//! Proxy-side cache: skip adaptation and analysis on subsequent runs.
//!
//! When a module is first loaded, the proxy adapts its PTX (seconds) and analyses it for lazy
//! compilation (seconds more). Both produce the same output for the same input. This cache stores
//! the results, keyed by a 128-bit hash of the original image, the policy, and the build identity
//! of the layer, so that a later run that loads the same module finds:
//!
//! - a **manifest**: every entry point with the flags the lazy path needs, the refusal reason when
//!   it is refused, the emulated-instruction counts when they are announced, what it is given with
//!   of reported traps (`vf_adapt::Reported`, format 9); and the
//!   architecture of the PTX that was retargeted, which the program's kernels are said to be
//!   compiled for (`hooks::versions`);
//! - **sub-module PTX** for each independent kernel, stored when the kernel is first extracted;
//! - **retargeted PTX/fatbin** (`.p`), the adapted image (LZ4-compressed) and its `Retargeted`
//!   outcome so subsequent runs reuse the adapted image when materializing a deferred library or
//!   loading a module or fat binary without re-running PTX lowering;
//! - for an image that has machine code only, its **translation**: the image to load in its
//!   place, with the kernels that are refused and those that have instructions added.
//!
//! On a cache hit the hooks behave exactly as without a cache: the same kernels are refused with
//! the same error at the same call, the same emulation warnings appear at the same moment, the
//! same kernel count and enumeration are reported.
//!
//! When the manifest or a sub-module cannot answer a request (a kernel that was never extracted, a
//! dependent entry, module variables, the whole-module fallback), the proxy adapts and analyses the
//! original image at that moment, as it does without a cache.
//!
//! # Location
//!
//! `VF_PROXY_CACHE` names the directory (`0` or empty: off). Without it, `$XDG_CACHE_HOME/volta-forever/proxy`
//! or `$HOME/.cache/volta-forever/proxy` is used. The cache is off when no directory can be formed,
//! and in the unit tests.
//!
//! # Trust
//!
//! Directories are created with mode 0700. Before use, the cache directory must be a real directory
//! owned by the effective uid and not writable by group or others; otherwise the cache is off.
//! Temporary files have a name of their own (`create_new`, and another name when one is taken) and
//! mode 0600; a write that fails removes its own, and the first use of the directory in a process
//! removes those that writes left more than ten minutes before. Files are read only when
//! they are regular files owned by the effective uid. Each file carries a magic, format version,
//! the full 128-bit key, the content length, and a checksum; anything that does not match is
//! ignored and may be deleted.
//!
//! # Size limit
//!
//! `VF_PROXY_CACHE_MAX` (bytes, default 2 GB: the translations of all of a stock PyTorch are
//! 0.6 GB) is enforced when writing, by deleting the files last used longest ago. Races between
//! processes are harmless: a file that disappeared is a cache miss, not an error.

use std::hash::Hasher;
use std::io::Write;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::{generated, log};

// ---- file format ---------------------------------------------------------------------------------

/// Magic at the start of every cached file.
const FILE_MAGIC: [u8; 4] = *b"S70C";
/// Format version; bump to invalidate all entries when the on-disk layout changes.
const FORMAT_VERSION: u32 = 9;
/// File header: magic(4) + format_version(4) + key(16) + content_len(4) + checksum(8) = 36.
const HEADER_LEN: usize = 36;
/// Default size limit for the whole cache directory.
const DEFAULT_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// A temporary file older than this was left by a write that did not end (its process was killed,
/// or the disk was full before temporary files were removed on failure): a write takes less.
const STALE_TEMPORARY: std::time::Duration = std::time::Duration::from_secs(600);
/// Names tried for a temporary file before a write gives up.
const TEMPORARY_NAMES: usize = 16;

/// Upper bound on an uncompressed retargeted image read from `.p` (1 GiB): NCCL 2.30.7 produces
/// ~248 MB of adapted PTX, so corrupt length prefixes cannot request multi-GiB allocations.
const MAX_RETARGETED_IMAGE_BYTES: usize = 1024 * 1024 * 1024;

/// Re-scan the directory at least every `LIMIT_SCAN_WRITES` writes even when accumulated bytes stay
/// small, so long runs eventually observe external writers without scanning on every cache write.
const LIMIT_SCAN_WRITES: u64 = 512;

static CACHE_STATE: OnceLock<Option<CacheState>> = OnceLock::new();

#[derive(Default)]
struct LimitState {
    scanned_total: Option<u64>,
    added_since_scan: u64,
    writes_since_scan: u64,
    scanning: bool,
}

struct CacheState {
    dir: PathBuf,
    max_bytes: u64,
    uid: u32,
    limit: std::sync::Mutex<LimitState>,
    #[cfg(test)]
    limit_scans: std::sync::atomic::AtomicUsize,
}

impl CacheState {
    fn new(dir: PathBuf, max_bytes: u64, uid: u32) -> Self {
        Self {
            dir,
            max_bytes,
            uid,
            limit: std::sync::Mutex::new(LimitState::default()),
            #[cfg(test)]
            limit_scans: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

fn state() -> Option<&'static CacheState> {
    CACHE_STATE
        .get_or_init(|| {
            let dir = cache_dir()?;
            let max_bytes = std::env::var("VF_PROXY_CACHE_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_MAX_BYTES);
            let uid = unsafe { libc::geteuid() };
            // Create the directory on first use, or verify it.
            if let Err(reason) = ensure_dir(&dir, uid) {
                log::warn(format_args!("{}", disabled(&reason)));
                return None;
            }
            remove_stale_temporaries(&dir, uid);
            Some(CacheState::new(dir, max_bytes, uid))
        })
        .as_ref()
}

/// The line said once, at the default level, when the cache is off for `reason`.
fn disabled(reason: &str) -> String {
    format!(
        "proxy cache disabled: {reason}; every start adapts again (nothing that the layer works out is kept, \
         `vf doctor` says what it found)"
    )
}

/// Determine the cache directory from the environment. None in the unit tests, whatever the
/// environment names: a test that adapts an image neither takes it from the cache of whoever runs
/// the tests nor keeps its translation there.
fn cache_dir() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    if let Ok(val) = std::env::var("VF_PROXY_CACHE") {
        if val == "0" || val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    // $XDG_CACHE_HOME/volta-forever/proxy, else $HOME/.cache/volta-forever/proxy
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Some(PathBuf::from(xdg).join("volta-forever/proxy"));
    }
    let home = std::env::var("HOME").ok()?;
    if home.is_empty() {
        return None;
    }
    Some(PathBuf::from(home).join(".cache/volta-forever/proxy"))
}

/// Create the directory (mode 0700) or verify an existing one is safe.
fn ensure_dir(dir: &Path, uid: u32) -> Result<(), String> {
    if dir.exists() {
        return verify_dir(dir, uid);
    }
    // Create with restrictive permissions. create_dir_all does not set the mode on intermediate
    // directories, so we set it on the leaf and accept that the parents may be wider.
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir:?}: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("{dir:?}: cannot set mode 0700: {e}"))?;
    }
    verify_dir(dir, uid)
}

/// The directory must be a directory (not a symlink), owned by us, and not writable by others.
fn verify_dir(dir: &Path, uid: u32) -> Result<(), String> {
    let meta = std::fs::symlink_metadata(dir).map_err(|e| format!("{dir:?}: {e}"))?;
    if !meta.is_dir() {
        return Err(format!("{dir:?} is not a directory"));
    }
    if meta.uid() != uid {
        return Err(format!("{dir:?} is not owned by uid {uid}"));
    }
    // (A umask of 002, the default of many distributions' accounts, makes directories that the
    // group can write.)
    let mode = meta.mode() & 0o7777;
    if mode & 0o022 != 0 {
        return Err(format!(
            "{dir:?} is writable by its group or by others (mode {mode:o}): chmod 700 {dir:?} makes it the user's \
             alone"
        ));
    }
    Ok(())
}

// ---- hashing -------------------------------------------------------------------------------------

/// Compute a 128-bit hash of the original image bytes together with every field that affects the
/// adapted output: the policy, any load option that matters, and the build identity of the layer.
/// Two independent XxHash64 values (different seeds) give negligible collision probability for a
/// local cache.
pub fn cache_hash(image: &[u8], policy: &vf_adapt::Policy) -> u128 {
    let hash_with_seed = |seed: u64| -> u64 {
        let mut h = twox_hash::XxHash64::with_seed(seed);
        h.write(image);
        // Every field of the policy, whatever fields it comes to have.
        h.write(format!("{policy:?}").as_bytes());
        h.write_u64(generated::BUILD_IDENTITY);
        h.finish()
    };
    let lo = hash_with_seed(0);
    let hi = hash_with_seed(0x9e3779b97f4a7c15);
    (hi as u128) << 64 | lo as u128
}

/// A content checksum for the file footer (not cryptographic, just corruption detection).
fn content_checksum(data: &[u8]) -> u64 {
    let mut h = twox_hash::XxHash64::with_seed(0xa5a5a5a5);
    h.write(data);
    h.finish()
}

// ---- file I/O ------------------------------------------------------------------------------------

/// Build the file header for `key` and `data`.
fn file_header(key: u128, data: &[u8]) -> [u8; HEADER_LEN] {
    let mut hdr = [0u8; HEADER_LEN];
    hdr[0..4].copy_from_slice(&FILE_MAGIC);
    hdr[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    hdr[8..24].copy_from_slice(&key.to_le_bytes());
    hdr[24..28].copy_from_slice(&(data.len() as u32).to_le_bytes());
    hdr[28..36].copy_from_slice(&content_checksum(data).to_le_bytes());
    hdr
}

/// Write `data` to `path` atomically: create a temporary file with a name of its own and mode 0600,
/// write header + data, sync, rename. Returns the number of bytes written (header + data). Nothing
/// stays of a write that fails (a full disk, a limit on the size of files).
fn atomic_write(path: &Path, data: &[u8], key: u128) -> std::io::Result<u64> {
    let dir = path.parent().unwrap();
    std::fs::create_dir_all(dir)?;
    let (temp, file) = create_temporary(dir, temporary_suffix)?;
    if let Err(e) = write_entry(file, data, key).and_then(|()| std::fs::rename(&temp, path)) {
        let _ = std::fs::remove_file(&temp);
        return Err(e);
    }
    Ok((HEADER_LEN + data.len()) as u64)
}

/// Header + data into `file`, synced; the file is closed on return.
fn write_entry(mut file: std::fs::File, data: &[u8], key: u128) -> std::io::Result<()> {
    file.write_all(&file_header(key, data))?;
    file.write_all(data)?;
    file.sync_data()
}

/// A new temporary file in `dir`, of this write's own: `.tmp-` and the 16 hexadecimal digits of
/// `suffix()`, created with `create_new` and mode 0600. A name that is taken (by a write of another
/// process, or a file left by a write that did not end) is passed over for the next.
fn create_temporary(dir: &Path, mut suffix: impl FnMut() -> u64) -> std::io::Result<(PathBuf, std::fs::File)> {
    let mut taken = std::io::Error::from(std::io::ErrorKind::AlreadyExists);
    for _ in 0..TEMPORARY_NAMES {
        let temp = dir.join(format!(".tmp-{:016x}", suffix()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)
        {
            Ok(file) => return Ok((temp, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => taken = e,
            Err(e) => return Err(e),
        }
    }
    Err(taken)
}

/// The process id, a count of the process's writes and the time, mixed: in a container, where
/// every start has the same process id, the first write of each start has a name of its own.
fn temporary_suffix() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut h = twox_hash::XxHash64::with_seed(0x517cc1b727220a95);
    h.write_u32(std::process::id());
    h.write_u64(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    h.write_u128(now.as_nanos());
    h.finish()
}

/// Remove the temporary files of the directory that are older than any write takes: a write that
/// did not end left them.
fn remove_stale_temporaries(dir: &Path, uid: u32) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        if !entry.file_name().as_encoded_bytes().starts_with(b".tmp-") {
            continue;
        }
        // (Not followed if it is a link.)
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let stale = meta
            .modified()
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > STALE_TEMPORARY);
        if meta.is_file() && meta.uid() == uid && stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Read and verify a cached file. Returns the content (without the header) on success.
fn checked_read(path: &Path, key: u128, uid: u32) -> Option<Vec<u8>> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    // Opened without following a link, then examined as opened: what is read is what was checked.
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .ok()?;
    let meta = file.metadata().ok()?;
    if !meta.is_file() || meta.uid() != uid {
        return None;
    }
    // Longer than any entry can be (its length is a 32-bit number): not read. Nor is a file that
    // there is not the memory for: a failed allocation would end the program.
    let len = usize::try_from(meta.len())
        .ok()
        .filter(|&len| len <= HEADER_LEN + u32::MAX as usize)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).ok()?;
    file.read_to_end(&mut bytes).ok()?;
    if bytes.len() < HEADER_LEN || bytes[0..4] != FILE_MAGIC {
        // Too short to be a cache entry, or not ours: do not delete.
        return None;
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    if version != FORMAT_VERSION {
        let _ = std::fs::remove_file(path); // stale
        return None;
    }
    let stored_key = u128::from_le_bytes(bytes[8..24].try_into().ok()?);
    if stored_key != key {
        return None; // hash collision in the file name, not our entry
    }
    let stored_len = u32::from_le_bytes(bytes[24..28].try_into().ok()?) as usize;
    let stored_checksum = u64::from_le_bytes(bytes[28..36].try_into().ok()?);
    let data = bytes.get(HEADER_LEN..)?;
    if data.len() != stored_len {
        let _ = std::fs::remove_file(path);
        return None;
    }
    if content_checksum(data) != stored_checksum {
        let _ = std::fs::remove_file(path);
        return None;
    }
    // Touch the file so LRU eviction sees it as recently used.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let times = [
        libc::timespec {
            tv_sec: now.as_secs() as _,
            tv_nsec: 0,
        },
        libc::timespec {
            tv_sec: now.as_secs() as _,
            tv_nsec: 0,
        },
    ];
    if let Ok(cpath) = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()) {
        unsafe { libc::utimensat(libc::AT_FDCWD, cpath.as_ptr(), times.as_ptr(), 0) };
    }
    Some(data.to_vec())
}

// ---- size limit ----------------------------------------------------------------------------------

/// Enforce the size limit by deleting the least recently modified files. Races are harmless:
/// a file that another process deleted while we enumerate is simply gone.
fn enforce_limit(cs: &CacheState) -> u64 {
    #[cfg(test)]
    cs.limit_scans.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut entries: Vec<(PathBuf, u64, i64)> = Vec::new(); // (path, size, mtime)
    let mut total: u64 = 0;
    let Ok(readdir) = std::fs::read_dir(&cs.dir) else {
        return 0;
    };
    for entry in readdir.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let size = meta.len();
        total += size;
        entries.push((entry.path(), size, meta.mtime()));
    }
    if total <= cs.max_bytes {
        return total;
    }
    // Oldest first.
    entries.sort_by_key(|e| e.2);
    for (path, size, _) in &entries {
        if total <= cs.max_bytes {
            break;
        }
        if std::fs::remove_file(path).is_ok() {
            total = total.saturating_sub(*size);
        }
    }
    total
}

/// Enforce the cache directory size limit incrementally after a successful write of `written`
/// bytes, avoiding an O(N) directory scan per write while still bounding overrun and reacting
/// when accumulated writes approach `max_bytes`.
fn maybe_enforce_limit(cs: &CacheState, written: u64) {
    let mut guard = cs.limit.lock().unwrap_or_else(|e| e.into_inner());
    guard.added_since_scan = guard.added_since_scan.saturating_add(written);
    guard.writes_since_scan = guard.writes_since_scan.saturating_add(1);
    if guard.scanning {
        return;
    }
    let periodic_bytes = (cs.max_bytes / 16).clamp(64 * 1024, 64 * 1024 * 1024);
    let should_scan = match guard.scanned_total {
        None => true,
        Some(total) => {
            total.saturating_add(guard.added_since_scan) > cs.max_bytes
                || guard.added_since_scan >= periodic_bytes
                || guard.writes_since_scan >= LIMIT_SCAN_WRITES
        }
    };
    if !should_scan {
        return;
    }
    guard.scanning = true;
    drop(guard);

    let total_after = enforce_limit(cs);

    let mut guard = cs.limit.lock().unwrap_or_else(|e| e.into_inner());
    guard.scanned_total = Some(total_after);
    guard.added_since_scan = 0;
    guard.writes_since_scan = 0;
    guard.scanning = false;
}

// ---- manifest ------------------------------------------------------------------------------------

/// A cached manifest for one module: every entry point with the flags the lazy path needs and the
/// information the kernel tracker needs to refuse and announce kernels exactly as without a cache.
#[derive(Clone)]
pub struct CachedManifest {
    pub entries: Vec<ManifestEntry>,
    /// The architecture of the program's PTX that was retargeted, when it was a later one than this
    /// GPU's (`images::Replacement::retargeted_from`; format 8).
    pub retargeted_from: Option<u32>,
}

/// One entry in the manifest.
#[derive(Clone)]
pub struct ManifestEntry {
    pub name: String,
    pub independent: bool,
    /// The read-only data that the sub-module of the entry holds a copy of
    /// (`vf_ptx::deps::ModuleInfo::copied_tables`).
    pub tables: Vec<String>,
    /// When the entry is refused: `(function, reason)`.
    pub refused: Option<(String, String)>,
    /// When the entry has emulated instructions.
    pub emulated: Option<vf_ptx::EmulatedCounts>,
    /// Its code is a trap and nothing else (`vf_adapt::Stubs::trap_only`): given at a lookup, not
    /// launched.
    pub trap_only: bool,
    /// When the entry is given with reported traps (format 9).
    pub reported: Option<vf_adapt::Reported>,
}

/// Encode the manifest to bytes.
fn encode_manifest(m: &CachedManifest) -> Vec<u8> {
    let mut buf = Vec::new();
    // (0: none.)
    buf.extend_from_slice(&m.retargeted_from.unwrap_or(0).to_le_bytes());
    buf.extend_from_slice(&(m.entries.len() as u32).to_le_bytes());
    for e in &m.entries {
        // name
        buf.extend_from_slice(&(e.name.len() as u32).to_le_bytes());
        buf.extend_from_slice(e.name.as_bytes());
        // flags
        let mut flags: u8 = 0;
        if e.independent {
            flags |= 1;
        }
        if e.refused.is_some() {
            flags |= 2;
        }
        if e.emulated.is_some() {
            flags |= 4;
        }
        if !e.tables.is_empty() {
            flags |= 8;
        }
        if e.trap_only {
            flags |= 16;
        }
        if e.reported.is_some() {
            flags |= 32;
        }
        buf.push(flags);
        // tables
        if !e.tables.is_empty() {
            buf.extend_from_slice(&(e.tables.len() as u32).to_le_bytes());
            for table in &e.tables {
                buf.extend_from_slice(&(table.len() as u32).to_le_bytes());
                buf.extend_from_slice(table.as_bytes());
            }
        }
        // refused
        if let Some((ref func, ref reason)) = e.refused {
            buf.extend_from_slice(&(func.len() as u32).to_le_bytes());
            buf.extend_from_slice(func.as_bytes());
            buf.extend_from_slice(&(reason.len() as u32).to_le_bytes());
            buf.extend_from_slice(reason.as_bytes());
        }
        // emulated counts (13 fields, each u32)
        if let Some(ref c) = e.emulated {
            for v in [
                c.fp16_tensor,
                c.integer_tensor,
                c.fp8_tensor,
                c.fp4_tensor,
                c.thread_tensor,
                c.ldmatrix,
                c.movmatrix,
                c.cp_async,
                c.tanh,
                c.cvt_packed,
                c.fp8_cvt,
                c.nan_minmax,
                c.other,
            ] {
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
        }
        // reported traps: the first function and form, the instructions and the functions
        if let Some(ref r) = e.reported {
            for text in [&r.function, &r.form] {
                buf.extend_from_slice(&(text.len() as u32).to_le_bytes());
                buf.extend_from_slice(text.as_bytes());
            }
            buf.extend_from_slice(&(r.instructions as u32).to_le_bytes());
            buf.extend_from_slice(&(r.functions as u32).to_le_bytes());
        }
    }
    buf
}

/// Decode the manifest from bytes.
fn decode_manifest(data: &[u8]) -> Option<CachedManifest> {
    let mut pos = 0;
    let read_u32 = |pos: &mut usize| -> Option<u32> {
        if *pos + 4 > data.len() {
            return None;
        }
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().ok()?);
        *pos += 4;
        Some(v)
    };
    let read_str = |pos: &mut usize| -> Option<String> {
        let len = read_u32(pos)? as usize;
        if *pos + len > data.len() {
            return None;
        }
        let s = std::str::from_utf8(&data[*pos..*pos + len]).ok()?.to_string();
        *pos += len;
        Some(s)
    };
    let retargeted_from = Some(read_u32(&mut pos)?).filter(|&sm| sm != 0);
    let count = read_u32(&mut pos)? as usize;
    // (An entry takes 5 bytes at least: no more room than the data can fill.)
    let mut entries = Vec::with_capacity(count.min(data.len() / 5));
    for _ in 0..count {
        let name = read_str(&mut pos)?;
        if pos >= data.len() {
            return None;
        }
        let flags = data[pos];
        pos += 1;
        let independent = flags & 1 != 0;
        let trap_only = flags & 16 != 0;
        let mut tables = Vec::new();
        if flags & 8 != 0 {
            let count = read_u32(&mut pos)? as usize;
            // (A name takes 4 bytes at least.)
            if count > (data.len() - pos) / 4 {
                return None;
            }
            for _ in 0..count {
                tables.push(read_str(&mut pos)?);
            }
        }
        let refused = if flags & 2 != 0 {
            let func = read_str(&mut pos)?;
            let reason = read_str(&mut pos)?;
            Some((func, reason))
        } else {
            None
        };
        let emulated = if flags & 4 != 0 {
            let mut vals = [0u32; 13];
            for v in &mut vals {
                *v = read_u32(&mut pos)?;
            }
            Some(vf_ptx::EmulatedCounts {
                fp16_tensor: vals[0] as usize,
                integer_tensor: vals[1] as usize,
                fp8_tensor: vals[2] as usize,
                fp4_tensor: vals[3] as usize,
                thread_tensor: vals[4] as usize,
                ldmatrix: vals[5] as usize,
                movmatrix: vals[6] as usize,
                cp_async: vals[7] as usize,
                tanh: vals[8] as usize,
                cvt_packed: vals[9] as usize,
                fp8_cvt: vals[10] as usize,
                nan_minmax: vals[11] as usize,
                other: vals[12] as usize,
                // Counted in translated machine code, which is not cached.
                machine_code: 0,
            })
        } else {
            None
        };
        let reported = if flags & 32 != 0 {
            Some(vf_adapt::Reported {
                kernel: name.clone(),
                function: read_str(&mut pos)?,
                form: read_str(&mut pos)?,
                instructions: read_u32(&mut pos)? as usize,
                functions: read_u32(&mut pos)? as usize,
            })
        } else {
            None
        };
        entries.push(ManifestEntry {
            name,
            independent,
            tables,
            refused,
            emulated,
            trap_only,
            reported,
        });
    }
    Some(CachedManifest {
        entries,
        retargeted_from,
    })
}

// ---- public API ----------------------------------------------------------------------------------

/// A cache key for a module, computed from the original image and adaptation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheKey {
    pub key: u128,
}

/// Compute the cache key for the given original image bytes and an explicit adaptation policy.
pub fn key_for_policy(image: &[u8], policy: &vf_adapt::Policy) -> CacheKey {
    CacheKey {
        key: cache_hash(image, policy),
    }
}

/// Compute the cache key for the given original image bytes, adapted by the policy of the load
/// this thread is making (`images::policy`).
pub fn key_for(image: &[u8]) -> CacheKey {
    key_for_policy(image, &crate::images::policy())
}

/// File path for the manifest of a module.
fn manifest_path(cs: &CacheState, key: u128) -> PathBuf {
    cs.dir.join(format!("{key:032x}.manifest"))
}

/// File path for a sub-module.
fn submodule_path(cs: &CacheState, key: u128, kernel_name: &str) -> PathBuf {
    let name_hash = {
        let mut h = twox_hash::XxHash64::with_seed(0xdeadbeef);
        h.write(kernel_name.as_bytes());
        h.finish()
    };
    cs.dir.join(format!("{key:032x}.s{name_hash:016x}"))
}

/// Try to load the manifest for a module.
pub fn load_manifest(ck: &CacheKey) -> Option<CachedManifest> {
    let cs = state()?;
    let data = checked_read(&manifest_path(cs, ck.key), ck.key, cs.uid)?;
    decode_manifest(&data)
}

/// Try to load a cached sub-module PTX for a kernel.
pub fn load_submodule(ck: &CacheKey, kernel_name: &str) -> Option<Vec<u8>> {
    let cs = state()?;
    let data = checked_read(&submodule_path(cs, ck.key, kernel_name), ck.key, cs.uid)?;
    // The file is named by a hash of the kernel's name: the name itself is in it.
    let length = u32::from_le_bytes(data.get(..4)?.try_into().ok()?) as usize;
    (data.get(4..4 + length)? == kernel_name.as_bytes()).then(|| data[4 + length..].to_vec())
}

/// Whether the cache is in use.
pub fn enabled() -> bool {
    state().is_some()
}

/// Save the manifest for a module. Enforces the size limit afterwards.
pub fn save_manifest(ck: &CacheKey, manifest: &CachedManifest) {
    let Some(cs) = state() else { return };
    let data = encode_manifest(manifest);
    match atomic_write(&manifest_path(cs, ck.key), &data, ck.key) {
        Ok(written) => maybe_enforce_limit(cs, written),
        Err(e) => log::trace(format_args!("cache: failed to write manifest: {e}")),
    }
}

/// Save a sub-module PTX to the cache. Enforces the size limit afterwards.
pub fn save_submodule(ck: &CacheKey, kernel_name: &str, ptx: &[u8]) {
    let Some(cs) = state() else { return };
    let mut data = Vec::with_capacity(4 + kernel_name.len() + ptx.len());
    data.extend_from_slice(&(kernel_name.len() as u32).to_le_bytes());
    data.extend_from_slice(kernel_name.as_bytes());
    data.extend_from_slice(ptx);
    match atomic_write(&submodule_path(cs, ck.key, kernel_name), &data, ck.key) {
        Ok(written) => maybe_enforce_limit(cs, written),
        Err(e) => log::trace(format_args!("cache: failed to write sub-module for {kernel_name}: {e}")),
    }
}

// ---- translations --------------------------------------------------------------------------------

fn put_usize(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u32).to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_usize(buf, s.len());
    buf.extend_from_slice(s.as_bytes());
}

fn take_usize(data: &[u8], pos: &mut usize) -> Option<usize> {
    let v = u32::from_le_bytes(data.get(*pos..*pos + 4)?.try_into().ok()?);
    *pos += 4;
    Some(v as usize)
}

fn take_str(data: &[u8], pos: &mut usize) -> Option<String> {
    let len = take_usize(data, pos)?;
    let s = std::str::from_utf8(data.get(*pos..pos.checked_add(len)?)?)
        .ok()?
        .to_string();
    *pos += len;
    Some(s)
}

/// The counts of emulated instructions, in the order they are written in cached entries.
fn counts_of(c: &vf_ptx::EmulatedCounts) -> [usize; 14] {
    [
        c.fp16_tensor,
        c.integer_tensor,
        c.fp8_tensor,
        c.fp4_tensor,
        c.thread_tensor,
        c.ldmatrix,
        c.movmatrix,
        c.cp_async,
        c.tanh,
        c.cvt_packed,
        c.fp8_cvt,
        c.nan_minmax,
        c.other,
        c.machine_code,
    ]
}

fn put_outcome_lists(
    buf: &mut Vec<u8>,
    refused: &[vf_adapt::Refused],
    emulated: &[vf_adapt::Emulated],
    stubs: &vf_adapt::Stubs,
) {
    put_usize(buf, refused.len());
    for r in refused {
        put_str(buf, &r.kernel);
        put_str(buf, &r.function);
        put_str(buf, &r.reason);
    }
    put_usize(buf, emulated.len());
    for e in emulated {
        put_str(buf, &e.kernel);
        for v in counts_of(&e.counts) {
            put_usize(buf, v);
        }
    }
    for list in [&stubs.trap_only, &stubs.empty] {
        put_usize(buf, list.len());
        for kernel in list {
            put_str(buf, kernel);
        }
    }
}

fn take_outcome_lists(
    data: &[u8],
    pos: &mut usize,
) -> Option<(Vec<vf_adapt::Refused>, Vec<vf_adapt::Emulated>, vf_adapt::Stubs)> {
    let mut refused = Vec::new();
    for _ in 0..take_usize(data, pos)? {
        refused.push(vf_adapt::Refused {
            kernel: take_str(data, pos)?,
            function: take_str(data, pos)?,
            reason: take_str(data, pos)?,
        });
    }
    let mut emulated = Vec::new();
    for _ in 0..take_usize(data, pos)? {
        let kernel = take_str(data, pos)?;
        let mut v = [0usize; 14];
        for x in &mut v {
            *x = take_usize(data, pos)?;
        }
        emulated.push(vf_adapt::Emulated {
            kernel,
            counts: vf_ptx::EmulatedCounts {
                fp16_tensor: v[0],
                integer_tensor: v[1],
                fp8_tensor: v[2],
                fp4_tensor: v[3],
                thread_tensor: v[4],
                ldmatrix: v[5],
                movmatrix: v[6],
                cp_async: v[7],
                tanh: v[8],
                cvt_packed: v[9],
                fp8_cvt: v[10],
                nan_minmax: v[11],
                other: v[12],
                machine_code: v[13],
            },
        });
    }
    let mut stubs = vf_adapt::Stubs::default();
    for list in [&mut stubs.trap_only, &mut stubs.empty] {
        for _ in 0..take_usize(data, pos)? {
            list.push(take_str(data, pos)?);
        }
    }
    Some((refused, emulated, stubs))
}

/// A translation as it is kept: what `vf_adapt` said of the image (how many modules and
/// functions, from which architecture, the kernels refused and those with instructions added),
/// then the image to load. `None` for anything that is not a translation with an image.
fn encode_translation(adapted: &vf_adapt::Adapted) -> Option<Vec<u8>> {
    let vf_adapt::Outcome::Translated {
        entries,
        from_sm,
        functions,
        refused,
        emulated,
        over_budget,
        stubs,
        paired,
    } = &adapted.outcome
    else {
        return None;
    };
    let image = adapted.image.as_ref()?;
    let mut buf = Vec::with_capacity(image.len() + 256);
    put_usize(&mut buf, *entries);
    put_usize(&mut buf, *from_sm as usize);
    put_usize(&mut buf, *functions);
    put_usize(&mut buf, *over_budget as usize);
    put_outcome_lists(&mut buf, refused, emulated, stubs);
    put_usize(&mut buf, paired.len());
    for name in paired {
        put_str(&mut buf, name);
    }
    buf.extend_from_slice(image);
    Some(buf)
}

fn decode_translation(data: &[u8]) -> Option<vf_adapt::Adapted> {
    let mut pos = 0;
    let entries = take_usize(data, &mut pos)?;
    let from_sm = take_usize(data, &mut pos)? as u32;
    let functions = take_usize(data, &mut pos)?;
    let over_budget = take_usize(data, &mut pos)? != 0;
    let (refused, emulated, stubs) = take_outcome_lists(data, &mut pos)?;
    let mut paired = Vec::new();
    for _ in 0..take_usize(data, &mut pos)? {
        paired.push(take_str(data, &mut pos)?);
    }
    Some(vf_adapt::Adapted {
        outcome: vf_adapt::Outcome::Translated {
            entries,
            from_sm,
            functions,
            refused,
            emulated,
            over_budget,
            stubs,
            paired,
        },
        image: Some(data.get(pos..)?.to_vec()),
    })
}

/// Serialize a retargeted PTX/fatbin adaptation outcome with LZ4-compressed image bytes.
pub(crate) fn encode_retargeted(adapted: &vf_adapt::Adapted) -> Option<Vec<u8>> {
    let vf_adapt::Outcome::Retargeted {
        entries,
        from_sm,
        tensor_rewritten,
        tanh_expanded,
        trapped,
        refused,
        emulated,
        replaced,
        reported,
        stubs,
    } = &adapted.outcome
    else {
        return None;
    };
    let image = adapted.image.as_ref()?;
    let compressed = lz4_flex::compress_prepend_size(image);
    let mut buf = Vec::with_capacity(compressed.len() + 256);
    put_usize(&mut buf, *entries);
    put_usize(&mut buf, *from_sm as usize);
    put_usize(&mut buf, *tensor_rewritten);
    put_usize(&mut buf, *tanh_expanded);
    put_outcome_lists(&mut buf, refused, emulated, stubs);
    put_usize(&mut buf, trapped.len());
    for t in trapped {
        put_str(&mut buf, &t.function);
        put_str(&mut buf, &t.reason);
    }
    put_usize(&mut buf, replaced.len());
    for rep in replaced {
        put_str(&mut buf, &rep.function);
        put_usize(&mut buf, rep.forms.len());
        for form in &rep.forms {
            put_str(&mut buf, form);
        }
    }
    put_usize(&mut buf, reported.len());
    for rep in reported {
        put_str(&mut buf, &rep.kernel);
        put_str(&mut buf, &rep.function);
        put_str(&mut buf, &rep.form);
        put_usize(&mut buf, rep.instructions);
        put_usize(&mut buf, rep.functions);
    }
    buf.extend_from_slice(&compressed);
    Some(buf)
}

/// Deserialize a cached retargeted PTX/fatbin outcome and decompress its image bytes.
pub(crate) fn decode_retargeted(data: &[u8]) -> Option<vf_adapt::Adapted> {
    let mut pos = 0;
    let entries = take_usize(data, &mut pos)?;
    let from_sm = take_usize(data, &mut pos)? as u32;
    let tensor_rewritten = take_usize(data, &mut pos)?;
    let tanh_expanded = take_usize(data, &mut pos)?;
    let (refused, emulated, stubs) = take_outcome_lists(data, &mut pos)?;
    let mut trapped = Vec::new();
    for _ in 0..take_usize(data, &mut pos)? {
        trapped.push(vf_adapt::Trapped {
            function: take_str(data, &mut pos)?,
            reason: take_str(data, &mut pos)?,
        });
    }
    let mut replaced = Vec::new();
    for _ in 0..take_usize(data, &mut pos)? {
        let function = take_str(data, &mut pos)?;
        let mut forms = Vec::new();
        for _ in 0..take_usize(data, &mut pos)? {
            forms.push(take_str(data, &mut pos)?);
        }
        replaced.push(vf_adapt::Replaced { function, forms });
    }
    let mut reported = Vec::new();
    for _ in 0..take_usize(data, &mut pos)? {
        reported.push(vf_adapt::Reported {
            kernel: take_str(data, &mut pos)?,
            function: take_str(data, &mut pos)?,
            form: take_str(data, &mut pos)?,
            instructions: take_usize(data, &mut pos)?,
            functions: take_usize(data, &mut pos)?,
        });
    }
    let payload = data.get(pos..)?;
    let raw_len = u32::from_le_bytes(payload.get(..4)?.try_into().ok()?) as usize;
    if raw_len > MAX_RETARGETED_IMAGE_BYTES {
        return None;
    }
    let mut image = Vec::new();
    image.try_reserve_exact(raw_len).ok()?;
    image.resize(raw_len, 0);
    let written = lz4_flex::block::decompress_into(&payload[4..], &mut image).ok()?;
    if written != raw_len {
        return None;
    }
    Some(vf_adapt::Adapted {
        outcome: vf_adapt::Outcome::Retargeted {
            entries,
            from_sm,
            tensor_rewritten,
            tanh_expanded,
            trapped,
            refused,
            emulated,
            replaced,
            reported,
            stubs,
        },
        image: Some(image),
    })
}

/// File path for the translation of a module.
fn translation_path(cs: &CacheState, key: u128) -> PathBuf {
    cs.dir.join(format!("{key:032x}.t"))
}

/// File path for a retargeted PTX/fatbin image.
fn retargeted_path(cs: &CacheState, key: u128) -> PathBuf {
    cs.dir.join(format!("{key:032x}.p"))
}

/// The translation of the image of this key, if an earlier run kept it.
pub fn load_translation(ck: &CacheKey) -> Option<vf_adapt::Adapted> {
    let cs = state()?;
    decode_translation(&checked_read(&translation_path(cs, ck.key), ck.key, cs.uid)?)
}

/// The retargeted PTX/fatbin image of this key, if an earlier run kept it.
pub fn load_retargeted(ck: &CacheKey) -> Option<vf_adapt::Adapted> {
    let cs = state()?;
    decode_retargeted(&checked_read(&retargeted_path(cs, ck.key), ck.key, cs.uid)?)
}

fn save_encoded(path: PathBuf, ck: &CacheKey, cs: &CacheState, data: &[u8], kind: &str) {
    // (The length of a file is written in 32 bits.)
    if data.len() > u32::MAX as usize {
        return;
    }
    match atomic_write(&path, data, ck.key) {
        Ok(written) => maybe_enforce_limit(cs, written),
        Err(e) => log::trace(format_args!("cache: failed to write {kind}: {e}")),
    }
}

/// Keep a translation for later runs (anything else is not kept). Enforces the size limit afterwards.
pub fn save_translation(ck: &CacheKey, adapted: &vf_adapt::Adapted) {
    let Some(cs) = state() else { return };
    let Some(data) = encode_translation(adapted) else {
        return;
    };
    save_encoded(translation_path(cs, ck.key), ck, cs, &data, "a translation");
}

/// Keep a retargeted PTX/fatbin adaptation for later runs. Enforces the size limit afterwards.
pub fn save_retargeted(ck: &CacheKey, adapted: &vf_adapt::Adapted) {
    let Some(cs) = state() else { return };
    let Some(data) = encode_retargeted(adapted) else {
        return;
    };
    save_encoded(retargeted_path(cs, ck.key), ck, cs, &data, "retargeted PTX");
}

// ---- tests ---------------------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let base = std::env::var("VF_TEST_DIR").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        let dir = PathBuf::from(base).join(format!("cache-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Set mode 0700 so verify_dir passes.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        dir
    }

    fn make_manifest() -> CachedManifest {
        CachedManifest {
            entries: vec![
                ManifestEntry {
                    name: "kernel_a".to_string(),
                    independent: true,
                    tables: vec!["grid".to_string(), "signs".to_string()],
                    refused: None,
                    emulated: Some({
                        let mut c = vf_ptx::EmulatedCounts::default();
                        c.tanh = 3;
                        c.fp16_tensor = 10;
                        c
                    }),
                    trap_only: false,
                    reported: Some(vf_adapt::Reported {
                        kernel: "kernel_a".to_string(),
                        function: "helper".to_string(),
                        form: "multimem.st.global.b32".to_string(),
                        instructions: 7,
                        functions: 2,
                    }),
                },
                ManifestEntry {
                    name: "kernel_b".to_string(),
                    independent: false,
                    tables: Vec::new(),
                    refused: Some(("bad_func".to_string(), "unknown instruction".to_string())),
                    emulated: None,
                    trap_only: false,
                    reported: None,
                },
                ManifestEntry {
                    name: "kernel_c".to_string(),
                    independent: true,
                    tables: Vec::new(),
                    refused: None,
                    emulated: None,
                    trap_only: true,
                    reported: None,
                },
            ],
            retargeted_from: Some(75),
        }
    }

    #[test]
    fn manifest_round_trip() {
        let manifest = make_manifest();
        let encoded = encode_manifest(&manifest);
        let decoded = decode_manifest(&encoded).unwrap();
        assert_eq!(decoded.entries.len(), 3);
        assert_eq!(decoded.entries[0].name, "kernel_a");
        assert!(decoded.entries[0].independent);
        assert!(decoded.entries[0].refused.is_none());
        assert_eq!(decoded.entries[0].tables, ["grid", "signs"]);
        assert!(decoded.entries[1].tables.is_empty());
        let emu = decoded.entries[0].emulated.as_ref().unwrap();
        assert_eq!(emu.tanh, 3);
        assert_eq!(emu.fp16_tensor, 10);
        assert_eq!(emu.ldmatrix, 0);
        assert_eq!(decoded.entries[1].name, "kernel_b");
        assert!(!decoded.entries[1].independent);
        let (func, reason) = decoded.entries[1].refused.as_ref().unwrap();
        assert_eq!(func, "bad_func");
        assert_eq!(reason, "unknown instruction");
        assert!(decoded.entries[1].emulated.is_none());
        assert_eq!(decoded.entries[2].name, "kernel_c");
        assert!(decoded.entries[2].independent);
        // A kernel whose code is a trap is kept as such, and only it.
        assert!(decoded.entries[2].trap_only && !decoded.entries[0].trap_only && !decoded.entries[1].trap_only);
        // What a kernel is given with of reported traps, and only it (format 9).
        assert_eq!(decoded.entries[0].reported, manifest.entries[0].reported);
        assert!(decoded.entries[1].reported.is_none() && decoded.entries[2].reported.is_none());
        // The architecture of the PTX retargeted, and none (format 8).
        assert_eq!(decoded.retargeted_from, Some(75));
        let none = CachedManifest {
            retargeted_from: None,
            ..manifest
        };
        assert_eq!(decode_manifest(&encode_manifest(&none)).unwrap().retargeted_from, None);
    }

    #[test]
    fn translation_round_trip() {
        let mut counts = vf_ptx::EmulatedCounts::default();
        counts.machine_code = 3;
        counts.fp4_tensor = 5;
        let adapted = vf_adapt::Adapted {
            outcome: vf_adapt::Outcome::Translated {
                entries: 1,
                from_sm: 75,
                functions: 7,
                refused: vec![vf_adapt::Refused {
                    kernel: "warp_sum".to_string(),
                    function: "warp_sum".to_string(),
                    reason: "its machine code for sm_75 is not translated (why)".to_string(),
                }],
                emulated: vec![vf_adapt::Emulated {
                    kernel: "block_sum".to_string(),
                    counts,
                }],
                over_budget: true,
                stubs: vf_adapt::Stubs {
                    trap_only: vec!["no_code".to_string()],
                    empty: vec!["noop".to_string(), "probe".to_string()],
                },
                paired: vec!["block_sum".to_string()],
            },
            image: Some(vec![0x7f, b'E', b'L', b'F', 1, 2, 3]),
        };
        let data = encode_translation(&adapted).unwrap();
        let back = decode_translation(&data).unwrap();
        assert_eq!(back.outcome, adapted.outcome);
        assert_eq!(back.image, adapted.image);
        match &back.outcome {
            vf_adapt::Outcome::Translated {
                emulated, over_budget, ..
            } => {
                assert!(over_budget);
                assert_eq!(emulated[0].counts.fp4_tensor, 5);
                assert_eq!(emulated[0].counts.machine_code, 3);
            }
            _ => panic!("not a translation"),
        }
        // Cut short anywhere before the image, it is nothing.
        for cut in 0..data.len() - 7 {
            assert!(decode_translation(&data[..cut]).is_none(), "{cut}");
        }
        // Only a translation with an image is kept.
        let native = vf_adapt::Adapted {
            outcome: vf_adapt::Outcome::Native,
            image: None,
        };
        assert!(encode_translation(&native).is_none());
    }

    #[test]
    fn file_round_trip() {
        let dir = test_dir("roundtrip");
        let key: u128 = 0x1234_5678_9abc_def0_1234_5678_9abc_def0;
        let uid = unsafe { libc::geteuid() };
        let path = dir.join("test.bin");
        let data = b"hello world";
        atomic_write(&path, data, key).unwrap();
        let read = checked_read(&path, key, uid).unwrap();
        assert_eq!(read, data);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The temporary files of the directory.
    fn temporaries(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(".tmp-"))
            .collect()
    }

    #[test]
    fn failed_write_leaves_nothing() {
        // The rename fails: the name of the entry is a directory with something in it.
        let dir = test_dir("failedrename");
        let path = dir.join("entry");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("inside"), b"x").unwrap();
        assert!(atomic_write(&path, &[7u8; 512], 1).is_err());
        assert_eq!(temporaries(&dir), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_over_the_size_limit_leaves_nothing() {
        // A limit on the size of files, in a process of its own (it holds for every thread): the
        // write of the data fails (EFBIG), as on a full disk.
        let dir = test_dir("sizelimitwrite");
        let path = dir.join("entry");
        let child = unsafe { libc::fork() };
        assert!(child >= 0);
        if child == 0 {
            let limit = libc::rlimit {
                rlim_cur: 100,
                rlim_max: 100,
            };
            let ok = unsafe {
                libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
                libc::setrlimit(libc::RLIMIT_FSIZE, &limit) == 0
            } && atomic_write(&path, &[7u8; 512], 1).is_err()
                && temporaries(&dir).is_empty()
                && !path.exists();
            unsafe { libc::_exit(if ok { 0 } else { 1 }) };
        }
        let mut status = 0;
        assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0, "{status:#x}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn temporary_name_taken_is_passed_over() {
        // A file left under the first name, as by a start of a container before this one.
        let dir = test_dir("nametaken");
        std::fs::write(dir.join(".tmp-0000000000000001"), b"left").unwrap();
        let mut names = [1u64, 1, 2].into_iter();
        let (temp, _file) = create_temporary(&dir, || names.next().unwrap()).unwrap();
        assert_eq!(temp, dir.join(".tmp-0000000000000002"));
        assert_eq!(std::fs::read(dir.join(".tmp-0000000000000001")).unwrap(), b"left");
        // Every name taken: the write gives up.
        let error = create_temporary(&dir, || 1).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_temporaries_removed() {
        let dir = test_dir("staletemp");
        let uid = unsafe { libc::geteuid() };
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        for name in [".tmp-00000000000000aa", "0123.manifest"] {
            let file = std::fs::File::create(dir.join(name)).unwrap();
            file.set_modified(old).unwrap();
        }
        std::fs::write(dir.join(".tmp-00000000000000bb"), b"being written").unwrap();
        remove_stale_temporaries(&dir, uid);
        // The old temporary goes; one being written and an entry of the cache stay.
        assert_eq!(temporaries(&dir), [".tmp-00000000000000bb"]);
        assert!(dir.join("0123.manifest").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn older_format_version_refused() {
        // A record written by version 2 had one field fewer (no `fp4_tensor`): the version is the
        // only thing that keeps it from being read with its fields shifted.
        let dir = test_dir("oldversion");
        let key: u128 = 7;
        let uid = unsafe { libc::geteuid() };
        let path = dir.join("old.bin");
        atomic_write(&path, b"payload", key).unwrap();
        assert_eq!(checked_read(&path, key, uid).unwrap(), b"payload");
        let mut full = std::fs::read(&path).unwrap();
        full[4..8].copy_from_slice(&2u32.to_le_bytes());
        std::fs::write(&path, &full).unwrap();
        assert!(checked_read(&path, key, uid).is_none());
        // A stale entry is deleted.
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_file_rejected() {
        let dir = test_dir("truncated");
        let key: u128 = 42;
        let uid = unsafe { libc::geteuid() };
        let path = dir.join("trunc.bin");
        let data = b"some content";
        atomic_write(&path, data, key).unwrap();
        // Truncate the file.
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..HEADER_LEN + 2]).unwrap();
        assert!(checked_read(&path, key, uid).is_none());
        // File should have been deleted.
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sizes_beyond_the_data_are_not_allocated() {
        // (Until 2026-11-22 each of these allocated what it was told, and the allocation of 1 TB
        // or more failed: that ends the program.)
        assert!(decode_manifest(&u32::MAX.to_le_bytes()).is_none());
        let dir = test_dir("long");
        let key: u128 = 5;
        let uid = unsafe { libc::geteuid() };
        let path = dir.join("long.bin");
        atomic_write(&path, b"payload", key).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1 << 40)
            .unwrap();
        assert!(checked_read(&path, key, uid).is_none());
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_checksum_rejected() {
        let dir = test_dir("corrupt");
        let key: u128 = 99;
        let uid = unsafe { libc::geteuid() };
        let path = dir.join("bad.bin");
        let data = b"payload";
        atomic_write(&path, data, key).unwrap();
        // Flip a byte in the payload.
        let mut full = std::fs::read(&path).unwrap();
        full[HEADER_LEN] ^= 0xff;
        std::fs::write(&path, &full).unwrap();
        assert!(checked_read(&path, key, uid).is_none());
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_key_rejected() {
        let dir = test_dir("wrongkey");
        let uid = unsafe { libc::geteuid() };
        let path = dir.join("wk.bin");
        atomic_write(&path, b"data", 100).unwrap();
        assert!(checked_read(&path, 200, uid).is_none());
        // Not deleted: different key, might belong to another module.
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn foreign_file_not_deleted() {
        let dir = test_dir("foreign");
        let uid = unsafe { libc::geteuid() };
        let path = dir.join("foreign.txt");
        std::fs::write(&path, b"not a cache file").unwrap();
        assert!(checked_read(&path, 0, uid).is_none());
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dir_writable_by_others_refused() {
        let dir = test_dir("perms");
        let uid = unsafe { libc::geteuid() };
        // World-writable, and writable by the group (as a umask of 002 makes a directory): refused,
        // with the fix, and the line says what it costs.
        #[cfg(unix)]
        for mode in [0o777, 0o775, 0o770] {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
            let reason = verify_dir(&dir, uid).unwrap_err();
            assert!(
                reason.contains(&format!(
                    "is writable by its group or by others (mode {mode:o}): chmod 700 {dir:?}"
                )),
                "{reason}"
            );
            assert!(
                disabled(&reason).starts_with("proxy cache disabled: ")
                    && disabled(&reason).contains("; every start adapts again"),
                "{}",
                disabled(&reason)
            );
        }
        // Restore and verify passes.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        assert!(verify_dir(&dir, uid).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn size_limit_evicts_old_files() {
        let dir = test_dir("sizelimit");
        let uid = unsafe { libc::geteuid() };
        let cs = CacheState::new(dir.clone(), 200, uid);
        // Write several files.
        for i in 0u128..10 {
            let path = dir.join(format!("f{i}.bin"));
            atomic_write(&path, &[0u8; 50], i).unwrap();
        }
        enforce_limit(&cs);
        // After eviction, total should be <= 200 bytes.
        let total: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum();
        assert!(total <= 200, "total {total} > 200");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_unit_tests_have_no_cache() {
        // (Until 2026-11-24 a test that translates machine code through `images::adapt` kept its
        // translation in ~/.cache/volta-forever/proxy at each run.) The functions of the cache are
        // tested here on directories of the tests' own.
        assert_eq!(cache_dir(), None);
        assert!(!enabled());
    }

    #[test]
    fn hash_includes_policy_and_build() {
        let img = b"test image";
        let p1 = vf_adapt::Policy::default();
        let mut p2 = vf_adapt::Policy::default();
        p2.told_sm = Some(80);
        let h1 = cache_hash(img, &p1);
        let h2 = cache_hash(img, &p2);
        assert_ne!(h1, h2, "different policy must give different hash");
        let mut grouped = vf_adapt::Policy::default();
        grouped.lowering.bf16_tc_groups = true;
        assert_ne!(
            h1,
            cache_hash(img, &grouped),
            "changing BF16 lowering changes the cache key"
        );
        let mut tiled = vf_adapt::Policy::default();
        tiled.lowering.bf16_tc_tiles = true;
        assert_ne!(cache_hash(img, &tiled), h1);
        assert_ne!(cache_hash(img, &tiled), cache_hash(img, &grouped));
        let mut column = vf_adapt::Policy::default();
        column.lowering.bf16_column = vf_ptx::lower::Bf16Column::Ordered;
        let ordered = cache_hash(img, &column);
        assert_ne!(ordered, h1);
        column.lowering.bf16_column = vf_ptx::lower::Bf16Column::Tree;
        assert_ne!(cache_hash(img, &column), ordered);
        // Same policy, same image: same hash.
        let h1b = cache_hash(img, &p1);
        assert_eq!(h1, h1b);
        // Safe mode (VF_SAFE=1): what a run without it kept is not given to a run with it, nor the
        // reverse, also where it makes the same bytes (an image without products of machine code).
        // A switch that safe mode turns off does not change its key; the pairs of machine code that
        // it turns off are in the key: VF_PAIRS=0 without safe mode has a key of its own, and so
        // would safe mode with them.
        let mut safe = vf_adapt::Policy::default();
        safe.make_safe();
        assert_ne!(cache_hash(img, &safe), h1, "safe mode changes the cache key");
        let mut grouped_safe = grouped;
        grouped_safe.make_safe();
        assert_ne!(cache_hash(img, &grouped_safe), cache_hash(img, &grouped));
        assert_eq!(cache_hash(img, &grouped_safe), cache_hash(img, &safe));
        let unpaired = vf_adapt::Policy {
            pairs: false,
            ..vf_adapt::Policy::default()
        };
        assert_ne!(cache_hash(img, &unpaired), h1);
        assert_ne!(cache_hash(img, &unpaired), cache_hash(img, &safe));
        assert_ne!(
            cache_hash(img, &vf_adapt::Policy { pairs: true, ..safe }),
            cache_hash(img, &safe)
        );
        let with_companions = vf_adapt::Policy {
            companions: true,
            ..vf_adapt::Policy::default()
        };
        assert_ne!(cache_hash(img, &with_companions), h1);
        assert_ne!(cache_hash(img, &with_companions), cache_hash(img, &unpaired));
        assert_ne!(cache_hash(img, &with_companions), cache_hash(img, &safe));
        let mut comp_safe = with_companions;
        comp_safe.make_safe();
        assert_eq!(cache_hash(img, &comp_safe), cache_hash(img, &safe));
    }

    #[test]
    fn retargeted_round_trip() {
        let adapted = vf_adapt::Adapted {
            outcome: vf_adapt::Outcome::Retargeted {
                entries: 2,
                from_sm: 120,
                tensor_rewritten: 42,
                tanh_expanded: 3,
                trapped: vec![vf_adapt::Trapped {
                    function: "helper_bad".to_string(),
                    reason: "unsupported opcode".to_string(),
                }],
                refused: vec![vf_adapt::Refused {
                    kernel: "k_refused".to_string(),
                    function: "helper_bad".to_string(),
                    reason: "unsupported opcode".to_string(),
                }],
                emulated: vec![vf_adapt::Emulated {
                    kernel: "k_emu".to_string(),
                    counts: vf_ptx::EmulatedCounts {
                        fp16_tensor: 8,
                        tanh: 3,
                        ..Default::default()
                    },
                }],
                replaced: vec![vf_adapt::Replaced {
                    function: "k_nccl".to_string(),
                    forms: vec!["multimem.ld_reduce.global.b32".to_string()],
                }],
                reported: vec![vf_adapt::Reported {
                    kernel: "k_nccl".to_string(),
                    function: "k_nccl".to_string(),
                    form: "multimem.ld_reduce.global.b32".to_string(),
                    instructions: 5,
                    functions: 1,
                }],
                stubs: vf_adapt::Stubs {
                    trap_only: vec!["k_stub".to_string()],
                    empty: vec!["k_empty".to_string()],
                },
            },
            image: Some(b".version 8.0\n.target sm_70\n.address_size 64\n.entry k_emu() { ret; }\n".to_vec()),
        };
        let encoded = encode_retargeted(&adapted).expect("encode_retargeted");
        let decoded = decode_retargeted(&encoded).expect("decode_retargeted");
        assert_eq!(decoded.outcome, adapted.outcome);
        assert_eq!(decoded.image, adapted.image);

        // Truncating or corrupting the LZ4 stream/length header must fail cleanly.
        assert!(decode_retargeted(&encoded[..encoded.len() - 1]).is_none());
        let mut bad_hdr = encoded.clone();
        let hdr_at = bad_hdr.len() - lz4_flex::compress_prepend_size(adapted.image.as_ref().unwrap()).len();
        bad_hdr[hdr_at + 4] = 0x0f;
        bad_hdr[hdr_at + 5] = 0x00;
        bad_hdr[hdr_at + 6] = 0x00;
        assert!(decode_retargeted(&bad_hdr).is_none());

        // File-level header + checksum round-trip rejects single-byte corruption too.
        let dir = test_dir("retargeted");
        let cs = CacheState::new(dir.clone(), DEFAULT_MAX_BYTES, unsafe { libc::geteuid() });
        let path = retargeted_path(&cs, 0x1234_5678_9abc_def0);
        atomic_write(&path, &encoded, 0x1234_5678_9abc_def0).unwrap();
        let read_back = checked_read(&path, 0x1234_5678_9abc_def0, cs.uid).unwrap();
        assert_eq!(decode_retargeted(&read_back).unwrap().image, adapted.image);
        let mut raw = std::fs::read(&path).unwrap();
        *raw.last_mut().unwrap() ^= 1;
        std::fs::write(&path, raw).unwrap();
        assert!(checked_read(&path, 0x1234_5678_9abc_def0, cs.uid).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_retargeted_rejects_mismatched_and_oversized_length() {
        let adapted = vf_adapt::Adapted {
            outcome: vf_adapt::Outcome::Retargeted {
                entries: 1,
                from_sm: 75,
                tensor_rewritten: 0,
                tanh_expanded: 0,
                trapped: Vec::new(),
                refused: Vec::new(),
                emulated: Vec::new(),
                replaced: Vec::new(),
                reported: Vec::new(),
                stubs: vf_adapt::Stubs::default(),
            },
            image: Some(b"abc".to_vec()),
        };
        let encoded = encode_retargeted(&adapted).expect("encode_retargeted");
        let stream_len = lz4_flex::compress_prepend_size(b"abc").len();
        let hdr_at = encoded.len() - stream_len;

        // Declared uncompressed length larger than actual stream output (64 vs 3).
        let mut wrong_len = encoded.clone();
        wrong_len[hdr_at..hdr_at + 4].copy_from_slice(&64u32.to_le_bytes());
        assert!(
            decode_retargeted(&wrong_len).is_none(),
            "declared length mismatch must be rejected"
        );

        // Oversized declared uncompressed length (> 1 GiB) must be rejected without allocating.
        let mut oversized = encoded.clone();
        oversized[hdr_at..hdr_at + 4].copy_from_slice(&((MAX_RETARGETED_IMAGE_BYTES as u32) + 1).to_le_bytes());
        assert!(decode_retargeted(&oversized).is_none());

        let mut max_u32 = encoded;
        max_u32[hdr_at..hdr_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_retargeted(&max_u32).is_none());

        // Empty retargeted image round-trips cleanly.
        let empty = vf_adapt::Adapted {
            outcome: adapted.outcome,
            image: Some(Vec::new()),
        };
        let empty_enc = encode_retargeted(&empty).expect("encode empty");
        let empty_dec = decode_retargeted(&empty_enc).expect("decode empty");
        assert_eq!(empty_dec.image.as_deref(), Some(&[][..]));
    }

    #[test]
    fn key_for_policy_tracks_explicit_policy() {
        let img = b"sample module bytes";
        let mut p75 = vf_adapt::Policy::default();
        p75.told_sm = Some(75);
        p75.keep_sources = true;
        let mut p80 = vf_adapt::Policy::default();
        p80.told_sm = Some(80);
        p80.keep_sources = false;
        assert_ne!(key_for_policy(img, &p75), key_for_policy(img, &p80));
        assert_ne!(key_for_policy(img, &p75), key_for(img));
    }

    #[test]
    fn maybe_enforce_limit_avoids_rescanning_every_write() {
        let dir = test_dir("maybelimit-2k");
        let uid = unsafe { libc::geteuid() };
        let cs = CacheState::new(dir.clone(), DEFAULT_MAX_BYTES, uid);
        let payload = [0xABu8; 512];
        for i in 0u128..2000 {
            let path = manifest_path(&cs, i + 1);
            let written = atomic_write(&path, &payload, i + 1).unwrap();
            maybe_enforce_limit(&cs, written);
        }
        let scans = cs.limit_scans.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            scans <= 6,
            "expected periodic scans (<= 6 across 2000 writes), got {scans}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn maybe_enforce_limit_evicts_when_writes_cross_max() {
        let dir = test_dir("maybelimit-evict");
        let uid = unsafe { libc::geteuid() };
        let cs = CacheState::new(dir.clone(), 200, uid);
        for i in 0u128..10 {
            let path = dir.join(format!("f{i}.bin"));
            let written = atomic_write(&path, &[0u8; 50], i).unwrap();
            maybe_enforce_limit(&cs, written);
        }
        let total: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum();
        assert!(total <= 200, "total {total} > 200");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
