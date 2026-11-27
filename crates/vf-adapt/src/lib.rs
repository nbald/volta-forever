//! Decide what the driver of a Volta GPU should load in place of an image that carries nothing for it.
//!
//! An image is a fat binary, a cubin or PTX text. The driver picks from a fat binary the entry made for
//! its GPU: machine code for the same SM major version with a minor version it reaches, or PTX whose
//! target it reaches. CUDA 13 builds stop at sm_75, so on sm_70 nothing is picked. What can be done:
//!
//! - PTX entries for a later target are rewritten as sm_70 PTX, which the driver's JIT compiles
//!   natively: the instruction stream nvcc emits for sm_75 is valid for sm_70 except for a few
//!   instructions, which `vf_ptx::lower_to_volta` replaces by sequences Volta has (`tanh`, the
//!   Turing tensor-core instructions). A function that uses a form without such a sequence is made
//!   to trap, and the kernels that need it are listed: they are not to be given to the program.
//! - Where there is no PTX, supported sm_75 and sm_80 instructions are translated to sm_70
//!   (`vf_sass`), function by function. The functions the translator refuses are left as stubs,
//!   and listed: their kernels are not to be given to the program either.
//! - When the driver's compiler still refuses functions of an adapted image, `without_refused` makes
//!   the image to load instead.

mod contraction;
pub mod emulation;

use vf_fatbin::{self as fatbin, Entry, FatBinary, KIND_ELF, KIND_PTX, NewEntry};
pub use vf_ptx::{Emulated, EmulatedCounts, Refused, RefusedFunction, Replaced, Reported, Trapped};

/// Kernels that an image holds without code: their code in the image is a trap and nothing else
/// (`vf_ptx::stubs`, `vf_sass::effects::entry_stubs`), which is how a kernel built for other
/// architectures only is emitted for the others, or it does nothing. Neither is refused at a lookup:
/// fat binaries are full of them, and programs look them up without launching them. The layer
/// refuses a launch of the first (launched, it would end the context on any GPU) and names the
/// second at log level 2. A kernel the layer refuses is in neither list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stubs {
    /// Kernels whose code is a trap and nothing else.
    pub trap_only: Vec<String>,
    /// Kernels whose code does nothing.
    pub empty: Vec<String>,
}

impl Stubs {
    /// `self` with what `more` found.
    pub fn extend(&mut self, more: Stubs) {
        self.trap_only.extend(more.trap_only);
        self.empty.extend(more.empty);
    }

    /// Whether neither list has a kernel.
    pub fn is_empty(&self) -> bool {
        self.trap_only.is_empty() && self.empty.is_empty()
    }

    /// `self` without the kernels that the layer refuses: their code is the layer's own trap, or
    /// their refusal says more than this.
    fn without(mut self, refused: &[Refused]) -> Stubs {
        let kept = |name: &String| !refused.iter().any(|r| r.kernel == *name);
        self.trap_only.retain(kept);
        self.empty.retain(kept);
        self
    }

    /// The kernels of PTX `text` (as the program gave it, before it is lowered) without code.
    fn of_ptx(text: &[u8]) -> Stubs {
        let found = vf_ptx::stubs::stubs(text);
        Stubs {
            trap_only: found.trap_only,
            empty: found.empty,
        }
    }

    /// The kernels of `cubin`, machine code for `sm`, without code. Only the encodings of sm_70 and
    /// sm_75 are read; others give nothing.
    fn of_cubin(cubin: &[u8], sm: u32) -> Stubs {
        if !matches!(sm, 70 | 72 | 75) {
            return Stubs::default();
        }
        vf_sass::effects::entry_stubs(cubin)
            .map(|found| Stubs {
                trap_only: found.trap_only,
                empty: found.empty,
            })
            .unwrap_or_default()
    }
}

/// The lowest CUDA 13 machine-code target and the default capability reported on Volta.
pub const TRANSLATED_FROM: u32 = 75;

/// The GPU and the driver the image is adapted for.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// SM architecture of the GPU (70 for Volta).
    pub device_sm: u32,
    /// Newest PTX ISA version the driver's JIT accepts (9.0 for the 580 branch).
    pub ptx_max: (u32, u32),
    /// What the rewriting of instructions may take for granted about the programs.
    pub lowering: vf_ptx::lower::Options,
    /// The architecture the programs are told the GPU has, when the user chose it
    /// (`VF_CAPABILITY`): the PTX taken from a fat binary is then the PTX for that architecture.
    /// Machine code is taken from sm_75 where a fat binary has some: a program whose code is taken
    /// from there is told 7.5 instead (`told_limit`), and the images it loads are adapted with 7.5
    /// here.
    /// `None`: a program is told the lowest architecture it has code for, and that code is taken.
    pub told_sm: Option<u32>,
    /// When supported machine code is translated for the device.
    pub translate: Translate,
    /// Whether pairs of products on whole rows are translated as such in machine code
    /// (`VF_PAIRS=0`: they are not, nor in safe mode). The same variable sets `lowering.pairs`, the
    /// pairs of the program-text lowering.
    pub pairs: bool,
    /// Opt-in companion register hoisting and per-group HMMA scheduling (`VF_COMPANIONS=1`, off by
    /// default): hoists loop-invariant whole-row load addresses into companion registers, assigns
    /// per-group scoreboards from 0..4 to interfering pairs, and marks paired `HMMA.884` steps
    /// with `usched=18` and `.reuse` (`docs/pairs.md`, sections 8 and 12). Unused when `pairs` is
    /// false, and turned off in safe mode (`make_safe`).
    pub companions: bool,
    /// Where the translation of machine code takes its registers (`VF_BORROW`; the values are
    /// those of `vf_sass::translate::Options::borrow`). The default, 7: above the program's
    /// wherever that translates, and for a kernel refused for want of registers, among the
    /// program's own that hold nothing, then with nothing kept in a register at all, kernels with
    /// tensor-core products included.
    pub borrow: u32,
    /// For an image that the driver loads for every GPU of the process (a library,
    /// `cuLibraryLoadData`): the entries of a fat binary that are rewritten or translated for the
    /// device are kept beside what replaces them, so that a later card in the same machine is
    /// still given its own code. The driver of the device takes the replacement (machine code for
    /// it, or PTX it reaches); the later card, its entries as they were.
    pub keep_sources: bool,
    /// Safe mode (`VF_SAFE=1`, `Safe`): the switches whose results are not bit-identical to the
    /// default's are at their default, and each product of machine code is translated by itself
    /// (`make_safe`). In the policy, so that what is made in safe mode and what is made without it
    /// are kept apart where the policy is a key (the proxy's cache hashes it), even where they would
    /// be the same bytes.
    pub safe: bool,
}

/// Safe mode, asked for with `VF_SAFE`: the layer keeps its conservative paths, those whose results
/// are bit-identical to its conservative reference's (`Policy::make_safe`). Only the exact value `1`
/// turns it on; unset, `0` and any other value (empty, `true`, `yes`, ` 1`, `01`) leave it off, as the
/// bf16 cuBLAS variants read the same variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Safe {
    /// Unset, or `0`.
    Off,
    /// `1`.
    On,
    /// Another value, which leaves safe mode off and is said once.
    NotOne(String),
}

impl Safe {
    /// What a value of `VF_SAFE` asks for (`None`: the variable is not set).
    pub fn from_value(value: Option<&std::ffi::OsStr>) -> Safe {
        match value {
            None => Safe::Off,
            Some(value) if value == "1" => Safe::On,
            Some(value) if value == "0" => Safe::Off,
            Some(value) => Safe::NotOne(value.to_string_lossy().into_owned()),
        }
    }

    pub fn on(&self) -> bool {
        *self == Safe::On
    }
}

/// `VF_SAFE`, read once in a process, at the first call.
pub fn safe_mode() -> &'static Safe {
    static SAFE: std::sync::OnceLock<Safe> = std::sync::OnceLock::new();
    SAFE.get_or_init(|| Safe::from_value(std::env::var_os("VF_SAFE").as_deref()))
}

/// The line to say once about safe mode, `safe` as `safe_mode` gives it and `turned_off` what
/// `Policy::make_safe` turned off: `None` when there is nothing to say (unset, or `0`).
pub fn safe_line(safe: &Safe, turned_off: &[&str]) -> Option<String> {
    match safe {
        Safe::Off => None,
        Safe::NotOne(value) => Some(format!("VF_SAFE='{value}' is not 1: safe mode stays off")),
        Safe::On if turned_off.is_empty() => Some(
            "VF_SAFE=1: the layer keeps its conservative paths (nothing that safe mode turns off is in use)"
                .to_string(),
        ),
        Safe::On => Some(format!(
            "VF_SAFE=1: the layer keeps its conservative paths: {} not used (results not bit-identical to the \
             reference's)",
            turned_off.join(", ")
        )),
    }
}

/// When machine code is translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Translate {
    /// Never: an image with nothing but such code is left as it is.
    Never,
    /// When an image has nothing else for the device: no PTX to rewrite.
    WithoutPtx,
    /// Whenever an image has such code, also where it has PTX that could be rewritten: to set the two
    /// ways against each other.
    First,
    /// Translate sm_75 machine code even when sm_70 code is available: to isolate the effect of the
    /// translation from the effect of the compiler target. The sm_70 entries are removed.
    Force,
}

/// A compute capability written `<major>.<minor>`, as an SM architecture number (8.6 is sm_86).
pub fn capability_sm(text: &str) -> Option<u32> {
    let (major, minor) = text.trim().split_once('.')?;
    let (major, minor): (u32, u32) = (major.parse().ok()?, minor.parse().ok()?);
    if minor >= 10 {
        return None;
    }
    major.checked_mul(10)?.checked_add(minor)
}

/// The capability that `value`, a value of `VF_CAPABILITY`, asks the programs to be told, as an
/// SM number: `Ok(None)` for an empty value; `Err`, with what to say, for a value that is not a
/// compute capability written `<major>.<minor>` and for one below the 7.0 of the GPU, which are
/// not used.
pub fn capability_from(value: &str) -> Result<Option<u32>, String> {
    if value.trim().is_empty() {
        return Ok(None);
    }
    match capability_sm(value) {
        None => Err(format!(
            "VF_CAPABILITY={value} is not a compute capability (<major>.<minor>, such as 8.0): it is not used"
        )),
        Some(sm) if sm < 70 => Err(format!(
            "VF_CAPABILITY={value} is below the 7.0 of the GPU: it is not used"
        )),
        Some(sm) => Ok(Some(sm)),
    }
}

/// `capability_from` the environment: `Ok(None)` when `VF_CAPABILITY` is not set.
pub fn requested_capability() -> Result<Option<u32>, String> {
    match std::env::var("VF_CAPABILITY") {
        Ok(value) => capability_from(&value),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err("VF_CAPABILITY is not text: it is not used".to_string()),
    }
}

/// With `VF_TRANSLATE_DUMP=DIR`, every cubin given to the translator is written to
/// `DIR/<pid>-<number>.sm_<target>.cubin` as it was received: the machine code a program really loads,
/// to be translated and checked apart (`reference/validate.py`). Libraries keep some of
/// theirs where `vf extract` does not find it. (The directory must exist; what cannot be
/// written is left out, and there is nowhere here to say so.)
fn dump_cubin(cubin: &[u8], sm: u32) {
    static NUMBER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if let Ok(dir) = std::env::var("VF_TRANSLATE_DUMP") {
        let number = NUMBER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = std::fs::write(format!("{dir}/{}-{number:04}.sm_{sm}.cubin", std::process::id()), cubin);
    }
}

impl Default for Policy {
    /// The layer takes programs to keep the rows that `ldmatrix` reads at multiples of 16, as the PTX
    /// manual requires of them (`vf_ptx::lower::Options::aligned_rows`): the programs measured do,
    /// and testing it costs every program 17 % to a factor of two on its emulated tensor-core kernels
    /// (`docs/emulations.md`, "Matrix loads and moves"). A program that does not stops with a misaligned-address fault;
    /// `with_environment` lets its user ask for the test.
    fn default() -> Self {
        Policy {
            device_sm: 70,
            ptx_max: (9, 0),
            lowering: vf_ptx::lower::Options {
                aligned_rows: true,
                pairs: true,
                reported_traps: true,
                ..Default::default()
            },
            told_sm: None,
            translate: Translate::WithoutPtx,
            pairs: true,
            companions: false,
            borrow: vf_sass::translate::BORROW_DEFAULT,
            keep_sources: false,
            safe: false,
        }
    }
}

/// What was found in an image and what was done about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The image already holds code the driver uses for this GPU: left as is.
    Native,
    /// PTX for architecture `from_sm` was rewritten for the device (`entries` fat-binary entries, or 1
    /// for PTX text). `tensor_rewritten` and `tanh_expanded` count the tensor-core and `tanh`
    /// instructions replaced by sequences the device has; the `trapped` functions use an instruction
    /// without such a sequence and now trap.
    Retargeted {
        entries: usize,
        from_sm: u32,
        tensor_rewritten: usize,
        tanh_expanded: usize,
        trapped: Vec<Trapped>,
        /// The kernels that are, or call, a trapped function: not to be given to the program.
        refused: Vec<Refused>,
        /// The kernels that run emulated instructions, with the counts by kind.
        emulated: Vec<Emulated>,
        /// The functions in which instructions were replaced by a reported trap
        /// (`vf_ptx::lower::Options::reported_traps`).
        replaced: Vec<Replaced>,
        /// The kernels given that are, or may reach, one of those functions.
        reported: Vec<Reported>,
        /// The kernels without code (`Stubs`).
        stubs: Stubs,
    },
    /// Machine code for architecture `from_sm` was translated to machine code for the device
    /// (`entries` fat-binary entries, or 1 for a cubin), `functions` functions in all.
    Translated {
        entries: usize,
        from_sm: u32,
        functions: usize,
        /// The functions the translator refused, which are stubs in the image: not to be given to
        /// the program.
        refused: Vec<Refused>,
        /// The functions that have more instructions than they had (`EmulatedCounts::machine_code`).
        emulated: Vec<Emulated>,
        /// A function of the translation takes more registers than its budget (what it declared,
        /// rounded up to eight, 32 at least): the driver may refuse large blocks of it, which a
        /// second translation within the budget would have (`Policy::kept_for_second`).
        over_budget: bool,
        /// The kernels without code in the machine code that was translated (`Stubs`).
        stubs: Stubs,
        /// The functions whose translation computes tensor-core products two at a time on whole
        /// rows (`docs/pairs.md`; with `VF_PAIRS=0` or `VF_SAFE=1`, each is computed by itself).
        paired: Vec<String>,
    },
    /// Machine code for `from_sm` that the translator could not read: left as is.
    NotTranslated { from_sm: u32, reason: String },
    /// Only machine code for newer architectures that nothing translates; `lowest_sm` is the closest.
    NeedsTranslation { lowest_sm: u32 },
    /// Nothing recognised: left as is.
    Unknown,
}

/// What the fat binaries inside a file (a shared library, an executable) hold for the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Embedded {
    /// No fat binary.
    Nothing,
    /// Code the driver uses on the device.
    Usable,
    /// Code for other architectures only; `lowest_sm` is the lowest of them.
    Other { lowest_sm: u32 },
}

/// The result of adapting an image: `image` is the replacement, when there is one.
#[derive(Debug)]
pub struct Adapted {
    pub outcome: Outcome,
    pub image: Option<Vec<u8>>,
}

/// What to load after the driver's compiler refused functions of an adapted image.
#[derive(Debug)]
pub struct Retry {
    /// The image with those functions made to trap.
    pub image: Vec<u8>,
    pub functions: Vec<RefusedFunction>,
    /// The kernels that are or call one of them: not to be given to the program.
    pub kernels: Vec<Refused>,
}

impl Adapted {
    fn unchanged(outcome: Outcome) -> Adapted {
        Adapted { outcome, image: None }
    }

    fn complete_ptx(&self) -> bool {
        matches!(&self.outcome, Outcome::Retargeted { trapped, refused, .. } if trapped.is_empty() && refused.is_empty())
    }
}

/// A cubin translated for the device.
struct Translation {
    cubin: Vec<u8>,
    functions: usize,
    refused: Vec<Refused>,
    emulated: Vec<Emulated>,
    over_budget: bool,
    stubs: Stubs,
    paired: Vec<String>,
}

impl Translation {
    /// Replacing this image with PTX cannot discard a working machine-code entry point.
    fn all_entries_refused(&self, cubin: &[u8]) -> bool {
        !self.refused.is_empty()
            && fatbin::cubin_entries(cubin).is_some_and(|names| {
                !names.is_empty() && names.iter().all(|name| self.refused.iter().any(|r| r.kernel == *name))
            })
    }
}

/// Reconstruct the single PTX module kept as NUL-separated lines by ptxas. Match the
/// architecture and the complete set of entry points before handing it to the usual PTX path.
fn embedded_ptx(cubin: &[u8], sm: u32) -> Option<(Vec<u8>, vf_ptx::Header)> {
    if fatbin::cubin_sm(cubin)? != sm {
        return None;
    }
    let source = fatbin::cubin_debug_ptx(cubin)?;
    if source.last() != Some(&0) || !source.is_ascii() {
        return None;
    }
    let mut text = source.to_vec();
    for byte in &mut text {
        if *byte == 0 {
            *byte = b'\n';
        }
    }
    let mut directives = text.split(|&b| b == b'\n').filter_map(|line| {
        let keyword = line.trim_ascii_start().split(|b| b.is_ascii_whitespace()).next()?;
        matches!(keyword, b".version" | b".target").then_some(keyword)
    });
    let header = vf_ptx::header(&text)?;
    if directives.next() != Some(b".version")
        || directives.next() != Some(b".target")
        || directives.next().is_some()
        || header.target != sm
    {
        return None;
    }
    let mut names = fatbin::cubin_entries(cubin)?;
    let mut entries: Vec<_> = vf_ptx::functions(&text)
        .into_iter()
        .filter(|f| f.entry)
        .map(|f| &text[f.name])
        .collect();
    names.sort_unstable();
    entries.sort_unstable();
    if entries.is_empty() || !names.iter().map(|name| name.as_bytes()).eq(entries) {
        return None;
    }
    // Compiled again, the source must call what the machine code calls (`__assertfail` in Triton's
    // bounds checks): every function the cubin leaves to the loader is one the source declares.
    let imports = fatbin::cubin_imports(cubin)?;
    if !imports.is_empty() {
        let declared: Vec<_> = vf_ptx::extern_functions(&text)
            .into_iter()
            .map(|name| &text[name])
            .collect();
        if !imports.iter().all(|name| declared.contains(&name.as_bytes())) {
            return None;
        }
    }
    // Compiled again, the source must round as the machine code does: what ptxas did not contract
    // stays so, and a source of which that is not known is not used (`contraction`).
    let text = contraction::as_compiled(cubin, text)?;
    Some((text, header))
}

impl Policy {
    /// Offline CUDA 12.9 tools accept PTX ISA through 8.8 on Volta. What they make is not
    /// announced by the layer where it is loaded, and the GPUs of the process that loads it are not
    /// known to them: no instruction is replaced by a reported trap (`reported_traps`); the functions
    /// that hold one trap, and their kernels are refused, as before 0.1.6.
    pub fn cuda_12_9() -> Self {
        let mut policy = Self {
            ptx_max: (8, 8),
            ..Self::default()
        };
        policy.lowering.reported_traps = false;
        policy
    }

    /// This policy with what the environment asks for. `VF_ALIGNED_ROWS=0`: do not take the rows
    /// that `ldmatrix` reads to be at multiples of 16; test their addresses when the code runs, for a
    /// program that does not follow the PTX manual there (slower: see `Policy::default`).
    /// `VF_BF16_TC_GROUPS=1`: use guarded Volta FP16 Tensor Cores for groups of BF16 products.
    /// `VF_BF16_TC_TILES=1`: apply the same guarded conversion to two-by-two BF16 tiles.
    /// `VF_BF16_COLUMN=ordered|tree`: specialize four BF16 products with one active B column.
    /// `VF_CAPABILITY=<major>.<minor>`: the compute capability the programs are told (`told_sm`);
    /// a value that is not one, or is below 7.0, is not used (`requested_capability`).
    /// `VF_TRANSLATE=0`: do not translate machine code; `VF_TRANSLATE=first`: translate it also
    /// where an image has PTX (`Translate`). `VF_PAIRS=0`: compute every tensor-core product by
    /// itself, in machine code (`pairs`) and in program text (`lowering.pairs`). `VF_COMPANIONS=1`:
    /// hoist loop-invariant whole-row load addresses into companion registers and enable per-group
    /// paired HMMA scheduling (`companions`; off by default, unused when `VF_PAIRS=0`). `VF_SAFE=1`: what
    /// these ask for whose results are not bit-identical to the default's is not used, nor the pairs
    /// of machine code (`make_safe`). `VF_REPORTED_TRAPS=0` (that value only: any other leaves it
    /// on): no instruction is replaced by a reported trap (`lowering.reported_traps`): the functions
    /// that hold one trap, and their kernels are refused.
    pub fn with_environment(self) -> Policy {
        self.with_environment_said().0
    }

    /// `with_environment`, and the line to say once about safe mode (`safe_line`; `None`: nothing
    /// to say).
    pub fn with_environment_said(self) -> (Policy, Option<String>) {
        let mut policy = self.with_switches();
        let safe = safe_mode();
        let turned_off = if safe.on() { policy.make_safe() } else { Vec::new() };
        (policy, safe_line(safe, &turned_off))
    }

    /// Safe mode (`Safe`): the settings of this policy whose results are not bit-identical to the
    /// reference's are put back to it, and `safe` is set. Returns what was turned off, as the
    /// variables and values that ask for it. The reference of lowered and translated code is the
    /// program's own instructions as the default computes them, with each product of machine code
    /// by itself:
    ///
    /// - `VF_BF16_TC_GROUPS=1`, `VF_BF16_TC_TILES=1`: bf16 products on Volta's fp16 tensor cores, whose
    ///   sums are truncated, and the path chosen by a vote over every operand (`docs/emulations.md`,
    ///   "bf16");
    /// - `VF_BF16_COLUMN=tree`, `VF_BF16_ROW=tree`: partial sums added in another order, the path
    ///   chosen by a vote (their `ordered` modes give the default's bits and stay);
    /// - `VF_TRANSLATE=first`, `VF_TRANSLATE=force`: another form of the program, its machine code
    ///   translated where the default takes its PTX, its sm_75 code translated where its sm_70 code
    ///   runs (`VF_TRANSLATE=0` refuses what it does not translate, and stays);
    /// - `pairs`, on by default (said as `VF_PAIRS in machine code`): two products of sm_75 machine
    ///   code computed together add the halves of k, where a product by itself adds the even k, then
    ///   the odd (`docs/pairs.md`, section 9), and whether they are made depends on the registers of
    ///   the kernel and on the block that it requires (none at level 3 of `borrow`). Each product is
    ///   then translated by itself at every level, as with `VF_PAIRS=0` in machine code.
    ///
    /// Not changed: what the programs are told (`told_sm`): the code a program then chooses is its
    /// own choice; nor `lowering.pairs` (the pairs of program text give the bits of each product by
    /// itself: `docs/pairs.md`, section 13), `borrow`, `lowering.aligned_rows` and
    /// `lowering.reported_traps` (a reported trap computes nothing: where a refusal would not have
    /// given the kernel, it stops the kernel if it is ever reached, and every path that does not
    /// reach it runs the program's own instructions; turning it off would refuse NCCL's collectives,
    /// and safe mode could then not say whether a difference of a run on several GPUs comes from
    /// the layer).
    pub fn make_safe(&mut self) -> Vec<&'static str> {
        use vf_ptx::lower::{Bf16Column, Bf16Row};
        let mut turned_off = Vec::new();
        if self.lowering.bf16_tc_groups {
            self.lowering.bf16_tc_groups = false;
            turned_off.push("VF_BF16_TC_GROUPS=1");
        }
        if self.lowering.bf16_tc_tiles {
            self.lowering.bf16_tc_tiles = false;
            turned_off.push("VF_BF16_TC_TILES=1");
        }
        if self.lowering.bf16_column == Bf16Column::Tree {
            self.lowering.bf16_column = Bf16Column::Off;
            turned_off.push("VF_BF16_COLUMN=tree");
        }
        if self.lowering.bf16_row == Bf16Row::Tree {
            self.lowering.bf16_row = Bf16Row::Off;
            turned_off.push("VF_BF16_ROW=tree");
        }
        match self.translate {
            Translate::First => turned_off.push("VF_TRANSLATE=first"),
            Translate::Force => turned_off.push("VF_TRANSLATE=force"),
            Translate::Never | Translate::WithoutPtx => {}
        }
        if matches!(self.translate, Translate::First | Translate::Force) {
            self.translate = Translate::WithoutPtx;
        }
        if self.companions {
            self.companions = false;
            turned_off.push("VF_COMPANIONS=1");
        }
        if self.pairs {
            self.pairs = false;
            turned_off.push("VF_PAIRS in machine code");
        }
        self.safe = true;
        turned_off
    }

    /// This policy with the switches that the environment sets, safe mode apart.
    fn with_switches(mut self) -> Policy {
        match std::env::var("VF_TRANSLATE").as_deref() {
            Ok("0") => self.translate = Translate::Never,
            Ok("first") => self.translate = Translate::First,
            Ok("force") => self.translate = Translate::Force,
            _ => {}
        }
        if let Ok(value) = std::env::var("VF_ALIGNED_ROWS") {
            self.lowering.aligned_rows = value != "0";
        }
        if let Ok(value) = std::env::var("VF_BF16_TC_GROUPS") {
            self.lowering.bf16_tc_groups = value == "1";
        }
        if let Ok(value) = std::env::var("VF_BF16_TC_TILES") {
            self.lowering.bf16_tc_tiles = value == "1";
        }
        if let Ok(value) = std::env::var("VF_BF16_COLUMN") {
            use vf_ptx::lower::Bf16Column;
            self.lowering.bf16_column = match value.as_str() {
                "ordered" => Bf16Column::Ordered,
                "tree" => Bf16Column::Tree,
                _ => Bf16Column::Off,
            };
        }
        if let Ok(value) = std::env::var("VF_BF16_ROW") {
            use vf_ptx::lower::Bf16Row;
            self.lowering.bf16_row = match value.as_str() {
                "ordered" => Bf16Row::Ordered,
                "tree" => Bf16Row::Tree,
                _ => Bf16Row::Off,
            };
        }
        if let Ok(Some(sm)) = requested_capability() {
            self.told_sm = Some(sm);
        }
        if let Ok(value) = std::env::var("VF_PAIRS") {
            self.pairs = value != "0";
            self.lowering.pairs = self.pairs;
        }
        if let Ok(value) = std::env::var("VF_COMPANIONS") {
            self.companions = value == "1";
        }
        if let Ok(value) = std::env::var("VF_BORROW") {
            self.borrow = vf_sass::translate::borrow_from(Some(&value));
        }
        if std::env::var("VF_REPORTED_TRAPS").as_deref() == Ok("0") {
            self.lowering.reported_traps = false;
        }
        self
    }

    /// Machine code for `sm` runs on the device: same major version, minor version not above.
    fn runs(&self, sm: u32) -> bool {
        sm / 10 == self.device_sm / 10 && sm <= self.device_sm
    }

    fn jit_accepts(&self, entry: &Entry<'_>) -> bool {
        let (major, minor) = entry.content_version();
        entry.sm() <= self.device_sm && (major as u32, minor as u32) <= self.ptx_max
    }

    /// PTX that can be retargeted to the device: PTX written for any later architecture. What decides
    /// is its instructions: `vf_ptx::lower_to_volta` rewrites those it has a sequence for and traps
    /// the functions that use others it knows of. It does not know everything later targets have, so a
    /// retargeted module can still be refused by the compiler, which is for the caller to handle.
    pub fn retargetable(&self, sm: u32) -> bool {
        sm > self.device_sm
    }

    /// The driver uses this fat-binary entry on the device.
    pub fn usable(&self, entry: &Entry<'_>) -> bool {
        match entry.kind() {
            KIND_ELF => self.runs(entry.sm()),
            KIND_PTX => self.jit_accepts(entry),
            _ => false,
        }
    }

    /// Look through `file` for fat binaries until one holds code for the device.
    pub fn embedded(&self, file: &[u8]) -> Embedded {
        let magic = fatbin::MAGIC.to_le_bytes();
        let mut lowest: Option<u32> = None;
        let mut offset = 0;
        while let Some(found) = file[offset..].windows(4).position(|w| w == magic) {
            let at = offset + found;
            offset = at + 4;
            let Ok(fat) = fatbin::parse(&file[at..]) else {
                continue;
            };
            if fat.entries.is_empty() {
                continue;
            }
            if fat.entries.iter().any(|entry| self.usable(entry)) {
                return Embedded::Usable;
            }
            let code = fat.entries.iter().filter(|e| matches!(e.kind(), KIND_ELF | KIND_PTX));
            lowest = code.map(|e| e.sm()).chain(lowest).min();
            offset = at + fat.len;
        }
        lowest.map_or(Embedded::Nothing, |lowest_sm| Embedded::Other { lowest_sm })
    }

    /// `text` for the device: Turing-only instructions lowered, then the header; and its kernels
    /// without code (`Stubs`), but those the lowering refuses.
    fn lower_ptx(&self, text: &[u8]) -> (Vec<u8>, vf_ptx::Lowering, Stubs) {
        let (mut lowered, lowering) = if !vf_ptx::needs_lowering(text) {
            (text.to_vec(), vf_ptx::Lowering::default())
        } else {
            vf_ptx::lower_to_volta_with(text, self.lowering)
        };
        if self.device_sm == 70 {
            vf_ptx::division::specialize(&mut lowered);
        }
        vf_ptx::retarget(&mut lowered, self.ptx_max, self.device_sm);
        let stubs = Stubs::of_ptx(text).without(&lowering.refused);
        (lowered, lowering, stubs)
    }

    /// Adapt PTX text (no terminating zero needed, none added).
    pub fn adapt_ptx(&self, text: &[u8]) -> Adapted {
        let Some(header) = vf_ptx::header(text) else {
            return Adapted::unchanged(Outcome::Unknown);
        };
        if header.target <= self.device_sm && header.version <= self.ptx_max {
            return Adapted::unchanged(Outcome::Native);
        }
        if header.target > self.device_sm && !self.retargetable(header.target) {
            return Adapted::unchanged(Outcome::NeedsTranslation {
                lowest_sm: header.target,
            });
        }
        let (image, lowering, stubs) = self.lower_ptx(text);
        let outcome = Outcome::Retargeted {
            entries: 1,
            from_sm: header.target,
            tensor_rewritten: lowering.tensor_rewritten,
            tanh_expanded: lowering.tanh_expanded,
            trapped: lowering.trapped,
            refused: lowering.refused,
            emulated: lowering.emulated,
            replaced: lowering.replaced,
            reported: lowering.reported,
            stubs,
        };
        Adapted {
            outcome,
            image: Some(image),
        }
    }

    /// Machine code for `sm` can be translated for the device.
    /// Whether the layer translates machine code for `sm` to the device's.
    pub fn translates(&self, sm: u32) -> bool {
        self.translatable(sm)
    }

    fn translatable(&self, sm: u32) -> bool {
        self.translate != Translate::Never && self.device_sm == 70 && matches!(sm, 75 | 80)
    }

    /// Translate one image of the selected architecture into machine code for the device.
    fn translate_cubin(&self, cubin: &[u8], sm: u32) -> Result<Translation, String> {
        if fatbin::cubin_sm(cubin) != Some(sm) {
            return Err(format!("the cubin header does not match sm_{sm}"));
        }
        dump_cubin(cubin, sm);
        let opts = vf_sass::translate::Options {
            pairs: self.pairs,
            aligned_rows: self.lowering.aligned_rows,
            companions: self.companions,
            borrow: self.borrow,
        };
        let translated = vf_sass::translate::translate_with_options(cubin, false, &opts)?;
        let (mut refused, mut emulated) = (Vec::new(), Vec::new());
        for function in &translated.report {
            match &function.refused {
                Some(reason) => refused.push(Refused {
                    kernel: function.name.clone(),
                    function: function.name.clone(),
                    reason: format!("its machine code for sm_{sm} is not translated ({reason})"),
                }),
                None if function.after > function.before => emulated.push(Emulated {
                    kernel: function.name.clone(),
                    counts: EmulatedCounts {
                        machine_code: function.after - function.before,
                        ..EmulatedCounts::default()
                    },
                }),
                None => {}
            }
        }
        let stubs = Stubs::of_cubin(cubin, sm).without(&refused);
        Ok(Translation {
            over_budget: translation_exceeds_budget(cubin, &translated.bytes),
            paired: translated
                .report
                .iter()
                .filter(|function| function.paired)
                .map(|function| function.name.clone())
                .collect(),
            cubin: translated.bytes,
            functions: translated.report.len(),
            refused,
            emulated,
            stubs,
        })
    }

    /// Adapt a cubin. Native images need only their header. Later executables can retain PTX or
    /// use an instruction set the machine-code translator understands.
    pub fn adapt_cubin(&self, cubin: &[u8]) -> Adapted {
        match fatbin::cubin_sm(cubin) {
            Some(sm) if self.runs(sm) => Adapted::unchanged(Outcome::Native),
            // Ampere images with retained PTX already run through the PTX lowerer.
            Some(80)
                if self.device_sm == 70
                    && self.translate == Translate::WithoutPtx
                    && let Some((text, _)) = embedded_ptx(cubin, 80) =>
            {
                self.adapt_ptx(&text)
            }
            Some(sm) if self.translatable(sm) => match self.translate_cubin(cubin, sm) {
                Ok(translation)
                    if sm == 75
                        && self.translate == Translate::WithoutPtx
                        && translation.all_entries_refused(cubin)
                        && let Some(recovered) = embedded_ptx(cubin, sm)
                            .map(|(text, _)| self.adapt_ptx(&text))
                            .filter(Adapted::complete_ptx) =>
                {
                    recovered
                }
                Ok(translation) => Adapted {
                    outcome: Outcome::Translated {
                        entries: 1,
                        from_sm: sm,
                        functions: translation.functions,
                        refused: translation.refused,
                        emulated: translation.emulated,
                        over_budget: translation.over_budget,
                        stubs: translation.stubs,
                        paired: translation.paired,
                    },
                    image: Some(translation.cubin),
                },
                Err(reason) => Adapted::unchanged(Outcome::NotTranslated { from_sm: sm, reason }),
            },
            Some(sm) if self.retargetable(sm) => embedded_ptx(cubin, sm)
                .map(|(text, _)| self.adapt_ptx(&text))
                .unwrap_or_else(|| Adapted::unchanged(Outcome::NeedsTranslation { lowest_sm: sm })),
            Some(sm) => Adapted::unchanged(Outcome::NeedsTranslation { lowest_sm: sm }),
            None => Adapted::unchanged(Outcome::Unknown),
        }
    }

    /// Whether `adapt_cubin` would need all of a cubin with this header, not the header alone.
    pub fn translates_cubin(&self, header: &[u8]) -> bool {
        fatbin::cubin_sm(header).is_some_and(|sm| !self.runs(sm) && (self.translatable(sm) || self.retargetable(sm)))
    }

    /// The kernels of `cubin`, the machine code for `source` taken from `fat`, that are stubs for
    /// the architecture the program is told (`told_sm`), when that is of a later generation: their
    /// code writes no memory that outlives them (`vf_sass::effects`), where the code of the same
    /// kernels for the architecture that a GPU of the capability told runs (the latest of `fat` up to
    /// it) writes. A library compiled for several architectures picks the variant of a kernel by the
    /// capability it is told, and its code for an earlier architecture can have the variant for a
    /// later one as such a stub: told 8.0 (`VF_CAPABILITY_ALL=1`), PyTorch launches attention kernels
    /// that in its code for sm_75 print an error and write nothing, and goes on with what the output
    /// held. Refused, they make the program fail instead. A kernel that writes nothing in both, as one
    /// that only waits or only checks, and one of which either answer is not known, is given as
    /// before; the code for the later architecture is only read, not translated.
    fn stubs_for_told(&self, fat: &FatBinary<'_>, source: u32, cubin: &[u8]) -> Vec<Refused> {
        use vf_sass::effects::{Effect, kernel_effects};
        let (Some(later), Some(told)) = (self.told_machine_code(fat, source), self.told_sm) else {
            return Vec::new();
        };
        let with = |cubin: &[u8], wanted: Effect| -> Vec<String> {
            let found = kernel_effects(cubin).unwrap_or_default();
            found
                .into_iter()
                .filter(|(_, e)| *e == wanted)
                .map(|(name, _)| name)
                .collect()
        };
        let quiet = with(cubin, Effect::None);
        if quiet.is_empty() {
            return Vec::new();
        }
        let mut writing = std::collections::HashSet::new();
        for entry in fat.entries.iter().filter(|e| e.kind() == KIND_ELF && e.sm() == later) {
            if let Ok(payload) = entry.payload() {
                writing.extend(with(&payload, Effect::Writes));
            }
        }
        quiet
            .into_iter()
            .filter(|name| writing.contains(name))
            .map(|kernel| Refused {
                function: kernel.clone(),
                reason: format!(
                    "its machine code for sm_{source} writes no memory (a stub), where its code for sm_{later}, which \
                     a GPU of the capability told ({}.{}) runs, does; that code is not translated",
                    told / 10,
                    told % 10
                ),
                kernel,
            })
            .collect()
    }

    /// The architecture of the machine code of `fat` that a GPU of the capability told (`told_sm`)
    /// loads, when the program is told a later generation than `given`, the architecture of the
    /// machine code that `adapt_fatbin` translated for it: the latest of `fat` after `given` and up
    /// to what is told. `None` when the program is told nothing later, or `fat` has nothing there.
    /// A library compiled for several architectures chooses its kernels by the capability it is
    /// told, and the code for `given` can have them as stubs (`stubs_for_told`) or not have them at
    /// all: a kernel that is compiled for the later architectures only, or under another name in
    /// the code for the earlier ones (a template whose `enable_if` takes another overload there).
    /// The lookup of such a kernel fails (`has_kernel`).
    pub fn told_machine_code(&self, fat: &FatBinary<'_>, given: u32) -> Option<u32> {
        let told = self.told_sm.filter(|told| told / 10 > given / 10)?;
        fat.entries
            .iter()
            .filter(|e| e.kind() == KIND_ELF && e.sm() > given && e.sm() <= told)
            .map(Entry::sm)
            .max()
    }

    /// Why a program told a later generation is given the machine code of an image for `given`, not
    /// its code for `loads` that a GPU of the capability told loads (`told_machine_code`).
    /// (`machine_source`: by default the earliest code that the layer translates; with
    /// `VF_TRANSLATE=first` or `force`, the latest up to what is told, which is then `loads` when
    /// it translates that.)
    pub fn why_given(&self, given: u32, loads: u32) -> String {
        if self.translatable(loads) {
            format!(
                "the layer gives an image that has machine code for sm_{given} and sm_{loads} its code for \
                 sm_{given}, whatever the capability told, as its translation of machine code for sm_{loads} \
                 covers fewer instructions and is not independently validated"
            )
        } else {
            format!("the layer does not translate machine code for sm_{loads}")
        }
    }

    /// Translate the machine code for one target, keeping entries for other GPUs. With the
    /// default policy, use a later target's complete retained PTX when it is available.
    fn translate_fatbin(&self, fat: &FatBinary<'_>) -> Result<Option<Adapted>, fatbin::Error> {
        let Some(source) = self.machine_source(fat) else {
            return Ok(None);
        };
        if self.translate == Translate::WithoutPtx
            && let Some(sm) = self.source_of(fat, KIND_ELF).filter(|&sm| sm >= 80)
        {
            let recovered = self.retarget_fatbin(fat, KIND_ELF, sm)?;
            if matches!(recovered.outcome, Outcome::Retargeted { .. }) {
                return Ok(Some(recovered));
            }
        }
        let mut entries = Vec::with_capacity(fat.entries.len());
        let (mut translated, mut functions, mut refused, mut emulated) = (0, 0, Vec::new(), Vec::new());
        let mut over_budget = false;
        let mut found = Stubs::default();
        let mut paired = Vec::new();
        let mut all_refused = source == 75 && self.translate == Translate::WithoutPtx;
        for entry in &fat.entries {
            if entry.kind() != KIND_ELF || entry.sm() != source {
                // In Force mode, drop native sm_70 entries so the driver uses the translated ones.
                if self.translate == Translate::Force && entry.kind() == KIND_ELF && self.runs(entry.sm()) {
                    continue;
                }
                entries.push(NewEntry::copy_of(entry));
                continue;
            }
            let payload = entry.payload()?;
            match self.translate_cubin(&payload, source) {
                Ok(translation) => {
                    all_refused = all_refused && translation.all_entries_refused(&payload);
                    if self.keep_sources {
                        entries.push(NewEntry::copy_of(entry));
                    }
                    entries.push(NewEntry::replace(
                        entry,
                        translation.cubin,
                        self.device_sm,
                        entry.content_version(),
                    ));
                    translated += 1;
                    functions += translation.functions;
                    over_budget |= translation.over_budget;
                    refused.extend(translation.refused);
                    let stubs: Vec<Refused> = self
                        .stubs_for_told(fat, source, &payload)
                        .into_iter()
                        .filter(|stub| !refused.iter().any(|r: &Refused| r.kernel == stub.kernel))
                        .collect();
                    refused.extend(stubs);
                    emulated.extend(translation.emulated);
                    found.extend(translation.stubs);
                    paired.extend(translation.paired);
                }
                Err(reason) => {
                    return Ok(Some(Adapted::unchanged(Outcome::NotTranslated {
                        from_sm: entry.sm(),
                        reason,
                    })));
                }
            }
        }
        if all_refused {
            let recovered = self.retarget_fatbin(fat, KIND_ELF, source)?;
            if recovered.complete_ptx() {
                return Ok(Some(recovered));
            }
        }
        // (A kernel refused for its told architecture is not one without code.)
        let stubs = found.without(&refused);
        Ok(Some(Adapted {
            outcome: Outcome::Translated {
                entries: translated,
                from_sm: source,
                functions,
                refused,
                emulated,
                over_budget,
                stubs,
                paired,
            },
            image: Some(fatbin::build(&entries)),
        }))
    }

    /// The architecture to use for PTX entries or, when none exist, PTX retained inside cubins.
    ///
    /// A program takes its decisions (which kernels to launch, with which parameters) for the GPU it
    /// is told it runs on, and the driver of that GPU would give it the code of the highest target
    /// the GPU reaches. The code for a lower target can be other code or none at all: the attention
    /// kernels of llama.cpp for sm_75 are traps where those for sm_80 are the real thing, and told
    /// 8.0 it launches them. So when the user chose what the programs are told, that entry is taken.
    /// Otherwise a program is told the lowest architecture it has code for (`callers` in the proxy):
    /// PTX at or below the device target in a dialect too new for the driver, else the nearest
    /// target above. The same when nothing reaches what the user chose.
    fn source_of(&self, fat: &FatBinary<'_>, kind: u16) -> Option<u32> {
        self.preferred_sm(
            fat.entries
                .iter()
                .filter(|e| e.kind() == kind)
                .map(Entry::sm)
                .filter(|&sm| sm <= self.device_sm || self.retargetable(sm)),
        )
    }

    fn machine_source(&self, fat: &FatBinary<'_>) -> Option<u32> {
        let targets = fat
            .entries
            .iter()
            .filter(|e| e.kind() == KIND_ELF)
            .map(Entry::sm)
            .filter(|&sm| self.translatable(sm));
        // The default Turing translation is the established path for multi-target libraries,
        // including PyTorch. Ampere's supported instruction set is currently narrower. A program
        // whose code is taken from there is told 7.5, whatever the user chose (`told_limit`).
        if self.translate == Translate::WithoutPtx {
            targets.min()
        } else {
            self.preferred_sm(targets)
        }
    }

    /// The architecture of the code that `adapt_fatbin` gives for `fat`: of the PTX that it
    /// retargets (an entry, or what the cubins of a later target retain, compressed or not) or of
    /// the machine code that it translates; `None` when it takes code that runs on the device, or
    /// nothing. (Where the translator refuses every function of the sm_75 code, `translate_fatbin`
    /// takes the PTX that this code retains, when it has some: PTX of the same architecture.)
    pub fn given_from(&self, fat: &FatBinary<'_>) -> Option<u32> {
        let machine = self.machine_source(fat);
        if self.translate == Translate::Force && machine.is_some() {
            return machine;
        }
        if fat.entries.iter().any(|entry| self.usable(entry)) {
            return None;
        }
        if self.translate == Translate::First && machine.is_some() {
            return machine;
        }
        if let Some(sm) = self.source_of(fat, KIND_PTX) {
            return Some(sm);
        }
        // The cubins of `sm` all retain their PTX (`retarget_fatbin`).
        let retain = |sm: u32| {
            fat.entries
                .iter()
                .filter(|entry| entry.kind() == KIND_ELF && entry.sm() == sm)
                .all(|entry| entry.payload().is_ok_and(|cubin| embedded_ptx(&cubin, sm).is_some()))
        };
        let retained = self.source_of(fat, KIND_ELF);
        if machine.is_some() {
            if self.translate == Translate::WithoutPtx
                && let Some(sm) = retained.filter(|&sm| sm >= 80 && retain(sm))
            {
                return Some(sm);
            }
            return machine;
        }
        retained.filter(|&sm| self.retargetable(sm) && retain(sm))
    }

    /// The capability that a program whose code includes `fat` can be told, as an SM number, when
    /// it cannot be told the one the user chose (`told_sm`): the architecture of the code given for
    /// it (`given_from`), when the chosen one is of a later generation (and 7.5 for code of
    /// Volta's generation). A program picks the variants of its kernels for the GPU it is told it
    /// runs on, and in the code compiled for an earlier architecture, machine code or PTX, the
    /// variants for a later one are other code or none (what `__CUDA_ARCH__` chose was chosen
    /// when it was compiled): told 8.0, PyTorch launches attention kernels for sm_80 that in its
    /// sm_75 code print an error and write nothing; told 9.0, kernels that stop the GPU with an
    /// assertion. Machine code for the told generation is not taken in place of sm_75 machine code,
    /// because the translation of sm_80 code covers fewer instructions than that of sm_75 (it does
    /// not cover PyTorch's simplest kernel); PTX for the told generation is taken where there is
    /// some (`source_of`). `None`: what the user chose can be told.
    ///
    /// A program told the limit is given, for every image it loads, the code for the limit: the
    /// images are adapted with `told_sm` set to it (the proxy's `callers::told_sm`), which takes
    /// the sm_75 code or the PTX for sm_75 where an image also has code for a later target. (An
    /// image with nothing for the limit or an earlier architecture is given the nearest above, as
    /// when the user asks for the limit.)
    pub fn told_limit(&self, fat: &FatBinary<'_>) -> Option<u32> {
        let told = self.told_sm?;
        let earlier = |sm: &u32| sm / 10 < told / 10;
        // Code that runs on the device as it is: what it chose for a later generation is not there
        // either (`given_from` answers `None` for it, as for no code at all).
        if self.translate != Translate::Force && fat.entries.iter().any(|entry| self.usable(entry)) {
            return earlier(&self.device_sm).then_some(self.device_sm.max(TRANSLATED_FROM));
        }
        // `given_from` is one of these: where none is earlier, the cubins need not be read.
        let candidates = [
            self.source_of(fat, KIND_PTX),
            self.machine_source(fat),
            self.source_of(fat, KIND_ELF),
        ];
        if !candidates.iter().flatten().any(earlier) {
            return None;
        }
        self.given_from(fat).filter(earlier).map(|sm| sm.max(TRANSLATED_FROM))
    }

    /// `told_limit` for the code embedded in `file` (a shared library, an executable): the lowest
    /// over its fat binaries, lowered until nothing lowers it more (`lowered`).
    pub fn told_limit_in(&self, file: &[u8]) -> Option<u32> {
        self.lowered(|policy| policy.told_limit_once_in(file))
    }

    /// What a program is told when `limit_at`, the limit that its code sets for a policy that tells
    /// `told_sm` (`told_limit` over all of that code), is below what this policy tells: told the
    /// limit, the program is given the code for the limit (the images it loads are adapted with
    /// `told_sm` set to it), which can set a lower one. Told 9.0, code with PTX for sm_80 sets 8.0;
    /// told 8.0, an image with PTX for sm_75 and sm_90 is given its PTX for sm_75, which sets 7.5.
    /// So the limit is lowered until nothing lowers it more. `None`: nothing lowers what is told.
    pub fn lowered(&self, mut limit_at: impl FnMut(&Policy) -> Option<u32>) -> Option<u32> {
        let mut policy = *self;
        let mut limit = None;
        while let Some(sm) = limit_at(&policy).filter(|&sm| policy.told_sm.is_some_and(|told| sm < told)) {
            limit = Some(sm);
            policy.told_sm = Some(sm);
        }
        limit
    }

    /// `told_limit` over the fat binaries of `file`, once.
    fn told_limit_once_in(&self, file: &[u8]) -> Option<u32> {
        // Nothing limits what Volta's generation is told.
        if self.told_sm? / 10 <= TRANSLATED_FROM / 10 {
            return None;
        }
        let magic = fatbin::MAGIC.to_le_bytes();
        let mut lowest: Option<u32> = None;
        let mut offset = 0;
        while let Some(found) = file[offset..].windows(4).position(|w| w == magic) {
            let at = offset + found;
            offset = at + 4;
            let Ok(fat) = fatbin::parse(&file[at..]) else {
                continue;
            };
            lowest = self.told_limit(&fat).into_iter().chain(lowest).min();
            if lowest == Some(TRANSLATED_FROM) {
                // The lowest limit: no other fat binary can lower it.
                break;
            }
            offset = at + fat.len.max(4);
        }
        lowest
    }

    fn preferred_sm(&self, targets: impl Iterator<Item = u32> + Clone) -> Option<u32> {
        let told = self
            .told_sm
            .and_then(|told| targets.clone().filter(|&sm| sm <= told).max());
        let nearest = targets.min_by_key(|&sm| if sm <= self.device_sm { 0 } else { sm });
        told.or(nearest)
    }

    /// Adapt a fat binary: when nothing in it is for the device, rewrite the PTX entries of one
    /// architecture (`source_of`) as device PTX. Without PTX, select one machine-code target.
    /// An executable for a later target can supply the PTX that ptxas kept with its line information.
    /// The other entries are kept.
    pub fn adapt_fatbin(&self, fat: &FatBinary<'_>) -> Result<Adapted, fatbin::Error> {
        if self.translate == Translate::Force
            && let Some(adapted) = self.translate_fatbin(fat)?
        {
            return Ok(adapted);
        }
        if fat.entries.iter().any(|entry| self.usable(entry)) {
            return Ok(Adapted::unchanged(Outcome::Native));
        }
        if self.translate == Translate::First
            && let Some(adapted) = self.translate_fatbin(fat)?
        {
            return Ok(adapted);
        }
        let (kind, source) = if let Some(source) = self.source_of(fat, KIND_PTX) {
            (KIND_PTX, source)
        } else {
            if let Some(adapted) = self.translate_fatbin(fat)? {
                return Ok(adapted);
            }
            let Some(source) = self.source_of(fat, KIND_ELF) else {
                return Ok(Adapted::unchanged(Outcome::Unknown));
            };
            (KIND_ELF, source)
        };
        self.retarget_fatbin(fat, kind, source)
    }

    fn retarget_fatbin(&self, fat: &FatBinary<'_>, kind: u16, source: u32) -> Result<Adapted, fatbin::Error> {
        let mut entries = Vec::with_capacity(fat.entries.len());
        let (mut rewritten, mut tensor_rewritten, mut tanh_expanded, mut trapped) = (0, 0, 0, Vec::new());
        let (mut refused, mut emulated, mut stubs) = (Vec::new(), Vec::new(), Stubs::default());
        let (mut replaced, mut reported) = (Vec::new(), Vec::new());
        for entry in &fat.entries {
            if entry.kind() != kind || entry.sm() != source {
                entries.push(NewEntry::copy_of(entry));
                continue;
            }
            let payload = entry.payload()?;
            let (text, version) = if kind == KIND_ELF {
                // Every cubin of the selected architecture must carry its complete source.
                // Returning a partially rewritten image would silently hide some entry points.
                let Some((ptx, header)) = embedded_ptx(&payload, source).filter(|_| self.retargetable(source)) else {
                    let lowest_sm = fat
                        .entries
                        .iter()
                        .filter(|e| e.kind() == KIND_ELF)
                        .map(Entry::sm)
                        .min()
                        .unwrap_or(source);
                    return Ok(Adapted::unchanged(Outcome::NeedsTranslation { lowest_sm }));
                };
                (std::borrow::Cow::Owned(ptx), header.version)
            } else {
                let (major, minor) = entry.content_version();
                (payload, (u32::from(major), u32::from(minor)))
            };
            let (text, lowering, found) = self.lower_ptx(&text);
            stubs.extend(found);
            if self.keep_sources {
                entries.push(NewEntry::copy_of(entry));
            }
            tensor_rewritten += lowering.tensor_rewritten;
            tanh_expanded += lowering.tanh_expanded;
            trapped.extend(lowering.trapped);
            refused.extend(lowering.refused);
            emulated.extend(lowering.emulated);
            replaced.extend(lowering.replaced);
            reported.extend(lowering.reported);
            let (major, minor) = version.min(self.ptx_max);
            let version = (major as u16, minor as u16);
            let target = self.device_sm.min(source);
            entries.push(if kind == KIND_ELF {
                NewEntry::from_ptx(text, target, version, entry)
            } else {
                NewEntry::replace(entry, text, target, version)
            });
            rewritten += 1;
        }
        Ok(Adapted {
            outcome: Outcome::Retargeted {
                entries: rewritten,
                from_sm: source,
                tensor_rewritten,
                tanh_expanded,
                trapped,
                refused,
                emulated,
                replaced,
                reported,
                stubs,
            },
            image: Some(fatbin::build(&entries)),
        })
    }

    /// The image to load in place of `image`, which this policy made and of which the driver's
    /// compiler refused functions: its `log` names the lines. `image` is PTX text, with or without a
    /// final zero, or a fat binary whose one PTX entry for the device is what was compiled. `None`
    /// when the log names no line of a function, or when what was compiled is not known.
    pub fn without_refused(&self, image: &[u8], log: &str) -> Option<Retry> {
        let text_of =
            |bytes: &[u8]| -> usize { bytes.len() - bytes.iter().rev().take_while(|&&byte| byte == 0).count() };
        if !image.starts_with(&fatbin::MAGIC.to_le_bytes()) {
            let length = text_of(image);
            vf_ptx::header(&image[..length])?;
            let removal = vf_ptx::without_refused(&image[..length], log)?;
            let mut text = removal.text;
            text.extend_from_slice(&image[length..]);
            return Some(Retry {
                image: text,
                functions: removal.functions,
                kernels: removal.kernels,
            });
        }
        let fat = fatbin::parse(image).ok()?;
        let mut compiled = fat
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kind() == KIND_PTX && self.jit_accepts(entry));
        let (index, entry) = compiled.next()?;
        if compiled.next().is_some() {
            return None;
        }
        let payload = entry.payload().ok()?;
        let removal = vf_ptx::without_refused(&payload[..text_of(&payload)], log)?;
        let mut text = Some(removal.text);
        let entries: Vec<NewEntry> = fat
            .entries
            .iter()
            .enumerate()
            .map(|(at, other)| match text.take_if(|_| at == index) {
                Some(text) => NewEntry::replace(other, text, other.sm(), other.content_version()),
                None => NewEntry::copy_of(other),
            })
            .collect();
        Some(Retry {
            image: fatbin::build(&entries),
            functions: removal.functions,
            kernels: removal.kernels,
        })
    }
}

/// Whether the machine code of `fat` for `sm` has a kernel `name`: an entry point of one of its
/// cubins for that architecture (`Policy::told_machine_code`). A cubin that cannot be read has none.
pub fn has_kernel(fat: &FatBinary<'_>, sm: u32, name: &str) -> bool {
    fat.entries
        .iter()
        .filter(|e| e.kind() == KIND_ELF && e.sm() == sm)
        .any(|entry| {
            entry
                .payload()
                .is_ok_and(|cubin| fatbin::cubin_entries(&cubin).is_some_and(|names| names.contains(&name)))
        })
}

/// Per-function register counts extracted from a cubin's ELF sections.
pub struct FunctionRegisters {
    pub name: String,
    pub registers: u32,
}

/// Read the register count of each `.text.<func>` section from a cubin.  The count is in the
/// upper 8 bits of the section's `info` field.
pub fn cubin_function_registers(cubin: &[u8]) -> Vec<FunctionRegisters> {
    let Ok(elf) = vf_sass::elf::Elf::parse(cubin) else {
        return Vec::new();
    };
    elf.sections
        .iter()
        .filter_map(|s| {
            s.name.strip_prefix(".text.").map(|name| FunctionRegisters {
                name: name.to_string(),
                registers: s.info >> 24,
            })
        })
        .collect()
}

/// Whether the translated cubin has any function whose register count exceeds its budget.  The
/// budget is `min(max(ceil8(declared), 32), 255)`.  `original` is the cubin before translation;
/// `translated` is the cubin after.  Returns `true` when a second translation within the budget
/// could lower the count of at least one function.
pub fn translation_exceeds_budget(original: &[u8], translated: &[u8]) -> bool {
    let orig = cubin_function_registers(original);
    let trans = cubin_function_registers(translated);
    for t in &trans {
        // Find the function's declared count in the original.
        let declared = orig
            .iter()
            .find(|o| o.name == t.name)
            .map_or(t.registers, |o| o.registers);
        let budget = (declared.div_ceil(8) * 8).clamp(32, 255);
        if t.registers > budget {
            return true;
        }
    }
    false
}

impl Policy {
    /// For a second translation at launch: what to keep of `original` (a cubin or a fat binary
    /// whose translation by this policy was `Outcome::Translated` with `over_budget`): the
    /// cubin, or a fat binary of the one entry that this policy translates, as it is stored
    /// (compressed: PyTorch's are a tenth of their size, and the whole fat binary holds code
    /// for six architectures). Nothing is decompressed here. `kept_cubin` gives the cubin back.
    pub fn kept_for_second(&self, original: &[u8]) -> Option<Vec<u8>> {
        if fatbin::is_elf(original) {
            return Some(original.to_vec());
        }
        let fat = fatbin::parse(original).ok()?;
        let source = self.machine_source(&fat)?;
        let mut entries = fat.entries.iter().filter(|e| e.kind() == KIND_ELF && e.sm() == source);
        let only = entries.next()?;
        // One module handle cannot identify which of several cubins owns a kernel.
        entries
            .next()
            .is_none()
            .then(|| fatbin::build(&[NewEntry::copy_of(only)]))
    }
}

/// The cubin in what `Policy::kept_for_second` gave to keep.
pub fn kept_cubin(kept: &[u8]) -> Option<std::borrow::Cow<'_, [u8]>> {
    if fatbin::is_elf(kept) {
        return Some(std::borrow::Cow::Borrowed(kept));
    }
    let fat = fatbin::parse(kept).ok()?;
    let cubin = fat.entries.first()?.payload().ok()?;
    Some(std::borrow::Cow::Owned(cubin.into_owned()))
}

/// Whether a second translation of a module must NOT be used for this kernel, because the kernel
/// can reach data of the module that code may write (`.nv.global`, `.nv.global.init`). A second
/// translation is loaded as another module: it has a copy of that data, which nothing keeps in step
/// with the first module's. True when, for the kernel or for a function of the cubin that its
/// relocations lead to:
///
/// - a relocation of the function's own sections (its code, its own constant banks) names such
///   data, or anything that is neither a constant section nor a function of the cubin;
/// - the function reads a constant bank that the module's kernels share (`.nv.constant4` holds the
///   addresses of the module's data) in which a relocation names such data. Tables that kernels
///   only read are there too (the compiler puts the tables of its mathematical functions in
///   `.nv.global.init`): they cannot be told from data that a kernel writes, and exclude the
///   kernel all the same;
/// - an instruction of the function cannot be decoded, or the cubin cannot be read.
///
/// On Turing, false for a name that the cubin has no function of. Other architectures require
/// a separate analysis of their constant-bank operands and stay in the original module.
pub fn kernel_names_global_data(cubin: &[u8], kernel_name: &str) -> bool {
    // A second translation at launch is of Turing code only. Other kernels stay in their
    // original module, where writable globals cannot diverge between two translations.
    if fatbin::cubin_sm(cubin) != Some(75) {
        return true;
    }
    names_global_data(cubin, kernel_name, 75)
}

/// What a kernel of an image's machine code can reach of the data of its image (`told_kernel_reach`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reach {
    /// Nothing: every word of it was read, and names none.
    Nothing,
    /// Data of the image, or what may be: what names it.
    Data(String),
    /// Not known: an instruction of it was not read (why). Its code is not translated either.
    Unknown(String),
}

/// What a kernel of an image's machine code for the capability told, which would be given from a
/// module of its own (`told_cubin`), can reach of the image's data: the question of
/// `kernel_names_global_data`, of Turing code or of Ampere code, and stricter. That module has
/// its own copy of everything the image holds, which nothing keeps in step with the copy that the
/// program reads and writes through the module it holds, by name (`cuModuleGetGlobal`,
/// `cudaMemcpyToSymbol`). So, beside the relocations and the banks that relocations taint:
///
/// - **every constant bank that the kernel reads must be the kernel's alone.** Bank 0 is the
///   kernel's own (`.nv.constant0.<kernel>`), and the driver fills it, but not all of it for the
///   launch: beside the parameters and the launch's constants it holds words of the module (at
///   0x50, the address of the module's `__constant__` bank: a kernel that stores that address, or
///   reads through it, has the layer's copy). So only the words whose meaning is known count as
///   the kernel's (`OWN_WORDS`), read at a fixed offset that is not negative: the parameters (from
///   0x160 to the end of the bank in the cubin), the sizes of
///   the block and of the grid (0x0 to 0x14), the stack pointer (0x28) and the descriptor of
///   global memory (0x118, the same in every module). Any other word of bank 0 is data. Another
///   bank is the kernel's when the cubin has a section of it named for the
///   kernel (`.nv.constant<N>.<kernel>`: the constants that the compiler made for that kernel),
///   none shared by the kernels (`.nv.constant<N>`), and no symbol of the cubin is in the
///   kernel's section: the program names data by a symbol, so nothing of the host can reach bytes
///   that no symbol names, and they are the image's own in every copy. A `__constant__`
///   variable is in a bank of the module (bank 3, `.nv.constant3`), read as `c[0x3][...]` with no
///   relocation: a kernel that reads it is not given.
/// - an Ampere instruction is read as the Turing instruction that it is by the two descriptions
///   (`vf_sass::translate::ampere_word`: taken by the translator or only described); the forms of
///   Ampere's own that the translator rewrites read no constant bank; of any other instruction
///   nothing is known (`Reach::Unknown`), unless something else of the kernel names data.
///
/// - a relocation in one of the kernel's own banks may name a function of the cubin and nothing
///   else, and an instruction on a texture or a surface (bound to the module by name, with no bank
///   in its operands) is data.
///
/// `Unknown` for code of any other architecture.
pub fn told_kernel_reach(cubin: &[u8], kernel_name: &str) -> Reach {
    match fatbin::cubin_sm(cubin) {
        Some(sm @ (75 | 80)) => reach(cubin, kernel_name, sm, true),
        sm => Reach::Unknown(format!("machine code for sm_{}", sm.unwrap_or(0))),
    }
}

/// Whether `told_kernel_reach` is anything but nothing.
pub fn told_kernel_names_global_data(cubin: &[u8], kernel_name: &str) -> bool {
    told_kernel_reach(cubin, kernel_name) != Reach::Nothing
}

/// The cubin of `fat` for `sm` that has kernel `name` (`has_kernel`), decompressed: the machine
/// code that a GPU of the capability told loads for it (`Policy::told_machine_code`).
pub fn told_cubin(fat: &FatBinary<'_>, sm: u32, name: &str) -> Option<Vec<u8>> {
    fat.entries
        .iter()
        .filter(|e| e.kind() == KIND_ELF && e.sm() == sm)
        .filter_map(|entry| entry.payload().ok())
        .find(|cubin| fatbin::cubin_entries(cubin).is_some_and(|names| names.contains(&name)))
        .map(|cubin| cubin.into_owned())
}

fn names_global_data(cubin: &[u8], kernel_name: &str, sm: u32) -> bool {
    reach(cubin, kernel_name, sm, false) != Reach::Nothing
}

/// The words of constant bank 0 that are a kernel's own (`told_kernel_reach`), as ranges of
/// bytes: the sizes of the block and of the grid, the stack pointer, the descriptor of global
/// memory, and the parameters. Over the sm_80 code of vLLM 0.34.0, PyTorch 2.14.0 and its linalg
/// library (609 cubins, as NVIDIA's disassembler prints them): 5,218,571 reads at 0x160 or
/// above, 104,254 at 0x0 to 0x14, 28,727 at 0x28, 30,514 at 0x118, and no other fixed offset but
/// 0x18 (89), 0x20 and 0x24 (262 each), 0x40 and 0x44 (309) and 60 that are negative, whose
/// meaning is not known here: not the kernel's. 0x50 is the address of the module's `__constant__`
/// bank (`tests/kernels/k_told.cu`, `k_const_address`).
/// The parameters end where the kernel's bank 0 does in the cubin (`.nv.constant0.<kernel>`, whose
/// size is 0x160 and the parameters): `own_word` bounds the last range by that. An offset is a
/// signed number of 16 bits once in bytes (the field of an ordinary operand is 14 bits of words,
/// that of the load of a uniform register 16 bits of bytes; whether NVIDIA reads the second with a
/// sign is not known here, and it is taken so): a negative one reads before the bank, and is not
/// the kernel's.
const OWN_WORDS: [(u32, u32); 4] = [(0x0, 0x18), (0x28, 0x2c), (0x118, 0x120), (PARAMETERS, u32::MAX)];
const PARAMETERS: u32 = 0x160;

/// Whether the `width` bytes of bank 0 at `at` (the offset as its field has it, in bytes) are the
/// kernel's own, in a bank of `size` bytes.
fn own_word(at: u32, width: u32, size: u32) -> bool {
    let negative = at >= 0x8000;
    let end = at.saturating_add(width);
    !negative
        && at.is_multiple_of(4)
        && OWN_WORDS
            .iter()
            .any(|&(from, to)| from <= at && end <= if from == PARAMETERS { to.min(size) } else { to })
}

/// The mnemonics of the instructions on textures and surfaces.
fn texture(mnemonic: &str) -> bool {
    [
        "TEX",
        "TLD",
        "TXQ",
        "TXD",
        "TMML",
        "SULD",
        "SUST",
        "SUATOM",
        "SURED",
        "SUQ",
        "FOOTPRINT",
    ]
    .iter()
    .any(|prefix| mnemonic.starts_with(prefix))
}

/// `kernel_names_global_data` and `told_kernel_reach`: with `own_banks`, every constant bank read
/// must be the kernel's alone.
fn reach(cubin: &[u8], kernel_name: &str, sm: u32, own_banks: bool) -> Reach {
    match vf_sass::elf::Elf::parse(cubin) {
        Ok(elf) => reach_in(&elf, kernel_name, sm, own_banks),
        Err(_) => Reach::Unknown("the cubin is not read".into()),
    }
}

fn reach_in(elf: &vf_sass::elf::Elf, kernel_name: &str, sm: u32, own_banks: bool) -> Reach {
    let Ok(symbols) = elf.symbols() else {
        return Reach::Unknown("the symbols of the cubin are not read".into());
    };
    let is_data = |section: usize| {
        elf.sections
            .get(section)
            .is_some_and(|s| s.name == ".nv.global" || s.name == ".nv.global.init")
    };
    // The banks shared by the module's kernels (`.nv.constant<N>`, no function's name after the
    // number) in which a relocation names the module's data.
    let mut tainted = Vec::new();
    for section in &elf.sections {
        let Some(rest) = relocation_target(section) else {
            continue;
        };
        let Some(bank) = rest.strip_prefix(".nv.constant").and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        if relocated_symbols(section).any(|symbol| symbols.get(symbol).is_none_or(|s| is_data(s.shndx as usize))) {
            tainted.push(bank);
        }
    }
    // Whether bank `bank` is the kernel's alone (`told_kernel_reach`).
    let own = |bank: u32| {
        if bank == 0 {
            return true;
        }
        let Some(section) = elf.section(&format!(".nv.constant{bank}.{kernel_name}")) else {
            return false;
        };
        elf.section(&format!(".nv.constant{bank}")).is_none() && !symbols.iter().any(|s| s.shndx as usize == section)
    };
    // Where the kernel's bank 0 ends: its parameters do not reach further.
    let bank_0_size = elf
        .section(&format!(".nv.constant0.{kernel_name}"))
        .map_or(0, |section| u32::try_from(elf.sections[section].size).unwrap_or(0));
    let mut unknown = None;
    let mut seen = vec![kernel_name.to_string()];
    let mut work = vec![kernel_name.to_string()];
    while let Some(function) = work.pop() {
        let Some(text) = elf.section(&format!(".text.{function}")) else {
            continue;
        };
        // What the relocations of the function's own sections name.
        for section in &elf.sections {
            let Some(rest) = relocation_target(section) else {
                continue;
            };
            let own_bank = rest
                .strip_prefix(".nv.constant")
                .and_then(|n| n.split_once('.'))
                .is_some_and(|(number, of)| number.parse::<u32>().is_ok() && of == function);
            if rest.strip_prefix(".text.") != Some(function.as_str()) && !own_bank {
                continue;
            }
            for symbol in relocated_symbols(section) {
                let Some(target) = symbols.get(symbol).and_then(|s| elf.sections.get(s.shndx as usize)) else {
                    return Reach::Data(format!("a relocation of {function} that names nothing of the cubin"));
                };
                if let Some(other) = target.name.strip_prefix(".text.") {
                    if !seen.iter().any(|name| name == other) {
                        seen.push(other.to_string());
                        work.push(other.to_string());
                    }
                } else if own_banks || !target.name.starts_with(".nv.constant") {
                    // (For a kernel given alone, also a place in a constant bank: its address
                    // would be the copy's.)
                    return Reach::Data(format!("a relocation of {function} to {}", target.name));
                }
            }
        }
        // The banks its instructions read.
        let code = &elf.sections[text].data;
        if !code.len().is_multiple_of(16) {
            return Reach::Unknown(format!("the code of {function} is not whole instructions"));
        }
        let mut decoder = vf_sass::decode::Decoder::new(75);
        for (i, word) in code.as_chunks::<16>().0.iter().enumerate() {
            let word = u128::from_le_bytes(*word);
            if sm == 80 {
                match vf_sass::translate::ampere_word(word) {
                    vf_sass::translate::AmpereWord::Same | vf_sass::translate::AmpereWord::Described => {}
                    vf_sass::translate::AmpereWord::Rewritten => continue,
                    vf_sass::translate::AmpereWord::Unsupported => {
                        unknown.get_or_insert_with(|| {
                            format!("the instruction of {function} at 0x{:x} is not read", 16 * i)
                        });
                        continue;
                    }
                }
            }
            let Some(d) = decoder.decode(word) else {
                unknown.get_or_insert_with(|| format!("the instruction of {function} at 0x{:x} is not read", 16 * i));
                continue;
            };
            if own_banks && texture(d.mnemonic) {
                return Reach::Data(format!(
                    "a texture or a surface of the image ({} at 0x{:x})",
                    d.mnemonic,
                    16 * i
                ));
            }
            for field in ["Sa_bank", "Sb_bank", "Sc_bank", "bank5"] {
                if !d.has_op_name(field) {
                    continue;
                }
                let bank = d.op_or_name(field, 0);
                if own_banks && bank == 0 {
                    // The word of bank 0 that is read, at a fixed offset: in words of four bytes,
                    // but in bytes in the load of a uniform register; the other loads from a
                    // bank (LDC, ULDC at an index) are at an offset that a register gives. Eight
                    // bytes for an operand of two registers.
                    let load = d.name.starts_with("ldc") || d.name.starts_with("uldc");
                    let fixed = !load || d.name == "uldc_const__RCR";
                    let unit = if load { 1 } else { 4 };
                    let offset = match field.strip_suffix("_bank").map(|operand| format!("{operand}_addr")) {
                        Some(address) if fixed && d.has_op_name(&address) => Some(unit * d.op_or_name(&address, 0)),
                        _ => None,
                    };
                    let wide = d.mnemonic.starts_with('D')
                        || d.name.contains("_wide")
                        || (d.mnemonic == "ULDC" && d.op_or_name("sz", 0) == 5);
                    let own = offset.is_some_and(|at| own_word(at, if wide { 8 } else { 4 }, bank_0_size));
                    if !own {
                        return Reach::Data(match offset {
                            Some(at) => format!(
                                "the word of constant bank 0 at 0x{at:x}, which is not known to be the kernel's \
                                 own (read at 0x{:x}; 0x50 is the address of the image's constants)",
                                16 * i
                            ),
                            None => format!("constant bank 0 at an offset that is computed (read at 0x{:x})", 16 * i),
                        });
                    }
                }
                if tainted.contains(&bank) {
                    return Reach::Data(format!(
                        "constant bank {bank}, which holds addresses of the image's variables (read at 0x{:x})",
                        16 * i
                    ));
                }
                if own_banks && bank != 0 && !(function == kernel_name && own(bank)) {
                    return Reach::Data(format!(
                        "constant bank {bank}, which is not this kernel's alone (read at 0x{:x})",
                        16 * i
                    ));
                }
            }
        }
    }
    unknown.map_or(Reach::Nothing, Reach::Unknown)
}

/// For a section of relocations (`SHT_REL`, `SHT_RELA`): the name of the section it relocates.
fn relocation_target(section: &vf_sass::elf::Section) -> Option<&str> {
    match section.stype {
        9 => section.name.strip_prefix(".rel"),
        4 => section.name.strip_prefix(".rela"),
        _ => None,
    }
}

/// The symbols that the entries of a section of relocations name, by their index.
fn relocated_symbols(section: &vf_sass::elf::Section) -> impl Iterator<Item = usize> + '_ {
    let entry = if section.stype == 9 { 16 } else { 24 };
    section
        .data
        .chunks_exact(entry)
        .map(|e| (u64::from_le_bytes(e[8..16].try_into().unwrap_or([0; 8])) >> 32) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELF75_PTX75: &[u8] = include_bytes!("../../vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin");
    const ELF70_ELF75_PTX70: &[u8] = include_bytes!("../../vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin");

    #[test]
    fn fat_binary_with_volta_code_is_left_alone() {
        let adapted = Policy::default()
            .adapt_fatbin(&fatbin::parse(ELF70_ELF75_PTX70).unwrap())
            .unwrap();
        assert_eq!(adapted.outcome, Outcome::Native);
        assert!(adapted.image.is_none());
    }

    #[test]
    fn turing_ptx_entry_becomes_volta_ptx() {
        let adapted = Policy::default()
            .adapt_fatbin(&fatbin::parse(ELF75_PTX75).unwrap())
            .unwrap();
        assert_eq!(
            adapted.outcome,
            Outcome::Retargeted {
                entries: 1,
                from_sm: 75,
                tensor_rewritten: 0,
                tanh_expanded: 0,
                trapped: vec![],
                refused: vec![],
                emulated: vec![],
                replaced: vec![],
                reported: vec![],
                stubs: Stubs::default(),
            }
        );
        let image = adapted.image.unwrap();
        let fat = fatbin::parse(&image).unwrap();
        assert_eq!(fat.entries.len(), 2);
        assert_eq!((fat.entries[0].kind(), fat.entries[0].sm()), (KIND_ELF, 75));
        let ptx = &fat.entries[1];
        assert_eq!(
            (ptx.kind(), ptx.sm(), ptx.content_version(), ptx.is_compressed()),
            (KIND_PTX, 70, (9, 0), false)
        );
        let text = ptx.payload().unwrap();
        assert_eq!(
            vf_ptx::header(&text),
            Some(vf_ptx::Header {
                version: (9, 0),
                target: 70
            })
        );
        // Adapting the result again finds it native.
        assert_eq!(Policy::default().adapt_fatbin(&fat).unwrap().outcome, Outcome::Native);
    }

    /// What the translator makes of the seven functions of the fixture's machine code: all are
    /// taken; one has an instruction more (the address of an atomic reduction, which is a
    /// parameter), one three (three accesses to shared memory at an address with a stride).
    fn translated_fixture(entries: usize) -> Outcome {
        let added = |kernel: &str, machine_code| Emulated {
            kernel: kernel.to_string(),
            counts: EmulatedCounts {
                machine_code,
                ..EmulatedCounts::default()
            },
        };
        Outcome::Translated {
            entries,
            from_sm: 75,
            functions: 7,
            refused: vec![],
            emulated: vec![added("warp_sum", 1), added("block_sum", 3)],
            over_budget: false,
            stubs: Stubs::default(),
            paired: vec![],
        }
    }

    #[test]
    fn machine_code_alone_is_translated() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let only_elf = fatbin::build(&[NewEntry::copy_of(&fat.entries[0])]);
        let adapted = Policy::default()
            .adapt_fatbin(&fatbin::parse(&only_elf).unwrap())
            .unwrap();
        assert_eq!(adapted.outcome, translated_fixture(1));
        let image = adapted.image.unwrap();
        let fat = fatbin::parse(&image).unwrap();
        assert_eq!(fat.entries.len(), 1);
        let entry = &fat.entries[0];
        assert_eq!((entry.kind(), entry.sm(), entry.is_compressed()), (KIND_ELF, 70, false));
        assert_eq!(fatbin::cubin_sm(&entry.payload().unwrap()), Some(70));
        // What comes out is for the device: nothing more to do to it.
        assert_eq!(Policy::default().adapt_fatbin(&fat).unwrap().outcome, Outcome::Native);
    }

    #[test]
    fn machine_code_before_ptx_when_asked() {
        // By default the PTX of this fat binary is rewritten (`turing_ptx_entry_becomes_volta_ptx`).
        let policy = Policy {
            translate: Translate::First,
            ..Policy::default()
        };
        let adapted = policy.adapt_fatbin(&fatbin::parse(ELF75_PTX75).unwrap()).unwrap();
        assert_eq!(adapted.outcome, translated_fixture(1));
        let image = adapted.image.unwrap();
        let fat = fatbin::parse(&image).unwrap();
        let kinds: Vec<(u16, u32)> = fat.entries.iter().map(|e| (e.kind(), e.sm())).collect();
        assert_eq!(kinds, [(KIND_ELF, 70), (KIND_PTX, 75)]);
    }

    #[test]
    fn a_cubin_is_translated() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let cubin = fat.entries[0].payload().unwrap();
        let policy = Policy::default();
        assert!(policy.translates_cubin(&cubin[..0x40]));
        let adapted = policy.adapt_cubin(&cubin);
        assert_eq!(adapted.outcome, translated_fixture(1));
        let image = adapted.image.unwrap();
        assert_eq!(fatbin::cubin_sm(&image), Some(70));
        assert_eq!(policy.adapt_cubin(&image).outcome, Outcome::Native);
        assert!(!policy.translates_cubin(&image[..0x40]));
        // Its header alone is not enough: said, and nothing loaded in its place.
        let cut = policy.adapt_cubin(&cubin[..0x40]);
        assert!(matches!(cut.outcome, Outcome::NotTranslated { from_sm: 75, .. }));
        assert!(cut.image.is_none());
    }

    #[test]
    fn machine_code_is_left_alone_when_told_so() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let only_elf = fatbin::build(&[NewEntry::copy_of(&fat.entries[0])]);
        let policy = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        let adapted = policy.adapt_fatbin(&fatbin::parse(&only_elf).unwrap()).unwrap();
        assert_eq!(adapted.outcome, Outcome::NeedsTranslation { lowest_sm: 75 });
        assert!(adapted.image.is_none());
        let cubin = fat.entries[0].payload().unwrap();
        assert_eq!(
            policy.adapt_cubin(&cubin).outcome,
            Outcome::NeedsTranslation { lowest_sm: 75 }
        );
    }

    #[test]
    fn ptx_for_a_later_generation_is_retargeted() {
        let policy = Policy::default();
        // Since the instruction-level check is the real guard, PTX from any later architecture
        // is accepted for retargeting. If it contained sm_90-only instructions, lower_to_volta
        // would replace or trap them.
        let adapted = policy.adapt_ptx(b".version 9.5\n.target sm_90\n");
        assert_eq!(
            adapted.outcome,
            Outcome::Retargeted {
                entries: 1,
                from_sm: 90,
                tensor_rewritten: 0,
                tanh_expanded: 0,
                trapped: vec![],
                refused: vec![],
                emulated: vec![],
                replaced: vec![],
                reported: vec![],
                stubs: Stubs::default(),
            }
        );
        assert_eq!(adapted.image.unwrap(), b".version 9.0\n.target sm_70\n");
    }

    #[test]
    fn cluster_directives_without_later_instructions_refuse_only_their_entry() {
        for directives in [
            ".explicitcluster\n.reqnctapercluster 2, 1, 1\n",
            ".explicitcluster .maxntid 128 .maxclusterrank 4 ",
            ".reqnctapercluster 2, /* rows */ 1, 1 ",
        ] {
            let ptx = format!(
                ".version 8.8\n.target sm_90\n.address_size 64\n\
                 .visible .entry clustered() {directives}{{ ret; }}\n\
                 .visible .entry plain() /* .explicitcluster */ {{ .reg .u32 %vf_tmp; /* .explicitcluster */ ret; }}\n"
            );
            let adapted = Policy::default().adapt_ptx(ptx.as_bytes());
            let Outcome::Retargeted { refused, .. } = adapted.outcome else {
                panic!("expected adapted PTX");
            };
            assert_eq!(refused.len(), 1);
            assert_eq!(refused[0].kernel, "clustered");
            let output = String::from_utf8(adapted.image.unwrap()).unwrap();
            let (cluster, plain) = output.split_once(".entry plain()").unwrap();
            assert!(cluster.contains("trap;"));
            assert!(!cluster.contains(".explicitcluster"));
            assert!(!cluster.contains(".reqnctapercluster"));
            assert!(!cluster.contains(".maxclusterrank"));
            if directives.contains(".maxntid") {
                assert!(cluster.contains(".maxntid 128"));
            }
            assert!(plain.contains(".reg .u32 %vf_tmp; /* .explicitcluster */ ret;"));
        }
    }

    #[test]
    fn ptx_text() {
        let policy = Policy::default();
        let adapted = policy.adapt_ptx(b".version 9.5\n.target sm_75\n.address_size 64\n");
        assert_eq!(
            adapted.outcome,
            Outcome::Retargeted {
                entries: 1,
                from_sm: 75,
                tensor_rewritten: 0,
                tanh_expanded: 0,
                trapped: vec![],
                refused: vec![],
                emulated: vec![],
                replaced: vec![],
                reported: vec![],
                stubs: Stubs::default(),
            }
        );
        assert_eq!(
            adapted.image.unwrap(),
            b".version 9.0\n.target sm_70\n.address_size 64\n"
        );
        assert_eq!(
            policy.adapt_ptx(b".version 8.8\n.target sm_70\n").outcome,
            Outcome::Native
        );
        // An sm_70 target in a dialect the JIT does not know yet: only the version changes.
        let adapted = policy.adapt_ptx(b".version 9.5\n.target sm_70\n");
        assert_eq!(adapted.image.unwrap(), b".version 9.0\n.target sm_70\n");
        assert_eq!(policy.adapt_ptx(b"hello").outcome, Outcome::Unknown);
    }

    #[test]
    fn multimem_instructions_are_reported_traps_by_default_and_not_kernels_without_code() {
        // A kernel whose one instruction is multimem.st, as NCCL's kernels that store to multicast
        // memory, and one that holds it behind a test.
        let text = b".version 9.3\n.target sm_90\n.address_size 64\n\
                     .visible .entry only(.param .u64 a)\n{\n\t.reg .b64 %rd<2>;\n\tld.param.u64 %rd1, [a];\n\
                     \tmultimem.st.global.b32 [%rd1], 1;\n\tret;\n}\n\
                     .visible .entry tested(.param .u64 a, .param .u32 f)\n{\n\t.reg .b64 %rd<2>;\n\t.reg .b32 %r<2>;\n\
                     \t.reg .pred %p<2>;\n\tld.param.u64 %rd1, [a];\n\tld.param.u32 %r1, [f];\n\tsetp.ne.u32 %p1, %r1, 0;\n\
                     \t@%p1 multimem.red.relaxed.sys.global.add.u32 [%rd1], 1;\n\tret;\n}\n";
        let adapted = Policy::default().adapt_ptx(text);
        let Outcome::Retargeted {
            trapped,
            refused,
            replaced,
            reported,
            stubs,
            ..
        } = &adapted.outcome
        else {
            panic!("{:?}", adapted.outcome);
        };
        assert!(trapped.is_empty() && refused.is_empty());
        let names: Vec<&str> = reported.iter().map(|kernel| kernel.kernel.as_str()).collect();
        assert_eq!(names, ["only", "tested"]);
        assert_eq!(replaced.len(), 2);
        // `only` is all a trap once its instruction is replaced; it is not a kernel without code: that
        // is read in the program's text, where it stores.
        assert!(stubs.is_empty(), "{stubs:?}");
        let image = String::from_utf8(adapted.image.unwrap()).unwrap();
        assert!(image.starts_with(".version 9.0\n.target sm_70\n.address_size 64\n.extern .func __assertfail"));
        // The offline tools of CUDA 12.9 make code that the layer does not announce: refused there.
        let Outcome::Retargeted { refused, reported, .. } = Policy::cuda_12_9().adapt_ptx(text).outcome else {
            panic!();
        };
        assert_eq!((refused.len(), reported.len()), (2, 0));
        let off = Policy {
            lowering: vf_ptx::lower::Options {
                reported_traps: false,
                ..Policy::default().lowering
            },
            ..Policy::default()
        };
        let Outcome::Retargeted { refused, reported, .. } = off.adapt_ptx(text).outcome else {
            panic!();
        };
        assert_eq!((refused.len(), reported.len()), (2, 0));
    }

    #[test]
    fn ptx_with_turing_instructions_is_lowered() {
        let text = b".version 9.5\n.target sm_75\n.entry a()\n{\n\ttanh.approx.f32 %f1, %f2;\n}\n\
                     .entry b()\n{\n\tmovmatrix.sync.aligned.m8n8.trans.b16 %r1, %r2;\n}\n\
                     .entry c()\n{\n\tmma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%r1, %r2}, {%r3}, {%r4}, {%r1, %r2};\n}\n";
        let adapted = Policy::default().adapt_ptx(text);
        let Outcome::Retargeted {
            entries: 1,
            from_sm: 75,
            tensor_rewritten: 1,
            tanh_expanded: 1,
            trapped,
            refused,
            emulated,
            stubs,
            ..
        } = &adapted.outcome
        else {
            panic!("{:?}", adapted.outcome);
        };
        // The trap that the layer put in c is not a kernel without code: c is refused, and said so.
        assert!(stubs.is_empty(), "{stubs:?}");
        // a has a tanh and b a movmatrix, both emulated; c is trapped, and refused as a kernel.
        let names: Vec<&str> = emulated.iter().map(|kernel| kernel.kernel.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(trapped.len(), 1);
        assert_eq!(trapped[0].function, "c");
        assert_eq!(refused.len(), 1);
        assert_eq!((refused[0].kernel.as_str(), refused[0].function.as_str()), ("c", "c"));
        let image = adapted.image.unwrap();
        assert!(vf_ptx::turing_only_features(&image).is_empty());
        assert_eq!(
            vf_ptx::header(&image),
            Some(vf_ptx::Header {
                version: (9, 0),
                target: 70
            })
        );
    }

    /// Two kernels, of which the second holds something the compiler refuses at line 9.
    const TWO_KERNELS: &str = ".version 9.5\n.target sm_75\n.address_size 64\n.entry good()\n{\n\tret;\n}\n.entry bad()\n{\n\tdiscard.global.L2 [0], 128;\n\tret;\n}\n";
    const LINE_10: &str =
        "ptxas application ptx input, line 10; error   : Feature 'discard' requires .target sm_80 or higher\n";

    #[test]
    fn text_without_what_the_compiler_refused() {
        let policy = Policy::default();
        for end in [&b""[..], b"\0"] {
            let mut image = policy.adapt_ptx(TWO_KERNELS.as_bytes()).image.unwrap();
            image.extend_from_slice(end);
            let retry = policy.without_refused(&image, LINE_10).unwrap();
            let text = String::from_utf8(retry.image).unwrap();
            assert!(text.contains(".entry good()\n{\n\tret;\n}\n.entry bad()\n{\n\ttrap;\n\tret;\n}\n"));
            assert_eq!(text.ends_with('\0'), !end.is_empty());
            assert_eq!(retry.functions.len(), 1);
            assert_eq!(retry.kernels.len(), 1);
            assert_eq!(retry.kernels[0].kernel, "bad");
            assert_eq!(
                retry.kernels[0].reason,
                "Feature 'discard' requires .target sm_80 or higher"
            );
            // Nothing the log names: nothing to try.
            assert!(
                policy
                    .without_refused(&image, "ptxas fatal   : Ptx assembly aborted due to errors\n")
                    .is_none()
            );
        }
        assert!(policy.without_refused(b"\x7fELF", LINE_10).is_none());
    }

    #[test]
    fn the_code_taken_is_the_code_for_what_the_program_is_told() {
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let ptx = |sm: u32| {
            let text = format!(".version 9.5\n.target sm_{sm}\n.address_size 64\n");
            NewEntry::replace(&fat.entries[1], text.into_bytes(), sm, (9, 5))
        };
        let program = fatbin::build(&[NewEntry::copy_of(&fat.entries[0]), ptx(90), ptx(75), ptx(80)]);
        let program = fatbin::parse(&program).unwrap();
        let taken = |told_sm: Option<u32>| {
            let policy = Policy {
                told_sm,
                ..Policy::default()
            };
            let adapted = policy.adapt_fatbin(&program).unwrap();
            let Outcome::Retargeted {
                entries: 1, from_sm, ..
            } = adapted.outcome
            else {
                panic!("{:?}", adapted.outcome);
            };
            // One entry became PTX for the device; the others are as they were.
            let image = adapted.image.unwrap();
            let adapted = fatbin::parse(&image).unwrap();
            let mut targets: Vec<u32> = adapted
                .entries
                .iter()
                .filter(|e| e.kind() == KIND_PTX)
                .map(|e| e.sm())
                .collect();
            targets.sort();
            let mut expected: Vec<u32> = [75, 80, 90]
                .iter()
                .map(|&sm| if sm == from_sm { 70 } else { sm })
                .collect();
            expected.sort();
            assert_eq!(targets, expected);
            from_sm
        };
        // Told nothing chosen by the user: the lowest architecture the program has code for.
        assert_eq!(taken(None), 75);
        // Told a capability: what the driver of such a GPU takes, the highest target it reaches.
        assert_eq!(taken(Some(75)), 75);
        assert_eq!(taken(Some(80)), 80);
        assert_eq!(taken(Some(86)), 80);
        assert_eq!(taken(Some(90)), 90);
        assert_eq!(taken(Some(121)), 90);
        // Nothing reaches it: the nearest above.
        assert_eq!(taken(Some(72)), 75);
        assert_eq!(capability_sm("8.6"), Some(86));
        assert_eq!(capability_sm("12.1"), Some(121));
        assert_eq!(capability_sm("8"), None);
        assert_eq!(capability_sm("8.10"), None);
    }

    #[test]
    fn a_capability_that_is_not_one_or_is_below_the_gpu_is_said_and_not_used() {
        for (value, sm) in [
            ("8.0", 80),
            (" 8.0", 80),
            ("8.0 ", 80),
            ("8.6", 86),
            ("7.0", 70),
            ("12.0", 120),
        ] {
            assert_eq!(capability_from(value), Ok(Some(sm)), "{value}");
        }
        assert_eq!(capability_from(""), Ok(None));
        assert_eq!(capability_from(" "), Ok(None));
        for value in ["8", "80", "sm_80", "8,0", "8.10", "abc", "4294967295.0"] {
            let said = capability_from(value).unwrap_err();
            assert!(
                said.contains(&format!("VF_CAPABILITY={value} is not a compute capability")),
                "{said}"
            );
        }
        for value in ["6.1", "0.0"] {
            let said = capability_from(value).unwrap_err();
            assert!(said.contains("below the 7.0 of the GPU"), "{said}");
        }
    }

    #[test]
    fn fat_binary_without_what_the_compiler_refused() {
        let policy = Policy::default();
        // The fat binary of the other tests, its PTX replaced by the two kernels, then adapted.
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let program = fatbin::build(&[
            NewEntry::copy_of(&fat.entries[0]),
            NewEntry::replace(&fat.entries[1], TWO_KERNELS.as_bytes().to_vec(), 75, (9, 5)),
        ]);
        let adapted = policy
            .adapt_fatbin(&fatbin::parse(&program).unwrap())
            .unwrap()
            .image
            .unwrap();
        let retry = policy.without_refused(&adapted, LINE_10).unwrap();
        assert_eq!(retry.kernels.len(), 1);
        let again = fatbin::parse(&retry.image).unwrap();
        assert_eq!(again.entries.len(), 2);
        assert_eq!(again.entries[0].stored, fat.entries[0].stored);
        let ptx = &again.entries[1];
        assert_eq!((ptx.kind(), ptx.sm(), ptx.content_version()), (KIND_PTX, 70, (9, 0)));
        let payload = ptx.payload().unwrap();
        let text = std::str::from_utf8(&payload).unwrap().trim_end_matches('\0');
        assert!(text.starts_with(".version 9.0\n.target sm_70\n"));
        assert!(text.ends_with(".entry bad()\n{\n\ttrap;\n\tret;\n}\n"));
        // It is still what the driver would compile.
        assert_eq!(policy.adapt_fatbin(&again).unwrap().outcome, Outcome::Native);
    }

    #[test]
    fn code_embedded_in_a_file() {
        let policy = Policy::default();
        assert_eq!(policy.embedded(b"\x7fELF no fat binary here"), Embedded::Nothing);
        let mut file = b"\x7fELF....".to_vec();
        file.extend_from_slice(ELF75_PTX75);
        file.extend_from_slice(&[0x50, 0xed, 0x55, 0xba, 9, 9]); // the magic number, not a fat binary
        assert_eq!(policy.embedded(&file), Embedded::Other { lowest_sm: 75 });
        file.extend_from_slice(ELF70_ELF75_PTX70);
        assert_eq!(policy.embedded(&file), Embedded::Usable);
    }

    const DEBUG_CUBIN: &[u8] = include_bytes!("../../../tests/kernels/ptx-cases/copy-octal.sm80.cubin");

    #[test]
    fn embedded_ptx_uses_the_same_lowering_in_a_cubin_and_fat_binary() {
        let policy = Policy::default();
        let adapted = policy.adapt_cubin(DEBUG_CUBIN);
        assert!(matches!(&adapted.outcome, Outcome::Retargeted { from_sm: 80, refused, .. } if refused.is_empty()));
        let image = adapted.image.unwrap();
        assert_eq!(vf_ptx::header(&image).unwrap().target, 70);
        let template = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
        let bytes = fatbin::build(&[NewEntry::replace(&template, DEBUG_CUBIN.to_vec(), 80, (1, 7))]);
        let rewritten = policy.adapt_fatbin(&fatbin::parse(&bytes).unwrap()).unwrap();
        assert_eq!(rewritten.outcome, adapted.outcome);
        let bytes = rewritten.image.unwrap();
        let fat = fatbin::parse(&bytes).unwrap();
        assert_eq!(fat.entries.len(), 1);
        let entry = fat.entries[0];
        assert_eq!(
            (entry.kind(), entry.sm(), entry.content_version()),
            (KIND_PTX, 70, (7, 0))
        );
        assert_eq!(&*entry.payload().unwrap(), image.as_slice());
    }

    /// `tests/kernels/k_stub.cu`: machine code for sm_75 and sm_80, no PTX.
    const STUBS: &[u8] = include_bytes!("../../../tests/kernels/k_stub.135.sm75-sm80.fatbin");

    /// `tests/kernels/k_nocode.cu`: PTX for sm_75, machine code for sm_75 and for sm_70, and a fat
    /// binary with PTX and machine code for sm_75.
    const NOCODE_PTX: &[u8] = include_bytes!("../../../tests/kernels/k_nocode.135.sm75.ptx");
    const NOCODE_75: &[u8] = include_bytes!("../../../tests/kernels/k_nocode.135.sm75.cubin");
    const NOCODE_70: &[u8] = include_bytes!("../../../tests/kernels/k_nocode.129.sm70.cubin");
    const NOCODE_FAT: &[u8] = include_bytes!("../../../tests/kernels/k_nocode.135.sm75.fatbin");

    fn stubs_of(outcome: &Outcome) -> &Stubs {
        match outcome {
            Outcome::Retargeted { stubs, .. } | Outcome::Translated { stubs, .. } => stubs,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn kernels_without_code_are_listed_whichever_code_is_taken() {
        let expected = Stubs {
            trap_only: vec!["no_code".to_string()],
            empty: vec!["empty".to_string()],
        };
        let fat = fatbin::parse(NOCODE_FAT).unwrap();
        // PTX text, and the PTX of a fat binary (taken by default).
        let policy = Policy::default();
        assert_eq!(stubs_of(&policy.adapt_ptx(NOCODE_PTX).outcome), &expected);
        let adapted = policy.adapt_fatbin(&fat).unwrap();
        assert!(matches!(adapted.outcome, Outcome::Retargeted { .. }));
        assert_eq!(stubs_of(&adapted.outcome), &expected);
        // Machine code for sm_75, translated: a cubin, and the cubin of the fat binary
        // (VF_TRANSLATE=first).
        assert_eq!(stubs_of(&policy.adapt_cubin(NOCODE_75).outcome), &expected);
        let first = Policy {
            translate: Translate::First,
            ..Policy::default()
        };
        let adapted = first.adapt_fatbin(&fat).unwrap();
        assert!(matches!(adapted.outcome, Outcome::Translated { .. }));
        assert_eq!(stubs_of(&adapted.outcome), &expected);
        // Code for sm_70 is loaded as it is: the layer says nothing of it.
        assert_eq!(policy.adapt_cubin(NOCODE_70).outcome, Outcome::Native);
    }

    #[test]
    fn a_stub_for_the_architecture_told_is_refused() {
        let fat = fatbin::parse(STUBS).unwrap();
        let refused = |told: Option<u32>| -> Vec<(String, String)> {
            let policy = Policy {
                told_sm: told,
                ..Policy::default()
            };
            match policy.adapt_fatbin(&fat).unwrap().outcome {
                Outcome::Translated {
                    from_sm: 75, refused, ..
                } => refused.into_iter().map(|r| (r.kernel, r.reason)).collect(),
                other => panic!("told {told:?}: {other:?}"),
            }
        };
        // Told nothing, or the generation of the code given: every kernel is given.
        assert_eq!(refused(None), []);
        assert_eq!(refused(Some(75)), []);
        // Told a later one: the kernel whose code for sm_75 writes nothing where its code for sm_80
        // writes. Not the one that writes in both, nor those that write nothing in both (one waits,
        // one asserts).
        for told in [80, 86, 90] {
            let found = refused(Some(told));
            assert_eq!(
                found.iter().map(|(kernel, _)| kernel.as_str()).collect::<Vec<_>>(),
                ["k_stub"],
                "told {told}"
            );
            assert!(
                found[0]
                    .1
                    .starts_with("its machine code for sm_75 writes no memory (a stub), where its code for sm_80"),
                "{}",
                found[0].1
            );
        }
        // Without the code for the later architecture, nothing tells a stub from a kernel that
        // writes nothing: given.
        let turing: Vec<NewEntry> = fat
            .entries
            .iter()
            .filter(|e| e.sm() == 75)
            .map(NewEntry::copy_of)
            .collect();
        let alone = fatbin::build(&turing);
        let policy = Policy {
            told_sm: Some(80),
            ..Policy::default()
        };
        assert!(matches!(
            policy.adapt_fatbin(&fatbin::parse(&alone).unwrap()).unwrap().outcome,
            Outcome::Translated { from_sm: 75, ref refused, .. } if refused.is_empty()
        ));
    }

    /// `tests/kernels/k_sections.cu`: machine code for sm_75 and sm_80, no PTX; `k_from_80` is in the
    /// code for sm_80 only, `k_before_80` in the code for sm_75 only, `k_both` in both.
    const SECTIONS: &[u8] = include_bytes!("../../../tests/kernels/k_sections.135.sm75-sm80.fatbin");

    /// `tests/kernels/k_told.cu`: machine code for sm_75 and sm_80, no PTX; in the code for sm_80
    /// only, `k_bf16_sum` and `k_divides`, which reach no data of the image, `k_reads_data`, which
    /// reads a variable of it, and `k_not_translated`.
    const TOLD: &[u8] = include_bytes!("../../../tests/kernels/k_told.135.sm75-sm80.fatbin");

    #[test]
    fn a_kernel_of_the_code_for_the_capability_told_is_given_alone_only_if_it_reaches_no_data() {
        let fat = fatbin::parse(TOLD).unwrap();
        let policy = Policy {
            told_sm: Some(80),
            keep_sources: true,
            ..Policy::default()
        };
        assert_eq!(policy.told_machine_code(&fat, 75), Some(80));
        assert!(policy.translates(80) && policy.translates(75) && !policy.translates(86));
        // The cubin that has the kernel, from the program's image and from the image given to
        // the driver, which keeps it.
        let cubin = told_cubin(&fat, 80, "k_bf16_sum").unwrap();
        assert_eq!(fatbin::cubin_sm(&cubin), Some(80));
        let given = policy.adapt_fatbin(&fat).unwrap().image.unwrap();
        assert_eq!(
            told_cubin(&fatbin::parse(&given).unwrap(), 80, "k_divides"),
            Some(cubin.clone())
        );
        assert_eq!(told_cubin(&fat, 80, "k_before_80"), None);
        assert_eq!(told_cubin(&fat, 86, "k_bf16_sum"), None);
        // Kernels that reach nothing of the image.
        for kernel in ["k_bf16_sum", "k_divides", "k_both"] {
            assert_eq!(told_kernel_reach(&cubin, kernel), Reach::Nothing, "{kernel}");
            // The second translation at launch is of Turing code only, as before.
            assert!(kernel_names_global_data(&cubin, kernel), "{kernel}");
        }
        // A variable that the program sets, and a table that nothing writes: through the bank
        // of the image's addresses, in instructions that the translator does not take and that
        // are read all the same. A `__constant__` variable: in a bank of the module, read with
        // no relocation, by an instruction that the translator takes (the kernel is translated).
        let data = |kernel: &str| match told_kernel_reach(&cubin, kernel) {
            Reach::Data(what) => what,
            other => panic!("{kernel}: {other:?}"),
        };
        for kernel in ["k_reads_data", "k_reads_table"] {
            assert!(
                data(kernel).starts_with("constant bank 4, which holds addresses of the image's variables"),
                "{}",
                data(kernel)
            );
        }
        assert!(
            data("k_adds_const").starts_with("constant bank 3, which is not this kernel's alone"),
            "{}",
            data("k_adds_const")
        );
        let translated = vf_sass::translate::translate(&cubin, false).unwrap();
        let refused = |name: &str| {
            translated
                .report
                .iter()
                .find(|r| r.name == name)
                .unwrap()
                .refused
                .clone()
        };
        assert_eq!(refused("k_adds_const"), None);
        // Bank 0 is not all the kernel's own: the driver puts the address of the module's
        // `__constant__` bank at 0x50. A kernel that stores that address is translated, every
        // word of it read, and has no relocation; one that reads through it has a load that is
        // not read, and is refused for the word of bank 0 all the same, whatever is read later.
        assert_eq!(refused("k_const_address"), None);
        for kernel in ["k_const_address", "k_mixed_ptr"] {
            assert!(
                data(kernel)
                    .starts_with("the word of constant bank 0 at 0x50, which is not known to be the kernel's own"),
                "{}",
                data(kernel)
            );
        }
        assert!(refused("k_reads_data").is_some() && refused("k_reads_table").is_some());
        // An instruction that is not read: nothing is known.
        assert!(matches!(
            told_kernel_reach(&cubin, "k_not_translated"),
            Reach::Unknown(_)
        ));
        assert!(told_kernel_names_global_data(&cubin, "k_not_translated"));

        // What makes a bank the kernel's alone, on copies of the cubin changed for it.
        let parsed = || vf_sass::elf::Elf::parse(&cubin).unwrap();
        let of = |elf: &vf_sass::elf::Elf, kernel: &str| reach_in(elf, kernel, 80, true);
        // Bank 3 named for the kernel, with the variable's symbol still in it: the program can
        // name those bytes. Without a symbol in it: the kernel's alone. With a bank 3 of the
        // module beside it: not its alone.
        let mut named = parsed();
        let bank = named.section(".nv.constant3").unwrap();
        named.sections[bank].name = ".nv.constant3.k_adds_const".into();
        assert!(matches!(of(&named, "k_adds_const"), Reach::Data(_)));
        let mut symbols = named.symbols().unwrap();
        for symbol in symbols.iter_mut().filter(|s| s.shndx as usize == bank) {
            symbol.shndx = 0;
        }
        named.set_symbols(&symbols);
        assert_eq!(of(&named, "k_adds_const"), Reach::Nothing);
        let mut both = parsed();
        let mut copy = named.sections[bank].clone();
        copy.name = ".nv.constant3.k_adds_const".into();
        both.sections.push(copy);
        assert!(matches!(of(&both, "k_adds_const"), Reach::Data(_)));
        // A bank named for another kernel is not this one's.
        let mut other = parsed();
        other.sections[bank].name = ".nv.constant3.k_both".into();
        assert!(matches!(of(&other, "k_adds_const"), Reach::Data(_)));
        // For the second translation at launch (Turing code, the other guard of fit.rs), a bank
        // without relocations is not looked at, as before.
        assert_eq!(reach_in(&parsed(), "k_adds_const", 80, false), Reach::Nothing);
        // A relocation of the kernel's code to a function that the cubin has not (a device
        // function of another cubin), or to the image's data.
        let relocation = |symbol: u64| {
            let mut elf = parsed();
            let mut section = elf.sections[elf.section(".rel.nv.constant4").unwrap()].clone();
            section.name = ".rel.text.k_bf16_sum".into();
            section.data = [0u64.to_le_bytes(), (symbol << 32 | 2).to_le_bytes()].concat();
            elf.sections.push(section);
            elf
        };
        let symbols = parsed().symbols().unwrap();
        let variable = symbols.iter().position(|s| s.name == "told_scale").unwrap() as u64;
        for symbol in [0, symbols.len() as u64 + 7, variable] {
            assert!(
                matches!(of(&relocation(symbol), "k_bf16_sum"), Reach::Data(_)),
                "{symbol}"
            );
        }
        let kernel = symbols.iter().position(|s| s.name == "k_both").unwrap() as u64;
        assert_eq!(of(&relocation(kernel), "k_bf16_sum"), Reach::Nothing);
        // k_mixed_ptr with the load that is not read made one that is (a word of the kernel that
        // reads no bank): every word read, no relocation, and still the word at 0x50.
        let mut readable = parsed();
        let text = readable.section(".text.k_mixed_ptr").unwrap();
        let words: Vec<u128> = readable.sections[text]
            .data
            .as_chunks::<16>()
            .0
            .iter()
            .map(|w| u128::from_le_bytes(*w))
            .collect();
        let read = |w: u128| {
            !matches!(
                vf_sass::translate::ampere_word(w),
                vf_sass::translate::AmpereWord::Unsupported
            )
        };
        let mut decoder = vf_sass::decode::Decoder::new(75);
        let plain = *words
            .iter()
            .find(|&&w| {
                matches!(vf_sass::translate::ampere_word(w), vf_sass::translate::AmpereWord::Same)
                    && decoder.decode(w).is_some_and(|d| {
                        !["Sa_bank", "Sb_bank", "Sc_bank", "bank5"]
                            .iter()
                            .any(|f| d.has_op_name(f))
                    })
            })
            .unwrap();
        let unread: Vec<usize> = (0..words.len()).filter(|&i| !read(words[i])).collect();
        assert!(!unread.is_empty());
        for i in unread {
            readable.sections[text].data[16 * i..16 * i + 16].copy_from_slice(&plain.to_le_bytes());
        }
        match of(&readable, "k_mixed_ptr") {
            Reach::Data(what) => assert!(what.starts_with("the word of constant bank 0 at 0x50"), "{what}"),
            other => panic!("{other:?}"),
        }
        // A relocation in a bank of the kernel's own to a place in a constant bank (a symbol of
        // bank 3): the address of the copy. To a function of the cubin: nothing.
        let in_bank = |symbol: u64| {
            let mut elf = relocation(symbol);
            elf.sections.last_mut().unwrap().name = ".rel.nv.constant0.k_bf16_sum".into();
            elf
        };
        let bank3 = parsed().section(".nv.constant3").unwrap();
        let constants = symbols.iter().position(|s| s.shndx as usize == bank3).unwrap() as u64;
        assert!(matches!(of(&in_bank(constants), "k_bf16_sum"), Reach::Data(_)));
        assert_eq!(of(&in_bank(kernel), "k_bf16_sum"), Reach::Nothing);
        // The words of bank 0, on the instruction of k_both that reads the stack pointer
        // (c[0x0][0x28]), made to read other words.
        let word_of_bank_0 = |offset: u32, size: Option<u64>| {
            let mut elf = parsed();
            if let Some(size) = size {
                let bank = elf.section(".nv.constant0.k_both").unwrap();
                elf.sections[bank].size = size;
            }
            let text = elf.section(".text.k_both").unwrap();
            let mut decoder = vf_sass::decode::Decoder::new(75);
            let first = u128::from_le_bytes(elf.sections[text].data[..16].try_into().unwrap());
            let d = decoder.decode(first).unwrap();
            let address = ["Sa_addr", "Sb_addr", "Sc_addr"]
                .into_iter()
                .find(|a| d.has_op_name(a))
                .unwrap();
            assert_eq!(4 * d.op_or_name(address, 0), 0x28, "{}", d.name);
            // (The address is a field of the word: found by where 0x28 / 4 is.)
            let bit = (0..114)
                .find(|&bit| {
                    (first >> bit) & 0x3fff == 0x28 / 4 && {
                        let changed = (first & !(0x3fffu128 << bit)) | (u128::from(0x50u32 / 4) << bit);
                        decoder
                            .decode(changed)
                            .is_some_and(|c| 4 * c.op_or_name(address, 0) == 0x50)
                    }
                })
                .unwrap();
            let changed = (first & !(0x3fffu128 << bit)) | (u128::from(offset / 4) << bit);
            elf.sections[text].data[..16].copy_from_slice(&changed.to_le_bytes());
            of(&elf, "k_both")
        };
        // (k_both has one parameter of eight bytes: its bank 0 ends at 0x168.)
        for own in [0x0, 0x4, 0x8, 0xc, 0x10, 0x14, 0x28, 0x118, 0x11c, 0x160, 0x164] {
            assert_eq!(word_of_bank_0(own, None), Reach::Nothing, "0x{own:x}");
        }
        // Past the parameters' end, and negative (c[0x0][-0x8] and [-0x4] are 0xfff8 and 0xfffc
        // in the field): data.
        for data in [
            0x18, 0x1c, 0x20, 0x24, 0x2c, 0x40, 0x44, 0x50, 0x54, 0x114, 0x120, 0x15c, 0x168, 0x400, 0x7ffc, 0x8000,
            0xfff8, 0xfffc,
        ] {
            assert!(matches!(word_of_bank_0(data, None), Reach::Data(_)), "0x{data:x}");
        }
        // In a bank as large as the field reaches, the parameters go as far, and a negative
        // offset is still not one.
        for own in [0x168, 0x400, 0x7ffc] {
            assert_eq!(word_of_bank_0(own, Some(0x1_0000)), Reach::Nothing, "0x{own:x}");
        }
        for data in [0x8000, 0xfff8, 0xfffc] {
            assert!(
                matches!(word_of_bank_0(data, Some(0x1_0000)), Reach::Data(_)),
                "0x{data:x}"
            );
        }
        // The load of a uniform register (ULDC), in a kernel that has one of its own bank 0: its
        // offset is in bytes, 16 bits.
        let uniform_load = |offset: u32, size: Option<u64>| {
            let mut elf = parsed();
            let mut decoder = vf_sass::decode::Decoder::new(75);
            let (kernel, text, at, word) = ["k_bf16_sum", "k_divides", "k_both"]
                .into_iter()
                .find_map(|kernel| {
                    let text = elf.section(&format!(".text.{kernel}")).unwrap();
                    elf.sections[text]
                        .data
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .enumerate()
                        .find_map(|(at, w)| {
                            let word = u128::from_le_bytes(*w);
                            let d = decoder.decode(word)?;
                            (d.name == "uldc_const__RCR" && d.op_or_name("Sa_bank", 9) == 0)
                                .then_some((kernel, text, at, word))
                        })
                })
                .expect("a ULDC of bank 0");
            let was = decoder.decode(word).unwrap().op_or_name("Sa_addr", 0);
            let bit = (0..112)
                .find(|&bit| {
                    (word >> bit) & 0xffff == u128::from(was) && {
                        let changed = (word & !(0xffffu128 << bit)) | (0x5554u128 << bit);
                        decoder
                            .decode(changed)
                            .is_some_and(|c| c.name == "uldc_const__RCR" && c.op_or_name("Sa_addr", 0) == 0x5554)
                    }
                })
                .unwrap();
            let changed = (word & !(0xffffu128 << bit)) | (u128::from(offset) << bit);
            elf.sections[text].data[16 * at..16 * at + 16].copy_from_slice(&changed.to_le_bytes());
            if let Some(size) = size {
                let bank = elf.section(&format!(".nv.constant0.{kernel}")).unwrap();
                elf.sections[bank].size = size;
            }
            of(&elf, kernel)
        };
        assert_eq!(uniform_load(0x160, None), Reach::Nothing);
        for data in [0x50, 0x162, 0x4000, 0x8000, 0xfff8, 0xfffc] {
            assert!(matches!(uniform_load(data, None), Reach::Data(_)), "0x{data:x}");
        }
        assert_eq!(uniform_load(0x4000, Some(0x1_0000)), Reach::Nothing);
        for data in [0x8000, 0xfff8, 0xfffc] {
            assert!(
                matches!(uniform_load(data, Some(0x1_0000)), Reach::Data(_)),
                "0x{data:x}"
            );
        }
        // (Eight bytes that end past the bank, a place that is not a word's.)
        assert!(own_word(0x160, 8, 0x168) && !own_word(0x164, 8, 0x168) && !own_word(0x162, 4, 0x168));
        assert!(texture("TEX") && texture("SULD") && texture("TLD4") && !texture("STG") && !texture("LDG"));

        // Turing code is read as for the second translation at launch, and with the banks.
        for (kernel, names_data) in [("k_fit_regs", false), ("k_fit_reads_global", true)] {
            assert_eq!(kernel_names_global_data(FIT, kernel), names_data, "{kernel}");
            assert_eq!(told_kernel_names_global_data(FIT, kernel), names_data, "{kernel}");
        }
        // Code of an architecture that the layer does not translate.
        let mut later = cubin.clone();
        let at = if later[8] >= 8 { 0x31 } else { 0x30 };
        later[at] = 86;
        assert!(matches!(told_kernel_reach(&later, "k_bf16_sum"), Reach::Unknown(_)));
    }

    #[test]
    fn the_code_that_a_gpu_of_the_capability_told_loads_is_found_beside_the_code_given() {
        let fat = fatbin::parse(SECTIONS).unwrap();
        for told_sm in [None, Some(75), Some(80), Some(86), Some(90)] {
            for keep_sources in [false, true] {
                let policy = Policy {
                    told_sm,
                    keep_sources,
                    ..Policy::default()
                };
                // Given from its sm_75 code, whatever is told, and nothing refused.
                let adapted = policy.adapt_fatbin(&fat).unwrap();
                assert!(
                    matches!(adapted.outcome, Outcome::Translated { from_sm: 75, ref refused, .. } if refused.is_empty()),
                    "told {told_sm:?}: {:?}",
                    adapted.outcome
                );
                // Told a later generation, a GPU of that capability loads the code for sm_80 (there is
                // nothing later up to 9.0): in the program's image, and in the image given to the
                // driver, which keeps that entry and where the proxy looks.
                let loads = told_sm.filter(|&sm| sm >= 80).map(|_| 80);
                assert_eq!(policy.told_machine_code(&fat, 75), loads, "told {told_sm:?}");
                let given = adapted.image.unwrap();
                assert_eq!(policy.told_machine_code(&fatbin::parse(&given).unwrap(), 75), loads);
            }
        }
        // Told the generation of the code given: nothing later.
        let told = Policy {
            told_sm: Some(80),
            ..Policy::default()
        };
        assert_eq!(told.told_machine_code(&fat, 80), None);
        // Which code has which kernel.
        assert!(has_kernel(&fat, 80, "k_from_80"));
        assert!(has_kernel(&fat, 80, "k_both"));
        assert!(!has_kernel(&fat, 80, "k_before_80"));
        assert!(!has_kernel(&fat, 80, "k_nowhere"));
        assert!(has_kernel(&fat, 75, "k_before_80"));
        assert!(!has_kernel(&fat, 75, "k_from_80"));
        assert!(!has_kernel(&fat, 86, "k_both"));
        // Why the sm_75 code is given; and for code that the layer does not translate.
        assert!(told.why_given(75, 80).ends_with(
            "its code for sm_75, whatever the capability told, as its translation of machine code for sm_80 covers \
             fewer instructions and is not independently validated"
        ));
        assert_eq!(
            told.why_given(75, 86),
            "the layer does not translate machine code for sm_86"
        );
    }

    #[test]
    fn choose_one_machine_target_and_keep_the_same_cubin_for_launch_retries() {
        let turing = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
        let ampere = || NewEntry::replace(&turing, DEBUG_CUBIN.to_vec(), 80, (1, 7));
        let policy = Policy::default();
        let told_ampere = Policy {
            told_sm: Some(80),
            ..policy
        };
        for order in [true, false] {
            let turing_entry = NewEntry::copy_of(&turing);
            let entries = if order {
                [turing_entry, ampere()]
            } else {
                [ampere(), turing_entry]
            };
            let image = fatbin::build(&entries);
            let fat = fatbin::parse(&image).unwrap();
            assert_eq!(policy.machine_source(&fat), Some(75));
            assert_eq!(policy.adapt_fatbin(&fat).unwrap().outcome, translated_fixture(1));
            // Told 8.0, the program is given the PTX that its sm_80 cubin retains: code for what
            // it is told, which it can be told.
            assert!(matches!(
                told_ampere.adapt_fatbin(&fat).unwrap().outcome,
                Outcome::Retargeted { from_sm: 80, .. }
            ));
            assert_eq!(told_ampere.given_from(&fat), Some(80));
            assert_eq!(told_ampere.told_limit(&fat), None);
            let first = Policy {
                translate: Translate::First,
                ..told_ampere
            };
            assert_eq!(first.machine_source(&fat), Some(80));
            assert_eq!(first.told_limit(&fat), None);
            assert_eq!(
                kept_cubin(&first.kept_for_second(&image).unwrap()).as_deref(),
                Some(DEBUG_CUBIN)
            );
            assert_eq!(
                kept_cubin(&policy.kept_for_second(&image).unwrap()).as_deref(),
                Some(&*turing.payload().unwrap())
            );
        }
        let mut stripped = vf_sass::elf::Elf::parse(DEBUG_CUBIN).unwrap();
        let section = stripped.section(".nv_debug_ptx_txt").unwrap();
        stripped.sections[section].data.fill(0);
        let mixed = fatbin::build(&[
            NewEntry::replace(&turing, stripped.write(), 80, (1, 7)),
            NewEntry::copy_of(&turing),
        ]);
        let mixed = fatbin::parse(&mixed).unwrap();
        // Without that PTX, what the program is given is the translation of its sm_75 code: told
        // 8.0, it would launch its kernels for sm_80, which that code has not. It is told 7.5.
        assert_eq!(told_ampere.adapt_fatbin(&mixed).unwrap().outcome, translated_fixture(1));
        assert_eq!(told_ampere.given_from(&mixed), Some(75));
        assert_eq!(told_ampere.told_limit(&mixed), Some(75));
        assert_eq!(policy.told_limit(&mixed), None);
        let duplicate = fatbin::build(&[ampere(), ampere()]);
        assert!(told_ampere.kept_for_second(&duplicate).is_none());
        assert!(kernel_names_global_data(DEBUG_CUBIN, "k"));
        let newer_device = Policy {
            device_sm: 90,
            ..policy
        };
        assert_eq!(
            newer_device.adapt_cubin(DEBUG_CUBIN).outcome,
            Outcome::NeedsTranslation { lowest_sm: 80 }
        );
        let native_ampere = Policy {
            device_sm: 80,
            ..policy
        };
        assert_eq!(native_ampere.adapt_cubin(DEBUG_CUBIN).outcome, Outcome::Native);
    }

    /// `entry` stored LZ4-compressed, as nvcc stores entries.
    fn compressed(entry: &Entry<'_>) -> NewEntry {
        let payload = entry.payload().unwrap();
        let packed = lz4_flex::block::compress(&payload);
        let mut stored = packed.clone();
        stored.resize(stored.len().next_multiple_of(8), 0);
        let mut header = entry.header.to_vec();
        header[0x08..0x10].copy_from_slice(&(stored.len() as u64).to_le_bytes());
        header[0x10..0x14].copy_from_slice(&(packed.len() as u32).to_le_bytes());
        header[0x28..0x30].copy_from_slice(&(entry.flags() | fatbin::FLAG_LZ4).to_le_bytes());
        header[0x38..0x40].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        NewEntry::copy_of(&Entry {
            header: &header,
            stored: &stored,
        })
    }

    #[test]
    fn what_a_program_is_told_and_given_agree_where_its_ampere_code_keeps_its_ptx() {
        let turing = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
        let plain = fatbin::build(&[
            NewEntry::copy_of(&turing),
            NewEntry::replace(&turing, DEBUG_CUBIN.to_vec(), 80, (1, 7)),
        ]);
        let ampere = fatbin::parse(&plain).unwrap().entries[1];
        // The same, with the sm_80 cubin stored compressed: its PTX is found all the same.
        let packed = fatbin::build(&[NewEntry::copy_of(&turing), compressed(&ampere)]);
        assert!(fatbin::parse(&packed).unwrap().entries[1].is_compressed());
        let told = |told_sm| Policy {
            told_sm,
            ..Policy::default()
        };
        for image in [&plain, &packed] {
            let fat = fatbin::parse(image).unwrap();
            // Told 8.0, the program is given the PTX of its sm_80 cubin: it can be told 8.0.
            assert_eq!(told(Some(80)).told_limit(&fat), None);
            assert!(matches!(
                told(Some(80)).adapt_fatbin(&fat).unwrap().outcome,
                Outcome::Retargeted { from_sm: 80, .. }
            ));
            // A program told 7.5 (for this code or other code of its own) is given its sm_75 code.
            assert_eq!(
                told(Some(75)).adapt_fatbin(&fat).unwrap().outcome,
                translated_fixture(1)
            );
            assert_eq!(told(None).adapt_fatbin(&fat).unwrap().outcome, translated_fixture(1));
        }
    }

    #[test]
    fn a_program_is_told_no_later_generation_than_the_code_it_is_given() {
        let fixture = fatbin::parse(ELF75_PTX75).unwrap();
        let (turing, ptx) = (fixture.entries[0], fixture.entries[1]);
        let stripped_ampere = include_bytes!("../../../tests/kernels/translate/k_ampere.sm80.cubin");
        let ampere = || NewEntry::replace(&turing, stripped_ampere.to_vec(), 80, (1, 7));
        // As PyTorch: machine code for several targets, sm_75 the first, no PTX.
        let multi = fatbin::build(&[NewEntry::copy_of(&turing), ampere()]);
        // As SGLang's kernels: machine code from sm_80 on.
        let from_ampere = fatbin::build(&[ampere()]);
        // Machine code and PTX for sm_75 only.
        let with_ptx = fatbin::build(&[NewEntry::copy_of(&turing), NewEntry::copy_of(&ptx)]);
        // As llama.cpp: PTX for sm_75, sm_80 and sm_90.
        let ptx_for = |sm: u32| {
            let text = format!(".version 9.5\n.target sm_{sm}\n.address_size 64\n");
            NewEntry::replace(&ptx, text.into_bytes(), sm, (9, 5))
        };
        let llama = fatbin::build(&[ptx_for(75), ptx_for(80), ptx_for(90)]);
        let native = ELF70_ELF75_PTX70.to_vec();
        // Machine code for sm_75 and for sm_80, which keeps its PTX (`-lineinfo`), stored as it is
        // and compressed.
        let keeps = fatbin::build(&[
            NewEntry::copy_of(&turing),
            NewEntry::replace(&turing, DEBUG_CUBIN.to_vec(), 80, (1, 7)),
        ]);
        let keeps_packed = fatbin::build(&[
            NewEntry::copy_of(&turing),
            compressed(&fatbin::parse(&keeps).unwrap().entries[1]),
        ]);
        let limit = |told_sm: Option<u32>, translate: Translate, image: &[u8]| {
            let policy = Policy {
                told_sm,
                translate,
                ..Policy::default()
            };
            let fat = fatbin::parse(image).unwrap();
            let limit = policy.told_limit(&fat);
            // The limit is the architecture of what is given, machine code or PTX; without a limit,
            // what is given is no code of an earlier generation than what is told.
            let outcome = policy.adapt_fatbin(&fat).unwrap().outcome;
            let given = match outcome {
                Outcome::Translated { from_sm, .. } | Outcome::Retargeted { from_sm, .. } => Some(from_sm),
                Outcome::Native => Some(policy.device_sm),
                _ => None,
            };
            match (limit, told_sm) {
                // (7.5 for code of Volta's generation.)
                (Some(sm), _) => assert_eq!(given.map(|g| g.max(TRANSLATED_FROM)), Some(sm), "{outcome:?}"),
                (None, Some(told)) => assert!(given.is_none_or(|sm| sm / 10 >= told / 10), "{outcome:?}"),
                (None, None) => {}
            }
            limit
        };
        let default = Translate::WithoutPtx;
        for told in [None, Some(70), Some(75)] {
            for image in [&multi, &from_ampere, &with_ptx, &llama, &native, &keeps, &keeps_packed] {
                assert_eq!(limit(told, default, image), None);
            }
        }
        for told in [80, 86, 90, 120] {
            assert_eq!(limit(Some(told), default, &multi), Some(75), "{told}");
            // Given its PTX for sm_75: what `__CUDA_ARCH__` chose there is for 7.5.
            assert_eq!(limit(Some(told), default, &with_ptx), Some(75), "{told}");
            // Code for this GPU, which runs as it is: what it chose for a later generation is not
            // there either (until 2026-11-23 it set no limit).
            assert_eq!(limit(Some(told), default, &native), Some(75), "{told}");
            // Machine code is not translated: nothing is given, and the driver says so.
            assert_eq!(limit(Some(told), Translate::Never, &multi), None, "{told}");
            // Machine code before PTX: what is given of this one is its sm_75 code.
            assert_eq!(limit(Some(told), Translate::First, &with_ptx), Some(75), "{told}");
            assert_eq!(limit(Some(told), Translate::Force, &with_ptx), Some(75), "{told}");
        }
        // An sm_80 translation is code of the same generation as 8.6, not of 9.0 and later; so is
        // the PTX of an sm_80 cubin, compressed or not.
        for (told, expected) in [(80, None), (86, None), (90, Some(80)), (120, Some(80))] {
            assert_eq!(limit(Some(told), default, &from_ampere), expected, "{told}");
            assert_eq!(limit(Some(told), default, &keeps), expected, "{told}");
            assert_eq!(limit(Some(told), default, &keeps_packed), expected, "{told}");
            // Asked for, the machine code of the told generation is translated where there is some.
            assert_eq!(limit(Some(told), Translate::First, &multi), expected, "{told}");
        }
        // In a file: the lowest over its fat binaries.
        let policy = Policy {
            told_sm: Some(90),
            ..Policy::default()
        };
        // With PTX for the told generation, or the one before: the PTX for that one.
        for (told, expected) in [(80, None), (86, None), (90, None), (120, Some(90))] {
            assert_eq!(limit(Some(told), default, &llama), expected, "{told}");
        }
        let mut file = b"\x7fELF....".to_vec();
        for image in [&llama, &from_ampere] {
            file.extend_from_slice(image);
        }
        assert_eq!(policy.told_limit_in(&file), Some(80));
        let mut with_native = file.clone();
        with_native.extend_from_slice(&native);
        assert_eq!(policy.told_limit_in(&with_native), Some(75));
        file.extend_from_slice(&native);
        file.extend_from_slice(&[0x50, 0xed, 0x55, 0xba, 9, 9]); // the magic number, not a fat binary
        file.extend_from_slice(&multi);
        assert_eq!(policy.told_limit_in(&file), Some(75));
        assert_eq!(Policy::default().told_limit_in(&file), None);
        assert_eq!(policy.told_limit_in(b"\x7fELF no fat binary here"), None);
        // PTX for sm_75 and sm_90 sets no limit at 9.0, and PTX for sm_80 sets 8.0; told 8.0, the
        // first is given its PTX for sm_75, which sets 7.5. Together, in either order, the limit is
        // 7.5, whatever the generation asked after Turing's (until 2026-11-23: 8.0 at 9.0 and 12.0,
        // and the program told 8.0 was given the code for sm_75 of the first).
        let ptx_75_90 = fatbin::build(&[ptx_for(75), ptx_for(90)]);
        let ptx_80 = fatbin::build(&[ptx_for(80)]);
        assert_eq!(limit(Some(90), default, &ptx_75_90), None);
        assert_eq!(limit(Some(90), default, &ptx_80), Some(80));
        assert_eq!(limit(Some(80), default, &ptx_75_90), Some(75));
        for images in [[&ptx_75_90, &ptx_80], [&ptx_80, &ptx_75_90]] {
            let mut file = b"\x7fELF....".to_vec();
            for image in images {
                file.extend_from_slice(image);
            }
            for told in [80, 86, 90, 120] {
                let asked = Policy {
                    told_sm: Some(told),
                    ..policy
                };
                assert_eq!(asked.told_limit_in(&file), Some(75), "{told}");
            }
        }
        // The fixed point itself: lowered while the limit is below what is told.
        let steps = |told: Option<u32>| match told {
            Some(120) => Some(90),
            Some(90) => Some(80),
            Some(80) => Some(75),
            _ => None,
        };
        let asked_12 = Policy {
            told_sm: Some(120),
            ..policy
        };
        assert_eq!(asked_12.lowered(|p| steps(p.told_sm)), Some(75));
        assert_eq!(asked_12.lowered(|p| p.told_sm), None);
        assert_eq!(Policy::default().lowered(|_| Some(75)), None);
        // A file with code that limits what its program is told to 7.5, and code for sm_75 and
        // sm_80 that alone would not: the program, told 7.5, is given the sm_75 code of both.
        let mut file = b"\x7fELF....".to_vec();
        for image in [&keeps_packed, &multi, &keeps] {
            file.extend_from_slice(image);
        }
        assert_eq!(policy.told_limit_in(&file), Some(75));
        let limited = Policy {
            told_sm: policy.told_limit_in(&file),
            ..policy
        };
        for image in [&keeps_packed, &multi, &keeps] {
            let fat = fatbin::parse(image).unwrap();
            assert!(matches!(
                limited.adapt_fatbin(&fat).unwrap().outcome,
                Outcome::Translated { from_sm: 75, .. }
            ));
            assert_eq!(limited.told_limit(&fat), None);
        }
        // With PTX for sm_75 and for sm_80: the PTX for sm_75.
        let both_ptx = fatbin::build(&[
            NewEntry::copy_of(&ptx),
            NewEntry::replace(
                &ptx,
                b".version 9.5\n.target sm_80\n.address_size 64\n".to_vec(),
                80,
                (9, 5),
            ),
        ]);
        let fat = fatbin::parse(&both_ptx).unwrap();
        assert!(matches!(
            limited.adapt_fatbin(&fat).unwrap().outcome,
            Outcome::Retargeted { from_sm: 75, .. }
        ));
        assert!(matches!(
            policy.adapt_fatbin(&fat).unwrap().outcome,
            Outcome::Retargeted { from_sm: 80, .. }
        ));
    }

    #[test]
    fn stripped_ampere_code_keeps_its_per_kernel_refusals_through_the_public_policy() {
        let cubin = include_bytes!("../../../tests/kernels/translate/k_ampere.sm80.cubin");
        let policy = Policy::default();
        let bare = policy.adapt_cubin(cubin);
        assert!(
            matches!(&bare.outcome, Outcome::Translated { from_sm: 80, functions: 2, refused, .. }
            if refused.len() == 1 && refused[0].kernel == "unsupported_atomic")
        );
        let native = fatbin::parse(ELF70_ELF75_PTX70).unwrap().entries[0];
        let ampere = || NewEntry::replace(&native, cubin.to_vec(), 80, (1, 7));
        let single = fatbin::build(&[ampere()]);
        let single = fatbin::parse(&single).unwrap();
        assert_eq!(policy.adapt_fatbin(&single).unwrap().outcome, bare.outcome);
        let never = Policy {
            translate: Translate::Never,
            ..policy
        };
        assert_eq!(
            never.adapt_fatbin(&single).unwrap().outcome,
            Outcome::NeedsTranslation { lowest_sm: 80 }
        );
        assert_eq!(
            never.adapt_cubin(cubin).outcome,
            Outcome::NeedsTranslation { lowest_sm: 80 }
        );
        let mixed = fatbin::build(&[NewEntry::copy_of(&native), ampere()]);
        let mixed = fatbin::parse(&mixed).unwrap();
        let first = Policy {
            translate: Translate::First,
            ..policy
        };
        assert_eq!(first.adapt_fatbin(&mixed).unwrap().outcome, Outcome::Native);
        let force = Policy {
            translate: Translate::Force,
            ..policy
        };
        let forced = force.adapt_fatbin(&mixed).unwrap();
        assert_eq!(forced.outcome, bare.outcome);
        let bytes = forced.image.unwrap();
        let fat = fatbin::parse(&bytes).unwrap();
        assert_eq!(fat.entries.len(), 1);
        assert_eq!(fat.entries[0].sm(), 70);
    }

    #[test]
    fn a_library_keeps_the_entries_of_other_gpus_beside_their_replacements() {
        let cubin = include_bytes!("../../../tests/kernels/translate/k_ampere.sm80.cubin");
        let native = fatbin::parse(ELF70_ELF75_PTX70).unwrap().entries[0];
        let ampere = fatbin::build(&[NewEntry::replace(&native, cubin.to_vec(), 80, (1, 7))]);
        let ptx = fatbin::parse(ELF75_PTX75).unwrap().entries[1];
        let ptx_80 = fatbin::build(&[NewEntry::replace(
            &ptx,
            b".version 9.5\n.target sm_80\n.address_size 64\n".to_vec(),
            80,
            (9, 5),
        )]);
        let kept = |image: &[u8]| -> Vec<(u16, u32)> {
            fatbin::parse(image)
                .unwrap()
                .entries
                .iter()
                .map(|e| (e.kind(), e.sm()))
                .collect()
        };
        let library = Policy {
            keep_sources: true,
            ..Policy::default()
        };
        for (image, alone, beside) in [
            (&ampere, vec![(KIND_ELF, 70)], vec![(KIND_ELF, 80), (KIND_ELF, 70)]),
            (&ptx_80, vec![(KIND_PTX, 70)], vec![(KIND_PTX, 80), (KIND_PTX, 70)]),
            (
                &fatbin::build(&[NewEntry::replace(&native, DEBUG_CUBIN.to_vec(), 80, (1, 7))]),
                vec![(KIND_PTX, 70)],
                vec![(KIND_ELF, 80), (KIND_PTX, 70)],
            ),
        ] {
            let fat = fatbin::parse(image).unwrap();
            // A module is loaded in the context of one GPU: the V100's code alone.
            let module = Policy::default().adapt_fatbin(&fat).unwrap();
            assert_eq!(kept(&module.image.unwrap()), alone);
            // A library, for every GPU of the process: the V100's, and the other cards' as it was
            // (until 2026-11-23 the V100's alone), the same outcome.
            let adapted = library.adapt_fatbin(&fat).unwrap();
            assert_eq!(adapted.outcome, Policy::default().adapt_fatbin(&fat).unwrap().outcome);
            let image = adapted.image.unwrap();
            assert_eq!(kept(&image), beside);
            let fat_again = fatbin::parse(&image).unwrap();
            assert_eq!(fat_again.entries[0].stored, fat.entries[0].stored);
            // What the driver takes on the V100 is the replacement, and the second retry still
            // finds the one PTX entry it reaches.
            assert_eq!(
                Policy::default().adapt_fatbin(&fat_again).unwrap().outcome,
                Outcome::Native
            );
        }
    }

    #[test]
    fn recover_turing_ptx_only_when_no_translated_entry_can_run() {
        // ptxas predicates the Turing HMMA, but the retained PTX branches around the MMA.
        let cubin = include_bytes!("../../../tests/kernels/ptx-cases/guarded-mma.sm75.cubin");
        let policy = Policy::default();
        let machine = Policy {
            translate: Translate::First,
            ..policy
        };
        assert!(
            matches!(machine.adapt_cubin(cubin).outcome, Outcome::Translated { ref refused, .. } if refused.len() == 1)
        );
        let recovered = policy.adapt_cubin(cubin);
        assert!(recovered.complete_ptx());
        let fixture = fatbin::parse(ELF75_PTX75).unwrap();
        let good = fixture.entries[0];
        let guarded = || NewEntry::replace(&good, cubin.to_vec(), 75, (1, 7));
        let one = fatbin::build(&[guarded()]);
        let one = fatbin::parse(&one).unwrap();
        assert_eq!(policy.adapt_fatbin(&one).unwrap().outcome, recovered.outcome);
        assert_eq!(
            machine.adapt_fatbin(&one).unwrap().outcome,
            machine.adapt_cubin(cubin).outcome
        );
        let mixed = fatbin::build(&[guarded(), NewEntry::copy_of(&good)]);
        assert!(
            matches!(policy.adapt_fatbin(&fatbin::parse(&mixed).unwrap()).unwrap().outcome,
            Outcome::Translated { ref refused, entries: 2, .. } if refused.len() == 1)
        );
        let mut missing = cubin.to_vec();
        let section = b".nv_debug_ptx_txt";
        let offset = missing.windows(section.len()).position(|s| s == section).unwrap();
        missing[offset] = b'_';
        assert_eq!(policy.adapt_cubin(&missing).outcome, machine.adapt_cubin(cubin).outcome);
    }

    #[test]
    fn only_compiler_helpers_inside_an_entry_may_be_weak_symbols() {
        let input = include_bytes!("../../../tests/kernels/ptx-cases/signed-division.sm80.cubin");
        let policy = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        assert!(matches!(
            policy.adapt_cubin(input).outcome,
            Outcome::Retargeted { from_sm: 80, .. }
        ));
        for edit in 0..4 {
            let mut elf = vf_sass::elf::Elf::parse(input).unwrap();
            let mut symbols = elf.symbols().unwrap();
            let index = symbols.iter().position(|s| s.name.ends_with("_div_s64")).unwrap();
            let helper = &mut symbols[index];
            assert_eq!(helper.info, 0x22);
            match edit {
                0 => helper.name_offset += 1, // outside ptxas's internal namespace
                1 => helper.value = u64::MAX, // outside its entry's range
                2 => helper.shndx = 0,        // external definition
                _ => {
                    let symtab = elf.section(".symtab").unwrap();
                    elf.sections[symtab].info = index as u32; // public weak symbol
                }
            }
            elf.set_symbols(&symbols);
            assert!(policy.adapt_cubin(&elf.write()).image.is_none());
        }
    }

    #[test]
    fn retained_ptx_may_call_what_its_source_declares_extern() {
        let input = include_bytes!("../../../tests/kernels/ptx-cases/assert-import.sm80.cubin");
        assert_eq!(fatbin::cubin_imports(input).unwrap(), ["__assertfail"]);
        for policy in [
            Policy::default(),
            Policy {
                translate: Translate::Never,
                ..Policy::default()
            },
        ] {
            assert!(matches!(
                policy.adapt_cubin(input).outcome,
                Outcome::Retargeted { from_sm: 80, .. }
            ));
        }
        let policy = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        for edit in 0..7 {
            let mut elf = vf_sass::elf::Elf::parse(input).unwrap();
            let mut symbols = elf.symbols().unwrap();
            let index = symbols.iter().position(|s| s.name == "__assertfail").unwrap();
            let import = &mut symbols[index];
            assert_eq!((import.info, import.other, import.shndx), (0x12, 0, 0));
            match edit {
                0 => import.name_offset += 2, // `assertfail`, which the source does not declare
                1 => import.shndx = 1,        // a definition: part of the module's interface
                2 => import.other = 2,        // hidden: not a plain call to the loader
                3 => import.info = 0x11,      // an undefined object, not a function
                4 => import.info = 0x22,      // weak: not a plain call to the loader
                5 => import.value = 0x10,     // a value: not an import
                _ => {
                    let symtab = elf.section(".symtab").unwrap();
                    elf.sections[symtab].info = index as u32 + 1; // in the local part
                }
            }
            elf.set_symbols(&symbols);
            assert!(policy.adapt_cubin(&elf.write()).image.is_none(), "edit {edit}");
        }
        // With an import, the source is compiled again as the machine code was: the addition that
        // ptxas did not contract (`--fmad=false`) keeps the rounding of its own.
        let separate = include_bytes!("../../../tests/kernels/ptx-cases/product-sum-assert.fmad-false.sm80.cubin");
        assert_eq!(fatbin::cubin_imports(separate).unwrap(), ["__assertfail"]);
        let text = retargeted_text(&Policy::default(), separate).unwrap();
        assert!(
            text.contains("\nmul.f32 %f4, %f1, %f2;\nadd.rn.f32 %f5, %f4, %f3;"),
            "{text}"
        );
    }

    /// The text of the image that `policy` makes of the sm_80 `cubin`, or what it found when it
    /// retargets nothing.
    fn retargeted_text(policy: &Policy, cubin: &[u8]) -> Result<String, Box<Outcome>> {
        let adapted = policy.adapt_cubin(cubin);
        match (&adapted.outcome, adapted.image) {
            (Outcome::Retargeted { from_sm: 80, .. }, Some(image)) => Ok(String::from_utf8(image).unwrap()),
            _ => Err(Box::new(adapted.outcome)),
        }
    }

    #[test]
    fn retained_ptx_rounds_as_its_machine_code_did() {
        let separate = include_bytes!("../../../tests/kernels/ptx-cases/product-sum.fmad-false.sm80.cubin");
        let fused = include_bytes!("../../../tests/kernels/ptx-cases/product-sum.fmad-true.sm80.cubin");
        let never = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        for policy in [Policy::default(), never] {
            // ptxas did not contract the product and the sum: the driver's compiler must not either.
            let text = retargeted_text(&policy, separate).unwrap();
            assert!(
                text.contains("\nmul.f32 %f4, %f1, %f2;\nadd.rn.f32 %f5, %f4, %f3;"),
                "{text}"
            );
            assert_eq!(text.matches(".rn.").count(), 1);
            // It did: the source is left as it is, and contracted again.
            let text = retargeted_text(&policy, fused).unwrap();
            assert!(
                text.contains("\nmul.f32 %f4, %f1, %f2;\nadd.f32 %f5, %f4, %f3;"),
                "{text}"
            );
            assert!(!text.contains(".rn."));
        }
        // The same in a fat binary.
        let template = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
        let bytes = fatbin::build(&[NewEntry::replace(&template, separate.to_vec(), 80, (1, 7))]);
        let image = Policy::default()
            .adapt_fatbin(&fatbin::parse(&bytes).unwrap())
            .unwrap()
            .image
            .unwrap();
        let entry = fatbin::parse(&image).unwrap().entries[0];
        assert!(String::from_utf8_lossy(&entry.payload().unwrap()).contains("\nadd.rn.f32 %f5, %f4, %f3;"));
        // Without the line table, what ptxas made of the sum is not known, whichever it was: the
        // source is not used, and the machine code is translated (or nothing is given).
        for cubin in [&separate[..], &fused[..]] {
            let mut unknown = cubin.to_vec();
            let name = b".nv_debug_line_sass";
            while let Some(at) = unknown.windows(name.len()).position(|w| w == name) {
                unknown[at] = b'_';
            }
            assert!(
                vf_sass::elf::Elf::parse(&unknown)
                    .unwrap()
                    .section("_nv_debug_line_sass")
                    .is_some()
            );
            assert!(matches!(
                Policy::default().adapt_cubin(&unknown).outcome,
                Outcome::Translated { from_sm: 80, .. }
            ));
            assert_eq!(
                never.adapt_cubin(&unknown).outcome,
                Outcome::NeedsTranslation { lowest_sm: 80 }
            );
        }
    }

    #[test]
    fn partial_mismatched_and_relocatable_debug_sources_are_left_alone() {
        let policy = Policy {
            translate: Translate::Never,
            ..Policy::default()
        };
        let edit = |from: &[u8], to: &[u8]| {
            let mut elf = vf_sass::elf::Elf::parse(DEBUG_CUBIN).unwrap();
            let index = elf.section(".nv_debug_ptx_txt").unwrap();
            let text = &mut elf.sections[index].data;
            let at = text.windows(from.len()).position(|w| w == from).unwrap();
            text.splice(at..at + from.len(), to.iter().copied());
            elf.write()
        };
        let mut relocatable = DEBUG_CUBIN.to_vec();
        relocatable[16] = 1;
        let mut with_global = vf_sass::elf::Elf::parse(DEBUG_CUBIN).unwrap();
        let mut symbols = with_global.symbols().unwrap();
        symbols.iter_mut().find(|s| s.name == "source").unwrap().info |= 0x10;
        with_global.set_symbols(&symbols);
        for bad in [
            edit(b".version 7.0", b".version 7.0\0.version 7.0"),
            edit(b".target sm_80", b".target sm_80\0.target sm_80"),
            edit(b".target sm_80", b".target sm_86"),
            edit(b".entry k(", b".entry q("),
            edit(b"ret;", b"\xff"),
            relocatable,
            with_global.write(),
        ] {
            assert_eq!(
                policy.adapt_cubin(&bad).outcome,
                Outcome::NeedsTranslation { lowest_sm: 80 }
            );
            let template = fatbin::parse(ELF75_PTX75).unwrap().entries[0];
            let bytes = fatbin::build(&[
                NewEntry::replace(&template, DEBUG_CUBIN.to_vec(), 80, (1, 7)),
                NewEntry::replace(&template, bad, 80, (1, 7)),
            ]);
            assert!(
                policy
                    .adapt_fatbin(&fatbin::parse(&bytes).unwrap())
                    .unwrap()
                    .image
                    .is_none()
            );
        }
    }

    #[test]
    fn cubins() {
        let policy = Policy::default();
        let fat = fatbin::parse(ELF70_ELF75_PTX70).unwrap();
        assert_eq!(policy.adapt_cubin(fat.entries[0].stored).outcome, Outcome::Native);
        // Machine code for Turing: translated, its seven functions taken.
        let adapted = policy.adapt_cubin(fat.entries[1].stored);
        let Outcome::Translated {
            from_sm: 75,
            functions: 7,
            refused,
            ..
        } = adapted.outcome
        else {
            panic!("{:?}", adapted.outcome);
        };
        assert_eq!(refused, []);
        assert_eq!(fatbin::cubin_sm(&adapted.image.unwrap()), Some(70));
        // The same with an instruction of one function that is none of Turing's: that function is
        // refused with the reason, the six others are translated.
        let mut damaged = fat.entries[1].stored.to_vec();
        let elf = vf_sass::elf::Elf::parse(&damaged).unwrap();
        let at = elf.sections[elf.section(".text.warp_sum").unwrap()].offset as usize;
        damaged[at..at + 16].fill(0xff);
        let adapted = policy.adapt_cubin(&damaged);
        let Outcome::Translated {
            functions: 7, refused, ..
        } = adapted.outcome
        else {
            panic!("{:?}", adapted.outcome);
        };
        assert_eq!(
            refused,
            [Refused {
                kernel: "warp_sum".to_string(),
                function: "warp_sum".to_string(),
                reason: "its machine code for sm_75 is not translated (the instruction \
                         ffffffffffffffffffffffffffffffff at 0x0 is not one of the sm_75 description)"
                    .to_string(),
            }]
        );
        assert_eq!(fatbin::cubin_sm(&adapted.image.unwrap()), Some(70));
        // For an architecture nothing translates.
        let mut later = fat.entries[1].stored.to_vec();
        later[0x30] = 86;
        assert_eq!(fatbin::cubin_sm(&later), Some(86));
        assert_eq!(
            policy.adapt_cubin(&later).outcome,
            Outcome::NeedsTranslation { lowest_sm: 86 }
        );
        assert_eq!(policy.adapt_cubin(b"\x7fELF").outcome, Outcome::Unknown);
    }

    /// The kernels of `tests/kernels/k_fit_test.cu` (the test of the second translation at launch).
    const FIT: &[u8] = include_bytes!("../../../tests/kernels/k_fit_test.135.sm75.cubin");

    #[test]
    fn data_of_the_module_excludes_a_kernel() {
        for kernel in ["k_fit_regs", "k_fit_shared", "k_fit_simple"] {
            assert!(!kernel_names_global_data(FIT, kernel), "{kernel}");
        }
        // These two read the bank that holds the addresses of `fit_scale` and `fit_seen`.
        for kernel in ["k_fit_reads_global", "k_fit_writes_global"] {
            assert!(kernel_names_global_data(FIT, kernel), "{kernel}");
        }
        assert!(!kernel_names_global_data(FIT, "no_such_kernel"));
        assert!(kernel_names_global_data(b"\x7fELF", "k_fit_regs"));
    }

    #[test]
    fn the_usual_translation_passes_the_budget_and_the_second_keeps_to_it() {
        let registers = |image: &[u8], name: &str| {
            cubin_function_registers(image)
                .into_iter()
                .find(|f| f.name == name)
                .map(|f| f.registers)
        };
        // k_fit_regs declares 64 registers: 1,024 threads of it can be launched. Translated the usual
        // way it takes 66, and they cannot.
        let usual = Policy::default().adapt_cubin(FIT).image.unwrap();
        assert_eq!(registers(FIT, "k_fit_regs"), Some(64));
        assert_eq!(registers(&usual, "k_fit_regs"), Some(66));
        assert!(translation_exceeds_budget(FIT, &usual));
        let second = Policy {
            borrow: 5,
            ..Policy::default()
        }
        .adapt_cubin(FIT)
        .image
        .unwrap();
        assert_eq!(registers(&second, "k_fit_regs"), Some(64));
        assert_eq!(registers(&second, "k_fit_reads_global"), Some(64));
        assert!(!translation_exceeds_budget(FIT, &second));
        // The translation says so itself, and what is kept for a second one is the cubin.
        let over = |policy: Policy| match policy.adapt_cubin(FIT).outcome {
            Outcome::Translated { over_budget, .. } => over_budget,
            other => panic!("{other:?}"),
        };
        assert!(over(Policy::default()));
        assert!(!over(Policy {
            borrow: 5,
            ..Policy::default()
        }));
        assert_eq!(Policy::default().kept_for_second(FIT).as_deref(), Some(FIT));
        assert_eq!(kept_cubin(FIT).as_deref(), Some(FIT));
        // From a fat binary whose machine code is for sm_75 (it has PTX too: translated when the
        // policy says machine code first): the one entry, as it is stored, and its cubin back.
        let policy = Policy {
            translate: Translate::First,
            ..Policy::default()
        };
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let adapted = policy.adapt_fatbin(&fat).unwrap();
        // (Its kernels are small: nothing of it is over its budget.)
        assert!(
            matches!(adapted.outcome, Outcome::Translated { over_budget: false, .. }),
            "{:?}",
            adapted.outcome
        );
        let cubin = fat.entries.iter().find(|entry| entry.kind() == KIND_ELF).unwrap();
        let kept = policy.kept_for_second(ELF75_PTX75).unwrap();
        assert!(kept.len() < ELF75_PTX75.len());
        assert_eq!(kept_cubin(&kept).as_deref(), Some(&*cubin.payload().unwrap()));
    }

    #[test]
    fn safe_mode_is_the_exact_value_one() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(Safe::from_value(None), Safe::Off);
        assert_eq!(Safe::from_value(Some(OsStr::new("1"))), Safe::On);
        assert_eq!(Safe::from_value(Some(OsStr::new("0"))), Safe::Off);
        for other in ["", "true", "yes", "on", " 1", "1 ", "01", "1\n", "2", "-1", "TRUE"] {
            assert_eq!(
                Safe::from_value(Some(OsStr::new(other))),
                Safe::NotOne(other.to_string()),
                "{other:?}"
            );
            assert!(!Safe::from_value(Some(OsStr::new(other))).on());
        }
        assert!(Safe::from_value(Some(OsStr::new("1"))).on());
        // A value that is not text is not 1 either.
        assert!(matches!(
            Safe::from_value(Some(OsStr::from_bytes(b"1\xff"))),
            Safe::NotOne(_)
        ));
        // What is said: nothing for unset and 0; the value that is not 1; what safe mode turned off.
        assert_eq!(safe_line(&Safe::Off, &[]), None);
        assert_eq!(
            safe_line(&Safe::NotOne(" 1".into()), &[]).as_deref(),
            Some("VF_SAFE=' 1' is not 1: safe mode stays off")
        );
        assert_eq!(
            safe_line(&Safe::On, &[]).as_deref(),
            Some("VF_SAFE=1: the layer keeps its conservative paths (nothing that safe mode turns off is in use)")
        );
        let line = safe_line(&Safe::On, &["VF_BF16_TC_GROUPS=1", "VF_BF16_ROW=tree"]).unwrap();
        assert!(
            line.starts_with(
                "VF_SAFE=1: the layer keeps its conservative paths: VF_BF16_TC_GROUPS=1, VF_BF16_ROW=tree not used"
            ),
            "{line}"
        );
    }

    #[test]
    fn safe_mode_turns_off_what_changes_results() {
        use vf_ptx::lower::{Bf16Column, Bf16Row};
        const PAIRS: &str = "VF_PAIRS in machine code";
        // The default in safe mode: the pairs of machine code off, those of program text on.
        let safe_default = Policy {
            safe: true,
            pairs: false,
            ..Policy::default()
        };
        assert!(safe_default.lowering.pairs);
        let mut default = Policy::default();
        assert_eq!(default.make_safe(), vec![PAIRS]);
        assert_eq!(format!("{default:?}"), format!("{safe_default:?}"));
        type Change<'a> = &'a dyn Fn(&mut Policy);
        let with = |change: Change| {
            let mut policy = Policy::default();
            change(&mut policy);
            policy
        };
        // Each switch whose results are not bit-identical to the default's: put back, and named.
        let changing: [(&str, Change); 7] = [
            ("VF_BF16_TC_GROUPS=1", &|p| p.lowering.bf16_tc_groups = true),
            ("VF_BF16_TC_TILES=1", &|p| p.lowering.bf16_tc_tiles = true),
            ("VF_BF16_COLUMN=tree", &|p| p.lowering.bf16_column = Bf16Column::Tree),
            ("VF_BF16_ROW=tree", &|p| p.lowering.bf16_row = Bf16Row::Tree),
            ("VF_TRANSLATE=first", &|p| p.translate = Translate::First),
            ("VF_TRANSLATE=force", &|p| p.translate = Translate::Force),
            ("VF_COMPANIONS=1", &|p| p.companions = true),
        ];
        for (name, change) in changing {
            let mut policy = with(change);
            assert_ne!(format!("{policy:?}"), format!("{:?}", Policy::default()), "{name}");
            assert_eq!(policy.make_safe(), vec![name, PAIRS]);
            assert_eq!(format!("{policy:?}"), format!("{safe_default:?}"), "{name}");
        }
        // All of them at once, in one list.
        let mut all = with(&|p| {
            p.lowering.bf16_tc_groups = true;
            p.lowering.bf16_tc_tiles = true;
            p.lowering.bf16_column = Bf16Column::Tree;
            p.lowering.bf16_row = Bf16Row::Tree;
            p.translate = Translate::Force;
            p.companions = true;
        });
        assert_eq!(all.make_safe().len(), 7);
        assert_eq!(format!("{all:?}"), format!("{safe_default:?}"));
        // With VF_PAIRS=0 there is nothing to turn off, and it stays as it is.
        let mut unpaired = with(&|p| {
            p.pairs = false;
            p.lowering.pairs = false;
        });
        let before = unpaired;
        assert!(unpaired.make_safe().is_empty(), "{before:?}");
        assert_eq!(
            format!("{unpaired:?}"),
            format!("{:?}", Policy { safe: true, ..before })
        );
        // What gives the default's bits, what refuses instead of computing otherwise, and what the
        // programs are told: left as they are (the pairs of machine code apart).
        let kept: [Change; 6] = [
            &|p| p.lowering.bf16_column = Bf16Column::Ordered,
            &|p| p.lowering.bf16_row = Bf16Row::Ordered,
            &|p| p.translate = Translate::Never,
            &|p| p.told_sm = Some(80),
            &|p| p.borrow = 9,
            &|p| p.lowering.aligned_rows = false,
        ];
        for change in kept {
            let mut policy = with(change);
            let before = policy;
            assert_eq!(policy.make_safe(), vec![PAIRS], "{before:?}");
            assert_eq!(
                format!("{policy:?}"),
                format!(
                    "{:?}",
                    Policy {
                        safe: true,
                        pairs: false,
                        ..before
                    }
                )
            );
        }
        // Made in safe mode or not, two policies are told apart, even when nothing was turned off.
        assert_ne!(format!("{:?}", Policy { safe: true, ..before }), format!("{before:?}"));
    }

    #[test]
    fn safe_mode_translates_each_product_of_machine_code_by_itself() {
        // Eight products a turn on two A, from whole-row loads (tests/kernels/k_pair_bits.cu): the
        // translator makes pairs of them by default, and none with VF_PAIRS=0 in machine code.
        let cubin = include_bytes!("../../../tests/kernels/k_pair_bits.135.sm75.cubin");
        let ptx = include_bytes!("../../../tests/kernels/k_pair_bits.135.sm75.ptx");
        let image = |policy: &Policy| policy.adapt_cubin(cubin).image.unwrap();
        let mut safe = Policy::default();
        safe.make_safe();
        // As the layer translates by itself, and within the budget (the second translation at
        // launch): pairs by default (the control), each product by itself in safe mode.
        for borrow in [vf_sass::translate::BORROW_DEFAULT, 5] {
            let default = Policy {
                borrow,
                ..Policy::default()
            };
            let unpaired = Policy {
                pairs: false,
                ..default
            };
            assert_ne!(image(&default), image(&unpaired), "VF_BORROW={borrow}");
            assert_eq!(
                image(&Policy { borrow, ..safe }),
                image(&unpaired),
                "VF_BORROW={borrow}"
            );
        }
        // The pairs of program text give the bits of each product by itself: they stay (in the PTX
        // of the same source, the readers after the loop are not where the 32 lanes are known to
        // run together, and it has none: tests/kernels/ptx-cases/pairs-adversarial.ptx has some).
        let text = |policy: &Policy, ptx: &[u8]| policy.adapt_ptx(ptx).image.unwrap();
        assert_eq!(text(&safe, ptx), text(&Policy::default(), ptx));
        let ptx = include_bytes!("../../../tests/kernels/ptx-cases/pairs-adversarial.ptx");
        let text = |policy: &Policy| text(policy, ptx);
        let mut unpaired = Policy::default();
        unpaired.lowering.pairs = false;
        assert_ne!(text(&Policy::default()), text(&unpaired));
        assert_eq!(text(&safe), text(&Policy::default()));
    }

    #[test]
    fn policy_companions_is_opt_in_and_respects_unpaired_and_safe() {
        let cubin = include_bytes!("../../../tests/kernels/k_pairs.135.sm75.cubin");
        let standalone = include_bytes!("../../../tests/kernels/k_tensor.135.sm75.cubin");
        let default = Policy::default();
        assert!(!default.companions);
        let with_comp = Policy {
            companions: true,
            ..default
        };
        let image = |policy: &Policy, bytes: &[u8]| policy.adapt_cubin(bytes).image.unwrap();
        assert_ne!(image(&default, cubin), image(&with_comp, cubin));
        // Standalone products (no pairs) are untouched whether companions are on or off.
        assert_eq!(image(&default, standalone), image(&with_comp, standalone));

        let unpaired = Policy {
            pairs: false,
            ..default
        };
        let unpaired_comp = Policy {
            companions: true,
            ..unpaired
        };
        assert_eq!(image(&unpaired, cubin), image(&unpaired_comp, cubin));
        assert_eq!(image(&unpaired, standalone), image(&unpaired_comp, standalone));

        let mut safe = default;
        safe.make_safe();
        let mut safe_comp = with_comp;
        let turned_off = safe_comp.make_safe();
        assert_eq!(turned_off, vec!["VF_COMPANIONS=1", "VF_PAIRS in machine code"]);
        assert!(!safe_comp.companions);
        assert_eq!(image(&safe, cubin), image(&safe_comp, cubin));
        assert_eq!(image(&safe, cubin), image(&unpaired, cubin));
    }

    #[test]
    fn safe_mode_gives_the_default_lowering() {
        // A bf16 product that the switches lower otherwise: in safe mode, the default's text.
        let groups = include_bytes!("../../../tests/kernels/ptx-cases/bf16-group.ptx");
        let row = include_bytes!("../../../tests/kernels/ptx-cases/bf16-row.ptx");
        let text = |policy: &Policy, ptx: &[u8]| policy.adapt_ptx(ptx).image.unwrap();
        let mut grouped = Policy::default();
        grouped.lowering.bf16_tc_groups = true;
        let mut tree = Policy::default();
        tree.lowering.bf16_row = vf_ptx::lower::Bf16Row::Tree;
        for (mut policy, ptx) in [(grouped, &groups[..]), (tree, &row[..])] {
            assert_ne!(text(&policy, ptx), text(&Policy::default(), ptx));
            policy.make_safe();
            assert_eq!(text(&policy, ptx), text(&Policy::default(), ptx));
        }
        // Machine code translated where the default takes the PTX: in safe mode, the PTX.
        let fat = fatbin::parse(ELF75_PTX75).unwrap();
        let mut first = Policy {
            translate: Translate::First,
            ..Policy::default()
        };
        assert!(matches!(
            first.adapt_fatbin(&fat).unwrap().outcome,
            Outcome::Translated { .. }
        ));
        first.make_safe();
        assert_eq!(
            first.adapt_fatbin(&fat).unwrap().image,
            Policy::default().adapt_fatbin(&fat).unwrap().image
        );
        // The program's sm_70 code, where force would translate its sm_75 code.
        let native = fatbin::parse(ELF70_ELF75_PTX70).unwrap();
        let mut force = Policy {
            translate: Translate::Force,
            ..Policy::default()
        };
        assert!(matches!(
            force.adapt_fatbin(&native).unwrap().outcome,
            Outcome::Translated { .. }
        ));
        force.make_safe();
        assert_eq!(force.adapt_fatbin(&native).unwrap().outcome, Outcome::Native);
    }
}
