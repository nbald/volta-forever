//! The machine code translator for Volta.
//!
//! A faithful port of `reference/translate.py`. It translates one cubin at a time,
//! function by function. Functions that call one another share their uniform register placement.
//! What it cannot do it refuses, and a refused function is left as a trap.

mod ampere;
mod attributes;
mod block_registers;
mod borrowed;
mod context;
mod cubin;
mod finish;
mod function;
mod instruction;
mod liveness;
mod matrix;
mod parallel;
mod parse;
mod return_addr;
mod schedule;
mod util;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use crate::decode::{Decoded, Kind};
use crate::generated::fields::*;
use crate::generated::special::*;

pub use self::util::find_relative_decoded;
pub use ampere::{Word as AmpereWord, described as ampere_class, word as ampere_word};
pub use cubin::{translate, translate_with_companions, translate_with_options};

// ---- Error type ----
#[derive(Debug)]
struct Refused(String);
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---- Item: one instruction of the result ----
struct Item {
    word: u128,
    origin: usize,
    what: &'static str,
    changed: bool,
    first: bool,
    target: Option<u64>,
    fix: Option<u64>,
    d: Option<Arc<Decoded>>,
    /// It reads, some time after it is issued, registers the translation gave it (`late`).
    late: bool,
    /// A scoreboard that the instruction after it waits for.
    then: Option<u32>,
    /// Other scoreboards the next instruction waits for (borrowed registers: a late result of
    /// the translation in a register the program may write next).
    also_mask: u8,
    /// An instruction of the program made to say when it has read (borrowed registers).
    says: bool,
    /// The borrowed registers that the sequence it is part of writes for the translation.
    borrows: Option<Box<RegSet>>,
    /// Its result is waited for with that of a later instruction of its sequence: the schedule
    /// treats it like a writer that names a scoreboard.
    covered: bool,
    /// Of code for sm_80: the distances that the program had are not those Volta needs.
    anew: bool,
    /// What it reads and writes besides what its operands say: the predicates that P2R saves and
    /// R2P gives back are in a mask.
    more_reads: Vec<(u8, u32, u8)>,
    more_writes: Vec<(u8, u32)>,
    /// An override for `what` in the map: when set, the map writes this instead of `what`.
    /// For level 3 items that include a uniform register number or a frame size.
    what_detail: Option<String>,
}

impl Item {
    fn new(word: u128, origin: usize, what: &'static str, changed: bool) -> Self {
        Self {
            word,
            origin,
            what,
            changed,
            first: false,
            target: None,
            fix: None,
            d: None,
            late: false,
            then: None,
            also_mask: 0,
            says: false,
            borrows: None,
            covered: false,
            anew: false,
            more_reads: Vec::new(),
            more_writes: Vec::new(),
            what_detail: None,
        }
    }
    /// The string to write in the map for this item.
    fn what_str(&self) -> &str {
        self.what_detail.as_deref().unwrap_or(self.what)
    }
}

// ---- Borrowed registers ----

/// A bit set of 256 bits (registers R0..R254; bit 255 unused since R255 is the zero register).
type RegSet = [u64; 4];

fn regset_empty() -> RegSet {
    [0u64; 4]
}

fn regset_set(s: &mut RegSet, r: u32) {
    s[(r / 64) as usize] |= 1u64 << (r % 64);
}

fn regset_test(s: &RegSet, r: u32) -> bool {
    s[(r / 64) as usize] & (1u64 << (r % 64)) != 0
}

fn regset_or(a: &RegSet, b: &RegSet) -> RegSet {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}

fn regset_andnot(a: &RegSet, b: &RegSet) -> RegSet {
    [a[0] & !b[0], a[1] & !b[1], a[2] & !b[2], a[3] & !b[3]]
}

/// Whether instruction `d` reads c[0x0][0x28] as an operand: the top of a thread's local memory.
fn stack_top(d: &Decoded) -> bool {
    // Look for any operand pair (Sx_bank=0, Sx_addr=0x28/4=10).
    for &(name_id, value) in &d.ops {
        let name = OP_NAMES[name_id as usize];
        if name.ends_with("_bank") && value == 0 {
            let addr_name = &name[..name.len() - 5];
            let addr_suffix = "_addr";
            let full = format!("{}{}", addr_name, addr_suffix);
            if let Some(&(_, addr_val)) = d.ops.iter().find(|&&(n, _)| OP_NAMES[n as usize] == full)
                && addr_val == 0x28 / 4
            {
                return true;
            }
        }
    }
    false
}

// ---- Constants ----
const OFFSET_LISTS: &[u8] = &[0x1c, 0x1d, 0x25, 0x27, 0x28, 0x2d, 0x31, 0x46];
const OFFSET_PAIRS: &[(u8, u8)] = &[(0x2e, 0), (0x44, 0), (0x55, 1)];
// (0x10, EIATTR_REQNTID: the block dimensions a kernel asks for, three words and no place in the
// code. Compilers of CUDA C never write it; Triton's cubins all do (`.reqntid` in its PTX), and
// until 2026-11-19 every one of its kernels was refused for it: vLLM could not start.)
const PLAIN_ATTRS: &[u8] = &[
    0x04, 0x05, 0x0a, 0x0f, 0x10, 0x11, 0x12, 0x17, 0x19, 0x1b, 0x1e, 0x23, 0x29, 0x2f, 0x33, 0x36, 0x37, 0x45, 0x4c,
    0x5f, 0x66,
];
/// What the translation puts around an instruction, which is not what the instruction became.
const AROUND: &[&str] = &[
    "wait",
    "save",
    "restore",
    "lane",
    "kept",
    "settled",
    "to pairs",
    "from pairs",
    "companion",
    "companion: lane",
];
/// The relocations that write the lower and the upper half of an address into the 32 bits of an
/// immediate (R_CUDA_ABS32_LO_32, R_CUDA_ABS32_HI_32).
const HALVES: &[u32] = &[56, 57];
/// The kinds whose results are written when they arrive (`docs/translator.md`, "Timing rules").
const LATE_WRITER_KINDS: &[Kind] = &[Kind::MioFast, Kind::MioSlow];
/// Fixed-time arithmetic whose result is written two cycles after the others'.
const DEEP_WRITER_MNEMONICS: &[&str] = &["DADD", "DMUL", "DFMA", "HMMA"];

/// The instruction writes its registers later than an instruction issued a cycle or two after it
/// would, and names no scoreboard that says when (`Borrowed`).
fn unannounced(d: &Decoded) -> bool {
    DEEP_WRITER_MNEMONICS.contains(&d.mnemonic)
        || (LATE_WRITER_KINDS.contains(&d.kind) && ((d.word >> WRITE_BARRIER) & 7) == 7)
}

/// The scoreboard on which a load added by the translation says its result is there.
const ARRIVES: u32 = 4;
/// The scoreboard of the instructions of a sequence that stands for a matrix instruction.
const SEQUENCE: u32 = 5;
/// The registers that the sequence of a matrix instruction computes into.
const MATRIX_SCRATCH: u32 = 8;
/// The registers of what a function with a product keeps of the lane (`kept`).
const KEPT: u32 = 4;
/// The registers that keep the elements of B a product gathered (`translate_matrix_product`).
const GATHERED: u32 = 4;
/// Turing's matrix load and product (`translate_matrix_load`, `translate_matrix_product`).
fn is_matrix(name: &str) -> bool {
    matches!(name, "ldsm__sImmOffset" | "ldsm__URsIR" | "hmma_x8_")
}

/// What ptxas 12.9.2 writes for `trap` on sm_70: `BPT.TRAP 0x1`, then a branch to itself.
const TRAP: u128 = 0x000fea0000300000_000000040000795c;
const STAY: u128 = 0x000fc0000383ffff_fffffff000007947;

/// An offset as the prototype prints it: `0x10`, `0x-10`.
fn hex(value: i64) -> String {
    if value < 0 {
        format!("0x-{:x}", value.unsigned_abs())
    } else {
        format!("0x{value:x}")
    }
}

#[inline]
fn stall_of(word: u128) -> u32 {
    ((word >> USCHED) & 15) as u32
}

#[inline]
/// `word` with a stall of `stall` cycles, keeping how it ends its group when it can.
///
/// The three bits above the wait mask go with the stall. In an instruction whose operands can bear
/// a mark of register reuse (`marks`), they are those marks when the stall ends a group (17 to 27
/// in the field) and name a batch with any other (NVIDIA's tables `TABLES_opex_*`). So an
/// instruction with marks that can no longer end its group loses them: they are hints, and
/// BATCH_START in their place is not one. And one that names a batch is not made to end its
/// group. (Until 2026-11-17 the marks were left where they were.)
fn with_stall(word: u128, stall: u32, marks: bool) -> u128 {
    let old = ((word >> USCHED) & 31) as u32;
    let mut word = word;
    let mut keeps = (old >> 4) != 0 && (1..=11).contains(&stall);
    if marks && word & REUSE != 0 {
        if (17..=27).contains(&old) {
            if !keeps {
                word &= !REUSE;
            }
        } else {
            keeps = false;
        }
    }
    let group = if keeps { 16u128 } else { 0 };
    (word & !(31u128 << USCHED)) | ((stall as u128 | group) << USCHED)
}

/// The decoded instruction has operands that can bear a mark of register reuse.
fn marked(d: &Option<Arc<Decoded>>) -> bool {
    d.as_ref().is_some_and(|d| {
        d.ops
            .iter()
            .any(|&(name, _)| OP_NAMES[name as usize].contains("reuse_src"))
    })
}

/// A listing entry: (function_name, new_offset, old_offset, what).
pub type MapEntry = (String, u64, u64, String);

/// A line of `MAP.lives`, for `reference/validate.py`, as `reference/translate.py` writes it
/// (`write_lives`): (function, two numbers, what, offsets). `free` or `saved`: a life of a
/// uniform predicate (the first number) and the ordinary predicate that stands for it there (the
/// second), saved around the life or not; `held`: a life of a uniform predicate held in a
/// register, `held-slot` in a word of the frame so many bytes under its top. With borrowed
/// registers: `home`, a uniform register in a register of the program along its life; `pool`, so
/// many registers from this one taken at an instruction; `room`, a register of the program put in
/// the frame for an instruction; `slot`, a uniform register so many bytes under the top of the
/// frame; `frame`, the bytes of the frame.
pub type LifeEntry = (String, i64, i64, &'static str, Vec<u64>);

/// What became of one function of a cubin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionReport {
    pub name: String,
    /// Why it was refused, when it was. Its code is then instructions that do nothing and an exit:
    /// it is never to run.
    pub refused: Option<String>,
    /// The instructions it had.
    pub before: usize,
    /// The instructions it has now: more where sequences stand for instructions sm_70 lacks, where
    /// uniform values are copied for the instructions that read them late, where waits were added.
    pub after: usize,
    /// Whether its translation computes tensor-core products two at a time on whole rows
    /// (`docs/pairs.md`); what a program is told after a fault in it (`VF_PAIRS=0` and `VF_SAFE=1`
    /// turn them off).
    pub paired: bool,
}

/// The result of translating a cubin.
pub struct TranslateResult {
    /// The cubin for sm_70.
    pub bytes: Vec<u8>,
    /// Its functions, in the order of their sections.
    pub report: Vec<FunctionReport>,
    pub map: Option<Vec<MapEntry>>,
    /// With the map: the lives of `MAP.lives`, function by function.
    pub lives: Option<Vec<LifeEntry>>,
}

/// Options for the translator.
#[derive(Debug, Clone)]
pub struct Options {
    /// Whether pairs of products on whole rows are translated as such (`VF_PAIRS=0`: they are
    /// not, and every product is translated by itself).
    pub pairs: bool,
    /// Whether the rows of a matrix load are taken to be aligned on 16 bytes (VF_ALIGNED_ROWS).
    pub aligned_rows: bool,
    /// Opt-in companion registers for whole-row matrix load addresses and per-group pair
    /// scheduling (`VF_COMPANIONS=1`; off by default, and unused when `pairs` is false).
    pub companions: bool,
    /// Whether the translation looks for room in the program's own registers before it takes new
    /// ones. 0: off. 1: homes and pool (levels 2, 1, 0). 2: and, for a kernel refused even so,
    /// everything at the instruction (level 3). 3: level 3 first, for tests. 4: by a kernel's
    /// register budget. 5: the same, with room made at level 3. 6: as at 0 wherever that
    /// translates, and for a kernel refused for want of registers, levels 1, 2, then 3 with room
    /// made. 7 (`BORROW_DEFAULT`): what the layer does by itself: what 6 does, and a function
    /// with tensor-core products may be translated at level 3 (the lane read at the product,
    /// its registers taken at the instruction). 8: level 3 first for every kernel, for tests.
    /// 9: try to preserve the source kernel’s register-limited maximum CTA size on GV100.
    pub borrow: u32,
}

/// The value of `Options::borrow` when nothing says otherwise (`VF_BORROW` unset or empty).
pub const BORROW_DEFAULT: u32 = 7;

/// `VF_BORROW` as given: a number; unset, empty or not a number: the default.
pub fn borrow_from(value: Option<&str>) -> u32 {
    value.and_then(|v| v.trim().parse().ok()).unwrap_or(BORROW_DEFAULT)
}

impl Default for Options {
    fn default() -> Self {
        Self {
            pairs: true,
            aligned_rows: true,
            companions: false,
            borrow: BORROW_DEFAULT,
        }
    }
}

impl Options {
    /// Read `VF_PAIRS`, `VF_ALIGNED_ROWS`, `VF_COMPANIONS` and `VF_BORROW` from the environment.
    pub fn from_environment() -> Self {
        let mut opts = Self::default();
        if let Ok(v) = std::env::var("VF_PAIRS") {
            opts.pairs = v != "0";
        }
        if let Ok(v) = std::env::var("VF_ALIGNED_ROWS") {
            opts.aligned_rows = v != "0";
        }
        if let Ok(v) = std::env::var("VF_COMPANIONS") {
            opts.companions = v == "1";
        }
        opts.borrow = borrow_from(std::env::var("VF_BORROW").ok().as_deref());
        opts
    }
}

struct TransResult {
    code: Vec<u8>,
    new_of: HashMap<u64, u64>,
    count: u32,
    items: Vec<Item>,
    /// Jump tables to move: (where in the constants of the function, the places as they are).
    relocate: Vec<(u32, Vec<u64>)>,
    /// Where each relocation on its code applies now (`follow`), in their order.
    moved: Vec<u64>,
    /// Level 3: the frame size in bytes added for uniform registers in local memory.
    frame_bytes: u32,
    /// Its lines of `MAP.lives`, without the name of the function.
    lives: Vec<(i64, i64, &'static str, Vec<u64>)>,
    /// Products made two at a time on whole rows (a plan with sites).
    paired: bool,
}

/// A relocation on a section of code: the section of relocations it is in and where in it, the
/// offset in the code it applies at, its kind.
struct Relocation {
    section: usize,
    at: usize,
    offset: u64,
    kind: u32,
}
