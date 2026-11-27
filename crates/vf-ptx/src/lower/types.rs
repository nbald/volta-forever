//! Internal types, register declarations, and small utility functions used throughout the
//! lowering: operand forms, element types, the register declaration tracker, address and
//! operand parsing helpers, and integer and constant recognition.

use std::collections::HashMap;

use crate::scan;

/// What is remembered about operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Form {
    /// Both registers of an FP16 A operand with their halves swapped (2 registers).
    Swapped,
    /// The two exact FP32 values used to make a packed BF16 word, low lane first.
    Bf16Lanes,
    /// The B operands of the two Volta instructions, made from an FP16 B register (4 registers).
    Columns,
    /// f32 accumulators in Volta layout (8 registers), for the Turing registers they were last
    /// converted to.
    Wide,
    /// A operand loaded at a `ldmatrix` in the chain's single-instruction layout (2 registers):
    /// the half of k this thread's octet handles, from the row it needs.
    ChainA,
    /// B operand loaded likewise (2 registers).
    ChainB,
    /// f16 accumulators in Volta layout (4 registers), likewise.
    Half,
    /// The row of A of a thread's group, for the integer mma (4 registers).
    Row,
    /// The two columns of B a thread needs, for the integer mma (8 registers).
    ColumnPair,
    /// The row of A a thread provides when an FP16 product is split in the halves of k: k 0..3 in
    /// registers 0 and 1, k 4..7 in registers 2 and 3.
    Rows,
    /// The column of B a thread provides in that case, likewise.
    Cols,
    /// For the products of a pair (`docs/pairs.md`): the row of A a lane gives to both, from the two
    /// A registers (4 registers: k 0..3, then k 4..7).
    QRows,
    /// For the products of a pair: the column of B a lane gives, from x's B register where `h` is
    /// clear and from y's where it is set (sources: x's register, y's register; 4 registers).
    QCols,
    /// The four `e4m3` bytes of a register as halves: bytes 0 and 1, then bytes 2 and 3 (2 registers).
    E4m3,
    /// The four `e4m3` bytes divided by 256 as halves (for deferred UE8M0 scale folding).
    E4m3Unscaled,
    /// The four `e5m2` bytes of a register, likewise.
    E5m2,
    /// The two bf16 elements of a register as f32 (2 registers).
    Bf16,
    /// A scalar BF16 register holding the exact negative zero bit pattern.
    Bf16NegativeZero,
    /// A tf32 operand as a product takes it: without the 13 low bits of its mantissa (1 register).
    Tf32,
}

/// The FP8 formats whose values a half holds exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fp8 {
    /// Sign, 4 bits of exponent (bias 7), 3 of mantissa; no infinity, and `0x7f` is the NaN.
    E4m3,
    /// Sign, 5 bits of exponent (bias 15), 2 of mantissa: a half without its low byte.
    E5m2,
}

impl Fp8 {
    pub(super) fn named(name: &str) -> Option<Fp8> {
        match name {
            "e4m3" | "e4m3x2" => Some(Fp8::E4m3),
            "e5m2" | "e5m2x2" => Some(Fp8::E5m2),
            _ => None,
        }
    }

    pub(super) fn form(self) -> Form {
        match self {
            Fp8::E4m3 => Form::E4m3,
            Fp8::E5m2 => Form::E5m2,
        }
    }
}

/// A block-scaled MMA product: the kind, operand types, scale type and block count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScaledKind {
    /// `kind::mxf4`, m16n8k64, E2M1, UE8M0 scale, `scale_vec::2X` (2 blocks of 32 k-elements).
    Mxf4,
    /// `kind::mxf4nvf4`, m16n8k64, E2M1, UE4M3 scale, `scale_vec::4X` (4 blocks of 16): NVFP4.
    Nvf4Ue4m3x4,
    /// `kind::mxf8f6f4`, m16n8k32, E4M3/E5M2 operands, UE8M0, `scale_vec::1X` (1 block of 32).
    Mxf8f6f4(Fp8, Fp8),
}

impl ScaledKind {
    /// Number of k-blocks the scale divides the K dimension into.
    pub(super) fn blocks(self) -> usize {
        match self {
            ScaledKind::Mxf4 => 2,
            ScaledKind::Nvf4Ue4m3x4 => 4,
            ScaledKind::Mxf8f6f4(_, _) => 1,
        }
    }

    /// Whether the scale type is UE4M3 (true) or UE8M0 (false).
    pub(super) fn ue4m3_scale(self) -> bool {
        matches!(self, ScaledKind::Nvf4Ue4m3x4)
    }
}

/// The element types of the products that each thread computes itself: a half does not hold them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Element {
    /// Two per register, each the upper half of an f32.
    Bf16,
    /// One per register: an f32 that the program rounded to 10 bits of mantissa.
    Tf32,
    /// One per 64-bit register.
    F64,
}

pub(super) type Key = (Form, Vec<String>);

/// Bit width of a register type (`.b32`, `.f16x2`, `.pred`...).
pub(super) fn type_width(name: &str) -> Option<u32> {
    match name {
        ".pred" => Some(1),
        ".f16x2" | ".bf16x2" => Some(32),
        _ => name
            .get(2..)?
            .parse()
            .ok()
            .filter(|_| matches!(name.as_bytes().get(1), Some(b'b' | b'u' | b's' | b'f'))),
    }
}

/// The registers declared by `.reg` statements, with their width.
#[derive(Debug, Default)]
pub(super) struct Declarations {
    pub(super) single: HashMap<String, u32>,
    /// `%name<n>` declares `%name0`..: (prefix, n, width).
    pub(super) families: Vec<(String, usize, u32)>,
}

impl Declarations {
    /// Record the declaration `.reg <operands>;` and return the names it declares (the prefix, for a
    /// family).
    pub(super) fn declare(&mut self, operands: &str) -> Vec<(String, bool)> {
        let operands = operands.trim();
        let (kind, names) = operands.split_once(char::is_whitespace).unwrap_or((operands, ""));
        let width = type_width(kind);
        let mut declared = Vec::new();
        for part in names.split(',') {
            let part = part.trim();
            let (name, count) = match part.split_once('<') {
                Some((name, count)) => (name.trim(), count.trim_end_matches('>').trim().parse::<usize>().ok()),
                None => (part, None),
            };
            if name.is_empty() {
                continue;
            }
            match (count, width) {
                (Some(count), Some(width)) => self.families.push((name.to_string(), count, width)),
                (None, Some(width)) => {
                    self.single.insert(name.to_string(), width);
                }
                _ => {}
            }
            declared.push((name.to_string(), part.contains('<')));
        }
        declared
    }

    /// The previous width of each single name that `operands` will declare: (name, previous width or
    /// None). Call this before `declare` to save what a scope close must restore.
    pub(super) fn save_singles(&self, operands: &str) -> Vec<(String, Option<u32>)> {
        let operands = operands.trim();
        let (_, names) = operands.split_once(char::is_whitespace).unwrap_or((operands, ""));
        names
            .split(',')
            .filter_map(|part| {
                let part = part.trim();
                if part.contains('<') || part.is_empty() {
                    return None;
                }
                Some((part.to_string(), self.single.get(part).copied()))
            })
            .collect()
    }

    /// The families from `from` on were declared in an inner block: a single name declared
    /// outside that one of them covers (`%r1` under `%r<4>`) is hidden until the block closes.
    /// Takes those names out and returns them with their widths, for the scope to put back.
    pub(super) fn hide_singles(&mut self, from: usize) -> Vec<(String, Option<u32>)> {
        let covered: Vec<String> = self
            .single
            .keys()
            .filter(|name| {
                self.families[from..].iter().any(|(prefix, count, _)| {
                    name.strip_prefix(prefix.as_str())
                        .and_then(|index| index.parse::<usize>().ok())
                        .is_some_and(|index| index < *count)
                })
            })
            .cloned()
            .collect();
        covered
            .into_iter()
            .map(|name| {
                let width = self.single.remove(&name);
                (name, width)
            })
            .collect()
    }

    pub(super) fn width(&self, name: &str) -> Option<u32> {
        if let Some(width) = self.single.get(name) {
            return Some(*width);
        }
        self.families.iter().rev().find_map(|(prefix, count, width)| {
            let index: usize = name.strip_prefix(prefix.as_str())?.parse().ok()?;
            (index < *count).then_some(*width)
        })
    }
}

/// A lexical scope in a `{ }` block: the names it declared and what must be restored when it closes.
#[derive(Debug, Default)]
pub(super) struct ScopeFrame {
    /// Names declared in this scope, for `redeclared()`.
    pub(super) names: Vec<(String, bool)>,
    /// For each single whose name was redeclared here: (name, previous width or None).
    pub(super) previous: Vec<(String, Option<u32>)>,
    /// Length of `Declarations.families` when this scope opened.
    pub(super) families_len: usize,
}

/// The address operand of `ldmatrix`: `[register]` or `[register+offset]`.
pub(super) fn address(operand: &str) -> Option<(&str, Option<&str>)> {
    let inner = operand.strip_prefix('[')?.strip_suffix(']')?.trim();
    let (base, offset) = match inner.split_once('+') {
        Some((base, offset)) => (base.trim(), Some(offset.trim())),
        None => (inner, None),
    };
    let numeric = |text: &str| !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    (scan::register(base).is_some() && offset.is_none_or(numeric)).then_some((base, offset))
}

/// The text of a statement's first operand: what the statement writes, when it writes registers.
pub(super) fn first_operand(operands: &str) -> &str {
    let mut depth = 0usize;
    for (at, byte) in operands.bytes().enumerate() {
        match byte {
            b'{' | b'[' | b'(' => depth += 1,
            b'}' | b']' | b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => return &operands[..at],
            _ => {}
        }
    }
    operands
}

/// An address operand `[register]` or `[register+offset]`, the offset a PTX integer literal.
pub(super) fn place_of(operand: &str) -> Option<(&str, u32)> {
    let inner = operand.strip_prefix('[')?.strip_suffix(']')?.trim();
    let (base, offset) = match inner.split_once('+') {
        Some((base, offset)) => (base.trim(), integer(offset.trim())?),
        None => (inner, 0),
    };
    scan::register(base).map(|base| (base, offset))
}

/// The operand is the constant with no bit set, as an integer or as a float.
pub(super) fn is_zero(operand: &str) -> bool {
    integer(operand) == Some(0) || operand.eq_ignore_ascii_case("0f00000000")
}

/// The operand is the constant with all 32 bits set.
pub(super) fn all_ones(operand: &str) -> bool {
    operand == "-1" || integer(operand) == Some(u32::MAX)
}

/// A nonnegative 32-bit PTX integer literal, including its optional unsigned suffix.
pub(super) fn integer(text: &str) -> Option<u32> {
    let text = text.strip_prefix('+').unwrap_or(text);
    let text = text.strip_suffix('U').unwrap_or(text);
    if let Some(digits) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(digits, 16).ok()
    } else if let Some(digits) = text.strip_prefix("0b").or_else(|| text.strip_prefix("0B")) {
        u32::from_str_radix(digits, 2).ok()
    } else if text.starts_with('0') {
        u32::from_str_radix(text, 8).ok()
    } else {
        text.parse().ok()
    }
}
