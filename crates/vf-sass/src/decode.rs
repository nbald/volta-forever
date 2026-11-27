//! Decoder: finds the class of an instruction word and extracts its operands.
//!
//! The decoder uses the generated tables to identify which instruction class a 128-bit
//! word belongs to, then extracts its operands, computes what it reads and writes, and
//! determines its scheduling kind. This mirrors `md.py`'s Decoder and `machine.py`'s decode.

use std::sync::Arc;

use crate::fast::FastMap;
use crate::field;
use crate::generated::classes::*;
use crate::generated::fields::{OP_NAMES, OP_PG, OP_UPG};
use crate::generated::special::*;

/// The scheduling kind of an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    FXU,
    IMAD,
    FMAI,
    FMALITE,
    FMA64HEAVY,
    MioFast,
    MioSlow,
    BRU,
    HMMA,
    IMMA,
    UDP,
    OTHER,
}

impl Kind {
    pub fn from_id(id: u8) -> Self {
        match id {
            0 => Kind::FXU,
            1 => Kind::IMAD,
            2 => Kind::FMAI,
            3 => Kind::FMALITE,
            4 => Kind::FMA64HEAVY,
            5 => Kind::MioFast,
            6 => Kind::MioSlow,
            7 => Kind::BRU,
            8 => Kind::HMMA,
            9 => Kind::IMMA,
            10 => Kind::UDP,
            _ => Kind::OTHER,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Kind::FXU => "FXU",
            Kind::IMAD => "IMAD",
            Kind::FMAI => "FMAI",
            Kind::FMALITE => "FMALITE",
            Kind::FMA64HEAVY => "FMA64HEAVY",
            Kind::MioFast => "MIO_FAST",
            Kind::MioSlow => "MIO_SLOW",
            Kind::BRU => "BRU",
            Kind::HMMA => "HMMA",
            Kind::IMMA => "IMMA",
            Kind::UDP => "UDP",
            Kind::OTHER => "OTHER",
        }
    }

    /// Whether instructions of this kind read their operands some time after they are issued:
    /// memory accesses and conversions, arithmetic on doubles, matrix products. They are the kinds
    /// that name a scoreboard for having read, in native code.
    pub fn is_late(&self) -> bool {
        matches!(self, Kind::MioFast | Kind::MioSlow | Kind::FMALITE | Kind::HMMA)
    }
}

pub const LETTER_R: u8 = 0;
pub const LETTER_U: u8 = 1;
pub const LETTER_P: u8 = 2;
pub const LETTER_Q: u8 = 3;

/// A decoded instruction with its class, operands, and what it reads and writes.
#[derive(Clone)]
pub struct Decoded {
    pub word: u128,
    pub class_id: u16,
    pub kind: Kind,
    pub mnemonic: &'static str,
    pub name: &'static str,
    /// (name_id, value) pairs.
    pub ops: Vec<(u16, u32)>,
    /// (letter, register, role) where role 0=data, 1=guard.
    pub reads: Vec<(u8, u32, u8)>,
    /// (letter, register).
    pub writes: Vec<(u8, u32)>,
}

impl Decoded {
    #[inline]
    pub fn op(&self, name_id: u16) -> Option<u32> {
        self.ops.iter().find(|&&(n, _)| n == name_id).map(|&(_, v)| v)
    }

    #[inline]
    pub fn op_or(&self, name_id: u16, default: u32) -> u32 {
        self.op(name_id).unwrap_or(default)
    }

    #[inline]
    pub fn has_op(&self, name_id: u16) -> bool {
        self.ops.iter().any(|&(n, _)| n == name_id)
    }

    /// Look up an operand by its string name (the name from `OP_NAMES`).
    pub fn op_or_name(&self, name: &str, default: u32) -> u32 {
        for &(n, v) in &self.ops {
            if OP_NAMES[n as usize] == name {
                return v;
            }
        }
        default
    }

    /// Whether the instruction has an operand with the given string name.
    pub fn has_op_name(&self, name: &str) -> bool {
        self.ops.iter().any(|&(n, _)| OP_NAMES[n as usize] == name)
    }
}

/// How many registers an operand takes in an instruction with the operands `ops`: `span` is that
/// number, or 0x8000 + the table that gives it by the values of some of them (a 64-bit address, a
/// load of 4, 8 or 16 bytes).
pub fn span_of(span: u16, ops: &[(u16, u32)]) -> u32 {
    if span < 0x8000 {
        return span as u32;
    }
    let (keys, count, values) = SPAN_TABLES[(span - 0x8000) as usize];
    let mut index = 0usize;
    for &(name_id, bits) in &SPAN_KEYS[keys as usize..keys as usize + count as usize] {
        let value = ops.iter().find(|&&(n, _)| n == name_id).map_or(0, |&(_, v)| v as usize);
        index = (index << bits) | (value & ((1 << bits) - 1));
    }
    SPAN_VALUES[values as usize + index] as u32
}

/// Calls `f(first_register, register_count)` for each uniform register operand (`U`), skipping
/// `UZ` (register 63).
pub(crate) fn for_each_uniform_operand<F: FnMut(u32, u32)>(d: &Decoded, mut f: F) {
    let cls = &SM75_CLASSES[d.class_id as usize];
    let reads = &SM75_READS[cls.12 as usize..(cls.12 + cls.13) as usize];
    let writes = &SM75_WRITES[cls.14 as usize..(cls.14 + cls.15) as usize];
    for &(letter, name_id, span) in reads.iter().chain(writes) {
        if letter == LETTER_U
            && let Some(first) = d.op(name_id)
            && first != ZERO_U
        {
            f(first, span_of(span, &d.ops));
        }
    }
}

/// The uniform registers that an sm_75 instruction reads or writes: for each of its operands, the
/// first register and how many it takes as one value.
pub fn uniform_operands(d: &Decoded) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for_each_uniform_operand(d, |first, count| out.push((first, count)));
    out
}

pub fn zero_of(letter: u8) -> u32 {
    match letter {
        LETTER_R => ZERO_R,
        LETTER_U => ZERO_U,
        LETTER_P => ZERO_P,
        LETTER_Q => ZERO_Q,
        _ => 0,
    }
}

/// A type alias for the class tuple from the generated tables.
type ClassTuple = (
    &'static str,
    &'static str,
    bool,
    u8,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u64,
    u64,
);

pub struct Decoder {
    sm: u32,
    cache: FastMap<u128, Option<Arc<Decoded>>>,
    class_cache: FastMap<u128, Option<u16>>,
}

impl Decoder {
    pub fn new(sm: u32) -> Self {
        Self {
            sm,
            cache: FastMap::default(),
            class_cache: FastMap::default(),
        }
    }

    pub fn decode(&mut self, word: u128) -> Option<Decoded> {
        self.decode_arc(word).map(|d| (*d).clone())
    }

    pub(crate) fn decode_arc(&mut self, word: u128) -> Option<Arc<Decoded>> {
        if let Some(cached) = self.cache.get(&word) {
            return cached.clone();
        }
        let is_75 = self.sm == 75;
        let result = self
            .match_class_cached(word, is_75)
            .map(|class_id| Arc::new(self.build_decoded(word, is_75, class_id)));
        self.cache.insert(word, result.clone());
        result
    }

    pub(crate) fn is_valid(&mut self, word: u128) -> bool {
        if let Some(cached) = self.cache.get(&word) {
            return cached.is_some();
        }
        self.match_class_cached(word, self.sm == 75).is_some()
    }

    #[inline]
    fn match_class_cached(&mut self, word: u128, is_75: bool) -> Option<u16> {
        // By the whole word: the wait mask is a constant field of some classes (the four `ttu*`
        // of sm_75 take no wait), so what a word decodes to may depend on it.
        if let Some(&cached) = self.class_cache.get(&word) {
            return cached;
        }
        let class_id = self.match_class(word, is_75);
        self.class_cache.insert(word, class_id);
        class_id
    }

    fn match_class(&self, word: u128, is_75: bool) -> Option<u16> {
        let classes: &[ClassTuple] = if is_75 { &SM75_CLASSES } else { &VF_CLASSES };
        let const_checks = if is_75 {
            &SM75_CONST_CHECKS[..]
        } else {
            &VF_CONST_CHECKS[..]
        };
        let domain_checks = if is_75 {
            &SM75_DOMAIN_CHECKS[..]
        } else {
            &VF_DOMAIN_CHECKS[..]
        };
        let domain_values = if is_75 {
            &SM75_DOMAIN_VALUES[..]
        } else {
            &VF_DOMAIN_VALUES[..]
        };
        let domain_offsets = if is_75 {
            &SM75_DOMAIN_OFFSETS[..]
        } else {
            &VF_DOMAIN_OFFSETS[..]
        };

        let mut matched: Option<u16> = None;
        let mut match_count: u32 = 0;

        // Try opcode fields; sm_75 has fields 0 (13-bit) and 34 (12-bit), sm_70 has field 34 only
        if is_75 {
            self.check_opcode(
                word,
                0,
                &SM75_OPCODE_0_OFFSETS,
                &SM75_OPCODE_0_CANDIDATES,
                classes,
                const_checks,
                domain_checks,
                domain_values,
                domain_offsets,
                false,
                &mut matched,
                &mut match_count,
            );
            if match_count < 2 {
                self.check_opcode(
                    word,
                    34,
                    &SM75_OPCODE_34_OFFSETS,
                    &SM75_OPCODE_34_CANDIDATES,
                    classes,
                    const_checks,
                    domain_checks,
                    domain_values,
                    domain_offsets,
                    false,
                    &mut matched,
                    &mut match_count,
                );
            }
        } else {
            self.check_opcode(
                word,
                34,
                &VF_OPCODE_34_OFFSETS,
                &VF_OPCODE_34_CANDIDATES,
                classes,
                const_checks,
                domain_checks,
                domain_values,
                domain_offsets,
                true,
                &mut matched,
                &mut match_count,
            );
        }

        if match_count != 1 {
            return None;
        }
        matched
    }

    fn build_decoded(&self, word: u128, is_75: bool, class_id: u16) -> Decoded {
        let classes: &[ClassTuple] = if is_75 { &SM75_CLASSES } else { &VF_CLASSES };
        let cls = &classes[class_id as usize];
        let operands = if is_75 { &SM75_OPERANDS[..] } else { &VF_OPERANDS[..] };
        let reads_desc = if is_75 { &SM75_READS[..] } else { &VF_READS[..] };
        let writes_desc = if is_75 { &SM75_WRITES[..] } else { &VF_WRITES[..] };

        let ops = self.extract_ops(word, &operands[cls.10 as usize..(cls.10 + cls.11) as usize]);
        let reads = self.compute_reads(&ops, &reads_desc[cls.12 as usize..(cls.12 + cls.13) as usize]);
        let writes = self.compute_writes(&ops, &writes_desc[cls.14 as usize..(cls.14 + cls.15) as usize]);

        Decoded {
            word,
            class_id,
            kind: Kind::from_id(cls.3),
            mnemonic: cls.1,
            name: cls.0,
            ops,
            reads,
            writes,
        }
    }

    /// The mnemonics of the classes of sm_75 whose opcode `word` has, whatever its other fields:
    /// what instruction a word that `decode` does not take is, for a later architecture that kept
    /// the opcodes of sm_75 (sm_80 for its stores, atomics and reductions).
    pub fn opcode_mnemonics_sm75(word: u128) -> Vec<&'static str> {
        let mut found = Vec::new();
        for (fid, offsets, candidates) in [
            (0, &SM75_OPCODE_0_OFFSETS[..], &SM75_OPCODE_0_CANDIDATES[..]),
            (34, &SM75_OPCODE_34_OFFSETS[..], &SM75_OPCODE_34_CANDIDATES[..]),
        ] {
            let value = field::get(fid, word) as usize;
            let Some(&(off, cnt)) = offsets.get(value) else {
                continue;
            };
            for &class_id in &candidates[off as usize..(off as usize + cnt as usize)] {
                let class = &SM75_CLASSES[class_id as usize];
                if !class.2 && !found.contains(&class.1) {
                    found.push(class.1);
                }
            }
        }
        found
    }

    #[allow(clippy::too_many_arguments)]
    fn check_opcode(
        &self,
        word: u128,
        opcode_fid: u16,
        offsets: &[(u32, u16)],
        candidates: &[u16],
        classes: &[ClassTuple],
        const_checks: &[(u16, u32)],
        domain_checks: &[(u16, u16)],
        domain_values: &[u32],
        domain_offsets: &[(u32, u16)],
        check_unused: bool,
        matched: &mut Option<u16>,
        match_count: &mut u32,
    ) {
        let opcode_val = field::get(opcode_fid, word) as usize;
        if opcode_val >= offsets.len() {
            return;
        }
        let (off, cnt) = offsets[opcode_val];
        for &class_id in &candidates[off as usize..(off as usize + cnt as usize)] {
            let cls = &classes[class_id as usize];
            // Skip alternates
            if cls.2 {
                continue;
            }
            // Check constant fields
            let cc = &const_checks[cls.6 as usize..(cls.6 + cls.7) as usize];
            let mut ok = true;
            for &(fid, expected) in cc {
                if field::get(fid, word) != expected as u64 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            // Check domain fields
            let dc = &domain_checks[cls.8 as usize..(cls.8 + cls.9) as usize];
            for &(fid, domain_id) in dc {
                let val = field::get(fid, word);
                let (doff, dlen) = domain_offsets[domain_id as usize];
                let domain = &domain_values[doff as usize..(doff as usize + dlen as usize)];
                if !domain.iter().any(|&d| d as u64 == val) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            // For sm_70: unused bits must be zero
            if check_unused {
                let unused_mask = (cls.17 as u128) << 64 | cls.16 as u128;
                if word & unused_mask != 0 {
                    continue;
                }
            }
            if *match_count == 0 {
                *matched = Some(class_id);
            }
            *match_count += 1;
            if *match_count > 1 {
                return;
            }
        }
    }

    fn extract_ops(&self, word: u128, operands: &[(u16, u16, u16, u8)]) -> Vec<(u16, u32)> {
        let mut ops = Vec::with_capacity(operands.len());
        for &(name_id, field_id, scale, op_kind) in operands {
            let raw = field::get(field_id, word);
            let value = if op_kind == 2 && scale > 0 {
                raw * scale as u64
            } else {
                raw
            };
            ops.push((name_id, value as u32));
        }
        ops
    }

    fn compute_reads(&self, ops: &[(u16, u32)], reads_desc: &[(u8, u16, u16)]) -> Vec<(u8, u32, u8)> {
        let mut reads = Vec::with_capacity(reads_desc.len());
        for &(letter, name_id, span) in reads_desc {
            if let Some(&(_, val)) = ops.iter().find(|&&(n, _)| n == name_id)
                && val != zero_of(letter)
            {
                let role = if name_id == OP_PG || name_id == OP_UPG {
                    1u8
                } else {
                    0u8
                };
                for i in 0..span_of(span, ops) {
                    reads.push((letter, val + i, role));
                }
            }
        }
        reads
    }

    fn compute_writes(&self, ops: &[(u16, u32)], writes_desc: &[(u8, u16, u16)]) -> Vec<(u8, u32)> {
        let mut writes = Vec::with_capacity(writes_desc.len());
        for &(letter, name_id, span) in writes_desc {
            if let Some(&(_, val)) = ops.iter().find(|&&(n, _)| n == name_id)
                && val != zero_of(letter)
            {
                for i in 0..span_of(span, ops) {
                    writes.push((letter, val + i));
                }
            }
        }
        writes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word of a class of sm_75: its opcode, its constant fields, the first value of each field of
    /// a domain, and nothing else.
    fn word_of(name: &str) -> (u16, u128) {
        let class_id = SM75_CLASSES
            .iter()
            .position(|c| c.0 == name)
            .unwrap_or_else(|| panic!("{name}")) as u16;
        let cls = &SM75_CLASSES[class_id as usize];
        let mut word = 0u128;
        let opcode = [
            (0u16, &SM75_OPCODE_0_OFFSETS[..], &SM75_OPCODE_0_CANDIDATES[..]),
            (34, &SM75_OPCODE_34_OFFSETS[..], &SM75_OPCODE_34_CANDIDATES[..]),
        ]
        .into_iter()
        .find_map(|(fid, offsets, candidates)| {
            offsets
                .iter()
                .position(|&(off, cnt)| candidates[off as usize..off as usize + cnt as usize].contains(&class_id))
                .map(|v| (fid, v))
        });
        let (fid, value) = opcode.unwrap_or_else(|| panic!("no opcode for {name}"));
        word = field::put(fid, word, value as u64);
        for &(fid, expected) in &SM75_CONST_CHECKS[cls.6 as usize..(cls.6 + cls.7) as usize] {
            word = field::put(fid, word, expected as u64);
        }
        for &(fid, domain_id) in &SM75_DOMAIN_CHECKS[cls.8 as usize..(cls.8 + cls.9) as usize] {
            let (offset, _) = SM75_DOMAIN_OFFSETS[domain_id as usize];
            word = field::put(fid, word, SM75_DOMAIN_VALUES[offset as usize] as u64);
        }
        (class_id, word)
    }

    /// Four classes of sm_75 (the tree-traversal unit's `ttugo_`, `ttuld_`, `ttuclose_`, `ttust_`)
    /// have the wait mask among their constant fields: such a word with a wait is not theirs. What
    /// a word decodes to does not depend on the words the decoder saw before it (one decoder a
    /// worker with `VF_THREADS`): the same word with a wait, before and after the word without one.
    #[test]
    fn a_word_decodes_the_same_whatever_the_decoder_saw_before() {
        for name in ["ttugo_", "ttuld_", "ttuclose_", "ttust_"] {
            let (class_id, word) = word_of(name);
            // (`ttuclose_` is an alternate form, which no word decodes to.)
            if !SM75_CLASSES[class_id as usize].2 {
                assert_eq!(
                    Decoder::new(75).decode(word).map(|d| d.class_id),
                    Some(class_id),
                    "{name}"
                );
            }
            let waits = word | (1u128 << WAIT);
            let alone = Decoder::new(75).decode(waits).map(|d| d.name);
            assert_ne!(alone, Some(name), "{name} with a wait");
            let mut after = Decoder::new(75);
            after.decode(word);
            assert_eq!(
                after.decode(waits).map(|d| d.name),
                alone,
                "{name} with a wait, after it without"
            );
            let mut checked = Decoder::new(75);
            checked.is_valid(word);
            assert_eq!(checked.is_valid(waits), alone.is_some(), "{name}: is_valid");
        }
    }
}
