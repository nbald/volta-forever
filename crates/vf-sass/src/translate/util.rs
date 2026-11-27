//! Utility functions for operand manipulation: find, read and write the operands
//! of a decoded instruction by name, class or field.

use crate::decode::Decoded;
use crate::field;
use crate::generated::classes::{SM75_CLASSES, SM75_OPERANDS, VF_CLASSES, VF_OPERANDS};
use crate::generated::fields::*;
use crate::generated::special::*;

use super::{Item, Refused};

// ---- Utility functions ----

/// (field_id, scale) of the operand of a decoded instruction that is a distance in the code, or None.
pub fn find_relative_decoded(d: &Decoded, sm: u32) -> Option<(u16, u32)> {
    find_relative(d, sm)
}

#[inline]
pub(super) fn find_relative(d: &Decoded, sm: u32) -> Option<(u16, u32)> {
    let cid = d.class_id as usize;
    let (name, op_types, op_type_offsets, op_start, op_len, operands) = if sm == 75 {
        let cls = &SM75_CLASSES[cid];
        (
            cls.0,
            &SM75_OP_TYPES[..],
            &SM75_OP_TYPE_OFFSETS[..],
            cls.10 as usize,
            cls.11 as usize,
            &SM75_OPERANDS[..],
        )
    } else {
        let cls = &VF_CLASSES[cid];
        (
            cls.0,
            &VF_OP_TYPES[..],
            &VF_OP_TYPE_OFFSETS[..],
            cls.10 as usize,
            cls.11 as usize,
            &VF_OPERANDS[..],
        )
    };
    let (off, len) = op_type_offsets[cid];
    let types = &op_types[off as usize..(off as usize + len as usize)];
    // A branch through a register goes to the instruction after it, plus the register, plus a
    // number: minus where that next instruction is, a distance in the code too (to offset 0).
    let rsimm_nid = if name == "brx_" {
        OP_RA_OFFSET
    } else {
        types.iter().find(|&&(_, tc)| tc == 5).map(|&(nid, _)| nid)?
    };
    let ops = &operands[op_start..(op_start + op_len)];
    for &(nid, fid, scale, kind) in ops {
        if nid == rsimm_nid && kind == 2 && scale > 0 {
            return Some((fid, scale as u32));
        }
    }
    None
}

pub(super) fn put_operand(d: &Decoded, name_id: u16, value: u64, sm: u32) -> Result<u128, Refused> {
    let classes = if sm == 75 {
        &crate::generated::classes::SM75_CLASSES[..]
    } else {
        &crate::generated::classes::VF_CLASSES[..]
    };
    let operands = if sm == 75 {
        &crate::generated::classes::SM75_OPERANDS[..]
    } else {
        &crate::generated::classes::VF_OPERANDS[..]
    };
    let cls = &classes[d.class_id as usize];
    let ops = &operands[cls.10 as usize..(cls.10 + cls.11) as usize];
    for &(nid, fid, _, _) in ops {
        if nid == name_id {
            return Ok(field::put(fid, d.word, value));
        }
    }
    Err(Refused(format!(
        "{} has no operand {}",
        d.name, OP_NAMES[name_id as usize]
    )))
}

pub(super) fn put_operand_on_word(d: &Decoded, word: u128, name_id: u16, value: u64, sm: u32) -> Result<u128, Refused> {
    let classes = if sm == 75 {
        &crate::generated::classes::SM75_CLASSES[..]
    } else {
        &crate::generated::classes::VF_CLASSES[..]
    };
    let operands = if sm == 75 {
        &crate::generated::classes::SM75_OPERANDS[..]
    } else {
        &crate::generated::classes::VF_OPERANDS[..]
    };
    let cls = &classes[d.class_id as usize];
    let ops = &operands[cls.10 as usize..(cls.10 + cls.11) as usize];
    for &(nid, fid, _, _) in ops {
        if nid == name_id {
            return Ok(field::put(fid, word, value));
        }
    }
    Err(Refused(format!(
        "{} has no operand {}",
        d.name, OP_NAMES[name_id as usize]
    )))
}

pub(super) fn put_operand_by_class(
    class_name: &str,
    word: u128,
    operand: &str,
    value: u64,
    _sm70: bool,
) -> Result<u128, Refused> {
    let classes = &crate::generated::classes::VF_CLASSES;
    let operands = &crate::generated::classes::VF_OPERANDS;
    let nid = OP_NAMES
        .iter()
        .position(|&n| n == operand)
        .ok_or_else(|| Refused(format!("unknown operand {}", operand)))? as u16;
    let idx = classes
        .iter()
        .position(|c| c.0 == class_name)
        .ok_or_else(|| Refused(format!("class {} not found", class_name)))?;
    let cls = &classes[idx];
    let ops = &operands[cls.10 as usize..(cls.10 + cls.11) as usize];
    for &(n, fid, _, _) in ops {
        if n == nid {
            return Ok(field::put(fid, word, value));
        }
    }
    Err(Refused(format!("{} has no operand {}", class_name, operand)))
}

/// The item for `word`, of sm_70 class `class_name`: an instruction that reads its operands some
/// time after it is issued (a memory access, a conversion), and that the translation gave a
/// register to read: the stand-in of a uniform register, or the registers of the pool.
///
/// Turing has a uniform value when the instruction is issued, and the program may change the
/// uniform register right after it; and the pool is written again by the next sequence that needs
/// it. Native code keeps what such an instruction reads until it has read: the instruction says so
/// on a scoreboard, and what writes one of its operands next waits for that. Here the instruction
/// says so, and the very next instruction waits for it: by then nothing it reads can change,
/// whatever comes after.
pub(super) fn late(class_name: &str, word: u128, i: usize, what: &'static str) -> Result<Item, Refused> {
    let mut item = Item::new(
        read_announced(class_name, unmarked(class_name, word, what), i)?,
        i,
        what,
        true,
    );
    item.late = true;
    Ok(item)
}

/// `word`, of sm_70 class `class_name` and of kind `what`, without its marks of register reuse.
/// In an instruction of the program (kept as it is, or on ordinary registers) the three bits are
/// marks only when its operands can bear them and its stall ends a group (`with_stall`);
/// otherwise they name a batch, which is no hint and stays. What the translation adds names no
/// batch.
pub(super) fn unmarked(class_name: &str, word: u128, what: &str) -> u128 {
    if matches!(what, "same" | "ordinary registers") {
        let classes = &crate::generated::classes::VF_CLASSES;
        let bears_marks = classes.iter().find(|class| class.0 == class_name).is_some_and(|class| {
            crate::generated::classes::VF_OPERANDS[class.10 as usize..(class.10 + class.11) as usize]
                .iter()
                .any(|&(name, ..)| OP_NAMES[name as usize].contains("reuse_src"))
        });
        if !(bears_marks && (17..=27).contains(&((word >> USCHED) & 31))) {
            return word;
        }
    }
    word & !REUSE
}

pub(super) fn read_announced(class_name: &str, word: u128, i: usize) -> Result<u128, Refused> {
    // Find the src_rel_sb function field in the sm_70 class
    // Look for the src_rel_sb operand
    let classes = &crate::generated::classes::VF_CLASSES;
    let class_id = classes
        .iter()
        .position(|c| c.0 == class_name)
        .ok_or_else(|| Refused(format!("{} not found", class_name)))? as u16;
    let cls = &classes[class_id as usize];
    let operands = &crate::generated::classes::VF_OPERANDS[cls.10 as usize..(cls.10 + cls.11) as usize];
    // Find src_rel_sb operand
    for &(nid, fid, _, _) in operands {
        if nid == OP_SRC_REL_SB {
            let current = field::get(fid, word);
            if current != 7 {
                return Ok(word);
            } // already has a scoreboard
            let write_sb = ((word >> WRITE_BARRIER) & 7) as u32;
            let sb = if write_sb != 5 { 5 } else { 4 };
            return Ok(field::put(fid, word, sb));
        }
    }
    Err(Refused(format!(
        "{} at 0x{:x} cannot say when it has read its operands",
        class_name,
        16 * i
    )))
}
