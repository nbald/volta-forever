//! Assembly of sm_70 instructions from a class name and operand values.
//!
//! Port of `machine.py`'s `Machine.assemble`: builds a 128-bit instruction word by filling in
//! constants, defaults, and given operand values. The generated ASM_TEMPLATES table holds the
//! pre-computed base word (all defaults/constants) and the operand setters.

use crate::field;
use crate::generated::assembly::*;

/// An operand value: either an integer or a string (enum symbol name).
#[derive(Clone)]
pub enum OpVal<'a> {
    Int(u32),
    Str(&'a str),
}

/// Find an assembly template by class name.
#[inline]
fn find_template(name: &str) -> Option<usize> {
    ASM_TEMPLATES.iter().position(|t| t.0 == name)
}

/// Assemble an sm_70 instruction word from a class name and operand values.
/// Operands not given take their default values (pre-baked in the template).
/// Enum operands can be given as strings (e.g., "LT", "U32") or integers.
/// Returns None if the class name is not in the assembly templates.
pub fn assemble(name: &str, operands: &[(&str, OpVal)]) -> Option<u128> {
    let idx = find_template(name)?;
    let t = &ASM_TEMPLATES[idx];
    let mut word = (t.2 as u128) << 64 | t.1 as u128;

    let setters = &ASM_SETTERS[t.3 as usize..(t.3 + t.4) as usize];
    let attr_setters = &ASM_ATTR_SETTERS[t.5 as usize..(t.5 + t.6) as usize];

    // Set operand fields
    for &(name_id, field_id, scale, default, enum_offset) in setters {
        let op_name = crate::generated::fields::OP_NAMES[name_id as usize];
        let given = operands.iter().find(|&&(n, _)| n == op_name);

        let value = if let Some((_, val)) = given {
            match val {
                OpVal::Int(v) => *v,
                OpVal::Str(s) => {
                    if enum_offset == 0xFFFFFFFF {
                        // Not an enum, try parsing as integer
                        s.parse().unwrap_or(default)
                    } else {
                        // Look up in enum entries
                        let mut idx = enum_offset as usize;
                        let mut found = default;
                        while ASM_ENUM_ENTRIES[idx].1 != 0xFFFFFFFF {
                            if ASM_ENUM_ENTRIES[idx].0 == *s {
                                found = ASM_ENUM_ENTRIES[idx].1;
                                break;
                            }
                            idx += 1;
                        }
                        found
                    }
                }
            }
        } else {
            default
        };

        let actual = if scale > 0 {
            debug_assert!(value % scale as u32 == 0);
            value / scale as u32
        } else {
            value
        };
        word = field::put(field_id, word, actual as u64);
    }

    // Set attribute fields
    for &(attr_name, field_id) in attr_setters {
        if let Some((_, val)) = operands.iter().find(|&&(n, _)| n == attr_name) {
            let v = match val {
                OpVal::Int(v) => *v,
                OpVal::Str(s) => s.parse().unwrap_or(0),
            };
            word = field::put(field_id, word, v as u64);
        }
    }

    // The result has a stall of 1 (from the base word which has USCHED bit 0 set)
    // and no scoreboard. Verify the base includes stall=1.
    Some(word)
}

/// Convenience: assemble with operands as (name, int_value) pairs.
pub fn assemble_int(name: &str, operands: &[(&str, u32)]) -> Option<u128> {
    let ops: Vec<(&str, OpVal)> = operands.iter().map(|&(n, v)| (n, OpVal::Int(v))).collect();
    assemble(name, &ops)
}

/// Assemble with mixed int and string operands. Each entry is (name, OpVal).
pub fn assemble_mixed(name: &str, operands: &[(&str, OpVal)]) -> Option<u128> {
    assemble(name, operands)
}
