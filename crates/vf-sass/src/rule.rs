//! Rule application: transforms sm_75 instruction words to sm_70 using pre-computed rules.
//!
//! Each rule maps an sm_75 class to an sm_70 class. It carries a set of field-to-field moves
//! (some with register/predicate conversion), a fixed base word for the sm_70 side, and
//! constraints on which values are valid. This is a port of `remap.py`'s `Rule.apply`.

use crate::field;
use crate::generated::rules::*;
use crate::generated::special::SCHEDULING;

/// Find a pre-computed rule by sm_75 class name, sm_70 class name, and drop key.
#[inline]
pub fn find_rule(sm75_name: &str, sm70_name: &str, drop_key: &str) -> Option<usize> {
    RULES
        .iter()
        .position(|r| r.0 == sm75_name && r.1 == sm70_name && r.10 == drop_key)
}

/// Find the rule_of target for an sm_75 class (by class ID).
#[inline]
pub fn rule_of_target(sm75_class_id: u16) -> Option<&'static str> {
    RULE_OF
        .iter()
        .find(|&&(id, _)| id == sm75_class_id)
        .map(|&(_, target)| target)
}

/// What applying a rule to an instruction word gave.
pub enum Applied {
    /// The sm_70 word.
    Word(u128),
    /// A field holds a value that sm_70 does not have.
    NotAllowed,
    /// A uniform register that no ordinary register stands for.
    NoRegister(u32),
    /// A uniform predicate that no ordinary predicate stands for.
    NoPredicate(u32),
}

/// Apply rule at index `rule_idx` to an sm_75 instruction word.
/// `register`: the ordinary register that stands for a uniform register, if one does.
/// `predicate`: the same for a uniform predicate.
pub fn apply(
    rule_idx: usize,
    word: u128,
    register: &dyn Fn(u32) -> Option<u32>,
    predicate: &dyn Fn(u32) -> Option<u32>,
) -> Applied {
    let r = &RULES[rule_idx];
    let fixed = (r.3 as u128) << 64 | r.2 as u128;

    // Check allowed values
    let allowed = &RULE_ALLOWED[r.6 as usize..(r.6 + r.7) as usize];
    for &(fid, values_offset) in allowed {
        let val = field::get(fid, word) as u32;
        // Walk the values list until sentinel (0xFFFFFFFF)
        let mut found = false;
        let mut idx = values_offset as usize;
        while RULE_ALLOWED_VALUES[idx] != 0xFFFFFFFF {
            if RULE_ALLOWED_VALUES[idx] == val {
                found = true;
                break;
            }
            idx += 1;
        }
        if !found {
            return Applied::NotAllowed;
        }
    }

    // Start with fixed bits and scheduling from the original
    let mut out = fixed | (word & SCHEDULING);

    // Apply moves
    let moves = &RULE_MOVES[r.4 as usize..(r.4 + r.5) as usize];
    for &(src_fid, dst_fid, conv) in moves {
        let raw = field::get(src_fid, word);
        let v: u64 = match conv {
            1 => {
                // Register: uniform -> ordinary
                if raw == 63 {
                    255
                } else {
                    match register(raw as u32) {
                        Some(r) => r as u64,
                        None => return Applied::NoRegister(raw as u32),
                    }
                }
            }
            2 => {
                // Predicate: uniform -> ordinary
                if raw == 7 {
                    7
                } else {
                    match predicate(raw as u32) {
                        Some(p) => p as u64,
                        None => return Applied::NoPredicate(raw as u32),
                    }
                }
            }
            // A signed number into a wider field: the same number.
            3 => field::signed(src_fid, word) as u64,
            _ => raw,
        };
        out = field::put(dst_fid, out, v);
    }

    Applied::Word(out)
}

/// Get the dropped fields of a rule (for uniform_address checks).
pub fn dropped_fields(rule_idx: usize) -> &'static [(&'static str, u16)] {
    let r = &RULES[rule_idx];
    &RULE_DROPPED[r.8 as usize..(r.8 + r.9) as usize]
}

/// Get the sm_70 class name of a rule.
pub fn rule_sm70_name(rule_idx: usize) -> &'static str {
    RULES[rule_idx].1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::Decoder;

    #[test]
    fn remap_uniform_operands_in_triton_kernels() {
        // Actual sm_75 words and their sm_70 forms as read by nvdisasm.
        // Ordinary registers start at R100 in this check.
        let cases = [
            // USGXT.U32 UR11, UR11, 0x4 -> SGXT.U32 R111, R111, 0x4
            (0x000fe20008000000_000000040b0b789a, 0x000fe20000000000_000000046f6f781a),
            // ULEA.HI.SX32 UR6, UR6, UR5, 0x1f -> LEA.HI.SX32 R106, R106, R105, 0x1f
            (0x000fc8000f8ffa3f_0000000506067291, 0x000fc800078ffaff_000000696a6a7211),
            // LEA.HI R0, R2, UR14, RZ, 0x1c -> LEA.HI R0, R2, R114, RZ, 0x1c
            (0x000fe4000f8fe0ff_0000000e02007c11, 0x000fe400078fe0ff_0000007202007211),
            // ULEA.HI UR4, UR4, 0x1, URZ, 0x1e -> LEA.HI R104, R104, 0x1, RZ, 0x1e
            (0x000fce000f8ff03f_0000000104047891, 0x000fce00078ff0ff_0000000168687811),
            // ULEA.HI in speculative attention: a different immediate, shift and destination.
            (0x000fe2000f8fe83f_0000800018057891, 0x000fe200078fe8ff_000080007c697811),
            // The same words with other fields, as nvdisasm reads them on both sides: a third
            // register that is not URZ, the negate bit, a guard, the other forms of USGXT.
            // ULEA.HI UR5, UR24, 0x8000, UR7, 0x1d -> LEA.HI R105, R124, 0x8000, R107, 0x1d
            (0x000fe2000f8fe807_0000800018057891, 0x000fe200078fe86b_000080007c697811),
            // ULEA.HI UR4, -UR4, 0x1, UR12, 0x1e -> LEA.HI R104, -R104, 0x1, R112, 0x1e
            (0x000fce000f8ff10c_0000000104047891, 0x000fce00078ff170_0000000168687811),
            // @!UP2 ULEA.HI UR5, UR24, 0x8000, UR3, 0x1d -> @!P2 LEA.HI R105, R124, 0x8000, R103, 0x1d
            (0x000fe2000f8fe803_000080001805a891, 0x000fe200078fe867_000080007c69a811),
            // USGXT UR2, UR9, 0xc -> SGXT R102, R109, 0xc
            (0x000fe20008000200_0000000c0902789a, 0x000fe20000000200_0000000c6d66781a),
            // @UP0 USGXT.W.U32 UR11, UR11, 0x4 -> @P0 SGXT.W.U32 R111, R111, 0x4
            (0x000fe20008000800_000000040b0b089a, 0x000fe20000000800_000000046f6f081a),
            // @!UP1 ULEA.HI.SX32 UR6, UR6, -UR8, 0x1f -> @!P1 LEA.HI.SX32 R106, R106, -R108, 0x1f
            (0x000fc8000f8ffa3f_8000000806069291, 0x000fc800078ffaff_8000006c6a6a9211),
            // @P3 LEA.HI R0, R2, -UR14, R7, 0x1c -> @P3 LEA.HI R0, R2, -R114, R7, 0x1c
            (0x000fe4000f8fe007_8000000e02003c11, 0x000fe400078fe007_8000007202003211),
        ];
        let mut decoder = Decoder::new(75);
        for (source, expected) in cases {
            let insn = decoder.decode(source).unwrap();
            let target = rule_of_target(insn.class_id).expect(insn.name);
            let rule = find_rule(insn.name, target, "").unwrap();
            let Applied::Word(mapped) = apply(rule, source, &|ur| Some(100 + ur), &Some) else {
                panic!("failed to map {}", insn.name);
            };
            assert_eq!(mapped, expected, "{}", insn.name);
        }
    }

    #[test]
    fn remap_uimad_and_carry_variants() {
        // Verified with nvdisasm -b SM75 and nvdisasm -b SM70; ordinary registers start at R100.
        let cases = [
            // UIMAD UR4, UR4, UR5, 0x240 -> IMAD R104, R104, R105, 0x240
            (
                0x001fcc000f8e0205_00000240040474a4_u128,
                0x001fcc00078e0269_0000024068687424_u128,
            ),
            // UIMAD UR4, UR4, UR5, -0x240 -> IMAD R104, R104, R105, -0x240
            (
                0x000fc2000f8e0205_fffffdc0040474a4_u128,
                0x000fc200078e0269_fffffdc068687424_u128,
            ),
            // UIMAD.X UR4, UR6, UR8, UR10, UP2 -> IMAD.X R104, R106, R108, R110, P2
            (
                0x000fc200090e060a_00000008060472a4_u128,
                0x000fc200010e066e_0000006c6a687224_u128,
            ),
            // UIMAD.X UR4, UR6, UR8, 0x240, UPT -> IMAD.X R104, R106, R108, 0x240, PT
            (
                0x000fc2000b8e0608_00000240060474a4_u128,
                0x000fc200038e066c_000002406a687424_u128,
            ),
            // @UP2 UIMAD.X UR4, UR6, 0x240, UR8, UP1 -> @P2 IMAD.X R104, R106, 0x240, R108, P1
            (
                0x000fc200088e0608_00000240060428a4_u128,
                0x000fc200008e066c_000002406a682824_u128,
            ),
            // UIMAD.WIDE.X UR4, UP1, UR6, UR8, UR10, UPT -> IMAD.WIDE.X R104, P1, R106, R108, R110, PT
            (
                0x000fc2000b82060a_00000008060472a5_u128,
                0x000fc2000382066e_0000006c6a687225_u128,
            ),
            // UIMAD.WIDE.X UR4, UR6, -0x240, UR10, UPT -> IMAD.WIDE.X R104, R106, -0x240, R110, PT
            (
                0x000fc2000b8e060a_fffffdc0060478a5_u128,
                0x000fc200038e066e_fffffdc06a687825_u128,
            ),
        ];
        let mut decoder = Decoder::new(75);
        for (source, expected) in cases {
            let insn = decoder.decode(source).unwrap();
            let target = rule_of_target(insn.class_id).expect(insn.name);
            let rule = find_rule(insn.name, target, "").unwrap();
            let Applied::Word(mapped) = apply(rule, source, &|ur| Some(100 + ur), &Some) else {
                panic!("failed to map {}", insn.name);
            };
            assert_eq!(mapped, expected, "{}", insn.name);
        }
    }
}
