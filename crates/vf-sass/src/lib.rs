//! Translates Turing (sm_75) cubins and selected executable Ampere (sm_80) instructions
//! into Volta (sm_70) cubins. Each refused function has a reason in the output report.
//!
//! It is the reference translator `reference/translate.py` carried over, and must give the same bytes:
//! `tests/sass/compare.sh` compares the two on real code, `tests/sass/fuzz.py` on damaged code. A
//! change to what is translated is a change to both.
//!
//! The tables (`generated`, which is the crate `vf-tables`) hold the two instruction sets (classes,
//! fields, what an instruction reads and writes), the rules for uniform operands and the distances
//! native code keeps. They are written by `reference/generate.py`, followed by
//! `cargo fmt -p vf-tables`.

pub mod assemble;
pub mod decode;
pub mod distance;
pub mod effects;
pub mod elf;
pub(crate) mod fast;
pub mod field;
pub mod pairing;
pub mod rule;
pub mod translate;

/// The instruction tables (the crate `vf-tables`, which has a licence of its own).
pub use vf_tables as generated;

#[cfg(test)]
mod tests {
    use crate::assemble::{self, OpVal};
    use crate::decode::Decoder;
    use crate::generated::fields::*;

    #[test]
    fn test_uldc_decode() {
        let w: u128 = 0x000fe40000000a0000005c0000047ab9;
        let mut dec = Decoder::new(75);
        let d = dec.decode(w).expect("should decode as uldc_const__RCR");
        assert_eq!(d.name, "uldc_const__RCR");
        assert_eq!(d.op_or(OP_SZ, 999), 5);
    }

    #[test]
    fn test_mov_assembly() {
        let w = assemble::assemble(
            "mov__RCR_RC",
            &[
                ("Pg", OpVal::Int(7)),
                ("Pg_not", OpVal::Int(0)),
                ("Rd", OpVal::Int(105)),
                ("Sb_bank", OpVal::Int(0)),
                ("Sb_addr", OpVal::Int(92)),
            ],
        )
        .unwrap();
        // Unused bits (32-39) should be zero
        assert_eq!((w >> 32) & 0xff, 0);
    }

    #[test]
    fn test_r2ur_decode() {
        let w: u128 = 0x004e2800000e000000000000000d03c2;
        let mut dec = Decoder::new(75);
        let d = dec.decode(w).expect("should decode as r2ur__noOR");
        assert_eq!(d.name, "r2ur__noOR");
        assert_eq!(d.op_or(OP_RA, 999), 0);
    }

    #[test]
    fn test_nop_decode() {
        let w: u128 = crate::generated::special::NOP;
        let mut dec = Decoder::new(70);
        let d = dec.decode(w).expect("NOP should decode");
        assert_eq!(d.name, "nop_");
    }

    #[test]
    fn test_field_put_get_roundtrip() {
        // Field 10 is BITS_8_23_16_Rd (8-bit, bits 23..16, contiguous, start=16)
        let word = 0u128;
        let word = crate::field::put(10, word, 42);
        assert_eq!(crate::field::get(10, word), 42);
    }
}
