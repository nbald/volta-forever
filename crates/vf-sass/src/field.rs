//! Field operations on 128-bit instruction words.
//!
//! A field is a set of bit positions within a 128-bit instruction word. The generator
//! stores each field as a slice of FIELD_BITS (bit positions, most significant first) and
//! a precomputed mask. Most fields are contiguous bit ranges; for those, the `start_bit`
//! in the FIELDS table enables a fast path.
//!
//! Values are u64 because some fields (branch offsets) are up to 48 bits wide.

use crate::generated::fields::{FIELD_BITS, FIELDS};

/// Get the value of field `fid` from a 128-bit instruction word.
/// Returns u64 because fields can be up to 48 bits (branch offsets).
#[inline]
pub fn get(fid: u16, word: u128) -> u64 {
    if fid == 0 {
        return ((((word >> 91) & 1) << 12) as u64) | (word as u64 & 0xfff);
    }
    if fid == 14 {
        return ((((word >> 122) & 7) << 5) | ((word >> 105) & 31)) as u64;
    }
    let (offset, width, _mask_lo, _mask_hi, start) = FIELDS[fid as usize];
    if start != 255 {
        ((word >> start) & ((1u128 << width) - 1)) as u64
    } else {
        let bits = &FIELD_BITS[offset as usize..(offset as usize + width as usize)];
        let mut value = 0u64;
        for &b in bits {
            value = (value << 1) | ((word >> b) as u64 & 1);
        }
        value
    }
}

/// Set the value of field `fid` in a 128-bit instruction word.
#[inline]
pub fn put(fid: u16, word: u128, value: u64) -> u128 {
    if fid == 14 {
        const MASK_14: u128 = 0x1c00_3e00_0000_0000u128 << 64;
        let v = value as u128;
        return (word & !MASK_14) | (((v >> 5) & 7) << 122) | ((v & 31) << 105);
    }
    let (offset, width, mask_lo, mask_hi, start) = FIELDS[fid as usize];
    let mask = (mask_hi as u128) << 64 | mask_lo as u128;
    if start != 255 {
        (word & !mask) | (((value as u128) << start) & mask)
    } else {
        let bits = &FIELD_BITS[offset as usize..(offset as usize + width as usize)];
        let mut w = word;
        for (i, &b) in bits.iter().rev().enumerate() {
            w = (w & !(1u128 << b)) | (((value as u128 >> i) & 1) << b);
        }
        w
    }
}

/// The number of bits in a field.
#[inline]
pub fn width(fid: u16) -> u8 {
    FIELDS[fid as usize].1
}

/// Sign-extend the value of field `fid` from the instruction word.
/// Returns i64 because fields can be up to 48 bits.
#[inline]
pub fn signed(fid: u16, word: u128) -> i64 {
    let v = get(fid, word);
    let w = width(fid);
    if w > 0 && w < 64 && (v >> (w - 1)) & 1 != 0 {
        v as i64 - (1i64 << w)
    } else {
        v as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generic_get(fid: u16, word: u128) -> u64 {
        let (offset, width, _, _, _) = FIELDS[fid as usize];
        let bits = &FIELD_BITS[offset as usize..(offset as usize + width as usize)];
        let mut value = 0u64;
        for &b in bits {
            value = (value << 1) | ((word >> b) as u64 & 1);
        }
        value
    }

    fn generic_put(fid: u16, word: u128, value: u64) -> u128 {
        let (offset, width, _, _, _) = FIELDS[fid as usize];
        let bits = &FIELD_BITS[offset as usize..(offset as usize + width as usize)];
        let mut w = word;
        for (i, &b) in bits.iter().rev().enumerate() {
            w = (w & !(1u128 << b)) | (((value as u128 >> i) & 1) << b);
        }
        w
    }

    #[test]
    fn fast_field_paths_match_bit_tables() {
        let words = [
            0u128,
            u128::MAX,
            0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128,
            (1u128 << 91) | 0xabc,
            (5u128 << 122) | (23u128 << 105) | (19u128 << 116),
        ];
        for &w in &words {
            for fid in [0u16, 14, 17, 18, 19, 34] {
                assert_eq!(get(fid, w), generic_get(fid, w), "get fid={fid} w={w:#x}");
            }
            for val in [0u64, 1, 0x1f, 0xff, 0xfff] {
                assert_eq!(put(14, w, val), generic_put(14, w, val), "put fid=14 val={val}");
            }
        }
    }
}
