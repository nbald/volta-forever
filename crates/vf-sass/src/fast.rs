//! Fast non-cryptographic hashing and compact register-set primitives.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

const K: u64 = 0x517cc1b727220a95;

#[derive(Default, Clone, Copy)]
pub(crate) struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        FxHasher { hash: 0 }
    }
}

pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn mix(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let (head, tail) = bytes.split_at(8);
            self.mix(u64::from_le_bytes(head.try_into().unwrap()));
            bytes = tail;
        }
        if bytes.len() >= 4 {
            let (head, tail) = bytes.split_at(4);
            self.mix(u32::from_le_bytes(head.try_into().unwrap()) as u64);
            bytes = tail;
        }
        if bytes.len() >= 2 {
            let (head, tail) = bytes.split_at(2);
            self.mix(u16::from_le_bytes(head.try_into().unwrap()) as u64);
            bytes = tail;
        }
        if let Some(&b) = bytes.first() {
            self.mix(b as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.mix(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.mix(i as u64);
        self.mix((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }
}

pub(crate) type FastMap<K, V> = HashMap<K, V, FxBuildHasher>;
pub(crate) type FastSet<T> = HashSet<T, FxBuildHasher>;

/// Bitset of registers 0..=255 (`R0..=R254`, `RZ = 255`).
pub(crate) type RegSet = [u64; 4];

#[inline]
pub(crate) fn regset_set(s: &mut RegSet, r: u32) {
    if r < 256 {
        s[(r >> 6) as usize] |= 1u64 << (r & 63);
    }
}

#[inline]
pub(crate) fn regset_clear(s: &mut RegSet, r: u32) {
    if r < 256 {
        s[(r >> 6) as usize] &= !(1u64 << (r & 63));
    }
}

#[inline]
pub(crate) fn regset_test(s: &RegSet, r: u32) -> bool {
    r < 256 && ((s[(r >> 6) as usize] >> (r & 63)) & 1) != 0
}

#[inline]
pub(crate) fn regset_is_empty(s: &RegSet) -> bool {
    *s == [0; 4]
}

#[inline]
pub(crate) fn regset_intersects(a: &RegSet, b: &RegSet) -> bool {
    ((a[0] & b[0]) | (a[1] & b[1]) | (a[2] & b[2]) | (a[3] & b[3])) != 0
}

#[inline]
pub(crate) fn regset_union(dst: &mut RegSet, src: &RegSet) -> bool {
    let mut changed = false;
    for i in 0..4 {
        let old = dst[i];
        let next = old | src[i];
        if next != old {
            dst[i] = next;
            changed = true;
        }
    }
    changed
}

#[inline]
pub(crate) fn regset_sub(dst: &mut RegSet, mask: &RegSet) {
    for i in 0..4 {
        dst[i] &= !mask[i];
    }
}

#[inline]
pub(crate) fn regset_for_each<F: FnMut(u32)>(s: &RegSet, mut f: F) {
    for (w, &bits) in s.iter().enumerate() {
        let mut left = bits;
        while left != 0 {
            let b = left.trailing_zeros();
            f((w as u32) * 64 + b);
            left &= left - 1;
        }
    }
}
