//! Stall-count distance lookup: the shortest distance native sm_70 code keeps between
//! a writer and a reader, by kind.
//!
//! Port of `machine.py`'s `Machine.least` and `Machine.longest`.

use std::sync::OnceLock;

use crate::decode::{Decoded, Kind};
use crate::generated::distances::{DISTANCES, LONGEST_KINDS};

const NUM_KINDS: usize = 12;

#[cfg(test)]
const ALL_KINDS: [Kind; NUM_KINDS] = [
    Kind::FXU,
    Kind::IMAD,
    Kind::FMAI,
    Kind::FMALITE,
    Kind::FMA64HEAVY,
    Kind::MioFast,
    Kind::MioSlow,
    Kind::BRU,
    Kind::HMMA,
    Kind::IMMA,
    Kind::UDP,
    Kind::OTHER,
];

#[inline]
const fn kind_idx(kind: Kind) -> usize {
    match kind {
        Kind::FXU => 0,
        Kind::IMAD => 1,
        Kind::FMAI => 2,
        Kind::FMALITE => 3,
        Kind::FMA64HEAVY => 4,
        Kind::MioFast => 5,
        Kind::MioSlow => 6,
        Kind::BRU => 7,
        Kind::HMMA => 8,
        Kind::IMMA => 9,
        Kind::UDP => 10,
        Kind::OTHER => 11,
    }
}

struct DistanceTables {
    spec_mnems: Vec<&'static str>,
    first_byte_mask: u128,
    stride_s: usize,
    least_flat: Vec<u8>,
    longest_flat: Vec<u8>,
}

static TABLES: OnceLock<DistanceTables> = OnceLock::new();

fn longest_kind_of(kind_str: &str) -> Option<Kind> {
    match kind_str {
        "FXU" => Some(Kind::FXU),
        "IMAD" => Some(Kind::IMAD),
        "FMAI" => Some(Kind::FMAI),
        "FMALITE" => Some(Kind::FMALITE),
        "MIO_FAST" => Some(Kind::MioFast),
        "MIO_SLOW" => Some(Kind::MioSlow),
        "BRU" => Some(Kind::BRU),
        "HMMA" => Some(Kind::HMMA),
        _ => None,
    }
}

#[cfg(test)]
fn ref_least_in_tables(letter: u8, role: u8, writer_kind: Kind, writer_mnem: &str, reader_kind: Kind) -> u32 {
    let letter_str = if letter == 2 { "P" } else { "R" };
    let role_str = if role == 1 { "guard" } else { "data" };
    let reader_str = reader_kind.name();

    for &(l, r, w, rk, d) in DISTANCES.iter() {
        if l == letter_str && r == role_str && w.strip_prefix('=') == Some(writer_mnem) && rk == reader_str {
            return d as u32;
        }
    }

    let writer_kind_str = writer_kind.name();
    for &(l, r, w, rk, d) in DISTANCES.iter() {
        if l == letter_str && r == role_str && w == writer_kind_str && rk == reader_str {
            return d as u32;
        }
    }

    let mut max_d = 0u32;
    let mut found = false;
    for &(l, _r, w, _rk, d) in DISTANCES.iter() {
        if l == letter_str && w == writer_kind_str {
            max_d = max_d.max(d as u32);
            found = true;
        }
    }
    if found { max_d } else { 15 }
}

#[cfg(test)]
#[inline]
fn ref_least(letter: u8, role: u8, writer_kind: Kind, writer_mnem: &str, reader_kind: Kind) -> u32 {
    let d = ref_least_in_tables(letter, role, writer_kind, writer_mnem, reader_kind);
    if letter == 2 && writer_mnem == "R2P" {
        return d.max(if reader_kind == Kind::FXU && role == 0 { 5 } else { 12 });
    }
    d
}

fn parse_kind(s: &str) -> Option<usize> {
    match s {
        "FXU" => Some(0),
        "IMAD" => Some(1),
        "FMAI" => Some(2),
        "FMALITE" => Some(3),
        "FMA64HEAVY" => Some(4),
        "MIO_FAST" => Some(5),
        "MIO_SLOW" => Some(6),
        "BRU" => Some(7),
        "HMMA" => Some(8),
        "IMMA" => Some(9),
        "UDP" => Some(10),
        "OTHER" => Some(11),
        _ => None,
    }
}

fn tables() -> &'static DistanceTables {
    TABLES.get_or_init(|| {
        let mut spec_mnems: Vec<&'static str> = Vec::new();
        for &(_, _, w, _, _) in DISTANCES.iter() {
            if let Some(mnem) = w.strip_prefix('=')
                && !spec_mnems.contains(&mnem)
            {
                spec_mnems.push(mnem);
            }
        }
        if !spec_mnems.contains(&"R2P") {
            spec_mnems.push("R2P");
        }
        let mut first_byte_mask = 0u128;
        for &m in &spec_mnems {
            if let Some(&b) = m.as_bytes().first() {
                first_byte_mask |= 1u128 << b;
            }
        }
        let stride_s = spec_mnems.len() + 1;

        let mut kind_max = [[0u8; NUM_KINDS]; 2];
        let mut has_kind = [[false; NUM_KINDS]; 2];
        for &(l, _r, w, _rk, d) in DISTANCES.iter() {
            let l_idx = usize::from(l == "P");
            if let Some(wk) = parse_kind(w)
                && (!has_kind[l_idx][wk] || d > kind_max[l_idx][wk])
            {
                kind_max[l_idx][wk] = d;
                has_kind[l_idx][wk] = true;
            }
        }

        let mut base = vec![0u8; 2 * 2 * NUM_KINDS * NUM_KINDS];
        for l_idx in 0..2 {
            for r_idx in 0..2 {
                for wk in 0..NUM_KINDS {
                    let def = if has_kind[l_idx][wk] { kind_max[l_idx][wk] } else { 15 };
                    let off = ((l_idx * 2 + r_idx) * NUM_KINDS + wk) * NUM_KINDS;
                    base[off..off + NUM_KINDS].fill(def);
                }
            }
        }
        for &(l, r, w, rk, d) in DISTANCES.iter() {
            if let (Some(wk), Some(rki)) = (parse_kind(w), parse_kind(rk)) {
                let l_idx = usize::from(l == "P");
                let r_idx = usize::from(r == "guard");
                base[(((l_idx * 2 + r_idx) * NUM_KINDS + wk) * NUM_KINDS) + rki] = d;
            }
        }

        let mut least_flat = vec![0u8; 2 * 2 * NUM_KINDS * stride_s * NUM_KINDS];
        for l_idx in 0..2 {
            for r_idx in 0..2 {
                for wk in 0..NUM_KINDS {
                    let b_off = ((l_idx * 2 + r_idx) * NUM_KINDS + wk) * NUM_KINDS;
                    for s_idx in 0..stride_s {
                        let d_off = (((l_idx * 2 + r_idx) * NUM_KINDS + wk) * stride_s + s_idx) * NUM_KINDS;
                        least_flat[d_off..d_off + NUM_KINDS].copy_from_slice(&base[b_off..b_off + NUM_KINDS]);
                    }
                }
            }
        }
        for &(l, r, w, rk, d) in DISTANCES.iter() {
            if let (Some(mnem), Some(rki)) = (w.strip_prefix('='), parse_kind(rk)) {
                let s_idx = spec_mnems.iter().position(|&m| m == mnem).unwrap() + 1;
                let l_idx = usize::from(l == "P");
                let r_idx = usize::from(r == "guard");
                for wk in 0..NUM_KINDS {
                    let idx = ((((l_idx * 2 + r_idx) * NUM_KINDS + wk) * stride_s + s_idx) * NUM_KINDS) + rki;
                    least_flat[idx] = d;
                }
            }
        }
        let r2p_s_idx = spec_mnems.iter().position(|&m| m == "R2P").unwrap() + 1;
        for r_idx in 0..2 {
            for wk in 0..NUM_KINDS {
                for rk in 0..NUM_KINDS {
                    let min_r2p = if rk == 0 && r_idx == 0 { 5 } else { 12 };
                    let idx = (((2 + r_idx) * NUM_KINDS + wk) * stride_s + r2p_s_idx) * NUM_KINDS + rk;
                    if least_flat[idx] < min_r2p {
                        least_flat[idx] = min_r2p;
                    }
                }
            }
        }

        let mut longest_flat = vec![0u8; 2 * NUM_KINDS * stride_s];
        for l_idx in 0..2 {
            let letter = if l_idx == 1 { 2 } else { 0 };
            let roles: &[u8] = if letter == 2 { &[0, 1] } else { &[0] };
            for wk in 0..NUM_KINDS {
                for s_idx in 0..stride_s {
                    let mut max_d = 0u32;
                    for &role in roles {
                        let r_idx = role as usize;
                        for &kind_str in LONGEST_KINDS.iter() {
                            if let Some(kind) = longest_kind_of(kind_str) {
                                let rk = kind_idx(kind);
                                let flat_idx =
                                    ((((l_idx * 2 + r_idx) * NUM_KINDS + wk) * stride_s + s_idx) * NUM_KINDS) + rk;
                                max_d = max_d.max(least_flat[flat_idx] as u32);
                            }
                        }
                    }
                    longest_flat[(l_idx * NUM_KINDS + wk) * stride_s + s_idx] = max_d as u8;
                }
            }
        }

        DistanceTables {
            spec_mnems,
            first_byte_mask,
            stride_s,
            least_flat,
            longest_flat,
        }
    })
}

impl DistanceTables {
    #[inline]
    fn spec_idx(&self, mnem: &str) -> usize {
        if let Some(&b) = mnem.as_bytes().first()
            && ((self.first_byte_mask >> b) & 1) != 0
        {
            for (idx, &m) in self.spec_mnems.iter().enumerate() {
                if m == mnem {
                    return idx + 1;
                }
            }
        }
        0
    }
}

/// The minimum distance native sm_70 code keeps from `writer` writing a register (letter "R")
/// or predicate (letter "P") with `role` ("data"/"guard") to a reader of kind `reader_kind`.
#[inline]
pub fn least(letter: u8, role: u8, writer: &Decoded, reader_kind: Kind) -> u32 {
    let t = tables();
    let l_idx = usize::from(letter == 2);
    let r_idx = usize::from(role == 1);
    let wk = kind_idx(writer.kind);
    let s_idx = t.spec_idx(writer.mnemonic);
    let rk = kind_idx(reader_kind);
    let idx = ((((l_idx * 2 + r_idx) * NUM_KINDS + wk) * t.stride_s + s_idx) * NUM_KINDS) + rk;
    t.least_flat[idx] as u32
}

/// The distance that native sm_70 code keeps from `writer` to a reader that waits for the
/// scoreboard on which it announces its result, where a wait is not all it keeps: `None` for every
/// writer but the four of double precision. A reader of what DADD, DMUL, DFMA or DSETP writes
/// never comes closer than this in Volta's own code, scoreboard or not (8 cycles to an instruction
/// of the same pipe, 9 to any other; a predicate of DSETP 15 as a guard or to a branch), where
/// code for sm_80 has 6 and 14 (docs/validation.md, section 8). Port of `Machine.waited`.
pub fn waited(letter: u8, role: u8, writer: &Decoded, reader_kind: Kind) -> Option<u32> {
    if !matches!(writer.mnemonic, "DADD" | "DMUL" | "DFMA" | "DSETP") {
        return None;
    }
    if letter == 2 && (role == 1 || reader_kind == Kind::BRU) {
        return Some(15);
    }
    Some(if reader_kind == Kind::FMALITE { 8 } else { 9 })
}

/// `waited` for any reader: what is kept from `writer` to the end of straight-line code, where
/// the reader is not seen (it may be the first instruction of a branch's target). Port of
/// `Machine.waited_longest`.
pub fn waited_longest(letter: u8, writer: &Decoded) -> Option<u32> {
    if !matches!(writer.mnemonic, "DADD" | "DMUL" | "DFMA" | "DSETP") {
        return None;
    }
    Some(if letter == 2 { 15 } else { 9 })
}

/// The distance that is enough from `writer` to any reader of what it writes.
#[inline]
pub fn longest(letter: u8, writer: &Decoded) -> u32 {
    let t = tables();
    let l_idx = usize::from(letter == 2);
    let wk = kind_idx(writer.kind);
    let s_idx = t.spec_idx(writer.mnemonic);
    t.longest_flat[(l_idx * NUM_KINDS + wk) * t.stride_s + s_idx] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_tables_match_reference_scan_exhaustively() {
        let mut mnems: Vec<&'static str> = vec!["", "FFMA", "NOP", "MOV", "IMAD", "R2P"];
        for &(_, _, w, _, _) in DISTANCES.iter() {
            if let Some(m) = w.strip_prefix('=')
                && !mnems.contains(&m)
            {
                mnems.push(m);
            }
        }
        for &letter in &[0u8, 2] {
            for &wk in &ALL_KINDS {
                for &mnem in &mnems {
                    let dec = Decoded {
                        word: 0,
                        class_id: 0,
                        kind: wk,
                        mnemonic: mnem,
                        name: "test",
                        ops: Vec::new(),
                        reads: Vec::new(),
                        writes: Vec::new(),
                    };
                    let mut ref_longest = 0u32;
                    let roles: &[u8] = if letter == 2 { &[0, 1] } else { &[0] };
                    for &role in roles {
                        for &ks in LONGEST_KINDS.iter() {
                            if let Some(rk) = longest_kind_of(ks) {
                                ref_longest = ref_longest.max(ref_least(letter, role, wk, mnem, rk));
                            }
                        }
                    }
                    assert_eq!(
                        longest(letter, &dec),
                        ref_longest,
                        "longest letter={letter} {wk:?} {mnem}"
                    );

                    for &role in &[0u8, 1] {
                        for &rk in &ALL_KINDS {
                            let got = least(letter, role, &dec, rk);
                            let expected = ref_least(letter, role, wk, mnem, rk);
                            assert_eq!(
                                got, expected,
                                "least letter={letter} role={role} {wk:?} {mnem} -> {rk:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
