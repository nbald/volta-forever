//! Borrowed registers: registers of the program that hold nothing where the translation
//! needs them. Liveness analysis of registers, homes for uniform register value-flow webs,
//! the pool at each instruction, and victims for room-making at level 3.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::decode::{Kind, LETTER_R, LETTER_U, for_each_uniform_pair};
use crate::generated::special::{READ_BARRIER, WRITE_BARRIER};
use crate::pairing::{BorrowFlow, unguarded};

use super::parse::FuncData;
use super::{Refused, RegSet, regset_andnot, regset_empty, regset_or, regset_set, regset_test, unannounced};

const FLAG_UNGUARDED: u8 = 1;
const FLAG_CALL: u8 = 2;
const FLAG_INEXACT_RET: u8 = 4;
const FLAG_KILL: u8 = 8;

enum BusyVec {
    W1(Vec<[u64; 1]>),
    W2(Vec<[u64; 2]>),
    W4(Vec<[u64; 4]>),
}

impl BusyVec {
    fn into_regsets(self) -> Vec<RegSet> {
        match self {
            Self::W1(v) => v.into_iter().map(|w| [w[0], 0, 0, 0]).collect(),
            Self::W2(v) => v.into_iter().map(|w| [w[0], w[1], 0, 0]).collect(),
            Self::W4(v) => v,
        }
    }
}

/// Prepared per-function register liveness and uniform value-flow units, computed once
/// per candidate function so L1/L2 occupancy checks need zero redundant flow/liveness work.
pub(super) struct BorrowPrep {
    pub(super) n: usize,
    pub(super) highest: u32,
    pub(super) flow: BorrowFlow,
    busy: BusyVec,
    touch_opt: Option<Vec<RegSet>>,
    never: RegSet,
    pub(super) late_read_counts: [u32; 256],
    pub(super) board_costs: [u32; 256],
    webs: Vec<Web>,
    units: Vec<Unit>,
}

struct Web {
    u: u32,
    span: Vec<usize>,
    busy: RegSet,
}

struct Unit {
    min_u: u32,
    width: u32,
    total_span_len: usize,
    first_insn: usize,
    members: Vec<usize>,
    pair_firsts: Vec<u32>,
    valid: bool,
}

pub(super) struct HomeSolution {
    pub(super) top: u32,
    pub(super) max_borrowed: u32,
    pub(super) unit_reg: Vec<Option<u32>>,
    pub(super) taken: Vec<RegSet>,
    pub(super) unhomed_u: HashSet<u32>,
    pub(super) unhomed_pairs: HashSet<u32>,
}

/// Registers of the program that hold nothing where the translation needs registers: homes
/// for each uniform register value-flow web through its life, and an even register and the next
/// at an instruction whose sequence computes into two (the pool). `docs/translator.md`, "Registers".
pub(super) struct Borrowed {
    /// The highest register the program uses.
    pub(super) top: u32,
    /// The function's own highest register (before any budget override).
    pub(super) highest: u32,
    /// At each instruction: a bit set of registers that are alive, read or written there.
    pub(super) busy: Vec<RegSet>,
    /// At each instruction: a bit set of registers taken by homes of uniform registers.
    pub(super) taken: Vec<RegSet>,
    /// At each instruction: a bit set of registers read or written by the instruction itself
    /// (for `victims_at`: registers the instruction does not touch can be put in the frame).
    pub(super) touch: Vec<RegSet>,
    /// A bit set of registers that some instruction writes late without saying when
    /// (never borrowed as victims either).
    pub(super) never: RegSet,
    /// Per-instruction count of unannounced late (`is_late()`, `rb == 7`) ordinary reads.
    pub(super) late_read_counts: [u32; 256],
    /// Weighted per-register count of scoreboard-bound (`wb < 7` or `rb < 7`) accesses.
    pub(super) board_costs: [u32; 256],
    /// Per-instruction home assignments: `(u, r)` sorted by `u` at each instruction `i`.
    pub(super) home_at_insn: Vec<Vec<(u32, u32)>>,
    /// Homed value-flow webs: `(uniform_reg, ordinary_reg, sorted_insn_indices)` for `MAP.lives`.
    pub(super) web_homes: Vec<(u32, u32, Vec<usize>)>,
    /// Uniform registers that have at least one unhomed web (need `placement` or frame `slots`).
    pub(super) unhomed_u: HashSet<u32>,
    /// Uniform 64-bit pair heads (`first` of `first:first+1`) with an unhomed pair web.
    pub(super) unhomed_pairs: HashSet<u32>,
    /// The ordinary registers taken for homes or for the pool.
    pub(super) used: HashSet<u32>,
    /// Level 3: everything at the instruction.
    pub(super) everything: bool,
    /// {uniform register: its word in the frame}.
    pub(super) slots: HashMap<u32, u32>,
    /// Held uniform predicates whose value must survive an interval with every register busy.
    pub(super) predicate_slots: HashMap<u32, u32>,
    /// The frame size in bytes (a multiple of 16).
    pub(super) frame_bytes: u32,
    /// Where the stack pointer is at each instruction: `depth[i]` is how far R1 is, before
    /// instruction i, under where the first instruction puts it. None when the depth is not
    /// known at that instruction.
    pub(super) depth: Vec<Option<i32>>,
    /// Whether the depth of R1 is known everywhere the first instruction leads.
    pub(super) r1_fixed: bool,
    /// Whether room is made where no register holds nothing (VF_BORROW >= 5): registers
    /// the instruction does not touch are put in the frame and brought back.
    pub(super) room: bool,
    /// Whether the frame is addressed from R1 itself (the stack pointer), not from a base
    /// register loaded from c[0x0][0x28].
    pub(super) from_r1: bool,
    /// The most registers any single instruction put away at once (for the final frame size).
    pub(super) spill: u32,
}

#[inline]
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

#[inline]
fn uf_union(parent: &mut [usize], a: usize, b: usize) -> bool {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[rb] = ra;
        true
    } else {
        false
    }
}

#[inline]
fn regset_disjoint(a: &RegSet, b: &RegSet) -> bool {
    (a[0] & b[0]) | (a[1] & b[1]) | (a[2] & b[2]) | (a[3] & b[3]) == 0
}

#[inline]
fn regset_is_zero(a: &RegSet) -> bool {
    (a[0] | a[1] | a[2] | a[3]) == 0
}

#[inline]
fn bits_or<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    let mut out = [0u64; W];
    for i in 0..W {
        out[i] = a[i] | b[i];
    }
    out
}

#[inline]
fn bits_andnot<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    let mut out = [0u64; W];
    for i in 0..W {
        out[i] = a[i] & !b[i];
    }
    out
}

#[inline]
fn bits_set<const W: usize>(a: &mut [u64; W], r: u32) {
    a[(r / 64) as usize] |= 1u64 << (r % 64);
}

#[inline]
fn bits_to_regset<const W: usize>(a: &[u64; W]) -> RegSet {
    let mut out = [0u64; 4];
    out[..W].copy_from_slice(a);
    out
}

fn build_prep<const W: usize>(
    fd: &FuncData,
    flow: BorrowFlow,
    need_late: bool,
    need_l3: bool,
    wrap_busy: impl FnOnce(Vec<[u64; W]>) -> BusyVec,
) -> BorrowPrep {
    let n = fd.src.len();
    let track_u = !fd.uniform.is_empty() || flow.has_calls;
    let mut reads_r: Vec<[u64; W]> = Vec::with_capacity(n);
    let mut writes_r: Vec<[u64; W]> = Vec::with_capacity(n);
    let mut reads_u: Vec<u64> = if track_u { Vec::with_capacity(n) } else { Vec::new() };
    let mut writes_u: Vec<u64> = if track_u { Vec::with_capacity(n) } else { Vec::new() };
    let mut flags: Vec<u8> = Vec::with_capacity(n);
    let mut every_r = [0u64; W];
    let mut every_u: u64 = 0;
    let mut never_w = [0u64; W];
    let mut late_read_counts = [0u32; 256];
    let mut board_costs = [0u32; 256];
    let mut pair_sites: Vec<(usize, u32)> = Vec::new();
    let mut pair_mask = 0u64;
    for &p in &fd.pairs {
        if p < 62 {
            pair_mask |= 3u64 << p;
        }
    }

    for (i, d) in fd.src.iter().enumerate() {
        let mut rr = [0u64; W];
        let mut wr = [0u64; W];
        let mut ru = 0u64;
        let mut wu = 0u64;
        for &(letter, x, _) in &d.reads {
            if letter == LETTER_R && (x as usize) < W * 64 && x != 255 {
                bits_set(&mut rr, x);
            } else if letter == LETTER_U && x != 63 {
                ru |= 1u64 << x;
            }
        }
        for &(letter, x) in &d.writes {
            if letter == LETTER_R && (x as usize) < W * 64 && x != 255 {
                bits_set(&mut wr, x);
            } else if letter == LETTER_U && x != 63 {
                wu |= 1u64 << x;
            }
        }
        reads_r.push(rr);
        writes_r.push(wr);
        if track_u {
            reads_u.push(ru);
            writes_u.push(wu);
        }
        if need_late {
            let wb = ((d.word >> WRITE_BARRIER) & 7) as u32;
            let rb = ((d.word >> READ_BARRIER) & 7) as u32;
            let w_cost = if d.kind == Kind::MioSlow { 4u32 } else { 1u32 };
            if wb < 7 {
                for (wi, &word) in wr.iter().enumerate() {
                    let mut m = word;
                    while m != 0 {
                        let reg = wi * 64 + m.trailing_zeros() as usize;
                        if reg < 255 {
                            board_costs[reg] += w_cost;
                        }
                        m &= m - 1;
                    }
                }
            }
            if rb < 7 {
                for (wi, &word) in rr.iter().enumerate() {
                    let mut m = word;
                    while m != 0 {
                        let reg = wi * 64 + m.trailing_zeros() as usize;
                        if reg < 255 {
                            board_costs[reg] += w_cost;
                        }
                        m &= m - 1;
                    }
                }
            }
            if d.kind.is_late() && rb == 7 {
                for (wi, &word) in rr.iter().enumerate() {
                    let mut m = word;
                    while m != 0 {
                        let reg = wi * 64 + m.trailing_zeros() as usize;
                        if reg < 255 {
                            late_read_counts[reg] += 1;
                        }
                        m &= m - 1;
                    }
                }
            }
        }
        if unannounced(d) {
            never_w = bits_or(&never_w, &wr);
        }
        if ((ru | wu) & pair_mask) != 0 {
            let mut pairs_buf: [u32; 4] = [0; 4];
            let mut np = 0usize;
            for_each_uniform_pair(d, |first| {
                if np < pairs_buf.len() {
                    pairs_buf[np] = first;
                    np += 1;
                } else {
                    pair_sites.push((i, first));
                }
            });
            pairs_buf[..np].sort_unstable();
            for &pf in &pairs_buf[..np] {
                pair_sites.push((i, pf));
            }
        }
        every_r = bits_or(&every_r, &bits_or(&rr, &wr));
        every_u |= ru | wu;
        let ung = unguarded(d);
        let mut f = if ung { FLAG_UNGUARDED | FLAG_KILL } else { 0 };
        if flow.is_call[i] {
            f |= FLAG_CALL;
            f &= !FLAG_KILL;
        }
        if d.mnemonic == "RET" {
            f |= FLAG_INEXACT_RET;
        }
        flags.push(f);
    }

    let touch_opt = if need_l3 {
        Some(
            reads_r
                .iter()
                .zip(&writes_r)
                .map(|(r, w)| bits_to_regset(&bits_or(r, w)))
                .collect(),
        )
    } else {
        None
    };

    let (raw_reads_u, raw_writes_u) = if flow.has_calls {
        (reads_u.clone(), writes_u.clone())
    } else {
        (Vec::new(), Vec::new())
    };

    let mut target_callers: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut callee_bodies: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    let mut exact_targets: HashSet<usize> = HashSet::new();
    if flow.has_calls && n > 0 {
        let mut main_reachable = vec![false; n];
        let mut work = vec![0usize];
        main_reachable[0] = true;
        while let Some(k) = work.pop() {
            for &t in flow.succ(k) {
                let t = t as usize;
                if !main_reachable[t] {
                    main_reachable[t] = true;
                    work.push(t);
                }
            }
        }
        for i in 0..n {
            if flow.is_call[i]
                && let Some(target) = flow.named[i]
            {
                target_callers.entry(target).or_default().push(i);
            }
        }
        for (target, callers) in &target_callers {
            let mut body = BTreeSet::new();
            let mut work = vec![*target];
            while let Some(k) = work.pop() {
                if !body.insert(k) || fd.src[k].mnemonic == "RET" {
                    continue;
                }
                work.extend(flow.succ(k).iter().map(|&t| t as usize));
            }
            let all_calls_exact = callers
                .iter()
                .all(|&c| fd.src[c].name.starts_with("call_rel") && flow.called.get(&c).is_some_and(Option::is_some));
            let isolated = !body.iter().any(|&k| main_reachable[k]);
            let clean_rets = body
                .iter()
                .all(|&k| fd.src[k].mnemonic != "RET" || (fd.src[k].name == "ret__REL" && unguarded(&fd.src[k])));
            if all_calls_exact && isolated && clean_rets {
                exact_targets.insert(*target);
                for &k in &body {
                    if fd.src[k].mnemonic == "RET" {
                        flags[k] &= !FLAG_INEXACT_RET;
                    }
                }
            }
            callee_bodies.insert(*target, body);
        }
        for i in 0..n {
            if !flow.is_call[i] {
                continue;
            }
            match flow.called.get(&i) {
                Some(None) | None => {
                    reads_r[i] = bits_or(&reads_r[i], &every_r);
                    reads_u[i] |= every_u;
                }
                Some(Some((alive, ws))) => {
                    for &x in alive {
                        if (x as usize) < W * 64 && x != 255 {
                            bits_set(&mut reads_r[i], x);
                        }
                    }
                    for &x in ws {
                        if (x as usize) < W * 64 && x != 255 {
                            bits_set(&mut writes_r[i], x);
                        }
                    }
                    if let Some(entry) = flow.named[i] {
                        let body = &callee_bodies[&entry];
                        let exit_u = if exact_targets.contains(&entry) { 0u64 } else { every_u };
                        let mut cal_live: BTreeMap<usize, u64> = body.iter().map(|&k| (k, 0)).collect();
                        let mut changed_u = true;
                        while changed_u {
                            changed_u = false;
                            for &k in body.iter().rev() {
                                let mut after = if fd.src[k].mnemonic == "RET" { exit_u } else { 0u64 };
                                for &t in flow.succ(k) {
                                    if let Some(&v) = cal_live.get(&(t as usize)) {
                                        after |= v;
                                    }
                                }
                                let ck = if (flags[k] & FLAG_UNGUARDED) != 0 {
                                    raw_writes_u[k]
                                } else {
                                    0
                                };
                                let now = raw_reads_u[k] | (after & !ck);
                                if now != cal_live[&k] {
                                    cal_live.insert(k, now);
                                    changed_u = true;
                                }
                            }
                        }
                        reads_u[i] |= cal_live.get(&entry).copied().unwrap_or(0);
                        for &k in body {
                            writes_u[i] |= raw_writes_u[k];
                        }
                    }
                }
            }
        }
    }

    let mut busy: Vec<[u64; W]> = vec![[0u64; W]; n];
    let mut live_in_u: Vec<u64> = if track_u { vec![0u64; n] } else { Vec::new() };
    let mut live_out_u: Vec<u64> = if flow.has_calls { vec![0u64; n] } else { Vec::new() };
    let mut busy_u: Vec<u64> = if track_u { vec![0u64; n] } else { Vec::new() };
    let mut live_in_r: Vec<[u64; W]> = if exact_targets.is_empty() {
        Vec::new()
    } else {
        vec![[0u64; W]; n]
    };

    let mut bb_bounds: Vec<(usize, usize)> = Vec::new();
    let mut bb_of: Vec<usize> = Vec::new();
    let mut bb_busy_r: Vec<[u64; W]> = Vec::new();
    let mut bb_any_u: Vec<u64> = Vec::new();
    let mut bb_whole_u: Vec<u64> = Vec::new();

    if n > 0 {
        let mut leader = vec![false; n];
        leader[0] = true;
        for i in 0..n {
            let s = flow.succ(i);
            if (flags[i] & (FLAG_CALL | FLAG_INEXACT_RET)) != 0 || s.len() != 1 || s[0] as usize != i + 1 {
                if i + 1 < n {
                    leader[i + 1] = true;
                }
                for &t in s {
                    leader[t as usize] = true;
                }
            }
        }
        bb_of.resize(n, 0usize);
        let mut start = 0usize;
        for (i, &is_lead) in leader.iter().enumerate().skip(1) {
            if is_lead {
                let b = bb_bounds.len();
                bb_bounds.push((start, i));
                bb_of[start..i].fill(b);
                start = i;
            }
        }
        let b = bb_bounds.len();
        bb_bounds.push((start, n));
        bb_of[start..n].fill(b);

        let num_bbs = bb_bounds.len();
        let mut bb_gen_r: Vec<[u64; W]> = Vec::with_capacity(num_bbs);
        let mut bb_kill_r: Vec<[u64; W]> = Vec::with_capacity(num_bbs);
        let mut bb_gen_u: Vec<u64> = if track_u {
            Vec::with_capacity(num_bbs)
        } else {
            Vec::new()
        };
        let mut bb_kill_u: Vec<u64> = if track_u {
            Vec::with_capacity(num_bbs)
        } else {
            Vec::new()
        };
        let mut bb_has_leaf: Vec<bool> = Vec::with_capacity(num_bbs);
        let mut bb_succ_off: Vec<u32> = Vec::with_capacity(num_bbs + 1);
        let mut bb_succ_data: Vec<usize> = Vec::with_capacity(num_bbs * 2);
        bb_succ_off.push(0);

        for &(s, e) in &bb_bounds {
            let mut gr = [0u64; W];
            let mut kr = [0u64; W];
            let mut gu = 0u64;
            let mut ku = 0u64;
            for i in (s..e).rev() {
                let kir = if (flags[i] & FLAG_KILL) != 0 {
                    writes_r[i]
                } else {
                    [0u64; W]
                };
                gr = bits_or(&reads_r[i], &bits_andnot(&gr, &kir));
                kr = bits_or(&kr, &kir);
                if track_u {
                    let kiu = if (flags[i] & FLAG_KILL) != 0 { writes_u[i] } else { 0 };
                    gu = reads_u[i] | (gu & !kiu);
                    ku |= kiu;
                }
            }
            bb_gen_r.push(gr);
            bb_kill_r.push(kr);
            if track_u {
                bb_gen_u.push(gu);
                bb_kill_u.push(ku);
            }
            let last = e - 1;
            bb_has_leaf.push((flags[last] & FLAG_INEXACT_RET) != 0);
            for &t in flow.succ(last) {
                bb_succ_data.push(bb_of[t as usize]);
            }
            bb_succ_off.push(bb_succ_data.len() as u32);
        }

        let mut bb_live_in_r = bb_gen_r.clone();
        let mut bb_live_in_u = bb_gen_u.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..num_bbs).rev() {
                let (mut out_r, mut out_u) = if bb_has_leaf[b] {
                    (every_r, every_u)
                } else {
                    ([0u64; W], 0u64)
                };
                let so = bb_succ_off[b] as usize;
                let eo = bb_succ_off[b + 1] as usize;
                for &sb in &bb_succ_data[so..eo] {
                    out_r = bits_or(&out_r, &bb_live_in_r[sb]);
                    if track_u {
                        out_u |= bb_live_in_u[sb];
                    }
                }
                let new_r = bits_or(&bb_gen_r[b], &bits_andnot(&out_r, &bb_kill_r[b]));
                if track_u {
                    let new_u = bb_gen_u[b] | (out_u & !bb_kill_u[b]);
                    if new_r != bb_live_in_r[b] || new_u != bb_live_in_u[b] {
                        bb_live_in_r[b] = new_r;
                        bb_live_in_u[b] = new_u;
                        changed = true;
                    }
                } else if new_r != bb_live_in_r[b] {
                    bb_live_in_r[b] = new_r;
                    changed = true;
                }
            }
        }

        if track_u {
            bb_busy_r.resize(num_bbs, [0u64; W]);
            bb_any_u.resize(num_bbs, 0u64);
            bb_whole_u.resize(num_bbs, 0u64);
        }

        for (b, &(s, e)) in bb_bounds.iter().enumerate() {
            let (mut cur_r, mut cur_u) = if bb_has_leaf[b] {
                (every_r, every_u)
            } else {
                ([0u64; W], 0u64)
            };
            let so = bb_succ_off[b] as usize;
            let eo = bb_succ_off[b + 1] as usize;
            for &sb in &bb_succ_data[so..eo] {
                cur_r = bits_or(&cur_r, &bb_live_in_r[sb]);
                if track_u {
                    cur_u |= bb_live_in_u[sb];
                }
            }
            let mut b_busy_acc = [0u64; W];
            let mut b_any = 0u64;
            let mut b_whole = !0u64;
            for i in (s..e).rev() {
                if !live_out_u.is_empty() {
                    live_out_u[i] = cur_u;
                }
                let kir = if (flags[i] & FLAG_KILL) != 0 {
                    writes_r[i]
                } else {
                    [0u64; W]
                };
                let in_r = bits_or(&reads_r[i], &bits_andnot(&cur_r, &kir));
                if !exact_targets.is_empty() {
                    live_in_r[i] = in_r;
                }
                let br = bits_or(&bits_or(&in_r, &cur_r), &bits_or(&reads_r[i], &writes_r[i]));
                busy[i] = br;
                cur_r = in_r;
                if track_u {
                    let kiu = if (flags[i] & FLAG_KILL) != 0 { writes_u[i] } else { 0 };
                    let in_u = reads_u[i] | (cur_u & !kiu);
                    live_in_u[i] = in_u;
                    let bu = in_u | cur_u | reads_u[i] | writes_u[i];
                    busy_u[i] = bu;
                    b_busy_acc = bits_or(&b_busy_acc, &br);
                    b_any |= bu;
                    if i + 1 < e {
                        b_whole &= live_in_u[i + 1];
                    } else {
                        b_whole &= bu;
                    }
                    cur_u = in_u;
                }
            }
            if track_u {
                bb_busy_r[b] = bits_or(&b_busy_acc, &never_w);
                bb_any_u[b] = b_any;
                bb_whole_u[b] = b_whole;
            }
        }
    }

    if !exact_targets.is_empty() {
        let mut leaves_r_map: HashMap<usize, [u64; W]> = HashMap::new();
        let mut leaves_u_map: HashMap<usize, u64> = HashMap::new();
        let mut exact_body_nodes: BTreeSet<usize> = BTreeSet::new();
        for &target in &exact_targets {
            let mut exit_r = [0u64; W];
            let mut exit_u = 0u64;
            for &c in &target_callers[&target] {
                let mut c_out_r = if (flags[c] & FLAG_INEXACT_RET) != 0 {
                    every_r
                } else {
                    [0u64; W]
                };
                let mut c_out_u = if (flags[c] & FLAG_INEXACT_RET) != 0 {
                    every_u
                } else {
                    0u64
                };
                for &t in flow.succ(c) {
                    let t = t as usize;
                    c_out_r = bits_or(&c_out_r, &live_in_r[t]);
                    c_out_u |= live_in_u[t];
                }
                exit_r = bits_or(&exit_r, &c_out_r);
                exit_u |= c_out_u;
            }
            for &k in &callee_bodies[&target] {
                exact_body_nodes.insert(k);
                if fd.src[k].mnemonic == "RET" {
                    let er = leaves_r_map.entry(k).or_insert_with(|| [0u64; W]);
                    *er = bits_or(er, &exit_r);
                    *leaves_u_map.entry(k).or_insert(0) |= exit_u;
                }
            }
        }
        let exact_vec: Vec<usize> = exact_body_nodes.into_iter().collect();
        let mut changed_c = true;
        while changed_c {
            changed_c = false;
            for &k in exact_vec.iter().rev() {
                let mut out_r = leaves_r_map.get(&k).copied().unwrap_or([0u64; W]);
                let mut out_u = leaves_u_map.get(&k).copied().unwrap_or(0);
                if (flags[k] & FLAG_INEXACT_RET) != 0 {
                    out_r = bits_or(&out_r, &every_r);
                    out_u |= every_u;
                }
                for &t in flow.succ(k) {
                    let t = t as usize;
                    out_r = bits_or(&out_r, &live_in_r[t]);
                    out_u |= live_in_u[t];
                }
                let (kir, kiu) = if (flags[k] & FLAG_KILL) != 0 {
                    (writes_r[k], writes_u[k])
                } else {
                    ([0u64; W], 0)
                };
                let new_r = bits_or(&reads_r[k], &bits_andnot(&out_r, &kir));
                let new_u = reads_u[k] | (out_u & !kiu);
                if new_r != live_in_r[k] || new_u != live_in_u[k] {
                    live_in_r[k] = new_r;
                    live_in_u[k] = new_u;
                    changed_c = true;
                }
            }
        }
        for &k in &exact_vec {
            let mut out_r = leaves_r_map.get(&k).copied().unwrap_or([0u64; W]);
            let mut out_u = leaves_u_map.get(&k).copied().unwrap_or(0);
            if (flags[k] & FLAG_INEXACT_RET) != 0 {
                out_r = bits_or(&out_r, &every_r);
                out_u |= every_u;
            }
            for &t in flow.succ(k) {
                let t = t as usize;
                out_r = bits_or(&out_r, &live_in_r[t]);
                out_u |= live_in_u[t];
            }
            live_out_u[k] = out_u;
            let br = bits_or(&bits_or(&live_in_r[k], &out_r), &bits_or(&reads_r[k], &writes_r[k]));
            let bu = live_in_u[k] | out_u | reads_u[k] | writes_u[k];
            busy[k] = br;
            busy_u[k] = bu;
            let bb = bb_of[k];
            bb_busy_r[bb] = bits_or(&bb_busy_r[bb], &br);
            bb_any_u[bb] |= bu;
        }
    }

    drop(reads_r);
    drop(writes_r);

    for b_item in &mut busy {
        *b_item = bits_or(b_item, &never_w);
    }

    // BB-accelerated SSA uniform value-flow webs & coupling units.
    let mut webs: Vec<Web> = Vec::new();
    let mut units: Vec<Unit> = Vec::new();
    let active_u_mask = every_u & ((1u64 << 63) - 1);
    if active_u_mask != 0 && !fd.uniform.is_empty() && n > 0 {
        if flow.has_calls {
            bb_whole_u.fill(0);
        }
        let return_sites: Vec<usize> = if flow.has_calls {
            (0..n)
                .filter(|&j| fd.src[j].name.starts_with("call_rel") && j + 1 < n)
                .map(|j| j + 1)
                .collect()
        } else {
            Vec::new()
        };
        let has_pairs = !pair_sites.is_empty();
        let mut is_pair_insn: Vec<bool> = if has_pairs { vec![false; n] } else { Vec::new() };
        for &(pi, _) in &pair_sites {
            is_pair_insn[pi] = true;
        }
        let mut web_of_pair_insn: HashMap<(usize, u32), usize> = if has_pairs {
            HashMap::with_capacity(pair_sites.len() * 2)
        } else {
            HashMap::new()
        };

        let mut u_list: Vec<u32> = fd
            .uniform
            .iter()
            .copied()
            .filter(|&u| u < 63 && (active_u_mask >> u) & 1 != 0)
            .collect();
        u_list.sort_unstable();
        u_list.dedup();

        let num_bbs = bb_bounds.len();
        let mut parent: Vec<usize> = vec![0usize; n];
        let mut root_to_wid: Vec<u32> = Vec::new();
        let mut touched_roots: Vec<usize> = Vec::new();
        let mut active_bbs: Vec<usize> = Vec::with_capacity(num_bbs);

        for &u in &u_list {
            let bit = 1u64 << u;
            active_bbs.clear();
            let mut components = 0usize;
            for (b, (&any_u, &whole_u)) in bb_any_u.iter().zip(&bb_whole_u).enumerate() {
                if (any_u & bit) == 0 {
                    continue;
                }
                active_bbs.push(b);
                let (s, e) = bb_bounds[b];
                if (whole_u & bit) != 0 {
                    parent[s] = s;
                    components += 1;
                } else {
                    for (i, &bu) in busy_u[s..e].iter().enumerate() {
                        if (bu & bit) != 0 {
                            parent[s + i] = s + i;
                            components += 1;
                        }
                    }
                }
            }
            if active_bbs.is_empty() {
                continue;
            }
            for &b in &active_bbs {
                let (s, e) = bb_bounds[b];
                if (bb_whole_u[b] & bit) != 0 {
                    for &t in flow.succ(e - 1) {
                        let t = t as usize;
                        if (live_in_u[t] & bit) != 0 && uf_union(&mut parent, s, t) {
                            components -= 1;
                        }
                    }
                } else {
                    for i in s..e {
                        if (busy_u[i] & bit) == 0 {
                            continue;
                        }
                        if i + 1 < e {
                            if (live_in_u[i + 1] & bit) != 0 && uf_union(&mut parent, i, i + 1) {
                                components -= 1;
                            }
                        } else {
                            for &t in flow.succ(i) {
                                let t = t as usize;
                                if (live_in_u[t] & bit) != 0 && uf_union(&mut parent, i, t) {
                                    components -= 1;
                                }
                            }
                        }
                        if flow.has_calls && fd.src[i].name == "ret__REL" && (flags[i] & FLAG_INEXACT_RET) != 0 {
                            for &t in &return_sites {
                                if (live_in_u[t] & bit) != 0 && uf_union(&mut parent, i, t) {
                                    components -= 1;
                                }
                            }
                        }
                        if flow.is_call[i]
                            && flow.called.get(&i).is_some_and(Option::is_some)
                            && let Some(entry) = flow.named[i]
                        {
                            if (busy_u[i] & bit) != 0
                                && (live_in_u[entry] & bit) != 0
                                && uf_union(&mut parent, i, entry)
                            {
                                components -= 1;
                            }
                            if ((writes_u[i] & live_out_u[i]) & bit) != 0 {
                                for &k in &callee_bodies[&entry] {
                                    if (raw_writes_u[k] & bit) != 0 && uf_union(&mut parent, i, k) {
                                        components -= 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if components == 1 {
                let wid = webs.len();
                let mut acc = [0u64; W];
                let mut span: Vec<usize> = Vec::new();
                for &b in &active_bbs {
                    let (s, e) = bb_bounds[b];
                    if (bb_whole_u[b] & bit) != 0 {
                        span.extend(s..e);
                        acc = bits_or(&acc, &bb_busy_r[b]);
                    } else {
                        for i in s..e {
                            if (busy_u[i] & bit) != 0 {
                                span.push(i);
                                acc = bits_or(&acc, &busy[i]);
                            }
                        }
                    }
                }
                if has_pairs {
                    for &i in &span {
                        if is_pair_insn[i] {
                            web_of_pair_insn.insert((i, u), wid);
                        }
                    }
                }
                webs.push(Web {
                    u,
                    span,
                    busy: bits_to_regset(&acc),
                });
            } else {
                if root_to_wid.is_empty() {
                    root_to_wid.resize(n, u32::MAX);
                }
                touched_roots.clear();
                let mut local_accs: Vec<[u64; W]> = Vec::new();
                let base_wid = webs.len();
                for &b in &active_bbs {
                    let (s, e) = bb_bounds[b];
                    if (bb_whole_u[b] & bit) != 0 {
                        let r = uf_find(&mut parent, s);
                        let mut wid = root_to_wid[r];
                        if wid == u32::MAX {
                            wid = webs.len() as u32;
                            root_to_wid[r] = wid;
                            touched_roots.push(r);
                            webs.push(Web {
                                u,
                                span: Vec::new(),
                                busy: regset_empty(),
                            });
                            local_accs.push([0u64; W]);
                        }
                        let idx = wid as usize;
                        webs[idx].span.extend(s..e);
                        local_accs[idx - base_wid] = bits_or(&local_accs[idx - base_wid], &bb_busy_r[b]);
                        if has_pairs {
                            for (off, &isp) in is_pair_insn[s..e].iter().enumerate() {
                                if isp {
                                    web_of_pair_insn.insert((s + off, u), idx);
                                }
                            }
                        }
                    } else {
                        for i in s..e {
                            if (busy_u[i] & bit) == 0 {
                                continue;
                            }
                            let r = uf_find(&mut parent, i);
                            let mut wid = root_to_wid[r];
                            if wid == u32::MAX {
                                wid = webs.len() as u32;
                                root_to_wid[r] = wid;
                                touched_roots.push(r);
                                webs.push(Web {
                                    u,
                                    span: Vec::new(),
                                    busy: regset_empty(),
                                });
                                local_accs.push([0u64; W]);
                            }
                            let idx = wid as usize;
                            webs[idx].span.push(i);
                            local_accs[idx - base_wid] = bits_or(&local_accs[idx - base_wid], &busy[i]);
                            if has_pairs && is_pair_insn[i] {
                                web_of_pair_insn.insert((i, u), idx);
                            }
                        }
                    }
                }
                for (off, acc) in local_accs.iter().enumerate() {
                    webs[base_wid + off].busy = bits_to_regset(acc);
                }
                for &r in &touched_roots {
                    root_to_wid[r] = u32::MAX;
                }
            }
        }

        let nw = webs.len();
        if pair_sites.is_empty() {
            units.reserve(nw);
            for (wid, w) in webs.iter().enumerate() {
                units.push(Unit {
                    min_u: w.u,
                    width: 1,
                    total_span_len: w.span.len(),
                    first_insn: w.span[0],
                    members: vec![wid],
                    pair_firsts: Vec::new(),
                    valid: true,
                });
            }
        } else {
            let mut unit_parent: Vec<usize> = (0..nw).collect();
            for &(i, first) in &pair_sites {
                if let (Some(&w0), Some(&w1)) =
                    (web_of_pair_insn.get(&(i, first)), web_of_pair_insn.get(&(i, first + 1)))
                {
                    uf_union(&mut unit_parent, w0, w1);
                }
            }
            let mut root_to_uid = vec![u32::MAX; nw];
            let mut unit_members: Vec<Vec<usize>> = Vec::new();
            let mut unit_pair_firsts: Vec<Vec<u32>> = Vec::new();
            for wid in 0..nw {
                let r = uf_find(&mut unit_parent, wid);
                let uid = if root_to_uid[r] == u32::MAX {
                    let id = unit_members.len() as u32;
                    root_to_uid[r] = id;
                    unit_members.push(Vec::new());
                    unit_pair_firsts.push(Vec::new());
                    id
                } else {
                    root_to_uid[r]
                };
                unit_members[uid as usize].push(wid);
            }
            for &(i, first) in &pair_sites {
                if let Some(&w0) = web_of_pair_insn.get(&(i, first)) {
                    let r = uf_find(&mut unit_parent, w0);
                    let uid = root_to_uid[r] as usize;
                    unit_pair_firsts[uid].push(first);
                }
            }
            let mut seen_stamp: Vec<u32> = Vec::new();
            let mut stamp = 0u32;
            units.reserve(unit_members.len());
            for (members, mut pfirsts) in unit_members.into_iter().zip(unit_pair_firsts) {
                let min_u = members.iter().map(|&w| webs[w].u).min().unwrap();
                let max_u = members.iter().map(|&w| webs[w].u).max().unwrap();
                let width = max_u - min_u + 1;
                let first_insn = members.iter().map(|&w| webs[w].span[0]).min().unwrap_or(0);
                let total_span_len = if members.len() == 1 {
                    webs[members[0]].span.len()
                } else {
                    if seen_stamp.is_empty() {
                        seen_stamp.resize(n, 0);
                    }
                    stamp += 1;
                    let mut cnt = 0usize;
                    for &w in &members {
                        for &i in &webs[w].span {
                            if seen_stamp[i] != stamp {
                                seen_stamp[i] = stamp;
                                cnt += 1;
                            }
                        }
                    }
                    cnt
                };
                pfirsts.sort_unstable();
                pfirsts.dedup();
                let valid = width == 1 || pfirsts.iter().all(|&f| (f - min_u).is_multiple_of(2));
                units.push(Unit {
                    min_u,
                    width,
                    total_span_len,
                    first_insn,
                    members,
                    pair_firsts: pfirsts,
                    valid,
                });
            }
        }
        units.sort_by(|a, b| {
            b.total_span_len
                .cmp(&a.total_span_len)
                .then(b.width.cmp(&a.width))
                .then(a.min_u.cmp(&b.min_u))
                .then(a.first_insn.cmp(&b.first_insn))
        });
    }

    BorrowPrep {
        n,
        highest: fd.highest,
        flow,
        busy: wrap_busy(busy),
        touch_opt,
        never: bits_to_regset(&never_w),
        late_read_counts,
        board_costs,
        webs,
        units,
    }
}

impl BorrowPrep {
    pub(super) fn new(fd: &FuncData, need_late: bool, need_l3: bool) -> Result<Self, Refused> {
        let flow = BorrowFlow::new(&fd.src, &fd.tables);
        if !flow.known {
            return Err(Refused("the ways through the function are not known".into()));
        }
        let prep = if fd.highest < 64 {
            build_prep::<1>(fd, flow, need_late, need_l3, BusyVec::W1)
        } else if fd.highest < 128 {
            build_prep::<2>(fd, flow, need_late, need_l3, BusyVec::W2)
        } else {
            build_prep::<4>(fd, flow, need_late, need_l3, BusyVec::W4)
        };
        Ok(prep)
    }

    pub(super) fn solve_homes(&self, all_in_memory: bool, top_override: Option<u32>) -> HomeSolution {
        let top = match top_override {
            Some(t) => t.max(self.highest),
            None => self.highest,
        };
        let mut taken: Vec<RegSet> = if self.units.is_empty() {
            Vec::new()
        } else {
            vec![regset_empty(); self.n]
        };
        let mut unit_reg: Vec<Option<u32>> = vec![None; self.units.len()];
        let mut unhomed_u: HashSet<u32> = HashSet::new();
        let mut unhomed_pairs: HashSet<u32> = HashSet::new();
        let mut any_taken = regset_empty();
        let mut max_borrowed = 0u32;

        let mark_unhomed = |unit: &Unit, unhomed_u: &mut HashSet<u32>, unhomed_pairs: &mut HashSet<u32>| {
            for &wid in &unit.members {
                unhomed_u.insert(self.webs[wid].u);
            }
            for &f in &unit.pair_firsts {
                unhomed_pairs.insert(f);
                unhomed_u.insert(f);
                unhomed_u.insert(f + 1);
            }
        };

        if all_in_memory {
            for unit in &self.units {
                mark_unhomed(unit, &mut unhomed_u, &mut unhomed_pairs);
            }
        } else {
            for (uid, unit) in self.units.iter().enumerate() {
                if !unit.valid || unit.width > top {
                    mark_unhomed(unit, &mut unhomed_u, &mut unhomed_pairs);
                    continue;
                }
                let mut small_buf = [regset_empty(); 4];
                let mut heap_buf: Vec<RegSet> = Vec::new();
                let union_by_off: &mut [RegSet] = if unit.width <= 4 {
                    &mut small_buf[..unit.width as usize]
                } else {
                    heap_buf.resize(unit.width as usize, regset_empty());
                    &mut heap_buf
                };
                for &wid in &unit.members {
                    let w = &self.webs[wid];
                    let off = (w.u - unit.min_u) as usize;
                    union_by_off[off] = regset_or(&union_by_off[off], &w.busy);
                    let mut rem = regset_andnot(&any_taken, &union_by_off[off]);
                    if !regset_is_zero(&rem) {
                        for &i in &w.span {
                            let ti = taken[i];
                            if !regset_disjoint(&ti, &rem) {
                                union_by_off[off] = regset_or(&union_by_off[off], &ti);
                                rem = regset_andnot(&rem, &ti);
                                if regset_is_zero(&rem) {
                                    break;
                                }
                            }
                        }
                    }
                }
                let step = if unit.width == 1 { 1u32 } else { 2 };
                let upper = top + 2 - unit.width;
                let mut r = 2u32;
                let found = loop {
                    if r >= upper {
                        break None;
                    }
                    let all_clear = (0..unit.width).all(|off| !regset_test(&union_by_off[off as usize], r + off));
                    if all_clear {
                        break Some(r);
                    }
                    r += step;
                };
                if let Some(r) = found {
                    unit_reg[uid] = Some(r);
                    for &wid in &unit.members {
                        let w = &self.webs[wid];
                        let reg = r + (w.u - unit.min_u);
                        regset_set(&mut any_taken, reg);
                        max_borrowed = max_borrowed.max(reg);
                        for &i in &w.span {
                            regset_set(&mut taken[i], reg);
                        }
                    }
                } else {
                    mark_unhomed(unit, &mut unhomed_u, &mut unhomed_pairs);
                }
            }
        }

        HomeSolution {
            top,
            max_borrowed,
            unit_reg,
            taken,
            unhomed_u,
            unhomed_pairs,
        }
    }

    pub(super) fn into_borrowed(self, fd: &FuncData, sol: HomeSolution, need_l3: bool) -> Borrowed {
        let n = self.n;
        let busy = self.busy.into_regsets();
        let taken = if sol.taken.is_empty() {
            vec![regset_empty(); n]
        } else {
            sol.taken
        };
        let any_homed = sol.unit_reg.iter().any(Option::is_some);
        let mut home_at_insn: Vec<Vec<(u32, u32)>> = if any_homed { vec![Vec::new(); n] } else { Vec::new() };
        let mut web_homes: Vec<(u32, u32, Vec<usize>)> = Vec::new();
        let mut used: HashSet<u32> = HashSet::new();
        if any_homed {
            for (uid, unit) in self.units.iter().enumerate() {
                if let Some(r) = sol.unit_reg[uid] {
                    for &wid in &unit.members {
                        let w = &self.webs[wid];
                        let reg = r + (w.u - unit.min_u);
                        web_homes.push((w.u, reg, w.span.clone()));
                        for &i in &w.span {
                            home_at_insn[i].push((w.u, reg));
                        }
                        used.insert(reg);
                    }
                }
            }
        }
        web_homes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for entry in &mut home_at_insn {
            entry.sort_unstable_by_key(|&(u, _)| u);
        }

        let touch = if need_l3 {
            self.touch_opt.unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut depth: Vec<Option<i32>> = Vec::new();
        let mut r1_fixed = false;
        if need_l3 {
            depth = vec![None; n];
            r1_fixed = n > 1;
            if n > 1 {
                let mut work: Vec<(usize, i32)> = self.flow.succ(0).iter().map(|&t| (t as usize, 0i32)).collect();
                while let Some((i, at)) = work.pop() {
                    if !r1_fixed {
                        break;
                    }
                    if let Some(existing) = depth[i] {
                        if existing != at {
                            r1_fixed = false;
                        }
                        continue;
                    }
                    depth[i] = Some(at);
                    let mut after = at;
                    if fd.src[i].writes.iter().any(|&(l, r)| l == LETTER_R && r == 1) {
                        let d = &fd.src[i];
                        if d.name == "iadd3_imm__RsIR_RIR"
                            && d.op_or(crate::generated::fields::OP_RD, 255) == 1
                            && d.op_or(crate::generated::fields::OP_RA, 255) == 1
                            && d.op_or(crate::generated::fields::OP_RC, 255) == 255
                            && unguarded(d)
                            && d.op_or(crate::generated::fields::OP_RA_AT_NEGATE, 0) == 0
                            && d.op_or(crate::generated::fields::OP_RC_AT_NEGATE, 0) == 0
                        {
                            let sb = d.op_or(crate::generated::fields::OP_SB, 0);
                            let imm = sb as i32;
                            after = at - imm;
                        } else {
                            r1_fixed = false;
                        }
                    }
                    for &t in self.flow.succ(i) {
                        work.push((t as usize, after));
                    }
                    if self.flow.is_call[i]
                        && self.flow.called.get(&i).is_some_and(Option::is_some)
                        && let Some(Some(entry)) = self.flow.named.get(i).copied()
                    {
                        work.push((entry, at));
                    }
                }
            }
        }

        Borrowed {
            top: sol.top,
            highest: self.highest,
            busy,
            taken,
            touch,
            never: self.never,
            late_read_counts: self.late_read_counts,
            board_costs: self.board_costs,
            home_at_insn,
            web_homes,
            unhomed_u: sol.unhomed_u,
            unhomed_pairs: sol.unhomed_pairs,
            used,
            everything: false,
            slots: HashMap::new(),
            predicate_slots: HashMap::new(),
            frame_bytes: 0,
            depth,
            r1_fixed,
            room: false,
            from_r1: false,
            spill: 0,
        }
    }
}

impl Borrowed {
    pub(super) fn new(fd: &FuncData, all_in_memory: bool, top_override: Option<u32>) -> Result<Self, Refused> {
        let prep = BorrowPrep::new(fd, true, true)?;
        let sol = prep.solve_homes(all_in_memory, top_override);
        Ok(prep.into_borrowed(fd, sol, true))
    }

    pub(super) fn home_at(&self, i: usize, u: u32) -> Option<u32> {
        self.home_at_insn
            .get(i)?
            .binary_search_by_key(&u, |&(reg_u, _)| reg_u)
            .ok()
            .map(|pos| self.home_at_insn[i][pos].1)
    }

    /// A register (or even-aligned run of `width`) holding nothing at instruction `i`, capped at
    /// `highest.min(top)` when borrowing into program registers (levels 1/2) or `top` at level 3.
    /// Prefers registers with zero/fewest scoreboard waits (`board_costs`) and unannounced late
    /// reads (`late_read_counts`) across the function (returning early on 0).
    pub(super) fn free_at(&self, i: usize, width: u32, given: &RegSet) -> Option<u32> {
        let held = regset_or(&regset_or(&self.busy[i], &self.taken[i]), given);
        let cap = if self.everything {
            self.top
        } else {
            self.highest.min(self.top)
        };
        if cap + 2 <= width {
            return None;
        }
        let upper = cap + 2 - width;
        let step = if width == 1 { 1u32 } else { 2 };
        let mut best: Option<(u32, u32)> = None;
        let mut r = 2u32;
        while r < upper {
            if (0..width).all(|k| !regset_test(&held, r + k)) {
                let cost: u32 = (0..width)
                    .map(|k| {
                        let idx = (r + k) as usize;
                        self.late_read_counts[idx] + self.board_costs[idx]
                    })
                    .sum();
                if cost == 0 {
                    return Some(r);
                }
                if best.is_none_or(|(bc, _)| cost < bc) {
                    best = Some((cost, r));
                }
            }
            r += step;
        }
        best.map(|(_, r)| r)
    }

    /// Where no register holds nothing: the lowest register, or run of `width` from an even
    /// one, that instruction `i` neither reads nor writes itself, that is no home there, was
    /// not given already, and receives no result that nothing announces: registers that hold
    /// something for the program, to be put in the frame for the time of the sequence and
    /// brought back after it (`Ctx::make_room`). None when there is none.
    pub(super) fn victims_at(&self, i: usize, width: u32, given: &RegSet) -> Option<u32> {
        let mut held = regset_or(&regset_or(&self.touch[i], &self.taken[i]), given);
        held = regset_or(&held, &self.never);
        regset_set(&mut held, 0);
        regset_set(&mut held, 1);
        let upper = self.highest + 2 - width;
        let step = if width == 1 { 1u32 } else { 2 };
        let mut r = 2u32;
        loop {
            if r >= upper {
                return None;
            }
            let all_clear = (0..width).all(|k| !regset_test(&held, r + k));
            if all_clear {
                return Some(r);
            }
            r += step;
        }
    }

    /// A register that holds nothing at any instruction of `span` and is no home there:
    /// taken through all of it. For a uniform predicate held in a register between the
    /// instructions that name it, at level 3. None when there is none.
    pub(super) fn kept(&mut self, span: &[usize]) -> Option<u32> {
        let mut union = regset_empty();
        for &i in span {
            union = regset_or(&union, &regset_or(&self.busy[i], &self.taken[i]));
        }
        for r in 2..=self.top {
            if !regset_test(&union, r) {
                for &i in span {
                    regset_set(&mut self.taken[i], r);
                }
                self.used.insert(r);
                return Some(r);
            }
        }
        None
    }

    /// A word of the frame for each uniform register that has unhomed webs (the two of a pair
    /// side by side, the lower one first). Returns the size of the frame in bytes.
    pub(super) fn assign_slots(&mut self, uniform: &[u32], pairs: &HashSet<u32>) -> u32 {
        let _ = (uniform, pairs);
        let mut list: Vec<u32> = self.unhomed_u.iter().copied().collect();
        list.sort_unstable();
        let mut slot = 0u32;
        for &u in &list {
            if u > 0 && self.unhomed_pairs.contains(&(u - 1)) {
                continue;
            }
            if self.unhomed_pairs.contains(&u) {
                self.slots.insert(u, slot);
                self.slots.insert(u + 1, slot + 1);
                slot += 2;
            } else {
                self.slots.insert(u, slot);
                slot += 1;
            }
        }
        self.frame_bytes = 4 * slot;
        if !self.frame_bytes.is_multiple_of(16) {
            self.frame_bytes = (self.frame_bytes + 15) & !15;
        }
        self.frame_bytes
    }

    /// Where frame word `word` is, from the top of the thread's local memory (what the stack
    /// pointer starts at): the frame is right under it.
    pub(super) fn slot_offset(&self, word: u32) -> i64 {
        4 * i64::from(word) - i64::from(self.frame_bytes)
    }

    pub(super) fn slot_count(&self) -> u32 {
        (self.slots.len() + self.predicate_slots.len()) as u32
    }
}
