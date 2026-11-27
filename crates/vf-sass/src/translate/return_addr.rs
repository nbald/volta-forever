//! Return addresses: find the MOV instructions in the translated code that hold return
//! addresses or jump-table entries, and rewrite them once the new offsets are known.

use std::collections::{HashMap, VecDeque};

use crate::decode::{Decoded, Kind, LETTER_R};
use crate::generated::fields::*;

use super::parse::FuncData;
use super::{Item, Refused, hex};

// ---- Return addresses ----

#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct RegSet([u128; 2]);

impl RegSet {
    const ALL_255: Self = Self([u128::MAX, (1u128 << 127) - 1]);

    #[inline]
    fn contains(self, r: u32) -> bool {
        if r >= 255 {
            return false;
        }
        let idx = (r >> 7) as usize;
        ((self.0[idx] >> (r & 127)) & 1) != 0
    }

    #[inline]
    fn insert(&mut self, r: u32) {
        if r < 255 {
            let idx = (r >> 7) as usize;
            self.0[idx] |= 1u128 << (r & 127);
        }
    }

    #[inline]
    fn union(self, other: Self) -> Self {
        Self([self.0[0] | other.0[0], self.0[1] | other.0[1]])
    }

    #[inline]
    fn intersect(self, other: Self) -> Self {
        Self([self.0[0] & other.0[0], self.0[1] & other.0[1]])
    }

    #[inline]
    fn diff(self, other: Self) -> Self {
        Self([self.0[0] & !other.0[0], self.0[1] & !other.0[1]])
    }

    #[inline]
    fn for_each(self, mut f: impl FnMut(u32)) {
        for (word_idx, &(mut bits)) in self.0.iter().enumerate() {
            let base = (word_idx as u32) << 7;
            while bits != 0 {
                let tz = bits.trailing_zeros();
                f(base + tz);
                bits &= bits - 1;
            }
        }
    }
}

#[inline]
fn unguarded(d: &Decoded) -> bool {
    d.op_or(OP_PG, 7) == 7 && d.op_or(OP_PG_AT_NOT, 0) == 0 && d.op_or(OP_UPG, 7) == 7 && d.op_or(OP_UPG_AT_NOT, 0) == 0
}

#[inline]
fn copy_src(d: &Decoded) -> Option<(u32, u32)> {
    if !unguarded(d) {
        return None;
    }
    let mut r_write = None;
    for &(l, r) in &d.writes {
        if l == LETTER_R && r < 255 {
            if r_write.is_some() {
                return None;
            }
            r_write = Some(r);
        }
    }
    let single_dst = r_write?;
    if d.name == "mov__RRR_RR" && d.op_or_name("PixMaskU04", 15) == 15 {
        let rb = d.op_or(OP_RB, 255);
        let rd = d.op_or(OP_RD, 255);
        if rb != 255 && rd == single_dst {
            return Some((rd, rb));
        }
    } else if d.name == "imad__RRR_RRR"
        && d.op_or(OP_RA, 255) == 255
        && d.op_or(OP_RB, 255) == 255
        && d.op_or(OP_RC_AT_NEGATE, 0) == 0
        && d.op_or_name("X", 0) == 0
    {
        let rc = d.op_or(OP_RC, 255);
        let rd = d.op_or(OP_RD, 255);
        if rc != 255 && rd == single_dst {
            return Some((rd, rc));
        }
    }
    None
}

fn room_tag_reg(item: &Item) -> Option<(u32, bool)> {
    let is_away = match item.what {
        "room: away" => true,
        "room: back" => false,
        _ => return None,
    };
    if let Some(detail) = item.what_detail.as_deref()
        && let Some(rest) = detail.strip_prefix("room: R")
    {
        let suffix = if is_away { " away" } else { " back" };
        if let Some(num) = rest.strip_suffix(suffix)
            && let Ok(r) = num.parse::<u32>()
        {
            return Some((r, is_away));
        }
    }
    let d = item.d.as_ref()?;
    let r = if is_away { d.op(OP_RB)? } else { d.op(OP_RD)? };
    (r < 255).then_some((r, is_away))
}

fn compute_room_regs(items: &[Item]) -> Vec<RegSet> {
    let n = items.len();
    let mut room_regs = vec![RegSet::default(); n];
    let mut away: Vec<(u32, usize)> = Vec::new();
    let mut back: Vec<(u32, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let orig = items[i].origin;
        let mut end = i + 1;
        while end < n && items[end].origin == orig {
            end += 1;
        }
        if end - i > 1 {
            away.clear();
            back.clear();
            for (x, item) in items[i..end].iter().enumerate() {
                if let Some((r, is_away)) = room_tag_reg(item) {
                    if is_away {
                        if !away.iter().any(|&(reg, _)| reg == r) {
                            away.push((r, i + x));
                        }
                    } else if let Some(slot) = back.iter_mut().find(|(reg, _)| *reg == r) {
                        slot.1 = i + x;
                    } else {
                        back.push((r, i + x));
                    }
                }
            }
            for &(r, ia) in &away {
                if let Some(&(_, ib)) = back.iter().find(|&&(reg, _)| reg == r)
                    && ia < ib
                {
                    for reg_set in &mut room_regs[ia..=ib] {
                        reg_set.insert(r);
                    }
                }
            }
        }
        i = end;
    }
    room_regs
}

struct Cfg {
    first_at: Vec<u32>,
    succ_off: Vec<u32>,
    succ_buf: Vec<u32>,
    pred_off: Vec<u32>,
    pred_buf: Vec<u32>,
    room_regs: Vec<RegSet>,
}

impl Cfg {
    #[inline]
    fn first_at(&self, byte_off: u64) -> Option<usize> {
        if (byte_off & 15) != 0 {
            return None;
        }
        let idx = (byte_off >> 4) as usize;
        let v = *self.first_at.get(idx)?;
        (v != u32::MAX).then_some(v as usize)
    }

    #[inline]
    fn succ(&self, i: usize) -> &[u32] {
        let s = self.succ_off[i] as usize;
        let e = self.succ_off[i + 1] as usize;
        &self.succ_buf[s..e]
    }

    #[inline]
    fn preds(&self, i: usize) -> &[u32] {
        let s = self.pred_off[i] as usize;
        let e = self.pred_off[i + 1] as usize;
        &self.pred_buf[s..e]
    }
}

struct WalkScratch {
    vis_stamp: Vec<u32>,
    stamp: u32,
    defs_stamp: Vec<u32>,
    d_stamp: u32,
    body_stamp: Vec<u32>,
    local_id: Vec<u32>,
    b_stamp: u32,
    work: Vec<u32>,
    stack: Vec<usize>,
}

impl WalkScratch {
    fn new(n: usize) -> Self {
        Self {
            vis_stamp: vec![0; n],
            stamp: 0,
            defs_stamp: vec![0; n],
            d_stamp: 0,
            body_stamp: vec![0; n],
            local_id: vec![0; n],
            b_stamp: 0,
            work: Vec::with_capacity(64),
            stack: Vec::with_capacity(8),
        }
    }

    #[inline]
    fn next_vis(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.vis_stamp.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }

    #[inline]
    fn next_defs(&mut self) -> u32 {
        self.d_stamp = self.d_stamp.wrapping_add(1);
        if self.d_stamp == 0 {
            self.defs_stamp.fill(0);
            self.d_stamp = 1;
        }
        self.d_stamp
    }

    #[inline]
    fn next_body(&mut self) -> u32 {
        self.b_stamp = self.b_stamp.wrapping_add(1);
        if self.b_stamp == 0 {
            self.body_stamp.fill(0);
            self.b_stamp = 1;
        }
        self.b_stamp
    }
}

#[inline]
fn is_leave_mnemonic(m: &str) -> bool {
    matches!(m, "BREAK" | "BRK" | "CONT" | "PEXIT" | "PRET" | "PLONGJMP" | "LONGJMP")
}

#[inline]
fn is_barrier_init_mnemonic(m: &str) -> bool {
    matches!(m, "BSSY" | "PBK" | "SSY" | "PCNT")
}

#[inline]
fn lookup_first(first_at: &[u32], byte_off: u64) -> Option<usize> {
    if (byte_off & 15) != 0 {
        return None;
    }
    let v = *first_at.get((byte_off >> 4) as usize)?;
    (v != u32::MAX).then_some(v as usize)
}

#[derive(Clone, Copy)]
struct BssyInfo {
    bar_slot: u16,
    meet_id: u32,
    unguarded_with_bar: bool,
}

fn build_cfg(items: &[Item], fd: &FuncData) -> Cfg {
    let n = items.len();
    let mut first_at = vec![u32::MAX; fd.src.len()];
    for (idx, it) in items.iter().enumerate() {
        if it.first && it.origin < first_at.len() {
            first_at[it.origin] = idx as u32;
        }
    }

    let mut base_off: Vec<u32> = Vec::with_capacity(n + 1);
    let mut base_buf: Vec<u32> = Vec::with_capacity(n + n / 8 + 8);
    let mut base_pred_count: Vec<u32> = vec![0; n];
    let mut leaves: Vec<usize> = Vec::new();
    let mut meets: Vec<usize> = Vec::new();
    let mut is_meet: Vec<bool> = vec![false; n];
    let mut bssy_raw: Vec<(usize, Option<u32>, usize, bool)> = Vec::new();
    let mut has_bmov_restore = false;

    for (i, it) in items.iter().enumerate() {
        base_off.push(base_buf.len() as u32);
        let d = it.d.as_ref().unwrap();
        let m = d.mnemonic;
        let ug = unguarded(d);
        if m == "BMOV" && d.op_or_name("clear", 0) == 0 && d.writes.iter().any(|&(l, _)| l == b'B') {
            has_bmov_restore = true;
        }
        let after = (i + 1 < n).then_some(i + 1);
        let push_edge = |buf: &mut Vec<u32>, pc: &mut [u32], tgt: usize| {
            buf.push(tgt as u32);
            pc[tgt] += 1;
        };
        if matches!(m, "EXIT" | "RET") {
            if !ug && let Some(a) = after {
                push_edge(&mut base_buf, &mut base_pred_count, a);
            }
        } else if d.name == "brx_"
            && let Some(places) = fd.table_at(16 * it.origin as u64)
        {
            if !ug && let Some(a) = after {
                push_edge(&mut base_buf, &mut base_pred_count, a);
            }
            for &t in places {
                if let Some(idx) = lookup_first(&first_at, t) {
                    push_edge(&mut base_buf, &mut base_pred_count, idx);
                }
            }
        } else if m == "BRA" {
            let always = ug && d.op_or(OP_PP, 7) == 7 && d.op_or(OP_PP_AT_NOT, 0) == 0 && d.op_or(OP_COND, 0) == 0;
            let t = it.target.and_then(|tgt| lookup_first(&first_at, tgt));
            if !always && let Some(a) = after {
                push_edge(&mut base_buf, &mut base_pred_count, a);
            }
            if let Some(ti) = t {
                push_edge(&mut base_buf, &mut base_pred_count, ti);
            }
        } else if is_barrier_init_mnemonic(m) {
            if let Some(a) = after {
                push_edge(&mut base_buf, &mut base_pred_count, a);
            }
            if let Some(ti) = it.target.and_then(|tgt| lookup_first(&first_at, tgt)) {
                if !is_meet[ti] {
                    is_meet[ti] = true;
                    meets.push(ti);
                }
                let b = if m == "BSSY" {
                    let v = d.op_or_name("barReg", u32::MAX);
                    (v != u32::MAX).then_some(v)
                } else {
                    None
                };
                bssy_raw.push((i, b, ti, ug));
            }
        } else if is_leave_mnemonic(m) {
            if let Some(a) = after {
                push_edge(&mut base_buf, &mut base_pred_count, a);
            }
            leaves.push(i);
        } else if d.name.starts_with("call") || d.name.starts_with("jcal") {
            if let Some(a) = after {
                push_edge(&mut base_buf, &mut base_pred_count, a);
            }
        } else if let Some(tgt) = it.target
            && !matches!(m, "BRX" | "JMX" | "JMP")
        {
            if let Some(a) = after {
                push_edge(&mut base_buf, &mut base_pred_count, a);
            }
            if let Some(ti) = lookup_first(&first_at, tgt) {
                push_edge(&mut base_buf, &mut base_pred_count, ti);
            }
        } else if let Some(a) = after {
            push_edge(&mut base_buf, &mut base_pred_count, a);
        }
    }
    base_off.push(base_buf.len() as u32);

    let (succ_off, succ_buf) = if !leaves.is_empty() && !meets.is_empty() {
        meets.sort_unstable();
        let num_meets = meets.len();
        let words_per_bar = num_meets.div_ceil(64);

        // Assign compact slot indices to distinct barrier keys appearing in bssy_raw.
        let mut bar_keys: Vec<Option<u32>> = Vec::new();
        let mut bssy_at_insn: Vec<Option<BssyInfo>> = vec![None; n];
        for &(i, b, ti, ug) in &bssy_raw {
            let bar_slot = if let Some(pos) = bar_keys.iter().position(|&k| k == b) {
                pos as u16
            } else {
                let pos = bar_keys.len() as u16;
                bar_keys.push(b);
                pos
            };
            let meet_id = meets.binary_search(&ti).unwrap() as u32;
            bssy_at_insn[i] = Some(BssyInfo {
                bar_slot,
                meet_id,
                unguarded_with_bar: ug && b.is_some(),
            });
        }
        let num_bars = bar_keys.len();
        let stride = num_bars * words_per_bar;

        let mut is_leave_insn: Vec<bool> = vec![false; n];
        for &l in &leaves {
            is_leave_insn[l] = true;
        }

        // Partition 0..n into basic blocks where every bssy_at and leave is a 1-instruction block
        // and every meet or control-flow join/split starts a new block.
        let mut is_leader: Vec<bool> = vec![false; n];
        if n > 0 {
            is_leader[0] = true;
        }
        for &m_idx in &meets {
            is_leader[m_idx] = true;
        }
        for i in 0..n {
            if base_pred_count[i] != 1 {
                is_leader[i] = true;
            }
            let s = base_off[i] as usize;
            let e = base_off[i + 1] as usize;
            let nexts = &base_buf[s..e];
            let straight = nexts.len() == 1 && nexts[0] as usize == i + 1;
            if !straight || bssy_at_insn[i].is_some() || is_leave_insn[i] {
                if bssy_at_insn[i].is_some() || is_leave_insn[i] {
                    is_leader[i] = true;
                }
                if i + 1 < n {
                    is_leader[i + 1] = true;
                }
                for &tgt in nexts {
                    is_leader[tgt as usize] = true;
                }
            }
        }

        let mut block_of: Vec<u32> = vec![0; n];
        let mut block_end: Vec<usize> = Vec::new();
        let mut cur_bb: u32 = 0;
        for i in 0..n {
            if i > 0 && is_leader[i] {
                block_end.push(i - 1);
                cur_bb += 1;
            }
            block_of[i] = cur_bb;
        }
        if n > 0 {
            block_end.push(n - 1);
        }
        let num_blocks = block_end.len();

        let meet_blocks: Vec<u32> = meets.iter().map(|&ti| block_of[ti]).collect();

        // Precompute per-block bssy/leave descriptors (each such block is a single instruction).
        let mut bb_bssy: Vec<Option<BssyInfo>> = vec![None; num_blocks];
        for &(i, _, _, _) in &bssy_raw {
            bb_bssy[block_of[i] as usize] = bssy_at_insn[i];
        }
        // leave_bar_slot: None => non-BREAK (all meets); Some(slot) => BREAK with bar slot (u16::MAX if bar never in bssy).
        let mut bb_leave: Vec<Option<(usize, Option<u16>)>> = vec![None; num_blocks];
        for (li, &i) in leaves.iter().enumerate() {
            let d = items[i].d.as_ref().unwrap();
            let bar_opt = if d.mnemonic == "BREAK" {
                let v = d.op_or_name("barReg", u32::MAX);
                if v != u32::MAX {
                    let slot = bar_keys
                        .iter()
                        .position(|&k| k == Some(v))
                        .map_or(u16::MAX, |p| p as u16);
                    Some(slot)
                } else {
                    None
                }
            } else {
                None
            };
            bb_leave[block_of[i] as usize] = Some((li, bar_opt));
        }

        let mut in_act: Vec<u64> = vec![0u64; num_blocks * stride];
        let mut out_buf: Vec<u64> = vec![0u64; stride];
        let mut q: VecDeque<usize> = VecDeque::with_capacity(num_blocks);
        let mut in_q: Vec<bool> = vec![false; num_blocks];
        let mut leave_reached: Vec<bool> = vec![false; leaves.len()];

        // bssy_raw is already in ascending instruction order.
        for &(r, _, _, _) in &bssy_raw {
            let bb = block_of[r] as usize;
            if !in_q[bb] {
                in_q[bb] = true;
                q.push_back(bb);
            }
        }

        let mut dyn_succ_blocks: Vec<u32> = Vec::new();
        while let Some(bb) = q.pop_front() {
            in_q[bb] = false;
            let base = bb * stride;
            out_buf.copy_from_slice(&in_act[base..base + stride]);

            if let Some(info) = bb_bssy[bb] {
                let b_off = info.bar_slot as usize * words_per_bar;
                if info.unguarded_with_bar {
                    out_buf[b_off..b_off + words_per_bar].fill(0);
                }
                let mid = info.meet_id as usize;
                out_buf[b_off + (mid >> 6)] |= 1u64 << (mid & 63);
            }

            let end_i = block_end[bb];
            let s = base_off[end_i] as usize;
            let e = base_off[end_i + 1] as usize;

            if let Some((li, bar_opt)) = bb_leave[bb] {
                leave_reached[li] = true;
                dyn_succ_blocks.clear();
                for &tgt in &base_buf[s..e] {
                    dyn_succ_blocks.push(block_of[tgt as usize]);
                }
                let use_all = match bar_opt {
                    None => true,
                    Some(u16::MAX) => has_bmov_restore,
                    Some(slot) => {
                        let b_off = base + slot as usize * words_per_bar;
                        let mut empty = true;
                        for w in 0..words_per_bar {
                            let mut bits = in_act[b_off + w];
                            if bits != 0 {
                                empty = false;
                                while bits != 0 {
                                    let tz = bits.trailing_zeros() as usize;
                                    let mid = (w << 6) + tz;
                                    dyn_succ_blocks.push(meet_blocks[mid]);
                                    bits &= bits - 1;
                                }
                            }
                        }
                        empty && has_bmov_restore
                    }
                };
                if use_all {
                    dyn_succ_blocks.extend_from_slice(&meet_blocks);
                }
                for &nxt_bb_u32 in &dyn_succ_blocks {
                    let nxt_bb = nxt_bb_u32 as usize;
                    let dst_base = nxt_bb * stride;
                    let mut changed = false;
                    for k in 0..stride {
                        let old = in_act[dst_base + k];
                        let merged = old | out_buf[k];
                        if merged != old {
                            in_act[dst_base + k] = merged;
                            changed = true;
                        }
                    }
                    if changed && !in_q[nxt_bb] {
                        in_q[nxt_bb] = true;
                        q.push_back(nxt_bb);
                    }
                }
            } else {
                for &tgt in &base_buf[s..e] {
                    let nxt_bb = block_of[tgt as usize] as usize;
                    let dst_base = nxt_bb * stride;
                    let mut changed = false;
                    for k in 0..stride {
                        let old = in_act[dst_base + k];
                        let merged = old | out_buf[k];
                        if merged != old {
                            in_act[dst_base + k] = merged;
                            changed = true;
                        }
                    }
                    if changed && !in_q[nxt_bb] {
                        in_q[nxt_bb] = true;
                        q.push_back(nxt_bb);
                    }
                }
            }
        }

        // Materialize final CSR succ with dynamic leave -> meet edges appended in sorted meet order.
        let mut succ_off: Vec<u32> = Vec::with_capacity(n + 1);
        let mut succ_buf: Vec<u32> = Vec::with_capacity(base_buf.len() + leaves.len() * 2);
        let mut leave_extra: Vec<u32> = Vec::new();
        for i in 0..n {
            succ_off.push(succ_buf.len() as u32);
            let s = base_off[i] as usize;
            let e = base_off[i + 1] as usize;
            let base_slice = &base_buf[s..e];
            succ_buf.extend_from_slice(base_slice);
            if is_leave_insn[i] {
                let bb = block_of[i] as usize;
                let (li, bar_opt) = bb_leave[bb].unwrap();
                if leave_reached[li] {
                    leave_extra.clear();
                    let base = bb * stride;
                    let use_all = match bar_opt {
                        None => true,
                        Some(u16::MAX) => has_bmov_restore,
                        Some(slot) => {
                            let b_off = base + slot as usize * words_per_bar;
                            let mut empty = true;
                            for w in 0..words_per_bar {
                                let mut bits = in_act[b_off + w];
                                if bits != 0 {
                                    empty = false;
                                    while bits != 0 {
                                        let tz = bits.trailing_zeros() as usize;
                                        let mid = (w << 6) + tz;
                                        leave_extra.push(meets[mid] as u32);
                                        bits &= bits - 1;
                                    }
                                }
                            }
                            empty && has_bmov_restore
                        }
                    };
                    if use_all {
                        for &t in &meets {
                            leave_extra.push(t as u32);
                        }
                    }
                    for &t in &leave_extra {
                        if !base_slice.contains(&t) {
                            succ_buf.push(t);
                        }
                    }
                }
            }
        }
        succ_off.push(succ_buf.len() as u32);
        (succ_off, succ_buf)
    } else {
        (base_off, base_buf)
    };

    // Build CSR preds in two linear passes (zero per-instruction Vec allocations).
    let mut pred_counts: Vec<u32> = vec![0; n];
    for &j in &succ_buf {
        pred_counts[j as usize] += 1;
    }
    let mut pred_off: Vec<u32> = Vec::with_capacity(n + 1);
    let mut total = 0u32;
    for &c in &pred_counts {
        pred_off.push(total);
        total += c;
    }
    pred_off.push(total);
    let mut pred_buf: Vec<u32> = vec![0; total as usize];
    let mut cursor = pred_off[..n].to_vec();
    for i in 0..n {
        let s = succ_off[i] as usize;
        let e = succ_off[i + 1] as usize;
        for &j in &succ_buf[s..e] {
            let pos = cursor[j as usize] as usize;
            pred_buf[pos] = i as u32;
            cursor[j as usize] = (pos + 1) as u32;
        }
    }

    Cfg {
        first_at,
        succ_off,
        succ_buf,
        pred_off,
        pred_buf,
        room_regs: compute_room_regs(items),
    }
}

#[derive(Clone, Copy)]
struct CalleeSummary {
    ret_reg: u32,
    may_write: RegSet,
    must_write: RegSet,
    data_read_inputs: RegSet,
}

const VAL_CLOB: i16 = -2;

fn initial_reg_map() -> [i16; 255] {
    let mut arr = [0i16; 255];
    for (i, slot) in arr.iter_mut().enumerate() {
        *slot = i as i16;
    }
    arr
}

fn analyze_callee(
    entry_idx: usize,
    items: &[Item],
    cfg: &Cfg,
    scratch: &mut WalkScratch,
    cache: &mut HashMap<usize, Option<CalleeSummary>>,
) -> Option<CalleeSummary> {
    if let Some(&cached) = cache.get(&entry_idx) {
        return cached;
    }
    if scratch.stack.contains(&entry_idx) {
        return None;
    }
    scratch.stack.push(entry_idx);

    // Collect callee body into a compact 0..m index space.
    let mut body: Vec<usize> = Vec::new();
    let mut nested_calls: Vec<(usize, usize)> = Vec::new();
    let mut rets: Vec<usize> = Vec::new();
    {
        let bst = scratch.next_body();
        let mut work: Vec<usize> = vec![entry_idx];
        while let Some(x) = work.pop() {
            if scratch.body_stamp[x] == bst {
                continue;
            }
            let slot = body.len();
            scratch.body_stamp[x] = bst;
            scratch.local_id[x] = slot as u32;
            body.push(x);
            let dx = items[x].d.as_ref().unwrap();
            if dx.mnemonic == "RET" {
                rets.push(slot);
            } else if dx.name.starts_with("call_rel") {
                let Some(t) = items[x].target.and_then(|tgt| cfg.first_at(tgt)) else {
                    scratch.stack.pop();
                    cache.insert(entry_idx, None);
                    return None;
                };
                nested_calls.push((slot, t));
            } else if (dx.name.starts_with("call")
                || dx.name.starts_with("jcal")
                || matches!(dx.mnemonic, "BRX" | "JMX" | "JMP"))
                && dx.name != "brx_"
            {
                scratch.stack.pop();
                cache.insert(entry_idx, None);
                return None;
            }
            for &nxt in cfg.succ(x) {
                let nxt = nxt as usize;
                if scratch.body_stamp[nxt] != bst {
                    work.push(nxt);
                }
            }
        }
    }

    if rets.is_empty()
        || rets.iter().any(|&rs| {
            let rx = body[rs];
            items[rx].d.as_ref().unwrap().name != "ret__REL" || items[rx].target != Some(0)
        })
    {
        scratch.stack.pop();
        cache.insert(entry_idx, None);
        return None;
    }

    let m = body.len();
    let mut nested_info: Vec<Option<CalleeSummary>> = if nested_calls.is_empty() {
        Vec::new()
    } else {
        vec![None; m]
    };
    for (cslot, ct) in nested_calls {
        let Some(res) = analyze_callee(ct, items, cfg, scratch, cache) else {
            let ct_in_stack = scratch.stack.contains(&ct);
            scratch.stack.pop();
            if !ct_in_stack {
                cache.insert(entry_idx, None);
            }
            return None;
        };
        nested_info[cslot] = Some(res);
    }

    // Re-stamp scratch.body_stamp/local_id because nested analyze_callee calls may have reused them.
    let bst = scratch.next_body();
    for (slot, &x) in body.iter().enumerate() {
        scratch.body_stamp[x] = bst;
        scratch.local_id[x] = slot as u32;
    }

    let mut in_st: Vec<[i16; 255]> = vec![[0i16; 255]; m];
    let mut in_must: Vec<RegSet> = vec![RegSet::default(); m];
    let mut visited: Vec<bool> = vec![false; m];
    let mut in_q: Vec<bool> = vec![false; m];

    in_st[0] = initial_reg_map();
    visited[0] = true;
    in_q[0] = true;
    let mut q: VecDeque<usize> = VecDeque::with_capacity(m);
    q.push_back(0);
    let mut may_write = RegSet::default();

    while let Some(slot) = q.pop_front() {
        in_q[slot] = false;
        let x = body[slot];
        let mut st = in_st[slot];
        let mut mw = in_must[slot];
        let dx = items[x].d.as_ref().unwrap();
        let rr = cfg.room_regs[x];
        let ug = unguarded(dx);

        if let Some((rd, rs)) = copy_src(dx) {
            if rd < 255 && !rr.contains(rd) {
                may_write.insert(rd);
                let old_v = if rs < 255 && !rr.contains(rs) {
                    st[rs as usize]
                } else {
                    VAL_CLOB
                };
                st[rd as usize] = old_v;
                if old_v != rd as i16 {
                    mw.insert(rd);
                }
            }
        } else if dx.name.starts_with("call_rel") {
            let n_info = nested_info[slot].unwrap();
            let n_may = n_info.may_write.diff(rr);
            may_write = may_write.union(n_may);
            n_may.for_each(|r| st[r as usize] = VAL_CLOB);
            if ug {
                mw = mw.union(n_info.must_write.diff(rr));
            }
        } else {
            for &(l, r) in &dx.writes {
                if l == LETTER_R && r < 255 && !rr.contains(r) {
                    may_write.insert(r);
                    st[r as usize] = VAL_CLOB;
                    if ug {
                        mw.insert(r);
                    }
                }
            }
        }

        for &nxt in cfg.succ(x) {
            let nxt = nxt as usize;
            if scratch.body_stamp[nxt] != bst {
                continue;
            }
            let nslot = scratch.local_id[nxt] as usize;
            if visited[nslot] {
                let cur = &mut in_st[nslot];
                let mut changed = false;
                for r in 0..255 {
                    if cur[r] != st[r] && cur[r] != VAL_CLOB {
                        cur[r] = VAL_CLOB;
                        changed = true;
                    }
                }
                let new_m = in_must[nslot].intersect(mw);
                if new_m != in_must[nslot] {
                    in_must[nslot] = new_m;
                    changed = true;
                }
                if changed && !in_q[nslot] {
                    in_q[nslot] = true;
                    q.push_back(nslot);
                }
            } else {
                visited[nslot] = true;
                in_st[nslot] = st;
                in_must[nslot] = mw;
                in_q[nslot] = true;
                q.push_back(nslot);
            }
        }
    }

    let mut chosen_ret: Option<u32> = None;
    let mut must_write = RegSet::ALL_255;
    for rslot in rets {
        if !visited[rslot] {
            scratch.stack.pop();
            cache.insert(entry_idx, None);
            return None;
        }
        let rx = body[rslot];
        let ra = items[rx].d.as_ref().unwrap().op_or(OP_RA, 255);
        let v = if ra < 255 { in_st[rslot][ra as usize] } else { VAL_CLOB };
        if v < 0 {
            scratch.stack.pop();
            cache.insert(entry_idx, None);
            return None;
        }
        let v = v as u32;
        if let Some(prev) = chosen_ret {
            if prev != v {
                scratch.stack.pop();
                cache.insert(entry_idx, None);
                return None;
            }
        } else {
            chosen_ret = Some(v);
        }
        must_write = must_write.intersect(in_must[rslot]);
    }

    let Some(ret_reg) = chosen_ret else {
        scratch.stack.pop();
        cache.insert(entry_idx, None);
        return None;
    };

    let mut data_read_inputs = RegSet::default();
    for (slot, &x) in body.iter().enumerate() {
        let dx = items[x].d.as_ref().unwrap();
        let rr = cfg.room_regs[x];
        if dx.mnemonic == "RET" || copy_src(dx).is_some() {
            continue;
        }
        let st_x = &in_st[slot];
        if dx.name.starts_with("call_rel") {
            let n_reads = nested_info[slot].unwrap().data_read_inputs.diff(rr);
            n_reads.for_each(|nr| {
                let v = st_x[nr as usize];
                if v >= 0 {
                    data_read_inputs.insert(v as u32);
                }
            });
            continue;
        }
        for &(l, r, _) in &dx.reads {
            if l == LETTER_R && r < 255 && !rr.contains(r) && st_x[r as usize] >= 0 {
                data_read_inputs.insert(st_x[r as usize] as u32);
            }
        }
    }

    if data_read_inputs.contains(ret_reg) {
        scratch.stack.pop();
        cache.insert(entry_idx, None);
        return None;
    }

    let summary = CalleeSummary {
        ret_reg,
        may_write,
        must_write,
        data_read_inputs,
    };
    scratch.stack.pop();
    cache.insert(entry_idx, Some(summary));
    Some(summary)
}

fn resolve_call_rel(
    k: usize,
    items: &[Item],
    cfg: &Cfg,
    scratch: &mut WalkScratch,
    cache: &mut HashMap<usize, Option<CalleeSummary>>,
) -> Result<Vec<usize>, Refused> {
    let origin = items[k].origin;
    let d_k = items[k].d.as_ref().unwrap();
    let t = items[k].target.and_then(|tgt| cfg.first_at(tgt));
    scratch.stack.clear();
    let Some(callee) = t.and_then(|ti| analyze_callee(ti, items, cfg, scratch, cache)) else {
        return Err(Refused(format!(
            "the callee of the call at 0x{:x} does not take its return address in one register",
            16 * origin
        )));
    };
    let ret_reg = callee.ret_reg;
    let back = (16 * origin + 16) as u32;

    if cfg.preds(k).is_empty() {
        return Err(Refused(format!(
            "the return address of the call at 0x{:x} does not reach it on every path",
            16 * origin
        )));
    }

    // Backward walk from predecessors of k: every incoming path must supply `MOV ret_reg, back` unguardedly.
    let vis_st = scratch.next_vis();
    let def_st = scratch.next_defs();
    scratch.work.clear();
    scratch.work.extend_from_slice(cfg.preds(k));
    let mut defs: Vec<usize> = Vec::new();

    while let Some(p_u32) = scratch.work.pop() {
        let p = p_u32 as usize;
        if scratch.vis_stamp[p] == vis_st {
            continue;
        }
        scratch.vis_stamp[p] = vis_st;
        let itp = &items[p];
        let dp = itp.d.as_ref().unwrap();
        if cfg.room_regs[p].contains(ret_reg) {
            if cfg.preds(p).is_empty() {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
            scratch.work.extend_from_slice(cfg.preds(p));
            continue;
        }
        if dp.name.starts_with("call_rel") {
            let pt = itp.target.and_then(|tgt| cfg.first_at(tgt));
            scratch.stack.clear();
            let pres = pt.and_then(|ti| analyze_callee(ti, items, cfg, scratch, cache));
            if pres.is_none_or(|c| c.may_write.contains(ret_reg)) || cfg.preds(p).is_empty() {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
            scratch.work.extend_from_slice(cfg.preds(p));
            continue;
        }
        if dp.name.starts_with("call") || dp.name.starts_with("jcal") {
            return Err(Refused(format!(
                "the return address of the call at 0x{:x} does not reach it on every path",
                16 * origin
            )));
        }
        if dp.writes.iter().any(|&(l, r)| l == LETTER_R && r == ret_reg) {
            if unguarded(dp) && immediate_of(itp) == Some(back) {
                if scratch.defs_stamp[p] != def_st {
                    scratch.defs_stamp[p] = def_st;
                    defs.push(p);
                }
            } else {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
        } else {
            if cfg.preds(p).is_empty() {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
            scratch.work.extend_from_slice(cfg.preds(p));
        }
    }

    if defs.is_empty() {
        return Err(Refused(format!(
            "the return address of the call at 0x{:x} does not reach it on every path",
            16 * origin
        )));
    }

    // Forward check from every reaching MOV before the call (no data read before kill).
    let f_st = scratch.next_vis();
    scratch.work.clear();
    for &j in &defs {
        scratch.work.extend_from_slice(cfg.succ(j));
    }
    while let Some(x_u32) = scratch.work.pop() {
        let x = x_u32 as usize;
        if x == k || scratch.vis_stamp[x] == f_st {
            continue;
        }
        scratch.vis_stamp[x] = f_st;
        let itx = &items[x];
        let dx = itx.d.as_ref().unwrap();
        if cfg.room_regs[x].contains(ret_reg) {
            scratch.work.extend_from_slice(cfg.succ(x));
            continue;
        }
        if dx.reads.iter().any(|&(l, r, _)| l == LETTER_R && r == ret_reg) {
            return Err(Refused(format!(
                "the return address of the call at 0x{:x} is read as data",
                16 * origin
            )));
        }
        if dx.name.starts_with("call_rel") {
            let pt = itx.target.and_then(|tgt| cfg.first_at(tgt));
            scratch.stack.clear();
            let pres = pt.and_then(|ti| analyze_callee(ti, items, cfg, scratch, cache));
            let Some(c) = pres else {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} is read as data",
                    16 * origin
                )));
            };
            if c.ret_reg == ret_reg || c.data_read_inputs.contains(ret_reg) {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} is read as data",
                    16 * origin
                )));
            }
            if unguarded(dx) && c.must_write.contains(ret_reg) {
                continue;
            }
        } else if dx.name.starts_with("call") || dx.name.starts_with("jcal") {
            return Err(Refused(format!(
                "the return address of the call at 0x{:x} is read as data",
                16 * origin
            )));
        }
        if unguarded(dx) && dx.writes.iter().any(|&(l, r)| l == LETTER_R && r == ret_reg) {
            continue;
        }
        scratch.work.extend_from_slice(cfg.succ(x));
    }

    // Post-call check if the call is guarded or the callee does not overwrite ret_reg on all paths.
    if !unguarded(d_k) || !callee.must_write.contains(ret_reg) {
        let p_st = scratch.next_vis();
        scratch.work.clear();
        scratch.work.extend_from_slice(cfg.succ(k));
        while let Some(x_u32) = scratch.work.pop() {
            let x = x_u32 as usize;
            if scratch.vis_stamp[x] == p_st {
                continue;
            }
            scratch.vis_stamp[x] = p_st;
            let itx = &items[x];
            let dx = itx.d.as_ref().unwrap();
            if cfg.room_regs[x].contains(ret_reg) {
                scratch.work.extend_from_slice(cfg.succ(x));
                continue;
            }
            if dx.reads.iter().any(|&(l, r, _)| l == LETTER_R && r == ret_reg) {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} is read as data",
                    16 * origin
                )));
            }
            if dx.name.starts_with("call_rel") {
                let pt = itx.target.and_then(|tgt| cfg.first_at(tgt));
                scratch.stack.clear();
                let pres = pt.and_then(|ti| analyze_callee(ti, items, cfg, scratch, cache));
                let Some(c) = pres else {
                    return Err(Refused(format!(
                        "the return address of the call at 0x{:x} is read as data",
                        16 * origin
                    )));
                };
                if c.ret_reg == ret_reg || c.data_read_inputs.contains(ret_reg) {
                    return Err(Refused(format!(
                        "the return address of the call at 0x{:x} is read as data",
                        16 * origin
                    )));
                }
                if unguarded(dx) && c.must_write.contains(ret_reg) {
                    continue;
                }
            } else if dx.name.starts_with("call") || dx.name.starts_with("jcal") {
                continue;
            }
            if dx.writes.iter().any(|&(l, r)| l == LETTER_R && r == ret_reg) {
                continue;
            }
            scratch.work.extend_from_slice(cfg.succ(x));
        }
    }

    defs.sort_unstable();
    Ok(defs)
}

/// Also finds the jump tables to move with the code: (where in the constants of the function, the
/// places as they are).
pub(super) fn return_addresses(
    items: &mut [Item],
    targets: &[u64],
    fd: &FuncData,
) -> Result<Vec<(u32, Vec<u64>)>, Refused> {
    let src = &fd.src;
    let mut relocate: Vec<(u32, Vec<u64>)> = Vec::new();
    let mut fixups: Vec<(usize, u64)> = Vec::new(); // (item_index, old_offset_it_stands_for)
    let mut cfg: Option<Cfg> = None;
    let mut scratch: Option<WalkScratch> = None;
    let mut callee_cache: HashMap<usize, Option<CalleeSummary>> = HashMap::new();

    for k in 0..items.len() {
        let d_name = items[k].d.as_ref().unwrap().name;
        let d_mnemonic = items[k].d.as_ref().unwrap().mnemonic;
        let origin = items[k].origin;

        if d_name == "brx_"
            && let Some(places) = fd.table_at(16 * origin as u64)
        {
            // Its table is constants of the function (bank 2), read into the register by the LDC
            // before it: where that reads, the places are to be moved with the code.
            let register = items[k].d.as_ref().unwrap().op_or(OP_RA, 255);
            let define = (0..k).rev().find(|&j| {
                let writes = &items[j].d.as_ref().unwrap().writes;
                writes.iter().any(|&(l, r)| l == LETTER_R && r == register)
            });
            let load = define.map(|j| items[j].d.as_ref().unwrap()).filter(|load| {
                load.name.starts_with("ldc__") && load.op(OP_SA_BANK) == Some(2) && load.op(OP_SZ) == Some(4)
            });
            let straight = define.is_some_and(|define| {
                !(define + 1..k).any(|j| {
                    let between = items[j].d.as_ref().unwrap();
                    (items[j].first && targets.contains(&(16 * items[j].origin as u64)))
                        || (between.kind == Kind::BRU && !matches!(between.mnemonic, "BSSY" | "BMOV"))
                })
            });
            let Some(load) = load.filter(|_| straight) else {
                return Err(Refused(format!(
                    "BRX at 0x{:x}: its table is not read by an LDC before it",
                    16 * origin
                )));
            };
            relocate.push((load.op_or(OP_RA_OFFSET, 0), places.clone()));
        } else if d_name.starts_with("call_rel")
            || d_name.starts_with("jmp")
            || matches!(d_mnemonic, "BRX" | "JMX" | "JMP")
        {
            if !d_name.starts_with("call_rel") {
                return Err(Refused(format!(
                    "{} at 0x{:x}: a jump that is not understood",
                    d_mnemonic,
                    16 * origin
                )));
            }
            let c = cfg.get_or_insert_with(|| build_cfg(items, fd));
            let sc = scratch.get_or_insert_with(|| WalkScratch::new(items.len()));
            let defs = resolve_call_rel(k, items, c, sc, &mut callee_cache)?;
            let back = (16 * origin + 16) as u64;
            for j in defs {
                fixups.push((j, back));
            }
        } else if d_name.starts_with("call_abs") {
            // Through a register: LEPC + IADD3 + two MOVs
            let lepc = (k.saturating_sub(15)..k)
                .rev()
                .find(|&j| items[j].d.as_ref().unwrap().mnemonic == "LEPC");
            let lepc = match lepc {
                Some(j) => j,
                None => {
                    return Err(Refused(format!(
                        "a call through a register at 0x{:x} without LEPC before it",
                        16 * origin
                    )));
                }
            };
            let pc_reg = items[lepc].d.as_ref().unwrap().op_or(OP_RD, 255);
            let add = (lepc + 1..k).find(|&j| {
                let d = items[j].d.as_ref().unwrap();
                d.name == "iadd3_noimm__RRR_RRR" && d.reads.iter().any(|&(l, r, _)| l == LETTER_R && r == pc_reg)
            });
            let add = match add {
                Some(j) => j,
                None => {
                    return Err(Refused(format!(
                        "the call at 0x{:x}: no addition on the program counter",
                        16 * origin
                    )));
                }
            };
            let add_d = items[add].d.as_ref().unwrap();
            let ra = add_d.op_or(OP_RA, 255);
            let rb = add_d.op_or(OP_RB, 255);
            let rc = add_d.op_or(OP_RC, 255);
            let ra_neg = add_d.op_or(OP_RA_AT_NEGATE, 0);
            let rb_neg = add_d.op_or(OP_RB_AT_NEGATE, 0);
            let rc_neg = add_d.op_or(OP_RC_AT_NEGATE, 0);
            let terms: Vec<(u32, u32)> = [(ra, ra_neg), (rb, rb_neg), (rc, rc_neg)]
                .iter()
                .filter(|&&(r, _)| r != pc_reg)
                .cloned()
                .collect();
            if terms.len() != 2 || {
                let mut negs: Vec<u32> = terms.iter().map(|&(_, n)| n).collect();
                negs.sort();
                negs != vec![0, 1]
            } {
                return Err(Refused(format!(
                    "the call at 0x{:x}: the addition on the program counter has another shape",
                    16 * origin
                )));
            }
            let mut values: HashMap<u32, usize> = HashMap::new(); // negated -> define index
            for &(register, negated) in &terms {
                let define = (0..add).rev().find(|&j| {
                    items[j]
                        .d
                        .as_ref()
                        .unwrap()
                        .writes
                        .iter()
                        .any(|&(l, r)| l == LETTER_R && r == register)
                });
                let define = match define {
                    Some(j) if immediate_of(&items[j]).is_some() => j,
                    _ => {
                        return Err(Refused(format!(
                            "the call at 0x{:x}: a term of its return address is not a MOV before it",
                            16 * origin
                        )));
                    }
                };
                // Check no branch target between define and call
                if (define + 1..=k).any(|j| items[j].first && targets.contains(&(16 * items[j].origin as u64))) {
                    return Err(Refused(format!(
                        "the call at 0x{:x}: a term of its return address is not a MOV before it",
                        16 * origin
                    )));
                }
                values.insert(negated, define);
            }
            if (lepc + 1..=k).any(|j| items[j].first && targets.contains(&(16 * items[j].origin as u64))) {
                return Err(Refused(format!(
                    "the call at 0x{:x}: a branch arrives between LEPC and the call",
                    16 * origin
                )));
            }
            let pos_val = immediate_of(&items[values[&0]]).unwrap_or(0) as i64;
            let neg_val = immediate_of(&items[values[&1]]).unwrap_or(0) as i64;
            let lepc_origin = items[lepc].origin;
            let back = 16 * lepc_origin as i64 + pos_val - neg_val;
            if back % 16 != 0 || back < 0 || back >= 16 * src.len() as i64 {
                return Err(Refused(format!(
                    "the call at 0x{:x}: its return address {} is not an instruction",
                    16 * origin,
                    hex(back)
                )));
            }
            fixups.push((values[&0], back as u64));
            fixups.push((values[&1], 16 * lepc_origin as u64));
        } else if d_mnemonic == "LEPC" {
            let limit = (k + 16).min(items.len());
            if !(k + 1..limit).any(|j| items[j].d.as_ref().unwrap().name.starts_with("call_abs")) {
                return Err(Refused(format!("LEPC at 0x{:x} without a call after it", 16 * origin)));
            }
        }
    }

    for (idx, old_offset) in fixups {
        items[idx].fix = Some(old_offset);
    }

    Ok(relocate)
}

fn immediate_of(item: &Item) -> Option<u32> {
    let d = item.d.as_ref()?;
    if d.name == "mov__RuIR_RI"
        || (d.mnemonic == "MOV"
            && d.has_op(OP_SB)
            && !d.has_op(OP_RB)
            && !d.has_op(OP_SB_BANK)
            && d.op_or_name("PixMaskU04", 15) == 15)
    {
        return d.op(OP_SB);
    }
    None
}
