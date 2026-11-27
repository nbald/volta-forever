//! Which products of a function the translator does as pairs on whole rows, which matrix loads
//! it does as whole rows, and where it converts the accumulator registers: its plan, by the
//! design of `docs/pairs.md`.
//!
//! A faithful port of `reference/pairing.py`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::decode::{Decoded, LETTER_P, LETTER_R};
use crate::fast::{
    FastSet, RegSet, regset_intersects, regset_is_empty, regset_set, regset_sub, regset_test, regset_union,
};
use crate::field;
use crate::generated::fields::*;

const LOADS: &[&str] = &["ldsm__sImmOffset", "ldsm__URsIR"];
const PRODUCT: &str = "hmma_x8_";

/// Mnemonics that park lanes, take them out, or bring them back.
const LANES: &[&str] = &[
    "BSYNC", "BREAK", "KILL", "PBK", "PCNT", "PEXIT", "PLONGJMP", "PRET", "SSY", "SYNC", "BRK", "CONT", "LONGJMP",
    "RTT",
];
/// Of those, the ones after which lanes go on elsewhere.
const LEAVES: &[&str] = &["BREAK", "BRK", "CONT", "PEXIT", "PLONGJMP", "PRET", "LONGJMP"];

/// What a register is next used as (ordered: the join is the larger).
const DEAD: u8 = 0;
const WANTS_Q: u8 = 1;
const WANTS_T: u8 = 2;

/// The state of a register at a place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    Z,
    T,
    Q,
}

/// The names of the values kept from the entry of the function for pairs.
pub const ROW_KEPT: &[&str] = &["kA", "kA2", "kB", "kB2", "kBT", "cB", "pi", "inv"];

// ---- Plan ----

/// A whole-row load entry in the plan.
#[derive(Debug, Clone)]
pub struct LoadEntry {
    pub role: LoadRole,
    pub rd: u32,
    pub also: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadRole {
    A,
    B,
    BT,
}

/// A product-of-a-pair entry in the plan.
#[derive(Debug, Clone)]
pub struct SiteEntry {
    pub half: u32,
    pub partner: usize,
    pub x: u32,
    pub y: u32,
    /// The accumulator of the product whose B register is the even one (None for zero).
    pub cx: Option<u32>,
    /// The accumulator of the other product (None for zero).
    pub cy: Option<u32>,
    pub ra: u32,
    pub a: u32,
    pub rb: u32,
    pub p: u32,
    pub transposed: bool,
}

/// A conversion: (register, target state).
pub type Conversion = (u32, State);

/// The plan for a function.
#[derive(Debug)]
pub struct Plan {
    pub loads: BTreeMap<usize, LoadEntry>,
    pub sites: BTreeMap<usize, SiteEntry>,
    pub share: BTreeMap<u32, u32>,
    pub before: BTreeMap<usize, Vec<Conversion>>,
    pub after: BTreeMap<usize, Vec<Conversion>>,
}

impl Plan {
    fn new() -> Self {
        Self {
            loads: BTreeMap::new(),
            sites: BTreeMap::new(),
            share: BTreeMap::new(),
            before: BTreeMap::new(),
            after: BTreeMap::new(),
        }
    }

    /// {destination of whole-row loads: how many registers of the translation its rows take}.
    pub fn rows(&self) -> BTreeMap<u32, u32> {
        let mut out = BTreeMap::new();
        for v in self.loads.values() {
            let count = if v.also { 8 } else { 4 };
            let entry = out.entry(v.rd).or_insert(4u32);
            *entry = (*entry).max(count);
        }
        out
    }

    /// Which values are kept from the entry of the function.
    pub fn kept(&self) -> BTreeSet<&'static str> {
        let mut out = BTreeSet::new();
        for v in self.loads.values() {
            match v.role {
                LoadRole::A => {
                    out.insert("kA");
                    out.insert("kA2");
                }
                LoadRole::B => {
                    out.insert("kB");
                    out.insert("kB2");
                }
                LoadRole::BT => {
                    out.insert("kBT");
                    out.insert("cB");
                }
            }
        }
        for cs in self.before.values().chain(self.after.values()) {
            for &(_, to) in cs {
                out.insert(if to == State::Q { "pi" } else { "inv" });
            }
        }
        out
    }
}

// ---- Helpers ----

#[inline]
fn true_guard(d: &Decoded, fid: u16, not_fid: u16) -> bool {
    d.op_or(fid, 7) == 7 && d.op_or(not_fid, 0) == 0
}

/// Bitmask (`0..7`) of the predicates an instruction may write: those its operands say, and for
/// R2P those of its mask (every one, when the mask is not a constant): they are not among its
/// operands. (Left out until 2026-11-18: a guard that R2P changed still hid a load from the
/// readers under it.)
#[inline]
fn written_predicates_mask(d: &Decoded) -> u8 {
    let mut mask = 0u8;
    for &(l, x) in &d.writes {
        if l == LETTER_P && x < 7 {
            mask |= 1u8 << x;
        }
    }
    if d.mnemonic == "R2P" {
        let m = if d.name == "r2p__RIR" {
            d.op_or(OP_SB, 0x7f) & 0x7f
        } else {
            0x7f
        };
        mask |= (m as u8) & 0x7f;
    }
    mask
}

#[inline]
pub(crate) fn unguarded(d: &Decoded) -> bool {
    true_guard(d, OP_PG, OP_PG_AT_NOT) && true_guard(d, OP_UPG, OP_UPG_AT_NOT)
}

/// Whether instruction `d` always goes on to the next instruction and does not leave the
/// section: it is not a branch (except BSSY and BMOV), not an exit, not a return, not an
/// external call.
pub(crate) fn is_straight(d: &Decoded) -> bool {
    if d.name.starts_with("call_abs") {
        return false;
    }
    if d.kind == crate::decode::Kind::BRU && d.mnemonic != "BSSY" && d.mnemonic != "BMOV" {
        return false;
    }
    true
}

/// (Rd, Ra, Rb, Rc) where Rc is the accumulator: the destination (in place), four other registers, or None for zero.
fn product_form(d: &Decoded) -> Option<(u32, u32, u32, Option<u32>)> {
    if d.name != PRODUCT || !unguarded(d) {
        return None;
    }
    if d.op_or(OP_RA_AT_NEGATE, 0) != 0
        || d.op_or(OP_RB_AT_NEGATE, 0) != 0
        || d.op_or(OP_SATFINITE, 0) != 0
        || d.op_or(OP_SIZE, 0) != 0
        || d.op_or(OP_ALTFMT, 0) != 0
        || d.op_or(OP_DSTFMT, 0) != 1
    {
        return None;
    }
    let rd = d.op_or(OP_RD, 255);
    let ra = d.op_or(OP_RA, 255);
    let rb = d.op_or(OP_RB, 255);
    let rc = d.op_or(OP_RC, 255);
    if [rd, ra, rb].contains(&255) || !rd.is_multiple_of(2) || !ra.is_multiple_of(2) || rd + 3 >= 255 {
        return None;
    }
    if (rd..rd + 4).any(|r| r == ra || r == ra + 1 || r == rb) {
        return None;
    }
    if rc == 255 {
        return Some((rd, ra, rb, None));
    }
    if !rc.is_multiple_of(2) || rc + 3 >= 255 || (rc != rd && (rc..rc + 4).any(|r| (rd..rd + 4).contains(&r))) {
        return None;
    }
    Some((rd, ra, rb, Some(rc)))
}

fn load_form(d: &Decoded) -> Option<(u32, bool)> {
    if !LOADS.contains(&d.name) || !unguarded(d) {
        return None;
    }
    if d.op_or(OP_SZ, 0) != 0 || d.op_or(OP_NUM, 0) != 2 || !matches!(d.op_or(OP_MODE, 0), 0 | 1) {
        return None;
    }
    let rd = d.op_or(OP_RD, 255);
    if rd == 255 || !rd.is_multiple_of(4) || rd + 3 >= 255 {
        return None;
    }
    Some((rd, d.op_or(OP_MODE, 0) == 1))
}

#[inline]
fn zero_writer(d: &Decoded) -> bool {
    if !unguarded(d) {
        return false;
    }
    if d.name == "cs2r_" {
        return d.op_or(OP_SRA, 0) == 255;
    }
    if d.mnemonic == "IMAD" && d.name.ends_with("__RRR_RRR") {
        return d.op_or(OP_RA, 0) == 255 && d.op_or(OP_RB, 0) == 255 && d.op_or(OP_RC, 0) == 255;
    }
    false
}

/// What a call reads and writes: (registers alive at entry, registers written).
pub(crate) type CallEffect = Option<(RegSet, RegSet)>;

/// The form of a product: (Rd, Ra, Rb, Rc).
type ProductForm = (u32, u32, u32, Option<u32>);

#[derive(Clone, Default)]
struct RegDefs {
    has_none: bool,
    locs: Vec<usize>,
}

type FormReach = HashMap<usize, [Option<RegDefs>; 4]>;

// ---- Flow ----

pub(crate) struct Flow {
    pub(crate) n: usize,
    pub(crate) kind: Vec<&'static str>,
    pub(crate) known: bool,
    pub(crate) named: Vec<Option<usize>>,
    pub(crate) succ: Vec<Vec<usize>>,
    pub(crate) pred: Vec<Vec<usize>>,
    pub(crate) called: HashMap<usize, CallEffect>,
    pub(crate) instr_reads_r: Vec<RegSet>,
    pub(crate) instr_writes_r: Vec<RegSet>,
    reads_r: Vec<RegSet>,
    writes_r: Vec<RegSet>,
    pub(crate) unguarded: Vec<bool>,
    pub(crate) blocks: Vec<(usize, usize)>,
    pub(crate) block_of: Vec<usize>,
    pub(crate) bb_succ: Vec<Vec<usize>>,
    pub(crate) bb_pred: Vec<Vec<usize>>,
    pub(crate) bb_rpo: Vec<usize>,
    pub(crate) bb_reachable: Vec<bool>,
    aligned: Vec<bool>,
    lead: Vec<bool>,
    uniform: Vec<bool>,
    whole: Vec<bool>,
    /// The entries of the function (`entries`): where it may be entered other than from the
    /// instruction before.
    entered: FastSet<usize>,
}

impl Flow {
    pub(crate) fn new(src: &[Decoded], tables: &[(u64, Vec<u64>)], entries: &Option<HashSet<usize>>) -> Self {
        Self::new_with_facts(src, tables, entries, true)
    }

    /// Analyze `src`, skipping `reads_r`/`writes_r` and `compute_whole_places` when `with_pairing` is false.
    pub(crate) fn new_with_facts(
        src: &[Decoded],
        tables: &[(u64, Vec<u64>)],
        entries: &Option<HashSet<usize>>,
        with_pairing: bool,
    ) -> Self {
        let n = src.len();
        let entered: FastSet<usize> = entries.iter().flatten().copied().collect();
        let kind: Vec<&'static str> = (0..n).map(|i| Self::kind_of(src, tables, i)).collect();
        let known = entries.is_some() && !kind.contains(&"unknown");
        if !known {
            return Self::empty(n, kind, entered);
        }

        let mut instr_reads_r = vec![[0u64; 4]; n];
        let mut instr_writes_r = vec![[0u64; 4]; n];
        let mut unguarded_vec = vec![false; n];
        let mut named = vec![None; n];
        let mut meets_set = FastSet::default();
        let mut meets = Vec::new();

        for (i, d) in src.iter().enumerate() {
            unguarded_vec[i] = unguarded(d);
            for &(l, r, _) in &d.reads {
                if l == LETTER_R && r < 255 {
                    regset_set(&mut instr_reads_r[i], r);
                }
            }
            for &(l, r) in &d.writes {
                if l == LETTER_R && r < 255 {
                    regset_set(&mut instr_writes_r[i], r);
                }
            }
            if let Some((rel_fid, scale)) = crate::translate::find_relative_decoded(d, 75) {
                let v = field::signed(rel_fid, d.word);
                let t = 16 * i as i64 + 16 + v * scale as i64;
                if t % 16 == 0 && t >= 0 && t < 16 * n as i64 {
                    let target = (t / 16) as usize;
                    named[i] = Some(target);
                    if matches!(d.mnemonic, "BSSY" | "PBK" | "SSY" | "PCNT") && meets_set.insert(target) {
                        meets.push(target);
                    }
                }
            }
        }
        meets.sort_unstable();

        let succ: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let has_after = i + 1 < n;
                let k = kind[i];
                let mut out = Vec::with_capacity(2);
                if k == "jump" {
                    if let Some(t) = named[i] {
                        out.push(t);
                    }
                } else if k == "cond" {
                    if has_after {
                        out.push(i + 1);
                    }
                    if let Some(t) = named[i] {
                        out.push(t);
                    }
                } else if k == "exit" || src[i].mnemonic == "RET" {
                    if !unguarded_vec[i] && has_after {
                        out.push(i + 1);
                    }
                } else if src[i].mnemonic == "BRX" {
                    let table = tables.iter().find(|(at, _)| *at == 16 * i as u64);
                    if let Some((_, places)) = table {
                        for &t in places {
                            if t % 16 == 0 && (t as usize) < 16 * n {
                                out.push((t / 16) as usize);
                            }
                        }
                    }
                    if !unguarded_vec[i] && has_after {
                        out.push(i + 1);
                    }
                } else if LEAVES.contains(&src[i].mnemonic) {
                    if has_after {
                        out.push(i + 1);
                    }
                    out.extend_from_slice(&meets);
                } else if has_after {
                    out.push(i + 1);
                }
                if out.len() > 1 {
                    out.sort_unstable();
                    out.dedup();
                }
                out
            })
            .collect();

        // Every branch goes to an instruction of the function (`reference/pairing.py`): one to
        // its end or outside leaves the ways unknown.
        let all_known = (0..n).all(|i| !(kind[i] == "jump" || kind[i] == "cond") || named[i].is_some());
        if !all_known {
            return Self::empty(n, kind, entered);
        }

        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, targets) in succ.iter().enumerate() {
            for &s in targets {
                pred[s].push(i);
            }
        }

        // Split into basic blocks once (a block starts iff `!simple(i)`).
        let mut blocks: Vec<(usize, usize)> = Vec::new();
        let mut block_of: Vec<usize> = vec![0; n];
        if n > 0 {
            let mut bstart = 0usize;
            for i in 1..n {
                if pred[i].len() == 1 && pred[i][0] == i - 1 && kind[i - 1] == "plain" && !entered.contains(&i) {
                    block_of[i] = blocks.len();
                } else {
                    blocks.push((bstart, i));
                    bstart = i;
                    block_of[i] = blocks.len();
                }
            }
            blocks.push((bstart, n));
        }
        let nb = blocks.len();
        let mut bb_succ: Vec<Vec<usize>> = Vec::with_capacity(nb);
        let mut bb_pred: Vec<Vec<usize>> = Vec::with_capacity(nb);
        for &(start_b, end_b) in &blocks {
            bb_succ.push(succ[end_b - 1].iter().map(|&s| block_of[s]).collect());
            bb_pred.push(pred[start_b].iter().map(|&p| block_of[p]).collect());
        }

        let mut bb_reachable = vec![false; nb];
        let mut bb_rpo = Vec::with_capacity(nb);
        if nb > 0 {
            let mut post = Vec::with_capacity(nb);
            let mut visited = vec![false; nb];
            let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
            visited[0] = true;
            while let Some((b, idx)) = stack.pop() {
                if idx < bb_succ[b].len() {
                    let next = bb_succ[b][idx];
                    stack.push((b, idx + 1));
                    if !visited[next] {
                        visited[next] = true;
                        stack.push((next, 0));
                    }
                } else {
                    post.push(b);
                }
            }
            bb_reachable = visited;
            post.reverse();
            bb_rpo = post;
        }

        // Effect of calls within the section: only when the callee is known whole and does not call anything itself.
        let mut called: HashMap<usize, CallEffect> = HashMap::new();
        let mut in_body = vec![usize::MAX; n];
        let mut body_nodes: Vec<usize> = Vec::new();
        let mut work: Vec<usize> = Vec::new();
        let mut alive_scratch: Vec<RegSet> = Vec::new();
        for i in 0..n {
            if kind[i] != "call" {
                continue;
            }
            // A function that is not here: anything.
            let entry = match named[i] {
                Some(entry) if src[i].name.starts_with("call_rel") => entry,
                _ => {
                    called.insert(i, None);
                    continue;
                }
            };
            body_nodes.clear();
            work.clear();
            work.push(entry);
            let mut simple = true;
            while let Some(k) = work.pop() {
                if in_body[k] != usize::MAX {
                    continue;
                }
                in_body[k] = 0;
                body_nodes.push(k);
                if kind[k] == "call" {
                    simple = false;
                    break;
                }
                work.extend_from_slice(&succ[k]);
            }
            if !simple {
                for &k in &body_nodes {
                    in_body[k] = usize::MAX;
                }
                called.insert(i, None);
                continue;
            }
            body_nodes.sort_unstable();
            for (pos, &k) in body_nodes.iter().enumerate() {
                in_body[k] = pos;
            }
            let bn = body_nodes.len();
            alive_scratch.clear();
            alive_scratch.resize(bn, [0u64; 4]);
            let mut changed = true;
            while changed {
                changed = false;
                for pos in (0..bn).rev() {
                    let k = body_nodes[pos];
                    let mut after_set = [0u64; 4];
                    for &t in &succ[k] {
                        let tpos = in_body[t];
                        if tpos != usize::MAX {
                            regset_union(&mut after_set, &alive_scratch[tpos]);
                        }
                    }
                    if unguarded_vec[k] {
                        regset_sub(&mut after_set, &instr_writes_r[k]);
                    }
                    regset_union(&mut after_set, &instr_reads_r[k]);
                    if after_set != alive_scratch[pos] {
                        alive_scratch[pos] = after_set;
                        changed = true;
                    }
                }
            }
            let entry_pos = in_body[entry];
            let entry_alive = alive_scratch[entry_pos];
            let mut writes = [0u64; 4];
            for &k in &body_nodes {
                regset_union(&mut writes, &instr_writes_r[k]);
                in_body[k] = usize::MAX;
            }
            called.insert(i, Some((entry_alive, writes)));
        }

        let (reads_r, writes_r, aligned) = if with_pairing {
            let mut reads_r = instr_reads_r.clone();
            let mut writes_r = instr_writes_r.clone();
            for i in 0..n {
                if kind[i] == "call" {
                    match called.get(&i) {
                        Some(Some((alive, ws))) => {
                            reads_r[i] = *alive;
                            writes_r[i] = *ws;
                        }
                        _ => {
                            reads_r[i] = [u64::MAX; 4];
                            writes_r[i] = [u64::MAX; 4];
                        }
                    }
                }
            }
            let aligned: Vec<bool> = (0..n)
                .map(|i| (LOADS.contains(&src[i].name) || src[i].name == PRODUCT) && unguarded_vec[i])
                .collect();
            (reads_r, writes_r, aligned)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        let mut flow = Self {
            n,
            kind,
            known: true,
            named,
            succ,
            pred,
            called,
            instr_reads_r,
            instr_writes_r,
            reads_r,
            writes_r,
            unguarded: unguarded_vec,
            blocks,
            block_of,
            bb_succ,
            bb_pred,
            bb_rpo,
            bb_reachable,
            aligned,
            lead: if with_pairing { vec![false; n] } else { Vec::new() },
            uniform: if with_pairing { vec![false; n] } else { Vec::new() },
            whole: if with_pairing { vec![false; n] } else { Vec::new() },
            entered,
        };
        if with_pairing {
            flow.compute_whole_places(src);
        }
        flow
    }

    fn empty(n: usize, kind: Vec<&'static str>, entered: FastSet<usize>) -> Self {
        Self {
            n,
            kind,
            known: false,
            named: Vec::new(),
            succ: Vec::new(),
            pred: Vec::new(),
            called: HashMap::new(),
            instr_reads_r: Vec::new(),
            instr_writes_r: Vec::new(),
            reads_r: Vec::new(),
            writes_r: Vec::new(),
            unguarded: Vec::new(),
            blocks: Vec::new(),
            block_of: Vec::new(),
            bb_succ: Vec::new(),
            bb_pred: Vec::new(),
            bb_rpo: Vec::new(),
            bb_reachable: Vec::new(),
            aligned: Vec::new(),
            lead: Vec::new(),
            uniform: Vec::new(),
            whole: Vec::new(),
            entered,
        }
    }

    fn kind_of(src: &[Decoded], tables: &[(u64, Vec<u64>)], i: usize) -> &'static str {
        let d = &src[i];
        let m = d.mnemonic;
        if m == "BRA" {
            // BRA.U (cond=1) is taken always when it is under no predicate. BRA.DIV and BRA.CONV stay conditional.
            return if unguarded(d) && true_guard(d, OP_PP, OP_PP_AT_NOT) && matches!(d.op_or(OP_COND, 0), 0 | 1) {
                "jump"
            } else {
                "cond"
            };
        }
        if m == "BRX" {
            let has_table = tables.iter().any(|(at, _)| *at == 16 * i as u64);
            return if d.name == "brx_" && has_table {
                "other"
            } else {
                "unknown"
            };
        }
        if m == "JMX" || m == "JMP" {
            return "unknown";
        }
        if m == "CALL" {
            return "call";
        }
        if m == "EXIT" {
            return if unguarded(d) { "exit" } else { "other" };
        }
        if m == "RET" || LANES.contains(&m) || matches!(m, "BSSY" | "BPT") {
            return "other";
        }
        "plain"
    }

    #[inline]
    fn target(&self, i: usize) -> Option<usize> {
        self.named[i]
    }

    #[inline]
    fn removes(&self, src: &[Decoded], i: usize) -> bool {
        let m = src[i].mnemonic;
        LANES.contains(&m) || (matches!(m, "EXIT" | "RET") && !self.unguarded[i])
    }

    #[inline]
    fn reads_reg(&self, i: usize, r: u32) -> bool {
        regset_test(&self.reads_r[i], r)
    }

    #[inline]
    fn writes_reg(&self, i: usize, r: u32) -> bool {
        regset_test(&self.writes_r[i], r)
    }

    #[inline]
    fn all_of(&self, i: usize, r: u32) -> bool {
        self.kind[i] != "call" && self.unguarded[i] && regset_test(&self.writes_r[i], r)
    }

    #[inline]
    fn goes_whole(&self, src: &[Decoded], p: usize) -> bool {
        if !self.whole[p] || self.removes(src, p) {
            return false;
        }
        self.kind[p] == "plain" || self.kind[p] == "jump" || (self.kind[p] == "cond" && self.uniform[p])
    }

    fn compute_whole_places(&mut self, src: &[Decoded]) {
        let n = self.n;
        let mut lead: Vec<Option<bool>> = vec![None; n];
        let mut in_path = vec![0u32; n];
        let mut epoch = 0u32;
        let mut path = Vec::new();

        for i in 0..n {
            if lead[i].is_some() {
                continue;
            }
            epoch += 1;
            path.clear();
            let mut cur = i;
            let result = loop {
                if cur >= n || lead[cur] == Some(false) || in_path[cur] == epoch {
                    break false;
                }
                if lead[cur] == Some(true) || self.aligned[cur] {
                    break true;
                }
                in_path[cur] = epoch;
                path.push(cur);
                if self.kind[cur] == "plain" && !self.removes(src, cur) {
                    cur += 1;
                } else if self.kind[cur] == "jump" {
                    match self.named[cur] {
                        Some(t) => cur = t,
                        None => break false,
                    }
                } else {
                    break false;
                }
            };
            for &k in &path {
                lead[k] = Some(result);
            }
            // Always set lead[i] to the result, even when i itself is aligned and the path is empty.
            lead[i] = Some(result);
        }
        let lead_bool: Vec<bool> = lead.into_iter().map(|x| x.unwrap_or(false)).collect();
        self.lead = lead_bool;

        self.uniform = (0..n)
            .map(|i| {
                self.kind[i] == "cond"
                    && (self.target(i).is_some_and(|t| self.lead[t]) || (i + 1 < n && self.lead[i + 1]))
            })
            .collect();

        let mut whole = self.lead.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for b in 0..n {
                if whole[b] || self.pred[b].is_empty() {
                    continue;
                }
                if self.pred[b].iter().all(|&p| {
                    if !whole[p] || self.removes(src, p) {
                        return false;
                    }
                    self.kind[p] == "plain" || self.kind[p] == "jump" || (self.kind[p] == "cond" && self.uniform[p])
                }) {
                    whole[b] = true;
                    changed = true;
                }
            }
        }
        self.whole = whole;
    }

    fn escapes(&self, src: &[Decoded], load: usize, rd: u32, sites: &FastSet<usize>) -> bool {
        let mut seen: FastSet<(usize, u16)> = FastSet::default();
        let mut work: Vec<(usize, u16)> = Vec::new();
        for r in rd..rd + 4 {
            seen.clear();
            work.clear();
            for &s in &self.succ[load] {
                work.push((s, 0));
            }
            while let Some((i, hidden)) = work.pop() {
                if !seen.insert((i, hidden)) {
                    continue;
                }
                let d = &src[i];
                // Guard analysis encoded into 16 bits: bit `(pg_not * 8 + pg)` for `0..8 x 0..2`.
                let guard_bit: Option<Option<u16>> = if self.unguarded[i] {
                    None
                } else if !d.has_op(OP_UPG) || true_guard(d, OP_UPG, OP_UPG_AT_NOT) {
                    let pg = d.op_or(OP_PG, 7).min(7);
                    let pg_not = d.op_or(OP_PG_AT_NOT, 0).min(1);
                    Some(Some(1u16 << (pg_not * 8 + pg)))
                } else {
                    Some(None)
                };
                if self.reads_reg(i, r) && !sites.contains(&i) {
                    match guard_bit {
                        None | Some(None) => return true,
                        Some(Some(bit)) => {
                            if (hidden & bit) == 0 {
                                return true;
                            }
                        }
                    }
                }
                let wp_mask = written_predicates_mask(d) as u16;
                let clear_mask = !((wp_mask & 0x7f) * 0x0101u16);
                let is_call = self.kind[i] == "call";
                if self.writes_reg(i, r) {
                    if self.all_of(i, r) {
                        continue;
                    }
                    let mut next_hidden = hidden;
                    if let Some(Some(bit)) = guard_bit {
                        next_hidden |= bit;
                    }
                    next_hidden &= clear_mask;
                    if is_call {
                        next_hidden = 0;
                    }
                    for &s in &self.succ[i] {
                        work.push((s, next_hidden));
                    }
                    continue;
                }
                let mut next_hidden = hidden & clear_mask;
                if is_call {
                    next_hidden = 0;
                }
                for &s in &self.succ[i] {
                    work.push((s, next_hidden));
                }
            }
        }
        false
    }

    fn reaching_for_forms(&self, forms: &BTreeMap<usize, Option<ProductForm>>, watch: &[u32]) -> FormReach {
        let mut out: FormReach = HashMap::new();
        if watch.is_empty() || self.n == 0 {
            return out;
        }
        let mut watch_idx = [u16::MAX; 256];
        let mut watch_mask = [0u64; 4];
        for (wi, &r) in watch.iter().enumerate() {
            if r < 256 {
                watch_idx[r as usize] = wi as u16;
                regset_set(&mut watch_mask, r);
            }
        }
        let nw = watch.len();
        let nb = self.blocks.len();

        // Map each form instruction (with valid product_form) to the 4 registers it reads from loads.
        // Slot mapping: 0 -> ra, 1 -> ra + 1, 2 -> rb & !1, 3 -> (rb & !1) + 1.
        let mut queries_by_reg: Vec<Vec<(usize, u8)>> = vec![Vec::new(); nw];
        for (&i, form) in forms {
            let Some(&(_, ra, rb, _)) = form.as_ref() else {
                continue;
            };
            let ib = self.block_of[i];
            if !self.bb_reachable[ib] {
                continue;
            }
            out.insert(i, [None, None, None, None]);
            let rb_base = rb & !1;
            let regs = [ra, ra + 1, rb_base, rb_base + 1];
            for (slot, &r) in regs.iter().enumerate() {
                if r < 256 {
                    let wi = watch_idx[r as usize];
                    if wi != u16::MAX {
                        queries_by_reg[wi as usize].push((i, slot as u8));
                    }
                }
            }
        }

        // Gather write events per watched register in a single pass over 0..n,
        // encoding `(i << 1) | is_call` so calls set bit 0 (`None`) cleanly.
        let mut events_by_reg: Vec<Vec<usize>> = vec![Vec::new(); nw];
        for (i, writes) in self.writes_r.iter().enumerate() {
            let mut masked = *writes;
            for w in 0..4 {
                masked[w] &= watch_mask[w];
            }
            if regset_is_empty(&masked) {
                continue;
            }
            let code = (i << 1) | usize::from(self.kind[i] == "call");
            crate::fast::regset_for_each(&masked, |r| {
                let wi = watch_idx[r as usize] as usize;
                events_by_reg[wi].push(code);
            });
        }

        // Reusable buffers for <= 63 writers fast path.
        let mut bb_in_u64 = vec![0u64; nb];
        let mut bb_out_u64 = vec![0u64; nb];
        let mut bb_gen_u64 = vec![0u64; nb];
        let mut bb_kill = vec![false; nb];
        let mut touched_blocks: Vec<usize> = Vec::new();
        let mut ev_range = vec![(0u32, 0u32); nb];

        for (wi, &_r) in watch.iter().enumerate() {
            let queries = &queries_by_reg[wi];
            if queries.is_empty() {
                continue;
            }
            let evs = &events_by_reg[wi];

            // Partition `evs` by block_of[i] and assign bit indices (bit 0 = None, bit 1 + ord = writers[ord]).
            // First count non-call writers and record (block, ev_start..ev_end).
            let mut writers: Vec<usize> = Vec::new();
            let mut ev_ordinals: Vec<usize> = Vec::with_capacity(evs.len());
            for &code in evs {
                if code & 1 != 0 {
                    ev_ordinals.push(0); // call -> bit 0
                } else {
                    let i = code >> 1;
                    let bit = 1 + writers.len();
                    writers.push(i);
                    ev_ordinals.push(bit);
                }
            }

            touched_blocks.clear();
            let mut pos = 0usize;
            while pos < evs.len() {
                let b = self.block_of[evs[pos] >> 1];
                let start = pos;
                pos += 1;
                while pos < evs.len() && self.block_of[evs[pos] >> 1] == b {
                    pos += 1;
                }
                touched_blocks.push(b);
                ev_range[b] = (start as u32, pos as u32);
            }

            if writers.len() <= 63 {
                // Fast u64 bitset path (covers >= 99.9% of cases).
                for &b in &touched_blocks {
                    let (s, e) = ev_range[b];
                    let mut b_gen = 0u64;
                    let mut kill = false;
                    for p in s as usize..e as usize {
                        let code = evs[p];
                        let bit = ev_ordinals[p];
                        if code & 1 != 0 {
                            b_gen |= 1u64;
                        } else {
                            let i = code >> 1;
                            if self.unguarded[i] {
                                kill = true;
                                b_gen = 1u64 << bit;
                            } else {
                                b_gen |= 1u64 << bit;
                            }
                        }
                    }
                    bb_gen_u64[b] = b_gen;
                    bb_kill[b] = kill;
                }

                for &b in &self.bb_rpo {
                    let init_in = if b == 0 { 1u64 } else { 0u64 };
                    bb_in_u64[b] = init_in;
                    bb_out_u64[b] = (if bb_kill[b] { 0 } else { init_in }) | bb_gen_u64[b];
                }
                let mut changed = true;
                while changed {
                    changed = false;
                    for &b in &self.bb_rpo {
                        let mut new_in = if b == 0 { 1u64 } else { 0u64 };
                        for &pb in &self.bb_pred[b] {
                            if self.bb_reachable[pb] {
                                new_in |= bb_out_u64[pb];
                            }
                        }
                        if new_in != bb_in_u64[b] {
                            bb_in_u64[b] = new_in;
                            let new_out = (if bb_kill[b] { 0 } else { new_in }) | bb_gen_u64[b];
                            if new_out != bb_out_u64[b] {
                                bb_out_u64[b] = new_out;
                                changed = true;
                            }
                        }
                    }
                }

                for &(form_i, slot) in queries {
                    let b = self.block_of[form_i];
                    let mut state = bb_in_u64[b];
                    if bb_gen_u64[b] != 0 || bb_kill[b] {
                        let (s, e) = ev_range[b];
                        for p in s as usize..e as usize {
                            let code = evs[p];
                            let ev_i = code >> 1;
                            if ev_i >= form_i {
                                break;
                            }
                            if code & 1 != 0 {
                                state |= 1u64;
                            } else if self.unguarded[ev_i] {
                                state = 1u64 << ev_ordinals[p];
                            } else {
                                state |= 1u64 << ev_ordinals[p];
                            }
                        }
                    }
                    let has_none = (state & 1) != 0;
                    let mut locs = Vec::with_capacity(state.count_ones() as usize);
                    let mut upper = state & !1u64;
                    while upper != 0 {
                        let bit = upper.trailing_zeros() as usize;
                        locs.push(writers[bit - 1]);
                        upper &= upper - 1;
                    }
                    out.get_mut(&form_i).unwrap()[slot as usize] = Some(RegDefs { has_none, locs });
                }

                for &b in &touched_blocks {
                    bb_gen_u64[b] = 0;
                    bb_kill[b] = false;
                }
            } else {
                // Multi-word bitset fallback when > 63 non-call writers exist for a single register.
                let words = (1 + writers.len()).div_ceil(64);
                let mut gen_words = vec![0u64; nb * words];
                for &b in &touched_blocks {
                    let (s, e) = ev_range[b];
                    let off = b * words;
                    let mut kill = false;
                    for p in s as usize..e as usize {
                        let code = evs[p];
                        let bit = ev_ordinals[p];
                        if code & 1 != 0 {
                            gen_words[off] |= 1u64;
                        } else {
                            let i = code >> 1;
                            if self.unguarded[i] {
                                kill = true;
                                gen_words[off..off + words].fill(0);
                            }
                            gen_words[off + (bit >> 6)] |= 1u64 << (bit & 63);
                        }
                    }
                    bb_kill[b] = kill;
                }

                let mut in_words = vec![0u64; nb * words];
                let mut out_words = vec![0u64; nb * words];
                for &b in &self.bb_rpo {
                    let off = b * words;
                    if b == 0 {
                        in_words[off] = 1;
                    }
                    if bb_kill[b] {
                        out_words[off..off + words].copy_from_slice(&gen_words[off..off + words]);
                    } else if b == 0 {
                        out_words[off..off + words].copy_from_slice(&gen_words[off..off + words]);
                        out_words[off] |= 1;
                    } else {
                        out_words[off..off + words].copy_from_slice(&gen_words[off..off + words]);
                    }
                }
                let mut new_in = vec![0u64; words];
                let mut changed = true;
                while changed {
                    changed = false;
                    for &b in &self.bb_rpo {
                        new_in.fill(0);
                        if b == 0 {
                            new_in[0] = 1;
                        }
                        for &pb in &self.bb_pred[b] {
                            if self.bb_reachable[pb] {
                                let poff = pb * words;
                                for w in 0..words {
                                    new_in[w] |= out_words[poff + w];
                                }
                            }
                        }
                        let off = b * words;
                        if new_in != in_words[off..off + words] {
                            in_words[off..off + words].copy_from_slice(&new_in);
                            let kill = bb_kill[b];
                            let mut out_changed = false;
                            for w in 0..words {
                                let nv = (if kill { 0 } else { new_in[w] }) | gen_words[off + w];
                                if nv != out_words[off + w] {
                                    out_words[off + w] = nv;
                                    out_changed = true;
                                }
                            }
                            if out_changed {
                                changed = true;
                            }
                        }
                    }
                }

                let mut state = vec![0u64; words];
                for &(form_i, slot) in queries {
                    let b = self.block_of[form_i];
                    let off = b * words;
                    state.copy_from_slice(&in_words[off..off + words]);
                    if bb_kill[b] || gen_words[off..off + words].iter().any(|&v| v != 0) {
                        let (s, e) = ev_range[b];
                        for p in s as usize..e as usize {
                            let code = evs[p];
                            let ev_i = code >> 1;
                            if ev_i >= form_i {
                                break;
                            }
                            let bit = ev_ordinals[p];
                            if code & 1 != 0 {
                                state[0] |= 1;
                            } else if self.unguarded[ev_i] {
                                state.fill(0);
                                state[bit >> 6] |= 1u64 << (bit & 63);
                            } else {
                                state[bit >> 6] |= 1u64 << (bit & 63);
                            }
                        }
                    }
                    let has_none = (state[0] & 1) != 0;
                    let mut locs = Vec::new();
                    for (w, &bits) in state.iter().enumerate() {
                        let mut left = if w == 0 { bits & !1u64 } else { bits };
                        while left != 0 {
                            let bit = (w << 6) + left.trailing_zeros() as usize;
                            locs.push(writers[bit - 1]);
                            left &= left - 1;
                        }
                    }
                    out.get_mut(&form_i).unwrap()[slot as usize] = Some(RegDefs { has_none, locs });
                }

                for &b in &touched_blocks {
                    bb_kill[b] = false;
                }
            }
        }
        out
    }
}

// ---- The plan ----

/// Compute the plan for a function, or None when it has nothing to do as pairs.
pub fn plan(src: &[Decoded], tables: &[(u64, Vec<u64>)], entries: &Option<HashSet<usize>>) -> Option<Plan> {
    if !src.iter().any(|d| d.name == PRODUCT) {
        return None;
    }
    let flow = Flow::new(src, tables, entries);
    plan_with_flow(src, &flow)
}

pub(crate) fn plan_with_flow(src: &[Decoded], flow: &Flow) -> Option<Plan> {
    if !flow.known || !src.iter().any(|d| d.name == PRODUCT) {
        return None;
    }
    debug_assert_eq!(flow.whole.len(), flow.n, "Flow built without pairing facts");
    let forms: BTreeMap<usize, Option<ProductForm>> = src
        .iter()
        .enumerate()
        .filter(|(_, d)| d.name == PRODUCT)
        .map(|(i, d)| (i, product_form(d)))
        .collect();
    let loads: BTreeMap<usize, (u32, bool)> = src
        .iter()
        .enumerate()
        .filter(|(_, d)| LOADS.contains(&d.name))
        .filter_map(|(i, d)| load_form(d).map(|v| (i, v)))
        .collect();
    if loads.is_empty() {
        return None;
    }
    let mut watch_mask = [0u64; 4];
    for &(rd, _) in loads.values() {
        for r in rd..rd + 4 {
            regset_set(&mut watch_mask, r);
        }
    }
    let mut watch = Vec::new();
    crate::fast::regset_for_each(&watch_mask, |r| watch.push(r));
    let reach = flow.reaching_for_forms(&forms, &watch);

    let mut excluded = FastSet::default();
    for _ in 0..forms.len() + 2 {
        let r = attempt(src, flow, &forms, &loads, &reach, &excluded);
        if r.bad.is_none() {
            let mut p = r.plan;
            if p.sites.is_empty() {
                return None;
            }
            sharing(&mut p, flow);
            return Some(p);
        }
        excluded.extend(r.bad.unwrap());
    }
    None
}

struct PlanWithBad {
    plan: Plan,
    bad: Option<FastSet<usize>>,
}

#[inline]
fn range_overlaps_4(a: u32, b: u32) -> bool {
    a < b + 4 && b < a + 4
}

#[inline]
fn regset_4(base: u32) -> RegSet {
    let mut m = [0u64; 4];
    for r in base..base + 4 {
        regset_set(&mut m, r);
    }
    m
}

// Sparse event kinds for accumulator register `r`.
const EV_Q: u8 = 1; // 'q': site read+write
const EV_C: u8 = 2; // 'c': site read-only
const EV_P: u8 = 3; // 'p': site write-only
const EV_W_ZERO: u8 = 4; // 'w': all_of zero writer
const EV_W_NONZERO: u8 = 5; // 'w': all_of non-zero writer
const EV_T_RO: u8 = 6; // 't': does not write r
const EV_T_RW: u8 = 7; // 't': writes r

#[inline]
fn ev_adjust(ev: u8, s: State) -> State {
    match ev {
        EV_Q | EV_C if s == State::T => State::Q,
        EV_T_RO | EV_T_RW if s == State::Q => State::T,
        _ => s,
    }
}

#[inline]
fn ev_after(ev: u8, s: State) -> State {
    match ev {
        EV_Q | EV_P => State::Q,
        EV_C => {
            if s == State::Z {
                State::Z
            } else {
                State::Q
            }
        }
        EV_W_ZERO => State::Z,
        EV_W_NONZERO => State::T,
        EV_T_RO => {
            if s == State::Z {
                State::Z
            } else {
                State::T
            }
        }
        EV_T_RW => State::T,
        _ => s,
    }
}

#[inline]
fn ev_want(ev: u8) -> u8 {
    match ev {
        EV_Q | EV_C => WANTS_Q,
        EV_T_RO | EV_T_RW => WANTS_T,
        EV_W_ZERO | EV_W_NONZERO | EV_P => DEAD,
        _ => DEAD,
    }
}

#[inline]
fn find_reg_defs(reach: &FormReach, i: usize, ra: u32, rb_base: u32, k: u32) -> Option<&RegDefs> {
    let slots = reach.get(&i)?;
    let idx = if k == ra {
        0
    } else if k == ra + 1 {
        1
    } else if k == rb_base {
        2
    } else if k == rb_base + 1 {
        3
    } else {
        return None;
    };
    slots[idx].as_ref()
}

fn attempt(
    src: &[Decoded],
    flow: &Flow,
    forms: &BTreeMap<usize, Option<ProductForm>>,
    loads: &BTreeMap<usize, (u32, bool)>,
    reach: &FormReach,
    excluded: &FastSet<usize>,
) -> PlanWithBad {
    let n = src.len();
    let mut out = Plan::new();

    // Which products could be half of a pair.
    let source = |i: usize, ra: u32, rb_base: u32, registers: &[u32]| -> Option<(u32, bool)> {
        let mut chosen: Option<(u32, bool)> = None;
        for &k in registers {
            let defs = find_reg_defs(reach, i, ra, rb_base, k)?;
            if defs.has_none || defs.locs.is_empty() {
                return None;
            }
            for &j in &defs.locs {
                let &pair = loads.get(&j)?;
                match chosen {
                    None => chosen = Some(pair),
                    Some(prev) if prev == pair => {}
                    Some(_) => return None,
                }
            }
        }
        let (rd, transposed) = chosen?;
        if registers.iter().all(|&k| rd <= k && k <= rd + 3) {
            Some((rd, transposed))
        } else {
            None
        }
    };

    #[derive(Clone)]
    struct Cand {
        d: u32,
        c: Option<u32>,
        ra: u32,
        a: u32,
        rb: u32,
        p: u32,
        e: u32,
        transposed: bool,
        first: bool,
    }

    let mut candidate: BTreeMap<usize, Cand> = BTreeMap::new();
    for (&i, form) in forms {
        if excluded.contains(&i) {
            continue;
        }
        let Some(&(rd, ra, rb, rc)) = form.as_ref() else {
            continue;
        };
        let b_base = rb - rb % 2;
        let a = source(i, ra, b_base, &[ra, ra + 1]);
        let b = source(i, ra, b_base, &[rb]);
        let (Some(a), Some(b)) = (a, b) else { continue };
        if a.1 || a.0 == b.0 || (ra as i32 - a.0 as i32) % 2 != 0 {
            continue;
        }
        let first = source(i, ra, b_base, &[b_base, b_base + 1]) == Some(b);
        candidate.insert(
            i,
            Cand {
                d: rd,
                c: rc,
                ra: a.0,
                a: (ra - a.0) / 2,
                rb: b.0,
                p: (rb - b.0) / 2,
                e: (rb - b.0) % 2,
                transposed: b.1,
                first,
            },
        );
    }

    let touches_mask = |k: usize, mask: &RegSet, writes_only: bool| -> bool {
        regset_intersects(&flow.writes_r[k], mask) || (!writes_only && regset_intersects(&flow.reads_r[k], mask))
    };

    // The main loop: build pairs, check loads, remove what doesn't work, repeat.
    let mut whole: HashMap<usize, LoadRole>;
    let mut used = FastSet::default();
    let mut gone = FastSet::default();
    let mut unpaired = Vec::new();
    let mut roles: HashMap<usize, Option<LoadRole>> = HashMap::new();

    loop {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        used.clear();
        for (&i, c) in &candidate {
            if used.contains(&i) || !c.first {
                continue;
            }
            let mine_mask = regset_4(c.d);
            for j in i + 1..n {
                // (As `reference/pairing.py`: an entry, or an instruction after one that is not
                // plain.)
                if flow.entered.contains(&j) || flow.kind[j - 1] != "plain" {
                    break;
                }
                if let Some(o) = candidate.get(&j)
                    && !used.contains(&j)
                    && (o.ra, o.a, o.rb, o.p, o.transposed) == (c.ra, c.a, c.rb, c.p, c.transposed)
                    && o.e != c.e
                    && !range_overlaps_4(c.d, o.d)
                {
                    let apart = [c, o].iter().all(|x| {
                        x.c == Some(x.d)
                            || x.c
                                .is_none_or(|xc| !range_overlaps_4(xc, c.d) && !range_overlaps_4(xc, o.d))
                    });
                    if apart {
                        let theirs_mask = regset_4(o.d);
                        let mut late_mask = [0u64; 4];
                        let has_late = match o.c {
                            Some(oc) if oc != o.d => {
                                late_mask = regset_4(oc);
                                regset_sub(&mut late_mask, &theirs_mask);
                                !regset_is_empty(&late_mask)
                            }
                            _ => false,
                        };
                        let ok = !(i + 1..j).any(|k| {
                            touches_mask(k, &theirs_mask, false) || (has_late && touches_mask(k, &late_mask, true))
                        });
                        if ok {
                            pairs.push((i, j));
                            used.insert(i);
                            used.insert(j);
                        }
                    }
                    break;
                }
                if touches_mask(j, &mine_mask, false)
                    || loads.get(&j).is_some_and(|&(lrd, _)| lrd == c.ra || lrd == c.rb)
                {
                    break;
                }
            }
        }

        let mut sites: BTreeMap<usize, SiteEntry> = BTreeMap::new();
        for &(i, j) in &pairs {
            let ci = &candidate[&i];
            let cj = &candidate[&j];
            let (first, second) = if ci.e == 0 { (ci, cj) } else { (cj, ci) };
            for &(k, half, partner) in &[(i, 0u32, j), (j, 1u32, i)] {
                sites.insert(
                    k,
                    SiteEntry {
                        half,
                        partner,
                        x: first.d,
                        y: second.d,
                        cx: first.c,
                        cy: second.c,
                        ra: ci.ra,
                        a: ci.a,
                        rb: ci.rb,
                        p: ci.p,
                        transposed: ci.transposed,
                    },
                );
            }
        }

        // Loads: whole rows when everything their results reach is a product of a pair, in one role.
        roles.clear();
        for (&u, s) in &sites {
            if s.half != 0 {
                continue;
            }
            let ra_base = s.ra + 2 * s.a;
            let rb_base = s.rb + 2 * s.p;
            let b_role = if s.transposed { LoadRole::BT } else { LoadRole::B };
            let operands = [([ra_base, ra_base + 1], LoadRole::A), ([rb_base, rb_base + 1], b_role)];
            for (regs, as_) in &operands {
                for &k in regs {
                    if let Some(defs) = find_reg_defs(reach, u, ra_base, rb_base, k) {
                        for &j in &defs.locs {
                            match roles.get(&j) {
                                None => {
                                    roles.insert(j, Some(*as_));
                                }
                                Some(Some(prev)) if *prev == *as_ => {}
                                Some(Some(_)) => {
                                    roles.insert(j, None);
                                }
                                Some(None) => {}
                            }
                        }
                    }
                }
            }
        }

        whole = roles.iter().filter_map(|(&j, &as_)| Some((j, as_?))).collect();

        // Check which pairs stay.
        gone.clear();
        for (&i, s) in &sites {
            if s.half != 0 {
                continue;
            }
            let ra_base = s.ra + 2 * s.a;
            let rb_base = s.rb + 2 * s.p;
            let b_allowed = [Some(LoadRole::B), Some(LoadRole::BT)];
            let a_bad = [ra_base, ra_base + 1].iter().any(|&k| {
                find_reg_defs(reach, i, ra_base, rb_base, k)
                    .is_some_and(|defs| defs.locs.iter().any(|&j| whole.get(&j) != Some(&LoadRole::A)))
            });
            let b_bad = [rb_base, rb_base + 1].iter().any(|&k| {
                find_reg_defs(reach, i, ra_base, rb_base, k)
                    .is_some_and(|defs| defs.locs.iter().any(|&j| !b_allowed.contains(&whole.get(&j).copied())))
            });
            if a_bad || b_bad {
                gone.insert(i);
                gone.insert(s.partner);
            }
        }

        unpaired.clear();
        for &i in candidate.keys() {
            if !sites.contains_key(&i) {
                unpaired.push(i);
            }
        }
        if gone.is_empty() && unpaired.is_empty() {
            out.sites = sites;
            break;
        }
        for i in &gone {
            candidate.remove(i);
        }
        for i in &unpaired {
            candidate.remove(i);
        }
    }

    // Build load entries.
    let sites_set: FastSet<usize> = out.sites.keys().copied().collect();
    out.loads = whole
        .iter()
        .map(|(&j, &as_)| {
            let (rd, _) = loads[&j];
            (
                j,
                LoadEntry {
                    role: as_,
                    rd,
                    also: flow.escapes(src, j, rd, &sites_set),
                },
            )
        })
        .collect();

    if out.sites.is_empty() {
        return PlanWithBad { plan: out, bad: None };
    }

    // ---- The accumulator registers: sparse basic-block state analysis and conversions ----
    let mut of_site_masks: Vec<(usize, RegSet, RegSet)> = Vec::with_capacity(out.sites.len());
    let mut site_reg_map: HashMap<usize, usize> = HashMap::with_capacity(out.sites.len());
    let mut all_regs_mask = [0u64; 4];
    for (&i, s) in &out.sites {
        let mut wrote = regset_4(s.x);
        regset_union(&mut wrote, &regset_4(s.y));
        let read = if s.half == 0 {
            let mut rm = [0u64; 4];
            if let Some(cx) = s.cx {
                regset_union(&mut rm, &regset_4(cx));
            }
            if let Some(cy) = s.cy {
                regset_union(&mut rm, &regset_4(cy));
            }
            rm
        } else {
            wrote
        };
        regset_union(&mut all_regs_mask, &wrote);
        regset_union(&mut all_regs_mask, &read);
        site_reg_map.insert(i, of_site_masks.len());
        of_site_masks.push((i, read, wrote));
    }

    let mut registers: Vec<u32> = Vec::new();
    let mut reg_slot = [u16::MAX; 256];
    crate::fast::regset_for_each(&all_regs_mask, |r| {
        reg_slot[r as usize] = registers.len() as u16;
        registers.push(r);
    });
    let num_regs = registers.len();

    // Build per-register events and common barriers in ONE single pass over 0..n.
    let mut events_by_reg: Vec<Vec<(usize, u8)>> = vec![Vec::new(); num_regs];
    let mut barriers: Vec<(usize, bool)> = Vec::new(); // (insn_idx, unknown_call_writes_all)
    for i in 0..n {
        if let Some(&sidx) = site_reg_map.get(&i) {
            let (_, ref read_m, ref wrote_m) = of_site_masks[sidx];
            let mut both = *read_m;
            regset_union(&mut both, wrote_m);
            crate::fast::regset_for_each(&both, |r| {
                let slot = reg_slot[r as usize] as usize;
                let is_r = regset_test(read_m, r);
                let is_w = regset_test(wrote_m, r);
                let code = match (is_r, is_w) {
                    (true, true) => EV_Q,
                    (true, false) => EV_C,
                    (false, true) => EV_P,
                    (false, false) => unreachable!(),
                };
                events_by_reg[slot].push((i, code));
            });
            continue;
        }
        let is_unknown_call = flow.kind[i] == "call" && !matches!(flow.called.get(&i), Some(Some(_)));
        let is_barrier =
            (flow.kind[i] == "other" && !matches!(src[i].mnemonic, "BSSY" | "BSYNC")) || flow.removes(src, i);
        if is_unknown_call || is_barrier {
            barriers.push((i, is_unknown_call));
            continue;
        }
        let mut r_masked = flow.reads_r[i];
        let mut w_masked = flow.writes_r[i];
        for w in 0..4 {
            r_masked[w] &= all_regs_mask[w];
            w_masked[w] &= all_regs_mask[w];
        }
        let mut rw = r_masked;
        regset_union(&mut rw, &w_masked);
        if regset_is_empty(&rw) {
            continue;
        }
        let is_zero = zero_writer(&src[i]);
        let call_flag = flow.kind[i] == "call";
        let unguarded_noncall = !call_flag && flow.unguarded[i];
        crate::fast::regset_for_each(&rw, |r| {
            let slot = reg_slot[r as usize] as usize;
            let reads = regset_test(&r_masked, r);
            let writes = regset_test(&w_masked, r);
            let code = if reads {
                if writes { EV_T_RW } else { EV_T_RO }
            } else if unguarded_noncall {
                if is_zero { EV_W_ZERO } else { EV_W_NONZERO }
            } else if writes {
                EV_T_RW
            } else {
                EV_T_RO
            };
            events_by_reg[slot].push((i, code));
        });
    }

    let nb = flow.blocks.len();
    let mut bad = FastSet::default();
    let mut forced: FastSet<(u32, usize)> = FastSet::default();
    let mut pending: Vec<u32> = registers.clone();

    let mut bb_want_def: Vec<Option<u8>> = vec![None; nb];
    let mut bb_want = vec![DEAD; nb];
    let mut bb_mode = vec![State::T; nb];
    let mut bb_zero = vec![true; nb];
    let mut bb_in: Vec<Option<State>> = vec![None; nb];
    let mut bb_out: Vec<Option<State>> = vec![None; nb];
    let mut bb_ev_range: Vec<(u32, u32)> = vec![(0, 0); nb];
    let mut merged_evs: Vec<(usize, u8)> = Vec::new();
    let mut touched_blocks: Vec<usize> = Vec::new();
    let mut again: FastSet<(u32, usize)> = FastSet::default();

    let refuse = |bad: &mut FastSet<usize>, r: u32| {
        for &(i, ref read_m, ref wrote_m) in &of_site_masks {
            if regset_test(read_m, r) || regset_test(wrote_m, r) {
                bad.insert(i);
            }
        }
    };

    while let Some(r) = pending.first().copied() {
        pending.remove(0);
        again.clear();

        for cs in out.before.values_mut() {
            cs.retain(|&(cr, _)| cr != r);
        }
        for cs in out.after.values_mut() {
            cs.retain(|&(cr, _)| cr != r);
        }
        out.before.retain(|_, cs| !cs.is_empty());
        out.after.retain(|_, cs| !cs.is_empty());

        let r_evs = &events_by_reg[reg_slot[r as usize] as usize];
        // Merge sorted `r_evs` and `barriers` (no instruction appears in both).
        merged_evs.clear();
        merged_evs.reserve(r_evs.len() + barriers.len());
        let (mut ei, mut bi) = (0usize, 0usize);
        while ei < r_evs.len() && bi < barriers.len() {
            let (ri, rc) = r_evs[ei];
            let (b_idx, unk_call) = barriers[bi];
            if ri < b_idx {
                merged_evs.push((ri, rc));
                ei += 1;
            } else {
                let b_code = if unk_call || flow.writes_reg(b_idx, r) {
                    EV_T_RW
                } else {
                    EV_T_RO
                };
                merged_evs.push((b_idx, b_code));
                bi += 1;
            }
        }
        merged_evs.extend_from_slice(&r_evs[ei..]);
        for &(b_idx, unk_call) in &barriers[bi..] {
            let b_code = if unk_call || flow.writes_reg(b_idx, r) {
                EV_T_RW
            } else {
                EV_T_RO
            };
            merged_evs.push((b_idx, b_code));
        }

        touched_blocks.clear();
        let mut pos = 0usize;
        while pos < merged_evs.len() {
            let b = flow.block_of[merged_evs[pos].0];
            let start = pos;
            pos += 1;
            while pos < merged_evs.len() && flow.block_of[merged_evs[pos].0] == b {
                pos += 1;
            }
            touched_blocks.push(b);
            bb_ev_range[b] = (start as u32, pos as u32);
            bb_want_def[b] = Some(ev_want(merged_evs[start].1));
        }

        // Backward: what the register is next used as (at every basic block head).
        for b in 0..nb {
            bb_want[b] = bb_want_def[b].unwrap_or(DEAD);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..nb).rev() {
                if bb_want_def[b].is_none() {
                    let new = flow.bb_succ[b].iter().map(|&sb| bb_want[sb]).max().unwrap_or(DEAD);
                    if new != bb_want[b] {
                        bb_want[b] = new;
                        changed = true;
                    }
                }
            }
        }

        for b in 0..nb {
            let start_b = flow.blocks[b].0;
            bb_mode[b] = if bb_want[b] == WANTS_Q && !forced.contains(&(r, start_b)) && start_b != 0 {
                State::Q
            } else {
                State::T
            };
            bb_zero[b] = start_b != 0;
            bb_in[b] = None;
            bb_out[b] = None;
        }

        // Forward: state at block heads (`bb_in[b]`) and block tails (`bb_out[b]`).
        let mut fchanged = true;
        while fchanged {
            fchanged = false;
            for b in 0..nb {
                let start_b = flow.blocks[b].0;
                let s = {
                    let z = bb_zero[b]
                        && flow.bb_pred[b]
                            .iter()
                            .all(|&pb| bb_out[pb].is_none_or(|st| st == State::Z));
                    if !z && bb_zero[b] {
                        bb_zero[b] = false;
                    }
                    if z && !flow.bb_pred[b].is_empty() {
                        State::Z
                    } else {
                        bb_mode[b]
                    }
                };
                let (ev_s, ev_e) = bb_ev_range[b];
                let head_ev = if ev_s < ev_e && merged_evs[ev_s as usize].0 == start_b {
                    Some(merged_evs[ev_s as usize].1)
                } else {
                    None
                };
                let adj_head = match head_ev {
                    Some(ev) => ev_adjust(ev, s),
                    None => s,
                };
                if Some(adj_head) != bb_in[b] {
                    bb_in[b] = Some(adj_head);
                    fchanged = true;
                }
                // Apply events from start_b..end_b to compute bb_out[b].
                let mut cur = match head_ev {
                    Some(ev) => ev_after(ev, adj_head),
                    None => adj_head,
                };
                let int_start = if head_ev.is_some() { ev_s + 1 } else { ev_s };
                for &(_, ev) in &merged_evs[int_start as usize..ev_e as usize] {
                    let a = ev_adjust(ev, cur);
                    cur = ev_after(ev, a);
                }
                bb_out[b] = Some(cur);
            }
        }

        // The conversions.
        for b in 0..nb {
            let start_b = flow.blocks[b].0;
            let (ev_s, ev_e) = bb_ev_range[b];
            let head_ev = if ev_s < ev_e && merged_evs[ev_s as usize].0 == start_b {
                Some(merged_evs[ev_s as usize].1)
            } else {
                None
            };

            // 1. Block head `!flow.simple(start_b)`.
            let head_handled = {
                let arrives = if matches!(head_ev, Some(EV_Q | EV_C | EV_T_RO | EV_T_RW)) {
                    if bb_zero[b]
                        && !flow.bb_pred[b].is_empty()
                        && flow.bb_pred[b].iter().all(|&pb| bb_out[pb] == Some(State::Z))
                    {
                        State::Z
                    } else {
                        bb_mode[b]
                    }
                } else {
                    bb_in[b].unwrap_or(State::T)
                };
                if bb_want[b] != DEAD && arrives != State::Z {
                    if matches!(head_ev, Some(EV_Q | EV_C)) && arrives == State::T {
                        if !flow.whole[start_b] {
                            refuse(&mut bad, r);
                        }
                        out.before.entry(start_b).or_default().push((r, State::Q));
                    }
                    for &p in &flow.pred[start_b] {
                        let pb = flow.block_of[p];
                        let came = bb_out[pb].unwrap();
                        if came == State::Z || came == arrives {
                            continue;
                        }
                        let kind = flow.kind[p];
                        let both = kind == "cond"
                            && flow.bb_succ[pb].iter().all(|&sb| {
                                bb_want[sb] != DEAD
                                    && bb_mode[sb] == arrives
                                    && !(bb_zero[sb] && bb_in[sb] == Some(State::Z))
                            });
                        let (place_is_before, ok) = if kind == "jump" || both {
                            (true, flow.whole[p])
                        } else if kind == "plain" && start_b == p + 1 {
                            (false, flow.goes_whole(src, p) || flow.lead[start_b])
                        } else if kind == "cond" && start_b == p + 1 && flow.target(p) != Some(start_b) {
                            (false, flow.goes_whole(src, p))
                        } else {
                            (false, false)
                        };
                        if !ok && arrives == State::Q {
                            again.insert((r, start_b));
                        } else if !ok {
                            refuse(&mut bad, r);
                        } else {
                            let place = if place_is_before {
                                &mut out.before
                            } else {
                                &mut out.after
                            };
                            let entry = place.entry(p).or_default();
                            if !entry.contains(&(r, arrives)) {
                                entry.push((r, arrives));
                            }
                        }
                    }
                }
                true
            };
            let _ = head_handled;

            // 2. Interior events inside block `b` (`flow.simple(ev_i)`).
            let int_start = if head_ev.is_some() { ev_s + 1 } else { ev_s };
            if int_start < ev_e {
                let st0 = bb_in[b].unwrap();
                let mut came = match head_ev {
                    Some(ev) => ev_after(ev, st0),
                    None => st0,
                };
                for &(ev_i, ev) in &merged_evs[int_start as usize..ev_e as usize] {
                    if (matches!(ev, EV_Q | EV_C) && came == State::T)
                        || (matches!(ev, EV_T_RO | EV_T_RW) && came == State::Q)
                    {
                        if !flow.whole[ev_i] {
                            refuse(&mut bad, r);
                        }
                        let to = if matches!(ev, EV_Q | EV_C) { State::Q } else { State::T };
                        out.before.entry(ev_i).or_default().push((r, to));
                    }
                    came = ev_after(ev, ev_adjust(ev, came));
                }
            }
        }

        for &b in &touched_blocks {
            bb_want_def[b] = None;
            bb_ev_range[b] = (0, 0);
        }

        let mut added_forced = false;
        for &pair in &again {
            if forced.insert(pair) {
                added_forced = true;
            }
        }
        if added_forced {
            pending.insert(0, r);
        }
    }

    out.before.retain(|_, cs| !cs.is_empty());
    out.after.retain(|_, cs| !cs.is_empty());

    if !bad.is_empty() {
        out.before.clear();
        out.after.clear();
        return PlanWithBad {
            plan: out,
            bad: Some(bad),
        };
    }

    PlanWithBad { plan: out, bad: None }
}

fn sharing(plan: &mut Plan, flow: &Flow) {
    let nb = flow.blocks.len();
    let size = plan.rows();
    let rds: Vec<u32> = size.keys().copied().collect();
    let nr = rds.len();
    if nr == 0 {
        return;
    }

    let mut reads_map: Vec<Vec<usize>> = vec![Vec::new(); nr];
    let mut writes_map: Vec<Vec<usize>> = vec![Vec::new(); nr];
    let mut rd_idx: [u16; 256] = [u16::MAX; 256];
    for (idx, &rd) in rds.iter().enumerate() {
        if rd < 256 {
            rd_idx[rd as usize] = idx as u16;
        }
    }

    for (&i, s) in &plan.sites {
        for &rd in &[s.ra, s.rb] {
            if rd < 256 {
                let idx = rd_idx[rd as usize];
                if idx != u16::MAX && (s.half != 0 || size[&rd] == 8) {
                    let v = &mut reads_map[idx as usize];
                    if v.last() != Some(&i) {
                        v.push(i);
                    }
                }
            }
        }
    }
    for (&j, v) in &plan.loads {
        if v.rd < 256 {
            let idx = rd_idx[v.rd as usize];
            if idx != u16::MAX {
                writes_map[idx as usize].push(j);
            }
        }
    }

    let mut reads_sets: Vec<FastSet<usize>> = reads_map.iter().map(|v| v.iter().copied().collect()).collect();
    let _ = &mut reads_sets;

    // Basic-block liveness for each row rd:
    // For each block `b`, `alive_at(i)`: live-out of instruction `i`.
    // `live_entry[b]` (live before `start_b` executes) = `in_block_gen[b] || (live_out_tail[b] && !block_writes[b])`,
    // where `live_out_tail[b]` = `any sb in bb_succ[b]: live_entry[sb]`.
    let mut bb_live_entry = vec![false; nb];
    let mut bb_gen = vec![false; nb];
    let mut bb_transparent = vec![true; nb];
    let mut bb_events: Vec<(u32, u32)> = vec![(0, 0); nb];
    let mut evs: Vec<(usize, bool, bool)> = Vec::new(); // (insn, is_read, is_write)
    let mut touched_b: Vec<usize> = Vec::new();
    let mut clash: Vec<Vec<bool>> = vec![vec![false; nr]; nr];

    for b_idx in 0..nr {
        let rm = &reads_map[b_idx];
        let wm = &writes_map[b_idx];

        // Merge sorted reads and writes for row b_idx.
        evs.clear();
        let (mut ri, mut wi) = (0usize, 0usize);
        while ri < rm.len() && wi < wm.len() {
            if rm[ri] < wm[wi] {
                evs.push((rm[ri], true, false));
                ri += 1;
            } else if wm[wi] < rm[ri] {
                evs.push((wm[wi], false, true));
                wi += 1;
            } else {
                evs.push((rm[ri], true, true));
                ri += 1;
                wi += 1;
            }
        }
        while ri < rm.len() {
            evs.push((rm[ri], true, false));
            ri += 1;
        }
        while wi < wm.len() {
            evs.push((wm[wi], false, true));
            wi += 1;
        }

        touched_b.clear();
        let mut pos = 0usize;
        while pos < evs.len() {
            let blk = flow.block_of[evs[pos].0];
            let start = pos;
            pos += 1;
            while pos < evs.len() && flow.block_of[evs[pos].0] == blk {
                pos += 1;
            }
            touched_b.push(blk);
            bb_events[blk] = (start as u32, pos as u32);
            let mut b_gen = false;
            let mut transp = true;
            for &(_, is_r, is_w) in evs[start..pos].iter().rev() {
                if is_w {
                    b_gen = false;
                    transp = false;
                }
                if is_r {
                    b_gen = true;
                }
            }
            bb_gen[blk] = b_gen;
            bb_transparent[blk] = transp;
        }

        bb_live_entry.fill(false);
        let mut changed = true;
        while changed {
            changed = false;
            for blk in (0..nb).rev() {
                let tail_live = flow.bb_succ[blk].iter().any(|&sb| bb_live_entry[sb]);
                let entry_live = bb_gen[blk] || (tail_live && bb_transparent[blk]);
                if entry_live != bb_live_entry[blk] {
                    bb_live_entry[blk] = entry_live;
                    changed = true;
                }
            }
        }

        // Check every write `j` of every other row `a_idx`: is row `b_idx` live out of `j` or read at `j`?
        for (a_idx, a_writes) in writes_map.iter().enumerate() {
            if a_idx == b_idx {
                continue;
            }
            let hit = a_writes.iter().any(|&j| {
                if reads_sets[b_idx].contains(&j) {
                    return true;
                }
                let blk = flow.block_of[j];
                let mut cur = flow.bb_succ[blk].iter().any(|&sb| bb_live_entry[sb]);
                let (s, e) = bb_events[blk];
                for &(ev_i, is_r, is_w) in evs[s as usize..e as usize].iter().rev() {
                    if ev_i <= j {
                        break;
                    }
                    if is_w {
                        cur = false;
                    }
                    if is_r {
                        cur = true;
                    }
                }
                cur
            });
            if hit {
                clash[a_idx][b_idx] = true;
                clash[b_idx][a_idx] = true;
            }
        }

        for &blk in &touched_b {
            bb_gen[blk] = false;
            bb_transparent[blk] = true;
            bb_events[blk] = (0, 0);
        }
    }

    let mut order: Vec<usize> = (0..nr).collect();
    order.sort_by(|&a, &b| size[&rds[b]].cmp(&size[&rds[a]]).then(rds[a].cmp(&rds[b])));

    let mut classes: Vec<Vec<usize>> = Vec::new();
    for idx in order {
        let mut placed = false;
        for members in &mut classes {
            if !members.iter().any(|&m| clash[idx][m]) {
                members.push(idx);
                placed = true;
                break;
            }
        }
        if !placed {
            classes.push(vec![idx]);
        }
    }

    plan.share = BTreeMap::new();
    for members in &classes {
        let rep = rds[members[0]];
        for &m in members {
            plan.share.insert(rds[m], rep);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_products_no_plan() {
        let src: Vec<Decoded> = Vec::new();
        let tables = Vec::new();
        let entries = Some(HashSet::new());
        assert!(plan(&src, &tables, &entries).is_none());
    }

    #[test]
    fn product_form_in_place() {
        let mut dec = crate::decode::Decoder::new(75);
        let word = crate::assemble::assemble(
            "hmma_x8_",
            &[
                ("Rd", crate::assemble::OpVal::Int(4)),
                ("Ra", crate::assemble::OpVal::Int(0)),
                ("Rb", crate::assemble::OpVal::Int(2)),
                ("Rc", crate::assemble::OpVal::Int(4)),
                ("dstfmt", crate::assemble::OpVal::Int(1)),
            ],
        );
        if let Some(w) = word
            && let Some(d) = dec.decode(w)
        {
            let form = product_form(&d);
            assert!(form.is_some());
            let (rd, ra, rb, rc) = form.unwrap();
            assert_eq!(rd, 4);
            assert_eq!(ra, 0);
            assert_eq!(rb, 2);
            assert_eq!(rc, Some(4));
        }
    }

    #[test]
    fn product_form_zero_accumulator() {
        let mut dec = crate::decode::Decoder::new(75);
        let word = crate::assemble::assemble(
            "hmma_x8_",
            &[
                ("Rd", crate::assemble::OpVal::Int(4)),
                ("Ra", crate::assemble::OpVal::Int(0)),
                ("Rb", crate::assemble::OpVal::Int(2)),
                ("Rc", crate::assemble::OpVal::Int(255)),
                ("dstfmt", crate::assemble::OpVal::Int(1)),
            ],
        );
        if let Some(w) = word
            && let Some(d) = dec.decode(w)
        {
            let form = product_form(&d);
            assert!(form.is_some());
            let (_, _, _, rc) = form.unwrap();
            assert_eq!(rc, None);
        }
    }

    #[test]
    fn product_form_different_accumulator() {
        let mut dec = crate::decode::Decoder::new(75);
        let word = crate::assemble::assemble(
            "hmma_x8_",
            &[
                ("Rd", crate::assemble::OpVal::Int(4)),
                ("Ra", crate::assemble::OpVal::Int(0)),
                ("Rb", crate::assemble::OpVal::Int(2)),
                ("Rc", crate::assemble::OpVal::Int(8)),
                ("dstfmt", crate::assemble::OpVal::Int(1)),
            ],
        );
        if let Some(w) = word
            && let Some(d) = dec.decode(w)
        {
            let form = product_form(&d);
            assert!(form.is_some());
            let (rd, _, _, rc) = form.unwrap();
            assert_eq!(rd, 4);
            assert_eq!(rc, Some(8));
        }
    }

    #[test]
    fn load_form_plain_and_transposed() {
        let mut dec = crate::decode::Decoder::new(75);
        let word = crate::assemble::assemble(
            "ldsm__sImmOffset",
            &[
                ("Rd", crate::assemble::OpVal::Int(0)),
                ("Ra", crate::assemble::OpVal::Int(10)),
                ("Ra_offset", crate::assemble::OpVal::Int(0)),
                ("sz", crate::assemble::OpVal::Int(0)),
                ("num", crate::assemble::OpVal::Int(2)),
                ("mode", crate::assemble::OpVal::Int(0)),
            ],
        );
        if let Some(w) = word
            && let Some(d) = dec.decode(w)
        {
            let form = load_form(&d);
            assert!(form.is_some());
            assert_eq!(form.unwrap(), (0, false));
        }
    }

    #[test]
    fn kind_of_bra_u_is_jump() {
        let _dec = crate::decode::Decoder::new(75);
        assert!(matches!(0u32, 0 | 1));
        assert!(!matches!(2u32, 0 | 1));
    }

    use crate::decode::Decoder;

    #[test]
    fn a_guard_that_r2p_may_change_hides_nothing_after_it() {
        let words: [u128; 8] = [
            0x000fc20000000200000000000008783b,
            0x000fc2000000020000000000010c783b,
            0x000fc200000010ff0000000c0810723c,
            0x000fc200000010ff0000000d0814723c,
            0x000fc20000000f000000000000080802,
            0x000fc200000000000000000102007804,
            0x000fc20000000f000000000800180202,
            0x000fc20003800000000000000000794d,
        ];
        let mut decoder = Decoder::new(75);
        let mut src: Vec<Decoded> = words.into_iter().map(|w| decoder.decode(w).unwrap()).collect();
        assert_eq!(src[5].mnemonic, "R2P");
        let entries = Some(HashSet::new());
        let p = plan(&src, &[], &entries).unwrap();
        assert!(
            p.loads[&0].also,
            "the load of R8 is read after R2P changed the guard that hid it"
        );
        src[5] = decoder.decode(0x000fc200000000000000000202007804).unwrap();
        assert_eq!((src[5].mnemonic, src[5].op_or_name("Sb", 0)), ("R2P", 2));
        let p = plan(&src, &[], &entries).unwrap();
        assert!(!p.loads[&0].also);
    }

    fn make_guarded_r8_writer(guarded: bool) -> Decoded {
        Decoded {
            word: 0,
            class_id: 0,
            kind: crate::decode::Kind::FXU,
            mnemonic: "IADD3",
            name: "iadd3_synthetic",
            ops: vec![
                (OP_PG, if guarded { 0 } else { 7 }),
                (OP_PG_AT_NOT, 0),
                (OP_UPG, 7),
                (OP_UPG_AT_NOT, 0),
                (OP_RD, 8),
            ],
            reads: if guarded {
                vec![(crate::decode::LETTER_P, 0, 1)]
            } else {
                Vec::new()
            },
            writes: vec![(LETTER_R, 8)],
        }
    }

    fn make_ldsm(rd: u32) -> Decoded {
        Decoded {
            word: 0,
            class_id: 0,
            kind: crate::decode::Kind::MioSlow,
            mnemonic: "LDSM",
            name: "ldsm__sImmOffset",
            ops: vec![
                (OP_PG, 7),
                (OP_PG_AT_NOT, 0),
                (OP_UPG, 7),
                (OP_UPG_AT_NOT, 0),
                (OP_RD, rd),
                (OP_RA, 20),
                (OP_SZ, 0),
                (OP_NUM, 2),
                (OP_MODE, 0),
            ],
            reads: vec![(LETTER_R, 20, 0)],
            writes: (rd..rd + 4).map(|r| (LETTER_R, r)).collect(),
        }
    }

    fn make_hmma(rd: u32, ra: u32, rb: u32, rc: u32) -> Decoded {
        let mut reads = vec![(LETTER_R, ra, 0), (LETTER_R, ra + 1, 0), (LETTER_R, rb, 0)];
        if rc != 255 {
            for r in rc..rc + 4 {
                reads.push((LETTER_R, r, 0));
            }
        }
        Decoded {
            word: 0,
            class_id: 0,
            kind: crate::decode::Kind::HMMA,
            mnemonic: "HMMA",
            name: "hmma_x8_",
            ops: vec![
                (OP_PG, 7),
                (OP_PG_AT_NOT, 0),
                (OP_UPG, 7),
                (OP_UPG_AT_NOT, 0),
                (OP_DSTFMT, 1),
                (OP_RD, rd),
                (OP_RA, ra),
                (OP_RB, rb),
                (OP_RC, rc),
            ],
            reads,
            writes: (rd..rd + 4).map(|r| (LETTER_R, r)).collect(),
        }
    }

    #[test]
    fn reaching_for_forms_multiword_fallback_handles_over_63_writers() {
        let mut decoder = Decoder::new(75);
        // 1 unguarded + 65 guarded writers to R8 before `hmma_idx0`, then unguarded `ldsm`s
        // writing R12..R15 and R8..R11 before `hmma_idx1` (67 writers total on R8 > 63 -> 2-word bitset).
        let mut src: Vec<Decoded> = Vec::with_capacity(72);
        src.push(make_guarded_r8_writer(false));
        for _ in 0..65 {
            src.push(make_guarded_r8_writer(true));
        }

        let hmma_idx0 = src.len();
        src.push(make_hmma(4, 8, 12, 4));

        src.push(make_ldsm(12));
        let ldsm_idx = src.len();
        src.push(make_ldsm(8));

        let hmma_idx1 = src.len();
        src.push(make_hmma(24, 8, 12, 255));
        let hmma_idx2 = src.len();
        src.push(make_hmma(28, 8, 13, 255));

        src.push(decoder.decode(0x000fc20003800000000000000000794d).unwrap());

        let entries = Some(HashSet::new());
        let flow = Flow::new(&src, &[], &entries);
        let mut forms = BTreeMap::new();
        forms.insert(hmma_idx0, product_form(&src[hmma_idx0]));
        forms.insert(hmma_idx1, product_form(&src[hmma_idx1]));
        let reach = flow.reaching_for_forms(&forms, &[8]);

        let defs0 = reach[&hmma_idx0][0].as_ref().expect("R8 defs at first HMMA");
        assert!(!defs0.has_none);
        assert_eq!(defs0.locs.len(), 66);
        assert_eq!(defs0.locs[0], 0);
        assert_eq!(*defs0.locs.last().unwrap(), 65);

        let defs1 = reach[&hmma_idx1][0].as_ref().expect("R8 defs after LDSM");
        assert!(!defs1.has_none);
        assert_eq!(defs1.locs, vec![ldsm_idx]);

        let p = plan(&src, &[], &entries).expect("valid pairing plan");
        assert!(!p.sites.contains_key(&hmma_idx0));
        assert!(p.sites.contains_key(&hmma_idx1));
        assert!(p.sites.contains_key(&hmma_idx2));
        assert!(p.loads.contains_key(&ldsm_idx));
    }
}
