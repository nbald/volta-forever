//! Borrowed registers: registers of the program that hold nothing where the translation
//! needs them. Liveness analysis of registers, homes for uniform registers, the pool at
//! each instruction, and victims for room-making at level 3.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::decode::LETTER_U;
use crate::fast::{FastSet, regset_sub, regset_union};
use crate::pairing::Flow;

#[cfg(test)]
use super::context::entries;
use super::parse::FuncData;
use super::{Refused, RegSet, regset_andnot, regset_empty, regset_or, regset_set, regset_test, unannounced};

/// Precomputed per-function liveness and stack-depth facts shared across borrow-level attempts.
pub(super) struct BorrowedPrepared {
    busy: Arc<Vec<RegSet>>,
    touch: Arc<Vec<RegSet>>,
    never: RegSet,
    depth: Arc<Vec<Option<i32>>>,
    r1_fixed: bool,
    units: Vec<(u32, u32, Vec<usize>)>,
}

impl BorrowedPrepared {
    pub(super) fn new(fd: &FuncData, flow: &Flow) -> Result<Self, Refused> {
        if !flow.known {
            return Err(Refused("the ways through the function are not known".into()));
        }
        let n = fd.src.len();
        let mut reads_r = flow.instr_reads_r.clone();
        let mut writes_r = flow.instr_writes_r.clone();
        let mut reads_u: Vec<u64> = vec![0u64; n];
        let mut writes_u: Vec<u64> = vec![0u64; n];
        let mut every_r: RegSet = regset_empty();
        let mut every_u: u64 = 0;

        for (i, d) in fd.src.iter().enumerate() {
            for &(letter, x, _) in &d.reads {
                if letter == LETTER_U && x != 63 {
                    reads_u[i] |= 1u64 << x;
                }
            }
            for &(letter, x) in &d.writes {
                if letter == LETTER_U && x != 63 {
                    writes_u[i] |= 1u64 << x;
                }
            }
            regset_union(&mut every_r, &reads_r[i]);
            regset_union(&mut every_r, &writes_r[i]);
            every_u |= reads_u[i] | writes_u[i];
        }

        let mut kills_r = writes_r.clone();
        let mut kills_u = writes_u.clone();
        let mut leaves_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut leaves_u: Vec<u64> = vec![0u64; n];

        let mut body_seen = FastSet::default();
        let mut work: Vec<usize> = Vec::new();
        for i in 0..n {
            if flow.kind[i] == "call" {
                match flow.called.get(&i) {
                    Some(None) | None => {
                        regset_union(&mut reads_r[i], &every_r);
                        reads_u[i] |= every_u;
                    }
                    Some(Some((alive, ws))) => {
                        regset_union(&mut reads_r[i], alive);
                        regset_union(&mut writes_r[i], ws);
                        if let Some(entry) = flow.named[i] {
                            body_seen.clear();
                            work.clear();
                            work.push(entry);
                            while let Some(k) = work.pop() {
                                if body_seen.insert(k) {
                                    work.extend_from_slice(&flow.succ[k]);
                                }
                            }
                            for &k in &body_seen {
                                for &(letter, x, _) in &fd.src[k].reads {
                                    if letter == LETTER_U && x != 63 {
                                        reads_u[i] |= 1u64 << x;
                                    }
                                }
                                for &(letter, x) in &fd.src[k].writes {
                                    if letter == LETTER_U && x != 63 {
                                        writes_u[i] |= 1u64 << x;
                                    }
                                }
                            }
                        }
                    }
                }
                kills_r[i] = regset_empty();
                kills_u[i] = 0;
            } else if !flow.unguarded[i] {
                kills_r[i] = regset_empty();
                kills_u[i] = 0;
            }
            if fd.src[i].mnemonic == "RET" {
                leaves_r[i] = every_r;
                leaves_u[i] = every_u;
            }
        }

        // Summarize each basic block's backward transfer function (`use`, `kill`, `leave`) and run
        // backward liveness over `nb` blocks only!
        let nb = flow.blocks.len();
        let mut bb_use_r = vec![regset_empty(); nb];
        let mut bb_kill_r = vec![regset_empty(); nb];
        let mut bb_leave_r = vec![regset_empty(); nb];
        let mut bb_use_u = vec![0u64; nb];
        let mut bb_kill_u = vec![0u64; nb];
        let mut bb_leave_u = vec![0u64; nb];

        for (b, &(start_b, end_b)) in flow.blocks.iter().enumerate() {
            let mut use_r = regset_empty();
            let mut use_u = 0u64;
            let mut kill_r = regset_empty();
            let mut kill_u = 0u64;
            for i in (start_b..end_b).rev() {
                if i == end_b - 1 {
                    bb_leave_r[b] = regset_andnot(&leaves_r[i], &kills_r[i]);
                    bb_leave_u[b] = leaves_u[i] & !kills_u[i];
                    use_r = regset_or(&reads_r[i], &bb_leave_r[b]);
                    use_u = reads_u[i] | bb_leave_u[b];
                    kill_r = kills_r[i];
                    kill_u = kills_u[i];
                } else {
                    regset_sub(&mut use_r, &kills_r[i]);
                    regset_union(&mut use_r, &reads_r[i]);
                    use_u = reads_u[i] | (use_u & !kills_u[i]);
                    regset_union(&mut kill_r, &kills_r[i]);
                    kill_u |= kills_u[i];
                    if leaves_r[i] != [0; 4] || leaves_u[i] != 0 {
                        regset_union(&mut use_r, &regset_andnot(&leaves_r[i], &kills_r[i]));
                        use_u |= leaves_u[i] & !kills_u[i];
                    }
                }
            }
            bb_use_r[b] = use_r;
            bb_kill_r[b] = kill_r;
            bb_use_u[b] = use_u;
            bb_kill_u[b] = kill_u;
        }

        let mut bb_live_in_r = vec![regset_empty(); nb];
        let mut bb_live_in_u = vec![0u64; nb];
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..nb).rev() {
                let mut out_r = regset_empty();
                let mut out_u = 0u64;
                for &sb in &flow.bb_succ[b] {
                    regset_union(&mut out_r, &bb_live_in_r[sb]);
                    out_u |= bb_live_in_u[sb];
                }
                let mut new_r = regset_andnot(&out_r, &bb_kill_r[b]);
                regset_union(&mut new_r, &bb_use_r[b]);
                let new_u = bb_use_u[b] | (out_u & !bb_kill_u[b]);
                if new_r != bb_live_in_r[b] || new_u != bb_live_in_u[b] {
                    bb_live_in_r[b] = new_r;
                    bb_live_in_u[b] = new_u;
                    changed = true;
                }
            }
        }

        let mut busy: Vec<RegSet> = vec![regset_empty(); n];
        let mut busy_u: Vec<u64> = vec![0u64; n];
        for (b, &(start_b, end_b)) in flow.blocks.iter().enumerate() {
            let mut live_after_r = regset_empty();
            let mut live_after_u = 0u64;
            for &sb in &flow.bb_succ[b] {
                regset_union(&mut live_after_r, &bb_live_in_r[sb]);
                live_after_u |= bb_live_in_u[sb];
            }
            for i in (start_b..end_b).rev() {
                let mut out_r = live_after_r;
                regset_union(&mut out_r, &leaves_r[i]);
                let out_u = live_after_u | leaves_u[i];
                let live_in_i_r = regset_or(&reads_r[i], &regset_andnot(&out_r, &kills_r[i]));
                let live_in_i_u = reads_u[i] | (out_u & !kills_u[i]);
                let mut bi = live_in_i_r;
                regset_union(&mut bi, &out_r);
                regset_union(&mut bi, &reads_r[i]);
                regset_union(&mut bi, &writes_r[i]);
                busy[i] = bi;
                busy_u[i] = live_in_i_u | out_u | reads_u[i] | writes_u[i];
                live_after_r = live_in_i_r;
                live_after_u = live_in_i_u;
            }
        }

        // Registers that some instruction writes late without saying when are never borrowed.
        let mut never = regset_empty();
        for (i, d) in fd.src.iter().enumerate() {
            if unannounced(d) {
                regset_union(&mut never, &flow.instr_writes_r[i]);
            }
        }
        let touch: Vec<RegSet> = (0..n).map(|i| regset_or(&reads_r[i], &writes_r[i])).collect();
        for b in &mut busy {
            regset_union(b, &never);
        }

        // Compute R1 stack depth along reachable paths from instruction 0.
        let mut depth: Vec<Option<i32>> = vec![None; n];
        let mut r1_fixed = n > 1;
        if n > 1 {
            let mut work: Vec<(usize, i32)> = flow.succ[0].iter().map(|&t| (t, 0i32)).collect();
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
                if (writes_r[i][0] & 2) != 0 {
                    let d = &fd.src[i];
                    if d.name == "iadd3_imm__RsIR_RIR"
                        && d.op_or(crate::generated::fields::OP_RD, 255) == 1
                        && d.op_or(crate::generated::fields::OP_RA, 255) == 1
                        && d.op_or(crate::generated::fields::OP_RC, 255) == 255
                        && flow.unguarded[i]
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
                for &t in &flow.succ[i] {
                    work.push((t, after));
                }
                if flow.kind[i] == "call"
                    && flow.called.get(&i).is_some_and(|effect| effect.is_some())
                    && let Some(Some(entry)) = flow.named.get(i).copied()
                {
                    work.push((entry, at));
                }
            }
        }

        // Collect all uniform register unit spans in a single pass over `busy_u`.
        let mut pairs_mask = 0u64;
        for &u in &fd.pairs {
            if u < 64 {
                pairs_mask |= 1u64 << u;
            }
        }
        let mut unit_meta: Vec<(u32, u32)> = Vec::new();
        let mut unit_of_u = [u16::MAX; 64];
        let mut relevant_mask = 0u64;
        for &u in &fd.uniform {
            if u > 0 && ((pairs_mask >> (u - 1)) & 1) != 0 {
                continue;
            }
            let width: u32 = if u < 64 && ((pairs_mask >> u) & 1) != 0 { 2 } else { 1 };
            let uidx = unit_meta.len() as u16;
            unit_meta.push((u, width));
            if u < 64 {
                unit_of_u[u as usize] = uidx;
                relevant_mask |= 1u64 << u;
                if width == 2 && u + 1 < 64 {
                    unit_of_u[(u + 1) as usize] = uidx;
                    relevant_mask |= 1u64 << (u + 1);
                }
            }
        }
        let mut unit_spans: Vec<Vec<usize>> = vec![Vec::new(); unit_meta.len()];
        for (i, &bu) in busy_u.iter().enumerate() {
            let mut active = bu & relevant_mask;
            while active != 0 {
                let ureg = active.trailing_zeros() as usize;
                let sp = &mut unit_spans[unit_of_u[ureg] as usize];
                if sp.last() != Some(&i) {
                    sp.push(i);
                }
                active &= active - 1;
            }
        }
        let mut units: Vec<(u32, u32, Vec<usize>)> = unit_meta
            .into_iter()
            .zip(unit_spans)
            .map(|((u, w), sp)| (u, w, sp))
            .collect();
        units.sort_by(|a, b| b.2.len().cmp(&a.2.len()).then(a.0.cmp(&b.0)));

        Ok(Self {
            busy: Arc::new(busy),
            touch: Arc::new(touch),
            never,
            depth: Arc::new(depth),
            r1_fixed,
            units,
        })
    }

    pub(super) fn instantiate(&self, fd: &FuncData, all_in_memory: bool, top_override: Option<u32>) -> Borrowed {
        let n = fd.src.len();
        let top = match top_override {
            Some(t) => t.max(fd.highest),
            None => fd.highest,
        };
        let mut taken: Vec<RegSet> = vec![regset_empty(); n];
        let mut homes: HashMap<u32, u32> = HashMap::new();
        let mut spans: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut used: HashSet<u32> = HashSet::new();

        if !all_in_memory {
            for (u, width, span) in &self.units {
                let width = *width;
                let mut union = regset_empty();
                for &i in span {
                    regset_union(&mut union, &self.busy[i]);
                    regset_union(&mut union, &taken[i]);
                }
                let mut r = 2u32;
                let upper = top + 2 - width;
                let found = loop {
                    if r >= upper {
                        break None;
                    }
                    let all_clear = (0..width).all(|k| !regset_test(&union, r + k));
                    if all_clear {
                        break Some(r);
                    }
                    r += width;
                };
                if let Some(r) = found {
                    for k in 0..width {
                        homes.insert(*u + k, r + k);
                        spans.insert(*u + k, span.clone());
                        used.insert(r + k);
                    }
                    for &i in span {
                        for k in 0..width {
                            regset_set(&mut taken[i], r + k);
                        }
                    }
                }
            }
        }

        Borrowed {
            top,
            highest: fd.highest,
            busy: Arc::clone(&self.busy),
            taken,
            touch: Arc::clone(&self.touch),
            never: self.never,
            homes,
            spans,
            used,
            everything: false,
            slots: HashMap::new(),
            predicate_slots: HashMap::new(),
            frame_bytes: 0,
            depth: Arc::clone(&self.depth),
            r1_fixed: self.r1_fixed,
            room: false,
            from_r1: false,
            spill: 0,
        }
    }
}

/// Registers of the program that hold nothing where the translation needs registers: a home
/// for each uniform register through its whole life, and an even register and the next at an
/// instruction whose sequence computes into two (the pool). `docs/translator.md`, "Registers".
///
/// A register is free at an instruction when it is dead before it, dead after it, and neither
/// read nor written by it (an instruction may write a register that nothing reads).
/// The lives follow the ways of `pairing::Flow`; a write
/// under a predicate ends no life; a call within the section reads and writes what the code
/// it calls does; a call elsewhere reads every register the function names; at a return every
/// one of them is alive. R0 and R1 are never taken (R1 is the stack).
pub(super) struct Borrowed {
    /// The highest register the program uses.
    pub(super) top: u32,
    /// The function's own highest register (before any budget override).
    pub(super) highest: u32,
    /// At each instruction: a bit set of registers that are alive, read or written there.
    pub(super) busy: Arc<Vec<RegSet>>,
    /// At each instruction: a bit set of registers taken by homes of uniform registers.
    pub(super) taken: Vec<RegSet>,
    /// At each instruction: a bit set of registers read or written by the instruction itself
    /// (for `victims_at`: registers the instruction does not touch can be put in the frame).
    pub(super) touch: Arc<Vec<RegSet>>,
    /// A bit set of registers that some instruction writes late without saying when
    /// (never borrowed as victims either).
    pub(super) never: RegSet,
    /// {uniform register: ordinary register it lives in through its whole life}.
    pub(super) homes: HashMap<u32, u32>,
    /// {uniform register with a home: the instructions of its life}.
    pub(super) spans: HashMap<u32, Vec<usize>>,
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
    /// known at that instruction. The program moves R1 by `IADD3 R1, R1, imm` only.
    pub(super) depth: Arc<Vec<Option<i32>>>,
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

impl Borrowed {
    #[cfg(test)]
    pub(super) fn new(fd: &FuncData, all_in_memory: bool, top_override: Option<u32>) -> Result<Self, Refused> {
        let ent = entries(fd);
        let flow = Flow::new_with_facts(&fd.src, &fd.tables, &ent, false);
        let prep = BorrowedPrepared::new(fd, &flow)?;
        Ok(prep.instantiate(fd, all_in_memory, top_override))
    }

    /// The lowest register that holds nothing at instruction `i`, or the lowest run of `width`
    /// registers from an even one: no home of a uniform register alive there, and not among
    /// `given` (a bit set: what the same instruction was given already). None when there is none.
    pub(super) fn free_at(&self, i: usize, width: u32, given: &RegSet) -> Option<u32> {
        let held = regset_or(&regset_or(&self.busy[i], &self.taken[i]), given);
        let upper = self.top + 2 - width;
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
            regset_union(&mut union, &self.busy[i]);
            regset_union(&mut union, &self.taken[i]);
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

    /// A word of the frame for each uniform register that has no home (the two of a pair
    /// side by side, the lower one first). Returns the size of the frame in bytes.
    pub(super) fn assign_slots(&mut self, uniform: &[u32], pairs: &HashSet<u32>) -> u32 {
        let mut slot = 0u32;
        for &u in uniform {
            if self.homes.contains_key(&u) || (u > 0 && pairs.contains(&(u - 1))) {
                continue;
            }
            if pairs.contains(&u) {
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
