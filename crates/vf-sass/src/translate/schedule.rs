//! Scheduling: set the stall of each item so that every instruction waits long enough
//! for the registers it reads and the scoreboards it depends on. The schedule handles
//! what the translation added (saves, restores, waits for borrowed registers) in the
//! same pass as the program's own instructions.

use std::sync::Arc;

use crate::decode::{Decoded, Kind, LETTER_P, LETTER_R};
use crate::generated::fields::*;
use crate::generated::special::*;

use super::{Item, Refused, marked, stall_of, with_stall};

// ---- Scheduling ----

const REG_SLOTS: usize = 264; // 0..256: R0..255, 256..264: P0..7

#[inline]
fn slot_of(letter: u8, reg: u32) -> Option<usize> {
    if letter == LETTER_R && reg < 256 {
        Some(reg as usize)
    } else if letter == LETTER_P && reg < 8 {
        Some(256 + reg as usize)
    } else {
        None
    }
}

#[inline]
fn slot_letter(slot: usize) -> u8 {
    if slot < 256 { LETTER_R } else { LETTER_P }
}

pub(super) fn schedule(items: &mut Vec<Item>, targets: &[u64]) -> Result<(), Refused> {
    let max_origin = items.iter().map(|it| it.origin).max().unwrap_or(0);
    let mut is_target = vec![false; max_origin + 1];
    for &t in targets {
        if t.is_multiple_of(16) {
            let idx = (t / 16) as usize;
            if idx < is_target.len() {
                is_target[idx] = true;
            }
        }
    }

    // Split into basic blocks
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for (k, item) in items.iter().enumerate() {
        if k > start && item.first && is_target[item.origin] {
            blocks.push((start, k));
            start = k;
        }
        if item.d.as_ref().unwrap().kind == Kind::BRU {
            blocks.push((start, k + 1));
            start = k + 1;
        }
    }
    if start < items.len() {
        blocks.push((start, items.len()));
    }

    let mut inserted: Vec<(usize, u128)> = Vec::new();
    let mut t: i32 = 0;

    type ExtraLastEntry = ((u8, u32), (i32, usize, bool));
    type ExtraReadEntry = ((u8, u32), (i32, Kind));

    // When each register or predicate was last written, by which item, and whether through the
    // mask of an R2P of the program (`masked`).
    let mut last_gen = [0u32; REG_SLOTS];
    let mut last_val = [(0i32, 0usize, false); REG_SLOTS];
    let mut last_epoch: u32 = 1;
    let mut active_last: Vec<u16> = Vec::with_capacity(64);
    let mut extra_last: Vec<ExtraLastEntry> = Vec::new();

    // The writers whose scoreboard an instruction since has waited for.
    let mut waited_gen = vec![0u32; items.len()];
    let mut waited_epoch: u32 = 1;

    // The time and kind of the last reader of each register, reset per block.
    let mut read_gen = [0u32; REG_SLOTS];
    let mut read_val = [(0i32, Kind::OTHER); REG_SLOTS];
    let mut read_epoch: u32 = 1;
    let mut extra_read: Vec<ExtraReadEntry> = Vec::new();

    let mut reads_buf: Vec<(u8, u32, u8)> = Vec::new();
    let mut writes_buf: Vec<(u8, u32)> = Vec::new();
    let mut through_buf: Vec<(u8, u32, u8)> = Vec::new();
    let mut made_up_buf: Vec<(u8, u32, u8)> = Vec::new();
    let mut masked_buf: Vec<(u8, u32)> = Vec::new();

    for &(a, b) in &blocks {
        // What was written before this block, and when, is known where the code before it goes
        // on into it (past a label, a branch not taken, the mark of a region): an instruction
        // the translation changed or added here is as far from those writers as from the ones
        // of its own block. Where the code before does not go on (or goes through other code
        // first: a call), nothing is known, as for what jumps here.
        let continued = goes_on(items, a);
        if !continued {
            t = 0;
            last_epoch += 1;
            if last_epoch == 0 {
                last_gen.fill(0);
                last_epoch = 1;
            }
            active_last.clear();
            extra_last.clear();
            waited_epoch += 1;
            if waited_epoch == 0 {
                waited_gen.fill(0);
                waited_epoch = 1;
            }
        }
        read_epoch += 1;
        if read_epoch == 0 {
            read_gen.fill(0);
            read_epoch = 1;
        }
        extra_read.clear();

        for k in a..b {
            let item_changed = items[k].changed;
            let item_origin = items[k].origin;
            let d: Arc<Decoded> = Arc::clone(items[k].d.as_ref().unwrap());
            let has_more_reads = !items[k].more_reads.is_empty();
            let has_more_writes = !items[k].more_writes.is_empty();
            let is_mask_op = d.mnemonic == "P2R" || d.mnemonic == "R2P";

            let reads_slice: &[(u8, u32, u8)] = if has_more_reads {
                reads_buf.clear();
                reads_buf.extend_from_slice(&d.reads);
                reads_buf.extend_from_slice(&items[k].more_reads);
                &reads_buf
            } else {
                &d.reads
            };

            let writes_slice: &[(u8, u32)] = if has_more_writes {
                writes_buf.clear();
                writes_buf.extend_from_slice(&d.writes);
                writes_buf.extend_from_slice(&items[k].more_writes);
                &writes_buf
            } else {
                &d.writes
            };

            let has_made_up = has_more_reads || is_mask_op;
            if is_mask_op {
                fill_all_predicates(&d, reads_slice, &mut through_buf);
            } else {
                through_buf.clear();
            }
            if has_made_up {
                made_up_buf.clear();
                made_up_buf.extend_from_slice(&items[k].more_reads);
                made_up_buf.extend_from_slice(&through_buf);
            } else {
                made_up_buf.clear();
            }

            let mut need: i32 = 0;
            for &(letter, register, role) in d.reads.iter().chain(made_up_buf.iter()) {
                let found = match slot_of(letter, register) {
                    Some(slot) if last_gen[slot] == last_epoch => Some(last_val[slot]),
                    Some(_) => None,
                    None => extra_last
                        .iter()
                        .find(|&&(k_reg, _)| k_reg == (letter, register))
                        .map(|&(_, v)| v),
                };
                if let Some((wt, wi, through)) = found {
                    let wr_changed = items[wi].changed;
                    let wr_word = items[wi].word;
                    if !(wr_changed || item_changed) {
                        continue;
                    }
                    let is_made = has_made_up && made_up_buf.contains(&(letter, register, role));
                    if through && !is_made {
                        // (Written through the mask of an R2P of the program, and read by what
                        // the program's instruction read: as far as the compiler had them.)
                        continue;
                    }
                    if ((wr_word >> WRITE_BARRIER) & 7) != 7 || items[wi].covered {
                        // Announced on a scoreboard: the reader waits for it. What the program's
                        // instruction waited for, its translation waits for; a read that the
                        // translation made up (a predicate saved, written down, or all of them
                        // for R2P) has nothing of the program that waits for it: it does here,
                        // unless an instruction since has waited for that scoreboard (`waited`).
                        if is_made && ((wr_word >> WRITE_BARRIER) & 7) != 7 && waited_gen[wi] != waited_epoch {
                            if !item_changed {
                                return Err(Refused(format!(
                                    "the {} at 0x{:x} reads a predicate of the translation that a scoreboard announces",
                                    d.mnemonic,
                                    16 * item_origin
                                )));
                            }
                            items[k].word |= 1u128 << (WAIT + ((wr_word >> WRITE_BARRIER) & 7) as u32);
                        }
                        // (A result of double precision is not read earlier than Volta's own
                        // code reads one, whatever is waited for: `distance::waited`.)
                        let wd = items[wi].d.as_ref().unwrap();
                        if let Some(floor) = crate::distance::waited(letter, role, wd, d.kind) {
                            need = need.max(wt + floor as i32 - t);
                        }
                        continue;
                    }
                    let wd = items[wi].d.as_ref().unwrap();
                    let dist = crate::distance::least(letter, role, wd, d.kind) as i32;
                    need = need.max(wt + dist - t);
                }
            }
            // A writer that is not plain arithmetic, after a reader of what it writes: the
            // compiler's tables ask for up to 9 cycles between them (2 in most cases); the
            // original had another kind of writer here, with its own distance. (Not a
            // product after a product: they are done in order, and the compiler has the
            // steps of one product two cycles apart, the third writing what the first read.)
            // R2P is kept to this too, for another reason than the one first given for it
            // (2026-11-18, the same day: a reader does NOT go by what an R2P issued after it
            // gives back, at any distance): the instruction before a restore reads
            // what stood in the predicate, so the restore is nine cycles into its block or
            // more, and no writer of a predicate that ran before the block, on a path the
            // schedule does not see, is close enough for the R2P to undo (`all_predicates`).
            // What R2P writes is in a mask: `more_writes`.
            if item_changed && (!matches!(d.kind, Kind::FXU | Kind::IMAD | Kind::FMAI) || d.mnemonic == "R2P") {
                for &(letter, register) in writes_slice {
                    let found = match slot_of(letter, register) {
                        Some(slot) if read_gen[slot] == read_epoch => Some(read_val[slot]),
                        Some(_) => None,
                        None => extra_read
                            .iter()
                            .find(|&&(k_reg, _)| k_reg == (letter, register))
                            .map(|&(_, v)| v),
                    };
                    if let Some((rt, reader_kind)) = found
                        && !(d.kind == Kind::HMMA && reader_kind == Kind::HMMA)
                    {
                        need = need.max(rt + 9 - t);
                    }
                }
            }
            if need > 0 {
                if k == 0 || (k == a && !continued) {
                    return Err(Refused(format!(
                        "scheduling: nothing before 0x{:x} to wait on",
                        16 * item_origin
                    )));
                }
                t += delay(items, k - 1, need as u32, &mut inserted) as i32;
            }
            // (As a reader for the nine cycles above: a P2R with the predicates of its mask; not
            // an R2P with those it only writes back.)
            let record_read = |letter: u8,
                               register: u32,
                               read_gen: &mut [u32; REG_SLOTS],
                               read_val: &mut [(i32, Kind); REG_SLOTS],
                               extra_read: &mut Vec<ExtraReadEntry>| {
                if let Some(slot) = slot_of(letter, register) {
                    read_gen[slot] = read_epoch;
                    read_val[slot] = (t, d.kind);
                } else if let Some(entry) = extra_read.iter_mut().find(|(k_reg, _)| *k_reg == (letter, register)) {
                    entry.1 = (t, d.kind);
                } else {
                    extra_read.push(((letter, register), (t, d.kind)));
                }
            };
            for &(letter, register, _) in reads_slice {
                record_read(letter, register, &mut read_gen, &mut read_val, &mut extra_read);
            }
            if d.mnemonic == "P2R" {
                for &(letter, register, _) in &through_buf {
                    record_read(letter, register, &mut read_gen, &mut read_val, &mut extra_read);
                }
            }
            // (What this instruction waits for is there when it is issued: the writers that
            // announced on those scoreboards are waited for.)
            let boards = (items[k].word >> WAIT) & 63;
            if boards != 0 {
                for &slot in &active_last {
                    let wi = last_val[slot as usize].1;
                    if (boards >> ((items[wi].word >> WRITE_BARRIER) & 7)) & 1 != 0 {
                        waited_gen[wi] = waited_epoch;
                    }
                }
                for &(_, (_, wi, _)) in &extra_last {
                    if (boards >> ((items[wi].word >> WRITE_BARRIER) & 7)) & 1 != 0 {
                        waited_gen[wi] = waited_epoch;
                    }
                }
            }
            let record_write = |letter: u8,
                                register: u32,
                                val: (i32, usize, bool),
                                last_gen: &mut [u32; REG_SLOTS],
                                last_val: &mut [(i32, usize, bool); REG_SLOTS],
                                active_last: &mut Vec<u16>,
                                extra_last: &mut Vec<ExtraLastEntry>| {
                if let Some(slot) = slot_of(letter, register) {
                    if last_gen[slot] != last_epoch {
                        last_gen[slot] = last_epoch;
                        active_last.push(slot as u16);
                    }
                    last_val[slot] = val;
                } else if let Some(entry) = extra_last.iter_mut().find(|(k_reg, _)| *k_reg == (letter, register)) {
                    entry.1 = val;
                } else {
                    extra_last.push(((letter, register), val));
                }
            };
            for &(letter, register) in writes_slice {
                record_write(
                    letter,
                    register,
                    (t, k, false),
                    &mut last_gen,
                    &mut last_val,
                    &mut active_last,
                    &mut extra_last,
                );
            }
            if d.mnemonic == "R2P" {
                fill_masked(&d, writes_slice, &mut masked_buf);
                for &(letter, register) in &masked_buf {
                    record_write(
                        letter,
                        register,
                        (t, k, true),
                        &mut last_gen,
                        &mut last_val,
                        &mut active_last,
                        &mut extra_last,
                    );
                }
            }
            t += stall_of(items[k].word) as i32;
        }
        // Tail: ensure enough distance for the last writer
        let mut tail_idx = if items[b - 1].d.as_ref().unwrap().kind != Kind::BRU || b - a == 1 {
            b - 1
        } else {
            b - 2
        };
        let mut need: i32 = 0;
        let check_tail = |letter: u8, when: i32, wi: usize, need: &mut i32| {
            let wr_changed = items[wi].changed;
            let wr_word = items[wi].word;
            if wr_changed && ((wr_word >> WRITE_BARRIER) & 7) == 7 && !items[wi].covered && wi != b - 1 {
                let wd = items[wi].d.as_ref().unwrap();
                *need = (*need).max(when + crate::distance::longest(letter, wd) as i32 - t);
            }
            // (A result of double precision of code for sm_80, scoreboard or not: its reader
            // may be the first instruction where a branch goes, which waits and is still too
            // early. Code for sm_75 keeps Volta's distances along its branches too: none shorter
            // in the 475 cubins of PyTorch, and the translation shortens none.)
            if items[wi].anew
                && wi != b - 1
                && let Some(floor) = crate::distance::waited_longest(letter, items[wi].d.as_ref().unwrap())
            {
                *need = (*need).max(when + floor as i32 - t);
            }
        };
        for &slot in &active_last {
            let (when, wi, _) = last_val[slot as usize];
            check_tail(slot_letter(slot as usize), when, wi, &mut need);
        }
        for &((letter, _), (when, wi, _)) in &extra_last {
            check_tail(letter, when, wi, &mut need);
        }
        let wd: Arc<Decoded> = Arc::clone(items[b - 1].d.as_ref().unwrap());
        let has_tail_more = !items[b - 1].more_writes.is_empty();
        let tail_writes: &[(u8, u32)] = if has_tail_more {
            writes_buf.clear();
            writes_buf.extend_from_slice(&wd.writes);
            writes_buf.extend_from_slice(&items[b - 1].more_writes);
            &writes_buf
        } else {
            &wd.writes
        };
        if items[b - 1].changed && !tail_writes.is_empty() && ((items[b - 1].word >> WRITE_BARRIER) & 7) == 7 {
            tail_idx = b - 1;
            for &(l, _) in tail_writes {
                need = need.max(crate::distance::longest(l, &wd) as i32 - stall_of(items[b - 1].word) as i32);
            }
        }
        if items[b - 1].anew {
            for &(l, _) in &wd.writes {
                if let Some(floor) = crate::distance::waited_longest(l, &wd) {
                    tail_idx = b - 1;
                    need = need.max(floor as i32 - stall_of(items[b - 1].word) as i32);
                }
            }
        }
        if need > 0 {
            t += delay(items, tail_idx, need as u32, &mut inserted) as i32;
        }
    }

    // Insert NOP fillers in O(N) while preserving ascending `(k, word)` insertion order.
    if !inserted.is_empty() {
        inserted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut rebuilt = Vec::with_capacity(items.len() + inserted.len());
        let mut ins_idx = 0usize;
        for (k, item) in items.drain(..).enumerate() {
            let origin = item.origin;
            rebuilt.push(item);
            while ins_idx < inserted.len() && inserted[ins_idx].0 == k {
                rebuilt.push(Item::new(inserted[ins_idx].1, origin, "wait", true));
                ins_idx += 1;
            }
        }
        *items = rebuilt;
    }

    // A wait for a scoreboard comes two cycles or more after the instruction that names it, for
    // its result or for having read: native code of both architectures never has less. The
    // translation adds waits and names.
    for k in 1..items.len() {
        let mask = (items[k].word >> WAIT) & 63;
        if mask == 0 {
            continue;
        }
        let mut distance = 0;
        for j in (0..k).rev() {
            distance += stall_of(items[j].word);
            if distance >= 2 {
                break;
            }
            let named = [
                (items[j].word >> WRITE_BARRIER) & 7,
                (items[j].word >> READ_BARRIER) & 7,
            ];
            if (items[j].changed || items[k].changed || items[j].says)
                && named.iter().any(|&s| s != 7 && (mask >> s) & 1 != 0)
            {
                let before = items[k - 1].word;
                items[k - 1].word = with_stall(before, stall_of(before) + 2 - distance, marked(&items[k - 1].d));
                break;
            }
        }
    }
    Ok(())
}

/// Whether the item before item `a`, the first of a block of the schedule, goes on into it with
/// nothing else run between: an instruction that is not a branch, a branch that may not be taken,
/// the mark of a region (BSSY, BMOV). Not a call (other code runs first), nor anything that leaves.
fn goes_on(items: &[Item], a: usize) -> bool {
    if a == 0 {
        return false;
    }
    let d = items[a - 1].d.as_ref().unwrap();
    if d.kind != Kind::BRU || matches!(d.mnemonic, "BSSY" | "BMOV") {
        return true;
    }
    let always = d.op_or(OP_PG, 7) == 7
        && d.op_or(OP_PG_AT_NOT, 0) == 0
        && d.op_or(OP_PP, 7) == 7
        && d.op_or(OP_PP_AT_NOT, 0) == 0
        && d.op_or(OP_COND, 0) == 0;
    d.mnemonic == "BRA" && !always
}

fn fill_all_predicates(d: &Decoded, reads: &[(u8, u32, u8)], out: &mut Vec<(u8, u32, u8)>) {
    out.clear();
    let named = |p: u32| reads.iter().any(|&(l, r, _)| l == LETTER_P && r == p);
    let wanted: u32 = match d.mnemonic {
        "P2R" => d.op_or(OP_SB, 0x7f) & 0x7f,
        "R2P" => {
            !(if d.name == "r2p__RIR" {
                d.op_or(OP_SB, 0) & 0x7f
            } else {
                0
            }) & 0x7f
        }
        _ => 0,
    };
    for p in 0..7u32 {
        if (wanted >> p) & 1 != 0 && !named(p) {
            out.push((LETTER_P, p, 0u8));
        }
    }
}

fn fill_masked(d: &Decoded, writes: &[(u8, u32)], out: &mut Vec<(u8, u32)>) {
    out.clear();
    if d.mnemonic != "R2P" {
        return;
    }
    let mask = if d.name == "r2p__RIR" {
        d.op_or(OP_SB, 0x7f) & 0x7f
    } else {
        0x7f
    };
    for p in 0..7u32 {
        if (mask >> p) & 1 != 0 && !writes.contains(&(LETTER_P, p)) {
            out.push((LETTER_P, p));
        }
    }
}

fn delay(items: &mut [Item], k: usize, cycles: u32, inserted: &mut Vec<(usize, u128)>) -> u32 {
    let room = 15 - stall_of(items[k].word);
    let take = room.min(cycles);
    items[k].word = with_stall(items[k].word, stall_of(items[k].word) + take, marked(&items[k].d));
    let mut left = cycles as i32 - take as i32;
    while left > 0 {
        let step = (left as u32).min(15);
        inserted.push((k, with_stall(NOP, step, false)));
        left -= step as i32;
    }
    cycles
}
