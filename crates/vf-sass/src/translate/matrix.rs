//! Turing's matrix instructions: `ldsm` (ldmatrix) and `hmma` (tensor-core product).
//! Volta has neither. What stands for them is what the layer already does in program text
//! (crates/vf-ptx/src/lower.rs says why each sequence is right, and tcrun checks every
//! register of every thread against a model), as the compiler writes it for sm_70.

use std::collections::HashMap;

use crate::assemble;
use crate::decode::{Decoded, LETTER_R};
use crate::field;
use crate::generated::fields::*;
use crate::generated::special::*;
use crate::rule;

use super::context::{Ctx, apply_rule};
use super::util::{late, put_operand_by_class};
use super::{ARRIVES, Item, LATE_WRITER_KINDS, Refused, SEQUENCE, with_stall};

// ---- Turing's matrix instructions -----------------------------------------------------------------------
// Volta has neither. What stands for them is what the layer already does in program text
// (crates/vf-ptx/src/lower.rs says why each sequence is right, and tcrun checks every
// register of every thread against a model), as the compiler writes it for sm_70.
//
// A lane is 4 * group + thread (group 0 to 7, thread 0 to 3). The instructions of a sequence
// that take their time (shuffles, loads, products) say on scoreboard SEQUENCE when their
// result is there, and what reads it waits for that: by the end of a sequence each has
// written, and so has read, and its registers are free for the next one.

/// LDSM (`ldmatrix`): each thread gets one word of each of 1, 2 or 4 matrices of 8 rows of 8
/// halves in shared memory. The address of row r of matrix m is what thread r + 8m has in the
/// address register. Plain: the thread's pair of halves in the row of its group. Transposed:
/// the halves of column `group` in the rows 2 * thread and 2 * thread + 1.
///
/// The address comes by a shuffle from the lane that holds it, into the destination itself
/// (the one destination that is the address register is shuffled last: what is issued before
/// has read by then), and plain loads follow.
///
/// A store before the load may not have read a destination yet: a store reads its registers some
/// time after it is issued, and the compiler counts on the load, a memory instruction like the
/// store, to write them after that (native code writes such a register with arithmetic only after
/// a barrier or a wait for a later memory instruction). So the first instruction of the sequence
/// that writes a destination is a shuffle, and arithmetic writes one only after waiting for the
/// shuffles: the lanes of the rows are computed in registers of the sequence, and in a
/// transposed load those of the first rows go into the destinations after the shuffles of the
/// second rows are there. (Until 2026-11-23 the lanes went into the destinations first, and a
/// store of such a register just before the load could store the lane.)
pub(super) fn translate_matrix_load(
    ctx: &mut Ctx,
    i: usize,
    d: &Decoded,
    guard: &[(&str, assemble::OpVal)],
    wait: u128,
) -> Result<Vec<Item>, Refused> {
    use assemble::OpVal::Int;
    let name = d.name;
    let sz = d.op_or(OP_SZ, 0);
    let mode = d.op_or(OP_MODE, 0);
    let num = d.op_or(OP_NUM, 0);
    if sz != 0 || !matches!(mode, 0..=1) || !matches!(num, 0..=2) {
        return Err(Refused(format!("this form of LDSM at 0x{:x}", 16 * i)));
    }
    let count = 1u32 << num;
    let transposed = mode == 1;
    let rd = d.op_or(OP_RD, 255);
    let ra = d.op_or(OP_RA, 255);
    let offset = d.op_or(OP_RA_OFFSET, 0);
    if rd == 255 || rd + count > 255 {
        return Err(Refused(format!("LDSM into no register at 0x{:x}", 16 * i)));
    }
    let uniform = if name == "ldsm__URsIR" {
        let u = ctx.r(d.op_or(OP_RA_URB, 63))?;
        if u == 255 { None } else { Some(u) }
    } else {
        None
    };
    let lane = ctx.lane()?;
    let s = ctx.scratch()?;
    let (part, group, other, sum_) = (s, s + 1, s + 2, s + 3);
    // s + 4 to s + 7: the second rows of a transposed load
    let result = d.op_or(OP_DST_WR_SB, 7);
    let w = d.word;

    let mut new: Vec<Item> = Vec::new();
    let mut add = |word: u128, what: &'static str, waits: bool| {
        let wait_bits = if waits { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        new.push(Item::new(word | wait | wait_bits, i, what, true));
    };
    let asm = assemble::assemble;
    let shf = |rd_: u32, ra_: u32, sb_: u32, rc_: u32, dir_: &str, hilo_: &str, fmt_: &str| {
        asm(
            "shf__RuIR_RIR",
            &[
                ("Rd", Int(rd_)),
                ("Ra", Int(ra_)),
                ("Sb", Int(sb_)),
                ("Rc", Int(rc_)),
                ("dir", assemble::OpVal::Str(dir_)),
                ("hilo", assemble::OpVal::Str(hilo_)),
                ("fmt", assemble::OpVal::Str(fmt_)),
            ],
        )
    };
    let lut = |rd_: u32, ra_: u32, sb_: u32, rc_: u32, imm8_: u32| {
        asm(
            "lop3_lut__RuIR_RIR",
            &[
                ("Pu", Int(7)),
                ("Pp", Int(7)),
                ("Pp_not", Int(1)),
                ("Rd", Int(rd_)),
                ("Ra", Int(ra_)),
                ("Sb", Int(sb_)),
                ("Rc", Int(rc_)),
                ("imm8", Int(imm8_)),
            ],
        )
    };

    add(
        shf(group, 255, 2, lane, "R", "HI", "U32").ok_or_else(|| Refused("cannot assemble SHF for LDSM".into()))?,
        "group",
        false,
    );

    // The one destination that is the address register comes last, and has its lane in a scratch register.
    let mut order: Vec<u32> = (0..count).filter(|&m| rd + m != ra).collect();
    order.extend((0..count).filter(|&m| rd + m == ra));
    let guarded = |r: u32| -> Vec<(&str, assemble::OpVal)> {
        if rd <= r && r < rd + count {
            guard.to_vec()
        } else {
            Vec::new()
        }
    };

    if !transposed {
        add(
            lut(part, lane, 3, 255, 0xc0).ok_or_else(|| Refused("cannot assemble LOP3 for LDSM".into()))?,
            "thread",
            false,
        );
        // The lane of the row: group + 8m, in a register of the sequence.
        let mut lanes: HashMap<u32, u32> = HashMap::new();
        for &m in &order {
            if m == 0 {
                lanes.insert(m, group);
            } else {
                let dest = s + 4 + m;
                lanes.insert(m, dest);
                let ops = vec![
                    ("Rd", Int(dest)),
                    ("Ra", Int(group)),
                    ("Sb", Int(8 * m)),
                    ("Rc", Int(255)),
                ];
                add(
                    asm("iadd3_imm__RsIR_RIR", &ops).ok_or_else(|| Refused("cannot assemble IADD3 for LDSM".into()))?,
                    "row",
                    false,
                );
            }
        }
        for &m in &order {
            let mut ops = vec![
                ("shflmd", Int(0)),
                ("Pu", Int(7)),
                ("Rd", Int(rd + m)),
                ("Ra", Int(ra)),
                ("Rb", Int(lanes[&m])),
                ("Sc", Int(31)),
                ("dst_wr_sb", Int(SEQUENCE)),
            ];
            ops.extend(guard.iter().map(|&(k, ref v)| (k, v.clone())));
            add(
                asm("shfl__RRI", &ops).ok_or_else(|| Refused("cannot assemble SHFL for LDSM".into()))?,
                "address of the row",
                false,
            );
        }
        // The pair of the thread in its row: 4 * thread further (and the uniform part of the address).
        if let Some(uniform_r) = uniform {
            add(
                asm(
                    "lea_lo_noimm__RRR_RRR",
                    &[
                        ("Rd", Int(sum_)),
                        ("Ra", Int(part)),
                        ("Rb", Int(uniform_r)),
                        ("scaleU5", Int(2)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble LEA for LDSM".into()))?,
                "place in the row",
                false,
            );
        }
        for (k, &m) in order.iter().enumerate() {
            let mut ops;
            if uniform.is_none() {
                ops = vec![
                    ("Rd", Int(rd + m)),
                    ("Ra", Int(part)),
                    ("Rb", Int(rd + m)),
                    ("scaleU5", Int(2)),
                ];
                ops.extend(guarded(rd + m).iter().map(|&(k, ref v)| (k, v.clone())));
                add(
                    asm("lea_lo_noimm__RRR_RRR", &ops).ok_or_else(|| Refused("cannot assemble LEA for LDSM".into()))?,
                    "address",
                    k == 0,
                );
            } else {
                ops = vec![
                    ("Rd", Int(rd + m)),
                    ("Ra", Int(rd + m)),
                    ("Rb", Int(sum_)),
                    ("Rc", Int(255)),
                ];
                ops.extend(guarded(rd + m).iter().map(|&(k, ref v)| (k, v.clone())));
                add(
                    asm("iadd3_noimm__RRR_RRR", &ops)
                        .ok_or_else(|| Refused("cannot assemble IADD3 for LDSM".into()))?,
                    "address",
                    k == 0,
                );
            }
        }
        for &m in &order {
            let mut ops = vec![
                ("sz", assemble::OpVal::Str("U.32")),
                ("Rd", Int(rd + m)),
                ("Ra", Int(rd + m)),
                ("Ra_offset", Int(offset)),
                ("dst_wr_sb", Int(if result == 7 { SEQUENCE } else { result })),
            ];
            ops.extend(guarded(rd + m).iter().map(|&(k, ref v)| (k, v.clone())));
            add(
                asm("lds__sImmOffset", &ops).ok_or_else(|| Refused("cannot assemble LDS for LDSM".into()))?,
                "word of the matrix",
                false,
            );
        }
        if result == 7 {
            new.last_mut().unwrap().then = Some(SEQUENCE);
        }
    } else {
        // Transposed: twice the thread: the first of its two rows.
        add(
            shf(part, lane, 1, 255, "L", "LO", "U32").ok_or_else(|| Refused("cannot assemble SHF for LDSM".into()))?,
            "thread",
            false,
        );
        add(
            lut(part, part, 6, 255, 0xc0).ok_or_else(|| Refused("cannot assemble LOP3 for LDSM".into()))?,
            "thread",
            false,
        );
        let mut first: HashMap<u32, u32> = HashMap::new();
        let mut second: HashMap<u32, u32> = HashMap::new();
        for &m in &order {
            second.insert(m, s + 4 + m);
            add(
                asm(
                    "iadd3_imm__RsIR_RIR",
                    &[
                        ("Rd", Int(s + 4 + m)),
                        ("Ra", Int(part)),
                        ("Sb", Int(8 * m + 1)),
                        ("Rc", Int(255)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble IADD3 for LDSM".into()))?,
                "row",
                false,
            );
        }
        for &m in &order {
            add(
                asm(
                    "shfl__RRI",
                    &[
                        ("shflmd", Int(0)),
                        ("Pu", Int(7)),
                        ("Rd", Int(second[&m])),
                        ("Ra", Int(ra)),
                        ("Rb", Int(second[&m])),
                        ("Sc", Int(31)),
                        ("dst_wr_sb", Int(SEQUENCE)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble SHFL for LDSM".into()))?,
                "address of the row",
                false,
            );
        }
        // The lanes of the first rows, in the destinations once the shuffles above are there.
        let mut waits = true;
        for &m in &order {
            if m == 0 {
                first.insert(m, part);
            } else {
                let dest = if rd + m == ra { other } else { rd + m };
                first.insert(m, dest);
                let mut ops = vec![
                    ("Rd", Int(dest)),
                    ("Ra", Int(part)),
                    ("Sb", Int(8 * m)),
                    ("Rc", Int(255)),
                ];
                ops.extend(guarded(dest).iter().map(|&(k, ref v)| (k, v.clone())));
                add(
                    asm("iadd3_imm__RsIR_RIR", &ops).ok_or_else(|| Refused("cannot assemble IADD3 for LDSM".into()))?,
                    "row",
                    waits,
                );
                waits = false;
            }
        }
        for &m in &order {
            let mut ops = vec![
                ("shflmd", Int(0)),
                ("Pu", Int(7)),
                ("Rd", Int(rd + m)),
                ("Ra", Int(ra)),
                ("Rb", Int(first[&m])),
                ("Sc", Int(31)),
                ("dst_wr_sb", Int(SEQUENCE)),
            ];
            ops.extend(guard.iter().map(|&(k, ref v)| (k, v.clone())));
            add(
                asm("shfl__RRI", &ops).ok_or_else(|| Refused("cannot assemble SHFL for LDSM".into()))?,
                "address of the row",
                false,
            );
        }
        // The half of the thread's column in each row: 2 * group further (and the uniform part).
        if let Some(uniform_r) = uniform {
            add(
                asm(
                    "lea_lo_noimm__RRR_RRR",
                    &[
                        ("Rd", Int(sum_)),
                        ("Ra", Int(group)),
                        ("Rb", Int(uniform_r)),
                        ("scaleU5", Int(1)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble LEA for LDSM".into()))?,
                "place in the row",
                false,
            );
        }
        for (k, &m) in order.iter().enumerate() {
            for &(r, is_guard) in &[(second[&m], false), (rd + m, true)] {
                let g = if is_guard { guard.to_vec() } else { Vec::new() };
                if uniform.is_none() {
                    let mut ops = vec![("Rd", Int(r)), ("Ra", Int(group)), ("Rb", Int(r)), ("scaleU5", Int(1))];
                    ops.extend(g.iter().map(|&(k, ref v)| (k, v.clone())));
                    add(
                        asm("lea_lo_noimm__RRR_RRR", &ops)
                            .ok_or_else(|| Refused("cannot assemble LEA for LDSM".into()))?,
                        "address",
                        k == 0 && r == second[&m],
                    );
                } else {
                    let mut ops = vec![("Rd", Int(r)), ("Ra", Int(r)), ("Rb", Int(sum_)), ("Rc", Int(255))];
                    ops.extend(g.iter().map(|&(k, ref v)| (k, v.clone())));
                    add(
                        asm("iadd3_noimm__RRR_RRR", &ops)
                            .ok_or_else(|| Refused("cannot assemble IADD3 for LDSM".into()))?,
                        "address",
                        k == 0 && r == second[&m],
                    );
                }
            }
        }
        for &m in &order {
            // Under the guard too, though it writes a register of the translation: it reads
            // memory, which the instruction does not when its guard is false. (Until 2026-11-17
            // it had none.)
            let mut ops = vec![
                ("sz", assemble::OpVal::Str("U.U16")),
                ("Rd", Int(second[&m])),
                ("Ra", Int(second[&m])),
                ("Ra_offset", Int(offset)),
                ("dst_wr_sb", Int(SEQUENCE)),
            ];
            ops.extend(guard.iter().map(|&(k, ref v)| (k, v.clone())));
            add(
                asm("lds__sImmOffset", &ops).ok_or_else(|| Refused("cannot assemble LDS for LDSM".into()))?,
                "half of the matrix",
                false,
            );
            let mut ops = vec![
                ("sz", assemble::OpVal::Str("U.U16")),
                ("Rd", Int(rd + m)),
                ("Ra", Int(rd + m)),
                ("Ra_offset", Int(offset)),
                ("dst_wr_sb", Int(SEQUENCE)),
            ];
            ops.extend(guarded(rd + m).iter().map(|&(k, ref v)| (k, v.clone())));
            add(
                asm("lds__sImmOffset", &ops).ok_or_else(|| Refused("cannot assemble LDS for LDSM".into()))?,
                "half of the matrix",
                false,
            );
        }
        // The two halves in one word: the one of the second row above the other.
        for (k, &m) in order.iter().enumerate() {
            let mut ops = vec![
                ("Rd", Int(rd + m)),
                ("Ra", Int(second[&m])),
                ("Rb", Int(rd + m)),
                ("scaleU5", Int(16)),
            ];
            ops.extend(guarded(rd + m).iter().map(|&(k, ref v)| (k, v.clone())));
            add(
                asm("lea_lo_noimm__RRR_RRR", &ops).ok_or_else(|| Refused("cannot assemble LEA for LDSM".into()))?,
                "word of the matrix",
                k == 0,
            );
        }
    }
    let last = new.last_mut().unwrap();
    last.word = (last.word & !(31u128 << USCHED)) | (w & (31u128 << USCHED));
    Ok(new)
}

/// HMMA.1688 (`mma.m16n8k8` on halves): D[16x8] = A[16x8] * B[8x8] + C over the warp. Volta's
/// own instruction makes four products of 8x8x4 a warp, one for each eight threads, in four
/// steps (or two) each. Two of them do it, one for the even k and one for the odd (the
/// prototype's `matrix_product` says how).
///
/// The lanes that hold what a thread gathers and the two masks are those the function keeps
/// (`Ctx::kept`). The four registers of the gathered B are kept too: a product on the same
/// register of B as the last one that gathered, with nothing since that writes it and no
/// other way to it, does not gather again (`Ctx::kept_gathering`). In floats: 25 instructions
/// when it gathers, 14 when it does not (33 every time until 2026-11-18).
pub(super) fn translate_matrix_product(ctx: &mut Ctx, i: usize, d: &Decoded, wait: u128) -> Result<Vec<Item>, Refused> {
    use assemble::OpVal::Int;
    let w = d.word;
    let asm = assemble::assemble;
    if d.op_or(OP_PG, 7) != 7 || d.op_or(OP_PG_AT_NOT, 0) != 0 {
        return Err(Refused(format!("HMMA.1688 under a predicate at 0x{:x}", 16 * i)));
    }
    if d.op_or(OP_RA_AT_NEGATE, 0) != 0
        || d.op_or(OP_RB_AT_NEGATE, 0) != 0
        || d.op_or(OP_SATFINITE, 0) != 0
        || d.op_or(OP_SIZE, 0) != 0
        || d.op_or(OP_ALTFMT, 0) != 0
        || !matches!(d.op_or(OP_DSTFMT, 0), 0 | 1)
    {
        return Err(Refused(format!("this form of HMMA.1688 at 0x{:x}", 16 * i)));
    }
    let floats = d.op_or(OP_DSTFMT, 0) == 1; // accumulators in floats (four registers), or in halves (two)
    let rd = d.op_or(OP_RD, 255);
    let ra = d.op_or(OP_RA, 255);
    let rb = d.op_or(OP_RB, 255);
    let rc = d.op_or(OP_RC, 255);
    // (The zero register for the accumulators: a product with nothing to add to. Volta's
    // instruction takes it too. In floats only: in halves the accumulators go through an
    // exchange first, which nothing would be needed for; no code at hand has that form.)
    let zero = rc == 255;
    if rd == 255
        || ra == 255
        || rb == 255
        || !rd.is_multiple_of(2)
        || !ra.is_multiple_of(2)
        || (!rc.is_multiple_of(2) && !zero)
        || (zero && !floats)
    {
        return Err(Refused(format!("HMMA.1688 on these registers at 0x{:x}", 16 * i)));
    }
    let d_set: Vec<u32> = (rd..rd + 4).collect();
    let c_set: Vec<u32> = if zero { Vec::new() } else { (rc..rc + 4).collect() };
    // (The destination may hold the register of B: what a product reads of B it reads first, into
    // registers of the translation; a product that does not gather does not read `rb` at all,
    // and the next one on that `rb` gathers again, because `kept_gathering` finds this product's
    // own write of it; a product of a pair on whole rows reads B from the rows that the
    // translation keeps. FlashInfer's attention kernels have such products. The destinations
    // that hold A stay refused: the steps read A again.)
    if floats
        && ((rd != rc && d_set.iter().any(|r| c_set.contains(r))) || d_set.iter().any(|r| *r == ra || *r == ra + 1))
    {
        return Err(Refused(format!(
            "HMMA.1688 that writes over its operands at 0x{:x}",
            16 * i
        )));
    }
    // Level 3: everything taken at the instruction. The lane is read there, the four values
    // (low, high, half, odd) computed there, B is gathered every time.
    let level3 = ctx.borrow.as_ref().is_some_and(|b| b.everything);
    let (low, high, half, near, even, other, sums, sums_hi, swapped, pair, b1, b2, s_all, takes);
    if level3 {
        let lane = ctx.lane()?;
        let low_r = ctx.take(1, None)?;
        let high_r = ctx.take(1, None)?;
        let half_r = ctx.take(1, None)?;
        // The scratch registers of the product, taken by their alignment need instead of
        // as one block of 8: room can then be made in small functions where 8 consecutive
        // registers are not available.
        let sums_lo_r = ctx.take(2, Some("product sums 0-1"))?;
        let sums_hi_r = ctx.take(2, Some("product sums 2-3"))?;
        let swapped_r = ctx.take(2, Some("product swapped"))?;
        let pair_r = if !floats { ctx.take(1, Some("product pair"))? } else { 0 };
        let b1_r = ctx.take(2, Some("product gathered B1"))?;
        let b2_r = ctx.take(2, Some("product gathered B2"))?;
        // Names for the existing code.
        low = low_r;
        high = high_r;
        half = half_r;
        sums = sums_lo_r;
        sums_hi = sums_hi_r;
        near = sums_lo_r;
        even = sums_lo_r + 1;
        other = sums_hi_r;
        swapped = swapped_r;
        pair = pair_r;
        b1 = b1_r;
        b2 = b2_r;
        s_all = [sums_lo_r, sums_lo_r + 1, sums_hi_r, sums_hi_r + 1];
        takes = false; // always gather at level 3
        let _ = lane; // used below by the closures that capture ctx via &mut
    } else {
        let (Some(low_r), Some(b1_r)) = (ctx.kept_at, ctx.gathered_at) else {
            return Err(Refused(
                "a product in a function that was not given the registers it keeps".into(),
            ));
        };
        let s = ctx.scratch()?;
        low = low_r;
        high = low_r + 1;
        half = low_r + 2;
        let b2_r = b1_r + 2;
        // Of the scratch registers: the four of the sums, three of which first hold what is being
        // gathered of B; A with its halves swapped; and, for halves, all ones in the threads 2
        // and 3 of a group.
        sums = s;
        sums_hi = s + 2;
        near = s;
        even = s + 1;
        other = s + 2;
        swapped = s + 4;
        pair = s + 6;
        b1 = b1_r;
        b2 = b2_r;
        s_all = [s, s + 1, s + 2, s + 3];
        takes = ctx.kept_gathering(i, rb);
        if !takes {
            ctx.holds = Some((i, rb));
        }
    }

    let mut new: Vec<Item> = Vec::new();
    let mut add = |word: u128, what: &'static str, waits: bool| {
        let wait_bits = if waits { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        new.push(Item::new(word | wait | wait_bits, i, what, true));
    };
    let shf = |rd_: u32, ra_: u32, sb_: u32, rc_: u32, dir_: &str, hilo_: &str, fmt_: &str| {
        asm(
            "shf__RuIR_RIR",
            &[
                ("Rd", Int(rd_)),
                ("Ra", Int(ra_)),
                ("Sb", Int(sb_)),
                ("Rc", Int(rc_)),
                ("dir", assemble::OpVal::Str(dir_)),
                ("hilo", assemble::OpVal::Str(hilo_)),
                ("fmt", assemble::OpVal::Str(fmt_)),
            ],
        )
    };
    let shf_cw = |rd_: u32, ra_: u32, sb_: u32, rc_: u32, dir_: &str, cw_: &str, hilo_: &str| {
        asm(
            "shf__RuIR_RIR",
            &[
                ("Rd", Int(rd_)),
                ("Ra", Int(ra_)),
                ("Sb", Int(sb_)),
                ("Rc", Int(rc_)),
                ("dir", assemble::OpVal::Str(dir_)),
                ("cw", assemble::OpVal::Str(cw_)),
                ("hilo", assemble::OpVal::Str(hilo_)),
                ("fmt", assemble::OpVal::Str("U32")),
            ],
        )
    };
    let choose = |rd_: u32, ra_: u32, rb_: u32, rc_: u32| {
        asm(
            "lop3_lut__RRR_RRR",
            &[
                ("Pu", Int(7)),
                ("Pp", Int(7)),
                ("Pp_not", Int(1)),
                ("imm8", Int(0xe4)),
                ("Rd", Int(rd_)),
                ("Ra", Int(ra_)),
                ("Rb", Int(rb_)),
                ("Rc", Int(rc_)),
            ],
        )
    };

    // `odd`: at level 3, computed after gathering into the register that held `half`
    // (since low/high/half are dead after the final shuffles). At other levels, from `kept`.
    let odd;
    let lut = |rd_: u32, ra_: u32, sb_: u32, rc_: u32, imm8: u32| {
        asm(
            "lop3_lut__RuIR_RIR",
            &[
                ("Pu", Int(7)),
                ("Pp", Int(7)),
                ("Pp_not", Int(1)),
                ("Rd", Int(rd_)),
                ("Ra", Int(ra_)),
                ("Sb", Int(sb_)),
                ("Rc", Int(rc_)),
                ("imm8", Int(imm8)),
            ],
        )
    };
    if level3 {
        // At level 3: compute low, high, half from lane, then gather B (always), then
        // compute odd into the register that held half (dead after gathering: its last use
        // is the two choose operations before the final shuffles to b1..b2+1).
        let lane = ctx.lane()?;
        add(shf(low, 255, 1, lane, "R", "HI", "U32").unwrap(), "kept: low", false);
        add(lut(low, low, 8, 255, 0xc0).unwrap(), "kept: low", false);
        add(lut(low, low, 3, lane, 0xf8).unwrap(), "kept: low", false);
        add(lut(high, low, 16, 255, 0xfc).unwrap(), "kept: high", false);
        add(shf(half, lane, 28, 255, "L", "LO", "U32").unwrap(), "kept: half", false);
        add(shf(half, 255, 31, half, "R", "HI", "S32").unwrap(), "kept: half", false);
        // Gather B.
        add(
            asm(
                "shfl__RII",
                &[
                    ("shflmd", Int(3)),
                    ("Pu", Int(7)),
                    ("Rd", Int(near)),
                    ("Ra", Int(rb)),
                    ("Sb", Int(4)),
                    ("Sc", Int(31)),
                    ("dst_wr_sb", Int(SEQUENCE)),
                ],
            )
            .unwrap(),
            "neighbour",
            false,
        );
        add(shf(even, rb, 16, 255, "L", "LO", "U32").unwrap(), "columns", false);
        add(
            shf_cw(swapped, ra, 16, ra, "L", "W", "HI").unwrap(),
            "halves swapped",
            false,
        );
        add(
            shf_cw(swapped + 1, ra + 1, 16, ra + 1, "L", "W", "HI").unwrap(),
            "halves swapped",
            false,
        );
        add(shf(other, 255, 16, near, "R", "HI", "U32").unwrap(), "columns", true);
        add(shf_cw(even, even, 16, near, "R", "W", "LO").unwrap(), "columns", false);
        add(shf_cw(other, rb, 16, other, "L", "W", "HI").unwrap(), "columns", false);
        add(choose(near, other, even, half).unwrap(), "columns", false);
        add(choose(even, even, other, half).unwrap(), "columns", false);
        for &(to, value, lanes) in &[
            (b1, near, low),
            (b1 + 1, near, high),
            (b2, even, low),
            (b2 + 1, even, high),
        ] {
            add(
                asm(
                    "shfl__RRI",
                    &[
                        ("shflmd", Int(0)),
                        ("Pu", Int(7)),
                        ("Rd", Int(to)),
                        ("Ra", Int(value)),
                        ("Rb", Int(lanes)),
                        ("Sc", Int(31)),
                        ("dst_wr_sb", Int(SEQUENCE)),
                    ],
                )
                .unwrap(),
                "gathered",
                false,
            );
        }
        // low, high, half are dead. Compute odd into the register that held half.
        odd = half;
        add(shf(odd, lane, 31, 255, "L", "LO", "U32").unwrap(), "kept: odd", false);
        add(shf(odd, 255, 31, odd, "R", "HI", "S32").unwrap(), "kept: odd", false);
    } else if takes {
        odd = low + 3;
        add(
            shf_cw(swapped, ra, 16, ra, "L", "W", "HI").unwrap(),
            "halves swapped",
            false,
        );
        add(
            shf_cw(swapped + 1, ra + 1, 16, ra + 1, "L", "W", "HI").unwrap(),
            "halves swapped",
            false,
        );
    } else {
        odd = low + 3;
        // The elements (2k, c), (2k, c + 1) and (2k + 1, c), (2k + 1, c + 1) of B, in the lanes of an even group c.
        add(
            asm(
                "shfl__RII",
                &[
                    ("shflmd", Int(3)),
                    ("Pu", Int(7)),
                    ("Rd", Int(near)),
                    ("Ra", Int(rb)),
                    ("Sb", Int(4)),
                    ("Sc", Int(31)),
                    ("dst_wr_sb", Int(SEQUENCE)),
                ],
            )
            .unwrap(),
            "neighbour",
            false,
        );
        add(shf(even, rb, 16, 255, "L", "LO", "U32").unwrap(), "columns", false);
        add(
            shf_cw(swapped, ra, 16, ra, "L", "W", "HI").unwrap(),
            "halves swapped",
            false,
        );
        add(
            shf_cw(swapped + 1, ra + 1, 16, ra + 1, "L", "W", "HI").unwrap(),
            "halves swapped",
            false,
        );
        add(shf(other, 255, 16, near, "R", "HI", "U32").unwrap(), "columns", true);
        add(shf_cw(even, even, 16, near, "R", "W", "LO").unwrap(), "columns", false);
        add(shf_cw(other, rb, 16, other, "L", "W", "HI").unwrap(), "columns", false);
        // The upper half gives the other parity: first the odd k there, then the even.
        add(choose(near, other, even, half).unwrap(), "columns", false);
        add(choose(even, even, other, half).unwrap(), "columns", false);
        for &(to, value, lanes) in &[
            (b1, near, low),
            (b1 + 1, near, high),
            (b2, even, low),
            (b2 + 1, even, high),
        ] {
            add(
                asm(
                    "shfl__RRI",
                    &[
                        ("shflmd", Int(0)),
                        ("Pu", Int(7)),
                        ("Rd", Int(to)),
                        ("Ra", Int(value)),
                        ("Rb", Int(lanes)),
                        ("Sc", Int(31)),
                        ("dst_wr_sb", Int(SEQUENCE)),
                    ],
                )
                .unwrap(),
                "gathered",
                false,
            );
        }
    }

    if !floats {
        return translate_product_in_halves(
            ctx,
            i,
            d,
            wait,
            &mut new,
            choose,
            shf,
            (rd, ra, rc),
            (pair, odd, b1, b2, swapped, sums, s_all),
            level3,
        );
    }

    let step = |n: u32, to: u32, a: u32, b: u32, c: u32, sb: Option<u32>, rel: Option<u32>| {
        let mut ops = vec![
            ("size", Int(0)),
            ("dstfmt", Int(1)),
            ("srcfmt", Int(1)),
            ("step", Int(n)),
            ("satfinite", Int(0)),
            ("Rd", Int(to)),
            ("Ra", Int(a)),
            ("tnspA", Int(1)),
            ("Rb", Int(b)),
            ("tnspB", Int(0)),
            ("Rc", Int(c)),
        ];
        if let Some(s) = sb {
            ops.push(("dst_wr_sb", Int(s)));
        }
        if let Some(r) = rel {
            ops.push(("src_rel_sb", Int(r)));
        }
        with_stall(asm("hmma__884_f32", &ops).unwrap(), 2, false)
    };
    // Registers 0 to 3 of Volta's eight in scratch registers, 4 to 7 in the destination.
    // (At level 3, sums may not be contiguous: s_all names the four registers.)
    // (Drop the add closure so we can borrow new directly for .covered.)
    drop(add);
    let upper = if zero { rc } else { rc + 2 };
    let (s_lo, s_hi) = (sums, sums_hi);
    for (n, (to, c)) in [(s_lo, rc), (s_hi, upper), (rd, rc), (rd + 2, upper)]
        .iter()
        .enumerate()
    {
        let sb = if n == 3 { Some(SEQUENCE) } else { None };
        let wait_bits = if n == 0 { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        let mut item = Item::new(
            step(n as u32, *to, ra, b1, *c, sb, None) | wait | wait_bits,
            i,
            "product",
            true,
        );
        item.covered = true;
        new.push(item);
    }
    for (n, to) in [s_lo, s_hi, rd, rd + 2].iter().enumerate() {
        let (sb, rel) = if n == 3 {
            (Some(SEQUENCE), Some(SEQUENCE))
        } else {
            (None, None)
        };
        let wait_bits = if n == 0 { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        let mut item = Item::new(
            step(n as u32, *to, swapped, b2, *to, sb, rel) | wait | wait_bits,
            i,
            "product",
            true,
        );
        item.covered = true;
        new.push(item);
    }
    {
        let last = new.last_mut().unwrap();
        last.word = with_stall(last.word, 8, false);
    }
    for k in 0..4u32 {
        let wait_bits = if k == 0 { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        new.push(Item::new(
            choose(rd + k, rd + k, s_all[k as usize], odd).unwrap() | wait | wait_bits,
            i,
            "the thread's own",
            true,
        ));
    }
    {
        let last = new.last_mut().unwrap();
        last.word = (last.word & !(31u128 << USCHED)) | (w & (31u128 << USCHED));
    }
    Ok(new)
}

/// The end of `matrix_product` for accumulators in halves (HMMA.1688.F16): Volta's
/// instruction then writes four registers of two halves, and a thread holds one row of its
/// eight threads' product.
/// At level 3: `pair` is computed from `lane` here (after gathering: the register that held
/// `half` now holds `odd`, and `pair` goes into a scratch register).
#[allow(clippy::too_many_arguments)]
fn translate_product_in_halves(
    ctx: &mut Ctx,
    i: usize,
    d: &Decoded,
    wait: u128,
    new: &mut Vec<Item>,
    choose: impl Fn(u32, u32, u32, u32) -> Option<u128>,
    shf: impl Fn(u32, u32, u32, u32, &str, &str, &str) -> Option<u128>,
    registers: (u32, u32, u32),
    scratch: (u32, u32, u32, u32, u32, u32, [u32; 4]),
    level3: bool,
) -> Result<Vec<Item>, Refused> {
    use assemble::OpVal::Int;
    let w = d.word;
    let asm = assemble::assemble;
    let lane = ctx.lane()?;
    let (rd, ra, rc) = registers;
    let (pair, odd, b1, b2, swapped, _sums, sa) = scratch;

    let mut add = |word: u128, what: &'static str, waits: bool| {
        let wait_bits = if waits { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        new.push(Item::new(word | wait | wait_bits, i, what, true));
    };

    // All ones in the threads 2 and 3 of each group (bit 1 of the lane).
    add(shf(pair, lane, 30, 255, "L", "LO", "U32").unwrap(), "pair", false);
    add(shf(pair, 255, 31, pair, "R", "HI", "S32").unwrap(), "pair", false);
    // The accumulator this thread holds for the other one, sent; then its row of the eight threads' matrix.
    add(choose(sa[2], rc, rc + 1, pair).unwrap(), "for the other", false);
    add(
        asm(
            "shfl__RII",
            &[
                ("shflmd", Int(3)),
                ("Pu", Int(7)),
                ("Rd", Int(sa[3])),
                ("Ra", Int(sa[2])),
                ("Sb", Int(2)),
                ("Sc", Int(31)),
                ("dst_wr_sb", Int(SEQUENCE)),
            ],
        )
        .unwrap(),
        "exchange",
        false,
    );
    add(choose(sa[0], sa[3], rc, pair).unwrap(), "row", true);
    add(choose(sa[1], rc + 1, sa[3], pair).unwrap(), "row", false);
    add(choose(sa[2], sa[3], rc, pair).unwrap(), "row", false);
    add(choose(sa[3], rc + 1, sa[3], pair).unwrap(), "row", false);

    let step = |n: u32, a: u32, b: u32, sb: Option<u32>, rel: Option<u32>| {
        let mut ops = vec![
            ("size", Int(0)),
            ("dstfmt", Int(0)),
            ("srcfmt", Int(0)),
            ("step", Int(n)),
            ("satfinite", Int(0)),
            ("Rd", Int(sa[2 * n as usize])),
            ("Ra", Int(a)),
            ("tnspA", Int(1)),
            ("Rb", Int(b)),
            ("tnspB", Int(0)),
            ("Rc", Int(sa[2 * n as usize])),
        ];
        if let Some(s) = sb {
            ops.push(("dst_wr_sb", Int(s)));
        }
        if let Some(r) = rel {
            ops.push(("src_rel_sb", Int(r)));
        }
        with_stall(asm("hmma__884_f16_f16", &ops).unwrap(), 4, false)
    };
    // As the compiler has them: four cycles apart, thirteen after the last.
    // (Drop the add closure so we can borrow new directly for .covered.)
    drop(add);
    for n in 0..2u32 {
        let sb = if n == 1 { Some(SEQUENCE) } else { None };
        let wait_bits = if n == 0 { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        let mut item = Item::new(step(n, ra, b1, sb, None) | wait | wait_bits, i, "product", true);
        item.covered = true;
        new.push(item);
    }
    for n in 0..2u32 {
        let (sb, rel) = if n == 1 {
            (Some(SEQUENCE), Some(SEQUENCE))
        } else {
            (None, None)
        };
        let wait_bits = if n == 0 { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        let mut item = Item::new(step(n, swapped, b2, sb, rel) | wait | wait_bits, i, "product", true);
        item.covered = true;
        new.push(item);
    }
    {
        let last = new.last_mut().unwrap();
        last.word = with_stall(last.word, 13, false);
    }
    // Back: the row of the thread, what it holds for the other one sent, and its own two registers.
    // (Reinstate the add helper as a function call: push with wait bits.)
    let push_item = |new: &mut Vec<Item>, word: u128, what: &'static str, waits: bool| {
        let wait_bits = if waits { 1u128 << (WAIT + SEQUENCE) } else { 0 };
        new.push(Item::new(word | wait | wait_bits, i, what, true));
    };
    push_item(new, choose(sa[0], sa[2], sa[0], odd).unwrap(), "row", true);
    push_item(new, choose(sa[1], sa[3], sa[1], odd).unwrap(), "row", false);
    push_item(new, choose(sa[2], sa[0], sa[1], pair).unwrap(), "for the other", false);
    push_item(
        new,
        asm(
            "shfl__RII",
            &[
                ("shflmd", Int(3)),
                ("Pu", Int(7)),
                ("Rd", Int(sa[3])),
                ("Ra", Int(sa[2])),
                ("Sb", Int(2)),
                ("Sc", Int(31)),
                ("dst_wr_sb", Int(SEQUENCE)),
            ],
        )
        .unwrap(),
        "exchange",
        false,
    );
    push_item(new, choose(rd, sa[3], sa[0], pair).unwrap(), "the thread's own", true);
    push_item(
        new,
        choose(rd + 1, sa[1], sa[3], pair).unwrap(),
        "the thread's own",
        false,
    );
    {
        let last = new.last_mut().unwrap();
        last.word = (last.word & !(31u128 << USCHED)) | (w & (31u128 << USCHED));
    }
    let _ = level3;
    Ok(std::mem::take(new))
}

pub(super) fn translate_uniform_address(
    ctx: &mut Ctx,
    i: usize,
    d: &Decoded,
    plain: &str,
) -> Result<Vec<Item>, Refused> {
    let w = d.word;
    let part_id = if d.has_op(OP_RA_URB) { OP_RA_URB } else { OP_RA_URC };
    let part_name = if part_id == OP_RA_URB { "Ra_URb" } else { "Ra_URc" };
    let shared = SHARED.contains(&plain);
    let drop_key = if shared {
        format!("{},stride,Ra", part_name)
    } else {
        format!("{},sp2,TABLE,Ra", part_name)
    };
    let rule_idx = rule::find_rule(d.name, plain, &drop_key)
        .ok_or_else(|| Refused(format!("no rule from {} to {}", d.name, plain)))?;
    for &(key, fid) in rule::dropped_fields(rule_idx) {
        // (What a load says to the cache about its neighbours, `sp2`, sm_70 has no way to say: it
        // is left out, as for a load at a plain address.)
        if !matches!(key, "Ra" | "stride" | "sp2") && key != part_name && field::get(fid, w) != 0 {
            return Err(Refused(format!(
                "{} at 0x{:x} asks for something sm_70 loads do not have ({})",
                d.name,
                16 * i,
                key
            )));
        }
    }
    let out = apply_rule(ctx, rule_idx, i, d)?;
    let part_val = d.op_or(part_id, 63);
    if part_val == 63 {
        return Err(Refused(format!(
            "{} at 0x{:x} has a value that sm_70 does not have",
            d.name,
            16 * i
        )));
    }
    let uniform_r = ctx.r(part_val)?;
    let ra = d.op_or(OP_RA, 255);
    let wait = w & (63u128 << WAIT);

    if ra == 255 {
        // Nothing but the uniform part: the plain instruction on its ordinary registers.
        let out_word = put_operand_by_class(plain, out, "Ra", uniform_r as u64, false)?;
        return Ok(vec![late(plain, out_word, i, "on the uniform part")?]);
    }

    if shared {
        // In shared memory: 32 bits, the register times its stride plus the uniform part.
        let scale = match d.op_or(OP_STRIDE, 0) {
            0 => 1,
            1 => 4,
            2 => 8,
            3 => 16,
            other => return Err(Refused(format!("unknown stride {}", other))),
        };
        let shadow = ctx.pool(1)?;
        let add = assemble::assemble(
            "imad__RsIR_RIR",
            &[
                ("Rd", assemble::OpVal::Int(shadow)),
                ("Ra", assemble::OpVal::Int(ra)),
                ("Sb", assemble::OpVal::Int(scale)),
                ("Rc", assemble::OpVal::Int(uniform_r)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble IMAD for address".into()))?;
        let out_word = put_operand_by_class(plain, out, "Ra", shadow as u64, false)?;
        return Ok(vec![
            Item::new(add | wait, i, "address", true),
            late(plain, out_word, i, "on the sum")?,
        ]);
    }

    // The lower word of Ra, taken as unsigned, added to the 64 bits of the uniform part; then the
    // upper word of Ra if it has one.
    let wide = d.name.ends_with("Ra64");
    let shadow = ctx.pool(2)?;
    let mut items = vec![];
    let add_lo = assemble::assemble(
        "imad_wide__RsIR_RIR",
        &[
            ("Rd", assemble::OpVal::Int(shadow)),
            ("Ra", assemble::OpVal::Int(ra)),
            ("Sb", assemble::OpVal::Int(1)),
            ("Rc", assemble::OpVal::Int(uniform_r)),
            ("fmt", assemble::OpVal::Str("U32")),
        ],
    )
    .ok_or_else(|| Refused("cannot assemble IMAD.WIDE for address".into()))?;
    items.push(Item::new(add_lo | wait, i, "address", true));
    if wide {
        let add_hi = assemble::assemble(
            "iadd3_noimm__RRR_RRR",
            &[
                ("Rd", assemble::OpVal::Int(shadow + 1)),
                ("Ra", assemble::OpVal::Int(shadow + 1)),
                ("Rb", assemble::OpVal::Int(ra + 1)),
                ("Rc", assemble::OpVal::Int(255)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble IADD3 for address".into()))?;
        items.push(Item::new(add_hi | wait, i, "address", true));
    }
    let out_word = put_operand_by_class(plain, out, "Ra", shadow as u64, false)?;
    items.push(late(plain, out_word, i, "on the sum")?);
    Ok(items)
}

pub(super) fn translate_rule(ctx: &mut Ctx, i: usize, d: &Decoded, target: &str) -> Result<Vec<Item>, Refused> {
    let rule_idx =
        rule::find_rule(d.name, target, "").ok_or_else(|| Refused(format!("no rule from {} to {}", d.name, target)))?;
    let mut out = apply_rule(ctx, rule_idx, i, d)?;
    let after = ctx.dec70.decode(out);
    let mut await_write = false;
    if let Some(ref after_d) = after {
        if LATE_WRITER_KINDS.contains(&after_d.kind) && ((out >> WRITE_BARRIER) & 7) == 7 {
            let writes_reg = after_d.writes.iter().any(|&(letter, r)| letter == LETTER_R && r != 255);
            let writes_borrowed = ctx.borrow.as_ref().is_some_and(|b| {
                after_d
                    .writes
                    .iter()
                    .any(|&(letter, r)| letter == LETTER_R && r != 255 && b.used.contains(&r))
            });
            let upgraded_writer = !LATE_WRITER_KINDS.contains(&d.kind) && writes_reg;
            if writes_borrowed || upgraded_writer {
                // A rule target whose result arrives late (`MIO_FAST` / `MIO_SLOW`, such as `UFLO`
                // -> `FLO` or `S2UR` -> `S2R`) must name a write scoreboard when writing a borrowed
                // register or when translated from a fixed-latency instruction (`!LATE_WRITER_KINDS`,
                // whose readers in the program wait on no scoreboard), before `late()` chooses the
                // read scoreboard so the two barriers stay distinct (`read_announced`).
                out = (out & !(7 << WRITE_BARRIER)) | (u128::from(ARRIVES) << WRITE_BARRIER);
                await_write = upgraded_writer;
            }
        }
        if after_d.kind.is_late() {
            let cls = &crate::generated::classes::VF_CLASSES[after_d.class_id as usize];
            let reads = &crate::generated::classes::VF_READS[cls.12 as usize..(cls.12 + cls.13) as usize];
            // Whether an operand of the result stands for a uniform operand of the instruction: the
            // register that stands for it in the whole function, or the one it was loaded into from
            // the frame for this instruction.
            let from_uniform = |name_id: u16| {
                let back = ordinary_back(OP_NAMES[name_id as usize], &d.ops);
                back != name_id
                    && d.op(back)
                        .is_some_and(|u| ctx.place.contains_key(&u) || ctx.mem_regs.contains_key(&u))
            };
            if reads
                .iter()
                .any(|&(letter, name_id, _)| letter == LETTER_R && after_d.has_op(name_id) && from_uniform(name_id))
            {
                // It reads a uniform register, and reads its operands late (`late`).
                let mut item = late(after_d.name, out, i, "ordinary registers")?;
                if await_write {
                    item.also.push(ARRIVES);
                }
                return Ok(vec![item]);
            }
        }
    }
    let mut item = Item::new(out, i, "ordinary registers", true);
    if await_write {
        item.also.push(ARRIVES);
    }
    Ok(vec![item])
}

fn ordinary_back(operand: &str, ops: &[(u16, u32)]) -> u16 {
    // Map back: for "Ra" check "URa", etc.
    let u_name = format!("U{}", operand);
    let ur_name = operand.replace("_R", "_UR");
    for &(nid, _) in ops {
        let name = OP_NAMES[nid as usize];
        if name == u_name || name == ur_name {
            return nid;
        }
    }
    // Return the original name ID
    OP_NAMES.iter().position(|&n| n == operand).unwrap_or(0) as u16
}
