//! Return addresses: find the MOV instructions in the translated code that hold return
//! addresses or jump-table entries, and rewrite them once the new offsets are known.

use std::collections::{HashMap, HashSet};

use crate::decode::{Kind, LETTER_R};
use crate::generated::fields::*;

use super::parse::FuncData;
use super::{Item, Refused, hex};

// ---- Return addresses ----

/// Also finds the jump tables to move with the code: (where in the constants of the function, the
/// places as they are).
pub(super) fn return_addresses(
    items: &mut [Item],
    targets: &[u64],
    fd: &FuncData,
) -> Result<Vec<(u32, Vec<u64>)>, Refused> {
    let src = &fd.src;
    let mut relocate: Vec<(u32, Vec<u64>)> = Vec::new();
    // Build MOV immediate index: immediate value -> list of item indices
    let mut movs: HashMap<u32, Vec<usize>> = HashMap::new();
    for (k, item) in items.iter().enumerate() {
        if let Some(v) = immediate_of(item) {
            movs.entry(v).or_default().push(k);
        }
    }

    // A LEPC immediately before a direct call is that call's return address: both instructions
    // move with the code, so the address the LEPC reads stays the instruction after the call.
    let mut link_lepc: HashSet<usize> = HashSet::new();
    for k in 1..items.len() {
        let d = items[k].d.as_ref().unwrap();
        if !d.name.starts_with("call_rel") || targets.contains(&(16 * items[k].origin as u64)) {
            continue;
        }
        let p = items[k - 1].d.as_ref().unwrap();
        if p.mnemonic == "LEPC" {
            link_lepc.insert(k - 1);
        }
    }

    // First pass: collect fixups
    let mut fixups: Vec<(usize, u64)> = Vec::new(); // (item_index, old_offset_it_stands_for)

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
            if k > 0 && link_lepc.contains(&(k - 1)) {
                // The direct call's return address is the program counter read by the LEPC
                // immediately before it. No number is fixed up: the LEPC moves with the code.
            } else {
                let back = (16 * origin + 16) as u32;
                let found: Vec<usize> = movs.get(&back).cloned().unwrap_or_default();
                if found.len() != 1 || found[0] > k {
                    return Err(Refused(format!(
                        "the return address of the call at 0x{:x} is not given by exactly one MOV",
                        16 * origin
                    )));
                }
                let j = found[0];
                let register = items[j].d.as_ref().unwrap().op_or(OP_RD, 255);
                for between in &items[j + 1..k] {
                    let bd = between.d.as_ref().unwrap();
                    if bd.writes.iter().any(|&(l, r)| l == LETTER_R && r == register)
                        || (between.first && targets.contains(&(16 * between.origin as u64)))
                        || bd.kind == Kind::BRU
                    {
                        return Err(Refused(format!(
                            "the return address of the call at 0x{:x} does not reach it in a straight line",
                            16 * origin
                        )));
                    }
                }
                fixups.push((j, (16 * origin + 16) as u64));
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
            let has_call = (k + 1..std::cmp::min(items.len(), k + 16))
                .any(|j| items[j].d.as_ref().unwrap().name.starts_with("call_abs"))
                || link_lepc.contains(&k);
            if !has_call {
                return Err(Refused(format!("LEPC at 0x{:x} without a call after it", 16 * origin)));
            }
        }
    }

    // Apply fixups
    for (idx, old_offset) in fixups {
        items[idx].fix = Some(old_offset);
    }

    Ok(relocate)
}

fn immediate_of(item: &Item) -> Option<u32> {
    let d = item.d.as_ref()?;
    if d.name == "mov__RuIR_RI" || (d.mnemonic == "MOV" && d.has_op(OP_SB) && !d.has_op(OP_RB) && !d.has_op(OP_SB_BANK))
    {
        return d.op(OP_SB);
    }
    None
}
