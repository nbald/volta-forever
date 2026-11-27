//! Cubin-level translation: parse the ELF, group functions by call edges, translate
//! each group, rewrite attributes and relocations, and re-tag for sm_70.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use crate::assemble;
use crate::decode::Decoder;
use crate::elf::Elf;
use crate::fast::{FastMap, FastSet};
use crate::generated::special::*;

use super::attributes::{check_attributes, find_exits, rewrite_attributes, strip_attributes, strip_attrs_by_kind};
use super::borrowed::BorrowedPrepared;
use super::context::{Predicates, entries, stand_ins};
use super::function::{follow, translate_function};
use super::parallel::{effective_threads, par_map_init};
use super::parse::{FuncData, parse_function, placement, tables_of};
use super::{
    FunctionReport, LifeEntry, MapEntry, Options, Refused, Relocation, STAY, TRAP, TransResult, TranslateResult,
    with_stall,
};

const PARALLEL_CODE_BYTES: usize = 64 * 1024;

/// Translate an sm_75 or supported sm_80 cubin to sm_70.
pub fn translate(data: &[u8], want_map: bool) -> Result<TranslateResult, String> {
    translate_with_options(data, want_map, &Options::default())
}

/// Translate an sm_75 or supported sm_80 cubin to sm_70 with the given options.
pub fn translate_with_options(data: &[u8], want_map: bool, opts: &Options) -> Result<TranslateResult, String> {
    translate_with_threads(data, want_map, opts, None)
}

pub(super) fn translate_with_threads(
    data: &[u8],
    want_map: bool,
    opts: &Options,
    forced_threads: Option<usize>,
) -> Result<TranslateResult, String> {
    let mut e = Elf::parse(data)?;
    let arch = super::ampere::source(&e.header)?;
    let mut symbols = e.symbols()?;
    let sec_map = e.section_index_map();
    let callgraph_sec = sec_map.get(".nv.callgraph").copied();
    let nv_info_sec = sec_map.get(".nv.info").copied();
    drop(sec_map);

    let mut texts: Vec<usize> = Vec::new();
    let mut text_names: FastSet<&str> = FastSet::default();
    let mut text_slot: FastMap<usize, usize> = FastMap::default();
    let mut text_by_name: FastMap<&str, usize> = FastMap::default();

    for (i, s) in e.sections.iter().enumerate() {
        if let Some(fn_name) = s.name.strip_prefix(".text.")
            && s.stype == 1
        {
            if !text_names.insert(s.name.as_str()) {
                return Err("two sections of code with the same name".into());
            }
            let slot = texts.len();
            texts.push(i);
            text_slot.insert(i, slot);
            text_by_name.insert(fn_name, slot);
        }
    }

    let n_texts = texts.len();
    let mut info_sec: Vec<Option<usize>> = vec![None; n_texts];
    // Every section of attributes of a function's name, for what it calls (the first one is
    // the function's for everything else, as `Elf::section` finds it).
    let mut info_secs: Vec<Vec<usize>> = vec![Vec::new(); n_texts];
    let mut const2_sec: Vec<Option<usize>> = vec![None; n_texts];
    for (i, s) in e.sections.iter().enumerate() {
        if let Some(fn_name) = s.name.strip_prefix(".nv.info.") {
            if let Some(&slot) = text_by_name.get(fn_name) {
                info_sec[slot].get_or_insert(i);
                info_secs[slot].push(i);
            }
        } else if let Some(fn_name) = s.name.strip_prefix(".nv.constant2.")
            && let Some(&slot) = text_by_name.get(fn_name)
        {
            const2_sec[slot].get_or_insert(i);
        }
    }

    let mut symbols_by_text: Vec<Vec<usize>> = vec![Vec::new(); n_texts];
    let mut by_symbol: FastMap<usize, usize> = FastMap::default();
    let mut entry_secs: FastSet<usize> = FastSet::default();
    let mut entered_inside: Vec<bool> = vec![false; n_texts];

    for (sym_idx, sym) in symbols.iter().enumerate() {
        let shndx = sym.shndx as usize;
        if let Some(&slot) = text_slot.get(&shndx) {
            symbols_by_text[slot].push(sym_idx);
            by_symbol.insert(sym_idx, slot);
            if sym.info & 0xf == 2 {
                if sym.other & 0x10 != 0 {
                    entry_secs.insert(shndx);
                }
                if sym.info >> 4 == 1 && sym.value != 0 {
                    entered_inside[slot] = true;
                }
            }
        }
    }

    // Union-find over `0..n_texts` for call groups.
    let mut group_parent: Vec<usize> = (0..n_texts).collect();
    fn find_group(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            let up = parent[parent[x]];
            parent[x] = up;
            x = up;
        }
        x
    }

    let mut relocations: FastMap<usize, Vec<Relocation>> = FastMap::default();
    let mut holders: FastSet<usize> = FastSet::default();
    let mut named: FastSet<usize> = FastSet::default();
    for (section, s) in e.sections.iter().enumerate() {
        let on = s.info as usize;
        if (s.stype != 4 && s.stype != 9) || on >= e.sections.len() {
            continue;
        }
        if e.sections[on].name.starts_with(".debug") || e.sections[on].name.starts_with(".nv_debug") {
            continue;
        }
        let step = if s.stype == 4 { 24 } else { 16 };
        let on_is_text = text_slot.contains_key(&on);
        for at in (0..s.data.len().saturating_sub(step - 1)).step_by(step) {
            let offset = u64::from_le_bytes(s.data[at..at + 8].try_into().unwrap());
            let info = u64::from_le_bytes(s.data[at + 8..at + 16].try_into().unwrap());
            let names = by_symbol.get(&((info >> 32) as usize)).map(|&slot| texts[slot]);
            if on_is_text {
                relocations.entry(on).or_default().push(Relocation {
                    section,
                    at,
                    offset,
                    kind: info as u32,
                });
                if names.is_some() {
                    holders.insert(on);
                }
            }
            if let Some(place) = names {
                named.insert(place);
            }
        }
    }

    // Edges from .nv.callgraph and .nv.info.* attribute 0x0f.
    if let Some(gi) = callgraph_sec {
        let gdata = &e.sections[gi].data;
        let mut at = 0;
        while at + 8 <= gdata.len() {
            let a = u32::from_le_bytes(gdata[at..at + 4].try_into().unwrap()) as usize;
            let b = i32::from_le_bytes(gdata[at + 4..at + 8].try_into().unwrap());
            if b >= 0
                && let (Some(&sa), Some(&sb)) = (by_symbol.get(&a), by_symbol.get(&(b as usize)))
            {
                let fa = find_group(&mut group_parent, sa);
                let fb = find_group(&mut group_parent, sb);
                if fa != fb {
                    group_parent[fa] = fb;
                }
            }
            at += 8;
        }
    }
    for (slot, sections) in info_secs.iter().enumerate() {
        for &ii in sections {
            let sdata = &e.sections[ii].data;
            let mut pos = 0;
            while pos + 4 <= sdata.len() {
                let fmt = sdata[pos];
                let attr = sdata[pos + 1];
                let val = u16::from_le_bytes(sdata[pos + 2..pos + 4].try_into().unwrap()) as usize;
                if fmt == 4 && attr == 0x0f {
                    let mut at = pos + 4;
                    while at + 4 <= (pos + 4 + val).min(sdata.len()) {
                        let callee = u32::from_le_bytes(sdata[at..at + 4].try_into().unwrap()) as usize;
                        if let Some(&cs) = by_symbol.get(&callee) {
                            let fa = find_group(&mut group_parent, slot);
                            let fb = find_group(&mut group_parent, cs);
                            if fa != fb {
                                group_parent[fa] = fb;
                            }
                        }
                        at += 4;
                    }
                }
                pos += if fmt == 4 { 4 + val } else { 4 };
            }
        }
    }

    let total_code_bytes: usize = texts.iter().map(|&si| e.sections[si].data.len()).sum();
    let large = total_code_bytes >= PARALLEL_CODE_BYTES;
    let parse_threads = forced_threads
        .map(|t| t.min(n_texts.max(1)).max(1))
        .unwrap_or_else(|| effective_threads(n_texts, large));

    // Parse each function (parallelized across `VF_THREADS`, preserving warm `dec75` per worker).
    let slot_indices: Vec<usize> = (0..n_texts).collect();
    let parsed_slots: Vec<Result<FuncData, Refused>> = par_map_init(
        &slot_indices,
        parse_threads,
        || Decoder::new(75),
        |dec75, _, &slot| -> Result<Result<FuncData, Refused>, std::convert::Infallible> {
            let res: Result<FuncData, Refused> = (|| {
                let si = texts[slot];
                let name = &e.sections[si].name[".text.".len()..];
                let registers = e.sections[si].info >> 24;
                let tables = match info_sec[slot] {
                    Some(ii) => tables_of(&e.sections[ii].data)?,
                    None => Vec::new(),
                };
                let code = &e.sections[si].data;
                let mut fd = if arch != 80 {
                    parse_function(dec75, name, code, registers, tables)?
                } else {
                    let mut helpers = symbols_by_text[slot].iter().copied().filter(|&index| {
                        let s = &symbols[index];
                        s.info & 15 == 2 && s.value != 0
                    });
                    let mut helper = None;
                    if let Some(index) = helpers.next() {
                        let s = &symbols[index];
                        if helpers.next().is_some()
                            || (s.info >> 4 != 0 && !vf_fatbin::cubin_has_assembler_helper(data, index))
                            || s.other & 0x10 != 0
                            || s.size == 0
                            || !s.value.is_multiple_of(16)
                            || s.value.checked_add(s.size) != Some(code.len() as u64)
                        {
                            return Err(Refused("unsupported secondary entry in sm_80 code".into()));
                        }
                        helper = Some(s.value as usize / 16);
                    }
                    if relocations.contains_key(&si) {
                        return Err(Refused("a relocation in sm_80 code".into()));
                    }
                    let normalized = super::ampere::normalize(code, dec75, helper)?;
                    let mut fd = parse_function(dec75, name, &normalized.words, registers, tables)?;
                    fd.ampere_ops = normalized.operations;
                    for (i, op) in &fd.ampere_ops {
                        match op {
                            super::ampere::Operation::GlobalToShared { .. } => {
                                fd.scratch_need = fd.scratch_need.max(6);
                            }
                            super::ampere::Operation::IncrementShared { .. } => {
                                fd.scratch_need = fd.scratch_need.max(2);
                            }
                            super::ampere::Operation::Bf16Add | super::ampere::Operation::Bf16Pack => {
                                fd.scratch_need = fd.scratch_need.max(3);
                            }
                            super::ampere::Operation::Pack => {
                                fd.needs_pool |= fd.src[*i].op_or(crate::generated::fields::OP_RC, 255) != 255;
                            }
                        }
                    }
                    fd.ampere = true;
                    fd
                };
                fd.entered_inside = entered_inside[slot];
                Ok(fd)
            })();
            Ok(res)
        },
    )
    .expect("parsing returns Result inside Vec");

    let before: Vec<usize> = texts.iter().map(|&si| e.sections[si].data.len() / 16).collect();
    let mut report_reasons: Vec<Option<String>> = vec![None; n_texts];
    let mut functions: Vec<Option<FuncData>> = Vec::with_capacity(n_texts);
    for (slot, res) in parsed_slots.into_iter().enumerate() {
        let si = texts[slot];
        match res {
            Ok(mut fd) => {
                if holders.contains(&si) {
                    report_reasons[slot] = Some("a relocation in its code names a place in code".into());
                    functions.push(None);
                } else if named.contains(&si) {
                    report_reasons[slot] = Some("a relocation names a place in its code".into());
                    functions.push(None);
                } else {
                    let _ = &mut fd;
                    functions.push(Some(fd));
                }
            }
            Err(why) => {
                report_reasons[slot] = Some(why.0);
                functions.push(None);
            }
        }
    }

    // Group functions by call edges, filtering out any group where a member was refused.
    let mut root_to_group: FastMap<usize, usize> = FastMap::default();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for slot in 0..n_texts {
        let root = find_group(&mut group_parent, slot);
        if let Some(&gidx) = root_to_group.get(&root) {
            groups[gidx].push(slot);
        } else {
            root_to_group.insert(root, groups.len());
            groups.push(vec![slot]);
        }
    }

    let mut active_groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for gslots in groups {
        if gslots.iter().any(|&slot| functions[slot].is_none()) {
            for &slot in &gslots {
                if functions[slot].is_some() {
                    report_reasons[slot] = Some("a function it calls, or that calls it, is refused".into());
                }
            }
        } else {
            let cost: usize = gslots
                .iter()
                .map(|&slot| functions[slot].as_ref().unwrap().src.len())
                .sum();
            active_groups.push((cost, gslots));
        }
    }

    // Sort longest-processing-time-first so large kernels schedule before small ones.
    active_groups.sort_by_key(|(cost, gslots)| (Reverse(*cost), gslots[0]));

    let say_borrows = std::env::var_os("VF_BORROW_SAY").is_some();
    let group_threads = forced_threads
        .map(|t| t.min(active_groups.len().max(1)).max(1))
        .unwrap_or_else(|| effective_threads(active_groups.len(), large));

    struct GroupOutput {
        outcome: Result<Vec<(usize, TransResult)>, String>,
        logs: Vec<String>,
    }

    let group_outputs: Vec<GroupOutput> = par_map_init(
        &active_groups,
        group_threads,
        || Decoder::new(70),
        |dec70, _, (_cost, gslots)| -> Result<GroupOutput, std::convert::Infallible> {
            let fs: Vec<&FuncData> = gslots.iter().map(|&s| functions[s].as_ref().unwrap()).collect();
            let indices: Vec<usize> = gslots.iter().map(|&s| texts[s]).collect();
            let logs = std::cell::RefCell::new(Vec::<String>::new());

            let declared = fs.iter().map(|f| f.registers).max().unwrap();
            let required = fs
                .iter()
                .zip(gslots.iter())
                .filter_map(|(_f, &slot)| {
                    let info = info_sec[slot].map_or(&[][..], |ii| e.sections[ii].data.as_slice());
                    let threads = super::block_registers::required_threads(info)?;
                    Some((super::block_registers::registers_for(threads), threads))
                })
                .min();

            if fs.len() > 1 && fs.iter().any(|f| !f.uniform_predicates.is_empty()) {
                return Ok(GroupOutput {
                    outcome: Err("uniform predicates in functions that call one another".into()),
                    logs: logs.into_inner(),
                });
            }
            if let Some(f) = fs.iter().find(|f| f.highest >= declared) {
                return Ok(GroupOutput {
                    outcome: Err(format!("register R{} is used and the function declares {}", f.highest, declared)),
                    logs: logs.into_inner(),
                });
            }

            let predicates: Result<Vec<Predicates>, Refused> = fs.iter().map(|f| stand_ins(f)).collect();
            let attempt_result = predicates.and_then(|predicates| {
                let saves = predicates.iter().map(|p| p.save_slots).max().unwrap_or(0);
                // Lazily compute Flow once per function (shared between pairing and Borrowed).
                let flow_cells: Vec<std::cell::OnceCell<crate::pairing::Flow>> =
                    (0..fs.len()).map(|_| std::cell::OnceCell::new()).collect();
                let flow_of = |k: usize| -> &crate::pairing::Flow {
                    flow_cells[k].get_or_init(|| {
                        let f = fs[k];
                        let ent = entries(f);
                        crate::pairing::Flow::new_with_facts(
                            &f.src,
                            &f.tables,
                            &ent,
                            opts.pairs && f.needs_lane && !f.entered_inside,
                        )
                    })
                };

                let paired = std::cell::LazyCell::new(|| {
                    fs.iter()
                        .enumerate()
                        .map(|(k, f)| {
                            if opts.pairs && f.needs_lane && !f.entered_inside {
                                crate::pairing::plan_with_flow(&f.src, flow_of(k))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                });
                let unpaired: Vec<Option<crate::pairing::Plan>> = fs.iter().map(|_| None).collect();

                // Cache BorrowedPrepared across borrow levels (1, 2, 3) for single-function groups.
                let borrowed_prep_cell: std::cell::OnceCell<Result<BorrowedPrepared, Refused>> =
                    std::cell::OnceCell::new();
                let get_borrowed_prep = || -> Result<&BorrowedPrepared, Refused> {
                    match borrowed_prep_cell.get_or_init(|| BorrowedPrepared::new(fs[0], flow_of(0))) {
                        Ok(p) => Ok(p),
                        Err(e) => Err(Refused(e.0.clone())),
                    }
                };

                let mut attempt = |do_pairs: bool,
                                   borrow_level: u32,
                                   top_override: Option<u32>,
                                   gave_up: &[String]|
                 -> Result<Vec<(usize, TransResult)>, Refused> {
                    let plans = if do_pairs { &*paired } else { &unpaired };
                    let has_product_flags: Vec<bool> = fs
                        .iter()
                        .zip(plans.iter())
                        .map(|(f, q)| {
                            q.as_ref().map_or_else(
                                || f.src.iter().any(|d| d.name == "hmma_x8_"),
                                |p| {
                                    f.src
                                        .iter()
                                        .enumerate()
                                        .any(|(i, d)| d.name == "hmma_x8_" && !p.sites.contains_key(&i))
                                },
                            )
                        })
                        .collect();
                    let mut rows: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
                    let mut shares: HashMap<u32, u32> = HashMap::new();
                    for plan in plans.iter().flatten() {
                        for (&rd, &count) in &plan.rows() {
                            let to = if fs.len() == 1 {
                                plan.share.get(&rd).copied().unwrap_or(rd)
                            } else {
                                rd
                            };
                            shares.insert(rd, to);
                            let entry = rows.entry(to).or_insert(0);
                            *entry = (*entry).max(count);
                        }
                    }
                    let row_kept: Vec<&'static str> = crate::pairing::ROW_KEPT
                        .iter()
                        .copied()
                        .filter(|&name| plans.iter().any(|q| q.as_ref().is_some_and(|p| p.kept().contains(name))))
                        .collect();
                    let held_preds = {
                        let mut s: Vec<u32> = predicates
                            .iter()
                            .flat_map(|p| p.held_predicates.iter().copied())
                            .collect::<HashSet<_>>()
                            .into_iter()
                            .collect();
                        s.sort_unstable();
                        s
                    };
                    let mut borrowed = if borrow_level > 0 {
                        let all_in_memory = borrow_level == 3 && (opts.borrow == 3 || opts.borrow == 8);
                        Some(get_borrowed_prep()?.instantiate(fs[0], all_in_memory, top_override))
                    } else {
                        None
                    };
                    if borrow_level == 3 {
                        let f = &fs[0];
                        if f.product_count > 0 && opts.borrow < 7 {
                            return Err(Refused("a matrix product in a function at level 3".into()));
                        }
                        let b = borrowed.as_mut().unwrap();
                        b.everything = true;
                        let pairs_set: HashSet<u32> = f.pairs.iter().copied().collect();
                        let mut sorted_uniform = f.uniform.clone();
                        sorted_uniform.sort_unstable();
                        b.assign_slots(&sorted_uniform, &pairs_set);
                        if opts.borrow >= 5 && b.r1_fixed {
                            b.room = true;
                            b.from_r1 = true;
                        }
                        let mut spans_by_q: std::collections::BTreeMap<u32, std::collections::BTreeSet<usize>> =
                            std::collections::BTreeMap::new();
                        for &(q, ref part) in &predicates[0].held_lives {
                            spans_by_q.entry(q).or_default().extend(part.iter().copied());
                        }
                        let mut held_at: HashMap<u32, u32> = HashMap::new();
                        for (&q, span_set) in &spans_by_q {
                            let span_sorted: Vec<usize> = span_set.iter().copied().collect();
                            if let Some(r) = b.kept(&span_sorted) {
                                held_at.insert(q, r);
                            } else if b.from_r1 {
                                let slot = b.slot_count();
                                b.predicate_slots.insert(q, slot);
                                b.frame_bytes = (4 * b.slot_count()).next_multiple_of(16);
                            } else {
                                return Err(Refused(format!(
                                    "no register holds nothing through the lives of uniform predicate UP{q}"
                                )));
                            }
                        }
                        let placed = placement(
                            b.top + 1,
                            &[],
                            &[],
                            false,
                            saves,
                            false,
                            0,
                            false,
                            None,
                            &[],
                            &[],
                        )?;
                        let slot = gslots[0];
                        let si = indices[0];
                        let place_map: HashMap<u32, u32> = b.homes.clone();
                        match translate_function(
                            dec70,
                            fs[0],
                            &predicates[0],
                            &place_map,
                            None,
                            placed.2,
                            None,
                            None,
                            None,
                            None,
                            placed.7,
                            None,
                            HashMap::new(),
                            HashMap::new(),
                            opts.aligned_rows,
                            false,
                            &held_at,
                            borrowed.as_mut(),
                        ) {
                            Ok(tr) => {
                                if say_borrows {
                                    let b = borrowed.as_ref().unwrap();
                                    let gu = if gave_up.is_empty() {
                                        "-".to_string()
                                    } else {
                                        gave_up.join("; ")
                                    };
                                    logs.borrow_mut().push(format!(
                                        "  registers\t{}\t{}\t{}\tlevel 3\tuniform {}\thomed {}\tin memory {}\tframe {}\tsaves {}\theld {}\tproducts {}\tgave up: {}",
                                        f.name,
                                        f.registers,
                                        tr.count,
                                        f.uniform.len(),
                                        b.homes.len(),
                                        b.slots.len(),
                                        b.frame_bytes,
                                        predicates[0].save_slots,
                                        held_at.len(),
                                        f.product_count,
                                        gu
                                    ));
                                }
                                let _ = slot;
                                return Ok(vec![(si, tr)]);
                            }
                            Err(refused) => return Err(refused),
                        }
                    }

                    let homes: HashMap<u32, u32> = borrowed.as_ref().map_or_else(HashMap::new, |b| b.homes.clone());
                    let homed_set: HashSet<u32> = homes.keys().copied().collect();
                    let uniform_for_placement: Vec<u32> = {
                        let mut all: Vec<u32> = fs.iter().flat_map(|f| f.uniform.iter().copied()).collect();
                        all.sort_unstable();
                        all.dedup();
                        all.into_iter().filter(|u| !homed_set.contains(u)).collect()
                    };
                    let pairs_for_placement: Vec<u32> = {
                        let mut all: Vec<u32> = fs.iter().flat_map(|f| f.pairs.iter().copied()).collect();
                        all.sort_unstable();
                        all.dedup();
                        all.into_iter().filter(|u| !homed_set.contains(u)).collect()
                    };
                    let placed = placement(
                        fs.iter().map(|f| f.highest).max().unwrap() + 1,
                        &uniform_for_placement,
                        &pairs_for_placement,
                        fs.iter().any(|f| f.needs_pool) && borrow_level != 2,
                        saves,
                        fs.iter().any(|f| f.needs_lane),
                        fs.iter().map(|f| f.scratch_need).max().unwrap_or(0),
                        fs.iter().zip(has_product_flags.iter()).any(|(_, &hp)| hp),
                        if rows.is_empty() { None } else { Some(&rows) },
                        &row_kept,
                        &held_preds,
                    )?;
                    let (
                        mut place,
                        pool,
                        save,
                        lane,
                        kept,
                        gathered,
                        scratch,
                        free,
                        rows_at_base,
                        kept_rows_base,
                        hold_at,
                    ) = placed;
                    for (&u, &r) in &homes {
                        place.insert(u, r);
                    }
                    let mut result = Vec::with_capacity(fs.len());
                    for (k, (&si, predicates)) in indices.iter().zip(&predicates).enumerate() {
                        let plan = &plans[k];
                        let mut func_rows_at = HashMap::new();
                        if let Some(plan) = plan {
                            for &rd in plan.rows().keys() {
                                let to = shares.get(&rd).copied().unwrap_or(rd);
                                if let Some(&at) = rows_at_base.get(&to) {
                                    func_rows_at.insert(rd, at);
                                }
                            }
                        }
                        let tr = translate_function(
                            dec70,
                            fs[k],
                            predicates,
                            &place,
                            pool,
                            save,
                            lane,
                            kept,
                            gathered,
                            scratch,
                            free,
                            plan.as_ref(),
                            func_rows_at,
                            kept_rows_base.clone(),
                            opts.aligned_rows,
                            has_product_flags[k],
                            &hold_at,
                            borrowed.as_mut(),
                        )?;
                        result.push((si, tr));
                    }
                    if say_borrows && fs.len() == 1 {
                        let f = &fs[0];
                        let count = result.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                        let n_homes = borrowed.as_ref().map_or(0, |b| b.homes.len());
                        let pool_count = if f.needs_pool && borrow_level != 2 { 2 } else { 0 };
                        let gu = if gave_up.is_empty() {
                            "-".to_string()
                        } else {
                            gave_up.join("; ")
                        };
                        logs.borrow_mut().push(format!(
                            "  registers\t{}\t{}\t{}\tlevel {}\tuniform {}\thomed {}\tpool {}\tsaves {}\theld {}\tproducts {}\tgave up: {}",
                            f.name,
                            f.registers,
                            count,
                            borrow_level,
                            f.uniform.len(),
                            n_homes,
                            pool_count,
                            saves,
                            held_preds.len(),
                            f.product_count,
                            gu
                        ));
                    }
                    Ok(result)
                };

                let alone = fs.len() == 1 && indices.iter().any(|si| entry_secs.contains(si));
                let has_pairs_to_drop = || paired.iter().any(Option::is_some);
                type Tried = Result<Vec<(usize, TransResult)>, Refused>;
                type AttemptFn<'a> = dyn FnMut(bool, u32, Option<u32>, &[String]) -> Tried + 'a;
                let empty_gave_up: Vec<String> = Vec::new();
                let usual = |attempt: &mut AttemptFn<'_>| {
                    let r = if opts.borrow > 0 && alone {
                        attempt(true, 2, None, &[])
                            .or_else(|_| attempt(true, 1, None, &[]))
                            .or_else(|_| attempt(true, 0, None, &[]))
                    } else {
                        attempt(true, 0, None, &[])
                    };
                    r.or_else(|refused| {
                        if has_pairs_to_drop() {
                            attempt(false, 0, None, &[])
                        } else {
                            Err(refused)
                        }
                    })
                };
                let plain = |attempt: &mut AttemptFn<'_>, gave_up: &[String]| {
                    attempt(true, 0, None, gave_up).or_else(|refused| {
                        if has_pairs_to_drop() {
                            attempt(false, 0, None, gave_up)
                        } else {
                            Err(refused)
                        }
                    })
                };

                let chosen: Tried = 'chain: {
                    if opts.borrow == 8 && alone {
                        let f = &fs[0];
                        let extra: u32 = if f.product_count > 0 { 16 } else { 0 };
                        let top_for_8 = if extra > 0 { Some(f.highest + extra) } else { None };
                        match attempt(false, 3, top_for_8, &empty_gave_up) {
                            Ok(d) => Ok(d),
                            Err(ref err) => {
                                logs.borrow_mut().push(format!("level 3 failed for {}: {}", fs[0].name, err.0));
                                usual(&mut attempt)
                            }
                        }
                    } else if matches!(opts.borrow, 7 | 9) && alone {
                        let mut gave_up: Vec<String> = Vec::new();
                        let plain_result = plain(&mut attempt, &gave_up);
                        let normal = match plain_result {
                            Ok(d) => Ok(d),
                            Err(original_why) if original_why.0 == "more registers than there are" => {
                                gave_up.push(original_why.0.clone());
                                let original_msg = original_why.0.clone();
                                let mut done = None;
                                for (do_p, bl) in [(true, 1u32), (true, 2), (false, 3)] {
                                    match attempt(do_p, bl, None, &gave_up) {
                                        Ok(d) => {
                                            done = Some(d);
                                            break;
                                        }
                                        Err(err) => {
                                            gave_up.push(err.0);
                                        }
                                    }
                                }
                                match done {
                                    Some(d) => Ok(d),
                                    None => {
                                        if say_borrows {
                                            logs.borrow_mut().push(format!(
                                                "  refused\t{}\t{}\tand with borrowed registers: {}",
                                                fs[0].name,
                                                original_msg,
                                                gave_up[1..].join("; ")
                                            ));
                                        }
                                        Err(Refused(original_msg))
                                    }
                                }
                            }
                            Err(err) => Err(err),
                        }?;
                        if opts.borrow == 9 {
                            let f = fs[0];
                            let info = info_sec[gslots[0]].map_or(&[][..], |ii| e.sections[ii].data.as_slice());
                            let limit = super::block_registers::budget(f.registers, info);
                            if normal[0].1.count > limit && !relocations.contains_key(&indices[0]) && f.tables.is_empty() {
                                let top = limit.saturating_sub(3).saturating_sub(saves);
                                for (pairs, level, bound) in [(true, 1, None), (true, 2, None), (false, 3, Some(top))] {
                                    if let Ok(candidate) = attempt(pairs, level, bound, &gave_up)
                                        && candidate[0].1.count <= limit
                                    {
                                        let old = &normal[0].1.new_of;
                                        let new = &candidate[0].1.new_of;
                                        if old.len() == new.len() && old.keys().all(|offset| new.contains_key(offset)) {
                                            break 'chain Ok(candidate);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        Ok(normal)
                    } else if opts.borrow == 6 && alone {
                        let mut gave_up: Vec<String> = Vec::new();
                        let plain_result = plain(&mut attempt, &gave_up);
                        match plain_result {
                            Ok(d) => Ok(d),
                            Err(original_why) if original_why.0 == "more registers than there are" => {
                                gave_up.push(original_why.0.clone());
                                let original_msg = original_why.0.clone();
                                match attempt(true, 1, None, &gave_up) {
                                    Ok(d) => Ok(d),
                                    Err(e1) => {
                                        gave_up.push(e1.0);
                                        match attempt(true, 2, None, &gave_up) {
                                            Ok(d) => Ok(d),
                                            Err(e2) => {
                                                gave_up.push(e2.0);
                                                match attempt(false, 3, None, &gave_up) {
                                                    Ok(d) => Ok(d),
                                                    Err(e3) => {
                                                        gave_up.push(e3.0);
                                                        if say_borrows {
                                                            logs.borrow_mut().push(format!(
                                                                "  refused\t{}\t{}\tand with borrowed registers: {}",
                                                                fs[0].name,
                                                                original_msg,
                                                                gave_up[1..].join("; ")
                                                            ));
                                                        }
                                                        Err(Refused(original_msg))
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(err) => Err(err),
                        }
                    } else if opts.borrow >= 4 && alone {
                        let f = &fs[0];
                        let budget = (f.registers.div_ceil(8) * 8).clamp(32, 255);
                        let top_for_l3 = Some(budget.saturating_sub(3).saturating_sub(saves));
                        struct Best {
                            count: u32,
                            level: u32,
                            result: Vec<(usize, TransResult)>,
                            within: bool,
                        }
                        let mut best: Option<Best> = None;
                        let mut first_why: Option<Refused> = None;
                        let mut gave_up: Vec<String> = Vec::new();
                        match plain(&mut attempt, &gave_up) {
                            Ok(d) => {
                                let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                                best = Some(Best {
                                    count,
                                    level: 0,
                                    result: d,
                                    within: count <= budget,
                                });
                            }
                            Err(refused) => {
                                gave_up.push(refused.0.clone());
                                first_why = Some(refused);
                            }
                        }
                        if !best.as_ref().is_some_and(|b| b.within) {
                            match attempt(true, 1, None, &gave_up) {
                                Ok(d) => {
                                    let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                                    if count <= budget {
                                        best = Some(Best {
                                            count,
                                            level: 1,
                                            result: d,
                                            within: true,
                                        });
                                    } else if best.is_none() || count < best.as_ref().unwrap().count {
                                        best = Some(Best {
                                            count,
                                            level: 1,
                                            result: d,
                                            within: false,
                                        });
                                    }
                                }
                                Err(refused) => {
                                    gave_up.push(refused.0.clone());
                                    if first_why.is_none() {
                                        first_why = Some(refused);
                                    }
                                }
                            }
                        }
                        if !best.as_ref().is_some_and(|b| b.within) {
                            match attempt(true, 2, None, &gave_up) {
                                Ok(d) => {
                                    let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                                    if count <= budget {
                                        best = Some(Best {
                                            count,
                                            level: 2,
                                            result: d,
                                            within: true,
                                        });
                                    } else if best.is_none() || count < best.as_ref().unwrap().count {
                                        best = Some(Best {
                                            count,
                                            level: 2,
                                            result: d,
                                            within: false,
                                        });
                                    }
                                }
                                Err(refused) => {
                                    gave_up.push(refused.0.clone());
                                    if first_why.is_none() {
                                        first_why = Some(refused);
                                    }
                                }
                            }
                        }
                        if !best.as_ref().is_some_and(|b| b.within) {
                            match attempt(false, 3, top_for_l3, &gave_up) {
                                Ok(d) => {
                                    let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                                    if count <= budget {
                                        best = Some(Best {
                                            count,
                                            level: 3,
                                            result: d,
                                            within: true,
                                        });
                                    } else if best.is_none() || count < best.as_ref().unwrap().count {
                                        best = Some(Best {
                                            count,
                                            level: 3,
                                            result: d,
                                            within: false,
                                        });
                                    }
                                }
                                Err(refused) => {
                                    gave_up.push(refused.0.clone());
                                    if first_why.is_none() {
                                        first_why = Some(refused);
                                    }
                                }
                            }
                        }
                        match best {
                            None => Err(first_why.unwrap()),
                            Some(mut b) => {
                                if !b.within && say_borrows {
                                    b.result = match b.level {
                                        0 => plain(&mut attempt, &gave_up)?,
                                        1 => attempt(true, 1, None, &gave_up)?,
                                        2 => attempt(true, 2, None, &gave_up)?,
                                        3 => attempt(false, 3, top_for_l3, &gave_up)?,
                                        _ => unreachable!(),
                                    };
                                }
                                if say_borrows {
                                    logs.borrow_mut().push(format!(
                                        "  budget\t{}\t{}\t{}\t{}\tlevel {}\t{}",
                                        f.name,
                                        f.registers,
                                        b.count,
                                        budget,
                                        b.level,
                                        if b.within { "within" } else { "over" }
                                    ));
                                }
                                Ok(b.result)
                            }
                        }
                    } else if opts.borrow >= 3 && alone {
                        match attempt(false, 3, None, &empty_gave_up) {
                            Ok(d) => Ok(d),
                            Err(ref err) => {
                                logs.borrow_mut().push(format!("level 3 failed for {}: {}", fs[0].name, err.0));
                                usual(&mut attempt)
                            }
                        }
                    } else if opts.borrow >= 2 && alone {
                        match usual(&mut attempt) {
                            Ok(d) => Ok(d),
                            Err(original_why) => match attempt(false, 3, None, &empty_gave_up) {
                                Ok(d) => Ok(d),
                                Err(also) => {
                                    if say_borrows {
                                        logs.borrow_mut().push(format!(
                                            "  refused\t{}\t{}\tat level 3: {}",
                                            fs[0].name, original_why.0, also.0
                                        ));
                                    }
                                    Err(original_why)
                                }
                            },
                        }
                    } else {
                        usual(&mut attempt)
                    }
                };

                let Some((limit, threads)) = required else {
                    return chosen;
                };
                let count = |d: &[(usize, TransResult)]| d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                let chosen = chosen?;
                if count(&chosen) <= limit {
                    return Ok(chosen);
                }
                if opts.borrow > 0 && alone {
                    let top = limit.saturating_sub(3).saturating_sub(saves);
                    for (pairs, level, bound) in [(true, 1, None), (true, 2, None), (false, 3, Some(top))] {
                        if let Ok(candidate) = attempt(pairs, level, bound, &empty_gave_up)
                            && count(&candidate) <= limit
                        {
                            return Ok(candidate);
                        }
                    }
                }
                Err(Refused(format!(
                    "more registers than its required block of {threads} threads allows ({limit})"
                )))
            });

            let mut done = match attempt_result {
                Ok(d) => d,
                Err(refused) => {
                    return Ok(GroupOutput {
                        outcome: Err(refused.0),
                        logs: logs.into_inner(),
                    });
                }
            };

            let count = done.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
            if count > declared && !indices.iter().any(|si| entry_secs.contains(si)) {
                return Ok(GroupOutput {
                    outcome: Err(
                        "called from code that is not in the file, and the translation needs registers above its own"
                            .into(),
                    ),
                    logs: logs.into_inner(),
                });
            }

            for (idx, (si, tr)) in done.iter_mut().enumerate() {
                if let Some(list) = relocations.get(si) {
                    match follow(fs[idx], tr, list) {
                        Ok(moved) => tr.moved = moved,
                        Err(refused) => {
                            return Ok(GroupOutput {
                                outcome: Err(refused.0),
                                logs: logs.into_inner(),
                            });
                        }
                    }
                }
            }

            for (idx, (si, tr)) in done.iter().enumerate() {
                let slot = gslots[idx];
                if let Some(ii) = info_sec[slot]
                    && let Err(refused) = check_attributes(&e.sections[ii].data, &tr.new_of)
                {
                    return Ok(GroupOutput {
                        outcome: Err(refused.0),
                        logs: logs.into_inner(),
                    });
                }
                let bank = const2_sec[slot];
                let found = |base: usize, places: &[u64]| {
                    bank.is_some_and(|bi| {
                        let data = &e.sections[bi].data;
                        base + 4 * places.len() <= data.len()
                            && places.iter().enumerate().all(|(j, &place)| {
                                let at = base + 4 * j;
                                u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as u64 == place
                            })
                    })
                };
                if tr.relocate.iter().any(|(base, places)| !found(*base as usize, places)) {
                    let _ = si;
                    return Ok(GroupOutput {
                        outcome: Err("a jump table is not in the constants where its load reads it".into()),
                        logs: logs.into_inner(),
                    });
                }
            }

            for (_, tr) in &mut done {
                tr.count = count;
            }
            Ok(GroupOutput {
                outcome: Ok(done),
                logs: logs.into_inner(),
            })
        },
    )
    .expect("group translation returns Result inside GroupOutput");

    // Reassemble group outputs deterministically (replay logs sorted by group min-slot to avoid interleaving).
    let mut results: Vec<Option<TransResult>> = (0..n_texts).map(|_| None).collect();
    let mut indexed_outputs: Vec<(usize, Vec<usize>, GroupOutput)> = active_groups
        .into_iter()
        .zip(group_outputs)
        .map(|((_, gslots), out)| (gslots[0], gslots, out))
        .collect();
    indexed_outputs.sort_by_key(|(min_slot, _, _)| *min_slot);
    for (_, gslots, gout) in indexed_outputs {
        for msg in gout.logs {
            eprintln!("{msg}");
        }
        match gout.outcome {
            Ok(done) => {
                for (si, tr) in done {
                    let slot = text_slot[&si];
                    results[slot] = Some(tr);
                }
            }
            Err(why) => {
                for slot in gslots {
                    report_reasons[slot] = Some(why.clone());
                }
            }
        }
    }

    // Build the listing, and what the validator is told beside it.
    let mut listing: Option<Vec<MapEntry>> = if want_map { Some(Vec::new()) } else { None };
    let mut lives: Option<Vec<LifeEntry>> = if want_map { Some(Vec::new()) } else { None };

    // Update the container.
    let mut counts: FastMap<u32, u32> = FastMap::default();
    let mut frames: FastMap<u32, u32> = FastMap::default();

    for (slot, &si) in texts.iter().enumerate() {
        let name = e.sections[si].name[".text.".len()..].to_string();
        if let Some(ref tr) = results[slot] {
            if let Some(ref mut map) = listing {
                for (k, item) in tr.items.iter().enumerate() {
                    map.push((
                        name.clone(),
                        (16 * k) as u64,
                        (16 * item.origin) as u64,
                        item.what_str().to_string(),
                    ));
                }
            }
            if let Some(ref mut lives) = lives {
                for (first, second, how, offsets) in &tr.lives {
                    lives.push((name.clone(), *first, *second, how, offsets.clone()));
                }
            }
            e.sections[si].data = tr.code.clone();
            e.sections[si].info = (tr.count << 24) | (e.sections[si].info & 0xffffff);
            counts.insert(e.sections[si].info & 0xffffff, tr.count);
            if tr.frame_bytes > 0 {
                frames.insert(e.sections[si].info & 0xffffff, tr.frame_bytes);
            }
            if let Some(ii) = info_sec[slot] {
                e.sections[ii].data = rewrite_attributes(&e.sections[ii].data, &tr.new_of);
            }
            if let Some(list) = relocations.get(&si) {
                for (r, now) in list.iter().zip(&tr.moved) {
                    e.sections[r.section].data[r.at..r.at + 8].copy_from_slice(&now.to_le_bytes());
                }
            }
            if let Some(bi) = const2_sec[slot] {
                for (base, places) in &tr.relocate {
                    for (j, place) in places.iter().enumerate() {
                        let at = *base as usize + 4 * j;
                        e.sections[bi].data[at..at + 4].copy_from_slice(&(tr.new_of[place] as u32).to_le_bytes());
                    }
                }
            }
            for &sym_idx in &symbols_by_text[slot] {
                let sym = &mut symbols[sym_idx];
                if let Some(&new_val) = tr.new_of.get(&sym.value) {
                    sym.value = new_val;
                } else if sym.value != 0 {
                    report_reasons[slot] = Some(format!("a symbol at 0x{:x}, which is not an instruction", sym.value));
                }
                if sym.info & 0xf == 2 {
                    sym.size = tr.code.len() as u64 - if arch == 80 { sym.value } else { 0 };
                }
            }
        } else {
            let mut exits = Vec::new();
            if let Some(ii) = info_sec[slot] {
                exits = find_exits(&e.sections[ii].data);
                e.sections[ii].data = strip_attributes(&e.sections[ii].data);
            }
            let nop_bytes = NOP.to_le_bytes();
            let n_insns = e.sections[si].data.len() / 16;
            let mut code = vec![0u8; n_insns * 16];
            for i in 0..n_insns {
                code[i * 16..(i + 1) * 16].copy_from_slice(&nop_bytes[..16]);
            }
            if let Some(leave) = assemble::assemble("exit_", &[]) {
                let leave = with_stall(leave, 5, false) | (16u128 << USCHED);
                let lb = leave.to_le_bytes();
                for &at in if exits.is_empty() {
                    &[0u32][..]
                } else {
                    exits.as_slice()
                } {
                    let off = at as usize;
                    if off + 16 <= code.len() {
                        code[off..off + 16].copy_from_slice(&lb[..16]);
                    }
                }
            }
            for (at, word) in [(0usize, TRAP), (16, STAY)] {
                if at + 16 <= code.len() && !exits.contains(&(at as u32)) {
                    code[at..at + 16].copy_from_slice(&word.to_le_bytes());
                }
            }
            e.sections[si].data = code;
        }
    }

    e.set_symbols(&symbols);

    // Update global info
    if let Some(ii) = nv_info_sec {
        let d = &mut e.sections[ii].data;
        let mut raised: FastSet<(u32, u8)> = FastSet::default();
        let mut pos = 0;
        while pos + 4 <= d.len() {
            let fmt = d[pos];
            let attr = d[pos + 1];
            let val = u16::from_le_bytes(d[pos + 2..pos + 4].try_into().unwrap()) as usize;
            if fmt == 4 && attr == 0x2f && pos + 12 <= d.len() {
                let symbol = u32::from_le_bytes(d[pos + 4..pos + 8].try_into().unwrap());
                if let Some(&count) = counts.get(&symbol) {
                    d[pos + 8..pos + 12].copy_from_slice(&count.to_le_bytes());
                }
            }
            if fmt == 4 && (attr == 0x11 || attr == 0x12 || attr == 0x23) && pos + 12 <= d.len() {
                let symbol = u32::from_le_bytes(d[pos + 4..pos + 8].try_into().unwrap());
                if let Some(&frame) = frames.get(&symbol) {
                    let old = u32::from_le_bytes(d[pos + 8..pos + 12].try_into().unwrap());
                    d[pos + 8..pos + 12].copy_from_slice(&(old + frame).to_le_bytes());
                    raised.insert((symbol, attr));
                }
            }
            pos += if fmt == 4 { 4 + val } else { 4 };
        }
        for &symbol in frames.keys() {
            if !raised.contains(&(symbol, 0x11)) || !raised.contains(&(symbol, 0x12)) {
                return Err("a frame was added to a function that has no frame attributes to raise".into());
            }
        }
    }

    // Retag as sm_70
    if e.header[7] == 0x41 {
        e.header[7] = 0x33;
        e.header[8] = 7;
        e.header[0x14..0x18].copy_from_slice(&0x80u32.to_le_bytes());
        let elf_flags = (u32::from(arch) << 16) | 0x0500 | 70;
        e.header[0x30..0x34].copy_from_slice(&elf_flags.to_le_bytes());
        for s in &mut e.sections {
            if s.name == ".note.nv.tkinfo" || s.name == ".note.nv.cuinfo" {
                s.stype = 0;
                s.data.clear();
                s.size = 0;
            } else if s.name.starts_with(".nv.info") {
                s.data = strip_attrs_by_kind(&s.data, &[0x5f, 0x66]);
            }
        }
    } else {
        e.header[0x30] = 70;
    }

    let ordered_report: Vec<FunctionReport> = texts
        .iter()
        .enumerate()
        .map(|(slot, &si)| {
            let name = e.sections[si].name[".text.".len()..].to_string();
            let refused = report_reasons[slot].clone();
            FunctionReport {
                name,
                paired: refused.is_none() && results[slot].as_ref().is_some_and(|tr| tr.paired),
                refused,
                before: before[slot],
                after: e.sections[si].data.len() / 16,
            }
        })
        .collect();

    Ok(TranslateResult {
        bytes: e.write(),
        report: ordered_report,
        map: listing,
        lives,
    })
}
