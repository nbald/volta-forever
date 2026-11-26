//! First-pass analysis of a function body: which operands of tensor-core products are loaded
//! from shared memory in the layout the Volta sequence takes, which registers are read in
//! Turing layout, and which products form chains.

use std::collections::{HashMap, HashSet};

use crate::scan::{self, Kind};

use super::parse::{Tensor, is_branch, is_tensor, parse};
use super::types::{Declarations, all_ones, first_operand, integer};
use super::{Options, Unsupported, may_change_shared};

/// What one `ldmatrix` loads into a register that nothing else writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Origin {
    /// Index of the instruction among the items of the body.
    pub(super) statement: usize,
    pub(super) matrix: u32,
    pub(super) transposed: bool,
}

/// What a first pass over the body decides: which operands of the products are loaded from shared
/// memory in the layout the Volta sequence takes, by the `ldmatrix` that loads them in Turing layout,
/// and which registers are read in Turing layout at all.
///
/// A register qualifies when one `ldmatrix` is the only statement of the function that writes it and
/// the body declares it once: whatever the control flow, a product that reads it reads what the last
/// execution of that `ldmatrix` loaded, and so does it read the registers loaded beside it.
#[derive(Debug, Default)]
pub(super) struct Plan<'a> {
    pub(super) origins: HashMap<&'a str, Origin>,
    /// A operands of FP16 products (register pairs) to load as rows.
    pub(super) rows: HashSet<(&'a str, &'a str)>,
    /// B operands of FP16 products to load as columns, from a plain `ldmatrix`.
    pub(super) cols: HashSet<&'a str>,
    /// B operands of FP16 products to load in the layout of `Form::Columns`, from `ldmatrix.trans`.
    pub(super) columns: HashSet<&'a str>,
    /// A operands of integer products whose row every product loads from shared memory.
    pub(super) row: HashSet<&'a str>,
    /// Highly-reused (>= 3 products) A operands in `row` loaded once with
    /// `ld.volatile.shared.v4.u32` and reused via `Form::Row` within the per-segment budget.
    pub(super) cached_row: HashSet<&'a str>,
    /// Registers that some statement reads in Turing layout.
    pub(super) plain: HashSet<&'a str>,
    /// Registers that hold all ones wherever they are read: the only statement of the function that
    /// writes them is an unguarded `mov` of that constant, and the body declares them once.
    /// Compilers put the member mask of a warp instruction in such a register, often in an earlier
    /// block than the instruction.
    pub(super) ones: HashSet<&'a str>,
    /// Labels that some statement of the function names (a branch, a list of branch targets) or
    /// that a line directive names: the only labels where control can come from elsewhere than
    /// the statement before. A label that nothing names (compilers put one before many statements,
    /// for their line tables) is reached by falling through to it alone, and what was known of
    /// the registers before it still holds after it.
    pub(super) targets: HashSet<&'a str>,
    /// Unshadowed 16-bit registers with one unconditional constant definition.
    pub(super) constants16: HashMap<&'a str, u32>,
    /// 32-bit registers that hold one of two numbers wherever they are read: the only statement of
    /// the function that writes them is an unguarded `selp` of two integers, and the body declares
    /// them once. Triton says so how much of the source of a `cp.async` is read
    /// (`selp.b32 %r47, 16, 0, %p2;`).
    pub(super) choices: HashMap<&'a str, [u32; 2]>,
    /// Products of packed BF16 FMAs whose sums are exactly representable.
    pub(super) exact_bf16_sums: HashMap<usize, super::known_bits::ExactSum<'a>>,
    /// Registers written by `movmatrix`: their layout is transposed, not suitable for the chain
    /// form's shuffle indices.
    pub(super) transposed: HashSet<&'a str>,
    /// Item indices of `mma.m16n8k8` f32 products that are in a chain: a run of products on the same
    /// accumulator registers in straight-line code, long enough for the single-instruction
    /// arrangement to pay off. The chain form uses one Volta `mma.m8n8k4` per Turing product (not
    /// two), with the accumulator split across octets and combined at the end of the chain.
    pub(super) chain_products: HashSet<usize>,
    /// A operands (register pairs) of products of chains, loaded by their `ldmatrix` in the layout
    /// of the chain's single instruction (2 registers).
    pub(super) chain_a: HashSet<(&'a str, &'a str)>,
    /// B operands of products of chains, likewise.
    pub(super) chain_b: HashSet<&'a str>,
    /// Pairs of FP16 products computed together, and what their accumulators need (`pairs`).
    pub(super) pairs: super::pairs::Pairs<'a>,
}

/// Collect labels that appear in statement operands or line directives, avoiding allocation when
/// the body contains no labels.
fn branch_targets<'a>(text: &'a str, items: &[scan::Item]) -> HashSet<&'a str> {
    if !items.iter().any(|item| item.kind == Kind::Label) {
        return HashSet::new();
    }
    let mut named: HashSet<&str> = HashSet::new();
    let mut labels: Vec<&str> = Vec::new();
    for item in items {
        match &item.kind {
            Kind::Label => labels.extend(scan::names(&text[item.range.clone()]).next()),
            Kind::Statement { operands, .. } => named.extend(scan::names(&text[operands.clone()])),
            Kind::Line => named.extend(scan::names(&text[item.range.clone()])),
            Kind::Open | Kind::Close => {}
        }
    }
    labels.into_iter().filter(|label| named.contains(label)).collect()
}

impl<'a> Plan<'a> {
    /// The FP16 `m16n8k8` product of A (two registers) and B is split in the halves of k: both
    /// registers of A come from one plain `ldmatrix` and B from a plain `ldmatrix`.
    pub(super) fn in_halves(&self, a: &[&str], b: &str) -> bool {
        match (self.origins.get(a[0]), self.origins.get(a[1]), self.origins.get(b)) {
            (Some(first), Some(second), Some(b)) => {
                first.statement == second.statement && !first.transposed && !b.transposed
            }
            _ => false,
        }
    }

    pub(super) fn loaded_transposed(&self, register: &str) -> bool {
        self.origins.get(register).is_some_and(|origin| origin.transposed)
    }

    pub(super) fn new(text: &'a str, items: &[scan::Item], options: Options) -> Plan<'a> {
        const TRIGGERS: [&str; 7] = [
            "mma.",
            "ldmatrix.",
            "movmatrix.",
            "stmatrix.",
            "cp.async.",
            "redux.sync",
            "fma.rn.bf16x2",
        ];
        if !TRIGGERS.iter().any(|needle| text.contains(needle)) {
            return Plan {
                targets: branch_targets(text, items),
                ..Plan::default()
            };
        }
        let statements = || {
            items.iter().enumerate().filter_map(|(index, item)| match &item.kind {
                Kind::Statement {
                    guarded,
                    opcode,
                    operands,
                } => Some((index, *guarded, &text[opcode.clone()], &text[operands.clone()])),
                _ => None,
            })
        };
        // Who writes what, and what is declared where.
        let mut writes: HashMap<&str, u32> = HashMap::new();
        let mut declarations = Declarations::default();
        let mut nested: Vec<(String, bool)> = Vec::new();
        let mut loads: Vec<(&str, Origin)> = Vec::new();
        let mut set_to_ones: Vec<&str> = Vec::new();
        let mut constants16 = Vec::new();
        let mut constants32 = Vec::new();
        let mut choices = Vec::new();
        let mut literals: Vec<&str> = Vec::new();
        let packed_fma = text.contains("fma.rn.bf16x2");
        let mut plan = Plan::default();
        let mut depth = 0usize;
        for (index, item) in items.iter().enumerate() {
            let (guarded, opcode, operands) = match &item.kind {
                Kind::Open => {
                    depth += 1;
                    continue;
                }
                Kind::Close => {
                    depth = depth.saturating_sub(1);
                    continue;
                }
                Kind::Statement {
                    guarded,
                    opcode,
                    operands,
                } => (*guarded, &text[opcode.clone()], &text[operands.clone()]),
                _ => continue,
            };
            if opcode == ".reg" {
                let names = declarations.declare(operands);
                if depth > 0 {
                    nested.extend(names);
                }
                continue;
            }
            for name in scan::names(first_operand(operands)) {
                *writes.entry(name).or_default() += 1;
            }
            if !guarded
                && matches!(
                    opcode,
                    "mov.b32"
                        | "mov.u32"
                        | "mov.s32"
                        | "mov.f32"
                        | "mov.b16"
                        | "mov.u16"
                        | "mov.s16"
                        | "selp.b32"
                        | "selp.u32"
                        | "selp.s32"
                )
            {
                let parts = scan::split_operands(operands);
                match (opcode, parts.as_slice()) {
                    ("mov.b32" | "mov.u32" | "mov.s32" | "mov.f32", [register, value]) => {
                        if let Some(name) = scan::register(register) {
                            if opcode != "mov.f32" {
                                if all_ones(value) {
                                    set_to_ones.push(name);
                                }
                                if packed_fma && let Some(bits) = super::known_bits::word_literal(value) {
                                    constants32.push((name, bits));
                                }
                            }
                            if super::pairs::literal(value) {
                                literals.push(name);
                            }
                        }
                    }
                    ("mov.b16" | "mov.u16" | "mov.s16", [register, literal]) => {
                        if let Some(name) = scan::register(register)
                            && let Some(value) = integer(literal)
                            && u16::try_from(value).is_ok()
                        {
                            constants16.push((name, value));
                        }
                    }
                    ("selp.b32" | "selp.u32" | "selp.s32", [register, first, second, _]) => {
                        if let Some(name) = scan::register(register)
                            && let (Some(first), Some(second)) = (integer(first), integer(second))
                        {
                            choices.push((name, [first, second]));
                        }
                    }
                    _ => {}
                }
            }
            if guarded || !is_tensor(opcode) {
                continue;
            }
            if let Ok(Tensor::Load { transposed, r, .. }) = parse(opcode, operands, &plan.constants16) {
                for (matrix, name) in r.into_iter().enumerate() {
                    loads.push((
                        name,
                        Origin {
                            statement: index,
                            matrix: matrix as u32,
                            transposed,
                        },
                    ));
                }
            }
        }
        // Items so far that could change shared memory or be reached from elsewhere: between an
        // `ldmatrix` and a product with the same count, the memory the first read is what the second
        // would read.
        let mut changes = Vec::with_capacity(items.len());
        let mut so_far = 0u32;
        for item in items {
            changes.push(so_far);
            so_far += u32::from(match &item.kind {
                Kind::Label => true,
                Kind::Statement { opcode, .. } => may_change_shared(&text[opcode.clone()]),
                _ => false,
            });
        }
        let shadowed = |name: &str| {
            nested
                .iter()
                .any(|(declared, family)| match name.strip_prefix(declared.as_str()) {
                    Some(rest) if *family => !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()),
                    Some(rest) => rest.is_empty(),
                    None => false,
                })
        };
        let written_once =
            |name: &str| writes.get(name) == Some(&1) && declarations.width(name).is_some() && !shadowed(name);
        for (name, origin) in loads {
            if written_once(name) {
                plan.origins.insert(name, origin);
            }
        }
        plan.ones = set_to_ones.into_iter().filter(|name| written_once(name)).collect();
        plan.targets = branch_targets(text, items);
        plan.constants16 = constants16
            .into_iter()
            .filter(|(name, _)| written_once(name) && declarations.width(name) == Some(16))
            .collect();
        plan.choices = choices
            .into_iter()
            .filter(|(name, _)| written_once(name) && declarations.width(name) == Some(32))
            .collect();
        if packed_fma {
            // Only unshadowed registers with one unconditional write retain constants across labels.
            let constants = constants32
                .into_iter()
                .filter(|(name, _)| written_once(name) && declarations.width(name) == Some(32))
                .collect();
            let shadowable: Vec<_> = writes.keys().copied().filter(|name| shadowed(name)).collect();
            plan.exact_bf16_sums = super::known_bits::exact_bf16_sums(text, items, &constants, &shadowable);
        }
        // Who reads what, and how.
        let mut gathered: HashSet<&str> = HashSet::new();
        let mut row_uses: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut products: Vec<(usize, Vec<&str>, Vec<&str>)> = Vec::new();
        for (index, guarded, opcode, operands) in statements() {
            if opcode == ".reg" {
                continue;
            }
            let tensor = if !guarded && is_tensor(opcode) {
                parse(opcode, operands, &plan.constants16).ok()
            } else {
                None
            };
            match tensor {
                Some(Tensor::Product { a, b, c, .. }) => {
                    plan.plain.extend(c);
                    // (How its operands are loaded depends on whether it is a product of a chain,
                    // which is known below.)
                    products.push((index, a, b));
                }
                Some(Tensor::Bytes { a, b, c, .. }) => {
                    plan.plain.extend(c);
                    plan.plain.insert(b);
                    let unchanged = plan.origins.get(a).is_some_and(|origin| {
                        !origin.transposed && origin.statement < index && changes[origin.statement] == changes[index]
                    });
                    if options.aligned_rows && unchanged {
                        plan.row.insert(a);
                        row_uses.entry(a).or_default().push(index);
                    } else {
                        gathered.insert(a);
                    }
                }
                Some(Tensor::AmpereBytes { a, b, c, .. }) => {
                    plan.plain.extend(c);
                    for b_reg in b {
                        plan.plain.insert(b_reg);
                    }
                    for a_reg in a {
                        let unchanged = plan.origins.get(a_reg).is_some_and(|origin| {
                            !origin.transposed
                                && origin.statement < index
                                && changes[origin.statement] == changes[index]
                        });
                        if options.aligned_rows && unchanged {
                            plan.row.insert(a_reg);
                            row_uses.entry(a_reg).or_default().push(index);
                        } else {
                            gathered.insert(a_reg);
                        }
                    }
                }
                // Read as the program has them: decoded where they are, or gathered from the lanes.
                Some(Tensor::Fp8Product { a, b, c, .. } | Tensor::Sums { a, b, c, .. }) => {
                    plan.plain.extend(a);
                    plan.plain.extend(b);
                    plan.plain.extend(c);
                }
                Some(Tensor::BlockScaled {
                    a,
                    b,
                    c,
                    scale_a,
                    scale_b,
                    ..
                }) => {
                    plan.plain.extend(a);
                    plan.plain.extend(b);
                    plan.plain.extend(c);
                    plan.plain.insert(scale_a);
                    plan.plain.insert(scale_b);
                }
                Some(Tensor::Load { base, .. }) => {
                    plan.plain.insert(base);
                }
                Some(Tensor::Store { src, base, .. }) => {
                    plan.plain.extend(src);
                    plan.plain.insert(base);
                }
                Some(Tensor::Transpose { d, a }) => {
                    plan.plain.insert(a);
                    plan.transposed.insert(d);
                }
                // Never parsed here: they are not tensor-core instructions.
                Some(Tensor::Copy { .. } | Tensor::Copied | Tensor::Pack { .. } | Tensor::Bound { .. }) => {}
                None => plan.plain.extend(scan::names(operands)),
            }
        }
        // One product that must gather it, and the register is gathered by all of them.
        plan.row.retain(|a| !gathered.contains(a));
        plan.plain.extend(gathered);

        // Promote highly-reused (>= 3 products) `plan.row` registers to volatile-once `Form::Row`
        // reuse up to 8 concurrently live rows per segment (<= 32 live registers) so dense MMQ
        // tiles (e.g. 2x4, 4x4) eliminate redundant SMEM reloads across rolled and unrolled turns
        // without hoisting spills or regressing 2-use / launch-bounds-capped kernels.
        const ROW_CACHE_MIN_USES: usize = 3;
        const ROW_CACHE_LIVE_BUDGET: usize = 8;
        struct RowUse<'r> {
            uses: usize,
            first: usize,
            last: usize,
            statement: usize,
            matrix: u32,
            reg: &'r str,
        }
        let mut by_segment: HashMap<u32, Vec<RowUse<'a>>> = HashMap::new();
        for &a_reg in &plan.row {
            if let Some(uses) = row_uses.get(a_reg)
                && uses.len() >= ROW_CACHE_MIN_USES
                && let Some(origin) = plan.origins.get(a_reg)
            {
                let (first, last) = (uses[0], *uses.last().unwrap());
                by_segment.entry(changes[origin.statement]).or_default().push(RowUse {
                    uses: uses.len(),
                    first,
                    last,
                    statement: origin.statement,
                    matrix: origin.matrix,
                    reg: a_reg,
                });
            }
        }
        for (_, mut entries) in by_segment {
            entries.sort_by(|x, y| {
                y.uses
                    .cmp(&x.uses)
                    .then_with(|| (x.last - x.first).cmp(&(y.last - y.first)))
                    .then_with(|| x.statement.cmp(&y.statement))
                    .then_with(|| x.matrix.cmp(&y.matrix))
            });
            let mut active: Vec<(usize, usize)> = Vec::new();
            for item in entries {
                let fits = std::iter::once(item.first)
                    .chain(
                        active
                            .iter()
                            .map(|&(s, _)| s)
                            .filter(|&s| item.first <= s && s <= item.last),
                    )
                    .all(|pt| active.iter().filter(|&&(s, e)| s <= pt && pt <= e).count() < ROW_CACHE_LIVE_BUDGET);
                if fits {
                    active.push((item.first, item.last));
                    plan.cached_row.insert(item.reg);
                }
            }
        }

        // Chain detection: find runs of f32 mma.m16n8k8 products on the same accumulator in
        // straight-line code, whose A and B operands come from plain ldmatrix (in-halves form).
        // The minimum chain length for the single-instruction arrangement to pay off: the chain end
        // costs ~44 instructions (combination + extraction), and each product saves ~4 HMMA steps
        // compared to the standard form. At 4 products the saving is ~16 HMMA steps.
        const CHAIN_MIN: usize = 4;
        struct ChainCandidate {
            indices: Vec<usize>,
        }
        // Multiple chains may be open at once: products of different accumulators interleaved.
        let mut open: HashMap<Vec<String>, ChainCandidate> = HashMap::new();
        let mut chains: Vec<ChainCandidate> = Vec::new();
        let flush_all = |open: &mut HashMap<Vec<String>, ChainCandidate>, chains: &mut Vec<ChainCandidate>| {
            for (_, c) in open.drain() {
                if c.indices.len() >= CHAIN_MIN {
                    chains.push(c);
                }
            }
        };
        // The chains whose accumulator the statement names end there, but the one it goes on with
        // (`own`). Names are compared whole: `%f1` is not named by a statement that names `%f10`.
        let close_named = |open: &mut HashMap<Vec<String>, ChainCandidate>,
                           chains: &mut Vec<ChainCandidate>,
                           operands: &str,
                           own: Option<&Vec<String>>| {
            let named: HashSet<&str> = scan::names(operands).collect();
            let touched: Vec<Vec<String>> = open
                .keys()
                .filter(|d| Some(*d) != own && d.iter().any(|name| named.contains(name.as_str())))
                .cloned()
                .collect();
            for d in touched {
                if let Some(c) = open.remove(&d).filter(|c| c.indices.len() >= CHAIN_MIN) {
                    chains.push(c);
                }
            }
        };
        for (index, item) in items.iter().enumerate() {
            match &item.kind {
                // A label, a branch (taken or not, under a predicate or not) and the bounds of a
                // scope end every chain: the accumulator must be in the program's layout wherever
                // the lanes may go, and a name may be another register in another scope.
                Kind::Label | Kind::Open | Kind::Close => flush_all(&mut open, &mut chains),
                Kind::Statement { opcode, .. } if is_branch(&text[opcode.clone()]) => flush_all(&mut open, &mut chains),
                Kind::Statement {
                    guarded,
                    opcode,
                    operands,
                } => {
                    let opcode_s = &text[opcode.clone()];
                    let operands_s = &text[operands.clone()];
                    // The accumulator of the chain this statement goes on with, or begins.
                    let product = if !guarded && is_tensor(opcode_s) {
                        parse(opcode_s, operands_s, &plan.constants16).ok()
                    } else {
                        None
                    };
                    let own = match product {
                        Some(Tensor::Product {
                            half: false,
                            d,
                            a,
                            b,
                            c,
                        }) if d == c
                            && d.len() == 4
                            && a.len() == 2
                            && b.len() == 1
                            && !plan.transposed.contains(b[0])
                            && !plan.loaded_transposed(b[0]) =>
                        {
                            Some(d.iter().map(|name| name.to_string()).collect::<Vec<String>>())
                        }
                        _ => None,
                    };
                    // Any statement, under a predicate or not, and a product of another chain too.
                    close_named(&mut open, &mut chains, operands_s, own.as_ref());
                    if let Some(d) = own {
                        open.entry(d)
                            .or_insert_with(|| ChainCandidate { indices: Vec::new() })
                            .indices
                            .push(index);
                    }
                }
                Kind::Line => {}
            }
        }
        flush_all(&mut open, &mut chains);
        for c in &chains {
            plan.chain_products.extend(c.indices.iter().copied());
        }
        // Pairs of products (`pairs`): their operands are loaded as the pairs take them, and their
        // products are not computed by themselves.
        if options.pairs && options.aligned_rows {
            let clean = |name: &str| declarations.width(name) == Some(32) && !shadowed(name);
            let uniform: HashSet<&str> = literals.into_iter().filter(|name| written_once(name)).collect();
            plan.pairs = super::pairs::plan(text, items, &plan, &clean, &uniform);
        }
        // How the operands of each FP16 product are loaded. Split in the halves of k (both
        // registers of A from one plain `ldmatrix`, B from a plain one): by their `ldmatrix`, in
        // the layout the product takes them in, which for a product of a chain is the one of its
        // single instruction (rows where the manual wants them) and for another the rows and
        // columns of its two. A register is loaded in the layout of the program as well only when
        // something reads it there: a product of a chain whose rows may be anywhere, a product
        // that is not split so, a statement that is not a product.
        for (index, a, b) in products {
            if plan.pairs.first.contains(&index) || plan.pairs.second.contains_key(&index) {
                continue;
            }
            let chained = plan.chain_products.contains(&index);
            for (a, b) in a.chunks(2).zip(b) {
                if !plan.in_halves(a, b) {
                    plan.plain.extend(a);
                    if plan.loaded_transposed(b) {
                        plan.columns.insert(b);
                    } else {
                        plan.plain.insert(b);
                    }
                } else if !chained {
                    plan.rows.insert((a[0], a[1]));
                    plan.cols.insert(b);
                } else if options.aligned_rows {
                    // The direct chain load selects the next matrix for the upper
                    // rows. Other legal pairings gather the actual A registers.
                    if plan.origins[a[1]].matrix == plan.origins[a[0]].matrix + 1 {
                        plan.chain_a.insert((a[0], a[1]));
                    } else {
                        plan.plain.extend(a);
                    }
                    plan.chain_b.insert(b);
                } else {
                    plan.plain.extend(a);
                    plan.plain.insert(b);
                }
            }
        }

        plan
    }
}

/// The plan promised an operand loaded by its `ldmatrix` and it is not there: a defect of this
/// module. The function is refused rather than given code that reads a register nothing wrote.
pub(super) fn not_loaded(register: &str) -> Unsupported {
    Unsupported(format!(
        "{register}: operand not loaded as planned (a defect of the rewriting)"
    ))
}
