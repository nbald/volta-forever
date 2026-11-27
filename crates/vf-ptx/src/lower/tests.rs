//! Unit tests for the PTX lowering: each family of instructions, the plan, chains, scopes,
//! and all the edge cases the code is built to handle.

use super::parse::is_tensor;
use super::types::Declarations;
use super::*;

fn lower(text: &str) -> String {
    body(text, 64, Options::default()).unwrap().0
}

/// Lowered for a program whose rows are where the manual wants them.
fn lower_aligned(text: &str) -> String {
    body(
        text,
        64,
        Options {
            aligned_rows: true,
            ..Default::default()
        },
    )
    .unwrap()
    .0
}

fn refusal(text: &str) -> String {
    body(text, 64, Options::default()).unwrap_err().0
}

fn count(text: &str, needle: &str) -> usize {
    text.matches(needle).count()
}

/// The generated registers `text` uses without a declaration that covers them.
fn undeclared(text: &str) -> Vec<String> {
    let mut declarations = Declarations::default();
    for line in text.lines() {
        if let Some(operands) = line.trim().strip_prefix(".reg ") {
            declarations.declare(operands.trim_end_matches(';'));
        }
    }
    let mut missing: Vec<String> = text
        .lines()
        .filter(|line| !line.trim().starts_with(".reg "))
        .flat_map(scan::names)
        .filter(|name| name.starts_with("%vf_") && declarations.width(name).is_none())
        .map(str::to_string)
        .collect();
    missing.sort();
    missing.dedup();
    missing
}

const F32: &str = "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32";
const F16: &str = "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16";
const S8: &str = "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32";
const VOLTA_F32: &str = "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32";
const VOLTA_F16: &str = "mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16";
/// The shuffle that starts the gathering of an FP16 B operand.
const GATHER: &str = ", 4, 31, -1;";

/// The operand vectors of the Volta instructions in `text`, in order.
fn volta_operands(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .filter_map(|line| line.trim().strip_prefix("mma.sync.aligned.m8n8k4."))
        .map(|line| {
            let operands = line.split_once(' ').unwrap().1.trim_end_matches(';');
            scan::split_operands(operands)
                .iter()
                .map(|part| part.to_string())
                .collect()
        })
        .collect()
}

#[test]
fn an_fp16_product_becomes_two_volta_products() {
    let text = format!("{{\n\t{F32} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}");
    let lowered = lower(&text);
    assert_eq!(count(&lowered, "mma."), 2);
    let products = volta_operands(&lowered);
    // First the registers as they are with the accumulators twice, then the swapped registers
    // with the result of the first.
    assert_eq!(products[0][1], "{%a0, %a1}");
    assert_eq!(products[0][3], "{%c0, %c1, %c2, %c3, %c0, %c1, %c2, %c3}");
    assert_eq!(products[1][3], products[0][0]);
    assert_eq!(count(&lowered, "shf.l.wrap.b32"), 3);
    assert_eq!(count(&lowered, "shfl.sync."), 5);
    // Even threads take the registers 0..3 of the result, odd ones 4..7.
    let result: Vec<&str> = products[1][0].trim_matches(['{', '}']).split(", ").collect();
    for (at, name) in ["%d0", "%d1", "%d2", "%d3"].iter().enumerate() {
        let select = format!("selp.b32 {name}, {}, {}, %vf_p1;", result[4 + at], result[at]);
        assert!(lowered.contains(&select), "{select} not in {lowered}");
    }
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    assert!(lowered.starts_with("{\n\t.reg .b32 %vf_r<"));
    assert!(lowered.ends_with(";\n\tret;\n}"));
}

#[test]
fn operands_used_again_are_gathered_once() {
    let product = |d: &str, a: &str, b: &str| {
        format!("\t{F32} {{{d}0, {d}1, {d}2, {d}3}}, {{{a}0, {a}1}}, {{{b}}}, {{{d}0, {d}1, {d}2, {d}3}};\n")
    };
    let text = format!(
        "{{\n{}{}{}}}",
        product("%x", "%a", "%b"),
        product("%y", "%a", "%b"),
        product("%z", "%e", "%b")
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, VOLTA_F32), 6);
    assert_eq!(count(&lowered, GATHER), 1);
    assert_eq!(count(&lowered, "shf.l.wrap.b32"), 1 + 2 + 2);
    // A statement that writes the B register between two uses.
    let text = format!(
        "{{\n{}\tmov.u32 %b, %other;\n{}}}",
        product("%x", "%a", "%b"),
        product("%y", "%a", "%b")
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, GATHER), 2);
    assert_eq!(count(&lowered, "shf.l.wrap.b32"), 2 + 2);
    // One that only reads it.
    let lowered = lower(&text.replace("mov.u32 %b, %other;", "mov.u32 %other, %b;"));
    assert_eq!(count(&lowered, GATHER), 1);
    // A label: the second instruction can be reached from elsewhere.
    let lowered = lower(&text.replace("mov.u32 %b, %other;", "$L__BB0_2:"));
    assert_eq!(count(&lowered, GATHER), 2);
    assert_eq!(count(&lowered, "shf.l.wrap.b32"), 4 + 2);
    // A register written as part of a vector, as a second result, by a call.
    for write in [
        "ld.global.v2.u32 {%other, %b}, [%rd1];",
        "shfl.sync.idx.b32 %other|%b, %a0, 1, 31, -1;",
        "call (%b), f, (%a0);",
    ] {
        let lowered = lower(&text.replace("mov.u32 %b, %other;", write));
        assert_eq!(count(&lowered, GATHER), 2, "{write}");
    }
    // A nested block that declares a register of the same name hides the outer one.
    let lowered = lower(&text.replace("mov.u32 %b, %other;", "{ .reg .b32 %b; }"));
    assert_eq!(count(&lowered, GATHER), 2);
    let lowered = lower(&text.replace("mov.u32 %b, %other;", "{ .reg .b32 %tmp; mov.u32 %tmp, %b; }"));
    assert_eq!(count(&lowered, GATHER), 1);
}

#[test]
fn chained_accumulators_stay_in_volta_layout() {
    for (turing, volta, accumulators) in [(F32, VOLTA_F32, "{%d0, %d1, %d2, %d3}"), (F16, VOLTA_F16, "{%d0, %d1}")] {
        let product =
            |a: &str, b: &str| format!("\t{turing} {accumulators}, {{{a}0, {a}1}}, {{{b}}}, {accumulators};\n");
        let text = format!("{{\n{}{}}}", product("%a", "%b"), product("%e", "%f"));
        let lowered = lower(&text);
        let products = volta_operands(&lowered);
        assert_eq!(products.len(), 4);
        // The second instruction continues from the Volta registers of the first.
        assert_eq!(products[2][3], products[1][0]);
        assert_eq!(count(&lowered, volta), 4);
        // The accumulators read in between are the Turing registers, which are up to date.
        let text = format!(
            "{{\n{}\tst.global.u32 [%rd1], %d0;\n{}}}",
            product("%a", "%b"),
            product("%e", "%f")
        );
        assert_eq!(volta_operands(&lower(&text))[2][3], products[1][0]);
        // Written in between, they are converted again.
        let text = text.replace("st.global.u32 [%rd1], %d0;", "mov.b32 %d1, %other;");
        let converted = volta_operands(&lower(&text));
        assert_ne!(converted[2][3], converted[1][0]);
        assert_eq!(undeclared(&lower(&text)), Vec::<String>::new());
    }
}

#[test]
fn long_blocks_are_ended_by_branches_never_taken() {
    // Twenty products on different accumulators (so they do not form a chain): five shuffles each.
    let products: String = (0..20)
            .map(|n| format!("\t{F32} {{%x{n}0, %x{n}1, %x{n}2, %x{n}3}}, {{%a0, %a1}}, {{%b{n}}}, {{%x{n}0, %x{n}1, %x{n}2, %x{n}3}};\n"))
            .collect();
    let lowered = lower(&format!("{{\n{products}\tret;\n}}"));
    assert_eq!(count(&lowered, "shfl.sync."), 100);
    // A branch before the product that follows 32 shuffles or more: after 7 products, then 14.
    let branches: Vec<&str> = lowered.lines().filter(|line| line.contains("bra $vf_never;")).collect();
    assert_eq!(branches, ["\t@%vf_n bra $vf_never;"; 2]);
    // Each tests the lane against another number that no lane has.
    assert!(lowered.contains("setp.eq.u32 %vf_n, %vf_l, 32;") && lowered.contains("setp.eq.u32 %vf_n, %vf_l, 33;"));
    // The block they lead to is jumped over when the function starts.
    assert!(lowered.contains("\n\tbra $vf_body;\n$vf_never:\n\ttrap;\n$vf_body:\n"));
    assert!(lowered.contains(".reg .pred %vf_n;"));
    // What was derived before a branch is still used after it: the A registers are swapped once.
    assert_eq!(count(&lowered, "shf.l.wrap.b32"), 2 + 20);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // A chain of 20 on the same accumulator: each product uses 4 shuffles (the chain form),
    // plus the extraction at the end.
    let chain: String = (0..20)
        .map(|n| format!("\t{F32} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b{n}}}, {{%d0, %d1, %d2, %d3}};\n"))
        .collect();
    let chain_lowered = lower(&format!("{{\n{chain}\tret;\n}}"));
    // 16 shuffles to split the accumulator, 6 a product (4 for A, 2 for B), 4 exchanges and
    // 8 shuffles to give it back.
    assert_eq!(count(&chain_lowered, "shfl.sync."), 16 + 20 * 6 + 4 + 8);
    assert_eq!(count(&chain_lowered, "mma.sync.aligned.m8n8k4."), 20);
    assert_eq!(undeclared(&chain_lowered), Vec::<String>::new());
    // A label starts a new block: twelve products on different accumulators, six on each side.
    let mut text = String::from("{\n");
    for n in 0..12 {
        if n == 6 {
            text.push_str("$L__BB0_2:\n");
        }
        text += &format!(
            "\t{F32} {{%y{n}0, %y{n}1, %y{n}2, %y{n}3}}, {{%a0, %a1}}, {{%b{n}}}, {{%y{n}0, %y{n}1, %y{n}2, %y{n}3}};\n"
        );
    }
    text.push_str("\tret;\n}");
    assert_eq!(count(&lower(&text), "$vf_never"), 0);
}

#[test]
fn half_precision_accumulators_change_layout_with_one_exchange_each_way() {
    let text = format!("{{\n\t{F16} {{%d0, %d1}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1}};\n}}");
    let lowered = lower(&text);
    assert_eq!(count(&lowered, VOLTA_F16), 2);
    assert_eq!(count(&lowered, ", 2, 31, -1;"), 2);
    assert_eq!(count(&lowered, "shfl.sync."), 2 + 5);
    let products = volta_operands(&lowered);
    let accumulators: Vec<&str> = products[0][3].trim_matches(['{', '}']).split(", ").collect();
    assert_eq!(accumulators[..2], accumulators[2..]);
    assert!(lowered.contains("selp.b32 %d0, ") && lowered.contains("selp.b32 %d1, "));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn an_integer_product_becomes_shuffles_and_dot_products() {
    let text = format!(
        "{{\n\t{S8} {{%d0, %d1}}, {{%a}}, {{%b}}, {{%c0, %c1}};\n\t{S8} {{%d2, %d3}}, {{%e}}, {{%b}}, {{%d2, %d3}};\n}}"
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, "mma."), 0);
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 16);
    // The row of A for each instruction, the two columns of B once.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4 + 8 + 4);
    assert!(lowered.contains(", %c0;") && lowered.contains(", %c1;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    let unsigned = lower(&text.replace(".s8.s8.", ".u8.s8."));
    assert_eq!(count(&unsigned, "dp4a.u32.s32 "), 16);
}

#[test]
fn ldmatrix_becomes_a_shuffle_of_the_row_address_and_a_load() {
    let text = "{\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%r1, %r2, %r3, %r4}, [%rd1];\n}";
    let lowered = lower(text);
    assert_eq!(count(&lowered, "cvta.to.shared.u64 %vf_q0, %rd1;"), 1);
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4);
    assert_eq!(count(&lowered, "ld.shared.u32 %r"), 4);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // Transposed: two rows per register.
    let lowered = lower("{\n\tldmatrix.sync.aligned.m8n8.x2.trans.b16 {%r1, %r2}, [%rd1+32];\n}");
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4);
    assert_eq!(count(&lowered, "ld.shared.u16 "), 4);
    assert!(lowered.contains(", 32;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // An address that is already an offset in shared memory, in a register of either width.
    let lowered = lower("{\n\t.reg .b32 %r<3>;\n\tldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r1}, [%r2];\n}");
    assert_eq!(count(&lowered, "cvta"), 0);
    assert!(lowered.contains(", %r2;"));
    let lowered = lower("{\n\t.reg .b64 %rd<3>;\n\tldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {%r1}, [%rd2];\n}");
    assert_eq!(count(&lowered, "cvta"), 0);
    assert!(lowered.contains("cvt.u32.u64 "));
    // A variable of shared memory as the address, with and without an offset, as Triton writes it
    // (until 2026-11-23 refused: "address register not recognised").
    let lowered = lower("{\n\tldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r1}, [global_smem];\n}");
    assert!(lowered.contains(", global_smem;"), "{lowered}");
    assert_eq!(count(&lowered, "cvta"), 0);
    assert_eq!(count(&lowered, "ld.shared.u32 %r1"), 1);
    let lowered =
        lower("{\n\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r1, %r2, %r3, %r4}, [global_smem+512];\n}");
    assert!(
        lowered.contains(", global_smem;") && lowered.contains(", 512;"),
        "{lowered}"
    );
    // In a module with 32-bit addresses.
    let lowered = body(text, 32, Options::default()).unwrap().0;
    assert_eq!(count(&lowered, "cvta.to.shared.u32 "), 1);
}

#[test]
fn movmatrix_becomes_two_shuffles() {
    let lowered = lower("{\n\tmovmatrix.sync.aligned.m8n8.trans.b16 %r1, %r2;\n}");
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 2);
    assert!(lowered.contains("shf.r.wrap.b32 %r1, "));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn what_is_not_rewritten_is_copied() {
    let text = "{\n\t.reg .b32 %r<9>; // registers\n\tmma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, {%a0, %a1}, {%b0, %b1}, {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8};\n\twmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%r1}, {%r2}, {%r3}, {%r4};\n$L1:\n\t@%p1 bra $L1;\n\tret;\n}";
    let (lowered, counts) = body(text, 64, Options::default()).unwrap();
    assert_eq!(lowered, text);
    assert_eq!(counts, Counts::default());
}

#[test]
fn refuses_the_instructions_of_later_targets_it_has_no_sequence_for() {
    let function = |statement: &str| format!("{{\n\t.reg .b32 %r<5>;\n\t{statement}\n\tret;\n}}");
    for (statement, reason) in [
        (
            "barrier.cluster.arrive;",
            "barrier.cluster.arrive: no Volta sequence for this form",
        ),
        (
            "mapa.shared::cluster.u32 %r1, %r2, 0;",
            "mapa.shared::cluster.u32: no Volta sequence for this form",
        ),
        (
            "st.shared::cluster.v2.b64 [%r1], {%rd1, %rd2};",
            "st.shared::cluster.v2.b64: no Volta sequence for this form",
        ),
        (
            "cvta.to.shared::cluster.u64 %rd1, %rd2;",
            "cvta.to.shared::cluster.u64: no Volta sequence for this form",
        ),
        (
            "mov.u32 %r1, %cluster_ctarank;",
            "%cluster_ctarank: a register of later targets",
        ),
        (
            "mov.u32 %r1, %reserved_smem_offset_1;",
            "%reserved_smem_offset_1: a register of later targets",
        ),
        (
            "min.f32x2 %rd1, %rd2, %rd3;",
            "min.f32x2: no Volta sequence for this form",
        ),
        (
            "ld.shared::cluster.L2::cache_hint.u32 %r1, [%rd1], %rd2;",
            "ld.shared::cluster.L2::cache_hint.u32: no Volta sequence for this form",
        ),
        (
            "mbarrier.arrive_drop.shared::cta.b64 %rd0, [%r1];",
            "mbarrier.arrive_drop.shared::cta.b64: no Volta sequence for this form",
        ),
        (
            "fence.proxy.async.shared::cta;",
            "fence.proxy.async.shared::cta: no Volta sequence for this form",
        ),
        (
            "{ .reg .u32 r4;cvt.pack.sat.s8.s32.b32 r4, %r1, %r2, 0;}",
            "cvt.pack.sat.s8.s32.b32: no Volta sequence for this form",
        ),
        (
            "wmma.load.a.sync.aligned.row.m16n16k16.shared.s8 {%r1, %r2}, [%r3], %r4;",
            "wmma.load.a.sync.aligned.row.m16n16k16.shared.s8: no Volta sequence for this form",
        ),
        // The types of later targets in forms the layer has no sequence for.
        (
            "{cvt.rz.satfinite.e4m3x2.f32 %rs1, %f1, %f2;}",
            "cvt.rz.satfinite.e4m3x2.f32: no Volta sequence for this form",
        ),
        (
            "{cvt.rn.satfinite.ue8m0x2.f32 %rs1, %f1, %f2;}",
            "cvt.rn.satfinite.ue8m0x2.f32: no Volta sequence for this form",
        ),
        (
            "add.rn.e5m2x4.e4m3x4.satfinite %r1, %r2, %r3;",
            "add.rn.e5m2x4.e4m3x4.satfinite: no Volta sequence for this form",
        ),
        (
            "cvt.rna.tf32.f32 %r1, %f1;",
            "cvt.rna.tf32.f32: no Volta sequence for this form",
        ),
        (
            "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16 {%r1, %r2}, {%r1, %r2}, {%r3}, {%r1, %r2};",
            "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16: no Volta sequence for this form",
        ),
        (
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32 {%r1, %r2, %r3, %r4}, {%r1, %r2, %r3, %r4}, {%r3, %r4}, {%r1, %r2, %r3, %r4};",
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32: no Volta sequence for this form",
        ),
    ] {
        assert_eq!(refusal(&function(statement)), reason, "{statement}");
    }
    // What Volta has under the same first words stays as it is.
    for statement in [
        "barrier.sync 0;",
        "bar.sync 0;",
        "fence.acq_rel.gpu;",
        "ld.shared::cta.u32 %r1, [%r2];",
        "wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {%r1, %r2, %r3, %r4}, [%r3], %r4;",
        "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%r1}, {%r2}, {%r3}, {%r4};",
        "cvt.rn.f16.f32 %r1, %r2;",
    ] {
        assert_eq!(lower(&function(statement)), function(statement), "{statement}");
    }
}

#[test]
fn refuses_what_it_has_no_sequence_for() {
    let operands = "{%d0, %d1, %d2, %d3}, {%a0, %a1}, {%b}, {%c0, %c1, %c2, %c3}";
    for (statement, reason) in [
        (format!("@%p1 {F32} {operands};"), "guarded by a predicate"),
        (
            format!("{F32} {{%d0, %d1}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1}};"),
            "no Volta sequence",
        ),
        (
            format!("{F32} {{%d0, %d1, %d2, %d3}}, {{%a0, 5}}, {{%b}}, {{%c0, %c1, %c2, %c3}};"),
            "operands",
        ),
        (
            "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {%d0, %d1}, {%a}, {%b}, {%c0, %c1};".to_string(),
            "no Volta sequence",
        ),
        (
            "ldmatrix.sync.aligned.m8n8.x4.b16 {%r1, %r2}, [%rd1];".to_string(),
            "operands",
        ),
        (
            "ldmatrix.sync.aligned.m8n8.x1.b16 {%r1}, [buffer+16];".to_string(),
            "address register not recognised",
        ),
        (
            "ldmatrix.sync.aligned.m8n8.x1.b16 {%r1}, [%rd1+%rd2];".to_string(),
            "operands",
        ),
        (
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r1}, [%unknown];".to_string(),
            "address register not recognised",
        ),
        (
            "ldmatrix.sync.aligned.m16n16.x1.trans.b8 {%r1, %r2}, [%rd1];".to_string(),
            "no Volta sequence",
        ),
        ("movmatrix.sync.aligned.m8n8.trans.b16 %r1, 5;".to_string(), "operands"),
        (
            "stmatrix.sync.aligned.m8n8.x1.b16 [%rd1], {%r1};".to_string(),
            "no Volta sequence",
        ),
    ] {
        let found = refusal(&format!("{{\n\t{statement}\n}}"));
        assert!(found.contains(reason), "{statement}: {found}");
    }
    assert_eq!(refusal("{ add.s32 %r1, %r1, 1 }"), "statements not recognised");
}

#[test]
fn x1_matrix_store_preserves_guards_and_materializes_loaded_fragments() {
    let store = "stmatrix.sync.aligned.m8n8.x1.shared.b16";
    let text = format!(
        "{{\n.reg .u64 %rd<2>;\n.reg .u32 %r<2>;\n.reg .pred %p;\n\
         ldmatrix.sync.aligned.m8n8.x1.shared.b16 {{%r0}}, [%rd0];\n\
         @!%p {store} [%rd1+16], {{%r0}};\n}}"
    );
    let result = lower(&text);
    assert!(result.contains("ld.shared.u32 %r0, ["), "{result}");
    for instruction in ["shfl.sync.idx.b32", "add.u32", "st.shared.b32", "bar.warp.sync"] {
        assert!(result.contains(&format!("@!%p {instruction}")), "{result}");
    }
    assert!(undeclared(&result).is_empty(), "{result}");

    for (opcode, operands) in [
        ("stmatrix.sync.aligned.m8n8.x2.shared.b16", "[%r1], {%r0}"),
        ("stmatrix.sync.aligned.m8n8.x4.shared.b16", "[%r1], {%r0, %r2}"),
        ("stmatrix.sync.aligned.m8n8.x3.shared.b16", "[%r1], {%r0, %r1, %r2}"),
        ("stmatrix.sync.aligned.m8n8.x1.trans.shared.b8", "[%r1], {%r0}"),
        ("stmatrix.sync.aligned.m8n8.x1.shared.b8", "[%r1], {%r0}"),
        (store, "[%r1], {%r0, %r2}"),
        (store, "[%r1+%r2], {%r0}"),
    ] {
        let result = refusal(&format!("{{\n{opcode} {operands};\n}}"));
        assert!(result.contains(opcode), "{result}");
    }
}

#[test]
fn each_matrix_store_fragment_uses_its_own_eight_row_addresses() {
    for (count, registers) in [(2, "%r0, %r1"), (4, "%r0, %r1, %r2, %r3")] {
        let text = format!(
            "{{\n.reg .u32 %r<6>;\n\
             ldmatrix.sync.aligned.m8n8.x{count}.shared.b16 {{{registers}}}, [%r4];\n\
             stmatrix.sync.aligned.m8n8.x{count}.shared.b16 [%r5], {{{registers}}};\n}}"
        );
        let lowered = lower(&text);
        let stores: Vec<_> = lowered.lines().filter(|l| l.contains("st.shared.b32")).collect();
        assert_eq!(stores.len(), count);
        for (m, line) in stores.iter().enumerate() {
            assert!(lowered.contains(&format!("ld.shared.u32 %r{m}, [")), "{lowered}");
            assert!(line.ends_with(&format!(", %r{m};")), "{line}");
            let row = if m == 0 {
                "%vf_g".to_string()
            } else {
                format!("%vf_lm{m}")
            };
            assert!(lowered.contains(&format!(", {row}, 31, -1;")), "{lowered}");
        }
        assert_eq!(lowered.matches("bar.warp.sync").count(), 1);
        assert!(undeclared(&lowered).is_empty());
    }
}

#[test]
fn transposed_matrix_store_writes_each_half_to_its_own_row() {
    for count in [1, 2, 4] {
        let registers = (0..count).map(|n| format!("%r{n}")).collect::<Vec<_>>().join(", ");
        let lowered = lower(&format!(
            "{{\n.reg .u32 %r<5>;\n.reg .pred %p;\n@%p stmatrix.sync.aligned.m8n8.x{count}.trans.shared::cta.b16 [%r4+16], {{{registers}}};\n}}"
        ));
        assert_eq!(lowered.matches("@%p st.shared.b16").count(), 2 * count);
        assert_eq!(lowered.matches("@%p shfl.sync.idx.b32").count(), 2 * count);
        assert_eq!(lowered.matches("@%p bar.warp.sync").count(), 1);
        assert!(undeclared(&lowered).is_empty());
    }
}

#[test]
fn tanh_is_expanded_where_it_stands() {
    let (lowered, counts) = body("{\n\ttanh.approx.f32 %f2, %f1;\n\tret;\n}", 64, Options::default()).unwrap();
    assert_eq!(
        counts,
        Counts {
            tanh: 1,
            tensor: 0,
            other: 0,
            reported: 0
        }
    );
    assert!(lowered.starts_with("{\n\t{\n\t.reg .f32 %vf_e, %vf_a;"));
    assert!(lowered.ends_with("selp.f32 %f2, %f1, %vf_e, %vf_p;\n}\n\tret;\n}"));
}

#[test]
fn register_declarations_give_widths() {
    let mut declarations = Declarations::default();
    declarations.declare(".b32 %r<4>, %x");
    declarations.declare(".b64 \t%rd<2>");
    declarations.declare(".pred %p");
    assert_eq!(declarations.width("%r3"), Some(32));
    assert_eq!(declarations.width("%r4"), None);
    assert_eq!(declarations.width("%x"), Some(32));
    assert_eq!(declarations.width("%rd1"), Some(64));
    assert_eq!(declarations.width("%p"), Some(1));
    assert_eq!(declarations.width("%y"), None);
}

/// A chain of four products on `%d0`..`%d3`, a statement that reads one of them, four more
/// products and stores: `read` is how that statement names the register, `product` how the
/// products after it name the accumulator they write and add to.
fn observed_chain(read: &str, product: &str) -> String {
    let products = |d: &str| -> String {
        (0..4)
            .map(|b| format!("\t{F32} {{{d}}}, {{%a0, %a1}}, {{%b{b}}}, {{{d}}};\n"))
            .collect()
    };
    format!(
        "{{\n\t.reg .b32 %d<4>, %a<2>, %b<4>, %observed;\n\t.reg .b64 %out;\n{}\tmov.b32 %observed, {read};\n{}\
         \tst.global.u32 [%out], %observed;\n\tst.global.u32 [%out+4], %d1;\n\tret;\n}}",
        products("%d0, %d1, %d2, %d3"),
        products(product),
    )
}

#[test]
fn a_member_of_a_family_written_with_zeros_is_that_member() {
    // (To the assembler `%d00` is `%d0` of `%d<4>`. Followed as another name, the statement that
    // reads the accumulator in the middle of a chain was given the register as it was before
    // the chain: the initial zero.)
    let plain = lower(&observed_chain("%d0", "%d0, %d1, %d2, %d3"));
    // The chain ends before the statement that reads its accumulator, and another begins after.
    let read = plain.find("mov.b32 %observed, %d0;").unwrap();
    assert_eq!(count(&plain[..read], "mma.sync.aligned.m8n8k4."), 4);
    assert_eq!(count(&plain[read..], "mma.sync.aligned.m8n8k4."), 4);
    assert!(plain[..read].contains("selp.b32 %d0,"), "{plain}");
    // The same program with the other spellings: the same code.
    for (read, product) in [
        ("%d00", "%d0, %d1, %d2, %d3"),
        ("%d000", "%d0, %d1, %d2, %d3"),
        ("%d0", "%d00, %d01, %d02, %d03"),
        ("%d00", "%d0, %d01, %d2, %d003"),
    ] {
        assert_eq!(lower(&observed_chain(read, product)), plain, "{read}; {product}");
    }
    // A guard is a register too.
    let guarded = "{\n\t.reg .pred %p<2>;\n\t.reg .f32 %x, %y;\n\t@%p01 tanh.approx.f32 %y, %x;\n\tret;\n}";
    assert_eq!(refusal(guarded), refusal(&guarded.replace("%p01", "%p1")));
    // A number beyond the family is no member, and is left as it is.
    let beyond = "{\n\t.reg .b32 %d<4>, %d04;\n\tmov.b32 %d04, 1;\n\tret;\n}";
    assert_eq!(lower(beyond), beyond);
}

#[test]
fn a_name_that_two_declarations_can_mean_is_not_followed() {
    // Inside the block `%d00` is the register declared there and `%d0` the member of the family
    // (the assembler takes it so); after the block `%d00` is the member again. Which register
    // the name means depends on the block: the function is refused rather than followed by names.
    let shadowed = "{\n\t.reg .b32 %d<4>, %x;\n\tmov.b32 %d0, 1;\n\t{\n\t.reg .b32 %d00;\n\tmov.b32 %d00, 2;\n\tmov.b32 %x, %d0;\n\t}\n\
                    \tmov.b32 %x, %d00;\n\tret;\n}";
    assert_eq!(
        refusal(shadowed),
        "%d00: a register of a family written with zeros before its number, in a function where %d00 is declared on its own beside a family that has that name"
    );
    // The same with a register declared on its own under the plain name of a member, when a
    // member is written with zeros somewhere.
    let plain_name =
        "{\n\t.reg .b32 %d<4>, %x;\n\t{\n\t.reg .b32 %d1;\n\tmov.b32 %d1, 2;\n\tmov.b32 %x, %d01;\n\t}\n\tret;\n}";
    assert_eq!(
        refusal(plain_name),
        "%d01: a register of a family written with zeros before its number, in a function where %d1 is declared on its own beside a family that has that name"
    );
    // With no member written so, nothing is renamed: each block with its declarations, as before.
    let unpadded = plain_name.replace("%d01", "%d2");
    assert_eq!(lower(&unpadded), unpadded);
    // Two families with a name in common.
    let two = "{\n\t.reg .b32 %d<4>, %d0<4>, %x;\n\tmov.b32 %x, %d00;\n\tret;\n}";
    assert_eq!(
        refusal(two),
        "%d00: a register of a family written with zeros before its number, which two families of registers can mean"
    );
}

#[test]
fn a_name_that_something_else_has_is_not_renamed() {
    let other =
        |name: &str| format!("where {name} is named in another declaration too (a variable, a parameter or a label)");
    // (What the assembler does with each of these was looked at in the machine code it gives.)
    // A variable declared in a block under the name of a member written with zeros is the
    // variable there: `mov.u32 %d1, %d00;` takes its address. Written `%d0`, the statement
    // would read the register of the family.
    let variable = "{\n\t.reg .b32 %d<4>;\n\t.reg .f32 %y;\n\tmov.b32 %d0, 12345;\n\t{\n\t.shared .b32 %d00;\n\tmov.u32 %d1, %d00;\n\
                    \ttanh.approx.f32 %y, 0f00000000;\n\t}\n\tret;\n}";
    assert_eq!(
        refusal(variable),
        format!(
            "%d00: a register of a family written with zeros before its number, in a function {}",
            other("%d00")
        )
    );
    // The same of a variable in the thread's own memory, with an alignment and a size.
    let local = variable.replace(".shared .b32 %d00;", ".local .align 4 .b32 %d00[2];");
    assert!(refusal(&local).ends_with(&other("%d00")), "{}", refusal(&local));
    // A variable under the plain name of a member: `%d00` in its block is still the register of
    // the family outside, which `%d0` there no longer names.
    let plain_name = variable.replace(".shared .b32 %d00;", ".shared .b32 %d0;");
    assert!(
        refusal(&plain_name).ends_with(&other("%d0")),
        "{}",
        refusal(&plain_name)
    );
    // A label.
    let label = "{\n\t.reg .b32 %d<4>;\n\t.reg .f32 %y;\n%d01:\n\tmov.b32 %d0, %d00;\n\ttanh.approx.f32 %y, 0f00000000;\n\tret;\n}";
    assert!(refusal(label).ends_with(&other("%d01")), "{}", refusal(label));
    // What the module names outside the body: the address of `ld.param` is looked for among the
    // parameters, whatever registers the body declares.
    let parameter = "{\n\t.reg .b32 %d<4>;\n\t.reg .f32 %y;\n\tld.param.u32 %d1, [%d00];\n\ttanh.approx.f32 %y, 0f00000000;\n\tret;\n}";
    let refused = body_in(
        parameter,
        &["%out".to_string(), "%d00".to_string()],
        64,
        Options::default(),
    )
    .unwrap_err();
    assert!(refused.0.ends_with(&other("%d00")), "{}", refused.0);
    // (Without that parameter the name can only be the register.)
    let renamed = body_in(parameter, &["%out".to_string()], 64, Options::default())
        .unwrap()
        .0;
    assert!(renamed.contains("ld.param.u32 %d1, [%d0];"), "{renamed}");
    // With no member written with zeros nothing is renamed, whatever else is declared: each
    // statement keeps the names it has.
    for name in ["%scratch", "%d0"] {
        let unpadded = variable.replace("%d00", name);
        let lowered = lower(&unpadded);
        assert!(
            lowered.contains(&format!(".shared .b32 {name};\n\tmov.u32 %d1, {name};")),
            "{lowered}"
        );
    }
}

#[test]
fn a_body_that_has_a_generated_name_is_refused() {
    // The expansion of `tanh` declares `%vf_e` and others: a program's own register of that
    // name would be declared twice, or written by the expansion.
    let own = "{\n\t.reg .f32 %f<3>, %vf_e;\n\tmov.f32 %vf_e, %f1;\n\ttanh.approx.f32 %f2, %f1;\n\tret;\n}";
    assert_eq!(
        refusal(own),
        "%vf_e: a name of the kind that the rewriting gives to what it adds"
    );
    // A label too.
    let label = "{\n\t.reg .f32 %f<3>;\n$vf_body:\n\ttanh.approx.f32 %f2, %f1;\n\tret;\n}";
    assert_eq!(
        refusal(label),
        "$vf_body: a name of the kind that the rewriting gives to what it adds"
    );
    // In a comment it is no name.
    let comment = "{\n\t.reg .f32 %f<3>;\n\t// %vf_e\n\ttanh.approx.f32 %f2, %f1;\n\tret;\n}";
    assert!(lower(comment).contains("%vf_e"));
}

/// `text` without its white space.
fn dense(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn a_comment_in_a_statement_is_not_read_with_it() {
    // A family declared with a comment beside it, before it or inside its count is the same
    // family: a member written with zeros is given the family's spelling as without the
    // comment. (Until 2026-11-21 a declaration was read with its comment: the family was not
    // seen, the member kept its other spelling, and a chain of products did not give the
    // accumulator its value before a statement that read it under that spelling.)
    let plain = "{\n\t.reg .b32 %d<4>;\n\t.reg .f32 %y;\n\tmov.b32 %d0, 12345;\n\tmov.b32 %d1, %d00;\n\ttanh.approx.f32 %y, 0f00000000;\n\tret;\n}";
    let expected = lower(plain);
    assert!(expected.contains("mov.b32 %d1, %d0;"), "{expected}");
    for declaration in [
        ".reg .b32 %d<4> /* register family */;",
        ".reg .b32 /* register family */ %d<4>;",
        ".reg .b32 %d< /* count */ 4>;",
        ".reg .b32 %d<4 // four\n\t>;",
        ".reg .b32 %d<4>; // %d00 is %d0 here",
    ] {
        let commented = plain.replace(".reg .b32 %d<4>;", declaration);
        assert_eq!(dense(&lower(&commented)), dense(&expected), "{declaration}");
    }
    // The case that was found: eight products of a chain, the accumulator read after the
    // first four under its other spelling. The read comes after the accumulator is given back
    // to the program's registers, with the comment as without it.
    let product =
        "\tmma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%d0,%d1,%d2,%d3}, {%a0,%a1}, {%b}, {%d0,%d1,%d2,%d3};\n";
    let chain = |declaration: &str| {
        format!(
            "{{\n\t.reg .u64 %out;\n\t{declaration}\n\tmov.b32 %d0, 0;\n\tmov.b32 %d1, 0;\n\tmov.b32 %d2, 0;\n\tmov.b32 %d3, 0;\n\
             {four}\tmov.b32 %observed, %d00;\n{four}\tst.global.u32 [%out], %observed;\n\tret;\n}}",
            four = product.repeat(4)
        )
    };
    let without = lower(&chain(".reg .b32 %d<4>, %a<2>, %b, %observed;"));
    assert!(without.contains("mov.b32 %observed, %d0;"), "{without}");
    for declaration in [
        ".reg .b32 %d<4> /* register family */, %a<2>, %b, %observed;",
        ".reg .b32 /* register family */ %d<4>, %a<2>, %b, %observed;",
        ".reg .b32 %d< /* count */ 4>, %a<2>, %b, %observed;",
    ] {
        assert_eq!(dense(&lower(&chain(declaration))), dense(&without), "{declaration}");
    }
    // In the operands of an instruction too.
    let operand = plain.replace("mov.b32 %d1, %d00;", "mov.b32 %d1 /* from */, /* the member */ %d00;");
    assert_eq!(dense(&lower(&operand)), dense(&expected));
}

#[test]
fn a_name_in_a_comment_declares_nothing() {
    // `%d0` named in the comment of another declaration is not what that declaration declares:
    // the member written with zeros can only be the register, and is given its spelling.
    let commented = "{\n\t.reg .b32 %d<4>;\n\t.reg .f32 %y;\n\t.shared .b32 scratch /* mentions %d0 only in this comment */;\n\
                     \tmov.b32 %d1, %d00;\n\ttanh.approx.f32 %y, 0f00000000;\n\tret;\n}";
    let lowered = lower(commented);
    assert!(lowered.contains("mov.b32 %d1, %d0;"), "{lowered}");
    // A label written in a comment is no label.
    let label = commented.replace(
        ".shared .b32 scratch /* mentions %d0 only in this comment */;",
        "/* %d01: */ // %d00:",
    );
    let lowered = lower(&label);
    assert!(lowered.contains("mov.b32 %d1, %d0;"), "{lowered}");
}

/// Declarations under which the plan follows `%r1`.. from the `ldmatrix` that loads them.
const DECLARED: &str = "\t.reg .b32 %r<9>, %d<4>;\n\t.reg .b64 %rd<3>;\n";
const LD: &str = "ldmatrix.sync.aligned.m8n8";
const HALVES_F32: &str = "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32";
const D4: &str = "{%d0, %d1, %d2, %d3}";

/// The registers of the 16-byte loads in `text`, in order.
fn wide_loads(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .filter_map(|line| line.trim().strip_prefix("ld.shared.v4.u32 {"))
        .map(|line| {
            line.split_once('}')
                .unwrap()
                .0
                .split(", ")
                .map(str::to_string)
                .collect()
        })
        .collect()
}

fn vector(registers: &[String]) -> String {
    format!("{{{}}}", registers.join(", "))
}

/// Two products whose operands all come from two plain `ldmatrix`.
fn loaded_products() -> String {
    format!(
        "{{\n{DECLARED}\t{LD}.x4.b16 {{%r1, %r2, %r3, %r4}}, [%rd1];\n\t{LD}.x2.b16 {{%r5, %r6}}, [%rd2];\n\
             \t{F32} {D4}, {{%r1, %r2}}, {{%r5}}, {D4};\n\t{F32} {D4}, {{%r3, %r4}}, {{%r6}}, {D4};\n\tret;\n}}"
    )
}

#[test]
fn operands_of_an_fp16_product_are_loaded_by_their_ldmatrix() {
    let lowered = lower(&loaded_products());
    // Split in the halves of k: four Volta instructions, no gathering. One shuffle per operand
    // brings the address of its row, and nothing is loaded in Turing layout.
    assert_eq!(count(&lowered, HALVES_F32), 4);
    assert_eq!(count(&lowered, "mma."), 4);
    assert_eq!(count(&lowered, "shfl.sync."), 4);
    assert_eq!(count(&lowered, "ld.shared.u32 %r"), 0);
    for lane in ["%vf_pr01", "%vf_pr23", "%vf_cr0", "%vf_cr1"] {
        assert_eq!(count(&lowered, &format!(", {lane}, 31, -1;")), 1, "{lane}");
    }
    // The rows of A for (%r1, %r2) and (%r3, %r4), then the columns of B for %r5 and %r6: k 0..3
    // in the first two registers, k 4..7 in the others.
    let loads = wide_loads(&lowered);
    assert_eq!(loads.len(), 4);
    let products = volta_operands(&lowered);
    assert_eq!(products[0][1..3], [vector(&loads[0][..2]), vector(&loads[2][..2])]);
    assert_eq!(products[1][1..3], [vector(&loads[0][2..]), vector(&loads[2][2..])]);
    assert_eq!(products[2][1..3], [vector(&loads[1][..2]), vector(&loads[3][..2])]);
    assert_eq!(products[3][1..3], [vector(&loads[1][2..]), vector(&loads[3][2..])]);
    // The accumulators go through as in the other scheme.
    assert_eq!(products[0][3], "{%d0, %d1, %d2, %d3, %d0, %d1, %d2, %d3}");
    assert_eq!(products[2][3], products[1][0]);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn sixteen_byte_loads_are_taken_only_at_aligned_addresses() {
    let lowered = lower(&loaded_products());
    // Each `ldmatrix` tests its two addresses together and has the same loads word by word.
    assert_eq!(count(&lowered, "ld.shared.v4.u32 "), 4);
    assert_eq!(count(&lowered, "ld.shared.u32 %vf_r"), 16);
    assert_eq!(count(&lowered, ", 15;"), 2);
    assert_eq!(count(&lowered, "setp.ne.u32 %vf_n, "), 2);
    for label in ["$vf_u0", "$vf_a0", "$vf_u1", "$vf_a1"] {
        assert_eq!(count(&lowered, &format!("\n{label}:\n")), 1, "{label}");
        assert_eq!(count(&lowered, &format!("bra {label};")), 1, "{label}");
    }
    assert!(lowered.contains(".reg .pred %vf_n;"));
    let words: Vec<&str> = lowered
        .lines()
        .skip_while(|line| *line != "$vf_u0:")
        .skip(1)
        .take_while(|line| *line != "$vf_a0:")
        .collect();
    let loads = wide_loads(&lowered);
    let address = |line: &str| line.split_once('[').unwrap().1.trim_end_matches("];").to_string();
    assert_eq!(words.len(), 8);
    for (at, line) in words.iter().enumerate() {
        assert!(
            line.starts_with(&format!("\tld.shared.u32 {}, ", loads[at / 4][at % 4])),
            "{line}"
        );
        let base = address(words[at - at % 4]);
        assert_eq!(
            address(line),
            if at % 4 == 0 {
                base
            } else {
                format!("{base}+{}", 4 * (at % 4))
            }
        );
    }
    // Without an instruction after the `ldmatrix` for the last label to stand on, word by word.
    let last = format!(
        "{{\n{DECLARED}$L1:\n\t{F32} {D4}, {{%r1, %r2}}, {{%r5}}, {D4};\n\t{LD}.x4.b16 {{%r1, %r2, %r5, %r6}}, [%rd1];\n$L2:\n\t@%p1 bra $L1;\n\tret;\n}}"
    );
    let lowered = lower(&last);
    assert_eq!(count(&lowered, "ld.shared.v4.u32 "), 0);
    assert_eq!(count(&lowered, "$vf_"), 0);
    assert_eq!(count(&lowered, "ld.shared.u32 %vf_r"), 8);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // A program whose rows are aligned is not tested: the 16-byte loads and nothing else.
    let lowered = lower_aligned(&loaded_products());
    assert_eq!(count(&lowered, "ld.shared.v4.u32 "), 4);
    assert_eq!(count(&lowered, "ld.shared.u32 %vf_r"), 0);
    assert_eq!(count(&lowered, "$vf_"), 0);
    assert_eq!(count(&lowered, "%vf_n"), 0);
    // The products take the loaded registers as with the test (see the test of the halves).
    let (loads, products) = (wide_loads(&lowered), volta_operands(&lowered));
    assert_eq!(products.len(), 4);
    assert_eq!(products[0][1..3], [vector(&loads[0][..2]), vector(&loads[2][..2])]);
    assert_eq!(products[3][1..3], [vector(&loads[1][2..]), vector(&loads[3][2..])]);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn a_register_that_something_else_writes_or_hides_is_gathered() {
    let plain = loaded_products();
    for (what, text) in [
        ("written again", plain.replace("\tret;", "\tmov.b32 %r5, %r7;\n\tret;")),
        (
            "written by a guarded statement",
            plain.replace("\tret;", "\t@%p1 mov.b32 %r5, %r7;\n\tret;"),
        ),
        (
            "hidden in a nested block",
            plain.replace("\tret;", "\t{ .reg .b32 %r5; }\n\tret;"),
        ),
        (
            "loaded twice",
            plain.replace("\tret;", &format!("\t{LD}.x1.b16 {{%r5}}, [%rd2];\n\tret;")),
        ),
        ("not declared in the body", plain.replace("%r<9>", "%r<5>")),
    ] {
        let lowered = lower(&text);
        // The first product gathers its B operand and takes its A registers as they are; the
        // second one, whose operands are not concerned, is as before.
        let expected = if what == "not declared in the body" { 0 } else { 2 };
        assert_eq!(count(&lowered, HALVES_F32), expected, "{what}");
        assert_eq!(count(&lowered, VOLTA_F32), 4 - expected, "{what}");
        assert_eq!(count(&lowered, GATHER), if expected == 2 { 1 } else { 2 }, "{what}");
        for register in ["%r1", "%r2", "%r5"] {
            assert!(
                lowered.contains(&format!("ld.shared.u32 {register}, [")),
                "{what}: {register}"
            );
        }
        assert_eq!(undeclared(&lowered), Vec::<String>::new(), "{what}");
    }
    // A guarded `ldmatrix` is refused as before.
    let guarded = plain.replace(&format!("\t{LD}.x2"), &format!("\t@%p1 {LD}.x2"));
    assert!(refusal(&guarded).contains("guarded by a predicate"));
}

#[test]
fn a_register_read_elsewhere_is_loaded_both_ways() {
    // %r1 and %r2 are stored, %r5 is the B operand of a product whose A is not loaded.
    let text = loaded_products().replace(
            "\tret;",
            &format!("\tst.global.u32 [%rd0], %r1;\n\tst.global.u32 [%rd0+4], %r2;\n\t{F32} {D4}, {{%a0, %a1}}, {{%r5}}, {D4};\n\tret;"),
        );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, HALVES_F32), 4);
    assert_eq!(count(&lowered, VOLTA_F32), 2);
    assert_eq!(count(&lowered, GATHER), 1);
    for (register, loaded) in [("%r1", 1), ("%r2", 1), ("%r3", 0), ("%r4", 0), ("%r5", 1), ("%r6", 0)] {
        assert_eq!(
            count(&lowered, &format!("ld.shared.u32 {register}, [")),
            loaded,
            "{register}"
        );
    }
    // The address of the rows of (%r1, %r2) is selected from the two that are there already.
    assert_eq!(count(&lowered, "%vf_pr01"), 0);
    assert!(lowered.contains(", %vf_p2;"));
    // Row addresses: %r1, %r2, (%r3, %r4); %r5 as it is and as a column, %r6; then the gathering.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 3 + 3 + 4);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn a_transposed_b_operand_is_loaded_as_four_words() {
    let text = format!(
        "{{\n{DECLARED}\t{LD}.x2.trans.b16 {{%r1, %r2}}, [%rd1];\n\
             \t{F16} {{%d0, %d1}}, {{%a0, %a1}}, {{%r1}}, {{%d0, %d1}};\n\
             \t{F16} {{%d0, %d1}}, {{%a2, %a3}}, {{%r2}}, {{%d0, %d1}};\n\tret;\n}}"
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, VOLTA_F16), 4);
    assert_eq!(count(&lowered, GATHER), 0);
    // Two row addresses per register; the accumulators change layout once in and once out
    // (after the second product: nothing reads them between the two).
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4);
    assert_eq!(count(&lowered, "shfl.sync.bfly.b32"), 2);
    assert_eq!(count(&lowered, "ld.shared.u16 "), 0);
    assert_eq!(count(&lowered, "ld.shared.u32 %vf_r"), 8);
    assert_eq!(count(&lowered, "+8];"), 4);
    for lane in ["%vf_tr00", "%vf_tr01", "%vf_tr10", "%vf_tr11"] {
        assert_eq!(count(&lowered, &format!(", {lane}, 31, -1;")), 1, "{lane}");
    }
    // The words loaded are the B operands of the Volta instructions, in order.
    let words: Vec<&str> = lowered
        .lines()
        .filter_map(|line| line.trim().strip_prefix("ld.shared.u32 "))
        .map(|line| line.split_once(',').unwrap().0)
        .collect();
    let products = volta_operands(&lowered);
    for (product, operands) in products.iter().enumerate() {
        assert_eq!(
            operands[2],
            format!("{{{}, {}}}", words[2 * product], words[2 * product + 1])
        );
    }
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // No alignment to test: these are loads of one word.
    assert_eq!(count(&lowered, "$vf_"), 0);
    // Read elsewhere, the register is loaded in Turing layout too.
    let lowered = lower(&text.replace("\tret;", "\tst.global.u32 [%rd0], %r2;\n\tret;"));
    assert_eq!(count(&lowered, "ld.shared.u16 "), 2);
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4 + 2);
}

/// An `ldmatrix` of two A operands and a product for each, with `between` after the load.
fn integer_products(between: &str) -> String {
    format!(
        "{{\n{DECLARED}\t{LD}.x2.b16 {{%r1, %r2}}, [%rd1];\n{between}\
             \t{S8} {{%d0, %d1}}, {{%r1}}, {{%b}}, {{%d0, %d1}};\n\t{S8} {{%d2, %d3}}, {{%r2}}, {{%b}}, {{%d2, %d3}};\n\tret;\n}}"
    )
}

#[test]
fn the_row_of_an_integer_product_is_loaded_at_the_product() {
    let lowered = lower_aligned(&integer_products(""));
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 16);
    // The address of each row at the `ldmatrix`, and the two columns of B once.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 2 + 8);
    assert_eq!(count(&lowered, "ld.shared.u32 "), 0);
    // One volatile load of 16 bytes per product, from the address its `ldmatrix` fetched.
    let lines: Vec<&str> = lowered.lines().map(str::trim).collect();
    let loads: Vec<&&str> = lines
        .iter()
        .filter(|line| line.starts_with("ld.volatile.shared.v4.u32 "))
        .collect();
    assert_eq!(loads.len(), 2);
    // The first two shuffles are those of the `ldmatrix`: the addresses of the rows.
    let addresses: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.strip_prefix("shfl.sync.idx.b32 "))
        .map(|operands| operands.split(',').next().unwrap())
        .take(2)
        .collect();
    let dots: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.strip_prefix("dp4a.s32.s32 "))
        .map(|operands| scan::split_operands(operands)[1])
        .collect();
    for (product, load) in loads.iter().enumerate() {
        assert!(load.ends_with(&format!("[{}];", addresses[product])), "{load}");
        // Each word of the row is the A operand of two dot products, one per column.
        let row: Vec<&str> = load[load.find('{').unwrap() + 1..load.find('}').unwrap()]
            .split(", ")
            .collect();
        let expected: Vec<&str> = [0, 1, 2, 0, 1, 2, 3, 3].iter().map(|&word| row[word]).collect();
        assert_eq!(dots[8 * product..8 * product + 8], expected);
    }
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // Without the option, the rows are gathered: four shuffles each.
    let gathered = lower(&integer_products(""));
    assert_eq!(count(&gathered, "ld.volatile"), 0);
    assert_eq!(count(&gathered, "shfl.sync.idx.b32"), 2 + 2 * 4 + 8);
}

#[test]
fn the_row_is_gathered_when_its_memory_may_have_changed() {
    // What may come between the load and the products without changing what they would read.
    for harmless in [
        "\tst.global.u32 [%rd0], %r7;\n",
        "\tadd.s32 %r7, %r7, 1;\n\tld.shared.u32 %r8, [%rd2];\n",
        "\t@%p1 bra $Lout;\n",
        "\t{LD}.x1.b16 {%r3}, [%rd2];\n",
    ] {
        let lowered = lower_aligned(&integer_products(&harmless.replace("{LD}", LD)));
        assert_eq!(count(&lowered, "ld.volatile.shared.v4.u32 "), 2, "{harmless}");
    }
    // What may: the products gather their rows as without the option.
    for change in [
        "\tst.shared.u32 [%rd2], %r7;\n",
        "\tst.u32 [%rd2], %r7;\n",
        "\t@%p1 st.shared.u8 [%rd2], %r7;\n",
        "\tatom.shared.add.u32 %r7, [%rd2], 1;\n",
        "\tbar.sync 0;\n",
        "\tcall helper, ();\n",
        "$Lagain:\n",
    ] {
        let lowered = lower_aligned(&integer_products(change));
        assert_eq!(count(&lowered, "ld.volatile"), 0, "{change}");
        assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 2 + 2 * 4 + 8, "{change}");
        assert_eq!(count(&lowered, "ld.shared.u32 %r"), 2, "{change}");
        assert_eq!(undeclared(&lowered), Vec::<String>::new());
    }
    // One product that cannot load its row, and none of that register's products does.
    let text = integer_products("").replace(
        "\tret;",
        &format!("\tbar.sync 0;\n\t{S8} {{%d0, %d1}}, {{%r1}}, {{%b}}, {{%d0, %d1}};\n\tret;"),
    );
    let lowered = lower_aligned(&text);
    assert_eq!(count(&lowered, "ld.volatile.shared.v4.u32 "), 1);
    assert_eq!(count(&lowered, "ld.shared.u32 %r1, "), 1);
    assert_eq!(count(&lowered, "ld.shared.u32 %r2, "), 0);
    // A product before its `ldmatrix` in the text reads what an earlier turn loaded.
    let looped = format!(
        "{{\n{DECLARED}$L1:\n\t{S8} {{%d0, %d1}}, {{%r1}}, {{%b}}, {{%d0, %d1}};\n\t{LD}.x1.b16 {{%r1}}, [%rd1];\n\t@%p1 bra $L1;\n\tret;\n}}"
    );
    assert_eq!(count(&lower_aligned(&looped), "ld.volatile"), 0);
}

#[test]
fn a_product_before_its_ldmatrix_in_the_text_reads_what_it_loads() {
    // A loop whose body starts with the product: it reads what the previous iteration loaded.
    let text = format!(
        "{{\n{DECLARED}$L1:\n\t{F32} {D4}, {{%r1, %r2}}, {{%r5}}, {D4};\n\
             \t{LD}.x2.b16 {{%r1, %r2}}, [%rd1];\n\t{LD}.x1.b16 {{%r5}}, [%rd2];\n\t@%p1 bra $L1;\n\tret;\n}}"
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, HALVES_F32), 2);
    let (loads, products) = (wide_loads(&lowered), volta_operands(&lowered));
    assert_eq!(loads.len(), 2);
    assert_eq!(products[0][1..3], [vector(&loads[0][..2]), vector(&loads[1][..2])]);
    assert_eq!(products[1][1..3], [vector(&loads[0][2..]), vector(&loads[1][2..])]);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn transposed_ldmatrix_accepts_nvcc_qualifier_order() {
    let canonical =
        "{\n\t.reg .b64 %rd<2>;\n\tldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r1,%r2,%r3,%r4}, [%rd1];\n}";
    assert_eq!(
        lower(canonical),
        lower(&canonical.replace("m8n8.x4.trans", "trans.m8n8.x4"))
    );
}

const K16_F32: &str = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32";

#[test]
fn f32_mma_accepts_component_constants_and_discarded_destinations() {
    let operands = "{%d0, _, %d2, _}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, 0., -2.5, 0f3FC00000}";
    let lowered = lower(&format!("{{\n\t{K16_F32} {operands};\n\tret;\n}}"));
    assert_eq!(count(&lowered, "selp.b32 %d"), 2);
    assert!(!lowered.contains("selp.b32 _,"));
    assert!(lowered.contains("0f3FC00000"));
    let malformed = operands.replace("0f3FC00000", "0f3FC0000X");
    assert!(refusal(&format!("{{\n\t{K16_F32} {malformed};\n}}")).contains("operands not recognised"));
}

#[test]
fn an_ampere_fp16_product_is_two_turing_products_in_a_row() {
    // Operands in registers that nothing loads: gathered, as for m16n8k8.
    let text = format!(
        "{{\n\t.reg .b32 %a<4>, %b<2>, %c<4>, %d<4>;\n\t{K16_F32}\n\t\t{{%d0, %d1, %d2, %d3}},\n\t\t{{%a0, %a1, %a2, %a3}},\n\t\t{{%b0, %b1}},\n\t\t{{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, "m16n8k16"), 0);
    let products = volta_operands(&lowered);
    assert_eq!(products.len(), 4);
    // The first product reads the accumulators, each next one the result of the one before.
    assert_eq!(products[0][3], "{%c0, %c1, %c2, %c3, %c0, %c1, %c2, %c3}");
    for at in 1..4 {
        assert_eq!(products[at][3], products[at - 1][0]);
    }
    // k 0..7 from A registers 0-1 and B register 0, k 8..15 from A registers 2-3 and B register 1.
    assert_eq!(products[0][1], "{%a0, %a1}");
    assert_eq!(products[2][1], "{%a2, %a3}");
    assert_eq!(count(&lowered, &format!("%b0{GATHER}")), 1);
    assert_eq!(count(&lowered, &format!("%b1{GATHER}")), 1);
    assert_eq!(count(&lowered, "selp.b32 %d"), 4);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // Operands loaded by `ldmatrix`, as Triton writes them: each half of k in the halves scheme.
    let loaded = format!(
        "{{\n\t.reg .b32 %r<9>, %d<4>;\n\t.reg .b64 %rd<3>;\n\t{LD}.x4.shared.b16 {{%r1, %r2, %r3, %r4}}, [%rd1];\n\
             \t{LD}.x4.shared.b16 {{%r5, %r6, %r7, %r8}}, [%rd2];\n\
             \t{K16_F32} {D4}, {{%r1, %r2, %r3, %r4}}, {{%r5, %r6}}, {D4};\n\tret;\n}}"
    );
    let lowered = lower(&loaded);
    assert_eq!(count(&lowered, HALVES_F32), 4);
    assert_eq!(count(&lowered, GATHER), 0);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // Halves in the accumulators of a product of bf16: not a form the layer knows.
    assert_eq!(
        refusal(
            "{\n\tmma.sync.aligned.m16n8k16.row.col.f16.bf16.bf16.f16 {%d0, %d1}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1};\n}"
        ),
        "mma.sync.aligned.m16n8k16.row.col.f16.bf16.bf16.f16: no Volta sequence for this form"
    );
    assert!(is_tensor("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"));
    assert!(!is_tensor("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"));
}

#[test]
fn an_asynchronous_copy_is_made_where_it_starts() {
    let text = "{\n\t.reg .b32 %r<4>;\n\t.reg .b64 %rd<2>;\n\t.reg .pred %p1;\n\
            \tcp.async.cg.shared.global [ %r1 + 0 ], [ %rd1 + 32 ], 0x10, 0x10;\n\
            \tcp.async.ca.shared.global [%r1+16], [%rd1], 8;\n\
            \tcp.async.ca.shared::cta.global.L2::128B [%r1+24], [%rd1+8], 4, 0;\n\
            \tcp.async.cg.shared.global [%r1+32], [%rd1+48], 16, %p1;\n\
            \tcp.async.commit_group;\n\tcp.async.wait_group 0;\n\tcp.async.wait_all;\n\tret;\n}";
    let lowered = lower(text);
    assert_eq!(count(&lowered, "cp.async"), 0);
    assert!(lowered.contains(
        "\tld.global.v4.u32 {%vf_r0, %vf_r1, %vf_r2, %vf_r3}, [%rd1+32];\n\
             \tst.shared.v4.u32 [%r1+0], {%vf_r0, %vf_r1, %vf_r2, %vf_r3};\n"
    ));
    assert!(
        lowered.contains(
            "\tld.global.v2.u32 {%vf_r4, %vf_r5}, [%rd1+0];\n\tst.shared.v2.u32 [%r1+16], {%vf_r4, %vf_r5};\n"
        )
    );
    // Nothing of the source: zeros, and no load.
    assert!(lowered.contains("\tmov.b32 %vf_r6, 0;\n\tst.shared.u32 [%r1+24], %vf_r6;\n"));
    assert_eq!(count(&lowered, "[%rd1+8]"), 0);
    // A predicate says whether the source is ignored.
    assert!(lowered.contains("\t@!%p1 ld.global.v4.u32 {%vf_r7, %vf_r8, %vf_r9, %vf_r10}, [%rd1+48];\n"));
    assert_eq!(count(&lowered, "$vf_c"), 0);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // An amount known only when the kernel runs: all, nothing, or the first bytes.
    let partial = "{\n\t.reg .b32 %r<4>;\n\t.reg .b64 %rd<2>;\n\
            \tcp.async.cg.shared.global [%r1], [%rd1], 16, %r2;\n\tret;\n}";
    let lowered = lower(partial);
    assert!(lowered.contains("\tsetp.eq.u32 %vf_n, %r2, 16;\n\t@%vf_n ld.global.v4.u32 {"));
    assert_eq!(count(&lowered, "@%vf_n ld.global.u8 "), 15);
    assert_eq!(count(&lowered, "@%vf_n st.shared.u8 [%r1+"), 15);
    assert!(lowered.contains("\t@!%vf_n bra $vf_c0;\n"));
    assert!(lowered.contains("\n$vf_c0:\n\tret;"));
    assert!(lowered.contains(".reg .pred %vf_n;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    for (statement, reason) in [
        (
            "cp.async.cg.shared.global [%r1], [%rd1], 8;",
            "no Volta sequence for this form",
        ),
        (
            "cp.async.ca.shared.global.L2::cache_hint [%r1], [%rd1], 16, %rd0;",
            "no Volta sequence for this form",
        ),
        (
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1], %r2, [%r3];",
            "no Volta sequence for this form",
        ),
    ] {
        let refused = refusal(&format!(
            "{{\n\t.reg .b32 %r<4>;\n\t.reg .pred %p<3>;\n\t{statement}\n\tret;\n}}"
        ));
        assert!(refused.ends_with(reason), "{statement}: {refused}");
    }
    // A copy under a predicate, as CUTLASS writes it: the load and the store under it.
    let guarded = lower(
        "{\n\t.reg .b32 %r<4>;\n\t{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %r2, 0;\n\t@p cp.async.cg.shared.global.L2::128B [%r1], [%rd1+32], 16;\n\t}\n\tret;\n}",
    );
    assert!(guarded.contains("\t@p ld.global.v4.u32 {%vf_r0, %vf_r1, %vf_r2, %vf_r3}, [%rd1+32];"));
    assert!(guarded.contains("\t@p st.shared.v4.u32 [%r1+0], {%vf_r0, %vf_r1, %vf_r2, %vf_r3};"));
    assert_eq!(count(&guarded, "cp.async"), 0);
    // A copy of a part of its source that ends its block: its last label has an instruction to
    // stand on.
    let last = lower("{\n\t.reg .b32 %r<4>;\n\tcp.async.ca.shared.global [%r1], [%rd1], 4, %r2;\n}");
    assert!(last.contains("\n$vf_c0:\n\tmov.b32 %vf_r"), "{last}");
    assert!(last.trim_end().ends_with(", 0;\n}"), "{last}");
    // A disabled runtime-sized copy must jump past both the vector store and the byte loop.
    for (guard, inverse) in [("@%p1", "@!%p1"), ("@!%p1", "@%p1")] {
        let code = lower(&format!(
            "{{\n\t.reg .b32 %r<4>;\n\t.reg .b64 %rd1;\n\t.reg .pred %p1;\n\t{guard} cp.async.ca.shared.global [%r1], [%rd1], 4, %r2;\n\tret;\n}}"
        ));
        let branch = code.find(&format!("{inverse} bra $vf_c0;")).unwrap();
        assert!(branch < code.find("ld.global.u32").unwrap());
        assert!(branch < code.find("st.shared.u32").unwrap());
        assert!(code.find("st.shared.u8").unwrap() < code.find("\n$vf_c0:").unwrap());
    }
    let ignored = lower(
        "{\n\t.reg .b32 %r1;\n\t.reg .b64 %rd1;\n\t.reg .pred %p<3>;\n\t@!%p1 cp.async.ca.shared.global [%r1], [%rd1], 4, %p2;\n\tret;\n}",
    );
    assert!(ignored.contains("@!%p1 not.pred %vf_n, %p2;"));
    assert!(ignored.contains("@%vf_n ld.global.u32"));
    assert!(ignored.contains("@!%p1 st.shared.u32"));
    // Of nothing: zeros are stored under it, and no load is made.
    let nothing = lower("{\n\t.reg .b32 %r<4>;\n\t@!%p1 cp.async.ca.shared.global [%r1], [%rd1], 4, 0;\n\tret;\n}");
    assert_eq!(count(&nothing, "ld."), 0);
    assert!(nothing.contains("\t@!%p1 st.shared.u32 [%r1+0], %vf_r0;"));
}

#[test]
fn an_amount_chosen_between_nothing_and_all_takes_no_byte_loop() {
    // Triton's pipelined loads: the amount is chosen once, between the whole size and nothing.
    let function = |declared: &str, before: &str, copy: &str| {
        format!(
            "{{\n\t.reg .b32 %r<8>;\n\t.reg .b64 %rd<2>;\n\t.reg .pred %p<3>;\n{declared}\
             \tsetp.ne.b32 %p1, %r3, 0;\n{before}\t{copy}\n\tcp.async.wait_all;\n\tret;\n}}"
        )
    };
    let chosen = "\tselp.b32 %r2, 16, 0, %p1;\n";
    let copy = "cp.async.cg.shared.global [ %r1 + 0 ], [ %rd1 + 0 ], 0x10, %r2;";
    let lowered = lower(&function("", chosen, copy));
    assert!(
        lowered.contains(
            "\tsetp.eq.u32 %vf_n, %r2, 16;\n\t@%vf_n ld.global.v4.u32 {%vf_r0, %vf_r1, %vf_r2, %vf_r3}, [%rd1+0];\n\
             \tst.shared.v4.u32 [%r1+0], {%vf_r0, %vf_r1, %vf_r2, %vf_r3};\n"
        ),
        "{lowered}"
    );
    // The zeros the destination gets when nothing is read.
    assert_eq!(count(&lowered, "mov.b32 %vf_r"), 4, "{lowered}");
    assert_eq!(count(&lowered, ".u8 "), 0, "{lowered}");
    assert_eq!(count(&lowered, "$vf_c"), 0, "{lowered}");
    assert!(lowered.contains(".reg .pred %vf_n;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // The other order, other spellings of the two numbers, and an 8-byte copy.
    for (choice, statement) in [
        ("\tselp.u32 %r2, 0, 16, %p1;\n", copy),
        ("\tselp.s32 %r2, 0x10, 00, %p1;\n", copy),
        (
            "\tselp.b32 %r2, 8, 0, %p1;\n",
            "cp.async.ca.shared.global [%r1], [%rd1], 8, %r2;",
        ),
    ] {
        let lowered = lower(&function("", choice, statement));
        assert_eq!(count(&lowered, ".u8 "), 0, "{choice}: {lowered}");
        assert_eq!(count(&lowered, "@%vf_n ld.global.v"), 1, "{choice}: {lowered}");
    }
    // Under a predicate: the load under both, the store under the predicate, the zeros always.
    let guarded = lower(&function("", chosen, &format!("@!%p2 {copy}")));
    assert!(
        guarded.contains("\tmov.pred %vf_n, 0;\n\t@!%p2 setp.eq.u32 %vf_n, %r2, 16;\n\t@%vf_n ld.global.v4.u32 {"),
        "{guarded}"
    );
    assert!(guarded.contains("\t@!%p2 st.shared.v4.u32 [%r1+0], {"), "{guarded}");
    assert_eq!(count(&guarded, ".u8 "), 0, "{guarded}");
    // A part of the source may be read, or the register may hold something else: the byte loop.
    for (declared, before, why) in [
        ("", "\tselp.b32 %r2, 8, 0, %p1;\n", "8 of 16 bytes"),
        ("", "\tselp.b32 %r2, 16, %r4, %p1;\n", "a register"),
        ("", "\t@%p2 selp.b32 %r2, 16, 0, %p1;\n", "a guarded choice"),
        (
            "",
            "\tselp.b32 %r2, 16, 0, %p1;\n\tadd.u32 %r2, %r2, 1;\n",
            "written twice",
        ),
        ("", "\tmov.u32 %r2, 16;\n", "written otherwise"),
        (
            "\t{\n\t.reg .b32 %r2;\n\tmov.u32 %r2, 3;\n\t}\n",
            chosen,
            "declared again in a block",
        ),
    ] {
        let lowered = lower(&function(declared, before, copy));
        assert_eq!(count(&lowered, "@%vf_n ld.global.u8 "), 15, "{why}: {lowered}");
        assert!(lowered.contains("@!%vf_n bra $vf_c0;"), "{why}: {lowered}");
    }
}

#[test]
fn packed_conversions_and_nan_bounds_use_what_volta_has() {
    let text = "{\n\t.reg .b32 %r<2>;\n\t.reg .f32 %f<4>;\n\tcvt.rn.f16x2.f32 %r1, %f1, %f2;\n\
            \tmax.NaN.f32 %f3, %f1, %f2;\n\tmin.ftz.NaN.f32 %f3, %f3, 0f3F800000;\n\tret;\n}";
    let lowered = lower(text);
    // The first input in the upper half.
    assert!(lowered.contains(
        "\tcvt.rn.f16.f32 %vf_x0, %f1;\n\tcvt.rn.f16.f32 %vf_x1, %f2;\n\tcvt.u32.u16 %vf_r0, %vf_x0;\n\
             \tcvt.u32.u16 %vf_r1, %vf_x1;\n\tshl.b32 %vf_r0, %vf_r0, 16;\n\tor.b32 %r1, %vf_r0, %vf_r1;\n"
    ));
    assert!(lowered.contains(
        "\tmax.f32 %vf_r2, %f1, %f2;\n\tsetp.nan.f32 %vf_n, %f1, %f2;\n\tselp.f32 %f3, 0f7FFFFFFF, %vf_r2, %vf_n;\n"
    ));
    assert!(lowered.contains("\tmin.ftz.f32 %vf_r3, %f3, 0f3F800000;\n"));
    assert!(lowered.contains(".reg .b16 %vf_x<2>;"));
    assert_eq!(count(&lowered, ".NaN."), 0);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    for statement in [
        "cvt.rn.relu.f16x2.f32 %r1, %f1, %f2;",
        "cvt.rn.satfinite.bf16.f32 %r1, %f1;",
        "add.rp.bf16 %r1, %r1, %r0;",
        "redux.sync.add.s32 %r1, %r0, %r1;",
        "max.NaN.abs.f32 %f3, %f1, %f2, %f0;",
        // Not instructions of the PTX ISA: a sum of floats (ptxas refuses it), and operations on
        // types that they do not take.
        "redux.sync.add.f32 %r1, %r0, -1;",
        "redux.sync.and.f32 %r1, %r0, -1;",
        "redux.sync.add.b32 %r1, %r0, -1;",
        "redux.sync.min.b32 %r1, %r0, -1;",
        "redux.sync.xor.u32 %r1, %r0, -1;",
    ] {
        let refused = refusal(&format!("{{\n\t.reg .b32 %r<2>;\n\t{statement}\n\tret;\n}}"));
        assert!(
            refused.ends_with("no Volta sequence for this form"),
            "{statement}: {refused}"
        );
    }
}

#[test]
fn instructions_of_later_targets_get_their_plain_equivalents() {
    let text = "{\n\t.reg .b32 %r<9>;\n\t.reg .b16 %h<2>;\n\t.reg .f32 %f<3>;\n\t.reg .b64 %rd<2>;\n\t.reg .f64 %fd<4>;\n\t.reg .pred %p1;\n\
            \tmin.relu.s32 %r1, %r2, %r3;\n\
            \t@%p1 cvt.rn.bf16.f32 %h0, %f1;\n\
            \tcvt.rz.bf16x2.f32 %r4, %f1, %f2;\n\
            \tcvt.f32.bf16 %f0, %h1;\n\
            \tredux.sync.max.s32 %r5, %r6, 0xffffffff;\n\
            \tld.global.v8.b32 {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, [%rd1+0x20];\n\
            \t@%p1 st.global.cg.v4.f64 [%rd1], {%fd0, %fd1, %fd2, %fd3};\n\
            \tld.global.L2::128B.u32 %r1, [%rd1];\n\
            \tst.volatile.local.b32 [%rd0], %r1;\n\
            \tgriddepcontrol.wait;\n\tret;\n}";
    let lowered = lower(text);
    assert!(lowered.contains("\tmin.s32 %vf_r0, %r2, %r3;\n\tmax.s32 %r1, %vf_r0, 0;\n"));
    // Nearest even on the lower half, a NaN kept a NaN, the guard on what the program reads.
    assert!(lowered.contains(
            "\tmov.b32 %vf_r1, %f1;\n\tshr.u32 %vf_r2, %vf_r1, 16;\n\tand.b32 %vf_r2, %vf_r2, 1;\n\
             \tadd.u32 %vf_r2, %vf_r2, 32767;\n\tadd.u32 %vf_r1, %vf_r1, %vf_r2;\n\tshr.u32 %vf_r1, %vf_r1, 16;\n\
             \tsetp.nan.f32 %vf_n, %f1, %f1;\n\tselp.b32 %vf_r1, 32767, %vf_r1, %vf_n;\n\t@%p1 cvt.u16.u32 %h0, %vf_r1;\n"
        ));
    // Truncation adds nothing; the first input goes to the upper half.
    assert_eq!(count(&lowered, "add.u32 "), 2);
    assert!(lowered.contains("\tshl.b32 %vf_r3, %vf_r3, 16;\n\tor.b32 %r4, %vf_r3, %vf_r5;\n"));
    assert!(lowered.contains("\tcvt.u32.u16 %vf_r7, %h1;\n\tshl.b32 %vf_r7, %vf_r7, 16;\n\tmov.b32 %f0, %vf_r7;\n"));
    // Five exchanges reach every lane.
    assert_eq!(count(&lowered, "shfl.sync.bfly.b32"), 5);
    assert_eq!(count(&lowered, "max.s32 %vf_r8, %vf_r8, "), 6);
    assert!(lowered.contains(
        "\tld.global.v4.b32 {%r1, %r2, %r3, %r4}, [%rd1+32];\n\tld.global.v4.b32 {%r5, %r6, %r7, %r8}, [%rd1+48];\n"
    ));
    assert!(lowered.contains(
        "\t@%p1 st.global.cg.v2.f64 [%rd1+0], {%fd0, %fd1};\n\t@%p1 st.global.cg.v2.f64 [%rd1+16], {%fd2, %fd3};\n"
    ));
    assert!(lowered.contains("\tld.global.u32 %r1, [%rd1];\n\tst.local.b32 [%rd0], %r1;\n"));
    for gone in [
        "relu",
        "bf16",
        "redux",
        ".v8.",
        ".v4.f64",
        "L2::",
        "volatile",
        "griddepcontrol",
    ] {
        assert_eq!(count(&lowered, gone), 0, "{gone}");
    }
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // The member mask through a register, as compilers write it: set in the same block, or in
    // another one by the only statement of the function that writes the register.
    let through = "{\n\t.reg .b32 %r<4>;\n\tmov.b32 %r1, -1;\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}";
    assert_eq!(count(&lower(through), "shfl.sync.bfly.b32"), 5);
    let rewritten =
        "{\n\t.reg .b32 %r<4>;\n\tmov.b32 %r1, 3;\n\tmov.b32 %r1, -1;\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}";
    assert_eq!(count(&lower(rewritten), "shfl.sync.bfly.b32"), 5);
    let earlier = "{\n\t.reg .b32 %r<4>;\n\tmov.u32 %r1, -1;\n$L1:\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}";
    assert_eq!(count(&lower(earlier), "shfl.sync.bfly.b32"), 5);
    // As Triton writes it: the `mov` before labels that no statement names (labels of the line
    // tables) and a branch out of the block, with the register written again later, in a loop
    // that comes after the instructions. Nothing comes to them but by falling through from the
    // `mov`, so the mask they read is all ones (until 2026-11-23 any label between ended what was
    // known, and the later writes ended the register's being a constant of the function).
    let reaching = "{\n\t.reg .b32 %r<4>;\n\t.reg .pred %p<2>;\n\tmov.b32 %r1, -1;\n$L__tmp1:\n\t\
                    redux.sync.add.s32 %r2, %r3, %r1;\n\tsetp.eq.s32 %p1, %r2, 0;\n\t@%p1 bra $L__BB0_4;\n\
                    $L__tmp2:\n\tredux.sync.min.u32 %r2, %r3, %r1;\n$L__BB0_2:\n\tadd.s32 %r1, %r1, 1;\n\t\
                    setp.lt.s32 %p1, %r1, 9;\n\t@%p1 bra $L__BB0_2;\n$L__BB0_4:\n\tret;\n}";
    assert_eq!(count(&lower(reaching), "shfl.sync.bfly.b32"), 10, "{reaching}");
    // Not when the register can hold something else where the instruction reads it: another
    // statement writes it and a label that a branch names lies between (before or after the
    // instruction in the text, or a label of a list of branch targets), the `mov` is guarded, an
    // inner scope declares a register of that name, or the register is written between.
    for unknown in [
        "{\n\t.reg .b32 %r<4>;\n\t.reg .pred %p1;\n\tmov.b32 %r1, -1;\n$L1:\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tadd.s32 %r1, %r1, 1;\n\t@%p1 bra $L1;\n\tret;\n}",
        "{\n\t.reg .b32 %r<4>;\n\t.reg .pred %p1;\n\tmov.b32 %r1, -1;\n\t@%p1 bra $L1;\n\tmov.b32 %r1, 5;\n$L1:\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}",
        "{\n\t.reg .b32 %r<4>;\n\tmov.b32 %r1, -1;\n$L1:\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tmov.b32 %r1, 3;\n$L2:\n\tbrx.idx %r1, $T;\n$T: .branchtargets $L1, $L2;\n\tret;\n}",
        "{\n\t.reg .b32 %r<4>;\n\tmov.b32 %r1, -1;\n$L1:\n\tadd.s32 %r1, %r1, 1;\n$L2:\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}",
        "{\n\t.reg .b32 %r<4>;\n\t.reg .pred %p1;\n\t@%p1 mov.b32 %r1, -1;\n$L1:\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}",
        "{\n\t.reg .b32 %r<4>;\n\tmov.b32 %r1, -1;\n$L1:\n\t{\n\t.reg .b32 %r1;\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\t}\n\tret;\n}",
        "{\n\t.reg .b32 %r<4>;\n\tmov.b32 %r1, 0x7fffffff;\n$L1:\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}",
        // (The outer register holds 15; the one set to -1 is the inner scope's, of a family declared again there.)
        "{\n\t.reg .b32 %r<4>;\n\tmov.b32 %r1, 15;\n\t{\n\t.reg .b32 %r<2>;\n\tmov.b32 %r1, -1;\n\t}\n\tredux.sync.add.s32 %r2, %r3, %r1;\n\tret;\n}",
    ] {
        assert!(
            refusal(unknown).ends_with("no Volta sequence for this form"),
            "{unknown}"
        );
    }
    // A group instruction that ends its block leaves something for a label to stand on.
    let last = lower("{\n\t.reg .b32 %r<2>;\n$L1:\n\tcp.async.wait_all;\n}");
    assert!(last.contains("$L1:\n\tmov.b32 %vf_r0, 0;\n}"));
    // The spellings of ldmatrix.
    for spelling in ["x4.m8n8.shared", "x4.trans.m8n8.shared", "m8n8.x4.trans"] {
        let text = format!(
            "{{\n\t.reg .b32 %r<5>;\n\t.reg .b64 %rd<2>;\n\tldmatrix.sync.aligned.{spelling}.b16 {{%r1, %r2, %r3, %r4}}, [%rd1];\n\tret;\n}}"
        );
        assert_eq!(count(&lower(&text), "ldmatrix"), 0, "{spelling}");
    }
}

const A_S8_K16: &str = "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32";
const A_S8_K32: &str = "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32";

#[test]
fn ampere_m16n8k16_integer_becomes_two_turing_products() {
    let text =
        format!("{{\n\t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}");
    let lowered = lower(&text);
    assert_eq!(count(&lowered, "m16n8k16"), 0);
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 16);
    // Rows of A: a0 gathered once for rows 0-7, a1 once for rows 8-15. B columns gathered once.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4 + 4 + 8);
    // Accumulates on {c0, c1} for the first product, {c2, c3} for the second.
    assert!(lowered.contains(", %c0;") && lowered.contains(", %c1;"));
    assert!(lowered.contains(", %c2;") && lowered.contains(", %c3;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // Unsigned and mixed element types.
    let unsigned = lower(&text.replace(".s8.s8.", ".u8.s8."));
    assert_eq!(count(&unsigned, "dp4a.u32.s32 "), 16);
    let mixed = lower(&text.replace(".s8.s8.", ".u8.u8."));
    assert_eq!(count(&mixed, "dp4a.u32.u32 "), 16);
}

#[test]
fn ampere_integer_results_that_are_operands_are_written_last() {
    // The accumulators of the two halves exchanged, and a result that is the B operand.
    for operands in [
        "{%d0, %d1, %d2, %d3}, {%a0, %a1}, {%b}, {%d2, %d3, %d0, %d1}",
        "{%d0, %b, %d2, %d3}, {%a0, %a1}, {%b}, {%c0, %c1, %c2, %c3}",
        "{%d0, %d1, %a1, %d3}, {%a0, %a1}, {%b}, {%d0, %d1, %c2, %d3}",
    ] {
        for opcode in [A_S8_K16.to_string(), format!("{A_S8_K16}.satfinite")] {
            let lowered = lower(&format!("{{\n\t{opcode} {operands};\n\tret;\n}}"));
            let lines: Vec<&str> = lowered.lines().map(str::trim).collect();
            // The four results are moved in after the last computation, and nothing writes a
            // register of the program before.
            let first_move = lines.iter().position(|line| line.starts_with("mov.b32 %")).unwrap();
            assert_eq!(
                lines[first_move..]
                    .iter()
                    .filter(|line| line.starts_with("mov.b32 %"))
                    .count(),
                4
            );
            for line in &lines[..first_move] {
                let written = line.split_whitespace().nth(1).unwrap_or("");
                assert!(
                    !line.contains(' ')
                        || written.starts_with("%vf_")
                        || line.starts_with('.')
                        || line.starts_with('{'),
                    "{line}"
                );
            }
            assert_eq!(undeclared(&lowered), Vec::<String>::new());
        }
    }
    // In place, each accumulator replaced by its result: no move.
    let in_place = lower(&format!(
        "{{\n\t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%d0, %d1, %d2, %d3}};\n\tret;\n}}"
    ));
    assert_eq!(count(&in_place, "mov.b32 %d"), 0);
}

#[test]
fn ampere_m16n8k32_integer_becomes_four_turing_products() {
    let text = format!(
        "{{\n\t{A_S8_K32} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, "m16n8k32"), 0);
    // 4 sub-products x 8 dp4a = 32.
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 32);
    // Rows: a0, a1, a2, a3 each gathered once (4 shuffles each). Columns: b0, b1 each once (8 shuffles each).
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4 * 4 + 2 * 8);
    // First product: accumulate on {c0, c1}; second chains on the result; third uses {c2, c3}.
    assert!(lowered.contains(", %c0;") && lowered.contains(", %c1;"));
    assert!(lowered.contains(", %c2;") && lowered.contains(", %c3;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn ampere_integer_with_loaded_rows() {
    // m16n8k16: A loaded by ldmatrix.x2, rows loaded at each product.
    let text = format!(
        "{{\n{DECLARED}\t{LD}.x2.b16 {{%r1, %r2}}, [%rd1];\n\
             \t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%r1, %r2}}, {{%b}}, {{%d0, %d1, %d2, %d3}};\n\tret;\n}}"
    );
    let lowered = lower_aligned(&text);
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 16);
    // Two volatile 16-byte loads (one per sub-product).
    assert_eq!(count(&lowered, "ld.volatile.shared.v4.u32 "), 2);
    // Row addresses for the ldmatrix (2) and columns of B (8).
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 2 + 8);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn ampere_integer_row_gathered_when_memory_changed() {
    // A label between ldmatrix and the product: rows gathered, not loaded.
    let text = format!(
        "{{\n{DECLARED}\t{LD}.x2.b16 {{%r1, %r2}}, [%rd1];\n\
             $Lagain:\n\
             \t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%r1, %r2}}, {{%b}}, {{%d0, %d1, %d2, %d3}};\n\tret;\n}}"
    );
    let lowered = lower_aligned(&text);
    assert_eq!(count(&lowered, "ld.volatile"), 0);
    // Rows gathered: 2*4 = 8 for rows, 8 for columns, 2 for ldmatrix addresses, 2 for plain loads.
    assert!(count(&lowered, "shfl.sync.idx.b32") >= 8 + 8);
}

const VOLTA: &str = "mma.sync.aligned.m8n8k4.";

#[test]
fn fp8_values_become_halves_where_they_are() {
    // A pair of bytes to a pair of halves: no other lane is asked, and e5m2 needs no arithmetic.
    let e5m2 = lower("{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\tcvt.rn.f16x2.e5m2x2 %r1, %rs1;\n\tret;\n}");
    assert_eq!(count(&e5m2, "e5m2"), 0);
    assert_eq!(count(&e5m2, "shfl."), 0);
    assert_eq!(count(&e5m2, "mul."), 0);
    // One permutation of bytes.
    assert_eq!(count(&e5m2, "prmt.b32 "), 1);
    assert_eq!(count(&e5m2, "shl.b32 "), 0);
    assert!(e5m2.contains("\tmov.b32 %r1, %vf_r"));
    assert_eq!(undeclared(&e5m2), Vec::<String>::new());
    // e4m3: both halves times 256 in one multiplication, and the NaN byte made a NaN.
    let e4m3 = lower("{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\t@%p1 cvt.rn.f16x2.e4m3x2 %r1, %rs1;\n\tret;\n}");
    assert_eq!(count(&e4m3, "e4m3"), 0);
    assert_eq!(count(&e4m3, "prmt.b32 "), 1);
    assert_eq!(count(&e4m3, "mul.rn.f16x2 "), 1);
    assert_eq!(count(&e4m3, "0x5c005c00;"), 1);
    assert_eq!(count(&e4m3, "cvt.rn.f16.f32 "), 0);
    assert_eq!(count(&e4m3, "0x00800080"), 1);
    // The guard is on the statement that writes the register of the program, and only there.
    assert_eq!(count(&e4m3, "@%p1 "), 1);
    assert!(e4m3.contains("\t@%p1 mov.b32 %r1, %vf_r"));
    assert_eq!(undeclared(&e4m3), Vec::<String>::new());
}

#[test]
fn ordinary_bfloat_bounds_select_numeric_lanes() {
    for op in ["min", "max"] {
        let text = lower(&format!(
            "{{\n.reg .b32 %r<2>;\n.reg .pred %p;\n@%p {op}.bf16x2 %r0, %r0, %r1;\n}}"
        ));
        assert_eq!(text.matches(&format!("{op}.f32 ")).count(), 2);
        assert!(!text.contains("setp.nan"));
        assert!(text.contains("@%p prmt.b32 %r0,"));
        assert!(undeclared(&text).is_empty());
    }
}

#[test]
fn packed_bfloat_sign_and_fp8_conversion_keep_both_lanes() {
    for kind in ["e4m3x2", "e5m2x2"] {
        let text = lower(&format!(
            "{{\n.reg .pred %p;\n.reg .b32 %r;\n.reg .b16 %h;\n@%p abs.bf16x2 %r, %r;\n@%p cvt.rn.satfinite.{kind}.bf16x2 %h, %r;\n}}"
        ));
        assert!(text.contains("@%p and.b32 %r, %r, 0x7fff7fff;"));
        assert!(text.contains("and.b32 %vf_r0, %r, 0xffff0000;"));
        assert!(text.contains("shl.b32 %vf_r1, %r, 16;"));
        assert!(text.contains("@%p cvt.u16.u32 %h,"));
        assert!(undeclared(&text).is_empty());
    }
}

#[test]
fn floats_become_fp8_bytes_with_one_rounding() {
    for (kind, largest, low) in [("e4m3x2", "0f43E00000", 7), ("e5m2x2", "0f47600000", 8)] {
        let lowered = lower(&format!(
            "{{\n\t.reg .f32 %f<3>;\n\t.reg .b16 %rs1;\n\tcvt.rn.satfinite.{kind}.f32 %rs1, %f1, %f2;\n\tret;\n}}"
        ));
        assert_eq!(count(&lowered, "satfinite"), 0, "{kind}");
        // Each value: capped at the largest finite one, cut to a half towards zero, and rounded
        // once, on the low bits and on what the cut dropped.
        assert_eq!(
            count(&lowered, &format!("min.f32 %vf_r0, %vf_r0, {largest};")),
            1,
            "{kind}"
        );
        assert_eq!(count(&lowered, "cvt.rz.f16.f32 "), 2, "{kind}");
        assert_eq!(count(&lowered, "cvt.rn.f16.f32 "), 0, "{kind}");
        assert_eq!(count(&lowered, "setp.neu.f32 "), 2, "{kind}");
        assert_eq!(
            count(&lowered, &format!("add.u32 %vf_r2, %vf_r2, {};", (1 << low) - 1)),
            1,
            "{kind}"
        );
        assert_eq!(
            count(&lowered, &format!("shr.u32 %vf_r5, %vf_r2, {};", low + 1)),
            1,
            "{kind}"
        );
        // The first value is the upper byte.
        assert!(lowered.contains("abs.f32 %vf_r0, %f1;"), "{kind}");
        assert!(lowered.contains("shl.b32 %vf_r5, %vf_r5, 8;"), "{kind}");
        assert!(lowered.contains("\tcvt.u16.u32 %rs1, %vf_r5;"), "{kind}");
        assert_eq!(undeclared(&lowered), Vec::<String>::new(), "{kind}");
        // From two halves: the same, on their values as f32.
        let halves = lower(&format!(
            "{{\n\t.reg .b32 %r1;\n\t.reg .b16 %rs1;\n\tcvt.rn.satfinite.{kind}.f16x2 %rs1, %r1;\n\tret;\n}}"
        ));
        assert_eq!(count(&halves, "cvt.f32.f16 "), 2 + 2, "{kind}");
        assert_eq!(count(&halves, "cvt.rz.f16.f32 "), 2, "{kind}");
        assert_eq!(undeclared(&halves), Vec::<String>::new(), "{kind}");
    }
    // To 10 bits of mantissa.
    let tf32 = lower("{\n\t.reg .f32 %f1;\n\t.reg .b32 %r1;\n\tcvt.rn.tf32.f32 %r1, %f1;\n\tret;\n}");
    assert_eq!(count(&tf32, "tf32"), 0);
    assert!(tf32.contains("add.u32 %vf_r1, %vf_r1, 4095;"));
    assert!(tf32.contains("and.b32 %vf_r0, %vf_r0, 0xffffe000;"));
    assert_eq!(undeclared(&tf32), Vec::<String>::new());
}

#[test]
fn e2m1_values_become_halves() {
    let lowered = lower("{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\tcvt.rn.f16x2.e2m1x2 %r1, %rs1;\n\tret;\n}");
    assert_eq!(count(&lowered, "e2m1"), 0);
    // Looked up by PRMT LUT and signed with PRMT+LOP3 (zero HMUL2).
    assert!(lowered.contains("cvt.u32.u16 %vf_r0, %rs1;"));
    assert_eq!(count(&lowered, "mul.rn.f16x2 "), 0);
    assert_eq!(count(&lowered, "cvt.f32.f16 "), 0);
    assert_eq!(count(&lowered, "cvt.rn.f16.f32 "), 0);
    assert_eq!(count(&lowered, "prmt.b32 "), 3);
    assert_eq!(count(&lowered, "lop3.b32 "), 1);
    assert!(lowered.contains("\tmov.b32 %r1, %vf_r"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    assert_eq!(count(&lowered, "mul.lo.u32 "), 0);
    // A register of 8 bits, which is what the type asks for.
    let byte = lower("{\n\t.reg .b8 %rb1;\n\t.reg .b32 %r1;\n\tcvt.rn.f16x2.e2m1x2 %r1, %rb1;\n\tret;\n}");
    assert!(byte.contains("cvt.u32.u8 %vf_r0, %rb1;"));
    assert_eq!(undeclared(&byte), Vec::<String>::new());
    // With .relu: generic prmt sign-extends bit 7 (0) of LUT bytes when selector bit 3 is set.
    let relu = lower("{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\tcvt.rn.relu.f16x2.e2m1x2 %r1, %rs1;\n\tret;\n}");
    assert_eq!(count(&relu, "e2m1"), 0);
    assert_eq!(count(&relu, "mul.lo.u32 "), 0);
    assert_eq!(count(&relu, "mul.rn.f16x2 "), 0);
    assert_eq!(count(&relu, "prmt.b32 "), 2);
    assert_eq!(undeclared(&relu), Vec::<String>::new());
}

#[test]
fn floats_become_e2m1_nibbles() {
    let lowered =
        lower("{\n\t.reg .f32 %f<3>;\n\t.reg .b16 %rs1;\n\tcvt.rn.satfinite.e2m1x2.f32 %rs1, %f1, %f2;\n\tret;\n}");
    assert_eq!(count(&lowered, "e2m1"), 0);
    assert_eq!(count(&lowered, "satfinite"), 0);
    // Each value against the seven midpoints, past which a tie counts or not; clamped at 6.
    assert_eq!(count(&lowered, "setp.gt.f32 %vf_n, "), 2 * 4);
    assert_eq!(count(&lowered, "setp.ge.f32 %vf_n, "), 2 * 3);
    assert_eq!(count(&lowered, "min.f32 "), 2);
    assert_eq!(count(&lowered, "0f40C00000"), 2);
    // A NaN is +6.
    assert_eq!(count(&lowered, "setp.nan.f32 "), 2);
    assert_eq!(count(&lowered, ", 7, "), 2);
    // The first value in the upper nibble.
    assert!(lowered.contains("shl.b32 %vf_r1, %vf_r1, 4;"));
    assert!(lowered.contains("\tcvt.u16.u32 %rs1, %vf_r1;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // With .relu: a negative value is +0, and no sign goes into the nibble.
    let relu = lower(
        "{\n\t.reg .f32 %f<3>;\n\t.reg .b16 %rs1;\n\tcvt.rn.relu.satfinite.e2m1x2.f32 %rs1, %f1, %f2;\n\tret;\n}",
    );
    assert_eq!(count(&relu, "e2m1"), 0);
    assert_eq!(count(&relu, "relu"), 0);
    assert_eq!(count(&relu, "setp.ne.u32 "), 2);
    assert_eq!(count(&relu, "or.b32 "), count(&lowered, "or.b32 ") - 2);
    assert_eq!(undeclared(&relu), Vec::<String>::new());
}

#[test]
fn floats_become_ue8m0_bytes() {
    let body = |rounding: &str| {
        format!(
            "{{\n\t.reg .f32 %f<3>;\n\t.reg .b16 %rs1;\n\tcvt.{rounding}.satfinite.ue8m0x2.f32 %rs1, %f1, %f2;\n\tret;\n}}"
        )
    };
    // Toward +infinity (the form NVIDIA's libraries have): the exponent field, and one more
    // when the float is above that power of two; for a subnormal float, above 2^-127.
    let up = lower(&body("rp"));
    assert_eq!(count(&up, "ue8m0"), 0);
    assert_eq!(count(&up, "satfinite"), 0);
    assert_eq!(count(&up, "abs.f32 "), 2);
    assert_eq!(count(&up, ", 23;"), 2);
    assert_eq!(count(&up, ", 0x400000, 0, %vf_n;"), 2);
    assert_eq!(count(&up, "setp.gt.u32 "), 2);
    // No more than the largest power; a NaN is 0xff.
    assert_eq!(count(&up, ", 0xfe;"), 2);
    assert_eq!(count(&up, ", 0xff, "), 2);
    assert!(up.contains("\tcvt.u16.u32 %rs1, "));
    assert_eq!(undeclared(&up), Vec::<String>::new());
    // Toward zero: the exponent field.
    let down = lower(&body("rz"));
    assert_eq!(count(&down, "ue8m0"), 0);
    assert_eq!(count(&down, "setp.gt.u32 "), 0);
    assert_eq!(count(&down, ", 0xfe;"), 2);
    assert_eq!(undeclared(&down), Vec::<String>::new());
}

#[test]
fn bf16_fma_with_a_known_negative_zero_addend_is_a_product() {
    let run = |setup: &str, guard: &str| {
        lower(&format!(
            "{{\n.reg .b16 %h<6>;\n.reg .pred %p;\n{setup}\n{guard}fma.rn.bf16 %h0,%h1,%h2,%h3;\nret;\n}}"
        ))
    };
    let known = run("mov.b16 %h3,0x8000;", "@%p ");
    assert_eq!(count(&known, "mul.rn.f32 "), 1);
    assert!(known.contains("@%p cvt.u16.u32 %h0,"));
    for setup in [
        "mov.b16 %h3,0;",
        "@%p mov.b16 %h3,0x8000;",
        "mov.b16 %h3,0x8000;\n@%p mov.b16 %h3,%h4;",
        "@%p bra skip;\nmov.b16 %h3,0x8000;\nskip:",
        "{\n.reg .b16 %h<6>;\nmov.b16 %h3,0x8000;\n}",
    ] {
        assert_eq!(count(&run(setup, ""), "fma.rm.f32 "), 1, "{setup}");
    }
}

#[test]
fn ue8m0_values_become_bf16() {
    let lowered = lower("{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\tcvt.rn.bf16x2.ue8m0x2 %r1, %rs1;\n\tret;\n}");
    assert_eq!(count(&lowered, "ue8m0"), 0);
    // The byte is the exponent field of its bf16; byte 0 is 2^-127 (a subnormal: bit 6 of the
    // mantissa), byte 0xff a NaN with the whole mantissa.
    assert!(lowered.contains(", 7;"));
    assert!(lowered.contains(", 15;"));
    assert_eq!(count(&lowered, ", 0x40, 0, %vf_n;"), 2);
    assert_eq!(count(&lowered, ", 0x7f, "), 2);
    assert!(lowered.contains("\tmov.b32 %r1, %vf_r"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn ue8m0_from_bf16x2() {
    let lowered =
        lower("{\n\t.reg .b32 %r1;\n\t.reg .b16 %rs1;\n\tcvt.rp.satfinite.ue8m0x2.bf16x2 %rs1, %r1;\n\tret;\n}");
    assert_eq!(count(&lowered, "ue8m0"), 0);
    assert_eq!(count(&lowered, "satfinite"), 0);
    // Each bf16 as the f32 it is, then as from a float.
    assert_eq!(count(&lowered, "abs.f32 "), 2);
    assert_eq!(count(&lowered, ", 0xfe;"), 2);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn an_fp8_product_is_four_fp16_products_of_the_halves() {
    let operands = "{%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}";
    for (a_type, b_type) in [("e4m3", "e4m3"), ("e5m2", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e5m2")] {
        let lowered = lower(&format!(
            "{{\n\tmma.sync.aligned.m16n8k32.row.col.f32.{a_type}.{b_type}.f32 {operands};\n\tret;\n}}"
        ));
        assert_eq!(count(&lowered, "m16n8k32"), 0);
        // Eight of Volta's products, and nothing computed outside the tensor cores.
        assert_eq!(count(&lowered, VOLTA), 8, "{a_type} {b_type}");
        assert_eq!(count(&lowered, "fma."), 0);
        // The bytes are decoded where they are: six registers, two pairs each; an e4m3 pair by
        // one multiplication of its two halves, an e5m2 pair by one permutation of bytes.
        let multiplied = 2 * (4 * (a_type == "e4m3") as usize + 2 * (b_type == "e4m3") as usize);
        assert_eq!(count(&lowered, "mul.rn.f16x2 "), multiplied, "{a_type} {b_type}");
        assert_eq!(count(&lowered, "prmt.b32 "), 12, "{a_type} {b_type}");
        assert_eq!(count(&lowered, "mul.f32 "), 0);
        // The shuffles are those of four FP16 products: five for each B operand.
        assert_eq!(count(&lowered, "shfl."), 4 * 5, "{a_type} {b_type}");
        assert_eq!(undeclared(&lowered), Vec::<String>::new());
    }
    // Operands used by two products are decoded once, and the accumulators stay in Volta layout.
    let twice = lower(&format!(
        "{{\n\tmma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {operands};\n\
             \tmma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%d0, %d1, %d2, %d3}};\n\tret;\n}}"
    ));
    assert_eq!(count(&twice, VOLTA), 16);
    assert_eq!(count(&twice, "mul.rn.f16x2 "), 12);
    assert_eq!(count(&twice, "shfl."), 4 * 5);
    assert_eq!(undeclared(&twice), Vec::<String>::new());
}

#[test]
fn products_of_what_a_half_does_not_hold_are_sums_in_each_thread() {
    let four = "{%d0, %d1, %d2, %d3}";
    for (opcode, operands, terms, shuffles, kind) in [
        // Two blocks of k: for each, B from 8 lanes and two A registers from 4 lanes.
        (
            "m16n8k16.row.col.f32.bf16.bf16.f32",
            format!("{four}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%c0, %c1, %c2, %c3}}"),
            16,
            32,
            "f32",
        ),
        (
            "m16n8k8.row.col.f32.bf16.bf16.f32",
            format!("{four}, {{%a0, %a1}}, {{%b0}}, {{%c0, %c1, %c2, %c3}}"),
            8,
            16,
            "f32",
        ),
        (
            "m16n8k8.row.col.f32.tf32.tf32.f32",
            format!("{four}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%c0, %c1, %c2, %c3}}"),
            8,
            32,
            "f32",
        ),
        (
            "m16n8k4.row.col.f32.tf32.tf32.f32",
            format!("{four}, {{%a0, %a1}}, {{%b0}}, {{%c0, %c1, %c2, %c3}}"),
            4,
            16,
            "f32",
        ),
    ] {
        let lowered = lower(&format!("{{\n\tmma.sync.aligned.{opcode} {operands};\n\tret;\n}}"));
        assert_eq!(count(&lowered, "mma."), 0, "{opcode}");
        assert_eq!(count(&lowered, &format!("fma.rn.{kind} ")), 4 * terms, "{opcode}");
        assert_eq!(count(&lowered, "shfl.sync.idx.b32"), shuffles, "{opcode}");
        // Each sum starts from its accumulator, and the results are written at the end.
        for register in 0..4 {
            assert_eq!(count(&lowered, &format!(", %c{register};")), 1, "{opcode}");
            assert!(
                lowered.contains(&format!("\tmov.b32 %d{register}, %vf_r{register};")),
                "{opcode}"
            );
        }
        let first_result = lowered.find("mov.b32 %d0").unwrap();
        assert!(lowered.rfind("fma.").unwrap() < first_result, "{opcode}");
        assert_eq!(undeclared(&lowered), Vec::<String>::new(), "{opcode}");
    }
    // bf16: the elements are the halves of the registers, as the upper halves of f32.
    let bf16 = lower(&format!(
        "{{\n\tmma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {four}, {{%a0, %a1}}, {{%b0}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    assert_eq!(count(&bf16, "shl.b32 "), 4 * 4 + 1);
    assert_eq!(count(&bf16, "0xffff0000"), 4 * 4);
    // f64: one half of the rows, registers of 64 bits gathered as two words.
    let f64 = lower(
        "{\n\tmma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd0, %fd1}, {%fa}, {%fb}, {%fc0, %fc1};\n\tret;\n}",
    );
    assert_eq!(count(&f64, "mma."), 0);
    assert_eq!(count(&f64, "fma.rn.f64 "), 8);
    assert_eq!(count(&f64, "shfl.sync.idx.b32"), 2 * (4 + 8));
    assert_eq!(count(&f64, "mov.b64 {"), 2);
    assert!(f64.contains("\tmov.b64 %fd0, %vf_q0;") && f64.contains("\tmov.b64 %fd1, %vf_q1;"));
    assert_eq!(undeclared(&f64), Vec::<String>::new());
    // A result that is an operand: nothing of the program is written before the last sum.
    let aliased = lower(
        "{\n\tmma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%b0, %a1, %d2, %d3}, {%a0, %a1}, {%b0}, {%a1, %b0, %c2, %c3};\n\tret;\n}",
    );
    assert!(aliased.rfind("fma.").unwrap() < aliased.find("mov.b32 %b0").unwrap());
    // bf16 operands used by two products are gathered once, and their elements made once.
    let again = lower(&format!(
        "{{\n\tmma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {four}, {{%a0, %a1}}, {{%b0}}, {{%c0, %c1, %c2, %c3}};\n\
             \tmma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {{%e0, %e1, %e2, %e3}}, {{%a0, %a1}}, {{%b0}}, {{%d0, %d1, %d2, %d3}};\n\tret;\n}}"
    ));
    assert_eq!(count(&again, "fma.rn.f32 "), 2 * 32);
    assert_eq!(count(&again, "shfl.sync.idx.b32"), 16);
    assert_eq!(count(&again, "0xffff0000"), 16);
    // An operand used by two products is gathered once.
    let twice = lower(&format!(
        "{{\n\tmma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {four}, {{%a0, %a1}}, {{%b0}}, {{%c0, %c1, %c2, %c3}};\n\
             \tmma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {{%e0, %e1, %e2, %e3}}, {{%a2, %a3}}, {{%b0}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    assert_eq!(count(&twice, "shfl.sync.idx.b32"), 8 + 4 * 4);
    // A tf32 operand is taken without the 13 low bits of its mantissa, as the hardware takes
    // it: cut once for each register, whatever the number of products that read it.
    assert_eq!(count(&twice, ", 0xffffe000;"), 4 + 1);
    assert!(twice.contains("and.b32 %vf_r"));
    assert!(!twice.contains("shfl.sync.idx.b32 %vf_r0, %a0"));
}

#[test]
fn pairs_of_floats_are_computed_one_float_at_a_time() {
    let function = |statement: &str| format!("{{\n\t.reg .b64 %rd<5>;\n\t{statement}\n\tret;\n}}");
    for (statement, single, sources) in [
        ("fma.rn.f32x2 %rd1, %rd2, %rd3, %rd1;", "fma.rn.f32", 3),
        ("mul.rm.f32x2 %rd1, %rd2, %rd3;", "mul.rm.f32", 2),
        ("add.rn.ftz.f32x2 %rd1, %rd2, %rd3;", "add.rn.ftz.f32", 2),
        ("sub.f32x2 %rd1, %rd2, %rd3;", "sub.f32", 2),
    ] {
        let lowered = lower(&function(statement));
        assert_eq!(count(&lowered, "f32x2"), 0, "{statement}");
        // Each operand apart in two words, the two results together again.
        assert_eq!(count(&lowered, "mov.b64 {"), sources, "{statement}");
        assert_eq!(count(&lowered, &format!("{single} %vf_r")), 2, "{statement}");
        assert!(lowered.contains("\tmov.b64 %rd1, {%vf_r"), "{statement}");
        // The first words together, and the second words.
        let first = lowered.lines().find(|line| line.trim().starts_with(single)).unwrap();
        assert!(first.contains("%vf_r0, %vf_r2"), "{first}");
        assert_eq!(undeclared(&lowered), Vec::<String>::new(), "{statement}");
    }
    // The guard is on the statement that writes the register of the program.
    let guarded = lower(&function("@%p1 mul.rn.f32x2 %rd1, %rd2, %rd3;"));
    assert!(guarded.contains("\t@%p1 mov.b64 %rd1, {%vf_r"));
    assert_eq!(count(&guarded, "@%p1 "), 1);
}

#[test]
fn what_a_load_says_to_the_second_level_cache_is_dropped() {
    let function = |statement: &str| format!("{{\n\t.reg .b32 %r<3>;\n\t.reg .b64 %rd<4>;\n\t{statement}\n\tret;\n}}");
    for (statement, plain) in [
        (
            "ld.global.L2::cache_hint.u32 %r1, [%rd1], %rd2;",
            "ld.global.u32 %r1, [%rd1];",
        ),
        (
            "ld.global.nc.L2::cache_hint.v2.f64 {%fd1, %fd2}, [%rd1+16], %rd2;",
            "ld.global.nc.v2.f64 {%fd1, %fd2}, [%rd1+16];",
        ),
        (
            "st.global.L2::cache_hint.b32 [%rd1], %r1, %rd2;",
            "st.global.b32 [%rd1], %r1;",
        ),
        (
            "@%p1 ld.global.L2::256B.L2::cache_hint.f32 %f1, [%rd1], %rd2;",
            "@%p1 ld.global.f32 %f1, [%rd1];",
        ),
        (
            "@!%p1 ld.global.L1::evict_first.L2::cache_hint.b16 %h1, [%rd1], %rd2;",
            "@!%p1 ld.global.L1::evict_first.b16 %h1, [%rd1];",
        ),
        // A cache policy is a value nothing else reads.
        (
            "createpolicy.fractional.L2::evict_last.b64 %rd2, 0.5;",
            "mov.b64 %rd2, 0;",
        ),
        (
            "createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %rd2, 1.0;",
            "mov.b64 %rd2, 0;",
        ),
    ] {
        assert_eq!(lower(&function(statement)), function(plain), "{statement}");
    }
    // A load of 256 bits with an eviction priority: two loads without it.
    let wide = lower(&function(
        "ld.global.L2::evict_last.v8.b32 {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, [%rd1];",
    ));
    assert_eq!(count(&wide, "ld.global.v4.b32 "), 2);
    assert_eq!(count(&wide, "L2::"), 0);
    // What Volta has is left alone.
    let first_level = function("ld.global.L1::evict_last.u32 %r1, [%rd1];");
    assert_eq!(lower(&first_level), first_level);
}

#[test]
fn satfinite_is_taken_at_both_places_the_assembler_takes_it() {
    // After the layouts (the manual, and what compilers write for Turing's shape) and after the
    // types (what they write for Ampere's shapes): the same instruction.
    for (shape, operands, products) in [
        ("m8n8k16", "{%d0, %d1}, {%a}, {%b}, {%c0, %c1}", 1),
        (
            "m16n8k16",
            "{%d0, %d1, %d2, %d3}, {%a0, %a1}, {%b}, {%c0, %c1, %c2, %c3}",
            2,
        ),
        (
            "m16n8k32",
            "{%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}",
            4,
        ),
    ] {
        let after_layouts = lower(&format!(
            "{{\n\tmma.sync.aligned.{shape}.row.col.satfinite.s32.u8.s8.s32 {operands};\n\tret;\n}}"
        ));
        let after_types = lower(&format!(
            "{{\n\tmma.sync.aligned.{shape}.row.col.s32.u8.s8.s32.satfinite {operands};\n\tret;\n}}"
        ));
        assert_eq!(after_layouts, after_types, "{shape}");
        assert_eq!(count(&after_layouts, "dp4a.u32.s32 "), 8 * products, "{shape}");
        assert_eq!(count(&after_layouts, "add.sat.s32 "), 2 * products.min(2), "{shape}");
        assert_eq!(count(&after_layouts, "mma."), 0, "{shape}");
    }
    // Anywhere else, or on a product of floats, it is not a form the layer knows.
    for statement in [
        "mma.sync.aligned.m16n8k16.satfinite.row.col.s32.s8.s8.s32 {%d0, %d1, %d2, %d3}, {%a0, %a1}, {%b}, {%c0, %c1, %c2, %c3};",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.satfinite {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3};",
    ] {
        let refused = refusal(&format!("{{\n\t{statement}\n\tret;\n}}"));
        assert!(refused.ends_with("no Volta sequence for this form"), "{refused}");
    }
}

#[test]
fn a_saturating_product_reads_its_accumulators_before_it_writes_a_result() {
    // The two accumulators exchanged with the two results.
    let lowered = lower(
        "{\n\tmma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%d0, %d1}, {%a}, {%b}, {%d1, %d0};\n\tret;\n}",
    );
    let lines: Vec<&str> = lowered.lines().map(str::trim).collect();
    let last_read = lines.iter().rposition(|line| line.starts_with("add.sat.s32 ")).unwrap();
    let first_write = lines.iter().position(|line| line.starts_with("mov.b32 %d")).unwrap();
    assert!(last_read < first_write, "{lowered}");
    assert_eq!(count(&lowered, "mov.b32 %d"), 2);
}

#[test]
fn satfinite_integer_uses_saturating_add() {
    // m8n8k16 with satfinite.
    let text =
        "{\n\tmma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%d0, %d1}, {%a}, {%b}, {%c0, %c1};\n\tret;\n}";
    let lowered = lower(text);
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 8);
    assert_eq!(count(&lowered, "add.sat.s32 "), 2);
    // The dot product starts from 0, not from c.
    assert!(lowered.contains(", 0;"));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // m16n8k16 with satfinite.
    let text = format!(
        "{{\n\t{A_S8_K16}.satfinite {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 16);
    assert_eq!(count(&lowered, "add.sat.s32 "), 4);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // m16n8k32 with satfinite: the entire dot product is computed from 0 first.
    let text = format!(
        "{{\n\t{A_S8_K32}.satfinite {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    );
    let lowered = lower(&text);
    // 4 sub-products x 8 dp4a = 32.
    assert_eq!(count(&lowered, "dp4a.s32.s32 "), 32);
    assert_eq!(count(&lowered, "add.sat.s32 "), 4);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn ampere_integer_b_columns_reused_across_subproducts() {
    // m16n8k16: both sub-products share the same B register, columns gathered once.
    let text =
        format!("{{\n\t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}");
    let lowered = lower(&text);
    // B columns: 8 shuffles, gathered once (not 16).
    let gather_count = count(&lowered, "shfl.sync.idx.b32") - 4 - 4; // minus rows of a0 and a1
    assert_eq!(gather_count, 8);
}

#[test]
fn mutations_of_the_decomposition_change_the_output() {
    // Mutation 1: swapping the A registers for rows 0-7 and 8-15 in m16n8k16 produces
    // different code (the first shuffle gathers from a different register).
    let normal = lower(&format!(
        "{{\n\t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    let swapped = lower(&format!(
        "{{\n\t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%a1, %a0}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    assert_ne!(normal, swapped);
    // The first row-gathering shuffle reads a different register.
    let first_row_shuffle = |text: &str| -> String {
        text.lines()
            .find(|line| line.contains("shfl.sync.idx.b32") && line.contains("%a"))
            .unwrap_or("")
            .to_string()
    };
    let n_shfl = first_row_shuffle(&normal);
    let s_shfl = first_row_shuffle(&swapped);
    assert_ne!(
        n_shfl, s_shfl,
        "swapping A registers should gather from different sources"
    );
    assert!(n_shfl.contains("%a0"), "normal gathers from %a0 first");
    assert!(s_shfl.contains("%a1"), "swapped gathers from %a1 first");

    // Mutation 2: swapping B registers in m16n8k32 changes which columns are used first.
    let normal_k32 = lower(&format!(
        "{{\n\t{A_S8_K32} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    let swapped_k32 = lower(&format!(
        "{{\n\t{A_S8_K32} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b1, %b0}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    assert_ne!(normal_k32, swapped_k32);

    // Mutation 3: satfinite uses add.sat.s32, non-satfinite does not.
    let without_sat = lower(&format!(
        "{{\n\t{A_S8_K16} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    let with_sat = lower(&format!(
        "{{\n\t{A_S8_K16}.satfinite {{%d0, %d1, %d2, %d3}}, {{%a0, %a1}}, {{%b}}, {{%c0, %c1, %c2, %c3}};\n\tret;\n}}"
    ));
    assert_eq!(count(&without_sat, "add.sat.s32"), 0);
    assert_eq!(count(&with_sat, "add.sat.s32"), 4);
    assert_ne!(without_sat, with_sat);
}

#[test]
fn bf16_arithmetic_through_f32() {
    let bf16_fn = |statement: &str| format!("{{\n\t.reg .b16 %rs<10>;\n\t{statement}\n\tret;\n}}");
    let bf16x2_fn = |statement: &str| format!("{{\n\t.reg .b32 %r<10>;\n\t{statement}\n\tret;\n}}");
    // neg.bf16: flip the sign bit.
    let neg = lower(&bf16_fn("neg.bf16 %rs1, %rs2;"));
    assert_eq!(count(&neg, "neg.bf16"), 0);
    assert_eq!(count(&neg, "xor.b32"), 1);
    assert!(neg.contains("0x8000"));
    assert_eq!(undeclared(&neg), Vec::<String>::new());
    // add.bf16: widen, add in f32, round to bf16.
    let add = lower(&bf16_fn("add.bf16 %rs1, %rs2, %rs3;"));
    assert_eq!(count(&add, "add.bf16"), 0);
    assert_eq!(count(&add, "add.rn.f32 "), 1);
    assert_eq!(count(&add, "cvt.u32.u16 "), 2);
    assert_eq!(count(&add, "shl.b32 "), 2); // two widens (rounding uses shr, not shl)
    assert_eq!(undeclared(&add), Vec::<String>::new());
    // mul.bf16: widen, multiply in f32, round to bf16.
    let mul = lower(&bf16_fn("mul.bf16 %rs1, %rs2, %rs3;"));
    assert_eq!(count(&mul, "mul.bf16"), 0);
    assert_eq!(count(&mul, "mul.rn.f32 "), 1);
    assert_eq!(undeclared(&mul), Vec::<String>::new());
    // fma.rn.bf16: widen, fma in f32 rounded to odd (from the two directed roundings), round to bf16.
    let fma = lower(&bf16_fn("fma.rn.bf16 %rs1, %rs2, %rs3, %rs4;"));
    assert_eq!(count(&fma, "fma.rn.bf16"), 0);
    assert_eq!(count(&fma, "fma.rn.f32 "), 0);
    assert_eq!((count(&fma, "fma.rm.f32 "), count(&fma, "fma.rp.f32 ")), (1, 1));
    assert_eq!(undeclared(&fma), Vec::<String>::new());
    // add.bf16x2: two scalar additions.
    let add2 = lower(&bf16x2_fn("add.bf16x2 %r1, %r2, %r3;"));
    assert_eq!(count(&add2, "add.bf16x2"), 0);
    assert_eq!(count(&add2, "add.rn.f32 "), 2);
    assert_eq!(undeclared(&add2), Vec::<String>::new());
    // fma.rn.bf16x2: two scalar fmas.
    let fma2 = lower(&bf16x2_fn("fma.rn.bf16x2 %r1, %r2, %r3, %r4;"));
    assert_eq!(count(&fma2, "fma.rn.bf16x2"), 0);
    assert_eq!((count(&fma2, "fma.rm.f32 "), count(&fma2, "fma.rp.f32 ")), (2, 2));
    assert_eq!(undeclared(&fma2), Vec::<String>::new());
    // cvt.rn.bf16.f64: through f32.
    let cvt64 = lower("{\n\t.reg .b16 %rs1;\n\t.reg .f64 %fd1;\n\tcvt.rn.bf16.f64 %rs1, %fd1;\n\tret;\n}");
    assert_eq!(count(&cvt64, "bf16.f64"), 0);
    assert_eq!(count(&cvt64, "cvt.rn.f32.f64 "), 0);
    assert_eq!(
        (count(&cvt64, "cvt.rm.f32.f64 "), count(&cvt64, "cvt.rp.f32.f64 ")),
        (1, 1)
    );
    assert_eq!(undeclared(&cvt64), Vec::<String>::new());
    // setp.eq.bf16: widen and compare.
    let setp = lower("{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .pred %p1;\n\tsetp.eq.bf16 %p1, %rs1, %rs2;\n\tret;\n}");
    assert_eq!(count(&setp, "setp.eq.bf16"), 0);
    assert_eq!(count(&setp, "setp.eq.f32 "), 1);
    assert_eq!(undeclared(&setp), Vec::<String>::new());
    // set.eq.bf16x2.bf16x2: compare pairs, produce bf16x2 result.
    let set = lower(&bf16x2_fn("set.eq.bf16x2.bf16x2 %r1, %r2, %r3;"));
    assert_eq!(count(&set, "set.eq"), 0);
    assert_eq!(count(&set, "setp.eq.f32 "), 2);
    assert!(set.contains("0x3f800000"));
    assert_eq!(undeclared(&set), Vec::<String>::new());
}

#[test]
fn bf16_atomic_add_uses_cas_loop() {
    let bf16_atom =
        lower("{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}");
    assert_eq!(count(&bf16_atom, "atom.add.noftz"), 0);
    assert_eq!(count(&bf16_atom, "atom.cas.b16 "), 1);
    assert!(bf16_atom.contains("$vf_at0:"));
    assert!(bf16_atom.contains("bra $vf_at0;"));
    assert_eq!(undeclared(&bf16_atom), Vec::<String>::new());
    // bf16x2 atomic: one compare-and-swap of the whole word.
    let bf16x2_atom =
        lower("{\n\t.reg .b32 %r1, %r2;\n\t.reg .b64 %rd1;\n\tatom.add.noftz.bf16x2 %r1, [%rd1], %r2;\n\tret;\n}");
    assert_eq!(count(&bf16x2_atom, "atom.add.noftz"), 0);
    assert_eq!(count(&bf16x2_atom, "atom.cas.b32 "), 1, "{bf16x2_atom}");
    assert_eq!(count(&bf16x2_atom, "atom.cas.b16 "), 0, "{bf16x2_atom}");
    assert!(bf16x2_atom.contains("$vf_at0:"));
    assert_eq!(undeclared(&bf16x2_atom), Vec::<String>::new());
    // With a space qualifier.
    let global_atom = lower(
        "{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom.global.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}",
    );
    assert_eq!(count(&global_atom, "atom.global.cas.b16 "), 1);
    assert!(global_atom.contains("ld.global.b16 "));
    assert_eq!(undeclared(&global_atom), Vec::<String>::new());
    // Under a guard, a thread whose predicate is false must not touch the memory: it jumps over
    // the whole loop (and the label it jumps to has a statement to stand on).
    for (guard, jump) in [("@%p1", "@!%p1 bra $vf_as0;"), ("@!%p1", "@%p1 bra $vf_as0;")] {
        let guarded = lower(&format!(
            "{{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\t.reg .pred %p1;\n\t{guard} atom.add.noftz.bf16 %rs1, [%rd1], %rs2;\n}}"
        ));
        assert!(guarded.contains(jump), "{guarded}");
        assert!(guarded.find(jump) < guarded.find("ld.b16 "), "{guarded}");
        let after = &guarded[guarded.find("$vf_as0:").expect("the label") + "$vf_as0:".len()..];
        assert!(after.contains("mov.b32 "), "{guarded}");
        assert_eq!(undeclared(&guarded), Vec::<String>::new());
    }
}

#[test]
fn bf16_atomic_preserves_memory_order_and_scope() {
    // Every sem x scope combination on the scalar form.
    let sems = ["", "relaxed", "acquire", "release", "acq_rel"];
    let scopes = ["", "cta", "gpu", "sys"];
    for sem in &sems {
        for scope in &scopes {
            let sem_dot = if sem.is_empty() {
                String::new()
            } else {
                format!(".{sem}")
            };
            let scope_dot = if scope.is_empty() {
                String::new()
            } else {
                format!(".{scope}")
            };
            let source = format!(
                "{{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom{sem_dot}{scope_dot}.global.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}}"
            );
            let lowered = lower(&source);
            let expected_cas = format!("atom{sem_dot}{scope_dot}.global.cas.b16 ");
            assert_eq!(
                count(&lowered, &expected_cas),
                1,
                "sem={sem:?} scope={scope:?}: expected {expected_cas}\n{lowered}"
            );
            assert_eq!(count(&lowered, "atom.add.noftz"), 0, "sem={sem:?} scope={scope:?}");
            assert_eq!(
                undeclared(&lowered),
                Vec::<String>::new(),
                "sem={sem:?} scope={scope:?}"
            );
        }
    }
    // .cluster widens to .gpu.
    let cluster = lower(
        "{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom.acq_rel.cluster.global.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}",
    );
    assert_eq!(count(&cluster, "atom.acq_rel.gpu.global.cas.b16 "), 1);
    // bf16x2 with sem and scope: one compare-and-swap of the word, with the qualifiers.
    for sem in &sems {
        for scope in &scopes {
            let sem_dot = if sem.is_empty() {
                String::new()
            } else {
                format!(".{sem}")
            };
            let scope_dot = if scope.is_empty() {
                String::new()
            } else {
                format!(".{scope}")
            };
            let source = format!(
                "{{\n\t.reg .b32 %r1, %r2;\n\t.reg .b64 %rd1;\n\tatom{sem_dot}{scope_dot}.global.add.noftz.bf16x2 %r1, [%rd1], %r2;\n\tret;\n}}"
            );
            let lowered = lower(&source);
            let expected_cas = format!("atom{sem_dot}{scope_dot}.global.cas.b32 ");
            assert_eq!(
                count(&lowered, &expected_cas),
                1,
                "bf16x2 sem={sem:?} scope={scope:?}: expected one {expected_cas}\n{lowered}"
            );
            assert_eq!(count(&lowered, "cas.b16 "), 0, "bf16x2 sem={sem:?} scope={scope:?}");
            assert_eq!(
                undeclared(&lowered),
                Vec::<String>::new(),
                "bf16x2 sem={sem:?} scope={scope:?}"
            );
        }
    }
    // Unqualified form (no sem, no scope, no space).
    let plain =
        lower("{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}");
    assert_eq!(count(&plain, "atom.cas.b16 "), 1);
    assert_eq!(undeclared(&plain), Vec::<String>::new());
    // shared space.
    let shared = lower(
        "{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom.release.gpu.shared.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}",
    );
    assert_eq!(count(&shared, "atom.release.gpu.shared.cas.b16 "), 1);
    assert!(shared.contains("ld.shared.b16 "));
    // Generic space (no space qualifier).
    let generic = lower(
        "{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom.acq_rel.sys.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}",
    );
    assert_eq!(count(&generic, "atom.acq_rel.sys.cas.b16 "), 1);
    assert!(generic.contains("ld.b16 "));
    // Guarded bf16x2 with qualifiers.
    let guarded_x2 = lower(
        "{\n\t.reg .b32 %r1, %r2;\n\t.reg .b64 %rd1;\n\t.reg .pred %p1;\n\t@%p1 atom.release.sys.global.add.noftz.bf16x2 %r1, [%rd1], %r2;\n}",
    );
    assert_eq!(count(&guarded_x2, "atom.release.sys.global.cas.b32 "), 1);
    assert!(guarded_x2.contains("@!%p1 bra $vf_as0;"));
    assert_eq!(undeclared(&guarded_x2), Vec::<String>::new());
    // An unknown qualifier is refused.
    let refused = body(
        "{\n\t.reg .b16 %rs1, %rs2;\n\t.reg .b64 %rd1;\n\tatom.seq_cst.global.add.noftz.bf16 %rs1, [%rd1], %rs2;\n\tret;\n}",
        64,
        Options::default(),
    );
    assert!(refused.is_err(), "unknown qualifier should be refused");
    assert!(
        refused.as_ref().unwrap_err().0.contains("unknown qualifier"),
        "{:?}",
        refused
    );
}

#[test]
fn wide_load_does_not_overwrite_its_address() {
    let function = |statement: &str| format!("{{\n\t.reg .b32 %r<3>;\n\t.reg .b64 %rd<4>;\n\t{statement}\n\tret;\n}}");
    // Base aliased with the first destination: the second half must be emitted first.
    let aliased_first = lower(&function("ld.global.v4.u64 {%rd1, %rd2, %rd3, %rd0}, [%rd1];"));
    let lines: Vec<&str> = aliased_first.lines().map(str::trim).collect();
    // The second half (with %rd3, %rd0) must come before the first (with %rd1, %rd2).
    let first_load = lines.iter().position(|l| l.contains("ld.global.v2.u64")).unwrap();
    let second_load = lines.iter().rposition(|l| l.contains("ld.global.v2.u64")).unwrap();
    assert!(first_load < second_load);
    // The first emitted load must be the one at [%rd1+16] (second half).
    assert!(lines[first_load].contains("[%rd1+16]"), "{aliased_first}");
    assert!(lines[second_load].contains("[%rd1+0]"), "{aliased_first}");
    assert_eq!(undeclared(&aliased_first), Vec::<String>::new());

    // Base aliased with the second destination of the first half.
    let aliased_second = lower(&function("ld.global.v4.u64 {%rd0, %rd1, %rd2, %rd3}, [%rd1];"));
    let lines: Vec<&str> = aliased_second.lines().map(str::trim).collect();
    let first_load = lines.iter().position(|l| l.contains("ld.global.v2.u64")).unwrap();
    assert!(
        lines[first_load].contains("[%rd1+16]"),
        "second half first: {aliased_second}"
    );

    // Base aliased with a destination in the second half only: original order is fine.
    let aliased_third = lower(&function("ld.global.v4.u64 {%rd0, %rd2, %rd1, %rd3}, [%rd1];"));
    let lines: Vec<&str> = aliased_third.lines().map(str::trim).collect();
    let first_load = lines.iter().position(|l| l.contains("ld.global.v2.u64")).unwrap();
    assert!(
        lines[first_load].contains("[%rd1+0]"),
        "first half first: {aliased_third}"
    );

    // Base not aliased with any destination.
    let no_alias = lower(&function("ld.global.v4.u64 {%rd0, %rd2, %rd3, %rd1}, [%rd1+16];"));
    // %rd1 is in the second half → original order is fine: first load at [%rd1+16].
    let lines: Vec<&str> = no_alias.lines().map(str::trim).collect();
    let first_load = lines.iter().position(|l| l.contains("ld.global.v2.u64")).unwrap();
    assert!(lines[first_load].contains("[%rd1+16]"), "{no_alias}");

    // Base aliased with both halves: must be refused.
    let both = body(
        &function("ld.global.v4.u64 {%rd1, %rd2, %rd1, %rd3}, [%rd1];"),
        64,
        Options::default(),
    );
    assert!(both.is_err(), "base in both halves");
    assert!(both.as_ref().unwrap_err().0.contains("both halves"), "{:?}", both);

    // With a predicate: guard on both halves.
    let guarded = lower(&function("@%p1 ld.global.v4.u64 {%rd1, %rd2, %rd3, %rd0}, [%rd1];"));
    assert_eq!(count(&guarded, "@%p1 ld.global.v2.u64"), 2, "{guarded}");
    // Second half still emitted first.
    let lines: Vec<&str> = guarded.lines().map(str::trim).collect();
    let first_load = lines.iter().position(|l| l.contains("ld.global.v2.u64")).unwrap();
    assert!(
        lines[first_load].contains("[%rd1+16]"),
        "guarded second half first: {guarded}"
    );

    // v8.b32: same address-aliasing logic.
    let wide32 = lower(&function(
        "ld.global.v8.b32 {%r1, %r2, %r0, %r1, %r2, %r0, %r1, %r2}, [%rd1];",
    ));
    assert_eq!(count(&wide32, "ld.global.v4.b32 "), 2, "{wide32}");

    // Store: order does not matter (no register overwrite).
    let store = lower(&function("st.global.v4.u64 [%rd1], {%rd1, %rd2, %rd3, %rd0};"));
    assert_eq!(count(&store, "st.global.v2.u64 "), 2, "{store}");
}

#[test]
fn inner_block_declarations_are_scoped() {
    // The review's PTX-03 program: outer .b32 %amount, inner .pred %amount, then cp.async
    // uses %amount as a size (should be b32, not pred).
    let scoped = lower(
        "{\n\t.reg .b64 %src;\n\t.reg .b32 %dst, %amount;\n\t.shared .align 16 .b8 buf[16];\n\tld.param.u64 %src, [p];\n\tld.param.u32 %amount, [n];\n\tmov.u32 %dst, buf;\n\t{\n\t\t.reg .pred %amount;\n\t\tsetp.eq.u32 %amount, 0, 0;\n\t}\n\tcp.async.ca.shared.global [%dst], [%src], 4, %amount;\n\tcp.async.wait_all;\n\tret;\n}",
    );
    // After the inner block closes, %amount is .b32 again: cp.async should NOT treat it as
    // a predicate. The output must NOT contain @!%amount (which is the predicate form).
    assert!(
        !scoped.contains("@!%amount"),
        "should not treat %amount as a predicate after inner block: {scoped}"
    );
    // It should use the partial-copy path with setp and amount as an integer.
    assert!(
        scoped.contains("setp.eq.u32 %vf_n, %amount,"),
        "should use amount as integer: {scoped}"
    );

    // Reverse shadowing: inner .b32 over outer .pred.
    let reverse = lower(
        "{\n\t.reg .b64 %src;\n\t.reg .b32 %dst;\n\t.reg .pred %amount;\n\t.shared .align 16 .b8 buf[16];\n\tld.param.u64 %src, [p];\n\tmov.u32 %dst, buf;\n\t{\n\t\t.reg .b32 %amount;\n\t}\n\tcp.async.ca.shared.global [%dst], [%src], 4, %amount;\n\tcp.async.wait_all;\n\tret;\n}",
    );
    // After the inner block, %amount is .pred again: cp.async treats it as the ignore predicate.
    assert!(
        reverse.contains("@!%amount"),
        "should treat %amount as predicate after reverse shadow: {reverse}"
    );

    // Nested twice: inner pred over b32, another inner pred.
    let nested = lower(
        "{\n\t.reg .b64 %src;\n\t.reg .b32 %dst, %amount;\n\t.shared .align 16 .b8 buf[16];\n\tld.param.u64 %src, [p];\n\tld.param.u32 %amount, [n];\n\tmov.u32 %dst, buf;\n\t{\n\t\t.reg .pred %amount;\n\t\t{\n\t\t\t.reg .pred %amount;\n\t\t\tsetp.eq.u32 %amount, 0, 0;\n\t\t}\n\t}\n\tcp.async.ca.shared.global [%dst], [%src], 4, %amount;\n\tcp.async.wait_all;\n\tret;\n}",
    );
    assert!(
        !nested.contains("@!%amount"),
        "should not treat %amount as pred after double nesting: {nested}"
    );

    // Renaming test: the same program with the inner register renamed to %flag must produce
    // the same output up to that name.
    let renamed = lower(
        "{\n\t.reg .b64 %src;\n\t.reg .b32 %dst, %amount;\n\t.shared .align 16 .b8 buf[16];\n\tld.param.u64 %src, [p];\n\tld.param.u32 %amount, [n];\n\tmov.u32 %dst, buf;\n\t{\n\t\t.reg .pred %flag;\n\t\tsetp.eq.u32 %flag, 0, 0;\n\t}\n\tcp.async.ca.shared.global [%dst], [%src], 4, %amount;\n\tcp.async.wait_all;\n\tret;\n}",
    );
    // Both should use the integer path for %amount.
    let scoped_norm = scoped.replace("%amount", "%SAME");
    let renamed_norm = renamed.replace("%amount", "%SAME").replace("%flag", "%amount");
    // After normalization, the cp.async lowering should be identical.
    let scoped_cp = scoped_norm
        .lines()
        .find(|l| l.contains("setp.eq.u32 %vf_n, %SAME,"))
        .unwrap();
    let renamed_cp = renamed_norm
        .lines()
        .find(|l| l.contains("setp.eq.u32 %vf_n, %SAME,"))
        .unwrap();
    assert_eq!(scoped_cp, renamed_cp, "renaming should not change the lowering");

    // Family declarations are also scoped.
    let family = lower(
        "{\n\t.reg .b64 %src;\n\t.reg .b32 %dst, %r<10>;\n\t.shared .align 16 .b8 buf[16];\n\tld.param.u64 %src, [p];\n\tmov.u32 %dst, buf;\n\t{\n\t\t.reg .pred %r<10>;\n\t}\n\tcp.async.ca.shared.global [%dst], [%src], 4, %r3;\n\tcp.async.wait_all;\n\tret;\n}",
    );
    // After the inner block, %r3 is .b32 again (from the outer family).
    assert!(!family.contains("@!%r3"), "family should be scoped: {family}");
}

#[test]
fn inner_family_hides_an_outer_name() {
    // Outside, %r3 is a predicate; in the block it is one of the family %r<5> of integers,
    // and is a predicate again after the block.
    let text = lower(
        "{\n\t.reg .b64 %src;\n\t.reg .b32 %dst;\n\t.reg .pred %r3;\n\tld.param.u64 %src, [p];\n\tmov.u32 %dst, buf;\n\tsetp.eq.u32 %r3, 0, 0;\n\t{\n\t\t.reg .b32 %r<5>;\n\t\tmov.u32 %r3, 4;\n\t\tcp.async.ca.shared.global [%dst], [%src], 4, %r3;\n\t}\n\tcp.async.ca.shared.global [%dst+16], [%src], 4, %r3;\n\tcp.async.wait_all;\n\tret;\n}",
    );
    // In the block: a number of bytes. After it: the predicate that says "copy nothing".
    assert_eq!(count(&text, "setp.eq.u32 %vf_n, %r3,"), 1, "{text}");
    assert_eq!(count(&text, "@!%r3 "), 1, "{text}");
    let inner = text.find("setp.eq.u32 %vf_n, %r3,").unwrap();
    let outer = text.find("@!%r3 ").unwrap();
    assert!(inner < outer, "{text}");
}

#[test]
fn packed_bf16_atomic_at_any_address() {
    // The pair is one word: the compare-and-swap has the memory operand as it is written,
    // whatever it is made of, and the qualifiers of the instruction.
    let function = |statement: &str| format!("{{\n\t.reg .b32 %r<3>;\n\t.reg .b64 %rd<2>;\n\t{statement}\n\tret;\n}}");
    for address in ["[flags]", "[flags+4]", "[%rd1]", "[%rd1+0x10]", "[%rd1+-4]"] {
        let text = lower(&function(&format!("atom.global.add.noftz.bf16x2 %r1, {address}, %r2;")));
        assert_eq!(
            count(&text, &format!("atom.global.cas.b32 %vf_r2, {address},")),
            1,
            "{address}: {text}"
        );
        assert_eq!(count(&text, "cas.b16"), 0, "{address}: {text}");
        assert_eq!(undeclared(&text), Vec::<String>::new(), "{address}: {text}");
    }
    let qualified = lower(&function("atom.acq_rel.sys.global.add.noftz.bf16x2 %r1, [%rd1], %r2;"));
    assert_eq!(count(&qualified, "atom.acq_rel.sys.global.cas.b32 "), 1, "{qualified}");
    let cluster = lower(&function("atom.release.cluster.add.noftz.bf16x2 %r1, [%rd1], %r2;"));
    assert_eq!(count(&cluster, "atom.release.gpu.cas.b32 "), 1, "{cluster}");
}

#[test]
fn xla_bf16_arithmetic_and_three_input_max() {
    let text = lower(
        "{\n\t.reg .b16 %h<3>;\n\t.reg .b32 %r<3>;\n\t.reg .pred %p;\n\
        \t@%p cvt.rn.bf16.f16 %h0, %h1;\n\tabs.bf16 %h0, %h0;\n\
        \tadd.rn.bf16 %h1, %h0, %h2;\n\tsub.rn.bf16 %h1, %h1, %h2;\n\
        \tmul.rn.bf16 %h2, %h0, %h1;\n\t@%p mul.rn.bf16x2 %r1, %r0, %r2;\n\
        \tadd.rn.f32.bf16 %r1, %h2, %r1;\n\tsub.rn.f32.bf16 %r1, %h2, %r1;\n\
        \tmax.NaN.f32 %r0, %r1, %r0, %r2;\n\tret;\n}",
    );
    assert!(text.contains("@%p cvt.u16.u32 %h0,"));
    assert!(text.contains("and.b16 %h0, %h0, 0x7fff;"));
    assert_eq!(count(&text, "mul.rn.f32 "), 3);
    assert!(text.contains("@%p or.b32 %r1,"));
    assert_eq!(count(&text, "max.f32 "), 2);
    assert_eq!(undeclared(&text), Vec::<String>::new());
    assert!(!text.contains(".bf16"));
    let invalid = refusal("{\n\t.reg .b16 %h;\n\tabs.ftz.bf16 %h, %h;\n\tret;\n}");
    assert!(invalid.contains("no Volta sequence for this form"));
}

#[test]
fn three_input_fp32_bounds_preserve_the_guard_and_destination_alias() {
    for opcode in ["min.f32", "max.f32", "min.ftz.f32", "max.ftz.f32"] {
        let input = format!("{{\n.reg .f32 %f<4>;\n.reg .pred %p;\n@%p {opcode} %f0, %f1, %f2, %f0;\n}}");
        let text = lower(&input);
        assert_eq!(text.matches(opcode).count(), 2);
        assert!(text.contains(&format!("@%p {opcode} %f0,")));
        assert!(text.contains(", %f0;"));
        assert!(undeclared(&text).is_empty());
        let native = format!("{{\n.reg .f32 %f<3>;\n{opcode} %f0, %f1, %f2;\n}}");
        assert_eq!(lower(&native), native);
    }
}

#[test]
fn unordered_bf16_predicates_preserve_boolean_operands() {
    for cmp in ["equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan"] {
        for (boolean, extra) in [("", ""), (".and", ", !%p2"), (".or", ", %p2"), (".xor", ", %p2")] {
            let input = format!(
                "{{\n.reg .b16 %h<2>;\n.reg .pred %p<3>;\n@%p0 setp.{cmp}{boolean}.bf16 %p1, %h0, %h1{extra};\n}}"
            );
            let text = lower(&input);
            let compare = text
                .lines()
                .find(|line| line.contains(&format!("@%p0 setp.{cmp}{boolean}.f32 %p1,")))
                .unwrap();
            assert!(compare.ends_with(&format!("{extra};")));
            assert!(undeclared(&text).is_empty());
        }
    }
}

#[test]
fn xla_nan_reductions_and_bf16_bounds() {
    let text = lower(
        "{\n\t.reg .b16 %h<3>;\n\t.reg .b32 %r<2>;\n\t.reg .pred %p;\n\
          \t@%p max.NaN.bf16 %h0, %h1, %h2;\n\tredux.sync.max.NaN.f32 %r0, %r0, -1;\n\tret;\n}",
    );
    assert!(text.contains("@%p cvt.u16.u32 %h0,"));
    assert!(text.contains("vote.sync.any.pred %vf_n, %vf_n, -1;"));
    assert!(text.contains("selp.f32 %r0, 0f7FFFFFFF,"));
    assert_eq!(undeclared(&text), Vec::<String>::new());
}

#[test]
fn triton_scalar_bf16_bounds() {
    // As Triton emits for a relu on bf16 tiles: max.bf16 against a zero register.
    let text = lower(
        "{\n\t.reg .b16 %h<3>;\n\t.reg .pred %p;\n\
          \tmax.bf16 %h0, %h1, %h2;\n\t@%p min.bf16 %h1, %h0, %h2;\n\tret;\n}",
    );
    assert_eq!(count(&text, "max.f32 "), 1);
    assert_eq!(count(&text, "min.f32 "), 1);
    assert!(text.contains("@%p cvt.u16.u32 %h1,"));
    assert!(!text.contains(".bf16"));
    assert!(!text.contains("setp.nan"), "no NaN propagation without .NaN");
    assert_eq!(undeclared(&text), Vec::<String>::new());
}

#[test]
fn triton_half_bounds_go_through_f32() {
    // sm_70 has no max.f16/min.f16 nor their f16x2 forms.
    let text = lower(
        "{\n\t.reg .b16 %h<3>;\n\t.reg .b32 %r<3>;\n\t.reg .pred %p;\n\
          \tmax.f16 %h0, %h1, %h2;\n\tmin.NaN.f16 %h1, %h0, %h2;\n\
          \tmax.f16x2 %r0, %r1, %r2;\n\t@%p min.NaN.f16x2 %r1, %r0, %r2;\n\tret;\n}",
    );
    assert_eq!(count(&text, "max.f32 "), 3);
    assert_eq!(count(&text, "min.f32 "), 3);
    assert_eq!(count(&text, "setp.nan.f32 "), 3, "only the .NaN forms propagate NaN");
    assert!(text.contains("cvt.rn.f16.f32 %h0,"));
    assert!(text.contains("@%p mov.b32 %r1, {"));
    assert!(!text.contains(".f16 %h") && !text.contains(".f16x2 %r"), "{text}");
    assert_eq!(undeclared(&text), Vec::<String>::new());
}

#[test]
fn ninfer_packed_bf16_weights() {
    let source = "{\n\t.reg .b16 %h<3>;\n\t.reg .b32 %r<4>;\n\t.reg .pred %p;\n\
                  \tcvt.rn.bf16x2.e4m3x2 %r0, %h0;\n\tcvt.rn.bf16x2.e5m2x2 %r1, %h1;\n\
                  \t@%p sub.rn.bf16x2 %r0, %r0, %r1;\n\tcvt.rn.f16.bf16 %h2, %h1;\n\tret;\n}";
    let text = lower(source);
    assert_eq!(count(&text, "sub.rn.f32 "), 2);
    assert!(text.contains("@%p or.b32 %r0,"));
    assert!(text.contains("cvt.rn.f16.f32 %h2,"));
    assert!(!text.contains(".bf16"));
    assert_eq!(undeclared(&text), Vec::<String>::new());
}

#[test]
fn block_scaled_mma_is_lowered() {
    // kind::mxf4 (E2M1 operands, UE8M0 scale, scale_vec::2X, m16n8k64).
    let mxf4 = lower(
        "{\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 \
             {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, \
             %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
    );
    // Should produce Volta tensor-core products (the E2M1 nibbles are decoded to halves)
    // and f64 arithmetic for the scales.
    assert!(
        count(&mxf4, "mma.sync.aligned.m8n8k4.") > 0,
        "mxf4 should have Volta products"
    );
    assert!(count(&mxf4, "mul.f64 ") > 0, "mxf4 should use f64 for scales");
    assert!(
        count(&mxf4, "cvt.rn.f32.f64 ") > 0,
        "mxf4 (2X) should use round-to-nearest for f64 to f32"
    );
    assert_eq!(undeclared(&mxf4), Vec::<String>::new());

    // kind::mxf4nvf4 with UE4M3, scale_vec::4X.
    let nvf4 = lower(
        "{\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \
             {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, \
             %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
    );
    assert!(
        count(&nvf4, "mma.sync.aligned.m8n8k4.") > 0,
        "nvf4 should have Volta products"
    );
    assert!(
        count(&nvf4, "mul.f32 ") > 0,
        "nvf4 should use f32 for block scales (exact for UE4M3)"
    );
    // 4X uses truncation (cvt.rz), not round-to-nearest.
    assert!(
        count(&nvf4, "cvt.rz.f32.f64 ") > 0,
        "nvf4 (4X) should use truncation for f64 to f32"
    );
    // 4X uses shuffles to redistribute block data between threads.
    assert!(
        count(&nvf4, "shfl.sync.idx.b32 ") > 0,
        "nvf4 (4X) should shuffle block data"
    );
    assert_eq!(undeclared(&nvf4), Vec::<String>::new());

    // kind::mxf8f6f4 (E4M3 operands, UE8M0 scale, scale_vec::1X, m16n8k32).
    let mxf8 = lower(
        "{\n\tmma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 \
             {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, \
             %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
    );
    assert!(
        count(&mxf8, "mma.sync.aligned.m8n8k4.") > 0,
        "mxf8f6f4 should have Volta products"
    );
    assert!(count(&mxf8, "mul.f64 ") > 0, "mxf8f6f4 should use f64 for scales");
    assert_eq!(undeclared(&mxf8), Vec::<String>::new());
}

#[test]
fn block_scaled_nvf4_with_ue8m0_is_refused() {
    // No reference vectors exist for these forms: the layer refuses them (with its own
    // message) instead of emitting a sequence nothing has checked.
    for vec in ["", ".scale_vec::2X", ".scale_vec::4X"] {
        let body = format!(
            "{{\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale{vec}.f32.e2m1.e2m1.f32.ue8m0 \
                 {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%c0, %c1, %c2, %c3}}, \
                 %sa, {{0, 0}}, %sb, {{0, 0}};\n\tret;\n}}"
        );
        let refused = refusal(&body);
        assert!(
            refused.contains("UE8M0 scales has no Volta sequence"),
            "{vec}: {refused}"
        );
    }
}

#[test]
fn block_scaled_selectors_out_of_range_are_refused() {
    let form = |kind: &str, selector_a: &str, selector_b: &str| {
        format!(
            "{{\n\tmma.sync.aligned.{kind} {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, \
                 {{%c0, %c1, %c2, %c3}}, %sa, {selector_a}, %sb, {selector_b};\n\tret;\n}}"
        )
    };
    let two = "m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0";
    let four = "m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3";
    let one = "m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0";
    // (the byte, the thread) of A, then of B.
    for (kind, a, b) in [
        (two, "{1, 0}", "{0, 0}"),
        (two, "{0, 0}", "{3, 0}"),
        (two, "{4, 0}", "{0, 0}"),
        (two, "{0, 2}", "{0, 0}"),
        (two, "{0, 0}", "{0, 4}"),
        (four, "{2, 0}", "{0, 0}"),
        (four, "{0, 0}", "{1, 0}"),
        (one, "{4, 0}", "{0, 0}"),
        (one, "{0, 0}", "{0, 7}"),
    ] {
        let refused = refusal(&form(kind, a, b));
        assert!(
            refused.ends_with("a scale selector that the instruction does not have"),
            "{kind} {a} {b}: {refused}"
        );
    }
    // The values the instruction has are lowered.
    for (kind, a, b) in [
        (two, "{2, 1}", "{2, 3}"),
        (four, "{0, 1}", "{0, 3}"),
        (one, "{3, 1}", "{1, 2}"),
    ] {
        let lowered = lower(&form(kind, a, b));
        assert!(count(&lowered, "mma.sync.aligned.m8n8k4.") > 0, "{kind} {a} {b}");
    }
}

#[test]
fn block_scaled_single_register_vectors_and_scalar_scales_have_the_same_lowering() {
    let product = "{\n\tmma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 \
        {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3}, %sa,{0,0},%sb,{0,0};\nret;\n}";
    let expected = lower(product);
    assert!(expected.contains("mma.sync.aligned.m8n8k4."));
    for (operand, side) in [("%sa", "scale_a"), ("%sb", "scale_b")] {
        assert_eq!(lower(&product.replace(operand, &format!("{{ {operand} }}"))), expected);
        for invalid in ["{}", "{%x,%y}", "{{%x}}", "{3}"] {
            assert!(refusal(&product.replace(operand, invalid)).contains(side), "{invalid}");
        }
    }
    assert_eq!(
        lower(&product.replace("%sa", "{%sa}").replace("%sb", "{%sb}")),
        expected
    );
}

#[test]
fn block_scaled_selectors_in_constant_u16_registers() {
    let product = "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0
        {%d0,%d1,%d2,%d3},{%a0,%a1,%a2,%a3},{%b0,%b1},{%c0,%c1,%c2,%c3},
        %sa,{%s0,%s1},%sb,{%s0,%s2};";
    let body = |assign: &str, product: &str| format!("{{\n.reg .b16 %s<3>;\n{assign}\n{product}\nret;\n}}");
    let assign = "mov.b16 %s0, 2; mov.u16 %s1, 1; mov.s16 %s2, 3;";
    let literal = product.replace("{%s0,%s1}", "{2,1}").replace("{%s0,%s2}", "{2,3}");
    assert_eq!(lower(&body(assign, product)), lower(&body(assign, &literal)));
    for extra in ["mov.b16 %s0, 0;", "@%p mov.b16 %s2, 0;", "{ .reg .b16 %s1; }"] {
        assert!(refusal(&body(&format!("{assign} {extra}"), product)).contains("selector_"));
    }
    let guarded = assign.replace("mov.u16", "@%p mov.u16");
    assert!(refusal(&body(&guarded, product)).contains("selector_a"));
    let invalid = assign.replace("%s0, 2", "%s0, 1");
    assert!(refusal(&body(&invalid, product)).contains("a scale selector that the instruction does not have"));
}

#[test]
fn ue4m3_outside_a_block_scaled_product_is_refused() {
    let refused = refusal("{\n\t.reg .b32 %r<2>;\n\tabs.ue4m3 %r0, %r1;\n\tret;\n}");
    assert!(refused.ends_with("no Volta sequence for this form"), "{refused}");
}

#[test]
fn block_scaled_emulated_counts() {
    let counts = super::super::emulated::count_emulated(
            b"mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 {%r}, {%a}, {%b}, {%c}, %sa, {0, 0}, %sb, {0, 0};\n",
        );
    assert_eq!(counts.fp4_tensor, 1);
    assert_eq!(counts.fp8_tensor, 0);
    assert_eq!(counts.fp16_tensor, 0);
}

#[test]
fn chain_of_four_uses_single_instruction_arrangement() {
    // Four products on the same accumulator: the chain form uses one m8n8k4.row.col per product
    // instead of two. The chain detection threshold is 4, so this is the minimum chain.
    let product = |a: &str, b: &str| {
        format!("\t{F32} {{%d0, %d1, %d2, %d3}}, {{{a}0, {a}1}}, {{{b}}}, {{%d0, %d1, %d2, %d3}};\n")
    };
    let text = format!(
        "{{\n{}{}{}{}}}",
        product("%a", "%b"),
        product("%e", "%f"),
        product("%g", "%h"),
        product("%i", "%j"),
    );
    let lowered = lower(&text);
    // The chain form uses m8n8k4.row.col (the same as the standard form) but only one per product.
    // 4 products * 1 m8n8k4 for the chain + combination shuffles at the end.
    let volta_count = count(&lowered, "mma.sync.aligned.m8n8k4.");
    assert_eq!(
        volta_count, 4,
        "chain of 4 should emit 4 m8n8k4 (one per product), got {volta_count}"
    );
    // The end of the chain: an exchange with the lane 8 away for each of the four sums the
    // lanes read, then two shuffles a register.
    assert_eq!(count(&lowered, ", 8, 31, -1;"), 4);
    assert_eq!(count(&lowered, "shfl.sync.bfly.b32"), 4);
    for from in ["%vf_co0", "%vf_co1"] {
        assert_eq!(count(&lowered, &format!(", {from}, 31, -1;")), 4, "{from}");
    }
    // Its start: 16 shuffles bring the accumulator of the program, and 6 a product its operands.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 16 + 4 * 6 + 8);
    // The final D registers must be written.
    assert!(lowered.contains("selp.b32 %d0,") || lowered.contains("b32 %d0, "));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn chain_of_two_uses_standard_form() {
    // Two products on the same accumulator: below the chain threshold, standard form is used.
    let product = |a: &str, b: &str| {
        format!("\t{F32} {{%d0, %d1, %d2, %d3}}, {{{a}0, {a}1}}, {{{b}}}, {{%d0, %d1, %d2, %d3}};\n")
    };
    let text = format!("{{\n{}{}}}", product("%a", "%b"), product("%e", "%f"));
    let lowered = lower(&text);
    // Standard form: 2 m8n8k4 per product = 4 total.
    let volta_count = count(&lowered, "mma.sync.aligned.m8n8k4.");
    assert_eq!(
        volta_count, 4,
        "chain of 2 should use standard form with 4 m8n8k4, got {volta_count}"
    );
}

#[test]
fn chain_broken_by_reader_uses_standard_form() {
    // A read of the accumulator between products breaks the chain.
    let product = |a: &str, b: &str| {
        format!("\t{F32} {{%d0, %d1, %d2, %d3}}, {{{a}0, {a}1}}, {{{b}}}, {{%d0, %d1, %d2, %d3}};\n")
    };
    let text = format!(
        "{{\n{}\tst.global.u32 [%rd1], %d0;\n{}{}{}{}\tret;\n}}",
        product("%a", "%b"),
        product("%e", "%f"),
        product("%g", "%h"),
        product("%i", "%j"),
        product("%k", "%l"),
    );
    let lowered = lower(&text);
    // The first product is standalone (standard form: 2 m8n8k4). The store reads %d0, breaking
    // any chain. Products 2-5 form a chain of 4 (4 m8n8k4). Total: 2 + 4 = 6 m8n8k4.
    let volta_count = count(&lowered, "mma.sync.aligned.m8n8k4.");
    assert_eq!(volta_count, 6, "1 standard + 4 chain = 6 m8n8k4, got {volta_count}");
}

#[test]
fn chain_of_sixteen_has_fewer_mma_than_standard() {
    // Sixteen products on the same accumulator: 16 m8n8k4 for the chain vs 32 for standard.
    let products: String = (0..16)
        .map(|n| format!("\t{F32} {{%d0, %d1, %d2, %d3}}, {{%a{n}0, %a{n}1}}, {{%b{n}}}, {{%d0, %d1, %d2, %d3}};\n"))
        .collect();
    let text = format!("{{\n{products}\tret;\n}}");
    let lowered = lower(&text);
    let volta_count = count(&lowered, "mma.sync.aligned.m8n8k4.");
    assert_eq!(
        volta_count, 16,
        "chain of 16: 16 m8n8k4 (one per product), got {volta_count}"
    );
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

/// `count` products of a chain on the accumulator `%{d}0` to `%{d}3`.
fn chain_products(d: &str, count: usize) -> String {
    (0..count)
            .map(|n| {
                format!("\t{F32} {{%{d}0, %{d}1, %{d}2, %{d}3}}, {{%a{n}0, %a{n}1}}, {{%b{n}}}, {{%{d}0, %{d}1, %{d}2, %{d}3}};\n")
            })
            .collect()
}

/// Where the `nth` (from 0) `needle` of `text` is.
fn nth(text: &str, needle: &str, nth: usize) -> usize {
    text.match_indices(needle)
        .nth(nth)
        .unwrap_or_else(|| panic!("no {needle:?} number {nth} in:\n{text}"))
        .0
}

/// What gives the accumulator back to the first register of the program at the end of a chain.
const GIVEN_BACK: &str = "selp.b32 %d0,";
const CHAIN: &str = "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32";

#[test]
fn a_way_out_ends_the_chain_before_it() {
    // Eight products on one accumulator, and a branch after four: the lanes that take it read
    // the accumulator in the registers of the program at the label, and the statement after the
    // branch is a product of the same accumulator, which says nothing of a reader. (So it went
    // wrong on the hardware: `t_chain_exit` of tests/kernels/k_chains.cu.)
    let four = chain_products("d", 4);
    let text = format!("{{\n{four}\t@%p1 bra $L_out;\n{four}$L_out:\n\tst.global.u32 [%rd1], %d0;\n\tret;\n}}");
    let lowered = lower(&text);
    // Two chains of four: a product each.
    assert_eq!(count(&lowered, CHAIN), 8);
    assert_eq!(count(&lowered, "mma.sync.aligned.m8n8k4."), 8);
    // The accumulator is given back twice: before the branch, and before the label.
    assert_eq!(count(&lowered, GIVEN_BACK), 2);
    let branch = nth(&lowered, "@%p1 bra $L_out;", 0);
    assert!(nth(&lowered, CHAIN, 3) < nth(&lowered, GIVEN_BACK, 0));
    assert!(nth(&lowered, GIVEN_BACK, 0) < branch);
    assert!(branch < nth(&lowered, CHAIN, 4));
    assert!(nth(&lowered, GIVEN_BACK, 1) < nth(&lowered, "$L_out:", 0));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn the_back_edge_of_a_loop_ends_the_chain() {
    // A loop entered without a test: no label follows its branch, and what follows is a product
    // of another accumulator, which this module rewrites. The lanes that go round must find the
    // accumulator in the registers of the program at the top.
    let text = format!(
        "{{\n$L_top:\n{}\tadd.u32 %n, %n, 1;\n\tsetp.lt.u32 %p1, %n, %turns;\n\t@%p1 bra $L_top;\n{}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
        chain_products("d", 4),
        chain_products("e", 1),
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 4);
    assert_eq!(count(&lowered, VOLTA_F32), 2);
    assert_eq!(count(&lowered, GIVEN_BACK), 1);
    let end = nth(&lowered, GIVEN_BACK, 0);
    assert!(nth(&lowered, CHAIN, 3) < end);
    assert!(end < nth(&lowered, "@%p1 bra $L_top;", 0));
    // (And so before the statements between the last product and the branch, which do not name
    // the accumulator.)
    assert!(end < nth(&lowered, "add.u32 %n, %n, 1;", 0));
    // An unconditional branch and a branch through a table likewise.
    for branch in ["bra.uni $L_top;", "brx.idx %n, $L_table;"] {
        let lowered = lower(&text.replace("@%p1 bra $L_top;", branch));
        assert!(nth(&lowered, GIVEN_BACK, 0) < nth(&lowered, branch, 0), "{branch}");
    }
}

#[test]
fn a_chain_does_not_go_through_the_bounds_of_a_scope() {
    // A name may be another register inside a scope: here `%d0` to `%d3`, declared again as a
    // family, which names none of them. The accumulator is given back before the scope opens,
    // not at the statement that names `%d0` inside it (after a product of another accumulator,
    // which this module rewrites: what is written at that statement comes after it).
    let four = chain_products("d", 4);
    let text = format!(
        "{{\n{four}\t{{\n\t.reg .b32 %d<4>;\n{}\tmov.b32 %d0, %other;\n\t}}\n\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
        chain_products("e", 1),
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 4);
    assert_eq!(count(&lowered, GIVEN_BACK), 1);
    assert!(nth(&lowered, GIVEN_BACK, 0) < nth(&lowered, "\t{\n\t.reg .b32 %d<4>;", 0));
    // Products inside a scope: their chain ends inside it.
    let text = format!(
        "{{\n\t{{\n\t.reg .b32 %d<4>;\n{four}\t}}\n{}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
        chain_products("e", 1),
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 4);
    assert_eq!(count(&lowered, GIVEN_BACK), 1);
    assert!(nth(&lowered, GIVEN_BACK, 0) < nth(&lowered, "\t}\n", 0));
}

#[test]
fn an_accumulator_is_named_whole() {
    // `%d0` is not named by a statement that names `%d00` or `%d01`: the chain goes on.
    let two = chain_products("d", 2);
    let text = format!("{{\n{two}\tadd.u32 %d00, %d01, 1;\n{two}\tret;\n}}");
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 4);
    assert_eq!(count(&lowered, "mma.sync.aligned.m8n8k4."), 4);
    assert_eq!(count(&lowered, GIVEN_BACK), 1);
    // Named, it ends the chain: two products on each side are not a chain.
    let lowered = lower(&text.replace("%d01, 1;", "%d1, 1;"));
    assert_eq!(count(&lowered, CHAIN), 0);
    assert_eq!(count(&lowered, VOLTA_F32), 8);
}

#[test]
fn a_statement_under_a_predicate_that_names_the_accumulator_ends_the_chain() {
    let text = format!(
        "{{\n{}\t@%p1 mov.b32 %d2, %other;\n{}\tret;\n}}",
        chain_products("d", 2),
        chain_products("d", 4)
    );
    let lowered = lower(&text);
    // Two products that are not a chain, then a chain of four.
    assert_eq!(count(&lowered, VOLTA_F32), 4);
    assert_eq!(count(&lowered, CHAIN), 4);
    assert!(nth(&lowered, "@%p1 mov.b32 %d2, %other;", 0) < nth(&lowered, CHAIN, 0));
    // After a chain: the accumulator is given back before the statement.
    let text = format!(
        "{{\n{}\t@%p1 mov.b32 %d2, %other;\n{}\tret;\n}}",
        chain_products("d", 4),
        chain_products("d", 4)
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 8);
    assert_eq!(count(&lowered, GIVEN_BACK), 2);
    assert!(nth(&lowered, GIVEN_BACK, 0) < nth(&lowered, "@%p1 mov.b32 %d2, %other;", 0));
    assert!(nth(&lowered, "@%p1 mov.b32 %d2, %other;", 0) < nth(&lowered, CHAIN, 4));
}

#[test]
fn a_product_that_names_the_accumulator_of_another_chain_ends_it() {
    // Two accumulators with two registers in common: `%d0` to `%d3`, then `%d2` to `%d5`.
    let second: String = (0..4)
        .map(|n| format!("\t{F32} {{%d2, %d3, %d4, %d5}}, {{%e{n}0, %e{n}1}}, {{%f{n}}}, {{%d2, %d3, %d4, %d5}};\n"))
        .collect();
    let text = format!(
        "{{\n{}{second}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
        chain_products("d", 4)
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 8);
    // The first chain gives `%d0` to `%d3` back before the first product of the second, which
    // reads `%d2` and `%d3`.
    assert!(nth(&lowered, GIVEN_BACK, 0) < nth(&lowered, CHAIN, 4));
    assert!(nth(&lowered, "selp.b32 %d3,", 0) < nth(&lowered, CHAIN, 4));
    // In turn, neither is ever a chain: each product ends the other's.
    let turns: String = (0..4)
        .map(|n| {
            format!(
                "\t{F32} {{%d0, %d1, %d2, %d3}}, {{%a{n}0, %a{n}1}}, {{%b{n}}}, {{%d0, %d1, %d2, %d3}};\n\
                     \t{F32} {{%d2, %d3, %d4, %d5}}, {{%e{n}0, %e{n}1}}, {{%f{n}}}, {{%d2, %d3, %d4, %d5}};\n"
            )
        })
        .collect();
    let lowered = lower(&format!("{{\n{turns}\tret;\n}}"));
    assert_eq!(count(&lowered, CHAIN), 0);
    assert_eq!(count(&lowered, VOLTA_F32), 16);
}

#[test]
fn a_product_that_reads_its_accumulator_as_an_operand_has_it_given_back_first() {
    // The second product of four takes `%d0`, a register of the accumulator, as its A (then
    // `%d1` as its B): the registers of the program must hold the accumulator before it.
    for (operand, with) in [("{%a10, %a11}", "{%d0, %a11}"), ("{%b1}", "{%d1}")] {
        let products = chain_products("d", 4);
        let text = format!(
            "{{\n{}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
            products.replacen(operand, with, 1)
        );
        assert_ne!(text, format!("{{\n{products}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}"));
        let lowered = lower(&text);
        // Given back after the first product and before anything of the second reads it.
        let given = nth(&lowered, GIVEN_BACK, 0);
        assert!(nth(&lowered, CHAIN, 0) < given, "{with}");
        assert!(given < nth(&lowered, CHAIN, 1), "{with}");
        // Every shuffle that reads the register after the first product (the chain begins
        // by reading its accumulator from the registers of the program) comes after it is
        // given back: the operand of the second product, and the accumulator again.
        let register = &with[1..4];
        let first = nth(&lowered, CHAIN, 0);
        let reads: Vec<usize> = lowered
            .match_indices(&format!(", {register},"))
            .map(|(at, _)| at)
            .filter(|at| *at > first && lowered[..*at].rsplit('\n').next().unwrap().contains("shfl.sync"))
            .collect();
        assert!(!reads.is_empty(), "{with}");
        assert!(
            reads
                .iter()
                .all(|at| *at > nth(&lowered, &format!("selp.b32 {register},"), 0)),
            "{with}"
        );
        // And again at the end, for the reader: twice in all.
        assert_eq!(count(&lowered, GIVEN_BACK), 2, "{with}");
        assert_eq!(undeclared(&lowered), Vec::<String>::new());
    }
}

#[test]
fn an_f16_product_that_reads_its_accumulator_as_an_operand_has_it_given_back_first() {
    // As its A, as its B, and as a C that has one register of it and not the other.
    let first = half_products("d", 1);
    for second in [
        format!("\t{F16} {{%d0, %d1}}, {{%d0, %a11}}, {{%b1}}, {{%d0, %d1}};\n"),
        format!("\t{F16} {{%d0, %d1}}, {{%a10, %a11}}, {{%d1}}, {{%d0, %d1}};\n"),
        format!("\t{F16} {{%d0, %d1}}, {{%a10, %a11}}, {{%b1}}, {{%d1, %e0}};\n"),
    ] {
        let lowered = lower(&format!("{{\n{first}{second}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}"));
        assert_eq!(count(&lowered, VOLTA_F16), 4, "{second}");
        // Given back between the two products (the Volta products 1 and 2, from 0).
        let given = nth(&lowered, "selp.b32 %d1,", 0);
        assert!(nth(&lowered, VOLTA_F16, 1) < given, "{second}");
        assert!(given < nth(&lowered, VOLTA_F16, 2), "{second}");
        assert_eq!(undeclared(&lowered), Vec::<String>::new());
    }
    // A C that is another accumulator altogether: nothing of `%d0`, `%d1` is read, and
    // nothing is given back between (the product writes them).
    let other = format!("\t{F16} {{%d0, %d1}}, {{%a10, %a11}}, {{%b1}}, {{%e0, %e1}};\n");
    let lowered = lower(&format!("{{\n{first}{other}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}"));
    assert_eq!(count(&lowered, "selp.b32 %d1,"), 1);
    assert!(nth(&lowered, VOLTA_F16, 3) < nth(&lowered, "selp.b32 %d1,", 0));
}

#[test]
fn a_family_declared_again_is_not_known_to_be_zero() {
    // The accumulators of the outer family hold 1.0; a scope declares the family again and
    // sets its own registers to zero. After it the names are the outer registers again.
    let ones: String = (0..4).map(|n| format!("\tmov.b32 %d{n}, 0x3f800000;\n")).collect();
    let zeros: String = (0..4).map(|n| format!("\tmov.b32 %d{n}, 0;\n")).collect();
    let shadow = format!(
        "{{\n\t.reg .b32 %d<4>;\n{ones}\t{{\n\t.reg .b32 %d<4>;\n{zeros}\t}}\n{}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
        chain_products("d", 4)
    );
    let fetched = 16 + 4 * 6 + 8;
    let lowered = lower(&shadow);
    assert_eq!(count(&lowered, CHAIN), 4);
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), fetched);
    // The same with another name for the inner family, which was right before.
    let renamed = shadow
        .replacen("\t{\n\t.reg .b32 %d<4>;", "\t{\n\t.reg .b32 %inner<4>;", 1)
        .replace(&zeros, &zeros.replace("%d", "%inner"));
    assert_ne!(renamed, shadow);
    assert_eq!(count(&lower(&renamed), "shfl.sync.idx.b32"), fetched);
    // The other way: the outer registers are zero, and the chain is on the inner family,
    // declared after them and not set.
    let inner = format!(
        "{{\n\t.reg .b32 %d<4>;\n{zeros}\t{{\n\t.reg .b32 %d<4>;\n{}\tst.global.u32 [%rd1], %d0;\n\t}}\n\tret;\n}}",
        chain_products("d", 4)
    );
    assert_eq!(count(&lower(&inner), "shfl.sync.idx.b32"), fetched);
    // And the control: set to zero in the same scope, nothing is fetched.
    let known = format!(
        "{{\n\t.reg .b32 %d<4>;\n{zeros}{}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
        chain_products("d", 4)
    );
    assert_eq!(count(&lower(&known), "shfl.sync.idx.b32"), 4 * 6 + 8);
}

#[test]
fn chains_in_turn_are_written_in_one_order() {
    // Products of three accumulators in turn: three chains open at once, ended together. The
    // text must be the same every time (a hash table has another order in every one).
    let turns: String = (0..4)
            .map(|n| {
                ["d", "e", "f"]
                    .iter()
                    .map(|d| {
                        format!(
                            "\t{F32} {{%{d}0, %{d}1, %{d}2, %{d}3}}, {{%a{n}0, %a{n}1}}, {{%b{n}}}, {{%{d}0, %{d}1, %{d}2, %{d}3}};\n"
                        )
                    })
                    .collect::<String>()
            })
            .collect();
    let text = format!("{{\n{turns}\tret;\n}}");
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 12);
    assert_eq!(count(&lowered, "mma.sync.aligned.m8n8k4."), 12);
    for _ in 0..16 {
        assert_eq!(lower(&text), lowered);
    }
    assert!(nth(&lowered, "selp.b32 %d0,", 0) < nth(&lowered, "selp.b32 %e0,", 0));
    assert!(nth(&lowered, "selp.b32 %e0,", 0) < nth(&lowered, "selp.b32 %f0,", 0));
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn a_chain_from_zero_fetches_nothing() {
    // The accumulator is set to zero through another register, as compilers write it.
    let zeroed = "\tmov.u32 %z, 0;\n\tmov.u32 %d0, %z;\n\tmov.u32 %d1, %z;\n\tmov.u32 %d2, %z;\n\tmov.u32 %d3, %z;\n";
    let text = format!(
        "{{\n{zeroed}{}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
        chain_products("d", 4)
    );
    let lowered = lower(&text);
    assert_eq!(count(&lowered, CHAIN), 4);
    // 6 shuffles a product for its operands and 8 at the end: none to split the accumulator.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4 * 6 + 8);
    let zeros = lowered
        .lines()
        .filter(|line| line.starts_with("\tmov.b32 %vf_r") && line.ends_with(", 0;"))
        .count();
    assert_eq!(zeros, 8);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
    // Not known to be zero: one register is not set, is set under a predicate, is set to
    // another value, or a label stands between (the lanes may come from elsewhere).
    for other in [
        text.replace("\tmov.u32 %d2, %z;\n", ""),
        text.replace("\tmov.u32 %d2, %z;", "\t@%p1 mov.u32 %d2, %z;"),
        text.replace("\tmov.u32 %d2, %z;", "\tmov.u32 %d2, 1;"),
        text.replace("\tmov.u32 %z, 0;", "\tmov.u32 %z, %y;"),
        text.replace("\tmov.u32 %d3, %z;\n", "\tmov.u32 %d3, %z;\n$L_top:\n"),
        text.replace("\tmov.u32 %d3, %z;\n", "\tmov.u32 %d3, %z;\n\tadd.f32 %d3, %d3, %y;\n"),
        text.replace("\tmov.u32 %d1, %z;", "\tmov.u32 %z, %y;\n\tmov.u32 %d1, %z;"),
    ] {
        assert_ne!(other, text);
        let lowered = lower(&other);
        assert_eq!(count(&lowered, CHAIN), 4, "{other}");
        assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 16 + 4 * 6 + 8, "{other}");
    }
    // The constant itself, as an integer or as a float.
    for constant in ["0", "0x0", "0f00000000"] {
        let direct = text.replace("%z;", &format!("{constant};"));
        assert_eq!(count(&lower(&direct), "shfl.sync.idx.b32"), 4 * 6 + 8, "{constant}");
    }
}

/// `count` products with f16 accumulators on `%{d}0` and `%{d}1`.
fn half_products(d: &str, count: usize) -> String {
    (0..count)
        .map(|n| format!("\t{F16} {{%{d}0, %{d}1}}, {{%a{n}0, %a{n}1}}, {{%b{n}}}, {{%{d}0, %{d}1}};\n"))
        .collect()
}

/// The exchange that brings an f16 accumulator into Volta layout, or back from it.
const HALF_EXCHANGE: &str = ", 2, 31, -1;";

#[test]
fn an_f16_accumulator_is_given_back_where_it_is_read() {
    // Four products, then a reader: one exchange to take the accumulator in, one to give it
    // back (not one after every product), and the reader after it.
    let text = format!("{{\n{}\tst.global.u32 [%rd1], %d0;\n\tret;\n}}", half_products("d", 4));
    let lowered = lower(&text);
    assert_eq!(count(&lowered, VOLTA_F16), 8);
    assert_eq!(count(&lowered, HALF_EXCHANGE), 2);
    assert_eq!(count(&lowered, "selp.b32 %d0,"), 1);
    assert_eq!(count(&lowered, "selp.b32 %d1,"), 1);
    assert!(nth(&lowered, VOLTA_F16, 7) < nth(&lowered, "selp.b32 %d0,", 0));
    assert!(nth(&lowered, "selp.b32 %d1,", 0) < nth(&lowered, "st.global.u32 [%rd1], %d0;", 0));
    // Every product goes on from the Volta registers of the one before.
    let products = volta_operands(&lowered);
    for pair in 1..4 {
        assert_eq!(products[2 * pair][3], products[2 * pair - 1][0]);
    }
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // Nothing reads it: it is given back all the same, at the end of the body (the registers
    // may be read by what the body is part of).
    let unread = lower(&format!("{{\n{}\tret;\n}}", half_products("d", 4)));
    assert_eq!(count(&unread, HALF_EXCHANGE), 2);
    assert!(nth(&unread, "selp.b32 %d1,", 0) < nth(&unread, "\tret;", 0));

    // Read between the products: given back there, and the products go on as before.
    let between = lower(&format!(
        "{{\n{}\tst.global.u32 [%rd1], %d1;\n{}\tret;\n}}",
        half_products("d", 2),
        half_products("d", 2)
    ));
    assert_eq!(count(&between, HALF_EXCHANGE), 3);
    assert!(nth(&between, "selp.b32 %d1,", 0) < nth(&between, "st.global.u32 [%rd1], %d1;", 0));
    let products = volta_operands(&between);
    assert_eq!(products[4][3], products[3][0]);

    // Written between the products: given back before (the other register keeps its value),
    // and taken in again after.
    let written = lower(&format!(
        "{{\n{}\tmov.b32 %d1, %other;\n{}\tret;\n}}",
        half_products("d", 2),
        half_products("d", 2)
    ));
    assert_eq!(count(&written, HALF_EXCHANGE), 4);
    assert!(nth(&written, "selp.b32 %d0,", 0) < nth(&written, "mov.b32 %d1, %other;", 0));
    let products = volta_operands(&written);
    assert_ne!(products[4][3], products[3][0]);
}

#[test]
fn an_f16_accumulator_is_given_back_before_the_lanes_may_go_elsewhere() {
    // A branch, a label, the bounds of a scope: as for a chain.
    for (between, mark) in [
        ("\t@%p1 bra $L_out;\n", "@%p1 bra $L_out;"),
        ("$L_in:\n", "$L_in:"),
        ("\t{\n\t.reg .b32 %d<2>;\n\t}\n", "\t{\n\t.reg"),
    ] {
        let text = format!(
            "{{\n{}{between}{}$L_out:\n\tst.global.u32 [%rd1], %d0;\n\tret;\n}}",
            half_products("d", 2),
            half_products("d", 2)
        );
        let lowered = lower(&text);
        assert_eq!(count(&lowered, VOLTA_F16), 8, "{mark}");
        // Given back before the mark, and again before the label of the end.
        assert_eq!(count(&lowered, "selp.b32 %d0,"), 2, "{mark}");
        assert!(nth(&lowered, "selp.b32 %d1,", 0) < nth(&lowered, mark, 0), "{mark}");
        assert!(
            nth(&lowered, VOLTA_F16, 3) < nth(&lowered, "selp.b32 %d0,", 0),
            "{mark}"
        );
        assert!(
            nth(&lowered, "selp.b32 %d1,", 1) < nth(&lowered, "$L_out:", 0),
            "{mark}"
        );
    }
}

#[test]
fn f16_accumulators_in_turn_are_each_given_back_once() {
    // llama.cpp's V product: two accumulators, two products each, and again.
    let turn = format!("{}{}", half_products("d", 2), half_products("e", 2));
    let text = format!("{{\n{turn}{turn}\tst.global.u32 [%rd1], %d0;\n\tst.global.u32 [%rd1], %e1;\n\tret;\n}}");
    let lowered = lower(&text);
    assert_eq!(count(&lowered, VOLTA_F16), 16);
    // In once and out once for each.
    assert_eq!(count(&lowered, HALF_EXCHANGE), 4);
    // Each before its reader. (What is written at a statement comes before the statements
    // since the last one rewritten, which are copied later: here before both readers.)
    assert!(nth(&lowered, VOLTA_F16, 15) < nth(&lowered, "selp.b32 %d0,", 0));
    assert!(nth(&lowered, "selp.b32 %d1,", 0) < nth(&lowered, "st.global.u32 [%rd1], %d0;", 0));
    assert!(nth(&lowered, "selp.b32 %e1,", 0) < nth(&lowered, "st.global.u32 [%rd1], %e1;", 0));
    // The same text every time.
    for _ in 0..8 {
        assert_eq!(lower(&text), lowered);
    }
    // A product that reads the other's registers as its operands ends it first.
    let crossed = format!(
        "{{\n{}\t{F16} {{%e0, %e1}}, {{%d0, %d1}}, {{%b9}}, {{%e0, %e1}};\n\tret;\n}}",
        half_products("d", 2)
    );
    let lowered = lower(&crossed);
    assert!(nth(&lowered, "selp.b32 %d1,", 0) < nth(&lowered, VOLTA_F16, 4));
}

/// A chain of four products whose operands two `ldmatrix` load, with `between` before it and
/// `after` behind it.
fn loaded_chain(between: &str, after: &str) -> String {
    format!(
        "{{\n{DECLARED}\t{LD}.x4.b16 {{%r1, %r2, %r3, %r4}}, [%rd1];\n\t{LD}.x2.b16 {{%r5, %r6}}, [%rd2];\n{between}\
             \t{F32} {D4}, {{%r1, %r2}}, {{%r5}}, {D4};\n\t{F32} {D4}, {{%r3, %r4}}, {{%r6}}, {D4};\n\
             \t{F32} {D4}, {{%r1, %r2}}, {{%r6}}, {D4};\n\t{F32} {D4}, {{%r3, %r4}}, {{%r5}}, {D4};\n{after}\tret;\n}}"
    )
}

#[test]
fn operands_of_a_chain_are_loaded_once() {
    let lowered = lower_aligned(&loaded_chain("", ""));
    assert_eq!(count(&lowered, CHAIN), 4);
    assert_eq!(count(&lowered, "mma."), 4);
    // Two words for each pair of A and each register of B, at an address that one shuffle
    // brings; nothing in the layout of the program, nor in the rows and columns that the
    // products that are not of a chain take.
    assert_eq!(count(&lowered, "ld.shared.v2.u32 "), 4);
    assert_eq!(count(&lowered, "ld.shared."), 4);
    // 4 for the addresses, 16 to split the accumulator, 8 to give it back.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4 + 16 + 8);
    // The products take what was loaded, in the order of the program.
    let loads: Vec<String> = lowered
        .lines()
        .filter_map(|line| line.trim().strip_prefix("ld.shared.v2.u32 "))
        .map(|line| line.split_once('}').unwrap().0.to_string() + "}")
        .collect();
    let products = volta_operands(&lowered);
    let taken: Vec<[&String; 2]> = products.iter().map(|product| [&product[1], &product[2]]).collect();
    // (The loads: the pairs (%r1, %r2) and (%r3, %r4) of A, then %r5 and %r6 of B.)
    assert_eq!(taken[0], [&loads[0], &loads[2]]);
    assert_eq!(taken[1], [&loads[1], &loads[3]]);
    assert_eq!(taken[2], [&loads[0], &loads[3]]);
    assert_eq!(taken[3], [&loads[1], &loads[2]]);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // Read by another statement as well, a register is loaded in the layout of the program too.
    let read = lower_aligned(&loaded_chain("", "\tst.global.u32 [%rd3], %r5;\n"));
    assert_eq!(count(&read, "ld.shared.v2.u32 "), 4);
    assert_eq!(count(&read, "ld.shared.u32 %r5, "), 1);
    assert_eq!(count(&read, "ld.shared."), 5);

    // Read by a product that is not of a chain as well: its rows and columns besides.
    let other = format!("\t{F32} {{%r7, %r8, %d2, %d3}}, {{%r1, %r2}}, {{%r5}}, {{%r7, %r8, %d2, %d3}};\n");
    let both = lower_aligned(&loaded_chain(&other, ""));
    assert_eq!(count(&both, "ld.shared.v2.u32 "), 4);
    assert_eq!(count(&both, "ld.shared.v4.u32 "), 2);
    assert_eq!(count(&both, CHAIN), 4 + 2);
    assert_eq!(undeclared(&both), Vec::<String>::new());

    // Rows that may be anywhere: the registers of the program are loaded and the products
    // gather their operands from them.
    let anywhere = lower(&loaded_chain("", ""));
    assert_eq!(count(&anywhere, CHAIN), 4);
    assert_eq!(count(&anywhere, "ld.shared.v2.u32 "), 0);
    assert_eq!(count(&anywhere, "ld.shared.u32 %r"), 6);
    assert_eq!(count(&anywhere, "shfl.sync.idx.b32"), 16 + 4 * 6 + 8 + 6);
}

#[test]
fn a_chain_before_its_loads_in_the_text_takes_what_they_load() {
    // A loop whose loads come after its products in the text: the products read what the
    // last turn loaded, from the registers the loads fill.
    let text = format!(
        "{{\n{DECLARED}$L_top:\n\
             \t{F32} {D4}, {{%r1, %r2}}, {{%r5}}, {D4};\n\t{F32} {D4}, {{%r3, %r4}}, {{%r6}}, {D4};\n\
             \t{F32} {D4}, {{%r1, %r2}}, {{%r6}}, {D4};\n\t{F32} {D4}, {{%r3, %r4}}, {{%r5}}, {D4};\n\
             \t{LD}.x4.b16 {{%r1, %r2, %r3, %r4}}, [%rd1];\n\t{LD}.x2.b16 {{%r5, %r6}}, [%rd2];\n\
             \t@%p1 bra $L_top;\n\tret;\n}}"
    );
    let lowered = lower_aligned(&text);
    assert_eq!(count(&lowered, CHAIN), 4);
    assert_eq!(count(&lowered, "ld.shared.v2.u32 "), 4);
    assert_eq!(count(&lowered, "ld.shared."), 4);
    let loads: Vec<String> = lowered
        .lines()
        .filter_map(|line| line.trim().strip_prefix("ld.shared.v2.u32 "))
        .map(|line| line.split_once('}').unwrap().0.to_string() + "}")
        .collect();
    let products = volta_operands(&lowered);
    assert_eq!([&products[0][1], &products[0][2]], [&loads[0], &loads[2]]);
    assert_eq!([&products[3][1], &products[3][2]], [&loads[1], &loads[2]]);
    // No operand is gathered: the shuffles are those of the accumulator and of the addresses.
    assert_eq!(count(&lowered, "shfl.sync.idx.b32"), 4 + 16 + 8);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());
}

#[test]
fn regression_chain_gathers_nonconsecutive_a_fragments() {
    for (low, high) in [("%r1", "%r3"), ("%r3", "%r1"), ("%r2", "%r2")] {
        let pair = format!("{{{low}, {high}}}");
        let source = loaded_chain("", "").replace("{%r1, %r2}", &pair);
        let lowered = lower_aligned(&source);
        assert_eq!(count(&lowered, CHAIN), 4);
        for name in [low, high] {
            assert!(lowered.contains(&format!("ld.shared.u32 {name},")));
            assert!(lowered.contains(&format!(", {name}, %vf_cw0, 31, -1;")));
        }
    }
}

#[test]
fn regression_reduction_keeps_only_live_lanes() {
    for op in ["add.s32", "min.s32", "max.u32", "and.b32", "or.b32", "xor.b32"] {
        let source = format!(
            "{{\n.reg .b32 %r<2>;\n.reg .pred %p;\nmov.u32 %r0, %laneid;\n\
             setp.ge.u32 %p, %r0, 16;\n@%p exit;\nredux.sync.{op} %r1, %r0, -1;\nret;\n}}"
        );
        let lowered = lower(&source);
        assert!(nth(&lowered, "vote.sync.ballot.b32", 0) < nth(&lowered, "shfl.sync.bfly", 0));
        assert!(lowered.contains("bfind.u32"));
        assert!(lowered.contains("shfl.sync.idx.b32"));
        assert_eq!(count(&lowered, "shfl.sync.bfly"), 5);
        assert!(undeclared(&lowered).is_empty());
    }
}

#[test]
fn regression_partial_reduction_in_one_order_for_every_lane() {
    // Every surviving lane starts from the input of the highest one and takes the others from
    // there down: the same operations on the same values, whichever lane it is. (Until
    // 2026-11-22 each started with its own input: the order then depends on the lane, and for
    // the minimum or maximum of floats so may the sign of a zero or the bits of a NaN.)
    for op in ["add.u32", "min.s32", "max.f32", "min.NaN.f32", "and.b32"] {
        let source = format!(
            "{{\n.reg .b32 %r<2>;\n.reg .pred %p;\nmov.u32 %r0, %laneid;\n\
             setp.ge.u32 %p, %r0, 16;\n@%p exit;\nredux.sync.{op} %r1, %r0, -1;\nret;\n}}"
        );
        let lowered = lower(&source);
        // (%vf_r0 the value reduced, %vf_r1 the lanes alive, %vf_r2 those still to take, %vf_r3
        // the one taken, %vf_r4 its bit.)
        let partial = &lowered[nth(&lowered, "$vf_partial", 1)..nth(&lowered, "$vf_peers", 0)];
        assert_eq!(
            partial.lines().skip(1).map(str::trim).collect::<Vec<_>>(),
            [
                "bfind.u32 %vf_r3, %vf_r1;",
                "shfl.sync.idx.b32 %vf_r0, %r0, %vf_r3, 31, %vf_r1;",
                "shl.b32 %vf_r4, 1, %vf_r3;",
                "xor.b32 %vf_r2, %vf_r1, %vf_r4;",
            ],
            "{op}"
        );
        let kind = op.rsplit('.').next().unwrap();
        let operation = op.split('.').next().unwrap();
        let reduce = format!("{operation}.{kind} %vf_r0, %vf_r0, ");
        assert_eq!(count(&lowered, &reduce), 6, "{op}:\n{lowered}");
        assert_eq!(count(&lowered, "%vf_l"), 0, "{op}");
        assert!(undeclared(&lowered).is_empty(), "{op}");
    }
}

#[test]
fn regression_reduction_of_floats_without_nan_gives_the_canonical_nan() {
    // Without .NaN, the minimum or maximum of floats is the canonical NaN when every input is a
    // NaN, and so when one lane alone is alive: its input is reduced with no other. (Until
    // 2026-11-23 the NaN of that lane was given as it was, 0x7fc00001 for 0x7fc00001.)
    let canonical = [
        "setp.nan.f32 %vf_n, %vf_r0, %vf_r0;",
        "selp.f32 %r1, 0f7FFFFFFF, %vf_r0, %vf_n;",
    ];
    let any_nan = [
        "setp.nan.f32 %vf_n, %r0, %r0;",
        "vote.sync.any.pred %vf_n, %vf_n, -1;",
        "selp.f32 %r1, 0f7FFFFFFF, %vf_r0, %vf_n;",
    ];
    let integer = ["mov.b32 %r1, %vf_r0;"];
    for (op, end) in [
        ("min.f32", &canonical[..]),
        ("max.f32", &canonical[..]),
        ("max.NaN.f32", &any_nan[..]),
        ("min.s32", &integer[..]),
        ("max.u32", &integer[..]),
    ] {
        let source = format!(
            "{{\n.reg .b32 %r<2>;\n.reg .pred %p;\nmov.u32 %r0, %laneid;\n\
             setp.ne.u32 %p, %r0, 7;\n@%p exit;\nredux.sync.{op} %r1, %r0, -1;\nret;\n}}"
        );
        let lowered = lower(&source);
        // What follows the end of the reduction, for every number of lanes alive.
        let lines: Vec<&str> = lowered
            .lines()
            .map(str::trim)
            .skip_while(|line| !(line.starts_with("$vf_reduced") && line.ends_with(':')))
            .skip(1)
            .take_while(|line| *line != "ret;")
            .collect();
        assert_eq!(lines, end, "{op}:\n{lowered}");
        assert!(undeclared(&lowered).is_empty(), "{op}");
    }
}

#[test]
fn regression_ptx_integer_radices_in_sizes_and_offsets() {
    let source = "{\n.reg .b32 %r<8>;\n.reg .b64 %rd;\n\
                  cp.async.ca.shared.global [%r0+32], [%rd+64], 16, 8;\n\
                  ld.global.v8.b32 {%r0,%r1,%r2,%r3,%r4,%r5,%r6,%r7}, [%rd+32];\nret;\n}";
    let expected = lower(source);
    for encoded in [
        source
            .replace("+32", "+040")
            .replace("+64", "+0100")
            .replace(", 8;", ", 010;"),
        source.replace(", 8;", ", +010U;"),
        source
            .replace("+32", "+0X20U")
            .replace("+64", "+0b1000000")
            .replace("16, 8;", "020, 0B1000U;"),
    ] {
        assert_eq!(lower(&encoded), expected);
    }
}

#[test]
fn cta_barriers_preserve_guards_and_separate_the_immutable_count_from_progress() {
    let body = lower(
        "{ .reg .pred p; .reg .b32 r; .reg .b64 token; .shared .align 8 .b64 gates[3];
           @p mbarrier.init.shared::cta.b64 [gates+010], 5;
           @!p cp.async.mbarrier.arrive.noinc.shared.b64 [gates+010];
           @p mbarrier.arrive.shared.b64 token, [gates+010];
           @p mbarrier.test_wait.parity.shared.b64 p, [gates+010], r; }",
    );
    assert!(body.contains("@p st.shared.u32 [gates+12], 5;"));
    assert!(body.contains("@!p atom.acq_rel.cta.shared.inc.u32"));
    assert!(body.contains("@p atom.acq_rel.cta.shared.inc.u32"));
    assert!(body.contains("@p mov.b64 token,"));
    assert!(body.contains("@p ld.acquire.cta.shared.u32"));
    assert!(body.contains("@p setp.ne.u32 p,"));
}

#[test]
fn other_barrier_operations_refuse_the_whole_body() {
    for instruction in [
        "mbarrier.arrive_drop.shared.b64 token, [r]",
        "mbarrier.arrive.expect_tx.shared.b64 token, [r], 16",
        "mbarrier.test_wait.shared.b64 p, [r], token",
        "mbarrier.test_wait.parity.shared::cluster.b64 p, [r], 0",
        "cp.async.mbarrier.arrive.shared.b64 [r]",
        "cp.async.bulk.global.shared::cta.bulk_group [r], [s], 16",
    ] {
        let body = format!("{{ mbarrier.init.shared.b64 [r], 5; {instruction}; }}");
        assert!(refusal(&body).starts_with(instruction.split_whitespace().next().unwrap()));
    }
}

#[test]
fn packed_bf16_fma_uses_exact_cancellation_when_both_lanes_qualify() {
    // Low lane: -128 + [128,143]. High lane: +2 + [-3.984375,-2].
    let setup = "mov.u32 %a,0x40004300;\nmov.b32 %b,0x3f80bf80;\nand.b32 %c,%x,0x007f000f;\nor.b32 %c,%c,0xc0004300;";
    let program = |body: &str| {
        lower(&format!(
            "{{\n.reg .b32 %a,%b,%c,%x;\n.reg .pred %p;\n{setup}\n{body}\nret;\n}}"
        ))
    };
    let fma = "@%p fma.rn.bf16x2 %c,%a,%b,%c;";
    let result = program(fma);
    assert_eq!(count(&result, "add.rn.f32 "), 2);
    assert!(result.contains("@%p prmt.b32 %c,"));
    assert!(result.contains("0fc3000000;") && result.contains("0f40000000;"));
    for change in [
        "@%p mov.u32 %c,%x;",                  // a predicated change makes its bounds unknown
        "branch_target:",                      // the addend no longer has a known local definition
        "mov.b32 %b,0xbf803f80;",              // both lanes now add terms of the same sign
        "or.b32 %c,%c,0x7f800000;",            // high lane may be nonfinite
        "mov.b32 %a,0x40000000;",              // zero multiplicand
        "{\n.reg .b32 %c;\nmov.b32 %c,%x;\n}", // no bounds survive a shadow
    ] {
        assert_eq!(
            count(&program(&format!("{change}\n{fma}")), "fma.rm.f32 "),
            2,
            "{change}"
        );
    }
    let after_join = program(&format!(
        "join:\nand.b32 %c,%x,0x007f000f;\nor.b32 %c,%c,0xc0004300;\n{fma}"
    ));
    assert_eq!(count(&after_join, "add.rn.f32 "), 2);
}

#[test]
fn packed_bf16_fma_with_a_unit_factor_uses_the_bounded_multiplicand() {
    let program = |op: &str, one: &str| {
        lower(&format!(
            "{{ .reg .b32 %x,%one,%bias,%packed;
        mov.u32 %one,{one}; mov.u32 %bias,-1023360256;
        and.b32 %packed,%x,0x000f000f; or.b32 %packed,%packed,0x43004300;
        fma.rn.bf16x2 %packed,{op},%bias; ret; }}"
        ))
    };
    for (op, one) in [("%packed,%one", "0x3f803f80"), ("%one,%packed", "1065369472")] {
        let output = program(op, one);
        assert_eq!(count(&output, "add.rn.f32 "), 2);
        assert!(output.contains("0fc3000000;"));
        assert_eq!(count(&output, "fma.rm.f32 "), 0);
    }
    for bad in ["0x3f813f80", "0x3f807fc0", "0xbf803f80"] {
        assert_eq!(count(&program("%packed,%one", bad), "fma.rm.f32 "), 2);
    }
}

// Pairs of products (`pairs.rs`, `docs/pairs.md` section 13).

/// Lowered with pairs, rows taken to be aligned (the policy's default).
fn lower_paired(text: &str) -> String {
    body(
        text,
        64,
        Options {
            aligned_rows: true,
            pairs: true,
            ..Default::default()
        },
    )
    .unwrap()
    .0
}

/// A loop as Triton writes it: accumulators set to zero by copies before the loop, one A and two
/// B loaded by `ldmatrix` in the loop, two `m16n8k16` products on the same A in place, the
/// accumulators stored after the loop. `{between}` is put between the two products, `{after}`
/// after the loop, before the stores; `{b2}` is the opcode of the second B load.
fn paired_loop(between: &str, after: &str, b2: &str) -> String {
    format!(
        "{{
	.reg .pred %p<4>;
	.reg .b32 %r<64>;
	.reg .b64 %rd<4>;
	mov.u32 %r20, %laneid;
	mov.b32 %r40, 0f00000000;
	mov.b32 %r41, %r40;
	mov.b32 %r42, %r40;
	mov.b32 %r43, %r40;
	mov.b32 %r44, %r40;
	mov.b32 %r45, %r40;
	mov.b32 %r46, %r40;
	mov.b32 %r47, %r40;
$L__loop:
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0, %r1, %r2, %r3}}, [%r30];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r4, %r5, %r6, %r7}}, [%r31];
	{b2} {{%r8, %r9, %r10, %r11}}, [%r32];
	{K16_F32} {{%r40, %r41, %r42, %r43}}, {{%r0, %r1, %r2, %r3}}, {{%r4, %r5}}, {{%r40, %r41, %r42, %r43}};
{between}
	{K16_F32} {{%r44, %r45, %r46, %r47}}, {{%r0, %r1, %r2, %r3}}, {{%r8, %r9}}, {{%r44, %r45, %r46, %r47}};
	add.s32 %r33, %r33, -1;
	setp.ne.s32 %p1, %r33, 0;
	@%p1 bra $L__loop;
{after}
	st.global.v4.b32 [%rd1], {{%r40, %r41, %r42, %r43}};
	st.global.v4.b32 [%rd1+16], {{%r44, %r45, %r46, %r47}};
	ret;
}}"
    )
}

const PLAIN_LOAD: &str = "ldmatrix.sync.aligned.m8n8.x4.shared.b16";

#[test]
fn two_products_on_one_a_take_two_volta_products_each() {
    let text = paired_loop("", "", PLAIN_LOAD);
    let paired = lower_paired(&text);
    let alone = lower_aligned(&text);
    // Four of Volta's products for the two, where each alone takes four.
    assert_eq!(count(&alone, HALVES_F32), 8);
    assert_eq!(count(&paired, HALVES_F32), 4);
    // In place on the program's eight registers, x's first, without a choice of results.
    let all = "%r40, %r41, %r42, %r43, %r44, %r45, %r46, %r47";
    for operands in volta_operands(&paired) {
        assert_eq!(operands[0], format!("{{{all}}}"));
        assert_eq!(operands[3], format!("{{{all}}}"));
    }
    assert_eq!(count(&paired, "selp.b32 %r4"), 0);
    // A as whole rows, B as whole rows from each of its two loads into one set of registers.
    assert_eq!(count(&paired, "@!%vf_p16 ld.shared.v4.u32"), 2);
    assert_eq!(count(&paired, "@%vf_p16 ld.shared.v4.u32"), 2);
    assert!(paired.contains(", %vf_yp01, 31, -1;") && paired.contains(", %vf_yp23, 31, -1;"));
    // Turing's layout comes back once for each register, after the loop, before the stores that
    // read it; nothing reads them after, so they are not put back.
    assert_eq!(count(&paired, ", %vf_yt, 31, -1;"), 8);
    assert_eq!(count(&paired, ", %vf_yq, 31, -1;"), 0);
    let back = paired.find("shfl.sync.idx.b32 %r40, %r40, %vf_yt, 31, -1;").unwrap();
    assert!(back > paired.find("@%p1 bra $L__loop;").unwrap());
    assert!(back < paired.find("st.global.v4.b32 [%rd1], {%r40").unwrap());
    // The products are counted as before.
    let counted = |options| body(&text, 64, options).unwrap().1.tensor;
    let on = Options {
        aligned_rows: true,
        pairs: true,
        ..Default::default()
    };
    assert_eq!(counted(on), counted(Options::default()));
    assert!(undeclared(&paired).is_empty(), "{:?}", undeclared(&paired));
}

#[test]
fn pairs_follow_the_option_and_aligned_rows() {
    let text = paired_loop("", "", PLAIN_LOAD);
    let without = |aligned_rows, pairs| {
        body(
            &text,
            64,
            Options {
                aligned_rows,
                pairs,
                ..Default::default()
            },
        )
        .unwrap()
        .0
    };
    assert_eq!(without(true, false), lower_aligned(&text));
    assert_eq!(without(false, true), lower(&text));
}

#[test]
fn what_a_pair_cannot_show_is_left_as_it_was() {
    let base = paired_loop("", "", PLAIN_LOAD);
    let unchanged = |text: &str, why: &str| {
        assert_eq!(lower_paired(text), lower_aligned(text), "{why}");
        assert_eq!(count(&lower_paired(text), ", %vf_yt, 31"), 0, "{why}");
    };
    // B of the second product by `ldmatrix.trans`: today's lowering adds even k, then odd k.
    let trans = paired_loop("", "", "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16");
    assert!(lower_aligned(&trans).contains(VOLTA_F32), "the even/odd form today");
    unchanged(&trans, "B transposed");
    // f16 accumulators.
    let f16 = base
        .replace(K16_F32, "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16")
        .replace("{%r40, %r41, %r42, %r43}", "{%r40, %r41}")
        .replace("{%r44, %r45, %r46, %r47}", "{%r44, %r45}");
    unchanged(&f16, "f16 accumulators");
    // Two different A.
    unchanged(
        &base.replacen(
            "{%r0, %r1, %r2, %r3}, {%r8, %r9}",
            "{%r0, %r1, %r6, %r7}, {%r8, %r9}",
            1,
        ),
        "another A",
    );
    // x's B loaded between the two, for the next turn: x reads what the previous turn loaded.
    let x_load = "	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r4, %r5, %r6, %r7}, [%r31];\n";
    let reloaded = paired_loop(x_load.trim_end(), "", PLAIN_LOAD).replacen(x_load, "", 1);
    assert_eq!(count(&reloaded, "{%r4, %r5, %r6, %r7}"), 1);
    unchanged(&reloaded, "x's B loaded between");
    // Not in place.
    unchanged(
        &base.replacen(
            "{%r8, %r9}, {%r44, %r45, %r46, %r47}",
            "{%r8, %r9}, {%r48, %r49, %r50, %r51}",
            1,
        ),
        "accumulator elsewhere",
    );
    let mutations: &[(&str, &str, &str)] = &[
        // An accumulator written by another instruction.
        ("", "\tadd.f32 %r40, %r40, 0f3F800000;", "written by an addition"),
        // Read where the 32 lanes may not be together: after a branch that depends on the lane.
        (
            "",
            "\tsetp.eq.u32 %p2, %r20, 0;\n\t@%p2 bra $L__skip;\n\tst.global.b32 [%rd2], %r41;\n$L__skip:",
            "read in code some lanes skip",
        ),
        // Read by a call.
        ("", "\tcall use, (%r42);", "read by a call"),
        // x's accumulator read between the two.
        ("\tst.global.b32 [%rd2], %r40;", "", "read between the products"),
        // A label something jumps to between the two.
        ("$L__in:\n\t@%p3 bra $L__in;", "", "entered between the products"),
        // x's operands loaded again between the two (then by two statements).
        (
            "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r4, %r5, %r6, %r7}, [%r31];",
            "",
            "x's B loaded twice",
        ),
        // A block between the two.
        ("\t{ .reg .b32 %t; mov.b32 %t, 0; }", "", "a block between"),
        // Copied from a register that differs from lane to lane.
        ("", "\tmov.b32 %r41, %r20;", "a copy of the lane"),
        // Loaded from memory.
        ("", "\tld.global.b32 %r43, [%rd2];", "loaded"),
        // Under a predicate.
        ("", "\t@%p1 mov.b32 %r44, 0;", "a guarded write"),
    ];
    for (between, after, why) in mutations {
        unchanged(&paired_loop(between, after, PLAIN_LOAD), why);
    }
    // An accumulator declared again in a block.
    unchanged(
        &base.replace("\tret;", "\t{ .reg .b32 %r45; mov.b32 %r45, 1; }\n\tret;"),
        "declared again",
    );
    // An indirect branch in the function.
    unchanged(
        &base.replace(
            "\tret;",
            "\t$L__t: .branchtargets $L__end;\n\tbrx.idx %r33, $L__t;\n$L__end:\n\tret;",
        ),
        "indirect branch",
    );
    // A guarded product: refused with and without pairs.
    let guarded = base.replacen(&format!("\t{K16_F32} {{%r44"), &format!("\t@%p1 {K16_F32} {{%r44"), 1);
    let options = |pairs| Options {
        aligned_rows: true,
        pairs,
        ..Default::default()
    };
    assert_eq!(
        body(&guarded, 64, options(true)).unwrap_err(),
        body(&guarded, 64, options(false)).unwrap_err()
    );
}

#[test]
fn a_pair_drops_only_its_own_registers() {
    // Two pairs on the same A; an addition to one register of the first drops the first alone.
    let text = format!(
        "{{
	.reg .b32 %r<80>;
	.reg .b64 %rd<4>;
	mov.b32 %r40, 0;
	mov.b32 %r41, 0; mov.b32 %r42, 0; mov.b32 %r43, 0; mov.b32 %r44, 0; mov.b32 %r45, 0; mov.b32 %r46, 0; mov.b32 %r47, 0;
	mov.b32 %r50, 0; mov.b32 %r51, 0; mov.b32 %r52, 0; mov.b32 %r53, 0; mov.b32 %r54, 0; mov.b32 %r55, 0; mov.b32 %r56, 0; mov.b32 %r57, 0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0, %r1, %r2, %r3}}, [%r30];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r4, %r5, %r6, %r7}}, [%r31];
	{K16_F32} {{%r40, %r41, %r42, %r43}}, {{%r0, %r1, %r2, %r3}}, {{%r4, %r5}}, {{%r40, %r41, %r42, %r43}};
	{K16_F32} {{%r44, %r45, %r46, %r47}}, {{%r0, %r1, %r2, %r3}}, {{%r6, %r7}}, {{%r44, %r45, %r46, %r47}};
	{K16_F32} {{%r50, %r51, %r52, %r53}}, {{%r0, %r1, %r2, %r3}}, {{%r4, %r5}}, {{%r50, %r51, %r52, %r53}};
	{K16_F32} {{%r54, %r55, %r56, %r57}}, {{%r0, %r1, %r2, %r3}}, {{%r6, %r7}}, {{%r54, %r55, %r56, %r57}};
	add.f32 %r40, %r40, 0f3F800000;
	st.global.v4.b32 [%rd1], {{%r40, %r41, %r42, %r43}};
	st.global.v4.b32 [%rd1+16], {{%r50, %r51, %r52, %r53}};
	ret;
}}"
    );
    let paired = lower_paired(&text);
    // The second pair, from one load: x's B is matrix 0, y's matrix 2, one shuffle with the row
    // chosen by the half of the warp, one load, no predicate.
    assert_eq!(count(&paired, HALVES_F32), 8 + 4);
    assert!(paired.contains(", %vf_yc02, 31, -1;") && paired.contains(", %vf_yc13, 31, -1;"));
    assert_eq!(count(&paired, "%vf_p16 ld.shared"), 0);
    // Only the second pair's registers that are read come back.
    assert_eq!(count(&paired, ", %vf_yt, 31, -1;"), 4);
    assert!(paired.contains("shfl.sync.idx.b32 %r50, %r50, %vf_yt, 31, -1;"));
    assert!(!paired.contains("shfl.sync.idx.b32 %r40, %r40, %vf_yt"));
}

#[test]
fn a_register_read_in_the_loop_is_put_back_where_it_is_read_again() {
    // A store of x's accumulator after the products, in the loop: in Turing's layout for the
    // store, then back for the products of the next turn.
    let text = paired_loop("", "", PLAIN_LOAD).replace(
        "\tadd.s32 %r33, %r33, -1;",
        "\tst.global.b32 [%rd2], %r41;\n\tadd.s32 %r33, %r33, -1;",
    );
    let paired = lower_paired(&text);
    assert_eq!(count(&paired, HALVES_F32), 4);
    let to = paired.find("shfl.sync.idx.b32 %r41, %r41, %vf_yt, 31, -1;").unwrap();
    let store = paired.find("st.global.b32 [%rd2], %r41;").unwrap();
    let back = paired.find("shfl.sync.idx.b32 %r41, %r41, %vf_yq, 31, -1;").unwrap();
    assert!(to < store && store < back);
    // After the loop, eight more to Turing's layout.
    assert_eq!(count(&paired, ", %vf_yt, 31, -1;"), 9);
    assert_eq!(count(&paired, ", %vf_yq, 31, -1;"), 1);
}

#[test]
fn a_branch_an_aligned_instruction_depends_on_is_uniform() {
    // A reader after a branch whose taken way reaches a product in straight-line code and whose
    // other way can leave without it: all 32 lanes take one way.
    let text = paired_loop(
        "",
        "\tsetp.eq.u32 %p2, %r33, 7;\n\t@%p2 bra $L__more;\n\tst.global.b32 [%rd2], %r41;\n\tbra.uni $L__done;\n$L__more:\n\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r12, %r13, %r14, %r15}, [%r34];\n\tst.global.b32 [%rd2], %r12;\n$L__done:",
        PLAIN_LOAD,
    );
    let paired = lower_paired(&text);
    assert_eq!(count(&paired, HALVES_F32), 4);
    assert!(paired.contains("shfl.sync.idx.b32 %r41, %r41, %vf_yt, 31, -1;\n\tst.global.b32 [%rd2], %r41;"));
    // Without the aligned instruction on the taken way, the branch may part the lanes.
    let parted = text.replace(
        "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r12, %r13, %r14, %r15}, [%r34];\n",
        "",
    );
    assert_eq!(lower_paired(&parted), lower_aligned(&parted));
    // Nor when the aligned instruction follows both ways: the lanes may part at the branch and meet
    // again before it, and the reader on one way is run by some lanes.
    let diamond = paired_loop(
        "",
        "\tsetp.eq.u32 %p2, %r20, 0;\n\t@%p2 bra $L__join;\n\tst.global.b32 [%rd2], %r41;\n$L__join:\n\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r12, %r13, %r14, %r15}, [%r34];\n\tst.global.b32 [%rd2], %r12;",
        PLAIN_LOAD,
    );
    assert_eq!(lower_paired(&diamond), lower_aligned(&diamond));
}

#[test]
fn a_loop_before_the_products_does_not_hide_that_they_depend_on_the_back_edge() {
    // An inner loop at the head of the loop (filling shared memory) before the first `ldmatrix`:
    // that `ldmatrix` still post-dominates the head, so the back edge is uniform and the code after
    // the loop is run by the 32 lanes together.
    let inner = "\tmov.u32 %r21, 0;\n$L__fill:\n\tadd.u32 %r21, %r21, 1;\n\tsetp.lt.u32 %p3, %r21, %r22;\n\t@%p3 bra $L__fill;\n";
    let text = paired_loop("", "", PLAIN_LOAD).replace("$L__loop:\n", &format!("$L__loop:\n{inner}"));
    let paired = lower_paired(&text);
    assert_eq!(count(&paired, HALVES_F32), 4);
    assert_eq!(count(&paired, ", %vf_yt, 31, -1;"), 8);
    // A way out of the inner loop that does not pass the products: the back edge may part the lanes.
    let out = text
        .replace("\t@%p3 bra $L__fill;\n", "\t@%p3 bra $L__fill;\n\t@%p2 bra $L__exit;\n")
        .replace("\tret;\n}", "$L__exit:\n\tret;\n}");
    assert_eq!(lower_paired(&out), lower_aligned(&out));
}

#[test]
fn copies_that_keep_the_permutation_are_allowed() {
    // A register that holds zero in every lane, copied into the accumulators in the loop's
    // preheader, and one accumulator copied into another: no conversion.
    let text = paired_loop("", "", PLAIN_LOAD)
        .replace(
            "\tmov.b32 %r40, 0f00000000;",
            "\tmov.b32 %r60, 0f00000000;\n\tmov.b32 %r40, %r60;",
        )
        .replace("\tmov.b32 %r47, %r40;", "\tmov.b32 %r47, %r46;");
    let paired = lower_paired(&text);
    assert_eq!(count(&paired, HALVES_F32), 4);
    assert_eq!(count(&paired, ", %vf_yt, 31, -1;"), 8);
}

#[test]
fn chains_and_m16n8k8_pairs() {
    // m16n8k8 pairs: two of Volta's products for the two.
    let text = format!(
        "{{
	.reg .b32 %r<64>;
	.reg .b64 %rd<4>;
	mov.b32 %r40, 0; mov.b32 %r41, 0; mov.b32 %r42, 0; mov.b32 %r43, 0;
	mov.b32 %r44, 0; mov.b32 %r45, 0; mov.b32 %r46, 0; mov.b32 %r47, 0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r0, %r1, %r2, %r3}}, [%r30];
	ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%r4, %r5}}, [%r31];
	{F32} {{%r40, %r41, %r42, %r43}}, {{%r2, %r3}}, {{%r4}}, {{%r40, %r41, %r42, %r43}};
	{F32} {{%r44, %r45, %r46, %r47}}, {{%r2, %r3}}, {{%r5}}, {{%r44, %r45, %r46, %r47}};
	st.global.v4.b32 [%rd1], {{%r40, %r41, %r42, %r43}};
	st.global.v4.b32 [%rd1+16], {{%r44, %r45, %r46, %r47}};
	ret;
}}"
    );
    let paired = lower_paired(&text);
    assert_eq!(count(&paired, HALVES_F32), 2);
    assert!(paired.contains(", %vf_yp23, 31, -1;") && paired.contains(", %vf_yc01, 31, -1;"));
    // Products of a chain (four on one accumulator in a row) keep the chain's form.
    let product =
        |b: &str| format!("\t{F32} {{%r40, %r41, %r42, %r43}}, {{%r2, %r3}}, {{{b}}}, {{%r40, %r41, %r42, %r43}};\n");
    let chained = text.replace(
        &product("%r4"),
        &[product("%r4"), product("%r5"), product("%r4"), product("%r5")].concat(),
    );
    assert_eq!(lower_paired(&chained), lower_aligned(&chained));
}

/// The value of each lane register of the prologue of `lowered`, in each of the 32 lanes: the
/// instructions the lane values are made with, interpreted.
fn lane_values(lowered: &str) -> HashMap<String, [u32; 32]> {
    let mut values: HashMap<String, [u32; 32]> = HashMap::new();
    let mut started = false;
    for line in lowered.lines() {
        let line = line.trim().trim_end_matches(';');
        if line.is_empty() || line == "{" || (!started && line.starts_with(".reg")) {
            continue;
        }
        let Some((opcode, operands)) = line.split_once(' ') else {
            break;
        };
        let operands: Vec<&str> = operands.split(',').map(str::trim).collect();
        if !operands[0].starts_with("%vf_") {
            break;
        }
        started = true;
        let get = |values: &HashMap<String, [u32; 32]>, operand: &str, lane: usize| -> u32 {
            values
                .get(operand)
                .map_or_else(|| operand.parse().unwrap(), |v| v[lane])
        };
        let mut result = [0u32; 32];
        for (lane, slot) in result.iter_mut().enumerate() {
            let at = |k: usize| get(&values, operands[k], lane);
            *slot = match opcode {
                "mov.u32" if operands[1] == "%laneid" => lane as u32,
                "and.b32" => at(1) & at(2),
                "or.b32" => at(1) | at(2),
                "shr.u32" => at(1) >> at(2),
                "shl.b32" => at(1) << at(2),
                "add.u32" => at(1) + at(2),
                "setp.ne.b32" => u32::from(at(1) != at(2)),
                "selp.b32" => {
                    if at(3) != 0 {
                        at(1)
                    } else {
                        at(2)
                    }
                }
                "bfi.b32" => {
                    let (position, length) = (at(3), at(4));
                    let mask = ((1u32 << length) - 1) << position;
                    (at(2) & !mask) | ((at(1) << position) & mask)
                }
                _ => panic!("{line}: not interpreted"),
            };
        }
        values.insert(operands[0].to_string(), result);
    }
    values
}

#[test]
fn the_lanes_of_a_pair_give_turings_layout_back() {
    // A host model of the arrangement: what lane L holds in each of the eight registers after
    // Volta's product on the rows and columns that the lane values of the lowering choose, then
    // after the shuffle by `%vf_yt`, is Turing's register of the accumulator of x or y. The layouts
    // are those of the PTX manual ("Matrix Fragments for mma.m8n8k4" and "for mma.m16n8k8"),
    // written here without the lowering's formulas.
    let text = paired_loop("", "", PLAIN_LOAD);
    let paired = lower_paired(&text);
    let values = lane_values(&paired);
    let lanes = |name: &str| values.get(name).unwrap_or_else(|| panic!("{name} not computed"));
    // Rows of A: the lane that gives the address, in matrix 0 (rows 0..7) or 1 (rows 8..15).
    // Columns of B: x's matrix 0 where h is clear, y's where it is set (both loads use %vf_yb*).
    let (row_lane, col_lane, back) = (lanes("%vf_yp01"), lanes("%vf_yb0"), lanes("%vf_yt"));
    // Row of the A operand (0..15: the address of row r of matrix m is in lane 8m + r, and A's
    // rows 0..7 are matrix 0) and column of B (0..7, product 0 or 1) that lane L gives.
    let a_row = |l: usize| row_lane[l] as usize;
    let b_col = |l: usize| (col_lane[l] as usize % 8, l >> 4);
    for l in 0..32 {
        assert!(row_lane[l] < 16 && col_lane[l] < 8, "lane {l}");
    }
    // Volta's m8n8k4: octet q = lanes {4q..4q+3, 16+4q..16+4q+3}; a lane gives row
    // (L % 4) + 4 (L >= 16) of the octet's A and that column of its B; with f32 results, register
    // i of lane L is (row (L & 1) + (i & 2) + 4 (L >= 16), column (i & 4) + (L & 2) + (i & 1)).
    let octet_lane = |q: usize, index: usize| {
        if index < 4 {
            4 * q + index
        } else {
            16 + 4 * q + index - 4
        }
    };
    // held[L][i] = (product, row of the 16x8 result, column).
    let mut held = [[(0usize, 0usize, 0usize); 8]; 32];
    for (l, registers) in held.iter_mut().enumerate() {
        let q = (l >> 2) & 3;
        for (i, register) in registers.iter_mut().enumerate() {
            let row = (l & 1) + (i & 2) + 4 * (l >> 4);
            let column = (i & 4) + (l & 2) + (i & 1);
            let (giver_a, giver_b) = (octet_lane(q, row), octet_lane(q, column));
            let (n, product) = b_col(giver_b);
            *register = (product, a_row(giver_a), n);
        }
    }
    // Turing's m16n8k8 f32 result: register j of lane T is (row T/4 + 8 (j >= 2), column 2 (T % 4)
    // + (j & 1)); registers 0..3 are x's, 4..7 y's.
    for (t, &source) in back.iter().enumerate() {
        for (i, &value) in held[source as usize].iter().enumerate() {
            let j = i & 3;
            let want = (i >> 2, t / 4 + 8 * (j >> 1), 2 * (t % 4) + (j & 1));
            assert_eq!(value, want, "lane {t}, register {i}");
        }
    }
    // And the index that puts them back is its inverse.
    let forward = lane_values(&lower_paired(&paired_loop("", "", PLAIN_LOAD).replace(
        "\tadd.s32 %r33, %r33, -1;",
        "\tst.global.b32 [%rd2], %r41;\n\tadd.s32 %r33, %r33, -1;",
    )));
    let (to, from) = (forward["%vf_yq"], forward["%vf_yt"]);
    for l in 0..32 {
        assert_eq!(from[to[l] as usize] as usize, l);
    }
}

#[test]
fn the_gpu_case_of_pairs_is_paired() {
    // `tests/gpu/run.sh`, `pairs-adversarial`: its three pairs are made (the words it checks are
    // those of each product by itself, so the case would pass without them).
    let module = include_str!("../../../../tests/kernels/ptx-cases/pairs-adversarial.ptx");
    let text = module[module.find("{\n").unwrap()..].trim_end();
    let paired = lower_paired(text);
    assert_eq!(count(&lower_aligned(text), HALVES_F32), 20);
    assert_eq!(count(&paired, HALVES_F32), 4 + 4 + 2);
    assert_eq!(count(&paired, "@%vf_p16 ld.shared.v4.u32"), 2);
    assert!(paired.contains(", %vf_yc02, 31, -1;"));
    // The running word in the loop, and the registers read twice after it, are put back.
    assert!(paired.contains("shfl.sync.idx.b32 dx1, dx1, %vf_yt, 31, -1;\n\txor.b32 running, running, dx1;\n\tshfl.sync.idx.b32 dx1, dx1, %vf_yq, 31, -1;"));
}

#[test]
fn a_product_with_b_by_ldmatrix_trans_is_summed_on_even_and_odd_k() {
    // `tests/gpu/run.sh`, `trans-order`: the same product with B by a plain `ldmatrix` (halves of
    // k, `row.col`) and by `ldmatrix.trans` (even then odd k, `col.row`); with pairs as without.
    let module = include_str!("../../../../tests/kernels/ptx-cases/trans-order.ptx");
    let text = module[module.find("{\n").unwrap()..].trim_end();
    let paired = lower_paired(text);
    assert_eq!(paired, lower_aligned(text));
    assert_eq!(count(&paired, HALVES_F32), 4);
    assert_eq!(count(&paired, VOLTA_F32), 4);
}

#[test]
fn fp8_e4m3_decode_uses_prmt_lop3_without_sub() {
    for dst in ["f16x2", "bf16x2"] {
        let pair = lower(&format!(
            "{{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\tcvt.rn.{dst}.e4m3x2 %r1, %rs1;\n\tret;\n}}"
        ));
        assert_eq!(
            count(&pair, "prmt.b32 "),
            1,
            "{dst}: expected 0x4140 prmt unpack:\n{pair}"
        );
        assert_eq!(count(&pair, "lop3.b32 "), 2, "{dst}: expected 2 lop3 ops:\n{pair}");
        assert_eq!(
            count(&pair, "sub.u32 "),
            0,
            "{dst}: unexpected sub.u32 NaN mask:\n{pair}"
        );
        assert_eq!(count(&pair, "0x00800080"), 1, "{dst}:\n{pair}");
        assert_eq!(undeclared(&pair), Vec::<String>::new(), "{dst}");
    }

    let mma = lower(
        "{\n\tmma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3};\n\tret;\n}",
    );
    assert_eq!(
        count(&mma, ", %a"),
        8,
        "only 8 prmt.b32 unpacking reads from %a0..%a3:\n{mma}"
    );
    assert_eq!(
        count(&mma, "shr.u32 %vf_r"),
        16,
        "12 NaN shifts + 4 column splits, no shr 16 unpack:\n{mma}"
    );
    assert_eq!(undeclared(&mma), Vec::<String>::new());
}

#[test]
fn mxf8_block_scaled_defers_hmul2_and_keeps_eight_step_chain() {
    // e4m3 x e4m3: omits all HMUL2, folds +16 into scale_A's UE8M0 bias (896 + 16 = 912), and
    // keeps main's single 8-step Volta summation chain (no split-octet re-association / add.f32).
    let e4_e4 = lower(
        "{\n\t.reg .b32 %d<4>, %a<4>, %b<2>, %c<4>, %sa, %sb;\n\tmma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
    );
    assert_eq!(
        count(&e4_e4, "mul.rn.f16x2 "),
        0,
        "expected deferred UE8M0 scale fold without HMUL2:\n{e4_e4}"
    );
    assert_eq!(
        count(&e4_e4, ", 912;"),
        2,
        "expected +16 UE8M0 bias (912) on scale_A lower/upper:\n{e4_e4}"
    );
    assert_eq!(
        count(&e4_e4, "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32"),
        8,
        "expected main's 8-step col.row m8n8k4 chain (not split-octet re-association):\n{e4_e4}"
    );
    assert_eq!(
        count(&e4_e4, "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"),
        0,
        "unexpected split-octet row.col re-association:\n{e4_e4}"
    );
    assert_eq!(
        count(&e4_e4, "add.f32 "),
        0,
        "unexpected partial-sum f32 re-association add:\n{e4_e4}"
    );
    assert_eq!(undeclared(&e4_e4), Vec::<String>::new());

    // e4m3 x e5m2: folds +8 (896 + 8 = 904) into scale_A and still uses 0 HMUL2 + 8-step chain.
    let e4_e5 = lower(
        "{\n\t.reg .b32 %d<4>, %a<4>, %b<2>, %c<4>, %sa, %sb;\n\tmma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e5m2.f32.ue8m0 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
    );
    assert_eq!(count(&e4_e5, "mul.rn.f16x2 "), 0);
    assert_eq!(count(&e4_e5, ", 904;"), 2);
    assert_eq!(count(&e4_e5, "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32"), 8);
    assert_eq!(count(&e4_e5, "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"), 0);
    assert_eq!(count(&e4_e5, "add.f32 "), 0);
    assert_eq!(undeclared(&e4_e5), Vec::<String>::new());
}

#[test]
fn e2m1_decode_uses_prmt_lut_without_hmul() {
    let lowered = lower("{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\tcvt.rn.f16x2.e2m1x2 %r1, %rs1;\n\tret;\n}");
    assert_eq!(count(&lowered, "e2m1"), 0);
    assert_eq!(count(&lowered, "mul.rn.f16x2 "), 0);
    assert_eq!(count(&lowered, "mul.lo.u32 "), 0);
    assert_eq!(count(&lowered, "0x74007400"), 0);
    assert!(lowered.contains("0x3E3C3800") && lowered.contains("0x46444240"));
    assert_eq!(count(&lowered, "prmt.b32 "), 3);
    assert_eq!(count(&lowered, "lop3.b32 "), 1);
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    let wide = lower("{\n\t.reg .b32 %r<2>;\n\tcvt.rn.f16x2.e2m1x2 %r1, %r0;\n\tret;\n}");
    assert_eq!(count(&wide, "cvt.u32."), 0);
    assert_eq!(undeclared(&wide), Vec::<String>::new());

    let relu = lower("{\n\t.reg .b16 %rs1;\n\t.reg .b32 %r1;\n\tcvt.rn.relu.f16x2.e2m1x2 %r1, %rs1;\n\tret;\n}");
    assert_eq!(count(&relu, "e2m1"), 0);
    assert_eq!(count(&relu, "mul.rn.f16x2 "), 0);
    assert_eq!(count(&relu, "mul.lo.u32 "), 0);
    assert_eq!(count(&relu, "lop3.b32 "), 0);
    assert_eq!(count(&relu, "prmt.b32 "), 2);
    assert_eq!(undeclared(&relu), Vec::<String>::new());
}

#[test]
fn block_scaled_fp4_uses_split_octet_and_prmt_lut() {
    for (label, mma, is_zero_c, max_shfl) in [
        (
            "nvf4-nonzero-c",
            "{\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
            false,
            50usize,
        ),
        (
            "nvf4-zero-c",
            "{\n\t.reg .f32 %z;\n\tmov.f32 %z, 0f00000000;\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%z, %z, %z, %z}, %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
            true,
            42usize,
        ),
        (
            "mxf4-nonzero-c",
            "{\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
            false,
            42usize,
        ),
        (
            "mxf4-zero-c",
            "{\n\t.reg .f32 %z;\n\tmov.f32 %z, 0f00000000;\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%z, %z, %z, %z}, %sa, {0, 0}, %sb, {0, 0};\n\tret;\n}",
            true,
            34usize,
        ),
    ] {
        let lowered = lower(mma);
        assert_eq!(
            count(&lowered, "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "),
            8,
            "{label}"
        );
        assert_eq!(count(&lowered, "mul.rn.f16x2 "), 0, "{label}");
        assert!(
            lowered.contains("0x3E3C3800") && lowered.contains("0x46444240"),
            "{label}"
        );
        let shuffles = count(&lowered, "shfl.sync.");
        assert!(
            shuffles <= max_shfl,
            "{label}: {shuffles} shuffles > {max_shfl} (zero_c={is_zero_c})"
        );
        if label.starts_with("nvf4") {
            assert_eq!(count(&lowered, "mul.f32 "), 36, "{label}");
        }
        assert_eq!(undeclared(&lowered), Vec::<String>::new(), "{label}");
    }
}

#[test]
fn block_scaled_fp4_split_octet_lane_permutations_match_host_model() {
    let lowered = lower(
        "{\n\tmma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1, %c2, %c3}, %sa, {0, 1}, %sb, {0, 3};\n\tret;\n}",
    );
    let values = lane_values(&lowered);
    let lanes = |name: &str| values.get(name).unwrap_or_else(|| panic!("{name} not computed"));
    let (ns0, ns1, sa1) = (lanes("%vf_ns0"), lanes("%vf_ns1"), lanes("%vf_sa1"));
    let (sc0, sc1) = (lanes("%vf_sc0"), lanes("%vf_sc1"));
    let sb = [
        lanes("%vf_sb0_3"),
        lanes("%vf_sb1_3"),
        lanes("%vf_sb2_3"),
        lanes("%vf_sb3_3"),
    ];
    for s in 0..32u32 {
        let (s4, s3, s2, s1, s0) = ((s >> 4) & 1, (s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1, s & 1);
        let group = 4 * s4 + 2 * s1 + s0;
        let idx = s as usize;
        assert_eq!(ns0[idx], 4 * group + s3, "ns0 lane {s}");
        assert_eq!(ns1[idx], 4 * group + 2 + s3, "ns1 lane {s}");
        let row = 8 * s2 + 4 * s4 + 2 * s3 + s0;
        assert_eq!(sa1[idx], 4 * (row & 7) + 2 + (row >> 3), "sa1 lane {s}");
        assert_eq!(sc0[idx], (s4 << 4) | (s3 << 3) | (s0 << 2) | s1, "sc0 lane {s}");
        assert_eq!(sc1[idx], sc0[idx] | 2, "sc1 lane {s}");
        for (col_idx, col) in [2 * s1, 4 + 2 * s1, 2 * s1 + 1, 4 + 2 * s1 + 1].into_iter().enumerate() {
            assert_eq!(sb[col_idx][idx], 4 * col + 3, "sb[{col_idx}] lane {s}");
        }
    }
}

#[test]
fn static_16b_aligned_ldmatrix_skips_runtime_alignment_diamond() {
    // Real-world shared-memory alignment via shl/mad/cvta/loop induction: no runtime & 15 check.
    let text = "{\n	.reg .b32 %r<8>, %d<4>, %a<4>, %b<2>;\n	.reg .b64 %rd<4>;\n	.reg .pred %p1;\n        	.shared .align 16 .b8 smem[4096];\n        	mov.u32 %r0, %tid.x;\n        	and.b32 %r1, %r0, 31;\n        	mad.lo.s32 %r2, %r1, 136, 0;\n        	shl.b32 %r3, %r2, 1;\n        	mov.u32 %r4, smem;\n        	add.s32 %r5, %r4, %r3;\n        	cvt.u64.u32 %rd0, %r5;\n        	cvta.shared.u64 %rd1, %rd0;\n        $L_loop:\n        	ldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd1];\n        	ldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd1+128];\n        	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n        	add.s64 %rd1, %rd1, 256;\n        	@%p1 bra $L_loop;\n	ret;\n}";
    let lowered = lower(text);
    assert_eq!(
        count(&lowered, "$vf_u"),
        0,
        "expected static 16B proof to drop $vf_u:\n{lowered}"
    );
    assert_eq!(count(&lowered, ", 15;"), 0, "expected no & 15 runtime test:\n{lowered}");
    assert_eq!(
        count(&lowered, "ld.shared.v4.u32 "),
        4,
        "expected direct v4 loads:\n{lowered}"
    );
    assert_eq!(undeclared(&lowered), Vec::<String>::new());

    // Negative soundness controls: .align 4 base, +4 loop step, or predicated +4 must keep runtime check.
    let unaligned_step = text.replace("add.s64 %rd1, %rd1, 256;", "add.s64 %rd1, %rd1, 4;");
    assert_eq!(
        count(&lower(&unaligned_step), ", 15;"),
        2,
        "unaligned loop step must keep runtime check"
    );
    let guarded_misalign = text.replace("add.s64 %rd1, %rd1, 256;", "@%p1 add.s64 %rd1, %rd1, 4;");
    assert_eq!(
        count(&lower(&guarded_misalign), ", 15;"),
        2,
        "guarded misalign must keep runtime check"
    );
    let align4_base = text.replace(".shared .align 16 .b8 smem[4096];", ".shared .align 4 .b8 smem[4096];");
    assert_eq!(
        count(&lower(&align4_base), ", 15;"),
        2,
        ".align 4 shared must keep runtime check"
    );
}

#[test]
fn module_level_aligned_extern_shared_skips_runtime_alignment_diamond() {
    let module = ".version 8.0\n.target sm_80\n.address_size 64\n.extern .shared .align 16 .b8 gsm[];\n.visible .entry k()\n{\n\t.reg .b32 %r<6>, %a<4>, %b<2>, %d<4>;\n\tmov.u32 %r0, %tid.x;\n\tshl.b32 %r1, %r0, 4;\n\tmov.u32 %r2, gsm;\n\tadd.u32 %r3, %r2, %r1;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%r3];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [gsm+128];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\tret;\n}\n";
    let (out, _) = crate::lower_to_volta_with(
        module.as_bytes(),
        Options {
            aligned_rows: false,
            ..Options::default()
        },
    );
    let s = std::str::from_utf8(&out).unwrap();
    assert!(
        !s.contains("$vf_u"),
        "expected module-level .extern .shared .align 16 to skip $vf_u:\n{s}"
    );
    assert!(!s.contains(", 15;"), "expected no & 15 runtime check:\n{s}");
}

#[test]
fn guarded_branch_in_middle_of_block_propagates_state_at_branch() {
    // Maintainer's negative example: fall-through re-establishes alignment after `@%p bra $L`,
    // which must NOT reach `$L` along the back-edge where `%rd0` became `smem + 4`.
    let text = "{\n\t.reg .b32 %r<4>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd<3>;\n\t.reg .pred %p;\n\t.shared .align 16 .b8 smem[4096], smem_b[1024];\n\tmov.u64 %rd0, smem;\n\tmov.u64 %rd2, smem_b;\n$L:\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd2];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\tadd.u64 %rd0, %rd0, 4;\n\t@%p bra $L;\n\tmov.u64 %rd0, smem;\n\tret;\n}";
    let lowered = lower(text);
    assert!(
        count(&lowered, "ld.shared.v4.u32 ") > 0,
        "expected wide_loads path:\n{lowered}"
    );
    assert_eq!(
        count(&lowered, ", 15;"),
        1,
        "back-edge from @%p bra $L carries smem+4 and must keep runtime alignment check:\n{lowered}"
    );
}

#[test]
fn barrier_red_and_mbarrier_arrive_clear_written_register_alignment() {
    let base = "{\n\t.reg .b32 %r<6>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd<4>;\n\t.reg .pred %p;\n\t.shared .align 16 .b8 smem[4096], smem_b[1024];\n\t.shared .align 8 .b64 mbar;\n\tmov.u32 %r0, smem;\n\tmov.u64 %rd0, smem;\n\tmov.u64 %rd3, smem_b;\n\t{WRITE}\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [{ADDR}];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd3];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\tret;\n}";
    for (op, addr) in [
        ("bar.red.popc.u32 %r0, 0, %p;\n\tcvt.u64.u32 %rd0, %r0;", "%rd0"),
        ("barrier.cta.red.popc.u32 %r0, 0, %p;\n\tcvt.u64.u32 %rd0, %r0;", "%rd0"),
        ("mbarrier.arrive.shared::cta.b64 %rd0, [mbar];", "%rd0"),
    ] {
        let ptx = base.replace("{WRITE}", op).replace("{ADDR}", addr);
        let lowered = lower(&ptx);
        assert!(count(&lowered, "ld.shared.v4.u32 ") > 0);
        assert_eq!(
            count(&lowered, ", 15;"),
            1,
            "{op} writes {addr} and must invalidate prior alignment:\n{lowered}"
        );
    }
}

#[test]
fn bfi_takes_position_and_length_modulo_256() {
    // pos=256 (mod 256 == 0) overwrites the low 4 bits with unaligned `%r1`; must NOT be proved aligned.
    let unaligned_pos_mod256 = "{\n\t.reg .b32 %r<6>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd<2>;\n\t.shared .align 16 .b8 smem[4096], smem_b[1024];\n\tmov.u32 %r0, smem;\n\tmov.u64 %rd1, smem_b;\n\tbfi.b32 %r2, %r1, %r0, 256, 4;\n\tcvt.u64.u32 %rd0, %r2;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd1];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\tret;\n}";
    assert_eq!(
        count(&lower(unaligned_pos_mod256), ", 15;"),
        1,
        "bfi with pos=256 (0 mod 256) must take bits 0..3 from unaligned source"
    );

    // len=256 (mod 256 == 0) inserts zero bits, keeping `%r1` (unaligned) rather than `smem`.
    let unaligned_len_mod256 = "{\n\t.reg .b32 %r<6>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd<2>;\n\t.shared .align 16 .b8 smem[4096], smem_b[1024];\n\tmov.u32 %r0, smem;\n\tmov.u64 %rd1, smem_b;\n\tbfi.b32 %r2, %r0, %r1, 0, 256;\n\tcvt.u64.u32 %rd0, %r2;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd1];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\tret;\n}";
    assert_eq!(
        count(&lower(unaligned_len_mod256), ", 15;"),
        1,
        "bfi with len=256 (0 mod 256) inserts nothing and must preserve unaligned base"
    );

    // Positive controls modulo 256: len=256 preserves aligned `%r0`; pos=256, len=4 inserts `smem`'s low 4 zero bits.
    let aligned_len_mod256 = "{\n\t.reg .b32 %r<6>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd<2>;\n\t.shared .align 16 .b8 smem[4096], smem_b[1024];\n\tmov.u32 %r0, smem;\n\tmov.u64 %rd1, smem_b;\n\tbfi.b32 %r2, %r1, %r0, 0, 256;\n\tcvt.u64.u32 %rd0, %r2;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd1];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\tret;\n}";
    let out_len = lower(aligned_len_mod256);
    assert!(count(&out_len, "ld.shared.v4.u32 ") > 0);
    assert_eq!(count(&out_len, ", 15;"), 0);

    let aligned_pos_mod256 = "{\n\t.reg .b32 %r<6>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd<2>;\n\t.shared .align 16 .b8 smem[4096], smem_b[1024];\n\tmov.u32 %r0, smem;\n\tmov.u64 %rd1, smem_b;\n\tbfi.b32 %r2, %r0, %r1, 256, 4;\n\tcvt.u64.u32 %rd0, %r2;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd1];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\tret;\n}";
    let out_pos = lower(aligned_pos_mod256);
    assert!(count(&out_pos, "ld.shared.v4.u32 ") > 0);
    assert_eq!(count(&out_pos, ", 15;"), 0);
}

#[test]
fn shared_symbols_are_scoped_and_malformed_directive_does_not_panic() {
    // 1. Module `.shared .align 16` shadowed inside kernel by `.shared .align 4` / unaligned `.shared`.
    for kernel_decl in ["\t.shared .align 4 .b8 buf[64];", "\t.shared .b8 buf[64];"] {
        let module = format!(
            ".version 8.0\n.target sm_80\n.address_size 64\n.extern .shared .align 16 .b8 buf[];\n.shared .align 16 .b8 smem_b[1024];\n.visible .entry k()\n{{\n\t.reg .b32 %r<4>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd<2>;\n{kernel_decl}\n\tmov.u64 %rd0, buf;\n\tmov.u64 %rd1, smem_b;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {{%a0, %a1, %a2, %a3}}, [%rd0];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {{%b0, %b1}}, [%rd1];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%d0, %d1, %d2, %d3}}, {{%a0, %a1, %a2, %a3}}, {{%b0, %b1}}, {{%d0, %d1, %d2, %d3}};\n\tret;\n}}\n"
        );
        let (out, _) = crate::lower_to_volta_with(
            module.as_bytes(),
            Options {
                aligned_rows: false,
                ..Options::default()
            },
        );
        let s = std::str::from_utf8(&out).unwrap();
        assert!(count(s, "ld.shared.v4.u32 ") > 0);
        assert_eq!(
            count(s, ", 15;"),
            1,
            "shadowing declaration {kernel_decl:?} must cancel module .align 16:\n{s}"
        );
    }

    // 2. Inner `{ .shared .align 16 buf; }` must not leak out to outer `.shared .align 4 buf;`.
    let inner_leak = "{\n\t.reg .b32 %r<4>, %a<4>, %b<2>, %e<2>, %d<4>, %c<4>;\n\t.reg .b64 %rd<4>;\n\t.shared .align 4 .b8 buf[64];\n\t.shared .align 16 .b8 smem_b[1024];\n\tmov.u64 %rd2, smem_b;\n\t{\n\t\t.shared .align 16 .b8 buf[64];\n\t\tmov.u64 %rd0, buf;\n\t\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\t\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd2];\n\t\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\t}\n\tmov.u64 %rd1, buf;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%c0, %c1, %c2, %c3}, [%rd1];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%e0, %e1}, [%rd2];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%c0, %c1, %c2, %c3}, {%e0, %e1}, {%d0, %d1, %d2, %d3};\n\tret;\n}";
    let lowered_leak = lower(inner_leak);
    assert!(count(&lowered_leak, "ld.shared.v4.u32 ") > 0);
    assert_eq!(
        count(&lowered_leak, ", 15;"),
        1,
        "only inner ldmatrix is aligned 16, outer must keep runtime check:\n{lowered_leak}"
    );

    // 3. Outer `.shared .align 16 buf` shadowed inside `{ .shared .align 4 buf; }`.
    let outer_shadow = "{\n\t.reg .b32 %r<4>, %a<4>, %b<2>, %e<2>, %d<4>, %c<4>;\n\t.reg .b64 %rd<4>;\n\t.shared .align 16 .b8 buf[64], smem_b[1024];\n\tmov.u64 %rd2, smem_b;\n\t{\n\t\t.shared .align 4 .b8 buf[64];\n\t\tmov.u64 %rd0, buf;\n\t\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\t\tldmatrix.sync.aligned.m8n8.x2.b16 {%b0, %b1}, [%rd2];\n\t\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n\t}\n\tmov.u64 %rd1, buf;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%c0, %c1, %c2, %c3}, [%rd1];\n\tldmatrix.sync.aligned.m8n8.x2.b16 {%e0, %e1}, [%rd2];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%c0, %c1, %c2, %c3}, {%e0, %e1}, {%d0, %d1, %d2, %d3};\n\tret;\n}";
    let lowered_shadow = lower(outer_shadow);
    assert!(count(&lowered_shadow, "ld.shared.v4.u32 ") > 0);
    assert_eq!(
        count(&lowered_shadow, ", 15;"),
        1,
        "inner shadowed ldmatrix must keep runtime check, outer skips it:\n{lowered_shadow}"
    );
}

#[test]
fn malformed_shared_directive_does_not_panic() {
    let malformed = ".version 8.0\n.target sm_80\n.address_size 64\n.shared .align 16 .align;\n.visible .entry k()\n{\n\t.shared .align 16 .align ;\n\t.reg .b32 %r<4>, %a<4>, %b<2>, %d<4>;\n\t.reg .b64 %rd0;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\tmma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1}, {%b0}, {%d0, %d1, %d2, %d3};\n\tret;\n}\n";
    let (out, _) = crate::lower_to_volta_with(
        malformed.as_bytes(),
        Options {
            aligned_rows: false,
            ..Options::default()
        },
    );
    assert!(!out.is_empty());
}

#[test]
fn function_level_shared_declaration_holds_from_its_place_in_the_text() {
    // ptxas resolves a name in the order of the text, whatever the branches: a statement before a
    // function-level declaration names the module's symbol of that name, one after it the
    // function's (ptxas 12.9 compiles `mov.u32 %r4, buf` before `.shared .align 16 .b8 buf[4096]`
    // in the body to 0x1004, the module's `.align 4` buffer after `pad`, and after it to 0). A
    // 16-byte load through the module's `buf` keeps its check; each case counts the checks left.
    let module4 = ".shared .align 4 .b8 pad[4];\n.shared .align 4 .b8 buf[4096];\n";
    let module16 = ".shared .align 4 .b8 pad[4];\n.shared .align 16 .b8 buf[4096];\n";
    let decl16 = "\t.shared .align 16 .b8 buf[4096];\n";
    let loads = "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%a0, %a1, %a2, %a3}, [%r4];\n\tldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0, %b1}, [%r4+128];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n";
    let mov = "\tmov.u32 %r4, buf;\n";
    let cases = [
        ("a use before the declaration", module4, format!("{mov}{decl16}{loads}"), 2),
        ("no function-level declaration", module4, format!("{mov}{loads}"), 2),
        ("a use after the declaration", module4, format!("{decl16}{mov}{loads}"), 0),
        (
            "a block before the declaration, reached only from a block after it",
            module4,
            format!("\tbra $A;\n$B:\n{mov}{loads}\tret;\n$A:\n{decl16}\tbra $B;\n"),
            2,
        ),
        (
            "a loop whose head is before the declaration, entered with the module's buf",
            module4,
            format!("{mov}$L:\n{loads}{decl16}{mov}\tsetp.ne.u32 %p1, %r0, 0;\n\t@%p1 bra $L;\n"),
            2,
        ),
        (
            "a use in a nested scope before the declaration",
            module4,
            format!("\t{{\n{mov}\t}}\n{decl16}{loads}"),
            2,
        ),
        (
            "a nested scope's declaration shadows the function's",
            module4,
            format!("{decl16}\t{{\n\t.shared .align 4 .b8 buf[64];\n{mov}\t}}\n{loads}"),
            2,
        ),
        (
            "the function's declaration holds again after that nested scope",
            module4,
            format!("{decl16}\t{{\n\t.shared .align 4 .b8 buf[64];\n\tmov.u32 %r6, buf;\n\t}}\n{mov}{loads}"),
            0,
        ),
        (
            "a use of the module's .align 16 buf before the function's .align 4 buf, and one after",
            module16,
            "\tmov.u32 %r4, buf;\n\t.shared .align 4 .b8 buf[4096];\n\tmov.u32 %r6, buf;\n\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%a0, %a1, %a2, %a3}, [%r4];\n\tldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0, %b1}, [%r6];\n\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0, %d1, %d2, %d3}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%d0, %d1, %d2, %d3};\n".to_string(),
            1,
        ),
    ];
    for (what, symbols, body, checks) in cases {
        let module = format!(
            ".version 8.0\n.target sm_80\n.address_size 64\n{symbols}.visible .entry k(.param .u64 out)\n{{\n\t.reg .b32 %r<8>, %a<4>, %b<2>, %d<4>;\n\t.reg .pred %p1;\n\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r5, pad;\n\tst.shared.u32 [%r5], %r0;\n{body}\tret;\n}}\n"
        );
        let (out, _) = crate::lower_to_volta_with(
            module.as_bytes(),
            Options {
                aligned_rows: false,
                ..Options::default()
            },
        );
        let s = std::str::from_utf8(&out).unwrap();
        assert!(
            count(s, "ld.shared.v4.u32 ") > 0,
            "{what}: expected the wide loads:\n{s}"
        );
        assert_eq!(count(s, ", 15;"), checks, "{what}:\n{s}");
    }
}
