//! FP8 and block-scaled MMA products: an FP8 product as four FP16 products of the decoded
//! halves, the block-scaled product with E2M1 or FP8 operands and UE8M0 or UE4M3 scales,
//! and the helpers that decode scale bytes and E2M1 nibbles into halves for the tensor cores.

use std::fmt::Write as _;

use super::body::Body;
use super::lane::Lane;
use super::types::{Fp8, ScaledKind};

impl<'a> Body<'a> {
    /// FP8 `mma.m16n8k32` with f32 accumulators, on Volta's tensor cores. A thread holds, in each
    /// register of A and of B, four consecutive k of its row or column (and the same k in both).
    /// Their halves two by two are the operands of an FP16 `mma.m16n8k8`, whose layout gives a
    /// thread two k: four such products in a row, for bytes 0-1 and 2-3 of the first sixteen k and
    /// of the last sixteen.
    pub(super) fn mma_fp8(&mut self, a_type: Fp8, b_type: Fp8, d: &[&str], a: &[&str], b: &[&str], c: &[&str]) {
        let mut of_a = Vec::new();
        for rows in a.chunks(2) {
            let (upper, lower) = (self.fp8_halves(a_type, rows[0]), self.fp8_halves(a_type, rows[1]));
            of_a.extend([upper[0].clone(), lower[0].clone(), upper[1].clone(), lower[1].clone()]);
        }
        let of_b: Vec<String> = b.iter().flat_map(|b| self.fp8_halves(b_type, b)).collect();
        let (of_a, of_b): (Vec<&str>, Vec<&str>) = (
            of_a.iter().map(String::as_str).collect(),
            of_b.iter().map(String::as_str).collect(),
        );
        self.mma_f32(d, &of_a, &of_b, c);
    }

    /// The eight E2M1 nibbles of a `.b32` register as four f16x2 halves, one per byte (nibble pair),
    /// using the PRMT/LOP3 LUT (`0x3E3C3800`, `0x46444240`, zero `HMUL2`).
    /// Halves `[0]` has nibbles 0,1 (`bits 7:0`), `[1]` has 2,3 (`bits 15:8`), `[2]` has 4,5
    /// (`bits 23:16`), `[3]` has 6,7 (`bits 31:24`).
    pub(super) fn e2m1_all_halves(&mut self, register: &str) -> Vec<String> {
        let [mag, mag_hi, m0, m1, sign_even]: [String; 5] = self.fresh_n();
        emit!(self, "and.b32 {mag}, {register}, 0x77777777;");
        emit!(self, "shr.u32 {mag_hi}, {mag}, 16;");
        emit!(self, "prmt.b32 {m0}, 0x3E3C3800, 0x46444240, {mag};");
        emit!(self, "prmt.b32 {m1}, 0x3E3C3800, 0x46444240, {mag_hi};");
        emit!(self, "shl.b32 {sign_even}, {register}, 4;");
        let mut halves = Vec::with_capacity(4);
        for ((m, unpack), sign_sel) in [(&m0, "0x1404"), (&m0, "0x3424"), (&m1, "0x1404"), (&m1, "0x3424")]
            .into_iter()
            .zip(["0xC080", "0xD090", "0xE0A0", "0xF0B0"])
        {
            let [p, s, h]: [String; 3] = self.fresh_n();
            emit!(self, "prmt.b32 {p}, {m}, 0, {unpack};");
            emit!(self, "prmt.b32 {s}, {sign_even}, {register}, {sign_sel};");
            emit!(self, "lop3.b32 {h}, {p}, {s}, 0x80008000, 0xF8;");
            halves.push(h);
        }
        halves
    }

    /// A UE4M3 scale byte decoded to `value * 2^-8` (or `value` if `mul256`) as f32.
    /// Every UE4M3 value (0 to 448) and `value * 2^-8` (`2^-17` to `1.75`) is exact and normal
    /// (or zero) in f32; byte `0x7f` produces canonical quiet NaN (`0x7fc00000`). Keeping the
    /// `2^-8` factor lets callers fold `2^16` once into row scale `scale_A` instead of multiplying
    /// both `scale_A` and four `scale_B` columns by `256` per block.
    pub(super) fn ue4m3_byte_f32(&mut self, register: &str, byte_offset: u32, mul256: bool) -> String {
        let placed = self.fresh();
        let half = self.fresh_half();
        let f32_val = self.fresh();
        self.scratch_predicate = true;
        match byte_offset {
            0 => emit!(self, "shl.b32 {placed}, {register}, 7;"),
            1 => emit!(self, "shr.u32 {placed}, {register}, 1;"),
            2 => emit!(self, "shr.u32 {placed}, {register}, 9;"),
            3 => emit!(self, "shr.u32 {placed}, {register}, 17;"),
            _ => unreachable!(),
        }
        emit!(self, "and.b32 {placed}, {placed}, 0x3f80;"); // keep bits 13:7
        emit!(self, "cvt.u16.u32 {half}, {placed};");
        emit!(self, "cvt.f32.f16 {f32_val}, {half};");
        if mul256 {
            emit!(self, "mul.f32 {f32_val}, {f32_val}, 0f43800000;"); // * 256
        }
        // Byte 0x7f (all seven bits set) is NaN: placed == 0x3f80 iff low 7 bits of the byte are 1.
        emit!(self, "setp.eq.u32 %vf_n, {placed}, 0x3f80;");
        emit!(self, "selp.b32 {f32_val}, 0x7fc00000, {f32_val}, %vf_n;");
        f32_val
    }

    /// A scale factor from a `.b32` register: the byte at position `byte_offset` within the register,
    /// decoded as UE8M0 (a power of two) or UE4M3 (unsigned FP8) to an f64. Using f64 avoids
    /// overflow: UE8M0 scales go up to 2^127, and their product can reach 2^254, which overflows
    /// f32 but not f64.
    pub(super) fn scale_byte_f64(&mut self, register: &str, byte_offset: u32, ue4m3: bool) -> String {
        self.scale_byte_f64_biased(register, byte_offset, ue4m3, 0)
    }

    pub(super) fn scale_byte_f64_biased(
        &mut self,
        register: &str,
        byte_offset: u32,
        ue4m3: bool,
        exp_boost: u32,
    ) -> String {
        let byte = self.fresh();
        match byte_offset {
            0 => emit!(self, "and.b32 {byte}, {register}, 0xff;"),
            1 => {
                emit!(self, "shr.u32 {byte}, {register}, 8;");
                emit!(self, "and.b32 {byte}, {byte}, 0xff;");
            }
            2 => {
                emit!(self, "shr.u32 {byte}, {register}, 16;");
                emit!(self, "and.b32 {byte}, {byte}, 0xff;");
            }
            3 => emit!(self, "shr.u32 {byte}, {register}, 24;"),
            _ => unreachable!(),
        }
        if ue4m3 {
            // UE4M3: unsigned, 4 exponent bits (bias 7), 3 mantissa bits. Max 448, min nonzero
            // 2^-9. Every value is an exact half: the seven bits, at bits 13:7 of a half, give the
            // value divided by 256 (exponent bias differs by 8), and times 256 through f32 is the
            // value. The byte 0x7f (the NaN of E4M3) becomes NaN.
            // All UE4M3 values fit exactly in f64 (they are at most 448).
            let [placed, nan_check]: [String; 2] = self.fresh_n();
            let half = self.fresh_half();
            let f32_val = self.fresh();
            let f64_val = self.fresh_wide();
            self.scratch_predicate = true;
            emit!(self, "shl.b32 {placed}, {byte}, 7;");
            emit!(self, "and.b32 {placed}, {placed}, 0x3f80;"); // keep bits 13:7
            emit!(self, "cvt.u16.u32 {half}, {placed};");
            emit!(self, "cvt.f32.f16 {f32_val}, {half};");
            emit!(self, "mul.f32 {f32_val}, {f32_val}, 0f43800000;"); // * 256
            emit!(self, "cvt.f64.f32 {f64_val}, {f32_val};");
            // Byte 0x7f (all seven bits set) is the NaN: make the f64 a NaN.
            emit!(self, "and.b32 {nan_check}, {byte}, 0x7f;");
            emit!(self, "setp.eq.u32 %vf_n, {nan_check}, 0x7f;");
            // 0x7ff8000000000000 is a quiet NaN in f64; the predicate `%vf_n` still holds the test.
            let [nan_hi, nan_lo]: [String; 2] = self.fresh_n();
            emit!(self, "mov.b32 {nan_hi}, 0x7ff80000;");
            emit!(self, "mov.b32 {nan_lo}, 0;");
            let nan_d = self.fresh_wide();
            emit!(self, "mov.b64 {nan_d}, {{{nan_lo}, {nan_hi}}};");
            let result = self.fresh_wide();
            emit!(self, "selp.b64 {result}, {nan_d}, {f64_val}, %vf_n;");
            result
        } else {
            // UE8M0: 2^(e - 127). Build as f64 directly: the f64 biased exponent is e - 127 + 1023 =
            // e + 896. The f64 is bits [62:52] = exponent, mantissa = 0.
            // Byte 0 is 2^-127 (not zero): f64 biased exponent 896 = 0x380.
            // Byte 0xff is NaN in UE8M0: return f64 NaN (0x7ff8000000000000).
            let [exp, hi, lo]: [String; 3] = self.fresh_n();
            let result = self.fresh_wide();
            self.scratch_predicate = true;
            emit!(self, "add.u32 {exp}, {byte}, {};", 896 + exp_boost);
            emit!(self, "shl.b32 {hi}, {exp}, 20;"); // bits 30:20 of hi word = exponent bits 10:0
            emit!(self, "mov.b32 {lo}, 0;");
            emit!(self, "mov.b64 {result}, {{{lo}, {hi}}};");
            // Byte 0xff: set NaN.
            let nan = self.fresh_wide();
            emit!(self, "setp.eq.u32 %vf_n, {byte}, 0xff;");
            let [nan_hi2, nan_lo2]: [String; 2] = self.fresh_n();
            emit!(self, "mov.b32 {nan_hi2}, 0x7ff80000;");
            emit!(self, "mov.b32 {nan_lo2}, 0;");
            emit!(self, "mov.b64 {nan}, {{{nan_lo2}, {nan_hi2}}};");
            emit!(self, "selp.b64 {result}, {nan}, {result}, %vf_n;");
            result
        }
    }

    /// Block-scaled MMA: D = (A * scale_A) * (B * scale_B) + C, on Volta's tensor cores.
    ///
    /// For E2M1 operands (kind::mxf4, kind::mxf4nvf4): the nibbles are decoded to exact halves, then
    /// each block of k is run as FP16 products with f32 accumulators; the f32 partial sum of each
    /// block is then multiplied by (scale_A * scale_B) in f64 and added to an f64 accumulator. The
    /// final sum is converted to f32 by truncation toward zero for four scales (`cvt.rz.f32.f64`) and
    /// by rounding to nearest even for two scales or one (`cvt.rn.f32.f64`).
    ///
    /// Why f64 for the scales: UE8M0 scales go up to 2^127; their product reaches 2^254, which
    /// overflows f32 (max ~2^128) but is exact in f64 (which covers 2^1023). The inner block sum of
    /// E2M1 products is at most 32 * 36 = 1152 (for 32 k-elements, all products +6 * +6 = 36), well
    /// within f32's exact integer range. The product scale_A * scale_B * block_sum fits in f64 without
    /// rounding when both scales are powers of two (UE8M0: a power of two times a 23-bit mantissa
    /// integer is exact in f64 which has 52 mantissa bits). For UE4M3 scales (max 448), the product
    /// scale_A * scale_B * block_sum <= 448 * 448 * 576 = 115,605,504, an integer that fits exactly
    /// in f64. The f64 accumulation across blocks is also exact for the sums that arise.
    ///
    /// For E4M3/E5M2 operands (kind::mxf8f6f4): the bytes are decoded to halves (the existing FP8
    /// path), with the single-block scale applied after.
    ///
    /// A NaN scale makes every result of its row or column 0x7fffffff (measured on the RTX 5090:
    /// a NaN in scale_A propagates to all columns of that row half; a NaN in scale_B to all rows of
    /// that column). The layer returns 0x7fffffff for any result whose scale product is NaN.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mma_block_scaled(
        &mut self,
        kind: ScaledKind,
        d: &[&str],
        a: &[&str],
        b: &[&str],
        c: &[&str],
        scale_a: &str,
        byte_id_a: u32,
        thread_id_a: u32,
        scale_b: &str,
        byte_id_b: u32,
        thread_id_b: u32,
    ) {
        if matches!(kind, ScaledKind::Mxf4 | ScaledKind::Nvf4Ue4m3x4) {
            self.mma_block_scaled_fp4(
                kind,
                d,
                a,
                b,
                c,
                scale_a,
                byte_id_a,
                thread_id_a,
                scale_b,
                byte_id_b,
                thread_id_b,
            );
            return;
        }
        let ScaledKind::Mxf8f6f4(a_type, b_type) = kind else {
            unreachable!();
        };

        let sa_lower_thread = if thread_id_a == 0 { 0 } else { 2 };
        let sa_upper_thread = sa_lower_thread + 1;
        let sa_lower_reg = self.shuffle(scale_a, Lane::InGroup(sa_lower_thread));
        let sa_upper_reg = self.shuffle(scale_a, Lane::InGroup(sa_upper_thread));
        let sb_col0_reg = self.shuffle(scale_b, Lane::InPair(thread_id_b));
        let sb_col1_reg = self.shuffle(scale_b, Lane::InPair(4 + thread_id_b));

        // Absorb 2^8 per unscaled E4M3 operand into UE8M0's exponent bias (eliminating HMUL2).
        let e4m3_exp_boost = 8 * u32::from(a_type == Fp8::E4m3) + 8 * u32::from(b_type == Fp8::E4m3);
        let scale_a_lower_f64 = self.scale_byte_f64_biased(&sa_lower_reg, byte_id_a, false, e4m3_exp_boost);
        let scale_a_upper_f64 = self.scale_byte_f64_biased(&sa_upper_reg, byte_id_a, false, e4m3_exp_boost);
        let scale_b_col0_f64 = self.scale_byte_f64(&sb_col0_reg, byte_id_b, false);
        let scale_b_col1_f64 = self.scale_byte_f64(&sb_col1_reg, byte_id_b, false);
        let mut combined_f64: [String; 4] = Default::default();
        for (i, (sa, sb)) in [
            (&scale_a_lower_f64, &scale_b_col0_f64),
            (&scale_a_lower_f64, &scale_b_col1_f64),
            (&scale_a_upper_f64, &scale_b_col0_f64),
            (&scale_a_upper_f64, &scale_b_col1_f64),
        ]
        .iter()
        .enumerate()
        {
            let comb = self.fresh_wide();
            emit!(self, "mul.f64 {comb}, {sa}, {sb};");
            combined_f64[i] = comb;
        }

        let sums: Vec<String> = (0..4).map(|_| self.fresh_wide()).collect();
        for (i, sum) in sums.iter().enumerate() {
            emit!(self, "cvt.f64.f32 {sum}, {};", c[i]);
        }

        let block_sums: Vec<String> = (0..4).map(|_| self.fresh()).collect();
        for sum in &block_sums {
            emit!(self, "mov.b32 {sum}, 0f00000000;");
        }
        let block_sum_names: Vec<&str> = block_sums.iter().map(String::as_str).collect();
        let mut of_a = Vec::new();
        for rows in a.chunks(2) {
            let (upper, lower) = (
                self.fp8_halves_scaled(a_type, rows[0], false),
                self.fp8_halves_scaled(a_type, rows[1], false),
            );
            of_a.extend([upper[0].clone(), lower[0].clone(), upper[1].clone(), lower[1].clone()]);
        }
        let of_b: Vec<String> = b
            .iter()
            .flat_map(|b_reg| self.fp8_halves_scaled(b_type, b_reg, false))
            .collect();
        let (of_a_refs, of_b_refs): (Vec<&str>, Vec<&str>) = (
            of_a.iter().map(String::as_str).collect(),
            of_b.iter().map(String::as_str).collect(),
        );
        self.mma_f32(&block_sum_names, &of_a_refs, &of_b_refs, &block_sum_names);
        for i in 0..4 {
            let f64_sum = self.fresh_wide();
            emit!(self, "cvt.f64.f32 {f64_sum}, {};", block_sums[i]);
            let scaled = self.fresh_wide();
            emit!(self, "mul.f64 {scaled}, {f64_sum}, {};", combined_f64[i]);
            emit!(self, "add.f64 {}, {}, {scaled};", sums[i], sums[i]);
        }

        for name in d {
            self.written(name);
        }
        self.scratch_predicate = true;
        for (i, name) in d.iter().enumerate() {
            let f32_val = self.fresh();
            emit!(self, "cvt.rn.f32.f64 {f32_val}, {};", sums[i]);
            emit!(self, "setp.nan.f64 %vf_n, {}, {};", sums[i], sums[i]);
            emit!(self, "selp.b32 {f32_val}, 0x7fffffff, {f32_val}, %vf_n;");
            emit!(self, "mov.b32 {name}, {f32_val};");
        }
    }

    /// Block-scaled FP4 (`kind::mxf4` and `kind::mxf4nvf4`) in the single-instruction split-octet
    /// `mma.m8n8k4.row.col` arrangement:
    /// - Packed E2M1 registers are shuffled before decoding (one 32-bit shuffle instead of four
    ///   decoded halves, cutting NVFP4 operand shuffles from 88 to 12 and eliminating `columns`),
    /// - Each 8-nibble register is decoded in place via `e2m1_all_halves` (PRMT/LOP3 LUT, 0 `HMUL2`),
    /// - Only 8 `mma.m8n8k4.row.col` instructions (= 32 Volta HMMA steps) execute instead of 16
    ///   (= 64 HMMA steps), because E2M1 block sums fit exactly in f32 (<= 1152 as multiples of 0.25),
    /// - The f64 accumulator stays in split-octet layout across all blocks and converts back to
    ///   Turing `d[0..4]` once at the end via `Lane::ChainOut`.
    #[allow(clippy::too_many_arguments)]
    fn mma_block_scaled_fp4(
        &mut self,
        kind: ScaledKind,
        d: &[&str],
        a: &[&str],
        b: &[&str],
        c: &[&str],
        scale_a: &str,
        byte_id_a: u32,
        thread_id_a: u32,
        scale_b: &str,
        byte_id_b: u32,
        thread_id_b: u32,
    ) {
        let ue4m3 = kind.ue4m3_scale();
        let n_blocks = kind.blocks();

        // Scale registers in split-octet layout `S`: one row (`sa_reg`) and four columns (`sb_regs`).
        let sa_reg = self.shuffle(scale_a, Lane::ScaledSa(thread_id_a));
        let sb_regs: [String; 4] =
            std::array::from_fn(|col_idx| self.shuffle(scale_b, Lane::ScaledSb(col_idx as u32, thread_id_b)));

        let byte_offset_a = |block: u32| -> u32 { if n_blocks == 2 { byte_id_a + block } else { block } };
        let byte_offset_b = |block: u32| -> u32 { if n_blocks == 2 { byte_id_b + block } else { block } };

        // Initialize the 4 f64 accumulators in split-octet layout (ordered as `2 * parity + b1`).
        let is_rh1 = self.lane(Lane::Has(4));
        let sums: [String; 4] = std::array::from_fn(|_| self.fresh_wide());
        if c.iter().all(|name| self.zeros.contains(*name)) {
            for sum in &sums {
                emit!(self, "mov.f64 {sum}, 0d0000000000000000;");
            }
        } else {
            let c_lanes = [self.lane(Lane::ScaledC(0)), self.lane(Lane::ScaledC(1))];
            for parity in 0..2 {
                for (b1, c_lane) in c_lanes.iter().enumerate() {
                    let idx = 2 * parity + b1;
                    let [from_rh0, from_rh1, chosen]: [String; 3] = self.fresh_n();
                    emit!(self, "shfl.sync.idx.b32 {from_rh0}, {}, {c_lane}, 31, -1;", c[parity],);
                    self.shuffles += 1;
                    emit!(
                        self,
                        "shfl.sync.idx.b32 {from_rh1}, {}, {c_lane}, 31, -1;",
                        c[2 + parity],
                    );
                    self.shuffles += 1;
                    emit!(self, "selp.b32 {chosen}, {from_rh1}, {from_rh0}, {is_rh1};");
                    emit!(self, "cvt.f64.f32 {}, {chosen};", sums[idx]);
                }
            }
        }

        // One zero register repeated across the 8 C slots compiles directly to RZ in HMMA.
        let zero_f32 = self.fresh();
        emit!(self, "mov.b32 {zero_f32}, 0f00000000;");
        let zero_c = [zero_f32.as_str(); 8].join(", ");
        let upper = self.lane(Lane::Has(8));

        for block in 0..n_blocks {
            let steps: Vec<(String, String, String, String)> = match kind {
                ScaledKind::Nvf4Ue4m3x4 => {
                    let reg_pair = block >> 1;
                    let src = Lane::Nvf4Src((block & 1) as u32);
                    let au_shuf = self.shuffle(a[2 * reg_pair], src);
                    let al_shuf = self.shuffle(a[2 * reg_pair + 1], src);
                    let b_shuf = self.shuffle(b[reg_pair], src);
                    let a_sel = self.fresh();
                    emit!(self, "selp.b32 {a_sel}, {al_shuf}, {au_shuf}, {is_rh1};");
                    let ah = self.e2m1_all_halves(&a_sel);
                    let bh = self.e2m1_all_halves(&b_shuf);
                    vec![
                        (ah[0].clone(), ah[2].clone(), bh[0].clone(), bh[2].clone()),
                        (ah[1].clone(), ah[3].clone(), bh[1].clone(), bh[3].clone()),
                    ]
                }
                ScaledKind::Mxf4 => {
                    let au_w0 = self.shuffle(a[2 * block], Lane::ChainW0);
                    let al_w0 = self.shuffle(a[2 * block + 1], Lane::ChainW0);
                    let a_w0 = self.fresh();
                    emit!(self, "selp.b32 {a_w0}, {al_w0}, {au_w0}, {is_rh1};");

                    let au_w1 = self.shuffle(a[2 * block], Lane::ChainW1);
                    let al_w1 = self.shuffle(a[2 * block + 1], Lane::ChainW1);
                    let a_w1 = self.fresh();
                    emit!(self, "selp.b32 {a_w1}, {al_w1}, {au_w1}, {is_rh1};");

                    let b_w0 = self.shuffle(b[block], Lane::ChainW0);
                    let b_w1 = self.shuffle(b[block], Lane::ChainW1);

                    let ah0 = self.e2m1_all_halves(&a_w0);
                    let ah1 = self.e2m1_all_halves(&a_w1);
                    let bh0 = self.e2m1_all_halves(&b_w0);
                    let bh1 = self.e2m1_all_halves(&b_w1);
                    (0..4)
                        .map(|s| (ah0[s].clone(), ah1[s].clone(), bh0[s].clone(), bh1[s].clone()))
                        .collect()
                }
                _ => unreachable!(),
            };

            let mut bs: [String; 8] = self.fresh_n();
            for (step, (aw0, aw1, bw0, bw1)) in steps.into_iter().enumerate() {
                let (c_ops, next_bs): (String, [String; 8]) = if step == 0 {
                    (zero_c.clone(), bs)
                } else {
                    (bs.join(", "), self.fresh_n())
                };
                emit!(
                    self,
                    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {{{}}}, {{{aw0}, {aw1}}}, {{{bw0}, {bw1}}}, {{{c_ops}}};",
                    next_bs.join(", "),
                );
                bs = next_bs;
            }

            // Combine k_half=0 and k_half=1 across lanes S and S^8 (index: 2 * parity + b1).
            let mut block_sums = Vec::with_capacity(4);
            for parity in 0..2 {
                for b1 in 0..2 {
                    let (low, high) = (&bs[parity + 4 * b1], &bs[parity + 4 * b1 + 2]);
                    let [mine, given, sum]: [String; 3] = self.fresh_n();
                    emit!(self, "selp.b32 {mine}, {high}, {low}, {upper};");
                    emit!(self, "selp.b32 {given}, {low}, {high}, {upper};");
                    let theirs = self.exchange(&given, 8);
                    emit!(self, "add.f32 {sum}, {mine}, {theirs};");
                    block_sums.push(sum);
                }
            }

            if ue4m3 {
                let sa_raw = self.ue4m3_byte_f32(&sa_reg, byte_offset_a(block as u32), false);
                let sa_f32 = self.fresh();
                emit!(self, "mul.f32 {sa_f32}, {sa_raw}, 0f47800000;"); // * 2^16 cancels two 2^-8s
                for i in 0..4 {
                    let sb_f32 = self.ue4m3_byte_f32(&sb_regs[i], byte_offset_b(block as u32), false);
                    let comb = self.fresh();
                    emit!(self, "mul.f32 {comb}, {sa_f32}, {sb_f32};");
                    let scaled = self.fresh();
                    emit!(self, "mul.f32 {scaled}, {}, {comb};", block_sums[i]);
                    let f64_scaled = self.fresh_wide();
                    emit!(self, "cvt.f64.f32 {f64_scaled}, {scaled};");
                    emit!(self, "add.f64 {}, {}, {f64_scaled};", sums[i], sums[i]);
                }
            } else {
                let sa_f64 = self.scale_byte_f64(&sa_reg, byte_offset_a(block as u32), false);
                for i in 0..4 {
                    let sb_f64 = self.scale_byte_f64(&sb_regs[i], byte_offset_b(block as u32), false);
                    let comb = self.fresh_wide();
                    emit!(self, "mul.f64 {comb}, {sa_f64}, {sb_f64};");
                    let f64_sum = self.fresh_wide();
                    emit!(self, "cvt.f64.f32 {f64_sum}, {};", block_sums[i]);
                    let scaled = self.fresh_wide();
                    emit!(self, "mul.f64 {scaled}, {f64_sum}, {comb};");
                    emit!(self, "add.f64 {}, {}, {scaled};", sums[i], sums[i]);
                }
            }
            self.split();
        }

        // Convert to f32 in S with the required rounding and NaN handling, then scatter to Turing D.
        let rounding = if n_blocks == 4 { "rz" } else { "rn" };
        self.scratch_predicate = true;
        let f32_sums: [String; 4] = std::array::from_fn(|i| {
            let f32_val = self.fresh();
            emit!(self, "cvt.{rounding}.f32.f64 {f32_val}, {};", sums[i]);
            emit!(self, "setp.nan.f64 %vf_n, {}, {};", sums[i], sums[i]);
            emit!(self, "selp.b32 {f32_val}, 0x7fffffff, {f32_val}, %vf_n;");
            f32_val
        });

        let from = [self.lane(Lane::ChainOut(false)), self.lane(Lane::ChainOut(true))];
        let second = self.lane(Lane::Has(2));
        for name in d {
            self.written(name);
        }
        for (j, name) in d.iter().enumerate() {
            let pair = [0, 1].map(|b1| {
                let fetched = self.fresh();
                emit!(
                    self,
                    "shfl.sync.idx.b32 {fetched}, {}, {}, 31, -1;",
                    f32_sums[2 * (j & 1) + b1],
                    from[j >> 1]
                );
                self.shuffles += 1;
                fetched
            });
            emit!(self, "selp.b32 {name}, {}, {}, {second};", pair[1], pair[0]);
        }
        self.split();
    }
}
