//! Type conversions and element-wise sums: gathering operands from other lanes, the sums
//! that each thread computes for bf16, tf32 and f64 products, and the decoding and encoding
//! of FP8 (e4m3, e5m2), E2M1 and UE8M0 values.

use std::fmt::Write as _;

use super::Unsupported;
use super::body::Body;
use super::lane::Lane;
use super::plan::not_loaded;
use super::types::{Element, Form, Fp8};

impl<'a> Body<'a> {
    /// The row of A and the columns of B for one integer `mma.m8n8k16` sub-product.
    pub(super) fn byte_operands(&mut self, a: &str, b: &str) -> Result<(Vec<String>, Vec<String>), Unsupported> {
        let row = if self.plan.row.contains(a) {
            // The row of the thread's group, from where the `ldmatrix` found it. Volatile: the
            // compiler must not keep the four registers from one product to the next.
            let address = self.row_at.get(a).cloned().ok_or_else(|| not_loaded(a))?;
            let row: [String; 4] = self.fresh_n();
            emit!(self, "ld.volatile.shared.v4.u32 {{{}}}, [{address}];", row.join(", "));
            row.to_vec()
        } else {
            self.gathered(Form::Row, a, false)
        };
        Ok((row, self.gathered(Form::ColumnPair, b, false)))
    }

    /// The register `source` of other lanes: of the four threads of the group (`Form::Row`: with a
    /// register of A, the row of the group) or of the eight threads of the groups `2 * thread` and
    /// `2 * thread + 1` (`Form::ColumnPair`: with a register of B, the two columns of the results of
    /// the thread). `wide`: the register has 64 bits, and so have the gathered ones.
    pub(super) fn gathered(&mut self, form: Form, source: &str, wide: bool) -> Vec<String> {
        if let Some(found) = self.recall(form, &[source]) {
            return found;
        }
        let lanes: Vec<Lane> = match form {
            Form::Row => (0..4).map(Lane::InGroup).collect(),
            _ => (0..8).map(Lane::InPair).collect(),
        };
        let values: Vec<String> = if wide {
            let [low, high]: [String; 2] = self.fresh_n();
            emit!(self, "mov.b64 {{{low}, {high}}}, {source};");
            lanes
                .into_iter()
                .map(|lane| {
                    let (low, high) = (self.shuffle(&low, lane), self.shuffle(&high, lane));
                    let value = self.fresh_wide();
                    emit!(self, "mov.b64 {value}, {{{low}, {high}}};");
                    value
                })
                .collect()
        } else {
            lanes.into_iter().map(|lane| self.shuffle(source, lane)).collect()
        };
        self.remember(form, &[source], &values);
        values
    }

    /// A product whose sums each thread computes itself (`Tensor::Sums`). The thread of group g and
    /// number t holds the results of rows g (and g + 8) and columns 2t and 2t + 1. The rows of A are
    /// spread over the threads of its group, the columns of B over the threads of the groups 2t and
    /// 2t + 1, the same k in the same thread of each: the sum over k is a sum over the four threads,
    /// the registers and the elements of a register.
    ///
    /// The terms are added one after the other with `fma`, in the order of k, each sum rounded to
    /// nearest. The manual does not say how the hardware adds; an RTX 5090 keeps more bits than an
    /// f32 while it adds and cuts the result toward zero, so that on values whose sums are not
    /// exact about one result in six of a bf16 product differs from this one in its last bits
    /// (`docs/emulations.md`, "bf16"). What the hardware does with a tf32 operand it does say, by its results: the
    /// 13 low bits of the mantissa are dropped, whatever the value (a NaN with nothing else is an
    /// infinity), and so they are here.
    pub(super) fn mma_sums(&mut self, element: Element, d: &[&str], a: &[&str], b: &[&str], c: &[&str]) {
        let wide = element == Element::F64;
        let (halves, kind) = (c.len() / 2, if wide { "f64" } else { "f32" });
        let fresh = |body: &mut Self| if wide { body.fresh_wide() } else { body.fresh() };
        // Each sum starts from its accumulator; all are complete before a result is written, which
        // may be an operand.
        let sums: Vec<String> = (0..c.len()).map(|_| fresh(self)).collect();
        let operands = |body: &mut Self, registers: &[&str]| -> Vec<String> {
            let as_taken = |register: &&str| match element {
                Element::Tf32 => body.tf32(register),
                _ => register.to_string(),
            };
            registers.iter().map(as_taken).collect()
        };
        let (a, b) = (operands(self, a), operands(self, b));
        let mut started = false;
        for (block, b) in b.iter().enumerate() {
            let columns = self.gathered(Form::ColumnPair, b, wide);
            let rows: Vec<Vec<String>> = (0..halves)
                .map(|half| self.gathered(Form::Row, &a[halves * block + half], wide))
                .collect();
            for thread in 0..4 {
                // The elements of the registers that this thread of the group and of the two column
                // groups holds: of a row of A for each half, of B for each of the two columns.
                let of_a: Vec<Vec<String>> = rows.iter().map(|row| self.elements(element, &row[thread])).collect();
                let of_b: Vec<Vec<String>> = (0..2)
                    .map(|column| self.elements(element, &columns[4 * column + thread]))
                    .collect();
                for part in 0..of_b[0].len() {
                    for (half, of_a) in of_a.iter().enumerate() {
                        for (column, of_b) in of_b.iter().enumerate() {
                            let sum = &sums[2 * half + column];
                            let so_far = if started { sum.as_str() } else { c[2 * half + column] };
                            emit!(self, "fma.rn.{kind} {sum}, {}, {}, {so_far};", of_a[part], of_b[part]);
                        }
                    }
                    started = true;
                }
            }
            self.split();
        }
        for name in d {
            self.written(name);
        }
        for (name, sum) in d.iter().zip(&sums) {
            emit!(self, "mov.b{} {name}, {sum};", if wide { 64 } else { 32 });
        }
    }

    /// `register`, a tf32 operand, with the 13 low bits of its mantissa cleared, made once for the
    /// products that read the register.
    pub(super) fn tf32(&mut self, register: &str) -> String {
        if let Some(found) = self.recall(Form::Tf32, &[register]) {
            return found[0].clone();
        }
        let kept = self.fresh();
        emit!(self, "and.b32 {kept}, {register}, 0xffffe000;");
        self.remember(Form::Tf32, &[register], std::slice::from_ref(&kept));
        kept
    }

    /// The elements of `register` as `fma` takes them: itself, or the two bf16 it holds as f32 (a bf16
    /// is the upper half of an f32), made once for the products of a block that read the register.
    pub(super) fn elements(&mut self, element: Element, register: &str) -> Vec<String> {
        if element != Element::Bf16 {
            return vec![register.to_string()];
        }
        if let Some(found) = self.recall(Form::Bf16, &[register]) {
            return found;
        }
        let both: [String; 2] = self.fresh_n();
        emit!(self, "shl.b32 {}, {register}, 16;", both[0]);
        emit!(self, "and.b32 {}, {register}, 0xffff0000;", both[1]);
        self.remember(Form::Bf16, &[register], &both);
        both.to_vec()
    }

    /// The four FP8 bytes of `register` as halves, exactly: bytes 0 and 1 in one register, bytes 2
    /// and 3 in another.
    pub(super) fn fp8_halves(&mut self, kind: Fp8, register: &str) -> Vec<String> {
        self.fp8_halves_scaled(kind, register, true)
    }

    /// Four FP8 bytes as halves; when `kind == Fp8::E4m3` and `scale_256 == false`, values remain
    /// divided by 256 so a subsequent power-of-two scale (such as UE8M0) absorbs the 2^8 factor.
    pub(super) fn fp8_halves_scaled(&mut self, kind: Fp8, register: &str, scale_256: bool) -> Vec<String> {
        let form = if kind == Fp8::E4m3 && !scale_256 {
            Form::E4m3Unscaled
        } else {
            kind.form()
        };
        if let Some(found) = self.recall(form, &[register]) {
            return found;
        }
        let halves = if kind == Fp8::E5m2 {
            let [lo, hi]: [String; 2] = self.fresh_n();
            emit!(self, "prmt.b32 {lo}, {register}, 0, 0x1404;");
            emit!(self, "prmt.b32 {hi}, {register}, 0, 0x3424;");
            vec![lo, hi]
        } else {
            let [lo_bytes, hi_bytes]: [String; 2] = self.fresh_n();
            emit!(self, "prmt.b32 {lo_bytes}, {register}, 0, 0x4140;");
            emit!(self, "prmt.b32 {hi_bytes}, {register}, 0, 0x4342;");
            vec![
                self.fp8_e4m3_bytes(&lo_bytes, scale_256),
                self.fp8_e4m3_bytes(&hi_bytes, scale_256),
            ]
        };
        self.remember(form, &[register], &halves);
        halves
    }

    /// The two FP8 bytes in the low 16 bits of `pair` as two halves, exactly: the lower byte in the
    /// lower half.
    pub(super) fn fp8_pair(&mut self, kind: Fp8, pair: &str) -> String {
        if kind == Fp8::E5m2 {
            // The byte is the upper byte of the half it equals: byte 0 goes to byte 1 and byte 1
            // to byte 3, with zeros below them. One permutation of bytes (`PRMT`; three shifts
            // and an `or` until 2026-11-18: the same halves for all 65,536 pairs of bytes on a
            // V100).
            let halves = self.fresh();
            emit!(self, "prmt.b32 {halves}, {pair}, 0, 0x1404;");
            return halves;
        }
        let bytes = self.fresh();
        emit!(self, "prmt.b32 {bytes}, {pair}, 0, 0x4140;");
        self.fp8_e4m3_bytes(&bytes, true)
    }

    /// Two E4M3 bytes unpacked at bits `[7:0]` and `[23:16]` of `bytes` (zero upper bytes) decoded
    /// into packed `f16x2` halves, multiplied by 256 when `scale_256` is true.
    fn fp8_e4m3_bytes(&mut self, bytes: &str, scale_256: bool) -> String {
        let [sh7, sh8, placed, smask, nan, nmask, scale, halves]: [String; 8] = self.fresh_n();
        // As the bits of a half, the sign in bit 15 and the seven other bits in bits 13..7 are the
        // value divided by 256, subnormal or not (the exponents of the formats differ by 8).
        emit!(self, "shl.b32 {sh7}, {bytes}, 7;");
        emit!(self, "shl.b32 {sh8}, {bytes}, 8;");
        emit!(self, "and.b32 {placed}, {sh7}, 0x3f803f80;");
        emit!(self, "mov.b32 {smask}, 0x80008000;");
        emit!(self, "lop3.b32 {placed}, {placed}, {sh8}, {smask}, 0xF8;");
        // A byte whose seven magnitude bits are all ones (0x7f / 0xff) is NaN, not 480: adding
        // 0x0080 sets bit 14 iff bits 13..7 are all 1 (bit 14 is 0 in `placed`, so carry never
        // reaches sign bit 15).
        emit!(self, "add.u32 {nan}, {placed}, 0x00800080;");
        if scale_256 {
            // Multiplying `placed` by 256 (`0x5c005c00`) recovers the exact half in one `HMUL2` on
            // a V100 (which takes subnormal halves as they are; each half went through f32 and back
            // until 2026-11-18: the same halves for all 65,536 pairs of bytes on a V100, and this
            // `prmt`/`lop3` form matches all 65,536 pairs bit-for-bit, NaN payloads included:
            // `0x5f80 | 0x2000 = 0x7f80`, `0xdf80 | 0x2000 = 0xff80`).
            emit!(self, "shr.u32 {nan}, {nan}, 1;");
            emit!(self, "mov.b32 {scale}, 0x5c005c00;");
            emit!(self, "mul.rn.f16x2 {halves}, {placed}, {scale};");
            emit!(self, "mov.b32 {nmask}, 0x20002000;");
            emit!(self, "lop3.b32 {halves}, {halves}, {nan}, {nmask}, 0xF8;");
        } else {
            // Defer the 2^8 factor to a power-of-two scale (UE8M0 in `kind::mxf8f6f4`); exact
            // because Volta's `HMMA` takes f16 subnormal inputs without flushing (`fp8-bits` pins
            // all 14 E4M3 subnormal bytes on a V100). NaN: `0x3f80 | 0x4000 = 0x7f80`.
            emit!(self, "mov.b32 {nmask}, 0x40004000;");
            emit!(self, "lop3.b32 {halves}, {placed}, {nan}, {nmask}, 0xF8;");
        }
        halves
    }

    /// The FP8 byte nearest to the f32 `value`, ties to the even byte, in a fresh register: the
    /// largest finite one for what is beyond it (`.satfinite`), the NaN of the format for a NaN.
    ///
    /// The value (divided by 256 for `e4m3`: see `fp8_e4m3_bytes`) is first cut to a half towards zero,
    /// with a bit that says whether something was cut off. The byte is that half without its low
    /// bits, rounded on them and on that bit: one rounding, of the value itself. (Rounding to a half
    /// first and then to the byte would round twice.)
    pub(super) fn fp8_byte(&mut self, kind: Fp8, value: &str) -> String {
        // The largest finite value, the low bits of the half that the byte does not have, and the
        // scale of the half.
        let (largest, low, scale) = match kind {
            Fp8::E4m3 => ("0f43E00000", 7, Some("0f3B800000")),
            Fp8::E5m2 => ("0f47600000", 8, None),
        };
        let [size, back, bits, sticky, odd, byte, sign, nan]: [String; 8] = self.fresh_n();
        let half = self.fresh_half();
        self.scratch_predicate = true;
        emit!(self, "abs.f32 {size}, {value};");
        // `min` returns the other operand for a NaN, which is dealt with at the end.
        emit!(self, "min.f32 {size}, {size}, {largest};");
        if let Some(scale) = scale {
            emit!(self, "mul.f32 {size}, {size}, {scale};");
        }
        emit!(self, "cvt.rz.f16.f32 {half}, {size};");
        emit!(self, "cvt.f32.f16 {back}, {half};");
        emit!(self, "setp.neu.f32 %vf_n, {back}, {size};");
        emit!(self, "selp.b32 {sticky}, 1, 0, %vf_n;");
        emit!(self, "cvt.u32.u16 {bits}, {half};");
        // To nearest, ties to even, on the low bits and the bit below them.
        emit!(self, "shr.u32 {odd}, {bits}, {low};");
        emit!(self, "and.b32 {odd}, {odd}, 1;");
        emit!(self, "shl.b32 {bits}, {bits}, 1;");
        emit!(self, "or.b32 {bits}, {bits}, {sticky};");
        emit!(self, "add.u32 {bits}, {bits}, {};", (1 << low) - 1);
        emit!(self, "add.u32 {bits}, {bits}, {odd};");
        emit!(self, "shr.u32 {byte}, {bits}, {};", low + 1);
        emit!(self, "mov.b32 {sign}, {value};");
        emit!(self, "shr.u32 {sign}, {sign}, 31;");
        emit!(self, "shl.b32 {sign}, {sign}, 7;");
        emit!(self, "or.b32 {byte}, {byte}, {sign};");
        emit!(self, "or.b32 {nan}, {sign}, 0x7f;");
        emit!(self, "setp.nan.f32 %vf_n, {value}, {value};");
        emit!(self, "selp.b32 {byte}, {nan}, {byte}, %vf_n;");
        byte
    }

    /// Two E2M1 values, the nibbles of the low byte of `a` (a `.b8` or wider register), as two
    /// halves in a fresh `.b32`: the lower nibble gives the lower half. Exact: every E2M1 value is
    /// a half.
    ///
    /// Every E2M1 magnitude `0..7` has low byte `0x00` and high byte in
    /// `{0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46}` (all with bit 7 clear), looked up by
    /// `prmt.b32` without `HMUL2`. Because generic `prmt` sign-extends bit 7 of the selected byte
    /// when selector bit 3 is set (`8..15`), `.relu` automatically clamps all negative nibbles
    /// (including `-0.0`) to `+0.0` (`0x0000`). Without `.relu`, the magnitudes are looked up on
    /// `bits & 0x77` and the sign bits are inserted via `prmt.b32` + `lop3.b32` (`tests/kernels/blackwell/ref/decode_e2m1{,_relu}.bin`).
    pub(super) fn e2m1_pair(&mut self, a: &str, relu: bool) -> String {
        let width = self.declarations.width(a).unwrap_or(16);
        let bits = if width >= 32 && crate::scan::register(a).is_some() {
            a.to_string()
        } else {
            let widened = self.fresh();
            let widen = if width <= 8 { "cvt.u32.u8" } else { "cvt.u32.u16" };
            emit!(self, "{widen} {widened}, {a};");
            widened
        };
        if relu {
            let [m, halves]: [String; 2] = self.fresh_n();
            emit!(self, "prmt.b32 {m}, 0x3E3C3800, 0x46444240, {bits};");
            emit!(self, "prmt.b32 {halves}, {m}, 0, 0x1404;");
            return halves;
        }
        let [mag, m, p, sign_even, signs, halves]: [String; 6] = self.fresh_n();
        emit!(self, "and.b32 {mag}, {bits}, 0x77;");
        emit!(self, "prmt.b32 {m}, 0x3E3C3800, 0x46444240, {mag};");
        emit!(self, "prmt.b32 {p}, {m}, 0, 0x1404;");
        emit!(self, "shl.b32 {sign_even}, {bits}, 4;");
        emit!(self, "prmt.b32 {signs}, {sign_even}, {bits}, 0xC080;");
        emit!(self, "lop3.b32 {halves}, {p}, {signs}, 0x80008000, 0xF8;");
        halves
    }

    /// The E2M1 value nearest to the f32 `value`, ties to even, as a nibble in a fresh register
    /// (`cvt.rn.satfinite`): what is beyond 6 in magnitude becomes 6 with its sign, a NaN becomes
    /// +6 (the manual: "NaN results are converted to positive MAX_NORM"), and what rounds to zero
    /// keeps its sign. (An RTX 5090 gives these nibbles at every boundary and one float on each
    /// side of it: `tests/kernels/blackwell/ref/encode_e2m1.bin`.) With `relu`, a negative value,
    /// minus zero and minus infinity included, gives +0, and a NaN of either sign still +6
    /// (`encode_e2m1_relu.bin`).
    pub(super) fn e2m1_nibble(&mut self, value: &str, relu: bool) -> String {
        // The magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4 and 6, numbered 0 to 7: the number of a
        // magnitude is how many of the seven midpoints it is past. A value at a midpoint goes to
        // the neighbour whose mantissa bit is 0, the one with the even number: it counts as past
        // the midpoint when the number below is odd.
        const MIDPOINTS: [(&str, &str); 7] = [
            ("0f3E800000", "gt"), // 0.25
            ("0f3F400000", "ge"), // 0.75
            ("0f3FA00000", "gt"), // 1.25
            ("0f3FE00000", "ge"), // 1.75
            ("0f40200000", "gt"), // 2.5
            ("0f40600000", "ge"), // 3.5
            ("0f40A00000", "gt"), // 5
        ];
        let [size, nibble, sign]: [String; 3] = self.fresh_n();
        self.scratch_predicate = true;
        // (`min` of a NaN and a number is the number: `size` is never a NaN.)
        emit!(self, "abs.f32 {size}, {value};");
        emit!(self, "min.f32 {size}, {size}, 0f40C00000;"); // 6
        emit!(self, "mov.b32 {nibble}, 0;");
        for (midpoint, past) in MIDPOINTS {
            let one = self.fresh();
            emit!(self, "setp.{past}.f32 %vf_n, {size}, {midpoint};");
            emit!(self, "selp.b32 {one}, 1, 0, %vf_n;");
            emit!(self, "add.u32 {nibble}, {nibble}, {one};");
        }
        emit!(self, "mov.b32 {sign}, {value};");
        emit!(self, "shr.u32 {sign}, {sign}, 28;");
        emit!(self, "and.b32 {sign}, {sign}, 0x8;");
        emit!(self, "setp.nan.f32 %vf_n, {value}, {value};");
        emit!(self, "selp.b32 {sign}, 0, {sign}, %vf_n;");
        emit!(self, "selp.b32 {nibble}, 7, {nibble}, %vf_n;");
        if relu {
            emit!(self, "setp.ne.u32 %vf_n, {sign}, 0;");
            emit!(self, "selp.b32 {nibble}, 0, {nibble}, %vf_n;");
        } else {
            emit!(self, "or.b32 {nibble}, {nibble}, {sign};");
        }
        nibble
    }

    /// The UE8M0 byte for the magnitude of the f32 `value`, rounded toward zero (`up` false) or
    /// toward +infinity, in a fresh register (`cvt.rz.satfinite`, `cvt.rp.satfinite`).
    ///
    /// A byte e stands for 2^(e - 127), from 2^-127 (byte 0: the format has no zero) to 2^127
    /// (byte 0xfe); 0xff is NaN. The manual does not say so; an RTX 5090 does
    /// (`tests/kernels/blackwell/ref/encode_ue8m0_*.bin`): the sign is dropped; zero
    /// and whatever is at or below 2^-127 give byte 0; an infinity, and what is above 2^127,
    /// give 0xfe; a NaN gives 0xff.
    pub(super) fn ue8m0_byte(&mut self, up: bool, value: &str) -> String {
        let [bits, exponent, fraction, byte]: [String; 4] = self.fresh_n();
        self.scratch_predicate = true;
        emit!(self, "abs.f32 {bits}, {value};");
        emit!(self, "shr.u32 {exponent}, {bits}, 23;");
        emit!(self, "and.b32 {fraction}, {bits}, 0x7fffff;");
        if up {
            // The exponent field of a float is the byte of the power of two at or below it; one
            // more when the float is above that power. A subnormal float (field 0) is its
            // fraction times 2^-149: above 2^-127, the power of byte 0, when its fraction is
            // above 2^22.
            let [least, more]: [String; 2] = self.fresh_n();
            emit!(self, "setp.eq.u32 %vf_n, {exponent}, 0;");
            emit!(self, "selp.b32 {least}, 0x400000, 0, %vf_n;");
            emit!(self, "setp.gt.u32 %vf_n, {fraction}, {least};");
            emit!(self, "selp.b32 {more}, 1, 0, %vf_n;");
            emit!(self, "add.u32 {byte}, {exponent}, {more};");
        } else {
            emit!(self, "mov.b32 {byte}, {exponent};");
        }
        // (An infinity has the field 0xff and no fraction: 0xfe, as what is above 2^127. A NaN
        // has it too, and is put back.)
        emit!(self, "min.u32 {byte}, {byte}, 0xfe;");
        emit!(self, "setp.nan.f32 %vf_n, {value}, {value};");
        emit!(self, "selp.b32 {byte}, 0xff, {byte}, %vf_n;");
        byte
    }

    /// Two UE8M0 bytes, the low 16 bits of `a`, as two bf16 values in a fresh `.b32`: the lower
    /// byte gives the lower half (`cvt.rn.bf16x2.ue8m0x2`). Exact: a byte e is 2^(e - 127), whose
    /// bf16 has the exponent field e and no mantissa, for e from 1 to 254. Byte 0 is 2^-127, the
    /// subnormal 0x0040; byte 0xff is a NaN, which an RTX 5090 gives as 0x7fff
    /// (`tests/kernels/blackwell/ref/decode_ue8m0.bin`, all 65,536 pairs).
    pub(super) fn ue8m0_pair(&mut self, a: &str) -> String {
        let [bits, lower, upper, halves]: [String; 4] = self.fresh_n();
        self.scratch_predicate = true;
        emit!(self, "cvt.u32.u16 {bits}, {a};");
        emit!(self, "and.b32 {lower}, {bits}, 0xff;");
        emit!(self, "and.b32 {upper}, {bits}, 0xff00;");
        emit!(self, "shl.b32 {halves}, {lower}, 7;");
        // What the two bytes that are not a power of two with that field need besides: bit 6 of
        // the mantissa for byte 0, the whole mantissa for byte 0xff.
        for (byte, least, greatest, to) in [(&lower, "0", "0xff", 0), (&upper, "0", "0xff00", 16)] {
            let [more, place]: [String; 2] = self.fresh_n();
            emit!(self, "setp.eq.u32 %vf_n, {byte}, {least};");
            emit!(self, "selp.b32 {more}, 0x40, 0, %vf_n;");
            emit!(self, "setp.eq.u32 %vf_n, {byte}, {greatest};");
            emit!(self, "selp.b32 {more}, 0x7f, {more}, %vf_n;");
            emit!(self, "shl.b32 {place}, {more}, {to};");
            emit!(self, "or.b32 {halves}, {halves}, {place};");
        }
        emit!(self, "shl.b32 {upper}, {upper}, 15;");
        emit!(self, "or.b32 {halves}, {halves}, {upper};");
        halves
    }
}
