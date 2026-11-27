// Bit-pinning kernels for FP8 decodes (all 65,536 pairs, f16x2 and bf16x2) and single m16n8k32
// FP8 products exercising all 14 E4M3 subnormal bytes on a V100 (driven by fp8_bits.c).
//
// Build (on the host, no GPU needed):
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_89 -O3 -ptx -o k_fp8_bits.135.sm89.ptx k_fp8_bits.cu
#define GI (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL(name) extern "C" __global__ void name(const unsigned* in, unsigned* out)

KERNEL(f8_bits_decode_f16_e4m3) {
    unsigned x = in[GI] & 0xffffu;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, tmp;\n\t}"
                 : "=r"(d) : "r"(x));
    out[GI] = d;
}

KERNEL(f8_bits_decode_f16_e5m2) {
    unsigned x = in[GI] & 0xffffu;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.f16x2.e5m2x2 %0, tmp;\n\t}"
                 : "=r"(d) : "r"(x));
    out[GI] = d;
}

KERNEL(f8_bits_decode_bf16_e4m3) {
    unsigned x = in[GI] & 0xffffu;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.bf16x2.e4m3x2 %0, tmp;\n\t}"
                 : "=r"(d) : "r"(x));
    out[GI] = d;
}

KERNEL(f8_bits_decode_bf16_e5m2) {
    unsigned x = in[GI] & 0xffffu;
    unsigned d;
    asm volatile("{\n\t.reg .b16 tmp;\n\tcvt.u16.u32 tmp, %1;\n\tcvt.rn.bf16x2.e5m2x2 %0, tmp;\n\t}"
                 : "=r"(d) : "r"(x));
    out[GI] = d;
}

KERNEL(f8_bits_mma_e4m3_e4m3) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
                   "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

KERNEL(f8_bits_mma_e4m3_e5m2) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
                   "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}

KERNEL(f8_bits_mma_e5m2_e4m3) {
    const unsigned* x = in + GI * 10;
    unsigned d[4];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]),
                   "r"(x[4]), "r"(x[5]),
                   "r"(x[6]), "r"(x[7]), "r"(x[8]), "r"(x[9]));
    for (int i = 0; i < 4; i++) out[GI * 4 + i] = d[i];
}
