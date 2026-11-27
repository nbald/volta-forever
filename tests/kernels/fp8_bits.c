// fp8_bits.c: bit-exact V100 checks for FP8 decodes and subnormal E4M3 m16n8k32 products.
//
// usage: fp8_bits [-d GPU] K_FP8_BITS_PTX K_MMA_SCALED_FORMS_PTX
//
// Checks:
//   1. All 256 x 256 (65,536) input byte pairs for cvt.rn.{f16x2,bf16x2}.{e4m3x2,e5m2x2},
//      pinning every bit including -0.0 and NaN payloads (e.g. e4m3 0x7f -> f16 0x7f80,
//      0xff -> f16 0xff80, bf16 NaN -> 0x7fff).
//   2. Single plain m16n8k32 FP8 products (e4m3xe4m3, e4m3xe5m2, e5m2xe4m3) and block-scaled
//      kind::mxf8f6f4 (s_mxf8_e4m3_1x) exercising all 14 E4M3 subnormal bytes (0x01..0x07 and
//      0x81..0x87): all 14 x 14 subnormal pairs across all 32 k-positions, dense 32-term sums,
//      subnormal x normal / E5M2 subnormal products, and non-unit / extreme / NaN UE8M0 scales.
//      All sums are constructed to be exact in f32 and checked bit-for-bit (uint32_t == uint32_t),
//      proving Volta's HMMA takes all 14 f16 subnormal inputs without FTZ so deferred 2^8 UE8M0
//      scale folding is bit-exact on hardware.
//
// Environment: DRVRUN_LIBCUDA (library to load, default libcuda.so.1).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;

static void* lib;
#define SYM(ret, nm, ...) static ret (*nm)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuCtxCreate_v2, CUcontext*, unsigned, CUdevice);
SYM(CUresult, cuCtxDestroy_v2, CUcontext);
SYM(CUresult, cuCtxSynchronize, void);
SYM(CUresult, cuModuleLoadDataEx, CUmodule*, const void*, unsigned, int*, void**);
SYM(CUresult, cuModuleGetFunction, CUfunction*, CUmodule, const char*);
SYM(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t);
SYM(CUresult, cuMemFree_v2, CUdeviceptr);
SYM(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void*, size_t);
SYM(CUresult, cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t);
SYM(CUresult, cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    unsigned, void*, void**, void**);
SYM(CUresult, cuGetErrorName, CUresult, const char**);

static void* sym(const char* n) {
    void* p = dlsym(lib, n);
    if (!p) { fprintf(stderr, "missing %s\n", n); exit(2); }
    return p;
}
static const char* ename(CUresult r) {
    const char* s = "?";
    cuGetErrorName(r, &s);
    return s ? s : "?";
}
#define CK(c) do { CUresult r_ = (c); if (r_) { fprintf(stderr, "FAIL %s -> %d %s\n", #c, r_, ename(r_)); exit(1); } } while (0)

static char* slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = malloc((size_t)n + 1);
    if (!b || (long)fread(b, 1, (size_t)n, f) != n) { fprintf(stderr, "read failed: %s\n", path); exit(2); }
    b[n] = 0;
    fclose(f);
    return b;
}

static uint32_t d2f_bits(double v) {
    float f = (float)v;
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}

static double fp8_val(int e5m2, uint8_t b) {
    int sign = b >> 7;
    int exp = e5m2 ? ((b >> 2) & 0x1f) : ((b >> 3) & 0x0f);
    int mant = e5m2 ? (b & 0x03) : (b & 0x07);
    if (e5m2) {
        if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
        double v = exp ? ldexp(1.0 + mant / 4.0, exp - 15) : ldexp(mant / 4.0, -14);
        return sign ? -v : v;
    }
    if (exp == 15 && mant == 7) return NAN;
    double v = exp ? ldexp(1.0 + mant / 8.0, exp - 7) : ldexp(mant / 8.0, -6);
    return sign ? -v : v;
}

static uint16_t ref_e4m3_f16(uint8_t b) {
    uint16_t sign = (uint16_t)(b & 0x80u) << 8;
    uint8_t mag = b & 0x7fu;
    if (mag == 0x7fu) return (uint16_t)(sign | 0x7f80u);
    if (mag == 0u) return sign;
    int exp = (mag >> 3) & 0xf;
    int mant = mag & 0x7;
    if (exp == 0) {
        int e = -9;
        while ((mant & 0x8) == 0) { mant <<= 1; e--; }
        int f16_exp = (e + 3) + 15;
        return (uint16_t)(sign | (f16_exp << 10) | ((mant & 0x7) << 7));
    }
    return (uint16_t)(sign | ((exp + 8) << 10) | (mant << 7));
}

static uint16_t ref_e5m2_f16(uint8_t b) {
    return (uint16_t)((uint16_t)b << 8);
}

static uint16_t ref_fp8_bf16(int e5m2, uint8_t b) {
    double v = fp8_val(e5m2, b);
    if (isnan(v)) return 0x7fffu;
    return (uint16_t)(d2f_bits(v) >> 16);
}

static int check_decode(CUmodule mod, const char* name, int is_bf16, int e5m2, CUdeviceptr d_in, CUdeviceptr d_out) {
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, name));
    static uint32_t h_in[65536], h_out[65536];
    for (uint32_t i = 0; i < 65536; i++) h_in[i] = i;
    CK(cuMemcpyHtoD_v2(d_in, h_in, sizeof(h_in)));
    void* args[] = { &d_in, &d_out };
    CK(cuLaunchKernel(fn, 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH_v2(h_out, d_out, sizeof(h_out)));

    int wrong = 0;
    for (uint32_t i = 0; i < 65536; i++) {
        uint8_t lo = (uint8_t)(i & 0xffu);
        uint8_t hi = (uint8_t)((i >> 8) & 0xffu);
        uint16_t elo = is_bf16 ? ref_fp8_bf16(e5m2, lo) : (e5m2 ? ref_e5m2_f16(lo) : ref_e4m3_f16(lo));
        uint16_t ehi = is_bf16 ? ref_fp8_bf16(e5m2, hi) : (e5m2 ? ref_e5m2_f16(hi) : ref_e4m3_f16(hi));
        uint32_t exp = (uint32_t)elo | ((uint32_t)ehi << 16);
        if (h_out[i] != exp) {
            if (wrong < 5)
                fprintf(stderr, "MISMATCH %s in=0x%04x got=0x%08x exp=0x%08x\n", name, i, h_out[i], exp);
            wrong++;
        }
    }
    printf("decode %-26s 65536 pairs, %d wrong\n", name, wrong);
    return wrong;
}

static void pack_mma_case(
    uint32_t* dst, int stride,
    const uint8_t A[16][32], const uint8_t B[32][8], const uint32_t C[16][8],
    uint32_t sa_word, uint32_t sb_word, int has_scales)
{
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        uint32_t* x = dst + lane * stride;
        for (int r = 0; r < 4; r++) {
            int row = (r == 0 || r == 2) ? g : g + 8;
            int k0 = 4 * t + (r >= 2 ? 16 : 0);
            x[r] = (uint32_t)A[row][k0]
                 | ((uint32_t)A[row][k0 + 1] << 8)
                 | ((uint32_t)A[row][k0 + 2] << 16)
                 | ((uint32_t)A[row][k0 + 3] << 24);
        }
        for (int r = 0; r < 2; r++) {
            int k0 = 4 * t + (r ? 16 : 0);
            x[4 + r] = (uint32_t)B[k0][g]
                     | ((uint32_t)B[k0 + 1][g] << 8)
                     | ((uint32_t)B[k0 + 2][g] << 16)
                     | ((uint32_t)B[k0 + 3][g] << 24);
        }
        x[6] = C[g][2 * t];
        x[7] = C[g][2 * t + 1];
        x[8] = C[g + 8][2 * t];
        x[9] = C[g + 8][2 * t + 1];
        if (has_scales) {
            x[10] = sa_word;
            x[11] = sb_word;
        }
    }
}

static void ref_mma_exact(
    int a_e5m2, int b_e5m2, int has_scales, uint8_t sa_byte, uint8_t sb_byte,
    const uint8_t A[16][32], const uint8_t B[32][8], const uint32_t C[16][8],
    uint32_t exp_out[32 * 4])
{
    double sa = (sa_byte == 0xff) ? NAN : ldexp(1.0, (int)sa_byte - 127);
    double sb = (sb_byte == 0xff) ? NAN : ldexp(1.0, (int)sb_byte - 127);
    double D[16][8];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float cf;
            memcpy(&cf, &C[i][j], 4);
            double sum = 0.0;
            for (int k = 0; k < 32; k++)
                sum += fp8_val(a_e5m2, A[i][k]) * fp8_val(b_e5m2, B[k][j]);
            if (has_scales) sum = (double)cf + sum * (sa * sb);
            else sum = (double)cf + sum;
            D[i][j] = sum;
        }
    }
    for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        double vals[4] = { D[g][2 * t], D[g][2 * t + 1], D[g + 8][2 * t], D[g + 8][2 * t + 1] };
        for (int r = 0; r < 4; r++) {
            if (isnan(vals[r])) exp_out[lane * 4 + r] = has_scales ? 0x7fffffffu : 0x7fc00000u;
            else exp_out[lane * 4 + r] = d2f_bits(vals[r]);
        }
    }
}

static int run_mma_batch(
    CUmodule mod, const char* name, int has_scales,
    const uint32_t* h_in, const uint32_t* h_exp, int n_cases,
    CUdeviceptr d_in, CUdeviceptr d_out, uint32_t* h_out_save)
{
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, name));
    int stride = has_scales ? 12 : 10;
    size_t in_bytes = (size_t)n_cases * 32 * (size_t)stride * 4;
    size_t out_bytes = (size_t)n_cases * 32 * 4 * 4;
    uint32_t* h_out = malloc(out_bytes);
    if (!h_out) exit(2);

    CK(cuMemcpyHtoD_v2(d_in, h_in, in_bytes));
    CUdeviceptr d_aux = 0;
    void* args_plain[] = { &d_in, &d_out };
    void* args_scaled[] = { &d_in, &d_out, &d_aux };
    CK(cuLaunchKernel(fn, (unsigned)n_cases, 1, 1, 32, 1, 1, 0, NULL,
                      has_scales ? args_scaled : args_plain, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH_v2(h_out, d_out, out_bytes));

    int wrong = 0;
    int total = n_cases * 128;
    for (int i = 0; i < total; i++) {
        uint32_t g = h_out[i], e = h_exp[i];
        int ok = (g == e);
        if (!ok && !has_scales && (e == 0x7fc00000u) && ((g & 0x7f800000u) == 0x7f800000u) && (g & 0x007fffffu))
            ok = 1;
        if (!ok) {
            if (wrong < 5)
                fprintf(stderr, "MISMATCH %s case=%d idx=%d got=0x%08x exp=0x%08x\n",
                        name, i / 128, i % 128, g, e);
            wrong++;
        }
    }
    if (h_out_save) memcpy(h_out_save, h_out, out_bytes);
    free(h_out);
    printf("mma    %-26s %d cases (%d results), %d wrong\n", name, n_cases, total, wrong);
    return wrong;
}

int main(int argc, char** argv) {
    int dev_id = 0, argi = 1;
    if (argc > 2 && strcmp(argv[1], "-d") == 0) { dev_id = atoi(argv[2]); argi = 3; }
    if (argc - argi != 2) {
        fprintf(stderr, "usage: %s [-d GPU] K_FP8_BITS_PTX K_MMA_SCALED_FORMS_PTX\n", argv[0]);
        return 2;
    }
    const char* libname = getenv("DRVRUN_LIBCUDA");
    if (!libname || !*libname) libname = "libcuda.so.1";
    lib = dlopen(libname, RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen(%s): %s\n", libname, dlerror()); return 2; }
    cuInit = sym("cuInit"); cuDeviceGet = sym("cuDeviceGet");
    cuCtxCreate_v2 = sym("cuCtxCreate_v2"); cuCtxDestroy_v2 = sym("cuCtxDestroy_v2");
    cuCtxSynchronize = sym("cuCtxSynchronize"); cuModuleLoadDataEx = sym("cuModuleLoadDataEx");
    cuModuleGetFunction = sym("cuModuleGetFunction"); cuMemAlloc_v2 = sym("cuMemAlloc_v2");
    cuMemFree_v2 = sym("cuMemFree_v2"); cuMemcpyHtoD_v2 = sym("cuMemcpyHtoD_v2");
    cuMemcpyDtoH_v2 = sym("cuMemcpyDtoH_v2"); cuLaunchKernel = sym("cuLaunchKernel");
    cuGetErrorName = sym("cuGetErrorName");

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, dev_id));
    CUcontext ctx; CK(cuCtxCreate_v2(&ctx, 0, dev));

    char* ptx_bits = slurp(argv[argi]);
    char* ptx_scaled = slurp(argv[argi + 1]);
    CUmodule mod_bits, mod_scaled;
    CK(cuModuleLoadDataEx(&mod_bits, ptx_bits, 0, NULL, NULL));
    CK(cuModuleLoadDataEx(&mod_scaled, ptx_scaled, 0, NULL, NULL));

    CUdeviceptr d_in, d_out;
    CK(cuMemAlloc_v2(&d_in, 1024 * 1024));
    CK(cuMemAlloc_v2(&d_out, 1024 * 1024));

    int fails = 0;
    fails += check_decode(mod_bits, "f8_bits_decode_f16_e4m3", 0, 0, d_in, d_out);
    fails += check_decode(mod_bits, "f8_bits_decode_f16_e5m2", 0, 1, d_in, d_out);
    fails += check_decode(mod_bits, "f8_bits_decode_bf16_e4m3", 1, 0, d_in, d_out);
    fails += check_decode(mod_bits, "f8_bits_decode_bf16_e5m2", 1, 1, d_in, d_out);

    static const uint8_t SUB14[14] = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87
    };
    static const uint8_t NORM_E4M3[8] = { 0x00, 0x80, 0x08, 0x38, 0xb8, 0x3c, 0x7e, 0xfe };
    static const uint8_t E5M2_SUB_NORM[8] = { 0x01, 0x02, 0x03, 0x81, 0x82, 0x83, 0x3c, 0xbc };

    enum { N_E4E4 = 40, N_SCALED_EXTRA = 6, N_MIXED = 16 };
    uint32_t* in_plain = calloc(N_E4E4 * 32 * 10, 4);
    uint32_t* in_scaled = calloc((N_E4E4 + N_SCALED_EXTRA) * 32 * 12, 4);
    uint32_t* exp_plain = calloc(N_E4E4 * 128, 4);
    uint32_t* exp_scaled = calloc((N_E4E4 + N_SCALED_EXTRA) * 128, 4);
    uint32_t* out_plain = calloc(N_E4E4 * 128, 4);
    uint32_t* out_scaled = calloc((N_E4E4 + N_SCALED_EXTRA) * 128, 4);

    for (int c = 0; c < N_E4E4; c++) {
        uint8_t A[16][32] = {{0}}, B[32][8] = {{0}};
        uint32_t C[16][8] = {{0}};
        if (c < 32) {
            for (int i = 0; i < 16; i++) A[i][c] = SUB14[(i + c) % 14];
            for (int j = 0; j < 8; j++)  B[c][j] = SUB14[(j * 3 + c * 5) % 14];
        } else if (c < 36) {
            int shift = c - 32;
            for (int i = 0; i < 16; i++)
                for (int k = 0; k < 32; k++)
                    A[i][k] = SUB14[(i * 3 + k + shift) % 14];
            for (int k = 0; k < 32; k++)
                for (int j = 0; j < 8; j++)
                    B[k][j] = SUB14[(k * 5 + j * 2 + shift) % 14];
            if (shift == 2) {
                for (int i = 0; i < 16; i++)
                    for (int j = 0; j < 8; j++)
                        C[i][j] = d2f_bits(ldexp((double)((i - 8) * (j + 1)), -12));
            }
        } else {
            int k = (c - 36) * 7 + 3;
            for (int i = 0; i < 16; i++) A[i][k] = SUB14[(i + c) % 14];
            for (int j = 0; j < 8; j++)  B[k][j] = NORM_E4M3[(j + c) % 8];
        }
        pack_mma_case(in_plain + c * 32 * 10, 10, A, B, C, 0, 0, 0);
        pack_mma_case(in_scaled + c * 32 * 12, 12, A, B, C, 0x7f7f7f7fu, 0x7f7f7f7fu, 1);
        ref_mma_exact(0, 0, 0, 127, 127, A, B, C, exp_plain + c * 128);
        ref_mma_exact(0, 0, 1, 127, 127, A, B, C, exp_scaled + c * 128);
    }

    static const uint8_t SA_EXTRA[N_SCALED_EXTRA] = { 123, 131, 0,   254, 0xff, 127 };
    static const uint8_t SB_EXTRA[N_SCALED_EXTRA] = { 131, 123, 254, 0,   127,  127 };
    for (int e = 0; e < N_SCALED_EXTRA; e++) {
        int c = N_E4E4 + e;
        uint8_t A[16][32] = {{0}}, B[32][8] = {{0}};
        uint32_t C[16][8] = {{0}};
        for (int i = 0; i < 16; i++)
            for (int k = 0; k < 32; k++)
                A[i][k] = SUB14[(i + k + e) % 14];
        for (int k = 0; k < 32; k++)
            for (int j = 0; j < 8; j++)
                B[k][j] = SUB14[(k * 3 + j + e) % 14];
        if (e == 5) { A[3][5] = 0x7f; A[11][19] = 0xff; }
        uint32_t sa_w = SA_EXTRA[e] * 0x01010101u;
        uint32_t sb_w = SB_EXTRA[e] * 0x01010101u;
        pack_mma_case(in_scaled + c * 32 * 12, 12, A, B, C, sa_w, sb_w, 1);
        ref_mma_exact(0, 0, 1, SA_EXTRA[e], SB_EXTRA[e], A, B, C, exp_scaled + c * 128);
    }

    fails += run_mma_batch(mod_bits, "f8_bits_mma_e4m3_e4m3", 0, in_plain, exp_plain, N_E4E4, d_in, d_out, out_plain);
    fails += run_mma_batch(mod_scaled, "s_mxf8_e4m3_1x", 1, in_scaled, exp_scaled, N_E4E4 + N_SCALED_EXTRA, d_in, d_out, out_scaled);

    int cross_diff = 0;
    for (int i = 0; i < N_E4E4 * 128; i++) {
        if (out_plain[i] != out_scaled[i]) {
            if (cross_diff < 5)
                fprintf(stderr, "CROSS-DIFF plain vs mxf8(1.0) idx=%d plain=0x%08x mxf8=0x%08x\n",
                        i, out_plain[i], out_scaled[i]);
            cross_diff++;
        }
    }
    printf("cross  plain vs s_mxf8_e4m3_1x(1.0) %d results, %d diff\n", N_E4E4 * 128, cross_diff);
    fails += cross_diff;

    uint32_t* in_mix_ab = calloc(N_MIXED * 32 * 10, 4);
    uint32_t* in_mix_ba = calloc(N_MIXED * 32 * 10, 4);
    uint32_t* exp_mix_ab = calloc(N_MIXED * 128, 4);
    uint32_t* exp_mix_ba = calloc(N_MIXED * 128, 4);
    for (int c = 0; c < N_MIXED; c++) {
        uint8_t A1[16][32] = {{0}}, B1[32][8] = {{0}};
        uint8_t A2[16][32] = {{0}}, B2[32][8] = {{0}};
        uint32_t C[16][8] = {{0}};
        int k = (c * 2) % 32;
        for (int i = 0; i < 16; i++) {
            A1[i][k] = SUB14[(i + c) % 14];
            A2[i][k] = E5M2_SUB_NORM[(i + c) % 8];
        }
        for (int j = 0; j < 8; j++) {
            B1[k][j] = E5M2_SUB_NORM[(j + c) % 8];
            B2[k][j] = SUB14[(j * 2 + c) % 14];
        }
        pack_mma_case(in_mix_ab + c * 32 * 10, 10, A1, B1, C, 0, 0, 0);
        ref_mma_exact(0, 1, 0, 127, 127, A1, B1, C, exp_mix_ab + c * 128);
        pack_mma_case(in_mix_ba + c * 32 * 10, 10, A2, B2, C, 0, 0, 0);
        ref_mma_exact(1, 0, 0, 127, 127, A2, B2, C, exp_mix_ba + c * 128);
    }
    fails += run_mma_batch(mod_bits, "f8_bits_mma_e4m3_e5m2", 0, in_mix_ab, exp_mix_ab, N_MIXED, d_in, d_out, NULL);
    fails += run_mma_batch(mod_bits, "f8_bits_mma_e5m2_e4m3", 0, in_mix_ba, exp_mix_ba, N_MIXED, d_in, d_out, NULL);

    free(in_plain); free(in_scaled); free(exp_plain); free(exp_scaled); free(out_plain); free(out_scaled);
    free(in_mix_ab); free(in_mix_ba); free(exp_mix_ab); free(exp_mix_ba);
    free(ptx_bits); free(ptx_scaled);
    CK(cuMemFree_v2(d_in)); CK(cuMemFree_v2(d_out));
    CK(cuCtxDestroy_v2(ctx));

    if (fails == 0) {
        printf("PASS fp8-bits: 262144 decode pairs + 14-subnormal plain/mxf8 products exact\n");
        return 0;
    }
    printf("FAIL fp8-bits: %d mismatches\n", fails);
    return 1;
}
