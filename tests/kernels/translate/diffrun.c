// diffrun: differential test for translated cubins.
// Loads a reference (sm_70) cubin and a translated cubin with the plain CUDA driver API,
// runs each kernel on the same inputs, and compares outputs bit-for-bit.
//
// usage: diffrun [-d GPU] [-r REPS] [-q] REFERENCE.cubin TRANSLATED.cubin [KERNEL_NAMES...]
//
// Only libcuda.so.1 is needed (dlopen). The translated cubin must load as native sm_70 code.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;
enum {
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
};

static void* lib;
#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuDeviceGetName, char*, int, CUdevice);
SYM(CUresult, cuCtxCreate_v2, CUcontext*, unsigned, CUdevice);
SYM(CUresult, cuCtxDestroy_v2, CUcontext);
SYM(CUresult, cuCtxSynchronize, void);
SYM(CUresult, cuModuleLoadDataEx, CUmodule*, const void*, unsigned, int*, void**);
SYM(CUresult, cuModuleGetFunction, CUfunction*, CUmodule, const char*);
SYM(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t);
SYM(CUresult, cuMemFree_v2, CUdeviceptr);
SYM(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void*, size_t);
SYM(CUresult, cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t);
SYM(CUresult, cuMemsetD8_v2, CUdeviceptr, unsigned char, size_t);
SYM(CUresult, cuLaunchKernel, CUfunction, unsigned,unsigned,unsigned, unsigned,unsigned,unsigned,
    unsigned, void*, void**, void**);
SYM(CUresult, cuGetErrorName, CUresult, const char**);

static void* sym(const char* name) {
    void* p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "missing symbol %s\n", name); exit(2); }
    return p;
}
static const char* ename(CUresult r) {
    const char* s = "?";
    cuGetErrorName(r, &s);
    return s ? s : "?";
}
#define CK(call) do { CUresult r_ = (call); if (r_) { \
    fprintf(stderr, "FAIL %s -> %d %s\n", #call, r_, ename(r_)); return -1; } } while (0)

// A kernel named in $DIFFRUN_SKIP (names separated by commas) is taken as absent: the ones a
// translator refused, whose bodies are not to run.
static CUresult get_function(CUfunction* f, CUmodule m, const char* name) {
    const char* skip = getenv("DIFFRUN_SKIP");
    size_t n = strlen(name);
    for (const char* p = skip; p && (p = strstr(p, name)); p += n)
        if ((p == skip || p[-1] == ',') && (p[n] == 0 || p[n] == ',')) return 500;
    return cuModuleGetFunction(f, m, name);
}

#define N (1 << 14)   // 16k elements
#define BLOCK 256
#define GRID (N / BLOCK)

// Global heap-allocated comparison buffers (avoid stack overflow)
static int32_t *g_oref, *g_otest;

static int quiet = 0;
static int total_pass = 0, total_fail = 0, total_skip = 0;

static CUdeviceptr dalloc(size_t bytes) {
    CUdeviceptr p;
    CUresult r = cuMemAlloc_v2(&p, bytes);
    if (r) { fprintf(stderr, "cuMemAlloc(%zu) -> %d\n", bytes, r); exit(1); }
    return p;
}

// Fill buffer with deterministic pseudo-random int32 values
static void fill_random_i32(int32_t* buf, int count, uint32_t seed) {
    uint32_t state = seed;
    for (int i = 0; i < count; i++) {
        state = state * 1103515245 + 12345;
        buf[i] = (int32_t)state;
    }
}

// Fill with deterministic pseudo-random uint32 values (non-zero for division)
static void fill_random_u32(uint32_t* buf, int count, uint32_t seed) {
    uint32_t state = seed;
    for (int i = 0; i < count; i++) {
        state = state * 1103515245 + 12345;
        buf[i] = state | 1;  // ensure non-zero
    }
}

// Fill with deterministic float values in a useful range
static void fill_random_f32(float* buf, int count, uint32_t seed) {
    uint32_t state = seed;
    for (int i = 0; i < count; i++) {
        state = state * 1103515245 + 12345;
        buf[i] = ((float)(int32_t)state) / (float)(1 << 20);  // range roughly [-2048, 2048]
    }
}

// Fill with adversarial int32 values at type bounds
static void fill_adversarial_i32(int32_t* buf, int count) {
    int32_t specials[] = {
        0, 1, -1, 2, -2, 0x7FFFFFFF, (int32_t)0x80000000, 0x7FFFFFFE, (int32_t)0x80000001,
        42, -42, 100, -100, 1000, -1000, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, (int32_t)0x8000,
        0xFFFF, 0x10000, 0x7FFFFF, (int32_t)0x800000, 0xFFFFFF, 0x1000000,
        0x55555555, (int32_t)0xAAAAAAAA, 0x12345678, (int32_t)0xDEADBEEF
    };
    int ns = sizeof(specials)/sizeof(specials[0]);
    for (int i = 0; i < count; i++) {
        if (i < ns) buf[i] = specials[i];
        else {
            uint32_t x = (uint32_t)i * 2654435761u;
            buf[i] = (int32_t)x;
        }
    }
}

// Compare two buffers and report differences
static int compare_i32(const char* kernel, const int32_t* ref, const int32_t* test, int count) {
    int bad = 0;
    for (int i = 0; i < count; i++) {
        if (ref[i] != test[i]) {
            if (bad < 5) {
                fprintf(stderr, "  %s DIFF at [%d]: ref=0x%08x test=0x%08x\n",
                        kernel, i, (unsigned)ref[i], (unsigned)test[i]);
            }
            bad++;
        }
    }
    return bad;
}

static int compare_u32(const char* kernel, const uint32_t* ref, const uint32_t* test, int count) {
    return compare_i32(kernel, (const int32_t*)ref, (const int32_t*)test, count);
}

static int compare_f32(const char* kernel, const float* ref, const float* test, int count) {
    return compare_i32(kernel, (const int32_t*)ref, (const int32_t*)test, count);
}

// Launch a kernel in both modules and compare results (int32 variant)
static int test_binary_i32(const char* kname, CUmodule rm, CUmodule tm,
                            const int32_t* ha, const int32_t* hb, int count,
                            CUdeviceptr da, CUdeviceptr db, CUdeviceptr dref, CUdeviceptr dtest) {
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP (not found: ref=%d test=%d)\n", kname, r1, r2);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(da, ha, count * 4));
    CK(cuMemcpyHtoD_v2(db, hb, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &da, &db, &n};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());

    void* argst[] = {&dtest, &da, &db, &n};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));

    int bad = compare_i32(kname, g_oref, g_otest, count);
    if (bad) {
        printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count);
        total_fail++;
    } else {
        printf("  %-30s PASS\n", kname);
        total_pass++;
    }
    return bad;
}

// Unary int32 kernel: out = f(x, n)
static int test_unary_i32(const char* kname, CUmodule rm, CUmodule tm,
                           const int32_t* hx, int count,
                           CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP (not found: ref=%d test=%d)\n", kname, r1, r2);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &n};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());

    void* argst[] = {&dtest, &dx, &n};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    // using g_oref, g_otest
    
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));

    int bad = compare_i32(kname, g_oref, g_otest, count);
    if (bad) {
        printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count);
        total_fail++;
    } else {
        printf("  %-30s PASS\n", kname);
        total_pass++;
    }
    return bad;
}

// Test k_clamp_s32: out = clamp(x, lo, hi)
static int test_clamp(CUmodule rm, CUmodule tm,
                      const int32_t* hx, int count,
                      CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_clamp_s32";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    int lo = -100, hi = 100;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &lo, &hi, &n};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest, &dx, &lo, &hi, &n};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));
    int bad = compare_i32(kname, g_oref, g_otest, count);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    return bad;
}

// Test k_div_const: out = x / 7u
static int test_div_const(CUmodule rm, CUmodule tm,
                          const uint32_t* hx, int count,
                          CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_div_const";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &n};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest, &dx, &n};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));
    int bad = compare_u32(kname, (uint32_t*)g_oref, (uint32_t*)g_otest, count);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    return bad;
}

// Test k_loop_minmax: out = loop_minmax(data, n, iters)
static int test_loop_minmax(CUmodule rm, CUmodule tm,
                            const int32_t* hx, int count,
                            CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_loop_minmax";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    int iters = 8;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &n, &iters};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest, &dx, &n, &iters};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));
    int bad = compare_i32(kname, g_oref, g_otest, count);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    return bad;
}

// Test k_loop_mulhi
static int test_loop_mulhi(CUmodule rm, CUmodule tm,
                           const uint32_t* hx, int count,
                           CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_loop_mulhi";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    int iters = 16;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &n, &iters};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest, &dx, &n, &iters};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));
    int bad = compare_u32(kname, (uint32_t*)g_oref, (uint32_t*)g_otest, count);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    return bad;
}

// Test shmem_reduce
static int test_shmem_reduce(CUmodule rm, CUmodule tm,
                             const int32_t* hx, int count,
                             CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_shmem_reduce";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    CUdeviceptr dref_out, dtest_out;
    dref_out = dalloc(grid * 4);
    dtest_out = dalloc(grid * 4);
    CK(cuMemsetD8_v2(dref_out, 0, grid * 4));
    CK(cuMemsetD8_v2(dtest_out, 0, grid * 4));

    int n = count;
    void* args[] = {&dref_out, &dx, &n};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest_out, &dx, &n};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    int32_t* oref = (int32_t*)malloc(grid * 4);
    int32_t* otest = (int32_t*)malloc(grid * 4);
    cuMemcpyDtoH_v2(g_oref, dref_out, grid * 4);
    cuMemcpyDtoH_v2(g_otest, dtest_out, grid * 4);
    int bad = compare_i32(kname, g_oref, g_otest, grid);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, (int)grid); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    free(oref); free(otest);
    cuMemFree_v2(dref_out); cuMemFree_v2(dtest_out);
    return bad;
}

// Test k_shared_scale
static int test_shared_scale(CUmodule rm, CUmodule tm,
                             const int32_t* hx, int count,
                             CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_shared_scale";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &n};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest, &dx, &n};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));
    int bad = compare_i32(kname, g_oref, g_otest, count);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    return bad;
}

// Test k_combined
static int test_combined(CUmodule rm, CUmodule tm,
                         const int32_t* hx, int count,
                         CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_combined";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &n};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest, &dx, &n};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));
    int bad = compare_i32(kname, g_oref, g_otest, count);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    return bad;
}

// Test k_call_loop with iters parameter
static int test_call_loop(CUmodule rm, CUmodule tm,
                          const int32_t* hx, int count,
                          CUdeviceptr dx, CUdeviceptr dref, CUdeviceptr dtest) {
    const char* kname = "k_call_loop";
    CUfunction rf, tf;
    CUresult r1 = get_function(&rf, rm, kname);
    CUresult r2 = get_function(&tf, tm, kname);
    if (r1 || r2) {
        if (!quiet) printf("  %-30s SKIP\n", kname);
        total_skip++;
        return 0;
    }

    CK(cuMemcpyHtoD_v2(dx, hx, count * 4));
    CK(cuMemsetD8_v2(dref, 0, count * 4));
    CK(cuMemsetD8_v2(dtest, 0, count * 4));

    int n = count;
    int iters = 4;
    unsigned grid = (count + BLOCK - 1) / BLOCK;
    void* args[] = {&dref, &dx, &n, &iters};
    CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    void* argst[] = {&dtest, &dx, &n, &iters};
    CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
    CK(cuCtxSynchronize());

    // using g_oref, g_otest
    CK(cuMemcpyDtoH_v2(g_oref, dref, count * 4));
    CK(cuMemcpyDtoH_v2(g_otest, dtest, count * 4));
    int bad = compare_i32(kname, g_oref, g_otest, count);
    if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
    else     { printf("  %-30s PASS\n", kname); total_pass++; }
    return bad;
}

static void* load_file(const char* path, long* size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* d = malloc(*size);
    if (fread(d, 1, *size, f) != (size_t)*size) { fprintf(stderr, "read error %s\n", path); exit(1); }
    fclose(f);
    return d;
}

// Run kernel `kname` of both modules on the same inputs and compare `bytes` of what each wrote:
// its first argument is the output, `rest` the `count` arguments after it.
static void test_same_output(const char* kname, CUmodule rm, CUmodule tm, size_t bytes, void** rest, int count) {
    CUfunction rf, tf;
    if (get_function(&rf, rm, kname) || get_function(&tf, tm, kname)) { printf("  %-30s SKIP\n", kname); total_skip++; return; }
    CUdeviceptr out[2] = {dalloc(bytes), dalloc(bytes)};
    unsigned char* host[2] = {malloc(bytes), malloc(bytes)};
    for (int which = 0; which < 2; which++) {
        void* args[16] = {&out[which]};
        for (int i = 0; i < count; i++) args[1 + i] = rest[i];
        cuMemsetD8_v2(out[which], 0, bytes);
        CUresult r = cuLaunchKernel(which ? tf : rf, GRID,1,1, BLOCK,1,1, 0, NULL, args, NULL);
        if (!r) r = cuCtxSynchronize();
        if (r) { printf("  %-30s FAIL (%s: error %d %s)\n", kname, which ? "translated" : "reference", r, ename(r)); total_fail++; return; }
        cuMemcpyDtoH_v2(host[which], out[which], bytes);
    }
    size_t bad = 0;
    for (size_t i = 0; i < bytes; i++) bad += host[0][i] != host[1][i];
    size_t written = 0;
    for (size_t i = 0; i < bytes; i++) written += host[0][i] != 0;
    if (bad) { printf("  %-30s FAIL (%zu of %zu bytes differ)\n", kname, bad, bytes); total_fail++; }
    else if (!written) { printf("  %-30s FAIL (the reference wrote nothing)\n", kname); total_fail++; }
    else if (getenv("DIFFRUN_TIME")) {
        // $DIFFRUN_TIME launches of each, timed: what the translation costs against native code.
        int launches = atoi(getenv("DIFFRUN_TIME"));
        double ms[2];
        for (int which = 0; which < 2; which++) {
            void* args[16] = {&out[which]};
            for (int i = 0; i < count; i++) args[1 + i] = rest[i];
            struct timespec t0, t1;
            cuCtxSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int l = 0; l < launches; l++) cuLaunchKernel(which ? tf : rf, GRID,1,1, BLOCK,1,1, 0, NULL, args, NULL);
            cuCtxSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ms[which] = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / launches;
        }
        printf("  %-30s PASS  native %.3f ms, translated %.3f ms a launch: %.2f times\n", kname, ms[0], ms[1], ms[1] / ms[0]);
        total_pass++;
    }
    else { printf("  %-30s PASS\n", kname); total_pass++; }
    cuMemFree_v2(out[0]); cuMemFree_v2(out[1]); free(host[0]); free(host[1]);
}

int main(int argc, char** argv) {
    int dev = 0;
    const char *ref_path = NULL, *test_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i+1 < argc) { dev = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-q")) { quiet = 1; }
        else if (!ref_path) { ref_path = argv[i]; }
        else if (!test_path) { test_path = argv[i]; }
    }
    if (!ref_path || !test_path) {
        fprintf(stderr, "usage: diffrun [-d GPU] [-q] REFERENCE.cubin TRANSLATED.cubin\n");
        return 2;
    }

    lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen(libcuda.so.1): %s\n", dlerror()); return 1; }
    #define L(n) *(void**)&n = sym(#n)
    L(cuInit); L(cuDeviceGet); L(cuDeviceGetName); L(cuCtxCreate_v2); L(cuCtxDestroy_v2);
    L(cuCtxSynchronize); L(cuModuleLoadDataEx); L(cuModuleGetFunction);
    L(cuMemAlloc_v2); L(cuMemFree_v2); L(cuMemcpyHtoD_v2); L(cuMemcpyDtoH_v2);
    L(cuMemsetD8_v2); L(cuLaunchKernel); L(cuGetErrorName);

    g_oref = (int32_t*)malloc(N * 8);  // enough for 64-bit elements too
    g_otest = (int32_t*)malloc(N * 8);

    CK(cuInit(0));
    CUdevice device;
    CK(cuDeviceGet(&device, dev));
    char name[256];
    cuDeviceGetName(name, sizeof(name), device);
    printf("GPU %d: %s\n", dev, name);

    CUcontext ctx;
    CK(cuCtxCreate_v2(&ctx, 0, device));

    // Load both cubins
    long rsize, tsize;
    void* rdata = load_file(ref_path, &rsize);
    void* tdata = load_file(test_path, &tsize);

    char info_log[4096] = {0}, err_log[4096] = {0};
    int opts[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                  CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* vals[] = {info_log, (void*)sizeof(info_log), err_log, (void*)sizeof(err_log)};

    CUmodule rm;
    CUresult r = cuModuleLoadDataEx(&rm, rdata, 4, opts, vals);
    if (r) {
        fprintf(stderr, "FAIL: load reference cubin %s -> %d %s\n  %s\n", ref_path, r, ename(r), err_log);
        return 1;
    }
    printf("Reference: %s (loaded)\n", ref_path);

    memset(info_log, 0, sizeof(info_log));
    memset(err_log, 0, sizeof(err_log));
    CUmodule tm;
    r = cuModuleLoadDataEx(&tm, tdata, 4, opts, vals);
    if (r) {
        fprintf(stderr, "FAIL: load translated cubin %s -> %d %s\n  %s\n", test_path, r, ename(r), err_log);
        return 1;
    }
    printf("Translated: %s (loaded)\n\n", test_path);

    // Allocate buffers
    CUdeviceptr da = dalloc(N*4), db = dalloc(N*4), dref = dalloc(N*4), dtest = dalloc(N*4);

    // Generate test data
    int32_t ha[N], hb[N];
    uint32_t hu[N];
    fill_adversarial_i32(ha, N);
    fill_random_i32(hb, N, 42);
    fill_random_u32(hu, N, 99);

    printf("--- k_translate tests ---\n");
    test_binary_i32("k_min_s32", rm, tm, ha, hb, N, da, db, dref, dtest);
    test_binary_i32("k_max_s32", rm, tm, ha, hb, N, da, db, dref, dtest);
    test_binary_i32("k_min_u32", rm, tm, (int32_t*)hu, hb, N, da, db, dref, dtest);
    test_binary_i32("k_max_u32", rm, tm, (int32_t*)hu, hb, N, da, db, dref, dtest);
    test_clamp(rm, tm, ha, N, da, dref, dtest);
    test_binary_i32("k_mulhi_s32", rm, tm, ha, hb, N, da, db, dref, dtest);
    test_binary_i32("k_mulhi_u32", rm, tm, (int32_t*)hu, hb, N, da, db, dref, dtest);
    test_div_const(rm, tm, hu, N, da, dref, dtest);

    test_binary_i32("k_branch_minmax", rm, tm, ha, hb, N, da, db, dref, dtest);
    test_loop_minmax(rm, tm, ha, N, da, dref, dtest);
    test_loop_mulhi(rm, tm, hu, N, da, dref, dtest);

    test_shared_scale(rm, tm, ha, N, da, dref, dtest);
    test_shmem_reduce(rm, tm, ha, N, da, dref, dtest);
    test_combined(rm, tm, ha, N, da, dref, dtest);

    // FP16 test: k_half_arith
    {
        const char* kname = "k_half_arith";
        CUfunction rf, tf;
        CUresult r1 = get_function(&rf, rm, kname);
        CUresult r2 = get_function(&tf, tm, kname);
        if (!r1 && !r2) {
            uint16_t hfa[N], hfb[N];
            for (int i = 0; i < N; i++) {
                // Small half values to avoid overflow
                float fa = (float)(ha[i] % 100) / 10.0f;
                float fb = (float)(hb[i] % 100) / 10.0f;
                uint32_t xa, xb;
                memcpy(&xa, &fa, 4); memcpy(&xb, &fb, 4);
                // Simple float->half: just truncate mantissa (not exact but good enough for small values)
                int ea = ((xa>>23)&0xff) - 127 + 15;
                int eb = ((xb>>23)&0xff) - 127 + 15;
                hfa[i] = (ea > 0 && ea < 31) ? ((xa>>16)&0x8000) | (ea<<10) | ((xa>>13)&0x3ff) : 0;
                hfb[i] = (eb > 0 && eb < 31) ? ((xb>>16)&0x8000) | (eb<<10) | ((xb>>13)&0x3ff) : 0;
            }
            CUdeviceptr dha = dalloc(N*2), dhb = dalloc(N*2);
            CK(cuMemcpyHtoD_v2(dha, hfa, N*2));
            CK(cuMemcpyHtoD_v2(dhb, hfb, N*2));
            CK(cuMemsetD8_v2(dref, 0, N*2));
            CK(cuMemsetD8_v2(dtest, 0, N*2));
            int n = N;
            unsigned grid = (N + BLOCK - 1) / BLOCK;
            void* args[] = {&dref, &dha, &dhb, &n};
            CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            void* argst[] = {&dtest, &dha, &dhb, &n};
            CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
            CK(cuCtxSynchronize());
            // using g_oref, g_otest (as uint16)
            cuMemcpyDtoH_v2(g_oref, dref, N*2);
            cuMemcpyDtoH_v2(g_otest, dtest, N*2);
            int bad = 0;
            for (int i = 0; i < N; i++) if (g_oref[i] != g_otest[i]) bad++;
            if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, N); total_fail++; }
            else     { printf("  %-30s PASS\n", kname); total_pass++; }
            cuMemFree_v2(dha); cuMemFree_v2(dhb);
        } else {
            printf("  %-30s SKIP\n", kname); total_skip++;
        }
    }

    // 64-bit test: k_int64_arith
    {
        const char* kname = "k_int64_arith";
        CUfunction rf, tf;
        CUresult r1 = get_function(&rf, rm, kname);
        CUresult r2 = get_function(&tf, tm, kname);
        if (!r1 && !r2) {
            int count = N / 2;  // 64-bit elements
            int64_t* h64a = (int64_t*)malloc(count * 8);
            int64_t* h64b = (int64_t*)malloc(count * 8);
            for (int i = 0; i < count; i++) {
                h64a[i] = ((int64_t)ha[i] << 16) | (int64_t)(unsigned)ha[(i+1)%N];
                h64b[i] = ((int64_t)hb[i] << 16) | (int64_t)(unsigned)hb[(i+1)%N];
            }
            CUdeviceptr d64a = dalloc(count*8), d64b = dalloc(count*8);
            CUdeviceptr d64ref = dalloc(count*8), d64test = dalloc(count*8);
            CK(cuMemcpyHtoD_v2(d64a, h64a, count*8));
            CK(cuMemcpyHtoD_v2(d64b, h64b, count*8));
            CK(cuMemsetD8_v2(d64ref, 0, count*8));
            CK(cuMemsetD8_v2(d64test, 0, count*8));
            int n = count;
            unsigned grid = (count + BLOCK - 1) / BLOCK;
            void* args[] = {&d64ref, &d64a, &d64b, &n};
            CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            void* argst[] = {&d64test, &d64a, &d64b, &n};
            CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
            CK(cuCtxSynchronize());
            int64_t* oref = (int64_t*)malloc(count*8);
            int64_t* otest = (int64_t*)malloc(count*8);
            cuMemcpyDtoH_v2(g_oref, d64ref, count*8);
            cuMemcpyDtoH_v2(g_otest, d64test, count*8);
            int bad = 0;
            for (int i = 0; i < count; i++) if (g_oref[i] != g_otest[i]) bad++;
            if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, count); total_fail++; }
            else     { printf("  %-30s PASS\n", kname); total_pass++; }
            free(h64a); free(h64b); free(oref); free(otest);
            cuMemFree_v2(d64a); cuMemFree_v2(d64b); cuMemFree_v2(d64ref); cuMemFree_v2(d64test);
        } else {
            printf("  %-30s SKIP\n", kname); total_skip++;
        }
    }

    // Uniform-address and uniform-branch tests
    {
        float hf[N]; fill_random_f32(hf, N, 77);
        CUdeviceptr df = dalloc(N*4);
        CK(cuMemcpyHtoD_v2(df, hf, N*4));

        const char* kname = "k_uniform_addr";
        CUfunction rf, tf;
        CUresult r1 = get_function(&rf, rm, kname);
        CUresult r2 = get_function(&tf, tm, kname);
        if (!r1 && !r2) {
            CK(cuMemsetD8_v2(dref, 0, N*4));
            CK(cuMemsetD8_v2(dtest, 0, N*4));
            int n = N, stride = BLOCK;
            unsigned grid = (N + BLOCK - 1) / BLOCK;
            void* args[] = {&dref, &df, &n, &stride};
            CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            void* argst[] = {&dtest, &df, &n, &stride};
            CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
            CK(cuCtxSynchronize());
            // using g_oref, g_otest (as float)
            cuMemcpyDtoH_v2(g_oref, dref, N*4);
            cuMemcpyDtoH_v2(g_otest, dtest, N*4);
            int bad = compare_f32(kname, (float*)g_oref, (float*)g_otest, N);
            if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, N); total_fail++; }
            else     { printf("  %-30s PASS\n", kname); total_pass++; }
        } else { printf("  %-30s SKIP\n", kname); total_skip++; }
        cuMemFree_v2(df);
    }

    // k_uniform_branch(out, x, n, threshold): the blocks above the threshold take one path, the others
    // the other one.
    {
        const char* kname = "k_uniform_branch";
        CUfunction rf, tf;
        if (!get_function(&rf, rm, kname) && !get_function(&tf, tm, kname)) {
            CK(cuMemcpyHtoD_v2(da, ha, N * 4));
            CK(cuMemsetD8_v2(dref, 0, N * 4));
            CK(cuMemsetD8_v2(dtest, 0, N * 4));
            int n = N, threshold = GRID / 2;
            void* args[] = {&dref, &da, &n, &threshold};
            CK(cuLaunchKernel(rf, GRID,1,1, BLOCK,1,1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            void* argst[] = {&dtest, &da, &n, &threshold};
            CK(cuLaunchKernel(tf, GRID,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
            CK(cuCtxSynchronize());
            CK(cuMemcpyDtoH_v2(g_oref, dref, N * 4));
            CK(cuMemcpyDtoH_v2(g_otest, dtest, N * 4));
            int bad = compare_i32(kname, g_oref, g_otest, N);
            if (bad) { printf("  %-30s FAIL (%d/%d differ)\n", kname, bad, N); total_fail++; }
            else     { printf("  %-30s PASS\n", kname); total_pass++; }
        } else { printf("  %-30s SKIP\n", kname); total_skip++; }
    }

    // Warp reduce minmax
    {
        const char* kname = "k_warp_reduce_minmax";
        CUfunction rf, tf;
        CUresult r1 = get_function(&rf, rm, kname);
        CUresult r2 = get_function(&tf, tm, kname);
        if (!r1 && !r2) {
            CK(cuMemcpyHtoD_v2(da, ha, N*4));
            int32_t init_min = 0x7FFFFFFF, init_max = (int32_t)0x80000000;
            CUdeviceptr dmin_r = dalloc(4), dmax_r = dalloc(4);
            CUdeviceptr dmin_t = dalloc(4), dmax_t = dalloc(4);
            CK(cuMemcpyHtoD_v2(dmin_r, &init_min, 4));
            CK(cuMemcpyHtoD_v2(dmax_r, &init_max, 4));
            CK(cuMemcpyHtoD_v2(dmin_t, &init_min, 4));
            CK(cuMemcpyHtoD_v2(dmax_t, &init_max, 4));
            int n = N;
            unsigned grid = (N + BLOCK - 1) / BLOCK;
            void* args[] = {&dmin_r, &dmax_r, &da, &n};
            CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            void* argst[] = {&dmin_t, &dmax_t, &da, &n};
            CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
            CK(cuCtxSynchronize());
            int32_t rmin, rmax, tmin, tmax;
            cuMemcpyDtoH_v2(&rmin, dmin_r, 4);
            cuMemcpyDtoH_v2(&rmax, dmax_r, 4);
            cuMemcpyDtoH_v2(&tmin, dmin_t, 4);
            cuMemcpyDtoH_v2(&tmax, dmax_t, 4);
            int bad = (rmin != tmin) + (rmax != tmax);
            if (bad) { printf("  %-30s FAIL (min: ref=%d test=%d, max: ref=%d test=%d)\n", kname, rmin,tmin,rmax,tmax); total_fail++; }
            else     { printf("  %-30s PASS\n", kname); total_pass++; }
            cuMemFree_v2(dmin_r); cuMemFree_v2(dmax_r); cuMemFree_v2(dmin_t); cuMemFree_v2(dmax_t);
        } else { printf("  %-30s SKIP\n", kname); total_skip++; }
    }

    // Atomic minmax
    {
        const char* kname = "k_atomic_minmax";
        CUfunction rf, tf;
        CUresult r1 = get_function(&rf, rm, kname);
        CUresult r2 = get_function(&tf, tm, kname);
        if (!r1 && !r2) {
            CK(cuMemcpyHtoD_v2(da, ha, N*4));
            int32_t init_min = 0x7FFFFFFF, init_max = (int32_t)0x80000000;
            CUdeviceptr dmin_r = dalloc(4), dmax_r = dalloc(4);
            CUdeviceptr dmin_t = dalloc(4), dmax_t = dalloc(4);
            CK(cuMemcpyHtoD_v2(dmin_r, &init_min, 4));
            CK(cuMemcpyHtoD_v2(dmax_r, &init_max, 4));
            CK(cuMemcpyHtoD_v2(dmin_t, &init_min, 4));
            CK(cuMemcpyHtoD_v2(dmax_t, &init_max, 4));
            int n = N;
            unsigned grid = (N + BLOCK - 1) / BLOCK;
            void* args[] = {&dmin_r, &dmax_r, &da, &n};
            CK(cuLaunchKernel(rf, grid,1,1, BLOCK,1,1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            void* argst[] = {&dmin_t, &dmax_t, &da, &n};
            CK(cuLaunchKernel(tf, grid,1,1, BLOCK,1,1, 0, NULL, argst, NULL));
            CK(cuCtxSynchronize());
            int32_t rmin, rmax, tmin, tmax;
            cuMemcpyDtoH_v2(&rmin, dmin_r, 4);
            cuMemcpyDtoH_v2(&rmax, dmax_r, 4);
            cuMemcpyDtoH_v2(&tmin, dmin_t, 4);
            cuMemcpyDtoH_v2(&tmax, dmax_t, 4);
            int bad = (rmin != tmin) + (rmax != tmax);
            if (bad) { printf("  %-30s FAIL (min: ref=%d test=%d, max: ref=%d test=%d)\n", kname, rmin,tmin,rmax,tmax); total_fail++; }
            else     { printf("  %-30s PASS\n", kname); total_pass++; }
            cuMemFree_v2(dmin_r); cuMemFree_v2(dmax_r); cuMemFree_v2(dmin_t); cuMemFree_v2(dmax_t);
        } else { printf("  %-30s SKIP\n", kname); total_skip++; }
    }

    // k_uniform.cu: memory at addresses with a uniform part.
    {
        enum { ROWS = 4, COLUMNS = N / ROWS, M = 2000 };
        int32_t scale = -7, table[M];
        int64_t bias = 0x123456789ll;
        static int64_t wide[N];
        static uint16_t small[N];
        static uint8_t bytes[N];
        for (int i = 0; i < M; i++) table[i] = hb[i] % 1000;
        for (int i = 0; i < N; i++) wide[i] = (int64_t)ha[i] * 0x10001 + hb[i], small[i] = (uint16_t)hb[i], bytes[i] = (uint8_t)(ha[i] >> 3);
        CUdeviceptr dscale = dalloc(4), dbias = dalloc(8), dtable = dalloc(sizeof table), dwide = dalloc(sizeof wide), dsmall = dalloc(sizeof small), dbytes = dalloc(sizeof bytes);
        CK(cuMemcpyHtoD_v2(da, ha, N * 4));
        CK(cuMemcpyHtoD_v2(dscale, &scale, 4)); CK(cuMemcpyHtoD_v2(dbias, &bias, 8)); CK(cuMemcpyHtoD_v2(dtable, table, sizeof table));
        CK(cuMemcpyHtoD_v2(dwide, wide, sizeof wide)); CK(cuMemcpyHtoD_v2(dsmall, small, sizeof small)); CK(cuMemcpyHtoD_v2(dbytes, bytes, sizeof bytes));
        int n = N, m = M, columns = COLUMNS, rows = ROWS;
        unsigned ucolumns = COLUMNS, urows = ROWS;
        test_same_output("k_uniform_scalars", rm, tm, N * 4, (void*[]){&da, &dscale, &dbias, &n}, 4);
        test_same_output("k_uniform_table", rm, tm, N * 4, (void*[]){&da, &dtable, &n, &m}, 4);
        test_same_output("k_uniform_rows", rm, tm, N * 8, (void*[]){&dwide, &columns, &rows}, 3);
        test_same_output("k_uniform_small", rm, tm, N * 2, (void*[]){&dsmall, &dbytes, &ucolumns, &urows}, 4);
        cuMemFree_v2(dscale); cuMemFree_v2(dbias); cuMemFree_v2(dtable); cuMemFree_v2(dwide); cuMemFree_v2(dsmall); cuMemFree_v2(dbytes);
    }

    // k_params.cu: parameters of 8 and 16 bits, at every place of a word.
    {
        uint8_t a = 0xc7, c = 0x81; int8_t b = -77, d = -128, g = -3;
        uint16_t e = 0xfedc, h = 0x8001; int16_t f = -12345, k = -32768;
        // $DIFFRUN_ROUNDS: how long the loops of these kernels run (for timing).
        int n = N, rounds = getenv("DIFFRUN_ROUNDS") ? atoi(getenv("DIFFRUN_ROUNDS")) : 5;
        CK(cuMemcpyHtoD_v2(da, ha, N * 4));
        test_same_output("k_small_params", rm, tm, N * 4, (void*[]){&da, &n, &a, &b, &c, &d, &e, &f, &g, &h, &k, &rounds}, 12);
        struct { int32_t dims[8]; int64_t strides[8]; } shape = {{7, 3, 11, 5, 13, 2, 17, 9}, {1, 7, 21, 231, 1155, 15015, 30030, 510510}};
        int eight = 8;
        test_same_output("k_const_table", rm, tm, N * 4, (void*[]){&da, &n, &rounds}, 3);
        test_same_output("k_param_table", rm, tm, N * 4, (void*[]){&da, &n, &shape, &eight}, 4);
    }

    // k_uniform_math.cu: arithmetic and conversions on uniform values.
    {
        int n = N, rounds = getenv("DIFFRUN_ROUNDS") ? atoi(getenv("DIFFRUN_ROUNDS")) : 40, few = 16;
        uint32_t seed = 0x9e3779b9u; int64_t step = 0x1234567 * 1000003ll;
        float scale = 0.5f; double bias = 0.25;
        CK(cuMemcpyHtoD_v2(da, ha, N * 4));
        test_same_output("k_uniform_int", rm, tm, N * 4, (void*[]){&da, &n, &rounds, &seed, &step}, 5);
        test_same_output("k_uniform_float", rm, tm, N * 4, (void*[]){&da, &n, &few, &scale, &bias}, 5);
    }

    // k_uimad.cu: sm_75 UIMAD with immediate addend (uimad__URURsI_URURI) and UIMAD.WIDE pair reads.
    {
        uint32_t a = 3u, b = 5u, stride = 8u, n = N;
        uint64_t d = 0x123456789abcdef0ull, e = 0x0fedcba987654321ull;
        int rounds = getenv("DIFFRUN_ROUNDS") ? atoi(getenv("DIFFRUN_ROUNDS")) : 40;
        CUdeviceptr dw = dalloc(N * 8);
        CK(cuMemcpyHtoD_v2(da, ha, N * 4));
        test_same_output("k_uimad_stride", rm, tm, N * 4, (void*[]){&da, &a, &b, &stride, &n}, 5);
        test_same_output("k_uimad_wide_hi", rm, tm, N * 4, (void*[]){&dw, &da, &a, &b, &d, &e, &rounds, &n}, 8);
        cuMemFree_v2(dw);
    }

    // k_switch.cu: jump tables (a table of code offsets in the constants, a branch through a register).
    {
        int n = N;
        CK(cuMemcpyHtoD_v2(da, ha, N * 4));
        for (int which = 0; which <= 8; which++)
            test_same_output("k_switch_uniform", rm, tm, N * 4, (void*[]){&da, &n, &which}, 3);
        test_same_output("k_switch_thread", rm, tm, N * 4, (void*[]){&da, &n}, 2);
        int which = 3, rounds = 9;
        test_same_output("k_switch_twice", rm, tm, N * 4, (void*[]){&da, &n, &which, &rounds}, 4);
        which = 6, rounds = 40;
        test_same_output("k_switch_twice", rm, tm, N * 4, (void*[]){&da, &n, &which, &rounds}, 4);
    }

    // k_memory.cu: atomic operations in shared memory at an index and at an address that is a
    // parameter, loads and stores through a pointer that is shared or global memory.
    {
        int n = N, perm[64];
        for (int i = 0; i < 64; i++) perm[i] = (i * 37 + 11) & 63;
        CUdeviceptr dother = dalloc(64 * 4), dperm = dalloc(sizeof perm);
        CK(cuMemcpyHtoD_v2(da, ha, N * 4));
        CK(cuMemcpyHtoD_v2(dperm, perm, sizeof perm));
        test_same_output("k_shared_atomic", rm, tm, N * 4, (void*[]){&da, &n}, 2);
        test_same_output("k_global_atomic", rm, tm, N * 4, (void*[]){&da, &n}, 2);
        test_same_output("k_global_atomic64", rm, tm, N * 8, (void*[]){&da, &n}, 2);
        for (int which = 0; which <= 1; which++) {
            test_same_output("k_generic", rm, tm, N * 4, (void*[]){&da, &n, &which}, 3);
            cuMemsetD8_v2(dother, 0, 64 * 4);
            test_same_output("k_generic_index", rm, tm, N * 4, (void*[]){&da, &n, &which, &dother, &dperm}, 5);
        }
        cuMemFree_v2(dother); cuMemFree_v2(dperm);
    }

    // k_convert.cu: parameters of 8 and 16 bits made floats and doubles.
    {
        static float xf[N];
        static double xd[N];
        for (int i = 0; i < N; i++) xf[i] = (float)(ha[i] % 1000) / 8.0f, xd[i] = (double)(hb[i] % 100000) / 64.0;
        CUdeviceptr dxf = dalloc(sizeof xf), dxd = dalloc(sizeof xd);
        CK(cuMemcpyHtoD_v2(dxf, xf, sizeof xf)); CK(cuMemcpyHtoD_v2(dxd, xd, sizeof xd));
        int n = N, rounds = getenv("DIFFRUN_ROUNDS") ? atoi(getenv("DIFFRUN_ROUNDS")) : 6;
        uint8_t a = 3, wide = 201; int8_t b = -7; uint16_t c = 40000; int16_t d = -1234;
        float scale = 2.5f, f = 0.375f; double bound = 100.0, m = 1.0009765625, k = -3.5;
        test_same_output("k_uniform_convert", rm, tm, N * 4, (void*[]){&dxf, &n, &a, &b, &c, &d, &scale, &bound, &rounds}, 9);
        test_same_output("k_uniform_double", rm, tm, N * 8, (void*[]){&dxd, &n, &wide, &c, &m, &k, &f, &rounds}, 8);
        cuMemFree_v2(dxf); cuMemFree_v2(dxd);
    }

    // k_guards.cu: uniform values changed under a condition that is the same in every thread
    // (instructions on uniform registers under a guard). The bounds are reached partway, so that
    // each guard holds in some rounds and not in others.
    {
        int n = N, rounds = getenv("DIFFRUN_ROUNDS") ? atoi(getenv("DIFFRUN_ROUNDS")) : 40, limit = 23;
        int64_t limit64 = 5000000000ll, step = 300000007ll;
        uint32_t seed = 0x9e3779b9u, mask = 0xedb88320u;
        CK(cuMemcpyHtoD_v2(da, ha, N * 4));
        test_same_output("k_guarded_count", rm, tm, N * 4, (void*[]){&da, &n, &rounds, &limit}, 4);
        test_same_output("k_guarded_count64", rm, tm, N * 8, (void*[]){&da, &n, &rounds, &limit64, &step}, 5);
        test_same_output("k_guarded_state", rm, tm, N * 4, (void*[]){&da, &n, &rounds, &seed, &mask}, 5);
    }

    // k_calls.cu: functions that are really called (the return address is a number the caller gives),
    // and k_simple_minmax.cu.
    test_unary_i32("k_call_clamp", rm, tm, ha, N, da, dref, dtest);
    test_binary_i32("k_call_mulhi", rm, tm, (int32_t*)hu, hb, N, da, db, dref, dtest);
    test_unary_i32("k_call_nested", rm, tm, ha, N, da, dref, dtest);
    test_call_loop(rm, tm, ha, N, da, dref, dtest);
    test_binary_i32("k_simple_min", rm, tm, ha, hb, N, da, db, dref, dtest);

    // k_call_rel_mov.cu: CALL.REL return-address resolution with a colliding data constant and
    // divergent branches across nested calls.
    {
        int n = N, rounds = getenv("DIFFRUN_ROUNDS") ? atoi(getenv("DIFFRUN_ROUNDS")) : 6;
        uint32_t a = 7u, b = 0x1357u;
        size_t chunk_bytes = (size_t)N * 72 * 4;
        uint32_t* hchunk = calloc(1, chunk_bytes);
        if (!hchunk) { fprintf(stderr, "oom\n"); return 2; }
        for (int i = 0; i < N; i++) {
            hchunk[i * 72] = hu[i];
            hchunk[i * 72 + 1] = (uint32_t)hb[i];
        }
        CUdeviceptr dchunk = dalloc(chunk_bytes);
        CK(cuMemcpyHtoD_v2(dchunk, hchunk, chunk_bytes));
        test_same_output("k_call_rel_collision", rm, tm, chunk_bytes, (void*[]){&dchunk, &a, &b, &n}, 4);
        cuMemFree_v2(dchunk);
        free(hchunk);
        CK(cuMemcpyHtoD_v2(da, hu, N * 4));
        test_same_output("k_call_rel_converge", rm, tm, N * 4, (void*[]){&da, &rounds, &n}, 3);
    }

    cuMemFree_v2(da); cuMemFree_v2(db); cuMemFree_v2(dref); cuMemFree_v2(dtest);
    cuCtxDestroy_v2(ctx);
    free(rdata); free(tdata);

    printf("\n=== SUMMARY: %d PASS, %d FAIL, %d SKIP ===\n", total_pass, total_fail, total_skip);
    return total_fail ? 1 : 0;
}
