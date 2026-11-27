#include <stdint.h>

extern "C" __global__ void k_uflo_u32(
    int32_t* out, const int32_t* __restrict__ table, int n, uint32_t mask)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lz = __clz((int)(gridDim.x + mask));
    const int32_t* row = table + (lz & 31);
    int32_t sum = 0;
    for (int i = 0; i <= (lz & 7); ++i) {
        sum += __ldcg(row + i);
    }
    if (tid < n) {
        out[tid] = sum + table[tid & 255];
    }
}

extern "C" __global__ void k_uflo_s32(
    int32_t* out, const int32_t* __restrict__ table, int n, int32_t seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t x = (int32_t)gridDim.x * 3 + seed;
    int lz, sh;
    asm("bfind.s32 %0, %1;" : "=r"(lz) : "r"(x));
    asm("bfind.shiftamt.s32 %0, %1;" : "=r"(sh) : "r"(x));
    const int32_t* p = table + ((lz & 15) + (sh & 15));
    int32_t acc = 0;
    for (int i = 0; i <= (lz & 7); ++i) {
        acc += __ldcg(p + i);
    }
    if (tid < n) {
        out[tid] = acc + table[tid & 255];
    }
}

extern "C" __global__ void k_uflo_u32_sh(
    int32_t* out, const int32_t* __restrict__ table, int n, uint32_t mask)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = (uint32_t)gridDim.x * 5u ^ mask;
    int sh;
    asm("bfind.shiftamt.u32 %0, %1;" : "=r"(sh) : "r"(x));
    const int32_t* p = table + (sh & 31);
    int32_t acc = 0;
    for (int i = 0; i <= (sh & 7); ++i) {
        acc += __ldcg(p + i);
    }
    if (tid < n) {
        out[tid] = acc + table[tid & 255];
    }
}
