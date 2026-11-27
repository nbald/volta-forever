// Kernels exercising CFG-based CALL.REL return-address resolution: a data constant matching a
// CALL.REL return address (must stay untouched while only the return-address MOV moves), nested
// CALL.REL calls, uniform operations shifting offsets before calls, and divergent branches with
// convergence barriers.
//   toolkits/cuda-12.9.2/bin/nvcc -ccbin <host> -arch=sm_70 -O3 -cubin -o k_call_rel_mov.sm70.cubin k_call_rel_mov.cu
//   toolkits/cuda-13.5.0/bin/nvcc -ccbin <host> -arch=sm_75 -O3 -cubin -o k_call_rel_mov.sm75.cubin k_call_rel_mov.cu
#include <stdint.h>

struct alignas(16) Chunk {
    uint32_t v[72]; // 288 == 0x120 bytes: collides with the CALL.REL return address
};

__device__ __noinline__ uint32_t dev_step(uint32_t x, uint32_t y) {
    return (x ^ y) * 0x9e3779b1u;
}

__device__ __noinline__ uint32_t dev_mix_acc(uint32_t x, uint32_t y) {
    uint32_t acc = x ^ y;
    if ((x & 1u) != 0u) {
        acc = acc * 0x9e3779b1u + 0x7f4a7c15u;
    } else {
        acc = (acc ^ 0x5bd1e995u) + (y << 3);
    }
    return acc ^ (acc >> 13);
}

__device__ __noinline__ uint32_t dev_nested_acc(uint32_t x, uint32_t y) {
    uint32_t a = dev_mix_acc(x, y);
    uint32_t b = dev_mix_acc(a ^ y, x + 7u);
    return a + (b << 5);
}

extern "C" __global__ void k_call_rel_collision(
    Chunk* out,
    const Chunk* in,
    uint32_t a,
    uint32_t b,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        uint32_t u = ((uint32_t)blockIdx.y * a + (b ^ 0x1234u)) * 3u + a;
        uint32_t r = dev_step(in[i].v[0] + u, in[i].v[1] ^ b);
        out[i].v[0] = r;
    }
}

extern "C" __global__ void k_call_rel_converge(
    uint32_t* out,
    const uint32_t* in,
    int rounds,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    uint32_t x = in[i];
    uint32_t acc = x ^ 0xa5a55a5au;
    for (int r = 0; r < rounds; ++r) {
        if ((acc & 3u) == 0u) {
            acc ^= dev_mix_acc(acc + (uint32_t)r, x);
        } else if ((acc & 3u) == 1u) {
            acc += dev_nested_acc(acc, x ^ (uint32_t)r);
        } else {
            acc = (acc >> 1) ^ 0xdeadbeefu;
        }
    }
    out[i] = acc;
}
