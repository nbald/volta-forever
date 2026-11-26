// A direct call whose return address is the program counter of a LEPC immediately before it:
// the shape cuDNN 9.24's sm_75 kernels use. nvcc gives a direct call a MOV of the return
// address; tests/kernels/translate/make_lepc_cubin.py turns that MOV into the LEPC and the
// callee's RET.REL into RET.ABS on its register, so that the checked-in cubin has the shape
// without being NVIDIA's code. The kernel computes the same result either way.
#include <stdint.h>

__device__ __noinline__ int32_t dev_lepc_inc(int32_t x) {
    return x + 1;
}

extern "C" __global__ void k_lepc(int32_t* out, const int32_t* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = dev_lepc_inc(in[i]);
    }
}
