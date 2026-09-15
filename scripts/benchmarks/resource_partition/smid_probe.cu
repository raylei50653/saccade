// SM-id probe: every block records the %smid it executed on. Launched on a
// caller-provided stream handle so the same kernel can be captured into a CUDA
// graph or run directly. Diagnostic only; never part of the production path.
#include <cuda_runtime.h>

__global__ void smid_kernel(unsigned *out, unsigned long long spin_ns) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    t = t0;
    // Keep every block resident long enough that all blocks of the grid cannot
    // fit on a subset of the partition's SMs.
    while (t - t0 < spin_ns) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    }
    if (threadIdx.x == 0) out[blockIdx.x] = smid;
}

extern "C" const char *smid_probe_launch(void *stream, unsigned *out, int blocks,
                                         unsigned long long spin_ns) {
    smid_kernel<<<blocks, 32, 0, static_cast<cudaStream_t>(stream)>>>(out, spin_ns);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? "" : cudaGetErrorString(e);
}
