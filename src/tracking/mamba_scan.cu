#include "tracking/mamba_scan.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// Number of D-dim channels processed per thread block.
// With N=16 states, this gives K*N threads per block:
//   K=4  →  64 threads (2 warps)  was 16 (0.5 warp)
//   K=8  → 128 threads (4 warps)
//   K=16 → 256 threads (8 warps)
#ifndef MAMBA_CHANNELS_PER_BLOCK
#define MAMBA_CHANNELS_PER_BLOCK 1
#endif

__device__ inline float softplus_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

template <typename T>
__global__ void selective_scan_fwd_kernel(
    const T* __restrict__ u,
    const T* __restrict__ delta_in,
    const T* __restrict__ A,
    const T* __restrict__ B_vals,
    const T* __restrict__ C_vals,
    const T* __restrict__ D,
    T* __restrict__ y,
    int B, int L, int D_dim, int N, int has_D, int a_per_channel
) {
    constexpr int K = MAMBA_CHANNELS_PER_BLOCK;

    int b = blockIdx.x;
    int d_start = blockIdx.y * K;

    int tid = threadIdx.x;
    int d_local = tid / N;
    int n = tid % N;

    int d = d_start + d_local;
    if (d >= D_dim || b >= B) return;

    float A_n = static_cast<float>(A[a_per_channel ? (d * N + n) : n]);
    float h = 0.0f;

    // Each channel's N=16 threads are consecutive within a single warp.
    // Compute the mask for __shfl_xor_sync using warp-local lane.
    int lane = tid % 32;
    int channel_in_warp = lane / N;
    unsigned mask = ((1u << N) - 1u) << (channel_in_warp * N);

    for (int t = 0; t < L; t++) {
        float u_btd = static_cast<float>(u[((b * L + t) * D_dim) + d]);
        float delta_btd = softplus_f32(static_cast<float>(delta_in[((b * L + t) * D_dim) + d]));
        float B_btn = static_cast<float>(B_vals[((b * L + t) * N) + n]);
        float C_btn = static_cast<float>(C_vals[((b * L + t) * N) + n]);

        float deltaA = expf(delta_btd * A_n);
        float deltaB_u = delta_btd * B_btn * u_btd;

        h = deltaA * h + deltaB_u;
        float acc = C_btn * h;

        // Warp shuffle reduction across N threads (N is power-of-2, each
        // channel group is within one warp).
        #pragma unroll
        for (int offset = N / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(mask, acc, offset);
        }

        // Thread 0 of each channel group writes the reduced result.
        if (n == 0) {
            y[((b * L + t) * D_dim) + d] = static_cast<T>(acc);
        }
    }

    // D skip connection — thread 0 of each channel group
    if (has_D && n == 0) {
        float D_d = static_cast<float>(D[d]);
        for (int t = 0; t < L; t++) {
            float val = static_cast<float>(y[((b * L + t) * D_dim) + d]);
            val += D_d * static_cast<float>(u[((b * L + t) * D_dim) + d]);
            y[((b * L + t) * D_dim) + d] = static_cast<T>(val);
        }
    }
}

void selective_scan_fwd(
    const float* u,
    const float* delta,
    const float* A,
    const float* B_ssm,
    const float* C_ssm,
    const float* D,
    float* y,
    const SelectiveScanParams& params,
    void* stream
) {
    constexpr int K = MAMBA_CHANNELS_PER_BLOCK;
    int B = params.B, L = params.L, D_dim = params.D, N = params.N;

    if (N > 1024 || K * N > 1024) {
        fprintf(stderr, "[mamba_scan] K*N=%d exceeds block thread limit 1024\n", K * N);
        return;
    }

    dim3 grid(B, (D_dim + K - 1) / K);
    dim3 block(K * N);
    size_t smem = 0;  // warp-shuffle reduction, no shared memory

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    selective_scan_fwd_kernel<float><<<grid, block, smem, s>>>(
        u, delta, A, B_ssm, C_ssm, D, y, B, L, D_dim, N,
        params.has_D ? 1 : 0, params.a_per_channel ? 1 : 0
    );
}

void selective_scan_fwd_half(
    const void* u,
    const void* delta,
    const void* A,
    const void* B_ssm,
    const void* C_ssm,
    const void* D,
    void* y,
    const SelectiveScanParams& params,
    void* stream
) {
    constexpr int K = MAMBA_CHANNELS_PER_BLOCK;
    int B = params.B, L = params.L, D_dim = params.D, N = params.N;

    if (N > 1024 || K * N > 1024) {
        fprintf(stderr, "[mamba_scan] K*N=%d exceeds block thread limit 1024\n", K * N);
        return;
    }

    dim3 grid2(B, (D_dim + K - 1) / K);
    dim3 block2(K * N);
    size_t smem2 = 0;  // warp-shuffle reduction

    cudaStream_t s2 = static_cast<cudaStream_t>(stream);
    selective_scan_fwd_kernel<__half><<<grid2, block2, smem2, s2>>>(
        static_cast<const __half*>(u),
        static_cast<const __half*>(delta),
        static_cast<const __half*>(A),
        static_cast<const __half*>(B_ssm),
        static_cast<const __half*>(C_ssm),
        static_cast<const __half*>(D),
        static_cast<__half*>(y),
        B, L, D_dim, N, params.has_D ? 1 : 0, params.a_per_channel ? 1 : 0
    );
}
