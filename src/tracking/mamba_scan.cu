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

__device__ inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Reverse-time backward. One thread block per (b, d); N threads (one per state
// n, all within a single warp). Mirrors selective_scan_fwd_kernel.
//
// Forward recurrence (per b, d):
//   delta_t = softplus(delta_in[t,d])
//   a_{t,n} = exp(delta_t * A_n)
//   bu_{t,n} = delta_t * B[t,n] * u[t,d]
//   h_{t,n} = a_{t,n} h_{t-1,n} + bu_{t,n}
//   y[t,d]  = sum_n C[t,n] h_{t,n}  (+ D_d u[t,d])
//
// Reverse: dh_{t,n} = g[t,d] C[t,n] + a_{t+1,n} dh_{t+1,n}.
__global__ void selective_scan_bwd_kernel(
    const float* __restrict__ grad_y,
    const float* __restrict__ u,
    const float* __restrict__ delta_in,
    const float* __restrict__ A,
    const float* __restrict__ B_vals,
    const float* __restrict__ C_vals,
    const float* __restrict__ D,
    float* __restrict__ h_buf,
    float* __restrict__ du,
    float* __restrict__ ddelta,
    float* __restrict__ dA,
    float* __restrict__ dB,
    float* __restrict__ dC,
    float* __restrict__ dD,
    int B, int L, int D_dim, int N, int has_D, int a_per_channel
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    if (d >= D_dim || b >= B) return;

    int n = threadIdx.x;  // 0 .. N-1, all lanes within one warp
    unsigned mask = (1u << N) - 1u;

    float A_n = A[a_per_channel ? (d * N + n) : n];

    // ---- Phase 1: forward recompute, store hidden states to h_buf ----
    float h = 0.0f;
    for (int t = 0; t < L; t++) {
        float delta_t = softplus_f32(delta_in[(b * L + t) * D_dim + d]);
        float u_td = u[(b * L + t) * D_dim + d];
        float a = expf(delta_t * A_n);
        float bu = delta_t * B_vals[(b * L + t) * N + n] * u_td;
        h = a * h + bu;
        h_buf[((b * L + t) * D_dim + d) * N + n] = h;
    }

    // ---- Phase 2: reverse pass ----
    float dh_next = 0.0f;  // dh_{t+1,n}
    float a_next = 0.0f;   // a_{t+1,n} (0 contribution at t=L-1)
    float dA_acc = 0.0f;   // sum_t da * a * delta  (per d,n)
    float dD_acc = 0.0f;   // sum_t g * u  (n==0 only)

    for (int t = L - 1; t >= 0; t--) {
        int idx_td = (b * L + t) * D_dim + d;
        int idx_tn = (b * L + t) * N + n;

        float delta_in_td = delta_in[idx_td];
        float delta_t = softplus_f32(delta_in_td);
        float sig = sigmoid_f32(delta_in_td);  // softplus'(delta_in)
        float u_td = u[idx_td];
        float g = grad_y[idx_td];
        float C_tn = C_vals[idx_tn];
        float B_tn = B_vals[idx_tn];

        float a = expf(delta_t * A_n);

        // dh_t = g C_tn + a_{t+1} dh_{t+1}
        float dh = g * C_tn + a_next * dh_next;

        float h_t = h_buf[(idx_td) * N + n];
        float h_tm1 = (t > 0) ? h_buf[(((b * L + (t - 1)) * D_dim + d) * N) + n]
                              : 0.0f;

        float dbu = dh;
        float da = h_tm1 * dh;

        // dA (per d,n) accumulate: da * a * delta_t
        dA_acc += da * a * delta_t;

        // dC[t,n] += g * h_t   (sum over d -> atomic)
        atomicAdd(&dC[idx_tn], g * h_t);
        // dB[t,n] += dbu * delta_t * u_td   (sum over d -> atomic)
        atomicAdd(&dB[idx_tn], dbu * delta_t * u_td);

        // du[t,d] = sum_n dbu * delta_t * B_tn + g * D_d  (reduce over n)
        float term_u = dbu * delta_t * B_tn;
        // ddelta[t,d] = (sum_n da*a*A_n + dh*B_tn*u_td) * sig  (reduce over n)
        float term_dlt = da * a * A_n + dh * B_tn * u_td;
        #pragma unroll
        for (int off = N / 2; off > 0; off >>= 1) {
            term_u += __shfl_xor_sync(mask, term_u, off);
            term_dlt += __shfl_xor_sync(mask, term_dlt, off);
        }
        if (n == 0) {
            float du_val = term_u;
            if (has_D) du_val += g * D[d];
            du[idx_td] = du_val;
            ddelta[idx_td] = term_dlt * sig;
            if (has_D) dD_acc += g * u_td;
        }

        dh_next = dh;
        a_next = a;
    }

    // dA: sum over t already in register; sum over b (and d if shared) via atomic
    atomicAdd(&dA[a_per_channel ? (d * N + n) : n], dA_acc);
    if (has_D && n == 0) atomicAdd(&dD[d], dD_acc);
}

void selective_scan_bwd(
    const float* grad_y,
    const float* u,
    const float* delta,
    const float* A,
    const float* B_ssm,
    const float* C_ssm,
    const float* D,
    float* h_buf,
    float* du,
    float* ddelta,
    float* dA,
    float* dB_ssm,
    float* dC_ssm,
    float* dD,
    const SelectiveScanParams& params,
    void* stream
) {
    int B = params.B, L = params.L, D_dim = params.D, N = params.N;
    if (N > 32) {
        fprintf(stderr, "[mamba_scan] bwd requires N<=32 (warp reduce), got %d\n", N);
        return;
    }
    dim3 grid(B, D_dim);
    dim3 block(N);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    selective_scan_bwd_kernel<<<grid, block, 0, s>>>(
        grad_y, u, delta, A, B_ssm, C_ssm, D, h_buf,
        du, ddelta, dA, dB_ssm, dC_ssm, dD,
        B, L, D_dim, N, params.has_D ? 1 : 0, params.a_per_channel ? 1 : 0
    );
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
