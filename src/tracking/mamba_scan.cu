#include "tracking/mamba_scan.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__device__ inline float softplus_f32(float x) {
    // Numerically stable softplus: log(1+exp(x))
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}

__global__ void selective_scan_fwd_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta_in,
    const float* __restrict__ A,
    const float* __restrict__ B_vals,
    const float* __restrict__ C_vals,
    const float* __restrict__ D,
    float* __restrict__ y,
    int B, int L, int D_dim, int N, int has_D
) {
    int b = blockIdx.x / D_dim;
    int d = blockIdx.x % D_dim;

    if (b >= B) return;

    int n = threadIdx.x;
    if (n >= N) return;

    float A_n = A[n];
    float h = 0.0f;

    extern __shared__ float y_shared[];

    for (int t = 0; t < L; t++) {
        float u_btd = u[((b * L + t) * D_dim) + d];
        float delta_btd = softplus_f32(delta_in[((b * L + t) * D_dim) + d]);
        float B_btn = B_vals[((b * L + t) * N) + n];
        float C_btn = C_vals[((b * L + t) * N) + n];

        float deltaA = expf(delta_btd * A_n);
        float deltaB_u = delta_btd * B_btn * u_btd;

        h = deltaA * h + deltaB_u;
        y_shared[n] = C_btn * h;

        __syncthreads();

        // Reduction across N threads (warp-level for N ≤ 32)
        if (N <= 32) {
            if (n == 0) {
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < N; k++) {
                    acc += y_shared[k];
                }
                y[((b * L + t) * D_dim) + d] = acc;
            }
        } else {
            // Block-level reduction for larger N
            for (int stride = N / 2; stride > 0; stride >>= 1) {
                if (n < stride) {
                    y_shared[n] += y_shared[n + stride];
                }
                __syncthreads();
            }
            if (n == 0) {
                y[((b * L + t) * D_dim) + d] = y_shared[0];
            }
        }
        __syncthreads();
    }

    // Skip connection D (one thread per block)
    if (has_D && n == 0) {
        float D_d = D[d];
        for (int t = 0; t < L; t++) {
            y[((b * L + t) * D_dim) + d] += D_d * u[((b * L + t) * D_dim) + d];
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
    int B = params.B, L = params.L, D_dim = params.D, N = params.N;

    // Ensure N ≤ 1024 (block thread limit)
    if (N > 1024) {
        fprintf(stderr, "[mamba_scan] N=%d exceeds block thread limit 1024\n", N);
        return;
    }

    dim3 grid(B * D_dim);
    dim3 block(N);
    size_t smem = N * sizeof(float);

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    selective_scan_fwd_kernel<<<grid, block, smem, s>>>(
        u, delta, A, B_ssm, C_ssm, D, y, B, L, D_dim, N, params.has_D ? 1 : 0
    );
}
