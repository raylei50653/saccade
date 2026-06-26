// FPN centre-pool + conv ReID CUDA kernels.
//
// Kernel-only translation unit — deliberately includes NO torch headers so it
// can be parsed by nvcc's front-end without hitting torch 2.x header
// incompatibilities under newer nvcc/gcc combos (the torch::Tensor orchestration
// lives in fpn_reid_binding.cpp, compiled by the host compiler directly).
#include <cuda_runtime.h>
#include <cmath>

namespace {

__global__ void centre_pool_kernel(
    const float* feat, int C, int H, int W,
    const float* boxes, int N,
    float* pooled, int img_size
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float cx = (boxes[n*4+0] + boxes[n*4+2]) * 0.5f;
    float cy = (boxes[n*4+1] + boxes[n*4+3]) * 0.5f;
    int cx_idx = static_cast<int>(cx * W / img_size);
    int cy_idx = static_cast<int>(cy * H / img_size);
    if (cx_idx < 0) cx_idx = 0; if (cx_idx >= W) cx_idx = W - 1;
    if (cy_idx < 0) cy_idx = 0; if (cy_idx >= H) cy_idx = H - 1;

    for (int c = 0; c < C; c++) {
        pooled[n * C + c] = feat[c * H * W + cy_idx * W + cx_idx];
    }
}

__global__ void conv1x1_kernel(
    const float* centre_feat, int C,
    const float* weight, int O,
    float* out, int N
) {
    int n = blockIdx.x;
    int o = threadIdx.x;
    if (n >= N || o >= O) return;

    float sum = 0.0f;
    const float* cf = centre_feat + n * C;
    const float* w = weight + o * C;
    for (int c = 0; c < C; c++) {
        sum += cf[c] * w[c];
    }
    out[n * O + o] = sum;
}

__global__ void l2_normalise_kernel(float* data, int D, int N, float eps) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float* row = data + n * D;
    float norm = 0.0f;
    for (int d = 0; d < D; d++) norm += row[d] * row[d];
    norm = sqrtf(norm + eps);
    for (int d = 0; d < D; d++) row[d] /= norm;
}

__global__ void bn1d_eval_kernel(
    float* data, int D, int N,
    const float* running_mean,
    const float* running_var,
    float eps
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float* row = data + n * D;
    for (int d = 0; d < D; d++) {
        row[d] = (row[d] - running_mean[d]) / sqrtf(running_var[d] + eps);
    }
}

__global__ void linear_kernel(
    const float* data, int D,
    const float* weight, int O,
    float* out, int N
) {
    int n = blockIdx.x;
    int o = threadIdx.x;
    if (n >= N || o >= O) return;

    float sum = 0.0f;
    const float* d = data + n * D;
    const float* w = weight + o * D;
    for (int i = 0; i < D; i++) sum += d[i] * w[i];
    out[n * O + o] = sum;
}

} // anonymous namespace

// ── C-linkage launchers (declared as extern "C" in fpn_reid_binding.cpp) ─────
// conv1x1 / linear launch with `out_dim` threads per block; the caller (binding)
// enforces out_dim <= 1024 via TORCH_CHECK before invoking.

extern "C" void fpn_centre_pool(
    const float* feat, int C, int H, int W,
    const float* boxes, int N,
    float* pooled, int img_size, cudaStream_t stream)
{
    if (N <= 0) return;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    centre_pool_kernel<<<blocks, threads, 0, stream>>>(
        feat, C, H, W, boxes, N, pooled, img_size);
}

extern "C" void fpn_conv1x1(
    const float* centre_feat, int C,
    const float* weight, int O,
    float* out, int N, cudaStream_t stream)
{
    if (N <= 0 || O <= 0) return;
    conv1x1_kernel<<<N, O, 0, stream>>>(centre_feat, C, weight, O, out, N);
}

extern "C" void fpn_l2_normalise(
    float* data, int D, int N, float eps, cudaStream_t stream)
{
    if (N <= 0) return;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    l2_normalise_kernel<<<blocks, threads, 0, stream>>>(data, D, N, eps);
}

extern "C" void fpn_bn1d(
    float* data, int D, int N,
    const float* running_mean, const float* running_var,
    float eps, cudaStream_t stream)
{
    if (N <= 0) return;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    bn1d_eval_kernel<<<blocks, threads, 0, stream>>>(
        data, D, N, running_mean, running_var, eps);
}

extern "C" void fpn_linear(
    const float* data, int D,
    const float* weight, int O,
    float* out, int N, cudaStream_t stream)
{
    if (N <= 0 || O <= 0) return;
    linear_kernel<<<N, O, 0, stream>>>(data, D, weight, O, out, N);
}
