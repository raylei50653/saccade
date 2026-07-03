#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <stdint.h>
#include <cmath>
#include <vector>

namespace saccade {

__global__ void normalize_chw_kernel(const uint8_t* src, float* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        int spatial_idx = y * w + x;
        int plane_size = w * h;
        
        // RGB Normalized CHW
        // 假設輸入是 RGB (NPP 轉換後的)
        dst[0 * plane_size + spatial_idx] = src[idx + 0] / 255.0f; // R
        dst[1 * plane_size + spatial_idx] = src[idx + 1] / 255.0f; // G
        dst[2 * plane_size + spatial_idx] = src[idx + 2] / 255.0f; // B
    }
}

void launch_normalize_chw(const uint8_t* src, float* dst, int w, int h, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    
    normalize_chw_kernel<<<grid, block, 0, stream>>>(src, dst, w, h);
}

// --- Cropper Kernel ---
//
// Antialiased bilinear (triangle) resampling. Matches PIL's
// `img.crop(box).resize(out, Image.BILINEAR)` filter convention so the native
// path and the PIL fallback stay pixel-aligned:
//   * pixel-center convention — source pixel j is sampled at j + 0.5 and the
//     output-pixel center maps to `x1 + (xx + 0.5) * scale` (the old kernel used
//     a corner convention with no half-pixel shift);
//   * triangle filter `1 - |t|` with support = max(1, scale), so downscaling
//     averages multiple source pixels instead of aliasing onto a single tap;
//   * separable weights, each axis normalized to sum 1 (2-D pass with
//     pre-normalized wx*wy is algebraically equal to PIL's H-then-V passes).
// Boxes stay sub-pixel accurate (no integer rounding) so the online pipeline
// keeps full detector precision; the only residual vs PIL is box rounding,
// which is sub-pixel and negligible for embedding cosine.
__global__ void batch_crop_resize_kernel(
    const float* src, float* dst,
    int src_w, int src_h,
    const float* boxes, int num_boxes,
    int crop_w, int crop_h)
{
    int box_idx = blockIdx.z;
    if (box_idx >= num_boxes) return;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= crop_w || y >= crop_h) return;

    const float* box = boxes + box_idx * 4;
    float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
    // Clamp the box to the image first, then derive the scale from the visible
    // region — matches PIL's `img.crop(clamped_box).resize(...)`: when a box
    // runs off the edge, PIL shrinks the crop to the visible pixels and resizes
    // that, instead of distorting the crop with edge-replicated pixels.
    if (x1 < 0.0f) x1 = 0.0f;
    if (y1 < 0.0f) y1 = 0.0f;
    if (x2 > (float)src_w) x2 = (float)src_w;
    if (y2 > (float)src_h) y2 = (float)src_h;
    float bw = fmaxf(x2 - x1, 1e-6f);
    float bh = fmaxf(y2 - y1, 1e-6f);

    float scale_x = bw / (float)crop_w;
    float scale_y = bh / (float)crop_h;
    float fscale_x = fmaxf(1.0f, scale_x);  // filter stretch when downscaling
    float fscale_y = fmaxf(1.0f, scale_y);
    float center_x = x1 + (x + 0.5f) * scale_x;
    float center_y = y1 + (y + 0.5f) * scale_y;

    // Support covers every source pixel whose triangle weight can be nonzero,
    // then truncated to the crop region [x1, x2] x [y1, y2] (and the image).
    // Truncating at the box edge mirrors PIL's crop-then-resize: the resize
    // never reads pixels outside the crop, so no background leaks in and the
    // renormalized weights match PIL exactly for integer boxes.
    int xmin = (int)floorf(center_x - fscale_x);
    int xmax = (int)ceilf(center_x + fscale_x);   // exclusive
    int ymin = (int)floorf(center_y - fscale_y);
    int ymax = (int)ceilf(center_y + fscale_y);   // exclusive
    int bx_lo = (int)floorf(x1);
    int by_lo = (int)floorf(y1);
    int bx_hi = (int)ceilf(x2);
    int by_hi = (int)ceilf(y2);
    if (bx_lo < 0) bx_lo = 0;
    if (by_lo < 0) by_lo = 0;
    if (bx_hi > src_w) bx_hi = src_w;
    if (by_hi > src_h) by_hi = src_h;
    if (xmin < bx_lo) xmin = bx_lo;
    if (ymin < by_lo) ymin = by_lo;
    if (xmax > bx_hi) xmax = bx_hi;
    if (ymax > by_hi) ymax = by_hi;

    // Normalization sums (separable). Two short passes avoid local-memory
    // arrays that would spill for large downscale ratios.
    float sum_wx = 0.0f, sum_wy = 0.0f;
    for (int j = xmin; j < xmax; ++j) {
        float t = ((float)j + 0.5f - center_x) / fscale_x;
        float w = 1.0f - fabsf(t);
        if (w > 0.0f) sum_wx += w;
    }
    for (int j = ymin; j < ymax; ++j) {
        float t = ((float)j + 0.5f - center_y) / fscale_y;
        float w = 1.0f - fabsf(t);
        if (w > 0.0f) sum_wy += w;
    }

    float* dst_ptr = dst + box_idx * (3 * crop_w * crop_h);
    int plane_size = crop_w * crop_h;
    int spatial_idx = y * crop_w + x;
    int src_plane = src_w * src_h;

    if (sum_wx <= 0.0f || sum_wy <= 0.0f) {
        // Degenerate box (zero-area after clamp): emit zeros.
        for (int c = 0; c < 3; ++c)
            dst_ptr[c * plane_size + spatial_idx] = 0.0f;
        return;
    }
    float inv_wx = 1.0f / sum_wx;
    float inv_wy = 1.0f / sum_wy;

    // src is CHW float (output of normalize_chw_kernel).
    for (int c = 0; c < 3; ++c) {
        const float* src_c = src + c * src_plane;
        float val = 0.0f;
        for (int jy = ymin; jy < ymax; ++jy) {
            float ty = ((float)jy + 0.5f - center_y) / fscale_y;
            float wy = 1.0f - fabsf(ty);
            if (wy <= 0.0f) continue;
            wy *= inv_wy;
            int row = jy * src_w;
            for (int jx = xmin; jx < xmax; ++jx) {
                float tx = ((float)jx + 0.5f - center_x) / fscale_x;
                float wx = 1.0f - fabsf(tx);
                if (wx <= 0.0f) continue;
                val += wy * wx * inv_wx * src_c[row + jx];
            }
        }
        dst_ptr[c * plane_size + spatial_idx] = val;
    }
}

void launch_batch_crop_resize(
    const float* src, float* dst, 
    int src_w, int src_h,
    const float* boxes, int num_boxes,
    int crop_w, int crop_h,
    cudaStream_t stream) 
{
    if (num_boxes <= 0) return;

    dim3 block(16, 16);
    dim3 grid((crop_w + block.x - 1) / block.x, (crop_h + block.y - 1) / block.y, num_boxes);
    
    batch_crop_resize_kernel<<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, boxes, num_boxes, crop_w, crop_h);
}

// --- Feature Extractor Kernels ---

__global__ void reid_pre_normalize_kernel(
    float* data, int num, int channels, int h, int w,
    const float* mean, const float* std, bool is_siglip) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = h * w;
    int total = num * channels * spatial_size;
    
    if (idx < total) {
        if (is_siglip) {
            // SigLIP: x * 2 - 1
            data[idx] = data[idx] * 2.0f - 1.0f;
        } else {
            // Standard: (x - mean) / std
            int c = (idx / spatial_size) % channels;
            data[idx] = (data[idx] - mean[c]) / std[c];
        }
    }
}

__global__ void l2_normalize_kernel(float* data, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float* row = data + i * dim;
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) sum += row[j] * row[j];
        float norm = rsqrtf(fmaxf(sum, 1e-12f));
        for (int j = 0; j < dim; ++j) row[j] *= norm;
    }
}

__global__ void reid_preprocess_kernel(
    const float* src, float* dst, int num, int channels, int h, int w,
    const float* mean, const float* std, bool is_siglip)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = h * w;
    int total = num * channels * spatial_size;

    if (idx < total) {
        float v = src[idx];
        if (is_siglip) {
            dst[idx] = v * 2.0f - 1.0f;
        } else {
            int c = (idx / spatial_size) % channels;
            dst[idx] = (v - mean[c]) / std[c];
        }
    }
}

void launch_reid_pre_normalize(
    float* data, int num, int channels, int h, int w,
    const float* mean, const float* std, bool is_siglip,
    cudaStream_t stream) 
{
    int total = num * channels * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reid_pre_normalize_kernel<<<blocks, threads, 0, stream>>>(data, num, channels, h, w, mean, std, is_siglip);
}

void launch_reid_preprocess(
    const float* src, float* dst, int num, int channels, int h, int w,
    const float* mean, const float* std, bool is_siglip,
    cudaStream_t stream)
{
    int total = num * channels * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reid_preprocess_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, num, channels, h, w, mean, std, is_siglip);
}

void launch_l2_normalize(float* data, int n, int dim, cudaStream_t stream) {
    int threads = 64;
    int blocks = (n + threads - 1) / threads;
    l2_normalize_kernel<<<blocks, threads, 0, stream>>>(data, n, dim);
}

// Weighted sum of 3 part embeddings: out[n] = w0*p0[n] + w1*p1[n] + w2*p2[n]
// parts_ptr layout: [3, num_dets, feat_dim] (part-major)
__global__ void parts_fuse_kernel(
    const float* __restrict__ parts_ptr,
    float* __restrict__ out_ptr,
    int num_dets, int feat_dim,
    float w0, float w1, float w2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets * feat_dim) return;
    int stride = num_dets * feat_dim;
    out_ptr[idx] = w0 * parts_ptr[idx]
                 + w1 * parts_ptr[stride + idx]
                 + w2 * parts_ptr[2 * stride + idx];
}

void launch_parts_fuse(
    const float* parts_ptr, float* out_ptr,
    int num_dets, int feat_dim,
    float w0, float w1, float w2,
    cudaStream_t stream)
{
    int total = num_dets * feat_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    parts_fuse_kernel<<<blocks, threads, 0, stream>>>(parts_ptr, out_ptr, num_dets, feat_dim, w0, w1, w2);
}

// =============================================================================
// LaSt-ViT kernels (arXiv:2602.22394)
// Input: last_hidden_state [B, N, C] from SigLIP2 ViT encoder
// =============================================================================

// Apply per-element Gaussian low-pass weight to complex freq domain.
// lhs_freq: [B*N, C/2+1] cufftComplex; freqs_w: [C/2+1] float Gaussian weights.
__global__ void apply_gauss_kernel(
    cufftComplex* lhs_freq, const float* freqs_w,
    int BN, int half_C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BN * half_C) return;
    int k = tid % half_C;
    float w = freqs_w[k];
    lhs_freq[tid].x *= w;
    lhs_freq[tid].y *= w;
}

// Compute per-patch stability score: s = clamp(1 - ||x - x_filt||^2 / ||x||^2, 0, 1)
// x: [B, N, C], x_filt: [B, N, C], scores_out: [B, N]
__global__ void stability_score_kernel(
    const float* __restrict__ x,
    const float* __restrict__ x_filt,
    float* __restrict__ scores_out,
    int BN, int C)
{
    // One warp per patch row; use warp shuffle for reduction.
    int patch = blockIdx.x;
    if (patch >= BN) return;

    const float* xr = x + patch * C;
    const float* xf = x_filt + patch * C;

    float diff_sq = 0.0f, norm_sq = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        float d = xr[j] - xf[j];
        diff_sq += d * d;
        norm_sq += xr[j] * xr[j];
    }
    // Block-level reduction via shared memory
    __shared__ float s_diff[256], s_norm[256];
    s_diff[threadIdx.x] = diff_sq;
    s_norm[threadIdx.x] = norm_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_diff[threadIdx.x] += s_diff[threadIdx.x + stride];
            s_norm[threadIdx.x] += s_norm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float ratio = s_diff[0] / fmaxf(s_norm[0], 1e-8f);
        scores_out[patch] = fmaxf(0.0f, fminf(1.0f, 1.0f - ratio));
    }
}

// Compute per-batch Top-K threshold from scores_embed [B, N].
// For each batch b: sort N scores descending, threshold = scores[topk_idx].
// N=196 is small — sequential scan per batch is fine.
// Find K-th largest score via O(N*K) sequential selection (N=196, K≤98 → ≤19K ops).
// Uses a descending "ceiling" pass: each iteration finds the max below the previous max.
__global__ void topk_threshold_kernel(
    const float* __restrict__ scores, float* __restrict__ thresholds,
    int B, int N, int K)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* s = scores + b * N;
    float ceiling = 1.0f + 1e-7f;
    float kth = 0.0f;
    for (int k = 0; k < K; ++k) {
        float cur_max = -1e9f;
        for (int i = 0; i < N; ++i) {
            if (s[i] < ceiling && s[i] > cur_max) {
                cur_max = s[i];
            }
        }
        kth = cur_max;
        ceiling = cur_max;
    }
    thresholds[b] = kth;
}

// Aggregate Top-K patches into embedding: mean of patches where score >= threshold.
// x: [B, N, C], scores: [B, N], thresholds: [B], out: [B, C]
// Grid: (B, ceil(C/BLOCK_C)); block: (BLOCK_C,)
__global__ void topk_aggregate_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scores,
    const float* __restrict__ thresholds,
    float* __restrict__ out,
    int B, int N, int C)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) return;

    float thr = thresholds[b];
    float sum = 0.0f;
    int cnt = 0;
    for (int n = 0; n < N; ++n) {
        if (scores[b * N + n] >= thr) {
            sum += x[b * N * C + n * C + c];
            ++cnt;
        }
    }
    out[b * C + c] = (cnt > 0) ? (sum / cnt) : 0.0f;
}

// Mean of scores_gate per batch: stability_out[b] = mean(scores_gate[b, :])
__global__ void mean_stability_kernel(
    const float* __restrict__ scores_gate,
    float* __restrict__ stability_out,
    int B, int N)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* sg = scores_gate + b * N;
    float s = 0.0f;
    for (int n = 0; n < N; ++n) s += sg[n];
    stability_out[b] = s / N;
}

// Build Gaussian frequency weights [C/2+1] on host for a given sigma and C.
// Builds Gaussian low-pass weights for cuFFT R2C output (DC at index 0).
// Includes 1/C normalization so that IRFFT(Gauss * FFT(x)) == PyTorch's
// normalized irfft output (cuFFT C2R is unnormalized by default).
static void build_gauss_weights(float* h_weights, int C, float sigma) {
    int half_C = C / 2 + 1;
    float inv_C = 1.0f / (float)C;
    for (int k = 0; k < half_C; ++k) {
        float f = (float)k / C;
        h_weights[k] = expf(-f * f / (2.0f * sigma * sigma)) * inv_C;
    }
}

void launch_last_vit_refinement(
    const float* lhs,          // [B, N, C] on device
    float* embedding_out,      // [B, C] on device (un-normalized; caller does L2)
    float* stability_out,      // [B] on device
    int B, int N, int C,
    float sigma_embed,
    float sigma_gate,
    float top_k_ratio,
    cudaStream_t stream)
{
    const int half_C = C / 2 + 1;
    const int BN = B * N;
    const int K = (int)fmaxf(1.0f, top_k_ratio * N);

    // --- Allocate workspace ---
    cufftComplex* d_freq = nullptr;
    float* d_filt = nullptr;
    float* d_scores_embed = nullptr;
    float* d_scores_gate  = nullptr;
    float* d_thresholds   = nullptr;
    float* d_gauss        = nullptr;

    cudaMalloc(&d_freq,         (size_t)BN * half_C * sizeof(cufftComplex));
    cudaMalloc(&d_filt,         (size_t)BN * C * sizeof(float));
    cudaMalloc(&d_scores_embed, (size_t)BN * sizeof(float));
    cudaMalloc(&d_scores_gate,  (size_t)BN * sizeof(float));
    cudaMalloc(&d_thresholds,   (size_t)B  * sizeof(float));
    cudaMalloc(&d_gauss,        (size_t)half_C * sizeof(float));

    // --- cuFFT plan: batch 1D RFFT over last dim C, for BN signals ---
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &C,
                  nullptr, 1, C,       // input
                  nullptr, 1, half_C,  // output
                  CUFFT_R2C, BN);
    cufftSetStream(plan, stream);

    cufftHandle iplan;
    cufftPlanMany(&iplan, 1, &C,
                  nullptr, 1, half_C,  // input
                  nullptr, 1, C,       // output
                  CUFFT_C2R, BN);
    cufftSetStream(iplan, stream);

    // --- Helper lambda: FFT → apply gauss → IFFT → store filtered ---
    auto fft_filter = [&](float sigma) {
        // Forward FFT: lhs [BN, C] → d_freq [BN, half_C]
        cufftExecR2C(plan, const_cast<float*>(lhs), d_freq);

        // Upload Gaussian weights for this sigma
        std::vector<float> h_gauss_vec(half_C);
        build_gauss_weights(h_gauss_vec.data(), C, sigma);
        cudaMemcpyAsync(d_gauss, h_gauss_vec.data(), half_C * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Apply Gaussian weights
        int total_freq = BN * half_C;
        int threads = 256;
        apply_gauss_kernel<<<(total_freq + threads - 1) / threads, threads, 0, stream>>>(
            d_freq, d_gauss, BN, half_C);

        // Inverse FFT: d_freq → d_filt [BN, C] (unnormalized by C, but ratio-based
        // stability score and original-lhs aggregation are both scale-invariant)
        cufftExecC2R(iplan, d_freq, d_filt);
    };

    // --- Pass 1: sigma_embed branch (for Top-K selection) ---
    fft_filter(sigma_embed);
    {
        int threads = 256;
        stability_score_kernel<<<BN, threads, 0, stream>>>(
            lhs, d_filt, d_scores_embed, BN, C);
    }

    // --- Pass 2: sigma_gate branch (for stability output) ---
    fft_filter(sigma_gate);
    {
        int threads = 256;
        stability_score_kernel<<<BN, threads, 0, stream>>>(
            lhs, d_filt, d_scores_gate, BN, C);
    }

    // --- Top-K threshold per batch ---
    {
        int threads = 64;
        topk_threshold_kernel<<<(B + threads - 1) / threads, threads, 0, stream>>>(
            d_scores_embed, d_thresholds, B, N, K);
    }

    // --- Aggregate Top-K patches → embedding ---
    {
        const int BLOCK_C = 128;
        dim3 grid(B, (C + BLOCK_C - 1) / BLOCK_C);
        topk_aggregate_kernel<<<grid, BLOCK_C, 0, stream>>>(
            lhs, d_scores_embed, d_thresholds, embedding_out, B, N, C);
    }

    // --- Mean gate stability per batch ---
    {
        int threads = 64;
        mean_stability_kernel<<<(B + threads - 1) / threads, threads, 0, stream>>>(
            d_scores_gate, stability_out, B, N);
    }

    cufftDestroy(plan);
    cufftDestroy(iplan);
    cudaFree(d_freq);
    cudaFree(d_filt);
    cudaFree(d_scores_embed);
    cudaFree(d_scores_gate);
    cudaFree(d_thresholds);
    cudaFree(d_gauss);
}

} // namespace saccade
