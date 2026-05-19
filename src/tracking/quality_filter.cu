#include "tracking/quality_filter.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace saccade {

// ─── quality_scale_scores ─────────────────────────────────────────────────────

__global__ static void quality_scale_kernel(
    float*       scores,
    const float* boxes,
    int n,
    float inv_fw, float inv_fh, float inv_frame_area,
    float w_aspect, float w_center, float w_area)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x1 = boxes[i * 4 + 0];
    float y1 = boxes[i * 4 + 1];
    float x2 = boxes[i * 4 + 2];
    float y2 = boxes[i * 4 + 3];

    float bw = fmaxf(x2 - x1, 1e-6f);
    float bh = fmaxf(y2 - y1, 1e-6f);

    // aspect_q: Gaussian peak at 2.5, sigma=1.2
    float aspect = bh / bw;
    float da = (aspect - 2.5f) / 1.2f;
    float aspect_q = expf(-0.5f * da * da);

    // center_q: proximity to frame interior
    float cx = (x1 + x2) * 0.5f * inv_fw;
    float cy = (y1 + y2) * 0.5f * inv_fh;
    float edge = fminf(fminf(cx, 1.0f - cx), fminf(cy, 1.0f - cy));
    float center_q = fminf(fmaxf(edge * 4.0f, 0.0f), 1.0f);

    // area_q: Gaussian peak at 0.01, sigma=0.01
    float area_ratio = bw * bh * inv_frame_area;
    float darea = (area_ratio - 0.01f) / 0.01f;
    float area_q = expf(-0.5f * darea * darea);

    float quality = w_aspect * aspect_q + w_center * center_q + w_area * area_q;
    scores[i] *= quality;
}

void quality_scale_scores(
    float*       scores,
    const float* boxes,
    int          n,
    int          frame_w,
    int          frame_h,
    float        w_aspect,
    float        w_center,
    float        w_area,
    cudaStream_t stream)
{
    if (n <= 0) return;
    float inv_fw = 1.0f / fmaxf((float)frame_w, 1.0f);
    float inv_fh = 1.0f / fmaxf((float)frame_h, 1.0f);
    float inv_fa = inv_fw * inv_fh;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    quality_scale_kernel<<<blocks, threads, 0, stream>>>(
        scores, boxes, n, inv_fw, inv_fh, inv_fa, w_aspect, w_center, w_area);
}

// ─── narrow_bonus_scores ──────────────────────────────────────────────────────

__global__ static void narrow_bonus_kernel(
    float*       scores,
    const float* boxes,
    const int*   classes,
    int n,
    float bonus,
    int person_class,
    float narrow_aspect_thresh,
    float narrow_height_thresh_abs)  // pre-multiplied: narrow_height_thresh * frame_h
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (classes[i] != person_class) return;

    float bw = fmaxf(boxes[i*4+2] - boxes[i*4+0], 1e-6f);
    float bh = fmaxf(boxes[i*4+3] - boxes[i*4+1], 1e-6f);
    float aspect = bh / bw;

    if (aspect < narrow_aspect_thresh && bh < narrow_height_thresh_abs) {
        scores[i] += bonus;
    }
}

void narrow_bonus_scores(
    float*       scores,
    const float* boxes,
    const int*   classes,
    int          n,
    float        bonus,
    int          person_class,
    float        narrow_aspect_thresh,
    float        narrow_height_thresh,
    int          frame_h,
    cudaStream_t stream)
{
    if (n <= 0 || bonus == 0.0f) return;
    float height_abs = narrow_height_thresh * (float)frame_h;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    narrow_bonus_kernel<<<blocks, threads, 0, stream>>>(
        scores, boxes, classes, n,
        bonus, person_class, narrow_aspect_thresh, height_abs);
}

// ─── fp_hard_filter ───────────────────────────────────────────────────────────

// Phase 1: mark keep flags
__global__ static void fp_keep_mask_kernel(
    const float* boxes,
    const float* scores,
    int n,
    float min_score,
    float max_area,
    float max_suspicious_score,
    int*  keep_mask)  // 1=keep, 0=drop
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float score = scores[i];
    if (score < min_score) { keep_mask[i] = 0; return; }

    float bw = fmaxf(boxes[i*4+2] - boxes[i*4+0], 1e-6f);
    float bh = fmaxf(boxes[i*4+3] - boxes[i*4+1], 1e-6f);
    float area = bw * bh;
    if (area > max_area && score < max_suspicious_score) { keep_mask[i] = 0; return; }

    keep_mask[i] = 1;
}

// Phase 2: prefix-sum scan (exclusive) — single-warp sequential for small N
// For N <= 2048 (typical post-NMS count), one block is fine.
__global__ static void prefix_sum_kernel(const int* in, int* out, int* total, int n) {
    // single-thread sequential scan in shared memory
    extern __shared__ int s[];
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            s[i] = sum;
            sum += in[i];
        }
        *total = sum;
        for (int i = 0; i < n; ++i) out[i] = s[i];
    }
}

// Phase 3: scatter kept elements
__global__ static void fp_scatter_kernel(
    const float* boxes_in,  const float* scores_in,  const int* classes_in,
    float*       boxes_out, float*       scores_out, int*       classes_out,
    const int* keep_mask, const int* prefix, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !keep_mask[i]) return;
    int dst = prefix[i];
    boxes_out[dst*4+0] = boxes_in[i*4+0];
    boxes_out[dst*4+1] = boxes_in[i*4+1];
    boxes_out[dst*4+2] = boxes_in[i*4+2];
    boxes_out[dst*4+3] = boxes_in[i*4+3];
    scores_out[dst]    = scores_in[i];
    classes_out[dst]   = classes_in[i];
}

int fp_hard_filter(
    const float* boxes_in,
    const float* scores_in,
    const int*   classes_in,
    int          n,
    float        min_score,
    float        max_area,
    float        max_suspicious_score,
    float*       boxes_out,
    float*       scores_out,
    int*         classes_out,
    cudaStream_t stream)
{
    if (n <= 0) return 0;

    // Allocate temporary keep_mask, prefix, total on device
    int* d_keep   = nullptr;
    int* d_prefix = nullptr;
    int* d_total  = nullptr;
    cudaMallocAsync(&d_keep,   n * sizeof(int), stream);
    cudaMallocAsync(&d_prefix, n * sizeof(int), stream);
    cudaMallocAsync(&d_total,  sizeof(int),     stream);

    // Phase 1: mark keep
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    fp_keep_mask_kernel<<<blocks, threads, 0, stream>>>(
        boxes_in, scores_in, n, min_score, max_area, max_suspicious_score, d_keep);

    // Phase 2: prefix sum — single block, shared mem = n * sizeof(int)
    // Safe for N up to ~8KB / 4 = 2048, which covers all post-NMS outputs.
    prefix_sum_kernel<<<1, 1, (size_t)n * sizeof(int), stream>>>(
        d_keep, d_prefix, d_total, n);

    // Phase 3: scatter
    fp_scatter_kernel<<<blocks, threads, 0, stream>>>(
        boxes_in, scores_in, classes_in,
        boxes_out, scores_out, classes_out,
        d_keep, d_prefix, n);

    // Read total (sync on stream)
    int count = 0;
    cudaMemcpyAsync(&count, d_total, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_keep,   stream);
    cudaFreeAsync(d_prefix, stream);
    cudaFreeAsync(d_total,  stream);

    return count;
}

// ─── parse_yolo_output ────────────────────────────────────────────────────────

__global__ static void parse_yolo_output_kernel(
    const float* __restrict__ raw,
    float* __restrict__ boxes,
    float* __restrict__ scores,
    int*   __restrict__ classes,
    int N,
    float inv_scale,
    float x_off,
    float y_off)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x1 = raw[i * 6 + 0];
    float y1 = raw[i * 6 + 1];
    float x2 = raw[i * 6 + 2];
    float y2 = raw[i * 6 + 3];

    boxes[i * 4 + 0] = (x1 - x_off) * inv_scale;
    boxes[i * 4 + 1] = (y1 - y_off) * inv_scale;
    boxes[i * 4 + 2] = (x2 - x_off) * inv_scale;
    boxes[i * 4 + 3] = (y2 - y_off) * inv_scale;

    scores[i]  = raw[i * 6 + 4];
    classes[i] = static_cast<int>(raw[i * 6 + 5]);
}

void parse_yolo_output(
    const float* raw,
    float*       boxes_out,
    float*       scores_out,
    int*         classes_out,
    int          N,
    float        inv_scale,
    float        x_off,
    float        y_off,
    cudaStream_t stream)
{
    if (N <= 0) return;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    parse_yolo_output_kernel<<<blocks, threads, 0, stream>>>(
        raw, boxes_out, scores_out, classes_out,
        N, inv_scale, x_off, y_off);
}

// ─── gather_boxes_cuda ────────────────────────────────────────────────────────

__global__ static void gather_boxes_kernel(
    const float* __restrict__ src,
    const int*   __restrict__ indices,
    float*       __restrict__ dst,
    int K)
{
    int k = blockIdx.x;
    int f = threadIdx.x;  // 0..3
    if (k >= K || f >= 4) return;
    dst[k * 4 + f] = src[indices[k] * 4 + f];
}

void gather_boxes_cuda(
    const float* src,
    const int*   d_indices,
    float*       dst,
    int          K,
    cudaStream_t stream)
{
    if (K <= 0) return;
    gather_boxes_kernel<<<K, 4, 0, stream>>>(src, d_indices, dst, K);
}

// ─── scatter_embeddings_cuda ──────────────────────────────────────────────────

__global__ static void scatter_embeddings_kernel(
    float*       __restrict__ dst,
    const float* __restrict__ src,
    const int*   __restrict__ indices,
    int K, int D)
{
    int k = blockIdx.x;
    if (k >= K) return;
    int dst_base = indices[k] * D;
    int src_base = k * D;
    for (int f = threadIdx.x; f < D; f += blockDim.x)
        dst[dst_base + f] = src[src_base + f];
}

void scatter_embeddings_cuda(
    float*       dst,
    const float* src,
    const int*   d_indices,
    int          K,
    int          D,
    cudaStream_t stream)
{
    if (K <= 0 || D <= 0) return;
    int threads = std::min(D, 512);
    scatter_embeddings_kernel<<<K, threads, 0, stream>>>(dst, src, d_indices, K, D);
}

} // namespace saccade
