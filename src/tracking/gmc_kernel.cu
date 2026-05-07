#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>
#include <iostream>

namespace saccade {

// Optimized: Handle CHW input (PyTorch default) directly to save Python overhead
__global__ void chw_to_grayscale_downscale_kernel(
    const float* src, float* dst,
    int src_w, int src_h, int dst_w, int dst_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_w && y < dst_h) {
        float scale_x = (float)src_w / dst_w;
        float scale_y = (float)src_h / dst_h;

        int sx = (int)(x * scale_x);
        int sy = (int)(y * scale_y);

        sx = min(sx, src_w - 1);
        sy = min(sy, src_h - 1);

        // Input is CHW: [c, y, x]
        size_t plane_size = (size_t)src_w * src_h;
        float r = src[0 * plane_size + sy * src_w + sx];
        float g = src[1 * plane_size + sy * src_w + sx];
        float b = src[2 * plane_size + sy * src_w + sx];

        float gray = 0.299f * r + 0.587f * g + 0.114f * b;

        // ADR 017: Apply Hanning Window to reduce FFT boundary artifacts
        float win_x = 0.5f * (1.0f - cosf(2.0f * M_PI * x / (dst_w - 1)));
        float win_y = 0.5f * (1.0f - cosf(2.0f * M_PI * y / (dst_h - 1)));

        dst[y * dst_w + x] = gray * win_x * win_y;
    }
}

void launch_grayscale_downscale(
    const float* src, float* dst,
    int src_w, int src_h, int dst_w, int dst_h,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    chw_to_grayscale_downscale_kernel<<<grid, block, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h);
}

// ── Phase Correlation Kernels ──────────────────────────────────────────────

__global__ void cross_power_spectrum_kernel(cuComplex* a, const cuComplex* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuComplex conj_a = cuConjf(a[i]);
        cuComplex prod = cuCmulf(conj_a, b[i]);
        float mag = cuCabsf(prod) + 1e-6f;
        a[i] = make_cuComplex(cuCrealf(prod) / mag, cuCimagf(prod) / mag);
    }
}

// Finds peak location with sub-pixel centroid refinement.
// Writes NaN to peak_x/peak_y when PCR (peak-to-RMS ratio) < kPCRThresh,
// which signals the shift estimate is unreliable (static camera, low texture, etc.).
// pcr_score always receives the actual peak/rms ratio so callers can observe quality.
__global__ void find_peak_subpixel_kernel(
    const float* data, int w, int h,
    float* peak_x, float* peak_y, float* peak_val, float* pcr_score)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int n = w * h;
    float max_v = -1e30f;
    int max_i = 0;
    float sum_sq = 0.0f;

    for (int i = 0; i < n; ++i) {
        float v = data[i];
        sum_sq += v * v;
        if (v > max_v) { max_v = v; max_i = i; }
    }

    *peak_val = max_v;

    const float kPCRThresh = 5.0f;
    float rms = sqrtf(sum_sq / n + 1e-12f);
    float ratio = max_v / (rms + 1e-12f);
    *pcr_score = ratio;  // always written so callers can read quality even on NaN

    // Reject if peak is not significantly above RMS — shift is noise
    if (ratio < kPCRThresh) {
        *peak_x = __int_as_float(0x7fc00000u);  // NaN
        *peak_y = __int_as_float(0x7fc00000u);
        return;
    }

    int py = max_i / w;
    int px = max_i % w;

    // Sub-pixel estimation using 3x3 centroid
    float sum_v = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int ny = (py + dy + h) % h;
            int nx = (px + dx + w) % w;
            float v = fmaxf(data[ny * w + nx], 0.0f);
            sum_v += v;
            sum_x += v * (px + dx);
            sum_y += v * (py + dy);
        }
    }

    *peak_x = sum_x / (sum_v + 1e-6f);
    *peak_y = sum_y / (sum_v + 1e-6f);
}

extern "C" void launch_phase_correlation(
    const float* prev_gray, const float* curr_gray,
    int w, int h, float* dx, float* dy,
    void* d_tmp_complex_a, void* d_tmp_complex_b, void* d_tmp_float,
    float* d_peak_x, float* d_peak_y, float* d_peak_val, float* d_pcr_score,
    cufftHandle plan_r2c, cufftHandle plan_c2r, cudaStream_t stream)
{
    // 1. FFT (Input is float Hanning-windowed gray)
    cufftExecR2C(plan_r2c, (cufftReal*)prev_gray, (cufftComplex*)d_tmp_complex_a);
    cufftExecR2C(plan_r2c, (cufftReal*)curr_gray, (cufftComplex*)d_tmp_complex_b);

    // 2. Cross-power spectrum: conj(FFT(prev)) * FFT(curr)
    int n_complex = h * (w / 2 + 1);
    int threads = 256;
    int blocks = (n_complex + threads - 1) / threads;
    cross_power_spectrum_kernel<<<blocks, threads, 0, stream>>>(
        (cuComplex*)d_tmp_complex_a, (cuComplex*)d_tmp_complex_b, n_complex);

    // 3. Inverse FFT
    cufftExecC2R(plan_c2r, (cufftComplex*)d_tmp_complex_a, (cufftReal*)d_tmp_float);

    // 4. Find peak with sub-pixel accuracy and PCR quality check
    find_peak_subpixel_kernel<<<1, 1, 0, stream>>>(
        (float*)d_tmp_float, w, h, d_peak_x, d_peak_y, d_peak_val, d_pcr_score);

    float h_peak_x = 0.0f, h_peak_y = 0.0f;
    cudaMemcpyAsync(&h_peak_x, d_peak_x, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_peak_y, d_peak_y, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // NaN from PCR check → unreliable shift, propagate to caller
    if (isnan(h_peak_x) || isnan(h_peak_y)) {
        *dx = h_peak_x;
        *dy = h_peak_y;
        return;
    }

    float fpx = h_peak_x;
    float fpy = h_peak_y;

    // Shift to [-w/2, w/2]
    if (fpx > w / 2.0f) fpx -= w;
    if (fpy > h / 2.0f) fpy -= h;

    *dx = fpx;
    *dy = fpy;
}

// ── Foreground Mask Kernel ─────────────────────────────────────────────────
// Zeroes out pixels inside any of the provided bounding boxes so that
// foreground objects do not bias the phase-correlation peak.
// boxes: flat [x1,y1,x2,y2, ...] in *original* frame coordinates.
// scale_x/scale_y: original_dim / downscaled_dim.
__global__ void zero_rects_kernel(
    float* gray, int w, int h,
    const float* boxes, int n_boxes,
    float scale_x, float scale_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= w || ty >= h) return;

    // Map downscaled pixel center back to original-frame coordinates.
    float fx = (tx + 0.5f) * scale_x;
    float fy = (ty + 0.5f) * scale_y;

    for (int i = 0; i < n_boxes; ++i) {
        float x1 = boxes[i * 4 + 0], y1 = boxes[i * 4 + 1];
        float x2 = boxes[i * 4 + 2], y2 = boxes[i * 4 + 3];
        if (fx >= x1 && fx <= x2 && fy >= y1 && fy <= y2) {
            gray[ty * w + tx] = 0.0f;
            return;
        }
    }
}

void launch_zero_fg_rects(
    float* gray, int w, int h,
    const float* d_boxes, int n_boxes,
    float orig_w, float orig_h,
    cudaStream_t stream)
{
    if (n_boxes <= 0) return;
    float scale_x = orig_w / w;
    float scale_y = orig_h / h;
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    zero_rects_kernel<<<grid, block, 0, stream>>>(gray, w, h, d_boxes, n_boxes, scale_x, scale_y);
}

} // namespace saccade
