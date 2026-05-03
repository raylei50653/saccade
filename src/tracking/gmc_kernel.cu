#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>

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
        cuComplex conj_b = cuConjf(b[i]);
        cuComplex prod = cuCmulf(a[i], conj_b);
        float mag = cuCabsf(prod) + 1e-6f;
        a[i] = make_cuComplex(cuCrealf(prod) / mag, cuCimagf(prod) / mag);
    }
}

__global__ void find_peak_subpixel_kernel(const float* data, int w, int h, float* peak_x, float* peak_y, float* peak_val) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    float max_v = -1e9f;
    int max_i = -1;
    for (int i = 0; i < w * h; ++i) {
        if (data[i] > max_v) {
            max_v = data[i];
            max_i = i;
        }
    }
    
    if (max_i == -1) return;
    
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
            float v = data[ny * w + nx];
            sum_v += v;
            sum_x += v * (px + dx);
            sum_y += v * (py + dy);
        }
    }
    
    *peak_x = sum_x / (sum_v + 1e-6f);
    *peak_y = sum_y / (sum_v + 1e-6f);
    *peak_val = max_v;
}

extern "C" void launch_phase_correlation(
    const float* prev_gray, const float* curr_gray,
    int w, int h, float* dx, float* dy,
    void* d_tmp_complex_a, void* d_tmp_complex_b, void* d_tmp_float,
    cufftHandle plan_r2c, cufftHandle plan_c2r, cudaStream_t stream)
{
    // 1. FFT (Input is float Hanning-windowed gray)
    cufftExecR2C(plan_r2c, (cufftReal*)prev_gray, (cufftComplex*)d_tmp_complex_a);
    cufftExecR2C(plan_r2c, (cufftReal*)curr_gray, (cufftComplex*)d_tmp_complex_b);

    // 2. Cross-power spectrum
    int n_complex = w * (h / 2 + 1);
    int threads = 256;
    int blocks = (n_complex + threads - 1) / threads;
    cross_power_spectrum_kernel<<<blocks, threads, 0, stream>>>((cuComplex*)d_tmp_complex_a, (cuComplex*)d_tmp_complex_b, n_complex);

    // 3. Inverse FFT
    cufftExecC2R(plan_c2r, (cufftComplex*)d_tmp_complex_a, (cufftReal*)d_tmp_float);

    // 4. Find peak with sub-pixel accuracy
    float h_peak_x = 0.0f, h_peak_y = 0.0f, h_peak_val = 0.0f;
    float *d_peak_x, *d_peak_y, *d_peak_val;
    cudaMalloc(&d_peak_x, sizeof(float));
    cudaMalloc(&d_peak_y, sizeof(float));
    cudaMalloc(&d_peak_val, sizeof(float));
    
    find_peak_subpixel_kernel<<<1, 1, 0, stream>>>((float*)d_tmp_float, w, h, d_peak_x, d_peak_y, d_peak_val);
    
    cudaMemcpyAsync(&h_peak_x, d_peak_x, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_peak_y, d_peak_y, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float fpx = h_peak_x;
    float fpy = h_peak_y;

    // Shift to [-w/2, w/2]
    if (fpx > w / 2.0f) fpx -= w;
    if (fpy > h / 2.0f) fpy -= h;

    *dx = fpx;
    *dy = fpy;

    cudaFree(d_peak_x); cudaFree(d_peak_y); cudaFree(d_peak_val);
}

} // namespace saccade
