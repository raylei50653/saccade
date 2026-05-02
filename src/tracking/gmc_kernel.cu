#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cufft.h>
#include <cuComplex.h>

namespace saccade {

__global__ void grayscale_downscale_kernel(
    const float* src, uint8_t* dst, 
    int src_w, int src_h, int dst_w, int dst_h) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_w && y < dst_h) {
        float scale_x = (float)src_w / dst_w;
        float scale_y = (float)src_h / dst_h;

        // Simple nearest neighbor / area-like sampling for downscale
        int sx = (int)(x * scale_x);
        int sy = (int)(y * scale_y);
        
        // Ensure within bounds
        sx = min(sx, src_w - 1);
        sy = min(sy, src_h - 1);

        int src_idx = (sy * src_w + sx) * 3;
        float r = src[src_idx + 0];
        float g = src[src_idx + 1];
        float b = src[src_idx + 2];

        // Grayscale conversion: 0.299R + 0.587G + 0.114B
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        
        // Clamp and scale to 0-255
        dst[y * dst_w + x] = (uint8_t)(fminf(fmaxf(gray, 0.0f), 1.0f) * 255.0f);
    }
}

void launch_grayscale_downscale(
    const float* src, uint8_t* dst, 
    int src_w, int src_h, int dst_w, int dst_h, 
    cudaStream_t stream) 
{
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    
    grayscale_downscale_kernel<<<grid, block, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h);
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

__global__ void find_peak_kernel(const float* data, int w, int h, int* peak_idx, float* peak_val) {
    // Note: Simple global max reduction. For production, use block-wise reduction.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float max_v = -1e9f;
    int max_i = -1;
    for (int i = 0; i < w * h; ++i) {
        if (data[i] > max_v) {
            max_v = data[i];
            max_i = i;
        }
    }
    *peak_idx = max_i;
    *peak_val = max_v;
}

extern "C" void launch_phase_correlation(
    const uint8_t* prev_gray, const uint8_t* curr_gray,
    int w, int h, float* dx, float* dy,
    void* d_tmp_complex_a, void* d_tmp_complex_b, void* d_tmp_float,
    cufftHandle plan_r2c, cufftHandle plan_c2r, cudaStream_t stream)
{
    // 1. FFT
    cufftExecR2C(plan_r2c, (cufftReal*)prev_gray, (cufftComplex*)d_tmp_complex_a);
    cufftExecR2C(plan_r2c, (cufftReal*)curr_gray, (cufftComplex*)d_tmp_complex_b);

    // 2. Cross-power spectrum
    int n_complex = w * (h / 2 + 1);
    int threads = 256;
    int blocks = (n_complex + threads - 1) / threads;
    cross_power_spectrum_kernel<<<blocks, threads, 0, stream>>>((cuComplex*)d_tmp_complex_a, (cuComplex*)d_tmp_complex_b, n_complex);

    // 3. Inverse FFT
    cufftExecC2R(plan_c2r, (cufftComplex*)d_tmp_complex_a, (cufftReal*)d_tmp_float);

    // 4. Find peak (D2H for simplicity now, could be done with cub)
    int h_peak_idx = 0;
    float h_peak_val = 0.0f;
    int* d_peak_idx; float* d_peak_val;
    cudaMalloc(&d_peak_idx, sizeof(int));
    cudaMalloc(&d_peak_val, sizeof(float));
    
    find_peak_kernel<<<1, 1, 0, stream>>>((float*)d_tmp_float, w, h, d_peak_idx, d_peak_val);
    
    cudaMemcpyAsync(&h_peak_idx, d_peak_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int px = h_peak_idx % w;
    int py = h_peak_idx / w;

    // Phase correlation peak is in [0, w) and [0, h)
    // Shift to [-w/2, w/2]
    if (px > w / 2) px -= w;
    if (py > h / 2) py -= h;

    *dx = (float)px;
    *dy = (float)py;

    cudaFree(d_peak_idx);
    cudaFree(d_peak_val);
}

} // namespace saccade
