#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

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

} // namespace saccade
