#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

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

} // namespace saccade
