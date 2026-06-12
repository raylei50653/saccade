#include <cuda_runtime.h>
#include <stdint.h>

namespace saccade {

__global__ void rgb_hwc_to_nv12_kernel(
    const uint8_t* __restrict__ rgb,
    uint8_t* __restrict__ nv12,
    int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int rgb_idx = (y * w + x) * 3;
    float r = (float)rgb[rgb_idx + 0];
    float g = (float)rgb[rgb_idx + 1];
    float b = (float)rgb[rgb_idx + 2];

    float yf = 0.299f * r + 0.587f * g + 0.114f * b;
    float cb = -0.168736f * r - 0.331264f * g + 0.5f * b + 128.0f;
    float cr = 0.5f * r - 0.418688f * g - 0.081312f * b + 128.0f;

    nv12[y * w + x] = (uint8_t)fminf(fmaxf(yf + 0.5f, 0.0f), 255.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        int uv_offset = h * w + (y / 2) * w + (x / 2) * 2;
        nv12[uv_offset]     = (uint8_t)fminf(fmaxf(cb + 0.5f, 0.0f), 255.0f);
        nv12[uv_offset + 1] = (uint8_t)fminf(fmaxf(cr + 0.5f, 0.0f), 255.0f);
    }
}

void launch_rgb_hwc_to_nv12_gpu(
    const uint8_t* rgb_ptr,
    uint8_t* nv12_ptr,
    int w, int h,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);
    rgb_hwc_to_nv12_kernel<<<grid, block, 0, stream>>>(rgb_ptr, nv12_ptr, w, h);
}

} // namespace saccade
