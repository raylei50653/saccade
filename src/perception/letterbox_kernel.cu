#include <cuda_runtime.h>
#include <math.h>

namespace saccade {

// Fused letterbox: bilinear resize + pad in a single pass.
// src: CHW float [0,1], src_w × src_h
// dst: CHW float [0,1], dst_size × dst_size
// x_off, y_off: top-left corner of the resized content in dst
// w_new, h_new: size of the content region in dst
// pad_val: value to fill outside the content region
__global__ void letterbox_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int src_w, int src_h,
    int dst_size,
    int x_off, int y_off,
    int w_new, int h_new,
    float pad_val)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z;

    if (dx >= dst_size || dy >= dst_size) return;

    int lx = dx - x_off;
    int ly = dy - y_off;

    float val;
    if (lx >= 0 && lx < w_new && ly >= 0 && ly < h_new) {
        // Nearest-neighbor: matches torch.nn.functional.interpolate(mode='nearest') default.
        // PyTorch formula: src_idx = floor(out_idx * src_size / dst_size)
        int sx = min((int)(lx * ((float)src_w / w_new)), src_w - 1);
        int sy = min((int)(ly * ((float)src_h / h_new)), src_h - 1);
        val = src[c * src_h * src_w + sy * src_w + sx];
    } else {
        val = pad_val;
    }

    dst[c * dst_size * dst_size + dy * dst_size + dx] = val;
}

void launch_letterbox_gpu(
    const float* src, int src_w, int src_h,
    float*       dst, int dst_size,
    int x_off, int y_off, int w_new, int h_new,
    float pad_val, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (dst_size + block.x - 1) / block.x,
        (dst_size + block.y - 1) / block.y,
        3);
    letterbox_kernel<<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h,
        dst_size, x_off, y_off, w_new, h_new,
        pad_val);
}

} // namespace saccade
