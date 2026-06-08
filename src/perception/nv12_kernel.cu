#include <cuda_runtime.h>
#include <stdint.h>

namespace saccade {

__global__ void nv12_to_chw_letterbox_kernel(
    const uint8_t* __restrict__ y_ptr,  int y_pitch,
    const uint8_t* __restrict__ uv_ptr, int uv_pitch,
    int src_w, int src_h,
    float* __restrict__ dst_ptr, int dst_size,
    int x_off, int y_off, int w_new, int h_new,
    float pad_val)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z;

    if (dx >= dst_size || dy >= dst_size) return;

    int lx = dx - x_off;
    int ly = dy - y_off;

    if (lx < 0 || lx >= w_new || ly < 0 || ly >= h_new) {
        dst_ptr[c * dst_size * dst_size + dy * dst_size + dx] = pad_val;
        return;
    }

    int sx = min((int)(lx * ((float)src_w / w_new)), src_w - 1);
    int sy = min((int)(ly * ((float)src_h / h_new)), src_h - 1);

    float yf = (float)y_ptr[sy * y_pitch + sx];

    float uv_fx = (float)sx * 0.5f;
    float uv_fy = (float)sy * 0.5f;
    int uv_col0 = (int)floorf(uv_fx);
    int uv_row0 = (int)floorf(uv_fy);
    int uv_col1 = min(uv_col0 + 1, (src_w >> 1) - 1);
    int uv_row1 = min(uv_row0 + 1, (src_h >> 1) - 1);
    float fx = uv_fx - (float)uv_col0;
    float fy = uv_fy - (float)uv_row0;

    const uint8_t* uv00 = uv_ptr + uv_row0 * uv_pitch + uv_col0 * 2;
    const uint8_t* uv01 = uv_ptr + uv_row0 * uv_pitch + uv_col1 * 2;
    const uint8_t* uv10 = uv_ptr + uv_row1 * uv_pitch + uv_col0 * 2;
    const uint8_t* uv11 = uv_ptr + uv_row1 * uv_pitch + uv_col1 * 2;

    float cb00 = (float)uv00[0], cr00 = (float)uv00[1];
    float cb01 = (float)uv01[0], cr01 = (float)uv01[1];
    float cb10 = (float)uv10[0], cr10 = (float)uv10[1];
    float cb11 = (float)uv11[0], cr11 = (float)uv11[1];

    float cb = ((1.0f - fy) * ((1.0f - fx) * cb00 + fx * cb01) +
                      fy  * ((1.0f - fx) * cb10 + fx * cb11)) - 128.0f;
    float cr = ((1.0f - fy) * ((1.0f - fx) * cr00 + fx * cr01) +
                      fy  * ((1.0f - fx) * cr10 + fx * cr11)) - 128.0f;

    float val;
    if (c == 0)       val = yf + 1.402f * cr;
    else if (c == 1)  val = yf - 0.344136f * cb - 0.714136f * cr;
    else              val = yf + 1.772f * cb;

    val = fminf(fmaxf(val, 0.0f), 255.0f) / 255.0f;
    dst_ptr[c * dst_size * dst_size + dy * dst_size + dx] = val;
}

__global__ void nv12_to_chw_resize_kernel(
    const uint8_t* __restrict__ y_ptr,  int y_pitch,
    const uint8_t* __restrict__ uv_ptr, int uv_pitch,
    int src_w, int src_h,
    float* __restrict__ dst_ptr, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z;

    if (dx >= dst_w || dy >= dst_h) return;

    int sx = min((int)(dx * ((float)src_w / dst_w)), src_w - 1);
    int sy = min((int)(dy * ((float)src_h / dst_h)), src_h - 1);

    float yf = (float)y_ptr[sy * y_pitch + sx];

    float uv_fx = (float)sx * 0.5f;
    float uv_fy = (float)sy * 0.5f;
    int uv_col0 = (int)floorf(uv_fx);
    int uv_row0 = (int)floorf(uv_fy);
    int uv_col1 = min(uv_col0 + 1, (src_w >> 1) - 1);
    int uv_row1 = min(uv_row0 + 1, (src_h >> 1) - 1);
    float fx = uv_fx - (float)uv_col0;
    float fy = uv_fy - (float)uv_row0;

    const uint8_t* uv00 = uv_ptr + uv_row0 * uv_pitch + uv_col0 * 2;
    const uint8_t* uv01 = uv_ptr + uv_row0 * uv_pitch + uv_col1 * 2;
    const uint8_t* uv10 = uv_ptr + uv_row1 * uv_pitch + uv_col0 * 2;
    const uint8_t* uv11 = uv_ptr + uv_row1 * uv_pitch + uv_col1 * 2;

    float cb00 = (float)uv00[0], cr00 = (float)uv00[1];
    float cb01 = (float)uv01[0], cr01 = (float)uv01[1];
    float cb10 = (float)uv10[0], cr10 = (float)uv10[1];
    float cb11 = (float)uv11[0], cr11 = (float)uv11[1];

    float cb = ((1.0f - fy) * ((1.0f - fx) * cb00 + fx * cb01) +
                      fy  * ((1.0f - fx) * cb10 + fx * cb11)) - 128.0f;
    float cr = ((1.0f - fy) * ((1.0f - fx) * cr00 + fx * cr01) +
                      fy  * ((1.0f - fx) * cr10 + fx * cr11)) - 128.0f;

    float val;
    if (c == 0)       val = yf + 1.402f * cr;
    else if (c == 1)  val = yf - 0.344136f * cb - 0.714136f * cr;
    else              val = yf + 1.772f * cb;

    val = fminf(fmaxf(val, 0.0f), 255.0f) / 255.0f;
    dst_ptr[c * dst_h * dst_w + dy * dst_w + dx] = val;
}

void launch_nv12_to_chw_letterbox(
    const uint8_t* y_ptr,  int y_pitch,
    const uint8_t* uv_ptr, int uv_pitch,
    int src_w, int src_h,
    float* dst_ptr, int dst_size,
    int x_off, int y_off, int w_new, int h_new,
    float pad_val, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (dst_size + block.x - 1) / block.x,
        (dst_size + block.y - 1) / block.y,
        3);
    nv12_to_chw_letterbox_kernel<<<grid, block, 0, stream>>>(
        y_ptr, y_pitch, uv_ptr, uv_pitch,
        src_w, src_h, dst_ptr, dst_size,
        x_off, y_off, w_new, h_new, pad_val);
}

void launch_nv12_to_chw_resize(
    const uint8_t* y_ptr,  int y_pitch,
    const uint8_t* uv_ptr, int uv_pitch,
    int src_w, int src_h,
    float* dst_ptr, int dst_w, int dst_h,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y,
        3);
    nv12_to_chw_resize_kernel<<<grid, block, 0, stream>>>(
        y_ptr, y_pitch, uv_ptr, uv_pitch,
        src_w, src_h, dst_ptr, dst_w, dst_h);
}

} // namespace saccade
