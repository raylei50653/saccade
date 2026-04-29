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

// --- Cropper Kernel ---

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

    if (x < crop_w && y < crop_h) {
        const float* box = boxes + box_idx * 4;
        float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
        float bw = fmaxf(x2 - x1, 1e-6f);
        float bh = fmaxf(y2 - y1, 1e-6f);

        float sx = x1 + (x + 0.5f) * (bw / crop_w);
        float sy = y1 + (y + 0.5f) * (bh / crop_h);

        // Bi-linear interpolation
        int x_low = (int)floorf(sx);
        int y_low = (int)floorf(sy);
        int x_high = x_low + 1;
        int y_high = y_low + 1;

        float dx = sx - x_low;
        float dy = sy - y_low;

        auto get_pixel = [&](int px, int py, int c) {
            px = max(0, min(px, src_w - 1));
            py = max(0, min(py, src_h - 1));
            return src[(py * src_w + px) * 3 + c];
        };

        float* dst_ptr = dst + box_idx * (3 * crop_w * crop_h);
        int plane_size = crop_w * crop_h;
        int spatial_idx = y * crop_w + x;

        for (int c = 0; c < 3; ++c) {
            float p00 = get_pixel(x_low, y_low, c);
            float p01 = get_pixel(x_high, y_low, c);
            float p10 = get_pixel(x_low, y_high, c);
            float p11 = get_pixel(x_high, y_high, c);

            float val = (1.0f - dx) * (1.0f - dy) * p00 +
                        dx * (1.0f - dy) * p01 +
                        (1.0f - dx) * dy * p10 +
                        dx * dy * p11;
            
            dst_ptr[c * plane_size + spatial_idx] = val;
        }
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

} // namespace saccade
