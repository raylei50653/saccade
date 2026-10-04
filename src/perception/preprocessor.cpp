#include "perception/preprocessor.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

namespace saccade {

// 外部宣告在 .cu 檔案中的啟動函數
void launch_normalize_chw(const uint8_t* src, float* dst, int w, int h, cudaStream_t stream);
void launch_batch_crop_resize(
    const float* src, float* dst, 
    int src_w, int src_h,
    const float* boxes, int num_boxes,
    int crop_w, int crop_h,
    cudaStream_t stream);

Preprocessor::Preprocessor(int target_width, int target_height)
    : target_width_(target_width), target_height_(target_height) {
    rgb_buffer_size_ = target_width_ * target_height_ * 3;
    cudaError_t err = cudaMalloc(&d_rgb_interleaved_, rgb_buffer_size_);
    if (err != cudaSuccess) {
        std::cerr << "[Preprocessor] cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        d_rgb_interleaved_ = nullptr;
    }
}

Preprocessor::~Preprocessor() {
    if (d_rgb_interleaved_) cudaFree(d_rgb_interleaved_);
}

void Preprocessor::process_gpu(void* input_cuda_ptr, int src_width, int src_height, void* output_cuda_ptr, cudaStream_t stream) {
    if (!input_cuda_ptr) {
        std::cerr << "[Preprocessor] input_cuda_ptr is null" << std::endl;
        return;
    }
    launch_normalize_chw(
        (uint8_t*)input_cuda_ptr, (float*)output_cuda_ptr, target_width_, target_height_, stream);
}

Cropper::Cropper(int crop_width, int crop_height) : crop_width_(crop_width), crop_height_(crop_height) {}
Cropper::~Cropper() = default;

void Cropper::process(void* input_ptr, int width, int height, float* boxes, int num_boxes, void* output_cuda_ptr, cudaStream_t stream) {
    // Legacy CPU path
}

void Cropper::process_gpu(void* input_cuda_ptr, int src_width, int src_height, 
                         float* boxes, int num_boxes, void* output_cuda_ptr, cudaStream_t stream) {
    launch_batch_crop_resize(
        (const float*)input_cuda_ptr, (float*)output_cuda_ptr,
        src_width, src_height,
        boxes, num_boxes,
        crop_width_, crop_height_,
        stream
    );
}

} // namespace saccade
