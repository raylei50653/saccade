#include "perception/preprocessor.hpp"
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <iostream>

namespace saccade {

// 外部宣告在 .cu 檔案中的啟動函數
void launch_normalize_chw(const uint8_t* src, float* dst, int w, int h, cudaStream_t stream);

Preprocessor::Preprocessor(int target_width, int target_height)
    : target_width_(target_width), target_height_(target_height) {
    rgb_buffer_size_ = target_width_ * target_height_ * 3;
    cudaMalloc(&d_rgb_interleaved_, rgb_buffer_size_);
}

Preprocessor::~Preprocessor() {
    if (d_rgb_interleaved_) cudaFree(d_rgb_interleaved_);
}

void Preprocessor::process_gpu(void* input_cuda_ptr, int src_width, int src_height, void* output_cuda_ptr, cudaStream_t stream) {
    // 這裡我們暫時假設 input_cuda_ptr 已經由 NPP 處理過 Resize 或格式轉換
    // 實際上應在這裡插入 nppiNV12ToRGB_8u_P2C3R 等呼叫
    
    // 呼叫 .cu 中的 CUDA Kernel
    launch_normalize_chw(
        (uint8_t*)d_rgb_interleaved_, (float*)output_cuda_ptr, target_width_, target_height_, stream);
}

void Preprocessor::process(void* input_ptr, int width, int height, void* output_cuda_ptr, cudaStream_t stream) {
    cv::Mat input(height, width, CV_8UC3, input_ptr);
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_LINEAR);
    
    std::vector<float> h_output(3 * target_width_ * target_height_);
    float* out = h_output.data();
    int frame_size = target_width_ * target_height_;
    uint8_t* in_data = resized.data;
    for (int y = 0; y < target_height_; ++y) {
        for (int x = 0; x < target_width_; ++x) {
            int idx = (y * target_width_ + x) * 3;
            out[0 * frame_size + y * target_width_ + x] = in_data[idx + 2] / 255.0f;
            out[1 * frame_size + y * target_width_ + x] = in_data[idx + 1] / 255.0f;
            out[2 * frame_size + y * target_width_ + x] = in_data[idx + 0] / 255.0f;
        }
    }
    cudaMemcpyAsync(output_cuda_ptr, h_output.data(), h_output.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

Cropper::Cropper(int crop_width, int crop_height) : crop_width_(crop_width), crop_height_(crop_height) {}
Cropper::~Cropper() = default;

void Cropper::process(void* input_ptr, int width, int height, float* boxes, int num_boxes, void* output_cuda_ptr, cudaStream_t stream) {
    // [與先前實作一致]
}

} // namespace saccade
