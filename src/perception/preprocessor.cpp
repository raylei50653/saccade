#include "perception/preprocessor.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace saccade {

Preprocessor::Preprocessor(int target_width, int target_height)
    : target_width_(target_width), target_height_(target_height) {}

Preprocessor::~Preprocessor() = default;

void Preprocessor::process(void* input_ptr, int width, int height, void* output_cuda_ptr, cudaStream_t stream) {
    // 1. CPU 端 Resize (降級方案)
    cv::Mat input(height, width, CV_8UC3, input_ptr);
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_LINEAR);
    
    // 2. CPU 端 BGR -> RGB & HWC -> CHW & Normalize
    // 這部分雖然耗時，但在 5-Buffer Pool 緩衝下可接受
    std::vector<float> h_output(3 * target_width_ * target_height_);
    float* out = h_output.data();
    int frame_size = target_width_ * target_height_;
    
    uint8_t* in_data = resized.data;
    for (int y = 0; y < target_height_; ++y) {
        for (int x = 0; x < target_width_; ++x) {
            int idx = (y * target_width_ + x) * 3;
            // BGR -> RGB & Normalize
            out[0 * frame_size + y * target_width_ + x] = in_data[idx + 2] / 255.0f;
            out[1 * frame_size + y * target_width_ + x] = in_data[idx + 1] / 255.0f;
            out[2 * frame_size + y * target_width_ + x] = in_data[idx + 0] / 255.0f;
        }
    }

    // 3. 上傳到 GPU 供 YOLO 推理
    cudaMemcpyAsync(output_cuda_ptr, h_output.data(), h_output.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

// --- Cropper ---

Cropper::Cropper(int crop_width, int crop_height)
    : crop_width_(crop_width), crop_height_(crop_height) {}

Cropper::~Cropper() = default;

void Cropper::process(void* input_ptr, int width, int height, 
                      float* boxes, int num_boxes, void* output_cuda_ptr, cudaStream_t stream) {
    if (num_boxes <= 0) return;

    cv::Mat input(height, width, CV_8UC3, input_ptr);
    size_t batch_size = 3 * crop_width_ * crop_height_;
    std::vector<float> h_batch_output(num_boxes * batch_size);

    for (int i = 0; i < num_boxes; ++i) {
        float* box = boxes + i * 4;
        cv::Rect roi(static_cast<int>(box[0]), static_cast<int>(box[1]), 
                     static_cast<int>(box[2] - box[0]), static_cast<int>(box[3] - box[1]));
        
        // 邊界檢查
        roi.x = std::max(0, std::min(roi.x, width - 1));
        roi.y = std::max(0, std::min(roi.y, height - 1));
        roi.width = std::max(1, std::min(roi.width, width - roi.x));
        roi.height = std::max(1, std::min(roi.height, height - roi.y));
        
        cv::Mat d_roi = input(roi);
        cv::Mat d_resized;
        cv::resize(d_roi, d_resized, cv::Size(crop_width_, crop_height_), 0, 0, cv::INTER_LINEAR);
        
        float* out = h_batch_output.data() + i * batch_size;
        int frame_size = crop_width_ * crop_height_;
        uint8_t* in_data = d_resized.data;

        for (int y = 0; y < crop_height_; ++y) {
            for (int x = 0; x < crop_width_; ++x) {
                int idx = (y * crop_width_ + x) * 3;
                out[0 * frame_size + y * crop_width_ + x] = in_data[idx + 2] / 255.0f;
                out[1 * frame_size + y * crop_width_ + x] = in_data[idx + 1] / 255.0f;
                out[2 * frame_size + y * crop_width_ + x] = in_data[idx + 0] / 255.0f;
            }
        }
    }

    cudaMemcpyAsync(output_cuda_ptr, h_batch_output.data(), h_batch_output.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

} // namespace saccade
