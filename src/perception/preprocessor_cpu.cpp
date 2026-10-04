// Preprocessor's legacy CPU path (#465 PR-11: moved verbatim out of
// preprocessor.cpp so that saccade_perception needs no OpenCV; only
// saccade_node calls it).
#include "perception/preprocessor.hpp"
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <vector>

namespace saccade {

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

} // namespace saccade
