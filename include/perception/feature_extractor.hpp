#pragma once

#include "saccade/common.hpp"
#include "perception/trt_engine.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>

namespace saccade {

enum class ModelType {
    SIGLIP2,
    DINOV2,
    TRANSREID
};

/**
 * @brief High-level Feature Extractor using TensorRT.
 * Handles model-specific normalization, batching, and inference.
 */
class SACCADE_PERCEPTION_API FeatureExtractor {
public:
    FeatureExtractor(const std::string& model_path, ModelType type, int max_batch = 32);
    ~FeatureExtractor();

    /**
     * @brief Extract features from a batch of images on GPU.
     * @param input_cuda_ptr GPU pointer to [N, 3, H, W] float32 RGB tensor in [0, 1]
     * @param num_images Number of images in batch
     * @param output_cuda_ptr GPU pointer to [N, feature_dim] float32 tensor
     * @param stream CUDA stream
     */
    void extract(void* input_cuda_ptr, int num_images, void* output_cuda_ptr, cudaStream_t stream);

    int get_feature_dim() const;
    int get_max_batch() const;
    std::pair<int, int> get_input_hw() const;

private:
    std::unique_ptr<TRTEngine> engine_;
    ModelType type_;
    int max_batch_;
    int feature_dim_;
    int input_h_, input_w_;
    bool is_dynamic_;

    std::string input_name_;
    std::string output_name_;
    std::vector<std::pair<std::string, void*>> scratch_buffers_;

    // Model-specific normalization parameters on GPU
    void* d_mean_ = nullptr;
    void* d_std_ = nullptr;
};

} // namespace saccade
