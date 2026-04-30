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
    TRANSREID,
    OSNET,
    FASTREID,
};

/**
 * @brief High-level Feature Extractor using TensorRT.
 * Handles model-specific normalization, batching, and inference.
 */
class SACCADE_PERCEPTION_API FeatureExtractor {
public:
    struct ProfileStats {
        double pre_normalize_ms = 0.0;
        double trt_enqueue_ms = 0.0;
        double l2_normalize_ms = 0.0;
        double total_ms = 0.0;
        int chunks = 0;
        int images = 0;
    };

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

    /**
     * @brief Fused 3-part extraction: takes [3*num_dets, C, H, W] crops (parts stacked),
     *        extracts features, applies weighted average [0.5, 0.3, 0.2], and L2-normalizes.
     *        Output: [num_dets, feature_dim].
     */
    void extract_parts_fused(void* input_cuda_ptr, int num_dets, void* output_cuda_ptr, cudaStream_t stream);

    int get_feature_dim() const;
    int get_max_batch() const;
    std::pair<int, int> get_input_hw() const;
    void set_profiling_enabled(bool enabled);
    void reset_profile_stats();
    ProfileStats get_profile_stats() const;

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
    bool profiling_enabled_ = false;
    ProfileStats last_profile_stats_{};

    // Model-specific normalization parameters on GPU
    void* d_mean_ = nullptr;
    void* d_std_ = nullptr;
};

} // namespace saccade
