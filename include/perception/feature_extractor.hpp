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
    MOBILENETV4_REID,
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

    /**
     * @brief LaSt-ViT refined extraction (SigLIP2 only).
     *        Runs standard TRT inference, then applies FFT-based frequency filtering
     *        on last_hidden_state to produce a stability-aware embedding.
     * @param input_cuda_ptr GPU [N, 3, H, W] float32 in [0,1]
     * @param num_images     Number of images
     * @param embedding_out  GPU [N, feature_dim] L2-normalized output
     * @param stability_out  GPU [N] float32 per-image stability score in [0,1]
     * @param stream         CUDA stream
     * @param sigma_embed    Gaussian sigma for Top-K patch selection (default 0.015)
     * @param sigma_gate     Gaussian sigma for stability gating (default 0.040)
     * @param top_k_ratio    Fraction of N patches to pool (default 0.5)
     */
    void extract_with_stability(
        void* input_cuda_ptr, int num_images,
        void* embedding_out, void* stability_out,
        cudaStream_t stream,
        float sigma_embed   = 0.015f,
        float sigma_gate    = 0.040f,
        float top_k_ratio   = 0.5f);

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
    std::string lhs_name_;   // "last_hidden_state" for SigLIP2, empty otherwise
    int lhs_N_ = 0;          // sequence length (196 for 224×224 SigLIP2)
    int lhs_C_ = 0;          // channel dim (768)
    std::vector<std::pair<std::string, void*>> scratch_buffers_;
    void* d_input_scratch_ = nullptr;
    bool profiling_enabled_ = false;
    ProfileStats last_profile_stats_{};

    // Model-specific normalization parameters on GPU
    void* d_mean_ = nullptr;
    void* d_std_ = nullptr;
};

} // namespace saccade
