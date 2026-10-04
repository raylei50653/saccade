#pragma once

#include <cuda_runtime.h>
#include <utility>

namespace saccade {

/**
 * @brief What PerceptionPipeline's ReID path needs from an embedder and a
 * cropper.
 *
 * Declared on the tracking side so that saccade_tracking does not link
 * saccade_perception (#465 PR-11): the perception classes (FeatureExtractor,
 * Cropper) are adapted to these in tracking/perception_reid_adapter.hpp, which
 * only the extensions include. The shipping runtime passes no ReID backend.
 */
struct ReidExtractStats {
    double pre_normalize_ms = 0.0;
    double trt_enqueue_ms = 0.0;
    double l2_normalize_ms = 0.0;
    double total_ms = 0.0;
    int chunks = 0;
};

class ReidExtractor {
public:
    virtual ~ReidExtractor() = default;

    // [N, 3, H, W] float32 RGB crops in [0, 1] -> [N, feature_dim] embeddings.
    virtual void extract(void* input_cuda_ptr, int num_images, void* output_cuda_ptr,
                         cudaStream_t stream) = 0;
    virtual int get_feature_dim() const = 0;
    virtual int get_max_batch() const = 0;
    virtual std::pair<int, int> get_input_hw() const = 0;
    virtual void set_profiling_enabled(bool enabled) = 0;
    virtual void reset_profile_stats() = 0;
    virtual ReidExtractStats get_profile_stats() const = 0;

    // The object the caller wired in (an adapter returns what it wraps);
    // PerceptionPipelineSnapshot::reid_ptr reports its address.
    virtual const void* wired_object() const { return this; }
};

class RoiCropper {
public:
    virtual ~RoiCropper() = default;

    // Batched RoI crop + resize + CHW on the GPU.
    virtual void process_gpu(void* input_cuda_ptr, int src_width, int src_height,
                             float* boxes, int num_boxes, void* output_cuda_ptr,
                             cudaStream_t stream) = 0;

    // As ReidExtractor::wired_object (PerceptionPipelineSnapshot::cropper_ptr).
    virtual const void* wired_object() const { return this; }
};

} // namespace saccade
