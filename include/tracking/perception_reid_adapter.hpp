#pragma once

// Header-only: FeatureExtractor / Cropper as PerceptionPipeline's ReID
// backend (tracking/reid_backend.hpp). Included by the Python extension that
// wires a perception ReID model into the pipeline -- never by a
// saccade_tracking source, which must not depend on perception (#465 PR-11).

#include "perception/feature_extractor.hpp"
#include "perception/preprocessor.hpp"
#include "tracking/reid_backend.hpp"
#include <memory>

namespace saccade {

class FeatureExtractorReid final : public ReidExtractor {
public:
    explicit FeatureExtractorReid(FeatureExtractor* fe) : fe_(fe) {}

    void extract(void* input_cuda_ptr, int num_images, void* output_cuda_ptr,
                 cudaStream_t stream) override {
        fe_->extract(input_cuda_ptr, num_images, output_cuda_ptr, stream);
    }
    int get_feature_dim() const override { return fe_->get_feature_dim(); }
    int get_max_batch() const override { return fe_->get_max_batch(); }
    std::pair<int, int> get_input_hw() const override { return fe_->get_input_hw(); }
    void set_profiling_enabled(bool enabled) override { fe_->set_profiling_enabled(enabled); }
    void reset_profile_stats() override { fe_->reset_profile_stats(); }
    ReidExtractStats get_profile_stats() const override {
        const FeatureExtractor::ProfileStats s = fe_->get_profile_stats();
        ReidExtractStats out;
        out.pre_normalize_ms = s.pre_normalize_ms;
        out.trt_enqueue_ms = s.trt_enqueue_ms;
        out.l2_normalize_ms = s.l2_normalize_ms;
        out.total_ms = s.total_ms;
        out.chunks = s.chunks;
        return out;
    }
    const void* wired_object() const override { return fe_; }

private:
    FeatureExtractor* fe_;  // not owned
};

class CropperRoi final : public RoiCropper {
public:
    explicit CropperRoi(Cropper* cropper) : cropper_(cropper) {}

    void process_gpu(void* input_cuda_ptr, int src_width, int src_height,
                     float* boxes, int num_boxes, void* output_cuda_ptr,
                     cudaStream_t stream) override {
        cropper_->process_gpu(input_cuda_ptr, src_width, src_height, boxes, num_boxes,
                              output_cuda_ptr, stream);
    }
    const void* wired_object() const override { return cropper_; }

private:
    Cropper* cropper_;  // not owned
};

// Null in, null out: the pipeline treats a missing backend as "no ReID".
inline std::unique_ptr<ReidExtractor> make_reid_extractor(FeatureExtractor* fe) {
    return fe ? std::make_unique<FeatureExtractorReid>(fe) : nullptr;
}
inline std::unique_ptr<RoiCropper> make_roi_cropper(Cropper* cropper) {
    return cropper ? std::make_unique<CropperRoi>(cropper) : nullptr;
}

} // namespace saccade
