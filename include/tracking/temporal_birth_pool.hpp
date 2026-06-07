#pragma once

#include "saccade/common.hpp"
#include <cuda_runtime.h>

namespace saccade {

class SACCADE_TRACKING_API TemporalBirthPool {
public:
    struct Config {
        int frames = 3;
        float iou_thresh = 0.50f;
        float boost = 0.25f;
        float min_score = 0.20f;
        float min_motion_px = 0.0f;
        float new_track_thresh = 0.70f;
        float high_thresh = 0.80f;
        int capacity = 2048;
    };

    explicit TemporalBirthPool(const Config& cfg);
    ~TemporalBirthPool();

    void reset();
    void update_and_apply(
        const float* boxes_ptr,
        float* scores_ptr,
        int num_dets,
        cudaStream_t stream);

private:
    void ensure_capacity(int num_dets);

    Config cfg_{};
    int capacity_ = 0;
    int required_window_ = 0;
    int window_capacity_ = 0;
    int next_slot_ = 0;
    int filled_ = 0;

    float* d_ring_boxes_ = nullptr;
    int* d_ring_counts_ = nullptr;
};

} // namespace saccade
