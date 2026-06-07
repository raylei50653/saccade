#pragma once

#include "saccade/common.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace saccade {

class SACCADE_TRACKING_API MultiSignalBirthPool {
public:
    struct Config {
        float new_track_thresh = 0.35f;
        float min_score = 0.12f;
        int min_frames = 3;
        float target_motion_px = 12.0f;
        float evidence_threshold = 0.60f;
        float iou_match = 0.30f;
        int ttl_frames = 5;
        float w_score = 0.35f;
        float w_motion = 0.30f;
        float w_quality = 0.20f;
        float w_streak = 0.15f;
        float min_aspect = 0.0f;
        int max_area_px = 0;
        bool replace_mode = false;
        float replace_evidence_threshold = 0.85f;
        int capacity = 2048;
        int max_candidates = 256;
        int max_history = 10;
    };

    struct Candidate {
        std::vector<float> history_boxes;
        std::vector<float> history_scores;
        std::vector<int> history_frames;
        int last_frame = 0;
    };

    explicit MultiSignalBirthPool(const Config& cfg);
    ~MultiSignalBirthPool();

    void reset();
    void update_and_apply(
        int frame_id,
        const float* boxes_ptr,
        const float* scores_ptr,
        int num_dets,
        bool* promote_mask_ptr,
        bool* replace_mask_ptr,
        cudaStream_t stream);

private:
    void ensure_capacity(int num_dets);

    Config cfg_{};
    int capacity_ = 0;
    std::vector<Candidate> candidates_;

    float* d_cand_last_boxes_ = nullptr;
    int* d_compact_idx_ = nullptr;
    int* d_compact_count_ = nullptr;
    int active_cand_count_ = 0;
    int cand_capacity_ = 0;
};

} // namespace saccade
