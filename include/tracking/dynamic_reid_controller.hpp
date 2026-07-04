#pragma once

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace saccade {

struct ReIDTrackObservation {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float det_score = 0;

    ReIDTrackObservation() = default;
    ReIDTrackObservation(float _x1, float _y1, float _x2, float _y2,
                         float _score)
        : x1(_x1), y1(_y1), x2(_x2), y2(_y2), det_score(_score) {}
};

struct ReIDFrameStats {
    int new_tracks = 0;
    int lost_tracks = 0;
    int unstable_tracks = 0;
};

class DynamicReIDController {
public:
    DynamicReIDController(
        int history_size = 5,
        const std::string& mode = "event_any",
        float unstable_iou = 0.50f,
        float unstable_center_shift = 0.30f,
        int crowd_threshold = 8,
        float long_memory_decay = 0.80f,
        float long_memory_trigger = 1.25f,
        float score_decay = 0.80f,
        float score_threshold = 2.0f,
        float score_threshold_low = 0.0f,
        float weight_new = 1.0f,
        float weight_lost = 1.4f,
        float weight_geom = 0.5f,
        float weight_conf = 0.5f,
        float birth_death_boost = 1.0f,
        float birth_death_lost_min = 0.0f,
        int lost_age_cap = 30,
        float unstable_shift_weight = 1.0f,
        float unstable_iou_weight = 1.0f,
        float conf_jitter_gate = 0.10f,
        int trigger_persist_frames = 1,
        int cooldown_frames = 0);

    void observe(
        const std::unordered_map<int, ReIDTrackObservation>& tracks,
        const std::vector<float>& gmc = {});

    bool should_reid(int det_count);

    std::unordered_map<int, float> get_priorities() const;

private:
    float box_iou_impl(float ax1, float ay1, float ax2, float ay2,
                       float bx1, float by1, float bx2, float by2) const;
    float center_shift_ratio_impl(float ax1, float ay1, float ax2,
                                  float ay2, float bx1, float by1,
                                  float bx2, float by2) const;
    bool persist(float trigger_score);

    int history_size_;
    std::string mode_;
    float unstable_iou_;
    float unstable_center_shift_;
    int crowd_threshold_;
    float long_memory_decay_;
    float long_memory_trigger_;
    float score_decay_;
    float score_threshold_;
    float score_threshold_low_;
    float weight_new_;
    float weight_lost_;
    float weight_geom_;
    float weight_conf_;
    float birth_death_boost_;
    float birth_death_lost_min_;
    int lost_age_cap_;
    float unstable_shift_weight_;
    float unstable_iou_weight_;
    float conf_jitter_gate_;
    int trigger_persist_frames_;
    int cooldown_frames_;

    std::deque<std::unordered_map<int, ReIDTrackObservation>> track_history_;
    std::deque<ReIDFrameStats> frame_stats_;
    float event_memory_ = 0;
    std::unordered_map<int, int> track_ages_;
    std::unordered_map<int, float> track_score_ema_;
    float score_new_ = 0;
    float score_lost_ = 0;
    float score_geom_ = 0;
    float score_conf_ = 0;
    float last_birth_death_boost_ = 0;
    std::vector<int> last_new_ids_;
    std::vector<int> last_lost_ids_;
    std::unordered_map<int, float> per_track_instability_;
    std::unordered_map<int, float> per_track_conf_jitter_;
    int persist_counter_ = 0;
    int cooldown_remaining_ = 0;
};

}  // namespace saccade
