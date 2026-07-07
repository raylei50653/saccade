#include "tracking/dynamic_reid_controller.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace saccade {

DynamicReIDController::DynamicReIDController(
    int history_size, const std::string& mode, float unstable_iou,
    float unstable_center_shift, int crowd_threshold,
    float long_memory_decay, float long_memory_trigger, float score_decay,
    float score_threshold, float score_threshold_low, float weight_new,
    float weight_lost, float weight_geom, float weight_conf,
    float birth_death_boost, float birth_death_lost_min, int lost_age_cap,
    float unstable_shift_weight, float unstable_iou_weight,
    float conf_jitter_gate, int trigger_persist_frames,
    int cooldown_frames)
    : history_size_(std::max(2, history_size)),
      mode_(mode),
      unstable_iou_(unstable_iou),
      unstable_center_shift_(unstable_center_shift),
      crowd_threshold_(crowd_threshold),
      long_memory_decay_(
          std::min(std::max(long_memory_decay, 0.0f), 0.99f)),
      long_memory_trigger_(std::max(long_memory_trigger, 0.0f)),
      score_decay_(std::min(std::max(score_decay, 0.0f), 0.99f)),
      score_threshold_(std::max(score_threshold, 0.0f)),
      score_threshold_low_(score_threshold_low > 0.0f
                               ? std::max(score_threshold_low, 0.0f)
                               : std::max(score_threshold, 0.0f)),
      weight_new_(weight_new),
      weight_lost_(weight_lost),
      weight_geom_(weight_geom),
      weight_conf_(weight_conf),
      birth_death_boost_(std::max(birth_death_boost, 0.0f)),
      birth_death_lost_min_(std::max(birth_death_lost_min, 0.0f)),
      lost_age_cap_(std::max(lost_age_cap, 1)),
      unstable_shift_weight_(std::max(unstable_shift_weight, 0.0f)),
      unstable_iou_weight_(std::max(unstable_iou_weight, 0.0f)),
      conf_jitter_gate_(std::max(conf_jitter_gate, 0.0f)),
      trigger_persist_frames_(std::max(trigger_persist_frames, 1)),
      cooldown_frames_(std::max(cooldown_frames, 0)) {}

float DynamicReIDController::box_iou_impl(float ax1, float ay1, float ax2,
                                          float ay2, float bx1, float by1,
                                          float bx2, float by2) const {
    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float iarea = iw * ih;
    float aarea = std::max(1e-6f, (ax2 - ax1) * (ay2 - ay1));
    float barea = std::max(1e-6f, (bx2 - bx1) * (by2 - by1));
    return iarea / (aarea + barea - iarea);
}

float DynamicReIDController::center_shift_ratio_impl(
    float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
    float bx2, float by2) const {
    float cx1 = (ax1 + ax2) * 0.5f;
    float cy1 = (ay1 + ay2) * 0.5f;
    float cx2 = (bx1 + bx2) * 0.5f;
    float cy2 = (by1 + by2) * 0.5f;
    float dx = cx1 - cx2;
    float dy = cy1 - cy2;
    float dist = std::sqrt(dx * dx + dy * dy);
    float diag = std::sqrt((ax2 - ax1) * (ax2 - ax1) +
                           (ay2 - ay1) * (ay2 - ay1));
    return diag > 1e-6f ? dist / diag : 0.0f;
}

void DynamicReIDController::observe(
    const std::unordered_map<int, ReIDTrackObservation>& tracks,
    const std::vector<float>& gmc) {

    const auto& prev =
        track_history_.empty()
            ? std::unordered_map<int, ReIDTrackObservation>()
            : track_history_.back();

    std::unordered_set<int> curr_ids, prev_ids;
    for (const auto& [tid, _] : tracks) curr_ids.insert(tid);
    for (const auto& [tid, _] : prev) prev_ids.insert(tid);

    std::vector<int> shared_ids;
    for (int tid : curr_ids)
        if (prev_ids.count(tid)) shared_ids.push_back(tid);

    float h00 = 1, h01 = 0, h02 = 0, h10 = 0, h11 = 1, h12 = 0;
    bool has_gmc = gmc.size() >= 6;
    if (has_gmc) {
        h00 = gmc[0];
        h01 = gmc[1];
        h02 = gmc[2];
        h10 = gmc[3];
        h11 = gmc[4];
        h12 = gmc[5];
    }

    int unstable = 0;
    float unstable_signal = 0;
    float conf_signal = 0;
    per_track_instability_.clear();
    per_track_conf_jitter_.clear();

    for (int tid : shared_ids) {
        auto cit = tracks.find(tid);
        auto pit = prev.find(tid);
        if (cit == tracks.end() || pit == prev.end()) continue;

        float cx1 = cit->second.x1, cy1 = cit->second.y1;
        float cx2 = cit->second.x2, cy2 = cit->second.y2;
        float px1 = pit->second.x1, py1 = pit->second.y1;
        float px2 = pit->second.x2, py2 = pit->second.y2;

        if (has_gmc) {
            float corners_x[4] = {px1, px2, px2, px1};
            float corners_y[4] = {py1, py1, py2, py2};
            float tx[4], ty[4];
            for (int k = 0; k < 4; ++k) {
                tx[k] = h00 * corners_x[k] + h01 * corners_y[k] + h02;
                ty[k] = h10 * corners_x[k] + h11 * corners_y[k] + h12;
            }
            px1 = *std::min_element(tx, tx + 4);
            py1 = *std::min_element(ty, ty + 4);
            px2 = *std::max_element(tx, tx + 4);
            py2 = *std::max_element(ty, ty + 4);
        }

        float iou =
            box_iou_impl(cx1, cy1, cx2, cy2, px1, py1, px2, py2);
        float shift = center_shift_ratio_impl(cx1, cy1, cx2, cy2, px1,
                                               py1, px2, py2);
        float shift_term =
            std::max(0.0f, shift - unstable_center_shift_);
        float iou_term = std::max(0.0f, unstable_iou_ - iou);
        float instability = unstable_shift_weight_ * shift_term +
                            unstable_iou_weight_ * iou_term;
        if (instability > 0.0f) {
            unstable++;
            unstable_signal += instability;
            per_track_instability_[tid] = instability;
        }

        float prev_ema = track_score_ema_.count(tid)
                             ? track_score_ema_[tid]
                             : pit->second.det_score;
        float jitter =
            std::max(0.0f, std::abs(cit->second.det_score - prev_ema) -
                               conf_jitter_gate_);
        if (jitter > 0.0f) {
            conf_signal += jitter;
            per_track_conf_jitter_[tid] = jitter;
        }
    }

    float new_signal = 0;
    for (int tid : curr_ids)
        if (!prev_ids.count(tid))
            new_signal += tracks.at(tid).det_score;

    float lost_signal = 0;
    for (int tid : prev_ids)
        if (!curr_ids.count(tid))
            lost_signal += std::min(
                1.0f,
                static_cast<float>(
                    track_ages_.count(tid) ? track_ages_[tid] : 1) /
                    static_cast<float>(lost_age_cap_));

    int matched_count = std::max(1, static_cast<int>(shared_ids.size()));
    unstable_signal /= static_cast<float>(matched_count);
    conf_signal /= static_cast<float>(matched_count);

    while (static_cast<int>(track_history_.size()) >= history_size_)
        track_history_.pop_front();
    track_history_.push_back(tracks);

    ReIDFrameStats stats;
    stats.new_tracks =
        static_cast<int>(curr_ids.size()) - static_cast<int>(shared_ids.size());
    stats.lost_tracks =
        static_cast<int>(prev_ids.size()) - static_cast<int>(shared_ids.size());
    stats.unstable_tracks = unstable;

    while (static_cast<int>(frame_stats_.size()) >= history_size_)
        frame_stats_.pop_front();
    frame_stats_.push_back(stats);

    float event_strength =
        static_cast<float>(stats.new_tracks + stats.lost_tracks) +
        0.5f * static_cast<float>(stats.unstable_tracks);
    event_memory_ =
        long_memory_decay_ * event_memory_ + event_strength;
    score_new_ = score_decay_ * score_new_ + new_signal;
    score_lost_ = score_decay_ * score_lost_ + lost_signal;
    score_geom_ = score_decay_ * score_geom_ + unstable_signal;
    score_conf_ = score_decay_ * score_conf_ + conf_signal;

    last_new_ids_.clear();
    last_lost_ids_.clear();
    for (int tid : curr_ids)
        if (!prev_ids.count(tid)) last_new_ids_.push_back(tid);
    for (int tid : prev_ids)
        if (!curr_ids.count(tid)) last_lost_ids_.push_back(tid);

    last_birth_death_boost_ =
        (new_signal > 0.0f && lost_signal > 0.0f &&
         lost_signal >= birth_death_lost_min_)
            ? birth_death_boost_
            : 0.0f;

    std::unordered_map<int, int> next_ages;
    std::unordered_map<int, float> next_score_ema;
    for (int tid : curr_ids) {
        next_ages[tid] =
            (prev_ids.count(tid) ? (track_ages_.count(tid)
                                        ? track_ages_[tid]
                                        : 0) + 1
                                 : 1);
        float prev_ema_val =
            prev_ids.count(tid) && track_score_ema_.count(tid)
                ? track_score_ema_[tid]
                : tracks.at(tid).det_score;
        next_score_ema[tid] =
            prev_ids.count(tid)
                ? score_decay_ * prev_ema_val +
                      (1.0f - score_decay_) * tracks.at(tid).det_score
                : tracks.at(tid).det_score;
    }
    track_ages_ = std::move(next_ages);
    track_score_ema_ = std::move(next_score_ema);
}

bool DynamicReIDController::persist(float trigger_score) {
    if (cooldown_remaining_ > 0) {
        cooldown_remaining_--;
        persist_counter_ = 0;
        return false;
    }
    bool triggered = persist_counter_ > 0
                         ? trigger_score >= score_threshold_low_
                         : trigger_score >= score_threshold_;
    if (triggered)
        persist_counter_++;
    else
        persist_counter_ = 0;
    if (persist_counter_ >= trigger_persist_frames_) {
        persist_counter_ = 0;
        cooldown_remaining_ = cooldown_frames_;
        return true;
    }
    return false;
}

bool DynamicReIDController::should_reid(int det_count) {
    if (det_count <= 0 || track_history_.empty()) return false;
    int active_count = static_cast<int>(track_history_.back().size());
    if (active_count <= 0) return false;
    if (frame_stats_.size() < 2) return false;

    if (det_count >= active_count + 2) return true;

    int recent_new = 0, recent_lost = 0, recent_unstable = 0;
    for (const auto& s : frame_stats_) {
        recent_new += s.new_tracks;
        recent_lost += s.lost_tracks;
        recent_unstable += s.unstable_tracks;
    }

    const auto& latest = frame_stats_.back();
    const auto& prev = frame_stats_[frame_stats_.size() - 2];
    int latest_events = latest.new_tracks + latest.lost_tracks;
    bool persistent_events =
        latest_events > 0 && (prev.new_tracks + prev.lost_tracks) > 0;
    int unstable_now = latest.unstable_tracks;
    bool unstable_persistent =
        unstable_now > 0 && prev.unstable_tracks > 0;

    if (mode_ == "count_jump")
        return det_count >= active_count + 2;

    if (mode_ == "event_memory") {
        if (event_memory_ >= long_memory_trigger_) return true;
        if (det_count >= active_count + 2) return true;
        return false;
    }

    if (mode_ == "score_ema") {
        float trigger_score =
            weight_new_ * score_new_ + weight_lost_ * score_lost_ +
            weight_geom_ * score_geom_ + weight_conf_ * score_conf_ +
            last_birth_death_boost_;
        return persist(trigger_score);
    }

    if (mode_ == "score_ema_geom") {
        float trigger_score = weight_new_ * score_new_ +
                              weight_lost_ * score_lost_ +
                              weight_geom_ * score_geom_ +
                              last_birth_death_boost_;
        return persist(trigger_score);
    }

    if (mode_ == "score_ema_conf") {
        float trigger_score = weight_new_ * score_new_ +
                              weight_lost_ * score_lost_ +
                              weight_conf_ * score_conf_ +
                              last_birth_death_boost_;
        return persist(trigger_score);
    }

    if (mode_ == "event_strict") {
        if (persistent_events) return true;
        if (unstable_now >= std::max(2, std::min(active_count, 3)) &&
            unstable_persistent)
            return true;
        if (event_memory_ >= long_memory_trigger_ && unstable_now > 0)
            return true;
        if (active_count >= crowd_threshold_ &&
            det_count >= active_count + 1 && unstable_persistent)
            return true;
        return false;
    }

    if (mode_ == "event_persist") {
        if (persistent_events) return true;
        if (unstable_now >= std::max(2, std::min(active_count, 4)) &&
            unstable_persistent)
            return true;
        if (event_memory_ >= long_memory_trigger_ && latest_events > 0)
            return true;
        if (active_count >= crowd_threshold_ &&
            det_count >= active_count && unstable_persistent)
            return true;
        return false;
    }

    if (recent_new > 0 || recent_lost > 0) return true;
    if (recent_unstable >= std::max(2, std::min(active_count, 4)))
        return true;
    if (active_count >= crowd_threshold_ &&
        det_count >= active_count && recent_unstable > 0)
        return true;
    return false;
}

std::unordered_map<int, float> DynamicReIDController::get_priorities()
    const {
    std::unordered_map<int, float> priorities;

    for (int tid : last_new_ids_) {
        priorities[tid] =
            weight_new_ *
                (track_score_ema_.count(tid) ? track_score_ema_.at(tid)
                                             : 0.5f) +
            last_birth_death_boost_;
    }

    for (const auto& [tid, age] : track_ages_) {
        if (std::find(last_new_ids_.begin(), last_new_ids_.end(), tid) !=
            last_new_ids_.end())
            continue;
        float inst = per_track_instability_.count(tid)
                         ? per_track_instability_.at(tid)
                         : 0.0f;
        float jitter = per_track_conf_jitter_.count(tid)
                           ? per_track_conf_jitter_.at(tid)
                           : 0.0f;
        float priority = weight_geom_ * inst + weight_conf_ * jitter;
        if (priority <= 0.0f)
            priority = 0.1f * (track_score_ema_.count(tid)
                                   ? track_score_ema_.at(tid)
                                   : 0.5f);
        priorities[tid] = priority;
    }

    for (int tid : last_lost_ids_) {
        priorities[tid] = weight_lost_ * 1.0f;
    }

    return priorities;
}

}  // namespace saccade
