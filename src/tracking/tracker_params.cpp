// TrackerParams setters (#465 Phase B PR-4b). The canonicalization below is
// the legacy GPUByteTracker setter behaviour, moved here unchanged so the
// tracker and the CUDA-free shipping tests run the same code. One deliberate
// exception: set_oao_params keeps a non-positive score_w as given (see there).
#include "tracking/tracker_params.hpp"

#include <algorithm>
#include <cmath>

namespace saccade {

void TrackerParams::set_params(float track_thresh, float high_thresh, float match_thresh,
                               int track_buffer, float mid_thresh, int confirm_streak,
                               float confirm_score_thresh, bool adaptive_confirmation,
                               float new_track_thresh, int kalman_adapt_mode, float r_scale,
                               float vel_dir_weight, float fuse_score_weight,
                               float stage2_match_thresh, float birth_low_score_thresh,
                               float birth_prox_norm_thresh) {
    core.track_thresh = track_thresh;
    core.high_thresh = high_thresh;
    core.match_thresh = match_thresh;
    core.track_buffer = track_buffer;
    core.mid_thresh = mid_thresh;
    core.new_track_thresh = new_track_thresh >= 0.0f ? new_track_thresh : mid_thresh;
    core.confirm_streak = std::max(confirm_streak, 1);
    core.confirm_score_thresh = confirm_score_thresh;
    core.adaptive_confirmation = adaptive_confirmation;
    core.kalman_adapt_mode = kalman_adapt_mode;
    core.r_scale = std::max(0.01f, r_scale);
    core.vel_dir_weight = fmaxf(0.0f, vel_dir_weight);
    core.fuse_score_weight = std::clamp(fuse_score_weight, 0.0f, 1.0f);
    core.stage2_match_thresh = std::clamp(stage2_match_thresh, 0.0f, 1.0f);
    core.birth_low_score_thresh = fmaxf(0.0f, birth_low_score_thresh);
    core.birth_prox_norm_thresh = fmaxf(0.0f, birth_prox_norm_thresh);
}

void TrackerParams::set_reid_params(float cos_threshold, float iou_low, float iou_high,
                                    float weight, float cost_cos_w, float cost_iou_w,
                                    float cost_score_w) {
    reid.cos_threshold = cos_threshold;
    reid.iou_low = iou_low;
    reid.iou_high = iou_high;
    reid.weight = weight;
    reid.cost_cos_w = cost_cos_w;
    reid.cost_iou_w = cost_iou_w;
    reid.cost_score_w = cost_score_w;
}

void TrackerParams::set_reid_min_candidates(int min_candidates) {
    reid_min_candidates = std::max(1, min_candidates);
}

void TrackerParams::set_relink_params(
    bool enabled, int bank_cap, float sim_thresh, float cheb_lambda, float spatial_gate,
    int max_age, bool bidirectional, float bridge_px, int bridge_at, int bridge_min_lost,
    int bridge_ttl, float bridge_max_speed, float bridge_person_height, float bridge_fps,
    float bridge_margin, float bridge_spatial_gate, int bridge_anchor,
    float bridge_anchor_rate, float bridge_h_lo, float bridge_h_hi, float bridge_dir_bonus,
    float occ_gate_cover, int occ_gap_min, float occ_expand_px, float occ_expand_cover,
    float bridge_app_veto) {
    relink.enabled = enabled;
    relink.bank_cap = std::max(1, bank_cap);
    relink.sim_thresh = sim_thresh;
    relink.cheb_lambda = std::max(0.0f, cheb_lambda);
    relink.spatial_gate = std::max(0.0f, spatial_gate);
    relink.max_age = std::max(1, max_age);
    // Phase-4 bidirectional foot-bridge params.
    relink.bidirectional = bidirectional;
    relink.bridge_px = std::max(0.0f, bridge_px);
    relink.bridge_at = std::max(1, bridge_at);
    relink.bridge_min_lost = std::max(0, bridge_min_lost);
    relink.bridge_ttl = std::max(1, bridge_ttl);
    relink.bridge_max_speed = std::max(0.0f, bridge_max_speed);
    relink.bridge_person_height = std::max(0.0f, bridge_person_height);
    relink.bridge_fps = bridge_fps > 0.0f ? bridge_fps : 30.0f;
    relink.bridge_margin = std::max(0.0f, bridge_margin);
    relink.bridge_spatial_gate = std::max(0.0f, bridge_spatial_gate);
    relink.bridge_anchor = (bridge_anchor < 0 || bridge_anchor > 2) ? 0 : bridge_anchor;
    relink.bridge_anchor_rate = std::max(0.0f, bridge_anchor_rate);
    relink.bridge_h_lo = std::max(0.0f, bridge_h_lo);
    relink.bridge_h_hi = std::max(0.0f, bridge_h_hi);
    relink.bridge_dir_bonus = std::max(0.0f, bridge_dir_bonus);
    relink.occ_gate_cover = std::clamp(occ_gate_cover, 0.0f, 1.0f);
    relink.occ_gap_min = std::max(1, occ_gap_min);
    relink.occ_expand_px = std::max(0.0f, occ_expand_px);
    relink.occ_expand_cover = std::clamp(occ_expand_cover, 0.0f, 1.0f);
    relink.bridge_app_veto = std::min(bridge_app_veto, 1.0f);  // <= -1 disables
}

void TrackerParams::set_unified_score_params(const UnifiedScoreParams& params) {
    unified = params;
}

void TrackerParams::set_frame_size(int w, int h) {
    frame.w = w;
    frame.h = h;
}

void TrackerParams::set_quality_params(bool enabled, float w_aspect, float w_center,
                                       float w_area) {
    quality.enabled = enabled;
    quality.w_aspect = w_aspect;
    quality.w_center = w_center;
    quality.w_area = w_area;
}

void TrackerParams::set_oao_params(float tau, float contest_thresh, float score_w,
                                   int occ_mode, float crowd_radius, float height_gate,
                                   float foot_gate, float ramp_frames) {
    oao.tau = std::clamp(tau, 0.0f, 1.0f);
    // contest_thresh < 0 keeps plain OAO (bit-exact); clamp the active range.
    oao.contest_thresh = (contest_thresh < 0.0f) ? -1.0f : std::clamp(contest_thresh, 0.0f, 1.0f);
    // score_w <= 0 → no score weighting (full penalty); the kernel tests
    // `<= 0` (oao_score_scale), so a non-positive value is stored as given
    // and the readback equals what the caller set. Only the active range is
    // clamped to (0, 1]. Before PR-4b this was clamp(score_w, 0, 1), which
    // turned the resolved -1 into 0 with the same kernel behaviour.
    oao.score_w = score_w <= 0.0f ? score_w : std::min(score_w, 1.0f);
    oao.occ_mode = (occ_mode == 1) ? 1 : 0;
    // crowd_radius <= 0 → off (multiplier 1); otherwise radius in units of box height.
    oao.crowd_radius = std::max(0.0f, crowd_radius);
    // height_gate <= 0 → off; otherwise relative height-diff tolerance for same-depth.
    oao.height_gate = std::max(0.0f, height_gate);
    // foot_gate <= 0 → off; otherwise relative foot-line gap tolerance for same-depth.
    oao.foot_gate = std::max(0.0f, foot_gate);
    // ramp_frames <= 0 → off; otherwise frames to ramp the penalty from 0 to full.
    oao.ramp_frames = std::max(0.0f, ramp_frames);
}

void TrackerParams::set_occ_params(bool enabled, float iou_thresh, float foot_gap, int ttl,
                                   float cost_weight) {
    occ.enabled = enabled;
    occ.iou_thresh = std::clamp(iou_thresh, 0.0f, 1.0f);
    occ.foot_gap = std::max(0.0f, foot_gap);
    occ.ttl = std::max(1, ttl);
    occ.cost_weight = std::max(0.0f, cost_weight);
}

void TrackerParams::set_multiplicative_cost(bool enabled) {
    multiplicative_cost = enabled;
}

void TrackerParams::set_sinkhorn_lambda(float lambda) {
    sinkhorn_lambda = std::max(1.0f, lambda);
}

void TrackerParams::set_stability_cost_w(float w) {
    stability_cost_w = w;
}

void TrackerParams::set_association_energy_params(bool enabled, float score_cost_w,
                                                  float height_cost_w) {
    association_energy.enabled = enabled;
    association_energy.score_cost_w = std::max(0.0f, score_cost_w);
    association_energy.height_cost_w = std::max(0.0f, height_cost_w);
}

void TrackerParams::set_homography(const float* h) {
    if (h == nullptr) {
        homography.reset();
        return;
    }
    std::array<float, 9> m{};
    std::copy(h, h + 9, m.begin());
    homography = m;
}

void TrackerParams::set_hatch_params(const Hatch& h) {
    hatch = h;
}

}  // namespace saccade
