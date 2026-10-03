// Post-detector host plan and host-side detection filters (#465 Phase B PR-5).
// See saccade_shipping/post_detector_plan.hpp.
#include "saccade_shipping/post_detector_plan.hpp"

#include <algorithm>
#include <string>

#include "saccade_shipping/native_config.hpp"

namespace saccade::shipping {
namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw ConfigError("shipping post-detector host: " + what +
                          " (the U3a host has no implementation for it)");
    }
}

}  // namespace

PostDetectorPlan plan_post_detector(const ResolvedShippingConfig& cfg) {
    const auto& hp = cfg.host_params;
    const auto& s = hp.steps;
    const auto& c = hp.cfg;

    // Gates on the post-detector path the host implements only one side of.
    // Each mirrors the oracle's own gate (exporter STEPS, or the cfg read the
    // gate evaluates where STEPS has no entry).
    require(s.post_native_postprocess, "steps.post.native_postprocess must be true");
    require(s.post_private_continuation, "steps.post.private_continuation must be true");
    require(!s.post_onms_priors && !hp.onms.enabled, "ONMS priors must be off");
    require(!s.post_detection_quality, "steps.post.detection_quality must be false");
    require(!s.filter_crowd_low_score, "steps.filter.crowd_low_score must be false");
    require(!s.filter_duplicate_suppression, "steps.filter.duplicate_suppression must be false");
    require(!s.filter_detection_cap, "steps.filter.detection_cap must be false");
    require(!s.birth_consecutive_gate && !s.birth_quality_gate && !s.birth_multi_birth,
            "birth gates must be off");
    require(!s.reid_work, "steps.reid.work must be false");
    require(!s.gmc_fg_mask, "steps.gmc.fg_mask must be false");
    require(!s.relink_semantic_relinker, "steps.relink.semantic_relinker must be false");
    require(!s.emit_pipeline_relink, "steps.emit.pipeline_relink must be false");
    require(!s.ingest_nv12_buffer, "steps.ingest.nv12_buffer must be false (GMC reads RGB)");
    require(s.track_graphed_update, "steps.track.graphed_update must be true");
    require(!s.track_workbench, "steps.track.workbench must be false (its own tracker path)");
    require(!s.post_scene_adapt && !s.post_narrow_person_bonus,
            "the narrow-person score bonus must be inactive (steps.post.scene_adapt, "
            "steps.post.narrow_person_bonus)");
    require(!s.filter_stage2_quality_gate, "steps.filter.stage2_quality_gate must be false");
    require(!s.track_score_jitter && !hp.env.score_jitter.has_value(),
            "SACCADE_SCORE_JITTER must be unset (steps.track.score_jitter)");
    if (s.filter_external_fp) {
        require(c.external_fp_filter_mode == "rule", "external FP filter mode must be \"rule\"");
        require(!(c.external_fp_penalty < 0.999), "the external FP penalty branch must be off");
    }
    // Birth config: the per-frame thresholds (no crowd mode) must be the ones
    // the tracker starts with, or the oracle would call set_params.
    const auto& init = hp.initial_tracker_thresholds;
    require(init.size() == 3 && init[0] == c.track_thresh && init[1] == c.mid_thresh &&
                init[2] == c.new_track_thresh,
            "per-frame tracker thresholds must equal initial_tracker_thresholds");

    const auto& k = cfg.native_params.tracker.constructor;
    PostDetectorPlan p{};
    p.nms_fixed_n = native_int(hp.capacities.nms_fixed_n, "capacities.nms_fixed_n");
    p.max_objects = native_int(k.max_objects, "max_objects");
    p.max_assoc = native_int(k.max_assoc, "max_assoc");
    require(p.nms_fixed_n == p.max_assoc, "capacities.nms_fixed_n must equal max_assoc");
    p.private_priors =
        c.private_prior_iou_threshold > 0.0 || c.private_prior_center_threshold > 0.0;
    p.private_prior_max_age = native_int(c.private_prior_max_age, "private_prior_max_age");
    p.external_fp = s.filter_external_fp;
    const auto& r = hp.external_fp_rule_config;
    p.external_fp_rule = ExternalFpRule{native_float(c.external_fp_max_score),
                                        native_float(r.min_score),
                                        native_float(r.low_score),
                                        native_float(r.medium_score),
                                        native_float(r.min_height),
                                        native_float(r.medium_height),
                                        native_float(r.min_aspect)};
    p.fp_hard = s.filter_fp_hard;
    p.fp_hard_filter = FpHardFilter{native_float(c.fp_hard_filter_min_score),
                                    static_cast<float>(c.fp_hard_filter_max_suspicious_area),
                                    native_float(c.fp_hard_filter_max_suspicious_score),
                                    native_float(hp.fp_hard_reject_score)};
    p.gmc = s.gmc;
    p.tracker_pre_roll = kGraphedTrackerUpdatePreRoll;
    return p;
}

namespace {

// `(hi - lo).clamp(min=1e-6)` in float32.
float span(float lo, float hi) { return std::max(hi - lo, 1e-6f); }

}  // namespace

DetectionRows apply_external_fp_rule(const DetectionRows& in, const ExternalFpRule& rule) {
    DetectionRows out;
    const std::size_t n = in.size();
    out.boxes.reserve(in.boxes.size());
    out.scores.reserve(n);
    out.classes.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float sc = in.scores[i];
        bool keep = true;
        if (sc <= rule.max_score) {
            const float* b = &in.boxes[i * 4];
            const float w = span(b[0], b[2]);
            const float h = span(b[1], b[3]);
            keep = sc >= rule.min_score;
            keep = keep && !(sc < rule.low_score && h < rule.min_height);
            keep = keep && !(sc < rule.medium_score && h < rule.medium_height &&
                             (h / w) < rule.min_aspect);
        }
        if (keep) {
            out.boxes.insert(out.boxes.end(), &in.boxes[i * 4], &in.boxes[i * 4] + 4);
            out.scores.push_back(sc);
            out.classes.push_back(in.classes[i]);
        }
    }
    return out;
}

std::vector<bool> fp_hard_reject_mask(const DetectionRows& rows, const FpHardFilter& f) {
    std::vector<bool> reject(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const float* b = &rows.boxes[i * 4];
        const float area = span(b[0], b[2]) * span(b[1], b[3]);
        const float sc = rows.scores[i];
        reject[i] = (sc < f.max_suspicious_score && area > f.max_suspicious_area) ||
                    sc < f.min_score;
    }
    return reject;
}

void apply_fp_hard_filter(DetectionRows& rows, const FpHardFilter& f) {
    const std::vector<bool> reject = fp_hard_reject_mask(rows, f);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (reject[i]) rows.scores[i] = f.reject_score;
    }
}

}  // namespace saccade::shipping
