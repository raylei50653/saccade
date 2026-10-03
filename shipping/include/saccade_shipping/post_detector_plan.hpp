// Post-detector host plan (#465 Phase B PR-5, U3a). CUDA-free.
//
// The U3a host replays the oracle's per-frame stages after the detector
// (evaluator.py `_run_frame`, serial): native tensor prep -> main NMS +
// private continuation append -> detection filters (external FP rule filter,
// FP hard filter) -> birth config -> GMC -> tracker update. `plan_post_detector`
// turns the resolved config into every value those stages read and fails
// closed (ConfigError) on any oracle gate on that path the host does not
// implement, so a config the host would run differently from the oracle is
// refused instead of approximated. Nothing here reads the process environment
// or carries a headline value of its own.
//
// The detection filters are host-side (CPU) twins of the Python tensor ops in
// detection_filters.py. They compute in float32 with every threshold cast to
// float32, as torch does for a float32 tensor against a Python scalar, so
// their results are bit-identical (tests/native/fixtures/
// shipping_detection_filters.json pins them against the Python functions).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "saccade_shipping/resolved_config.hpp"

namespace saccade::shipping {

// RuleBaselineConfig thresholds plus the low-score subset gate
// (`external_fp_max_score`), as float32.
struct ExternalFpRule {
    float max_score;
    float min_score, low_score, medium_score, min_height, medium_height, min_aspect;
};

struct FpHardFilter {
    float min_score;
    float max_suspicious_area;  // an int in the config; compared as float32
    float max_suspicious_score;
    float reject_score;  // host_params.fp_hard_reject_score (masked_fill value)
};

// GraphedTrackerUpdate runs `update_into` on its zeroed scratch inputs before
// the first real update of a sequence: once in `_warmup`, then
// `make_graphed_callables`' warm-up iterations (default 3; capture itself runs
// nothing). tests/unit/test_post_detector_host_oracle_pins.py pins this to the
// oracle source.
inline constexpr int kGraphedTrackerUpdatePreRoll = 4;

struct PostDetectorPlan {
    int nms_fixed_n;  // capacities.nms_fixed_n: main NMS input padding
    int max_objects, max_assoc;  // tracker constructor
    // Private-continuation priors: active tracks of age <= max_age, any score
    // (built only when an IoU or center prior threshold is set).
    bool private_priors;
    int private_prior_max_age;
    bool external_fp;  // steps.filter.external_fp (rule mode only)
    ExternalFpRule external_fp_rule;
    bool fp_hard;
    FpHardFilter fp_hard_filter;
    bool gmc;
    int tracker_pre_roll;  // empty updates before a sequence's first real update
};

PostDetectorPlan plan_post_detector(const ResolvedShippingConfig& cfg);

// Host-side detection rows: boxes xyxy [n*4], scores [n], classes [n].
struct DetectionRows {
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<std::int32_t> classes;
    std::size_t size() const { return scores.size(); }
};

// detection_filters.py `_apply_external_fp_filter`, mode "rule", penalty off:
// rows with score <= max_score are kept only if they pass the rule; the rest
// are kept; survivors stay in order, scores unchanged.
DetectionRows apply_external_fp_rule(const DetectionRows& in, const ExternalFpRule& rule);

// detection_filters.py `_fp_hard_reject_mask` + `masked_fill`: rejected rows
// keep their place, their score becomes reject_score.
void apply_fp_hard_filter(DetectionRows& rows, const FpHardFilter& f);
std::vector<bool> fp_hard_reject_mask(const DetectionRows& rows, const FpHardFilter& f);

}  // namespace saccade::shipping
