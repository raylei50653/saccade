#pragma once

// Configuration of PerceptionPipeline and GMC, and their read-only snapshots
// (#465 Phase B PR-4b). CUDA-free so the shipping loader tests can build the
// expected values on a stock CI runner. Nothing here reads the environment;
// tracking/legacy_env.hpp resolves the legacy SACCADE_* hatches.

#include <cstdint>
#include <string>

namespace saccade {

// How filter_detections_cuda compacts the kept detections.
enum class FilterCompactionMode {
    // Default: keep-mask kernel + CUB exclusive scan; stable order.
    kStableScan = 0,
    // One-thread stable kernel (legacy SACCADE_DETERMINISTIC_FILTER_COMPACTION).
    kSerialStable = 1,
    // atomicAdd compaction, order not stable (legacy SACCADE_ATOMIC_FILTER_BASELINE).
    kAtomicBaseline = 2,
};

inline const char* to_string(FilterCompactionMode mode) {
    switch (mode) {
        case FilterCompactionMode::kStableScan: return "stable_scan";
        case FilterCompactionMode::kSerialStable: return "serial_stable";
        case FilterCompactionMode::kAtomicBaseline: return "atomic_baseline";
    }
    return "unknown";
}

// The legacy env precedence: deterministic wins over atomic.
inline FilterCompactionMode filter_compaction_mode(bool deterministic, bool atomic_baseline) {
    if (deterministic) return FilterCompactionMode::kSerialStable;
    if (atomic_baseline) return FilterCompactionMode::kAtomicBaseline;
    return FilterCompactionMode::kStableScan;
}

// PerceptionPipeline::Config (the class keeps that name as an alias).
struct PerceptionPipelineConfig {
    float score_threshold       = 0.05f;
    int   person_class          = 0;
    bool  person_only           = true;
    float nms_threshold         = 0.50f;
    bool  person_geometry_prior = true;
    bool  geometry_suspect_support = true;
    float geometry_suspect_support_score = 0.25f;
    float person_min_height_ratio = 0.018f;
    float person_min_aspect       = 1.0f;
    float person_max_aspect       = 5.5f;
    float person_min_area_ratio   = 0.00006f;
    float person_max_area_ratio   = 0.0f;
    int   max_detections          = 2048;
    bool  private_continuation_enabled = false;
    float private_candidate_nms_iou = 0.70f;
    float private_min_score = 0.25f;
    int   private_max_candidates = 0;
    float private_prior_iou_threshold = 0.0f;
    float private_prior_center_threshold = 0.0f;
    bool  private_low_stage_only = false;
    float private_track_thresh = 0.05f;
    float private_mid_thresh = 0.10f;
    float private_new_track_thresh = 0.35f;
    float private_score_eps = 1e-4f;

    // Keys are the pybind field names (= the resolved JSON's
    // native_params.PerceptionPipelineConfig.fields keys).
    template <class V> void visit(V&& v) const {
        v("score_threshold", score_threshold);
        v("person_class", person_class);
        v("person_only", person_only);
        v("nms_threshold", nms_threshold);
        v("person_geometry_prior", person_geometry_prior);
        v("geometry_suspect_support", geometry_suspect_support);
        v("geometry_suspect_support_score", geometry_suspect_support_score);
        v("person_min_height_ratio", person_min_height_ratio);
        v("person_min_aspect", person_min_aspect);
        v("person_max_aspect", person_max_aspect);
        v("person_min_area_ratio", person_min_area_ratio);
        v("person_max_area_ratio", person_max_area_ratio);
        v("max_detections", max_detections);
        v("private_continuation_enabled", private_continuation_enabled);
        v("private_candidate_nms_iou", private_candidate_nms_iou);
        v("private_min_score", private_min_score);
        v("private_max_candidates", private_max_candidates);
        v("private_prior_iou_threshold", private_prior_iou_threshold);
        v("private_prior_center_threshold", private_prior_center_threshold);
        v("private_low_stage_only", private_low_stage_only);
        v("private_track_thresh", private_track_thresh);
        v("private_mid_thresh", private_mid_thresh);
        v("private_new_track_thresh", private_new_track_thresh);
        v("private_score_eps", private_score_eps);
    }
};

// Read-only view of everything that configures one PerceptionPipeline.
struct PerceptionPipelineSnapshot {
    std::uintptr_t reid_ptr = 0;     // constructor (0 = no ReID extractor)
    std::uintptr_t cropper_ptr = 0;  // constructor (0 = no cropper)
    PerceptionPipelineConfig config;
    bool postprocess_profiling_enabled = false;
    FilterCompactionMode filter_compaction = FilterCompactionMode::kStableScan;
    bool private_workload_stats_enabled = false;  // diagnostic (legacy SACCADE_ASSOC_STATS)

    // v(key, field); config fields are keyed `constructor.config.<field>`.
    template <class V> void visit(V&& v) const {
        v(std::string("constructor.reid_ptr"), reid_ptr);
        v(std::string("constructor.cropper_ptr"), cropper_ptr);
        config.visit([&](const char* key, const auto& field) {
            v(std::string("constructor.config.") + key, field);
        });
        v(std::string("set_postprocess_profiling_enabled.enabled"), postprocess_profiling_enabled);
        v(std::string("filter_compaction"), filter_compaction);
        v(std::string("diagnostic.private_workload_stats"), private_workload_stats_enabled);
    }
};

// Read-only view of everything that configures one GMC.
struct GmcSnapshot {
    int downscale = 0;
    int max_corners = 0;
    float quality_level = 0.0f;
    float min_distance = 0.0f;
    int min_inliers = 0;
    float ransac_threshold = 0.0f;
    float pcr_thresh = 0.0f;  // phase-correlation accept threshold (legacy SACCADE_GMC_PCR_THRESH)
    bool profiling_enabled = false;

    template <class V> void visit(V&& v) const {
        v(std::string("constructor.downscale"), downscale);
        v(std::string("constructor.max_corners"), max_corners);
        v(std::string("constructor.quality_level"), quality_level);
        v(std::string("constructor.min_distance"), min_distance);
        v(std::string("constructor.min_inliers"), min_inliers);
        v(std::string("constructor.ransac_threshold"), ransac_threshold);
        v(std::string("set_profiling_enabled.enabled"), profiling_enabled);
        v(std::string("native_env.SACCADE_GMC_PCR_THRESH"), pcr_thresh);
    }
};

}  // namespace saccade
