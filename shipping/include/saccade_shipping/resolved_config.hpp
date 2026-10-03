// Typed shipping runtime config: strict loader for
// configs/shipping/mamba_whole_graph.resolved.json (#465 Phase B PR-4a, U2b-a).
//
// The JSON (written by scripts/model/export_resolved_shipping_config.py, PR-3)
// is the only source of every value here. The loader:
//   * requires `schema` == kResolvedConfigSchema;
//   * requires every field the schema declares and rejects every key it does
//     not declare, at every level;
//   * checks each value's JSON type (int vs float literal included), its enum
//     domain or range, and finiteness;
//     Numeric ranges are shipping-admissibility guards (fail-closed ABI policy),
//     not the accepted domain of the legacy native setters, which clamp or
//     canonicalize many inputs; PR-4b may tighten them where exact readback
//     needs a canonical domain.
//   * never reads the process environment (`SACCADE_*` or otherwise) and has no
//     fallback/default values: a field is either in the JSON or the load fails.
//
// Each struct's `visit` is the one schema declaration; the same walk parses
// JSON into the struct and serializes it back, so JSON -> typed -> snapshot
// cannot disagree about field names, order or types. The structs are plain
// aggregates for read-only consumers; their default-constructed values mean
// nothing, and a ResolvedShippingConfig can only be obtained from the loader.
//
// Scope (PR-4a): parsing only. Nothing here calls a tracker/GMC/pipeline
// setter or claims that native code consumes this config; that is PR-4b.
#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "saccade_shipping/strict_json.hpp"

namespace saccade::shipping {

inline constexpr std::string_view kResolvedConfigSchema = "saccade.resolved_shipping_config/v1";

// ─── leaf types ───────────────────────────────────────────────────────────

struct NullValue {};  // the field exists and must be JSON null
using IntList = std::vector<std::int64_t>;
using FloatList = std::vector<double>;
using BoolList = std::vector<bool>;
using StringList = std::vector<std::string>;

// {"per_sequence": "seqinfo.ini:Sequence.imWidth"}: a value the host reads per
// sequence, not a constant.
enum class PerSequenceValue { ImWidth, ImHeight };
using PerSequenceList = std::vector<PerSequenceValue>;

// A native `SACCADE_*` getenv with no default: {"unset_effect": "..."}.
struct UnsetEnv {
    std::string unset_effect;
    template <class V> void visit(V& v) { v.field("unset_effect", unset_effect); }
};

// Marker base for an ordered `calls` array of {"method", "args"} records. The
// method names and their order are part of the schema.
struct CallSequence {};

// ─── value checks ─────────────────────────────────────────────────────────

namespace check {
struct Any {};
struct FloatRange {
    double lo;
    double hi;
    bool lo_open = false;
    bool hi_open = false;
};
struct IntRange {
    std::int64_t lo;
    std::int64_t hi;
};
struct IntOneOf {
    std::vector<std::int64_t> values;
};
struct StringOneOf {
    std::vector<std::string> values;
};
struct NonEmpty {};
struct Sha256Hex {};
struct BoolIs {
    bool value;
};
struct PerSequenceIs {
    PerSequenceValue value;
};
// Exact sequence of per-sequence references.
struct PerSequenceListIs {
    PerSequenceList values;
};
// Every element is one of `values` and each appears exactly once.
struct PermutationOf {
    std::vector<std::string> values;
};
struct FloatListOf {
    std::size_t size;
    FloatRange element;
};
struct BoolListSize {
    std::size_t size;
};
struct IntListSize {
    std::size_t size;
    IntRange element;
};
}  // namespace check

inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr std::int64_t kInt32Max = std::numeric_limits<std::int32_t>::max();

inline check::FloatRange unit() { return {0.0, 1.0}; }
inline check::FloatRange signed_unit() { return {-1.0, 1.0}; }
inline check::FloatRange non_negative() { return {0.0, kInf}; }
inline check::FloatRange positive() { return {0.0, kInf, true}; }
inline check::FloatRange at_least(double lo) { return {lo, kInf}; }
inline check::IntRange count() { return {0, kInt32Max}; }      // native int, >= 0
inline check::IntRange count_pos() { return {1, kInt32Max}; }  // native int, >= 1

// ─── native_params ────────────────────────────────────────────────────────

struct EmptyArgs {
    template <class V> void visit(V&) {}
};

struct EnabledArgs {
    bool enabled;
    template <class V> void visit(V& v) { v.field("enabled", enabled); }
};

// PerceptionPipelineConfig: the 24 fields the oracle sets on the config object.
struct PerceptionPipelineConfigFields {
    double score_threshold;
    std::int64_t person_class;
    bool person_only;
    double nms_threshold;
    bool person_geometry_prior;
    bool geometry_suspect_support;
    double geometry_suspect_support_score;
    double person_min_height_ratio;
    double person_min_aspect;
    double person_max_aspect;
    double person_min_area_ratio;
    double person_max_area_ratio;
    std::int64_t max_detections;
    bool private_continuation_enabled;
    double private_candidate_nms_iou;
    double private_min_score;
    std::int64_t private_max_candidates;
    double private_prior_iou_threshold;
    double private_prior_center_threshold;
    bool private_low_stage_only;
    double private_track_thresh;
    double private_mid_thresh;
    double private_new_track_thresh;
    double private_score_eps;

    template <class V> void visit(V& v) {
        v.field("score_threshold", score_threshold, unit());
        v.field("person_class", person_class, count());
        v.field("person_only", person_only);
        v.field("nms_threshold", nms_threshold, unit());
        v.field("person_geometry_prior", person_geometry_prior);
        v.field("geometry_suspect_support", geometry_suspect_support);
        v.field("geometry_suspect_support_score", geometry_suspect_support_score, unit());
        v.field("person_min_height_ratio", person_min_height_ratio, unit());
        v.field("person_min_aspect", person_min_aspect, non_negative());
        v.field("person_max_aspect", person_max_aspect, non_negative());
        v.field("person_min_area_ratio", person_min_area_ratio, unit());
        v.field("person_max_area_ratio", person_max_area_ratio, unit());
        v.field("max_detections", max_detections, count_pos());
        v.field("private_continuation_enabled", private_continuation_enabled);
        v.field("private_candidate_nms_iou", private_candidate_nms_iou, unit());
        v.field("private_min_score", private_min_score, unit());
        v.field("private_max_candidates", private_max_candidates, count());
        v.field("private_prior_iou_threshold", private_prior_iou_threshold, unit());
        v.field("private_prior_center_threshold", private_prior_center_threshold, non_negative());
        v.field("private_low_stage_only", private_low_stage_only);
        v.field("private_track_thresh", private_track_thresh, unit());
        v.field("private_mid_thresh", private_mid_thresh, unit());
        v.field("private_new_track_thresh", private_new_track_thresh, unit());
        v.field("private_score_eps", private_score_eps, non_negative());
    }
};

struct PerceptionPipelineConfigParams {
    EmptyArgs constructor;
    PerceptionPipelineConfigFields fields;
    template <class V> void visit(V& v) {
        v.field("constructor", constructor);
        v.field("fields", fields);
    }
};

struct ObjectRef {
    std::string ref;
    template <class V> void visit(V& v) {
        v.field("ref", ref, check::StringOneOf{{"PerceptionPipelineConfig"}});
    }
};

struct PerceptionPipelineParams {
    struct Constructor {
        std::int64_t reid_ptr;     // shipping has no ReID extractor: must be 0
        std::int64_t cropper_ptr;  // nor a cropper: must be 0
        ObjectRef config;
        template <class V> void visit(V& v) {
            v.field("reid_ptr", reid_ptr, check::IntOneOf{{0}});
            v.field("cropper_ptr", cropper_ptr, check::IntOneOf{{0}});
            v.field("config", config);
        }
    };
    struct Calls : CallSequence {
        EnabledArgs set_postprocess_profiling_enabled;
        template <class V> void visit(V& v) {
            v.call("set_postprocess_profiling_enabled", set_postprocess_profiling_enabled);
        }
    };
    Constructor constructor;
    Calls calls;
    template <class V> void visit(V& v) {
        v.field("constructor", constructor);
        v.field("calls", calls);
    }
};

struct TrackerRuntimeParams {
    struct Constructor {
        std::int64_t max_objects;
        std::int64_t embedding_dim;
        std::int64_t max_assoc;
        template <class V> void visit(V& v) {
            v.field("max_objects", max_objects, count_pos());
            v.field("embedding_dim", embedding_dim, count_pos());
            v.field("max_assoc", max_assoc, count_pos());
        }
    };
    struct HomographyArgs {
        NullValue h;  // shipping has no homography; any matrix is rejected
        template <class V> void visit(V& v) { v.field("h", h); }
    };
    struct ReidArgs {
        double cos_threshold, iou_low, iou_high, weight, cost_cos_w, cost_iou_w, cost_score_w;
        template <class V> void visit(V& v) {
            v.field("cos_threshold", cos_threshold, signed_unit());
            v.field("iou_low", iou_low, unit());
            v.field("iou_high", iou_high, unit());
            v.field("weight", weight, non_negative());
            v.field("cost_cos_w", cost_cos_w, non_negative());
            v.field("cost_iou_w", cost_iou_w, non_negative());
            v.field("cost_score_w", cost_score_w, non_negative());
        }
    };
    struct RelinkArgs {
        bool enabled;
        std::int64_t bank_cap;
        double sim_thresh, cheb_lambda, spatial_gate;
        std::int64_t max_age;
        bool bidirectional;
        double bridge_px;
        std::int64_t bridge_at, bridge_min_lost, bridge_ttl;
        double bridge_max_speed, bridge_person_height, bridge_fps, bridge_margin,
            bridge_spatial_gate;
        std::int64_t bridge_anchor;
        double bridge_anchor_rate, bridge_h_lo, bridge_h_hi, bridge_dir_bonus, occ_gate_cover;
        std::int64_t occ_gap_min;
        double occ_expand_px, occ_expand_cover, bridge_app_veto;
        template <class V> void visit(V& v) {
            v.field("enabled", enabled);
            v.field("bank_cap", bank_cap, count_pos());
            v.field("sim_thresh", sim_thresh, signed_unit());
            v.field("cheb_lambda", cheb_lambda, non_negative());
            v.field("spatial_gate", spatial_gate, non_negative());
            v.field("max_age", max_age, count());
            v.field("bidirectional", bidirectional);
            v.field("bridge_px", bridge_px, non_negative());
            v.field("bridge_at", bridge_at, count_pos());
            v.field("bridge_min_lost", bridge_min_lost, count());
            v.field("bridge_ttl", bridge_ttl, count());
            v.field("bridge_max_speed", bridge_max_speed, non_negative());
            v.field("bridge_person_height", bridge_person_height, positive());
            v.field("bridge_fps", bridge_fps, positive());
            v.field("bridge_margin", bridge_margin, non_negative());
            v.field("bridge_spatial_gate", bridge_spatial_gate, non_negative());
            // tracker_gpu.cu: 0 = centre, 1 = foot, 2 = adaptive edges
            v.field("bridge_anchor", bridge_anchor, check::IntOneOf{{0, 1, 2}});
            v.field("bridge_anchor_rate", bridge_anchor_rate, non_negative());
            v.field("bridge_h_lo", bridge_h_lo, positive());
            v.field("bridge_h_hi", bridge_h_hi, positive());
            v.field("bridge_dir_bonus", bridge_dir_bonus, non_negative());
            v.field("occ_gate_cover", occ_gate_cover, unit());
            v.field("occ_gap_min", occ_gap_min, count());
            v.field("occ_expand_px", occ_expand_px, non_negative());
            v.field("occ_expand_cover", occ_expand_cover, unit());
            v.field("bridge_app_veto", bridge_app_veto, signed_unit());  // -1 = off
        }
    };
    struct UnifiedScoreArgs {
        struct Params {
            double w_sim_base, w_iou_base, w_maha_base, shift_ambiguity, shift_lost_age;
            template <class V> void visit(V& v) {
                v.field("w_sim_base", w_sim_base, non_negative());
                v.field("w_iou_base", w_iou_base, non_negative());
                v.field("w_maha_base", w_maha_base, non_negative());
                v.field("shift_ambiguity", shift_ambiguity, non_negative());
                v.field("shift_lost_age", shift_lost_age, non_negative());
            }
        };
        Params params;
        template <class V> void visit(V& v) { v.field("params", params); }
    };
    struct FrameSizeArgs {
        PerSequenceValue w, h;
        template <class V> void visit(V& v) {
            v.field("w", w, check::PerSequenceIs{PerSequenceValue::ImWidth});
            v.field("h", h, check::PerSequenceIs{PerSequenceValue::ImHeight});
        }
    };
    struct QualityArgs {
        bool enabled;
        double w_aspect, w_center, w_area;
        template <class V> void visit(V& v) {
            v.field("enabled", enabled);
            v.field("w_aspect", w_aspect, non_negative());
            v.field("w_center", w_center, non_negative());
            v.field("w_area", w_area, non_negative());
        }
    };
    struct Params {
        double track_thresh, high_thresh, match_thresh;
        std::int64_t track_buffer;
        double mid_thresh;
        std::int64_t confirm_streak;
        double confirm_score_thresh;
        bool adaptive_confirmation;
        double new_track_thresh;
        std::int64_t kalman_adapt_mode;
        double r_scale, vel_dir_weight, fuse_score_weight, stage2_match_thresh,
            birth_low_score_thresh, birth_prox_norm_thresh;
        template <class V> void visit(V& v) {
            v.field("track_thresh", track_thresh, unit());
            v.field("high_thresh", high_thresh, unit());
            v.field("match_thresh", match_thresh, unit());
            v.field("track_buffer", track_buffer, count());
            v.field("mid_thresh", mid_thresh, unit());
            v.field("confirm_streak", confirm_streak, count());
            v.field("confirm_score_thresh", confirm_score_thresh, unit());
            v.field("adaptive_confirmation", adaptive_confirmation);
            v.field("new_track_thresh", new_track_thresh, unit());
            // tracker_gpu.cu adapt_measurement_noise: modes 0..4
            v.field("kalman_adapt_mode", kalman_adapt_mode, check::IntOneOf{{0, 1, 2, 3, 4}});
            v.field("r_scale", r_scale, positive());
            v.field("vel_dir_weight", vel_dir_weight, non_negative());
            v.field("fuse_score_weight", fuse_score_weight, unit());
            v.field("stage2_match_thresh", stage2_match_thresh, unit());
            v.field("birth_low_score_thresh", birth_low_score_thresh, unit());
            v.field("birth_prox_norm_thresh", birth_prox_norm_thresh, non_negative());
        }
    };
    struct OaoArgs {
        double tau, contest_thresh, score_w;
        std::int64_t occ_mode;
        double crowd_radius, height_gate, foot_gate, ramp_frames;
        template <class V> void visit(V& v) {
            v.field("tau", tau, non_negative());
            v.field("contest_thresh", contest_thresh, at_least(-1.0));  // -1 = off
            v.field("score_w", score_w, at_least(-1.0));                // -1 = off
            v.field("occ_mode", occ_mode, check::IntOneOf{{0, 1}});     // 1 = union
            v.field("crowd_radius", crowd_radius, non_negative());
            v.field("height_gate", height_gate, non_negative());
            v.field("foot_gate", foot_gate, non_negative());
            v.field("ramp_frames", ramp_frames, non_negative());
        }
    };
    struct OccArgs {
        bool enabled;
        double iou_thresh, foot_gap;
        std::int64_t ttl;
        double cost_weight;
        template <class V> void visit(V& v) {
            v.field("enabled", enabled);
            v.field("iou_thresh", iou_thresh, unit());
            v.field("foot_gap", foot_gap, non_negative());
            v.field("ttl", ttl, count());
            v.field("cost_weight", cost_weight, non_negative());
        }
    };
    struct SinkhornArgs {
        double lambda;
        template <class V> void visit(V& v) { v.field("lambda", lambda, at_least(1.0)); }
    };
    struct StabilityArgs {
        double w;
        template <class V> void visit(V& v) { v.field("w", w, non_negative()); }
    };
    struct AssociationEnergyArgs {
        bool enabled;
        double score_cost_w, height_cost_w;
        template <class V> void visit(V& v) {
            v.field("enabled", enabled);
            v.field("score_cost_w", score_cost_w, non_negative());
            v.field("height_cost_w", height_cost_w, non_negative());
        }
    };
    // The oracle's setter order (EvalPipeline.__init__); later setters may
    // depend on earlier ones, so the order is schema.
    struct Calls : CallSequence {
        HomographyArgs set_homography;
        ReidArgs set_reid_params;
        RelinkArgs set_relink_params;
        UnifiedScoreArgs set_unified_score_params;
        FrameSizeArgs set_frame_size;
        QualityArgs set_quality_params;
        Params set_params;
        OaoArgs set_oao_params;
        OccArgs set_occ_params;
        EnabledArgs set_multiplicative_cost;
        SinkhornArgs set_sinkhorn_lambda;
        StabilityArgs set_stability_cost_w;
        AssociationEnergyArgs set_association_energy_params;
        template <class V> void visit(V& v) {
            v.call("set_homography", set_homography);
            v.call("set_reid_params", set_reid_params);
            v.call("set_relink_params", set_relink_params);
            v.call("set_unified_score_params", set_unified_score_params);
            v.call("set_frame_size", set_frame_size);
            v.call("set_quality_params", set_quality_params);
            v.call("set_params", set_params);
            v.call("set_oao_params", set_oao_params);
            v.call("set_occ_params", set_occ_params);
            v.call("set_multiplicative_cost", set_multiplicative_cost);
            v.call("set_sinkhorn_lambda", set_sinkhorn_lambda);
            v.call("set_stability_cost_w", set_stability_cost_w);
            v.call("set_association_energy_params", set_association_energy_params);
        }
    };
    Constructor constructor;
    Calls calls;
    template <class V> void visit(V& v) {
        v.field("constructor", constructor);
        v.field("calls", calls);
    }
};

struct GmcRuntimeParams {
    struct Constructor {
        std::int64_t downscale, max_corners;
        double quality_level, min_distance;
        std::int64_t min_inliers;
        double ransac_threshold;
        template <class V> void visit(V& v) {
            v.field("downscale", downscale, count_pos());
            v.field("max_corners", max_corners, count_pos());
            v.field("quality_level", quality_level, check::FloatRange{0.0, 1.0, true});
            v.field("min_distance", min_distance, non_negative());
            v.field("min_inliers", min_inliers, count());
            v.field("ransac_threshold", ransac_threshold, positive());
        }
    };
    struct Calls : CallSequence {
        EnabledArgs set_profiling_enabled;
        template <class V> void visit(V& v) { v.call("set_profiling_enabled", set_profiling_enabled); }
    };
    Constructor constructor;
    Calls calls;
    template <class V> void visit(V& v) {
        v.field("constructor", constructor);
        v.field("calls", calls);
    }
};

struct NativeParams {
    PerceptionPipelineConfigParams perception_pipeline_config;
    PerceptionPipelineParams perception_pipeline;
    TrackerRuntimeParams tracker;
    GmcRuntimeParams gmc;
    template <class V> void visit(V& v) {
        v.field("PerceptionPipelineConfig", perception_pipeline_config);
        v.field("PerceptionPipeline", perception_pipeline);
        v.field("GPUByteTracker", tracker);
        v.field("GMC", gmc);
    }
};

// ─── native_env: every native SACCADE_* getenv, as an explicit value ─────────

struct NativeEnvParams {
    UnsetEnv assoc_dump;
    UnsetEnv assoc_stats;
    bool atomic_filter_baseline;
    double coast_max_age;
    double coast_occ_thresh;
    double coast_score_decay;
    double dda_max_cost;
    bool deterministic_filter_compaction;
    bool enable_dda;
    double freshness_w;
    double gate_adapt_r_mult;
    double gmc_pcr_thresh;
    UnsetEnv ho_debug_level;
    UnsetEnv kalman_adapt_mode;
    double occ_vel_damp;
    double occ_vel_occ_thresh;
    bool output_measurement;
    double stability_w;

    template <class V> void visit(V& v) {
        v.field("SACCADE_ASSOC_DUMP", assoc_dump);
        v.field("SACCADE_ASSOC_STATS", assoc_stats);
        v.field("SACCADE_ATOMIC_FILTER_BASELINE", atomic_filter_baseline);
        v.field("SACCADE_COAST_MAX_AGE", coast_max_age, non_negative());
        v.field("SACCADE_COAST_OCC_THRESH", coast_occ_thresh, unit());
        v.field("SACCADE_COAST_SCORE_DECAY", coast_score_decay, unit());
        v.field("SACCADE_DDA_MAX_COST", dda_max_cost, non_negative());
        v.field("SACCADE_DETERMINISTIC_FILTER_COMPACTION", deterministic_filter_compaction);
        v.field("SACCADE_ENABLE_DDA", enable_dda);
        v.field("SACCADE_FRESHNESS_W", freshness_w, non_negative());
        v.field("SACCADE_GATE_ADAPT_R_MULT", gate_adapt_r_mult, positive());
        v.field("SACCADE_GMC_PCR_THRESH", gmc_pcr_thresh, non_negative());
        v.field("SACCADE_HO_DEBUG_LEVEL", ho_debug_level);
        v.field("SACCADE_KALMAN_ADAPT_MODE", kalman_adapt_mode);
        v.field("SACCADE_OCC_VEL_DAMP", occ_vel_damp, unit());
        v.field("SACCADE_OCC_VEL_OCC_THRESH", occ_vel_occ_thresh, unit());
        v.field("SACCADE_OUTPUT_MEASUREMENT", output_measurement);
        v.field("SACCADE_STABILITY_W", stability_w, non_negative());
    }
};

// ─── host_params ──────────────────────────────────────────────────────────

// Gate of every conditional host step. Steps the boundary doc (§5 B2/B4)
// freezes as off in the shipping tail are pinned to false: shipping has no
// implementation for them.
struct HostSteps {
    bool ingest_gpu_decode, ingest_nv12_buffer, schedule_double_buffer, post_native_postprocess,
        post_private_continuation, post_onms_priors, post_detection_quality,
        filter_crowd_low_score, filter_external_fp, filter_fp_hard, filter_duplicate_suppression,
        filter_detection_cap, birth_consecutive_gate, birth_quality_gate, birth_multi_birth,
        reid_work, gmc, gmc_fg_mask, relink_semantic_relinker, relink_native_bridge,
        track_graphed_update, emit_pipeline_relink, tail_cheb_gr_or_occ_audit,
        tail_post_lifecycle_merge, tail_deferred_alias, tail_tracklet_quality_filter,
        tail_interpolation, tail_write_output;

    template <class V> void visit(V& v) {
        const check::BoolIs off{false};
        v.field("ingest.gpu_decode", ingest_gpu_decode);
        v.field("ingest.nv12_buffer", ingest_nv12_buffer);
        v.field("schedule.double_buffer", schedule_double_buffer);
        v.field("post.native_postprocess", post_native_postprocess);
        v.field("post.private_continuation", post_private_continuation);
        v.field("post.onms_priors", post_onms_priors);
        v.field("post.detection_quality", post_detection_quality);
        v.field("filter.crowd_low_score", filter_crowd_low_score);
        v.field("filter.external_fp", filter_external_fp);
        v.field("filter.fp_hard", filter_fp_hard);
        v.field("filter.duplicate_suppression", filter_duplicate_suppression);
        v.field("filter.detection_cap", filter_detection_cap);
        v.field("birth.consecutive_gate", birth_consecutive_gate);
        v.field("birth.quality_gate", birth_quality_gate);
        v.field("birth.multi_birth", birth_multi_birth);
        v.field("reid.work", reid_work);
        v.field("gmc", gmc);
        v.field("gmc.fg_mask", gmc_fg_mask);
        v.field("relink.semantic_relinker", relink_semantic_relinker);
        v.field("relink.native_bridge", relink_native_bridge);
        v.field("track.graphed_update", track_graphed_update);
        v.field("emit.pipeline_relink", emit_pipeline_relink);
        v.field("tail.cheb_gr_or_occ_audit", tail_cheb_gr_or_occ_audit, off);
        v.field("tail.post_lifecycle_merge", tail_post_lifecycle_merge, off);
        v.field("tail.deferred_alias", tail_deferred_alias, off);
        v.field("tail.tracklet_quality_filter", tail_tracklet_quality_filter, off);
        v.field("tail.interpolation", tail_interpolation);
        v.field("tail.write_output", tail_write_output);
    }
};

struct DetectorHostConfig {
    struct Build {
        std::string yolo_pt_path, teacher_ckpt, mamba_ckpt;
        std::int64_t img_size;
        std::string device;
        double conf_thr;
        std::int64_t max_det;
        std::string trt_backbone_engine, trt_head_engine;
        NullValue temporal_T_override;
        bool use_cuda_graph, use_whole_graph;
        double small_p3_max_threshold;
        bool postprocess_compile;
        template <class V> void visit(V& v) {
            v.field("yolo_pt_path", yolo_pt_path, check::NonEmpty{});
            v.field("teacher_ckpt", teacher_ckpt, check::NonEmpty{});
            v.field("mamba_ckpt", mamba_ckpt, check::NonEmpty{});
            v.field("img_size", img_size, count_pos());
            v.field("device", device, check::StringOneOf{{"cuda"}});
            v.field("conf_thr", conf_thr, unit());
            v.field("max_det", max_det, count_pos());
            v.field("trt_backbone_engine", trt_backbone_engine, check::NonEmpty{});
            v.field("trt_head_engine", trt_head_engine);
            v.field("temporal_T_override", temporal_T_override);
            v.field("use_cuda_graph", use_cuda_graph);
            v.field("use_whole_graph", use_whole_graph);
            v.field("small_p3_max_threshold", small_p3_max_threshold, non_negative());
            v.field("postprocess_compile", postprocess_compile);
        }
    };
    struct HeadCalls {
        BoolList set_head_compile, set_block_compile;  // positional args
        template <class V> void visit(V& v) {
            v.field("set_head_compile", set_head_compile, check::BoolListSize{1});
            v.field("set_block_compile", set_block_compile, check::BoolListSize{1});
        }
    };
    struct Contract {
        std::int64_t feature_dim;
        bool fpn_reid_mode;
        std::string box_format;
        template <class V> void visit(V& v) {
            v.field("feature_dim", feature_dim, count());
            v.field("fpn_reid_mode", fpn_reid_mode);
            v.field("box_format", box_format, check::StringOneOf{{"xyxy", "cxcywh"}});
        }
    };
    struct Calls : CallSequence {
        PerSequenceList set_whole_graph_img_dims;  // positional (height, width)
        template <class V> void visit(V& v) {
            v.call("set_whole_graph_img_dims", set_whole_graph_img_dims,
                   check::PerSequenceListIs{{PerSequenceValue::ImHeight, PerSequenceValue::ImWidth}});
        }
    };
    Build build;
    HeadCalls head_calls;
    std::string detect_fn;
    Contract contract;
    Calls calls;
    template <class V> void visit(V& v) {
        v.field("build", build);
        v.field("head_calls", head_calls);
        v.field("detect_fn", detect_fn, check::NonEmpty{});
        v.field("contract", contract);
        v.field("calls", calls);
    }
};

struct HostCapacities {
    std::int64_t track_result_cap, nms_fixed_n;
    template <class V> void visit(V& v) {
        v.field("track_result_cap", track_result_cap, count_pos());
        v.field("nms_fixed_n", nms_fixed_n, count_pos());
    }
};

// RuleBaselineConfig() as run_eval builds it (boundary S5a).
struct ExternalFpRuleConfig {
    double min_score, low_score, medium_score, min_height, medium_height, min_aspect;
    template <class V> void visit(V& v) {
        v.field("min_score", min_score, unit());
        v.field("low_score", low_score, unit());
        v.field("medium_score", medium_score, unit());
        v.field("min_height", min_height, non_negative());
        v.field("medium_height", medium_height, non_negative());
        v.field("min_aspect", min_aspect, non_negative());
    }
};

struct OnmsConfig {
    bool enabled;
    double prior_iou_threshold;
    std::int64_t min_track_age;
    double min_track_score;
    template <class V> void visit(V& v) {
        v.field("enabled", enabled);
        v.field("prior_iou_threshold", prior_iou_threshold, unit());
        v.field("min_track_age", min_track_age, count());
        v.field("min_track_score", min_track_score, unit());
    }
};

// The SACCADE_* values the Python host saw (null = unset). Recorded so the
// shipping host can be checked against them; shipping itself reads no env.
struct HostEnv {
    std::optional<std::string> assoc_stats, build_path, capture_debug, detect_barrier,
        disable_cuda_scan_bwd, double_buffer, enable_onms, fixed_postbuf, flow_timing, gpu_decode,
        gpu_relink_gate, main_nms_graphed, main_nms_graphed_shadow, main_nms_graph_shadow,
        main_nms_shadow, main_nms_split, nms_graphed_eager_test, nms_logical_n, nv12_buffer,
        occ_dump, occ_log, research_bridge_fidelity_capture_capacity,
        research_bridge_fidelity_capture_dir, research_bridge_fidelity_capture_shadow,
        research_portable_or_tail_policy, research_r1_temporal_reduction_capture_dir,
        score_jitter, stream_debug, stream_mode;

    template <class V> void visit(V& v) {
        v.field("SACCADE_ASSOC_STATS", assoc_stats);
        v.field("SACCADE_BUILD_PATH", build_path);
        v.field("SACCADE_CAPTURE_DEBUG", capture_debug);
        // pipeline.py _detect_barrier_mode
        v.field("SACCADE_DETECT_BARRIER", detect_barrier,
                check::StringOneOf{{"full", "no_postproc", "event"}});
        v.field("SACCADE_DISABLE_CUDA_SCAN_BWD", disable_cuda_scan_bwd);
        v.field("SACCADE_DOUBLE_BUFFER", double_buffer);
        v.field("SACCADE_ENABLE_ONMS", enable_onms);
        v.field("SACCADE_FIXED_POSTBUF", fixed_postbuf);
        v.field("SACCADE_FLOW_TIMING", flow_timing);
        v.field("SACCADE_GPU_DECODE", gpu_decode);
        v.field("SACCADE_GPU_RELINK_GATE", gpu_relink_gate);
        v.field("SACCADE_MAIN_NMS_GRAPHED", main_nms_graphed);
        v.field("SACCADE_MAIN_NMS_GRAPHED_SHADOW", main_nms_graphed_shadow);
        v.field("SACCADE_MAIN_NMS_GRAPH_SHADOW", main_nms_graph_shadow);
        v.field("SACCADE_MAIN_NMS_SHADOW", main_nms_shadow);
        v.field("SACCADE_MAIN_NMS_SPLIT", main_nms_split);
        v.field("SACCADE_NMS_GRAPHED_EAGER_TEST", nms_graphed_eager_test);
        v.field("SACCADE_NMS_LOGICAL_N", nms_logical_n);
        v.field("SACCADE_NV12_BUFFER", nv12_buffer);
        v.field("SACCADE_OCC_DUMP", occ_dump);
        v.field("SACCADE_OCC_LOG", occ_log);
        v.field("SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_CAPACITY",
                research_bridge_fidelity_capture_capacity);
        v.field("SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_DIR", research_bridge_fidelity_capture_dir);
        v.field("SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_SHADOW",
                research_bridge_fidelity_capture_shadow);
        v.field("SACCADE_RESEARCH_PORTABLE_OR_TAIL_POLICY", research_portable_or_tail_policy);
        v.field("SACCADE_RESEARCH_R1_TEMPORAL_REDUCTION_CAPTURE_DIR",
                research_r1_temporal_reduction_capture_dir);
        v.field("SACCADE_SCORE_JITTER", score_jitter);
        v.field("SACCADE_STREAM_DEBUG", stream_debug);
        v.field("SACCADE_STREAM_MODE", stream_mode);
    }
};

// Every cfg/kwargs key the oracle host reads, typed from the generated list
// (scripts/model/render_shipping_host_cfg_schema.py). This records what the
// oracle read; whether a key has behavior is answered by `steps`. Values get
// type and finiteness checks only -- values native code consumes are checked
// where they enter native code (native_params, steps, the typed host sections).
struct HostCfg {
#define SACCADE_HOST_CFG_FIELD(member, key, type) type member;
#include "saccade_shipping/host_cfg_fields.inc"
#undef SACCADE_HOST_CFG_FIELD

    template <class V> void visit(V& v) {
#define SACCADE_HOST_CFG_FIELD(member, key, type) v.field(key, member);
#include "saccade_shipping/host_cfg_fields.inc"
#undef SACCADE_HOST_CFG_FIELD
    }
};

inline const std::vector<std::string>& host_stage_names() {
    static const std::vector<std::string> names = {
        "_run_detect",         "_run_native_tensor_prep", "_run_nms",
        "_run_post_nms_finalize", "_run_detection_filters", "_run_birth_config",
        "_run_reid_and_gmc",   "_run_track",              "_run_materialize",
        "_run_emit"};
    return names;
}

struct ShippingHostConfig {
    StringList stage_order;
    HostSteps steps;
    DetectorHostConfig detector;
    HostCapacities capacities;
    ExternalFpRuleConfig external_fp_rule_config;
    OnmsConfig onms;
    FloatList initial_tracker_thresholds;
    double fp_hard_reject_score;
    HostEnv env;
    HostCfg cfg;

    template <class V> void visit(V& v) {
        v.field("stage_order", stage_order, check::PermutationOf{host_stage_names()});
        v.field("steps", steps);
        v.field("detector", detector);
        v.field("capacities", capacities);
        v.field("external_fp_rule_config", external_fp_rule_config);
        v.field("onms", onms);
        v.field("initial_tracker_thresholds", initial_tracker_thresholds,
                check::FloatListOf{3, unit()});
        v.field("fp_hard_reject_score", fp_hard_reject_score, signed_unit());  // -1 = off
        v.field("env", env);
        v.field("cfg", cfg);
    }
};

// ─── source ───────────────────────────────────────────────────────────────

struct SourceInfo {
    struct NativeObjectsNotShipped {
        std::string dynamic_reid_controller, tracklet_lifecycle_merger;
        template <class V> void visit(V& v) {
            v.field("DynamicReIDController", dynamic_reid_controller, check::NonEmpty{});
            v.field("TrackletLifecycleMerger", tracklet_lifecycle_merger, check::NonEmpty{});
        }
    };
    struct HarnessReadsNotExported {
        std::string data_root, kwargs_detector, kwargs_sequence_result_callback,
            kwargs_stage_probe_callback, output_root, seqs, split;
        template <class V> void visit(V& v) {
            v.field("data_root", data_root, check::NonEmpty{});
            v.field("kwargs.detector", kwargs_detector, check::NonEmpty{});
            v.field("kwargs.sequence_result_callback", kwargs_sequence_result_callback,
                    check::NonEmpty{});
            v.field("kwargs.stage_probe_callback", kwargs_stage_probe_callback, check::NonEmpty{});
            v.field("output_root", output_root, check::NonEmpty{});
            v.field("seqs", seqs, check::NonEmpty{});
            v.field("split", split, check::NonEmpty{});
        }
    };
    std::string preset, preset_sha256;
    StringList oracle_argv;
    std::string exporter;
    PerSequenceList per_sequence_values;
    NativeObjectsNotShipped native_objects_not_shipped;
    HarnessReadsNotExported harness_reads_not_exported;

    template <class V> void visit(V& v) {
        v.field("preset", preset, check::NonEmpty{});
        v.field("preset_sha256", preset_sha256, check::Sha256Hex{});
        v.field("oracle_argv", oracle_argv);
        v.field("exporter", exporter, check::NonEmpty{});
        v.field("per_sequence_values", per_sequence_values,
                check::PerSequenceListIs{{PerSequenceValue::ImWidth, PerSequenceValue::ImHeight}});
        v.field("native_objects_not_shipped", native_objects_not_shipped);
        v.field("harness_reads_not_exported", harness_reads_not_exported);
    }
};

// ─── top level ────────────────────────────────────────────────────────────

class ResolvedShippingConfig {
public:
    std::string schema;
    SourceInfo source;
    NativeParams native_params;
    NativeEnvParams native_env;
    ShippingHostConfig host_params;

    template <class V> void visit(V& v) {
        v.field("schema", schema, check::StringOneOf{{std::string(kResolvedConfigSchema)}});
        v.field("source", source);
        v.field("native_params", native_params);
        v.field("native_env", native_env);
        v.field("host_params", host_params);
    }

private:
    ResolvedShippingConfig() = default;
    friend ResolvedShippingConfig load_resolved_shipping_config(const JsonValue& document);
};

// Strict load. Throws ConfigError naming the JSON path on any violation.
ResolvedShippingConfig load_resolved_shipping_config(const JsonValue& document);
ResolvedShippingConfig parse_resolved_shipping_config(std::string_view json_text);
ResolvedShippingConfig load_resolved_shipping_config_file(const std::string& path);

// Typed -> JSON. `canonical_snapshot` is the exporter's file format (indent 2,
// trailing newline); for the committed file it is byte-identical to the input.
JsonValue to_json(const ResolvedShippingConfig& config);
std::string canonical_snapshot(const ResolvedShippingConfig& config);

}  // namespace saccade::shipping
