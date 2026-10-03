// Resolved shipping config -> native tracking state, and readback
// (#465 Phase B PR-4b, U2b-b). CUDA-free: the GPU objects are built in
// native_build.hpp from the same mapping.
//
// Mapping. `apply_tracker_config` is the one place where JSON values become
// tracker setter calls. It is a template so the CUDA-free TrackerParams (the
// tracker's parameter store, tracking/tracker_params.hpp) and the real
// GPUByteTracker run the identical call sequence: `native_env` hatches first,
// then the oracle's `calls` in schema order. Values pass through the native
// setters' own canonicalization; nothing here clamps, defaults or reads the
// process environment.
//
// Readback. `expected_*_snapshot` lists, for every key of a native snapshot's
// visit(), the value the JSON says it must hold. `readback_mismatches`
// compares a snapshot to it: every snapshot key needs an expectation, every
// expectation must be consumed, floats compare bit-exactly after the
// float32 cast the setter applies. A non-empty result means the native state
// is not the resolved config (a setter canonicalized a value, a field has no
// JSON source, or a JSON value has no native consumer); the GPU builders turn
// that into a ConfigError.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "saccade_shipping/resolved_config.hpp"
#include "tracking/perception_params.hpp"
#include "tracking/tracker_params.hpp"

namespace saccade::shipping {

// Per-sequence values the resolved JSON marks `per_sequence`
// (seqinfo.ini Sequence.imWidth / Sequence.imHeight).
struct SequenceGeometry {
    int im_width = 0;
    int im_height = 0;
};

// Checked int64 -> native int. The loader already bounds every native int
// argument to int32; this keeps the narrowing explicit.
int native_int(std::int64_t value, const char* what);

inline float native_float(double value) { return static_cast<float>(value); }

TrackerParams::Hatch tracker_hatch(const NativeEnvParams& env);

template <class Tracker>
void apply_tracker_config(Tracker& t, const ResolvedShippingConfig& cfg, SequenceGeometry g) {
    if (g.im_width <= 0 || g.im_height <= 0) {
        throw ConfigError("shipping: sequence geometry must be positive (imWidth/imHeight)");
    }
    t.set_hatch_params(tracker_hatch(cfg.native_env));

    const auto& c = cfg.native_params.tracker.calls;
    t.set_homography(nullptr);  // the loader admits only `h: null`
    {
        const auto& a = c.set_reid_params;
        t.set_reid_params(native_float(a.cos_threshold), native_float(a.iou_low),
                          native_float(a.iou_high), native_float(a.weight),
                          native_float(a.cost_cos_w), native_float(a.cost_iou_w),
                          native_float(a.cost_score_w));
    }
    {
        const auto& a = c.set_relink_params;
        t.set_relink_params(
            a.enabled, native_int(a.bank_cap, "bank_cap"), native_float(a.sim_thresh),
            native_float(a.cheb_lambda), native_float(a.spatial_gate),
            native_int(a.max_age, "max_age"), a.bidirectional, native_float(a.bridge_px),
            native_int(a.bridge_at, "bridge_at"), native_int(a.bridge_min_lost, "bridge_min_lost"),
            native_int(a.bridge_ttl, "bridge_ttl"), native_float(a.bridge_max_speed),
            native_float(a.bridge_person_height), native_float(a.bridge_fps),
            native_float(a.bridge_margin), native_float(a.bridge_spatial_gate),
            native_int(a.bridge_anchor, "bridge_anchor"), native_float(a.bridge_anchor_rate),
            native_float(a.bridge_h_lo), native_float(a.bridge_h_hi),
            native_float(a.bridge_dir_bonus), native_float(a.occ_gate_cover),
            native_int(a.occ_gap_min, "occ_gap_min"), native_float(a.occ_expand_px),
            native_float(a.occ_expand_cover), native_float(a.bridge_app_veto));
    }
    {
        const auto& a = c.set_unified_score_params.params;
        UnifiedScoreParams u;
        u.w_sim_base = native_float(a.w_sim_base);
        u.w_iou_base = native_float(a.w_iou_base);
        u.w_maha_base = native_float(a.w_maha_base);
        u.shift_ambiguity = native_float(a.shift_ambiguity);
        u.shift_lost_age = native_float(a.shift_lost_age);
        t.set_unified_score_params(u);
    }
    t.set_frame_size(g.im_width, g.im_height);
    {
        const auto& a = c.set_quality_params;
        t.set_quality_params(a.enabled, native_float(a.w_aspect), native_float(a.w_center),
                             native_float(a.w_area));
    }
    {
        const auto& a = c.set_params;
        t.set_params(native_float(a.track_thresh), native_float(a.high_thresh),
                     native_float(a.match_thresh), native_int(a.track_buffer, "track_buffer"),
                     native_float(a.mid_thresh), native_int(a.confirm_streak, "confirm_streak"),
                     native_float(a.confirm_score_thresh), a.adaptive_confirmation,
                     native_float(a.new_track_thresh),
                     native_int(a.kalman_adapt_mode, "kalman_adapt_mode"),
                     native_float(a.r_scale), native_float(a.vel_dir_weight),
                     native_float(a.fuse_score_weight), native_float(a.stage2_match_thresh),
                     native_float(a.birth_low_score_thresh),
                     native_float(a.birth_prox_norm_thresh));
    }
    {
        const auto& a = c.set_oao_params;
        t.set_oao_params(native_float(a.tau), native_float(a.contest_thresh),
                         native_float(a.score_w), native_int(a.occ_mode, "occ_mode"),
                         native_float(a.crowd_radius), native_float(a.height_gate),
                         native_float(a.foot_gate), native_float(a.ramp_frames));
    }
    {
        const auto& a = c.set_occ_params;
        t.set_occ_params(a.enabled, native_float(a.iou_thresh), native_float(a.foot_gap),
                         native_int(a.ttl, "ttl"), native_float(a.cost_weight));
    }
    t.set_multiplicative_cost(c.set_multiplicative_cost.enabled);
    t.set_sinkhorn_lambda(native_float(c.set_sinkhorn_lambda.lambda));
    t.set_stability_cost_w(native_float(c.set_stability_cost_w.w));
    {
        const auto& a = c.set_association_energy_params;
        t.set_association_energy_params(a.enabled, native_float(a.score_cost_w),
                                        native_float(a.height_cost_w));
    }
}

// The tracker parameter state the config asks for, computed on the CUDA-free
// TrackerParams with the tracker's own setters.
TrackerParams planned_tracker_params(const ResolvedShippingConfig& cfg, SequenceGeometry g);
PerceptionPipelineConfig planned_pipeline_config(const ResolvedShippingConfig& cfg);
FilterCompactionMode planned_filter_compaction(const ResolvedShippingConfig& cfg);
// What the GMC / PerceptionPipeline builders construct and set, as the
// snapshot those objects must read back. native_build.cpp builds from these.
GmcSnapshot planned_gmc_snapshot(const ResolvedShippingConfig& cfg);
PerceptionPipelineSnapshot planned_pipeline_snapshot(const ResolvedShippingConfig& cfg);

// Expected value per snapshot key. Values are JSON leaves copied from the
// config (per-sequence markers resolved, `unset_effect` entries -> false), or
// one of the documented native-only expectations below.
using SnapshotExpectation = std::map<std::string, JsonValue>;
SnapshotExpectation expected_tracker_snapshot(const ResolvedShippingConfig& cfg, SequenceGeometry g);
SnapshotExpectation expected_gmc_snapshot(const ResolvedShippingConfig& cfg);
SnapshotExpectation expected_pipeline_snapshot(const ResolvedShippingConfig& cfg);

// Tracker snapshot keys the resolved JSON has no value for, with the value
// shipping requires and why. Exactly these; anything else without a JSON
// source is a readback failure.
struct NativeOnlyExpectation {
    std::string key;
    JsonValue value;
    std::string reason;
};
const std::vector<NativeOnlyExpectation>& tracker_native_only_expectations();

// Which native object consumes each `native_env` key ("GPUByteTracker",
// "GMC", "PerceptionPipeline"), or "" with the reason it has no shipping
// consumer. Covers every key of NativeEnvParams.
struct NativeEnvConsumer {
    const char* key;
    const char* consumer;
    const char* reason;  // only when consumer is ""
};
const std::vector<NativeEnvConsumer>& native_env_consumers();

std::vector<std::string> readback_mismatches(const SnapshotExpectation& expected,
                                             const TrackerSnapshot& snapshot);
std::vector<std::string> readback_mismatches(const SnapshotExpectation& expected,
                                             const GmcSnapshot& snapshot);
std::vector<std::string> readback_mismatches(const SnapshotExpectation& expected,
                                             const PerceptionPipelineSnapshot& snapshot);

// Throws ConfigError("<object> readback != resolved config: ...") listing every
// mismatch; no-op when there is none.
void require_readback(const char* object, const std::vector<std::string>& mismatches);

}  // namespace saccade::shipping
