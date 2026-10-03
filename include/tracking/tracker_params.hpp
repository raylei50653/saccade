#pragma once

// GPUByteTracker runtime parameters: the single authority (#465 Phase B PR-4b).
//
// Every value the tracker's kernels and host update path read is a field of
// TrackerParams. GPUByteTracker holds exactly one instance (`params_`); its
// setters write it through the member functions below, update() launches
// kernels from it, and snapshot() returns a copy of it. There is no second
// copy to drift from and no per-field getter.
//
// Two front-ends feed this one state:
//   * legacy (pybind eval harness, seq_runner): env hatches are resolved by
//     tracking/legacy_env.hpp and passed to set_hatch_params(); set_* receive
//     the Python-resolved values;
//   * shipping (shipping/): every value comes from the resolved JSON
//     (configs/shipping/*.resolved.json), applied through the same setters.
//
// Nothing in this header or its .cpp reads the process environment.
//
// CUDA-free on purpose: the shipping loader tests apply the resolved JSON to a
// TrackerParams and check readback on a stock (no-GPU) CI runner.

#include <array>
#include <optional>

namespace saccade {

struct UnifiedScoreParams {
    float w_sim_base = 0.0f;
    float w_iou_base = 0.0f;
    float w_maha_base = 0.0f;
    float shift_ambiguity = 0.0f;
    float shift_lost_age = 0.0f;
};

struct TrackerParams {
    // Field names follow the pybind argument names of the setter that writes
    // them, so `visit` keys read `<setter>.<arg>` and line up with the
    // resolved JSON's `native_params.GPUByteTracker.calls[*].args`.

    struct Core {  // set_params
        float track_thresh = 0.1f;
        float high_thresh = 0.5f;
        float match_thresh = 0.8f;
        int track_buffer = 30;
        float mid_thresh = 0.40f;
        int confirm_streak = 3;
        float confirm_score_thresh = 0.50f;
        bool adaptive_confirmation = false;
        float new_track_thresh = 0.40f;
        int kalman_adapt_mode = 0;
        float r_scale = 1.0f;
        float vel_dir_weight = 0.0f;
        float fuse_score_weight = 0.0f;
        float stage2_match_thresh = 0.5f;
        float birth_low_score_thresh = 0.0f;
        float birth_prox_norm_thresh = 0.0f;
    };

    struct Reid {  // set_reid_params
        float cos_threshold = 0.90f;
        float iou_low = 0.3f;
        float iou_high = 0.6f;
        float weight = 0.4f;
        float cost_cos_w = 0.55f;
        float cost_iou_w = 0.30f;
        float cost_score_w = 0.15f;
    };

    struct Relink {  // set_relink_params
        bool enabled = false;
        int bank_cap = 256;
        float sim_thresh = 0.6f;
        float cheb_lambda = 2.5f;
        float spatial_gate = 4.0f;
        int max_age = 300;
        bool bidirectional = false;
        float bridge_px = 0.25f;
        int bridge_at = 4;
        int bridge_min_lost = 2;
        int bridge_ttl = 120;
        float bridge_max_speed = 0.0f;
        float bridge_person_height = 1.65f;
        float bridge_fps = 30.0f;
        float bridge_margin = 0.0f;
        float bridge_spatial_gate = 0.0f;
        int bridge_anchor = 0;          // 0=center 1=foot 2=adaptive (residual-weighted)
        float bridge_anchor_rate = 0.0f;  // adaptive deformation gate; 0=always-on
        float bridge_h_lo = 0.0f;       // scale gate: min ema_lost/ema_cand ratio
        float bridge_h_hi = 0.0f;       // scale gate: max ratio (<=0 disables the gate)
        float bridge_dir_bonus = 0.0f;  // directional consistency relaxation multiplier
        float occ_gate_cover = 0.0f;    // gap-occupancy veto: min occ_cover (0=off)
        int occ_gap_min = 30;           // occ gates apply only to gaps >= this
        float occ_expand_px = 0.0f;     // tiered expansion: looser bridge_px when occ high
        float occ_expand_cover = 0.9f;  // min occ_cover to unlock the expanded threshold
        float bridge_app_veto = -1.0f;  // appearance cosine veto floor (<=-1 off)
    };

    struct FrameSize {  // set_frame_size
        int w = 1920;
        int h = 1080;
    };

    struct Quality {  // set_quality_params
        bool enabled = false;
        float w_aspect = 0.50f;
        float w_center = 0.30f;
        float w_area = 0.20f;
    };

    struct Oao {  // set_oao_params
        float tau = 0.0f;
        // Contention gate: < 0 → plain OAO (bit-exact legacy); >= 0 → apply the
        // penalty only when the detection is also claimed by t's max-overlap
        // partner (partner-pred IoU >= thresh). Spares uncontested side-by-side
        // real tracks.
        float contest_thresh = -1.0f;
        // Soft score weight: <= 0 → off (full penalty); the penalty is scaled
        // by (1 - score_w * det_score), so confident detections get a reduced
        // penalty without cutting it entirely. The kernel tests `<= 0`
        // (oao_score_scale), so any non-positive value means off.
        float score_w = -1.0f;
        // Occlusion signal: 0 = max single inter-track IoU (default, bit-exact);
        // 1 = union coverage (fraction of t covered by union of other boxes).
        int occ_mode = 0;
        // Crowd multiplier: <= 0 → off; > 0 → scale penalty by (1 - 1/N), N =
        // tracks (incl. self) within crowd_radius * h of t.
        float crowd_radius = 0.0f;
        // Same-height gate: <= 0 → off; > 0 → only partners with |h_t - h_j| <=
        // gate * max(h) contribute to occ_coeff (same-depth occlusions only).
        float height_gate = 0.0f;
        // Same-foot gate: <= 0 → off; > 0 → only partners with |footy_t -
        // footy_j| <= gate * h_ref contribute.
        float foot_gate = 0.0f;
        // Duration ramp: <= 0 → off; > 0 → penalty *= min(1, overlap_frames/ramp).
        float ramp_frames = 0.0f;
    };

    struct Occ {  // set_occ_params: occluder-side depth mutual exclusion
        bool enabled = false;
        float iou_thresh = 0.45f;
        float foot_gap = 0.15f;  // same-height gate: flag only at similar depth
        int ttl = 4;
        float cost_weight = 0.50f;
    };

    struct AssociationEnergy {  // set_association_energy_params
        bool enabled = false;
        float score_cost_w = 0.0f;
        float height_cost_w = 0.0f;
    };

    // Association/motion/output knobs that used to be read from SACCADE_*
    // in the constructor and update path. Defaults are the values those
    // reads resolved to with the variable unset.
    struct Hatch {  // set_hatch_params
        bool enable_dda = true;           // SACCADE_ENABLE_DDA
        float dda_max_cost = 0.12f;       // SACCADE_DDA_MAX_COST
        float gate_adapt_r_mult = 1.0f;   // SACCADE_GATE_ADAPT_R_MULT
        // Occlusion-gated velocity damping during miss-gap coasting: scales the
        // x,y position-mean extrapolation when a track is coasting (age >= 1)
        // and was occluded last frame (occ_coeff >= occ_vel_occ_thresh).
        // 1.0 = bit-identical no-op; 0.0 = full static hold.
        float occ_vel_damp = 1.0f;        // SACCADE_OCC_VEL_DAMP
        float occ_vel_occ_thresh = 0.05f; // SACCADE_OCC_VEL_OCC_THRESH
        // NSA-Kalman gating/output decouple: matched tracks emit their
        // associated detection box instead of the filtered state (the filter
        // state itself is untouched). false = emit filtered state.
        bool output_measurement = false;  // SACCADE_OUTPUT_MEASUREMENT
        // Predict-through-occlusion: confirmed tracks coast-emit their predicted
        // box for up to coast_max_age missed frames, score decayed per frame,
        // only when occ_coeff >= coast_occ_thresh (0 = no gate).
        // coast_max_age == 0 ⇒ matched-only output.
        int coast_max_age = 0;            // SACCADE_COAST_MAX_AGE (truncated to int)
        float coast_score_decay = 1.0f;   // SACCADE_COAST_SCORE_DECAY
        float coast_occ_thresh = 0.0f;    // SACCADE_COAST_OCC_THRESH
        float freshness_w = 0.0f;         // SACCADE_FRESHNESS_W: bid += w/(1+age)
        float stability_w = 0.1f;         // SACCADE_STABILITY_W: bid += w/(1+dh_rel)
    };

    Core core;
    Reid reid;
    int reid_min_candidates = 2;  // set_reid_min_candidates (no pybind binding)
    Relink relink;
    // Stored for readback; no native kernel reads it (the Python semantic
    // reranker applies these weights).
    UnifiedScoreParams unified;
    FrameSize frame;
    Quality quality;
    Oao oao;
    Occ occ;
    bool multiplicative_cost = false;
    float sinkhorn_lambda = 30.0f;
    float stability_cost_w = 0.0f;
    AssociationEnergy association_energy;
    // Row-major 3x3; nullopt = null (MMD off, device matrix zeroed).
    std::optional<std::array<float, 9>> homography;
    Hatch hatch;

    // ── setters: the only writers; canonicalization lives here ─────────────
    void set_params(float track_thresh, float high_thresh, float match_thresh,
                    int track_buffer, float mid_thresh, int confirm_streak,
                    float confirm_score_thresh, bool adaptive_confirmation,
                    float new_track_thresh, int kalman_adapt_mode, float r_scale,
                    float vel_dir_weight, float fuse_score_weight,
                    float stage2_match_thresh, float birth_low_score_thresh,
                    float birth_prox_norm_thresh);
    void set_reid_params(float cos_threshold, float iou_low, float iou_high, float weight,
                         float cost_cos_w, float cost_iou_w, float cost_score_w);
    void set_reid_min_candidates(int min_candidates);
    void set_relink_params(bool enabled, int bank_cap, float sim_thresh, float cheb_lambda,
                           float spatial_gate, int max_age, bool bidirectional,
                           float bridge_px, int bridge_at, int bridge_min_lost,
                           int bridge_ttl, float bridge_max_speed,
                           float bridge_person_height, float bridge_fps,
                           float bridge_margin, float bridge_spatial_gate,
                           int bridge_anchor, float bridge_anchor_rate,
                           float bridge_h_lo, float bridge_h_hi, float bridge_dir_bonus,
                           float occ_gate_cover, int occ_gap_min, float occ_expand_px,
                           float occ_expand_cover, float bridge_app_veto);
    void set_unified_score_params(const UnifiedScoreParams& params);
    void set_frame_size(int w, int h);
    void set_quality_params(bool enabled, float w_aspect, float w_center, float w_area);
    void set_oao_params(float tau, float contest_thresh, float score_w, int occ_mode,
                        float crowd_radius, float height_gate, float foot_gate,
                        float ramp_frames);
    void set_occ_params(bool enabled, float iou_thresh, float foot_gap, int ttl,
                        float cost_weight);
    void set_multiplicative_cost(bool enabled);
    void set_sinkhorn_lambda(float lambda);
    void set_stability_cost_w(float w);
    void set_association_energy_params(bool enabled, float score_cost_w, float height_cost_w);
    void set_homography(const float* h);  // nullptr = null
    void set_hatch_params(const Hatch& hatch);

    // ── the one schema walk ────────────────────────────────────────────────
    // v(key, field) once per field, in declaration order. Keys are
    // `<setter>.<arg>` (unified score: `set_unified_score_params.params.<f>`;
    // hatches: `native_env.<SACCADE_*>` after the variable they replace).
    template <class V> void visit(V&& v) const {
        v("set_params.track_thresh", core.track_thresh);
        v("set_params.high_thresh", core.high_thresh);
        v("set_params.match_thresh", core.match_thresh);
        v("set_params.track_buffer", core.track_buffer);
        v("set_params.mid_thresh", core.mid_thresh);
        v("set_params.confirm_streak", core.confirm_streak);
        v("set_params.confirm_score_thresh", core.confirm_score_thresh);
        v("set_params.adaptive_confirmation", core.adaptive_confirmation);
        v("set_params.new_track_thresh", core.new_track_thresh);
        v("set_params.kalman_adapt_mode", core.kalman_adapt_mode);
        v("set_params.r_scale", core.r_scale);
        v("set_params.vel_dir_weight", core.vel_dir_weight);
        v("set_params.fuse_score_weight", core.fuse_score_weight);
        v("set_params.stage2_match_thresh", core.stage2_match_thresh);
        v("set_params.birth_low_score_thresh", core.birth_low_score_thresh);
        v("set_params.birth_prox_norm_thresh", core.birth_prox_norm_thresh);

        v("set_reid_params.cos_threshold", reid.cos_threshold);
        v("set_reid_params.iou_low", reid.iou_low);
        v("set_reid_params.iou_high", reid.iou_high);
        v("set_reid_params.weight", reid.weight);
        v("set_reid_params.cost_cos_w", reid.cost_cos_w);
        v("set_reid_params.cost_iou_w", reid.cost_iou_w);
        v("set_reid_params.cost_score_w", reid.cost_score_w);
        v("set_reid_min_candidates.min_candidates", reid_min_candidates);

        v("set_relink_params.enabled", relink.enabled);
        v("set_relink_params.bank_cap", relink.bank_cap);
        v("set_relink_params.sim_thresh", relink.sim_thresh);
        v("set_relink_params.cheb_lambda", relink.cheb_lambda);
        v("set_relink_params.spatial_gate", relink.spatial_gate);
        v("set_relink_params.max_age", relink.max_age);
        v("set_relink_params.bidirectional", relink.bidirectional);
        v("set_relink_params.bridge_px", relink.bridge_px);
        v("set_relink_params.bridge_at", relink.bridge_at);
        v("set_relink_params.bridge_min_lost", relink.bridge_min_lost);
        v("set_relink_params.bridge_ttl", relink.bridge_ttl);
        v("set_relink_params.bridge_max_speed", relink.bridge_max_speed);
        v("set_relink_params.bridge_person_height", relink.bridge_person_height);
        v("set_relink_params.bridge_fps", relink.bridge_fps);
        v("set_relink_params.bridge_margin", relink.bridge_margin);
        v("set_relink_params.bridge_spatial_gate", relink.bridge_spatial_gate);
        v("set_relink_params.bridge_anchor", relink.bridge_anchor);
        v("set_relink_params.bridge_anchor_rate", relink.bridge_anchor_rate);
        v("set_relink_params.bridge_h_lo", relink.bridge_h_lo);
        v("set_relink_params.bridge_h_hi", relink.bridge_h_hi);
        v("set_relink_params.bridge_dir_bonus", relink.bridge_dir_bonus);
        v("set_relink_params.occ_gate_cover", relink.occ_gate_cover);
        v("set_relink_params.occ_gap_min", relink.occ_gap_min);
        v("set_relink_params.occ_expand_px", relink.occ_expand_px);
        v("set_relink_params.occ_expand_cover", relink.occ_expand_cover);
        v("set_relink_params.bridge_app_veto", relink.bridge_app_veto);

        v("set_unified_score_params.params.w_sim_base", unified.w_sim_base);
        v("set_unified_score_params.params.w_iou_base", unified.w_iou_base);
        v("set_unified_score_params.params.w_maha_base", unified.w_maha_base);
        v("set_unified_score_params.params.shift_ambiguity", unified.shift_ambiguity);
        v("set_unified_score_params.params.shift_lost_age", unified.shift_lost_age);

        v("set_frame_size.w", frame.w);
        v("set_frame_size.h", frame.h);

        v("set_quality_params.enabled", quality.enabled);
        v("set_quality_params.w_aspect", quality.w_aspect);
        v("set_quality_params.w_center", quality.w_center);
        v("set_quality_params.w_area", quality.w_area);

        v("set_oao_params.tau", oao.tau);
        v("set_oao_params.contest_thresh", oao.contest_thresh);
        v("set_oao_params.score_w", oao.score_w);
        v("set_oao_params.occ_mode", oao.occ_mode);
        v("set_oao_params.crowd_radius", oao.crowd_radius);
        v("set_oao_params.height_gate", oao.height_gate);
        v("set_oao_params.foot_gate", oao.foot_gate);
        v("set_oao_params.ramp_frames", oao.ramp_frames);

        v("set_occ_params.enabled", occ.enabled);
        v("set_occ_params.iou_thresh", occ.iou_thresh);
        v("set_occ_params.foot_gap", occ.foot_gap);
        v("set_occ_params.ttl", occ.ttl);
        v("set_occ_params.cost_weight", occ.cost_weight);

        v("set_multiplicative_cost.enabled", multiplicative_cost);
        v("set_sinkhorn_lambda.lambda", sinkhorn_lambda);
        v("set_stability_cost_w.w", stability_cost_w);

        v("set_association_energy_params.enabled", association_energy.enabled);
        v("set_association_energy_params.score_cost_w", association_energy.score_cost_w);
        v("set_association_energy_params.height_cost_w", association_energy.height_cost_w);

        v("set_homography.h", homography);

        v("native_env.SACCADE_ENABLE_DDA", hatch.enable_dda);
        v("native_env.SACCADE_DDA_MAX_COST", hatch.dda_max_cost);
        v("native_env.SACCADE_GATE_ADAPT_R_MULT", hatch.gate_adapt_r_mult);
        v("native_env.SACCADE_OCC_VEL_DAMP", hatch.occ_vel_damp);
        v("native_env.SACCADE_OCC_VEL_OCC_THRESH", hatch.occ_vel_occ_thresh);
        v("native_env.SACCADE_OUTPUT_MEASUREMENT", hatch.output_measurement);
        v("native_env.SACCADE_COAST_MAX_AGE", hatch.coast_max_age);
        v("native_env.SACCADE_COAST_SCORE_DECAY", hatch.coast_score_decay);
        v("native_env.SACCADE_COAST_OCC_THRESH", hatch.coast_occ_thresh);
        v("native_env.SACCADE_FRESHNESS_W", hatch.freshness_w);
        v("native_env.SACCADE_STABILITY_W", hatch.stability_w);
    }
};

// Research hooks and diagnostics: off unless a research harness turns them on.
// They are not configuration (the resolved JSON never sets them), but they can
// change decisions or force host I/O, so the snapshot reports them and the
// shipping loader requires every one to be off.
struct TrackerInstrumentation {
    bool portable_or_tail = false;       // set_research_portable_or_tail
    bool bridge_shadow = false;          // set_research_bridge_shadow
    bool bridge_fidelity_audit = false;  // set_research_bridge_fidelity_audit
    bool h0_bridge_trace = false;        // set_research_h0_bridge_trace
    bool assoc_dump = false;             // set_assoc_dump_path (SACCADE_ASSOC_DUMP)

    template <class V> void visit(V&& v) const {
        v("research.portable_or_tail", portable_or_tail);
        v("research.bridge_shadow", bridge_shadow);
        v("research.bridge_fidelity_audit", bridge_fidelity_audit);
        v("research.h0_bridge_trace", h0_bridge_trace);
        v("diagnostic.assoc_dump", assoc_dump);
    }
};

// Read-only view of everything that configures one GPUByteTracker.
struct TrackerSnapshot {
    int max_objects = 0;    // constructor
    int embedding_dim = 0;  // constructor
    int max_assoc = 0;      // constructor, after max(1, max_assoc)
    TrackerParams params;
    TrackerInstrumentation instrumentation;
    // True once update()/update_into() ran inside a CUDA stream capture; every
    // setter that writes params_ or arms a research hook then throws.
    bool config_frozen = false;

    template <class V> void visit(V&& v) const {
        v("constructor.max_objects", max_objects);
        v("constructor.embedding_dim", embedding_dim);
        v("constructor.max_assoc", max_assoc);
        params.visit(v);
        instrumentation.visit(v);
        v("config_frozen", config_frozen);
    }
};

}  // namespace saccade
