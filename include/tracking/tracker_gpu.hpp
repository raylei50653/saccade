#pragma once

#include "saccade/common.hpp"
#include <cstdint>
#include <utility>
#include <vector>

namespace saccade {

/**
 * @brief GPU 追蹤結果結構體 (Zero-Sync)
 */
struct TrackResult {
    float x1, y1, x2, y2;
    int obj_id;
    float score;
    int class_id;
    int det_idx;  // detection index matched to this track; -1 if unmatched/predicted
};

struct TrackStateSnapshot {
    int obj_id;
    uint64_t track_uid = 0;
    int generation = 0;
    int class_id;
    int age;
    float score;
    std::vector<float> state;      // [cx, cy, a, h, vx, vy, va, vh]
    std::vector<float> covariance; // row-major 8x8 covariance
};

struct TrackCandidateSnapshot {
    int obj_id;
    uint64_t track_uid = 0;
    int generation = 0;
    int class_id;
    int age;
    int hit_streak;
    int required_confirm_streak;
    float score;
    float x1, y1, x2, y2;
};

// Default-off research capture of the values actually computed by the
// Consumer-A bridge kernel. This POD is written directly by CUDA and read back
// unchanged by the host; no field participates in the bridge decision.
struct BridgeFidelityEvent {
    int frame = 0;
    int lost_id = -1;
    int cand_id = -1;
    int lost_slot = -1;
    int cand_slot = -1;
    int lost_last_frame = 0;
    int cand_first_frame = 0;
    int gap = 0;
    int bridge_at = 0;
    int la = 0;
    int anchor_mode = 0;
    float anchor_rate = 0.0f;
    float bdist = 0.0f;
    float dist_h = 0.0f;
    float fwd_r = 0.0f;
    float bwd_r = 0.0f;
    float v_lost_x = 0.0f;
    float v_lost_y = 0.0f;
    float v_cand_x = 0.0f;
    float v_cand_y = 0.0f;
    float ax = 0.0f;
    float ay = 0.0f;
    float cx0 = 0.0f;
    float cy0 = 0.0f;
    float ema_lost = 0.0f;
    float ema_cand = 0.0f;
    float h_ref = 0.0f;
    float s_lost = 0.0f;
    float w = 0.0f;
    float production_threshold = 0.0f;
    // R1 temporal-reduction capture: exact chronological windows consumed by
    // bridge_anchor4. The candidate uses its head-four; the lost track uses
    // its last-four, or a single fallback point when it has fewer than four
    // samples. These are observational inputs only, never bridge state.
    int lost_window_size = 0;
    int cand_window_size = 0;
    float lost_anchor_window[12] = {};  // 4 × (cx, cy, h)
    float cand_anchor_window[12] = {};  // 4 × (cx, cy, h)
    float bridge_dir_bonus = 0.0f;
};

struct BridgeFidelityCapture {
    std::vector<BridgeFidelityEvent> events;
    int total_events = 0;
    int overflow_events = 0;
};

// H0 v1 records the native bridge policy as four separate streams.  These POD
// structures are the capture ABI: CUDA writes them directly and the host only
// drains/serializes their fields.  They never participate in bridge selection.
enum H0ScalarStatus : uint8_t {
    H0_NOT_COMPUTED = 0,
    H0_COMPUTED_FINITE = 1,
    H0_COMPUTED_POS_INF = 2,
    H0_COMPUTED_NEG_INF = 3,
    H0_COMPUTED_NAN = 4,
};

struct H0Float32 {
    uint32_t bits = 0;
    uint8_t status = H0_NOT_COMPUTED;
    uint8_t reserved[3] = {};
};

enum H0Verdict : uint8_t {
    H0_NOT_EVALUATED = 0,
    H0_PASS = 1,
    H0_REJECT = 2,
    H0_DISABLED = 3,
};

enum H0PairRejectReason : uint8_t {
    H0_PAIR_REJECT_NONE = 0,
    H0_PAIR_REJECT_HEIGHT = 1,
    H0_PAIR_REJECT_SPEED = 2,
    H0_PAIR_REJECT_SPATIAL = 3,
    H0_PAIR_REJECT_CUTOFF = 4,
    H0_PAIR_REJECT_OCCUPANCY = 5,
    H0_PAIR_REJECT_APPEARANCE = 6,
    H0_PAIR_REJECT_PORTABLE_TAIL = 7,
};

enum H0CandidateStatus : uint8_t {
    H0_CAND_NO_STRUCTURAL_COMPETITORS = 0,
    H0_CAND_ALL_REJECTED_PRE_SCORE = 1,
    H0_CAND_ALL_REJECTED_CUTOFF_OR_VETO = 2,
    H0_CAND_MARGIN_REJECTED = 3,
    H0_CAND_PROPOSAL_EMITTED = 4,
};

enum H0ProposalRejectReason : uint8_t {
    H0_PROPOSAL_REJECT_NONE = 0,
    H0_PROPOSAL_REJECT_NO_COMPETITOR = 1,
    H0_PROPOSAL_REJECT_MARGIN = 2,
};

// Stable key: (frame, cand_slot, cand_instance_uid, lost_slot,
// lost_instance_uid).  The sequence is supplied by the caller at drain time;
// a tracker instance has no dataset-sequence identity.
struct H0BridgePairRecord {
    uint32_t schema_version = 1;
    int frame = 0;
    int cand_slot = -1;
    int lost_slot = -1;
    int cand_precommit_track_id = -1;
    int lost_precommit_track_id = -1;
    uint64_t cand_instance_uid = 0;
    uint64_t lost_instance_uid = 0;
    int la = 0;
    int bridge_at = 0;
    int cand_ring_length = 0;
    int lost_ring_length = 0;
    H0Float32 ema_lost;
    H0Float32 ema_cand;
    H0Float32 height_ratio;
    uint8_t height_verdict = H0_NOT_EVALUATED;
    H0Float32 speed;
    uint8_t speed_verdict = H0_NOT_EVALUATED;
    H0Float32 spatial_distance;
    uint8_t spatial_verdict = H0_NOT_EVALUATED;
    H0Float32 lost_anchor_x;
    H0Float32 lost_anchor_y;
    H0Float32 cand_anchor_x;
    H0Float32 cand_anchor_y;
    H0Float32 lost_velocity_x;
    H0Float32 lost_velocity_y;
    H0Float32 cand_velocity_x;
    H0Float32 cand_velocity_y;
    H0Float32 h_ref;
    H0Float32 fwd_r;
    H0Float32 bwd_r;
    H0Float32 dist_h;
    H0Float32 s_lost;
    H0Float32 w;
    H0Float32 direction_cosine;
    H0Float32 directional_alpha;
    H0Float32 directional_cross_bdist;
    H0Float32 bdist_before_direction;
    H0Float32 bdist_after_direction;
    uint8_t cutoff_verdict = H0_NOT_EVALUATED;
    uint8_t occupancy_verdict = H0_NOT_EVALUATED;
    H0Float32 occupancy_coverage;
    uint8_t appearance_verdict = H0_NOT_EVALUATED;
    H0Float32 appearance_cosine;
    uint8_t portable_tail_verdict = H0_NOT_EVALUATED;
    int portable_tail_mask = 0;
    uint8_t final_pair_eligible = H0_NOT_EVALUATED;
    uint8_t reject_reason = H0_PAIR_REJECT_NONE;
};

// Stable key: (frame, cand_slot, cand_instance_uid).
struct H0BridgeCandidateRecord {
    uint32_t schema_version = 1;
    int frame = 0;
    int cand_slot = -1;
    int cand_precommit_track_id = -1;
    uint64_t cand_instance_uid = 0;
    int structural_competitors = 0;
    int pre_score_passes = 0;
    int final_pair_eligible_count = 0;
    int best_lost_slot = -1;
    int second_lost_slot = -1;
    int best_lost_precommit_track_id = -1;
    int second_lost_precommit_track_id = -1;
    uint64_t best_lost_instance_uid = 0;
    uint64_t second_lost_instance_uid = 0;
    H0Float32 best_bdist;
    H0Float32 second_best_bdist;
    H0Float32 margin;
    uint8_t no_second_competitor = 0;
    uint8_t margin_verdict = H0_NOT_EVALUATED;
    uint8_t proposal_emitted = H0_NOT_EVALUATED;
    uint8_t proposal_reject_reason = H0_PROPOSAL_REJECT_NONE;
    uint8_t candidate_status = H0_CAND_NO_STRUCTURAL_COMPETITORS;
};

// Stable key: (frame, proposing_cand_slot, proposing_cand_instance_uid,
// proposed_lost_slot, proposed_lost_instance_uid).
struct H0BridgeClaimRecord {
    uint32_t schema_version = 1;
    int frame = 0;
    int proposing_cand_slot = -1;
    int proposed_lost_slot = -1;
    int proposing_cand_precommit_track_id = -1;
    int proposed_lost_precommit_track_id = -1;
    uint64_t proposing_cand_instance_uid = 0;
    uint64_t proposed_lost_instance_uid = 0;
    H0Float32 detection_score;
    int sq = 0;
    int packed_atomic_key = 0;
    int candidate_index_component = -1;
    int winning_cand_slot = -1;
    int winning_cand_precommit_track_id = -1;
    uint64_t winning_cand_instance_uid = 0;
    uint8_t claim_won = H0_NOT_EVALUATED;
};

// Stable key is the winning H0BridgeClaimRecord key.
struct H0BridgeCommitRecord {
    uint32_t schema_version = 1;
    int frame = 0;
    int cand_slot = -1;
    int lost_slot = -1;
    int cand_precommit_track_id = -1;
    int lost_precommit_track_id = -1;
    int cand_postcommit_track_id = -1;
    int lost_postcommit_track_id = -1;
    uint64_t cand_instance_uid = 0;
    uint64_t lost_instance_uid = 0;
    uint8_t cand_active_before = 0;
    uint8_t cand_active_after = 0;
    uint8_t lost_active_before = 0;
    uint8_t lost_active_after = 0;
    uint8_t commit_executed = H0_NOT_EVALUATED;
    uint8_t lost_slot_deactivated = H0_NOT_EVALUATED;
};

// These keys are emitted at native decision points independently of the four
// record append paths. They make a dropped record observable to the host
// verifier rather than letting the packet define its own completeness domain.
struct H0BridgeCandidateKey {
    int frame = 0;
    int cand_slot = -1;
    uint64_t cand_instance_uid = 0;
};

struct H0BridgePairKey {
    int frame = 0;
    int cand_slot = -1;
    int lost_slot = -1;
    uint64_t cand_instance_uid = 0;
    uint64_t lost_instance_uid = 0;
};

struct H0BridgeClaimKey {
    int frame = 0;
    int proposing_cand_slot = -1;
    int proposed_lost_slot = -1;
    uint64_t proposing_cand_instance_uid = 0;
    uint64_t proposed_lost_instance_uid = 0;
};

struct H0BridgeDecisionTraceCapture {
    std::vector<H0BridgePairRecord> pair_records;
    std::vector<H0BridgeCandidateRecord> candidate_records;
    std::vector<H0BridgeClaimRecord> claim_records;
    std::vector<H0BridgeCommitRecord> commit_records;
    std::vector<H0BridgeCandidateKey> native_candidate_keys;
    std::vector<H0BridgePairKey> native_pair_keys;
    std::vector<H0BridgeClaimKey> native_proposal_keys;
    std::vector<H0BridgeClaimKey> native_claim_winner_keys;
    std::vector<H0BridgePairKey> native_commit_keys;
    int total_pair_records = 0;
    int total_candidate_records = 0;
    int total_claim_records = 0;
    int total_commit_records = 0;
    int overflow_pair_records = 0;
    int overflow_candidate_records = 0;
    int overflow_claim_records = 0;
    int overflow_commit_records = 0;
    int total_native_candidate_keys = 0;
    int total_native_pair_keys = 0;
    int total_native_proposal_keys = 0;
    int total_native_claim_winner_keys = 0;
    int total_native_commit_keys = 0;
    int overflow_native_candidate_keys = 0;
    int overflow_native_pair_keys = 0;
    int overflow_native_proposal_keys = 0;
    int overflow_native_claim_winner_keys = 0;
    int overflow_native_commit_keys = 0;
    int identity_uid_wrap_events = 0;
};

struct UnifiedScoreParams {
    float w_sim_base = 0.0f;
    float w_iou_base = 0.0f;
    float w_maha_base = 0.0f;
    float shift_ambiguity = 0.0f;
    float shift_lost_age = 0.0f;
};

struct TrackerGPUBuffers {
    uintptr_t states;     // float*,  device pointer [max_objs * 8]
    uintptr_t covs;       // float*,  device pointer [max_objs * 64]
    uintptr_t track_ids;  // int*,    device pointer [max_objs]
    uintptr_t track_uids; // uint64_t*, device pointer [max_objs]
    int max_objs;
};

/**
 * @brief ITracker 接口
 */
class SACCADE_TRACKING_API ITracker {
public:
    virtual ~ITracker() = default;
    virtual std::vector<TrackResult> update(
        float* boxes_ptr, 
        float* scores_ptr, 
        int* classes_ptr, 
        int num_dets,
        cudaStream_t stream,
        float* embeddings_ptr = nullptr,
        float* gmc_ptr = nullptr,
        float light_factor = 0.0f,
        float mid_thresh_scale = 1.0f
    ) = 0;
};

/**
 * @brief GPUByteTracker 核心實作
 */
class SACCADE_TRACKING_API GPUByteTracker : public ITracker {
public:
    GPUByteTracker(int max_objects = 2048, int embedding_dim = 768, int max_assoc = 1024);
    ~GPUByteTracker();

    void set_params(
        float track_thresh,
        float high_thresh,
        float match_thresh,
        int track_buffer,
        float mid_thresh = 0.40f,
        int confirm_streak = 3,
        float confirm_score_thresh = 0.50f,
        bool adaptive_confirmation = false,
        float new_track_thresh = -1.0f,
        int kalman_adapt_mode = 0,
        float r_scale = 1.0f,
        float vel_dir_weight = 0.0f,
        float fuse_score_weight = 0.0f,
        float stage2_match_thresh = 0.5f,
        float birth_low_score_thresh = 0.0f,
        float birth_prox_norm_thresh = 0.0f
    );
    void set_reid_params(float cos_threshold, float iou_low, float iou_high, float weight,
                         float cost_cos_w = 0.55f, float cost_iou_w = 0.30f, float cost_score_w = 0.15f);
    void set_reid_min_candidates(int min_candidates);
    void set_relink_params(bool enabled, int bank_cap, float sim_thresh,
                           float cheb_lambda, float spatial_gate, int max_age,
                           bool bidirectional = false, float bridge_px = 0.25f,
                           int bridge_at = 4, int bridge_min_lost = 2, int bridge_ttl = 120,
                           float bridge_max_speed = 0.0f, float bridge_person_height = 1.65f,
                           float bridge_fps = 30.0f, float bridge_margin = 0.0f,
                           float bridge_spatial_gate = 0.0f, int bridge_anchor = 0,
                           float bridge_anchor_rate = 0.0f,
                            float bridge_h_lo = 0.0f, float bridge_h_hi = 0.0f,
                            float bridge_dir_bonus = 0.0f,
                           float occ_gate_cover = 0.0f, int occ_gap_min = 30,
                           float occ_expand_px = 0.0f, float occ_expand_cover = 0.9f,
                           float bridge_app_veto = -1.0f);
    /**
     * Research-only M-B1 portable OR-tail hook (default-off).
     * When enabled, reject bridge pairs if any of 5 frozen tail thresholds fire.
     * thr must have length 5 in order:
     *   score_m_bridge, abs_log_h, dist_h, abs_ratio_m1, resid_mean
     * Production path: enabled=false is bit-identical to prior behavior.
     */
    void set_research_portable_or_tail(bool enabled,
                                       const std::vector<float>& thr,
                                       bool audit_enabled = false);
    std::vector<int> get_relink_debug();

    // Issue #112: default-off, decision-neutral native bridge-score capture.
    void set_research_bridge_fidelity_audit(bool enabled, int capacity = 65536);
    void clear_research_bridge_fidelity_audit();
    BridgeFidelityCapture drain_research_bridge_fidelity_events();

    // H0 full decision-path capture.  This is default-off observational
    // instrumentation for the real bridge commit path; it is not shadow mode.
    void set_research_h0_bridge_trace(bool enabled,
                                      int pair_capacity = 65536,
                                      int candidate_capacity = 16384,
                                      int claim_capacity = 16384,
                                      int commit_capacity = 16384);
    // Bind the caller-owned, stable device scalar containing the actual
    // evaluation-frame ID.  H0 kernels read this scalar at replay time, so a
    // CUDA graph never freezes a host-side frame argument.  The caller must
    // retain the allocation until H0 tracing is disabled.
    void bind_research_h0_bridge_trace_frame_device(const int* frame_ptr);
    void clear_research_h0_bridge_trace();
    H0BridgeDecisionTraceCapture drain_research_h0_bridge_trace();

    /**
     * @brief Issue #112 shadow bridge: propose (and capture) but never commit.
     *
     * The fidelity capture only runs inside relink_bidir_propose_kernel, which
     * is gated on the bridge being enabled -- but an enabled bridge rewrites
     * track identity in relink_bidir_commit_kernel, so captured events cannot
     * be joined against a bridge-off pair cohort. Shadow mode skips the commit
     * kernel: propose writes only bridge-private state (bridge_claim,
     * bridge_cand_lost, track_revived, dbg, fidelity events), none of which is
     * read by compact_results_kernel, and spawn only zeroes that state on slot
     * reuse. Commit is the sole bridge write to track_ids/active, so skipping
     * it makes output bit-identical to a bridge-off run while still emitting
     * real float32 CUDA bridge scores.
     */
    void set_research_bridge_shadow(bool enabled);

    /**
     * @brief OA-SORT Occlusion-Aware Offset (OAO) penalty weight.
     * @param tau Cost penalty scale in [0, 1]. 0 = disabled (default).
     *            When > 0, tracks whose predicted boxes overlap other tracks get
     *            a cost increase of tau * IoU_overlap, reducing incorrect associations
     *            during occlusion (cost confusion).
     */
    void set_oao_params(float tau, float contest_thresh = -1.0f, float score_w = -1.0f,
                        int occ_mode = 0, float crowd_radius = 0.0f, float height_gate = 0.0f,
                        float foot_gate = 0.0f, float ramp_frames = 0.0f);

    /**
     * @brief Configure the depth-gated occlusion-state machine (default off).
     *
     * When enabled, a confirmed track that loses its detection behind a higher-IoU
     * partner whose foot is decisively lower (closer to camera) is held in an
     * OCCLUDED state bound to that occluder, and a depth-consistency cost term biases
     * its re-acquisition toward the depth-consistent re-emerging box — resolving
     * occlusion crossing-swaps without appearance or velocity-direction cues.
     */
    void set_occ_params(bool enabled, float iou_thresh, float foot_gap, int ttl,
                        float cost_weight);
    std::vector<int> get_occ_front_info();

    /**
     * @brief Enable multiplicative log-linear cost form (default off).
     *
     * When enabled, cost = 1 - quality * exp(-Σ penalty) instead of the
     * additive clamp chain. The value is kept per tracker instance and passed
     * to each cost-kernel launch.
     */
    void set_multiplicative_cost(bool enabled);

    /**
     * @brief Set stability reward weight for multiplicative cost form.
     *
     * When >0, size-consistent matches get reduced cost via
     * penalty -= w / (1 + |h_diff|/h_det). Default 0 (off).
     */
    void set_stability_cost_w(float w);

    /**
     * @brief Optional energy terms for association scoring.
     *
     * Baseline keeps this disabled. When enabled with multiplicative cost,
     * score and height-consistency terms are added to the log-linear energy
     * before cost = 1 - quality * exp(-energy).
     */
    void set_association_energy_params(
        bool enabled, float score_cost_w, float height_cost_w);

    /**
     * @brief Set Sinkhorn lambda (cost→prob scaling, default 30.0).
     *
     * Lower values (10-15) give softer discrimination and room for reward
     * terms to influence auction outcomes. Higher values sharpen the
     * transition but leave no room for small cost differences.
     */
    void set_sinkhorn_lambda(float lambda);

    /**
     * @brief Set Detection Quality Scaling (A6) parameters.
     * @param enabled Enable scaling
     * @param w_aspect Weight for aspect ratio quality
     * @param w_center Weight for center bias quality
     * @param w_area Weight for area ratio quality
     */
    void set_quality_params(bool enabled, float w_aspect = 0.50f, float w_center = 0.30f, float w_area = 0.20f);

    /**
     * @brief Set frame size for quality scaling and other geometry-aware logic.
     */
    void set_frame_size(int w, int h);

    /**
     * @brief Set homography matrix for 2D Ground Plane Mapping (MMD).
     * @param h 9-float array (3x3 row-major). If all zeros, MMD is disabled.
     */
    void set_homography(const float* h);

    void set_unified_score_params(const UnifiedScoreParams& params);
    void update_reference_features(int* track_ids, float* features_ptr, int num, cudaStream_t stream);
    void set_clean_embedding_flags(int* track_ids, bool* flags, int n, cudaStream_t stream);
    void set_clean_embedding_flags_host(int* h_tids, bool* h_flags, int n, cudaStream_t stream);
    void bind_features_buffer(float* ptr);
    std::vector<std::pair<int,int>> get_active_tid_slot_pairs();
    std::vector<TrackStateSnapshot> get_state_snapshots(cudaStream_t stream);

    /// GPU compaction kernel: reads active Kalman track states on device and
    /// writes compacted xyxy prior boxes + classes into caller-provided
    /// buffers.  Returns the compacted count (host-side int via 4-byte D2H).
    /// Replaces get_state_snapshots() (634 KB D2H + Python loop) for the
    /// prior-building hot path.
    int build_track_priors_gpu(
        float* d_out_boxes, int* d_out_classes,
        int min_track_age, int max_track_age, float min_track_score,
        cudaStream_t stream
    );

    std::vector<TrackStateSnapshot> get_motion_snapshots_for_track_ids(
        const std::vector<int>& track_ids,
        cudaStream_t stream
    );
    TrackerGPUBuffers get_gpu_buffers() const;
    int max_objects() const;
    int max_assoc() const;
    std::vector<TrackCandidateSnapshot> get_tentative_candidates(cudaStream_t stream);
    void update_into(
        float* boxes_ptr,
        float* scores_ptr,
        int* classes_ptr,
        int num_dets,
        cudaStream_t stream,
        float* out_boxes_ptr,
        float* out_scores_ptr,
        int* out_ids_ptr,
        int* out_classes_ptr,
        int* out_det_idx_ptr,
        int* out_count_ptr,
        float* embeddings_ptr = nullptr,
        float* gmc_ptr = nullptr,
        float light_factor = 0.0f,
        float mid_thresh_scale = 1.0f,
        int out_capacity = -1
    );

    std::vector<TrackResult> update(
        float* boxes_ptr, 
        float* scores_ptr, 
        int* classes_ptr, 
        int num_dets,
        cudaStream_t stream,
        float* embeddings_ptr = nullptr,
        float* gmc_ptr = nullptr,
        float light_factor = 0.0f,
        float mid_thresh_scale = 1.0f
    ) override;

    int compact_output_to_host(float* host_boxes, float* host_scores,
                                int* host_ids, int* host_classes,
                                int capacity, cudaStream_t stream);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

void SACCADE_TRACKING_API merge_cross_tile_duplicates_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    int num_dets,
    int* anchor_indices_ptr,
    float* box_sums_ptr,
    float* score_sums_ptr,
    int* score_bits_max_ptr,
    float* best_boxes_ptr,
    int* best_key_bits_ptr,
    int* cluster_counts_ptr,
    float* out_boxes_ptr,
    float* out_scores_ptr,
    int* out_classes_ptr,
    int* out_count_ptr,
    float iou_threshold,
    float center_threshold,
    float area_ratio_threshold,
    int tiling_mode,
    int frame_w,
    int frame_h,
    float seam_margin_canvas_px,
    float seam_center_scale,
    float seam_area_ratio_threshold,
    float seam_min_overlap_ratio,
    cudaStream_t stream
);

void SACCADE_TRACKING_API filter_detections_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    int num_dets,
    int* keep_indices_ptr,
    bool* suspect_flags_ptr,
    float* quality_scores_ptr,
    int* out_count_ptr,
    float score_threshold,
    bool track_person_only,
    int person_class,
    bool is_tiled,
    int frame_w,
    int frame_h,
    bool person_geometry_prior,
    bool geometry_suspect_support,
    float person_min_height_ratio,
    float person_min_aspect,
    float person_max_aspect,
    float person_min_area_ratio,
    float person_max_area_ratio,
    cudaStream_t stream,
    int* d_keep_flags = nullptr,
    int* d_prefix = nullptr,
    bool* d_suspect_tmp = nullptr,
    void* d_scan_tmp = nullptr,
    size_t scan_tmp_bytes = 0
);

void SACCADE_TRACKING_API nms_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    const int64_t* order_indices_ptr,
    int num_dets,
    int* keep_indices_ptr,
    uint64_t* suppression_masks_ptr,
    uint64_t* remv_ptr,
    int* out_count_ptr,
    float iou_threshold,
    bool class_aware,
    cudaStream_t stream
);

void SACCADE_TRACKING_API nms_counted_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    const int64_t* order_indices_ptr,
    int max_dets,
    const int* valid_count_ptr,
    int* keep_indices_ptr,
    uint64_t* suppression_masks_ptr,
    uint64_t* remv_ptr,
    int* out_count_ptr,
    float iou_threshold,
    bool class_aware,
    const float* priors_ptr,
    const int* prior_classes_ptr,
    int num_priors,
    float prior_iou_threshold,
    bool* immunity_mask_ptr,
    cudaStream_t stream
);

// M1: GPU gather-compact helpers for PerceptionPipeline
void SACCADE_TRACKING_API gather_compact3_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes,
    float* dst_boxes, float* dst_scores, int* dst_classes,
    const int* indices, int n, cudaStream_t stream);

void SACCADE_TRACKING_API gather_compact3_counted_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes,
    float* dst_boxes, float* dst_scores, int* dst_classes,
    const int* indices, const int* count_ptr, int max_n, cudaStream_t stream);

void SACCADE_TRACKING_API gather_compact4_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes, const bool* src_suspect,
    float* dst_boxes, float* dst_scores, int* dst_classes, bool* dst_suspect,
    const int* indices, int n, cudaStream_t stream);

void SACCADE_TRACKING_API gather_compact4_counted_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes, const bool* src_suspect,
    float* dst_boxes, float* dst_scores, int* dst_classes, bool* dst_suspect,
    const int* indices, const int* count_ptr, int max_n, cudaStream_t stream);

void SACCADE_TRACKING_API copy_bool_counted_cuda(
    const bool* src, bool* dst, const int* count_ptr, int max_n, cudaStream_t stream);

void SACCADE_TRACKING_API penalize_suspect_scores_cuda(
    float* scores, const bool* suspect, const int* count_ptr, float penalty_score, int max_n, cudaStream_t stream);

void SACCADE_TRACKING_API append_private_continuation_cuda(
    const float* src_boxes,
    const float* src_scores,
    const int* src_classes,
    const bool* src_suspect,
    float* dst_boxes,
    float* dst_scores,
    int* dst_classes,
    bool* dst_suspect,
    int* out_count_ptr,
    int output_capacity,
    const int* baseline_keep_indices,
    const int* baseline_count_ptr,
    bool* baseline_mask,
    const int* candidate_keep_indices,
    const int* candidate_count_ptr,
    const float* private_priors_ptr,
    int num_private_priors,
    float private_prior_iou_threshold,
    float private_prior_center_threshold,
    float score_floor,
    float score_ceiling,
    int max_private_candidates,
    int* private_added_count_ptr,
    cudaStream_t stream);

// Merged compact+sort+NMS with grid spatial indexing (#1,#2,#3,#5 optimizations)
void SACCADE_TRACKING_API compact_grid_nms_cuda(
    const float* boxes_ptr, const float* scores_ptr, const int* classes_ptr,
    int num_dets, const int* keep_indices, int valid_count,
    float* out_boxes, float* out_scores, int* out_classes,
    bool* out_suspect, int* out_count,
    float iou_threshold,
    cudaStream_t stream);

size_t SACCADE_TRACKING_API argsort_scores_descending_bytes(int n);

size_t SACCADE_TRACKING_API filter_stable_scan_temp_bytes(int n);

// Stable descending argsort: equal-score ties break toward lower original index.
// d_keys_in / d_keys_out are uint64_t scratch buffers of length n each.
void SACCADE_TRACKING_API argsort_scores_descending_cuda(
    const float* d_scores, int n,
    int64_t* d_order_out, uint64_t* d_keys_in, uint64_t* d_keys_out,
    void* d_cub_tmp, size_t cub_tmp_bytes, cudaStream_t stream);

} // namespace saccade
