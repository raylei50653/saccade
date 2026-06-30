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
    int class_id;
    int age;
    float score;
    std::vector<float> state;      // [cx, cy, a, h, vx, vy, va, vh]
    std::vector<float> covariance; // row-major 8x8 covariance
};

struct TrackCandidateSnapshot {
    int obj_id;
    int class_id;
    int age;
    int hit_streak;
    int required_confirm_streak;
    float score;
    float x1, y1, x2, y2;
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
                           float occ_expand_px = 0.0f, float occ_expand_cover = 0.9f);
    std::vector<int> get_relink_debug();

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
    cudaStream_t stream
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

// Stable descending argsort: equal-score ties break toward lower original index.
// d_keys_in / d_keys_out are uint64_t scratch buffers of length n each.
void SACCADE_TRACKING_API argsort_scores_descending_cuda(
    const float* d_scores, int n,
    int64_t* d_order_out, uint64_t* d_keys_in, uint64_t* d_keys_out,
    void* d_cub_tmp, size_t cub_tmp_bytes, cudaStream_t stream);

} // namespace saccade
