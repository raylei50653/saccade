from dataclasses import dataclass
from typing import Any
from pathlib import Path
from saccade.perception.eval.preprocess import parse_preprocess
from saccade.perception.eval.assoc_basis import (
    PERSON_HEIGHT_M,
    REF_HEIGHT_RATIO,
    SCENE_FPS,
)


# ---------------------------------------------------------------------------
# Phase 4B: Module view dataclasses (projections of EvalConfig flat fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreView:
    """Core tracking and I/O parameters. Auto-projected from EvalConfig flat fields. Frozen."""

    workbench: bool
    threads: int
    data_root: str
    split: str
    debug_dump_seq: str
    debug_dump_frames: str
    debug_dump_csv: str
    debug_birth_csv: str
    conf_threshold: float
    track_thresh: float
    high_thresh: float
    mid_thresh: float
    new_track_thresh: float
    match_thresh: float
    confirm_streak: int
    confirm_score_thresh: float
    adaptive_confirmation: bool
    gmc_mode: str
    gmc_downscale: int
    per_seq_adapt: bool
    profile_stages: bool
    profile_frame_csv: bool
    latency_only: bool


@dataclass(frozen=True)
class DetectionView:
    """Detection and preprocessing parameters. Auto-projected from EvalConfig flat fields. Frozen."""

    gamma: float
    gamma_luma_threshold: float
    contrast: float
    tiling: str
    cross_tile_merge: bool
    cross_tile_score_penalty: float
    tile_diagnostics: bool
    tile_seam_score_penalty: float
    tile_seam_margin_canvas_px: float
    cross_tile_seam_center_scale: float
    cross_tile_seam_area_ratio_threshold: float
    cross_tile_seam_min_overlap_ratio: float
    person_class: int
    track_person_only: bool
    detector_box_format: str
    nms_iou_threshold: float
    private_continuation_enabled: bool
    private_candidate_nms_iou: float
    private_min_score: float
    private_max_candidates: int
    private_prior_iou_threshold: float
    private_prior_center_threshold: float
    private_prior_max_age: int
    private_low_stage_only: bool
    private_selection_mode: str
    private_energy_margin: float
    crowd_low_score_mode: bool
    crowd_low_score_trigger: int
    crowd_conf_threshold: float
    fp_hard_filter_enabled: bool
    fp_hard_filter_min_score: float
    fp_hard_filter_max_suspicious_area: int
    fp_hard_filter_max_suspicious_score: float
    external_fp_filter_mode: str
    external_fp_logistic_threshold: float
    external_fp_max_score: float
    external_fp_penalty: float
    external_fp_softmax_min_scale: float
    external_fp_logistic_model: str
    narrow_person_score_bonus: float
    narrow_person_max_width_ratio: float
    narrow_person_min_height_ratio: float
    narrow_person_min_aspect: float
    narrow_person_max_aspect: float
    per_frame_detection_cap: int
    detection_cap_rank_method: str
    adaptive_detection_cap: bool
    adaptive_cap_base: int
    adaptive_cap_max: int
    adaptive_cap_min: int
    temporal_consistency_min_frames: int
    scene_adapt_enabled: bool
    scene_adapt_window: int
    scene_adapt_crowd_thresh: float
    scene_adapt_narrow_aspect_thresh: float
    scene_adapt_narrow_width_thresh: float


@dataclass(frozen=True)
class GeometryView:
    """Geometry priors, OAO, Kalman, association. Auto-projected from EvalConfig flat fields. Frozen."""

    crowd_track_thresh: float
    crowd_mid_thresh: float
    crowd_new_track_thresh: float
    geometry_mid_scale: bool
    geometry_ref_height_ratio: float
    geometry_min_scale: float
    geometry_max_scale: float
    geometry_ema_beta: float
    geometry_loosen_step: float
    geometry_tighten_step: float
    geometry_min_samples: int
    id_stability_min_hits: int
    id_stability_min_iou: float
    id_stability_max_center_shift: float
    id_stability_max_gap: int
    id_stability_score_ema: float
    id_stability_min_score_ema: float
    person_geometry_prior: bool
    person_min_height_ratio: float
    person_min_aspect: float
    person_max_aspect: float
    person_min_area_ratio: float
    person_max_area_ratio: float
    detection_quality_scaling: bool
    detection_quality_w_aspect: float
    detection_quality_w_center: float
    detection_quality_w_area: float
    geometry_suspect_support: bool
    geometry_suspect_score: float
    kalman_adapt_mode: int
    kalman_r_scale: float
    vel_dir_weight: float
    fuse_score_weight: float
    stage2_match_thresh: float
    birth_low_score_thresh: float
    birth_prox_norm_thresh: float
    association_scoring_mode: str
    assoc_score_cost_w: float
    assoc_height_cost_w: float
    assoc_energy_diagnostics: bool
    oao_tau: float
    oao_contest_thresh: float
    oao_score_w: float
    oao_occ_mode: int
    oao_crowd_radius: float
    oao_height_gate: float
    oao_foot_gate: float
    oao_ramp_frames: float
    occ_state_enabled: bool
    occ_iou_thresh: float
    occ_foot_gap: float
    occ_ttl: int
    occ_cost_weight: float
    multiplicative_cost: bool
    sinkhorn_lambda: float
    stability_cost_w: float


@dataclass(frozen=True)
class MotionView:
    """Motion-based relinking parameters. Auto-projected from EvalConfig flat fields. Frozen."""

    vel_alpha: float
    acc_alpha: float
    min_motion_observations: int
    w_motion_iou: float
    motion_consistency_check: bool
    consistency_tol: float
    enable_motion_only: bool
    motion_only_lost_frames: int
    motion_only_iou_threshold: float
    motion_only_min_lost_frames: int


@dataclass(frozen=True)
class ReIDView:
    """ReID backbone and embedding parameters. Auto-projected from EvalConfig flat fields. Frozen."""

    reid_mode: str
    reid_model: str
    reid_interval: int
    reid_crop_mode: str
    reid_crop_padding: float
    reid_crop_layout: str
    lazy_reid_min_hit_streak: int
    lazy_reid_self_threshold: float
    async_reid: bool
    pipeline_relink: bool
    gmc_fg_mask: bool
    gmc_pcr_uncertain_thresh: float
    homography_root: str
    profile_lazy_reid_candidates: bool
    profile_lazy_reid_embeddings: bool
    exp_velocity_aligned_bank: bool


@dataclass(frozen=True)
class SemanticView:
    """Semantic relink and appearance bank. Auto-projected from EvalConfig flat fields. Frozen."""

    semantic_cheb_gr_claim: bool
    semantic_cheb_gr_max_cost: float
    semantic_cheb_gr_margin: float
    semantic_cheb_gr_min_head: int
    semantic_cheb_gr_pool_frac: float
    semantic_cheb_gr_min_sim: float
    semantic_gpu_relink_gate: bool | None
    semantic_gpu_relink_gate_graph: bool
    semantic_gpu_relink_gate_init_query_cap: int
    semantic_gpu_relink_gate_init_candidate_cap: int
    semantic_cheb_gr_graph_init_cap: int
    semantic_buffer_size: int
    semantic_min_consistency: float
    semantic_rerank_mode: str
    semantic_reciprocal_margin: float
    semantic_iou_weight: float
    semantic_mahalanobis_weight: float
    semantic_dynamic_margin_crowd: float
    semantic_dynamic_margin_age: float
    semantic_biometric_threshold: float
    semantic_w_sim_base: float
    semantic_w_iou_base: float
    semantic_w_maha_base: float
    semantic_shift_ambiguity: float
    semantic_shift_lost_age: float
    force_python_relinker: bool
    semantic_exp_density_gating: bool
    semantic_exp_density_k: float
    semantic_exp_density_eta: float
    semantic_bank_inject: bool
    appearance_bank_size: int
    appearance_bank_min_score: float
    appearance_bank_min_iou: float
    appearance_bank_consistency_threshold: float
    appearance_bank_high_quality_min_score: float
    appearance_bank_min_aspect: float
    appearance_bank_max_aspect: float
    appearance_occlusion_gate: bool
    appearance_occlusion_cov: float
    bank_quality_v2: bool
    bank_weighted_mean: bool
    bank_quality_w_det: float
    bank_quality_w_iou: float
    bank_quality_w_aspect: float
    bank_quality_w_center: float
    bank_quality_w_area: float
    semantic_clean_score_threshold: float
    semantic_clean_min_aspect: float
    semantic_clean_max_aspect: float
    semantic_clean_margin_ratio: float
    semantic_strict_sim_threshold: float


@dataclass(frozen=True)
class TriggerView:
    """Dynamic ReID trigger policy (Experimental). No fields in EvalConfig (experimental/KWARGS_DIRECT)."""

    pass


@dataclass(frozen=True)
class LifecycleView:
    """Track lifecycle, merge, interpolation. Auto-projected from EvalConfig flat fields. Frozen."""

    birth_quality_gate: bool
    birth_min_quality: float
    birth_quality_score_bias: float
    stage2_quality_gate: bool
    stage2_quality_min: float
    birth_consecutive_gate: bool
    birth_consecutive_frames: int
    birth_consecutive_iou: float
    birth_consecutive_boost: float
    birth_consecutive_min_score: float
    birth_consecutive_min_motion: float
    lifecycle_ttl: int
    lifecycle_min_gap: int
    lifecycle_spatial_gate: float
    lifecycle_min_iou: float
    lifecycle_sim_threshold: float
    lifecycle_require_embedding: bool
    lifecycle_ema: float
    post_lifecycle_merge: bool
    post_lifecycle_ttl: int
    post_lifecycle_min_gap: int
    post_lifecycle_velocity_samples: int
    post_lifecycle_spatial_weight: float
    post_lifecycle_motion_weight: float
    post_lifecycle_time_weight: float
    post_lifecycle_direction_weight: float
    post_lifecycle_max_cost: float
    post_lifecycle_appearance_gate: bool
    post_lifecycle_appearance_threshold: float
    post_lifecycle_appearance_min_samples: int
    post_lifecycle_appearance_max_samples: int
    post_lifecycle_appearance_min_score: float
    post_lifecycle_appearance_min_consistency: float
    post_lifecycle_appearance_weight: float
    post_lifecycle_gap_uncertainty_weight: float
    post_lifecycle_consistency_weight: float
    post_lifecycle_missing_appearance_cost: float
    cheb_gr_merge_enabled: bool
    cheb_gr_merge_max_cost: float
    cheb_gr_merge_max_gap: int
    cheb_gr_merge_min_overlap: int
    cheb_gr_merge_n_samples: int
    cheb_gr_pool_frac: float
    cheb_gr_lambda: float
    cheb_gr_k2: int
    cheb_gr_max_fwd: int
    cheb_gr_fuse_lambda: float
    cheb_gr_engine: str
    cheb_gr_model: str
    cheb_gr_online: bool
    cheb_gr_online_decide_n: int
    cheb_gr_online_max_cost: float
    cheb_gr_online_min_head: int
    cheb_gr_online_margin: float
    cheb_gr_online_key_sim_min: float
    cheb_gr_online_key_sim_cost_floor: float
    cheb_gr_online_key_margin_min: float
    cheb_gr_online_center_dist_veto: float
    cheb_gr_online_pollution_veto: float
    cheb_gr_online_neighbor_iou_max: float
    cheb_gr_online_bank_mode: str
    cheb_gr_online_bank_n: int
    cheb_gr_online_log: bool
    occ_audit: bool
    occ_audit_tau: float
    occ_audit_ref_n: int
    occ_audit_min_ref: int
    occ_audit_crops: int
    occ_audit_window: int
    occ_audit_min_occ: int
    occ_audit_log: bool
    track_buffer: int
    relink_enabled: bool
    relink_bank_cap: int
    relink_sim_thresh: float
    relink_lambda: float
    relink_spatial_gate: float
    relink_max_age: int
    relink_bridge_enabled: bool
    relink_bridge_px: float
    relink_bridge_at: int
    relink_bridge_min_lost: int
    relink_bridge_ttl: int
    relink_bridge_max_speed: float
    relink_bridge_person_height: float
    relink_bridge_fps: float
    relink_bridge_margin: float
    relink_bridge_spatial_gate: float
    relink_bridge_anchor: str
    relink_bridge_anchor_rate: float
    relink_bridge_h_lo: float
    relink_bridge_h_hi: float
    relink_bridge_dir_bonus: float
    relink_bridge_occ_gate_cover: float
    relink_bridge_occ_gap_min: int
    relink_bridge_occ_expand_px: float
    relink_bridge_occ_expand_cover: float
    relink_bridge_app_veto: float
    duplicate_suppression_iou_threshold: float
    duplicate_suppression_min_score_ratio: float
    multi_birth_enabled: bool
    multi_birth_min_score: float
    multi_birth_min_frames: int
    multi_birth_target_motion: float
    multi_birth_evidence_threshold: float
    multi_birth_iou_match: float
    multi_birth_ttl_frames: int
    multi_birth_w_score: float
    multi_birth_w_motion: float
    multi_birth_w_quality: float
    multi_birth_w_streak: float
    multi_birth_min_aspect: float
    multi_birth_max_area_px: int
    multi_birth_replace_mode: bool
    multi_birth_replace_evidence_threshold: float
    min_tracklet_len: int
    min_tracklet_score: float
    interpolate_tracklets: bool
    interpolate_max_gap: int
    interpolate_min_track_len: int
    interpolate_min_h: float


@dataclass
class EvalConfig:
    data_root: str
    split: str
    conf_threshold: float
    output_root: Path
    debug_dump_seq: str
    debug_dump_frames: str
    debug_dump_csv: str
    debug_birth_csv: str
    profile_stages: bool
    profile_frame_csv: bool
    latency_only: bool
    workbench: bool
    threads: int

    reid_mode: str
    reid_enabled: bool
    reid_model: str
    reid_engine: str
    reid_interval: int
    reid_crop_mode: str
    reid_crop_padding: float
    reid_crop_layout: str
    crop_hw: tuple[int, int]
    profile_lazy_reid_embeddings: bool
    profile_lazy_reid_candidates: bool
    reid_work_enabled: bool

    gmc_enabled: bool
    gmc_mode: str
    gmc_downscale: int
    gmc_fg_mask: bool
    gmc_pcr_uncertain_thresh: float
    homography_root: str

    semantic_buffer_size: int
    semantic_min_consistency: float
    semantic_rerank_mode: str
    semantic_reciprocal_margin: float
    semantic_bank_inject: bool
    semantic_iou_weight: float
    semantic_mahalanobis_weight: float
    semantic_dynamic_margin_crowd: float
    semantic_dynamic_margin_age: float
    semantic_biometric_threshold: float
    semantic_w_sim_base: float
    semantic_w_iou_base: float
    semantic_w_maha_base: float
    semantic_shift_ambiguity: float
    semantic_shift_lost_age: float
    semantic_clean_score_threshold: float
    semantic_clean_margin_ratio: float
    semantic_clean_min_aspect: float
    semantic_clean_max_aspect: float
    semantic_strict_sim_threshold: float
    semantic_exp_density_gating: bool
    semantic_exp_density_k: float
    semantic_exp_density_eta: float
    semantic_cheb_gr_claim: bool
    semantic_cheb_gr_max_cost: float
    semantic_cheb_gr_margin: float
    semantic_cheb_gr_min_head: int
    semantic_cheb_gr_pool_frac: float
    semantic_cheb_gr_min_sim: float
    semantic_gpu_relink_gate: bool | None
    semantic_gpu_relink_gate_graph: bool
    semantic_gpu_relink_gate_init_query_cap: int
    semantic_gpu_relink_gate_init_candidate_cap: int
    semantic_cheb_gr_graph_init_cap: int

    cross_tile_score_penalty: float
    tile_diagnostics: bool
    tile_seam_score_penalty: float
    tile_seam_margin_canvas_px: float
    cross_tile_seam_center_scale: float
    cross_tile_seam_area_ratio_threshold: float
    cross_tile_seam_min_overlap_ratio: float

    force_python_relinker: bool
    use_semantic_mode: bool
    use_tracker_reid: bool

    person_class: int
    track_person_only: bool
    detector_box_format: str
    track_thresh: float
    high_thresh: float
    match_thresh: float
    mid_thresh: float
    new_track_thresh: float
    confirm_streak: int
    confirm_score_thresh: float
    adaptive_confirmation: bool
    track_buffer: int
    birth_quality_gate: bool
    birth_min_quality: float
    birth_quality_score_bias: float
    stage2_quality_gate: bool
    stage2_quality_min: float
    birth_consecutive_gate: bool
    birth_consecutive_frames: int
    birth_consecutive_iou: float
    birth_consecutive_boost: float
    birth_consecutive_min_score: float
    birth_consecutive_min_motion: float
    exp_velocity_aligned_bank: bool

    # Duplicate suppression: remove near-duplicate detections within the same frame
    duplicate_suppression: bool
    duplicate_suppression_iou_threshold: float
    duplicate_suppression_min_score_ratio: float
    # Multi-signal birth policy (P5-1): joint evidence over score × streak × motion × geometry
    multi_birth_enabled: bool
    multi_birth_min_score: float
    multi_birth_min_frames: int
    multi_birth_target_motion: float
    multi_birth_evidence_threshold: float
    multi_birth_iou_match: float
    multi_birth_ttl_frames: int
    multi_birth_w_score: float
    multi_birth_w_motion: float
    multi_birth_w_quality: float
    multi_birth_w_streak: float
    multi_birth_min_aspect: float
    multi_birth_max_area_px: int
    # Multi-birth replace mode: suppress competing detection when evidence is high
    multi_birth_replace_mode: bool
    multi_birth_replace_evidence_threshold: float

    crowd_low_score_mode: bool
    crowd_low_score_trigger: int
    crowd_conf_threshold: float
    crowd_track_thresh: float
    crowd_mid_thresh: float
    crowd_new_track_thresh: float

    narrow_person_score_bonus: float
    narrow_person_max_width_ratio: float
    narrow_person_min_height_ratio: float
    narrow_person_min_aspect: float
    narrow_person_max_aspect: float

    tiling: str
    nms_iou_threshold: float
    cross_tile_merge: bool
    private_continuation_enabled: bool
    private_candidate_nms_iou: float
    private_min_score: float
    private_max_candidates: int
    private_prior_iou_threshold: float
    private_prior_center_threshold: float
    private_prior_max_age: int
    private_low_stage_only: bool
    private_selection_mode: str
    private_energy_margin: float

    geometry_mid_scale: bool
    geometry_ref_height_ratio: float
    geometry_min_scale: float
    geometry_max_scale: float
    geometry_ema_beta: float
    geometry_loosen_step: float
    geometry_tighten_step: float
    geometry_min_samples: int

    lazy_reid_min_hit_streak: int
    lazy_reid_self_threshold: float

    preprocess_modes: list[str]
    gamma: float
    gamma_luma_threshold: float
    contrast: float

    id_stability_filter_enabled: bool
    id_stability_min_hits: int
    id_stability_min_iou: float
    id_stability_max_center_shift: float
    id_stability_max_gap: int
    id_stability_score_ema: float
    id_stability_min_score_ema: float

    person_geometry_prior: bool
    person_min_height_ratio: float
    person_min_aspect: float
    person_max_aspect: float
    person_min_area_ratio: float
    person_max_area_ratio: float

    detection_quality_scaling: bool
    detection_quality_w_aspect: float
    detection_quality_w_center: float
    detection_quality_w_area: float

    geometry_suspect_support: bool
    geometry_suspect_score: float
    geometry_suspect_support_score: float

    reid_budget_raw: float

    lifecycle_merge_enabled: bool
    lifecycle_ttl: int
    lifecycle_min_gap: int
    lifecycle_spatial_gate: float
    lifecycle_min_iou: float
    lifecycle_sim_threshold: float
    lifecycle_require_embedding: bool
    lifecycle_ema: float

    post_lifecycle_merge: bool
    post_lifecycle_ttl: int
    post_lifecycle_min_gap: int
    post_lifecycle_velocity_samples: int
    post_lifecycle_spatial_weight: float
    post_lifecycle_motion_weight: float
    post_lifecycle_time_weight: float
    post_lifecycle_direction_weight: float
    post_lifecycle_max_cost: float
    post_lifecycle_appearance_gate: bool
    post_lifecycle_appearance_threshold: float
    post_lifecycle_appearance_min_samples: int
    post_lifecycle_appearance_max_samples: int
    post_lifecycle_appearance_min_score: float
    post_lifecycle_appearance_min_consistency: float
    post_lifecycle_appearance_weight: float
    post_lifecycle_gap_uncertainty_weight: float
    post_lifecycle_consistency_weight: float
    post_lifecycle_missing_appearance_cost: float

    # Cheb-GR offline tracklet merge (path 2; default off)
    cheb_gr_merge_enabled: bool
    cheb_gr_merge_max_cost: float
    cheb_gr_merge_max_gap: int
    cheb_gr_merge_min_overlap: int
    cheb_gr_merge_n_samples: int
    cheb_gr_pool_frac: float
    cheb_gr_lambda: float
    cheb_gr_k2: int
    cheb_gr_max_fwd: int
    cheb_gr_fuse_lambda: float
    cheb_gr_engine: str
    cheb_gr_model: str
    cheb_gr_online: bool
    cheb_gr_online_decide_n: int
    cheb_gr_online_max_cost: float
    cheb_gr_online_min_head: int
    cheb_gr_online_margin: float
    cheb_gr_online_key_sim_min: float
    cheb_gr_online_key_sim_cost_floor: float
    cheb_gr_online_key_margin_min: float
    cheb_gr_online_center_dist_veto: float
    cheb_gr_online_pollution_veto: float
    cheb_gr_online_neighbor_iou_max: float
    cheb_gr_online_bank_mode: str
    cheb_gr_online_bank_n: int
    cheb_gr_online_log: bool

    # Causal occ-exit identity audit (ABSORB-side twin of the online handover;
    # default off). Shares the cheb_gr extractor + visclean coverage rule.
    occ_audit: bool
    occ_audit_tau: float
    occ_audit_ref_n: int
    occ_audit_min_ref: int
    occ_audit_crops: int
    occ_audit_window: int
    occ_audit_min_occ: int
    occ_audit_log: bool
    occ_audit_bank_reference: bool
    occ_audit_bank_n: int

    # Birth-time lost-bank ReID relink (online, GPU; default off)
    relink_enabled: bool
    relink_bank_cap: int
    relink_sim_thresh: float
    relink_lambda: float
    relink_spatial_gate: float
    relink_max_age: int

    # GPU tracker-core bidirectional foot-bridge relink (Kalman-free; default off)
    relink_bridge_enabled: bool
    relink_bridge_px: float
    relink_bridge_at: int
    relink_bridge_min_lost: int
    relink_bridge_ttl: int
    relink_bridge_max_speed: float
    relink_bridge_person_height: float
    relink_bridge_fps: float
    relink_bridge_margin: float
    relink_bridge_spatial_gate: float
    relink_bridge_anchor: str
    relink_bridge_anchor_rate: float
    relink_bridge_h_lo: float
    relink_bridge_h_hi: float
    relink_bridge_dir_bonus: float
    relink_bridge_occ_gate_cover: float
    relink_bridge_occ_gap_min: int
    relink_bridge_occ_expand_px: float
    relink_bridge_occ_expand_cover: float
    relink_bridge_app_veto: float

    min_tracklet_len: int
    min_tracklet_score: float
    interpolate_tracklets: bool
    interpolate_max_gap: int
    interpolate_min_track_len: int
    interpolate_min_h: float
    kalman_adapt_mode: int
    kalman_r_scale: float
    vel_dir_weight: float
    occ_vel_weight: float
    fuse_score_weight: float
    stage2_match_thresh: float
    birth_low_score_thresh: float
    birth_prox_norm_thresh: float
    oao_tau: float
    oao_contest_thresh: float
    oao_score_w: float
    oao_occ_mode: int
    oao_crowd_radius: float
    oao_height_gate: float
    oao_foot_gate: float
    oao_ramp_frames: float

    # Depth-gated occlusion-state machine (Occluded(by=A)); default off.
    occ_state_enabled: bool
    occ_iou_thresh: float
    occ_foot_gap: float
    occ_ttl: int
    occ_cost_weight: float
    multiplicative_cost: bool
    sinkhorn_lambda: float
    stability_cost_w: float
    association_scoring_mode: str
    assoc_score_cost_w: float
    assoc_height_cost_w: float
    assoc_energy_diagnostics: bool

    # Motion-based relinking (Phase 2B: promoted from cfg.kwargs.get)
    vel_alpha: float
    acc_alpha: float
    min_motion_observations: int
    w_motion_iou: float
    motion_consistency_check: bool
    consistency_tol: float
    enable_motion_only: bool
    motion_only_lost_frames: int
    motion_only_iou_threshold: float
    motion_only_min_lost_frames: int

    # Temporal consistency filter
    temporal_consistency_min_frames: int
    # Per-frame detection cap
    per_frame_detection_cap: int
    # Detection cap ranking method: "score" | "quality" | "fp_filter" | "fp_filter_quality"
    detection_cap_rank_method: str
    # Adaptive detection cap (overrides per_frame_detection_cap when > 0)
    adaptive_detection_cap: bool
    adaptive_cap_base: int
    adaptive_cap_max: int
    adaptive_cap_min: int
    # FP hard filter: removes extremely suspicious low-score large-area detections
    fp_hard_filter_enabled: bool
    fp_hard_filter_min_score: float
    fp_hard_filter_max_suspicious_area: int
    fp_hard_filter_max_suspicious_score: float
    external_fp_filter_mode: str
    external_fp_logistic_model: str
    external_fp_logistic_threshold: float
    external_fp_max_score: float
    external_fp_penalty: float
    external_fp_softmax_min_scale: float
    appearance_bank_enabled: bool
    appearance_bank_size: int
    appearance_bank_min_score: float
    appearance_bank_min_iou: float
    appearance_bank_consistency_threshold: float
    appearance_bank_high_quality_min_score: float
    appearance_bank_min_aspect: float
    appearance_bank_max_aspect: float
    appearance_occlusion_gate: bool
    appearance_occlusion_cov: float

    bank_quality_v2: bool
    bank_quality_w_det: float
    bank_quality_w_iou: float
    bank_quality_w_aspect: float
    bank_quality_w_center: float
    bank_quality_w_area: float
    bank_weighted_mean: bool

    need_reid_enabled: bool
    async_reid: bool
    pipeline_relink: bool
    per_seq_adapt: bool
    seqs: list[str]
    kwargs: dict[str, Any]

    # Pose-guided box expansion
    pose_box_expand: bool
    pose_expand_ankle_conf: float
    pose_expand_margin: float
    pose_expand_flat_aspect: float

    # Scene-adaptive policy (P5-4)
    scene_adapt_enabled: bool
    scene_adapt_window: int
    scene_adapt_crowd_thresh: float
    scene_adapt_narrow_aspect_thresh: float
    scene_adapt_narrow_width_thresh: float

    def __post_init__(self) -> None:
        """Build module views as projections of flat fields.

        These are read-only typed views — they reflect the same values
        as the flat fields, not a second source of truth.
        """
        self.core = CoreView(
            workbench=self.workbench,
            threads=self.threads,
            data_root=self.data_root,
            split=self.split,
            debug_dump_seq=self.debug_dump_seq,
            debug_dump_frames=self.debug_dump_frames,
            debug_dump_csv=self.debug_dump_csv,
            debug_birth_csv=self.debug_birth_csv,
            conf_threshold=self.conf_threshold,
            track_thresh=self.track_thresh,
            high_thresh=self.high_thresh,
            mid_thresh=self.mid_thresh,
            new_track_thresh=self.new_track_thresh,
            match_thresh=self.match_thresh,
            confirm_streak=self.confirm_streak,
            confirm_score_thresh=self.confirm_score_thresh,
            adaptive_confirmation=self.adaptive_confirmation,
            gmc_mode=self.gmc_mode,
            gmc_downscale=self.gmc_downscale,
            per_seq_adapt=self.per_seq_adapt,
            profile_stages=self.profile_stages,
            profile_frame_csv=self.profile_frame_csv,
            latency_only=self.latency_only,
        )
        self.detection = DetectionView(
            gamma=self.gamma,
            gamma_luma_threshold=self.gamma_luma_threshold,
            contrast=self.contrast,
            tiling=self.tiling,
            cross_tile_merge=self.cross_tile_merge,
            cross_tile_score_penalty=self.cross_tile_score_penalty,
            tile_diagnostics=self.tile_diagnostics,
            tile_seam_score_penalty=self.tile_seam_score_penalty,
            tile_seam_margin_canvas_px=self.tile_seam_margin_canvas_px,
            cross_tile_seam_center_scale=self.cross_tile_seam_center_scale,
            cross_tile_seam_area_ratio_threshold=self.cross_tile_seam_area_ratio_threshold,
            cross_tile_seam_min_overlap_ratio=self.cross_tile_seam_min_overlap_ratio,
            person_class=self.person_class,
            track_person_only=self.track_person_only,
            detector_box_format=self.detector_box_format,
            nms_iou_threshold=self.nms_iou_threshold,
            private_continuation_enabled=self.private_continuation_enabled,
            private_candidate_nms_iou=self.private_candidate_nms_iou,
            private_min_score=self.private_min_score,
            private_max_candidates=self.private_max_candidates,
            private_prior_iou_threshold=self.private_prior_iou_threshold,
            private_prior_center_threshold=self.private_prior_center_threshold,
            private_prior_max_age=self.private_prior_max_age,
            private_low_stage_only=self.private_low_stage_only,
            private_selection_mode=self.private_selection_mode,
            private_energy_margin=self.private_energy_margin,
            crowd_low_score_mode=self.crowd_low_score_mode,
            crowd_low_score_trigger=self.crowd_low_score_trigger,
            crowd_conf_threshold=self.crowd_conf_threshold,
            fp_hard_filter_enabled=self.fp_hard_filter_enabled,
            fp_hard_filter_min_score=self.fp_hard_filter_min_score,
            fp_hard_filter_max_suspicious_area=self.fp_hard_filter_max_suspicious_area,
            fp_hard_filter_max_suspicious_score=self.fp_hard_filter_max_suspicious_score,
            external_fp_filter_mode=self.external_fp_filter_mode,
            external_fp_logistic_threshold=self.external_fp_logistic_threshold,
            external_fp_max_score=self.external_fp_max_score,
            external_fp_penalty=self.external_fp_penalty,
            external_fp_softmax_min_scale=self.external_fp_softmax_min_scale,
            external_fp_logistic_model=self.external_fp_logistic_model,
            narrow_person_score_bonus=self.narrow_person_score_bonus,
            narrow_person_max_width_ratio=self.narrow_person_max_width_ratio,
            narrow_person_min_height_ratio=self.narrow_person_min_height_ratio,
            narrow_person_min_aspect=self.narrow_person_min_aspect,
            narrow_person_max_aspect=self.narrow_person_max_aspect,
            per_frame_detection_cap=self.per_frame_detection_cap,
            detection_cap_rank_method=self.detection_cap_rank_method,
            adaptive_detection_cap=self.adaptive_detection_cap,
            adaptive_cap_base=self.adaptive_cap_base,
            adaptive_cap_max=self.adaptive_cap_max,
            adaptive_cap_min=self.adaptive_cap_min,
            temporal_consistency_min_frames=self.temporal_consistency_min_frames,
            scene_adapt_enabled=self.scene_adapt_enabled,
            scene_adapt_window=self.scene_adapt_window,
            scene_adapt_crowd_thresh=self.scene_adapt_crowd_thresh,
            scene_adapt_narrow_aspect_thresh=self.scene_adapt_narrow_aspect_thresh,
            scene_adapt_narrow_width_thresh=self.scene_adapt_narrow_width_thresh,
        )
        self.geometry = GeometryView(
            crowd_track_thresh=self.crowd_track_thresh,
            crowd_mid_thresh=self.crowd_mid_thresh,
            crowd_new_track_thresh=self.crowd_new_track_thresh,
            geometry_mid_scale=self.geometry_mid_scale,
            geometry_ref_height_ratio=self.geometry_ref_height_ratio,
            geometry_min_scale=self.geometry_min_scale,
            geometry_max_scale=self.geometry_max_scale,
            geometry_ema_beta=self.geometry_ema_beta,
            geometry_loosen_step=self.geometry_loosen_step,
            geometry_tighten_step=self.geometry_tighten_step,
            geometry_min_samples=self.geometry_min_samples,
            id_stability_min_hits=self.id_stability_min_hits,
            id_stability_min_iou=self.id_stability_min_iou,
            id_stability_max_center_shift=self.id_stability_max_center_shift,
            id_stability_max_gap=self.id_stability_max_gap,
            id_stability_score_ema=self.id_stability_score_ema,
            id_stability_min_score_ema=self.id_stability_min_score_ema,
            person_geometry_prior=self.person_geometry_prior,
            person_min_height_ratio=self.person_min_height_ratio,
            person_min_aspect=self.person_min_aspect,
            person_max_aspect=self.person_max_aspect,
            person_min_area_ratio=self.person_min_area_ratio,
            person_max_area_ratio=self.person_max_area_ratio,
            detection_quality_scaling=self.detection_quality_scaling,
            detection_quality_w_aspect=self.detection_quality_w_aspect,
            detection_quality_w_center=self.detection_quality_w_center,
            detection_quality_w_area=self.detection_quality_w_area,
            geometry_suspect_support=self.geometry_suspect_support,
            geometry_suspect_score=self.geometry_suspect_score,
            kalman_adapt_mode=self.kalman_adapt_mode,
            kalman_r_scale=self.kalman_r_scale,
            vel_dir_weight=self.vel_dir_weight,
            fuse_score_weight=self.fuse_score_weight,
            stage2_match_thresh=self.stage2_match_thresh,
            birth_low_score_thresh=self.birth_low_score_thresh,
            birth_prox_norm_thresh=self.birth_prox_norm_thresh,
            association_scoring_mode=self.association_scoring_mode,
            assoc_score_cost_w=self.assoc_score_cost_w,
            assoc_height_cost_w=self.assoc_height_cost_w,
            assoc_energy_diagnostics=self.assoc_energy_diagnostics,
            oao_tau=self.oao_tau,
            oao_contest_thresh=self.oao_contest_thresh,
            oao_score_w=self.oao_score_w,
            oao_occ_mode=self.oao_occ_mode,
            oao_crowd_radius=self.oao_crowd_radius,
            oao_height_gate=self.oao_height_gate,
            oao_foot_gate=self.oao_foot_gate,
            oao_ramp_frames=self.oao_ramp_frames,
            occ_state_enabled=self.occ_state_enabled,
            occ_iou_thresh=self.occ_iou_thresh,
            occ_foot_gap=self.occ_foot_gap,
            occ_ttl=self.occ_ttl,
            occ_cost_weight=self.occ_cost_weight,
            multiplicative_cost=self.multiplicative_cost,
            sinkhorn_lambda=self.sinkhorn_lambda,
            stability_cost_w=self.stability_cost_w,
        )
        self.motion = MotionView(
            vel_alpha=self.vel_alpha,
            acc_alpha=self.acc_alpha,
            min_motion_observations=self.min_motion_observations,
            w_motion_iou=self.w_motion_iou,
            motion_consistency_check=self.motion_consistency_check,
            consistency_tol=self.consistency_tol,
            enable_motion_only=self.enable_motion_only,
            motion_only_lost_frames=self.motion_only_lost_frames,
            motion_only_iou_threshold=self.motion_only_iou_threshold,
            motion_only_min_lost_frames=self.motion_only_min_lost_frames,
        )
        self.reid = ReIDView(
            reid_mode=self.reid_mode,
            reid_model=self.reid_model,
            reid_interval=self.reid_interval,
            reid_crop_mode=self.reid_crop_mode,
            reid_crop_padding=self.reid_crop_padding,
            reid_crop_layout=self.reid_crop_layout,
            lazy_reid_min_hit_streak=self.lazy_reid_min_hit_streak,
            lazy_reid_self_threshold=self.lazy_reid_self_threshold,
            async_reid=self.async_reid,
            pipeline_relink=self.pipeline_relink,
            gmc_fg_mask=self.gmc_fg_mask,
            gmc_pcr_uncertain_thresh=self.gmc_pcr_uncertain_thresh,
            homography_root=self.homography_root,
            profile_lazy_reid_candidates=self.profile_lazy_reid_candidates,
            profile_lazy_reid_embeddings=self.profile_lazy_reid_embeddings,
            exp_velocity_aligned_bank=self.exp_velocity_aligned_bank,
        )
        self.semantic = SemanticView(
            semantic_cheb_gr_claim=self.semantic_cheb_gr_claim,
            semantic_cheb_gr_max_cost=self.semantic_cheb_gr_max_cost,
            semantic_cheb_gr_margin=self.semantic_cheb_gr_margin,
            semantic_cheb_gr_min_head=self.semantic_cheb_gr_min_head,
            semantic_cheb_gr_pool_frac=self.semantic_cheb_gr_pool_frac,
            semantic_cheb_gr_min_sim=self.semantic_cheb_gr_min_sim,
            semantic_gpu_relink_gate=self.semantic_gpu_relink_gate,
            semantic_gpu_relink_gate_graph=self.semantic_gpu_relink_gate_graph,
            semantic_gpu_relink_gate_init_query_cap=self.semantic_gpu_relink_gate_init_query_cap,
            semantic_gpu_relink_gate_init_candidate_cap=self.semantic_gpu_relink_gate_init_candidate_cap,
            semantic_cheb_gr_graph_init_cap=self.semantic_cheb_gr_graph_init_cap,
            semantic_buffer_size=self.semantic_buffer_size,
            semantic_min_consistency=self.semantic_min_consistency,
            semantic_rerank_mode=self.semantic_rerank_mode,
            semantic_reciprocal_margin=self.semantic_reciprocal_margin,
            semantic_iou_weight=self.semantic_iou_weight,
            semantic_mahalanobis_weight=self.semantic_mahalanobis_weight,
            semantic_dynamic_margin_crowd=self.semantic_dynamic_margin_crowd,
            semantic_dynamic_margin_age=self.semantic_dynamic_margin_age,
            semantic_biometric_threshold=self.semantic_biometric_threshold,
            semantic_w_sim_base=self.semantic_w_sim_base,
            semantic_w_iou_base=self.semantic_w_iou_base,
            semantic_w_maha_base=self.semantic_w_maha_base,
            semantic_shift_ambiguity=self.semantic_shift_ambiguity,
            semantic_shift_lost_age=self.semantic_shift_lost_age,
            force_python_relinker=self.force_python_relinker,
            semantic_exp_density_gating=self.semantic_exp_density_gating,
            semantic_exp_density_k=self.semantic_exp_density_k,
            semantic_exp_density_eta=self.semantic_exp_density_eta,
            semantic_bank_inject=self.semantic_bank_inject,
            appearance_bank_size=self.appearance_bank_size,
            appearance_bank_min_score=self.appearance_bank_min_score,
            appearance_bank_min_iou=self.appearance_bank_min_iou,
            appearance_bank_consistency_threshold=self.appearance_bank_consistency_threshold,
            appearance_bank_high_quality_min_score=self.appearance_bank_high_quality_min_score,
            appearance_bank_min_aspect=self.appearance_bank_min_aspect,
            appearance_bank_max_aspect=self.appearance_bank_max_aspect,
            appearance_occlusion_gate=self.appearance_occlusion_gate,
            appearance_occlusion_cov=self.appearance_occlusion_cov,
            bank_quality_v2=self.bank_quality_v2,
            bank_weighted_mean=self.bank_weighted_mean,
            bank_quality_w_det=self.bank_quality_w_det,
            bank_quality_w_iou=self.bank_quality_w_iou,
            bank_quality_w_aspect=self.bank_quality_w_aspect,
            bank_quality_w_center=self.bank_quality_w_center,
            bank_quality_w_area=self.bank_quality_w_area,
            semantic_clean_score_threshold=self.semantic_clean_score_threshold,
            semantic_clean_min_aspect=self.semantic_clean_min_aspect,
            semantic_clean_max_aspect=self.semantic_clean_max_aspect,
            semantic_clean_margin_ratio=self.semantic_clean_margin_ratio,
            semantic_strict_sim_threshold=self.semantic_strict_sim_threshold,
        )
        # TriggerView has no fields in EvalConfig — skip
        self.lifecycle = LifecycleView(
            birth_quality_gate=self.birth_quality_gate,
            birth_min_quality=self.birth_min_quality,
            birth_quality_score_bias=self.birth_quality_score_bias,
            stage2_quality_gate=self.stage2_quality_gate,
            stage2_quality_min=self.stage2_quality_min,
            birth_consecutive_gate=self.birth_consecutive_gate,
            birth_consecutive_frames=self.birth_consecutive_frames,
            birth_consecutive_iou=self.birth_consecutive_iou,
            birth_consecutive_boost=self.birth_consecutive_boost,
            birth_consecutive_min_score=self.birth_consecutive_min_score,
            birth_consecutive_min_motion=self.birth_consecutive_min_motion,
            lifecycle_ttl=self.lifecycle_ttl,
            lifecycle_min_gap=self.lifecycle_min_gap,
            lifecycle_spatial_gate=self.lifecycle_spatial_gate,
            lifecycle_min_iou=self.lifecycle_min_iou,
            lifecycle_sim_threshold=self.lifecycle_sim_threshold,
            lifecycle_require_embedding=self.lifecycle_require_embedding,
            lifecycle_ema=self.lifecycle_ema,
            post_lifecycle_merge=self.post_lifecycle_merge,
            post_lifecycle_ttl=self.post_lifecycle_ttl,
            post_lifecycle_min_gap=self.post_lifecycle_min_gap,
            post_lifecycle_velocity_samples=self.post_lifecycle_velocity_samples,
            post_lifecycle_spatial_weight=self.post_lifecycle_spatial_weight,
            post_lifecycle_motion_weight=self.post_lifecycle_motion_weight,
            post_lifecycle_time_weight=self.post_lifecycle_time_weight,
            post_lifecycle_direction_weight=self.post_lifecycle_direction_weight,
            post_lifecycle_max_cost=self.post_lifecycle_max_cost,
            post_lifecycle_appearance_gate=self.post_lifecycle_appearance_gate,
            post_lifecycle_appearance_threshold=self.post_lifecycle_appearance_threshold,
            post_lifecycle_appearance_min_samples=self.post_lifecycle_appearance_min_samples,
            post_lifecycle_appearance_max_samples=self.post_lifecycle_appearance_max_samples,
            post_lifecycle_appearance_min_score=self.post_lifecycle_appearance_min_score,
            post_lifecycle_appearance_min_consistency=self.post_lifecycle_appearance_min_consistency,
            post_lifecycle_appearance_weight=self.post_lifecycle_appearance_weight,
            post_lifecycle_gap_uncertainty_weight=self.post_lifecycle_gap_uncertainty_weight,
            post_lifecycle_consistency_weight=self.post_lifecycle_consistency_weight,
            post_lifecycle_missing_appearance_cost=self.post_lifecycle_missing_appearance_cost,
            cheb_gr_merge_enabled=self.cheb_gr_merge_enabled,
            cheb_gr_merge_max_cost=self.cheb_gr_merge_max_cost,
            cheb_gr_merge_max_gap=self.cheb_gr_merge_max_gap,
            cheb_gr_merge_min_overlap=self.cheb_gr_merge_min_overlap,
            cheb_gr_merge_n_samples=self.cheb_gr_merge_n_samples,
            cheb_gr_pool_frac=self.cheb_gr_pool_frac,
            cheb_gr_lambda=self.cheb_gr_lambda,
            cheb_gr_k2=self.cheb_gr_k2,
            cheb_gr_max_fwd=self.cheb_gr_max_fwd,
            cheb_gr_fuse_lambda=self.cheb_gr_fuse_lambda,
            cheb_gr_engine=self.cheb_gr_engine,
            cheb_gr_model=self.cheb_gr_model,
            cheb_gr_online=self.cheb_gr_online,
            cheb_gr_online_decide_n=self.cheb_gr_online_decide_n,
            cheb_gr_online_max_cost=self.cheb_gr_online_max_cost,
            cheb_gr_online_min_head=self.cheb_gr_online_min_head,
            cheb_gr_online_margin=self.cheb_gr_online_margin,
            cheb_gr_online_key_sim_min=self.cheb_gr_online_key_sim_min,
            cheb_gr_online_key_sim_cost_floor=self.cheb_gr_online_key_sim_cost_floor,
            cheb_gr_online_key_margin_min=self.cheb_gr_online_key_margin_min,
            cheb_gr_online_center_dist_veto=self.cheb_gr_online_center_dist_veto,
            cheb_gr_online_pollution_veto=self.cheb_gr_online_pollution_veto,
            cheb_gr_online_neighbor_iou_max=self.cheb_gr_online_neighbor_iou_max,
            cheb_gr_online_bank_mode=self.cheb_gr_online_bank_mode,
            cheb_gr_online_bank_n=self.cheb_gr_online_bank_n,
            cheb_gr_online_log=self.cheb_gr_online_log,
            occ_audit=self.occ_audit,
            occ_audit_tau=self.occ_audit_tau,
            occ_audit_ref_n=self.occ_audit_ref_n,
            occ_audit_min_ref=self.occ_audit_min_ref,
            occ_audit_crops=self.occ_audit_crops,
            occ_audit_window=self.occ_audit_window,
            occ_audit_min_occ=self.occ_audit_min_occ,
            occ_audit_log=self.occ_audit_log,
            track_buffer=self.track_buffer,
            relink_enabled=self.relink_enabled,
            relink_bank_cap=self.relink_bank_cap,
            relink_sim_thresh=self.relink_sim_thresh,
            relink_lambda=self.relink_lambda,
            relink_spatial_gate=self.relink_spatial_gate,
            relink_max_age=self.relink_max_age,
            relink_bridge_enabled=self.relink_bridge_enabled,
            relink_bridge_px=self.relink_bridge_px,
            relink_bridge_at=self.relink_bridge_at,
            relink_bridge_min_lost=self.relink_bridge_min_lost,
            relink_bridge_ttl=self.relink_bridge_ttl,
            relink_bridge_max_speed=self.relink_bridge_max_speed,
            relink_bridge_person_height=self.relink_bridge_person_height,
            relink_bridge_fps=self.relink_bridge_fps,
            relink_bridge_margin=self.relink_bridge_margin,
            relink_bridge_spatial_gate=self.relink_bridge_spatial_gate,
            relink_bridge_anchor=self.relink_bridge_anchor,
            relink_bridge_anchor_rate=self.relink_bridge_anchor_rate,
            relink_bridge_h_lo=self.relink_bridge_h_lo,
            relink_bridge_h_hi=self.relink_bridge_h_hi,
            relink_bridge_dir_bonus=self.relink_bridge_dir_bonus,
            relink_bridge_occ_gate_cover=self.relink_bridge_occ_gate_cover,
            relink_bridge_occ_gap_min=self.relink_bridge_occ_gap_min,
            relink_bridge_occ_expand_px=self.relink_bridge_occ_expand_px,
            relink_bridge_occ_expand_cover=self.relink_bridge_occ_expand_cover,
            relink_bridge_app_veto=self.relink_bridge_app_veto,
            duplicate_suppression_iou_threshold=self.duplicate_suppression_iou_threshold,
            duplicate_suppression_min_score_ratio=self.duplicate_suppression_min_score_ratio,
            multi_birth_enabled=self.multi_birth_enabled,
            multi_birth_min_score=self.multi_birth_min_score,
            multi_birth_min_frames=self.multi_birth_min_frames,
            multi_birth_target_motion=self.multi_birth_target_motion,
            multi_birth_evidence_threshold=self.multi_birth_evidence_threshold,
            multi_birth_iou_match=self.multi_birth_iou_match,
            multi_birth_ttl_frames=self.multi_birth_ttl_frames,
            multi_birth_w_score=self.multi_birth_w_score,
            multi_birth_w_motion=self.multi_birth_w_motion,
            multi_birth_w_quality=self.multi_birth_w_quality,
            multi_birth_w_streak=self.multi_birth_w_streak,
            multi_birth_min_aspect=self.multi_birth_min_aspect,
            multi_birth_max_area_px=self.multi_birth_max_area_px,
            multi_birth_replace_mode=self.multi_birth_replace_mode,
            multi_birth_replace_evidence_threshold=self.multi_birth_replace_evidence_threshold,
            min_tracklet_len=self.min_tracklet_len,
            min_tracklet_score=self.min_tracklet_score,
            interpolate_tracklets=self.interpolate_tracklets,
            interpolate_max_gap=self.interpolate_max_gap,
            interpolate_min_track_len=self.interpolate_min_track_len,
            interpolate_min_h=self.interpolate_min_h,
        )


# ---------------------------------------------------------------------------
# Phase 2 alias resolution
#
# ALIAS_MAP: canonical_name → [legacy_alias_1, legacy_alias_2, ...]
# _resolve_alias(kwargs, canonical, default, coerce=None) resolves a value
# from kwargs, checking the canonical key first, then legacy aliases.
# If multiple keys are present with different values, it raises ValueError.
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, list[str]] = {
    # Phase 2B: motion params consumed with motion_ prefix in pipeline.py
    "vel_alpha": ["motion_vel_alpha"],
    "acc_alpha": ["motion_acc_alpha"],
    "min_motion_observations": ["motion_min_observations"],
    "w_motion_iou": ["motion_w_iou"],
    "consistency_tol": ["motion_consistency_tol"],
    "enable_motion_only": ["motion_enable_motion_only"],
    "motion_only_lost_frames": ["motion_motion_only_lost_frames"],
    "motion_only_iou_threshold": ["motion_motion_only_iou_threshold"],
    "motion_only_min_lost_frames": ["motion_motion_only_min_lost_frames"],
    # Phase 2C: NAME_MAP entries (argparse dest → EvalConfig field)
    "gmc_enabled": ["gmc"],
    "id_stability_filter_enabled": ["id_stability_filter"],
    "lifecycle_merge_enabled": ["lifecycle_merge"],
    "need_reid_enabled": ["need_reid"],
    "appearance_bank_enabled": ["appearance_bank"],
    "appearance_bank": [],  # canonical is the short form
    "duplicate_suppression": ["duplicate_suppression_enabled"],
    "preprocess_modes": ["preprocess"],
    "reid_budget_raw": ["reid_budget"],
    "reid_engine": ["reid_engine_path"],
}


def _resolve_alias(
    kwargs: dict[str, Any],
    canonical: str,
    default: Any,
    coerce: type | None = None,
) -> Any:
    """Resolve a config value supporting both canonical and legacy alias keys.

    Checks canonical key first, then legacy aliases from ALIAS_MAP.
    Raises ValueError if multiple present keys have conflicting values.
    """
    candidates = [canonical] + ALIAS_MAP.get(canonical, [])
    present: dict[str, Any] = {}
    for key in candidates:
        if key in kwargs:
            val = kwargs[key]
            if coerce is not None:
                val = coerce(val)
            present[key] = val

    if len(present) == 0:
        return default

    values = list(present.values())
    if len(present) > 1:
        first_val = values[0]
        for key, val in present.items():
            if val != first_val:
                raise ValueError(
                    f"Conflicting config values for '{canonical}': "
                    f"{', '.join(f'{k}={v!r}' for k, v in present.items())}"
                )

    return values[0]


# ---------------------------------------------------------------------------
# Phase 4A: Auto-resolve default registry.
#
# _DEFAULTS maps each EvalConfig field name to its default value.
# Generated from the 8 module dataclasses in scripts/eval/config/.
# Phase 3 verified: dataclass defaults == argparse defaults (0 mismatches).
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "data_root": "",
    "split": "",
    "conf_threshold": 0.05,
    "output_root": None,
    "debug_dump_seq": "",
    "debug_dump_frames": "",
    "debug_dump_csv": "",
    "debug_birth_csv": "",
    "profile_stages": False,
    "profile_frame_csv": False,
    "latency_only": False,
    "workbench": False,
    "threads": 1,
    "reid_mode": "off",
    "reid_enabled": False,
    "reid_model": "siglip2",
    "reid_engine": "",
    "reid_interval": 20,
    "reid_crop_mode": "tight",
    "reid_crop_padding": 0.0,
    "reid_crop_layout": "full",
    "crop_hw": None,
    "profile_lazy_reid_embeddings": False,
    "profile_lazy_reid_candidates": False,
    "reid_work_enabled": False,
    "gmc_enabled": True,
    "gmc_mode": "gpu",
    "gmc_downscale": 8,
    "gmc_fg_mask": False,
    "gmc_pcr_uncertain_thresh": 8.0,
    "homography_root": "",
    "semantic_buffer_size": 10,
    "semantic_min_consistency": 0.0,
    "semantic_rerank_mode": "mean",
    "semantic_reciprocal_margin": 0.0,
    "semantic_bank_inject": False,
    "semantic_iou_weight": 0.0,
    "semantic_mahalanobis_weight": 0.0,
    "semantic_dynamic_margin_crowd": 0.0,
    "semantic_dynamic_margin_age": 0.0,
    "semantic_biometric_threshold": 0.0,
    "semantic_w_sim_base": 0.8,
    "semantic_w_iou_base": 0.34,
    "semantic_w_maha_base": 0.31,
    "semantic_shift_ambiguity": 0.34,
    "semantic_shift_lost_age": 0.18,
    "semantic_clean_score_threshold": 0.65,
    "semantic_clean_margin_ratio": 0.0,
    "semantic_clean_min_aspect": 1.2,
    "semantic_clean_max_aspect": 4.5,
    "semantic_strict_sim_threshold": 0.0,
    "semantic_exp_density_gating": False,
    "semantic_exp_density_k": 2.0,
    "semantic_exp_density_eta": 0.15,
    "semantic_cheb_gr_claim": False,
    "semantic_cheb_gr_max_cost": 0.45,
    "semantic_cheb_gr_margin": 0.05,
    "semantic_cheb_gr_min_head": 2,
    "semantic_cheb_gr_pool_frac": 0.3,
    "semantic_cheb_gr_min_sim": 0.0,
    "semantic_gpu_relink_gate": None,
    "semantic_gpu_relink_gate_graph": True,
    "semantic_gpu_relink_gate_init_query_cap": 64,
    "semantic_gpu_relink_gate_init_candidate_cap": 128,
    "semantic_cheb_gr_graph_init_cap": 32,
    "cross_tile_score_penalty": 1.0,
    "tile_diagnostics": False,
    "tile_seam_score_penalty": 1.0,
    "tile_seam_margin_canvas_px": 24.0,
    "cross_tile_seam_center_scale": 1.8,
    "cross_tile_seam_area_ratio_threshold": 0.30,
    "cross_tile_seam_min_overlap_ratio": 0.45,
    "force_python_relinker": False,
    "use_semantic_mode": False,
    "use_tracker_reid": False,
    "person_class": 0,
    "track_person_only": True,
    "detector_box_format": "xyxy",
    "track_thresh": 0.05,
    "high_thresh": 0.45,
    "match_thresh": 0.75,
    "mid_thresh": 0.10,
    "new_track_thresh": 0.35,
    "confirm_streak": 1,
    "confirm_score_thresh": 0.0,
    "adaptive_confirmation": False,
    "track_buffer": 30,
    "birth_quality_gate": False,
    "birth_min_quality": 0.0,
    "birth_quality_score_bias": 0.15,
    "stage2_quality_gate": False,
    "stage2_quality_min": 0.40,
    "birth_consecutive_gate": False,
    "birth_consecutive_frames": 2,
    "birth_consecutive_iou": 0.40,
    "birth_consecutive_boost": 0.05,
    "birth_consecutive_min_score": 0.20,
    "birth_consecutive_min_motion": 0.0,
    "exp_velocity_aligned_bank": False,
    "duplicate_suppression": False,
    "duplicate_suppression_iou_threshold": 0.85,
    "duplicate_suppression_min_score_ratio": 1.05,
    "multi_birth_enabled": False,
    "multi_birth_min_score": 0.12,
    "multi_birth_min_frames": 3,
    "multi_birth_target_motion": 12.0,
    "multi_birth_evidence_threshold": 0.60,
    "multi_birth_iou_match": 0.30,
    "multi_birth_ttl_frames": 5,
    "multi_birth_w_score": 0.35,
    "multi_birth_w_motion": 0.30,
    "multi_birth_w_quality": 0.20,
    "multi_birth_w_streak": 0.15,
    "multi_birth_min_aspect": 0.0,
    "multi_birth_max_area_px": 0,
    "multi_birth_replace_mode": False,
    "multi_birth_replace_evidence_threshold": 0.85,
    "crowd_low_score_mode": False,
    "crowd_low_score_trigger": 25,
    "crowd_conf_threshold": 0.02,
    "crowd_track_thresh": 0.02,
    "crowd_mid_thresh": 0.05,
    "crowd_new_track_thresh": 0.25,
    "narrow_person_score_bonus": 0.0,
    "narrow_person_max_width_ratio": 0.018,
    "narrow_person_min_height_ratio": 0.045,
    "narrow_person_min_aspect": 2.0,
    "narrow_person_max_aspect": 4.8,
    "tiling": "native_960",
    "nms_iou_threshold": None,
    "cross_tile_merge": True,
    "private_continuation_enabled": False,
    "private_candidate_nms_iou": 0.70,
    "private_min_score": 0.25,
    "private_max_candidates": 0,
    "private_prior_iou_threshold": 0.0,
    "private_prior_center_threshold": 0.0,
    "private_prior_max_age": 2,
    "private_low_stage_only": False,
    "private_selection_mode": "global",
    "private_energy_margin": 0.0,
    "geometry_mid_scale": False,
    "geometry_ref_height_ratio": REF_HEIGHT_RATIO,
    "geometry_min_scale": 0.875,
    "geometry_max_scale": 1.20,
    "geometry_ema_beta": 0.80,
    "geometry_loosen_step": 0.08,
    "geometry_tighten_step": 0.03,
    "geometry_min_samples": 5,
    "lazy_reid_min_hit_streak": 2,
    "lazy_reid_self_threshold": 0.85,
    "preprocess_modes": [],
    "gamma": 0.8,
    "gamma_luma_threshold": 0.35,
    "contrast": 1.2,
    "id_stability_filter_enabled": True,
    "id_stability_min_hits": 2,
    "id_stability_min_iou": 0.05,
    "id_stability_max_center_shift": 2.0,
    "id_stability_max_gap": 1,
    "id_stability_score_ema": 0.70,
    "id_stability_min_score_ema": 0.15,
    "person_geometry_prior": True,
    "person_min_height_ratio": 0.018,
    "person_min_aspect": 1.0,
    "person_max_aspect": 5.5,
    "person_min_area_ratio": 0.00006,
    "person_max_area_ratio": 0.0,
    "detection_quality_scaling": True,
    "detection_quality_w_aspect": 0.50,
    "detection_quality_w_center": 0.30,
    "detection_quality_w_area": 0.20,
    "geometry_suspect_support": True,
    "geometry_suspect_score": 0.0,
    "geometry_suspect_support_score": 0.0,
    "reid_budget_raw": 0.2,
    "lifecycle_merge_enabled": False,
    "lifecycle_ttl": 45,
    "lifecycle_min_gap": 2,
    "lifecycle_spatial_gate": 0.08,
    "lifecycle_min_iou": 0.0,
    "lifecycle_sim_threshold": 0.90,
    "lifecycle_require_embedding": False,
    "lifecycle_ema": 0.83,
    "post_lifecycle_merge": False,
    "post_lifecycle_ttl": 60,
    "post_lifecycle_min_gap": 1,
    "post_lifecycle_velocity_samples": 5,
    "post_lifecycle_spatial_weight": 0.35,
    "post_lifecycle_motion_weight": 0.45,
    "post_lifecycle_time_weight": 0.10,
    "post_lifecycle_direction_weight": 0.25,
    "post_lifecycle_max_cost": 1.25,
    "post_lifecycle_appearance_gate": False,
    "post_lifecycle_appearance_threshold": 0.90,
    "post_lifecycle_appearance_min_samples": 1,
    "post_lifecycle_appearance_max_samples": 5,
    "post_lifecycle_appearance_min_score": 0.0,
    "post_lifecycle_appearance_min_consistency": 0.0,
    "post_lifecycle_appearance_weight": 0.0,
    "post_lifecycle_gap_uncertainty_weight": 0.0,
    "post_lifecycle_consistency_weight": 0.0,
    "post_lifecycle_missing_appearance_cost": 0.5,
    "cheb_gr_merge_enabled": False,
    "cheb_gr_merge_max_cost": 0.55,
    "cheb_gr_merge_max_gap": 60,
    "cheb_gr_merge_min_overlap": 1,
    "cheb_gr_merge_n_samples": 50,
    "cheb_gr_pool_frac": 0.3,
    "cheb_gr_lambda": 2.0,
    "cheb_gr_k2": 6,
    "cheb_gr_max_fwd": 50,
    "cheb_gr_fuse_lambda": 0.3,
    "cheb_gr_engine": "",
    "cheb_gr_model": "siglip2_reid",
    "cheb_gr_online": False,
    "cheb_gr_online_decide_n": 5,
    "cheb_gr_online_max_cost": 0.45,
    "cheb_gr_online_min_head": 1,
    "cheb_gr_online_margin": 0.0,
    "cheb_gr_online_key_sim_min": 0.0,
    "cheb_gr_online_key_sim_cost_floor": 0.0,
    "cheb_gr_online_key_margin_min": 0.0,
    "cheb_gr_online_center_dist_veto": 0.0,
    "cheb_gr_online_pollution_veto": 0.0,
    "cheb_gr_online_neighbor_iou_max": 0.0,
    "cheb_gr_online_bank_mode": "spread",
    "cheb_gr_online_bank_n": 0,
    "cheb_gr_online_log": False,
    "occ_audit": False,
    "occ_audit_tau": 0.45,
    "occ_audit_ref_n": 5,
    "occ_audit_min_ref": 2,
    "occ_audit_crops": 3,
    "occ_audit_window": 30,
    "occ_audit_min_occ": 2,
    "occ_audit_log": False,
    "occ_audit_bank_reference": False,
    "occ_audit_bank_n": 20,
    "relink_enabled": False,
    "relink_bank_cap": 256,
    "relink_sim_thresh": 0.6,
    "relink_lambda": 2.5,
    "relink_spatial_gate": 4.0,
    "relink_max_age": 300,
    "relink_bridge_enabled": False,
    "relink_bridge_px": 0.25,
    "relink_bridge_at": 4,
    "relink_bridge_min_lost": 2,
    "relink_bridge_ttl": 120,
    "relink_bridge_max_speed": 0.0,
    "relink_bridge_person_height": PERSON_HEIGHT_M,
    "relink_bridge_fps": SCENE_FPS,
    "relink_bridge_margin": 0.0,
    "relink_bridge_spatial_gate": 0.0,
    "relink_bridge_anchor": "adaptive",
    "relink_bridge_anchor_rate": 0.03,
    "relink_bridge_h_lo": 0.0,
    "relink_bridge_h_hi": 0.0,
    "relink_bridge_dir_bonus": 0.0,
    "relink_bridge_occ_gate_cover": 0.0,
    "relink_bridge_occ_gap_min": 30,
    "relink_bridge_occ_expand_px": 0.0,
    "relink_bridge_occ_expand_cover": 0.9,
    "relink_bridge_app_veto": -1.0,
    "min_tracklet_len": 1,
    "min_tracklet_score": 0.0,
    "interpolate_tracklets": True,
    "interpolate_max_gap": 20,
    "interpolate_min_track_len": 5,
    "interpolate_min_h": 0.0,
    "kalman_adapt_mode": 0,
    "kalman_r_scale": 0.75,
    "vel_dir_weight": 0.0,
    "occ_vel_weight": 0.0,
    "fuse_score_weight": 0.0,
    "stage2_match_thresh": 0.5,
    "birth_low_score_thresh": 0.0,
    "birth_prox_norm_thresh": 0.0,
    "oao_tau": 0.0,
    "oao_contest_thresh": -1.0,
    "oao_score_w": -1.0,
    "oao_occ_mode": 0,
    "oao_crowd_radius": 0.0,
    "oao_height_gate": 0.0,
    "oao_foot_gate": 0.0,
    "oao_ramp_frames": 0.0,
    "occ_state_enabled": True,
    "occ_iou_thresh": 0.45,
    "occ_foot_gap": 0.15,
    "occ_ttl": 4,
    "occ_cost_weight": 0.50,
    "multiplicative_cost": False,
    "sinkhorn_lambda": 30.0,
    "stability_cost_w": 0.0,
    "association_scoring_mode": "baseline",
    "assoc_score_cost_w": 0.0,
    "assoc_height_cost_w": 0.0,
    "assoc_energy_diagnostics": False,
    "vel_alpha": 0.3,
    "acc_alpha": 0.15,
    "min_motion_observations": 2,
    "w_motion_iou": 0.3,
    "motion_consistency_check": True,
    "consistency_tol": 2.0,
    "enable_motion_only": True,
    "motion_only_lost_frames": 5,
    "motion_only_iou_threshold": 0.15,
    "motion_only_min_lost_frames": 1,
    "temporal_consistency_min_frames": 3,
    "per_frame_detection_cap": 0,
    "detection_cap_rank_method": "fp_filter_quality",
    "adaptive_detection_cap": False,
    "adaptive_cap_base": 40,
    "adaptive_cap_max": 60,
    "adaptive_cap_min": 15,
    "fp_hard_filter_enabled": True,
    "fp_hard_filter_min_score": 0.10,
    "fp_hard_filter_max_suspicious_area": 40000,
    "fp_hard_filter_max_suspicious_score": 0.40,
    "external_fp_filter_mode": "rule",
    "external_fp_logistic_model": "",
    "external_fp_logistic_threshold": 0.5,
    "external_fp_max_score": 0.18,
    "external_fp_penalty": 1.0,
    "external_fp_softmax_min_scale": 0.7,
    "appearance_bank_enabled": False,
    "appearance_bank_size": 5,
    "appearance_bank_min_score": 0.45,
    "appearance_bank_min_iou": 0.35,
    "appearance_bank_consistency_threshold": 0.75,
    "appearance_bank_high_quality_min_score": 0.75,
    "appearance_bank_min_aspect": 1.2,
    "appearance_bank_max_aspect": 4.5,
    "appearance_occlusion_gate": False,
    "appearance_occlusion_cov": 0.4,
    "bank_quality_v2": True,
    "bank_quality_w_det": 0.45,
    "bank_quality_w_iou": 0.20,
    "bank_quality_w_aspect": 0.15,
    "bank_quality_w_center": 0.10,
    "bank_quality_w_area": 0.10,
    "bank_weighted_mean": False,
    "need_reid_enabled": False,
    "async_reid": False,
    "pipeline_relink": False,
    "per_seq_adapt": True,
    "seqs": None,
    "kwargs": None,
    "pose_box_expand": False,
    "pose_expand_ankle_conf": 0.30,
    "pose_expand_margin": 0.05,
    "pose_expand_flat_aspect": 0.0,
    "scene_adapt_enabled": False,
    "scene_adapt_window": 30,
    "scene_adapt_crowd_thresh": 15.0,
    "scene_adapt_narrow_aspect_thresh": 2.1,
    "scene_adapt_narrow_width_thresh": 0.035,
}


def _resolve_fields(
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Auto-resolve all EvalConfig fields from kwargs using _DEFAULTS + aliases.

    Returns a dict that can be passed to EvalConfig(**fields).
    Caller should then override computed/derived fields before constructing.
    """
    result: dict[str, Any] = {}
    for field_name, default in _DEFAULTS.items():
        val = _resolve_alias(kwargs, field_name, default)
        result[field_name] = val
    return result


def parse_eval_config(
    output: str,
    data_root: str,
    split: str,
    sequences: str,
    conf_threshold: float,
    reid_mode: str,
    reid_model: str,
    profile_stages: bool,
    kwargs: dict[str, Any],
    profile_frame_csv: bool = False,
) -> EvalConfig:
    output_root = Path(output)

    reid_enabled = reid_mode != "off"
    profile_lazy_reid_embeddings = bool(
        kwargs.get("profile_lazy_reid_embeddings", False)
    )
    profile_lazy_reid_candidates = (
        bool(kwargs.get("profile_lazy_reid_candidates", False))
        or profile_lazy_reid_embeddings
    )
    reid_work_enabled = reid_enabled or profile_lazy_reid_embeddings

    crop_hw = (
        (256, 128) if reid_model in {"transreid", "osnet", "fastreid"} else (224, 224)
    )

    track_thresh = float(kwargs.get("track_thresh", 0.05))
    mid_thresh = float(kwargs.get("mid_thresh", 0.10))

    suspect_score_arg = kwargs.get("geometry_suspect_score", None)
    if suspect_score_arg is None:
        geometry_suspect_score = track_thresh + max(
            (mid_thresh - track_thresh) * 0.5, 1e-4
        )
    else:
        geometry_suspect_score = float(suspect_score_arg)
    geometry_suspect_support_score = min(
        max(geometry_suspect_score, track_thresh + 1e-4),
        max(mid_thresh - 1e-4, track_thresh + 1e-4),
    )

    _detector_suffixes = {"SDP", "DPM", "FRCNN"}
    if sequences:
        filters = [s.strip() for s in sequences.split(",")]
        if all(f in _detector_suffixes for f in filters):
            _all = sorted(
                d.name for d in (Path(data_root) / split).iterdir() if d.is_dir()
            )
            seqs = [d for d in _all if any(d.endswith(f"-{f}") for f in filters)]
        else:
            seqs = filters
    else:
        seqs = sorted(d.name for d in (Path(data_root) / split).iterdir() if d.is_dir())

    tiling = kwargs.get("tiling", "native_960")
    _nms_default = 0.35 if tiling == "960p_3x2" else 0.5

    post_lifecycle_merge = bool(kwargs.get("post_lifecycle_merge", False))
    post_lifecycle_appearance_gate = bool(
        kwargs.get("post_lifecycle_appearance_gate", False)
    )
    post_lifecycle_appearance_weight = float(
        kwargs.get("post_lifecycle_appearance_weight", 0.0)
    )
    if post_lifecycle_merge and not post_lifecycle_appearance_gate:
        if post_lifecycle_appearance_weight <= 0.0:
            post_lifecycle_appearance_gate = True

    private_selection_mode = (
        str(kwargs.get("private_selection_mode", "global")).strip().lower()
    )
    if private_selection_mode not in {
        "global",
        "per_track",
        "suppressor_aware",
        "sparse_symmetric",
        "energy",
    }:
        raise ValueError(f"unknown private_selection_mode: {private_selection_mode}")

    association_scoring_mode = (
        str(kwargs.get("association_scoring_mode", "baseline")).strip().lower()
    )
    if association_scoring_mode not in {"baseline", "energy"}:
        raise ValueError(
            f"unknown association_scoring_mode: {association_scoring_mode}"
        )

    fields = _resolve_fields(kwargs)

    # Function parameters
    fields["data_root"] = data_root
    fields["split"] = split
    fields["conf_threshold"] = conf_threshold
    fields["output_root"] = output_root
    fields["profile_stages"] = profile_stages
    fields["profile_frame_csv"] = profile_frame_csv

    # Derived fields
    fields["reid_mode"] = reid_mode
    fields["reid_model"] = reid_model
    fields["reid_enabled"] = reid_enabled
    fields["reid_work_enabled"] = reid_work_enabled
    fields["crop_hw"] = crop_hw
    fields["profile_lazy_reid_embeddings"] = profile_lazy_reid_embeddings
    fields["profile_lazy_reid_candidates"] = profile_lazy_reid_candidates
    fields["use_semantic_mode"] = (
        reid_mode in {"semantic", "hybrid"}
        or bool(kwargs.get("semantic_kalman_gate", False))
        or bool(kwargs.get("semantic_cheb_gr_claim", False))
    )
    fields["use_tracker_reid"] = reid_mode in {"tracker", "hybrid"}
    fields["seqs"] = seqs
    fields["kwargs"] = kwargs

    # Fields with special processing
    fields["debug_dump_seq"] = str(kwargs.get("debug_dump_seq", "")).strip()
    fields["debug_dump_frames"] = str(kwargs.get("debug_dump_frames", "")).strip()
    fields["debug_dump_csv"] = str(kwargs.get("debug_dump_csv", "")).strip()
    fields["debug_birth_csv"] = str(kwargs.get("debug_birth_csv", "")).strip()
    fields["reid_interval"] = max(1, int(kwargs.get("reid_interval", 20)))
    fields["gmc_downscale"] = max(1, int(kwargs.get("gmc_downscale", 8)))
    fields["semantic_buffer_size"] = max(1, int(kwargs.get("semantic_buffer_size", 10)))
    fields["reid_budget_raw"] = float(kwargs.get("reid_budget", 0.2))

    # Conditional logic
    fields["post_lifecycle_merge"] = post_lifecycle_merge
    fields["post_lifecycle_appearance_gate"] = post_lifecycle_appearance_gate
    fields["post_lifecycle_appearance_weight"] = post_lifecycle_appearance_weight

    # Validated enums
    fields["private_selection_mode"] = private_selection_mode
    fields["association_scoring_mode"] = association_scoring_mode

    # Computed geometry fields
    fields["geometry_suspect_score"] = geometry_suspect_score
    fields["geometry_suspect_support_score"] = geometry_suspect_support_score

    # Track threshold locals (already computed)
    fields["track_thresh"] = track_thresh
    fields["mid_thresh"] = mid_thresh

    # NMS threshold (computed from tiling)
    fields["nms_iou_threshold"] = float(kwargs.get("nms_iou_threshold") or _nms_default)

    # Tiling (computed)
    fields["tiling"] = tiling

    # New track thresh (conditional)
    _new_track_thresh = kwargs.get("new_track_thresh")
    fields["new_track_thresh"] = (
        0.35 if _new_track_thresh is None else float(_new_track_thresh)
    )

    # Kalman adapt mode (NSA backward compat)
    fields["kalman_adapt_mode"] = (
        1
        if int(kwargs.get("kalman_adapt_mode", 0)) == 0
        and kwargs.get("nsa_kalman", False)
        else int(kwargs.get("kalman_adapt_mode", 0))
    )

    # Pipeline relink (conditional)
    fields["pipeline_relink"] = (
        bool(kwargs.get("pipeline_relink", False)) and not profile_stages
    )

    # Preprocess modes (transformed)
    fields["preprocess_modes"] = parse_preprocess(
        kwargs.get("preprocess", "letterbox,gamma,contrast")
    )

    # Cheb-GR offline→online legacy key compatibility
    for _new, _old, _coerce in [
        ("cheb_gr_online", "cheb_gr_offline_handover", bool),
        ("cheb_gr_online_decide_n", "cheb_gr_offline_decide_n", int),
        ("cheb_gr_online_max_cost", "cheb_gr_offline_max_cost", float),
        ("cheb_gr_online_min_head", "cheb_gr_offline_min_head", int),
        ("cheb_gr_online_margin", "cheb_gr_offline_margin", float),
        ("cheb_gr_online_key_sim_min", "cheb_gr_offline_key_sim_min", float),
        (
            "cheb_gr_online_key_sim_cost_floor",
            "cheb_gr_offline_key_sim_cost_floor",
            float,
        ),
        ("cheb_gr_online_key_margin_min", "cheb_gr_offline_key_margin_min", float),
        ("cheb_gr_online_center_dist_veto", "cheb_gr_offline_center_dist_veto", float),
        ("cheb_gr_online_pollution_veto", "cheb_gr_offline_pollution_veto", float),
        ("cheb_gr_online_neighbor_iou_max", "cheb_gr_offline_neighbor_iou_max", float),
        ("cheb_gr_online_bank_mode", "cheb_gr_offline_bank_mode", str),
        ("cheb_gr_online_bank_n", "cheb_gr_offline_bank_n", int),
        ("cheb_gr_online_log", "cheb_gr_offline_log", bool),
    ]:
        if _old in kwargs:
            fields[_new] = _coerce(kwargs[_old])

    # ReID engine: empty string fallback
    fields["reid_engine"] = kwargs.get("reid_engine_path", "") or ""

    return EvalConfig(**fields)
