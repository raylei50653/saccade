from dataclasses import dataclass
from typing import Any
from pathlib import Path
from saccade.perception.eval.preprocess import parse_preprocess


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
    track_thresh: float
    high_thresh: float
    match_thresh: float
    mid_thresh: float
    new_track_thresh: float
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
    relink_bridge_occ_gate_cover: float
    relink_bridge_occ_gap_min: int
    relink_bridge_occ_expand_px: float
    relink_bridge_occ_expand_cover: float

    min_tracklet_len: int
    min_tracklet_score: float
    interpolate_tracklets: bool
    interpolate_max_gap: int
    interpolate_min_track_len: int
    interpolate_min_h: float
    nsa_kalman: bool
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

    tiling = kwargs.get("tiling", "960p_2x2")
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

    return EvalConfig(
        data_root=data_root,
        split=split,
        conf_threshold=conf_threshold,
        output_root=output_root,
        debug_dump_seq=str(kwargs.get("debug_dump_seq", "")).strip(),
        debug_dump_frames=str(kwargs.get("debug_dump_frames", "")).strip(),
        debug_dump_csv=str(kwargs.get("debug_dump_csv", "")).strip(),
        debug_birth_csv=str(kwargs.get("debug_birth_csv", "")).strip(),
        profile_stages=profile_stages,
        latency_only=bool(kwargs.get("latency_only", False)),
        workbench=bool(kwargs.get("workbench", False)),
        threads=int(kwargs.get("threads", 1)),
        reid_mode=reid_mode,
        reid_enabled=reid_enabled,
        reid_model=reid_model,
        reid_engine=kwargs.get("reid_engine_path", "") or "",
        reid_interval=max(1, int(kwargs.get("reid_interval", 20))),
        reid_crop_mode=kwargs.get("reid_crop_mode", "tight"),
        reid_crop_padding=float(kwargs.get("reid_crop_padding", 0.0)),
        reid_crop_layout=kwargs.get("reid_crop_layout", "full"),
        crop_hw=crop_hw,
        profile_lazy_reid_embeddings=profile_lazy_reid_embeddings,
        profile_lazy_reid_candidates=profile_lazy_reid_candidates,
        reid_work_enabled=reid_work_enabled,
        gmc_enabled=bool(kwargs.get("gmc", False)),
        gmc_mode=str(kwargs.get("gmc_mode", "gpu")),
        gmc_downscale=max(1, int(kwargs.get("gmc_downscale", 8))),
        gmc_fg_mask=bool(kwargs.get("gmc_fg_mask", False)),
        gmc_pcr_uncertain_thresh=float(kwargs.get("gmc_pcr_uncertain_thresh", 8.0)),
        homography_root=str(kwargs.get("homography_root", "")),
        semantic_buffer_size=max(1, int(kwargs.get("semantic_buffer_size", 1))),
        semantic_min_consistency=float(kwargs.get("semantic_min_consistency", 0.0)),
        semantic_rerank_mode=str(kwargs.get("semantic_rerank_mode", "mean")),
        semantic_reciprocal_margin=float(kwargs.get("semantic_reciprocal_margin", 0.0)),
        semantic_bank_inject=bool(kwargs.get("semantic_bank_inject", False)),
        semantic_iou_weight=float(kwargs.get("semantic_iou_weight", 0.0)),
        semantic_mahalanobis_weight=float(
            kwargs.get("semantic_mahalanobis_weight", 0.0)
        ),
        semantic_dynamic_margin_crowd=float(
            kwargs.get("semantic_dynamic_margin_crowd", 0.0)
        ),
        semantic_dynamic_margin_age=float(
            kwargs.get("semantic_dynamic_margin_age", 0.0)
        ),
        semantic_biometric_threshold=float(
            kwargs.get("semantic_biometric_threshold", 0.0)
        ),
        semantic_w_sim_base=float(kwargs.get("semantic_w_sim_base", 0.0)),
        semantic_w_iou_base=float(kwargs.get("semantic_w_iou_base", 0.0)),
        semantic_w_maha_base=float(kwargs.get("semantic_w_maha_base", 0.0)),
        semantic_shift_ambiguity=float(kwargs.get("semantic_shift_ambiguity", 0.0)),
        semantic_shift_lost_age=float(kwargs.get("semantic_shift_lost_age", 0.0)),
        semantic_clean_score_threshold=float(
            kwargs.get("semantic_clean_score_threshold", 0.60)
        ),
        semantic_clean_margin_ratio=float(
            kwargs.get("semantic_clean_margin_ratio", 0.0)
        ),
        semantic_clean_min_aspect=float(kwargs.get("semantic_clean_min_aspect", 1.2)),
        semantic_clean_max_aspect=float(kwargs.get("semantic_clean_max_aspect", 4.5)),
        semantic_strict_sim_threshold=float(
            kwargs.get("semantic_strict_sim_threshold", 0.0)
        ),
        semantic_exp_density_gating=bool(
            kwargs.get("semantic_exp_density_gating", False)
        ),
        semantic_exp_density_k=float(kwargs.get("semantic_exp_density_k", 2.0)),
        semantic_exp_density_eta=float(kwargs.get("semantic_exp_density_eta", 0.15)),
        cross_tile_score_penalty=float(kwargs.get("cross_tile_score_penalty", 1.0)),
        tile_diagnostics=bool(kwargs.get("tile_diagnostics", False)),
        tile_seam_score_penalty=float(kwargs.get("tile_seam_score_penalty", 1.0)),
        tile_seam_margin_canvas_px=float(
            kwargs.get("tile_seam_margin_canvas_px", 24.0)
        ),
        cross_tile_seam_center_scale=float(
            kwargs.get("cross_tile_seam_center_scale", 1.8)
        ),
        cross_tile_seam_area_ratio_threshold=float(
            kwargs.get("cross_tile_seam_area_ratio_threshold", 0.30)
        ),
        cross_tile_seam_min_overlap_ratio=float(
            kwargs.get("cross_tile_seam_min_overlap_ratio", 0.45)
        ),
        force_python_relinker=bool(kwargs.get("force_python_relinker", False)),
        use_semantic_mode=reid_mode in {"semantic", "hybrid"}
        or bool(kwargs.get("semantic_kalman_gate", False)),
        use_tracker_reid=reid_mode in {"tracker", "hybrid"},
        person_class=int(kwargs.get("person_class", 0)),
        track_person_only=bool(kwargs.get("track_person_only", True)),
        track_thresh=track_thresh,
        high_thresh=float(kwargs.get("high_thresh", 0.5)),
        match_thresh=float(kwargs.get("match_thresh", 0.8)),
        mid_thresh=mid_thresh,
        new_track_thresh=0.35
        if kwargs.get("new_track_thresh", None) is None
        else float(kwargs.get("new_track_thresh")),  # type: ignore[arg-type]
        birth_quality_gate=bool(kwargs.get("birth_quality_gate", False)),
        birth_min_quality=float(kwargs.get("birth_min_quality", 0.0)),
        birth_quality_score_bias=float(kwargs.get("birth_quality_score_bias", 0.15)),
        stage2_quality_gate=bool(kwargs.get("stage2_quality_gate", False)),
        stage2_quality_min=float(kwargs.get("stage2_quality_min", 0.40)),
        birth_consecutive_gate=bool(kwargs.get("birth_consecutive_gate", False)),
        birth_consecutive_frames=max(2, int(kwargs.get("birth_consecutive_frames", 2))),
        birth_consecutive_iou=float(kwargs.get("birth_consecutive_iou", 0.40)),
        birth_consecutive_boost=float(kwargs.get("birth_consecutive_boost", 0.05)),
        birth_consecutive_min_score=float(
            kwargs.get("birth_consecutive_min_score", 0.20)
        ),
        birth_consecutive_min_motion=float(
            kwargs.get("birth_consecutive_min_motion", 0.0)
        ),
        # Duplicate suppression
        duplicate_suppression=bool(kwargs.get("duplicate_suppression", False)),
        duplicate_suppression_iou_threshold=float(
            kwargs.get("duplicate_suppression_iou_threshold", 0.85)
        ),
        duplicate_suppression_min_score_ratio=float(
            kwargs.get("duplicate_suppression_min_score_ratio", 1.05)
        ),
        multi_birth_enabled=bool(kwargs.get("multi_birth_enabled", False)),
        multi_birth_min_score=float(kwargs.get("multi_birth_min_score", 0.12)),
        multi_birth_min_frames=max(2, int(kwargs.get("multi_birth_min_frames", 3))),
        multi_birth_target_motion=float(kwargs.get("multi_birth_target_motion", 12.0)),
        multi_birth_evidence_threshold=float(
            kwargs.get("multi_birth_evidence_threshold", 0.60)
        ),
        multi_birth_iou_match=float(kwargs.get("multi_birth_iou_match", 0.30)),
        multi_birth_ttl_frames=max(1, int(kwargs.get("multi_birth_ttl_frames", 5))),
        multi_birth_w_score=float(kwargs.get("multi_birth_w_score", 0.35)),
        multi_birth_w_motion=float(kwargs.get("multi_birth_w_motion", 0.30)),
        multi_birth_w_quality=float(kwargs.get("multi_birth_w_quality", 0.20)),
        multi_birth_w_streak=float(kwargs.get("multi_birth_w_streak", 0.15)),
        multi_birth_min_aspect=float(kwargs.get("multi_birth_min_aspect", 0.0)),
        multi_birth_max_area_px=int(kwargs.get("multi_birth_max_area_px", 0)),
        multi_birth_replace_mode=bool(kwargs.get("multi_birth_replace_mode", False)),
        multi_birth_replace_evidence_threshold=float(
            kwargs.get("multi_birth_replace_evidence_threshold", 0.85)
        ),
        exp_velocity_aligned_bank=bool(kwargs.get("exp_velocity_aligned_bank", False)),
        crowd_low_score_mode=bool(kwargs.get("crowd_low_score_mode", False)),
        crowd_low_score_trigger=int(kwargs.get("crowd_low_score_trigger", 25)),
        crowd_conf_threshold=float(kwargs.get("crowd_conf_threshold", 0.02)),
        crowd_track_thresh=float(kwargs.get("crowd_track_thresh", 0.02)),
        crowd_mid_thresh=float(kwargs.get("crowd_mid_thresh", 0.05)),
        crowd_new_track_thresh=float(kwargs.get("crowd_new_track_thresh", 0.25)),
        narrow_person_score_bonus=float(kwargs.get("narrow_person_score_bonus", 0.0)),
        narrow_person_max_width_ratio=float(
            kwargs.get("narrow_person_max_width_ratio", 0.018)
        ),
        narrow_person_min_height_ratio=float(
            kwargs.get("narrow_person_min_height_ratio", 0.045)
        ),
        narrow_person_min_aspect=float(kwargs.get("narrow_person_min_aspect", 2.0)),
        narrow_person_max_aspect=float(kwargs.get("narrow_person_max_aspect", 4.8)),
        tiling=tiling,
        nms_iou_threshold=float(kwargs.get("nms_iou_threshold") or _nms_default),
        cross_tile_merge=bool(kwargs.get("cross_tile_merge", False)),
        geometry_mid_scale=bool(kwargs.get("geometry_mid_scale", False)),
        geometry_ref_height_ratio=float(kwargs.get("geometry_ref_height_ratio", 0.12)),
        geometry_min_scale=float(kwargs.get("geometry_min_scale", 0.875)),
        geometry_max_scale=float(kwargs.get("geometry_max_scale", 1.20)),
        geometry_ema_beta=float(kwargs.get("geometry_ema_beta", 0.80)),
        geometry_loosen_step=float(kwargs.get("geometry_loosen_step", 0.08)),
        geometry_tighten_step=float(kwargs.get("geometry_tighten_step", 0.03)),
        geometry_min_samples=int(kwargs.get("geometry_min_samples", 5)),
        lazy_reid_min_hit_streak=int(kwargs.get("lazy_reid_min_hit_streak", 2)),
        lazy_reid_self_threshold=float(kwargs.get("lazy_reid_self_threshold", 0.85)),
        preprocess_modes=parse_preprocess(
            kwargs.get("preprocess", "letterbox,gamma,contrast")
        ),
        gamma=float(kwargs.get("gamma", 0.8)),
        gamma_luma_threshold=float(kwargs.get("gamma_luma_threshold", 0.35)),
        contrast=float(kwargs.get("contrast", 1.2)),
        id_stability_filter_enabled=bool(kwargs.get("id_stability_filter", True)),
        id_stability_min_hits=int(kwargs.get("id_stability_min_hits", 2)),
        id_stability_min_iou=float(kwargs.get("id_stability_min_iou", 0.05)),
        id_stability_max_center_shift=float(
            kwargs.get("id_stability_max_center_shift", 2.0)
        ),
        id_stability_max_gap=int(kwargs.get("id_stability_max_gap", 1)),
        id_stability_score_ema=float(kwargs.get("id_stability_score_ema", 0.70)),
        id_stability_min_score_ema=float(
            kwargs.get("id_stability_min_score_ema", 0.15)
        ),
        person_geometry_prior=bool(kwargs.get("person_geometry_prior", True)),
        person_min_height_ratio=float(kwargs.get("person_min_height_ratio", 0.018)),
        person_min_aspect=float(kwargs.get("person_min_aspect", 1.0)),
        person_max_aspect=float(kwargs.get("person_max_aspect", 5.5)),
        person_min_area_ratio=float(kwargs.get("person_min_area_ratio", 0.00006)),
        person_max_area_ratio=float(kwargs.get("person_max_area_ratio", 0.0)),
        detection_quality_scaling=bool(kwargs.get("detection_quality_scaling", False)),
        detection_quality_w_aspect=float(
            kwargs.get("detection_quality_w_aspect", 0.50)
        ),
        detection_quality_w_center=float(
            kwargs.get("detection_quality_w_center", 0.30)
        ),
        detection_quality_w_area=float(kwargs.get("detection_quality_w_area", 0.20)),
        geometry_suspect_support=bool(kwargs.get("geometry_suspect_support", True)),
        geometry_suspect_score=geometry_suspect_score,
        geometry_suspect_support_score=geometry_suspect_support_score,
        reid_budget_raw=float(kwargs.get("reid_budget", 0.0)),
        lifecycle_merge_enabled=bool(kwargs.get("lifecycle_merge", False)),
        lifecycle_ttl=int(kwargs.get("lifecycle_ttl", 45)),
        lifecycle_min_gap=int(kwargs.get("lifecycle_min_gap", 2)),
        lifecycle_spatial_gate=float(kwargs.get("lifecycle_spatial_gate", 0.08)),
        lifecycle_min_iou=float(kwargs.get("lifecycle_min_iou", 0.0)),
        lifecycle_sim_threshold=float(kwargs.get("lifecycle_sim_threshold", 0.90)),
        lifecycle_require_embedding=bool(
            kwargs.get("lifecycle_require_embedding", False)
        ),
        lifecycle_ema=float(kwargs.get("lifecycle_ema", 0.83)),
        post_lifecycle_merge=post_lifecycle_merge,
        post_lifecycle_ttl=int(kwargs.get("post_lifecycle_ttl", 60)),
        post_lifecycle_min_gap=int(kwargs.get("post_lifecycle_min_gap", 1)),
        post_lifecycle_velocity_samples=int(
            kwargs.get("post_lifecycle_velocity_samples", 5)
        ),
        post_lifecycle_spatial_weight=float(
            kwargs.get("post_lifecycle_spatial_weight", 0.35)
        ),
        post_lifecycle_motion_weight=float(
            kwargs.get("post_lifecycle_motion_weight", 0.45)
        ),
        post_lifecycle_time_weight=float(
            kwargs.get("post_lifecycle_time_weight", 0.10)
        ),
        post_lifecycle_direction_weight=float(
            kwargs.get("post_lifecycle_direction_weight", 0.25)
        ),
        post_lifecycle_max_cost=float(kwargs.get("post_lifecycle_max_cost", 1.25)),
        post_lifecycle_appearance_gate=post_lifecycle_appearance_gate,
        post_lifecycle_appearance_threshold=float(
            kwargs.get("post_lifecycle_appearance_threshold", 0.90)
        ),
        post_lifecycle_appearance_min_samples=max(
            1, int(kwargs.get("post_lifecycle_appearance_min_samples", 1))
        ),
        post_lifecycle_appearance_max_samples=max(
            1, int(kwargs.get("post_lifecycle_appearance_max_samples", 5))
        ),
        post_lifecycle_appearance_min_score=float(
            kwargs.get("post_lifecycle_appearance_min_score", 0.0)
        ),
        post_lifecycle_appearance_min_consistency=float(
            kwargs.get("post_lifecycle_appearance_min_consistency", 0.0)
        ),
        post_lifecycle_appearance_weight=post_lifecycle_appearance_weight,
        post_lifecycle_gap_uncertainty_weight=float(
            kwargs.get("post_lifecycle_gap_uncertainty_weight", 0.0)
        ),
        post_lifecycle_consistency_weight=float(
            kwargs.get("post_lifecycle_consistency_weight", 0.0)
        ),
        post_lifecycle_missing_appearance_cost=float(
            kwargs.get("post_lifecycle_missing_appearance_cost", 0.5)
        ),
        cheb_gr_merge_enabled=bool(kwargs.get("cheb_gr_merge_enabled", False)),
        cheb_gr_merge_max_cost=float(kwargs.get("cheb_gr_merge_max_cost", 0.55)),
        cheb_gr_merge_max_gap=int(kwargs.get("cheb_gr_merge_max_gap", 60)),
        cheb_gr_merge_min_overlap=int(kwargs.get("cheb_gr_merge_min_overlap", 1)),
        cheb_gr_merge_n_samples=int(kwargs.get("cheb_gr_merge_n_samples", 50)),
        cheb_gr_pool_frac=float(kwargs.get("cheb_gr_pool_frac", 0.3)),
        cheb_gr_lambda=float(kwargs.get("cheb_gr_lambda", 2.0)),
        cheb_gr_k2=int(kwargs.get("cheb_gr_k2", 6)),
        cheb_gr_max_fwd=int(kwargs.get("cheb_gr_max_fwd", 50)),
        cheb_gr_fuse_lambda=float(kwargs.get("cheb_gr_fuse_lambda", 0.3)),
        cheb_gr_engine=str(kwargs.get("cheb_gr_engine", "") or ""),
        relink_enabled=bool(kwargs.get("relink_enabled", False)),
        relink_bank_cap=int(kwargs.get("relink_bank_cap", 256)),
        relink_sim_thresh=float(kwargs.get("relink_sim_thresh", 0.6)),
        relink_lambda=float(kwargs.get("relink_lambda", 2.5)),
        relink_spatial_gate=float(kwargs.get("relink_spatial_gate", 4.0)),
        relink_max_age=int(kwargs.get("relink_max_age", 300)),
        relink_bridge_enabled=bool(kwargs.get("relink_bridge_enabled", False)),
        relink_bridge_px=float(kwargs.get("relink_bridge_px", 0.25)),
        relink_bridge_at=int(kwargs.get("relink_bridge_at", 4)),
        relink_bridge_min_lost=int(kwargs.get("relink_bridge_min_lost", 2)),
        relink_bridge_ttl=int(kwargs.get("relink_bridge_ttl", 120)),
        relink_bridge_max_speed=float(kwargs.get("relink_bridge_max_speed", 0.0)),
        relink_bridge_person_height=float(
            kwargs.get("relink_bridge_person_height", 1.65)
        ),
        relink_bridge_fps=float(kwargs.get("relink_bridge_fps", 30.0)),
        relink_bridge_margin=float(kwargs.get("relink_bridge_margin", 0.05)),
        relink_bridge_spatial_gate=float(kwargs.get("relink_bridge_spatial_gate", 0.0)),
        relink_bridge_anchor=str(kwargs.get("relink_bridge_anchor", "adaptive")),
        relink_bridge_anchor_rate=float(kwargs.get("relink_bridge_anchor_rate", 0.03)),
        relink_bridge_h_lo=float(kwargs.get("relink_bridge_h_lo", 0.75)),
        relink_bridge_h_hi=float(kwargs.get("relink_bridge_h_hi", 1.33)),
        relink_bridge_occ_gate_cover=float(
            kwargs.get("relink_bridge_occ_gate_cover", 0.0)
        ),
        relink_bridge_occ_gap_min=int(kwargs.get("relink_bridge_occ_gap_min", 30)),
        relink_bridge_occ_expand_px=float(
            kwargs.get("relink_bridge_occ_expand_px", 0.0)
        ),
        relink_bridge_occ_expand_cover=float(
            kwargs.get("relink_bridge_occ_expand_cover", 0.9)
        ),
        min_tracklet_len=max(1, int(kwargs.get("min_tracklet_len", 1))),
        min_tracklet_score=float(kwargs.get("min_tracklet_score", 0.0)),
        interpolate_tracklets=bool(kwargs.get("interpolate_tracklets", True)),
        interpolate_max_gap=max(0, int(kwargs.get("interpolate_max_gap", 20))),
        interpolate_min_track_len=max(
            1, int(kwargs.get("interpolate_min_track_len", 5))
        ),
        interpolate_min_h=float(kwargs.get("interpolate_min_h", 0.0)),
        nsa_kalman=bool(kwargs.get("nsa_kalman", False)),
        kalman_r_scale=float(kwargs.get("kalman_r_scale", 1.0)),
        vel_dir_weight=float(kwargs.get("vel_dir_weight", 0.0)),
        occ_vel_weight=float(kwargs.get("occ_vel_weight", 0.0)),
        fuse_score_weight=float(kwargs.get("fuse_score_weight", 0.0)),
        stage2_match_thresh=float(kwargs.get("stage2_match_thresh", 0.5)),
        birth_low_score_thresh=float(kwargs.get("birth_low_score_thresh", 0.0)),
        birth_prox_norm_thresh=float(kwargs.get("birth_prox_norm_thresh", 0.0)),
        oao_tau=float(kwargs.get("oao_tau", 0.0)),
        oao_contest_thresh=float(kwargs.get("oao_contest_thresh", -1.0)),
        oao_score_w=float(kwargs.get("oao_score_w", -1.0)),
        oao_occ_mode=int(kwargs.get("oao_occ_mode", 0)),
        oao_crowd_radius=float(kwargs.get("oao_crowd_radius", 0.0)),
        oao_height_gate=float(kwargs.get("oao_height_gate", 0.0)),
        oao_foot_gate=float(kwargs.get("oao_foot_gate", 0.0)),
        oao_ramp_frames=float(kwargs.get("oao_ramp_frames", 0.0)),
        occ_state_enabled=bool(kwargs.get("occ_state_enabled", True)),
        occ_iou_thresh=float(kwargs.get("occ_iou_thresh", 0.45)),
        occ_foot_gap=float(kwargs.get("occ_foot_gap", 0.15)),
        occ_ttl=int(kwargs.get("occ_ttl", 4)),
        occ_cost_weight=float(kwargs.get("occ_cost_weight", 0.50)),
        temporal_consistency_min_frames=int(
            kwargs.get("temporal_consistency_min_frames", 3)
        ),
        per_frame_detection_cap=int(kwargs.get("per_frame_detection_cap", 0)),
        detection_cap_rank_method=str(
            kwargs.get("detection_cap_rank_method", "fp_filter_quality")
        ),
        adaptive_detection_cap=bool(kwargs.get("adaptive_detection_cap", True)),
        adaptive_cap_base=int(kwargs.get("adaptive_cap_base", 40)),
        adaptive_cap_max=int(kwargs.get("adaptive_cap_max", 60)),
        adaptive_cap_min=int(kwargs.get("adaptive_cap_min", 15)),
        # FP hard filter: removes extremely suspicious low-score large-area detections
        fp_hard_filter_enabled=bool(kwargs.get("fp_hard_filter_enabled", True)),
        fp_hard_filter_min_score=float(kwargs.get("fp_hard_filter_min_score", 0.10)),
        fp_hard_filter_max_suspicious_area=int(
            kwargs.get("fp_hard_filter_max_suspicious_area", 40000)
        ),
        fp_hard_filter_max_suspicious_score=float(
            kwargs.get("fp_hard_filter_max_suspicious_score", 0.40)
        ),
        external_fp_filter_mode=str(kwargs.get("external_fp_filter_mode", "off")),
        external_fp_logistic_model=str(kwargs.get("external_fp_logistic_model", "")),
        external_fp_logistic_threshold=float(
            kwargs.get("external_fp_logistic_threshold", 0.5)
        ),
        external_fp_max_score=float(kwargs.get("external_fp_max_score", 0.18)),
        external_fp_penalty=float(kwargs.get("external_fp_penalty", 1.0)),
        external_fp_softmax_min_scale=float(
            kwargs.get("external_fp_softmax_min_scale", 0.7)
        ),
        appearance_bank_enabled=bool(kwargs.get("appearance_bank", False)),
        appearance_bank_size=max(1, int(kwargs.get("appearance_bank_size", 5))),
        appearance_bank_min_score=float(kwargs.get("appearance_bank_min_score", 0.45)),
        appearance_bank_min_iou=float(kwargs.get("appearance_bank_min_iou", 0.35)),
        appearance_bank_consistency_threshold=float(
            kwargs.get("appearance_bank_consistency_threshold", 0.75)
        ),
        appearance_bank_high_quality_min_score=float(
            kwargs.get("appearance_bank_high_quality_min_score", 0.70)
        ),
        appearance_bank_min_aspect=float(kwargs.get("appearance_bank_min_aspect", 1.2)),
        appearance_bank_max_aspect=float(kwargs.get("appearance_bank_max_aspect", 4.5)),
        bank_quality_v2=bool(kwargs.get("bank_quality_v2", False)),
        bank_quality_w_det=float(kwargs.get("bank_quality_w_det", 0.45)),
        bank_quality_w_iou=float(kwargs.get("bank_quality_w_iou", 0.20)),
        bank_quality_w_aspect=float(kwargs.get("bank_quality_w_aspect", 0.15)),
        bank_quality_w_center=float(kwargs.get("bank_quality_w_center", 0.10)),
        bank_quality_w_area=float(kwargs.get("bank_quality_w_area", 0.10)),
        bank_weighted_mean=bool(kwargs.get("bank_weighted_mean", False)),
        need_reid_enabled=bool(kwargs.get("need_reid", False)),
        async_reid=bool(kwargs.get("async_reid", True)),
        pipeline_relink=bool(kwargs.get("pipeline_relink", True))
        and not profile_stages,
        per_seq_adapt=bool(kwargs.get("per_seq_adapt", True)),
        seqs=seqs,
        kwargs=kwargs,
        pose_box_expand=bool(kwargs.get("pose_box_expand", False)),
        pose_expand_ankle_conf=float(kwargs.get("pose_expand_ankle_conf", 0.30)),
        pose_expand_margin=float(kwargs.get("pose_expand_margin", 0.05)),
        pose_expand_flat_aspect=float(kwargs.get("pose_expand_flat_aspect", 0.0)),
        scene_adapt_enabled=bool(kwargs.get("scene_adapt_enabled", False)),
        scene_adapt_window=int(kwargs.get("scene_adapt_window", 30)),
        scene_adapt_crowd_thresh=float(kwargs.get("scene_adapt_crowd_thresh", 15.0)),
        scene_adapt_narrow_aspect_thresh=float(
            kwargs.get("scene_adapt_narrow_aspect_thresh", 2.1)
        ),
        scene_adapt_narrow_width_thresh=float(
            kwargs.get("scene_adapt_narrow_width_thresh", 0.035)
        ),
    )
