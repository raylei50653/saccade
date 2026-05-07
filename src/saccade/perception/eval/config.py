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
    profile_stages: bool

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

    min_tracklet_len: int
    min_tracklet_score: float
    nsa_kalman: bool

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

    need_reid_enabled: bool
    async_reid: bool
    pipeline_relink: bool
    per_seq_adapt: bool
    seqs: list[str]
    kwargs: dict[str, Any]


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

    seqs = (
        sequences.split(",")
        if sequences
        else [d.name for d in (Path(data_root) / split).iterdir() if d.is_dir()]
    )

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
        profile_stages=profile_stages,
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
        semantic_bank_inject=bool(kwargs.get("semantic_bank_inject", True)),
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
        use_semantic_mode=reid_mode in {"semantic", "hybrid"},
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
        min_tracklet_len=max(1, int(kwargs.get("min_tracklet_len", 1))),
        min_tracklet_score=float(kwargs.get("min_tracklet_score", 0.0)),
        nsa_kalman=bool(kwargs.get("nsa_kalman", False)),
        appearance_bank_enabled=bool(kwargs.get("appearance_bank", True)),
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
        need_reid_enabled=bool(kwargs.get("need_reid", False)),
        async_reid=bool(kwargs.get("async_reid", True)),
        pipeline_relink=bool(kwargs.get("pipeline_relink", True))
        and not profile_stages,
        per_seq_adapt=bool(kwargs.get("per_seq_adapt", True)),
        seqs=seqs,
        kwargs=kwargs,
    )
