# mypy: ignore-errors
"""Multi-threaded C++ evaluation path (CppEvaluatorPool) extracted from
evaluator.py.

Detection+tracking run GIL-free in C++ for all sequences in parallel;
Python post-processing (post-merge, Cheb-GR handover/merge, occ-audit,
interpolation, metrics) runs afterwards.
"""

from typing import Any

from saccade.perception.feature_extractor import TRTFeatureExtractor

from .utils import append_dict_csv as _append_dict_csv


def _build_cpp_seq_config(
    cfg: Any,
    seq: str,
    data_root: str,
    split: str = "train",
    trt_input_size: int = 640,
    max_raw_dets: int = 8400,
) -> Any:
    """Build a CppSequenceConfig from eval config + sequence name."""
    from saccade_eval_ext import CppSequenceConfig
    from pathlib import Path

    seq_dir = Path(data_root) / split / seq / "img1"
    frame_paths = sorted(str(p) for p in seq_dir.glob("*.jpg"))

    # Read seqinfo.ini for frame dimensions
    seqinfo = Path(data_root) / split / seq / "seqinfo.ini"
    w, h = 1920, 1080
    if seqinfo.exists():
        import configparser

        cp = configparser.ConfigParser()
        cp.read(str(seqinfo))
        w = int(cp.get("Sequence", "imWidth", fallback=str(w)))
        h = int(cp.get("Sequence", "imHeight", fallback=str(h)))

    c = CppSequenceConfig()
    c.name = seq
    c.frame_paths = frame_paths
    c.width = w
    c.height = h

    # Quality filter params
    c.w_aspect = getattr(cfg, "detection_quality_w_aspect", 0.5)
    c.w_center = getattr(cfg, "detection_quality_w_center", 0.3)
    c.w_area = getattr(cfg, "detection_quality_w_area", 0.2)
    c.fp_hard_filter = getattr(cfg, "fp_hard_filter_enabled", True)
    c.fp_min_score = getattr(cfg, "fp_hard_filter_min_score", 0.25)
    c.fp_max_area = float(getattr(cfg, "fp_hard_filter_max_suspicious_area", 10000))
    c.fp_max_susp_score = getattr(cfg, "fp_hard_filter_max_suspicious_score", 0.45)
    c.narrow_bonus = 0.0  # scene-adapt handled separately if needed
    c.person_class = getattr(cfg, "person_class", 0)
    c.trt_input_size = trt_input_size
    c.max_raw_dets = max_raw_dets

    # Tracker params — match Python baseline set_params call
    c.track_thresh = float(getattr(cfg, "track_thresh", 0.05))
    c.high_thresh = float(getattr(cfg, "high_thresh", 0.45))
    c.match_thresh = float(getattr(cfg, "match_thresh", 0.66))
    c.new_track_thresh = float(getattr(cfg, "new_track_thresh", 0.28))
    c.mid_thresh = float(getattr(cfg, "mid_thresh", 0.10))
    c.confirm_streak = cfg.core.confirm_streak
    c.confirm_score_thresh = cfg.core.confirm_score_thresh
    c.fuse_score_weight = float(getattr(cfg, "fuse_score_weight", 0.4))
    c.vel_dir_weight = float(getattr(cfg, "vel_dir_weight", 0.0))
    c.stage2_match_thresh = float(getattr(cfg, "stage2_match_thresh", 0.5))
    c.birth_low_score_thresh = float(getattr(cfg, "birth_low_score_thresh", 0.0))
    c.birth_prox_norm_thresh = float(getattr(cfg, "birth_prox_norm_thresh", 0.0))
    # NB: OAO is configured on the tracker via set_oao_params(cfg.oao_tau); the
    # C++ SequenceConfig has no oao_tau field, so do not set it here.
    c.track_buffer = cfg.track_buffer

    # GMC — always enabled (GPU phase correlation, matches Python workbench default)
    c.gmc_enabled = True
    c.gmc_downscale = 8
    c.gmc_phase_corr = True

    # ReID — wire from cfg when reid is active and an engine path is provided
    _reid_engine_path = getattr(cfg, "reid_engine", "") or ""
    _reid_enabled = getattr(cfg, "reid_enabled", False)
    c.reid_engine_path = _reid_engine_path if _reid_enabled else ""
    _reid_model = getattr(cfg, "reid_model", "siglip2")
    _model_type_map = {
        "siglip2": 0,
        "dinov2": 1,
        "transreid": 2,
        "osnet": 3,
        "fastreid": 4,
        "mobilenetv4_reid": 5,
    }
    c.reid_model_type = _model_type_map.get(_reid_model, 0)
    _budget_raw = float(getattr(cfg, "reid_budget_raw", 0.0))
    c.reid_budget = int(_budget_raw) if _budget_raw > 0 else 64
    c.reid_interval = max(1, int(getattr(cfg, "reid_interval", 1)))
    _crop_hw = getattr(cfg, "crop_hw", (224, 224))
    c.reid_crop_h = int(_crop_hw[0])
    c.reid_crop_w = int(_crop_hw[1])

    return c


def run_eval_cpp(
    engine: str,
    output: str,
    data_root: str,
    split: str,
    sequences: str,
    n_threads: int = 4,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Multi-threaded C++ evaluation loop via CppEvaluatorPool.

    Runs detection+tracking for all sequences in parallel (GIL-free in C++).
    Post-processing (relink, post_merge, metrics) runs in Python after all
    sequences complete.
    """
    try:
        from saccade_eval_ext import CppEvaluatorPool
    except ImportError:
        raise RuntimeError(
            "saccade_eval_ext not available — build with cmake and copy .so to project root"
        )
    from .config import parse_eval_config
    from saccade.perception.detector_trt import TRTYoloDetector

    cfg = parse_eval_config(
        output=output,
        data_root=data_root,
        split=split,
        sequences=sequences,
        conf_threshold=float(kwargs.pop("conf_threshold", 0.05)),
        reid_mode=str(kwargs.pop("reid_mode", "off")),
        reid_model=str(kwargs.pop("reid_model", "siglip2")),
        profile_stages=bool(kwargs.pop("profile_stages", False)),
        kwargs=kwargs,
    )
    # Build PerceptionPipelineConfig (native)
    from saccade_tracking_ext import PerceptionPipelineConfig

    native_cfg = PerceptionPipelineConfig()
    native_cfg.score_threshold = cfg.core.conf_threshold
    native_cfg.person_class = cfg.detection.person_class
    native_cfg.nms_threshold = cfg.detection.nms_iou_threshold
    native_cfg.person_geometry_prior = cfg.person_geometry_prior
    native_cfg.person_min_height_ratio = cfg.person_min_height_ratio
    native_cfg.person_min_aspect = cfg.person_min_aspect
    native_cfg.person_max_aspect = cfg.person_max_aspect

    detector = kwargs.pop("detector", None)
    if detector is None:
        from saccade.perception.detector_trt import TRTYoloDetector

        detector = TRTYoloDetector(engine)

    if hasattr(detector, "cpp_ptr"):
        detect_detector_ptr = int(detector.cpp_ptr)
    elif hasattr(detector, "cpp_engine") and hasattr(detector.cpp_engine, "cpp_ptr"):
        detect_detector_ptr = int(detector.cpp_engine.cpp_ptr)
    else:
        raise RuntimeError(
            f"run_eval_cpp: detector {detector} has no cpp_ptr or cpp_engine.cpp_ptr"
        )

    # Read engine's actual input/output shapes to configure seq_configs correctly.
    if hasattr(detector, "cpp_engine"):
        _in_shape = detector.cpp_engine.get_tensor_shape("images")  # [B,C,H,W]
        _out_shape = detector.cpp_engine.get_tensor_shape("output0")  # [B,N,6]
        trt_input_size = int(_in_shape[2])  # spatial side (H == W)
        max_raw_dets = int(_out_shape[1])  # N detections
    else:
        # It's a MambaGatedDetector
        trt_input_size = getattr(detector, "img_size", 640)
        max_raw_dets = 8400  # YOLO default

    seq_configs = [
        _build_cpp_seq_config(
            cfg, seq, data_root, cfg.core.split, trt_input_size, max_raw_dets
        )
        for seq in cfg.seqs
    ]

    pool = CppEvaluatorPool(
        detect_detector_ptr=detect_detector_ptr,
        pipe_cfg=native_cfg,
        n_threads=min(n_threads, len(cfg.seqs)),
        max_dets=2048,
        max_tracks=256,
        device_id=0,
    )

    import time
    import numpy as np
    from pathlib import Path as _Path
    from .post_merge import (
        post_merge_output_tracklets,
        filter_low_quality_tracklets,
        interpolate_tracklets,
    )

    output_root = cfg.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cheb_gr_online_log_path = output_root / "_cheb_gr_offline_handover.csv"
    if getattr(cfg, "cheb_gr_online_log", False):
        cheb_gr_online_log_path.unlink(missing_ok=True)
    occ_audit_log_path = output_root / "_occ_audit.csv"
    if getattr(cfg, "occ_audit_log", False):
        occ_audit_log_path.unlink(missing_ok=True)

    t0 = time.monotonic()
    cpp_results = pool.run_sequences(seq_configs)  # GIL released here
    elapsed = time.monotonic() - t0
    print(
        f"[run_eval_cpp] {len(cfg.seqs)} sequences in {elapsed:.1f}s "
        f"({n_threads} threads)"
    )

    # Cheb-GR offline tracklet merge (path 2) / output-layer handover: build
    # the ReID extractor once. C++ eval emits no per-det embedding, so tracklet
    # crops are re-cut from img1 inside the post-process loop.
    cheb_gr_extractor = None
    cheb_gr_online = getattr(cfg, "cheb_gr_online", False)
    occ_audit_enabled = getattr(cfg, "occ_audit", False)
    _live_bank_enabled = bool(
        getattr(cfg, "kwargs", {}).get("cheb_gr_online_live_bank", False)
    )
    if (
        cfg.cheb_gr_merge_enabled
        or cheb_gr_online
        or occ_audit_enabled
        or _live_bank_enabled
    ):
        from .cheb_gr_merge import (
            cheb_gr_merge_output_tracklets,
            extract_tracklet_embeddings,
        )
        from .cheb_gr_online import (
            causal_handover_lines,
            extract_handover_embeddings,
        )
        from .occ_audit import (
            extract_audit_embeddings,
            occ_exit_audit_lines,
        )

        cheb_gr_extractor = TRTFeatureExtractor(
            engine_path=cfg.cheb_gr_engine,
            model_type=getattr(cfg, "cheb_gr_model", "siglip2_reid"),
            max_batch=64,
        )

    # ── Per-sequence post-processing ──────────────────────────────────────────
    for seq in cfg.seqs:
        if seq not in cpp_results:
            print(f"⚠️  {seq}: no C++ results, skipping")
            continue

        res = cpp_results[seq]
        frame_ids: np.ndarray = res["frame_ids"]
        track_ids: np.ndarray = res["track_ids"]
        boxes: np.ndarray = res["boxes"]  # [N,4] x1 y1 x2 y2
        scores: np.ndarray = res["scores"]  # [N]

        # Convert to MOT17 lines: frame,id,x,y,w,h,score,-1,-1,-1
        results_lines: list[str] = []
        for i in range(len(frame_ids)):
            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1
            results_lines.append(
                f"{frame_ids[i]},{track_ids[i]},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                f"{scores[i]:.4f},-1,-1,-1"
            )

        results_lines, _ = post_merge_output_tracklets(
            results_lines,
            enabled=cfg.post_lifecycle_merge,
            ttl=cfg.post_lifecycle_ttl,
            min_gap=cfg.post_lifecycle_min_gap,
            velocity_samples=cfg.post_lifecycle_velocity_samples,
            spatial_weight=cfg.post_lifecycle_spatial_weight,
            motion_weight=cfg.post_lifecycle_motion_weight,
            time_weight=cfg.post_lifecycle_time_weight,
            direction_weight=cfg.post_lifecycle_direction_weight,
            max_cost=cfg.post_lifecycle_max_cost,
            appearance_bank=None,
        )

        if cheb_gr_extractor is not None and occ_audit_enabled:
            seq_img_dir = str(_Path(cfg.core.data_root) / cfg.core.split / seq / "img1")
            if getattr(cfg, "occ_audit_bank_reference", False):
                from .clean_fifo_bank import build_filled_bank
                from .occ_audit import (
                    extract_audit_embeddings_post_exit,
                    occ_exit_audit_lines_from_bank,
                )

                occ_bank = build_filled_bank(
                    results_lines,
                    seq_img_dir,
                    cheb_gr_extractor,
                    appearance_occlusion_cov=cfg.appearance_occlusion_cov,
                    fifo_n=getattr(cfg, "occ_audit_bank_n", 20),
                    crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                )
                audit_embs = extract_audit_embeddings_post_exit(
                    results_lines,
                    seq_img_dir,
                    cheb_gr_extractor,
                    ref_n=cfg.occ_audit_ref_n,
                    audit_crops=cfg.occ_audit_crops,
                    audit_window=cfg.occ_audit_window,
                    min_occ_frames=cfg.occ_audit_min_occ,
                    crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                    appearance_occlusion_cov=cfg.appearance_occlusion_cov,
                )
                oa_log_rows: list[dict[str, Any]] = []
                results_lines, oa_stats = occ_exit_audit_lines_from_bank(
                    results_lines,
                    occ_bank,
                    audit_embs,
                    enabled=True,
                    tau=cfg.occ_audit_tau,
                    min_ref=cfg.occ_audit_min_ref,
                    ref_n=cfg.occ_audit_ref_n,
                    audit_crops=cfg.occ_audit_crops,
                    audit_window=cfg.occ_audit_window,
                    min_occ_frames=cfg.occ_audit_min_occ,
                    appearance_occlusion_cov=cfg.appearance_occlusion_cov,
                    decision_log=oa_log_rows
                    if getattr(cfg, "occ_audit_log", False)
                    else None,
                )
            else:
                audit_embs = extract_audit_embeddings(
                    results_lines,
                    seq_img_dir,
                    cheb_gr_extractor,
                    ref_n=cfg.occ_audit_ref_n,
                    audit_crops=cfg.occ_audit_crops,
                    audit_window=cfg.occ_audit_window,
                    min_occ_frames=cfg.occ_audit_min_occ,
                    crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                    appearance_occlusion_cov=cfg.appearance_occlusion_cov,
                )
                oa_log_rows: list[dict[str, Any]] = []
                results_lines, oa_stats = occ_exit_audit_lines(
                    results_lines,
                    audit_embs,
                    enabled=True,
                    tau=cfg.occ_audit_tau,
                    min_ref=cfg.occ_audit_min_ref,
                    ref_n=cfg.occ_audit_ref_n,
                    audit_crops=cfg.occ_audit_crops,
                    audit_window=cfg.occ_audit_window,
                    min_occ_frames=cfg.occ_audit_min_occ,
                    appearance_occlusion_cov=cfg.appearance_occlusion_cov,
                    decision_log=oa_log_rows
                    if getattr(cfg, "occ_audit_log", False)
                    else None,
                )
            if getattr(cfg, "occ_audit_log", False) and oa_log_rows:
                _append_dict_csv(
                    occ_audit_log_path,
                    [{"seq": seq, **row} for row in oa_log_rows],
                )
            print(
                f"  {seq}: occ-audit {oa_stats['flags']} flags / "
                f"{oa_stats['audited']} audited "
                f"({oa_stats['episodes']} episodes, "
                f"no_ref={oa_stats['abstain_no_ref']} "
                f"no_crops={oa_stats['abstain_no_crops']}, "
                f"ids {oa_stats['ids_before']}->{oa_stats['ids_after']})"
            )

        if cheb_gr_extractor is not None and cheb_gr_online:
            seq_img_dir = str(_Path(cfg.core.data_root) / cfg.core.split / seq / "img1")
            head_embs, bank_embs = extract_handover_embeddings(
                results_lines,
                seq_img_dir,
                cheb_gr_extractor,
                decide_n=cfg.cheb_gr_online_decide_n,
                n_samples=cfg.cheb_gr_merge_n_samples,
                crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                appearance_occlusion_cov=cfg.appearance_occlusion_cov,
                neighbor_iou_max=cfg.cheb_gr_online_neighbor_iou_max,
                bank_mode=cfg.cheb_gr_online_bank_mode,
                bank_n=cfg.cheb_gr_online_bank_n,
            )
            ho_log_rows: list[dict[str, Any]] = []
            results_lines, ho_stats = causal_handover_lines(
                results_lines,
                head_embs,
                bank_embs,
                enabled=True,
                max_cost=cfg.cheb_gr_online_max_cost,
                max_gap=cfg.cheb_gr_merge_max_gap,
                decide_n=cfg.cheb_gr_online_decide_n,
                min_head_samples=cfg.cheb_gr_online_min_head,
                margin=cfg.cheb_gr_online_margin,
                key_sim_min=cfg.cheb_gr_online_key_sim_min,
                key_sim_cost_floor=cfg.cheb_gr_online_key_sim_cost_floor,
                key_margin_min=cfg.cheb_gr_online_key_margin_min,
                center_dist_veto=cfg.cheb_gr_online_center_dist_veto,
                pollution_veto=cfg.cheb_gr_online_pollution_veto,
                pool_frac=cfg.cheb_gr_pool_frac,
                cheb_lambda=cfg.cheb_gr_lambda,
                k2=cfg.cheb_gr_k2,
                max_fwd=cfg.cheb_gr_max_fwd,
                fuse_lambda=cfg.cheb_gr_fuse_lambda,
                decision_log=ho_log_rows
                if getattr(cfg, "cheb_gr_online_log", False)
                else None,
            )
            if getattr(cfg, "cheb_gr_online_log", False) and ho_log_rows:
                _append_dict_csv(
                    cheb_gr_online_log_path,
                    [{"seq": seq, **row} for row in ho_log_rows],
                )
            print(
                f"  {seq}: cheb-gr offline handover {ho_stats['ids_before']}→"
                f"{ho_stats['ids_after']} ({ho_stats['handovers']} handovers, "
                f"{ho_stats['events_with_candidates']}/{ho_stats['events']} "
                "events had candidates, "
                f"reject_cost={ho_stats['reject_cost']} "
                f"reject_margin={ho_stats['reject_margin']} "
                f"reject_key_sim={ho_stats['reject_key_sim']} "
                f"reject_key_margin={ho_stats['reject_key_margin']} "
                f"reject_center_dist={ho_stats['reject_center_dist']} "
                f"reject_pollution={ho_stats['reject_pollution']} "
                f"reject_min_head={ho_stats['reject_min_head']})"
            )
        elif cheb_gr_extractor is not None:
            seq_img_dir = str(_Path(cfg.core.data_root) / cfg.core.split / seq / "img1")
            cheb_embeddings = extract_tracklet_embeddings(
                results_lines,
                seq_img_dir,
                cheb_gr_extractor,
                n_samples=cfg.cheb_gr_merge_n_samples,
                crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                appearance_occlusion_gate=(
                    cfg.appearance_occlusion_gate
                    or getattr(cfg, "cheb_gr_model", "") == "mobilenetv4_reid"
                ),
                appearance_occlusion_cov=cfg.appearance_occlusion_cov,
            )
            results_lines, cheb_stats = cheb_gr_merge_output_tracklets(
                results_lines,
                cheb_embeddings,
                enabled=True,
                max_cost=cfg.cheb_gr_merge_max_cost,
                max_gap=cfg.cheb_gr_merge_max_gap,
                min_overlap_frames=cfg.cheb_gr_merge_min_overlap,
                pool_frac=cfg.cheb_gr_pool_frac,
                cheb_lambda=cfg.cheb_gr_lambda,
                k2=cfg.cheb_gr_k2,
                max_fwd=cfg.cheb_gr_max_fwd,
                fuse_lambda=cfg.cheb_gr_fuse_lambda,
            )
            print(
                f"  {seq}: cheb-gr merge {cheb_stats['ids_before']}→"
                f"{cheb_stats['ids_after']} ({cheb_stats['merges']} merges)"
            )

        results_lines, quality_stats = filter_low_quality_tracklets(
            results_lines,
            min_len=cfg.min_tracklet_len,
            min_score=cfg.min_tracklet_score,
        )
        if quality_stats["removed"] > 0:
            print(
                f"  {seq}: quality filter removed {quality_stats['removed']} tracklets"
            )

        if cfg.interpolate_tracklets:
            results_lines, interp_stats = interpolate_tracklets(
                results_lines,
                max_gap=cfg.interpolate_max_gap,
                min_track_len=cfg.interpolate_min_track_len,
                min_h=cfg.interpolate_min_h,
            )
            print(
                f"  {seq}: interpolation gaps={interp_stats['gaps_filled']} "
                f"frames_added={interp_stats['frames_added']}"
            )

        _Path(output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        print(f"✅ {seq} written ({len(results_lines)} lines)")

    from .metrics import run_motmetrics_evaluation

    return run_motmetrics_evaluation(
        data_root=cfg.core.data_root,
        split=cfg.core.split,
        output=str(cfg.output_root),
        sequences=",".join(cfg.seqs),
        detector=cfg.kwargs.get("detector"),
        score_on_gt_frames=bool(cfg.kwargs.get("score_on_gt_frames", False)),
    )
