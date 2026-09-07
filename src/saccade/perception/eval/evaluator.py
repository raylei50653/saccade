# mypy: ignore-errors
import json
import os
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from typing import Any

from .quality import (
    compute_detection_quality_batch as _compute_detection_quality_batch,
)
from .utils import (
    append_dict_csv as _append_dict_csv,
    parse_debug_frame_ranges as _parse_debug_frame_ranges,
    debug_frame_selected as _debug_frame_selected,
    append_stage_dump_rows as _append_stage_dump_rows,
    safe_cpp_ptr as _safe_cpp_ptr,
    mot_result_line as _mot_result_line,
    apply_narrow_person_score_bonus as _apply_narrow_person_score_bonus,
    tile_seam_mask as _tile_seam_mask,
    count_tile_seam_boxes as _count_tile_seam_boxes,
)
from .post_merge import (
    post_merge_output_tracklets,
    apply_deferred_alias,
    filter_low_quality_tracklets,
    interpolate_tracklets,
)
from .external_fp_model import (
    RuleBaselineConfig,
    load_logistic_model,
)
from .helpers import (
    read_deferred_result as _read_deferred_result,
    fast_emit_mot_lines as _fast_emit_mot_lines,
)

# Perception/eval modules load local extensions before any torchvision fallback.
from saccade.perception.cropper import ZeroCopyCropper


# Cross-thread coordination handles for multi-stream eval. They are populated by
# MultiStreamRunner (see multi_stream.py) before workers are submitted and reset
# to None afterwards; they must exist at module scope so workers can read them
# without AttributeError even when multi-stream is not in use.
_worker_init_lock: "threading.Lock | None" = None
_worker_init_barrier: "threading.Barrier | None" = None
_graph_capture_lock: "threading.Lock | None" = None


def _release_worker_init_lock() -> None:
    """Release the per-wave worker-init lock if this thread holds it.

    Safe to call multiple times and when the lock is absent or already
    released, so it can be used both as an explicit hand-off point and as a
    finally-block safety net.
    """
    lock = _worker_init_lock
    if lock is None:
        return
    try:
        lock.release()
    except RuntimeError:
        # Lock was not held (already released or never acquired) — no-op.
        pass


from saccade.perception.detector_trt import (  # noqa: E402
    TRTYoloDetector,
    TwostageDetector,
    ConcurrentDetectorProxy,
    BatchedDetectorProxy,
)
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

from saccade.perception.eval.detection import (  # noqa: E402
    detect_adaptive_960_tiled,
    detect_mamba_global_2x2,
    detect_960p_3x2_tiled,
    detect_native_640,
    detect_native_960,
    detect_native_960_tta,
    detect_sahi_960p_2x2,
    filter_detections_fast,
    match_keypoints_to_boxes,
    merge_cross_tile_duplicates_fast,
    nms_fast,
)
from saccade.perception.eval.pool import (  # noqa: E402
    rgb_chw_to_nv12_gpu,
    rgb_hwc_to_nv12_gpu,
)
from saccade.perception.eval.preprocess import (  # noqa: E402
    apply_frame_preprocess,
)

try:
    from saccade_tracking_ext import (
        TrackletLifecycleMerger as _CppTrackletLifecycleMerger,
        IdentityResolver as _CppIdentityResolver,
    )

    _LIFECYCLE_CLS: type | None = _CppTrackletLifecycleMerger
except ImportError:
    _CppIdentityResolver = None
    _LIFECYCLE_CLS = None
from saccade.perception.eval.tracking import GlobalTrackIdMapper  # noqa: E402

try:
    from saccade_tracking_ext import (
        PerceptionPipeline,
        PerceptionPipelineConfig,
        copy_pad_detections,
    )
except ImportError:
    PerceptionPipeline = None
    PerceptionPipelineConfig = None
    copy_pad_detections = None


# Functions moved to output_bank.py and helpers.py


# Frame tracking helpers moved to helpers.py and utils.py


# Internal helpers moved to helpers.py, quality.py, and utils.py


# Post-merge functions moved to post_merge.py


from .detection_filters import (  # noqa: E402,F401
    _FP_HARD_REJECT_SCORE,
    _SOFTMAX3_TORCH_CACHE,
    _append_private_continuation_candidates,
    _apply_detection_cap,
    _apply_external_fp_filter,
    _apply_fp_hard_filter,
    _apply_private_energy_margin,
    _apply_stage2_quality_gate,
    _build_active_track_priors,
    _compute_adaptive_cap,
    _compute_fp_filter_ranking,
    _consecutive_birth_check,
    _fp_hard_reject_mask,
    _get_softmax3_torch_params,
    _predict_softmax3_probs_torch,
    _prior_iou_and_center_distance,
    _private_height_log_ratio,
    _private_prior_pair_keep,
    _sparse_symmetric_detection_support,
    _suppress_duplicate_detections,
)


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized not in {"0", "false", "no", "off"}


def _flush_deferred_emit(
    event: "torch.cuda.Event",
    pinned: dict[str, torch.Tensor],
    *,
    default_class_id: int | None,
    global_id_mapper: Any,
    seq: str,
    frame_id: int,
    frame_w: int,
    frame_h: int,
) -> tuple[list[str], set[int]]:
    """Read one deferred async-materialize result and emit its MOT lines.

    Mirrors the inline ``_defer_emit`` read that runs at the top of frame N+1
    (and once more after the loop to flush the final frame). Returns the emitted
    lines plus the track-id set the caller assigns to ``prev_track_ids`` (the
    end-of-sequence flush discards it).
    """
    track_results = _read_deferred_result(
        event,
        pinned,
        default_class_id=default_class_id,
        include_det_idx=False,
    )
    lines: list[str] = []
    if track_results["count"] > 0:
        lines = _fast_emit_mot_lines(
            track_results=track_results,
            global_id_mapper=global_id_mapper,
            seq=seq,
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
        )
    track_ids = set(int(x) for x in track_results["ids"].tolist())
    return lines, track_ids


from .cpp_runner import (  # noqa: E402,F401
    _build_cpp_seq_config,
    run_eval_cpp,
)
from .pipeline import (  # noqa: E402,F401
    DetectionContract,
    EvalPipeline,
    FPNConfig,
    _build_gmc_estimator,
    _detect_barrier_mode,
    _double_buffer_eligible,
    _explicit_stream_probe_enabled,
    _stream_mode_ptds_probe,
    _resolve_kalman_fps,
)
from .stages import (  # noqa: E402,F401
    FrameCtx,
    PreparedDetection,
    _flush_db_tracker_out,
    _launch_double_buffer_detect,
    _record_profile_scope,
    _run_birth_config,
    _run_detect,
    _run_detection_filters,
    _run_emit,
    _run_gmc_estimate,
    _run_materialize,
    _run_native_tensor_prep,
    _run_nms,
    _run_nms_shadow_compare,
    _capture_main_nms_graph,
    _run_post_nms_finalize,
    _run_reid_and_gmc,
    _run_track,
    _stash_crop_ring,
)


def _record_frame_timing(state: EvalPipeline, *, latency_started_at: float) -> None:
    """Record latency and throughput without deriving one from the other."""

    if state.current_frame_id <= state.warmup_frames:
        return
    completed_at = time.perf_counter()
    state.frame_latencies.append((completed_at - latency_started_at) * 1000.0)
    state.throughput_frames += 1
    state.throughput_finished_at = completed_at


def _run_frame(
    state: "EvalPipeline",
    *,
    frame_id: int,
    prepared_detection: PreparedDetection | None = None,
) -> bool:
    """Execute one frame. Returns False to stop iteration, True to continue."""
    state.current_frame_id = frame_id
    if frame_id == state.warmup_frames + 1:
        # This is the start of the measured pipeline interval, independently of
        # this frame's launch-to-output latency.
        state.throughput_started_at = time.perf_counter()
    # ── flush deferred tracker output from the previous frame ────────────
    if state.db_emit_frame_id > 0 and state.db_emit_event is not None:
        _flush_db_tracker_out(state)
    # --- unpack pure-input fields ---
    _annotate_birth_events = state.annotate_birth_events
    _append_birth_event_rows = state.append_birth_event_rows
    _consec_birth_window = state._consec_birth_window
    _defer_emit = state.defer_emit
    _fpn_backbone = state.fpn.backbone
    _fpn_img_size = state.fpn.img_size
    _fpn_reid_conv_weights = state.fpn.conv_weights
    _fpn_reid_dim = state.contract.feature_dim
    _fpn_reid_mode = state.contract.fpn_reid_mode
    _fpn_reid_proj_weight = state.fpn.proj_weight
    _fpn_reid_running_mean = state.fpn.running_mean
    _fpn_reid_running_var = state.fpn.running_var
    _multi_birth_manager = state._multi_birth_manager
    _pinned_result_bufs = state.pinned_result_bufs
    _scene_policy = state._scene_policy
    _use_direct_gmc = state.use_direct_gmc
    cfg = state.cfg
    cropper = state.cropper
    debug_dump_frames = state.debug_dump_frames
    debug_dump_seq = state.debug_dump_seq
    debug_stage_dump_rows = state.debug_stage_dump_rows
    detect_fn = state.detect_fn
    detector = state.detector
    detector_box_format = state.detector_box_format
    enable_onms = state.enable_onms
    extractor = state.extractor
    frame_end = state.frame_end
    global_id_mapper = state.global_id_mapper
    h_orig = state.h_orig
    lazy_reid_prev_embeddings = state.lazy_reid_prev_embeddings
    nv12_direct_from_hwc = state.nv12_direct_from_hwc
    onms_min_track_age = state.onms_min_track_age
    onms_min_track_score = state.onms_min_track_score
    onms_prior_iou_threshold = state.onms_prior_iou_threshold
    perception_pipeline = state.perception_pipeline
    pool = state.pool
    profile_stages = state.profile_stages
    record_stage_sample = state.record_stage_sample
    results_lines = state.results_lines
    seq = state.seq
    seq_post_counts = state.seq_post_counts
    seq_segment_samples = state.seq_segment_samples
    seq_stage_samples = state.seq_stage_samples
    seq_stage_totals = state.seq_stage_totals
    seq_tile_diag = state.seq_tile_diag
    stream_iter = state.stream_iter
    time_stage = state.time_stage
    top_level_stage_names = state.top_level_stage_names
    w_orig = state.w_orig
    warmup_frames = state.warmup_frames
    wb = state.wb
    wb_scene_policy = state.wb_scene_policy
    _stage_probe_callback = getattr(state, "stage_probe_callback", None)
    # -----------------------------------------------
    if _defer_emit and state.defer_emit_event is not None:
        _lines, state.prev_track_ids = _flush_deferred_emit(
            state.defer_emit_event,
            _pinned_result_bufs,
            default_class_id=cfg.detection.person_class
            if cfg.detection.track_person_only
            else None,
            global_id_mapper=global_id_mapper,
            seq=seq,
            frame_id=state.defer_emit_fid,
            frame_w=w_orig,
            frame_h=h_orig,
        )
        results_lines.extend(_lines)
        state.defer_emit_event = None
    current_stage_sample_active = frame_id > warmup_frames
    current_frame_stage_elapsed = (
        {name: 0.0 for name in top_level_stage_names}
        if current_stage_sample_active
        else None
    )
    if state.frame_ledger is not None:
        state._frame_det_counts = None
        state._frame_stage_times = None
        state._frame_reid_stats = None
        if state.perception_pipeline is not None:
            try:
                state._prev_post_sync_stats = (
                    state.perception_pipeline.get_postprocess_profile_stats()
                )
            except Exception:
                state._prev_post_sync_stats = None
    t_e2e_start = time.perf_counter()
    if state.frame_ledger is not None and frame_id > warmup_frames:
        state._frame_stage_times = {}
    _t_ledger_last: float | None = (
        t_e2e_start
        if (state.frame_ledger is not None and frame_id > warmup_frames)
        else None
    )

    def _ledger_stage_done(name: str) -> None:
        nonlocal _t_ledger_last
        st = state._frame_stage_times
        if _t_ledger_last is not None and st is not None:
            now = time.perf_counter()
            st[name] = round((now - _t_ledger_last) * 1000, 6)
            _t_ledger_last = now

    _reid_side_pending = False
    _reid_async_embeddings: torch.Tensor | None = None
    _reid_async_indices: torch.Tensor | None = None
    _reid_frame_hwc_ref: torch.Tensor | None = None
    if prepared_detection is None:
        try:
            frame_gpu, _fetch_ms = time_stage(
                seq_stage_totals,
                "fetch",
                lambda: next(stream_iter),
                sync_cuda=False,
            )
        except StopIteration:
            return False
    else:
        if prepared_detection.frame_id != frame_id:
            raise ValueError(
                "prepared detection frame does not match tracker frame: "
                f"{prepared_detection.frame_id} != {frame_id}"
            )
        # ``wait_event`` is a stream dependency, not a host/device barrier.
        # It makes the side-stream detector result visible before this frame's
        # postprocess while allowing the preceding tracker update to overlap.
        torch.cuda.current_stream().wait_event(prepared_detection.ready_event)
        pool = prepared_detection.pool
        state.pool = pool
        frame_gpu = prepared_detection.frame_gpu
        t_frame_start = prepared_detection.latency_started_at
    if prepared_detection is None:
        # End-to-end latency: clock from before decode/ingest (t_e2e_start),
        # not after, so the reported latency includes JPEG decode and matches
        # the wall-clock throughput period.
        t_frame_start = t_e2e_start

    if getattr(cfg, "workbench", False) and wb is not None:
        _, _ingest_elapsed = time_stage(
            seq_stage_totals,
            "ingest_preprocess",
            lambda: (
                (
                    pool.frame_buffer_nv12.copy_(rgb_hwc_to_nv12_gpu(frame_gpu)),
                    pool.mark_nv12_current(),
                )
                if nv12_direct_from_hwc
                else (
                    pool.frame_buffer.copy_(frame_gpu.permute(2, 0, 1).float() / 255.0),
                    apply_frame_preprocess(
                        pool.frame_buffer,
                        cfg.preprocess_modes,
                        cfg.detection.gamma,
                        cfg.detection.gamma_luma_threshold,
                        cfg.detection.contrast,
                    ),
                    pool.mark_rgb_current(),
                    (
                        pool.frame_buffer_nv12.copy_(
                            rgb_chw_to_nv12_gpu(pool.frame_buffer)
                        )
                        if pool.use_nv12
                        else None
                    ),
                )
            ),
            sync_cuda=False,
        )

        # Fetch priors for ONMS if enabled
        priors_tensor, prior_classes_tensor = None, None
        if enable_onms:
            priors_tensor, prior_classes_tensor = _build_active_track_priors(
                detector.tracker,
                pool.frame_buffer.device,
                min_track_age=onms_min_track_age,
                min_track_score=onms_min_track_score,
            )

        # Process frame
        wb_result, _detect_elapsed = time_stage(
            seq_stage_totals,
            "detect",
            lambda: wb.process_frame(
                pool.as_rgb_chw(),
                frame_w=w_orig,
                frame_h=h_orig,
                priors=priors_tensor if enable_onms else None,
                prior_classes=prior_classes_tensor if enable_onms else None,
            ),
            sync_cuda=True,
        )

        track_results = {
            "count": len(wb_result.ids),
            "ids": wb_result.ids,
            "boxes": wb_result.boxes,
            "scores": wb_result.scores,
            "classes": wb_result.classes,
            "det_idx": wb_result.det_idx,
        }

        # Mock missing variables
        fused_boxes = wb_result.boxes
        fused_scores = wb_result.scores
        fused_classes = wb_result.classes
        geometry_suspect_mask = torch.zeros(
            len(fused_boxes), dtype=torch.bool, device=fused_boxes.device
        )
        embeddings = None
        state.gmc_warp = None
        aligned_keypoints = None
        raw_dump_boxes = fused_boxes
        raw_dump_scores = fused_scores
        raw_dump_classes = fused_classes

        # Update seq_stage_totals for the skipped stages to match legacy timing expectations
        seq_stage_totals["postprocess"] += 0.0
        seq_stage_totals["track"] += 0.0
        seq_stage_totals["materialize"] += 0.0
    else:
        if getattr(cfg, "workbench", False) and wb is not None:
            # Step 1: ingest + preprocess (same as non-workbench path)
            _, _ = time_stage(
                seq_stage_totals,
                "ingest_preprocess",
                lambda: (
                    (
                        pool.frame_buffer_nv12.copy_(rgb_hwc_to_nv12_gpu(frame_gpu)),
                        pool.mark_nv12_current(),
                    )
                    if nv12_direct_from_hwc
                    else (
                        pool.frame_buffer.copy_(
                            frame_gpu.permute(2, 0, 1).float() / 255.0
                        ),
                        apply_frame_preprocess(
                            pool.frame_buffer,
                            cfg.preprocess_modes,
                            cfg.detection.gamma,
                            cfg.detection.gamma_luma_threshold,
                            cfg.detection.contrast,
                        ),
                        pool.mark_rgb_current(),
                        (
                            pool.frame_buffer_nv12.copy_(
                                rgb_chw_to_nv12_gpu(pool.frame_buffer)
                            )
                            if pool.use_nv12
                            else None
                        ),
                    )
                ),
                sync_cuda=True,
            )

            # Step 2: YOLO detection via detect_fn — produces boxes in original coords
            (
                (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    is_tiled,
                    source_keypoints,
                ),
                _,
            ) = time_stage(
                seq_stage_totals,
                "detect",
                lambda: detect_fn(
                    detector,
                    pool,
                    h_orig,
                    w_orig,
                    cfg.preprocess_modes,
                    detector_box_format,
                ),
                sync_cuda=True,
            )

            # ── Scene-adapt: observe & classify on first frames ────────
            if wb_scene_policy is not None and not wb_scene_policy.is_classified:
                wb_scene_policy.observe(fused_boxes, fused_scores, w_orig, h_orig)
                if wb_scene_policy.is_classified and wb_scene_policy.stats is not None:
                    st = wb_scene_policy.stats
                    if st.scene_type == "crowded_narrow":
                        state.seq_narrow_bonus = cfg.detection.narrow_person_score_bonus
                    wb.narrow_bonus = state.seq_narrow_bonus
                    print(
                        f"  [scene_adapt] {seq} @ frame {frame_id}: {st}"
                        + (
                            f" → narrow_bonus={state.seq_narrow_bonus:.2f}"
                            if cfg.detection.scene_adapt_enabled
                            and cfg.detection.narrow_person_score_bonus > 0
                            else ""
                        )
                    )

            # ── Quality-aware processing (baseline-aligned pipeline) ────
            wb_result, _ = time_stage(
                seq_stage_totals,
                "postprocess",
                lambda: wb.process_detections_quality_aware(
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    frame_w=w_orig,
                    frame_h=h_orig,
                    frame_chw=pool.as_rgb_chw(),
                    frame_id=frame_id,
                    last_reid_frame=state.last_reid_frame,
                    prev_gray=state.prev_gray,
                    is_tiled=is_tiled,
                ),
                sync_cuda=True,
            )
            # Update scene-adapt narrow bonus from workbench
            state.seq_narrow_bonus = wb.narrow_bonus
            state.last_reid_frame = wb.last_reid_frame

            # D2H once, share across tracker_result_buffers, track_results, and
            # MOT line writing — avoids 10 redundant device syncs (was 13 .cpu()
            # calls per frame × 7 threads = 91 syncs/cycle; now 5).
            wb_count = int(len(wb_result.ids))
            _wb_boxes_cpu = wb_result.boxes.cpu()
            _wb_scores_cpu = wb_result.scores.cpu()
            _wb_ids_cpu = wb_result.ids.cpu()
            _wb_classes_cpu = wb_result.classes.cpu()
            _wb_det_idx_cpu = wb_result.det_idx.cpu()

            state.tracker_result_buffers = {
                "count": torch.tensor([wb_count], dtype=torch.int32, device="cpu"),
                "boxes": _wb_boxes_cpu,
                "scores": _wb_scores_cpu,
                "ids": _wb_ids_cpu,
                "classes": _wb_classes_cpu,
                "det_idx": _wb_det_idx_cpu,
            }
            track_results = {
                "count": wb_count,
                "boxes": _wb_boxes_cpu,
                "scores": _wb_scores_cpu,
                "ids": _wb_ids_cpu,
                "classes": _wb_classes_cpu,
                "det_idx": _wb_det_idx_cpu,
            }

            # Write MOT result lines for workbench path (C++ already tracked).
            # Build lines: "frame_id, global_tid, x1, y1, w, h, score, -1, -1, -1"
            wb_mot_lines: list[str] = []
            if wb_count > 0:
                wb_ids_np = _wb_ids_cpu.numpy().astype(int)
                wb_boxes_np = _wb_boxes_cpu.numpy()
                wb_scores_np = _wb_scores_cpu.numpy()
                for i in range(wb_count):
                    global_tid = int(global_id_mapper.map(seq, wb_ids_np[i]))
                    box = (
                        float(wb_boxes_np[i, 0]),
                        float(wb_boxes_np[i, 1]),
                        float(wb_boxes_np[i, 2]),
                        float(wb_boxes_np[i, 3]),
                    )
                    score = float(wb_scores_np[i])
                    wb_mot_lines.append(
                        _mot_result_line(
                            frame_id, global_tid, box, score, w_orig, h_orig
                        )
                    )

            # Add MOT lines to results_lines so they go through post-processing
            if wb_mot_lines:
                results_lines.extend(wb_mot_lines)

            # Update prev_track_ids for downstream (lazy ReID, etc.)
            if wb_count > 0:
                state.prev_track_ids = set(wb_ids_np.tolist())

            # Mock missing variables for downstream compatibility
            fused_boxes = wb_result.boxes
            fused_scores = wb_result.scores
            fused_classes = wb_result.classes
            geometry_suspect_mask = torch.zeros(
                len(fused_boxes), dtype=torch.bool, device=fused_boxes.device
            )
            embeddings = None
            state.gmc_warp = None
            aligned_keypoints = None
            raw_dump_boxes = fused_boxes
            raw_dump_scores = fused_scores
            raw_dump_classes = fused_classes
            if _stage_probe_callback is not None:
                _stage_probe_callback(
                    seq,
                    frame_id,
                    "detector_output",
                    raw_dump_boxes,
                    raw_dump_scores,
                    raw_dump_classes,
                )
            frame_birth_events = []

            # Update seq_stage_totals for the skipped stages
            seq_stage_totals["postprocess"] += 0.0
            seq_stage_totals["track"] += 0.0
            seq_stage_totals["materialize"] += 0.0

            # Save gray frame for GMC in next frame
            state.prev_gray = pool.get_frame_luma().clone()

            if _t_ledger_last is not None and state._frame_stage_times is not None:
                state._frame_stage_times["fetch"] = round(_ingest_elapsed, 6)
                state._frame_stage_times["detect"] = round(_detect_elapsed, 6)
                _t_ledger_last = time.perf_counter()
        else:
            _ledger_stage_done("fetch")
            if _explicit_stream_probe_enabled() and state._pp_streams[0]:
                _p = frame_id % 2
                state.stream_detect = state._pp_streams[_p]["detect"]
                state.stream_post = state._pp_streams[_p]["post"]
                state.stream_detect_event = state._pp_detect_done[_p]
            if prepared_detection is None:
                (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    is_tiled,
                    source_keypoints,
                ) = _run_detect(
                    state,
                    pool=pool,
                    frame_gpu=frame_gpu,
                    nv12_direct_from_hwc=nv12_direct_from_hwc,
                    detect_fn=detect_fn,
                    detector_box_format=detector_box_format,
                )
            else:
                fused_boxes = prepared_detection.fused_boxes
                fused_scores = prepared_detection.fused_scores
                fused_classes = prepared_detection.fused_classes
                is_tiled = prepared_detection.is_tiled
                source_keypoints = prepared_detection.source_keypoints

            if _explicit_stream_probe_enabled() and state._pp_streams[0]:
                _pp = frame_id % 2
                state._pp_fused[_pp] = (
                    fused_boxes.clone(),
                    fused_scores.clone(),
                    fused_classes.clone(),
                )
                state._pp_streams[_pp]["is_tiled"] = is_tiled
                state._pp_streams[_pp]["source_keypoints"] = source_keypoints

            _ledger_stage_done("detect")
            _fpn_cache: dict[str, torch.Tensor] = {}
            if _fpn_backbone is not None and fused_boxes.numel() > 0:
                _fpn_in = pool.canvas_960p.unsqueeze(0)
                if _fpn_in.shape[2] != _fpn_img_size:
                    _fpn_in = torch.nn.functional.interpolate(
                        _fpn_in,
                        size=(_fpn_img_size, _fpn_img_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                p3, p4, p5 = _fpn_backbone.infer(_fpn_in)
                _fpn_cache = {"p3": p3, "p4": p4, "p5": p5}

            source_boxes_for_keypoints = fused_boxes
            debug_dump_active = _debug_frame_selected(
                seq,
                frame_id,
                debug_dump_seq,
                debug_dump_frames,
            )
            raw_dump_boxes = fused_boxes
            raw_dump_scores = fused_scores
            raw_dump_classes = fused_classes
            if _stage_probe_callback is not None:
                _stage_probe_callback(
                    seq,
                    frame_id,
                    "detector_output",
                    raw_dump_boxes,
                    raw_dump_scores,
                    raw_dump_classes,
                )

            # P5-4: scene-adaptive observation and one-shot classification.
            if _scene_policy is not None and not _scene_policy.is_classified:
                _scene_policy.observe(fused_boxes, fused_scores, w_orig, h_orig)
                if _scene_policy.is_classified and _scene_policy.stats is not None:
                    st = _scene_policy.stats
                    if st.scene_type == "crowded_narrow":
                        state.seq_narrow_bonus = cfg.detection.narrow_person_score_bonus
                    print(
                        f"  [scene_adapt] {seq} @ frame {frame_id}: {st}"
                        + (
                            f" → narrow_bonus={state.seq_narrow_bonus:.2f}"
                            if cfg.detection.scene_adapt_enabled
                            and cfg.detection.narrow_person_score_bonus > 0
                            else ""
                        )
                    )

            if fused_boxes.numel() == 0:
                if debug_dump_active:
                    _append_stage_dump_rows(
                        debug_stage_dump_rows,
                        seq=seq,
                        frame_id=frame_id,
                        stage="raw",
                        boxes=raw_dump_boxes,
                        scores=raw_dump_scores,
                        classes=raw_dump_classes,
                    )
                _record_frame_timing(state, latency_started_at=t_frame_start)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (
                        time.perf_counter() - t_e2e_start
                    ) * 1000
                    state.seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                return True

            t_keypoint_align_start = None
            if profile_stages:
                torch.cuda.synchronize()
                t_keypoint_align_start = time.perf_counter()
            aligned_keypoints = match_keypoints_to_boxes(
                fused_boxes,
                source_boxes_for_keypoints,
                source_keypoints,
            )
            if (
                profile_stages
                and current_stage_sample_active
                and t_keypoint_align_start is not None
            ):
                torch.cuda.synchronize()
                seq_stage_totals["post_keypoint_align"] += (
                    time.perf_counter() - t_keypoint_align_start
                ) * 1000

            # Keep low-score boxes down to cfg.core.track_thresh so ByteTrack's
            # second-stage association can actually use them.
            post_gpu_start_event = None
            post_gpu_end_event = None
            # Sync-free CUDA-event markers partitioning the postprocess GPU span
            # (post_gpu_elapsed) into contiguous segments. Each entry carries the
            # name of the segment ENDING at that marker; the residual from the last
            # marker to post_gpu_end_event is attributed to "post_seg_python_tail".
            # No torch.cuda.synchronize() is inserted between markers, so timing is
            # not distorted. Used only to locate the "unattributed GPU" residual.
            post_seg_events: list[tuple[str, "torch.cuda.Event"]] = []
            if profile_stages:
                torch.cuda.synchronize()
                t_post_start = time.perf_counter()
                post_gpu_start_event = torch.cuda.Event(enable_timing=True)
                post_gpu_end_event = torch.cuda.Event(enable_timing=True)
                post_gpu_start_event.record(torch.cuda.current_stream())
            raw_box_count = int(fused_scores.numel())
            private_added_count = 0

            if state._frame_stage_times is not None:
                _t_post_sub = time.perf_counter()

            if _explicit_stream_probe_enabled() and state.stream_post is not None:
                torch.cuda.set_stream(state.stream_post)

            if perception_pipeline is not None:
                t_native_prep_start = None
                if profile_stages:
                    torch.cuda.synchronize()
                    t_native_prep_start = time.perf_counter()
                _fctx = _run_native_tensor_prep(
                    state,
                    fused_boxes=fused_boxes,
                    fused_scores=fused_scores,
                    fused_classes=fused_classes,
                    seq_narrow_bonus=state.seq_narrow_bonus,
                    enable_onms=enable_onms,
                    onms_min_track_age=onms_min_track_age,
                    onms_min_track_score=onms_min_track_score,
                    raw_box_count=raw_box_count,
                )
                raw_boxes_contig = _fctx.raw_boxes_contig
                raw_scores_contig = _fctx.raw_scores_contig
                raw_classes_contig = _fctx.raw_classes_contig
                num_priors = _fctx.num_priors
                if state._frame_stage_times is not None:
                    _t_post_after_prep = time.perf_counter()
                    state._frame_stage_times["post_tensor_prep"] = round(
                        (_t_post_after_prep - _t_post_sub) * 1000, 6
                    )
                if (
                    profile_stages
                    and current_stage_sample_active
                    and t_native_prep_start is not None
                ):
                    torch.cuda.synchronize()
                    seq_stage_totals["post_tensor_prep"] += (
                        time.perf_counter() - t_native_prep_start
                    ) * 1000
                if profile_stages and current_stage_sample_active:
                    _seg_ev = torch.cuda.Event(enable_timing=True)
                    _seg_ev.record(torch.cuda.current_stream())
                    post_seg_events.append(("post_seg_prep", _seg_ev))

                _shadow_enabled = os.environ.get("SACCADE_MAIN_NMS_SHADOW", "") in (
                    "1",
                    "true",
                    "yes",
                )
                _graph_shadow_enabled = os.environ.get(
                    "SACCADE_MAIN_NMS_GRAPH_SHADOW", ""
                ) in ("1", "true", "yes")
                _graphed_shadow_enabled = os.environ.get(
                    "SACCADE_MAIN_NMS_GRAPHED_SHADOW", ""
                ) in ("1", "true", "yes")
                if _shadow_enabled or _graph_shadow_enabled or _graphed_shadow_enabled:
                    n_post, state.nms_graph = _run_nms_shadow_compare(
                        state,
                        raw_boxes_contig=raw_boxes_contig,
                        raw_scores_contig=raw_scores_contig,
                        raw_classes_contig=raw_classes_contig,
                        raw_box_count=raw_box_count,
                        priors_tensor=_fctx.priors_tensor,
                        prior_classes_tensor=_fctx.prior_classes_tensor,
                        num_priors=num_priors,
                        private_prior_boxes=_fctx.private_prior_boxes,
                        num_private_priors=_fctx.num_private_priors,
                        native_private_enabled=_fctx.native_private_enabled,
                        is_tiled=is_tiled,
                        nms_graph=state.nms_graph,
                    )
                else:
                    n_post, state.nms_graph = _run_nms(
                        state,
                        raw_boxes_contig=raw_boxes_contig,
                        raw_scores_contig=raw_scores_contig,
                        raw_classes_contig=raw_classes_contig,
                        raw_box_count=raw_box_count,
                        priors_tensor=_fctx.priors_tensor,
                        prior_classes_tensor=_fctx.prior_classes_tensor,
                        num_priors=num_priors,
                        private_prior_boxes=_fctx.private_prior_boxes,
                        num_private_priors=_fctx.num_private_priors,
                        native_private_enabled=_fctx.native_private_enabled,
                        is_tiled=is_tiled,
                        nms_graph=state.nms_graph,
                    )
                if state._frame_stage_times is not None:
                    _t_now = time.perf_counter()
                    state._frame_stage_times["post_pre_nms"] = round(
                        (_t_now - _t_post_sub) * 1000, 6
                    )
                    _t_post_sub = _t_now
                    state._frame_stage_times["_t_post_after_nms"] = _t_now
                if profile_stages and current_stage_sample_active:
                    _post_stats = perception_pipeline.get_postprocess_profile_stats()
                    _post_filter_ms = float(_post_stats.get("filter_ms", 0.0))
                    _post_nms_ms = float(_post_stats.get("nms_ms", 0.0))
                    _post_count_sync_ms = float(_post_stats.get("count_d2h_ms", 0.0))
                    _post_total_ms = float(_post_stats.get("total_ms", 0.0))
                    _native_private_candidate_nms_ms = float(
                        _post_stats.get("native_private_candidate_nms_ms", 0.0)
                    )
                    _native_private_append_ms = float(
                        _post_stats.get("native_private_append_ms", 0.0)
                    )
                    _native_private_ms = (
                        _native_private_candidate_nms_ms + _native_private_append_ms
                    )
                    seq_stage_totals["post_filter"] += float(_post_filter_ms)
                    seq_stage_totals["post_nms"] += max(
                        0.0, float(_post_nms_ms) - _native_private_ms
                    )
                    seq_stage_totals["post_count_sync"] += _post_count_sync_ms
                    seq_stage_totals["native_filter_gather"] += float(
                        _post_stats.get("native_filter_gather_ms", 0.0)
                    )
                    seq_stage_totals["native_filter_kernel"] += float(
                        _post_stats.get("native_filter_kernel_ms", 0.0)
                    )
                    seq_stage_totals["native_gather_compact3"] += float(
                        _post_stats.get("native_gather_compact3_ms", 0.0)
                    )
                    seq_stage_totals["native_copy_suspect"] += float(
                        _post_stats.get("native_copy_suspect_ms", 0.0)
                    )
                    seq_stage_totals["native_filter_count_sync"] += float(
                        _post_stats.get("native_filter_count_sync_ms", 0.0)
                    )
                    seq_stage_totals["native_small_nms"] += float(
                        _post_stats.get("native_small_nms_ms", 0.0)
                    )
                    seq_stage_totals["native_suspect_penalty"] += float(
                        _post_stats.get("native_suspect_penalty_ms", 0.0)
                    )
                    seq_stage_totals["native_large_sort_nms"] += float(
                        _post_stats.get("native_large_sort_nms_ms", 0.0)
                    )
                    seq_stage_totals["native_large_argsort"] += float(
                        _post_stats.get("native_large_argsort_ms", 0.0)
                    )
                    seq_stage_totals["native_large_nms"] += float(
                        _post_stats.get("native_large_nms_ms", 0.0)
                    )
                    seq_stage_totals["native_compact_copy"] += float(
                        _post_stats.get("native_compact_copy_ms", 0.0)
                    )
                    seq_stage_totals["native_large_gather4"] += float(
                        _post_stats.get("native_large_gather4_ms", 0.0)
                    )
                    seq_stage_totals["native_large_copyback"] += float(
                        _post_stats.get("native_large_copyback_ms", 0.0)
                    )
                    seq_stage_totals["native_private_candidate_nms"] += (
                        _native_private_candidate_nms_ms
                    )
                    seq_stage_totals["native_private_append"] += (
                        _native_private_append_ms
                    )
                    seq_stage_totals["post_private_continuation"] += _native_private_ms
                    seq_stage_totals["post_native_other"] += max(
                        0.0,
                        _post_total_ms
                        - _post_filter_ms
                        - _post_nms_ms
                        - _post_count_sync_ms,
                    )
                    private_added_count = int(_post_stats.get("private_boxes", 0))
                if profile_stages and current_stage_sample_active:
                    _seg_ev = torch.cuda.Event(enable_timing=True)
                    _seg_ev.record(torch.cuda.current_stream())
                    post_seg_events.append(("post_seg_native", _seg_ev))
                (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    geometry_suspect_mask,
                    aligned_keypoints,
                    after_filter_count,
                    after_nms_count,
                ) = _run_post_nms_finalize(
                    state,
                    _fctx,
                    n_post=n_post,
                    source_boxes_for_keypoints=source_boxes_for_keypoints,
                    source_keypoints=source_keypoints,
                    current_stage_sample_active=current_stage_sample_active,
                    post_seg_events=post_seg_events,
                )
            else:
                fused_scores = _apply_narrow_person_score_bonus(
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    frame_w=w_orig,
                    frame_h=h_orig,
                    person_class=cfg.detection.person_class,
                    bonus=state.seq_narrow_bonus,
                    max_width_ratio=cfg.detection.narrow_person_max_width_ratio,
                    min_height_ratio=cfg.detection.narrow_person_min_height_ratio,
                    min_aspect=cfg.detection.narrow_person_min_aspect,
                    max_aspect=cfg.detection.narrow_person_max_aspect,
                )
                t_sub_start = time.perf_counter()
                if state._frame_stage_times is not None:
                    _t_pre_filter = time.perf_counter()
                keep_indices, geometry_suspect_mask, _ = filter_detections_fast(
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    score_threshold=min(
                        cfg.core.conf_threshold,
                        cfg.core.track_thresh,
                        cfg.detection.crowd_conf_threshold
                        if cfg.detection.crowd_low_score_mode
                        else cfg.core.conf_threshold,
                        cfg.geometry.crowd_track_thresh
                        if cfg.detection.crowd_low_score_mode
                        else cfg.core.track_thresh,
                    ),
                    track_person_only=cfg.detection.track_person_only,
                    person_class=cfg.detection.person_class,
                    is_tiled=is_tiled,
                    frame_w=w_orig,
                    frame_h=h_orig,
                    person_geometry_prior=cfg.geometry.person_geometry_prior,
                    geometry_suspect_support=cfg.geometry.geometry_suspect_support,
                    person_min_height_ratio=cfg.geometry.person_min_height_ratio,
                    person_min_aspect=cfg.geometry.person_min_aspect,
                    person_max_aspect=cfg.geometry.person_max_aspect,
                    person_min_area_ratio=cfg.geometry.person_min_area_ratio,
                    person_max_area_ratio=cfg.geometry.person_max_area_ratio,
                )
                fused_boxes = fused_boxes[keep_indices]
                fused_scores = fused_scores[keep_indices]
                fused_classes = fused_classes[keep_indices]
                if state._frame_stage_times is not None:
                    _t_after_filter = time.perf_counter()
                    state._frame_stage_times["_t_post_after_filter"] = _t_after_filter
                if aligned_keypoints is not None:
                    aligned_keypoints = aligned_keypoints[keep_indices]

                if _fpn_reid_mode and fused_boxes.numel() > 0:
                    valid_w = (fused_boxes[:, 2] - fused_boxes[:, 0]) > 0
                    if state._frame_stage_times is not None:
                        _t_any = time.perf_counter()
                    if not valid_w.all():
                        if state._frame_stage_times is not None:
                            _t_elapsed = (time.perf_counter() - _t_any) * 1000
                            state._frame_stage_times.setdefault("_post_any_sync", 0.0)
                            state._frame_stage_times["_post_any_sync"] += _t_elapsed
                        fused_boxes = fused_boxes[valid_w]
                        fused_scores = fused_scores[valid_w]
                        fused_classes = fused_classes[valid_w]
                        if geometry_suspect_mask.numel() > 0:
                            geometry_suspect_mask = geometry_suspect_mask[valid_w]

                if cfg.geometry.detection_quality_scaling and fused_boxes.numel() > 0:
                    quality_factors = _compute_detection_quality_batch(
                        fused_boxes,
                        w_orig,
                        h_orig,
                        w_aspect=cfg.geometry.detection_quality_w_aspect,
                        w_center=cfg.geometry.detection_quality_w_center,
                        w_area=cfg.geometry.detection_quality_w_area,
                    )
                    fused_scores = fused_scores * quality_factors
                elif (
                    cfg.geometry.geometry_suspect_support
                    and geometry_suspect_mask.numel() > 0
                ):
                    if state._frame_stage_times is not None:
                        _t_any = time.perf_counter()
                    _suspect_any = geometry_suspect_mask.any()
                    if state._frame_stage_times is not None:
                        _t_elapsed = (time.perf_counter() - _t_any) * 1000
                        state._frame_stage_times.setdefault("_post_any_sync", 0.0)
                        state._frame_stage_times["_post_any_sync"] += _t_elapsed
                    if _suspect_any:
                        fused_scores = fused_scores.clone()
                        fused_scores[geometry_suspect_mask] = torch.minimum(
                            fused_scores[geometry_suspect_mask],
                            torch.full_like(
                                fused_scores[geometry_suspect_mask],
                                cfg.geometry_suspect_support_score,
                            ),
                        )
                if profile_stages:
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t_sub_start) * 1000
                    seq_stage_totals["post_filter"] += elapsed_ms
                after_filter_count = int(fused_scores.numel())

            if debug_dump_active:
                _append_stage_dump_rows(
                    debug_stage_dump_rows,
                    seq=seq,
                    frame_id=frame_id,
                    stage="raw",
                    boxes=raw_dump_boxes,
                    scores=raw_dump_scores,
                    classes=raw_dump_classes,
                )
                _append_stage_dump_rows(
                    debug_stage_dump_rows,
                    seq=seq,
                    frame_id=frame_id,
                    stage="post_filter",
                    boxes=fused_boxes,
                    scores=fused_scores,
                    classes=fused_classes,
                )

            if fused_boxes.numel() == 0:
                _record_frame_timing(state, latency_started_at=t_frame_start)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (
                        time.perf_counter() - t_e2e_start
                    ) * 1000
                    state.seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                return True

            pre_private_boxes = None
            pre_private_scores = None
            pre_private_classes = None
            pre_private_geometry_suspect_mask = None
            pre_private_aligned_keypoints = None
            private_baseline_keep = None
            private_priors = None
            private_prior_classes = None
            private_motion_prior_boxes = None
            if (
                perception_pipeline is None
                and (is_tiled or cfg.detection.nms_iou_threshold is not None)
                and fused_boxes.numel() > 0
            ):
                if profile_stages:
                    torch.cuda.synchronize()
                    t_sub_start = time.perf_counter()

                # Fetch priors for Occlusion-aware NMS
                priors = None
                prior_classes = None
                if enable_onms:
                    priors, prior_classes = _build_active_track_priors(
                        detector.tracker,
                        fused_boxes.device,
                        min_track_age=onms_min_track_age,
                        min_track_score=onms_min_track_score,
                    )

                pre_private_boxes = fused_boxes
                pre_private_scores = fused_scores
                pre_private_classes = fused_classes
                pre_private_geometry_suspect_mask = geometry_suspect_mask
                pre_private_aligned_keypoints = aligned_keypoints
                keep = nms_fast(
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    cfg.detection.nms_iou_threshold,
                    class_aware=not cfg.detection.track_person_only,
                    priors=priors,
                    prior_classes=prior_classes,
                    prior_iou_threshold=onms_prior_iou_threshold,
                )
                private_baseline_keep = keep
                private_priors = priors
                private_prior_classes = prior_classes
                fused_boxes = fused_boxes[keep]
                fused_scores = fused_scores[keep]
                fused_classes = fused_classes[keep]
                geometry_suspect_mask = geometry_suspect_mask[keep]
                if aligned_keypoints is not None:
                    aligned_keypoints = aligned_keypoints[keep]
                if profile_stages:
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t_sub_start) * 1000
                    seq_stage_totals["post_nms"] += elapsed_ms
            if perception_pipeline is None:
                after_nms_count = int(fused_scores.numel())
            if debug_dump_active:
                _append_stage_dump_rows(
                    debug_stage_dump_rows,
                    seq=seq,
                    frame_id=frame_id,
                    stage="post_nms",
                    boxes=fused_boxes,
                    scores=fused_scores,
                    classes=fused_classes,
                )

            if _stage_probe_callback is not None:
                _stage_probe_callback(
                    seq,
                    frame_id,
                    "post_nms",
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                )

            if cfg.detection.tile_diagnostics and is_tiled:
                seq_tile_diag["frames_tiled"] += 1
                seq_tile_diag["pre_merge_seam_boxes"] += _count_tile_seam_boxes(
                    fused_boxes,
                    tiling=cfg.detection.tiling,
                    h_orig=h_orig,
                    w_orig=w_orig,
                )

            use_repo_cross_tile_merge = (
                cfg.detection.cross_tile_merge
                and is_tiled
                and cfg.detection.tiling != "sahi_960p_2x2"
                and fused_boxes.numel() > 1
            )
            if use_repo_cross_tile_merge:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_sub_start = time.perf_counter()
                fused_boxes, fused_scores, fused_classes, _merge_counts = (
                    merge_cross_tile_duplicates_fast(
                        fused_boxes,
                        fused_scores,
                        fused_classes,
                        tiling=cfg.detection.tiling,
                        frame_w=w_orig,
                        frame_h=h_orig,
                        seam_margin_canvas_px=cfg.detection.tile_seam_margin_canvas_px,
                        seam_center_scale=cfg.detection.cross_tile_seam_center_scale,
                        seam_area_ratio_threshold=cfg.detection.cross_tile_seam_area_ratio_threshold,
                        seam_min_overlap_ratio=cfg.detection.cross_tile_seam_min_overlap_ratio,
                    )
                )
                # MOT17-b: penalise boxes that were merged from multiple tiles.
                # Merged boxes have uncertain positions; lowering their score makes
                # ByteTracker treat them more conservatively during association.
                if cfg.detection.cross_tile_score_penalty < 1.0:
                    merged_mask = _merge_counts > 1
                    if merged_mask.any():
                        fused_scores = fused_scores.clone()
                        fused_scores[merged_mask] = (
                            fused_scores[merged_mask]
                            * cfg.detection.cross_tile_score_penalty
                        )
                if cfg.detection.tile_diagnostics:
                    merged_mask = _merge_counts > 1
                    seq_tile_diag["merged_clusters"] += int(merged_mask.sum().item())
                    seq_tile_diag["merged_members"] += int(
                        _merge_counts[merged_mask].sum().item()
                    )
                    seq_tile_diag["merged_outputs"] += int(_merge_counts.numel())
                geometry_suspect_mask = torch.zeros_like(fused_scores, dtype=torch.bool)
                aligned_keypoints = None
                if profile_stages:
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t_sub_start) * 1000
                    seq_stage_totals["post_merge"] += elapsed_ms
            after_merge_count = int(fused_scores.numel())
            crowd_low_active = (
                cfg.detection.crowd_low_score_mode
                and after_merge_count >= cfg.detection.crowd_low_score_trigger
            )
            frame_conf_threshold = (
                cfg.detection.crowd_conf_threshold
                if crowd_low_active
                else cfg.core.conf_threshold
            )
            frame_track_thresh = (
                cfg.geometry.crowd_track_thresh
                if crowd_low_active
                else cfg.core.track_thresh
            )
            frame_mid_thresh = (
                cfg.geometry.crowd_mid_thresh
                if crowd_low_active
                else cfg.core.mid_thresh
            )
            frame_new_track_thresh = (
                cfg.geometry.crowd_new_track_thresh
                if crowd_low_active
                else cfg.core.new_track_thresh
            )
            frame_score_floor = min(frame_conf_threshold, frame_track_thresh)
            base_score_floor = min(cfg.core.conf_threshold, cfg.core.track_thresh)
            (
                fused_boxes,
                fused_scores,
                fused_classes,
                geometry_suspect_mask,
                aligned_keypoints,
                after_merge_count,
            ) = _run_detection_filters(
                state,
                fused_boxes=fused_boxes,
                fused_scores=fused_scores,
                fused_classes=fused_classes,
                geometry_suspect_mask=geometry_suspect_mask,
                aligned_keypoints=aligned_keypoints,
                after_merge_count=after_merge_count,
                frame_score_floor=frame_score_floor,
                base_score_floor=base_score_floor,
                frame_id=frame_id,
                current_stage_sample_active=current_stage_sample_active,
                post_seg_events=post_seg_events,
                debug_dump_active=debug_dump_active,
                debug_stage_dump_rows=debug_stage_dump_rows,
            )
            if state._frame_stage_times is not None:
                state._frame_stage_times["_t_post_after_det_filters"] = (
                    time.perf_counter()
                )

            # === Stage 2 Quality Gate ===
            # Remove mid-score-band detections with poor geometry before the tracker's
            # Stage 2 association step, preventing bad lost-track assignments → IDs.
            _s2_quality = None
            _s2_pre_n = fused_boxes.shape[0]
            if cfg.stage2_quality_gate and fused_scores.numel() > 0:
                _s2_quality = _compute_detection_quality_batch(
                    fused_boxes,
                    w_orig,
                    h_orig,
                    w_aspect=cfg.geometry.detection_quality_w_aspect,
                    w_center=cfg.geometry.detection_quality_w_center,
                    w_area=cfg.geometry.detection_quality_w_area,
                )
                (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    geometry_suspect_mask,
                    aligned_keypoints,
                ) = _apply_stage2_quality_gate(
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    geometry_suspect_mask,
                    aligned_keypoints,
                    track_thresh=frame_track_thresh,
                    mid_thresh=frame_mid_thresh,
                    quality_min=cfg.stage2_quality_min,
                    quality=_s2_quality,
                )
                after_merge_count = int(fused_scores.numel())
            # Share quality factors with birth config when stage2 gate did not
            # remove any boxes (most-common fast path). When boxes change, let
            # birth_config recompute on its own.
            _birth_quality = (
                _s2_quality
                if _s2_quality is not None and _s2_pre_n == fused_boxes.shape[0]
                else None
            )

            frame_birth_events, fused_scores = _run_birth_config(
                state,
                frame_id=frame_id,
                frame_mid_thresh=frame_mid_thresh,
                frame_new_track_thresh=frame_new_track_thresh,
                frame_track_thresh=frame_track_thresh,
                fused_boxes=fused_boxes,
                fused_scores=fused_scores,
                fused_quality_factors=_birth_quality,
            )
            after_merge_count_before_private = after_merge_count
            after_private_count = after_merge_count
            if perception_pipeline is not None and private_added_count > 0:
                after_merge_count_before_private = max(
                    0, after_merge_count - private_added_count
                )
            if (
                cfg.detection.private_continuation_enabled
                and pre_private_boxes is not None
            ):
                if (
                    cfg.detection.private_prior_iou_threshold > 0.0
                    or cfg.detection.private_prior_center_threshold > 0.0
                    or (
                        cfg.detection.private_selection_mode
                        in {
                            "per_track",
                            "suppressor_aware",
                            "sparse_symmetric",
                            "energy",
                        }
                    )
                ):
                    private_motion_prior_boxes = _fctx.private_prior_boxes
                t_private_start = None
                if profile_stages:
                    torch.cuda.synchronize()
                    t_private_start = time.perf_counter()
                (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    geometry_suspect_mask,
                    aligned_keypoints,
                    private_added_count,
                ) = _append_private_continuation_candidates(
                    fused_boxes=fused_boxes,
                    fused_scores=fused_scores,
                    fused_classes=fused_classes,
                    geometry_suspect_mask=geometry_suspect_mask,
                    aligned_keypoints=aligned_keypoints,
                    pre_nms_boxes=pre_private_boxes,
                    pre_nms_scores=pre_private_scores,
                    pre_nms_classes=pre_private_classes,
                    pre_nms_geometry_suspect_mask=pre_private_geometry_suspect_mask,
                    pre_nms_aligned_keypoints=pre_private_aligned_keypoints,
                    baseline_keep=private_baseline_keep,
                    baseline_nms_iou=cfg.detection.nms_iou_threshold,
                    candidate_nms_iou=cfg.detection.private_candidate_nms_iou,
                    class_aware=not cfg.detection.track_person_only,
                    priors=private_priors,
                    prior_classes=private_prior_classes,
                    prior_iou_threshold=onms_prior_iou_threshold,
                    private_prior_boxes=private_motion_prior_boxes,
                    private_prior_iou_threshold=cfg.detection.private_prior_iou_threshold,
                    private_prior_center_threshold=cfg.detection.private_prior_center_threshold,
                    frame_track_thresh=frame_track_thresh,
                    frame_mid_thresh=frame_mid_thresh,
                    frame_new_track_thresh=frame_new_track_thresh,
                    low_stage_only=cfg.detection.private_low_stage_only,
                    private_min_score=cfg.detection.private_min_score,
                    private_max_candidates=cfg.detection.private_max_candidates,
                    private_selection_mode=cfg.detection.private_selection_mode,
                    private_energy_margin=cfg.detection.private_energy_margin,
                )
                after_merge_count = int(fused_scores.numel())
                after_private_count = after_merge_count
                if profile_stages and t_private_start is not None:
                    torch.cuda.synchronize()
                    seq_stage_totals["post_private_continuation"] += (
                        time.perf_counter() - t_private_start
                    ) * 1000
                if debug_dump_active:
                    _append_stage_dump_rows(
                        debug_stage_dump_rows,
                        seq=seq,
                        frame_id=frame_id,
                        stage="post_private_continuation",
                        boxes=fused_boxes,
                        scores=fused_scores,
                        classes=fused_classes,
                    )
            if cfg.detection.tile_diagnostics and is_tiled:
                seq_tile_diag["post_merge_seam_boxes"] += _count_tile_seam_boxes(
                    fused_boxes,
                    tiling=cfg.detection.tiling,
                    h_orig=h_orig,
                    w_orig=w_orig,
                    seam_margin_canvas_px=cfg.detection.tile_seam_margin_canvas_px,
                )
            if (
                cfg.detection.tile_seam_score_penalty < 1.0
                and is_tiled
                and fused_boxes.numel() > 0
            ):
                seam_mask = _tile_seam_mask(
                    fused_boxes,
                    tiling=cfg.detection.tiling,
                    h_orig=h_orig,
                    w_orig=w_orig,
                    seam_margin_canvas_px=cfg.detection.tile_seam_margin_canvas_px,
                )
                if seam_mask.any():
                    fused_scores = fused_scores.clone()
                    fused_scores[seam_mask] = (
                        fused_scores[seam_mask] * cfg.detection.tile_seam_score_penalty
                    )
            if state.frame_ledger is not None and frame_id > warmup_frames:
                state._frame_det_counts = {
                    "raw_boxes": raw_box_count,
                    "after_filter": after_filter_count,
                    "after_nms": after_nms_count,
                    "after_merge": after_merge_count_before_private,
                    "private_candidates": private_added_count,
                    "after_private": after_private_count,
                }
            if profile_stages:
                if post_gpu_end_event is not None:
                    post_gpu_end_event.record(torch.cuda.current_stream())
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t_post_start) * 1000
                seq_stage_totals["postprocess"] += elapsed_ms
                record_stage_sample("postprocess", elapsed_ms)
                if (
                    current_stage_sample_active
                    and post_gpu_start_event is not None
                    and post_gpu_end_event is not None
                ):
                    seq_stage_totals["post_gpu_elapsed"] += float(
                        post_gpu_start_event.elapsed_time(post_gpu_end_event)
                    )
                    # Partition the GPU span into contiguous segments. Each marker
                    # carries the name of the segment ending at it; the residual to
                    # post_gpu_end_event is the Python tail (gates 2814-3147).
                    _prev_ev = post_gpu_start_event
                    for _seg_name, _seg_ev in post_seg_events:
                        _seg_ms = float(_prev_ev.elapsed_time(_seg_ev))
                        seq_stage_totals[_seg_name] += _seg_ms
                        seq_segment_samples[_seg_name].append(_seg_ms)
                        _prev_ev = _seg_ev
                    _tail_ms = float(_prev_ev.elapsed_time(post_gpu_end_event))
                    seq_stage_totals["post_seg_python_tail"] += _tail_ms
                    seq_segment_samples["post_seg_python_tail"].append(_tail_ms)
                if frame_id > warmup_frames:
                    seq_post_counts["raw_boxes"] += raw_box_count
                    seq_post_counts["after_filter"] += after_filter_count
                    seq_post_counts["after_nms"] += after_nms_count
                    seq_post_counts["after_merge"] += after_merge_count_before_private
                    seq_post_counts["private_candidates"] += private_added_count
                    seq_post_counts["after_private"] += after_private_count
            _ledger_stage_done("post")
            if (
                state._frame_stage_times is not None
                and "_t_post_after_nms" in state._frame_stage_times
            ):
                _t_post_end = time.perf_counter()
                _t_after_nms = state._frame_stage_times.pop("_t_post_after_nms")
                _t_after_filter = state._frame_stage_times.pop(
                    "_t_post_after_filter", None
                )
                _t_after_det = state._frame_stage_times.pop(
                    "_t_post_after_det_filters", None
                )
                state._frame_stage_times["post_finalize"] = round(
                    (_t_post_end - _t_after_nms) * 1000, 6
                )
                if _t_after_filter is not None:
                    state._frame_stage_times["post_filter_d2h"] = round(
                        (_t_after_filter - _t_after_nms) * 1000, 6
                    )
                if _t_after_det is not None:
                    state._frame_stage_times["post_quality_filters"] = (
                        round((_t_after_det - _t_after_filter) * 1000, 6)
                        if _t_after_filter is not None
                        else 0
                    )
                    state._frame_stage_times["post_tail"] = round(
                        (_t_post_end - _t_after_det) * 1000, 6
                    )
            if _explicit_stream_probe_enabled() and state.stream_post is not None:
                _pp = frame_id % 2
                state._pp_post_done[_pp].record(state.stream_post)
                if (
                    _stream_mode_ptds_probe()
                    and state._pp_track_streams[_pp] is not None
                ):
                    _s_track = state._pp_track_streams[_pp]
                    torch.cuda.set_stream(_s_track)
                    _s_track.wait_event(state._pp_post_done[_pp])
                    if frame_id > 0:
                        _prev_p = 1 - _pp
                        _s_track.wait_event(state._pp_track_done[_prev_p])
                else:
                    # detect_post_event (production): restore the default stream
                    # so tracker/ReID/GMC/output — all of which assume the legacy
                    # (default) stream for ordering — are fenced after postprocess.
                    _default = torch.cuda.default_stream()
                    torch.cuda.set_stream(_default)
                    _default.wait_event(state._pp_post_done[_pp])
            # Sync previous frame's background relink_write before accessing shared
            # mutable state (dynamic_reid, primary_appearance_bank, relinker).
            if state.bg_future is not None:
                if profile_stages or (_t_ledger_last is not None):
                    t_bg_wait_start = time.perf_counter()
                (
                    _bg_rw_lines,
                    state.prev_track_ids,
                    _bg_det_idx_to_local_id,
                    _bg_output_by_local,
                ) = state.bg_future.result()
                if profile_stages:
                    elapsed_ms = (time.perf_counter() - t_bg_wait_start) * 1000
                    seq_stage_totals["bg_relink_wait"] += elapsed_ms
                    record_stage_sample("bg_relink_wait", elapsed_ms)
                elif _t_ledger_last is not None:
                    _ledger_stage_done("handover")
                results_lines.extend(_bg_rw_lines)
                if state.bg_birth_events is not None:
                    _annotate_birth_events(
                        state.bg_birth_events,
                        _det_idx_to_local_id=_bg_det_idx_to_local_id,
                        _output_by_local=_bg_output_by_local,
                    )
                state.bg_future = None
                state.bg_birth_events = None

            embeddings, mid_thresh_scale = _run_reid_and_gmc(
                state,
                frame_id=frame_id,
                fused_boxes=fused_boxes,
                fused_scores=fused_scores,
                fused_classes=fused_classes,
                after_merge_count=after_merge_count,
                current_stage_sample_active=current_stage_sample_active,
                _fpn_cache=_fpn_cache,
            )
            if _t_ledger_last is not None and state._frame_stage_times is not None:
                _end = state._frame_stage_times.pop("_stage_end", None)
                if isinstance(_end, (int, float)):
                    _t_ledger_last = float(_end)
                else:
                    _t_ledger_last = time.perf_counter()
                state._frame_stage_times.pop("_reid_enter", None)
                state._frame_stage_times.pop("_gmc_enter", None)
            appearance_occlusion_mask = getattr(
                state, "appearance_occlusion_mask", None
            )
            if (
                appearance_occlusion_mask is not None
                and appearance_occlusion_mask.shape == geometry_suspect_mask.shape
            ):
                geometry_suspect_mask = (
                    geometry_suspect_mask | appearance_occlusion_mask
                )

            if _stage_probe_callback is not None:
                _stage_probe_callback(
                    seq,
                    frame_id,
                    "tracker_input",
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                )

            state.tracker_result_buffers = _run_track(
                state,
                fused_boxes=fused_boxes,
                fused_scores=fused_scores,
                fused_classes=fused_classes,
                gmc_warp=state.gmc_warp,
                embeddings=embeddings,
                mid_thresh_scale=mid_thresh_scale,
                tracker_result_buffers=state.tracker_result_buffers,
                synchronize=state.double_buffer_stream is None,
            )
            if _stream_mode_ptds_probe() and state._pp_track_streams[0]:
                _pp = frame_id % 2
                state._pp_track_done[_pp].record(state._pp_track_streams[_pp])
                torch.cuda.set_stream(torch.cuda.default_stream())
            if (
                state.double_buffer_stream is not None
                and state.double_buffer_tracker_out_pinned
            ):
                parity = frame_id % 2
                pinned = state.double_buffer_tracker_out_pinned[parity]
                db_bufs = state.tracker_result_buffers
                for key in ("boxes", "scores", "ids", "classes", "det_idx", "count"):
                    pinned[key].copy_(db_bufs[key], non_blocking=True)
                _stash_crop_ring(
                    state,
                    {
                        "count": db_bufs["count"],
                        "ids": db_bufs["ids"],
                        "boxes": db_bufs["boxes"],
                    },
                    frame_id,
                )
                ev = state.double_buffer_tracker_out_events[parity]
                ev.record()
                state.double_buffer_tracker_out_fids[parity] = frame_id
                state.db_emit_frame_id = frame_id
                state.db_emit_event = ev
                state.db_emit_parity = parity
                state.db_emit_ctx = {
                    "tracker_result_buffers": db_bufs,
                    "fused_boxes": fused_boxes,
                    "fused_scores": fused_scores,
                    "fused_classes": fused_classes,
                    "geometry_suspect_mask": geometry_suspect_mask,
                    "embeddings": embeddings,
                    "gmc_warp": state.gmc_warp,
                }
                track_results = {
                    "count": 0,
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "ids": torch.empty((0,), dtype=torch.int32),
                    "classes": torch.empty((0,), dtype=torch.int32),
                    "det_idx": torch.empty((0,), dtype=torch.int32),
                }
            else:
                track_results = _run_materialize(
                    state,
                    tracker_result_buffers=state.tracker_result_buffers,
                    embeddings=embeddings,
                    aligned_keypoints=aligned_keypoints,
                    frame_id=frame_id,
                )

            _ledger_stage_done("track")

    if (
        aligned_keypoints is not None
        and track_results["det_idx"] is not None
        and track_results["count"] > 0
    ):
        det_idx = track_results["det_idx"]
        valid = (det_idx >= 0) & (det_idx < aligned_keypoints.shape[0])
        if valid.any():
            detector.tracker.push_keypoints(
                track_results["ids"][valid].to(
                    device=aligned_keypoints.device, dtype=torch.int32
                ),
                aligned_keypoints[det_idx[valid]],
            )

    if cfg.profile_lazy_reid_candidates:
        candidates = detector.tracker.get_tentative_candidates()
        ready_candidates = [
            c
            for c in candidates
            if c.hit_streak >= cfg.lazy_reid_min_hit_streak
            and c.hit_streak < c.required_confirm_streak
        ]
        state.seq_lazy_reid_candidates += len(ready_candidates)
        state.seq_lazy_reid_frames += 1
        if cfg.profile_lazy_reid_embeddings and extractor and cropper and candidates:
            ready_ids = {int(c.obj_id) for c in ready_candidates}

            def _profile_lazy_reid_embeddings() -> tuple[
                int, int, int, float, int, int, set[int]
            ]:
                embed_candidates = [
                    c
                    for c in candidates
                    if int(c.class_id) == cfg.detection.person_class
                    and c.hit_streak >= 1
                ]
                if not embed_candidates:
                    return 0, 0, 0, 0.0, 0, 0, set()
                cand_boxes = torch.tensor(
                    [[c.x1, c.y1, c.x2, c.y2] for c in embed_candidates],
                    device=pool.frame_buffer.device,
                    dtype=torch.float32,
                )
                crops = cropper.process(pool.as_rgb_chw().unsqueeze(0), cand_boxes)
                if crops.numel() == 0:
                    return 0, 0, 0, 0.0, 0, 0, set()
                cand_embeddings = extractor.extract(crops)
                pairs, passed, sim_sum = 0, 0, 0.0
                arbiter_checks, arbiter_approve = 0, 0
                seen_ids: set[int] = set()
                for cand, emb in zip(embed_candidates, cand_embeddings):
                    tid = int(cand.obj_id)
                    seen_ids.add(tid)
                    prev = lazy_reid_prev_embeddings.get(tid)
                    if prev is not None:
                        sim = float(torch.dot(prev, emb).item())
                        pairs += 1
                        sim_sum += sim
                        if sim >= cfg.lazy_reid_self_threshold:
                            passed += 1
                        if tid in ready_ids:
                            arbiter_checks += 1
                            if sim >= cfg.lazy_reid_self_threshold:
                                arbiter_approve += 1
                    lazy_reid_prev_embeddings[tid] = emb.detach()
                return (
                    len(embed_candidates),
                    pairs,
                    passed,
                    sim_sum,
                    arbiter_checks,
                    arbiter_approve,
                    seen_ids,
                )

            (
                (
                    crop_count,
                    pair_count,
                    pass_count,
                    sim_sum,
                    arbiter_checks,
                    arbiter_approve,
                    seen_ids,
                ),
                _,
            ) = time_stage(
                seq_stage_totals,
                "lazy_reid",
                _profile_lazy_reid_embeddings,
                sync_cuda=True,
            )
            state.seq_lazy_reid_crops += crop_count
            state.seq_lazy_reid_self_pairs += pair_count
            state.seq_lazy_reid_self_pass += pass_count
            state.seq_lazy_reid_self_sim_sum += sim_sum
            state.seq_lazy_reid_arbiter_checks += arbiter_checks
            state.seq_lazy_reid_arbiter_approve += arbiter_approve
            if seen_ids:
                for stale_id in set(lazy_reid_prev_embeddings.keys()) - seen_ids:
                    lazy_reid_prev_embeddings.pop(stale_id, None)

    if state.double_buffer_stream is None:
        (
            state.prev_track_ids,
            _emit_lines,
        ) = _run_emit(
            state,
            track_results=track_results,
            tracker_result_buffers=state.tracker_result_buffers,
            fused_boxes=fused_boxes,
            fused_scores=fused_scores,
            geometry_suspect_mask=geometry_suspect_mask,
            embeddings=embeddings,
            gmc_warp=state.gmc_warp,
            frame_birth_events=frame_birth_events,
            frame_id=frame_id,
            prev_track_ids=state.prev_track_ids,
            track_results_on_host=True,
        )
        results_lines.extend(_emit_lines)

    _ledger_stage_done("output")
    _record_frame_timing(state, latency_started_at=t_frame_start)
    if state.frame_ledger is not None and frame_id > warmup_frames:
        total_ms = (time.perf_counter() - t_frame_start) * 1000
        det = state._frame_det_counts or {}
        st = state._frame_stage_times or {}
        reid = state._frame_reid_stats or {}
        state.frame_ledger.add(
            seq=seq,
            frame=frame_id,
            total_ms=round(total_ms, 6),
            n_dets_raw=det.get("raw_boxes", 0),
            n_dets_after_filter=det.get("after_filter", 0),
            n_dets_after_nms=det.get("after_nms", 0),
            n_dets_final=det.get("after_private", track_results.get("count", 0)),
            fetch_ms=st.get("fetch", 0),
            detect_ms=st.get("detect", 0),
            detect_ingest_barrier_ms=st.get("detect_ingest_barrier", 0),
            detect_trt_enqueue_ms=st.get("detect_trt_enqueue", 0),
            detect_postproc_barrier_ms=st.get("detect_postproc_barrier", 0),
            post_ms=st.get("post", 0),
            post_graph_replay_ms=st.get("post_graph_replay", 0),
            post_graph_count_wait_ms=round(
                max(0.0, st.get("post", 0) - st.get("post_graph_replay", 0)), 6
            ),
            post_pre_nms_ms=st.get("post_pre_nms", 0),
            reid_ms=st.get("reid", 0),
            gmc_ms=st.get("gmc", 0),
            track_ms=st.get("track", 0),
            handover_ms=st.get("handover", 0),
            output_ms=st.get("output", 0),
            reid_submitted=1 if reid.get("submitted") else 0,
            reid_waited_this_frame=1 if reid.get("waited") else 0,
            reid_blocking_wait_ms=reid.get("blocking_wait_ms", 0),
            reid_crop_stash_ms=reid.get("crop_ms", 0),
            reid_extract_submit_ms=reid.get("extract_ms", 0),
            n_crops=reid.get("n_crops", 0),
            detect_stream_handle=int(st.get("detect_stream_handle", 0)),
            post_capture_stream_handle=int(st.get("post_capture_stream_handle", 0)),
            post_replay_stream_handle=int(st.get("post_replay_stream_handle", 0)),
            trt_enqueue_stream_handle=int(st.get("trt_enqueue_stream_handle", 0)),
            dets_hash="",
            explicit_stream_dispatch_ms=st.get("explicit_stream_dispatch_ms", 0),
            trt_enqueue_host_ms=st.get("trt_enqueue_host_ms", 0),
            event_record_ms=st.get("event_record_ms", 0),
            post_graph_launch_host_ms=st.get("post_graph_launch_host_ms", 0),
            post_tensor_prep_ms=st.get("post_tensor_prep", 0),
            post_nms_replay_ms=st.get("post_graph_replay", 0),
            post_finalize_ms=st.get("post_finalize", 0),
            post_any_sync_ms=round(st.pop("_post_any_sync", 0.0), 6),
            post_item_sync_ms=0,
            post_filter_d2h_ms=st.get("post_filter_d2h", 0),
            post_quality_filters_ms=st.get("post_quality_filters", 0),
            post_tail_ms=st.get("post_tail", 0),
        )
        if state.perception_pipeline is not None:
            try:
                cur = state.perception_pipeline.get_postprocess_profile_stats()
                prev = state._prev_post_sync_stats or {}
                state.frame_ledger._rows[-1]["post_filter_count_sync_ms"] = round(
                    float(cur.get("native_filter_count_sync_ms", 0))
                    - float(prev.get("native_filter_count_sync_ms", 0)),
                    6,
                )
                state.frame_ledger._rows[-1]["post_nms_count_sync_ms"] = round(
                    float(cur.get("native_nms_count_sync_ms", 0))
                    - float(prev.get("native_nms_count_sync_ms", 0)),
                    6,
                )
                state.frame_ledger._rows[-1]["post_final_count_sync_ms"] = round(
                    float(cur.get("count_d2h_ms", 0))
                    - float(prev.get("count_d2h_ms", 0)),
                    6,
                )
            except Exception:
                pass
        if state.frame_ledger is not None and state.frame_ledger._rows:
            import hashlib

            _row = state.frame_ledger._rows[-1]
            _counts = (
                str(int(float(_row.get("n_dets_raw", 0))))
                + str(int(float(_row.get("n_dets_final", 0))))
                + str(int(float(_row.get("n_tracks", 0))))
            )
            _row["dets_hash"] = hashlib.md5(_counts.encode()).hexdigest()[:8]
    if profile_stages and frame_id > warmup_frames:
        elapsed_ms = (time.perf_counter() - t_frame_start) * 1000
        seq_stage_totals["frame_total"] += elapsed_ms
        record_stage_sample("frame_total", elapsed_ms)
        if current_frame_stage_elapsed is not None:
            for (
                stage_name,
                stage_elapsed,
            ) in current_frame_stage_elapsed.items():
                seq_stage_samples[stage_name].append(stage_elapsed)
        state.seq_profiled_frames += 1
    if frame_id % 100 == 0:
        print(f"🎬 {seq} [{frame_id}/{frame_end}]")
    return True


def _mot_row_keys(lines: list[str]) -> set[tuple[int, float, float, float, float]]:
    """Detection-identity keys of MOT rows, independent of the track id.

    A post-process stage may relabel a row, add an interpolated one, or drop
    one; only the box itself survives all three, so it is what stage records
    are joined on.
    """
    keys: set[tuple[int, float, float, float, float]] = set()
    for line in lines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        keys.add(
            (
                int(parts[0]),
                round(float(parts[2]), 1),
                round(float(parts[3]), 1),
                round(float(parts[4]), 1),
                round(float(parts[5]), 1),
            )
        )
    return keys


def _postproc_stage_record(
    before: list[str],
    after: list[str],
    origin_keys: set[tuple[int, float, float, float, float]],
) -> dict[str, Any]:
    """What one output-layer stage actually did to the lines it was handed.

    Identity links are recovered rather than reported by the stage itself, so
    the record is comparable across stages that count their own work
    differently (a merge can span several breakpoints that a handover would
    count one at a time). Zero links is a legitimate outcome and is recorded
    as such.

    ``origin_keys`` are the rows that entered the first stage; rows in this
    stage's input outside that set were synthesised upstream and carry no
    detection, which is the one way a chained stage sees input a single-stage
    run never produces.
    """

    def _by_key(lines: list[str]) -> dict[tuple[int, float, float, float, float], int]:
        out: dict[tuple[int, float, float, float, float], int] = {}
        for line in lines:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            out[
                (
                    int(parts[0]),
                    round(float(parts[2]), 1),
                    round(float(parts[3]), 1),
                    round(float(parts[4]), 1),
                    round(float(parts[5]), 1),
                )
            ] = int(parts[1])
        return out

    src, dst = _by_key(before), _by_key(after)
    grouped: dict[int, set[int]] = {}
    for key, src_id in src.items():
        dst_id = dst.get(key)
        if dst_id is not None:
            grouped.setdefault(dst_id, set()).add(src_id)

    parent: dict[int, int] = {}

    def _find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in grouped.values():
        ordered = sorted(members)
        for other in ordered[1:]:
            ra, rb = _find(ordered[0]), _find(other)
            if ra != rb:
                parent[ra] = rb

    comps: dict[int, set[int]] = {}
    for src_id in {v for v in src.values()}:
        comps.setdefault(_find(src_id), set()).add(src_id)
    linked = [m for m in comps.values() if len(m) > 1]

    return {
        "rows_in": len(before),
        "rows_out": len(after),
        "rows_added": len(dst.keys() - src.keys()),
        "rows_dropped": len(src.keys() - dst.keys()),
        "rows_in_synthetic": len(src.keys() - origin_keys),
        "ids_in": len({v for v in src.values()}),
        "ids_out": len({v for v in dst.values()}),
        "link_pairs": sum(len(m) * (len(m) - 1) // 2 for m in linked),
        "link_components": len(linked),
        "largest_component": max((len(m) for m in linked), default=0),
    }


def run_eval(
    engine: str,
    output: str,
    data_root: str,
    split: str,
    sequences: str,
    max_frames: int,
    conf_threshold: float,
    reid_mode: str = "semantic",
    reid_model: str = "siglip2",
    detector: TRTYoloDetector = None,
    extractor: TRTFeatureExtractor = None,
    pose_engine: str = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    from .config import parse_eval_config

    cfg = parse_eval_config(
        output=output,
        data_root=data_root,
        split=split,
        sequences=sequences,
        conf_threshold=conf_threshold,
        reid_mode=reid_mode,
        reid_model=reid_model,
        profile_stages=bool(kwargs.get("profile_stages", False)),
        profile_frame_csv=bool(kwargs.get("profile_frame_csv", False)),
        kwargs=kwargs,
    )
    if cfg.detection.private_continuation_enabled:
        if cfg.core.workbench:
            raise ValueError(
                "private continuation is not implemented for the Workbench "
                "hot path; disable --workbench"
            )
        if cfg.detection.private_candidate_nms_iou < cfg.detection.nms_iou_threshold:
            raise ValueError("private-candidate-nms-iou must be >= nms-iou-threshold")

    output_root = cfg.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cheb_gr_online_log_path = output_root / "_cheb_gr_offline_handover.csv"
    if getattr(cfg, "cheb_gr_online_log", False):
        cheb_gr_online_log_path.unlink(missing_ok=True)
    occ_audit_log_path = output_root / "_occ_audit.csv"
    if getattr(cfg, "occ_audit_log", False):
        occ_audit_log_path.unlink(missing_ok=True)
    if cfg.geometry.assoc_energy_diagnostics:
        scoring_profile = {
            "association_scoring_mode": cfg.geometry.association_scoring_mode,
            "multiplicative_cost": bool(
                cfg.geometry.multiplicative_cost
                or cfg.geometry.association_scoring_mode == "energy"
            ),
            "sinkhorn_lambda": float(cfg.geometry.sinkhorn_lambda),
            "stability_cost_w": float(cfg.geometry.stability_cost_w),
            "assoc_score_cost_w": float(cfg.geometry.assoc_score_cost_w),
            "assoc_height_cost_w": float(cfg.geometry.assoc_height_cost_w),
            "private_continuation_enabled": bool(
                cfg.detection.private_continuation_enabled
            ),
            "private_selection_mode": cfg.detection.private_selection_mode,
            "private_energy_margin": float(cfg.detection.private_energy_margin),
        }
        (output_root / "_association_scoring_profile.json").write_text(
            json.dumps(scoring_profile, indent=2) + "\n"
        )
    fps_summary_lines = []
    overall_latency_ms = []
    overall_throughput_frames = 0
    overall_throughput_seconds = 0.0
    debug_dump_seq = cfg.core.debug_dump_seq
    debug_dump_frames = _parse_debug_frame_ranges(cfg.core.debug_dump_frames)
    debug_dump_csv = cfg.core.debug_dump_csv
    debug_stage_dump_rows: list[dict[str, float | int | str]] = []
    debug_birth_csv = cfg.core.debug_birth_csv
    debug_birth_rows: list[dict[str, float | int | str | bool]] = []
    profile_stages = cfg.core.profile_stages
    profile_frame_csv = cfg.core.profile_frame_csv
    if profile_frame_csv:
        from .frame_ledger import FrameLedger

        frame_ledger = FrameLedger()
    else:
        frame_ledger = None

    if os.environ.get("SACCADE_STREAM_MODE", "") == "ptds_probe":
        cfg.kwargs["use_tracker_graph"] = True
        print("[STREAM] ptds_probe: forcing use_tracker_graph=True")

    if os.environ.get("SACCADE_STREAM_DEBUG", "") in ("1", "true", "yes"):
        import os as _os

        _cs = torch.cuda.current_stream()
        _ds = torch.cuda.default_stream()
        _legacy = torch.cuda.Stream(0) if hasattr(torch.cuda, "Stream") else None
        print(
            f"[STREAM] run_eval: current={_cs.cuda_stream:#x} default={_ds.cuda_stream:#x}"
        )
        print(
            f"[STREAM] barrier={_detect_barrier_mode()} workbench={cfg.core.workbench} whole_graph={getattr(detector, 'use_whole_graph', False)} double_buffer={_double_buffer_eligible(cfg, detector, profile_stages)}"
        )
        print(
            f"[STREAM] async_reid={cfg.async_reid} gmc_mode={cfg.core.gmc_mode} decode={'NVJPEG' if _os.environ.get('SACCADE_GPU_DECODE') == '1' else 'DALI'}"
        )
    detector_box_format = cfg.detection.detector_box_format
    stage_summary_lines = []
    global_id_mapper = GlobalTrackIdMapper()
    external_fp_rule_config = RuleBaselineConfig()
    external_fp_logistic_model = None
    if cfg.detection.external_fp_filter_mode in {"logistic", "softmax3"}:
        model_path = Path(cfg.detection.external_fp_logistic_model)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"external FP logistic model not found: {model_path}"
            )
        external_fp_logistic_model = load_logistic_model(model_path)

    if not isinstance(
        detector,
        (
            TRTYoloDetector,
            TwostageDetector,
            ConcurrentDetectorProxy,
            BatchedDetectorProxy,
        ),
    ):
        from saccade.perception.temporal_yolo.mamba_gated_detector import (
            MambaGatedDetector,
        )
        from saccade.perception.multistream_mamba_server import (
            MambaStreamProxy,
        )

        if isinstance(detector, (MambaGatedDetector, MambaStreamProxy)) or (
            detector is not None and hasattr(detector, "detect_raw")
        ):
            # Pre-built detector (Mamba head, or a TeacherHeadDetector-style
            # drop-in exposing detect_raw) — use as-is, do not construct a TRT
            # engine from the placeholder engine string.
            pass
        else:
            import os as _os

            tiling = kwargs.get("tiling", "native_960")
            if tiling in {"960p_2x2", "sahi_960p_2x2"} and "_960_batch1" in engine:
                candidate = engine.replace("_960_batch1", "_batch4")
                if _os.path.exists(candidate):
                    engine = candidate
            elif tiling == "960p_3x2" and "_960_batch1" in engine:
                candidate = engine.replace("_960_batch1", "_batch6")
                if _os.path.exists(candidate):
                    engine = candidate
            elif tiling == "native_640" and "_960_batch1" in engine:
                candidate = engine.replace("_960_batch1", "_batch4")
                if _os.path.exists(candidate):
                    engine = candidate
            if pose_engine:
                detector = TwostageDetector(det_engine=engine, pose_engine=pose_engine)
            else:
                detector = TRTYoloDetector(engine_path=engine)

    if reid_mode not in {"off", "tracker", "semantic", "hybrid", "extract"}:
        raise ValueError(f"Unsupported reid_mode: {reid_mode}")
    if bool(kwargs.get("semantic_cheb_gr_claim", False)) and reid_mode == "off":
        raise ValueError("--semantic-cheb-gr-claim requires a non-off --reid-mode")

    _fpn_reid_mode = reid_model in {"fpn_raw", "fpn_trained"}
    _fpn_reid_conv_weights = None
    _fpn_reid_proj_weight = _fpn_reid_running_mean = _fpn_reid_running_var = None
    _fpn_reid_dim = 0
    _fpn_backbone = None
    _fpn_cache: dict[str, torch.Tensor] = {}
    _fpn_img_size = 640

    if _fpn_reid_mode:
        fpn_reid_ckpt = kwargs.get("fpn_reid_ckpt", "")
        if reid_model == "fpn_trained" and fpn_reid_ckpt:
            _fpn_reid_dim = 128
            ckpt_path = Path(fpn_reid_ckpt)
            if not ckpt_path.is_absolute():
                ckpt_path = Path.cwd() / ckpt_path
            _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            in_channels = _ckpt.get("in_channels", [128, 256, 512])
            _fpn_reid_conv_weights = [
                _ckpt["head"][f"convs.{i}.weight"].to(device="cuda")
                for i in range(len(in_channels))
            ]
            mid_dim = 128 * len(in_channels)
            if mid_dim != 128:
                _fpn_reid_proj_weight = _ckpt["head"]["proj.0.weight"].to(device="cuda")
                _fpn_reid_running_mean = _ckpt["head"]["proj.1.running_mean"].to(
                    device="cuda"
                )
                _fpn_reid_running_var = _ckpt["head"]["proj.1.running_var"].to(
                    device="cuda"
                )
        else:
            _fpn_reid_dim = 896
        extractor = None
        cropper = None
        _fpn_backbone = None
        _fpn_cache: dict[str, torch.Tensor] = {}
        if not hasattr(detector, "teacher"):
            _bb_engine = kwargs.get("fpn_backbone_engine", "")
            if _bb_engine:
                from saccade.perception.temporal_yolo.mamba_gated_detector import (
                    TRTYoloBackbone as _TRTYoloBackbone,
                )

                _fpn_backbone = _TRTYoloBackbone(str(Path(_bb_engine).resolve()))
                print(f"  FPN backbone engine: {_bb_engine}")
                _fpn_img_size = 960 if "960" in _bb_engine else 640

    elif extractor is None and cfg.reid_work_enabled:
        extractor = TRTFeatureExtractor(
            engine_path=cfg.reid_engine,
            model_type=reid_model,
            max_batch=64,
        )
        cropper = (
            ZeroCopyCropper(
                output_size=cfg.crop_hw,
                mode=cfg.reid_crop_mode,
                padding=cfg.reid_crop_padding,
            )
            if cfg.reid_work_enabled
            else None
        )
    else:
        cropper = (
            ZeroCopyCropper(
                output_size=cfg.crop_hw,
                mode=cfg.reid_crop_mode,
                padding=cfg.reid_crop_padding,
            )
            if cfg.reid_work_enabled
            else None
        )

    if cfg.reid_crop_layout not in {"full", "parts"}:
        raise ValueError(f"Unsupported reid_crop_layout: {cfg.reid_crop_layout}")

    if cfg.detection.tiling == "mamba_global_2x2":
        detect_fn = detect_mamba_global_2x2
    elif cfg.detection.tiling == "960p_3x2":
        detect_fn = detect_960p_3x2_tiled
    elif cfg.detection.tiling == "sahi_960p_2x2":
        detect_fn = detect_sahi_960p_2x2
    elif cfg.detection.tiling == "native_640":
        detect_fn = detect_native_640
    elif cfg.detection.tiling in (
        "native_960",
        "mamba_960",
        "native_1024",
        "native_1280",
    ):
        detect_fn = (
            detect_native_960_tta if getattr(cfg, "tta", False) else detect_native_960
        )
    else:
        detect_fn = detect_adaptive_960_tiled

    contract = DetectionContract(
        feature_dim=_fpn_reid_dim,
        fpn_reid_mode=_fpn_reid_mode,
    )
    fpn = FPNConfig(
        backbone=_fpn_backbone,
        img_size=_fpn_img_size,
        conv_weights=_fpn_reid_conv_weights,
        proj_weight=_fpn_reid_proj_weight,
        running_mean=_fpn_reid_running_mean,
        running_var=_fpn_reid_running_var,
    )

    extractor_cpp_ptr = _safe_cpp_ptr(extractor) if extractor is not None else 0
    cropper_cpp_ptr = _safe_cpp_ptr(cropper) if cropper is not None else 0
    native_postprocess_available = (
        PerceptionPipeline is not None and PerceptionPipelineConfig is not None
    )
    native_private_mode = str(cfg.detection.private_selection_mode).strip().lower()
    native_private_blockers: list[str] = []
    if cfg.detection.private_continuation_enabled:
        if native_private_mode != "global":
            native_private_blockers.append(f"selection_mode={native_private_mode}")
        for _flag_name in (
            "crowd_low_score_mode",
            "duplicate_suppression",
            "stage2_quality_gate",
            "birth_consecutive_gate",
            "birth_quality_gate",
            "multi_birth_enabled",
        ):
            if bool(getattr(cfg, _flag_name, False)):
                native_private_blockers.append(_flag_name)
    native_private_available = bool(
        cfg.detection.private_continuation_enabled and not native_private_blockers
    )
    if (
        cfg.detection.private_continuation_enabled
        and native_postprocess_available
        and not native_private_available
    ):
        native_postprocess_available = False
        print(
            "  [private_continuation] disabled native postprocess for "
            + ", ".join(native_private_blockers)
        )
    native_reid_available = (
        native_postprocess_available
        and extractor_cpp_ptr != 0
        and cropper_cpp_ptr != 0
        and cfg.reid_crop_layout == "full"
    )
    native_cfg = None
    perception_pipeline = None
    if native_postprocess_available:
        native_cfg = PerceptionPipelineConfig()
        native_cfg.score_threshold = min(
            conf_threshold,
            cfg.core.track_thresh,
            cfg.detection.crowd_conf_threshold
            if cfg.detection.crowd_low_score_mode
            else conf_threshold,
            cfg.geometry.crowd_track_thresh
            if cfg.detection.crowd_low_score_mode
            else cfg.core.track_thresh,
        )
        native_cfg.person_class = cfg.detection.person_class
        native_cfg.person_only = cfg.detection.track_person_only
        native_cfg.nms_threshold = cfg.detection.nms_iou_threshold
        native_cfg.person_geometry_prior = cfg.geometry.person_geometry_prior
        native_cfg.geometry_suspect_support = cfg.geometry.geometry_suspect_support
        native_cfg.geometry_suspect_support_score = cfg.geometry_suspect_support_score
        native_cfg.person_min_height_ratio = cfg.geometry.person_min_height_ratio
        native_cfg.person_min_aspect = cfg.geometry.person_min_aspect
        native_cfg.person_max_aspect = cfg.geometry.person_max_aspect
        native_cfg.person_min_area_ratio = cfg.geometry.person_min_area_ratio
        native_cfg.person_max_area_ratio = cfg.geometry.person_max_area_ratio
        native_cfg.max_detections = 2048
        native_cfg.private_continuation_enabled = native_private_available
        native_cfg.private_candidate_nms_iou = cfg.detection.private_candidate_nms_iou
        native_cfg.private_min_score = cfg.detection.private_min_score
        native_cfg.private_max_candidates = cfg.detection.private_max_candidates
        native_cfg.private_prior_iou_threshold = (
            cfg.detection.private_prior_iou_threshold
        )
        native_cfg.private_prior_center_threshold = (
            cfg.detection.private_prior_center_threshold
        )
        native_cfg.private_low_stage_only = cfg.detection.private_low_stage_only
        native_cfg.private_track_thresh = cfg.core.track_thresh
        native_cfg.private_mid_thresh = cfg.core.mid_thresh
        native_cfg.private_new_track_thresh = cfg.core.new_track_thresh
        native_cfg.private_score_eps = 1e-4
        perception_pipeline = PerceptionPipeline(
            extractor_cpp_ptr if native_reid_available else 0,
            cropper_cpp_ptr if native_reid_available else 0,
            native_cfg,
        )
        perception_pipeline.set_postprocess_profiling_enabled(profile_stages)
        if native_reid_available:
            perception_pipeline.set_reid_profiling_enabled(profile_stages)
    enable_onms = _env_flag_enabled("SACCADE_ENABLE_ONMS", False)
    onms_prior_iou_threshold = 0.70
    onms_min_track_age = 2
    onms_min_track_score = cfg.core.high_thresh

    top_level_stage_names = (
        "fetch",
        "ingest_preprocess",
        "detect",
        "postprocess",
        "reid_bank_sync",
        "reid_budget",
        "reid_crop",
        "reid_extract",
        "lazy_reid",
        "gmc",
        "track",
        "materialize",
        "bg_relink_wait",
        "relink_write",
        "frame_total",
    )
    breakdown_stage_names = (
        "post_gpu_elapsed",
        "post_tensor_prep",
        "post_filter",
        "post_nms",
        "post_private_continuation",
        "post_count_sync",
        "post_native_other",
        "native_filter_gather",
        "native_filter_kernel",
        "native_gather_compact3",
        "native_copy_suspect",
        "native_filter_count_sync",
        "native_small_nms",
        "native_suspect_penalty",
        "native_large_sort_nms",
        "native_large_argsort",
        "native_large_nms",
        "native_compact_copy",
        "native_large_gather4",
        "native_large_copyback",
        "native_private_candidate_nms",
        "native_private_append",
        "post_keypoint_align",
        "post_output_slicing",
        "post_quality_scale",
        "post_tail_filtering",
        "post_merge",
        # Sync-free CUDA-event span partition (diagnostic; excluded from attribution)
        "post_seg_prep",
        "post_seg_native",
        "post_seg_slice_quality",
        "post_seg_tail_filter",
        "post_seg_fp_hard",
        "post_seg_python_tail",
    )

    native_reid_breakdown_names = (
        "native_reid_crop",
        "native_reid_pre_normalize",
        "native_reid_trt_enqueue",
        "native_reid_l2_normalize",
    )
    gmc_breakdown_names = (
        "gmc_gray_downscale",
        "gmc_fg_mask",
        "gmc_phase_corr",
        "gmc_handoff",
    )
    # Sync-free CUDA-event partition of the postprocess GPU span; sampled per
    # frame so the report can show P95/P99 (not just the mean) per segment.
    segment_breakdown_names = tuple(
        name for name in breakdown_stage_names if name.startswith("post_seg_")
    )
    current_frame_stage_elapsed: dict[str, float] | None = None
    current_stage_sample_active = False

    def record_stage_sample(stage_name: str, elapsed_ms: float) -> None:
        if (
            profile_stages
            and current_stage_sample_active
            and current_frame_stage_elapsed is not None
            and stage_name in current_frame_stage_elapsed
        ):
            current_frame_stage_elapsed[stage_name] += elapsed_ms

    def time_stage(stage_totals, stage_name, fn, sync_cuda=False):
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if profile_stages:
            stage_totals[stage_name] += elapsed_ms
            record_stage_sample(stage_name, elapsed_ms)
        return result, elapsed_ms

    overall_stage_totals = OrderedDict(
        (name, 0.0) for name in (*top_level_stage_names, *breakdown_stage_names)
    )
    overall_stage_samples = OrderedDict((name, []) for name in top_level_stage_names)
    overall_gmc_samples = OrderedDict((name, []) for name in gmc_breakdown_names)
    overall_segment_samples: "OrderedDict[str, list[float]]" = OrderedDict(
        (name, []) for name in segment_breakdown_names
    )
    overall_profiled_frames = 0
    overall_post_counts = OrderedDict(
        (name, 0)
        for name in (
            "raw_boxes",
            "after_filter",
            "after_nms",
            "after_merge",
            "private_candidates",
            "after_private",
        )
    )
    overall_lazy_reid_candidates = 0
    overall_lazy_reid_frames = 0
    overall_lazy_reid_crops = 0
    overall_lazy_reid_self_pairs = 0
    overall_lazy_reid_self_pass = 0
    overall_lazy_reid_self_sim_sum = 0.0
    overall_lazy_reid_arbiter_checks = 0
    overall_lazy_reid_arbiter_approve = 0

    reid_side_stream: torch.cuda.Stream | None = (
        torch.cuda.Stream() if cfg.async_reid and torch.cuda.is_available() else None
    )
    reid_side_event: torch.cuda.Event | None = (
        torch.cuda.Event(enable_timing=False) if reid_side_stream is not None else None
    )
    reid_main_ready: torch.cuda.Event | None = (
        torch.cuda.Event(enable_timing=False) if reid_side_stream is not None else None
    )

    _rw_executor: ThreadPoolExecutor | None = None

    all_seq_profile: list[dict] = []

    # Cheb-GR offline tracklet merge (path 2) / output-layer handover:
    # ReID extractor built once.
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

    for seq in cfg.seqs:
        _seq_path = Path(cfg.core.data_root) / cfg.core.split / seq
        if not (_seq_path / "seqinfo.ini").exists():
            continue
        _seq_state = EvalPipeline(
            cfg=cfg,
            seq=seq,
            profile_stages=profile_stages,
            contract=contract,
            detector=detector,
            cropper=cropper,
            extractor=extractor,
            fpn=fpn,
            debug_birth_rows=debug_birth_rows,
            global_id_mapper=global_id_mapper,
            gmc_breakdown_names=gmc_breakdown_names,
            max_frames=max_frames,
            native_cfg=native_cfg,
            native_reid_breakdown_names=native_reid_breakdown_names,
            overall_stage_totals=overall_stage_totals,
            segment_breakdown_names=segment_breakdown_names,
            top_level_stage_names=top_level_stage_names,
            time_stage=time_stage,
            record_stage_sample=record_stage_sample,
            _rw_executor=_rw_executor,
            perception_pipeline=perception_pipeline,
            reid_main_ready=reid_main_ready,
            reid_side_event=reid_side_event,
            reid_side_stream=reid_side_stream,
            debug_dump_frames=debug_dump_frames,
            debug_dump_seq=debug_dump_seq,
            debug_stage_dump_rows=debug_stage_dump_rows,
            detect_fn=detect_fn,
            detector_box_format=detector_box_format,
            enable_onms=enable_onms,
            onms_min_track_age=onms_min_track_age,
            onms_min_track_score=onms_min_track_score,
            onms_prior_iou_threshold=onms_prior_iou_threshold,
            native_reid_available=native_reid_available,
            external_fp_rule_config=external_fp_rule_config,
            external_fp_logistic_model=external_fp_logistic_model,
            frame_ledger=frame_ledger,
        )
        _seq_state.stage_probe_callback = kwargs.get("stage_probe_callback")

        from .pipeline import _explicit_stream_probe_enabled, _stream_mode_ptds_probe

        if _explicit_stream_probe_enabled():
            _s_detect_handles = []
            _s_post_handles = []
            _s_track_handles = []
            _ptds = _stream_mode_ptds_probe()
            for _p in (0, 1):
                _pp = _seq_state._pp_streams[_p]
                _pp["detect"] = torch.cuda.Stream()
                _pp["post"] = torch.cuda.Stream()
                _s_detect_handles.append(_pp["detect"].cuda_stream)
                _s_post_handles.append(_pp["post"].cuda_stream)
                _seq_state._pp_detect_done[_p] = torch.cuda.Event()
                _seq_state._pp_post_done[_p] = torch.cuda.Event()
                if _ptds:
                    _seq_state._pp_track_streams[_p] = torch.cuda.Stream()
                    _seq_state._pp_track_done[_p] = torch.cuda.Event()
                    _s_track_handles.append(
                        _seq_state._pp_track_streams[_p].cuda_stream
                    )
            _seq_state.stream_detect = _seq_state._pp_streams[0]["detect"]
            _seq_state.stream_post = _seq_state._pp_streams[0]["post"]
            _seq_state.stream_detect_event = _seq_state._pp_detect_done[0]
            _track_info = (
                f"S_track=({_s_track_handles[0]:#x},{_s_track_handles[1]:#x})"
                if _ptds
                else "track=0x0 (legacy)"
            )
            print(
                f"[STREAM] {('ptds_probe (experimental)' if _ptds else 'detect_post_event (production)'):>35s}: "
                f"S_detect=({_s_detect_handles[0]:#x},{_s_detect_handles[1]:#x}) "
                f"S_post=({_s_post_handles[0]:#x},{_s_post_handles[1]:#x}) "
                f"{_track_info}"
            )
            if _ptds:
                forced_graph = cfg.kwargs.setdefault("use_tracker_graph", True)
                if not forced_graph:
                    cfg.kwargs["use_tracker_graph"] = True
                    print("[STREAM] ptds_probe: forced use_tracker_graph=True")

        def _run_frame_diagnostics(frame_id: int) -> None:
            import os as _os  # noqa: E402

            if _os.environ.get("SACCADE_OCC_LOG", "") or _os.environ.get(
                "SACCADE_OCC_DUMP", ""
            ):
                _trk = detector.tracker
                _buf = _seq_state.tracker_result_buffers
                if _buf is not None:
                    _snaps = _trk.get_state_snapshots()
                    _ids = [s.obj_id for s in _snaps]
                    _sts = [v for s in _snaps for v in s.state]
                    _ndet = _buf["count"].item()
                    if hasattr(_trk, "_occ_log_maybe"):
                        _trk._occ_log_maybe(frame_id, _sts, _ids, _ndet, seq)
                    if hasattr(_trk, "_occ_dump_maybe"):
                        _trk._occ_dump_maybe(frame_id, _sts, _ids, seq)

        if _seq_state.double_buffer_stream is None:
            for frame_id in range(1, _seq_state.frame_end + 1):
                if not _run_frame(_seq_state, frame_id=frame_id):
                    break
                _run_frame_diagnostics(frame_id)
        else:
            # Prime frame 1, then always enqueue detect(N+1) before tracking N.
            # There is one detector task in flight and two frame pools selected
            # by parity, so the tracker sees exactly the serial frame order.
            print("  [double-buffer] detect(N+1) overlaps tracker(N) on a side stream")

            def _schedule(frame_id: int) -> PreparedDetection | None:
                # Start the end-to-end latency clock before decode/ingest.
                latency_started_at = time.perf_counter()
                try:
                    frame_gpu = next(_seq_state.stream_iter)
                except StopIteration:
                    return None
                pool = _seq_state.double_buffer_pools[(frame_id - 1) % 2]
                input_ready, ready_event = _seq_state.double_buffer_events[
                    (frame_id - 1) % 2
                ]
                return _launch_double_buffer_detect(
                    _seq_state,
                    frame_id=frame_id,
                    pool=pool,
                    frame_gpu=frame_gpu,
                    input_ready=input_ready,
                    ready_event=ready_event,
                    latency_started_at=latency_started_at,
                )

            pending = _schedule(1)
            for frame_id in range(1, _seq_state.frame_end + 1):
                if pending is None:
                    break
                next_pending = (
                    _schedule(frame_id + 1) if frame_id < _seq_state.frame_end else None
                )
                if not _run_frame(
                    _seq_state,
                    frame_id=frame_id,
                    prepared_detection=pending,
                ):
                    break
                _run_frame_diagnostics(frame_id)
                pending = next_pending

        # Flush deferred materialize from the last frame.
        if _seq_state.defer_emit and _seq_state.defer_emit_event is not None:
            _lines, _ = _flush_deferred_emit(
                _seq_state.defer_emit_event,
                _seq_state.pinned_result_bufs,
                default_class_id=cfg.detection.person_class
                if cfg.detection.track_person_only
                else None,
                global_id_mapper=global_id_mapper,
                seq=seq,
                frame_id=_seq_state.defer_emit_fid,
                frame_w=_seq_state.w_orig,
                frame_h=_seq_state.h_orig,
            )
            _seq_state.results_lines.extend(_lines)
            _seq_state.defer_emit_event = None

        # Flush deferred double-buffer tracker output (last frame).
        if (
            _seq_state.double_buffer_stream is not None
            and _seq_state.db_emit_frame_id > 0
            and _seq_state.db_emit_event is not None
        ):
            _flush_db_tracker_out(_seq_state)

        # Flush any last background relink_write future before post-processing results.
        if _seq_state.bg_future is not None:
            (
                _bg_rw_lines,
                _seq_state.prev_track_ids,
                _bg_det_idx_to_local_id,
                _bg_output_by_local,
            ) = _seq_state.bg_future.result()
            _seq_state.results_lines.extend(_bg_rw_lines)
            if _seq_state.bg_birth_events is not None:
                _seq_state.annotate_birth_events(
                    _seq_state.bg_birth_events,
                    _det_idx_to_local_id=_bg_det_idx_to_local_id,
                    _output_by_local=_bg_output_by_local,
                )
            _seq_state.bg_future = None
            _seq_state.bg_birth_events = None

        if _seq_state.frame_ledger is not None:
            csv_path = output_root / f"_frame_ledger_{seq}.csv"
            _seq_state.frame_ledger.write_csv(csv_path)
            print(
                f"📋 Frame ledger written: {csv_path} ({len(_seq_state.frame_ledger)} frames)"
            )

        if _seq_state.frame_latencies:
            lats = np.array(_seq_state.frame_latencies)
            mean_ms = float(np.mean(lats))
            p95_ms = float(np.percentile(lats, 95))
            p99_ms = float(np.percentile(lats, 99))
            throughput_seconds = max(
                0.0,
                (_seq_state.throughput_finished_at or 0.0)
                - (_seq_state.throughput_started_at or 0.0),
            )
            throughput_fps = (
                _seq_state.throughput_frames / throughput_seconds
                if throughput_seconds > 0.0
                else 0.0
            )
            print(f"\n📊 Production Latency Report for {seq}:")
            print(f"  - Mean latency: {mean_ms:.2f} ms")
            print(f"  - P95: {p95_ms:.2f} ms")
            print(f"  - P99: {p99_ms:.2f} ms")
            print(f"  - Throughput: {throughput_fps:.2f} FPS")
            fps_summary_lines.append(
                f"{seq}\tfps={throughput_fps:.2f}\tmean_ms={mean_ms:.2f}"
                f"\tframes={_seq_state.throughput_frames}"
            )
            overall_latency_ms.extend(_seq_state.frame_latencies)
            overall_throughput_frames += _seq_state.throughput_frames
            overall_throughput_seconds += throughput_seconds
            latency_profile = {
                "sequence": seq,
                "frames": len(_seq_state.frame_latencies),
                "throughput_fps": round(throughput_fps, 4),
                "throughput_seconds": round(throughput_seconds, 6),
                "mean_ms": round(mean_ms, 6),
                "std_ms": round(float(np.std(lats)), 6),
                "p95_ms": round(p95_ms, 6),
                "p99_ms": round(p99_ms, 6),
                "samples_ms": [round(float(x), 6) for x in _seq_state.frame_latencies],
            }
            (output_root / f"_latency_profile_{seq}.json").write_text(
                json.dumps(latency_profile, indent=2) + "\n"
            )
        else:
            fps_summary_lines.append(f"{seq}\tfps=n/a\tmean_ms=n/a\tframes=0")

        if (cfg.relink_enabled or _seq_state._bridge_enabled) and hasattr(
            detector.tracker, "get_relink_debug"
        ):
            _rd = detector.tracker.get_relink_debug()
            # Host layout (cursor + d_relink_dbg_): see portable_or_tail.RELINK_DEBUG_HOST_INDEX.
            # Slots 5+ are portable OR-tail hook counters when Stage 1 hook is built in;
            # legacy Cheb-gate names no longer apply to these indices.
            from saccade.perception.eval.portable_or_tail import (
                RELINK_DEBUG_HOST_INDEX as _RD_IDX,
            )

            _rd_named = {
                name: int(_rd[idx]) if idx < len(_rd) else None
                for name, idx in _RD_IDX.items()
            }
            _rd_named["raw"] = [int(x) for x in _rd]
            (output_root / f"_relink_debug_{seq}.json").write_text(
                json.dumps(_rd_named, indent=2) + "\n"
            )
            print(
                f"🔗 Relink debug {seq}: archived={_rd_named.get('archived_cursor')} "
                f"births={_rd_named.get('births')} revived={_rd_named.get('revived')} "
                f"bridge_attempts={_rd_named.get('bridge_attempts')} "
                f"bridge_accepts={_rd_named.get('bridge_accepts')} "
                f"hook_eligible={_rd_named.get('hook_eligible')} "
                f"hook_rejected={_rd_named.get('hook_rejected')} "
                f"atom0={_rd_named.get('atom0_score_m_bridge')} "
                f"atom1={_rd_named.get('atom1_abs_log_h')} "
                f"atom2={_rd_named.get('atom2_dist_h')} "
                f"atom3={_rd_named.get('atom3_abs_ratio_m1')} "
                f"atom4={_rd_named.get('atom4_resid_mean')} "
                f"app_veto={_rd_named.get('app_veto')}"
            )

        _d0_capture_dir = str(
            (getattr(cfg, "kwargs", {}) or {}).get(
                "research_bridge_fidelity_capture_dir", ""
            )
        ).strip()
        if _d0_capture_dir:
            _d0_drain = getattr(
                detector.tracker, "drain_research_bridge_fidelity_events", None
            )
            if _d0_drain is None:
                raise RuntimeError(
                    "bridge-fidelity capture was enabled but the tracker cannot drain it"
                )
            _d0_capture = _d0_drain(seq=seq)
            try:
                _d0_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[4],
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                _d0_commit = "unknown"
            _d0_capture["provenance"] = {
                # D0's sealed packet remains on its frozen contract. R1 uses
                # the same observation buffer but emits a separate versioned
                # temporal-reduction payload through its own exporter.
                "capture_contract": str(
                    (getattr(cfg, "kwargs", {}) or {}).get(
                        "research_bridge_fidelity_capture_contract",
                        "d0_runtime_cuda_v1",
                    )
                ),
                "git_commit": _d0_commit,
                # Shadow = propose+capture without commit. A committing bridge
                # rewrites track identity, so a non-shadow capture cannot be
                # joined against a bridge-off pair cohort; the v2 exporter
                # fails closed on it.
                "shadow": bool(
                    (getattr(cfg, "kwargs", {}) or {}).get(
                        "research_bridge_fidelity_capture_shadow", False
                    )
                ),
                "bridge": {
                    "px": float(cfg.relink_bridge_px),
                    "at": int(cfg.relink_bridge_at),
                    "min_lost": int(cfg.relink_bridge_min_lost),
                    "ttl": int(cfg.relink_bridge_ttl),
                    "anchor": str(cfg.relink_bridge_anchor),
                    "anchor_rate": float(cfg.relink_bridge_anchor_rate),
                    "dir_bonus": float(cfg.relink_bridge_dir_bonus),
                },
                "detector": {
                    **dict(
                        (getattr(cfg, "kwargs", {}) or {}).get(
                            "research_bridge_fidelity_detector_provenance", {}
                        )
                    ),
                    "tiling": str(getattr(cfg, "tiling", "")),
                },
            }
            _d0_path = Path(_d0_capture_dir)
            _d0_path.mkdir(parents=True, exist_ok=True)
            (_d0_path / f"{seq}.json").write_text(
                json.dumps(_d0_capture, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not bool(_d0_capture.get("complete", False)):
                raise RuntimeError(
                    "bridge-fidelity capture overflowed; refuse an incomplete packet"
                )

        if _seq_state.relinker is not None and hasattr(
            _seq_state.relinker, "handover_count"
        ):
            ho_n = _seq_state.relinker.handover_count
            if ho_n > 0:
                print(f"🔗 Online handover {seq}: {ho_n} handovers accepted")

        _seq_state.results_lines, post_merge_stats = post_merge_output_tracklets(
            _seq_state.results_lines,
            enabled=cfg.post_lifecycle_merge,
            ttl=cfg.post_lifecycle_ttl,
            min_gap=cfg.post_lifecycle_min_gap,
            velocity_samples=cfg.post_lifecycle_velocity_samples,
            spatial_weight=cfg.post_lifecycle_spatial_weight,
            motion_weight=cfg.post_lifecycle_motion_weight,
            time_weight=cfg.post_lifecycle_time_weight,
            direction_weight=cfg.post_lifecycle_direction_weight,
            max_cost=cfg.post_lifecycle_max_cost,
            appearance_bank=_seq_state.output_appearance_bank,
            appearance_gate=cfg.post_lifecycle_appearance_gate,
            appearance_threshold=cfg.post_lifecycle_appearance_threshold,
            appearance_min_samples=cfg.post_lifecycle_appearance_min_samples,
            appearance_weight=cfg.post_lifecycle_appearance_weight,
            gap_uncertainty_weight=cfg.post_lifecycle_gap_uncertainty_weight,
            consistency_weight=cfg.post_lifecycle_consistency_weight,
            missing_appearance_cost=cfg.post_lifecycle_missing_appearance_cost,
        )

        if cheb_gr_extractor is not None and occ_audit_enabled:
            seq_img_dir = str(Path(cfg.core.data_root) / cfg.core.split / seq / "img1")
            if getattr(cfg, "occ_audit_bank_reference", False):
                from .clean_fifo_bank import build_filled_bank
                from .occ_audit import (
                    extract_audit_embeddings_post_exit,
                    occ_exit_audit_lines_from_bank,
                )

                occ_bank = build_filled_bank(
                    _seq_state.results_lines,
                    seq_img_dir,
                    cheb_gr_extractor,
                    appearance_occlusion_cov=cfg.appearance_occlusion_cov,
                    fifo_n=getattr(cfg, "occ_audit_bank_n", 20),
                    crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                )
                audit_embs = extract_audit_embeddings_post_exit(
                    _seq_state.results_lines,
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
                _seq_state.results_lines, oa_stats = occ_exit_audit_lines_from_bank(
                    _seq_state.results_lines,
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
                    chebgr_probe=bool(getattr(cfg, "occ_audit_chebgr_probe", False)),
                    chebgr_max_cost=float(
                        getattr(cfg, "occ_audit_chebgr_max_cost", 0.45)
                    ),
                    chebgr_margin=float(getattr(cfg, "occ_audit_chebgr_margin", 0.0)),
                    chebgr_pool_frac=float(
                        getattr(cfg, "occ_audit_chebgr_pool_frac", 0.3)
                    ),
                    chebgr_lambda=float(getattr(cfg, "occ_audit_chebgr_lambda", 2.0)),
                    chebgr_k2=int(getattr(cfg, "occ_audit_chebgr_k2", 6)),
                    chebgr_max_fwd=int(getattr(cfg, "occ_audit_chebgr_max_fwd", 50)),
                    chebgr_fuse_lambda=float(
                        getattr(cfg, "occ_audit_chebgr_fuse_lambda", 0.3)
                    ),
                )
            else:
                audit_embs = extract_audit_embeddings(
                    _seq_state.results_lines,
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
                _seq_state.results_lines, oa_stats = occ_exit_audit_lines(
                    _seq_state.results_lines,
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

        _live_evfifo = getattr(_seq_state, "live_evfifo", None)
        _ho_available = _live_evfifo is not None or (
            cheb_gr_extractor is not None and cheb_gr_online
        )
        # Legacy dispatch tested only for the extractor, so an occ-audit-only
        # run also reaches the merge stage; that behaviour is preserved below.
        _merge_legacy = cheb_gr_extractor is not None
        _merge_requested = _merge_legacy and cfg.cheb_gr_merge_enabled

        def _stage_handover() -> None:
            if _live_evfifo is not None:
                # Live evfifo-5-20-w3 bank accumulated during tracking (bounded
                # VRAM, no end-of-sequence disk re-read) — reproduces the offline
                # decision online.
                head_embs, bank_embs = _live_evfifo.build_embeddings(
                    _seq_state.results_lines,
                    extractor=cheb_gr_extractor,
                    bank_mode=getattr(cfg, "cheb_gr_online_bank_mode", "spread"),
                    bank_n=getattr(cfg, "cheb_gr_online_bank_n", 0),
                    decide_n=getattr(cfg, "cheb_gr_online_decide_n", 5),
                    n_samples=getattr(cfg, "cheb_gr_merge_n_samples", 50),
                    appearance_occlusion_cov=getattr(
                        cfg, "appearance_occlusion_cov", 0.4
                    ),
                    neighbor_iou_max=getattr(
                        cfg, "cheb_gr_online_neighbor_iou_max", 0.0
                    ),
                    crop_ring=(
                        _seq_state.perception_pipeline
                        if _seq_state.perception_pipeline is not None
                        and hasattr(_seq_state.perception_pipeline, "crop_ring_enabled")
                        and _seq_state.perception_pipeline.crop_ring_enabled()
                        else None
                    ),
                    alias=getattr(_seq_state.relinker, "deferred_alias", None),
                    stream_ptr=torch.cuda.current_stream().cuda_stream,
                )
            else:
                seq_img_dir = str(
                    Path(cfg.core.data_root) / cfg.core.split / seq / "img1"
                )
                head_embs, bank_embs = extract_handover_embeddings(
                    _seq_state.results_lines,
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
            _seq_state.results_lines, ho_stats = causal_handover_lines(
                _seq_state.results_lines,
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
                f"🧬 Cheb-GR Offline Handover: ids={ho_stats['ids_before']}->"
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

        def _stage_merge() -> None:
            seq_img_dir = str(Path(cfg.core.data_root) / cfg.core.split / seq / "img1")
            cheb_embeddings = extract_tracklet_embeddings(
                _seq_state.results_lines,
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
            _seq_state.results_lines, cheb_stats = cheb_gr_merge_output_tracklets(
                _seq_state.results_lines,
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
                f"🧬 Cheb-GR Merge: ids={cheb_stats['ids_before']}->"
                f"{cheb_stats['ids_after']} ({cheb_stats['merges']} merges)"
            )

        _order = getattr(cfg, "cheb_gr_postproc_order", "")
        if _order:
            if not (_ho_available and _merge_requested):
                raise ValueError(
                    f"cheb_gr_postproc_order={_order!r} chains both output-layer repairs, "
                    f"but handover_available={_ho_available} merge_enabled={cfg.cheb_gr_merge_enabled}. "
                    "Enable both, or leave the order empty to run a single stage."
                )
            _stages = [("handover", _stage_handover), ("merge", _stage_merge)]
            if _order == "merge_then_handover":
                _stages.reverse()
        elif _ho_available:
            if cfg.cheb_gr_merge_enabled:
                print(
                    "⚠️  Both offline handover and tracklet merge are configured, but "
                    "no --cheb-gr-postproc-order was given: running HANDOVER only, "
                    "merge is SKIPPED. This run is not a stacked result."
                )
            _stages = [("handover", _stage_handover)]
        elif _merge_legacy:
            _stages = [("merge", _stage_merge)]
        else:
            _stages = []

        _origin_keys = _mot_row_keys(_seq_state.results_lines)
        for _stage_idx, (_stage_name, _stage_fn) in enumerate(_stages, start=1):
            _before = list(_seq_state.results_lines)
            _stage_fn()
            if _order:
                _record = _postproc_stage_record(
                    _before, _seq_state.results_lines, _origin_keys
                )
                _append_dict_csv(
                    output_root / "_postproc_stage_log.csv",
                    [
                        {
                            "seq": seq,
                            "order": _order,
                            "stage_index": _stage_idx,
                            "stage": _stage_name,
                            **_record,
                        }
                    ],
                )
                _stage_dir = output_root / f"_postproc_stage{_stage_idx}_{_stage_name}"
                _stage_dir.mkdir(parents=True, exist_ok=True)
                (_stage_dir / f"{seq}.txt").write_text(
                    "\n".join(_seq_state.results_lines)
                )

        if cfg.post_lifecycle_merge:
            print(
                "🔗 Post Lifecycle Merge: "
                f"candidates={post_merge_stats['candidates']} "
                f"accepted={post_merge_stats['accepted']} "
                f"ids={post_merge_stats['ids_before']}->{post_merge_stats['ids_after']} "
                f"reject_app={post_merge_stats['reject_appearance']} "
                f"reject_app_missing={post_merge_stats['reject_appearance_missing']} "
                f"reject_app_consistency={post_merge_stats['reject_appearance_consistency']} "
                f"reject_cost={post_merge_stats['reject_cost']}"
            )

        if _seq_state.relinker is not None and cfg.kwargs.get(
            "semantic_delayed_claim", False
        ):
            local_alias = getattr(_seq_state.relinker, "deferred_alias", {})
            if local_alias:
                global_alias = {
                    int(global_id_mapper.map(seq, int(raw_id))): int(
                        global_id_mapper.map(seq, int(canonical_id))
                    )
                    for raw_id, canonical_id in dict(local_alias).items()
                    if int(raw_id) != int(canonical_id)
                }
                _seq_state.results_lines, deferred_stats = apply_deferred_alias(
                    _seq_state.results_lines,
                    global_alias,
                )
                if deferred_stats["lines_remapped"] > 0:
                    print(
                        "🔁 Deferred Claim Remap: "
                        f"aliases={deferred_stats['aliases']} "
                        f"lines={deferred_stats['lines_remapped']} "
                        f"ids={deferred_stats['ids_before']}->{deferred_stats['ids_after']}"
                    )

        if cfg.min_tracklet_len > 1 or cfg.min_tracklet_score > 0.0:
            _seq_state.results_lines, quality_stats = filter_low_quality_tracklets(
                _seq_state.results_lines,
                min_len=cfg.min_tracklet_len,
                min_score=cfg.min_tracklet_score,
            )
            if quality_stats["removed"] > 0:
                print(
                    f"🧹 Quality Filter: removed={quality_stats['removed']} "
                    f"ids={quality_stats['before']}->{quality_stats['after']}"
                )
        else:
            quality_stats = {"removed": 0, "before": 0, "after": 0}

        if cfg.interpolate_tracklets:
            _seq_state.results_lines, interp_stats = interpolate_tracklets(
                _seq_state.results_lines,
                max_gap=cfg.interpolate_max_gap,
                min_track_len=cfg.interpolate_min_track_len,
                min_h=cfg.interpolate_min_h,
            )
            print(
                f"🔀 Interpolation: tracks={interp_stats['tracks_interpolated']} "
                f"gaps={interp_stats['gaps_filled']} "
                f"frames_added={interp_stats['frames_added']}"
            )

        if not cfg.core.latency_only:
            Path(output_root / f"{seq}.txt").write_text(
                "\n".join(_seq_state.results_lines)
            )
        _sequence_result_callback = kwargs.get("sequence_result_callback")
        if _sequence_result_callback is not None:
            _sequence_result_callback(seq, tuple(_seq_state.results_lines))
        print(
            f"✅ Finished {seq} (Total Time: {time.time() - _seq_state.start_time:.2f}s)"
        )
        if _seq_state.relinker:
            _seq_state.relinker.report()
        _seq_state.lifecycle_merger.report()
        from .reporting import print_sequence_summary

        print_sequence_summary(
            cfg=cfg,
            seq=seq,
            seq_tile_diag=_seq_state.seq_tile_diag,
            profile_stages=profile_stages,
            seq_profiled_frames=_seq_state.seq_profiled_frames,
            top_level_stage_names=top_level_stage_names,
            seq_stage_samples=_seq_state.seq_stage_samples,
            overall_stage_totals=overall_stage_totals,
            overall_stage_samples=overall_stage_samples,
            breakdown_stage_names=breakdown_stage_names,
            seq_stage_totals=_seq_state.seq_stage_totals,
            native_reid_breakdown_names=native_reid_breakdown_names,
            seq_native_reid_samples=_seq_state.seq_native_reid_samples,
            gmc_breakdown_names=gmc_breakdown_names,
            seq_gmc_samples=_seq_state.seq_gmc_samples,
            overall_gmc_samples=overall_gmc_samples,
            segment_breakdown_names=segment_breakdown_names,
            seq_segment_samples=_seq_state.seq_segment_samples,
            overall_segment_samples=overall_segment_samples,
            seq_post_counts=_seq_state.seq_post_counts,
            overall_post_counts=overall_post_counts,
            seq_lazy_reid_frames=_seq_state.seq_lazy_reid_frames,
            seq_lazy_reid_candidates=_seq_state.seq_lazy_reid_candidates,
            overall_lazy_reid_candidates=overall_lazy_reid_candidates,
            overall_lazy_reid_frames=overall_lazy_reid_frames,
            overall_lazy_reid_crops=overall_lazy_reid_crops,
            overall_lazy_reid_self_pairs=overall_lazy_reid_self_pairs,
            overall_lazy_reid_self_pass=overall_lazy_reid_self_pass,
            overall_lazy_reid_self_sim_sum=overall_lazy_reid_self_sim_sum,
            overall_lazy_reid_arbiter_checks=overall_lazy_reid_arbiter_checks,
            overall_lazy_reid_arbiter_approve=overall_lazy_reid_arbiter_approve,
            seq_lazy_reid_crops=_seq_state.seq_lazy_reid_crops,
            seq_lazy_reid_self_pairs=_seq_state.seq_lazy_reid_self_pairs,
            seq_lazy_reid_self_pass=_seq_state.seq_lazy_reid_self_pass,
            seq_lazy_reid_self_sim_sum=_seq_state.seq_lazy_reid_self_sim_sum,
            seq_lazy_reid_arbiter_checks=_seq_state.seq_lazy_reid_arbiter_checks,
            seq_lazy_reid_arbiter_approve=_seq_state.seq_lazy_reid_arbiter_approve,
            overall_profiled_frames=overall_profiled_frames,
            stage_summary_lines=stage_summary_lines,
        )

        overall_profiled_frames += _seq_state.seq_profiled_frames
        if profile_stages and _seq_state.seq_profiled_frames > 0:
            seq_entry: dict = {
                "seq": seq,
                "frames": _seq_state.seq_profiled_frames,
                "stages": {},
            }
            for _sn in top_level_stage_names:
                _samp = _seq_state.seq_stage_samples.get(_sn, [])
                if _samp:
                    _arr = np.array(_samp, dtype=np.float64)
                    seq_entry["stages"][_sn] = {
                        "mean_ms": float(_arr.mean()),
                        "std_ms": float(_arr.std()),
                        "p95_ms": float(np.percentile(_arr, 95)),
                        "p99_ms": float(np.percentile(_arr, 99)),
                    }
            for _sn in breakdown_stage_names:
                _tot = _seq_state.seq_stage_totals.get(_sn, 0.0)
                if _tot > 0.0:
                    seq_entry["stages"][_sn] = {
                        "mean_ms": _tot / _seq_state.seq_profiled_frames
                    }
            _seq_post_means = {
                _sn: _seq_state.seq_stage_totals[_sn] / _seq_state.seq_profiled_frames
                for _sn in breakdown_stage_names
                if _seq_state.seq_stage_totals.get(_sn, 0.0) > 0.0
            }
            _seq_post_wall_ms = (
                seq_entry["stages"].get("postprocess", {}).get("mean_ms", 0.0)
            )
            _seq_post_gpu_ms = float(_seq_post_means.get("post_gpu_elapsed", 0.0))
            _seq_has_native_breakdown = any(
                _k.startswith("native_") for _k in _seq_post_means
            )
            _seq_excluded_post_stages = (
                {"post_filter", "post_nms"} if _seq_has_native_breakdown else set()
            )
            _seq_known_gpu_ms = float(
                sum(
                    _v
                    for _k, _v in _seq_post_means.items()
                    if (
                        _k.startswith("post_")
                        and _k != "post_gpu_elapsed"
                        and _k not in _seq_excluded_post_stages
                    )
                    or _k.startswith("native_")
                )
            )
            seq_entry["postprocess_attribution"] = {
                "wall_ms": float(_seq_post_wall_ms),
                "gpu_ms": _seq_post_gpu_ms,
                "unattributed_gpu_ms": max(0.0, _seq_post_gpu_ms - _seq_known_gpu_ms),
                "overhead_ms": max(0.0, float(_seq_post_wall_ms) - _seq_post_gpu_ms),
            }
            all_seq_profile.append(seq_entry)

    from .reporting import print_overall_summary

    if overall_latency_ms:
        _overall_lats = np.array(overall_latency_ms)
        _overall_profile = {
            "sequence": "OVERALL",
            "frames": len(overall_latency_ms),
            "mean_ms": round(float(np.mean(_overall_lats)), 6),
            "std_ms": round(float(np.std(_overall_lats)), 6),
            "p95_ms": round(float(np.percentile(_overall_lats, 95)), 6),
            "p99_ms": round(float(np.percentile(_overall_lats, 99)), 6),
        }
        (output_root / "_latency_profile.json").write_text(
            json.dumps(_overall_profile, indent=2) + "\n"
        )

    print_overall_summary(
        cfg=cfg,
        output_root=output_root,
        fps_summary_lines=fps_summary_lines,
        overall_latency_ms=overall_latency_ms,
        overall_throughput_frames=overall_throughput_frames,
        overall_throughput_seconds=overall_throughput_seconds,
        global_id_mapper=global_id_mapper,
        overall_profiled_frames=overall_profiled_frames,
        top_level_stage_names=top_level_stage_names,
        overall_stage_samples=overall_stage_samples,
        stage_summary_lines=stage_summary_lines,
        breakdown_stage_names=breakdown_stage_names,
        overall_stage_totals=overall_stage_totals,
        overall_post_counts=overall_post_counts,
        gmc_breakdown_names=gmc_breakdown_names,
        overall_gmc_samples=overall_gmc_samples,
        segment_breakdown_names=segment_breakdown_names,
        overall_segment_samples=overall_segment_samples,
        overall_lazy_reid_frames=overall_lazy_reid_frames,
        overall_lazy_reid_candidates=overall_lazy_reid_candidates,
        overall_lazy_reid_crops=overall_lazy_reid_crops,
        overall_lazy_reid_self_sim_sum=overall_lazy_reid_self_sim_sum,
        overall_lazy_reid_self_pairs=overall_lazy_reid_self_pairs,
        overall_lazy_reid_self_pass=overall_lazy_reid_self_pass,
        overall_lazy_reid_arbiter_checks=overall_lazy_reid_arbiter_checks,
        overall_lazy_reid_arbiter_approve=overall_lazy_reid_arbiter_approve,
        debug_dump_csv=debug_dump_csv,
        debug_stage_dump_rows=debug_stage_dump_rows,
        debug_birth_csv=debug_birth_csv,
        debug_birth_rows=debug_birth_rows,
        all_seq_profile=all_seq_profile,
    )

    # ── MOTMetrics Evaluation ──────────────────────────────────────────────────
    if cfg.core.latency_only:
        return {}

    from .metrics import run_motmetrics_evaluation

    return run_motmetrics_evaluation(
        data_root=cfg.core.data_root,
        split=cfg.core.split,
        output=str(cfg.output_root),
        sequences=",".join(cfg.seqs),
        detector=cfg.kwargs.get("detector"),
        score_on_gt_frames=bool(cfg.kwargs.get("score_on_gt_frames", False)),
    )
