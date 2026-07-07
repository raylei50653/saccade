# mypy: ignore-errors
"""Per-frame pipeline stage functions extracted from evaluator.py.

Each _run_* stage consumes/mutates EvalPipeline state (pipeline.py) plus the
per-frame FrameCtx. _run_frame (evaluator.py) sequences them.
"""

# mypy: ignore-errors
import os
import threading
import time
import dataclasses
from contextlib import nullcontext

import numpy as np
import torch

from typing import Any

from .quality import (
    compute_detection_quality_batch as _compute_detection_quality_batch,
)
from .utils import (
    append_stage_dump_rows as _append_stage_dump_rows,
    apply_narrow_person_score_bonus as _apply_narrow_person_score_bonus,
)
from .helpers import (
    materialize_gpu_track_results as _materialize_gpu_track_results,
    materialize_gpu_track_results_pinned as _materialize_gpu_track_results_pinned,
    materialize_gpu_track_results_async as _materialize_gpu_track_results_async,
    fast_emit_mot_lines as _fast_emit_mot_lines,
    prepare_host_track_batch as _prepare_host_track_batch,
    resolve_frame_tracks as _resolve_frame_tracks,
    prepare_track_candidates as _prepare_track_candidates,
    emit_resolved_tracks as _emit_resolved_tracks,
    finalize_frame_side_effects as _finalize_frame_side_effects,
    budget_reid_candidates as _budget_reid_candidates,
    front_occlusion_mask_xyxy as _front_occlusion_mask_xyxy,
)

# Perception/eval modules load local extensions before any torchvision fallback.


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


from saccade.perception.eval.detection import (  # noqa: E402
    match_keypoints_to_boxes,
)
from saccade.perception.eval.gmc import (  # noqa: E402
    PyGraphedGMC,
)
from saccade.perception.eval.pool import (  # noqa: E402
    rgb_chw_to_nv12_gpu,
    rgb_hwc_to_nv12_gpu,
)
from saccade.perception.eval.preprocess import (  # noqa: E402
    apply_frame_preprocess,
    geometry_mid_thresh_scale,
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
from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    need_reid_frame,
)

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


from .pipeline import (  # noqa: E402
    EvalPipeline,
    _detect_barrier_mode,
    _explicit_stream_probe_enabled,
)


def _record_profile_scope(name: str):
    profiler_mod = getattr(torch, "profiler", None)
    if profiler_mod is None:
        return nullcontext()
    record_fn = getattr(profiler_mod, "record_function", None)
    if record_fn is None:
        return nullcontext()
    return record_fn(name)


@dataclasses.dataclass
class FrameCtx:
    """Per-frame working tensors produced by the postprocess stage helpers.

    Companion to EvalPipeline (per-sequence): collects the outputs an extracted
    postprocess stage hands back to the frame loop, so a stage returns one
    object instead of a long tuple. Grows as more postprocess sub-stages move
    out. Only the values the loop actually consumes downstream are carried —
    Native NMS priors are carried here so the tensors stay alive until the C++
    call consumes their raw pointers.

    feature_dim carries the active FPN embedding dimension so downstream
    consumers (kalman gate, relink, tracker) can validate input shapes
    against the detection pipeline's current configuration.
    """

    raw_boxes_contig: torch.Tensor
    raw_scores_contig: torch.Tensor
    raw_classes_contig: torch.Tensor
    post_boxes: torch.Tensor
    post_scores: torch.Tensor
    post_classes: torch.Tensor
    geometry_suspect_mask: torch.Tensor
    priors_tensor: "torch.Tensor | None"
    prior_classes_tensor: "torch.Tensor | None"
    num_priors: int
    private_prior_boxes: "torch.Tensor | None"
    num_private_priors: int
    native_private_enabled: bool
    feature_dim: int


def _run_gmc_estimate(
    state: EvalPipeline,
    *,
    fused_boxes: torch.Tensor,
    _frame_gmc: torch.Tensor,
) -> tuple[torch.Tensor | None, bool]:
    """Estimate the per-frame GMC affine warp (extracted from run_eval).

    Returns ``(gmc_warp, gmc_uncertain)``. Mutates the caller-owned graph cell
    ``state.gmc_cuda_graph[0]`` and the reused ``state.shared_gmc_warp`` /
    ``state.gmc_frame_buf`` buffers in place, exactly as the original closure.
    """
    cfg = state.cfg
    gmc_estimator = state.gmc_estimator
    _shared_gmc_warp = state.shared_gmc_warp
    _use_direct_gmc = state.use_direct_gmc
    _gmc_graphable = state.gmc_graphable
    _gmc_cuda_graph = state.gmc_cuda_graph
    _gmc_frame_buf = state.gmc_frame_buf
    seq = state.seq
    w_orig = state.w_orig
    h_orig = state.h_orig
    profile_stages = state.profile_stages
    seq_gmc_samples = state.seq_gmc_samples
    local_gmc_warp: torch.Tensor | None = None
    local_gmc_uncertain = False
    _raw_warp = None
    if cfg.gmc_fg_mask and hasattr(gmc_estimator, "set_fg_mask_boxes_tensor"):
        if fused_boxes.numel() > 0:
            gmc_estimator.set_fg_mask_boxes_tensor(fused_boxes)
    elif cfg.gmc_fg_mask and hasattr(gmc_estimator, "set_fg_mask_boxes_gpu"):
        if fused_boxes.numel() > 0:
            gmc_estimator.set_fg_mask_boxes_gpu(
                fused_boxes.data_ptr(),
                fused_boxes.shape[0],
                torch.cuda.current_stream().cuda_stream,
            )
    elif cfg.gmc_fg_mask and hasattr(gmc_estimator, "set_fg_mask_boxes"):
        if fused_boxes.numel() > 0:
            _flat = fused_boxes.detach().cpu().view(-1).tolist()
            gmc_estimator.set_fg_mask_boxes(_flat)

    if isinstance(gmc_estimator, PyGraphedGMC):
        gmc_estimator.estimate_into_direct(
            _frame_gmc,
            _shared_gmc_warp,
        )
        local_gmc_warp = _shared_gmc_warp
    elif _use_direct_gmc:
        _w = _frame_gmc.shape[2]
        _h = _frame_gmc.shape[1]
        if not _gmc_graphable or cfg.gmc_fg_mask:
            # FG mask varies per frame → eager (non-graph) path.
            gmc_estimator.estimate_into_direct(
                _frame_gmc.data_ptr(),
                _w,
                _h,
                torch.cuda.current_stream().cuda_stream,
                _shared_gmc_warp.data_ptr(),
            )
        elif _gmc_cuda_graph[0] is None:
            # Warmup once on the fixed input buffer, then capture.
            # d_prev_gray_ state carries across replays
            # (verified bit-exact by test_gmc_cudagraph.py).
            _gmc_frame_buf.copy_(_frame_gmc)
            gmc_estimator.estimate_into_direct(
                _gmc_frame_buf.data_ptr(),
                _w,
                _h,
                torch.cuda.current_stream().cuda_stream,
                _shared_gmc_warp.data_ptr(),
            )
            torch.cuda.synchronize()
            _g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(_g):
                gmc_estimator.estimate_into_direct(
                    _gmc_frame_buf.data_ptr(),
                    _w,
                    _h,
                    torch.cuda.current_stream().cuda_stream,
                    _shared_gmc_warp.data_ptr(),
                )
            _gmc_cuda_graph[0] = _g
            print(
                f"🕯️ [GMCGraph] Captured C++ cuFFT GMC graph "
                f"for seq {seq} (img={_h}×{_w} "
                f"ds={cfg.gmc_downscale})"
            )
        else:
            # Steady state: copy new frame into the captured
            # input buffer and replay the recorded graph.
            _gmc_frame_buf.copy_(_frame_gmc)
            _gmc_cuda_graph[0].replay()
        local_gmc_warp = _shared_gmc_warp
    elif hasattr(gmc_estimator, "estimate_into"):
        local_gmc_warp = torch.empty(6, dtype=torch.float32, device=fused_boxes.device)
        gmc_estimator.estimate_into(
            _frame_gmc.data_ptr(),
            _frame_gmc.shape[2],
            _frame_gmc.shape[1],
            torch.cuda.current_stream().cuda_stream,
            local_gmc_warp.data_ptr(),
        )
    elif hasattr(gmc_estimator, "estimate_mat"):
        # C++ version or GlobalMotionCompensator
        _raw_warp = gmc_estimator.estimate(
            _frame_gmc.data_ptr(),
            w_orig,
            h_orig,
            torch.cuda.current_stream().cuda_stream,
        )
    else:
        _raw_warp = gmc_estimator.estimate(_frame_gmc)

    if _raw_warp is not None:
        if isinstance(_raw_warp, list):
            local_gmc_warp = torch.tensor(
                _raw_warp,
                dtype=torch.float32,
                device=fused_boxes.device,
            )
        else:
            local_gmc_warp = _raw_warp.to(fused_boxes.device)

    # A4: PCR quality feedback — flag marginal motion estimates so the
    # ReID budget function widens appearance coverage on uncertain frames.
    if hasattr(gmc_estimator, "pcr_score"):
        _pcr = gmc_estimator.pcr_score()
        local_gmc_uncertain = (
            local_gmc_warp is not None and 0.0 < _pcr < cfg.gmc_pcr_uncertain_thresh
        )
    if profile_stages and hasattr(gmc_estimator, "get_profile_stats"):
        _gmc_stats = gmc_estimator.get_profile_stats()
        if _gmc_stats:
            seq_gmc_samples["gmc_gray_downscale"].append(
                float(_gmc_stats.get("gray_downscale_ms", 0.0))
            )
            seq_gmc_samples["gmc_fg_mask"].append(
                float(_gmc_stats.get("fg_mask_ms", 0.0))
            )
            seq_gmc_samples["gmc_phase_corr"].append(
                float(_gmc_stats.get("phase_corr_ms", 0.0))
            )
            seq_gmc_samples["gmc_handoff"].append(
                float(_gmc_stats.get("handoff_ms", 0.0))
            )
    return local_gmc_warp, local_gmc_uncertain


def _run_materialize(
    state: EvalPipeline,
    *,
    tracker_result_buffers: Any,
    embeddings: "torch.Tensor | None",
    aligned_keypoints: "torch.Tensor | None",
    frame_id: int,
) -> Any:
    """Materialize tracker results for one frame (extracted from run_eval).

    Defer mode launches the async D2H + records the event; otherwise reads the
    pinned/host result synchronously. Returns ``track_results``. The deferred
    emit event/fid are carried on ``state`` (set on the defer path, left
    untouched on the non-defer path) and drained by the frame loop.
    """
    _defer_emit = state.defer_emit
    cfg = state.cfg
    _pinned_result_bufs = state.pinned_result_bufs
    _use_pinned_materialize = state.use_pinned_materialize
    seq_stage_totals = state.seq_stage_totals
    time_stage = state.time_stage
    _defer_emit_event = state.defer_emit_event
    _defer_emit_fid = state.defer_emit_fid
    if _defer_emit:
        _defer_emit_event, _ = _materialize_gpu_track_results_async(
            tracker_result_buffers,
            _pinned_result_bufs,
            default_class_id=cfg.person_class if cfg.track_person_only else None,
            include_det_idx=False,
        )
        _defer_emit_fid = frame_id
        track_results = {
            "count": 0,
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "ids": torch.empty((0,), dtype=torch.int32),
            "classes": None,
            "det_idx": None,
        }
    else:
        track_results, _ = time_stage(
            seq_stage_totals,
            "materialize",
            lambda: (
                _materialize_gpu_track_results_pinned(
                    tracker_result_buffers,
                    _pinned_result_bufs,
                    default_class_id=cfg.person_class
                    if cfg.track_person_only
                    else None,
                    include_det_idx=(
                        embeddings is not None or aligned_keypoints is not None
                    ),
                )
                if _use_pinned_materialize
                else _materialize_gpu_track_results(
                    tracker_result_buffers,
                    default_class_id=cfg.person_class
                    if cfg.track_person_only
                    else None,
                    include_det_idx=(
                        embeddings is not None or aligned_keypoints is not None
                    ),
                )
            ),
            sync_cuda=True,
        )
    state.defer_emit_event = _defer_emit_event
    state.defer_emit_fid = _defer_emit_fid
    return track_results


def _flush_db_tracker_out(state: EvalPipeline) -> None:
    """Flush one deferred double-buffer tracker output.

    Syncs the parity D2H event, builds CPU track_results from the pinned
    buffer, then runs the full emit/relink tail.  Called at the *start* of
    the next frame so the CPU emit overlaps the GPU postproc+GMC+ReID work
    while preserving relink→tracker ordering.
    """

    fid = state.db_emit_frame_id
    ev = state.db_emit_event
    if fid == 0 or ev is None:
        return
    ev.synchronize()
    parity = state.db_emit_parity
    pinned = state.double_buffer_tracker_out_pinned[parity]
    count = int(pinned["count"].item())
    track_results = {
        "count": count,
        "boxes": pinned["boxes"][:count].clone(),
        "scores": pinned["scores"][:count].clone(),
        "ids": pinned["ids"][:count].clone(),
        "classes": pinned["classes"][:count].clone(),
        "det_idx": pinned["det_idx"][:count].clone(),
    }
    ctx = state.db_emit_ctx
    tracker_result_bufs = ctx["tracker_result_buffers"]
    fused_boxes = ctx["fused_boxes"]
    fused_scores = ctx["fused_scores"]
    geometry_suspect_mask = ctx["geometry_suspect_mask"]
    embeddings = ctx["embeddings"]
    gmc_warp = ctx["gmc_warp"]
    (
        state.prev_track_ids,
        _emit_lines,
    ) = _run_emit(
        state,
        track_results=track_results,
        tracker_result_buffers=tracker_result_bufs,
        fused_boxes=fused_boxes,
        fused_scores=fused_scores,
        geometry_suspect_mask=geometry_suspect_mask,
        embeddings=embeddings,
        gmc_warp=gmc_warp,
        frame_birth_events=[],
        frame_id=fid,
        prev_track_ids=state.prev_track_ids,
        track_results_on_host=True,
    )
    state.results_lines.extend(_emit_lines)
    state.db_emit_frame_id = 0
    state.db_emit_event = None
    state.db_emit_ctx.clear()


def _apply_score_jitter(
    state: EvalPipeline, fused_scores: torch.Tensor, fused_boxes: torch.Tensor
) -> None:
    """Perturbation-robustness probe: deterministic tiny noise on scores/boxes.

    Enabled by SACCADE_SCORE_JITTER="<seed>[:<score_eps>[:<box_eps_px>]]"
    (defaults 1e-3 / 0.05 px). Seeded per (seed, sequence) so each run is
    reproducible. Emulates implementation-level bit perturbations (fp16
    rounding, kernel reorders): box jitter is the dominant channel — the
    tracker amplifies geometry flips through birth/confirm/bridge decisions —
    while score-only jitter barely moves metrics. Used to check that a tuning
    delta survives perturbation redraws instead of being one lottery draw
    (see perf_attribution_whole_graph_m.md).
    """
    jit = getattr(state, "_score_jitter", None)
    if jit is None:
        spec = os.environ.get("SACCADE_SCORE_JITTER", "")
        if not spec:
            state._score_jitter = ()
            return
        import zlib

        parts = spec.split(":")
        gen = torch.Generator(device=fused_scores.device)
        gen.manual_seed((int(parts[0]) << 32) ^ zlib.crc32(state.seq.encode()))
        eps = float(parts[1]) if len(parts) > 1 and parts[1] else 1e-3
        box_eps = float(parts[2]) if len(parts) > 2 and parts[2] else 0.05
        jit = state._score_jitter = (gen, eps, box_eps)
    if jit:
        gen, eps, box_eps = jit
        fused_scores.add_(
            torch.randn(fused_scores.shape, generator=gen, device=fused_scores.device),
            alpha=eps,
        )
        if box_eps > 0:
            fused_boxes.add_(
                torch.randn(
                    fused_boxes.shape, generator=gen, device=fused_boxes.device
                ),
                alpha=box_eps,
            )


def _run_track(
    state: EvalPipeline,
    *,
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    fused_classes: torch.Tensor,
    gmc_warp: "torch.Tensor | None",
    embeddings: "torch.Tensor | None",
    mid_thresh_scale: float,
    tracker_result_buffers: Any,
    synchronize: bool = True,
) -> Any:
    """Tracker update for one frame (extracted from run_eval).

    Uses the captured GraphedTrackerUpdate replay when available, else the
    direct ``update_into``. Returns the result buffers (the graph path returns
    fresh ``gtu.out_*`` tensors; the direct path writes in place).
    """
    _apply_score_jitter(state, fused_scores, fused_boxes)
    gtu = state.gtu
    detector = state.detector
    cfg = state.cfg
    seq_stage_totals = state.seq_stage_totals
    time_stage = state.time_stage
    if embeddings is not None:
        expected_dim = (
            state.contract.feature_dim
            if state.contract.fpn_reid_mode
            else (state.extractor.feature_dim if state.extractor is not None else 0)
        )
        actual_dim = embeddings.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch in _run_track: "
                f"got {actual_dim}, expected {expected_dim} "
                f"(fpn_reid_mode={state.contract.fpn_reid_mode}). "
                f"The detection FPN config has likely changed — "
                f"update kalman gate r_scale and tracker embedding_dim accordingly."
            )
    if gtu is not None:
        gtu.copy_inputs(
            fused_boxes,
            fused_scores,
            fused_classes.to(torch.int32),
            gmc=gmc_warp,
        )
        tracker_result_buffers, _ = time_stage(
            seq_stage_totals,
            "track",
            lambda: gtu.replay(),
            sync_cuda=synchronize,
        )
    else:
        _, _ = time_stage(
            seq_stage_totals,
            "track",
            lambda: detector.tracker.update_into(
                fused_boxes,
                fused_scores,
                fused_classes.to(torch.int32),
                tracker_result_buffers,
                embeddings=embeddings if cfg.use_tracker_reid else None,
                gmc=gmc_warp,
                mid_thresh_scale=mid_thresh_scale,
            ),
            sync_cuda=synchronize,
        )
    return tracker_result_buffers


def _run_nms(
    state: EvalPipeline,
    *,
    raw_boxes_contig: torch.Tensor,
    raw_scores_contig: torch.Tensor,
    raw_classes_contig: torch.Tensor,
    raw_box_count: int,
    priors_tensor: "torch.Tensor | None",
    prior_classes_tensor: "torch.Tensor | None",
    num_priors: int,
    private_prior_boxes: "torch.Tensor | None",
    num_private_priors: int,
    native_private_enabled: bool,
    is_tiled: bool,
    nms_graph: Any,
) -> tuple[int, Any]:
    """Native NMS + detection filter for one frame (extracted).

    Uses the fixed CUDA graph only for the no-prior/no-private fast path. ONMS
    priors and private continuation require per-frame pointers, so they call the
    synchronous native wrapper directly.
    """
    perception_pipeline = state.perception_pipeline
    _nms_in = state.nms_in
    _post_bufs = state.post_bufs
    _NMS_FIXED_N = state.nms_fixed_n
    w_orig = state.w_orig
    h_orig = state.h_orig
    _nms_graph = nms_graph
    _explicit_probe = _explicit_stream_probe_enabled()
    if _explicit_probe and state.stream_post is not None:
        current_stream = state.stream_post.cuda_stream
    else:
        current_stream = torch.cuda.current_stream().cuda_stream
    priors_ptr = (
        priors_tensor.data_ptr() if num_priors > 0 and priors_tensor is not None else 0
    )
    prior_classes_ptr = (
        prior_classes_tensor.data_ptr()
        if num_priors > 0 and prior_classes_tensor is not None
        else 0
    )
    private_priors_ptr = (
        private_prior_boxes.data_ptr()
        if native_private_enabled
        and num_private_priors > 0
        and private_prior_boxes is not None
        else 0
    )

    if native_private_enabled:
        _use_split = os.environ.get("SACCADE_MAIN_NMS_SPLIT", "") in (
            "1",
            "true",
            "yes",
        )
        if _use_split:
            out_count_buf = state.nms_graph_out_count
            if out_count_buf is None:
                out_count_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
                state.nms_graph_out_count = out_count_buf
            n_post = perception_pipeline.process_detections_split_pipeline(
                raw_boxes_contig.data_ptr(),
                raw_scores_contig.data_ptr(),
                raw_classes_contig.data_ptr(),
                raw_box_count,
                w_orig,
                h_orig,
                is_tiled,
                _post_bufs["boxes"].data_ptr(),
                _post_bufs["scores"].data_ptr(),
                _post_bufs["classes"].data_ptr(),
                _post_bufs["suspect"].data_ptr(),
                out_count_buf.data_ptr(),
                priors_ptr,
                prior_classes_ptr,
                num_priors,
                state.onms_prior_iou_threshold,
                private_priors_ptr,
                num_private_priors,
                current_stream,
            )
            return n_post, _nms_graph
        n_post = perception_pipeline.process_detections_n_private(
            raw_boxes_contig.data_ptr(),
            raw_scores_contig.data_ptr(),
            raw_classes_contig.data_ptr(),
            raw_box_count,
            w_orig,
            h_orig,
            is_tiled,
            _post_bufs["boxes"].data_ptr(),
            _post_bufs["scores"].data_ptr(),
            _post_bufs["classes"].data_ptr(),
            _post_bufs["suspect"].data_ptr(),
            priors_ptr,
            prior_classes_ptr,
            num_priors,
            state.onms_prior_iou_threshold,
            private_priors_ptr,
            num_private_priors,
            current_stream,
        )
        return n_post, _nms_graph

    if num_priors > 0:
        n_post = perception_pipeline.process_detections_n(
            raw_boxes_contig.data_ptr(),
            raw_scores_contig.data_ptr(),
            raw_classes_contig.data_ptr(),
            raw_box_count,
            w_orig,
            h_orig,
            is_tiled,
            _post_bufs["boxes"].data_ptr(),
            _post_bufs["scores"].data_ptr(),
            _post_bufs["classes"].data_ptr(),
            _post_bufs["suspect"].data_ptr(),
            priors_ptr,
            prior_classes_ptr,
            num_priors,
            state.onms_prior_iou_threshold,
            current_stream,
        )
        return n_post, _nms_graph

    # process_detections_n releases GIL for the full filter+NMS+sync
    # sequence so sibling threads can run Python while GPU is busy.
    _use_graph_nms = perception_pipeline is not None
    if _use_graph_nms:
        _pad_stream = current_stream
        copy_pad_detections(
            raw_boxes_contig.data_ptr(),
            raw_scores_contig.data_ptr(),
            raw_classes_contig.data_ptr(),
            min(raw_box_count, _NMS_FIXED_N),
            _nms_in["boxes"].data_ptr(),
            _nms_in["scores"].data_ptr(),
            _nms_in["classes"].data_ptr(),
            _NMS_FIXED_N,
            _pad_stream,
        )

    if _nms_graph is None:
        out_count_buf = state.nms_graph_out_count
        if out_count_buf is None:
            out_count_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
            state.nms_graph_out_count = out_count_buf
        perception_pipeline.process_detections_graph(
            _nms_in["boxes"].data_ptr(),
            _nms_in["scores"].data_ptr(),
            _nms_in["classes"].data_ptr(),
            _NMS_FIXED_N,
            w_orig,
            h_orig,
            is_tiled,
            _post_bufs["boxes"].data_ptr(),
            _post_bufs["scores"].data_ptr(),
            _post_bufs["classes"].data_ptr(),
            _post_bufs["suspect"].data_ptr(),
            out_count_buf.data_ptr(),
            0,
            0,
            0,
            0.0,
            current_stream,
        )
        torch.cuda.synchronize()
        if _explicit_probe and state.stream_post is not None:
            _capture_ctx = torch.cuda.stream(state.stream_post)
        else:
            _capture_ctx = nullcontext()
        _nms_graph = torch.cuda.CUDAGraph()
        if os.environ.get("SACCADE_STREAM_DEBUG", "") in ("1", "true", "yes"):
            _cs = torch.cuda.current_stream()
            _cs_stream_post = (
                state.stream_post.cuda_stream if state.stream_post is not None else 0
            )
            print(
                f"[STREAM] _run_nms graph capture: current={_cs.cuda_stream:#x} stream_post={_cs_stream_post:#x}"
            )
        with _capture_ctx, torch.cuda.graph(_nms_graph):
            perception_pipeline.process_detections_graph(
                _nms_in["boxes"].data_ptr(),
                _nms_in["scores"].data_ptr(),
                _nms_in["classes"].data_ptr(),
                _NMS_FIXED_N,
                w_orig,
                h_orig,
                is_tiled,
                _post_bufs["boxes"].data_ptr(),
                _post_bufs["scores"].data_ptr(),
                _post_bufs["classes"].data_ptr(),
                _post_bufs["suspect"].data_ptr(),
                out_count_buf.data_ptr(),
                0,
                0,
                0,
                0.0,
                current_stream,
            )
        print("🕯️ [NMSGraph] Captured NMS graph")
    else:
        if state._frame_stage_times is not None:
            _t_graph_replay = time.perf_counter()
        if os.environ.get("SACCADE_STREAM_DEBUG", "") in ("1", "true", "yes"):
            _cs = torch.cuda.current_stream()
            _cs_stream_post = (
                state.stream_post.cuda_stream if state.stream_post is not None else 0
            )
            print(
                f"[STREAM] _run_nms graph replay: current={_cs.cuda_stream:#x} stream_post={_cs_stream_post:#x}"
            )
        if _explicit_probe and state.stream_post is not None:
            torch.cuda.set_stream(state.stream_post)
            _nms_graph.replay()
        else:
            _nms_graph.replay()
        if state._frame_stage_times is not None:
            state._frame_stage_times["post_graph_replay"] = round(
                (time.perf_counter() - _t_graph_replay) * 1000, 6
            )
    _logical_cap = int(os.environ.get("SACCADE_NMS_LOGICAL_N", str(_NMS_FIXED_N)))
    n_post = min(_logical_cap, _NMS_FIXED_N)
    if state._frame_stage_times is not None and _explicit_probe:
        _sp = state.stream_post
        state._frame_stage_times["post_capture_stream_handle"] = (
            _sp.cuda_stream if _sp is not None else 0
        )
        state._frame_stage_times["post_replay_stream_handle"] = (
            _sp.cuda_stream if _sp is not None else 0
        )
    return n_post, _nms_graph


def _capture_main_nms_graph(
    state: EvalPipeline,
    *,
    raw_box_count: int,
    is_tiled: bool,
) -> None:
    """Capture process_detections_main_nms_graph into a torch.cuda.CUDAGraph.

    Works around torch.cuda.CUDAGraph() memory snapshotting by allocating
    fresh output tensors for the graph to write into. Input data is copied
    to _main_nms_in before each replay (outside the graph).
    """
    _perception_pipeline = state.perception_pipeline
    if _perception_pipeline is None:
        return
    _main_nms_in = state.main_nms_in
    _graph_out = state.main_nms_graph_out
    _NMS_FIXED_N = state.nms_fixed_n
    n_in = min(raw_box_count, _NMS_FIXED_N)

    out_count_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
    state.main_nms_graph_out_count = out_count_buf

    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Warm up (eager, outside graph capture)
    _perception_pipeline.process_detections_main_nms_graph(
        _main_nms_in["boxes"].data_ptr(),
        _main_nms_in["scores"].data_ptr(),
        _main_nms_in["classes"].data_ptr(),
        n_in,
        state.w_orig,
        state.h_orig,
        is_tiled,
        _graph_out["boxes"].data_ptr(),
        _graph_out["scores"].data_ptr(),
        _graph_out["classes"].data_ptr(),
        _graph_out["suspect"].data_ptr(),
        out_count_buf.data_ptr(),
        0,
        0,
        0,
        0.0,
        stream_ptr,
    )
    torch.cuda.synchronize()

    _graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(_graph):
        _perception_pipeline.process_detections_main_nms_graph(
            _main_nms_in["boxes"].data_ptr(),
            _main_nms_in["scores"].data_ptr(),
            _main_nms_in["classes"].data_ptr(),
            n_in,
            state.w_orig,
            state.h_orig,
            is_tiled,
            _graph_out["boxes"].data_ptr(),
            _graph_out["scores"].data_ptr(),
            _graph_out["classes"].data_ptr(),
            _graph_out["suspect"].data_ptr(),
            out_count_buf.data_ptr(),
            0,
            0,
            0,
            0.0,
            stream_ptr,
        )
    state.main_nms_graph = _graph
    print("🕯️ [MainNMSGraph] Captured main NMS graph")


def _run_nms_shadow_compare(
    state: EvalPipeline,
    *,
    raw_boxes_contig: torch.Tensor,
    raw_scores_contig: torch.Tensor,
    raw_classes_contig: torch.Tensor,
    raw_box_count: int,
    priors_tensor: "torch.Tensor | None",
    prior_classes_tensor: "torch.Tensor | None",
    num_priors: int,
    private_prior_boxes: "torch.Tensor | None",
    num_private_priors: int,
    native_private_enabled: bool,
    is_tiled: bool,
    nms_graph: Any,
) -> tuple[int, Any]:
    """Shadow compare: runs monolithic on both post_bufs and shadow bufs,
    then compares split (main_nms + private_append) against monolithic.
    """

    _post_bufs = state.post_bufs
    _NMS_FIXED_N = state.nms_fixed_n
    _perception_pipeline = state.perception_pipeline
    _cfg = state.cfg

    # ── Monolithic on post_bufs (production path) ──────────────────
    n_post_mono, _nms_graph_out = _run_nms(
        state,
        raw_boxes_contig=raw_boxes_contig,
        raw_scores_contig=raw_scores_contig,
        raw_classes_contig=raw_classes_contig,
        raw_box_count=raw_box_count,
        priors_tensor=priors_tensor,
        prior_classes_tensor=prior_classes_tensor,
        num_priors=num_priors,
        private_prior_boxes=private_prior_boxes,
        num_private_priors=num_private_priors,
        native_private_enabled=native_private_enabled,
        is_tiled=is_tiled,
        nms_graph=nms_graph,
    )
    torch.cuda.synchronize()

    mono_boxes = _post_bufs["boxes"][:n_post_mono].clone()
    mono_scores = _post_bufs["scores"][:n_post_mono].clone()
    mono_classes = _post_bufs["classes"][:n_post_mono].clone()
    mono_suspect = _post_bufs["suspect"][:n_post_mono].clone()

    # ── Split path on shadow buffers ─────────────────────────────
    _shadow_boxes = torch.empty((_NMS_FIXED_N, 4), dtype=torch.float32, device="cuda")
    _shadow_scores = torch.empty((_NMS_FIXED_N,), dtype=torch.float32, device="cuda")
    _shadow_classes = torch.empty((_NMS_FIXED_N,), dtype=torch.int32, device="cuda")
    _shadow_suspect = torch.empty((_NMS_FIXED_N,), dtype=torch.bool, device="cuda")
    _shadow_count = torch.zeros(1, dtype=torch.int32, device="cuda")

    _priors_ptr = (
        priors_tensor.data_ptr() if num_priors > 0 and priors_tensor is not None else 0
    )
    _prior_classes_ptr = (
        prior_classes_tensor.data_ptr()
        if num_priors > 0 and prior_classes_tensor is not None
        else 0
    )
    _private_priors_ptr = (
        private_prior_boxes.data_ptr()
        if num_private_priors > 0 and private_prior_boxes is not None
        else 0
    )
    current_stream = torch.cuda.current_stream().cuda_stream

    assert _perception_pipeline is not None
    n_post_split = _perception_pipeline.process_detections_split_pipeline(
        raw_boxes_contig.data_ptr(),
        raw_scores_contig.data_ptr(),
        raw_classes_contig.data_ptr(),
        raw_box_count,
        state.w_orig,
        state.h_orig,
        is_tiled,
        _shadow_boxes.data_ptr(),
        _shadow_scores.data_ptr(),
        _shadow_classes.data_ptr(),
        _shadow_suspect.data_ptr(),
        _shadow_count.data_ptr(),
        _priors_ptr,
        _prior_classes_ptr,
        num_priors,
        state.onms_prior_iou_threshold,
        _private_priors_ptr,
        num_private_priors,
        current_stream,
    )

    # ── Compare ───────────────────────────────────────────────────
    if n_post_mono != n_post_split:
        print(
            f"[NMS_SHADOW] frame={state.current_frame_id} COUNT MISMATCH: "
            f"mono={n_post_mono} split={n_post_split}"
        )
    else:
        _split_boxes = _shadow_boxes[:n_post_split]
        _split_scores = _shadow_scores[:n_post_split]
        _split_classes = _shadow_classes[:n_post_split]
        _split_suspect = _shadow_suspect[:n_post_split]

        boxes_ok = torch.allclose(mono_boxes, _split_boxes, atol=1e-4)
        scores_ok = torch.allclose(mono_scores, _split_scores, atol=1e-4)
        classes_ok = bool((mono_classes == _split_classes).all())
        suspect_ok = bool((mono_suspect == _split_suspect).all())

        if not (boxes_ok and scores_ok and classes_ok and suspect_ok):
            # ── Canonical sort compare: check if it's ordering-only ──
            def _canonical_idx(boxes, scores, classes, suspect):
                """Lexsort by (class, -score, x1, y1, x2, y2)."""
                cb, cs, cc = boxes.cpu(), scores.cpu(), classes.cpu()
                keys = [
                    cb[:, 3].double(),
                    cb[:, 2].double(),
                    cb[:, 1].double(),
                    cb[:, 0].double(),
                    (-cs).double(),
                    cc.double(),
                ]
                idx = torch.arange(boxes.shape[0])
                for key in reversed(keys):
                    idx = idx[torch.argsort(key[idx], stable=True)]
                return idx.to(boxes.device)

            _mi = _canonical_idx(mono_boxes, mono_scores, mono_classes, mono_suspect)
            _si = _canonical_idx(
                _split_boxes, _split_scores, _split_classes, _split_suspect
            )

            sorted_boxes_ok = torch.allclose(
                mono_boxes[_mi], _split_boxes[_si], atol=1e-4
            )
            sorted_scores_ok = torch.allclose(
                mono_scores[_mi], _split_scores[_si], atol=1e-4
            )
            sorted_classes_ok = bool((mono_classes[_mi] == _split_classes[_si]).all())
            sorted_suspect_ok = bool((mono_suspect[_mi] == _split_suspect[_si]).all())

            if (
                sorted_boxes_ok
                and sorted_scores_ok
                and sorted_classes_ok
                and sorted_suspect_ok
            ):
                print(
                    f"[NMS_SHADOW] frame={state.current_frame_id} "
                    f"ordering-only (same set, different index order)"
                )
            else:
                issues = []
                if not sorted_boxes_ok:
                    diff = (mono_boxes[_mi] - _split_boxes[_si]).abs()
                    issues.append(f"boxes max diff={diff.max().item():.6f}")
                if not sorted_scores_ok:
                    diff = (mono_scores[_mi] - _split_scores[_si]).abs()
                    issues.append(f"scores max diff={diff.max().item():.6f}")
                if not sorted_classes_ok:
                    issues.append("classes")
                if not sorted_suspect_ok:
                    issues.append("suspect")
                print(
                    f"[NMS_SHADOW] frame={state.current_frame_id} "
                    f"REAL MISMATCH: " + " ".join(issues)
                )

    # ── Graph shadow: main NMS graph vs eager ──────────────────────
    _graph_shadow_enabled = os.environ.get("SACCADE_MAIN_NMS_GRAPH_SHADOW", "") in (
        "1",
        "true",
        "yes",
    )
    if _graph_shadow_enabled:
        _graph_in = state.main_nms_in
        _graph_out = state.main_nms_graph_out

        # Copy per-frame raw detections to fixed graph input buffers
        from saccade_tracking_ext import copy_pad_detections

        copy_pad_detections(
            raw_boxes_contig.data_ptr(),
            raw_scores_contig.data_ptr(),
            raw_classes_contig.data_ptr(),
            min(raw_box_count, _NMS_FIXED_N),
            _graph_in["boxes"].data_ptr(),
            _graph_in["scores"].data_ptr(),
            _graph_in["classes"].data_ptr(),
            _NMS_FIXED_N,
            current_stream,
        )

        if state.main_nms_graph is None:
            _capture_main_nms_graph(
                state,
                raw_box_count=raw_box_count,
                is_tiled=is_tiled,
            )

        if state.main_nms_graph is None:
            return n_post_mono, _nms_graph_out

        state.main_nms_graph.replay()
        torch.cuda.synchronize()

        # Save graph main NMS output before private append
        _graph_count = int(state.main_nms_graph_out_count.item())
        _graph_boxes = _graph_out["boxes"][:_graph_count].clone()
        _graph_scores = _graph_out["scores"][:_graph_count].clone()
        _graph_classes = _graph_out["classes"][:_graph_count].clone()
        _graph_suspect = _graph_out["suspect"][:_graph_count].clone()

        # Run private append on graph output
        _perception_pipeline.process_private_continuation_append(
            _graph_out["boxes"].data_ptr(),
            _graph_out["scores"].data_ptr(),
            _graph_out["classes"].data_ptr(),
            _graph_out["suspect"].data_ptr(),
            state.main_nms_graph_out_count.data_ptr(),
            _NMS_FIXED_N,
            _private_priors_ptr,
            num_private_priors,
            current_stream,
        )
        torch.cuda.synchronize()

        _graph_final_count = int(state.main_nms_graph_out_count.item())

        # Compare graph+private final vs monolithic final
        if n_post_mono != _graph_final_count:
            print(
                f"[NMS_GRAPH_SHADOW] frame={state.current_frame_id} "
                f"COUNT: mono={n_post_mono} graph={_graph_final_count}"
            )
        else:
            _gf_boxes = _graph_out["boxes"][:_graph_final_count]
            _gf_scores = _graph_out["scores"][:_graph_final_count]
            _gf_classes = _graph_out["classes"][:_graph_final_count]
            _gf_suspect = _graph_out["suspect"][:_graph_final_count]

            boxes_ok = torch.allclose(mono_boxes, _gf_boxes, atol=1e-4)
            scores_ok = torch.allclose(mono_scores, _gf_scores, atol=1e-4)
            classes_ok = bool((mono_classes == _gf_classes).all())
            suspect_ok = bool((mono_suspect == _gf_suspect).all())

            if not (boxes_ok and scores_ok and classes_ok and suspect_ok):
                issues = []
                if not boxes_ok:
                    diff = (mono_boxes - _gf_boxes).abs()
                    issues.append(f"boxes max diff={diff.max().item():.6f}")
                if not scores_ok:
                    diff = (mono_scores - _gf_scores).abs()
                    issues.append(f"scores max diff={diff.max().item():.6f}")
                if not classes_ok:
                    issues.append("classes")
                if not suspect_ok:
                    issues.append("suspect")
                print(
                    f"[NMS_GRAPH_SHADOW] frame={state.current_frame_id} "
                    + " ".join(issues)
                )

    return n_post_mono, _nms_graph_out


def _run_native_tensor_prep(
    state: EvalPipeline,
    *,
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    fused_classes: torch.Tensor,
    seq_narrow_bonus: float,
    enable_onms: bool,
    onms_min_track_age: int,
    onms_min_track_score: float,
    raw_box_count: int,
) -> FrameCtx:
    """Cast detections to native NMS dtypes + slice output buffers (extracted).

    Casts the fused detections to the contiguous float32/int32 layout the native
    NMS expects, applies the narrow-person score bonus, slices the reusable
    post-process output buffers, and (when ONMS is enabled) builds the active-track
    priors. Returns a FrameCtx with the values the postprocess tail consumes,
    including prior tensors whose raw pointers are passed to native NMS.
    """
    cfg = state.cfg
    w_orig = state.w_orig
    h_orig = state.h_orig
    detector = state.detector
    _post_bufs = state.post_bufs
    native_private_enabled = bool(
        state.native_cfg is not None
        and getattr(state.native_cfg, "private_continuation_enabled", False)
    )
    with _record_profile_scope("post.native_tensor_prep"):
        raw_boxes_contig = fused_boxes.to(torch.float32).contiguous()
        raw_scores_contig = fused_scores.to(torch.float32).contiguous()
        raw_classes_contig = fused_classes.to(torch.int32).contiguous()
        raw_scores_contig = _apply_narrow_person_score_bonus(
            raw_boxes_contig,
            raw_scores_contig,
            raw_classes_contig,
            frame_w=w_orig,
            frame_h=h_orig,
            person_class=cfg.person_class,
            bonus=seq_narrow_bonus,
            max_width_ratio=cfg.narrow_person_max_width_ratio,
            min_height_ratio=cfg.narrow_person_min_height_ratio,
            min_aspect=cfg.narrow_person_min_aspect,
            max_aspect=cfg.narrow_person_max_aspect,
        )
        post_boxes = _post_bufs["boxes"][:raw_box_count]
        post_scores = _post_bufs["scores"][:raw_box_count]
        post_classes = _post_bufs["classes"][:raw_box_count]
        geometry_suspect_mask = _post_bufs["suspect"][:raw_box_count]

        # Fetch priors for Occlusion-aware NMS
        priors_tensor = None
        prior_classes_tensor = None
        num_priors = 0
        if enable_onms:
            priors_tensor, prior_classes_tensor = _build_active_track_priors(
                detector.tracker,
                raw_boxes_contig.device,
                min_track_age=onms_min_track_age,
                min_track_score=onms_min_track_score,
            )
        if (
            enable_onms
            and priors_tensor is not None
            and prior_classes_tensor is not None
        ):
            num_priors = priors_tensor.size(0)
        private_prior_boxes = None
        num_private_priors = 0
        if native_private_enabled and (
            cfg.private_prior_iou_threshold > 0.0
            or cfg.private_prior_center_threshold > 0.0
        ):
            private_prior_boxes, _ = _build_active_track_priors(
                detector.tracker,
                raw_boxes_contig.device,
                min_track_age=0,
                max_track_age=cfg.private_prior_max_age,
                min_track_score=0.0,
            )
            if private_prior_boxes is not None:
                num_private_priors = private_prior_boxes.size(0)
    feature_dim = (
        state.contract.feature_dim
        if state.contract.fpn_reid_mode
        else (state.extractor.feature_dim if state.extractor is not None else 0)
    )
    return FrameCtx(
        raw_boxes_contig=raw_boxes_contig,
        raw_scores_contig=raw_scores_contig,
        raw_classes_contig=raw_classes_contig,
        post_boxes=post_boxes,
        post_scores=post_scores,
        post_classes=post_classes,
        geometry_suspect_mask=geometry_suspect_mask,
        priors_tensor=priors_tensor,
        prior_classes_tensor=prior_classes_tensor,
        num_priors=num_priors,
        private_prior_boxes=private_prior_boxes,
        num_private_priors=num_private_priors,
        native_private_enabled=native_private_enabled,
        feature_dim=feature_dim,
    )


def _run_post_nms_finalize(
    state: EvalPipeline,
    fctx: FrameCtx,
    *,
    n_post: int,
    source_boxes_for_keypoints: "torch.Tensor | None",
    source_keypoints: "torch.Tensor | None",
    current_stage_sample_active: bool,
    post_seg_events: list,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    "torch.Tensor | None",
    int,
    int,
]:
    """Slice native-NMS outputs to n_post, align keypoints, quality-scale (extracted).

    Final postprocess tail on the perception_pipeline path: slices the fixed output
    buffers down to the kept count, matches source keypoints to the kept boxes, and
    (CPU-tracker only) multiplies in per-detection quality factors. Returns
    ``(fused_boxes, fused_scores, fused_classes, geometry_suspect_mask,
    aligned_keypoints, after_filter_count, after_nms_count)``. The per-frame
    profiling locals (``current_stage_sample_active`` flag, ``post_seg_events``
    marker list which is appended in place) are threaded through unchanged.
    """
    cfg = state.cfg
    w_orig = state.w_orig
    h_orig = state.h_orig
    detector = state.detector
    profile_stages = state.profile_stages
    seq_stage_totals = state.seq_stage_totals
    post_boxes = fctx.post_boxes
    post_scores = fctx.post_scores
    post_classes = fctx.post_classes
    geometry_suspect_mask = fctx.geometry_suspect_mask
    t_output_slicing_start = None
    if profile_stages:
        torch.cuda.synchronize()
        t_output_slicing_start = time.perf_counter()
    with _record_profile_scope("post.output_slicing"):
        fused_boxes = post_boxes[:n_post]
        fused_scores = post_scores[:n_post]
        fused_classes = post_classes[:n_post]
        geometry_suspect_mask = geometry_suspect_mask[:n_post]
    if (
        profile_stages
        and current_stage_sample_active
        and t_output_slicing_start is not None
    ):
        torch.cuda.synchronize()
        seq_stage_totals["post_output_slicing"] += (
            time.perf_counter() - t_output_slicing_start
        ) * 1000
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

    t_quality_scale_start = None
    if profile_stages:
        torch.cuda.synchronize()
        t_quality_scale_start = time.perf_counter()
    with _record_profile_scope("post.quality_scale"):
        if (
            cfg.detection_quality_scaling
            and n_post > 0
            and not getattr(detector.tracker, "is_cuda", False)
        ):
            quality_factors = _compute_detection_quality_batch(
                fused_boxes,
                w_orig,
                h_orig,
                w_aspect=cfg.detection_quality_w_aspect,
                w_center=cfg.detection_quality_w_center,
                w_area=cfg.detection_quality_w_area,
            )
            fused_scores = fused_scores * quality_factors
    if (
        profile_stages
        and current_stage_sample_active
        and t_quality_scale_start is not None
    ):
        torch.cuda.synchronize()
        seq_stage_totals["post_quality_scale"] += (
            time.perf_counter() - t_quality_scale_start
        ) * 1000
    if profile_stages and current_stage_sample_active:
        _seg_ev = torch.cuda.Event(enable_timing=True)
        _seg_ev.record(torch.cuda.current_stream())
        post_seg_events.append(("post_seg_slice_quality", _seg_ev))
    after_filter_count = int(n_post)
    after_nms_count = int(n_post)
    return (
        fused_boxes,
        fused_scores,
        fused_classes,
        geometry_suspect_mask,
        aligned_keypoints,
        after_filter_count,
        after_nms_count,
    )


@dataclasses.dataclass(frozen=True)
class PreparedDetection:
    """One frame's detector output, produced on the detection stream.

    The tensors are clones, rather than views into the whole-graph callable's
    static output storage.  This is the ownership boundary that permits the
    next graph replay while the tracker is still consuming the previous frame.
    """

    frame_id: int
    pool: Any
    frame_gpu: torch.Tensor
    fused_boxes: torch.Tensor
    fused_scores: torch.Tensor
    fused_classes: torch.Tensor
    is_tiled: bool
    source_keypoints: "torch.Tensor | None"
    ready_event: "torch.cuda.Event"
    latency_started_at: float


def _launch_double_buffer_detect(
    state: EvalPipeline,
    *,
    frame_id: int,
    pool: Any,
    frame_gpu: torch.Tensor,
    input_ready: "torch.cuda.Event",
    ready_event: "torch.cuda.Event",
    latency_started_at: float,
) -> PreparedDetection:
    """Queue detect(frame_id) on the side stream without blocking the host.

    The main stream waits only on ``ready_event`` when it reaches this frame.
    Until then it is free to run GMC/tracker/materialization for frame N-1.
    There is intentionally one outstanding detection: that preserves detector
    graph ownership while still providing the desired double-buffer overlap.

    ``latency_started_at`` is captured by the caller *before* JPEG decode so the
    reported per-frame latency is genuinely end-to-end (decode→detect→track→out).
    """

    # Per-parity stream/event swap for explicit probe mode.
    # _run_frame also does this swap, but _launch_double_buffer_detect runs
    # before _run_frame for the same frame_id — we must assign the correct
    # parity's streams before _run_detect enqueues work.
    #
    # IMPORTANT: _run_detect is called on double_buffer_stream (explicit probe's
    # per-parity stream_detect is temporarily nulled).  The whole-detect CUDA
    # graph is captured once on the first call's stream — switching to a
    # different per-parity stream on later frames would replay the graph on a
    # stream whose kernels differ from the capture stream, making detect_done
    # fire at the wrong point.  We record detect_done on double_buffer_stream
    # instead and fence stream_post here so _run_frame's postprocessing is
    # correctly ordered.
    _p_db = -1
    if _explicit_stream_probe_enabled() and state._pp_streams[0]:
        _p_db = frame_id % 2
        state.stream_post = state._pp_streams[_p_db]["post"]
        # Save and null per-parity detect stream so _run_detect falls back to
        # the legacy path (TRT on current = double_buffer_stream).
        _saved_detect = state.stream_detect
        _saved_detect_event = state.stream_detect_event
        state.stream_detect = None
        state.stream_detect_event = None

    stream = state.double_buffer_stream
    if stream is None:
        raise RuntimeError("double-buffer launch requested without a CUDA stream")
    # ``frame_gpu`` is produced on the caller's stream.  Establish an explicit
    # producer→detector dependency before the side stream reads it; PyTorch does
    # not infer ordering merely because both streams reference the same tensor.
    main_stream = torch.cuda.current_stream()
    input_ready.record(main_stream)
    with torch.cuda.stream(stream):
        stream.wait_event(input_ready)
        frame_gpu.record_stream(stream)
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
            nv12_direct_from_hwc=state.nv12_direct_from_hwc,
            detect_fn=state.detect_fn,
            detector_box_format=state.detector_box_format,
            synchronize=False,
        )
        # Record detect_done on double_buffer_stream so _run_frame's
        # stream_post can be fenced after the detection output is ready.
        if _p_db >= 0:
            _detect_done_ev = state._pp_detect_done[_p_db]
            _detect_done_ev.record(stream)
            _sp = state.stream_post
            if _sp is not None:
                _sp.wait_event(_detect_done_ev)
            # Restore per-parity detect state for _run_frame's parity swap.
            state.stream_detect = _saved_detect
            state.stream_detect_event = _saved_detect_event
        # Whole-graph replays return views into reusable static buffers.  Clone
        # every tensor crossing the frame boundary before another replay can
        # overwrite those buffers.
        fused_boxes = fused_boxes.clone()
        fused_scores = fused_scores.clone()
        fused_classes = fused_classes.clone()
        source_keypoints = (
            source_keypoints.clone() if source_keypoints is not None else None
        )
        ready_event.record()

    for tensor in (fused_boxes, fused_scores, fused_classes, source_keypoints):
        if tensor is not None:
            tensor.record_stream(main_stream)
    return PreparedDetection(
        frame_id=frame_id,
        pool=pool,
        frame_gpu=frame_gpu,
        fused_boxes=fused_boxes,
        fused_scores=fused_scores,
        fused_classes=fused_classes,
        is_tiled=is_tiled,
        source_keypoints=source_keypoints,
        ready_event=ready_event,
        latency_started_at=latency_started_at,
    )


def _run_detect(
    state: EvalPipeline,
    *,
    pool: Any,
    frame_gpu: torch.Tensor,
    nv12_direct_from_hwc: bool,
    detect_fn: Any,
    detector_box_format: Any,
    synchronize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, "torch.Tensor | None"]:
    """Ingest+preprocess the frame then run YOLO detection (extracted).

    Copies the frame into the pool's working buffer (NV12-direct, or RGB →
    preprocess → optional NV12) and dispatches the configured ``detect_fn`` (tiled
    vs native, chosen once at setup; opaque here). Returns ``(fused_boxes,
    fused_scores, fused_classes, is_tiled, source_keypoints)`` in original-image
    coordinates. This is the non-workbench detect path only.
    """
    cfg = state.cfg
    detector = state.detector
    h_orig = state.h_orig
    w_orig = state.w_orig
    seq_stage_totals = state.seq_stage_totals
    time_stage = state.time_stage
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
                pool.frame_buffer.copy_(frame_gpu.permute(2, 0, 1).float() / 255.0),
                apply_frame_preprocess(
                    pool.frame_buffer,
                    cfg.preprocess_modes,
                    cfg.gamma,
                    cfg.gamma_luma_threshold,
                    cfg.contrast,
                ),
                pool.mark_rgb_current(),
                (
                    pool.frame_buffer_nv12.copy_(rgb_chw_to_nv12_gpu(pool.frame_buffer))
                    if pool.use_nv12
                    else None
                ),
            )
        ),
        sync_cuda=synchronize,
    )

    # Serialise the GPU pipeline between ingest/preprocess and YOLO detection.
    # The NVJPEG/DALI decode hardware engine and the TRT enqueue both operate
    # outside the default CUDA stream, so ingest -> detect carries no implicit
    # device barrier.  Without an explicit synchronize the YOLO engine can read
    # partially-stale pool-buffer data, producing run-to-run output drift.
    # Profiling mode masks this by synchronizing at every stage boundary; this
    # is the minimal single-barrier fix.
    _barrier_mode = _detect_barrier_mode()
    _explicit_probe = _explicit_stream_probe_enabled()
    if os.environ.get("SACCADE_STREAM_DEBUG", "") in ("1", "true", "yes"):
        _cs = torch.cuda.current_stream()
        _ds = torch.cuda.default_stream()
        _s_detect_str = (
            f" S_detect={state.stream_detect.cuda_stream:#x}"
            if state.stream_detect is not None
            else ""
        )
        _s_post_str = (
            f" S_post={state.stream_post.cuda_stream:#x}"
            if state.stream_post is not None
            else ""
        )
        print(
            f"[STREAM] _run_detect: current={_cs.cuda_stream:#x} default={_ds.cuda_stream:#x}{_s_detect_str}{_s_post_str} trt_enqueue_on_current=True barrier={_barrier_mode} explicit_probe={_explicit_probe}"
        )
    _t_ds1_start = time.perf_counter() if state._frame_stage_times is not None else 0.0
    if not synchronize or _barrier_mode == "event":
        pass
    else:
        torch.cuda.synchronize()
    if state._frame_stage_times is not None:
        state._frame_stage_times["detect_ingest_barrier"] = round(
            (time.perf_counter() - _t_ds1_start) * 1000, 6
        )

    _t_trt_start = time.perf_counter() if state._frame_stage_times is not None else 0.0
    if _explicit_probe and state.stream_detect is not None:
        _trt_sync = False
        _t_explicit_dispatch_start = (
            time.perf_counter() if state._frame_stage_times is not None else 0.0
        )
        _s_detect = state.stream_detect
        _s_post = state.stream_post
        _prev_stream = torch.cuda.current_stream()
        if _prev_stream.cuda_stream != _s_detect.cuda_stream:
            torch.cuda.set_stream(_s_detect)
        _t_stream_switch = (
            time.perf_counter() if state._frame_stage_times is not None else 0.0
        )
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
            sync_cuda=_trt_sync,
        )
        _t_trt_done = (
            time.perf_counter() if state._frame_stage_times is not None else 0.0
        )
        state.stream_detect_event.record(_s_detect)
        _s_post.wait_event(state.stream_detect_event)
        _t_event_fence = (
            time.perf_counter() if state._frame_stage_times is not None else 0.0
        )
        _default = torch.cuda.default_stream()
        torch.cuda.set_stream(_default)
        _default.wait_event(state.stream_detect_event)
        if state._frame_stage_times is not None:
            _dispatch_total = (_t_event_fence - _t_explicit_dispatch_start) * 1000
            state._frame_stage_times["explicit_stream_dispatch_ms"] = round(
                _dispatch_total, 6
            )
            state._frame_stage_times["trt_enqueue_host_ms"] = round(
                (_t_trt_done - _t_stream_switch) * 1000, 6
            )
            state._frame_stage_times["event_record_ms"] = round(
                (_t_event_fence - _t_trt_done) * 1000, 6
            )
    else:
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
            sync_cuda=synchronize,
        )
    if state._frame_stage_times is not None:
        state._frame_stage_times["detect_trt_enqueue"] = round(
            (time.perf_counter() - _t_trt_start) * 1000, 6
        )

    # Ensure the TRT output is fully written before the postprocess stage
    # reads the raw detection tensors (views into the shared output buffer).
    # In whole_graph mode the TRT enqueue and the postprocess graphs both launch
    # from the current stream, so this ordering is likely already implicit; the
    # no_postproc/event modes drop the redundant full barrier (gated on the same
    # N>=6 determinism check).
    _t_ds2_start = time.perf_counter() if state._frame_stage_times is not None else 0.0
    if not _explicit_probe:
        if not synchronize or _barrier_mode in ("no_postproc", "event"):
            pass
        else:
            torch.cuda.synchronize()
    if state._frame_stage_times is not None:
        state._frame_stage_times["detect_postproc_barrier"] = round(
            (time.perf_counter() - _t_ds2_start) * 1000, 6
        )
    if state._frame_stage_times is not None:
        _sd = state.stream_detect
        _sp = state.stream_post
        state._frame_stage_times["detect_stream_handle"] = (
            _sd.cuda_stream if _sd is not None else 0
        )
        state._frame_stage_times["trt_enqueue_stream_handle"] = (
            _sd.cuda_stream if _sd is not None else 0
        )

    return fused_boxes, fused_scores, fused_classes, is_tiled, source_keypoints


def _stash_crop_ring(state: "EvalPipeline", track_results: Any, frame_id: int) -> None:
    pp = state.perception_pipeline
    if pp is None or not hasattr(pp, "crop_ring_enabled") or not pp.crop_ring_enabled():
        return
    live_evfifo = getattr(state, "live_evfifo", None)
    if (
        live_evfifo is not None
        and frame_id % max(1, getattr(live_evfifo, "stride", 5)) != 0
    ):
        return
    count_raw = track_results.get("count")
    if count_raw is None:
        return
    count_i = int(count_raw.item() if hasattr(count_raw, "item") else count_raw)
    if count_i == 0:
        return
    ids_t = track_results["ids"]
    ids_cpu = ids_t[:count_i].cpu().tolist()
    boxes_xyxy = track_results["boxes"][:count_i].detach().cpu()
    from .helpers import front_occlusion_mask_xyxy as _front_occlusion_mask_xyxy

    occluded = _front_occlusion_mask_xyxy(
        boxes_xyxy, getattr(state.cfg, "appearance_occlusion_cov", 0.4)
    )
    clean_flags = (~occluded).to(torch.int32).tolist()
    _hwc_cache = getattr(state, "frame_hwc_cache", None)
    if _hwc_cache is not None and _hwc_cache[0] == frame_id:
        frame_hwc = _hwc_cache[1]
    else:
        frame_hwc = state.pool.as_rgb_chw().permute(1, 2, 0).contiguous()
        state.frame_hwc_cache = (frame_id, frame_hwc)
    boxes_dev = boxes_xyxy.to("cuda", torch.float32).contiguous()
    uids_np = np.asarray(ids_cpu, dtype=np.uint64)
    frames_np = np.full(count_i, frame_id, dtype=np.int32)
    clean_np = np.asarray(clean_flags, dtype=bool)
    pp.stash_crops(
        uids_np.ctypes.data,
        frames_np.ctypes.data,
        frame_hwc.data_ptr(),
        state.h_orig,
        state.w_orig,
        boxes_dev.data_ptr(),
        count_i,
        clean_np.ctypes.data,
        torch.cuda.current_stream().cuda_stream,
    )
    if live_evfifo is not None:
        live_evfifo.record_box_uids(frame_id, ids_cpu, boxes_xyxy)


def _run_emit(
    state: EvalPipeline,
    *,
    track_results: Any,
    tracker_result_buffers: Any,
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    geometry_suspect_mask: torch.Tensor,
    embeddings: "torch.Tensor | None",
    gmc_warp: "torch.Tensor | None",
    frame_birth_events: list,
    frame_id: int,
    prev_track_ids: set,
    track_results_on_host: bool = False,
) -> tuple[set, list]:
    """Emit MOT lines for one frame's tracker result (extracted from run_eval).

    Either submits the relink-write pipeline to the background executor (when
    pipeline_relink + semantic work is active) or runs the synchronous emit path
    (fast emit, or full prepare/resolve/emit + lifecycle prune + side-effect
    finalize). Returns ``(prev_track_ids, lines)``: the background path passes
    prev_track_ids through and stashes the new future/birth-events on ``state``
    with no lines this frame; the synchronous path returns the updated
    prev_track_ids, leaves the bg cells on ``state`` untouched, and returns the
    emitted lines for the caller to extend into the run-level accumulator. The
    bg future/birth-events live on ``state`` and are drained by the frame loop.
    """
    cfg = state.cfg
    detector = state.detector
    seq = state.seq
    w_orig = state.w_orig
    h_orig = state.h_orig
    profile_stages = state.profile_stages
    seq_stage_totals = state.seq_stage_totals
    relinker = state.relinker
    id_stability_filter = state.id_stability_filter

    if (
        relinker is not None
        and hasattr(relinker, "feed_frame_embeddings")
        and embeddings is not None
        and embeddings.numel() > 0
    ):
        try:
            from .helpers import FLOW_TIMING, flow_add, flow_now, flow_report

            ids_t = track_results["ids"]
            count_raw = track_results["count"]
            count_i = int(count_raw.item() if hasattr(count_raw, "item") else count_raw)
            if count_i > 0:
                _t0 = flow_now() if FLOW_TIMING else 0.0
                ids_cpu = ids_t[:count_i].cpu().tolist()
                scores_cpu = track_results["scores"][:count_i].cpu().tolist()
                emb_dim = embeddings.shape[1]
                # embeddings is detection-indexed (fused_boxes rows, zero rows
                # for un-budgeted dets); gather to track order via det_idx.
                # Tracks with no matched detection this frame get a zero row
                # (skipped by the C++ side).
                det_idx = track_results.get("det_idx")
                emb_src = embeddings.detach()
                emb_rows = torch.zeros((count_i, emb_dim), dtype=emb_src.dtype)
                if det_idx is not None:
                    di = det_idx[:count_i].long().cpu()
                    ok = (di >= 0) & (di < emb_src.shape[0])
                    if ok.any():
                        # Gather on-device, then D2H only the matched rows —
                        # not the full detection-indexed tensor.
                        sel = di[ok].to(emb_src.device, non_blocking=True)
                        emb_rows[ok] = emb_src.index_select(0, sel).cpu()
                else:
                    n_copy = min(count_i, emb_src.shape[0])
                    emb_rows[:n_copy] = emb_src[:n_copy].cpu()
                # Head/bank samples must be visually clean (Python-reference
                # parity): front-occlusion gate over this frame's track boxes.
                boxes_xyxy = track_results["boxes"][:count_i].detach().cpu()
                occluded = _front_occlusion_mask_xyxy(
                    boxes_xyxy, getattr(cfg, "appearance_occlusion_cov", 0.4)
                )
                clean_flags = (~occluded).to(torch.int32).tolist()
                if FLOW_TIMING:
                    flow_add("ho_gather", flow_now() - _t0)
                    _t0 = flow_now()
                if hasattr(relinker, "feed_frame_embeddings_arr"):
                    # Buffer-protocol fast path: no per-element Python float
                    # round-trip through the pybind std::vector caster.
                    relinker.feed_frame_embeddings_arr(
                        np.ascontiguousarray(emb_rows.numpy()),
                        emb_dim,
                        frame_id,
                        np.asarray(ids_cpu, dtype=np.int32),
                        np.asarray(scores_cpu, dtype=np.float32),
                        (~occluded).numpy().astype(np.int32),
                    )
                else:
                    relinker.feed_frame_embeddings(
                        emb_rows.numpy().ravel().tolist(),
                        emb_dim,
                        frame_id,
                        ids_cpu,
                        scores_cpu,
                        clean_flags,
                    )
                if FLOW_TIMING:
                    flow_add("ho_feed", flow_now() - _t0)
                    _t0 = flow_now()
                if state.live_evfifo is not None:
                    state.live_evfifo.observe_frame(
                        frame_id, ids_cpu, boxes_xyxy, emb_rows
                    )
                if FLOW_TIMING:
                    flow_add("ho_observe", flow_now() - _t0)
                    _t0 = flow_now()
                # Stash each confirmed track's crop into the re-query ring,
                # keyed by the same track id fed above (= handover archive tid).
                # Align with evfifo stride: the planner selects 1 every N frames,
                # so stashing every frame wastes GPU copy bandwidth (24 MB/frame
                # HWC permute + contiguous at 1080p).
                pp = state.perception_pipeline
                if (
                    pp is not None
                    and hasattr(pp, "crop_ring_enabled")
                    and pp.crop_ring_enabled()
                    and (
                        state.live_evfifo is None
                        or frame_id % max(1, getattr(state.live_evfifo, "stride", 5))
                        == 0
                    )
                ):
                    # Reuse the HWC frame copy the ReID stage built this frame
                    # (a full-frame permute+contiguous is ~24MB at 1080p).
                    _hwc_cache = getattr(state, "frame_hwc_cache", None)
                    if _hwc_cache is not None and _hwc_cache[0] == frame_id:
                        frame_hwc = _hwc_cache[1]
                    else:
                        frame_hwc = (
                            state.pool.as_rgb_chw().permute(1, 2, 0).contiguous()
                        )
                        state.frame_hwc_cache = (frame_id, frame_hwc)
                    boxes_dev = (
                        track_results["boxes"][:count_i]
                        .to("cuda", torch.float32)
                        .contiguous()
                    )
                    uids_np = np.asarray(ids_cpu, dtype=np.uint64)
                    frames_np = np.full(count_i, frame_id, dtype=np.int32)
                    clean_np = np.asarray(clean_flags, dtype=bool)
                    if FLOW_TIMING:
                        flow_add("ho_hwc_boxes", flow_now() - _t0)
                        _t0 = flow_now()
                    pp.stash_crops(
                        uids_np.ctypes.data,
                        frames_np.ctypes.data,
                        frame_hwc.data_ptr(),
                        state.h_orig,
                        state.w_orig,
                        boxes_dev.data_ptr(),
                        count_i,
                        clean_np.ctypes.data,
                        torch.cuda.current_stream().cuda_stream,
                    )
                    if FLOW_TIMING:
                        flow_add("ho_stash", flow_now() - _t0)
            if FLOW_TIMING and frame_id % 100 == 0:
                print(f"[flow t] f{frame_id}: {flow_report()}")
        except Exception as exc:
            print(f"[online_ho] ERROR frame={frame_id}: {exc}")
            import traceback

            traceback.print_exc()

    primary_appearance_bank = state.primary_appearance_bank
    output_appearance_bank = state.output_appearance_bank
    dynamic_reid = state.dynamic_reid
    lifecycle_merger = state.lifecycle_merger
    identity_resolver = state.identity_resolver
    global_id_mapper = state.global_id_mapper
    _rw_executor = state.rw_executor
    record_stage_sample = state.record_stage_sample
    _bg_relink_write = state.bg_relink_write
    _collect_output_metadata = state.collect_output_metadata
    _annotate_birth_events = state.annotate_birth_events
    _lines_out: list[str] = []
    _bg_future = state.bg_future
    _bg_birth_events = state.bg_birth_events
    _needs_emit_pipeline = (
        (relinker is not None and state.live_evfifo is None)
        or id_stability_filter is not None
        or primary_appearance_bank is not None
        or dynamic_reid is not None
    )
    if (
        cfg.pipeline_relink
        and not getattr(cfg, "workbench", False)
        and _needs_emit_pipeline
    ):
        # Pre-materialize: host_track_batch + motion snapshots (need main CUDA stream)
        # then D2H-copy GPU tensors so background thread stays off CUDA streams.
        _pm_host_batch = _prepare_host_track_batch(
            track_results,
            tracker_result_buffers,
            dynamic_reid_enabled=dynamic_reid is not None,
            person_class=cfg.person_class,
        )
        _pm_motion_cids: list[int] = []
        _pm_motion_snaps = None
        if relinker:
            _pm_motion_cids = relinker.motion_candidate_ids(frame_id)
            if _pm_motion_cids:
                _pm_motion_snaps = detector.tracker.get_motion_snapshots_for_track_ids(
                    _pm_motion_cids
                )
        _pm_host_batch_cpu = dataclasses.replace(
            _pm_host_batch, boxes_gpu=_pm_host_batch.boxes_gpu.cpu()
        )
        _pm_fused_boxes = fused_boxes.cpu()
        _pm_fused_scores = fused_scores.cpu()
        _pm_geom_mask = geometry_suspect_mask.cpu()
        _pm_embeddings = embeddings.cpu() if embeddings is not None else None
        _pm_gmc = (
            gmc_warp.cpu()
            if (gmc_warp is not None and gmc_warp.device.type == "cuda")
            else gmc_warp
        )
        if _rw_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            _rw_executor = ThreadPoolExecutor(max_workers=1)
            state.rw_executor = _rw_executor
        _bg_future = _rw_executor.submit(  # type: ignore[union-attr]
            _bg_relink_write,
            frame_id,
            track_results,
            _pm_host_batch_cpu,
            _pm_fused_boxes,
            _pm_fused_scores,
            _pm_geom_mask,
            _pm_embeddings,
            _pm_gmc,
            _pm_motion_cids,
            _pm_motion_snaps,
            prev_track_ids,
        )
        _bg_birth_events = frame_birth_events
    else:
        if profile_stages:
            torch.cuda.synchronize()
            t_relink_write_start = time.perf_counter()
        if relinker:
            motion_candidate_ids = relinker.motion_candidate_ids(frame_id)
            if motion_candidate_ids:
                relinker.update_motion_snapshots(
                    detector.tracker.get_motion_snapshots_for_track_ids(
                        motion_candidate_ids
                    ),
                    frame_id,
                )

        _use_fast_emit = (
            not _needs_emit_pipeline
            and cfg.reid_mode in ("off", "tracker", "extract")
            and not bool(cfg.kwargs.get("id_stability_filter", False))
        )
        if _use_fast_emit:
            # When the caller already materialized track_results to host
            # (DB flush via _flush_db_tracker_out, or _run_materialize),
            # use _fast_emit_mot_lines directly on the host data. This avoids
            # emit_tracks_unified's redundant synchronous D2H via
            # compact_output_to_host, which re-reads d_res_* from GPU after
            # the data is already on host via the pinned copy.
            _can_use_host_fast_emit = track_results_on_host or not hasattr(
                detector.tracker.tracker, "compact_output_to_host"
            )
            if _can_use_host_fast_emit:
                frame_result_lines = _fast_emit_mot_lines(
                    track_results=track_results,
                    global_id_mapper=global_id_mapper,
                    seq=seq,
                    frame_id=frame_id,
                    frame_w=w_orig,
                    frame_h=h_orig,
                )
                curr_track_ids = set(int(x) for x in track_results["ids"].tolist())
            else:
                import saccade_tracking_ext

                count, boxes_list, scores_list, ids_list, _ = (
                    saccade_tracking_ext.emit_tracks_unified(
                        detector.tracker.tracker,
                        relinker if relinker else None,
                        torch.cuda.current_stream().cuda_stream,
                        frame_id,
                        w_orig,
                        h_orig,
                        detector.tracker.max_objects,
                        embeddings if cfg.use_tracker_reid else None,
                    )
                )
                frame_result_lines = []
                for i in range(count):
                    gid = global_id_mapper.map(seq, int(ids_list[i]))
                    x1, y1 = float(boxes_list[i * 4]), float(boxes_list[i * 4 + 1])
                    x2, y2 = float(boxes_list[i * 4 + 2]), float(boxes_list[i * 4 + 3])
                    s = float(scores_list[i])
                    frame_result_lines.append(
                        f"{frame_id},{gid},{x1:.2f},{y1:.2f},"
                        f"{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},-1,-1,-1"
                    )
                curr_track_ids = set(ids_list[:count])
        else:
            from .helpers import FLOW_TIMING, flow_add, flow_now

            _t0 = flow_now() if FLOW_TIMING else 0.0
            host_track_batch = _prepare_host_track_batch(
                track_results,
                tracker_result_buffers,
                dynamic_reid_enabled=dynamic_reid is not None,
                person_class=cfg.person_class,
            )
            if FLOW_TIMING:
                flow_add("emit_host_batch", flow_now() - _t0)
                _t0 = flow_now()

            prepared_candidates = _prepare_track_candidates(
                frame_id=frame_id,
                track_results=track_results,
                host_batch=host_track_batch,
                person_class=cfg.person_class,
                track_person_only=cfg.track_person_only,
                geometry_suspect_support=cfg.geometry_suspect_support,
                geometry_suspect_support_score=cfg.geometry_suspect_support_score,
                id_stability_filter=id_stability_filter,
                embeddings=embeddings,
                fused_boxes=fused_boxes,
                fused_scores=fused_scores,
                geometry_suspect_mask=geometry_suspect_mask,
                primary_appearance_bank=primary_appearance_bank,
                frame_w=w_orig,
                frame_h=h_orig,
                bank_quality_v2=cfg.bank_quality_v2,
                bank_quality_w_det=cfg.bank_quality_w_det,
                bank_quality_w_iou=cfg.bank_quality_w_iou,
                bank_quality_w_aspect=cfg.bank_quality_w_aspect,
                bank_quality_w_center=cfg.bank_quality_w_center,
                bank_quality_w_area=cfg.bank_quality_w_area,
            )
            if FLOW_TIMING:
                flow_add("emit_prepare_cand", flow_now() - _t0)
                _t0 = flow_now()
            resolved_tracks = _resolve_frame_tracks(
                frame_id=frame_id,
                frame_w=w_orig,
                frame_h=h_orig,
                prepared_candidates=prepared_candidates,
                lifecycle_merger=lifecycle_merger,
                identity_resolver=identity_resolver,
            )
            if FLOW_TIMING:
                flow_add("emit_resolve", flow_now() - _t0)
                _t0 = flow_now()
            frame_result_lines = _emit_resolved_tracks(
                seq=seq,
                frame_id=frame_id,
                frame_w=w_orig,
                frame_h=h_orig,
                resolved_tracks=resolved_tracks,
                global_id_mapper=global_id_mapper,
                output_appearance_bank=output_appearance_bank,
            )
            if FLOW_TIMING:
                flow_add("emit_lines", flow_now() - _t0)
            det_idx_to_local_id = {
                int(det_idx): int(local_id)
                for local_id, det_idx in zip(
                    host_track_batch.ids,
                    host_track_batch.det_idx or [],
                )
                if int(det_idx) >= 0
            }
            output_by_local = _collect_output_metadata(resolved_tracks)
            _annotate_birth_events(
                frame_birth_events,
                _det_idx_to_local_id=det_idx_to_local_id,
                _output_by_local=output_by_local,
            )
            curr_track_ids = set(host_track_batch.ids)
        _lines_out = frame_result_lines
        if state.double_buffer_stream is None:
            _stash_crop_ring(state, track_results, frame_id)
        lifecycle_merger.prune(frame_id)
        prev_track_ids = _finalize_frame_side_effects(
            curr_track_ids=curr_track_ids,
            prev_track_ids=prev_track_ids,
            relinker=relinker,
            semantic_bank_inject=cfg.semantic_bank_inject,
            primary_appearance_bank=primary_appearance_bank,
            dynamic_reid=dynamic_reid,
            person_observations=(
                host_track_batch.person_observations if not _use_fast_emit else []
            ),
            gmc_warp=gmc_warp,
            gmc_enabled=cfg.gmc_enabled,
        )
        if hasattr(detector.tracker, "observe_track_observations"):
            detector.tracker.observe_track_observations(
                host_track_batch.person_observations if not _use_fast_emit else {},
                gmc=gmc_warp if cfg.gmc_enabled else None,
            )
        if relinker is not None and hasattr(relinker, "prune_and_archive"):
            try:
                relinker.prune_and_archive(list(curr_track_ids), frame_id)
            except Exception as exc:
                print(f"[online_ho] prune ERROR frame={frame_id}: {exc}")
        if profile_stages:
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t_relink_write_start) * 1000
            seq_stage_totals["relink_write"] += elapsed_ms
            record_stage_sample("relink_write", elapsed_ms)
    state.bg_future = _bg_future
    state.bg_birth_events = _bg_birth_events
    return prev_track_ids, _lines_out


def _run_detection_filters(
    state: EvalPipeline,
    *,
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    fused_classes: torch.Tensor,
    geometry_suspect_mask: torch.Tensor,
    aligned_keypoints: "torch.Tensor | None",
    after_merge_count: int,
    frame_score_floor: float,
    base_score_floor: float,
    frame_id: int,
    current_stage_sample_active: bool,
    post_seg_events: list,
    debug_dump_active: bool,
    debug_stage_dump_rows: list,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | None", int
]:
    """Detection tail-filters for one frame (extracted from run_eval).

    The trap-free first half of the post-merge tail: per-frame score-floor filter,
    external FP filter, FP-hard mask-in-place, duplicate suppression, and per-frame
    detection cap. Each is an independent ``if cfg.<feature>:`` step mutating the
    fused detections (and keeping geometry_suspect_mask / keypoints aligned), with
    debug dumps and segment-event markers preserved. Returns the (possibly filtered)
    ``(fused_boxes, fused_scores, fused_classes, geometry_suspect_mask,
    aligned_keypoints, after_merge_count)``; after_merge_count passes through
    unchanged when no filter fires. The birth gates + tracker config that follow
    stay inline (they carry cross-frame state).
    """
    cfg = state.cfg
    seq = state.seq
    w_orig = state.w_orig
    h_orig = state.h_orig
    profile_stages = state.profile_stages
    seq_stage_totals = state.seq_stage_totals
    external_fp_rule_config = state.external_fp_rule_config
    external_fp_logistic_model = state.external_fp_logistic_model
    t_tail_filtering_start = None
    if profile_stages:
        torch.cuda.synchronize()
        t_tail_filtering_start = time.perf_counter()
    with _record_profile_scope("post.tail_filtering"):
        if frame_score_floor > base_score_floor and fused_scores.numel() > 0:
            floor_keep = fused_scores > frame_score_floor
            fused_boxes = fused_boxes[floor_keep]
            fused_scores = fused_scores[floor_keep]
            fused_classes = fused_classes[floor_keep]
            geometry_suspect_mask = geometry_suspect_mask[floor_keep]
            if aligned_keypoints is not None:
                aligned_keypoints = aligned_keypoints[floor_keep]
            after_merge_count = int(fused_scores.numel())
    if (
        profile_stages
        and current_stage_sample_active
        and t_tail_filtering_start is not None
    ):
        torch.cuda.synchronize()
        seq_stage_totals["post_tail_filtering"] += (
            time.perf_counter() - t_tail_filtering_start
        ) * 1000
    if profile_stages and current_stage_sample_active:
        _seg_ev = torch.cuda.Event(enable_timing=True)
        _seg_ev.record(torch.cuda.current_stream())
        post_seg_events.append(("post_seg_tail_filter", _seg_ev))
    if debug_dump_active:
        _append_stage_dump_rows(
            debug_stage_dump_rows,
            seq=seq,
            frame_id=frame_id,
            stage="post_merge",
            boxes=fused_boxes,
            scores=fused_scores,
            classes=fused_classes,
        )

    if cfg.external_fp_filter_mode != "off" and fused_scores.numel() > 0:
        fused_boxes, fused_scores, fused_classes = _apply_external_fp_filter(
            fused_boxes,
            fused_scores,
            fused_classes,
            image_width=w_orig,
            image_height=h_orig,
            mode=cfg.external_fp_filter_mode,
            rule_config=external_fp_rule_config,
            logistic_model=external_fp_logistic_model,
            logistic_threshold=cfg.external_fp_logistic_threshold,
            max_score=cfg.external_fp_max_score,
            penalty=cfg.external_fp_penalty,
            min_score=frame_score_floor,
            softmax_min_scale=cfg.external_fp_softmax_min_scale,
        )
        after_merge_count = int(fused_scores.numel())
        if debug_dump_active:
            _append_stage_dump_rows(
                debug_stage_dump_rows,
                seq=seq,
                frame_id=frame_id,
                stage=f"external_fp_{cfg.external_fp_filter_mode}",
                boxes=fused_boxes,
                scores=fused_scores,
                classes=fused_classes,
            )

    # === FP hard filter ===
    # Removes extremely suspicious low-score large-area detections
    # that are likely false positives based on FP analysis.
    if cfg.fp_hard_filter_enabled and fused_scores.numel() > 0:
        # Mask-in-place (fixed shape, sync-free): set rejected scores
        # below all tracker thresholds rather than compacting. The GPU
        # tracker's score gate drops them without a host .nonzero()
        # sync, and geometry_suspect_mask / keypoints stay aligned.
        _fp_reject = _fp_hard_reject_mask(
            fused_boxes,
            fused_scores,
            min_score=cfg.fp_hard_filter_min_score,
            max_suspicious_area=cfg.fp_hard_filter_max_suspicious_area,
            max_suspicious_score=cfg.fp_hard_filter_max_suspicious_score,
        )
        fused_scores = fused_scores.masked_fill(_fp_reject, _FP_HARD_REJECT_SCORE)
        after_merge_count = int(fused_scores.numel())
        if debug_dump_active:
            _append_stage_dump_rows(
                debug_stage_dump_rows,
                seq=seq,
                frame_id=frame_id,
                stage="fp_hard_filter",
                boxes=fused_boxes,
                scores=fused_scores,
                classes=fused_classes,
            )
    if profile_stages and current_stage_sample_active:
        _seg_ev = torch.cuda.Event(enable_timing=True)
        _seg_ev.record(torch.cuda.current_stream())
        post_seg_events.append(("post_seg_fp_hard", _seg_ev))

    # === Duplicate suppression ===
    # Remove near-duplicate detections within the same frame before
    # birth gates / multi_birth / detection cap. This eliminates detector
    # artifacts where the same person is detected at slightly different
    # positions with different scores.
    if cfg.duplicate_suppression and fused_scores.numel() > 0:
        dup_keep = _suppress_duplicate_detections(
            fused_boxes,
            fused_scores,
            iou_threshold=cfg.duplicate_suppression_iou_threshold,
            min_score_ratio=cfg.duplicate_suppression_min_score_ratio,
        )
        if not dup_keep.all():
            fused_boxes = fused_boxes[dup_keep]
            fused_scores = fused_scores[dup_keep]
            fused_classes = fused_classes[dup_keep]
            geometry_suspect_mask = geometry_suspect_mask[dup_keep]
            if aligned_keypoints is not None:
                aligned_keypoints = aligned_keypoints[dup_keep]
            after_merge_count = int(fused_scores.numel())
            if debug_dump_active:
                _append_stage_dump_rows(
                    debug_stage_dump_rows,
                    seq=seq,
                    frame_id=frame_id,
                    stage="duplicate_suppression",
                    boxes=fused_boxes,
                    scores=fused_scores,
                    classes=fused_classes,
                )

            # === Per-frame detection cap ===
    # Cap detections per frame to prevent overwhelming association.
    # Uses FP-filter-aware ranking to preferentially keep high-score,
    # appropriately-sized detections while filtering suspicious large boxes.
    if cfg.per_frame_detection_cap > 0 and fused_scores.numel() > 0:
        quality_factors_for_cap = None
        if cfg.detection_quality_scaling and fused_scores.numel() > 0:
            quality_factors_for_cap = _compute_detection_quality_batch(
                fused_boxes,
                w_orig,
                h_orig,
                w_aspect=cfg.detection_quality_w_aspect,
                w_center=cfg.detection_quality_w_center,
                w_area=cfg.detection_quality_w_area,
            )

        # Compute adaptive cap if enabled
        max_det = cfg.per_frame_detection_cap
        if cfg.adaptive_detection_cap:
            max_det = _compute_adaptive_cap(
                fused_boxes,
                fused_scores,
                base_cap=cfg.adaptive_cap_base,
                max_cap=cfg.adaptive_cap_max,
                min_cap=cfg.adaptive_cap_min,
            )

        rank_method = cfg.detection_cap_rank_method
        # Only apply cap if we have more detections than the cap
        if fused_scores.numel() > max_det:
            fused_boxes, fused_scores, fused_classes = _apply_detection_cap(
                fused_boxes,
                fused_scores,
                fused_classes,
                max_detections=max_det,
                quality_factors=quality_factors_for_cap,
                rank_method=rank_method,
            )
        after_merge_count = int(fused_scores.numel())
    return (
        fused_boxes,
        fused_scores,
        fused_classes,
        geometry_suspect_mask,
        aligned_keypoints,
        after_merge_count,
    )


def _run_birth_config(
    state: "EvalPipeline",
    *,
    frame_id: int,
    frame_mid_thresh: float,
    frame_new_track_thresh: float,
    frame_track_thresh: float,
    fused_boxes: "torch.Tensor",
    fused_scores: "torch.Tensor",
    fused_quality_factors: "torch.Tensor | None" = None,
) -> "tuple[list[dict[str, float | int | str | bool]], torch.Tensor]":
    cfg = state.cfg
    h_orig = state.h_orig
    w_orig = state.w_orig
    seq_track_buffer = state.seq_track_buffer
    detector = state.detector
    _consec_birth_window = state._consec_birth_window
    _multi_birth_manager = state._multi_birth_manager
    _append_birth_event_rows = state.append_birth_event_rows
    frame_birth_events: list[dict[str, float | int | str | bool]] = []

    # === Consecutive-Frame Birth Gate ===
    # Boost sub-threshold detections that have appeared in the last N frames.
    # More selective than birth_quality_gate: requires temporal evidence, not
    # just per-frame geometry quality.
    scores_cloned = False
    if cfg.birth_consecutive_gate and fused_scores.numel() > 0:
        if len(_consec_birth_window) >= cfg.birth_consecutive_frames - 1:
            below_birth = fused_scores < frame_new_track_thresh
            if below_birth.any():
                sub_boxes = fused_boxes[below_birth]
                confirmed = _consecutive_birth_check(
                    sub_boxes,
                    list(_consec_birth_window),
                    cfg.birth_consecutive_iou,
                    min_motion_px=cfg.birth_consecutive_min_motion,
                )
                if confirmed.any():
                    boost_idx = below_birth.nonzero(as_tuple=True)[0][confirmed]
                    eligible = (
                        fused_scores[boost_idx] >= cfg.birth_consecutive_min_score
                    )
                    boost_idx = boost_idx[eligible]
                    if boost_idx.numel() > 0:
                        if not scores_cloned:
                            fused_scores = fused_scores.clone()
                            scores_cloned = True
                        score_before = fused_scores[boost_idx].clone()
                        fused_scores[boost_idx] = torch.clamp(
                            fused_scores[boost_idx] + cfg.birth_consecutive_boost,
                            max=cfg.high_thresh,
                        )
                        _append_birth_event_rows(
                            frame_birth_events,
                            _policy="birth_consecutive_gate",
                            _det_indices=boost_idx,
                            _score_before=score_before,
                            _score_after=fused_scores[boost_idx],
                            _boxes=fused_boxes[boost_idx],
                        )
    # Only keep sub-threshold boxes in window: prevents boosting detections
    # that match against already-tracked (high-score) detections from prior frames.
    _sub_thresh_boxes = fused_boxes[fused_scores < frame_new_track_thresh]
    _consec_birth_window.append(_sub_thresh_boxes.detach())

    # === Birth quality gate ===
    # Boost scores of high-quality detections that fall below new_track_thresh
    # so they can spawn new tracks without globally lowering the score floor.
    # Only detections with quality above birth_min_quality receive a boost.
    # Scores are capped at high_thresh to avoid Stage-1 promotion side effects.
    if (
        cfg.birth_quality_gate
        and fused_scores.numel() > 0
        and cfg.birth_quality_score_bias > 0.0
    ):
        if fused_quality_factors is not None:
            birth_quality = fused_quality_factors
        else:
            birth_quality = _compute_detection_quality_batch(
                fused_boxes,
                w_orig,
                h_orig,
                w_aspect=cfg.detection_quality_w_aspect,
                w_center=cfg.detection_quality_w_center,
                w_area=cfg.detection_quality_w_area,
            )
        below_birth = fused_scores < frame_new_track_thresh
        high_quality = birth_quality > cfg.birth_min_quality
        boost_mask = below_birth & high_quality
        if boost_mask.any():
            if not scores_cloned:
                fused_scores = fused_scores.clone()
                scores_cloned = True
            boost_idx = boost_mask.nonzero(as_tuple=True)[0]
            score_before = fused_scores[boost_idx].clone()
            boost = (
                birth_quality[boost_mask] - cfg.birth_min_quality
            ) * cfg.birth_quality_score_bias
            fused_scores[boost_mask] = torch.clamp(
                fused_scores[boost_mask] + boost,
                max=cfg.high_thresh,
            )
            _append_birth_event_rows(
                frame_birth_events,
                _policy="birth_quality_gate",
                _det_indices=boost_idx,
                _score_before=score_before,
                _score_after=fused_scores[boost_idx],
                _boxes=fused_boxes[boost_idx],
            )

    # === Multi-signal birth policy (P5-1) ===
    # Joint evidence (score × streak × motion × geometry) selectively promotes
    # sub-threshold detections that have accumulated enough multi-frame evidence.
    if _multi_birth_manager is not None and fused_scores.numel() > 0:
        below_birth = fused_scores < frame_new_track_thresh
        above_min = fused_scores >= cfg.multi_birth_min_score
        cand_mask = below_birth & above_min
        if cand_mask.any():
            promote_mask, replace_mask = _multi_birth_manager.update(
                frame_id,
                fused_boxes[cand_mask],
                fused_scores[cand_mask],
            )
            if promote_mask.any():
                if not scores_cloned:
                    fused_scores = fused_scores.clone()
                    scores_cloned = True
                boost_idx = cand_mask.nonzero(as_tuple=True)[0][promote_mask]
                score_before = fused_scores[boost_idx].clone()
                fused_scores[boost_idx] = frame_new_track_thresh + 0.01
                _append_birth_event_rows(
                    frame_birth_events,
                    _policy="multi_birth",
                    _det_indices=boost_idx,
                    _score_before=score_before,
                    _score_after=fused_scores[boost_idx],
                    _boxes=fused_boxes[boost_idx],
                )
        else:
            _multi_birth_manager.update(frame_id, fused_boxes[:0], fused_scores[:0])

    frame_tracker_thresholds = (
        frame_track_thresh,
        frame_mid_thresh,
        frame_new_track_thresh,
    )
    if frame_tracker_thresholds != state.active_tracker_thresholds:
        detector.tracker.set_params(
            track_thresh=frame_track_thresh,
            high_thresh=cfg.high_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=seq_track_buffer,
            mid_thresh=frame_mid_thresh,
            confirm_streak=int(cfg.kwargs.get("confirm_streak", 1)),
            confirm_score_thresh=float(cfg.kwargs.get("confirm_score_thresh", 0.0)),
            adaptive_confirmation=bool(cfg.kwargs.get("adaptive_confirmation", False)),
            new_track_thresh=frame_new_track_thresh,
            kalman_adapt_mode=cfg.kalman_adapt_mode,
            r_scale=cfg.kalman_r_scale,
            vel_dir_weight=cfg.vel_dir_weight,
            fuse_score_weight=cfg.fuse_score_weight,
            stage2_match_thresh=cfg.stage2_match_thresh,
            birth_low_score_thresh=cfg.birth_low_score_thresh,
            birth_prox_norm_thresh=cfg.birth_prox_norm_thresh,
        )
        state.active_tracker_thresholds = frame_tracker_thresholds
    return frame_birth_events, fused_scores


def _run_reid_and_gmc(
    state: "EvalPipeline",
    *,
    frame_id: int,
    fused_boxes: "torch.Tensor",
    fused_scores: "torch.Tensor",
    fused_classes: "torch.Tensor",
    after_merge_count: int,
    current_stage_sample_active: bool,
    _fpn_cache: "dict[str, torch.Tensor]",
) -> "tuple[torch.Tensor | None, float]":
    _fpn_reid_mode = state.contract.fpn_reid_mode
    _fpn_img_size = state.fpn.img_size
    _fpn_reid_conv_weights = state.fpn.conv_weights
    _fpn_reid_dim = state.contract.feature_dim
    _fpn_reid_proj_weight = state.fpn.proj_weight
    _fpn_reid_running_mean = state.fpn.running_mean
    _fpn_reid_running_var = state.fpn.running_var
    cfg = state.cfg
    cropper = state.cropper
    detector = state.detector
    dynamic_reid = state.dynamic_reid
    extractor = state.extractor
    geometry_scale_state = state.geometry_scale_state
    gmc_estimator = state.gmc_estimator
    h_orig = state.h_orig
    native_reid_available = state.native_reid_available
    perception_pipeline = state.perception_pipeline
    pool = state.pool
    primary_appearance_bank = state.primary_appearance_bank
    profile_stages = state.profile_stages
    record_stage_sample = state.record_stage_sample
    reid_main_ready = state.reid_main_ready
    reid_side_event = state.reid_side_event
    reid_side_stream = state.reid_side_stream
    seq_native_reid_samples = state.seq_native_reid_samples
    seq_reid_interval = state.seq_reid_interval
    seq_stage_totals = state.seq_stage_totals
    time_stage = state.time_stage
    w_orig = state.w_orig
    _use_direct_gmc = state.use_direct_gmc
    _reid_side_pending = False
    _reid_async_embeddings: torch.Tensor | None = None
    _reid_async_indices: torch.Tensor | None = None
    _reid_frame_hwc_ref: torch.Tensor | None = None
    embeddings = None
    if state._frame_stage_times is not None:
        state._frame_stage_times["_reid_enter"] = time.perf_counter()
    state.appearance_occlusion_mask = None
    _fpn_ready = _fpn_reid_mode and fused_boxes.numel() > 0
    _do_reid = _fpn_ready or (
        cfg.reid_work_enabled
        and extractor is not None
        and cropper is not None
        and fused_boxes.numel() > 0
    )
    if _do_reid:
        if not _fpn_ready:
            MIN_REID_GAP = (
                max(1, getattr(state.live_evfifo, "stride", 5))
                if state.live_evfifo is not None
                else 2
            )
            time_since_last_reid = frame_id - state.last_reid_frame

            if time_since_last_reid < MIN_REID_GAP:
                _do_reid = False
            elif cfg.need_reid_enabled:
                if hasattr(detector.tracker, "need_reid"):
                    _do_reid = detector.tracker.need_reid(after_merge_count)
                elif dynamic_reid is not None:
                    _do_reid = dynamic_reid.should_reid(after_merge_count)
                else:
                    _do_reid = need_reid_frame(state.prev_track_ids, after_merge_count)
            else:
                _do_reid = seq_reid_interval > 0 and frame_id % seq_reid_interval == 0

        if _do_reid:
            state.last_reid_frame = frame_id
            if state.frame_ledger is not None:
                state._frame_reid_stats = {
                    "submitted": True,
                    "n_crops": 0,
                    "n_requery": 0,
                    "crop_ms": 0.0,
                    "extract_ms": 0.0,
                    "blocking_wait_ms": 0.0,
                    "waited": False,
                }
                _t_reid_start = time.perf_counter()
            if primary_appearance_bank is not None:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_reid_bank_sync_start = time.perf_counter()
                bank_reps = primary_appearance_bank.representatives()
                if bank_reps:
                    detector.tracker.set_reference_features_from_bank(bank_reps)
                clean_ids = primary_appearance_bank.clean_ids()
                if clean_ids:
                    _clean_ids_list = list(clean_ids)
                    _ids_t = torch.tensor(_clean_ids_list, dtype=torch.int32)
                    _flags_t = torch.ones(len(_clean_ids_list), dtype=torch.bool)
                    detector.tracker.set_clean_embedding_flags(_ids_t, _flags_t)
                else:
                    detector.tracker.set_clean_embedding_flags(
                        torch.zeros(0, dtype=torch.int32),
                        torch.zeros(0, dtype=torch.bool),
                    )
                if profile_stages:
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t_reid_bank_sync_start) * 1000
                    seq_stage_totals["reid_bank_sync"] += elapsed_ms
                    record_stage_sample("reid_bank_sync", elapsed_ms)

    if cfg.reid_enabled and _do_reid:
        # ── FPN ReID fast path ──
        if _fpn_reid_mode and fused_boxes.shape[0] > 0:
            _trt_feat_cache = getattr(detector, "_trt_feat_cache", None)
            if _fpn_cache:
                _img_sz = _fpn_img_size
            elif _trt_feat_cache:
                _p3_cache = _trt_feat_cache.get("p3")
                _img_sz = int(_p3_cache.shape[2] * 8) if _p3_cache is not None else 640
            elif hasattr(detector, "teacher"):
                _p3_cache = detector.teacher._gate_layers["p3"]._feat_cache.get("p3")
                _img_sz = int(_p3_cache.shape[2] * 8) if _p3_cache is not None else 640
            else:
                _img_sz = 960
            boxes_rescaled = fused_boxes.clone()
            sx = _img_sz / w_orig
            sy = _img_sz / h_orig
            boxes_rescaled[:, 0] *= sx
            boxes_rescaled[:, 1] *= sy
            boxes_rescaled[:, 2] *= sx
            boxes_rescaled[:, 3] *= sy
            if _trt_feat_cache:
                fpn = [_trt_feat_cache[s] for s in ("p3", "p4", "p5")]
            elif hasattr(detector, "teacher"):
                fpn = [
                    detector.teacher._gate_layers[s]._feat_cache[s]
                    for s in ("p3", "p4", "p5")
                ]
            elif _fpn_cache:
                fpn = [_fpn_cache[s] for s in ("p3", "p4", "p5")]
            else:
                fpn = None
            if fpn is not None:
                if _fpn_reid_conv_weights is not None:
                    from saccade.perception.tracking.fpn_reid_cuda import (
                        fpn_reid_extract_cuda,
                    )

                    embeddings = fpn_reid_extract_cuda(
                        fpn,
                        _fpn_reid_conv_weights,
                        boxes_rescaled,
                        img_size=_img_sz,
                        proj_weight=_fpn_reid_proj_weight,
                        running_mean=_fpn_reid_running_mean,
                        running_var=_fpn_reid_running_var,
                    )
                else:
                    embeddings = detector.extract_fpn_embeddings(None, boxes_rescaled)
        else:
            if profile_stages:
                torch.cuda.synchronize()
                t_reid_budget_start = time.perf_counter()

            num_dets = fused_boxes.shape[0]
            appearance_occlusion_mask = None
            if cfg.appearance_occlusion_gate:
                appearance_occlusion_mask = _front_occlusion_mask_xyxy(
                    fused_boxes, cfg.appearance_occlusion_cov
                )
                state.appearance_occlusion_mask = appearance_occlusion_mask
            if cfg.reid_budget_raw >= 1.0:
                actual_budget = int(cfg.reid_budget_raw)
            elif cfg.reid_budget_raw > 0.0:
                actual_budget = max(1, int(cfg.reid_budget_raw * num_dets))
            else:
                actual_budget = 0  # Unlimited or handled by _budget_reid_candidates

            budget_indices = _budget_reid_candidates(
                fused_boxes,
                fused_scores,
                actual_budget,
                dynamic_reid=dynamic_reid,
                gmc_warp=state.gmc_warp if cfg.gmc_enabled else None,
                gmc_uncertain=state.gmc_uncertain,
            )
            if appearance_occlusion_mask is not None and budget_indices.numel() > 0:
                budget_indices = budget_indices[
                    ~appearance_occlusion_mask[budget_indices]
                ]

            if profile_stages:
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t_reid_budget_start) * 1000
                seq_stage_totals["reid_budget"] += elapsed_ms
                record_stage_sample("reid_budget", elapsed_ms)

            _reid_feat_dim = _fpn_reid_dim if _fpn_reid_mode else extractor.feature_dim
            # Initialize full embeddings with zeros. Detections without budget
            # will have neutral features for association.
            embeddings = torch.zeros(
                (fused_boxes.shape[0], _reid_feat_dim),
                device=fused_boxes.device,
                dtype=torch.float32,
            )

            if budget_indices.numel() > 0:
                budgeted_boxes = fused_boxes[budget_indices].contiguous()
                if state._frame_reid_stats is not None:
                    state._frame_reid_stats["n_crops"] = int(budgeted_boxes.shape[0])
                    _t_reid_crop = time.perf_counter()

                if native_reid_available and perception_pipeline is not None:
                    from .helpers import FLOW_TIMING, flow_add, flow_now

                    _t0 = flow_now() if FLOW_TIMING else 0.0
                    frame_hwc = pool.as_rgb_chw().permute(1, 2, 0).contiguous()
                    # Shared with the emit-stage crop-ring stash (read-only).
                    state.frame_hwc_cache = (frame_id, frame_hwc)
                    if FLOW_TIMING:
                        flow_add("reid_hwc", flow_now() - _t0)
                    budget_embeddings = torch.empty(
                        (budget_indices.numel(), extractor.feature_dim),
                        device=fused_boxes.device,
                        dtype=torch.float32,
                    )

                    if cfg.async_reid and reid_side_stream is not None:
                        reid_main_ready.record()
                        with torch.cuda.stream(reid_side_stream):
                            reid_side_stream.wait_event(reid_main_ready)
                            perception_pipeline.extract_reid(
                                frame_hwc.data_ptr(),
                                h_orig,
                                w_orig,
                                budgeted_boxes.data_ptr(),
                                int(budgeted_boxes.shape[0]),
                                budget_embeddings.data_ptr(),
                                reid_side_stream.cuda_stream,
                            )
                        reid_side_event.record(reid_side_stream)
                        _reid_side_pending = True
                        _reid_async_embeddings = budget_embeddings
                        _reid_async_indices = budget_indices
                        _reid_frame_hwc_ref = frame_hwc
                        if state._frame_reid_stats is not None:
                            state._frame_reid_stats["crop_ms"] = round(
                                (time.perf_counter() - _t_reid_crop) * 1000, 6
                            )
                    else:
                        if profile_stages:
                            perception_pipeline.reset_reid_profile_stats()
                            torch.cuda.synchronize()
                            t_reid_extract_start = time.perf_counter()

                        _t0 = flow_now() if FLOW_TIMING else 0.0
                        perception_pipeline.extract_reid(
                            frame_hwc.data_ptr(),
                            h_orig,
                            w_orig,
                            budgeted_boxes.data_ptr(),
                            int(budgeted_boxes.shape[0]),
                            budget_embeddings.data_ptr(),
                            torch.cuda.current_stream().cuda_stream,
                        )
                        if FLOW_TIMING:
                            flow_add("reid_extract_call", flow_now() - _t0)

                        if state._frame_reid_stats is not None:
                            state._frame_reid_stats["extract_ms"] = round(
                                (time.perf_counter() - _t_reid_crop) * 1000, 6
                            )
                        if profile_stages:
                            torch.cuda.synchronize()
                            elapsed_ms = (
                                time.perf_counter() - t_reid_extract_start
                            ) * 1000
                            seq_stage_totals["reid_extract"] += elapsed_ms
                            record_stage_sample("reid_extract", elapsed_ms)
                            if current_stage_sample_active:
                                native_stats = (
                                    perception_pipeline.get_reid_profile_stats()
                                )
                                seq_native_reid_samples["native_reid_crop"].append(
                                    float(native_stats.get("crop_ms", 0.0))
                                )
                                seq_native_reid_samples[
                                    "native_reid_pre_normalize"
                                ].append(
                                    float(
                                        native_stats.get(
                                            "extract_pre_normalize_ms",
                                            0.0,
                                        )
                                    )
                                )
                                seq_native_reid_samples[
                                    "native_reid_trt_enqueue"
                                ].append(
                                    float(
                                        native_stats.get(
                                            "extract_trt_enqueue_ms",
                                            0.0,
                                        )
                                    )
                                )
                                seq_native_reid_samples[
                                    "native_reid_l2_normalize"
                                ].append(
                                    float(
                                        native_stats.get(
                                            "extract_l2_normalize_ms",
                                            0.0,
                                        )
                                    )
                                )
                        embeddings[budget_indices] = budget_embeddings
                else:
                    frame_batch = pool.as_rgb_chw().unsqueeze(0)
                    if cfg.reid_crop_layout == "parts":
                        if profile_stages:
                            torch.cuda.synchronize()
                            t_reid_crop_start = time.perf_counter()
                        crops = cropper.process_parts(frame_batch, budgeted_boxes)
                        if profile_stages:
                            torch.cuda.synchronize()
                            elapsed_ms = (
                                time.perf_counter() - t_reid_crop_start
                            ) * 1000
                            seq_stage_totals["reid_crop"] += elapsed_ms
                            record_stage_sample("reid_crop", elapsed_ms)

                        if crops.numel() > 0:
                            if profile_stages:
                                torch.cuda.synchronize()
                                t_reid_extract_start = time.perf_counter()
                            budget_embeddings = extractor.extract_parts_fused(crops)
                            if profile_stages:
                                torch.cuda.synchronize()
                                elapsed_ms = (
                                    time.perf_counter() - t_reid_extract_start
                                ) * 1000
                                seq_stage_totals["reid_extract"] += elapsed_ms
                                record_stage_sample("reid_extract", elapsed_ms)
                            embeddings[budget_indices] = budget_embeddings
                    else:
                        if profile_stages:
                            torch.cuda.synchronize()
                            t_reid_crop_start = time.perf_counter()
                        crops = cropper.process(frame_batch, budgeted_boxes)
                        if profile_stages:
                            torch.cuda.synchronize()
                            elapsed_ms = (
                                time.perf_counter() - t_reid_crop_start
                            ) * 1000
                            seq_stage_totals["reid_crop"] += elapsed_ms
                            record_stage_sample("reid_crop", elapsed_ms)

                        if crops.numel() > 0:
                            if profile_stages:
                                torch.cuda.synchronize()
                                t_reid_extract_start = time.perf_counter()
                            budget_embeddings = extractor.extract(crops)
                            if profile_stages:
                                torch.cuda.synchronize()
                                elapsed_ms = (
                                    time.perf_counter() - t_reid_extract_start
                                ) * 1000
                                seq_stage_totals["reid_extract"] += elapsed_ms
                                record_stage_sample("reid_extract", elapsed_ms)
                            embeddings[budget_indices] = budget_embeddings

    mid_thresh_scale = geometry_mid_thresh_scale(
        fused_boxes,
        fused_classes,
        h_orig,
        enabled=cfg.geometry_mid_scale,
        person_class=cfg.person_class,
        track_person_only=cfg.track_person_only,
        ref_height_ratio=cfg.geometry_ref_height_ratio,
        min_scale=cfg.geometry_min_scale,
        max_scale=cfg.geometry_max_scale,
        ema_beta=cfg.geometry_ema_beta,
        loosen_step=cfg.geometry_loosen_step,
        tighten_step=cfg.geometry_tighten_step,
        min_samples=cfg.geometry_min_samples,
        state=geometry_scale_state,
    )
    if (
        state._frame_stage_times is not None
        and "_reid_enter" in state._frame_stage_times
    ):
        _now = time.perf_counter()
        state._frame_stage_times["reid"] = round(
            (_now - state._frame_stage_times["_reid_enter"]) * 1000, 6
        )
        state._frame_stage_times["_gmc_enter"] = _now
    state.gmc_warp = None
    state.gmc_uncertain = False
    # GMC estimator takes luma from frame_buffer (RGB path)
    # or from NV12 Y-plane directly (NV12 path).
    # C++ GMC estimator expects [3, H, W] — clone luma to 3 channels in NV12 mode.
    if pool.use_nv12:
        _luma = pool.get_frame_luma()
        _frame_gmc = _luma.repeat(3, 1, 1)
    else:
        _frame_gmc = pool.frame_buffer

    if gmc_estimator is not None:
        (state.gmc_warp, state.gmc_uncertain), _ = time_stage(
            seq_stage_totals,
            "gmc",
            lambda: _run_gmc_estimate(
                state,
                fused_boxes=fused_boxes,
                _frame_gmc=_frame_gmc,
            ),
            # Only the pure-CPU estimators need a host sync for timing;
            # the GPU direct/graph and PyGraphed paths feed the tracker
            # (which syncs) so an extra sync here only stalls overlap.
            sync_cuda=not _use_direct_gmc
            and not isinstance(gmc_estimator, PyGraphedGMC),
        )
        if hasattr(detector, "set_gmc_warp"):
            detector.set_gmc_warp(state.gmc_warp, h_orig, w_orig)
    # Async ReID: sync side stream and inject fresh embeddings right before
    # tracker.update_into so the cost matrix still has detection-side appearance.
    # GMC on main stream overlapped with reid on side stream during the gap above.
    if _reid_side_pending and reid_side_event is not None:
        if profile_stages:
            torch.cuda.synchronize()
            t_reid_extract_start = time.perf_counter()
        if state._frame_reid_stats is not None:
            _t_blocking = time.perf_counter()
        reid_side_event.synchronize()
        if state._frame_reid_stats is not None:
            state._frame_reid_stats["blocking_wait_ms"] = round(
                (time.perf_counter() - _t_blocking) * 1000, 6
            )
            state._frame_reid_stats["waited"] = True
        if profile_stages:
            elapsed_ms = (time.perf_counter() - t_reid_extract_start) * 1000
            seq_stage_totals["reid_extract"] += elapsed_ms
            record_stage_sample("reid_extract", elapsed_ms)
        embeddings[_reid_async_indices] = _reid_async_embeddings
        _reid_side_pending = False
        _reid_frame_hwc_ref = None
    if (
        state._frame_stage_times is not None
        and "_gmc_enter" in state._frame_stage_times
    ):
        _now = time.perf_counter()
        state._frame_stage_times["gmc"] = round(
            (_now - state._frame_stage_times["_gmc_enter"]) * 1000, 6
        )
        state._frame_stage_times["_stage_end"] = _now
    return embeddings, mid_thresh_scale
