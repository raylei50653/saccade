# mypy: ignore-errors
"""Eval pipeline state extracted from evaluator.py.

EvalPipeline is the per-sequence state bag built once before the frame loop:
detector/streamer/tracker/relinker construction, buffers, CUDA graphs, and
all config-derived constants. Stage functions (stages.py) and the frame loop
(evaluator.py) read and mutate this state.
"""

# mypy: ignore-errors
import configparser
import os
import threading
import time
from collections import OrderedDict, deque
import dataclasses
from pathlib import Path

import numpy as np
import torch

from typing import Any

from .types import (
    HostTrackResultView,
    HostTrackBatch,
)
from .lifecycle import (
    IdStabilityFilter,
    TrackletLifecycleMerger,
)
from .output_bank import OutputAppearanceBank
from .helpers import (
    resolve_frame_tracks as _resolve_frame_tracks,
    prepare_track_candidates as _prepare_track_candidates,
    emit_resolved_tracks as _emit_resolved_tracks,
    finalize_frame_side_effects as _finalize_frame_side_effects,
)

# Perception/eval modules load local extensions before any torchvision fallback.
from .scene_adapt import SceneAdaptivePolicy


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


from saccade.perception.eval.gmc import (  # noqa: E402
    SparseOpticalFlowGMC,
    PyGraphedGMC,
    TilePhaseCorrAffineGMC,
)
from saccade.perception.eval.multi_birth import MultiSignalBirthManager  # noqa: E402
from saccade.perception.eval.pool import (  # noqa: E402
    AdaptiveFramePool,
)
from saccade.perception.eval.preprocess import (  # noqa: E402
    GeometryScaleState,
)
from saccade.perception.eval.relink import (  # noqa: E402
    IdentityResolver,
    PythonSemanticRelinker,
    SemanticRelinker,
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
from saccade.perception.eval.streaming import (  # noqa: E402
    DALIStreamerStream,
    TorchvisionGpuStreamer,
)
from saccade.perception.tracking.dynamic_reid import DynamicReIDController  # noqa: E402
from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    TrackAppearanceBank,
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


def _resolve_kalman_fps(explicit_fps: float, seq_path: "Path") -> float:
    """Resolve the fps used by the physical Kalman relink model.

    Priority: explicit flag (>0) → seqinfo.ini frameRate → sibling/parent .mp4
    probe (cv2) → 30.0. The physical diffusion converts m/s² to px/frame² and is
    sensitive to fps², so it must match the real source rather than a hard default.
    """
    if explicit_fps and explicit_fps > 0.0:
        return float(explicit_fps)
    # 1. seqinfo.ini (MOT-standard; configparser lowercases keys so "framerate"
    #    and "frameRate" both match)
    seqinfo = seq_path / "seqinfo.ini"
    if seqinfo.exists():
        cp = configparser.ConfigParser()
        cp.read(str(seqinfo))
        fr = cp.getfloat("Sequence", "frameRate", fallback=0.0)
        if fr > 0.0:
            return float(fr)
    # 2. probe a video file (mp4/mov/...) in the seq dir or its parent
    try:
        import cv2  # type: ignore

        for base in (seq_path, seq_path.parent):
            for vid in sorted(base.glob("*.mp4")) + sorted(base.glob("*.mov")):
                cap = cv2.VideoCapture(str(vid))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0.0:
                    return float(fps)
    except Exception:
        pass
    # 3. fallback
    return 30.0


def _detect_barrier_mode() -> str:
    """Stream-ordering policy between the ingest / detect / postprocess stages.

    The two ``torch.cuda.synchronize()`` calls in ``_run_detect`` are full-device
    barriers added as the determinism fix for the ingest->detect stale-buffer race
    (commit 3046ae60). They are correct but block the host ~2.9ms/frame, serialising
    the per-frame stages and preventing cross-frame overlap.

    Modes (env ``SACCADE_DETECT_BARRIER``):
      * ``full`` (default): both full-device barriers — current, zero-risk behaviour.
      * ``no_postproc``: drop the detect->postprocess barrier only (likely redundant
        in whole_graph mode, where TRT + postprocess graphs share the launch stream).
        Keeps the ingest->detect (decode-race) barrier.
      * ``event``: ``no_postproc`` + replace the ingest->detect full sync with a
        narrower same-stream ordering. EXPERIMENTAL.

    Any non-``full`` mode is GPU-decode-determinism-sensitive and MUST pass N>=6
    repeat runs (zero run-to-run drift) + bit-exact A/B before use; see
    project_eval_nondeterminism_source.
    """
    return (os.getenv("SACCADE_DETECT_BARRIER", "full") or "full").strip().lower()


def _double_buffer_eligible(cfg: Any, detector: Any, profile_stages: bool) -> bool:
    """Return whether the safe, frame-independent overlap path can run.

    CUDA graph outputs and tracker state are both mutable, so this deliberately
    has a narrow contract.  Temporal detectors, the workbench path and stage
    profiling retain the serial path.  The ingest->detect full-device barrier
    also makes overlap impossible; ``event`` is the previously documented,
    determinism-gated narrow-barrier mode.
    """

    requested = os.getenv("SACCADE_DOUBLE_BUFFER", "0").strip().lower()
    if requested not in {"1", "true", "yes", "on"}:
        return False
    # Whole-graph forward bypasses the detector's temporal ring buffer even
    # when the checkpoint advertises a temporal training configuration.
    # Eager forward, in contrast, mutates that buffer and must stay serial.
    frame_independent_detect = int(getattr(detector, "_temporal_T", 0)) == 0 or bool(
        getattr(detector, "use_whole_graph", False)
    )
    return bool(
        torch.cuda.is_available()
        and not profile_stages
        and not getattr(cfg, "workbench", False)
        and frame_independent_detect
        and _detect_barrier_mode() == "event"
    )


@dataclasses.dataclass(frozen=True)
class DetectionContract:
    """Immutable contract between detection pipeline and tracking consumers.

    Captures the data-format assumptions that downstream modules (kalman gate,
    relink, tracker) depend on.  When the detection pipeline configuration
    changes (e.g. different FPN backbone, different feature dim), this
    contract must be updated so that validation in _run_track catches
    mismatches before silent corruption.
    """

    feature_dim: int
    fpn_reid_mode: bool
    box_format: str = "xyxy"


@dataclasses.dataclass(frozen=True)
class FPNConfig:
    """FPN backbone runtime configuration (only meaningful when fpn_reid_mode)."""

    backbone: Any = None
    img_size: int = 640
    conv_weights: Any = None
    proj_weight: Any = None
    running_mean: Any = None
    running_var: Any = None


class EvalPipeline:
    """Per-sequence evaluation pipeline state.

    Holds references to the per-sequence locals (buffers, estimators, profiling
    accumulators) so each stage helper takes ``state`` instead of a dozen
    explicit params. Buffers are mutated in place exactly as before — this is a
    passing convenience, not a second source of truth.

    This is the nucleus of the eval pipeline object: stage helpers will migrate
    onto it as methods and the per-frame cross-frame loop-locals (gmc_warp,
    last_reid_frame, prev_track_ids, bg_future/bg_birth_events, deferred-emit
    state) will become fields, dissolving the hand-threaded in/out plumbing.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        seq: str,
        profile_stages: bool,
        contract: DetectionContract,
        detector: Any,
        cropper: Any,
        extractor: Any,
        fpn: FPNConfig | None = None,
        debug_birth_rows: Any = None,
        global_id_mapper: Any = None,
        gmc_breakdown_names: Any = None,
        max_frames: int | None = None,
        native_cfg: Any = None,
        native_reid_breakdown_names: Any = None,
        overall_stage_totals: Any = None,
        segment_breakdown_names: Any = None,
        top_level_stage_names: Any = None,
        time_stage: Any = None,
        record_stage_sample: Any = None,
        _rw_executor: Any = None,
        perception_pipeline: Any = None,
        reid_main_ready: Any = None,
        reid_side_event: Any = None,
        reid_side_stream: Any = None,
        debug_dump_frames: Any = None,
        debug_dump_seq: Any = None,
        debug_stage_dump_rows: Any = None,
        detect_fn: Any = None,
        detector_box_format: Any = None,
        enable_onms: Any = None,
        onms_min_track_age: Any = None,
        onms_min_track_score: Any = None,
        onms_prior_iou_threshold: Any = None,
        native_reid_available: Any = None,
        external_fp_rule_config: Any = None,
        external_fp_logistic_model: Any = None,
    ) -> None:
        # ── params that aren't computed by setup ──────────────────────
        self.cfg = cfg
        self.seq = seq
        self.profile_stages = profile_stages
        self.contract = contract
        self.detector = detector
        self.cropper = cropper
        self.extractor = extractor
        self.fpn = fpn if fpn is not None else FPNConfig()
        self.global_id_mapper = global_id_mapper
        self.native_cfg = native_cfg
        self.top_level_stage_names = top_level_stage_names
        self.time_stage = time_stage
        self.record_stage_sample = record_stage_sample
        self.rw_executor = _rw_executor
        self.perception_pipeline = perception_pipeline
        self.reid_main_ready = reid_main_ready
        self.reid_side_event = reid_side_event
        self.reid_side_stream = reid_side_stream
        self.debug_dump_frames = debug_dump_frames
        self.debug_dump_seq = debug_dump_seq
        self.debug_stage_dump_rows = debug_stage_dump_rows
        self.detect_fn = detect_fn
        self.detector_box_format = detector_box_format
        self.enable_onms = enable_onms
        self.onms_min_track_age = onms_min_track_age
        self.onms_min_track_score = onms_min_track_score
        self.onms_prior_iou_threshold = onms_prior_iou_threshold
        self.native_reid_available = native_reid_available
        self.external_fp_rule_config = external_fp_rule_config
        self.external_fp_logistic_model = external_fp_logistic_model
        # ── cross-frame state (fresh per seq) ─────────────────────────
        self.defer_emit_event: torch.cuda.Event | None = None
        self.defer_emit_fid: int = 0
        self.bg_future: Any = None
        self.bg_birth_events: Any = None
        self.nms_graph: Any = None
        # The graph writes the post-NMS count through this pointer on every
        # replay. It must outlive the CUDA graph; a function-local tensor may
        # otherwise be returned to PyTorch's allocator after capture.
        self.nms_graph_out_count: torch.Tensor | None = None
        self.gmc_uncertain: bool = False
        self.last_reid_frame: int = -100
        self.prev_gray: torch.Tensor | None = None
        self.seq_profiled_frames: int = 0
        self.seq_lazy_reid_candidates: int = 0
        self.seq_lazy_reid_frames: int = 0
        self.seq_lazy_reid_crops: int = 0
        self.seq_lazy_reid_self_pairs: int = 0
        self.seq_lazy_reid_self_pass: int = 0
        self.seq_lazy_reid_self_sim_sum: float = 0.0
        self.seq_lazy_reid_arbiter_checks: int = 0
        self.seq_lazy_reid_arbiter_approve: int = 0
        self.gmc_warp: torch.Tensor | None = None
        self.prev_track_ids: set = set()
        self.current_frame_id: int = 0
        # Latency is frame-local; throughput is measured independently from the
        # wall-clock interval over completed, non-warmup frames.
        self.throughput_started_at: float | None = None
        self.throughput_finished_at: float | None = None
        self.throughput_frames: int = 0
        # ── per-seq setup ─────────────────────────────────────────────
        wb = None
        wb_scene_policy = None
        if getattr(cfg, "workbench", False):
            from saccade.perception.workbench import Workbench

            # Build workbench with quality/ReID/GMC components (baseline-aligned)
            wb = Workbench(
                detector,
                native_cfg,
                device=str(detector.device),
                max_dets=2048,
                max_tracks=256,
                # Quality/ReID/GMC components
                extractor=extractor,
                cropper=cropper,
                gmc_estimator=None,  # set below after gmc_estimator is created
                narrow_bonus=0.0,
                # Quality filter params
                quality_weights=(
                    cfg.detection_quality_w_aspect,
                    cfg.detection_quality_w_center,
                    cfg.detection_quality_w_area,
                )
                if cfg.detection_quality_scaling
                else (0.5, 0.3, 0.2),
                max_detections=cfg.per_frame_detection_cap or 30,
                fp_hard_filter=cfg.fp_hard_filter_enabled,
                fp_min_score=cfg.fp_hard_filter_min_score,
                fp_max_suspicious_area=cfg.fp_hard_filter_max_suspicious_area,
                fp_max_suspicious_score=cfg.fp_hard_filter_max_suspicious_score,
                # ReID params
                reid_budget_raw=cfg.reid_budget_raw,
                reid_interval=cfg.reid_interval,
                need_reid=cfg.need_reid_enabled,
                dynamic_reid=None,
                gmc_uncertain=False,
            )
        else:
            tracker_feature_dim = (
                contract.feature_dim
                if contract.fpn_reid_mode
                else (extractor.feature_dim if extractor is not None else 0)
            )
            if tracker_feature_dim > 0:
                from saccade.perception.tracking import GPUByteTracker

                detector.tracker = GPUByteTracker(
                    max_objects=2048, embedding_dim=tracker_feature_dim
                )
            else:
                detector.reset_tracker()

        geometry_scale_state = GeometryScaleState()

        gmc_estimator, _use_direct_gmc, _gmc_graphable, _gmc_cuda_graph = (
            _build_gmc_estimator(cfg, profile_stages)
        )

        # Set scene-adapt policy on workbench
        if wb is not None:
            wb_scene_policy = (
                SceneAdaptivePolicy(
                    window=cfg.scene_adapt_window,
                    crowd_box_thresh=cfg.scene_adapt_crowd_thresh,
                    narrow_aspect_thresh=cfg.scene_adapt_narrow_aspect_thresh,
                    narrow_width_thresh=cfg.scene_adapt_narrow_width_thresh,
                )
                if cfg.scene_adapt_enabled
                else None
            )
            wb.scene_adapt_policy = wb_scene_policy

        # Set GMC estimator on workbench (after gmc_estimator is created)
        if wb is not None and gmc_estimator is not None:
            wb.gmc_estimator = gmc_estimator

        # Load homography if provided (ADR 017)
        if cfg.homography_root:
            h_path = Path(cfg.homography_root) / f"{seq}.txt"
            if h_path.exists():
                try:
                    h_mat = np.loadtxt(h_path).astype(np.float32).flatten()
                    if h_mat.size == 9:
                        detector.tracker.set_homography(h_mat)
                except Exception:
                    detector.tracker.set_homography(None)
            else:
                detector.tracker.set_homography(None)
        else:
            detector.tracker.set_homography(None)

        id_stability_filter = (
            IdStabilityFilter(
                min_hits=cfg.id_stability_min_hits,
                min_iou=cfg.id_stability_min_iou,
                max_center_shift=cfg.id_stability_max_center_shift,
                max_gap=cfg.id_stability_max_gap,
                score_ema=cfg.id_stability_score_ema,
                min_score_ema=cfg.id_stability_min_score_ema,
            )
            if cfg.id_stability_filter_enabled
            else None
        )
        lifecycle_merger = (_LIFECYCLE_CLS or TrackletLifecycleMerger)(
            enabled=cfg.lifecycle_merge_enabled,
            ttl=cfg.lifecycle_ttl,
            min_gap=cfg.lifecycle_min_gap,
            spatial_gate=cfg.lifecycle_spatial_gate,
            min_iou=cfg.lifecycle_min_iou,
            sim_threshold=cfg.lifecycle_sim_threshold,
            require_embedding=cfg.lifecycle_require_embedding,
            ema=cfg.lifecycle_ema,
        )
        detector.tracker.set_reid_params(
            cos_threshold=float(cfg.kwargs.get("reid_cos_threshold", 0.90)),
            iou_low=float(cfg.kwargs.get("reid_iou_low", 0.30)),
            iou_high=float(cfg.kwargs.get("reid_iou_high", 0.60)),
            weight=float(cfg.kwargs.get("reid_weight", 0.80)),
            cost_cos_w=float(cfg.kwargs.get("reid_cost_cos_w", 0.55)),
            cost_iou_w=float(cfg.kwargs.get("reid_cost_iou_w", 0.30)),
            cost_score_w=float(cfg.kwargs.get("reid_cost_score_w", 0.15)),
        )
        if contract.fpn_reid_mode and hasattr(
            detector.tracker, "set_reid_min_candidates"
        ):
            detector.tracker.set_reid_min_candidates(1)
        _bridge_enabled = bool(getattr(cfg, "relink_bridge_enabled", False))
        if (cfg.relink_enabled or _bridge_enabled) and hasattr(
            detector.tracker, "set_relink_params"
        ):
            detector.tracker.set_relink_params(
                enabled=cfg.relink_enabled,
                bank_cap=cfg.relink_bank_cap,
                sim_thresh=cfg.relink_sim_thresh,
                cheb_lambda=cfg.relink_lambda,
                spatial_gate=cfg.relink_spatial_gate,
                max_age=cfg.relink_max_age,
                bidirectional=_bridge_enabled,
                bridge_px=cfg.relink_bridge_px,
                bridge_at=cfg.relink_bridge_at,
                bridge_min_lost=cfg.relink_bridge_min_lost,
                bridge_ttl=cfg.relink_bridge_ttl,
                bridge_max_speed=cfg.relink_bridge_max_speed,
                bridge_person_height=cfg.relink_bridge_person_height,
                bridge_fps=cfg.relink_bridge_fps,
                bridge_margin=cfg.relink_bridge_margin,
                bridge_spatial_gate=cfg.relink_bridge_spatial_gate,
                bridge_anchor={"center": 0, "foot": 1, "adaptive": 2}.get(
                    cfg.relink_bridge_anchor, 0
                ),
                bridge_anchor_rate=cfg.relink_bridge_anchor_rate,
                bridge_h_lo=cfg.relink_bridge_h_lo,
                bridge_h_hi=cfg.relink_bridge_h_hi,
                bridge_dir_bonus=cfg.relink_bridge_dir_bonus,
                occ_gate_cover=cfg.relink_bridge_occ_gate_cover,
                occ_gap_min=cfg.relink_bridge_occ_gap_min,
                occ_expand_px=cfg.relink_bridge_occ_expand_px,
                occ_expand_cover=cfg.relink_bridge_occ_expand_cover,
                bridge_app_veto=getattr(cfg, "relink_bridge_app_veto", -1.0),
            )

        if hasattr(detector.tracker, "set_unified_score_params"):
            detector.tracker.set_unified_score_params(
                w_sim_base=cfg.semantic_w_sim_base,
                w_iou_base=cfg.semantic_w_iou_base,
                w_maha_base=cfg.semantic_w_maha_base,
                shift_ambiguity=cfg.semantic_shift_ambiguity,
                shift_lost_age=cfg.semantic_shift_lost_age,
            )

        _semantic_delayed_claim = bool(cfg.kwargs.get("semantic_delayed_claim", False))
        _semantic_cheb_gr_claim = bool(cfg.semantic_cheb_gr_claim)
        _semantic_bidirectional = bool(cfg.kwargs.get("semantic_bidirectional", False))
        _semantic_gpu_relink_gate = (
            bool(cfg.semantic_gpu_relink_gate)
            if cfg.semantic_gpu_relink_gate is not None
            else _semantic_cheb_gr_claim
        )
        _use_python_relinker = (
            cfg.force_python_relinker
            or cfg.semantic_rerank_mode != "mean"
            or _semantic_gpu_relink_gate
        )
        _relinker_cls = (
            PythonSemanticRelinker if _use_python_relinker else SemanticRelinker
        )
        _relinker_common_kwargs: dict = dict(
            sim_threshold=cfg.kwargs.get("semantic_threshold", 0.90),
            ttl=cfg.kwargs.get("semantic_ttl", 45),
            ema_beta=cfg.kwargs.get("semantic_ema", 0.83),
            spatial_gate=cfg.kwargs.get("semantic_spatial_gate", 0.20),
            min_lost_frames=cfg.kwargs.get("semantic_min_lost_frames", 2),
            min_iou=cfg.kwargs.get("semantic_min_iou", 0.20),
            mahalanobis_threshold=cfg.kwargs.get("semantic_mahalanobis_threshold", 0.0),
            kalman_gate=cfg.kwargs.get("semantic_kalman_gate", False),
            kalman_chi2=cfg.kwargs.get("semantic_kalman_chi2", 9.4877),
            kalman_penalty_weight=cfg.kwargs.get("semantic_kalman_penalty_weight", 0.0),
            kalman_dir_min_cos=cfg.kwargs.get("semantic_kalman_dir_min_cos", -1.0),
            kalman_dir_min_speed=cfg.kwargs.get("semantic_kalman_dir_min_speed", 1.0),
            kalman_person_height_m=cfg.kwargs.get(
                "semantic_kalman_person_height_m", 0.0
            ),
            kalman_accel_long=cfg.kwargs.get("semantic_kalman_accel_long", 2.0),
            kalman_accel_lat=cfg.kwargs.get("semantic_kalman_accel_lat", 1.0),
            kalman_fps=_resolve_kalman_fps(
                cfg.kwargs.get("semantic_kalman_fps", 0.0),
                Path(cfg.data_root) / cfg.split / seq,
            ),
            kalman_max_speed_mps=cfg.kwargs.get("semantic_kalman_max_speed_mps", 0.0),
            buffer_size=cfg.semantic_buffer_size,
            min_consistency=cfg.semantic_min_consistency,
            rerank_mode=cfg.semantic_rerank_mode,
            reciprocal_margin=cfg.semantic_reciprocal_margin,
            clean_score_threshold=cfg.semantic_clean_score_threshold,
            clean_margin_ratio=cfg.semantic_clean_margin_ratio,
            clean_min_aspect=cfg.semantic_clean_min_aspect,
            clean_max_aspect=cfg.semantic_clean_max_aspect,
            strict_sim_threshold=cfg.semantic_strict_sim_threshold,
            w_sim_base=cfg.semantic_w_sim_base,
            w_iou_base=cfg.semantic_w_iou_base,
            w_maha_base=cfg.semantic_w_maha_base,
            shift_ambiguity=cfg.semantic_shift_ambiguity,
            shift_lost_age=cfg.semantic_shift_lost_age,
            iou_weight=cfg.semantic_iou_weight,
            mahalanobis_weight=cfg.semantic_mahalanobis_weight,
            dynamic_margin_crowd=cfg.semantic_dynamic_margin_crowd,
            dynamic_margin_age=cfg.semantic_dynamic_margin_age,
            debug=cfg.kwargs.get("semantic_debug", False),
            exp_density_gating=cfg.semantic_exp_density_gating,
            exp_density_k=cfg.semantic_exp_density_k,
            exp_density_eta=cfg.semantic_exp_density_eta,
        )
        if _semantic_delayed_claim or _semantic_cheb_gr_claim:
            _relinker_common_kwargs.update(
                delayed_claim=True,
                claim_warmup_frames=cfg.kwargs.get("semantic_claim_warmup_frames", 3),
            )
        if _semantic_cheb_gr_claim:
            _relinker_common_kwargs.update(
                cheb_gr_claim=_semantic_cheb_gr_claim,
                cheb_gr_max_cost=cfg.semantic_cheb_gr_max_cost,
                cheb_gr_margin=cfg.semantic_cheb_gr_margin,
                cheb_gr_min_head=cfg.semantic_cheb_gr_min_head,
                cheb_gr_pool_frac=cfg.semantic_cheb_gr_pool_frac,
                cheb_gr_min_sim=cfg.semantic_cheb_gr_min_sim,
                cheb_gr_lambda=cfg.cheb_gr_lambda,
                cheb_gr_k2=cfg.cheb_gr_k2,
                cheb_gr_max_fwd=cfg.cheb_gr_max_fwd,
                cheb_gr_fuse_lambda=cfg.cheb_gr_fuse_lambda,
            )
        if _semantic_bidirectional:
            _relinker_common_kwargs.update(
                bidirectional=True,
                bridge_px=cfg.kwargs.get("semantic_bridge_px", 1.5),
                bridge_h_lo=cfg.relink_bridge_h_lo,
                bridge_h_hi=cfg.relink_bridge_h_hi,
            )
        if _use_python_relinker:
            _relinker_common_kwargs.update(
                experimental_mode=str(
                    cfg.kwargs.get("semantic_experimental_mode", "standard")
                ),
                appearance_first_sim_threshold=float(
                    cfg.kwargs.get("semantic_appearance_first_sim_threshold", 0.95)
                ),
                appearance_first_margin=float(
                    cfg.kwargs.get("semantic_appearance_first_margin", 0.03)
                ),
                # Motion-based relinking (PythonSemanticRelinker only)
                motion_vel_alpha=cfg.kwargs.get("motion_vel_alpha", 0.3),
                motion_acc_alpha=cfg.kwargs.get("motion_acc_alpha", 0.15),
                motion_min_observations=cfg.kwargs.get("motion_min_observations", 2),
                motion_w_iou=cfg.kwargs.get("motion_w_iou", 0.3),
                motion_consistency_check=cfg.kwargs.get(
                    "motion_consistency_check", True
                ),
                motion_consistency_tol=cfg.kwargs.get("motion_consistency_tol", 2.0),
                motion_enable_motion_only=cfg.kwargs.get(
                    "motion_enable_motion_only", not _semantic_cheb_gr_claim
                ),
                motion_motion_only_lost_frames=cfg.kwargs.get(
                    "motion_motion_only_lost_frames", 5
                ),
                motion_motion_only_iou_threshold=cfg.kwargs.get(
                    "motion_motion_only_iou_threshold", 0.15
                ),
                motion_motion_only_min_lost_frames=cfg.kwargs.get(
                    "motion_motion_only_min_lost_frames", 1
                ),
                gpu_relink_gate=_semantic_gpu_relink_gate,
                gpu_relink_gate_graph=cfg.semantic_gpu_relink_gate_graph,
                gpu_relink_gate_init_query_cap=(
                    cfg.semantic_gpu_relink_gate_init_query_cap
                ),
                gpu_relink_gate_init_candidate_cap=(
                    cfg.semantic_gpu_relink_gate_init_candidate_cap
                ),
                cheb_gr_graph_init_cap=cfg.semantic_cheb_gr_graph_init_cap,
            )
        relinker = None
        if cfg.use_semantic_mode:
            try:
                relinker = _relinker_cls(**_relinker_common_kwargs)
            except TypeError:
                if (
                    not _semantic_delayed_claim
                    or _relinker_cls is PythonSemanticRelinker
                ):
                    raise
                _fallback_kwargs = dict(_relinker_common_kwargs)
                _fallback_kwargs.update(
                    experimental_mode=str(
                        cfg.kwargs.get("semantic_experimental_mode", "standard")
                    ),
                    appearance_first_sim_threshold=float(
                        cfg.kwargs.get("semantic_appearance_first_sim_threshold", 0.95)
                    ),
                    appearance_first_margin=float(
                        cfg.kwargs.get("semantic_appearance_first_margin", 0.03)
                    ),
                    motion_vel_alpha=cfg.kwargs.get("motion_vel_alpha", 0.3),
                    motion_acc_alpha=cfg.kwargs.get("motion_acc_alpha", 0.15),
                    motion_min_observations=cfg.kwargs.get(
                        "motion_min_observations", 2
                    ),
                    motion_w_iou=cfg.kwargs.get("motion_w_iou", 0.3),
                    motion_consistency_check=cfg.kwargs.get(
                        "motion_consistency_check", True
                    ),
                    motion_consistency_tol=cfg.kwargs.get(
                        "motion_consistency_tol", 2.0
                    ),
                    motion_enable_motion_only=cfg.kwargs.get(
                        "motion_enable_motion_only", True
                    ),
                    motion_motion_only_lost_frames=cfg.kwargs.get(
                        "motion_motion_only_lost_frames", 5
                    ),
                    motion_motion_only_iou_threshold=cfg.kwargs.get(
                        "motion_motion_only_iou_threshold", 0.15
                    ),
                    motion_motion_only_min_lost_frames=cfg.kwargs.get(
                        "motion_motion_only_min_lost_frames", 1
                    ),
                )
                print(
                    "ℹ️  [Semantic] C++ relinker lacks delayed-claim kwargs; "
                    "falling back to Python relinker"
                )
                relinker = PythonSemanticRelinker(**_fallback_kwargs)

        # Online handover is experimental and default-off. The output-layer
        # --cheb-gr-offline-handover path below is the supported handover route;
        # do not enable the live feedback loop just because ReID embeddings are
        # present. Accepted in both "tracker" (embeddings also drive association)
        # and "extract" (embeddings feed only the handover; the tracker output is
        # identical to reid-off, giving the handover the same fragmentation base
        # as the offline reid-off run).
        _tracker_online_handover = bool(
            getattr(cfg, "kwargs", {}).get("tracker_online_handover", False)
        )
        if (
            relinker is None
            and _tracker_online_handover
            and cfg.reid_mode in ("tracker", "extract")
            and _relinker_cls is not PythonSemanticRelinker
        ):
            try:
                relinker = _relinker_cls(**_relinker_common_kwargs)
            except TypeError:
                pass

        if relinker is not None:
            if (
                _CppIdentityResolver is not None
                and _LIFECYCLE_CLS is not None
                and isinstance(lifecycle_merger, _LIFECYCLE_CLS)
                and not isinstance(relinker, PythonSemanticRelinker)
            ):
                identity_resolver = _CppIdentityResolver(relinker, lifecycle_merger)
            else:
                identity_resolver = IdentityResolver(relinker, lifecycle_merger)
        else:
            identity_resolver = None

        # Live Cheb-GR handover: configure the C++ relinker for per-frame
        # handover decisions only when the experimental live feedback loop is
        # explicitly requested. Plain tracker-mode ReID must not enable this.
        _has_reid = cfg.reid_mode != "off"
        self.live_evfifo = None
        if (
            _tracker_online_handover
            and _has_reid
            and relinker is not None
            and not isinstance(relinker, PythonSemanticRelinker)
        ):
            ho_max_cost = getattr(cfg, "cheb_gr_online_max_cost", 0.45)
            ho_margin = getattr(cfg, "cheb_gr_online_margin", 0.0)
            ho_max_gap = getattr(cfg, "cheb_gr_merge_max_gap", 60)
            ho_decide_n = getattr(cfg, "cheb_gr_online_decide_n", 5)
            ho_min_head = getattr(cfg, "cheb_gr_online_min_head", 1)
            # Live evfifo bank mode: build the offline evfifo-5-20-w3 bank
            # incrementally during tracking and run the decision at sequence end
            # (bounded VRAM, no disk re-read). The C++ relinker's own per-frame
            # handover is disabled to avoid double application — it still
            # accumulates fed embeddings harmlessly (enabled=False).
            _live_bank = bool(
                getattr(cfg, "kwargs", {}).get("cheb_gr_online_live_bank", False)
            )
            relinker.set_handover_params(
                enabled=not _live_bank,
                max_cost=ho_max_cost,
                margin=ho_margin,
                max_gap=ho_max_gap,
                decide_n=ho_decide_n,
                min_head=ho_min_head,
                pool_frac=getattr(cfg, "cheb_gr_pool_frac", 0.3),
                cheb_lambda=getattr(cfg, "cheb_gr_lambda", 2.0),
                k2=getattr(cfg, "cheb_gr_k2", 6),
                max_fwd=getattr(cfg, "cheb_gr_max_fwd", 50),
                fuse_lambda=getattr(cfg, "cheb_gr_fuse_lambda", 0.3),
            )
            print(
                f"🔗 Online handover enabled: decide_n={ho_decide_n} max_cost={ho_max_cost} margin={ho_margin}"
            )

            if _live_bank:
                from .streaming_handover import LiveEvfifoHandover

                _lb_kwargs = getattr(cfg, "kwargs", {})
                self.live_evfifo = LiveEvfifoHandover(
                    fifo_n=int(_lb_kwargs.get("cheb_gr_online_bank_fifo_n", 20)),
                    stride=int(_lb_kwargs.get("cheb_gr_online_bank_stride", 5)),
                    decide_n=int(ho_decide_n),
                    preocc_window=int(
                        _lb_kwargs.get("cheb_gr_online_preocc_window", 3)
                    ),
                    appearance_occlusion_cov=getattr(
                        cfg, "appearance_occlusion_cov", 0.4
                    ),
                    neighbor_iou_max=getattr(
                        cfg, "cheb_gr_online_neighbor_iou_max", 0.0
                    ),
                )
                # Enable the crop ring for raw-crop extraction at finalize.
                _ring_depth = int(_lb_kwargs.get("cheb_gr_online_requery_ring_n", 20))
                _ring_cap = int(
                    _lb_kwargs.get("cheb_gr_online_requery_ring_capacity", 4096)
                )
                if (
                    hasattr(perception_pipeline, "enable_crop_ring")
                    and perception_pipeline is not None
                ):
                    perception_pipeline.enable_crop_ring(_ring_cap, _ring_depth)
                print(
                    "🧬 Live handover bank enabled: "
                    f"fifo={self.live_evfifo.fifo_n} "
                    f"stride={self.live_evfifo.stride} "
                    f"w{self.live_evfifo.preocc_window} "
                    f"ring={_ring_cap}x{_ring_depth} "
                    "(C++ per-frame handover disabled)"
                )

            # Borderline re-query: attach the PerceptionPipeline crop ring as the
            # relinker's ReidCropStore so a flippable decision can re-extract
            # dense recent-tail banks. The ring is keyed by the same track ids
            # fed to feed_frame_embeddings (= the handover archive tids), so
            # ByteTrack's never-reused ids double as track_uids. Default-off
            # (requery_band == 0).
            _ho_kwargs = getattr(cfg, "kwargs", {})
            _ho_requery_band = float(_ho_kwargs.get("cheb_gr_online_requery_band", 0.0))
            if (
                _ho_requery_band > 0.0
                and hasattr(relinker, "set_crop_store")
                and perception_pipeline is not None
                and hasattr(perception_pipeline, "enable_crop_ring")
            ):
                _ring_depth = int(_ho_kwargs.get("cheb_gr_online_requery_ring_n", 20))
                _ring_cap = int(
                    _ho_kwargs.get("cheb_gr_online_requery_ring_capacity", 4096)
                )
                _ho_requery_top = int(_ho_kwargs.get("cheb_gr_online_requery_top", 0))
                # The live-bank block above may already have enabled the ring
                # (same kwargs); re-enabling would drop and reallocate it.
                if not perception_pipeline.crop_ring_enabled():
                    perception_pipeline.enable_crop_ring(_ring_cap, _ring_depth)
                relinker.set_crop_store(perception_pipeline)
                relinker.set_handover_requery(_ho_requery_band, _ho_requery_top)
                print(
                    f"🔎 Borderline re-query enabled: band={_ho_requery_band} "
                    f"top={_ho_requery_top} ring={_ring_cap}x{_ring_depth}"
                )

        seq_path = Path(cfg.data_root) / cfg.split / seq
        config = configparser.ConfigParser()
        config.read(seq_path / "seqinfo.ini")
        w_orig = config.getint("Sequence", "imWidth")
        h_orig = config.getint("Sequence", "imHeight")
        frame_end = min(max_frames or int(1e9), config.getint("Sequence", "seqLength"))
        seq_fps = config.getint("Sequence", "frameRate", fallback=30)

        # F-1: Per-sequence adaptive params — scale temporal params by fps/30
        seq_reid_interval = cfg.reid_interval
        _track_buffer_base = int(cfg.kwargs.get("track_buffer", 30))
        seq_track_buffer = _track_buffer_base
        if cfg.per_seq_adapt and seq_fps != 30:
            fps_scale = seq_fps / 30.0
            seq_reid_interval = max(1, round(cfg.reid_interval * fps_scale))
            seq_track_buffer = max(10, round(_track_buffer_base * fps_scale))
        if hasattr(detector, "set_whole_graph_img_dims"):
            detector.set_whole_graph_img_dims(h_orig, w_orig)
        detector.tracker.set_frame_size(w_orig, h_orig)
        _gmc_frame_buf = None
        if _use_direct_gmc:
            _gmc_frame_buf = torch.zeros(
                3, h_orig, w_orig, dtype=torch.float32, device="cuda"
            )
        detector.tracker.set_quality_params(
            enabled=cfg.detection_quality_scaling,
            w_aspect=cfg.detection_quality_w_aspect,
            w_center=cfg.detection_quality_w_center,
            w_area=cfg.detection_quality_w_area,
        )
        detector.tracker.set_params(
            track_thresh=cfg.track_thresh,
            high_thresh=cfg.high_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=seq_track_buffer,
            mid_thresh=cfg.mid_thresh,
            confirm_streak=int(cfg.kwargs.get("confirm_streak", 1)),
            confirm_score_thresh=float(cfg.kwargs.get("confirm_score_thresh", 0.0)),
            adaptive_confirmation=bool(cfg.kwargs.get("adaptive_confirmation", False)),
            new_track_thresh=cfg.new_track_thresh,
            kalman_adapt_mode=cfg.kalman_adapt_mode,
            r_scale=cfg.kalman_r_scale,
            vel_dir_weight=cfg.vel_dir_weight,
            fuse_score_weight=cfg.fuse_score_weight,
            stage2_match_thresh=cfg.stage2_match_thresh,
            birth_low_score_thresh=cfg.birth_low_score_thresh,
            birth_prox_norm_thresh=cfg.birth_prox_norm_thresh,
        )
        detector.tracker.set_oao_params(
            cfg.oao_tau,
            cfg.oao_contest_thresh,
            cfg.oao_score_w,
            cfg.oao_occ_mode,
            cfg.oao_crowd_radius,
            cfg.oao_height_gate,
            cfg.oao_foot_gate,
            cfg.oao_ramp_frames,
        )
        detector.tracker.set_occ_params(
            enabled=cfg.occ_state_enabled,
            iou_thresh=cfg.occ_iou_thresh,
            foot_gap=cfg.occ_foot_gap,
            ttl=cfg.occ_ttl,
            cost_weight=cfg.occ_cost_weight,
        )
        association_scoring_mode = (
            str(getattr(cfg, "association_scoring_mode", "baseline")).strip().lower()
        )
        if association_scoring_mode not in {"baseline", "energy"}:
            raise ValueError(
                f"unknown association_scoring_mode: {association_scoring_mode}"
            )
        association_energy_enabled = association_scoring_mode == "energy"
        if getattr(cfg, "multiplicative_cost", False) or association_energy_enabled:
            detector.tracker.set_multiplicative_cost(enabled=True)
            lam = float(getattr(cfg, "sinkhorn_lambda", 30.0))
            detector.tracker.set_sinkhorn_lambda(lam)
            stab_w = float(getattr(cfg, "stability_cost_w", 0.0))
            if stab_w > 0:
                setter = getattr(detector.tracker, "set_stability_cost_w", None)
                if setter:
                    setter(stab_w)
        energy_setter = getattr(detector.tracker, "set_association_energy_params", None)
        if energy_setter:
            energy_setter(
                association_energy_enabled,
                float(getattr(cfg, "assoc_score_cost_w", 0.0)),
                float(getattr(cfg, "assoc_height_cost_w", 0.0)),
            )
        active_tracker_thresholds = (
            cfg.track_thresh,
            cfg.mid_thresh,
            cfg.new_track_thresh,
        )

        pool = AdaptiveFramePool(h_orig, w_orig)
        if os.environ.get("SACCADE_NV12_BUFFER") == "1":
            pool.use_nv12 = True
        nv12_direct_from_hwc = pool.use_nv12 and not cfg.preprocess_modes
        # Detection owns its input buffers until its CUDA stream signals ready.
        # Keep two independent pools so launching frame N+1 cannot overwrite the
        # frame N pixels still consumed by GMC/ReID/tracking on the main stream.
        double_buffer_pools = [pool]
        double_buffer_stream = None
        double_buffer_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if _double_buffer_eligible(cfg, detector, profile_stages):
            next_pool = AdaptiveFramePool(h_orig, w_orig)
            next_pool.use_nv12 = pool.use_nv12
            double_buffer_pools.append(next_pool)
            double_buffer_stream = torch.cuda.Stream()
            for _ in range(2):
                double_buffer_events.append(
                    (
                        torch.cuda.Event(enable_timing=False),
                        torch.cuda.Event(enable_timing=False),
                    )
                )
        # SACCADE_GPU_DECODE=1 routes JPEG decode to the GPU's NVJPG hardware
        # engine (torchvision/nvJPEG) instead of CPU (DALI), offloading decode
        # off the CPU. Both yield [H, W, C] uint8 CUDA frames.
        if os.environ.get("SACCADE_GPU_DECODE") == "1":
            streamer: Any = TorchvisionGpuStreamer(seq_path / "img1")
        else:
            streamer = DALIStreamerStream(seq_path / "img1")
        stream_iter = iter(streamer)
        results_lines, frame_latencies = [], []
        output_appearance_bank = (
            OutputAppearanceBank(
                max_samples=cfg.post_lifecycle_appearance_max_samples,
                min_score=cfg.post_lifecycle_appearance_min_score,
                min_consistency=cfg.post_lifecycle_appearance_min_consistency,
            )
            if cfg.post_lifecycle_appearance_gate
            else None
        )
        primary_appearance_bank = (
            TrackAppearanceBank(
                k=cfg.appearance_bank_size,
                min_score=cfg.appearance_bank_min_score,
                min_iou=cfg.appearance_bank_min_iou,
                consistency_threshold=cfg.appearance_bank_consistency_threshold,
                high_quality_min_score=cfg.appearance_bank_high_quality_min_score,
                min_aspect=cfg.appearance_bank_min_aspect,
                max_aspect=cfg.appearance_bank_max_aspect,
                bank_weighted_mean=cfg.bank_weighted_mean,
                exp_velocity_aligned_bank=cfg.exp_velocity_aligned_bank,
            )
            if cfg.appearance_bank_enabled
            else None
        )
        dynamic_reid = (
            DynamicReIDController(
                history_size=max(2, int(cfg.kwargs.get("reid_history_size", 5))),
                mode=str(cfg.kwargs.get("reid_trigger_mode", "event_any")),
                long_memory_decay=float(cfg.kwargs.get("reid_long_memory_decay", 0.80)),
                long_memory_trigger=float(
                    cfg.kwargs.get("reid_long_memory_trigger", 1.25)
                ),
                score_decay=float(cfg.kwargs.get("reid_score_decay", 0.80)),
                score_threshold=float(cfg.kwargs.get("reid_score_threshold", 2.0)),
                score_threshold_low=float(
                    cfg.kwargs["reid_score_threshold_low"]
                    if cfg.kwargs.get("reid_score_threshold_low") is not None
                    else 2.0
                ),
                weight_new=float(cfg.kwargs.get("reid_weight_new", 1.0)),
                weight_lost=float(cfg.kwargs.get("reid_weight_lost", 1.4)),
                weight_geom=float(cfg.kwargs.get("reid_weight_geom", 0.5)),
                weight_conf=float(cfg.kwargs.get("reid_weight_conf", 0.5)),
                birth_death_boost=float(cfg.kwargs.get("reid_birth_death_boost", 1.0)),
                lost_age_cap=int(cfg.kwargs.get("reid_lost_age_cap", 30)),
                unstable_shift_weight=float(
                    cfg.kwargs.get("reid_unstable_shift_weight", 1.0)
                ),
                unstable_iou_weight=float(
                    cfg.kwargs.get("reid_unstable_iou_weight", 1.0)
                ),
                conf_jitter_gate=float(cfg.kwargs.get("reid_conf_jitter_gate", 0.10)),
                trigger_persist_frames=int(
                    cfg.kwargs.get("reid_trigger_persist_frames", 2)
                ),
                cooldown_frames=int(cfg.kwargs.get("reid_cooldown_frames", 4)),
                birth_death_lost_min=float(
                    cfg.kwargs.get("reid_birth_death_lost_min", 0.1)
                ),
            )
            if cfg.need_reid_enabled
            else None
        )
        # Update workbench with dynamic_reid (created after Workbench init)
        if wb is not None:
            wb.dynamic_reid = dynamic_reid

        _consec_birth_window: deque[torch.Tensor] = deque(
            maxlen=max(1, cfg.birth_consecutive_frames - 1)
        )
        _multi_birth_manager = (
            MultiSignalBirthManager(
                new_track_thresh=cfg.new_track_thresh,
                min_score=cfg.multi_birth_min_score,
                min_frames=cfg.multi_birth_min_frames,
                target_motion_px=cfg.multi_birth_target_motion,
                evidence_threshold=cfg.multi_birth_evidence_threshold,
                iou_match=cfg.multi_birth_iou_match,
                ttl_frames=cfg.multi_birth_ttl_frames,
                w_score=cfg.multi_birth_w_score,
                w_motion=cfg.multi_birth_w_motion,
                w_quality=cfg.multi_birth_w_quality,
                w_streak=cfg.multi_birth_w_streak,
                min_aspect=cfg.multi_birth_min_aspect,
                max_area_px=cfg.multi_birth_max_area_px,
            )
            if cfg.multi_birth_enabled
            else None
        )
        # P5-4: scene-adaptive policy — classifies scene from first N frames and
        # applies seq-local narrow_person_score_bonus for crowded_narrow scenes.
        _scene_policy = (
            SceneAdaptivePolicy(
                window=cfg.scene_adapt_window,
                crowd_box_thresh=cfg.scene_adapt_crowd_thresh,
                narrow_aspect_thresh=cfg.scene_adapt_narrow_aspect_thresh,
                narrow_width_thresh=cfg.scene_adapt_narrow_width_thresh,
            )
            if cfg.scene_adapt_enabled
            else None
        )
        # Start with 0 bonus; if scene_adapt disabled, use the configured value directly.
        seq_narrow_bonus: float = (
            0.0 if cfg.scene_adapt_enabled else cfg.narrow_person_score_bonus
        )
        start_time = time.time()
        warmup_frames = int(cfg.kwargs.get("warmup_frames", 50))
        seq_stage_totals = OrderedDict(
            (name, 0.0) for name in overall_stage_totals.keys()
        )
        seq_stage_samples = OrderedDict((name, []) for name in top_level_stage_names)
        seq_native_reid_samples = OrderedDict(
            (name, []) for name in native_reid_breakdown_names
        )
        seq_gmc_samples = OrderedDict((name, []) for name in gmc_breakdown_names)
        seq_segment_samples: "OrderedDict[str, list[float]]" = OrderedDict(
            (name, []) for name in segment_breakdown_names
        )
        seq_post_counts = OrderedDict(
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
        seq_tile_diag = OrderedDict(
            (name, 0)
            for name in (
                "frames_tiled",
                "pre_merge_seam_boxes",
                "post_merge_seam_boxes",
                "merged_clusters",
                "merged_members",
                "merged_outputs",
            )
        )
        lazy_reid_prev_embeddings: dict[int, torch.Tensor] = {}
        tracker_result_buffers = detector.tracker.allocate_result_buffers(
            device=pool.frame_buffer.device
        )
        _TRACK_RESULT_CAP = int(getattr(detector.tracker, "max_objects", 2048))
        _pinned_result_bufs: dict[str, torch.Tensor] = {
            "boxes": torch.empty(
                (_TRACK_RESULT_CAP, 4), dtype=torch.float32, pin_memory=True
            ),
            "scores": torch.empty(
                (_TRACK_RESULT_CAP,), dtype=torch.float32, pin_memory=True
            ),
            "ids": torch.empty(
                (_TRACK_RESULT_CAP,), dtype=torch.int32, pin_memory=True
            ),
            "classes": torch.empty(
                (_TRACK_RESULT_CAP,), dtype=torch.int32, pin_memory=True
            ),
            "det_idx": torch.empty(
                (_TRACK_RESULT_CAP,), dtype=torch.int32, pin_memory=True
            ),
            "count": torch.empty((), dtype=torch.int32, pin_memory=True),
        }
        # ── double-buffer tracker output pipelining ────────────────────────
        # Parity-slotted pinned buffers so the D2H for frame N-1 can complete
        # while the GPU runs tracker(N).  Emit/relink(N-1) executes at the
        # start of frame N's iteration, before tracker(N) is submitted,
        # preserving the relink→tracker ordering requirement.
        _db_tracker_out_pinned: list[dict[str, torch.Tensor]] = []
        _db_tracker_out_events: list[torch.cuda.Event] = []
        _db_tracker_out_fids: list[int] = [0, 0]
        if double_buffer_stream is not None:
            for _ in range(2):
                _db_tracker_out_pinned.append(
                    {
                        "boxes": torch.empty(
                            (_TRACK_RESULT_CAP, 4),
                            dtype=torch.float32,
                            pin_memory=True,
                        ),
                        "scores": torch.empty(
                            (_TRACK_RESULT_CAP,),
                            dtype=torch.float32,
                            pin_memory=True,
                        ),
                        "ids": torch.empty(
                            (_TRACK_RESULT_CAP,),
                            dtype=torch.int32,
                            pin_memory=True,
                        ),
                        "classes": torch.empty(
                            (_TRACK_RESULT_CAP,),
                            dtype=torch.int32,
                            pin_memory=True,
                        ),
                        "det_idx": torch.empty(
                            (_TRACK_RESULT_CAP,),
                            dtype=torch.int32,
                            pin_memory=True,
                        ),
                        "count": torch.empty((), dtype=torch.int32, pin_memory=True),
                    }
                )
                _db_tracker_out_events.append(torch.cuda.Event(enable_timing=False))
        _use_pinned_materialize = not cfg.pipeline_relink
        _defer_emit = (
            relinker is None
            and id_stability_filter is None
            and primary_appearance_bank is None
            and dynamic_reid is None
        )
        _shared_gmc_warp = torch.zeros(6, dtype=torch.float32, device="cuda")
        _NMS_FIXED_N = int(getattr(detector.tracker, "max_assoc", 1024))
        _post_bufs: dict[str, torch.Tensor] = {
            "boxes": torch.empty((_NMS_FIXED_N, 4), dtype=torch.float32, device="cuda"),
            "scores": torch.empty((_NMS_FIXED_N,), dtype=torch.float32, device="cuda"),
            "classes": torch.empty((_NMS_FIXED_N,), dtype=torch.int32, device="cuda"),
            "suspect": torch.empty((_NMS_FIXED_N,), dtype=torch.bool, device="cuda"),
        }
        _nms_in: dict[str, torch.Tensor] = {
            "boxes": torch.empty((_NMS_FIXED_N, 4), dtype=torch.float32, device="cuda"),
            "scores": torch.empty((_NMS_FIXED_N,), dtype=torch.float32, device="cuda"),
            "classes": torch.empty((_NMS_FIXED_N,), dtype=torch.int32, device="cuda"),
        }

        gtu: Any = None
        # GraphedTrackerUpdate.copy_inputs does not feed per-detection embeddings,
        # so the captured graph path cannot do appearance association / relink.
        # Fall back to direct update_into (which passes embeddings) when relink is on.
        if cfg.kwargs.get("use_tracker_graph", False) and not cfg.relink_enabled:
            from saccade.perception.tracking.tracker_gpu import GraphedTrackerUpdate

            gtu = GraphedTrackerUpdate(detector.tracker)
            print(f"🕯️ [TrackerGraph] Captured tracker update for seq {seq}")
        elif cfg.relink_enabled and cfg.kwargs.get("use_tracker_graph", False):
            print(
                "ℹ️  [Relink] tracker graph disabled (relink needs embeddings via update_into)"
            )

        # Inter-frame pipelining: relink_write for frame N runs in background while
        # frame N+1 runs detect+postprocess on the main thread. All GPU tensors are
        # pre-materialized to CPU before submit to avoid CUDA stream conflicts.
        def _collect_output_metadata(
            _resolved_tracks: list,
        ) -> dict[int, dict[str, float | int]]:
            output_by_local: dict[int, dict[str, float | int]] = {}
            for _track in _resolved_tracks:
                _global_tid = global_id_mapper.map(seq, _track.resolved_track_id)
                output_by_local[int(_track.local_track_id)] = {
                    "output_local_track_id": int(_track.local_track_id),
                    "output_track_id": int(_global_tid),
                    "output_score": float(_track.score),
                    "output_x1": float(_track.box[0]),
                    "output_y1": float(_track.box[1]),
                    "output_x2": float(_track.box[2]),
                    "output_y2": float(_track.box[3]),
                }
            return output_by_local

        def _annotate_birth_events(
            _frame_birth_events: list[dict[str, float | int | str | bool]],
            *,
            _det_idx_to_local_id: dict[int, int],
            _output_by_local: dict[int, dict[str, float | int]],
        ) -> None:
            if not _frame_birth_events:
                return
            for _event in _frame_birth_events:
                _det_idx = int(_event["det_idx"])
                _local_id = _det_idx_to_local_id.get(_det_idx, -1)
                _meta = _output_by_local.get(_local_id)
                if _meta is None:
                    _event.update(
                        {
                            "output_emitted": False,
                            "output_local_track_id": _local_id,
                            "output_track_id": -1,
                            "output_score": float("nan"),
                            "output_x1": float("nan"),
                            "output_y1": float("nan"),
                            "output_x2": float("nan"),
                            "output_y2": float("nan"),
                        }
                    )
                else:
                    _event.update(
                        {
                            "output_emitted": True,
                            **_meta,
                        }
                    )
                debug_birth_rows.append(_event)

        def _append_birth_event_rows(
            _frame_birth_events: list[dict[str, float | int | str | bool]],
            *,
            _policy: str,
            _det_indices: torch.Tensor,
            _score_before: torch.Tensor,
            _score_after: torch.Tensor,
            _boxes: torch.Tensor,
        ) -> None:
            if _det_indices.numel() == 0:
                return
            _idx_cpu = _det_indices.detach().to(torch.int64).cpu().tolist()
            _before_cpu = _score_before.detach().to(torch.float32).cpu().tolist()
            _after_cpu = _score_after.detach().to(torch.float32).cpu().tolist()
            _boxes_cpu = _boxes.detach().to(torch.float32).cpu().tolist()
            for _det_idx, _before, _after, _box in zip(
                _idx_cpu,
                _before_cpu,
                _after_cpu,
                _boxes_cpu,
            ):
                _frame_birth_events.append(
                    {
                        "seq": seq,
                        "frame": int(self.current_frame_id),
                        "policy": _policy,
                        "det_idx": int(_det_idx),
                        "score_before": float(_before),
                        "score_after": float(_after),
                        "x1": float(_box[0]),
                        "y1": float(_box[1]),
                        "x2": float(_box[2]),
                        "y2": float(_box[3]),
                        "w": float(_box[2] - _box[0]),
                        "h": float(_box[3] - _box[1]),
                    }
                )

        def _bg_relink_write(
            _frame_id: int,
            _track_results: "HostTrackResultView",
            _host_batch: "HostTrackBatch",
            _fused_boxes: torch.Tensor,
            _fused_scores: torch.Tensor,
            _geometry_suspect_mask: torch.Tensor,
            _embeddings: "torch.Tensor | None",
            _gmc_warp: "torch.Tensor | None",
            _motion_candidate_ids: "list[int]",
            _motion_snapshots: "list | None",
            _prev_track_ids: "set[int]",
        ) -> "tuple[list[str], set[int], dict[int, int], dict[int, dict[str, float | int]]]":
            # motion snapshot update (pre-computed in main thread)
            if relinker and _motion_candidate_ids and _motion_snapshots is not None:
                relinker.update_motion_snapshots(_motion_snapshots, _frame_id)

            prepared_candidates = _prepare_track_candidates(
                frame_id=_frame_id,
                track_results=_track_results,
                host_batch=_host_batch,
                person_class=cfg.person_class,
                track_person_only=cfg.track_person_only,
                geometry_suspect_support=cfg.geometry_suspect_support,
                geometry_suspect_support_score=cfg.geometry_suspect_support_score,
                id_stability_filter=id_stability_filter,
                embeddings=_embeddings,
                fused_boxes=_fused_boxes,
                fused_scores=_fused_scores,
                geometry_suspect_mask=_geometry_suspect_mask,
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
            if (
                int(os.environ.get("SACCADE_GPU_RELINK_GATE", "1") or "1")
                and relinker is not None
                and hasattr(relinker, "build_gate_table")
            ):
                tracker_kwargs = {}
                if hasattr(detector.tracker, "get_gpu_buffers"):
                    states, covs, tids, uids, maxn = detector.tracker.get_gpu_buffers()
                    tracker_kwargs = dict(
                        tracker_states=int(states),
                        tracker_covs=int(covs),
                        tracker_tids=int(tids),
                        tracker_max_objs=int(maxn),
                    )
                # Row-aligned query embeddings: one row per candidate in the
                # same order as local_track_id / box, with a zero row where a
                # candidate has no embedding. Filtering out the None rows would
                # desync the row index from raw_ids and trip the n_q==n_query
                # guard in build_gate_table, silently disabling the GPU scoring
                # path for the whole frame. A zero row yields sim=0 for that
                # query, so an embedding-less detection is simply not relinked.
                _cand_embs = [c.embedding for c in prepared_candidates]
                _ref_emb = next((e for e in _cand_embs if e is not None), None)
                _query_embs = None
                if _ref_emb is not None:
                    _zero_emb = torch.zeros_like(_ref_emb)
                    _query_embs = (
                        torch.stack(
                            [e if e is not None else _zero_emb for e in _cand_embs]
                        )
                        .float()
                        .cpu()
                    )
                relinker.build_gate_table(
                    [c.local_track_id for c in prepared_candidates],
                    [c.box for c in prepared_candidates],
                    _frame_id,
                    w_orig,
                    h_orig,
                    **tracker_kwargs,
                    query_embs=_query_embs,
                )
            resolved_tracks = _resolve_frame_tracks(
                frame_id=_frame_id,
                frame_w=w_orig,
                frame_h=h_orig,
                prepared_candidates=prepared_candidates,
                lifecycle_merger=lifecycle_merger,
                identity_resolver=identity_resolver,
            )
            frame_result_lines = _emit_resolved_tracks(
                seq=seq,
                frame_id=_frame_id,
                frame_w=w_orig,
                frame_h=h_orig,
                resolved_tracks=resolved_tracks,
                global_id_mapper=global_id_mapper,
                output_appearance_bank=output_appearance_bank,
            )
            det_idx_to_local_id = {
                int(_det_idx): int(_local_id)
                for _local_id, _det_idx in zip(
                    _host_batch.ids,
                    _host_batch.det_idx or [],
                )
                if int(_det_idx) >= 0
            }
            output_by_local = _collect_output_metadata(resolved_tracks)
            curr_track_ids = set(_host_batch.ids)
            lifecycle_merger.prune(_frame_id)
            new_prev_track_ids = _finalize_frame_side_effects(
                curr_track_ids=curr_track_ids,
                prev_track_ids=_prev_track_ids,
                relinker=relinker,
                semantic_bank_inject=cfg.semantic_bank_inject,
                primary_appearance_bank=primary_appearance_bank,
                dynamic_reid=dynamic_reid,
                person_observations=_host_batch.person_observations,
                gmc_warp=_gmc_warp,
                gmc_enabled=cfg.gmc_enabled,
            )
            return (
                frame_result_lines,
                new_prev_track_ids,
                det_idx_to_local_id,
                output_by_local,
            )

        # ── bind computed state onto self ──────────────────────────
        self._bridge_enabled = _bridge_enabled
        self.gmc_estimator = gmc_estimator
        self.shared_gmc_warp = _shared_gmc_warp
        self.use_direct_gmc = _use_direct_gmc
        self.gmc_graphable = _gmc_graphable
        self.gmc_cuda_graph = _gmc_cuda_graph
        self.gmc_frame_buf = _gmc_frame_buf
        self.seq_gmc_samples = seq_gmc_samples
        self.seq_stage_totals = seq_stage_totals
        self.pinned_result_bufs = _pinned_result_bufs
        self.use_pinned_materialize = _use_pinned_materialize
        self.defer_emit = _defer_emit
        self.gtu = gtu
        self.nms_in = _nms_in
        self.post_bufs = _post_bufs
        self.nms_fixed_n = _NMS_FIXED_N
        self.relinker = relinker
        self.id_stability_filter = id_stability_filter
        self.primary_appearance_bank = primary_appearance_bank
        self.output_appearance_bank = output_appearance_bank
        self.dynamic_reid = dynamic_reid
        self.lifecycle_merger = lifecycle_merger
        self.identity_resolver = identity_resolver
        self.bg_relink_write = _bg_relink_write
        self.collect_output_metadata = _collect_output_metadata
        self.annotate_birth_events = _annotate_birth_events
        self.active_tracker_thresholds = active_tracker_thresholds
        self._consec_birth_window = _consec_birth_window
        self._multi_birth_manager = _multi_birth_manager
        self._scene_policy = _scene_policy
        self.wb = wb
        self.wb_scene_policy = wb_scene_policy
        self.geometry_scale_state = geometry_scale_state
        self.nv12_direct_from_hwc = nv12_direct_from_hwc
        self.pool = pool
        self.double_buffer_pools = double_buffer_pools
        self.double_buffer_stream = double_buffer_stream
        self.double_buffer_events = double_buffer_events
        self.double_buffer_tracker_out_pinned = _db_tracker_out_pinned
        self.double_buffer_tracker_out_events = _db_tracker_out_events
        self.double_buffer_tracker_out_fids = _db_tracker_out_fids
        # ── deferred emit context (one pending frame at a time) ─────────
        self.db_emit_frame_id: int = 0
        self.db_emit_event: "torch.cuda.Event | None" = None
        self.db_emit_parity: int = 0
        self.db_emit_ctx: dict[str, Any] = {}
        self.stream_iter = stream_iter
        self.frame_end = frame_end
        self.frame_latencies = frame_latencies
        self.results_lines = results_lines
        self.seq_track_buffer = seq_track_buffer
        self.seq_reid_interval = seq_reid_interval
        self.warmup_frames = warmup_frames
        self.seq_stage_samples = seq_stage_samples
        self.seq_segment_samples = seq_segment_samples
        self.seq_native_reid_samples = seq_native_reid_samples
        self.seq_post_counts = seq_post_counts
        self.seq_tile_diag = seq_tile_diag
        self.lazy_reid_prev_embeddings = lazy_reid_prev_embeddings
        self.tracker_result_buffers = tracker_result_buffers
        self.append_birth_event_rows = _append_birth_event_rows
        self.seq_narrow_bonus = seq_narrow_bonus
        self.w_orig = w_orig
        self.h_orig = h_orig
        self.start_time = start_time


def _build_gmc_estimator(
    cfg: Any, profile_stages: bool
) -> tuple[Any, bool, bool, list[Any]]:
    """Construct the per-sequence GMC estimator and its graph-capture flags.

    Picks the C++ cuFFT GMC (default gpu mode), falling back to PyGraphedGMC then
    SparseOpticalFlowGMC if the native extension is unavailable. Returns
    ``(gmc_estimator, use_direct_gmc, gmc_graphable, gmc_cuda_graph)`` where
    gmc_cuda_graph is the single-cell mutable list the frame loop captures into.
    """
    # A8: Uniform CMC & 2D MMD
    gmc_estimator = None
    if cfg.gmc_enabled:
        if cfg.gmc_mode == "gpu":
            # Default: C++ cuFFT GMC, graph-captured in the frame loop below.
            # Falls back to the pure-Python PyGraphedGMC (also graph-capturable)
            # only if the native extension is unavailable.
            try:
                from saccade_tracking_ext import GMC as CppGMC

                gmc_estimator = CppGMC(downscale=cfg.gmc_downscale)
                if hasattr(gmc_estimator, "set_profiling_enabled"):
                    gmc_estimator.set_profiling_enabled(profile_stages)
            except (ImportError, AttributeError):
                try:
                    gmc_estimator = PyGraphedGMC(downscale=cfg.gmc_downscale)
                except Exception:
                    gmc_estimator = SparseOpticalFlowGMC(downscale=cfg.gmc_downscale)
        elif cfg.gmc_mode == "tile":
            # Tile-based phase-correlation similarity GMC (4-DOF: s, θ, tx, ty).
            # Eager Python prototype; falls back to global PCR translation when
            # the affine fit is not confident/plausible (never to identity).
            gmc_estimator = TilePhaseCorrAffineGMC(downscale=cfg.gmc_downscale)
        else:
            gmc_estimator = SparseOpticalFlowGMC(downscale=cfg.gmc_downscale)
    _use_direct_gmc = hasattr(gmc_estimator, "estimate_into_direct")
    # C++ estimate_into_direct is CUDA-graph-capturable (PyGraphedGMC self-graphs
    # via make_graphed_callables, so it is excluded here). Capture lazily on the
    # first eligible frame; replay thereafter. FG mask (variable box count) is
    # not graph-compatible, so those runs stay eager.
    _gmc_graphable = _use_direct_gmc and not isinstance(gmc_estimator, PyGraphedGMC)
    _gmc_cuda_graph = [None]  # mutable for closure capture
    return gmc_estimator, _use_direct_gmc, _gmc_graphable, _gmc_cuda_graph
