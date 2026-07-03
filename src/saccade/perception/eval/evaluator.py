# mypy: ignore-errors
import configparser
import csv
import json
import os
import threading
import time
from collections import OrderedDict, deque
import dataclasses
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from typing import Any

from saccade.perception.box_ops import torch_box_iou_matrix
from .types import (
    HostTrackResultView,
    HostTrackBatch,
)
from .lifecycle import (
    IdStabilityFilter,
    TrackletLifecycleMerger,
)
from .quality import (
    compute_detection_quality_batch as _compute_detection_quality_batch,
)
from .utils import (
    parse_debug_frame_ranges as _parse_debug_frame_ranges,
    debug_frame_selected as _debug_frame_selected,
    append_stage_dump_rows as _append_stage_dump_rows,
    safe_cpp_ptr as _safe_cpp_ptr,
    mot_result_line as _mot_result_line,
    apply_narrow_person_score_bonus as _apply_narrow_person_score_bonus,
    tile_seam_mask as _tile_seam_mask,
    count_tile_seam_boxes as _count_tile_seam_boxes,
)
from .output_bank import OutputAppearanceBank
from .post_merge import (
    post_merge_output_tracklets,
    apply_deferred_alias,
    filter_low_quality_tracklets,
    interpolate_tracklets,
)
from .external_fp_model import (
    BandedLogisticModel,
    LogisticModel,
    RuleBaselineConfig,
    SoftmaxLinearModel,
    load_logistic_model,
    predict_external_fp_matrix,
)
from .helpers import (
    materialize_gpu_track_results as _materialize_gpu_track_results,
    materialize_gpu_track_results_pinned as _materialize_gpu_track_results_pinned,
    materialize_gpu_track_results_async as _materialize_gpu_track_results_async,
    read_deferred_result as _read_deferred_result,
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
from saccade.perception.cropper import ZeroCopyCropper
from .scene_adapt import SceneAdaptivePolicy


_SOFTMAX3_TORCH_CACHE: dict[
    tuple[SoftmaxLinearModel, str],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _append_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


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


def _record_profile_scope(name: str):
    profiler_mod = getattr(torch, "profiler", None)
    if profiler_mod is None:
        return nullcontext()
    record_fn = getattr(profiler_mod, "record_function", None)
    if record_fn is None:
        return nullcontext()
    return record_fn(name)


# ---------------------------------------------------------------------------
# Stage 2 quality gate: remove detections in the mid-score band
# [track_thresh, mid_thresh) whose geometry quality is below a floor.
# Bad-geometry stage-2 detections cause spurious lost-track associations → IDs.
# ---------------------------------------------------------------------------
def _apply_stage2_quality_gate(
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    fused_classes: torch.Tensor,
    geometry_suspect_mask: torch.Tensor,
    aligned_keypoints: "torch.Tensor | None",
    *,
    track_thresh: float,
    mid_thresh: float,
    quality_min: float,
    quality: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    "torch.Tensor | None",
]:
    n = fused_boxes.shape[0]
    # Guard against stale mask (detection cap / fp-filter may not update it)
    if geometry_suspect_mask.shape[0] != n:
        geometry_suspect_mask = torch.zeros(
            n, dtype=torch.bool, device=fused_boxes.device
        )
    stage2_mask = (fused_scores >= track_thresh) & (fused_scores < mid_thresh)
    if not stage2_mask.any():
        return (
            fused_boxes,
            fused_scores,
            fused_classes,
            geometry_suspect_mask,
            aligned_keypoints,
        )
    remove_mask = stage2_mask & (quality < quality_min)
    if not remove_mask.any():
        return (
            fused_boxes,
            fused_scores,
            fused_classes,
            geometry_suspect_mask,
            aligned_keypoints,
        )
    keep = ~remove_mask
    kp = aligned_keypoints[keep] if aligned_keypoints is not None else None
    return (
        fused_boxes[keep],
        fused_scores[keep],
        fused_classes[keep],
        geometry_suspect_mask[keep],
        kp,
    )


def _prior_iou_and_center_distance(
    boxes: torch.Tensor,
    prior_boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_iou = torch_box_iou_matrix(boxes, prior_boxes)
    det_centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    prior_centers = (prior_boxes[:, :2] + prior_boxes[:, 2:]) * 0.5
    prior_heights = (prior_boxes[:, 3] - prior_boxes[:, 1]).clamp(min=1.0)
    dist = torch.linalg.vector_norm(
        det_centers[:, None, :] - prior_centers[None, :, :], dim=2
    )
    return prior_iou, dist / prior_heights[None, :]


def _private_prior_pair_keep(
    prior_iou: torch.Tensor,
    norm_dist: torch.Tensor,
    *,
    iou_threshold: float,
    center_threshold: float,
) -> torch.Tensor:
    if iou_threshold > 0.0 or center_threshold > 0.0:
        pair_keep = torch.zeros_like(prior_iou, dtype=torch.bool)
        if iou_threshold > 0.0:
            pair_keep |= prior_iou >= iou_threshold
        if center_threshold > 0.0:
            pair_keep |= norm_dist <= center_threshold
        return pair_keep
    return (prior_iou > 0.0) | (norm_dist <= 1.0)


def _private_height_log_ratio(
    boxes: torch.Tensor, prior_boxes: torch.Tensor
) -> torch.Tensor:
    det_h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    prior_h = (prior_boxes[:, 3] - prior_boxes[:, 1]).clamp(min=1.0)
    return torch.abs(torch.log((det_h[:, None] / prior_h[None, :]).clamp(min=1e-3)))


def _apply_private_energy_margin(
    pair_scores: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if margin <= 0.0 or pair_scores.numel() == 0:
        return torch.ones_like(pair_scores, dtype=torch.bool)

    finite = torch.isfinite(pair_scores)
    row_scores = pair_scores.masked_fill(~finite, float("-inf"))
    row_top = torch.topk(row_scores, k=min(2, row_scores.shape[1]), dim=1)
    row_best = row_top.values[:, 0]
    row_idx = row_top.indices[:, 0]
    row_second = (
        row_top.values[:, 1]
        if row_top.values.shape[1] > 1
        else torch.full_like(row_best, float("-inf"))
    )
    row_margin = (row_best - row_second).clamp(min=0.0)

    col_scores = pair_scores.masked_fill(~finite, float("-inf"))
    col_top = torch.topk(col_scores, k=min(2, col_scores.shape[0]), dim=0)
    col_best = col_top.values[0]
    col_idx = col_top.indices[0]
    col_second = (
        col_top.values[1]
        if col_top.values.shape[0] > 1
        else torch.full_like(col_best, float("-inf"))
    )
    col_margin = (col_best - col_second).clamp(min=0.0)

    rows = torch.arange(pair_scores.shape[0], device=pair_scores.device)
    cols = torch.arange(pair_scores.shape[1], device=pair_scores.device)
    return (
        finite
        & (cols[None, :] == row_idx[:, None])
        & (rows[:, None] == col_idx[None, :])
        & (row_margin[:, None] >= margin)
        & (col_margin[None, :] >= margin)
    )


def _sparse_symmetric_detection_support(
    candidate_boxes: torch.Tensor,
    candidate_classes: torch.Tensor,
    field_boxes: torch.Tensor,
    field_scores: torch.Tensor,
    field_classes: torch.Tensor,
    *,
    class_aware: bool,
) -> torch.Tensor:
    if candidate_boxes.numel() == 0 or field_boxes.numel() == 0:
        return candidate_boxes.new_zeros((candidate_boxes.shape[0],))

    cand_centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) * 0.5
    cand_wh = (candidate_boxes[:, 2:] - candidate_boxes[:, :2]).clamp(min=1.0)
    field_centers = (field_boxes[:, :2] + field_boxes[:, 2:]) * 0.5

    min_side = cand_wh.min(dim=1).values
    step = (0.35 * min_side).clamp(min=2.0, max=12.0)
    sigma = (0.45 * min_side).clamp(min=2.0, max=16.0)
    offsets = candidate_boxes.new_tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )
    sample_points = cand_centers[:, None, :] + offsets[None, :, :] * step[:, None, None]
    delta = sample_points[:, :, None, :] - field_centers[None, None, :, :]
    dist2 = (delta * delta).sum(dim=3)
    weights = torch.exp(-0.5 * dist2 / (sigma[:, None, None] * sigma[:, None, None]))
    weighted_scores = weights * field_scores[None, None, :]
    if class_aware:
        same_class = candidate_classes[:, None] == field_classes[None, :]
        weighted_scores = weighted_scores.masked_fill(~same_class[:, None, :], 0.0)

    support = weighted_scores.max(dim=2).values
    center_support = support[:, 0]
    pair_a = support[:, [1, 3, 5, 7]]
    pair_b = support[:, [2, 4, 6, 8]]
    pair_strength = 0.5 * (pair_a + pair_b)
    pair_balance = torch.minimum(pair_a, pair_b) / torch.maximum(
        torch.maximum(pair_a, pair_b), candidate_boxes.new_tensor(1e-4)
    )
    return (
        0.45 * center_support
        + 0.35 * pair_strength.mean(dim=1)
        + 0.20 * pair_balance.mean(dim=1)
    ).clamp(min=0.0, max=1.0)


def _append_private_continuation_candidates(
    *,
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    fused_classes: torch.Tensor,
    geometry_suspect_mask: torch.Tensor,
    aligned_keypoints: "torch.Tensor | None",
    pre_nms_boxes: torch.Tensor,
    pre_nms_scores: torch.Tensor,
    pre_nms_classes: torch.Tensor,
    pre_nms_geometry_suspect_mask: torch.Tensor,
    pre_nms_aligned_keypoints: "torch.Tensor | None",
    baseline_keep: torch.Tensor | None,
    baseline_nms_iou: float,
    candidate_nms_iou: float,
    class_aware: bool,
    priors: "torch.Tensor | None",
    prior_classes: "torch.Tensor | None",
    prior_iou_threshold: float,
    private_prior_boxes: "torch.Tensor | None",
    private_prior_iou_threshold: float,
    private_prior_center_threshold: float,
    frame_track_thresh: float,
    frame_mid_thresh: float,
    frame_new_track_thresh: float,
    low_stage_only: bool,
    private_min_score: float,
    private_max_candidates: int,
    private_selection_mode: str,
    private_energy_margin: float,
    score_eps: float = 1e-4,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    "torch.Tensor | None",
    int,
]:
    """Append wider-NMS candidates as continuation-only tracker inputs."""
    if (
        baseline_keep is None
        or pre_nms_boxes.numel() == 0
        or candidate_nms_iou <= baseline_nms_iou
    ):
        return (
            fused_boxes,
            fused_scores,
            fused_classes,
            geometry_suspect_mask,
            aligned_keypoints,
            0,
        )

    birth_ceiling = float(frame_new_track_thresh) - score_eps
    if low_stage_only:
        birth_ceiling = min(birth_ceiling, float(frame_mid_thresh) - score_eps)
    if birth_ceiling <= float(frame_track_thresh):
        return (
            fused_boxes,
            fused_scores,
            fused_classes,
            geometry_suspect_mask,
            aligned_keypoints,
            0,
        )

    candidate_keep = nms_fast(
        pre_nms_boxes,
        pre_nms_scores,
        pre_nms_classes,
        candidate_nms_iou,
        class_aware=class_aware,
        priors=priors,
        prior_classes=prior_classes,
        prior_iou_threshold=prior_iou_threshold,
    )
    if candidate_keep.numel() == 0:
        return (
            fused_boxes,
            fused_scores,
            fused_classes,
            geometry_suspect_mask,
            aligned_keypoints,
            0,
        )

    baseline_mask = torch.zeros(
        pre_nms_scores.shape[0], dtype=torch.bool, device=pre_nms_scores.device
    )
    baseline_mask[baseline_keep.to(torch.long)] = True
    private_keep = candidate_keep[~baseline_mask[candidate_keep.to(torch.long)]]
    if private_keep.numel() == 0:
        return (
            fused_boxes,
            fused_scores,
            fused_classes,
            geometry_suspect_mask,
            aligned_keypoints,
            0,
        )

    private_scores = pre_nms_scores[private_keep]
    score_floor = max(float(private_min_score), float(frame_track_thresh))
    score_keep = private_scores >= score_floor
    if not score_keep.any():
        return (
            fused_boxes,
            fused_scores,
            fused_classes,
            geometry_suspect_mask,
            aligned_keypoints,
            0,
        )
    private_keep = private_keep[score_keep]
    private_scores = private_scores[score_keep]

    if (
        private_prior_boxes is not None
        and private_prior_boxes.numel() > 0
        and (private_prior_iou_threshold > 0.0 or private_prior_center_threshold > 0.0)
    ):
        private_boxes_for_gate = pre_nms_boxes[private_keep]
        prior_keep = torch.zeros(
            private_boxes_for_gate.shape[0],
            dtype=torch.bool,
            device=private_boxes_for_gate.device,
        )
        prior_iou, norm_dist = _prior_iou_and_center_distance(
            private_boxes_for_gate, private_prior_boxes
        )
        if private_prior_iou_threshold > 0.0:
            prior_keep |= prior_iou.max(dim=1).values >= private_prior_iou_threshold
        if private_prior_center_threshold > 0.0:
            prior_keep |= norm_dist.min(dim=1).values <= private_prior_center_threshold
        if not prior_keep.any():
            return (
                fused_boxes,
                fused_scores,
                fused_classes,
                geometry_suspect_mask,
                aligned_keypoints,
                0,
            )
        private_keep = private_keep[prior_keep]
        private_scores = private_scores[prior_keep]

    private_selection_mode = private_selection_mode.strip().lower()
    if private_selection_mode in {
        "per_track",
        "suppressor_aware",
        "sparse_symmetric",
        "energy",
    }:
        if private_selection_mode in {"sparse_symmetric", "energy"}:
            private_boxes_for_support = pre_nms_boxes[private_keep]
            sparse_support = _sparse_symmetric_detection_support(
                private_boxes_for_support,
                pre_nms_classes[private_keep],
                pre_nms_boxes,
                pre_nms_scores,
                pre_nms_classes,
                class_aware=class_aware,
            )
        else:
            sparse_support = None
        if private_prior_boxes is None or private_prior_boxes.numel() == 0:
            if private_selection_mode != "sparse_symmetric":
                return (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    geometry_suspect_mask,
                    aligned_keypoints,
                    0,
                )
            rank_scores = 0.65 * sparse_support + 0.35 * private_scores
            if (
                private_max_candidates > 0
                and rank_scores.numel() > private_max_candidates
            ):
                _, top_idx = torch.topk(rank_scores, private_max_candidates)
                private_keep = private_keep[top_idx]
                private_scores = private_scores[top_idx]
            private_prior_boxes = None
        if private_prior_boxes is None:
            pass
        else:
            private_boxes_for_rank = pre_nms_boxes[private_keep]
            prior_iou, norm_dist = _prior_iou_and_center_distance(
                private_boxes_for_rank, private_prior_boxes
            )
            pair_keep = _private_prior_pair_keep(
                prior_iou,
                norm_dist,
                iou_threshold=private_prior_iou_threshold,
                center_threshold=private_prior_center_threshold,
            )
            if private_selection_mode == "suppressor_aware":
                public_keep = baseline_keep.to(torch.long)
                public_boxes = pre_nms_boxes[public_keep]
                if public_boxes.numel() == 0:
                    return (
                        fused_boxes,
                        fused_scores,
                        fused_classes,
                        geometry_suspect_mask,
                        aligned_keypoints,
                        0,
                    )
                suppressor_iou = torch_box_iou_matrix(
                    private_boxes_for_rank, public_boxes
                )
                if class_aware:
                    private_classes = pre_nms_classes[private_keep]
                    public_classes = pre_nms_classes[public_keep]
                    same_class = private_classes[:, None] == public_classes[None, :]
                    suppressor_iou = suppressor_iou.masked_fill(
                        ~same_class, float("-inf")
                    )
                suppressor_max_iou, suppressor_idx = suppressor_iou.max(dim=1)
                has_suppressor = suppressor_max_iou >= baseline_nms_iou
                if not has_suppressor.any():
                    return (
                        fused_boxes,
                        fused_scores,
                        fused_classes,
                        geometry_suspect_mask,
                        aligned_keypoints,
                        0,
                    )
                suppressor_boxes = public_boxes[suppressor_idx]
                suppressor_prior_iou, suppressor_norm_dist = (
                    _prior_iou_and_center_distance(
                        suppressor_boxes, private_prior_boxes
                    )
                )
                suppressor_pair_keep = _private_prior_pair_keep(
                    suppressor_prior_iou,
                    suppressor_norm_dist,
                    iou_threshold=private_prior_iou_threshold,
                    center_threshold=private_prior_center_threshold,
                )
                suppressor_has_prior = suppressor_pair_keep.any(dim=1)
                pair_keep &= (
                    has_suppressor[:, None]
                    & suppressor_has_prior[:, None]
                    & ~suppressor_pair_keep
                )
            if not pair_keep.any():
                return (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    geometry_suspect_mask,
                    aligned_keypoints,
                    0,
                )
            center_affinity = 1.0 / (1.0 + norm_dist)
            if private_selection_mode == "energy":
                height_energy = _private_height_log_ratio(
                    private_boxes_for_rank, private_prior_boxes
                )
                iou_energy = -torch.log(prior_iou.clamp(min=1e-4))
                score_energy = 1.0 - private_scores[:, None].clamp(min=0.0, max=1.0)
                support_energy = (
                    1.0 - sparse_support[:, None].clamp(min=0.0, max=1.0)
                    if sparse_support is not None
                    else private_scores[:, None].new_zeros(prior_iou.shape)
                )
                energy = (
                    0.60 * iou_energy
                    + 0.25 * norm_dist
                    + 0.20 * score_energy
                    + 0.20 * height_energy
                    + 0.20 * support_energy
                )
                pair_scores = -energy
            elif sparse_support is None:
                pair_scores = (
                    0.70 * prior_iou
                    + 0.20 * center_affinity
                    + 0.10 * private_scores[:, None]
                )
            else:
                pair_scores = (
                    0.45 * prior_iou
                    + 0.15 * center_affinity
                    + 0.30 * sparse_support[:, None]
                    + 0.10 * private_scores[:, None]
                )
            pair_scores = pair_scores.masked_fill(~pair_keep, float("-inf"))
            if private_selection_mode == "energy":
                pair_scores = pair_scores.masked_fill(
                    ~_apply_private_energy_margin(
                        pair_scores,
                        margin=private_energy_margin,
                    ),
                    float("-inf"),
                )
            flat_scores_cpu = pair_scores.flatten().detach().cpu()
            flat_order = torch.argsort(flat_scores_cpu, descending=True)
            n_priors = int(private_prior_boxes.shape[0])
            used_candidates: set[int] = set()
            used_priors: set[int] = set()
            selected_candidates: list[int] = []
            for flat_idx in flat_order.tolist():
                score = float(flat_scores_cpu[flat_idx].item())
                if score == float("-inf"):
                    break
                cand_idx = int(flat_idx // n_priors)
                prior_idx = int(flat_idx % n_priors)
                if cand_idx in used_candidates or prior_idx in used_priors:
                    continue
                used_candidates.add(cand_idx)
                used_priors.add(prior_idx)
                selected_candidates.append(cand_idx)
                if (
                    private_max_candidates > 0
                    and len(selected_candidates) >= private_max_candidates
                ):
                    break
            if not selected_candidates:
                return (
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    geometry_suspect_mask,
                    aligned_keypoints,
                    0,
                )
            selected = torch.tensor(
                selected_candidates, dtype=torch.long, device=private_keep.device
            )
            private_keep = private_keep[selected]
            private_scores = private_scores[selected]
    elif private_selection_mode == "global":
        pass
    else:
        raise ValueError(f"unknown private selection mode: {private_selection_mode}")

    if (
        private_selection_mode == "global"
        and private_max_candidates > 0
        and private_scores.numel() > private_max_candidates
    ):
        _, top_idx = torch.topk(private_scores, private_max_candidates)
        private_keep = private_keep[top_idx]
        private_scores = private_scores[top_idx]

    private_scores = torch.minimum(
        private_scores,
        torch.full_like(private_scores, birth_ceiling),
    )
    if geometry_suspect_mask.shape[0] != fused_scores.shape[0]:
        geometry_suspect_mask = torch.zeros(
            fused_scores.shape[0], dtype=torch.bool, device=fused_scores.device
        )
    if (
        aligned_keypoints is not None
        and aligned_keypoints.shape[0] != fused_scores.shape[0]
    ):
        aligned_keypoints = None
    private_boxes = pre_nms_boxes[private_keep]
    private_classes = pre_nms_classes[private_keep]
    private_geometry = pre_nms_geometry_suspect_mask[private_keep]

    fused_boxes = torch.cat((fused_boxes, private_boxes), dim=0)
    fused_scores = torch.cat((fused_scores, private_scores), dim=0)
    fused_classes = torch.cat((fused_classes, private_classes), dim=0)
    geometry_suspect_mask = torch.cat((geometry_suspect_mask, private_geometry), dim=0)
    if aligned_keypoints is not None and pre_nms_aligned_keypoints is not None:
        aligned_keypoints = torch.cat(
            (aligned_keypoints, pre_nms_aligned_keypoints[private_keep]), dim=0
        )
    else:
        aligned_keypoints = None

    return (
        fused_boxes,
        fused_scores,
        fused_classes,
        geometry_suspect_mask,
        aligned_keypoints,
        int(private_scores.numel()),
    )


# ---------------------------------------------------------------------------
# Consecutive-frame birth gate: boost sub-threshold detections that appear
# in the last N consecutive frames (rolling window IoU check).
# More selective than birth_quality_gate alone — requires temporal evidence.
# ---------------------------------------------------------------------------
def _consecutive_birth_check(
    sub_boxes: torch.Tensor,
    window: "list[torch.Tensor]",
    iou_thresh: float,
    min_motion_px: float = 0.0,
) -> torch.Tensor:
    """Return bool mask for sub_boxes that have an IoU match in EVERY window frame
    and (optionally) have moved at least min_motion_px from their oldest-frame match.

    Motion gate: static FPs (chairs, signs) have near-zero GMC-compensated displacement;
    real people have measurable displacement even across just 2 frames (~10-15 px/frame).
    """
    n = sub_boxes.shape[0]
    confirmed = torch.ones(n, dtype=torch.bool, device=sub_boxes.device)
    oldest_match_cx: "torch.Tensor | None" = None
    oldest_match_cy: "torch.Tensor | None" = None
    for frame_idx, prev_boxes in enumerate(window):
        if prev_boxes.numel() == 0:
            confirmed[:] = False
            break
        iou = torch_box_iou_matrix(
            sub_boxes, prev_boxes, union_mode="clamp", area_min=1.0
        )
        max_iou, best_idx = iou.max(dim=1)
        confirmed &= max_iou >= iou_thresh
        if frame_idx == 0 and min_motion_px > 0:
            matched = prev_boxes[best_idx]
            oldest_match_cx = (matched[:, 0] + matched[:, 2]) / 2
            oldest_match_cy = (matched[:, 1] + matched[:, 3]) / 2
    # Motion gate: exclude detections whose center hasn't moved from oldest frame match
    if min_motion_px > 0 and oldest_match_cx is not None and confirmed.any():
        curr_cx = (sub_boxes[:, 0] + sub_boxes[:, 2]) / 2
        curr_cy = (sub_boxes[:, 1] + sub_boxes[:, 3]) / 2
        disp = (
            (curr_cx - oldest_match_cx) ** 2 + (curr_cy - oldest_match_cy) ** 2
        ).sqrt()
        confirmed &= disp >= min_motion_px
    return confirmed


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Duplicate detection suppression within the same frame.
# When the detector produces multiple overlapping detections of the same
# person at slightly different positions (detector artifact), suppress the
# lower-score ones before they reach multi_birth / tracker.
# ---------------------------------------------------------------------------
def _suppress_duplicate_detections(
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    *,
    iou_threshold: float = 0.85,
    min_score_ratio: float = 1.05,
) -> torch.Tensor:
    """Suppress near-duplicate detections within the same frame.

    When two boxes have IoU > iou_threshold but different scores (ratio > min_score_ratio),
    suppress the lower-score detection. This eliminates detector artifacts where the
    detector produces multiple overlapping detections of the same person at slightly
    different positions.

    Returns a keep_mask of shape (N,).
    """
    n = fused_boxes.shape[0]
    if n <= 1:
        return torch.ones(n, dtype=torch.bool, device=fused_boxes.device)

    keep = torch.ones(n, dtype=torch.bool, device=fused_boxes.device)
    scores = fused_scores.clone()

    iou_matrix = torch_box_iou_matrix(fused_boxes, fused_boxes, union_mode="clamp")

    # Sort by score descending, process highest-score first
    sorted_idx = torch.argsort(scores, descending=True)
    suppressed = torch.zeros(n, dtype=torch.bool, device=fused_boxes.device)

    for i in range(n):
        si = sorted_idx[i]
        if suppressed[si]:
            continue
        # Check IoU against all higher-score detections
        iou_with_higher = iou_matrix[si][sorted_idx[:i]]
        # Find indices in sorted_idx that have high IoU with this box
        high_iou_mask = iou_with_higher >= iou_threshold
        if high_iou_mask.any():
            # Check if score ratio is significant enough
            higher_scores = scores[sorted_idx[:i]]
            ratios = higher_scores / (scores[si] + 1e-6)
            # Suppress if any higher-score detection has IoU >= threshold AND score ratio >= min_ratio
            needs_suppress = high_iou_mask & (ratios >= min_score_ratio)
            if needs_suppress.any():
                keep[si] = False
                suppressed[si] = True

    return keep


# Per-frame detection cap: keep top-K detections per frame by combined
# score+quality metric.  This prevents excessive detections in crowded
# scenes from overwhelming the association stage.
# ---------------------------------------------------------------------------
def _apply_detection_cap(
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    fused_classes: torch.Tensor,
    *,
    max_detections: int = 30,
    quality_factors: torch.Tensor | None = None,
    rank_method: str = "score",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cap the number of detections per frame.

    Keeps top-K detections sorted by the selected ranking method.

    Methods:
      - "score": raw score (default)
      - "quality": score * quality_factor
      - "fp_filter": FP-filter-aware scoring (penalizes large+moderate-score boxes)
      - "fp_filter_quality": FP-filter-aware scoring * quality_factor
    """
    n = fused_boxes.shape[0]
    if n <= max_detections or max_detections <= 0:
        return fused_boxes, fused_scores, fused_classes

    if rank_method == "fp_filter":
        ranking = _compute_fp_filter_ranking(fused_boxes, fused_scores)
    elif rank_method == "fp_filter_quality":
        ranking_raw = _compute_fp_filter_ranking(fused_boxes, fused_scores)
        if quality_factors is not None and quality_factors.numel() == n:
            ranking = ranking_raw * quality_factors
        else:
            ranking = ranking_raw
    elif quality_factors is not None and quality_factors.numel() == n:
        ranking = fused_scores * quality_factors
    else:
        ranking = fused_scores.clone()

    _, top_indices = torch.topk(ranking, min(max_detections, n), dim=0)
    top_indices = top_indices.sort().values

    return (
        fused_boxes[top_indices],
        fused_scores[top_indices],
        fused_classes[top_indices],
    )


def _compute_fp_filter_ranking(
    boxes: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """Ranking that penalizes likely-FP detections.

    Based on FP analysis:
      - ~68% of FP have area > 5000px
      - ~38% of FP have scores 0.4-0.6 (moderate)
      - ~60% of FP have person-like H/W ratios (2.0-4.0)

    Strategy: gentle penalty for large boxes with moderate scores.
    High-score large boxes are kept (real large persons/groups).
    Moderate-score large boxes are penalized (likely FP).
    """
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.float32)

    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    area = bw * bh
    aspect = bh / bw

    # Size penalty: gentle penalty for large boxes (area > 4000)
    # Real persons in MOT17 are typically 2000-8000px
    # FP tend to be larger (>5000px) with moderate scores
    _log_max = area.new_tensor(float(np.log(12000.0 / 4000.0)))
    size_penalty = torch.where(
        area > 4000,
        1.0 - 0.3 * torch.clamp(torch.log(area / 4000.0) / _log_max, 0.0, 1.0),
        torch.ones_like(area),
    )

    # Moderate-score penalty: boxes with score 0.4-0.6 are suspicious
    # These are the ~38% of FPs that have moderate confidence
    # Apply gentler penalty - only if size is also large
    mod_score_mask = (scores > 0.35) & (scores < 0.65)
    large_area_mask = area > 4000
    suspicious_mask = mod_score_mask & large_area_mask
    suspicious_penalty = torch.where(
        suspicious_mask,
        0.70,  # Moderate penalty for suspicious detections
        torch.ones_like(scores),
    )

    # Aspect ratio bonus: genuine persons have H/W 2.0-4.0
    # But ~60% of FPs also have this range, so small bonus only
    aspect_bonus = torch.exp(-0.5 * ((aspect - 2.5) / 1.5) ** 2) * 0.05 + 1.0

    # Combined ranking score
    ranking = scores * size_penalty * suspicious_penalty * aspect_bonus

    return ranking


# Score assigned to detections rejected by the FP hard filter when masking in
# place (instead of compacting). Below every tracker score threshold, so the GPU
# tracker's score gate drops them with no host sync. Negative is safe: score
# fusion uses non-negative thresholds only.
_FP_HARD_REJECT_SCORE = -1.0


def _fp_hard_reject_mask(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    min_score: float,
    max_suspicious_area: int,
    max_suspicious_score: float,
) -> torch.Tensor:
    """Boolean mask (True = reject) for the FP hard filter.

    Rejects detections that are either very low score, or low score AND
    oversized — the tail of the FP distribution. Pure elementwise GPU ops, no
    host sync.
    """
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    area = bw * bh
    # Suspicious: low score AND large area. Also reject very low score regardless.
    suspicious = (scores < max_suspicious_score) & (area > max_suspicious_area)
    very_low_score = scores < min_score
    return suspicious | very_low_score


def _apply_fp_hard_filter(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    *,
    min_score: float = 0.25,
    max_suspicious_area: int = 10000,
    max_suspicious_score: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compacting hard filter: removes extremely suspicious detections.

    Retained for the workbench / unit-test contract. The hot eval path uses
    `_fp_hard_reject_mask` + masked_fill instead (sync-free, fixed shape).
    """
    if boxes.numel() == 0:
        return boxes, scores, classes

    reject = _fp_hard_reject_mask(
        boxes,
        scores,
        min_score=min_score,
        max_suspicious_area=max_suspicious_area,
        max_suspicious_score=max_suspicious_score,
    )
    # Single .nonzero() (one host sync) — three separate bool index ops would
    # each re-run .nonzero() internally, triggering three syncs.
    keep_idx = (~reject).nonzero(as_tuple=True)[0]
    return boxes[keep_idx], scores[keep_idx], classes[keep_idx]


def _get_softmax3_torch_params(
    model: SoftmaxLinearModel,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (model, f"{device.type}:{device.index}:{dtype}")
    cached = _SOFTMAX3_TORCH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    params = (
        torch.as_tensor(model.weights, device=device, dtype=dtype),
        torch.as_tensor(model.bias, device=device, dtype=dtype),
        torch.as_tensor(model.mean, device=device, dtype=dtype),
        torch.as_tensor(model.std, device=device, dtype=dtype),
    )
    _SOFTMAX3_TORCH_CACHE[cache_key] = params
    return params


def _predict_softmax3_probs_torch(
    model: SoftmaxLinearModel,
    *,
    subset_scores: torch.Tensor,
    subset_widths: torch.Tensor,
    subset_heights: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    edge_margin: torch.Tensor,
    touches_edge: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    feature_map = {
        "score": subset_scores,
        "width": subset_widths,
        "height": subset_heights,
        "area": subset_widths * subset_heights,
        "aspect_ratio": subset_heights / subset_widths,
        "center_x_norm": center_x / max(float(image_width), 1.0),
        "center_y_norm": center_y / max(float(image_height), 1.0),
        "edge_margin_norm": edge_margin
        / max(min(float(image_width), float(image_height)), 1.0),
        "touches_edge": touches_edge,
    }
    unknown_features = [name for name in model.feature_names if name not in feature_map]
    if unknown_features:
        raise ValueError(
            "softmax3 model has unsupported feature_names: "
            + ", ".join(unknown_features)
        )
    feature_matrix = torch.stack(
        [feature_map[name] for name in model.feature_names],
        dim=1,
    )
    weights, bias, mean, std = _get_softmax3_torch_params(
        model,
        device=feature_matrix.device,
        dtype=feature_matrix.dtype,
    )
    standardized = (feature_matrix - mean) / std
    logits = standardized @ weights + bias
    return torch.softmax(logits, dim=1)


def _apply_external_fp_filter(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    mode: str,
    rule_config: RuleBaselineConfig,
    logistic_model: LogisticModel | BandedLogisticModel | SoftmaxLinearModel | None,
    logistic_threshold: float,
    max_score: float,
    penalty: float,
    min_score: float,
    softmax_min_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0 or str(mode).lower() == "off":
        return boxes, scores, classes

    mode = str(mode).lower()
    low_score_mask = scores <= max_score
    # Resolve the mask to integer indices with a single nonzero() (one host
    # sync). Reusing low_idx for every gather/scatter below avoids the implicit
    # per-op nonzero that bool-mask indexing would otherwise re-trigger each time
    # (profiled: this function alone was ~6-9 host syncs/frame).
    low_idx = low_score_mask.nonzero(as_tuple=True)[0]
    if low_idx.numel() == 0:
        return boxes, scores, classes

    keep = torch.ones(scores.shape[0], dtype=torch.bool, device=boxes.device)
    subset_boxes = boxes[low_idx]
    subset_scores = scores[low_idx]
    adjusted_scores = scores.clone()
    if mode in {"rule", "rule_score"}:
        subset_widths = (subset_boxes[:, 2] - subset_boxes[:, 0]).clamp(min=1e-6)
        subset_heights = (subset_boxes[:, 3] - subset_boxes[:, 1]).clamp(min=1e-6)
        subset_keep = subset_scores >= rule_config.min_score
        subset_keep &= ~(
            (subset_scores < rule_config.low_score)
            & (subset_heights < rule_config.min_height)
        )
        subset_keep &= ~(
            (subset_scores < rule_config.medium_score)
            & (subset_heights < rule_config.medium_height)
            & ((subset_heights / subset_widths) < rule_config.min_aspect)
        )
    elif mode in {"logistic", "softmax3"}:
        if logistic_model is None:
            raise ValueError(f"external_fp_filter_mode={mode} requires a loaded model")
        subset_widths = (subset_boxes[:, 2] - subset_boxes[:, 0]).clamp(min=1e-6)
        subset_heights = (subset_boxes[:, 3] - subset_boxes[:, 1]).clamp(min=1e-6)
        center_x = (subset_boxes[:, 0] + subset_boxes[:, 2]) * 0.5
        center_y = (subset_boxes[:, 1] + subset_boxes[:, 3]) * 0.5
        left_margin = subset_boxes[:, 0]
        top_margin = subset_boxes[:, 1]
        right_margin = torch.clamp(float(image_width) - subset_boxes[:, 2], min=0.0)
        bottom_margin = torch.clamp(float(image_height) - subset_boxes[:, 3], min=0.0)
        edge_margin = torch.minimum(
            torch.minimum(left_margin, right_margin),
            torch.minimum(top_margin, bottom_margin),
        )
        touches_edge = (edge_margin <= 1.0).to(torch.float32)
        if mode == "logistic":
            features = np.stack(
                [
                    subset_scores.detach().cpu().numpy().astype(np.float64, copy=False),
                    subset_widths.detach().cpu().numpy().astype(np.float64, copy=False),
                    subset_heights.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False),
                    (subset_widths * subset_heights)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False),
                    (subset_heights / subset_widths)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False),
                    (center_x / max(float(image_width), 1.0))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False),
                    (center_y / max(float(image_height), 1.0))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False),
                    (
                        edge_margin
                        / max(min(float(image_width), float(image_height)), 1.0)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False),
                    touches_edge.detach().cpu().numpy().astype(np.float64, copy=False),
                ],
                axis=1,
            )
            probs = predict_external_fp_matrix(logistic_model, features)
            subset_keep = torch.from_numpy(probs >= logistic_threshold).to(
                device=boxes.device, dtype=torch.bool
            )
        else:
            if not isinstance(logistic_model, SoftmaxLinearModel):
                raise ValueError(
                    "external_fp_filter_mode=softmax3 requires a softmax3 model JSON"
                )
            probs = _predict_softmax3_probs_torch(
                logistic_model,
                subset_scores=subset_scores,
                subset_widths=subset_widths,
                subset_heights=subset_heights,
                center_x=center_x,
                center_y=center_y,
                edge_margin=edge_margin,
                touches_edge=touches_edge.to(dtype=subset_scores.dtype),
                image_width=image_width,
                image_height=image_height,
            )
            tp_idx = logistic_model.class_names.index("tp")
            fp_idx = logistic_model.class_names.index("fp")
            tp_probs = probs[:, tp_idx]
            fp_probs = probs[:, fp_idx]
            tp_vs_fp = tp_probs / (tp_probs + fp_probs).clamp(min=1e-6)
            score_scale = tp_vs_fp.clamp(min=softmax_min_scale, max=1.0)
            subset_adjusted = subset_scores * score_scale.to(dtype=subset_scores.dtype)
            adjusted_scores[low_idx] = subset_adjusted
            subset_keep = subset_adjusted >= min_score
    else:
        raise ValueError(f"Unknown external FP filter mode: {mode}")

    if penalty < 0.999 or mode == "rule_score":
        # torch.where avoids the bool-mask scatter (~subset_keep) host sync.
        penalized_scores = torch.where(
            subset_keep, subset_scores, subset_scores * penalty
        )
        adjusted_scores[low_idx] = penalized_scores
        subset_keep = penalized_scores >= min_score
    keep[low_idx] = subset_keep
    # Single nonzero() for the final compaction instead of three bool-index ops
    # (boxes/adjusted_scores/classes) plus a keep.any() guard.
    keep_idx = keep.nonzero(as_tuple=True)[0]
    return boxes[keep_idx], adjusted_scores[keep_idx], classes[keep_idx]


def _compute_adaptive_cap(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    base_cap: int = 40,
    max_cap: int = 60,
    min_cap: int = 15,
) -> int:
    """Compute adaptive detection cap based on scene characteristics.

    Crowded scenes (many detections, high average score) get lower caps.
    Sparse scenes (few detections) get higher caps to avoid over-filtering.
    """
    n = boxes.shape[0]
    if n == 0:
        return base_cap

    avg_score = scores.mean().item()
    median_area = float(
        torch.median((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    )

    # Crowdedness signal: high avg score + large boxes + many detections
    crowd_factor = 0.0
    if avg_score > 0.55:
        crowd_factor += 0.3
    if n > 50:
        crowd_factor += 0.3
    if median_area > 8000:
        crowd_factor += 0.2

    # Cap adjustment
    if crowd_factor > 0.5:
        cap = max(min_cap, int(base_cap * (1.0 - crowd_factor * 0.5)))
    elif crowd_factor < 0.2:
        cap = min(max_cap, int(base_cap * (1.0 + crowd_factor)))
    else:
        cap = base_cap

    return max(min_cap, min(max_cap, cap))


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
from saccade.perception.eval.gmc import (  # noqa: E402
    SparseOpticalFlowGMC,
    PyGraphedGMC,
    TilePhaseCorrAffineGMC,
)
from saccade.perception.eval.multi_birth import MultiSignalBirthManager  # noqa: E402
from saccade.perception.eval.pool import (  # noqa: E402
    AdaptiveFramePool,
    rgb_chw_to_nv12_gpu,
    rgb_hwc_to_nv12_gpu,
)
from saccade.perception.eval.preprocess import (  # noqa: E402
    GeometryScaleState,
    apply_frame_preprocess,
    geometry_mid_thresh_scale,
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
from saccade.perception.eval.tracking import GlobalTrackIdMapper  # noqa: E402
from saccade.perception.tracking.dynamic_reid import DynamicReIDController  # noqa: E402
from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    TrackAppearanceBank,
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


def _build_active_track_priors(
    tracker: Any,
    device: torch.device,
    *,
    min_track_age: int = 0,
    max_track_age: int | None = None,
    min_track_score: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    snapshots = tracker.get_state_snapshots()
    if not snapshots:
        return None, None

    prior_boxes: list[list[float]] = []
    prior_classes: list[int] = []
    for snap in snapshots:
        if int(snap.age) < min_track_age:
            continue
        if max_track_age is not None and int(snap.age) > max_track_age:
            continue
        if float(snap.score) < min_track_score:
            continue
        cx, cy, a, h = snap.state[0], snap.state[1], snap.state[2], snap.state[3]
        w = a * h
        prior_boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        prior_classes.append(int(snap.class_id))

    if not prior_boxes:
        return None, None

    return (
        torch.tensor(prior_boxes, device=device, dtype=torch.float32).contiguous(),
        torch.tensor(prior_classes, device=device, dtype=torch.int32).contiguous(),
    )


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized not in {"0", "false", "no", "off"}


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
    c.confirm_streak = int(cfg.kwargs.get("confirm_streak", 1))
    c.confirm_score_thresh = float(cfg.kwargs.get("confirm_score_thresh", 0.0))
    c.fuse_score_weight = float(getattr(cfg, "fuse_score_weight", 0.4))
    c.vel_dir_weight = float(getattr(cfg, "vel_dir_weight", 0.0))
    c.stage2_match_thresh = float(getattr(cfg, "stage2_match_thresh", 0.5))
    c.birth_low_score_thresh = float(getattr(cfg, "birth_low_score_thresh", 0.0))
    c.birth_prox_norm_thresh = float(getattr(cfg, "birth_prox_norm_thresh", 0.0))
    # NB: OAO is configured on the tracker via set_oao_params(cfg.oao_tau); the
    # C++ SequenceConfig has no oao_tau field, so do not set it here.
    c.track_buffer = 30

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
    native_cfg.score_threshold = cfg.conf_threshold
    native_cfg.person_class = cfg.person_class
    native_cfg.nms_threshold = cfg.nms_iou_threshold
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
            cfg, seq, data_root, cfg.split, trt_input_size, max_raw_dets
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
    cheb_gr_online_log_path = output_root / "_cheb_gr_online_handover.csv"
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

    # Cheb-GR offline tracklet merge (path 2) / causal online handover: build
    # the ReID extractor once. C++ eval emits no per-det embedding, so tracklet
    # crops are re-cut from img1 inside the post-process loop.
    cheb_gr_extractor = None
    cheb_gr_online = getattr(cfg, "cheb_gr_online", False)
    occ_audit_enabled = getattr(cfg, "occ_audit", False)
    if cfg.cheb_gr_merge_enabled or cheb_gr_online or occ_audit_enabled:
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
            seq_img_dir = str(_Path(cfg.data_root) / cfg.split / seq / "img1")
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
            seq_img_dir = str(_Path(cfg.data_root) / cfg.split / seq / "img1")
            head_embs, bank_embs = extract_handover_embeddings(
                results_lines,
                seq_img_dir,
                cheb_gr_extractor,
                decide_n=cfg.cheb_gr_online_decide_n,
                n_samples=cfg.cheb_gr_merge_n_samples,
                crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                appearance_occlusion_cov=cfg.appearance_occlusion_cov,
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
                f"  {seq}: cheb-gr online handover {ho_stats['ids_before']}→"
                f"{ho_stats['ids_after']} ({ho_stats['handovers']} handovers, "
                f"{ho_stats['events_with_candidates']}/{ho_stats['events']} "
                "events had candidates, "
                f"reject_cost={ho_stats['reject_cost']} "
                f"reject_margin={ho_stats['reject_margin']} "
                f"reject_min_head={ho_stats['reject_min_head']})"
            )
        elif cheb_gr_extractor is not None:
            seq_img_dir = str(_Path(cfg.data_root) / cfg.split / seq / "img1")
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
        data_root=cfg.data_root,
        split=cfg.split,
        output=str(cfg.output_root),
        sequences=",".join(cfg.seqs),
        detector=cfg.kwargs.get("detector"),
        score_on_gt_frames=bool(cfg.kwargs.get("score_on_gt_frames", False)),
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
            or _semantic_delayed_claim
            or _semantic_cheb_gr_claim
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
                    states, covs, tids, maxn = detector.tracker.get_gpu_buffers()
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
    )
    state.results_lines.extend(_emit_lines)
    state.db_emit_frame_id = 0
    state.db_emit_event = None
    state.db_emit_ctx.clear()


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
        copy_pad_detections(
            raw_boxes_contig.data_ptr(),
            raw_scores_contig.data_ptr(),
            raw_classes_contig.data_ptr(),
            min(raw_box_count, _NMS_FIXED_N),
            _nms_in["boxes"].data_ptr(),
            _nms_in["scores"].data_ptr(),
            _nms_in["classes"].data_ptr(),
            _NMS_FIXED_N,
            torch.cuda.current_stream().cuda_stream,
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
            torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        _nms_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(_nms_graph):
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
                torch.cuda.current_stream().cuda_stream,
            )
        print("🕯️ [NMSGraph] Captured NMS graph")
    else:
        _nms_graph.replay()
    n_post = _NMS_FIXED_N
    return n_post, _nms_graph


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
        # Whole-graph replays return views into reusable static buffers.  Clone
        # every tensor crossing the frame boundary before another replay can
        # overwrite those buffers.
        fused_boxes = fused_boxes.clone()
        fused_scores = fused_scores.clone()
        fused_classes = fused_classes.clone()
        source_keypoints = (
            source_keypoints.clone() if source_keypoints is not None else None
        )
        ready_event.record(stream)

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
    if not synchronize or _barrier_mode == "event":
        # EXPERIMENTAL: drop the ingest->detect full barrier to probe whether the
        # decode is already ordered w.r.t. the current (TRT) stream and to measure
        # the recoverable host stall. The decode (nvJPEG/DALI) exposes no stream/
        # event handle, so there is no narrow fence here yet -- this mode is only
        # valid if N>=6 GPU-decode runs show zero drift; otherwise the decode must
        # be fenced onto the current stream first.
        pass
    else:
        torch.cuda.synchronize()

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

    # Ensure the TRT output is fully written before the postprocess stage
    # reads the raw detection tensors (views into the shared output buffer).
    # In whole_graph mode the TRT enqueue and the postprocess graphs both launch
    # from the current stream, so this ordering is likely already implicit; the
    # no_postproc/event modes drop the redundant full barrier (gated on the same
    # N>=6 determinism check).
    if not synchronize or _barrier_mode in ("no_postproc", "event"):
        pass
    else:
        torch.cuda.synchronize()

    return fused_boxes, fused_scores, fused_classes, is_tiled, source_keypoints


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
        relinker is not None
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
            and cfg.reid_mode == "off"
            and not bool(cfg.kwargs.get("id_stability_filter", False))
        )
        if _use_fast_emit:
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
            host_track_batch = _prepare_host_track_batch(
                track_results,
                tracker_result_buffers,
                dynamic_reid_enabled=dynamic_reid is not None,
                person_class=cfg.person_class,
            )

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
            resolved_tracks = _resolve_frame_tracks(
                frame_id=frame_id,
                frame_w=w_orig,
                frame_h=h_orig,
                prepared_candidates=prepared_candidates,
                lifecycle_merger=lifecycle_merger,
                identity_resolver=identity_resolver,
            )
            frame_result_lines = _emit_resolved_tracks(
                seq=seq,
                frame_id=frame_id,
                frame_w=w_orig,
                frame_h=h_orig,
                resolved_tracks=resolved_tracks,
                global_id_mapper=global_id_mapper,
                output_appearance_bank=output_appearance_bank,
            )
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
            MIN_REID_GAP = 2
            time_since_last_reid = frame_id - state.last_reid_frame

            if time_since_last_reid < MIN_REID_GAP:
                _do_reid = False
            elif cfg.need_reid_enabled:
                if dynamic_reid is not None:
                    _do_reid = dynamic_reid.should_reid(after_merge_count)
                else:
                    _do_reid = need_reid_frame(state.prev_track_ids, after_merge_count)
            else:
                _do_reid = frame_id % seq_reid_interval == 0

        if _do_reid:
            state.last_reid_frame = frame_id
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

                if native_reid_available and perception_pipeline is not None:
                    frame_hwc = pool.as_rgb_chw().permute(1, 2, 0).contiguous()
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
                    else:
                        if profile_stages:
                            perception_pipeline.reset_reid_profile_stats()
                            torch.cuda.synchronize()
                            t_reid_extract_start = time.perf_counter()

                        perception_pipeline.extract_reid(
                            frame_hwc.data_ptr(),
                            h_orig,
                            w_orig,
                            budgeted_boxes.data_ptr(),
                            int(budgeted_boxes.shape[0]),
                            budget_embeddings.data_ptr(),
                            torch.cuda.current_stream().cuda_stream,
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
        reid_side_event.synchronize()
        if profile_stages:
            elapsed_ms = (time.perf_counter() - t_reid_extract_start) * 1000
            seq_stage_totals["reid_extract"] += elapsed_ms
            record_stage_sample("reid_extract", elapsed_ms)
        embeddings[_reid_async_indices] = _reid_async_embeddings
        _reid_side_pending = False
        _reid_frame_hwc_ref = None
    return embeddings, mid_thresh_scale


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
    # -----------------------------------------------
    if _defer_emit and state.defer_emit_event is not None:
        _lines, state.prev_track_ids = _flush_deferred_emit(
            state.defer_emit_event,
            _pinned_result_bufs,
            default_class_id=cfg.person_class if cfg.track_person_only else None,
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
    t_e2e_start = time.perf_counter()
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
        wb_result, _ = time_stage(
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
                            cfg.gamma,
                            cfg.gamma_luma_threshold,
                            cfg.contrast,
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
                        state.seq_narrow_bonus = cfg.narrow_person_score_bonus
                    wb.narrow_bonus = state.seq_narrow_bonus
                    print(
                        f"  [scene_adapt] {seq} @ frame {frame_id}: {st}"
                        + (
                            f" → narrow_bonus={state.seq_narrow_bonus:.2f}"
                            if cfg.scene_adapt_enabled
                            and cfg.narrow_person_score_bonus > 0
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
            frame_birth_events = []

            # Update seq_stage_totals for the skipped stages
            seq_stage_totals["postprocess"] += 0.0
            seq_stage_totals["track"] += 0.0
            seq_stage_totals["materialize"] += 0.0

            # Save gray frame for GMC in next frame
            state.prev_gray = pool.get_frame_luma().clone()
        else:
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

            # P5-4: scene-adaptive observation and one-shot classification.
            if _scene_policy is not None and not _scene_policy.is_classified:
                _scene_policy.observe(fused_boxes, fused_scores, w_orig, h_orig)
                if _scene_policy.is_classified and _scene_policy.stats is not None:
                    st = _scene_policy.stats
                    if st.scene_type == "crowded_narrow":
                        state.seq_narrow_bonus = cfg.narrow_person_score_bonus
                    print(
                        f"  [scene_adapt] {seq} @ frame {frame_id}: {st}"
                        + (
                            f" → narrow_bonus={state.seq_narrow_bonus:.2f}"
                            if cfg.scene_adapt_enabled
                            and cfg.narrow_person_score_bonus > 0
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

            # Keep low-score boxes down to cfg.track_thresh so ByteTrack's
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
                    person_class=cfg.person_class,
                    bonus=state.seq_narrow_bonus,
                    max_width_ratio=cfg.narrow_person_max_width_ratio,
                    min_height_ratio=cfg.narrow_person_min_height_ratio,
                    min_aspect=cfg.narrow_person_min_aspect,
                    max_aspect=cfg.narrow_person_max_aspect,
                )
                t_sub_start = time.perf_counter()
                keep_indices, geometry_suspect_mask, _ = filter_detections_fast(
                    fused_boxes,
                    fused_scores,
                    fused_classes,
                    score_threshold=min(
                        cfg.conf_threshold,
                        cfg.track_thresh,
                        cfg.crowd_conf_threshold
                        if cfg.crowd_low_score_mode
                        else cfg.conf_threshold,
                        cfg.crowd_track_thresh
                        if cfg.crowd_low_score_mode
                        else cfg.track_thresh,
                    ),
                    track_person_only=cfg.track_person_only,
                    person_class=cfg.person_class,
                    is_tiled=is_tiled,
                    frame_w=w_orig,
                    frame_h=h_orig,
                    person_geometry_prior=cfg.person_geometry_prior,
                    geometry_suspect_support=cfg.geometry_suspect_support,
                    person_min_height_ratio=cfg.person_min_height_ratio,
                    person_min_aspect=cfg.person_min_aspect,
                    person_max_aspect=cfg.person_max_aspect,
                    person_min_area_ratio=cfg.person_min_area_ratio,
                    person_max_area_ratio=cfg.person_max_area_ratio,
                )
                fused_boxes = fused_boxes[keep_indices]
                fused_scores = fused_scores[keep_indices]
                fused_classes = fused_classes[keep_indices]
                if aligned_keypoints is not None:
                    aligned_keypoints = aligned_keypoints[keep_indices]

                if _fpn_reid_mode and fused_boxes.numel() > 0:
                    valid_w = (fused_boxes[:, 2] - fused_boxes[:, 0]) > 0
                    if not valid_w.all():
                        fused_boxes = fused_boxes[valid_w]
                        fused_scores = fused_scores[valid_w]
                        fused_classes = fused_classes[valid_w]
                        if geometry_suspect_mask.numel() > 0:
                            geometry_suspect_mask = geometry_suspect_mask[valid_w]

                if cfg.detection_quality_scaling and fused_boxes.numel() > 0:
                    quality_factors = _compute_detection_quality_batch(
                        fused_boxes,
                        w_orig,
                        h_orig,
                        w_aspect=cfg.detection_quality_w_aspect,
                        w_center=cfg.detection_quality_w_center,
                        w_area=cfg.detection_quality_w_area,
                    )
                    fused_scores = fused_scores * quality_factors
                elif cfg.geometry_suspect_support and geometry_suspect_mask.any():
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
                and (is_tiled or cfg.nms_iou_threshold is not None)
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
                    cfg.nms_iou_threshold,
                    class_aware=not cfg.track_person_only,
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

            if cfg.tile_diagnostics and is_tiled:
                seq_tile_diag["frames_tiled"] += 1
                seq_tile_diag["pre_merge_seam_boxes"] += _count_tile_seam_boxes(
                    fused_boxes,
                    tiling=cfg.tiling,
                    h_orig=h_orig,
                    w_orig=w_orig,
                )

            use_repo_cross_tile_merge = (
                cfg.cross_tile_merge
                and is_tiled
                and cfg.tiling != "sahi_960p_2x2"
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
                        tiling=cfg.tiling,
                        frame_w=w_orig,
                        frame_h=h_orig,
                        seam_margin_canvas_px=cfg.tile_seam_margin_canvas_px,
                        seam_center_scale=cfg.cross_tile_seam_center_scale,
                        seam_area_ratio_threshold=cfg.cross_tile_seam_area_ratio_threshold,
                        seam_min_overlap_ratio=cfg.cross_tile_seam_min_overlap_ratio,
                    )
                )
                # MOT17-b: penalise boxes that were merged from multiple tiles.
                # Merged boxes have uncertain positions; lowering their score makes
                # ByteTracker treat them more conservatively during association.
                if cfg.cross_tile_score_penalty < 1.0:
                    merged_mask = _merge_counts > 1
                    if merged_mask.any():
                        fused_scores = fused_scores.clone()
                        fused_scores[merged_mask] = (
                            fused_scores[merged_mask] * cfg.cross_tile_score_penalty
                        )
                if cfg.tile_diagnostics:
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
                cfg.crowd_low_score_mode
                and after_merge_count >= cfg.crowd_low_score_trigger
            )
            frame_conf_threshold = (
                cfg.crowd_conf_threshold if crowd_low_active else cfg.conf_threshold
            )
            frame_track_thresh = (
                cfg.crowd_track_thresh if crowd_low_active else cfg.track_thresh
            )
            frame_mid_thresh = (
                cfg.crowd_mid_thresh if crowd_low_active else cfg.mid_thresh
            )
            frame_new_track_thresh = (
                cfg.crowd_new_track_thresh if crowd_low_active else cfg.new_track_thresh
            )
            frame_score_floor = min(frame_conf_threshold, frame_track_thresh)
            base_score_floor = min(cfg.conf_threshold, cfg.track_thresh)
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
                    w_aspect=cfg.detection_quality_w_aspect,
                    w_center=cfg.detection_quality_w_center,
                    w_area=cfg.detection_quality_w_area,
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
            if cfg.private_continuation_enabled and pre_private_boxes is not None:
                if (
                    cfg.private_prior_iou_threshold > 0.0
                    or cfg.private_prior_center_threshold > 0.0
                    or (
                        cfg.private_selection_mode
                        in {
                            "per_track",
                            "suppressor_aware",
                            "sparse_symmetric",
                            "energy",
                        }
                    )
                ):
                    private_motion_prior_boxes, _ = _build_active_track_priors(
                        detector.tracker,
                        fused_boxes.device,
                        min_track_age=0,
                        max_track_age=cfg.private_prior_max_age,
                        min_track_score=0.0,
                    )
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
                    baseline_nms_iou=cfg.nms_iou_threshold,
                    candidate_nms_iou=cfg.private_candidate_nms_iou,
                    class_aware=not cfg.track_person_only,
                    priors=private_priors,
                    prior_classes=private_prior_classes,
                    prior_iou_threshold=onms_prior_iou_threshold,
                    private_prior_boxes=private_motion_prior_boxes,
                    private_prior_iou_threshold=cfg.private_prior_iou_threshold,
                    private_prior_center_threshold=cfg.private_prior_center_threshold,
                    frame_track_thresh=frame_track_thresh,
                    frame_mid_thresh=frame_mid_thresh,
                    frame_new_track_thresh=frame_new_track_thresh,
                    low_stage_only=cfg.private_low_stage_only,
                    private_min_score=cfg.private_min_score,
                    private_max_candidates=cfg.private_max_candidates,
                    private_selection_mode=cfg.private_selection_mode,
                    private_energy_margin=cfg.private_energy_margin,
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
            if cfg.tile_diagnostics and is_tiled:
                seq_tile_diag["post_merge_seam_boxes"] += _count_tile_seam_boxes(
                    fused_boxes,
                    tiling=cfg.tiling,
                    h_orig=h_orig,
                    w_orig=w_orig,
                    seam_margin_canvas_px=cfg.tile_seam_margin_canvas_px,
                )
            if (
                cfg.tile_seam_score_penalty < 1.0
                and is_tiled
                and fused_boxes.numel() > 0
            ):
                seam_mask = _tile_seam_mask(
                    fused_boxes,
                    tiling=cfg.tiling,
                    h_orig=h_orig,
                    w_orig=w_orig,
                    seam_margin_canvas_px=cfg.tile_seam_margin_canvas_px,
                )
                if seam_mask.any():
                    fused_scores = fused_scores.clone()
                    fused_scores[seam_mask] = (
                        fused_scores[seam_mask] * cfg.tile_seam_score_penalty
                    )
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
            # Sync previous frame's background relink_write before accessing shared
            # mutable state (dynamic_reid, primary_appearance_bank, relinker).
            if state.bg_future is not None:
                if profile_stages:
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
            if (
                state.double_buffer_stream is not None
                and state.double_buffer_tracker_out_pinned
            ):
                parity = frame_id % 2
                pinned = state.double_buffer_tracker_out_pinned[parity]
                db_bufs = state.tracker_result_buffers
                for key in ("boxes", "scores", "ids", "classes", "det_idx", "count"):
                    pinned[key].copy_(db_bufs[key], non_blocking=True)
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
                    if int(c.class_id) == cfg.person_class and c.hit_streak >= 1
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
        )
        results_lines.extend(_emit_lines)

    _record_frame_timing(state, latency_started_at=t_frame_start)
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
        kwargs=kwargs,
    )
    if cfg.private_continuation_enabled:
        if cfg.workbench:
            raise ValueError(
                "private continuation is not implemented for the Workbench "
                "hot path; disable --workbench"
            )
        if cfg.private_candidate_nms_iou < cfg.nms_iou_threshold:
            raise ValueError("private-candidate-nms-iou must be >= nms-iou-threshold")

    output_root = cfg.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cheb_gr_online_log_path = output_root / "_cheb_gr_online_handover.csv"
    if getattr(cfg, "cheb_gr_online_log", False):
        cheb_gr_online_log_path.unlink(missing_ok=True)
    occ_audit_log_path = output_root / "_occ_audit.csv"
    if getattr(cfg, "occ_audit_log", False):
        occ_audit_log_path.unlink(missing_ok=True)
    if cfg.assoc_energy_diagnostics:
        scoring_profile = {
            "association_scoring_mode": cfg.association_scoring_mode,
            "multiplicative_cost": bool(
                cfg.multiplicative_cost or cfg.association_scoring_mode == "energy"
            ),
            "sinkhorn_lambda": float(cfg.sinkhorn_lambda),
            "stability_cost_w": float(cfg.stability_cost_w),
            "assoc_score_cost_w": float(cfg.assoc_score_cost_w),
            "assoc_height_cost_w": float(cfg.assoc_height_cost_w),
            "private_continuation_enabled": bool(cfg.private_continuation_enabled),
            "private_selection_mode": cfg.private_selection_mode,
            "private_energy_margin": float(cfg.private_energy_margin),
        }
        (output_root / "_association_scoring_profile.json").write_text(
            json.dumps(scoring_profile, indent=2) + "\n"
        )
    fps_summary_lines = []
    overall_latency_ms = []
    overall_throughput_frames = 0
    overall_throughput_seconds = 0.0
    debug_dump_seq = cfg.debug_dump_seq
    debug_dump_frames = _parse_debug_frame_ranges(cfg.debug_dump_frames)
    debug_dump_csv = cfg.debug_dump_csv
    debug_stage_dump_rows: list[dict[str, float | int | str]] = []
    debug_birth_csv = cfg.debug_birth_csv
    debug_birth_rows: list[dict[str, float | int | str | bool]] = []
    profile_stages = cfg.profile_stages
    detector_box_format = str(kwargs.get("detector_box_format", "xyxy"))
    stage_summary_lines = []
    global_id_mapper = GlobalTrackIdMapper()
    external_fp_rule_config = RuleBaselineConfig()
    external_fp_logistic_model = None
    if cfg.external_fp_filter_mode in {"logistic", "softmax3"}:
        model_path = Path(cfg.external_fp_logistic_model)
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

        if isinstance(detector, (MambaGatedDetector, MambaStreamProxy)):
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

    if reid_mode not in {"off", "tracker", "semantic", "hybrid"}:
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

    if cfg.tiling == "mamba_global_2x2":
        detect_fn = detect_mamba_global_2x2
    elif cfg.tiling == "960p_3x2":
        detect_fn = detect_960p_3x2_tiled
    elif cfg.tiling == "sahi_960p_2x2":
        detect_fn = detect_sahi_960p_2x2
    elif cfg.tiling == "native_640":
        detect_fn = detect_native_640
    elif cfg.tiling in ("native_960", "mamba_960", "native_1024", "native_1280"):
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
    native_private_mode = str(cfg.private_selection_mode).strip().lower()
    native_private_blockers: list[str] = []
    if cfg.private_continuation_enabled:
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
        cfg.private_continuation_enabled and not native_private_blockers
    )
    if (
        cfg.private_continuation_enabled
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
            cfg.track_thresh,
            cfg.crowd_conf_threshold if cfg.crowd_low_score_mode else conf_threshold,
            cfg.crowd_track_thresh if cfg.crowd_low_score_mode else cfg.track_thresh,
        )
        native_cfg.person_class = cfg.person_class
        native_cfg.person_only = cfg.track_person_only
        native_cfg.nms_threshold = cfg.nms_iou_threshold
        native_cfg.person_geometry_prior = cfg.person_geometry_prior
        native_cfg.geometry_suspect_support = cfg.geometry_suspect_support
        native_cfg.geometry_suspect_support_score = cfg.geometry_suspect_support_score
        native_cfg.person_min_height_ratio = cfg.person_min_height_ratio
        native_cfg.person_min_aspect = cfg.person_min_aspect
        native_cfg.person_max_aspect = cfg.person_max_aspect
        native_cfg.person_min_area_ratio = cfg.person_min_area_ratio
        native_cfg.person_max_area_ratio = cfg.person_max_area_ratio
        native_cfg.max_detections = 2048
        native_cfg.private_continuation_enabled = native_private_available
        native_cfg.private_candidate_nms_iou = cfg.private_candidate_nms_iou
        native_cfg.private_min_score = cfg.private_min_score
        native_cfg.private_max_candidates = cfg.private_max_candidates
        native_cfg.private_prior_iou_threshold = cfg.private_prior_iou_threshold
        native_cfg.private_prior_center_threshold = cfg.private_prior_center_threshold
        native_cfg.private_low_stage_only = cfg.private_low_stage_only
        native_cfg.private_track_thresh = cfg.track_thresh
        native_cfg.private_mid_thresh = cfg.mid_thresh
        native_cfg.private_new_track_thresh = cfg.new_track_thresh
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
    onms_min_track_score = cfg.high_thresh

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

    _rw_executor: ThreadPoolExecutor | None = (
        ThreadPoolExecutor(max_workers=1) if cfg.pipeline_relink else None
    )

    all_seq_profile: list[dict] = []

    # Cheb-GR offline tracklet merge (path 2) / causal online handover:
    # ReID extractor built once.
    cheb_gr_extractor = None
    cheb_gr_online = getattr(cfg, "cheb_gr_online", False)
    occ_audit_enabled = getattr(cfg, "occ_audit", False)
    if cfg.cheb_gr_merge_enabled or cheb_gr_online or occ_audit_enabled:
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
        _seq_path = Path(cfg.data_root) / cfg.split / seq
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
        )

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
                default_class_id=cfg.person_class if cfg.track_person_only else None,
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
            (output_root / "_latency_profile.json").write_text(
                json.dumps(latency_profile, indent=2) + "\n"
            )
        else:
            fps_summary_lines.append(f"{seq}\tfps=n/a\tmean_ms=n/a\tframes=0")

        if (cfg.relink_enabled or _seq_state._bridge_enabled) and hasattr(
            detector.tracker, "get_relink_debug"
        ):
            _rd = detector.tracker.get_relink_debug()
            _gates = (
                (
                    f" | no_emb={_rd[5]} bank_lt3={_rd[6]} spatial_ok={_rd[7]} "
                    f"cheb_ok={_rd[8]} floor_ok={_rd[9]} both_ok={_rd[10]}"
                )
                if len(_rd) > 10
                else ""
            )
            if len(_rd) > 11:
                _gates += f" bridge_veto={_rd[11]}"
            print(
                f"🔗 Relink debug {seq}: archived={_rd[0]} "
                f"birth_candidates={_rd[1]} revived={_rd[2]} "
                f"bridge_attempts={_rd[3]} bridge_accepts={_rd[4]}{_gates}"
            )

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
            seq_img_dir = str(Path(cfg.data_root) / cfg.split / seq / "img1")
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

        if cheb_gr_extractor is not None and cheb_gr_online:
            seq_img_dir = str(Path(cfg.data_root) / cfg.split / seq / "img1")
            head_embs, bank_embs = extract_handover_embeddings(
                _seq_state.results_lines,
                seq_img_dir,
                cheb_gr_extractor,
                decide_n=cfg.cheb_gr_online_decide_n,
                n_samples=cfg.cheb_gr_merge_n_samples,
                crop_hw=getattr(cheb_gr_extractor, "input_hw", (224, 224)),
                appearance_occlusion_cov=cfg.appearance_occlusion_cov,
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
                f"🧬 Cheb-GR Online Handover: ids={ho_stats['ids_before']}->"
                f"{ho_stats['ids_after']} ({ho_stats['handovers']} handovers, "
                f"{ho_stats['events_with_candidates']}/{ho_stats['events']} "
                "events had candidates, "
                f"reject_cost={ho_stats['reject_cost']} "
                f"reject_margin={ho_stats['reject_margin']} "
                f"reject_min_head={ho_stats['reject_min_head']})"
            )
        elif cheb_gr_extractor is not None:
            seq_img_dir = str(Path(cfg.data_root) / cfg.split / seq / "img1")
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

        if not cfg.latency_only:
            Path(output_root / f"{seq}.txt").write_text(
                "\n".join(_seq_state.results_lines)
            )
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
    if cfg.latency_only:
        return {}

    from .metrics import run_motmetrics_evaluation

    return run_motmetrics_evaluation(
        data_root=cfg.data_root,
        split=cfg.split,
        output=str(cfg.output_root),
        sequences=",".join(cfg.seqs),
        detector=cfg.kwargs.get("detector"),
        score_on_gt_frames=bool(cfg.kwargs.get("score_on_gt_frames", False)),
    )
