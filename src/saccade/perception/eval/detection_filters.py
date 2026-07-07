# mypy: ignore-errors
"""Detection-filter helpers extracted from evaluator.py.

Pure(-ish) per-frame detection filtering: stage-2 quality gate, private
continuation (wider-NMS) candidates, duplicate suppression, detection caps,
and the external FP-filter family. All operate on detection tensors plus
tracker priors; no EvalPipeline state.
"""

import numpy as np
import torch

from typing import Any

from saccade.perception.box_ops import torch_box_iou_matrix
from saccade.perception.eval.detection import nms_fast
from .external_fp_model import (
    BandedLogisticModel,
    LogisticModel,
    RuleBaselineConfig,
    SoftmaxLinearModel,
    predict_external_fp_matrix,
)


_SOFTMAX3_TORCH_CACHE: dict[
    tuple[SoftmaxLinearModel, str],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


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
