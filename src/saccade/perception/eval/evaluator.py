# mypy: ignore-errors
import configparser
import json
import os
import time
from collections import OrderedDict, deque
import dataclasses
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, Future
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
    prepare_host_track_batch as _prepare_host_track_batch,
    resolve_frame_tracks as _resolve_frame_tracks,
    prepare_track_candidates as _prepare_track_candidates,
    emit_resolved_tracks as _emit_resolved_tracks,
    finalize_frame_side_effects as _finalize_frame_side_effects,
    budget_reid_candidates as _budget_reid_candidates,
)

# Perception/eval modules load local extensions before any torchvision fallback.
from saccade.perception.cropper import ZeroCopyCropper
from .scene_adapt import SceneAdaptivePolicy


_SOFTMAX3_TORCH_CACHE: dict[
    tuple[int, str],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


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
    sx1 = sub_boxes[:, 0].unsqueeze(1)
    sy1 = sub_boxes[:, 1].unsqueeze(1)
    sx2 = sub_boxes[:, 2].unsqueeze(1)
    sy2 = sub_boxes[:, 3].unsqueeze(1)
    sa = ((sx2 - sx1) * (sy2 - sy1)).clamp(min=1.0)
    oldest_match_cx: "torch.Tensor | None" = None
    oldest_match_cy: "torch.Tensor | None" = None
    for frame_idx, prev_boxes in enumerate(window):
        if prev_boxes.numel() == 0:
            confirmed[:] = False
            break
        px1 = prev_boxes[:, 0].unsqueeze(0)
        py1 = prev_boxes[:, 1].unsqueeze(0)
        px2 = prev_boxes[:, 2].unsqueeze(0)
        py2 = prev_boxes[:, 3].unsqueeze(0)
        pa = ((px2 - px1) * (py2 - py1)).clamp(min=1.0)
        ix1 = torch.maximum(sx1, px1)
        iy1 = torch.maximum(sy1, py1)
        ix2 = torch.minimum(sx2, px2)
        iy2 = torch.minimum(sy2, py2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        union = sa + pa - inter
        iou = inter / union.clamp(min=1e-6)
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

    # Compute IoU matrix
    ax1, ay1, ax2, ay2 = (
        fused_boxes[:, 0],
        fused_boxes[:, 1],
        fused_boxes[:, 2],
        fused_boxes[:, 3],
    )
    areas = (ax2 - ax1) * (ay2 - ay1)

    ix1 = torch.maximum(ax1.unsqueeze(1), ax1.unsqueeze(0))
    iy1 = torch.maximum(ay1.unsqueeze(1), ay1.unsqueeze(0))
    ix2 = torch.minimum(ax2.unsqueeze(1), ax2.unsqueeze(0))
    iy2 = torch.minimum(ay2.unsqueeze(1), ay2.unsqueeze(0))
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    iou_matrix = inter / (areas.unsqueeze(1) + areas.unsqueeze(0) - inter).clamp(
        min=1e-6
    )

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
    _log_max = torch.tensor(
        torch.log(torch.tensor(12000.0 / 4000.0)), device=area.device
    )
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
    cache_key = (id(model), f"{device.type}:{device.index}:{dtype}")
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
    if not low_score_mask.any():
        return boxes, scores, classes

    keep = torch.ones(scores.shape[0], dtype=torch.bool, device=boxes.device)
    subset_boxes = boxes[low_score_mask]
    subset_scores = scores[low_score_mask]
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
            adjusted_scores[low_score_mask] = subset_scores * score_scale.to(
                dtype=subset_scores.dtype
            )
            subset_keep = adjusted_scores[low_score_mask] >= min_score
    else:
        raise ValueError(f"Unknown external FP filter mode: {mode}")

    if penalty < 0.999 or mode == "rule_score":
        penalized_scores = subset_scores.clone()
        penalized_scores[~subset_keep] = penalized_scores[~subset_keep] * penalty
        adjusted_scores[low_score_mask] = penalized_scores
        subset_keep = penalized_scores >= min_score
    keep[low_score_mask] = subset_keep
    if not keep.any():
        return boxes[:0], scores[:0], classes[:0]
    return boxes[keep], adjusted_scores[keep], classes[keep]


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
from saccade.perception.eval.gmc import SparseOpticalFlowGMC  # noqa: E402
from saccade.perception.eval.multi_birth import MultiSignalBirthManager  # noqa: E402
from saccade.perception.eval.pool import AdaptiveFramePool  # noqa: E402
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
    from saccade_tracking_ext import PerceptionPipeline, PerceptionPipelineConfig
except ImportError:
    PerceptionPipeline = None
    PerceptionPipelineConfig = None


# Functions moved to output_bank.py and helpers.py


# Frame tracking helpers moved to helpers.py and utils.py


# Internal helpers moved to helpers.py, quality.py, and utils.py


# Post-merge functions moved to post_merge.py


def _build_active_track_priors(
    tracker: Any,
    device: torch.device,
    *,
    min_track_age: int = 0,
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
    if not value:
        return default
    return value.lower() not in {"0", "false"}


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

    t0 = time.monotonic()
    cpp_results = pool.run_sequences(seq_configs)  # GIL released here
    elapsed = time.monotonic() - t0
    print(
        f"[run_eval_cpp] {len(cfg.seqs)} sequences in {elapsed:.1f}s "
        f"({n_threads} threads)"
    )

    # Cheb-GR offline tracklet merge (path 2): build the siglip2_reid extractor
    # once. C++ eval emits no per-det embedding, so tracklet crops are re-cut
    # from img1 inside the post-process loop.
    cheb_gr_extractor = None
    if cfg.cheb_gr_merge_enabled:
        from .cheb_gr_merge import (
            cheb_gr_merge_output_tracklets,
            extract_tracklet_embeddings,
        )

        cheb_gr_extractor = TRTFeatureExtractor(
            engine_path=cfg.cheb_gr_engine,
            model_type="siglip2_reid",
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

        if cheb_gr_extractor is not None:
            seq_img_dir = str(_Path(cfg.data_root) / cfg.split / seq / "img1")
            cheb_embeddings = extract_tracklet_embeddings(
                results_lines,
                seq_img_dir,
                cheb_gr_extractor,
                n_samples=cfg.cheb_gr_merge_n_samples,
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
    )


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

    output_root = cfg.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    fps_summary_lines = []
    overall_latency_ms = []
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

    if cfg.tiling == "960p_3x2":
        detect_fn = detect_960p_3x2_tiled
    elif cfg.tiling == "sahi_960p_2x2":
        detect_fn = detect_sahi_960p_2x2
    elif cfg.tiling == "native_640":
        detect_fn = detect_native_640
    elif cfg.tiling == "native_960" or cfg.tiling == "mamba_960":
        detect_fn = (
            detect_native_960_tta if getattr(cfg, "tta", False) else detect_native_960
        )
    else:
        detect_fn = detect_adaptive_960_tiled

    extractor_cpp_ptr = _safe_cpp_ptr(extractor) if extractor is not None else 0
    cropper_cpp_ptr = _safe_cpp_ptr(cropper) if cropper is not None else 0
    native_postprocess_available = (
        PerceptionPipeline is not None and PerceptionPipelineConfig is not None
    )
    native_reid_available = (
        native_postprocess_available
        and extractor_cpp_ptr != 0
        and cropper_cpp_ptr != 0
        and cfg.reid_crop_layout == "full"
    )
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

    # Cheb-GR offline tracklet merge (path 2): siglip2_reid extractor built once.
    cheb_gr_extractor = None
    if cfg.cheb_gr_merge_enabled:
        from .cheb_gr_merge import (
            cheb_gr_merge_output_tracklets,
            extract_tracklet_embeddings,
        )

        cheb_gr_extractor = TRTFeatureExtractor(
            engine_path=cfg.cheb_gr_engine,
            model_type="siglip2_reid",
            max_batch=64,
        )

    for seq in cfg.seqs:
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
            if _fpn_reid_mode:
                from saccade.perception.tracking import GPUByteTracker

                detector.tracker = GPUByteTracker(
                    max_objects=2048, embedding_dim=_fpn_reid_dim
                )
            else:
                detector.reset_tracker()

        geometry_scale_state = GeometryScaleState()

        # A8: Uniform CMC & 2D MMD
        gmc_estimator = None
        if cfg.gmc_enabled:
            if cfg.gmc_mode == "gpu":
                try:
                    from saccade_tracking_ext import GMC as CppGMC

                    gmc_estimator = CppGMC(downscale=cfg.gmc_downscale)
                    if hasattr(gmc_estimator, "set_profiling_enabled"):
                        gmc_estimator.set_profiling_enabled(profile_stages)
                except ImportError:
                    gmc_estimator = SparseOpticalFlowGMC(downscale=cfg.gmc_downscale)
            else:
                gmc_estimator = SparseOpticalFlowGMC(downscale=cfg.gmc_downscale)

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
        if _fpn_reid_mode and hasattr(detector.tracker, "set_reid_min_candidates"):
            detector.tracker.set_reid_min_candidates(1)
        if cfg.relink_enabled and hasattr(detector.tracker, "set_relink_params"):
            detector.tracker.set_relink_params(
                enabled=True,
                bank_cap=cfg.relink_bank_cap,
                sim_thresh=cfg.relink_sim_thresh,
                cheb_lambda=cfg.relink_lambda,
                spatial_gate=cfg.relink_spatial_gate,
                max_age=cfg.relink_max_age,
            )

        if hasattr(detector.tracker, "set_unified_score_params"):
            detector.tracker.set_unified_score_params(
                w_sim_base=cfg.semantic_w_sim_base,
                w_iou_base=cfg.semantic_w_iou_base,
                w_maha_base=cfg.semantic_w_maha_base,
                shift_ambiguity=cfg.semantic_shift_ambiguity,
                shift_lost_age=cfg.semantic_shift_lost_age,
            )

        _use_python_relinker = (
            cfg.force_python_relinker or cfg.semantic_rerank_mode != "mean"
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
            kalman_fps=cfg.kwargs.get("semantic_kalman_fps", 30.0),
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
        relinker = (
            _relinker_cls(**_relinker_common_kwargs) if cfg.use_semantic_mode else None
        )

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
        if not (seq_path / "seqinfo.ini").exists():
            continue
        config = configparser.ConfigParser()
        config.read(seq_path / "seqinfo.ini")
        w_orig = config.getint("Sequence", "imWidth")
        h_orig = config.getint("Sequence", "imHeight")
        frame_end = min(max_frames or int(1e9), config.getint("Sequence", "seqLength"))
        seq_fps = config.getint("Sequence", "frameRate", fallback=30)

        # F-1: Per-sequence adaptive params — scale temporal params by fps/30
        seq_reid_interval = cfg.reid_interval
        seq_track_buffer = 30
        if cfg.per_seq_adapt and seq_fps != 30:
            fps_scale = seq_fps / 30.0
            seq_reid_interval = max(1, round(cfg.reid_interval * fps_scale))
            seq_track_buffer = max(10, round(30 * fps_scale))
        detector.tracker.set_frame_size(w_orig, h_orig)
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
            nsa_kalman=cfg.nsa_kalman,
            r_scale=cfg.kalman_r_scale,
            vel_dir_weight=cfg.vel_dir_weight,
            fuse_score_weight=cfg.fuse_score_weight,
            stage2_match_thresh=cfg.stage2_match_thresh,
            birth_low_score_thresh=cfg.birth_low_score_thresh,
            birth_prox_norm_thresh=cfg.birth_prox_norm_thresh,
        )
        detector.tracker.set_oao_params(cfg.oao_tau)
        active_tracker_thresholds = (
            cfg.track_thresh,
            cfg.mid_thresh,
            cfg.new_track_thresh,
        )

        pool = AdaptiveFramePool(h_orig, w_orig)
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

        prev_track_ids: set[int] = set()
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
            for name in ("raw_boxes", "after_filter", "after_nms", "after_merge")
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
        seq_lazy_reid_candidates = 0
        seq_lazy_reid_frames = 0
        seq_lazy_reid_crops = 0
        seq_lazy_reid_self_pairs = 0
        seq_lazy_reid_self_pass = 0
        seq_lazy_reid_self_sim_sum = 0.0
        seq_lazy_reid_arbiter_checks = 0
        seq_lazy_reid_arbiter_approve = 0
        lazy_reid_prev_embeddings: dict[int, torch.Tensor] = {}
        seq_profiled_frames = 0
        last_reid_frame = -100
        gmc_warp = None
        gmc_uncertain = False
        prev_gray: torch.Tensor | None = None  # for GMC (workbench path)
        tracker_result_buffers = detector.tracker.allocate_result_buffers(
            device=pool.frame_buffer.device
        )

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
                        "frame": int(frame_id),
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

        _bg_future: "Future[tuple[list[str], set[int], dict[int, int], dict[int, dict[str, float | int]]]] | None" = None
        _bg_birth_events: list[dict[str, float | int | str | bool]] | None = None

        for frame_id in range(1, frame_end + 1):
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
            try:
                frame_gpu, _fetch_ms = time_stage(
                    seq_stage_totals,
                    "fetch",
                    lambda: next(stream_iter),
                    sync_cuda=False,
                )
            except StopIteration:
                break
            t_frame_start = time.perf_counter()

            if getattr(cfg, "workbench", False) and wb is not None:
                # Step 1: ingest + preprocess (same as non-workbench path)
                _, _ = time_stage(
                    seq_stage_totals,
                    "ingest_preprocess",
                    lambda: (
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
                    if (
                        wb_scene_policy.is_classified
                        and wb_scene_policy.stats is not None
                    ):
                        st = wb_scene_policy.stats
                        if st.scene_type == "crowded_narrow":
                            seq_narrow_bonus = cfg.narrow_person_score_bonus
                        wb.narrow_bonus = seq_narrow_bonus
                        print(
                            f"  [scene_adapt] {seq} @ frame {frame_id}: {st}"
                            + (
                                f" → narrow_bonus={seq_narrow_bonus:.2f}"
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
                        frame_chw=pool.frame_buffer,
                        frame_id=frame_id,
                        last_reid_frame=last_reid_frame,
                        prev_gray=prev_gray,
                        is_tiled=is_tiled,
                    ),
                    sync_cuda=True,
                )
                # Update scene-adapt narrow bonus from workbench
                seq_narrow_bonus = wb.narrow_bonus
                last_reid_frame = wb.last_reid_frame

                # D2H once, share across tracker_result_buffers, track_results, and
                # MOT line writing — avoids 10 redundant device syncs (was 13 .cpu()
                # calls per frame × 7 threads = 91 syncs/cycle; now 5).
                wb_count = int(len(wb_result.ids))
                _wb_boxes_cpu = wb_result.boxes.cpu()
                _wb_scores_cpu = wb_result.scores.cpu()
                _wb_ids_cpu = wb_result.ids.cpu()
                _wb_classes_cpu = wb_result.classes.cpu()
                _wb_det_idx_cpu = wb_result.det_idx.cpu()

                tracker_result_buffers = {
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
                    prev_track_ids = set(wb_ids_np.tolist())

                # Mock missing variables for downstream compatibility
                fused_boxes = wb_result.boxes
                fused_scores = wb_result.scores
                fused_classes = wb_result.classes
                geometry_suspect_mask = torch.zeros(
                    len(fused_boxes), dtype=torch.bool, device=fused_boxes.device
                )
                embeddings = None
                gmc_warp = None
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
                prev_gray = (
                    (
                        0.299 * pool.frame_buffer[2]
                        + 0.587 * pool.frame_buffer[1]
                        + 0.114 * pool.frame_buffer[0]
                    )
                    .unsqueeze(0)
                    .clone()
                )
            else:
                _, _ = time_stage(
                    seq_stage_totals,
                    "ingest_preprocess",
                    lambda: (
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
                    ),
                    sync_cuda=True,
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
                    sync_cuda=True,
                )

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
                            seq_narrow_bonus = cfg.narrow_person_score_bonus
                        print(
                            f"  [scene_adapt] {seq} @ frame {frame_id}: {st}"
                            + (
                                f" → narrow_bonus={seq_narrow_bonus:.2f}"
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
                    if frame_id > warmup_frames:
                        frame_latencies.append(
                            (time.perf_counter() - t_frame_start) * 1000
                        )
                    if profile_stages and frame_id > warmup_frames:
                        seq_stage_totals["frame_total"] += (
                            time.perf_counter() - t_e2e_start
                        ) * 1000
                        seq_profiled_frames += 1
                    if frame_id % 100 == 0:
                        print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                    continue

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

                if perception_pipeline is not None:
                    t_native_prep_start = None
                    if profile_stages:
                        torch.cuda.synchronize()
                        t_native_prep_start = time.perf_counter()
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
                        post_boxes = torch.empty_like(raw_boxes_contig)
                        post_scores = torch.empty_like(raw_scores_contig)
                        post_classes = torch.empty_like(raw_classes_contig)
                        geometry_suspect_mask = torch.empty(
                            (raw_box_count,),
                            device=raw_boxes_contig.device,
                            dtype=torch.bool,
                        )

                        # Fetch priors for Occlusion-aware NMS
                        priors_tensor = None
                        prior_classes_tensor = None
                        priors_ptr = 0
                        prior_classes_ptr = 0
                        num_priors = 0
                        if enable_onms:
                            priors_tensor, prior_classes_tensor = (
                                _build_active_track_priors(
                                    detector.tracker,
                                    raw_boxes_contig.device,
                                    min_track_age=onms_min_track_age,
                                    min_track_score=onms_min_track_score,
                                )
                            )
                        if (
                            enable_onms
                            and priors_tensor is not None
                            and prior_classes_tensor is not None
                        ):
                            priors_ptr = priors_tensor.data_ptr()
                            prior_classes_ptr = prior_classes_tensor.data_ptr()
                            num_priors = priors_tensor.size(0)
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

                    # process_detections_n releases GIL for the full filter+NMS+sync
                    # sequence so sibling threads can run Python while GPU is busy.
                    with _record_profile_scope("post.native_process_detections_n"):
                        n_post = perception_pipeline.process_detections_n(
                            raw_boxes_contig.data_ptr(),
                            raw_scores_contig.data_ptr(),
                            raw_classes_contig.data_ptr(),
                            raw_box_count,
                            w_orig,
                            h_orig,
                            is_tiled,
                            post_boxes.data_ptr(),
                            post_scores.data_ptr(),
                            post_classes.data_ptr(),
                            geometry_suspect_mask.data_ptr(),
                            priors_ptr,
                            prior_classes_ptr,
                            num_priors,
                            onms_prior_iou_threshold,
                            torch.cuda.current_stream().cuda_stream,
                        )
                    if profile_stages and current_stage_sample_active:
                        _post_stats = (
                            perception_pipeline.get_postprocess_profile_stats()
                        )
                        _post_filter_ms = float(_post_stats.get("filter_ms", 0.0))
                        _post_nms_ms = float(_post_stats.get("nms_ms", 0.0))
                        _post_count_sync_ms = float(
                            _post_stats.get("count_d2h_ms", 0.0)
                        )
                        _post_total_ms = float(_post_stats.get("total_ms", 0.0))
                        seq_stage_totals["post_filter"] += float(_post_filter_ms)
                        seq_stage_totals["post_nms"] += float(_post_nms_ms)
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
                        seq_stage_totals["post_native_other"] += max(
                            0.0,
                            _post_total_ms
                            - _post_filter_ms
                            - _post_nms_ms
                            - _post_count_sync_ms,
                        )
                    if profile_stages and current_stage_sample_active:
                        _seg_ev = torch.cuda.Event(enable_timing=True)
                        _seg_ev.record(torch.cuda.current_stream())
                        post_seg_events.append(("post_seg_native", _seg_ev))
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
                else:
                    fused_scores = _apply_narrow_person_score_bonus(
                        fused_boxes,
                        fused_scores,
                        fused_classes,
                        frame_w=w_orig,
                        frame_h=h_orig,
                        person_class=cfg.person_class,
                        bonus=seq_narrow_bonus,
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
                    if frame_id > warmup_frames:
                        frame_latencies.append(
                            (time.perf_counter() - t_frame_start) * 1000
                        )
                    if profile_stages and frame_id > warmup_frames:
                        seq_stage_totals["frame_total"] += (
                            time.perf_counter() - t_e2e_start
                        ) * 1000
                        seq_profiled_frames += 1
                    if frame_id % 100 == 0:
                        print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                    continue

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
                        seq_tile_diag["merged_clusters"] += int(
                            merged_mask.sum().item()
                        )
                        seq_tile_diag["merged_members"] += int(
                            _merge_counts[merged_mask].sum().item()
                        )
                        seq_tile_diag["merged_outputs"] += int(_merge_counts.numel())
                    geometry_suspect_mask = torch.zeros_like(
                        fused_scores, dtype=torch.bool
                    )
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
                    cfg.crowd_new_track_thresh
                    if crowd_low_active
                    else cfg.new_track_thresh
                )
                frame_score_floor = min(frame_conf_threshold, frame_track_thresh)
                base_score_floor = min(cfg.conf_threshold, cfg.track_thresh)
                t_tail_filtering_start = None
                if profile_stages:
                    torch.cuda.synchronize()
                    t_tail_filtering_start = time.perf_counter()
                with _record_profile_scope("post.tail_filtering"):
                    if (
                        frame_score_floor > base_score_floor
                        and fused_scores.numel() > 0
                    ):
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
                    fused_boxes, fused_scores, fused_classes = (
                        _apply_external_fp_filter(
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
                    fused_scores = fused_scores.masked_fill(
                        _fp_reject, _FP_HARD_REJECT_SCORE
                    )
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

                # === Stage 2 Quality Gate ===
                # Remove mid-score-band detections with poor geometry before the tracker's
                # Stage 2 association step, preventing bad lost-track assignments → IDs.
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

                frame_birth_events: list[dict[str, float | int | str | bool]] = []

                # === Consecutive-Frame Birth Gate ===
                # Boost sub-threshold detections that have appeared in the last N frames.
                # More selective than birth_quality_gate: requires temporal evidence, not
                # just per-frame geometry quality.
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
                                boost_idx = below_birth.nonzero(as_tuple=True)[0][
                                    confirmed
                                ]
                                # Only boost detections that are close enough to new_track_thresh
                                # (above min_score) — prevents very-low-score noise from crossing
                                eligible = (
                                    fused_scores[boost_idx]
                                    >= cfg.birth_consecutive_min_score
                                )
                                boost_idx = boost_idx[eligible]
                                if boost_idx.numel() > 0:
                                    score_before = fused_scores[boost_idx].clone()
                                    fused_scores = fused_scores.clone()
                                    fused_scores[boost_idx] = torch.clamp(
                                        fused_scores[boost_idx]
                                        + cfg.birth_consecutive_boost,
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
                        boost_idx = boost_mask.nonzero(as_tuple=True)[0]
                        score_before = fused_scores[boost_idx].clone()
                        boost = (
                            birth_quality[boost_mask] - cfg.birth_min_quality
                        ) * cfg.birth_quality_score_bias
                        fused_scores = fused_scores.clone()
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
                            boost_idx = cand_mask.nonzero(as_tuple=True)[0][
                                promote_mask
                            ]
                            score_before = fused_scores[boost_idx].clone()
                            fused_scores = fused_scores.clone()
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
                        _multi_birth_manager.update(
                            frame_id, fused_boxes[:0], fused_scores[:0]
                        )

                frame_tracker_thresholds = (
                    frame_track_thresh,
                    frame_mid_thresh,
                    frame_new_track_thresh,
                )
                if frame_tracker_thresholds != active_tracker_thresholds:
                    detector.tracker.set_params(
                        track_thresh=frame_track_thresh,
                        high_thresh=cfg.high_thresh,
                        match_thresh=cfg.match_thresh,
                        track_buffer=seq_track_buffer,
                        mid_thresh=frame_mid_thresh,
                        confirm_streak=int(cfg.kwargs.get("confirm_streak", 1)),
                        confirm_score_thresh=float(
                            cfg.kwargs.get("confirm_score_thresh", 0.0)
                        ),
                        adaptive_confirmation=bool(
                            cfg.kwargs.get("adaptive_confirmation", False)
                        ),
                        new_track_thresh=frame_new_track_thresh,
                        nsa_kalman=cfg.nsa_kalman,
                        r_scale=cfg.kalman_r_scale,
                        vel_dir_weight=cfg.vel_dir_weight,
                        fuse_score_weight=cfg.fuse_score_weight,
                        stage2_match_thresh=cfg.stage2_match_thresh,
                        birth_low_score_thresh=cfg.birth_low_score_thresh,
                        birth_prox_norm_thresh=cfg.birth_prox_norm_thresh,
                    )
                    active_tracker_thresholds = frame_tracker_thresholds
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
                        seq_post_counts["after_merge"] += after_merge_count
                # Sync previous frame's background relink_write before accessing shared
                # mutable state (dynamic_reid, primary_appearance_bank, relinker).
                if _bg_future is not None:
                    if profile_stages:
                        t_bg_wait_start = time.perf_counter()
                    (
                        _bg_rw_lines,
                        prev_track_ids,
                        _bg_det_idx_to_local_id,
                        _bg_output_by_local,
                    ) = _bg_future.result()
                    if profile_stages:
                        elapsed_ms = (time.perf_counter() - t_bg_wait_start) * 1000
                        seq_stage_totals["bg_relink_wait"] += elapsed_ms
                        record_stage_sample("bg_relink_wait", elapsed_ms)
                    results_lines.extend(_bg_rw_lines)
                    if _bg_birth_events is not None:
                        _annotate_birth_events(
                            _bg_birth_events,
                            _det_idx_to_local_id=_bg_det_idx_to_local_id,
                            _output_by_local=_bg_output_by_local,
                        )
                    _bg_future = None
                    _bg_birth_events = None

                embeddings = None
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
                        time_since_last_reid = frame_id - last_reid_frame

                        if time_since_last_reid < MIN_REID_GAP:
                            _do_reid = False
                        elif cfg.need_reid_enabled:
                            if dynamic_reid is not None:
                                _do_reid = dynamic_reid.should_reid(after_merge_count)
                            else:
                                _do_reid = need_reid_frame(
                                    prev_track_ids, after_merge_count
                                )
                        else:
                            _do_reid = frame_id % seq_reid_interval == 0

                    if _do_reid:
                        last_reid_frame = frame_id
                        if primary_appearance_bank is not None:
                            if profile_stages:
                                torch.cuda.synchronize()
                                t_reid_bank_sync_start = time.perf_counter()
                            bank_reps = primary_appearance_bank.representatives()
                            if bank_reps:
                                detector.tracker.set_reference_features_from_bank(
                                    bank_reps
                                )
                            clean_ids = primary_appearance_bank.clean_ids()
                            if clean_ids:
                                _clean_ids_list = list(clean_ids)
                                _ids_t = torch.tensor(
                                    _clean_ids_list, dtype=torch.int32
                                )
                                _flags_t = torch.ones(
                                    len(_clean_ids_list), dtype=torch.bool
                                )
                                detector.tracker.set_clean_embedding_flags(
                                    _ids_t, _flags_t
                                )
                            else:
                                detector.tracker.set_clean_embedding_flags(
                                    torch.zeros(0, dtype=torch.int32),
                                    torch.zeros(0, dtype=torch.bool),
                                )
                            if profile_stages:
                                torch.cuda.synchronize()
                                elapsed_ms = (
                                    time.perf_counter() - t_reid_bank_sync_start
                                ) * 1000
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
                            _img_sz = (
                                int(_p3_cache.shape[2] * 8)
                                if _p3_cache is not None
                                else 640
                            )
                        elif hasattr(detector, "teacher"):
                            _p3_cache = detector.teacher._gate_layers[
                                "p3"
                            ]._feat_cache.get("p3")
                            _img_sz = (
                                int(_p3_cache.shape[2] * 8)
                                if _p3_cache is not None
                                else 640
                            )
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
                                embeddings = detector.extract_fpn_embeddings(
                                    None, boxes_rescaled
                                )
                    else:
                        if profile_stages:
                            torch.cuda.synchronize()
                            t_reid_budget_start = time.perf_counter()

                        num_dets = fused_boxes.shape[0]
                        if cfg.reid_budget_raw >= 1.0:
                            actual_budget = int(cfg.reid_budget_raw)
                        elif cfg.reid_budget_raw > 0.0:
                            actual_budget = max(1, int(cfg.reid_budget_raw * num_dets))
                        else:
                            actual_budget = (
                                0  # Unlimited or handled by _budget_reid_candidates
                            )

                        budget_indices = _budget_reid_candidates(
                            fused_boxes,
                            fused_scores,
                            actual_budget,
                            dynamic_reid=dynamic_reid,
                            gmc_warp=gmc_warp if cfg.gmc_enabled else None,
                            gmc_uncertain=gmc_uncertain,
                        )

                        if profile_stages:
                            torch.cuda.synchronize()
                            elapsed_ms = (
                                time.perf_counter() - t_reid_budget_start
                            ) * 1000
                            seq_stage_totals["reid_budget"] += elapsed_ms
                            record_stage_sample("reid_budget", elapsed_ms)

                        _reid_feat_dim = (
                            _fpn_reid_dim if _fpn_reid_mode else extractor.feature_dim
                        )
                        # Initialize full embeddings with zeros. Detections without budget
                        # will have neutral features for association.
                        embeddings = torch.zeros(
                            (fused_boxes.shape[0], _reid_feat_dim),
                            device=fused_boxes.device,
                            dtype=torch.float32,
                        )

                        if budget_indices.numel() > 0:
                            budgeted_boxes = fused_boxes[budget_indices].contiguous()

                            if (
                                native_reid_available
                                and perception_pipeline is not None
                            ):
                                frame_hwc = pool.frame_buffer.permute(
                                    1, 2, 0
                                ).contiguous()
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
                                            native_stats = perception_pipeline.get_reid_profile_stats()
                                            seq_native_reid_samples[
                                                "native_reid_crop"
                                            ].append(
                                                float(native_stats.get("crop_ms", 0.0))
                                            )
                                            seq_native_reid_samples[
                                                "native_reid_pre_normalize"
                                            ].append(
                                                float(
                                                    native_stats.get(
                                                        "extract_pre_normalize_ms", 0.0
                                                    )
                                                )
                                            )
                                            seq_native_reid_samples[
                                                "native_reid_trt_enqueue"
                                            ].append(
                                                float(
                                                    native_stats.get(
                                                        "extract_trt_enqueue_ms", 0.0
                                                    )
                                                )
                                            )
                                            seq_native_reid_samples[
                                                "native_reid_l2_normalize"
                                            ].append(
                                                float(
                                                    native_stats.get(
                                                        "extract_l2_normalize_ms", 0.0
                                                    )
                                                )
                                            )
                                    embeddings[budget_indices] = budget_embeddings
                            else:
                                frame_batch = pool.frame_buffer.unsqueeze(0)
                                if cfg.reid_crop_layout == "parts":
                                    if profile_stages:
                                        torch.cuda.synchronize()
                                        t_reid_crop_start = time.perf_counter()
                                    crops = cropper.process_parts(
                                        frame_batch, budgeted_boxes
                                    )
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
                                        budget_embeddings = (
                                            extractor.extract_parts_fused(crops)
                                        )
                                        if profile_stages:
                                            torch.cuda.synchronize()
                                            elapsed_ms = (
                                                time.perf_counter()
                                                - t_reid_extract_start
                                            ) * 1000
                                            seq_stage_totals["reid_extract"] += (
                                                elapsed_ms
                                            )
                                            record_stage_sample(
                                                "reid_extract", elapsed_ms
                                            )
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
                                                time.perf_counter()
                                                - t_reid_extract_start
                                            ) * 1000
                                            seq_stage_totals["reid_extract"] += (
                                                elapsed_ms
                                            )
                                            record_stage_sample(
                                                "reid_extract", elapsed_ms
                                            )
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
                gmc_warp = None
                gmc_uncertain = False
                if gmc_estimator is not None:

                    def _run_gmc() -> tuple[torch.Tensor | None, bool]:
                        local_gmc_warp: torch.Tensor | None = None
                        local_gmc_uncertain = False
                        _raw_warp = None
                        # A4: foreground mask — zero out current-frame detection regions before FFT.
                        if cfg.gmc_fg_mask and hasattr(
                            gmc_estimator, "set_fg_mask_boxes"
                        ):
                            if fused_boxes.numel() > 0:
                                _flat = fused_boxes.detach().cpu().view(-1).tolist()
                                gmc_estimator.set_fg_mask_boxes(_flat)

                        if hasattr(gmc_estimator, "estimate_into"):
                            local_gmc_warp = torch.empty(
                                6, dtype=torch.float32, device=fused_boxes.device
                            )
                            gmc_estimator.estimate_into(
                                pool.frame_buffer.data_ptr(),
                                pool.frame_buffer.shape[2],
                                pool.frame_buffer.shape[1],
                                torch.cuda.current_stream().cuda_stream,
                                local_gmc_warp.data_ptr(),
                            )
                        elif hasattr(gmc_estimator, "estimate_mat"):
                            # C++ version or GlobalMotionCompensator
                            _raw_warp = gmc_estimator.estimate(
                                pool.frame_buffer.data_ptr(),
                                w_orig,
                                h_orig,
                                torch.cuda.current_stream().cuda_stream,
                            )
                        else:
                            _raw_warp = gmc_estimator.estimate(pool.frame_buffer)

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
                                local_gmc_warp is not None
                                and 0.0 < _pcr < cfg.gmc_pcr_uncertain_thresh
                            )
                        if profile_stages and hasattr(
                            gmc_estimator, "get_profile_stats"
                        ):
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

                    (gmc_warp, gmc_uncertain), _ = time_stage(
                        seq_stage_totals,
                        "gmc",
                        _run_gmc,
                        sync_cuda=hasattr(gmc_estimator, "estimate_mat"),
                    )
                    if hasattr(detector, "set_gmc_warp"):
                        detector.set_gmc_warp(gmc_warp, h_orig, w_orig)
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

                if gtu is not None:
                    gtu.copy_inputs(
                        fused_boxes,
                        fused_scores,
                        fused_classes.to(torch.int32),
                        gmc=gmc_warp,
                    )
                    # replay() returns gtu.out_* tensors directly; use them as
                    # tracker_result_buffers to skip the extra D2D copy + item() sync
                    # that read_outputs() would introduce.
                    tracker_result_buffers, _ = time_stage(
                        seq_stage_totals,
                        "track",
                        lambda: gtu.replay(),
                        sync_cuda=True,
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
                        sync_cuda=True,
                    )
                track_results, _ = time_stage(
                    seq_stage_totals,
                    "materialize",
                    lambda: _materialize_gpu_track_results(
                        tracker_result_buffers,
                        default_class_id=cfg.person_class
                        if cfg.track_person_only
                        else None,
                        include_det_idx=(
                            embeddings is not None or aligned_keypoints is not None
                        ),
                    ),
                    sync_cuda=True,
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
                seq_lazy_reid_candidates += len(ready_candidates)
                seq_lazy_reid_frames += 1
                if (
                    cfg.profile_lazy_reid_embeddings
                    and extractor
                    and cropper
                    and candidates
                ):
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
                        crops = cropper.process(
                            pool.frame_buffer.unsqueeze(0), cand_boxes
                        )
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
                    seq_lazy_reid_crops += crop_count
                    seq_lazy_reid_self_pairs += pair_count
                    seq_lazy_reid_self_pass += pass_count
                    seq_lazy_reid_self_sim_sum += sim_sum
                    seq_lazy_reid_arbiter_checks += arbiter_checks
                    seq_lazy_reid_arbiter_approve += arbiter_approve
                    if seen_ids:
                        for stale_id in (
                            set(lazy_reid_prev_embeddings.keys()) - seen_ids
                        ):
                            lazy_reid_prev_embeddings.pop(stale_id, None)

            if cfg.pipeline_relink and not getattr(cfg, "workbench", False):
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
                        _pm_motion_snaps = (
                            detector.tracker.get_motion_snapshots_for_track_ids(
                                _pm_motion_cids
                            )
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
                results_lines.extend(frame_result_lines)
                curr_track_ids = set(host_track_batch.ids)
                lifecycle_merger.prune(frame_id)
                prev_track_ids = _finalize_frame_side_effects(
                    curr_track_ids=curr_track_ids,
                    prev_track_ids=prev_track_ids,
                    relinker=relinker,
                    semantic_bank_inject=cfg.semantic_bank_inject,
                    primary_appearance_bank=primary_appearance_bank,
                    dynamic_reid=dynamic_reid,
                    person_observations=host_track_batch.person_observations,
                    gmc_warp=gmc_warp,
                    gmc_enabled=cfg.gmc_enabled,
                )
                if profile_stages:
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t_relink_write_start) * 1000
                    seq_stage_totals["relink_write"] += elapsed_ms
                    record_stage_sample("relink_write", elapsed_ms)

            if frame_id > warmup_frames:
                frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
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
                seq_profiled_frames += 1
            if frame_id % 100 == 0:
                print(f"🎬 {seq} [{frame_id}/{frame_end}]")

        # Flush any last background relink_write future before post-processing results.
        if _bg_future is not None:
            (
                _bg_rw_lines,
                prev_track_ids,
                _bg_det_idx_to_local_id,
                _bg_output_by_local,
            ) = _bg_future.result()
            results_lines.extend(_bg_rw_lines)
            if _bg_birth_events is not None:
                _annotate_birth_events(
                    _bg_birth_events,
                    _det_idx_to_local_id=_bg_det_idx_to_local_id,
                    _output_by_local=_bg_output_by_local,
                )
            _bg_future = None
            _bg_birth_events = None

        if frame_latencies:
            lats = np.array(frame_latencies)
            mean_ms = float(np.mean(lats))
            fps = 1000.0 / mean_ms
            print(f"\n📊 Production Latency Report for {seq}:")
            print(f"  - FPS:  {fps:.2f}")
            print(f"  - Mean latency: {mean_ms:.2f} ms")
            fps_summary_lines.append(
                f"{seq}\tfps={fps:.2f}\tmean_ms={mean_ms:.2f}\tframes={len(frame_latencies)}"
            )
            overall_latency_ms.extend(frame_latencies)
            latency_profile = {
                "sequence": seq,
                "frames": len(frame_latencies),
                "fps": round(fps, 4),
                "mean_ms": round(mean_ms, 6),
                "std_ms": round(float(np.std(lats)), 6),
                "p95_ms": round(float(np.percentile(lats, 95)), 6),
                "p99_ms": round(float(np.percentile(lats, 99)), 6),
                "samples_ms": [round(float(x), 6) for x in frame_latencies],
            }
            (output_root / "_latency_profile.json").write_text(
                json.dumps(latency_profile, indent=2) + "\n"
            )
        else:
            fps_summary_lines.append(f"{seq}\tfps=n/a\tmean_ms=n/a\tframes=0")

        if cfg.relink_enabled and hasattr(detector.tracker, "get_relink_debug"):
            _rd = detector.tracker.get_relink_debug()
            print(
                f"🔗 Relink debug {seq}: archived={_rd[0]} "
                f"birth_candidates={_rd[1]} revived={_rd[2]}"
            )

        results_lines, post_merge_stats = post_merge_output_tracklets(
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
            appearance_bank=output_appearance_bank,
            appearance_gate=cfg.post_lifecycle_appearance_gate,
            appearance_threshold=cfg.post_lifecycle_appearance_threshold,
            appearance_min_samples=cfg.post_lifecycle_appearance_min_samples,
            appearance_weight=cfg.post_lifecycle_appearance_weight,
            gap_uncertainty_weight=cfg.post_lifecycle_gap_uncertainty_weight,
            consistency_weight=cfg.post_lifecycle_consistency_weight,
            missing_appearance_cost=cfg.post_lifecycle_missing_appearance_cost,
        )

        if cheb_gr_extractor is not None:
            seq_img_dir = str(Path(cfg.data_root) / cfg.split / seq / "img1")
            cheb_embeddings = extract_tracklet_embeddings(
                results_lines,
                seq_img_dir,
                cheb_gr_extractor,
                n_samples=cfg.cheb_gr_merge_n_samples,
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

        results_lines, quality_stats = filter_low_quality_tracklets(
            results_lines,
            min_len=cfg.min_tracklet_len,
            min_score=cfg.min_tracklet_score,
        )
        if quality_stats["removed"] > 0:
            print(
                f"🧹 Quality Filter: removed={quality_stats['removed']} "
                f"ids={quality_stats['before']}->{quality_stats['after']}"
            )

        if cfg.interpolate_tracklets:
            results_lines, interp_stats = interpolate_tracklets(
                results_lines,
                max_gap=cfg.interpolate_max_gap,
                min_track_len=cfg.interpolate_min_track_len,
            )
            print(
                f"🔀 Interpolation: tracks={interp_stats['tracks_interpolated']} "
                f"gaps={interp_stats['gaps_filled']} "
                f"frames_added={interp_stats['frames_added']}"
            )

        if not cfg.latency_only:
            Path(output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        print(f"✅ Finished {seq} (Total Time: {time.time() - start_time:.2f}s)")
        if relinker:
            relinker.report()
        lifecycle_merger.report()
        from .reporting import print_sequence_summary

        print_sequence_summary(
            cfg=cfg,
            seq=seq,
            seq_tile_diag=seq_tile_diag,
            profile_stages=profile_stages,
            seq_profiled_frames=seq_profiled_frames,
            top_level_stage_names=top_level_stage_names,
            seq_stage_samples=seq_stage_samples,
            overall_stage_totals=overall_stage_totals,
            overall_stage_samples=overall_stage_samples,
            breakdown_stage_names=breakdown_stage_names,
            seq_stage_totals=seq_stage_totals,
            native_reid_breakdown_names=native_reid_breakdown_names,
            seq_native_reid_samples=seq_native_reid_samples,
            gmc_breakdown_names=gmc_breakdown_names,
            seq_gmc_samples=seq_gmc_samples,
            overall_gmc_samples=overall_gmc_samples,
            segment_breakdown_names=segment_breakdown_names,
            seq_segment_samples=seq_segment_samples,
            overall_segment_samples=overall_segment_samples,
            seq_post_counts=seq_post_counts,
            overall_post_counts=overall_post_counts,
            seq_lazy_reid_frames=seq_lazy_reid_frames,
            seq_lazy_reid_candidates=seq_lazy_reid_candidates,
            overall_lazy_reid_candidates=overall_lazy_reid_candidates,
            overall_lazy_reid_frames=overall_lazy_reid_frames,
            overall_lazy_reid_crops=overall_lazy_reid_crops,
            overall_lazy_reid_self_pairs=overall_lazy_reid_self_pairs,
            overall_lazy_reid_self_pass=overall_lazy_reid_self_pass,
            overall_lazy_reid_self_sim_sum=overall_lazy_reid_self_sim_sum,
            overall_lazy_reid_arbiter_checks=overall_lazy_reid_arbiter_checks,
            overall_lazy_reid_arbiter_approve=overall_lazy_reid_arbiter_approve,
            seq_lazy_reid_crops=seq_lazy_reid_crops,
            seq_lazy_reid_self_pairs=seq_lazy_reid_self_pairs,
            seq_lazy_reid_self_pass=seq_lazy_reid_self_pass,
            seq_lazy_reid_self_sim_sum=seq_lazy_reid_self_sim_sum,
            seq_lazy_reid_arbiter_checks=seq_lazy_reid_arbiter_checks,
            seq_lazy_reid_arbiter_approve=seq_lazy_reid_arbiter_approve,
            overall_profiled_frames=overall_profiled_frames,
            stage_summary_lines=stage_summary_lines,
        )

        overall_profiled_frames += seq_profiled_frames
        if profile_stages and seq_profiled_frames > 0:
            seq_entry: dict = {"seq": seq, "frames": seq_profiled_frames, "stages": {}}
            for _sn in top_level_stage_names:
                _samp = seq_stage_samples.get(_sn, [])
                if _samp:
                    _arr = np.array(_samp, dtype=np.float64)
                    seq_entry["stages"][_sn] = {
                        "mean_ms": float(_arr.mean()),
                        "std_ms": float(_arr.std()),
                        "p95_ms": float(np.percentile(_arr, 95)),
                        "p99_ms": float(np.percentile(_arr, 99)),
                    }
            for _sn in breakdown_stage_names:
                _tot = seq_stage_totals.get(_sn, 0.0)
                if _tot > 0.0:
                    seq_entry["stages"][_sn] = {"mean_ms": _tot / seq_profiled_frames}
            _seq_post_means = {
                _sn: seq_stage_totals[_sn] / seq_profiled_frames
                for _sn in breakdown_stage_names
                if seq_stage_totals.get(_sn, 0.0) > 0.0
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
    )
