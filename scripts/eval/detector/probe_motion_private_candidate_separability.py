#!/usr/bin/env python3
"""Probe CenterTrack-lite motion signal on private suppressed candidates.

This extends ``probe_private_candidate_separability.py``. The first probe asks
whether a wider-NMS private pool contains recoverable boxes. This one asks the
next question: can previous-track geometry/motion rank those private boxes well
enough to justify a CenterTrack-lite continuation path without ReID?

The script uses ground truth only to label the offline experiment:
  * baseline NMS detections matched to GT create a proxy active-track state,
  * wider-NMS-only detections are private candidates,
  * a private candidate is positive if it overlaps an active-track GT that
    baseline NMS missed in the current frame,
  * deployment-available signals are then measured: detector score, distance to
    last/predicted track center, and IoU with last/predicted track box.

No ReID feature is used. Private candidates are evaluated as continuation-only
candidates; they are not allowed to start new IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

# Import native extension before torchvision to avoid the libjpeg conflict.
import saccade_tracking_ext  # noqa: E402, F401
import torchvision  # noqa: E402

from saccade.perception.eval.detection import detect_single_patch_640  # noqa: E402
from saccade.perception.eval.pool import AdaptiveFramePool  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
    set_postprocess_compile,
)


@dataclass
class TrackState:
    last_box: torch.Tensor
    last_frame: int
    last_score: float
    prev_box: torch.Tensor | None = None
    prev_frame: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mamba-ckpt",
        default="runs/reduction_candidates_full30/baseline_conv/best_recall.ckpt",
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="MOT17-02-SDP")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument(
        "--trt-backbone-engine",
        default="models/yolo/yolo26s_backbone_640_best.engine",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--baseline-nms-iou", type=float, default=0.50)
    parser.add_argument("--candidate-nms-iou", type=float, default=0.70)
    parser.add_argument("--score-thresholds", default="0.001,0.10,0.25")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument(
        "--max-active-gap",
        type=int,
        default=2,
        help="Frames since last baseline match for a proxy track to stay active.",
    )
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-records", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def _load_frame(path: Path, device: str) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to load image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(rgb)
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .contiguous()
        .div_(255.0)
    )


def _load_ground_truth(
    path: Path,
    *,
    max_frames: int,
) -> dict[int, list[dict[str, float | int | list[float]]]]:
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    by_frame: dict[int, list[dict[str, float | int | list[float]]]] = defaultdict(list)
    for row in rows:
        frame_id = int(row[0])
        if max_frames > 0 and frame_id > max_frames:
            continue
        mark = int(row[6])
        class_id = int(row[7])
        visibility = float(row[8]) if len(row) > 8 else 1.0
        if mark != 1 or class_id != 1 or visibility < 0.1:
            continue
        track_id = int(row[1])
        x, y, width, height = (float(value) for value in row[2:6])
        by_frame[frame_id].append(
            {
                "id": track_id,
                "bbox": [x, y, x + width, y + height],
                "width": width,
                "height": height,
                "visibility": visibility,
            }
        )
    return by_frame


def _match_gt_to_pred(
    gt_boxes: torch.Tensor,
    pred_indices: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    iou_threshold: float,
) -> dict[int, int]:
    """Greedy score-ordered prediction-to-GT matching.

    Returns ``{gt_index: raw_prediction_index}``.
    """
    if gt_boxes.numel() == 0 or pred_indices.numel() == 0:
        return {}
    pred_boxes = boxes[pred_indices]
    pred_scores = scores[pred_indices]
    ious = torchvision.ops.box_iou(pred_boxes.cpu(), gt_boxes.cpu())
    used_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    out: dict[int, int] = {}
    for local_pred_idx in pred_scores.cpu().argsort(descending=True).tolist():
        available = ious[local_pred_idx].masked_fill(used_gt, -1.0)
        best_iou, gt_idx = available.max(dim=0)
        if float(best_iou) < iou_threshold:
            continue
        gt_int = int(gt_idx)
        used_gt[gt_int] = True
        out[gt_int] = int(pred_indices[local_pred_idx])
    return out


def _center(box: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        ((box[..., 0] + box[..., 2]) * 0.5, (box[..., 1] + box[..., 3]) * 0.5), dim=-1
    )


def _predict_box(state: TrackState, frame_id: int) -> torch.Tensor:
    last = state.last_box
    last_center = _center(last)
    width = (last[2] - last[0]).clamp_min(1.0)
    height = (last[3] - last[1]).clamp_min(1.0)
    if state.prev_box is not None and state.prev_frame is not None:
        dt_prev = max(state.last_frame - state.prev_frame, 1)
        velocity = (last_center - _center(state.prev_box)) / float(dt_prev)
        center = last_center + velocity * float(max(frame_id - state.last_frame, 1))
    else:
        center = last_center
    return torch.tensor(
        [
            center[0] - width * 0.5,
            center[1] - height * 0.5,
            center[0] + width * 0.5,
            center[1] + height * 0.5,
        ],
        dtype=torch.float32,
    )


def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank = (start + csum + 1) / 2.0
    ranks = avg_rank[inv]
    sum_pos = ranks[: pos.size].sum()
    u_pos = sum_pos - pos.size * (pos.size + 1) / 2.0
    return float(u_pos / (pos.size * neg.size))


def _auc_from_records(
    records: list[dict[str, Any]], label_key: str
) -> dict[str, float]:
    labels = np.asarray([bool(rec[label_key]) for rec in records], dtype=bool)
    signals = {
        "score": "score",
        "neg_nearest_pred_dist_norm": "neg_nearest_pred_dist_norm",
        "neg_nearest_last_dist_norm": "neg_nearest_last_dist_norm",
        "best_pred_iou": "best_pred_iou",
        "best_last_iou": "best_last_iou",
        "score_times_motion": "score_times_motion",
        "score_times_pred_iou": "score_times_pred_iou",
    }
    if labels.sum() == 0 or (~labels).sum() == 0:
        return {name: float("nan") for name in signals}
    out: dict[str, float] = {}
    for name, key in signals.items():
        values = np.asarray([float(rec[key]) for rec in records], dtype=np.float64)
        out[name] = _rank_auc(values[labels], values[~labels])
    return out


def _precision_at_k(
    records: list[dict[str, Any]],
    label_key: str,
    score_key: str,
    ks: tuple[int, ...] = (20, 50, 100, 200),
) -> dict[str, float]:
    ordered = sorted(records, key=lambda rec: float(rec[score_key]), reverse=True)
    out: dict[str, float] = {}
    for k in ks:
        top = ordered[: min(k, len(ordered))]
        out[f"p_at_{k}"] = (
            float(np.mean([bool(rec[label_key]) for rec in top]))
            if top
            else float("nan")
        )
    return out


def _motion_features(
    candidate_box: torch.Tensor,
    candidate_score: float,
    active_states: dict[int, TrackState],
    frame_id: int,
) -> dict[str, Any]:
    if not active_states:
        return {
            "nearest_pred_track_id": -1,
            "nearest_pred_dist_norm": 1e6,
            "nearest_last_dist_norm": 1e6,
            "neg_nearest_pred_dist_norm": -1e6,
            "neg_nearest_last_dist_norm": -1e6,
            "best_pred_iou": 0.0,
            "best_last_iou": 0.0,
            "score_times_motion": 0.0,
            "score_times_pred_iou": 0.0,
        }

    track_ids = list(active_states)
    last_boxes = torch.stack([active_states[tid].last_box for tid in track_ids])
    pred_boxes = torch.stack(
        [_predict_box(active_states[tid], frame_id) for tid in track_ids]
    )
    cand = candidate_box.cpu().to(torch.float32)
    cand_center = _center(cand)
    cand_h = (cand[3] - cand[1]).clamp_min(1.0)

    pred_centers = _center(pred_boxes)
    last_centers = _center(last_boxes)
    pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp_min(1.0)
    last_h = (last_boxes[:, 3] - last_boxes[:, 1]).clamp_min(1.0)
    pred_norm = ((cand_h + pred_h) * 0.5).clamp_min(1.0)
    last_norm = ((cand_h + last_h) * 0.5).clamp_min(1.0)
    pred_dist = (pred_centers - cand_center).norm(dim=1) / pred_norm
    last_dist = (last_centers - cand_center).norm(dim=1) / last_norm
    nearest_pred_pos = int(pred_dist.argmin())

    pred_iou = torchvision.ops.box_iou(cand.view(1, 4), pred_boxes).squeeze(0)
    last_iou = torchvision.ops.box_iou(cand.view(1, 4), last_boxes).squeeze(0)
    nearest_pred_dist = float(pred_dist[nearest_pred_pos])
    motion_score = math.exp(-min(nearest_pred_dist, 50.0))
    best_pred_iou = float(pred_iou.max()) if pred_iou.numel() else 0.0
    best_last_iou = float(last_iou.max()) if last_iou.numel() else 0.0
    return {
        "nearest_pred_track_id": int(track_ids[nearest_pred_pos]),
        "nearest_pred_dist_norm": nearest_pred_dist,
        "nearest_last_dist_norm": float(last_dist.min()),
        "neg_nearest_pred_dist_norm": -nearest_pred_dist,
        "neg_nearest_last_dist_norm": -float(last_dist.min()),
        "best_pred_iou": best_pred_iou,
        "best_last_iou": best_last_iou,
        "score_times_motion": float(candidate_score * motion_score),
        "score_times_pred_iou": float(candidate_score * best_pred_iou),
    }


def _update_track_states(
    states: dict[int, TrackState],
    matches: dict[int, int],
    gt_items: list[dict[str, Any]],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    frame_id: int,
) -> None:
    for gt_idx, raw_idx in matches.items():
        gt_id = int(gt_items[gt_idx]["id"])
        old = states.get(gt_id)
        box = boxes[raw_idx].detach().cpu().to(torch.float32)
        states[gt_id] = TrackState(
            last_box=box,
            last_frame=frame_id,
            last_score=float(scores[raw_idx]),
            prev_box=old.last_box if old is not None else None,
            prev_frame=old.last_frame if old is not None else None,
        )


def _prune_active(
    states: dict[int, TrackState],
    frame_id: int,
    *,
    max_active_gap: int,
) -> dict[int, TrackState]:
    return {
        tid: state
        for tid, state in states.items()
        if 0 < frame_id - state.last_frame <= max_active_gap
    }


def main() -> None:
    args = build_parser().parse_args()
    thresholds = tuple(
        sorted({float(value) for value in args.score_thresholds.split(",")})
    )
    if args.candidate_nms_iou < args.baseline_nms_iou:
        raise ValueError("candidate-nms-iou must be >= baseline-nms-iou")

    set_postprocess_compile(False)
    detector = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=640,
        device=args.device,
        conf_thr=args.conf_floor,
        max_det=args.max_det,
        trt_backbone_engine=args.trt_backbone_engine,
        temporal_T_override=0,
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    detector.eval()
    detector.mamba_head.set_head_compile(False)

    label = args.label or Path(args.mamba_ckpt).parent.name
    records_by_threshold: dict[float, list[dict[str, Any]]] = {
        threshold: [] for threshold in thresholds
    }
    counts_by_threshold: dict[float, dict[str, int]] = {
        threshold: defaultdict(int) for threshold in thresholds
    }
    unique_recovered_by_threshold: dict[float, set[tuple[str, int, int]]] = {
        threshold: set() for threshold in thresholds
    }
    unique_nearest_recovered_by_threshold: dict[float, set[tuple[str, int, int]]] = {
        threshold: set() for threshold in thresholds
    }
    counts: dict[str, int] = defaultdict(int)

    for sequence in (
        item.strip() for item in args.sequences.split(",") if item.strip()
    ):
        seq_root = Path(args.data_root) / args.split / sequence
        frame_paths = sorted((seq_root / "img1").glob("*.jpg"))
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]
        if not frame_paths:
            raise RuntimeError(f"No frames found for {sequence}: {seq_root}")
        ground_truths = _load_ground_truth(
            seq_root / "gt" / "gt.txt", max_frames=args.max_frames
        )
        first_frame = _load_frame(frame_paths[0], args.device)
        orig_h, orig_w = first_frame.shape[-2:]
        pool = AdaptiveFramePool(orig_h, orig_w, device=args.device)
        states_by_threshold: dict[float, dict[int, TrackState]] = {
            threshold: {} for threshold in thresholds
        }

        detector.reset_tracker()
        with torch.inference_mode():
            for frame_index, frame_path in enumerate(frame_paths, start=1):
                frame_id = int(frame_path.stem)
                frame = (
                    first_frame
                    if frame_index == 1
                    else _load_frame(frame_path, args.device)
                )
                pool.frame_buffer.copy_(frame)
                boxes, scores, classes = detect_single_patch_640(
                    detector,
                    pool,
                    orig_h,
                    orig_w,
                    preprocess_modes=[],
                    detector_box_format="xyxy",
                )
                person = (classes.to(torch.int32) == 0) & (scores >= args.conf_floor)
                boxes = boxes[person]
                scores = scores[person]
                if boxes.numel() == 0:
                    continue

                gt_items = ground_truths.get(frame_id, [])
                gt_boxes = torch.tensor(
                    [item["bbox"] for item in gt_items],
                    dtype=torch.float32,
                ).reshape(-1, 4)
                gt_ids = [int(item["id"]) for item in gt_items]

                baseline_keep = torchvision.ops.nms(
                    boxes, scores, args.baseline_nms_iou
                )
                candidate_keep = torchvision.ops.nms(
                    boxes, scores, args.candidate_nms_iou
                )
                baseline_set = {int(idx) for idx in baseline_keep.cpu().tolist()}
                private_indices = sorted(
                    set(int(idx) for idx in candidate_keep.cpu().tolist())
                    - baseline_set
                )
                counts["frames"] += 1
                counts["raw_person"] += int(scores.numel())
                counts["baseline_kept"] += int(baseline_keep.numel())
                counts["candidate_kept"] += int(candidate_keep.numel())
                counts["private"] += len(private_indices)

                if gt_boxes.numel() > 0 and private_indices:
                    private_tensor = torch.tensor(
                        private_indices, dtype=torch.long, device=boxes.device
                    )
                    private_boxes = boxes[private_tensor]
                    gt_iou = torchvision.ops.box_iou(
                        private_boxes.cpu(), gt_boxes.cpu()
                    )
                    best_gt_iou, best_gt_idx = gt_iou.max(dim=1)
                else:
                    private_tensor = torch.tensor(
                        private_indices, dtype=torch.long, device=boxes.device
                    )
                    best_gt_iou = torch.zeros(len(private_indices), dtype=torch.float32)
                    best_gt_idx = torch.full(
                        (len(private_indices),), -1, dtype=torch.long
                    )

                matches_by_threshold: dict[float, dict[int, int]] = {}
                matched_ids_by_threshold: dict[float, set[int]] = {}
                active_by_threshold: dict[float, dict[int, TrackState]] = {}
                for threshold in thresholds:
                    selected = baseline_keep[scores[baseline_keep] >= threshold]
                    matches = _match_gt_to_pred(
                        gt_boxes,
                        selected,
                        boxes,
                        scores,
                        iou_threshold=args.match_iou,
                    )
                    matches_by_threshold[threshold] = matches
                    matched_ids = {gt_ids[gt_idx] for gt_idx in matches}
                    matched_ids_by_threshold[threshold] = matched_ids
                    active = _prune_active(
                        states_by_threshold[threshold],
                        frame_id,
                        max_active_gap=args.max_active_gap,
                    )
                    active_by_threshold[threshold] = active
                    active_missed = [
                        gt_id
                        for gt_id in gt_ids
                        if gt_id in active and gt_id not in matched_ids
                    ]
                    counts_by_threshold[threshold]["baseline_fn"] += len(gt_ids) - len(
                        matched_ids
                    )
                    counts_by_threshold[threshold]["active_baseline_fn"] += len(
                        active_missed
                    )

                for local_idx, raw_idx in enumerate(private_indices):
                    best_idx = int(best_gt_idx[local_idx])
                    best_iou = float(best_gt_iou[local_idx])
                    best_gt_id = (
                        gt_ids[best_idx]
                        if best_idx >= 0 and best_iou >= args.match_iou
                        else -1
                    )
                    score = float(scores[raw_idx])
                    for threshold in thresholds:
                        active = active_by_threshold[threshold]
                        features = _motion_features(
                            boxes[raw_idx],
                            score,
                            active,
                            frame_id,
                        )
                        active_ids = set(active)
                        matched_ids = matched_ids_by_threshold[threshold]
                        continues_active = best_gt_id in active_ids
                        recovers_active_missed = (
                            continues_active and best_gt_id not in matched_ids
                        )
                        nearest_id = int(features["nearest_pred_track_id"])
                        nearest_is_same = best_gt_id >= 0 and nearest_id == best_gt_id
                        rec = {
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "score_threshold": threshold,
                            "raw_index": int(raw_idx),
                            "score": score,
                            "best_gt_iou": best_iou,
                            "best_gt_id": int(best_gt_id),
                            "continues_active_gt": bool(continues_active),
                            "nearest_continues_active_gt": bool(
                                continues_active and nearest_is_same
                            ),
                            "recovers_active_missed_gt": bool(recovers_active_missed),
                            "nearest_recovers_active_missed_gt": bool(
                                recovers_active_missed and nearest_is_same
                            ),
                            **features,
                        }
                        records_by_threshold[threshold].append(rec)
                        if continues_active:
                            counts_by_threshold[threshold][
                                "private_continues_active_gt"
                            ] += 1
                        if continues_active and nearest_is_same:
                            counts_by_threshold[threshold][
                                "private_nearest_continues_active_gt"
                            ] += 1
                        if recovers_active_missed:
                            counts_by_threshold[threshold][
                                "private_recovers_active_missed_gt"
                            ] += 1
                            unique_recovered_by_threshold[threshold].add(
                                (sequence, frame_id, best_gt_id)
                            )
                        if recovers_active_missed and nearest_is_same:
                            counts_by_threshold[threshold][
                                "private_nearest_recovers_active_missed_gt"
                            ] += 1
                            unique_nearest_recovered_by_threshold[threshold].add(
                                (sequence, frame_id, best_gt_id)
                            )

                for threshold in thresholds:
                    _update_track_states(
                        states_by_threshold[threshold],
                        matches_by_threshold[threshold],
                        gt_items,
                        boxes,
                        scores,
                        frame_id,
                    )

                if frame_index % 100 == 0:
                    print(
                        f"[{label}] {sequence} [{frame_index}/{len(frame_paths)}] "
                        f"private={counts['private']}"
                    )

    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        recs = records_by_threshold[threshold]
        counts_t = dict(counts_by_threshold[threshold])
        rec_label = "recovers_active_missed_gt"
        nearest_rec_label = "nearest_recovers_active_missed_gt"
        by_threshold[f"{threshold:g}"] = {
            "counts": counts_t,
            "records": len(recs),
            "unique_recovered_active_missed_gt": len(
                unique_recovered_by_threshold[threshold]
            ),
            "unique_nearest_recovered_active_missed_gt": len(
                unique_nearest_recovered_by_threshold[threshold]
            ),
            "auc_recovers_active_missed": _auc_from_records(recs, rec_label),
            "auc_nearest_recovers_active_missed": _auc_from_records(
                recs, nearest_rec_label
            ),
            "precision_at_k_recovers_active_missed": {
                "score": _precision_at_k(recs, rec_label, "score"),
                "score_times_motion": _precision_at_k(
                    recs, rec_label, "score_times_motion"
                ),
                "score_times_pred_iou": _precision_at_k(
                    recs, rec_label, "score_times_pred_iou"
                ),
                "best_pred_iou": _precision_at_k(recs, rec_label, "best_pred_iou"),
            },
            "precision_at_k_nearest_recovers_active_missed": {
                "score_times_motion": _precision_at_k(
                    recs, nearest_rec_label, "score_times_motion"
                ),
                "score_times_pred_iou": _precision_at_k(
                    recs, nearest_rec_label, "score_times_pred_iou"
                ),
            },
        }

    report: dict[str, Any] = {
        "label": label,
        "checkpoint": args.mamba_ckpt,
        "conf_floor": args.conf_floor,
        "baseline_nms_iou": args.baseline_nms_iou,
        "candidate_nms_iou": args.candidate_nms_iou,
        "match_iou": args.match_iou,
        "max_active_gap": args.max_active_gap,
        "counts": dict(counts),
        "thresholds": by_threshold,
    }
    if args.save_records:
        report["records_by_threshold"] = records_by_threshold

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")

    print(f"\n[{label}] motion-private candidate separability")
    print(
        f"  baseline_nms={args.baseline_nms_iou:.2f} "
        f"candidate_nms={args.candidate_nms_iou:.2f} "
        f"private={counts.get('private', 0)}"
    )
    for threshold in thresholds:
        key = f"{threshold:g}"
        row = by_threshold[key]
        c = row["counts"]
        auc = row["auc_recovers_active_missed"]
        patk = row["precision_at_k_recovers_active_missed"]
        active_fn = c.get("active_baseline_fn", 0)
        recovered = c.get("private_recovers_active_missed_gt", 0)
        nearest_recovered = c.get("private_nearest_recovers_active_missed_gt", 0)
        print(
            f"  score>={threshold:g}: active_fn={active_fn} "
            f"private_recover_rows={recovered} nearest_rows={nearest_recovered} "
            f"AUC(score)={auc['score']:.3f} "
            f"AUC(score*motion)={auc['score_times_motion']:.3f} "
            f"P@100(score*motion)={patk['score_times_motion']['p_at_100']:.3f}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
