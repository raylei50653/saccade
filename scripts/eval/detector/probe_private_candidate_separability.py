#!/usr/bin/env python3
"""Probe detector-side separability for private suppressed candidates.

This is the signal-level gate for CenterTrack-lite / PTDS-style continuation:
before adding tracker-conditioned logic, ask whether boxes suppressed by the
current baseline NMS contain a usable, separable signal.

For each frame the script:
  * runs the detector once at a low confidence floor,
  * applies baseline NMS and a wider candidate NMS to the same raw person boxes,
  * labels wide-NMS-only boxes as private candidates,
  * marks whether each private candidate is a GT-overlapping TP, a duplicate of
    an already-recalled GT, or a recoverable missed-GT candidate, and
  * reports AUC / top-k precision for deployment-available signals such as
    detector score, suppressor IoU, score gap, and box size.

The key readout is not final MOT accuracy. It is whether the private candidate
pool is rankable enough to justify a tracker-private continuation path without
opening a broad FP birth channel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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

# Import the native extension before torchvision to avoid the libjpeg conflict.
import saccade_tracking_ext  # noqa: E402, F401
import torchvision  # noqa: E402

from saccade.perception.eval.detection import detect_single_patch_640  # noqa: E402
from saccade.perception.eval.pool import AdaptiveFramePool  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
    set_postprocess_compile,
)

HEIGHT_BINS = (
    ("h_lt32", 0.0, 32.0),
    ("h_32to64", 32.0, 64.0),
    ("h_64to128", 64.0, 128.0),
    ("h_ge128", 128.0, float("inf")),
)
MIN_SIDE_BINS = (
    ("min_lt4", 0.0, 4.0),
    ("min_4to8", 4.0, 8.0),
    ("min_8to16", 8.0, 16.0),
    ("min_ge16", 16.0, float("inf")),
)


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
    parser.add_argument("--candidate-nms-iou", type=float, default=0.90)
    parser.add_argument(
        "--score-thresholds",
        default="0.001,0.10,0.25",
        help="Baseline score thresholds used to define missed GT.",
    )
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--save-records",
        action="store_true",
        help="Include private candidate rows in the output JSON.",
    )
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
) -> dict[int, list[dict[str, float | list[float]]]]:
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    by_frame: dict[int, list[dict[str, float | list[float]]]] = defaultdict(list)
    for row in rows:
        frame_id = int(row[0])
        if max_frames > 0 and frame_id > max_frames:
            continue
        mark = int(row[6])
        class_id = int(row[7])
        visibility = float(row[8]) if len(row) > 8 else 1.0
        if mark != 1 or class_id != 1 or visibility < 0.1:
            continue
        x, y, width, height = (float(value) for value in row[2:6])
        by_frame[frame_id].append(
            {
                "bbox": [x, y, x + width, y + height],
                "width": width,
                "height": height,
                "visibility": visibility,
            }
        )
    return by_frame


def _bin_name(value: float, bins: tuple[tuple[str, float, float], ...]) -> str:
    for name, lower, upper in bins:
        if lower <= value < upper:
            return name
    return bins[-1][0]


def _match_gt_indices(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    *,
    iou_threshold: float,
) -> set[int]:
    """Greedy score-ordered detection-to-GT matching; returns matched GT indices."""
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return set()
    ious = torchvision.ops.box_iou(pred_boxes.cpu(), gt_boxes.cpu())
    used_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for pred_idx in pred_scores.cpu().argsort(descending=True).tolist():
        available = ious[pred_idx].masked_fill(used_gt, -1.0)
        best_iou, gt_idx = available.max(dim=0)
        if float(best_iou) < iou_threshold:
            continue
        used_gt[gt_idx] = True
    return {int(i) for i, value in enumerate(used_gt.tolist()) if value}


def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(signal(pos) > signal(neg)), ties count as 0.5."""
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
    signals = {
        "score": "score",
        "height": "height",
        "min_side": "min_side",
        "sqrt_area": "sqrt_area",
        "suppress_iou": "suppress_iou",
        "score_over_suppressor": "score_over_suppressor",
        "neg_score_gap": "neg_score_gap",
        "neg_center_dist_norm": "neg_center_dist_norm",
    }
    out: dict[str, float] = {}
    labels = np.asarray([bool(rec[label_key]) for rec in records], dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return {name: float("nan") for name in signals}
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
    if not records:
        return {f"p_at_{k}": float("nan") for k in ks}
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


def _score_bins(records: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    bins = (
        ("score_001_010", 0.001, 0.10),
        ("score_010_025", 0.10, 0.25),
        ("score_025_050", 0.25, 0.50),
        ("score_ge050", 0.50, float("inf")),
    )
    out: dict[str, Any] = {}
    for name, lo, hi in bins:
        subset = [rec for rec in records if lo <= float(rec["score"]) < hi]
        positives = sum(1 for rec in subset if bool(rec[label_key]))
        out[name] = {
            "n": len(subset),
            "positive": positives,
            "precision": positives / len(subset) if subset else float("nan"),
        }
    return out


def _compact_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in rec.items()
        if key
        in {
            "sequence",
            "frame_id",
            "score",
            "height",
            "min_side",
            "best_gt_iou",
            "best_gt_index",
            "is_potential_tp",
            "is_duplicate_tp",
            "suppress_iou",
            "suppressor_score",
            "score_gap_to_suppressor",
            "center_dist_norm",
            "height_bin",
            "min_side_bin",
        }
        or key.startswith("recovers_missed_gt_at_")
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
    private_records: list[dict[str, Any]] = []
    per_sequence: dict[str, Any] = {}
    pooled_counts: dict[str, int] = defaultdict(int)
    pooled_recovered_unique: dict[float, set[tuple[str, int, int]]] = {
        threshold: set() for threshold in thresholds
    }
    pooled_baseline_fn: dict[float, int] = defaultdict(int)

    for sequence in (
        item.strip() for item in args.sequences.split(",") if item.strip()
    ):
        seq_root = Path(args.data_root) / args.split / sequence
        frame_paths = sorted((seq_root / "img1").glob("*.jpg"))
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]
        ground_truths = _load_ground_truth(
            seq_root / "gt" / "gt.txt", max_frames=args.max_frames
        )
        if not frame_paths:
            raise RuntimeError(f"No frames found for {sequence}: {seq_root}")
        first_frame = _load_frame(frame_paths[0], args.device)
        orig_h, orig_w = first_frame.shape[-2:]
        pool = AdaptiveFramePool(orig_h, orig_w, device=args.device)
        seq_records: list[dict[str, Any]] = []
        seq_recovered_unique: dict[float, set[tuple[int, int]]] = {
            threshold: set() for threshold in thresholds
        }
        seq_baseline_fn: dict[float, int] = defaultdict(int)
        seq_counts: dict[str, int] = defaultdict(int)

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
                gt_count = gt_boxes.shape[0]
                seq_counts["frames"] += 1
                seq_counts["raw_person"] += int(scores.numel())

                baseline_keep = torchvision.ops.nms(
                    boxes, scores, args.baseline_nms_iou
                )
                candidate_keep = torchvision.ops.nms(
                    boxes, scores, args.candidate_nms_iou
                )
                baseline_set = {int(idx) for idx in baseline_keep.cpu().tolist()}
                candidate_set = {int(idx) for idx in candidate_keep.cpu().tolist()}
                private_indices = sorted(candidate_set - baseline_set)
                seq_counts["baseline_kept"] += len(baseline_set)
                seq_counts["candidate_kept"] += len(candidate_set)
                seq_counts["private"] += len(private_indices)

                baseline_matched: dict[float, set[int]] = {}
                for threshold in thresholds:
                    keep_thr = baseline_keep[scores[baseline_keep] >= threshold]
                    matched = _match_gt_indices(
                        gt_boxes,
                        boxes[keep_thr],
                        scores[keep_thr],
                        iou_threshold=args.match_iou,
                    )
                    baseline_matched[threshold] = matched
                    fn = gt_count - len(matched)
                    seq_baseline_fn[threshold] += fn
                    pooled_baseline_fn[threshold] += fn

                if not private_indices:
                    continue

                private_tensor = torch.tensor(
                    private_indices, dtype=torch.long, device=boxes.device
                )
                private_boxes = boxes[private_tensor]
                private_scores = scores[private_tensor]
                if gt_boxes.numel() > 0:
                    gt_iou = torchvision.ops.box_iou(
                        private_boxes.cpu(), gt_boxes.cpu()
                    )
                    best_gt_iou, best_gt_idx = gt_iou.max(dim=1)
                else:
                    best_gt_iou = torch.zeros(len(private_indices), dtype=torch.float32)
                    best_gt_idx = torch.full(
                        (len(private_indices),), -1, dtype=torch.long
                    )

                baseline_boxes = boxes[baseline_keep]
                baseline_scores = scores[baseline_keep]
                if baseline_boxes.numel() > 0:
                    suppress_ious = torchvision.ops.box_iou(
                        private_boxes.cpu(), baseline_boxes.cpu()
                    )
                    # The private candidate itself is not in baseline_keep, so the
                    # best overlapping baseline box is its practical suppressor.
                    sup_iou, sup_pos = suppress_ious.max(dim=1)
                    sup_scores = baseline_scores.cpu()[sup_pos]
                    sup_boxes = baseline_boxes.cpu()[sup_pos]
                else:
                    sup_iou = torch.zeros(len(private_indices), dtype=torch.float32)
                    sup_scores = torch.zeros(len(private_indices), dtype=torch.float32)
                    sup_boxes = torch.zeros(
                        len(private_indices), 4, dtype=torch.float32
                    )

                pb = private_boxes.cpu()
                ps = private_scores.cpu()
                centers = torch.stack(
                    [(pb[:, 0] + pb[:, 2]) * 0.5, (pb[:, 1] + pb[:, 3]) * 0.5],
                    dim=1,
                )
                sup_centers = torch.stack(
                    [
                        (sup_boxes[:, 0] + sup_boxes[:, 2]) * 0.5,
                        (sup_boxes[:, 1] + sup_boxes[:, 3]) * 0.5,
                    ],
                    dim=1,
                )
                heights = (pb[:, 3] - pb[:, 1]).clamp_min(1e-6)
                widths = (pb[:, 2] - pb[:, 0]).clamp_min(1e-6)
                center_dist = (centers - sup_centers).norm(dim=1)
                center_dist_norm = center_dist / heights.clamp_min(1.0)

                for local_idx, raw_idx in enumerate(private_indices):
                    height = float(heights[local_idx])
                    width = float(widths[local_idx])
                    min_side = min(
                        width * 640.0 / max(float(orig_w), 1.0),
                        height * 640.0 / max(float(orig_h), 1.0),
                    )
                    best_idx = int(best_gt_idx[local_idx])
                    best_iou = float(best_gt_iou[local_idx])
                    is_potential_tp = best_idx >= 0 and best_iou >= args.match_iou
                    rec: dict[str, Any] = {
                        "sequence": sequence,
                        "frame_id": frame_id,
                        "raw_index": int(raw_idx),
                        "score": float(ps[local_idx]),
                        "height": height,
                        "width": width,
                        "min_side": float(min_side),
                        "sqrt_area": float(math.sqrt(max(width * height, 0.0))),
                        "height_bin": _bin_name(height, HEIGHT_BINS),
                        "min_side_bin": _bin_name(min_side, MIN_SIDE_BINS),
                        "best_gt_iou": best_iou,
                        "best_gt_index": best_idx,
                        "is_potential_tp": bool(is_potential_tp),
                        "is_duplicate_tp": bool(False),
                        "suppress_iou": float(sup_iou[local_idx]),
                        "suppressor_score": float(sup_scores[local_idx]),
                        "score_gap_to_suppressor": float(
                            sup_scores[local_idx] - ps[local_idx]
                        ),
                        "score_over_suppressor": float(
                            ps[local_idx] / max(float(sup_scores[local_idx]), 1e-6)
                        ),
                        "neg_score_gap": float(ps[local_idx] - sup_scores[local_idx]),
                        "center_dist_norm": float(center_dist_norm[local_idx]),
                        "neg_center_dist_norm": float(-center_dist_norm[local_idx]),
                    }
                    for threshold in thresholds:
                        recovers = (
                            bool(is_potential_tp)
                            and float(ps[local_idx]) >= args.conf_floor
                            and best_idx not in baseline_matched[threshold]
                        )
                        duplicate_tp = (
                            bool(is_potential_tp)
                            and best_idx in baseline_matched[threshold]
                        )
                        rec[f"recovers_missed_gt_at_{threshold:g}"] = bool(recovers)
                        rec[f"duplicate_tp_at_{threshold:g}"] = bool(duplicate_tp)
                        if recovers:
                            seq_recovered_unique[threshold].add((frame_id, best_idx))
                            pooled_recovered_unique[threshold].add(
                                (sequence, frame_id, best_idx)
                            )
                    rec["is_duplicate_tp"] = any(
                        bool(rec[f"duplicate_tp_at_{threshold:g}"])
                        for threshold in thresholds
                    )
                    seq_records.append(rec)

                if frame_index % 100 == 0:
                    print(
                        f"[{label}] {sequence} [{frame_index}/{len(frame_paths)}] "
                        f"private={len(seq_records)}"
                    )

        private_records.extend(seq_records)
        for key, value in seq_counts.items():
            pooled_counts[key] += int(value)
        per_sequence[sequence] = {
            "counts": dict(seq_counts),
            "private_potential_tp": sum(
                1 for rec in seq_records if bool(rec["is_potential_tp"])
            ),
            "private_fp": sum(
                1 for rec in seq_records if not bool(rec["is_potential_tp"])
            ),
            "baseline_fn": {
                f"{threshold:g}": int(seq_baseline_fn[threshold])
                for threshold in thresholds
            },
            "unique_recovered_missed_gt": {
                f"{threshold:g}": len(seq_recovered_unique[threshold])
                for threshold in thresholds
            },
        }

    report: dict[str, Any] = {
        "label": label,
        "checkpoint": args.mamba_ckpt,
        "conf_floor": args.conf_floor,
        "baseline_nms_iou": args.baseline_nms_iou,
        "candidate_nms_iou": args.candidate_nms_iou,
        "match_iou": args.match_iou,
        "score_thresholds": thresholds,
        "counts": dict(pooled_counts),
        "private_potential_tp": sum(
            1 for rec in private_records if bool(rec["is_potential_tp"])
        ),
        "private_fp": sum(
            1 for rec in private_records if not bool(rec["is_potential_tp"])
        ),
        "baseline_fn": {
            f"{threshold:g}": int(pooled_baseline_fn[threshold])
            for threshold in thresholds
        },
        "unique_recovered_missed_gt": {
            f"{threshold:g}": len(pooled_recovered_unique[threshold])
            for threshold in thresholds
        },
        "sequences": per_sequence,
    }
    if private_records:
        tp_count = report["private_potential_tp"]
        report["private_tp_precision"] = tp_count / len(private_records)
        report["auc_private_tp_vs_fp"] = _auc_from_records(
            private_records, "is_potential_tp"
        )
        report["score_bins_private_tp"] = _score_bins(
            private_records, "is_potential_tp"
        )
        report["precision_at_k_private_tp"] = {
            "score": _precision_at_k(private_records, "is_potential_tp", "score"),
            "score_over_suppressor": _precision_at_k(
                private_records, "is_potential_tp", "score_over_suppressor"
            ),
            "suppress_iou": _precision_at_k(
                private_records, "is_potential_tp", "suppress_iou"
            ),
        }
        by_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            key = f"recovers_missed_gt_at_{threshold:g}"
            positives = sum(1 for rec in private_records if bool(rec[key]))
            by_threshold[f"{threshold:g}"] = {
                "candidate_positive": positives,
                "candidate_precision": positives / len(private_records),
                "auc_recoverable_vs_other": _auc_from_records(private_records, key),
                "precision_at_k_recoverable": {
                    "score": _precision_at_k(private_records, key, "score"),
                    "score_over_suppressor": _precision_at_k(
                        private_records, key, "score_over_suppressor"
                    ),
                    "suppress_iou": _precision_at_k(
                        private_records, key, "suppress_iou"
                    ),
                },
                "score_bins_recoverable": _score_bins(private_records, key),
            }
        report["recoverable_by_baseline_threshold"] = by_threshold
    else:
        report["private_tp_precision"] = float("nan")
        report["auc_private_tp_vs_fp"] = {}
        report["recoverable_by_baseline_threshold"] = {}

    if args.save_records:
        report["private_records"] = [_compact_record(rec) for rec in private_records]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")

    print(f"\n[{label}] private candidate separability")
    print(
        f"  raw_person={pooled_counts.get('raw_person', 0)} "
        f"baseline_kept={pooled_counts.get('baseline_kept', 0)} "
        f"candidate_kept={pooled_counts.get('candidate_kept', 0)} "
        f"private={len(private_records)}"
    )
    print(
        f"  private potential TP={report['private_potential_tp']} "
        f"FP={report['private_fp']} "
        f"precision={report['private_tp_precision']:.3f}"
    )
    if private_records:
        aucs = report["auc_private_tp_vs_fp"]
        print(
            "  AUC private TP>FP: "
            f"score={aucs['score']:.3f} "
            f"suppress_iou={aucs['suppress_iou']:.3f} "
            f"score/suppressor={aucs['score_over_suppressor']:.3f}"
        )
        for threshold in thresholds:
            unique = report["unique_recovered_missed_gt"][f"{threshold:g}"]
            fn = report["baseline_fn"][f"{threshold:g}"]
            rec = report["recoverable_by_baseline_threshold"][f"{threshold:g}"]
            auc = rec["auc_recoverable_vs_other"]["score"]
            print(
                f"  score>={threshold:g}: unique recoverable missed GT={unique}/{fn} "
                f"({unique / fn if fn else 0.0:.3f}); "
                f"candidate positives={rec['candidate_positive']} "
                f"AUC(score)={auc:.3f}"
            )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
