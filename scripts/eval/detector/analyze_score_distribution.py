#!/usr/bin/env python3
"""Mamba detector score distribution vs box height / crowd density / visibility.

For each ground-truth person box we record the score of its best-matching
detection (IoU >= match_iou, greedy by descending score) plus covariates:
  - height: original-pixel GT box height
  - visibility: MOT17 gt.txt visibility column (occlusion proxy)
  - frame_gt: number of valid GT boxes in the frame (global crowding)
  - neighbors: number of other GT whose center lies within this box (local crowding)
  - max_overlap: max IoU with any other GT box (direct occlusion proxy)

Unmatched GT get matched_score = NaN (they contribute to detection-rate but not
to the score distribution). Output: a JSON with binned summaries + Spearman
correlations, and optionally the flat per-GT records for downstream plotting.
"""
# status: stable

from __future__ import annotations

import argparse
import json
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
    ("h_128to256", 128.0, 256.0),
    ("h_ge256", 256.0, float("inf")),
)
VIS_BINS = (
    ("vis_lt30", 0.0, 0.30),
    ("vis_30to50", 0.30, 0.50),
    ("vis_50to70", 0.50, 0.70),
    ("vis_70to90", 0.70, 0.90),
    ("vis_ge90", 0.90, 1.01),
)
NEIGHBOR_BINS = (
    ("nb_0", 0.0, 1.0),
    ("nb_1", 1.0, 2.0),
    ("nb_2to3", 2.0, 4.0),
    ("nb_ge4", 4.0, float("inf")),
)
OVERLAP_BINS = (
    ("ov_0", 0.0, 0.05),
    ("ov_lt20", 0.05, 0.20),
    ("ov_20to40", 0.20, 0.40),
    ("ov_ge40", 0.40, 1.01),
)
FRAME_GT_BINS = (
    ("frame_lt15", 0.0, 15.0),
    ("frame_15to30", 15.0, 30.0),
    ("frame_30to50", 30.0, 50.0),
    ("frame_ge50", 50.0, float("inf")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mamba-ckpt", default="runs/mamba_gt_v14replica_t3_t1/best.ckpt"
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument(
        "--trt-backbone-engine",
        default="models/yolo/yolo26s_backbone_640_best.engine",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--save-records",
        action="store_true",
        help="Also dump the flat per-GT records (height/vis/density/score).",
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
    path: Path, *, max_frames: int
) -> dict[int, list[dict[str, Any]]]:
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
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


def _match_scores(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    *,
    iou_threshold: float,
) -> torch.Tensor:
    """Return a per-GT matched score (NaN where unmatched), greedy by score."""
    matched_score = torch.full((gt_boxes.shape[0],), float("nan"))
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return matched_score
    ious = torchvision.ops.box_iou(pred_boxes.cpu(), gt_boxes.cpu())
    used_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    cpu_scores = pred_scores.cpu()
    for pred_idx in cpu_scores.argsort(descending=True).tolist():
        available = ious[pred_idx].masked_fill(used_gt, -1.0)
        best_iou, gt_idx = available.max(dim=0)
        if float(best_iou) < iou_threshold:
            continue
        used_gt[gt_idx] = True
        matched_score[gt_idx] = float(cpu_scores[pred_idx])
    return matched_score


def _local_density(gt_boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-GT (neighbor_count, max_overlap_iou) with the other GT boxes."""
    n = gt_boxes.shape[0]
    if n == 0:
        return torch.zeros(0), torch.zeros(0)
    ious = torchvision.ops.box_iou(gt_boxes, gt_boxes)
    ious.fill_diagonal_(0.0)
    max_overlap = ious.max(dim=1).values if n > 1 else torch.zeros(n)
    centers = torch.stack(
        [
            (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5,
            (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5,
        ],
        dim=1,
    )
    # A neighbor is another GT whose center lies inside this box.
    inside = (
        (centers[None, :, 0] >= gt_boxes[:, 0:1])
        & (centers[None, :, 0] <= gt_boxes[:, 2:3])
        & (centers[None, :, 1] >= gt_boxes[:, 1:2])
        & (centers[None, :, 1] <= gt_boxes[:, 3:4])
    )
    inside.fill_diagonal_(False)
    neighbors = inside.sum(dim=1).to(torch.float32)
    return neighbors, max_overlap


def _summ(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"n": 0}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "frac_lt_0.05": float((arr < 0.05).mean()),
        "frac_lt_0.10": float((arr < 0.10).mean()),
        "frac_lt_0.28": float((arr < 0.28).mean()),
        "frac_ge_0.50": float((arr >= 0.50).mean()),
    }


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    xr = x.argsort().argsort().astype(np.float64)
    yr = y.argsort().argsort().astype(np.float64)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = np.sqrt((xr * xr).sum() * (yr * yr).sum())
    return float((xr * yr).sum() / denom) if denom > 0 else float("nan")


def _binned_summary(
    records: list[dict[str, float]],
    key: str,
    bins: tuple[tuple[str, float, float], ...],
) -> dict[str, dict[str, float]]:
    out: dict[str, list[float]] = {b[0]: [] for b in bins}
    det_total: dict[str, int] = {b[0]: 0 for b in bins}
    for rec in records:
        name = _bin_name(rec[key], bins)
        det_total[name] += 1
        if not np.isnan(rec["score"]):
            out[name].append(rec["score"])
    summary: dict[str, dict[str, float]] = {}
    for name in out:
        s = _summ(out[name])
        s["gt_total"] = det_total[name]
        s["detected"] = len(out[name])
        s["detect_rate"] = len(out[name]) / det_total[name] if det_total[name] else 0.0
        summary[name] = s
    return summary


def main() -> None:
    args = build_parser().parse_args()
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
    all_records: list[dict[str, float]] = []
    per_seq: dict[str, Any] = {}

    for sequence in (s.strip() for s in args.sequences.split(",") if s.strip()):
        seq_root = Path(args.data_root) / args.split / sequence
        frame_paths = sorted((seq_root / "img1").glob("*.jpg"))
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]
        ground_truths = _load_ground_truth(
            seq_root / "gt" / "gt.txt", max_frames=args.max_frames
        )
        first_frame = _load_frame(frame_paths[0], args.device)
        orig_h, orig_w = first_frame.shape[-2:]
        pool = AdaptiveFramePool(orig_h, orig_w, device=args.device)
        seq_records: list[dict[str, float]] = []

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
                frame_gt = ground_truths.get(frame_id, [])
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
                if boxes.numel() > 0:
                    keep = torchvision.ops.nms(boxes, scores, args.nms_iou)
                    boxes = boxes[keep]
                    scores = scores[keep]

                if not frame_gt:
                    continue
                gt_boxes = torch.tensor(
                    [item["bbox"] for item in frame_gt], dtype=torch.float32
                ).reshape(-1, 4)
                matched = _match_scores(
                    gt_boxes, boxes, scores, iou_threshold=args.match_iou
                )
                neighbors, max_overlap = _local_density(gt_boxes)
                frame_count = float(len(frame_gt))
                for idx, item in enumerate(frame_gt):
                    seq_records.append(
                        {
                            "score": float(matched[idx]),
                            "height": float(item["height"]),
                            "visibility": float(item["visibility"]),
                            "neighbors": float(neighbors[idx]),
                            "max_overlap": float(max_overlap[idx]),
                            "frame_gt": frame_count,
                        }
                    )
                if frame_index % 100 == 0:
                    print(f"[{label}] {sequence} [{frame_index}/{len(frame_paths)}]")

        matched_scores = [r["score"] for r in seq_records if not np.isnan(r["score"])]
        per_seq[sequence] = {
            "frames": len(frame_paths),
            "gt_total": len(seq_records),
            "detected": len(matched_scores),
            "overall_score": _summ(matched_scores),
            "by_height": _binned_summary(seq_records, "height", HEIGHT_BINS),
            "by_visibility": _binned_summary(seq_records, "visibility", VIS_BINS),
            "by_neighbors": _binned_summary(seq_records, "neighbors", NEIGHBOR_BINS),
            "by_overlap": _binned_summary(seq_records, "max_overlap", OVERLAP_BINS),
            "by_frame_gt": _binned_summary(seq_records, "frame_gt", FRAME_GT_BINS),
            "spearman_matched_only": {
                "score_vs_height": _spearman(
                    [r["height"] for r in seq_records if not np.isnan(r["score"])],
                    matched_scores,
                ),
                "score_vs_visibility": _spearman(
                    [r["visibility"] for r in seq_records if not np.isnan(r["score"])],
                    matched_scores,
                ),
                "score_vs_neighbors": _spearman(
                    [r["neighbors"] for r in seq_records if not np.isnan(r["score"])],
                    matched_scores,
                ),
                "score_vs_overlap": _spearman(
                    [r["max_overlap"] for r in seq_records if not np.isnan(r["score"])],
                    matched_scores,
                ),
            },
        }
        all_records.extend(seq_records)
        _print_seq(label, sequence, per_seq[sequence])

    matched_all = [r["score"] for r in all_records if not np.isnan(r["score"])]
    report: dict[str, Any] = {
        "label": label,
        "checkpoint": args.mamba_ckpt,
        "conf_floor": args.conf_floor,
        "match_iou": args.match_iou,
        "pooled": {
            "gt_total": len(all_records),
            "detected": len(matched_all),
            "overall_score": _summ(matched_all),
            "by_height": _binned_summary(all_records, "height", HEIGHT_BINS),
            "by_visibility": _binned_summary(all_records, "visibility", VIS_BINS),
            "by_neighbors": _binned_summary(all_records, "neighbors", NEIGHBOR_BINS),
            "by_overlap": _binned_summary(all_records, "max_overlap", OVERLAP_BINS),
            "by_frame_gt": _binned_summary(all_records, "frame_gt", FRAME_GT_BINS),
        },
        "sequences": per_seq,
    }
    if args.save_records:
        report["records"] = all_records

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {output_path}")
    _print_seq("POOLED", "all-sequences", report["pooled"])


def _print_table(title: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"  {title}")
    print(
        f"    {'bin':14s} {'GT':>6s} {'det%':>6s} "
        f"{'mean':>6s} {'med':>6s} {'p25':>6s} {'p75':>6s} "
        f"{'<.10':>6s} {'<.28':>6s} {'>=.50':>6s}"
    )
    for name, s in summary.items():
        if not s.get("gt_total"):
            continue
        if s.get("n", 0) == 0:
            print(f"    {name:14s} {s['gt_total']:6d} {'0%':>6s}  (no detections)")
            continue
        print(
            f"    {name:14s} {s['gt_total']:6d} {100 * s['detect_rate']:5.1f}% "
            f"{s['mean']:6.3f} {s['median']:6.3f} {s['p25']:6.3f} {s['p75']:6.3f} "
            f"{s['frac_lt_0.10']:6.2f} {s['frac_lt_0.28']:6.2f} {s['frac_ge_0.50']:6.2f}"
        )


def _print_seq(label: str, sequence: str, rep: dict[str, Any]) -> None:
    ov = rep["overall_score"]
    print(f"\n[{label}] {sequence}  GT={rep['gt_total']} detected={rep['detected']}")
    if ov.get("n"):
        print(
            f"  overall matched score: mean={ov['mean']:.3f} median={ov['median']:.3f} "
            f"p05={ov['p05']:.3f} p95={ov['p95']:.3f} | "
            f"frac<.10={ov['frac_lt_0.10']:.2f} <.28={ov['frac_lt_0.28']:.2f} "
            f">=.50={ov['frac_ge_0.50']:.2f}"
        )
    _print_table("by GT height (orig px)", rep["by_height"])
    _print_table("by visibility", rep["by_visibility"])
    _print_table("by neighbors-in-box", rep["by_neighbors"])
    _print_table("by max GT-GT overlap IoU", rep["by_overlap"])
    _print_table("by frame GT count", rep["by_frame_gt"])
    sp = rep.get("spearman_matched_only")
    if sp:
        print(
            "  spearman(score, .) matched-only: "
            f"height={sp['score_vs_height']:+.3f} "
            f"vis={sp['score_vs_visibility']:+.3f} "
            f"neighbors={sp['score_vs_neighbors']:+.3f} "
            f"overlap={sp['score_vs_overlap']:+.3f}"
        )


if __name__ == "__main__":
    main()
