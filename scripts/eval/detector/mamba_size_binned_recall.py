#!/usr/bin/env python3
"""Detector-only Mamba recall stratified by ground-truth box size."""

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
from saccade.perception.temporal_yolo.mamba_head import (  # noqa: E402
    build_strip_oracle_positions,
)

ORIGINAL_HEIGHT_BINS = (
    ("h_lt32", 0.0, 32.0),
    ("h_32to64", 32.0, 64.0),
    ("h_64to128", 64.0, 128.0),
    ("h_ge128", 128.0, float("inf")),
)
RESIZED_MIN_SIDE_BINS = (
    ("min_lt4", 0.0, 4.0),
    ("min_4to8", 4.0, 8.0),
    ("min_8to16", 8.0, 16.0),
    ("min_ge16", 16.0, float("inf")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate detector-only recall by GT size for Mamba checkpoints."
    )
    parser.add_argument("--mamba-ckpt", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="MOT17-02-SDP,MOT17-05-SDP")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument(
        "--trt-backbone-engine",
        default="models/yolo/yolo26s_backbone_640_best.engine",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--score-thresholds", default="0.001,0.10,0.25")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--save-frame-records",
        action="store_true",
        help="Include per-frame GT/match counts for paired bootstrap analysis.",
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


def _bin_name(
    value: float,
    bins: tuple[tuple[str, float, float], ...],
) -> str:
    for name, lower, upper in bins:
        if lower <= value < upper:
            return name
    raise ValueError(f"No size bin for value {value}")


def _match_ground_truth(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    *,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    matched_iou = torch.zeros(gt_boxes.shape[0], dtype=torch.float32)
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return matched, matched_iou

    ious = torchvision.ops.box_iou(pred_boxes.cpu(), gt_boxes.cpu())
    used_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for pred_idx in pred_scores.cpu().argsort(descending=True).tolist():
        available = ious[pred_idx].masked_fill(used_gt, -1.0)
        best_iou, gt_idx = available.max(dim=0)
        if float(best_iou) < iou_threshold:
            continue
        used_gt[gt_idx] = True
        matched[gt_idx] = True
        matched_iou[gt_idx] = best_iou
    return matched, matched_iou


def _new_counter() -> dict[str, float]:
    return {"gt": 0.0, "matched": 0.0, "iou_sum": 0.0}


def _record_matches(
    counters: dict[str, dict[str, float]],
    ground_truth: list[dict[str, float | list[float]]],
    matched: torch.Tensor,
    matched_iou: torch.Tensor,
    *,
    orig_h: int,
    orig_w: int,
) -> None:
    for index, item in enumerate(ground_truth):
        width = float(item["width"])
        height = float(item["height"])
        resized_min_side = min(width * 640.0 / orig_w, height * 640.0 / orig_h)
        names = (
            "all",
            _bin_name(height, ORIGINAL_HEIGHT_BINS),
            _bin_name(resized_min_side, RESIZED_MIN_SIDE_BINS),
        )
        for name in names:
            counter = counters.setdefault(name, _new_counter())
            counter["gt"] += 1
            if bool(matched[index]):
                counter["matched"] += 1
                counter["iou_sum"] += float(matched_iou[index])


def _finalize(counters: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for name, counter in counters.items():
        gt_count = int(counter["gt"])
        matched = int(counter["matched"])
        finalized[name] = {
            "gt": gt_count,
            "matched": matched,
            "fn": gt_count - matched,
            "recall": matched / gt_count if gt_count else 0.0,
            "matched_mean_iou": (counter["iou_sum"] / matched if matched else 0.0),
        }
    return finalized


def _print_threshold_summary(
    label: str,
    sequence: str,
    threshold: float,
    metrics: dict[str, dict[str, float]],
) -> None:
    print(f"\n[{label}] {sequence} score>={threshold:g}")
    for name in (
        "all",
        *(item[0] for item in ORIGINAL_HEIGHT_BINS),
        *(item[0] for item in RESIZED_MIN_SIDE_BINS),
    ):
        item = metrics.get(name)
        if item is None:
            continue
        print(
            f"  {name:12s} GT={int(item['gt']):5d} "
            f"recall={100.0 * item['recall']:6.2f}% "
            f"FN={int(item['fn']):5d} "
            f"matched_IoU={item['matched_mean_iou']:.3f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    thresholds = tuple(
        sorted({float(value) for value in args.score_thresholds.split(",")})
    )
    if not thresholds or min(thresholds) < args.conf_floor:
        raise ValueError("score thresholds must be >= conf-floor")

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
    report: dict[str, Any] = {
        "label": label,
        "checkpoint": args.mamba_ckpt,
        "conf_floor": args.conf_floor,
        "score_thresholds": thresholds,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "sequences": {},
    }

    for sequence in (
        item.strip() for item in args.sequences.split(",") if item.strip()
    ):
        seq_root = Path(args.data_root) / args.split / sequence
        frame_paths = sorted((seq_root / "img1").glob("*.jpg"))
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]
        ground_truths = _load_ground_truth(
            seq_root / "gt" / "gt.txt",
            max_frames=args.max_frames,
        )
        first_frame = _load_frame(frame_paths[0], args.device)
        orig_h, orig_w = first_frame.shape[-2:]
        pool = AdaptiveFramePool(orig_h, orig_w, device=args.device)
        counters = {threshold: {} for threshold in thresholds}
        prediction_counts = {threshold: 0 for threshold in thresholds}
        frame_records: dict[float, list[dict[str, Any]]] = {
            threshold: [] for threshold in thresholds
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
                frame_gt = ground_truths.get(frame_id, [])
                if detector.use_strip_detail:
                    resized_gt = torch.tensor(
                        [item["bbox"] for item in frame_gt],
                        device=args.device,
                        dtype=torch.float32,
                    ).reshape(-1, 4)
                    if resized_gt.numel() > 0:
                        resized_gt[:, [0, 2]] *= 640.0 / orig_w
                        resized_gt[:, [1, 3]] *= 640.0 / orig_h
                    route_positions, route_mask = build_strip_oracle_positions(
                        [resized_gt],
                        p3_hw=(80, 80),
                        img_size=640,
                        budget=detector.strip_route_budget,
                        small_threshold=detector.strip_small_threshold,
                        device=torch.device(args.device),
                        generator=torch.Generator().manual_seed(20260613 + frame_id),
                    )
                    detector.set_strip_detail_positions(route_positions, route_mask)
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

                gt_boxes = torch.tensor(
                    [item["bbox"] for item in frame_gt],
                    dtype=torch.float32,
                ).reshape(-1, 4)
                for threshold in thresholds:
                    selected = scores >= threshold
                    threshold_boxes = boxes[selected]
                    threshold_scores = scores[selected]
                    prediction_counts[threshold] += int(threshold_scores.numel())
                    matched, matched_iou = _match_ground_truth(
                        gt_boxes,
                        threshold_boxes,
                        threshold_scores,
                        iou_threshold=args.match_iou,
                    )
                    _record_matches(
                        counters[threshold],
                        frame_gt,
                        matched,
                        matched_iou,
                        orig_h=orig_h,
                        orig_w=orig_w,
                    )
                    if args.save_frame_records:
                        frame_counter: dict[str, dict[str, float]] = {}
                        _record_matches(
                            frame_counter,
                            frame_gt,
                            matched,
                            matched_iou,
                            orig_h=orig_h,
                            orig_w=orig_w,
                        )
                        frame_records[threshold].append(
                            {
                                "frame_id": frame_id,
                                "predictions": int(threshold_scores.numel()),
                                "bins": _finalize(frame_counter),
                            }
                        )
                if frame_index % 100 == 0:
                    print(f"[{label}] {sequence} [{frame_index}/{len(frame_paths)}]")

        sequence_report: dict[str, Any] = {
            "frames": len(frame_paths),
            "original_hw": [orig_h, orig_w],
            "thresholds": {},
        }
        for threshold in thresholds:
            metrics = _finalize(counters[threshold])
            sequence_report["thresholds"][str(threshold)] = {
                "predictions": prediction_counts[threshold],
                "bins": metrics,
            }
            if args.save_frame_records:
                sequence_report["thresholds"][str(threshold)]["frames"] = frame_records[
                    threshold
                ]
            _print_threshold_summary(label, sequence, threshold, metrics)
        report["sequences"][sequence] = sequence_report

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
