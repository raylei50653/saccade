#!/usr/bin/env python3
# mypy: ignore-errors
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

# MUST IMPORT THIS BEFORE torchvision TO AVOID LIBJPEG CONFLICT
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402, F401
from saccade.perception.eval.detection import detect_single_patch_960  # noqa: E402
from saccade.perception.eval.metrics import compute_detection_mean_ap  # noqa: E402
from saccade.perception.eval.pool import AdaptiveFramePool  # noqa: E402
from saccade.perception.eval.preprocess import (  # noqa: E402
    apply_frame_preprocess,
    parse_preprocess,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detector-only mAP evaluation on MOT-format sequences."
    )
    parser.add_argument(
        "--model",
        default="models/yolo/yolo26s_960_batch1.engine",
        help="Detector path. Supports TRT .engine and Ultralytics .pt.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "trt", "ultralytics"],
        default="auto",
        help="Detector backend. auto infers from model suffix.",
    )
    parser.add_argument(
        "--data-root",
        default="datasets/MOT17",
        help="Dataset root containing <split>/<sequence>/img1 and gt/gt.txt.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split under data-root.",
    )
    parser.add_argument(
        "--sequences",
        default="MOT17-04-SDP",
        help="Comma-separated sequence names.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames per sequence to evaluate. 0 means all frames.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.001,
        help="Prediction score threshold before AP ranking.",
    )
    parser.add_argument(
        "--preprocess",
        default="letterbox,gamma,contrast",
        help="Comma-separated preprocess modes. Supports letterbox,gamma,contrast.",
    )
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--gamma-luma-threshold", type=float, default=0.35)
    parser.add_argument("--contrast", type=float, default=1.2)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for detector inference.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Ultralytics inference size. TRT path currently uses the repo's 960 flow.",
    )
    parser.add_argument(
        "--gt-class-id",
        type=int,
        default=1,
        help="MOT ground-truth class id to score. Default 1=pedestrian.",
    )
    parser.add_argument(
        "--gt-mark",
        type=int,
        default=1,
        help="MOT ground-truth mark/conf flag required for scoring. Default 1.",
    )
    return parser


def _resolve_backend(model_path: str, backend: str) -> str:
    if backend != "auto":
        return backend
    suffix = Path(model_path).suffix.lower()
    if suffix == ".engine":
        return "trt"
    if suffix == ".pt":
        return "ultralytics"
    raise ValueError(
        f"Cannot infer backend from model suffix {suffix!r}; pass --backend explicitly."
    )


def _load_frame_rgb_tensor(image_path: Path, device: str) -> torch.Tensor:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1).contiguous() / 255.0


def _load_sequence_ground_truth(
    gt_path: Path,
    *,
    max_frames: int,
    gt_mark: int,
    gt_class_id: int,
) -> dict[str, list[dict[str, object]]]:
    rows = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    ground_truths: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        frame_id = int(row[0])
        if max_frames > 0 and frame_id > max_frames:
            continue
        mark = int(row[6])
        class_id = int(row[7])
        if mark != gt_mark or class_id != gt_class_id:
            continue
        x, y, w, h = [float(v) for v in row[2:6]]
        image_id = f"{frame_id:06d}"
        ground_truths.setdefault(image_id, []).append(
            {
                "bbox": [x, y, x + w, y + h],
                "class_id": 0,
            }
        )
    return ground_truths


def _sequence_frame_ids(img_dir: Path, max_frames: int) -> list[str]:
    frame_ids = sorted(path.stem for path in img_dir.glob("*.jpg"))
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def _print_sequence_summary(
    sequence: str,
    frame_count: int,
    gt_boxes: int,
    pred_boxes: int,
    metrics_50: dict[str, object],
    metrics_coco: dict[str, object],
) -> None:
    print(
        f"{sequence}: "
        f"frames={frame_count} "
        f"gt_boxes={gt_boxes} "
        f"pred_boxes={pred_boxes} "
        f"mAP@0.5={float(metrics_50['mAP']):.4f} "
        f"mAP@0.5:0.95={float(metrics_coco['mAP']):.4f}"
    )


def _predict_with_trt(
    detector: TRTYoloDetector,
    pool: AdaptiveFramePool,
    frame: torch.Tensor,
    *,
    preprocess_modes: list[str],
    gamma: float,
    gamma_luma_threshold: float,
    contrast: float,
    conf_threshold: float,
) -> list[dict[str, object]]:
    h_orig, w_orig = int(frame.shape[1]), int(frame.shape[2])
    pool.frame_buffer.copy_(frame)
    apply_frame_preprocess(
        pool.frame_buffer,
        [mode for mode in preprocess_modes if mode != "letterbox"],
        gamma=gamma,
        gamma_luma_threshold=gamma_luma_threshold,
        contrast=contrast,
    )
    boxes, scores, classes, _ = detect_single_patch_960(
        detector,
        pool,
        h_orig,
        w_orig,
        preprocess_modes=preprocess_modes,
        detector_box_format="xyxy",
    )

    keep = (classes.to(torch.int32) == 0) & (scores >= conf_threshold)
    frame_predictions: list[dict[str, object]] = []
    if bool(keep.any()):
        kept_boxes = boxes[keep].detach().cpu().numpy()
        kept_scores = scores[keep].detach().cpu().numpy()
        for box, score in zip(kept_boxes, kept_scores):
            frame_predictions.append(
                {
                    "bbox": [float(v) for v in box.tolist()],
                    "score": float(score),
                    "class_id": 0,
                }
            )
    return frame_predictions


def _predict_with_ultralytics(
    model: object,
    image_path: Path,
    *,
    imgsz: int,
    conf_threshold: float,
    device: str,
) -> list[dict[str, object]]:
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf_threshold,
        classes=[0],
        device=device,
        verbose=False,
        stream=False,
    )
    result = results[0]
    frame_predictions: list[dict[str, object]] = []
    if result.boxes is None or result.boxes.xyxy is None:
        return frame_predictions
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    for box, score in zip(boxes, scores):
        frame_predictions.append(
            {
                "bbox": [float(v) for v in box.tolist()],
                "score": float(score),
                "class_id": 0,
            }
        )
    return frame_predictions


def main() -> None:
    args = build_parser().parse_args()
    preprocess_modes = parse_preprocess(args.preprocess)
    backend = _resolve_backend(args.model, args.backend)

    detector = None
    ul_model = None
    if backend == "trt":
        detector = TRTYoloDetector(engine_path=args.model, device=args.device)
    else:
        from ultralytics import YOLO

        ul_model = YOLO(args.model)

    all_ground_truths: dict[str, list[dict[str, object]]] = {}
    all_predictions: dict[str, list[dict[str, object]]] = {}
    pool: AdaptiveFramePool | None = None
    pool_shape: tuple[int, int] | None = None

    with torch.inference_mode():
        for sequence in [
            item.strip() for item in args.sequences.split(",") if item.strip()
        ]:
            seq_root = Path(args.data_root) / args.split / sequence
            img_dir = seq_root / "img1"
            gt_path = seq_root / "gt" / "gt.txt"
            if not img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            if not gt_path.exists():
                raise FileNotFoundError(f"Ground truth not found: {gt_path}")

            frame_ids = _sequence_frame_ids(img_dir, args.max_frames)
            sequence_ground_truths = _load_sequence_ground_truth(
                gt_path,
                max_frames=args.max_frames,
                gt_mark=args.gt_mark,
                gt_class_id=args.gt_class_id,
            )
            sequence_predictions: dict[str, list[dict[str, object]]] = {}

            for frame_id in frame_ids:
                image_path = img_dir / f"{frame_id}.jpg"
                if backend == "trt":
                    frame = _load_frame_rgb_tensor(image_path, args.device)
                    h_orig, w_orig = int(frame.shape[1]), int(frame.shape[2])
                    if pool is None or pool_shape != (h_orig, w_orig):
                        pool = AdaptiveFramePool(h_orig, w_orig, device=args.device)
                        pool_shape = (h_orig, w_orig)
                    sequence_predictions[frame_id] = _predict_with_trt(
                        detector,
                        pool,
                        frame,
                        preprocess_modes=preprocess_modes,
                        gamma=args.gamma,
                        gamma_luma_threshold=args.gamma_luma_threshold,
                        contrast=args.contrast,
                        conf_threshold=args.conf_threshold,
                    )
                else:
                    sequence_predictions[frame_id] = _predict_with_ultralytics(
                        ul_model,
                        image_path,
                        imgsz=args.imgsz,
                        conf_threshold=args.conf_threshold,
                        device=args.device,
                    )

            prefixed_ground_truths = {
                f"{sequence}/{frame_id}": items
                for frame_id, items in sequence_ground_truths.items()
            }
            prefixed_predictions = {
                f"{sequence}/{frame_id}": items
                for frame_id, items in sequence_predictions.items()
            }
            all_ground_truths.update(prefixed_ground_truths)
            all_predictions.update(prefixed_predictions)

            metrics_50 = compute_detection_mean_ap(
                prefixed_ground_truths,
                prefixed_predictions,
                iou_thresholds=(0.5,),
                class_ids=(0,),
            )
            metrics_coco = compute_detection_mean_ap(
                prefixed_ground_truths,
                prefixed_predictions,
                iou_thresholds=tuple(np.arange(0.5, 1.0, 0.05).tolist()),
                class_ids=(0,),
            )
            _print_sequence_summary(
                sequence,
                len(frame_ids),
                sum(len(items) for items in sequence_ground_truths.values()),
                sum(len(items) for items in sequence_predictions.values()),
                metrics_50,
                metrics_coco,
            )

    overall_50 = compute_detection_mean_ap(
        all_ground_truths,
        all_predictions,
        iou_thresholds=(0.5,),
        class_ids=(0,),
    )
    overall_coco = compute_detection_mean_ap(
        all_ground_truths,
        all_predictions,
        iou_thresholds=tuple(np.arange(0.5, 1.0, 0.05).tolist()),
        class_ids=(0,),
    )

    print("\n=== DETECTION mAP SUMMARY ===")
    print(f"backend: {backend}")
    print(f"model: {args.model}")
    print(f"split: {args.split}")
    print(f"sequences: {args.sequences}")
    print(f"max_frames: {args.max_frames}")
    print(f"preprocess: {args.preprocess}")
    print(f"conf_threshold: {args.conf_threshold}")
    print(f"gt_boxes: {sum(len(items) for items in all_ground_truths.values())}")
    print(f"pred_boxes: {sum(len(items) for items in all_predictions.values())}")
    print(f"mAP@0.5: {float(overall_50['mAP']):.4f}")
    print(f"mAP@0.5:0.95: {float(overall_coco['mAP']):.4f}")


if __name__ == "__main__":
    main()
