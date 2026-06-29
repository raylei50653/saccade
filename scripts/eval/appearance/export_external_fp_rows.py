#!/usr/bin/env python3
# mypy: ignore-errors
import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import torch

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from saccade.perception.detector_trt import TRTYoloDetector  # noqa: E402, F401
from saccade.perception.eval.detection import detect_single_patch_960  # noqa: E402
from saccade.perception.eval.external_fp_rows import (  # noqa: E402
    build_structural_row,
    count_false_negatives,
    label_detection_rows,
    load_crowdhuman_records,
    ExternalGroundTruthBox,
    ExternalImageRecord,
)
from saccade.perception.eval.pool import AdaptiveFramePool  # noqa: E402
from saccade.perception.eval.preprocess import (  # noqa: E402
    apply_frame_preprocess,
    parse_preprocess,
)

CSV_FIELDNAMES = [
    "dataset",
    "split",
    "image_id",
    "image_path",
    "image_width",
    "image_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "label",
    "matched_iou",
    "width",
    "height",
    "area",
    "aspect_ratio",
    "center_x_norm",
    "center_y_norm",
    "edge_margin_norm",
    "touches_edge",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export external dataset detection rows for FP/TP analysis."
    )
    parser.add_argument(
        "--dataset", choices=["crowdhuman", "mot17"], default="crowdhuman"
    )
    # CrowdHuman
    parser.add_argument(
        "--annotations",
        default="datasets/CrowdHuman/annotations/annotation_crowdhuman_val.odgt",
        help="Dataset annotation file (CrowdHuman: .odgt, MOT17: dir).",
    )
    parser.add_argument(
        "--images",
        default="datasets/CrowdHuman/images/CrowdHuman_val",
        help="Dataset image directory.",
    )
    # MOT17
    parser.add_argument(
        "--mot17-sequences",
        default="datasets/MOT17/train",
        help="MOT17 sequences root directory (contains MOT17-XX-XX/ folders).",
    )
    parser.add_argument(
        "--mot17-seq-list",
        default="",
        help="Comma-separated list of MOT17 sequence names (e.g. MOT17-02-SDP,MOT17-04-SDP). Empty=all.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split name to embed in exported rows.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination CSV path for detection-level rows.",
    )
    parser.add_argument(
        "--model",
        default="models/yolo/yolo26m_960_batch1.engine",
        help="Detector path. Supports TRT .engine and Ultralytics .pt.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "trt", "ultralytics"],
        default="auto",
        help="Detector backend. auto infers from model suffix.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for detector inference.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Ultralytics inference size. TRT path uses the repo 960 flow.",
    )
    parser.add_argument(
        "--preprocess",
        default="letterbox,gamma,contrast",
        help="Comma-separated preprocess modes for TRT inference.",
    )
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--gamma-luma-threshold", type=float, default=0.35)
    parser.add_argument("--contrast", type=float, default=1.2)
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.05,
        help="Prediction score threshold before matching/export.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for TP matching.",
    )
    parser.add_argument(
        "--ignore-iou",
        type=float,
        default=0.5,
        help="IoU threshold for ignore-region suppression.",
    )
    parser.add_argument(
        "--edge-touch-px",
        type=float,
        default=2.0,
        help="Margin in pixels for touches_edge.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit images for debugging. 0 means all images.",
    )
    parser.add_argument(
        "--pool-cache-size",
        type=int,
        default=8,
        help="Maximum number of AdaptiveFramePool instances to keep in VRAM.",
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


def _get_or_create_pool(
    *,
    pool_cache: OrderedDict[tuple[int, int], AdaptiveFramePool],
    frame_shape: tuple[int, int],
    device: str,
    max_cache_size: int,
) -> AdaptiveFramePool:
    pool = pool_cache.get(frame_shape)
    if pool is not None:
        pool_cache.move_to_end(frame_shape)
        return pool
    while len(pool_cache) >= max(max_cache_size, 1):
        pool_cache.popitem(last=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    h_orig, w_orig = frame_shape
    pool = AdaptiveFramePool(h_orig, w_orig, device=device)
    pool_cache[frame_shape] = pool
    return pool


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


def _load_detector(model_path: str, backend: str, device: str) -> object:
    if backend == "trt":
        return TRTYoloDetector(engine_path=model_path, device=device)
    if backend == "ultralytics":
        from ultralytics import YOLO

        return YOLO(model_path)
    raise ValueError(f"Unsupported backend: {backend}")


def _load_mot17_records(
    mot17_root: Path,
    seq_list: str = "",
) -> list[ExternalImageRecord]:
    """Load MOT17 sequence records from MOTChallenge format.

    Expected structure:
        mot17_root/
          MOT17-02-SDP/
            gt/gt.txt
            img1/000001.jpg, ...
            seqinfo.ini
    """
    records: list[ExternalImageRecord] = []
    seq_dirs = list(mot17_root.iterdir())
    if not seq_dirs:
        raise ValueError(f"No sequence directories found in {mot17_root}")

    # Filter by seq_list if provided
    if seq_list:
        allowed = {s.strip() for s in seq_list.split(",")}
        seq_dirs = [d for d in seq_dirs if d.name in allowed]

    for seq_dir in seq_dirs:
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"  Skipping {seq_name}: no gt.txt")
            continue

        # Parse seqinfo for image dimensions and extension
        seqinfo_path = seq_dir / "seqinfo.ini"
        im_ext = ".jpg"
        if seqinfo_path.exists():
            for line in seqinfo_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("imWidth"):
                    int(line.split("=")[-1])
                elif line.startswith("imHeight"):
                    int(line.split("=")[-1])
                elif line.startswith("imExt"):
                    im_ext = line.split("=")[-1]

        # Parse gt.txt: frame, id, x, y, w, h, visible, class, ignore
        gt_boxes: list[ExternalGroundTruthBox] = []
        with gt_path.open("r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                try:
                    fid = int(parts[0])
                    int(parts[7])
                    ignore = int(parts[8])
                    x, y, w, h = (
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                    )
                    if w <= 0 or h <= 0:
                        continue
                    # Class 1 = person, 2 = vehicle (keep all for now)
                    gt_boxes.append(
                        ExternalGroundTruthBox(
                            bbox=(x, y, x + w, y + h),
                            ignore=ignore != 0,
                        )
                    )
                except (ValueError, IndexError):
                    continue

        if not gt_boxes:
            continue

        img_dir = seq_dir / "img1"
        if not img_dir.exists():
            print(f"  Skipping {seq_name}: no img1/")
            continue

        # Create one record per frame
        frame_files = sorted(img_dir.glob(f"*{im_ext}"))
        # Build frame -> gt mapping
        frame_gt: dict[int, list[ExternalGroundTruthBox]] = {}
        for box in gt_boxes:
            # We need frame number from gt - but gt has frame index
            pass

        # Actually, in MOT gt.txt each row has a frame number.
        # We need to group gt by frame.
        frame_gt: dict[int, list[ExternalGroundTruthBox]] = {}
        with gt_path.open("r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                try:
                    fid = int(parts[0])
                    ignore = int(parts[8])
                    x, y, w, h = (
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                    )
                    if w <= 0 or h <= 0:
                        continue
                    frame_gt.setdefault(fid, []).append(
                        ExternalGroundTruthBox(
                            bbox=(x, y, x + w, y + h),
                            ignore=ignore != 0,
                        )
                    )
                except (ValueError, IndexError):
                    continue

        for img_file in frame_files:
            frame_num = int(img_file.stem.lstrip("0") or "0")
            records.append(
                ExternalImageRecord(
                    image_id=f"{seq_name}_{img_file.stem}",
                    image_path=img_file,
                    gt_boxes=tuple(frame_gt.get(frame_num, [])),
                )
            )

    print(f"  Loaded {len(records)} image records from MOT17")
    return records


def _dataset_records(dataset: str, annotation_path: Path, image_dir: Path, **kwargs):
    if dataset == "crowdhuman":
        return load_crowdhuman_records(annotation_path, image_dir)
    if dataset == "mot17":
        mot17_root = Path(kwargs.get("mot17_sequences", ""))
        seq_list = kwargs.get("mot17_seq_list", "")
        return _load_mot17_records(mot17_root, seq_list)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _write_rows_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    annotation_path = Path(args.annotations)
    image_dir = Path(args.images)
    output_csv = Path(args.output_csv)
    backend = _resolve_backend(args.model, args.backend)
    preprocess_modes = parse_preprocess(args.preprocess)

    records = _dataset_records(
        args.dataset,
        annotation_path,
        image_dir,
        mot17_sequences=args.mot17_sequences,
        mot17_seq_list=args.mot17_seq_list,
    )
    if args.max_images > 0:
        records = records[: args.max_images]

    detector = _load_detector(args.model, backend, args.device)
    pool_cache: OrderedDict[tuple[int, int], AdaptiveFramePool] = OrderedDict()

    rows: list[dict[str, float | int | str]] = []
    tp_count = 0
    fp_count = 0
    ignored_count = 0
    fn_count = 0

    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            if not record.image_path.exists():
                raise FileNotFoundError(f"Missing image: {record.image_path}")

            if backend == "trt":
                frame_tensor = _load_frame_rgb_tensor(record.image_path, args.device)
                h_orig, w_orig = int(frame_tensor.shape[1]), int(frame_tensor.shape[2])
                pool = _get_or_create_pool(
                    pool_cache=pool_cache,
                    frame_shape=(h_orig, w_orig),
                    device=args.device,
                    max_cache_size=args.pool_cache_size,
                )
                image_height, image_width = h_orig, w_orig
                predictions = _predict_with_trt(
                    detector,
                    pool,
                    frame_tensor,
                    preprocess_modes=preprocess_modes,
                    gamma=args.gamma,
                    gamma_luma_threshold=args.gamma_luma_threshold,
                    contrast=args.contrast,
                    conf_threshold=args.conf_threshold,
                )
            else:
                bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Failed to load image: {record.image_path}")
                image_height, image_width = bgr.shape[:2]
                predictions = _predict_with_ultralytics(
                    detector,
                    record.image_path,
                    imgsz=args.imgsz,
                    conf_threshold=args.conf_threshold,
                    device=args.device,
                )

            labels = label_detection_rows(
                predictions,
                record.gt_boxes,
                match_iou=args.match_iou,
                ignore_iou=args.ignore_iou,
            )
            fn_count += count_false_negatives(
                predictions,
                record.gt_boxes,
                match_iou=args.match_iou,
            )

            for prediction, label in zip(predictions, labels):
                if label.label == "ignore":
                    ignored_count += 1
                elif label.label == "tp":
                    tp_count += 1
                elif label.label == "fp":
                    fp_count += 1
                rows.append(
                    build_structural_row(
                        dataset=args.dataset,
                        split=args.split,
                        image_id=record.image_id,
                        image_path=str(record.image_path),
                        image_width=image_width,
                        image_height=image_height,
                        prediction=prediction,
                        label=label,
                        edge_touch_px=args.edge_touch_px,
                    )
                )

            if index % 100 == 0 or index == len(records):
                print(
                    f"[{index}/{len(records)}] rows={len(rows)} tp={tp_count} fp={fp_count} "
                    f"ignored={ignored_count} fn={fn_count}"
                )

    _write_rows_csv(output_csv, rows)

    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tp_count / max(tp_count + fn_count, 1)
    print(f"CSV written to {output_csv}")
    print(
        f"summary: TP={tp_count} FP={fp_count} IGNORE={ignored_count} FN={fn_count} "
        f"precision={precision:.4f} recall={recall:.4f}"
    )


if __name__ == "__main__":
    main()
