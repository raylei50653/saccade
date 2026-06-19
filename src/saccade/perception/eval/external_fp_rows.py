import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from saccade.perception.box_ops import box_iou


@dataclass(frozen=True)
class ExternalGroundTruthBox:
    bbox: tuple[float, float, float, float]
    ignore: bool = False


@dataclass(frozen=True)
class ExternalImageRecord:
    image_id: str
    image_path: Path
    gt_boxes: tuple[ExternalGroundTruthBox, ...]


@dataclass(frozen=True)
class DetectionRowLabel:
    label: str
    matched_iou: float


def load_crowdhuman_records(
    annotation_path: Path,
    image_dir: Path,
) -> list[ExternalImageRecord]:
    records: list[ExternalImageRecord] = []
    with annotation_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image_id = str(payload["ID"])
            gt_boxes = tuple(_parse_crowdhuman_gtboxes(payload.get("gtboxes", [])))
            records.append(
                ExternalImageRecord(
                    image_id=image_id,
                    image_path=image_dir / f"{image_id}.jpg",
                    gt_boxes=gt_boxes,
                )
            )
    return records


def _parse_crowdhuman_gtboxes(
    gtboxes: Iterable[dict[str, Any]],
) -> list[ExternalGroundTruthBox]:
    parsed: list[ExternalGroundTruthBox] = []
    for item in gtboxes:
        fbox = item.get("fbox")
        if not isinstance(fbox, Sequence) or len(fbox) != 4:
            continue
        x, y, w, h = [float(v) for v in fbox]
        if w <= 0.0 or h <= 0.0:
            continue
        tag = str(item.get("tag", "person"))
        extra = item.get("extra")
        head_attr = item.get("head_attr")
        ignore_flag = bool(isinstance(extra, dict) and int(extra.get("ignore", 0)) != 0)
        head_ignore_flag = bool(
            isinstance(head_attr, dict) and int(head_attr.get("ignore", 0)) != 0
        )
        ignore = ignore_flag or head_ignore_flag or tag != "person"
        parsed.append(
            ExternalGroundTruthBox(
                bbox=(x, y, x + w, y + h),
                ignore=ignore,
            )
        )
    return parsed


def label_detection_rows(
    predictions: Sequence[dict[str, float | int]],
    gt_boxes: Sequence[ExternalGroundTruthBox],
    *,
    match_iou: float = 0.5,
    ignore_iou: float = 0.5,
) -> list[DetectionRowLabel]:
    valid_gt = [box.bbox for box in gt_boxes if not box.ignore]
    ignore_gt = [box.bbox for box in gt_boxes if box.ignore]
    matched_gt = [False] * len(valid_gt)

    ordered_indices = sorted(
        range(len(predictions)),
        key=lambda idx: float(predictions[idx]["score"]),
        reverse=True,
    )
    labels: list[DetectionRowLabel | None] = [None] * len(predictions)

    for pred_idx in ordered_indices:
        pred_box = _coerce_xyxy(predictions[pred_idx]["bbox"])

        best_gt_idx = -1
        best_gt_iou = 0.0
        for gt_idx, gt_box in enumerate(valid_gt):
            if matched_gt[gt_idx]:
                continue
            iou = _box_iou_xyxy(pred_box, gt_box)
            if iou > best_gt_iou:
                best_gt_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_gt_iou >= match_iou:
            matched_gt[best_gt_idx] = True
            labels[pred_idx] = DetectionRowLabel(label="tp", matched_iou=best_gt_iou)
            continue

        best_ignore_iou = 0.0
        for ignore_box in ignore_gt:
            best_ignore_iou = max(best_ignore_iou, _box_iou_xyxy(pred_box, ignore_box))
        if best_ignore_iou >= ignore_iou:
            labels[pred_idx] = DetectionRowLabel(
                label="ignore",
                matched_iou=best_ignore_iou,
            )
            continue

        labels[pred_idx] = DetectionRowLabel(label="fp", matched_iou=best_gt_iou)

    return [label for label in labels if label is not None]


def count_false_negatives(
    predictions: Sequence[dict[str, float | int]],
    gt_boxes: Sequence[ExternalGroundTruthBox],
    *,
    match_iou: float = 0.5,
) -> int:
    valid_gt = [box.bbox for box in gt_boxes if not box.ignore]
    matched_gt = [False] * len(valid_gt)
    ordered_indices = sorted(
        range(len(predictions)),
        key=lambda idx: float(predictions[idx]["score"]),
        reverse=True,
    )
    for pred_idx in ordered_indices:
        pred_box = _coerce_xyxy(predictions[pred_idx]["bbox"])
        best_gt_idx = -1
        best_gt_iou = 0.0
        for gt_idx, gt_box in enumerate(valid_gt):
            if matched_gt[gt_idx]:
                continue
            iou = _box_iou_xyxy(pred_box, gt_box)
            if iou > best_gt_iou:
                best_gt_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_gt_iou >= match_iou:
            matched_gt[best_gt_idx] = True
    return sum(1 for is_matched in matched_gt if not is_matched)


def build_structural_row(
    *,
    dataset: str,
    split: str,
    image_id: str,
    image_path: str,
    image_width: int,
    image_height: int,
    prediction: dict[str, float | int],
    label: DetectionRowLabel,
    edge_touch_px: float = 2.0,
) -> dict[str, float | int | str]:
    x1, y1, x2, y2 = _coerce_xyxy(prediction["bbox"])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    aspect_ratio = height / max(width, 1e-6)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    frame_w = max(float(image_width), 1.0)
    frame_h = max(float(image_height), 1.0)
    edge_margin_px = min(x1, y1, max(frame_w - x2, 0.0), max(frame_h - y2, 0.0))
    edge_margin_norm = edge_margin_px / min(frame_w, frame_h)
    touches_edge = int(edge_margin_px <= edge_touch_px)

    return {
        "dataset": dataset,
        "split": split,
        "image_id": image_id,
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": float(prediction["score"]),
        "label": label.label,
        "matched_iou": label.matched_iou,
        "width": width,
        "height": height,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "center_x_norm": center_x / frame_w,
        "center_y_norm": center_y / frame_h,
        "edge_margin_norm": edge_margin_norm,
        "touches_edge": touches_edge,
    }


def _coerce_xyxy(
    raw_box: float | int | Sequence[float | int],
) -> tuple[float, float, float, float]:
    if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
        raise ValueError(f"Expected xyxy box with 4 values, got: {raw_box!r}")
    x1, y1, x2, y2 = [float(v) for v in raw_box]
    return (x1, y1, x2, y2)


def _box_iou_xyxy(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    return box_iou(box_a, box_b, union_mode="zero")
