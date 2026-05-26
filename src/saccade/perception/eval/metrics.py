import os
import glob
from pathlib import Path
from collections import OrderedDict
from typing import Any, Iterable, Sequence, TypedDict
import numpy as np


class DetectionGroundTruth(TypedDict):
    bbox: Sequence[float]
    class_id: int


class DetectionPrediction(TypedDict):
    bbox: Sequence[float]
    score: float
    class_id: int


def _evaluate_single_sequence(
    gt_name: str,
    gt_path: str,
    ts_path: str,
) -> dict[str, float | int]:
    import motmetrics as mm

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
    ts = mm.io.loadtxt(ts_path, fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "idtp",
            "idfp",
            "idfn",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_objects",
            "num_detections",
            "num_predictions",
        ],
        name=gt_name,
    )
    row = summary.loc[gt_name]
    return {
        "idtp": int(row["idtp"]),
        "idfp": int(row["idfp"]),
        "idfn": int(row["idfn"]),
        "num_false_positives": int(row["num_false_positives"]),
        "num_misses": int(row["num_misses"]),
        "num_switches": int(row["num_switches"]),
        "num_objects": int(row["num_objects"]),
        "num_detections": int(row["num_detections"]),
        "num_predictions": int(row["num_predictions"]),
    }


def _format_overall_metrics_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    idtp = counts["idtp"]
    idfp = counts["idfp"]
    idfn = counts["idfn"]
    fp = counts["num_false_positives"]
    fn = counts["num_misses"]
    ids = counts["num_switches"]
    num_objects = counts["num_objects"]
    num_detections = counts["num_detections"]
    num_predictions = counts["num_predictions"]

    idf1_den = 2 * idtp + idfp + idfn
    mota_den = max(num_objects, 1)
    recall_den = max(num_objects, 1)
    precision_den = max(num_predictions, 1)

    idf1 = (2 * idtp / idf1_den) if idf1_den > 0 else 0.0
    mota = 1.0 - ((fn + fp + ids) / mota_den)
    recall = num_detections / recall_den
    precision = num_detections / precision_den

    return {
        "IDF1": f"{idf1 * 100:.1f}%",
        "MOTA": f"{mota * 100:.1f}%",
        "IDs": ids,
        "FP": fp,
        "FN": fn,
        "Rcll": f"{recall * 100:.1f}%",
        "Prcn": f"{precision * 100:.1f}%",
    }


def run_motmetrics_evaluation(
    data_root: str,
    split: str,
    output: str,
    sequences: str,
    detector: str | None = None,
) -> dict[str, Any] | None:
    import importlib.util

    if importlib.util.find_spec("motmetrics") is None:
        print("motmetrics not found, skipping evaluation.")
        return None

    # Force NumPy 2.0 compatibility
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

    gt_folder = os.path.join(data_root, split)
    gtfiles = glob.glob(os.path.join(gt_folder, "*/gt/gt.txt"))
    # Filter by selected sequences and detector
    if sequences:
        seq_list = sequences.split(",")
        gtfiles = [f for f in gtfiles if Path(f).parts[-3] in seq_list]
    elif detector:
        gtfiles = [f for f in gtfiles if f"-{detector}" in Path(f).parts[-3]]

    tsfiles = sorted(glob.glob(os.path.join(output, "*.txt")))
    # Filter tsfiles similarly
    if sequences:
        seq_list = sequences.split(",")
        tsfiles = [f for f in tsfiles if Path(f).stem in seq_list]

    if not gtfiles or not tsfiles:
        return None

    gt_by_name = OrderedDict((Path(f).parts[-3], f) for f in gtfiles)
    jobs: list[tuple[str, str, str]] = []
    for ts_path in tsfiles:
        ts_name = Path(ts_path).stem
        matched_gt = None
        for gt_name in gt_by_name.keys():
            if ts_name == gt_name or ts_name.startswith(gt_name + "_"):
                matched_gt = gt_name
                break
        if matched_gt:
            jobs.append((matched_gt, gt_by_name[matched_gt], ts_path))

    if not jobs:
        return None

    counts = {
        "idtp": 0,
        "idfp": 0,
        "idfn": 0,
        "num_false_positives": 0,
        "num_misses": 0,
        "num_switches": 0,
        "num_objects": 0,
        "num_detections": 0,
        "num_predictions": 0,
    }
    results = [
        _evaluate_single_sequence(gt_name, gt_path, ts_path)
        for gt_name, gt_path, ts_path in jobs
    ]

    for result in results:
        for key in counts:
            counts[key] += int(result[key])

    return _format_overall_metrics_from_counts(counts)


def _box_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _compute_average_precision(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    if recalls.size == 0 or precisions.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum(
            (mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1]
        )
    )


def compute_detection_mean_ap(
    ground_truths: dict[str, list[DetectionGroundTruth]],
    predictions: dict[str, list[DetectionPrediction]],
    *,
    iou_thresholds: Iterable[float] = (0.5,),
    class_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Compute class-aware detection AP/mAP for xyxy boxes.

    The returned `mAP` is averaged over:
    1. classes that have at least one ground-truth instance
    2. all requested IoU thresholds
    """

    thresholds = tuple(float(thr) for thr in iou_thresholds)
    if not thresholds:
        raise ValueError("iou_thresholds must not be empty")

    if class_ids is None:
        class_pool = {
            int(item["class_id"])
            for image_items in ground_truths.values()
            for item in image_items
        }
        class_pool.update(
            int(item["class_id"])
            for image_items in predictions.values()
            for item in image_items
        )
        class_id_list = sorted(class_pool)
    else:
        class_id_list = sorted({int(class_id) for class_id in class_ids})

    ap_by_threshold: dict[float, dict[int, float]] = {}
    valid_classes: list[int] = []

    for class_id in class_id_list:
        gt_for_class: dict[str, list[DetectionGroundTruth]] = {}
        gt_count = 0
        for image_id, items in ground_truths.items():
            matched_items = [
                item for item in items if int(item["class_id"]) == class_id
            ]
            if matched_items:
                gt_for_class[image_id] = matched_items
                gt_count += len(matched_items)

        if gt_count == 0:
            continue

        valid_classes.append(class_id)
        preds_for_class: list[tuple[str, float, Sequence[float]]] = []
        for image_id, pred_items in predictions.items():
            for pred_item in pred_items:
                if int(pred_item["class_id"]) == class_id:
                    preds_for_class.append(
                        (image_id, float(pred_item["score"]), pred_item["bbox"])
                    )

        preds_for_class.sort(key=lambda item: item[1], reverse=True)
        ap_by_threshold_for_class: dict[float, float] = {}

        for thr in thresholds:
            matched_flags = {
                image_id: [False] * len(items)
                for image_id, items in gt_for_class.items()
            }
            true_positives = np.zeros(len(preds_for_class), dtype=np.float64)
            false_positives = np.zeros(len(preds_for_class), dtype=np.float64)

            for pred_idx, (image_id, _score, pred_box) in enumerate(preds_for_class):
                image_gts = gt_for_class.get(image_id, [])
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(image_gts):
                    if matched_flags[image_id][gt_idx]:
                        continue
                    iou = _box_iou_xyxy(pred_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0 and best_iou >= thr:
                    matched_flags[image_id][best_gt_idx] = True
                    true_positives[pred_idx] = 1.0
                else:
                    false_positives[pred_idx] = 1.0

            cum_tp = np.cumsum(true_positives)
            cum_fp = np.cumsum(false_positives)
            recalls = cum_tp / gt_count
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            ap_by_threshold_for_class[thr] = _compute_average_precision(
                recalls, precisions
            )

        for thr, ap in ap_by_threshold_for_class.items():
            ap_by_threshold.setdefault(thr, {})[class_id] = ap

    threshold_map = {
        thr: (float(np.mean(list(class_map.values()))) if class_map else 0.0)
        for thr, class_map in ap_by_threshold.items()
    }

    overall_map = float(np.mean(list(threshold_map.values()))) if threshold_map else 0.0

    return {
        "mAP": overall_map,
        "thresholds": threshold_map,
        "per_class_ap": ap_by_threshold,
        "classes": valid_classes,
    }
