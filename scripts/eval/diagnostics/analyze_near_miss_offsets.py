#!/usr/bin/env python3

"""Analyze spatial offsets of near-miss associations."""

# status: stable
from __future__ import annotations

import argparse
import configparser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# NumPy 2.0 compatibility for motmetrics.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm


GT_COLUMNS = ["FrameId", "Id", "X", "Y", "W", "H", "Active", "Class", "Visibility"]
TS_COLUMNS = ["FrameId", "Id", "X", "Y", "W", "H", "Score", "A", "B", "C"]
BUCKETS = ("true_miss", "near_miss", "threshold_sensitive")


@dataclass(frozen=True)
class BoxTransform:
    name: str
    mode: str
    width_scale: float = 0.0
    top_scale: float = 0.0
    bottom_scale: float = 0.0
    max_area_growth: float = 1.20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose MOT near-miss localization offsets and optionally simulate "
            "conservative box refinement transforms."
        )
    )
    parser.add_argument("--results", default="results/MOT17_rule_promotion")
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--visibility-threshold", type=float, default=0.6)
    parser.add_argument(
        "--output-csv",
        default="results/mot17_rule_promotion_near_miss_offsets.csv",
    )
    parser.add_argument(
        "--output-json",
        default="results/mot17_rule_promotion_near_miss_offsets.json",
    )
    parser.add_argument("--simulate-box-refine", action="store_true")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold used for near-miss bucket boundaries.",
    )
    return parser


def load_gt(gt_file: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_file, header=None)
    gt.columns = GT_COLUMNS
    return gt[(gt["Active"] == 1) & (gt["Class"] == 1)].copy()


def load_ts(ts_file: Path) -> pd.DataFrame:
    if not ts_file.exists() or ts_file.stat().st_size == 0:
        return pd.DataFrame(columns=TS_COLUMNS)
    ts = pd.read_csv(ts_file, header=None)
    ts.columns = TS_COLUMNS
    return ts


def load_seq_meta(seq_dir: Path) -> tuple[int, int]:
    seqinfo_path = seq_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        return 0, 0
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    if "Sequence" not in config:
        return 0, 0
    frame_w = config.getint("Sequence", "imWidth", fallback=0)
    frame_h = config.getint("Sequence", "imHeight", fallback=0)
    return frame_w, frame_h


def xywh_to_xyxy(
    x: float, y: float, w: float, h: float
) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def bbox_iou_xywh(gt_row: pd.Series, ts_row: pd.Series) -> float:
    return bbox_iou_xyxy(
        xywh_to_xyxy(
            float(gt_row["X"]),
            float(gt_row["Y"]),
            float(gt_row["W"]),
            float(gt_row["H"]),
        ),
        xywh_to_xyxy(
            float(ts_row["X"]),
            float(ts_row["Y"]),
            float(ts_row["W"]),
            float(ts_row["H"]),
        ),
    )


def bbox_iou_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def classify_bucket(best_iou: float, match_iou: float = 0.5) -> str:
    if best_iou <= 0.0:
        return "true_miss"
    if best_iou < match_iou:
        return "near_miss"
    return "threshold_sensitive"


def best_prediction_row(
    gt_row: pd.Series,
    ts_frame: pd.DataFrame,
) -> tuple[float, pd.Series | None]:
    best_iou = 0.0
    best_row: pd.Series | None = None
    for _, ts_row in ts_frame.iterrows():
        current_iou = bbox_iou_xywh(gt_row, ts_row)
        if current_iou > best_iou:
            best_iou = current_iou
            best_row = ts_row
    return best_iou, best_row


def is_edge_box(
    x: float, y: float, w: float, h: float, frame_w: int, frame_h: int
) -> bool:
    if frame_w <= 0 or frame_h <= 0:
        return False
    return x <= 0.0 or y <= 0.0 or x + w >= frame_w or y + h >= frame_h


def build_offset_row(
    *,
    seq: str,
    frame_id: int,
    gt_row: pd.Series,
    pred_row: pd.Series | None,
    best_iou: float,
    frame_w: int,
    frame_h: int,
    visibility_threshold: float,
    match_iou: float,
) -> dict[str, Any]:
    gx = float(gt_row["X"])
    gy = float(gt_row["Y"])
    gw = float(gt_row["W"])
    gh = float(gt_row["H"])
    vis = float(gt_row["Visibility"])
    row: dict[str, Any] = {
        "seq": seq,
        "frame": frame_id,
        "gt_id": int(float(gt_row["Id"])),
        "vis": vis,
        "is_high_vis": vis >= visibility_threshold,
        "bucket": classify_bucket(best_iou, match_iou=match_iou),
        "gt_x": gx,
        "gt_y": gy,
        "gt_w": gw,
        "gt_h": gh,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "best_iou": best_iou,
        "is_edge_gt": is_edge_box(gx, gy, gw, gh, frame_w, frame_h),
    }
    if pred_row is None:
        row.update(
            {
                "pred_track_id": -1,
                "pred_score": 0.0,
                "pred_x": np.nan,
                "pred_y": np.nan,
                "pred_w": np.nan,
                "pred_h": np.nan,
                "center_dx_norm": np.nan,
                "center_dy_norm": np.nan,
                "width_ratio": np.nan,
                "height_ratio": np.nan,
                "left_delta_norm": np.nan,
                "right_delta_norm": np.nan,
                "top_delta_norm": np.nan,
                "bottom_delta_norm": np.nan,
                "is_edge_pred": False,
            }
        )
        return row

    px = float(pred_row["X"])
    py = float(pred_row["Y"])
    pw = float(pred_row["W"])
    ph = float(pred_row["H"])
    gt_cx = gx + gw * 0.5
    gt_cy = gy + gh * 0.5
    pred_cx = px + pw * 0.5
    pred_cy = py + ph * 0.5
    row.update(
        {
            "pred_track_id": int(float(pred_row["Id"])),
            "pred_score": float(pred_row["Score"]),
            "pred_x": px,
            "pred_y": py,
            "pred_w": pw,
            "pred_h": ph,
            "center_dx_norm": (pred_cx - gt_cx) / max(gw, 1e-6),
            "center_dy_norm": (pred_cy - gt_cy) / max(gh, 1e-6),
            "width_ratio": pw / max(gw, 1e-6),
            "height_ratio": ph / max(gh, 1e-6),
            "left_delta_norm": (px - gx) / max(gw, 1e-6),
            "right_delta_norm": ((px + pw) - (gx + gw)) / max(gw, 1e-6),
            "top_delta_norm": (py - gy) / max(gh, 1e-6),
            "bottom_delta_norm": ((py + ph) - (gy + gh)) / max(gh, 1e-6),
            "is_edge_pred": is_edge_box(px, py, pw, ph, frame_w, frame_h),
        }
    )
    return row


def clip_xyxy(
    box: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if frame_w > 0:
        x1 = min(max(x1, 0.0), float(frame_w))
        x2 = min(max(x2, 0.0), float(frame_w))
    if frame_h > 0:
        y1 = min(max(y1, 0.0), float(frame_h))
        y2 = min(max(y2, 0.0), float(frame_h))
    return (x1, y1, max(x2, x1), max(y2, y1))


def apply_box_transform(
    pred_xywh: tuple[float, float, float, float],
    transform: BoxTransform,
    *,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float] | None:
    x, y, w, h = pred_xywh
    if w <= 0.0 or h <= 0.0:
        return None
    x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h)
    x1 -= w * transform.width_scale
    x2 += w * transform.width_scale
    y1 -= h * transform.top_scale
    y2 += h * transform.bottom_scale
    x1, y1, x2, y2 = clip_xyxy((x1, y1, x2, y2), frame_w, frame_h)
    new_w = max(0.0, x2 - x1)
    new_h = max(0.0, y2 - y1)
    area_growth = (new_w * new_h) / max(w * h, 1e-6)
    if area_growth > transform.max_area_growth:
        return None
    return (x1, y1, x2, y2)


def default_transforms() -> list[BoxTransform]:
    transforms: list[BoxTransform] = []
    for scale in (0.025, 0.05, 0.075, 0.10):
        transforms.append(
            BoxTransform(
                name=f"uniform_expand_{scale:.3f}",
                mode="uniform_expand",
                width_scale=scale,
                top_scale=scale,
                bottom_scale=scale,
            )
        )
    for scale in (0.05, 0.10, 0.15):
        transforms.append(
            BoxTransform(
                name=f"bottom_expand_{scale:.3f}",
                mode="bottom_expand",
                bottom_scale=scale,
            )
        )
    for top_scale, bottom_scale in ((0.025, 0.075), (0.05, 0.10)):
        transforms.append(
            BoxTransform(
                name=f"vertical_expand_t{top_scale:.3f}_b{bottom_scale:.3f}",
                mode="vertical_expand",
                top_scale=top_scale,
                bottom_scale=bottom_scale,
            )
        )
    for scale in (0.05, 0.10):
        transforms.append(
            BoxTransform(
                name=f"width_expand_{scale:.3f}",
                mode="width_expand",
                width_scale=scale,
            )
        )
    return transforms


def simulate_transform(
    rows: pd.DataFrame, transform: BoxTransform, match_iou: float = 0.5
) -> dict[str, Any]:
    valid = rows[rows["pred_track_id"] >= 0].copy()
    if valid.empty:
        return {
            "name": transform.name,
            "near_miss_recovered_count": 0,
            "near_miss_recovered_share": 0.0,
            "median_iou_before": 0.0,
            "median_iou_after": 0.0,
            "threshold_sensitive_iou_drop_count": 0,
            "box_area_growth_median": 0.0,
        }
    after_ious: list[float] = []
    area_growths: list[float] = []
    accepted_mask: list[bool] = []
    for _, row in valid.iterrows():
        refined = apply_box_transform(
            (
                float(row["pred_x"]),
                float(row["pred_y"]),
                float(row["pred_w"]),
                float(row["pred_h"]),
            ),
            transform,
            frame_w=int(row.get("frame_w", 0) or 0),
            frame_h=int(row.get("frame_h", 0) or 0),
        )
        if refined is None:
            after_ious.append(float(row["best_iou"]))
            area_growths.append(1.0)
            accepted_mask.append(False)
            continue
        gt_box = xywh_to_xyxy(
            float(row["gt_x"]),
            float(row["gt_y"]),
            float(row["gt_w"]),
            float(row["gt_h"]),
        )
        after_ious.append(bbox_iou_xyxy(gt_box, refined))
        new_area = max(0.0, refined[2] - refined[0]) * max(0.0, refined[3] - refined[1])
        old_area = max(float(row["pred_w"]) * float(row["pred_h"]), 1e-6)
        area_growths.append(new_area / old_area)
        accepted_mask.append(True)

    valid = valid.assign(
        _after_iou=after_ious, _area_growth=area_growths, _accepted=accepted_mask
    )
    near = valid[valid["bucket"] == "near_miss"]
    recovered = near[near["_after_iou"] >= match_iou]
    threshold_sensitive = valid[valid["bucket"] == "threshold_sensitive"]
    dropped = threshold_sensitive[threshold_sensitive["_after_iou"] < match_iou]
    return {
        "name": transform.name,
        "mode": transform.mode,
        "near_miss_recovered_count": int(recovered.shape[0]),
        "near_miss_recovered_share": float(recovered.shape[0] / max(near.shape[0], 1)),
        "median_iou_before": float(valid["best_iou"].median()),
        "median_iou_after": float(valid["_after_iou"].median()),
        "threshold_sensitive_iou_drop_count": int(dropped.shape[0]),
        "box_area_growth_median": float(valid["_area_growth"].median()),
        "accepted_count": int(valid["_accepted"].sum()),
    }


def summarize_rows(rows: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": int(rows.shape[0]),
        "bucket_counts": {
            bucket: int((rows["bucket"] == bucket).sum()) for bucket in BUCKETS
        },
        "high_vis_bucket_counts": {
            bucket: int(((rows["bucket"] == bucket) & rows["is_high_vis"]).sum())
            for bucket in BUCKETS
        },
    }
    per_seq = {}
    for seq, group in rows.groupby("seq"):
        per_seq[str(seq)] = {
            "total": int(group.shape[0]),
            "bucket_counts": {
                bucket: int((group["bucket"] == bucket).sum()) for bucket in BUCKETS
            },
            "near_miss_median_center_dx_norm": float(
                group.loc[group["bucket"] == "near_miss", "center_dx_norm"].median()
            ),
            "near_miss_median_center_dy_norm": float(
                group.loc[group["bucket"] == "near_miss", "center_dy_norm"].median()
            ),
            "near_miss_median_width_ratio": float(
                group.loc[group["bucket"] == "near_miss", "width_ratio"].median()
            ),
            "near_miss_median_height_ratio": float(
                group.loc[group["bucket"] == "near_miss", "height_ratio"].median()
            ),
        }
    summary["per_sequence"] = per_seq
    return summary


def analyze_sequence(
    *,
    seq: str,
    results_folder: Path,
    gt_root: Path,
    visibility_threshold: float,
    match_iou: float,
) -> pd.DataFrame:
    seq_dir = gt_root / seq
    gt_file = seq_dir / "gt" / "gt.txt"
    ts_file = results_folder / f"{seq}.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing GT file: {gt_file}")
    if not ts_file.exists():
        raise FileNotFoundError(f"Missing results file: {ts_file}")

    frame_w, frame_h = load_seq_meta(seq_dir)
    gt_raw = load_gt(gt_file)
    gt_indexed = gt_raw.set_index(["FrameId", "Id"])
    ts_raw = load_ts(ts_file)

    gt_mm = mm.io.loadtxt(gt_file.as_posix(), fmt="mot15-2D", min_confidence=1)
    ts_mm = mm.io.loadtxt(ts_file.as_posix(), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_mm, ts_mm, "iou", distth=match_iou)
    miss_events = acc.events[acc.events["Type"] == "MISS"]

    rows: list[dict[str, Any]] = []
    for _, event in miss_events.reset_index().iterrows():
        frame_id = int(event["FrameId"])
        gt_id = float(event["OId"])
        try:
            gt_row = gt_indexed.loc[(frame_id, gt_id)]
        except KeyError:
            continue
        gt_row = gt_row.copy()
        gt_row["FrameId"] = frame_id
        gt_row["Id"] = gt_id
        ts_frame = ts_raw[ts_raw["FrameId"] == frame_id]
        best_iou, pred_row = best_prediction_row(gt_row, ts_frame)
        rows.append(
            build_offset_row(
                seq=seq,
                frame_id=frame_id,
                gt_row=gt_row,
                pred_row=pred_row,
                best_iou=best_iou,
                frame_w=frame_w,
                frame_h=frame_h,
                visibility_threshold=visibility_threshold,
                match_iou=match_iou,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    sequences = [seq.strip() for seq in args.sequences.split(",") if seq.strip()]
    results_folder = Path(args.results)
    gt_root = Path(args.gt_root)
    frames = [
        analyze_sequence(
            seq=seq,
            results_folder=results_folder,
            gt_root=gt_root,
            visibility_threshold=args.visibility_threshold,
            match_iou=args.match_iou,
        )
        for seq in sequences
    ]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)

    summary = summarize_rows(rows)
    if args.simulate_box_refine:
        summary["simulation"] = [
            simulate_transform(rows, transform, match_iou=args.match_iou)
            for transform in default_transforms()
        ]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved near-miss rows to {output_csv}")
    print(f"Saved near-miss summary to {output_json}")


if __name__ == "__main__":
    main()
