#!/usr/bin/env python3

"""Attribute near-misses to pipeline stages."""

# status: diagnostic
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))

from scripts.eval.analyze_near_miss_offsets import bbox_iou_xyxy, xywh_to_xyxy  # noqa: E402


DEFAULT_STAGES = ("raw", "post_filter", "post_nms", "post_merge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute MOT near misses to detection stages using evaluator "
            "debug-dump CSVs."
        )
    )
    parser.add_argument("--near-miss-csv", required=True)
    parser.add_argument("--stage-dump-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--good-iou", type=float, default=0.5)
    parser.add_argument("--min-best-iou", type=float, default=0.0)
    return parser


def _stage_box_iou(gt_row: pd.Series, det_row: pd.Series) -> float:
    gt_box = xywh_to_xyxy(
        float(gt_row["gt_x"]),
        float(gt_row["gt_y"]),
        float(gt_row["gt_w"]),
        float(gt_row["gt_h"]),
    )
    det_box = (
        float(det_row["x1"]),
        float(det_row["y1"]),
        float(det_row["x2"]),
        float(det_row["y2"]),
    )
    return bbox_iou_xyxy(gt_box, det_box)


def best_stage_match(
    gt_row: pd.Series,
    stage_rows: pd.DataFrame,
) -> dict[str, float | int]:
    if stage_rows.empty:
        return {
            "iou": 0.0,
            "det_idx": -1,
            "score": 0.0,
            "x1": float("nan"),
            "y1": float("nan"),
            "x2": float("nan"),
            "y2": float("nan"),
        }
    best_iou = -1.0
    best: pd.Series | None = None
    for _, det_row in stage_rows.iterrows():
        current_iou = _stage_box_iou(gt_row, det_row)
        if current_iou > best_iou:
            best_iou = current_iou
            best = det_row
    if best is None:
        raise RuntimeError("stage_rows was non-empty but no best match was selected")
    return {
        "iou": float(max(best_iou, 0.0)),
        "det_idx": int(best["det_idx"]),
        "score": float(best["score"]),
        "x1": float(best["x1"]),
        "y1": float(best["y1"]),
        "x2": float(best["x2"]),
        "y2": float(best["y2"]),
    }


def classify_stage_attribution(
    row: pd.Series, stages: tuple[str, ...], good_iou: float
) -> str:
    stage_ious = [float(row[f"{stage}_iou"]) for stage in stages]
    final_iou = float(row["final_best_iou"])
    if not stage_ious or max(stage_ious) <= 0.0:
        return "raw_no_box"
    good_stages = [stage for stage in stages if float(row[f"{stage}_iou"]) >= good_iou]
    if good_stages and final_iou < good_iou:
        last_good = good_stages[-1]
        if last_good == stages[-1]:
            return "stage_good_final_lost"
        return f"lost_after_{last_good}"
    if stage_ious[0] >= good_iou and stage_ious[-1] < good_iou:
        return "postprocess_degraded"
    if stage_ious[-1] > final_iou + 1e-6:
        return "tracker_degraded"
    return "raw_never_good"


def attribute_rows(
    near_miss_rows: pd.DataFrame,
    stage_dump_rows: pd.DataFrame,
    *,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    good_iou: float = 0.5,
    min_best_iou: float = 0.0,
) -> pd.DataFrame:
    candidates = near_miss_rows[near_miss_rows["best_iou"] >= min_best_iou].copy()
    stage_dump_rows = stage_dump_rows[stage_dump_rows["stage"].isin(stages)].copy()
    if not stage_dump_rows.empty:
        covered_frames = stage_dump_rows[["seq", "frame"]].drop_duplicates()
        candidates = candidates.merge(
            covered_frames,
            on=["seq", "frame"],
            how="inner",
        )
    grouped = {
        key: group
        for key, group in stage_dump_rows.groupby(["seq", "frame", "stage"], sort=False)
    }

    output_rows: list[dict[str, Any]] = []
    for _, gt_row in candidates.iterrows():
        base: dict[str, Any] = {
            "seq": gt_row["seq"],
            "frame": int(gt_row["frame"]),
            "gt_id": int(gt_row["gt_id"]),
            "bucket": gt_row["bucket"],
            "vis": float(gt_row["vis"]),
            "is_high_vis": bool(gt_row["is_high_vis"]),
            "gt_x": float(gt_row["gt_x"]),
            "gt_y": float(gt_row["gt_y"]),
            "gt_w": float(gt_row["gt_w"]),
            "gt_h": float(gt_row["gt_h"]),
            "final_best_iou": float(gt_row["best_iou"]),
            "final_pred_track_id": int(gt_row["pred_track_id"]),
            "final_pred_score": float(gt_row["pred_score"]),
        }
        for stage in stages:
            matches = grouped.get(
                (gt_row["seq"], int(gt_row["frame"]), stage), pd.DataFrame()
            )
            match = best_stage_match(gt_row, matches)
            base[f"{stage}_iou"] = match["iou"]
            base[f"{stage}_det_idx"] = match["det_idx"]
            base[f"{stage}_score"] = match["score"]
            base[f"{stage}_x1"] = match["x1"]
            base[f"{stage}_y1"] = match["y1"]
            base[f"{stage}_x2"] = match["x2"]
            base[f"{stage}_y2"] = match["y2"]
        series = pd.Series(base)
        base["stage_attribution"] = classify_stage_attribution(
            series,
            stages=stages,
            good_iou=good_iou,
        )
        output_rows.append(base)
    return pd.DataFrame(output_rows)


def summarize_attribution(
    rows: pd.DataFrame, stages: tuple[str, ...], good_iou: float
) -> dict[str, Any]:
    if rows.empty:
        return {
            "total": 0,
            "good_iou": good_iou,
            "attribution_counts": {},
            "stage_good_counts": {stage: 0 for stage in stages},
            "per_sequence": {},
        }
    summary: dict[str, Any] = {
        "total": int(rows.shape[0]),
        "good_iou": good_iou,
        "attribution_counts": {
            str(k): int(v) for k, v in rows["stage_attribution"].value_counts().items()
        },
        "stage_good_counts": {
            stage: int((rows[f"{stage}_iou"] >= good_iou).sum()) for stage in stages
        },
        "stage_median_iou": {
            stage: float(rows[f"{stage}_iou"].median()) for stage in stages
        },
    }
    per_sequence = {}
    for seq, group in rows.groupby("seq"):
        per_sequence[str(seq)] = {
            "total": int(group.shape[0]),
            "attribution_counts": {
                str(k): int(v)
                for k, v in group["stage_attribution"].value_counts().items()
            },
            "stage_good_counts": {
                stage: int((group[f"{stage}_iou"] >= good_iou).sum())
                for stage in stages
            },
        }
    summary["per_sequence"] = per_sequence
    return summary


def main() -> None:
    args = build_parser().parse_args()
    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())
    near_miss_rows = pd.read_csv(args.near_miss_csv)
    stage_dump_rows = pd.read_csv(args.stage_dump_csv)
    attributed = attribute_rows(
        near_miss_rows,
        stage_dump_rows,
        stages=stages,
        good_iou=args.good_iou,
        min_best_iou=args.min_best_iou,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    attributed.to_csv(output_csv, index=False)

    summary = summarize_attribution(attributed, stages=stages, good_iou=args.good_iou)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved stage attribution rows to {output_csv}")
    print(f"Saved stage attribution summary to {output_json}")


if __name__ == "__main__":
    main()
