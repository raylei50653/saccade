#!/usr/bin/env python3

"""Analyze near-miss events in final tracker output CSVs."""

# status: stable
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stage-attributed near misses with final MOT output from the "
            "same evaluator run."
        )
    )
    parser.add_argument("--attribution-csv", required=True)
    parser.add_argument("--mot-result", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--stage", default="post_merge")
    parser.add_argument("--good-iou", type=float, default=0.5)
    parser.add_argument("--near-iou", type=float, default=0.1)
    parser.add_argument("--same-box-iou", type=float, default=0.95)
    return parser


def load_mot_result(path: Path, seq: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(
            columns=["seq", "frame", "track_id", "x", "y", "w", "h", "score"]
        )
    rows = pd.read_csv(
        path,
        header=None,
        names=["frame", "track_id", "x", "y", "w", "h", "score", "c1", "c2", "c3"],
        usecols=[0, 1, 2, 3, 4, 5, 6],
    )
    rows.insert(0, "seq", seq)
    return rows


def _box_from_stage(row: pd.Series, stage: str) -> tuple[float, float, float, float]:
    return (
        float(row[f"{stage}_x1"]),
        float(row[f"{stage}_y1"]),
        float(row[f"{stage}_x2"]),
        float(row[f"{stage}_y2"]),
    )


def _best_iou(
    target_box: tuple[float, float, float, float],
    frame_outputs: pd.DataFrame,
) -> tuple[float, int, float]:
    best_iou = 0.0
    best_track_id = -1
    best_score = 0.0
    for _, output in frame_outputs.iterrows():
        output_box = xywh_to_xyxy(
            float(output["x"]),
            float(output["y"]),
            float(output["w"]),
            float(output["h"]),
        )
        iou = bbox_iou_xyxy(target_box, output_box)
        if iou > best_iou:
            best_iou = iou
            best_track_id = int(output["track_id"])
            best_score = float(output["score"])
    return best_iou, best_track_id, best_score


def classify_final_output(
    row: pd.Series, *, good_iou: float, near_iou: float, same_box_iou: float
) -> str:
    if float(row["stage_iou"]) < good_iou:
        return "stage_not_good"
    if float(row["final_gt_iou"]) >= good_iou:
        return "final_preserved_gt_match"
    if int(row["final_frame_outputs"]) == 0:
        return "final_frame_empty"
    if float(row["final_stage_iou"]) >= same_box_iou:
        return "final_preserved_but_metric_miss"
    if float(row["final_stage_iou"]) >= good_iou:
        return "final_similar_box_but_gt_miss"
    if float(row["final_gt_iou"]) >= near_iou:
        return "final_near_miss"
    return "final_candidate_absent"


def analyze_final_output(
    attribution_rows: pd.DataFrame,
    mot_rows: pd.DataFrame,
    *,
    stage: str = "post_merge",
    good_iou: float = 0.5,
    near_iou: float = 0.1,
    same_box_iou: float = 0.95,
) -> pd.DataFrame:
    grouped_outputs = {
        key: group for key, group in mot_rows.groupby(["seq", "frame"], sort=False)
    }
    output_rows: list[dict[str, Any]] = []
    for _, row in attribution_rows.iterrows():
        gt_box = xywh_to_xyxy(
            float(row["gt_x"]),
            float(row["gt_y"]),
            float(row["gt_w"]),
            float(row["gt_h"]),
        )
        stage_box = _box_from_stage(row, stage)
        frame_outputs = grouped_outputs.get(
            (row["seq"], int(row["frame"])),
            pd.DataFrame(),
        )
        final_gt_iou, final_gt_track_id, final_gt_score = _best_iou(
            gt_box, frame_outputs
        )
        final_stage_iou, final_stage_track_id, final_stage_score = _best_iou(
            stage_box,
            frame_outputs,
        )
        base = {
            "seq": row["seq"],
            "frame": int(row["frame"]),
            "gt_id": int(row["gt_id"]),
            "is_high_vis": bool(row["is_high_vis"]),
            "stage": stage,
            "stage_iou": float(row[f"{stage}_iou"]),
            "stage_score": float(row[f"{stage}_score"]),
            "stage_attribution": row.get("stage_attribution", ""),
            "final_gt_iou": final_gt_iou,
            "final_gt_track_id": final_gt_track_id,
            "final_gt_score": final_gt_score,
            "final_stage_iou": final_stage_iou,
            "final_stage_track_id": final_stage_track_id,
            "final_stage_score": final_stage_score,
            "final_frame_outputs": int(frame_outputs.shape[0]),
        }
        base["final_output_attribution"] = classify_final_output(
            pd.Series(base),
            good_iou=good_iou,
            near_iou=near_iou,
            same_box_iou=same_box_iou,
        )
        output_rows.append(base)
    return pd.DataFrame(output_rows)


def summarize(rows: pd.DataFrame, *, good_iou: float, stage: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "total": 0,
            "stage": stage,
            "good_iou": good_iou,
            "final_output_counts": {},
            "stage_good_total": 0,
        }
    stage_good = rows[rows["stage_iou"] >= good_iou]
    return {
        "total": int(rows.shape[0]),
        "stage": stage,
        "good_iou": good_iou,
        "stage_good_total": int(stage_good.shape[0]),
        "final_output_counts": {
            str(k): int(v)
            for k, v in rows["final_output_attribution"].value_counts().items()
        },
        "stage_good_final_output_counts": {
            str(k): int(v)
            for k, v in stage_good["final_output_attribution"].value_counts().items()
        },
        "median_final_gt_iou_for_stage_good": (
            float(stage_good["final_gt_iou"].median()) if not stage_good.empty else 0.0
        ),
        "median_final_stage_iou_for_stage_good": (
            float(stage_good["final_stage_iou"].median())
            if not stage_good.empty
            else 0.0
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    attribution_rows = pd.read_csv(args.attribution_csv)
    seqs = attribution_rows["seq"].dropna().unique().tolist()
    if len(seqs) != 1:
        raise SystemExit(
            "--attribution-csv must contain exactly one sequence when paired with one MOT result"
        )
    mot_rows = load_mot_result(Path(args.mot_result), str(seqs[0]))
    rows = analyze_final_output(
        attribution_rows,
        mot_rows,
        stage=args.stage,
        good_iou=args.good_iou,
        near_iou=args.near_iou,
        same_box_iou=args.same_box_iou,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)

    summary = summarize(rows, good_iou=args.good_iou, stage=args.stage)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved final output attribution rows to {output_csv}")
    print(f"Saved final output attribution summary to {output_json}")


if __name__ == "__main__":
    main()
