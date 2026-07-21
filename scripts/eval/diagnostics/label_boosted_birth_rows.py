#!/usr/bin/env python3

"""Label and export birth rows boosted by GT for diagnostics."""

# status: diagnostic
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from saccade.perception.eval.external_fp_rows import (
    DetectionRowLabel,
    ExternalGroundTruthBox,
    label_detection_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label boosted birth-promotion rows against final MOT outputs as "
            "tp/fp/ignore/dropped."
        )
    )
    parser.add_argument("--boosted-csv", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--ignore-iou", type=float, default=0.5)
    return parser


def load_mot_results(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(results_dir.glob("MOT*.txt")):
        rows = pd.read_csv(
            path,
            header=None,
            names=["frame", "track_id", "x", "y", "w", "h", "score", "c1", "c2", "c3"],
            usecols=[0, 1, 2, 3, 4, 5, 6],
        )
        rows.insert(0, "seq", path.stem)
        frames.append(rows)
    if not frames:
        return pd.DataFrame(
            columns=["seq", "frame", "track_id", "x", "y", "w", "h", "score"]
        )
    return pd.concat(frames, ignore_index=True)


def load_mot17_gt(
    gt_root: Path, seqs: list[str]
) -> dict[tuple[str, int], list[ExternalGroundTruthBox]]:
    gt_by_frame: dict[tuple[str, int], list[ExternalGroundTruthBox]] = {}
    for seq in seqs:
        gt_path = gt_root / seq / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                try:
                    frame_id = int(float(parts[0]))
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    active = int(float(parts[6])) != 0
                    class_id = int(float(parts[7]))
                except ValueError:
                    continue
                if w <= 0.0 or h <= 0.0:
                    continue
                ignore = (not active) or class_id != 1
                gt_by_frame.setdefault((seq, frame_id), []).append(
                    ExternalGroundTruthBox(
                        bbox=(x, y, x + w, y + h),
                        ignore=ignore,
                    )
                )
    return gt_by_frame


def build_output_label_map(
    mot_rows: pd.DataFrame,
    gt_by_frame: dict[tuple[str, int], list[ExternalGroundTruthBox]],
    *,
    match_iou: float,
    ignore_iou: float,
) -> dict[tuple[str, int, int], DetectionRowLabel]:
    label_map: dict[tuple[str, int, int], DetectionRowLabel] = {}
    if mot_rows.empty:
        return label_map
    for (seq, frame_id), group in mot_rows.groupby(["seq", "frame"], sort=False):
        predictions: list[dict[str, float | int]] = []
        keys: list[tuple[str, int, int]] = []
        for _, row in group.iterrows():
            predictions.append(
                {
                    "bbox": (
                        float(row["x"]),
                        float(row["y"]),
                        float(row["x"] + row["w"]),
                        float(row["y"] + row["h"]),
                    ),
                    "score": float(row["score"]),
                }
            )
            keys.append((str(seq), int(frame_id), int(row["track_id"])))
        labels = label_detection_rows(
            predictions,
            gt_by_frame.get((str(seq), int(frame_id)), []),
            match_iou=match_iou,
            ignore_iou=ignore_iou,
        )
        for key, label in zip(keys, labels):
            label_map[key] = label
    return label_map


def label_boosted_rows(
    boosted_rows: pd.DataFrame,
    label_map: dict[tuple[str, int, int], DetectionRowLabel],
) -> pd.DataFrame:
    if boosted_rows.empty:
        empty = boosted_rows.copy()
        empty["final_label"] = pd.Series(dtype=str)
        empty["final_matched_iou"] = pd.Series(dtype=float)
        return empty
    output_rows: list[dict[str, Any]] = []
    for _, row in boosted_rows.iterrows():
        labeled = dict(row)
        output_track_id = (
            int(row["output_track_id"]) if pd.notna(row["output_track_id"]) else -1
        )
        if not bool(row.get("output_emitted", False)) or output_track_id < 0:
            labeled["final_label"] = "dropped"
            labeled["final_matched_iou"] = 0.0
        else:
            label = label_map.get((str(row["seq"]), int(row["frame"]), output_track_id))
            if label is None:
                labeled["final_label"] = "dropped"
                labeled["final_matched_iou"] = 0.0
            else:
                labeled["final_label"] = label.label
                labeled["final_matched_iou"] = float(label.matched_iou)
        output_rows.append(labeled)
    return pd.DataFrame(output_rows)


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"total": 0, "final_label_counts": {}, "policy_counts": {}}
    summary: dict[str, Any] = {
        "total": int(rows.shape[0]),
        "final_label_counts": {
            str(k): int(v) for k, v in rows["final_label"].value_counts().items()
        },
        "policy_counts": {},
    }
    for policy, group in rows.groupby("policy", sort=False):
        summary["policy_counts"][str(policy)] = {
            str(k): int(v) for k, v in group["final_label"].value_counts().items()
        }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    boosted_rows = pd.read_csv(args.boosted_csv)
    mot_rows = load_mot_results(Path(args.results_dir))
    seqs = sorted({str(seq) for seq in boosted_rows["seq"].dropna().unique().tolist()})
    gt_by_frame = load_mot17_gt(Path(args.gt_root), seqs)
    label_map = build_output_label_map(
        mot_rows,
        gt_by_frame,
        match_iou=args.match_iou,
        ignore_iou=args.ignore_iou,
    )
    labeled = label_boosted_rows(boosted_rows, label_map)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_csv, index=False)

    summary = summarize(labeled)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved labeled boosted rows to {output_csv}")
    print(f"Saved boosted-row summary to {output_json}")


if __name__ == "__main__":
    main()
