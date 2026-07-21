#!/usr/bin/env python3

"""General FN breakdown diagnostics."""

# status: stable
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# NumPy 2.0 compatibility for motmetrics.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore

import motmetrics as mm


GT_COLUMNS = ["FrameId", "Id", "X", "Y", "W", "H", "Active", "Class", "Visibility"]
TS_COLUMNS = ["FrameId", "Id", "X", "Y", "W", "H", "Score", "A", "B", "C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose MOT false negatives by sequence. Reports how many MISS events "
            "are high-visibility, whether a nearby prediction exists, and which "
            "contiguous miss runs are most actionable."
        )
    )
    parser.add_argument("--results", default="results/MOT17_eval")
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--sequences", default="MOT17-02-SDP")
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.6,
        help="Visibility threshold used for high-visibility miss reporting.",
    )
    parser.add_argument(
        "--top-runs",
        type=int,
        default=15,
        help="How many longest contiguous miss runs to print per sequence.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV path for per-run diagnostics.",
    )
    parser.add_argument(
        "--frame-csv",
        default="",
        help="Optional CSV path for per-frame MISS diagnostics.",
    )
    parser.add_argument(
        "--focus-gt-ids",
        default="",
        help="Optional comma-separated GT ids to keep in the per-frame CSV.",
    )
    return parser.parse_args()


def load_gt(gt_file: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_file, header=None)
    gt.columns = GT_COLUMNS
    return gt[(gt["Active"] == 1) & (gt["Class"] == 1)].copy()


def load_ts(ts_file: Path) -> pd.DataFrame:
    ts = pd.read_csv(ts_file, header=None)
    ts.columns = TS_COLUMNS
    return ts


def bbox_iou(gt_row: pd.Series, ts_row: pd.Series) -> float:
    gx1 = float(gt_row["X"])
    gy1 = float(gt_row["Y"])
    gx2 = gx1 + float(gt_row["W"])
    gy2 = gy1 + float(gt_row["H"])
    tx1 = float(ts_row["X"])
    ty1 = float(ts_row["Y"])
    tx2 = tx1 + float(ts_row["W"])
    ty2 = ty1 + float(ts_row["H"])

    ix1 = max(gx1, tx1)
    iy1 = max(gy1, ty1)
    ix2 = min(gx2, tx2)
    iy2 = min(gy2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (
        float(gt_row["W"]) * float(gt_row["H"])
        + float(ts_row["W"]) * float(ts_row["H"])
        - inter
    )
    return inter / union if union > 0.0 else 0.0


def best_prediction_iou(
    gt_row: pd.Series, ts_frame: pd.DataFrame
) -> tuple[float, int | None]:
    best_iou = 0.0
    best_id = None
    for _, ts_row in ts_frame.iterrows():
        current_iou = bbox_iou(gt_row, ts_row)
        if current_iou > best_iou:
            best_iou = current_iou
            best_id = int(ts_row["Id"])
    return best_iou, best_id


def contiguous_runs(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    runs = [[frames[0]]]
    for frame in frames[1:]:
        if frame == runs[-1][-1] + 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def analyze_sequence(
    seq: str,
    results_folder: Path,
    gt_root: Path,
    visibility_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_file = gt_root / seq / "gt/gt.txt"
    ts_file = results_folder / f"{seq}.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"Missing GT file: {gt_file}")
    if not ts_file.exists():
        raise FileNotFoundError(f"Missing results file: {ts_file}")

    gt_raw = load_gt(gt_file)
    gt_indexed = gt_raw.set_index(["FrameId", "Id"])
    ts_raw = load_ts(ts_file)

    gt_mm = mm.io.loadtxt(gt_file.as_posix(), fmt="mot15-2D", min_confidence=1)
    ts_mm = mm.io.loadtxt(ts_file.as_posix(), fmt="mot15-2D", min_confidence=-1.0)
    acc = mm.utils.compare_to_groundtruth(gt_mm, ts_mm, "iou", distth=0.5)
    miss_events = acc.events[acc.events["Type"] == "MISS"]

    miss_rows: list[dict[str, float | int | str | bool]] = []
    for _, event in miss_events.reset_index().iterrows():
        frame_id = int(event["FrameId"])
        gt_id = float(event["OId"])
        try:
            gt_row = gt_indexed.loc[(frame_id, gt_id)]
        except KeyError:
            continue
        ts_frame = ts_raw[ts_raw["FrameId"] == frame_id]
        best_iou, best_id = best_prediction_iou(gt_row, ts_frame)
        miss_rows.append(
            {
                "seq": seq,
                "frame": frame_id,
                "gt_id": int(gt_id),
                "vis": float(gt_row["Visibility"]),
                "is_high_vis": float(gt_row["Visibility"]) >= visibility_threshold,
                "x": float(gt_row["X"]),
                "y": float(gt_row["Y"]),
                "w": float(gt_row["W"]),
                "h": float(gt_row["H"]),
                "best_iou": best_iou,
                "best_track_id": best_id if best_id is not None else -1,
                "pred_count": int(ts_frame.shape[0]),
            }
        )

    miss_df = pd.DataFrame(miss_rows)
    run_rows: list[dict[str, float | int | str | bool]] = []
    for gt_id, group in miss_df.groupby("gt_id"):
        frames = sorted(int(v) for v in group["frame"].tolist())
        for run in contiguous_runs(frames):
            run_df = group[group["frame"].isin(run)]
            mid = run[len(run) // 2]
            mid_row = run_df[run_df["frame"] == mid].iloc[0]
            run_rows.append(
                {
                    "seq": seq,
                    "gt_id": int(gt_id),
                    "first_frame": int(run[0]),
                    "last_frame": int(run[-1]),
                    "run_len": len(run),
                    "vis_median": float(run_df["vis"].median()),
                    "vis_max": float(run_df["vis"].max()),
                    "is_high_vis_run": float(run_df["vis"].median())
                    >= visibility_threshold,
                    "best_iou_median": float(run_df["best_iou"].median()),
                    "best_iou_max": float(run_df["best_iou"].max()),
                    "zero_iou_frames": int((run_df["best_iou"] == 0.0).sum()),
                    "frames_with_iou_ge_05": int((run_df["best_iou"] >= 0.5).sum()),
                    "x": float(mid_row["x"]),
                    "y": float(mid_row["y"]),
                    "w": float(mid_row["w"]),
                    "h": float(mid_row["h"]),
                }
            )
    run_df = pd.DataFrame(run_rows)
    return miss_df, run_df


def print_sequence_summary(
    seq: str,
    miss_df: pd.DataFrame,
    run_df: pd.DataFrame,
    visibility_threshold: float,
    top_runs: int,
) -> None:
    total_misses = len(miss_df)
    high_vis = miss_df[miss_df["is_high_vis"]]
    print(f"\n## {seq}")
    print(
        f"MISS={total_misses}  "
        f"high_vis(vis>={visibility_threshold:.2f})={len(high_vis)} "
        f"({len(high_vis) / max(total_misses, 1):.1%})"
    )
    print(
        f"high_vis best_iou==0: {(high_vis['best_iou'] == 0.0).sum()}  "
        f"best_iou>=0.1: {(high_vis['best_iou'] >= 0.1).sum()}  "
        f"best_iou>=0.5: {(high_vis['best_iou'] >= 0.5).sum()}"
    )
    print(
        f"miss h median={miss_df['h'].median():.1f}  "
        f"high_vis h median={high_vis['h'].median():.1f}"
    )

    columns = [
        "gt_id",
        "run_len",
        "vis_median",
        "vis_max",
        "best_iou_median",
        "best_iou_max",
        "zero_iou_frames",
        "frames_with_iou_ge_05",
        "first_frame",
        "last_frame",
        "x",
        "y",
        "w",
        "h",
    ]
    actionable = run_df[
        (run_df["vis_max"] >= visibility_threshold) & (run_df["run_len"] >= 10)
    ].sort_values(["run_len", "vis_max"], ascending=[False, False])
    if actionable.empty:
        print("No high-visibility miss runs >= 10 frames.")
        return
    print(f"\nTop actionable miss runs (vis_max>={visibility_threshold:.2f}, len>=10):")
    print(actionable[columns].head(top_runs).to_string(index=False))


def main() -> None:
    args = parse_args()
    results_folder = Path(args.results)
    gt_root = Path(args.gt_root)
    sequences = [seq.strip() for seq in args.sequences.split(",") if seq.strip()]
    focus_gt_ids = {
        int(value.strip()) for value in args.focus_gt_ids.split(",") if value.strip()
    }

    all_runs: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    for seq in sequences:
        miss_df, run_df = analyze_sequence(
            seq=seq,
            results_folder=results_folder,
            gt_root=gt_root,
            visibility_threshold=args.visibility_threshold,
        )
        print_sequence_summary(
            seq=seq,
            miss_df=miss_df,
            run_df=run_df,
            visibility_threshold=args.visibility_threshold,
            top_runs=args.top_runs,
        )
        if not run_df.empty:
            all_runs.append(run_df)
        if not miss_df.empty:
            if focus_gt_ids:
                miss_df = miss_df[miss_df["gt_id"].isin(focus_gt_ids)].copy()
            if not miss_df.empty:
                all_frames.append(
                    miss_df.sort_values(["seq", "gt_id", "frame"]).reset_index(
                        drop=True
                    )
                )

    if args.csv and all_runs:
        output_path = Path(args.csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_runs, ignore_index=True).to_csv(output_path, index=False)
        print(f"\nSaved per-run diagnostics to {output_path}")
    if args.frame_csv and all_frames:
        output_path = Path(args.frame_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_frames, ignore_index=True).to_csv(output_path, index=False)
        print(f"Saved per-frame diagnostics to {output_path}")


if __name__ == "__main__":
    main()
