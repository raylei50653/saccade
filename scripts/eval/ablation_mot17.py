#!/usr/bin/env python3
# mypy: ignore-errors
"""
Unified ablation runner for scripts/eval/mot17.py.

This keeps the tuning environment centered on mot17.py itself instead of
spreading category studies across many small entry points.

Categories mirror the grouped knobs in mot17.py:
- detection
- association
- geometry
- reid
- semantic
- trigger
- lifecycle

Usage:
    uv run python scripts/eval/ablation_mot17.py --category detection
    uv run python scripts/eval/ablation_mot17.py --category detection,geometry
    uv run python scripts/eval/ablation_mot17.py --category all --detector SDP
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

import numpy as np  # noqa: E402

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import motmetrics as mm  # noqa: E402


_METRICS = [
    "idf1",
    "recall",
    "precision",
    "mota",
    "num_switches",
    "num_false_positives",
    "num_misses",
]
_DISPLAY = ["IDF1", "Rcll", "Prcn", "MOTA", "IDs", "FP", "FN"]
_PCT = {"idf1", "recall", "precision", "mota"}
_CATEGORY_ORDER = [
    "detection",
    "association",
    "geometry",
    "reid",
    "semantic",
    "trigger",
    "lifecycle",
]

_CATEGORY_EXPERIMENTS: dict[str, list[tuple[str, str, list[str]]]] = {
    "detection": [
        ("conf=0.03", "conf003", ["--conf-threshold", "0.03"]),
        ("conf=0.10", "conf010", ["--conf-threshold", "0.10"]),
        ("letterbox only", "letterbox_only", ["--preprocess", "letterbox"]),
        ("letterbox+gamma", "letterbox_gamma", ["--preprocess", "letterbox,gamma"]),
        ("contrast=1.0", "contrast100", ["--contrast", "1.0"]),
        ("cross-tile merge", "cross_tile_merge", ["--cross-tile-merge"]),
        ("nms=0.60", "nms060", ["--nms-iou-threshold", "0.60"]),
    ],
    "association": [
        ("high=0.45", "high045", ["--high-thresh", "0.45"]),
        ("high=0.55", "high055", ["--high-thresh", "0.55"]),
        ("new-track=0.35", "newtrack035", ["--new-track-thresh", "0.35"]),
        ("new-track=0.50", "newtrack050", ["--new-track-thresh", "0.50"]),
        ("match=0.75", "match075", ["--match-thresh", "0.75"]),
        ("match=0.85", "match085", ["--match-thresh", "0.85"]),
        ("confirm=2", "confirm2", ["--confirm-streak", "2"]),
        ("adaptive confirm", "adaptive_confirm", ["--adaptive-confirmation"]),
        ("no gmc", "no_gmc", ["--no-gmc"]),
    ],
    "geometry": [
        ("geometry mid-scale", "mid_scale", ["--geometry-mid-scale"]),
        (
            "geometry tight",
            "geometry_tight",
            ["--geometry-min-scale", "0.92", "--geometry-max-scale", "1.12"],
        ),
        (
            "geometry loose",
            "geometry_loose",
            ["--geometry-min-scale", "0.80", "--geometry-max-scale", "1.35"],
        ),
        ("no id stability", "no_id_stability", ["--no-id-stability-filter"]),
        (
            "strict id stability",
            "strict_id_stability",
            [
                "--id-stability-min-iou",
                "0.10",
                "--id-stability-max-center-shift",
                "1.5",
            ],
        ),
        ("no person prior", "no_person_prior", ["--no-person-geometry-prior"]),
        ("min height 0.024", "min_height_024", ["--person-min-height-ratio", "0.024"]),
    ],
    "reid": [
        ("reid off", "reid_off", ["--reid-mode", "off"]),
        ("tracker mode", "tracker_mode", ["--reid-mode", "tracker"]),
        ("hybrid mode", "hybrid_mode", ["--reid-mode", "hybrid"]),
        ("transreid", "transreid", ["--reid-model", "transreid"]),
        ("osnet", "osnet", ["--reid-model", "osnet"]),
        ("cos=0.88", "cos088", ["--reid-cos-threshold", "0.88"]),
        ("cos=0.92", "cos092", ["--reid-cos-threshold", "0.92"]),
    ],
    "semantic": [
        ("thr=0.88", "thr088", ["--semantic-threshold", "0.88"]),
        ("thr=0.92", "thr092", ["--semantic-threshold", "0.92"]),
        ("ttl=90", "ttl090", ["--semantic-ttl", "90"]),
        ("gate=0.18", "gate018", ["--semantic-spatial-gate", "0.18"]),
        ("gate=0.22", "gate022", ["--semantic-spatial-gate", "0.22"]),
        ("bank inject off", "no_bank_inject", ["--no-semantic-bank-inject"]),
        ("margin=0.05", "margin005", ["--semantic-reciprocal-margin", "0.05"]),
    ],
    "trigger": [
        (
            "fixed interval",
            "fixed_interval",
            ["--no-need-reid", "--reid-interval", "16"],
        ),
        ("count jump", "count_jump", ["--reid-trigger-mode", "count_jump"]),
        ("event persist", "event_persist", ["--reid-trigger-mode", "event_persist"]),
        ("event memory", "event_memory", ["--reid-trigger-mode", "event_memory"]),
        ("score ema", "score_ema", ["--reid-trigger-mode", "score_ema"]),
        (
            "score ema strict",
            "score_ema_strict",
            [
                "--reid-trigger-mode",
                "score_ema",
                "--reid-score-threshold",
                "2.5",
                "--reid-cooldown-frames",
                "8",
            ],
        ),
        (
            "score ema hysteresis",
            "score_ema_hyst",
            [
                "--reid-trigger-mode",
                "score_ema",
                "--reid-score-threshold",
                "2.0",
                "--reid-score-threshold-low",
                "1.5",
            ],
        ),
    ],
    "lifecycle": [
        ("nsa kalman", "nsa_kalman", ["--nsa-kalman"]),
        ("min len 2", "min_len_2", ["--min-tracklet-len", "2"]),
        ("min score 0.10", "min_score_010", ["--min-tracklet-score", "0.10"]),
        ("lifecycle merge", "lifecycle_merge", ["--lifecycle-merge"]),
        (
            "post merge ttl60",
            "postmerge_t60",
            ["--post-lifecycle-merge", "--post-lifecycle-ttl", "60"],
        ),
        (
            "post merge app gate",
            "postmerge_app",
            [
                "--post-lifecycle-merge",
                "--post-lifecycle-appearance-gate",
                "--post-lifecycle-appearance-threshold",
                "0.90",
            ],
        ),
        (
            "qf2 + post merge",
            "qf2_postmerge",
            [
                "--min-tracklet-len",
                "2",
                "--post-lifecycle-merge",
                "--post-lifecycle-ttl",
                "60",
            ],
        ),
    ],
}


def is_mot_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("MOT") and name.endswith(".txt")


def evaluate_dir(results_dir: str, gt_root: str, detector: str | None) -> dict | None:
    import glob
    import json

    cache_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    files = sorted(
        f for f in glob.glob(os.path.join(results_dir, "*.txt")) if is_mot_file(f)
    )
    if detector:
        files = [f for f in files if f"-{detector}" in Path(f).stem]
    if not files:
        return None

    gt_files = glob.glob(os.path.join(gt_root, "*/gt/gt.txt"))
    if detector:
        gt_files = [f for f in gt_files if f"-{detector}" in Path(f).parts[-3]]
    gt = {
        Path(f).parts[-3]: mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)
        for f in gt_files
    }

    accs, names = [], []
    for f in files:
        name = os.path.splitext(Path(f).name)[0]
        if name not in gt:
            continue
        ts = mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt[name], ts, "iou", distth=0.5))
        names.append(name)
    if not accs:
        return None

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names, metrics=_METRICS, generate_overall=True
    )
    results = {m: summary.loc["OVERALL", m] for m in _METRICS}

    # Save to cache
    try:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    return results


def fmt(val, metric: str) -> str:
    if metric in _PCT:
        return f"{val * 100:.1f}%"
    return f"{int(val)}"


def delta_str(base, val, metric: str) -> str:
    if metric in _PCT:
        d = (val - base) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}pp"
    d = int(val) - int(base)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d}"


def run_eval(
    label: str, output_dir: str, extra_args: list[str], base_args: list[str], dry: bool
) -> bool:
    cmd = (
        ["uv", "run", "python", "scripts/eval/mot17.py", "--output", output_dir]
        + base_args
        + extra_args
    )
    print(f"\n{'=' * 68}")
    print(f"  Running: {label}")
    print(f"  Cmd:     {' '.join(cmd)}")
    print(f"{'=' * 68}")
    if dry:
        return True
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def parse_categories(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(_CATEGORY_ORDER)
    unknown = [name for name in names if name not in _CATEGORY_EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unsupported categories: {', '.join(unknown)}")
    return [name for name in _CATEGORY_ORDER if name in names]


def print_table(title: str, results: list[tuple[str, dict | None]]) -> None:
    print(f"\n{'=' * 88}")
    print(f"  {title}")
    print(f"{'=' * 88}")

    base_vals = results[0][1]
    if base_vals is None:
        print("Baseline results not found.")
        return

    lw = 28
    mw = 8
    header = f"{'Config':<{lw}}"
    for d in _DISPLAY:
        header += f" {d:>{mw}}"
    header += "  IDF1 D  IDs D  MOTA D"
    print(header)
    print("-" * len(header))

    for label, r in results:
        if r is None:
            print(f"{label:<{lw}}  (no results)")
            continue
        row = f"{label:<{lw}}"
        for metric in _METRICS:
            row += f" {fmt(r[metric], metric):>{mw}}"
        row += (
            f"  {delta_str(base_vals['idf1'], r['idf1'], 'idf1'):>7}"
            f"  {delta_str(base_vals['num_switches'], r['num_switches'], 'num_switches'):>5}"
            f"  {delta_str(base_vals['mota'], r['mota'], 'mota'):>7}"
        )
        print(row)
    print(f"{'=' * 88}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", default="all", help="Comma-separated categories or 'all'."
    )
    parser.add_argument("--detector", choices=["SDP", "DPM", "FRCNN"], default="SDP")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--output-root", default="scripts/eval/output/ablation_mot17")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    categories = parse_categories(args.category)
    base_args = ["--detector", args.detector]
    if args.max_frames:
        base_args += ["--max-frames", str(args.max_frames)]

    baseline_dir = f"{args.output_root}/baseline"
    if not args.skip_run:
        run_eval("Baseline", baseline_dir, [], base_args, args.dry_run)

    baseline_metrics = evaluate_dir(baseline_dir, args.gt_root, args.detector)

    for category in categories:
        experiments = [("Baseline", baseline_dir, [])]
        for label, slug, extra in _CATEGORY_EXPERIMENTS[category]:
            experiments.append((label, f"{args.output_root}/{category}/{slug}", extra))

        if not args.skip_run:
            for label, out_dir, extra in experiments[1:]:
                ok = run_eval(
                    f"{category}: {label}", out_dir, extra, base_args, args.dry_run
                )
                if not ok:
                    print(f"  [WARN] {category} / {label} failed, continuing...")

        results: list[tuple[str, dict | None]] = [("Baseline", baseline_metrics)]
        for label, out_dir, _ in experiments[1:]:
            results.append((label, evaluate_dir(out_dir, args.gt_root, args.detector)))
        print_table(f"MOT17 Ablation - {category} - detector={args.detector}", results)


if __name__ == "__main__":
    main()
