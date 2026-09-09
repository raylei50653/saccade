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
# status: stable

import argparse
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

import numpy as np  # noqa: E402

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import motmetrics as mm  # noqa: E402


from .ablation_experiments import (  # noqa: E402
    _METRICS,
    _DISPLAY,
    _PCT,
    _CATEGORY_ORDER,
    _CATEGORY_EXPERIMENTS,
)


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

    # Cast to float/int for JSON serialization
    results = {
        m: (float(v) if m in _PCT or "num" not in m else int(v))
        for m, v in results.items()
    }

    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

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
    env = os.environ.copy()
    pythonpath_entries = [str(project_root)]
    if src_path.exists():
        pythonpath_entries.append(str(src_path))
    if build_path.exists():
        pythonpath_entries.append(str(build_path))
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    print(f"\n{'=' * 68}")
    print(f"  Running: {label}")
    print(f"  Cmd:     {' '.join(cmd)}")
    print(f"{'=' * 68}")
    if dry:
        return True
    result = subprocess.run(cmd, cwd=str(project_root), env=env)
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
    parser.add_argument(
        "--sequences", default="", help="Comma-separated sequence names."
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--output-root", default="scripts/eval/output/ablation_mot17")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="mot17-ablation",
        help="MLflow experiment name for ablation runs.",
    )
    args = parser.parse_args()

    base_args = ["--detector", args.detector]
    if args.sequences:
        base_args += ["--sequences", args.sequences]
    if args.max_frames:
        base_args += ["--max-frames", str(args.max_frames)]
    base_args += [
        "--mlflow-uri",
        args.mlflow_uri,
        "--mlflow-experiment",
        args.mlflow_experiment,
    ]

    categories = parse_categories(args.category)

    baseline_dir = f"{args.output_root}/baseline"
    if not args.skip_run:
        run_eval("Baseline", baseline_dir, [], base_args, args.dry_run)

    baseline_metrics = evaluate_dir(baseline_dir, args.gt_root, args.detector)

    for category in categories:
        # Expand experiment tuples: (label, slug, extra[, path_override])
        # path_override points to an existing output dir to reuse (skips run if
        # metrics.json already present there).
        expanded: list[tuple[str, str, list[str]]] = []
        for entry in _CATEGORY_EXPERIMENTS[category]:
            label, slug, extra = entry[0], entry[1], entry[2]
            path_override: str | None = entry[3] if len(entry) > 3 else None  # type: ignore[misc]
            if path_override:
                out_dir = f"{args.output_root}/{path_override}"
            else:
                out_dir = f"{args.output_root}/{category}/{slug}"
            expanded.append((label, out_dir, extra))

        experiments: list[tuple[str, str, list[str]]] = [
            ("Baseline", baseline_dir, [])
        ] + expanded

        if not args.skip_run:
            for label, out_dir, extra in experiments[1:]:
                # Skip if cached — supports both resume and path_override reuse.
                metrics_cache = os.path.join(out_dir, "metrics.json")
                if os.path.exists(metrics_cache):
                    print(f"  [CACHED] {category} / {label}  ({out_dir})")
                    continue
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
