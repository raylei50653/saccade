#!/usr/bin/env python3
# mypy: ignore-errors
"""
Summarize MOT17 ablation experiment folders against a shared baseline.

This script scans the output tree produced by ablation_mot17.py, recomputes
tracking metrics from the per-sequence MOT result files, and reports how each
experiment moved the key metrics relative to baseline.

Examples:
    uv run python scripts/eval/summarize_ablation_mot17.py
    uv run python scripts/eval/summarize_ablation_mot17.py --category detection
    uv run python scripts/eval/summarize_ablation_mot17.py --sort-by mota_delta
    uv run python scripts/eval/summarize_ablation_mot17.py --csv output/ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

_METRICS = [
    "idf1",
    "recall",
    "precision",
    "mota",
    "num_switches",
    "num_false_positives",
    "num_misses",
]
_PCT = {"idf1", "recall", "precision", "mota"}
_CATEGORY_ORDER = [
    "detection",
    "association",
    "geometry",
    "reid",
    "semantic",
    "trigger",
    "lifecycle",
    "combination",
    "combination_fine",
    "mot17_a",
    "mot17_b",
    "mot17_c",
]
_SORT_KEYS = {
    "category",
    "experiment",
    "idf1",
    "idf1_delta",
    "mota",
    "mota_delta",
    "recall_delta",
    "precision_delta",
    "num_switches",
    "num_switches_delta",
    "num_false_positives_delta",
    "num_misses_delta",
    "fps",
    "fps_delta",
}


def is_mot_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("MOT") and name.endswith(".txt")


def parse_categories(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(_CATEGORY_ORDER)
    unknown = [name for name in names if name not in _CATEGORY_ORDER]
    if unknown:
        raise ValueError(f"Unsupported categories: {', '.join(unknown)}")
    return [name for name in _CATEGORY_ORDER if name in names]


def load_eval_fps_summary(results_dir: Path) -> dict[str, dict[str, str]]:
    summary_path = results_dir / "_fps_summary.txt"
    if not summary_path.exists():
        return {}

    eval_meta: dict[str, dict[str, str]] = {}
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        name = parts[0]
        fields: dict[str, str] = {}
        for field in parts[1:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            fields[key] = value
        eval_meta[name] = fields
    return eval_meta


def load_overall_fps(results_dir: Path) -> float | None:
    overall = load_eval_fps_summary(results_dir).get("OVERALL")
    if not overall:
        return None
    fps = overall.get("fps")
    if not fps or fps == "n/a":
        return None
    try:
        return float(fps)
    except ValueError:
        return None


def evaluate_dir_per_seq(
    results_dir: Path, gt_root: Path, detector: str | None
) -> dict[str, dict[str, float]] | None:
    """Return per-sequence metrics dict, keyed by sequence name."""
    import numpy as np
    import motmetrics as mm

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

    files = sorted(
        f
        for f in glob.glob(os.path.join(results_dir.as_posix(), "*.txt"))
        if is_mot_file(f)
    )
    if detector:
        files = [f for f in files if f"-{detector}" in Path(f).stem]
    if not files:
        return None

    gt_files = glob.glob(os.path.join(gt_root.as_posix(), "*/gt/gt.txt"))
    if detector:
        gt_files = [f for f in gt_files if f"-{detector}" in Path(f).parts[-3]]
    gt = {
        Path(f).parts[-3]: mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)
        for f in gt_files
    }

    per_seq: dict[str, dict[str, float]] = {}
    mh = mm.metrics.create()
    for f in files:
        name = Path(f).stem
        if name not in gt:
            continue
        ts = mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1.0)
        acc = mm.utils.compare_to_groundtruth(gt[name], ts, "iou", distth=0.5)
        summary = mh.compute(acc, metrics=_METRICS, name=name)
        per_seq[name] = {m: float(summary.loc[name, m]) for m in _METRICS}

    return per_seq if per_seq else None


def print_per_seq_table(
    title: str,
    experiments: list[tuple[str, dict[str, dict[str, float]] | None]],
) -> None:
    """Print per-sequence IDF1 / IDs / MOTA breakdown across experiments."""
    # Collect all sequence names in sorted order
    all_seqs: list[str] = []
    for _, seq_data in experiments:
        if seq_data:
            for seq in seq_data:
                if seq not in all_seqs:
                    all_seqs.append(seq)
    all_seqs.sort()

    print(f"\n{'=' * 120}")
    print(f"  {title}  [per-sequence IDF1 / IDs / MOTA]")
    print(f"{'=' * 120}")

    col_w = 12
    header = f"{'Sequence':<22}"
    for label, _ in experiments:
        short = label[:col_w]
        header += f"  {short:>{col_w}}"
    print(header)
    print("-" * len(header))

    base_data = experiments[0][1] if experiments else None

    for seq in all_seqs:
        row = f"{seq:<22}"
        for label, seq_data in experiments:
            if seq_data is None or seq not in seq_data:
                row += f"  {'n/a':>{col_w}}"
                continue
            m = seq_data[seq]
            cell = f"{m['idf1']*100:.1f}/{int(m['num_switches'])}/{m['mota']*100:.1f}"
            # mark delta vs baseline
            if base_data and seq in base_data and label != experiments[0][0]:
                d_ids = int(m["num_switches"]) - int(base_data[seq]["num_switches"])
                sign = "+" if d_ids >= 0 else ""
                cell += f"({sign}{d_ids})"
            row += f"  {cell:>{col_w}}"
        print(row)

    print(f"{'=' * 120}")
    print("  Format: IDF1% / IDs / MOTA%  (delta IDs vs baseline in parens)")
    print(f"{'=' * 120}")


def evaluate_dir(
    results_dir: Path, gt_root: Path, detector: str | None
) -> dict[str, float] | None:
    import numpy as np
    import motmetrics as mm
    import json

    cache_path = results_dir / "metrics.json"
    if cache_path.exists():
        try:
            with cache_path.open("r") as f:
                return json.load(f)
        except Exception:
            pass

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

    files = sorted(
        f
        for f in glob.glob(os.path.join(results_dir.as_posix(), "*.txt"))
        if is_mot_file(f)
    )
    if detector:
        files = [f for f in files if f"-{detector}" in Path(f).stem]
    if not files:
        return None

    gt_files = glob.glob(os.path.join(gt_root.as_posix(), "*/gt/gt.txt"))
    if detector:
        gt_files = [f for f in gt_files if f"-{detector}" in Path(f).parts[-3]]
    gt = {
        Path(f).parts[-3]: mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)
        for f in gt_files
    }

    accs, names = [], []
    for f in files:
        name = Path(f).stem
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
    metrics = {metric: float(summary.loc["OVERALL", metric]) for metric in _METRICS}
    fps = load_overall_fps(results_dir)
    if fps is not None:
        metrics["fps"] = fps

    # Save to cache
    try:
        with cache_path.open("w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    return metrics


def pct_text(val: float) -> str:
    return f"{val * 100:.1f}%"


def delta_pp_text(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}pp"


def int_text(val: float) -> str:
    return f"{int(round(val))}"


def delta_int_text(val: float) -> str:
    rounded = int(round(val))
    sign = "+" if rounded >= 0 else ""
    return f"{sign}{rounded}"


def fps_text(val: float | None) -> str:
    return "n/a" if val is None else f"{val:.2f}"


def fps_delta_text(val: float | None) -> str:
    if val is None:
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}"


def make_row(
    category: str,
    experiment: str,
    path: Path,
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "category": category,
        "experiment": experiment,
        "path": path.as_posix(),
    }
    for metric in _METRICS:
        row[metric] = metrics[metric]
        row[f"{metric}_delta"] = metrics[metric] - baseline[metric]
    fps = metrics.get("fps")
    base_fps = baseline.get("fps")
    row["fps"] = fps
    row["fps_delta"] = None if fps is None or base_fps is None else fps - base_fps
    return row


def print_table(title: str, rows: list[dict[str, object]]) -> None:
    print(f"\n{'=' * 140}")
    print(f"  {title}")
    print(f"{'=' * 140}")
    header = (
        f"{'Category':<12} {'Experiment':<22} {'IDF1':>8} {'dIDF1':>8} "
        f"{'MOTA':>8} {'dMOTA':>8} {'IDs':>7} {'dIDs':>7} "
        f"{'FP':>7} {'dFP':>7} {'FN':>7} {'dFN':>7} {'FPS':>8} {'dFPS':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['category']):<12} {str(row['experiment']):<22} "
            f"{pct_text(float(row['idf1'])):>8} {delta_pp_text(float(row['idf1_delta'])):>8} "
            f"{pct_text(float(row['mota'])):>8} {delta_pp_text(float(row['mota_delta'])):>8} "
            f"{int_text(float(row['num_switches'])):>7} {delta_int_text(float(row['num_switches_delta'])):>7} "
            f"{int_text(float(row['num_false_positives'])):>7} {delta_int_text(float(row['num_false_positives_delta'])):>7} "
            f"{int_text(float(row['num_misses'])):>7} {delta_int_text(float(row['num_misses_delta'])):>7} "
            f"{fps_text(row['fps'] if isinstance(row['fps'], float) else None):>8} "
            f"{fps_delta_text(row['fps_delta'] if isinstance(row['fps_delta'], float) else None):>8}"
        )
    print(f"{'=' * 140}")


def sort_rows(rows: list[dict[str, object]], sort_by: str) -> list[dict[str, object]]:
    if sort_by not in _SORT_KEYS:
        raise ValueError(f"Unsupported sort key: {sort_by}")
    reverse = sort_by not in {"category", "experiment"}

    def key(row: dict[str, object]) -> tuple:
        value = row.get(sort_by)
        if value is None:
            normalized = float("-inf") if reverse else float("inf")
        else:
            normalized = value
        return (
            (row["category"], row["experiment"])
            if sort_by == "category"
            else (
                normalized,
                row["category"],
                row["experiment"],
            )
        )

    return sorted(rows, key=key, reverse=reverse)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "experiment",
        "path",
        "idf1",
        "idf1_delta",
        "recall",
        "recall_delta",
        "precision",
        "precision_delta",
        "mota",
        "mota_delta",
        "num_switches",
        "num_switches_delta",
        "num_false_positives",
        "num_false_positives_delta",
        "num_misses",
        "num_misses_delta",
        "fps",
        "fps_delta",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="scripts/eval/output/ablation_mot17",
        help="Ablation output root produced by ablation_mot17.py.",
    )
    parser.add_argument(
        "--category",
        default="all",
        help="Comma-separated categories or 'all'.",
    )
    parser.add_argument("--detector", choices=["SDP", "DPM", "FRCNN"], default="SDP")
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument(
        "--sort-by",
        default="idf1_delta",
        help=f"Sort key: {', '.join(sorted(_SORT_KEYS))}",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--per-seq",
        action="store_true",
        default=False,
        help="Print per-sequence IDF1/IDs/MOTA breakdown for selected experiments.",
    )
    parser.add_argument(
        "--per-seq-experiments",
        default="",
        help="Comma-separated experiment slug names to include in --per-seq table (default: all in category).",
    )
    parser.add_argument(
        "--per-seq-extra-dirs",
        default="",
        help="Comma-separated dirs relative to --root to append as extra columns in --per-seq table.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    gt_root = Path(args.gt_root)
    categories = parse_categories(args.category)

    baseline_dir = root / "baseline"
    baseline_metrics = evaluate_dir(baseline_dir, gt_root, args.detector)
    if baseline_metrics is None:
        raise SystemExit(f"Baseline results not found in {baseline_dir}")

    rows: list[dict[str, object]] = []
    for category in categories:
        category_dir = root / category
        if not category_dir.exists():
            print(f"[WARN] Missing category dir: {category_dir}")
            continue
        for experiment_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            metrics = evaluate_dir(experiment_dir, gt_root, args.detector)
            if metrics is None:
                print(f"[WARN] No MOT results in {experiment_dir}")
                continue
            rows.append(
                make_row(
                    category=category,
                    experiment=experiment_dir.name,
                    path=experiment_dir,
                    metrics=metrics,
                    baseline=baseline_metrics,
                )
            )

    if not rows:
        raise SystemExit("No experiment results found.")

    sorted_rows = sort_rows(rows, args.sort_by)
    print_table(
        f"MOT17 Ablation Summary - detector={args.detector} - baseline={baseline_dir.as_posix()}",
        sorted_rows,
    )

    print("\nBaseline")
    print(
        f"IDF1={pct_text(baseline_metrics['idf1'])} "
        f"MOTA={pct_text(baseline_metrics['mota'])} "
        f"IDs={int_text(baseline_metrics['num_switches'])} "
        f"FP={int_text(baseline_metrics['num_false_positives'])} "
        f"FN={int_text(baseline_metrics['num_misses'])} "
        f"FPS={fps_text(baseline_metrics.get('fps'))}"
    )

    if args.csv:
        csv_path = Path(args.csv)
        write_csv(csv_path, sorted_rows)
        print(f"CSV written to {csv_path}")

    if args.per_seq:
        filter_slugs = {s.strip() for s in args.per_seq_experiments.split(",") if s.strip()}
        # baseline always first
        per_seq_experiments: list[tuple[str, dict[str, dict[str, float]] | None]] = [
            ("baseline", evaluate_dir_per_seq(baseline_dir, gt_root, args.detector))
        ]
        for category in categories:
            category_dir = root / category
            if not category_dir.exists():
                continue
            for experiment_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                slug = experiment_dir.name
                if filter_slugs and slug not in filter_slugs:
                    continue
                seq_data = evaluate_dir_per_seq(experiment_dir, gt_root, args.detector)
                if seq_data:
                    per_seq_experiments.append((slug, seq_data))
        for extra_rel in (s.strip() for s in args.per_seq_extra_dirs.split(",") if s.strip()):
            extra_dir = root / extra_rel
            seq_data = evaluate_dir_per_seq(extra_dir, gt_root, args.detector)
            if seq_data:
                per_seq_experiments.append((extra_dir.name, seq_data))
            else:
                print(f"[WARN] No results in extra dir: {extra_dir}")
        print_per_seq_table(
            f"Per-sequence breakdown - detector={args.detector}",
            per_seq_experiments,
        )


if __name__ == "__main__":
    main()
