#!/usr/bin/env python3
"""Analyse per-frame CSV ledger from --profile-frame-csv.

Input:  _frame_ledger_<seq>.csv
Output: percentile table, top slowest frames, correlations, tail attribution.

Usage:
    uv run scripts/eval/profile_analyze.py path/to/_frame_ledger_MOT17-04-SDP.csv
    uv run scripts/eval/profile_analyze.py path.csv --top-n 10 --out-json out.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TIMING_COLS = [
    "total_ms",
    "fetch_ms",
    "preprocess_ms",
    "detect_ms",
    "post_ms",
    "gmc_ms",
    "reid_ms",
    "track_ms",
    "handover_ms",
    "output_ms",
    "reid_blocking_wait_ms",
    "reid_crop_stash_ms",
    "reid_extract_submit_ms",
    "reid_gpu_ms",
    "reid_d2h_ms",
    "handover_snapshot_ms",
    "handover_score_ms",
    "handover_apply_ms",
]

TOP_FRAME_COLS = [
    "seq",
    "frame",
    "total_ms",
    "detect_ms",
    "post_ms",
    "gmc_ms",
    "reid_ms",
    "track_ms",
    "handover_ms",
    "output_ms",
    "n_dets_final",
    "n_tracks",
    "n_dead_archive",
    "n_crops",
    "n_requery",
    "reid_waited_this_frame",
    "reid_blocking_wait_ms",
]


def load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(rows: list[dict], col: str) -> np.ndarray:
    return np.array([float(r.get(col, 0)) for r in rows], dtype=np.float64)


def _to_int(rows: list[dict], col: str) -> np.ndarray:
    return np.array([int(r.get(col, 0)) for r in rows], dtype=np.int64)


def percentile_table(rows: list[dict], timing_cols: list[str]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for col in timing_cols:
        vals = _to_float(rows, col)
        if not np.any(vals):
            continue
        table.append(
            {
                "stage": col,
                "mean": round(float(np.mean(vals)), 4),
                "p50": round(float(np.percentile(vals, 50)), 4),
                "p90": round(float(np.percentile(vals, 90)), 4),
                "p95": round(float(np.percentile(vals, 95)), 4),
                "p99": round(float(np.percentile(vals, 99)), 4),
                "max": round(float(np.max(vals)), 4),
            }
        )
    return table


def top_slowest(rows: list[dict], top_n: int = 20) -> list[dict[str, Any]]:
    indexed = sorted(
        enumerate(rows), key=lambda x: float(x[1].get("total_ms", 0)), reverse=True
    )
    result: list[dict[str, Any]] = []
    for idx, _ in indexed[:top_n]:
        r = rows[idx]
        result.append({c: r.get(c, 0) for c in TOP_FRAME_COLS})
    return result


def correlation_table(rows: list[dict], timing_cols: list[str]) -> list[dict[str, Any]]:
    pairs = [
        ("total_ms", "n_dets_raw"),
        ("total_ms", "n_dets_after_nms"),
        ("total_ms", "n_dets_final"),
        ("total_ms", "n_tracks"),
        ("total_ms", "n_dead_archive"),
        ("total_ms", "n_crops"),
        ("total_ms", "n_requery"),
        ("total_ms", "reid_blocking_wait_ms"),
        ("post_ms", "n_dets_after_nms"),
        ("reid_ms", "n_crops"),
        ("reid_blocking_wait_ms", "n_requery"),
        ("handover_ms", "n_dead_archive"),
        ("handover_ms", "handover_score_matrix_n"),
    ]

    result: list[dict[str, Any]] = []
    for x_col, y_col in pairs:
        x = _to_float(rows, x_col)
        y = _to_float(rows, y_col)

        if np.std(x) == 0 or np.std(y) == 0:
            result.append(
                {
                    "x": x_col,
                    "y": y_col,
                    "pearson_r": None,
                    "note": "skipped: constant column",
                }
            )
            continue

        r = float(np.corrcoef(x, y)[0, 1])
        result.append({"x": x_col, "y": y_col, "pearson_r": round(r, 4)})

    return result


def tail_attribution(
    rows: list[dict], timing_cols: list[str], top_pct: float = 5.0
) -> list[dict[str, Any]]:
    vals = _to_float(rows, "total_ms")
    threshold = float(np.percentile(vals, 100.0 - top_pct))
    top_idx = vals >= threshold
    top_rows = [r for i, r in enumerate(rows) if top_idx[i]]

    result: list[dict[str, Any]] = []
    for col in timing_cols:
        all_vals = _to_float(rows, col)
        top_vals = _to_float(top_rows, col)
        mean_all = float(np.mean(all_vals))
        mean_top = float(np.mean(top_vals))
        delta = mean_top - mean_all
        ratio = round(mean_top / mean_all, 3) if mean_all > 0 else 0.0
        result.append(
            {
                "stage": col,
                "mean_all": round(mean_all, 4),
                "mean_top5": round(mean_top, 4),
                "delta": round(delta, 4),
                "ratio": ratio,
            }
        )

    result.sort(key=lambda d: d["delta"], reverse=True)
    return result


def print_percentile_table(table: list[dict[str, Any]]) -> None:
    header = f"{'stage':<30s} {'mean':>8s} {'p50':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}"
    print("\n" + "=" * len(header))
    print("STAGE TIMING PERCENTILES (ms)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in table:
        print(
            f"{r['stage']:<30s} "
            f"{r['mean']:>8.2f} {r['p50']:>8.2f} {r['p90']:>8.2f} "
            f"{r['p95']:>8.2f} {r['p99']:>8.2f} {r['max']:>8.2f}"
        )


def print_top_slowest(table: list[dict[str, Any]], top_n: int) -> None:
    print(f"\n{'=' * 110}")
    print(f"TOP {min(top_n, len(table))} SLOWEST FRAMES")
    print(f"{'=' * 110}")
    header = (
        f"{'frame':>6s} {'total':>8s} {'detect':>8s} {'post':>7s} {'gmc':>7s} "
        f"{'reid':>7s} {'track':>7s} {'handover':>8s} {'output':>7s} "
        f"{'dets':>5s} {'tracks':>6s} {'dead_arch':>9s} "
        f"{'crops':>5s} {'requery':>7s} {'reid_wait':>9s} {'reid_block':>10s}"
    )
    print(header)
    print("-" * len(header))
    for r in table:
        print(
            f"{r['frame']:>6s} {float(r.get('total_ms', 0)):>8.2f} "
            f"{float(r.get('detect_ms', 0)):>8.2f} {float(r.get('post_ms', 0)):>7.2f} "
            f"{float(r.get('gmc_ms', 0)):>7.2f} {float(r.get('reid_ms', 0)):>7.2f} "
            f"{float(r.get('track_ms', 0)):>7.2f} {float(r.get('handover_ms', 0)):>8.2f} "
            f"{float(r.get('output_ms', 0)):>7.2f} "
            f"{str(r.get('n_dets_final', 0)):>5s} "
            f"{str(r.get('n_tracks', 0)):>6s} "
            f"{str(r.get('n_dead_archive', 0)):>9s} "
            f"{str(r.get('n_crops', 0)):>5s} "
            f"{str(r.get('n_requery', 0)):>7s} "
            f"{str(r.get('reid_waited_this_frame', 0)):>9s} "
            f"{float(r.get('reid_blocking_wait_ms', 0)):>10.2f}"
        )


def print_correlations(table: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 85}")
    print("CORRELATIONS (Pearson r)")
    print(f"{'=' * 85}")
    header = f"{'x':<28s} {'y':<28s} {'r':>8s}  note"
    print(header)
    print("-" * len(header))
    for r in table:
        if r["pearson_r"] is not None:
            print(f"{r['x']:<28s} {r['y']:<28s} {r['pearson_r']:>8.4f}")
        else:
            print(f"{r['x']:<28s} {r['y']:<28s} {'-':>8s}  {r.get('note', '')}")


def print_tail_attribution(table: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 80}")
    print("TAIL ATTRIBUTION — top 5% slowest frames vs all frames (ms)")
    print(f"{'=' * 80}")
    header = (
        f"{'stage':<30s} {'mean_all':>9s} {'mean_top5':>9s} {'delta':>9s} {'ratio':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in table:
        print(
            f"{r['stage']:<30s} "
            f"{r['mean_all']:>9.2f} {r['mean_top5']:>9.2f} "
            f"{r['delta']:>9.2f} {r['ratio']:>7.2f}"
            if r["ratio"] > 0
            else f"{r['stage']:<30s} {r['mean_all']:>9.2f} {r['mean_top5']:>9.2f} {r['delta']:>9.2f} {'-':>7s}"
        )


def maybe_plot(
    rows: list[dict],
    timing_cols: list[str],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠  matplotlib not available — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total = _to_float(rows, "total_ms")

    # 1. Total_ms histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(total, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("total_ms")
    ax.set_ylabel("frames")
    ax.set_title("Frame latency distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_total_ms.png", dpi=120)
    plt.close(fig)

    # 2. Stage breakdown waterfall of mean
    nonzero_cols = [
        c for c in timing_cols if c != "total_ms" and np.any(_to_float(rows, c) > 0)
    ]
    if nonzero_cols:
        means = [float(np.mean(_to_float(rows, c))) for c in nonzero_cols]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(nonzero_cols[::-1], means[::-1], color="steelblue")
        ax.set_xlabel("mean wall time (ms)")
        ax.set_title("Stage breakdown (mean)")
        fig.tight_layout()
        fig.savefig(output_dir / "stage_breakdown_mean.png", dpi=120)
        plt.close(fig)

    # 3. Scatter: total_ms vs n_dets_final
    if np.any(dets := _to_int(rows, "n_dets_final")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(dets, total, alpha=0.3, s=4, color="steelblue")
        ax.set_xlabel("n_dets_final")
        ax.set_ylabel("total_ms")
        ax.set_title("Latency vs final detection count")
        fig.tight_layout()
        fig.savefig(output_dir / "scatter_total_vs_ndets.png", dpi=120)
        plt.close(fig)

    # 4. Scatter: total_ms vs reid_blocking_wait_ms
    if np.any(block := _to_float(rows, "reid_blocking_wait_ms")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(block, total, alpha=0.3, s=4, color="coral")
        ax.set_xlabel("reid_blocking_wait_ms")
        ax.set_ylabel("total_ms")
        ax.set_title("Latency vs ReID blocking wait")
        fig.tight_layout()
        fig.savefig(output_dir / "scatter_total_vs_reid_block.png", dpi=120)
        plt.close(fig)

    print(f"\n📊 Plots saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse per-frame CSV ledger from --profile-frame-csv."
    )
    parser.add_argument("csv", type=str, help="Path to _frame_ledger_<seq>.csv")
    parser.add_argument(
        "--top-n", type=int, default=20, help="Number of slowest frames to show"
    )
    parser.add_argument(
        "--out-json", type=str, default=None, help="Write full results to JSON file"
    )
    parser.add_argument(
        "--plots", action="store_true", help="Generate plots (requires matplotlib)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(str(csv_path))
    if not rows:
        print("CSV is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} frames from {csv_path.name}")

    pct_table = percentile_table(rows, TIMING_COLS)
    top_frames = top_slowest(rows, args.top_n)
    corr_table = correlation_table(rows, TIMING_COLS)
    tail_attr = tail_attribution(rows, TIMING_COLS)

    print_percentile_table(pct_table)
    print_top_slowest(top_frames, args.top_n)
    print_correlations(corr_table)
    print_tail_attribution(tail_attr)

    if args.out_json:
        out = {
            "source": str(csv_path),
            "n_frames": len(rows),
            "percentiles": pct_table,
            "top_slowest": top_frames,
            "correlations": corr_table,
            "tail_attribution": tail_attr,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n📄 JSON written to {args.out_json}")

    if args.plots:
        maybe_plot(rows, TIMING_COLS, csv_path.parent / "plots")


if __name__ == "__main__":
    main()
