#!/usr/bin/env python3
"""Paired moving-block bootstrap for size-binned detector recall."""
# status: diagnostic

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two frame-level size recall reports with paired blocks."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--sequence", default="MOT17-02-SDP")
    parser.add_argument("--thresholds", default="0.001,0.1,0.25")
    parser.add_argument("--bins", default="all,min_4to8,min_8to16")
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--output", required=True)
    return parser


def _load_frame_counts(
    report: dict[str, Any],
    sequence: str,
    threshold: str,
    bin_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold_report = report["sequences"][sequence]["thresholds"][threshold]
    frames = threshold_report.get("frames")
    if frames is None:
        raise ValueError(
            f"{sequence} threshold {threshold} has no frame records; "
            "rerun evaluator with --save-frame-records"
        )

    frame_ids: list[int] = []
    gt_counts: list[int] = []
    matched_counts: list[int] = []
    for frame in frames:
        item = frame["bins"].get(bin_name, {})
        frame_ids.append(int(frame["frame_id"]))
        gt_counts.append(int(item.get("gt", 0)))
        matched_counts.append(int(item.get("matched", 0)))
    return (
        np.asarray(frame_ids, dtype=np.int64),
        np.asarray(gt_counts, dtype=np.int64),
        np.asarray(matched_counts, dtype=np.int64),
    )


def _moving_block_indices(
    rng: np.random.Generator,
    *,
    frame_count: int,
    block_length: int,
    samples: int,
) -> np.ndarray:
    blocks_per_sample = int(np.ceil(frame_count / block_length))
    starts = rng.integers(
        0,
        frame_count,
        size=(samples, blocks_per_sample),
        endpoint=False,
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets) % frame_count
    return indices.reshape(samples, -1)[:, :frame_count]


def _percentile_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def _analyze_bin(
    baseline_gt: np.ndarray,
    baseline_matched: np.ndarray,
    candidate_gt: np.ndarray,
    candidate_matched: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    if not np.array_equal(baseline_gt, candidate_gt):
        raise ValueError("Paired reports have different per-frame GT counts")

    baseline_total_gt = int(baseline_gt.sum())
    candidate_total_gt = int(candidate_gt.sum())
    baseline_total_matched = int(baseline_matched.sum())
    candidate_total_matched = int(candidate_matched.sum())
    baseline_recall = baseline_total_matched / max(baseline_total_gt, 1)
    candidate_recall = candidate_total_matched / max(candidate_total_gt, 1)

    sampled_gt = baseline_gt[indices].sum(axis=1)
    sampled_baseline_matched = baseline_matched[indices].sum(axis=1)
    sampled_candidate_matched = candidate_matched[indices].sum(axis=1)
    valid = sampled_gt > 0
    sampled_gt = sampled_gt[valid]
    sampled_baseline_matched = sampled_baseline_matched[valid]
    sampled_candidate_matched = sampled_candidate_matched[valid]

    baseline_boot_recall = sampled_baseline_matched / sampled_gt
    candidate_boot_recall = sampled_candidate_matched / sampled_gt
    recall_delta_pp = (candidate_boot_recall - baseline_boot_recall) * 100.0

    baseline_fn = sampled_gt - sampled_baseline_matched
    candidate_fn = sampled_gt - sampled_candidate_matched
    valid_fn = baseline_fn > 0
    fn_reduction = (
        (baseline_fn[valid_fn] - candidate_fn[valid_fn]) / baseline_fn[valid_fn] * 100.0
    )

    observed_baseline_fn = baseline_total_gt - baseline_total_matched
    observed_candidate_fn = candidate_total_gt - candidate_total_matched
    observed_fn_reduction = (
        (observed_baseline_fn - observed_candidate_fn)
        / max(observed_baseline_fn, 1)
        * 100.0
    )
    return {
        "gt": baseline_total_gt,
        "baseline_matched": baseline_total_matched,
        "candidate_matched": candidate_total_matched,
        "baseline_recall": baseline_recall,
        "candidate_recall": candidate_recall,
        "recall_delta_pp": (candidate_recall - baseline_recall) * 100.0,
        "recall_delta_pp_ci95": _percentile_interval(recall_delta_pp),
        "probability_recall_delta_gt_zero": float(np.mean(recall_delta_pp > 0.0)),
        "baseline_fn": observed_baseline_fn,
        "candidate_fn": observed_candidate_fn,
        "fn_reduction_percent": observed_fn_reduction,
        "fn_reduction_percent_ci95": _percentile_interval(fn_reduction),
    }


def main() -> None:
    args = build_parser().parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    thresholds = [item.strip() for item in args.thresholds.split(",") if item.strip()]
    bins = [item.strip() for item in args.bins.split(",") if item.strip()]

    first_threshold = thresholds[0]
    first_bin = bins[0]
    baseline_frame_ids, _, _ = _load_frame_counts(
        baseline, args.sequence, first_threshold, first_bin
    )
    candidate_frame_ids, _, _ = _load_frame_counts(
        candidate, args.sequence, first_threshold, first_bin
    )
    if not np.array_equal(baseline_frame_ids, candidate_frame_ids):
        raise ValueError("Paired reports have different frame IDs")
    if args.block_length <= 0 or args.block_length > len(baseline_frame_ids):
        raise ValueError("block-length must be between 1 and the frame count")

    rng = np.random.default_rng(args.seed)
    indices = _moving_block_indices(
        rng,
        frame_count=len(baseline_frame_ids),
        block_length=args.block_length,
        samples=args.samples,
    )
    results: dict[str, Any] = {
        "baseline": baseline["label"],
        "candidate": candidate["label"],
        "sequence": args.sequence,
        "frame_count": len(baseline_frame_ids),
        "block_length": args.block_length,
        "samples": args.samples,
        "seed": args.seed,
        "thresholds": {},
    }

    for threshold in thresholds:
        threshold_results: dict[str, Any] = {}
        for bin_name in bins:
            baseline_ids, baseline_gt, baseline_matched = _load_frame_counts(
                baseline, args.sequence, threshold, bin_name
            )
            candidate_ids, candidate_gt, candidate_matched = _load_frame_counts(
                candidate, args.sequence, threshold, bin_name
            )
            if not np.array_equal(baseline_ids, candidate_ids):
                raise ValueError(f"Frame mismatch at threshold {threshold}")
            threshold_results[bin_name] = _analyze_bin(
                baseline_gt,
                baseline_matched,
                candidate_gt,
                candidate_matched,
                indices,
            )
        results["thresholds"][threshold] = threshold_results

    for threshold, threshold_results in results["thresholds"].items():
        print(f"\nscore>={threshold}")
        for bin_name, item in threshold_results.items():
            ci = item["recall_delta_pp_ci95"]
            fn_ci = item["fn_reduction_percent_ci95"]
            print(
                f"  {bin_name:12s} delta={item['recall_delta_pp']:+.3f}pp "
                f"CI95=[{ci[0]:+.3f},{ci[1]:+.3f}] "
                f"P(delta>0)={item['probability_recall_delta_gt_zero']:.3f} "
                f"FNred={item['fn_reduction_percent']:+.2f}% "
                f"CI95=[{fn_ci[0]:+.2f},{fn_ci[1]:+.2f}]"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
