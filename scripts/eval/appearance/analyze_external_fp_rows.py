#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

project_root = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from saccade.perception.eval.external_fp_model import (  # noqa: E402
    CascadeFilterConfig,
    RuleBaselineConfig,
    apply_cascade_filter,
    apply_rule_baseline,
    bucketize_feature,
    compute_quantiles,
    count_labels,
    load_external_rows_csv,
    load_external_fp_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze TP/FP distributions from external rows CSV."
    )
    parser.add_argument(
        "--rows-csv", required=True, help="CSV exported by export_external_fp_rows.py"
    )
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    # Cascade filter options
    parser.add_argument(
        "--cascade-model",
        default="",
        help="Path to cascade Stage 2 logistic model JSON",
    )
    parser.add_argument("--cascade-log-threshold", type=float, default=0.25)
    parser.add_argument("--cascade-log-max-score", type=float, default=0.25)
    parser.add_argument("--cascade-log-penalty", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_external_rows_csv(Path(args.rows_csv))
    counts = count_labels(rows)

    score_bins = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]
    height_bins = [0.0, 32.0, 64.0, 96.0, 128.0, 192.0, 256.0, 512.0, 4096.0]
    aspect_bins = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0]

    score_summary = bucketize_feature(rows, feature="score", bins=score_bins)
    height_summary = bucketize_feature(rows, feature="height", bins=height_bins)
    aspect_summary = bucketize_feature(rows, feature="aspect_ratio", bins=aspect_bins)
    quantiles = {
        feature: compute_quantiles(rows, feature=feature, labels=("tp", "fp"))
        for feature in ("score", "height", "aspect_ratio", "area")
    }

    _, rule_metrics = apply_rule_baseline(rows, config=RuleBaselineConfig())
    result: dict = {
        "counts": counts,
        "quantiles": quantiles,
        "score_buckets": score_summary,
        "height_buckets": height_summary,
        "aspect_buckets": aspect_summary,
        "rule_baseline": {key: value for key, value in rule_metrics.__dict__.items()},
    }

    # Cascade filter analysis
    if args.cascade_model:
        model = load_external_fp_model(Path(args.cascade_model))
        cascade_config = CascadeFilterConfig(
            log_threshold=args.cascade_log_threshold,
            log_max_score=args.cascade_log_max_score,
            log_penalty=args.cascade_log_penalty,
        )
        _, cascade_metrics = apply_cascade_filter(
            rows, config=cascade_config, stage2_model=model
        )
        result["cascade_filter"] = {
            key: value for key, value in cascade_metrics.__dict__.items()
        }
        print("\n" + "=" * 60)
        print("Cascade Filter Results:")
        print("=" * 60)
        cm = cascade_metrics
        print(f"  Total rows:  {len(rows)}")
        print(f"  Stage 1 kept: {cm.s1_kept} (removed {cm.s1_removed})")
        print(f"  Stage 2 kept: {cm.s2_kept} (removed {cm.s2_removed})")
        print(f"  TP: {cm.tp_total} -> {cm.tp_kept} (removed {cm.tp_removed})")
        print(f"  FP: {cm.fp_total} -> {cm.fp_kept} (removed {cm.fp_removed})")
        print(
            f"  Precision: {cm.precision_before * 100:.1f}% -> {cm.precision_after * 100:.1f}%"
        )
        print(f"  Recall: {cm.recall_after * 100:.1f}%")
        print(f"  FP Reduction: {cm.fp_reduction * 100:.1f}%")

    print(json.dumps(result, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
