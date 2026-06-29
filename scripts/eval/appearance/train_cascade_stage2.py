#!/usr/bin/env python3
"""Train Stage 2 logistic model for the cascade filter.

The Stage 2 model is trained on the output of the rule baseline (Stage 1),
so it learns the distribution of "hard FP" that survives the rule filter.
This avoids the distribution mismatch that caused the original logistic
to underperform the rule baseline on MOT17.

Usage:
    python scripts/eval/appearance/train_cascade_stage2.py \
        --rows-csv results/crowdhuman_val_external_fp_rows.csv \
        --output-model models/external_fp/cascade_stage2_logistic.json \
        --output-config models/external_fp/cascade_config.json \
        --rule-min-score 0.05 --rule-low-score 0.10 --rule-medium-score 0.18 \
        --rule-min-height 72 --rule-medium-height 96 --rule-min-aspect 1.6 \
        --log-threshold 0.50 --log-max-score 0.18 --log-penalty 0.4
"""

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
    RuleBaselineConfig,
    CascadeFilterConfig,
    apply_cascade_filter,
    train_cascade_stage2_model,
    load_external_rows_csv,
    save_logistic_model,
    evaluate_logistic_classifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Stage 2 logistic model for cascade filter."
    )
    parser.add_argument("--rows-csv", required=True, help="Full detection rows CSV")
    parser.add_argument(
        "--output-model",
        required=True,
        help="Path to save trained logistic model JSON",
    )
    parser.add_argument(
        "--output-config",
        default="",
        help="Path to save cascade config JSON",
    )
    # Rule baseline params
    parser.add_argument("--rule-min-score", type=float, default=0.05)
    parser.add_argument("--rule-low-score", type=float, default=0.10)
    parser.add_argument("--rule-medium-score", type=float, default=0.18)
    parser.add_argument("--rule-min-height", type=float, default=72.0)
    parser.add_argument("--rule-medium-height", type=float, default=96.0)
    parser.add_argument("--rule-min-aspect", type=float, default=1.6)
    # Logistic params
    parser.add_argument("--log-threshold", type=float, default=0.25)
    parser.add_argument("--log-max-score", type=float, default=0.25)
    parser.add_argument("--log-penalty", type=float, default=None)
    parser.add_argument("--log-min-score", type=float, default=0.05)
    # Training params
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load rows
    rows = load_external_rows_csv(Path(args.rows_csv))
    print(f"Loaded {len(rows)} rows")

    # Build rule config
    rule_cfg = RuleBaselineConfig(
        min_score=args.rule_min_score,
        low_score=args.rule_low_score,
        medium_score=args.rule_medium_score,
        min_height=args.rule_min_height,
        medium_height=args.rule_medium_height,
        min_aspect=args.rule_min_aspect,
    )

    # Show Stage 1 baseline stats
    from saccade.perception.eval.external_fp_model import apply_rule_baseline

    _, s1_metrics = apply_rule_baseline(rows, config=rule_cfg)
    print("\nStage 1 (Rule Baseline):")
    print(
        f"  TP: {s1_metrics.tp_total} -> kept={s1_metrics.tp_kept}, removed={s1_metrics.tp_removed}"
    )
    print(
        f"  FP: {s1_metrics.fp_total} -> kept={s1_metrics.fp_kept}, removed={s1_metrics.fp_removed}"
    )
    print(
        f"  Precision: {s1_metrics.precision_after * 100:.1f}%, Recall: {s1_metrics.recall_after * 100:.1f}%"
    )

    # Train Stage 2 model
    print(
        f"\nTraining Stage 2 logistic model ({args.epochs} epochs, lr={args.lr}, l2={args.l2})..."
    )
    model = train_cascade_stage2_model(
        rows,
        rule_config=rule_cfg,
        epochs=args.epochs,
        learning_rate=args.lr,
        l2=args.l2,
    )
    print(
        f"Trained model: {len(model.feature_names)} features, {len(model.weights)} weights"
    )

    # Save model
    model_path = Path(args.output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_logistic_model(model_path, model)
    print(f"Model saved to {model_path}")

    # Evaluate on Stage 1 output
    print("\nStage 2 model evaluation on Stage 1 output:")
    stage1_rows, _ = apply_rule_baseline(rows, config=rule_cfg)
    stage2_eval = evaluate_logistic_classifier(
        model, stage1_rows, threshold=args.log_threshold
    )
    print(f"  Precision: {stage2_eval['precision'] * 100:.1f}%")
    print(f"  Recall: {stage2_eval['recall'] * 100:.1f}%")
    print(f"  Accuracy: {stage2_eval['accuracy'] * 100:.1f}%")

    # Apply full cascade and show results
    config = CascadeFilterConfig(
        rule=rule_cfg,
        log_threshold=args.log_threshold,
        log_max_score=args.log_max_score,
        log_penalty=args.log_penalty,
    )
    final_rows, cascade_metrics = apply_cascade_filter(
        rows, config=config, stage2_model=model
    )
    print(f"\n{'=' * 60}")
    print("Cascade Results:")
    print(f"{'=' * 60}")
    print(f"  Total rows:  {len(rows)}")
    print(
        f"  Stage 1 kept: {cascade_metrics.s1_kept} (removed {cascade_metrics.s1_removed})"
    )
    print(
        f"  Stage 2 kept: {cascade_metrics.s2_kept} (removed {cascade_metrics.s2_removed})"
    )
    print(
        f"  TP: {cascade_metrics.tp_total} -> {cascade_metrics.tp_kept} (removed {cascade_metrics.tp_removed})"
    )
    print(
        f"  FP: {cascade_metrics.fp_total} -> {cascade_metrics.fp_kept} (removed {cascade_metrics.fp_removed})"
    )
    print(
        f"  Precision: {cascade_metrics.precision_before * 100:.1f}% -> {cascade_metrics.precision_after * 100:.1f}%"
    )
    print(f"  Recall: {cascade_metrics.recall_after * 100:.1f}%")
    print(f"  FP Reduction: {cascade_metrics.fp_reduction * 100:.1f}%")

    # Save config if requested
    if args.output_config:
        config_path = Path(args.output_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_data = {
            "rule": {
                "min_score": args.rule_min_score,
                "low_score": args.rule_low_score,
                "medium_score": args.rule_medium_score,
                "min_height": args.rule_min_height,
                "medium_height": args.rule_medium_height,
                "min_aspect": args.rule_min_aspect,
            },
            "log_threshold": args.log_threshold,
            "log_max_score": args.log_max_score,
            "log_penalty": args.log_penalty,
            "model_path": str(model_path),
        }
        config_path.write_text(
            json.dumps(config_data, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Config saved to {config_path}")

    # Print metrics as JSON for integration
    print("\nMetrics JSON:")
    print(
        json.dumps(
            {
                "tp_total": cascade_metrics.tp_total,
                "fp_total": cascade_metrics.fp_total,
                "tp_kept": cascade_metrics.tp_kept,
                "fp_kept": cascade_metrics.fp_kept,
                "tp_removed": cascade_metrics.tp_removed,
                "fp_removed": cascade_metrics.fp_removed,
                "precision_before": cascade_metrics.precision_before,
                "precision_after": cascade_metrics.precision_after,
                "recall_after": cascade_metrics.recall_after,
                "fp_reduction": cascade_metrics.fp_reduction,
                "s1_kept": cascade_metrics.s1_kept,
                "s1_removed": cascade_metrics.s1_removed,
                "s2_kept": cascade_metrics.s2_kept,
                "s2_removed": cascade_metrics.s2_removed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
