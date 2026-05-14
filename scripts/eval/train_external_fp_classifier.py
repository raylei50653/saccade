#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from saccade.perception.eval.external_fp_model import (  # noqa: E402
    STRUCTURAL_FEATURE_COLUMNS,
    evaluate_softmax_classifier,
    evaluate_logistic_classifier,
    fit_banded_logistic_classifier,
    fit_logistic_classifier,
    fit_softmax_classifier,
    load_external_rows_csv,
    save_logistic_model,
    split_rows_train_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a small logistic TP/FP classifier from external rows."
    )
    parser.add_argument(
        "--rows-csv", required=True, help="CSV exported by export_external_fp_rows.py"
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="JSON output path for trained logistic model.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio by image_id.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--model-kind",
        choices=["single", "banded", "softmax3"],
        default="single",
        help="Train one global logistic model or score-band weighted logistic models.",
    )
    parser.add_argument(
        "--score-bands",
        default="0.05,0.10,0.18",
        help="Comma-separated score band edges for --model-kind banded.",
    )
    parser.add_argument(
        "--output-metrics-json",
        default="",
        help="Optional JSON path for train/eval metrics.",
    )
    parser.add_argument(
        "--softmax3-class-weight-multipliers",
        default="1.0,1.0,1.0",
        help="Comma-separated multipliers applied to inverse-frequency class weights for tp,fp,np.",
    )
    return parser


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def main() -> None:
    args = build_parser().parse_args()
    rows = load_external_rows_csv(Path(args.rows_csv))
    train_rows, eval_rows = split_rows_train_eval(rows, eval_ratio=args.eval_ratio)
    if not train_rows or not eval_rows:
        raise ValueError("Need non-empty train/eval split to train classifier")
    softmax3_class_weight_multipliers: list[float] = []

    if args.model_kind == "banded":
        score_bands = _parse_float_list(args.score_bands)
        model = fit_banded_logistic_classifier(
            train_rows,
            band_edges=score_bands,
            feature_names=STRUCTURAL_FEATURE_COLUMNS,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    elif args.model_kind == "softmax3":
        score_bands = []
        softmax3_class_weight_multipliers = _parse_float_list(
            args.softmax3_class_weight_multipliers
        )
        model = fit_softmax_classifier(
            train_rows,
            feature_names=STRUCTURAL_FEATURE_COLUMNS,
            class_weight_multipliers=softmax3_class_weight_multipliers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    else:
        score_bands = []
        model = fit_logistic_classifier(
            train_rows,
            feature_names=STRUCTURAL_FEATURE_COLUMNS,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    save_logistic_model(Path(args.output_model), model)

    if args.model_kind == "softmax3":
        train_metrics = evaluate_softmax_classifier(model, train_rows)
        eval_metrics = evaluate_softmax_classifier(model, eval_rows)
    else:
        train_metrics = evaluate_logistic_classifier(
            model, train_rows, threshold=args.threshold
        )
        eval_metrics = evaluate_logistic_classifier(
            model, eval_rows, threshold=args.threshold
        )
    summary = {
        "model_kind": args.model_kind,
        "score_bands": score_bands,
        "softmax3_class_weight_multipliers": softmax3_class_weight_multipliers,
        "feature_names": list(model.feature_names),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_label_counts": {},
        "eval_label_counts": {},
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    for split_name, split_rows in (
        ("train_label_counts", train_rows),
        ("eval_label_counts", eval_rows),
    ):
        counts: dict[str, int] = {}
        for row in split_rows:
            label = str(row["label"]).lower()
            counts[label] = counts.get(label, 0) + 1
        summary[split_name] = counts
    print(json.dumps(summary, indent=2))

    if args.output_metrics_json:
        output_path = Path(args.output_metrics_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
