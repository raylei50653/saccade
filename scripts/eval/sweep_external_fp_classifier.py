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
    load_external_rows_csv,
    load_logistic_model,
    sweep_low_score_logistic_filter,
)


def _parse_float_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep low-score logistic TP/FP filters on CrowdHuman external rows."
    )
    parser.add_argument("--rows-csv", required=True)
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--max-scores", default="0.10,0.12,0.15,0.18,0.22")
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--penalties", default="0.70,0.80,0.90")
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-recall", type=float, default=0.94)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_external_rows_csv(Path(args.rows_csv))
    model = load_logistic_model(Path(args.model_json))
    candidates = sweep_low_score_logistic_filter(
        rows,
        model=model,
        max_scores=_parse_float_list(args.max_scores),
        thresholds=_parse_float_list(args.thresholds),
        penalties=_parse_float_list(args.penalties),
        min_score=float(args.min_score),
    )
    candidates_json = [candidate.__dict__ for candidate in candidates]
    pareto = [
        row for row in candidates_json if row["recall_after"] >= float(args.min_recall)
    ]
    pareto.sort(
        key=lambda row: (
            -float(row["fp_reduction"]),
            -float(row["precision_after"]),
            -float(row["recall_after"]),
            float(row["max_score"]),
            float(row["threshold"]),
            float(row["penalty"]),
        )
    )
    result = {
        "rows_csv": str(Path(args.rows_csv)),
        "model_json": str(Path(args.model_json)),
        "grid": {
            "max_scores": _parse_float_list(args.max_scores),
            "thresholds": _parse_float_list(args.thresholds),
            "penalties": _parse_float_list(args.penalties),
            "min_score": float(args.min_score),
            "min_recall": float(args.min_recall),
        },
        "top_candidates": pareto[: max(int(args.top_k), 1)],
        "all_candidates": candidates_json,
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
