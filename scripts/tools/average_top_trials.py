"""Print top Optuna trials and mean of their parameters."""

# status: experiment
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print top Optuna trials and the mean of their parameter values."
    )
    parser.add_argument(
        "--storage",
        default="postgresql://saccade:saccade@localhost:5432/optuna",
        help="Optuna storage URL.",
    )
    parser.add_argument("--study", required=True, help="Optuna study name.")
    parser.add_argument(
        "--min-value",
        type=float,
        default=65.0,
        help="Minimum objective value required for a trial to be included.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import optuna

    study = optuna.load_study(study_name=args.study, storage=args.storage)
    df = study.trials_dataframe()

    if "value" not in df.columns:
        print("Study has no 'value' column; nothing to average.")
        return 1

    top_trials = df[df["value"] > args.min_value].copy()
    if top_trials.empty:
        print(f"No trials found with score > {args.min_value}.")
        return 0

    param_cols = [col for col in top_trials.columns if col.startswith("params_")]
    if not param_cols:
        print("No parameter columns found in the selected trials.")
        return 1

    averages = top_trials[param_cols].mean(numeric_only=True)

    print(f"Found {len(top_trials)} trials with score > {args.min_value}.")
    print("\nIndividual Top Trials:")
    print(top_trials[["number", "value"] + param_cols].to_string(index=False))

    print("\nAveraged Parameters:")
    for param, value in averages.items():
        clean_name = param.replace("params_", "", 1)
        print(f"  {clean_name}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
