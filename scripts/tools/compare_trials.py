"""Compare a selected set of Optuna trial results."""

# status: diagnostic
import argparse


DISPLAY_COLUMNS = {
    "number": "Trial",
    "value": "Score",
    "params_crowd_conf_threshold": "conf_thr",
    "params_crowd_low_score_trigger": "trigger",
    "params_crowd_new_track_thresh": "new_track_thr",
    "params_scene_adapt_narrow_aspect_thresh": "aspect_thr",
    "user_attrs_IDF1": "IDF1",
    "user_attrs_MOTA": "MOTA",
    "user_attrs_IDs": "IDs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a selected set of Optuna trials."
    )
    parser.add_argument(
        "--storage",
        default="postgresql://saccade:saccade@localhost:5432/optuna",
        help="Optuna storage URL.",
    )
    parser.add_argument("--study", required=True, help="Optuna study name.")
    parser.add_argument(
        "--trials",
        required=True,
        nargs="+",
        type=int,
        help="Trial numbers to compare.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import optuna

    study = optuna.load_study(study_name=args.study, storage=args.storage)
    df = study.trials_dataframe()

    compare_df = df[df["number"].isin(args.trials)].copy()
    if compare_df.empty:
        print(f"No matching trials found for: {args.trials}")
        return 1

    final_cols: list[str] = []
    rename_map: dict[str, str] = {}
    for column, display_name in DISPLAY_COLUMNS.items():
        resolved = column
        if resolved not in compare_df.columns and not column.startswith("user_attrs_"):
            fallback = f"user_attrs_{column}"
            if fallback in compare_df.columns:
                resolved = fallback
        if resolved in compare_df.columns:
            final_cols.append(resolved)
            rename_map[resolved] = display_name

    if not final_cols:
        print("None of the requested display columns are present in the study.")
        return 1

    result = compare_df[final_cols].rename(columns=rename_map)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
