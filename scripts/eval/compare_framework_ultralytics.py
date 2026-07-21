# mypy: ignore-errors
"""
Side-by-side MOTA comparison: Saccade vs Ultralytics (or any two result sets).

Usage:
    python scripts/eval/compare_framework_ultralytics.py \
        --saccade   results/MOT17_eval \
        --ultralytics results/MOT17_ultralytics_eval \
        [--gt-root datasets/MOT17/train] \
        [--detector SDP]
"""
# status: diagnostic

import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore

import motmetrics as mm


_METRICS = [
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "mota",
    "motp",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "num_fragmentations",
]
_DISPLAY = [
    "IDF1",
    "IDP",
    "IDR",
    "Rcll",
    "Prcn",
    "MOTA",
    "MOTP",
    "IDs",
    "FP",
    "FN",
    "FM",
]
_PCT = {"idf1", "idp", "idr", "recall", "precision", "mota", "motp"}


def is_mot_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("MOT") and name.endswith(".txt")


def load_results(results_dir: str, detector: str | None) -> dict[str, any]:
    files = sorted(
        f for f in glob.glob(os.path.join(results_dir, "*.txt")) if is_mot_file(f)
    )
    if detector:
        files = [f for f in files if f"-{detector}" in Path(f).stem]
    return OrderedDict(
        (
            os.path.splitext(Path(f).name)[0],
            mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1.0),
        )
        for f in files
    )


def load_gt(gt_root: str, names: list[str]) -> dict[str, any]:
    gt = {}
    for name in names:
        # strip detector suffix to find GT folder: MOT17-02-SDP → MOT17-02-SDP
        gt_file = Path(gt_root) / name / "gt" / "gt.txt"
        if not gt_file.exists():
            # try without detector suffix: MOT17-02-SDP result matches MOT17-02-SDP gt
            # gt folder name equals sequence name in MOT17
            continue
        gt[name] = mm.io.loadtxt(str(gt_file), fmt="mot15-2D", min_confidence=1)
    return gt


def evaluate(ts: dict, gt: dict) -> tuple[dict, dict]:
    mh = mm.metrics.create()
    accs, names = [], []
    for name, tsacc in ts.items():
        if name not in gt:
            print(f"  [warn] no GT for {name}, skipping")
            continue
        accs.append(mm.utils.compare_to_groundtruth(gt[name], tsacc, "iou", distth=0.5))
        names.append(name)
    summary = mh.compute_many(
        accs, names=names, metrics=_METRICS, generate_overall=True
    )
    return summary, {n: a for n, a in zip(names, accs)}


def fmt(val, metric: str) -> str:
    if metric in _PCT:
        return f"{val * 100:.1f}%"
    return f"{int(val)}"


def delta_str(a, b, metric: str) -> str:
    if metric in _PCT:
        d = (b - a) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}pp"
    d = int(b) - int(a)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saccade", default="results/MOT17_eval", metavar="DIR")
    parser.add_argument(
        "--ultralytics", default="results/MOT17_ultralytics_eval", metavar="DIR"
    )
    parser.add_argument("--gt-root", default="datasets/MOT17/train")
    parser.add_argument("--detector", choices=["SDP", "DPM", "FRCNN"], default=None)
    parser.add_argument(
        "--metric",
        choices=_METRICS,
        default=None,
        help="Show only this metric in the per-sequence table.",
    )
    args = parser.parse_args()

    label_a = f"Saccade{f'/{args.detector}' if args.detector else ''}"
    label_b = f"Ultralytics{f'/{args.detector}' if args.detector else ''}"

    print(f"Loading {label_a} from {args.saccade} ...")
    ts_a = load_results(args.saccade, args.detector)
    print(f"Loading {label_b} from {args.ultralytics} ...")
    ts_b = load_results(args.ultralytics, args.detector)

    all_names = sorted(set(ts_a) | set(ts_b))
    if not all_names:
        print("No result files found. Check --saccade / --ultralytics paths.")
        return

    print(f"Loading GT from {args.gt_root} ...")
    gt = load_gt(args.gt_root, all_names)
    if not gt:
        print("No GT loaded. Check --gt-root.")
        return

    print(f"\nEvaluating {label_a} ...")
    summary_a, _ = evaluate(ts_a, gt)
    print(f"Evaluating {label_b} ...")
    summary_b, _ = evaluate(ts_b, gt)

    # --- OVERALL table ---
    cw = 10
    header = f"{'Metric':<6} {label_a:>{cw}} {label_b:>{cw}} {'Delta':>8}"
    print(f"\n{label_a} vs {label_b} — OVERALL")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for metric, display in zip(_METRICS, _DISPLAY):
        va = summary_a.loc["OVERALL", metric]
        vb = summary_b.loc["OVERALL", metric]
        print(
            f"{display:<6} {fmt(va, metric):>{cw}} {fmt(vb, metric):>{cw}} {delta_str(va, vb, metric):>8}"
        )

    # --- Per-sequence table (key metrics) ---
    show_metrics = (
        [args.metric]
        if args.metric
        else ["mota", "idf1", "recall", "precision", "num_switches"]
    )
    show_display = [_DISPLAY[_METRICS.index(m)] for m in show_metrics]

    common_names = sorted(set(summary_a.index) & set(summary_b.index) - {"OVERALL"})
    if common_names:
        sc, dc = 7, 6
        hdr = f"{'Sequence':<18}"
        for d in show_display:
            hdr += f" {d:>{sc}} {d[:sc]:>{sc}} {'Δ':>{dc}}"
        # replace display headers properly
        hdr = f"{'Sequence':<18}"
        for d in show_display:
            hdr += f" {'Sac':>{sc}} {'Ult':>{sc}} {'Δ':>{dc}}"
        sub = f"{'':18}"
        for d in show_display:
            sub += f" {d:>{sc}} {d:>{sc}} {'':>{dc}}"
        print(f"\nPer-sequence — {', '.join(show_display)}")
        print("-" * len(hdr))
        print(hdr)
        print(sub)
        print("-" * len(hdr))
        for name in common_names:
            row = f"{name:<18}"
            for metric in show_metrics:
                va = summary_a.loc[name, metric]
                vb = summary_b.loc[name, metric]
                row += f" {fmt(va, metric):>{sc}} {fmt(vb, metric):>{sc}} {delta_str(va, vb, metric):>{dc}}"
            print(row)

    print(
        f"\nDone. Sequences evaluated: Saccade={len(ts_a)}, Ultralytics={len(ts_b)}, GT={len(gt)}"
    )


if __name__ == "__main__":
    main()
