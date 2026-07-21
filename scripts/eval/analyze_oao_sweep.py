#!/usr/bin/env python3
"""OAO (Occlusion-Aware Object) τ sweep analysis: metrics trend + per-sequence breakdown.

Inputs:
  scripts/eval/output/ablation_mot17/baseline/          → τ=0
  scripts/eval/output/ablation_mot17/association/oao_tau{01..04}/
Each has:
  metrics.json           → aggregate MOTA/IDF1/IDs/FP/FN
  MOT17-{XX}-SDP.txt    → per-sequence tracker output

Outputs:
  stdout tables + matplotlib charts (saved as .png in --output dir)
"""
# status: experiment

import argparse
import json
from pathlib import Path

import numpy as np


# NumPy 2.0 compat for motmetrics
np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm

SEQUENCES = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

TAU_EXPERIMENTS = {
    "τ=0.0 (baseline)": "baseline",
    "τ=0.1": "association/oao_tau01",
    "τ=0.2": "association/oao_tau02",
    "τ=0.3": "association/oao_tau03",
    "τ=0.4": "association/oao_tau04",
}

ABLATION_DIR = Path("scripts/eval/output/ablation_mot17")
GT_ROOT = Path("datasets/MOT17/train")


def load_aggregated_metrics(exp_path: Path) -> dict:
    with open(exp_path / "metrics.json") as f:
        return json.load(f)


def compute_per_sequence_metrics(result_dir: Path, seq: str, gt_path: Path) -> dict:
    """Compute motmetrics per-sequence from MOT output file."""
    result_file = result_dir / f"{seq}.txt"
    if not result_file.exists():
        return None

    gt_file = gt_path / seq / "gt" / "gt.txt"
    if not gt_file.exists():
        # try without detector suffix
        base = seq.rsplit("-", 1)[0]  # MOT17-02
        gt_file = gt_path / f"{base}-SDP" / "gt" / "gt.txt"
    if not gt_file.exists():
        return None

    gt = mm.io.loadtxt(str(gt_file), fmt="mot15-2D", min_confidence=1)
    ts = mm.io.loadtxt(str(result_file), fmt="mot15-2D", min_confidence=-1.0)

    acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, metrics=mm.metrics.motchallenge_metrics + ["num_objects"], name=seq
    )
    row = summary.iloc[0]
    return {
        "idf1": float(row.get("idf1", 0)),
        "mota": float(row.get("mota", 0)),
        "recall": float(row.get("recall", 0)) / 100.0,
        "precision": float(row.get("precision", 0)) / 100.0,
        "num_switches": int(row.get("num_switches", 0)),
        "num_false_positives": int(row.get("num_false_positives", 0)),
        "num_misses": int(row.get("num_misses", 0)),
        "num_objects": int(row.get("num_objects", 0)),
    }


def collect_all_metrics():
    data = {}
    for label, exp_rel in TAU_EXPERIMENTS.items():
        exp_dir = ABLATION_DIR / exp_rel
        agg = load_aggregated_metrics(exp_dir)
        per_seq = {}
        for seq in SEQUENCES:
            m = compute_per_sequence_metrics(exp_dir, seq, GT_ROOT)
            if m:
                per_seq[seq] = m
        data[label] = {"aggregate": agg, "per_sequence": per_seq}
    return data


def print_aggregate_table(data: dict):
    print("=" * 102)
    print("Aggregate metrics across OAO τ sweep (all 7 sequences)")
    print("=" * 102)
    header = f"{'Experiment':<22} {'MOTA':>7} {'IDF1':>7} {'Prec':>7} {'Recall':>7} {'IDs':>6} {'FP':>7} {'FN':>7}"
    print(header)
    print("-" * 102)

    baseline = data["τ=0.0 (baseline)"]["aggregate"]
    for label in TAU_EXPERIMENTS:
        m = data[label]["aggregate"]
        mota_d = m["mota"] - baseline["mota"]
        idf1_d = m["idf1"] - baseline["idf1"]
        print(
            f"{label:<22} {m['mota']:.4f} {m['idf1']:.4f} {m['precision']:.4f} "
            f"{m['recall']:.4f} {m['num_switches']:>6} {m['num_false_positives']:>7} "
            f"{m['num_misses']:>7}  (ΔMOTA={mota_d:+.4f} ΔIDF1={idf1_d:+.4f})"
        )
    print()

    # Tradeoff analysis
    print("Tradeoff analysis (vs baseline):")
    print("-" * 60)
    for label in TAU_EXPERIMENTS:
        if "baseline" in label:
            continue
        m = data[label]["aggregate"]
        b = baseline
        fp_delta = m["num_false_positives"] - b["num_false_positives"]
        fn_delta = m["num_misses"] - b["num_misses"]
        ids_delta = m["num_switches"] - b["num_switches"]
        cost_per_fp = fn_delta / abs(fp_delta) if fp_delta != 0 else float("inf")
        print(
            f"  {label}: FP {fp_delta:+d}  FN {fn_delta:+d}  IDs {ids_delta:+d}  "
            f"(FN cost per FP saved = {cost_per_fp:.2f})"
        )


def print_per_sequence_table(data: dict):
    print("\n" + "=" * 120)
    print("Per-sequence MOTA breakdown")
    print("=" * 120)
    seqs = SEQUENCES
    labels = list(TAU_EXPERIMENTS.keys())

    header = f"{'Sequence':<18}"
    for lbl in labels:
        short = lbl.split("(")[0].strip() if "(" in lbl else lbl
        header += f" {short:>10}"
    header += "  Best τ"
    print(header)
    print("-" * 120)

    for seq in seqs:
        row = f"{seq:<18}"
        best_tau = None
        best_mota = -1
        for label in labels:
            m = data[label]["per_sequence"].get(seq, {})
            mota = m.get("mota", 0)
            row += f" {mota:.4f}   "
            if mota > best_mota:
                best_mota = mota
                best_tau = label
        row += f"  {best_tau}"
        print(row)

    print("\nPer-sequence IDF1 breakdown:")
    print("-" * 120)
    header = f"{'Sequence':<18}"
    for lbl in labels:
        short = lbl.split("(")[0].strip() if "(" in lbl else lbl
        header += f" {short:>10}"
    print(header)
    for seq in seqs:
        row = f"{seq:<18}"
        for label in labels:
            m = data[label]["per_sequence"].get(seq, {})
            row += f" {m.get('idf1', 0):.4f}   "
        print(row)

    print("\nPer-sequence IDs (switches) breakdown:")
    print("-" * 120)
    hdr = f"{'Sequence':<18}"
    for lbl in labels:
        short = lbl.split("(")[0].strip() if "(" in lbl else lbl
        hdr += f" {short:>10}"
    print(hdr)
    for seq in seqs:
        row = f"{seq:<18}"
        for label in labels:
            m = data[label]["per_sequence"].get(seq, {})
            row += f" {m.get('num_switches', 0):>10}"
        print(row)

    print("\nPer-sequence FP breakdown:")
    print("-" * 120)
    for seq in seqs:
        row = f"{seq:<18}"
        for label in labels:
            m = data[label]["per_sequence"].get(seq, {})
            row += f" {m.get('num_false_positives', 0):>10}"
        print(row)

    print("\nPer-sequence FN breakdown:")
    print("-" * 120)
    for seq in seqs:
        row = f"{seq:<18}"
        for label in labels:
            m = data[label]["per_sequence"].get(seq, {})
            row += f" {m.get('num_misses', 0):>10}"
        print(row)


def interpret(data: dict):
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)

    baseline = data["τ=0.0 (baseline)"]["aggregate"]
    best_mota_label = max(
        [lbl for lbl in TAU_EXPERIMENTS if lbl != "τ=0.0 (baseline)"],
        key=lambda lbl: data[lbl]["aggregate"]["mota"],
    )
    best_mota = data[best_mota_label]["aggregate"]
    fp_saved = baseline["num_false_positives"] - best_mota["num_false_positives"]
    fn_cost = best_mota["num_misses"] - baseline["num_misses"]
    ids_cost = best_mota["num_switches"] - baseline["num_switches"]

    print(
        f"Best MOTA: {best_mota_label} (MOTA={best_mota['mota']:.4f} vs baseline {baseline['mota']:.4f})"
    )
    print(
        f"  FP reduction: {fp_saved} ({fp_saved / baseline['num_false_positives'] * 100:.1f}%)"
    )
    print(f"  FN increase:  {fn_cost} ({fn_cost / baseline['num_misses'] * 100:.1f}%)")
    print(f"  IDs change:   {ids_cost:+d}")
    print(f"  Precision:    {baseline['precision']:.4f} → {best_mota['precision']:.4f}")
    print(f"  Recall:       {baseline['recall']:.4f} → {best_mota['recall']:.4f}")

    tau_labels = ["τ=0.0 (baseline)", "τ=0.1", "τ=0.2", "τ=0.3", "τ=0.4"]
    tau_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
    fp_vals = [data[lbl]["aggregate"]["num_false_positives"] for lbl in tau_labels]
    fn_vals = [data[lbl]["aggregate"]["num_misses"] for lbl in tau_labels]
    ids_vals = [data[lbl]["aggregate"]["num_switches"] for lbl in tau_labels]

    fp_mono = all(fp_vals[i] >= fp_vals[i + 1] for i in range(len(fp_vals) - 1))
    fn_mono = all(fn_vals[i] <= fn_vals[i + 1] for i in range(len(fn_vals) - 1))
    print("\nMonotonicity check:")
    print(f"  FP decreasing: {fp_mono}")
    print(f"  FN increasing: {fn_mono}")
    print(f"  IDs trend:     {ids_vals}")

    # Find knee: where marginal FP reduction no longer worth marginal FN increase
    print("\nMarginal analysis:")
    for i in range(1, len(tau_vals)):
        fp_d = fp_vals[i] - fp_vals[i - 1]
        fn_d = fn_vals[i] - fn_vals[i - 1]
        ids_d = ids_vals[i] - ids_vals[i - 1]
        ratio = fn_d / abs(fp_d) if fp_d != 0 else float("inf")
        print(
            f"  τ={tau_vals[i - 1]:.1f}→{tau_vals[i]:.1f}: FP {fp_d:+d}, FN {fn_d:+d}, IDs {ids_d:+d} (FN/FP={ratio:.2f})"
        )


def plot(data: dict, output_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[skip] matplotlib not available, plotting disabled")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    tau_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
    labels_all = ["τ=0.0 (baseline)", "τ=0.1", "τ=0.2", "τ=0.3", "τ=0.4"]

    def get(k):
        return [data[lbl]["aggregate"][k] for lbl in labels_all]

    # Chart 1: MOTA & IDF1 trend
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(tau_vals, get("mota"), "o-", color="tab:blue", linewidth=2, label="MOTA")
    ax1.plot(
        tau_vals, get("idf1"), "s--", color="tab:orange", linewidth=2, label="IDF1"
    )
    ax1.set_xlabel("OAO τ")
    ax1.set_ylabel("Score")
    ax1.set_title("MOTA / IDF1 vs OAO τ")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "oao_mota_idf1.png", dpi=120)
    plt.close()

    # Chart 2: FP vs FN tradeoff (dual axis)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1, color2 = "tab:red", "tab:green"
    ax1.plot(
        tau_vals,
        get("num_false_positives"),
        "o-",
        color=color1,
        linewidth=2,
        label="FP",
    )
    ax1.set_xlabel("OAO τ")
    ax1.set_ylabel("False Positives", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.plot(tau_vals, get("num_misses"), "s--", color=color2, linewidth=2, label="FN")
    ax2.set_ylabel("False Negatives (MISS)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("FP ↓ vs FN ↑ tradeoff as OAO τ increases")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "oao_fp_fn_tradeoff.png", dpi=120)
    plt.close()

    # Chart 3: Per-sequence MOTA waterfall (baseline vs best-τ=0.3)
    best_idx = 3
    best_label = labels_all[best_idx]
    base_label = labels_all[0]
    seqs = SEQUENCES
    base_mota = [
        data[base_label]["per_sequence"].get(s, {}).get("mota", 0) for s in seqs
    ]
    best_mota = [
        data[best_label]["per_sequence"].get(s, {}).get("mota", 0) for s in seqs
    ]
    delta = [b - a for a, b in zip(base_mota, best_mota)]

    x = np.arange(len(seqs))
    width = 0.35
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True
    )
    ax_top.bar(
        x - width / 2,
        base_mota,
        width,
        label="τ=0.0 (baseline)",
        color="tab:blue",
        alpha=0.8,
    )
    ax_top.bar(
        x + width / 2, best_mota, width, label=best_label, color="tab:orange", alpha=0.8
    )
    ax_top.set_ylabel("MOTA")
    ax_top.set_title(f"Per-sequence MOTA: baseline vs {best_label}")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)
    colors = ["tab:green" if d >= 0 else "tab:red" for d in delta]
    ax_bot.bar(x, delta, width, color=colors, alpha=0.8)
    ax_bot.set_ylabel("ΔMOTA")
    ax_bot.axhline(y=0, color="black", linewidth=0.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([s.replace("MOT17-", "").replace("-SDP", "") for s in seqs])
    plt.tight_layout()
    plt.savefig(output_dir / "oao_per_sequence_mota.png", dpi=120)
    plt.close()

    # Chart 4: IDs trend
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tau_vals, get("num_switches"), "o-", color="tab:purple", linewidth=2)
    ax.set_xlabel("OAO τ")
    ax.set_ylabel("Identity Switches")
    ax.set_title("ID Switches vs OAO τ")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "oao_ids.png", dpi=120)
    plt.close()

    print(f"\nCharts saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="OAO τ sweep analysis")
    parser.add_argument(
        "--output",
        default="scripts/eval/output/oao_analysis",
        help="Output directory for charts",
    )
    parser.add_argument(
        "--text-only", action="store_true", help="Skip chart generation"
    )
    args = parser.parse_args()

    print("Collecting metrics...")
    data = collect_all_metrics()

    print_aggregate_table(data)
    print_per_sequence_table(data)
    interpret(data)

    if not args.text_only:
        plot(data, Path(args.output))


if __name__ == "__main__":
    main()
