#!/usr/bin/env python3
"""Offline Cheb-GR handover ablation on an existing substrate's output files.

The handover is a pure line-level post-process, so the cleanest A/B applies it
directly to a frozen no-handover substrate (identical tracker output, zero
eval nondeterminism) and re-scores every variant with the same motmetrics
call. Embeddings are extracted once per distinct ``neighbor_iou_max`` value
and reused across variants.

Usage
-----
  .venv/bin/python scripts/eval/run_offline_handover_ablation.py \
      --substrate results/diag_m_no_reid_current_20260704 \
      --variant cdveto2 center_dist_veto=2.0 \
      --variant polveto05 pollution_veto=0.5 \
      --variant nbriou05 neighbor_iou_max=0.5 \
      --variant combined center_dist_veto=2.0 neighbor_iou_max=0.5
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

PROJECT_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm  # noqa: E402
import pandas as pd  # noqa: E402

from saccade.perception.eval.cheb_gr_online import (  # noqa: E402
    causal_handover_lines,
    extract_handover_embeddings,
)
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]
KEY_METRICS = [
    "idf1",
    "mota",
    "recall",
    "precision",
    "num_switches",
    "num_false_positives",
    "num_misses",
]
VARIANT_KEYS = {
    "max_cost": float,
    "margin": float,
    "key_sim_min": float,
    "key_sim_cost_floor": float,
    "key_margin_min": float,
    "min_head": int,
    "center_dist_veto": float,
    "pollution_veto": float,
    "neighbor_iou_max": float,
    "bank_mode": str,
    "bank_n": int,
}


def _mot_lines_to_df(lines: list[str], *, min_confidence: float = -1.0) -> pd.DataFrame:
    text = "\n".join(line for line in lines if line.strip())
    columns = [
        "FrameId",
        "Id",
        "X",
        "Y",
        "Width",
        "Height",
        "Confidence",
        "ClassId",
        "Visibility",
        "unused",
    ]
    if not text:
        empty = pd.DataFrame(columns=columns[:-1])
        empty.index = pd.MultiIndex.from_arrays([[], []], names=["FrameId", "Id"])
        return empty
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"\s+|\t+|,",
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=columns,
        engine="python",
    )
    df[["X", "Y"]] -= (1, 1)
    del df["unused"]
    return df[df["Confidence"] >= min_confidence]


def _load_gt(args) -> dict[str, pd.DataFrame]:
    return {
        seq: mm.io.loadtxt(
            str(PROJECT_ROOT / args.data_root / args.split / seq / "gt" / "gt.txt"),
            fmt="mot15-2D",
            min_confidence=1,
        )
        for seq in args.seqs
    }


def score_lines(lines_by_seq: dict[str, list[str]], args, gt_by_seq):
    accs, names = [], []
    for seq in args.seqs:
        hyp_df = _mot_lines_to_df(lines_by_seq[seq], min_confidence=-1.0)
        accs.append(
            mm.utils.compare_to_groundtruth(gt_by_seq[seq], hyp_df, "iou", distth=0.5)
        )
        names.append(seq)
    mh = mm.metrics.create()
    return mh.compute_many(
        accs,
        names=names,
        metrics=KEY_METRICS,
        generate_overall=True,
    )


def parse_variants(raw: list[list[str]] | None) -> list[tuple[str, dict]]:
    variants: list[tuple[str, dict]] = [("control", {})]
    for spec in raw or []:
        name, kvs = spec[0], spec[1:]
        params: dict = {}
        for kv in kvs:
            key, _, val = kv.partition("=")
            if key not in VARIANT_KEYS:
                raise SystemExit(
                    f"variant {name}: unknown key {key!r} "
                    f"(allowed: {sorted(VARIANT_KEYS)})"
                )
            params[key] = VARIANT_KEYS[key](val)
        variants.append((name, params))
    return variants


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--substrate", type=Path, required=True)
    ap.add_argument(
        "--variant",
        nargs="+",
        action="append",
        metavar=("NAME", "KEY=VAL"),
        help="Variant name followed by param overrides; repeatable. "
        f"Keys: {sorted(VARIANT_KEYS)}. A no-override control always runs.",
    )
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--engine", default="models/embedding/mobilenetv4_reid_visclean_224.engine"
    )
    ap.add_argument("--model-type", default="mobilenetv4_reid")
    # Baseline = current GO operating point (h2 margin0.05 maxcost0.45).
    ap.add_argument("--decide-n", type=int, default=5)
    ap.add_argument("--max-cost", type=float, default=0.45)
    ap.add_argument("--min-head", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max-gap", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--cov", type=float, default=0.4)
    ap.add_argument(
        "--no-write-results",
        action="store_true",
        help="Do not write per-variant MOT txt files; score directly from memory.",
    )
    ap.add_argument(
        "--no-score",
        action="store_true",
        help="Run handover variants and logs only; skip motmetrics scoring.",
    )
    args = ap.parse_args()

    variants = parse_variants(args.variant)

    extractor = TRTFeatureExtractor(
        engine_path=str(PROJECT_ROOT / args.engine),
        model_type=args.model_type,
        max_batch=64,
    )
    crop_hw = tuple(getattr(extractor, "input_hw", (224, 224)))

    lines_by_seq: dict[str, list[str]] = {
        seq: (args.substrate / f"{seq}.txt").read_text().splitlines()
        for seq in args.seqs
    }

    # Embeddings depend only on (substrate lines, decide_n, neighbor_iou_max,
    # bank_mode, bank_n) -> extract once per distinct combo, reuse across
    # variants.
    def _ext_key(params: dict) -> tuple[float, str, int]:
        return (
            params.get("neighbor_iou_max", 0.0),
            params.get("bank_mode", "spread"),
            params.get("bank_n", 0),
        )

    ext_keys = sorted({_ext_key(v) for _, v in variants})
    embs: dict[tuple[tuple[float, str, int], str], tuple[dict, dict]] = {}
    for ext_key in ext_keys:
        nbr_max, bank_mode, bank_n = ext_key
        for seq in args.seqs:
            head_embs, bank_embs = extract_handover_embeddings(
                lines_by_seq[seq],
                str(PROJECT_ROOT / args.data_root / args.split / seq / "img1"),
                extractor,
                decide_n=args.decide_n,
                n_samples=args.n_samples,
                crop_hw=crop_hw,
                appearance_occlusion_cov=args.cov,
                neighbor_iou_max=nbr_max,
                bank_mode=bank_mode,
                bank_n=bank_n,
            )
            embs[(ext_key, seq)] = (head_embs, bank_embs)
            print(
                f"extracted nbr_iou_max={nbr_max:g} bank={bank_mode}/{bank_n} "
                f"{seq}: {len(head_embs)} heads / {len(bank_embs)} banks"
            )

    runs: list[tuple[str, Path]] = []
    variant_lines: dict[str, dict[str, list[str]]] = {}
    for name, params in variants:
        out_dir = Path(f"{args.substrate}_ho_{name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        ext_key = _ext_key(params)
        log_rows: list[dict] = []
        seq_lines: dict[str, list[str]] = {}
        for seq in args.seqs:
            head_embs, bank_embs = embs[(ext_key, seq)]
            rows: list[dict] = []
            out_lines, stats = causal_handover_lines(
                lines_by_seq[seq],
                head_embs,
                bank_embs,
                enabled=True,
                max_cost=params.get("max_cost", args.max_cost),
                max_gap=args.max_gap,
                decide_n=args.decide_n,
                min_head_samples=params.get("min_head", args.min_head),
                margin=params.get("margin", args.margin),
                key_sim_min=params.get("key_sim_min", 0.0),
                key_sim_cost_floor=params.get("key_sim_cost_floor", 0.0),
                key_margin_min=params.get("key_margin_min", 0.0),
                center_dist_veto=params.get("center_dist_veto", 0.0),
                pollution_veto=params.get("pollution_veto", 0.0),
                decision_log=rows,
            )
            seq_lines[seq] = out_lines
            if not args.no_write_results:
                (out_dir / f"{seq}.txt").write_text("\n".join(out_lines) + "\n")
            log_rows.extend({"seq": seq, **r} for r in rows)
            print(
                f"  {name} {seq}: {stats['handovers']} handovers "
                f"(ids {stats['ids_before']}->{stats['ids_after']}, "
                f"reject_cost={stats['reject_cost']} "
                f"reject_margin={stats['reject_margin']} "
                f"reject_key_sim={stats['reject_key_sim']} "
                f"reject_key_margin={stats['reject_key_margin']} "
                f"reject_center_dist={stats['reject_center_dist']} "
                f"reject_pollution={stats['reject_pollution']} "
                f"reject_min_head={stats['reject_min_head']})"
            )
        if log_rows:
            with (out_dir / "_cheb_gr_offline_handover.csv").open(
                "w", newline=""
            ) as fh:
                w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
                w.writeheader()
                w.writerows(log_rows)
        variant_lines[name] = seq_lines
        runs.append((name, out_dir))

    if args.no_score:
        return

    print("\n=== scoring ===")
    gt_by_seq = _load_gt(args)
    substrate = score_lines(lines_by_seq, args, gt_by_seq)
    variants_scored = {
        name: score_lines(variant_lines[name], args, gt_by_seq) for name, _ in runs
    }

    def fmt(v: float, key: str) -> str:
        return (
            f"{v * 100:.2f}"
            if key in ("idf1", "mota", "recall", "precision")
            else f"{int(v)}"
        )

    header = ["metric", "substrate"] + [name for name, _ in runs]
    print("\n" + "  ".join(f"{h:>12}" for h in header))
    ov = "OVERALL"
    for key in KEY_METRICS:
        row = [key, fmt(substrate.loc[ov, key], key)] + [
            fmt(variants_scored[name].loc[ov, key], key) for name, _ in runs
        ]
        print("  ".join(f"{c:>12}" for c in row))

    print("\nper-seq IDF1 delta vs control:")
    control = variants_scored["control"]
    for seq in args.seqs:
        base = control.loc[seq, "idf1"] * 100
        deltas = "  ".join(
            f"{name}:{variants_scored[name].loc[seq, 'idf1'] * 100 - base:+.2f}"
            for name, _ in runs
            if name != "control"
        )
        print(f"  {seq}: {base:.2f}  {deltas}")


if __name__ == "__main__":
    main()
