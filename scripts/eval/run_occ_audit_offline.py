#!/usr/bin/env python3
"""Offline occ-exit-audit ablation on an existing substrate's output files.

The audit is a pure line-level post-process, so the cleanest A/B applies it
directly to a frozen substrate (identical tracker output, zero eval
nondeterminism) and re-scores both dirs with the same motmetrics call.

Usage
-----
  .venv/bin/python scripts/eval/run_occ_audit_offline.py \
      --substrate results/analysis_m_semantic_delayed_claim_control_20260703 \
      --taus 0.386 0.45 0.498
"""
# status: stable

from __future__ import annotations

import argparse
import csv
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

from saccade.perception.eval.occ_audit import (  # noqa: E402
    extract_audit_embeddings,
    occ_exit_audit_lines,
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


def score(output_dir: Path, args):
    accs, names = [], []
    for seq in args.seqs:
        gt_path = PROJECT_ROOT / args.data_root / args.split / seq / "gt" / "gt.txt"
        gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
        hyp_df = mm.io.loadtxt(
            str(output_dir / f"{seq}.txt"), fmt="mot15-2D", min_confidence=-1.0
        )
        accs.append(mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5))
        names.append(seq)
    mh = mm.metrics.create()
    return mh.compute_many(
        accs,
        names=names,
        metrics=KEY_METRICS,
        generate_overall=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", type=Path, required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.45])
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--engine", default="models/embedding/mobilenetv4_reid_visclean_224.engine"
    )
    ap.add_argument("--model-type", default="mobilenetv4_reid")
    ap.add_argument("--ref-n", type=int, default=5)
    ap.add_argument("--min-ref", type=int, default=2)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--min-occ", type=int, default=2)
    ap.add_argument("--cov", type=float, default=0.4)
    ap.add_argument("--consensus", type=int, default=1)
    ap.add_argument("--self-min", type=float, default=0.0)
    ap.add_argument("--occluder-margin", type=float, default=-1.0)
    args = ap.parse_args()

    extractor = TRTFeatureExtractor(
        engine_path=str(PROJECT_ROOT / args.engine),
        model_type=args.model_type,
        max_batch=64,
    )

    # Embeddings depend only on geometry -> extract once per seq, reuse per tau.
    embs_by_seq: dict[str, dict] = {}
    lines_by_seq: dict[str, list[str]] = {}
    for seq in args.seqs:
        lines = (args.substrate / f"{seq}.txt").read_text().splitlines()
        lines_by_seq[seq] = lines
        embs_by_seq[seq] = extract_audit_embeddings(
            lines,
            str(PROJECT_ROOT / args.data_root / args.split / seq / "img1"),
            extractor,
            ref_n=args.ref_n,
            audit_crops=args.crops,
            audit_window=args.window,
            min_occ_frames=args.min_occ,
            appearance_occlusion_cov=args.cov,
        )
        print(f"{seq}: {len(embs_by_seq[seq])} audit crops extracted")

    runs: list[tuple[str, Path]] = []
    for tau in args.taus:
        suffix = f"_om{args.occluder_margin:g}" if args.occluder_margin >= 0 else ""
        out_dir = Path(f"{args.substrate}_occaudit_tau{tau:g}{suffix}")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_rows: list[dict] = []
        for seq in args.seqs:
            rows: list[dict] = []
            out_lines, stats = occ_exit_audit_lines(
                lines_by_seq[seq],
                embs_by_seq[seq],
                enabled=True,
                tau=tau,
                min_ref=args.min_ref,
                ref_n=args.ref_n,
                audit_crops=args.crops,
                audit_window=args.window,
                min_occ_frames=args.min_occ,
                appearance_occlusion_cov=args.cov,
                flag_consensus=args.consensus,
                self_consistency_min=args.self_min,
                occluder_margin=args.occluder_margin,
                decision_log=rows,
            )
            (out_dir / f"{seq}.txt").write_text("\n".join(out_lines) + "\n")
            log_rows.extend({"seq": seq, **r} for r in rows)
            print(
                f"  tau={tau:g} {seq}: {stats['flags']} flags / "
                f"{stats['audited']} audited ({stats['episodes']} episodes, "
                f"no_ref={stats['abstain_no_ref']} "
                f"no_crops={stats['abstain_no_crops']})"
            )
        if log_rows:
            with (out_dir / "_occ_audit.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
                w.writeheader()
                w.writerows(log_rows)
        runs.append((f"tau{tau:g}", out_dir))

    print("\n=== scoring ===")
    control = score(args.substrate, args)
    variants = {name: score(out_dir, args) for name, out_dir in runs}

    def fmt(v: float, key: str) -> str:
        return (
            f"{v * 100:.2f}"
            if key in ("idf1", "mota", "recall", "precision")
            else f"{int(v)}"
        )

    header = ["metric", "control"] + [name for name, _ in runs]
    print("\n" + "  ".join(f"{h:>12}" for h in header))
    ov = "OVERALL"
    for key in KEY_METRICS:
        row = [key, fmt(control.loc[ov, key], key)] + [
            fmt(variants[name].loc[ov, key], key) for name, _ in runs
        ]
        print("  ".join(f"{c:>12}" for c in row))

    print("\nper-seq IDF1 delta vs control:")
    for seq in args.seqs:
        base = control.loc[seq, "idf1"] * 100
        deltas = "  ".join(
            f"{name}:{variants[name].loc[seq, 'idf1'] * 100 - base:+.2f}"
            for name, _ in runs
        )
        print(f"  {seq}: {base:.2f}  {deltas}")


if __name__ == "__main__":
    main()
