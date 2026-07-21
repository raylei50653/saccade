#!/usr/bin/env python3
"""Probe: online assoc cost via forwarded (copy) clean-FIFO embeddings.

Tests the "copy embedding for graph-external use" finding (probe 2026-07-04,
registry #58): duplicate embeddings are safe for direct cosine assoc cost
but must NOT enter Cheb-GR k-reciprocal graph.

On a frozen substrate, simulate adding an appearance cosine term to the
post-merge assoc cost, using ``CleanFifoBank.representative()`` (mean of
FIFO-20 clean samples) as the per-track appearance vector. Compare:

  (a) Current: post-merge with no appearance cost (pure IoU/Kalman/spatial)
  (b) FIFO appearance: post-merge with appearance_weight on FIFO reps

Metrics: IDF1 / IDs / MOTA delta, accept count diff, per-seq breakdown.

Usage:
  .venv/bin/python scripts/eval/diagnostics/probe_forwarded_embedding_assoc_cost.py \
      --substrate results/diag_m_no_reid_current_20260704 \
      --data-root datasets/MOT17 --split train
"""
# status: stable

from __future__ import annotations

import argparse
import json
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
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore

import motmetrics as mm  # noqa: E402

from saccade.perception.eval.clean_fifo_bank import build_filled_bank  # noqa: E402
from saccade.perception.eval.post_merge import post_merge_output_tracklets  # noqa: E402
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]


def score_dir(output_dir: Path, args) -> dict:
    accs, names = [], []
    for seq in args.seqs:
        gt_path = PROJECT_ROOT / args.data_root / args.split / seq / "gt" / "gt.txt"
        gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
        hyp_path = output_dir / f"{seq}.txt"
        if not hyp_path.exists():
            continue
        hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5))
        names.append(seq)
    if not accs:
        return {"idf1": 0.0, "ids": 0, "mota": 0.0}
    mh = mm.metrics.create()
    r = mh.compute_many(
        accs,
        names=names,
        metrics=["idf1", "num_switches", "mota"],
        generate_overall=True,
    )
    return {
        "idf1": float(r.loc["OVERALL", "idf1"]),
        "ids": int(r.loc["OVERALL", "num_switches"]),
        "mota": float(r.loc["OVERALL", "mota"]),
    }


class FifoAppearanceBank:
    """Adapter that mimics OutputAppearanceBank using CleanFifoBank reps."""

    def __init__(self, fifo_bank):
        self._bank = fifo_bank

    def count(self, track_id: int) -> int:
        return self._bank.count(track_id)

    def consistency(self, track_id: int) -> float:
        s = self._bank.samples(track_id)
        if s is None or s.shape[0] < 2:
            return 1.0
        sim = s @ s.T
        n = s.shape[0]
        return float((sim.sum() - n) / max(1, n * (n - 1)))

    def similarity(self, a: int, b: int) -> float | None:
        ra = self._bank.representative(a)
        rb = self._bank.representative(b)
        if ra is None or rb is None:
            return None
        return float(ra @ rb)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--substrate", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--engine", default="models/embedding/mobilenetv4_reid_visclean_224.engine"
    )
    ap.add_argument("--model-type", default="mobilenetv4_reid")
    ap.add_argument("--fifo-n", type=int, default=20)
    ap.add_argument("--cov", type=float, default=0.4)
    ap.add_argument("--appearance-weight", type=float, default=0.15)
    ap.add_argument("--appearance-threshold", type=float, default=0.0)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    extractor = TRTFeatureExtractor(
        engine_path=str(PROJECT_ROOT / args.engine),
        model_type=args.model_type,
        max_batch=64,
    )

    control_dir = Path(f"{args.substrate}_assoc_control")
    fifo_dir = Path(f"{args.substrate}_assoc_fifo")
    control_dir.mkdir(parents=True, exist_ok=True)
    fifo_dir.mkdir(parents=True, exist_ok=True)

    per_seq = {}

    for seq in args.seqs:
        p = args.substrate / f"{seq}.txt"
        if not p.exists():
            continue
        lines = p.read_text().splitlines()
        seq_dir = str(PROJECT_ROOT / args.data_root / args.split / seq / "img1")

        print(f"\n=== {seq} ===")

        control_out, control_stats = post_merge_output_tracklets(
            lines,
            enabled=True,
            appearance_bank=None,
            appearance_weight=0.0,
        )
        (control_dir / f"{seq}.txt").write_text("\n".join(control_out) + "\n")

        fifo_bank = build_filled_bank(
            lines,
            seq_dir,
            extractor,
            appearance_occlusion_cov=args.cov,
            fifo_n=args.fifo_n,
            crop_hw=getattr(extractor, "input_hw", (224, 224)),
        )
        app_bank = FifoAppearanceBank(fifo_bank)

        fifo_out, fifo_stats = post_merge_output_tracklets(
            lines,
            enabled=True,
            appearance_bank=app_bank,
            appearance_weight=args.appearance_weight,
            appearance_threshold=args.appearance_threshold,
        )
        (fifo_dir / f"{seq}.txt").write_text("\n".join(fifo_out) + "\n")

        print(f"  Control: accepted={control_stats.get('accepted', 0)}")
        print(
            f"  FIFO:    accepted={fifo_stats.get('accepted', 0)} "
            f"reject_app={fifo_stats.get('reject_appearance', 0)} "
            f"reject_app_missing={fifo_stats.get('reject_appearance_missing', 0)}"
        )

        per_seq[seq] = {
            "control_accepted": control_stats.get("accepted", 0),
            "fifo_accepted": fifo_stats.get("accepted", 0),
            "fifo_reject_appearance": fifo_stats.get("reject_appearance", 0),
            "fifo_reject_appearance_missing": fifo_stats.get(
                "reject_appearance_missing", 0
            ),
        }

    control_metrics = score_dir(control_dir, args)
    fifo_metrics = score_dir(fifo_dir, args)
    substrate_metrics = score_dir(args.substrate, args)

    print("\n=== Summary ===")
    print(
        f"Substrate (no merge):  IDF1 {substrate_metrics['idf1'] * 100:.2f}  "
        f"IDs {substrate_metrics['ids']}  MOTA {substrate_metrics['mota'] * 100:.2f}"
    )
    print(
        f"Control (merge, no app): IDF1 {control_metrics['idf1'] * 100:.2f}  "
        f"IDs {control_metrics['ids']}  MOTA {control_metrics['mota'] * 100:.2f}"
    )
    print(
        f"FIFO app (w={args.appearance_weight}):       IDF1 {fifo_metrics['idf1'] * 100:.2f}  "
        f"IDs {fifo_metrics['ids']}  MOTA {fifo_metrics['mota'] * 100:.2f}"
    )

    d_idf1 = (fifo_metrics["idf1"] - control_metrics["idf1"]) * 100
    d_ids = fifo_metrics["ids"] - control_metrics["ids"]
    print(f"\nDelta (FIFO - control): IDF1 {d_idf1:+.2f}pp  IDs {d_ids:+d}")

    result = {
        "schema": "forwarded_embedding_assoc_cost_probe/v1",
        "substrate": str(args.substrate),
        "params": {
            "fifo_n": args.fifo_n,
            "cov": args.cov,
            "appearance_weight": args.appearance_weight,
            "appearance_threshold": args.appearance_threshold,
        },
        "substrate_metrics": substrate_metrics,
        "control_metrics": control_metrics,
        "fifo_metrics": fifo_metrics,
        "delta_idf1_pp": d_idf1,
        "delta_ids": d_ids,
        "per_seq": per_seq,
    }

    out_path = args.out_json or Path(
        f"results/probe_forwarded_embedding_assoc_cost_{args.substrate.name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult JSON: {out_path}")


if __name__ == "__main__":
    main()
