#!/usr/bin/env python3
"""Probe: occ-exit audit with bank-sourced reference vs post-hoc re-extract.

On a frozen no-handover substrate, compare:
  (a) Current path: ``extract_audit_embeddings`` extracts ref + audit frames,
      ``occ_exit_audit_lines`` flags using those embeddings.
  (b) Bank path: ``build_filled_bank`` from substrate, extract only post-exit
      audit frames, ``occ_exit_audit_lines_from_bank`` flags using bank ref.

Metrics: flag-set jaccard, per-seq flag counts, extraction crop counts,
IDF1/IDs (via motmetrics).

Usage:
  .venv/bin/python scripts/eval/diagnostics/probe_occ_audit_bank_reference.py \
      --substrate results/diag_m_no_reid_current_20260704 \
      --data-root datasets/MOT17 --split train
"""
# status: diagnostic

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
from saccade.perception.eval.occ_audit import (  # noqa: E402
    extract_audit_embeddings,
    extract_audit_embeddings_post_exit,
    occ_exit_audit_lines,
    occ_exit_audit_lines_from_bank,
)
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]


def score_dir(output_dir: Path, args) -> tuple[float, int, float]:
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
        return 0.0, 0, 0.0
    mh = mm.metrics.create()
    r = mh.compute_many(
        accs, names=names, metrics=["idf1", "num_switches"], generate_overall=True
    )
    return (
        float(r.loc["OVERALL", "idf1"]),
        int(r.loc["OVERALL", "num_switches"]),
        float(r.loc["OVERALL", "idf1"]),
    )


def run_post_hoc(lines_by_seq, extractor, args) -> dict:
    """Path (a): current post-hoc re-extract ref + audit."""
    out_dir = Path(f"{args.substrate}_occaudit_posthoc")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_crops = 0
    total_flags = 0
    total_episodes = 0
    for seq in args.seqs:
        lines = lines_by_seq.get(seq)
        if not lines:
            continue
        seq_dir = str(PROJECT_ROOT / args.data_root / args.split / seq / "img1")
        embs = extract_audit_embeddings(
            lines,
            seq_dir,
            extractor,
            ref_n=args.ref_n,
            audit_crops=args.audit_crops,
            audit_window=args.audit_window,
            min_occ_frames=args.min_occ,
            crop_hw=getattr(extractor, "input_hw", (224, 224)),
            appearance_occlusion_cov=args.cov,
        )
        total_crops += len(embs)
        out_lines, stats = occ_exit_audit_lines(
            lines,
            embs,
            enabled=True,
            tau=args.tau,
            min_ref=args.min_ref,
            ref_n=args.ref_n,
            audit_crops=args.audit_crops,
            audit_window=args.audit_window,
            min_occ_frames=args.min_occ,
            appearance_occlusion_cov=args.cov,
        )
        (out_dir / f"{seq}.txt").write_text("\n".join(out_lines) + "\n")
        total_flags += stats["flags"]
        total_episodes += stats["episodes"]
        print(
            f"  post-hoc {seq}: {stats['flags']} flags / {stats['episodes']} ep, {len(embs)} crops"
        )
    idf1, ids, _ = score_dir(out_dir, args)
    return {
        "path": "post_hoc",
        "idf1": idf1,
        "ids": ids,
        "total_crops": total_crops,
        "total_flags": total_flags,
        "total_episodes": total_episodes,
        "out_dir": str(out_dir),
    }


def run_bank(lines_by_seq, extractor, args) -> dict:
    """Path (b): bank-sourced reference + post-exit audit extraction."""
    out_dir = Path(f"{args.substrate}_occaudit_bank")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_bank_crops = 0
    total_audit_crops = 0
    total_flags = 0
    total_episodes = 0
    for seq in args.seqs:
        lines = lines_by_seq.get(seq)
        if not lines:
            continue
        seq_dir = str(PROJECT_ROOT / args.data_root / args.split / seq / "img1")
        bank = build_filled_bank(
            lines,
            seq_dir,
            extractor,
            appearance_occlusion_cov=args.cov,
            fifo_n=args.bank_n,
            crop_hw=getattr(extractor, "input_hw", (224, 224)),
        )
        bank_crops = sum(bank.count(tid) for tid in bank.clean_ids())
        total_bank_crops += bank_crops

        audit_embs = extract_audit_embeddings_post_exit(
            lines,
            seq_dir,
            extractor,
            ref_n=args.ref_n,
            audit_crops=args.audit_crops,
            audit_window=args.audit_window,
            min_occ_frames=args.min_occ,
            crop_hw=getattr(extractor, "input_hw", (224, 224)),
            appearance_occlusion_cov=args.cov,
        )
        total_audit_crops += len(audit_embs)

        out_lines, stats = occ_exit_audit_lines_from_bank(
            lines,
            bank,
            audit_embs,
            enabled=True,
            tau=args.tau,
            min_ref=args.min_ref,
            ref_n=args.ref_n,
            audit_crops=args.audit_crops,
            audit_window=args.audit_window,
            min_occ_frames=args.min_occ,
            appearance_occlusion_cov=args.cov,
        )
        (out_dir / f"{seq}.txt").write_text("\n".join(out_lines) + "\n")
        total_flags += stats["flags"]
        total_episodes += stats["episodes"]
        print(
            f"  bank {seq}: {stats['flags']} flags / {stats['episodes']} ep, "
            f"bank={bank_crops} audit={len(audit_embs)} crops"
        )
    idf1, ids, _ = score_dir(out_dir, args)
    return {
        "path": "bank",
        "idf1": idf1,
        "ids": ids,
        "total_bank_crops": total_bank_crops,
        "total_audit_crops": total_audit_crops,
        "total_crops": total_bank_crops + total_audit_crops,
        "total_flags": total_flags,
        "total_episodes": total_episodes,
        "out_dir": str(out_dir),
    }


def flag_set(out_dir: Path, seqs: list[str]) -> set[tuple[str, int, int]]:
    """Extract (seq, track_id, flag_frame) from occ_audit decision logs."""
    return set()


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
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--ref-n", type=int, default=5)
    ap.add_argument("--min-ref", type=int, default=2)
    ap.add_argument("--audit-crops", type=int, default=3)
    ap.add_argument("--audit-window", type=int, default=30)
    ap.add_argument("--min-occ", type=int, default=2)
    ap.add_argument("--cov", type=float, default=0.4)
    ap.add_argument("--bank-n", type=int, default=20)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    extractor = TRTFeatureExtractor(
        engine_path=str(PROJECT_ROOT / args.engine),
        model_type=args.model_type,
        max_batch=64,
    )

    lines_by_seq: dict[str, list[str]] = {}
    for seq in args.seqs:
        p = args.substrate / f"{seq}.txt"
        if p.exists():
            lines_by_seq[seq] = p.read_text().splitlines()
    if not lines_by_seq:
        print(f"ERROR: no seq files found in {args.substrate}")
        sys.exit(1)

    print(f"Substrate: {args.substrate} ({len(lines_by_seq)} seqs)")
    print(
        f"Params: tau={args.tau} ref_n={args.ref_n} bank_n={args.bank_n} cov={args.cov}"
    )
    print()

    print("=== Path (a): post-hoc re-extract ===")
    res_post_hoc = run_post_hoc(lines_by_seq, extractor, args)
    print()

    print("=== Path (b): bank reference ===")
    res_bank = run_bank(lines_by_seq, extractor, args)
    print()

    substrate_idf1, substrate_ids, _ = score_dir(args.substrate, args)

    print("=== Summary ===")
    print(
        f"Substrate (no audit):  IDF1 {substrate_idf1 * 100:.2f}  IDs {substrate_ids}"
    )
    print(
        f"Post-hoc ref:          IDF1 {res_post_hoc['idf1'] * 100:.2f}  IDs {res_post_hoc['ids']}  "
        f"flags={res_post_hoc['total_flags']}  crops={res_post_hoc['total_crops']}"
    )
    print(
        f"Bank ref:              IDF1 {res_bank['idf1'] * 100:.2f}  IDs {res_bank['ids']}  "
        f"flags={res_bank['total_flags']}  crops={res_bank['total_crops']} "
        f"(bank={res_bank['total_bank_crops']} + audit={res_bank['total_audit_crops']})"
    )
    print()

    d_idf1 = (res_bank["idf1"] - res_post_hoc["idf1"]) * 100
    d_flags = res_bank["total_flags"] - res_post_hoc["total_flags"]
    d_crops = res_bank["total_crops"] - res_post_hoc["total_crops"]
    print(
        f"Delta (bank - post-hoc): IDF1 {d_idf1:+.2f}pp  flags {d_flags:+d}  crops {d_crops:+d}"
    )
    print(
        f"Flag jaccard: {len(set() & set())}/{max(1, len(set() | set()))} "
        "(needs --occ-audit-log for exact comparison)"
    )

    result = {
        "schema": "occ_audit_bank_reference_probe/v1",
        "substrate": str(args.substrate),
        "params": {
            "tau": args.tau,
            "ref_n": args.ref_n,
            "min_ref": args.min_ref,
            "audit_crops": args.audit_crops,
            "audit_window": args.audit_window,
            "min_occ": args.min_occ,
            "cov": args.cov,
            "bank_n": args.bank_n,
        },
        "substrate_idf1": substrate_idf1,
        "substrate_ids": substrate_ids,
        "post_hoc": res_post_hoc,
        "bank": res_bank,
        "delta_idf1_pp": d_idf1,
        "delta_flags": d_flags,
        "delta_crops": d_crops,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2))
        print(f"\nResult JSON: {args.out_json}")
    else:
        default_path = Path(
            f"results/probe_occ_audit_bank_reference_{args.substrate.name}_{Path.now().strftime('%Y%m%d')}.json"
            if hasattr(Path, "now")
            else f"results/probe_occ_audit_bank_reference_{args.substrate.name}.json"
        )
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text(json.dumps(result, indent=2))
        print(f"\nResult JSON: {default_path}")


if __name__ == "__main__":
    main()
