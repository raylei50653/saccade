#!/usr/bin/env python3
"""WP3: frozen-substrate occ-exit control/treatment + promotion decision inputs.

RESEARCH only. Does **not** enable a sequence gate or change production defaults.

Control = frozen substrate MOT txts (no audit).
Treatment = bank-reference occ-exit audit on the same lines with:
  - cosine flag_frame / cuts (existing production-path audit)
  - ``chebgr_probe=True`` decision-log columns (log-only; does not change cuts)

Outputs under ``--out-dir`` (default ``results/occ_exit_p55_wp3``):
  - treatment/  (MOT txts + ``_occ_audit.csv``)
  - occ_audit_metrics.json
  - occ_audit_seq_applicability.{json,md}
  - wp3_summary.json

Usage:
  .venv/bin/python scripts/eval/diagnostics/run_occ_audit_wp3_promotion.py \\
      --substrate results/diag_m_no_reid_current_20260704 \\
      --out-dir results/occ_exit_p55_wp3
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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

from saccade.perception.eval.clean_fifo_bank import build_filled_bank  # noqa: E402
from saccade.perception.eval.occ_audit import (  # noqa: E402
    extract_audit_embeddings_post_exit,
    occ_exit_audit_lines_from_bank,
)
from saccade.perception.eval.occ_audit_seq_conditioning import (  # noqa: E402
    Thresholds,
    aggregate_occ_audit_rows,
    attach_metrics,
    build_applicability_table,
    decide_promotion,
    render_applicability_md,
    rollup_by_seq_type,
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
SCHEMA = "occ_exit_audit_wp3_promotion/v1"


def score_dir(output_dir: Path, seqs: list[str], data_root: Path, split: str):
    accs, names = [], []
    for seq in seqs:
        gt_path = data_root / split / seq / "gt" / "gt.txt"
        hyp_path = output_dir / f"{seq}.txt"
        if not hyp_path.is_file():
            continue
        gt_df = mm.io.loadtxt(str(gt_path), fmt="mot15-2D", min_confidence=1)
        hyp_df = mm.io.loadtxt(str(hyp_path), fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt_df, hyp_df, "iou", distth=0.5))
        names.append(seq)
    if not accs:
        raise SystemExit(f"no scored sequences under {output_dir}")
    mh = mm.metrics.create()
    return mh.compute_many(
        accs, names=names, metrics=KEY_METRICS, generate_overall=True
    )


def build_metrics_json(
    control, treatment, seqs: list[str]
) -> dict[str, dict[str, float | int]]:
    """WP2-compatible metrics map: per-seq control/treatment + deltas."""
    out: dict[str, dict[str, float | int]] = {}
    for seq in seqs:
        if seq not in control.index or seq not in treatment.index:
            continue
        idf1_c = float(control.loc[seq, "idf1"]) * 100.0
        idf1_t = float(treatment.loc[seq, "idf1"]) * 100.0
        ids_c = int(control.loc[seq, "num_switches"])
        ids_t = int(treatment.loc[seq, "num_switches"])
        out[seq] = {
            "idf1_control": idf1_c,
            "idf1_treatment": idf1_t,
            "ids_control": ids_c,
            "ids_treatment": ids_t,
            "idf1_delta": idf1_t - idf1_c,
            "ids_delta": ids_t - ids_c,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--substrate",
        type=Path,
        default=Path("results/diag_m_no_reid_current_20260704"),
        help="Frozen no-audit MOT txt directory (control).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/occ_exit_p55_wp3"),
        help="WP3 artifact root.",
    )
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--engine",
        default="models/embedding/mobilenetv4_reid_visclean_224.engine",
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
    ap.add_argument("--chebgr-max-cost", type=float, default=0.45)
    ap.add_argument("--chebgr-margin", type=float, default=0.0)
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing treatment/_occ_audit.csv + MOT txts (score only).",
    )
    args = ap.parse_args(argv)

    substrate = (
        args.substrate
        if args.substrate.is_absolute()
        else PROJECT_ROOT / args.substrate
    )
    out_dir = (
        args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    )
    treatment_dir = out_dir / "treatment"
    data_root = PROJECT_ROOT / args.data_root
    out_dir.mkdir(parents=True, exist_ok=True)
    treatment_dir.mkdir(parents=True, exist_ok=True)

    seqs = [s for s in args.seqs if (substrate / f"{s}.txt").is_file()]
    if not seqs:
        raise SystemExit(f"no MOT txts in {substrate}")

    print(f"control substrate: {substrate}")
    print(f"treatment out:     {treatment_dir}")
    print(f"seqs: {seqs}")

    log_rows: list[dict[str, Any]] = []
    if not args.skip_extract:
        extractor = TRTFeatureExtractor(
            engine_path=str(PROJECT_ROOT / args.engine),
            model_type=args.model_type,
            max_batch=64,
        )
        for seq in seqs:
            lines = (substrate / f"{seq}.txt").read_text().splitlines()
            seq_img = str(data_root / args.split / seq / "img1")
            print(f"\n=== {seq}: bank + post-exit extract ===")
            bank = build_filled_bank(
                lines,
                seq_img,
                extractor,
                appearance_occlusion_cov=args.cov,
                fifo_n=args.bank_n,
                crop_hw=getattr(extractor, "input_hw", (224, 224)),
            )
            audit_embs = extract_audit_embeddings_post_exit(
                lines,
                seq_img,
                extractor,
                ref_n=args.ref_n,
                audit_crops=args.audit_crops,
                audit_window=args.audit_window,
                min_occ_frames=args.min_occ,
                crop_hw=getattr(extractor, "input_hw", (224, 224)),
                appearance_occlusion_cov=args.cov,
            )
            rows: list[dict[str, Any]] = []
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
                decision_log=rows,
                chebgr_probe=True,
                chebgr_max_cost=args.chebgr_max_cost,
                chebgr_margin=args.chebgr_margin,
            )
            (treatment_dir / f"{seq}.txt").write_text("\n".join(out_lines) + "\n")
            log_rows.extend({"seq": seq, **r} for r in rows)
            print(
                f"  flags={stats['flags']} audited={stats['audited']} "
                f"episodes={stats['episodes']} "
                f"ids {stats['ids_before']}->{stats['ids_after']} "
                f"log_rows={len(rows)}"
            )
        if log_rows:
            with (treatment_dir / "_occ_audit.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
                w.writeheader()
                w.writerows(log_rows)
            print(f"\nwrote {treatment_dir / '_occ_audit.csv'} ({len(log_rows)} rows)")
    else:
        csv_path = treatment_dir / "_occ_audit.csv"
        if not csv_path.is_file():
            raise SystemExit(f"--skip-extract but missing {csv_path}")
        with csv_path.open(newline="") as fh:
            log_rows = list(csv.DictReader(fh))
        print(f"reused {csv_path} ({len(log_rows)} rows)")

    print("\n=== scoring ===")
    control = score_dir(substrate, seqs, data_root, args.split)
    treatment = score_dir(treatment_dir, seqs, data_root, args.split)
    ov = "OVERALL"
    idf1_c = float(control.loc[ov, "idf1"]) * 100.0
    idf1_t = float(treatment.loc[ov, "idf1"]) * 100.0
    ids_c = int(control.loc[ov, "num_switches"])
    ids_t = int(treatment.loc[ov, "num_switches"])
    d_idf1 = idf1_t - idf1_c
    d_ids = ids_t - ids_c
    print(f"control:   IDF1 {idf1_c:.2f}  IDs {ids_c}")
    print(
        f"treatment: IDF1 {idf1_t:.2f}  IDs {ids_t}  (Δ {d_idf1:+.2f}pp / {d_ids:+d})"
    )

    metrics = build_metrics_json(control, treatment, seqs)
    metrics_path = out_dir / "occ_audit_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema": "occ_exit_audit_metrics/v1",
                "control_dir": str(substrate),
                "treatment_dir": str(treatment_dir),
                "overall": {
                    "idf1_control": idf1_c,
                    "idf1_treatment": idf1_t,
                    "ids_control": ids_c,
                    "ids_treatment": ids_t,
                    "idf1_delta": d_idf1,
                    "ids_delta": d_ids,
                },
                "per_sequence": metrics,
                "note": (
                    "Treatment applies cosine occ-exit cuts; Cheb-GR probe columns "
                    "are log-only and do not change flag_frame / rewritten lines."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {metrics_path}")

    by_seq = aggregate_occ_audit_rows(log_rows)
    attach_metrics(by_seq, metrics)
    th = Thresholds()
    table = build_applicability_table(by_seq, th)
    rollup = rollup_by_seq_type(table)
    provenance = {
        "control_dir": str(substrate),
        "treatment_dir": str(treatment_dir),
        "occ_audit_csv": str(treatment_dir / "_occ_audit.csv"),
        "metrics_json": str(metrics_path),
        "n_csv_rows": len(log_rows),
        "n_seq": len(table),
        "tau": args.tau,
        "bank_n": args.bank_n,
        "chebgr_probe": True,
        "chebgr_max_cost": args.chebgr_max_cost,
    }
    app_json = {
        "schema": "occ_exit_audit_seq_conditioning/v1",
        "objective": "RESEARCH+DEBUG",
        "gate_enabled": False,
        "thresholds": {
            "min_audited": th.min_audited,
            "min_useful_flags": th.min_useful_flags,
            "idf1_noise_pp": th.idf1_noise_pp,
            "idf1_harm_pp": th.idf1_harm_pp,
            "ids_material": th.ids_material,
            "chebgr_only_domination": th.chebgr_only_domination,
        },
        "provenance": provenance,
        "per_sequence": table,
        "by_seq_type": rollup,
    }
    app_json_path = out_dir / "occ_audit_seq_applicability.json"
    app_md_path = out_dir / "occ_audit_seq_applicability.md"
    app_json_path.write_text(json.dumps(app_json, indent=2) + "\n")
    app_md_path.write_text(
        render_applicability_md(
            table,
            title="occ-exit WP3 sequence applicability (frozen substrate)",
            thresholds=th,
            provenance=provenance,
        )
    )
    print(f"wrote {app_json_path}")
    print(f"wrote {app_md_path}")

    promotion = decide_promotion(
        table, overall_idf1_delta_pp=d_idf1, overall_ids_delta=d_ids
    )
    # Per-seq rationale strings
    seq_rationale = []
    for row in table:
        seq_rationale.append(
            {
                "seq": row["seq"],
                "recommendation": row["recommendation"],
                "idf1_delta": row["idf1_delta"],
                "ids_delta": row["ids_delta"],
                "audited": row["audited"],
                "cosine_flags": row["cosine_flags"],
                "chebgr_flags": row["chebgr_flags"],
                "flag_delta_same": row["flag_delta_same"],
                "flag_delta_cosine_only": row["flag_delta_cosine_only"],
                "flag_delta_chebgr_only": row["flag_delta_chebgr_only"],
                "note": (
                    f"{row['recommendation']}: ΔIDF1="
                    f"{row['idf1_delta'] if row['idf1_delta'] is not None else 'n/a'} "
                    f"ΔIDs={row['ids_delta'] if row['ids_delta'] is not None else 'n/a'} "
                    f"flags cos={row['cosine_flags']} chebgr={row['chebgr_flags']} "
                    f"delta(same/cos_only/chebgr_only)="
                    f"{row['flag_delta_same']}/{row['flag_delta_cosine_only']}/"
                    f"{row['flag_delta_chebgr_only']}"
                ),
            }
        )

    summary = {
        "schema": SCHEMA,
        "objective": "RESEARCH+DEBUG",
        "gate_implemented": False,
        "params": {
            "substrate": str(substrate),
            "tau": args.tau,
            "ref_n": args.ref_n,
            "min_ref": args.min_ref,
            "audit_crops": args.audit_crops,
            "audit_window": args.audit_window,
            "min_occ": args.min_occ,
            "cov": args.cov,
            "bank_n": args.bank_n,
            "chebgr_probe": True,
            "chebgr_max_cost": args.chebgr_max_cost,
            "chebgr_margin": args.chebgr_margin,
            "engine": args.engine,
            "model_type": args.model_type,
        },
        "aggregate": {
            "idf1_control": idf1_c,
            "idf1_treatment": idf1_t,
            "ids_control": ids_c,
            "ids_treatment": ids_t,
            "idf1_delta_pp": d_idf1,
            "ids_delta": d_ids,
        },
        "promotion": promotion,
        "per_sequence": seq_rationale,
        "artifacts": {
            "metrics_json": str(metrics_path),
            "applicability_json": str(app_json_path),
            "applicability_md": str(app_md_path),
            "treatment_dir": str(treatment_dir),
            "control_dir": str(substrate),
        },
    }
    summary_path = out_dir / "wp3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    print(f"\nPROMOTION DECISION: {promotion['decision']}")
    print(f"  {promotion['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
