#!/usr/bin/env python3
"""Aggregate occ-exit audit logs into a per-seq / scene-type applicability map.

WP2 (#55): RESEARCH + DEBUG analysis only. Does **not** enable sequence gates
or change production defaults.

Usage:
  .venv/bin/python scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py \\
      --occ-audit-csv results/run/_occ_audit.csv \\
      --metrics-json results/run/occ_audit_metrics.json \\
      --out-json results/run/occ_audit_seq_applicability.json \\
      --out-md results/run/occ_audit_seq_applicability.md

Metrics JSON (optional) maps seq → idf1_delta / ids_delta (treatment − control),
or control/treatment absolute pairs. Without metrics, rows classify as
insufficient_evidence (or harmful if chebgr_only domination is extreme).
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saccade.perception.eval.occ_audit_seq_conditioning import (  # noqa: E402
    Thresholds,
    attach_metrics,
    aggregate_occ_audit_rows,
    build_applicability_table,
    load_metrics_json,
    load_occ_audit_csv,
    render_applicability_md,
    rollup_by_seq_type,
)

SCHEMA = "occ_exit_audit_seq_conditioning/v1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--occ-audit-csv",
        type=Path,
        required=True,
        help="Path to _occ_audit.csv (probe-off or probe-on).",
    )
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional per-seq metric deltas JSON (control vs treatment).",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--min-audited", type=int, default=5)
    ap.add_argument("--min-useful-flags", type=int, default=2)
    ap.add_argument("--idf1-noise-pp", type=float, default=0.15)
    ap.add_argument("--idf1-harm-pp", type=float, default=0.30)
    ap.add_argument("--ids-material", type=int, default=5)
    ap.add_argument("--chebgr-only-domination", type=float, default=0.70)
    args = ap.parse_args(argv)

    th = Thresholds(
        min_audited=args.min_audited,
        min_useful_flags=args.min_useful_flags,
        idf1_noise_pp=args.idf1_noise_pp,
        idf1_harm_pp=args.idf1_harm_pp,
        ids_material=args.ids_material,
        chebgr_only_domination=args.chebgr_only_domination,
    )

    rows = load_occ_audit_csv(args.occ_audit_csv)
    by_seq = aggregate_occ_audit_rows(rows)
    metrics_path = str(args.metrics_json) if args.metrics_json else None
    if args.metrics_json is not None:
        attach_metrics(by_seq, load_metrics_json(args.metrics_json))

    table = build_applicability_table(by_seq, th)
    rollup = rollup_by_seq_type(table)
    provenance = {
        "occ_audit_csv": str(args.occ_audit_csv),
        "metrics_json": metrics_path or "",
        "n_csv_rows": len(rows),
        "n_seq": len(table),
    }
    payload = {
        "schema": SCHEMA,
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
        "notes": [
            "Applicability map only — no sequence gate is activated.",
            "enable_candidate is a research recommendation, not a runtime switch.",
            "WP3 owns promotion / gating decisions.",
        ],
    }

    md = render_applicability_md(
        table,
        title="occ-exit audit sequence conditioning (WP2)",
        thresholds=th,
        provenance=provenance,
    )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out_json}")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"wrote {args.out_md}")

    # Always print a short console summary.
    from collections import Counter

    counts = Counter(r["recommendation"] for r in table)
    print(f"schema={SCHEMA} seqs={len(table)} rows={len(rows)}")
    for c in (
        "enable_candidate",
        "abstain",
        "harmful",
        "insufficient_evidence",
    ):
        print(f"  {c}: {counts.get(c, 0)}")
    if not args.out_md and not args.out_json:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
