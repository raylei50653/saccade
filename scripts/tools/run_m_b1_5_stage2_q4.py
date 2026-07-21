#!/usr/bin/env python3
"""M-B1.5 Stage 2 Q4 runner: signal separability on D_online.

Does **not** search thresholds, Boolean rules, or change production presets.

Usage:
  uv run python scripts/tools/run_m_b1_5_stage2_q4.py \\
    --q1q3-study out/signal_study/m_b1_5_stage2_q1q3_20260710 \\
    --out out/signal_study/m_b1_5_stage2_q4_20260710
"""
# status: stable

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from saccade.perception.eval.d_online_stage2_q4 import (
    Q1Q3_STUDY_ID,
    run_stage2_q4_audit,
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--q1q3-study",
        type=Path,
        default=Path(f"out/signal_study/{Q1Q3_STUDY_ID}"),
        help="Q1–Q3 study dir with d_online_events",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output study directory",
    )
    ap.add_argument("--study-id", type=str, default=None)
    args = ap.parse_args(argv)

    if not args.q1q3_study.is_dir():
        print(f"ERROR: q1q3 study not found: {args.q1q3_study}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path(f"out/signal_study/m_b1_5_stage2_q4_{stamp}")
    study_id = args.study_id or out.name

    summary = run_stage2_q4_audit(
        q1q3_study_dir=args.q1q3_study,
        out_dir=out,
        git_commit=_git_commit(),
        study_id=study_id,
    )

    print(f"wrote study → {out}")
    print(f"Q4={summary['stage2_q4_separability']}  ({summary['terminal_letter']})")
    print(
        f"primary: neg={summary['n_primary_negative']}  "
        f"pos_protect={summary['n_primary_positive_protect']}"
    )
    if summary.get("best_feature"):
        print(
            f"best frozen oriented AUC: {summary.get('best_oriented_auc'):.3f} "
            f"({summary.get('best_feature')})"
        )
    print(f"next: {summary['next_authorized_step']}")
    print(f"production_preset: {summary['production_preset']}")

    if summary.get("reconciliation_acceptance") == "FAIL_CLOSED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
