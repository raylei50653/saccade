#!/usr/bin/env python3
"""M-B1.5 Stage 2 Q1–Q3 runner: D_online label join + safe-negative mass audit.

Does **not** search thresholds, Boolean rules, or change production presets.

Usage:
  uv run python scripts/tools/run_m_b1_5_stage2_q1q3.py \\
    --stage1-study out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close \\
    --out out/signal_study/m_b1_5_stage2_q1q3_<stamp>

Contract:
  docs/modules/semantic/research/m_b1_5_stage2_entry_contract_20260710.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from saccade.perception.eval.d_online_stage2 import (
    AUTHORITATIVE_STAGE1_STUDY,
    EXPECTED_D_ONLINE_N,
    run_stage2_q1q3_audit,
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
        "--stage1-study",
        type=Path,
        default=Path(f"out/signal_study/{AUTHORITATIVE_STAGE1_STUDY}"),
        help="Stage 1 CLOSED study dir with hook_candidate_events + e2e_A1_hook_off",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output study directory (default: out/signal_study/m_b1_5_stage2_q1q3_<utc>)",
    )
    ap.add_argument(
        "--gt-root",
        type=Path,
        default=Path("datasets/MOT17/train"),
        help="MOT17 train root containing <seq>/gt/gt.txt",
    )
    ap.add_argument(
        "--study-id",
        type=str,
        default=None,
        help="Optional study id override",
    )
    ap.add_argument(
        "--min-join-coverage",
        type=float,
        default=0.5,
        help="Q1 fail if resolved join coverage below this (default 0.5)",
    )
    ap.add_argument(
        "--allow-n-mismatch",
        action="store_true",
        help="Do not fail if n_total != 244 (fixtures / subset studies)",
    )
    args = ap.parse_args(argv)

    if not args.stage1_study.is_dir():
        print(f"ERROR: stage1 study not found: {args.stage1_study}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path(f"out/signal_study/m_b1_5_stage2_q1q3_{stamp}")
    study_id = args.study_id or out.name

    summary = run_stage2_q1q3_audit(
        stage1_study_dir=args.stage1_study,
        out_dir=out,
        gt_root=args.gt_root,
        git_commit=_git_commit(),
        study_id=study_id,
        expected_n=None if args.allow_n_mismatch else EXPECTED_D_ONLINE_N,
        min_join_coverage=args.min_join_coverage,
        enforce_n_total=not args.allow_n_mismatch,
    )

    print(f"wrote study → {out}")
    print(
        f"Q1={summary['stage2_q1_label_join']}  "
        f"Q2={summary['stage2_q2_population_support']}  "
        f"Q3={summary['stage2_q3_safe_negative_mass']}"
    )
    print(
        f"D_online={summary['D_online_total']}  "
        f"resolved={summary['label_resolved']}  "
        f"neg={summary['negative']}  "
        f"safe_removable={summary['safe_removable_negative']}"
    )
    print(f"next: {summary['next_authorized_step']}")
    print(f"production_preset: {summary['production_preset']}")

    # Exit non-zero only on hard FAIL_CLOSED recon when expecting 244
    if summary.get("reconciliation_acceptance") == "FAIL_CLOSED":
        print("ERROR: reconciliation FAIL_CLOSED", file=sys.stderr)
        return 1
    if summary["stage2_q1_label_join"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
