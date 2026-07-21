#!/usr/bin/env python3
"""M-B1.5 Stage 2 Q4.5: structured threshold-combination atlas.

Descriptive atlas only. Does **not** promote thresholds/rules or change presets.

Usage:
  uv run python scripts/tools/run_m_b1_5_stage2_q45_atlas.py \\
    --q1q3-study out/signal_study/m_b1_5_stage2_q1q3_20260710 \\
    --out out/signal_study/m_b1_5_stage2_q45_20260710
"""
# status: stable

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from saccade.perception.eval.d_online_stage2_q45_atlas import (
    Q1Q3_STUDY_ID,
    run_stage2_q45_atlas,
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
    )
    ap.add_argument(
        "--stage1-study",
        type=Path,
        default=Path("out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close"),
        help="Stage1 study for frame provenance MOT cross-check",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--study-id", type=str, default=None)
    ap.add_argument(
        "--no-secondary-competition",
        action="store_true",
        help="Skip secondary competition-relative columns",
    )
    args = ap.parse_args(argv)

    if not args.q1q3_study.is_dir():
        print(f"ERROR: q1q3 study not found: {args.q1q3_study}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path(f"out/signal_study/m_b1_5_stage2_q45_{stamp}")
    study_id = args.study_id or out.name

    summary = run_stage2_q45_atlas(
        q1q3_study_dir=args.q1q3_study,
        out_dir=out,
        stage1_study_dir=args.stage1_study,
        git_commit=_git_commit(),
        study_id=study_id,
        include_secondary_competition=not args.no_secondary_competition,
    )

    print(f"wrote study → {out}")
    print(f"terminal={summary['stage2_q45_terminal']}  ({summary['terminal_letter']})")
    print(
        f"primary: neg={summary['n_primary_negative']}  "
        f"pos={summary['n_primary_positive_protect']}"
    )
    print(
        f"atlas: single={summary['n_atom_atlas_rows']}  "
        f"AND={summary['n_pairwise_and_rows']}  "
        f"OR={summary['n_pairwise_or_rows']}"
    )
    print(
        f"productive_safe: single={summary['n_productive_safe_single']}  "
        f"AND={summary['n_productive_safe_and']}  "
        f"OR={summary['n_productive_safe_or']}"
    )
    print(
        f"frame: {summary['frame_provenance_kind']}  "
        f"absolute_mot={summary['frame_is_absolute_mot']}"
    )
    print(f"next: {summary['next_authorized_step']}")
    print(f"production_preset: {summary['production_preset']}")

    if summary.get("reconciliation_acceptance") == "FAIL_CLOSED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
