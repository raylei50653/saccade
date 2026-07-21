#!/usr/bin/env python3
"""Run Safe-Region Assetization R1 study (Phase A assets + Phase B linear probe).

Research-only. Does not touch production hooks/presets.
"""
# status: stable

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from saccade.perception.eval.safe_region_assetization_r1 import run_study  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--q45-dir",
        type=Path,
        default=REPO / "out/signal_study/m_b1_5_stage2_q45_20260710",
    )
    p.add_argument(
        "--events",
        type=Path,
        default=REPO
        / "out/signal_study/m_b1_5_stage2_q1q3_20260710/d_online_events.parquet",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Study output directory (default: out/signal_study/safe_region_assetization_r1_<ts>/)",
    )
    p.add_argument("--study-id", type=str, default=None)
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    study_id = args.study_id or f"safe_region_assetization_r1_{ts}"
    out = args.out or (REPO / "out/signal_study" / study_id)

    if not args.q45_dir.is_dir():
        print(f"ERROR: q45 dir missing: {args.q45_dir}", file=sys.stderr)
        return 2
    if not args.events.is_file():
        print(f"ERROR: events missing: {args.events}", file=sys.stderr)
        return 2

    summary = run_study(
        q45_dir=args.q45_dir,
        events_path=args.events,
        out_dir=out,
        study_id=study_id,
    )
    v = summary.get("verdict", {})
    print(f"study_id={summary.get('study_id')}")
    print(f"out={out}")
    print(f"verdict={v.get('verdict_code')}: {v.get('verdict_text', '')[:200]}")
    print(f"terminal_b_retained={summary.get('terminal_b_retained')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
