#!/usr/bin/env python3
"""Run R1.1 Transfer Failure Attribution Pack (authorized research only)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from saccade.perception.eval.safe_region_assetization_r11 import (  # noqa: E402
    run_r11_study,
)


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
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--study-id", type=str, default=None)
    args = p.parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    study_id = args.study_id or f"safe_region_assetization_r11_{ts}"
    out = args.out or (REPO / "out/signal_study" / study_id)
    if not args.q45_dir.is_dir():
        print(f"ERROR: missing {args.q45_dir}", file=sys.stderr)
        return 2
    if not args.events.is_file():
        print(f"ERROR: missing {args.events}", file=sys.stderr)
        return 2
    summary = run_r11_study(
        q45_dir=args.q45_dir,
        events_path=args.events,
        out_dir=out,
        study_id=study_id,
    )
    tax = summary.get("failure_taxonomy", {})
    print(f"study_id={summary.get('study_id')}")
    print(f"out={out}")
    print(f"primary={tax.get('primary')} secondary={tax.get('secondary')}")
    print(f"mapping={tax.get('decision_mapping')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
