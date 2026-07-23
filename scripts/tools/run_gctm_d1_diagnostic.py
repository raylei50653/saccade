#!/usr/bin/env python3
"""GCTM D1 substrate-agnostic ranking diagnostic runner.

Synthetic / sealed non-runtime inputs only. Does not activate runtime B1/O1,
does not re-enter H0, and does not modify production presets or tracker state.

Usage:
  python scripts/tools/run_gctm_d1_diagnostic.py
  python scripts/tools/run_gctm_d1_diagnostic.py --out-dir \\
      docs/modules/semantic/research/evidence/gctm_d1_substrate_agnostic_ranking_20260723
"""

# status: experiment

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tools"))

from gctm_d1.runner import emit_packet  # noqa: E402


DEFAULT_OUT = (
    REPO
    / "docs"
    / "modules"
    / "semantic"
    / "research"
    / "evidence"
    / "gctm_d1_substrate_agnostic_ranking_20260723"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for sealed diagnostic packet artifacts",
    )
    args = parser.parse_args(argv)
    result = emit_packet(args.out_dir)
    print(
        json.dumps(
            {
                "selected_terminal": result["selected_terminal"],
                "all_invariants_passed": result["all_invariants_passed"],
                "out_dir": result["out_dir"],
                "manifest_status": result["manifest"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["all_invariants_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
