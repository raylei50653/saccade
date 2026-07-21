#!/usr/bin/env python3
"""Full all-7 sequence order-contamination validation.

Generates the complete 28-run order matrix across all seven MOT17-SDP
sequences (forward, reverse, forward again) and compares every occurrence
against the first occurrence of each sequence.

This is deep / release validation.  Routine pre-push uses the continuous
chain sentinel in ``check_decimal_chain_routine.py``.  Directional forensics
after a failure use ``check_decimal_matrix_2x2.py``.

All unrecognised options are forwarded to ``scripts/eval/mot17.py``.
"""
# status: diagnostic

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_SRC_DIR))

from saccade.perception.eval._decimal_hash_tools import (  # noqa: E402
    CANONICAL_FIELDS,
    HASH_METADATA,
    compare_chain_to_first_occurrence,
    run_sequences,
    verdict_from_comparisons,
    write_csv,
    write_summary,
)

_ALL7_SDP = [
    "MOT17-04-SDP",
    "MOT17-02-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

_ALL7_ORDER: list[str] = []
for _seq in _ALL7_SDP:
    _ALL7_ORDER.append(_seq)
    _ALL7_ORDER.append(_seq)
for _i in range(1, len(_ALL7_SDP)):
    _ALL7_ORDER.append(_ALL7_SDP[_i])
    _ALL7_ORDER.append(_ALL7_SDP[0])
    _ALL7_ORDER.append(_ALL7_SDP[_i])
for _seq in reversed(_ALL7_SDP[1:]):
    _ALL7_ORDER.append(_ALL7_SDP[0])
    _ALL7_ORDER.append(_seq)
for _seq in reversed(_ALL7_SDP):
    _ALL7_ORDER.append(_seq)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"out/determinism/matrix_all7_{_timestamp()}"),
        help="Output artifact directory",
    )
    parser.add_argument(
        "--max-mismatch-records",
        type=int,
        default=20,
    )
    args, forwarded = parser.parse_known_args()
    for flag in ("--processes", "--cpp-threads", "--output", "--sequences"):
        if flag in forwarded or any(item.startswith(f"{flag}=") for item in forwarded):
            parser.error(
                f"{flag} is managed by this tool; do not pass it in forwarded arguments"
            )
    return args, forwarded


def main() -> int:
    args, forwarded = _parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    eval_out = root / "mot_output"

    print(f"all-7 matrix: {len(_ALL7_ORDER)} runs across {len(_ALL7_SDP)} sequences")
    print(f"order: {','.join(_ALL7_ORDER)}")

    _, runs = run_sequences(_ALL7_ORDER, eval_out, forwarded)

    run_rows = [
        {
            "run_index": run.index,
            "sequence": run.sequence,
            "record_count": len(run.records),
            "decimal_hash": run.decimal_hash,
        }
        for run in runs
    ]

    comparisons = compare_chain_to_first_occurrence(
        runs, max_records=args.max_mismatch_records
    )
    final_verdict = verdict_from_comparisons(comparisons)

    write_csv(root / "runs.csv", run_rows)
    write_csv(
        root / "hashes.csv",
        [
            {"run_index": run.index, "sequence": run.sequence, "hash": run.decimal_hash}
            for run in runs
        ],
    )
    write_csv(
        root / "comparisons.csv",
        [
            {k: v for k, v in c.items() if k != "frame_multiset_differences"}
            for c in comparisons
        ],
    )

    summary: dict[str, Any] = {
        "matrix": "all7",
        "sequences": _ALL7_SDP,
        "sequence_order": _ALL7_ORDER,
        "runs": run_rows,
        "comparisons": comparisons,
        "metadata": {
            "canonical_fields": list(CANONICAL_FIELDS),
            "decimal_hash": HASH_METADATA,
        },
        "verdict": final_verdict,
    }
    write_summary(root, summary)

    print(f"\nall-7 determinism verdict: {final_verdict}; artifacts: {root}")
    return 0 if final_verdict == "decimal_exact_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
