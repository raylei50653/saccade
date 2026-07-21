#!/usr/bin/env python3
"""Routine pre-push continuous-chain determinism sentinel.

Runs a fixed six-sequence chain in one Python evaluator process::

    A, A, B, A, B, B

where:

* Sequence A: ``MOT17-04-SDP``
* Sequence B: ``MOT17-02-SDP``

Each sequence's first occurrence is the reference.  Later same-sequence
occurrences are compared against that reference (record count + serialized
decimal hash).  Any divergence fails the guard and retains diagnostic
artifacts.

This tool answers only: "does continuous same-process execution produce
inconsistent final MOT decimal output?"  It does **not** attribute
directional contamination.  After a failure:

* use ``check_decimal_matrix_2x2.py`` for directional/forensic cells;
* use ``check_decimal_matrix_all7.py`` for deep/release validation.

All unrecognised options are forwarded to ``scripts/eval/mot17.py``.
"""
# status: stable

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

SEQUENCE_A = "MOT17-04-SDP"
SEQUENCE_B = "MOT17-02-SDP"
ROUTINE_CHAIN: tuple[str, ...] = (
    SEQUENCE_A,
    SEQUENCE_A,
    SEQUENCE_B,
    SEQUENCE_A,
    SEQUENCE_B,
    SEQUENCE_B,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"out/determinism/routine_chain_{_timestamp()}"),
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
    chain = list(ROUTINE_CHAIN)

    print(
        f"routine continuous chain ({len(chain)} runs, one process): {','.join(chain)}"
    )

    _, runs = run_sequences(chain, eval_out, forwarded)
    comparisons = compare_chain_to_first_occurrence(
        runs, max_records=args.max_mismatch_records
    )
    final_verdict = verdict_from_comparisons(comparisons)

    run_rows: list[dict[str, Any]] = [
        {
            "run_index": run.index,
            "sequence": run.sequence,
            "sequence_occurrence": comparisons[index]["sequence_occurrence"],
            "record_count": len(run.records),
            "decimal_hash": run.decimal_hash,
        }
        for index, run in enumerate(runs)
    ]

    write_csv(root / "runs.csv", run_rows)
    write_csv(
        root / "hashes.csv",
        [
            {
                "run_index": run.index,
                "sequence": run.sequence,
                "hash": run.decimal_hash,
            }
            for run in runs
        ],
    )
    write_csv(
        root / "mismatches.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key != "frame_multiset_differences"
            }
            for row in comparisons
        ],
    )

    summary: dict[str, Any] = {
        "guard": "routine_continuous_chain",
        "sequence_a": SEQUENCE_A,
        "sequence_b": SEQUENCE_B,
        "sequence_order": chain,
        "runs": run_rows,
        "comparisons": comparisons,
        "metadata": {
            "canonical_fields": list(CANONICAL_FIELDS),
            "decimal_hash": HASH_METADATA,
            "reference_rule": (
                "first occurrence of each sequence is the reference; "
                "later same-sequence occurrences compare against it"
            ),
            "attribution": (
                "routine sentinel only — use 2x2 for directional forensics, "
                "all-7 for deep/release validation"
            ),
        },
        "verdict": final_verdict,
    }
    write_summary(root, summary)

    print(f"\nroutine chain verdict: {final_verdict}; artifacts: {root}")
    for row in comparisons:
        status = "pass" if row["classification"] == "decimal_exact_pass" else "FAIL"
        print(
            f"  [{row['run_index']}] {row['sequence']} "
            f"occ={row['sequence_occurrence']}: {status}  "
            f"recs={row['record_count_test']}  "
            f"hash_equal={row['decimal_hash_equal']}"
        )
        if row["classification"] == "decimal_exact_pass":
            continue
        print(
            f"    reference_hash={row['reference_hash']}  "
            f"observed_hash={row['observed_hash']}"
        )
        if row["first_diff_frame"] is not None:
            print(
                f"    first divergent frame: {row['first_diff_frame']}  "
                f"(different frames: {row['different_frame_count']})"
            )
        samples = row.get("frame_multiset_differences", [])
        if samples:
            sample = samples[0]
            ref_recs = sample.get("reference_records", [])
            cmp_recs = sample.get("compared_records", [])
            if ref_recs:
                print(f"    reference record[0]: {ref_recs[0]}")
            if cmp_recs:
                print(f"    observed record[0]: {cmp_recs[0]}")

    return 0 if final_verdict == "decimal_exact_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
