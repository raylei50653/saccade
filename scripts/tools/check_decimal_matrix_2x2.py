#!/usr/bin/env python3
"""Forensic / directional 2×2 sequence-order determinism matrix.

Runs the complete directed matrix::

    A → A    B → A
    A → B    B → B

Each cell means two full sequences executed consecutively within the same
process and runtime state.  This tool is **not** the routine pre-push guard;
use ``check_decimal_chain_routine.py`` for that.  The 2×2 matrix is retained
for post-failure directional diagnosis:

1. same-sequence continuous-run instability (A→A, B→B);
2. cross-sequence state or buffer contamination (B→A, A→B);
3. directional contamination differences (B→A vs A→B);
4. final serialized-MOT decimal divergence with cleaner cell-level references.

Fixed sequence pair (frozen 2026-07-10):

* Sequence A: ``MOT17-04-SDP``  (1 050 frames, 44 248 output records, 1 920×1 080)
* Sequence B: ``MOT17-02-SDP``  (600 frames, 11 722 output records, 1 920×1 080)

Selection rationale for B:  MOT17-02-SDP appears interleaved with 04 in the
all-7 matrix where it was the nearest-size SDP neighbour.  Its 4× smaller
detection footprint exercises substantially different postprocess buffer
utilisation and CUDA-graph static sizing on either side of the 04 footprint.
It was self-consistent in all all-7 comparisons and the nearest-neighbour
runs survived directional cross-contamination at the record level.

All unrecognised options are forwarded to ``scripts/eval/mot17.py``.
"""
# status: diagnostic

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parents[1] / "src"
sys.path.insert(0, str(_SRC_DIR))

from saccade.perception.eval._decimal_hash_tools import (  # noqa: E402
    Run,
    CANONICAL_FIELDS,
    HASH_METADATA,
    diagnose,
    generate_manifest,
    run_sequences,
    write_csv,
    write_summary,
)

_DEFAULT_SEQUENCE_A = "MOT17-04-SDP"
_DEFAULT_SEQUENCE_B = "MOT17-02-SDP"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _determinism_env() -> dict[str, str]:
    relevant = [
        "SACCADE_DETERMINISTIC_FILTER_COMPACTION",
        "CUDA_LAUNCH_BLOCKING",
        "SACCADE_DISABLE_CUDA_GRAPHS",
    ]
    return {k: v for k in relevant if (v := os.environ.get(k)) is not None}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-a",
        default=_DEFAULT_SEQUENCE_A,
        help=f"Sequence A (default: {_DEFAULT_SEQUENCE_A})",
    )
    parser.add_argument(
        "--sequence-b",
        default=_DEFAULT_SEQUENCE_B,
        help=f"Sequence B (default: {_DEFAULT_SEQUENCE_B})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"out/determinism/matrix_2x2_{_timestamp()}"),
        help="Output artifact directory",
    )
    parser.add_argument(
        "--max-mismatch-records",
        type=int,
        default=20,
    )
    args, forwarded = parser.parse_known_args()
    _reject_forwarded(parser, forwarded)
    return args, forwarded


def _reject_forwarded(parser: argparse.ArgumentParser, forwarded: list[str]) -> None:
    for flag in ("--processes", "--cpp-threads", "--output"):
        if flag in forwarded or any(item.startswith(f"{flag}=") for item in forwarded):
            parser.error(
                f"{flag} is managed by this tool; do not pass it in forwarded arguments"
            )


def _run_cell(
    label: str,
    preceding: str,
    target: str,
    root_output: Path,
    forwarded: list[str],
) -> tuple[Run, Run]:
    cell_dir = root_output / label
    eval_out = cell_dir / "mot_output"
    cell_dir.mkdir(parents=True, exist_ok=True)
    _, runs = run_sequences([preceding, target], eval_out, forwarded)
    return runs[0], runs[1]


def _build_comparison(
    cell_label: str,
    preceding: str,
    target: str,
    reference: Run,
    test_run: Run,
    max_records: int,
) -> dict[str, Any]:
    diag = diagnose(reference, test_run, max_records)
    return {
        "cell": cell_label,
        "preceding_sequence": preceding,
        "target_sequence": target,
        "reference_hash": reference.decimal_hash,
        "observed_hash": test_run.decimal_hash,
        "record_count_ref": len(reference.records),
        "record_count_test": len(test_run.records),
        "decimal_hash_equal": diag["decimal_hash_equal"],
        "first_diff_frame": diag["first_diff_frame"],
        "different_frame_count": diag["different_frame_count"],
        "classification": diag["classification"],
        "frame_multiset_differences": diag.get("frame_multiset_differences", []),
    }


def _extract_config(forwarded: list[str]) -> tuple[str, str]:
    config = "unknown"
    for flag in ("--preset", "--config"):
        if flag in forwarded:
            idx = forwarded.index(flag)
            if idx + 1 < len(forwarded):
                config = forwarded[idx + 1]
                break
    detector = "unknown"
    if "--detector" in forwarded:
        idx = forwarded.index("--detector")
        if idx + 1 < len(forwarded):
            detector = forwarded[idx + 1]
    return config, detector


def main() -> int:
    args, forwarded = _parse_args()
    seq_a = args.sequence_a
    seq_b = args.sequence_b
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    det_env = _determinism_env()
    ts_start = _timestamp()

    cells: list[tuple[str, str, str]] = [
        ("A_to_A", seq_a, seq_a),
        ("B_to_B", seq_b, seq_b),
        ("B_to_A", seq_b, seq_a),
        ("A_to_B", seq_a, seq_b),
    ]

    all_target_runs: dict[str, Run] = {}

    for label, pre, tgt in cells:
        print(f"── cell {label}  ({pre} → {tgt})", flush=True)
        try:
            prec_run, tgt_run = _run_cell(label, pre, tgt, root, forwarded)
        except Exception as exc:
            print(f"   FAILED: cell {label} raised {type(exc).__name__}: {exc}")
            return 1
        all_target_runs[f"{label}_preceding"] = prec_run
        all_target_runs[f"{label}_target"] = tgt_run
        print(
            f"   {pre}: {len(prec_run.records)} records  "
            f"{tgt}: {len(tgt_run.records)} records"
        )

    a_ref = all_target_runs["A_to_A_preceding"]
    b_ref = all_target_runs["B_to_B_preceding"]

    comparisons = [
        _build_comparison(
            "A_to_A",
            seq_a,
            seq_a,
            a_ref,
            all_target_runs["A_to_A_target"],
            args.max_mismatch_records,
        ),
        _build_comparison(
            "B_to_B",
            seq_b,
            seq_b,
            b_ref,
            all_target_runs["B_to_B_target"],
            args.max_mismatch_records,
        ),
        _build_comparison(
            "B_to_A",
            seq_b,
            seq_a,
            a_ref,
            all_target_runs["B_to_A_target"],
            args.max_mismatch_records,
        ),
        _build_comparison(
            "A_to_B",
            seq_a,
            seq_b,
            b_ref,
            all_target_runs["A_to_B_target"],
            args.max_mismatch_records,
        ),
    ]

    classifications = {c["classification"] for c in comparisons}
    final_verdict = "decimal_exact_pass"
    for v in ("structural_divergence", "decimal_divergence"):
        if v in classifications:
            final_verdict = v
            break

    config_name, detector_name = _extract_config(forwarded)
    manifest = generate_manifest(
        ts_start,
        seq_a,
        seq_b,
        config_name,
        detector_name,
        "--double-buffer" in forwarded,
        forwarded,
        det_env,
    )
    manifest["completion_timestamp"] = _timestamp()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    run_rows: list[dict[str, Any]] = []
    for key in (
        "A_to_A_preceding",
        "A_to_A_target",
        "B_to_B_preceding",
        "B_to_B_target",
        "B_to_A_preceding",
        "B_to_A_target",
        "A_to_B_preceding",
        "A_to_B_target",
    ):
        r = all_target_runs[key]
        run_rows.append(
            {
                "key": key,
                "sequence": r.sequence,
                "record_count": len(r.records),
                "decimal_hash": r.decimal_hash,
            }
        )

    hash_rows = [
        {"key": c["cell"], "sequence": c["target_sequence"], "hash": c["observed_hash"]}
        for c in comparisons
    ]

    write_csv(root / "runs.csv", run_rows)
    write_csv(root / "hashes.csv", hash_rows)
    write_csv(
        root / "comparisons.csv",
        [
            {k: v for k, v in c.items() if k != "frame_multiset_differences"}
            for c in comparisons
        ],
    )

    summary: dict[str, Any] = {
        "matrix": "2x2",
        "sequence_a": seq_a,
        "sequence_b": seq_b,
        "cells": [
            {
                "label": c["cell"],
                "preceding": c["preceding_sequence"],
                "target": c["target_sequence"],
                "record_count": c["record_count_test"],
                "classification": c["classification"],
                "decimal_hash_equal": c["decimal_hash_equal"],
            }
            for c in comparisons
        ],
        "comparisons": comparisons,
        "metadata": {
            "canonical_fields": list(CANONICAL_FIELDS),
            "decimal_hash": HASH_METADATA,
        },
        "verdict": final_verdict,
    }
    write_summary(root, summary)

    print(f"\n2×2 determinism verdict: {final_verdict}; artifacts: {root}")
    for c in comparisons:
        status = "pass" if c["classification"] == "decimal_exact_pass" else "FAIL"
        print(
            f"  {c['cell']}: {status}  "
            f"({c['target_sequence']} recs={c['record_count_test']}, "
            f"hash_equal={c['decimal_hash_equal']})"
        )

    for c in comparisons:
        if c["classification"] == "decimal_exact_pass":
            continue
        ff = c["first_diff_frame"]
        if ff is not None:
            print(
                f"    first divergent frame: {ff}  "
                f"(different frames: {c['different_frame_count']})"
            )
        samples = c.get("frame_multiset_differences", [])
        if samples:
            s = samples[0]
            ref_recs = s.get("reference_records", [])
            cmp_recs = s.get("compared_records", [])
            if ref_recs:
                print(f"    reference record[0]: {json.dumps(ref_recs[0])}")
            if cmp_recs:
                print(f"    observed record[0]: {json.dumps(cmp_recs[0])}")

    return 0 if final_verdict == "decimal_exact_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
