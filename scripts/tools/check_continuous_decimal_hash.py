#!/usr/bin/env python3
"""Validate ID-free final-MOT decimal output consistency in one process.

All unrecognised options are forwarded to ``scripts/eval/mot17.py``.  For
example, pass ``--preset mamba_whole_graph --detector SDP --max-frames 120``
after this tool's options.  The evaluator is invoked once with the complete
sequence list, so repeated entries exercise the same detector and runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import dataclasses
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from saccade.perception.eval.decimal_hash import (  # noqa: E402
    FIELDS,
    CanonicalRecord,
    canonicalize_mot_lines,
    decimal_hash,
    record_as_dict,
)


@dataclasses.dataclass(frozen=True)
class Run:
    index: int
    sequence: str
    records: tuple[CanonicalRecord, ...]
    decimal_hash: str


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences", required=True, help="Comma-separated ordered MOT sequences"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for validation artifacts"
    )
    parser.add_argument("--max-mismatch-records", type=int, default=20)
    args, forwarded = parser.parse_known_args()
    if "--processes" in forwarded or any(
        item.startswith("--processes=") for item in forwarded
    ):
        parser.error(
            "--processes is incompatible: this probe must remain in one process"
        )
    if "--cpp-threads" in forwarded or any(
        item.startswith("--cpp-threads=") for item in forwarded
    ):
        parser.error(
            "--cpp-threads is incompatible: the C++ evaluator cannot expose final MOT records to this probe"
        )
    if "--quantization" in forwarded or any(
        item.startswith("--quantization=") for item in forwarded
    ):
        parser.error(
            "--quantization was removed: this probe hashes final serialized decimal values exactly"
        )
    if "--output" in forwarded or any(
        item.startswith("--output=") for item in forwarded
    ):
        parser.error("use this tool's --output; evaluator output is managed internally")
    return args, forwarded


def _frame_multiset(
    records: Sequence[CanonicalRecord],
) -> dict[int, Counter[CanonicalRecord]]:
    result: dict[int, Counter[CanonicalRecord]] = {}
    for record in records:
        result.setdefault(record.frame, Counter())[record] += 1
    return result


def _diagnose(reference: Run, compared: Run, max_records: int) -> dict[str, Any]:
    count_equal = len(reference.records) == len(compared.records)
    if not count_equal:
        classification = "structural_divergence"
    else:
        classification = (
            "decimal_exact_pass"
            if reference.decimal_hash == compared.decimal_hash
            else "decimal_divergence"
        )
    reference_by_frame = _frame_multiset(reference.records)
    compared_by_frame = _frame_multiset(compared.records)
    diff_frames = sorted(
        frame
        for frame in set(reference_by_frame) | set(compared_by_frame)
        if reference_by_frame.get(frame, Counter())
        != compared_by_frame.get(frame, Counter())
    )
    samples = [
        {
            "frame": frame,
            "reference_records": [
                record_as_dict(record)
                for record, count in sorted(
                    reference_by_frame.get(frame, Counter()).items(),
                    key=lambda item: (item[0].frame, *item[0].values),
                )
                for _ in range(count)
            ],
            "compared_records": [
                record_as_dict(record)
                for record, count in sorted(
                    compared_by_frame.get(frame, Counter()).items(),
                    key=lambda item: (item[0].frame, *item[0].values),
                )
                for _ in range(count)
            ],
        }
        for frame in diff_frames[:max_records]
    ]
    return {
        "sequence": compared.sequence,
        "reference_run": reference.index,
        "compared_run": compared.index,
        "record_count_ref": len(reference.records),
        "record_count_test": len(compared.records),
        "decimal_hash_equal": reference.decimal_hash == compared.decimal_hash,
        "first_diff_frame": diff_frames[0] if diff_frames else None,
        "different_frame_count": len(diff_frames),
        "classification": classification,
        "frame_multiset_differences": samples,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    args, forwarded = _parse_args()
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    if not sequences:
        raise SystemExit("--sequences must contain at least one sequence")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluator_output = output / "mot_output"
    captured: list[tuple[str, tuple[str, ...]]] = []

    import saccade.perception.eval.runner as runner

    original_run_eval = runner.run_eval

    def capture_run_eval(**kwargs: Any) -> Any:
        if int(kwargs.get("cpp_threads", 0)) > 0:
            raise RuntimeError(
                "resolved cpp_threads is non-zero; the C++ evaluator cannot expose final MOT records to this probe"
            )

        def callback(sequence: str, lines: tuple[str, ...]) -> None:
            captured.append((sequence, lines))

        kwargs["sequence_result_callback"] = callback
        return original_run_eval(**kwargs)

    runner.run_eval = capture_run_eval
    original_argv = sys.argv[:]
    eval_script_dir = str(ROOT / "scripts" / "eval")
    inserted_eval_script_dir = eval_script_dir not in sys.path
    if inserted_eval_script_dir:
        sys.path.insert(0, eval_script_dir)
    try:
        sys.argv = [
            str(ROOT / "scripts" / "eval" / "mot17.py"),
            "--sequences",
            ",".join(sequences),
            "--output",
            str(evaluator_output),
            *forwarded,
        ]
        runpy.run_path(str(ROOT / "scripts" / "eval" / "mot17.py"), run_name="__main__")
    finally:
        runner.run_eval = original_run_eval
        sys.argv = original_argv
        if inserted_eval_script_dir:
            sys.path.remove(eval_script_dir)

    if len(captured) != len(sequences):
        raise RuntimeError(
            f"evaluator completed {len(captured)} sequences; expected {len(sequences)}"
        )

    runs: list[Run] = []
    for index, (sequence, lines) in enumerate(captured, start=1):
        records = tuple(canonicalize_mot_lines(lines))
        runs.append(Run(index, sequence, records, decimal_hash(records)))

    mismatches = [
        _diagnose(
            next(run for run in runs if run.sequence == current.sequence),
            current,
            args.max_mismatch_records,
        )
        for current in runs
    ]
    verdicts = {row["classification"] for row in mismatches}
    final_verdict = next(
        (
            value
            for value in ("structural_divergence", "decimal_divergence")
            if value in verdicts
        ),
        "decimal_exact_pass",
    )
    metadata = {
        "canonical_fields": list(FIELDS),
        "excluded_fields": [
            "global_track_id",
            "timestamps",
            "run UUID",
            "output path",
            "process metadata",
        ],
        "sort_key": "frame,x_centipixel,y_centipixel,w_centipixel,h_centipixel,score_1e4 (lexicographical; no identity field)",
        "decimal_hash": {
            "algorithm": "sha256",
            "source": "final serialized MOT text",
            "endian": "little",
            "binary_record": "int64 frame,x100,y100,w100,h100,score10000",
            "bbox_scale": 100,
            "score_scale": 10000,
            "non_finite": "rejected",
            "negative_zero": "canonicalized to integer zero",
        },
    }
    run_rows = [
        {
            "run_index": run.index,
            "sequence": run.sequence,
            "record_count": len(run.records),
            "decimal_hash": run.decimal_hash,
        }
        for run in runs
    ]
    hash_rows = [
        {
            "run_index": run.index,
            "sequence": run.sequence,
            "mode": "serialized_decimal",
            "hash": run.decimal_hash,
        }
        for run in runs
    ]
    _write_csv(output / "runs.csv", run_rows)
    _write_csv(output / "hashes.csv", hash_rows)
    _write_csv(
        output / "mismatches.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key != "frame_multiset_differences"
            }
            for row in mismatches
        ],
    )
    summary = {
        "command": [
            sys.executable,
            str(Path(__file__).relative_to(ROOT)),
            *original_argv[1:],
        ],
        "git_commit": _git_commit(),
        "config": forwarded,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "sequence_order": sequences,
        "runs": run_rows,
        "comparisons": mismatches,
        "metadata": metadata,
        "verdict": final_verdict,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"continuous decimal hash verdict: {final_verdict}; artifacts: {output}")
    return 0 if final_verdict == "decimal_exact_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
