"""Shared low-level utilities for decimal-hash determinism tools.

This internal module provides the capture, comparison, and artifact-serialization
primitives consumed by ``check_continuous_decimal_hash.py``,
``check_decimal_matrix_2x2.py``, and ``check_decimal_matrix_all7.py``.  It is not
part of the public API.
"""

from __future__ import annotations

import csv
import dataclasses
from collections import Counter
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from .decimal_hash import (
    CanonicalRecord,
    canonicalize_mot_lines,
    decimal_hash,
    record_as_dict,
)

ROOT = Path(__file__).resolve().parents[4]


@dataclasses.dataclass(frozen=True)
class Run:
    index: int
    sequence: str
    records: tuple[CanonicalRecord, ...]
    decimal_hash: str


@dataclasses.dataclass
class CellResult:
    label: str
    preceding: str
    target: str
    preceding_run: Run
    target_run: Run
    target_record_count: int
    target_is_ref: bool
    decimal_hash_equal: bool | None
    classification: str


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool:
    try:
        return (
            subprocess.call(
                ["git", "diff", "--quiet"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            != 0
        )
    except OSError:
        return False


def _frame_multiset(
    records: Sequence[CanonicalRecord],
) -> dict[int, Counter[CanonicalRecord]]:
    result: dict[int, Counter[CanonicalRecord]] = {}
    for record in records:
        result.setdefault(record.frame, Counter())[record] += 1
    return result


def diagnose(
    reference: Run,
    compared: Run,
    max_records: int = 20,
) -> dict[str, Any]:
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
    frame_samples: list[dict[str, Any]] = []
    for frame in diff_frames[:max_records]:
        ref_counter = reference_by_frame.get(frame, Counter())
        cmp_counter = compared_by_frame.get(frame, Counter())
        frame_samples.append(
            {
                "frame": frame,
                "reference_records": [
                    record_as_dict(record)
                    for record, count in sorted(
                        ref_counter.items(),
                        key=lambda item: (item[0].frame, *item[0].values),
                    )
                    for _ in range(count)
                ],
                "compared_records": [
                    record_as_dict(record)
                    for record, count in sorted(
                        cmp_counter.items(),
                        key=lambda item: (item[0].frame, *item[0].values),
                    )
                    for _ in range(count)
                ],
            }
        )
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
        "frame_multiset_differences": frame_samples,
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def write_summary(output: Path, summary: dict[str, Any]) -> None:
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def generate_manifest(
    timestamp: str,
    sequence_a: str,
    sequence_b: str,
    config: str,
    detector: str,
    double_buffer: bool,
    forwarded: list[str],
    determinism_env: dict[str, str],
) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "dirty_tree": _git_dirty(),
        "command": [sys.executable, *sys.argv],
        "model": config,
        "detector": detector,
        "sequence_a": sequence_a,
        "sequence_b": sequence_b,
        "double_buffer": double_buffer,
        "determinism_environment": determinism_env,
        "start_timestamp": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run_sequences(
    sequences: list[str],
    evaluator_output: Path,
    forwarded: list[str],
) -> tuple[list[tuple[str, tuple[str, ...]]], list[Run]]:
    """Run evaluator once and capture final MOT lines per sequence.

    Returns (captured, runs) where *captured* is a list of (sequence_name,
    mot_lines) pairs and *runs* are parsed, sorted, and hashed records.
    """
    import runpy

    evaluator_script = str(ROOT / "scripts" / "eval" / "mot17.py")
    eval_script_dir = str(ROOT / "scripts" / "eval")

    captured: list[tuple[str, tuple[str, ...]]] = []

    import saccade.perception.eval.runner as runner

    original_run_eval = runner.run_eval

    def _capture(**kwargs: Any) -> Any:
        def _cb(seq: str, lines: tuple[str, ...]) -> None:
            captured.append((seq, lines))

        kwargs["sequence_result_callback"] = _cb
        return original_run_eval(**kwargs)

    runner.run_eval = _capture

    original_argv = sys.argv[:]
    inserted_dir = eval_script_dir not in sys.path
    if inserted_dir:
        sys.path.insert(0, eval_script_dir)
    try:
        sys.argv = [
            evaluator_script,
            "--sequences",
            ",".join(sequences),
            "--output",
            str(evaluator_output),
            *forwarded,
        ]
        runpy.run_path(evaluator_script, run_name="__main__")
    finally:
        runner.run_eval = original_run_eval
        sys.argv = original_argv
        if inserted_dir:
            sys.path.remove(eval_script_dir)

    if len(captured) != len(sequences):
        raise RuntimeError(
            f"evaluator completed {len(captured)} sequences; expected {len(sequences)}"
        )

    runs: list[Run] = []
    for index, (sequence, lines) in enumerate(captured, start=1):
        records = tuple(canonicalize_mot_lines(lines))
        runs.append(Run(index, sequence, records, decimal_hash(records)))
    return captured, runs


CANONICAL_FIELDS = (
    "frame",
    "x_centipixel",
    "y_centipixel",
    "w_centipixel",
    "h_centipixel",
    "score_1e4",
)

HASH_METADATA = {
    "algorithm": "sha256",
    "source": "final serialized MOT text",
    "endian": "little",
    "binary_record": "int64 frame,x100,y100,w100,h100,score10000",
    "bbox_scale": 100,
    "score_scale": 10000,
    "non_finite": "rejected",
    "negative_zero": "canonicalized to integer zero",
    "excluded_fields": [
        "global_track_id",
        "timestamps",
        "run UUID",
        "output path",
        "process metadata",
    ],
    "sort_key": (
        "frame,x_centipixel,y_centipixel,w_centipixel,h_centipixel,score_1e4 "
        "(lexicographical; no identity field)"
    ),
}
