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
import hashlib
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
STAGE_PROBE_STAGES = ("detector_output", "post_nms", "tracker_input")

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
    parser.add_argument(
        "--stage-probe-frames",
        default="",
        help="Comma-separated frame numbers/ranges (for example 120-130) to hash detector and tracker-input tensors",
    )
    parser.add_argument(
        "--stage-probe-mode",
        choices=("passive", "fenced"),
        default="passive",
        help="Passive uses D2D snapshots without a fence; fenced synchronizes before each snapshot",
    )
    parser.add_argument(
        "--stage-probe-stages",
        default=",".join(STAGE_PROBE_STAGES),
        help="Comma-separated subset of detector_output,post_nms,tracker_input",
    )
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


def _parse_frame_ranges(value: str) -> set[int]:
    frames: set[int] = set()
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        try:
            if "-" in item:
                start_text, end_text = item.split("-", maxsplit=1)
                start, end = int(start_text), int(end_text)
                if start <= 0 or end < start:
                    raise ValueError
                frames.update(range(start, end + 1))
            else:
                frame = int(item)
                if frame <= 0:
                    raise ValueError
                frames.add(frame)
        except ValueError as exc:
            raise ValueError(f"invalid --stage-probe-frames item: {item!r}") from exc
    return frames


def _parse_stage_names(value: str) -> set[str]:
    stages = {item.strip() for item in value.split(",") if item.strip()}
    invalid = stages - set(STAGE_PROBE_STAGES)
    if invalid:
        raise ValueError(f"invalid --stage-probe-stages values: {sorted(invalid)}")
    return stages


def _tensor_hash(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _detection_multiset_hash(boxes: Any, scores: Any, classes: Any) -> str:
    boxes_array = boxes.cpu().contiguous().numpy()
    scores_array = scores.cpu().contiguous().numpy()
    classes_array = classes.cpu().contiguous().numpy()
    row_hashes: list[bytes] = []
    for index in range(scores_array.shape[0]):
        digest = hashlib.sha256()
        digest.update(boxes_array[index].tobytes())
        digest.update(scores_array[index : index + 1].tobytes())
        digest.update(classes_array[index : index + 1].tobytes())
        row_hashes.append(digest.digest())
    digest = hashlib.sha256()
    digest.update(len(row_hashes).to_bytes(8, "little"))
    for row_hash in sorted(row_hashes):
        digest.update(row_hash)
    return digest.hexdigest()


def _score_tie_stats(scores: Any) -> tuple[int, int]:
    scores_array = scores.cpu().contiguous().numpy()
    counts = Counter(
        scores_array[index : index + 1].tobytes()
        for index in range(scores_array.shape[0])
    )
    return len(counts), max(counts.values(), default=0)


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
    stage_probe_frames = _parse_frame_ranges(args.stage_probe_frames)
    stage_probe_stages = _parse_stage_names(args.stage_probe_stages)
    stage_snapshots: list[dict[str, Any]] = []

    import saccade.perception.eval.runner as runner

    original_run_eval = runner.run_eval

    def capture_run_eval(**kwargs: Any) -> Any:
        if int(kwargs.get("cpp_threads", 0)) > 0:
            raise RuntimeError(
                "resolved cpp_threads is non-zero; the C++ evaluator cannot expose final MOT records to this probe"
            )

        def callback(sequence: str, lines: tuple[str, ...]) -> None:
            captured.append((sequence, lines))

        def stage_callback(
            sequence: str,
            frame_id: int,
            stage: str,
            boxes: Any,
            scores: Any,
            classes: Any,
        ) -> None:
            if frame_id not in stage_probe_frames or stage not in stage_probe_stages:
                return
            import torch

            if args.stage_probe_mode == "fenced":
                torch.cuda.current_stream().synchronize()
            count = int(scores.shape[0])
            stage_snapshots.append(
                {
                    "sequence": sequence,
                    "sequence_occurrence": sum(item[0] == sequence for item in captured)
                    + 1,
                    "frame": frame_id,
                    "stage": stage,
                    "count": count,
                    "boxes": boxes[:count].detach().clone(),
                    "scores": scores[:count].detach().clone(),
                    "classes": classes[:count].detach().clone(),
                }
            )

        kwargs["sequence_result_callback"] = callback
        if stage_probe_frames:
            kwargs["stage_probe_callback"] = stage_callback
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

    stage_rows: list[dict[str, int | str]] = []
    if stage_snapshots:
        import torch

        torch.cuda.synchronize()
        for snapshot in stage_snapshots:
            boxes = snapshot.pop("boxes")
            scores = snapshot.pop("scores")
            classes = snapshot.pop("classes")
            boxes_hash = _tensor_hash(boxes)
            scores_hash = _tensor_hash(scores)
            classes_hash = _tensor_hash(classes)
            unique_scores, max_score_multiplicity = _score_tie_stats(scores)
            stage_rows.append(
                {
                    **snapshot,
                    "boxes_hash": boxes_hash,
                    "scores_hash": scores_hash,
                    "classes_hash": classes_hash,
                    "ordered_hash": hashlib.sha256(
                        (boxes_hash + scores_hash + classes_hash).encode()
                    ).hexdigest(),
                    "multiset_hash": _detection_multiset_hash(boxes, scores, classes),
                    "unique_score_count": unique_scores,
                    "max_score_multiplicity": max_score_multiplicity,
                }
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
    if stage_probe_frames:
        _write_csv(output / "stage_hashes.csv", stage_rows)
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
    stage_comparisons: list[dict[str, int | str | bool]] = []
    if stage_probe_frames:
        reference_rows: dict[tuple[str, int, str], dict[str, int | str]] = {}
        for row in stage_rows:
            key = (str(row["sequence"]), int(row["frame"]), str(row["stage"]))
            reference = reference_rows.setdefault(key, row)
            if row is reference:
                continue
            stage_comparisons.append(
                {
                    "sequence": row["sequence"],
                    "sequence_occurrence": row["sequence_occurrence"],
                    "frame": row["frame"],
                    "stage": row["stage"],
                    "reference_boxes_hash": reference["boxes_hash"],
                    "boxes_hash_equal": row["boxes_hash"] == reference["boxes_hash"],
                    "scores_hash_equal": row["scores_hash"] == reference["scores_hash"],
                    "classes_hash_equal": row["classes_hash"]
                    == reference["classes_hash"],
                    "ordered_hash_equal": row["ordered_hash"]
                    == reference["ordered_hash"],
                    "multiset_hash_equal": row["multiset_hash"]
                    == reference["multiset_hash"],
                    "count_equal": row["count"] == reference["count"],
                }
            )
        _write_csv(output / "stage_comparisons.csv", stage_comparisons)
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
        "stage_probe": {
            "frames": sorted(stage_probe_frames),
            "mode": args.stage_probe_mode,
            "stages": sorted(stage_probe_stages),
            "comparisons": stage_comparisons,
        }
        if stage_probe_frames
        else None,
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
