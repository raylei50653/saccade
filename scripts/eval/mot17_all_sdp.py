#!/usr/bin/env python3

"""Batch-run MOT17 SDP sequences through the standard eval path."""
# status: stable
# mypy: ignore-errors

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from saccade.perception.eval.metrics import run_motmetrics_evaluation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all MOT17 SDP sequences by dispatching one mot17.py subprocess "
            "per sequence, then merge outputs and compute overall MOT metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="results/MOT17_eval_all_sdp")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of concurrent sequence jobs. Typical useful range is 8-16.",
    )
    parser.add_argument(
        "--sequences",
        default="",
        help="Optional comma-separated SDP subset. Empty means all SDP sequences.",
    )
    parser.add_argument(
        "--detector",
        default="SDP",
        help="Detector suffix to select. This entrypoint is intended for SDP runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Finish other sequences even if one subprocess fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned per-sequence commands without executing them.",
    )
    return parser


def _strip_flag(argv: list[str], flag: str, takes_value: bool) -> list[str]:
    result: list[str] = []
    skip_next = False
    prefix = f"{flag}="
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == flag:
            if takes_value:
                skip_next = True
            continue
        if takes_value and token.startswith(prefix):
            continue
        result.append(token)
    return result


def _has_flag(argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(token == flag or token.startswith(prefix) for token in argv)


def _discover_sequences(data_root: Path, split: str, selected: str) -> list[str]:
    if selected.strip():
        seqs = [item.strip() for item in selected.split(",") if item.strip()]
        invalid = [seq for seq in seqs if not seq.endswith("-SDP")]
        if invalid:
            raise ValueError(
                "mot17_all_sdp.py only accepts SDP sequences; got: "
                + ", ".join(invalid)
            )
        return seqs

    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    return sorted(
        d.name for d in split_dir.iterdir() if d.is_dir() and d.name.endswith("-SDP")
    )


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(project_root)]
    if src_path.exists():
        pythonpath_entries.append(str(src_path))
    if build_path.exists():
        pythonpath_entries.append(str(build_path))
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _run_sequence(
    seq: str,
    cmd: list[str],
    output_dir: Path,
    env: dict[str, str],
    dry_run: bool,
) -> dict[str, object]:
    log_path = output_dir / f"{seq}.log"
    if dry_run:
        return {
            "sequence": seq,
            "returncode": 0,
            "log_path": str(log_path),
            "cmd": cmd,
            "dry_run": True,
            "wall_sec": 0.0,
        }

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        text=True,
        capture_output=True,
    )
    wall_sec = time.perf_counter() - start
    log_path.write_text(proc.stdout + proc.stderr)
    return {
        "sequence": seq,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "cmd": cmd,
        "dry_run": False,
        "wall_sec": wall_sec,
    }


def _parse_kv_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    parts = [part for part in line.strip().split("\t") if part]
    if parts:
        fields["name"] = parts[0]
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def _load_latency_summary(
    output_root: Path,
    per_seq_root: Path,
    results: list[dict[str, object]],
) -> dict[str, object]:
    result_by_seq = {str(item["sequence"]): item for item in results}
    per_sequence: dict[str, dict[str, object]] = {}
    total_frames = 0
    weighted_mean_ms = 0.0
    weighted_wall_sec = 0.0
    profile_enabled = False
    overall_samples_ms: list[float] = []

    for seq_dir in sorted(per_seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        fps_path = seq_dir / "_fps_summary.txt"
        if not fps_path.exists():
            continue
        seq_line = ""
        for line in fps_path.read_text().splitlines():
            if line.startswith(seq + "\t"):
                seq_line = line
                break
        if not seq_line:
            continue

        fields = _parse_kv_fields(seq_line)
        frames = int(fields.get("frames", "0") or 0)
        mean_ms_raw = fields.get("mean_ms", "n/a")
        fps_raw = fields.get("fps", "n/a")
        mean_ms = None if mean_ms_raw == "n/a" else float(mean_ms_raw)
        fps = None if fps_raw == "n/a" else float(fps_raw)
        wall_sec = float(result_by_seq.get(seq, {}).get("wall_sec", 0.0))

        seq_summary: dict[str, object] = {
            "frames": frames,
            "fps": fps,
            "mean_latency_ms": mean_ms,
            "wall_sec": round(wall_sec, 3),
        }

        stage_profile_path = seq_dir / "_stage_profile.json"
        if stage_profile_path.exists():
            profile_enabled = True
            stage_profile = json.loads(stage_profile_path.read_text())
            frame_total = stage_profile.get("overall", {}).get("frame_total", {}) or {}
            if frame_total:
                seq_summary["p95_latency_ms"] = float(frame_total.get("p95_ms", 0.0))
                seq_summary["p99_latency_ms"] = float(frame_total.get("p99_ms", 0.0))
                seq_summary["std_latency_ms"] = float(frame_total.get("std_ms", 0.0))

        latency_profile_path = seq_dir / "_latency_profile.json"
        if latency_profile_path.exists():
            latency_profile = json.loads(latency_profile_path.read_text())
            samples_ms = [float(x) for x in latency_profile.get("samples_ms", [])]
            if samples_ms:
                overall_samples_ms.extend(samples_ms)
                seq_summary["mean_latency_ms"] = float(latency_profile["mean_ms"])
                seq_summary["std_latency_ms"] = float(latency_profile["std_ms"])
                seq_summary["p95_latency_ms"] = float(latency_profile["p95_ms"])
                seq_summary["p99_latency_ms"] = float(latency_profile["p99_ms"])
                seq_summary["fps"] = float(latency_profile["fps"])
                seq_summary["frames"] = int(latency_profile["frames"])

        per_sequence[seq] = seq_summary

        if frames > 0 and mean_ms is not None:
            total_frames += frames
            weighted_mean_ms += mean_ms * frames
        weighted_wall_sec += wall_sec

    overall: dict[str, object] = {
        "sequence_count": len(per_sequence),
        "profile_stages_enabled": profile_enabled,
        "total_frames": total_frames,
        "total_wall_sec": round(weighted_wall_sec, 3),
    }
    if total_frames > 0:
        overall_mean_ms = weighted_mean_ms / total_frames
        overall["mean_latency_ms"] = round(overall_mean_ms, 3)
        overall["fps"] = (
            round(1000.0 / overall_mean_ms, 3) if overall_mean_ms > 0 else 0.0
        )
    if overall_samples_ms:
        arr = np.array(overall_samples_ms, dtype=np.float64)
        overall["std_latency_ms"] = round(float(arr.std()), 3)
        overall["p95_latency_ms"] = round(float(np.percentile(arr, 95)), 3)
        overall["p99_latency_ms"] = round(float(np.percentile(arr, 99)), 3)
    elif profile_enabled:
        overall["p95_latency_ms"] = None
        overall["p99_latency_ms"] = None
        overall["note"] = (
            "Overall P95/P99 unavailable because per-sequence latency samples were not present."
        )

    payload = {"overall": overall, "per_sequence": per_sequence}
    (output_root / "_latency_summary.json").write_text(json.dumps(payload, indent=2))

    lines = [("sequence\tframes\tfps\tmean_ms\tp95_ms\tp99_ms\tstd_ms\twall_sec")]
    for seq, info in per_sequence.items():
        lines.append(
            "\t".join(
                [
                    seq,
                    str(info.get("frames", 0)),
                    "n/a" if info.get("fps") is None else f"{float(info['fps']):.2f}",
                    (
                        "n/a"
                        if info.get("mean_latency_ms") is None
                        else f"{float(info['mean_latency_ms']):.2f}"
                    ),
                    (
                        "n/a"
                        if info.get("p95_latency_ms") is None
                        else f"{float(info['p95_latency_ms']):.2f}"
                    ),
                    (
                        "n/a"
                        if info.get("p99_latency_ms") is None
                        else f"{float(info['p99_latency_ms']):.2f}"
                    ),
                    (
                        "n/a"
                        if info.get("std_latency_ms") is None
                        else f"{float(info['std_latency_ms']):.2f}"
                    ),
                    f"{float(info.get('wall_sec', 0.0)):.2f}",
                ]
            )
        )
    if overall.get("mean_latency_ms") is not None:
        lines.append("")
        lines.append(
            "OVERALL\t"
            f"frames={overall['total_frames']}\t"
            f"fps={overall['fps']}\t"
            f"mean_ms={overall['mean_latency_ms']}\t"
            f"p95_ms={overall.get('p95_latency_ms', 'n/a')}\t"
            f"p99_ms={overall.get('p99_latency_ms', 'n/a')}\t"
            f"wall_sec={overall['total_wall_sec']}"
        )
    if overall.get("note"):
        lines.append(str(overall["note"]))
    (output_root / "_latency_summary.txt").write_text("\n".join(lines) + "\n")
    return payload


def main() -> int:
    parser = build_parser()
    args, _ = parser.parse_known_args()

    if args.detector and args.detector != "SDP":
        raise ValueError("mot17_all_sdp.py is restricted to SDP detector runs")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    sequences = _discover_sequences(Path(args.data_root), args.split, args.sequences)
    if not sequences:
        raise ValueError("No SDP sequences found to run")

    output_root = Path(args.output)
    per_seq_root = output_root / "_per_seq"
    output_root.mkdir(parents=True, exist_ok=True)
    per_seq_root.mkdir(parents=True, exist_ok=True)

    passthrough = list(sys.argv[1:])
    for flag, takes_value in (
        ("--jobs", True),
        ("--continue-on-error", False),
        ("--dry-run", False),
        ("--sequences", True),
        ("--output", True),
        ("--detector", True),
    ):
        passthrough = _strip_flag(passthrough, flag, takes_value)
    if not _has_flag(sys.argv[1:], "--engine"):
        default_engine = project_root / "models" / "yolo" / "yolo26m_960_batch16.engine"
        if default_engine.exists():
            passthrough += ["--engine", str(default_engine)]

    mot17_script = project_root / "scripts" / "eval" / "mot17.py"
    env = _base_env()
    plan: list[dict[str, object]] = []
    futures = []

    print(
        f"Dispatching {len(sequences)} SDP sequences with jobs={args.jobs} "
        f"into {output_root}"
    )
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(sequences))) as executor:
        for seq in sequences:
            child_output = per_seq_root / seq
            child_output.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(mot17_script),
                "--output",
                str(child_output),
                "--sequences",
                seq,
            ] + passthrough
            plan.append(
                {
                    "sequence": seq,
                    "output": str(child_output),
                    "cmd": cmd,
                }
            )
            futures.append(
                executor.submit(
                    _run_sequence,
                    seq=seq,
                    cmd=cmd,
                    output_dir=output_root,
                    env=env,
                    dry_run=args.dry_run,
                )
            )

        results: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            seq = str(result["sequence"])
            code = int(result["returncode"])
            if code == 0:
                print(f"[OK] {seq}")
            else:
                print(f"[FAIL] {seq} (log: {result['log_path']})")
                failures.append(result)
                if not args.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    break

    (output_root / "_dispatch_plan.json").write_text(json.dumps(plan, indent=2))

    if args.dry_run:
        print("Dry run only; no subprocesses executed.")
        return 0

    if failures:
        (output_root / "_dispatch_failures.json").write_text(
            json.dumps(failures, indent=2)
        )
        if not args.continue_on_error:
            return 1

    completed = sorted(
        item["sequence"] for item in results if int(item["returncode"]) == 0
    )
    for seq in completed:
        src = per_seq_root / seq / f"{seq}.txt"
        dst = output_root / f"{seq}.txt"
        if not src.exists():
            print(f"[MISS] {seq} finished but result file is missing: {src}")
            continue
        shutil.copy2(src, dst)

    latency_summary = _load_latency_summary(output_root, per_seq_root, results)
    overall_latency = latency_summary.get("overall", {})
    if overall_latency:
        print("\n=== LATENCY SUMMARY ===")
        if overall_latency.get("mean_latency_ms") is not None:
            print(f"  Mean latency: {overall_latency['mean_latency_ms']} ms")
            print(f"  FPS: {overall_latency['fps']}")
        if overall_latency.get("p95_latency_ms") is not None:
            print(f"  P95 latency: {overall_latency['p95_latency_ms']} ms")
        if overall_latency.get("p99_latency_ms") is not None:
            print(f"  P99 latency: {overall_latency['p99_latency_ms']} ms")
        print(f"  Total frames: {overall_latency.get('total_frames', 0)}")
        print(f"  Total wall time: {overall_latency.get('total_wall_sec', 0.0)} s")
        if overall_latency.get("note"):
            print(f"  Note: {overall_latency['note']}")

    metrics = run_motmetrics_evaluation(
        data_root=args.data_root,
        split=args.split,
        output=str(output_root),
        sequences=",".join(completed),
        detector="SDP",
    )
    if metrics:
        print("\n=== OVERALL METRICS ===")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    else:
        print("No overall metrics produced.")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
