"""Validate routing evidence and summarize true-partition throughput, latency and jitter."""

# status: diagnostic
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.resource_partition.sweep import point_name  # noqa: E402
from scripts.benchmarks.resource_sensitivity.summarize import (  # noqa: E402
    distribution,
    telemetry_summary,
)


def summarize_point(path, mode, level, deadline=None, expected_ranges=None):
    """Same timing semantics as the proxy summarizer (#35/#376 boundary):
    latency from the production profile, period from completion timestamps,
    FPS from completed frames over the production throughput interval."""
    data = json.loads((path / "partition_point.json").read_text())
    if data["error"]:
        raise ValueError(f"failed point: {data['error']}")
    requested = None if level == "full" else level
    if data["requested_sm_count"] != requested:
        raise ValueError("requested SM budget differs from sweep schedule")
    if not data["completions"]:
        raise ValueError("no completed frames")
    if expected_ranges is not None and set(data["completions"]) != set(expected_ranges):
        raise ValueError("completed sequences differ from requested sequences")
    latencies, periods = [], []
    frames, seconds = 0, 0.0
    intervals = []
    for seq, rows in data["completions"].items():
        if data["double_buffer_enabled"][seq] != (mode == "double"):
            raise ValueError("actual double-buffer route differs from requested mode")
        profile = json.loads(
            (path / "eval" / f"_latency_profile_{seq}.json").read_text()
        )
        ids, times = zip(*rows, strict=True)
        if expected_ranges is not None and list(ids) != list(
            range(*expected_ranges[seq])
        ):
            raise ValueError("completed frame bounds differ from requested bounds")
        if any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("nonconsecutive or duplicate completed frames")
        if len(rows) != profile["frames"] or len(rows) != len(profile["samples_ms"]):
            raise ValueError("latency/completion sample count mismatch")
        latencies.extend(profile["samples_ms"])
        periods.extend(np.diff(times) * 1000)
        frames += profile["frames"]
        seconds += profile["throughput_seconds"]
        intervals.append((times[0], times[-1]))
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("nonpositive throughput interval")
    outputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (path / "eval").glob("MOT17-*.txt")
    }
    if set(outputs) != {f"{seq}.txt" for seq in data["completions"]}:
        raise ValueError("missing or unexpected MOT outputs")
    owner = data.get("owner") or {}
    audit = data.get("audit") or {}
    row = {
        "measurement_intervals_monotonic": intervals,
        "clock_anchor": data.get("clock_anchor"),
        "output_sha256": outputs,
        "fps": frames / seconds,
        "frames": frames,
        "latency": distribution(latencies),
        "period": distribution(periods),
        "tracker_latency": None,
        "partition_kind": data["partition_kind"],
        "requested_sm_count": data["requested_sm_count"],
        "actual_sm_count": data["actual_sm_count"],
        "device_sm_count": owner.get("device_sm_count"),
        "audit_enabled": data["audit_enabled"],
        "owner_checks_passed": bool(data["owner_checks_passed"]),
        "audit_checks_passed": bool(data["audit_checks_passed"]),
        "point_true_partition_validated": bool(data["true_partition_validated"]),
        "point_routing_validated": bool(data["routing_validated"]),
        "failed_checks": data["failed_checks"],
        "routing_verification": data["routing_verification"],
        "routing_evidence": {
            "green_context": owner.get("green_context"),
            "execution_context": owner.get("execution_context"),
            "cupti_execution_context_id": owner.get("cupti_execution_context_id"),
            "main_stream": owner.get("main_stream"),
            "owned_streams": [s["handle"] for s in owner.get("streams", [])],
            "double_buffer_stream": {
                seq: r["double_buffer_stream_handle"]
                for seq, r in data["routes"].items()
            },
            "thread_context_switches": owner.get("thread_context_switches"),
            "set_device_reasserts": owner.get("set_device_reasserts"),
            "probe_sm_ids": owner.get("probe"),
            "audit_windows": {
                seq: {
                    "sequence": w["sequence"],
                    "measured": w["measured"],
                    "source_coverage_heuristic": w["source_coverage_heuristic"],
                }
                for seq, w in (audit.get("windows") or {}).items()
            },
            "setup_before_first_sequence": audit.get("setup_before_first_sequence"),
            "audit_contexts": audit.get("contexts"),
            "audit_dropped": audit.get("dropped"),
        },
    }
    if deadline is not None:
        row["deadline_ms"] = deadline
        row["deadline_miss_fraction"] = float(np.mean(np.array(latencies) > deadline))
    return row


def summarize(root, deadline=None):
    sweep = json.loads((root / "sweep.json").read_text())
    if sweep["kind"] != "green_context_true_partition":
        raise ValueError("not a true-partition sweep")
    sequence_lengths = None
    for filename in ("input_identity_before.json", "input_identity_after.json"):
        candidate = root / filename
        if candidate.exists():
            sequence_lengths = json.loads(candidate.read_text()).get("sequence_lengths")
            if sequence_lengths:
                break
    if not sequence_lengths:
        raise ValueError("missing input sequence bounds identity")
    expected_ranges = {
        seq: (51, min(sweep["arguments"]["max_frames"] or length, length) + 1)
        for seq, length in sequence_lengths.items()
    }
    if set(expected_ranges) != set(sweep["arguments"]["sequences"].split(",")):
        raise ValueError("input identity sequences differ from sweep")
    levels = sweep["arguments"]["levels"]
    observed = [
        (e["rep"], str(e["level"]), e["mode"], e["audit"]) for e in sweep["schedule"]
    ]
    expected = {
        (rep, str(level), mode, audit)
        for rep in range(sweep["arguments"]["repeats"])
        for level in levels
        for mode in ("serial", "double")
        for audit in (True, False)
    }
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("incomplete or duplicate sweep schedule")
    points = {}
    for entry in sweep["schedule"]:
        if entry.get("returncode") != 0:
            raise ValueError("incomplete or failed sweep")
        name = point_name(entry)
        points[name] = {
            "name": name,
            "rep": entry["rep"],
            "level": entry["level"],
            "mode": entry["mode"],
            "audit": entry["audit"],
            **summarize_point(
                root / name,
                entry["mode"],
                entry["level"],
                deadline,
                expected_ranges,
            ),
        }
    rows = []
    for point in points.values():
        if point["audit"]:
            continue
        twin = points[point["name"] + "_audit"]
        row = dict(point)
        row["telemetry"] = telemetry_summary(root / f"{row['name']}.telemetry.csv", row)
        consistency = {
            "output_sha256_equal": twin["output_sha256"] == point["output_sha256"],
            "owned_stream_count_equal": len(twin["routing_evidence"]["owned_streams"])
            == len(point["routing_evidence"]["owned_streams"]),
            "thread_context_switches_equal": twin["routing_evidence"][
                "thread_context_switches"
            ]
            == point["routing_evidence"]["thread_context_switches"],
            "set_device_reasserts_equal": twin["routing_evidence"][
                "set_device_reasserts"
            ]
            == point["routing_evidence"]["set_device_reasserts"],
            "actual_sm_count_equal": twin["actual_sm_count"]
            == point["actual_sm_count"],
            "double_buffer_route_equal": twin["routing_evidence"][
                "double_buffer_stream"
            ].keys()
            == point["routing_evidence"]["double_buffer_stream"].keys(),
        }
        row["audit_twin"] = {
            "name": twin["name"],
            "fps": twin["fps"],
            "fps_measurement_over_audit": point["fps"] / twin["fps"],
            "frame_p99_ms": twin["latency"]["p99_ms"],
            "owner_checks_passed": twin["owner_checks_passed"],
            "audit_checks_passed": twin["audit_checks_passed"],
            "point_true_partition_validated": twin["point_true_partition_validated"],
            "point_routing_validated": twin["point_routing_validated"],
            "failed_checks": twin["failed_checks"],
            "routing_evidence": twin["routing_evidence"],
            "consistency_with_measurement": consistency,
        }
        green = point["level"] != "full"
        routed = (
            point["owner_checks_passed"]
            and twin["point_routing_validated"]
            and all(consistency.values())
        )
        row["routing_validated"] = routed
        row["true_partition_validated"] = green and routed
        row["curve_eligible"] = routed
        rows.append(row)
    pairs = []
    for rep in range(sweep["arguments"]["repeats"]):
        for level in levels:
            pair = {
                r["mode"]: r for r in rows if r["rep"] == rep and r["level"] == level
            }
            pairs.append(
                {
                    "rep": rep,
                    "level": level,
                    "actual_sm_count": pair["serial"]["actual_sm_count"],
                    "partition_kind": pair["serial"]["partition_kind"],
                    "serial_fps": pair["serial"]["fps"],
                    "double_fps": pair["double"]["fps"],
                    "db_gain": pair["double"]["fps"] / pair["serial"]["fps"],
                    "serial_frame_p99_ms": pair["serial"]["latency"]["p99_ms"],
                    "double_frame_p99_ms": pair["double"]["latency"]["p99_ms"],
                    "serial_period_std_ms": pair["serial"]["period"]["std_ms"],
                    "double_period_std_ms": pair["double"]["period"]["std_ms"],
                    "true_partition_validated": pair["serial"][
                        "true_partition_validated"
                    ]
                    and pair["double"]["true_partition_validated"],
                    "curve_eligible": pair["serial"]["curve_eligible"]
                    and pair["double"]["curve_eligible"],
                }
            )
    for row in rows:
        baseline = next(
            r
            for r in rows
            if r["rep"] == row["rep"]
            and r["mode"] == row["mode"]
            and r["level"] == "full"
        )
        row["output_byte_equal_to_same_mode_full_baseline"] = (
            row["output_sha256"] == baseline["output_sha256"]
        )
    not_validated = sorted(r["name"] for r in rows if not r["routing_validated"])
    partition_rows = [r for r in rows if r["level"] != "full"]
    return {
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema": "saccade-partition-summary-v1",
        "kind": sweep["kind"],
        "rows": rows,
        "pairs": pairs,
        "audit_overhead": [
            {
                "name": r["name"],
                "level": r["level"],
                "mode": r["mode"],
                "fps_measurement_over_audit": r["audit_twin"][
                    "fps_measurement_over_audit"
                ],
            }
            for r in rows
        ],
        "validated_actual_sm_counts": sorted(
            {
                r["actual_sm_count"]
                for r in partition_rows
                if r["true_partition_validated"]
            }
        ),
        "points_not_validated": not_validated,
        "true_partition_validated": bool(partition_rows)
        and all(r["true_partition_validated"] for r in partition_rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--deadline-ms", type=float)
    args = parser.parse_args()
    if args.deadline_ms is not None and (
        not np.isfinite(args.deadline_ms) or args.deadline_ms <= 0
    ):
        parser.error("deadline must be positive and finite")
    result = summarize(args.root, args.deadline_ms)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for pair in result["pairs"]:
        print(pair)
    if result["points_not_validated"]:
        print("routing verification failed:", result["points_not_validated"])
        sys.exit(2)
