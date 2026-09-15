"""Validate and summarize same-run throughput, latency, jitter and pressure."""

# status: diagnostic
import argparse
import csv
import hashlib
from datetime import datetime
import json
from pathlib import Path

import numpy as np


def distribution(values):
    arr = np.asarray(values, dtype=float)
    if not len(arr) or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("empty, negative or nonfinite timing samples")
    return {
        "n": len(arr),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        **{f"p{p}_ms": float(np.percentile(arr, p)) for p in (50, 95, 99)},
    }


def summarize_point(
    path, mode, deadline=None, expected_blocks=None, expected_ranges=None
):
    data = json.loads((path / "resource_point.json").read_text())
    if expected_blocks is not None and data["blocks_requested"] != expected_blocks:
        raise ValueError("requested pressure level differs from sweep schedule")
    if data["error"]:
        raise ValueError(f"failed point: {data['error']}")
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
        profile = json.loads((path / f"_latency_profile_{seq}.json").read_text())
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
    row = {
        "measurement_intervals_monotonic": intervals,
        "clock_anchor": data.get("clock_anchor"),
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.glob("MOT17-*.txt")
        },
        "fps": frames / seconds,
        "frames": frames,
        "latency": distribution(latencies),
        "period": distribution(periods),
        "tracker_latency": None,
    }
    if set(row["output_sha256"]) != {f"{seq}.txt" for seq in data["completions"]}:
        raise ValueError("missing or unexpected MOT outputs")
    if deadline is not None:
        row["deadline_ms"] = deadline
        row["deadline_miss_fraction"] = float(np.mean(np.array(latencies) > deadline))
    if data["blocks_requested"]:
        k = data["blocks_requested"]
        if data["blocker_info"]["max_blocks_per_sm"] != 1:
            raise ValueError("blocker occupancy was not verified")
        pulses = [
            p
            for p in data["pulses"]
            if any(a <= p["host_begin"] and p["host_end"] <= b for a, b in intervals)
        ]
        if not pulses:
            raise ValueError("no complete blocker pulses inside measured interval")
        concurrent_ns = 0
        resident_ns = 0
        span_ns = 0
        gaps = []
        verified = 0
        previous_end = None
        for p in pulses:
            blocks = p["blocks"]
            if len(blocks) != k or len({b[0] for b in blocks}) != k:
                raise ValueError("blocker blocks did not occupy K distinct SMs")
            begin = min(b[1] for b in blocks)
            end = max(b[2] for b in blocks)
            simultaneous = min(b[2] for b in blocks) - max(b[1] for b in blocks)
            if simultaneous <= 0:
                raise ValueError("K blocks were not simultaneously resident")
            verified += 1
            concurrent_ns += simultaneous
            resident_ns += sum(b[2] - b[1] for b in blocks)
            span_ns += end - begin
            # Do not bridge sequence setup gaps in the duty denominator.
            if previous_end is not None and p["host_begin"] <= previous_end[1]:
                gap = max(0, begin - previous_end[0])
                gaps.append(gap / 1e6)
                span_ns += gap
            enclosing_end = next(
                b for a, b in intervals if a <= p["host_begin"] and p["host_end"] <= b
            )
            previous_end = (end, enclosing_end)
        coverages = []
        covered_seconds = 0.0
        for a, b in intervals:
            selected = [
                p for p in pulses if a <= p["host_begin"] and p["host_end"] <= b
            ]
            covered = max((p["host_end"] for p in selected), default=a) - min(
                (p["host_begin"] for p in selected), default=a
            )
            coverages.append(covered / (b - a) if b > a else 0.0)
            covered_seconds += covered
        coverage = covered_seconds / sum(b - a for a, b in intervals)
        if min(coverages) < 0.9:
            raise ValueError(
                "blocker pulse span covers less than 90% of a measurement interval"
            )
        row["pressure"] = {
            "selected_pulse_span_fraction_of_measured_time": coverage,
            "verified_pulses": verified,
            "k_simultaneous_fraction_within_selected_pulses": concurrent_ns / span_ns,
            "mean_resident_blocks_within_selected_pulses": resident_ns / span_ns,
            "restart_gap": distribution(gaps) if gaps else None,
            "info": data["blocker_info"],
        }
    return row


def telemetry_summary(path, point):
    anchor = point["clock_anchor"]
    if anchor is None:
        raise ValueError("missing wall/monotonic clock anchor")
    offset = anchor["epoch"] - anchor["monotonic"]
    values = {}
    count = 0
    with path.open() as handle:
        for raw in csv.DictReader(handle, skipinitialspace=True):
            timestamp = datetime.strptime(
                raw["timestamp"], "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
            if not any(
                a + offset <= timestamp <= b + offset
                for a, b in point["measurement_intervals_monotonic"]
            ):
                continue
            count += 1
            for key, value in raw.items():
                if key in {"timestamp", "uuid", "pstate"}:
                    continue
                try:
                    value = float(value.split()[0])
                except (ValueError, IndexError):
                    continue
                if np.isfinite(value):
                    values.setdefault(key, []).append(value)
    if not count:
        raise ValueError("no telemetry samples inside measured intervals")
    return {
        "samples": count,
        "values": {
            key: {"mean": float(np.mean(v)), "min": min(v), "max": max(v)}
            for key, v in values.items()
        },
    }


def summarize(root, deadline=None):
    sweep = json.loads((root / "sweep.json").read_text())
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
    rows = []
    observed = [(e["rep"], e["blocks"], e["mode"]) for e in sweep["schedule"]]
    expected = {
        (rep, k, mode)
        for rep in range(sweep["arguments"]["repeats"])
        for k in sweep["arguments"]["levels"]
        for mode in ("serial", "double")
    }
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("incomplete or duplicate sweep schedule")
    for entry in sweep["schedule"]:
        if entry.get("returncode") != 0:
            raise ValueError("incomplete or failed sweep")
        name = f"r{entry['rep']}_k{entry['blocks']}_{entry['mode']}"
        rows.append(
            {
                "name": name,
                "rep": entry["rep"],
                "blocks": entry["blocks"],
                "mode": entry["mode"],
                **summarize_point(
                    root / name,
                    entry["mode"],
                    deadline,
                    entry["blocks"],
                    expected_ranges,
                ),
            }
        )
    exclusion_path = root / "verification_window.json"
    if exclusion_path.exists():
        exclusion = json.loads(exclusion_path.read_text())
        for row in rows:
            offset = row["clock_anchor"]["epoch"] - row["clock_anchor"]["monotonic"]
            if any(
                a + offset <= exclusion["end_epoch"]
                and b + offset >= exclusion["begin_epoch"]
                for a, b in row["measurement_intervals_monotonic"]
            ):
                raise ValueError(
                    f"{row['name']} overlaps an excluded verification window; rerun required"
                )
    for row in rows:
        row["telemetry"] = telemetry_summary(root / f"{row['name']}.telemetry.csv", row)
    pairs = []
    for rep in range(sweep["arguments"]["repeats"]):
        for k in sweep["arguments"]["levels"]:
            pair = {r["mode"]: r for r in rows if r["rep"] == rep and r["blocks"] == k}
            pairs.append(
                {
                    "rep": rep,
                    "blocks": k,
                    "serial_fps": pair["serial"]["fps"],
                    "double_fps": pair["double"]["fps"],
                    "db_gain": pair["double"]["fps"] / pair["serial"]["fps"],
                }
            )
    for row in rows:
        baseline = next(
            r
            for r in rows
            if r["rep"] == row["rep"] and r["mode"] == row["mode"] and r["blocks"] == 0
        )
        row["output_byte_equal_to_same_mode_baseline"] = (
            row["output_sha256"] == baseline["output_sha256"]
        )
    return {
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema": "saccade-resource-summary-v1",
        "kind": sweep["kind"],
        "rows": rows,
        "pairs": pairs,
        "available_sm_count": None,
        "true_partition_validated": False,
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
