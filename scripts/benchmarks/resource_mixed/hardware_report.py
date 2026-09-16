"""Relate the sealed mixed frontier to CUPTI timelines and block residency."""

# status: diagnostic
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmarks.resource_mixed.report import STAMP
from scripts.benchmarks.resource_partition.green_owner import parse_trace

POLICIES = ("fixed", "shared", "dynamic")
ITERATIONS = (2048, 8192)
THEORETICAL_BLOCKS_PER_SM = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if begin >= end:
            continue
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intersect_intervals(left, right):
    overlap = []
    i = j = 0
    while i < len(left) and j < len(right):
        begin, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if begin < end:
            overlap.append([begin, end])
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def interval_ns(intervals):
    return sum(end - begin for begin, end in intervals)


def clipped(kernel, begin, end):
    return max(begin, kernel["start"]), min(end, kernel["end"])


def quantiles(values):
    return dict(
        zip(
            ("p50", "p95", "p99", "max"),
            [*np.percentile(values, [50, 95, 99]).tolist(), float(max(values))],
        )
    )


def sm_residency(intervals):
    timestamps = np.concatenate([intervals["begin"], intervals["end"]])
    changes = np.concatenate(
        [
            np.ones(len(intervals), dtype=np.int8),
            -np.ones(len(intervals), dtype=np.int8),
        ]
    )
    # End before begin at an identical clock avoids a false instantaneous overlap.
    order = np.lexsort((changes, timestamps))
    timestamps, changes = timestamps[order], changes[order]
    unique, indices = np.unique(timestamps, return_index=True)
    resident = np.cumsum(np.add.reduceat(changes, indices))[:-1]
    durations = np.diff(unique)
    active_cycles = int(durations[resident > 0].sum())
    require(active_cycles > 0, "empty per-SM residency")
    return {
        "peak_blocks": int(resident.max()),
        "active_cycles": active_cycles,
        "resident_block_cycles": int((resident * durations).sum()),
        "full_residency_cycles": int(
            durations[resident == THEORETICAL_BLOCKS_PER_SM].sum()
        ),
    }


def pool_residency(stamps, records, lanes):
    blocks = stamps[np.isin(records[:, 1], lanes)].ravel()
    per_sm = [
        sm_residency(blocks[blocks["sm"] == sm]) for sm in np.unique(blocks["sm"])
    ]
    active_cycles = sum(row["active_cycles"] for row in per_sm)
    mean_blocks = sum(row["resident_block_cycles"] for row in per_sm) / active_cycles
    full_cycles = sum(row["full_residency_cycles"] for row in per_sm)
    return {
        "lanes": lanes,
        "blocks": len(blocks),
        "observed_sms": len(per_sm),
        "peak_blocks_per_sm": max(row["peak_blocks"] for row in per_sm),
        "active_sm_cycle_weighted_mean_blocks": mean_blocks,
        "active_sm_cycle_weighted_residency_fraction": mean_blocks
        / THEORETICAL_BLOCKS_PER_SM,
        "full_residency_fraction_of_active_sm_cycles": full_cycles / active_cycles,
    }


def cup_timeline(directory):
    point = json.loads((directory / "point.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    require(report["valid"] and point["arguments"]["audit"], f"invalid {directory}")
    trace = parse_trace(directory / "cupti.trace")
    mark = next(m for m in point["owner"]["marks"] if m["label"] == "mixed_begin")
    # Pair the host and CUPTI clocks at mixed_begin, then stay within each clock.
    begin = mark["cupti"] - round((mark["host"] - point["origin"]) * 1e9)
    end = begin + round((point["frames"][-1]["finished"] - point["origin"]) * 1e9)
    kernels = [
        kernel
        for kernel in trace["kernels"]
        if kernel["end"] > begin
        and kernel["start"] < end
        and "smid_kernel" not in kernel["name"]
    ]
    stable_kernels = [k for k in kernels if "mixed_elastic" not in k["name"]]
    elastic_kernels = [k for k in kernels if "mixed_elastic" in k["name"]]
    require(stable_kernels and elastic_kernels, f"missing mixed kernels in {directory}")
    stable = merge_intervals(clipped(k, begin, end) for k in stable_kernels)
    elastic = merge_intervals(clipped(k, begin, end) for k in elastic_kernels)
    overlap = intersect_intervals(stable, elastic)
    horizon = end - begin
    stable_ns, elastic_ns, overlap_ns = map(interval_ns, (stable, elastic, overlap))
    durations = np.array([(k["end"] - k["start"]) / 1e6 for k in elastic_kernels])
    result = {
        "policy": report["policy"],
        "iterations": report["iterations"],
        "window": report["window"],
        "horizon_s": horizon / 1e9,
        "stable_kernel_present_ms": stable_ns / 1e6,
        "stable_kernel_present_fraction": stable_ns / horizon,
        "elastic_kernel_present_ms": elastic_ns / 1e6,
        "elastic_kernel_present_fraction": elastic_ns / horizon,
        "stable_elastic_overlap_ms": overlap_ns / 1e6,
        "stable_fraction_with_elastic_kernel": overlap_ns / stable_ns,
        "any_kernel_present_fraction": interval_ns(merge_intervals(stable + elastic))
        / horizon,
        "elastic_kernel_duration_ms": quantiles(durations),
        "stable_sms": point["owner"]["actual_sm_count"],
        "elastic_sms": point["elastic_pool"]["actual_sms"],
    }
    records = np.fromfile(directory / "elastic.records", dtype="<u8").reshape(-1, 4)
    stamps = np.fromfile(directory / "elastic.stamps", dtype=STAMP).reshape(-1, 256)
    expected_sets = {
        0: set(point["elastic_pool"]["probe"]["direct_sm_ids"]),
        1: set(point["owner"]["probe"]["before"]["direct_sm_ids"]),
    }
    lane_coverage = {}
    for lane in (0, 1):
        ids = set(map(int, np.unique(stamps["sm"][records[:, 1] == lane])))
        expected = expected_sets[
            1 if report["policy"] == "dynamic" and lane == 1 else 0
        ]
        lane_coverage[str(lane)] = {
            "units": int((records[:, 1] == lane).sum()),
            "observed_sm_count": len(ids),
            "expected_sm_count": len(expected),
            "exact_expected_set": ids == expected,
        }
    require(
        all(row["exact_expected_set"] for row in lane_coverage.values()),
        f"incomplete elastic SM coverage in {directory}",
    )
    result["elastic_lane_coverage"] = lane_coverage
    groups = (
        {"shared_elastic_pool": [0, 1]}
        if report["policy"] in ("fixed", "shared")
        else {"permanent_elastic_pool": [0], "borrowed_stable_pool": [1]}
    )
    result["elastic_block_residency"] = {
        name: pool_residency(stamps, records, lanes) for name, lanes in groups.items()
    }
    if report["policy"] == "dynamic":
        stable_context = point["owner"]["cupti_execution_context_id"]
        borrowed_kernels = [
            k for k in elastic_kernels if k["context"] == stable_context
        ]
        borrowed = merge_intervals(clipped(k, begin, end) for k in borrowed_kernels)
        borrowed_ns = interval_ns(borrowed)
        borrowed_overlap_ns = interval_ns(intersect_intervals(stable, borrowed))
        result.update(
            borrowed_units=len(borrowed_kernels),
            borrowed_unit_fraction=len(borrowed_kernels) / len(elastic_kernels),
            borrowed_kernel_present_ms=borrowed_ns / 1e6,
            borrowed_stable_overlap_ns=borrowed_overlap_ns,
            stable_pool_kernel_present_fraction=interval_ns(
                merge_intervals(stable + borrowed)
            )
            / horizon,
            borrowed_fraction_of_stable_kernel_free_time=borrowed_ns
            / (horizon - stable_ns),
        )
        require(
            borrowed_overlap_ns == 0
            and len(borrowed_kernels) == report["borrowed_units"],
            f"borrow replay mismatch in {directory}",
        )
    return result


def launch_window(directory):
    report = json.loads((directory / "report.json").read_text())
    records = np.fromfile(directory / "elastic.records", dtype="<u8").reshape(-1, 4)
    gaps_us = []
    for burst in np.unique(records[:, 0]):
        for lane in (0, 1):
            lane_rows = records[(records[:, 0] == burst) & (records[:, 1] == lane)]
            gaps_us.extend(
                (lane_rows[1:, 2].astype(np.int64) - lane_rows[:-1, 3].astype(np.int64))
                / 1e3
            )
    gaps_us = np.asarray(gaps_us)
    return {
        "iterations": report["iterations"],
        "window": report["window"],
        "within_burst_successor_pairs": len(gaps_us),
        "successor_enqueued_before_prior_completion_fraction": float(
            np.mean(gaps_us < 0)
        ),
        "audit_transition_p99_ms": report["transition_ms"]["p99"],
        "audit_burst_completion_p95_ms": report["burst_completion_ms"]["p95"],
        "borrowed_units": report["borrowed_units"],
    }


def build(original_root):
    sweep = json.loads((original_root / "sweep.json").read_text())
    timelines = []
    for iterations in ITERATIONS:
        for policy in POLICIES:
            timelines.append(
                cup_timeline(original_root / f"r0_{policy}_i{iterations}_w4_audit")
            )
    launch_windows = [
        launch_window(original_root / f"r0_dynamic_i{iterations}_w{window}_audit")
        for iterations in ITERATIONS
        for window in (1, 4)
    ]
    return {
        "schema": "saccade-resource-hardware-v1",
        "frontier_source_head": sweep["head"],
        "original_archive": str(original_root),
        "elastic_kernel_limits": {
            "threads_per_block": 256,
            "registers_per_thread": 14,
            "static_shared_memory_bytes": 2048,
            "device_max_threads_per_sm": 1536,
            "device_max_blocks_per_sm": 24,
            "device_registers_per_sm": 65536,
            "device_shared_memory_per_sm_bytes": 102400,
            "theoretical_blocks_per_sm": THEORETICAL_BLOCKS_PER_SM,
            "limiter": "threads",
        },
        "cupti_timelines": timelines,
        "dynamic_launch_windows": launch_windows,
        "verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_archive", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = build(args.original_archive.resolve())
    text = json.dumps(result, indent=2) + "\n"
    if args.json_output:
        args.json_output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
