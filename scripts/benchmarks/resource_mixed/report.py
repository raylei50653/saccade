"""Replay mixed-workload timestamps, conservation, placement and CUPTI evidence."""

# status: diagnostic
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmarks.resource_partition.green_owner import parse_trace  # noqa: E402
from scripts.benchmarks.resource_partition.routing import (  # noqa: E402
    native_kernel_names,
    source_coverage,
)

STAMP = np.dtype([("begin", "<u8"), ("end", "<u8"), ("sm", "<u4"), ("value", "<f4")])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    return (
        dict(
            zip(
                ("p50", "p95", "p99", "max"),
                [*np.percentile(values, [50, 95, 99]).tolist(), float(max(values))],
            )
        )
        if len(values)
        else None
    )


def reference(iterations):
    """Independent FP32 FMA and tree-reduction replay, without loading CUDA."""
    values = np.float32(1) + np.arange(256, dtype=np.float32) * np.float32(0.001)
    multiplier, addend = float(np.float32(1.000001)), float(np.float32(0.000001))
    for _ in range(iterations):
        values = (values.astype(np.float64) * multiplier + addend).astype(np.float32)
    for width in (128, 64, 32, 16, 8, 4, 2, 1):
        values[:width] += values[width : 2 * width]
    return float(values[0])


def derive(directory):
    point = json.loads((directory / "point.json").read_text())
    args, frames, owner = point["arguments"], point["frames"], point["owner"]
    checks = {}
    checks["host_clock_pair"] = (
        0 <= point["native_anchor_after"] - point["origin"] <= 0.001
    )
    checks["no_error"] = point["error"] is None
    checks["complete_frames"] = [f["frame"] for f in frames] == list(
        range(51, args["frames"] + 1)
    )
    checks["owned_frames"] = all(f["context_owned"] for f in frames)
    checks["owned_streams"] = bool(owner["streams"]) and all(
        s["owned"] for s in owner["streams"]
    )
    checks["owned_finalize"] = owner["main_thread_context_still_owned_at_finalize"]
    checks["thread_hooks"] = owner["thread_context_switch_errors"] == []
    stable_size = (
        32 if args["policy"] == "shared" else 24 if args["policy"] == "headroom" else 16
    )
    checks["stable_size"] = owner["actual_sm_count"] == stable_size
    stable_ids = set(owner["probe"]["before"]["direct_sm_ids"])
    elastic_ids = set(point["elastic_pool"]["probe"]["direct_sm_ids"])
    checks["stable_probes"] = len(stable_ids) == stable_size and all(
        set(owner["probe"][phase][kind + "_sm_ids"]) == stable_ids
        for phase in ("before", "after")
        for kind in ("direct", "graph")
    )
    checks["elastic_probe"] = (
        len(elastic_ids) == point["elastic_pool"]["actual_sms"]
        and set(point["elastic_pool"]["probe"]["graph_sm_ids"]) == elastic_ids
    )
    checks["envelope"] = (
        (stable_ids == elastic_ids)
        if args["policy"] in ("shared", "control")
        else not (stable_ids & elastic_ids) and len(stable_ids | elastic_ids) == 32
    )
    checks["timestamps"] = all(
        f["arrival"]
        <= f["request"]
        <= f["drained"]
        <= f["admitted"]
        <= f["output"]
        <= f["finished"]
        for f in frames
    )
    checks["arrivals"] = all(
        abs(f["arrival"] - (point["origin"] + (i + 1) * args["period_ms"] / 1000))
        < 1e-8
        for i, f in enumerate(frames)
    )
    # The conservative all-owned-stream completion includes the pipeline's
    # output timestamp and any residual pipeline work before releasing capacity.
    response = [(f["finished"] - f["arrival"]) * 1000 for f in frames]
    output_response = [(f["output"] - f["arrival"]) * 1000 for f in frames]
    drain = [(f["drained"] - f["request"]) * 1000 for f in frames]
    transition = [(f["admitted"] - f["request"]) * 1000 for f in frames]
    horizon = frames[-1]["finished"] - point["origin"]
    result = dict(
        policy=args["policy"],
        iterations=args["iterations"],
        window=args["window"],
        audit=args["audit"],
        frames=len(frames),
        response_ms=quantiles(response),
        output_response_ms=quantiles(output_response),
        drain_ms=quantiles(drain),
        transition_ms=quantiles(transition),
        dispatch_lateness_ms=quantiles(
            [(f["request"] - f["arrival"]) * 1000 for f in frames]
        ),
        stable_fps=len(frames) / horizon,
        horizon_s=horizon,
        miss_count=sum(v > args["deadline_ms"] for v in response),
        miss_fraction=sum(v > args["deadline_ms"] for v in response) / len(frames),
        output_sha256=point["output_sha256"],
    )
    result["service_pass"] = (
        result["response_ms"]["p99"] <= args["deadline_ms"]
        and result["miss_fraction"] <= 0.01
    )
    if args["policy"] != "control":
        records = np.fromfile(directory / "elastic.records", dtype="<u8").reshape(-1, 4)
        stamps = np.fromfile(directory / "elastic.stamps", dtype=STAMP).reshape(-1, 256)
        checks["elastic_count"] = (
            len(records) == len(stamps) == point["bursts"] * args["units"]
        )
        checks["elastic_bursts"] = np.array_equal(
            records[:, 0], np.arange(len(records)) // args["units"]
        )
        checks["elastic_lanes"] = bool(np.isin(records[:, 1], [0, 1]).all())
        checks["reference_matches_cpu"] = point["reference"] == reference(
            args["iterations"]
        )
        checks["elastic_output"] = bool(
            np.isfinite(stamps["value"]).all()
            and (stamps["value"] == np.float32(point["reference"])).all()
        )
        checks["elastic_times"] = bool(
            (records[:, 2] <= records[:, 3]).all()
            and (stamps["begin"] <= stamps["end"]).all()
        )
        checks["elastic_releases"] = bool(
            (
                records[:, 2] >= point["native_origin"] + records[:, 0] * 100_000_000
            ).all()
        )
        checks["elastic_sm_membership"] = all(
            set(stamps["sm"][records[:, 1] == lane].ravel())
            <= (
                stable_ids if args["policy"] == "dynamic" and lane == 1 else elastic_ids
            )
            for lane in (0, 1)
        )
        for lane in (0, 1):
            events = [(int(r[2]), 1) for r in records if r[1] == lane] + [
                (int(r[3]), -1) for r in records if r[1] == lane
            ]
            pending = peak = 0
            for _, delta in sorted(events):
                pending += delta
                peak = max(peak, pending)
            checks[f"window_{lane}"] = pending == 0 and peak <= args["window"]
        cutoff = (
            point["native_origin"]
            + (frames[-1]["finished"] - point["native_anchor_after"]) * 1e9
        )
        completed = int((records[:, 3] <= cutoff).sum())
        last = int(records[:, 3].max())
        result.update(
            elastic_completed=completed,
            elastic_total=len(records),
            elastic_units_s=completed / horizon,
            elastic_all_done_s=(last - point["native_origin"]) / 1e9,
            elastic_drain_after_stable_s=max(0, (last - cutoff) / 1e9),
            borrowed_units=int((records[:, 1] == 1).sum())
            if args["policy"] == "dynamic"
            else 0,
        )
        burst_latencies = [
            (
                int(records[records[:, 0] == b, 3].max())
                - point["native_origin"]
                - int(b) * 100_000_000
            )
            / 1e6
            for b in np.unique(records[:, 0])
        ]
        result["burst_completion_ms"] = quantiles(burst_latencies)
    if args["audit"]:
        trace = parse_trace(directory / "cupti.trace")
        # Exclude only known probes and elastic kernels. All other kernels,
        # including initialization, must execute inside the stable context.
        begin = next(m["cupti"] for m in trace["labels"] if m["label"] == "mixed_begin")
        end = next(m["cupti"] for m in trace["labels"] if m["label"] == "mixed_end")
        measured_kernels = [k for k in trace["kernels"] if begin <= k["start"] <= end]
        stable = [
            k
            for k in measured_kernels
            if "mixed_elastic" not in k["name"] and "smid_kernel" not in k["name"]
        ]
        elastic = [k for k in trace["kernels"] if "mixed_elastic" in k["name"]]
        context_id = owner["cupti_execution_context_id"]
        checks["audit_no_drops"] = trace["dropped"] == 0
        checks["audit_stable_context"] = bool(stable) and all(
            k["context"] == context_id for k in stable
        )
        checks["audit_graphs"] = any(k["graph"] for k in stable)
        checks["audit_elastic_count"] = len(elastic) == (
            point["bursts"] * args["units"] if args["policy"] != "control" else 0
        )
        checks["audit_memsets"] = all(
            k["context"] == context_id
            for k in trace["memsets"]
            if begin <= k["start"] <= end
        )
        allowed_elastic = (
            {context_id, point["elastic_pool"]["cupti_context"]}
            if args["policy"] == "dynamic"
            else {point["elastic_pool"]["cupti_context"]}
        )
        checks["audit_elastic_contexts"] = all(
            k["context"] in allowed_elastic for k in elastic
        )
        if args["policy"] == "dynamic":
            borrowed = sorted(
                (k["start"], k["end"]) for k in elastic if k["context"] == context_id
            )
            stable_intervals = sorted((k["start"], k["end"]) for k in stable)
            j = 0
            overlap = 0
            for start, end in borrowed:
                while j < len(stable_intervals) and stable_intervals[j][1] <= start:
                    j += 1
                if j < len(stable_intervals) and stable_intervals[j][0] < end:
                    overlap += 1
            checks["audit_borrow_no_stable_overlap"] = overlap == 0
            checks["audit_borrow_count"] = len(borrowed) == result["borrowed_units"]
            result["borrowed_stable_kernel_overlaps"] = overlap
        result["kernel_source_coverage"] = source_coverage(
            stable, native_kernel_names()
        )
        result["graph_kernels"] = sum(bool(k["graph"]) for k in stable)
    checks["output_present"] = bool(point["output_sha256"])
    checks["output_hashes"] = all(
        digest(directory / name) == h for name, h in point["output_sha256"].items()
    )
    result["checks"] = {k: bool(v) for k, v in checks.items()}
    result["failed_checks"] = [k for k, v in checks.items() if not v]
    result["valid"] = not result["failed_checks"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    result = derive(args.directory)
    (args.directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
