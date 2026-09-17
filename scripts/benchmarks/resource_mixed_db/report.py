"""Replay double-buffer mixed-workload timing, placement and GPU evidence."""

# status: diagnostic
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmarks.resource_mixed.report import (  # noqa: E402
    STAMP,
    digest,
    quantiles,
    reference,
)
from scripts.benchmarks.resource_partition.green_owner import parse_trace  # noqa: E402
from scripts.benchmarks.resource_partition.routing import (  # noqa: E402
    native_kernel_names,
    source_coverage,
)


def derive(directory):
    point = json.loads((directory / "point.json").read_text())
    args, frames, owner = point["arguments"], point["frames"], point["owner"]
    transitions, windows = point["transitions"], point["windows"]
    checks = {}
    checks["schema"] = point["schema"] == "saccade-mixed-db-point-v1"
    checks["host_clock_pair"] = (
        0 <= point["native_anchor_after"] - point["origin"] <= 0.001
    )
    checks["no_error"] = point["error"] is None
    expected_frames = list(range(51, args["frames"] + 1))
    checks["complete_frames"] = [f["frame"] for f in frames] == expected_frames
    checks["owned_frames"] = all(f["context_owned"] for f in frames)
    checks["frame_timestamps"] = all(
        f["start"] <= f["cycle_finished"] <= f["output"] for f in frames
    )
    checks["double_buffer"] = bool(point["route"].get("enabled"))
    checks["double_buffer_warmup"] = point["route"].get("warmup_frames") == 50
    checks["owned_db_stream"] = bool(point["route"].get("owned_class")) and point[
        "route"
    ].get("stream_handle") in {s["handle"] for s in owner["streams"]}
    checks["owned_streams"] = bool(owner["streams"]) and all(
        s["owned"] for s in owner["streams"]
    )
    checks["owned_finalize"] = owner["main_thread_context_still_owned_at_finalize"]
    checks["thread_hooks"] = owner["thread_context_switch_errors"] == []
    stable_size = 32 if args["policy"] == "shared" else 16
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
        stable_ids == elastic_ids
        if args["policy"] in ("shared", "control")
        else not (stable_ids & elastic_ids) and len(stable_ids | elastic_ids) == 32
    )
    checks["transitions"] = [
        t["scheduled_frame"] for t in transitions
    ] == expected_frames
    checks["transition_timestamps"] = all(
        t["request"] <= t["drained"] <= t["admitted"] for t in transitions
    )
    in_service = [w for w in windows if w["within_stable_service"]]
    tail = [w for w in windows if not w["within_stable_service"]]
    checks["window_count"] = len(in_service) == len(frames) - 1 and len(tail) == (
        0 if args["policy"] == "control" else 1
    )
    checks["window_timestamps"] = all(
        w["release_requested"] <= w["released"] <= w["request"] <= w["drained"]
        for w in windows
    )
    if in_service:
        checks["window_links"] = [w["release_after_frame"] for w in in_service] == list(
            range(50, args["frames"] - 1)
        ) and [w["reclaim_before_scheduled_frame"] for w in in_service] == list(
            range(52, args["frames"] + 1)
        )
    else:
        checks["window_links"] = False

    latency = [(f["output"] - f["start"]) * 1000 for f in frames]
    periods = np.diff([f["output"] for f in frames]) * 1000
    drain = [(w["drained"] - w["request"]) * 1000 for w in in_service]
    transition = [(w["admitted"] - w["request"]) * 1000 for w in in_service]
    open_windows = [(w["request"] - w["released"]) * 1000 for w in in_service]
    lease_windows = [(w["drained"] - w["released"]) * 1000 for w in in_service]
    horizon = frames[-1]["output"] - frames[0]["start"]
    result = dict(
        policy=args["policy"],
        iterations=args["iterations"],
        window=args["window"],
        audit=args["audit"],
        frames=len(frames),
        latency_ms=quantiles(latency),
        period_ms={**quantiles(periods), "std": float(np.std(periods))},
        drain_ms=quantiles(drain),
        transition_ms=quantiles(transition),
        borrow_open_ms=quantiles(open_windows),
        borrow_lease_ms=quantiles(lease_windows),
        initial_admission_ms=(
            (transitions[0]["admitted"] - transitions[0]["request"]) * 1000
        ),
        db_fps=len(frames) / horizon,
        horizon_s=horizon,
        miss_count=sum(v > args["deadline_ms"] for v in latency),
        miss_fraction=sum(v > args["deadline_ms"] for v in latency) / len(frames),
        output_sha256=point["output_sha256"],
    )
    result["service_pass"] = (
        result["latency_ms"]["p99"] <= args["deadline_ms"]
        and result["miss_fraction"] <= 0.01
    )
    if args["policy"] != "control":
        records = np.fromfile(directory / "elastic.records", dtype="<u8").reshape(-1, 4)
        stamps = np.fromfile(directory / "elastic.stamps", dtype=STAMP).reshape(-1, 256)
        expected_total = args["bursts"] * args["units"]
        checks["elastic_count"] = len(records) == len(stamps) == expected_total
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
        period_ns = int(args["burst_period_ms"] * 1_000_000)
        checks["elastic_releases"] = bool(
            (records[:, 2] >= point["native_origin"] + records[:, 0] * period_ns).all()
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
            checks[f"queue_window_{lane}"] = pending == 0 and peak <= args["window"]
        cutoff = (
            point["native_origin"]
            + (frames[-1]["output"] - point["native_anchor_after"]) * 1e9
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
            borrowed_completed_before_cutoff=int(
                ((records[:, 1] == 1) & (records[:, 3] <= cutoff)).sum()
            )
            if args["policy"] == "dynamic"
            else 0,
            borrowed_enqueued_before_cutoff=int(
                ((records[:, 1] == 1) & (records[:, 2] <= cutoff)).sum()
            )
            if args["policy"] == "dynamic"
            else 0,
        )
        burst_latencies = [
            (
                int(records[records[:, 0] == burst, 3].max())
                - point["native_origin"]
                - int(burst) * period_ns
            )
            / 1e6
            for burst in np.unique(records[:, 0])
        ]
        result["burst_completion_ms"] = quantiles(burst_latencies)
    if args["audit"]:
        trace = parse_trace(directory / "cupti.trace")
        begin = next(m["cupti"] for m in trace["labels"] if m["label"] == "mixed_begin")
        end = next(m["cupti"] for m in trace["labels"] if m["label"] == "mixed_end")
        measured = [k for k in trace["kernels"] if begin <= k["start"] <= end]
        stable = [
            k
            for k in measured
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
            args["bursts"] * args["units"] if args["policy"] != "control" else 0
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
            overlap = 0
            j = 0
            for start, stop in borrowed:
                while j < len(stable_intervals) and stable_intervals[j][1] <= start:
                    j += 1
                if j < len(stable_intervals) and stable_intervals[j][0] < stop:
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
        digest(directory / name) == value
        for name, value in point["output_sha256"].items()
    )
    result["checks"] = {name: bool(value) for name, value in checks.items()}
    result["failed_checks"] = [name for name, value in checks.items() if not value]
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
