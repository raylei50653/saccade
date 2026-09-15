"""The true-partition reporter must fail closed on any routing escape."""

# scope: eval
# function: contract
# lifecycle: active

import json
from pathlib import Path

import pytest

from scripts.benchmarks.resource_partition.green_owner import (
    audit_window,
    host_to_cupti,
    parse_trace,
)
from scripts.benchmarks.resource_partition.routing import build_point_report
from scripts.benchmarks.resource_partition.summarize import summarize_point

GREEN_ID = 2
PRIMARY_ID = 1


def owner_evidence(*, actual=16, green=True, streams=3, owned=True):
    ctx = 0xB000 if green else 0xA000
    return {
        "device_sm_count": 46,
        "min_partition": 8,
        "alignment": 8,
        "requested_sm_count": actual if green else None,
        "actual_sm_count": actual,
        "primary_context": 0xA000,
        "execution_context": ctx,
        "green_context": 0xC000 if green else None,
        "cupti_primary_context_id": PRIMARY_ID,
        "cupti_execution_context_id": GREEN_ID if green else PRIMARY_ID,
        "main_thread_context_still_owned_at_finalize": True,
        "priority_range": [0, -5],
        "main_stream": 0x100,
        "streams": [
            {
                "handle": 0x100 + i,
                "priority": 0,
                "thread": 1,
                "created_at": 0.0,
                "context": ctx,
                "green_context": (0xC000 if green else 0) if owned else 0,
                "owned": owned,
            }
            for i in range(streams)
        ],
        "thread_context_switches": 2,
        "thread_context_switch_errors": [],
        "set_device_reasserts": 5,
        "marks": [],
        # host seconds -> CUPTI ns with a constant 1e12 offset
        "calibration": [(1.0, 1e12 + 1e9), (3.0, 1e12 + 3e9)],
        "probe": {
            phase: {
                "direct_sm_ids": list(range(actual)),
                "graph_sm_ids": list(range(actual)),
            }
            for phase in ("before", "after")
        },
        "trace_path": "trace",
    }


def point_dict(*, green=True, double=True, actual=16, error=None, audit=True):
    return {
        "schema": "saccade-partition-point-v1",
        "kind": "green_context_true_partition"
        if green
        else "primary_full_device_baseline",
        "clock_anchor": {"epoch": 0, "monotonic": 0},
        "requested_sm_count": actual if green else None,
        "audit_enabled": audit,
        "owner": owner_evidence(actual=actual, green=green),
        "routes": {
            "MOT17-04-SDP": {
                "double_buffer_enabled": double,
                "double_buffer_stream_handle": 0x101 if double else None,
                "double_buffer_stream_owned_class": True if double else None,
                "warmup_frames": 1,
            }
        },
        # frame 1 (warmup) starts at host 1.5 s, frame 2 at 1.7 s
        "frame_starts": {"MOT17-04-SDP": [[1, 1.5], [2, 1.7], [3, 1.9]]},
        "completions": {"MOT17-04-SDP": [[2, 1.8], [3, 2.0]]},
        "frames_completed_with_unowned_main_thread_context": [],
        "double_buffer_enabled": {"MOT17-04-SDP": double},
        "error": error,
    }


def write_trace(path, kernels, extra=""):
    """kernels: list of (start_host_seconds, context_id, graph_id, name)."""
    lines = [
        "A green_context_kind enabled",
        f"C {PRIMARY_ID} 0 7 0 0 0",
        f"G {GREEN_ID} {PRIMARY_ID} 0 8 16 255",
        "N 0 predict_kernel",
        "N 1 sm120_trt_kernel",
    ]
    for i, (start, ctx, graph, name) in enumerate(kernels):
        name_id = {"predict_kernel": 0, "sm120_trt_kernel": 1}[name]
        ts = int(1e12 + start * 1e9)
        lines.append(f"K {ts} {ts + 1000} {ctx} 15 {graph} {i} {name_id} 0")
    lines.append(extra)
    lines.append("E dropped=0")
    path.write_text("\n".join(lines) + "\n")


def good_kernels():
    return [
        (1.55, GREEN_ID, 0, "predict_kernel"),
        (1.75, GREEN_ID, 3, "sm120_trt_kernel"),
        (1.95, GREEN_ID, 3, "predict_kernel"),
    ]


def test_validated_when_every_sequence_kernel_runs_in_the_green_context(tmp_path):
    trace = tmp_path / "trace"
    write_trace(trace, good_kernels())
    report = build_point_report(point_dict(), trace)
    assert report["failed_checks"] == []
    assert report["true_partition_validated"] is True
    window = report["audit"]["windows"]["MOT17-04-SDP"]
    assert window["sequence"]["kernels"] == {
        "total": 3,
        "in_execution_context": 3,
        "outside": 0,
        "graph_launched": 2,
        "on_null_stream": None,
    }
    assert window["measured"]["kernels"]["total"] == 2
    assert window["source_coverage_heuristic"] == {"native": 2, "tensorrt": 1}


def test_one_escaped_kernel_fails_closed_and_is_localized(tmp_path):
    trace = tmp_path / "trace"
    kernels = good_kernels() + [(1.85, PRIMARY_ID, 0, "sm120_trt_kernel")]
    write_trace(trace, kernels)
    report = build_point_report(point_dict(), trace)
    assert report["true_partition_validated"] is False
    assert "sequence_kernels_all_in_execution_context" in report["failed_checks"]
    escapes = report["audit"]["windows"]["MOT17-04-SDP"]["sequence"]["kernel_escapes"]
    assert escapes == [
        {"context": PRIMARY_ID, "stream": 15, "name": "sm120_trt_kernel", "count": 1}
    ]


def test_setup_escape_before_the_sequence_is_reported_not_fatal(tmp_path):
    trace = tmp_path / "trace"
    write_trace(trace, [(0.5, PRIMARY_ID, 0, "sm120_trt_kernel")] + good_kernels())
    report = build_point_report(point_dict(), trace)
    assert report["true_partition_validated"] is True
    assert report["audit"]["setup_before_first_sequence"]["kernels"]["outside"] == 1


@pytest.mark.parametrize(
    "mutation, check",
    [
        (
            lambda p: p["owner"]["streams"][1].update(owned=False, green_context=0),
            "all_created_streams_owned",
        ),
        (
            lambda p: p["routes"]["MOT17-04-SDP"].update(
                double_buffer_stream_handle=0x999
            ),
            "double_buffer_stream_owned",
        ),
        (
            lambda p: p["routes"]["MOT17-04-SDP"].update(
                double_buffer_stream_owned_class=False
            ),
            "double_buffer_stream_owned",
        ),
        (
            lambda p: p.update(
                frames_completed_with_unowned_main_thread_context=[["MOT17-04-SDP", 2]]
            ),
            "main_thread_context_owned_at_every_completion",
        ),
        (
            lambda p: p["owner"].update(
                main_thread_context_still_owned_at_finalize=False
            ),
            "main_thread_context_owned_at_finalize",
        ),
        (
            lambda p: p["owner"]["probe"]["after"].update(graph_sm_ids=list(range(46))),
            "probe_after_graph_sm_count_matches_actual",
        ),
        (
            lambda p: p["owner"].update(actual_sm_count=24),
            "actual_sm_count_matches_requested",
        ),
        (
            lambda p: p["owner"].update(thread_context_switch_errors=["boom"]),
            "no_thread_context_switch_errors",
        ),
        (lambda p: p.update(error="RuntimeError('x')"), "eval_error_none"),
    ],
)
def test_owner_evidence_failures_fail_closed(tmp_path, mutation, check):
    trace = tmp_path / "trace"
    write_trace(trace, good_kernels())
    point = point_dict()
    mutation(point)
    report = build_point_report(point, trace)
    assert check in report["failed_owner_checks"]
    assert report["true_partition_validated"] is False


def test_dropped_records_and_missing_green_record_fail_closed(tmp_path):
    trace = tmp_path / "trace"
    write_trace(trace, good_kernels(), extra="D 3")
    report = build_point_report(point_dict(), trace)
    assert "audit_no_dropped_records" in report["failed_audit_checks"]
    lines = trace.read_text().splitlines()
    trace.write_text(
        "\n".join(line for line in lines if not line.startswith("G ")) + "\n"
    )
    report = build_point_report(point_dict(), trace)
    assert "audit_execution_context_record_present" in report["failed_audit_checks"]
    assert report["true_partition_validated"] is False


def test_unaudited_point_passes_owner_checks_but_is_never_validated():
    report = build_point_report(point_dict(audit=False), None)
    assert report["owner_checks_passed"] is True
    assert report["audit_checks_passed"] is False
    assert report["true_partition_validated"] is False


def test_primary_full_device_baseline_is_routed_but_never_a_partition(tmp_path):
    trace = tmp_path / "trace"
    write_trace(trace, [(s, PRIMARY_ID, g, n) for s, _, g, n in good_kernels()])
    report = build_point_report(point_dict(green=False, actual=46), trace)
    assert report["routing_validated"] is True
    assert report["true_partition_validated"] is False
    assert report["partition_kind"] == "primary_full_device_baseline"


def test_trace_parser_reads_green_context_records_and_clock_offset(tmp_path):
    trace = tmp_path / "trace"
    write_trace(trace, good_kernels())
    parsed = parse_trace(trace)
    green = next(c for c in parsed["contexts"] if c["context"] == GREEN_ID)
    assert (
        green["is_green"] == 1
        and green["sm_count"] == 16
        and green["tpc_mask"] == [255]
    )
    assert parsed["dropped"] == 0
    cupti, spread = host_to_cupti([(1.0, 1e12 + 1e9), (3.0, 1e12 + 3e9)], 2.0)
    assert cupti == pytest.approx(1e12 + 2e9)
    assert spread == 0
    summary = audit_window(parsed, GREEN_ID, 0, 2e12, 142)
    assert summary["kernels"]["in_execution_context"] == 3


def write_run(
    root, name, *, green=True, double=True, actual=16, kernels=None, output="a"
):
    point_dir = root / name
    (point_dir / "eval").mkdir(parents=True)
    audit = name.endswith("_audit")
    if audit:
        write_trace(point_dir / "trace", kernels or good_kernels())
    point = point_dict(green=green, double=double, actual=actual, audit=audit)
    report = build_point_report(point, point_dir / "trace" if audit else None)
    (point_dir / "partition_point.json").write_text(json.dumps(report))
    profile = {"frames": 2, "samples_ms": [50, 70], "throughput_seconds": 0.02}
    (point_dir / "eval" / "_latency_profile_MOT17-04-SDP.json").write_text(
        json.dumps(profile)
    )
    (point_dir / "eval" / "MOT17-04-SDP.txt").write_text(output)
    return point_dir


def test_summarize_point_keeps_timing_semantics_and_route_check(tmp_path):
    path = write_run(tmp_path, "r0_sm16_double_audit")
    row = summarize_point(path, "double", 16, deadline=60)
    assert row["fps"] == 100  # completed frames / production throughput interval
    assert row["latency"]["p50_ms"] == 60
    assert row["period"]["n"] == 1 and row["period"]["p50_ms"] == pytest.approx(200)
    assert row["deadline_miss_fraction"] == pytest.approx(0.5)
    assert row["tracker_latency"] is None
    assert row["point_true_partition_validated"] is True
    with pytest.raises(ValueError, match="route"):
        summarize_point(path, "serial", 16)
    with pytest.raises(ValueError, match="requested SM budget"):
        summarize_point(path, "double", 24)


def test_scripts_are_diagnostic_and_documented():
    here = Path(__file__).resolve().parents[3] / "scripts/benchmarks/resource_partition"
    for script in here.glob("*.py"):
        assert "# status: diagnostic" in script.read_text(), script
    readme = (here / "README.md").read_text()
    assert "true_partition_validated" in readme


def write_sweep(root, *, twin_output="a", twin_kernels=None):
    from datetime import datetime

    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for level in ("full", 16):
        for mode in ("serial", "double"):
            for audit in (True, False):
                entries.append(
                    {
                        "rep": 0,
                        "level": level,
                        "mode": mode,
                        "audit": audit,
                        "returncode": 0,
                    }
                )
    (root / "sweep.json").write_text(
        json.dumps(
            {
                "kind": "green_context_true_partition",
                "arguments": {
                    "levels": ["full", 16],
                    "repeats": 1,
                    "max_frames": 52,
                    "sequences": "MOT17-04-SDP",
                },
                "schedule": entries,
            }
        )
    )
    (root / "input_identity_before.json").write_text(
        json.dumps({"sequence_lengths": {"MOT17-04-SDP": 1050}})
    )
    stamp = datetime.fromtimestamp(1.9).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    for entry in entries:
        name = f"r0_sm{entry['level']}_{entry['mode']}" + (
            "_audit" if entry["audit"] else ""
        )
        green = entry["level"] != "full"
        kernels = None
        if entry["audit"] and green and twin_kernels is not None:
            kernels = twin_kernels
        if entry["audit"] and not green:
            kernels = [(s, PRIMARY_ID, g, n) for s, _, g, n in good_kernels()]
        point_dir = root / name
        (point_dir / "eval").mkdir(parents=True)
        if entry["audit"]:
            write_trace(point_dir / "trace", kernels or good_kernels())
        point = point_dict(
            green=green,
            double=entry["mode"] == "double",
            actual=16 if green else 46,
            audit=entry["audit"],
        )
        point["routes"]["MOT17-04-SDP"]["warmup_frames"] = 50
        point["frame_starts"] = {"MOT17-04-SDP": [[1, 1.5], [51, 1.7], [52, 1.9]]}
        point["completions"] = {"MOT17-04-SDP": [[51, 1.8], [52, 2.0]]}
        report = build_point_report(
            point, point_dir / "trace" if entry["audit"] else None
        )
        (point_dir / "partition_point.json").write_text(json.dumps(report))
        (point_dir / "eval" / "_latency_profile_MOT17-04-SDP.json").write_text(
            json.dumps(
                {"frames": 2, "samples_ms": [50, 70], "throughput_seconds": 0.02}
            )
        )
        output = twin_output if (entry["audit"] and green) else "a"
        (point_dir / "eval" / "MOT17-04-SDP.txt").write_text(output)
        (root / f"{name}.telemetry.csv").write_text(
            f"timestamp, uuid, pstate, power.draw [W]\n{stamp}, GPU-x, P0, 100.00 W\n"
        )


def test_summary_validates_only_consistent_audited_twins(tmp_path):
    from scripts.benchmarks.resource_partition.summarize import summarize

    write_sweep(tmp_path)
    result = summarize(tmp_path)
    assert result["true_partition_validated"] is True
    assert result["validated_actual_sm_counts"] == [16]
    assert result["points_not_validated"] == []
    pair = next(p for p in result["pairs"] if p["level"] == 16)
    assert pair["db_gain"] == pytest.approx(1.0)
    assert pair["true_partition_validated"] is True
    full = next(r for r in result["rows"] if r["level"] == "full")
    assert (
        full["routing_validated"] is True and full["true_partition_validated"] is False
    )


def test_summary_fails_closed_when_twin_output_or_audit_differs(tmp_path):
    from scripts.benchmarks.resource_partition.summarize import summarize

    write_sweep(tmp_path / "output", twin_output="b")
    result = summarize(tmp_path / "output")
    assert result["true_partition_validated"] is False
    assert set(result["points_not_validated"]) == {"r0_sm16_serial", "r0_sm16_double"}
    escaped = good_kernels() + [(1.85, PRIMARY_ID, 0, "sm120_trt_kernel")]
    write_sweep(tmp_path / "escape", twin_kernels=escaped)
    result = summarize(tmp_path / "escape")
    assert result["true_partition_validated"] is False
    row = next(r for r in result["rows"] if r["name"] == "r0_sm16_serial")
    assert row["audit_twin"]["failed_checks"] == [
        "sequence_kernels_all_in_execution_context"
    ]
    assert row["curve_eligible"] is False
