"""Fail-closed Green Context routing verdicts from owner evidence and CUPTI traces."""

# status: diagnostic
import re
from pathlib import Path

from scripts.benchmarks.resource_partition.green_owner import (
    audit_window,
    host_to_cupti,
    parse_trace,
)

ROOT = Path(__file__).resolve().parents[3]
NATIVE_SOURCES = ("src/tracking", "src/perception")
CALIBRATION_SPREAD_LIMIT_NS = 1_000_000

# Heuristic execution-source labels for coverage reporting only. The verdict
# never depends on these labels: every kernel is judged by its context ID.
SOURCE_PATTERNS = (
    ("nvjpeg", re.compile(r"nvjpeg|Nvjpeg|jpeg", re.I)),
    ("tensorrt", re.compile(r"^sm\d+_|^__myl|trt|nvinfer|genericReformat|Reformat")),
    (
        "torch",
        re.compile(
            r"at::native|_ZN2at|at_cuda_detail|c10::|_ZN3c10|cutlass|cub::|_ZN3cub"
            r"|thrust|^triton_"
        ),
    ),
    ("cufft", re.compile(r"regular_fft|_fft_|cufft", re.I)),
    ("cublas_or_cudnn", re.compile(r"gemm|cudnn|cublas|xmma|nchw|gemv", re.I)),
)


def native_kernel_names(root=ROOT):
    names = set()
    for rel in NATIVE_SOURCES:
        for path in (root / rel).glob("*.cu"):
            names.update(
                re.findall(
                    r"__global__\s+void\s+([A-Za-z_][A-Za-z_0-9]*)", path.read_text()
                )
            )
    return names


def classify(name, native):
    for candidate in native:
        if candidate in name:
            return "native"
    for label, pattern in SOURCE_PATTERNS:
        if pattern.search(name):
            return label
    return "other"


def source_coverage(kernels, native):
    counts = {}
    for kernel in kernels:
        label = classify(kernel["name"], native)
        counts[label] = counts.get(label, 0) + 1
    return counts


AUDIT_CHECKS = (
    "audit_enabled",
    "audit_no_dropped_records",
    "audit_execution_context_record_present",
    "clock_calibration_spread_within_limit",
    "sequence_windows_present",
    "sequence_kernels_present",
    "sequence_kernels_all_in_execution_context",
    "sequence_memsets_all_in_execution_context",
    "graph_launched_kernels_present",
)


def build_point_report(point, trace_path):
    """Attach routing verification and a fail-closed verdict to a point.

    ``owner_checks`` come from the owner's own evidence (stream registry,
    context ownership per completed frame, SM-id probes, double-buffer route).
    ``audit_checks`` come from the CUPTI activity trace and exist only for an
    audited run. A point is ``true_partition_validated`` only when it is a
    Green Context point and both groups pass; an unaudited point can never be
    validated on its own (the sweep pairs it with an audited twin).
    """
    owner = point.get("owner") or {}
    green = point["requested_sm_count"] is not None
    checks = {
        "eval_error_none": point["error"] is None,
        "owner_evidence_present": bool(owner),
        "completions_present": bool(point["completions"]),
        "execution_context_created": bool(owner.get("execution_context")),
        "all_created_streams_owned": bool(owner.get("streams"))
        and all(s["owned"] for s in owner["streams"]),
        "main_thread_context_owned_at_finalize": bool(
            owner.get("main_thread_context_still_owned_at_finalize")
        ),
        "no_thread_context_switch_errors": owner.get("thread_context_switch_errors")
        == [],
        "main_thread_context_owned_at_every_completion": point.get(
            "frames_completed_with_unowned_main_thread_context"
        )
        == [],
    }
    actual = owner.get("actual_sm_count")
    if green:
        checks["actual_sm_count_matches_requested"] = (
            actual == point["requested_sm_count"]
        )
    probe = owner.get("probe") or {}
    for phase in ("before", "after"):
        for kind in ("direct", "graph"):
            ids = (probe.get(phase) or {}).get(f"{kind}_sm_ids")
            checks[f"probe_{phase}_{kind}_sm_count_matches_actual"] = (
                ids is not None and actual is not None and len(ids) == actual
            )
    owned_handles = {s["handle"] for s in owner.get("streams", []) if s["owned"]}
    checks["double_buffer_stream_owned"] = all(
        (not r["double_buffer_enabled"])
        or (
            r["double_buffer_stream_handle"] in owned_handles
            and r["double_buffer_stream_owned_class"] is True
        )
        for r in point["routes"].values()
    )
    checks["double_buffer_route_consistent"] = bool(point["routes"]) and (
        len({r["double_buffer_enabled"] for r in point["routes"].values()}) == 1
    )
    report = dict(point)
    report["actual_sm_count"] = actual
    report["partition_kind"] = (
        "green_context" if green else "primary_full_device_baseline"
    )
    owner_checks = dict(checks)
    audit = None
    audit_checks = {}
    checks = {}
    if trace_path is not None and Path(trace_path).exists() and owner:
        trace = parse_trace(trace_path)
        checks["audit_enabled"] = bool(point["audit_enabled"])
        execution_id = owner.get("cupti_execution_context_id")
        context_record = next(
            (c for c in trace["contexts"] if c["context"] == execution_id), None
        )
        checks["audit_no_dropped_records"] = trace["dropped"] == 0
        checks["audit_execution_context_record_present"] = context_record is not None
        if green:
            checks["audit_context_record_is_green_with_actual_sm_count"] = bool(
                context_record
            ) and (
                context_record["is_green"] == 1 and context_record["sm_count"] == actual
            )
        calibration = owner.get("calibration") or []
        spread = None
        windows = {}
        native = native_kernel_names()
        if calibration and point["frame_starts"]:
            _, spread = host_to_cupti(calibration, 0.0)
            checks["clock_calibration_spread_within_limit"] = (
                spread <= CALIBRATION_SPREAD_LIMIT_NS
            )
            null_stream = context_record["null_stream"] if context_record else None
            for seq, starts in point["frame_starts"].items():
                completions = point["completions"].get(seq) or []
                if not completions:
                    continue
                warmup = point["routes"][seq]["warmup_frames"]
                measured_start = next(
                    (t for f, t in starts if f == warmup + 1), starts[0][1]
                )
                begin, _ = host_to_cupti(calibration, starts[0][1])
                measured_begin, _ = host_to_cupti(calibration, measured_start)
                end, _ = host_to_cupti(calibration, completions[-1][1])
                seq_window = audit_window(trace, execution_id, begin, end, null_stream)
                measured = audit_window(
                    trace, execution_id, measured_begin, end, null_stream
                )
                inside = [k for k in trace["kernels"] if begin <= k["start"] <= end]
                windows[seq] = {
                    "cupti_begin": begin,
                    "cupti_measured_begin": measured_begin,
                    "cupti_end": end,
                    "sequence": seq_window,
                    "measured": measured,
                    "source_coverage_heuristic": source_coverage(inside, native),
                }
            first_begin = (
                min(w["cupti_begin"] for w in windows.values()) if windows else None
            )
            setup = (
                audit_window(trace, execution_id, 0, first_begin, None)
                if first_begin
                else None
            )
            checks["sequence_windows_present"] = bool(windows)
            checks["sequence_kernels_present"] = bool(windows) and all(
                w["sequence"]["kernels"]["total"] > 0 for w in windows.values()
            )
            checks["sequence_kernels_all_in_execution_context"] = bool(windows) and all(
                w["sequence"]["kernels"]["outside"] == 0 for w in windows.values()
            )
            checks["sequence_memsets_all_in_execution_context"] = bool(windows) and all(
                w["sequence"]["memsets"]["outside"] == 0 for w in windows.values()
            )
            checks["graph_launched_kernels_present"] = bool(windows) and all(
                w["sequence"]["kernels"]["graph_launched"] > 0 for w in windows.values()
            )
        else:
            checks["clock_calibration_spread_within_limit"] = False
            checks["sequence_windows_present"] = False
            setup = None
        audit = {
            "attributes": trace["attributes"],
            "dropped": trace["dropped"],
            "contexts": trace["contexts"],
            "execution_context_id": execution_id,
            "calibration_spread_ns": spread,
            "kernel_records": len(trace["kernels"]),
            "copy_records": len(trace["copies"]),
            "memset_records": len(trace["memsets"]),
            "setup_before_first_sequence": setup,
            "windows": windows,
        }
        audit_checks = checks
        for name in AUDIT_CHECKS:
            audit_checks.setdefault(name, False)
    else:
        audit_checks = {name: False for name in AUDIT_CHECKS}
    report["audit"] = audit
    report["routing_verification"] = {
        "owner_checks": owner_checks,
        "audit_checks": audit_checks,
    }
    report["failed_owner_checks"] = sorted(k for k, v in owner_checks.items() if not v)
    report["failed_audit_checks"] = sorted(k for k, v in audit_checks.items() if not v)
    report["failed_checks"] = (
        report["failed_owner_checks"] + report["failed_audit_checks"]
    )
    report["owner_checks_passed"] = not report["failed_owner_checks"]
    report["audit_checks_passed"] = not report["failed_audit_checks"]
    report["routing_validated"] = not report["failed_checks"]
    # A full-device primary baseline is never a partition, however well routed.
    report["true_partition_validated"] = green and report["routing_validated"]
    return report
