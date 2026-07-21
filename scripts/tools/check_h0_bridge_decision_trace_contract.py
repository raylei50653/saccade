#!/usr/bin/env python3
"""Statically admit H0's complete capture ABI before an owner seal.

The emitted ``h0_coverage_v2`` artifact is a source-level admission check. It
validates the frozen field schema, exact append wiring, writer-local field
evidence, cursor separation, the fail-closed native envelope, and every H0
source consumed by export or replay. It never runs a capture or emits an H0
execution terminal.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from export_headline_bridge_decision_trace import (
    CAPTURE_SCHEMA,
    RECORD_FIELDS,
    SCHEMA_PATH,
    SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
HEADER_PATH = ROOT / "include/tracking/tracker_gpu.hpp"
CUDA_PATH = ROOT / "src/tracking/tracker_gpu.cu"
PYTHON_BINDING_PATH = ROOT / "src/tracking/tracker_gpu_python.cpp"
TRACKER_WRAPPER_PATH = ROOT / "src/saccade/perception/tracking/tracker_gpu.py"
EXPORT_PATH = ROOT / "scripts/tools/export_headline_bridge_decision_trace.py"
VERIFIER_PATH = ROOT / "scripts/tools/verify_headline_bridge_decision_trace.py"
CHECKER_PATH = Path(__file__).resolve()


class AppendSpec(NamedTuple):
    buffer: str
    capacity: str
    cursor: str
    overflow: str
    record: str
    declaration: str
    scope: str
    field_instances: tuple[str, ...]
    preappend_fields: tuple[str, ...]


class AppendCall(NamedTuple):
    start: int
    end: int
    args: tuple[str, ...]


class KernelScope(NamedTuple):
    start: int
    end: int


RECORD_STRUCTS = {
    "pair_records": "H0BridgePairRecord",
    "candidate_records": "H0BridgeCandidateRecord",
    "claim_records": "H0BridgeClaimRecord",
    "commit_records": "H0BridgeCommitRecord",
}
RECORD_SERIALIZER_END = {
    "pair_records": "pairs.append(std::move(row));",
    "candidate_records": "candidates.append(std::move(row));",
    "claim_records": "claims.append(std::move(row));",
    "commit_records": "commits.append(std::move(row));",
}
RECORD_APPEND_SPECS = {
    "pair_records": AppendSpec(
        "h0.pair_records",
        "h0.pair_capacity",
        "h0.pair_cursor",
        "h0.pair_overflow",
        "h0_pair",
        "H0BridgePairRecord h0_pair{};",
        "propose",
        ("h0_pair",),
        ("frame", "cand_slot", "lost_slot", "cand_instance_uid", "lost_instance_uid"),
    ),
    "candidate_records": AppendSpec(
        "h0.candidate_records",
        "h0.candidate_capacity",
        "h0.candidate_cursor",
        "h0.candidate_overflow",
        "h0_candidate",
        "H0BridgeCandidateRecord h0_candidate{};",
        "propose",
        ("h0_candidate",),
        ("frame", "cand_slot", "cand_instance_uid"),
    ),
    "claim_records": AppendSpec(
        "h0.claim_records",
        "h0.claim_capacity",
        "h0.claim_cursor",
        "h0.claim_overflow",
        "h0_claim",
        "H0BridgeClaimRecord h0_claim{};",
        "propose",
        ("h0_claim", "claim"),
        (
            "frame",
            "proposing_cand_slot",
            "proposed_lost_slot",
            "proposing_cand_instance_uid",
            "proposed_lost_instance_uid",
        ),
    ),
    "commit_records": AppendSpec(
        "h0.commit_records",
        "h0.commit_capacity",
        "h0.commit_cursor",
        "h0.commit_overflow",
        "h0_commit",
        "H0BridgeCommitRecord h0_commit{};",
        "commit",
        ("h0_commit",),
        ("frame", "cand_slot", "lost_slot", "cand_instance_uid", "lost_instance_uid"),
    ),
}
NATIVE_APPEND_SPECS = {
    "native_candidate_keys": AppendSpec(
        "h0.native_candidate_keys",
        "h0.native_candidate_capacity",
        "h0.native_candidate_cursor",
        "h0.native_candidate_overflow",
        "key",
        "H0BridgeCandidateKey key{};",
        "propose",
        ("key",),
        ("frame", "cand_slot", "cand_instance_uid"),
    ),
    "native_pair_keys": AppendSpec(
        "h0.native_pair_keys",
        "h0.native_pair_capacity",
        "h0.native_pair_cursor",
        "h0.native_pair_overflow",
        "key",
        "H0BridgePairKey key{};",
        "propose",
        ("key",),
        ("frame", "cand_slot", "lost_slot", "cand_instance_uid", "lost_instance_uid"),
    ),
    "native_proposal_keys": AppendSpec(
        "h0.native_proposal_keys",
        "h0.native_proposal_capacity",
        "h0.native_proposal_cursor",
        "h0.native_proposal_overflow",
        "proposal",
        "H0BridgeClaimKey proposal{};",
        "propose",
        ("proposal",),
        (
            "frame",
            "proposing_cand_slot",
            "proposed_lost_slot",
            "proposing_cand_instance_uid",
            "proposed_lost_instance_uid",
        ),
    ),
    "native_claim_winner_keys": AppendSpec(
        "h0.native_claim_winner_keys",
        "h0.native_claim_winner_capacity",
        "h0.native_claim_winner_cursor",
        "h0.native_claim_winner_overflow",
        "winner",
        "H0BridgeClaimKey winner{};",
        "commit",
        ("winner",),
        (
            "frame",
            "proposing_cand_slot",
            "proposed_lost_slot",
            "proposing_cand_instance_uid",
            "proposed_lost_instance_uid",
        ),
    ),
    "native_commit_keys": AppendSpec(
        "h0.native_commit_keys",
        "h0.native_commit_capacity",
        "h0.native_commit_cursor",
        "h0.native_commit_overflow",
        "commit",
        "H0BridgePairKey commit{};",
        "commit",
        ("commit",),
        ("frame", "cand_slot", "lost_slot", "cand_instance_uid", "lost_instance_uid"),
    ),
}
NATIVE_OBSERVED_CURSORS = {
    "native_candidate_keys": "h0.candidate_cursor",
    "native_pair_keys": "h0.pair_cursor",
    "native_proposal_keys": "h0.claim_cursor",
    "native_claim_winner_keys": "h0.claim_cursor",
    "native_commit_keys": "h0.commit_cursor",
}
NATIVE_ENVELOPE_FIELDS = (
    "trace_armed",
    "processed_frame_count",
    "bridge_attempt_count",
    "bridge_commit_count",
    "identity_uid_wrap_events",
)


def _read(path: Path, overrides: Mapping[Path, str]) -> str:
    return overrides.get(path, path.read_text(encoding="utf-8"))


def _sha256(path: Path, overrides: Mapping[Path, str]) -> str:
    return hashlib.sha256(_read(path, overrides).encode("utf-8")).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalise(value: str) -> str:
    return re.sub(r"\s+", "", value)


def strip_cpp_comments(source: str) -> str:
    """Mask C/C++ comments while preserving every offset and newline.

    The static checker uses source offsets to constrain each append and writer
    slice to its production kernel. Replacing comments with spaces (rather
    than deleting them) preserves that relation while preventing commented-out
    calls or assignments from becoming admission evidence. String, character,
    and raw-string literal contents are retained unchanged.
    """
    masked = list(source)
    index = 0
    normal = "normal"
    state = normal
    raw_end = ""
    while index < len(source):
        char = source[index]
        if state == normal:
            if source.startswith("//", index):
                end = source.find("\n", index)
                end = len(source) if end < 0 else end
                for masked_index in range(index, end):
                    masked[masked_index] = " "
                index = end
                continue
            if source.startswith("/*", index):
                end = source.find("*/", index + 2)
                end = len(source) if end < 0 else end + 2
                for masked_index in range(index, end):
                    if masked[masked_index] != "\n":
                        masked[masked_index] = " "
                index = end
                continue
            if source.startswith('R"', index):
                delimiter_end = source.find("(", index + 2)
                if delimiter_end >= 0:
                    raw_end = ")" + source[index + 2 : delimiter_end] + '"'
                    state = "raw"
                    index = delimiter_end + 1
                    continue
            if char == '"':
                state = "string"
            elif char == "'":
                state = "character"
        elif state == "raw":
            if source.startswith(raw_end, index):
                index += len(raw_end)
                state = normal
                continue
        elif state in {"string", "character"}:
            if char == "\\":
                index += 2
                continue
            if (state == "string" and char == '"') or (
                state == "character" and char == "'"
            ):
                state = normal
        index += 1
    return "".join(masked)


def _split_arguments(value: str) -> tuple[str, ...]:
    args: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(value[start:index])
            start = index + 1
    args.append(value[start:])
    return tuple(args)


def append_calls(cuda: str) -> list[AppendCall]:
    calls: list[AppendCall] = []
    for match in re.finditer(r"\bh0_append_record\s*\(", cuda):
        depth = 1
        index = match.end()
        while index < len(cuda) and depth:
            if cuda[index] == "(":
                depth += 1
            elif cuda[index] == ")":
                depth -= 1
            index += 1
        if depth:
            continue
        calls.append(
            AppendCall(
                match.start(), index, _split_arguments(cuda[match.end() : index - 1])
            )
        )
    return calls


def struct_body(header: str, name: str) -> str:
    match = re.search(rf"struct {re.escape(name)} \{{(.*?)\n\}};", header, re.DOTALL)
    if match is None:
        raise ValueError(f"missing C++ struct {name}")
    return match.group(1)


def serializer_body(binding: str, name: str, end_marker: str) -> str:
    match = re.search(
        rf"for \(const {re.escape(name)}\s*&\s*ev\s*:\s*capture\.[a-z_]+\) \{{",
        binding,
    )
    if match is None:
        raise ValueError(f"missing Python serializer loop for {name}")
    end = binding.find(end_marker, match.start())
    if end < 0:
        raise ValueError(f"missing Python serializer end for {name}")
    return binding[match.start() : end + len(end_marker)]


def kernel_scopes(cuda: str) -> dict[str, KernelScope]:
    ranges = {
        "propose": (
            "__global__ void relink_bidir_propose_kernel(",
            "__global__ void relink_bidir_commit_kernel(",
        ),
        "commit": (
            "__global__ void relink_bidir_commit_kernel(",
            "__global__ void compact_results_kernel(",
        ),
    }
    scopes: dict[str, KernelScope] = {}
    for name, (start_marker, end_marker) in ranges.items():
        start = cuda.find(start_marker)
        end = cuda.find(end_marker, start)
        if start < 0 or end < 0:
            raise ValueError(f"missing {name} kernel boundary")
        scopes[name] = KernelScope(start, end)
    return scopes


def _matching_calls(calls: list[AppendCall], spec: AppendSpec) -> list[AppendCall]:
    return [
        call for call in calls if call.args and _normalise(call.args[0]) == spec.buffer
    ]


def _validate_append_wiring(
    label: str,
    cuda: str,
    calls: list[AppendCall],
    spec: AppendSpec,
    scope: KernelScope,
) -> tuple[list[str], list[str]]:
    matching = _matching_calls(calls, spec)
    if not matching:
        return [], [f"{label} has no h0_append_record call for {spec.buffer}"]
    expected = (spec.buffer, spec.capacity, spec.cursor, spec.overflow, spec.record)
    failures: list[str] = []
    slices: list[str] = []
    for call in matching:
        actual = tuple(_normalise(arg) for arg in call.args)
        if actual != expected:
            failures.append(f"{label} append wiring must be {expected}, found {actual}")
            continue
        if not scope.start <= call.start < scope.end:
            failures.append(f"{label} append is outside its {spec.scope} writer kernel")
            continue
        construct_start = cuda.rfind(spec.declaration, 0, call.start)
        if construct_start < scope.start:
            failures.append(
                f"{label} append has no local {spec.declaration} construction"
            )
            continue
        slices.append(cuda[construct_start : call.start])
    return slices, failures


def _validate_preappend_fields(
    label: str,
    fields: tuple[str, ...],
    instances: tuple[str, ...],
    append_slices: list[str],
) -> list[str]:
    """Require each frozen mapping inside a writer-construction-to-append slice.

    This deliberately does not accept an assignment elsewhere in the CUDA file:
    it catches a stale writer, a post-append assignment, and a similarly named
    assignment in another kernel.
    """
    if not append_slices:
        return [f"{label} has no valid writer-construction-to-append slice"]
    writer_text = "\n".join(append_slices)
    missing = [
        field
        for field in fields
        if not any(
            re.search(
                rf"\b{re.escape(instance)}\.{re.escape(field)}\s*=",
                writer_text,
            )
            for instance in instances
        )
    ]
    if not missing:
        return []
    return [f"{label} fields must be assigned before their append: {missing}"]


def _validate_record_component(
    stream: str,
    header: str,
    binding: str,
    scopes: Mapping[str, KernelScope],
    cuda: str,
    calls: list[AppendCall],
) -> list[str]:
    struct_name = RECORD_STRUCTS[stream]
    spec = RECORD_APPEND_SPECS[stream]
    fields = RECORD_FIELDS[stream]
    failures: list[str] = []
    try:
        declared = struct_body(header, struct_name)
        serialized = serializer_body(
            binding, struct_name, RECORD_SERIALIZER_END[stream]
        )
    except ValueError as exc:
        return [str(exc)]
    missing_declared = [
        field
        for field in fields
        if field != "seq" and re.search(rf"\b{re.escape(field)}\b", declared) is None
    ]
    missing_serialized = [
        field
        for field in fields
        if field != "seq" and f'row["{field}"]' not in serialized
    ]
    if missing_declared:
        failures.append(f"{stream} C++ field checklist missing {missing_declared}")
    if missing_serialized:
        failures.append(f"{stream} Python field checklist missing {missing_serialized}")
    append_slices, append_failures = _validate_append_wiring(
        stream, cuda, calls, spec, scopes[spec.scope]
    )
    preappend_fields = tuple(
        field for field in fields if field not in {"seq", "schema_version"}
    )
    return (
        failures
        + append_failures
        + _validate_preappend_fields(
            stream, preappend_fields, spec.field_instances, append_slices
        )
    )


def _validate_native_component(
    header: str,
    binding: str,
    wrapper: str,
    cuda: str,
    calls: list[AppendCall],
    scopes: Mapping[str, KernelScope],
) -> list[str]:
    failures: list[str] = []
    for stream, spec in NATIVE_APPEND_SPECS.items():
        if spec.declaration.split()[0] not in header:
            failures.append(f"{stream} key struct is missing from the H0 header")
        if f'result["{stream}"]' not in binding:
            failures.append(f"{stream} is not drained by the Python serializer")
        if f'"{stream}"' not in wrapper:
            failures.append(f"{stream} is not sequenced by the Python wrapper")
        append_slices, append_failures = _validate_append_wiring(
            stream, cuda, calls, spec, scopes[spec.scope]
        )
        failures.extend(append_failures)
        failures.extend(
            _validate_preappend_fields(
                stream, spec.preappend_fields, spec.field_instances, append_slices
            )
        )
        observed_cursor = NATIVE_OBSERVED_CURSORS[stream]
        if spec.cursor == observed_cursor:
            failures.append(f"{stream} reuses observed record cursor {observed_cursor}")
    return failures


def _validate_envelope_component(
    header: str, binding: str, wrapper: str, exporter: str, verifier: str
) -> list[str]:
    failures: list[str] = []
    capture_struct = struct_body(header, "H0BridgeDecisionTraceCapture")
    wrapper_owned_fields = {
        "capture_schema_version",
        "capture_run_uuid",
        "capture_phase",
        "require_candidate_exposure",
        "require_commit_exposure",
    }
    for field in CAPTURE_SCHEMA["envelope_fields"]:
        if field in wrapper_owned_fields:
            continue
        if f'result["{field}"]' not in binding:
            failures.append(f"Python binding does not emit envelope field {field}")
        if f'"{field}"' not in wrapper:
            failures.append(f"Python wrapper does not require envelope field {field}")
    for field in NATIVE_ENVELOPE_FIELDS:
        if re.search(rf"\b{re.escape(field)}\b", capture_struct) is None:
            failures.append(f"H0 capture struct missing native envelope field {field}")
        if f'result["{field}"]' not in binding:
            failures.append(f"Python binding missing native envelope field {field}")
        if f'"{field}"' not in wrapper:
            failures.append(
                f"Python wrapper does not require native envelope field {field}"
            )
    for field in (
        "capture_schema_version",
        "capture_run_uuid",
        "capture_phase",
        "require_candidate_exposure",
        "require_commit_exposure",
    ):
        if f'"{field}"' not in wrapper:
            failures.append(f"Python wrapper does not write envelope field {field}")
    exporter_markers = (
        "missing = [field for field in ENVELOPE_FIELDS if field not in capture]",
        'capture["trace_armed"] is not True',
        "capture[OVERFLOW_KEYS[stream]]",
        "capture[TOTAL_KEYS[stream]]",
        "capture[UNIVERSE_OVERFLOW_KEYS[stream]]",
        "capture[UNIVERSE_TOTAL_KEYS[stream]]",
        'envelope["bridge_attempt_count"] != len(',
        'envelope["bridge_commit_count"] != len(',
        "required candidate exposure",
        "required commit exposure",
    )
    for marker in exporter_markers:
        if marker not in exporter:
            failures.append(
                f"exporter is missing fail-closed envelope marker {marker!r}"
            )
    if "capture.get(" in exporter:
        failures.append("exporter retains a fail-open capture.get default")
    h0_wrapper_start = wrapper.find("def drain_research_h0_bridge_trace")
    if h0_wrapper_start < 0:
        failures.append("Python wrapper is missing the H0 drain method")
        return failures
    h0_wrapper_end = wrapper.find("\n    def ", h0_wrapper_start + 1)
    h0_wrapper = wrapper[h0_wrapper_start:h0_wrapper_end]
    if "native.get(" in h0_wrapper:
        failures.append("Python wrapper retains a fail-open native.get default")
    if "packet = canonical_semantic_packet(capture)" not in verifier:
        failures.append("verifier does not enter through the fail-closed envelope gate")
    return failures


def coverage_report(
    source_overrides: Mapping[Path, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    overrides = source_overrides or {}
    header = _read(HEADER_PATH, overrides)
    cuda = _read(CUDA_PATH, overrides)
    analysis_cuda = strip_cpp_comments(cuda)
    binding = _read(PYTHON_BINDING_PATH, overrides)
    wrapper = _read(TRACKER_WRAPPER_PATH, overrides)
    exporter = _read(EXPORT_PATH, overrides)
    verifier = _read(VERIFIER_PATH, overrides)
    components: dict[str, bool] = {}
    failures: list[str] = []

    uid_markers = (
        "__device__ inline uint64_t h0_instance_uid(",
        "h0_slot_generation[slot] = current_generation + 1u;",
        "d_h0_slot_generation_",
    )
    uid_failures = [marker for marker in uid_markers if marker not in analysis_cuda]
    components["track_instance_uid_v1"] = not uid_failures
    if uid_failures:
        failures.append(f"track_instance_uid_v1 markers are incomplete: {uid_failures}")

    try:
        scopes = kernel_scopes(analysis_cuda)
        calls = append_calls(analysis_cuda)
    except ValueError as exc:
        scopes = {}
        calls = []
        failures.append(str(exc))
    for stream in RECORD_STRUCTS:
        component = stream.removesuffix("s")
        component_failures = (
            _validate_record_component(
                stream, header, binding, scopes, analysis_cuda, calls
            )
            if scopes
            else ["CUDA kernel scopes unavailable"]
        )
        components[component] = not component_failures
        failures.extend(component_failures)

    native_failures = (
        _validate_native_component(
            header, binding, wrapper, analysis_cuda, calls, scopes
        )
        if scopes
        else ["CUDA kernel scopes unavailable"]
    )
    components["native_universe_v2"] = not native_failures
    failures.extend(native_failures)

    try:
        envelope_failures = _validate_envelope_component(
            header, binding, wrapper, exporter, verifier
        )
    except ValueError as exc:
        envelope_failures = [str(exc)]
    components["capture_envelope_v2"] = not envelope_failures
    failures.extend(envelope_failures)

    required = tuple(CAPTURE_SCHEMA["coverage_components"])
    if tuple(components) != required:
        failures.append("coverage component ordering does not match the frozen schema")
    all_true = all(components.get(component, False) for component in required)
    report: dict[str, Any] = {
        "coverage_schema_version": "h0_coverage_v2",
        "capture_schema_version": SCHEMA_VERSION,
        "coverage_components": components,
        "all_components_true": all_true,
        "checker_sha256": _sha256(CHECKER_PATH, overrides),
        "schema_sha256": _sha256(SCHEMA_PATH, overrides),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path, overrides)
            for path in (
                HEADER_PATH,
                CUDA_PATH,
                PYTHON_BINDING_PATH,
                TRACKER_WRAPPER_PATH,
                EXPORT_PATH,
                VERIFIER_PATH,
            )
        },
    }
    return report, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="write canonical coverage JSON here"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report, failures = coverage_report()
    payload = canonical_json(report) + b"\n"
    if args.output:
        args.output.write_bytes(payload)
    if not args.quiet:
        print(payload.decode("utf-8"), end="")
    if failures:
        for failure in failures:
            print(f"H0 contract check failed: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
