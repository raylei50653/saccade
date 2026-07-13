#!/usr/bin/env python3
"""Statically admit the complete H0 capture ABI before an owner seal.

The check emits the machine-readable ``h0_coverage_v2`` artifact.  It checks
the frozen field schema against the C++ record declarations and Python drain
serializer, and verifies that each native-universe writer is independent of
the corresponding record append path.  It is an engineering-time check: it
does not run a capture and cannot produce an H0 execution terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

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
CHECKER_PATH = Path(__file__).resolve()

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
RECORD_WRITER_MARKERS = {
    "pair_records": "h0_append_record(h0.pair_records",
    "candidate_records": "h0_append_record(h0.candidate_records",
    "claim_records": "h0.claim_records",
    "commit_records": "h0_append_record(h0.commit_records",
}
RECORD_WRITER_INSTANCES = {
    "pair_records": ("h0_pair",),
    "candidate_records": ("h0_candidate",),
    "claim_records": ("h0_claim", "claim"),
    "commit_records": ("h0_commit",),
}
NATIVE_UNIVERSE_MARKERS = {
    "native_candidate_keys": ("H0BridgeCandidateKey", "h0.native_candidate_keys"),
    "native_pair_keys": ("H0BridgePairKey", "h0.native_pair_keys"),
    "native_proposal_keys": ("H0BridgeClaimKey", "h0.native_proposal_keys"),
    "native_claim_winner_keys": (
        "H0BridgeClaimKey",
        "h0.native_claim_winner_keys",
    ),
    "native_commit_keys": ("H0BridgePairKey", "h0.native_commit_keys"),
}
NATIVE_UNIVERSE_WRITER_INSTANCES = {
    "native_candidate_keys": ("key",),
    "native_pair_keys": ("key",),
    "native_proposal_keys": ("proposal",),
    "native_claim_winner_keys": ("winner",),
    "native_commit_keys": ("commit",),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
    start = match.start()
    end = binding.find(end_marker, start)
    if end < 0:
        raise ValueError(f"missing Python serializer end for {name}")
    return binding[start : end + len(end_marker)]


def coverage_report() -> tuple[dict[str, Any], list[str]]:
    header = HEADER_PATH.read_text(encoding="utf-8")
    cuda = CUDA_PATH.read_text(encoding="utf-8")
    binding = PYTHON_BINDING_PATH.read_text(encoding="utf-8")
    wrapper = TRACKER_WRAPPER_PATH.read_text(encoding="utf-8")
    components: dict[str, bool] = {}
    failures: list[str] = []

    uid_markers = ("track_instance_uid_v1", "h0_instance_uid", "d_h0_slot_generation_")
    uid_ok = all(marker in header or marker in cuda for marker in uid_markers)
    components["track_instance_uid_v1"] = uid_ok
    if not uid_ok:
        failures.append("track_instance_uid_v1 markers are incomplete")

    for stream, struct_name in RECORD_STRUCTS.items():
        component = stream.removesuffix("s")
        fields = RECORD_FIELDS[stream]
        try:
            declared = struct_body(header, struct_name)
            serialized = serializer_body(
                binding, struct_name, RECORD_SERIALIZER_END[stream]
            )
        except ValueError as exc:
            components[component] = False
            failures.append(str(exc))
            continue
        missing_declared = [
            field
            for field in fields
            if field != "seq"
            and re.search(rf"\b{re.escape(field)}\b", declared) is None
        ]
        missing_serialized = [
            field
            for field in fields
            if field != "seq" and f'row["{field}"]' not in serialized
        ]
        missing_written = [
            field
            for field in fields
            if field not in {"seq", "schema_version"}
            and not any(
                f"{instance}.{field}" in cuda
                for instance in RECORD_WRITER_INSTANCES[stream]
            )
        ]
        writer_ok = RECORD_WRITER_MARKERS[stream] in cuda
        ok = (
            not missing_declared
            and not missing_serialized
            and not missing_written
            and writer_ok
        )
        components[component] = ok
        if missing_declared:
            failures.append(f"{stream} C++ field checklist missing {missing_declared}")
        if missing_serialized:
            failures.append(
                f"{stream} Python field checklist missing {missing_serialized}"
            )
        if missing_written:
            failures.append(
                f"{stream} CUDA writer field checklist missing {missing_written}"
            )
        if not writer_ok:
            failures.append(f"{stream} native append marker is missing")

    native_ok = True
    for stream, markers in NATIVE_UNIVERSE_MARKERS.items():
        if not all(marker in header or marker in cuda for marker in markers):
            native_ok = False
            failures.append(f"{stream} native-universe writer marker is missing")
        if f'result["{stream}"]' not in binding:
            native_ok = False
            failures.append(f"{stream} is not drained by the Python serializer")
        if f'"{stream}"' not in wrapper:
            native_ok = False
            failures.append(f"{stream} is not sequenced by the Python wrapper")
        missing_native_fields = [
            field
            for field in CAPTURE_SCHEMA["native_universe_keys"][stream]
            if field != "seq"
            and not any(
                f"{instance}.{field}" in cuda
                for instance in NATIVE_UNIVERSE_WRITER_INSTANCES[stream]
            )
        ]
        if missing_native_fields:
            native_ok = False
            failures.append(
                f"{stream} CUDA key field checklist missing {missing_native_fields}"
            )
    if "independent from the record" not in cuda:
        native_ok = False
        failures.append("native-universe cursor independence marker is missing")
    if "h0_bridge_decision_trace_v2" not in wrapper or 'row["seq"]' not in wrapper:
        native_ok = False
        failures.append("Python wrapper does not stamp the frozen v2 schema and seq")
    components["native_universe_v2"] = native_ok

    required = tuple(CAPTURE_SCHEMA["coverage_components"])
    if tuple(components) != required:
        failures.append("coverage component ordering does not match the frozen schema")
    all_true = all(components.get(component, False) for component in required)
    report: dict[str, Any] = {
        "coverage_schema_version": "h0_coverage_v2",
        "capture_schema_version": SCHEMA_VERSION,
        "coverage_components": components,
        "all_components_true": all_true,
        "checker_sha256": sha256(CHECKER_PATH),
        "schema_sha256": sha256(SCHEMA_PATH),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                HEADER_PATH,
                CUDA_PATH,
                PYTHON_BINDING_PATH,
                TRACKER_WRAPPER_PATH,
                EXPORT_PATH,
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
