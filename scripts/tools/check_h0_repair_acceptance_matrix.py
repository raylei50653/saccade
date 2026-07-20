#!/usr/bin/env python3
"""Validate the prospective H0 repair/qualification acceptance matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "scripts/tools/h0_repair_acceptance_matrix_v1.json"
SCHEMA = "h0_repair_acceptance_matrix_v1"
GATES = (
    "repair_qualification",
    "seal",
    "authoritative_execution",
)
REQUIRED_BY_GATE = {
    "repair_qualification": (
        "packet_admission",
        "historical_archive_verification",
        "build_tool_binding_dry_run",
        "host_independent_ci",
        "owner_acceptance_matrix",
        "qualification_report",
        "qualification_report_bound_head_sha",
    ),
    "seal": (
        "instrumentation_head",
        "freeze_commit",
        "seal_commit",
        "exact_ifs_topology",
    ),
    "authoritative_execution": (
        "clean_seal_checkout",
        "independent_preflight",
        "controller_exactly_once",
        "phase_b_fail_closed",
    ),
}
QUALIFICATION_STEPS = (
    "configure",
    "build",
    "build_identity",
    "runtime_closure",
    "cuda_runtime_confinement",
    "extension_load",
    "t1_verdict_semantics",
    "runner_launch_preflight",
    "failure_envelope_serialization",
    "preseal_freeze_assembly",
)
REPAIR_UNITS = ("h0_build_tool_provenance_closure",)


class MatrixError(ValueError):
    """The prospective repair process is under-specified."""


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read acceptance matrix: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixError("acceptance matrix is not an object")
    return value


def _require_strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MatrixError(f"{label} is not a string array")
    return value


def validate_matrix(value: Mapping[str, Any]) -> None:
    expected = {
        "algorithm",
        "correction_budget",
        "gates",
        "qualification",
        "repair_units",
        "schema",
    }
    if (
        set(value) != expected
        or value.get("algorithm") != SCHEMA
        or value.get("schema") != SCHEMA
    ):
        raise MatrixError("acceptance matrix identity is malformed")
    if value.get("correction_budget") != "one_batch_then_restart":
        raise MatrixError("correction budget is not one batch then restart")
    if (
        tuple(_require_strings(value.get("repair_units"), "repair units"))
        != REPAIR_UNITS
    ):
        raise MatrixError("admissible repair unit differs from the H0 repair contract")
    gates = value.get("gates")
    if not isinstance(gates, list) or len(gates) != len(GATES):
        raise MatrixError("acceptance matrix gate cardinality is malformed")
    by_id: dict[str, Mapping[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, Mapping) or set(gate) != {"id", "required"}:
            raise MatrixError("acceptance matrix gate shape is malformed")
        gate_id = gate.get("id")
        if not isinstance(gate_id, str) or gate_id in by_id:
            raise MatrixError("acceptance matrix gate identity is malformed")
        required = tuple(
            _require_strings(gate.get("required"), f"gate {gate_id} required")
        )
        expected_required = REQUIRED_BY_GATE.get(gate_id)
        if expected_required is None or required != expected_required:
            raise MatrixError(
                f"gate {gate_id} requirements differ from the acceptance contract"
            )
        by_id[gate_id] = gate
    if tuple(by_id) != GATES:
        raise MatrixError("acceptance matrix gate order differs from H0 process")
    qualification = value.get("qualification")
    if not isinstance(qualification, Mapping) or set(qualification) != {
        "authority",
        "forbidden",
        "required_steps",
    }:
        raise MatrixError("qualification section is malformed")
    if qualification.get("authority") != "non_authoritative":
        raise MatrixError("qualification is not explicitly non-authoritative")
    forbidden = _require_strings(
        qualification.get("forbidden"), "qualification forbidden"
    )
    if set(forbidden) != {
        "phase_b",
        "research_capture",
        "research_inputs",
        "terminal_claim",
    }:
        raise MatrixError("qualification forbidden surface is incomplete")
    if (
        tuple(
            _require_strings(qualification.get("required_steps"), "qualification steps")
        )
        != QUALIFICATION_STEPS
    ):
        raise MatrixError("qualification substrate path is incomplete or reordered")


def main() -> int:
    try:
        validate_matrix(load_matrix())
    except MatrixError as exc:
        print(f"H0 repair acceptance matrix rejected: {exc}", file=sys.stderr)
        return 1
    print("H0 repair acceptance matrix: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
