#!/usr/bin/env python3
"""Verify a non-authoritative H0 R4 repair qualification report.

Binds the report to an exact 40-character candidate head SHA and asserts the
Repair acceptance boundary: no capture, no F/S, no terminal claim, no actual
guarantee.
"""
# status: stable

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "h0_r4_qualification_report_v1"
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
TERMINALS = frozenset(
    {
        "H0_R4_REPAIR_INVALID",
        "H0_R4_REPAIR_REQUIRES_ABI_DELTA",
        "H0_R4_REPAIR_QUALIFIED_SEALABLE",
    }
)


class ReportError(ValueError):
    """Qualification report is incomplete or over-claims."""


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read qualification report: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError("qualification report is not an object")
    return value


def validate_report(value: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "authority",
        "repair_unit",
        "candidate_head_sha",
        "qualification_result",
        "repair_terminal",
        "known_s3_defect_removed",
        "declaration_runtime_bound",
        "declaration_authority_overlay_bound",
        "trace_v2_abi_changed",
        "launch_hygiene",
        "registration_v3_downstream_target",
        "execution_authorized",
        "actual_guarantee_established",
        "next_owner_decision",
        "forbidden_claims",
    }
    if set(value) < required or value.get("schema") != SCHEMA:
        raise ReportError("qualification report identity is malformed")
    if value.get("authority") != "non_authoritative":
        raise ReportError("qualification report must be non-authoritative")
    if value.get("repair_unit") != "h0_authority_overlay_runtime_binding_split_v1":
        raise ReportError("repair unit is not the Amendment-10 sole unit")
    head = value.get("candidate_head_sha")
    if not isinstance(head, str) or not HEAD_RE.fullmatch(head):
        raise ReportError("candidate head must be an exact 40-character SHA")
    if value.get("qualification_result") not in {"passed", "failed", "not_run"}:
        raise ReportError("qualification_result is invalid")
    terminal = value.get("repair_terminal")
    if terminal not in TERMINALS:
        raise ReportError("repair terminal is not an ordered H0_R4 terminal")
    if value.get("declaration_runtime_bound") is not False:
        raise ReportError("declaration must not be runtime-bound")
    if value.get("execution_authorized") is not False:
        raise ReportError("execution must not be authorized")
    if value.get("actual_guarantee_established") is not False:
        raise ReportError("actual guarantee must not be established")
    if value.get("trace_v2_abi_changed") is not False and terminal == (
        "H0_R4_REPAIR_QUALIFIED_SEALABLE"
    ):
        raise ReportError("qualified-sealable terminal requires unchanged trace-v2 ABI")
    forbidden = value.get("forbidden_claims")
    if not isinstance(forbidden, list) or not forbidden:
        raise ReportError("forbidden claims must be enumerated")
    for claim in (
        "I_selected",
        "F_created",
        "S_created",
        "SEALED_accepted",
        "execution_authorized",
        "H0_baseline_accepted",
        "runtime_substrate_established",
        "actual_guarantee_registered",
        "runtime_compatibility_established",
    ):
        if claim not in forbidden:
            raise ReportError(f"forbidden claim missing: {claim}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_report(load_report(args.report))
    except ReportError as exc:
        print(f"H0 R4 qualification report rejected: {exc}", file=sys.stderr)
        return 1
    print("H0 R4 qualification report: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
