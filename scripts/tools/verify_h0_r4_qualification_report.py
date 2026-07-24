#!/usr/bin/env python3
"""Verify a non-authoritative H0 R4 repair qualification report.

Mechanically binds the report to repository evidence:

* exact candidate commit exists in the local repository;
* candidate tree equals the reported tree SHA;
* qualification summary is readable and agrees on head/tree/result;
* repair terminal is consistent with qualification outcome and required flags.

A self-declared ``H0_R4_REPAIR_QUALIFIED_SEALABLE`` without a matching passed
summary and real commit/tree binding is rejected.
"""
# status: stable

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "h0_r4_qualification_report_v1"
SUMMARY_SCHEMA = "h0_phase_a_qualification_v1"
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
TERMINALS = frozenset(
    {
        "H0_R4_REPAIR_INVALID",
        "H0_R4_REPAIR_REQUIRES_ABI_DELTA",
        "H0_R4_REPAIR_QUALIFIED_SEALABLE",
    }
)
REQUIRED_KEYS = frozenset(
    {
        "schema",
        "authority",
        "repair_unit",
        "candidate_head_sha",
        "candidate_tree_sha",
        "qualification_result",
        "qualification_summary_path",
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
)
FORBIDDEN_CLAIMS = (
    "I_selected",
    "F_created",
    "S_created",
    "SEALED_accepted",
    "execution_authorized",
    "H0_baseline_accepted",
    "runtime_substrate_established",
    "actual_guarantee_registered",
    "runtime_compatibility_established",
)
LAUNCH_HYGIENE_VALUES = frozenset(
    {
        "clear",
        "rejected",
        "not_run",
        "predicate_single_source_verified",
    }
)
REGISTRATION_TARGETS = frozenset(
    {
        "structurally_reachable",
        "blocked",
    }
)
NEXT_OWNER_DECISIONS = frozenset(
    {
        "separate Seal PR",
        "exact ABI-delta charter",
        "repair contract",
    }
)
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
    "landing_discovery_dry_run",
)


class ReportError(ValueError):
    """Qualification report is incomplete, unbound, or over-claims."""


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read qualification report: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError("qualification report is not an object")
    return value


def _git(*args: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReportError(f"git {' '.join(args)} failed: {exc}") from exc


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEAD_RE.fullmatch(value):
        raise ReportError(f"{label} must be an exact 40-character SHA")
    return value


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read qualification summary: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError("qualification summary is not an object")
    return value


def _verify_repository_binding(head: str, tree: str, root: Path) -> None:
    try:
        _git("cat-file", "-e", f"{head}^{{commit}}", cwd=root)
    except ReportError as exc:
        raise ReportError(
            f"candidate_head_sha is not a commit in repository: {head}"
        ) from exc
    actual_tree = _git("rev-parse", f"{head}^{{tree}}", cwd=root)
    if actual_tree != tree:
        raise ReportError(
            "candidate_tree_sha does not match git tree of candidate_head_sha: "
            f"reported={tree} actual={actual_tree}"
        )


def _verify_summary_binding(
    report: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    head: str,
    tree: str,
) -> None:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ReportError("qualification summary schema mismatch")
    if summary.get("authority") != "non_authoritative":
        raise ReportError("qualification summary is not non-authoritative")
    for key, expected in (
        ("capture", "forbidden"),
        ("phase_b", "forbidden"),
        ("terminal_claim", "forbidden"),
        ("research_inputs", "forbidden"),
    ):
        if summary.get(key) != expected:
            raise ReportError(f"qualification summary {key} must be {expected}")
    if summary.get("repository_head_sha") != head:
        raise ReportError("summary repository_head_sha disagrees with report")
    if summary.get("repository_tree_sha") != tree:
        raise ReportError("summary repository_tree_sha disagrees with report")
    if summary.get("requested_ref") != head:
        raise ReportError("summary requested_ref must equal candidate_head_sha")
    summary_result = summary.get("result")
    if summary_result not in {"passed", "failed"}:
        raise ReportError("qualification summary result is invalid")
    report_result = report.get("qualification_result")
    if report_result == "not_run":
        if summary_result == "passed":
            raise ReportError("not_run report cannot reference a passed summary")
    elif report_result != summary_result:
        raise ReportError("report qualification_result disagrees with summary.result")
    steps = summary.get("steps")
    if not isinstance(steps, list) or len(steps) != len(QUALIFICATION_STEPS):
        raise ReportError("qualification summary step tuple is incomplete")
    names = [step.get("name") if isinstance(step, Mapping) else None for step in steps]
    if tuple(names) != QUALIFICATION_STEPS:
        raise ReportError("qualification summary steps are reordered or incomplete")
    if summary_result == "passed":
        if any(
            not isinstance(step, Mapping) or step.get("state") != "passed"
            for step in steps
        ):
            raise ReportError("passed summary has a non-passed qualification step")


def _verify_terminal_consistency(report: Mapping[str, Any]) -> None:
    terminal = report.get("repair_terminal")
    qual = report.get("qualification_result")
    defect = report.get("known_s3_defect_removed")
    overlay = report.get("declaration_authority_overlay_bound")
    runtime_bound = report.get("declaration_runtime_bound")
    abi = report.get("trace_v2_abi_changed")
    hygiene = report.get("launch_hygiene")
    registration = report.get("registration_v3_downstream_target")
    next_decision = report.get("next_owner_decision")
    execution = report.get("execution_authorized")
    guarantee = report.get("actual_guarantee_established")

    if runtime_bound is not False:
        raise ReportError("declaration must not be runtime-bound")
    if execution is not False:
        raise ReportError("execution must not be authorized")
    if guarantee is not False:
        raise ReportError("actual guarantee must not be established")
    if hygiene not in LAUNCH_HYGIENE_VALUES:
        raise ReportError("launch_hygiene value is invalid")
    if registration not in REGISTRATION_TARGETS:
        raise ReportError("registration_v3_downstream_target value is invalid")
    if next_decision not in NEXT_OWNER_DECISIONS:
        raise ReportError("next_owner_decision value is invalid")
    if (
        not isinstance(defect, bool)
        or not isinstance(overlay, bool)
        or not isinstance(abi, bool)
    ):
        raise ReportError("boolean terminal flags are malformed")

    if terminal == "H0_R4_REPAIR_QUALIFIED_SEALABLE":
        if qual != "passed":
            raise ReportError("QUALIFIED_SEALABLE requires qualification_result=passed")
        if defect is not True:
            raise ReportError(
                "QUALIFIED_SEALABLE requires known_s3_defect_removed=true"
            )
        if overlay is not True:
            raise ReportError(
                "QUALIFIED_SEALABLE requires declaration_authority_overlay_bound=true"
            )
        if abi is not False:
            raise ReportError("QUALIFIED_SEALABLE requires trace_v2_abi_changed=false")
        if registration != "structurally_reachable":
            raise ReportError(
                "QUALIFIED_SEALABLE requires registration-v3 structurally_reachable"
            )
        if next_decision != "separate Seal PR":
            raise ReportError(
                "QUALIFIED_SEALABLE next_owner_decision must be separate Seal PR"
            )
        return

    if terminal == "H0_R4_REPAIR_REQUIRES_ABI_DELTA":
        if abi is not True:
            raise ReportError("REQUIRES_ABI_DELTA requires trace_v2_abi_changed=true")
        if next_decision != "exact ABI-delta charter":
            raise ReportError(
                "REQUIRES_ABI_DELTA next_owner_decision must be exact ABI-delta charter"
            )
        # Qualification may still pass substrate steps while the repair scope
        # discovers an ABI dependency; still forbid over-claiming sealability.
        return

    if terminal == "H0_R4_REPAIR_INVALID":
        if next_decision != "repair contract":
            raise ReportError("INVALID next_owner_decision must be repair contract")
        if qual == "passed" and defect is True and overlay is True and abi is False:
            raise ReportError(
                "INVALID cannot be selected when qualification passed and "
                "defect/overlay/ABI flags are all green"
            )
        return

    raise ReportError("repair terminal is not an ordered H0_R4 terminal")


def validate_report(value: Mapping[str, Any], *, root: Path = ROOT) -> None:
    if not REQUIRED_KEYS.issubset(value) or value.get("schema") != SCHEMA:
        missing = sorted(REQUIRED_KEYS - set(value))
        raise ReportError(
            "qualification report identity is malformed"
            + (f"; missing={missing}" if missing else "")
        )
    if value.get("authority") != "non_authoritative":
        raise ReportError("qualification report must be non-authoritative")
    if value.get("repair_unit") != "h0_authority_overlay_runtime_binding_split_v1":
        raise ReportError("repair unit is not the Amendment-10 sole unit")
    if value.get("qualification_result") not in {"passed", "failed", "not_run"}:
        raise ReportError("qualification_result is invalid")
    terminal = value.get("repair_terminal")
    if terminal not in TERMINALS:
        raise ReportError("repair terminal is not an ordered H0_R4 terminal")

    head = _require_sha(value.get("candidate_head_sha"), "candidate_head_sha")
    tree = _require_sha(value.get("candidate_tree_sha"), "candidate_tree_sha")
    _verify_repository_binding(head, tree, root)

    summary_rel = value.get("qualification_summary_path")
    if (
        not isinstance(summary_rel, str)
        or not summary_rel
        or summary_rel.startswith("/")
    ):
        raise ReportError(
            "qualification_summary_path must be a relative repository path"
        )
    summary_path = (root / summary_rel).resolve()
    try:
        summary_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ReportError("qualification_summary_path escapes repository root") from exc
    if not summary_path.is_file() or summary_path.is_symlink():
        raise ReportError("qualification summary is missing or not a regular file")
    summary = _load_summary(summary_path)
    _verify_summary_binding(value, summary, head=head, tree=tree)

    _verify_terminal_consistency(value)

    forbidden = value.get("forbidden_claims")
    if not isinstance(forbidden, list) or not all(
        isinstance(item, str) for item in forbidden
    ):
        raise ReportError("forbidden claims must be a string array")
    for claim in FORBIDDEN_CLAIMS:
        if claim not in forbidden:
            raise ReportError(f"forbidden claim missing: {claim}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root used for git and summary path resolution",
    )
    args = parser.parse_args(argv)
    try:
        validate_report(load_report(args.report), root=args.root.resolve())
    except ReportError as exc:
        print(f"H0 R4 qualification report rejected: {exc}", file=sys.stderr)
        return 1
    print("H0 R4 qualification report: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
