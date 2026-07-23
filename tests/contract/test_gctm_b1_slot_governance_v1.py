"""Contract tests for B1 slot identity and substrate-agnostic GCTM isolation."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "tools"
RECORD = (
    ROOT / "docs" / "research" / "contracts" / "gctm_b1_slot_identity_decision_v1.json"
)
REGISTRY = ROOT / "docs" / "research" / "contracts" / "claim_state_registry.md"
TODO = ROOT / "docs" / "modules" / "semantic" / "TODO.md"
D1_CHARTER = (
    ROOT
    / "docs"
    / "research"
    / "threads"
    / "gctm_d1_substrate_agnostic_ranking_diagnostic_task.md"
)
sys.path.insert(0, TOOLS.as_posix())

import validate_research_slot_governance as governance  # noqa: E402


def _record() -> dict[str, Any]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _slot(record: dict[str, Any], slot_id: str) -> dict[str, Any]:
    return next(slot for slot in record["slots"] if slot["slot_id"] == slot_id)


def _relation(record: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    endpoints = {source, target}
    return next(
        relation
        for relation in record["relations"]
        if {relation["source_slot_id"], relation["target_slot_id"]} == endpoints
    )


def _assert_error(
    record: dict[str, Any], error_class: str
) -> governance.SlotGovernanceValidationError:
    with pytest.raises(governance.SlotGovernanceValidationError) as raised:
        governance.validate_record(record)
    assert raised.value.error_class == error_class
    return raised.value


def test_authoritative_record_and_schema_validate() -> None:
    schema = governance.load_json(governance.SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"] == governance.SCHEMA_ID
    assert schema["additionalProperties"] is False

    report = governance.validate_record_file(RECORD)
    assert report == {
        "schema": "research_slot_governance_v1",
        "record_id": "gctm_b1_slot_identity_decision_v1",
        "valid": True,
        "owner_decision_status": "accepted",
        "authority_verified": True,
        "activation_eligible_slots": [],
        "decision_relevant_candidates": [],
        "active_wip": [],
    }


def test_gctm_b1_is_not_an_h0_route5_alias() -> None:
    record = _record()
    relation = _relation(record, "GCTM_B1", "H0_ROUTE5_B1")
    assert relation["aliases"] is False
    assert (
        _slot(record, "GCTM_B1")["slot_id"] != _slot(record, "H0_ROUTE5_B1")["slot_id"]
    )

    invalid = deepcopy(record)
    _relation(invalid, "GCTM_B1", "H0_ROUTE5_B1")["aliases"] = True
    _assert_error(invalid, "relation_semantics")


def test_owner_decision_relation_is_coexist_not_equal_or_supersede() -> None:
    relation = _relation(_record(), "GCTM_B1", "H0_ROUTE5_B1")
    assert relation == {
        "source_slot_id": "GCTM_B1",
        "target_slot_id": "H0_ROUTE5_B1",
        "relation": "coexist",
        "aliases": False,
        "supersedes": False,
        "shares_activation_authority": False,
        "change_requires_owner_transition": True,
    }


def test_diagnostic_evidence_cannot_satisfy_h0_provenance_or_identity_gates() -> None:
    record = _record()
    diagnostic = _slot(record, "GCTM_D1")
    assert set(diagnostic["cannot_satisfy_gate_classes"]) == (
        governance.RUNTIME_GATE_CLASSES
    )
    assert "h0_runtime_evidence" not in diagnostic["allowed_evidence_classes"]

    invalid = deepcopy(record)
    _slot(invalid, "GCTM_D1")["cannot_satisfy_gate_classes"].remove(
        "runtime_provenance"
    )
    _assert_error(invalid, "diagnostic_evidence_boundary")


def test_runtime_b1_cannot_activate_while_h0_substrate_blocker_exists() -> None:
    record = _record()
    runtime = _slot(record, "H0_ROUTE5_B1")
    assert runtime["state"] == "proposed"
    assert runtime["blocked_by"] == ["h0_runtime_substrate"]
    assert runtime["state"] not in {"active", "failed", "superseded"}

    invalid = deepcopy(record)
    activated = _slot(invalid, "H0_ROUTE5_B1")
    activated["state"] = "active"
    activated["owner_acceptance_id"] = "invalid_activation_while_blocked"
    next(
        item
        for item in invalid["registry_projection"]["slot_states"]
        if item["slot_id"] == "H0_ROUTE5_B1"
    )["state"] = "active"
    _assert_error(invalid, "activation_blocked")


def test_accepted_score_contract_alone_produces_no_b1_candidate() -> None:
    record = _record()
    assert (
        "accepted_score_contract"
        in _slot(record, "H0_ROUTE5_B1")["allowed_evidence_classes"]
    )
    assert (
        "accepted_score_contract"
        in _slot(record, "GCTM_B1")["allowed_evidence_classes"]
    )
    assert record["registry_projection"]["decision_relevant_candidates"] == []


def test_diagnostic_charter_and_terminal_do_not_unlock_o1() -> None:
    record = _record()
    policy = next(
        policy
        for policy in record["terminal_policies"]
        if policy["slot_id"] == "GCTM_D1"
    )
    assert policy["unlocks_slot_ids"] == []
    assert policy["generates_decision_relevant_candidate"] is False
    assert record["registry_projection"]["o1_state"] == "proposed"

    invalid = deepcopy(record)
    next(
        policy
        for policy in invalid["terminal_policies"]
        if policy["slot_id"] == "GCTM_D1"
    )["unlocks_slot_ids"] = ["GCTM_B1"]
    _assert_error(invalid, "diagnostic_terminal_boundary")


def test_diagnostic_terminal_cannot_rewrite_runtime_b1_state() -> None:
    record = _record()
    policy = next(
        policy
        for policy in record["terminal_policies"]
        if policy["slot_id"] == "GCTM_D1"
    )
    assert policy["may_transition_slot_ids"] == ["GCTM_D1"]

    invalid = deepcopy(record)
    next(
        policy
        for policy in invalid["terminal_policies"]
        if policy["slot_id"] == "GCTM_D1"
    )["may_transition_slot_ids"] = ["GCTM_D1", "H0_ROUTE5_B1"]
    _assert_error(invalid, "diagnostic_terminal_boundary")


def test_missing_compatibility_verdict_fails_closed() -> None:
    gate = _record()["compatibility_gates"][0]
    assert gate["status"] == "missing"
    assert gate["verdict_owner_acceptance_id"] is None
    assert gate["fail_closed"] is True
    assert gate["incompatible_behavior"] == "reject_runtime_consumption"
    assert set(gate["required_checks"]) == governance.RUNTIME_COMPATIBILITY_CHECKS

    invalid = _record()
    invalid["compatibility_gates"][0]["fail_closed"] = False
    _assert_error(invalid, "schema_rejection")


def test_registry_repush_has_no_false_active_wip() -> None:
    record = _record()
    projection = record["registry_projection"]
    assert projection["decision_relevant_candidates"] == []
    assert projection["active_wip"] == []
    assert projection["h0_reentry_authorized"] is False

    invalid = deepcopy(record)
    invalid["registry_projection"]["active_wip"] = ["GCTM_D1"]
    _assert_error(invalid, "false_active_wip")


def test_registry_todo_and_charter_project_the_machine_record() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    todo = TODO.read_text(encoding="utf-8")
    charter = D1_CHARTER.read_text(encoding="utf-8")

    for text in (registry, todo):
        assert "H0_ROUTE5_B1" in text
        assert "GCTM_B1" in text
        assert "GCTM_D1" in text
        assert "blocked_by: h0_runtime_substrate" in text
    assert "無 active" in todo
    assert "decision_relevant_candidates: []" in registry
    assert "active_wip: []" in registry
    assert "PROPOSED / non-WIP / not owner-accepted" in charter
    assert "runtime faithful" in charter
    assert "reject_runtime_consumption" in charter
