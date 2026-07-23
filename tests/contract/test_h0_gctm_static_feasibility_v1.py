"""Contract tests for the bounded H0 to GCTM static feasibility audit."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "tools"
SCHEMA = TOOLS / "h0_gctm_static_feasibility_schema_v1.json"
VALIDATOR = TOOLS / "validate_h0_gctm_static_feasibility.py"
MATRIX = (
    ROOT
    / "docs/modules/semantic/research/evidence"
    / "h0_gctm_interface_static_feasibility_20260723"
    / "responsibility_coverage_matrix.json"
)
FIXTURES = (
    ROOT
    / "tests/contract/fixtures"
    / "h0_gctm_static_feasibility_fixture_catalog_v1.json"
)
REGISTRATION_FIXTURE = (
    ROOT
    / "tests/contract/fixtures"
    / "h0_gctm_guarantee_registration_candidate_sources_v2.json"
)
PACKET = MATRIX.parent
sys.path.insert(0, TOOLS.as_posix())

import validate_h0_gctm_static_feasibility as audit  # noqa: E402
import verify_h0_gctm_guarantee_registration as registration  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_mutation(record: dict[str, Any], mutation: dict[str, Any]) -> None:
    path = list(mutation["path"])
    target: Any = record
    for part in path[:-1]:
        target = target[part]
    leaf = path[-1]
    if mutation["op"] == "replace":
        target[leaf] = deepcopy(mutation["value"])
    elif mutation["op"] == "remove":
        del target[leaf]
    else:  # pragma: no cover - fixture catalog contract
        raise AssertionError(f"unsupported fixture mutation {mutation['op']!r}")


def _fixture_record(fixture: dict[str, Any]) -> dict[str, Any]:
    record = _load(MATRIX)
    for mutation in fixture["mutations"]:
        _apply_mutation(record, mutation)
    return record


def test_schema_and_canonical_matrix_validate() -> None:
    schema = _load(SCHEMA)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"] == audit.SCHEMA_ID
    assert schema["additionalProperties"] is False

    report = audit.validate_record_file(MATRIX)
    assert report == {
        "schema": audit.SCHEMA_ID,
        "audit_id": "h0_gctm_interface_static_feasibility_20260723",
        "valid": True,
        "selected_terminal": audit.TERMINAL_INSUFFICIENT,
        "responsibility_counts": {
            "H0_EXACT": 1,
            "H0_DERIVED": 1,
            "GCTM_DERIVED": 6,
            "DECLARATION_CONSTANT": 8,
            "B1_OFFLINE": 1,
            "OUTSIDE_ENVELOPE": 1,
            "UNAVAILABLE": 0,
        },
        "unresolved_runtime_objects": [
            "candidate_universe",
            "event_membership",
        ],
        "authority_verified": False,
        "runtime_compatibility_established": False,
        "h0_runtime_substrate_established": False,
        "activation_eligible": False,
    }


def test_every_required_d1_field_and_requested_boundary_has_one_row() -> None:
    record = _load(MATRIX)
    consumer_path = next(
        ROOT / item["path"]
        for item in record["frozen_inputs"]
        if item["role"] == "gctm_consumer_interface"
    )
    consumer = _load(consumer_path)
    rows = {row["consumer_object"]: row for row in record["rows"]}

    required_fields = {item["name"] for item in consumer["required_runtime_fields"]}
    assert required_fields <= rows.keys()
    assert {
        "score_transform",
        "normalization",
        "candidate_universe",
        "event_membership",
        "operator_offset_position",
    } <= rows.keys()
    assert all(isinstance(row["responsibility_class"], str) for row in rows.values())


def test_physical_gap_is_gctm_derived_not_h0_exact_or_proxy() -> None:
    record = _load(MATRIX)
    row = next(item for item in record["rows"] if item["consumer_object"] == "g_phys")
    assert row["responsibility_class"] == "GCTM_DERIVED"
    assert row["relation"] == "derived"
    assert row["derivation_inputs"] == ["la", "bridge_at"]
    definition = next(
        item
        for item in record["derivation_definitions"]
        if item["definition_id"] == row["derivation_binding"]["definition_id"]
    )
    assert "g_phys := la - bridge_at + 1" in definition["expression"]
    assert "la remains Delta_on" in definition["expression"]
    assert row["current_registration_v2_coverage"] == "candidate_source_eligible"
    assert all(
        source["registered_guarantee_claimed"] is False
        for source in row["producer_sources"]
    )


def test_residual_covariance_context_and_label_ownership_stay_separate() -> None:
    rows = {row["consumer_object"]: row for row in _load(MATRIX)["rows"]}
    assert rows["residual_position"]["responsibility_class"] == "GCTM_DERIVED"
    assert rows["S_innovation"]["responsibility_class"] == "GCTM_DERIVED"
    assert rows["S_innovation"]["current_abi_coverage"] == "gctm_owned_not_abi"
    assert (
        rows["context_drift_position"]["responsibility_class"] == "DECLARATION_CONSTANT"
    )
    assert rows["context_drift_position"]["requires_gctm_restriction"] is True
    assert rows["true_match_label"]["responsibility_class"] == "B1_OFFLINE"
    assert rows["true_match_label"]["runtime_observable"] is False


def test_candidate_source_fixture_remains_unusable_registration_input() -> None:
    report = registration.verify_record_file(REGISTRATION_FIXTURE)
    assert report["valid"] is True
    assert report["structurally_usable"] is False
    assert report["authority_verified"] is False
    assert report["disposition"] == "candidate-source"


def test_consumer_bindings_name_and_entail_their_frozen_sources() -> None:
    rows = {row["consumer_object"]: row for row in _load(MATRIX)["rows"]}
    bindings = {name: row["consumer_binding"] for name, row in rows.items()}

    assert all(binding["frozen_input_role"] for binding in bindings.values())
    assert bindings["candidate_universe"] == {
        "binding_kind": "top_level_policy",
        "frozen_input_role": "gctm_consumer_interface",
        "source_key": "candidate_universe",
    }
    assert bindings["event_membership"] == {
        "binding_kind": "compatibility_contract_requirement",
        "frozen_input_role": "h0_consumer_compatibility_contract",
        "source_key": "lost_candidate_identities_and_event_membership",
    }
    assert bindings["operator_offset_position"] == {
        "binding_kind": "audit_boundary",
        "frozen_input_role": "gctm_theory",
        "source_key": "production_cv_null_offset",
    }


def test_fixture_catalog_positive_and_per_category_negatives() -> None:
    catalog = _load(FIXTURES)
    assert catalog["schema"] == "h0_gctm_static_feasibility_fixture_catalog_v1"
    categories = {fixture["category"] for fixture in catalog["fixtures"]}
    assert {
        "positive",
        "exact_source_relation",
        "h0_derived_identity",
        "gctm_derived_identity",
        "declaration_constant_boundary",
        "b1_offline_boundary",
        "outside_envelope_boundary",
        "unavailable_boundary",
        "candidate_source_boundary",
        "shape_unit_availability_causality",
        "compatibility_contract_binding",
        "top_level_policy_binding",
        "audit_boundary_binding",
        "binding_kind_exhaustiveness",
        "coverage_conservation",
        "independent_runtime_gates",
        "terminal_selection",
    } <= categories

    for fixture in catalog["fixtures"]:
        record = _fixture_record(fixture)
        if fixture["expected_valid"]:
            report = audit.validate_record(record)
            assert report["selected_terminal"] == fixture["expected_terminal"]
            continue
        with pytest.raises(audit.AuditValidationError) as raised:
            audit.validate_record(record)
        assert raised.value.error_class == fixture["expected_error_class"], (
            fixture["fixture_id"],
            raised.value.error_class,
        )


def test_both_runtime_gates_remain_independent_and_missing() -> None:
    record = _load(MATRIX)
    gates = {
        gate["consumer_slot_id"]: gate for gate in record["runtime_consumer_gates"]
    }
    assert set(gates) == {"H0_ROUTE5_B1", "GCTM_B1"}
    assert all(gate["input_status"] == "missing" for gate in gates.values())
    assert all(gate["output_status"] == "missing" for gate in gates.values())
    assert all(gate["independent"] is True for gate in gates.values())
    assert all(gate["changed_by_audit"] is False for gate in gates.values())


def test_packet_identity_conservation_and_manifest_hashes() -> None:
    matrix = _load(MATRIX)
    identities = _load(PACKET / "frozen_input_identities.json")
    coverage = _load(PACKET / "coverage_conservation_report.json")
    terminal = _load(PACKET / "terminal_report.json")
    manifest = _load(PACKET / "manifest.json")

    assert identities["inputs"] == matrix["frozen_inputs"]
    assert (
        coverage["responsibility_counts"]
        == (matrix["coverage_conservation"]["responsibility_counts"])
    )
    assert [
        item["consumer_object"] for item in coverage["unresolved_runtime_objects"]
    ] == matrix["coverage_conservation"]["unresolved_runtime_objects"]
    assert terminal["selected_terminal"] == matrix["selected_terminal"]
    assert terminal["fixed_validator_output"] == {
        "authority_verified": False,
        "runtime_compatibility_established": False,
        "h0_runtime_substrate_established": False,
        "activation_eligible": False,
    }

    for name, expected_hash in manifest["artifacts"].items():
        assert _sha256(PACKET / name) == expected_hash
    assert _sha256(SCHEMA) == manifest["tooling"]["schema"]["sha256"]
    assert _sha256(VALIDATOR) == manifest["tooling"]["validator"]["sha256"]
    assert _sha256(FIXTURES) == manifest["tooling"]["fixture_catalog"]["sha256"]

    bindings = terminal["artifact_bindings"]
    assert bindings["schema_sha256"] == _sha256(SCHEMA)
    assert bindings["validator_sha256"] == _sha256(VALIDATOR)
    assert bindings["responsibility_matrix_sha256"] == _sha256(MATRIX)
    assert bindings["frozen_input_record_sha256"] == _sha256(
        PACKET / "frozen_input_identities.json"
    )
    assert bindings["coverage_report_sha256"] == _sha256(
        PACKET / "coverage_conservation_report.json"
    )
    assert bindings["boundary_verdicts_sha256"] == _sha256(
        PACKET / "boundary_ownership_verdicts.json"
    )
    assert bindings["fixture_catalog_sha256"] == _sha256(FIXTURES)


def test_cli_invalid_output_keeps_all_non_authority_flags_false(
    tmp_path: Path,
) -> None:
    record = _load(MATRIX)
    record["selected_terminal"] = audit.TERMINAL_FEASIBLE
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(record), encoding="utf-8")

    import subprocess

    completed = subprocess.run(
        [sys.executable, VALIDATOR.as_posix(), invalid.as_posix()],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["selected_terminal"] == audit.TERMINAL_INVALID
    assert report["authority_verified"] is False
    assert report["runtime_compatibility_established"] is False
    assert report["h0_runtime_substrate_established"] is False
    assert report["activation_eligible"] is False
