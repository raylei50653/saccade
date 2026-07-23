"""Contract for the fail-closed L2 score-ranking declaration v1 validator."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "tools"
FIXTURE = (
    ROOT / "tests" / "contract" / "fixtures" / "score_ranking_declaration_valid_v1.json"
)
sys.path.insert(0, TOOLS.as_posix())

import validate_score_ranking_declaration as score_contract  # noqa: E402


def _valid() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _active_sr6() -> dict[str, Any]:
    declaration = _valid()
    declaration["record_scope"] = "actual"
    binding = declaration["contract_binding"]
    binding["contract_status"] = "active"
    binding["owner_acceptance_id"] = "fixture_owner_acceptance_v1"
    binding["registry_binding_id"] = "fixture_registry_binding_v1"

    policy = declaration["policy"]
    spaces = declaration["spaces"]
    spaces["assignment_space_id"] = "fixture_assignment_space_v1"
    spaces["system_space_id"] = "fixture_system_space_v1"
    spaces["reductions"] = [
        {
            "reduction_id": "fixture_event_to_assignment_v1",
            "source_space_id": policy["event_space_id"],
            "target_space_id": spaces["assignment_space_id"],
        },
        {
            "reduction_id": "fixture_assignment_to_system_v1",
            "source_space_id": spaces["assignment_space_id"],
            "target_space_id": spaces["system_space_id"],
        },
    ]

    claim = declaration["claim"]
    claim["target_rung"] = "SR6"
    claim["claim_space"] = "system"
    claim["kappa"]["quantification_space"] = "system"
    claim["primary_ranking_metric"] = "correct_assignment_rate"
    claim["metric_parameters"] = {}
    claim["target_metric_id"] = "fixture_track_sequence_system_metric_v1"

    for rung in ("SR4", "SR5", "SR6"):
        obligation_ids = sorted(score_contract.RUNG_OBLIGATIONS[rung])
        declaration["rung_obligations"].append(
            {
                "rung": rung,
                "obligation_ids": obligation_ids,
                "artifact_bindings": {
                    obligation_id: f"fixture_{obligation_id}_binding_v1"
                    for obligation_id in obligation_ids
                },
            }
        )
    declaration["terminals"][-1]["state_transition"]["target_state"] = "SR6"
    return declaration


def _assert_error(
    declaration: dict[str, Any], expected_error_class: str
) -> score_contract.ScoreRankingValidationError:
    with pytest.raises(score_contract.ScoreRankingValidationError) as raised:
        score_contract.validate_declaration(declaration)
    assert raised.value.error_class == expected_error_class
    return raised.value


def test_schema_is_valid_draft_2020_12() -> None:
    schema = score_contract.load_json(score_contract.SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"] == score_contract.SCHEMA_ID
    assert schema["additionalProperties"] is False


def test_checked_in_fixture_is_structurally_complete_but_never_authoritative() -> None:
    report = score_contract.validate_declaration_file(FIXTURE)

    assert report == {
        "schema": "score_ranking_declaration_v1",
        "declaration_id": "fixture_score_ranking_sr3_v1",
        "valid": True,
        "structurally_complete": True,
        "binding_fields_complete": False,
        "authority_verified": False,
        "activation_eligible": False,
        "target_rung": "SR3",
        "disposition": "fixture",
    }


def test_full_sr6_prefix_can_be_structurally_validated_without_granting_authority() -> (
    None
):
    report = score_contract.validate_declaration(_active_sr6())

    assert report["valid"] is True
    assert report["target_rung"] == "SR6"
    assert report["binding_fields_complete"] is True
    assert report["authority_verified"] is False
    assert report["activation_eligible"] is False
    assert report["disposition"] == "declaration_candidate"


def test_missing_policy_tuple_member_fails_schema() -> None:
    declaration = _valid()
    del declaration["policy"]["score"]["orientation"]
    _assert_error(declaration, "schema_rejection")


def test_unknown_enum_fails_schema() -> None:
    declaration = _valid()
    declaration["claim"]["target_rung"] = "SR7"
    _assert_error(declaration, "schema_rejection")


def test_unknown_field_fails_schema() -> None:
    declaration = _valid()
    declaration["policy"]["score"]["raw_score_escape_hatch"] = True
    _assert_error(declaration, "schema_rejection")


def test_duplicate_json_member_fails_before_schema(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"score_ranking_declaration_v1",'
        '"schema":"score_ranking_declaration_v1"}',
        encoding="utf-8",
    )

    with pytest.raises(score_contract.ScoreRankingValidationError) as raised:
        score_contract.load_json(path)

    assert raised.value.error_class == "duplicate_json_key"


def _assert_json_non_finite_fails(tmp_path: Path, value: float) -> None:
    declaration = _valid()
    declaration["claim"]["minimum_effect"]["value"] = value
    path = tmp_path / "non_finite.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")

    with pytest.raises(score_contract.ScoreRankingValidationError) as raised:
        score_contract.load_json(path)

    assert raised.value.error_class == "non_finite_number"


def test_nan_minimum_effect_fails_closed(tmp_path: Path) -> None:
    _assert_json_non_finite_fails(tmp_path, float("nan"))


def test_positive_infinity_minimum_effect_fails_closed(tmp_path: Path) -> None:
    _assert_json_non_finite_fails(tmp_path, float("inf"))


def test_in_memory_non_finite_number_fails_closed() -> None:
    declaration = _valid()
    declaration["claim"]["minimum_effect"]["value"] = float("-inf")
    _assert_error(declaration, "non_finite_number")


def test_in_memory_decimal_nan_fails_closed() -> None:
    declaration = _valid()
    declaration["claim"]["minimum_effect"]["value"] = Decimal("NaN")
    _assert_error(declaration, "non_finite_number")


def test_in_memory_decimal_infinity_fails_closed() -> None:
    declaration = _valid()
    declaration["claim"]["minimum_effect"]["value"] = Decimal("Infinity")
    _assert_error(declaration, "non_finite_number")


def test_identity_transform_cannot_hide_multiple_components() -> None:
    declaration = _valid()
    declaration["policy"]["score"]["transform_kind"] = "identity"
    _assert_error(declaration, "policy_transform")


def test_stable_tie_key_must_equal_candidate_identity_key() -> None:
    declaration = _valid()
    declaration["policy"]["tie_rule"]["key_fields"].pop()
    _assert_error(declaration, "tie_rule")


def test_assignment_cutoff_requires_assignment_space() -> None:
    declaration = _valid()
    cutoff = declaration["policy"]["cutoff"]
    cutoff["role"] = "assignment_rule"
    cutoff["imported_gate_id"] = None
    _assert_error(declaration, "cutoff_space")


def test_duplicate_reduction_edge_fails_closed() -> None:
    declaration = _active_sr6()
    duplicate = deepcopy(declaration["spaces"]["reductions"][0])
    duplicate["reduction_id"] = "fixture_duplicate_edge_v1"
    declaration["spaces"]["reductions"].append(duplicate)
    _assert_error(declaration, "duplicate_identity")


def test_assignment_space_requires_declared_reduction() -> None:
    declaration = _active_sr6()
    declaration["spaces"]["reductions"].pop(0)
    _assert_error(declaration, "reduction_graph")


def test_claim_space_cannot_exceed_target_rung() -> None:
    declaration = _valid()
    declaration["claim"]["claim_space"] = "system"
    _assert_error(declaration, "claim_above_rung")


def test_kappa_quantification_must_match_claim_space() -> None:
    declaration = _valid()
    declaration["claim"]["kappa"]["quantification_space"] = "assignment"
    _assert_error(declaration, "claim_space")


def test_assignment_metric_requires_assignment_space() -> None:
    declaration = _valid()
    declaration["claim"]["primary_ranking_metric"] = "correct_assignment_rate"
    _assert_error(declaration, "claim_space")


def test_top_k_metric_requires_predeclared_positive_k() -> None:
    declaration = _valid()
    declaration["claim"]["primary_ranking_metric"] = "top_k_gt_recall"
    _assert_error(declaration, "schema_rejection")

    declaration["claim"]["metric_parameters"] = {"top_k": 0}
    _assert_error(declaration, "schema_rejection")


def test_top_k_metric_accepts_predeclared_positive_k() -> None:
    declaration = _valid()
    declaration["claim"]["primary_ranking_metric"] = "top_k_gt_recall"
    declaration["claim"]["metric_parameters"] = {"top_k": 5}

    assert score_contract.validate_declaration(declaration)["valid"] is True


def test_non_top_k_metric_rejects_unused_top_k_parameter() -> None:
    declaration = _valid()
    declaration["claim"]["metric_parameters"] = {"top_k": 5}
    _assert_error(declaration, "schema_rejection")


def test_calibration_claim_discriminator_is_required() -> None:
    declaration = _valid()
    del declaration["calibration_claim"]
    _assert_error(declaration, "schema_rejection")


def test_no_calibration_claim_rejects_unbound_evidence_fields() -> None:
    declaration = _valid()
    declaration["calibration_claim"]["reference_id"] = "fixture_reference_v1"
    _assert_error(declaration, "schema_rejection")


def test_typed_calibration_claim_requires_complete_evidence_binding() -> None:
    declaration = _valid()
    declaration["calibration_claim"] = {"kind": "probabilistic"}
    _assert_error(declaration, "schema_rejection")

    declaration["calibration_claim"] = {
        "kind": "probabilistic",
        "reference_id": "fixture_gt_probability_target_v1",
        "calibration_unit_id": "fixture_candidate_event_unit_v1",
        "estimator_id": "fixture_isotonic_estimator_v1",
        "proper_score_id": "fixture_brier_score_v1",
        "held_out_rule_id": "fixture_held_out_calibration_v1",
        "minimum_exposure": 100,
    }
    assert score_contract.validate_declaration(declaration)["valid"] is True


def test_target_rung_requires_exact_lower_rung_prefix() -> None:
    declaration = _valid()
    declaration["rung_obligations"].pop(2)
    _assert_error(declaration, "rung_prefix")


def test_each_rung_uses_the_frozen_obligation_set() -> None:
    declaration = _valid()
    declaration["rung_obligations"][1]["obligation_ids"].pop()
    _assert_error(declaration, "rung_obligations")


def test_sr4_through_sr6_require_one_artifact_binding_per_obligation() -> None:
    declaration = _active_sr6()
    del declaration["rung_obligations"][4]["artifact_bindings"]["quantity_fidelity"]
    _assert_error(declaration, "rung_bindings")


def test_high_rung_obligations_cannot_share_one_artifact_identity() -> None:
    declaration = _active_sr6()
    bindings = declaration["rung_obligations"][4]["artifact_bindings"]
    bindings["quantity_fidelity"] = bindings["substrate_identity"]
    _assert_error(declaration, "duplicate_identity")


def test_high_rung_artifact_identity_cannot_be_reused_across_rungs() -> None:
    declaration = _active_sr6()
    sr4_bindings = declaration["rung_obligations"][4]["artifact_bindings"]
    sr5_bindings = declaration["rung_obligations"][5]["artifact_bindings"]
    sr5_bindings["online_hook"] = sr4_bindings["substrate_identity"]
    _assert_error(declaration, "duplicate_identity")


def test_sr0_through_sr3_reject_unscoped_artifact_bindings() -> None:
    declaration = _valid()
    declaration["rung_obligations"][3]["artifact_bindings"] = {
        "candidate_universe_invariance": "fixture_unscoped_binding_v1"
    }
    _assert_error(declaration, "rung_bindings")


def test_conservation_claim_cannot_be_false() -> None:
    declaration = _valid()
    declaration["conservation"]["pair_count_reconciliation"] = False
    _assert_error(declaration, "schema_rejection")


def test_terminal_partition_requires_all_three_outcome_classes() -> None:
    declaration = _valid()
    declaration["terminals"][1]["outcome_class"] = "invalid"
    _assert_error(declaration, "terminal_partition")


def test_valid_positive_terminal_must_transition_to_target_rung() -> None:
    declaration = _valid()
    declaration["terminals"][-1]["state_transition"] = {"kind": "none"}
    _assert_error(declaration, "terminal_transition")


def test_terminal_cannot_transition_above_target_rung() -> None:
    declaration = _valid()
    declaration["terminals"][0]["state_transition"] = {
        "kind": "transition",
        "target_state": "SR4",
    }
    _assert_error(declaration, "terminal_transition")


def test_duplicate_terminal_identity_fails_closed() -> None:
    declaration = _valid()
    declaration["terminals"][1]["terminal_id"] = declaration["terminals"][0][
        "terminal_id"
    ]
    _assert_error(declaration, "duplicate_identity")


def test_fixture_cannot_claim_active_owner_and_registry_authority() -> None:
    declaration = _valid()
    binding = declaration["contract_binding"]
    binding["contract_status"] = "active"
    binding["owner_acceptance_id"] = "fixture_owner_acceptance_v1"
    binding["registry_binding_id"] = "fixture_registry_binding_v1"
    _assert_error(declaration, "fixture_authority")


def test_active_binding_requires_both_acceptance_identities() -> None:
    declaration = _valid()
    binding = declaration["contract_binding"]
    binding["contract_status"] = "active"
    binding["owner_acceptance_id"] = "fixture_owner_acceptance_v1"
    _assert_error(declaration, "schema_rejection")
