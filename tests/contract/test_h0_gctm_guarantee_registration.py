from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "tools"
FIXTURE = (
    ROOT
    / "tests"
    / "contract"
    / "fixtures"
    / "h0_gctm_guarantee_registration_candidate_identity_v1.json"
)
sys.path.insert(0, TOOLS.as_posix())

import verify_h0_gctm_guarantee_registration as registration  # noqa: E402


PAIR_INSTANCE_KEY = [
    "seq",
    "frame",
    "cand_slot",
    "cand_instance_uid",
    "lost_slot",
    "lost_instance_uid",
]


def _candidate_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _registered_identity_record() -> dict[str, object]:
    """An in-memory branch test, not a checked-in H0 acceptance record."""
    return {
        "schema": "h0_gctm_guarantee_registration_v1",
        "record_id": "contract_registered_pair_identity_v1",
        "record_scope": "actual",
        "record_state": "registered-guarantee",
        "consumer": {
            "consumer_id": "contract_gctm_pair_identity",
            "maximum_claim_layer": "bridge_runtime_b1",
            "required_guarantee_ids": ["h0_pair_instance_identity_v1"],
        },
        "baseline": {
            "baseline_id": "contract_h0_baseline_identity_v1",
            "h0_terminal": "H0_FULL_COMMIT_CAPTURE_FAITHFUL",
            "h0_evidence_id": "contract_h0_evidence_identity_v1",
            "h0_packet_hash": "a" * 64,
            "h0_schema_version": "h0_bridge_decision_trace_v2",
            "runtime_instrumentation_identity": "contract_runtime_identity_v1",
            "policy_base_id": "contract_policy_base_v1",
            "resolved_preset_id": "contract_preset_m_v1",
            "capture_mode": "contract_capture_mode_v1",
            "event_key_version": "h0_event_key_v1",
            "observation_state_semantics_version": "h0_observation_state_v1",
            "dataset_sequence_domain": "contract_sequence_domain_v1",
            "accepted_by": "h0_owner",
            "accepted_at": "2026-07-19T00:00:00Z",
        },
        "guarantees": [
            {
                "guarantee_id": "h0_pair_instance_identity_v1",
                "guarantee_class": "identity",
                "consumer_object": "event_runtime_instance_identity",
                "stream": "pair_record",
                "covered_fields": PAIR_INSTANCE_KEY,
                "relation": "exact",
                "causal_availability": "online",
                "declared_domain": {
                    "preset_id": "contract_preset_m_v1",
                    "runtime_identity": "contract_runtime_identity_v1",
                    "schema_id": "h0_bridge_decision_trace_v2",
                    "dataset_sequence_domain": "contract_sequence_domain_v1",
                },
                "basis": ["replay", "shadow_nonperturbation"],
                "invalidation_inputs": sorted(
                    registration.IDENTITY_REQUIRED_INVALIDATION_INPUTS
                ),
            }
        ],
    }


def test_candidate_source_fixture_is_valid_but_not_usable() -> None:
    report = registration.verify_record_file(FIXTURE)

    assert report == {
        "schema": "h0_gctm_guarantee_registration_v1",
        "record_id": "fixture_h0_gctm_candidate_pair_identity_v1",
        "valid": True,
        "structurally_usable": False,
        "authority_verified": False,
        "disposition": "candidate-source",
    }


def test_fully_bound_owner_accepted_identity_record_is_structurally_usable() -> None:
    report = registration.validate_record(_registered_identity_record())

    assert report["valid"] is True
    assert report["structurally_usable"] is True
    assert report["authority_verified"] is False
    assert report["disposition"] == "registered-guarantee"


@pytest.mark.parametrize(
    "missing_input", sorted(registration.IDENTITY_REQUIRED_INVALIDATION_INPUTS)
)
def test_each_required_identity_invalidation_input_is_required(
    missing_input: str,
) -> None:
    record = _registered_identity_record()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["invalidation_inputs"].remove(missing_input)

    with pytest.raises(
        registration.RegistrationValidationError, match="identity invalidation_inputs"
    ):
        registration.validate_record(record)


def test_arbitrary_invalidation_input_is_rejected() -> None:
    record = _registered_identity_record()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["invalidation_inputs"] = ["anything"]

    with pytest.raises(
        registration.RegistrationValidationError, match="identity invalidation_inputs"
    ):
        registration.validate_record(record)


def test_guarantee_domain_must_match_the_accepted_baseline() -> None:
    record = _registered_identity_record()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["declared_domain"]["preset_id"] = "different_preset"

    with pytest.raises(
        registration.RegistrationValidationError, match="declared_domain"
    ):
        registration.validate_record(record)


def test_second_unvalidated_guarantee_is_rejected() -> None:
    record = _registered_identity_record()
    guarantees = record["guarantees"]
    consumer = record["consumer"]
    assert isinstance(guarantees, list)
    assert isinstance(consumer, dict)
    unvalidated = deepcopy(guarantees[0])
    unvalidated["guarantee_id"] = "h0_unvalidated_runtime_state_v1"
    unvalidated["guarantee_class"] = "runtime_state"
    guarantees.append(unvalidated)
    consumer["required_guarantee_ids"].append("h0_unvalidated_runtime_state_v1")

    with pytest.raises(registration.RegistrationValidationError, match="too long"):
        registration.validate_record(record)


def test_visible_track_id_cannot_be_registered_as_runtime_instance_identity() -> None:
    record = _registered_identity_record()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["covered_fields"] = [
        "seq",
        "frame",
        "cand_slot",
        "cand_instance_uid",
        "lost_slot",
        "visible_track_id",
    ]

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_fixture_scope_can_never_claim_registered_guarantee() -> None:
    record = _registered_identity_record()
    record["record_scope"] = "fixture_only"

    with pytest.raises(registration.RegistrationValidationError, match="record_scope"):
        registration.validate_record(record)


def test_unknown_fields_fail_closed() -> None:
    record = deepcopy(_registered_identity_record())
    record["baseline"]["unexpected"] = "forbidden"

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)
