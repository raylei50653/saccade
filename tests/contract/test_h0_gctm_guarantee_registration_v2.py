"""Contract for the H0 GCTM guarantee-registration verifier (candidate-sources v2)."""

# scope: system
# function: contract
# lifecycle: active

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
    / "h0_gctm_guarantee_registration_candidate_sources_v2.json"
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
BASELINE = {
    "baseline_id": "contract_h0_baseline_v2",
    "h0_terminal": "H0_FULL_COMMIT_CAPTURE_FAITHFUL",
    "h0_evidence_id": "contract_h0_evidence_v2",
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
    "accepted_at": "2026-07-20T00:00:00Z",
}
DECLARED_DOMAIN = {
    "preset_id": "contract_preset_m_v1",
    "runtime_identity": "contract_runtime_identity_v1",
    "schema_id": "h0_bridge_decision_trace_v2",
    "dataset_sequence_domain": "contract_sequence_domain_v1",
}


def _candidate_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _guarantee(
    guarantee_id: str,
    guarantee_class: str,
    consumer_object: str,
    stream: str,
    covered_fields: list[str],
    relation: str = "exact",
) -> dict[str, object]:
    invalidation = registration._required_invalidation_inputs(guarantee_class, relation)
    return {
        "guarantee_id": guarantee_id,
        "guarantee_class": guarantee_class,
        "consumer_object": consumer_object,
        "stream": stream,
        "covered_fields": covered_fields,
        "relation": relation,
        "causal_availability": "online",
        "declared_domain": dict(DECLARED_DOMAIN),
        "basis": ["replay", "shadow_nonperturbation"],
        "invalidation_inputs": sorted(invalidation),
    }


def _registered_record_v2() -> dict[str, object]:
    """An in-memory branch test, not a checked-in H0 acceptance record."""
    guarantees = [
        _guarantee(
            "h0_pair_instance_identity_v2",
            "identity",
            "event_runtime_instance_identity",
            "pair_record",
            list(PAIR_INSTANCE_KEY),
        ),
        _guarantee(
            "h0_pair_exit_entry_snapshot_v2",
            "snapshot",
            "native_exit_entry_snapshot",
            "pair_record",
            ["lost_anchor_x", "lost_anchor_y", "lost_velocity_x", "lost_velocity_y"],
        ),
        _guarantee(
            "h0_pair_operational_horizon_v2",
            "timing",
            "operational_horizon_observation_point",
            "pair_record",
            ["seq", "frame", "la", "bridge_at"],
        ),
        _guarantee(
            "h0_pair_score_context_v2",
            "competition",
            "candidate_competition_pair_score_context",
            "pair_record",
            ["bdist_after_direction", "final_pair_eligible", "reject_reason"],
        ),
        _guarantee(
            "h0_candidate_competition_context_v2",
            "competition",
            "candidate_competition_pair_score_context",
            "candidate_record",
            ["seq", "frame", "cand_slot", "best_bdist", "margin", "proposal_emitted"],
        ),
        _guarantee(
            "h0_claim_audit_boundary_v2",
            "audit",
            "claim_commit_audit_boundary",
            "claim_record",
            ["seq", "frame", "detection_score", "packed_atomic_key", "claim_won"],
        ),
        _guarantee(
            "h0_commit_audit_boundary_v2",
            "audit",
            "claim_commit_audit_boundary",
            "commit_record",
            ["seq", "frame", "commit_executed", "lost_slot_deactivated"],
        ),
    ]
    return {
        "schema": "h0_gctm_guarantee_registration_v2",
        "record_id": "contract_registered_guarantees_v2",
        "record_scope": "actual",
        "record_state": "registered-guarantee",
        "consumer": {
            "consumer_id": "contract_gctm_consumer_v2",
            "maximum_claim_layer": "bridge_runtime_b1",
            "required_guarantee_ids": [g["guarantee_id"] for g in guarantees],
        },
        "baseline": dict(BASELINE),
        "guarantees": guarantees,
    }


def test_candidate_sources_fixture_is_valid_but_not_usable() -> None:
    report = registration.verify_record_file(FIXTURE)

    assert report == {
        "schema": "h0_gctm_guarantee_registration_v2",
        "record_id": "fixture_h0_gctm_candidate_sources_v2",
        "valid": True,
        "structurally_usable": False,
        "authority_verified": False,
        "disposition": "candidate-source",
    }


def test_fixture_covers_the_full_candidate_inventory() -> None:
    record = _candidate_fixture()
    sources = record["candidate_sources"]
    assert isinstance(sources, list)
    coordinates = {(s["consumer_object"], s["stream"]) for s in sources}

    expected = {
        (obj, stream)
        for _, (obj, streams) in registration.V2_CLASS_SPECS.items()
        for stream in streams
    }
    assert coordinates == expected


def test_every_fixture_field_exists_in_the_capture_abi() -> None:
    record = _candidate_fixture()
    sources = record["candidate_sources"]
    assert isinstance(sources, list)
    for source in sources:
        abi_fields = registration._capture_abi_fields(source["stream"])
        assert set(source["source_fields"]) <= abi_fields


def test_fully_bound_multi_class_record_is_structurally_usable() -> None:
    report = registration.validate_record(_registered_record_v2())

    assert report["valid"] is True
    assert report["structurally_usable"] is True
    assert report["authority_verified"] is False
    assert report["disposition"] == "registered-guarantee"


DERIVATION_BINDING = {
    "definition_id": "contract_snapshot_derivation_v1",
    "definition_version": "1",
    "content_hash": "b" * 64,
}


def test_snapshot_derived_relation_requires_derivation_invalidation() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    snapshot = guarantees[1]
    snapshot["relation"] = "derived"
    snapshot["derivation"] = dict(DERIVATION_BINDING)

    with pytest.raises(
        registration.RegistrationValidationError, match="snapshot invalidation_inputs"
    ):
        registration.validate_record(record)

    snapshot["invalidation_inputs"] = sorted(
        set(snapshot["invalidation_inputs"]) | {"derivation_definition"}
    )
    report = registration.validate_record(record)
    assert report["structurally_usable"] is True


def test_derived_relation_requires_an_immutable_derivation_binding() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    snapshot = guarantees[1]
    snapshot["relation"] = "derived"
    snapshot["invalidation_inputs"] = sorted(
        set(snapshot["invalidation_inputs"]) | {"derivation_definition"}
    )

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)

    incomplete = {"definition_id": "contract_snapshot_derivation_v1"}
    snapshot["derivation"] = incomplete
    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)

    snapshot["derivation"] = dict(DERIVATION_BINDING)
    report = registration.validate_record(record)
    assert report["structurally_usable"] is True


def test_exact_relation_forbids_a_derivation_binding() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[1]["derivation"] = dict(DERIVATION_BINDING)

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_baseline_schema_version_must_match_the_validated_capture_abi() -> None:
    record = _registered_record_v2()
    baseline = record["baseline"]
    guarantees = record["guarantees"]
    assert isinstance(baseline, dict)
    assert isinstance(guarantees, list)
    baseline["h0_schema_version"] = "some_other_schema"
    for guarantee in guarantees:
        guarantee["declared_domain"]["schema_id"] = "some_other_schema"

    with pytest.raises(registration.RegistrationValidationError, match="capture ABI"):
        registration.validate_record(record)


def test_non_snapshot_classes_reject_derived_relation() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[2]["relation"] = "derived"

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_fields_outside_the_sealed_allowlist_are_rejected() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[1]["covered_fields"] = ["lost_anchor_x", "detection_score"]

    with pytest.raises(
        registration.RegistrationValidationError, match="sealed allowlist"
    ):
        registration.validate_record(record)


def test_track_id_is_rejected_outside_the_audit_boundary() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[4]["covered_fields"] = ["seq", "frame", "best_lost_precommit_track_id"]

    with pytest.raises(registration.RegistrationValidationError, match="track_id"):
        registration.validate_record(record)


def test_track_id_is_registrable_as_audit_observation_only() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[5]["covered_fields"] = [
        "seq",
        "frame",
        "winning_cand_precommit_track_id",
        "claim_won",
    ]

    report = registration.validate_record(record)
    assert report["structurally_usable"] is True


def test_identity_guarantee_keeps_the_sealed_pair_key() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["covered_fields"] = PAIR_INSTANCE_KEY[:-1] + ["h_ref"]

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_duplicate_class_stream_guarantee_is_rejected() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    consumer = record["consumer"]
    assert isinstance(guarantees, list)
    assert isinstance(consumer, dict)
    duplicate = deepcopy(guarantees[2])
    duplicate["guarantee_id"] = "h0_pair_operational_horizon_v2_dup"
    removed = guarantees.pop()
    guarantees.append(duplicate)
    consumer["required_guarantee_ids"].remove(removed["guarantee_id"])
    consumer["required_guarantee_ids"].append(duplicate["guarantee_id"])

    with pytest.raises(
        registration.RegistrationValidationError, match="duplicate guarantee"
    ):
        registration.validate_record(record)


def test_required_guarantee_ids_must_bind_every_registered_guarantee() -> None:
    record = _registered_record_v2()
    consumer = record["consumer"]
    assert isinstance(consumer, dict)
    consumer["required_guarantee_ids"] = consumer["required_guarantee_ids"][:-1]

    with pytest.raises(
        registration.RegistrationValidationError, match="required_guarantee_ids"
    ):
        registration.validate_record(record)


def test_guarantee_domain_must_match_the_accepted_baseline() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[3]["declared_domain"]["preset_id"] = "different_preset"

    with pytest.raises(
        registration.RegistrationValidationError, match="declared_domain"
    ):
        registration.validate_record(record)


def test_wrong_stream_for_class_is_rejected() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[1]["stream"] = "claim_record"

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_each_shared_invalidation_input_is_required() -> None:
    record = _registered_record_v2()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    for missing in sorted(registration.SHARED_INVALIDATION_INPUTS):
        mutated = deepcopy(record)
        mutated_guarantees = mutated["guarantees"]
        assert isinstance(mutated_guarantees, list)
        mutated_guarantees[2]["invalidation_inputs"].remove(missing)

        with pytest.raises(
            registration.RegistrationValidationError,
            match="timing invalidation_inputs",
        ):
            registration.validate_record(mutated)


def test_fixture_scope_can_never_claim_registered_guarantee() -> None:
    record = _registered_record_v2()
    record["record_scope"] = "fixture_only"

    with pytest.raises(registration.RegistrationValidationError, match="record_scope"):
        registration.validate_record(record)


def test_unknown_fields_fail_closed() -> None:
    record = deepcopy(_registered_record_v2())
    record["baseline"]["unexpected"] = "forbidden"

    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_unsupported_schema_version_is_rejected() -> None:
    record = _registered_record_v2()
    record["schema"] = "h0_gctm_guarantee_registration_v3"

    with pytest.raises(
        registration.RegistrationValidationError, match="unsupported registration"
    ):
        registration.validate_record(record)
