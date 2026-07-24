"""Contract for the H0 GCTM guarantee-registration verifier (universe-completeness v3)."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

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
    / "h0_gctm_guarantee_registration_candidate_universe_v3.json"
)
V1_FIXTURE = (
    ROOT
    / "tests"
    / "contract"
    / "fixtures"
    / "h0_gctm_guarantee_registration_candidate_identity_v1.json"
)
V2_FIXTURE = (
    ROOT
    / "tests"
    / "contract"
    / "fixtures"
    / "h0_gctm_guarantee_registration_candidate_sources_v2.json"
)
sys.path.insert(0, TOOLS.as_posix())

import verify_h0_gctm_guarantee_registration as registration  # noqa: E402


CAPTURE_RUN_UUID = "contract-capture-run-uuid-v3-0001"
BASELINE = {
    "baseline_id": "contract_h0_baseline_universe_v3",
    "h0_terminal": "H0_FULL_COMMIT_CAPTURE_FAITHFUL",
    "h0_evidence_id": "contract_h0_evidence_universe_v3",
    "h0_packet_hash": "a" * 64,
    "h0_schema_version": "h0_bridge_decision_trace_v2",
    "runtime_instrumentation_identity": "contract_runtime_identity_v1",
    "policy_base_id": "contract_policy_base_v1",
    "resolved_preset_id": "contract_preset_m_v1",
    "capture_mode": "contract_capture_mode_v1",
    "capture_run_uuid": CAPTURE_RUN_UUID,
    "event_key_version": "gctm_runtime_event_key_v1",
    "dataset_sequence_domain": "contract_sequence_domain_v1",
    "accepted_by": "h0_owner",
    "accepted_at": "2026-07-24T00:00:00Z",
}
DECLARED_DOMAIN = {
    "preset_id": "contract_preset_m_v1",
    "runtime_identity": "contract_runtime_identity_v1",
    "schema_id": "h0_bridge_decision_trace_v2",
    "dataset_sequence_domain": "contract_sequence_domain_v1",
    "consumer_universe_id": "gctm_runtime_native_candidate_universe_v1",
}

# Two lost events on the same frame.
EVENT_E1 = {
    "seq": 0,
    "frame": 10,
    "lost_slot": 2,
    "lost_instance_uid": 200,
    "event_key_version": "gctm_runtime_event_key_v1",
}
EVENT_E2 = {
    "seq": 0,
    "frame": 10,
    "lost_slot": 4,
    "lost_instance_uid": 400,
    "event_key_version": "gctm_runtime_event_key_v1",
}

# Native exposure candidates (frame-local; may join 0..N events).
CAND_A = {"seq": 0, "frame": 10, "cand_slot": 1, "cand_instance_uid": 100}
CAND_B = {"seq": 0, "frame": 10, "cand_slot": 3, "cand_instance_uid": 101}
CAND_C = {"seq": 0, "frame": 10, "cand_slot": 5, "cand_instance_uid": 102}

# Pairs: A joins both events; B joins only E1; C has no pre-score pairs.
PAIR_A_E1 = {
    "seq": 0,
    "frame": 10,
    "cand_slot": 1,
    "cand_instance_uid": 100,
    "lost_slot": 2,
    "lost_instance_uid": 200,
}
PAIR_A_E2 = {
    "seq": 0,
    "frame": 10,
    "cand_slot": 1,
    "cand_instance_uid": 100,
    "lost_slot": 4,
    "lost_instance_uid": 400,
}
PAIR_B_E1 = {
    "seq": 0,
    "frame": 10,
    "cand_slot": 3,
    "cand_instance_uid": 101,
    "lost_slot": 2,
    "lost_instance_uid": 200,
}


def _source_bindings() -> list[dict[str, object]]:
    return [
        {
            "stream": "pair_record",
            "source_fields": sorted(registration.V3_PAIR_COMPLETENESS_FIELDS),
        },
        {
            "stream": "candidate_record",
            "source_fields": sorted(registration.V3_CANDIDATE_COMPLETENESS_FIELDS),
        },
        {
            "stream": "event_universe_sidecar",
            "source_fields": sorted(registration.EVENT_UNIVERSE_SIDECAR_FIELDS),
        },
    ]


def _completeness_evidence() -> dict[str, object]:
    """Multi-event same exposure + zero pre_score_passes exposure."""
    return {
        "capture_run_uuid": CAPTURE_RUN_UUID,
        "capture_schema_version": "h0_bridge_decision_trace_v2",
        "native_pair_keys": [dict(PAIR_A_E1), dict(PAIR_A_E2), dict(PAIR_B_E1)],
        "pair_record_keys": [dict(PAIR_A_E1), dict(PAIR_A_E2), dict(PAIR_B_E1)],
        "pre_score_eligible_pair_keys": [
            dict(PAIR_A_E1),
            dict(PAIR_A_E2),
            dict(PAIR_B_E1),
        ],
        "native_candidate_keys": [dict(CAND_A), dict(CAND_B), dict(CAND_C)],
        "candidate_record_keys": [dict(CAND_A), dict(CAND_B), dict(CAND_C)],
        "candidate_pre_score_passes": [
            {"candidate_key": dict(CAND_A), "pre_score_passes": 2},
            {"candidate_key": dict(CAND_B), "pre_score_passes": 1},
            {"candidate_key": dict(CAND_C), "pre_score_passes": 0},
        ],
        "retained_candidate_membership": [
            {
                "event_key": dict(EVENT_E1),
                "cand_slot": 1,
                "cand_instance_uid": 100,
            },
            {
                "event_key": dict(EVENT_E2),
                "cand_slot": 1,
                "cand_instance_uid": 100,
            },
            {
                "event_key": dict(EVENT_E1),
                "cand_slot": 3,
                "cand_instance_uid": 101,
            },
        ],
        "totals": {
            "total_native_pair_keys": 3,
            "total_pair_records": 3,
            "total_native_candidate_keys": 3,
            "total_candidate_records": 3,
            "overflow_native_pair_keys": 0,
            "overflow_pair_records": 0,
            "overflow_native_candidate_keys": 0,
            "overflow_candidate_records": 0,
            "bridge_attempt_count": 3,
            "identity_uid_wrap_events": 0,
        },
        "partial_events": [],
        "silent_truncation_possible": False,
        "claim_commit_label_participation": False,
        "exposure_predicates": {
            "require_candidate_exposure": True,
            "bridge_attempt_equals_native_candidate_keys": True,
        },
        "m0_m1_m2_identical_universe": {
            "same_event_set": True,
            "same_C_e": True,
            "event_set_fingerprint": "contract_event_set_fp_v1",
            "c_e_fingerprint": "contract_c_e_fp_v1",
        },
        "inclusion_stage_score_independent": True,
    }


def _guarantee(guarantee_id: str, consumer_object: str) -> dict[str, object]:
    return {
        "guarantee_id": guarantee_id,
        "guarantee_class": "universe_completeness",
        "consumer_object": consumer_object,
        "consumer_universe_id": "gctm_runtime_native_candidate_universe_v1",
        "baseline_id": BASELINE["baseline_id"],
        "runtime_instrumentation_identity": BASELINE[
            "runtime_instrumentation_identity"
        ],
        "resolved_preset_id": BASELINE["resolved_preset_id"],
        "h0_schema_version": BASELINE["h0_schema_version"],
        "capture_run_uuid": CAPTURE_RUN_UUID,
        "dataset_sequence_domain": BASELINE["dataset_sequence_domain"],
        "event_key_version": "gctm_runtime_event_key_v1",
        "candidate_key_version": "gctm_runtime_candidate_key_v1",
        "inclusion_stage_identity": "pre_score_eligible_v1",
        "mask_identity": "h0_pair_pre_score_mask_v1",
        "gate_retained_band_identity": "native_non_score_gates_height_speed_spatial_v1",
        "source_bindings": _source_bindings(),
        "completeness_predicate_id": "h0_native_universe_completeness_predicate_v1",
        "replay_non_perturbation_basis": ["replay", "shadow_nonperturbation"],
        "causal_availability": "online",
        "declared_domain": dict(DECLARED_DOMAIN),
        "invalidation_inputs": sorted(registration.V3_REQUIRED_INVALIDATION_INPUTS),
        "completeness_evidence": _completeness_evidence(),
    }


def _registered_record_v3() -> dict[str, object]:
    """In-memory structural branch only — not a checked-in accepted H0 guarantee."""
    guarantees = [
        _guarantee(
            "h0_runtime_candidate_universe_completeness_contract_v1",
            "runtime_candidate_universe",
        ),
        _guarantee(
            "h0_runtime_event_membership_completeness_contract_v1",
            "runtime_event_membership",
        ),
    ]
    return {
        "schema": "h0_gctm_guarantee_registration_v3",
        "record_id": "contract_registered_universe_completeness_v3",
        "record_scope": "actual",
        "record_state": "registered-guarantee",
        "consumer": {
            "consumer_id": "contract_gctm_consumer_v3",
            "consumer_universe_id": "gctm_runtime_native_candidate_universe_v1",
            "maximum_claim_layer": "bridge_runtime_b1",
            "required_guarantee_ids": [g["guarantee_id"] for g in guarantees],
        },
        "identity_bindings": dict(registration.V3_IDENTITY_BINDINGS),
        "baseline": dict(BASELINE),
        "guarantees": guarantees,
    }


def _candidate_fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_candidate_source_fixture_is_valid_but_not_usable() -> None:
    report = registration.verify_record_file(FIXTURE)

    assert report == {
        "schema": "h0_gctm_guarantee_registration_v3",
        "record_id": "fixture_h0_gctm_candidate_universe_completeness_v3",
        "valid": True,
        "structurally_usable": False,
        "authority_verified": False,
        "runtime_substrate_established": False,
        "runtime_compatibility_established": False,
        "h0_reentry_authorized": False,
        "disposition": "candidate-source",
    }


def test_fixture_covers_completeness_streams_and_objects() -> None:
    record = _candidate_fixture()
    sources = record["candidate_sources"]
    assert isinstance(sources, list)
    coordinates = {(s["consumer_object"], s["stream"]) for s in sources}
    expected = {
        (obj, stream)
        for obj in registration.V3_CONSUMER_OBJECTS
        for stream in registration.V3_STREAMS
    }
    assert coordinates == expected


def test_sidecar_fields_exist_in_capture_abi_envelope() -> None:
    envelope = registration._capture_abi_envelope_fields()
    assert registration.EVENT_UNIVERSE_SIDECAR_FIELDS <= envelope


def test_structurally_complete_registered_guarantee_is_usable_not_authoritative() -> (
    None
):
    report = registration.validate_record(_registered_record_v3())

    assert report == {
        "schema": "h0_gctm_guarantee_registration_v3",
        "record_id": "contract_registered_universe_completeness_v3",
        "valid": True,
        "structurally_usable": True,
        "authority_verified": False,
        "runtime_substrate_established": False,
        "runtime_compatibility_established": False,
        "h0_reentry_authorized": False,
        "disposition": "registered-guarantee",
    }


def test_same_exposure_candidate_may_join_multiple_events() -> None:
    """Consumer candidate_key is event-local; same exposure across events is legal."""
    report = registration.validate_record(_registered_record_v3())
    assert report["structurally_usable"] is True
    evidence = _completeness_evidence()
    membership = evidence["retained_candidate_membership"]
    a_events = [
        m["event_key"]["lost_slot"]
        for m in membership
        if m["cand_slot"] == 1 and m["cand_instance_uid"] == 100
    ]
    assert sorted(a_events) == [2, 4]


def test_zero_pre_score_passes_has_no_membership() -> None:
    report = registration.validate_record(_registered_record_v3())
    assert report["structurally_usable"] is True
    evidence = _completeness_evidence()
    membership_exposures = {
        (
            m["event_key"]["seq"],
            m["event_key"]["frame"],
            m["cand_slot"],
            m["cand_instance_uid"],
        )
        for m in evidence["retained_candidate_membership"]
    }
    assert (0, 10, 5, 102) not in membership_exposures
    passes = {
        (
            row["candidate_key"]["cand_slot"],
            row["candidate_key"]["cand_instance_uid"],
            row["pre_score_passes"],
        )
        for row in evidence["candidate_pre_score_passes"]
    }
    assert (5, 102, 0) in passes


def test_contract_audit_selects_sealable_without_abi_delta() -> None:
    audit = registration.audit_registration_v3_contract()
    assert audit["selected_terminal"] == registration.TERMINAL_V3_CONTRACT_SEALABLE
    assert audit["trace_v2_abi_change_required"] is False
    assert audit["actual_guarantee_established"] is False
    assert audit["runtime_compatibility_established"] is False
    assert audit["abi_coverage"]["all_available"] is True
    basis = audit["selection_basis"]
    assert basis["candidate_source_fixture_valid"] is True
    assert basis["registered_positive_branch_structurally_usable"] is True
    assert basis["multi_event_same_exposure_accepted"] is True
    assert basis["zero_pre_score_passes_accepted"] is True
    assert basis["negative_catalog_fail_closed"] is True
    assert basis["completeness_predicate_complete"] is True
    assert audit["audit_issues"] == []


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_wrong_consumer_universe_identity() -> None:
    record = _registered_record_v3()
    record["identity_bindings"]["consumer_universe_id"] = "wrong_universe"
    with pytest.raises(
        registration.RegistrationValidationError,
        match="schema rejection|consumer_universe",
    ):
        registration.validate_record(record)


def test_wrong_event_or_candidate_key_version() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["event_key_version"] = "wrong_event_key"
    with pytest.raises(
        registration.RegistrationValidationError,
        match="schema rejection|event_key_version",
    ):
        registration.validate_record(record)

    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["candidate_key_version"] = "wrong_candidate_key"
    with pytest.raises(
        registration.RegistrationValidationError,
        match="schema rejection|candidate_key_version",
    ):
        registration.validate_record(record)


def test_post_score_inclusion_stage_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["inclusion_stage_identity"] = "final_eligible_set"
    with pytest.raises(
        registration.RegistrationValidationError,
        match="schema rejection|inclusion_stage",
    ):
        registration.validate_record(record)


def test_missing_pair_candidate_or_sidecar_binding() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["source_bindings"] = guarantees[0]["source_bindings"][:2]
    with pytest.raises(
        registration.RegistrationValidationError,
        match="source_bindings|schema rejection",
    ):
        registration.validate_record(record)


def test_count_mismatch() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["completeness_evidence"]["totals"]["total_pair_records"] = 99
    with pytest.raises(
        registration.RegistrationValidationError, match="total_pair_records"
    ):
        registration.validate_record(record)


def test_native_key_without_emitted_row() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    evidence = guarantees[0]["completeness_evidence"]
    evidence["pair_record_keys"] = [dict(PAIR_A_E1), dict(PAIR_B_E1)]
    evidence["totals"]["total_pair_records"] = 2
    with pytest.raises(
        registration.RegistrationValidationError, match="native_pair_key"
    ):
        registration.validate_record(record)


def test_emitted_row_without_native_key() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    evidence = guarantees[0]["completeness_evidence"]
    evidence["native_pair_keys"] = [dict(PAIR_A_E1), dict(PAIR_B_E1)]
    evidence["totals"]["total_native_pair_keys"] = 2
    with pytest.raises(
        registration.RegistrationValidationError, match="native_pair_key"
    ):
        registration.validate_record(record)


def test_duplicate_candidate_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    evidence = guarantees[0]["completeness_evidence"]
    evidence["candidate_record_keys"] = [dict(CAND_A), dict(CAND_A), dict(CAND_B)]
    evidence["native_candidate_keys"] = [dict(CAND_A), dict(CAND_A), dict(CAND_B)]
    evidence["totals"]["total_native_candidate_keys"] = 3
    evidence["totals"]["total_candidate_records"] = 3
    evidence["totals"]["bridge_attempt_count"] = 3
    with pytest.raises(
        registration.RegistrationValidationError, match="duplicate candidate"
    ):
        registration.validate_record(record)


def test_duplicate_event_local_membership_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    evidence = guarantees[0]["completeness_evidence"]
    evidence["retained_candidate_membership"].append(
        {
            "event_key": dict(EVENT_E1),
            "cand_slot": 1,
            "cand_instance_uid": 100,
        }
    )
    with pytest.raises(
        registration.RegistrationValidationError,
        match="duplicate event-local candidate membership",
    ):
        registration.validate_record(record)


def test_membership_must_equal_pre_score_pair_projection() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    evidence = guarantees[0]["completeness_evidence"]
    # Drop one membership while leaving the pre-score pair.
    evidence["retained_candidate_membership"] = evidence[
        "retained_candidate_membership"
    ][:2]
    with pytest.raises(
        registration.RegistrationValidationError,
        match="projection of pre_score_eligible_pair_keys|pre_score_passes",
    ):
        registration.validate_record(record)


def test_pre_score_passes_mismatch_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    evidence = guarantees[0]["completeness_evidence"]
    evidence["candidate_pre_score_passes"][0]["pre_score_passes"] = 1
    with pytest.raises(
        registration.RegistrationValidationError, match="pre_score_passes"
    ):
        registration.validate_record(record)


def test_overflow_or_truncation_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["completeness_evidence"]["totals"]["overflow_native_pair_keys"] = 1
    with pytest.raises(
        registration.RegistrationValidationError, match="overflow_native_pair_keys"
    ):
        registration.validate_record(record)

    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["completeness_evidence"]["silent_truncation_possible"] = True
    with pytest.raises(
        registration.RegistrationValidationError, match="silent truncation"
    ):
        registration.validate_record(record)


def test_partial_event_fail_closed() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["completeness_evidence"]["partial_events"] = [dict(EVENT_E1)]
    with pytest.raises(
        registration.RegistrationValidationError, match="partial events"
    ):
        registration.validate_record(record)


def test_uid_wrap_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["completeness_evidence"]["totals"]["identity_uid_wrap_events"] = 1
    with pytest.raises(
        registration.RegistrationValidationError, match="identity_uid_wrap_events"
    ):
        registration.validate_record(record)


def test_claim_commit_derived_membership_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["completeness_evidence"]["claim_commit_label_participation"] = True
    with pytest.raises(
        registration.RegistrationValidationError, match="claim/commit/label"
    ):
        registration.validate_record(record)

    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["source_bindings"][0]["source_fields"] = [
        "seq",
        "frame",
        "cand_slot",
        "cand_instance_uid",
        "lost_slot",
        "lost_instance_uid",
        "reject_reason",
        "margin",
    ]
    with pytest.raises(
        registration.RegistrationValidationError,
        match="forbidden|allowlist|schema rejection",
    ):
        registration.validate_record(record)


def test_label_derived_membership_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["source_bindings"][1]["source_fields"] = [
        "seq",
        "frame",
        "cand_slot",
        "cand_instance_uid",
        "true_match_label",
    ]
    with pytest.raises(
        registration.RegistrationValidationError,
        match="forbidden|allowlist|schema rejection",
    ):
        registration.validate_record(record)


def test_missing_replay_non_perturbation_basis() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["replay_non_perturbation_basis"] = ["replay"]
    with pytest.raises(
        registration.RegistrationValidationError,
        match="replay_non_perturbation_basis|schema rejection",
    ):
        registration.validate_record(record)


def test_wrong_causal_availability() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["causal_availability"] = "offline_only"
    with pytest.raises(
        registration.RegistrationValidationError, match="causal_availability"
    ):
        registration.validate_record(record)


def test_incomplete_or_extra_invalidation_set() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["invalidation_inputs"] = sorted(
        registration.V3_REQUIRED_INVALIDATION_INPUTS - {"capture_run_uuid"}
    )
    with pytest.raises(
        registration.RegistrationValidationError, match="invalidation_inputs"
    ):
        registration.validate_record(record)

    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["invalidation_inputs"] = sorted(
        registration.V3_REQUIRED_INVALIDATION_INPUTS | {"unexpected_extra"}
    )
    with pytest.raises(
        registration.RegistrationValidationError, match="invalidation_inputs"
    ):
        registration.validate_record(record)


def test_v3_class_with_v2_consumer_object_rejected() -> None:
    record = _registered_record_v3()
    guarantees = record["guarantees"]
    assert isinstance(guarantees, list)
    guarantees[0]["consumer_object"] = "event_runtime_instance_identity"
    with pytest.raises(
        registration.RegistrationValidationError,
        match="schema rejection|consumer_object",
    ):
        registration.validate_record(record)


def test_v2_record_attempting_universe_completeness_rejected() -> None:
    record = {
        "schema": "h0_gctm_guarantee_registration_v2",
        "record_id": "contract_v2_universe_attempt",
        "record_scope": "actual",
        "record_state": "registered-guarantee",
        "consumer": {
            "consumer_id": "contract_v2_universe_attempt",
            "maximum_claim_layer": "bridge_runtime_b1",
            "required_guarantee_ids": ["h0_universe_attempt"],
        },
        "baseline": {
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
        },
        "guarantees": [
            {
                "guarantee_id": "h0_universe_attempt",
                "guarantee_class": "universe_completeness",
                "consumer_object": "runtime_candidate_universe",
                "stream": "pair_record",
                "covered_fields": ["seq", "frame"],
                "relation": "exact",
                "causal_availability": "online",
                "declared_domain": {
                    "preset_id": "contract_preset_m_v1",
                    "runtime_identity": "contract_runtime_identity_v1",
                    "schema_id": "h0_bridge_decision_trace_v2",
                    "dataset_sequence_domain": "contract_sequence_domain_v1",
                },
                "basis": ["replay", "shadow_nonperturbation"],
                "invalidation_inputs": sorted(registration.SHARED_INVALIDATION_INPUTS),
            }
        ],
    }
    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_candidate_source_promoted_to_registered_guarantee_rejected() -> None:
    record = _candidate_fixture()
    record["record_state"] = "registered-guarantee"
    record["record_scope"] = "actual"
    with pytest.raises(
        registration.RegistrationValidationError, match="schema rejection"
    ):
        registration.validate_record(record)


def test_fixture_scope_cannot_claim_registered_guarantee() -> None:
    record = _registered_record_v3()
    record["record_scope"] = "fixture_only"
    with pytest.raises(registration.RegistrationValidationError, match="record_scope"):
        registration.validate_record(record)


# ---------------------------------------------------------------------------
# v1/v2 non-regression
# ---------------------------------------------------------------------------


def test_v1_candidate_fixture_unchanged_result() -> None:
    report = registration.verify_record_file(V1_FIXTURE)
    assert report == {
        "schema": "h0_gctm_guarantee_registration_v1",
        "record_id": "fixture_h0_gctm_candidate_pair_identity_v1",
        "valid": True,
        "structurally_usable": False,
        "authority_verified": False,
        "disposition": "candidate-source",
    }


def test_v2_candidate_fixture_unchanged_result() -> None:
    report = registration.verify_record_file(V2_FIXTURE)
    assert report == {
        "schema": "h0_gctm_guarantee_registration_v2",
        "record_id": "fixture_h0_gctm_candidate_sources_v2",
        "valid": True,
        "structurally_usable": False,
        "authority_verified": False,
        "disposition": "candidate-source",
    }


def test_v1_v2_class_allowlists_remain_frozen() -> None:
    assert set(registration.V2_CLASS_SPECS) == {
        "identity",
        "snapshot",
        "timing",
        "competition",
        "audit",
    }
    assert "universe_completeness" not in registration.V2_CLASS_SPECS
    assert "runtime_candidate_universe" not in registration.V2_OBJECT_TO_CLASS
    assert "runtime_event_membership" not in registration.V2_OBJECT_TO_CLASS


def test_unsupported_schema_still_rejected() -> None:
    record = _registered_record_v3()
    record["schema"] = "h0_gctm_guarantee_registration_v99"
    with pytest.raises(
        registration.RegistrationValidationError, match="unsupported registration"
    ):
        registration.validate_record(record)


def test_guarantees_do_not_mutually_imply() -> None:
    record = _registered_record_v3()
    consumer = record["consumer"]
    assert isinstance(consumer, dict)
    consumer["required_guarantee_ids"] = consumer["required_guarantee_ids"][:1]
    with pytest.raises(
        registration.RegistrationValidationError, match="required_guarantee_ids"
    ):
        registration.validate_record(record)
