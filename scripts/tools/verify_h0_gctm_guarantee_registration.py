#!/usr/bin/env python3
"""Fail-closed validator for H0-to-GCTM guarantee registration records.

This validator never verifies an H0 capture or owner acceptance. It determines
only whether a record is structurally and semantically sufficient for the
identity-v1 contract, the registration-v2 candidate-row inventory, or the
registration-v3 native-universe completeness contract. A synthetic
candidate-source record is deliberately valid but can never produce
``structurally_usable: true``. V3 never establishes runtime substrate,
compatibility, authority, or H0 re-entry.
"""
# status: stable

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATHS = {
    "h0_gctm_guarantee_registration_v1": ROOT
    / "scripts/tools/h0_gctm_guarantee_registration_schema_v1.json",
    "h0_gctm_guarantee_registration_v2": ROOT
    / "scripts/tools/h0_gctm_guarantee_registration_schema_v2.json",
    "h0_gctm_guarantee_registration_v3": ROOT
    / "scripts/tools/h0_gctm_guarantee_registration_schema_v3.json",
}
CAPTURE_ABI_SCHEMA_PATH = ROOT / "scripts/tools/h0_bridge_decision_trace_schema_v2.json"
CAPTURE_ABI_SCHEMA_ID = "h0_bridge_decision_trace_v2"
SCHEMA_V1 = "h0_gctm_guarantee_registration_v1"
SCHEMA_V2 = "h0_gctm_guarantee_registration_v2"
SCHEMA_V3 = "h0_gctm_guarantee_registration_v3"

PAIR_INSTANCE_KEY = (
    "seq",
    "frame",
    "cand_slot",
    "cand_instance_uid",
    "lost_slot",
    "lost_instance_uid",
)
IDENTITY_OBJECT = "event_runtime_instance_identity"
SHARED_INVALIDATION_INPUTS = frozenset(
    {
        "runtime_instrumentation_identity",
        "policy_base_id",
        "resolved_preset_id",
        "h0_schema_version",
        "capture_mode",
        "event_key_version",
        "observation_state_semantics_version",
        "consumer_domain",
        "causal_availability",
    }
)
IDENTITY_REQUIRED_INVALIDATION_INPUTS = SHARED_INVALIDATION_INPUTS | {
    "identity_lifecycle"
}
DERIVED_RELATION_INVALIDATION_INPUT = "derivation_definition"

# Sealed registrable-field allowlists for the v2 candidate rows (requirements
# doc section 0.2). Every allowlist must stay a subset of the capture-ABI
# record fields; the validator re-checks that containment fail-closed.
SNAPSHOT_PAIR_FIELDS = frozenset(
    {
        "cand_ring_length",
        "lost_ring_length",
        "ema_lost",
        "ema_cand",
        "h_ref",
        "lost_anchor_x",
        "lost_anchor_y",
        "cand_anchor_x",
        "cand_anchor_y",
        "lost_velocity_x",
        "lost_velocity_y",
        "cand_velocity_x",
        "cand_velocity_y",
        "fwd_r",
        "bwd_r",
        "dist_h",
        "s_lost",
        "w",
        "direction_cosine",
        "directional_alpha",
        "height_ratio",
        "speed",
        "spatial_distance",
    }
)
TIMING_PAIR_FIELDS = frozenset({"seq", "frame", "la", "bridge_at"})
COMPETITION_PAIR_FIELDS = frozenset(
    {
        "directional_cross_bdist",
        "bdist_before_direction",
        "bdist_after_direction",
        "height_verdict",
        "speed_verdict",
        "spatial_verdict",
        "cutoff_verdict",
        "occupancy_verdict",
        "occupancy_coverage",
        "appearance_verdict",
        "appearance_cosine",
        "portable_tail_verdict",
        "portable_tail_mask",
        "final_pair_eligible",
        "reject_reason",
    }
)
# candidate_record carries its own record keys because identity-v1 binds only
# the pair_record stream.
COMPETITION_CANDIDATE_FIELDS = frozenset(
    {
        "seq",
        "frame",
        "cand_slot",
        "cand_instance_uid",
        "structural_competitors",
        "pre_score_passes",
        "final_pair_eligible_count",
        "best_lost_slot",
        "second_lost_slot",
        "best_lost_instance_uid",
        "second_lost_instance_uid",
        "best_bdist",
        "second_best_bdist",
        "margin",
        "no_second_competitor",
        "margin_verdict",
        "proposal_emitted",
        "proposal_reject_reason",
        "candidate_status",
    }
)
AUDIT_CLAIM_FIELDS = frozenset(
    {
        "seq",
        "frame",
        "proposing_cand_slot",
        "proposed_lost_slot",
        "proposing_cand_precommit_track_id",
        "proposed_lost_precommit_track_id",
        "proposing_cand_instance_uid",
        "proposed_lost_instance_uid",
        "detection_score",
        "sq",
        "packed_atomic_key",
        "candidate_index_component",
        "winning_cand_slot",
        "winning_cand_precommit_track_id",
        "winning_cand_instance_uid",
        "claim_won",
    }
)
AUDIT_COMMIT_FIELDS = frozenset(
    {
        "seq",
        "frame",
        "cand_slot",
        "lost_slot",
        "cand_precommit_track_id",
        "lost_precommit_track_id",
        "cand_postcommit_track_id",
        "lost_postcommit_track_id",
        "cand_instance_uid",
        "lost_instance_uid",
        "cand_active_before",
        "cand_active_after",
        "lost_active_before",
        "lost_active_after",
        "commit_executed",
        "lost_slot_deactivated",
    }
)

# guarantee_class -> (consumer_object, {stream -> allowed field set}).
V2_CLASS_SPECS: dict[str, tuple[str, dict[str, frozenset[str]]]] = {
    "identity": (IDENTITY_OBJECT, {"pair_record": frozenset(PAIR_INSTANCE_KEY)}),
    "snapshot": ("native_exit_entry_snapshot", {"pair_record": SNAPSHOT_PAIR_FIELDS}),
    "timing": (
        "operational_horizon_observation_point",
        {"pair_record": TIMING_PAIR_FIELDS},
    ),
    "competition": (
        "candidate_competition_pair_score_context",
        {
            "pair_record": COMPETITION_PAIR_FIELDS,
            "candidate_record": COMPETITION_CANDIDATE_FIELDS,
        },
    ),
    "audit": (
        "claim_commit_audit_boundary",
        {"claim_record": AUDIT_CLAIM_FIELDS, "commit_record": AUDIT_COMMIT_FIELDS},
    ),
}
V2_OBJECT_TO_CLASS = {spec[0]: cls for cls, spec in V2_CLASS_SPECS.items()}
# Visible track ids stay banned everywhere except the claim/commit audit
# boundary, where they are declared audit observations, never identities.
TRACK_ID_ALLOWED_CLASSES = frozenset({"audit"})

# ---------------------------------------------------------------------------
# registration-v3: native-universe completeness (additive; frozen v1/v2)
# ---------------------------------------------------------------------------
V3_CONSUMER_UNIVERSE_ID = "gctm_runtime_native_candidate_universe_v1"
V3_EVENT_KEY_VERSION = "gctm_runtime_event_key_v1"
V3_CANDIDATE_KEY_VERSION = "gctm_runtime_candidate_key_v1"
V3_INCLUSION_STAGE_IDENTITY = "pre_score_eligible_v1"
V3_MASK_IDENTITY = "h0_pair_pre_score_mask_v1"
V3_GATE_RETAINED_BAND_IDENTITY = "native_non_score_gates_height_speed_spatial_v1"
V3_COMPLETENESS_PREDICATE_ID = "h0_native_universe_completeness_predicate_v1"
V3_NATIVE_UNIVERSE_SEMANTICS_VERSION = "native_universe_v2"
V3_CONSUMER_OBJECTS = frozenset(
    {"runtime_candidate_universe", "runtime_event_membership"}
)
V3_STREAMS = frozenset({"pair_record", "candidate_record", "event_universe_sidecar"})
V3_REPLAY_BASIS = frozenset({"replay", "shadow_nonperturbation"})
V3_REQUIRED_CAUSAL_AVAILABILITY = "online"

# Registration-level sidecar coordinate: binds frozen trace-v2 envelope fields
# only. Not a data-plane ABI stream; no capture-schema change required.
EVENT_UNIVERSE_SIDECAR_FIELDS = frozenset(
    {
        "native_candidate_keys",
        "native_pair_keys",
        "total_native_candidate_keys",
        "total_native_pair_keys",
        "total_candidate_records",
        "total_pair_records",
        "overflow_native_candidate_keys",
        "overflow_native_pair_keys",
        "overflow_candidate_records",
        "overflow_pair_records",
        "bridge_attempt_count",
        "identity_uid_wrap_events",
        "capture_schema_version",
        "capture_run_uuid",
    }
)
V3_PAIR_COMPLETENESS_FIELDS = frozenset(PAIR_INSTANCE_KEY) | {"reject_reason"}
V3_CANDIDATE_COMPLETENESS_FIELDS = frozenset(
    {
        "seq",
        "frame",
        "cand_slot",
        "cand_instance_uid",
        "pre_score_passes",
    }
)
V3_FORBIDDEN_SOURCE_FIELDS = frozenset(
    {
        "best_lost_slot",
        "second_lost_slot",
        "best_bdist",
        "second_best_bdist",
        "margin",
        "no_second_competitor",
        "margin_verdict",
        "proposal_emitted",
        "proposal_reject_reason",
        "claim_won",
        "commit_executed",
        "winning_cand_slot",
        "winning_cand_precommit_track_id",
        "winning_cand_instance_uid",
        "packed_atomic_key",
        "true_match_label",
        "detection_score",
    }
)
V3_FORBIDDEN_STREAMS = frozenset({"claim_record", "commit_record"})
V3_SCORE_DEPENDENT_STAGES = frozenset(
    {"final_eligible_set", "post_score_eligible_v1", "score_dependent"}
)
V3_REQUIRED_INVALIDATION_INPUTS = frozenset(
    {
        "runtime_instrumentation_identity",
        "policy_base_id",
        "resolved_preset_id",
        "h0_schema_version",
        "capture_mode",
        "capture_run_uuid",
        "consumer_universe_id",
        "event_key_version",
        "candidate_key_version",
        "inclusion_stage_identity",
        "mask_identity",
        "gate_retained_band_identity",
        "native_universe_semantics_version",
        "identity_uid_lifecycle",
        "consumer_domain",
        "causal_availability",
        "completeness_predicate_id",
        "replay_non_perturbation_basis",
    }
)
V3_IDENTITY_BINDINGS = {
    "consumer_universe_id": V3_CONSUMER_UNIVERSE_ID,
    "event_key_version": V3_EVENT_KEY_VERSION,
    "candidate_key_version": V3_CANDIDATE_KEY_VERSION,
    "inclusion_stage_identity": V3_INCLUSION_STAGE_IDENTITY,
    "mask_identity": V3_MASK_IDENTITY,
    "gate_retained_band_identity": V3_GATE_RETAINED_BAND_IDENTITY,
    "completeness_predicate_id": V3_COMPLETENESS_PREDICATE_ID,
    "native_universe_semantics_version": V3_NATIVE_UNIVERSE_SEMANTICS_VERSION,
}
V3_FIXED_NON_AUTHORITY = {
    "authority_verified": False,
    "runtime_substrate_established": False,
    "runtime_compatibility_established": False,
    "h0_reentry_authorized": False,
}
PAIR_KEY_FIELDS = (
    "seq",
    "frame",
    "cand_slot",
    "cand_instance_uid",
    "lost_slot",
    "lost_instance_uid",
)
CANDIDATE_KEY_FIELDS = ("seq", "frame", "cand_slot", "cand_instance_uid")
EVENT_KEY_FIELDS = (
    "seq",
    "frame",
    "lost_slot",
    "lost_instance_uid",
    "event_key_version",
)

# Ordered terminals for the registration-v3 contract seal itself.
TERMINAL_V3_AUDIT_INVALID = "H0_REGISTRATION_V3_AUDIT_INVALID"
TERMINAL_V3_REQUIRES_ABI_DELTA = "H0_REGISTRATION_V3_REQUIRES_ABI_DELTA"
TERMINAL_V3_CONTRACT_SEALABLE = "H0_REGISTRATION_V3_CONTRACT_SEALABLE"
TERMINAL_V3_ORDER = (
    TERMINAL_V3_AUDIT_INVALID,
    TERMINAL_V3_REQUIRES_ABI_DELTA,
    TERMINAL_V3_CONTRACT_SEALABLE,
)


class RegistrationValidationError(ValueError):
    """The record cannot satisfy the applicable registration contract."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistrationValidationError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistrationValidationError(f"malformed JSON {path}: {exc}") from exc


def _record_schema_id(record: object) -> str:
    record_map = _as_mapping(record, "record")
    schema_id = record_map.get("schema")
    if schema_id not in SCHEMA_PATHS:
        raise RegistrationValidationError(
            f"unsupported registration schema {schema_id!r}"
        )
    return schema_id


def _schema_document(schema_id: str) -> Mapping[str, Any]:
    value = load_json(SCHEMA_PATHS[schema_id])
    if not isinstance(value, Mapping):  # pragma: no cover - static repository file
        raise RegistrationValidationError("registration schema must be a JSON object")
    return value


def _capture_abi_document() -> Mapping[str, Any]:
    document_map = _as_mapping(load_json(CAPTURE_ABI_SCHEMA_PATH), "capture ABI schema")
    declared = document_map.get("capture_schema_version")
    if declared != CAPTURE_ABI_SCHEMA_ID:
        raise RegistrationValidationError(
            f"capture ABI document declares {declared!r}, "
            f"expected {CAPTURE_ABI_SCHEMA_ID!r}"
        )
    return document_map


def _capture_abi_fields(stream: str) -> frozenset[str]:
    document_map = _capture_abi_document()
    record_fields = _as_mapping(document_map["record_fields"], "record_fields")
    key = f"{stream}s"
    if key not in record_fields:
        raise RegistrationValidationError(f"capture ABI has no stream {stream!r}")
    return frozenset(_as_string_sequence(record_fields[key], key))


def _allowlist(guarantee_class: str, stream: str) -> frozenset[str]:
    _, stream_fields = V2_CLASS_SPECS[guarantee_class]
    if stream not in stream_fields:
        raise RegistrationValidationError(
            f"{guarantee_class} may not bind stream {stream!r}"
        )
    allowed = stream_fields[stream]
    abi_fields = _capture_abi_fields(stream)
    if not allowed <= abi_fields:
        drifted = sorted(allowed - abi_fields)
        raise RegistrationValidationError(
            f"sealed {guarantee_class}/{stream} allowlist drifted from the "
            f"capture ABI (unknown fields: {drifted})"
        )
    return allowed


def _schema_validate(record: object, schema_id: str) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RegistrationValidationError("jsonschema dependency unavailable") from exc

    schema = _schema_document(schema_id)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = sorted(
            jsonschema.Draft202012Validator(
                schema, format_checker=jsonschema.FormatChecker()
            ).iter_errors(record),
            key=lambda error: list(error.absolute_path),
        )
    except jsonschema.SchemaError as exc:  # pragma: no cover - static schema
        raise RegistrationValidationError(
            f"invalid registration schema: {exc.message}"
        ) from exc
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise RegistrationValidationError(
            f"schema rejection at {location}: {error.message}"
        )


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistrationValidationError(f"{name} must be an object")
    return value


def _as_string_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RegistrationValidationError(f"{name} must be an array of strings")
    if not all(isinstance(item, str) for item in value):
        raise RegistrationValidationError(f"{name} must contain only strings")
    return tuple(value)


def _validate_candidate_identity_source(record: Mapping[str, Any]) -> None:
    sources = record["candidate_sources"]
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise RegistrationValidationError("candidate_sources must be an array")

    if len(sources) != 1:
        raise RegistrationValidationError(
            "identity-v1 candidate registration requires exactly one source"
        )
    source_map = _as_mapping(sources[0], "candidate_sources entry")
    if source_map["consumer_object"] != IDENTITY_OBJECT:
        raise RegistrationValidationError(
            "identity-v1 supports only event_runtime_instance_identity"
        )
    if source_map["stream"] != "pair_record":
        raise RegistrationValidationError(
            "event_runtime_instance_identity must use pair_record"
        )
    if (
        _as_string_sequence(source_map["key_fields"], "candidate key_fields")
        != PAIR_INSTANCE_KEY
    ):
        raise RegistrationValidationError(
            "candidate identity source must use the sealed pair instance key"
        )


def _validate_registered_identity_guarantee(record: Mapping[str, Any]) -> None:
    baseline = _as_mapping(record["baseline"], "baseline")
    if baseline["accepted_by"] != "h0_owner":  # Schema repeats this intentionally.
        raise RegistrationValidationError("only h0_owner may accept a baseline")
    expected_domain = {
        "preset_id": baseline["resolved_preset_id"],
        "runtime_identity": baseline["runtime_instrumentation_identity"],
        "schema_id": baseline["h0_schema_version"],
        "dataset_sequence_domain": baseline["dataset_sequence_domain"],
    }

    guarantees = record["guarantees"]
    if not isinstance(guarantees, Sequence) or isinstance(guarantees, (str, bytes)):
        raise RegistrationValidationError("guarantees must be an array")
    if len(guarantees) != 1:
        raise RegistrationValidationError(
            "identity-v1 registered record requires exactly one guarantee"
        )
    guarantee_map = _as_mapping(guarantees[0], "guarantee entry")
    declared_domain = _as_mapping(guarantee_map["declared_domain"], "declared_domain")
    if dict(declared_domain) != expected_domain:
        raise RegistrationValidationError(
            "guarantee declared_domain must exactly bind the accepted baseline"
        )
    if guarantee_map["consumer_object"] != IDENTITY_OBJECT:
        raise RegistrationValidationError(
            "identity-v1 supports only event_runtime_instance_identity"
        )
    if guarantee_map["guarantee_class"] != "identity":
        raise RegistrationValidationError(
            "event_runtime_instance_identity requires identity guarantee_class"
        )
    if guarantee_map["relation"] != "exact":
        raise RegistrationValidationError(
            "event_runtime_instance_identity may be structurally usable only as exact"
        )
    if guarantee_map["stream"] != "pair_record":
        raise RegistrationValidationError(
            "event_runtime_instance_identity must bind pair_record"
        )
    fields = _as_string_sequence(guarantee_map["covered_fields"], "covered_fields")
    if fields != PAIR_INSTANCE_KEY:
        raise RegistrationValidationError(
            "identity guarantee must cover exactly the sealed pair instance key"
        )
    if any("track_id" in field for field in fields):
        raise RegistrationValidationError(
            "visible track_id may not be used as a runtime-instance identity"
        )
    invalidation_inputs = frozenset(
        _as_string_sequence(guarantee_map["invalidation_inputs"], "invalidation_inputs")
    )
    if invalidation_inputs != IDENTITY_REQUIRED_INVALIDATION_INPUTS:
        missing = sorted(IDENTITY_REQUIRED_INVALIDATION_INPUTS - invalidation_inputs)
        unexpected = sorted(invalidation_inputs - IDENTITY_REQUIRED_INVALIDATION_INPUTS)
        raise RegistrationValidationError(
            "identity invalidation_inputs must equal the sealed identity-v1 set "
            f"(missing={missing}, unexpected={unexpected})"
        )

    consumer = _as_mapping(record["consumer"], "consumer")
    required_ids = _as_string_sequence(
        consumer["required_guarantee_ids"], "consumer.required_guarantee_ids"
    )
    if required_ids != (guarantee_map["guarantee_id"],):
        raise RegistrationValidationError(
            "consumer.required_guarantee_ids must bind the sole identity guarantee"
        )


def _check_track_id_ban(fields: Sequence[str], guarantee_class: str) -> None:
    if guarantee_class in TRACK_ID_ALLOWED_CLASSES:
        return
    if any("track_id" in field for field in fields):
        raise RegistrationValidationError(
            "visible track_id fields are registrable only at the "
            "claim/commit audit boundary"
        )


def _validate_candidate_sources_v2(record: Mapping[str, Any]) -> None:
    sources = record["candidate_sources"]
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise RegistrationValidationError("candidate_sources must be an array")

    seen: set[tuple[str, str]] = set()
    for index, source in enumerate(sources):
        source_map = _as_mapping(source, f"candidate_sources[{index}]")
        consumer_object = source_map["consumer_object"]
        stream = source_map["stream"]
        guarantee_class = V2_OBJECT_TO_CLASS[consumer_object]
        coordinate = (consumer_object, stream)
        if coordinate in seen:
            raise RegistrationValidationError(
                f"duplicate candidate source for {consumer_object}/{stream}"
            )
        seen.add(coordinate)
        fields = _as_string_sequence(
            source_map["source_fields"], f"candidate_sources[{index}].source_fields"
        )
        _check_track_id_ban(fields, guarantee_class)
        if guarantee_class == "identity":
            if fields != PAIR_INSTANCE_KEY:
                raise RegistrationValidationError(
                    "candidate identity source must use the sealed pair instance key"
                )
            continue
        allowed = _allowlist(guarantee_class, stream)
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise RegistrationValidationError(
                f"candidate {guarantee_class}/{stream} source names fields outside "
                f"the sealed allowlist: {unknown}"
            )


def _required_invalidation_inputs(
    guarantee_class: str, relation: str
) -> frozenset[str]:
    required = (
        IDENTITY_REQUIRED_INVALIDATION_INPUTS
        if guarantee_class == "identity"
        else SHARED_INVALIDATION_INPUTS
    )
    if relation == "derived":
        required = required | {DERIVED_RELATION_INVALIDATION_INPUT}
    return required


def _validate_registered_guarantees_v2(record: Mapping[str, Any]) -> None:
    baseline = _as_mapping(record["baseline"], "baseline")
    if baseline["accepted_by"] != "h0_owner":  # Schema repeats this intentionally.
        raise RegistrationValidationError("only h0_owner may accept a baseline")
    _capture_abi_document()
    if baseline["h0_schema_version"] != CAPTURE_ABI_SCHEMA_ID:
        raise RegistrationValidationError(
            "baseline h0_schema_version must match the capture ABI this validator "
            f"anchors covered fields to ({CAPTURE_ABI_SCHEMA_ID!r}); a different "
            "schema requires its own registration contract"
        )
    expected_domain = {
        "preset_id": baseline["resolved_preset_id"],
        "runtime_identity": baseline["runtime_instrumentation_identity"],
        "schema_id": baseline["h0_schema_version"],
        "dataset_sequence_domain": baseline["dataset_sequence_domain"],
    }

    guarantees = record["guarantees"]
    if not isinstance(guarantees, Sequence) or isinstance(guarantees, (str, bytes)):
        raise RegistrationValidationError("guarantees must be an array")

    guarantee_ids: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, guarantee in enumerate(guarantees):
        guarantee_map = _as_mapping(guarantee, f"guarantees[{index}]")
        guarantee_class = guarantee_map["guarantee_class"]
        stream = guarantee_map["stream"]
        expected_object, _ = V2_CLASS_SPECS[guarantee_class]
        if guarantee_map["consumer_object"] != expected_object:
            raise RegistrationValidationError(
                f"{guarantee_class} guarantee must bind {expected_object}"
            )
        coordinate = (guarantee_class, stream)
        if coordinate in seen:
            raise RegistrationValidationError(
                f"duplicate guarantee for {guarantee_class}/{stream}"
            )
        seen.add(coordinate)

        declared_domain = _as_mapping(
            guarantee_map["declared_domain"], "declared_domain"
        )
        if dict(declared_domain) != expected_domain:
            raise RegistrationValidationError(
                "guarantee declared_domain must exactly bind the accepted baseline"
            )

        fields = _as_string_sequence(guarantee_map["covered_fields"], "covered_fields")
        _check_track_id_ban(fields, guarantee_class)
        if guarantee_class == "identity":
            if fields != PAIR_INSTANCE_KEY:
                raise RegistrationValidationError(
                    "identity guarantee must cover exactly the sealed pair instance key"
                )
        else:
            allowed = _allowlist(guarantee_class, stream)
            unknown = sorted(set(fields) - allowed)
            if unknown:
                raise RegistrationValidationError(
                    f"{guarantee_class}/{stream} guarantee covers fields outside "
                    f"the sealed allowlist: {unknown}"
                )

        relation = guarantee_map["relation"]
        # Schema couples relation and derivation; repeated here intentionally.
        if relation == "derived" and "derivation" not in guarantee_map:
            raise RegistrationValidationError(
                "derived relation requires an immutable derivation binding"
            )
        invalidation_inputs = frozenset(
            _as_string_sequence(
                guarantee_map["invalidation_inputs"], "invalidation_inputs"
            )
        )
        required = _required_invalidation_inputs(guarantee_class, relation)
        if invalidation_inputs != required:
            missing = sorted(required - invalidation_inputs)
            unexpected = sorted(invalidation_inputs - required)
            raise RegistrationValidationError(
                f"{guarantee_class} invalidation_inputs must equal the sealed set "
                f"(missing={missing}, unexpected={unexpected})"
            )

        guarantee_id = guarantee_map["guarantee_id"]
        if guarantee_id in guarantee_ids:
            raise RegistrationValidationError(
                f"duplicate guarantee_id {guarantee_id!r}"
            )
        guarantee_ids.append(guarantee_id)

    consumer = _as_mapping(record["consumer"], "consumer")
    required_ids = _as_string_sequence(
        consumer["required_guarantee_ids"], "consumer.required_guarantee_ids"
    )
    if sorted(required_ids) != sorted(guarantee_ids):
        raise RegistrationValidationError(
            "consumer.required_guarantee_ids must bind exactly the registered "
            "guarantees"
        )


def _tuple_from_key(
    value: Mapping[str, Any], fields: Sequence[str], name: str
) -> tuple:
    missing = [field for field in fields if field not in value]
    if missing:
        raise RegistrationValidationError(f"{name} missing fields {missing}")
    return tuple(value[field] for field in fields)


def _capture_abi_envelope_fields() -> frozenset[str]:
    document_map = _capture_abi_document()
    return frozenset(
        _as_string_sequence(document_map["envelope_fields"], "envelope_fields")
    )


def _v3_stream_allowlist(stream: str) -> frozenset[str]:
    if stream == "event_universe_sidecar":
        envelope = _capture_abi_envelope_fields()
        if not EVENT_UNIVERSE_SIDECAR_FIELDS <= envelope:
            drifted = sorted(EVENT_UNIVERSE_SIDECAR_FIELDS - envelope)
            raise RegistrationValidationError(
                "event_universe_sidecar allowlist drifted from the frozen "
                f"capture ABI envelope (unknown fields: {drifted})"
            )
        return EVENT_UNIVERSE_SIDECAR_FIELDS
    if stream == "pair_record":
        allowed = V3_PAIR_COMPLETENESS_FIELDS
        abi = _capture_abi_fields("pair_record")
        if not allowed <= abi:
            drifted = sorted(allowed - abi)
            raise RegistrationValidationError(
                f"v3 pair_record allowlist drifted from capture ABI: {drifted}"
            )
        return allowed
    if stream == "candidate_record":
        allowed = V3_CANDIDATE_COMPLETENESS_FIELDS
        abi = _capture_abi_fields("candidate_record")
        if not allowed <= abi:
            drifted = sorted(allowed - abi)
            raise RegistrationValidationError(
                f"v3 candidate_record allowlist drifted from capture ABI: {drifted}"
            )
        return allowed
    raise RegistrationValidationError(f"v3 does not admit stream {stream!r}")


def _validate_v3_identity_bindings(record: Mapping[str, Any]) -> None:
    bindings = _as_mapping(record["identity_bindings"], "identity_bindings")
    for key, expected in V3_IDENTITY_BINDINGS.items():
        if bindings.get(key) != expected:
            raise RegistrationValidationError(
                f"identity binding {key} must be {expected!r}, "
                f"got {bindings.get(key)!r}"
            )
    consumer = _as_mapping(record["consumer"], "consumer")
    if consumer.get("consumer_universe_id") != V3_CONSUMER_UNIVERSE_ID:
        raise RegistrationValidationError(
            f"consumer.consumer_universe_id must exact-bind {V3_CONSUMER_UNIVERSE_ID!r}"
        )


def _validate_v3_source_fields(
    stream: str, fields: Sequence[str], context: str
) -> None:
    if stream in V3_FORBIDDEN_STREAMS:
        raise RegistrationValidationError(
            f"{context}: claim/commit streams may not form universe membership"
        )
    forbidden = sorted(set(fields) & V3_FORBIDDEN_SOURCE_FIELDS)
    if forbidden:
        raise RegistrationValidationError(
            f"{context}: claim/score/label-derived fields forbidden for "
            f"universe completeness: {forbidden}"
        )
    if any("track_id" in field for field in fields):
        raise RegistrationValidationError(
            f"{context}: visible track_id fields may not form universe membership"
        )
    allowed = _v3_stream_allowlist(stream)
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise RegistrationValidationError(
            f"{context}: fields outside the sealed v3 {stream} allowlist: {unknown}"
        )


def _validate_candidate_sources_v3(record: Mapping[str, Any]) -> None:
    _validate_v3_identity_bindings(record)
    sources = record["candidate_sources"]
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise RegistrationValidationError("candidate_sources must be an array")

    seen: set[tuple[str, str]] = set()
    streams_seen: set[str] = set()
    objects_seen: set[str] = set()
    for index, source in enumerate(sources):
        source_map = _as_mapping(source, f"candidate_sources[{index}]")
        consumer_object = source_map["consumer_object"]
        stream = source_map["stream"]
        if consumer_object not in V3_CONSUMER_OBJECTS:
            raise RegistrationValidationError(
                f"v3 consumer_object must be one of {sorted(V3_CONSUMER_OBJECTS)}"
            )
        if stream not in V3_STREAMS:
            raise RegistrationValidationError(
                f"v3 stream must be one of {sorted(V3_STREAMS)}"
            )
        coordinate = (consumer_object, stream)
        if coordinate in seen:
            raise RegistrationValidationError(
                f"duplicate candidate source for {consumer_object}/{stream}"
            )
        seen.add(coordinate)
        streams_seen.add(stream)
        objects_seen.add(consumer_object)
        fields = _as_string_sequence(
            source_map["source_fields"], f"candidate_sources[{index}].source_fields"
        )
        _validate_v3_source_fields(
            stream, fields, f"candidate_sources[{index}] ({consumer_object}/{stream})"
        )
        if stream == "event_universe_sidecar":
            # Sidecar inventory must expose every mandatory completeness field.
            missing = sorted(EVENT_UNIVERSE_SIDECAR_FIELDS - set(fields))
            if missing:
                raise RegistrationValidationError(
                    "event_universe_sidecar candidate source must bind every "
                    f"mandatory completeness field (missing={missing})"
                )

    missing_streams = sorted(V3_STREAMS - streams_seen)
    if missing_streams:
        raise RegistrationValidationError(
            "v3 candidate-source inventory must cover pair_record, "
            f"candidate_record, and event_universe_sidecar (missing={missing_streams})"
        )
    # At least one of the two completeness consumer objects must appear.
    if not objects_seen & V3_CONSUMER_OBJECTS:
        raise RegistrationValidationError(
            "v3 candidate-source inventory must bind at least one completeness "
            "consumer object"
        )


def _validate_completeness_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_capture_run_uuid: str,
    context: str,
) -> None:
    if evidence.get("capture_run_uuid") != expected_capture_run_uuid:
        raise RegistrationValidationError(
            f"{context}: completeness evidence capture_run_uuid must bind the "
            "declared capture_run_uuid"
        )
    if evidence.get("capture_schema_version") != CAPTURE_ABI_SCHEMA_ID:
        raise RegistrationValidationError(
            f"{context}: completeness evidence must bind {CAPTURE_ABI_SCHEMA_ID!r}"
        )
    if evidence.get("inclusion_stage_score_independent") is not True:
        raise RegistrationValidationError(
            f"{context}: inclusion stage must be score-independent"
        )
    if evidence.get("silent_truncation_possible") is not False:
        raise RegistrationValidationError(
            f"{context}: silent truncation must be impossible under the "
            "declared completeness basis"
        )
    if evidence.get("claim_commit_label_participation") is not False:
        raise RegistrationValidationError(
            f"{context}: claim/commit/label fields must not participate in "
            "universe or membership formation"
        )

    native_pairs = [
        _tuple_from_key(_as_mapping(item, "native_pair_key"), PAIR_KEY_FIELDS, "pair")
        for item in evidence["native_pair_keys"]
    ]
    pair_rows = [
        _tuple_from_key(_as_mapping(item, "pair_record_key"), PAIR_KEY_FIELDS, "pair")
        for item in evidence["pair_record_keys"]
    ]
    native_pair_counter = Counter(native_pairs)
    pair_row_counter = Counter(pair_rows)
    if native_pair_counter != pair_row_counter:
        only_native = sorted(native_pair_counter - pair_row_counter)
        only_rows = sorted(pair_row_counter - native_pair_counter)
        raise RegistrationValidationError(
            f"{context}: every native_pair_key must map to exactly one "
            f"pair_record (native_without_row={only_native}, "
            f"row_without_native={only_rows})"
        )
    if any(count != 1 for count in native_pair_counter.values()):
        raise RegistrationValidationError(
            f"{context}: duplicate native_pair_key entries are rejected"
        )

    native_candidates = [
        _tuple_from_key(
            _as_mapping(item, "native_candidate_key"),
            CANDIDATE_KEY_FIELDS,
            "candidate",
        )
        for item in evidence["native_candidate_keys"]
    ]
    candidate_rows = [
        _tuple_from_key(
            _as_mapping(item, "candidate_record_key"),
            CANDIDATE_KEY_FIELDS,
            "candidate",
        )
        for item in evidence["candidate_record_keys"]
    ]
    native_cand_counter = Counter(native_candidates)
    cand_row_counter = Counter(candidate_rows)
    if native_cand_counter != cand_row_counter:
        only_native = sorted(native_cand_counter - cand_row_counter)
        only_rows = sorted(cand_row_counter - native_cand_counter)
        raise RegistrationValidationError(
            f"{context}: every native_candidate_key must map to exactly one "
            f"candidate_record (native_without_row={only_native}, "
            f"row_without_native={only_rows})"
        )
    if any(count != 1 for count in native_cand_counter.values()):
        raise RegistrationValidationError(
            f"{context}: duplicate candidate rows are rejected"
        )

    membership = evidence["retained_candidate_membership"]
    if not isinstance(membership, Sequence) or isinstance(membership, (str, bytes)):
        raise RegistrationValidationError(
            f"{context}: retained_candidate_membership must be an array"
        )
    membership_by_candidate: dict[tuple, tuple] = {}
    for index, entry in enumerate(membership):
        entry_map = _as_mapping(entry, f"retained_candidate_membership[{index}]")
        cand = _tuple_from_key(
            _as_mapping(entry_map["candidate_key"], "candidate_key"),
            CANDIDATE_KEY_FIELDS,
            "candidate_key",
        )
        event = _tuple_from_key(
            _as_mapping(entry_map["event_key"], "event_key"),
            EVENT_KEY_FIELDS,
            "event_key",
        )
        if event[-1] != V3_EVENT_KEY_VERSION:
            raise RegistrationValidationError(
                f"{context}: event_key_version must be {V3_EVENT_KEY_VERSION!r}"
            )
        # Candidate key (seq,frame) must match event (seq,frame).
        if cand[0] != event[0] or cand[1] != event[1]:
            raise RegistrationValidationError(
                f"{context}: retained candidate must share seq/frame with its event"
            )
        if cand in membership_by_candidate:
            if membership_by_candidate[cand] != event:
                raise RegistrationValidationError(
                    f"{context}: cross-event candidate split rejected for {cand!r}"
                )
            raise RegistrationValidationError(
                f"{context}: duplicate candidate membership for {cand!r}"
            )
        membership_by_candidate[cand] = event

    retained = set(membership_by_candidate)
    if retained != set(native_cand_counter):
        missing = sorted(set(native_cand_counter) - retained)
        extra = sorted(retained - set(native_cand_counter))
        raise RegistrationValidationError(
            f"{context}: every retained candidate key must belong to exactly one "
            f"event and reconcile with native keys (missing={missing}, extra={extra})"
        )

    totals = _as_mapping(evidence["totals"], "totals")
    expected_totals = {
        "total_native_pair_keys": len(native_pairs),
        "total_pair_records": len(pair_rows),
        "total_native_candidate_keys": len(native_candidates),
        "total_candidate_records": len(candidate_rows),
        "overflow_native_pair_keys": 0,
        "overflow_pair_records": 0,
        "overflow_native_candidate_keys": 0,
        "overflow_candidate_records": 0,
        "bridge_attempt_count": len(native_candidates),
        "identity_uid_wrap_events": 0,
    }
    for key, expected in expected_totals.items():
        if totals.get(key) != expected:
            raise RegistrationValidationError(
                f"{context}: totals.{key} must equal {expected}, got {totals.get(key)!r}"
            )

    partial_events = evidence["partial_events"]
    if not isinstance(partial_events, Sequence) or isinstance(
        partial_events, (str, bytes)
    ):
        raise RegistrationValidationError(f"{context}: partial_events must be an array")
    if len(partial_events) != 0:
        raise RegistrationValidationError(
            f"{context}: partial events fail closed; partial_events must be empty"
        )

    exposure = _as_mapping(evidence["exposure_predicates"], "exposure_predicates")
    if exposure.get("require_candidate_exposure") is not True:
        raise RegistrationValidationError(
            f"{context}: capture exposure predicates require candidate exposure"
        )
    if exposure.get("bridge_attempt_equals_native_candidate_keys") is not True:
        raise RegistrationValidationError(
            f"{context}: bridge_attempt_count must equal native candidate key count"
        )
    if totals["bridge_attempt_count"] != len(native_candidates):
        raise RegistrationValidationError(
            f"{context}: bridge_attempt_count does not reconcile with "
            "native_candidate_keys"
        )

    conservation = _as_mapping(
        evidence["m0_m1_m2_identical_universe"], "m0_m1_m2_identical_universe"
    )
    if conservation.get("same_event_set") is not True:
        raise RegistrationValidationError(
            f"{context}: M0/M1/M2 must consume identical event sets"
        )
    if conservation.get("same_C_e") is not True:
        raise RegistrationValidationError(
            f"{context}: M0/M1/M2 must consume identical C_e"
        )
    if not conservation.get("event_set_fingerprint"):
        raise RegistrationValidationError(
            f"{context}: event_set_fingerprint is required for M0/M1/M2 conservation"
        )
    if not conservation.get("c_e_fingerprint"):
        raise RegistrationValidationError(
            f"{context}: c_e_fingerprint is required for M0/M1/M2 conservation"
        )


def _validate_v3_source_bindings(
    bindings: object, context: str
) -> dict[str, tuple[str, ...]]:
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        raise RegistrationValidationError(
            f"{context}: source_bindings must be an array"
        )
    if len(bindings) != 3:
        raise RegistrationValidationError(
            f"{context}: source_bindings must include exactly pair_record, "
            "candidate_record, and event_universe_sidecar"
        )
    by_stream: dict[str, tuple[str, ...]] = {}
    for index, binding in enumerate(bindings):
        binding_map = _as_mapping(binding, f"{context}.source_bindings[{index}]")
        stream = binding_map["stream"]
        if stream in by_stream:
            raise RegistrationValidationError(
                f"{context}: duplicate source binding for stream {stream!r}"
            )
        fields = _as_string_sequence(
            binding_map["source_fields"], f"{context}.source_bindings[{index}]"
        )
        _validate_v3_source_fields(stream, fields, f"{context}/{stream}")
        if stream == "event_universe_sidecar":
            missing = sorted(EVENT_UNIVERSE_SIDECAR_FIELDS - set(fields))
            if missing:
                raise RegistrationValidationError(
                    f"{context}: event_universe_sidecar binding missing {missing}"
                )
        by_stream[stream] = fields
    missing_streams = sorted(V3_STREAMS - set(by_stream))
    if missing_streams:
        raise RegistrationValidationError(
            f"{context}: missing source bindings for {missing_streams}"
        )
    return by_stream


def _validate_registered_guarantees_v3(record: Mapping[str, Any]) -> None:
    _validate_v3_identity_bindings(record)
    baseline = _as_mapping(record["baseline"], "baseline")
    if baseline["accepted_by"] != "h0_owner":
        raise RegistrationValidationError("only h0_owner may accept a baseline")
    if baseline["h0_schema_version"] != CAPTURE_ABI_SCHEMA_ID:
        raise RegistrationValidationError(
            "baseline h0_schema_version must match the capture ABI "
            f"({CAPTURE_ABI_SCHEMA_ID!r})"
        )
    if baseline["event_key_version"] != V3_EVENT_KEY_VERSION:
        raise RegistrationValidationError(
            f"baseline event_key_version must be {V3_EVENT_KEY_VERSION!r}"
        )

    expected_domain = {
        "preset_id": baseline["resolved_preset_id"],
        "runtime_identity": baseline["runtime_instrumentation_identity"],
        "schema_id": baseline["h0_schema_version"],
        "dataset_sequence_domain": baseline["dataset_sequence_domain"],
        "consumer_universe_id": V3_CONSUMER_UNIVERSE_ID,
    }

    guarantees = record["guarantees"]
    if not isinstance(guarantees, Sequence) or isinstance(guarantees, (str, bytes)):
        raise RegistrationValidationError("guarantees must be an array")

    guarantee_ids: list[str] = []
    seen_objects: set[str] = set()
    for index, guarantee in enumerate(guarantees):
        guarantee_map = _as_mapping(guarantee, f"guarantees[{index}]")
        context = f"guarantees[{index}]"
        if guarantee_map["guarantee_class"] != "universe_completeness":
            raise RegistrationValidationError(
                f"{context}: v3 admits only guarantee_class=universe_completeness"
            )
        consumer_object = guarantee_map["consumer_object"]
        if consumer_object not in V3_CONSUMER_OBJECTS:
            raise RegistrationValidationError(
                f"{context}: consumer_object must be one of "
                f"{sorted(V3_CONSUMER_OBJECTS)}"
            )
        if consumer_object in seen_objects:
            raise RegistrationValidationError(
                f"{context}: duplicate guarantee for consumer_object "
                f"{consumer_object!r}"
            )
        seen_objects.add(consumer_object)

        if guarantee_map["consumer_universe_id"] != V3_CONSUMER_UNIVERSE_ID:
            raise RegistrationValidationError(
                f"{context}: consumer_universe_id must be {V3_CONSUMER_UNIVERSE_ID!r}"
            )
        if guarantee_map["baseline_id"] != baseline["baseline_id"]:
            raise RegistrationValidationError(
                f"{context}: baseline_id must bind the accepted baseline"
            )
        if (
            guarantee_map["runtime_instrumentation_identity"]
            != baseline["runtime_instrumentation_identity"]
        ):
            raise RegistrationValidationError(
                f"{context}: runtime_instrumentation_identity must bind baseline"
            )
        if guarantee_map["resolved_preset_id"] != baseline["resolved_preset_id"]:
            raise RegistrationValidationError(
                f"{context}: resolved_preset_id must bind baseline"
            )
        if guarantee_map["h0_schema_version"] != baseline["h0_schema_version"]:
            raise RegistrationValidationError(
                f"{context}: h0_schema_version must bind baseline"
            )
        if guarantee_map["capture_run_uuid"] != baseline["capture_run_uuid"]:
            raise RegistrationValidationError(
                f"{context}: capture_run_uuid must bind baseline"
            )
        if (
            guarantee_map["dataset_sequence_domain"]
            != baseline["dataset_sequence_domain"]
        ):
            raise RegistrationValidationError(
                f"{context}: dataset_sequence_domain must bind baseline"
            )
        for field, expected in (
            ("event_key_version", V3_EVENT_KEY_VERSION),
            ("candidate_key_version", V3_CANDIDATE_KEY_VERSION),
            ("inclusion_stage_identity", V3_INCLUSION_STAGE_IDENTITY),
            ("mask_identity", V3_MASK_IDENTITY),
            ("gate_retained_band_identity", V3_GATE_RETAINED_BAND_IDENTITY),
            ("completeness_predicate_id", V3_COMPLETENESS_PREDICATE_ID),
        ):
            if guarantee_map[field] != expected:
                raise RegistrationValidationError(
                    f"{context}: {field} must be {expected!r}"
                )
        if guarantee_map["inclusion_stage_identity"] in V3_SCORE_DEPENDENT_STAGES:
            raise RegistrationValidationError(
                f"{context}: post-score or score-dependent inclusion stage rejected"
            )
        if guarantee_map["causal_availability"] != V3_REQUIRED_CAUSAL_AVAILABILITY:
            raise RegistrationValidationError(
                f"{context}: causal_availability must be "
                f"{V3_REQUIRED_CAUSAL_AVAILABILITY!r} for pre-score membership"
            )
        basis = frozenset(
            _as_string_sequence(
                guarantee_map["replay_non_perturbation_basis"],
                f"{context}.replay_non_perturbation_basis",
            )
        )
        if basis != V3_REPLAY_BASIS:
            raise RegistrationValidationError(
                f"{context}: replay_non_perturbation_basis must equal "
                f"{sorted(V3_REPLAY_BASIS)}"
            )

        declared_domain = _as_mapping(
            guarantee_map["declared_domain"], f"{context}.declared_domain"
        )
        if dict(declared_domain) != expected_domain:
            raise RegistrationValidationError(
                f"{context}: declared_domain must exactly bind the accepted baseline "
                "and consumer universe"
            )

        invalidation_inputs = frozenset(
            _as_string_sequence(
                guarantee_map["invalidation_inputs"], f"{context}.invalidation_inputs"
            )
        )
        if invalidation_inputs != V3_REQUIRED_INVALIDATION_INPUTS:
            missing = sorted(V3_REQUIRED_INVALIDATION_INPUTS - invalidation_inputs)
            unexpected = sorted(invalidation_inputs - V3_REQUIRED_INVALIDATION_INPUTS)
            raise RegistrationValidationError(
                f"{context}: invalidation_inputs must equal the sealed v3 set "
                f"(missing={missing}, unexpected={unexpected})"
            )

        _validate_v3_source_bindings(guarantee_map["source_bindings"], context)
        _validate_completeness_evidence(
            _as_mapping(
                guarantee_map["completeness_evidence"],
                f"{context}.completeness_evidence",
            ),
            expected_capture_run_uuid=baseline["capture_run_uuid"],
            context=context,
        )

        guarantee_id = guarantee_map["guarantee_id"]
        if guarantee_id in guarantee_ids:
            raise RegistrationValidationError(
                f"duplicate guarantee_id {guarantee_id!r}"
            )
        guarantee_ids.append(guarantee_id)

    # The two consumer objects may share baseline/sidecar evidence but never
    # imply each other: required_guarantee_ids must bind exactly the registered set.
    consumer = _as_mapping(record["consumer"], "consumer")
    required_ids = _as_string_sequence(
        consumer["required_guarantee_ids"], "consumer.required_guarantee_ids"
    )
    if sorted(required_ids) != sorted(guarantee_ids):
        raise RegistrationValidationError(
            "consumer.required_guarantee_ids must bind exactly the registered "
            "guarantees (no mutual implication between universe and membership)"
        )


def mandatory_completeness_fields_available_in_trace_v2() -> dict[str, object]:
    """Return whether every v3 completeness binding exists on frozen trace-v2."""
    envelope = _capture_abi_envelope_fields()
    pair_fields = _capture_abi_fields("pair_record")
    candidate_fields = _capture_abi_fields("candidate_record")
    missing_sidecar = sorted(EVENT_UNIVERSE_SIDECAR_FIELDS - envelope)
    missing_pair = sorted(V3_PAIR_COMPLETENESS_FIELDS - pair_fields)
    missing_candidate = sorted(V3_CANDIDATE_COMPLETENESS_FIELDS - candidate_fields)
    return {
        "missing_sidecar_fields": missing_sidecar,
        "missing_pair_fields": missing_pair,
        "missing_candidate_fields": missing_candidate,
        "all_available": not (missing_sidecar or missing_pair or missing_candidate),
    }


def audit_registration_v3_contract() -> dict[str, object]:
    """Mechanically select the registration-v3 ordered terminal.

    This audits the *contract* (schema + validator + ABI coverage), not an
    actual H0 guarantee or owner-accepted baseline.
    """
    issues: list[str] = []
    try:
        _schema_document(SCHEMA_V3)
        _schema_validate(
            {
                "schema": SCHEMA_V3,
                "record_id": "audit_probe_candidate_source_v3",
                "record_scope": "fixture_only",
                "record_state": "candidate-source",
                "consumer": {
                    "consumer_id": "audit_probe_consumer_v3",
                    "consumer_universe_id": V3_CONSUMER_UNIVERSE_ID,
                    "maximum_claim_layer": "gctm_model_specification",
                    "required_guarantee_ids": [],
                },
                "identity_bindings": dict(V3_IDENTITY_BINDINGS),
                "candidate_sources": [
                    {
                        "source_id": "audit_pair",
                        "consumer_object": "runtime_candidate_universe",
                        "stream": "pair_record",
                        "source_fields": sorted(V3_PAIR_COMPLETENESS_FIELDS),
                        "causal_availability": "online",
                    },
                    {
                        "source_id": "audit_candidate",
                        "consumer_object": "runtime_candidate_universe",
                        "stream": "candidate_record",
                        "source_fields": sorted(V3_CANDIDATE_COMPLETENESS_FIELDS),
                        "causal_availability": "online",
                    },
                    {
                        "source_id": "audit_sidecar",
                        "consumer_object": "runtime_candidate_universe",
                        "stream": "event_universe_sidecar",
                        "source_fields": sorted(EVENT_UNIVERSE_SIDECAR_FIELDS),
                        "causal_availability": "online",
                    },
                ],
            },
            SCHEMA_V3,
        )
    except RegistrationValidationError as exc:
        issues.append(f"schema_or_probe_invalid: {exc}")

    coverage = mandatory_completeness_fields_available_in_trace_v2()
    if issues:
        terminal = TERMINAL_V3_AUDIT_INVALID
    elif not coverage["all_available"]:
        terminal = TERMINAL_V3_REQUIRES_ABI_DELTA
    else:
        terminal = TERMINAL_V3_CONTRACT_SEALABLE

    return {
        "registration_identity": SCHEMA_V3,
        "consumer_universe": V3_CONSUMER_UNIVERSE_ID,
        "guarantee_class": "universe_completeness",
        "consumer_objects": sorted(V3_CONSUMER_OBJECTS),
        "selected_terminal": terminal,
        "terminal_order": list(TERMINAL_V3_ORDER),
        "trace_v2_abi_change_required": terminal == TERMINAL_V3_REQUIRES_ABI_DELTA,
        "abi_coverage": coverage,
        "audit_issues": issues,
        "actual_guarantee_established": False,
        "runtime_compatibility_established": False,
        "runtime_substrate_established": False,
        "authority_verified": False,
        "h0_reentry_authorized": False,
        "maximum_supported_conclusion": (
            "The registration-v3 contract is structurally capable of registering "
            "future H0 native-universe and event-membership completeness guarantees "
            f"for {V3_CONSUMER_UNIVERSE_ID}."
            if terminal == TERMINAL_V3_CONTRACT_SEALABLE
            else None
        ),
        "not_established": [
            "actual H0 guarantee",
            "accepted runtime baseline",
            "runtime substrate",
            "runtime compatibility",
            "H0 re-entry authority",
            "H0_ROUTE5_B1 activation",
            "GCTM_B1 activation",
            "O1 activation",
            "production claim",
        ],
    }


def validate_record(record: object) -> dict[str, object]:
    """Validate one registration record without asserting external authority."""
    schema_id = _record_schema_id(record)
    _schema_validate(record, schema_id)
    record_map = _as_mapping(record, "record")
    state = record_map["record_state"]
    if state == "candidate-source":
        if schema_id == SCHEMA_V1:
            _validate_candidate_identity_source(record_map)
        elif schema_id == SCHEMA_V2:
            _validate_candidate_sources_v2(record_map)
        elif schema_id == SCHEMA_V3:
            _validate_candidate_sources_v3(record_map)
        else:  # pragma: no cover - guarded by SCHEMA_PATHS
            raise RegistrationValidationError(
                f"unsupported registration schema {schema_id!r}"
            )
        result: dict[str, object] = {
            "schema": schema_id,
            "record_id": record_map["record_id"],
            "valid": True,
            "structurally_usable": False,
            "authority_verified": False,
            "disposition": "candidate-source",
        }
        if schema_id == SCHEMA_V3:
            result.update(V3_FIXED_NON_AUTHORITY)
        return result

    if state == "registered-guarantee":
        if schema_id == SCHEMA_V1:
            _validate_registered_identity_guarantee(record_map)
        elif schema_id == SCHEMA_V2:
            _validate_registered_guarantees_v2(record_map)
        elif schema_id == SCHEMA_V3:
            _validate_registered_guarantees_v3(record_map)
        else:  # pragma: no cover - guarded by SCHEMA_PATHS
            raise RegistrationValidationError(
                f"unsupported registration schema {schema_id!r}"
            )
        result = {
            "schema": schema_id,
            "record_id": record_map["record_id"],
            "valid": True,
            "structurally_usable": True,
            "authority_verified": False,
            "disposition": "registered-guarantee",
        }
        if schema_id == SCHEMA_V3:
            result.update(V3_FIXED_NON_AUTHORITY)
        return result

    raise RegistrationValidationError(f"unsupported record_state {state!r}")


def verify_record_file(path: Path) -> dict[str, object]:
    return validate_record(load_json(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record",
        type=Path,
        nargs="?",
        default=None,
        help="registration record JSON",
    )
    parser.add_argument(
        "--audit-contract",
        action="store_true",
        help="mechanically select the registration-v3 ordered terminal",
    )
    args = parser.parse_args()
    try:
        if args.audit_contract:
            result = audit_registration_v3_contract()
        elif args.record is not None:
            result = verify_record_file(args.record)
        else:
            parser.error("record path is required unless --audit-contract is set")
    except RegistrationValidationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
