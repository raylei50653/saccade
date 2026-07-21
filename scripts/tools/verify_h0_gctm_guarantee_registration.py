#!/usr/bin/env python3
"""Fail-closed validator for H0-to-GCTM guarantee registration records.

This validator never verifies an H0 capture or owner acceptance. It determines
only whether a record is structurally and semantically sufficient for the
identity-v1 contract or the registration-v2 candidate-row inventory. A
synthetic candidate-source record is deliberately valid but can never produce
``structurally_usable: true``.
"""
# status: stable

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATHS = {
    "h0_gctm_guarantee_registration_v1": ROOT
    / "scripts/tools/h0_gctm_guarantee_registration_schema_v1.json",
    "h0_gctm_guarantee_registration_v2": ROOT
    / "scripts/tools/h0_gctm_guarantee_registration_schema_v2.json",
}
CAPTURE_ABI_SCHEMA_PATH = ROOT / "scripts/tools/h0_bridge_decision_trace_schema_v2.json"
CAPTURE_ABI_SCHEMA_ID = "h0_bridge_decision_trace_v2"

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


class RegistrationValidationError(ValueError):
    """The record cannot satisfy the identity-v1 registration contract."""


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


def validate_record(record: object) -> dict[str, object]:
    """Validate one registration record without asserting external authority."""
    schema_id = _record_schema_id(record)
    _schema_validate(record, schema_id)
    record_map = _as_mapping(record, "record")
    state = record_map["record_state"]
    if state == "candidate-source":
        if schema_id == "h0_gctm_guarantee_registration_v1":
            _validate_candidate_identity_source(record_map)
        else:
            _validate_candidate_sources_v2(record_map)
        return {
            "schema": schema_id,
            "record_id": record_map["record_id"],
            "valid": True,
            "structurally_usable": False,
            "authority_verified": False,
            "disposition": "candidate-source",
        }

    if state == "registered-guarantee":
        if schema_id == "h0_gctm_guarantee_registration_v1":
            _validate_registered_identity_guarantee(record_map)
        else:
            _validate_registered_guarantees_v2(record_map)
        return {
            "schema": schema_id,
            "record_id": record_map["record_id"],
            "valid": True,
            "structurally_usable": True,
            "authority_verified": False,
            "disposition": "registered-guarantee",
        }

    raise RegistrationValidationError(f"unsupported record_state {state!r}")


def verify_record_file(path: Path) -> dict[str, object]:
    return validate_record(load_json(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="registration record JSON")
    args = parser.parse_args()
    try:
        result = verify_record_file(args.record)
    except RegistrationValidationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
