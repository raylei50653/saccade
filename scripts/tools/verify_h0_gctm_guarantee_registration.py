#!/usr/bin/env python3
"""Fail-closed validator for H0-to-GCTM guarantee registration records.

This validator never verifies an H0 capture or owner acceptance. It determines
only whether a record is structurally and semantically sufficient for the
identity-v1 contract. A synthetic candidate-source record is deliberately
valid but can never produce ``structurally_usable: true``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "scripts/tools/h0_gctm_guarantee_registration_schema_v1.json"

PAIR_INSTANCE_KEY = (
    "seq",
    "frame",
    "cand_slot",
    "cand_instance_uid",
    "lost_slot",
    "lost_instance_uid",
)
IDENTITY_OBJECT = "event_runtime_instance_identity"
IDENTITY_REQUIRED_INVALIDATION_INPUTS = frozenset(
    {
        "runtime_instrumentation_identity",
        "policy_base_id",
        "resolved_preset_id",
        "h0_schema_version",
        "capture_mode",
        "event_key_version",
        "observation_state_semantics_version",
        "identity_lifecycle",
        "consumer_domain",
        "causal_availability",
    }
)


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


def _schema_document() -> Mapping[str, Any]:
    value = load_json(SCHEMA_PATH)
    if not isinstance(value, Mapping):  # pragma: no cover - static repository file
        raise RegistrationValidationError("registration schema must be a JSON object")
    return value


def _schema_validate(record: object) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RegistrationValidationError("jsonschema dependency unavailable") from exc

    schema = _schema_document()
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


def validate_record(record: object) -> dict[str, object]:
    """Validate one identity-v1 record without asserting external authority."""
    _schema_validate(record)
    record_map = _as_mapping(record, "record")
    state = record_map["record_state"]
    if state == "candidate-source":
        _validate_candidate_identity_source(record_map)
        return {
            "schema": "h0_gctm_guarantee_registration_v1",
            "record_id": record_map["record_id"],
            "valid": True,
            "structurally_usable": False,
            "authority_verified": False,
            "disposition": "candidate-source",
        }

    if state == "registered-guarantee":
        _validate_registered_identity_guarantee(record_map)
        return {
            "schema": "h0_gctm_guarantee_registration_v1",
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
