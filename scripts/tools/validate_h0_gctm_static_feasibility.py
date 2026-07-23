#!/usr/bin/env python3
"""Fail-closed validator for the bounded H0 to GCTM static feasibility audit.

The validator checks identities, responsibility conservation, immutable
derivations, producer-source eligibility, consumer semantics, and the
independence of the two runtime consumer gates.  It never validates H0
authority, runtime fidelity, compatibility, substrate establishment, or
activation eligibility.
"""
# status: experiment

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "scripts/tools/h0_gctm_static_feasibility_schema_v1.json"
CAPTURE_ABI_PATH = ROOT / "scripts/tools/h0_bridge_decision_trace_schema_v2.json"
REGISTRATION_TOOLS = ROOT / "scripts/tools"
if REGISTRATION_TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, REGISTRATION_TOOLS.as_posix())

import verify_h0_gctm_guarantee_registration as registration  # noqa: E402


SCHEMA_ID = "h0_gctm_interface_static_feasibility_audit_v1"
TERMINAL_INVALID = "H0_GCTM_STATIC_AUDIT_INVALID"
TERMINAL_INSUFFICIENT = "H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT"
TERMINAL_FEASIBLE = "H0_GCTM_INTERFACE_STRUCTURALLY_FEASIBLE"
TERMINAL_ORDER = [TERMINAL_INVALID, TERMINAL_INSUFFICIENT, TERMINAL_FEASIBLE]

RESPONSIBILITY_CLASSES = (
    "H0_EXACT",
    "H0_DERIVED",
    "GCTM_DERIVED",
    "DECLARATION_CONSTANT",
    "B1_OFFLINE",
    "OUTSIDE_ENVELOPE",
    "UNAVAILABLE",
)

ESSENTIAL_INPUT_ROLES = {
    "gctm_d1_terminal_acceptance",
    "gctm_consumer_interface",
    "gctm_compatibility_requirements",
    "h0_consumer_compatibility_contract",
    "h0_capture_abi",
    "h0_guarantee_registration_schema",
    "h0_guarantee_registration_validator",
    "gctm_theory",
}

MANDATORY_CONSUMER_OBJECTS = {
    "event_id",
    "cand_id",
    "g_phys",
    "residual_position",
    "S_innovation",
    "covariance_semantics",
    "coordinate_dim_d",
    "observation_mode",
    "stratum_id",
    "context_drift_position",
    "score_orientation",
    "score_transform",
    "normalization",
    "tie_rule",
    "true_match_label",
    "candidate_universe",
    "event_membership",
}

EXPECTED_GATE_IDS = {
    "gctm_d1_to_h0_route5_b1_compatibility_v1": "H0_ROUTE5_B1",
    "gctm_d1_to_gctm_b1_compatibility_v1": "GCTM_B1",
}

FIXED_NON_AUTHORITY = {
    "authority_verified": False,
    "runtime_compatibility_established": False,
    "h0_runtime_substrate_established": False,
    "activation_eligible": False,
}

CONSUMER_INTERFACE_ROLE = "gctm_consumer_interface"
H0_COMPATIBILITY_CONTRACT_ROLE = "h0_consumer_compatibility_contract"
GCTM_THEORY_ROLE = "gctm_theory"

# These projections are part of the validator identity.  Each projection names
# exact text that must exist in the already path/hash-frozen source and fixes
# the only row semantics and availability that the source key may entail.
COMPATIBILITY_REQUIREMENT_PROJECTIONS = {
    "lost_candidate_identities_and_event_membership": {
        "frozen_input_role": H0_COMPATIBILITY_CONTRACT_ROLE,
        "consumer_object": "event_membership",
        "consumer_required_semantics": (
            "Complete event membership over the declared candidate universe, "
            "with expected native keys reconciled to observed candidate and "
            "pair records"
        ),
        "required_availability_time": (
            "after native event formation and before a runtime ranking claim"
        ),
        "source_markers": (
            "lost/candidate identities and event membership",
            "A pair-local state alone cannot determine an online decision.",
            "be the full candidate universe at the bridge\nevent.",
        ),
    },
    "production_cv_null_offset_treatment": {
        "frozen_input_role": H0_COMPATIBILITY_CONTRACT_ROLE,
        "consumer_object": "operator_offset_position",
        "consumer_required_semantics": (
            "Production operator horizon mismatch offset must remain separate "
            "from canonical residual mean and M2 context drift"
        ),
        "required_availability_time": "at candidate entry/bridge event",
        "source_markers": (
            "production-CV null offset and its declared treatment",
            "GCTM owns the physical-time mapping and the treatment of any\n"
            "induced null offset.",
        ),
    },
}

AUDIT_BOUNDARY_PROJECTIONS = {
    "production_cv_null_offset": {
        "frozen_input_role": GCTM_THEORY_ROLE,
        "consumer_object": "operator_offset_position",
        "consumer_required_semantics": (
            "Production operator horizon mismatch offset must remain separate "
            "from canonical residual mean and M2 context drift"
        ),
        "required_availability_time": "at candidate entry/bridge event",
        "source_markers": (
            "| 8 | `production_cv_null_offset` |",
            "| 9 | `null_offset_treatment` |",
            "`operator_offset` | required-if `operator_offset_declared=true`",
        ),
    },
}


class AuditValidationError(ValueError):
    """The static audit is structurally invalid."""

    def __init__(self, error_class: str, message: str):
        super().__init__(message)
        self.error_class = error_class


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditValidationError(
                "malformed_json", f"duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except AuditValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditValidationError(
            "malformed_json", f"malformed JSON {path}: {exc}"
        ) from exc


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditValidationError("semantic_shape", f"{name} must be an object")
    return value


def _as_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditValidationError("semantic_shape", f"{name} must be an array")
    return value


def _schema_validate(record: object) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - project dependency
        raise AuditValidationError(
            "dependency_unavailable", "jsonschema dependency unavailable"
        ) from exc

    schema = load_json(SCHEMA_PATH)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(record),
            key=lambda error: list(error.absolute_path),
        )
    except jsonschema.SchemaError as exc:  # pragma: no cover - repository file
        raise AuditValidationError(
            "invalid_schema", f"invalid audit schema: {exc.message}"
        ) from exc
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise AuditValidationError(
            "schema_rejection",
            f"schema rejection at {location}: {error.message}",
        )


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AuditValidationError(
            "frozen_identity", f"cannot read frozen input {path}: {exc}"
        ) from exc


def _input_by_role(record: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    matches = [
        _as_mapping(item, f"frozen_inputs[{index}]")
        for index, item in enumerate(
            _as_sequence(record["frozen_inputs"], "frozen_inputs")
        )
        if _as_mapping(item, f"frozen_inputs[{index}]").get("role") == role
    ]
    if len(matches) != 1:
        raise AuditValidationError(
            "frozen_identity",
            f"frozen input role {role!r} must occur exactly once",
        )
    return matches[0]


def _frozen_text(record: Mapping[str, Any], role: str) -> str:
    frozen_input = _input_by_role(record, role)
    path = ROOT / str(frozen_input["path"])
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditValidationError(
            "frozen_identity", f"cannot read frozen input {path}: {exc}"
        ) from exc


def _validate_frozen_inputs(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    inputs = _as_sequence(record["frozen_inputs"], "frozen_inputs")
    roles: list[str] = []
    identities: list[str] = []
    documents: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(inputs):
        item_map = _as_mapping(item, f"frozen_inputs[{index}]")
        role = str(item_map["role"])
        roles.append(role)
        identities.append(str(item_map["identity_id"]))
        path = ROOT / str(item_map["path"])
        actual = _sha256_path(path)
        if actual != item_map["sha256"]:
            raise AuditValidationError(
                "frozen_identity",
                f"{role} hash mismatch: declared={item_map['sha256']} actual={actual}",
            )
        if path.suffix == ".json":
            document = load_json(path)
            if isinstance(document, Mapping):
                documents[role] = document

    if not ESSENTIAL_INPUT_ROLES <= set(roles):
        missing = sorted(ESSENTIAL_INPUT_ROLES - set(roles))
        raise AuditValidationError(
            "frozen_identity", f"missing essential frozen input roles: {missing}"
        )
    if len(roles) != len(set(roles)):
        raise AuditValidationError(
            "frozen_identity", "frozen input roles must be unique"
        )
    if len(identities) != len(set(identities)):
        raise AuditValidationError(
            "frozen_identity", "frozen input identity_id values must be unique"
        )

    acceptance = documents.get("gctm_d1_terminal_acceptance")
    if acceptance is None:
        raise AuditValidationError(
            "frozen_identity", "D1 terminal acceptance must be JSON"
        )
    if acceptance.get("selected_terminal") != "GCTM_D1_INTERFACE_READY":
        raise AuditValidationError(
            "frozen_identity", "D1 terminal identity is not GCTM_D1_INTERFACE_READY"
        )
    if (
        acceptance.get("owner_acceptance_id")
        != "gctm_d1_terminal_owner_acceptance_20260723"
    ):
        raise AuditValidationError(
            "frozen_identity", "D1 terminal owner acceptance identity drifted"
        )

    capture_abi = documents.get("h0_capture_abi")
    if capture_abi is None or (
        capture_abi.get("capture_schema_version") != registration.CAPTURE_ABI_SCHEMA_ID
    ):
        raise AuditValidationError("frozen_identity", "H0 capture ABI identity drifted")

    registration_schema = documents.get("h0_guarantee_registration_schema")
    if registration_schema is None or (
        registration_schema.get("$id") != "h0_gctm_guarantee_registration_v2"
    ):
        raise AuditValidationError(
            "frozen_identity", "registration-v2 schema identity drifted"
        )
    return documents


def _canonical_definition_payload(definition: Mapping[str, Any]) -> bytes:
    payload = {
        key: definition[key]
        for key in (
            "expression",
            "inputs",
            "output",
            "causal_rule",
            "invalidation_set",
        )
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def definition_content_hash(definition: Mapping[str, Any]) -> str:
    """Return the sealed hash for one derivation's semantic payload."""
    return hashlib.sha256(_canonical_definition_payload(definition)).hexdigest()


def _validate_derivations(
    record: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    definitions = _as_sequence(
        record["derivation_definitions"], "derivation_definitions"
    )
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(definitions):
        definition = _as_mapping(item, f"derivation_definitions[{index}]")
        definition_id = str(definition["definition_id"])
        if definition_id in by_id:
            raise AuditValidationError(
                "derivation_identity", f"duplicate derivation {definition_id!r}"
            )
        actual_hash = definition_content_hash(definition)
        if definition["content_hash"] != actual_hash:
            raise AuditValidationError(
                "derivation_identity",
                f"derivation {definition_id!r} content hash mismatch: "
                f"declared={definition['content_hash']} actual={actual_hash}",
            )
        by_id[definition_id] = definition
    return by_id


def _capture_fields() -> dict[str, set[str]]:
    document = _as_mapping(load_json(CAPTURE_ABI_PATH), "capture ABI")
    record_fields = _as_mapping(document["record_fields"], "record_fields")
    result = {
        "pair_record": set(record_fields["pair_records"]),
        "candidate_record": set(record_fields["candidate_records"]),
        "claim_record": set(record_fields["claim_records"]),
        "commit_record": set(record_fields["commit_records"]),
        "capture_envelope": set(document["envelope_fields"]),
    }
    return result


def _registrable_fields() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {
        "pair_record": set(),
        "candidate_record": set(),
        "claim_record": set(),
        "commit_record": set(),
    }
    for _, stream_specs in registration.V2_CLASS_SPECS.values():
        for stream, fields in stream_specs.items():
            result[stream].update(fields)
    return result


def _validate_source_fields(row: Mapping[str, Any]) -> None:
    abi_fields = _capture_fields()
    registrable = _registrable_fields()
    h0_sources = []
    for index, item in enumerate(
        _as_sequence(row["producer_sources"], "producer_sources")
    ):
        source = _as_mapping(item, f"producer_sources[{index}]")
        if source["registered_guarantee_claimed"] is not False:
            raise AuditValidationError(
                "candidate_source_promotion",
                f"{row['consumer_object']} promotes a candidate source to a guarantee",
            )
        if source["source_state"] != "candidate-source":
            continue
        h0_sources.append(source)
        stream = str(source["stream"])
        field = str(source["field"])
        if stream not in abi_fields or field not in abi_fields[stream]:
            raise AuditValidationError(
                "producer_source",
                f"{row['consumer_object']} names absent H0 source {stream}/{field}",
            )

    registration_status = row["current_registration_v2_coverage"]
    if registration_status == "candidate_source_eligible":
        if not h0_sources:
            raise AuditValidationError(
                "registration_coverage",
                f"{row['consumer_object']} claims candidate eligibility without H0 fields",
            )
        unavailable = [
            f"{source['stream']}/{source['field']}"
            for source in h0_sources
            if source["stream"] not in registrable
            or source["field"] not in registrable[str(source["stream"])]
        ]
        if unavailable:
            raise AuditValidationError(
                "registration_coverage",
                f"{row['consumer_object']} sources are not registrable in v2: "
                f"{sorted(unavailable)}",
            )
    if (
        registration_status == "not_registrable_current_v2"
        and h0_sources
        and all(
            source["stream"] in registrable
            and source["field"] in registrable[str(source["stream"])]
            for source in h0_sources
        )
    ):
        raise AuditValidationError(
            "registration_coverage",
            f"{row['consumer_object']} is marked non-registrable but all H0 "
            "sources are registrable",
        )


def _validate_projected_binding(
    *,
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    source_key: str,
    projections: Mapping[str, Mapping[str, Any]],
) -> None:
    projection = projections.get(source_key)
    if projection is None:
        raise AuditValidationError(
            "consumer_binding",
            f"{row['consumer_object']} binds unknown frozen source {source_key!r}",
        )
    binding = _as_mapping(row["consumer_binding"], "consumer_binding")
    expected_role = str(projection["frozen_input_role"])
    if binding["frozen_input_role"] != expected_role:
        raise AuditValidationError(
            "consumer_binding",
            f"{row['consumer_object']} binds {source_key!r} to the wrong frozen input",
        )

    source_text = _frozen_text(record, expected_role)
    missing_markers = [
        marker
        for marker in _as_sequence(projection["source_markers"], "source_markers")
        if str(marker) not in source_text
    ]
    if missing_markers:
        raise AuditValidationError(
            "consumer_binding",
            f"{row['consumer_object']} frozen source {source_key!r} is absent or drifted",
        )

    for key in (
        "consumer_object",
        "consumer_required_semantics",
        "required_availability_time",
    ):
        if row[key] != projection[key]:
            raise AuditValidationError(
                "consumer_binding",
                f"{row['consumer_object']} {key} does not match frozen "
                f"projection {source_key!r}",
            )


def _validate_consumer_binding(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    consumer: Mapping[str, Any],
) -> None:
    binding = _as_mapping(row["consumer_binding"], "consumer_binding")
    binding_kind = binding["binding_kind"]
    source_key = str(binding["source_key"])
    if binding_kind == "required_runtime_field":
        if binding["frozen_input_role"] != CONSUMER_INTERFACE_ROLE:
            raise AuditValidationError(
                "consumer_binding",
                f"{row['consumer_object']} runtime field binds the wrong frozen input",
            )
        fields = _as_sequence(
            consumer.get("required_runtime_fields"), "consumer.required_runtime_fields"
        )
        matches = [
            _as_mapping(item, "required_runtime_field")
            for item in fields
            if _as_mapping(item, "required_runtime_field").get("name") == source_key
        ]
        if len(matches) != 1 or row["consumer_object"] != source_key:
            raise AuditValidationError(
                "consumer_coverage",
                f"{row['consumer_object']} does not uniquely bind D1 field {source_key}",
            )
        field = matches[0]
        expected = {
            "consumer_required_semantics": field["semantic_meaning"],
            "shape": field["shape"],
            "unit": field["units"],
            "sharing": field["event_shared_or_candidate_specific"],
            "required_availability_time": field["available_when"],
        }
        for key, value in expected.items():
            if row[key] != value:
                raise AuditValidationError(
                    "consumer_semantics",
                    f"{row['consumer_object']} {key} drifted from D1 interface",
                )
    elif binding_kind == "top_level_policy":
        if (
            binding["frozen_input_role"] != CONSUMER_INTERFACE_ROLE
            or source_key not in consumer
            or row["consumer_object"] != source_key
        ):
            raise AuditValidationError(
                "consumer_coverage",
                f"{row['consumer_object']} binds absent D1 policy {source_key}",
            )
        if row["consumer_required_semantics"] != consumer[source_key]:
            raise AuditValidationError(
                "consumer_semantics",
                f"{row['consumer_object']} semantic value drifted from D1 policy",
            )
    elif binding_kind == "compatibility_contract_requirement":
        _validate_projected_binding(
            row=row,
            record=record,
            source_key=source_key,
            projections=COMPATIBILITY_REQUIREMENT_PROJECTIONS,
        )
    elif binding_kind == "audit_boundary":
        _validate_projected_binding(
            row=row,
            record=record,
            source_key=source_key,
            projections=AUDIT_BOUNDARY_PROJECTIONS,
        )
    else:
        raise AuditValidationError(
            "consumer_binding",
            f"{row['consumer_object']} uses unsupported binding kind {binding_kind!r}",
        )


def _validate_relation_and_responsibility(
    row: Mapping[str, Any], definitions: Mapping[str, Mapping[str, Any]]
) -> None:
    obj = str(row["consumer_object"])
    responsibility = str(row["responsibility_class"])
    relation = str(row["relation"])
    sources = [
        _as_mapping(item, f"{obj}.producer_sources")
        for item in _as_sequence(row["producer_sources"], "producer_sources")
    ]
    h0_sources = [
        item for item in sources if item["source_state"] == "candidate-source"
    ]

    expected_relation = {
        "H0_EXACT": "exact",
        "H0_DERIVED": "derived",
        "GCTM_DERIVED": "derived",
        "DECLARATION_CONSTANT": "declaration",
        "B1_OFFLINE": "offline",
        "OUTSIDE_ENVELOPE": "unavailable",
        "UNAVAILABLE": "unavailable",
    }[responsibility]
    if relation != expected_relation:
        raise AuditValidationError(
            "responsibility_relation",
            f"{obj} responsibility {responsibility} requires {expected_relation}",
        )

    has_binding = "derivation_binding" in row
    if responsibility in {"H0_DERIVED", "GCTM_DERIVED"}:
        if not has_binding:
            raise AuditValidationError(
                "derivation_identity", f"{obj} derived relation lacks binding"
            )
        binding = _as_mapping(row["derivation_binding"], "derivation_binding")
        definition_id = str(binding["definition_id"])
        definition = definitions.get(definition_id)
        if definition is None:
            raise AuditValidationError(
                "derivation_identity", f"{obj} binds unknown derivation {definition_id}"
            )
        for key in ("definition_version", "content_hash"):
            if binding[key] != definition[key]:
                raise AuditValidationError(
                    "derivation_identity",
                    f"{obj} derivation {key} does not match definition",
                )
        if definition["owner_class"] != responsibility:
            raise AuditValidationError(
                "derivation_identity",
                f"{obj} derivation owner does not match {responsibility}",
            )
        input_names = [item["name"] for item in definition["inputs"]]
        if list(row["derivation_inputs"]) != input_names:
            raise AuditValidationError(
                "derivation_identity", f"{obj} derivation inputs drifted"
            )
        if list(row["derivation_invalidation_set"]) != list(
            definition["invalidation_set"]
        ):
            raise AuditValidationError(
                "derivation_identity", f"{obj} invalidation set drifted"
            )
        if definition["output"] != {"shape": row["shape"], "unit": row["unit"]}:
            raise AuditValidationError(
                "shape_unit",
                f"{obj} derivation output shape/unit does not match consumer row",
            )
    elif has_binding or row["derivation_inputs"] or row["derivation_invalidation_set"]:
        raise AuditValidationError(
            "derivation_identity",
            f"{obj} non-derived responsibility carries derivation material",
        )

    if responsibility == "H0_EXACT":
        if not h0_sources:
            raise AuditValidationError(
                "producer_source", f"{obj} H0_EXACT lacks an H0 candidate field"
            )
        if row["current_abi_coverage"] != "candidate_source_available":
            raise AuditValidationError(
                "producer_source", f"{obj} H0_EXACT is not marked candidate-source"
            )
    if responsibility == "H0_DERIVED" and not h0_sources:
        raise AuditValidationError(
            "producer_source", f"{obj} H0_DERIVED lacks H0 candidate inputs"
        )
    if responsibility == "DECLARATION_CONSTANT":
        if h0_sources or any(item["stream"] != "declaration" for item in sources):
            raise AuditValidationError(
                "declaration_promotion",
                f"{obj} declaration constant masquerades as a runtime field",
            )
    if responsibility == "B1_OFFLINE":
        if obj == "true_match_label" and (
            row["requires_b1_only_treatment"] is not True or h0_sources
        ):
            raise AuditValidationError(
                "b1_boundary", "true_match_label must stay B1-only and outside H0"
            )
        if row["runtime_observable"] is not False:
            raise AuditValidationError(
                "b1_boundary", f"{obj} B1 offline row cannot be runtime-observable"
            )
    if responsibility in {"OUTSIDE_ENVELOPE", "UNAVAILABLE"}:
        if row["disposition_status"] not in {"outside_envelope", "unavailable"}:
            raise AuditValidationError(
                "outside_envelope_promotion",
                f"{obj} unavailable responsibility masquerades as covered",
            )
        if row["current_abi_coverage"] not in {"outside_envelope", "unavailable"}:
            raise AuditValidationError(
                "outside_envelope_promotion",
                f"{obj} unavailable responsibility claims ABI coverage",
            )

    causal = _as_mapping(row["causal_requirement"], "causal_requirement")
    if row["runtime_observable"] and causal["future_info_leak_allowed"] is not False:
        raise AuditValidationError(
            "causal_availability",
            f"{obj} runtime object allows future information leakage",
        )
    if causal["causal_status"] == "unavailable" and row["disposition_status"] not in {
        "outside_envelope",
        "unavailable",
    }:
        raise AuditValidationError(
            "causal_availability",
            f"{obj} unavailable causal source masquerades as covered",
        )


def _unresolved_runtime_objects(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        str(row["consumer_object"])
        for row in rows
        if row["runtime_observable"]
        and (
            row["responsibility_class"] in {"OUTSIDE_ENVELOPE", "UNAVAILABLE"}
            or row["requires_h0_delta"]
            or row["disposition_status"] in {"outside_envelope", "unavailable"}
        )
    )


def _validate_conservation(
    record: Mapping[str, Any], rows: list[Mapping[str, Any]]
) -> None:
    conservation = _as_mapping(record["coverage_conservation"], "coverage_conservation")
    row_objects = [str(row["consumer_object"]) for row in rows]
    if len(row_objects) != len(set(row_objects)):
        raise AuditValidationError(
            "coverage_conservation", "consumer_object rows must be unique"
        )
    if not MANDATORY_CONSUMER_OBJECTS <= set(row_objects):
        missing = sorted(MANDATORY_CONSUMER_OBJECTS - set(row_objects))
        raise AuditValidationError(
            "coverage_conservation",
            f"required consumer objects have no disposition: {missing}",
        )
    if set(conservation["required_consumer_objects"]) != MANDATORY_CONSUMER_OBJECTS:
        raise AuditValidationError(
            "coverage_conservation", "required consumer inventory drifted"
        )
    if list(conservation["audited_consumer_objects"]) != row_objects:
        raise AuditValidationError(
            "coverage_conservation", "audited consumer object order drifted"
        )

    counts = Counter(str(row["responsibility_class"]) for row in rows)
    declared_counts = _as_mapping(
        conservation["responsibility_counts"], "responsibility_counts"
    )
    expected_counts = {name: counts.get(name, 0) for name in RESPONSIBILITY_CLASSES}
    if dict(declared_counts) != expected_counts:
        raise AuditValidationError(
            "coverage_conservation",
            f"responsibility counts do not conserve rows: "
            f"declared={dict(declared_counts)} actual={expected_counts}",
        )
    if conservation["total_rows"] != len(rows) or sum(counts.values()) != len(rows):
        raise AuditValidationError(
            "coverage_conservation", "total row conservation failed"
        )
    runtime_count = sum(bool(row["runtime_observable"]) for row in rows)
    offline_count = sum(not bool(row["runtime_observable"]) for row in rows)
    if conservation["runtime_observable_count"] != runtime_count:
        raise AuditValidationError(
            "coverage_conservation", "runtime-observable count drifted"
        )
    if conservation["offline_count"] != offline_count:
        raise AuditValidationError("coverage_conservation", "offline count drifted")
    unresolved = _unresolved_runtime_objects(rows)
    if list(conservation["unresolved_runtime_objects"]) != unresolved:
        raise AuditValidationError(
            "coverage_conservation",
            f"unresolved runtime inventory drifted: expected {unresolved}",
        )


def _validate_runtime_gates(
    record: Mapping[str, Any], compatibility: Mapping[str, Any]
) -> None:
    audit_gates = [
        _as_mapping(item, f"runtime_consumer_gates[{index}]")
        for index, item in enumerate(
            _as_sequence(record["runtime_consumer_gates"], "runtime_consumer_gates")
        )
    ]
    by_id = {str(item["gate_id"]): item for item in audit_gates}
    if set(by_id) != set(EXPECTED_GATE_IDS):
        raise AuditValidationError(
            "gate_conservation", "the two runtime consumer gates are not conserved"
        )
    compatibility_gates = {
        str(item["gate_id"]): _as_mapping(item, "compatibility gate")
        for item in _as_sequence(compatibility["gates"], "compatibility.gates")
    }
    for gate_id, slot_id in EXPECTED_GATE_IDS.items():
        gate = by_id[gate_id]
        source_gate = compatibility_gates.get(gate_id)
        if source_gate is None:
            raise AuditValidationError(
                "gate_conservation", f"frozen compatibility gate {gate_id} is absent"
            )
        if (
            gate["consumer_slot_id"] != slot_id
            or source_gate["consumer_slot_id"] != slot_id
        ):
            raise AuditValidationError(
                "gate_conservation", f"gate {gate_id} consumer identity drifted"
            )
        if source_gate["status"] != "missing":
            raise AuditValidationError(
                "gate_conservation", f"gate {gate_id} must remain missing"
            )


def validate_record(record: object) -> dict[str, object]:
    """Validate a canonical audit and mechanically select its bounded terminal."""
    _schema_validate(record)
    record_map = _as_mapping(record, "record")
    if record_map["schema"] != SCHEMA_ID:
        raise AuditValidationError("schema_rejection", "audit schema identity drifted")
    if list(record_map["terminal_order"]) != TERMINAL_ORDER:
        raise AuditValidationError("terminal_order", "terminal order drifted")
    documents = _validate_frozen_inputs(record_map)
    definitions = _validate_derivations(record_map)
    consumer = documents.get("gctm_consumer_interface")
    compatibility = documents.get("gctm_compatibility_requirements")
    if consumer is None or compatibility is None:
        raise AuditValidationError(
            "frozen_identity", "consumer and compatibility inputs must be JSON"
        )

    rows = [
        _as_mapping(item, f"rows[{index}]")
        for index, item in enumerate(_as_sequence(record_map["rows"], "rows"))
    ]
    for row in rows:
        _validate_consumer_binding(row, record_map, consumer)
        _validate_relation_and_responsibility(row, definitions)
        _validate_source_fields(row)
    _validate_conservation(record_map, rows)
    _validate_runtime_gates(record_map, compatibility)

    for key, value in FIXED_NON_AUTHORITY.items():
        if record_map["fixed_non_authority"][key] is not value:
            raise AuditValidationError(
                "non_authority_boundary", f"{key} must remain false"
            )

    unresolved = _unresolved_runtime_objects(rows)
    selected = TERMINAL_INSUFFICIENT if unresolved else TERMINAL_FEASIBLE
    if record_map["selected_terminal"] != selected:
        raise AuditValidationError(
            "terminal_selection",
            f"declared terminal {record_map['selected_terminal']} does not match "
            f"mechanical selection {selected}",
        )

    counts = Counter(str(row["responsibility_class"]) for row in rows)
    return {
        "schema": SCHEMA_ID,
        "audit_id": record_map["audit_id"],
        "valid": True,
        "selected_terminal": selected,
        "responsibility_counts": {
            name: counts.get(name, 0) for name in RESPONSIBILITY_CLASSES
        },
        "unresolved_runtime_objects": unresolved,
        **FIXED_NON_AUTHORITY,
    }


def validate_record_file(path: Path) -> dict[str, object]:
    return validate_record(load_json(path))


def _invalid_result(path: Path, exc: AuditValidationError) -> dict[str, object]:
    return {
        "schema": SCHEMA_ID,
        "audit_path": path.as_posix(),
        "valid": False,
        "selected_terminal": TERMINAL_INVALID,
        "error_class": exc.error_class,
        "error": str(exc),
        **FIXED_NON_AUTHORITY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path, help="responsibility matrix JSON")
    args = parser.parse_args()
    try:
        result = validate_record_file(args.audit)
    except AuditValidationError as exc:
        print(json.dumps(_invalid_result(args.audit, exc), sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
