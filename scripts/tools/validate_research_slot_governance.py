#!/usr/bin/env python3
"""Fail-closed validator for research slot identity and authority records."""

# status: stable

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "research_slot_governance_v1"
SCHEMA_PATH = ROOT / "scripts/tools/research_slot_governance_schema_v1.json"

RUNTIME_GATE_CLASSES = frozenset(
    {
        "runtime_substrate",
        "runtime_provenance",
        "runtime_evidence_identity",
        "runtime_checksum",
        "runtime_consumer_compatibility",
        "runtime_activation_authority",
    }
)
RUNTIME_COMPATIBILITY_CHECKS = frozenset(
    {
        "canonical_h0_evidence_manifest",
        "stable_evidence_identity",
        "checksum_verification",
        "producer_consumer_schema_compatibility",
        "observation_semantics_compatibility",
        "parameterization_compatibility",
        "score_transformation_normalization_compatibility",
        "ordering_preservation_verdict",
    }
)


class SlotGovernanceValidationError(ValueError):
    """A slot-governance record violates one fail-closed rule."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SlotGovernanceValidationError(
                "duplicate_json_key", f"duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except SlotGovernanceValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlotGovernanceValidationError(
            "malformed_json", f"malformed JSON {path}: {exc}"
        ) from exc


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SlotGovernanceValidationError(
            "semantic_type", f"{name} must be an object"
        )
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SlotGovernanceValidationError("semantic_type", f"{name} must be an array")
    return value


def _schema_validate(record: object) -> None:
    schema = load_json(SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = "/".join(str(item) for item in error.absolute_path) or "<root>"
        raise SlotGovernanceValidationError(
            "schema_rejection", f"{location}: {error.message}"
        )


def _unique_by(
    items: Sequence[Any], field: str, collection: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(items):
        item = _mapping(value, f"{collection}[{index}]")
        identity = item[field]
        if identity in result:
            raise SlotGovernanceValidationError(
                "duplicate_identity", f"duplicate {field} {identity!r}"
            )
        result[identity] = item
    return result


def _validate_relations(
    record: Mapping[str, Any], slots: Mapping[str, Mapping[str, Any]]
) -> None:
    seen_pairs: set[frozenset[str]] = set()
    for index, value in enumerate(_sequence(record["relations"], "relations")):
        relation = _mapping(value, f"relations[{index}]")
        source = relation["source_slot_id"]
        target = relation["target_slot_id"]
        if source not in slots or target not in slots or source == target:
            raise SlotGovernanceValidationError(
                "relation_reference",
                f"relation endpoints must be two distinct declared slots: {source}, {target}",
            )
        pair = frozenset({source, target})
        if pair in seen_pairs:
            raise SlotGovernanceValidationError(
                "duplicate_relation", f"duplicate relation for {sorted(pair)}"
            )
        seen_pairs.add(pair)

        kind = relation["relation"]
        if kind in {"coexist", "isolated"} and (
            relation["aliases"]
            or relation["supersedes"]
            or relation["shares_activation_authority"]
            or not relation["change_requires_owner_transition"]
        ):
            raise SlotGovernanceValidationError(
                "relation_semantics",
                f"{kind} slots cannot alias, supersede, or share activation authority",
            )
        if kind == "equal" and not relation["aliases"]:
            raise SlotGovernanceValidationError(
                "relation_semantics", "equal relation must explicitly declare aliasing"
            )
        if kind == "supersede" and not relation["supersedes"]:
            raise SlotGovernanceValidationError(
                "relation_semantics",
                "supersede relation must explicitly declare supersession",
            )


def _validate_slots(slots: Mapping[str, Mapping[str, Any]]) -> None:
    for slot_id, slot in slots.items():
        authority = slot["authority_class"]
        evidence = set(slot["allowed_evidence_classes"])
        exclusions = set(slot["cannot_satisfy_gate_classes"])
        if authority == "diagnostic_only":
            if "h0_runtime_evidence" in evidence:
                raise SlotGovernanceValidationError(
                    "diagnostic_evidence_boundary",
                    f"diagnostic slot {slot_id} cannot admit H0 runtime evidence",
                )
            if exclusions != RUNTIME_GATE_CLASSES:
                raise SlotGovernanceValidationError(
                    "diagnostic_evidence_boundary",
                    f"diagnostic slot {slot_id} must disclaim every runtime gate class",
                )
        if slot["state"] == "active" and slot["owner_acceptance_id"] is None:
            raise SlotGovernanceValidationError(
                "activation_authority",
                f"active slot {slot_id} lacks owner acceptance",
            )
        if slot["state"] == "active" and slot["blocked_by"]:
            raise SlotGovernanceValidationError(
                "activation_blocked",
                f"active slot {slot_id} still has blockers {slot['blocked_by']}",
            )


def _validate_compatibility(
    record: Mapping[str, Any], slots: Mapping[str, Mapping[str, Any]]
) -> None:
    gates = _unique_by(
        _sequence(record["compatibility_gates"], "compatibility_gates"),
        "gate_id",
        "compatibility_gates",
    )
    for gate_id, gate in gates.items():
        producer = gate["producer_slot_id"]
        consumer = gate["consumer_slot_id"]
        if producer not in slots or consumer not in slots or producer == consumer:
            raise SlotGovernanceValidationError(
                "compatibility_reference",
                f"compatibility gate {gate_id} endpoints must be distinct declared slots",
            )
        if set(gate["required_checks"]) != RUNTIME_COMPATIBILITY_CHECKS:
            raise SlotGovernanceValidationError(
                "compatibility_completeness",
                f"compatibility gate {gate_id} does not declare the complete runtime gate",
            )
        if gate["status"] != "compatible" and gate["verdict_owner_acceptance_id"]:
            raise SlotGovernanceValidationError(
                "compatibility_authority",
                f"non-compatible gate {gate_id} cannot carry an acceptance id",
            )
        if gate["status"] == "compatible" and not gate["verdict_owner_acceptance_id"]:
            raise SlotGovernanceValidationError(
                "compatibility_authority",
                f"compatible gate {gate_id} lacks owner acceptance",
            )


def _validate_terminal_policies(
    record: Mapping[str, Any], slots: Mapping[str, Mapping[str, Any]]
) -> None:
    policies = _unique_by(
        _sequence(record["terminal_policies"], "terminal_policies"),
        "slot_id",
        "terminal_policies",
    )
    for slot_id, policy in policies.items():
        if slot_id not in slots:
            raise SlotGovernanceValidationError(
                "terminal_reference",
                f"terminal policy references unknown slot {slot_id}",
            )
        affected = set(policy["may_transition_slot_ids"])
        unlocked = set(policy["unlocks_slot_ids"])
        if not affected <= slots.keys() or not unlocked <= slots.keys():
            raise SlotGovernanceValidationError(
                "terminal_reference",
                f"terminal policy {slot_id} references unknown affected slots",
            )
        if slots[slot_id]["authority_class"] == "diagnostic_only" and (
            affected != {slot_id}
            or unlocked
            or policy["generates_decision_relevant_candidate"]
        ):
            raise SlotGovernanceValidationError(
                "diagnostic_terminal_boundary",
                f"diagnostic terminal for {slot_id} must be local and non-activating",
            )


def _validate_registry_projection(
    record: Mapping[str, Any], slots: Mapping[str, Mapping[str, Any]]
) -> None:
    projection = _mapping(record["registry_projection"], "registry_projection")
    projected = _unique_by(
        _sequence(projection["slot_states"], "registry_projection.slot_states"),
        "slot_id",
        "registry_projection.slot_states",
    )
    if set(projected) != set(slots):
        raise SlotGovernanceValidationError(
            "registry_projection",
            "registry projection must cover every declared slot exactly once",
        )
    for slot_id, slot in slots.items():
        item = projected[slot_id]
        if item["state"] != slot["state"] or set(item["blocked_by"]) != set(
            slot["blocked_by"]
        ):
            raise SlotGovernanceValidationError(
                "registry_projection",
                f"registry projection drifts from authoritative slot {slot_id}",
            )

    candidates = set(projection["decision_relevant_candidates"])
    active_wip = set(projection["active_wip"])
    if not candidates <= slots.keys() or not active_wip <= slots.keys():
        raise SlotGovernanceValidationError(
            "registry_projection", "candidate or WIP projection references unknown slot"
        )
    for slot_id, slot in slots.items():
        blocked = bool(slot["blocked_by"])
        if (blocked or slot["state"] != "active") and slot_id in active_wip:
            raise SlotGovernanceValidationError(
                "false_active_wip", f"non-active or blocked slot {slot_id} holds WIP"
            )
        if (blocked or slot["state"] != "active") and slot_id in candidates:
            raise SlotGovernanceValidationError(
                "false_candidate",
                f"non-active or blocked slot {slot_id} is a decision candidate",
            )


def validate_record(record: object) -> dict[str, object]:
    """Validate one machine-readable research-slot governance record."""
    record_map = _mapping(record, "record")
    if record_map.get("schema") != SCHEMA_ID:
        raise SlotGovernanceValidationError(
            "unsupported_schema",
            f"unsupported slot-governance schema {record_map.get('schema')!r}",
        )
    _schema_validate(record)
    slots = _unique_by(_sequence(record_map["slots"], "slots"), "slot_id", "slots")
    _validate_slots(slots)
    _validate_relations(record_map, slots)
    _validate_compatibility(record_map, slots)
    _validate_terminal_policies(record_map, slots)
    _validate_registry_projection(record_map, slots)

    decision = _mapping(record_map["owner_decision"], "owner_decision")
    return {
        "schema": SCHEMA_ID,
        "record_id": record_map["record_id"],
        "valid": True,
        "owner_decision_status": decision["status"],
        "authority_verified": decision["status"] == "accepted",
        "activation_eligible_slots": sorted(
            slot_id
            for slot_id, slot in slots.items()
            if slot["state"] == "active"
            and not slot["blocked_by"]
            and slot["owner_acceptance_id"] is not None
        ),
        "decision_relevant_candidates": list(
            _mapping(record_map["registry_projection"], "registry_projection")[
                "decision_relevant_candidates"
            ]
        ),
        "active_wip": list(
            _mapping(record_map["registry_projection"], "registry_projection")[
                "active_wip"
            ]
        ),
    }


def validate_record_file(path: Path) -> dict[str, object]:
    return validate_record(load_json(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="slot-governance record JSON")
    args = parser.parse_args()
    try:
        report = validate_record_file(args.record)
    except SlotGovernanceValidationError as exc:
        parser.error(f"{exc.error_class}: {exc}")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
