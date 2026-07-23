#!/usr/bin/env python3
"""Fail-closed validator for L2 score-ranking declaration v1 records."""

# status: stable

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ID = "score_ranking_declaration_v1"
SCHEMA_PATH = ROOT / "scripts/tools/score_ranking_declaration_schema_v1.json"

RUNG_ORDER = ("SR0", "SR1", "SR2", "SR3", "SR4", "SR5", "SR6")
RUNG_OBLIGATIONS: dict[str, frozenset[str]] = {
    "SR0": frozenset(
        {
            "policy_identity",
            "space_identity",
            "orientation_transform_tie",
            "denominator_semantics",
        }
    ),
    "SR1": frozenset(
        {
            "policy_search_space",
            "fit_selection_data",
            "exposure",
            "primary_metric_effect",
            "candidate_universe_conservation",
        }
    ),
    "SR2": frozenset(
        {
            "disjoint_partitions",
            "blind_reveal_binding",
            "per_fold_result",
            "dependence_uncertainty",
            "no_refit",
        }
    ),
    "SR3": frozenset(
        {
            "cross_sequence_stability",
            "gap_context_stability",
            "protected_stratum_retention",
            "short_gap_retention",
            "candidate_universe_invariance",
            "fallback_audit",
        }
    ),
    "SR4": frozenset(
        {
            "quantity_fidelity",
            "hook_semantic_equivalence",
            "runtime_causal_availability",
            "candidate_universe_parity",
            "transform_fallback_parity",
        }
    ),
    "SR5": frozenset(
        {
            "default_off_ab",
            "applied_rejected_audit",
            "online_state_provenance",
            "online_fallback_abstention",
            "disabled_baseline_unchanged",
        }
    ),
    "SR6": frozenset(
        {
            "system_efficacy_contract",
            "track_sequence_metrics",
            "latency_resource_audit",
            "failure_mode_rollback",
            "explicit_acceptance",
        }
    ),
}
CLAIM_SPACE_BY_RUNG = {
    "SR0": "score_observation",
    "SR1": "ranking",
    "SR2": "ranking",
    "SR3": "ranking",
    "SR4": "portable_ranking",
    "SR5": "assignment",
    "SR6": "system",
}
QUANTIFICATION_BY_CLAIM_SPACE = {
    "score_observation": "candidate_event",
    "ranking": "candidate_event",
    "portable_ranking": "candidate_event",
    "assignment": "assignment",
    "system": "system",
}
OUTCOME_CLASSES = frozenset({"invalid", "valid_negative", "valid_positive"})


class ScoreRankingValidationError(ValueError):
    """A declaration fails one fail-closed L2 contract class."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScoreRankingValidationError(
                "duplicate_json_key", f"duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except ScoreRankingValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreRankingValidationError(
            "malformed_json", f"malformed JSON {path}: {exc}"
        ) from exc


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScoreRankingValidationError("semantic_type", f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScoreRankingValidationError("semantic_type", f"{name} must be an array")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    values = _sequence(value, name)
    if not all(isinstance(item, str) for item in values):
        raise ScoreRankingValidationError(
            "semantic_type", f"{name} must contain only strings"
        )
    return tuple(values)


def _schema_document() -> Mapping[str, Any]:
    value = load_json(SCHEMA_PATH)
    return _mapping(value, "score-ranking schema")


def _schema_validate(record: object) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ScoreRankingValidationError(
            "schema_runtime", "jsonschema dependency unavailable"
        ) from exc

    schema = _schema_document()
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(record),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
    except jsonschema.SchemaError as exc:  # pragma: no cover - static schema
        raise ScoreRankingValidationError(
            "invalid_schema", f"invalid score-ranking schema: {exc.message}"
        ) from exc
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ScoreRankingValidationError(
            "schema_rejection",
            f"schema rejection at {location}: {error.message}",
        )


def _validate_contract_binding(record: Mapping[str, Any]) -> None:
    binding = _mapping(record["contract_binding"], "contract_binding")
    if record["record_scope"] == "fixture_only":
        if binding["contract_status"] != "proposed":
            raise ScoreRankingValidationError(
                "fixture_authority",
                "fixture_only declarations may bind only a proposed contract",
            )
        if (
            binding["owner_acceptance_id"] is not None
            or binding["registry_binding_id"] is not None
        ):
            raise ScoreRankingValidationError(
                "fixture_authority",
                "fixture_only declarations may not claim acceptance or registry binding",
            )


def _validate_policy(record: Mapping[str, Any]) -> None:
    policy = _mapping(record["policy"], "policy")
    candidate_universe = _mapping(
        policy["candidate_universe"], "policy.candidate_universe"
    )
    score = _mapping(policy["score"], "policy.score")
    tie_rule = _mapping(policy["tie_rule"], "policy.tie_rule")
    cutoff = _mapping(policy["cutoff"], "policy.cutoff")
    spaces = _mapping(record["spaces"], "spaces")

    component_ids = _strings(score["component_ids"], "score.component_ids")
    if score["transform_kind"] == "identity" and len(component_ids) != 1:
        raise ScoreRankingValidationError(
            "policy_transform",
            "identity transform requires exactly one score component",
        )

    candidate_keys = _strings(
        candidate_universe["candidate_key_fields"],
        "candidate_universe.candidate_key_fields",
    )
    tie_keys = _strings(tie_rule["key_fields"], "tie_rule.key_fields")
    if tie_rule["kind"] == "stable_candidate_key" and tie_keys != candidate_keys:
        raise ScoreRankingValidationError(
            "tie_rule",
            "stable tie key_fields must exactly equal candidate_key_fields",
        )

    if cutoff["role"] == "assignment_rule" and spaces["assignment_space_id"] is None:
        raise ScoreRankingValidationError(
            "cutoff_space",
            "assignment_rule cutoff requires an assignment_space_id",
        )


def _validate_reductions(record: Mapping[str, Any]) -> None:
    policy = _mapping(record["policy"], "policy")
    spaces = _mapping(record["spaces"], "spaces")
    reductions = _sequence(spaces["reductions"], "spaces.reductions")
    seen_ids: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    for index, reduction in enumerate(reductions):
        item = _mapping(reduction, f"spaces.reductions[{index}]")
        reduction_id = item["reduction_id"]
        edge = (item["source_space_id"], item["target_space_id"])
        if reduction_id in seen_ids or edge in seen_edges:
            raise ScoreRankingValidationError(
                "duplicate_identity",
                f"duplicate reduction identity or edge at index {index}",
            )
        if edge[0] == edge[1]:
            raise ScoreRankingValidationError(
                "reduction_graph", "a reduction may not map a space to itself"
            )
        seen_ids.add(reduction_id)
        seen_edges.add(edge)

    assignment_space = spaces["assignment_space_id"]
    system_space = spaces["system_space_id"]
    if assignment_space is not None:
        required_edge = (policy["event_space_id"], assignment_space)
        if required_edge not in seen_edges:
            raise ScoreRankingValidationError(
                "reduction_graph",
                "assignment_space_id requires an event-to-assignment reduction",
            )
    if system_space is not None:
        if assignment_space is None:
            raise ScoreRankingValidationError(
                "reduction_graph",
                "system_space_id requires an assignment_space_id",
            )
        required_edge = (assignment_space, system_space)
        if required_edge not in seen_edges:
            raise ScoreRankingValidationError(
                "reduction_graph",
                "system_space_id requires an assignment-to-system reduction",
            )


def _validate_claim(record: Mapping[str, Any]) -> None:
    claim = _mapping(record["claim"], "claim")
    spaces = _mapping(record["spaces"], "spaces")
    kappa = _mapping(claim["kappa"], "claim.kappa")
    target_rung = claim["target_rung"]
    expected_claim_space = CLAIM_SPACE_BY_RUNG[target_rung]
    if claim["claim_space"] != expected_claim_space:
        raise ScoreRankingValidationError(
            "claim_above_rung",
            f"{target_rung} requires claim_space {expected_claim_space!r}",
        )
    expected_quantification = QUANTIFICATION_BY_CLAIM_SPACE[expected_claim_space]
    if kappa["quantification_space"] != expected_quantification:
        raise ScoreRankingValidationError(
            "claim_space",
            f"{expected_claim_space} requires quantification_space "
            f"{expected_quantification!r}",
        )

    if claim["primary_ranking_metric"] == "correct_assignment_rate":
        if spaces["assignment_space_id"] is None:
            raise ScoreRankingValidationError(
                "claim_space",
                "correct_assignment_rate requires an assignment_space_id",
            )
    if target_rung in {"SR5", "SR6"} and spaces["assignment_space_id"] is None:
        raise ScoreRankingValidationError(
            "claim_space", f"{target_rung} requires an assignment_space_id"
        )
    if target_rung == "SR6" and spaces["system_space_id"] is None:
        raise ScoreRankingValidationError(
            "claim_space", "SR6 requires a system_space_id"
        )


def _validate_rungs(record: Mapping[str, Any]) -> None:
    claim = _mapping(record["claim"], "claim")
    target_rung = claim["target_rung"]
    target_index = RUNG_ORDER.index(target_rung)
    expected_rungs = RUNG_ORDER[: target_index + 1]
    obligations = _sequence(record["rung_obligations"], "rung_obligations")
    declared_rungs: list[str] = []
    for index, obligation in enumerate(obligations):
        item = _mapping(obligation, f"rung_obligations[{index}]")
        rung = item["rung"]
        if rung in declared_rungs:
            raise ScoreRankingValidationError(
                "duplicate_identity", f"duplicate rung obligation {rung}"
            )
        declared_rungs.append(rung)
        declared = frozenset(
            _strings(item["obligation_ids"], f"rung_obligations[{index}]")
        )
        expected = RUNG_OBLIGATIONS[rung]
        if declared != expected:
            missing = sorted(expected - declared)
            unexpected = sorted(declared - expected)
            raise ScoreRankingValidationError(
                "rung_obligations",
                f"{rung} obligation set mismatch "
                f"(missing={missing}, unexpected={unexpected})",
            )
    if tuple(declared_rungs) != expected_rungs:
        raise ScoreRankingValidationError(
            "rung_prefix",
            f"target {target_rung} requires exact rung prefix {expected_rungs}",
        )


def _validate_terminals(record: Mapping[str, Any]) -> None:
    claim = _mapping(record["claim"], "claim")
    target_rung = claim["target_rung"]
    target_index = RUNG_ORDER.index(target_rung)
    terminals = _sequence(record["terminals"], "terminals")
    seen_ids: set[str] = set()
    outcomes: set[str] = set()
    for index, terminal in enumerate(terminals):
        item = _mapping(terminal, f"terminals[{index}]")
        terminal_id = item["terminal_id"]
        if terminal_id in seen_ids:
            raise ScoreRankingValidationError(
                "duplicate_identity", f"duplicate terminal_id {terminal_id!r}"
            )
        seen_ids.add(terminal_id)
        outcome = item["outcome_class"]
        outcomes.add(outcome)
        transition = _mapping(
            item["state_transition"], f"terminals[{index}].state_transition"
        )
        if transition["kind"] == "transition":
            transition_index = RUNG_ORDER.index(transition["target_state"])
            if transition_index > target_index:
                raise ScoreRankingValidationError(
                    "terminal_transition",
                    f"terminal {terminal_id} transitions above target {target_rung}",
                )
        if outcome == "valid_positive":
            if (
                transition["kind"] != "transition"
                or transition["target_state"] != target_rung
            ):
                raise ScoreRankingValidationError(
                    "terminal_transition",
                    "valid_positive terminal must transition to the target rung",
                )
    if outcomes != OUTCOME_CLASSES:
        raise ScoreRankingValidationError(
            "terminal_partition",
            f"terminal outcome classes must equal {sorted(OUTCOME_CLASSES)}",
        )


def validate_declaration(record: object) -> dict[str, object]:
    """Validate one declaration without asserting owner or registry authority."""
    record_map = _mapping(record, "declaration")
    if record_map.get("schema") != SCHEMA_ID:
        raise ScoreRankingValidationError(
            "unsupported_schema",
            f"unsupported score-ranking schema {record_map.get('schema')!r}",
        )
    _schema_validate(record)
    _validate_contract_binding(record_map)
    _validate_policy(record_map)
    _validate_reductions(record_map)
    _validate_claim(record_map)
    _validate_rungs(record_map)
    _validate_terminals(record_map)

    binding = _mapping(record_map["contract_binding"], "contract_binding")
    return {
        "schema": SCHEMA_ID,
        "declaration_id": record_map["declaration_id"],
        "valid": True,
        "structurally_complete": True,
        "binding_fields_complete": binding["contract_status"] == "active",
        "authority_verified": False,
        "activation_eligible": False,
        "target_rung": _mapping(record_map["claim"], "claim")["target_rung"],
        "disposition": (
            "fixture"
            if record_map["record_scope"] == "fixture_only"
            else "declaration_candidate"
        ),
    }


def validate_declaration_file(path: Path) -> dict[str, object]:
    return validate_declaration(load_json(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declaration", type=Path, help="score declaration JSON")
    args = parser.parse_args()
    try:
        report = validate_declaration_file(args.declaration)
    except ScoreRankingValidationError as exc:
        parser.error(f"{exc.error_class}: {exc}")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
