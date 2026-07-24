#!/usr/bin/env python3
"""Fail-closed validator for the GCTM runtime-native candidate-universe contract.

Validates consumer-contract structure only. Never authorizes H0 capture,
registration, re-entry, runtime guarantees, compatibility, or B1 activation.
"""

# status: experiment

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts/tools"
if TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, TOOLS.as_posix())

import h0_declaration_frozen_identity as decl_id  # noqa: E402

SCHEMA_PATH = ROOT / "scripts/tools/gctm_runtime_universe_schema_v1.json"
PACKET_DIR = (
    ROOT
    / "docs/modules/semantic/research/evidence"
    / "gctm_runtime_native_candidate_universe_20260724"
)
FROZEN_INPUTS_PATH = PACKET_DIR / "frozen_input_identities.json"
REG_REQ_PATH = (
    PACKET_DIR / "h0_native_universe_completeness_registration_requirements_v1.json"
)
MANIFEST_PATH = PACKET_DIR / "manifest.json"
TERMINAL_REPORT_PATH = PACKET_DIR / "terminal_report.json"
DECLARATION_PATH = PACKET_DIR / "universe_declaration.json"

SCHEMA_ID = "gctm_runtime_native_candidate_universe_declaration_v1"
DECLARATION_ID = "gctm_runtime_native_candidate_universe_v1"
PACKET_ID = "gctm_runtime_native_candidate_universe_20260724"
MANIFEST_SCHEMA = "gctm_runtime_universe_manifest_v1"
TERMINAL_REPORT_SCHEMA = "gctm_runtime_universe_terminal_report_v1"

REQUIRED_PACKET_ARTIFACTS = (
    "universe_declaration.json",
    "frozen_input_identities.json",
    "event_candidate_identity.json",
    "inclusion_stage_decision.json",
    "composition_completeness_contract.json",
    "h0_native_universe_completeness_registration_requirements_v1.json",
    "terminal_report.json",
)
REQUIRED_TOOLING_KEYS = (
    "schema",
    "validator",
    "fixture_catalog",
    "targeted_tests",
)
# Maps terminal_report.artifact_bindings keys → (manifest_section, key).
TERMINAL_BINDING_MAP: dict[str, tuple[str, str]] = {
    "universe_declaration_sha256": ("artifacts", "universe_declaration.json"),
    "frozen_input_record_sha256": ("artifacts", "frozen_input_identities.json"),
    "event_candidate_identity_sha256": ("artifacts", "event_candidate_identity.json"),
    "inclusion_stage_decision_sha256": ("artifacts", "inclusion_stage_decision.json"),
    "composition_completeness_sha256": (
        "artifacts",
        "composition_completeness_contract.json",
    ),
    "registration_requirements_sha256": (
        "artifacts",
        "h0_native_universe_completeness_registration_requirements_v1.json",
    ),
    "schema_sha256": ("tooling", "schema"),
    "validator_sha256": ("tooling", "validator"),
    "fixture_catalog_sha256": ("tooling", "fixture_catalog"),
}

TERMINAL_INVALID = "GCTM_RUNTIME_UNIVERSE_AUDIT_INVALID"
TERMINAL_UNSEALABLE = "GCTM_RUNTIME_UNIVERSE_UNSEALABLE"
TERMINAL_SEALABLE = "GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE"
TERMINAL_ORDER = [TERMINAL_INVALID, TERMINAL_UNSEALABLE, TERMINAL_SEALABLE]

FIXED_NON_AUTHORITY = {
    "authority_verified": False,
    "runtime_guarantee_established": False,
    "runtime_compatibility_established": False,
    "h0_reentry_authorized": False,
    "b1_activation_eligible": False,
}

REQUIRED_EVENT_KEY_FIELDS = [
    "seq",
    "frame",
    "lost_slot",
    "lost_instance_uid",
    "event_key_version",
]
REQUIRED_CANDIDATE_KEY_FIELDS = ["event_key", "cand_slot", "cand_instance_uid"]
REQUIRED_DROP_REASONS = {
    "not_structural",
    "rejected_height",
    "rejected_speed",
    "rejected_spatial",
    "overflow_truncated",
    "partial_event",
    "duplicate_candidate_row",
    "cross_event_candidate_split",
    "missing_pair_row",
    "identity_uid_wrap",
}
REQUIRED_M_CONSERVATION = {
    "same event set",
    "same event key",
    "same candidate identity",
    "same C_e",
    "same duplicate handling",
    "same missing/overflow handling",
    "same label partition",
}
REQUIRED_LABELS = {
    "true_match_label",
    "GT",
    "FP",
    "GT_present",
    "GT_absent",
    "ambiguous",
    "fold",
    "blind",
    "reveal",
}
REQUIRED_COMPLETENESS_PROOFS = {
    "all native candidates satisfying inclusion semantics are represented",
    "no candidate row is silently dropped",
    "reported total reconciles with emitted rows",
    "overflow/truncation is explicit",
    "partial events fail closed",
    "each candidate belongs to exactly one event",
}
REQUIRED_REG_BINDINGS = {
    "universe_identity",
    "event_key_version",
    "inclusion_stage_identity",
    "mask_identity",
    "emitted_row_count",
    "native_candidate_count",
    "overflow_truncation_status",
    "exposure_completeness_counters",
    "replay_non_perturbation_basis",
    "causal_availability",
    "invalidation_set",
}
FORBIDDEN_INCLUSION_MARKERS = {
    "best_lost_slot",
    "second_lost_slot",
    "best_bdist",
    "second_best_bdist",
    "margin",
    "proposal_emitted",
    "claim_won",
    "commit_executed",
    "winning_cand_slot",
    "true_match_label",
}
SCORE_DEPENDENT_STAGES = {"final_eligible_set"}


class RuntimeUniverseValidationError(ValueError):
    """Declaration fails a fail-closed runtime-universe contract class."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeUniverseValidationError(
                "duplicate_json_key", f"duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise RuntimeUniverseValidationError(
        "non_finite_number", f"non-standard JSON numeric token {token!r}"
    )


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_non_finite,
        )
    except RuntimeUniverseValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeUniverseValidationError(
            "malformed_json", f"malformed JSON {path}: {exc}"
        ) from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeUniverseValidationError(
            "invalid_shape", f"{name} must be an object"
        )
    return value


def require_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeUniverseValidationError(
            "invalid_shape", f"{name} must be an array"
        )
    return value


def require_true(condition: bool, error_class: str, message: str) -> None:
    if not condition:
        raise RuntimeUniverseValidationError(error_class, message)


def require_false(condition: bool, error_class: str, message: str) -> None:
    if condition:
        raise RuntimeUniverseValidationError(error_class, message)


def validate_schema_shape(record: Mapping[str, Any]) -> None:
    schema = load_json(SCHEMA_PATH)
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeUniverseValidationError(
            "missing_dependency", "jsonschema is required"
        ) from exc
    try:
        jsonschema.Draft202012Validator(schema).validate(record)
    except jsonschema.ValidationError as exc:
        raise RuntimeUniverseValidationError(
            "schema_violation", f"schema violation: {exc.message}"
        ) from exc


def validate_frozen_inputs(record: Mapping[str, Any], frozen_path: Path) -> None:
    frozen = load_json(frozen_path)
    require_mapping(frozen, "frozen_input_identities")
    require_true(
        frozen.get("identity_policy") == "path_plus_sha256_only_no_mutable_branch_tip",
        "identity_policy",
        "frozen inputs must use path+sha256 identity only",
    )
    require_false(
        bool(frozen.get("mutable_branch_tip_used")),
        "mutable_branch_tip",
        "mutable branch tip identity is forbidden",
    )
    prereq = require_mapping(record.get("prerequisite_binding"), "prerequisite_binding")
    require_true(
        prereq.get("terminal") == "H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT",
        "prerequisite_terminal",
        "must bind H0_GCTM_INTERFACE_STRUCTURALLY_INSUFFICIENT",
    )
    require_true(
        prereq.get("owner_acceptance_id")
        == "h0_gctm_static_audit_terminal_owner_acceptance_20260723",
        "prerequisite_acceptance",
        "must bind h0_gctm_static_audit_terminal_owner_acceptance_20260723",
    )
    retained = set(require_sequence(prereq.get("retained_conclusions"), "retained"))
    require_true(
        "current ABI / registration-v2 is structurally insufficient" in retained,
        "retained_conclusions",
        "missing retained insufficient conclusion",
    )
    require_true(
        "repeat capture under unchanged interface is forbidden" in retained,
        "retained_conclusions",
        "missing retained no-repeat-capture conclusion",
    )

    inputs = require_sequence(frozen.get("inputs"), "frozen.inputs")
    require_true(len(inputs) >= 10, "frozen_inputs", "too few frozen inputs")
    for item in inputs:
        row = require_mapping(item, "frozen input row")
        path = ROOT / str(row["path"])
        require_true(path.is_file(), "frozen_path", f"missing frozen input {path}")
        expected = str(row["sha256"])
        try:
            disk_bytes = path.read_bytes()
        except OSError as exc:
            raise RuntimeUniverseValidationError(
                "frozen_path", f"cannot read frozen input {path}: {exc}"
            ) from exc
        actual = hashlib.sha256(disk_bytes).hexdigest()
        # H0 capture declaration may gain pure trailing SEALED owner-event rows
        # after package freeze (Amendment 10 Seal append).  All other inputs stay
        # strict path+sha256.
        if not decl_id.frozen_path_hash_ok(
            path=str(row["path"]),
            disk_bytes=disk_bytes,
            expected_sha256=expected,
        ):
            require_true(
                False,
                "frozen_hash_mismatch",
                f"hash mismatch for {row['path']}: expected {expected}, got {actual}",
            )


def validate_score_policy_spaces(record: Mapping[str, Any]) -> None:
    spaces = require_mapping(record.get("score_policy_spaces"), "score_policy_spaces")
    for key in ("U_src", "U_evt", "rho", "C_e"):
        member = require_mapping(spaces.get(key), key)
        require_true(bool(member), "score_policy_space", f"{key} must be non-empty")
        if key == "U_src":
            require_true(
                "space_id" in member and "source_stream" in member,
                "U_src_identity",
                "U_src must declare immutable space_id and source_stream",
            )
        if key == "U_evt":
            require_true(
                member.get("event_key_version"),
                "U_evt_identity",
                "U_evt must declare event_key_version",
            )
        if key == "rho":
            require_true(member.get("total") is True, "rho_total", "rho must be total")
            require_true(
                member.get("functional") is True,
                "rho_functional",
                "rho must be functional",
            )
            require_true(
                "exactly one" in str(member.get("identity", "")).lower()
                or "exactly one" in str(member.get("rule", "")).lower()
                or member.get("mapping_id"),
                "rho_unique",
                "rho must map each pair to exactly one event",
            )
        if key == "C_e":
            require_true(
                member.get("inclusion_stage_id"),
                "C_e_inclusion",
                "C_e must bind inclusion_stage_id",
            )


def validate_identities(record: Mapping[str, Any]) -> None:
    event = require_mapping(record.get("event_identity"), "event_identity")
    require_true(
        list(event.get("event_key_fields") or []) == REQUIRED_EVENT_KEY_FIELDS,
        "event_key_fields",
        f"event_key_fields must equal {REQUIRED_EVENT_KEY_FIELDS}",
    )
    require_true(
        event.get("consistent_across_candidate_pair_rows") is True,
        "event_key_consistency",
        "event key must be consistent across candidate pair rows",
    )
    require_true(
        event.get("represents_trajectory_identity") is False,
        "event_not_trajectory",
        "event identity must not represent trajectory identity",
    )
    require_true(
        event.get("represents_single_bridge_competition") is True,
        "event_single_competition",
        "event identity must represent a single bridge competition",
    )
    forbidden = set(require_sequence(event.get("must_not_be_inferred_from"), "infer"))
    for marker in ("claim winner", "commit result"):
        require_true(
            any(marker in item for item in forbidden),
            "event_inference_boundary",
            f"event key must forbid inference from {marker}",
        )

    cand = require_mapping(record.get("candidate_identity"), "candidate_identity")
    require_true(
        list(cand.get("candidate_key_fields") or []) == REQUIRED_CANDIDATE_KEY_FIELDS,
        "candidate_key_fields",
        f"candidate_key_fields must equal {REQUIRED_CANDIDATE_KEY_FIELDS}",
    )
    prevents = require_mapping(cand.get("prevents"), "prevents")
    for key in (
        "visible_track_id_as_stable_identity",
        "candidate_slot_reuse_without_instance_uid",
        "duplicate_candidate_rows_in_same_event",
        "same_candidate_assigned_to_multiple_events",
        "local_or_global_id_fallback",
    ):
        require_true(
            prevents.get(key) is True,
            "candidate_prevention",
            f"candidate_identity.prevents.{key} must be true",
        )


def validate_inclusion(record: Mapping[str, Any]) -> list[str]:
    """Return unsealable reasons if any."""
    unsealable: list[str] = []
    incl = require_mapping(record.get("inclusion_stage"), "inclusion_stage")
    require_true(
        incl.get("exactly_one_stage_selected") is True,
        "unique_inclusion_stage",
        "exactly one inclusion stage must be selected",
    )
    selected = incl.get("selected_stage")
    require_true(
        selected in set(require_sequence(incl.get("allowed_native_stages"), "stages")),
        "inclusion_stage_allowed",
        "selected inclusion stage must be one allowed native stage",
    )
    require_true(
        incl.get("circular_if_score_dependent") is True,
        "circular_rule",
        "score-dependent inclusion must be declared circular",
    )
    score_independent = incl.get("score_independent")
    require_true(
        isinstance(score_independent, bool),
        "score_independent_type",
        "score_independent must be boolean",
    )
    forbidden = set(
        require_sequence(incl.get("forbidden_inclusion_fields"), "forbidden fields")
    )
    require_true(
        FORBIDDEN_INCLUSION_MARKERS.issubset(forbidden),
        "forbidden_inclusion_fields",
        "missing claim/commit/winner/label fields from forbidden inclusion set",
    )
    if selected in SCORE_DEPENDENT_STAGES or score_independent is False:
        unsealable.append("inclusion stage is score-dependent or circular")
    if selected == "pre_score_eligible_set":
        proof = require_mapping(incl.get("writer_proof"), "writer_proof")
        require_true(
            "pre_score_gates" in proof and "score_and_later_gates" in proof,
            "writer_proof",
            "pre_score inclusion requires writer proof of gate order",
        )
    return unsealable


def validate_composition_and_completeness(record: Mapping[str, Any]) -> list[str]:
    unsealable: list[str] = []
    composition = require_mapping(record.get("composition"), "composition")
    drop_vocab = set(
        require_sequence(composition.get("drop_reason_vocabulary"), "drop_reason")
    )
    require_true(
        REQUIRED_DROP_REASONS.issubset(drop_vocab),
        "drop_reason_vocabulary",
        f"drop-reason vocabulary missing {sorted(REQUIRED_DROP_REASONS - drop_vocab)}",
    )
    require_true(
        composition.get("duplicate_rule") == "reject",
        "duplicate_rule",
        "duplicate_rule must be reject",
    )
    for key in (
        "empty_event_rule",
        "singleton_event_rule",
        "overflow_rule",
        "truncation_rule",
        "partial_event_rule",
        "pair_row_reconciliation_rule",
        "candidate_mask_identity",
        "gate_retained_band_identity",
        "candidate_count_source",
        "candidate_inclusion_source",
    ):
        require_true(
            bool(composition.get(key)),
            "composition_member",
            f"composition.{key} must be non-empty",
        )
    partial = str(composition.get("partial_event_rule", "")).lower()
    require_true(
        "fail" in partial
        and "closed" in partial.replace("-", "_").replace(" ", "_")
        or "fail_closed" in partial.replace(" ", "_"),
        "partial_event_rule",
        "partial events must fail closed",
    )

    completeness = require_mapping(
        record.get("completeness_requirements"), "completeness_requirements"
    )
    require_true(
        completeness.get("h0_completeness_guarantee_established") is False,
        "no_h0_completeness_claim",
        "must not claim an established H0 completeness guarantee",
    )
    proofs = set(
        require_sequence(
            completeness.get("future_producer_must_prove"), "completeness proofs"
        )
    )
    require_true(
        REQUIRED_COMPLETENESS_PROOFS.issubset(proofs),
        "completeness_requirements",
        f"missing completeness proofs {sorted(REQUIRED_COMPLETENESS_PROOFS - proofs)}",
    )
    if not completeness.get("required_counters"):
        unsealable.append("completeness counters missing")
    if (
        "silent" in str(composition.get("truncation_rule", "")).lower()
        and "forbidden" not in str(composition.get("truncation_rule", "")).lower()
    ):
        unsealable.append("truncation rule does not forbid silent truncation")
    return unsealable


def validate_labels_and_conservation(record: Mapping[str, Any]) -> None:
    labels = require_mapping(record.get("label_boundary"), "label_boundary")
    offline = set(require_sequence(labels.get("b1_offline_only"), "b1_offline_only"))
    require_true(
        REQUIRED_LABELS.issubset(offline),
        "label_boundary",
        f"missing B1_OFFLINE labels {sorted(REQUIRED_LABELS - offline)}",
    )
    require_true(
        labels.get("runtime_event_formation_depends_on_labels") is False,
        "labels_not_in_event",
        "runtime event formation must not depend on labels",
    )
    require_true(
        labels.get("runtime_candidate_inclusion_depends_on_labels") is False,
        "labels_not_in_inclusion",
        "runtime candidate inclusion must not depend on labels",
    )
    require_true(
        labels.get("inside_h0_guarantee_envelope") is False,
        "labels_outside_h0",
        "labels must remain outside H0 guarantee envelope",
    )
    require_true(
        bool(labels.get("attachment_rule_for_future_b1_evaluation")),
        "label_attachment_rule",
        "must predeclare offline label attachment to frozen runtime universe",
    )

    cons = require_mapping(record.get("m0_m1_m2_conservation"), "m0_m1_m2_conservation")
    invariants = set(require_sequence(cons.get("required_invariants"), "invariants"))
    require_true(
        REQUIRED_M_CONSERVATION.issubset(invariants),
        "m_conservation",
        f"missing M0/M1/M2 invariants {sorted(REQUIRED_M_CONSERVATION - invariants)}",
    )
    require_true(
        cons.get("m0_to_m1_may_change_candidate_universe") is False,
        "m0_m1_universe",
        "M0→M1 must not change candidate universe",
    )
    require_true(
        cons.get("m1_to_m2_may_change_candidate_universe") is False,
        "m1_m2_universe",
        "M1→M2 must not change candidate universe",
    )


def validate_claim_commit_boundary(record: Mapping[str, Any]) -> None:
    boundary = require_mapping(
        record.get("claim_commit_boundary"), "claim_commit_boundary"
    )
    for key in (
        "claim_participates_in_universe_formation",
        "commit_participates_in_universe_formation",
        "winner_fields_participate_in_universe_formation",
    ):
        require_true(
            boundary.get(key) is False,
            "claim_commit_boundary",
            f"{key} must be false",
        )


def validate_registration_requirements(
    record: Mapping[str, Any], reg_path: Path
) -> None:
    binding = require_mapping(
        record.get("registration_requirements_binding"),
        "registration_requirements_binding",
    )
    require_true(
        binding.get("modifies_registration_v2") is False,
        "no_reg_v2_mutation",
        "must not modify registration-v2",
    )
    require_true(
        binding.get("claims_registration_v3_exists") is False,
        "no_reg_v3_claim",
        "must not claim registration-v3 exists",
    )
    require_true(reg_path.is_file(), "registration_requirements", f"missing {reg_path}")
    req = load_json(reg_path)
    require_mapping(req, "registration requirements")
    require_true(
        req.get("schema")
        == "h0_native_universe_completeness_registration_requirements_v1",
        "registration_schema",
        "registration requirements schema mismatch",
    )
    require_true(
        req.get("guarantee_class") == "universe_completeness",
        "guarantee_class",
        "guarantee_class must be universe_completeness",
    )
    consumer_objects = set(
        require_sequence(req.get("consumer_objects"), "consumer_objects")
    )
    require_true(
        {"runtime_candidate_universe", "runtime_event_membership"}.issubset(
            consumer_objects
        ),
        "consumer_objects",
        "registration requirements missing runtime consumer objects",
    )
    streams = set(require_sequence(req.get("required_streams"), "required_streams"))
    require_true(
        {"pair_record", "candidate_record", "event_universe_sidecar"}.issubset(streams),
        "required_streams",
        "registration requirements missing required streams",
    )
    bindings = require_sequence(req.get("required_bindings"), "required_bindings")
    names = {require_mapping(item, "binding").get("binding") for item in bindings}
    require_true(
        REQUIRED_REG_BINDINGS.issubset(names),
        "required_bindings",
        f"missing registration bindings {sorted(REQUIRED_REG_BINDINGS - names)}",
    )
    require_true(
        req.get("modifies_registration_v2") is False,
        "reg_artifact_no_v2",
        "registration requirements artifact must not modify v2",
    )
    require_true(
        req.get("claims_registration_v3_exists") is False,
        "reg_artifact_no_v3",
        "registration requirements artifact must not claim v3 exists",
    )


def validate_fixed_outputs_and_terminal(
    record: Mapping[str, Any], unsealable_reasons: Sequence[str]
) -> str:
    fixed = require_mapping(
        record.get("fixed_validator_output"), "fixed_validator_output"
    )
    for key, expected in FIXED_NON_AUTHORITY.items():
        require_true(
            fixed.get(key) is expected,
            "fixed_non_authority",
            f"fixed_validator_output.{key} must be {expected}",
        )
    ordered = list(
        require_sequence(record.get("ordered_terminals"), "ordered_terminals")
    )
    require_true(
        ordered == TERMINAL_ORDER,
        "terminal_order",
        f"ordered_terminals must equal {TERMINAL_ORDER}",
    )
    seal = require_mapping(record.get("sealability"), "sealability")
    require_true(
        seal.get("complete_as_h0_guarantee") is False,
        "no_h0_guarantee",
        "must not claim H0 completeness guarantee",
    )
    require_true(
        seal.get("current_registration_v2_sufficient_to_bind_completeness") is False,
        "reg_v2_insufficient",
        "must retain registration-v2 insufficiency for completeness binding",
    )

    selected = record.get("selected_terminal")
    if unsealable_reasons:
        computed = TERMINAL_UNSEALABLE
    else:
        sealable = all(
            [
                seal.get("structurally_defined") is True,
                seal.get("score_independent") is True,
                seal.get("non_circular") is True,
                seal.get("event_local") is True,
                seal.get("candidate_conserving") is True,
                seal.get("complete_as_consumer_semantics") is True,
                seal.get("current_trace_v2_sufficient_to_define_consumer_universe")
                is True,
                record.get("inclusion_stage", {}).get("score_independent") is True,
            ]
        )
        computed = TERMINAL_SEALABLE if sealable else TERMINAL_UNSEALABLE

    require_true(
        selected == computed,
        "terminal_selection",
        f"selected_terminal {selected!r} does not match computed {computed!r}"
        + (f" reasons={list(unsealable_reasons)}" if unsealable_reasons else ""),
    )
    if computed == TERMINAL_SEALABLE:
        conclusion = str(record.get("maximum_supported_conclusion", ""))
        require_true(
            "structurally defined" in conclusion.lower()
            and "registration" in conclusion.lower(),
            "maximum_conclusion",
            "sealable terminal requires the structural consumer-target conclusion",
        )
        not_established = set(
            require_sequence(record.get("not_established"), "not_established")
        )
        for item in (
            "H0 completeness guarantee",
            "runtime compatibility",
            "runtime substrate",
            "B1 activation",
        ):
            require_true(
                item in not_established,
                "not_established",
                f"missing not_established item {item!r}",
            )
    return computed


def _require_fixed_non_authority(container: Mapping[str, Any], name: str) -> None:
    for key, expected in FIXED_NON_AUTHORITY.items():
        # Manifest stores the five fixed flags at top level; terminal_report
        # nests them under fixed_validator_output.
        if key in container:
            actual = container.get(key)
        else:
            nested = require_mapping(
                container.get("fixed_validator_output"),
                f"{name}.fixed_validator_output",
            )
            actual = nested.get(key)
        require_true(
            actual is expected,
            "fixed_non_authority",
            f"{name}.{key} must be {expected}",
        )


def validate_packet_bindings(packet_dir: Path = PACKET_DIR) -> dict[str, Any]:
    """Fail-closed integrity check for the exact on-disk consumer packet.

    Verifies manifest and terminal_report artifact/tooling SHA-256 bindings
    against disk, and cross-checks identity/terminal fields across declaration,
    manifest, and terminal report. Does not authorize H0/runtime/B1 authority.
    """
    packet_dir = packet_dir.resolve()
    manifest_path = packet_dir / "manifest.json"
    terminal_path = packet_dir / "terminal_report.json"
    declaration_path = packet_dir / "universe_declaration.json"
    for path in (manifest_path, terminal_path, declaration_path):
        require_true(path.is_file(), "packet_missing", f"missing packet file {path}")

    manifest = require_mapping(load_json(manifest_path), "manifest")
    terminal = require_mapping(load_json(terminal_path), "terminal_report")
    declaration = require_mapping(load_json(declaration_path), "universe_declaration")

    require_true(
        manifest.get("schema") == MANIFEST_SCHEMA,
        "manifest_schema",
        f"manifest schema must be {MANIFEST_SCHEMA}",
    )
    require_true(
        terminal.get("schema") == TERMINAL_REPORT_SCHEMA,
        "terminal_schema",
        f"terminal_report schema must be {TERMINAL_REPORT_SCHEMA}",
    )
    require_true(
        manifest.get("packet_id") == PACKET_ID,
        "packet_id",
        f"manifest.packet_id must be {PACKET_ID}",
    )
    require_true(
        terminal.get("packet_id") == PACKET_ID,
        "packet_id",
        f"terminal_report.packet_id must be {PACKET_ID}",
    )

    artifacts = require_mapping(manifest.get("artifacts"), "manifest.artifacts")
    for name in REQUIRED_PACKET_ARTIFACTS:
        require_true(
            name in artifacts,
            "manifest_artifacts",
            f"manifest.artifacts missing {name}",
        )
        path = packet_dir / name
        require_true(
            path.is_file(), "packet_missing", f"missing packet artifact {path}"
        )
        actual = sha256_file(path)
        expected = str(artifacts[name])
        require_true(
            actual == expected,
            "packet_artifact_hash_mismatch",
            f"manifest.artifacts[{name!r}] hash mismatch: expected {expected}, got {actual}",
        )

    tooling = require_mapping(manifest.get("tooling"), "manifest.tooling")
    for key in REQUIRED_TOOLING_KEYS:
        require_true(
            key in tooling, "manifest_tooling", f"manifest.tooling missing {key}"
        )
        entry = require_mapping(tooling.get(key), f"manifest.tooling.{key}")
        rel = str(entry.get("path", ""))
        path = ROOT / rel
        require_true(path.is_file(), "tooling_missing", f"missing tooling file {path}")
        actual = sha256_file(path)
        expected = str(entry.get("sha256", ""))
        require_true(
            actual == expected,
            "packet_tooling_hash_mismatch",
            f"manifest.tooling[{key!r}] hash mismatch for {rel}: expected {expected}, got {actual}",
        )

    bindings = require_mapping(
        terminal.get("artifact_bindings"), "terminal_report.artifact_bindings"
    )
    for binding_key, (section, section_key) in TERMINAL_BINDING_MAP.items():
        require_true(
            binding_key in bindings,
            "terminal_bindings",
            f"terminal_report.artifact_bindings missing {binding_key}",
        )
        if section == "artifacts":
            expected = str(artifacts[section_key])
        else:
            expected = str(
                require_mapping(tooling.get(section_key), f"tooling.{section_key}")[
                    "sha256"
                ]
            )
        actual = str(bindings[binding_key])
        require_true(
            actual == expected,
            "terminal_manifest_binding_mismatch",
            f"terminal_report.artifact_bindings[{binding_key!r}]={actual} "
            f"does not match manifest {section}[{section_key!r}]={expected}",
        )
        # Also re-check disk for terminal bindings (defends against manifest/tooling
        # agreement that both point at a stale value relative to disk).
        if section == "artifacts":
            disk = sha256_file(packet_dir / section_key)
        else:
            disk = sha256_file(ROOT / str(tooling[section_key]["path"]))
        require_true(
            actual == disk,
            "terminal_binding_disk_mismatch",
            f"terminal_report.artifact_bindings[{binding_key!r}]={actual} "
            f"does not match on-disk digest {disk}",
        )

    # Cross-identity: declaration ↔ manifest ↔ terminal_report.
    decl_id = declaration.get("declaration_id")
    decl_terminal = declaration.get("selected_terminal")
    require_true(
        decl_id == DECLARATION_ID,
        "declaration_id",
        f"declaration_id must be {DECLARATION_ID}",
    )
    require_true(
        manifest.get("runtime_consumer_identity") == decl_id,
        "identity_cross_check",
        "manifest.runtime_consumer_identity must equal declaration.declaration_id",
    )
    require_true(
        terminal.get("frozen_runtime_universe") == decl_id,
        "identity_cross_check",
        "terminal_report.frozen_runtime_universe must equal declaration.declaration_id",
    )
    require_true(
        manifest.get("selected_terminal") == decl_terminal,
        "terminal_cross_check",
        "manifest.selected_terminal must equal declaration.selected_terminal",
    )
    require_true(
        terminal.get("selected_terminal") == decl_terminal,
        "terminal_cross_check",
        "terminal_report.selected_terminal must equal declaration.selected_terminal",
    )
    ownership = require_mapping(declaration.get("ownership"), "ownership")
    require_true(
        ownership.get("runtime_consumer_identity") == decl_id,
        "identity_cross_check",
        "ownership.runtime_consumer_identity must equal declaration_id",
    )

    _require_fixed_non_authority(manifest, "manifest")
    _require_fixed_non_authority(terminal, "terminal_report")

    return {
        "valid": True,
        "packet_id": PACKET_ID,
        "declaration_id": decl_id,
        "selected_terminal": decl_terminal,
        "artifacts_checked": sorted(artifacts),
        "tooling_checked": sorted(tooling),
        "terminal_bindings_checked": sorted(TERMINAL_BINDING_MAP),
        "fixed_validator_output": dict(FIXED_NON_AUTHORITY),
    }


def validate_declaration(
    record: Mapping[str, Any],
    *,
    frozen_path: Path = FROZEN_INPUTS_PATH,
    reg_path: Path = REG_REQ_PATH,
    packet_dir: Path | None = None,
) -> dict[str, Any]:
    require_mapping(record, "declaration")
    validate_schema_shape(record)
    require_true(
        record.get("schema") == SCHEMA_ID,
        "schema_id",
        f"schema must be {SCHEMA_ID}",
    )
    require_true(
        record.get("declaration_id") == DECLARATION_ID,
        "declaration_id",
        f"declaration_id must be {DECLARATION_ID}",
    )
    ownership = require_mapping(record.get("ownership"), "ownership")
    require_true(
        ownership.get("runtime_consumer_identity") == DECLARATION_ID,
        "runtime_consumer_identity",
        "runtime consumer identity mismatch",
    )
    for flag in (
        "is_gctm_d1_candidate_universe",
        "is_h0_guarantee",
        "is_runtime_evidence_substrate",
        "is_b1_activation_declaration",
    ):
        require_true(
            ownership.get(flag) is False,
            "ownership_boundary",
            f"ownership.{flag} must be false",
        )

    validate_frozen_inputs(record, frozen_path)
    validate_score_policy_spaces(record)
    validate_identities(record)
    unsealable = validate_inclusion(record)
    unsealable.extend(validate_composition_and_completeness(record))
    validate_labels_and_conservation(record)
    validate_claim_commit_boundary(record)
    validate_registration_requirements(record, reg_path)
    terminal = validate_fixed_outputs_and_terminal(record, unsealable)

    result: dict[str, Any] = {
        "valid": True,
        "selected_terminal": terminal,
        "unsealable_reasons": list(unsealable),
        "fixed_validator_output": dict(FIXED_NON_AUTHORITY),
        "declaration_id": DECLARATION_ID,
    }
    if packet_dir is not None:
        packet = validate_packet_bindings(packet_dir)
        # When validating an in-memory declaration against a packet, the on-disk
        # declaration digest is already checked by packet bindings; additionally
        # require the in-memory selected_terminal to match the packet terminal.
        require_true(
            record.get("selected_terminal") == packet["selected_terminal"],
            "packet_declaration_terminal_mismatch",
            "in-memory declaration selected_terminal must match packet terminal",
        )
        result["packet_bindings"] = packet
    return result


def build_report(
    *,
    ok: bool,
    terminal: str,
    errors: Sequence[Mapping[str, str]] | None = None,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "validator": "validate_gctm_runtime_universe.py",
        "schema": SCHEMA_ID,
        "ok": ok,
        "selected_terminal": terminal,
        "errors": list(errors or []),
        "fixed_validator_output": dict(FIXED_NON_AUTHORITY),
        "authority_verified": False,
        "runtime_guarantee_established": False,
        "runtime_compatibility_established": False,
        "h0_reentry_authorized": False,
        "b1_activation_eligible": False,
    }
    if result is not None:
        report["result"] = dict(result)
    return report


def _resolve_packet_dir(
    declaration_path: Path, packet_dir_arg: Path | None
) -> Path | None:
    if packet_dir_arg is not None:
        return packet_dir_arg
    candidate = declaration_path.resolve().parent
    if (candidate / "manifest.json").is_file() and (
        candidate / "terminal_report.json"
    ).is_file():
        return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a GCTM runtime-native candidate-universe declaration and, "
            "when present, the exact on-disk packet bindings. "
            "Never authorizes H0/runtime/B1 authority."
        )
    )
    parser.add_argument(
        "declaration",
        type=Path,
        help="Path to the canonical runtime-universe declaration JSON",
    )
    parser.add_argument(
        "--frozen-inputs",
        type=Path,
        default=FROZEN_INPUTS_PATH,
        help="Path to frozen-input identity record",
    )
    parser.add_argument(
        "--registration-requirements",
        type=Path,
        default=REG_REQ_PATH,
        help="Path to registration-v3 requirements-only artifact",
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=None,
        help=(
            "Packet directory containing manifest.json and terminal_report.json. "
            "Defaults to the declaration's parent when those files are present."
        ),
    )
    parser.add_argument(
        "--skip-packet-bindings",
        action="store_true",
        help="Validate declaration structure only (does not accept the exact packet).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report on stdout",
    )
    args = parser.parse_args(argv)

    try:
        record = load_json(args.declaration)
        packet_dir = None
        if not args.skip_packet_bindings:
            packet_dir = _resolve_packet_dir(args.declaration, args.packet_dir)
            require_true(
                packet_dir is not None,
                "packet_dir_required",
                "canonical validation requires a packet directory with "
                "manifest.json and terminal_report.json; pass --packet-dir "
                "or --skip-packet-bindings",
            )
        result = validate_declaration(
            record,
            frozen_path=args.frozen_inputs,
            reg_path=args.registration_requirements,
            packet_dir=packet_dir,
        )
    except RuntimeUniverseValidationError as exc:
        report = build_report(
            ok=False,
            terminal=TERMINAL_INVALID,
            errors=[{"error_class": exc.error_class, "message": str(exc)}],
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"INVALID: {exc.error_class}: {exc}", file=sys.stderr)
            print(json.dumps(FIXED_NON_AUTHORITY, indent=2, sort_keys=True))
        return 2

    report = build_report(
        ok=True,
        terminal=str(result["selected_terminal"]),
        result=result,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"OK terminal={result['selected_terminal']}")
        if result.get("packet_bindings"):
            print("packet_bindings: verified")
        print(json.dumps(FIXED_NON_AUTHORITY, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
