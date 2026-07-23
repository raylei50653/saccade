"""Machine-readable future runtime consumer interface for GCTM D1.

Interface-ready only. Not an H0 compatibility verdict.
"""

# status: experiment

from __future__ import annotations

from typing import Any

from . import (
    CANDIDATE_UNIVERSE,
    DIAGNOSTIC_ID,
    DIAGNOSTIC_SLOT_ID,
    EVENT_KEY,
    GCTM_THEORY_IDENTITY,
    NORMALIZATION,
    OBSERVATION_FAMILY,
    ORDERING_ACTIVE_MECHANISM,
    PARAMETERIZATION_FAMILY,
    SCORE_ORIENTATION,
    SCORE_TRANSFORM,
    TIE_RULE,
)


def _field(
    *,
    name: str,
    semantic_meaning: str,
    units: str,
    shape: str,
    event_shared_or_candidate_specific: str,
    available_when: str,
    future_info_leak: bool,
    consumed_by_invariant: str,
    absence_selects_reject_runtime_consumption: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "semantic_meaning": semantic_meaning,
        "units": units,
        "shape": shape,
        "event_shared_or_candidate_specific": event_shared_or_candidate_specific,
        "available_when": available_when,
        "future_info_leak": future_info_leak,
        "consumed_by_invariant": consumed_by_invariant,
        "absence_selects_reject_runtime_consumption": (
            absence_selects_reject_runtime_consumption
        ),
    }


REQUIRED_INTERFACE_TOP_LEVEL_KEYS = frozenset(
    {
        "consumer_slot_id",
        "gctm_theory_identity",
        "observation_family",
        "parameterization_family",
        "coordinate_semantics",
        "required_runtime_fields",
        "field_shapes",
        "field_units",
        "time_conversion",
        "causal_availability_by_field",
        "candidate_specific_fields",
        "event_shared_fields",
        "missing_value_rule",
        "context_definition",
        "context_fallback",
        "covariance_semantics",
        "score_transform",
        "score_orientation",
        "normalization",
        "tie_rule",
        "candidate_universe",
        "event_key",
        "ordering_active_mechanism",
        "identifiability_limits",
        "compatibility_checks",
        "reject_runtime_consumption_conditions",
    }
)

REQUIRED_FIELD_KEYS = frozenset(
    {
        "name",
        "semantic_meaning",
        "units",
        "shape",
        "event_shared_or_candidate_specific",
        "available_when",
        "future_info_leak",
        "consumed_by_invariant",
        "absence_selects_reject_runtime_consumption",
    }
)


def interface_is_complete(consumer: dict[str, Any]) -> bool:
    """True iff the consumer interface carries all INTERFACE_READY requirements."""
    if not REQUIRED_INTERFACE_TOP_LEVEL_KEYS.issubset(consumer.keys()):
        return False
    fields = consumer.get("required_runtime_fields")
    if not isinstance(fields, list) or len(fields) < 8:
        return False
    for field in fields:
        if not isinstance(field, dict):
            return False
        if not REQUIRED_FIELD_KEYS.issubset(field.keys()):
            return False
    if not consumer.get("ordering_active_mechanism"):
        return False
    if not consumer.get("reject_runtime_consumption_conditions"):
        return False
    if not consumer.get("identifiability_limits"):
        return False
    if consumer.get("not_an_h0_compatibility_verdict") is not True:
        return False
    return True


def build_consumer_interface() -> dict[str, Any]:
    required_runtime_fields = [
        _field(
            name="event_id",
            semantic_meaning="Stable event key grouping one lost exit with its candidate set",
            units="opaque string id",
            shape="scalar",
            event_shared_or_candidate_specific="event_shared",
            available_when="at event formation / exit endpoint",
            future_info_leak=False,
            consumed_by_invariant="I1,I2,I3",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="cand_id",
            semantic_meaning="Stable candidate identity inside an event",
            units="opaque string id",
            shape="scalar",
            event_shared_or_candidate_specific="candidate_specific",
            available_when="at candidate entry endpoint",
            future_info_leak=False,
            consumed_by_invariant="I1,I2,I3,tie_rule",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="g_phys",
            semantic_meaning="Physical exit→entry gap in frame intervals (canonical Δ)",
            units="frames",
            shape="scalar >= 1",
            event_shared_or_candidate_specific="candidate_specific",
            available_when="at candidate entry endpoint (exit already known)",
            future_info_leak=False,
            consumed_by_invariant="I6 shared gap condition; M1/M2 parameterization",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="residual_position",
            semantic_meaning=(
                "Position innovation r = y1 - H m^-_Δ under declared mean model; "
                "Hx observation mode only"
            ),
            units="coordinate units of declared substrate",
            shape="(d,)",
            event_shared_or_candidate_specific="candidate_specific",
            available_when="at entry endpoint after mean prediction from exit-causal state",
            future_info_leak=False,
            consumed_by_invariant="I5,I6,I7,I12 ranking scores",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="S_innovation",
            semantic_meaning=(
                "Total innovation covariance S_Δ used for Mahalanobis/NLL. "
                "Must declare whether event-shared or candidate-specific and the source"
            ),
            units="coordinate_units^2",
            shape="(d,d) SPD",
            event_shared_or_candidate_specific="either; must be declared",
            available_when=(
                "no later than entry endpoint; parameters used to form S must be "
                "exit-causal or entry-causal as declared"
            ),
            future_info_leak=False,
            consumed_by_invariant="I5,I6,I8,I10,I12",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="covariance_semantics",
            semantic_meaning=(
                "Enum: isotropic_shared | anisotropic_shared | candidate_specific; "
                "source and causal availability of each component"
            ),
            units="enum + declaration text",
            shape="scalar enum + provenance object",
            event_shared_or_candidate_specific="event_shared declaration",
            available_when="declaration time before scoring",
            future_info_leak=False,
            consumed_by_invariant="ranking_active_mechanism_test; I5; I6",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="coordinate_dim_d",
            semantic_meaning="Coordinate dimension d for residual and covariance",
            units="dimensionless",
            shape="scalar integer >= 1",
            event_shared_or_candidate_specific="event_shared",
            available_when="declaration time",
            future_info_leak=False,
            consumed_by_invariant="I6 dimension match; I10 shape checks",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="observation_mode",
            semantic_meaning="Must be H_x for this interface family (position-only)",
            units="enum",
            shape="scalar",
            event_shared_or_candidate_specific="event_shared",
            available_when="declaration time",
            future_info_leak=False,
            consumed_by_invariant="I11 Hx non-identifiability boundary",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="stratum_id",
            semantic_meaning="Protected-stratum label (must include short_gap partition)",
            units="enum string",
            shape="scalar",
            event_shared_or_candidate_specific="event_shared",
            available_when="at event formation from g_phys",
            future_info_leak=False,
            consumed_by_invariant="I9",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="context_drift_position",
            semantic_meaning=(
                "Optional M2 only: H d_Δ(c) position correction; exit-causal context only"
            ),
            units="coordinate units",
            shape="(d,) or null when M2 inactive",
            event_shared_or_candidate_specific="candidate_specific values; mapping exit-shared",
            available_when="must be computable from exit-causal context only",
            future_info_leak=True,  # if sourced from entry/future frames
            consumed_by_invariant="I2 M2 residual; I12 CEX_m2",
            absence_selects_reject_runtime_consumption=False,  # only required if M2 claimed
        ),
        _field(
            name="score_orientation",
            semantic_meaning="Frozen lower_better for q and NLL native scores",
            units="enum",
            shape="scalar",
            event_shared_or_candidate_specific="event_shared",
            available_when="declaration time",
            future_info_leak=False,
            consumed_by_invariant="I7",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="tie_rule",
            semantic_meaning="Frozen stable_cand_id_asc",
            units="enum",
            shape="scalar",
            event_shared_or_candidate_specific="event_shared",
            available_when="declaration time",
            future_info_leak=False,
            consumed_by_invariant="I10",
            absence_selects_reject_runtime_consumption=True,
        ),
        _field(
            name="true_match_label",
            semantic_meaning=(
                "Diagnostic-only label for RANK metrics on synthetic/sealed non-runtime "
                "inputs; not a runtime producer field for online scoring"
            ),
            units="boolean",
            shape="scalar",
            event_shared_or_candidate_specific="candidate_specific",
            available_when="offline diagnostic labels only",
            future_info_leak=False,
            consumed_by_invariant="I9 diagnostic ranks",
            absence_selects_reject_runtime_consumption=False,
        ),
    ]

    return {
        "consumer_slot_id": DIAGNOSTIC_SLOT_ID,
        "interface_id": f"{DIAGNOSTIC_ID}_consumer_interface",
        "interface_status": "interface_ready_only",
        "not_an_h0_compatibility_verdict": True,
        "gctm_theory_identity": GCTM_THEORY_IDENTITY,
        "observation_family": OBSERVATION_FAMILY,
        "parameterization_family": PARAMETERIZATION_FAMILY,
        "coordinate_semantics": {
            "space": "R^{2d} latent state; observed residual is position block R^d",
            "observation_mode": "H_x",
            "production_declared_target": "S_A height-normalized image-plane position",
            "binding_class": "declared-target; not runtime-fidelity",
        },
        "required_runtime_fields": required_runtime_fields,
        "field_shapes": {f["name"]: f["shape"] for f in required_runtime_fields},
        "field_units": {f["name"]: f["units"] for f in required_runtime_fields},
        "time_conversion": {
            "frame_time_unit": "1 frame interval",
            "canonical_delta": "g_phys frames (exit endpoint → entry endpoint)",
            "continuous_dt": "dt ≡ 1 frame; wall-clock/fps mapping must be separately declared",
            "production_horizon_la_not_in_canonical_kernel": True,
        },
        "causal_availability_by_field": {
            f["name"]: {
                "available_when": f["available_when"],
                "future_info_leak": f["future_info_leak"],
            }
            for f in required_runtime_fields
        },
        "candidate_specific_fields": [
            f["name"]
            for f in required_runtime_fields
            if "candidate" in f["event_shared_or_candidate_specific"]
        ],
        "event_shared_fields": [
            f["name"]
            for f in required_runtime_fields
            if f["event_shared_or_candidate_specific"].startswith("event_shared")
        ],
        "missing_value_rule": "fail_closed_reject_record_no_imputation",
        "context_definition": (
            "Exit-causal context c only; fixed on [0, Δ]; no entry/future frames"
        ),
        "context_fallback": "if context unavailable, M2 inactive; M0/M1 still legal with d_Δ=0",
        "covariance_semantics": {
            "S_delta": "HP^-H^T + R1 under canonical C=0",
            "shared_isotropic_scalar": "calibration_only_not_ranking_active",
            "shared_anisotropic": "ranking_active_vs_euclidean",
            "candidate_specific": (
                "ranking_active_only_when_source_and_causal_availability_declared"
            ),
            "singular_or_missing": "reject_runtime_consumption",
        },
        "score_transform": SCORE_TRANSFORM,
        "score_orientation": SCORE_ORIENTATION,
        "normalization": NORMALIZATION,
        "tie_rule": TIE_RULE,
        "candidate_universe": CANDIDATE_UNIVERSE,
        "event_key": EVENT_KEY,
        "ordering_active_mechanism": ORDERING_ACTIVE_MECHANISM,
        "identifiability_limits": [
            "P_xx vs R_1 split structurally non-identifiable without gauge_fixing",
            "asym(P_xv) invisible under H_x",
            "gamma unknown requires unproven joint-map regime; D1 treats gamma as declared",
            "CAL scale alpha_Delta not identifiable from RANK order",
            "single Hx event cannot identify full {P0,R1,D} quotient",
        ],
        "compatibility_checks": [
            "canonical_h0_evidence_manifest",
            "stable_evidence_identity",
            "checksum_verification",
            "producer_consumer_schema_compatibility",
            "observation_semantics_compatibility",
            "parameterization_compatibility",
            "score_transformation_normalization_compatibility",
            "ordering_preservation_verdict",
        ],
        "reject_runtime_consumption_conditions": [
            "missing_required_runtime_field",
            "singular_or_non_psd_S_innovation",
            "undeclared_candidate_specific_covariance_source",
            "future_leaking_context_drift",
            "observation_mode_not_H_x_for_this_family",
            "score_orientation_or_tie_rule_mismatch",
            "normalization_not_frozen_for_scale_comparison",
            "h0_compatibility_verdict_missing_or_rejected",
            "attempt_to_use_diagnostic_evidence_as_runtime_substrate",
        ],
        "models": {
            "M0": "frozen deterministic Euclidean residual energy; same residual mean",
            "M1": "gap-conditioned Mahalanobis/NLL under same mean residual interface",
            "M2": (
                "optional leakage-free context drift mean correction; "
                "observation interface, candidate universe, missing-value rule, "
                "normalization, score composition, fitting semantics held fixed"
            ),
        },
        "m1_to_m2_forbidden_changes": [
            "observation_interface",
            "candidate_universe",
            "missing_value_rule",
            "normalization",
            "score_composition",
            "fitting_semantics",
        ],
    }


def build_compatibility_matrix() -> dict[str, Any]:
    checks = [
        "canonical_h0_evidence_manifest",
        "stable_evidence_identity",
        "checksum_verification",
        "producer_consumer_schema_compatibility",
        "observation_semantics_compatibility",
        "parameterization_compatibility",
        "score_transformation_normalization_compatibility",
        "ordering_preservation_verdict",
    ]
    return {
        "matrix_id": f"{DIAGNOSTIC_ID}_compatibility_requirements_v1",
        "producer_slot_id": DIAGNOSTIC_SLOT_ID,
        "consumer_slot_ids": ["H0_ROUTE5_B1", "GCTM_B1"],
        "status": "requirements_only_no_verdict",
        "fail_closed_default": "reject_runtime_consumption",
        "gates": [
            {
                "gate_id": "gctm_d1_to_h0_route5_b1_compatibility_v1",
                "consumer_slot_id": "H0_ROUTE5_B1",
                "status": "missing",
                "required_checks": checks,
            },
            {
                "gate_id": "gctm_d1_to_gctm_b1_compatibility_v1",
                "consumer_slot_id": "GCTM_B1",
                "status": "missing",
                "required_checks": checks,
            },
        ],
        "check_definitions": {
            "canonical_h0_evidence_manifest": (
                "Producer supplies owner-accepted H0 evidence manifest identity"
            ),
            "stable_evidence_identity": "Stable evidence id + schema version",
            "checksum_verification": "Canonical checksum verifies bit-identical payload",
            "producer_consumer_schema_compatibility": (
                "Consumer required fields ⊆ producer schema with type/unit match"
            ),
            "observation_semantics_compatibility": (
                "Observation family, H mode, residual definition, time index match"
            ),
            "parameterization_compatibility": (
                "M0/M1/M2 parameterization family and covariance semantics match"
            ),
            "score_transformation_normalization_compatibility": (
                "Orientation, transform, normalization, tie rule identical"
            ),
            "ordering_preservation_verdict": (
                "Separately owner-accepted verdict that consumer ranking uses the "
                "same within-event order semantics"
            ),
        },
        "explicitly_not_satisfied_by_this_diagnostic": [
            "runtime_substrate",
            "runtime_provenance",
            "runtime_evidence_identity",
            "runtime_checksum",
            "runtime_consumer_compatibility",
            "runtime_activation_authority",
        ],
    }
