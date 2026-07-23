"""GCTM D1 substrate-agnostic ranking diagnostic core.

Diagnostic-only package. Synthetic / sealed non-runtime inputs only.
Does not activate H0_ROUTE5_B1, GCTM_B1, GCTM_O1, or any runtime substrate.
"""

# status: experiment

from __future__ import annotations

DIAGNOSTIC_ID = "gctm_d1_substrate_agnostic_ranking_v1"
DIAGNOSTIC_SLOT_ID = "GCTM_D1"
GCTM_THEORY_IDENTITY = (
    "docs/research/models/gap_conditioned_stochastic_transition_spec_v1.md"
)
GCTM_LEMMAS_IDENTITY = (
    "docs/research/models/gap_conditioned_stochastic_transition_lemmas_v1.md"
)
SCORE_CONTRACT_IDENTITY = "score_ranking_evidence_contract_v1"

OBSERVATION_FAMILY = "position_innovation_residual_v1"
PARAMETERIZATION_FAMILY = "gctm_affine_m0_m1_m2_shared_interface_v1"
CANDIDATE_UNIVERSE = "synthetic_event_candidate_set_v1"
EVENT_KEY = "event_id"
SCORE_ORIENTATION = "lower_better"
SCORE_TRANSFORM = "identity_after_declared_score"
NORMALIZATION = "frozen_identity_no_free_scale"
TIE_RULE = "stable_cand_id_asc"

ORDERING_ACTIVE_MECHANISM = "anisotropic_shared_innovation_covariance"
CALIBRATION_ONLY_MECHANISM = "shared_isotropic_scalar_covariance"

ALLOWED_TERMINALS = (
    "GCTM_D1_DIAGNOSTIC_SEAL",
    "GCTM_D1_BOUNDED_NO_GO",
    "GCTM_D1_INTERFACE_READY",
)

__all__ = [
    "DIAGNOSTIC_ID",
    "DIAGNOSTIC_SLOT_ID",
    "OBSERVATION_FAMILY",
    "PARAMETERIZATION_FAMILY",
    "ORDERING_ACTIVE_MECHANISM",
    "ALLOWED_TERMINALS",
]
