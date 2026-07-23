"""M0/M1/M2 diagnostic runner and sealed packet emitter for GCTM D1."""

# status: experiment

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import (
    ALLOWED_TERMINALS,
    CANDIDATE_UNIVERSE,
    DIAGNOSTIC_ID,
    DIAGNOSTIC_SLOT_ID,
    EVENT_KEY,
    GCTM_LEMMAS_IDENTITY,
    GCTM_THEORY_IDENTITY,
    NORMALIZATION,
    OBSERVATION_FAMILY,
    ORDERING_ACTIVE_MECHANISM,
    PARAMETERIZATION_FAMILY,
    SCORE_CONTRACT_IDENTITY,
    SCORE_ORIENTATION,
    SCORE_TRANSFORM,
    TIE_RULE,
)
from .consumer_interface import (
    build_compatibility_matrix,
    build_consumer_interface,
    interface_is_complete,
)
from .fixtures import (
    FIXTURE_PACK_ID,
    canonical_json_bytes,
    pack_to_candidates,
    write_fixture_pack,
)
from .invariants import run_all_invariants
from .models import ordering_tuple, score_candidate


REPO = Path(__file__).resolve().parents[3]

# Fixed metadata for bit-identical seal-candidate regeneration.
SEALED_CREATED_AT_UTC = "2026-07-23T00:00:00Z"
CANONICAL_PACKET_REL = (
    "docs/modules/semantic/research/evidence/"
    "gctm_d1_substrate_agnostic_ranking_20260723"
)
CANONICAL_ARTIFACTS = (
    "fixture_pack.json",
    "declaration_sidecar.json",
    "invariant_report.json",
    "event_level_diagnostic_packet.json",
    "consumer_interface.json",
    "compatibility_requirements_matrix.json",
    "terminal_report.json",
    "identities.json",
    "manifest.json",
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(obj) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def event_cov_mode(event: dict[str, Any]) -> str:
    return str(event["cov_mode"])


def run_models(pack: dict[str, Any]) -> dict[str, Any]:
    cands = pack_to_candidates(pack)
    by_event: dict[str, list] = {}
    for c in cands:
        by_event.setdefault(c.event_id, []).append(c)

    event_packets: list[dict[str, Any]] = []
    for event in pack["events"]:
        eid = event["event_id"]
        group = by_event[eid]
        mode = event_cov_mode(event)
        model_block: dict[str, Any] = {}
        for model in ("M0", "M1", "M2"):
            rank_score = "euclid" if model == "M0" else "q"
            scored = [
                score_candidate(
                    c,
                    model,  # type: ignore[arg-type]
                    cov_mode=mode,  # type: ignore[arg-type]
                    rank_score=rank_score,  # type: ignore[arg-type]
                )
                for c in group
            ]
            order = ordering_tuple(scored)
            model_block[model] = {
                "ordering": list(order),
                "scores": [
                    {
                        "cand_id": s.cand_id,
                        "q": s.q,
                        "nll": s.nll,
                        "score_for_rank": s.score_for_rank,
                        "is_true_match": s.is_true_match,
                    }
                    for s in sorted(scored, key=lambda x: x.cand_id)
                ],
            }
            if model == "M1" and mode != "candidate_specific":
                scored_nll = [
                    score_candidate(
                        c,
                        "M1",
                        cov_mode=mode,  # type: ignore[arg-type]
                        rank_score="nll",
                    )
                    for c in group
                ]
                model_block[model]["nll_ordering"] = list(ordering_tuple(scored_nll))

        true_id = next(c["cand_id"] for c in event["candidates"] if c["is_true_match"])
        event_packets.append(
            {
                "event_id": eid,
                "stratum": event["stratum"],
                "delta": event["delta"],
                "cov_mode": mode,
                "true_match_cand_id": true_id,
                "candidate_ids": sorted(c.cand_id for c in group),
                "models": model_block,
                "notes": event.get("notes"),
            }
        )

    return {
        "schema": "gctm_d1_event_level_diagnostic_packet_v1",
        "diagnostic_id": DIAGNOSTIC_ID,
        "events": event_packets,
        "model_sequence": {
            "M0": "frozen deterministic baseline (Euclidean residual energy)",
            "M1": "gap-conditioned uncertainty under the same mean residual",
            "M2": "leakage-free context drift; all other interface fields fixed",
        },
        "m1_to_m2_held_fixed": [
            "observation_interface",
            "candidate_universe",
            "missing_value_rule",
            "normalization",
            "score_composition",
            "fitting_semantics",
        ],
    }


def ranking_active_requirements_met(mechanism: dict[str, Any]) -> bool:
    active = mechanism["admissible_ranking_active"]
    if not active.get("anisotropic_shared_innovation_covariance"):
        return False
    if not active.get("candidate_specific_observation_covariance"):
        return False
    if not mechanism["calibration_only"].get(
        "shared_isotropic_scalar_covariance", False
    ):
        return False
    return True


def select_terminal(
    invariant_report: dict[str, Any],
    mechanism: dict[str, Any],
    *,
    interface_complete: bool,
) -> str:
    """Three-way mechanical terminal selection.

    - invariants or ranking-active falsification fail → BOUNDED_NO_GO
    - invariants + ranking-active pass, interface incomplete → DIAGNOSTIC_SEAL
    - invariants + ranking-active + complete interface → INTERFACE_READY
    """
    if not invariant_report["all_passed"]:
        return "GCTM_D1_BOUNDED_NO_GO"
    if not ranking_active_requirements_met(mechanism):
        return "GCTM_D1_BOUNDED_NO_GO"
    if not interface_complete:
        return "GCTM_D1_DIAGNOSTIC_SEAL"
    return "GCTM_D1_INTERFACE_READY"


def build_declaration_sidecar(
    *,
    fixture_sha: str,
    theory_sha: str,
    lemmas_sha: str,
    contract_sha: str,
) -> dict[str, Any]:
    return {
        "schema": "gctm_d1_declaration_sidecar_v1",
        "diagnostic_id": DIAGNOSTIC_ID,
        "slot_id": DIAGNOSTIC_SLOT_ID,
        "authority_class": "diagnostic_only",
        "record_scope": "diagnostic_seal_candidate",
        "generation_kind": "pre_activation_synthetic_seal_candidate",
        "owner_acceptance_status": "pending_owner_review",
        "activation_status": "not_activated",
        "wip_status": "non_wip",
        "not_charter_execution": True,
        "not_canonical_registry_terminal_transition": True,
        "accepted_gctm_theory_identity": {
            "path": GCTM_THEORY_IDENTITY,
            "sha256": theory_sha,
        },
        "accepted_gctm_lemmas_identity": {
            "path": GCTM_LEMMAS_IDENTITY,
            "sha256": lemmas_sha,
        },
        "accepted_score_contract_identity": {
            "contract_id": SCORE_CONTRACT_IDENTITY,
            "sha256": contract_sha,
        },
        "input_substrate_class": "synthetic",
        "input_identity": FIXTURE_PACK_ID,
        "input_checksum": fixture_sha,
        "input_schema": "gctm_d1_fixture_pack_v1",
        "observation_family": OBSERVATION_FAMILY,
        "parameterization_family": PARAMETERIZATION_FAMILY,
        "candidate_universe": CANDIDATE_UNIVERSE,
        "event_key": EVENT_KEY,
        "score_orientation": SCORE_ORIENTATION,
        "score_transform": SCORE_TRANSFORM,
        "normalization": NORMALIZATION,
        "tie_rule": TIE_RULE,
        "ordering_active_mechanism": ORDERING_ACTIVE_MECHANISM,
        "invariants": [f"I{i}" for i in range(1, 13)],
        "counterexample_search_space": "synthetic_d2_controlled_residuals_and_S_structures",
        "identifiability_questions": [
            "P_xx_R1_split",
            "asym_Pxv_under_Hx",
            "gamma_unknown_regime",
            "cal_scale_from_rank",
            "single_event_quotient",
        ],
        "terminal_order": list(ALLOWED_TERMINALS),
        "prohibited_claims": [
            "runtime_faithful",
            "online_grounded",
            "h0_equivalent_capture",
            "h0_gctm_compatibility_completed",
            "activate_H0_ROUTE5_B1",
            "activate_GCTM_B1",
            "activate_GCTM_O1",
            "decision_relevant_candidate",
            "automatic_wip_acquisition",
        ],
        "models": ["M0", "M1", "M2"],
    }


def build_terminal_report(
    *,
    terminal: str,
    invariant_report: dict[str, Any],
    identities: dict[str, Any],
    mechanism: dict[str, Any],
) -> dict[str, Any]:
    assert terminal in ALLOWED_TERMINALS
    max_claims = []
    blocked_claims = [
        "runtime fidelity or H0 capture equivalence",
        "H0→GCTM consumer compatibility completed",
        "activation of H0_ROUTE5_B1 or GCTM_B1 or GCTM_O1",
        "decision-relevant registry candidate",
        "WIP acquisition",
        "production observation-mode freeze",
        "pooled-row independence ranking evidence",
        "repair or re-entry of historical H0 packets",
    ]
    if terminal == "GCTM_D1_INTERFACE_READY":
        max_claims = [
            "seal-candidate only: declared substrate-agnostic family is machine-checkable on synthetic fixtures",
            "calibration-only vs ranking-active mechanisms are mechanically distinguishable",
            "ordering-active mechanism precisely specified: anisotropic shared innovation covariance",
            "future runtime consumer interface fields/semantics complete for separate H0 verdict",
            "non-identifiable quantities explicitly listed and bounded by reject conditions",
            "all declared invariants pass on synthetic non-runtime fixtures",
        ]
    elif terminal == "GCTM_D1_DIAGNOSTIC_SEAL":
        max_claims = [
            "seal-candidate only: declared diagnostic object is internally sealed on synthetic substrate",
            "consumer interface incomplete or not yet validated for INTERFACE_READY",
        ]
    else:
        max_claims = [
            "bounded diagnostic no-go within declared scope; constructive failures retained"
        ]

    return {
        "schema": "gctm_d1_terminal_report_v1",
        "diagnostic_id": DIAGNOSTIC_ID,
        "slot_id": DIAGNOSTIC_SLOT_ID,
        "record_scope": "diagnostic_seal_candidate",
        "selected_terminal": terminal,
        "terminal_family": list(ALLOWED_TERMINALS),
        "terminal_status": "seal_candidate_pending_owner_acceptance",
        "invariants_all_passed": invariant_report["all_passed"],
        "n_invariants_passed": invariant_report["n_passed"],
        "n_invariants": invariant_report["n_invariants"],
        "ordering_active_mechanism": mechanism["ordering_active_mechanism"],
        "ranking_active_mechanism_test": mechanism,
        "maximum_claims": max_claims,
        "blocked_claims": blocked_claims,
        "identities": identities,
        "registry_effect": {
            "may_transition_slot_ids": [DIAGNOSTIC_SLOT_ID],
            "canonical_registry_state_transition": False,
            "records_seal_candidate_only": True,
            "must_not_alter": [
                "quantity.bridge_capture_provenance",
                "H0_ROUTE5_B1",
                "GCTM_B1",
                "GCTM_O1",
                "decision_relevant_candidate_set",
            ],
            "owner_acceptance_required_for_charter_activation": True,
            "scheduling_required_for_wip": True,
        },
        "canonical_conclusion": {
            "maximum_claims": max_claims,
            "blocked_claims": blocked_claims,
            "selected_terminal": terminal,
            "authority": "seal_candidate_not_owner_accepted_execution",
        },
    }


def emit_packet(out_dir: Path) -> dict[str, Any]:
    """Generate a deterministic seal-candidate packet (not charter execution)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = out_dir / "fixture_pack.json"
    fixture_meta = write_fixture_pack(fixture_path)
    pack = fixture_meta["pack"]
    fixture_sha = fixture_meta["sha256"]

    theory_path = REPO / GCTM_THEORY_IDENTITY
    lemmas_path = REPO / GCTM_LEMMAS_IDENTITY
    contract_path = REPO / "docs/research/contracts/score_ranking_evidence_contract.md"
    theory_sha = _file_sha256(theory_path)
    lemmas_sha = _file_sha256(lemmas_path)
    contract_sha = _file_sha256(contract_path)

    invariant_report = run_all_invariants(pack)
    mechanism = invariant_report["ranking_active_mechanism_test"]
    event_packet = run_models(pack)
    consumer = build_consumer_interface()
    interface_complete = interface_is_complete(consumer)
    compat = build_compatibility_matrix()
    declaration = build_declaration_sidecar(
        fixture_sha=fixture_sha,
        theory_sha=theory_sha,
        lemmas_sha=lemmas_sha,
        contract_sha=contract_sha,
    )
    terminal = select_terminal(
        invariant_report,
        mechanism,
        interface_complete=interface_complete,
    )

    identities = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "packet_id": "gctm_d1_substrate_agnostic_ranking_20260723",
        "canonical_packet_rel": CANONICAL_PACKET_REL,
        "fixture_pack_id": FIXTURE_PACK_ID,
        "fixture_sha256": fixture_sha,
        "gctm_theory_path": GCTM_THEORY_IDENTITY,
        "gctm_theory_sha256": theory_sha,
        "gctm_lemmas_path": GCTM_LEMMAS_IDENTITY,
        "gctm_lemmas_sha256": lemmas_sha,
        "score_contract_id": SCORE_CONTRACT_IDENTITY,
        "score_contract_sha256": contract_sha,
        "observation_family": OBSERVATION_FAMILY,
        "parameterization_family": PARAMETERIZATION_FAMILY,
        "runner": "scripts/tools/run_gctm_d1_diagnostic.py",
        "core_package": "scripts/tools/gctm_d1/",
        "record_scope": "diagnostic_seal_candidate",
    }

    terminal_report = build_terminal_report(
        terminal=terminal,
        invariant_report=invariant_report,
        identities=identities,
        mechanism=mechanism,
    )

    # Write non-manifest artifacts first. Manifest digests cover those files only
    # (no circular self-hash) so content is bit-identical across regenerations.
    artifact_bodies: dict[str, Any] = {
        "declaration_sidecar.json": declaration,
        "invariant_report.json": invariant_report,
        "event_level_diagnostic_packet.json": event_packet,
        "consumer_interface.json": consumer,
        "compatibility_requirements_matrix.json": compat,
        "terminal_report.json": terminal_report,
        "identities.json": identities,
    }
    artifact_digests: dict[str, str] = {"fixture_pack.json": fixture_sha}
    for name, body in artifact_bodies.items():
        artifact_digests[name] = _write_json(out_dir / name, body)

    manifest = {
        "schema": "gctm_d1_evidence_packet_manifest_v1",
        "diagnostic_id": DIAGNOSTIC_ID,
        "slot_id": DIAGNOSTIC_SLOT_ID,
        "created_at_utc": SEALED_CREATED_AT_UTC,
        "substrate_class": "synthetic",
        "non_runtime": True,
        "h0_forbidden": True,
        "record_scope": "diagnostic_seal_candidate",
        "selected_terminal": terminal,
        "interface_complete": interface_complete,
        "artifacts": artifact_digests,
        "identities": identities,
        "status": "SEAL_CANDIDATE_GENERATED",
        "not_charter_execution": True,
        "not_owner_accepted_terminal_transition": True,
    }
    manifest_sha = _write_json(out_dir / "manifest.json", manifest)

    return {
        "out_dir": str(out_dir),
        "selected_terminal": terminal,
        "all_invariants_passed": invariant_report["all_passed"],
        "interface_complete": interface_complete,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "artifact_digests": artifact_digests,
    }
