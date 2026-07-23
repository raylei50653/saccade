"""Contract tests for the GCTM D1 ranking diagnostic interface package."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts" / "tools"
PACKET = (
    ROOT
    / "docs"
    / "modules"
    / "semantic"
    / "research"
    / "evidence"
    / "gctm_d1_substrate_agnostic_ranking_20260723"
)
sys.path.insert(0, TOOLS.as_posix())

from gctm_d1 import (  # noqa: E402
    ALLOWED_TERMINALS,
    DIAGNOSTIC_ID,
    ORDERING_ACTIVE_MECHANISM,
)
from gctm_d1.consumer_interface import (  # noqa: E402
    build_compatibility_matrix,
    build_consumer_interface,
    interface_is_complete,
)
from gctm_d1.fixtures import build_fixture_pack, pack_to_candidates  # noqa: E402
from gctm_d1.invariants import (  # noqa: E402
    protected_stratum_guard,
    run_all_invariants,
)
from gctm_d1.models import (  # noqa: E402
    CandidateObservation,
    FailClosedError,
    mahalanobis_q,
    ordering_tuple,
    resolve_covariance,
    score_candidate,
)
from gctm_d1.runner import (  # noqa: E402
    CANONICAL_ARTIFACTS,
    emit_packet,
    select_terminal,
)


REQUIRED_INTERFACE_KEYS = {
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


def test_all_twelve_invariants_pass_on_synthetic_pack() -> None:
    report = run_all_invariants(build_fixture_pack())
    assert report["all_passed"] is True
    assert report["n_invariants"] == 12
    assert report["n_passed"] == 12
    for result in report["results"]:
        assert result["passed"], result


def test_ranking_active_vs_calibration_only_distinguished() -> None:
    report = run_all_invariants(build_fixture_pack())
    mech = report["ranking_active_mechanism_test"]
    assert mech["ordering_active_mechanism"] == ORDERING_ACTIVE_MECHANISM
    assert mech["admissible_ranking_active"]["anisotropic_shared_innovation_covariance"]
    assert mech["admissible_ranking_active"][
        "candidate_specific_observation_covariance"
    ]
    assert mech["calibration_only"]["shared_isotropic_scalar_covariance"]
    assert "pooled_AUC" in mech["rejected_as_ranking_evidence"]


def test_candidate_specific_ranking_active_is_not_hardcoded_true() -> None:
    report = run_all_invariants(build_fixture_pack())
    mech = report["ranking_active_mechanism_test"]
    evidence = mech["admissible_ranking_active"]["evidence"][
        "candidate_specific_q_vs_nll"
    ]
    assert evidence["q_order"] != evidence["nll_order"]
    # Mutation: equalize S so q and NLL share order → must report False.
    pack = build_fixture_pack()
    for event in pack["events"]:
        if event["event_id"] != "E_cand_spec":
            continue
        for cand in event["candidates"]:
            cand["candidate_S"] = [[1.0, 0.0], [0.0, 1.0]]
            cand["residual"] = [1.0, 0.0] if cand["cand_id"] == "c_a" else [2.0, 0.0]
    mutated = run_all_invariants(pack)
    assert (
        mutated["ranking_active_mechanism_test"]["admissible_ranking_active"][
            "candidate_specific_observation_covariance"
        ]
        is False
    )


def test_anisotropic_shared_reorders_euclidean_baseline() -> None:
    pack = build_fixture_pack()
    cands = [c for c in pack_to_candidates(pack) if c.event_id == "E_shared_aniso"]
    m0 = [
        score_candidate(c, "M0", cov_mode="anisotropic_shared", rank_score="euclid")
        for c in cands
    ]
    m1 = [
        score_candidate(c, "M1", cov_mode="anisotropic_shared", rank_score="q")
        for c in cands
    ]
    assert ordering_tuple(m0) != ordering_tuple(m1)
    assert ordering_tuple(m1)[0] == "c_true"


def test_shared_isotropic_is_calibration_only() -> None:
    pack = build_fixture_pack()
    cands = [c for c in pack_to_candidates(pack) if c.event_id == "E_shared_iso"]
    m0 = [
        score_candidate(c, "M0", cov_mode="isotropic_shared", rank_score="euclid")
        for c in cands
    ]
    m1 = [
        score_candidate(c, "M1", cov_mode="isotropic_shared", rank_score="q")
        for c in cands
    ]
    assert ordering_tuple(m0) == ordering_tuple(m1)


def test_q_and_nll_identical_under_shared_anisotropic() -> None:
    pack = build_fixture_pack()
    cands = [c for c in pack_to_candidates(pack) if c.event_id == "E_shared_aniso"]
    by_q = [
        score_candidate(c, "M1", cov_mode="anisotropic_shared", rank_score="q")
        for c in cands
    ]
    by_nll = [
        score_candidate(c, "M1", cov_mode="anisotropic_shared", rank_score="nll")
        for c in cands
    ]
    assert ordering_tuple(by_q) == ordering_tuple(by_nll)


def test_fail_closed_on_singular_covariance() -> None:
    with pytest.raises(FailClosedError) as raised:
        mahalanobis_q(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert raised.value.code == "singular_covariance"


def test_candidate_specific_missing_cov_fails_closed_no_shared_fallback() -> None:
    cand = CandidateObservation(
        event_id="e",
        cand_id="c",
        residual=np.array([1.0, 0.0]),
        delta=1.0,
        is_true_match=True,
        stratum="short_gap",
        cov_shared=np.eye(2),
        cov_candidate=None,
    )
    with pytest.raises(FailClosedError) as raised:
        resolve_covariance(cand, "candidate_specific")
    assert raised.value.code == "missing_candidate_covariance"


def test_i9_rejects_protected_stratum_regression_hiding() -> None:
    # Aggregate top-1 improves while short_gap true rank regresses → must fail.
    result = protected_stratum_guard(
        strata_true_ranks={
            "short_gap": {"baseline_true_rank": 1, "challenger_true_rank": 2},
            "long_gap": {"baseline_true_rank": 2, "challenger_true_rank": 1},
        },
        protected_strata=["short_gap"],
        aggregate_top1_baseline=1,
        aggregate_top1_challenger=2,
    )
    assert result["passed"] is False
    assert result["would_hide_under_aggregate"] is True
    assert "short_gap" in result["protected_regressions"]


def test_i9_rejects_protected_regression_even_without_aggregate_gain() -> None:
    result = protected_stratum_guard(
        strata_true_ranks={
            "short_gap": {"baseline_true_rank": 1, "challenger_true_rank": 2},
        },
        protected_strata=["short_gap"],
        aggregate_top1_baseline=1,
        aggregate_top1_challenger=0,
    )
    assert result["passed"] is False


def test_consumer_interface_has_required_keys_and_field_semantics() -> None:
    iface = build_consumer_interface()
    missing = REQUIRED_INTERFACE_KEYS - set(iface)
    assert not missing, missing
    assert interface_is_complete(iface) is True
    assert iface["ordering_active_mechanism"] == ORDERING_ACTIVE_MECHANISM
    assert iface["not_an_h0_compatibility_verdict"] is True
    assert iface["score_orientation"] == "lower_better"
    fields = iface["required_runtime_fields"]
    assert len(fields) >= 8
    for field in fields:
        for key in (
            "semantic_meaning",
            "units",
            "shape",
            "event_shared_or_candidate_specific",
            "available_when",
            "future_info_leak",
            "consumed_by_invariant",
            "absence_selects_reject_runtime_consumption",
        ):
            assert key in field, (field.get("name"), key)


def test_compatibility_matrix_is_requirements_only() -> None:
    matrix = build_compatibility_matrix()
    assert matrix["status"] == "requirements_only_no_verdict"
    assert matrix["fail_closed_default"] == "reject_runtime_consumption"
    assert {g["status"] for g in matrix["gates"]} == {"missing"}
    assert "runtime_substrate" in matrix["explicitly_not_satisfied_by_this_diagnostic"]


def test_select_terminal_three_way_paths() -> None:
    report = run_all_invariants(build_fixture_pack())
    mech = report["ranking_active_mechanism_test"]

    assert (
        select_terminal(report, mech, interface_complete=True)
        == "GCTM_D1_INTERFACE_READY"
    )
    assert (
        select_terminal(report, mech, interface_complete=False)
        == "GCTM_D1_DIAGNOSTIC_SEAL"
    )

    failed = deepcopy(report)
    failed["all_passed"] = False
    assert (
        select_terminal(failed, mech, interface_complete=True)
        == "GCTM_D1_BOUNDED_NO_GO"
    )

    weak_mech = deepcopy(mech)
    weak_mech["admissible_ranking_active"][
        "anisotropic_shared_innovation_covariance"
    ] = False
    assert (
        select_terminal(report, weak_mech, interface_complete=True)
        == "GCTM_D1_BOUNDED_NO_GO"
    )

    for terminal in (
        "GCTM_D1_INTERFACE_READY",
        "GCTM_D1_DIAGNOSTIC_SEAL",
        "GCTM_D1_BOUNDED_NO_GO",
    ):
        assert terminal in ALLOWED_TERMINALS


def test_sealed_packet_is_bit_identical_to_fresh_emit(tmp_path: Path) -> None:
    result = emit_packet(tmp_path)
    assert result["selected_terminal"] == "GCTM_D1_INTERFACE_READY"
    assert result["all_invariants_passed"] is True
    assert result["interface_complete"] is True
    assert result["manifest"]["status"] == "SEAL_CANDIDATE_GENERATED"
    assert result["manifest"]["not_charter_execution"] is True
    assert result["manifest"]["created_at_utc"] == "2026-07-23T00:00:00Z"

    for name in CANONICAL_ARTIFACTS:
        sealed_path = PACKET / name
        fresh_path = tmp_path / name
        assert sealed_path.exists(), name
        assert fresh_path.exists(), name
        assert sealed_path.read_bytes() == fresh_path.read_bytes(), (
            f"bit mismatch for {name}"
        )

    fixture = PACKET / "fixture_pack.json"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    recorded = (PACKET / "fixture_pack.json.sha256").read_text(encoding="utf-8").strip()
    assert digest == recorded

    sealed = json.loads((PACKET / "terminal_report.json").read_text(encoding="utf-8"))
    assert sealed["record_scope"] == "diagnostic_seal_candidate"
    assert sealed["registry_effect"]["canonical_registry_state_transition"] is False
    assert sealed["registry_effect"]["records_seal_candidate_only"] is True
    assert sealed["diagnostic_id"] == DIAGNOSTIC_ID


def test_no_h0_or_runtime_activation_language_in_terminal() -> None:
    sealed = json.loads((PACKET / "terminal_report.json").read_text(encoding="utf-8"))
    blocked = " ".join(sealed["blocked_claims"]).lower()
    assert "runtime fidelity" in blocked
    assert "h0" in blocked
    assert sealed["canonical_conclusion"]["selected_terminal"] == (
        "GCTM_D1_INTERFACE_READY"
    )
    assert "seal_candidate" in sealed["canonical_conclusion"]["authority"]
