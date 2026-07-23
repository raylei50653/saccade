"""Contract tests for the GCTM D1 ranking diagnostic interface package."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

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
)
from gctm_d1.fixtures import build_fixture_pack, pack_to_candidates  # noqa: E402
from gctm_d1.invariants import run_all_invariants  # noqa: E402
from gctm_d1.models import (  # noqa: E402
    FailClosedError,
    mahalanobis_q,
    ordering_tuple,
    score_candidate,
)
from gctm_d1.runner import emit_packet, select_terminal  # noqa: E402


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
    assert mech["calibration_only"]["shared_isotropic_scalar_covariance"]
    assert "pooled_AUC" in mech["rejected_as_ranking_evidence"]


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
    import numpy as np

    with pytest.raises(FailClosedError) as raised:
        mahalanobis_q(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert raised.value.code == "singular_covariance"


def test_consumer_interface_has_required_keys_and_field_semantics() -> None:
    iface = build_consumer_interface()
    missing = REQUIRED_INTERFACE_KEYS - set(iface)
    assert not missing, missing
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


def test_select_terminal_interface_ready_when_invariants_pass() -> None:
    report = run_all_invariants(build_fixture_pack())
    terminal = select_terminal(report, report["ranking_active_mechanism_test"])
    assert terminal == "GCTM_D1_INTERFACE_READY"
    assert terminal in ALLOWED_TERMINALS


def test_sealed_packet_terminal_and_identities(tmp_path: Path) -> None:
    # Fresh emit into tmp, then compare sealed packet on disk for terminal.
    result = emit_packet(tmp_path)
    assert result["selected_terminal"] == "GCTM_D1_INTERFACE_READY"
    assert result["all_invariants_passed"] is True

    sealed = json.loads((PACKET / "terminal_report.json").read_text(encoding="utf-8"))
    assert sealed["selected_terminal"] == "GCTM_D1_INTERFACE_READY"
    assert sealed["diagnostic_id"] == DIAGNOSTIC_ID
    assert sealed["registry_effect"]["may_transition_slot_ids"] == ["GCTM_D1"]
    assert "H0_ROUTE5_B1" in sealed["registry_effect"]["must_not_alter"]
    assert "GCTM_B1" in sealed["registry_effect"]["must_not_alter"]

    manifest = json.loads((PACKET / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["non_runtime"] is True
    assert manifest["h0_forbidden"] is True
    assert manifest["selected_terminal"] == "GCTM_D1_INTERFACE_READY"

    fixture = PACKET / "fixture_pack.json"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    recorded = (PACKET / "fixture_pack.json.sha256").read_text(encoding="utf-8").strip()
    assert digest == recorded

    identities = json.loads((PACKET / "identities.json").read_text(encoding="utf-8"))
    for key in (
        "fixture_sha256",
        "gctm_theory_sha256",
        "gctm_lemmas_sha256",
        "score_contract_sha256",
        "runner",
    ):
        assert identities[key]


def test_no_h0_or_runtime_activation_language_in_terminal() -> None:
    sealed = json.loads((PACKET / "terminal_report.json").read_text(encoding="utf-8"))
    blocked = " ".join(sealed["blocked_claims"]).lower()
    assert "runtime fidelity" in blocked
    assert "h0" in blocked
    assert sealed["canonical_conclusion"]["selected_terminal"] == (
        "GCTM_D1_INTERFACE_READY"
    )
