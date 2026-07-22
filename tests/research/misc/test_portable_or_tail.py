"""Unit tests for frozen portable OR-tail policy loader / evaluator."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from saccade.perception.eval.portable_or_tail import (
    ACTIVATION_CONTROL_ATOM0_THR,
    CONTROL_DISABLED_THR,
    EXPECTED_CANDIDATE_ID,
    FORCE_REJECT_ATOM0_THR,
    FROZEN_POLICY_SHA256,
    FROZEN_THR_VECTOR,
    ORDERED_ATOM_IDS,
    PortableAuditNotImplementedError,
    PortablePolicyError,
    STAGE1_OP,
    classify_e2e_status,
    classify_stage1_milestones,
    evaluate_policy,
    evaluate_policy_row,
    fire_class_counts,
    load_portable_policy,
    reconcile_fire_classes,
    require_online_audit_available,
    resolve_policy_path_from_env,
    snapshot_policy,
)

CONTROL_FIX_DIR = Path("scripts/tools/fixtures/m_b1_stage1")

FREEZE = Path(
    "out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json"
)


def _synthetic_policy(
    tmp_path: Path,
    thr: list[float],
    *,
    candidate_id: str = EXPECTED_CANDIDATE_ID,
    op: str = ">",
    kind0: str = "tail_q",
) -> Path:
    raw = {
        "clauses": [[aid] for aid in ORDERED_ATOM_IDS],
        "atom_specs": {
            aid: {
                "atom_id": aid,
                "signal": aid.split(":")[0],
                "kind": kind0 if i == 0 else "tail_q",
                "op": op,
                "thr": thr[i],
            }
            for i, aid in enumerate(ORDERED_ATOM_IDS)
        },
        "eps": 0.0,
        "candidate_id": candidate_id,
    }
    p = tmp_path / "portable_policy.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


@pytest.mark.skipif(not FREEZE.is_file(), reason="freeze portable_policy.json missing")
def test_load_freeze_policy_strict_lock() -> None:
    pol = load_portable_policy(FREEZE, enforce_freeze_lock=True)
    assert pol.candidate_id == EXPECTED_CANDIDATE_ID
    assert pol.freeze_locked is True
    assert pol.file_hash == FROZEN_POLICY_SHA256
    assert all(
        abs(a - b) < 1e-12
        for a, b in zip(pol.thr_vector, FROZEN_THR_VECTOR, strict=True)
    )
    assert all(a.op == STAGE1_OP for a in pol.atoms)
    snap = snapshot_policy(pol)
    assert snap["ordered_atom_ids"] == list(ORDERED_ATOM_IDS)
    assert snap["freeze_locked"] is True


def test_load_rejects_zone_atom(tmp_path: Path) -> None:
    p = _synthetic_policy(tmp_path, [1.0, 2.0, 3.0, 4.0, 5.0], kind0="zone_q")
    with pytest.raises(PortablePolicyError, match="zone/gap"):
        load_portable_policy(p, enforce_freeze_lock=False)


def test_load_rejects_missing_atom(tmp_path: Path) -> None:
    incomplete_ids = list(ORDERED_ATOM_IDS)[:4]
    bad = {
        "clauses": [[aid] for aid in incomplete_ids],
        "atom_specs": {
            aid: {
                "atom_id": aid,
                "signal": aid.split(":")[0],
                "kind": "tail_q",
                "op": ">",
                "thr": 1.0,
            }
            for aid in incomplete_ids
        },
        "eps": 0.0,
        "candidate_id": EXPECTED_CANDIDATE_ID,
    }
    p = tmp_path / "portable_policy.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PortablePolicyError):
        load_portable_policy(p, enforce_freeze_lock=False)


def test_load_rejects_missing_candidate_id_soft_fallback(tmp_path: Path) -> None:
    thr = list(FROZEN_THR_VECTOR)
    # Parent dir is NOT the freeze id and no candidate_id field.
    study = tmp_path / "some_other_study"
    study.mkdir()
    raw = {
        "clauses": [[aid] for aid in ORDERED_ATOM_IDS],
        "atom_specs": {
            aid: {
                "atom_id": aid,
                "signal": aid.split(":")[0],
                "kind": "tail_q",
                "op": ">",
                "thr": thr[i],
            }
            for i, aid in enumerate(ORDERED_ATOM_IDS)
        },
        "eps": 0.0,
        # intentionally omit candidate_id
    }
    p = study / "portable_policy.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PortablePolicyError, match="refuse soft fallback"):
        load_portable_policy(p, enforce_freeze_lock=False)


def test_load_rejects_ge_op(tmp_path: Path) -> None:
    p = _synthetic_policy(tmp_path, [1.0, 2.0, 3.0, 4.0, 5.0], op=">=")
    with pytest.raises(PortablePolicyError, match="unsupported"):
        load_portable_policy(p, enforce_freeze_lock=False)


def test_load_rejects_thr_drift_under_freeze_lock(tmp_path: Path) -> None:
    thr = list(FROZEN_THR_VECTOR)
    thr[0] = thr[0] + 0.01  # drift
    p = _synthetic_policy(tmp_path, thr)
    with pytest.raises(PortablePolicyError, match="thr_vector mismatch"):
        load_portable_policy(p, enforce_freeze_lock=True)


def test_load_rejects_hash_drift_under_freeze_lock(tmp_path: Path) -> None:
    # Valid thr + candidate_id but not the freeze file bytes.
    p = _synthetic_policy(tmp_path, list(FROZEN_THR_VECTOR))
    with pytest.raises(PortablePolicyError, match="file hash mismatch"):
        load_portable_policy(p, enforce_freeze_lock=True)


def test_evaluate_or_semantics(tmp_path: Path) -> None:
    thr = [10.0, 1.0, 5.0, 2.0, 8.0]
    p = _synthetic_policy(tmp_path, thr)
    pol = load_portable_policy(p, enforce_freeze_lock=False)

    # zero fire
    z = evaluate_policy_row(
        pol,
        {
            "score_m_bridge": 0.1,
            "abs_log_h": 0.1,
            "dist_h": 0.1,
            "abs_ratio_m1": 0.1,
            "resid_mean": 0.1,
        },
    )
    assert z["reject"] is False
    assert z["fire_class"] == "zero"
    assert z["n_atoms_fired"] == 0

    # singleton: abs_log_h only
    s = evaluate_policy_row(
        pol,
        {
            "score_m_bridge": 0.1,
            "abs_log_h": 1.5,
            "dist_h": 0.1,
            "abs_ratio_m1": 0.1,
            "resid_mean": 0.1,
        },
    )
    assert s["reject"] is True
    assert s["fire_class"] == "singleton"
    assert s["fired_atom_ids"] == ["abs_log_h:tail_q85"]

    # cofire
    c = evaluate_policy_row(
        pol,
        {
            "score_m_bridge": 11.0,
            "abs_log_h": 1.5,
            "dist_h": 0.1,
            "abs_ratio_m1": 0.1,
            "resid_mean": 0.1,
        },
    )
    assert c["fire_class"] == "cofire"
    assert c["n_atoms_fired"] == 2

    # vector path
    sig = {
        "score_m_bridge": np.array([0.1, 11.0, 0.1]),
        "abs_log_h": np.array([0.1, 1.5, 1.5]),
        "dist_h": np.array([0.1, 0.1, 0.1]),
        "abs_ratio_m1": np.array([0.1, 0.1, 0.1]),
        "resid_mean": np.array([0.1, 0.1, 0.1]),
    }
    out = evaluate_policy(pol, sig)
    assert out["reject"].tolist() == [False, True, True]
    counts = fire_class_counts(out["fire_class"])
    assert counts["n_zero_fire"] == 1
    assert counts["n_singleton"] == 1
    assert counts["n_cofire"] == 1
    errs = reconcile_fire_classes(
        n_hook_eligible=3,
        n_zero=counts["n_zero_fire"],
        n_singleton=counts["n_singleton"],
        n_cofire=counts["n_cofire"],
        n_rejected=2,
    )
    assert errs == []


def test_audit_flag_fail_closed() -> None:
    require_online_audit_available(audit_enabled=False)  # no-op
    with pytest.raises(PortableAuditNotImplementedError, match="NOT implemented"):
        require_online_audit_available(audit_enabled=True)


def test_resolve_path_default_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SACCADE_RESEARCH_PORTABLE_OR_TAIL_POLICY", raising=False)
    assert resolve_policy_path_from_env(None) is None
    p = tmp_path / "p.json"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SACCADE_RESEARCH_PORTABLE_OR_TAIL_POLICY", str(p))
    assert resolve_policy_path_from_env(None) == p
    assert (
        resolve_policy_path_from_env(str(tmp_path / "cli.json"))
        == tmp_path / "cli.json"
    )


def test_classify_e2e_status() -> None:
    assert (
        classify_e2e_status(hook_off_identity_ok=False, n_rejected=0)
        == "online_inconclusive"
    )
    assert (
        classify_e2e_status(
            hook_off_identity_ok=True, n_rejected=0, metrics_delta={"IDF1": 0.0}
        )
        == "online_effect_neutral_but_safe"
    )
    assert (
        classify_e2e_status(
            hook_off_identity_ok=True,
            n_rejected=10,
            metrics_delta={"IDF1": -1.0},
            per_seq_regression=True,
        )
        == "online_unsafe"
    )


def test_classify_stage1_milestones_entry_only_keeps_overall_open() -> None:
    m = classify_stage1_milestones(
        evaluation_entry_ok=True,
        frozen_policy_null_effect=True,
        activation_action_path_ok=None,
        force_reject_path_ok=None,
        soft_a0_identity_ok=True,
        strict_a0_identity_ok=False,
    )
    assert m["stage1a_evaluation_entry"] == "PASSED"
    assert m["stage1b_action_path"] == "NOT_ACTIVATED"
    assert m["stage1_overall"] == "OPEN"
    assert m["a0_identity"] == "soft_pass_strict_unresolved"
    assert "evaluation-entry" in m["headline_claim_allowed"]
    assert "unactivated" in m["headline_claim_allowed"]


def test_classify_stage1_milestones_action_path_still_not_full_close() -> None:
    m = classify_stage1_milestones(
        evaluation_entry_ok=True,
        frozen_policy_null_effect=True,
        activation_action_path_ok=True,
        force_reject_path_ok=True,
        online_baudit_ok=False,
        strict_a0_identity_ok=False,
        soft_a0_identity_ok=True,
    )
    assert m["stage1b_action_path"] == "PASSED"
    assert m["stage1b_eng_milestone"] == "PASSED"
    # Full Stage 1 still open without B-audit + strict A0 + determinism.
    assert m["stage1_overall"] == "OPEN"


@pytest.mark.skipif(
    not (CONTROL_FIX_DIR / "activation_control_policy.json").is_file(),
    reason="activation control fixture missing",
)
def test_load_activation_control_skips_freeze_lock() -> None:
    pol = load_portable_policy(
        CONTROL_FIX_DIR / "activation_control_policy.json",
        enforce_freeze_lock=True,
    )
    assert pol.control_arm == "activation"
    assert pol.freeze_locked is False
    assert pol.thr_vector[0] == ACTIVATION_CONTROL_ATOM0_THR
    assert all(t >= CONTROL_DISABLED_THR * 0.5 for t in pol.thr_vector[1:])
    # In-domain synthetic: thr=0.2 fires on score 0.3
    out = evaluate_policy_row(
        pol,
        {
            "score_m_bridge": 0.3,
            "abs_log_h": 0.0,
            "dist_h": 0.0,
            "abs_ratio_m1": 0.0,
            "resid_mean": 0.0,
        },
    )
    assert out["reject"] is True


@pytest.mark.skipif(
    not (CONTROL_FIX_DIR / "force_reject_policy.json").is_file(),
    reason="force_reject fixture missing",
)
def test_load_force_reject_control() -> None:
    pol = load_portable_policy(
        CONTROL_FIX_DIR / "force_reject_policy.json",
        enforce_freeze_lock=True,
    )
    assert pol.control_arm == "force_reject"
    assert pol.thr_vector[0] == FORCE_REJECT_ATOM0_THR
    out = evaluate_policy(
        pol,
        {
            "score_m_bridge": np.array([0.0, 0.1, 0.5]),
            "abs_log_h": np.zeros(3),
            "dist_h": np.zeros(3),
            "abs_ratio_m1": np.zeros(3),
            "resid_mean": np.zeros(3),
        },
    )
    assert bool(out["reject"].all())
