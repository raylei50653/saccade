"""Unit tests for frozen portable OR-tail policy loader / evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from saccade.perception.eval.portable_or_tail import (
    EXPECTED_CANDIDATE_ID,
    ORDERED_ATOM_IDS,
    PortablePolicyError,
    classify_e2e_status,
    evaluate_policy,
    evaluate_policy_row,
    fire_class_counts,
    load_portable_policy,
    reconcile_fire_classes,
    resolve_policy_path_from_env,
    snapshot_policy,
)

FREEZE = Path(
    "out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json"
)


@pytest.mark.skipif(not FREEZE.is_file(), reason="freeze portable_policy.json missing")
def test_load_freeze_policy() -> None:
    pol = load_portable_policy(FREEZE)
    assert pol.candidate_id == EXPECTED_CANDIDATE_ID
    assert len(pol.thr_vector) == 5
    assert all(t > 0 for t in pol.thr_vector)
    assert pol.file_hash
    snap = snapshot_policy(pol)
    assert snap["ordered_atom_ids"] == list(ORDERED_ATOM_IDS)


def test_load_rejects_zone_atom(tmp_path: Path) -> None:
    bad = {
        "clauses": [[aid] for aid in ORDERED_ATOM_IDS],
        "atom_specs": {
            aid: {
                "atom_id": aid,
                "signal": aid.split(":")[0],
                "kind": "zone_q" if i == 0 else "tail_q",
                "op": ">",
                "thr": 1.0 + i,
            }
            for i, aid in enumerate(ORDERED_ATOM_IDS)
        },
        "eps": 0.0,
        "candidate_id": EXPECTED_CANDIDATE_ID,
    }
    p = tmp_path / "portable_policy.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PortablePolicyError, match="zone/gap"):
        load_portable_policy(p)


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
        load_portable_policy(p)


def test_evaluate_or_semantics(tmp_path: Path) -> None:
    thr = [10.0, 1.0, 5.0, 2.0, 8.0]
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
        "candidate_id": EXPECTED_CANDIDATE_ID,
    }
    p = tmp_path / "portable_policy.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    pol = load_portable_policy(p)

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
