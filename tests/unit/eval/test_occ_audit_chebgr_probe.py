"""WP1: occ-exit Cheb-GR graph decision probe (diagnostic / default-off)."""

# scope: eval
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import torch
import torch.nn.functional as F

from saccade.perception.eval.clean_fifo_bank import CleanFifoBank
from saccade.perception.eval.occ_audit import (
    _chebgr_occ_exit_probe,
    occ_exit_audit_lines_from_bank,
    plan_occ_audit_episodes,
)
from saccade.perception.eval.post_merge import _parse_mot_lines

_DIM = 16


def _identity_emb(seed: int, n: int = 5) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    center = F.normalize(torch.randn(_DIM, generator=g), dim=0)
    return [
        F.normalize(center + 0.02 * torch.randn(_DIM, generator=g), dim=0)
        for _ in range(n)
    ]


def _stack(seed: int, n: int = 4) -> torch.Tensor:
    return torch.stack(_identity_emb(seed, n))


def _make_substrate_with_occ(
    track_id: int = 1,
    clean_before: int = 10,
    occ_start: int = 11,
    occ_end: int = 14,
    clean_after: int = 5,
    x: float = 0.0,
) -> list[str]:
    lines: list[str] = []
    for f in range(1, clean_before + 1):
        lines.append(f"{f},{track_id},{x},0,10,20,0.9,-1,-1,-1")
    for f in range(occ_start, occ_end + 1):
        lines.append(f"{f},{track_id},{x},0,10,20,0.9,-1,-1,-1")
        lines.append(f"{f},99,{x},5,10,20,0.9,-1,-1,-1")
    for f in range(occ_end + 1, occ_end + 1 + clean_after):
        lines.append(f"{f},{track_id},{x},0,10,20,0.9,-1,-1,-1")
    return lines


def _bank_fixture() -> tuple[
    list[str], CleanFifoBank, dict[tuple[int, int], torch.Tensor]
]:
    """Occlusion episode with distinct pre-episode vs post-exit identity."""
    lines = _make_substrate_with_occ()
    records = _parse_mot_lines(lines)
    episodes = plan_occ_audit_episodes(
        records,
        appearance_occlusion_cov=0.3,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
    )
    ep = episodes[0]
    identity_a = _identity_emb(42)
    identity_b = _identity_emb(99)
    bank = CleanFifoBank(fifo_n=20, stride=1, decide_n=5)
    for i, f in enumerate(range(1, 11)):
        bank.store(1, identity_a[i % len(identity_a)], f)
        bank.store(99, identity_b[i % len(identity_b)], f)
    audit_embs = {(1, f): identity_b[f % len(identity_b)] for f in ep.audit_frames}
    return lines, bank, audit_embs


# --- Contract tests (WP1 acceptance) -----------------------------------------


def test_1_disabled_behavior_unchanged():
    """enabled=False returns same object / no flags."""
    lines = _make_substrate_with_occ()
    bank = CleanFifoBank(fifo_n=20)
    audit_embs: dict[tuple[int, int], torch.Tensor] = {}
    out, stats = occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=False,
    )
    assert out is lines
    assert stats["flags"] == 0
    assert stats["episodes"] == 0
    assert stats["audited"] == 0


def test_2_chebgr_probe_default_off_no_chebgr_fields():
    """decision_log without chebgr_probe must not contain chebgr_* fields."""
    lines, bank, audit_embs = _bank_fixture()
    log: list[dict] = []
    occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        appearance_occlusion_cov=0.3,
        decision_log=log,
        # chebgr_probe defaults to False
    )
    assert log
    row = log[0]
    chebgr_keys = [k for k in row if k.startswith("chebgr_")]
    assert chebgr_keys == [], f"unexpected chebgr_* keys when probe off: {chebgr_keys}"
    assert "flag_delta" not in row
    assert "cosine_min" not in row
    assert "min_cos" in row
    assert "median_cos" in row


def test_3_chebgr_probe_logs_fields_when_enabled():
    """chebgr_probe=True populates required diagnostic columns."""
    lines, bank, audit_embs = _bank_fixture()
    log: list[dict] = []
    occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=True,
        chebgr_probe=True,
        decision_log=log,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        appearance_occlusion_cov=0.3,
    )
    assert log
    row = log[0]
    for key in (
        "chebgr_self_cost",
        "chebgr_margin",
        "chebgr_flag",
        "flag_delta",
        "chebgr_ref_n",
        "chebgr_audit_n",
    ):
        assert key in row, f"missing decision_log field: {key}"
    assert row["flag_delta"] in ("same", "cosine_only", "chebgr_only")
    assert int(row["chebgr_ref_n"]) >= 2
    assert int(row["chebgr_audit_n"]) >= 1


def test_4_chebgr_probe_is_log_only():
    """output lines/stats match exactly between probe off and on."""
    lines, bank, audit_embs = _bank_fixture()
    common = dict(
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        appearance_occlusion_cov=0.3,
    )
    log_off: list[dict] = []
    out_off, stats_off = occ_exit_audit_lines_from_bank(
        lines, bank, audit_embs, decision_log=log_off, chebgr_probe=False, **common
    )
    log_on: list[dict] = []
    out_on, stats_on = occ_exit_audit_lines_from_bank(
        lines, bank, audit_embs, decision_log=log_on, chebgr_probe=True, **common
    )

    assert out_on == out_off
    assert stats_on == stats_off
    # cosine decision surface identical; only extra diagnostic keys on probe path
    for k in (
        "flagged",
        "flag_frame",
        "min_cos",
        "median_cos",
        "audit_n",
        "ref_n_used",
    ):
        assert log_on[0][k] == log_off[0][k]


def test_5_raw_unique_sample_guard_no_prototype_collapse():
    """3 distinct ref rows + 2 audit rows stay as 3 and 2 (not collapsed to 1)."""
    g = torch.Generator().manual_seed(0)
    ref_stack = F.normalize(torch.randn(3, _DIM, generator=g), dim=1)
    audit_stack = F.normalize(torch.randn(2, _DIM, generator=g), dim=1)
    # Ensure rows are actually distinct (not accidental duplicates)
    assert torch.unique(ref_stack, dim=0).shape[0] == 3
    assert torch.unique(audit_stack, dim=0).shape[0] == 2

    out = _chebgr_occ_exit_probe(ref_stack, audit_stack, max_cost=0.45)
    assert out["chebgr_ref_n"] == 3
    assert out["chebgr_audit_n"] == 2
    # Must not collapse to a single prototype row for either side
    assert out["chebgr_ref_n"] != 1
    assert out["chebgr_audit_n"] != 1


# --- Helper behavior (supporting) --------------------------------------------


def test_probe_same_identity_accepted():
    ref = _stack(1)
    audit = _stack(1)
    out = _chebgr_occ_exit_probe(ref, audit, max_cost=0.45)
    assert out["chebgr_flag"] is False
    assert out["chebgr_reason"] == "accepted"
    assert float(out["chebgr_self_cost"]) <= 0.45


def test_probe_identity_transfer_flags_high_self_cost():
    ref = _stack(1)
    audit = _stack(99)
    out = _chebgr_occ_exit_probe(ref, audit, max_cost=0.45)
    assert float(out["chebgr_self_cost"]) > 0.45
    assert out["chebgr_flag"] is True
    assert out["chebgr_reason"] == "cost"


def test_probe_margin_gate_can_suppress_flag():
    ref = _stack(1)
    audit = _stack(99)
    occ = _stack(7)
    out = _chebgr_occ_exit_probe(ref, audit, occ, max_cost=0.1, margin=0.5)
    assert float(out["chebgr_self_cost"]) > 0.1
    assert float(out["chebgr_margin"]) < 0.5
    assert out["chebgr_flag"] is False
    assert out["chebgr_reason"] == "margin"


def test_probe_does_not_mutate_inputs():
    ref = _stack(1)
    audit = _stack(2)
    occ = _stack(3)
    ref_before, audit_before, occ_before = ref.clone(), audit.clone(), occ.clone()
    _chebgr_occ_exit_probe(ref, audit, occ)
    assert torch.equal(ref, ref_before)
    assert torch.equal(audit, audit_before)
    assert torch.equal(occ, occ_before)
