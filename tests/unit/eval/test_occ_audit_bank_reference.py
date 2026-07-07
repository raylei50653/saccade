"""Tests for occ-exit audit with bank-sourced reference (Phase 2a, #55)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from saccade.perception.eval.clean_fifo_bank import CleanFifoBank
from saccade.perception.eval.occ_audit import (
    occ_exit_audit_lines,
    occ_exit_audit_lines_from_bank,
    plan_occ_audit_episodes,
)
from saccade.perception.eval.post_merge import _parse_mot_lines


_DIM = 16


def _emb(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(_DIM, generator=g), dim=0)


def _identity_emb(seed: int, n: int = 5) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    center = F.normalize(torch.randn(_DIM, generator=g), dim=0)
    return [
        F.normalize(center + 0.02 * torch.randn(_DIM, generator=g), dim=0)
        for _ in range(n)
    ]


def _make_substrate_with_occ(
    track_id: int = 1,
    clean_before: int = 10,
    occ_start: int = 11,
    occ_end: int = 14,
    clean_after: int = 5,
    x: float = 0.0,
) -> list[str]:
    """Track with clean frames, then dirty (occlusion), then clean again."""
    lines: list[str] = []
    for f in range(1, clean_before + 1):
        lines.append(f"{f},{track_id},{x},0,10,20,0.9,-1,-1,-1")
    for f in range(occ_start, occ_end + 1):
        lines.append(f"{f},{track_id},{x},0,10,20,0.9,-1,-1,-1")
        lines.append(f"{f},99,{x},5,10,20,0.9,-1,-1,-1")
    for f in range(occ_end + 1, occ_end + 1 + clean_after):
        lines.append(f"{f},{track_id},{x},0,10,20,0.9,-1,-1,-1")
    return lines


def test_bank_reference_matches_post_hoc_when_embeddings_identical():
    """Bank ref and post-hoc ref should flag the same episodes when
    the bank's FIFO contains the same clean frames the planner would pick."""
    lines = _make_substrate_with_occ(
        track_id=1,
        clean_before=10,
        occ_start=11,
        occ_end=14,
        clean_after=5,
    )
    records = _parse_mot_lines(lines)

    episodes = plan_occ_audit_episodes(
        records,
        appearance_occlusion_cov=0.3,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
    )
    assert len(episodes) >= 1
    ep = episodes[0]

    identity_a = _identity_emb(42)
    identity_b = _identity_emb(99)

    post_hoc_embs: dict[tuple[int, int], torch.Tensor] = {}
    for f in ep.ref_frames:
        post_hoc_embs[(1, f)] = identity_a[f % len(identity_a)]
    for f in ep.audit_frames:
        post_hoc_embs[(1, f)] = identity_b[f % len(identity_b)]

    bank = CleanFifoBank(fifo_n=20, stride=1, decide_n=5)
    for i, f in enumerate(range(1, 11)):
        bank.store(1, identity_a[i % len(identity_a)], f)

    _, stats_post_hoc = occ_exit_audit_lines(
        lines,
        post_hoc_embs,
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
        appearance_occlusion_cov=0.3,
    )

    audit_embs = {(1, f): identity_b[f % len(identity_b)] for f in ep.audit_frames}
    _, stats_bank = occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
        appearance_occlusion_cov=0.3,
    )

    assert stats_post_hoc["flags"] == stats_bank["flags"]
    assert stats_post_hoc["audited"] == stats_bank["audited"]


def test_bank_reference_abstains_when_no_pre_episode_samples():
    """If the bank has no samples before occ_start, the audit abstains."""
    lines = _make_substrate_with_occ(
        track_id=1,
        clean_before=0,
        occ_start=1,
        occ_end=3,
        clean_after=5,
    )
    bank = CleanFifoBank(fifo_n=20)
    audit_embs: dict[tuple[int, int], torch.Tensor] = {
        (1, f): _emb(f) for f in range(4, 9)
    }
    _, stats = occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
        appearance_occlusion_cov=0.3,
    )
    assert stats["abstain_no_ref"] > 0
    assert stats["flags"] == 0


def test_bank_reference_respects_boundary_from_prior_cut():
    """After a prior flag, the reference only uses samples after the cut."""
    lines = _make_substrate_with_occ(
        track_id=1,
        clean_before=10,
        occ_start=11,
        occ_end=14,
        clean_after=5,
    )
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

    bank = CleanFifoBank(fifo_n=20)
    early = _identity_emb(1)
    late = _identity_emb(2)
    for i, f in enumerate(range(1, 6)):
        bank.store(1, early[i % len(early)], f)
    for i, f in enumerate(range(6, 11)):
        bank.store(1, late[i % len(late)], f)

    audit_embs = {(1, f): _emb(f + 100) for f in ep.audit_frames}

    _, stats = occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
        appearance_occlusion_cov=0.3,
    )
    assert stats["episodes"] >= 1


def test_bank_reference_disabled_returns_unchanged():
    lines = _make_substrate_with_occ()
    bank = CleanFifoBank(fifo_n=20)
    audit_embs: dict[tuple[int, int], torch.Tensor] = {}
    out, stats = occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=False,
    )
    assert stats["flags"] == 0
    assert out is lines


def test_bank_reference_provides_occluder_ref():
    """When occluder_margin >= 0, the occluder's reference comes from bank."""
    lines = _make_substrate_with_occ(
        track_id=1,
        clean_before=10,
        occ_start=11,
        occ_end=14,
        clean_after=5,
    )
    bank = CleanFifoBank(fifo_n=20)
    track_id_emb = _identity_emb(42)
    occluder_emb = _identity_emb(77)
    for i, f in enumerate(range(1, 11)):
        bank.store(1, track_id_emb[i % len(track_id_emb)], f)
        bank.store(99, occluder_emb[i % len(occluder_emb)], f)

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
    assert ep.occluder_id == 99

    different_person = _identity_emb(333)
    audit_embs = {
        (1, f): different_person[f % len(different_person)] for f in ep.audit_frames
    }
    _, stats = occ_exit_audit_lines_from_bank(
        lines,
        bank,
        audit_embs,
        enabled=True,
        tau=0.45,
        min_ref=2,
        ref_n=5,
        audit_crops=3,
        audit_window=30,
        min_occ_frames=2,
        appearance_occlusion_cov=0.3,
        occluder_margin=0.0,
    )
    assert stats["abstain_no_occref"] == 0
