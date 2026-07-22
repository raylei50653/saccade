"""Unit tests for the causal occ-exit identity audit (numeric core)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import torch
import torch.nn.functional as F

from saccade.perception.eval.occ_audit import (
    occ_exit_audit_lines,
    plan_occ_audit_episodes,
)
from saccade.perception.eval.post_merge import _parse_mot_lines


def _line(
    fr: int, tid: int, x: float, y: float, w: float = 20.0, h: float = 40.0
) -> str:
    return f"{fr},{tid},{x},{y},{w},{h},0.9,-1,-1,-1"


def _vec(d: int, hot: int) -> torch.Tensor:
    v = torch.zeros(d)
    v[hot] = 1.0
    return F.normalize(v, dim=0)


def _crossing_lines(n_frames: int = 30, occ_span: range = range(10, 15)) -> list[str]:
    """Track 1 walks in place; track 2's box fronts (lower foot) and covers it
    during ``occ_span`` — a geometric occlusion episode for track 1."""
    lines: list[str] = []
    for fr in range(1, n_frames + 1):
        lines.append(_line(fr, 1, x=100.0, y=100.0))
        if fr in occ_span:
            # same box, slightly lower foot: fronts and fully covers track 1
            lines.append(_line(fr, 2, x=100.0, y=104.0))
        else:
            lines.append(_line(fr, 2, x=300.0, y=100.0))
    return lines


def _embs_for(
    lines: list[str], ident_of: dict[tuple[int, int], int], d: int = 8
) -> dict[tuple[int, int], torch.Tensor]:
    """One-hot identity embeddings; default = the track's own id."""
    out: dict[tuple[int, int], torch.Tensor] = {}
    for r in _parse_mot_lines(lines):
        key = (r.track_id, r.frame)
        out[key] = _vec(d, ident_of.get(key, r.track_id))
    return out


def _ids_by_frame(lines: list[str], tid_pred) -> dict[int, int]:
    out = {}
    for ln in lines:
        p = ln.split(",")
        fr, tid = int(p[0]), int(p[1])
        if tid_pred(tid):
            out[fr] = tid
    return out


def test_plan_finds_crossing_episode():
    lines = _crossing_lines()
    episodes = plan_occ_audit_episodes(_parse_mot_lines(lines))
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.track_id == 1
    assert (ep.occ_start, ep.occ_end) == (10, 14)
    assert ep.ref_frames == [5, 6, 7, 8, 9]
    assert ep.audit_frames == [15, 16, 17]


def test_disabled_is_noop():
    lines = _crossing_lines()
    out, stats = occ_exit_audit_lines(lines, {}, enabled=False)
    assert out is lines
    assert stats["flags"] == 0


def test_consistent_identity_passes():
    lines = _crossing_lines()
    out, stats = occ_exit_audit_lines(lines, _embs_for(lines, {}), enabled=True)
    assert stats["episodes"] == 1
    assert stats["audited"] == 1
    assert stats["flags"] == 0
    assert out == lines


def test_identity_transfer_splits_from_flag_frame():
    lines = _crossing_lines()
    # After the occlusion, track 1's boxes are a different person (identity 7).
    swapped = {(1, fr): 7 for fr in range(15, 31)}
    log: list[dict] = []
    out, stats = occ_exit_audit_lines(
        lines, _embs_for(lines, swapped), enabled=True, decision_log=log
    )
    assert stats["flags"] == 1
    assert stats["ids_after"] == stats["ids_before"] + 1
    assert log[0]["flagged"] and log[0]["flag_frame"] == 15

    track1 = _ids_by_frame(out, lambda t: t != 2)
    # causal: frames before the decision keep the old id, from it a fresh one
    assert all(track1[fr] == 1 for fr in range(1, 15))
    new_ids = {track1[fr] for fr in range(15, 31)}
    assert len(new_ids) == 1 and new_ids != {1}
    assert next(iter(new_ids)) > 2


def test_second_episode_reference_respects_earlier_cut():
    # two episodes; identity changes at the first exit and stays the new person
    lines = _crossing_lines(n_frames=45, occ_span=range(10, 15))
    # add a second occlusion of track 1 at frames 30-34
    lines = [
        ln
        if not (30 <= int(ln.split(",")[0]) <= 34 and ln.split(",")[1] == "2")
        else _line(int(ln.split(",")[0]), 2, x=100.0, y=104.0)
        for ln in lines
    ]
    swapped = {(1, fr): 7 for fr in range(15, 46)}
    out, stats = occ_exit_audit_lines(lines, _embs_for(lines, swapped), enabled=True)
    # first episode flags; second episode's reference is rebuilt from the new
    # identity's frames (>= cut), so the consistent person 7 passes it.
    assert stats["episodes"] == 2
    assert stats["flags"] == 1


def test_no_reference_abstains():
    # occlusion right after birth: no clean pre-occlusion frames
    lines = _crossing_lines(occ_span=range(1, 6))
    out, stats = occ_exit_audit_lines(lines, _embs_for(lines, {}), enabled=True)
    assert stats["abstain_no_ref"] == 1
    assert stats["flags"] == 0
    assert out == lines
