"""Parity test: IdentityResolver.resolve_pass must be byte-equal to the legacy
_resolve_frame_tracks path (relink → lifecycle) for every candidate across every frame."""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.relink import IdentityResolver, PythonSemanticRelinker  # noqa: E402
from saccade.perception.eval.lifecycle import TrackletLifecycleMerger  # noqa: E402
from saccade.perception.eval.types import PreparedTrackCandidate  # noqa: E402
from saccade.perception.eval.helpers import (  # noqa: E402
    resolve_frame_tracks as _resolve_frame_tracks,
)


def _make_relinker(**overrides):
    defaults = dict(
        sim_threshold=0.70,
        ttl=30,
        ema_beta=0.80,
        spatial_gate=0.30,
        min_lost_frames=1,
        min_iou=0.10,
        mahalanobis_threshold=0.0,
    )
    defaults.update(overrides)
    return PythonSemanticRelinker(**defaults)


def _make_lifecycle(**overrides):
    defaults = dict(
        enabled=True,
        ttl=20,
        min_gap=1,
        spatial_gate=0.30,
        min_iou=0.10,
        sim_threshold=0.50,
        require_embedding=False,
        ema=0.70,
    )
    defaults.update(overrides)
    return TrackletLifecycleMerger(**defaults)


def _make_candidates(
    rng: torch.Generator,
    n: int,
    ids: list[int],
    emb_dim: int = 8,
    frame_w: int = 640,
    frame_h: int = 480,
) -> list[PreparedTrackCandidate]:
    candidates = []
    for i in range(n):
        x1 = float(torch.rand(1, generator=rng).item() * (frame_w - 50))
        y1 = float(torch.rand(1, generator=rng).item() * (frame_h - 50))
        x2 = x1 + float(torch.rand(1, generator=rng).item() * 80 + 20)
        y2 = y1 + float(torch.rand(1, generator=rng).item() * 80 + 20)
        score = float(torch.rand(1, generator=rng).item() * 0.5 + 0.5)
        emb = torch.nn.functional.normalize(torch.randn(emb_dim, generator=rng), dim=0)
        candidates.append(
            PreparedTrackCandidate(
                local_track_id=ids[i % len(ids)],
                box=(x1, y1, x2, y2),
                score=score,
                embedding=emb,
            )
        )
    return candidates


def _run_via_frame_tracks(
    frames: list[list[PreparedTrackCandidate]],
    relinker: PythonSemanticRelinker,
    lifecycle: TrackletLifecycleMerger,
    frame_w: int,
    frame_h: int,
) -> list[list[int]]:
    """Run through _resolve_frame_tracks (via IdentityResolver) — tests wiring."""
    resolver = IdentityResolver(relinker, lifecycle)
    results = []
    for frame_id, candidates in enumerate(frames, start=1):
        resolved = _resolve_frame_tracks(
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
            prepared_candidates=candidates,
            lifecycle_merger=lifecycle,
            identity_resolver=resolver,
        )
        results.append([t.resolved_track_id for t in resolved])
        lifecycle.prune(frame_id)
    return results


def _run_resolver(
    frames: list[list[PreparedTrackCandidate]],
    resolver: IdentityResolver,
    lifecycle: TrackletLifecycleMerger,
    frame_w: int,
    frame_h: int,
) -> list[list[int]]:
    results = []
    for frame_id, candidates in enumerate(frames, start=1):
        resolved_ids = resolver.resolve_pass(
            [c.local_track_id for c in candidates],
            [c.embedding for c in candidates],
            [c.box for c in candidates],
            [c.score for c in candidates],
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        results.append([int(rid) for rid in resolved_ids])
        lifecycle.prune(frame_id)
    return results


def test_resolve_pass_ids_match_frame_tracks() -> None:
    """_resolve_frame_tracks(identity_resolver=...) must produce same IDs as direct resolve_pass."""
    rng = torch.Generator()
    rng.manual_seed(42)
    frame_w, frame_h = 640, 480
    n_frames, n_per_frame = 10, 5
    track_ids = [1, 2, 3, 4, 5]

    frames = [
        _make_candidates(rng, n_per_frame, track_ids, frame_w=frame_w, frame_h=frame_h)
        for _ in range(n_frames)
    ]

    rk_ft = _make_relinker()
    lc_ft = _make_lifecycle()
    ft_ids = _run_via_frame_tracks(frames, rk_ft, lc_ft, frame_w, frame_h)

    rk_new = _make_relinker()
    lc_new = _make_lifecycle()
    resolver = IdentityResolver(rk_new, lc_new)
    resolver_ids = _run_resolver(frames, resolver, lc_new, frame_w, frame_h)

    assert ft_ids == resolver_ids, (
        f"ID mismatch:\n  frame_tracks = {ft_ids}\n  resolver     = {resolver_ids}"
    )


def test_resolve_pass_alias_and_stats_match_frame_tracks() -> None:
    """_resolve_frame_tracks and direct resolve_pass must leave identical alias/stats."""
    rng = torch.Generator()
    rng.manual_seed(99)
    frame_w, frame_h = 1920, 1080
    n_frames, n_per_frame = 8, 4
    track_ids = [10, 20, 30, 40]

    frames = [
        _make_candidates(rng, n_per_frame, track_ids, frame_w=frame_w, frame_h=frame_h)
        for _ in range(n_frames)
    ]

    rk_ft = _make_relinker()
    lc_ft = _make_lifecycle()
    _run_via_frame_tracks(frames, rk_ft, lc_ft, frame_w, frame_h)

    rk_new = _make_relinker()
    lc_new = _make_lifecycle()
    _run_resolver(frames, IdentityResolver(rk_new, lc_new), lc_new, frame_w, frame_h)

    assert dict(rk_ft.alias) == dict(rk_new.alias), "relinker alias mismatch"
    assert dict(lc_ft.alias) == dict(lc_new.alias), "lifecycle alias mismatch"
    assert dict(rk_ft.stats) == dict(rk_new.stats), "relinker stats mismatch"
    assert dict(lc_ft.stats) == dict(lc_new.stats), "lifecycle stats mismatch"


def test_resolve_pass_empty_candidates() -> None:
    rk = _make_relinker()
    lc = _make_lifecycle()
    resolver = IdentityResolver(rk, lc)
    result = resolver.resolve_pass([], [], [], [], frame_id=1, frame_w=640, frame_h=480)
    assert result == []


def test_resolve_pass_with_none_embeddings() -> None:
    """Candidates without embeddings (embedding=None) must not crash."""
    rk = _make_relinker()
    lc = _make_lifecycle()
    resolver = IdentityResolver(rk, lc)

    candidates = [
        PreparedTrackCandidate(
            local_track_id=7, box=(10.0, 10.0, 50.0, 50.0), score=0.8, embedding=None
        ),
        PreparedTrackCandidate(
            local_track_id=8, box=(60.0, 60.0, 100.0, 100.0), score=0.9, embedding=None
        ),
    ]
    result = resolver.resolve_pass(
        [c.local_track_id for c in candidates],
        [c.embedding for c in candidates],
        [c.box for c in candidates],
        [c.score for c in candidates],
        frame_id=1,
        frame_w=640,
        frame_h=480,
    )
    assert len(result) == 2


def test_resolve_pass_with_relink_match() -> None:
    """After a track dies and re-appears, frame_tracks path and direct resolver agree."""
    rng = torch.Generator()
    rng.manual_seed(7)
    frame_w, frame_h = 640, 480

    rng.manual_seed(7)
    emb_a = torch.nn.functional.normalize(torch.randn(8, generator=rng), dim=0)
    box_a = (100.0, 100.0, 150.0, 160.0)

    def run_two_frame(relinker, lifecycle):
        resolver = IdentityResolver(relinker, lifecycle)
        # frame 1: establish track 1
        _resolve_frame_tracks(
            frame_id=1,
            frame_w=frame_w,
            frame_h=frame_h,
            prepared_candidates=[PreparedTrackCandidate(1, box_a, 0.9, emb_a)],
            lifecycle_merger=lifecycle,
            identity_resolver=resolver,
        )
        lifecycle.prune(1)
        emb_b = torch.nn.functional.normalize(
            emb_a * 0.99 + torch.randn(8) * 0.01, dim=0
        )
        result = _resolve_frame_tracks(
            frame_id=5,
            frame_w=frame_w,
            frame_h=frame_h,
            prepared_candidates=[PreparedTrackCandidate(99, box_a, 0.85, emb_b)],
            lifecycle_merger=lifecycle,
            identity_resolver=resolver,
        )
        lifecycle.prune(5)
        return [t.resolved_track_id for t in result]

    def run_two_frame_resolver(relinker, lifecycle):
        resolver = IdentityResolver(relinker, lifecycle)
        resolver.resolve_pass(
            [1],
            [emb_a],
            [box_a],
            [0.9],
            frame_id=1,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        lifecycle.prune(1)
        emb_b = torch.nn.functional.normalize(
            emb_a * 0.99 + torch.randn(8) * 0.01, dim=0
        )
        result = resolver.resolve_pass(
            [99],
            [emb_b],
            [box_a],
            [0.85],
            frame_id=5,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        lifecycle.prune(5)
        return [int(rid) for rid in result]

    legacy_ids = run_two_frame(_make_relinker(), _make_lifecycle())
    resolver_ids = run_two_frame_resolver(_make_relinker(), _make_lifecycle())
    assert legacy_ids == resolver_ids
