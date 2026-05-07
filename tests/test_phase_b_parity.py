"""Phase B parity: C++ TrackletLifecycleMerger and IdentityResolver must produce
byte-equal IDs, alias, and stats as the Python counterparts for identical inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

import pytest  # noqa: E402

try:
    from saccade_tracking_ext import (
        TrackletLifecycleMerger as CppLifecycleMerger,
        IdentityResolver as CppIdentityResolver,
        SemanticRelinker as CppRelinker,
    )

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

pytestmark = pytest.mark.skipif(not _HAS_CPP, reason="C++ ext not available")

from saccade.perception.eval.relink import (  # noqa: E402
    IdentityResolver as PyIdentityResolver,
    PythonSemanticRelinker,
)
from saccade.perception.eval.lifecycle import (  # noqa: E402
    TrackletLifecycleMerger as PyLifecycleMerger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_py_lifecycle(**kw):
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
    defaults.update(kw)
    return PyLifecycleMerger(**defaults)


def _make_cpp_lifecycle(**kw):
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
    defaults.update(kw)
    return CppLifecycleMerger(**defaults)


def _make_py_relinker(**kw):
    defaults = dict(
        sim_threshold=0.70,
        ttl=30,
        ema_beta=0.80,
        spatial_gate=0.30,
        min_lost_frames=1,
        min_iou=0.10,
        mahalanobis_threshold=0.0,
    )
    defaults.update(kw)
    return PythonSemanticRelinker(**defaults)


def _make_cpp_relinker(**kw):
    defaults = dict(
        sim_threshold=0.70,
        ttl=30,
        ema_beta=0.80,
        spatial_gate=0.30,
        min_lost_frames=1,
        min_iou=0.10,
        mahalanobis_threshold=0.0,
    )
    defaults.update(kw)
    return CppRelinker(**defaults)


def _frames(
    rng: torch.Generator,
    n_frames: int,
    n_per: int,
    ids: list[int],
    frame_w=640,
    frame_h=480,
    emb_dim=8,
):
    frames = []
    for _ in range(n_frames):
        batch = []
        for k in range(n_per):
            x1 = float(torch.rand(1, generator=rng).item() * (frame_w - 60))
            y1 = float(torch.rand(1, generator=rng).item() * (frame_h - 60))
            x2 = x1 + float(torch.rand(1, generator=rng).item() * 80 + 20)
            y2 = y1 + float(torch.rand(1, generator=rng).item() * 80 + 20)
            emb = torch.nn.functional.normalize(
                torch.randn(emb_dim, generator=rng), dim=0
            )
            batch.append(
                (
                    ids[k % len(ids)],
                    (x1, y1, x2, y2),
                    float(torch.rand(1, generator=rng).item() * 0.5 + 0.5),
                    emb,
                )
            )
        frames.append(batch)
    return frames


# ---------------------------------------------------------------------------
# TrackletLifecycleMerger parity
# ---------------------------------------------------------------------------


def _run_lifecycle_many(lc, frames, frame_w, frame_h):
    all_ids = []
    for frame_id, batch in enumerate(frames, start=1):
        local_ids = [b[0] for b in batch]
        boxes = [b[1] for b in batch]
        scores = [b[2] for b in batch]
        embeddings = [b[3] for b in batch]
        ids = lc.resolve_many_packed(
            local_ids,
            boxes,
            scores,
            embeddings,
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        all_ids.append([int(i) for i in ids])
        lc.prune(frame_id)
    return all_ids


def test_lifecycle_ids_match():
    rng = torch.Generator()
    rng.manual_seed(42)
    frames = _frames(rng, 12, 5, [1, 2, 3, 4, 5])
    py_lc = _make_py_lifecycle()
    cpp_lc = _make_cpp_lifecycle()
    py_ids = _run_lifecycle_many(py_lc, frames, 640, 480)
    cpp_ids = _run_lifecycle_many(cpp_lc, frames, 640, 480)
    assert py_ids == cpp_ids, f"ID mismatch:\n  py = {py_ids}\n  cpp= {cpp_ids}"


def test_lifecycle_alias_matches():
    rng = torch.Generator()
    rng.manual_seed(77)
    frames = _frames(rng, 8, 4, [10, 20, 30, 40])
    py_lc = _make_py_lifecycle()
    cpp_lc = _make_cpp_lifecycle()
    _run_lifecycle_many(py_lc, frames, 1920, 1080)
    _run_lifecycle_many(cpp_lc, frames, 1920, 1080)
    assert dict(py_lc.alias) == dict(cpp_lc.alias), (
        f"alias mismatch:\n  py = {py_lc.alias}\n  cpp= {dict(cpp_lc.alias)}"
    )


def test_lifecycle_stats_match():
    rng = torch.Generator()
    rng.manual_seed(13)
    frames = _frames(rng, 10, 6, [1, 2, 3, 4, 5, 6])
    py_lc = _make_py_lifecycle()
    cpp_lc = _make_cpp_lifecycle()
    _run_lifecycle_many(py_lc, frames, 640, 480)
    _run_lifecycle_many(cpp_lc, frames, 640, 480)
    assert dict(py_lc.stats) == dict(cpp_lc.stats), (
        f"stats mismatch:\n  py = {dict(py_lc.stats)}\n  cpp= {dict(cpp_lc.stats)}"
    )


def test_lifecycle_none_embeddings():
    """None embeddings must not crash and must match."""
    py_lc = _make_py_lifecycle()
    cpp_lc = _make_cpp_lifecycle()
    boxes = [(10.0, 10.0, 50.0, 50.0), (60.0, 60.0, 100.0, 100.0)]
    scores = [0.9, 0.8]
    embs = [None, None]
    ids_py = [
        int(i)
        for i in py_lc.resolve_many_packed(
            [1, 2], boxes, scores, embs, frame_id=1, frame_w=640, frame_h=480
        )
    ]
    ids_cpp = [
        int(i)
        for i in cpp_lc.resolve_many_packed(
            [1, 2], boxes, scores, embs, frame_id=1, frame_w=640, frame_h=480
        )
    ]
    assert ids_py == ids_cpp


def test_lifecycle_disabled():
    py_lc = _make_py_lifecycle(enabled=False)
    cpp_lc = _make_cpp_lifecycle(enabled=False)
    rng = torch.Generator()
    rng.manual_seed(5)
    frames = _frames(rng, 5, 3, [1, 2, 3])
    py_ids = _run_lifecycle_many(py_lc, frames, 640, 480)
    cpp_ids = _run_lifecycle_many(cpp_lc, frames, 640, 480)
    assert py_ids == cpp_ids


# ---------------------------------------------------------------------------
# IdentityResolver (C++ vs Python) parity
# ---------------------------------------------------------------------------


def _run_resolver(resolver, lc, frames, frame_w, frame_h):
    all_ids = []
    for frame_id, batch in enumerate(frames, start=1):
        local_ids = [b[0] for b in batch]
        embs = [b[3] for b in batch]
        boxes = [b[1] for b in batch]
        scores = [b[2] for b in batch]
        ids = resolver.resolve_pass(
            local_ids,
            embs,
            boxes,
            scores,
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        all_ids.append([int(i) for i in ids])
        lc.prune(frame_id)
    return all_ids


def test_identity_resolver_cpp_vs_python_ids():
    rng = torch.Generator()
    rng.manual_seed(42)
    frames = _frames(rng, 12, 5, [1, 2, 3, 4, 5])

    py_rk = _make_py_relinker()
    py_lc = _make_py_lifecycle()
    cpp_rk = _make_cpp_relinker()
    cpp_lc = _make_cpp_lifecycle()

    py_resolver = PyIdentityResolver(py_rk, py_lc)
    cpp_resolver = CppIdentityResolver(cpp_rk, cpp_lc)

    py_ids = _run_resolver(py_resolver, py_lc, frames, 640, 480)
    cpp_ids = _run_resolver(cpp_resolver, cpp_lc, frames, 640, 480)

    assert py_ids == cpp_ids, f"ID mismatch:\n  py = {py_ids}\n  cpp= {cpp_ids}"


def test_identity_resolver_cpp_vs_python_alias_and_stats():
    rng = torch.Generator()
    rng.manual_seed(99)
    frames = _frames(rng, 8, 4, [10, 20, 30, 40])

    py_rk = _make_py_relinker()
    py_lc = _make_py_lifecycle()
    cpp_rk = _make_cpp_relinker()
    cpp_lc = _make_cpp_lifecycle()

    _run_resolver(PyIdentityResolver(py_rk, py_lc), py_lc, frames, 640, 480)
    _run_resolver(CppIdentityResolver(cpp_rk, cpp_lc), cpp_lc, frames, 640, 480)

    assert dict(py_rk.alias) == dict(cpp_rk.alias), "relinker alias mismatch"
    assert dict(py_lc.alias) == dict(cpp_lc.alias), "lifecycle alias mismatch"
    assert dict(py_rk.stats) == dict(cpp_rk.stats), "relinker stats mismatch"
    assert dict(py_lc.stats) == dict(cpp_lc.stats), "lifecycle stats mismatch"


def test_identity_resolver_cpp_relink_match_scenario():
    """After track dies at frame 2, re-appears at frame 6 with similar embedding:
    C++ and Python must agree on whether the relink fires."""
    rng = torch.Generator()
    rng.manual_seed(7)
    emb_a = torch.nn.functional.normalize(torch.randn(8, generator=rng), dim=0)
    box_a = (100.0, 100.0, 150.0, 160.0)
    emb_b = torch.nn.functional.normalize(emb_a * 0.97 + torch.randn(8) * 0.03, dim=0)

    def run(resolver, lc):
        # Frame 1: establish track 1
        resolver.resolve_pass(
            [1], [emb_a], [box_a], [0.9], frame_id=1, frame_w=640, frame_h=480
        )
        lc.prune(1)
        # Frame 6: re-appear with similar embedding — relink candidate
        result = resolver.resolve_pass(
            [99], [emb_b], [box_a], [0.85], frame_id=6, frame_w=640, frame_h=480
        )
        lc.prune(6)
        return [int(i) for i in result]

    py_rk = _make_py_relinker()
    py_lc = _make_py_lifecycle()
    cpp_rk = _make_cpp_relinker()
    cpp_lc = _make_cpp_lifecycle()

    py_ids = run(PyIdentityResolver(py_rk, py_lc), py_lc)
    cpp_ids = run(CppIdentityResolver(cpp_rk, cpp_lc), cpp_lc)

    assert py_ids == cpp_ids, (
        f"relink scenario mismatch: py={py_ids} cpp={cpp_ids}\n"
        f"  py_rk.alias={py_rk.alias}  cpp_rk.alias={dict(cpp_rk.alias)}"
    )
