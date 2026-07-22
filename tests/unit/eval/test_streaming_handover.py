"""Unit tests for the streaming (live) online Cheb-GR handover substrate."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from saccade.perception.eval.cheb_gr_online import causal_handover_lines
from saccade.perception.eval.streaming_handover import (
    EmbeddingRing,
    StreamingHandoverDriver,
    TrackCropRing,
)


def _normed(
    rng: np.random.Generator, n: int, d: int, center: np.ndarray
) -> torch.Tensor:
    raw = center[None, :] + 0.05 * rng.standard_normal((n, d)).astype(np.float32)
    return F.normalize(torch.from_numpy(raw), dim=1)


def _crop(val: float, hw: tuple[int, int] = (2, 2)) -> torch.Tensor:
    return torch.full((3, *hw), val, dtype=torch.float32)


# --------------------------------------------------------------------------
# TrackCropRing
# --------------------------------------------------------------------------
def test_crop_ring_gather_returns_recent_order():
    ring = TrackCropRing(capacity=8, crop_hw=(2, 2), depth=4, device="cpu")
    for f in range(3):
        ring.stash(uid=7, frame=f, crop=_crop(float(f)))
    crops, frames, clean = ring.gather(7)
    assert frames == [0, 1, 2]
    assert clean == [True, True, True]
    assert torch.allclose(crops[0], _crop(0.0))
    assert torch.allclose(crops[-1], _crop(2.0))


def test_crop_ring_per_uid_depth_caps_tail():
    ring = TrackCropRing(capacity=16, crop_hw=(2, 2), depth=3, device="cpu")
    for f in range(6):
        ring.stash(uid=1, frame=f, crop=_crop(float(f)))
    crops, frames, _ = ring.gather(1)
    assert frames == [3, 4, 5]  # only the most recent `depth` survive
    assert torch.allclose(crops[0], _crop(3.0))


def test_crop_ring_global_lru_eviction_bounds_memory():
    ring = TrackCropRing(capacity=3, crop_hw=(2, 2), depth=3, device="cpu")
    ring.stash(uid=1, frame=0, crop=_crop(0.0))
    ring.stash(uid=2, frame=0, crop=_crop(1.0))
    ring.stash(uid=3, frame=0, crop=_crop(2.0))
    assert len(ring) == 3
    # capacity is full; a new uid evicts the least-recently-stashed (uid 1).
    ring.stash(uid=4, frame=1, crop=_crop(3.0))
    assert ring.gather(1) is None
    assert ring.has(4)
    assert len(ring) == 3


def test_crop_ring_self_lru_eviction_keeps_bookkeeping():
    ring = TrackCropRing(capacity=3, crop_hw=(2, 2), depth=3, device="cpu")
    ring.stash(uid=1, frame=0, crop=_crop(0.0))
    ring.stash(uid=2, frame=0, crop=_crop(1.0))
    ring.stash(uid=3, frame=0, crop=_crop(2.0))
    # Pool full and uid 1 is below depth: its own slot is the global LRU
    # victim of its next stash. The new crop must stay reachable (regression:
    # the eviction deleted the uid's map entry while a stale deque was held).
    ring.stash(uid=1, frame=5, crop=_crop(9.0))
    g = ring.gather(1)
    assert g is not None
    crops, frames, _ = g
    assert frames == [5]
    assert torch.allclose(crops[0], _crop(9.0))
    assert len(ring) == 3


def test_cache_key_survives_mot_line_roundtrip():
    from saccade.perception.eval.cheb_gr_online import _cache_key

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 1920.0, size=(500, 4))
    raw[:, 2:] = raw[:, :2] + rng.uniform(1.0, 200.0, size=(500, 2))
    # .2f half-boundaries that flip a coarser (0.1px) quantization grid
    adversarial = np.array(
        [
            [12.345, 7.005, 12.345 + 30.055, 7.005 + 61.115],
            [0.125, 0.335, 50.555, 90.005],
        ]
    )
    for x1, y1, x2, y2 in np.vstack([raw, adversarial]):
        live_key = _cache_key(3, (x1, y1, x2, y2))
        # emit writes x,y,w,h at .2f; finalize parses and rebuilds xyxy
        x = float(f"{x1:.2f}")
        y = float(f"{y1:.2f}")
        w = float(f"{x2 - x1:.2f}")
        h = float(f"{y2 - y1:.2f}")
        assert _cache_key(3, (x, y, x + w, y + h)) == live_key


def test_crop_ring_evict_frees_slots():
    ring = TrackCropRing(capacity=4, crop_hw=(2, 2), depth=4, device="cpu")
    ring.stash(uid=5, frame=0, crop=_crop(0.0))
    ring.stash(uid=5, frame=1, crop=_crop(1.0))
    assert len(ring) == 2
    ring.evict(5)
    assert ring.gather(5) is None
    assert len(ring) == 0
    # slots are reusable after eviction
    ring.stash(uid=6, frame=0, crop=_crop(9.0))
    assert ring.has(6)


def test_crop_ring_batch_stash_matches_single():
    ring = TrackCropRing(capacity=8, crop_hw=(2, 2), depth=4, device="cpu")
    crops = torch.stack([_crop(float(i)) for i in range(3)])
    ring.stash_batch(uids=[1, 1, 2], frames=[0, 1, 0], crops=crops)
    g1 = ring.gather(1)
    g2 = ring.gather(2)
    assert g1 is not None and g2 is not None
    assert g1[1] == [0, 1]
    assert g2[1] == [0]
    assert torch.allclose(g2[0][0], _crop(2.0))


# --------------------------------------------------------------------------
# EmbeddingRing
# --------------------------------------------------------------------------
def test_embedding_ring_incremental_tail_and_stride():
    ring = EmbeddingRing(embed_dim=4, tail_n=6, head_n=3)
    for i in range(6):
        ring.append_tail(1, torch.full((4,), float(i)))
    # dense = full recent tail
    dense = ring.dense(1, n=6)
    assert dense is not None and dense.shape == (6, 4)
    # sparse strided bank preserves multiple samples (not a mean-1 prototype)
    bank = ring.bank(1, bank_n=3, bank_stride=2)
    assert bank is not None and bank.shape[0] == 3
    assert bank.shape[0] > 1  # guards against mean-collapse regression


def test_embedding_ring_head_freezes():
    ring = EmbeddingRing(embed_dim=4, tail_n=6, head_n=2)
    ring.append_head(1, torch.zeros(4))
    ring.append_head(1, torch.ones(4))
    ring.freeze_head(1)
    ring.append_head(1, torch.full((4,), 9.0))  # ignored after freeze
    head = ring.head(1)
    assert head is not None and head.shape[0] == 2


# --------------------------------------------------------------------------
# StreamingHandoverDriver
# --------------------------------------------------------------------------
def _feed_lines(driver: StreamingHandoverDriver, lines: list[str]) -> None:
    by_frame: dict[int, list[str]] = {}
    for ln in lines:
        fr = int(ln.split(",")[0])
        by_frame.setdefault(fr, []).append(ln)
    for fr in sorted(by_frame):
        active = [int(x.split(",")[1]) for x in by_frame[fr]]
        driver.observe_frame(fr, by_frame[fr], active)


def _handover_scenario(rng: np.random.Generator, d: int = 32):
    c0 = rng.standard_normal(d).astype(np.float32)
    c1 = rng.standard_normal(d).astype(np.float32)
    lines: list[str] = []
    for fr in range(1, 11):
        lines.append(f"{fr},1,10,10,20,40,0.9,-1,-1,-1")  # dies frame 10
    for fr in range(16, 31):
        lines.append(f"{fr},2,10,10,20,40,0.9,-1,-1,-1")  # newborn, same id
    for fr in range(1, 31):
        lines.append(f"{fr},3,200,10,20,40,0.9,-1,-1,-1")  # distractor
    head = {2: _normed(rng, 3, d, c0), 3: _normed(rng, 3, d, c1)}
    bank = {1: _normed(rng, 8, d, c0), 3: _normed(rng, 8, d, c1)}
    return lines, head, bank


def test_driver_lazy_gate_skips_until_newborn_matures():
    ring = EmbeddingRing(embed_dim=8, tail_n=8, head_n=5)
    driver = StreamingHandoverDriver(embeddings=ring, decide_n=5)
    # newborn born frame 16; nothing matured before 16+5.
    for fr in range(16, 20):
        driver.observe_frame(fr, [f"{fr},2,10,10,20,40,0.9,-1,-1,-1"], [2])
    assert driver.should_run(19) is False
    driver.observe_frame(21, ["21,2,10,10,20,40,0.9,-1,-1,-1"], [2])
    assert driver.should_run(21) is True


def test_driver_alias_matches_offline_decision():
    rng = np.random.default_rng(0)
    lines, head, bank = _handover_scenario(rng)

    # Reference: the validated offline decision core on the same inputs.
    _, ref_stats = causal_handover_lines(
        lines, head, bank, enabled=True, max_cost=0.9, max_fwd=0
    )
    assert ref_stats["handovers"] == 1

    ring = EmbeddingRing(embed_dim=32, tail_n=8, head_n=3)
    for uid, h in head.items():
        for row in h:
            ring.append_head(uid, row)
    for uid, b in bank.items():
        for row in b:
            ring.append_tail(uid, row)
    driver = StreamingHandoverDriver(
        embeddings=ring,
        decide_n=5,
        handover_kwargs={"max_cost": 0.9, "max_fwd": 0},
    )
    _feed_lines(driver, lines)
    assert driver.run(31, force=True) is True
    alias = driver.finalize(31)
    assert alias == {2: 1}
    assert driver.resolve(2) == 1
    assert driver.resolve(3) == 3
    assert driver.stats["handovers"] == 1


def test_driver_async_pass_produces_same_alias():
    rng = np.random.default_rng(0)
    lines, head, bank = _handover_scenario(rng)
    ring = EmbeddingRing(embed_dim=32, tail_n=8, head_n=3)
    for uid, h in head.items():
        for row in h:
            ring.append_head(uid, row)
    for uid, b in bank.items():
        for row in b:
            ring.append_tail(uid, row)
    driver = StreamingHandoverDriver(
        embeddings=ring,
        decide_n=5,
        handover_kwargs={"max_cost": 0.9, "max_fwd": 0},
        async_pass=True,
    )
    _feed_lines(driver, lines)
    driver.run(31, force=True)
    alias = driver.finalize(31)  # joins the worker
    assert alias == {2: 1}
