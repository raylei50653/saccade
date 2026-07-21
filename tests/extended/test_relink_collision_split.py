"""Regression tests for the per-frame id-uniqueness guard in the relinker.

Online relink commits ``alias[raw]→canonical`` permanently. When the *original*
canonical id re-appears after a long absence while a relinked alias is still
live, both raw tracks fast-path to the same canonical and would emit a duplicate
id in a single timestep (TrackEval rejects such output). The guard splits the
losing raw track onto a fresh surrogate id; with bidirectional relink the
higher-confidence incumbent keeps the canonical and the low-confidence fragment
is the one that splits.
"""

# scope: eval
# function: regression
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.relink import PythonSemanticRelinker  # noqa: E402

_BOX = torch.tensor([100.0, 100.0, 140.0, 200.0])
_EMB = torch.ones(4)  # non-None so resolve() takes the main path, not motion-only
_W, _H = 1920, 1080
_SPLIT_BASE = 1_000_000


def _seed_committed_alias(relinker: PythonSemanticRelinker) -> None:
    """Alias 100→50 committed earlier; native 50 maps to itself. Both are past
    first-claim so resolve() takes the fast-path (no matching)."""
    relinker.alias = {50: 50, 100: 50}
    relinker.features = {50: torch.zeros(1)}
    relinker.last_seen = {50: 0}
    relinker.last_boxes = {50: _BOX}


def test_fastpath_collision_splits_to_surrogate() -> None:
    relinker = PythonSemanticRelinker(sim_threshold=0.0, ttl=120)
    _seed_committed_alias(relinker)

    out = relinker.resolve_many_packed(
        [100, 50],
        [_EMB, _EMB],
        [_BOX, _BOX],
        [0.9, 0.1],
        frame_id=200,
        w=_W,
        h=_H,
    )

    # No duplicate id in the frame, and the loser got a surrogate id.
    assert len(set(out)) == 2, out
    assert out[0] == 50  # incumbent alias emitted first keeps the canonical
    assert out[1] >= _SPLIT_BASE  # native 50 re-appearing splits off
    assert relinker.stats["relink_split_collision"] == 1


def test_bidirectional_keeps_high_confidence_incumbent() -> None:
    relinker = PythonSemanticRelinker(sim_threshold=0.0, ttl=120, bidirectional=True)
    _seed_committed_alias(relinker)

    # Detector order: native 50 (low conf) first, alias 100 (high conf) second.
    # Score-descending processing must let the high-conf track keep id 50.
    out = relinker.resolve_many_packed(
        [50, 100],
        [_EMB, _EMB],
        [_BOX, _BOX],
        [0.1, 0.9],
        frame_id=200,
        w=_W,
        h=_H,
    )

    assert len(set(out)) == 2, out
    assert out[1] == 50  # high-confidence alias keeps the canonical id
    assert out[0] >= _SPLIT_BASE  # low-confidence native fragment splits


def test_split_is_stable_across_frames() -> None:
    relinker = PythonSemanticRelinker(sim_threshold=0.0, ttl=120)
    _seed_committed_alias(relinker)

    first = relinker.resolve_many_packed(
        [100, 50], [_EMB, _EMB], [_BOX, _BOX], [0.9, 0.1], frame_id=200, w=_W, h=_H
    )
    second = relinker.resolve_many_packed(
        [100, 50], [_EMB, _EMB], [_BOX, _BOX], [0.9, 0.1], frame_id=201, w=_W, h=_H
    )

    # The split fragment keeps the same surrogate id on the next frame and does
    # not trigger a second split.
    assert second == first, (first, second)
    assert relinker.stats["relink_split_collision"] == 1


def test_no_collision_leaves_ids_untouched() -> None:
    relinker = PythonSemanticRelinker(sim_threshold=0.0, ttl=120)
    relinker.alias = {10: 10, 20: 20}
    relinker.features = {10: torch.zeros(1), 20: torch.zeros(1)}
    relinker.last_seen = {10: 0, 20: 0}
    relinker.last_boxes = {10: _BOX, 20: _BOX}

    out = relinker.resolve_many_packed(
        [10, 20], [_EMB, _EMB], [_BOX, _BOX], [0.9, 0.8], frame_id=5, w=_W, h=_H
    )

    assert out == [10, 20]
    assert relinker.stats["relink_split_collision"] == 0
