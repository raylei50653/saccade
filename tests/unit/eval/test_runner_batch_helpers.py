"""Tests for eval runner batch helpers (perception.eval relink/lifecycle/output_bank/helpers)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.relink import IdentityResolver  # noqa: E402
from saccade.perception.eval.lifecycle import IdStabilityFilter, TrackletLifecycleMerger  # noqa: E402
from saccade.perception.eval.output_bank import OutputAppearanceBank  # noqa: E402
from saccade.perception.eval.types import (  # noqa: E402
    HostTrackBatch,
    PreparedTrackCandidate,
    ResolvedTrack,
)
from saccade.perception.eval.helpers import (  # noqa: E402
    emit_resolved_tracks as _emit_resolved_tracks,
    finalize_frame_side_effects as _finalize_frame_side_effects,
    front_occlusion_mask_xyxy,
    inject_lost_track_references as _inject_lost_track_references,
    prepare_track_candidates as _prepare_track_candidates,
    resolve_frame_tracks as _resolve_frame_tracks,
)
from saccade.perception.tracking.tracker_gpu import TrackAppearanceBank  # noqa: E402


def test_id_stability_accept_many_matches_sequential_accept() -> None:
    candidates = [
        (11, (0.0, 0.0, 10.0, 10.0), 0.90),
        (22, (20.0, 20.0, 32.0, 34.0), 0.85),
    ]
    batch_filter = IdStabilityFilter(
        min_hits=2,
        min_iou=0.5,
        max_center_shift=0.4,
        max_gap=1,
        score_ema=0.5,
        min_score_ema=0.6,
    )
    sequential_filter = IdStabilityFilter(
        min_hits=2,
        min_iou=0.5,
        max_center_shift=0.4,
        max_gap=1,
        score_ema=0.5,
        min_score_ema=0.6,
    )

    batch_first = batch_filter.accept_many(candidates, frame_id=1)
    sequential_first = [
        sequential_filter.accept(obj_id, box, score, frame_id=1)
        for obj_id, box, score in candidates
    ]
    assert batch_first == sequential_first == [False, False]

    next_candidates = [
        (11, (0.5, 0.5, 10.5, 10.5), 0.92),
        (22, (20.5, 20.5, 32.5, 34.5), 0.88),
    ]
    batch_second = batch_filter.accept_many(next_candidates, frame_id=2)
    sequential_second = [
        sequential_filter.accept(obj_id, box, score, frame_id=2)
        for obj_id, box, score in next_candidates
    ]

    assert batch_second == sequential_second == [True, True]
    assert batch_filter.states == sequential_filter.states


def test_id_stability_resets_tentative_track_after_gap() -> None:
    stability = IdStabilityFilter(
        min_hits=2,
        min_iou=0.5,
        max_center_shift=0.2,
        max_gap=1,
        score_ema=0.5,
        min_score_ema=0.6,
    )

    assert stability.accept(7, (0.0, 0.0, 10.0, 10.0), 0.9, frame_id=1) is False
    first_state = stability.states[7]
    assert first_state.stable_hits == 1
    assert first_state.total_hits == 1

    assert stability.accept(7, (0.2, 0.2, 10.2, 10.2), 0.9, frame_id=4) is False
    reset_state = stability.states[7]
    assert reset_state.frame_id == 4
    assert reset_state.stable_hits == 1
    assert reset_state.total_hits == 1


def test_id_stability_requires_score_ema_before_confirming() -> None:
    stability = IdStabilityFilter(
        min_hits=2,
        min_iou=0.5,
        max_center_shift=0.2,
        max_gap=1,
        score_ema=0.5,
        min_score_ema=0.6,
    )

    assert stability.accept(11, (0.0, 0.0, 10.0, 10.0), 0.4, frame_id=1) is False
    assert stability.accept(11, (0.1, 0.1, 10.1, 10.1), 0.5, frame_id=2) is False
    assert stability.states[11].stable_hits == 2
    assert stability.states[11].score_ema == pytest.approx(0.45)

    assert stability.accept(11, (0.2, 0.2, 10.2, 10.2), 0.9, frame_id=3) is True
    assert stability.states[11].score_ema == pytest.approx(0.675)


def test_output_appearance_bank_update_many_filters_and_truncates() -> None:
    bank = OutputAppearanceBank(
        max_samples=2,
        min_score=0.5,
        min_consistency=0.8,
    )

    bank.update_many(
        [
            (101, torch.tensor([1.0, 0.0], dtype=torch.float32), 0.70, 1),
            (101, torch.tensor([0.9, 0.1], dtype=torch.float32), 0.95, 2),
            (101, torch.tensor([0.8, 0.2], dtype=torch.float32), 0.80, 3),
            (202, None, 0.90, 1),
            (202, torch.tensor([0.0, 1.0], dtype=torch.float32), 0.40, 2),
        ]
    )

    assert bank.count(101) == 2
    assert bank.count(202) == 0
    representative = bank.representative(101)
    assert representative is not None
    assert torch.isclose(torch.linalg.norm(representative), torch.tensor(1.0))
    kept_scores = [score for score, _, _ in bank.samples[101]]
    assert kept_scores == [0.95, 0.8]


class _StubRelinker:
    def __init__(self) -> None:
        self.injected_many: list[list[tuple[int, torch.Tensor]]] = []

    def canonical_id(self, raw_id: int) -> int:
        return raw_id + 1000

    def has_feature(self, canonical_id: int) -> bool:
        return canonical_id != 1003

    def inject_references_many(
        self, references: list[tuple[int, torch.Tensor]]
    ) -> None:
        self.injected_many.append(references)


class _StubPackedResolveRelinker:
    def resolve_many_packed(
        self,
        raw_ids: list[int],
        embeddings: list[torch.Tensor | None],
        boxes: list[tuple[float, float, float, float]],
        scores: list[float],
        *,
        frame_id: int,
        w: int,
        h: int,
    ) -> list[int]:
        assert raw_ids == [11, 22]
        assert embeddings[0] is not None
        assert embeddings[1] is None
        assert boxes == [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)]
        assert scores == [0.9, 0.8]
        assert frame_id == 9
        assert w == 1280
        assert h == 720
        return [211, 222]


class _StubPackedLifecycleMerger:
    def resolve_many_packed(
        self,
        local_ids: list[int],
        boxes: list[tuple[float, float, float, float]],
        scores: list[float],
        embeddings: list[torch.Tensor | None],
        *,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> list[int]:
        assert local_ids == [211, 222]
        assert boxes == [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)]
        assert scores == [0.9, 0.8]
        assert embeddings[0] is not None
        assert embeddings[1] is None
        assert frame_id == 9
        assert frame_w == 1280
        assert frame_h == 720
        return [311, 322]


class _StubDynamicReID:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[int, object], torch.Tensor | None]] = []

    def observe(
        self, tracks: dict[int, object], gmc: torch.Tensor | None = None
    ) -> None:
        self.calls.append((tracks, gmc))


class _StubPrimaryAppearanceBank:
    def __init__(self) -> None:
        self.pruned: list[set[int]] = []

    def is_high_quality(self, track_id: int) -> bool:
        return False

    def high_quality_representative(self, track_id: int) -> torch.Tensor | None:
        return None

    def is_consistent(self, track_id: int) -> bool:
        return track_id == 1

    def representative(self, track_id: int) -> torch.Tensor | None:
        if track_id != 1:
            return None
        return torch.tensor([1.0, 0.0], dtype=torch.float32)

    def prune(self, active_ids: set[int]) -> None:
        self.pruned.append(set(active_ids))


class _FallbackLifecycleMerger:
    def __init__(self) -> None:
        self.calls: list[
            list[tuple[int, tuple[float, float, float, float], float, None]]
        ] = []

    def resolve_many(
        self,
        candidates: list[tuple[int, tuple[float, float, float, float], float, None]],
        *,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> list[int]:
        assert frame_id == 12
        assert frame_w == 640
        assert frame_h == 480
        self.calls.append(candidates)
        return [local_id + 500 for local_id, *_ in candidates]


class _CollectingAppearanceBank:
    def __init__(self) -> None:
        self.updated: list[tuple[int, torch.Tensor | None, float, int]] = []

    def update_many(
        self, updates: list[tuple[int, torch.Tensor | None, float, int]]
    ) -> None:
        self.updated.extend(updates)


class _Mapper:
    def map(self, seq: str, track_id: int) -> int:
        assert seq == "MOT17-02-SDP"
        return track_id + 10_000


class _ConsistencyBank:
    def __init__(self, consistent_ids: set[int]) -> None:
        self.consistent_ids = consistent_ids
        self.updated: list[
            tuple[
                int,
                torch.Tensor,
                float,
                float,
                int,
                bool,
                bool,
                float,
                float,
                tuple[float, float, float, float],
            ]
        ] = []

    def update_many(
        self,
        updates: list[
            tuple[
                int,
                torch.Tensor,
                float,
                float,
                int,
                bool,
                bool,
                float,
                float,
                tuple[float, float, float, float],
            ]
        ],
    ) -> None:
        self.updated.extend(updates)

    def is_consistent(self, track_id: int) -> bool:
        return track_id in self.consistent_ids


def test_front_occlusion_mask_xyxy_matches_visclean_lower_foot_rule() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 5.0, 10.0, 15.0],
            [30.0, 0.0, 40.0, 10.0],
        ],
        dtype=torch.float32,
    )

    mask = front_occlusion_mask_xyxy(boxes, 0.4)

    assert mask.tolist() == [True, False, False]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for TrackAppearanceBank"
)
def test_inject_lost_track_references_batches_consistent_tracks() -> None:
    relinker = _StubRelinker()
    bank = TrackAppearanceBank(
        k=2,
        min_score=0.5,
        min_iou=0.3,
        consistency_threshold=0.8,
    )
    bank.update_many(
        [
            (
                1,
                torch.tensor([1.0, 0.0], dtype=torch.float32),
                0.95,
                0.9,
                1,
                True,
                False,
                0.0,
                0.0,
            ),
            (
                1,
                torch.tensor([0.98, 0.02], dtype=torch.float32),
                0.90,
                0.85,
                2,
                True,
                False,
                0.0,
                0.0,
            ),
            (
                2,
                torch.tensor([0.0, 1.0], dtype=torch.float32),
                0.93,
                0.9,
                1,
                True,
                False,
                0.0,
                0.0,
            ),
            (
                2,
                torch.tensor([0.0, -1.0], dtype=torch.float32),
                0.92,
                0.85,
                2,
                True,
                False,
                0.0,
                0.0,
            ),
            (
                3,
                torch.tensor([1.0, 1.0], dtype=torch.float32),
                0.96,
                0.9,
                1,
                True,
                False,
                0.0,
                0.0,
            ),
            (
                3,
                torch.tensor([1.0, 1.0], dtype=torch.float32),
                0.91,
                0.85,
                2,
                True,
                False,
                0.0,
                0.0,
            ),
        ]
    )

    _inject_lost_track_references(
        relinker=relinker,
        primary_appearance_bank=bank,
        prev_track_ids={1, 2, 3, 4},
        curr_track_ids={4},
    )

    assert len(relinker.injected_many) == 1
    injected = relinker.injected_many[0]
    assert [canonical_id for canonical_id, _ in injected] == [1001]
    assert torch.isclose(torch.linalg.norm(injected[0][1]), torch.tensor(1.0))


def test_finalize_frame_side_effects_runs_batch_observe_inject_and_prune() -> None:
    relinker = _StubRelinker()
    primary_bank = _StubPrimaryAppearanceBank()
    dynamic_reid = _StubDynamicReID()
    person_observations = {7: object()}
    gmc = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=torch.float32)

    next_prev_track_ids = _finalize_frame_side_effects(
        curr_track_ids={4},
        prev_track_ids={1, 4},
        relinker=relinker,
        semantic_bank_inject=True,
        primary_appearance_bank=primary_bank,
        dynamic_reid=dynamic_reid,
        person_observations=person_observations,
        gmc_warp=gmc,
        gmc_enabled=True,
    )

    assert next_prev_track_ids == {4}
    assert len(relinker.injected_many) == 1
    assert [canonical_id for canonical_id, _ in relinker.injected_many[0]] == [1001]
    assert primary_bank.pruned == [{4}]
    assert len(dynamic_reid.calls) == 1
    assert dynamic_reid.calls[0][0] is person_observations
    assert torch.equal(dynamic_reid.calls[0][1], gmc)


def test_tracklet_lifecycle_resolve_many_matches_sequential_resolve() -> None:
    batch_merger = TrackletLifecycleMerger(
        enabled=True,
        ttl=10,
        min_gap=1,
        spatial_gate=0.3,
        min_iou=0.1,
        sim_threshold=0.8,
        require_embedding=False,
        ema=0.5,
    )
    sequential_merger = TrackletLifecycleMerger(
        enabled=True,
        ttl=10,
        min_gap=1,
        spatial_gate=0.3,
        min_iou=0.1,
        sim_threshold=0.8,
        require_embedding=False,
        ema=0.5,
    )

    seed_candidates = [
        (11, (0.0, 0.0, 10.0, 10.0), 0.9, torch.tensor([1.0, 0.0])),
        (22, (20.0, 20.0, 30.0, 30.0), 0.85, torch.tensor([0.0, 1.0])),
    ]
    assert batch_merger.resolve_many(
        seed_candidates,
        frame_id=1,
        frame_w=100,
        frame_h=100,
    ) == [11, 22]
    assigned_outputs: set[int] = set()
    sequential_seed = [
        sequential_merger.resolve(
            local_id,
            box,
            score,
            1,
            100,
            100,
            embedding,
            assigned_outputs,
        )
        for local_id, box, score, embedding in seed_candidates
    ]
    assert sequential_seed == [11, 22]

    next_candidates = [
        (31, (0.5, 0.5, 10.5, 10.5), 0.92, torch.tensor([0.99, 0.01])),
        (44, (20.3, 20.3, 30.3, 30.3), 0.83, torch.tensor([0.01, 0.99])),
    ]
    batch_resolved = batch_merger.resolve_many(
        next_candidates,
        frame_id=2,
        frame_w=100,
        frame_h=100,
    )
    assigned_outputs = set()
    sequential_resolved = [
        sequential_merger.resolve(
            local_id,
            box,
            score,
            2,
            100,
            100,
            embedding,
            assigned_outputs,
        )
        for local_id, box, score, embedding in next_candidates
    ]

    assert batch_resolved == sequential_resolved == [11, 22]
    assert batch_merger.alias == sequential_merger.alias
    assert set(batch_merger.states) == set(sequential_merger.states)


def test_tracklet_lifecycle_resolve_many_packed_matches_sequential_resolve() -> None:
    batch_merger = TrackletLifecycleMerger(
        enabled=True,
        ttl=10,
        min_gap=1,
        spatial_gate=0.3,
        min_iou=0.1,
        sim_threshold=0.8,
        require_embedding=False,
        ema=0.5,
    )
    sequential_merger = TrackletLifecycleMerger(
        enabled=True,
        ttl=10,
        min_gap=1,
        spatial_gate=0.3,
        min_iou=0.1,
        sim_threshold=0.8,
        require_embedding=False,
        ema=0.5,
    )

    batch_merger.resolve_many_packed(
        [11, 22],
        [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)],
        [0.9, 0.85],
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        frame_id=1,
        frame_w=100,
        frame_h=100,
    )
    assigned_outputs: set[int] = set()
    for local_id, box, score, embedding in [
        (11, (0.0, 0.0, 10.0, 10.0), 0.9, torch.tensor([1.0, 0.0])),
        (22, (20.0, 20.0, 30.0, 30.0), 0.85, torch.tensor([0.0, 1.0])),
    ]:
        sequential_merger.resolve(
            local_id, box, score, 1, 100, 100, embedding, assigned_outputs
        )

    batch_resolved = batch_merger.resolve_many_packed(
        [31, 44],
        [(0.5, 0.5, 10.5, 10.5), (20.3, 20.3, 30.3, 30.3)],
        [0.92, 0.83],
        [torch.tensor([0.99, 0.01]), torch.tensor([0.01, 0.99])],
        frame_id=2,
        frame_w=100,
        frame_h=100,
    )
    assigned_outputs = set()
    sequential_resolved = [
        sequential_merger.resolve(
            local_id, box, score, 2, 100, 100, embedding, assigned_outputs
        )
        for local_id, box, score, embedding in [
            (31, (0.5, 0.5, 10.5, 10.5), 0.92, torch.tensor([0.99, 0.01])),
            (44, (20.3, 20.3, 30.3, 30.3), 0.83, torch.tensor([0.01, 0.99])),
        ]
    ]

    assert batch_resolved == sequential_resolved == [11, 22]
    assert batch_merger.alias == sequential_merger.alias


def test_tracklet_lifecycle_reuses_lost_track_within_ttl_once_per_frame() -> None:
    merger = TrackletLifecycleMerger(
        enabled=True,
        ttl=5,
        min_gap=2,
        spatial_gate=0.4,
        min_iou=0.1,
        sim_threshold=0.5,
        require_embedding=False,
        ema=0.5,
    )

    assert merger.resolve_many(
        [(10, (0.0, 0.0, 10.0, 10.0), 0.9, None)],
        frame_id=1,
        frame_w=100,
        frame_h=100,
    ) == [10]

    resolved = merger.resolve_many(
        [
            (20, (0.5, 0.5, 10.5, 10.5), 0.85, None),
            (30, (0.6, 0.6, 10.6, 10.6), 0.80, None),
        ],
        frame_id=3,
        frame_w=100,
        frame_h=100,
    )

    assert resolved == [10, 30]
    assert merger.alias[20] == 10
    assert merger.alias[30] == 30
    assert merger.stats["accepted"] == 1
    assert merger.stats["reject_assigned"] >= 1


def test_tracklet_lifecycle_prune_evicts_expired_lost_track() -> None:
    merger = TrackletLifecycleMerger(
        enabled=True,
        ttl=2,
        min_gap=1,
        spatial_gate=0.4,
        min_iou=0.1,
        sim_threshold=0.5,
        require_embedding=False,
        ema=0.5,
    )

    assert merger.resolve_many(
        [(10, (0.0, 0.0, 10.0, 10.0), 0.9, None)],
        frame_id=1,
        frame_w=100,
        frame_h=100,
    ) == [10]

    merger.prune(frame_id=4)
    assert merger.states == {}

    assert merger.resolve_many(
        [(20, (0.2, 0.2, 10.2, 10.2), 0.8, None)],
        frame_id=4,
        frame_w=100,
        frame_h=100,
    ) == [20]
    assert merger.stats["new_ids"] == 2


def test_resolve_frame_tracks_falls_back_to_resolve_many() -> None:
    lifecycle = _FallbackLifecycleMerger()

    resolved = _resolve_frame_tracks(
        frame_id=12,
        frame_w=640,
        frame_h=480,
        prepared_candidates=[
            PreparedTrackCandidate(
                local_track_id=1,
                box=(1.0, 2.0, 11.0, 22.0),
                score=0.9,
                embedding=torch.tensor([1.0, 0.0], dtype=torch.float32),
            ),
            PreparedTrackCandidate(
                local_track_id=2,
                box=(30.0, 40.0, 50.0, 80.0),
                score=0.7,
                embedding=None,
            ),
        ],
        lifecycle_merger=lifecycle,
    )

    assert [track.resolved_track_id for track in resolved] == [501, 502]
    assert lifecycle.calls == [
        [
            (1, (1.0, 2.0, 11.0, 22.0), 0.9, None),
            (2, (30.0, 40.0, 50.0, 80.0), 0.7, None),
        ]
    ]


def test_resolve_frame_tracks_uses_identity_resolver() -> None:
    relinker = _StubPackedResolveRelinker()
    lifecycle = _StubPackedLifecycleMerger()
    resolved = _resolve_frame_tracks(
        frame_id=9,
        frame_w=1280,
        frame_h=720,
        prepared_candidates=[
            PreparedTrackCandidate(
                local_track_id=11,
                box=(1.0, 2.0, 3.0, 4.0),
                score=0.9,
                embedding=torch.tensor([1.0, 0.0], dtype=torch.float32),
            ),
            PreparedTrackCandidate(
                local_track_id=22,
                box=(5.0, 6.0, 7.0, 8.0),
                score=0.8,
                embedding=None,
            ),
        ],
        lifecycle_merger=lifecycle,
        identity_resolver=IdentityResolver(relinker, lifecycle),
    )

    assert [track.resolved_track_id for track in resolved] == [311, 322]


def test_prepare_track_candidates_holds_tentative_until_stable() -> None:
    track_results = {
        "count": 1,
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
        "scores": torch.tensor([0.9], dtype=torch.float32),
        "ids": torch.tensor([42], dtype=torch.int32),
        "classes": torch.tensor([0], dtype=torch.int32),
        "det_idx": torch.tensor([0], dtype=torch.int32),
    }
    host_batch = HostTrackBatch(
        boxes_gpu=track_results["boxes"],
        boxes=[(0.0, 0.0, 10.0, 10.0)],
        scores=[0.9],
        ids=[42],
        classes=[0],
        det_idx=[0],
        person_observations=None,
    )
    stability = IdStabilityFilter(
        min_hits=2,
        min_iou=0.5,
        max_center_shift=0.2,
        max_gap=1,
        score_ema=0.5,
        min_score_ema=0.6,
    )
    bank = _ConsistencyBank(consistent_ids={42})

    first = _prepare_track_candidates(
        frame_id=1,
        track_results=track_results,
        host_batch=host_batch,
        person_class=0,
        track_person_only=False,
        geometry_suspect_support=False,
        geometry_suspect_support_score=0.0,
        id_stability_filter=stability,
        embeddings=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        fused_boxes=track_results["boxes"],
        fused_scores=track_results["scores"],
        geometry_suspect_mask=torch.tensor([False]),
        primary_appearance_bank=bank,
    )
    assert first == []
    assert bank.updated == []

    second = _prepare_track_candidates(
        frame_id=2,
        track_results=track_results,
        host_batch=host_batch,
        person_class=0,
        track_person_only=False,
        geometry_suspect_support=False,
        geometry_suspect_support_score=0.0,
        id_stability_filter=stability,
        embeddings=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        fused_boxes=track_results["boxes"],
        fused_scores=track_results["scores"],
        geometry_suspect_mask=torch.tensor([False]),
        primary_appearance_bank=bank,
    )

    assert len(second) == 1
    assert second[0].local_track_id == 42
    assert second[0].embedding is not None
    assert len(bank.updated) == 1
    assert bank.updated[0][0] == 42


def test_prepare_track_candidates_drops_inconsistent_embedding_after_update() -> None:
    track_results = {
        "count": 1,
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
        "scores": torch.tensor([0.9], dtype=torch.float32),
        "ids": torch.tensor([7], dtype=torch.int32),
        "classes": torch.tensor([0], dtype=torch.int32),
        "det_idx": torch.tensor([0], dtype=torch.int32),
    }
    host_batch = HostTrackBatch(
        boxes_gpu=track_results["boxes"],
        boxes=[(0.0, 0.0, 10.0, 10.0)],
        scores=[0.9],
        ids=[7],
        classes=[0],
        det_idx=[0],
        person_observations=None,
    )
    bank = _ConsistencyBank(consistent_ids=set())

    prepared = _prepare_track_candidates(
        frame_id=5,
        track_results=track_results,
        host_batch=host_batch,
        person_class=0,
        track_person_only=False,
        geometry_suspect_support=False,
        geometry_suspect_support_score=0.0,
        id_stability_filter=None,
        embeddings=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        fused_boxes=track_results["boxes"],
        fused_scores=track_results["scores"],
        geometry_suspect_mask=torch.tensor([False]),
        primary_appearance_bank=bank,
    )

    assert len(prepared) == 1
    assert prepared[0].local_track_id == 7
    assert prepared[0].embedding is None
    assert len(bank.updated) == 1
    assert bank.updated[0][0] == 7


def test_emit_resolved_tracks_updates_output_bank_and_formats_lines() -> None:
    output_bank = _CollectingAppearanceBank()

    lines = _emit_resolved_tracks(
        seq="MOT17-02-SDP",
        frame_id=8,
        frame_w=640,
        frame_h=480,
        resolved_tracks=[
            ResolvedTrack(
                local_track_id=2,
                resolved_track_id=20,
                box=(10.0, 20.0, 40.0, 60.0),
                score=0.875,
                embedding=torch.tensor([1.0, 0.0], dtype=torch.float32),
            )
        ],
        global_id_mapper=_Mapper(),
        output_appearance_bank=output_bank,
    )

    assert lines == ["8,10020,10.00,20.00,30.00,40.00,0.8750,-1,-1,-1"]
    assert len(output_bank.updated) == 1
    assert output_bank.updated[0][0] == 10020
    assert torch.equal(
        output_bank.updated[0][1], torch.tensor([1.0, 0.0], dtype=torch.float32)
    )
    assert output_bank.updated[0][2:] == (0.875, 8)
