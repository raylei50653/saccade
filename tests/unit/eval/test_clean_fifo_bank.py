"""Unit tests for CleanFifoBank — the reusable clean-FIFO embedding substrate."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from saccade.perception.eval.clean_fifo_bank import (
    CleanFifoBank,
    plan_clean_fifo_crops,
)
from saccade.perception.eval.post_merge import _parse_mot_lines


def _emb(seed: int, dim: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(dim, generator=g), dim=0)


def _mot_line(frame: int, tid: int, x: float = 0.0, y: float = 0.0) -> str:
    return f"{frame},{tid},{x},{y},10,20,0.9,-1,-1,-1"


class TestCleanFifoBankContainer:
    def test_store_and_samples(self):
        bank = CleanFifoBank(fifo_n=3)
        e1, e2 = _emb(1), _emb(2)
        bank.store(1, e1, frame_id=10)
        bank.store(1, e2, frame_id=20)
        out = bank.samples(1)
        assert out is not None
        assert out.shape == (2, 8)
        torch.testing.assert_close(out[0], F.normalize(e1, dim=0))
        torch.testing.assert_close(out[1], F.normalize(e2, dim=0))

    def test_fifo_eviction_keeps_most_recent(self):
        bank = CleanFifoBank(fifo_n=2)
        for i in range(5):
            bank.store(1, _emb(i), frame_id=i * 10)
        out = bank.samples(1)
        assert out is not None
        assert out.shape == (2, 8)
        assert bank.frames(1) == [30, 40]

    def test_samples_before_frame(self):
        bank = CleanFifoBank(fifo_n=10)
        bank.store(1, _emb(1), frame_id=10)
        bank.store(1, _emb(2), frame_id=20)
        bank.store(1, _emb(3), frame_id=30)
        pre = bank.samples_before(1, 25)
        assert pre is not None
        assert pre.shape == (2, 8)
        pre_none = bank.samples_before(1, 5)
        assert pre_none is None

    def test_samples_empty_track(self):
        bank = CleanFifoBank()
        assert bank.samples(99) is None
        assert bank.representative(99) is None
        assert bank.count(99) == 0

    def test_representative_is_mean(self):
        bank = CleanFifoBank(fifo_n=5)
        bank.store(1, _emb(1), frame_id=1)
        bank.store(1, _emb(2), frame_id=2)
        rep = bank.representative(1)
        assert rep is not None
        expected = F.normalize(
            (F.normalize(_emb(1), dim=0) + F.normalize(_emb(2), dim=0)) / 2,
            dim=0,
        )
        torch.testing.assert_close(rep, expected, rtol=1e-5, atol=1e-6)

    def test_clean_ids_and_count(self):
        bank = CleanFifoBank(fifo_n=5)
        bank.store(1, _emb(1), frame_id=1)
        bank.store(2, _emb(2), frame_id=1)
        bank.store(2, _emb(3), frame_id=2)
        assert bank.clean_ids() == {1, 2}
        assert bank.count(1) == 1
        assert bank.count(2) == 2
        assert bank.count(99) == 0

    def test_prune(self):
        bank = CleanFifoBank(fifo_n=5)
        bank.store(1, _emb(1), frame_id=1)
        bank.store(2, _emb(2), frame_id=1)
        bank.store(3, _emb(3), frame_id=1)
        bank.prune({1, 3})
        assert bank.clean_ids() == {1, 3}
        assert bank.samples(2) is None


class TestCleanFifoBankScheduling:
    def test_update_birth_window_always_accepts(self):
        bank = CleanFifoBank(fifo_n=10, stride=3, decide_n=5)
        for offset in range(5):
            stored = bank.update(
                1,
                _emb(offset),
                frame_id=offset + 1,
                is_clean=True,
                frame_offset_from_birth=offset,
            )
            assert stored, f"birth frame offset={offset} should be stored"
        assert bank.count(1) == 5

    def test_update_stride_after_birth(self):
        bank = CleanFifoBank(fifo_n=20, stride=3, decide_n=5)
        stored_frames = []
        for offset in range(20):
            stored = bank.update(
                1,
                _emb(offset),
                frame_id=offset + 1,
                is_clean=True,
                frame_offset_from_birth=offset,
            )
            if stored:
                stored_frames.append(offset)
        birth = stored_frames[:5]
        post = stored_frames[5:]
        assert len(birth) == 5
        # stride=3 counts all clean frames from birth (including birth window),
        # so first post-birth accept is at seen=6 (offset 6), then 9, 12, ...
        assert post == [6, 9, 12, 15, 18]

    def test_update_skips_dirty(self):
        bank = CleanFifoBank(fifo_n=10, stride=1, decide_n=5)
        stored = bank.update(
            1, _emb(1), frame_id=1, is_clean=False, frame_offset_from_birth=0
        )
        assert not stored
        assert bank.count(1) == 0

    def test_clean_seen_tracks_all_clean_frames(self):
        bank = CleanFifoBank(fifo_n=20, stride=3, decide_n=2)
        for offset in range(10):
            bank.update(
                1,
                _emb(offset),
                frame_id=offset,
                is_clean=True,
                frame_offset_from_birth=offset,
            )
        assert bank.clean_seen(1) == 10


class TestPlanCleanFifoCrops:
    def test_basic_head_and_bank(self):
        lines = [
            _mot_line(1, 1),
            _mot_line(2, 1),
            _mot_line(3, 1),
            _mot_line(4, 1),
            _mot_line(5, 1),
            _mot_line(6, 1),
        ]
        records = _parse_mot_lines(lines)
        plan = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.99,
            fifo_n=3,
            stride=1,
            decide_n=5,
        )
        assert set(plan.head_crops) == {1}
        assert len(plan.head_crops[1]) == 5
        assert len(plan.bank_crops[1]) == 3
        bank_frames = [f for f, _ in plan.bank_crops[1]]
        assert bank_frames == [4, 5, 6]

    def test_fifo_n_caps_bank(self):
        lines = [_mot_line(f, 1) for f in range(1, 11)]
        records = _parse_mot_lines(lines)
        plan = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.99,
            fifo_n=4,
            stride=1,
            decide_n=3,
        )
        assert len(plan.bank_crops[1]) == 4
        bank_frames = [f for f, _ in plan.bank_crops[1]]
        assert bank_frames == [7, 8, 9, 10]

    def test_stride_reduces_extraction(self):
        lines = [_mot_line(f, 1) for f in range(1, 21)]
        records = _parse_mot_lines(lines)
        plan = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.99,
            fifo_n=20,
            stride=3,
            decide_n=5,
        )
        bank_frames = [f for f, _ in plan.bank_crops[1]]
        birth_frames = bank_frames[:5]
        post_frames = bank_frames[5:]
        assert len(birth_frames) == 5
        # stride=3: frames 1-5 (birth), then every 3rd clean frame after
        # seen counts from 0; first post-birth accept at seen=6 (frame 7)
        assert post_frames == [7, 10, 13, 16, 19]

    def test_neighbor_iou_pollutes_head(self):
        lines = [
            "1,1,0,0,10,10,0.9,-1,-1,-1",
            "1,2,2,0,10,10,0.9,-1,-1,-1",
            "2,1,20,0,10,10,0.9,-1,-1,-1",
            "2,2,22,0,10,10,0.9,-1,-1,-1",
        ]
        records = _parse_mot_lines(lines)
        plan_clean = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.99,
            neighbor_iou_max=0.0,
        )
        plan_polluted = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.99,
            neighbor_iou_max=0.5,
        )
        assert len(plan_polluted.head_crops[1]) < len(plan_clean.head_crops[1])
        assert len(plan_polluted.head_crops[2]) < len(plan_clean.head_crops[2])

    def test_bank_falls_back_when_pollution_empties(self):
        lines = [
            "1,1,0,0,10,10,0.9,-1,-1,-1",
            "1,2,2,0,10,10,0.9,-1,-1,-1",
            "2,1,2,0,10,10,0.9,-1,-1,-1",
            "2,2,2,0,10,10,0.9,-1,-1,-1",
        ]
        records = _parse_mot_lines(lines)
        plan = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.99,
            neighbor_iou_max=0.5,
        )
        assert plan.head_crops[1] == []
        assert plan.head_crops[2] == []
        assert len(plan.bank_crops[1]) == 2
        assert len(plan.bank_crops[2]) == 2

    def test_visclean_dirty_excluded(self):
        lines = [
            "1,1,0,0,10,20,0.9,-1,-1,-1",
            "1,2,0,5,10,20,0.9,-1,-1,-1",
            "2,1,0,0,10,20,0.9,-1,-1,-1",
            "2,2,0,5,10,20,0.9,-1,-1,-1",
        ]
        records = _parse_mot_lines(lines)
        plan = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.3,
        )
        # Track 1 (y2=20) is covered by track 2 (y2=25) → all dirty.
        assert 1 not in plan.head_crops or len(plan.head_crops[1]) == 0
        # Track 2 is in front → clean, has head + bank crops.
        assert len(plan.head_crops[2]) == 2
        assert len(plan.bank_crops[2]) == 2

    def test_fill_from_completes_fifo(self):
        lines = [_mot_line(f, 1) for f in range(1, 11)]
        records = _parse_mot_lines(lines)
        plan = plan_clean_fifo_crops(
            records,
            appearance_occlusion_cov=0.99,
            fifo_n=3,
            stride=1,
            decide_n=5,
        )
        crops = plan.bank_crops[1]
        fake_embs = torch.stack([_emb(f) for f, _ in crops])
        plan.bank.fill_from({1: crops}, {1: fake_embs})
        out = plan.bank.samples(1)
        assert out is not None
        assert out.shape[0] == 3
        assert plan.bank.frames(1) == [f for f, _ in crops]
