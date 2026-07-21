"""Tests for the evaluator double-buffer path (perception.eval.evaluator)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from saccade.perception.eval.evaluator import (
    _double_buffer_eligible,
    _launch_double_buffer_detect,
)


class _Detector:
    _temporal_T = 0


def test_double_buffer_requires_explicit_narrow_barrier_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("SACCADE_DOUBLE_BUFFER", "1")
    monkeypatch.setenv("SACCADE_DETECT_BARRIER", "event")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert _double_buffer_eligible(SimpleNamespace(workbench=False), _Detector(), False)


def test_double_buffer_rejects_paths_that_cannot_preserve_frame_independence(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SACCADE_DOUBLE_BUFFER", "1")
    monkeypatch.setenv("SACCADE_DETECT_BARRIER", "event")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert not _double_buffer_eligible(
        SimpleNamespace(workbench=True), _Detector(), False
    )
    assert not _double_buffer_eligible(
        SimpleNamespace(workbench=False), _Detector(), True
    )
    assert not _double_buffer_eligible(
        SimpleNamespace(workbench=False), SimpleNamespace(_temporal_T=3), False
    )
    assert _double_buffer_eligible(
        SimpleNamespace(workbench=False),
        SimpleNamespace(_temporal_T=3, use_whole_graph=True),
        False,
    )


def test_double_buffer_rejects_full_device_detect_barrier(monkeypatch) -> None:
    monkeypatch.setenv("SACCADE_DOUBLE_BUFFER", "1")
    monkeypatch.setenv("SACCADE_DETECT_BARRIER", "full")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert not _double_buffer_eligible(
        SimpleNamespace(workbench=False), _Detector(), False
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_double_buffer_clones_reused_detector_output_before_next_replay() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.use_nv12 = False
            self.frame_buffer = torch.zeros((3, 4, 4), device="cuda")
            self.frame_buffer_nv12 = torch.empty(0, device="cuda", dtype=torch.uint8)

        def mark_rgb_current(self) -> None:
            pass

    # Emulates a whole-graph detector whose next replay overwrites its output.
    shared_boxes = torch.empty((1, 4), device="cuda")
    shared_scores = torch.empty((1,), device="cuda")
    shared_classes = torch.empty((1,), device="cuda")

    def detect_fn(_detector, pool, *_args):
        marker = pool.frame_buffer[0, 0, 0]
        shared_boxes.fill_(marker)
        shared_scores.fill_(marker)
        shared_classes.fill_(1)
        return shared_boxes, shared_scores, shared_classes, False, None

    def time_stage(_totals, _name, fn, sync_cuda=False):
        del sync_cuda
        return fn(), 0.0

    state = SimpleNamespace(
        _frame_stage_times=None,
        double_buffer_stream=torch.cuda.Stream(),
        # Nested detection view matches EvalConfig module-view shape used by stages.py.
        cfg=SimpleNamespace(
            preprocess_modes=[],
            detection=SimpleNamespace(
                gamma=1.0, gamma_luma_threshold=0.0, contrast=1.0
            ),
        ),
        detector=object(),
        h_orig=4,
        w_orig=4,
        seq_stage_totals={},
        time_stage=time_stage,
        nv12_direct_from_hwc=False,
        detect_fn=detect_fn,
        detector_box_format="xyxy",
    )
    first = _launch_double_buffer_detect(
        state,
        frame_id=1,
        pool=_Pool(),
        frame_gpu=torch.full((4, 4, 3), 10, device="cuda", dtype=torch.uint8),
        input_ready=torch.cuda.Event(enable_timing=False),
        ready_event=torch.cuda.Event(enable_timing=False),
        latency_started_at=0.0,
    )
    second = _launch_double_buffer_detect(
        state,
        frame_id=2,
        pool=_Pool(),
        frame_gpu=torch.full((4, 4, 3), 20, device="cuda", dtype=torch.uint8),
        input_ready=torch.cuda.Event(enable_timing=False),
        ready_event=torch.cuda.Event(enable_timing=False),
        latency_started_at=0.0,
    )
    main = torch.cuda.current_stream()
    main.wait_event(first.ready_event)
    main.wait_event(second.ready_event)
    torch.cuda.synchronize()

    assert first.fused_boxes[0, 0].item() == pytest.approx(10 / 255.0)
    assert second.fused_boxes[0, 0].item() == pytest.approx(20 / 255.0)
