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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_explicit_post_waits_for_clones(monkeypatch) -> None:
    from saccade.perception.eval import stages

    monkeypatch.setenv("SACCADE_STREAM_MODE", "detect_post_event")
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    post = torch.cuda.Stream()
    raw = torch.ones((4,), device="cuda")
    frame = torch.zeros((2, 2, 3), device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()
    state = SimpleNamespace(
        double_buffer_stream=side,
        _pp_streams=[{"post": post}, {"post": post}],
        _pp_detect_done=[torch.cuda.Event(), torch.cuda.Event()],
        stream_detect=None,
        stream_detect_event=None,
        stream_post=post,
        nv12_direct_from_hwc=False,
        detect_fn=None,
        detector_box_format="xyxy",
    )
    monkeypatch.setattr(
        stages, "_run_detect", lambda *a, **kw: (raw, raw, raw, False, raw)
    )
    original_clone = torch.Tensor.clone
    delay_pending = True

    def delayed_clone(tensor, *args, **kwargs):
        nonlocal delay_pending
        if delay_pending:
            delay_pending = False
            # Widen the interval between detector completion and clone writes.
            torch.cuda._sleep(300_000_000)
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", delayed_clone)
    try:
        prepared = _launch_double_buffer_detect(
            state,
            frame_id=1,
            pool=None,
            frame_gpu=frame,
            input_ready=torch.cuda.Event(),
            ready_event=torch.cuda.Event(),
            latency_started_at=0.0,
        )
        # Match _run_frame: wait on main, then switch to the post stream.
        main.wait_event(prepared.ready_event)
        with torch.cuda.stream(post):
            reached = torch.cuda.Event()
            reached.record()
        reached.synchronize()
        assert prepared.ready_event.query(), "post acquired unfinished clones"
        with torch.cuda.stream(post):
            observed = torch.stack(
                [
                    prepared.fused_boxes,
                    prepared.fused_scores,
                    prepared.fused_classes,
                    prepared.source_keypoints,
                ]
            )
        post.synchronize()
        assert torch.equal(observed.cpu(), torch.ones((4, 4)))
    finally:
        torch.cuda.synchronize()


@pytest.mark.parametrize("frame_id,current_frame_id", [(1, 2), (2, 3), (3, 3)])
@pytest.mark.parametrize("background", [False, True])
def test_deferred_timing_counts_only_completed_output(
    monkeypatch, frame_id, current_frame_id, background
) -> None:
    from saccade.perception.eval import stages

    clock = [10.0]
    monkeypatch.setattr(stages.time, "perf_counter", lambda: clock[0])
    pinned = {
        key: torch.zeros((1,))
        for key in ("boxes", "scores", "ids", "classes", "det_idx")
    }
    pinned["count"] = torch.tensor(0)
    context = dict.fromkeys(
        (
            "tracker_result_buffers",
            "fused_boxes",
            "fused_scores",
            "geometry_suspect_mask",
            "embeddings",
            "gmc_warp",
        )
    )
    context["latency_started_at"] = 9.0
    state = SimpleNamespace(
        current_frame_id=current_frame_id,
        warmup_frames=1,
        frame_latencies=[],
        throughput_frames=0,
        throughput_finished_at=None,
        db_emit_frame_id=frame_id,
        db_emit_parity=0,
        double_buffer_tracker_out_pinned=[pinned],
        db_emit_ctx=context,
        prev_track_ids=set(),
        results_lines=[],
        bg_future=None,
        db_background_timing=None,
    )

    def synchronize():
        assert state.throughput_frames == 0
        clock[0] = 11.0

    state.db_emit_event = SimpleNamespace(synchronize=synchronize)

    def emit(*args, **kwargs):
        assert clock[0] == 11.0
        assert state.throughput_frames == 0
        assert kwargs["frame_id"] == frame_id
        clock[0] = 12.0
        if background:
            state.bg_future = object()
            return set(), []
        return set(), ["output"]

    monkeypatch.setattr(stages, "_run_emit", emit)
    stages._flush_db_tracker_out(state)
    if background:
        assert state.frame_latencies == []
        clock[0] = 14.0
        # The frame loop calls this after future.result() and appending lines.
        state.results_lines.append("output")
        stages._record_db_background_timing(state)
        stages._record_db_background_timing(state)
    stages._flush_db_tracker_out(state)  # Final drain must not double-count.
    measured = frame_id > state.warmup_frames
    assert state.throughput_frames == int(measured)
    assert state.frame_latencies == (
        [5000.0 if background else 3000.0] if measured else []
    )
    assert state.throughput_finished_at == (clock[0] if measured else None)
    assert state.results_lines == ["output"]
