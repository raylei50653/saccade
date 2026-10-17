"""#537 C0-11: pin how AsyncDispatcher calls its tracker, and what happens after that call fails.

Characterization only (R-1). These record the status quo on ``main@82126e7c``.
They do not choose the target contract; that needs an owner decision.

``_process_batch()`` calls ``tracker.set_degradation_params(level)`` with
``tracker.update(boxes, scores, classes)`` when level >= FAST_PATH. Below
FAST_PATH it calls ``tracker.update(boxes, scores, classes, frame_tensor=...,
stream_id=...)`` and unpacks a 3-tuple. Neither the native binding (``update``
takes raw pointers and returns ``list[TrackResult]``) nor the Python facade
(``update`` takes tensors, has no ``frame_tensor``/``stream_id``, and returns
``list[TrackResult]``) supports either call. The ``MagicMock`` trackers in
``tests/extended/test_dispatcher.py`` hide this.

The worker-death cases are #538 D2 observations: they pin what a dead worker
task leaves behind. They do not define how errors should propagate.
"""

# scope: perception, resource
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from saccade.perception import dispatcher as dmod
from saccade.perception.tracking import tracker_gpu as facade
from saccade.resource.resource_manager import DegradationLevel, ResourceManager

_HW = (8, 8)


class _StrictNativeSurfaceTracker:
    """The part of ``saccade_tracking_ext.GPUByteTracker`` the dispatcher touches.

    ``update`` keeps the binding's parameter names and accepts only ints for the
    pointer/count arguments, which is how pybind11 rejects tensors. There is no
    ``set_degradation_params``. No GPU is needed.
    """

    def update(
        self,
        boxes_ptr: int,
        scores_ptr: int,
        classes_ptr: int,
        num_dets: int,
        stream_ptr: int,
        embeddings_ptr: int = 0,
        gmc_ptr: int = 0,
        light_factor: float = 0.0,
        mid_thresh_scale: float = 1.0,
    ) -> list[Any]:
        for value in (boxes_ptr, scores_ptr, classes_ptr, num_dets, stream_ptr):
            if not isinstance(value, int):
                raise TypeError("update(): incompatible function arguments")
        return []


def _detector(n_boxes: int) -> MagicMock:
    det = MagicMock()
    det.detect_batch.side_effect = lambda x, conf_threshold: [
        (
            torch.tensor([[10.0, 10.0, 50.0, 50.0]] * n_boxes).reshape(n_boxes, 4),
            torch.full((n_boxes,), 0.9),
            torch.zeros(n_boxes, dtype=torch.int32),
            None,
        )
        for _ in range(x.shape[0])
    ]
    return det


def _dispatcher(level: DegradationLevel, n_boxes: int = 1) -> dmod.AsyncDispatcher:
    rm = MagicMock(spec=ResourceManager)
    rm.decide_degradation_level.return_value = level
    disp = dmod.AsyncDispatcher(_detector(n_boxes), rm, max_batch=1, input_hw=_HW)
    disp._make_tracker = _StrictNativeSurfaceTracker  # type: ignore[method-assign]
    return disp


async def _wait_done(task: asyncio.Task[None], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not task.done():
        assert asyncio.get_running_loop().time() < deadline, "worker still running"
        await asyncio.sleep(0.01)


# ── Tracker surfaces (no dispatcher) ────────────────────────────────────


def test_facade_does_not_support_dispatcher_calls() -> None:
    sig = inspect.signature(facade.GPUByteTracker.update)
    b, s, c = torch.zeros(1, 4), torch.zeros(1), torch.zeros(1, dtype=torch.int32)
    sig.bind(None, b, s, c)  # the tensor form itself is accepted
    with pytest.raises(TypeError, match="frame_tensor"):
        sig.bind(None, b, s, c, frame_tensor=torch.zeros(3, *_HW), stream_id=1)
    assert not hasattr(facade.GPUByteTracker, "set_degradation_params")
    assert "TrackResult" in str(sig.return_annotation)


def test_native_binding_has_no_degradation_api() -> None:
    ext = pytest.importorskip("saccade_tracking_ext")
    assert not hasattr(ext.GPUByteTracker, "set_degradation_params")
    first = (ext.GPUByteTracker.update.__doc__ or "").splitlines()[0]
    assert "boxes_ptr" in first and "frame_tensor" not in first
    assert first.endswith("-> list[saccade_tracking_ext.TrackResult]")


@pytest.mark.gpu
def test_native_binding_rejects_dispatcher_calls() -> None:
    ext = pytest.importorskip("saccade_tracking_ext")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to construct the native tracker")
    t = ext.GPUByteTracker(max_objects=16, embedding_dim=0)
    b = torch.tensor([[10.0, 10.0, 50.0, 80.0]], device="cuda")
    s = torch.tensor([0.9], device="cuda")
    c = torch.tensor([0], dtype=torch.int32, device="cuda")
    with pytest.raises(AttributeError, match="set_degradation_params"):
        t.set_degradation_params(DegradationLevel.FAST_PATH)
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.update(b, s, c)
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.update(b, s, c, frame_tensor=torch.zeros(3, 8, 8, device="cuda"), stream_id=1)


# ── Dispatcher with a native-surface tracker ────────────────────────────

_EXPECTED = [
    (DegradationLevel.NORMAL, TypeError, "frame_tensor"),
    (DegradationLevel.REDUCED, TypeError, "frame_tensor"),
    (DegradationLevel.FAST_PATH, AttributeError, "set_degradation_params"),
    (DegradationLevel.EMERGENCY, AttributeError, "set_degradation_params"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("level", "exc", "needle"), _EXPECTED)
async def test_first_detection_kills_worker(
    level: DegradationLevel, exc: type[Exception], needle: str
) -> None:
    with patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()):
        disp = _dispatcher(level)
        await disp.start()
        task = disp._worker_task
        assert task is not None
        await disp.put_frame("cam0", torch.rand(3, *_HW), 0.0)
        await _wait_done(task)
        err = task.exception()
        assert type(err) is exc and needle in str(err)
        task.cancel()
        disp._running = False


@pytest.mark.asyncio
async def test_empty_frames_never_touch_tracker() -> None:
    """The failure depends on input: frames without detections keep the worker alive."""
    with patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()):
        disp = _dispatcher(DegradationLevel.FAST_PATH, n_boxes=0)
        await disp.start()
        for i in range(5):
            await disp.put_frame("cam0", torch.rand(3, *_HW), float(i))
        await asyncio.wait_for(disp.queue.join(), timeout=2.0)
        assert disp._worker_task is not None and not disp._worker_task.done()
        assert disp.trackers == {}
        await disp.stop()


# ── #538 D2 observations: what a dead worker leaves behind ───────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(("level", "exc", "needle"), [_EXPECTED[0], _EXPECTED[2]])
async def test_dead_worker_aftermath(
    level: DegradationLevel, exc: type[Exception], needle: str
) -> None:
    writer = MagicMock()
    with patch.object(dmod, "VRAMLevelWriter", return_value=writer):
        disp = _dispatcher(level)
        await disp.start()
        task = disp._worker_task
        assert task is not None
        await disp.put_frame("cam0", torch.rand(3, *_HW), 0.0)
        await _wait_done(task)

        # Nothing outside the task records the failure.
        assert disp._running is True
        assert disp.queue._unfinished_tasks == 1  # type: ignore[attr-defined]
        for i in range(150):
            await disp.put_frame("cam0", torch.rand(3, *_HW), float(i))  # no error
        stats = disp.get_stats()
        assert stats["queue_depth"] == disp.queue.maxsize == 100
        assert stats["current_level"] == level.name
        assert not [k for k in stats if "error" in k or "health" in k or "drop" in k]

        # stop() re-raises the worker's exception and skips its cleanup.
        with pytest.raises(exc, match=needle):
            await asyncio.wait_for(disp.stop(), timeout=2.0)
        assert disp._running is False
        assert disp._worker_task is task
        assert len(disp.trackers) == 1
        writer.close.assert_not_called()
