"""#537 C0-11: pin the tracker surfaces behind AsyncDispatcher, and what a dead worker leaves behind.

Characterization only. Pinned on ``main@82126e7c``; trimmed by the R-1 fix.

Before the fix, ``_process_batch()`` called ``tracker.set_degradation_params(level)``
and ``tracker.update(boxes, scores, classes)`` at FAST_PATH and above. Below
FAST_PATH it called ``tracker.update(..., frame_tensor=..., stream_id=...)`` and
unpacked a 3-tuple. Neither the native binding nor the Python facade supports
those calls. The surface checks below still hold, because the fix added
``OnlineTrackerAdapter`` and left both surfaces unchanged. The dispatcher cases
that showed the worker dying moved to ``test_dispatcher_tracker_contract_537.py``
as target-contract tests. Failure evidence from before the fix is in
failure_degradation_537.md §9 and PR #600.

The worker-death case is a #538 D2 observation. It pins what a dead worker task
leaves behind, and D2-AD will change it. It does not define how errors should
propagate.
"""

# scope: perception, resource
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch

from saccade.perception import dispatcher as dmod
from saccade.perception.online_tracker_adapter import OnlineTrackerAdapter
from saccade.perception.tracking import tracker_gpu as facade
from saccade.resource.resource_manager import DegradationLevel, ResourceManager

_HW = (8, 8)


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


# ── #538 D2 observation: what a dead worker leaves behind ────────────────


@pytest.mark.asyncio
async def test_dead_worker_aftermath() -> None:
    fake = create_autospec(facade.GPUByteTracker, instance=True)
    fake.update.side_effect = RuntimeError("native tracker failed")
    writer = MagicMock()
    with (
        patch.object(dmod, "native_unavailable_reason", return_value=None),
        patch.object(dmod, "VRAMLevelWriter", return_value=writer),
    ):
        disp = _dispatcher(DegradationLevel.FAST_PATH)
        disp._make_tracker = lambda: OnlineTrackerAdapter(_tracker=fake)  # type: ignore[method-assign]
        await disp.start()
        task = disp._worker_task
        assert task is not None
        await disp.put_frame("cam0", torch.rand(3, *_HW), 0.0)
        await _wait_done(task)
        assert isinstance(task.exception(), RuntimeError)

        # Nothing outside the task records the failure.
        assert disp._running is True
        assert disp.queue._unfinished_tasks == 1  # type: ignore[attr-defined]
        for i in range(150):
            await disp.put_frame("cam0", torch.rand(3, *_HW), float(i))  # no error
        stats = disp.get_stats()
        assert stats["queue_depth"] == disp.queue.maxsize == 100
        assert stats["current_level"] == "FAST_PATH"
        assert not [k for k in stats if "error" in k or "health" in k or "drop" in k]

        # stop() re-raises the worker's exception and skips its cleanup.
        with pytest.raises(RuntimeError, match="native tracker failed"):
            await asyncio.wait_for(disp.stop(), timeout=2.0)
        assert disp._running is False
        assert disp._worker_task is task
        assert len(disp.trackers) == 1
        writer.close.assert_not_called()
