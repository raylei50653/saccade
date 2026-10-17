"""#538 D2-AD: AsyncDispatcher worker failure, health, error propagation, queue accounting, stop cleanup.

Owner decisions (2026-10-17):

- When the worker raises, the dispatcher enters an observable FAILED state
  right away and keeps the original error.
- New work is refused after a failure.
- ``stop()`` always tries every cleanup step, and no exception can skip one.

These pin the contract that follows from those decisions:

- ``state``/``failure`` and ``get_stats()["state"/"failure"]`` show the failure.
- ``put_frame()`` and ``start()`` raise ``DispatcherFailed``, whose
  ``__cause__`` is the original error.
- Frames in the failed batch and frames still queued are ``task_done()`` and
  counted, so ``queue.join()`` cannot hang.
- ``stop()`` cleans up, then raises ``DispatcherFailed`` for a failed run, or
  the first cleanup error for a run that did not fail.

Host-only: the tracker, detector and VRAM writer are fakes. No GPU is used.
"""

# scope: perception, resource
# function: contract
# lifecycle: active

from __future__ import annotations

import asyncio
from typing import Any, Iterator
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch

from saccade.perception import dispatcher as dmod
from saccade.perception.dispatcher import DispatcherFailed, DispatcherState
from saccade.perception.online_tracker_adapter import OnlineTrackerAdapter
from saccade.perception.tracking import tracker_gpu as facade
from saccade.resource.resource_manager import DegradationLevel, ResourceManager

_HW = (8, 8)


@pytest.fixture(autouse=True)
def _host_only() -> Iterator[MagicMock]:
    writer = MagicMock()
    with (
        patch.object(dmod, "native_unavailable_reason", return_value=None),
        patch.object(dmod, "VRAMLevelWriter", return_value=writer),
    ):
        yield writer


def _facade(error: Exception | None = None) -> MagicMock:
    fake = create_autospec(facade.GPUByteTracker, instance=True)
    fake.update.return_value = []
    if error is not None:
        fake.update.side_effect = error
    return fake


def _dispatcher(
    *,
    tracker_error: Exception | None = None,
    detector_error: Exception | None = None,
    decide_error: Exception | None = None,
    on_track_result: Any = None,
    max_batch: int = 1,
) -> dmod.AsyncDispatcher:
    det = MagicMock()
    if detector_error is not None:
        det.detect_batch.side_effect = detector_error
    else:
        det.detect_batch.side_effect = lambda x, conf_threshold: [
            (
                torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                torch.tensor([0.9]),
                torch.zeros(1, dtype=torch.int32),
                None,
            )
            for _ in range(x.shape[0])
        ]
    rm = MagicMock(spec=ResourceManager)
    if decide_error is not None:
        rm.decide_degradation_level.side_effect = decide_error
    else:
        rm.decide_degradation_level.return_value = DegradationLevel.NORMAL
    disp = dmod.AsyncDispatcher(
        det, rm, max_batch=max_batch, input_hw=_HW, on_track_result=on_track_result
    )
    disp._make_tracker = lambda: OnlineTrackerAdapter(  # type: ignore[method-assign]
        _tracker=_facade(tracker_error)
    )
    return disp


def _frame() -> torch.Tensor:
    return torch.rand(3, *_HW)


async def _until_done(task: asyncio.Task[None]) -> None:
    await asyncio.wait_for(asyncio.wait({task}), timeout=2.0)


async def _failing_callback(*_: Any) -> None:
    raise KeyError("callback failed")


_SOURCES = {
    "tracker": dict(tracker_error=RuntimeError("tracker failed")),
    "detector": dict(detector_error=RuntimeError("detector failed")),
    "resource_manager": dict(decide_error=ZeroDivisionError("total == 0")),
    "callback": dict(on_track_result=_failing_callback),
}


# ── FAILED state and the original error ─────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("source", list(_SOURCES))
async def test_worker_error_enters_failed_with_original_error(source: str) -> None:
    disp = _dispatcher(**_SOURCES[source])
    await disp.start()
    task = disp._worker_task
    assert task is not None and disp.state is DispatcherState.RUNNING
    await disp.put_frame("cam0", _frame(), 0.0)
    await _until_done(task)

    # The state is set inside the worker, so it is visible once the task is done.
    assert disp.state is DispatcherState.FAILED
    original = disp.failure
    assert original is not None and task.exception() is None
    expected = {
        "tracker": RuntimeError,
        "detector": RuntimeError,
        "resource_manager": ZeroDivisionError,
        "callback": KeyError,
    }[source]
    assert type(original) is expected
    stats = disp.get_stats()
    assert stats["state"] == "failed" and stats["failure"] == repr(original)
    assert disp._running is False

    with pytest.raises(DispatcherFailed) as exc:
        await disp.stop()
    assert exc.value.__cause__ is original


@pytest.mark.asyncio
async def test_new_work_is_refused_after_failure() -> None:
    disp = _dispatcher(tracker_error=RuntimeError("tracker failed"))
    await disp.start()
    assert disp._worker_task is not None
    await disp.put_frame("cam0", _frame(), 0.0)
    await _until_done(disp._worker_task)
    original = disp.failure

    with pytest.raises(DispatcherFailed) as exc:
        await disp.put_frame("cam0", _frame(), 1.0)
    assert exc.value.__cause__ is original
    assert disp.queue.qsize() == 0
    with pytest.raises(DispatcherFailed):
        await disp.start()  # a failed run must be stopped first
    with pytest.raises(DispatcherFailed):
        await disp.stop()


# ── Queue accounting ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failure_settles_the_failed_batch_and_the_queue() -> None:
    gate = asyncio.Event()
    calls = 0

    async def slow_then_fail(*_: Any) -> None:
        nonlocal calls
        calls += 1
        await gate.wait()
        raise RuntimeError("callback failed")

    disp = _dispatcher(on_track_result=slow_then_fail, max_batch=2)
    disp.batch_timeout_ms = 200.0
    await disp.start()
    assert disp._worker_task is not None
    for i in range(2):
        await disp.put_frame("cam0", _frame(), float(i))
    while calls == 0:
        await asyncio.sleep(0.01)
    for i in range(5):  # queued behind the batch that is about to fail
        await disp.put_frame("cam0", _frame(), float(10 + i))
    gate.set()
    await _until_done(disp._worker_task)

    await asyncio.wait_for(disp.queue.join(), timeout=1.0)  # nothing is left unfinished
    stats = disp.get_stats()
    assert stats["failed_batch_frames"] == 2
    assert stats["discarded_on_failure"] == 5
    assert stats["queue_depth"] == 0
    with pytest.raises(DispatcherFailed):
        await disp.stop()
    assert disp.get_stats()["discarded_on_stop"] == 0


@pytest.mark.asyncio
async def test_queue_full_drops_are_counted() -> None:
    disp = _dispatcher()
    for i in range(disp.queue.maxsize + 5):
        await disp.put_frame("cam0", _frame(), float(i))
    assert disp.get_stats()["dropped_queue_full"] == 5


@pytest.mark.asyncio
async def test_stop_settles_queued_and_inflight_frames() -> None:
    gate = asyncio.Event()
    started = asyncio.Event()

    async def blocked(*_: Any) -> None:
        started.set()
        await gate.wait()

    disp = _dispatcher(on_track_result=blocked)
    await disp.start()
    await disp.put_frame("cam0", _frame(), 0.0)
    await asyncio.wait_for(started.wait(), timeout=2.0)
    for i in range(3):
        await disp.put_frame("cam0", _frame(), float(1 + i))

    await asyncio.wait_for(disp.stop(), timeout=2.0)  # cancels the blocked batch
    await asyncio.wait_for(disp.queue.join(), timeout=1.0)
    stats = disp.get_stats()
    assert stats["discarded_on_stop"] == 4  # 1 in flight + 3 queued
    assert stats["state"] == "stopped" and disp.failure is None


# ── stop() cleanup ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stop_after_failure_cleans_up_then_raises(_host_only: MagicMock) -> None:
    disp = _dispatcher(tracker_error=RuntimeError("tracker failed"))
    await disp.start()
    assert disp._worker_task is not None
    await disp.put_frame("cam0", _frame(), 0.0)
    await _until_done(disp._worker_task)
    assert len(disp.trackers) == 1

    with pytest.raises(DispatcherFailed):
        await disp.stop()
    assert disp.trackers == {} and disp._worker_task is None
    assert disp._vram_writer is None
    _host_only.close.assert_called_once()
    assert disp.state is DispatcherState.STOPPED
    assert isinstance(disp.failure, RuntimeError)  # kept until the next start()

    await disp.stop()  # a second stop() has nothing left to report


@pytest.mark.asyncio
async def test_cleanup_continues_past_a_failing_step(_host_only: MagicMock) -> None:
    _host_only.close.side_effect = OSError("close failed")
    disp = _dispatcher()
    await disp.start()
    disp.get_tracker("cam0")
    disp.queue.put_nowait(dmod._QueuedFrame("cam0", _frame(), 0.0, 0.0))
    with patch.object(disp, "_stop_settle", side_effect=ValueError("settle failed")):
        with pytest.raises(ValueError, match="settle failed"):
            await disp.stop()
    assert disp.trackers == {}  # later steps still ran
    _host_only.close.assert_called_once()
    assert disp._vram_writer is None and disp.state is DispatcherState.STOPPED


@pytest.mark.asyncio
async def test_failed_run_keeps_cleanup_errors_as_notes(_host_only: MagicMock) -> None:
    _host_only.close.side_effect = OSError("close failed")
    disp = _dispatcher(tracker_error=RuntimeError("tracker failed"))
    await disp.start()
    assert disp._worker_task is not None
    await disp.put_frame("cam0", _frame(), 0.0)
    await _until_done(disp._worker_task)
    with pytest.raises(DispatcherFailed) as exc:
        await disp.stop()
    assert any("close failed" in note for note in exc.value.__notes__)
    assert disp.trackers == {}


# ── Lifecycle ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_normal_stop_is_not_a_failure() -> None:
    disp = _dispatcher()
    assert disp.state is DispatcherState.CREATED
    await disp.start()
    await disp.put_frame("cam0", _frame(), 0.0)
    await asyncio.wait_for(disp.queue.join(), timeout=2.0)
    await disp.stop()
    assert disp.state is DispatcherState.STOPPED and disp.failure is None
    assert disp.get_stats()["failure"] == ""


@pytest.mark.asyncio
async def test_start_rules() -> None:
    disp = _dispatcher(tracker_error=RuntimeError("tracker failed"))
    await disp.start()
    with pytest.raises(RuntimeError, match="already running"):
        await disp.start()
    assert disp._worker_task is not None
    await disp.put_frame("cam0", _frame(), 0.0)
    await _until_done(disp._worker_task)
    with pytest.raises(DispatcherFailed):
        await disp.stop()

    await disp.start()  # restart after stop clears the old failure
    assert disp.state is DispatcherState.RUNNING and disp.failure is None
    await disp.stop()
