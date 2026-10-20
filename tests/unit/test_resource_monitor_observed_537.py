"""#537 C0-05: characterize NVML failure and the existing dispatcher boundary.

These are diagnostic observations at d6724f11, not the pending monitor policy.
In particular, unavailable zero stats and newly published NORMAL decisions
remain incorrect evidence of health. Post-init read errors instead reach the
already approved #538 D2-AD failure boundary. A future C0-05 fix must update
these observations alongside its approved policy.

Host-only: NVML and the snapshot writer are fakes, batch processing is replaced,
and no CUDA operation or shared-memory name is used.
"""

# scope: resource, perception
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import saccade.resource.resource_manager as rm
from saccade.perception import dispatcher as dmod
from saccade.perception.dispatcher import DispatcherFailed, DispatcherState
from saccade.resource.resource_manager import DegradationLevel as L

_MIB = 1024**2
_FAULTS = ("handle", "memory", "total_zero")


class _FakeNVML:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.init_error: Exception | None = None
        self.handle_error: Exception | None = None
        self.memory_error: Exception | None = None
        self.info = SimpleNamespace(free=3 * _MIB, used=97 * _MIB, total=100 * _MIB)
        self.handle = object()

    def nvmlInit(self) -> None:
        self.calls.append("init")
        if self.init_error is not None:
            raise self.init_error

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        self.calls.append(f"handle:{index}")
        if self.handle_error is not None:
            raise self.handle_error
        return self.handle

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> SimpleNamespace:
        assert handle is self.handle
        self.calls.append("memory")
        if self.memory_error is not None:
            raise self.memory_error
        return self.info

    def nvmlShutdown(self) -> None:
        self.calls.append("shutdown")

    def fail_read(self, fault: str) -> Exception | None:
        if fault == "handle":
            self.handle_error = RuntimeError("handle unavailable")
            return self.handle_error
        if fault == "memory":
            self.memory_error = RuntimeError("memory unavailable")
            return self.memory_error
        assert fault == "total_zero"
        self.info = SimpleNamespace(free=0, used=0, total=0)
        return None  # Python creates the ZeroDivisionError in get_stats().

    def recover_empty(self) -> None:
        self.init_error = self.handle_error = self.memory_error = None
        self.info = SimpleNamespace(free=100 * _MIB, used=0, total=100 * _MIB)


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeNVML]:
    fake = _FakeNVML()
    monkeypatch.setattr(rm, "pynvml", fake)
    yield fake


def test_valid_zero_usage_and_unavailable_zero_usage_have_different_stats(
    nvml: _FakeNVML,
) -> None:
    nvml.recover_empty()
    available = rm.ResourceManager().get_stats()
    assert vars(available) == {
        "free_mb": 100.0,
        "used_mb": 0.0,
        "total_mb": 100.0,
        "usage_percent": 0.0,
    }

    nvml.init_error = RuntimeError("NVML unavailable")
    unavailable = rm.ResourceManager().get_stats()
    assert vars(unavailable) == {
        "free_mb": 0.0,
        "used_mb": 0.0,
        "total_mb": 0.0,
        "usage_percent": 0.0,
    }
    # A consumer of usage alone cannot distinguish a real empty GPU from the
    # init-failed sentinel. Neither result carries an explicit validity status.
    assert available.usage_percent == unavailable.usage_percent
    assert available.total_mb != unavailable.total_mb


def test_init_failure_never_retries_after_backend_recovery(nvml: _FakeNVML) -> None:
    nvml.init_error = RuntimeError("NVML unavailable")
    manager = rm.ResourceManager()
    manager.current_level = L.EMERGENCY
    nvml.recover_empty()

    for expected in (L.FAST_PATH, L.REDUCED, L.NORMAL):
        assert manager.get_stats().total_mb == 0.0
        assert manager.decide_degradation_level() is expected
    manager.close()
    assert nvml.calls == ["init"]
    assert manager._nvml_initialized is False


@pytest.mark.parametrize("fault", _FAULTS)
def test_post_init_failure_retains_level_and_next_read_can_recover(
    nvml: _FakeNVML, fault: str
) -> None:
    manager = rm.ResourceManager(gpu_id=3)
    assert manager.decide_degradation_level() is L.EMERGENCY
    original = nvml.fail_read(fault)

    with pytest.raises(
        ZeroDivisionError if fault == "total_zero" else RuntimeError
    ) as exc:
        manager.decide_degradation_level()
    if original is not None:
        assert exc.value is original
    assert manager.current_level is L.EMERGENCY
    assert manager._nvml_initialized is True

    nvml.recover_empty()
    assert manager.get_stats().total_mb == 100.0
    assert manager.decide_degradation_level() is L.FAST_PATH
    assert nvml.calls.count("init") == 1
    assert "handle:3" in nvml.calls
    manager.close()
    assert nvml.calls[-1] == "shutdown"


def _dispatcher(
    monkeypatch: pytest.MonkeyPatch, manager: rm.ResourceManager
) -> tuple[dmod.AsyncDispatcher, list[MagicMock], AsyncMock]:
    writers = [MagicMock(spec=rm.VRAMLevelWriter), MagicMock(spec=rm.VRAMLevelWriter)]
    feed = iter(writers)
    monkeypatch.setattr(dmod, "native_unavailable_reason", lambda: None)
    monkeypatch.setattr(dmod, "VRAMLevelWriter", lambda: next(feed))
    dispatcher = dmod.AsyncDispatcher(
        detector=MagicMock(), resource_manager=manager, max_batch=1
    )
    process = AsyncMock()
    monkeypatch.setattr(dispatcher, "_process_batch", process)
    return dispatcher, writers, process


async def _frame(dispatcher: dmod.AsyncDispatcher) -> None:
    await dispatcher.put_frame("cam0", torch.zeros(3, 8, 8), 0.0)
    await asyncio.wait_for(dispatcher.queue.join(), timeout=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", _FAULTS)
async def test_post_init_failure_stops_dispatcher_without_new_decision(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML, fault: str
) -> None:
    manager = rm.ResourceManager()
    dispatcher, writers, process = _dispatcher(monkeypatch, manager)
    await dispatcher.start()
    task = dispatcher._worker_task
    assert task is not None
    try:
        await _frame(dispatcher)
        assert manager.current_level is L.EMERGENCY
        assert writers[0].write.call_count == 1
        assert writers[0].write.call_args.args[0] is L.EMERGENCY
        process.assert_awaited_once()

        original = nvml.fail_read(fault)
        await _frame(dispatcher)
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        failure = dispatcher.failure
        assert type(failure) is (
            ZeroDivisionError if fault == "total_zero" else RuntimeError
        )
        if original is not None:
            assert failure is original
        assert dispatcher.state is DispatcherState.FAILED
        assert task.exception() is None
        assert manager.current_level is L.EMERGENCY
        stats = dispatcher.get_stats()
        assert (stats["state"], stats["failure"], stats["current_level"]) == (
            "failed",
            repr(failure),
            "EMERGENCY",
        )
        assert stats["failed_batch_frames"] == 1
        # No new snapshot is published on this path, and D2-AD leaves the old
        # writer open until stop(); its prior decision can age into STALE.
        assert writers[0].write.call_count == 1
        writers[0].close.assert_not_called()
        process.assert_awaited_once()

        nvml.recover_empty()
        with pytest.raises(DispatcherFailed) as refused:
            await dispatcher.put_frame("cam0", torch.zeros(3, 8, 8), 1.0)
        assert refused.value.__cause__ is failure
        with pytest.raises(DispatcherFailed) as stopped:
            await dispatcher.stop()
        assert stopped.value.__cause__ is failure
        writers[0].close.assert_called_once()

        # Restoring NVML alone does not restart a failed worker. Explicit
        # stop/start can reuse the initialized manager and its retained level.
        await dispatcher.start()
        await _frame(dispatcher)
        assert dispatcher.state is DispatcherState.RUNNING
        assert dispatcher.failure is None
        assert manager.current_level is L.FAST_PATH
        assert writers[1].write.call_args.args[0] is L.FAST_PATH
        assert nvml.calls.count("init") == 1
        assert process.await_count == 2
    finally:
        try:
            await dispatcher.stop()
        except DispatcherFailed:
            pass
        manager.close()


@pytest.mark.asyncio
async def test_init_failure_still_publishes_normal_and_processes_batch(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML
) -> None:
    nvml.init_error = RuntimeError("NVML unavailable")
    manager = rm.ResourceManager()
    dispatcher, writers, process = _dispatcher(monkeypatch, manager)
    await dispatcher.start()
    try:
        await _frame(dispatcher)
        assert dispatcher.state is DispatcherState.RUNNING
        assert dispatcher.failure is None
        assert dispatcher.get_stats()["current_level"] == "NORMAL"
        writers[0].write.assert_called_once()
        assert writers[0].write.call_args.args[0] is L.NORMAL
        assert writers[0].write.call_args.args[1] > 0
        process.assert_awaited_once()
        assert nvml.calls == ["init"]
    finally:
        await dispatcher.stop()
        manager.close()
