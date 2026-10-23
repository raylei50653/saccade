"""#537 D-537-4A: monitor preflight, failure reporting and explicit recovery.

Real ResourceManager and dispatcher decisions run against fake NVML. Detector,
processing and (except for snapshot tests) writer are host fakes. No CUDA or
public shared-memory name is used. Pre-fix observations are preserved at
6cb6d121 and in failure_degradation_537.md §19.1.
"""

# scope: resource, perception
# function: contract
# lifecycle: active

from __future__ import annotations

import asyncio
import dataclasses
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import saccade.resource.resource_manager as rm
from saccade.perception import dispatcher as dmod
from saccade.perception.dispatcher import (
    DispatcherFailed,
    DispatcherNotRunning,
    DispatcherState,
)
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
        if fault == "init":
            self.init_error = RuntimeError("NVML unavailable")
            return self.init_error
        if fault == "handle":
            self.handle_error = RuntimeError("handle unavailable")
            return self.handle_error
        if fault == "memory":
            self.memory_error = RuntimeError("memory unavailable")
            return self.memory_error
        assert fault == "total_zero"
        self.info = SimpleNamespace(free=0, used=0, total=0)
        return None

    def recover_empty(self) -> None:
        self.init_error = self.handle_error = self.memory_error = None
        self.info = SimpleNamespace(free=100 * _MIB, used=0, total=100 * _MIB)


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> _FakeNVML:
    fake = _FakeNVML()
    monkeypatch.setattr(rm, "pynvml", fake)
    return fake


def _dispatcher(
    monkeypatch: pytest.MonkeyPatch, manager: rm.ResourceManager
) -> tuple[dmod.AsyncDispatcher, list[MagicMock], AsyncMock, MagicMock]:
    writers = [MagicMock(spec=rm.VRAMLevelWriter), MagicMock(spec=rm.VRAMLevelWriter)]
    factory = MagicMock(side_effect=writers)
    monkeypatch.setattr(dmod, "native_unavailable_reason", lambda: None)
    monkeypatch.setattr(dmod, "VRAMLevelWriter", factory)
    dispatcher = dmod.AsyncDispatcher(
        detector=MagicMock(), resource_manager=manager, max_batch=1
    )
    process = AsyncMock()
    monkeypatch.setattr(dispatcher, "_process_batch", process)
    return dispatcher, writers, process, factory


async def _frame(dispatcher: dmod.AsyncDispatcher) -> None:
    await dispatcher.put_frame("cam0", torch.zeros(3, 8, 8), 0.0)
    await asyncio.wait_for(dispatcher.queue.join(), timeout=2.0)


async def _stop(dispatcher: dmod.AsyncDispatcher) -> None:
    try:
        await dispatcher.stop()
    except DispatcherFailed:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("init", *_FAULTS))
async def test_unavailable_preflight_has_no_writer_worker_or_frame_side_effect(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML, fault: str
) -> None:
    original = nvml.fail_read(fault)
    manager = rm.ResourceManager()
    dispatcher, _, process, factory = _dispatcher(monkeypatch, manager)
    assert dispatcher.get_stats()["current_level"] == "UNKNOWN"
    assert dispatcher.get_stats()["resource_monitor_status"] == "unprobed"
    assert dispatcher.get_stats()["resource_monitor_error"] == ""
    tasks_before = asyncio.all_tasks()
    try:
        with pytest.raises(rm.ResourceMonitorUnavailable) as refused:
            await dispatcher.start()
        if original is None:
            assert isinstance(refused.value.__cause__, ValueError)
        else:
            assert refused.value.__cause__ is original
        assert dispatcher.state is DispatcherState.CREATED
        assert dispatcher.failure is None
        assert dispatcher._worker_task is None
        assert dispatcher._vram_writer is None
        assert asyncio.all_tasks() == tasks_before
        factory.assert_not_called()
        process.assert_not_awaited()
        with pytest.raises(DispatcherNotRunning):
            await dispatcher.put_frame("cam0", torch.zeros(3, 8, 8), 0.0)
        assert dispatcher.queue.empty()
        for _ in range(3):
            stats = dispatcher.get_stats()
            assert stats["resource_monitor_status"] == "unavailable"
            assert stats["resource_monitor_error"] == repr(refused.value)
            assert stats["current_level"] == "UNKNOWN"
            assert stats["failure"] == ""
    finally:
        await _stop(dispatcher)
        manager.close()


@pytest.mark.asyncio
async def test_preflight_does_not_decide_or_publish_and_valid_zero_is_available(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML
) -> None:
    nvml.recover_empty()
    manager = rm.ResourceManager()
    manager.current_level = L.EMERGENCY
    dispatcher, writers, process, _ = _dispatcher(monkeypatch, manager)
    try:
        await dispatcher.start()
        assert dispatcher.state is DispatcherState.RUNNING
        assert manager.current_level is L.EMERGENCY
        assert nvml.calls == ["init", "handle:0", "memory"]
        writers[0].write.assert_not_called()
        process.assert_not_awaited()
        calls = list(nvml.calls)
        for _ in range(3):
            stats = dispatcher.get_stats()
            assert stats["current_level"] == "UNKNOWN"
            assert stats["resource_monitor_status"] == "available"
            assert stats["resource_monitor_error"] == ""
        assert nvml.calls == calls  # reporting never probes or changes a level
        await _frame(dispatcher)
        assert manager.current_level is L.FAST_PATH
        assert dispatcher.get_stats()["current_level"] == "FAST_PATH"
        writers[0].write.assert_called_once()
        assert writers[0].write.call_args.args[0] is L.FAST_PATH
    finally:
        await _stop(dispatcher)
        manager.close()


@pytest.mark.asyncio
async def test_init_failure_requires_a_new_manager_after_backend_recovery(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML
) -> None:
    original = nvml.fail_read("init")
    manager = rm.ResourceManager()
    dispatcher, _, _, factory = _dispatcher(monkeypatch, manager)
    recovered: dmod.AsyncDispatcher | None = None
    fresh_manager: rm.ResourceManager | None = None
    try:
        with pytest.raises(rm.ResourceMonitorUnavailable):
            await dispatcher.start()
        nvml.recover_empty()
        with pytest.raises(rm.ResourceMonitorUnavailable) as refused:
            await dispatcher.start()
        assert refused.value.__cause__ is original
        assert nvml.calls == ["init"]
        factory.assert_not_called()
        fresh_manager = rm.ResourceManager()
        recovered, writers, _, _ = _dispatcher(monkeypatch, fresh_manager)
        await recovered.start()
        assert recovered.get_stats()["resource_monitor_status"] == "available"
        assert recovered.get_stats()["resource_monitor_error"] == ""
        assert recovered.get_stats()["current_level"] == "UNKNOWN"
        assert fresh_manager.current_level is L.NORMAL
        await _frame(recovered)
        assert recovered.get_stats()["current_level"] == "NORMAL"
        assert writers[0].write.call_args.args[0] is L.NORMAL
    finally:
        if recovered is not None:
            await _stop(recovered)
        await _stop(dispatcher)
        if fresh_manager is not None:
            fresh_manager.close()
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", _FAULTS)
async def test_runtime_failure_accounts_for_queue_and_preserves_cause_on_restart(
    monkeypatch: pytest.MonkeyPatch,
    nvml: _FakeNVML,
    fault: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = rm.ResourceManager()
    dispatcher, writers, process, factory = _dispatcher(monkeypatch, manager)
    try:
        await dispatcher.start()
        await _frame(dispatcher)
        assert manager.current_level is L.EMERGENCY
        original = nvml.fail_read(fault)
        task = dispatcher._worker_task
        for _ in range(3):
            await dispatcher.put_frame("cam0", torch.zeros(3, 8, 8), 1.0)
        await asyncio.wait_for(dispatcher.queue.join(), timeout=2.0)
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        failure = dispatcher.failure
        assert isinstance(failure, rm.ResourceMonitorUnavailable)
        if original is None:
            assert isinstance(failure.__cause__, ValueError)
        else:
            assert failure.__cause__ is original
        assert dispatcher.state is DispatcherState.FAILED
        assert task.exception() is None
        assert manager.current_level is L.EMERGENCY
        stats = dispatcher.get_stats()
        assert stats["current_level"] == "UNKNOWN"
        assert stats["resource_monitor_status"] == "unavailable"
        assert stats["resource_monitor_error"] == repr(failure)
        assert stats["failure"] == repr(failure)
        assert stats["failed_batch_frames"] == 1
        assert stats["discarded_on_failure"] == 2
        writers[0].write.assert_called_once()
        writers[0].close.assert_not_called()
        process.assert_awaited_once()
        for action in (
            lambda: dispatcher.put_frame("cam0", torch.zeros(3, 8, 8), 2.0),
            dispatcher.start,
        ):
            with pytest.raises(DispatcherFailed) as refused:
                await action()
            assert refused.value.__cause__ is failure
        with pytest.raises(DispatcherFailed) as stopped:
            await dispatcher.stop()
        assert stopped.value.__cause__ is failure
        writers[0].close.assert_called_once()

        # A rejected restart stays STOPPED and keeps the previous worker cause;
        # startup's new error is independently observable in monitor fields.
        with pytest.raises(rm.ResourceMonitorUnavailable) as startup:
            await dispatcher.start()
        assert dispatcher.state is DispatcherState.STOPPED
        assert dispatcher.failure is failure
        assert dispatcher._worker_task is None
        assert dispatcher._vram_writer is None
        assert factory.call_count == 1
        assert dispatcher.get_stats()["resource_monitor_error"] == repr(startup.value)
        assert dispatcher.get_stats()["current_level"] == "UNKNOWN"

        nvml.recover_empty()
        await dispatcher.start()
        assert dispatcher.failure is None
        assert dispatcher.get_stats()["resource_monitor_status"] == "available"
        assert dispatcher.get_stats()["resource_monitor_error"] == ""
        assert dispatcher.get_stats()["current_level"] == "UNKNOWN"
        assert manager.current_level is L.EMERGENCY
        await _frame(dispatcher)
        assert manager.current_level is L.FAST_PATH
        assert writers[1].write.call_args.args[0] is L.FAST_PATH
        assert nvml.calls.count("init") == 1

        nvml.fail_read(fault)
        await _frame(dispatcher)
        assert dispatcher.state is DispatcherState.FAILED
        assert isinstance(dispatcher.failure, rm.ResourceMonitorUnavailable)
        assert dispatcher.failure is not failure
        assert dispatcher.get_stats()["current_level"] == "UNKNOWN"
        writers[1].write.assert_called_once()
        process.assert_awaited()
        assert process.await_count == 2
        for _ in range(3):
            dispatcher.get_stats()
        assert capsys.readouterr().out.count("[Dispatcher] Worker failed:") == 2
    finally:
        await _stop(dispatcher)
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [-1.0, Decimal("NaN")], ids=repr)
async def test_out_of_domain_usage_fails_the_worker_without_publishing(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML, usage: object
) -> None:
    """O0: a replaced get_stats() yielding invalid usage takes the 4A path."""
    manager = rm.ResourceManager()
    dispatcher, writers, process, _ = _dispatcher(monkeypatch, manager)
    try:
        await dispatcher.start()
        await _frame(dispatcher)
        assert manager.current_level is L.EMERGENCY
        valid = manager.get_stats
        monkeypatch.setattr(
            manager,
            "get_stats",
            lambda: dataclasses.replace(valid(), usage_percent=usage),
        )
        task = dispatcher._worker_task
        await _frame(dispatcher)
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        failure = dispatcher.failure
        assert isinstance(failure, rm.ResourceMonitorUnavailable)
        assert isinstance(failure.__cause__, ValueError)
        assert dispatcher.state is DispatcherState.FAILED
        assert manager.current_level is L.EMERGENCY
        stats = dispatcher.get_stats()
        assert stats["current_level"] == "UNKNOWN"
        assert stats["resource_monitor_status"] == "unavailable"
        assert stats["resource_monitor_error"] == repr(failure)
        writers[0].write.assert_called_once()
        assert writers[0].write.call_args.args[0] is L.EMERGENCY
        process.assert_awaited_once()
    finally:
        await _stop(dispatcher)
        manager.close()


@pytest.mark.asyncio
async def test_detector_failure_keeps_the_successful_monitor_observation(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML
) -> None:
    manager = rm.ResourceManager()
    dispatcher, writers, process, _ = _dispatcher(monkeypatch, manager)
    original = RuntimeError("detector failed")
    process.side_effect = original
    try:
        await dispatcher.start()
        await _frame(dispatcher)
        assert dispatcher.state is DispatcherState.FAILED
        assert dispatcher.failure is original
        stats = dispatcher.get_stats()
        assert stats["resource_monitor_status"] == "available"
        assert stats["resource_monitor_error"] == ""
        assert stats["current_level"] == "EMERGENCY"
        writers[0].write.assert_called_once()
    finally:
        await _stop(dispatcher)
        manager.close()


@pytest.mark.asyncio
async def test_rejected_preflight_leaves_existing_fallback_snapshot_to_age(
    monkeypatch: pytest.MonkeyPatch, nvml: _FakeNVML, tmp_path: Path
) -> None:
    monkeypatch.setattr(rm, "_SHM_DIR", str(tmp_path))
    now = [10_000]
    reader = rm.VRAMLevelReader(max_age_ns=100, clock=lambda: now[0])
    previous = rm.VRAMLevelWriter()
    previous.write(L.NORMAL, now[0])  # an older init-fallback publication
    path = tmp_path / rm._SHM_NAME
    before = path.read_bytes()
    nvml.fail_read("init")
    manager = rm.ResourceManager()
    dispatcher, _, _, factory = _dispatcher(monkeypatch, manager)
    try:
        with pytest.raises(rm.ResourceMonitorUnavailable):
            await dispatcher.start()
        factory.assert_not_called()
        await dispatcher.stop()
        assert path.read_bytes() == before
        assert reader.read_snapshot().level is L.NORMAL
        now[0] += 100
        assert reader.read_snapshot().status is rm.VRAMReadStatus.FRESH
        now[0] += 1
        assert reader.read_snapshot().status is rm.VRAMReadStatus.STALE
    finally:
        previous.close()
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement_fails", (False, True))
async def test_failed_monitor_snapshot_retires_or_ages_when_retirement_fails(
    monkeypatch: pytest.MonkeyPatch,
    nvml: _FakeNVML,
    tmp_path: Path,
    retirement_fails: bool,
) -> None:
    monkeypatch.setattr(rm, "_SHM_DIR", str(tmp_path))
    now = [10_000]
    monkeypatch.setattr(dmod, "vram_level_clock_ns", lambda: now[0])
    reader = rm.VRAMLevelReader(max_age_ns=100, clock=lambda: now[0])
    manager = rm.ResourceManager()
    dispatcher, _, _, _ = _dispatcher(monkeypatch, manager)
    monkeypatch.setattr(dmod, "VRAMLevelWriter", rm.VRAMLevelWriter)
    try:
        await dispatcher.start()
        await _frame(dispatcher)
        assert reader.read_snapshot().level is L.EMERGENCY
        before = (tmp_path / rm._SHM_NAME).read_bytes()
        nvml.fail_read("memory")
        now[0] += 1
        await _frame(dispatcher)
        assert (tmp_path / rm._SHM_NAME).read_bytes() == before
        assert reader.read_snapshot().level is L.EMERGENCY
        writer = dispatcher._vram_writer
        assert writer is not None
        if retirement_fails:
            monkeypatch.setattr(
                writer, "_publish", MagicMock(side_effect=OSError("rename"))
            )
        with pytest.raises(DispatcherFailed):
            await dispatcher.stop()
        assert reader.read_snapshot().status is (
            rm.VRAMReadStatus.FRESH if retirement_fails else rm.VRAMReadStatus.STALE
        )
        now[0] += 99
        assert reader.read_snapshot().status is (
            rm.VRAMReadStatus.FRESH if retirement_fails else rm.VRAMReadStatus.STALE
        )
        now[0] += 1
        assert reader.read_snapshot().status is rm.VRAMReadStatus.STALE
    finally:
        await _stop(dispatcher)
        manager.close()
