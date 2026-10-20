"""#538 caller lifecycle: main.py and run_8stream_perception.py stop their workers and resources.

Owner's final ruling (2026-10-17):

- ``main.py`` tracks geometry only. It starts no SigLIP/EmbeddingDispatcher
  branch, logs that ReID is unsupported, and fails fast on ``--reid``.
- Both callers try ``stop()`` on a normal end, on a worker failure, on a
  producer error and on cancellation. ``run_full()`` cancels the orchestrator
  when perception fails.
- ``run_8stream_perception.py`` releases everything that does not depend on a
  live worker, even when ``dispatcher.stop()`` or ``pool.stop()`` raises.
- The first error is kept; cleanup errors become its notes.
- A ``WorkbenchPool.stop()`` timeout never forces the batcher out.

Each scenario runs the real entry point in a fresh interpreter under an
outside timeout. Media, NVML, detector, orchestrator and VRAM writer are fakes, and
the native check is bypassed, so there is no GPU, Redis or RTSP. The
WorkbenchPool scenarios reuse the D2-WB fakes.
"""

# scope: perception, pipeline
# function: contract
# lifecycle: active

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_workbench_pool_lifecycle_observed_538 import (
    _PRELUDE as _WB_PRELUDE,
)

REPO = Path(__file__).resolve().parents[2]

_COMMON = r"""
import asyncio, sys, time, types
from unittest.mock import MagicMock
import torch

for name, attrs in {
    "saccade.media.dali_pipeline": ["DALIMediaClient"],
    "saccade.media.mediamtx_client": ["MediaMTXClient"],
    "saccade.cognition.orchestrator": ["PipelineOrchestrator"],
}.items():
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, object)
    sys.modules[name] = mod

import saccade.perception.dispatcher as d
import saccade.resource.resource_manager as resource_module
# main.py constructs its process-global manager at import time, before the
# caller-specific manager is supplied below. Keep that import host-only too.
resource_module.pynvml = MagicMock()
d.native_unavailable_reason = lambda: None
WRITER = MagicMock()
d.VRAMLevelWriter = lambda: WRITER
INSTANCES = []

class RecordingDispatcher(d.AsyncDispatcher):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        INSTANCES.append(self)

class FakeDetector:
    fail = False
    def __init__(self, *a, **k):
        pass
    def detect_batch(self, x, conf_threshold):
        if FakeDetector.fail:
            raise RuntimeError("detector failed")
        z = torch.empty((0, 4))
        return [(z, torch.empty(0), torch.empty(0, dtype=torch.int32), None)] * x.shape[0]

def fake_rm():
    rm = MagicMock()
    from saccade.resource.resource_manager import VRAMStats
    rm.get_stats.return_value = VRAMStats(100.0, 0.0, 100.0, 0.0)
    rm.decide_degradation_level.return_value = d.DegradationLevel.NORMAL
    return rm

class LateStartFailure(RecordingDispatcher):
    # start() gets the writer and worker running, then fails.
    async def start(self):
        await super().start()
        raise RuntimeError("start failed late")

async def checked(coro):
    # Await coro; afterwards report tasks other than this one still alive.
    try:
        await coro
    finally:
        left = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        print(f"@leftover_tasks={len(left)}")

def report(run):
    try:
        run()
        print("@outcome=ok")
    except BaseException as e:
        print(f"@outcome={type(e).__name__}")
        print(f"@cause={type(e.__cause__).__name__ if e.__cause__ else None}")
        for note in getattr(e, "__notes__", []):
            print(f"@note={note.split(':')[0]}")
        print(f"@notes_full={getattr(e, '__notes__', [])}")
"""

_MAIN = (
    _COMMON
    + r"""
class FakeOrchestrator:
    cancelled = False
    fail = False
    async def run(self):
        try:
            while True:
                await asyncio.sleep(0.05)
                if FakeOrchestrator.fail:
                    raise ConnectionError("orchestrator failed")
        except asyncio.CancelledError:
            FakeOrchestrator.cancelled = True
            raise

sys.path.insert(0, ".")
import main
main.AsyncDispatcher = RecordingDispatcher
main.resource_manager = fake_rm()
RELEASED = []

def producer(n_frames=None, fail_after=None, idle_after=None):
    async def run(sid, dispatcher, source):
        try:
            i = 0
            while n_frames is None or i < n_frames:
                if idle_after is not None and i == idle_after:
                    await asyncio.sleep(3600)  # stops sending; only a watcher can notice
                if fail_after is not None and i == fail_after and sid == "s1":
                    raise ValueError("producer failed")
                await dispatcher.put_frame(sid, torch.zeros(3, 8, 8), float(i))
                i += 1
                await asyncio.sleep(0.01)
        finally:
            RELEASED.append(sid)
    return run

def perception(**k):
    return main.run_perception(
        detector_factory=FakeDetector, producer=producer(**k), streams=("s1", "s2"), source="x"
    )

def after():
    disp = INSTANCES[-1]
    print(f"@state={disp.state.value}")
    print(f"@writer_closed={WRITER.close.call_count}")
    print(f"@released={sorted(RELEASED)}")
"""
)


def _run(body: str, prelude: str = _MAIN, timeout: float = 60.0) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), str(REPO), env.get("PYTHONPATH")) if p
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", prelude + body],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("@"):
            key, _, value = line[1:].partition("=")
            out[key] = f"{out[key]}|{value}" if key in out else value
    out["_stdout"] = proc.stdout
    return out


def _pick(out: dict[str, str], *keys: str) -> dict[str, str]:
    return {k: out[k] for k in keys if k in out}


# ── main.py ─────────────────────────────────────────────────────────────

_MAIN_KEYS = ("outcome", "cause", "note", "state", "writer_closed", "released")


def test_main_normal_end_stops_the_dispatcher() -> None:
    out = _run("report(lambda: asyncio.run(perception(n_frames=3)))\nafter()")
    assert _pick(out, *_MAIN_KEYS) == {
        "outcome": "ok",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
    }
    assert "online tracking = geometry-only" in out["_stdout"]
    assert "ReID is unsupported in the online tracking path" in out["_stdout"]
    assert "SigLIP" not in out["_stdout"]


def test_main_worker_failure_cancels_producers_and_stops() -> None:
    out = _run(
        "FakeDetector.fail = True\nreport(lambda: asyncio.run(perception()))\nafter()"
    )
    assert _pick(out, *_MAIN_KEYS) == {
        "outcome": "DispatcherFailed",
        "cause": "RuntimeError",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
    }


def test_main_worker_failure_is_noticed_while_producers_are_idle() -> None:
    out = _run(
        "FakeDetector.fail = True\nreport(lambda: asyncio.run(perception(idle_after=1)))\nafter()"
    )
    assert _pick(out, *_MAIN_KEYS) == {
        "outcome": "DispatcherFailed",
        "cause": "RuntimeError",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
    }


def test_main_producer_error_cancels_the_rest_and_stops() -> None:
    out = _run("report(lambda: asyncio.run(perception(fail_after=2)))\nafter()")
    assert _pick(out, *_MAIN_KEYS) == {
        "outcome": "ValueError",
        "cause": "None",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
    }


def test_main_cancellation_stops_the_dispatcher() -> None:
    out = _run(
        """
async def outer():
    t = asyncio.create_task(perception())
    await asyncio.sleep(0.3)
    t.cancel()
    await t
report(lambda: asyncio.run(outer()))
after()
"""
    )
    assert _pick(out, *_MAIN_KEYS) == {
        "outcome": "CancelledError",
        "cause": "None",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
    }


def test_main_cleanup_error_does_not_replace_the_original() -> None:
    out = _run(
        'WRITER.close.side_effect = OSError("close failed")\n'
        "report(lambda: asyncio.run(perception(fail_after=2)))\nafter()"
    )
    assert _pick(out, *_MAIN_KEYS) == {
        "outcome": "ValueError",
        "cause": "None",
        "note": "cleanup dispatcher.stop() also failed",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
    }
    assert "OSError('close failed')" in out["notes_full"]


def test_run_full_cancels_the_orchestrator_when_perception_fails() -> None:
    out = _run(
        """
FakeDetector.fail = True
async def outer():
    try:
        await main.run_full(perception=perception, orchestrator_factory=FakeOrchestrator)
    finally:  # checked before asyncio.run() tears down leftover tasks
        print(f"@orchestrator_cancelled={FakeOrchestrator.cancelled}")
report(lambda: asyncio.run(outer()))
after()
"""
    )
    assert _pick(out, *_MAIN_KEYS, "orchestrator_cancelled") == {
        "outcome": "DispatcherFailed",
        "cause": "RuntimeError",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
        "orchestrator_cancelled": "True",
    }


def test_run_full_stops_perception_when_the_orchestrator_fails() -> None:
    out = _run(
        """
FakeOrchestrator.fail = True
async def outer():
    try:
        await main.run_full(perception=perception, orchestrator_factory=FakeOrchestrator)
    finally:  # checked before asyncio.run() tears down leftover tasks
        print(f"@stopped_before_return={INSTANCES[-1].state.value}")
report(lambda: asyncio.run(outer()))
after()
"""
    )
    assert _pick(out, *_MAIN_KEYS, "stopped_before_return") == {
        "stopped_before_return": "stopped",
        "outcome": "ConnectionError",
        "cause": "None",
        "state": "stopped",
        "writer_closed": "1",
        "released": "['s1', 's2']",
    }


# ── Start-up exceptions (owner's merge-gate ruling) ───────────────────────

_START_KEYS = (*_MAIN_KEYS, "leftover_tasks")


def test_run_full_orchestrator_constructor_failure_leaves_nothing_running() -> None:
    out = _run(
        """
class BrokenOrchestrator:
    def __init__(self):
        raise ConnectionError("redis down")
report(lambda: asyncio.run(checked(main.run_full(perception=perception, orchestrator_factory=BrokenOrchestrator))))
print(f"@dispatchers={len(INSTANCES)}")
"""
    )
    assert _pick(out, "outcome", "cause", "leftover_tasks", "dispatchers") == {
        "outcome": "ConnectionError",
        "cause": "None",
        "leftover_tasks": "0",
        "dispatchers": "0",
    }


def test_main_partial_dispatcher_start_is_cleaned_up() -> None:
    out = _run(
        "main.AsyncDispatcher = LateStartFailure\n"
        "report(lambda: asyncio.run(checked(perception())))\nafter()"
    )
    assert _pick(out, *_START_KEYS) == {
        "outcome": "RuntimeError",
        "cause": "None",
        "leftover_tasks": "0",
        "state": "stopped",
        "writer_closed": "1",
        "released": "[]",
    }


def test_main_partial_start_keeps_the_original_over_a_cleanup_error() -> None:
    out = _run(
        'main.AsyncDispatcher = LateStartFailure\nWRITER.close.side_effect = OSError("close failed")\n'
        "report(lambda: asyncio.run(checked(perception())))\nafter()"
    )
    assert _pick(out, *_START_KEYS) == {
        "outcome": "RuntimeError",
        "cause": "None",
        "note": "cleanup dispatcher.stop() also failed",
        "leftover_tasks": "0",
        "state": "stopped",
        "writer_closed": "1",
        "released": "[]",
    }
    assert "start failed late" not in out["notes_full"]


def test_main_producer_task_creation_failure_is_cleaned_up() -> None:
    out = _run(
        """
good = producer()
def picky(sid, dispatcher, source):
    if sid == "s2":
        raise TypeError("bad stream config")
    return good(sid, dispatcher, source)
report(lambda: asyncio.run(checked(main.run_perception(
    detector_factory=FakeDetector, producer=picky, streams=("s1", "s2"), source="x"))))
after()
"""
    )
    assert _pick(out, *_START_KEYS) == {
        "outcome": "TypeError",
        "cause": "None",
        "leftover_tasks": "0",
        "state": "stopped",
        "writer_closed": "1",
        # s1's task was created but cancelled before it ran its first step
        "released": "[]",
    }


def test_main_reid_request_fails_fast() -> None:
    out = _run(
        """
import contextlib, io
err = io.StringIO()
sys.argv = ["main.py", "--mode", "perception", "--reid"]
with contextlib.redirect_stderr(err):
    try:
        main.main()
    except SystemExit as e:
        print(f"@exit={e.code}")
print(f"@message={'ReID is unsupported in the online tracking path' in err.getvalue()}")
print(f"@dispatchers={len(INSTANCES)}")
"""
    )
    assert _pick(out, "exit", "message", "dispatchers") == {
        "exit": "2",
        "message": "True",
        "dispatchers": "0",
    }


# ── run_8stream_perception.py ────────────────────────────────────────────

_SCRIPT = r"""
import importlib.util
spec = importlib.util.spec_from_file_location("run8", "scripts/ops/run_8stream_perception.py")
run8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run8)

RELEASED = []
class FakeMedia:
    grab_error = None
    frames = None  # None = endless; afterwards grab_tensor() reports no frame
    def __init__(self, rtsp_url):
        self.url = rtsp_url
    def connect(self):
        return True
    def grab_tensor(self):
        if FakeMedia.grab_error is not None:
            raise FakeMedia.grab_error
        if FakeMedia.frames is not None:
            if FakeMedia.frames <= 0:
                return False, None
            FakeMedia.frames -= 1
        return True, torch.zeros(16, 16, 3, dtype=torch.uint8)
    def release(self):
        RELEASED.append(self.url[-1])

class FakeTelemetry:
    last = None
    def __init__(self, sample_interval_sec):
        self.stopped = 0
        FakeTelemetry.last = self
    def start(self):
        pass
    def stop(self):
        self.stopped += 1
    def summary(self):
        return {}
    def observe(self, *a):
        pass
    def increment(self, *a):
        pass

RMS = []
def recording_rm():
    rm = fake_rm()
    RMS.append(rm)
    return rm

run8.MediaMTXClient = FakeMedia
run8.TRTYoloDetector = FakeDetector
run8.ResourceManager = recording_rm
run8.OnlineTelemetry = FakeTelemetry
run8.AsyncDispatcher = RecordingDispatcher

def args(*extra):
    return run8.build_parser().parse_args(
        ["--streams", "2", "--stats-interval-sec", "0.05", "--online-telemetry",
         "--input-size", "8", "--rtsp-prefix", "rtsp://x/s", *extra]
    )

def cancel_after(coro, delay):
    async def outer():
        t = asyncio.create_task(coro)
        await asyncio.sleep(delay)
        t.cancel()
        await t
    return outer()

def after():
    print(f"@telemetry_stopped={FakeTelemetry.last.stopped}")
    print(f"@released={sorted(RELEASED)}")
"""

_DISPATCHER_SCRIPT = (
    _COMMON
    + _SCRIPT
    + r"""
def after_dispatcher():
    after()
    print(f"@state={INSTANCES[-1].state.value}")
    print(f"@rm_closed={RMS[-1].close.call_count}")
"""
)

_POOL_SCRIPT = (
    _WB_PRELUDE
    + _COMMON
    + _SCRIPT
    + r"""
class RecordingPool(d.WorkbenchPool):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        globals()["POOL"] = self
run8.WorkbenchPool = RecordingPool

def after_pool():
    after()
    print(f"@pool_state={POOL.state.value}")
    print(f"@batcher_down={FakeBatcher.last.down}")
"""
)

_DISPATCHER_KEYS = (
    "outcome",
    "cause",
    "note",
    "state",
    "rm_closed",
    "telemetry_stopped",
    "released",
)


def test_run8_worker_failure_releases_everything() -> None:
    out = _run(
        "FakeDetector.fail = True\n"
        "report(lambda: asyncio.run(run8.run_multistream_perception(args())))\n"
        "after_dispatcher()",
        prelude=_DISPATCHER_SCRIPT,
    )
    assert _pick(out, *_DISPATCHER_KEYS) == {
        "outcome": "DispatcherFailed",
        "cause": "RuntimeError",
        "state": "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "['0', '1']",
    }


def test_run8_dispatcher_stop_error_still_releases_the_rest() -> None:
    out = _run(
        'FakeDetector.fail = True\nWRITER.close.side_effect = OSError("close failed")\n'
        "report(lambda: asyncio.run(run8.run_multistream_perception(args())))\n"
        "after_dispatcher()",
        prelude=_DISPATCHER_SCRIPT,
    )
    assert _pick(out, *_DISPATCHER_KEYS) == {
        "outcome": "DispatcherFailed",
        "cause": "RuntimeError",
        "note": "cleanup dispatcher.stop()",  # stop()'s own note: close failed
        "state": "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "['0', '1']",
    }
    assert "OSError('close failed')" in out["notes_full"]


def test_run8_worker_failure_is_noticed_while_streams_are_idle() -> None:
    out = _run(
        "FakeDetector.fail = True\nFakeMedia.frames = 1\n"
        "report(lambda: asyncio.run(run8.run_multistream_perception(args())))\n"
        "after_dispatcher()",
        prelude=_DISPATCHER_SCRIPT,
    )
    assert _pick(out, *_DISPATCHER_KEYS) == {
        "outcome": "DispatcherFailed",
        "cause": "RuntimeError",
        "state": "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "['0', '1']",
    }


def test_run8_producer_error_is_not_swallowed() -> None:
    out = _run(
        'FakeMedia.grab_error = IOError("rtsp read failed")\n'
        "report(lambda: asyncio.run(run8.run_multistream_perception(args())))\n"
        "after_dispatcher()",
        prelude=_DISPATCHER_SCRIPT,
    )
    assert _pick(out, *_DISPATCHER_KEYS) == {
        "outcome": "OSError",
        "cause": "None",
        "state": "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "['0', '1']",
    }


def test_run8_cancellation_releases_everything() -> None:
    out = _run(
        "report(lambda: asyncio.run(cancel_after(run8.run_multistream_perception(args()), 0.3)))\n"
        "after_dispatcher()",
        prelude=_DISPATCHER_SCRIPT,
    )
    assert _pick(out, *_DISPATCHER_KEYS) == {
        "outcome": "CancelledError",
        "cause": "None",
        "state": "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "['0', '1']",
    }


def test_run8_workbench_worker_failure_releases_everything() -> None:
    out = _run(
        'MODE["process"] = "raise"\n'
        'report(lambda: asyncio.run(run8.run_multistream_perception(args("--workbench"))))\n'
        "after_pool()",
        prelude=_POOL_SCRIPT,
    )
    assert _pick(
        out,
        "outcome",
        "cause",
        "note",
        "pool_state",
        "batcher_down",
        "telemetry_stopped",
        "released",
    ) == {
        "outcome": "WorkbenchPoolFailed",
        "cause": "RuntimeError",
        "pool_state": "stopped",
        "batcher_down": "True",
        "telemetry_stopped": "1",
        "released": "['0', '1']",
    }


def test_run8_workbench_stop_timeout_keeps_the_batcher() -> None:
    out = _run(
        'MODE["process"] = "block"\n'
        'report(lambda: asyncio.run(cancel_after(run8.run_multistream_perception(args("--workbench")), 0.3)))\n'
        "after_pool()\nGATE.set()",
        prelude=_POOL_SCRIPT,
    )
    assert _pick(
        out,
        "outcome",
        "note",
        "pool_state",
        "batcher_down",
        "telemetry_stopped",
        "released",
    ) == {
        "outcome": "CancelledError",
        "note": "cleanup pool.stop() also failed",
        "pool_state": "stopping",
        "batcher_down": "False",
        "telemetry_stopped": "1",
        "released": "['0', '1']",
    }


def test_run8_stream_url_failure_cancels_created_tasks() -> None:
    out = _run(
        """
real_url = run8.build_stream_url
def flaky_url(args, i):
    if i == 1:
        raise ValueError("bad rtsp url")
    return real_url(args, i)
run8.build_stream_url = flaky_url
report(lambda: asyncio.run(checked(run8.run_multistream_perception(args()))))
after_dispatcher()
""",
        prelude=_DISPATCHER_SCRIPT,
    )
    assert _pick(out, *_DISPATCHER_KEYS, "leftover_tasks") == {
        "outcome": "ValueError",
        "cause": "None",
        "leftover_tasks": "0",
        "state": "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "[]",
    }


def test_run8_partial_dispatcher_start_is_cleaned_up() -> None:
    out = _run(
        "run8.AsyncDispatcher = LateStartFailure\n"
        "report(lambda: asyncio.run(checked(run8.run_multistream_perception(args()))))\n"
        "after_dispatcher()",
        prelude=_DISPATCHER_SCRIPT,
    )
    assert _pick(out, *_DISPATCHER_KEYS, "leftover_tasks") == {
        "outcome": "RuntimeError",
        "cause": "None",
        "leftover_tasks": "0",
        "state": "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "[]",
    }


# ── #537 D-537-4A: monitor refusal, diagnosis and caller ownership ────────

_MONITOR = r"""
from saccade.resource.resource_manager import ResourceMonitorUnavailable
BACKEND = OSError("NVML driver unavailable")
MONITOR = ResourceMonitorUnavailable("NVML probe failed: NVML driver unavailable")
MONITOR.__cause__ = BACKEND
WRITER_FACTORY = MagicMock(return_value=WRITER)
d.VRAMLevelWriter = WRITER_FACTORY

async def checked_monitor(coro):
    try:
        await checked(coro)
    except BaseException as error:
        monitor = error if isinstance(error, ResourceMonitorUnavailable) else error.__cause__
        print(f"@monitor_identity={monitor is MONITOR}")
        print(f"@backend_identity={monitor.__cause__ is BACKEND}")
        raise

def monitor_after():
    dispatcher = INSTANCES[-1]
    stats = dispatcher.get_stats()
    print(f"@level={stats['current_level']}")
    print(f"@monitor_status={stats['resource_monitor_status']}")
    print(f"@monitor_error={stats['resource_monitor_error']}")
    print(f"@writer_created={WRITER_FACTORY.call_count}")
    print(f"@worker_missing={dispatcher._worker_task is None}")
    print(f"@queue_empty={dispatcher.queue.empty()}")
"""


@pytest.mark.parametrize("phase", ["startup", "runtime"])
def test_main_monitor_failure_preserves_diagnosis_and_cleans_up(phase: str) -> None:
    method = "get_stats" if phase == "startup" else "decide_degradation_level"
    out = _run(
        f"main.resource_manager.{method}.side_effect = MONITOR\n"
        "report(lambda: asyncio.run(checked_monitor(perception())))\n"
        "after()\nmonitor_after()",
        prelude=_MAIN + _MONITOR,
    )
    assert _pick(out, *_MAIN_KEYS, "leftover_tasks") == {
        "outcome": "ResourceMonitorUnavailable"
        if phase == "startup"
        else "DispatcherFailed",
        "cause": "OSError" if phase == "startup" else "ResourceMonitorUnavailable",
        "leftover_tasks": "0",
        "state": "created" if phase == "startup" else "stopped",
        "writer_closed": "0" if phase == "startup" else "1",
        "released": "[]" if phase == "startup" else "['s1', 's2']",
    }
    assert _pick(
        out,
        "monitor_identity",
        "backend_identity",
        "level",
        "monitor_status",
        "writer_created",
        "worker_missing",
        "queue_empty",
    ) == {
        "monitor_identity": "True",
        "backend_identity": "True",
        "level": "UNKNOWN",
        "monitor_status": "unavailable",
        "writer_created": "0" if phase == "startup" else "1",
        "worker_missing": "True",
        "queue_empty": "True",
    }
    assert "ResourceMonitorUnavailable" in out["monitor_error"]
    assert "NVML driver unavailable" in out["monitor_error"]


def test_main_orchestrator_only_remains_importable_when_nvml_init_fails() -> None:
    out = _run(
        r"""
import saccade.resource.resource_manager as rm
INIT_CALLS = []
class UnavailableNVML:
    def nvmlInit(self):
        INIT_CALLS.append("init")
        raise OSError("NVML driver unavailable")
rm.pynvml = UnavailableNVML()
sys.path.insert(0, ".")
import main
class OnlyOrchestrator:
    async def run(self):
        print("@orchestrator_ran=True")
main.PipelineOrchestrator = OnlyOrchestrator
sys.argv = ["main.py", "--mode", "orchestrator"]
report(main.main)
print(f"@init_calls={INIT_CALLS}")
print(f"@dispatchers={len(INSTANCES)}")
print(f"@initialized={main.resource_manager._nvml_initialized}")
""",
        prelude=_COMMON,
    )
    assert _pick(
        out, "outcome", "orchestrator_ran", "init_calls", "dispatchers", "initialized"
    ) == {
        "outcome": "ok",
        "orchestrator_ran": "True",
        "init_calls": "['init']",
        "dispatchers": "0",
        "initialized": "False",
    }


@pytest.mark.parametrize("phase", ["startup", "runtime"])
def test_run8_monitor_failure_preserves_diagnosis_and_releases_owned_resources(
    phase: str,
) -> None:
    method = "get_stats" if phase == "startup" else "decide_degradation_level"
    out = _run(
        f"""
def unavailable_rm():
    manager = recording_rm()
    manager.{method}.side_effect = MONITOR
    return manager
run8.ResourceManager = unavailable_rm
report(lambda: asyncio.run(checked_monitor(run8.run_multistream_perception(args()))))
after_dispatcher()
monitor_after()
""",
        prelude=_DISPATCHER_SCRIPT + _MONITOR,
    )
    assert _pick(out, *_DISPATCHER_KEYS, "leftover_tasks") == {
        "outcome": "ResourceMonitorUnavailable"
        if phase == "startup"
        else "DispatcherFailed",
        "cause": "OSError" if phase == "startup" else "ResourceMonitorUnavailable",
        "leftover_tasks": "0",
        "state": "created" if phase == "startup" else "stopped",
        "rm_closed": "1",
        "telemetry_stopped": "1",
        "released": "[]" if phase == "startup" else "['0', '1']",
    }
    assert _pick(
        out,
        "monitor_identity",
        "backend_identity",
        "level",
        "monitor_status",
        "writer_created",
        "worker_missing",
        "queue_empty",
    ) == {
        "monitor_identity": "True",
        "backend_identity": "True",
        "level": "UNKNOWN",
        "monitor_status": "unavailable",
        "writer_created": "0" if phase == "startup" else "1",
        "worker_missing": "True",
        "queue_empty": "True",
    }
    assert "ResourceMonitorUnavailable" in out["monitor_error"]
    assert "NVML driver unavailable" in out["monitor_error"]


@pytest.mark.parametrize("monitor_status", ["available", "unavailable"])
def test_run8_text_report_preserves_unknown_level_and_monitor_reason(
    monitor_status: str,
) -> None:
    out = _run(
        f"MONITOR_STATUS = {monitor_status!r}\n"
        + r"""
class MonitorStatsDispatcher(RecordingDispatcher):
    def get_stats(self):
        stats = super().get_stats()
        stats.update(
            current_level="UNKNOWN", resource_monitor_status=MONITOR_STATUS,
            resource_monitor_error=repr(MONITOR) if MONITOR_STATUS == "unavailable" else "",
        )
        return stats
run8.AsyncDispatcher = MonitorStatsDispatcher
async def idle_producer(*args):
    await asyncio.Event().wait()
run8.run_stream_producer = idle_producer
async def report_once(interval, report):
    report()
    raise RuntimeError("report complete")
run8._stats_loop = report_once
report(lambda: asyncio.run(checked(run8.run_multistream_perception(args()))))
after_dispatcher()
""",
        prelude=_DISPATCHER_SCRIPT + _MONITOR,
    )
    lines = [
        line for line in out["_stdout"].splitlines() if "Dispatcher Stats:" in line
    ]
    assert len(lines) == 1
    assert f"level=UNKNOWN monitor={monitor_status} " in lines[0]
    if monitor_status == "unavailable":
        assert "ResourceMonitorUnavailable" in lines[0]
        assert "NVML driver unavailable" in lines[0]
    else:
        assert "monitor_error='' " in lines[0]
    assert _pick(
        out, "outcome", "leftover_tasks", "rm_closed", "telemetry_stopped"
    ) == {
        "outcome": "RuntimeError",
        "leftover_tasks": "0",
        "rm_closed": "1",
        "telemetry_stopped": "1",
    }
