"""#538 D2-WB-0: pin WorkbenchPool's CPU lifecycle under fault injection (status quo).

Characterization only. These record what ``WorkbenchPool`` does on ``main``
(R-1 and D2-AD did not touch it). They do not choose the target contract. The
owner's target is in failure_degradation_537.md §12, and D2-WB-1 owns the fix.

Every scenario runs in a fresh interpreter under an outside timeout, because
some of them hang by design (``stop()`` blocks forever). The batcher, Workbench
and pipeline config are CPU fakes. A real ``BatchingTRTDetector`` has a server
thread; after ``shutdown()``, a worker that calls ``submit()`` waits on an event
that nothing will set. The fake records such calls instead of blocking.

Not covered here: CUDA streams, events and buffer lifetime (D2-GPU).
"""

# scope: perception
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PRELUDE = r"""
import asyncio, sys, threading, time, types
from types import SimpleNamespace
import torch
import saccade.perception.dispatcher as d

MODE = {"process": "ok", "construct_fail_at": None}
GATE = threading.Event()
THREAD_ERRORS = []
threading.excepthook = lambda a: THREAD_ERRORS.append(f"{a.thread.name}:{type(a.exc_value).__name__}")

class FakeBatcher:
    last = None
    def __init__(self, engine_path, batch_size):
        self._device = "cpu"
        self.down = False
        self.used_after_shutdown = 0
        self.alive_at_shutdown = None
        FakeBatcher.last = self
    def make_proxy(self):
        return self
    def shutdown(self):
        self.down = True
        self.alive_at_shutdown = sum(w.is_alive() for w in POOL._workers)

class FakeWorkbench:
    built = 0
    def __init__(self, proxy, cfg, device, max_dets, max_tracks):
        if MODE["construct_fail_at"] == FakeWorkbench.built:
            raise RuntimeError("workbench construction failed")
        FakeWorkbench.built += 1
        self.batcher = proxy
    def process_frame(self, padded, frame_w, frame_h):
        mode = MODE["process"]
        if mode == "raise":
            raise RuntimeError("process_frame failed")
        if mode == "block":
            GATE.wait()
        if self.batcher.down:
            self.batcher.used_after_shutdown += 1
        z = torch.zeros(0)
        return SimpleNamespace(ids=z, boxes=z, classes=z)

sys.modules["saccade.perception.workbench"] = types.SimpleNamespace(Workbench=FakeWorkbench)
d.BatchingTRTDetector = FakeBatcher
d.PerceptionPipelineConfig = lambda: object()

RESULTS = []
def on_result(*args):
    RESULTS.append(args[1])

def frame():
    return torch.zeros(16, 16, 3, dtype=torch.uint8)

def make_pool(n=1):
    global POOL
    POOL = d.WorkbenchPool("fake.engine", n_streams=n, on_result=on_result, input_hw=(8, 8))
    return POOL

def wait_until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()

def timed_stop(pool):
    t0 = time.monotonic()
    pool.stop()
    return round(time.monotonic() - t0, 1)
"""


def _run(body: str, timeout: float = 30.0) -> dict[str, str] | None:
    """Run PRELUDE+body; return its ``key=value`` lines, or None when it hung."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH")) if p
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PRELUDE + body],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    assert proc.returncode == 0, proc.stderr[-3000:]
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("@"):
            key, _, value = line[1:].partition("=")
            out[key] = value
    return out


def test_normal_start_stop() -> None:
    out = _run(
        """
async def main():
    pool = make_pool(2)
    await pool.start()
    for i in range(3):
        pool.put_frame("stream_0", frame(), float(i))
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    await asyncio.sleep(0.1)
    print(f"@results={len(RESULTS)}")
    print(f"@stop_s={timed_stop(pool)}")
    print(f"@alive_at_shutdown={FakeBatcher.last.alive_at_shutdown}")
asyncio.run(main())
"""
    )
    assert out == {"results": "3", "stop_s": "0.0", "alive_at_shutdown": "0"}


def test_stuck_worker_stop_claims_success_and_shuts_down_batcher_in_use() -> None:
    out = _run(
        """
async def main():
    MODE["process"] = "block"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    print(f"@stop_s={timed_stop(pool)}")          # join(timeout=5.0), then shutdown
    print(f"@alive_at_shutdown={FakeBatcher.last.alive_at_shutdown}")
    print(f"@worker_alive_after_stop={pool._workers[0].is_alive()}")
    GATE.set()                                       # the worker resumes and uses the batcher
    wait_until(lambda: FakeBatcher.last.used_after_shutdown > 0)
    print(f"@used_after_shutdown={FakeBatcher.last.used_after_shutdown}")
asyncio.run(main())
"""
    )
    assert out == {
        "stop_s": "5.0",
        "alive_at_shutdown": "1",
        "worker_alive_after_stop": "True",
        "used_after_shutdown": "1",
    }


def test_stop_with_full_queue_and_stuck_worker_hangs() -> None:
    body = """
async def main():
    MODE["process"] = "block"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    for i in range(4):                               # fills the maxsize=4 queue
        pool.put_frame("stream_0", frame(), float(1 + i))
    print("@queue_depth=" + str(pool.get_stats()["queue_depth"]), flush=True)
    pool.stop()                                      # q.put(None) has no timeout
    print("@stop_returned=True")
asyncio.run(main())
"""
    assert _run(body, timeout=20.0) is None


def test_worker_exception_is_silent() -> None:
    out = _run(
        """
async def main():
    MODE["process"] = "raise"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: not pool._workers[0].is_alive())
    print(f"@worker_alive={pool._workers[0].is_alive()}")
    print(f"@thread_errors={THREAD_ERRORS}")
    print(f"@running={pool._running}")
    for i in range(6):                               # accepted; latest-wins drains
        pool.put_frame("stream_0", frame(), float(1 + i))
    stats = pool.get_stats()
    print(f"@drops={stats['drops']}")
    print(f"@error_keys={[k for k in stats if 'err' in k or 'health' in k or 'fail' in k]}")
    print(f"@stop_s={timed_stop(pool)}")
asyncio.run(main())
"""
    )
    assert out == {
        "worker_alive": "False",
        "thread_errors": "['wb-stream_0:RuntimeError']",
        "running": "True",
        "drops": "1",
        "error_keys": "[]",
        "stop_s": "0.0",
    }


def test_dead_worker_with_full_queue_makes_stop_hang() -> None:
    body = """
async def main():
    MODE["process"] = "raise"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: not pool._workers[0].is_alive())
    for i in range(4):
        pool.put_frame("stream_0", frame(), float(1 + i))
    pool.stop()
    print("@stop_returned=True")
asyncio.run(main())
"""
    assert _run(body, timeout=20.0) is None


def test_partial_start_failure_leaves_resources_running() -> None:
    out = _run(
        """
async def main():
    MODE["construct_fail_at"] = 1
    pool = make_pool(3)
    try:
        await pool.start()
    except RuntimeError as e:
        print(f"@start_error={e}")
    print(f"@running={pool._running}")
    print(f"@workers_alive={sum(w.is_alive() for w in pool._workers)}")
    print(f"@queues={len(pool._queues)}")
    print(f"@batcher_down={FakeBatcher.last.down}")
asyncio.run(main())
"""
    )
    assert out == {
        "start_error": "workbench construction failed",
        "running": "True",
        "workers_alive": "1",
        "queues": "2",
        "batcher_down": "False",
    }


def test_missing_extension_after_batcher_leaks_batcher() -> None:
    out = _run(
        """
async def main():
    d.PerceptionPipelineConfig = None
    pool = make_pool(1)
    try:
        await pool.start()
    except RuntimeError as e:
        print(f"@start_error={type(e).__name__}")
    print(f"@running={pool._running}")
    print(f"@batcher_created={FakeBatcher.last is not None}")
    print(f"@batcher_down={FakeBatcher.last.down}")
asyncio.run(main())
"""
    )
    assert out == {
        "start_error": "RuntimeError",
        "running": "True",
        "batcher_created": "True",
        "batcher_down": "False",
    }


def test_repeated_stop_and_restart() -> None:
    out = _run(
        """
async def main():
    pool = make_pool(1)
    await pool.start()
    print(f"@stop1_s={timed_stop(pool)}")
    print(f"@stop2_s={timed_stop(pool)}")       # a second stop() returns
    await pool.start()                           # restart appends to the old lists
    print(f"@queues={len(pool._queues)}")
    print(f"@workers_alive={[w.is_alive() for w in pool._workers]}")
    pool.put_frame("stream_0", frame(), 0.0)     # index 0 is the old, dead queue
    await asyncio.sleep(0.3)
    print(f"@results={len(RESULTS)}")
    print(f"@old_queue_depth={pool._queues[0].qsize()}")
    timed_stop(pool)
asyncio.run(main())
"""
    )
    assert out == {
        "stop1_s": "0.0",
        "stop2_s": "0.0",
        "queues": "2",
        "workers_alive": "[False, True]",
        "results": "0",
        "old_queue_depth": "2",  # the second stop's sentinel + the new frame
    }


def test_late_frame_after_stop_is_accepted_silently() -> None:
    out = _run(
        """
async def main():
    pool = make_pool(1)
    await pool.start()
    timed_stop(pool)
    for i in range(6):
        pool.put_frame("stream_0", frame(), float(i))
    print(f"@queue_depth={pool.get_stats()['queue_depth']}")
    print(f"@drops={pool.get_stats()['drops']}")
asyncio.run(main())
"""
    )
    # The stop sentinel was consumed; 6 puts into maxsize=4 → one drain → 2 left.
    assert out == {"queue_depth": "2", "drops": "1"}


def test_late_callback_after_loop_closed_kills_worker() -> None:
    out = _run(
        """
async def main():
    MODE["process"] = "block"
    pool = make_pool(1)
    await pool.start()
    pool.put_frame("stream_0", frame(), 0.0)
    wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
    return pool
pool = asyncio.run(main())                       # the loop is closed now
GATE.set()
wait_until(lambda: not pool._workers[0].is_alive())
print(f"@thread_errors={THREAD_ERRORS}")
print(f"@results={len(RESULTS)}")
"""
    )
    assert out == {"thread_errors": "['wb-stream_0:RuntimeError']", "results": "0"}


def test_async_on_result_is_never_awaited() -> None:
    out = _run(
        """
import warnings
CALLED = []
async def async_on_result(*args):
    CALLED.append(args[1])
async def main():
    pool = d.WorkbenchPool("fake.engine", n_streams=1, on_result=async_on_result, input_hw=(8, 8))
    globals()["POOL"] = pool
    await pool.start()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pool.put_frame("stream_0", frame(), 0.0)
        wait_until(lambda: pool.get_stats()["queue_depth"] == 0)
        await asyncio.sleep(0.2)
        import gc; gc.collect()
    print(f"@called={len(CALLED)}")
    print(f"@never_awaited={any('never awaited' in str(w.message) for w in caught)}")
    timed_stop(pool)
asyncio.run(main())
"""
    )
    assert out == {"called": "0", "never_awaited": "True"}


@pytest.mark.parametrize("stream_id", ["stream_9", "cam"])
def test_put_frame_stream_mapping(stream_id: str) -> None:
    out = _run(
        f"""
async def main():
    pool = make_pool(1)
    await pool.start()
    pool.put_frame({stream_id!r}, frame(), 0.0)
    await asyncio.sleep(0.3)
    print(f"@results={{len(RESULTS)}}")
    timed_stop(pool)
asyncio.run(main())
"""
    )
    # "stream_9" is out of range and silently dropped; an id without "_" maps to index 0.
    assert out == {"results": "0" if stream_id == "stream_9" else "1"}
