"""#538 D2-WB-0: the WorkbenchPool normal path, plus the harness that the contract tests share.

Characterization only. D2-WB-0 pinned 13 scenarios. Two later changes moved the
rest to ``test_workbench_pool_lifecycle_contract_538.py``:
- D2-WB-1 changed eight of them (WB-02..08, WB-10);
- the owner's final ruling changed four more (WB-09, WB-11..13).
Evidence from before each fix is in failure_degradation_537.md §12/§13/§15 and in
the PRs. The normal path (WB-01) remains here.

This module also holds the shared harness (``_PRELUDE`` and ``_run``). Every
scenario runs in a fresh interpreter under an outside timeout, with CPU fakes
for the batcher, Workbench and pipeline config. CUDA is out of scope (D2-GPU).
"""

# scope: perception
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

_PRELUDE = r"""
import asyncio, sys, threading, time, types
from types import SimpleNamespace
import torch
import saccade.perception.dispatcher as d

MODE = {"process": "ok", "construct_fail_at": None, "shutdown": None}
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
        # MODE["shutdown"]: None (legacy, no report), False (server still alive),
        # or "raise".
        if MODE["shutdown"] == "raise":
            raise OSError("batcher shutdown failed")
        self.down = True
        self.alive_at_shutdown = sum(w.is_alive() for w in POOL._workers)
        return MODE["shutdown"]

class FakeWorkbench:
    built = 0
    processed = 0
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
        FakeWorkbench.processed += 1
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
