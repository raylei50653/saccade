"""#538 D2-GPU-0: CUDA ordering, output ownership and batcher failure, as they are.

Characterization only; no production code changes here. Owner ruling (2026-10-22):
pin the three gaps that the read-only probe reproduced, before D2-GPU-1 fixes them.

- G1: ``_WorkbenchWorker._process`` preprocesses on the worker thread's current
  stream (legacy stream 0). ``Workbench.process_frame`` then switches to its own
  non-blocking stream, and ``BatchingTRTDetector.submit`` records the input
  event on that stream. Nothing orders the preprocessing before the server's
  read. Shown with an injected delay only; no claim on the incidence under
  natural load.
- G2: the result tensors handed to ``on_result`` are views of
  ``Workbench.out_*``, which the next frame overwrites.
- G3: an exception in the batcher server ends its thread. ``submit()`` waits
  without a timeout, so the worker blocks for good and the pool never reaches
  FAILED.

Positive control: AsyncDispatcher hands out tensors that later frames leave intact.

G1..G3 run the real WorkbenchPool, Workbench and BatchingTRTDetector, in a fresh
interpreter under an outside timeout. Only ``TRTYoloDetector`` is a fake (no
engine). The fake reports which input values the server read. See
failure_degradation_537.md §17.
"""

# scope: perception
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from saccade.perception import dispatcher as dmod
from saccade.resource.resource_manager import DegradationLevel, ResourceManager


REPO = Path(__file__).resolve().parents[2]

_PRELUDE = r"""
import asyncio, os, sys, threading, time
import torch
import saccade.perception.detector_trt as dt
import saccade.perception.dispatcher as d

THREAD_ERRORS = []
threading.excepthook = lambda a: THREAD_ERRORS.append(
    f"{a.thread.name}:{type(a.exc_value).__name__}:{a.exc_value}"
)

PLAN = {"boxes": [], "raise": False}  # detections per submitted frame, in order
SEEN = []  # max input value the server read, per frame

class FakeTRT:
    output_name = "output"
    def __init__(self, *args, **kwargs):
        pass
    def infer_raw_batch(self, frames):
        if PLAN["raise"]:
            raise RuntimeError("injected TRT failure")
        b = frames.shape[0]
        # frames is the server's torch.cat copy, so this is what the server read.
        SEEN.extend(frames.reshape(b, -1).amax(dim=1).tolist())
        out = torch.zeros((b, 300, 6), device=frames.device)
        for i in range(b):
            if PLAN["boxes"]:
                bx = PLAN["boxes"].pop(0)
                out[i, : bx.shape[0], :4] = bx
                out[i, : bx.shape[0], 4] = 0.9
        return {"output": out}

dt.TRTYoloDetector = FakeTRT

# Delay injection: a GPU sleep on the calling stream right before the letterbox
# interpolate, so the last preprocessing writes land late on that stream.
INJECT = {"delay": 0, "sync": False}
_interpolate = torch.nn.functional.interpolate
def _delayed_interpolate(*args, **kwargs):
    if INJECT["delay"]:
        torch.cuda._sleep(INJECT["delay"])
    out = _interpolate(*args, **kwargs)
    if INJECT["sync"]:
        torch.cuda.current_stream().synchronize()
    return out
torch.nn.functional.interpolate = _delayed_interpolate

RESULTS = []
def on_result(sid, ts, ids, boxes, classes, kps):
    RESULTS.append(dict(
        ts=ts, ids=ids, boxes=boxes, classes=classes,
        ids0=ids.clone(), boxes0=boxes.clone(), classes0=classes.clone(),
    ))

def frame(value):
    # 48x64 HWC letterboxes into 64x64 with no scaling; the max is value/255.
    return torch.full((48, 64, 3), value, dtype=torch.uint8, device="cuda")

def make_pool():
    return d.WorkbenchPool(
        "fake.engine", n_streams=1, on_result=on_result, input_hw=(64, 64),
        frame_size=(1920, 1080), max_dets=300, max_tracks=64,
    )

async def feed(pool, frames, timeout=20.0):
    # One frame at a time: put, then wait until its result was delivered.
    for i, f in enumerate(frames):
        n0 = len(RESULTS)
        pool.put_frame("stream_0", f, float(i))
        end = time.monotonic() + timeout
        while len(RESULTS) == n0:
            if time.monotonic() > end:
                raise TimeoutError(f"frame {i} was not delivered")
            await asyncio.sleep(0.005)

def done():
    sys.stdout.flush()
    os._exit(0)  # do not finalize with worker threads still blocked (G3)
"""


def _run(body: str, timeout: float = 60.0) -> dict[str, str]:
    """Run PRELUDE+body on the GPU; return its ``@key=value`` lines."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH")) if p
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PRELUDE + body],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"scenario hung past the {timeout}s outside timeout: {e}")
    assert proc.returncode == 0, proc.stderr[-3000:]
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("@"):
            key, _, value = line[1:].partition("=")
            out[key] = value
    return out


def _need_native() -> None:
    pytest.importorskip("saccade_tracking_ext")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


@pytest.mark.gpu
def test_g1_preprocessing_is_not_ordered_before_the_server_read() -> None:
    _need_native()
    out = _run(
        """
N = 10
async def main():
    pool = make_pool()
    await pool.start()
    for mode, sync in (("delay", False), ("delay_sync", True)):
        INJECT.update(delay=200_000_000, sync=sync)
        SEEN.clear()
        values = [10 * (i + 1) for i in range(N)]
        await feed(pool, [frame(v) for v in values])
        stale = sum(abs(s - v / 255.0) > 1e-4 for s, v in zip(SEEN, values))
        print(f"@{mode}_read={len(SEEN)}")
        print(f"@{mode}_stale={stale}")
    print(f"@state={pool.state.name}")
    print(f"@failure={pool.failure!r}")
    pool.stop()
asyncio.run(main())
done()
"""
    )
    # Every frame reached the server once in both modes.
    assert out["delay_read"] == out["delay_sync_read"] == "10"
    # Missing ordering: with the injected delay the server read inputs that
    # preprocessing had not finished writing. The probe saw 11/20; the count
    # varies with timing, so only "at least one" is pinned.
    assert int(out["delay_stale"]) >= 1, out
    # Control: the same delay with a host sync after it gives no stale reads, so
    # the stale reads come from the missing ordering, not from the harness.
    assert out["delay_sync_stale"] == "0", out
    # The gap is silent: the pool stays RUNNING with no failure.
    assert (out["state"], out["failure"]) == ("RUNNING", "None")


@pytest.mark.gpu
def test_g2_result_tensors_alias_buffers_the_next_frame_overwrites() -> None:
    _need_native()
    out = _run(
        """
A = torch.tensor([[100., 100, 150, 200], [300, 100, 350, 200]], device="cuda")
SHIFT = torch.tensor([[0., 0, 0, 0], [5, 0, 5, 0]], device="cuda")
async def main():
    pool = make_pool()
    await pool.start()
    PLAN["boxes"] = [A + k for k in range(3)] + [A + 3 + SHIFT]
    await feed(pool, [frame(10)] * 4)
    a, b = RESULTS[2], RESULTS[3]  # frame 3 (tracks confirmed) and frame 4
    print(f"@n={a['ids'].numel()},{b['ids'].numel()}")
    for name in ("ids", "boxes", "classes"):
        print(f"@{name}_same_ptr={a[name].data_ptr() == b[name].data_ptr()}")
        print(f"@{name}_ptr={a[name].data_ptr()}")
        print(f"@{name}_moved={not torch.equal(a[name], a[name + '0'])}")
    print(f"@a_boxes_now_b={torch.equal(a['boxes'], b['boxes0'])}")
    print(f"@b_differs_from_a={not torch.equal(a['boxes0'], b['boxes0'])}")
    pool.stop()
asyncio.run(main())
done()
"""
    )
    assert out["n"] == "2,2", out
    # All three outputs of frame 3 share storage with frame 4's.
    for name in ("ids", "boxes", "classes"):
        assert out[f"{name}_same_ptr"] == "True", out
        assert int(out[f"{name}_ptr"]) != 0, out
    # Frame 4 had different boxes, and frame 3's held boxes now show them.
    assert out["b_differs_from_a"] == "True", out
    assert out["boxes_moved"] == "True", out
    assert out["a_boxes_now_b"] == "True", out
    # ids/classes keep their values here only because the tracks persist with
    # the same class; the shared storage above is the ownership evidence.
    assert out["ids_moved"] == out["classes_moved"] == "False", out


@pytest.mark.gpu
def test_g3_batcher_server_failure_blocks_the_worker_and_the_pool() -> None:
    _need_native()
    out = _run(
        """
async def main():
    PLAN["raise"] = True
    pool = make_pool()
    await pool.start()
    batcher = pool._batcher
    pool.put_frame("stream_0", frame(10), 0.0)
    await asyncio.sleep(3.0)
    print(f"@server_alive={batcher._server.is_alive()}")
    print(f"@worker_alive={pool._workers[0].is_alive()}")
    print(f"@state={pool.state.name}")
    print(f"@failure={pool.failure!r}")
    print(f"@thread_errors={THREAD_ERRORS}")
    # A new submit after the server ended blocks as well.
    late = threading.Thread(
        target=lambda: batcher.make_proxy().detect_raw(
            torch.zeros(3, 64, 64, device="cuda")
        ),
        daemon=True,
    )
    late.start()
    late.join(2.0)
    print(f"@late_submit_blocked={late.is_alive()}")
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            pool.stop(timeout=1.0)
            result = "returned"
        except Exception as e:
            result = type(e).__name__
        print(f"@stop{attempt}={result}")
        print(f"@stop{attempt}_s={time.monotonic() - t0:.1f}")
        print(f"@stop{attempt}_state={pool.state.name}")
    print(f"@batcher_kept={pool._batcher is batcher}")
asyncio.run(main())
done()
"""
    )
    # The server thread ended; the worker is still blocked in submit().
    assert (out["server_alive"], out["worker_alive"]) == ("False", "True"), out
    # The pool did not see the failure.
    assert (out["state"], out["failure"]) == ("RUNNING", "None"), out
    # The original exception reached only threading.excepthook.
    assert "BatchTRT-server:RuntimeError:injected TRT failure" in out["thread_errors"]
    assert out["late_submit_blocked"] == "True", out
    # stop() keeps its bound but can never complete: each retry times out, the
    # state stays STOPPING and the batcher is kept.
    for attempt in ("1", "2"):
        assert out[f"stop{attempt}"] == "WorkbenchPoolStopTimeout", out
        assert float(out[f"stop{attempt}_s"]) < 3.0, out
        assert out[f"stop{attempt}_state"] == "STOPPING", out
    assert out["batcher_kept"] == "True", out


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_async_dispatcher_results_are_owned_by_the_consumer() -> None:
    """Positive control: same method as G2, on the AsyncDispatcher path."""
    _need_native()
    held: list[dict[str, torch.Tensor]] = []

    async def on_result(sid, ts, ids, boxes, classes, kpts) -> None:  # type: ignore[no-untyped-def]
        held.append(
            dict(
                ids=ids,
                boxes=boxes,
                classes=classes,
                ids0=ids.clone(),
                boxes0=boxes.clone(),
                classes0=classes.clone(),
            )
        )

    calls = iter(range(100))

    def detect_batch(x, conf_threshold):  # type: ignore[no-untyped-def]
        out = []
        for _ in range(x.shape[0]):
            k = float(next(calls))
            out.append(
                (
                    torch.tensor(
                        [[200.0, 200.0, 260.0, 330.0], [500.0, 200.0, 560.0, 330.0]],
                        device="cuda",
                    )
                    + 2.0 * k,
                    torch.tensor([0.9, 0.9], device="cuda"),
                    torch.tensor([0, 0], dtype=torch.int32, device="cuda"),
                    None,
                )
            )
        return out

    det = MagicMock()
    det.detect_batch.side_effect = detect_batch
    rm = MagicMock(spec=ResourceManager)
    rm.decide_degradation_level.return_value = DegradationLevel.NORMAL
    with patch.object(dmod, "VRAMLevelWriter", return_value=MagicMock()):
        disp = dmod.AsyncDispatcher(
            det, rm, max_batch=1, input_hw=(64, 64), on_track_result=on_result
        )
        await disp.start()
        for i in range(6):
            await disp.put_frame("cam0", torch.rand(3, 64, 64, device="cuda"), float(i))
            await asyncio.wait_for(disp.queue.join(), timeout=10.0)
        await disp.stop()

    confirmed = [h for h in held if h["ids"].numel() == 2]
    assert len(confirmed) >= 2
    # Consecutive frames differ, so an aliased buffer would show here.
    assert not torch.equal(confirmed[-2]["boxes0"], confirmed[-1]["boxes0"])
    for h in held:
        for name in ("ids", "boxes", "classes"):
            assert torch.equal(h[name], h[name + "0"]), name
            assert h[name].device.type == "cuda"
    for name in ("ids", "boxes", "classes"):
        ptrs = [h[name].data_ptr() for h in confirmed]
        assert len(set(ptrs)) == len(ptrs), name
