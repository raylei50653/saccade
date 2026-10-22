"""#538 D2-GPU-1: CUDA ordering, output ownership and batcher failure contracts.

D2-GPU-0 (#617) pinned three gaps; this module holds the contracts that fix them.
Owner rulings (2026-10-22):

- G1: Workbench's stream waits for the worker's preprocessing stream before it
  reads the input. The RGB path and the NV12 kernel are both covered.
- G2: ``on_result`` receives copies that the consumer owns, ordered after the
  Workbench's writes and before the next frame overwrites its buffers. Not a
  "valid until the next frame" contract.
- G3: a terminal server error wakes every waiting ``submit()`` with
  ``BatchingDetectorFailed`` (``__cause__`` is the original error); later
  submits fail at once; the batcher is FAILED; WorkbenchPool reports FAILED
  through its worker-failure channel and ``stop()`` completes. A shut-down
  batcher also wakes its waiters and refuses new submits.

Positive control: AsyncDispatcher hands out tensors that later frames leave intact.

The pool scenarios run the real WorkbenchPool, Workbench and BatchingTRTDetector,
in a fresh interpreter under an outside timeout. Only ``TRTYoloDetector`` is a
fake (no engine); it reports which input values the server read. The NV12
kernel is a fake that writes the destination on the stream it is given (this
build has no ``nv12_to_chw_letterbox``). See failure_degradation_537.md §17/§18.
"""

# scope: perception
# function: contract
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
import asyncio, os, struct, sys, threading, time
import torch
import saccade.perception.detector_trt as dt
import saccade.perception.dispatcher as d
from saccade.perception.zero_copy import Nv12Frame

THREAD_ERRORS = []
threading.excepthook = lambda a: THREAD_ERRORS.append(
    f"{a.thread.name}:{type(a.exc_value).__name__}:{a.exc_value}"
)

PLAN = {"boxes": [], "raise": False, "gate": None}
SEEN = []  # max input value the server read, per frame
IN_INFER = threading.Event()
ORIGINAL = RuntimeError("injected TRT failure")

class FakeTRT:
    output_name = "output"
    def __init__(self, *args, **kwargs):
        pass
    def infer_raw_batch(self, frames):
        IN_INFER.set()
        if PLAN["gate"] is not None:
            PLAN["gate"].wait()
        if PLAN["raise"]:
            raise ORIGINAL
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

DELAY = 200_000_000  # GPU sleep cycles for the injected delays

# RGB delay injection: a GPU sleep on the calling stream right before the
# letterbox interpolate, so the last preprocessing writes land late.
INJECT = {"delay": False}
_interpolate = torch.nn.functional.interpolate
def _delayed_interpolate(*args, **kwargs):
    if INJECT["delay"]:
        torch.cuda._sleep(DELAY)
    return _interpolate(*args, **kwargs)
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

def make_pool(n=1, **kwargs):
    return d.WorkbenchPool(
        "fake.engine", n_streams=n, on_result=on_result, input_hw=(64, 64),
        frame_size=(1920, 1080), max_dets=300, max_tracks=64, **kwargs,
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

async def until(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.005)

def track_boxes(k):
    # Two boxes moving 4 px per frame; frames far enough apart to tell apart.
    base = torch.tensor([[100., 100, 150, 200], [300, 100, 350, 200]], device="cuda")
    return base + 4.0 * k

def check_frames(results, n_frames):
    # Each delivered result with both tracks must show its own frame's boxes.
    ok = mismatched = 0
    for r in results:
        if r["ids0"].numel() != 2:
            continue
        k = int(r["ts"])
        want = track_boxes(k)
        got = r["boxes0"][r["boxes0"][:, 0].argsort()]
        if torch.allclose(got, want, atol=1.5):
            ok += 1
        else:
            mismatched += 1
    return ok, mismatched

def done():
    sys.stdout.flush()
    os._exit(0)  # never finalize with threads still running
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


# ── G1: preprocessing is ordered before the server's read ────────────────


@pytest.mark.gpu
def test_g1_rgb_preprocessing_is_ordered_before_the_server_read() -> None:
    _need_native()
    out = _run(
        """
N = 10
async def main():
    pool = make_pool()
    await pool.start()
    INJECT["delay"] = True
    values = [10 * (i + 1) for i in range(N)]
    await feed(pool, [frame(v) for v in values])
    stale = sum(abs(s - v / 255.0) > 1e-4 for s, v in zip(SEEN, values))
    print(f"@read={len(SEEN)}")
    print(f"@stale={stale}")
    print(f"@state={pool.state.name}")
    pool.stop()
asyncio.run(main())
done()
"""
    )
    # D2-GPU-0 saw 9/10 stale reads with this delay; the server now waits.
    assert (out["read"], out["stale"], out["state"]) == ("10", "0", "RUNNING"), out


@pytest.mark.gpu
def test_g1_nv12_kernel_writes_are_ordered_before_the_server_read() -> None:
    _need_native()
    out = _run(
        """
from cuda.bindings import driver as cu
N = 10
EXPECTED = []
STREAMS = []
def fake_nv12(y_ptr, y_pitch, uv_ptr, uv_pitch, src_w, src_h, dst_ptr, dst_w,
              x_off, y_off, new_w, new_h, pad, stream):
    # Writes the whole 3x64x64 destination late, on the stream it was given.
    STREAMS.append(stream)
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        torch.cuda._sleep(DELAY)
    bits = struct.unpack("<I", struct.pack("<f", EXPECTED.pop(0)))[0]
    (err,) = cu.cuMemsetD32Async(dst_ptr, bits, 3 * 64 * 64, stream)
    assert int(err) == 0, err
d._nv12_letterbox_kernel = fake_nv12

def nv12(value):
    buf = torch.full((72 * 64,), value, dtype=torch.uint8, device="cuda")
    return Nv12Frame(buf=buf, width=64, height=48, y_pitch=64, uv_pitch=64)

async def main():
    pool = make_pool(use_nv12=True)
    await pool.start()
    values = [(i + 1) / 16.0 for i in range(N)]
    EXPECTED.extend(values)
    await feed(pool, [nv12(1) for _ in values])
    stale = sum(abs(s - v) > 1e-6 for s, v in zip(SEEN, values))
    print(f"@read={len(SEEN)}")
    print(f"@stale={stale}")
    print(f"@streams={sorted(set(STREAMS))}")
    pool.stop()
asyncio.run(main())
done()
"""
    )
    assert (out["read"], out["stale"]) == ("10", "0"), out
    # The kernel ran on the worker's current stream (the legacy stream).
    assert out["streams"] == "[0]", out


# ── G2: the consumer owns what on_result receives ────────────────────────


@pytest.mark.gpu
def test_g2_delivered_results_survive_later_frames() -> None:
    _need_native()
    out = _run(
        """
async def main():
    pool = make_pool()
    await pool.start()
    PLAN["boxes"] = [track_boxes(k) for k in range(4)]
    await feed(pool, [frame(10)] * 4)
    a, b = RESULTS[2], RESULTS[3]
    print(f"@n={a['ids'].numel()},{b['ids'].numel()}")
    for name in ("ids", "boxes", "classes"):
        print(f"@{name}_same_ptr={a[name].data_ptr() == b[name].data_ptr()}")
        print(f"@{name}_kept={torch.equal(a[name], a[name + '0'])}")
    print(f"@b_differs_from_a={not torch.equal(a['boxes0'], b['boxes0'])}")
    print(f"@frames={check_frames(RESULTS, 4)}")
    pool.stop()
asyncio.run(main())
done()
"""
    )
    assert out["n"] == "2,2", out
    assert out["b_differs_from_a"] == "True", out
    for name in ("ids", "boxes", "classes"):
        assert out[f"{name}_same_ptr"] == "False", out
        assert out[f"{name}_kept"] == "True", out
    assert out["frames"].endswith(", 0)"), out


@pytest.mark.gpu
def test_g2_results_queued_behind_a_blocked_event_loop_keep_their_frame() -> None:
    _need_native()
    out = _run(
        """
N = 6
async def main():
    pool = make_pool()
    await pool.start()
    PLAN["boxes"] = [track_boxes(k) for k in range(N)]
    for k in range(N):
        pool.put_frame("stream_0", frame(10), float(k))
        # Block the event loop until the worker has processed this frame, so
        # every callback is still queued while the next frame runs.
        while len(pool._stats_window) <= k:
            time.sleep(0.005)
    print(f"@queued_before_delivery={len(RESULTS)}")
    await until(lambda: len(RESULTS) == N)
    ok, bad = check_frames(RESULTS, N)
    print(f"@ok={ok}")
    print(f"@mismatched={bad}")
    pool.stop()
asyncio.run(main())
done()
"""
    )
    assert out["queued_before_delivery"] == "0", out
    assert int(out["ok"]) >= 2 and out["mismatched"] == "0", out


@pytest.mark.gpu
def test_g2_the_next_frame_waits_for_the_previous_copies() -> None:
    _need_native()
    out = _run(
        """
N = 4  # the per-stream queue holds 4; more would overflow and drain it
_own = d._own_result
def delayed_own(result):
    # Queue the copies late on the worker's stream; the next frame must not
    # overwrite the Workbench buffers before they run.
    torch.cuda._sleep(DELAY)
    return _own(result)
d._own_result = delayed_own

async def main():
    pool = make_pool()
    await pool.start()
    PLAN["boxes"] = [track_boxes(k) for k in range(N)]
    for k in range(N):  # all queued at once: the worker runs ahead of the copies
        pool.put_frame("stream_0", frame(10), float(k))
    await until(lambda: len(RESULTS) == N)
    ok, bad = check_frames(RESULTS, N)
    print(f"@ok={ok}")
    print(f"@mismatched={bad}")
    pool.stop()
asyncio.run(main())
done()
"""
    )
    assert int(out["ok"]) >= 2 and out["mismatched"] == "0", out


# ── G3: batcher failure reaches every waiter and the pool ────────────────


@pytest.mark.gpu
def test_g3_server_failure_wakes_every_waiter_with_the_original_error() -> None:
    _need_native()
    out = _run(
        """
PLAN["raise"] = True
PLAN["gate"] = threading.Event()
batcher = dt.BatchingTRTDetector("fake", batch_size=2)
proxy = batcher.make_proxy()
caught = {}
def submit(name):
    try:
        proxy.detect_raw(torch.zeros(3, 8, 8, device="cuda"))
        caught[name] = "returned"
    except Exception as e:
        caught[name] = (type(e).__name__, e.__cause__ is ORIGINAL)
threads = [threading.Thread(target=submit, args=(n,)) for n in ("a", "b")]
for t in threads:
    t.start()
assert IN_INFER.wait(10.0)  # a and b are the batch in flight
threads.append(threading.Thread(target=submit, args=("c",)))  # c stays pending
threads[-1].start()
time.sleep(0.2)
print(f"@pending_before={len(batcher._pending)}")
PLAN["gate"].set()
for t in threads:
    t.join(5.0)
print(f"@blocked={[t.is_alive() for t in threads]}")
print(f"@caught={sorted(caught.items())}")
print(f"@state={batcher.state.name}")
print(f"@failure_is_original={batcher.failure is ORIGINAL}")
print(f"@server_alive={batcher._server.is_alive()}")
t0 = time.monotonic()
try:
    proxy.detect_raw(torch.zeros(3, 8, 8, device="cuda"))
    late = "returned"
except Exception as e:
    late = (type(e).__name__, e.__cause__ is ORIGINAL)
print(f"@late={late}")
print(f"@late_s={time.monotonic() - t0:.2f}")
print(f"@shutdown={batcher.shutdown()}")
print(f"@thread_errors={THREAD_ERRORS}")
done()
"""
    )
    assert out["pending_before"] == "1", out
    assert out["blocked"] == "[False, False, False]", out
    failed = "('BatchingDetectorFailed', True)"
    assert out["caught"] == f"[('a', {failed}), ('b', {failed}), ('c', {failed})]"
    assert (out["state"], out["failure_is_original"]) == ("FAILED", "True"), out
    assert out["server_alive"] == "False", out
    assert out["late"] == failed and float(out["late_s"]) < 0.5, out
    assert out["shutdown"] == "True", out
    # The error is handled, not left to threading.excepthook.
    assert out["thread_errors"] == "[]", out


@pytest.mark.gpu
def test_g3_shutdown_wakes_pending_submits_and_refuses_new_ones() -> None:
    _need_native()
    out = _run(
        """
PLAN["gate"] = threading.Event()
batcher = dt.BatchingTRTDetector("fake", batch_size=1)
proxy = batcher.make_proxy()
caught = {}
def submit(name):
    try:
        proxy.detect_raw(torch.zeros(3, 8, 8, device="cuda"))
        caught[name] = "returned"
    except Exception as e:
        caught[name] = type(e).__name__
first = threading.Thread(target=submit, args=("in_flight",))
first.start()
assert IN_INFER.wait(10.0)
second = threading.Thread(target=submit, args=("pending",))
second.start()
time.sleep(0.2)
# shutdown() while the server is busy: its join times out and reports False.
print(f"@shutdown_busy={batcher.shutdown()}")
PLAN["gate"].set()
first.join(5.0)
second.join(5.0)
print(f"@blocked={[first.is_alive(), second.is_alive()]}")
print(f"@caught={sorted(caught.items())}")
print(f"@state={batcher.state.name}")
t0 = time.monotonic()
try:
    proxy.detect_raw(torch.zeros(3, 8, 8, device="cuda"))
    late = "returned"
except Exception as e:
    late = type(e).__name__
print(f"@late={late}")
print(f"@late_s={time.monotonic() - t0:.2f}")
print(f"@shutdown_again={batcher.shutdown()}")
done()
"""
    )
    assert out["shutdown_busy"] == "False", out
    assert out["blocked"] == "[False, False]", out
    # The batch in flight finishes normally; the pending submit is woken.
    assert out["caught"] == (
        "[('in_flight', 'returned'), ('pending', 'BatchingDetectorShutDown')]"
    ), out
    assert out["state"] == "SHUT_DOWN", out
    assert out["late"] == "BatchingDetectorShutDown", out
    assert float(out["late_s"]) < 0.5, out
    assert out["shutdown_again"] == "True", out


@pytest.mark.gpu
def test_g3_pool_reports_failed_and_stop_completes() -> None:
    _need_native()
    out = _run(
        """
async def main():
    PLAN["raise"] = True
    pool = make_pool(n=2)
    await pool.start()
    batcher = pool._batcher
    pool.put_frame("stream_0", frame(10), 0.0)
    pool.put_frame("stream_1", frame(10), 0.0)
    await until(lambda: not any(w.is_alive() for w in pool._workers))
    f = pool.failure
    print(f"@state={pool.state.name}")
    print(f"@failure={type(f).__name__}")
    print(f"@cause_is_original={f.__cause__ is ORIGINAL}")
    print(f"@batcher={batcher.state.name}")
    t0 = time.monotonic()
    try:
        pool.stop(timeout=2.0)
        result = "returned"
    except Exception as e:
        result = (type(e).__name__, e.__cause__ is f)
    print(f"@stop={result}")
    print(f"@stop_s={time.monotonic() - t0:.1f}")
    print(f"@after={pool.state.name}")
    print(f"@batcher_released={pool._batcher is None}")
    print(f"@thread_errors={THREAD_ERRORS}")
asyncio.run(main())
done()
"""
    )
    assert out["state"] == "FAILED", out
    assert (out["failure"], out["cause_is_original"]) == (
        "BatchingDetectorFailed",
        "True",
    ), out
    assert out["batcher"] == "FAILED", out
    assert out["stop"] == "('WorkbenchPoolFailed', True)", out
    assert float(out["stop_s"]) < 3.0, out
    assert (out["after"], out["batcher_released"]) == ("STOPPED", "True"), out
    assert out["thread_errors"] == "[]", out


@pytest.mark.gpu
def test_g3_failure_while_stop_waits_ends_the_stop() -> None:
    _need_native()
    out = _run(
        """
async def main():
    PLAN["gate"] = threading.Event()
    pool = make_pool()
    await pool.start()
    pool.put_frame("stream_0", frame(10), 0.0)
    await until(IN_INFER.is_set)
    result = {}
    def stop():
        t0 = time.monotonic()
        try:
            pool.stop(timeout=10.0)
            result["stop"] = "returned"
        except Exception as e:
            result["stop"] = type(e).__name__
        result["s"] = time.monotonic() - t0
    stopper = threading.Thread(target=stop)
    stopper.start()
    await asyncio.sleep(0.5)
    print(f"@state_while_waiting={pool.state.name}")
    PLAN["raise"] = True
    t_release = time.monotonic()
    PLAN["gate"].set()
    stopper.join(10.0)
    print(f"@stopper_alive={stopper.is_alive()}")
    print(f"@stop={result.get('stop')}")
    print(f"@after_release_s={time.monotonic() - t_release:.1f}")
    print(f"@after={pool.state.name}")
    print(f"@failure={type(pool.failure).__name__}")
asyncio.run(main())
done()
"""
    )
    assert out["state_while_waiting"] == "STOPPING", out
    assert out["stopper_alive"] == "False", out
    assert out["stop"] == "WorkbenchPoolFailed", out
    assert float(out["after_release_s"]) < 3.0, out
    assert (out["after"], out["failure"]) == ("STOPPED", "BatchingDetectorFailed")


@pytest.mark.gpu
def test_batcher_normal_path_routes_each_slice_to_its_submitter() -> None:
    """Concurrent eval's use: many threads share one batcher; results unchanged.

    Uses only the pre-#538 surface, so it passes on the pre-fix source too: the
    fix leaves the normal path's values alone.
    """
    _need_native()
    out = _run(
        """
class EchoTRT(FakeTRT):
    def infer_raw_batch(self, frames):
        b = frames.shape[0]
        out = torch.zeros((b, 300, 6), device=frames.device)
        out[:, :, 0] = frames.reshape(b, -1)[:, :1]  # echo each input's id
        return {"output": out}
dt.TRTYoloDetector = EchoTRT
batcher = dt.BatchingTRTDetector("fake", batch_size=4)
got = {}
def worker(i):
    proxy = batcher.make_proxy()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        vals = []
        for j in range(25):
            x = torch.full((3, 8, 8), float(i * 1000 + j), device="cuda")
            vals.append(proxy.detect_raw(x)[0, 0, 0].item())
    got[i] = vals == [float(i * 1000 + j) for j in range(25)]
threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join(30.0)
print(f"@all_match={all(got.get(i) for i in range(6))}")
print(f"@shutdown={batcher.shutdown()}")
done()
"""
    )
    assert out["all_match"] == "True", out
    assert out["shutdown"] == "True", out


@pytest.mark.gpu
def test_batcher_reports_running_then_shut_down() -> None:
    _need_native()
    out = _run(
        """
batcher = dt.BatchingTRTDetector("fake", batch_size=1)
proxy = batcher.make_proxy()
proxy.detect_raw(torch.zeros(3, 8, 8, device="cuda"))
print(f"@state={batcher.state.name}")
print(f"@failure={batcher.failure}")
print(f"@shutdown={batcher.shutdown()}")
print(f"@after={batcher.state.name}")
print(f"@failure_after={batcher.failure}")
done()
"""
    )
    assert (out["state"], out["failure"]) == ("RUNNING", "None"), out
    assert out["shutdown"] == "True", out
    assert (out["after"], out["failure_after"]) == ("SHUT_DOWN", "None"), out


# ── Positive control: AsyncDispatcher ────────────────────────────────────


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
