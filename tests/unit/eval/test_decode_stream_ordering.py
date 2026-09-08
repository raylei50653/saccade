"""Decode producer stream contract (issue #340 Phase 2B, TorchvisionGpuStreamer).

The claim under test is not "a run finished without crashing".  It is:

  * the decode worker issues no work on the legacy default stream, so the second
    precondition of Rule B is gone regardless of who opens a blocking capture;
  * each frame crosses the queue with an event recorded on the decode stream, and
    the consumer orders itself against that event -- an invariant of this module
    rather than a private detail of torchvision's decoder;
  * the decode stream belongs to the worker generation, so a worker that outlived
    its join timeout cannot share a CUDA ordering domain with its successor.

The queue protocol and the allocator-lifetime half (``record_stream``) are
asserted in ``test_decode_producer_contract.py``; ordering and lifetime are two
invariants and neither test stands in for the other.

``test_blocking_capture_survives_concurrent_decode`` is a reproduction test: it
carries its own positive control, so a pass means the harness can still detect
the hazard it is asserting the absence of.
"""

# scope: eval
# function: behavior
# lifecycle: active

import contextlib
import queue as queue_mod
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from saccade.perception.eval import cuda_capture
from saccade.perception.eval.streaming import TorchvisionGpuStreamer

REPO_ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = REPO_ROOT / "datasets" / "MOT17" / "train" / "MOT17-02-SDP" / "img1"


def _streamer(n: int = 4) -> TorchvisionGpuStreamer:
    if not IMG_DIR.is_dir():
        pytest.skip(f"MOT17 sequence not present at {IMG_DIR}")
    streamer = TorchvisionGpuStreamer(IMG_DIR)
    if len(streamer.img_files) < n:
        pytest.skip("sequence too short")
    streamer.img_files = streamer.img_files[:n]
    return streamer


@pytest.mark.gpu
def test_worker_decodes_off_the_legacy_stream() -> None:
    """The worker's current stream is its own, never the default.

    This is the precondition removal: torchvision joins the decode to whatever
    the calling thread's current torch stream is, so the worker's current stream
    *is* its legacy-stream footprint.
    """
    import torch

    streamer = _streamer()
    seen: list[int] = []
    inner = streamer._decode

    def observing_decode(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(torch.cuda.current_stream().cuda_stream)
        return inner(*args, **kwargs)

    streamer._decode = observing_decode  # type: ignore[assignment]
    frames = list(streamer)

    assert len(frames) == len(streamer.img_files)
    assert seen, "decode was never called"
    decode_stream = streamer._decode_stream.cuda_stream
    assert set(seen) == {decode_stream}
    assert decode_stream != torch.cuda.default_stream().cuda_stream


@pytest.mark.gpu
def test_frames_are_handed_over_with_a_decode_stream_event() -> None:
    """Every queued frame carries the event the consumer orders itself against."""
    import torch

    streamer = _streamer()
    iterator = iter(streamer)
    frame, ready = streamer._queue.get()

    assert isinstance(ready, torch.cuda.Event)
    assert frame.is_cuda and frame.dtype == torch.uint8
    # The frame is [H, W, C] to match DALIStreamerStream.
    assert frame.ndim == 3 and frame.shape[2] == 3

    streamer._queue.put((frame, ready))
    consumed = next(iterator)
    assert consumed.data_ptr() == frame.data_ptr()


class _Event:
    """Stand-in for ``torch.cuda.Event``; remembers where it was recorded."""

    def __init__(self) -> None:
        self.recorded_on: object | None = None

    def record(self, stream: object) -> None:
        self.recorded_on = stream


class _Stream:
    """Identity-comparable stand-in for a torch CUDA stream."""

    _n = 0

    def __init__(self, name: str | None = None):
        type(self)._n += 1
        self.name = name or f"stream-{type(self)._n}"
        self.waited_on: list[object] = []

    def wait_event(self, event: object) -> None:
        self.waited_on.append(event)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"<stream {self.name}>"


class _Frame:
    """Stand-in for a decoded CUDA tensor; ``permute`` is a view, not a copy."""

    def __init__(self, name: str):
        self.name = name
        self.recorded_on: list[object] = []

    def permute(self, *_dims: int) -> "_Frame":
        return self

    def record_stream(self, stream: object) -> None:
        self.recorded_on.append(stream)


def _stub_streamer(files: list[str], consumer: _Stream | None = None):
    """A streamer with every torch/torchvision dependency stubbed out.

    Lets the stream-ownership and ordering contracts be asserted on hosts with no
    CUDA device, where the interesting part is which stream object goes where.
    """
    consumer_stream = consumer or _Stream("consumer")

    streamer = TorchvisionGpuStreamer.__new__(TorchvisionGpuStreamer)
    streamer.img_files = list(files)
    streamer._prefetch = 2
    streamer._idx = 0
    streamer._stop = threading.Event()
    streamer._worker = None
    streamer._queue = None  # type: ignore[assignment]
    streamer._decode_stream = None
    streamer._rgb = None
    streamer.relaxed_capture_mode_from = None

    class _Cuda:
        Stream = staticmethod(_Stream)
        Event = staticmethod(_Event)
        stream = staticmethod(lambda _s: contextlib.nullcontext())
        current_stream = staticmethod(lambda: consumer_stream)

    class _Torch:
        cuda = _Cuda

    streamer._torch = _Torch  # type: ignore[assignment]
    streamer._read_file = lambda path: path  # type: ignore[assignment]
    streamer._decode = lambda data, device, mode: _Frame(data)  # type: ignore[assignment]
    return streamer, consumer_stream


@pytest.fixture(autouse=True)
def stub_capture_mode_exchange(monkeypatch):
    """Keep the device-free tests on the stream contract, off the Rule A exemption.

    ``_decode_worker`` enters Relaxed capture mode through a real cudart call
    that raises on a host with the library but no usable driver. That contract is
    tested in ``test_capture_mode_binding.py``; coupling it in here would make a
    stream-ownership failure unreadable as one.
    """
    monkeypatch.setattr(cuda_capture, "enter_relaxed_capture_mode", lambda: "global")


def test_consumer_waits_on_the_frames_decode_event() -> None:
    """Execution ordering is this module's invariant, not the decoder's.

    ``record_stream`` (asserted in ``test_decode_producer_contract.py``) tells the
    allocator the block is still in use; it orders nothing. The consuming stream
    has to wait on the event recorded by the producer, and on the same stream the
    producer decoded on.
    """
    streamer, consumer_stream = _stub_streamer(["a.jpg", "b.jpg"])

    frames = list(streamer)

    assert len(frames) == 2
    assert len(consumer_stream.waited_on) == 2
    decode_stream = streamer._decode_stream
    for event in consumer_stream.waited_on:
        assert isinstance(event, _Event)
        assert event.recorded_on is decode_stream


def test_each_worker_generation_gets_its_own_decode_stream() -> None:
    """Stream ownership follows the worker, not the streamer object.

    ``_stop_worker`` joins with a 3 s timeout, so a stale worker can still be
    decoding when the next sequence starts. It keeps the stream it was handed;
    were the stream owned by the streamer, two generations would share one CUDA
    ordering domain and record their per-frame events on it.
    """
    streamer, _ = _stub_streamer(["a.jpg"])

    list(streamer)
    first = streamer._decode_stream
    list(streamer)
    second = streamer._decode_stream

    assert first is not None and second is not None
    assert first is not second


def test_a_stale_worker_keeps_its_own_decode_stream() -> None:
    """A worker that outlived its join cannot record onto the successor's stream."""
    streamer, _ = _stub_streamer(["a.jpg"])

    stale_queue: queue_mod.Queue = queue_mod.Queue()
    stale_stream = _Stream("stale-decode")
    fresh_queue: queue_mod.Queue = queue_mod.Queue()
    fresh_stream = _Stream("fresh-decode")
    streamer._queue = fresh_queue
    streamer._decode_stream = fresh_stream

    # Run the stale worker inline against the queue and stream it was handed,
    # exactly as a thread that outlived its join would.
    streamer._decode_worker(stale_queue, stale_stream)

    assert fresh_queue.empty(), "a stale worker reached the next sequence's queue"
    assert streamer._decode_stream is fresh_stream
    _frame, ready = stale_queue.get_nowait()
    assert ready.recorded_on is stale_stream


_CAPTURE_PROBE = textwrap.dedent(
    """
    # Blocking-stream capture with a concurrent producer, in its own process:
    # a failed capture poisons the CUDA context, so each case must be isolated.
    import ctypes, json, sys, threading
    from pathlib import Path
    sys.path.insert(0, {src!r})
    import torch

    which = sys.argv[1]
    rt = None
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            rt = ctypes.CDLL(name)
            break
        except OSError:
            continue

    torch.cuda.init()
    dev = torch.device("cuda")
    main_buf = torch.ones(256, device=dev)
    work_buf = torch.ones(256, device=dev)

    raw = ctypes.c_void_p()
    rt.cudaStreamCreateWithFlags(ctypes.byref(raw), ctypes.c_uint(0))  # 0 = BLOCKING
    blocking = torch.cuda.ExternalStream(raw.value)
    flags = ctypes.c_uint(9)
    rt.cudaStreamGetFlags(ctypes.c_void_p(blocking.cuda_stream), ctypes.byref(flags))
    torch.cuda.synchronize()

    capture_open = threading.Event()
    result = {{"capture_error": None, "producer_error": None,
              "capture_stream_flags": int(flags.value)}}

    if which == "control":
        # Positive control: the hazard this test asserts the absence of.
        def body():
            with torch.cuda.stream(torch.cuda.default_stream()):
                work_buf.add_(1.0)
    else:
        from saccade.perception.eval.streaming import TorchvisionGpuStreamer
        streamer = TorchvisionGpuStreamer(Path({img_dir!r}))
        streamer.img_files = streamer.img_files[:8]
        def body():
            for _ in streamer:
                pass

    def producer():
        # Rule B is raised in the *offending* thread; the capturing thread only
        # sees the downstream "previous error during capture".
        try:
            capture_open.wait(10)
            body()
        except BaseException as exc:
            result["producer_error"] = (
                f"{{type(exc).__name__}}: {{str(exc).splitlines()[0][:160]}}"
            )

    run = threading.Thread(target=producer, daemon=True)

    run.start()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=blocking, capture_error_mode="thread_local"):
            capture_open.set()
            for _ in range(4000):
                main_buf.mul_(1.0)
    except Exception as exc:
        result["capture_error"] = f"{{type(exc).__name__}}: {{str(exc).splitlines()[0][:160]}}"
    capture_open.set()
    run.join(30)
    print(json.dumps(result))
    """
)


def _run_capture_probe(which: str) -> dict:
    import json

    script = _CAPTURE_PROBE.format(src=str(REPO_ROOT / "src"), img_dir=str(IMG_DIR))
    proc = subprocess.run(
        [sys.executable, "-c", script, which],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert lines, (
        f"probe produced no result: rc={proc.returncode}\n{proc.stderr[-2000:]}"
    )
    return json.loads(lines[-1])


@pytest.mark.gpu
def test_blocking_capture_survives_concurrent_decode() -> None:
    """Rule B is unreachable from the decode thread, shown against a live control.

    A blocking capturing stream is the other precondition for
    ``cudaErrorStreamCaptureImplicit``.  We cannot prove no component in the
    process ever opens one, so instead we open one deliberately and check the
    decode worker no longer trips it -- while the control case, a thread touching
    the legacy stream, still does.

    Passing says the decode-side precondition is gone. It says nothing about the
    production incident's own capturing stream, which is still unidentified.
    """
    if not IMG_DIR.is_dir():
        pytest.skip(f"MOT17 sequence not present at {IMG_DIR}")

    control = _run_capture_probe("control")
    assert control["capture_stream_flags"] == 0, "control stream was not blocking"
    assert (
        control["producer_error"] is not None and control["capture_error"] is not None
    ), (
        "the positive control did not reproduce the hazard, so a clean result "
        f"from the decode case would prove nothing: {control}"
    )
    # cudaErrorStreamCaptureImplicit, spelled out by the driver.
    assert (
        "legacy stream depend on a capturing blocking stream"
        in (control["producer_error"])
    ), control["producer_error"]

    decode = _run_capture_probe("decode")
    assert decode["capture_stream_flags"] == 0
    assert decode["producer_error"] is None, decode["producer_error"]
    assert decode["capture_error"] is None, decode["capture_error"]
