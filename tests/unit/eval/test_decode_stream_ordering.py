"""Decode producer stream contract (issue #340 Phase 2B, TorchvisionGpuStreamer).

The claim under test is not "a run finished without crashing".  It is:

  * the decode worker issues no work on the legacy default stream, so the second
    precondition of Rule B is gone regardless of who opens a blocking capture;
  * each frame crosses the queue with an event recorded on the decode stream, so
    the ordering the consumer relies on belongs to this module rather than to a
    private detail of torchvision's decoder;
  * a decode failure reaches the consumer as a failure, not as end-of-sequence.

``test_blocking_capture_survives_concurrent_decode`` is a reproduction test: it
carries its own positive control, so a pass means the harness can still detect
the hazard it is asserting the absence of.
"""

# scope: eval
# function: behavior
# lifecycle: active

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from saccade.perception.eval.streaming import TorchvisionGpuStreamer

REPO_ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = REPO_ROOT / "datasets" / "MOT17" / "train" / "MOT17-02-SDP" / "img1"


def _streamer(tmp_path: Path, n: int = 4) -> TorchvisionGpuStreamer:
    if not IMG_DIR.is_dir():
        pytest.skip(f"MOT17 sequence not present at {IMG_DIR}")
    streamer = TorchvisionGpuStreamer(IMG_DIR)
    if len(streamer.img_files) < n:
        pytest.skip("sequence too short")
    streamer.img_files = streamer.img_files[:n]
    return streamer


@pytest.mark.gpu
def test_worker_decodes_off_the_legacy_stream(tmp_path: Path) -> None:
    """The worker's current stream is the dedicated one, never the default.

    This is the precondition removal: torchvision joins the decode to whatever
    the calling thread's current torch stream is, so the worker's current stream
    *is* its legacy-stream footprint.
    """
    import torch

    streamer = _streamer(tmp_path)
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
    default_stream = torch.cuda.default_stream().cuda_stream
    assert set(seen) == {decode_stream}
    assert decode_stream != default_stream


@pytest.mark.gpu
def test_frames_are_handed_over_with_a_decode_stream_event(tmp_path: Path) -> None:
    """Every queued frame carries the event the consumer orders itself against."""
    import torch

    streamer = _streamer(tmp_path)
    iterator = iter(streamer)
    frame, ready = streamer._queue.get()
    assert isinstance(ready, torch.cuda.Event)
    assert frame.is_cuda and frame.dtype == torch.uint8
    # The frame is [H, W, C] to match DALIStreamerStream.
    assert frame.ndim == 3 and frame.shape[2] == 3
    streamer._queue.put((frame, ready))
    consumed = next(iterator)
    assert consumed.data_ptr() == frame.data_ptr()


@pytest.mark.gpu
def test_relaxed_capture_mode_is_actually_entered(tmp_path: Path) -> None:
    """Rule A exemption is reported, not assumed.

    The previous implementation dropped the return code, so a failed exchange was
    indistinguishable from a successful one.
    """
    streamer = _streamer(tmp_path, n=2)
    list(streamer)
    assert streamer.relaxed_capture_mode_from == "global"


def _stub_streamer(files: list[str]) -> TorchvisionGpuStreamer:
    """A streamer with every torch/torchvision dependency stubbed out.

    Lets the queue-handoff and failure-propagation contracts be asserted on hosts
    without a CUDA device, where the interesting part is pure control flow.
    """
    import contextlib

    streamer = TorchvisionGpuStreamer.__new__(TorchvisionGpuStreamer)
    streamer.img_files = list(files)
    streamer._prefetch = 2
    streamer._idx = 0
    streamer._stop = threading.Event()
    streamer._worker = None
    streamer._queue = None  # type: ignore[assignment]
    streamer._decode_stream = object()
    streamer.relaxed_capture_mode_from = None
    streamer._rgb = None

    class _Event:
        def record(self, _stream):  # type: ignore[no-untyped-def]
            return None

    class _Cuda:
        Stream = staticmethod(lambda: object())
        Event = staticmethod(_Event)
        stream = staticmethod(lambda _s: contextlib.nullcontext())
        current_stream = staticmethod(lambda: object())

    class _Torch:
        cuda = _Cuda

    streamer._torch = _Torch  # type: ignore[assignment]
    streamer._read_file = lambda path: path  # type: ignore[assignment]
    streamer._decode = lambda data, device, mode: _Frame(data)  # type: ignore[assignment]
    return streamer


class _Frame:
    """Stand-in for the decoded CUDA tensor: permute is a view, not a copy."""

    def __init__(self, name: str):
        self.name = name

    def permute(self, *_dims: int) -> "_Frame":
        return self

    def record_stream(self, _stream) -> None:  # type: ignore[no-untyped-def]
        return None


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_decode_failure_is_not_end_of_sequence() -> None:
    """A failing decode must fail the run, not silently truncate the sequence.

    The worker used to queue a bare ``None`` sentinel, which ``__next__`` turned
    into ``StopIteration`` — a decode error and a finished sequence were the same
    event, so a broken run produced a short, plausible-looking result file.
    """
    streamer = _stub_streamer(["a.jpg", "b.jpg", "c.jpg"])

    def boom(_path):  # type: ignore[no-untyped-def]
        raise OSError("truncated JPEG")

    streamer._read_file = boom  # type: ignore[assignment]

    iterator = iter(streamer)
    with pytest.raises(RuntimeError, match="GPU JPEG decode failed") as excinfo:
        next(iterator)
    assert isinstance(excinfo.value.__cause__, OSError)
    # The worker re-raises after handing the error over; join it here so its
    # traceback is reported against this test rather than the next one.
    streamer._worker.join(timeout=5)


def test_a_stale_worker_cannot_write_into_a_fresh_queue() -> None:
    """The worker owns its queue, so a join timeout cannot interleave sequences.

    ``_stop_worker`` joins with a timeout; before the queue was passed in, a
    worker that outlived that timeout would resume pushing into whichever queue
    ``self._queue`` pointed at by then — the *next* sequence's.
    """
    import queue as queue_mod

    streamer = _stub_streamer(["a.jpg"])
    stale_queue: queue_mod.Queue = queue_mod.Queue()
    fresh_queue: queue_mod.Queue = queue_mod.Queue()
    streamer._queue = fresh_queue
    streamer._decode_worker(stale_queue)  # runs inline, as the stale worker would

    assert fresh_queue.empty()
    assert stale_queue.qsize() == 1


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
    decode worker no longer trips it — while the control case, a thread touching
    the legacy stream, still does.
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
