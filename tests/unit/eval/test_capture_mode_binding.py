"""Binding contract for cudaThreadExchangeStreamCaptureMode.

ctypes defaults to int-sized arguments and no return type, so calling a CUDA
entry point without declaring its signature is undefined behaviour rather than a
type error, and a dropped return code makes a call that did not complete cleanly
indistinguishable from one that did. Both were true of the copy that used to
live in ``streaming.py``.

Note what the failure contract deliberately does *not* say. A CUDA runtime call
may report an error left by a prior asynchronous launch, so a non-success code
does not establish that the exchange failed to take effect — only that the
post-call mode is unverified. The assertions below are written to that weaker
claim.

These assert the declaration, the success and failure semantics of the shared
helper, and that the caller uses it rather than keeping a second raw binding.
They say nothing about Rule B or about who opens a blocking capture.
"""

# scope: eval
# function: contract
# lifecycle: active

import ctypes
import queue as queue_mod
import threading

import pytest

from saccade.perception.eval import cuda_capture
from saccade.perception.eval.cuda_capture import (
    CaptureModeExchangeError,
    enter_relaxed_capture_mode,
)
from saccade.perception.eval.streaming import _DecodeFailure, TorchvisionGpuStreamer

_RELAXED = 2  # cudaStreamCaptureModeRelaxed


class _Entry:
    """Records the argtypes/restype a binding layer declares on it."""

    def __init__(self, impl=None):
        self.restype = None
        self.argtypes = None
        self._impl = impl

    def __call__(self, *args):
        return self._impl(*args) if self._impl is not None else 0


@pytest.fixture
def reset_cudart():
    """The loaded handle is a module global; keep tests from leaking into it."""
    saved_lib, saved_tried = cuda_capture._cudart_lib, cuda_capture._cudart_tried
    yield
    cuda_capture._cudart_lib, cuda_capture._cudart_tried = saved_lib, saved_tried


def test_exchange_binding_declares_its_signature(monkeypatch, reset_cudart) -> None:
    """argtypes/restype are declared, so the call is not undefined behaviour."""

    class _Lib:
        def __init__(self):
            self.cudaStreamGetFlags = _Entry()
            self.cudaStreamIsCapturing = _Entry()
            self.cudaGetLastError = _Entry()
            self.cudaThreadExchangeStreamCaptureMode = _Entry()

    lib = _Lib()
    monkeypatch.setattr(cuda_capture.ctypes, "CDLL", lambda _name: lib)
    cuda_capture._cudart_lib, cuda_capture._cudart_tried = None, False

    assert cuda_capture._cudart() is lib
    entry = lib.cudaThreadExchangeStreamCaptureMode
    assert entry.restype is ctypes.c_int
    assert entry.argtypes == [ctypes.POINTER(ctypes.c_int)]


def _fake_cudart(rc: int, previous_mode: int = 0):
    """A cudart whose exchange returns ``rc`` and reports ``previous_mode``."""
    seen: dict = {}

    def exchange(arg):
        seen["requested"] = arg._obj.value
        if rc == 0:
            arg._obj.value = previous_mode  # the entry point exchanges
        return rc

    class _Lib:
        cudaThreadExchangeStreamCaptureMode = staticmethod(exchange)
        cudaGetLastError = staticmethod(lambda: 0)

    return _Lib(), seen


def test_successful_exchange_requests_relaxed_and_returns_the_prior_mode(
    monkeypatch,
) -> None:
    lib, seen = _fake_cudart(
        rc=0, previous_mode=cuda_capture._CAPTURE_MODE_CODE["global"]
    )
    monkeypatch.setattr(cuda_capture, "_cudart", lambda: lib)

    assert enter_relaxed_capture_mode() == "global"
    assert seen["requested"] == _RELAXED


def test_failed_exchange_raises_instead_of_reporting_success(monkeypatch) -> None:
    """A non-success code must not read as an established Relaxed entry.

    It also must not read as the opposite. The message is required to say the
    post-call mode is *unverified*, because the code may have been left by a
    prior asynchronous launch rather than by this call.
    """
    lib, _ = _fake_cudart(rc=999)
    monkeypatch.setattr(cuda_capture, "_cudart", lambda: lib)

    with pytest.raises(CaptureModeExchangeError) as excinfo:
        enter_relaxed_capture_mode()

    message = str(excinfo.value)
    assert "cudaThreadExchangeStreamCaptureMode" in message  # API name
    assert "999" in message  # CUDA return code
    assert "relaxed" in message  # requested mode
    # Calibration: unverified, not "the thread is not exempt".
    assert "unverified" in message
    assert "prior asynchronous launch" in message


def test_missing_cudart_is_not_a_failed_exchange(monkeypatch) -> None:
    """No library is a different event from an exchange that failed."""
    monkeypatch.setattr(cuda_capture, "_cudart", lambda: None)
    assert enter_relaxed_capture_mode() is None


def test_streamer_keeps_no_second_raw_binding() -> None:
    """The caller uses the shared helper; the private copy is gone."""
    assert not hasattr(TorchvisionGpuStreamer, "_enter_relaxed_capture_mode")


def _stub_streamer(files: list[str]):
    streamer = TorchvisionGpuStreamer.__new__(TorchvisionGpuStreamer)
    streamer.img_files = list(files)
    streamer._prefetch = 2
    streamer._idx = 0
    streamer._stop = threading.Event()
    streamer._worker = None
    streamer._queue = None  # type: ignore[assignment]
    streamer._rgb = None
    streamer.relaxed_capture_mode_from = None

    class _Frame:
        def permute(self, *_dims: int) -> "_Frame":
            return self

        def record_stream(self, _stream) -> None:  # type: ignore[no-untyped-def]
            return None

    class _Cuda:
        current_stream = staticmethod(lambda: object())

    class _Torch:
        cuda = _Cuda

    streamer._torch = _Torch  # type: ignore[assignment]
    streamer._read_file = lambda path: path  # type: ignore[assignment]
    streamer._decode = lambda data, device, mode: _Frame()  # type: ignore[assignment]
    return streamer


def test_worker_records_the_prior_mode_from_the_shared_helper(monkeypatch) -> None:
    monkeypatch.setattr(cuda_capture, "enter_relaxed_capture_mode", lambda: "global")

    streamer = _stub_streamer(["a.jpg"])
    out: queue_mod.Queue = queue_mod.Queue()
    streamer._decode_worker(out)

    assert streamer.relaxed_capture_mode_from == "global"
    assert out.qsize() == 1


def test_a_failed_exchange_is_not_recorded_as_a_successful_entry(monkeypatch) -> None:
    """The failure surfaces and no prior mode is recorded.

    Not because the thread is known not to be exempt — that is unknowable from
    the return code — but because after a call that did not complete cleanly the
    reported prior mode cannot be trusted either.
    """

    def boom() -> str:
        raise CaptureModeExchangeError("cudaThreadExchangeStreamCaptureMode failed")

    monkeypatch.setattr(cuda_capture, "enter_relaxed_capture_mode", boom)

    streamer = _stub_streamer(["a.jpg"])
    out: queue_mod.Queue = queue_mod.Queue()
    with pytest.raises(CaptureModeExchangeError):
        streamer._decode_worker(out)

    assert streamer.relaxed_capture_mode_from is None
    # It reaches the consumer as a failure rather than as end-of-sequence.
    failure = out.get_nowait()
    assert isinstance(failure, _DecodeFailure)
    assert isinstance(failure.error, CaptureModeExchangeError)
