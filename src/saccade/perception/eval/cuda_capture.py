"""CUDA graph capture policy for this repo, and the Rule B diagnostic.

Capturing a CUDA graph while another thread is issuing CUDA work is not one
hazard but **two independent ones**, and only the first is governed by
``capture_error_mode``.  Both were established with a deterministic probe (see
issue #340 Phase 1; probes retained at
``~/.local/state/saccade/perf/capture-race-audit-20260906/``), not inferred
from documentation:

**Rule A — the unsafe-API rule.**  Another thread calling the *allocator* while
a capture is open.  The stream that thread uses is irrelevant; a plain kernel
launch never trips it.  Governed by ``capture_error_mode``: ``"global"``
errors, ``"thread_local"`` exempts other threads, and a thread that has called
``cudaThreadExchangeStreamCaptureMode(Relaxed)`` is exempt even under
``"global"``.  It surfaces twice for one event — ``cudaErrorStreamCaptureUnsupported``
in the offending thread and ``cudaErrorStreamCaptureInvalidated`` at
``capture_end``.

**Rule B — the legacy-stream implicit-dependency rule.**  Another thread doing
*any* work on the legacy stream while a **blocking** stream is capturing.
``"global"``, ``"thread_local"`` and ``"relaxed"`` all fail identically, and
exempting the other thread does not help either.  It surfaces as
``cudaErrorStreamCaptureImplicit``.  **Changing the capture mode cannot fix
it.**

Every capture here is issued from the main eval thread onto a torch-owned
stream, and every torch stream is ``cudaStreamNonBlocking`` — so Rule B should
be unreachable from our own captures, yet the failure has been observed once in
production.  The blocking capturing stream responsible is unidentified, which
is why :func:`describe_capture_state` exists: it is meant to turn the next
occurrence into a diagnosis instead of a mystery.  Enable it with
``SACCADE_CAPTURE_DEBUG=1``; the failure-time dump is unconditional because the
failure is rare and already fatal.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
from typing import Any, Iterator

import torch

# Rule A only.  Our captures never depend on decoder-thread work, so another
# thread's allocator activity is not something they need protection from.
# Deliberately not "relaxed": that would also stop the *capturing* thread's own
# unsafe calls from being reported, which is a real bug detector we want to keep.
CAPTURE_ERROR_MODE = "thread_local"

_CUDART_NAMES = ("libcudart.so.13", "libcudart.so.12", "libcudart.so")
_CUDA_STREAM_LEGACY = ctypes.c_void_p(1)

_cudart_lib: Any = None
_cudart_tried = False

# Which of our captures is open, if any.  A decode-thread failure raised while
# this is None proves the capturing stream belongs to someone else, which is
# the open question Rule B leaves.
_open_capture: str | None = None
_open_capture_lock = threading.Lock()


def _cudart() -> Any:
    """Load cudart once, with argtypes pinned.

    ctypes defaults to int-sized arguments and no return type.  Calling a CUDA
    entry point without declaring its signature is undefined behaviour, not a
    type error: ``cudaStreamGetCaptureInfo`` takes six out-parameters in this
    cudart, and invoking it with three segfaults the process.  Everything below
    is therefore declared explicitly, and only two-argument, stable entry
    points are used.
    """
    global _cudart_lib, _cudart_tried
    if not _cudart_tried:
        _cudart_tried = True
        for name in _CUDART_NAMES:
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            lib.cudaStreamGetFlags.restype = ctypes.c_int
            lib.cudaStreamGetFlags.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
            ]
            lib.cudaStreamIsCapturing.restype = ctypes.c_int
            lib.cudaStreamIsCapturing.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.cudaGetLastError.restype = ctypes.c_int
            lib.cudaGetLastError.argtypes = []
            _cudart_lib = lib
            break
    return _cudart_lib


def stream_flags(stream_ptr: int) -> int | None:
    """``cudaStreamGetFlags``: 0 = blocking (the Rule B precondition), 1 = non-blocking."""
    rt = _cudart()
    if rt is None:
        return None
    flags = ctypes.c_uint(0)
    rc = rt.cudaStreamGetFlags(ctypes.c_void_p(stream_ptr), ctypes.byref(flags))
    if rc != 0:
        rt.cudaGetLastError()
        return None
    return int(flags.value)


def capture_status(stream_ptr: int) -> int | None:
    """``cudaStreamIsCapturing`` status: 0 none, 1 active.

    Deliberately not ``cudaStreamGetCaptureInfo``, whose six-out-parameter
    signature is easy to get wrong from ctypes and fatal when wrong.  A query
    that reports a failure clears the sticky error so it cannot be mistaken for
    a real one later.
    """
    rt = _cudart()
    if rt is None:
        return None
    status = ctypes.c_int(-1)
    rc = rt.cudaStreamIsCapturing(ctypes.c_void_p(stream_ptr), ctypes.byref(status))
    if rc != 0:
        rt.cudaGetLastError()
        return None
    return int(status.value)


_FLAG_NAMES = {0: "BLOCKING", 1: "non-blocking"}
_STATUS_NAMES = {0: "none", 1: "active"}


def _describe_stream(label: str, stream_ptr: int) -> str:
    flags = stream_flags(stream_ptr)
    kind = _FLAG_NAMES.get(flags, f"flags={flags}") if flags is not None else "flags=?"
    status = capture_status(stream_ptr)
    st = _STATUS_NAMES.get(status, str(status)) if status is not None else "?"
    return f"{label}=0x{stream_ptr:x}({kind},capture={st})"


def describe_capture_state(where: str) -> str:
    """One-line snapshot of every stream that matters to Rules A and B.

    ``open_capture=None`` together with a capture-related failure is the
    discriminator: it means the capture in progress is not one of ours.
    """
    parts = [f"[capture-state] at={where}", f"open_capture={_open_capture!r}"]
    try:
        cur = torch.cuda.current_stream().cuda_stream
        parts.append(_describe_stream("current", cur))
        default = torch.cuda.default_stream().cuda_stream
        if default != cur:
            parts.append(_describe_stream("default", default))
        parts.append(_describe_stream("legacy", int(_CUDA_STREAM_LEGACY.value or 1)))
        cap: "torch.cuda.Stream | None" = getattr(
            torch.cuda.graph, "default_capture_stream", None
        )
        if cap is not None:
            parts.append(_describe_stream("torch_capture", cap.cuda_stream))
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never mask the real error
        parts.append(f"probe_failed={type(exc).__name__}")
    return " ".join(parts)


def capture_debug_enabled() -> bool:
    return os.environ.get("SACCADE_CAPTURE_DEBUG", "") in ("1", "true", "yes")


@contextlib.contextmanager
def graph_capture(
    cuda_graph: "torch.cuda.CUDAGraph",
    *,
    label: str,
    pool: Any = None,
    stream: "torch.cuda.Stream | None" = None,
) -> Iterator[None]:
    """``torch.cuda.graph`` with this repo's capture-error-mode policy attached.

    ``label`` names the capture site so a failure elsewhere in the process can
    say which capture was open at the time.
    """
    global _open_capture
    ctx = torch.cuda.graph(
        cuda_graph, pool=pool, stream=stream, capture_error_mode=CAPTURE_ERROR_MODE
    )
    if capture_debug_enabled():
        # torch types capture_stream as optional; it is set by __init__ unless
        # the caller passed stream=None *and* the class default failed to init.
        cap_stream = ctx.capture_stream
        where = (
            _describe_stream("stream", cap_stream.cuda_stream)
            if cap_stream is not None
            else "stream=?"
        )
        print(f"[capture-site] {label} mode={CAPTURE_ERROR_MODE} {where}", flush=True)
    with _open_capture_lock:
        _open_capture = label
    try:
        with ctx:
            if capture_debug_enabled():
                print(f"  {describe_capture_state(f'{label}:open')}", flush=True)
            yield
    finally:
        with _open_capture_lock:
            _open_capture = None
