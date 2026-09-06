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
stream, and every torch stream is ``cudaStreamNonBlocking``.  **That does not
exclude Rule B.**  A stream can participate in an existing capture without ever
issuing ``cudaStreamBeginCapture`` itself — it can be joined into an in-progress
capture through an event dependency — so enumerating the streams that *begin*
captures does not enumerate the streams that *participate* in them.  A control
run under the #340 attribution harness showed exactly that: a blocking side
stream joined a non-blocking origin's capture via an event wait, another
thread's ``cudaStreamIsCapturing`` on the legacy stream then returned
``cudaErrorStreamCaptureImplicit``, and the origin's ``capture_end`` still
succeeded.

So the blocking capturing stream behind the one production failure is
unidentified, and an enumeration of our own capture origins cannot identify it.
A bounded production-path trace on 2026-09-06 observed four classified captures
with no in-capture joins and no blocking participant, but its structure check
did not pass, which makes it a snapshot rather than an exclusion — see
``docs/research/pipeline/capture_failure_provenance_20260906.md`` and
``scripts/tools/capture_attribution/README.md``.  #340 stays open on this.

:func:`describe_capture_state` exists to turn the next occurrence into a
diagnosis instead of a mystery.  Enable it with ``SACCADE_CAPTURE_DEBUG=1``; the
failure-time dump is unconditional because the failure is rare and already
fatal.
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

# Which of our captures is open, if any.  Set by :func:`graph_capture` only, so
# it covers the direct ``torch.cuda.graph`` sites and *not* the
# ``make_graphed_callables`` ones (detector whole-graph, tracker), which take no
# wrapper.  ``None`` therefore narrows a failure to "no direct site was open";
# it does not establish that the capture belongs to another component.
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
            lib.cudaThreadExchangeStreamCaptureMode.restype = ctypes.c_int
            lib.cudaThreadExchangeStreamCaptureMode.argtypes = [
                ctypes.POINTER(ctypes.c_int)
            ]
            _cudart_lib = lib
            break
    return _cudart_lib


# ``cudaStreamCaptureMode``.  These are CUDA API constants, mapped here once so
# no caller re-spells the integers; they are not torch's mode strings, which
# happen to share the names.
_CAPTURE_MODE_CODE = {"global": 0, "thread_local": 1, "relaxed": 2}
_CAPTURE_MODE_NAME = {code: name for name, code in _CAPTURE_MODE_CODE.items()}


class CaptureModeExchangeError(RuntimeError):
    """``cudaThreadExchangeStreamCaptureMode`` returned a non-success code."""


def enter_relaxed_capture_mode() -> str | None:
    """Exempt the calling thread from Rule A, and say what it was exempt from.

    Producer threads allocate through torch, which under a ``"global"`` capture
    elsewhere in the process invalidates that capture (Rule A).  ``thread_local``
    on our own captures does not cover it: ``make_graphed_callables`` captures at
    torch's default ``"global"`` and accepts no ``capture_error_mode``, so the
    exemption has to come from the producer side.  This is **only** about Rule A;
    it does nothing for Rule B, which no capture mode addresses.

    Returns the mode the thread was in *before* the exchange, or ``None`` when
    cudart could not be loaded at all — a host without the library, which is not
    the same event as a call that failed.  A non-success return code raises
    :class:`CaptureModeExchangeError`; the previous implementation dropped the
    code, so a call that did not complete cleanly was indistinguishable from one
    that did.

    What a non-success code does *not* establish is that the exchange did not
    happen.  A CUDA runtime call may report an error left by a prior
    asynchronous launch, so the code need not describe this call at all.  What
    follows is only that the call did not complete cleanly and the thread's
    post-call capture mode is therefore **unverified**: Relaxed entry must not be
    assumed established, and the reported prior mode must not be trusted or
    recorded.
    """
    rt = _cudart()
    if rt is None:
        return None
    requested = _CAPTURE_MODE_CODE["relaxed"]
    mode = ctypes.c_int(requested)
    rc = rt.cudaThreadExchangeStreamCaptureMode(ctypes.byref(mode))
    if rc != 0:
        # Clear the sticky error so it cannot be mistaken for a later one; the
        # code itself is carried in the message rather than dropped.  It may have
        # been left by a prior asynchronous launch rather than by this call,
        # which is exactly why the message below claims nothing about whether the
        # exchange took effect.
        rt.cudaGetLastError()
        raise CaptureModeExchangeError(
            f"cudaThreadExchangeStreamCaptureMode did not complete cleanly: "
            f"rc={rc}, requested mode 'relaxed' ({requested}); the return code "
            f"may belong to a prior asynchronous launch, so this thread's "
            f"post-call capture mode is unverified — Relaxed entry must not be "
            f"assumed established"
        )
    previous = int(mode.value)
    return _CAPTURE_MODE_NAME.get(previous, f"unknown({previous})")


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

    ``open_capture=None`` alongside a capture-related failure narrows the
    search: no direct ``graph_capture`` site was open.  It is not proof that the
    capture belongs to someone else — the ``make_graphed_callables`` sites set no
    label, and a stream can join a capture it never began.
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
