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

The production topology has since directly shown four blocking streams joining
``detector.whole`` through event dependencies.  Their owning component is still
unidentified: the retained trace did not bind those handles to observed stream
creation stacks.  It also did not observe failure-time overlap with decode's
legacy-stream status query or any new 900/901/906.  This establishes topology,
not causality, so #340 remains open.  See
``docs/research/pipeline/capture_failure_provenance_20260906.md`` and
``scripts/tools/capture_attribution/README.md``.

Rule B has two preconditions, and the second one is ours.  Phase 2B removed the
legacy-stream half rather than continuing to hunt the capturing stream: the JPEG
decode producer now runs on a stream of its own, so no producer thread issues
work on the legacy stream at all and Rule B is unreachable from the decode side
*whoever* opens the blocking capture.  See
:class:`saccade.perception.eval.streaming.TorchvisionGpuStreamer` for the stream
contract that replaced it.  That is a removed precondition, not an explanation
of the production incident, which stays open.

:func:`describe_capture_state` exists to turn the next occurrence into a
diagnosis instead of a mystery, and #340's first reproduction showed the shape it
has to have.  That failure came out of ``make_graphed_callables`` on the **main
eval thread**, while the only unconditional dump in the tree hung off the decode
worker's exception handler, so the campaign log carried no capture state and no
stream flags at all.  See issue #374 and
``docs/research/pipeline/closed/capture_race_incidence_closure_20260908.md``.

Every production capture entrance now goes through :func:`capture_diagnostics` --
:func:`graph_capture` for ``torch.cuda.graph`` and :func:`graphed_callables` for
``torch.cuda.make_graphed_callables``.  It emits one line whenever an exception
escapes a capture entrance, on whatever thread ran it, and re-raises that
exception untouched.  ``SACCADE_CAPTURE_DEBUG=1`` additionally logs each capture
site as it opens; the failure-time line is deliberately **not** gated on it,
because a campaign that does not set the variable would otherwise observe
nothing -- which is exactly how the 20260908 campaign lost its evidence.

What that line is: a **post-failure snapshot**, taken after the exception has
unwound out of torch.  It records that an exception escaped a capture entrance
and what the streams looked like once it had.  It does not classify the error,
does not establish that a capture was invalidated, and does not describe the
state at the instant of invalidation.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
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

# Which capture entrances are inside their capture routine, per thread.  Set by
# :func:`capture_diagnostics`, so it now covers both the direct
# ``torch.cuda.graph`` sites and the ``make_graphed_callables`` ones.
#
# Per-thread, and a stack rather than a single slot, because a single global is
# not safe to read at failure time: ``multi_stream`` runs several ``StreamWorker``
# threads that capture concurrently (its ``_graph_capture_lock`` is created and
# injected but never acquired anywhere in ``src/``), so one worker's exit would
# otherwise clear a label another worker had set, and the failing thread would
# report a site it never entered.  A thread pops only its own entry.  This is
# robustness under a topology change and a fix for that read hazard; it does not
# serialise anything and no nesting is reachable in the current call graph.
_open_captures: dict[int, list[tuple[str, str]]] = {}
_open_capture_lock = threading.Lock()

# Every interpolated value is capped at this many characters.  The dump is one
# physical line by contract (see :func:`describe_capture_state`), so an
# unbounded value would be both unparseable and unbounded in the log.
_FIELD_LIMIT = 200


def _one_line(value: object) -> str:
    """Collapse a value to one bounded line.

    Any run of whitespace -- newlines included -- becomes a single space, so no
    field can split the dump across physical lines.  Used for the trailing
    free-form ``diagnostic_failed`` field only; every embedded field goes through
    :func:`_token`.
    """
    text = " ".join(str(value).split())
    if len(text) > _FIELD_LIMIT:
        text = text[:_FIELD_LIMIT] + "<truncated>"
    return text


def _token(value: object) -> str:
    """As :func:`_one_line`, but with no internal spaces either.

    Embedded fields must survive ``line.split()``, and a value is not always
    ours to trust: a thread name is whatever the code that created the thread
    chose.  Spaces become underscores rather than being dropped, so the
    substitution is visible in the log instead of silently joining two words.
    """
    return _one_line(value).replace(" ", "_")


def _push_open_capture(label: str) -> None:
    thread = threading.current_thread()
    entry = (label, f"{thread.name}#{thread.ident}")
    with _open_capture_lock:
        _open_captures.setdefault(threading.get_ident(), []).append(entry)


def _pop_open_capture() -> None:
    """Release this thread's innermost label, and only this thread's."""
    ident = threading.get_ident()
    with _open_capture_lock:
        stack = _open_captures.get(ident)
        if not stack:
            return
        stack.pop()
        if not stack:
            del _open_captures[ident]


def _render_open_captures() -> str:
    """``none``, or ``label@thread#ident`` for every entrance currently open.

    Rendered without spaces so it stays one whitespace-delimited field.
    """
    with _open_capture_lock:
        snapshot = [entry for stack in _open_captures.values() for entry in stack]
    if not snapshot:
        return "none"
    return ",".join(f"{_token(label)}@{_token(thread)}" for label, thread in snapshot)


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


def describe_capture_state(
    where: str, *, event: str = "capture_probe", error: str | None = None
) -> str:
    """One physical line describing every stream that matters to Rules A and B.

    The grammar is fixed-order, whitespace-delimited ``key=value`` with no spaces
    inside a value, so the line is ``split()``-parseable and can never wrap onto a
    second physical line.  ``diagnostic_failed=``, added by
    :func:`emit_capture_failure_state`, is by contract the last field, because its
    value is the only free-form one.

    The identity fields -- ``event``, ``at``, ``thread``, ``error``,
    ``open_capture`` -- are built before the stream probe, so a probe that fails
    degrades to ``probe_failed=<Type>`` without taking the identity half with it.

    ``error`` carries the exception's **type name only**.  The #340 incidence
    harness counts capture-error hits by substring-matching every log line against
    its signature table, so echoing a CUDA message here would let this diagnostic
    inject hits into that count.  The message and traceback reach the log through
    the interpreter anyway.

    What a label means, and what it does not:

    * a label is present while that site is inside its capture *routine* on that
      thread.  For ``make_graphed_callables`` that window includes torch's warmup
      iterations, so it is wider than ``cudaStreamBeginCapture``..``capture_end``;
    * ``open_capture=none`` means no *wrapped* entrance on any thread was inside
      its capture routine.  It is still not proof that a capture belongs to
      someone else — a stream can join a capture it never began;
    * the labels listed do not identify *which* capture was invalidated;
    * the line is a post-failure snapshot taken after the exception unwound out of
      torch, so the stream state it reports may already differ from the state at
      the moment of invalidation.
    """
    thread = threading.current_thread()
    parts = [
        "[capture-state]",
        f"event={_token(event)}",
        f"at={_token(where)}",
        f"thread={_token(thread.name)}#{thread.ident}",
    ]
    if error is not None:
        parts.append(f"error={_token(error)}")
    parts.append(f"open_capture={_render_open_captures()}")
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


_UNPRINTABLE = (
    "[capture-state] event=capture_failure at=? diagnostic_failed=unprintable"
)


def _diagnostic_failure_line(where: str, exc: BaseException) -> str:
    """Secondary evidence: the diagnostic itself failed, and this is how."""
    try:
        detail = _one_line(f"{type(exc).__name__}: {exc}")
    except BaseException:  # noqa: BLE001 - ``str(exc)`` can raise too
        detail = "unprintable"
    try:
        site = _token(where)
    except BaseException:  # noqa: BLE001 - same reason
        site = "?"
    # ``diagnostic_failed`` is last by contract; nothing may follow it.
    return f"[capture-state] event=capture_failure at={site} diagnostic_failed={detail}"


def emit_capture_failure_state(where: str, *, error: str | None = None) -> None:
    """Print the failure-time dump.  Total: this must not raise, ever.

    A diagnostic that swallows the error it exists to explain leaves the next
    failure with *less* evidence than this one, so the contract is layered:

    1. a dump that cannot be built is reported as ``diagnostic_failed=`` rather
       than dropped — the diagnostic's own failure is secondary evidence;
    2. a ``print`` that fails is retried once on stderr;
    3. if both streams are gone the line is lost.  That is the accepted floor:
       the alternative is raising from here, which would mask the original CUDA
       error, and no record is strictly better than a replaced one.

    One accepted side effect.  :func:`stream_flags` and :func:`capture_status`
    call ``cudaGetLastError`` when a query returns non-zero, and that now happens
    on the main eval thread as well as the decode worker.  The error the log
    needs is the already-raised Python exception, which the caller re-raises, so
    it cannot be lost here; what could be cleared is a sticky flag some later
    torch call would have reported.  That is tolerable only because this runs
    while an exception is already propagating out of a capture entrance.  Never
    add this call to a path that continues.
    """
    try:
        line = describe_capture_state(where, event="capture_failure", error=error)
    except BaseException as exc:  # noqa: BLE001 - a diagnostic must never mask the real error
        try:
            line = _diagnostic_failure_line(where, exc)
        except BaseException:  # noqa: BLE001 - last resort
            line = _UNPRINTABLE
    try:
        print(line, flush=True)
        return
    except BaseException:  # noqa: BLE001 - stdout may be closed or replaced
        pass
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except BaseException:  # noqa: BLE001 - see the docstring's accepted floor
        pass


@contextlib.contextmanager
def capture_diagnostics(label: str) -> Iterator[None]:
    """Report any exception escaping a capture entrance, and re-raise it as-is.

    This does not detect, classify or prove a *capture* failure.  It fires on any
    exception that escapes the wrapped capture routine and records what the
    streams looked like once it had.

    ``except Exception``, not ``BaseException``: CUDA errors surface as
    ``RuntimeError``, while a ``KeyboardInterrupt`` mid-capture is not evidence
    worth a dump.  The label is released on that path too.

    The dump runs before the ``finally``, so it still sees this site's own label
    in ``open_capture=``.  The bare ``raise`` keeps the original exception object,
    its ``__traceback__``, ``__cause__`` and ``__context__``; that holds because
    :func:`emit_capture_failure_state` is total, so it can never chain a
    "during handling of the above exception" of its own onto the real error.
    """
    _push_open_capture(label)
    try:
        yield
    except Exception as exc:
        emit_capture_failure_state(label, error=type(exc).__name__)
        raise
    finally:
        _pop_open_capture()


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
    say which entrance was open at the time, and so
    :func:`capture_diagnostics` can name it in the failure-time dump.

    ``with ctx`` must stay lexically inside this generator.  The attribution
    harness resolves a capture's label by walking ``f_back`` from the patched
    ``CUDAGraph.capture_begin`` for a frame *named* ``graph_capture`` and reading
    its ``label`` local (``scripts/tools/capture_attribution/run.py``); hoisting
    the ``with`` into a helper would silently break that.
    """
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
    with capture_diagnostics(label):
        with ctx:
            if capture_debug_enabled():
                print(
                    f"  {describe_capture_state(label, event='capture_open')}",
                    flush=True,
                )
            yield


def graphed_callables(
    callables: Any, sample_args: Any, *, label: str, **kwargs: Any
) -> Any:
    """``torch.cuda.make_graphed_callables`` under the failure-time diagnostic.

    torch hard-codes this capture at ``capture_error_mode="global"`` and exposes
    no way to change it, so unlike :func:`graph_capture` this attaches no policy
    -- only observation.  The graphed callable is returned by identity, so replay
    is the same call it was before.

    The open-time line is gated on ``SACCADE_CAPTURE_DEBUG`` for the same reason
    it is in :func:`graph_capture`: an unconditional print here would sit on the
    capture path itself.  Only the failure-time dump is unconditional.
    """
    if capture_debug_enabled():
        print(
            f"[capture-site] {label} mode=global(torch-fixed) "
            f"{describe_capture_state(label, event='capture_open')}",
            flush=True,
        )
    with capture_diagnostics(label):
        return torch.cuda.make_graphed_callables(callables, sample_args, **kwargs)
