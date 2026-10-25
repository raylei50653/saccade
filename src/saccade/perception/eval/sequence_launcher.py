"""Per-sequence child launcher for ``mot17.py --processes`` (#537 F01).

One child process per sequence, at most ``max_workers`` at a time. A child
that exits non-zero, dies on a signal or cannot be started fails the whole
launch (fail-fast):

* sequences not yet started are not started (no child is spawned after the
  abort is set);
* running children get SIGTERM, then SIGKILL after ``grace_s``, and every
  child is reaped before :func:`run_sequence_children` returns;
* each original failure keeps its sequence, return code, stdout and stderr;
  a child that dies from a signal the launcher sent is reported as
  ``terminated``, never as a failure of its own sequence, while a child that
  exits by itself after the abort keeps its own status;
* an error in the launcher while supervising a child fails that sequence and
  aborts the launch like a child failure (child output is decoded with
  replacement, so invalid UTF-8 is not such an error).

An exception in the waiting thread (``KeyboardInterrupt`` included) stops and
reaps the children the same way before it propagates; the caller exits with
:data:`EXIT_INTERRUPTED` for ``KeyboardInterrupt``. SIGTERM delivered to the
parent itself is not handled. Contract:
docs/architecture/failure_degradation_537.md §25.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import IO
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

OK = "ok"
FAILED = "failed"
TERMINATED = "terminated"
NOT_STARTED = "not_started"

# Parent exit codes that do not come from a child's own exit status.
EXIT_START_FAILED = 127  # the original failure is a child that could not be started
EXIT_INCOMPLETE = 1  # no child failed, yet not every child completed
EXIT_INTERRUPTED = 130  # KeyboardInterrupt, after the children were reaped
EXIT_LAUNCHER_ERROR = 70  # the launcher itself failed supervising a child (EX_SOFTWARE)


@dataclass(frozen=True)
class ChildResult:
    sequence: str
    status: str  # OK / FAILED / TERMINATED / NOT_STARTED
    returncode: int | None  # None: never ran, or Popen itself raised
    stdout: str = ""
    stderr: str = ""
    # time.monotonic(). spawned_at: right after Popen returned, under the
    # spawn lock. failed_at: when the failure was observed (orders failures).
    # abort_at: when that failure set the abort, under the spawn lock.
    spawned_at: float | None = None
    failed_at: float | None = None
    abort_at: float | None = None
    launcher_error: bool = False  # FAILED by the launcher, not by the child


def run_sequence_children(
    sequences: Sequence[str],
    command_for: Callable[[str], list[str]],
    *,
    max_workers: int,
    env: Mapping[str, str] | None = None,
    grace_s: float = 10.0,
    poll_s: float = 0.2,
) -> list[ChildResult]:
    """Run one child per sequence; results come back in ``sequences`` order."""
    abort = threading.Event()
    # The abort check plus Popen, and setting the abort, share this lock, so
    # no child is spawned after the abort was set. It is held only around
    # Popen; the command is built outside it.
    spawn_lock = threading.Lock()

    def _abort() -> float:
        with spawn_lock:
            abort.set()
            return time.monotonic()

    def _run(seq: str) -> ChildResult:
        # Any error in this worker, wherever it happens (capture files too),
        # kills and reaps its child and aborts the launch as a launcher error.
        proc: subprocess.Popen[bytes] | None = None
        spawned_at: float | None = None
        with contextlib.ExitStack() as files:
            try:
                if abort.is_set():
                    return ChildResult(seq, NOT_STARTED, None)
                cmd = command_for(seq)
                out_f = files.enter_context(tempfile.TemporaryFile())
                err_f = files.enter_context(tempfile.TemporaryFile())
                with spawn_lock:
                    if abort.is_set():
                        return ChildResult(seq, NOT_STARTED, None)
                    try:
                        # Files, not pipes: a descendant that keeps the
                        # output open cannot hide the child's own exit.
                        proc = subprocess.Popen(
                            cmd,
                            stdout=out_f,
                            stderr=err_f,
                            env=None if env is None else dict(env),
                        )
                    except OSError as exc:
                        failed_at = time.monotonic()
                        abort.set()
                        return ChildResult(
                            seq, FAILED, None, "", _exc(exc), None, failed_at, failed_at
                        )
                    spawned_at = time.monotonic()
                print(f" 🚀 Spawned process for {seq} (pid {proc.pid})", flush=True)
                return _supervise(seq, proc, spawned_at, out_f, err_f)
            except BaseException as exc:
                # Record the failure and abort first; cleanup may be slow.
                failed_at = time.monotonic()
                abort_at = _abort()
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    proc.wait()
                err = f"launcher error for {seq}: {_exc(exc)}"
                code = None if proc is None else proc.returncode
                return ChildResult(
                    seq, FAILED, code, "", err, spawned_at, failed_at, abort_at, True
                )
        raise AssertionError("unreachable")  # pragma: no cover

    def _signal(proc: subprocess.Popen[bytes], sig: int, sent: set[int]) -> None:
        # Popen.send_signal polls first and sends nothing when that reaps the
        # child; only a signal that was actually sent can make it TERMINATED.
        proc.send_signal(sig)
        if proc.returncode is None:
            sent.add(sig)

    def _supervise(
        seq: str,
        proc: subprocess.Popen[bytes],
        spawned_at: float,
        out_f: IO[bytes],
        err_f: IO[bytes],
    ) -> ChildResult:
        sent: set[int] = set()
        terminated_at: float | None = None
        while True:
            try:
                proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                if not abort.is_set():
                    continue
                now = time.monotonic()
                if terminated_at is None:
                    _signal(proc, signal.SIGTERM, sent)
                    terminated_at = time.monotonic()  # grace counts from the send
                elif now - terminated_at >= grace_s:
                    _signal(proc, signal.SIGKILL, sent)
        code = proc.returncode
        # Only a death by a signal the launcher sent is TERMINATED; any other
        # exit (before the signal landed, by handling it, or by a signal the
        # child raised itself before it was signalled) keeps the child's own
        # status. A child that dies of that same signal after receiving it
        # (e.g. cleanup, then re-raising SIGTERM) is counted as stopped.
        if code < 0 and -code in sent:
            return ChildResult(
                seq, TERMINATED, code, _read(out_f), _read(err_f), spawned_at
            )
        if code != 0:
            # Record the failure and abort before collecting its output.
            failed_at = time.monotonic()
            abort_at = _abort()
            out, err = _read(out_f), _read(err_f)
            return ChildResult(
                seq, FAILED, code, out, err, spawned_at, failed_at, abort_at
            )
        return ChildResult(seq, OK, 0, _read(out_f), _read(err_f), spawned_at)

    executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
    try:
        futures = [executor.submit(_run, seq) for seq in sequences]
        return [fut.result() for fut in futures]
    except BaseException:
        _abort()
        raise
    finally:
        # Workers observe ``abort`` and reap their own child; wait for them.
        executor.shutdown(wait=True, cancel_futures=True)


def _read(f: IO[bytes]) -> str:
    """The capture as it is now, without ever raising.

    Bytes up to the size seen on entry, read with pread from offset 0: the
    file offset is shared with every process that inherited the descriptor
    (seeking would let a late writer clobber it), and a descendant that
    keeps appending must not keep the read going. The child's status is
    already known here, so a read error must not replace it: it is reported
    after whatever was readable.
    """
    chunks: list[bytes] = []
    try:
        fd = f.fileno()
        size, offset = os.fstat(fd).st_size, 0
        while offset < size:
            chunk = os.pread(fd, min(1 << 20, size - offset), offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
    except Exception as exc:
        chunks.append(f"\n[launcher could not read the rest: {_exc(exc)}]".encode())
    return b"".join(chunks).decode(errors="replace")


def _exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def failures(results: Sequence[ChildResult]) -> list[ChildResult]:
    """FAILED results, earliest first; the first one is the original failure."""
    failed = [r for r in results if r.status == FAILED]
    return sorted(failed, key=lambda r: r.failed_at or 0.0)


def launch_exit_code(results: Sequence[ChildResult]) -> int:
    """0 only when every child succeeded; otherwise from the original failure.

    Its own positive exit code, ``128 + N`` when it died on signal N,
    :data:`EXIT_START_FAILED` when it could not be started and
    :data:`EXIT_LAUNCHER_ERROR` when the launcher failed supervising it.
    :data:`EXIT_INCOMPLETE` when nothing failed but some child did not complete.
    """
    if all(r.status == OK for r in results):
        return 0
    failed = failures(results)
    if not failed:
        return EXIT_INCOMPLETE
    if failed[0].launcher_error:
        return EXIT_LAUNCHER_ERROR
    code = failed[0].returncode
    if code is None:
        return EXIT_START_FAILED
    return code if code > 0 else 128 - code


def report(results: Sequence[ChildResult], exit_code: int) -> None:
    """Print every result and, on failure, the summary line; best effort.

    The exit code is decided before this runs; a closed stdout/stderr
    (BrokenPipeError) must not replace it, so a stream that fails is pointed
    at /dev/null, which also keeps the interpreter's flush at exit quiet.
    """
    _best_effort(lambda: [print(describe(r)) for r in results])
    if exit_code != 0:
        failed = failures(results)
        origin = failed[0].sequence if failed else "none"
        _best_effort(
            lambda: print(
                f"\n❌ [Multi-Process Launcher] Launch failed (original failure: "
                f"{origin}); metrics not computed, exit {exit_code}",
                file=sys.stderr,
            )
        )


def report_interrupt() -> None:
    _best_effort(
        lambda: print(
            "\n🛑 [Multi-Process Launcher] Interrupted; children stopped and reaped",
            file=sys.stderr,
        )
    )


def _best_effort(write: Callable[[], object]) -> None:
    # Any failure here is swallowed: a stream may be broken (EPIPE) or absent
    # (sys.stdout is None when the parent started with fd 1 closed, so even
    # .flush() raises). A stream that cannot be flushed is pointed at
    # /dev/null so the interpreter's flush at exit stays quiet.
    try:
        write()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                try:
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stream.fileno())
                    os.close(devnull)
                except Exception:
                    pass


def describe(result: ChildResult) -> str:
    if result.status == OK:
        return f"✅ Process for {result.sequence} finished successfully"
    if result.status == NOT_STARTED:
        return f"⏭️  Process for {result.sequence} was not started (launch aborted)"
    if result.status == TERMINATED:
        return (
            f"⛔ Process for {result.sequence} was terminated by the launcher "
            f"(code {_code(result.returncode)})"
        )
    return (
        f"❌ Process for {result.sequence} failed with code "
        f"{_code(result.returncode)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def _code(returncode: int | None) -> str:
    if returncode is None:
        return "none (could not start)"
    if returncode < 0:
        try:
            return f"{returncode} ({signal.Signals(-returncode).name})"
        except ValueError:
            return str(returncode)
    return str(returncode)
