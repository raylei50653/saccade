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

import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
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
    # time.monotonic(), both taken under the spawn lock: when this child was
    # spawned, and when its failure set the abort.
    spawned_at: float | None = None
    failed_at: float | None = None
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
    # Spawning and setting abort share this lock, so no child starts after
    # abort was set: a sequence is NOT_STARTED or its child is stopped.
    spawn_lock = threading.Lock()

    def _abort() -> float:
        with spawn_lock:
            abort.set()
            return time.monotonic()

    def _run(seq: str) -> ChildResult:
        with spawn_lock:
            if abort.is_set():
                return ChildResult(seq, NOT_STARTED, None)
            print(f" 🚀 Spawning process for {seq}...", flush=True)
            cmd = command_for(seq)
            spawned_at = time.monotonic()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=None if env is None else dict(env),
                )
            except OSError as exc:
                abort.set()
                err = f"{type(exc).__name__}: {exc}"
                return ChildResult(
                    seq, FAILED, None, "", err, spawned_at, time.monotonic()
                )
        try:
            return _supervise(seq, proc, spawned_at)
        except BaseException as exc:
            # A launcher-side error must still stop the launch and this child.
            failed_at = _abort()
            proc.kill()
            proc.wait()
            err = f"launcher error while supervising {seq}: {type(exc).__name__}: {exc}"
            return ChildResult(
                seq, FAILED, proc.returncode, "", err, spawned_at, failed_at, True
            )

    def _supervise(
        seq: str, proc: subprocess.Popen[bytes], spawned_at: float
    ) -> ChildResult:
        sent: set[int] = set()
        terminated_at: float | None = None
        while True:
            try:
                raw_out, raw_err = proc.communicate(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                if not abort.is_set():
                    continue
                now = time.monotonic()
                if terminated_at is None:
                    proc.terminate()
                    sent.add(signal.SIGTERM)
                    terminated_at = now
                elif now - terminated_at >= grace_s:
                    proc.kill()
                    sent.add(signal.SIGKILL)
        out = raw_out.decode(errors="replace")
        err = raw_err.decode(errors="replace")
        code = proc.returncode
        # Only a death by a signal the launcher sent is TERMINATED; any other
        # exit (before the signal landed, by handling it, or by a signal the
        # child raised itself) keeps the child's own status.
        if code < 0 and -code in sent:
            return ChildResult(seq, TERMINATED, code, out, err, spawned_at)
        if code != 0:
            return ChildResult(seq, FAILED, code, out, err, spawned_at, _abort())
        return ChildResult(seq, OK, 0, out, err, spawned_at)

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
