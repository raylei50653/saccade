"""Per-sequence child launcher for ``mot17.py --processes`` (#537 F01).

One child process per sequence, at most ``max_workers`` at a time. A child
that exits non-zero, dies on a signal or cannot be started fails the whole
launch (fail-fast):

* sequences not yet started are not started;
* running children get SIGTERM, then SIGKILL after ``grace_s``, and every
  child is reaped before :func:`run_sequence_children` returns;
* each original failure keeps its sequence, return code, stdout and stderr;
  children stopped by the launcher are reported as ``terminated``, never as a
  failure of their own sequence.

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


@dataclass(frozen=True)
class ChildResult:
    sequence: str
    status: str  # OK / FAILED / TERMINATED / NOT_STARTED
    returncode: int | None  # None: never ran, or Popen itself raised
    stdout: str = ""
    stderr: str = ""
    failed_at: float | None = None  # time.monotonic() when FAILED was seen


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

    def _failed(seq: str, code: int | None, out: str, err: str) -> ChildResult:
        failed_at = time.monotonic()
        abort.set()
        return ChildResult(seq, FAILED, code, out, err, failed_at)

    def _run(seq: str) -> ChildResult:
        if abort.is_set():
            return ChildResult(seq, NOT_STARTED, None)
        print(f" 🚀 Spawning process for {seq}...", flush=True)
        try:
            proc = subprocess.Popen(
                command_for(seq),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=None if env is None else dict(env),
            )
        except OSError as exc:
            return _failed(seq, None, "", f"{type(exc).__name__}: {exc}")
        terminated_at: float | None = None
        while True:
            try:
                out, err = proc.communicate(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                if not abort.is_set():
                    continue
                now = time.monotonic()
                if terminated_at is None:
                    proc.terminate()
                    terminated_at = now
                elif now - terminated_at >= grace_s:
                    proc.kill()
        if terminated_at is not None:
            return ChildResult(seq, TERMINATED, proc.returncode, out, err)
        if proc.returncode != 0:
            return _failed(seq, proc.returncode, out, err)
        return ChildResult(seq, OK, 0, out, err)

    executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
    try:
        futures = [executor.submit(_run, seq) for seq in sequences]
        return [fut.result() for fut in futures]
    except BaseException:
        abort.set()
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

    Its own positive exit code, ``128 + N`` when it died on signal N, and
    :data:`EXIT_START_FAILED` when it could not be started.
    :data:`EXIT_INCOMPLETE` when nothing failed but some child did not complete.
    """
    if all(r.status == OK for r in results):
        return 0
    failed = failures(results)
    if not failed:
        return EXIT_INCOMPLETE
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
