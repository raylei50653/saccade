"""#537 F01: ``mot17.py --processes`` child failure fails the launch.

CPU fault injection with real child processes (``sys.executable`` running a
scripted child): exit codes, signal deaths, start failures, a child that
ignores SIGTERM, concurrent failures and parent interruption. Every case also
checks that no child is left running or unreaped. Contract:
docs/architecture/failure_degradation_537.md §25.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from saccade.perception.eval import sequence_launcher as sl

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="POSIX signals and /proc semantics"
)

CHILD = r"""
import json, os, signal, sys, time
seq, spec_path, state = sys.argv[1], sys.argv[2], sys.argv[3]
spec = json.loads(open(spec_path).read())[seq]
kind = spec["kind"]
if kind == "stubborn":
    def _note_term(*_):
        with open(os.path.join(state, seq + ".term"), "w") as f:
            f.write(repr(time.monotonic()))
    signal.signal(signal.SIGTERM, _note_term)
if kind == "trap":
    def _own_exit(*_):
        print("trapped-" + seq, file=sys.stderr, flush=True)
        os._exit(spec["code"])
    signal.signal(signal.SIGTERM, _own_exit)
if kind == "trap_kill":
    signal.signal(signal.SIGTERM, lambda *_: os.kill(os.getpid(), signal.SIGKILL))
# Published atomically: a reader never sees an empty pid file.
tmp = os.path.join(state, seq + ".pid.tmp")
with open(tmp, "w") as f:
    f.write(str(os.getpid()))
os.replace(tmp, os.path.join(state, seq + ".pid"))
for peer in spec.get("wait_for", []):
    while not os.path.exists(os.path.join(state, peer + ".pid")):
        time.sleep(0.01)
time.sleep(spec.get("delay", 0.0))
print("out-" + seq, flush=True)
if kind == "ok":
    sys.exit(0)
if kind == "fail":
    print("boom-" + seq, file=sys.stderr, flush=True)
    sys.exit(spec["code"])
if kind == "selfterm":
    os.kill(os.getpid(), signal.SIGTERM)
if kind == "orphaner":
    import subprocess
    subprocess.Popen([sys.executable, sys.argv[0], spec["grandchild"], spec_path, state])
    while not os.path.exists(os.path.join(state, spec["grandchild"] + ".pid")):
        time.sleep(0.01)
    sys.exit(spec["code"])
if kind == "sigkill":
    os.kill(os.getpid(), signal.SIGKILL)
if kind == "badutf8":
    sys.stderr.buffer.write(b"bad-\xff\xfe-" + seq.encode() + b"\n")
    sys.stderr.flush()
    sys.exit(spec["code"])
time.sleep(30)  # "sleep", "stubborn", "trap*": only the launcher ends these early
sys.exit(0)
"""


class Rig:
    def __init__(self, tmp_path: Path, spec: dict[str, dict[str, Any]]) -> None:
        self.state = tmp_path / "state"
        self.state.mkdir()
        self.child = tmp_path / "child.py"
        self.child.write_text(CHILD)
        self.spec_path = tmp_path / "spec.json"
        self.spec_path.write_text(json.dumps(spec))

    def command_for(self, seq: str) -> list[str]:
        return [
            sys.executable,
            str(self.child),
            seq,
            str(self.spec_path),
            str(self.state),
        ]

    def pids(self) -> dict[str, int]:
        return {p.stem: int(p.read_text()) for p in self.state.glob("*.pid")}

    def run(self, seqs: list[str], **kw: Any) -> tuple[list[sl.ChildResult], float]:
        kw.setdefault("max_workers", len(seqs))
        t0 = time.monotonic()
        results = sl.run_sequence_children(seqs, self.command_for, **kw)
        return results, time.monotonic() - t0


def _alive(pid: int) -> bool:
    """True for a running process or an unreaped zombie with this pid."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture(autouse=True)
def _kill_leftovers(tmp_path: Path) -> Any:
    """Teardown even when a test fails: SIGKILL scripted children still alive.

    Only pids whose command line is this test's child script, so a recycled
    pid is never touched.
    """
    yield
    child = str(tmp_path / "child.py")
    for pid_file in (tmp_path / "state").glob("*.pid"):
        try:
            pid = int(pid_file.read_text())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if child.encode() in cmdline:
                os.kill(pid, signal.SIGKILL)
        except (OSError, ValueError):
            pass


def _assert_no_leftovers(rig: Rig, ignore: tuple[str, ...] = ()) -> None:
    leftovers = {
        seq: pid for seq, pid in rig.pids().items() if seq not in ignore and _alive(pid)
    }
    assert leftovers == {}


def _by_seq(results: list[sl.ChildResult]) -> dict[str, sl.ChildResult]:
    return {r.sequence: r for r in results}


def test_all_children_succeed(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {s: {"kind": "ok"} for s in "ABC"})
    results, _ = rig.run(list("ABC"), max_workers=2)
    assert [r.sequence for r in results] == list("ABC")
    assert [r.status for r in results] == [sl.OK] * 3
    assert all(r.stdout.strip() == f"out-{r.sequence}" for r in results)
    assert sl.launch_exit_code(results) == 0
    _assert_no_leftovers(rig)


def test_failure_terminates_running_and_skips_pending(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 3, "wait_for": ["B"], "delay": 0.2},
            "B": {"kind": "sleep"},
            "C": {"kind": "ok"},
        },
    )
    results, elapsed = rig.run(list("ABC"), max_workers=2)
    by = _by_seq(results)
    assert by["A"].status == sl.FAILED
    assert by["A"].returncode == 3
    assert "boom-A" in by["A"].stderr and "out-A" in by["A"].stdout
    assert by["B"].status == sl.TERMINATED
    assert by["B"].returncode == -signal.SIGTERM
    assert by["C"].status == sl.NOT_STARTED
    assert "C" not in rig.pids()
    assert sl.launch_exit_code(results) == 3
    assert [r.sequence for r in sl.failures(results)] == ["A"]
    assert elapsed < 10  # B would sleep 30 s if it were not stopped
    _assert_no_leftovers(rig)


def test_pending_sequences_build_no_command_after_the_abort(tmp_path: Path) -> None:
    """A pending sequence is NOT_STARTED without its command being built (the
    builder may be slow or may raise; neither may turn into a result)."""
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 3, "wait_for": ["B"], "delay": 0.2},
            "B": {"kind": "sleep"},
            "C": {"kind": "ok"},
            "D": {"kind": "ok"},
        },
    )
    built: list[str] = []

    def command_for(seq: str) -> list[str]:
        built.append(seq)
        if seq in ("C", "D"):
            raise AssertionError(f"command built for pending {seq}")
        return rig.command_for(seq)

    results = sl.run_sequence_children(list("ABCD"), command_for, max_workers=2)
    by = _by_seq(results)
    assert by["C"].status == by["D"].status == sl.NOT_STARTED
    assert sorted(built) == ["A", "B"]
    assert sl.launch_exit_code(results) == 3
    _assert_no_leftovers(rig)


def test_child_ignoring_sigterm_is_killed_after_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGKILL is sent no earlier than grace_s after SIGTERM, measured at the
    send_signal calls (Codex r3/r4)."""
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 2, "wait_for": ["B"], "delay": 0.2},
            "B": {"kind": "stubborn"},
        },
    )
    sends: list[tuple[int, float]] = []

    class Popen(subprocess.Popen):  # type: ignore[type-arg]
        def send_signal(self, sig: int) -> None:
            if "B" in list(self.args):  # type: ignore[arg-type]
                sends.append((sig, time.monotonic()))
            super().send_signal(sig)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    results, elapsed = rig.run(["A", "B"], grace_s=1.0)
    by = _by_seq(results)
    assert by["B"].status == sl.TERMINATED
    assert by["B"].returncode == -signal.SIGKILL
    assert [sig for sig, _ in sends] == [signal.SIGTERM, signal.SIGKILL]
    assert 1.0 <= sends[1][1] - sends[0][1] < 2.0
    assert sl.launch_exit_code(results) == 2
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_child_exiting_by_itself_after_abort_keeps_its_own_status(
    tmp_path: Path,
) -> None:
    """Seen on the GPU probe: a child failed by itself while the launcher was
    already stopping it. Only a death by the launcher's own signal is
    TERMINATED; here B handles SIGTERM and exits 4 on its own.
    """
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 3, "wait_for": ["B"], "delay": 0.2},
            "B": {"kind": "trap", "code": 4},
        },
    )
    results, elapsed = rig.run(["A", "B"])
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED
    assert by["B"].returncode == 4
    assert "trapped-B" in by["B"].stderr
    assert [r.sequence for r in sl.failures(results)] == ["A", "B"]
    assert sl.launch_exit_code(results) == 3
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_child_raising_sigkill_on_sigterm_is_not_terminated(tmp_path: Path) -> None:
    """Codex r1: B answers the launcher's SIGTERM by SIGKILLing itself before
    the grace period ends. The launcher never sent SIGKILL, so it is B's own
    failure, not ``terminated``.
    """
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 3, "wait_for": ["B"], "delay": 0.2},
            "B": {"kind": "trap_kill"},
        },
    )
    results, elapsed = rig.run(["A", "B"], grace_s=20.0)
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED
    assert by["B"].returncode == -signal.SIGKILL
    assert sl.launch_exit_code(results) == 3
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_invalid_utf8_output_is_kept_and_still_aborts(tmp_path: Path) -> None:
    """Codex r1: invalid UTF-8 used to raise in the worker and hide B's failure."""
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "sleep"},
            "B": {"kind": "badutf8", "code": 7, "wait_for": ["A"], "delay": 0.2},
        },
    )
    results, elapsed = rig.run(["A", "B"])
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED
    assert by["B"].returncode == 7
    assert "bad-\ufffd\ufffd-B" in by["B"].stderr
    assert by["A"].status == sl.TERMINATED
    assert sl.launch_exit_code(results) == 7
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_launcher_error_fails_the_sequence_and_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex r1: an exception while supervising B must stop A and reap B."""
    rig = Rig(tmp_path, {"A": {"kind": "sleep"}, "B": {"kind": "sleep"}})

    class Popen(subprocess.Popen):  # type: ignore[type-arg]
        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None and "B" in list(self.args):  # type: ignore[arg-type]
                raise RuntimeError("injected supervision failure")
            return super().wait(timeout)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    results, elapsed = rig.run(["A", "B"])
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED and by["B"].launcher_error
    assert "injected supervision failure" in by["B"].stderr
    assert by["A"].status == sl.TERMINATED
    assert sl.launch_exit_code(results) == sl.EXIT_LAUNCHER_ERROR == 70
    assert elapsed < 10
    _assert_no_leftovers(rig)


def _wait_for_pid(rig: Rig, seq: str) -> None:
    deadline = time.monotonic() + 20
    while seq not in rig.pids():
        assert time.monotonic() < deadline, f"{seq} did not start"
        time.sleep(0.01)


@pytest.mark.parametrize("slow", ["popen", "command"])
def test_no_child_is_spawned_after_the_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slow: str
) -> None:
    """Codex r1/r2: a sequence must not be spawned after the abort was set.

    A is spawned first (B's command waits for A's pid file) and fails after
    0.1 s. ``popen``: B's Popen takes 0.5 s, holding the spawn lock, so B
    must be spawned before A's failure could set the abort (``abort_at``).
    ``command``: building B's command takes 0.5 s, so normally the abort is
    set before B reaches the spawn lock and B is not started.
    """
    rig = Rig(
        tmp_path,
        {"A": {"kind": "fail", "code": 3, "delay": 0.1}, "B": {"kind": "sleep"}},
    )

    def command_for(seq: str) -> list[str]:
        if seq == "B":
            _wait_for_pid(rig, "A")
            if slow == "command":
                time.sleep(0.5)
        return rig.command_for(seq)

    class SlowPopen(subprocess.Popen):  # type: ignore[type-arg]
        def __init__(self, args: Any, **kw: Any) -> None:
            if slow == "popen" and "B" in list(args):
                time.sleep(0.5)
            super().__init__(args, **kw)

    monkeypatch.setattr(subprocess, "Popen", SlowPopen)
    results = sl.run_sequence_children(["A", "B"], command_for, max_workers=2)
    by = _by_seq(results)
    assert by["A"].status == sl.FAILED and by["A"].abort_at is not None
    # Under any scheduling, B is either not started or was spawned before the
    # abort (Codex r3: a late A may legitimately let B spawn first).
    if by["B"].status == sl.NOT_STARTED:
        assert "B" not in rig.pids()
    else:
        assert by["B"].status == sl.TERMINATED
        assert by["B"].spawned_at is not None
        assert by["B"].spawned_at < by["A"].abort_at
    assert sl.launch_exit_code(results) == 3
    _assert_no_leftovers(rig)


def test_signal_not_sent_when_popen_reaps_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex r2: B died by its own SIGTERM before the launcher signalled it;
    terminate() reaped it in its poll and sent nothing, so B is FAILED.

    The fake timeout lets B exit (unreaped, WNOWAIT) and only then returns to
    the launcher, which by then has seen A's failure.
    """
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 3},
            "B": {"kind": "selfterm", "wait_for": ["A"], "delay": 0.3},
        },
    )

    class Popen(subprocess.Popen):  # type: ignore[type-arg]
        faked = False

        def wait(self, timeout: float | None = None) -> int:
            if "B" in list(self.args) and not self.faked:  # type: ignore[arg-type]
                self.faked = True
                os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOWAIT)
                raise subprocess.TimeoutExpired(self.args, float(timeout or 0))
            return super().wait(timeout)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    results, elapsed = rig.run(["A", "B"])
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED
    assert by["B"].returncode == -signal.SIGTERM
    assert sl.launch_exit_code(results) == 3
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_command_builder_error_aborts(tmp_path: Path) -> None:
    """Codex r2: an exception from command_for used to escape the worker
    without aborting; A kept running and C could start."""
    rig = Rig(tmp_path, {"A": {"kind": "sleep"}, "C": {"kind": "ok"}})

    def command_for(seq: str) -> list[str]:
        if seq == "B":
            _wait_for_pid(rig, "A")
            raise KeyError("no command for B")
        return rig.command_for(seq)

    t0 = time.monotonic()
    results = sl.run_sequence_children(["A", "B", "C"], command_for, max_workers=2)
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED and by["B"].launcher_error
    assert "KeyError" in by["B"].stderr and "no command for B" in by["B"].stderr
    assert by["A"].status == sl.TERMINATED
    assert by["C"].status == sl.NOT_STARTED
    assert sl.launch_exit_code(results) == sl.EXIT_LAUNCHER_ERROR
    assert time.monotonic() - t0 < 10
    _assert_no_leftovers(rig)


def test_popen_non_oserror_aborts_as_launcher_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex r2: a non-OSError from Popen used to escape the worker."""
    rig = Rig(tmp_path, {"A": {"kind": "sleep"}, "C": {"kind": "ok"}})

    class Popen(subprocess.Popen):  # type: ignore[type-arg]
        def __init__(self, args: Any, **kw: Any) -> None:
            if "B" in list(args):
                raise ValueError("bad Popen arguments")
            super().__init__(args, **kw)

    def command_for(seq: str) -> list[str]:
        if seq == "B":
            _wait_for_pid(rig, "A")
        return rig.command_for(seq)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    results = sl.run_sequence_children(["A", "B", "C"], command_for, max_workers=2)
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED and by["B"].launcher_error
    assert "ValueError: bad Popen arguments" in by["B"].stderr
    assert by["A"].status == sl.TERMINATED
    assert by["C"].status == sl.NOT_STARTED
    assert sl.launch_exit_code(results) == sl.EXIT_LAUNCHER_ERROR
    _assert_no_leftovers(rig)


def test_earliest_observed_failure_wins_over_lock_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex r2: A exits 5 after 0.1 s while B holds the spawn lock in a
    0.5 s Popen that then fails to start. A's failure was observed first, so
    it is the original failure (exit 5), although B set the abort first."""
    rig = Rig(tmp_path, {"A": {"kind": "fail", "code": 5, "delay": 0.1}})

    def command_for(seq: str) -> list[str]:
        if seq == "B":
            _wait_for_pid(rig, "A")
        return rig.command_for(seq)

    class SlowFailingPopen(subprocess.Popen):  # type: ignore[type-arg]
        def __init__(self, args: Any, **kw: Any) -> None:
            if "B" in list(args):
                time.sleep(0.5)
                raise FileNotFoundError("no-such-interpreter")
            super().__init__(args, **kw)

    monkeypatch.setattr(subprocess, "Popen", SlowFailingPopen)
    results = sl.run_sequence_children(["A", "B"], command_for, max_workers=2)
    by = _by_seq(results)
    assert by["A"].status == by["B"].status == sl.FAILED
    assert by["B"].abort_at is not None and by["A"].abort_at is not None
    assert by["B"].abort_at < by["A"].abort_at  # B won the lock ...
    assert [r.sequence for r in sl.failures(results)] == ["A", "B"]  # ... A first
    assert sl.launch_exit_code(results) == 5
    _assert_no_leftovers(rig)


def test_descendant_holding_the_output_does_not_hide_the_exit(tmp_path: Path) -> None:
    """Codex r3: A exits 7 while its own child G keeps A's stdout/stderr
    open. With pipes the launcher waited for EOF (G sleeps 30 s) and B's
    later exit 9 decided the code; the exit is seen at once now."""
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "orphaner", "code": 7, "grandchild": "G", "wait_for": ["B"]},
            "G": {"kind": "sleep"},
            "B": {"kind": "fail", "code": 9, "delay": 1.0},
        },
    )
    results, elapsed = rig.run(["A", "B"])
    by = _by_seq(results)
    assert by["A"].status == sl.FAILED and by["A"].returncode == 7
    assert sl.launch_exit_code(results) == 7
    assert elapsed < 10
    # G is A's child, not the launcher's: out of scope (§25), killed by teardown.
    _assert_no_leftovers(rig, ignore=("G",))


def test_spawn_message_error_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex r3: printing the spawn message (e.g. BrokenPipeError on a closed
    stdout) raised outside every guard; B's child kept running unsupervised."""
    rig = Rig(
        tmp_path, {"A": {"kind": "sleep"}, "B": {"kind": "sleep"}, "C": {"kind": "ok"}}
    )

    def fake_print(*args: Any, **kw: Any) -> None:
        if any("for B" in str(a) for a in args):
            raise BrokenPipeError(32, "Broken pipe")

    def command_for(seq: str) -> list[str]:
        if seq == "B":
            _wait_for_pid(rig, "A")
        return rig.command_for(seq)

    monkeypatch.setattr(sl, "print", fake_print, raising=False)
    results = sl.run_sequence_children(["A", "B", "C"], command_for, max_workers=2)
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED and by["B"].launcher_error
    assert "BrokenPipeError" in by["B"].stderr
    assert by["A"].status == sl.TERMINATED
    assert by["C"].status == sl.NOT_STARTED
    assert sl.launch_exit_code(results) == sl.EXIT_LAUNCHER_ERROR
    _assert_no_leftovers(rig)


def test_capture_file_error_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex r4: B's capture file could not be created (EMFILE); that escaped
    the worker guard, A kept running and C started."""
    rig = Rig(tmp_path, {"A": {"kind": "sleep"}, "C": {"kind": "ok"}})
    real = tempfile.TemporaryFile
    calls = []

    def temporary_file(*a: Any, **kw: Any) -> Any:
        calls.append(1)
        if len(calls) == 3:  # A took the first two (B waits for A's pid)
            raise OSError(24, "Too many open files")
        return real(*a, **kw)

    def command_for(seq: str) -> list[str]:
        if seq == "B":
            _wait_for_pid(rig, "A")
        return rig.command_for(seq)

    monkeypatch.setattr(tempfile, "TemporaryFile", temporary_file)
    results = sl.run_sequence_children(["A", "B", "C"], command_for, max_workers=2)
    by = _by_seq(results)
    assert by["B"].status == sl.FAILED and by["B"].launcher_error
    assert "Too many open files" in by["B"].stderr
    assert by["A"].status == sl.TERMINATED
    assert by["C"].status == sl.NOT_STARTED
    assert sl.launch_exit_code(results) == sl.EXIT_LAUNCHER_ERROR
    _assert_no_leftovers(rig)


def test_failure_is_recorded_before_its_output_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex r4: reading A's (large or slow) output delayed A's failure, so
    B's later exit 7 won. A failed first, so the exit code is 5."""
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 5, "wait_for": ["B"], "delay": 0.1},
            "B": {"kind": "fail", "code": 7, "wait_for": ["A"], "delay": 0.6},
        },
    )
    real = sl._read
    reads = []

    def slow_read(f: Any) -> str:
        reads.append(1)
        if len(reads) == 1:
            time.sleep(1.0)
        return real(f)

    monkeypatch.setattr(sl, "_read", slow_read)
    results, _ = rig.run(["A", "B"])
    assert [r.sequence for r in sl.failures(results)][0] == "A"
    assert sl.launch_exit_code(results) == 5
    _assert_no_leftovers(rig)


def test_reading_the_capture_leaves_the_shared_offset_alone(tmp_path: Path) -> None:
    """Codex r4: seek(0)+read moved the offset shared with descendants that
    inherited the descriptor, so a late writer overwrote the child's output."""
    with tempfile.TemporaryFile() as f:
        f.write(b"ORIGINAL-CHILD-OUTPUT\n")
        f.flush()
        f.seek(9)  # wherever a writer sharing the descriptor left it
        assert sl._read(f) == "ORIGINAL-CHILD-OUTPUT\n"
        assert f.tell() == 9


def test_signal_death_is_a_failure_with_128_plus_n(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path, {"A": {"kind": "sigkill", "wait_for": ["B"]}, "B": {"kind": "sleep"}}
    )
    results, elapsed = rig.run(["A", "B"])
    by = _by_seq(results)
    assert by["A"].status == sl.FAILED
    assert by["A"].returncode == -signal.SIGKILL
    assert by["B"].status == sl.TERMINATED
    assert sl.launch_exit_code(results) == 128 + signal.SIGKILL
    assert "SIGKILL" in sl.describe(by["A"])
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_start_failure_has_a_stable_exit_code(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"B": {"kind": "sleep"}, "C": {"kind": "ok"}})

    def command_for(seq: str) -> list[str]:
        if seq == "A":
            _wait_for_pid(rig, "B")
            return [str(tmp_path / "no-such-interpreter")]
        return rig.command_for(seq)

    # B starts first so A's start failure has a running child to stop.
    t0 = time.monotonic()
    results = sl.run_sequence_children(
        ["B", "A", "C"], command_for, max_workers=2, poll_s=0.05
    )
    elapsed = time.monotonic() - t0
    by = _by_seq(results)
    assert by["A"].status == sl.FAILED
    assert by["A"].returncode is None
    assert "FileNotFoundError" in by["A"].stderr
    assert by["B"].status == sl.TERMINATED
    assert by["C"].status == sl.NOT_STARTED
    assert sl.launch_exit_code(results) == sl.EXIT_START_FAILED == 127
    assert "could not start" in sl.describe(by["A"])
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_first_of_concurrent_failures_sets_the_exit_code(tmp_path: Path) -> None:
    """Two children fail on their own while others still run.

    A long ``poll_s`` keeps the launcher from stopping B before it fails by
    itself; B is listed first so the exit code follows time, not list order.
    """
    rig = Rig(
        tmp_path,
        {
            "B": {"kind": "fail", "code": 7, "wait_for": ["A", "C"], "delay": 0.8},
            "A": {"kind": "fail", "code": 5, "wait_for": ["B", "C"], "delay": 0.1},
            "C": {"kind": "sleep"},
            "D": {"kind": "ok"},
        },
    )
    results, elapsed = rig.run(["B", "A", "C", "D"], max_workers=3, poll_s=3.0)
    by = _by_seq(results)
    assert by["A"].status == by["B"].status == sl.FAILED
    assert [r.sequence for r in sl.failures(results)] == ["A", "B"]
    assert "boom-B" in by["B"].stderr and "boom-A" in by["A"].stderr
    assert by["C"].status == sl.TERMINATED
    assert by["D"].status == sl.NOT_STARTED
    assert sl.launch_exit_code(results) == 5
    assert elapsed < 15
    _assert_no_leftovers(rig)


HARNESS = r"""
import sys
from saccade.perception.eval import sequence_launcher as sl
child, spec, state = sys.argv[1:4]
try:
    sl.run_sequence_children(
        ["A", "B", "C"],
        lambda s: [sys.executable, child, s, spec, state],
        max_workers=2,
        grace_s=0.5,
    )
except KeyboardInterrupt:
    sys.exit(sl.EXIT_INTERRUPTED)
sys.exit(0)
"""


def test_parent_interrupt_reaps_children_and_exits_130(tmp_path: Path) -> None:
    """SIGINT reaches only the parent (as from ``kill -INT``, not a tty)."""
    rig = Rig(
        tmp_path,
        {"A": {"kind": "sleep"}, "B": {"kind": "stubborn"}, "C": {"kind": "ok"}},
    )
    harness = tmp_path / "harness.py"
    harness.write_text(HARNESS)
    # The harness must import the launcher under test, not whatever copy
    # the interpreter's site-packages would resolve.
    src_root = Path(sl.__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": str(src_root)}
    proc = subprocess.Popen(
        [
            sys.executable,
            str(harness),
            str(rig.child),
            str(rig.spec_path),
            str(rig.state),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 20
        while set(rig.pids()) != {"A", "B"}:
            assert time.monotonic() < deadline, "children did not start"
            assert proc.poll() is None, proc.communicate()
            time.sleep(0.05)
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == sl.EXIT_INTERRUPTED == 130, err
    assert "C" not in rig.pids()
    _assert_no_leftovers(rig)


@pytest.mark.parametrize(
    ("results", "code"),
    [
        ([], 0),
        ([sl.ChildResult("A", sl.OK, 0)], 0),
        ([sl.ChildResult("A", sl.FAILED, 9, failed_at=1.0)], 9),
        ([sl.ChildResult("A", sl.FAILED, -15, failed_at=1.0)], 143),
        ([sl.ChildResult("A", sl.FAILED, None, failed_at=1.0)], 127),
        ([sl.ChildResult("A", sl.FAILED, -9, failed_at=1.0, launcher_error=True)], 70),
        (
            [
                sl.ChildResult("A", sl.FAILED, 4, failed_at=2.0),
                sl.ChildResult("B", sl.FAILED, 6, failed_at=1.0),
            ],
            6,
        ),
        ([sl.ChildResult("A", sl.OK, 0), sl.ChildResult("B", sl.TERMINATED, -15)], 1),
        ([sl.ChildResult("A", sl.NOT_STARTED, None)], 1),
    ],
)
def test_launch_exit_code_table(results: list[sl.ChildResult], code: int) -> None:
    assert sl.launch_exit_code(results) == code
