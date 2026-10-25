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
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(os.path.join(state, seq + ".pid"), "w").write(str(os.getpid()))
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
if kind == "sigkill":
    os.kill(os.getpid(), signal.SIGKILL)
time.sleep(30)  # "sleep" and "stubborn": only the launcher ends these early
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


def _assert_no_leftovers(rig: Rig) -> None:
    leftovers = {seq: pid for seq, pid in rig.pids().items() if _alive(pid)}
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


def test_child_ignoring_sigterm_is_killed_after_grace(tmp_path: Path) -> None:
    rig = Rig(
        tmp_path,
        {
            "A": {"kind": "fail", "code": 2, "wait_for": ["B"], "delay": 0.2},
            "B": {"kind": "stubborn"},
        },
    )
    results, elapsed = rig.run(["A", "B"], grace_s=0.5)
    by = _by_seq(results)
    assert by["B"].status == sl.TERMINATED
    assert by["B"].returncode == -signal.SIGKILL
    assert sl.launch_exit_code(results) == 2
    assert elapsed < 10
    _assert_no_leftovers(rig)


def test_signal_death_is_a_failure_with_128_plus_n(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"A": {"kind": "sigkill"}, "B": {"kind": "sleep"}})
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
