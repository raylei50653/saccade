"""#537 B0-F07: VRAM-level snapshot across real processes (no GPU).

Covers the restart sequences that kept an attached reader on a dead segment
(S-A/S-B), a reader process exit that unlinked the writer's segment (S-C),
an immediate restart after SIGKILL, and concurrent publish/read where every
accepted read must be one complete, in-order record.
"""

# scope: resource
# function: contract
# lifecycle: active

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterator

import pytest

import saccade.resource.resource_manager as rm
from saccade.resource.resource_manager import (
    DegradationLevel as L,
    VRAMLevelReader,
    VRAMLevelWriter,
    VRAMReadStatus as S,
    vram_level_clock_ns,
)

TIMEOUT_S = 30
PRELUDE = "import sys, time\nimport saccade.resource.resource_manager as rm\n"


@pytest.fixture
def shm_name(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    name = f"saccade_vram_level_t537x_{os.getpid()}"
    monkeypatch.setattr(rm, "_SHM_NAME", name)
    yield name
    for path in Path(rm._SHM_DIR).glob(f"{name}*"):
        path.unlink()


def _spawn(name: str, body: str) -> subprocess.Popen[str]:
    code = PRELUDE + f"rm._SHM_NAME = {name!r}\n" + textwrap.dedent(body)
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # SIGKILL to the group takes any helper with it
    )


def _wait_line(proc: subprocess.Popen[str], expected: str) -> None:
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line == expected, (line, proc.stderr.read() if proc.stderr else "")


def _kill_group(proc: subprocess.Popen[str]) -> None:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=TIMEOUT_S)


WRITER_HOLDS = """
w = rm.VRAMLevelWriter()
w.write(rm.DegradationLevel({level}), rm.vram_level_clock_ns())
print("up", flush=True)
time.sleep({TIMEOUT_S})
"""


def test_v8_reader_process_exit_keeps_writer_snapshot(shm_name: str) -> None:
    """S-C: a reader process exiting no longer unlinks the writer's record."""
    with VRAMLevelWriter() as writer:
        writer.write(L.EMERGENCY, vram_level_clock_ns())
        child = _spawn(
            shm_name,
            """
            s = rm.VRAMLevelReader(max_age_ns=10**12).read_snapshot()
            print(s.status.value, int(s.level), flush=True)
            """,
        )
        out, err = child.communicate(timeout=TIMEOUT_S)
        assert child.returncode == 0, err
        assert out.split() == ["fresh", "3"]
        assert "leaked" not in err
        snap = VRAMLevelReader(max_age_ns=10**12).read_snapshot()
        assert (snap.status, snap.level, snap.generation) == (
            S.FRESH,
            L.EMERGENCY,
            writer.generation,
        )


def test_v9_v10_sigkilled_writer_ages_out_then_restart_is_followed(
    shm_name: str,
) -> None:
    """S-A/S-B on one reader object: leftover EMERGENCY -> STALE -> new FRESH."""
    max_age_ns = 300_000_000
    reader = VRAMLevelReader(max_age_ns=max_age_ns)
    first = _spawn(shm_name, WRITER_HOLDS.format(level=3, TIMEOUT_S=TIMEOUT_S))
    _wait_line(first, "up")
    _kill_group(first)

    snap = reader.read_snapshot()
    assert (snap.status, snap.level) == (S.FRESH, L.EMERGENCY)  # within T
    old_generation = snap.generation
    time.sleep(max_age_ns / 1e9 + 0.1)
    assert reader.read_snapshot().status is S.STALE

    second = _spawn(shm_name, WRITER_HOLDS.format(level=0, TIMEOUT_S=TIMEOUT_S))
    try:
        _wait_line(second, "up")
        snap = reader.read_snapshot()
        assert (snap.status, snap.level) == (S.FRESH, L.NORMAL)
        assert snap.generation != old_generation
    finally:
        _kill_group(second)


def test_immediate_restart_after_sigkill_is_not_undone(shm_name: str) -> None:
    """No tracker owns the record, so nothing of the dead writer can remove
    the successor's snapshot after it is published."""
    first = _spawn(shm_name, WRITER_HOLDS.format(level=3, TIMEOUT_S=TIMEOUT_S))
    _wait_line(first, "up")
    os.killpg(first.pid, signal.SIGKILL)  # successor starts before reaping
    with VRAMLevelWriter() as writer:
        first.wait(timeout=TIMEOUT_S)
        writer.write(L.REDUCED, vram_level_clock_ns())
        time.sleep(1.0)  # room for any delayed cleanup of the predecessor
        snap = VRAMLevelReader().read_snapshot()
        assert snap.generation == writer.generation
        writer.write(L.NORMAL, vram_level_clock_ns())
        snap = VRAMLevelReader().read_snapshot()
        assert (snap.status, snap.level) == (S.FRESH, L.NORMAL)


BASE = 1 << 40
CLOCK = 1 << 50


def test_v12_concurrent_reads_see_complete_records_in_order(shm_name: str) -> None:
    writer = _spawn(
        shm_name,
        f"""
        w = rm.VRAMLevelWriter()
        print("up", flush=True)
        i, end = 0, time.monotonic() + 1.5
        while time.monotonic() < end:
            i += 1
            assert w.write(rm.DegradationLevel(i % 4), {BASE} + i)
        w.close()
        print(i, flush=True)
        """,
    )
    _wait_line(writer, "up")
    reader = VRAMLevelReader(max_age_ns=CLOCK, clock=lambda: CLOCK)
    counts = {status: 0 for status in S}
    mismatches = regressions = 0
    generations: set[int | None] = set()
    seen: set[int] = set()
    last = 0
    while writer.poll() is None:
        snap = reader.read_snapshot()
        counts[snap.status] += 1
        if snap.status is not S.FRESH:
            continue
        assert snap.age_ns is not None and snap.level is not None
        i = CLOCK - snap.age_ns - BASE
        generations.add(snap.generation)
        if snap.level != i % 4:
            mismatches += 1
        if i < last:
            regressions += 1
        last = i
        seen.add(i)
    out, err = writer.communicate(timeout=TIMEOUT_S)
    assert writer.returncode == 0, err
    published = int(out.strip())

    report = (counts, mismatches, regressions, len(seen), published)
    assert counts[S.INVALID] == 0, report
    assert mismatches == 0 and regressions == 0, report
    assert counts[S.FRESH] >= 1000 and len(seen) >= 100, report
    assert published >= 1000 and len(generations) == 1, report
