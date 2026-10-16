"""#537 B0-F07: VRAM-level snapshot across real processes (no GPU).

Covers the restart sequences that kept an attached reader on a dead segment
(S-A/S-B), a reader process exit that unlinked the writer's segment (S-C),
an immediate restart after SIGKILL, a writer killed between writing its temp
file and the rename, a legacy SharedMemory attacher whose tracker removes the
name, and concurrent publish/read where every accepted read must be one
complete, in-order record. Ageing uses an injected clock offset, and every
wait is bounded, so scheduling delays cannot turn a correct run red.
"""

# scope: resource
# function: contract
# lifecycle: active

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Callable, Iterator

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
MINUTE_NS = 60 * 10**9
PRELUDE = "import os, sys, time\nimport saccade.resource.resource_manager as rm\n"


@pytest.fixture
def shm_name(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    name = f"saccade_vram_level_t537x_{os.getpid()}"
    monkeypatch.setattr(rm, "_SHM_NAME", name)
    yield name
    for path in Path(rm._SHM_DIR).glob(f"{name}*"):
        path.unlink()


Spawn = Callable[[str], "subprocess.Popen[str]"]


@pytest.fixture
def spawn(shm_name: str) -> Iterator[Spawn]:
    """Start children on the private name; always kill and reap them."""
    children: list[subprocess.Popen[str]] = []

    def start(body: str) -> subprocess.Popen[str]:
        code = PRELUDE + f"rm._SHM_NAME = {shm_name!r}\n" + textwrap.dedent(body)
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # SIGKILL to the group takes helpers with it
        )
        children.append(proc)
        return proc

    yield start
    for proc in children:
        if proc.poll() is None:
            _kill_group(proc)
        proc.communicate(timeout=TIMEOUT_S)


def _wait_line(proc: subprocess.Popen[str], expected: str) -> None:
    """Read one line byte by byte under a single deadline (never blocks past it)."""
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + TIMEOUT_S
    line = b""
    while not line.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"no full line within {TIMEOUT_S}s: {line!r}"
        ready, _, _ = select.select([fd], [], [], remaining)
        if ready:
            chunk = os.read(fd, 1)
            assert chunk, f"child closed stdout: {line!r}"
            line += chunk
    assert line.decode().strip() == expected, line


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=TIMEOUT_S)


def _offset_reader(offset_ns: list[int]) -> VRAMLevelReader:
    """Real clock plus a test-controlled offset; one-minute freshness bound."""
    return VRAMLevelReader(
        max_age_ns=MINUTE_NS, clock=lambda: vram_level_clock_ns() + offset_ns[0]
    )


WRITER_HOLDS = """
w = rm.VRAMLevelWriter()
w.write(rm.DegradationLevel({level}), rm.vram_level_clock_ns())
print("up", flush=True)
time.sleep(600)
"""


def test_v8_reader_process_exit_keeps_writer_snapshot(spawn: Spawn) -> None:
    """S-C: a reader process exiting no longer removes the writer's record."""
    with VRAMLevelWriter() as writer:
        writer.write(L.EMERGENCY, vram_level_clock_ns())
        child = spawn(
            """
            s = rm.VRAMLevelReader(max_age_ns=10**12).read_snapshot()
            print(s.status.value, int(s.level), flush=True)
            """
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
    spawn: Spawn,
) -> None:
    """S-A/S-B on one reader object: leftover EMERGENCY -> STALE -> new FRESH."""
    offset = [0]
    reader = _offset_reader(offset)
    first = spawn(WRITER_HOLDS.format(level=3))
    _wait_line(first, "up")
    _kill_group(first)

    snap = reader.read_snapshot()
    assert (snap.status, snap.level) == (S.FRESH, L.EMERGENCY)  # within the bound
    old_generation = snap.generation
    offset[0] = 2 * MINUTE_NS  # the dead writer's decision is now too old
    assert reader.read_snapshot().status is S.STALE

    offset[0] = 0
    second = spawn(WRITER_HOLDS.format(level=0))
    _wait_line(second, "up")
    snap = reader.read_snapshot()
    assert (snap.status, snap.level) == (S.FRESH, L.NORMAL)
    assert snap.generation != old_generation


def test_immediate_restart_after_sigkill_is_not_undone(spawn: Spawn) -> None:
    """No tracker owns the record, so nothing of the dead writer can remove
    the successor's snapshot after it is published."""
    first = spawn(WRITER_HOLDS.format(level=3))
    _wait_line(first, "up")
    os.killpg(first.pid, signal.SIGKILL)  # successor starts before reaping
    reader = _offset_reader([0])
    with VRAMLevelWriter() as writer:
        first.wait(timeout=TIMEOUT_S)
        writer.write(L.REDUCED, vram_level_clock_ns())
        time.sleep(1.0)  # room for any delayed cleanup of the predecessor
        assert reader.read_snapshot().generation == writer.generation
        writer.write(L.NORMAL, vram_level_clock_ns())
        snap = reader.read_snapshot()
        assert (snap.status, snap.level) == (S.FRESH, L.NORMAL)


def test_writer_killed_between_temp_write_and_rename(
    shm_name: str, spawn: Spawn
) -> None:
    """Death mid-publication leaves the previous complete record readable and
    an orphan temp file that the next writer removes."""
    child = spawn(
        """
        w = rm.VRAMLevelWriter()
        w.write(rm.DegradationLevel.EMERGENCY, rm.vram_level_clock_ns())
        rm.os.replace = lambda *_: os._exit(9)  # die after the temp file is written
        w.write(rm.DegradationLevel.NORMAL, rm.vram_level_clock_ns())
        """
    )
    child.communicate(timeout=TIMEOUT_S)
    assert child.returncode == 9
    orphans = list(Path(rm._SHM_DIR).glob(f"{shm_name}{rm._TMP_INFIX}{child.pid}.*"))
    assert len(orphans) == 1

    reader = _offset_reader([0])
    snap = reader.read_snapshot()
    assert (snap.status, snap.level) == (S.FRESH, L.EMERGENCY)
    with VRAMLevelWriter() as writer:
        assert not orphans[0].exists()
        writer.write(L.NORMAL, vram_level_clock_ns())
        assert reader.read_snapshot().level is L.NORMAL


def test_legacy_attacher_tracker_removal_heals_on_next_decision(
    shm_name: str, spawn: Spawn
) -> None:
    """Why the deployment rule stops legacy processes first: a pre-fix
    SharedMemory attacher's tracker removes the name when it exits. Readers
    then see MISSING (never a wrong level) until the next publication."""
    reader = _offset_reader([0])
    with VRAMLevelWriter() as writer:
        writer.write(L.EMERGENCY, vram_level_clock_ns())
        child = spawn(
            f"""
            from multiprocessing.shared_memory import SharedMemory
            SharedMemory(name={shm_name!r}, create=False).close()
            """
        )
        child.communicate(timeout=TIMEOUT_S)
        assert child.returncode == 0
        deadline = time.monotonic() + TIMEOUT_S
        while reader.read_snapshot().status is not S.MISSING:
            assert time.monotonic() < deadline, "legacy tracker never removed the name"
            time.sleep(0.05)
        writer.write(L.REDUCED, vram_level_clock_ns())
        snap = reader.read_snapshot()
        assert (snap.status, snap.level) == (S.FRESH, L.REDUCED)


BASE = 1 << 40
CLOCK = 1 << 50


def test_v12_concurrent_reads_see_complete_records_in_order(
    tmp_path: Path, spawn: Spawn
) -> None:
    stop = tmp_path / "stop"
    writer = spawn(
        f"""
        w = rm.VRAMLevelWriter()
        print("up", flush=True)
        i, end = 0, time.monotonic() + {TIMEOUT_S}
        while not os.path.exists({str(stop)!r}) and time.monotonic() < end:
            i += 1
            assert w.write(rm.DegradationLevel(i % 4), {BASE} + i)
        w.close()
        print(i, flush=True)
        """
    )
    _wait_line(writer, "up")
    reader = VRAMLevelReader(max_age_ns=CLOCK, clock=lambda: CLOCK)
    counts = {status: 0 for status in S}
    mismatches = regressions = 0
    generations: set[int | None] = set()
    seen: set[int] = set()
    last = 0
    deadline = time.monotonic() + TIMEOUT_S
    while writer.poll() is None and time.monotonic() < deadline:
        if counts[S.FRESH] >= 1000 and len(seen) >= 100:
            stop.touch()  # enough evidence: let the writer finish
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
    stop.touch()
    out, err = writer.communicate(timeout=TIMEOUT_S)
    assert writer.returncode == 0, err
    published = int(out.strip())

    report = (counts, mismatches, regressions, len(seen), published)
    assert counts[S.INVALID] == 0, report
    assert mismatches == 0 and regressions == 0, report
    assert counts[S.FRESH] >= 1000 and len(seen) >= 100, report
    assert published >= 100 and len(generations) == 1, report
