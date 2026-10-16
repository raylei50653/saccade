"""#537 B0-F07: VRAM-level snapshot contract (rename-published record, stateless reader).

Pins the target semantics adopted for F07: every read re-resolves the public
name and sees one complete record; status is FRESH / STALE / MISSING / INVALID
and the reader never raises; a restarted writer is followed on the next read;
the decision timestamp is captured before the decision.
"""

# scope: resource
# function: contract
# lifecycle: active

from __future__ import annotations

import asyncio
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

import saccade.resource.resource_manager as rm
from saccade.resource.resource_manager import (
    DegradationLevel as L,
    VRAMLevelReader,
    VRAMLevelWriter,
    VRAMReadStatus as S,
    vram_level_clock_ns,
)

T = 1_000_000_000
NOW = 50 * T


@pytest.fixture
def shm_name(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Route reader/writer to a private name so a live service is untouched."""
    name = f"saccade_vram_level_t537c_{os.getpid()}"
    monkeypatch.setattr(rm, "_SHM_NAME", name)
    yield name
    for path in Path(rm._SHM_DIR).glob(f"{name}*"):
        path.unlink()


def _path(name: str) -> Path:
    return Path(rm._SHM_DIR) / name


def _reader(now: int = NOW, max_age_ns: int = T) -> VRAMLevelReader:
    return VRAMLevelReader(max_age_ns=max_age_ns, clock=lambda: now)


def _record(
    level: int = 3,
    version: int = 1,
    generation: int = 7,
    decided_at_ns: int = NOW,
    crc: int | None = None,
) -> bytes:
    fmt = rm._SNAPSHOT
    if crc is None:
        crc = zlib.crc32(fmt.pack(level, version, 0, generation, decided_at_ns))
    return fmt.pack(level, version, crc, generation, decided_at_ns)


# ── V1-V4: status of a well-formed lifecycle ────────────────────────────────
def test_v1_no_writer_reads_missing(shm_name: str) -> None:
    snap = _reader().read_snapshot()
    assert (snap.status, snap.level) == (S.MISSING, None)


def test_v2_writer_before_first_decision_reads_stale(shm_name: str) -> None:
    with VRAMLevelWriter() as writer:
        snap = _reader().read_snapshot()
        assert (snap.status, snap.level, snap.reason) == (
            S.STALE,
            None,
            "no decision yet",
        )
        assert snap.generation == writer.generation


def test_v3_decision_reads_fresh_with_age(shm_name: str) -> None:
    with VRAMLevelWriter() as writer:
        assert writer.write(L.EMERGENCY, NOW - 5)
        snap = _reader().read_snapshot()
        assert (snap.status, snap.level, snap.age_ns) == (S.FRESH, L.EMERGENCY, 5)


def test_v4_age_boundary_is_inclusive(shm_name: str) -> None:
    with VRAMLevelWriter() as writer:
        writer.write(L.FAST_PATH, NOW - T)
        assert _reader().read_snapshot().status is S.FRESH
        snap = _reader(now=NOW + 1).read_snapshot()
        assert (snap.status, snap.level, snap.reason) == (S.STALE, None, "expired")


def test_future_timestamp_is_invalid(shm_name: str) -> None:
    with VRAMLevelWriter() as writer:
        writer.write(L.NORMAL, NOW + 1)
        snap = _reader().read_snapshot()
        assert (snap.status, snap.level) == (S.INVALID, None)


# ── V5-V6: malformed records are INVALID, never an exception ─────────────────
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_record(level=7), "level 7"),
        (_record(version=2), "version 2"),
        (_record(crc=1), "crc"),
        (bytes([3]), "size 1"),  # legacy 1-byte segment
        (b"", "size 0"),  # writer died between create and size
        (_record() + b"\0", "size 25"),
    ],
    ids=["level", "version", "crc", "legacy-1-byte", "empty", "oversized"],
)
def test_v5_v6_malformed_record_is_invalid(
    shm_name: str, payload: bytes, reason: str
) -> None:
    _path(shm_name).write_bytes(payload)
    snap = _reader().read_snapshot()
    assert (snap.status, snap.level, snap.reason) == (S.INVALID, None, reason)


def test_non_regular_file_is_invalid(shm_name: str) -> None:
    _path(shm_name).mkdir()
    try:
        assert _reader().read_snapshot().status is S.INVALID
    finally:
        _path(shm_name).rmdir()


def test_reader_never_raises(shm_name: str) -> None:
    with mock.patch.object(rm.os, "pread", side_effect=RuntimeError("boom")):
        _path(shm_name).write_bytes(_record())
        snap = _reader().read_snapshot()
    assert (snap.status, snap.reason) == (S.INVALID, "unexpected RuntimeError")


# ── V7, restart and recovery ─────────────────────────────────────────────────
def test_v7_writer_close_reads_missing(shm_name: str) -> None:
    writer = VRAMLevelWriter()
    writer.write(L.REDUCED, NOW)
    writer.close()
    writer.close()  # idempotent
    assert _reader().read_snapshot().status is S.MISSING
    assert list(Path(rm._SHM_DIR).glob(f"{shm_name}*")) == []


def test_closing_predecessor_keeps_successor_snapshot(shm_name: str) -> None:
    old = VRAMLevelWriter()
    new = VRAMLevelWriter()
    try:
        new.write(L.NORMAL, NOW)
        old.close()
        snap = _reader().read_snapshot()
        assert (snap.status, snap.level, snap.generation) == (
            S.FRESH,
            L.NORMAL,
            new.generation,
        )
    finally:
        new.close()


def test_restart_replaces_leftover_level_on_same_reader(shm_name: str) -> None:
    """F07 target: a writer died at EMERGENCY; the reader is built first."""
    crashed = VRAMLevelWriter()
    crashed.write(L.EMERGENCY, NOW)  # crashed: never closed
    reader = _reader()
    assert reader.read_snapshot().level is L.EMERGENCY  # within T: still FRESH
    with VRAMLevelWriter() as writer:
        assert reader.read_snapshot().status is S.STALE  # leftover replaced
        for level in (L.NORMAL, L.REDUCED):
            writer.write(level, NOW)
            snap = reader.read_snapshot()
            assert (snap.status, snap.level, snap.generation) == (
                S.FRESH,
                level,
                writer.generation,
            )
    assert reader.read_snapshot().status is S.MISSING


def test_writer_recovers_from_zero_byte_leftover_and_orphan_temps(
    shm_name: str,
) -> None:
    _path(shm_name).write_bytes(b"")
    dead_pid = 2**22 + 1  # above the default pid_max: never a live process
    orphan = _path(f"{shm_name}{rm._TMP_INFIX}{dead_pid}.1")
    orphan.write_bytes(b"\0")
    live = _path(f"{shm_name}{rm._TMP_INFIX}{os.getpid()}.2")
    live.write_bytes(b"\0")
    with VRAMLevelWriter() as writer:
        writer.write(L.FAST_PATH, NOW)
        assert _reader().read_snapshot().level is L.FAST_PATH
        assert not orphan.exists()
        assert live.exists()  # owner alive: not ours to remove


def test_v11_repeated_restarts(shm_name: str) -> None:
    reader = _reader()
    generations = set()
    for i in range(20):
        with VRAMLevelWriter() as writer:
            assert reader.read_snapshot().status is S.STALE
            writer.write(L(i % 4), NOW)
            snap = reader.read_snapshot()
            assert (snap.status, snap.level) == (S.FRESH, L(i % 4))
            generations.add(snap.generation)
    assert len(generations) == 20
    assert list(Path(rm._SHM_DIR).glob(f"{shm_name}*")) == []


# ── writer API ───────────────────────────────────────────────────────────────
def test_write_rejects_missing_timestamp_and_bad_level(shm_name: str) -> None:
    with VRAMLevelWriter() as writer:
        with pytest.raises(ValueError):
            writer.write(L.NORMAL, 0)
        with pytest.raises(ValueError):
            writer.write(7, NOW)  # type: ignore[arg-type]
        assert _reader().read_snapshot().status is S.STALE


def test_failed_publish_keeps_previous_snapshot(shm_name: str) -> None:
    with VRAMLevelWriter() as writer:
        writer.write(L.REDUCED, NOW)
        with mock.patch.object(rm.os, "replace", side_effect=OSError("ENOSPC")):
            assert writer.write(L.EMERGENCY, NOW) is False
        assert _reader().read_snapshot().level is L.REDUCED
        assert not Path(writer._tmp).exists()
        assert writer.write(L.NORMAL, NOW) is True


def test_clock_is_boottime() -> None:
    before = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    stamp = vram_level_clock_ns()
    assert before <= stamp <= time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def test_record_layout_keeps_level_in_byte_zero() -> None:
    assert rm._SNAPSHOT.size == 24
    assert struct.unpack_from("<B", _record(level=2))[0] == 2


# ── decision stamp: captured before the decision (dispatcher) ───────────────
def test_dispatcher_stamps_before_deciding() -> None:
    from saccade.perception import dispatcher as dmod

    calls: list[str] = []
    manager = mock.MagicMock()

    def decide() -> L:
        calls.append("decide")
        return L.FAST_PATH

    def clock() -> int:
        calls.append("clock")
        return 1234

    manager.decide_degradation_level.side_effect = decide
    dispatcher = dmod.AsyncDispatcher(
        detector=mock.MagicMock(), resource_manager=manager
    )
    dispatcher._vram_writer = mock.MagicMock()
    dispatcher._running = True

    async def process(*_: object) -> None:
        dispatcher._running = False

    async def run() -> None:
        dispatcher.queue.put_nowait(("s", mock.MagicMock(), 0.0))  # type: ignore[arg-type]
        with (
            mock.patch.object(dmod, "vram_level_clock_ns", clock),
            mock.patch.object(dispatcher, "_process_batch", process),
        ):
            await asyncio.wait_for(dispatcher._worker_loop(), timeout=10)

    asyncio.run(run())
    assert calls == ["clock", "decide"]
    dispatcher._vram_writer.write.assert_called_once_with(L.FAST_PATH, 1234)
