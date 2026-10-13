"""#537 C0: pin the observed resource degradation / VRAM-level SHM behavior.

These are characterization tests, not a specification. Each assertion records
what the code at the C0 baseline does, including behavior that the adopted
target (D-537-2, docs/modules/resource/architecture_degradation.md §5) or
REQ-535-07-01 says is wrong. A later fix PR changes the code and these
expectations together; the matrix lives in
docs/architecture/failure_degradation_537.md.
"""

# scope: resource
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import ast
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Iterator
from unittest import mock

import pynvml
import pytest

import saccade.resource.resource_manager as rm
from saccade.resource.resource_manager import (
    DegradationLevel as L,
    ResourceManager,
    VRAMLevelReader,
    VRAMLevelWriter,
    VRAMStats,
)

REPO = Path(__file__).resolve().parents[2]


def _manager(usages: list[float], current: L = L.NORMAL) -> ResourceManager:
    with mock.patch.object(pynvml, "nvmlInit", return_value=None):
        manager = ResourceManager()
    feed = iter(usages)
    manager.get_stats = lambda: VRAMStats(0.0, 0.0, 0.0, next(feed))  # type: ignore[method-assign]
    manager.current_level = current
    return manager


def _decide(current: L, usage: float) -> L:
    return _manager([usage], current).decide_degradation_level()


# ── C2: single-decision transition table (observed as-is) ────────────────────
# Upward thresholds are strict (> 85 / 92 / 96). Rows marked "target differs"
# violate the adopted D-537-2 semantics.
OBSERVED_STEPS = [
    # from NORMAL
    (L.NORMAL, 85.0, L.NORMAL),
    (L.NORMAL, 85.1, L.REDUCED),
    (L.NORMAL, 92.1, L.FAST_PATH),
    (L.NORMAL, 96.1, L.EMERGENCY),  # escalation may skip levels
    # from REDUCED: the only working -5% deadband (< 80)
    (L.REDUCED, 80.0, L.REDUCED),
    (L.REDUCED, 79.9, L.NORMAL),
    (L.REDUCED, 92.1, L.FAST_PATH),
    # from FAST_PATH: no effective hysteresis, leaves at <= 92
    (L.FAST_PATH, 92.1, L.FAST_PATH),
    (L.FAST_PATH, 92.0, L.REDUCED),  # target differs: stay until < 87
    (L.FAST_PATH, 90.0, L.REDUCED),  # issue counterexample; target differs
    (L.FAST_PATH, 0.0, L.REDUCED),
    # from EMERGENCY: non-monotonic in usage
    (L.EMERGENCY, 96.1, L.EMERGENCY),
    (L.EMERGENCY, 96.0, L.FAST_PATH),  # target differs: stay until < 91
    (L.EMERGENCY, 94.0, L.FAST_PATH),  # issue counterexample; target differs
    (L.EMERGENCY, 92.1, L.FAST_PATH),  # target differs
    (L.EMERGENCY, 92.0, L.REDUCED),  # two levels in one decision; target differs
    (L.EMERGENCY, 91.0, L.REDUCED),  # two levels in one decision; target differs
    (L.EMERGENCY, 90.9, L.FAST_PATH),
    (L.EMERGENCY, 0.0, L.FAST_PATH),
]


@pytest.mark.parametrize(
    ("current", "usage", "observed"),
    OBSERVED_STEPS,
    ids=[f"{c.name}@{u}" for c, u, _ in OBSERVED_STEPS],
)
def test_c2_observed_single_decision(current: L, usage: float, observed: L) -> None:
    assert _decide(current, usage) is observed


# ── C3: NVML unavailable is reported as zero usage ───────────────────────────
def test_c3_nvml_init_failure_reports_zero_stats() -> None:
    with mock.patch.object(pynvml, "nvmlInit", side_effect=RuntimeError("nvml")):
        manager = ResourceManager()
    assert vars(manager.get_stats()) == {
        "free_mb": 0.0,
        "used_mb": 0.0,
        "total_mb": 0.0,
        "usage_percent": 0.0,
    }


def test_c3_nvml_init_failure_decays_emergency_to_normal() -> None:
    with mock.patch.object(pynvml, "nvmlInit", side_effect=RuntimeError("nvml")):
        manager = ResourceManager()
    manager.current_level = L.EMERGENCY
    seen = [manager.decide_degradation_level() for _ in range(3)]
    # "cannot measure" ends as NORMAL; no distinct unknown state exists.
    assert seen == [L.FAST_PATH, L.REDUCED, L.NORMAL]


# ── C3 / B0-F07: VRAM-level shared memory ────────────────────────────────────
@pytest.fixture
def shm_name(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Route reader/writer to a private segment so a live service is untouched."""
    name = f"saccade_vram_level_t537_{os.getpid()}"
    monkeypatch.setattr(rm, "_SHM_NAME", name)
    yield name
    try:
        leftover = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    leftover.close()
    leftover.unlink()


def test_c3_reader_without_segment_reads_normal(shm_name: str) -> None:
    reader = VRAMLevelReader()
    try:
        assert reader.read() is L.NORMAL
    finally:
        reader.close()


def test_c3_reader_attaches_to_writer_created_later(shm_name: str) -> None:
    reader = VRAMLevelReader()
    writer = VRAMLevelWriter()
    try:
        assert reader.read() is L.NORMAL
        writer.write(L.FAST_PATH)
        assert reader.read() is L.FAST_PATH
    finally:
        reader.close()
        writer.close()


def test_c3_reader_rejects_out_of_range_byte(shm_name: str) -> None:
    writer = VRAMLevelWriter()
    reader = VRAMLevelReader()
    try:
        writer._shm.buf[0] = 7  # type: ignore[index]
        with pytest.raises(ValueError, match="7 is not a valid DegradationLevel"):
            reader.read()
    finally:
        reader.close()
        writer.close()


def test_b0_f07_reader_attached_before_writer_restart_keeps_stale_level(
    shm_name: str,
) -> None:
    # A previous writer died without unlinking its segment.
    crashed = VRAMLevelWriter()
    crashed.write(L.EMERGENCY)
    crashed._shm.close()

    stale_reader = VRAMLevelReader()  # orchestrator built first
    writer = VRAMLevelWriter()  # dispatcher.start(): unlink stale, recreate
    try:
        assert stale_reader.read() is L.EMERGENCY
        for level in (L.NORMAL, L.REDUCED):
            writer.write(level)
            assert stale_reader.read() is L.EMERGENCY

        fresh_reader = VRAMLevelReader()
        try:
            assert fresh_reader.read() is L.REDUCED
        finally:
            fresh_reader.close()
    finally:
        writer.close()

    # After the writer stops, the old reader still reports its last value.
    try:
        assert stale_reader.read() is L.EMERGENCY
    finally:
        stale_reader.close()


def test_b0_f07_full_mode_builds_orchestrator_before_dispatcher_start() -> None:
    """`main.py --mode full` evaluates PipelineOrchestrator() as a gather()
    argument, so its reader attaches before run_perception() starts the writer."""
    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    run_full = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_full"
    )
    gather = next(
        node
        for node in ast.walk(run_full)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "gather"
    )
    rendered = [ast.unparse(arg) for arg in gather.args]
    assert rendered == ["run_perception()", "PipelineOrchestrator().run()"]
