"""#537 C0: pin the observed resource degradation / VRAM-level SHM behavior.

These are characterization tests, not a specification. Each assertion records
what the code at the C0 baseline does, including behavior that the adopted
target (D-537-2, docs/modules/resource/architecture_degradation.md §5) or
REQ-535-07-01 says is wrong. A later fix PR changes the code and these
expectations together; the matrix lives in
docs/architecture/failure_degradation_537.md. The B0-F07 fix flipped the
VRAM-level snapshot tests (C0-06/07/08) to the target semantics. The C2
hysteresis fix moved the seven rows that changed to the contract tests in
test_resource_degradation_hysteresis_537.py; the rows kept here hold under both
the C0 baseline and the D-537-2 target.
"""

# scope: resource
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterator
from unittest import mock

import pynvml
import pytest

import saccade.resource.resource_manager as rm
from saccade.resource.resource_manager import (
    DegradationLevel as L,
    ResourceManager,
    ResourceMonitorUnavailable,
    VRAMLevelReader,
    VRAMLevelWriter,
    VRAMReadStatus,
    VRAMStats,
    vram_level_clock_ns,
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


# ── C2: single-decision transition table (rows unchanged by the C2 fix) ──────
# Upward thresholds are strict (> 85 / 92 / 96). The seven C0 rows that
# violated D-537-2 (FAST_PATH@92.0/90.0, EMERGENCY@96.0/94.0/92.1/92.0/91.0)
# now live, with the target values, in test_resource_degradation_hysteresis_537.py.
OBSERVED_STEPS = [
    # from NORMAL
    (L.NORMAL, 85.0, L.NORMAL),
    (L.NORMAL, 85.1, L.REDUCED),
    (L.NORMAL, 92.1, L.FAST_PATH),
    (L.NORMAL, 96.1, L.EMERGENCY),  # escalation may skip levels
    # from REDUCED
    (L.REDUCED, 80.0, L.REDUCED),
    (L.REDUCED, 79.9, L.NORMAL),
    (L.REDUCED, 92.1, L.FAST_PATH),
    # from FAST_PATH
    (L.FAST_PATH, 92.1, L.FAST_PATH),
    (L.FAST_PATH, 0.0, L.REDUCED),
    # from EMERGENCY
    (L.EMERGENCY, 96.1, L.EMERGENCY),
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


# ── C3: NVML unavailable is explicit (#537 D-537-4A) ─────────────────────────
def test_c3_nvml_init_failure_raises_named_monitor_error() -> None:
    original = RuntimeError("nvml")
    with mock.patch.object(pynvml, "nvmlInit", side_effect=original):
        manager = ResourceManager()
    with pytest.raises(ResourceMonitorUnavailable) as exc:
        manager.get_stats()
    assert exc.value.__cause__ is original


def test_c3_nvml_init_failure_preserves_hysteresis_state() -> None:
    with mock.patch.object(pynvml, "nvmlInit", side_effect=RuntimeError("nvml")):
        manager = ResourceManager()
    manager.current_level = L.EMERGENCY
    for _ in range(3):
        with pytest.raises(ResourceMonitorUnavailable):
            manager.decide_degradation_level()
        assert manager.current_level is L.EMERGENCY


# ── C3 / B0-F07: VRAM-level snapshot (flipped from observed to target) ──────
# C0-06/07/08 were fixed by the F07 PR; the full contract lives in
# test_vram_level_snapshot_contract_537.py and the cross-process sequences in
# tests/integration/test_vram_level_snapshot_xproc_537.py.
@pytest.fixture
def shm_name(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Route reader/writer to a private name so a live service is untouched."""
    name = f"saccade_vram_level_t537_{os.getpid()}"
    monkeypatch.setattr(rm, "_SHM_NAME", name)
    yield name
    for path in Path(rm._SHM_DIR).glob(f"{name}*"):
        path.unlink()


def _now_reader() -> VRAMLevelReader:
    return VRAMLevelReader(max_age_ns=10**12)


def test_c3_reader_without_segment_reads_missing_not_normal(shm_name: str) -> None:
    snap = _now_reader().read_snapshot()
    assert (snap.status, snap.level) == (VRAMReadStatus.MISSING, None)


def test_c3_reader_follows_writer_created_later(shm_name: str) -> None:
    reader = _now_reader()
    assert reader.read_snapshot().status is VRAMReadStatus.MISSING
    with VRAMLevelWriter() as writer:
        writer.write(L.FAST_PATH, vram_level_clock_ns())
        snap = reader.read_snapshot()
        assert (snap.status, snap.level) == (VRAMReadStatus.FRESH, L.FAST_PATH)


def test_c3_out_of_range_level_is_invalid_not_raised(shm_name: str) -> None:
    fmt = rm._SNAPSHOT
    stamp = vram_level_clock_ns()
    crc = rm._record_crc(fmt.pack(7, 1, 0, 1, stamp))
    (Path(rm._SHM_DIR) / shm_name).write_bytes(fmt.pack(7, 1, crc, 1, stamp))
    snap = _now_reader().read_snapshot()
    assert (snap.status, snap.level, snap.reason) == (
        VRAMReadStatus.INVALID,
        None,
        "level 7",
    )


def test_b0_f07_reader_built_before_writer_restart_follows_new_writer(
    shm_name: str,
) -> None:
    # A previous writer died at EMERGENCY without removing its snapshot.
    crashed = VRAMLevelWriter()
    crashed.write(L.EMERGENCY, vram_level_clock_ns())

    reader = _now_reader()  # orchestrator built first
    with VRAMLevelWriter() as writer:  # dispatcher.start()
        assert reader.read_snapshot().status is VRAMReadStatus.STALE
        for level in (L.NORMAL, L.REDUCED):
            writer.write(level, vram_level_clock_ns())
            assert reader.read_snapshot().level is level

    # After the writer stops, the same reader no longer sees a usable level.
    assert reader.read_snapshot().status is VRAMReadStatus.STALE


def test_b0_f07_full_mode_builds_orchestrator_before_dispatcher_start() -> None:
    """`main.py --mode full` builds the orchestrator (and its reader) before
    run_perception() starts the writer. Kept after F07: the order is unchanged,
    and the stateless reader makes it harmless. The #538 caller-lifecycle fix
    builds the orchestrator before creating any task, so a failing constructor
    leaves nothing running; orchestrator_factory defaults to
    PipelineOrchestrator."""
    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    run_full = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_full"
    )
    statements = [ast.unparse(stmt) for stmt in run_full.body]
    built = statements.index("orchestrator = orchestrator_factory()")
    spawns = [
        ast.unparse(node)
        for node in ast.walk(run_full)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "spawn"
    ]
    assert spawns == [
        "spawn(tasks, perception or run_perception)",
        "spawn(tasks, orchestrator.run)",
    ]
    first_task = next(i for i, text in enumerate(statements) if "spawn(" in text)
    assert built < first_task
    default = run_full.args.defaults[-1]
    assert ast.unparse(default) == "PipelineOrchestrator"
