"""#537 B0-F07: orchestrator consumer policy for VRAM-level snapshots.

Only a FRESH EMERGENCY may shed (drop and ACK) a non-anomaly event. A level
that is STALE, MISSING, INVALID or unreadable is never reused from an earlier
read: every event is persisted, no new event-triggered RAG starts, and the
event is counted as unknown. Storage failures are still ACKed (observed as-is;
PEL reclaim is out of scope, #538).
"""

# scope: cognition, resource
# function: behavior
# lifecycle: active

from __future__ import annotations

import asyncio
from typing import Any, Iterator
from unittest import mock

import pytest

import saccade.cognition.orchestrator as omod
from saccade.cognition.orchestrator import PipelineOrchestrator
from saccade.resource.resource_manager import (
    DegradationLevel as L,
    VRAMLevelSnapshot,
    VRAMReadStatus as S,
)

FRESH_EMERGENCY = VRAMLevelSnapshot(S.FRESH, L.EMERGENCY, 1, 0)
FRESH_FAST_PATH = VRAMLevelSnapshot(S.FRESH, L.FAST_PATH, 1, 0)
FRESH_NORMAL = VRAMLevelSnapshot(S.FRESH, L.NORMAL, 2, 0)
STALE = VRAMLevelSnapshot(S.STALE, None, 1, 5, "expired")
MISSING = VRAMLevelSnapshot(S.MISSING, None, reason="no snapshot")
INVALID = VRAMLevelSnapshot(S.INVALID, None, reason="crc")
RAISES = RuntimeError("reader bug")

# One plain high-entropy event (RAG-eligible) and one anomaly per batch.
PLAIN, ANOMALY = 1, 2


def _batch(step: int) -> list[tuple[str, dict[str, Any]]]:
    def event(kind: int, objects: list[str]) -> tuple[str, dict[str, Any]]:
        frame = step * 10 + kind
        meta = {"frame_id": frame, "objects": objects, "entropy_value": 0.95}
        return (f"{frame}-0", {"metadata": meta})

    return [event(PLAIN, ["person"]), event(ANOMALY, ["knife"])]


class ScriptedReader:
    def __init__(self) -> None:
        self.next: VRAMLevelSnapshot | Exception = MISSING

    def read_snapshot(self) -> VRAMLevelSnapshot:
        if isinstance(self.next, Exception):
            raise self.next
        return self.next


class Harness:
    def __init__(self, orchestrator: PipelineOrchestrator) -> None:
        self.o = orchestrator
        self.reader = ScriptedReader()
        orchestrator._vram_reader = self.reader  # type: ignore[assignment]
        self.persisted: list[int] = []
        self.acked: list[str] = []
        self.rag: list[str] = []
        self.store_fails = False

        def add_memory(content: str, metadata: dict[str, Any]) -> None:
            if self.store_fails:
                raise RuntimeError("chroma down")
            self.persisted.append(metadata["frame_id"])

        async def acknowledge(ids: list[str]) -> None:
            self.acked.extend(ids)

        async def rag(query: str) -> None:
            self.rag.append(query)

        orchestrator.memory_store.add_memory = add_memory  # type: ignore[method-assign]
        orchestrator.redis_cache.acknowledge = acknowledge  # type: ignore[method-assign]
        orchestrator._trigger_rag_analysis = rag  # type: ignore[method-assign]

    def step(
        self, step: int, snapshot: VRAMLevelSnapshot | Exception
    ) -> dict[str, Any]:
        self.reader.next = snapshot
        before = (len(self.persisted), len(self.acked), len(self.rag))

        async def run() -> None:
            await self.o.process_event_batch(_batch(step))
            await asyncio.gather(*list(self.o._background_tasks))

        asyncio.run(run())
        return {
            "persisted": [f % 10 for f in self.persisted[before[0] :]],
            "acked": len(self.acked) - before[1],
            "rag": len(self.rag) - before[2],
            **self.o.vram_policy_stats(),
        }


@pytest.fixture
def harness() -> Iterator[Harness]:
    with (
        mock.patch.object(omod, "RedisCache"),
        mock.patch.object(omod, "ChromaStore"),
        mock.patch.object(omod, "HAS_LLAMA_INDEX", False),
    ):
        orchestrator = PipelineOrchestrator()
    with mock.patch.object(omod, "HAS_LLAMA_INDEX", True):
        yield Harness(orchestrator)


def test_sequence_never_reuses_an_earlier_emergency(harness: Harness) -> None:
    """FRESH EMERGENCY -> each unknown status -> recovered FRESH NORMAL,
    through one consumer, so a consumer that remembers the last level fails."""
    steps: list[tuple[VRAMLevelSnapshot | Exception, dict[str, Any]]] = [
        (FRESH_EMERGENCY, {"persisted": [ANOMALY], "rag": 0, "shed": 1, "unknown": 0}),
        (STALE, {"persisted": [PLAIN, ANOMALY], "rag": 0, "shed": 1, "unknown": 2}),
        (MISSING, {"persisted": [PLAIN, ANOMALY], "rag": 0, "shed": 1, "unknown": 4}),
        (INVALID, {"persisted": [PLAIN, ANOMALY], "rag": 0, "shed": 1, "unknown": 6}),
        (RAISES, {"persisted": [PLAIN, ANOMALY], "rag": 0, "shed": 1, "unknown": 8}),
        (
            FRESH_NORMAL,
            {"persisted": [PLAIN, ANOMALY], "rag": 2, "shed": 1, "unknown": 8},
        ),
    ]
    for i, (snapshot, want) in enumerate(steps):
        got = harness.step(i, snapshot)
        assert got["acked"] == 2, (i, got)
        assert sorted(got["persisted"]) == want["persisted"], (i, got)
        assert got["rag"] == want["rag"], (i, got)
        assert (got["shed_events"], got["unknown_events"]) == (
            want["shed"],
            want["unknown"],
        ), (i, got)
    assert harness.o.vram_policy_stats()["last_status"] == "fresh"


def test_fresh_fast_path_persists_all_and_skips_rag(harness: Harness) -> None:
    got = harness.step(0, FRESH_FAST_PATH)
    assert (sorted(got["persisted"]), got["acked"], got["rag"]) == ([1, 2], 2, 0)
    assert (got["shed_events"], got["unknown_events"]) == (0, 0)


def test_status_transitions_are_logged_once(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    for i, snapshot in enumerate([MISSING, MISSING, FRESH_NORMAL, FRESH_NORMAL]):
        harness.step(i, snapshot)
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "VRAM level status" in line
    ]
    assert len(lines) == 2
    assert "none -> missing" in lines[0]
    assert "missing -> fresh" in lines[1]


def test_storage_failure_is_still_acked_observed(harness: Harness) -> None:
    """Observed as-is (not a target): I-1 does not hold for storage failures."""
    harness.store_fails = True
    got = harness.step(0, FRESH_NORMAL)
    assert (got["persisted"], got["acked"], got["rag"]) == ([], 2, 0)
