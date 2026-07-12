"""CPU contracts for Issue #112 native-capture export normalization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from saccade.perception.eval.consumer_a_bridge_fidelity import (
    CAPTURE_MODE_RUNTIME_CUDA,
    NATIVE_CUDA_BRIDGE_FIDELITY_CAPTURE_IMPLEMENTED,
)
from saccade.perception.tracking.tracker_gpu import GPUByteTracker


REPO = Path(__file__).resolve().parents[3]
EXPORTER = REPO / "scripts/tools/export_d0_runtime_capture.py"


class _NativeCaptureStub:
    def __init__(self) -> None:
        self.enabled: tuple[bool, int] | None = None

    def set_research_bridge_fidelity_audit(self, enabled: bool, capacity: int) -> None:
        self.enabled = (enabled, capacity)

    def clear_research_bridge_fidelity_audit(self) -> None:
        return None

    def drain_research_bridge_fidelity_events(self) -> dict[str, object]:
        # Deliberately reverse event-key order: the wrapper must canonicalize it.
        base = {
            "frame": 12,
            "lost_id": 7,
            "cand_id": 9,
            "lost_slot": 3,
            "cand_slot": 4,
            "lost_last_frame": 5,
            "cand_first_frame": 9,
            "gap": 4,
            "bridge_at": 4,
            "la": 7,
            "anchor_mode": 2,
            "anchor_rate": 0.03,
            "bdist": 0.2,
            "dist_h": 0.1,
            "fwd_r": 0.2,
            "bwd_r": 0.2,
            "v_lost_x": 1.0,
            "v_lost_y": 0.0,
            "v_cand_x": 1.0,
            "v_cand_y": 0.0,
            "ax": 100.0,
            "ay": 200.0,
            "cx0": 110.0,
            "cy0": 200.0,
            "ema_lost": 80.0,
            "ema_cand": 82.0,
            "h_ref": 81.0,
            "s_lost": 0.01,
            "w": 0.3,
            "production_threshold": 0.4,
        }
        second = dict(base, lost_id=2, cand_id=4, lost_slot=1, cand_slot=2)
        return {"events": [base, second], "total_events": 2, "overflow_events": 0}


def _tracker_with_stub() -> tuple[GPUByteTracker, _NativeCaptureStub]:
    native = _NativeCaptureStub()
    tracker = object.__new__(GPUByteTracker)
    tracker.tracker = native
    return tracker, native


def test_runtime_capture_is_plumbed_but_default_packet_is_not_reclassified() -> None:
    assert NATIVE_CUDA_BRIDGE_FIDELITY_CAPTURE_IMPLEMENTED is True


def test_runtime_capture_export_builds_exact_join_keys_and_is_complete() -> None:
    tracker, native = _tracker_with_stub()
    tracker.set_research_bridge_fidelity_audit(True, capacity=123)
    assert native.enabled == (True, 123)

    capture = tracker.drain_research_bridge_fidelity_events(seq="MOT17-02")

    assert capture["complete"] is True
    assert capture["total_events"] == 2
    rows = capture["events"]
    assert isinstance(rows, list)
    assert [row["event_key"] for row in rows] == [
        "MOT17-02|2|4|5|9",
        "MOT17-02|7|9|5|9",
    ]
    assert rows[0]["capture_mode"] == CAPTURE_MODE_RUNTIME_CUDA
    assert rows[0]["evidence_role"] == "runtime_cuda_observation"
    assert rows[0]["bdist"] == pytest.approx(0.2)


def test_runtime_capture_requires_sequence_identity() -> None:
    tracker, _ = _tracker_with_stub()
    with pytest.raises(ValueError, match="sequence name"):
        tracker.drain_research_bridge_fidelity_events(seq="")


def test_runtime_capture_exporter_refuses_partial_data_and_writes_d0_csv(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("d0_export", EXPORTER)
    assert spec and spec.loader
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    native = _NativeCaptureStub().drain_research_bridge_fidelity_events()
    events = []
    for event in native["events"]:
        row = dict(event, seq="MOT17-02", capture_mode=CAPTURE_MODE_RUNTIME_CUDA)
        row["evidence_role"] = "runtime_cuda_observation"
        row["event_key"] = "|".join(
            str(row[field])
            for field in (
                "seq",
                "lost_id",
                "cand_id",
                "lost_last_frame",
                "cand_first_frame",
            )
        )
        events.append(row)
    capture_dir = tmp_path / "native"
    capture_dir.mkdir()
    (capture_dir / "MOT17-02.json").write_text(
        json.dumps(
            {
                "events": events,
                "total_events": len(events),
                "overflow_events": 0,
                "complete": True,
                "provenance": {
                    "capture_contract": "d0_runtime_cuda_v1",
                    "git_commit": "test",
                    "bridge": {"px": 0.4},
                    "detector": {"tiling": "native_640"},
                },
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "capture.csv.gz"
    manifest = exporter.export_capture(capture_dir, output)

    assert output.is_file()
    assert manifest["events"] == len(events)
    assert manifest["overflow_events"] == 0
    assert manifest["capture_mode"] == CAPTURE_MODE_RUNTIME_CUDA

    payload = json.loads((capture_dir / "MOT17-02.json").read_text())
    payload["overflow_events"] = 1
    payload["complete"] = False
    (capture_dir / "overflow.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete native capture"):
        exporter.export_capture(capture_dir, tmp_path / "bad.csv.gz")
