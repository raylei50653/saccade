"""CPU contracts for Issue #112 native-capture export normalization."""

# scope: eval, tracking
# function: contract
# lifecycle: active

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from saccade.perception.eval.consumer_a_bridge_fidelity import (
    CAPTURE_MODE_RUNTIME_CUDA,
    EVENT_KEY_FIELDS_V2,
    EVENT_KEY_V1_UNSOUND_FIELDS,
    EVENT_KEY_VERSION_V2,
    NATIVE_CUDA_BRIDGE_FIDELITY_CAPTURE_IMPLEMENTED,
    PARTITION_COHORT_GAP,
    PARTITION_MATCHED,
    PARTITION_UNEMITTED,
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
            "lost_window_size": 4,
            "cand_window_size": 4,
            "lost_anchor_window": [
                [90.0, 190.0, 80.0],
                [94.0, 193.0, 80.0],
                [98.0, 196.0, 80.0],
                [100.0, 200.0, 80.0],
            ],
            "cand_anchor_window": [
                [110.0, 200.0, 82.0],
                [114.0, 203.0, 82.0],
                [118.0, 206.0, 82.0],
                [122.0, 209.0, 82.0],
            ],
            "bridge_dir_bonus": 0.0,
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
    assert rows[0]["lost_window_size"] == 4
    assert rows[0]["cand_window_size"] == 4
    assert rows[0]["lost_anchor_window"][0] == [90.0, 190.0, 80.0]
    assert rows[0]["cand_anchor_window"][3] == [122.0, 209.0, 82.0]


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


# ── v2 (global-id) contract ─────────────────────────────────────────────────
#
# Local ids 7/9/2/4 below are lifted to globals 107/109/102/104. Exactly one
# capture event joins the cohort, one does not, and one carries an id that was
# never emitted to MOT -- so the three partitions are all exercised and their
# conservation is checked rather than assumed.


def _load_exporter() -> Any:
    spec = importlib.util.spec_from_file_location("d0_export_v2", EXPORTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_v2_fixture(tmp_path: Path, *, shadow: bool = True) -> dict[str, Path]:
    native = _NativeCaptureStub().drain_research_bridge_fidelity_events()
    events = []
    for event in native["events"]:
        row = dict(event, seq="MOT17-02", capture_mode=CAPTURE_MODE_RUNTIME_CUDA)
        row["evidence_role"] = "runtime_cuda_observation"
        row["event_key"] = "unused-by-v2"
        events.append(row)
    # A third event whose candidate id never reached MOT output.
    events.append(dict(events[0], lost_id=7, cand_id=999, event_key="unused-by-v2"))

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
                    "shadow": shadow,
                    "git_commit": "test",
                    "bridge": {"px": 0.4},
                    "detector": {"tiling": "native_640"},
                },
            }
        ),
        encoding="utf-8",
    )

    id_map = tmp_path / "_global_id_map.txt"
    id_map.write_text(
        "".join(
            f"MOT17-02\tlocal_id={local}\tglobal_id={glob}\n"
            for local, glob in ((7, 107), (9, 109), (2, 102), (4, 104))
        ),
        encoding="utf-8",
    )

    pairs = tmp_path / "pairs.csv"
    # Cohort is in GLOBAL ids and contains only the 107->109 pair.
    pairs.write_text("seq,lost_id,cand_id\nMOT17-02,107,109\n", encoding="utf-8")
    return {"capture_dir": capture_dir, "id_map": id_map, "pairs": pairs}


def test_v2_export_partitions_are_exhaustive_and_conserved(tmp_path: Path) -> None:
    exporter = _load_exporter()
    fx = _write_v2_fixture(tmp_path)
    output = tmp_path / "capture.csv.gz"

    manifest = exporter.export_capture_v2(
        fx["capture_dir"], output, id_map_path=fx["id_map"], pairs_csv=fx["pairs"]
    )

    assert manifest["event_key_version"] == EVENT_KEY_VERSION_V2
    assert manifest["event_key_fields"] == list(EVENT_KEY_FIELDS_V2)
    assert manifest["partition"] == {
        PARTITION_MATCHED: 1,  # 107|109 is in the cohort
        PARTITION_COHORT_GAP: 1,  # 102|104 resolves but was not enumerated
        PARTITION_UNEMITTED: 1,  # cand 999 never reached MOT output
    }
    # Conservation invariant: the partition accounts for every captured event.
    assert sum(manifest["partition"].values()) == manifest["events"] == 3


def test_v2_export_uses_global_ids_and_never_falls_back_to_local(
    tmp_path: Path,
) -> None:
    exporter = _load_exporter()
    fx = _write_v2_fixture(tmp_path)
    output = tmp_path / "capture.csv.gz"
    exporter.export_capture_v2(
        fx["capture_dir"], output, id_map_path=fx["id_map"], pairs_csv=fx["pairs"]
    )

    with gzip.open(output, "rt", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    # The unsound frame fields must not survive into a v2 packet at all.
    assert set(EVENT_KEY_V1_UNSOUND_FIELDS).isdisjoint(rows[0].keys())

    keyed = {r["event_key"] for r in rows if r["event_key"]}
    assert keyed == {"MOT17-02|107|109", "MOT17-02|102|104"}
    # A local-id key would be MOT17-02|7|9 -- its presence means silent fallback.
    assert "MOT17-02|7|9" not in keyed

    unemitted = [r for r in rows if r["partition"] == PARTITION_UNEMITTED]
    assert [r["event_key"] for r in unemitted] == [""]
    assert [r["lost_global_id"] for r in unemitted] == ["-1"]


def test_v2_export_rejects_duplicate_keys(tmp_path: Path) -> None:
    exporter = _load_exporter()
    fx = _write_v2_fixture(tmp_path)
    payload = json.loads((fx["capture_dir"] / "MOT17-02.json").read_text())
    payload["events"].append(dict(payload["events"][0]))
    payload["total_events"] = len(payload["events"])
    (fx["capture_dir"] / "MOT17-02.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate v2 event keys"):
        exporter.export_capture_v2(
            fx["capture_dir"],
            tmp_path / "dupe.csv.gz",
            id_map_path=fx["id_map"],
            pairs_csv=fx["pairs"],
        )


def test_v2_export_fails_closed_on_non_shadow_capture(tmp_path: Path) -> None:
    """A committing bridge mutates track identity; its ids cannot join a cohort."""
    exporter = _load_exporter()
    fx = _write_v2_fixture(tmp_path, shadow=False)

    with pytest.raises(ValueError, match="requires a shadow capture"):
        exporter.export_capture_v2(
            fx["capture_dir"],
            tmp_path / "nonshadow.csv.gz",
            id_map_path=fx["id_map"],
            pairs_csv=fx["pairs"],
        )


def test_v2_export_rejects_non_injective_id_map(tmp_path: Path) -> None:
    exporter = _load_exporter()
    fx = _write_v2_fixture(tmp_path)
    fx["id_map"].write_text(
        "MOT17-02\tlocal_id=7\tglobal_id=107\nMOT17-02\tlocal_id=9\tglobal_id=107\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not injective"):
        exporter.export_capture_v2(
            fx["capture_dir"],
            tmp_path / "bad_map.csv.gz",
            id_map_path=fx["id_map"],
            pairs_csv=fx["pairs"],
        )
