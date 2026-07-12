"""R1 capture-contract tests: nested windows, fail-closed export, label-free replay."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from saccade.perception.eval.consumer_a_bridge_fidelity import (
    ANCHOR_MODE,
    consumer_a_estimate_from_rings,
)


REPO = Path(__file__).resolve().parents[3]
EXPORTER = REPO / "scripts/tools/export_r1_temporal_reduction_capture.py"
VERIFIER = REPO / "scripts/tools/verify_r1_temporal_reduction_replay.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_event(*, short_lost: bool = False) -> dict[str, object]:
    lost = [
        (90.0, 190.0, 80.0),
        (94.0, 193.0, 80.0),
        (98.0, 196.0, 80.0),
        (100.0, 200.0, 80.0),
    ]
    candidate = [
        (110.0, 200.0, 82.0),
        (114.0, 203.0, 82.0),
        (118.0, 206.0, 82.0),
        (122.0, 209.0, 82.0),
    ]
    effective_lost = lost[-1:] if short_lost else lost
    estimate = consumer_a_estimate_from_rings(
        effective_lost,
        candidate,
        gap=4,
        bridge_at=4,
        ema_lost=80.0,
        ema_cand=82.0,
        anchor_mode=ANCHOR_MODE["adaptive"],
        rate_gate=0.03,
        bridge_dir_bonus=0.0,
    )
    return {
        "seq": "MOT17-02",
        "frame": 12,
        "lost_id": 7,
        "cand_id": 9,
        "lost_slot": 3,
        "cand_slot": 4,
        "capture_mode": "runtime_cuda_event_ring",
        "evidence_role": "runtime_cuda_observation",
        "gap": estimate.gap,
        "bridge_at": estimate.bridge_at,
        "la": estimate.la,
        "anchor_mode": ANCHOR_MODE["adaptive"],
        "anchor_rate": 0.03,
        "bdist": estimate.bdist,
        "dist_h": estimate.dist_h,
        "fwd_r": estimate.fwd_r,
        "bwd_r": estimate.bwd_r,
        "v_lost_x": estimate.v_lost_x,
        "v_lost_y": estimate.v_lost_y,
        "v_cand_x": estimate.v_cand_x,
        "v_cand_y": estimate.v_cand_y,
        "ax": estimate.ax,
        "ay": estimate.ay,
        "cx0": estimate.cx0,
        "cy0": estimate.cy0,
        "ema_lost": estimate.ema_lost,
        "ema_cand": estimate.ema_cand,
        "h_ref": estimate.h_ref,
        "s_lost": estimate.s_lost,
        "w": estimate.w,
        "production_threshold": 0.4,
        "bridge_dir_bonus": 0.0,
        "lost_window_size": 1 if short_lost else 4,
        "cand_window_size": 4,
        "lost_anchor_window": [list(sample) for sample in effective_lost]
        + [[0.0, 0.0, 0.0]] * (4 - len(effective_lost)),
        "cand_anchor_window": [list(sample) for sample in candidate],
    }


def _capture_fixture(tmp_path: Path, *, short_lost: bool = False) -> tuple[Path, Path]:
    capture_dir = tmp_path / "native"
    capture_dir.mkdir()
    event = _native_event(short_lost=short_lost)
    payload = {
        "events": [event],
        "total_events": 1,
        "overflow_events": 0,
        "complete": True,
        "provenance": {
            "capture_contract": "r1_temporal_reduction_capture_v1",
            "shadow": True,
            "git_commit": "test",
            "bridge": {
                "px": 0.4,
                "at": 4,
                "anchor": "adaptive",
                "anchor_rate": 0.03,
                "dir_bonus": 0.0,
            },
            "detector": {"tiling": "native_640"},
        },
    }
    (capture_dir / "MOT17-02.json").write_text(json.dumps(payload), encoding="utf-8")
    id_map = tmp_path / "_global_id_map.txt"
    id_map.write_text(
        "MOT17-02\tlocal_id=7\tglobal_id=107\nMOT17-02\tlocal_id=9\tglobal_id=109\n",
        encoding="utf-8",
    )
    return capture_dir, id_map


def test_r1_export_preserves_effective_windows_and_global_event_key(
    tmp_path: Path,
) -> None:
    exporter = _load(EXPORTER, "r1_export")
    capture_dir, id_map = _capture_fixture(tmp_path)
    output = tmp_path / "events.jsonl"

    manifest = exporter.export_r1_capture(capture_dir, output, id_map_path=id_map)

    assert manifest["capture_contract"] == "r1_temporal_reduction_capture_v1"
    record = json.loads(output.read_text().strip())
    assert record["event_key"] == "MOT17-02|capture_ordinal=0|lost_slot=3|cand_slot=4"
    assert record["lost_global_id"] == 107
    assert record["cand_global_id"] == 109
    assert record["lost_reduction"]["branch"] == "bridge_anchor4_last4"
    assert record["lost_reduction"]["chronological_cx_cy_h"][0] == [90.0, 190.0, 80.0]
    assert record["candidate_reduction"]["chronological_cx_cy_h"][3] == [
        122.0,
        209.0,
        82.0,
    ]
    assert "lost_last_frame" not in record
    assert "cand_first_frame" not in record


def test_r1_verifier_replays_without_labels_or_score_fit(tmp_path: Path) -> None:
    exporter = _load(EXPORTER, "r1_export_replay")
    verifier = _load(VERIFIER, "r1_verify")
    capture_dir, id_map = _capture_fixture(tmp_path)
    output = tmp_path / "events.jsonl"
    exporter.export_r1_capture(capture_dir, output, id_map_path=id_map)

    result = verifier.verify(output)

    assert result["all_terms_within_abs_tolerance"] is True
    assert result["predicate_disagreements"] == 0
    assert result["outcome_labels_read"] is False
    assert result["score_fit_performed"] is False
    assert result["causal_sensitivity_interpretable"] is True
    assert {row["mutation"] for row in result["causal_sensitivity"]} == {
        "cyclic_window_shift",
        "omit_oldest_candidate_sample",
        "omit_oldest_lost_sample",
    }


def test_r1_verifier_counts_replay_tie_as_order_disagreement_regardless_of_order(
    tmp_path: Path,
) -> None:
    exporter = _load(EXPORTER, "r1_export_order")
    verifier = _load(VERIFIER, "r1_verify_order")
    capture_dir, id_map = _capture_fixture(tmp_path)
    output = tmp_path / "events.jsonl"
    exporter.export_r1_capture(capture_dir, output, id_map_path=id_map)
    first = json.loads(output.read_text())
    second = copy.deepcopy(first)
    second["event_key"] = "MOT17-02|capture_ordinal=1|lost_slot=4|cand_slot=4"
    second["native_capture_ordinal"] = 1
    second["lost_slot"] = 4
    second["kernel_terms"]["bdist"] = float(first["kernel_terms"]["bdist"]) + 0.1
    output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in (first, second)) + "\n",
        encoding="utf-8",
    )

    forward = verifier.verify(output)
    output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in (second, first)) + "\n",
        encoding="utf-8",
    )
    reverse = verifier.verify(output)

    assert forward["event_local_order"]["comparable_pairs"] == 1
    assert forward["event_local_order"]["order_disagreements"] == 1
    assert reverse["event_local_order"]["order_disagreements"] == 1


def test_r1_sensitivity_uses_untouched_host_replay_and_is_uninterpretable_on_r0_failure(
    tmp_path: Path,
) -> None:
    exporter = _load(EXPORTER, "r1_export_sensitivity")
    verifier = _load(VERIFIER, "r1_verify_sensitivity")
    capture_dir, id_map = _capture_fixture(tmp_path)
    output = tmp_path / "events.jsonl"
    exporter.export_r1_capture(capture_dir, output, id_map_path=id_map)
    row = json.loads(output.read_text())
    baseline = verifier.replay_record(row)
    lost = verifier._ring(row["lost_reduction"], name="lost")
    candidate = verifier._ring(row["candidate_reduction"], name="candidate")
    shifted = verifier._estimate_values(
        row, lost=lost[1:] + lost[:1], candidate=candidate[1:] + candidate[:1]
    )
    expected = abs(float(shifted["bdist"]) - float(baseline["bdist"]))
    row["kernel_terms"]["bdist"] = float(baseline["bdist"]) + 0.2
    output.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    result = verifier.verify(output)
    cyclic = next(
        item
        for item in result["causal_sensitivity"]
        if item["mutation"] == "cyclic_window_shift"
    )

    assert cyclic["max_abs_mutation_delta"]["bdist"] == pytest.approx(expected)
    assert result["causal_sensitivity_interpretable"] is False


def test_r1_verifier_rejects_post_seal_tolerance_override(tmp_path: Path) -> None:
    exporter = _load(EXPORTER, "r1_export_tolerance")
    verifier = _load(VERIFIER, "r1_verify_tolerance")
    capture_dir, id_map = _capture_fixture(tmp_path)
    output = tmp_path / "events.jsonl"
    exporter.export_r1_capture(capture_dir, output, id_map_path=id_map)

    with pytest.raises(ValueError, match="tolerance is sealed"):
        verifier.verify(output, abs_tolerance=1e-3)


def test_r1_export_and_replay_preserve_short_lost_fallback(tmp_path: Path) -> None:
    exporter = _load(EXPORTER, "r1_export_short_lost")
    verifier = _load(VERIFIER, "r1_verify_short_lost")
    capture_dir, id_map = _capture_fixture(tmp_path, short_lost=True)
    output = tmp_path / "short-lost.jsonl"
    exporter.export_r1_capture(capture_dir, output, id_map_path=id_map)

    record = json.loads(output.read_text().strip())
    assert record["lost_reduction"]["branch"] == "short_lost_last_point_zero_velocity"
    assert record["lost_reduction"]["consumed_samples"] == 1
    assert verifier.verify(output)["all_terms_within_abs_tolerance"] is True


def test_r1_export_rejects_d0_provenance_but_preserves_unemitted_native_state(
    tmp_path: Path,
) -> None:
    exporter = _load(EXPORTER, "r1_export_failure")
    capture_dir, id_map = _capture_fixture(tmp_path)
    payload_path = capture_dir / "MOT17-02.json"
    payload = json.loads(payload_path.read_text())
    payload["provenance"]["capture_contract"] = "d0_runtime_cuda_v1"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not an R1 capture contract"):
        exporter.export_r1_capture(
            capture_dir, tmp_path / "bad.jsonl", id_map_path=id_map
        )

    payload["provenance"]["capture_contract"] = "r1_temporal_reduction_capture_v1"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    id_map.write_text("MOT17-02\tlocal_id=7\tglobal_id=107\n", encoding="utf-8")
    output = tmp_path / "unemitted.jsonl"
    manifest = exporter.export_r1_capture(capture_dir, output, id_map_path=id_map)
    assert manifest["events_without_global_id"] == 1
    assert json.loads(output.read_text())["cand_global_id"] is None


def test_r1_export_rejects_all_reduction_configuration_mismatches(
    tmp_path: Path,
) -> None:
    exporter = _load(EXPORTER, "r1_export_provenance")
    capture_dir, id_map = _capture_fixture(tmp_path)
    payload_path = capture_dir / "MOT17-02.json"
    payload = json.loads(payload_path.read_text())
    checks = (
        ("at", 5, "bridge_at"),
        ("anchor", "foot", "anchor_mode"),
        ("anchor_rate", 0.04, "anchor_rate"),
        ("px", 0.5, "production_threshold"),
        ("dir_bonus", 0.1, "bridge_dir_bonus"),
    )
    for field, value, expected in checks:
        mutated = copy.deepcopy(payload)
        mutated["provenance"]["bridge"][field] = value
        payload_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            exporter.export_r1_capture(
                capture_dir, tmp_path / f"bad-{field}.jsonl", id_map_path=id_map
            )
