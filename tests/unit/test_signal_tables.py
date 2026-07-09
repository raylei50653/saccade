"""Unit tests for signal_tables schema helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saccade.perception.eval.signal_tables import (
    CutDesign,
    MethodId,
    PipelineLayer,
    SchemaError,
    StudyMeta,
    UniverseId,
    compare_accept_masks,
    cumulative_enabled_nodes,
    error_set_diff,
    load_study_meta,
    make_det_key,
    pipeline_layer_of_method,
    save_study_meta,
    validate_columns,
    validate_method_order,
)


def test_make_det_key() -> None:
    assert make_det_key("MOT17-04-SDP", 3, "post_nms", 1) == (
        "MOT17-04-SDP:3:post_nms:1"
    )


def test_compare_accept_masks_with_labels() -> None:
    a = [True, True, False, False]
    b = [True, False, True, False]
    y = [True, True, False, False]
    r = compare_accept_masks(a, b, y)
    assert r.both == 1
    assert r.only_a == 1
    assert r.only_b == 1
    assert r.jaccard == pytest.approx(1 / 3)
    assert r.tp_both == 1
    assert r.tp_only_a == 1
    assert r.tp_only_b == 0
    assert r.fp_only_b == 1


def test_error_set_diff() -> None:
    only_a, only_b, both = error_set_diff(["e1", "e2"], ["e2", "e3"])
    assert only_a == {"e1"}
    assert only_b == {"e3"}
    assert both == {"e2"}


def test_validate_columns_missing() -> None:
    with pytest.raises(SchemaError, match="missing"):
        validate_columns(UniverseId.U_DET, ["seq", "frame"])


def test_study_meta_roundtrip(tmp_path: Path) -> None:
    meta = StudyMeta(
        study_id="t1",
        created_utc="2026-07-09T00:00:00Z",
        commit="abc1234",
        preset="mamba_whole_graph_m",
        detector="SDP",
        double_buffer=True,
        universes=[UniverseId.U_DET.value],
        method_ids=[MethodId.POST_NMS.value],
        cut_design=CutDesign.SINGLE.value,
    )
    save_study_meta(meta, tmp_path)
    loaded = load_study_meta(tmp_path)
    assert loaded.study_id == "t1"
    assert loaded.iou_match == 0.5
    assert loaded.score_field_for(UniverseId.U_DET) == "score"
    assert loaded.cut_design == CutDesign.SINGLE.value
    raw = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert raw["preset"] == "mamba_whole_graph_m"


def test_pipeline_method_order_ok() -> None:
    validate_method_order(
        [
            MethodId.RAW.value,
            MethodId.POST_NMS.value,
            MethodId.BARE_GMC.value,
            MethodId.BARE_BRIDGE.value,
            MethodId.FULL_PRESET.value,
        ]
    )


def test_pipeline_method_order_rejects_regression() -> None:
    with pytest.raises(ValueError, match="not in pipeline order"):
        validate_method_order([MethodId.FULL_PRESET.value, MethodId.RAW.value])


def test_pipeline_layer_and_cumulative() -> None:
    assert pipeline_layer_of_method(MethodId.POST_NMS) is PipelineLayer.L2_POST
    assert pipeline_layer_of_method(MethodId.BARE_BRIDGE) is PipelineLayer.L5_IDENTITY
    enabled = cumulative_enabled_nodes("bridge_relink")
    assert "detect" in enabled
    assert "track" in enabled
    assert "bridge_relink" in enabled
    assert "interpolate" not in enabled


def test_cumulative_meta_loads(tmp_path: Path) -> None:
    meta = StudyMeta(
        study_id="cum",
        created_utc="2026-07-09T00:00:00Z",
        commit="abc",
        preset="mamba_whole_graph_m",
        detector="SDP",
        double_buffer=True,
        cut_design=CutDesign.CUMULATIVE.value,
        method_ids=[
            MethodId.RAW.value,
            MethodId.POST_NMS.value,
            MethodId.BARE_GMC.value,
        ],
    )
    save_study_meta(meta, tmp_path)
    loaded = load_study_meta(tmp_path)
    assert loaded.resolved_terminal(MethodId.BARE_GMC.value) == "track"


def test_cumulative_meta_rejects_bad_order(tmp_path: Path) -> None:
    meta = StudyMeta(
        study_id="bad",
        created_utc="2026-07-09T00:00:00Z",
        commit="abc",
        preset="mamba_whole_graph_m",
        detector="SDP",
        double_buffer=True,
        cut_design=CutDesign.CUMULATIVE.value,
        method_ids=[MethodId.FULL_PRESET.value, MethodId.RAW.value],
    )
    save_study_meta(meta, tmp_path)
    with pytest.raises(SchemaError, match="pipeline order"):
        load_study_meta(tmp_path)
