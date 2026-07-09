"""Unit tests for signal_tables schema helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saccade.perception.eval.signal_tables import (
    DEFAULT_RELINK_HARD_POOL_RULE,
    RELINK_PAIR_REQUIRED,
    CutDesign,
    MethodId,
    PipelineLayer,
    RelinkPairRow,
    SchemaError,
    StudyMeta,
    SweepAxis,
    SweepGridKind,
    SweepMode,
    UniverseId,
    apply_hard_pool_mask,
    auc_full_and_hard_pool,
    compare_accept_masks,
    cumulative_enabled_nodes,
    error_set_diff,
    expand_sweep_grid,
    load_study_meta,
    make_det_key,
    make_run_id,
    offline_threshold_curve,
    parse_simple_hard_pool_rule,
    pipeline_layer_of_method,
    save_study_meta,
    summarize_sweep_metric,
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


def test_expand_linspace_and_grid() -> None:
    axis = SweepAxis(
        name="score_floor",
        node_id="post_filter",
        kind=SweepGridKind.LINSPACE.value,
        lo=0.1,
        hi=0.5,
        num=5,
        offline=True,
        offline_universe=UniverseId.U_DET.value,
        offline_score_col="score",
    )
    vals = axis.expanded_values()
    assert len(vals) == 5
    assert vals[0] == pytest.approx(0.1)
    assert vals[-1] == pytest.approx(0.5)
    grid = expand_sweep_grid([axis])
    assert len(grid) == 5
    assert make_run_id("post_nms", grid[0]).startswith("post_nms__")


def test_offline_threshold_curve_and_summary() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    y = [True, True, False, False]
    curve = offline_threshold_curve(scores, y, [0.15, 0.5, 0.85])
    assert len(curve) == 3
    # high threshold keeps only strong TPs
    hi = curve[-1]
    assert hi["tp"] == 1
    assert hi["fp"] == 0
    rows = [{"score_floor": r["threshold"], "f1": r["f1"]} for r in curve]
    summary = summarize_sweep_metric(
        rows, axis_name="score_floor", metric="f1", higher_is_better=True
    )
    assert summary["n_points"] == 3
    assert "x_best" in summary


def test_sweep_meta_roundtrip(tmp_path: Path) -> None:
    axis = SweepAxis(
        name="match_thresh",
        node_id="track",
        kind=SweepGridKind.ARANGE.value,
        lo=0.4,
        hi=0.6,
        step=0.1,
        offline=False,
    )
    meta = StudyMeta(
        study_id="sweep1",
        created_utc="2026-07-09T00:00:00Z",
        commit="abc",
        preset="mamba_whole_graph_m",
        detector="SDP",
        double_buffer=True,
        cut_design=CutDesign.SWEEP.value,
        sweep_mode=SweepMode.ONLINE.value,
        sweep_base_method=MethodId.BARE_GMC.value,
        method_ids=[MethodId.BARE_GMC.value],
        sweep_axes=[axis.to_json_dict()],
    )
    save_study_meta(meta, tmp_path)
    loaded = load_study_meta(tmp_path)
    assert loaded.cut_design == CutDesign.SWEEP.value
    assert len(loaded.parsed_sweep_axes()) == 1
    assert len(loaded.parsed_sweep_axes()[0].expanded_values()) >= 2


def test_relink_pair_columns_and_gt_match_cast() -> None:
    validate_columns(UniverseId.U_RELINK_PAIR, RELINK_PAIR_REQUIRED)
    assert RelinkPairRow.gt_match_as_bool(1) is True
    assert RelinkPairRow.gt_match_as_bool(0) is False
    assert RelinkPairRow.gt_match_as_bool(True) is True


def test_relink_meta_requires_hard_pool(tmp_path: Path) -> None:
    meta = StudyMeta(
        study_id="b1_bad",
        created_utc="2026-07-09T00:00:00Z",
        commit="abc",
        preset="mamba_whole_graph",
        detector="SDP",
        double_buffer=True,
        universes=[UniverseId.U_RELINK_PAIR.value],
        hard_pool_rule="",  # missing
        study_line="B1",
    )
    save_study_meta(meta, tmp_path)
    with pytest.raises(SchemaError, match="hard_pool_rule"):
        load_study_meta(tmp_path)


def test_relink_meta_ok(tmp_path: Path) -> None:
    meta = StudyMeta(
        study_id="b1_ok",
        created_utc="2026-07-09T00:00:00Z",
        commit="abc",
        preset="mamba_whole_graph",
        detector="SDP",
        double_buffer=True,
        universes=[UniverseId.U_RELINK_PAIR.value],
        hard_pool_rule=DEFAULT_RELINK_HARD_POOL_RULE,
        study_line="B1",
        report_base_rate=True,
    )
    save_study_meta(meta, tmp_path)
    loaded = load_study_meta(tmp_path)
    assert loaded.uses_relink_pair()
    assert loaded.score_field_for(UniverseId.U_RELINK_PAIR) == "bridge_dist"
    assert "gt_match" in loaded.y_definition_for(UniverseId.U_RELINK_PAIR)


def test_auc_full_and_hard_pool() -> None:
    # positives: small distance; negatives: mix of near and far
    scores = [0.2, 0.3, 0.4, 2.0, 3.0, 4.0, 0.5, 5.0]
    y = [True, True, True, False, False, False, False, False]
    hard = apply_hard_pool_mask(scores, "bridge_dist<=1.0")
    assert hard == [True, True, True, False, False, False, True, False]
    col, op, thr = parse_simple_hard_pool_rule("bridge_dist<=1.0")
    assert col == "bridge_dist" and op == "<=" and thr == 1.0
    out = auc_full_and_hard_pool(scores, y, hard, lower_is_better=True, min_n=4)
    assert out["full"]["n_pos"] == 3
    assert out["full"]["n_neg"] == 5
    assert out["full"]["skipped_reason"] == ""
    assert out["full"]["auc"] > 0.5
    assert out["hard"]["n"] == 4  # three pos + one near neg
    assert out["citation_ok"] is True
