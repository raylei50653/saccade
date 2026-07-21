"""Tests for the near-miss offsets analyzer (scripts.eval.diagnostics.analyze_near_miss_offsets)."""

# scope: eval
# function: behavior
# lifecycle: active

from pathlib import Path

import pandas as pd
import pytest

from scripts.eval.diagnostics.analyze_near_miss_offsets import (
    BoxTransform,
    analyze_sequence,
    apply_box_transform,
    bbox_iou_xyxy,
    build_offset_row,
    classify_bucket,
    simulate_transform,
)


def test_classify_bucket() -> None:
    assert classify_bucket(0.0) == "true_miss"
    assert classify_bucket(0.49) == "near_miss"
    assert classify_bucket(0.5) == "threshold_sensitive"


def test_build_offset_row_centered_prediction_has_zero_offsets() -> None:
    gt = pd.Series(
        {"Id": 1, "X": 10.0, "Y": 20.0, "W": 40.0, "H": 80.0, "Visibility": 1.0}
    )
    pred = pd.Series(
        {"Id": 9, "X": 10.0, "Y": 20.0, "W": 40.0, "H": 80.0, "Score": 0.7}
    )

    row = build_offset_row(
        seq="MOT17-00-SDP",
        frame_id=1,
        gt_row=gt,
        pred_row=pred,
        best_iou=1.0,
        frame_w=100,
        frame_h=120,
        visibility_threshold=0.6,
        match_iou=0.5,
    )

    assert row["center_dx_norm"] == pytest.approx(0.0)
    assert row["center_dy_norm"] == pytest.approx(0.0)
    assert row["width_ratio"] == pytest.approx(1.0)
    assert row["height_ratio"] == pytest.approx(1.0)
    assert row["bucket"] == "threshold_sensitive"


def test_build_offset_row_shifted_prediction_has_expected_offsets() -> None:
    gt = pd.Series(
        {"Id": 1, "X": 10.0, "Y": 20.0, "W": 40.0, "H": 80.0, "Visibility": 1.0}
    )
    pred = pd.Series(
        {"Id": 9, "X": 20.0, "Y": 10.0, "W": 20.0, "H": 40.0, "Score": 0.7}
    )

    row = build_offset_row(
        seq="MOT17-00-SDP",
        frame_id=1,
        gt_row=gt,
        pred_row=pred,
        best_iou=0.25,
        frame_w=100,
        frame_h=120,
        visibility_threshold=0.6,
        match_iou=0.5,
    )

    assert row["center_dx_norm"] == pytest.approx(0.0)
    assert row["center_dy_norm"] == pytest.approx(-0.375)
    assert row["width_ratio"] == pytest.approx(0.5)
    assert row["height_ratio"] == pytest.approx(0.5)
    assert row["top_delta_norm"] == pytest.approx(-0.125)
    assert row["bottom_delta_norm"] == pytest.approx(-0.625)
    assert row["bucket"] == "near_miss"


def test_apply_box_transform_uniform_expansion_preserves_center() -> None:
    refined = apply_box_transform(
        (10.0, 20.0, 40.0, 80.0),
        BoxTransform(
            name="uniform",
            mode="uniform_expand",
            width_scale=0.1,
            top_scale=0.1,
            bottom_scale=0.1,
            max_area_growth=2.0,
        ),
        frame_w=200,
        frame_h=200,
    )

    assert refined == pytest.approx((6.0, 12.0, 54.0, 108.0))
    assert ((refined[0] + refined[2]) * 0.5) == pytest.approx(30.0)
    assert ((refined[1] + refined[3]) * 0.5) == pytest.approx(60.0)


def test_apply_box_transform_bottom_expansion_changes_only_y2() -> None:
    refined = apply_box_transform(
        (10.0, 20.0, 40.0, 80.0),
        BoxTransform(
            name="bottom",
            mode="bottom_expand",
            bottom_scale=0.1,
            max_area_growth=2.0,
        ),
        frame_w=200,
        frame_h=200,
    )

    assert refined == pytest.approx((10.0, 20.0, 50.0, 108.0))


def test_apply_box_transform_clips_and_rejects_area_growth() -> None:
    clipped = apply_box_transform(
        (0.0, 0.0, 20.0, 20.0),
        BoxTransform(
            name="clip",
            mode="uniform_expand",
            width_scale=0.5,
            top_scale=0.5,
            bottom_scale=0.5,
            max_area_growth=4.0,
        ),
        frame_w=25,
        frame_h=25,
    )
    rejected = apply_box_transform(
        (0.0, 0.0, 20.0, 20.0),
        BoxTransform(
            name="reject",
            mode="uniform_expand",
            width_scale=0.5,
            top_scale=0.5,
            bottom_scale=0.5,
            max_area_growth=1.1,
        ),
        frame_w=100,
        frame_h=100,
    )

    assert clipped == pytest.approx((0.0, 0.0, 25.0, 25.0))
    assert rejected is None


def test_simulate_transform_reports_recovered_near_miss() -> None:
    rows = pd.DataFrame(
        [
            {
                "bucket": "near_miss",
                "best_iou": bbox_iou_xyxy((0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 9.0, 9.0)),
                "pred_track_id": 1,
                "pred_x": 1.0,
                "pred_y": 1.0,
                "pred_w": 8.0,
                "pred_h": 8.0,
                "gt_x": 0.0,
                "gt_y": 0.0,
                "gt_w": 10.0,
                "gt_h": 10.0,
            }
        ]
    )

    summary = simulate_transform(
        rows,
        BoxTransform(
            name="recover",
            mode="uniform_expand",
            width_scale=0.2,
            top_scale=0.2,
            bottom_scale=0.2,
            max_area_growth=2.0,
        ),
    )

    assert summary["near_miss_recovered_count"] == 1
    assert summary["near_miss_recovered_share"] == pytest.approx(1.0)


def test_analyze_sequence_synthetic_mot_buckets(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    results = tmp_path / "results"
    seq_dir = gt_root / "MOT17-00-SDP"
    (seq_dir / "gt").mkdir(parents=True)
    results.mkdir()
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nimWidth=100\nimHeight=100\n", encoding="utf-8"
    )
    (seq_dir / "gt" / "gt.txt").write_text(
        "\n".join(
            [
                "1,1,0,0,10,10,1,1,1",
                "2,1,0,0,10,10,1,1,1",
                "3,1,0,0,10,10,1,1,1",
                "4,1,0,0,10,10,1,1,1",
                "4,2,0,0,10,10,1,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (results / "MOT17-00-SDP.txt").write_text(
        "\n".join(
            [
                "1,1,0,0,10,10,0.9,-1,-1,-1",
                "3,3,7,0,10,10,0.8,-1,-1,-1",
                "4,4,0,0,10,10,0.7,-1,-1,-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = analyze_sequence(
        seq="MOT17-00-SDP",
        results_folder=results,
        gt_root=gt_root,
        visibility_threshold=0.6,
        match_iou=0.5,
    )

    assert rows["bucket"].value_counts().to_dict() == {
        "true_miss": 1,
        "near_miss": 1,
        "threshold_sensitive": 1,
    }
