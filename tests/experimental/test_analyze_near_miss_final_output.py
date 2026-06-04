from __future__ import annotations

import pandas as pd

from scripts.eval.analyze_near_miss_final_output import (
    analyze_final_output,
    classify_final_output,
    summarize,
)


def _attr_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "seq": "MOT17-02-SDP",
        "frame": 10,
        "gt_id": 1,
        "is_high_vis": True,
        "gt_x": 100.0,
        "gt_y": 100.0,
        "gt_w": 40.0,
        "gt_h": 100.0,
        "post_merge_iou": 0.8,
        "post_merge_score": 0.9,
        "post_merge_x1": 100.0,
        "post_merge_y1": 100.0,
        "post_merge_x2": 140.0,
        "post_merge_y2": 200.0,
        "stage_attribution": "stage_good_final_lost",
    }
    row.update(overrides)
    return row


def _mot_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "seq": "MOT17-02-SDP",
        "frame": 10,
        "track_id": 7,
        "x": 100.0,
        "y": 100.0,
        "w": 40.0,
        "h": 100.0,
        "score": 0.8,
    }
    row.update(overrides)
    return row


def test_analyze_final_output_marks_preserved_gt_match() -> None:
    rows = analyze_final_output(
        pd.DataFrame([_attr_row()]),
        pd.DataFrame([_mot_row()]),
    )

    assert rows.loc[0, "final_output_attribution"] == "final_preserved_gt_match"
    assert rows.loc[0, "final_gt_track_id"] == 7


def test_analyze_final_output_marks_candidate_absent() -> None:
    rows = analyze_final_output(
        pd.DataFrame([_attr_row()]),
        pd.DataFrame([_mot_row(x=400.0, y=400.0)]),
    )

    assert rows.loc[0, "final_output_attribution"] == "final_candidate_absent"


def test_analyze_final_output_marks_near_miss() -> None:
    rows = analyze_final_output(
        pd.DataFrame([_attr_row()]),
        pd.DataFrame([_mot_row(x=130.0, y=100.0)]),
    )

    assert rows.loc[0, "final_output_attribution"] == "final_near_miss"
    assert 0.1 <= rows.loc[0, "final_gt_iou"] < 0.5


def test_classify_final_output_marks_stage_not_good_first() -> None:
    row = pd.Series(
        {
            "stage_iou": 0.4,
            "final_gt_iou": 0.7,
            "final_frame_outputs": 1,
            "final_stage_iou": 0.8,
        }
    )

    assert (
        classify_final_output(row, good_iou=0.5, near_iou=0.1, same_box_iou=0.95)
        == "stage_not_good"
    )


def test_summarize_counts_stage_good_subset() -> None:
    rows = pd.DataFrame(
        [
            {
                "stage_iou": 0.6,
                "final_gt_iou": 0.0,
                "final_stage_iou": 0.0,
                "final_output_attribution": "final_candidate_absent",
            },
            {
                "stage_iou": 0.4,
                "final_gt_iou": 0.7,
                "final_stage_iou": 0.8,
                "final_output_attribution": "stage_not_good",
            },
        ]
    )

    summary = summarize(rows, good_iou=0.5, stage="post_merge")

    assert summary["total"] == 2
    assert summary["stage_good_total"] == 1
    assert summary["stage_good_final_output_counts"] == {"final_candidate_absent": 1}
