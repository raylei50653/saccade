import pandas as pd

from scripts.eval.diagnostics.analyze_near_miss_stage_attribution import (
    attribute_rows,
    best_stage_match,
    classify_stage_attribution,
    summarize_attribution,
)


def _near_row(best_iou: float = 0.2) -> dict[str, object]:
    return {
        "seq": "MOT17-00-SDP",
        "frame": 1,
        "gt_id": 7,
        "bucket": "near_miss",
        "vis": 1.0,
        "is_high_vis": True,
        "gt_x": 0.0,
        "gt_y": 0.0,
        "gt_w": 10.0,
        "gt_h": 10.0,
        "best_iou": best_iou,
        "pred_track_id": 3,
        "pred_score": 0.8,
    }


def _stage_row(stage: str, x: float, y: float, w: float, h: float) -> dict[str, object]:
    return {
        "seq": "MOT17-00-SDP",
        "frame": 1,
        "stage": stage,
        "det_idx": 0,
        "x1": x,
        "y1": y,
        "x2": x + w,
        "y2": y + h,
        "w": w,
        "h": h,
        "score": 0.9,
        "cls": 0,
    }


def test_best_stage_match_selects_highest_iou_box() -> None:
    gt = pd.Series(_near_row())
    stage_rows = pd.DataFrame(
        [
            _stage_row("raw", 20.0, 20.0, 10.0, 10.0),
            _stage_row("raw", 0.0, 0.0, 10.0, 10.0),
        ]
    )

    match = best_stage_match(gt, stage_rows)

    assert match["iou"] == 1.0
    assert match["det_idx"] == 0


def test_attribute_rows_classifies_good_raw_box_lost_after_post_filter() -> None:
    near = pd.DataFrame([_near_row(best_iou=0.2)])
    dump = pd.DataFrame(
        [
            _stage_row("raw", 0.0, 0.0, 10.0, 10.0),
            _stage_row("post_filter", 7.0, 0.0, 10.0, 10.0),
            _stage_row("post_nms", 7.0, 0.0, 10.0, 10.0),
            _stage_row("post_merge", 7.0, 0.0, 10.0, 10.0),
        ]
    )

    rows = attribute_rows(near, dump)

    assert rows.loc[0, "raw_iou"] == 1.0
    assert rows.loc[0, "post_filter_iou"] < 0.5
    assert rows.loc[0, "stage_attribution"] == "lost_after_raw"


def test_attribute_rows_classifies_tracker_degraded_when_stage_remains_better() -> None:
    near = pd.DataFrame([_near_row(best_iou=0.2)])
    dump = pd.DataFrame(
        [
            _stage_row("raw", 7.0, 0.0, 10.0, 10.0),
            _stage_row("post_filter", 7.0, 0.0, 10.0, 10.0),
            _stage_row("post_nms", 7.0, 0.0, 10.0, 10.0),
            _stage_row("post_merge", 4.0, 0.0, 10.0, 10.0),
        ]
    )

    rows = attribute_rows(near, dump)

    assert rows.loc[0, "post_merge_iou"] > rows.loc[0, "final_best_iou"]
    assert rows.loc[0, "stage_attribution"] == "tracker_degraded"


def test_classify_stage_attribution_handles_no_box() -> None:
    row = pd.Series(
        {
            "raw_iou": 0.0,
            "post_filter_iou": 0.0,
            "post_nms_iou": 0.0,
            "post_merge_iou": 0.0,
            "final_best_iou": 0.0,
        }
    )

    assert (
        classify_stage_attribution(
            row, ("raw", "post_filter", "post_nms", "post_merge"), 0.5
        )
        == "raw_no_box"
    )


def test_summarize_attribution_counts_stage_good() -> None:
    near = pd.DataFrame([_near_row(best_iou=0.2)])
    dump = pd.DataFrame(
        [
            _stage_row("raw", 0.0, 0.0, 10.0, 10.0),
            _stage_row("post_filter", 7.0, 0.0, 10.0, 10.0),
        ]
    )
    rows = attribute_rows(near, dump, stages=("raw", "post_filter"))

    summary = summarize_attribution(rows, stages=("raw", "post_filter"), good_iou=0.5)

    assert summary["total"] == 1
    assert summary["stage_good_counts"]["raw"] == 1
    assert summary["stage_good_counts"]["post_filter"] == 0


def test_attribute_rows_limits_to_stage_dump_frames() -> None:
    near = pd.DataFrame(
        [
            _near_row(best_iou=0.2),
            {**_near_row(best_iou=0.2), "frame": 2},
        ]
    )
    dump = pd.DataFrame([_stage_row("raw", 0.0, 0.0, 10.0, 10.0)])

    rows = attribute_rows(near, dump, stages=("raw",))

    assert rows.shape[0] == 1
    assert int(rows.loc[0, "frame"]) == 1
