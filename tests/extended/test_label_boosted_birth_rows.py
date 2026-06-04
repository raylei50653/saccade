from __future__ import annotations

import pandas as pd

from scripts.eval.label_boosted_birth_rows import (
    build_output_label_map,
    label_boosted_rows,
    summarize,
)
from saccade.perception.eval.external_fp_rows import ExternalGroundTruthBox


def test_build_output_label_map_marks_tp_fp_and_ignore() -> None:
    mot_rows = pd.DataFrame(
        [
            {
                "seq": "MOT17-02-SDP",
                "frame": 1,
                "track_id": 11,
                "x": 0.0,
                "y": 0.0,
                "w": 10.0,
                "h": 10.0,
                "score": 0.9,
            },
            {
                "seq": "MOT17-02-SDP",
                "frame": 1,
                "track_id": 12,
                "x": 20.0,
                "y": 20.0,
                "w": 10.0,
                "h": 10.0,
                "score": 0.8,
            },
            {
                "seq": "MOT17-02-SDP",
                "frame": 1,
                "track_id": 13,
                "x": 40.0,
                "y": 40.0,
                "w": 10.0,
                "h": 10.0,
                "score": 0.7,
            },
        ]
    )
    gt_by_frame = {
        ("MOT17-02-SDP", 1): [
            ExternalGroundTruthBox(bbox=(0.0, 0.0, 10.0, 10.0), ignore=False),
            ExternalGroundTruthBox(bbox=(40.0, 40.0, 50.0, 50.0), ignore=True),
        ]
    }

    label_map = build_output_label_map(
        mot_rows,
        gt_by_frame,
        match_iou=0.5,
        ignore_iou=0.5,
    )

    assert label_map[("MOT17-02-SDP", 1, 11)].label == "tp"
    assert label_map[("MOT17-02-SDP", 1, 12)].label == "fp"
    assert label_map[("MOT17-02-SDP", 1, 13)].label == "ignore"


def test_label_boosted_rows_marks_dropped_when_no_output() -> None:
    boosted_rows = pd.DataFrame(
        [
            {
                "seq": "MOT17-02-SDP",
                "frame": 1,
                "policy": "multi_birth",
                "output_emitted": False,
                "output_track_id": -1,
            }
        ]
    )

    labeled = label_boosted_rows(boosted_rows, {})

    assert labeled.loc[0, "final_label"] == "dropped"
    assert labeled.loc[0, "final_matched_iou"] == 0.0


def test_summarize_groups_by_policy() -> None:
    rows = pd.DataFrame(
        [
            {"policy": "multi_birth", "final_label": "tp"},
            {"policy": "multi_birth", "final_label": "fp"},
            {"policy": "birth_consecutive_gate", "final_label": "dropped"},
        ]
    )

    summary = summarize(rows)

    assert summary["total"] == 3
    assert summary["final_label_counts"] == {"tp": 1, "fp": 1, "dropped": 1}
    assert summary["policy_counts"]["multi_birth"] == {"tp": 1, "fp": 1}
