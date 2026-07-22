"""Synthetic tests for the external-FP row apparatus: CrowdHuman record loading,
ignore-region handling, TP/ignore/FP row labelling, and structural feature rows."""

# scope: eval
# function: behavior
# lifecycle: quarantined
# lifecycle-note: external-FP study line inactive; DISPOSITION.md proposes T4 (delete)
#   — row-extraction apparatus for the concluded study, preserved by recipe/git.

from pathlib import Path

from saccade.perception.eval.external_fp_rows import (
    DetectionRowLabel,
    ExternalGroundTruthBox,
    build_structural_row,
    count_false_negatives,
    label_detection_rows,
    load_crowdhuman_records,
)


def test_load_crowdhuman_records_marks_ignore_boxes(tmp_path: Path) -> None:
    ann_path = tmp_path / "ann.odgt"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    ann_path.write_text(
        "\n".join(
            [
                '{"ID":"img001","gtboxes":[{"fbox":[10,20,30,40],"tag":"person"},{"fbox":[0,0,5,5],"tag":"mask","extra":{"ignore":1}}]}',
                '{"ID":"img002","gtboxes":[{"fbox":[1,2,3,4],"tag":"person","extra":{"ignore":1}}]}',
                '{"ID":"img003","gtboxes":[{"fbox":[1,2,3,4],"tag":"person","head_attr":{"ignore":1}}]}',
            ]
        ),
        encoding="utf-8",
    )

    records = load_crowdhuman_records(ann_path, image_dir)

    assert [record.image_id for record in records] == ["img001", "img002", "img003"]
    assert records[0].image_path == image_dir / "img001.jpg"
    assert records[0].gt_boxes[0].ignore is False
    assert records[0].gt_boxes[0].bbox == (10.0, 20.0, 40.0, 60.0)
    assert records[0].gt_boxes[1].ignore is True
    assert records[1].gt_boxes[0].ignore is True
    assert records[2].gt_boxes[0].ignore is True


def test_label_detection_rows_prefers_tp_then_ignore_then_fp() -> None:
    predictions = [
        {"bbox": [0, 0, 10, 10], "score": 0.95, "class_id": 0},
        {"bbox": [30, 30, 40, 40], "score": 0.80, "class_id": 0},
        {"bbox": [50, 50, 60, 60], "score": 0.60, "class_id": 0},
    ]
    gt_boxes = [
        ExternalGroundTruthBox(bbox=(0.0, 0.0, 10.0, 10.0), ignore=False),
        ExternalGroundTruthBox(bbox=(30.0, 30.0, 40.0, 40.0), ignore=True),
    ]

    labels = label_detection_rows(predictions, gt_boxes, match_iou=0.5, ignore_iou=0.5)

    assert [label.label for label in labels] == ["tp", "ignore", "fp"]
    assert labels[0].matched_iou == 1.0
    assert labels[1].matched_iou == 1.0
    assert labels[2].matched_iou == 0.0


def test_count_false_negatives_ignores_ignore_regions() -> None:
    predictions = [
        {"bbox": [0, 0, 10, 10], "score": 0.95, "class_id": 0},
    ]
    gt_boxes = [
        ExternalGroundTruthBox(bbox=(0.0, 0.0, 10.0, 10.0), ignore=False),
        ExternalGroundTruthBox(bbox=(20.0, 20.0, 30.0, 30.0), ignore=False),
        ExternalGroundTruthBox(bbox=(40.0, 40.0, 50.0, 50.0), ignore=True),
    ]

    assert count_false_negatives(predictions, gt_boxes, match_iou=0.5) == 1


def test_build_structural_row_computes_expected_features() -> None:
    row = build_structural_row(
        dataset="crowdhuman",
        split="val",
        image_id="img001",
        image_path="/tmp/img001.jpg",
        image_width=200,
        image_height=100,
        prediction={"bbox": [0, 10, 20, 50], "score": 0.75, "class_id": 0},
        label=DetectionRowLabel(label="fp", matched_iou=0.25),
        edge_touch_px=2.0,
    )

    assert row["label"] == "fp"
    assert row["width"] == 20.0
    assert row["height"] == 40.0
    assert row["area"] == 800.0
    assert row["aspect_ratio"] == 2.0
    assert row["center_x_norm"] == 0.05
    assert row["center_y_norm"] == 0.30
    assert row["edge_margin_norm"] == 0.0
    assert row["touches_edge"] == 1
