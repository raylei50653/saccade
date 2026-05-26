from saccade.perception.eval.metrics import (
    _format_overall_metrics_from_counts,
    compute_detection_mean_ap,
)


def test_detection_map_perfect_single_class() -> None:
    ground_truths = {
        "img1": [{"bbox": [0, 0, 10, 10], "class_id": 0}],
        "img2": [{"bbox": [20, 20, 40, 40], "class_id": 0}],
    }
    predictions = {
        "img1": [{"bbox": [0, 0, 10, 10], "score": 0.95, "class_id": 0}],
        "img2": [{"bbox": [20, 20, 40, 40], "score": 0.90, "class_id": 0}],
    }

    metrics = compute_detection_mean_ap(
        ground_truths, predictions, iou_thresholds=(0.5,)
    )

    assert metrics["classes"] == [0]
    assert metrics["thresholds"][0.5] == 1.0
    assert metrics["mAP"] == 1.0
    assert metrics["per_class_ap"][0.5][0] == 1.0


def test_detection_map_false_positive_ranked_first_reduces_ap() -> None:
    ground_truths = {
        "img1": [{"bbox": [0, 0, 10, 10], "class_id": 0}],
    }
    predictions = {
        "img1": [
            {"bbox": [50, 50, 60, 60], "score": 0.99, "class_id": 0},
            {"bbox": [0, 0, 10, 10], "score": 0.80, "class_id": 0},
        ],
    }

    metrics = compute_detection_mean_ap(
        ground_truths, predictions, iou_thresholds=(0.5,)
    )

    assert metrics["thresholds"][0.5] == 0.5
    assert metrics["mAP"] == 0.5


def test_detection_map_averages_across_classes() -> None:
    ground_truths = {
        "img1": [
            {"bbox": [0, 0, 10, 10], "class_id": 0},
            {"bbox": [20, 20, 30, 30], "class_id": 1},
        ],
    }
    predictions = {
        "img1": [
            {"bbox": [0, 0, 10, 10], "score": 0.95, "class_id": 0},
            {"bbox": [100, 100, 120, 120], "score": 0.90, "class_id": 1},
        ],
    }

    metrics = compute_detection_mean_ap(
        ground_truths, predictions, iou_thresholds=(0.5,)
    )

    assert metrics["classes"] == [0, 1]
    assert metrics["per_class_ap"][0.5][0] == 1.0
    assert metrics["per_class_ap"][0.5][1] == 0.0
    assert metrics["mAP"] == 0.5


def test_detection_map_averages_across_iou_thresholds() -> None:
    ground_truths = {
        "img1": [{"bbox": [0, 0, 10, 10], "class_id": 0}],
    }
    predictions = {
        "img1": [{"bbox": [1, 1, 11, 11], "score": 0.95, "class_id": 0}],
    }

    metrics = compute_detection_mean_ap(
        ground_truths,
        predictions,
        iou_thresholds=(0.5, 0.75),
    )

    assert metrics["thresholds"][0.5] == 1.0
    assert metrics["thresholds"][0.75] == 0.0
    assert metrics["mAP"] == 0.5


def test_detection_map_ignores_classes_without_ground_truth() -> None:
    ground_truths = {
        "img1": [{"bbox": [0, 0, 10, 10], "class_id": 0}],
    }
    predictions = {
        "img1": [{"bbox": [0, 0, 10, 10], "score": 0.95, "class_id": 2}],
    }

    metrics = compute_detection_mean_ap(
        ground_truths,
        predictions,
        iou_thresholds=(0.5,),
        class_ids=(0, 2),
    )

    assert metrics["classes"] == [0]
    assert metrics["per_class_ap"][0.5][0] == 0.0
    assert metrics["mAP"] == 0.0


def test_format_overall_metrics_from_counts_matches_expected_formula() -> None:
    metrics = _format_overall_metrics_from_counts(
        {
            "idtp": 80,
            "idfp": 10,
            "idfn": 20,
            "num_false_positives": 12,
            "num_misses": 18,
            "num_switches": 5,
            "num_objects": 100,
            "num_detections": 82,
            "num_predictions": 94,
        }
    )

    assert metrics["IDF1"] == "84.2%"
    assert metrics["MOTA"] == "65.0%"
    assert metrics["IDs"] == 5
    assert metrics["FP"] == 12
    assert metrics["FN"] == 18
    assert metrics["Rcll"] == "82.0%"
    assert metrics["Prcn"] == "87.2%"
