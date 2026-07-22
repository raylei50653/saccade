"""Tests for the runtime external-FP filter (perception.eval.evaluator._apply_external_fp_filter)."""

# scope: eval
# function: behavior
# lifecycle: active

import torch

from saccade.perception.eval.evaluator import _apply_external_fp_filter
from saccade.perception.eval.external_fp_model import RuleBaselineConfig


def test_rule_external_fp_filter_drops_small_low_score_boxes() -> None:
    boxes = torch.tensor(
        [
            [10.0, 10.0, 30.0, 90.0],  # height=80, aspect=4.0 -> keep
            [40.0, 10.0, 60.0, 70.0],  # height=60, score<0.10 -> drop
            [80.0, 20.0, 120.0, 110.0],  # height=90, aspect=2.25 -> keep
            [140.0, 20.0, 200.0, 100.0],  # height=80, aspect=1.33 -> drop
            [220.0, 20.0, 260.0, 70.0],  # score above max_score -> untouched
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.09, 0.08, 0.17, 0.17, 0.25], dtype=torch.float32)
    classes = torch.zeros(5, dtype=torch.float32)

    filtered_boxes, filtered_scores, filtered_classes = _apply_external_fp_filter(
        boxes,
        scores,
        classes,
        image_width=320,
        image_height=180,
        mode="rule",
        rule_config=RuleBaselineConfig(),
        logistic_model=None,
        logistic_threshold=0.5,
        max_score=0.18,
        penalty=1.0,
        min_score=0.05,
        softmax_min_scale=0.7,
    )

    assert filtered_boxes.shape == (3, 4)
    assert torch.equal(filtered_scores, torch.tensor([0.09, 0.17, 0.25]))
    assert torch.equal(filtered_classes, torch.zeros(3, dtype=torch.float32))


def test_rule_score_external_fp_filter_penalizes_instead_of_hard_dropping() -> None:
    boxes = torch.tensor(
        [
            [40.0, 10.0, 60.0, 70.0],  # would fail low-score rule
            [140.0, 20.0, 200.0, 100.0],  # would fail medium-score rule
            [220.0, 20.0, 260.0, 70.0],  # above max_score, unchanged
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.08, 0.17, 0.25], dtype=torch.float32)
    classes = torch.zeros(3, dtype=torch.float32)

    filtered_boxes, filtered_scores, filtered_classes = _apply_external_fp_filter(
        boxes,
        scores,
        classes,
        image_width=320,
        image_height=180,
        mode="rule_score",
        rule_config=RuleBaselineConfig(),
        logistic_model=None,
        logistic_threshold=0.5,
        max_score=0.18,
        penalty=0.7,
        min_score=0.05,
        softmax_min_scale=0.7,
    )

    assert filtered_boxes.shape == (3, 4)
    assert torch.allclose(filtered_scores, torch.tensor([0.056, 0.119, 0.25]))
    assert torch.equal(filtered_classes, classes)
