"""Unit tests for the teacher-head detector (perception.temporal_yolo.teacher_head_detector)."""

# scope: detection
# function: behavior
# lifecycle: active

from __future__ import annotations

from types import SimpleNamespace

import torch

from saccade.perception.temporal_yolo.teacher_head_detector import (
    _get_topk_index_graph_safe,
)


def test_graph_safe_topk_preserves_class_aware_candidates() -> None:
    head = SimpleNamespace(export=False, agnostic_nms=False)
    scores = torch.tensor(
        [
            [
                [0.90, 0.95, 0.05],
                [0.89, 0.01, 0.01],
            ]
        ],
        dtype=torch.float32,
    )

    top_scores, classes, anchor_idx = _get_topk_index_graph_safe(head, scores, 2)

    assert torch.allclose(top_scores[..., 0], torch.tensor([[0.95, 0.90]]))
    assert torch.equal(classes[..., 0], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(anchor_idx[..., 0], torch.tensor([[0, 0]]))
