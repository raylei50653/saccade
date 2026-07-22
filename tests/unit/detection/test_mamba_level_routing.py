"""Unit tests for mamba gated-detector level routing (perception.temporal_yolo.mamba_gated_detector)."""

# scope: detection
# function: behavior
# lifecycle: active

import torch

from saccade.perception.temporal_yolo.mamba_gated_detector import (
    _postprocess_mamba_fixed_eager,
)


def _predictions() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    cls_preds = [
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[3.0]]]]),
        torch.tensor([[[[2.0]]]]),
    ]
    reg_preds = [
        torch.full((1, 4, 1, 1), 0.5),
        torch.full((1, 4, 1, 1), 0.25),
        torch.full((1, 4, 1, 1), 0.125),
    ]
    return cls_preds, reg_preds


def test_small_p3_max_uses_coarse_score_with_p3_box() -> None:
    cls_preds, reg_preds = _predictions()
    strides = torch.tensor([8.0, 16.0, 32.0])

    baseline = _postprocess_mamba_fixed_eager(
        cls_preds, reg_preds, strides, conf_thr=0.0, max_det=3
    )
    routed = _postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=3,
        small_p3_max_threshold=10.0,
    )

    assert torch.allclose(baseline[0, 0, 4], torch.sigmoid(torch.tensor(3.0)))
    assert torch.allclose(routed[0, 0, 4], torch.sigmoid(torch.tensor(3.0)))
    assert torch.allclose(routed[0, 0, :4], torch.tensor([0.0, 0.0, 8.0, 8.0]))


def test_small_p3_max_threshold_uses_original_image_scale() -> None:
    cls_preds, reg_preds = _predictions()
    strides = torch.tensor([8.0, 16.0, 32.0])

    routed = _postprocess_mamba_fixed_eager(
        cls_preds,
        reg_preds,
        strides,
        conf_thr=0.0,
        max_det=3,
        small_p3_max_threshold=10.0,
        box_scale_x=torch.tensor([2.0]),
        box_scale_y=torch.tensor([2.0]),
    )

    assert torch.allclose(routed[0, 0, 4], torch.sigmoid(torch.tensor(3.0)))
