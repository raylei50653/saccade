"""Unit tests for the YOLO gated detector (perception.temporal_yolo.yolo_gated_detector)."""

# scope: detection
# function: behavior
# lifecycle: active

import torch.nn as nn

from saccade.perception.temporal_yolo.yolo_gated_detector import (
    GatedDetConfig,
    GatedYOLODetector,
)


def _stub_detector(*, freeze_backbone: bool) -> GatedYOLODetector:
    model = GatedYOLODetector.__new__(GatedYOLODetector)
    nn.Module.__init__(model)
    model.cfg = GatedDetConfig(freeze_backbone=freeze_backbone)
    model.yolo_model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout2d())
    model.gate = nn.Sequential(nn.Dropout())
    model.fusion = None
    model._gate_layers = {}
    return model


def test_train_keeps_frozen_yolo_and_batchnorm_in_eval_mode():
    model = _stub_detector(freeze_backbone=True)

    model.train()

    assert model.training
    assert model.yolo_model.training
    assert not model.yolo_model[0].training
    assert model.gate.training


def test_train_allows_unfrozen_yolo_batchnorm_updates():
    model = _stub_detector(freeze_backbone=False)

    model.train()

    assert model.yolo_model.training
    assert model.yolo_model[0].training
