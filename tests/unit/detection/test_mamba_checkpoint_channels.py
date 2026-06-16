import torch

from saccade.perception.temporal_yolo.mamba_head import (
    DEFAULT_MAMBA_IN_CHANNELS,
    resolve_mamba_in_channels,
)


def test_resolve_mamba_in_channels_prefers_metadata() -> None:
    checkpoint = {"mamba_args": {"in_channels": [256, 512, 512]}}

    assert resolve_mamba_in_channels(checkpoint) == (256, 512, 512)


def test_resolve_mamba_in_channels_from_input_projection_weights() -> None:
    checkpoint = {
        "student": {
            "input_proj.0.weight": torch.empty(128, 256, 1, 1),
            "input_proj.1.weight": torch.empty(128, 512, 1, 1),
            "input_proj.2.weight": torch.empty(128, 512, 1, 1),
        }
    }

    assert resolve_mamba_in_channels(checkpoint) == (256, 512, 512)


def test_resolve_mamba_in_channels_legacy_fallback() -> None:
    assert resolve_mamba_in_channels({}) == DEFAULT_MAMBA_IN_CHANNELS
