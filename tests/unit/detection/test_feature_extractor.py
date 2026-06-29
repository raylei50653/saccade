from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from saccade.perception.feature_extractor import (
    TRTFeatureExtractor,
    _last_vit_dual,
)


def _stub_extractor(
    *,
    model_type: str = "siglip2",
    feature_dim: int = 4,
) -> TRTFeatureExtractor:
    extractor = TRTFeatureExtractor.__new__(TRTFeatureExtractor)
    extractor.model_type = model_type
    extractor.device = "cpu"
    extractor.max_batch = 2
    extractor.feature_dim = feature_dim
    extractor.input_hw = (224, 224)
    extractor.is_dynamic = False
    extractor.output_names = []
    extractor.output_buffers = {}
    extractor._embed_key = "image_embeds"
    extractor._imagenet_mean = None
    extractor._imagenet_std = None
    extractor._cpp = None
    return extractor


def test_normalize_siglip_maps_unit_range_to_signed_range() -> None:
    extractor = _stub_extractor(model_type="siglip2")
    pixels = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float64)

    normalized = extractor._normalize(pixels)

    assert normalized.dtype == torch.float32
    assert normalized.is_contiguous()
    torch.testing.assert_close(
        normalized, torch.tensor([[[[-1.0, 0.0, 1.0]]]], dtype=torch.float32)
    )


def test_normalize_imagenet_model_uses_cached_mean_and_std() -> None:
    extractor = _stub_extractor(model_type="dinov2")
    extractor._imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    extractor._imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels = torch.ones((1, 3, 1, 1), dtype=torch.float32)

    normalized = extractor._normalize(pixels)

    expected = (pixels - extractor._imagenet_mean) / extractor._imagenet_std
    torch.testing.assert_close(normalized, expected)
    assert normalized.is_contiguous()


def test_cpp_ptr_raises_when_cpp_extension_not_attached() -> None:
    extractor = _stub_extractor()

    with pytest.raises(RuntimeError, match="C\\+\\+ extension not available"):
        _ = extractor.cpp_ptr


def test_extract_parts_fused_python_fallback_weights_and_normalizes() -> None:
    extractor = _stub_extractor(feature_dim=4)

    def fake_extract(
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        assert stream is None
        assert input_tensor.shape == (3, 3, 224, 224)
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

    extractor.extract = fake_extract  # type: ignore[method-assign]
    crops = torch.zeros((3, 3, 224, 224), dtype=torch.float32)

    fused = extractor.extract_parts_fused(crops)

    expected = F.normalize(torch.tensor([[0.5, 0.3, 0.2, 0.0]]), dim=-1)
    torch.testing.assert_close(fused, expected)


def test_extract_parts_fused_empty_batch_returns_empty_embeddings() -> None:
    extractor = _stub_extractor(feature_dim=8)
    crops = torch.empty((0, 3, 224, 224), dtype=torch.float32)

    fused = extractor.extract_parts_fused(crops)

    assert fused.shape == (0, 8)
    assert fused.device.type == "cpu"


def test_extract_parts_fused_rejects_batches_that_are_not_three_parts() -> None:
    extractor = _stub_extractor()
    crops = torch.zeros((4, 3, 224, 224), dtype=torch.float32)

    with pytest.raises(ValueError, match="multiple of 3"):
        extractor.extract_parts_fused(crops)


def test_extract_with_stability_without_hidden_state_returns_default_stability() -> (
    None
):
    extractor = _stub_extractor(feature_dim=3)
    expected_embed = F.normalize(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        dim=-1,
    )

    def fake_extract(
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        assert stream is None
        assert input_tensor.shape == (2, 3, 224, 224)
        return expected_embed

    extractor.extract = fake_extract  # type: ignore[method-assign]
    images = torch.zeros((2, 3, 224, 224), dtype=torch.float32)

    embed, stability = extractor.extract_with_stability(images)

    torch.testing.assert_close(embed, expected_embed)
    torch.testing.assert_close(stability, torch.ones(2))


@pytest.mark.parametrize("top_k_ratio", [0.0, -0.1, 1.1])
def test_extract_with_stability_rejects_invalid_top_k_ratio(
    top_k_ratio: float,
) -> None:
    extractor = _stub_extractor()
    images = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    with pytest.raises(ValueError, match="top_k_ratio"):
        extractor.extract_with_stability(images, top_k_ratio=top_k_ratio)


def test_last_vit_dual_returns_embeddings_and_patch_scores() -> None:
    lhs = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

    embed, scores = _last_vit_dual(
        lhs,
        sigma_embed=0.015,
        sigma_gate=0.040,
        top_k_ratio=0.5,
    )

    assert embed.shape == (2, 3)
    assert scores.shape == (2, 4)


def test_last_vit_dual_rejects_invalid_top_k_ratio() -> None:
    lhs = torch.zeros((1, 4, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="top_k_ratio"):
        _last_vit_dual(lhs, sigma_embed=0.015, sigma_gate=0.040, top_k_ratio=2.0)
