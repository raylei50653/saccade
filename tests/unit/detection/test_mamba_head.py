import torch
import torch.nn as nn
from saccade.perception.temporal_yolo.mamba_head import (
    MambaBlock,
    MambaDetectionHead,
    _selective_scan_legacy_n1,
    _gather_strip_pixels,
)


def test_legacy_n1_scan_reproduces_flattened_stride_bug():
    u = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    delta = torch.zeros_like(u)
    a = torch.tensor([[-1.0, -2.0, -3.0]])
    b = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 1
    c = (torch.arange(4, dtype=torch.float32).reshape(2, 2, 1) + 1) / 10

    actual = _selective_scan_legacy_n1(u, delta, a, b, c)

    softplus_zero = torch.log(torch.tensor(2.0))
    b_flat = b.flatten()[:4].reshape(2, 2)
    c_flat = c.flatten()[:4].reshape(2, 2)
    expected = torch.zeros_like(u)
    h = torch.zeros(2, 1)
    for t in range(2):
        h = (
            torch.exp(softplus_zero * a.flatten()[0]) * h
            + softplus_zero * b_flat[:, t : t + 1] * u[:, t]
        )
        expected[:, t] = h * c_flat[:, t : t + 1]

    torch.testing.assert_close(actual, expected)
    assert b_flat[1, 0].item() == b[0, 0, 2].item()
    torch.testing.assert_close(c_flat, c.squeeze(-1))


def test_legacy_n1_flag_propagates_to_spatial_and_temporal_blocks():
    model = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        use_temporal_mamba=True,
        legacy_n1_scan=True,
    )

    assert model.legacy_n1_scan
    assert all(block.legacy_n1_scan for level in model.mamba_blocks for block in level)
    assert model.temporal_blocks is not None
    assert all(
        block.legacy_n1_scan for level in model.temporal_blocks for block in level
    )
    temporal = model.temporal_blocks[0][0]
    assert temporal.x_proj.out_features == temporal.d_state * 2 + 1
    assert isinstance(model.mamba_blocks[0][0], MambaBlock)


def test_mamba_head_creation():
    # 1. Create original bilinear model
    model_bilinear = MambaDetectionHead(
        in_channels=(128, 256, 512), d_model=128, use_pixel_shuffle=False
    )
    assert not model_bilinear.use_pixel_shuffle
    assert not model_bilinear.upsample_loaded
    assert hasattr(model_bilinear, "upsample")

    # 2. Create new PixelShuffle model
    model_shuffle = MambaDetectionHead(
        in_channels=(128, 256, 512), d_model=128, use_pixel_shuffle=True
    )
    assert model_shuffle.use_pixel_shuffle
    assert model_shuffle.upsample_loaded
    assert hasattr(model_shuffle, "upsample")


def test_mamba_head_forward():
    B = 2
    feats = [
        torch.randn(B, 128, 80, 80),
        torch.randn(B, 256, 40, 40),
        torch.randn(B, 512, 20, 20),
    ]

    # Test bilinear forward
    model_bilinear = MambaDetectionHead(
        in_channels=(128, 256, 512),
        d_model=64,  # Use smaller d_model for speed
        use_pixel_shuffle=False,
    )
    cls_preds, reg_preds = model_bilinear(feats)
    assert len(cls_preds) == 3
    assert len(reg_preds) == 3
    assert cls_preds[0].shape == (B, 80, 80, 80)  # num_classes = 80
    assert reg_preds[0].shape == (B, 4, 80, 80)  # reg_max * 4 = 1 * 4

    # Test PixelShuffle forward
    model_shuffle = MambaDetectionHead(
        in_channels=(128, 256, 512), d_model=64, use_pixel_shuffle=True
    )
    cls_preds_shuf, reg_preds_shuf = model_shuffle(feats)
    assert len(cls_preds_shuf) == 3
    assert len(reg_preds_shuf) == 3
    assert cls_preds_shuf[0].shape == (B, 80, 80, 80)
    assert reg_preds_shuf[0].shape == (B, 4, 80, 80)


def test_mamba_head_state_dict_fallback():
    # Model A: Trained model without upsample layers (e.g. old checkpoints)
    model_old = MambaDetectionHead(
        in_channels=(128, 256, 512), d_model=64, use_pixel_shuffle=False
    )
    old_state = model_old.state_dict()

    # Verify 'upsample' keys are NOT in the old state dict
    assert not any("upsample" in k for k in old_state.keys())

    # Model B: New model initialized with PixelShuffle enabled
    model_new = MambaDetectionHead(
        in_channels=(128, 256, 512), d_model=64, use_pixel_shuffle=True
    )
    assert model_new.use_pixel_shuffle
    assert model_new.upsample_loaded

    # Load old state dict into the new model (strict=False, replicating checkpoint load)
    missing, unexpected = model_new.load_state_dict(old_state, strict=False)

    # The new upsample parameters should be missing
    assert any("upsample" in k for k in missing)

    # Crucial: upsample_loaded should now be False, triggering the safe bilinear fallback!
    assert not model_new.upsample_loaded

    # Run forward pass to verify it falls back to bilinear successfully and outputs correct shapes
    B = 2
    feats = [
        torch.randn(B, 128, 80, 80),
        torch.randn(B, 256, 40, 40),
        torch.randn(B, 512, 20, 20),
    ]
    cls_preds, reg_preds = model_new(feats)
    assert cls_preds[0].shape == (B, 80, 80, 80)
    assert reg_preds[0].shape == (B, 4, 80, 80)


def test_detail_fusion_is_identity_at_initialization():
    torch.manual_seed(7)
    base = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=1,
        spatial_reduction=2,
    )
    detail = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=1,
        spatial_reduction=2,
        use_detail_fusion=True,
        detail_channels=8,
    )
    detail.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    detail.eval()

    feats = [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]
    detail_images = torch.randn(2, 3, 48, 80)
    valid_hw = torch.tensor([[48, 80], [32, 64]])

    with torch.no_grad():
        base_cls, base_reg = base(feats)
        detail_cls, detail_reg = detail(
            feats,
            detail_images=detail_images,
            detail_valid_hw=valid_hw,
        )

    for expected, actual in zip(base_cls + base_reg, detail_cls + detail_reg):
        torch.testing.assert_close(actual, expected)


def test_detail_fusion_projection_receives_gradients():
    model = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=1,
        spatial_reduction=2,
        use_detail_fusion=True,
        detail_channels=8,
    )
    feats = [
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
    ]
    detail_images = torch.randn(1, 3, 48, 80)
    cls_preds, reg_preds = model(
        feats,
        detail_images=detail_images,
        detail_valid_hw=torch.tensor([[40, 64]]),
    )
    (cls_preds[0].sum() + reg_preds[0].sum()).backward()

    assert model.detail_fusion is not None
    assert model.detail_fusion.cls_proj.weight.grad is not None
    assert model.detail_fusion.reg_proj.weight.grad is not None
    assert model.detail_fusion.cls_proj.weight.grad.abs().sum() > 0
    assert model.detail_fusion.reg_proj.weight.grad.abs().sum() > 0


def test_detail_encoder_can_copy_matching_yolo_stem():
    class Stem(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(8)

    model = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        use_detail_fusion=True,
        detail_channels=8,
    )
    stem = Stem()
    with torch.no_grad():
        stem.conv.weight.fill_(0.25)
        stem.bn.weight.fill_(0.75)

    assert model.initialize_detail_from_yolo_stem(stem)
    assert model.detail_fusion is not None
    torch.testing.assert_close(model.detail_fusion.encoder[0].weight, stem.conv.weight)
    torch.testing.assert_close(model.detail_fusion.encoder[1].weight, stem.bn.weight)


def test_wide_detail_encoder_preserves_yolo_stem_warm_start():
    class Stem(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(32)

    model = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        use_detail_fusion=True,
        detail_channels=64,
    )
    stem = Stem()

    assert model.initialize_detail_from_yolo_stem(stem)
    assert model.detail_fusion is not None
    assert model.detail_fusion.encoder[0].out_channels == 32
    assert model.detail_fusion.encoder[3].in_channels == 32
    assert model.detail_fusion.encoder[3].out_channels == 64
    torch.testing.assert_close(model.detail_fusion.encoder[0].weight, stem.conv.weight)


def test_external_p3_detail_feature_path_is_identity_and_trainable():
    base = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=1,
        spatial_reduction=2,
    )
    oracle = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=1,
        spatial_reduction=2,
        use_detail_fusion=True,
        detail_channels=8,
        detail_feature_channels=12,
    )
    oracle.load_state_dict(base.state_dict(), strict=False)
    feats = [
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
    ]
    native_p3 = torch.randn(1, 12, 12, 20)
    valid_hw = torch.tensor([[80, 144]])

    base.eval()
    oracle.eval()
    with torch.no_grad():
        expected_cls, expected_reg = base(feats)
        actual_cls, actual_reg = oracle(
            feats,
            detail_features=native_p3,
            detail_valid_hw=valid_hw,
            detail_feature_stride=8,
        )
    for expected, actual in zip(expected_cls + expected_reg, actual_cls + actual_reg):
        torch.testing.assert_close(actual, expected)

    oracle.train()
    cls_preds, reg_preds = oracle(
        feats,
        detail_features=native_p3,
        detail_valid_hw=valid_hw,
        detail_feature_stride=8,
    )
    (cls_preds[0].sum() + reg_preds[0].sum()).backward()
    assert oracle.detail_fusion is not None
    assert oracle.detail_fusion.feature_adapter is not None
    assert oracle.detail_fusion.cls_proj.weight.grad is not None


def test_strip_detail_gather_has_fixed_four_direction_shape_and_reversals():
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    positions = torch.tensor([[[0, 0], [3, 3]]])

    strips = _gather_strip_pixels(
        image,
        positions,
        output_hw=(4, 4),
        valid_hw=torch.tensor([[8, 8]]),
        strip_length=4,
        strip_width=3,
    ).reshape(1, 2, 4, 3, 12)

    assert strips.shape == (1, 2, 4, 3, 12)
    torch.testing.assert_close(strips[:, :, 1], strips[:, :, 0].flip(-1))
    torch.testing.assert_close(strips[:, :, 3], strips[:, :, 2].flip(-1))


def test_strip_detail_is_identity_at_initialization_and_receives_gradients():
    torch.manual_seed(11)
    base = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=1,
        spatial_reduction=2,
    )
    strip = MambaDetectionHead(
        in_channels=(8, 16, 32),
        d_model=8,
        d_state=4,
        num_blocks=1,
        num_classes=1,
        spatial_reduction=2,
        use_strip_detail=True,
        strip_stem_channels=4,
        strip_length=8,
        strip_width=3,
    )
    strip.load_state_dict(base.state_dict(), strict=False)
    feats = [
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
    ]
    detail_images = torch.randn(1, 3, 48, 80)
    positions = torch.tensor([[[0, 0], [8, 9], [15, 15]]])
    position_mask = torch.tensor([[True, True, False]])

    base.eval()
    strip.eval()
    with torch.no_grad():
        expected_cls, expected_reg = base(feats)
        actual_cls, actual_reg = strip(
            feats,
            detail_images=detail_images,
            detail_valid_hw=torch.tensor([[40, 64]]),
            detail_positions=positions,
            detail_position_mask=position_mask,
        )
    for expected, actual in zip(expected_cls + expected_reg, actual_cls + actual_reg):
        torch.testing.assert_close(actual, expected)

    strip.train()
    cls_preds, reg_preds = strip(
        feats,
        detail_images=detail_images,
        detail_valid_hw=torch.tensor([[40, 64]]),
        detail_positions=positions,
        detail_position_mask=position_mask,
    )
    (cls_preds[0].sum() + reg_preds[0].sum()).backward()
    assert strip.strip_detail_fusion is not None
    assert strip.strip_detail_fusion.cls_proj.weight.grad is not None
    assert strip.strip_detail_fusion.reg_proj.weight.grad is not None
    assert strip.strip_detail_fusion.cls_proj.weight.grad.abs().sum() > 0
    assert strip.strip_detail_fusion.reg_proj.weight.grad.abs().sum() > 0

    # Zero-init projections intentionally block stem/Mamba gradients on the
    # first step. Once the projections move, gradients reach the full branch.
    strip.zero_grad(set_to_none=True)
    with torch.no_grad():
        strip.strip_detail_fusion.cls_proj.weight.fill_(0.01)
        strip.strip_detail_fusion.reg_proj.weight.fill_(0.01)
    cls_preds, reg_preds = strip(
        feats,
        detail_images=detail_images,
        detail_valid_hw=torch.tensor([[40, 64]]),
        detail_positions=positions,
        detail_position_mask=position_mask,
    )
    (cls_preds[0].sum() + reg_preds[0].sum()).backward()
    assert strip.strip_detail_fusion.stem[0].weight.grad is not None
    assert strip.strip_detail_fusion.stem[0].weight.grad.abs().sum() > 0
    assert strip.strip_detail_fusion.mamba.in_proj.weight.grad is not None
    assert strip.strip_detail_fusion.mamba.in_proj.weight.grad.abs().sum() > 0
