import torch
from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead


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
