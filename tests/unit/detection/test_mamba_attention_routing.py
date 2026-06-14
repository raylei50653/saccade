import torch
from saccade.perception.temporal_yolo.mamba_head import (
    MambaDetectionHead,
    P3AttentionStripDetailFusion,
)


def test_p3_attention_strip_detail_fusion_shapes():
    d_model = 64
    batch = 2
    routed = 8
    h, w = 16, 16
    img_h, img_w = 48, 80

    detail_images = torch.randn(batch, 3, img_h, img_w)
    positions = torch.randint(0, 16, (batch, routed, 2))
    x_proj = torch.randn(batch, d_model, h, w)
    position_mask = torch.ones(batch, routed, dtype=torch.bool)

    module = P3AttentionStripDetailFusion(
        d_model=d_model,
        stem_channels=16,
        strip_length=8,
        strip_width=3,
        route_chunk_size=4,
        num_heads=2,
    )

    cls_delta, reg_delta = module(
        detail_images=detail_images,
        positions=positions,
        output_hw=(h, w),
        x_proj=x_proj,
        position_mask=position_mask,
    )

    assert cls_delta.shape == (batch, routed, d_model)
    assert reg_delta.shape == (batch, routed, d_model)


def test_attention_strip_detail_is_identity_at_initialization_and_receives_gradients():
    torch.manual_seed(42)
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
        strip_detail_type="attention",
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

    # Zero-init projections block stem/attn gradients on the first step.
    # Move projections to verify gradients reach full branch.
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
    assert strip.strip_detail_fusion.attn.in_proj_weight.grad is not None
    assert strip.strip_detail_fusion.attn.in_proj_weight.grad.abs().sum() > 0
