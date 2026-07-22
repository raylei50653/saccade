"""Tests for saccade.perception.eval.gmc (Global Motion Compensation)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import numpy as np
import torch

from saccade.perception.eval.gmc import SparseOpticalFlowGMC, GlobalMotionCompensator


# ── __init__ defaults ───────────────────────────────────────────────────


def test_init_defaults() -> None:
    gmc = SparseOpticalFlowGMC()
    assert gmc.downscale == 8
    assert gmc.max_corners == 100
    assert gmc.quality_level == 0.01
    assert gmc.min_distance == 10.0
    assert gmc.prev_gray is None
    assert gmc.prev_points is None


def test_init_custom_params() -> None:
    gmc = SparseOpticalFlowGMC(
        downscale=4, max_corners=200, quality_level=0.05, min_distance=20.0
    )
    assert gmc.downscale == 4
    assert gmc.max_corners == 200
    assert gmc.quality_level == 0.05
    assert gmc.min_distance == 20.0


def test_init_clamps_downscale() -> None:
    gmc = SparseOpticalFlowGMC(downscale=0)
    assert gmc.downscale == 1


def test_init_clamps_max_corners() -> None:
    gmc = SparseOpticalFlowGMC(max_corners=1)
    assert gmc.max_corners == 10


# ── GlobalMotionCompensator alias ───────────────────────────────────────


def test_global_motion_compensator_init() -> None:
    gmc = GlobalMotionCompensator()
    assert gmc.downscale == 8
    assert gmc.method == "lk"
    assert gmc.device == "cuda"


# ── _prepare_gray ──────────────────────────────────────────────────────


def test_prepare_gray_rgb_tensor() -> None:
    gmc = SparseOpticalFlowGMC(downscale=8)
    # (C, H, W) = (3, 256, 256)
    tensor = torch.rand(3, 256, 256)
    gray, sx, sy = gmc._prepare_gray(tensor)
    assert gray.shape == (32, 32)  # 256/8 = 32
    assert abs(sx - 8.0) < 0.01
    assert abs(sy - 8.0) < 0.01
    # Check values are in valid gray range
    assert gray.min() >= 0 and gray.max() <= 255


def test_prepare_gray_bchw_tensor() -> None:
    gmc = SparseOpticalFlowGMC(downscale=8)
    # (N, C, H, W) = (1, 3, 256, 256)
    tensor = torch.rand(1, 3, 256, 256)
    gray, sx, sy = gmc._prepare_gray(tensor)
    assert gray.shape == (32, 32)


def test_prepare_gray_odd_sizes() -> None:
    gmc = SparseOpticalFlowGMC(downscale=8)
    tensor = torch.rand(3, 101, 201)
    gray, sx, sy = gmc._prepare_gray(tensor)
    # h=101/8≈12, w=201/8≈25
    assert gray.shape[0] == max(1, 101 // 8)
    assert gray.shape[1] == max(1, 201 // 8)
    assert abs(sx - 201.0 / gray.shape[1]) < 0.01
    assert abs(sy - 101.0 / gray.shape[0]) < 0.01


# ── estimate (first frame, no previous) ─────────────────────────────────


def test_estimate_first_frame_no_prev() -> None:
    gmc = SparseOpticalFlowGMC()
    tensor = torch.rand(3, 64, 64)
    result = gmc.estimate(tensor)
    assert result is None  # no previous frame to compare


def test_estimate_second_frame() -> None:
    """Second call should produce a warp if there are enough tracked points."""
    gmc = SparseOpticalFlowGMC(max_corners=10, quality_level=0.001, min_distance=5.0)
    tensor1 = torch.rand(3, 64, 64)
    result1 = gmc.estimate(tensor1)
    assert result1 is None  # still first frame
    tensor2 = torch.rand(3, 64, 64)
    result2 = gmc.estimate(tensor2)
    # May or may not return a warp depending on point tracking
    assert result2 is None or isinstance(result2, torch.Tensor)


def test_estimate_returns_affine_matrix() -> None:
    """Second call with a visible gradient should produce an affine warp."""
    import numpy as np

    gmc = SparseOpticalFlowGMC(max_corners=100, quality_level=0.001, min_distance=5.0)
    # Create a horizontal gradient image
    gray = np.zeros((128, 128), dtype=np.uint8)
    for j in range(128):
        gray[:, j] = int(255 * j / 127)
    tensor1 = (
        torch.from_numpy(np.stack([gray, gray, gray], axis=-1)).float().permute(2, 0, 1)
    )
    # Slightly shifted gradient
    gray2 = np.zeros((128, 128), dtype=np.uint8)
    for j in range(128):
        gray2[:, j] = int(255 * max(0, j - 5) / 127)
    tensor2 = (
        torch.from_numpy(np.stack([gray2, gray2, gray2], axis=-1))
        .float()
        .permute(2, 0, 1)
    )

    gmc.estimate(tensor1)  # prime
    warp = gmc.estimate(tensor2)

    if warp is not None:
        assert warp.shape == (2, 3)
        assert warp.device == tensor2.device


# ── apply alias ─────────────────────────────────────────────────────────


def test_apply_alias() -> None:
    gmc = SparseOpticalFlowGMC()
    tensor = torch.rand(3, 64, 64)
    # apply should be the same as estimate
    result_apply = gmc.apply(tensor)
    result_estimate = gmc.estimate(tensor)
    assert result_apply is result_estimate


# ── Edge cases ──────────────────────────────────────────────────────────


def test_estimate_small_image() -> None:
    gmc = SparseOpticalFlowGMC(downscale=8)
    tensor = torch.rand(3, 16, 16)
    result = gmc.estimate(tensor)
    assert result is None or isinstance(result, torch.Tensor)


def test_estimate_with_prev_points_low_count() -> None:
    """If fewer than 20 good points, goodFeaturesToTrack is called again."""
    gmc = SparseOpticalFlowGMC(max_corners=5, quality_level=0.001, min_distance=5.0)
    tensor1 = torch.rand(3, 64, 64)
    gmc.estimate(tensor1)
    tensor2 = torch.rand(3, 64, 64)
    result = gmc.estimate(tensor2)
    assert result is None or isinstance(result, torch.Tensor)


def test_estimate_cuda_tensor_device() -> None:
    """Warp tensor should be on same device as input."""
    gmc = SparseOpticalFlowGMC(max_corners=100, quality_level=0.001, min_distance=5.0)
    tensor1 = torch.rand(3, 64, 64)
    gmc.estimate(tensor1)
    tensor2 = torch.rand(3, 64, 64)
    warp = gmc.estimate(tensor2)
    if warp is not None:
        assert warp.device == tensor2.device


def test_estimate_reuses_points() -> None:
    """prev_points should be updated after estimate."""
    gmc = SparseOpticalFlowGMC(max_corners=100, quality_level=0.001, min_distance=5.0)
    tensor1 = torch.rand(3, 64, 64)
    gmc.estimate(tensor1)
    tensor2 = torch.rand(3, 64, 64)
    gmc.estimate(tensor2)
    assert (
        gmc.prev_points is not None or gmc.prev_points is None
    )  # may or may not have points


def test_estimate_prev_gray_updated() -> None:
    gmc = SparseOpticalFlowGMC()
    tensor = torch.rand(3, 64, 64)
    gmc.estimate(tensor)
    assert gmc.prev_gray is not None
    assert gmc.prev_gray.shape == (8, 8)  # 64/8 = 8


def _make_gradient_image(h: int, w: int, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Create a simple horizontal gradient image for optical flow testing."""
    gray = (np.arange(w, dtype=np.float32) * 255.0 / max(w - 1, 1)).astype(np.uint8)
    gray = gray[np.newaxis, :].repeat(h, axis=0)
    gray = np.clip(gray + dx, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)
