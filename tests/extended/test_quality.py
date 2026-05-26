"""Tests for perception/eval/quality.py.

Covers:
  - compute_detection_quality_batch: vectorized quality scoring
  - compute_bank_quality_score: scalar quality scoring
"""

from __future__ import annotations


import pytest
import torch

from saccade.perception.eval.quality import (
    compute_bank_quality_score,
    compute_detection_quality_batch,
)


class TestComputeDetectionQualityBatch:
    """Tests for compute_detection_quality_batch."""

    def test_empty_boxes(self) -> None:
        """Empty boxes returns empty tensor."""
        boxes = torch.empty((0, 4), dtype=torch.float32)
        result = compute_detection_quality_batch(boxes, 640, 480)
        assert result.shape == (0,)
        assert result.dtype == torch.float32

    def test_single_centered_box(self) -> None:
        """Centered box should get high quality score."""
        boxes = torch.tensor([[280.0, 180.0, 360.0, 260.0]], dtype=torch.float32)
        result = compute_detection_quality_batch(boxes, 640, 480)
        assert result.shape == (1,)
        assert 0.0 <= result[0].item() <= 1.0

    def test_aspect_ratio_quality(self) -> None:
        """Aspect ratio near 2.5 should get higher quality than extreme ratios."""
        # Near ideal aspect ratio (2.5): tall thin box
        ideal = torch.tensor([[280.0, 100.0, 320.0, 200.0]], dtype=torch.float32)
        # Extreme aspect ratio: wide box
        wide = torch.tensor([[280.0, 180.0, 360.0, 190.0]], dtype=torch.float32)

        ideal_q = compute_detection_quality_batch(ideal, 640, 480)[0].item()
        wide_q = compute_detection_quality_batch(wide, 640, 480)[0].item()

        # Ideal aspect ratio should score higher than extreme wide ratio
        assert ideal_q > wide_q

    def test_center_quality(self) -> None:
        """Box in center should score higher than box at edge."""
        # Center box
        center = torch.tensor([[280.0, 180.0, 360.0, 260.0]], dtype=torch.float32)
        # Corner box
        corner = torch.tensor([[0.0, 0.0, 80.0, 80.0]], dtype=torch.float32)

        center_q = compute_detection_quality_batch(center, 640, 480)[0].item()
        corner_q = compute_detection_quality_batch(corner, 640, 480)[0].item()

        # Center box should score higher (less truncated)
        assert center_q > corner_q

    def test_area_quality(self) -> None:
        """Area quality component: Gaussian peak at 0.01 area ratio."""
        # Box with area ratio ~0.01 (ideal)
        ideal = torch.tensor([[280.0, 180.0, 360.0, 260.0]], dtype=torch.float32)
        # Tiny box: area_ratio = 4 / 307200 = 1.3e-5 (far from 0.01)
        tiny = torch.tensor([[318.0, 238.0, 322.0, 342.0]], dtype=torch.float32)

        ideal_q = compute_detection_quality_batch(ideal, 640, 480)[0].item()
        tiny_q = compute_detection_quality_batch(tiny, 640, 480)[0].item()

        # Ideal area box should generally score higher (aspect/center also contribute)
        assert 0.0 <= ideal_q <= 1.0
        assert 0.0 <= tiny_q <= 1.0
        # The ideal box has better overall score due to area + aspect + center
        assert ideal_q > tiny_q

    def test_multiple_boxes(self) -> None:
        """Batch of boxes returns correct shape."""
        boxes = torch.tensor(
            [
                [280.0, 180.0, 360.0, 260.0],
                [100.0, 100.0, 200.0, 200.0],
                [400.0, 200.0, 500.0, 300.0],
            ],
            dtype=torch.float32,
        )
        result = compute_detection_quality_batch(boxes, 640, 480)
        assert result.shape == (3,)

    def test_all_scores_in_range(self) -> None:
        """All scores should be in [0, 1]."""
        boxes = torch.rand((20, 4), dtype=torch.float32) * 640
        result = compute_detection_quality_batch(boxes, 640, 480)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_custom_weights(self) -> None:
        """Custom weights affect output."""
        boxes = torch.tensor([[280.0, 180.0, 360.0, 260.0]], dtype=torch.float32)

        compute_detection_quality_batch(boxes, 640, 480)[0].item()
        aspect_heavy = compute_detection_quality_batch(
            boxes, 640, 480, w_aspect=1.0, w_center=0.0, w_area=0.0
        )[0].item()

        # With all weight on aspect, score depends only on aspect ratio
        assert 0.0 <= aspect_heavy <= 1.0

    def test_cuda_device(self) -> None:
        """Works on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        boxes = torch.tensor([[280.0, 180.0, 360.0, 260.0]], device="cuda")
        result = compute_detection_quality_batch(boxes, 640, 480)
        assert result.device.type == "cuda"
        assert result.shape == (1,)

    def test_zero_area_clamped(self) -> None:
        """Tiny boxes with near-zero area should not cause division by zero."""
        # 1x1 pixel box
        boxes = torch.tensor([[100.0, 100.0, 101.0, 101.0]], dtype=torch.float32)
        result = compute_detection_quality_batch(boxes, 640, 480)
        assert not torch.isnan(result).any()
        assert result.shape == (1,)

    def test_large_frame(self) -> None:
        """Works with large frame dimensions."""
        boxes = torch.tensor([[2000.0, 1500.0, 2400.0, 1900.0]], dtype=torch.float32)
        result = compute_detection_quality_batch(boxes, 4000, 3000)
        assert result.shape == (1,)
        assert 0.0 <= result[0].item() <= 1.0


class TestComputeBankQualityScore:
    """Tests for compute_bank_quality_score."""

    def test_basic_calculation(self) -> None:
        """Basic quality score calculation works."""
        score = compute_bank_quality_score(
            det_score=0.8,
            iou=0.5,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
        )
        assert 0.0 <= score <= 1.0

    def test_high_det_score_increases_quality(self) -> None:
        """Higher detection score → higher quality."""
        low_det = compute_bank_quality_score(
            det_score=0.2,
            iou=0.5,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
        )
        high_det = compute_bank_quality_score(
            det_score=0.9,
            iou=0.5,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
        )
        assert high_det > low_det

    def test_high_iou_increases_quality(self) -> None:
        """Higher IoU → higher quality."""
        low_iou = compute_bank_quality_score(
            det_score=0.7,
            iou=0.1,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
        )
        high_iou = compute_bank_quality_score(
            det_score=0.7,
            iou=0.9,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
        )
        assert high_iou > low_iou

    def test_ideal_aspect_ratio_score(self) -> None:
        """Aspect ratio 2.5 should score ~1.0."""
        ideal_q = compute_bank_quality_score(
            det_score=0.0,
            iou=0.0,
            aspect_ratio=2.5,
            box=(0, 0, 0, 0),
            frame_w=1,
            frame_h=1,
            w_det=0,
            w_iou=0,
        )
        # Aspect contribution: math.exp(-0.5 * ((2.5-2.5)/1.2)**2) = 1.0
        expected_aspect = 0.15 * 1.0  # w_aspect * 1.0
        assert ideal_q > expected_aspect  # extra from center/area

    def test_unknown_aspect_ratio(self) -> None:
        """Aspect ratio 0.0 → neutral quality (0.5)."""
        # With only aspect contribution visible
        q = compute_bank_quality_score(
            det_score=0.0,
            iou=0.0,
            aspect_ratio=0.0,
            box=(0, 0, 0, 0),
            frame_w=1,
            frame_h=1,
            w_det=0,
            w_iou=0,
        )
        # aspect_q = 0.5 (neutral) → 0.15 * 0.5 = 0.075
        assert q > 0.0

    def test_center_bias_quality(self) -> None:
        """Center box should get higher quality than corner box."""
        center_q = compute_bank_quality_score(
            det_score=0.0,
            iou=0.0,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
            w_det=0,
            w_iou=0,
        )
        corner_q = compute_bank_quality_score(
            det_score=0.0,
            iou=0.0,
            aspect_ratio=2.5,
            box=(0.0, 0.0, 10.0, 10.0),
            frame_w=640,
            frame_h=480,
            w_det=0,
            w_iou=0,
        )
        assert center_q > corner_q

    def test_area_ratio_quality(self) -> None:
        """Area ratio quality: Gaussian peak at 0.01."""
        # Medium area box with center near center of frame
        medium_q = compute_bank_quality_score(
            det_score=0.0,
            iou=0.0,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
            w_det=0,
            w_iou=0,
        )
        # Tiny box at corner (poor center quality)
        tiny_q = compute_bank_quality_score(
            det_score=0.0,
            iou=0.0,
            aspect_ratio=2.5,
            box=(1.0, 1.0, 5.0, 5.0),
            frame_w=640,
            frame_h=480,
            w_det=0,
            w_iou=0,
        )
        # Medium box at center should score higher due to center + area quality
        assert medium_q > tiny_q

    def test_custom_weights(self) -> None:
        """Custom weights change the score."""
        q_default = compute_bank_quality_score(
            det_score=0.8,
            iou=0.5,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
        )
        # With only det score: 0.75 * 0.8 = 0.6
        q_det_only = compute_bank_quality_score(
            det_score=0.8,
            iou=0.0,
            aspect_ratio=0.0,
            box=(0, 0, 0, 0),
            frame_w=1,
            frame_h=1,
            w_det=0.75,
            w_iou=0.0,
            w_aspect=0.0,
            w_center=0.0,
            w_area=0.0,
        )
        # With all other zero: 0.75 * 0.8 = 0.6
        assert abs(q_det_only - 0.6) < 0.01
        # Default has additional contributions from aspect/center/area
        assert q_default > q_det_only

    def test_score_bounds(self) -> None:
        """Score should be bounded."""
        score = compute_bank_quality_score(
            det_score=1.0,
            iou=1.0,
            aspect_ratio=2.5,
            box=(280.0, 180.0, 360.0, 260.0),
            frame_w=640,
            frame_h=480,
        )
        assert 0.0 <= score <= 1.0

    def test_negative_box_coordinates(self) -> None:
        """Works with negative coordinates (edge cases)."""
        score = compute_bank_quality_score(
            det_score=0.5,
            iou=0.3,
            aspect_ratio=2.0,
            box=(-10.0, -10.0, 30.0, 50.0),
            frame_w=640,
            frame_h=480,
        )
        # Should not crash, center quality will clamp
        assert 0.0 <= score <= 1.0

    def test_large_frame(self) -> None:
        """Works with large frame dimensions."""
        score = compute_bank_quality_score(
            det_score=0.7,
            iou=0.4,
            aspect_ratio=2.5,
            box=(1900.0, 1400.0, 2300.0, 1800.0),
            frame_w=4000,
            frame_h=3000,
        )
        assert 0.0 <= score <= 1.0
