"""Additional tests for saccade.perception.eval.detection — _decode_detector_boxes,
expand_boxes_with_ankle_keypoints, match_keypoints_to_boxes, _tile_seam_mask_for_boxes,
_get_detector_static_batch_size.

These cover the Python-side pure functions that were previously untested.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from saccade.perception.eval.detection import (
    _decode_detector_boxes,
    expand_boxes_with_ankle_keypoints,
    match_keypoints_to_boxes,
    _tile_seam_mask_for_boxes,
    _get_detector_static_batch_size,
    detect_single_patch_640,
)


# ── _decode_detector_boxes ──────────────────────────────────────────────


def test_decode_boxes_xyxy_identity() -> None:
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0], [5.0, 5.0, 100.0, 200.0]])
    result = _decode_detector_boxes(boxes, "xyxy")
    assert torch.equal(result, boxes)
    assert result is not boxes  # should be a clone


def test_decode_boxes_cxcywh_to_xyxy() -> None:
    # cxcywh: center_x, center_y, width, height
    # (cx=50, cy=50, w=20, h=20) → xyxy (40, 40, 60, 60)
    boxes = torch.tensor([[50.0, 50.0, 20.0, 20.0]])
    result = _decode_detector_boxes(boxes, "cxcywh")
    expected = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
    assert torch.allclose(result, expected)


def test_decode_boxes_cxcywh_multiple() -> None:
    # Two boxes: (cx=100, cy=100, w=40, h=40) and (cx=200, cy=200, w=60, h=80)
    boxes = torch.tensor(
        [
            [100.0, 100.0, 40.0, 40.0],
            [200.0, 200.0, 60.0, 80.0],
        ]
    )
    result = _decode_detector_boxes(boxes, "cxcywh")
    # box 0: (80, 80, 120, 120)
    # box 1: (170, 160, 230, 240)
    expected = torch.tensor(
        [
            [80.0, 80.0, 120.0, 120.0],
            [170.0, 160.0, 230.0, 240.0],
        ]
    )
    assert torch.allclose(result, expected)


def test_decode_boxes_cxcywh_odd_sizes() -> None:
    # (cx=10, cy=10, w=5, h=7) → xyxy (7.5, 6.5, 12.5, 13.5)
    boxes = torch.tensor([[10.0, 10.0, 5.0, 7.0]])
    result = _decode_detector_boxes(boxes, "cxcywh")
    expected = torch.tensor([[7.5, 6.5, 12.5, 13.5]])
    assert torch.allclose(result, expected, atol=1e-6)


def test_decode_boxes_unsupported_format() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    with pytest.raises(ValueError, match="Unsupported detector_box_format"):
        _decode_detector_boxes(boxes, "nhwc")


def test_decode_boxes_empty() -> None:
    boxes = torch.empty((0, 4), dtype=torch.float32)
    result = _decode_detector_boxes(boxes, "xyxy")
    assert result.shape == (0, 4)
    assert torch.equal(result, boxes)


def test_detect_single_patch_640_nv12_whole_graph_uses_preprocessed_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStream:
        cuda_stream = 0

    class _FakePool:
        use_nv12 = True

        def __init__(self) -> None:
            self.frame_buffer_nv12 = torch.zeros(640 * 480 * 3 // 2, dtype=torch.uint8)
            self.canvas_640p = torch.zeros((3, 640, 640), dtype=torch.float32)

        def as_rgb_chw(self) -> torch.Tensor:
            raise AssertionError("legacy RGB fallback should not be used")

        def prepare_canvas_640_stretch(self, h_orig: int, w_orig: int) -> torch.Tensor:
            assert (h_orig, w_orig) == (480, 640)
            return self.canvas_640p

    class _FakeDetector:
        use_whole_graph = True
        _trt_backbone = object()

        def __init__(self) -> None:
            self.calls = 0

        def detect_raw_preprocessed(self, input_tensor: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert tuple(input_tensor.shape) == (1, 3, 640, 640)
            raw = torch.zeros((1, 300, 6), dtype=torch.float32)
            raw[0, 0, :4] = torch.tensor([10.0, 90.0, 110.0, 190.0])
            raw[0, 0, 4] = 0.9
            raw[0, 0, 5] = 1.0
            return raw

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream())

    pool = _FakePool()
    detector = _FakeDetector()
    boxes, scores, classes = detect_single_patch_640(
        detector=detector,
        pool=pool,
        h_orig=480,
        w_orig=640,
        preprocess_modes=[],
        detector_box_format="xyxy",
    )

    assert detector.calls == 1
    assert torch.allclose(boxes[0], torch.tensor([10.0, 67.5, 110.0, 142.5]))
    assert scores[0].item() == pytest.approx(0.9)
    assert classes[0].item() == pytest.approx(1.0)


# ── expand_boxes_with_ankle_keypoints ──────────────────────────────────


def test_expand_boxes_no_keypoints() -> None:
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    result = expand_boxes_with_ankle_keypoints(boxes, None, frame_h=480)
    assert torch.equal(result, boxes)


def test_expand_boxes_empty_boxes() -> None:
    boxes = torch.empty((0, 4), dtype=torch.float32)
    keypoints = torch.randn(0, 17, 3)
    result = expand_boxes_with_ankle_keypoints(boxes, keypoints, frame_h=480)
    assert result.shape == (0, 4)


def test_expand_boxes_ankle_below_box() -> None:
    """Box bottom should extend to ankle position when ankle is below box."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 80.0]])  # bottom=80
    # Ankle at y=100, confidence 0.5 (> threshold 0.30)
    keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0]] * 15 + [[20.0, 100.0, 0.5], [30.0, 100.0, 0.5]]]
    )
    result = expand_boxes_with_ankle_keypoints(boxes, keypoints, frame_h=480)
    # Expected: bottom = 100 + 0.05 * 70 = 103.5 (clamped to 480)
    assert result[0, 1] == 10.0  # top unchanged
    assert abs(result[0, 3] - 103.5) < 0.01  # bottom extended


def test_expand_boxes_ankle_inside_box() -> None:
    """Box should NOT expand when ankle is inside or above box bottom."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 120.0]])  # bottom=120
    # Ankle at y=100, which is inside the box
    keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0]] * 15 + [[20.0, 100.0, 0.5], [30.0, 100.0, 0.5]]]
    )
    result = expand_boxes_with_ankle_keypoints(boxes, keypoints, frame_h=480)
    assert torch.equal(result, boxes)  # no expansion


def test_expand_boxes_low_ankle_confidence() -> None:
    """Box should NOT expand when ankle confidence is below threshold."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 80.0]])
    # Ankle at y=100 but confidence 0.1 (< threshold 0.30)
    keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0]] * 15 + [[20.0, 100.0, 0.1], [30.0, 100.0, 0.1]]]
    )
    result = expand_boxes_with_ankle_keypoints(boxes, keypoints, frame_h=480)
    assert torch.equal(result, boxes)  # no expansion


def test_expand_boxes_nan_keypoints() -> None:
    """Unmatched detections get NaN keypoints; should not expand."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 80.0]])
    # Both ankle y values are NaN
    keypoints = torch.tensor(
        [
            [[0.0, 0.0, 0.0]] * 15
            + [[20.0, float("nan"), 0.5], [30.0, float("nan"), 0.5]]
        ]
    )
    result = expand_boxes_with_ankle_keypoints(boxes, keypoints, frame_h=480)
    assert torch.equal(result, boxes)  # NaN should be masked out, no expansion


def test_expand_boxes_clamped_to_frame() -> None:
    """Expanded box bottom should be clamped to frame height."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 400.0]])  # bottom=400
    # Ankle at y=500, beyond frame_h=480
    keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0]] * 15 + [[20.0, 500.0, 0.5], [30.0, 500.0, 0.5]]]
    )
    result = expand_boxes_with_ankle_keypoints(boxes, keypoints, frame_h=480)
    assert result[0, 3] == 480.0  # clamped to frame_h


def test_expand_boxes_multiple_detections() -> None:
    """Only some detections should expand."""
    boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 80.0],  # ankle below → expand
            [60.0, 10.0, 100.0, 120.0],  # ankle inside → no expand
        ]
    )
    keypoints = torch.tensor(
        [
            [[0.0, 0.0, 0.0]] * 15 + [[20.0, 100.0, 0.5], [30.0, 100.0, 0.5]],
            [[0.0, 0.0, 0.0]] * 15 + [[80.0, 100.0, 0.5], [90.0, 100.0, 0.5]],
        ]
    )
    result = expand_boxes_with_ankle_keypoints(boxes, keypoints, frame_h=480)
    # box 0: bottom should expand
    assert result[0, 1] == 10.0  # top unchanged
    assert result[1, 3] == 120.0  # box 1 unchanged


def test_expand_boxes_flat_aspect_filter() -> None:
    """flat_aspect_thresh should only expand flat boxes (h/w < thresh)."""
    # Flat box: h=30, w=100, h/w=0.3 (< 1.5)
    # Normal box: h=100, w=50, h/w=2.0 (> 1.5)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 100.0, 30.0],  # flat
            [0.0, 0.0, 50.0, 100.0],  # normal
        ]
    )
    keypoints = torch.tensor(
        [
            [[0.0, 0.0, 0.0]] * 15
            + [[50.0, 60.0, 0.5], [60.0, 60.0, 0.5]],  # ankle below flat box
            [[0.0, 0.0, 0.0]] * 15
            + [[25.0, 160.0, 0.5], [35.0, 160.0, 0.5]],  # ankle below normal box
        ]
    )
    result = expand_boxes_with_ankle_keypoints(
        boxes, keypoints, frame_h=480, flat_aspect_thresh=1.5
    )
    # Only flat box should expand
    assert result[0, 3] > 30.0  # flat box expanded
    assert result[1, 3] == 100.0  # normal box unchanged


def test_expand_boxes_custom_margin() -> None:
    """Margin parameter should affect expansion amount."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 80.0]])
    keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0]] * 15 + [[20.0, 100.0, 0.5], [30.0, 100.0, 0.5]]]
    )

    result_small = expand_boxes_with_ankle_keypoints(
        boxes, keypoints, frame_h=480, margin=0.0
    )
    result_large = expand_boxes_with_ankle_keypoints(
        boxes, keypoints, frame_h=480, margin=0.1
    )

    # margin=0.0: bottom = 100 + 0 * 70 = 100
    # margin=0.1: bottom = 100 + 0.1 * 70 = 107
    assert result_small[0, 3] == 100.0
    assert abs(result_large[0, 3] - 107.0) < 0.01


def test_expand_boxes_ankle_conf_threshold_custom() -> None:
    """Custom ankle_conf_thresh should change which keypoints are valid."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 80.0]])
    keypoints = torch.tensor(
        [[[0.0, 0.0, 0.0]] * 15 + [[20.0, 100.0, 0.4], [30.0, 100.0, 0.4]]]
    )

    # With default thresh=0.30, ankle is valid
    result_valid = expand_boxes_with_ankle_keypoints(
        boxes, keypoints, frame_h=480, ankle_conf_thresh=0.30
    )
    # With thresh=0.5, ankle is too low confidence
    result_invalid = expand_boxes_with_ankle_keypoints(
        boxes, keypoints, frame_h=480, ankle_conf_thresh=0.5
    )

    assert result_valid[0, 3] > 80.0  # expanded
    assert torch.equal(result_invalid, boxes)  # not expanded


# ── match_keypoints_to_boxes ───────────────────────────────────────────


def test_match_keypoints_empty_source() -> None:
    target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    source = torch.empty((0, 4), dtype=torch.float32)
    kps = torch.empty((0, 17, 3))
    result = match_keypoints_to_boxes(target, source, kps)
    assert result is None


def test_match_keypoints_empty_target() -> None:
    target = torch.empty((0, 4), dtype=torch.float32)
    source = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    kps = torch.randn(1, 17, 3)
    result = match_keypoints_to_boxes(target, source, kps)
    assert result is None


def test_match_keypoints_no_keypoints() -> None:
    target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    source = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    result = match_keypoints_to_boxes(target, source, None)
    assert result is None


def test_match_keypoints_single() -> None:
    """Single target matches single source by center distance."""
    target = torch.tensor([[50.0, 50.0, 60.0, 60.0]])  # center (55, 55)
    source = torch.tensor([[0.0, 0.0, 20.0, 20.0]])  # center (10, 10)
    kps = torch.tensor([[[1.0, 2.0, 3.0]]])
    result = match_keypoints_to_boxes(target, source, kps)
    # nearest index is 0, source_keypoints[[0]] returns first element
    assert result.shape[0] == 1  # one target box
    assert result.shape[-1] == 3  # 3 coords per keypoint
    assert torch.allclose(result.squeeze(), kps[0])  # matched kps[0]


def test_match_keypoints_nearest() -> None:
    """Target should match nearest source detection."""
    target = torch.tensor([[50.0, 50.0, 60.0, 60.0]])  # center (55, 55)
    # source[0]: center (5, 5), source[1]: center (50, 50)
    source = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [45.0, 45.0, 55.0, 55.0],
        ]
    )
    # 2 sources, 2 keypoints each, 3 coords per keypoint
    kps = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]],  # source 0 kps
            [[2.0, 0.0, 0.0], [20.0, 0.0, 0.0]],  # source 1 kps
        ]
    )
    result = match_keypoints_to_boxes(target, source, kps)
    assert result.shape[0] == 1  # one target
    assert result.shape[-1] == 3
    # source[1] is closer to target → should match source[1]
    assert result.squeeze()[0, 0].item() == 2.0


def test_match_keypoints_multiple_targets() -> None:
    """Each target independently matches nearest source."""
    target = torch.tensor(
        [
            [0.0, 0.0, 20.0, 20.0],  # center (10, 10)
            [40.0, 40.0, 60.0, 60.0],  # center (50, 50)
        ]
    )
    source = torch.tensor([[10.0, 10.0, 20.0, 20.0]])  # center (15, 15)
    kps = torch.tensor([[[100.0, 200.0, 300.0]]])
    result = match_keypoints_to_boxes(target, source, kps)
    assert result.shape[0] == 2  # two target boxes
    assert result.shape[-1] == 3
    # Both targets match the single source
    assert torch.allclose(result[0].squeeze(), kps[0])
    assert torch.allclose(result[1].squeeze(), kps[0])


def test_match_keypoints_multiple_sources() -> None:
    """Multiple targets each match their nearest source."""
    target = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # center (5, 5)
            [100.0, 100.0, 110.0, 110.0],  # center (105, 105)
        ]
    )
    source = torch.tensor(
        [
            [5.0, 5.0, 15.0, 15.0],  # center (10, 10) — closer to target 0
            [100.0, 100.0, 110.0, 110.0],  # center (105, 105) — closer to target 1
        ]
    )
    kps = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    result = match_keypoints_to_boxes(target, source, kps)
    assert result.shape[0] == 2  # two target boxes
    assert result.shape[-1] == 3
    assert result[0].squeeze()[0] == 1.0  # target 0 → source 0
    assert result[1].squeeze()[0] == 4.0  # target 1 → source 1


# ── _tile_seam_mask_for_boxes ──────────────────────────────────────────


def test_tile_seam_mask_no_boxes() -> None:
    boxes = torch.empty((0, 4), dtype=torch.float32)
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="960p_2x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert result.shape == (0,)
    assert result.dtype == torch.bool


def test_tile_seam_mask_non_tiled() -> None:
    """Non-tiled tiling should return all False."""
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="native_960",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert result.shape == (1,)
    assert not result.any()


def test_tile_seam_mask_2x2_crossing_seam() -> None:
    """Box that crosses the seam line should be marked."""
    # For frame_w=1920, frame_h=1080, r=0.5:
    # seams in orig coords: x=[640,1280], y=[220,860]
    # seam_margin_orig = 10/0.5 = 20
    boxes = torch.tensor([[630.0, 300.0, 650.0, 400.0]])  # crosses x=640 seam
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="960p_2x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert bool(result[0])


def test_tile_seam_mask_2x2_not_crossing() -> None:
    """Box far from seams should not be marked."""
    # Seams in orig coords: x=[640,1280], y=[220,860]
    boxes = torch.tensor([[50.0, 50.0, 100.0, 100.0]])  # far from all seams
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="960p_2x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert not bool(result[0])


def test_tile_seam_mask_near_seam_margin() -> None:
    """Box center near seam (within margin) should be marked."""
    # seam_margin_orig = 20; box center at (635, 300) is 5px from seam x=640
    boxes = torch.tensor(
        [[625.0, 300.0, 645.0, 400.0]]
    )  # center at x=635, 5px from seam
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="960p_2x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    # 5px < 20px margin → should be marked
    assert bool(result[0])


def test_tile_seam_mask_3x2_tiling() -> None:
    """3x2 tiling has seams at 160, 320, 640, 800."""
    # For 3x2: seam_x_canvas=[160,320,640,800], seam_y_canvas=[320,640]
    # r=0.5, x_off=0, y_off=210
    # seam_x_orig = [(160-0)/0.5, (320-0)/0.5, (640-0)/0.5, (800-0)/0.5] = [320,640,1280,1600]
    # seam_y_orig = [(320-210)/0.5, (640-210)/0.5] = [220,860]
    boxes = torch.tensor([[310.0, 300.0, 330.0, 400.0]])  # near x=320 seam
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="960p_3x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert bool(result[0])

    # Box in middle, far from all seams
    boxes_far = torch.tensor([[900.0, 300.0, 950.0, 400.0]])
    result_far = _tile_seam_mask_for_boxes(
        boxes_far,
        tiling="960p_3x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert not bool(result_far[0])


def test_tile_seam_mask_multiple_boxes() -> None:
    """Some boxes near seam, some not."""
    # Box 0 crosses x=640 seam, Box 1 is far from seams
    boxes = torch.tensor(
        [
            [630.0, 300.0, 650.0, 400.0],  # crosses x=640 seam → True
            [50.0, 50.0, 100.0, 100.0],  # far from seams → False
        ]
    )
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="960p_2x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert bool(result[0])
    assert not bool(result[1])


def test_tile_seam_mask_horizontal_seam() -> None:
    """Box crossing horizontal seam (y=220 in orig coords) should also be marked."""
    boxes = torch.tensor([[100.0, 215.0, 200.0, 225.0]])  # crosses y=220 seam
    result = _tile_seam_mask_for_boxes(
        boxes,
        tiling="960p_2x2",
        frame_w=1920,
        frame_h=1080,
        seam_margin_canvas_px=10.0,
    )
    assert bool(result[0])


# ── _get_detector_static_batch_size ────────────────────────────────────


def test_get_batch_size_dynamic() -> None:
    detector = MagicMock(is_dynamic=True)
    assert _get_detector_static_batch_size(detector) == 1


def test_get_batch_size_from_shape() -> None:
    detector = MagicMock()
    detector.is_dynamic = False
    detector.input_shape = (4, 3, 960, 960)
    assert _get_detector_static_batch_size(detector) == 4


def test_get_batch_size_from_engine() -> None:
    detector = MagicMock()
    detector.is_dynamic = False
    detector.input_shape = None
    mock_engine = MagicMock()
    mock_engine.get_tensor_shape.return_value = (8, 3, 960, 960)
    detector.engine = mock_engine
    detector.input_name = "input"
    assert _get_detector_static_batch_size(detector) == 8


def test_get_batch_size_no_shape() -> None:
    detector = MagicMock()
    detector.is_dynamic = False
    detector.input_shape = None
    # engine is not set (no 'engine' attribute)
    del detector.engine
    assert _get_detector_static_batch_size(detector) == 1


def test_get_batch_size_invalid_shape() -> None:
    detector = MagicMock()
    detector.is_dynamic = False
    detector.input_shape = "invalid"
    assert _get_detector_static_batch_size(detector) == 1


def test_get_batch_size_zero_batch() -> None:
    detector = MagicMock()
    detector.is_dynamic = False
    detector.input_shape = (0, 3, 960, 960)
    assert _get_detector_static_batch_size(detector) == 1  # max(1, 0) = 1


def test_get_batch_size_negative_batch() -> None:
    detector = MagicMock()
    detector.is_dynamic = False
    detector.input_shape = (-1, 3, 960, 960)
    assert _get_detector_static_batch_size(detector) == 1  # max(1, -1) = 1
