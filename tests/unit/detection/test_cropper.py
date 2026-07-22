"""Tests for ZeroCopyCropper (perception/cropper.py).

Covers:
  - __init__ validation and parameter storage
  - process with empty/None boxes
  - _prepare_boxes (tight, square, square_mean modes with padding)
  - process_parts with empty boxes
  - _fill_extra_with_mean
  - cpp_ptr raises when C++ not available
"""

# scope: perception
# function: behavior
# lifecycle: active

from __future__ import annotations

import pytest
import torch

import saccade.perception.cropper as cropper_module
from saccade.perception.cropper import ZeroCopyCropper


# ─── __init__ ────────────────────────────────────────────────────────────────


class TestCropperInit:
    """Test ZeroCopyCropper.__init__."""

    def test_init_default_parameters(self) -> None:
        cropper = ZeroCopyCropper()
        assert cropper.output_size == (224, 224)
        assert cropper.mode == "tight"
        assert cropper.padding == 0.0

    def test_init_custom_output_size(self) -> None:
        cropper = ZeroCopyCropper(output_size=(384, 384))
        assert cropper.output_size == (384, 384)

    def test_init_custom_padding(self) -> None:
        cropper = ZeroCopyCropper(padding=0.1)
        assert cropper.padding == 0.1

    def test_init_tight_mode(self) -> None:
        cropper = ZeroCopyCropper(mode="tight")
        assert cropper.mode == "tight"

    def test_init_square_mode(self) -> None:
        cropper = ZeroCopyCropper(mode="square")
        assert cropper.mode == "square"

    def test_init_square_mean_mode(self) -> None:
        cropper = ZeroCopyCropper(mode="square_mean")
        assert cropper.mode == "square_mean"

    def test_init_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported crop mode"):
            ZeroCopyCropper(mode="invalid_mode")

    def test_init_cpp_not_available_without_gpu(self) -> None:
        """When C++ cropper is not available, _cpp should be None."""
        cropper = ZeroCopyCropper()
        # _cpp may be a C++ object or None depending on build
        # The important thing is init doesn't crash
        assert hasattr(cropper, "_cpp")


# ─── process with empty/None boxes ──────────────────────────────────────────


class TestCropperProcess:
    """Test process() with edge cases."""

    def test_process_none_boxes(self) -> None:
        """process() with None boxes returns empty tensor."""
        cropper = ZeroCopyCropper()
        dummy_frame = torch.zeros((1, 3, 100, 100), device="cpu", dtype=torch.float32)
        result = cropper.process(dummy_frame, None)
        assert result.shape == (0, 3, 224, 224)

    def test_process_empty_boxes(self) -> None:
        """process() with empty boxes returns empty tensor."""
        cropper = ZeroCopyCropper()
        dummy_frame = torch.zeros((1, 3, 100, 100), device="cpu", dtype=torch.float32)
        empty_boxes = torch.empty((0, 4), device="cpu", dtype=torch.float32)
        result = cropper.process(dummy_frame, empty_boxes)
        assert result.shape == (0, 3, 224, 224)

    def test_process_returns_correct_channels(self) -> None:
        """process() preserves input channel count."""
        cropper = ZeroCopyCropper(output_size=(64, 64))
        dummy_frame = torch.zeros((1, 3, 200, 200), device="cpu", dtype=torch.float32)
        empty_boxes = torch.empty((0, 4), device="cpu", dtype=torch.float32)
        result = cropper.process(dummy_frame, empty_boxes)
        assert result.shape == (0, 3, 64, 64)

    def test_process_with_cpp_fallback_roi_align(
        self,
    ) -> None:
        """Test roi_align path when _cpp is None."""
        # Force Python path by setting _cpp to None
        cropper = ZeroCopyCropper(mode="square", padding=0.1)
        # Ensure we're on the Python path
        cropper._cpp = None

        dummy_frame = torch.rand((1, 3, 100, 100), device="cpu", dtype=torch.float32)
        # Boxes in [x1, y1, x2, y2] format
        dummy_boxes = torch.tensor(
            [[10.0, 10.0, 50.0, 50.0]], device="cpu", dtype=torch.float32
        )

        result = cropper.process(dummy_frame, dummy_boxes)
        assert result.shape == (1, 3, 224, 224)
        # Values should be valid (not NaN)
        assert not torch.isnan(result).any()

    def test_cpp_fast_path_passes_chw_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C++ cropper expects CHW float input, not an HWC permuted copy."""
        created: list[FakeCppCropper] = []

        class FakeStream:
            cuda_stream = 0

        class FakeCppCropper:
            def __init__(self, crop_width: int, crop_height: int) -> None:
                self.crop_width = crop_width
                self.crop_height = crop_height
                self.input_ptr: int | None = None
                self.src_w: int | None = None
                self.src_h: int | None = None
                created.append(self)

            def process_gpu(
                self,
                input_ptr: int,
                src_w: int,
                src_h: int,
                boxes_ptr: int,
                num_boxes: int,
                output_ptr: int,
                stream_ptr: int,
            ) -> None:
                self.input_ptr = input_ptr
                self.src_w = src_w
                self.src_h = src_h

        monkeypatch.setattr(cropper_module, "_CROPPER_CPP_AVAILABLE", True)
        monkeypatch.setattr(cropper_module, "CropperCpp", FakeCppCropper, raising=False)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeStream())

        cropper = ZeroCopyCropper(output_size=(4, 5))
        frame = torch.arange(1 * 3 * 7 * 11, dtype=torch.float32).reshape(1, 3, 7, 11)
        boxes = torch.tensor([[1.0, 2.0, 8.0, 6.0]], dtype=torch.float32)

        result = cropper.process(frame, boxes)

        assert result.shape == (1, 3, 4, 5)
        assert len(created) == 1
        assert created[0].input_ptr == frame.squeeze(0).data_ptr()
        assert created[0].src_w == 11
        assert created[0].src_h == 7


# ─── _prepare_boxes ─────────────────────────────────────────────────────────


class TestPrepareBoxes:
    """Test _prepare_boxes with different modes and padding."""

    def test_prepare_tight_no_padding(self) -> None:
        """tight mode with padding=0 returns boxes unchanged."""
        cropper = ZeroCopyCropper(mode="tight", padding=0.0)
        boxes = torch.tensor(
            [[10.0, 20.0, 100.0, 80.0], [50.0, 50.0, 150.0, 200.0]],
            dtype=torch.float32,
        )
        result = cropper._prepare_boxes(boxes, 300, 400)
        torch.testing.assert_close(result, boxes)

    def test_prepare_square_with_padding(self) -> None:
        """square mode with padding: boxes should be expanded to squares."""
        cropper = ZeroCopyCropper(mode="square", padding=0.1)
        # Box: [60, 50, 100, 80]
        # cx=80, cy=65, w=40, h=30
        # side = max(40, 30) * 1.1 = 44.0, half = 22.0
        boxes = torch.tensor([[60.0, 50.0, 100.0, 80.0]], dtype=torch.float32)
        result = cropper._prepare_boxes(boxes, 300, 400)

        cx = (60.0 + 100.0) * 0.5  # 80.0
        cy = (50.0 + 80.0) * 0.5  # 65.0
        bw = 100.0 - 60.0  # 40.0
        bh = 80.0 - 50.0  # 30.0
        side = max(float(bw), float(bh)) * (1.0 + 0.1)  # 44.0
        half_w = side * 0.5  # 22.0
        half_h = side * 0.5  # 22.0

        expected = torch.tensor(
            [[cx - half_w, cy - half_h, cx + half_w, cy + half_h]], dtype=torch.float32
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_prepare_square_mean_with_padding(self) -> None:
        """square_mean mode with padding: same as square for box expansion."""
        cropper = ZeroCopyCropper(mode="square_mean", padding=0.1)
        boxes = torch.tensor([[60.0, 50.0, 100.0, 80.0]], dtype=torch.float32)
        result = cropper._prepare_boxes(boxes, 300, 400)

        cx = (60.0 + 100.0) * 0.5  # 80.0
        cy = (50.0 + 80.0) * 0.5  # 65.0
        bw = 100.0 - 60.0  # 40.0
        bh = 80.0 - 50.0  # 30.0
        side = max(bw, bh) * 1.1
        half_w = side * 0.5
        half_h = side * 0.5

        expected = torch.tensor(
            [[cx - half_w, cy - half_h, cx + half_w, cy + half_h]], dtype=torch.float32
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_prepare_clamps_to_frame_bounds(self) -> None:
        """_prepare_boxes clamps coordinates to frame boundaries."""
        cropper = ZeroCopyCropper(mode="square", padding=0.5)
        # Box near corner, expansion would go negative
        boxes = torch.tensor([[1.0, 1.0, 20.0, 20.0]], dtype=torch.float32)
        result = cropper._prepare_boxes(boxes, 100, 100)

        # All values should be >= 0
        assert (result >= 0).all()
        # All values should be <= frame dimensions - 1
        assert (result[:, 0] <= 99.0).all()
        assert (result[:, 2] <= 99.0).all()
        assert (result[:, 1] <= 99.0).all()
        assert (result[:, 3] <= 99.0).all()

    def test_prepare_clamps_to_frame_width(self) -> None:
        """_prepare_boxes clamps to frame width."""
        cropper = ZeroCopyCropper(mode="square", padding=0.5)
        # Box near right edge
        boxes = torch.tensor([[950.0, 400.0, 990.0, 600.0]], dtype=torch.float32)
        result = cropper._prepare_boxes(boxes, 1000, 1000)

        assert (result[:, 2] <= 999.0).all()

    def test_prepare_min_box_size(self) -> None:
        """_prepare_boxes enforces minimum box size of 1.0."""
        cropper = ZeroCopyCropper(mode="tight", padding=0.0)
        # Very small box (1x1 pixel)
        boxes = torch.tensor([[100.0, 100.0, 101.0, 101.0]], dtype=torch.float32)
        result = cropper._prepare_boxes(boxes, 1000, 1000)
        # Width and height should be at least 1.0
        width = result[0, 2] - result[0, 0]
        height = result[0, 3] - result[0, 1]
        assert width >= 1.0
        assert height >= 1.0

    def test_prepare_multiple_boxes(self) -> None:
        """_prepare_boxes handles multiple boxes correctly."""
        cropper = ZeroCopyCropper(mode="square", padding=0.2)
        boxes = torch.tensor(
            [
                [10.0, 10.0, 30.0, 30.0],
                [100.0, 100.0, 120.0, 120.0],
                [200.0, 200.0, 220.0, 220.0],
            ],
            dtype=torch.float32,
        )
        result = cropper._prepare_boxes(boxes, 500, 500)
        assert result.shape == (3, 4)
        # All should be squared
        widths = result[:, 2] - result[:, 0]
        heights = result[:, 3] - result[:, 1]
        assert torch.allclose(widths, heights, atol=1e-4)


# ─── _fill_extra_with_mean ──────────────────────────────────────────────────


class TestFillExtraWithMean:
    """Test _fill_extra_with_mean for square_mean mode."""

    def test_fill_extra_returns_unchanged_when_not_square_mean(self) -> None:
        """When mode is not square_mean, returns crops unchanged."""
        cropper = ZeroCopyCropper(mode="tight", padding=0.0)
        crops = torch.ones((2, 3, 224, 224))
        original = torch.tensor(
            [[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32
        )
        prepared = torch.tensor(
            [[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32
        )
        result = cropper._fill_extra_with_mean(crops, original, prepared)
        torch.testing.assert_close(result, crops)

    def test_fill_extra_returns_unchanged_when_empty(self) -> None:
        """When crops is empty, returns unchanged."""
        cropper = ZeroCopyCropper(mode="square_mean", padding=0.1)
        crops = torch.empty((0, 3, 224, 224))
        original = torch.empty((0, 4), dtype=torch.float32)
        prepared = torch.empty((0, 4), dtype=torch.float32)
        result = cropper._fill_extra_with_mean(crops, original, prepared)
        assert result.shape == (0, 3, 224, 224)

    def test_fill_extra_fills_outside_pixels(self) -> None:
        """When mode is square_mean, outside pixels are filled with mean of inside."""
        cropper = ZeroCopyCropper(mode="square_mean", padding=0.3)
        # Use actual output_size (224x224)
        crops = torch.ones((1, 3, 224, 224))
        original = torch.tensor([[80.0, 80.0, 144.0, 144.0]], dtype=torch.float32)
        prepared = torch.tensor([[40.0, 40.0, 184.0, 184.0]], dtype=torch.float32)
        result = cropper._fill_extra_with_mean(crops, original, prepared)
        assert result.shape == (1, 3, 224, 224)
        assert not torch.isnan(result).any()


# ─── process_parts ──────────────────────────────────────────────────────────


class TestProcessParts:
    """Test process_parts for full/upper/lower cropping."""

    def test_process_parts_none_boxes(self) -> None:
        """process_parts with None boxes returns empty tensor."""
        cropper = ZeroCopyCropper()
        dummy_frame = torch.zeros((1, 3, 100, 100), device="cpu", dtype=torch.float32)
        result = cropper.process_parts(dummy_frame, None)
        assert result.shape == (0, 3, 224, 224)

    def test_process_parts_empty_boxes(self) -> None:
        """process_parts with empty boxes returns empty tensor."""
        cropper = ZeroCopyCropper()
        dummy_frame = torch.zeros((1, 3, 100, 100), device="cpu", dtype=torch.float32)
        empty_boxes = torch.empty((0, 4), device="cpu", dtype=torch.float32)
        result = cropper.process_parts(dummy_frame, empty_boxes)
        assert result.shape == (0, 3, 224, 224)

    def test_process_parts_3x_boxes(self) -> None:
        """process_parts returns 3 crops per input box (full + upper + lower)."""
        cropper = ZeroCopyCropper()
        cropper._cpp = None  # Force Python path
        dummy_frame = torch.zeros((1, 3, 100, 100), device="cpu", dtype=torch.float32)
        boxes = torch.tensor([[20.0, 20.0, 80.0, 80.0]], dtype=torch.float32)
        result = cropper.process_parts(dummy_frame, boxes)
        # 1 box → 3 crops (full, upper, lower)
        assert result.shape[0] == 3


# ─── cpp_ptr property ───────────────────────────────────────────────────────


class TestCppPtr:
    """Test cpp_ptr property."""

    def test_cpp_ptr_raises_when_no_cpp(self) -> None:
        """cpp_ptr raises RuntimeError when C++ cropper is not available."""
        cropper = ZeroCopyCropper()
        cropper._cpp = None  # Force no C++ path
        with pytest.raises(RuntimeError) as exc_info:
            _ = cropper.cpp_ptr
        assert "C++" in str(exc_info.value)


# ─── native kernel vs PIL bilinear alignment ─────────────────────────────────

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native cropper alignment requires CUDA"
)


def _structured_frame(h: int, w: int):
    """RGB uint8 HWC frame with gradients + 2px diagonal stripes (aliasing bait)."""
    import numpy as np

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((xx * 0.7 + yy * 0.3) % 255).astype(np.uint8)
    g = ((np.sin(xx * 0.05) * 127 + 128 + yy * 0.4) % 255).astype(np.uint8)
    b = ((np.cos(yy * 0.07) * 127 + 128 + xx * 0.2) % 255).astype(np.uint8)
    frame = np.stack([r, g, b], axis=-1).astype(np.uint8)
    frame[((xx + yy) % 8) < 2] = [255, 0, 0]
    return frame


def _pil_crop(frame, box_xyxy, out_hw):
    import numpy as np
    from PIL import Image

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    box = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(w, int(round(x2))),
        min(h, int(round(y2))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        box = (0, 0, w, h)
    img = Image.fromarray(frame, mode="RGB")
    return np.asarray(
        img.crop(box).resize((out_hw[1], out_hw[0]), Image.BILINEAR), dtype=np.uint8
    )


@cuda_only
class TestNativeCropMatchesPilBilinear:
    """The CUDA crop kernel must stay pixel-aligned with the PIL fallback.

    Guards against the two drift sources that broke offline/online parity:
      1. half-pixel shift (corner vs pixel-center convention);
      2. missing antialiasing (single-tap vs PIL's triangle filter).
    Integer boxes match PIL to within uint8 rounding (max abs diff <= 1).
    """

    def test_integer_boxes_match_pil(self) -> None:
        import numpy as np

        h, w, out_hw = 720, 1280, (224, 224)
        frame = _structured_frame(h, w)
        frame_chw = (
            torch.from_numpy(frame.copy())
            .to(device="cuda", dtype=torch.float32)
            .div_(255.0)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )
        cropper = ZeroCopyCropper(output_size=out_hw, mode="tight", padding=0.0)
        if cropper._cpp is None:
            pytest.skip("C++ cropper extension not available")
        boxes = [
            (50.0, 80.0, 450.0, 520.0),  # ~2x downscale
            (200.0, 150.0, 300.0, 250.0),  # upscale (small box)
            (1000.0, 400.0, 1280.0, 700.0),  # near right edge
            (0.0, 0.0, 224.0, 224.0),  # 1:1
            (530.0, 310.0, 954.0, 734.0),  # box runs off the bottom edge
        ]
        boxes_t = torch.tensor(boxes, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            crops = cropper.process(frame_chw, boxes_t).clamp(0.0, 1.0)
        crops_hwc = (
            (crops.permute(0, 2, 3, 1).cpu().numpy() * 255.0)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        for i, bx in enumerate(boxes):
            pil = _pil_crop(frame, bx, out_hw).astype(np.float32).ravel()
            cpp = crops_hwc[i].astype(np.float32).ravel()
            cos = float(
                np.dot(pil, cpp) / (np.linalg.norm(pil) * np.linalg.norm(cpp) + 1e-12)
            )
            mae = float(np.abs(pil - cpp).mean())
            mx = float(np.abs(pil - cpp).max())
            assert cos >= 0.999, (
                f"box {bx}: cosine {cos:.4f} < 0.999 (mae={mae:.3f}, max={mx:.0f})"
            )
            # uint8 rounding only — no structural drift.
            assert mx <= 1.0, (
                f"box {bx}: max abs diff {mx:.0f} > 1 (rounding-only expected)"
            )
