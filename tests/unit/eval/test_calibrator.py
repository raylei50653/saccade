"""Tests for SaccadeInt8Calibrator (perception/calibrator.py).

Covers:
  - __init__ parameter storage
  - get_batch_size
  - read_calibration_cache / write_calibration_cache (file I/O)
  - get_batch with mock images
"""

# scope: perception
# function: behavior
# lifecycle: active

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from saccade.perception.calibrator import SaccadeInt8Calibrator


def _has_gpu() -> bool:
    """Check if a CUDA GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_gpu(),
    reason="No NVIDIA GPU available (required for INT8 calibrator)",
)
@pytest.mark.gpu
class TestSaccadeInt8CalibratorInit:
    """Test __init__ parameter storage."""

    def test_init_stores_cache_file(self, tmp_path: tempfile.TempPathFactory) -> None:
        cache_file = str(tmp_path / "cal.bin")
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=cache_file,
        )
        assert cal.cache_file == cache_file

    def test_init_stores_batch_size(self) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir="/tmp",
            cache_file="/tmp/cal.bin",
            batch_size=16,
        )
        assert cal.batch_size == 16

    def test_init_stores_input_shape(self) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir="/tmp",
            cache_file="/tmp/cal.bin",
            input_shape=(384, 384),
        )
        assert cal.input_shape == (384, 384)

    def test_init_default_batch_size(self) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir="/tmp",
            cache_file="/tmp/cal.bin",
        )
        assert cal.batch_size == 8

    def test_init_default_input_shape(self) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir="/tmp",
            cache_file="/tmp/cal.bin",
        )
        assert cal.input_shape == (640, 640)

    def test_init_limits_images_to_200(
        self, tmp_path: tempfile.TempPathFactory
    ) -> None:
        # Create 250 fake jpg files
        for i in range(250):
            (tmp_path / f"img_{i:04d}.jpg").touch()
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=str(tmp_path / "cal.bin"),
        )
        assert cal.count == 200
        assert len(cal.images) == 200

    def test_init_counts_non_jpg_files(
        self, tmp_path: tempfile.TempPathFactory
    ) -> None:
        # Create jpg and non-jpg files
        (tmp_path / "img1.jpg").touch()
        (tmp_path / "img2.png").touch()
        (tmp_path / "img3.jpg").touch()
        (tmp_path / "img4.txt").touch()
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=str(tmp_path / "cal.bin"),
        )
        assert cal.count == 2
        assert len(cal.images) == 2


@pytest.mark.skipif(
    not _has_gpu(),
    reason="No NVIDIA GPU available (required for INT8 calibrator)",
)
@pytest.mark.gpu
class TestCalibratorBatchSize:
    """Test get_batch_size."""

    def test_get_batch_size_returns_value(self) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir="/tmp",
            cache_file="/tmp/cal.bin",
            batch_size=12,
        )
        assert cal.get_batch_size() == 12


@pytest.mark.skipif(
    not _has_gpu(),
    reason="No NVIDIA GPU available (required for INT8 calibrator)",
)
@pytest.mark.gpu
class TestCalibratorCacheFileIO:
    """Test read_calibration_cache / write_calibration_cache."""

    def test_write_and_read_cache(self, tmp_path: tempfile.TempPathFactory) -> None:
        cache_file = str(tmp_path / "cal.bin")
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=cache_file,
        )
        cache_data = b"calibration data here"
        cal.write_calibration_cache(cache_data)
        result = cal.read_calibration_cache()
        assert result == cache_data

    def test_read_nonexistent_cache(self, tmp_path: tempfile.TempPathFactory) -> None:
        cache_file = str(tmp_path / "nonexistent.bin")
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=cache_file,
        )
        assert cal.read_calibration_cache() is None

    def test_write_overwrites_existing_cache(
        self, tmp_path: tempfile.TempPathFactory
    ) -> None:
        cache_file = str(tmp_path / "cal.bin")
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=cache_file,
        )
        cal.write_calibration_cache(b"first")
        assert cal.read_calibration_cache() == b"first"
        cal.write_calibration_cache(b"second")
        assert cal.read_calibration_cache() == b"second"

    def test_write_cache_creates_file(self, tmp_path: tempfile.TempPathFactory) -> None:
        cache_file = str(tmp_path / "cal.bin")
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=cache_file,
        )
        assert not os.path.exists(cache_file)
        cal.write_calibration_cache(b"data")
        assert os.path.exists(cache_file)


@pytest.mark.skipif(
    not _has_gpu(),
    reason="No NVIDIA GPU available (required for INT8 calibrator)",
)
@pytest.mark.gpu
class TestCalibratorGetBatch:
    """Test get_batch with mocked images."""

    @pytest.fixture
    def mock_image_dir(self, tmp_path: tempfile.TempPathFactory) -> str:
        """Create 10 fake jpg files for testing."""
        for i in range(10):
            (tmp_path / f"img_{i:04d}.jpg").touch()
        return str(tmp_path)

    def test_get_batch_returns_none_when_exhausted(self, mock_image_dir: str) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir=mock_image_dir,
            cache_file="/tmp/cal.bin",
            batch_size=8,
        )
        # First call consumes 8 images
        cal.current_index = 8
        assert cal.get_batch(None) is None

    def test_get_batch_returns_none_when_no_valid_images(
        self, tmp_path: tempfile.TempPathFactory
    ) -> None:
        """When cv2.imread returns None for all images, get_batch returns None."""
        # Create 0 jpg files, or images that can't be read
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=str(tmp_path / "cal.bin"),
            batch_size=4,
        )
        assert cal.count == 0
        assert cal.get_batch(None) is None

    def test_get_batch_advances_index(self, mock_image_dir: str) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir=mock_image_dir,
            cache_file="/tmp/cal.bin",
            batch_size=3,
        )
        # Create valid image files for cv2 to read
        import cv2 as _cv2

        blank = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(10):
            _cv2.imwrite(mock_image_dir + f"/img_{i:04d}.jpg", blank)
        # Recalculate images and count
        cal.images = [
            os.path.join(mock_image_dir, f)
            for f in os.listdir(mock_image_dir)
            if f.endswith(".jpg")
        ][:200]
        cal.count = len(cal.images)

        cal.get_batch(None)
        assert cal.current_index == 3
        cal.get_batch(None)
        assert cal.current_index == 6
        # 6 + 3 = 9 <= 10, so returns a batch
        cal.get_batch(None)
        assert cal.current_index == 9
        # 9 + 3 = 12 > 10, so returns None
        assert cal.get_batch(None) is None


@pytest.mark.skipif(
    not _has_gpu(),
    reason="No NVIDIA GPU available (required for INT8 calibrator)",
)
@pytest.mark.gpu
class TestCalibratorWithMockedCUDA:
    """Test that init allocates CUDA memory."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not available",
    )
    def test_init_allocates_cuda_memory(
        self, tmp_path: tempfile.TempPathFactory
    ) -> None:
        """Init should allocate CUDA memory for the input buffer."""
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=str(tmp_path / "cal.bin"),
            batch_size=4,
        )
        # device_input should be a non-zero pointer (int)
        assert isinstance(cal.device_input, int)
        assert cal.device_input != 0

    def test_init_sets_current_batch_none(
        self, tmp_path: tempfile.TempPathFactory
    ) -> None:
        cal = SaccadeInt8Calibrator(
            image_dir=str(tmp_path),
            cache_file=str(tmp_path / "cal.bin"),
        )
        assert cal.current_batch is None
