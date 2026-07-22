"""
Tests for DALIMediaClient / DALIVideoPipeline (NVDEC GPU decode path).

Requires: CUDA device + nvidia-dali-cuda120 installed.
Run:  pytest tests/test_dali_pipeline.py -m dali -v
"""

# scope: media
# function: behavior
# lifecycle: active

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.dali]

# Skip the entire module when CUDA or DALI is unavailable.
pytest.importorskip(
    "nvidia.dali", reason="nvidia-dali not installed", exc_type=ImportError
)
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from saccade.media.dali_pipeline import DALIMediaClient  # noqa: E402

VIDEO = "datasets/MOT17_videos/MOT17-05-SDP.mp4"


@pytest.fixture(scope="module")
def client():
    c = DALIMediaClient(video_path=VIDEO, batch_size=1, device_id=0)
    ok = c.connect()
    if not ok:
        pytest.skip(f"Could not open video {VIDEO}")
    yield c
    c.release()


# ── connect ──────────────────────────────────────────────────────────────────


@pytest.mark.dali
def test_connect_returns_true():
    c = DALIMediaClient(video_path=VIDEO, batch_size=1)
    assert c.connect() is True
    c.release()


@pytest.mark.dali
def test_connect_missing_file_returns_false():
    c = DALIMediaClient(video_path="does_not_exist.mp4")
    assert c.connect() is False


# ── grab_tensor ───────────────────────────────────────────────────────────────


@pytest.mark.dali
def test_grab_tensor_returns_success(client: DALIMediaClient):
    ok, tensor = client.grab_tensor()
    assert ok is True
    assert tensor is not None


@pytest.mark.dali
def test_grab_tensor_shape(client: DALIMediaClient):
    _, tensor = client.grab_tensor()
    assert tensor is not None
    # batch=1 → [1, C, H, W]; sequence dim squeezed inside grab_tensor
    assert tensor.ndim == 4
    assert tensor.shape == (1, 3, 640, 640)


@pytest.mark.dali
def test_grab_tensor_dtype(client: DALIMediaClient):
    _, tensor = client.grab_tensor()
    assert tensor is not None
    assert tensor.dtype == torch.float32


@pytest.mark.dali
def test_grab_tensor_on_gpu(client: DALIMediaClient):
    _, tensor = client.grab_tensor()
    assert tensor is not None
    assert tensor.is_cuda


@pytest.mark.dali
def test_grab_tensor_value_range(client: DALIMediaClient):
    _, tensor = client.grab_tensor()
    assert tensor is not None
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


# ── grab (CPU numpy) ──────────────────────────────────────────────────────────


@pytest.mark.dali
def test_grab_cpu_returns_success(client: DALIMediaClient):
    ok, arr = client.grab()
    assert ok is True
    assert arr is not None


@pytest.mark.dali
def test_grab_cpu_shape(client: DALIMediaClient):
    _, arr = client.grab()
    assert arr is not None
    assert arr.ndim == 3
    assert arr.shape == (640, 640, 3)


@pytest.mark.dali
def test_grab_cpu_dtype(client: DALIMediaClient):
    import numpy as np

    _, arr = client.grab()
    assert arr is not None
    assert arr.dtype == np.uint8


# ── multi-frame ───────────────────────────────────────────────────────────────


@pytest.mark.dali
def test_multiple_frames_consistent_shape(client: DALIMediaClient):
    shapes = []
    for _ in range(10):
        ok, tensor = client.grab_tensor()
        if not ok:
            break
        assert tensor is not None
        shapes.append(tensor.shape)
    assert len(shapes) == 10
    assert all(s == (1, 3, 640, 640) for s in shapes)


# ── release ───────────────────────────────────────────────────────────────────


@pytest.mark.dali
def test_release_cleans_up():
    c = DALIMediaClient(video_path=VIDEO, batch_size=1)
    c.connect()
    c.release()
    assert c.iterator is None
    assert c.pipeline is None
