"""Unit tests for GPU (nvJPEG) clip decoding used by the full-cadence GT trainer.

The full-cadence PersonPath22 frame set (1080p, ~78k frames) is too large to
RAM-preload, and CPU JPEG decode bottlenecks the loop. In --gpu-decode mode the
DataLoader workers return raw JPEG bytes (no decode, CUDA-free) and the train
loop batch-decodes them on the GPU. This checks the decode helper produces the
expected (B, T, 3, S, S) float tensor in [0, 255], regardless of source aspect
ratio (stretch-resize matches the non-letterbox loader's GT scaling).
"""

# scope: detection
# function: behavior
# lifecycle: active

import pytest
import torch

from saccade.perception.temporal_yolo.dataset import gpu_decode_clip_batch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="nvJPEG GPU decode requires CUDA"
)


def _jpeg_bytes(h: int, w: int) -> bytes:
    from torchvision.io import encode_jpeg

    img = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
    return encode_jpeg(img).numpy().tobytes()


@cuda_only
def test_shape_dtype_range():
    # B=2 clips, T=3 frames; differing source resolutions per frame.
    batch = [
        [_jpeg_bytes(1080, 1920), _jpeg_bytes(480, 640), _jpeg_bytes(720, 720)],
        [_jpeg_bytes(200, 300), _jpeg_bytes(1080, 1920), _jpeg_bytes(640, 480)],
    ]
    out = gpu_decode_clip_batch(batch, img_size=640, device="cuda")
    assert out.shape == (2, 3, 3, 640, 640)
    assert out.dtype == torch.float32
    assert out.is_cuda
    assert 0.0 <= float(out.min()) and float(out.max()) <= 255.0


@cuda_only
def test_batch_major_ordering():
    # A distinctly-bright frame placed at (clip 1, t 2) must land at out[1, 2].
    dark = _jpeg_bytes(64, 64)
    from torchvision.io import encode_jpeg

    bright = (
        encode_jpeg(torch.full((3, 64, 64), 255, dtype=torch.uint8)).numpy().tobytes()
    )
    batch = [[dark, dark, dark], [dark, dark, bright]]
    out = gpu_decode_clip_batch(batch, img_size=64, device="cuda")
    means = out.mean(dim=(2, 3, 4))  # (B, T)
    assert means[1, 2] == means.max()
    assert means[1, 2] > 250.0
