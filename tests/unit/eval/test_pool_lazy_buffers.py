"""Lazy acquire / release of AdaptiveFramePool tiling buffers.

The 960 / tile / flip-TTA scratch buffers (~71 MB) are only used by the
high-res / tiled / TTA detection paths, never by the default 640 whole_graph
path. They must be allocated lazily on first access and freed by
release_tiling_buffers() so single- and double-buffer 640 eval never pays for
VRAM it does not use.
"""

# scope: eval
# function: behavior
# lifecycle: active

import pytest
import torch

from saccade.perception.eval.pool import AdaptiveFramePool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _alloc_mb() -> float:
    return torch.cuda.memory_allocated() / 1e6


def test_tiling_buffers_not_allocated_until_accessed() -> None:
    pool = AdaptiveFramePool(1080, 1920)
    # Eager buffers (frame_buffer, canvas_640p, nv12) exist; tiling ones do not.
    assert pool._canvas_960p is None
    assert pool._canvas_960p_flip is None
    assert pool._tiles_batch4 is None
    assert pool._tiles_batch6 is None


def test_lazy_acquire_then_release_roundtrips_vram() -> None:
    base = _alloc_mb()
    pool = AdaptiveFramePool(1080, 1920)
    only_640 = _alloc_mb() - base

    # First access acquires each tiling buffer (申請).
    assert pool.canvas_960p.shape == (3, 960, 960)
    assert pool.canvas_960p_flip.shape == (3, 960, 960)
    assert pool.tiles_batch4.shape == (4, 3, 640, 640)
    assert pool.tiles_batch6.shape == (6, 3, 640, 640)
    with_tiling = _alloc_mb() - base
    assert with_tiling > only_640 + 60.0  # ~71 MB of scratch added

    # Release frees them (釋放); next access re-acquires transparently.
    pool.release_tiling_buffers(empty_cache=True)
    assert pool._canvas_960p is None
    released = _alloc_mb() - base
    assert released == pytest.approx(only_640, abs=1.0)

    reacquired = pool.tiles_batch6
    assert reacquired.shape == (6, 3, 640, 640)


def test_lazy_buffer_returns_same_instance_until_released() -> None:
    pool = AdaptiveFramePool(64, 64)
    a = pool.canvas_960p
    b = pool.canvas_960p
    assert a is b
    pool.release_tiling_buffers()
    c = pool.canvas_960p
    assert c is not a
