"""Tests for the FPN ReID CUDA extension (saccade_fpn_reid_cuda).

Covers the split binding: kernel launchers live in fpn_reid_cuda.cu (nvcc,
no torch headers) and the torch orchestration in fpn_reid_binding.cpp (host
compiler).  Verifies both the no-projection and projection+BN paths, the
L2 normalisation invariant, and the out_dim guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "build"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pytest.importorskip("saccade_fpn_reid_cuda", exc_type=ImportError)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

import saccade_fpn_reid_cuda as m  # noqa: E402


def _make_inputs(
    *, n: int = 3, c: int = 8, hw: int = 16, out_dim: int = 4
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    feats = [torch.randn(1, c, hw, hw, device="cuda")]
    conv_weights = [torch.randn(out_dim, c, device="cuda")]
    boxes = torch.tensor(
        [[2, 2, 10, 14], [4, 4, 8, 12], [1, 1, 15, 15]],
        dtype=torch.float32,
        device="cuda",
    )[:n]
    return feats, conv_weights, boxes


def test_module_exposes_extract() -> None:
    assert hasattr(m, "fpn_reid_extract")
    assert callable(m.fpn_reid_extract)


def test_no_proj_path_shape_and_l2() -> None:
    feats, cw, boxes = _make_inputs()
    empty = torch.empty(0, device="cuda")
    out = m.fpn_reid_extract(feats, cw, boxes, 16, empty, empty, empty, 1e-5)
    assert out.shape == (3, 4)
    assert out.dtype == torch.float32
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3, device="cuda"), atol=1e-4)


def test_proj_bn_path_shape_and_l2() -> None:
    feats, cw, boxes = _make_inputs(c=8, out_dim=4)
    # mid_dim = n_scales * out_dim = 4; proj maps mid_dim -> out_dim
    proj = torch.randn(4, 4, device="cuda")
    mean = torch.zeros(4, device="cuda")
    var = torch.ones(4, device="cuda")
    out = m.fpn_reid_extract(feats, cw, boxes, 16, proj, mean, var, 1e-5)
    assert out.shape == (3, 4)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3, device="cuda"), atol=1e-4)


def test_multi_scale_concat() -> None:
    # Two scales with different C; concatenated mid_dim = n_scales * out_dim.
    c0, c1, out_dim = 6, 8, 4
    feats = [
        torch.randn(1, c0, 16, 16, device="cuda"),
        torch.randn(1, c1, 16, 16, device="cuda"),
    ]
    cw = [
        torch.randn(out_dim, c0, device="cuda"),
        torch.randn(out_dim, c1, device="cuda"),
    ]
    boxes = torch.tensor([[2, 2, 10, 14]], dtype=torch.float32, device="cuda")
    # mid_dim = 2 * out_dim = 8; proj maps 8 -> out_dim=4
    proj = torch.randn(8, out_dim, device="cuda")
    mean = torch.zeros(out_dim, device="cuda")
    var = torch.ones(out_dim, device="cuda")
    out = m.fpn_reid_extract(feats, cw, boxes, 16, proj, mean, var, 1e-5)
    assert out.shape == (1, out_dim)
    assert torch.allclose(out.norm(dim=1), torch.ones(1, device="cuda"), atol=1e-4)


def test_out_dim_guard_rejects_oversized() -> None:
    # conv1x1/linear launch with out_dim threads/block; >1024 must be rejected.
    feats, _cw, boxes = _make_inputs(out_dim=2048)
    cw = [torch.randn(2048, 8, device="cuda")]
    empty = torch.empty(0, device="cuda")
    with pytest.raises(RuntimeError, match="out_dim"):
        m.fpn_reid_extract(feats, cw, boxes, 16, empty, empty, empty, 1e-5)


def test_out_dim_guard_rejects_zero() -> None:
    feats, _cw, boxes = _make_inputs(out_dim=0)
    cw = [torch.randn(0, 8, device="cuda")]
    empty = torch.empty(0, device="cuda")
    with pytest.raises(RuntimeError, match="out_dim"):
        m.fpn_reid_extract(feats, cw, boxes, 16, empty, empty, empty, 1e-5)


def test_zero_boxes_returns_empty() -> None:
    feats = [torch.randn(1, 8, 16, 16, device="cuda")]
    cw = [torch.randn(4, 8, device="cuda")]
    boxes = torch.empty((0, 4), dtype=torch.float32, device="cuda")
    empty = torch.empty(0, device="cuda")
    out = m.fpn_reid_extract(feats, cw, boxes, 16, empty, empty, empty, 1e-5)
    assert out.shape == (0, 4)
