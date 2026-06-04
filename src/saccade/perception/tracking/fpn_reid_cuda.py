"""
FPN ReID CUDA kernel — loads prebuilt libfpn_reid_cuda.so.

Provides fpn_reid_extract_cuda() as a drop-in replacement for
DimReduceHead.forward_boxes().
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

_project_root = Path(__file__).resolve().parents[4]
_build_dir = _project_root / "build/cuda_reid"

_mod = None

_so_path = _build_dir / "libfpn_reid_cuda.so"
if _so_path.exists():
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "saccade_fpn_reid_cuda", str(_so_path)
        )
        _mod = importlib.util.module_from_spec(spec)
        sys.modules["saccade_fpn_reid_cuda"] = _mod
        spec.loader.exec_module(_mod)
    except Exception as e:
        print(f"[FPN ReID CUDA] Load failed: {e}")
        _mod = None


def fpn_reid_extract_cuda(
    feats: list[torch.Tensor],
    conv_weights: list[torch.Tensor],
    boxes: torch.Tensor,
    img_size: int = 640,
    proj_weight: torch.Tensor | None = None,
    running_mean: torch.Tensor | None = None,
    running_var: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """CUDA-accelerated FPN centre-pool + 1x1 conv + project + L2."""
    if _mod is None:
        return _fpn_reid_extract_python(
            feats,
            conv_weights,
            boxes,
            img_size,
            proj_weight,
            running_mean,
            running_var,
            eps,
        )
    return _mod.fpn_reid_extract(
        feats,
        conv_weights,
        boxes,
        img_size,
        proj_weight if proj_weight is not None else torch.empty(0),
        running_mean if running_mean is not None else torch.empty(0),
        running_var if running_var is not None else torch.empty(0),
        eps,
    )


def _fpn_reid_extract_python(
    feats,
    conv_weights,
    boxes,
    img_size,
    proj_weight,
    running_mean,
    running_var,
    eps,
) -> torch.Tensor:
    import torch.nn.functional as F

    parts = []
    for feat, cw in zip(feats, conv_weights):
        w = cw.view(cw.shape[0], cw.shape[1], 1, 1)
        x = F.conv2d(feat.contiguous(), w)[0]
        fh, fw = x.shape[1], x.shape[2]
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        cx_idx = (cx / img_size * fw).long().clamp(0, fw - 1)
        cy_idx = (cy / img_size * fh).long().clamp(0, fh - 1)
        centre = x[:, cy_idx, cx_idx].mT
        parts.append(centre)

    pooled = torch.cat(parts, dim=1)

    if proj_weight is not None and proj_weight.numel() > 0:
        out = pooled @ proj_weight.T
        if running_mean is not None and running_var is not None:
            out = (out - running_mean) / torch.sqrt(running_var + eps)
    else:
        out = pooled

    return F.normalize(out, dim=1)
