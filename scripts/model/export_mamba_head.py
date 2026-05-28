#!/usr/bin/env python3
"""Export PyTorch Mamba Head to TorchScript via wrapper.

Usage:
    uv run scripts/model/export_mamba_head.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# MUST import saccade_tracking_ext before torchvision/PIL to avoid shared library conflict
import saccade_tracking_ext  # noqa: F401

import torch
import torch.nn as nn
from saccade.perception.temporal_yolo.mamba_gated_detector import (
    build_mamba_gated_detector,
)


class MambaHeadWrapper(nn.Module):
    def __init__(self, mamba_head: nn.Module):
        super().__init__()
        self.mamba_head = mamba_head

    def forward(
        self, feats: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cls_preds, reg_preds = self.mamba_head(feats, return_embeddings=False)
        return cls_preds, reg_preds


def export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build student detector
    print("Building detector...")
    detector = build_mamba_gated_detector(
        yolo_pt_path=str(project_root / "models/yolo/yolo26s.pt"),
        teacher_ckpt=str(project_root / "runs/gated_det_v1/best.ckpt"),
        mamba_ckpt=str(project_root / "runs/mamba_gt_960_v2/best.ckpt"),
        img_size=640,
        device=device,
        emb_dim=128,
    )
    detector.eval()

    wrapper = MambaHeadWrapper(detector.mamba_head).eval()

    # Prepare dummy FPN features: P3, P4, P5
    dummy_fpn = [
        torch.zeros(1, 128, 80, 80, device=device),
        torch.zeros(1, 256, 40, 40, device=device),
        torch.zeros(1, 512, 20, 20, device=device),
    ]

    # Trace the mamba head wrapper
    print("Tracing MambaHeadWrapper to TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, [dummy_fpn], strict=False)

    output_path = project_root / "models/yolo/mamba_head_best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    print(f"🎉 Successfully exported and saved TorchScript head to: {output_path}")


if __name__ == "__main__":
    export()
