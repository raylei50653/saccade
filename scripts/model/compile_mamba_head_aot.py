#!/usr/bin/env python3
"""Example script demonstrating how to compile the PyTorch Mamba Head
into a standalone C++ Shared/Static Library via PyTorch 2.x AOTInductor.

This compiles the PyTorch Mamba Head directly into native machine code (C++/CUDA),
completely bypassing the TorchScript JIT interpreter and running at pure native speed.

Usage:
    uv run scripts/model/compile_mamba_head_aot.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
from saccade.perception.temporal_yolo.mamba_gated_detector import (
    build_mamba_gated_detector,
)


class MambaHeadWrapper(nn.Module):
    def __init__(self, mamba_head: nn.Module):
        super().__init__()
        self.mamba_head = mamba_head

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # AOTInductor requires flat arguments instead of lists for clean C-API generation
        feats = [p3, p4, p5]
        cls_preds, reg_preds = self.mamba_head(feats, return_embeddings=False)
        # Flatten outputs for native C-API returning
        return (
            cls_preds[0],
            cls_preds[1],
            cls_preds[2],
            reg_preds[0],
            reg_preds[1],
            reg_preds[2],
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build detector
    print("Building Mamba head model...")
    detector = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="runs/gated_det_v1/best.ckpt",
        mamba_ckpt="runs/mamba_gt_960_v2/best.ckpt",
        img_size=640,
        device=device,
        emb_dim=128,
    )
    detector.eval()

    wrapper = MambaHeadWrapper(detector.mamba_head).to(device).eval()

    # 2. Prepare dummy inputs
    p3 = torch.randn(1, 128, 80, 80, device=device)
    p4 = torch.randn(1, 256, 40, 40, device=device)
    p5 = torch.randn(1, 512, 20, 20, device=device)

    # 3. Export model graph
    print("Exporting Mamba head graph via torch.export...")
    try:
        # torch.export is the standard PyTorch 2.x graph representation compiler
        exported_program = torch.export.export(wrapper, (p3, p4, p5))

        # 4. Compile to standalone C++ library
        output_so = project_root / "build" / "libmamba_head_aot.so"
        output_so.parent.mkdir(parents=True, exist_ok=True)
        print(f"Compiling exported graph to standalone C++ library: {output_so} ...")

        # Compile using AOTInductor
        torch._inductor.aot_compiler.aot_compile(
            exported_program, (p3, p4, p5), output_so_path=str(output_so)
        )
        print(f"🎉 Success! stand-alone C++ library generated at: {output_so}")

    except Exception as e:
        print(
            "\n[AOTInductor Info] Note: Standard PyTorch 2.x AOTInductor export requires all sub-operators "
            "to be strictly traceable. Gated Mamba's custom CUDA scanning layers might need custom custom-op registrations "
            "for production compilation. Exception raised during export:"
        )
        print(f"Error details: {e}")


if __name__ == "__main__":
    main()
