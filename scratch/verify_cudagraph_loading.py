#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.mamba_gated_detector import (
    build_mamba_gated_detector,
)


def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the preset to see if use_cuda_graph is true
    preset_path = project_root / "configs" / "presets" / "mamba_optimal.yaml"
    with preset_path.open() as f:
        preset = yaml.safe_load(f)
    print(f"Preset loaded. use_cuda_graph in preset: {preset.get('use_cuda_graph')}")

    # Build the detector using the preset's configs
    detector = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="",
        mamba_ckpt=preset["mamba_ckpt"],
        img_size=640,
        device=device,
        emb_dim=128,
        use_cuda_graph=preset.get("use_cuda_graph", False),
    )
    detector.eval()

    print("Detector built.")
    print(f"detector.mamba_head.use_cuda_graph: {detector.mamba_head.use_cuda_graph}")
    print(
        f"Number of captured graphs before forward: {len(detector.mamba_head._cuda_graphs)}"
    )

    # Run one forward pass to trigger capture
    frame = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        detector(frame)

    print(
        f"Number of captured graphs after forward: {len(detector.mamba_head._cuda_graphs)}"
    )
    for key, g in detector.mamba_head._cuda_graphs.items():
        print(f"Captured graph key: {key}")
        print(f"Graph object: {g}")

    # Run another forward pass to verify it replays
    with torch.no_grad():
        detector(frame)
    print("Replay successful!")


if __name__ == "__main__":
    verify()
