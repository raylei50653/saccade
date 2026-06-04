#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.mamba_gated_detector import (
    build_mamba_gated_detector,
)


def test_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build eager model
    print("Building detector...")
    model = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="",
        mamba_ckpt="runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        img_size=640,
        device=device,
        emb_dim=128,
        use_cuda_graph=False,
    )
    model.eval()

    # Generate dummy frame
    frame = torch.randn(1, 3, 640, 640, device=device)

    # Warmup and run eager
    with torch.no_grad():
        for _ in range(5):
            det_eager, extra_eager = model(frame)

    # Enable CUDA graph
    model.set_use_cuda_graph(True)
    print("CUDA Graph enabled.")

    # Run and trigger capture + replay
    with torch.no_grad():
        det_graph, extra_graph = model(frame)  # Triggers capture and first replay

        # Run a few times to test replay
        for _ in range(5):
            det_graph, extra_graph = model(frame)

    # Check shape and equivalence
    print("Checking mathematical equivalence between eager and CUDA graph outputs...")
    # Compare the detections
    diff_det = torch.abs(det_eager - det_graph).max().item()
    print(f"Max absolute difference in detections: {diff_det:.6e}")

    if "emb_preds" in extra_eager:
        for idx, (e_emb, g_emb) in enumerate(
            zip(extra_eager["emb_preds"], extra_graph["emb_preds"])
        ):
            diff_emb = torch.abs(e_emb - g_emb).max().item()
            print(f"Scale {idx} embedding max difference: {diff_emb:.6e}")
            assert diff_emb < 1e-4, f"Embedding difference too high at scale {idx}!"

    assert diff_det < 1e-4, "Detection difference too high!"
    print(
        "🎉 Success! CUDA Graph outputs are mathematically identical to Eager outputs."
    )


if __name__ == "__main__":
    test_equivalence()
