#!/usr/bin/env python3
import torch

# Force cuBLAS and PyTorch CUDA context initialization early
_ = torch.zeros(10, 10, device="cuda") @ torch.zeros(10, 10, device="cuda")

import sys
import ctypes

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
import saccade_tracking_ext  # noqa: F401
from pathlib import Path
import cv2

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.mamba_gated_detector import (
    build_mamba_gated_detector,
)
from saccade_perception_ext import MambaGatedDetector as CppMambaGatedDetector
from saccade.perception.temporal_yolo.data_pipeline import resize_stretch_batch_gpu


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load a real frame from MOT17 if available, otherwise fallback to random
    img_path = project_root / "datasets/MOT17/train/MOT17-04-SDP/img1/000001.jpg"
    if img_path.exists():
        print(f"Loading real frame from: {img_path}")
        frame_np = cv2.imread(str(img_path))
        frame_uint8 = (
            torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        frame = resize_stretch_batch_gpu(frame_uint8, 640, device)
    else:
        print("Real frame not found, using dummy frame...")
        frame = torch.randn(1, 3, 640, 640, device=device)

    # 2. Build Python Reference Detector
    print("\nBuilding Python Reference Detector...")
    py_model = build_mamba_gated_detector(
        yolo_pt_path="models/yolo/yolo26s.pt",
        teacher_ckpt="runs/gated_det_v1/best.ckpt",
        mamba_ckpt="runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        img_size=640,
        device=device,
        conf_thr=0.05,
        max_det=30000,
        emb_dim=128,
        trt_backbone_engine="models/yolo/yolo26s_backbone_640_best.engine",
        use_cuda_graph=False,
    )
    py_model.eval()
    if getattr(py_model, "_trt_backbone", None) is not None:
        print(f"TRT Backbone output names: {py_model._trt_backbone.output_names}")

    # 3. Build C++ LibTorch Detector
    print("\nBuilding C++ LibTorch Detector...")
    cpp_model = CppMambaGatedDetector(
        trt_backbone_path=str(
            (project_root / "models/yolo/yolo26s_backbone_640_best.engine").resolve()
        ),
        mamba_head_script_path=str(
            (project_root / "models/yolo/mamba_head_best.pt").resolve()
        ),
        img_size=640,
        conf_thr=0.05,
    )

    # 4. Run Python Reference forward pass
    print("\nRunning Python Reference forward...")
    with torch.no_grad():
        py_dets, py_extra = py_model(frame)
    if len(py_dets.shape) == 3:
        py_dets = py_dets[0]
    # Filter Python detections by conf threshold (Python returns padded max_det tensor)
    print(f"  py_dets shape: {py_dets.shape}")
    print(
        f"  py_dets scores min/max: {py_dets[:, 4].min().item():.6f} / {py_dets[:, 4].max().item():.6f}"
    )
    print(
        f"  py_dets non-zero count (score > 0.0): {(py_dets[:, 4] > 0.0).sum().item()}"
    )
    print(f"  py_dets first 20 scores:\n{py_dets[:20, 4]}")
    py_mask = py_dets[:, 4] >= 0.05
    py_dets_filtered = py_dets[py_mask]
    print(f"  Python Detections shape before NMS: {py_dets_filtered.shape}")

    # Apply NMS to Python detections to match C++ internal NMS
    from torchvision.ops import nms as tv_nms

    def cpp_nms_style(bboxes, scores, iou_threshold):
        if bboxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=bboxes.device)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            idx = order[1:]
            xx1 = torch.clamp_min(x1[idx], x1[i])
            yy1 = torch.clamp_min(y1[idx], y1[i])
            xx2 = torch.clamp_max(x2[idx], x2[i])
            yy2 = torch.clamp_max(y2[idx], y2[i])
            w = torch.clamp_min(xx2 - xx1, 0.0)
            h = torch.clamp_min(yy2 - yy1, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[idx] - inter)
            mask = ovr <= iou_threshold
            if not mask.any().item():
                break
            order = idx[mask]
        return torch.tensor(keep, dtype=torch.long, device=bboxes.device)

    if py_dets_filtered.shape[0] > 0:
        py_keep_tv = tv_nms(py_dets_filtered[:, :4], py_dets_filtered[:, 4], 0.5)
        py_keep_cpp = cpp_nms_style(
            py_dets_filtered[:, :4], py_dets_filtered[:, 4], 0.5
        )
        print(f"  torchvision NMS count: {py_keep_tv.shape[0]}")
        print(f"  cpp_nms_style NMS count: {py_keep_cpp.shape[0]}")
        py_dets = py_dets_filtered[py_keep_tv]
    else:
        py_dets = py_dets_filtered

    print(f"  Python Detections shape (after NMS): {py_dets.shape}")

    # 5. Run C++ LibTorch forward pass
    print("\nRunning C++ LibTorch forward...")
    with torch.no_grad():
        cpp_dets = cpp_model.forward(frame)
    print(f"  C++ Detections shape: {cpp_dets.shape}")

    # 6. Compare Detections (Bit-exact alignment)
    print("\nComparing Detections...")
    if py_dets.shape != cpp_dets.shape:
        print(
            f"⚠️ Warning: Detection count mismatch! Python: {py_dets.shape[0]}, C++: {cpp_dets.shape[0]}"
        )
        # Try running cpp_nms_style on C++ detections or compare
        print(f"Python first 5:\n{py_dets[:5]}")
        print(f"C++ first 5:\n{cpp_dets[:5]}")
        sys.exit(1)

    det_diff = torch.abs(py_dets - cpp_dets).max().item()
    print(f"Max absolute difference in detections (Python vs C++): {det_diff:.6e}")

    # 7. Compare FPN Embeddings
    print("\nComparing FPN Embeddings...")
    if py_dets.shape[0] > 0:
        boxes = py_dets[:, :4]
        with torch.no_grad():
            py_embs = py_model.extract_fpn_embeddings(None, boxes)
            cpp_embs = cpp_model.extract_fpn_embeddings(boxes)

        emb_diff = torch.abs(py_embs - cpp_embs).max().item()
        print(
            f"Max absolute difference in FPN Embeddings (Python vs C++): {emb_diff:.6e}"
        )
    else:
        emb_diff = 0.0
        print("No detections to extract FPN embeddings.")

    # Print detailed differences
    box_diff = torch.abs(py_dets[:, :4] - cpp_dets[:, :4]).max().item()
    score_diff = torch.abs(py_dets[:, 4] - cpp_dets[:, 4]).max().item()
    class_diff = torch.abs(py_dets[:, 5] - cpp_dets[:, 5]).max().item()
    print(f"Max absolute difference in boxes: {box_diff:.6e}")
    print(f"Max absolute difference in scores: {score_diff:.6e}")
    print(f"Max absolute difference in classes: {class_diff:.6e}")

    # 8. Assert absolute mathematical alignment
    assert det_diff < 2e-4, f"Detection mismatch detected: max diff = {det_diff:.6e}"
    assert emb_diff < 1e-5, f"Embedding mismatch detected: max diff = {emb_diff:.6e}"

    print(
        "\n🎉 Success! Python reference and C++ LibTorch Mamba implementations are mathematically aligned!"
    )


if __name__ == "__main__":
    main()
