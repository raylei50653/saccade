# ADR 004: YOLO26 Perception (L1) Upgrade

## Status
Accepted (已落地，見 ADR 005 擴展整合)

## Context
Saccade currently uses **YOLO11** as its default perception engine (L1). While YOLO11 is high-performing, **YOLO26** (released January 2026) offers several key advantages for our edge-first, zero-copy architecture:
- **NMS-Free (End-to-End):** Eliminates the Non-Maximum Suppression post-processing step, reducing GPU-to-CPU synchronization overhead.
- **Edge-First Optimization:** Up to 43% faster on CPU (for nano variants) and significantly more efficient for TensorRT/ONNX export.
- **Improved Small Object Detection:** Leverages STAL and ProgLoss to better capture distant or small features.
- **MuSGD Optimization:** Faster convergence during training, though our primary concern is inference.

## Decision
Upgrade the L1 perception layer from YOLO11 to **YOLO26**.
Specifically:
- Change the default model to `yolo26n` (Nano) or `yolo26s` (Small) depending on VRAM constraints.
- Update `Detector` and `configs/yolo_config.yaml` to reflect this change.
- Re-export models to TensorRT (.engine) format for maximum performance.

## Consequences
- **Positive:** Lower latency, simpler deployment pipeline, better small-object detection.
- **Negative:** Requires updating `ultralytics` package and re-generating TRT engines.
- **Neutral:** The API remains mostly identical since both use the `ultralytics` framework.
