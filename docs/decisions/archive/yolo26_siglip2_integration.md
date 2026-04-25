# Integration Roadmap: YOLO26 & SigLIP 2

## Objective
Seamlessly integrate the YOLO26 perception layer (L1) and the SigLIP 2 extraction layer (L2) to achieve NMS-free high-speed inference and superior semantic precision.

## Architecture Context (L1-L2)
1. **L1 (Perception):** YOLO26 (NMS-Free, ~43% faster on CPU).
2. **L2 (Extraction):** SigLIP 2 (ViT-B/16 or SO400M).

## VRAM Resource Estimation
| Module | Model Variant | VRAM Usage (Est.) |
| :--- | :--- | :--- |
| **YOLO26** | yolo26n (Nano) | ~150 MB |
| **SigLIP 2** | ViT-B/16 (FP16) | ~300 MB |
| **Total Core Stack** | | **~450 MB** |

## Integration Tasks
- [x] **YOLO26 Basic Integration:**
    - [x] Download weights (`yolo26n.pt`).
    - [x] Update `Detector` and `configs/yolo_config.yaml`.
    - [x] Export to TensorRT (`yolo26n.engine`).
- [x] **SigLIP 2 Integration:**
    - [x] Export `google/siglip2-base-patch16-224` to ONNX and TensorRT.
    - [x] Update `TRTFeatureExtractor` to handle SigLIP 2 input dimensions.
    - [x] Update `perception/feature_extractor.py` output logic.
- [x] **Drift Handler Calibration:**
    - [x] Test Cosine Similarity thresholds for SigLIP 2 features.
    - [x] Adjust `SemanticDriftHandler` similarity threshold (Baseline: 0.95).

## Verification Milestones
- [x] **L1 Loading Test:** Verify `Detector` correctly loads YOLO26 engine.
- [x] **L1-L2 Pipeline Test:** End-to-end zero-copy inference from YOLO26 to SigLIP 2.
- [x] **Drift Precision Test:** Confirm SigLIP 2 correctly filters semantic redundancy.

Last Updated: 2026-04-13
