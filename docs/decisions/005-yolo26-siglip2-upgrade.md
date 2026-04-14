# ADR 005: Next-Gen Perception Stack (YOLO26 & SigLIP 2)

## Status
Proposed

## Context
The Saccade project aims for maximum efficiency on edge AI hardware. Our current stack (YOLO11 + Jina-CLIP-v2) is performant but faces two bottlenecks:
1. **Inference Latency:** YOLO11 requires Non-Maximum Suppression (NMS), which involves GPU-to-CPU synchronization and limits throughput.
2. **Semantic Precision:** While Jina-CLIP-v2 is strong, the recently released **SigLIP 2** (Feb 2025) offers superior zero-shot classification and localization capabilities, especially for "Semantic Drift" detection in dynamic environments.

## Decision
Upgrade the core perception and extraction modules (L1 & L2) to:
- **L1 (Perception):** **YOLO26**. Leverage its NMS-Free (End-to-End) architecture to eliminate synchronization overhead.
- **L2 (Extraction):** **SigLIP 2** (ViT-B/16 or SO400M variant). Utilize its improved dense feature extraction for more accurate semantic indexing.

## Technical Path
1. **YOLO26 Integration:**
    - Standardize on `yolo26n.engine` (Nano) for the 1.5GB VRAM target.
    - Update `Detector` to handle NMS-free result formats.
2. **SigLIP 2 Integration:**
    - Export `google/siglip2-base-patch16-224` or similar to ONNX/TensorRT FP16.
    - Update `TRTFeatureExtractor` to handle the new input dimensions and output feature vectors.
    - Calibrate `SemanticDriftHandler` thresholds for the new embedding space.
3. **Zero-Copy Pipeline:**
    - Maintain the `MediaMTX -> NVDEC -> CUDA Tensor -> YOLO26 -> RoiAlign -> SigLIP 2` path.

## Consequences
- **Positive:** Lower E2E latency, higher semantic accuracy, reduced VRAM-to-CPU copies.
- **Negative:** Requires re-calibration of all similarity thresholds.
- **Neutral:** Model sizes remain comparable to the previous stack.
