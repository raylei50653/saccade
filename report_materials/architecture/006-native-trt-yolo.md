# ADR 006: Native TensorRT API for YOLO (L1) Extreme Optimization

## Status
Proposed

## Context
Our current Perception (L1) uses the `ultralytics` Python library for YOLO26 inference. While YOLO26 itself is fast (NMS-free), the `ultralytics` wrapper introduces several sources of overhead and jitter:
1. **Python Object Overhead:** Creating and managing `Results` objects for every frame.
2. **Synchronous Calls:** Internal library calls that may block the GIL or perform unnecessary CPU/GPU synchronizations.
3. **Black-box Logic:** Limited control over CUDA streams and memory management within the library's internal `predict` or `track` methods.

## Decision
Implement a custom **`TRTYoloDetector`** using the native **TensorRT Python API**.
This implementation will:
- Directly bind GPU memory pointers using `torch.data_ptr()`.
- Explicitly manage inference on the current CUDA stream to allow for better pipelining.
- Extract the NMS-free output (boxes, scores, classes) directly into GPU Tensors without creating intermediate Python objects.

## Technical Path
1. **Engine Compatibility:** Re-build the YOLO26 engine using the project's standard `tensorrt` version (10.x) to ensure seamless loading.
2. **Zero-Copy Binding:** Map input `images` and output `output0` directly to PyTorch tensors.
3. **Performance Target:** Reduce L1 latency from ~18ms to **10-12ms** and significantly lower the Standard Deviation (Jitter).

## Consequences
- **Positive:** Extreme low latency, near-zero jitter, full control over the inference pipeline.
- **Negative:** More complex code to maintain than the high-level `ultralytics` API. Requires manual handling of the model's output format (scaling back to original frame size).
- **Neutral:** The underlying YOLO26 model remains the same.
