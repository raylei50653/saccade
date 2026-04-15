# Saccade Optimization Progress: Phases [A, B, C, D]

## 1. Overview
This document records the results of the "Non-Inference Overhead Elimination" plan, which targeted the ~22ms gap between raw GPU inference time and end-to-end latency.

## 2. Benchmark Results (32 Streams)
| Phase | Configuration | E2E Latency (Avg) | GPU Util | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Python + Single Stream | 55.15 ms | 95% | Initial state after engine rebuild |
| **Phase 1** | CUDA Streams Parallelism | 13.30 ms | 97% | L1/L2 overlapping |
| **Phase 4** | C++ Core Migration | **12.69 ms** | 98% | Minimized Python/GIL overhead |

## 3. Implementation Details

### (A) DALI GPU Preprocessing
- **Status**: ✅ Completed (2026-04-15)
- **File**: `media/dali_pipeline.py`
- **Impact**: Moved decoding, resizing, and normalization to GPU. Replaced `MediaMTXClient` for file-based inputs in `main.py`.
- **Benchmark**: Reduced preprocessing latency from ~8ms (CPU) to <1.5ms (GPU).

### (B) CUDA Streams Parallelism
- **Status**: ✅ Completed
- **Files**: `perception/dispatcher.py`, `perception/embedding_dispatcher.py`
- **Impact**: Created dedicated streams `l1_stream` and `l2_stream`. Allowed GPU to overlap YOLO detection of frame N+1 with SigLIP2 extraction of frame N.

### (C) C++ Core Migration (ADR 007)
- **Status**: ✅ Completed
- **Files**: `src/perception/trt_engine.cpp`, `src/perception/perception_python.cpp`
- **Impact**: Ported TensorRT execution to C++. Created `saccade_perception_ext` pybind11 module. Eliminated micro-stutters caused by Python's `execute_async_v3` calls.

### (D) Redis Pipelining
- **Status**: ✅ Completed
- **File**: `storage/redis_cache.py`
- **Impact**: Implemented `add_to_stream_batch` using Redis Pipeline. Reduced system call overhead for high-frequency event logging.

## 4. Stability & Verification
- **Unit Tests**: 10/10 passed (including updated legacy tests).
- **Type Checking**: 100% compliant (`mypy perception/ storage/`).
- **Stress Test**: 32 streams stable at ~78 FPS per stream on single GPU.
- **Breakthrough**: Achieved 10-stream 3000 FPS aggregate throughput (300 FPS per stream).

## 5. Next Steps
- Expand C++ migration to include the `Cropper` and `Dispatcher` queue management.
- Integrate DALI for RTSP streams (currently optimized for files).

最後更新：2026-04-15
