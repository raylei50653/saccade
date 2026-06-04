# Tracker Refactoring Plan (C++ Core Preservation)

## Objective
Refactor the tracking pipeline to improve modularity and type safety while preserving the high-performance C++ Sinkhorn and Kalman Filter implementations.

## Architectural Changes

### 1. Unified Tracking Interface
- Standardize the interface between Python and C++.
- Ensure all tracking components follow a consistent life-cycle (Init -> Update -> Filter -> Reset).

### 2. Refactor `perception/tracker.py`
- Separate `ReorderingBuffer` into a dedicated utility if necessary, or streamline its integration.
- Ensure `SmartTracker` is strictly typed and follows the "Zero-Copy First" principle.
- Improve error handling and logging for latency spikes.

### 3. C++ Component Preservation
- **Sinkhorn Algorithm**: Keep `include/tracking/sinkhorn.hpp` as the core association engine.
- **Kalman Filter**: Keep `include/tracking/kalman_gpu.cuh` for GPU-accelerated state estimation.
- **SmartTracker (C++)**: Maintain the existing `SmartTracker` class in C++ to handle the heavy lifting of association and filtering.

### 4. Integration & Validation
- Update `perception/detector_trt.py` to use the refactored tracking interface.
- Add comprehensive tests for the refactored components.
- Run `mypy` to ensure strict type safety.

## Implementation Steps

1. [x] Create a consolidated `perception/tracking/` directory.
2. [x] Move and refactor `ReorderingBuffer` to `perception/tracking/reorder.py`.
3. [x] Refactor `SmartTracker` in `perception/tracking/tracker.py`.
4. [x] Update imports in `perception/detector_trt.py` and `main.py`.
5. [x] Verify performance and accuracy with existing tests.

## Why this approach?
- **Performance**: C++ Sinkhorn and Kalman on GPU are critical for the 3000 FPS aggregate throughput goal.
- **Maintainability**: Clear separation of concerns between reordering (business logic) and tracking (mathematical/performance core).
- **Type Safety**: Moving towards strict typing reduces runtime errors in complex pipeline interactions.
