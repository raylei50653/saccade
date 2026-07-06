#pragma once
#include <cuda_runtime.h>

namespace saccade {

void copy_pad_detections(
    const float* src_boxes,
    const float* src_scores,
    const int*   src_classes,
    int n_copy,
    float* dst_boxes,
    float* dst_scores,
    int*   dst_classes,
    int padded_n,
    cudaStream_t stream);

void interleaved_to_split(
    const float* det_6d,
    int n,
    float* dst_boxes,
    float* dst_scores,
    int*   dst_classes,
    cudaStream_t stream);

/// Copy contiguous [n_rows, elem] float rows to per-row destination pointers
/// (nullptr rows are skipped). One launch replaces n_rows cudaMemcpyAsync
/// calls in launch-bound paths (crop-ring stash).
void scatter_copy_rows(
    const float* src,
    float* const* dst_ptrs,
    int n_rows,
    int elem,
    cudaStream_t stream);

} // namespace saccade
