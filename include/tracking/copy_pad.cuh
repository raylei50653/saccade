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

} // namespace saccade
