#include "tracking/copy_pad.cuh"
#include <cuda_runtime.h>

namespace saccade {

namespace {

__global__ void copy_pad_detections_kernel(
    const float* src_boxes,
    const float* src_scores,
    const int*   src_classes,
    int n_copy,
    float* dst_boxes,
    float* dst_scores,
    int*   dst_classes,
    int padded_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_n) return;
    if (idx < n_copy) {
        dst_boxes[idx * 4 + 0] = src_boxes[idx * 4 + 0];
        dst_boxes[idx * 4 + 1] = src_boxes[idx * 4 + 1];
        dst_boxes[idx * 4 + 2] = src_boxes[idx * 4 + 2];
        dst_boxes[idx * 4 + 3] = src_boxes[idx * 4 + 3];
        dst_scores[idx] = src_scores[idx];
        dst_classes[idx] = src_classes[idx];
    } else {
        dst_boxes[idx * 4 + 0] = 0.0f;
        dst_boxes[idx * 4 + 1] = 0.0f;
        dst_boxes[idx * 4 + 2] = 0.0f;
        dst_boxes[idx * 4 + 3] = 0.0f;
        dst_scores[idx] = 0.0f;
        dst_classes[idx] = 0;
    }
}

__global__ void interleaved_to_split_kernel(
    const float* det_6d,
    int n,
    float* dst_boxes,
    float* dst_scores,
    int*   dst_classes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* s = det_6d + i * 6;
    dst_boxes[i * 4 + 0] = s[0];
    dst_boxes[i * 4 + 1] = s[1];
    dst_boxes[i * 4 + 2] = s[2];
    dst_boxes[i * 4 + 3] = s[3];
    dst_scores[i] = s[4];
    dst_classes[i] = __float2int_rd(s[5]);
}

} // namespace

void copy_pad_detections(
    const float* src_boxes,
    const float* src_scores,
    const int*   src_classes,
    int n_copy,
    float* dst_boxes,
    float* dst_scores,
    int*   dst_classes,
    int padded_n,
    cudaStream_t stream)
{
    if (padded_n <= 0) return;
    const int THREADS = 256;
    const int BLOCKS = (padded_n + THREADS - 1) / THREADS;
    copy_pad_detections_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        src_boxes, src_scores, src_classes,
        n_copy,
        dst_boxes, dst_scores, dst_classes,
        padded_n);
}

void interleaved_to_split(
    const float* det_6d,
    int n,
    float* dst_boxes,
    float* dst_scores,
    int*   dst_classes,
    cudaStream_t stream)
{
    if (n <= 0) return;
    const int THREADS = 256;
    const int BLOCKS = (n + THREADS - 1) / THREADS;
    interleaved_to_split_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        det_6d, n, dst_boxes, dst_scores, dst_classes);
}

} // namespace saccade
