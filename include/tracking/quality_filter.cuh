#pragma once
#include "saccade/common.hpp"
#include <cuda_runtime.h>

namespace saccade {

/**
 * GPU quality ops — ports of Python quality.py / workbench.py equivalents.
 * All kernels run on the provided stream and return immediately (async).
 */

/**
 * Multiply scores in-place by quality factors derived from box geometry.
 *
 * quality = w_aspect * aspect_q + w_center * center_q + w_area * area_q
 *   aspect_q:  Gaussian peak at aspect_ratio=2.5, sigma=1.2  (bh/bw)
 *   center_q:  proximity to frame center, clamped to [0,1]
 *   area_q:    Gaussian peak at area_ratio=0.01, sigma=0.01   (box_area/frame_area)
 *
 * Matches compute_detection_quality_batch() in quality.py.
 */
SACCADE_TRACKING_API
void quality_scale_scores(
    float*       scores,      // [N] float32 GPU — modified in-place
    const float* boxes,       // [N, 4] float32 GPU  (x1, y1, x2, y2)
    int          n,
    int          frame_w,
    int          frame_h,
    float        w_aspect,    // default 0.50
    float        w_center,    // default 0.30
    float        w_area,      // default 0.20
    cudaStream_t stream);

/**
 * Add a score bonus to narrow/tall person detections.
 *
 * Applies `bonus` to detections where:
 *   class == person_class  AND
 *   aspect_ratio (bh/bw) < narrow_aspect_thresh  AND
 *   bh / frame_h < narrow_height_thresh
 *
 * Matches _apply_narrow_bonus() in workbench.py.
 */
SACCADE_TRACKING_API
void narrow_bonus_scores(
    float*       scores,             // [N] float32 GPU — modified in-place
    const float* boxes,              // [N, 4] float32 GPU
    const int*   classes,            // [N] int32 GPU
    int          n,
    float        bonus,
    int          person_class,       // default 0
    float        narrow_aspect_thresh,  // default 2.1
    float        narrow_height_thresh,  // default 0.5 (fraction of frame_h)
    int          frame_h,
    cudaStream_t stream);

/**
 * Compact detections that pass the FP hard filter.
 *
 * Removes detections where:
 *   score < min_score   (very low confidence), OR
 *   area > max_area AND score < max_suspicious_score  (large + uncertain)
 *
 * Writes compacted boxes/scores/classes to out_* and stores count in *out_count.
 * out_count is written to device memory (int32) AND returned as a CPU value.
 *
 * Caller must ensure out_* are at least `n` elements.
 * Matches _apply_fp_hard_filter() in workbench.py.
 *
 * Returns: number of kept detections (synchronous result).
 */
SACCADE_TRACKING_API
int fp_hard_filter(
    const float* boxes_in,           // [N, 4] float32 GPU
    const float* scores_in,          // [N] float32 GPU
    const int*   classes_in,         // [N] int32 GPU
    int          n,
    float        min_score,          // default 0.25
    float        max_area,           // default 10000.0
    float        max_suspicious_score,  // default 0.45
    float*       boxes_out,          // [N, 4] float32 GPU
    float*       scores_out,         // [N] float32 GPU
    int*         classes_out,        // [N] int32 GPU
    cudaStream_t stream);

/**
 * Parse YOLO flat output [N, 6] → separate arrays + inverse letterbox transform.
 *
 * raw[i] = {x1_lb, y1_lb, x2_lb, y2_lb, score, class} in letterboxed coords.
 * boxes_out[i] = {x1, y1, x2, y2} in original image coordinates.
 *
 * inv_scale = 1.0 / min(input_size / frame_w, input_size / frame_h)
 * x_off, y_off: left/top padding in the letterboxed image
 */
SACCADE_TRACKING_API
void parse_yolo_output(
    const float* raw,        // [N, 6] float32 GPU
    float*       boxes_out,  // [N, 4] float32 GPU
    float*       scores_out, // [N] float32 GPU
    int*         classes_out,// [N] int32 GPU
    int          N,
    float        inv_scale,
    float        x_off,
    float        y_off,
    cudaStream_t stream);

} // namespace saccade
