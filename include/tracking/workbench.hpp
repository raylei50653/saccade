#pragma once

#include "saccade/common.hpp"
#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"
#include <cuda_runtime.h>

namespace saccade {

/**
 * @brief Per-thread isolated workspace for the post-YOLO hot path.
 *
 * Composes a PerceptionPipeline (filter+NMS+geometry) and a GPUByteTracker
 * (multi-object tracker) plus per-workbench GPU scratch and a CUDA stream.
 * Multiple Workbenches in multiple threads can execute the hot path truly
 * in parallel because:
 *   - Each owns its own PerceptionPipeline (the pipeline's d_filter_*, d_nms_*,
 *     d_compact_* scratch is mutable and would race if shared).
 *   - Each owns its own GPUByteTracker (per-stream tracking state).
 *   - Each owns its own cudaStream_t so GPU kernels don't serialize.
 *   - process_frame_postyolo() is invoked under py::gil_scoped_release in the
 *     pybind11 binding, so the GIL is dropped for the entire ~3-6 ms call.
 *
 * Pipeline and tracker pointers are borrowed; the caller is responsible for
 * their lifetimes (typically Python objects via uintptr_t through pybind11).
 */
class SACCADE_TRACKING_API Workbench {
public:
    Workbench(PerceptionPipeline* pipeline,
              GPUByteTracker*    tracker,
              cudaStream_t       stream,
              int max_dets   = 2048,
              int max_tracks = 256);
    ~Workbench();

    Workbench(const Workbench&) = delete;
    Workbench& operator=(const Workbench&) = delete;

    /**
     * @brief Post-YOLO hot path: filter+NMS, then tracker.update_into.
     *
     * All work runs on the workbench's CUDA stream. Pointer inputs are raw
     * GPU pointers; no Python objects are touched, so the binding wraps this
     * call in py::gil_scoped_release for true cross-thread parallelism.
     *
     * @return Number of output tracks (also stored at *out_count on device).
     */
    int process_frame_postyolo(
        const float* raw_boxes,
        const float* raw_scores,
        const int*   raw_classes,
        int n_in,
        int frame_w, int frame_h,
        bool is_tiled,
        const float* priors_ptr,
        const int*   prior_classes_ptr,
        int          num_priors,
        float        prior_iou_threshold,
        const float* embeddings_ptr,    // may be nullptr
        const float* gmc_ptr,           // may be nullptr (length 9 row-major)
        float        light_factor,
        float        mid_thresh_scale,
        float* out_boxes,
        float* out_scores,
        int*   out_ids,
        int*   out_classes,
        int*   out_det_idx,
        int*   out_count);

private:
    PerceptionPipeline* pipeline_;   // borrowed
    GPUByteTracker*     tracker_;    // borrowed
    cudaStream_t        stream_;     // borrowed
    int max_dets_;
    int max_tracks_;

    // Per-workbench post-NMS staging — fed by pipeline_, consumed by tracker_.
    float* d_post_boxes_   = nullptr;
    float* d_post_scores_  = nullptr;
    int*   d_post_classes_ = nullptr;
    bool*  d_post_suspect_ = nullptr;
    int*   d_post_count_   = nullptr;
};

} // namespace saccade
