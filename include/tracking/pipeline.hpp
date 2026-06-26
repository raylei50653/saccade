#pragma once

#include "saccade/common.hpp"
#include "perception/feature_extractor.hpp"
#include "perception/preprocessor.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace saccade {

/**
 * @brief PerceptionPipeline — C++ facade for the per-frame hot path.
 *
 * Consolidates detection postprocessing (filter + NMS) and ReID extraction
 * into a single C++ call, eliminating Python boundary crossings for each step.
 *
 * Usage:
 *   1. Python tiling logic fills a batch input tensor and calls detect_raw().
 *   2. Pass the raw detection output to process_detections() along with a
 *      pre-allocated scratch workspace.
 *   3. Optionally call extract_reid() on the filtered boxes.
 */
class SACCADE_TRACKING_API PerceptionPipeline {
public:
    struct ReIDProfileStats {
        double crop_ms = 0.0;
        double extract_pre_normalize_ms = 0.0;
        double extract_trt_enqueue_ms = 0.0;
        double extract_l2_normalize_ms = 0.0;
        double extract_total_ms = 0.0;
        double total_ms = 0.0;
        int chunks = 0;
        int images = 0;
    };

    struct PostprocessProfileStats {
        double filter_ms = 0.0;
        double nms_ms = 0.0;
        double count_d2h_ms = 0.0;
        double total_ms = 0.0;
        double native_filter_gather_ms = 0.0;
        double native_filter_kernel_ms = 0.0;
        double native_gather_compact3_ms = 0.0;
        double native_copy_suspect_ms = 0.0;
        double native_filter_count_sync_ms = 0.0;
        double native_small_nms_ms = 0.0;
        double native_suspect_penalty_ms = 0.0;
        double native_large_sort_nms_ms = 0.0;
        double native_large_argsort_ms = 0.0;
        double native_large_nms_ms = 0.0;
        double native_compact_copy_ms = 0.0;
        double native_large_gather4_ms = 0.0;
        double native_large_copyback_ms = 0.0;
        int input_boxes = 0;
        int filtered_boxes = 0;
        int output_boxes = 0;
    };

    struct Config {
        float score_threshold       = 0.05f;
        int   person_class          = 0;
        bool  person_only           = true;
        float nms_threshold         = 0.50f;
        bool  person_geometry_prior = true;
        bool  geometry_suspect_support = true;
        float geometry_suspect_support_score = 0.25f;
        float person_min_height_ratio = 0.018f;
        float person_min_aspect       = 1.0f;
        float person_max_aspect       = 5.5f;
        float person_min_area_ratio   = 0.00006f;
        float person_max_area_ratio   = 0.0f;
        int   max_detections          = 2048;
    };

    PerceptionPipeline(FeatureExtractor* reid, Cropper* cropper, Config cfg);
    ~PerceptionPipeline();

    PerceptionPipeline(const PerceptionPipeline&) = delete;
    PerceptionPipeline& operator=(const PerceptionPipeline&) = delete;
    PerceptionPipeline(PerceptionPipeline&&) = delete;
    PerceptionPipeline& operator=(PerceptionPipeline&&) = delete;

    /**
     * @brief Filter + NMS on flat detection arrays already on GPU.
     *
     * @param boxes_ptr   [N, 4] float32 GPU
     * @param scores_ptr  [N] float32 GPU
     * @param classes_ptr [N] int32 GPU
     * @param n_in        Number of raw detections
     * @param frame_w     Original frame width (for geometry prior)
     * @param frame_h     Original frame height
     * @param is_tiled    Whether detections come from a tiled pass
     * @param out_boxes   [max_detections, 4] float32 GPU — caller allocates
     * @param out_scores  [max_detections] float32 GPU
     * @param out_classes [max_detections] int32 GPU
     * @param out_suspect [max_detections] bool GPU — geometry-suspect flags
     * @param stream      CUDA stream
     * @return Number of detections after filter + NMS
     */
    int process_detections(
        const float* boxes_ptr,
        const float* scores_ptr,
        const int*   classes_ptr,
        int n_in,
        int frame_w, int frame_h,
        bool is_tiled,
        float* out_boxes,
        float* out_scores,
        int*   out_classes,
        bool*  out_suspect,
        cudaStream_t stream
    );

    void process_detections_into(
        const float* boxes_ptr,
        const float* scores_ptr,
        const int*   classes_ptr,
        int n_in,
        int frame_w, int frame_h,
        bool is_tiled,
        float* out_boxes,
        float* out_scores,
        int*   out_classes,
        bool*  out_suspect,
        int*   out_count,
        const float* priors_ptr,
        const int* prior_classes_ptr,
        int num_priors,
        float prior_iou_threshold,
        cudaStream_t stream);

    /**
     * @brief Like process_detections_into() but fully synchronous and returns
     *        the post-NMS count directly.  Python binding uses
     *        py::gil_scoped_release so GIL is released for the entire call
     *        (including the internal stream syncs), allowing other Python
     *        threads to make progress during GPU execution.
     */
    int process_detections_n(
        const float* boxes_ptr,
        const float* scores_ptr,
        const int*   classes_ptr,
        int n_in,
        int frame_w, int frame_h,
        bool is_tiled,
        float* out_boxes,
        float* out_scores,
        int*   out_classes,
        bool*  out_suspect,
        const float* priors_ptr,
        const int* prior_classes_ptr,
        int num_priors,
        float prior_iou_threshold,
        cudaStream_t stream);

    /**
     * @brief Near-sync-free variant of process_detections_n.
     *
     * Runs the same filter + NMS pipeline and skips only the final n_post
     * D2H read, returning n_in (the raw input count) instead of the actual
     * post-NMS count.  NOTE: process_detections_into still performs one
     * filter_count D2H sync internally to select the small/large NMS path,
     * so this is NOT fully sync-free — use process_detections_graph() for a
     * truly sync-free (graph-capturable) path.
     *
     * Output buffers are always n_in elements: valid detections are compacted
     * to the front, remaining slots are zero-padded by the gather kernels.
     *
     * Callers MUST NOT slice output by the returned count; instead rely on
     * the tracker's score gate (track_thresh > 0) to ignore zero-score slots.
     */
    int process_detections_n_fixed(
        const float* boxes_ptr,
        const float* scores_ptr,
        const int*   classes_ptr,
        int n_in,
        int frame_w, int frame_h,
        bool is_tiled,
        float* out_boxes,
        float* out_scores,
        int*   out_classes,
        bool*  out_suspect,
        const float* priors_ptr,
        const int* prior_classes_ptr,
        int num_priors,
        float prior_iou_threshold,
        cudaStream_t stream);

    /**
     * @brief Graph-capture-safe NMS: always uses the large path, skips all
     * D2H sync points. Input must be zero-padded to n_in.
     */
    void process_detections_graph(
        const float* boxes_ptr,
        const float* scores_ptr,
        const int*   classes_ptr,
        int n_in,
        int frame_w, int frame_h,
        bool is_tiled,
        float* out_boxes,
        float* out_scores,
        int*   out_classes,
        bool*  out_suspect,
        int*   out_count,
        const float* priors_ptr,
        const int* prior_classes_ptr,
        int num_priors,
        float prior_iou_threshold,
        cudaStream_t stream);

    /**
     * @brief Combined interleaved→split→NMS graph. Takes a (N,6) interleaved
     * source (x1,y1,x2,y2,score,class), splits to intermediate buffers, then
     * runs graph-safe NMS. All fixed-size — safe for CUDA graph capture.
     */
    void process_detections_interleaved_graph(
        const float* det_6d,
        int n_in,
        float* split_boxes,
        float* split_scores,
        int*   split_classes,
        int frame_w, int frame_h,
        bool is_tiled,
        float* out_boxes,
        float* out_scores,
        int*   out_classes,
        bool*  out_suspect,
        int*   out_count,
        cudaStream_t stream);

    const Config& get_config() const { return cfg_; }

    /**
     * @brief Crop boxes from frame_ptr and extract ReID embeddings.
     *
     * @param frame_ptr   [3, H, W] float32 GPU (CHW, [0,1])
     * @param frame_h     Frame height
     * @param frame_w     Frame width
     * @param boxes_ptr   [N, 4] float32 GPU — boxes to crop
     * @param n_boxes     Number of boxes
     * @param out_embeds  [N, feat_dim] float32 GPU — caller allocates
     * @param stream      CUDA stream
     */
    void extract_reid(
        const float* frame_ptr, int frame_h, int frame_w,
        const float* boxes_ptr, int n_boxes,
        float* out_embeds,
        cudaStream_t stream
    );

    int get_embed_dim() const;
    void set_reid_profiling_enabled(bool enabled);
    void reset_reid_profile_stats();
    ReIDProfileStats get_reid_profile_stats() const;
    void set_postprocess_profiling_enabled(bool enabled);
    void reset_postprocess_profile_stats();
    PostprocessProfileStats get_postprocess_profile_stats() const;

private:
    FeatureExtractor* reid_;
    Cropper*          cropper_;
    Config            cfg_;

    // Scratch buffers for filter and NMS stages
    int*      d_filter_keep_indices_  = nullptr;
    bool*     d_filter_suspect_flags_ = nullptr;
    int*      d_filter_count_         = nullptr;
    int64_t*  d_nms_order_            = nullptr;
    int*      d_nms_keep_             = nullptr;
    uint64_t* d_nms_suppression_      = nullptr;
    uint64_t* d_nms_remv_             = nullptr;
    int*      d_nms_count_            = nullptr;
    bool*     d_nms_immunity_mask_    = nullptr;
    float*    d_crop_buf_             = nullptr;
    int       scratch_capacity_       = 0;
    int       crop_buf_capacity_      = 0;
    // M1: GPU compaction scratch (NMS in-place scatter needs temp)
    float*    d_compact_boxes_        = nullptr;
    float*    d_compact_scores_       = nullptr;
    int*      d_compact_classes_      = nullptr;
    bool*     d_compact_suspect_      = nullptr;
    // M1: GPU argsort scratch (compound uint64 keys for stable sort)
    uint64_t* d_sort_keys_in_         = nullptr;  // unsorted compound keys
    uint64_t* d_sort_keys_out_        = nullptr;  // CUB output scratch
    void*     d_cub_sort_tmp_         = nullptr;
    size_t    cub_sort_tmp_bytes_     = 0;

    void ensure_scratch(int n_dets, cudaStream_t stream);
    void ensure_crop_buf(int n_boxes);
    bool reid_profiling_enabled_ = false;
    ReIDProfileStats last_reid_profile_stats_{};
    bool postprocess_profiling_enabled_ = false;
    PostprocessProfileStats last_postprocess_profile_stats_{};
};

} // namespace saccade
