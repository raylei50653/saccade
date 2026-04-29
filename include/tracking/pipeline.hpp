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
    struct Config {
        float score_threshold       = 0.05f;
        int   person_class          = 0;
        bool  person_only           = true;
        float nms_threshold         = 0.50f;
        bool  person_geometry_prior = true;
        bool  geometry_suspect_support = true;
        float person_min_height_ratio = 0.018f;
        float person_min_aspect       = 1.0f;
        float person_max_aspect       = 5.5f;
        float person_min_area_ratio   = 0.00006f;
        float person_max_area_ratio   = 0.0f;
        int   max_detections          = 2048;
    };

    PerceptionPipeline(FeatureExtractor* reid, Cropper* cropper, Config cfg);
    ~PerceptionPipeline();

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
    float*    d_crop_buf_             = nullptr;
    int       scratch_capacity_       = 0;
    int       crop_buf_capacity_      = 0;

    void ensure_scratch(int n_dets, cudaStream_t stream);
    void ensure_crop_buf(int n_boxes);
};

} // namespace saccade
