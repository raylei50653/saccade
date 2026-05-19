#pragma once
#include "saccade/common.hpp"
#include "tracking/pipeline.hpp"
#include "tracking/workbench.hpp"
#include "tracking/tracker_gpu.hpp"
#include "tracking/gmc.hpp"
#include "perception/trt_engine.hpp"
#include "perception/feature_extractor.hpp"
#include "perception/preprocessor.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace saccade {

/**
 * Per-sequence configuration for the C++ evaluator.
 * All frame paths must be absolute JPEG paths in display order.
 */
struct SequenceConfig {
    std::string              name;
    std::vector<std::string> frame_paths;
    int   width  = 1920;
    int   height = 1080;

    // Quality scaling (matches compute_detection_quality_batch in quality.py)
    float w_aspect          = 0.50f;
    float w_center          = 0.30f;
    float w_area            = 0.20f;

    // FP hard filter (matches _apply_fp_hard_filter in workbench.py)
    bool  fp_hard_filter    = true;
    float fp_min_score      = 0.25f;
    float fp_max_area       = 10000.0f;
    float fp_max_susp_score = 0.45f;

    // Narrow-person bonus
    float narrow_bonus            = 0.0f;
    float narrow_aspect_thresh    = 2.1f;
    float narrow_height_thresh    = 0.5f;
    int   person_class            = 0;

    // Tracker params (matching GPUByteTracker::set_params)
    float track_thresh           = 0.05f;
    float high_thresh            = 0.45f;
    float match_thresh           = 0.66f;
    float new_track_thresh       = 0.28f;
    float mid_thresh             = 0.10f;
    int   confirm_streak         = 1;
    float confirm_score_thresh   = 0.0f;
    float fuse_score_weight      = 0.4f;
    float vel_dir_weight         = 0.0f;
    float stage2_match_thresh    = 0.5f;
    float birth_low_score_thresh = 0.0f;
    float birth_prox_norm_thresh = 0.0f;  // proximity birth gate (0=off)
    int   track_buffer           = 30;

    // YOLO TRT geometry
    int   trt_input_size = 640;    // square input side (e.g. 640)
    int   max_raw_dets   = 8400;   // max detections from TRT output

    // GMC (camera motion compensation)
    bool  gmc_enabled    = true;
    int   gmc_downscale  = 8;
    bool  gmc_phase_corr = true;

    // ReID embedding extraction (empty path = disabled)
    std::string reid_engine_path = "";
    int   reid_model_type = 0;     // 0=SIGLIP2, 1=DINOV2, 2=TRANSREID, 3=OSNET, 4=FASTREID
    int   reid_budget     = 64;    // max detections to embed per frame
    int   reid_interval   = 1;     // extract every N frames
    int   reid_crop_h     = 224;
    int   reid_crop_w     = 224;
};

/**
 * Per-frame tracking output.
 * frame_id is 1-indexed (MOT17 convention).
 */
struct FrameResult {
    int                              frame_id;
    std::vector<int>                 track_ids;
    std::vector<std::array<float,4>> boxes;   // [x1, y1, x2, y2] in original coords
    std::vector<float>               scores;
};

/**
 * Runs a single MOT sequence in one C++ thread.
 * Owns per-thread GPU resources (stream, TRT context, tracker, pipeline, workbench).
 * The TRTEngine is borrowed (shared across threads).
 */
class SACCADE_TRACKING_API SequenceRunner {
public:
    /**
     * @param detect_engine  Shared (read-only) YOLO TRT engine. Thread-safe.
     * @param pipe_cfg       PerceptionPipeline config (filter + NMS thresholds).
     * @param max_dets       Max detections fed to the workbench.
     * @param max_tracks     Max concurrent tracks.
     * @param device_id      CUDA device index.
     */
    SequenceRunner(TRTEngine*                        detect_engine,
                   const PerceptionPipeline::Config& pipe_cfg,
                   int                               max_dets   = 2048,
                   int                               max_tracks = 256,
                   int                               device_id  = 0);
    ~SequenceRunner();

    SequenceRunner(const SequenceRunner&) = delete;
    SequenceRunner& operator=(const SequenceRunner&) = delete;

    /**
     * Run the full frame loop for one sequence.
     * Blocking — must be called from a worker thread.
     */
    std::vector<FrameResult> run(const SequenceConfig& cfg);

private:
    void alloc_buffers(int input_size, int max_raw_dets, int max_dets, int max_tracks);
    void free_buffers();

    // Load JPEG → GPU CHW float32 [0,1] (CPU-side decode + H2D upload)
    // Writes letterboxed 640×640 CHW into d_input_.
    // Returns the letterbox scale r = min(S/W, S/H) and offsets x_off, y_off.
    void load_frame(const std::string& path,
                    int frame_w, int frame_h, int S,
                    float& out_scale, int& out_x_off, int& out_y_off);

    TRTEngine*                   detect_engine_;  // borrowed, thread-safe (read-only)
    nvinfer1::IExecutionContext* detect_ctx_ = nullptr;

    // Pipeline is created once; tracker + workbench are created fresh per run()
    std::unique_ptr<PerceptionPipeline> pipeline_;
    cudaStream_t                        stream_ = nullptr;
    int                                 device_id_;

    // GMC — lazy-init on first run() with gmc_enabled=true
    std::unique_ptr<GMC> gmc_;
    float* d_gmc_warp_ = nullptr;   // [6] float32, GPU

    // ReID — lazy-init on first run() when reid_engine_path non-empty
    std::unique_ptr<FeatureExtractor> reid_extractor_;
    std::unique_ptr<Cropper>          reid_cropper_;
    float* d_reid_embeds_     = nullptr;  // [max_raw * feat_dim], GPU, zeroed per frame
    float* d_reid_sel_embeds_ = nullptr;  // [reid_budget * feat_dim], GPU
    float* d_reid_sel_boxes_  = nullptr;  // [reid_budget, 4], GPU
    int*   d_reid_sel_idx_    = nullptr;  // [reid_budget], GPU
    float* d_crop_buf_        = nullptr;  // [reid_budget, 3, crop_h, crop_w], GPU
    float* h_raw_scores_pin_  = nullptr;  // [max_raw], pinned host
    float* h_reid_boxes_pin_  = nullptr;  // [reid_budget, 4], pinned host
    int*   h_reid_idx_pin_    = nullptr;  // [reid_budget], pinned host
    int    reid_feat_dim_     = 0;
    int    reid_crop_h_       = 0;
    int    reid_crop_w_       = 0;

    // GPU buffers (owned)
    float* d_src_chw_     = nullptr;  // [3, H_max, W_max] original CHW frame
    float* d_input_       = nullptr;  // [3, S, S] letterboxed YOLO input
    float* d_yolo_out_    = nullptr;  // [max_raw_dets, 6]
    float* d_raw_boxes_   = nullptr;  // [max_raw_dets, 4]
    float* d_raw_scores_  = nullptr;  // [max_raw_dets]
    int*   d_raw_classes_ = nullptr;  // [max_raw_dets]
    float* d_out_boxes_   = nullptr;  // [max_tracks, 4]
    float* d_out_scores_  = nullptr;  // [max_tracks]
    int*   d_out_ids_     = nullptr;  // [max_tracks]
    int*   d_out_classes_ = nullptr;  // [max_tracks]
    int*   d_out_det_idx_ = nullptr;  // [max_tracks]
    int*   d_out_count_   = nullptr;  // [1]

    int max_raw_dets_   = 0;
    int max_tracks_     = 0;
    int trt_input_size_ = 0;

    // Pinned host staging buffer for CHW frame
    float* h_frame_pinned_ = nullptr;
    size_t h_frame_bytes_  = 0;

    // Pinned host output buffers. cudaMemcpyAsync with pageable destinations
    // leaks driver-side staging pinned memory rapidly under WSL2's WDDM stack
    // (~12MB/frame measured), so we allocate proper pinned buffers up front.
    int*   h_out_count_pinned_  = nullptr;
    int*   h_out_ids_pinned_    = nullptr;
    float* h_out_boxes_pinned_  = nullptr;
    float* h_out_scores_pinned_ = nullptr;
};

} // namespace saccade
