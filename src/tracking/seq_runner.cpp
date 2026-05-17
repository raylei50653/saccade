#include "tracking/seq_runner.hpp"
#include "tracking/quality_filter.cuh"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <stdexcept>

namespace saccade {

// Declared in letterbox_kernel.cu
void launch_letterbox_gpu(
    const float* src, int src_w, int src_h,
    float*       dst, int dst_size,
    int x_off, int y_off, int w_new, int h_new,
    float pad_val, cudaStream_t stream);

// ─── SequenceRunner ───────────────────────────────────────────────────────────

SequenceRunner::SequenceRunner(TRTEngine*                        detect_engine,
                               const PerceptionPipeline::Config& pipe_cfg,
                               int                               max_dets,
                               int                               max_tracks,
                               int                               device_id)
    : detect_engine_(detect_engine)
    , device_id_(device_id)
    , max_tracks_(max_tracks)
{
    cudaSetDevice(device_id_);
    cudaStreamCreate(&stream_);

    detect_ctx_ = detect_engine_->create_context();
    if (!detect_ctx_)
        throw std::runtime_error("SequenceRunner: failed to create TRT execution context");

    pipeline_ = std::make_unique<PerceptionPipeline>(nullptr, nullptr, pipe_cfg);
}

SequenceRunner::~SequenceRunner() {
    free_buffers();
    if (detect_ctx_) {
        TRTEngine::delete_context(detect_ctx_);
        detect_ctx_ = nullptr;
    }
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void SequenceRunner::alloc_buffers(int S, int max_raw, int /*max_dets*/, int max_tracks) {
    if (trt_input_size_ == S && max_raw_dets_ == max_raw && max_tracks_ == max_tracks)
        return;
    free_buffers();

    trt_input_size_ = S;
    max_raw_dets_   = max_raw;
    max_tracks_     = max_tracks;

    cudaMalloc(&d_input_,       3 * (size_t)S * S * sizeof(float));
    cudaMalloc(&d_yolo_out_,    (size_t)max_raw * 6 * sizeof(float));
    cudaMalloc(&d_raw_boxes_,   (size_t)max_raw * 4 * sizeof(float));
    cudaMalloc(&d_raw_scores_,  (size_t)max_raw     * sizeof(float));
    cudaMalloc(&d_raw_classes_, (size_t)max_raw     * sizeof(int));
    // GPUByteTracker::update_into copies max_objs = max_tracks*8 elements.
    // Output buffers must be at least that large to avoid buffer overflow.
    const size_t max_objs = (size_t)max_tracks * 8;
    cudaMalloc(&d_out_boxes_,   max_objs * 4 * sizeof(float));
    cudaMalloc(&d_out_scores_,  max_objs     * sizeof(float));
    cudaMalloc(&d_out_ids_,     max_objs     * sizeof(int));
    cudaMalloc(&d_out_classes_, max_objs     * sizeof(int));
    cudaMalloc(&d_out_det_idx_, max_objs     * sizeof(int));
    cudaMalloc(&d_out_count_,                  sizeof(int));

    // Pinned host destinations for D2H — required to avoid per-call driver
    // staging allocations under WSL2 WDDM.
    cudaMallocHost(&h_out_count_pinned_,  sizeof(int));
    cudaMallocHost(&h_out_ids_pinned_,    max_objs * sizeof(int));
    cudaMallocHost(&h_out_boxes_pinned_,  max_objs * 4 * sizeof(float));
    cudaMallocHost(&h_out_scores_pinned_, max_objs * sizeof(float));
}

void SequenceRunner::free_buffers() {
    auto cfree = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto cfreehost = [](void*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
    cfree((void*&)d_src_chw_);
    cfree((void*&)d_input_);
    cfree((void*&)d_yolo_out_);
    cfree((void*&)d_raw_boxes_);
    cfree((void*&)d_raw_scores_);
    cfree((void*&)d_raw_classes_);
    cfree((void*&)d_out_boxes_);
    cfree((void*&)d_out_scores_);
    cfree((void*&)d_out_ids_);
    cfree((void*&)d_out_classes_);
    cfree((void*&)d_out_det_idx_);
    cfree((void*&)d_out_count_);
    cfreehost((void*&)h_out_count_pinned_);
    cfreehost((void*&)h_out_ids_pinned_);
    cfreehost((void*&)h_out_boxes_pinned_);
    cfreehost((void*&)h_out_scores_pinned_);
    if (h_frame_pinned_) {
        cudaFreeHost(h_frame_pinned_);
        h_frame_pinned_ = nullptr;
        h_frame_bytes_  = 0;
    }
}

void SequenceRunner::load_frame(const std::string& path,
                                 int frame_w, int frame_h, int S,
                                 float& out_scale, int& out_x_off, int& out_y_off)
{
    // Ensure src CHW buffer is large enough
    size_t needed_src = 3 * (size_t)frame_h * frame_w * sizeof(float);
    if (d_src_chw_ == nullptr)
        cudaMalloc(&d_src_chw_, needed_src);

    if (needed_src > h_frame_bytes_) {
        if (h_frame_pinned_) cudaFreeHost(h_frame_pinned_);
        cudaMallocHost(&h_frame_pinned_, needed_src);
        h_frame_bytes_ = needed_src;
    }

    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty())
        throw std::runtime_error("SequenceRunner: cannot read: " + path);

    if (bgr.cols != frame_w || bgr.rows != frame_h)
        cv::resize(bgr, bgr, cv::Size(frame_w, frame_h));

    cv::Mat f32;
    bgr.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(f32, channels);

    const size_t n_pixels = (size_t)frame_h * frame_w;
    std::memcpy(h_frame_pinned_,                  channels[2].ptr<float>(), n_pixels * sizeof(float));
    std::memcpy(h_frame_pinned_ + n_pixels,       channels[1].ptr<float>(), n_pixels * sizeof(float));
    std::memcpy(h_frame_pinned_ + 2 * n_pixels,   channels[0].ptr<float>(), n_pixels * sizeof(float));

    // H2D upload
    cudaMemcpyAsync(d_src_chw_, h_frame_pinned_, needed_src,
                    cudaMemcpyHostToDevice, stream_);

    // Letterbox params
    float scale = std::min((float)S / frame_w, (float)S / frame_h);
    int   w_new = static_cast<int>(frame_w * scale);
    int   h_new = static_cast<int>(frame_h * scale);
    int   x_off = (S - w_new) / 2;
    int   y_off = (S - h_new) / 2;

    out_scale = scale;
    out_x_off = x_off;
    out_y_off = y_off;

    launch_letterbox_gpu(d_src_chw_, frame_w, frame_h,
                         d_input_, S,
                         x_off, y_off, w_new, h_new,
                         114.0f / 255.0f, stream_);
}

std::vector<FrameResult> SequenceRunner::run(const SequenceConfig& cfg) {
    cudaSetDevice(device_id_);
    const int S       = cfg.trt_input_size;
    const int max_raw = cfg.max_raw_dets;
    alloc_buffers(S, max_raw, 2048, max_tracks_);

    // Create a fresh tracker for this sequence (no reset() method available)
    GPUByteTracker tracker(max_tracks_ * 8);
    tracker.set_params(
        cfg.track_thresh,
        cfg.high_thresh,
        cfg.match_thresh,
        cfg.track_buffer,
        cfg.mid_thresh,
        cfg.confirm_streak,
        cfg.confirm_score_thresh,
        /*adaptive_confirmation=*/false,
        cfg.new_track_thresh,
        /*nsa_kalman=*/false,
        /*r_scale=*/1.0f,
        cfg.vel_dir_weight,
        cfg.fuse_score_weight,
        cfg.stage2_match_thresh,
        cfg.birth_low_score_thresh
    );
    Workbench workbench(pipeline_.get(), &tracker, stream_, 2048, max_tracks_);

    // Configure TRT context input shape (only needed for dynamic/profile engines)
    {
        nvinfer1::Dims engine_dims = detect_engine_->getTensorDims("images");
        bool is_dynamic = false;
        for (int i = 0; i < engine_dims.nbDims; ++i) {
            if (engine_dims.d[i] < 0) { is_dynamic = true; break; }
        }
        if (is_dynamic) {
            nvinfer1::Dims4 in_dims{1, 3, S, S};
            detect_ctx_->setInputShape("images", in_dims);
        }
    }

    const int n_frames = static_cast<int>(cfg.frame_paths.size());
    std::vector<FrameResult> results;
    results.reserve(n_frames);

    // Use the pre-allocated pinned host buffers (see alloc_buffers).
    int*   h_ids    = h_out_ids_pinned_;
    float* h_boxes  = h_out_boxes_pinned_;
    float* h_scores = h_out_scores_pinned_;

    for (int fi = 0; fi < n_frames; ++fi) {
        // 1. Load + letterbox frame → d_input_
        float scale; int x_off, y_off;
        load_frame(cfg.frame_paths[fi], cfg.width, cfg.height, S, scale, x_off, y_off);

        // 2. TRT detection (async)
        {
            void* bindings[2] = {d_input_, d_yolo_out_};
            detect_engine_->infer_with_context(detect_ctx_, {bindings[0], bindings[1]}, stream_);
        }

        // 3. Parse YOLO output: [max_raw, 6] → boxes/scores/classes + inverse letterbox
        parse_yolo_output(d_yolo_out_,
                          d_raw_boxes_, d_raw_scores_, d_raw_classes_,
                          max_raw, 1.0f / scale, (float)x_off, (float)y_off,
                          stream_);

        // 4. Quality scaling (in-place)
        quality_scale_scores(d_raw_scores_, d_raw_boxes_, max_raw,
                             cfg.width, cfg.height,
                             cfg.w_aspect, cfg.w_center, cfg.w_area,
                             stream_);

        // 5. Narrow bonus (optional, in-place)
        if (cfg.narrow_bonus > 0.0f) {
            narrow_bonus_scores(d_raw_scores_, d_raw_boxes_, d_raw_classes_, max_raw,
                                cfg.narrow_bonus, cfg.person_class,
                                cfg.narrow_aspect_thresh, cfg.narrow_height_thresh,
                                cfg.height, stream_);
        }

        // 6. Workbench: filter + NMS + tracker
        int n_out = workbench.process_frame_postyolo(
            d_raw_boxes_, d_raw_scores_, d_raw_classes_, max_raw,
            cfg.width, cfg.height,
            /*is_tiled=*/false,
            /*priors_ptr=*/nullptr, /*prior_classes_ptr=*/nullptr,
            /*num_priors=*/0, /*prior_iou_threshold=*/0.5f,
            /*embeddings_ptr=*/nullptr,
            /*gmc_ptr=*/nullptr,
            /*light_factor=*/0.0f,
            /*mid_thresh_scale=*/1.0f,
            d_out_boxes_, d_out_scores_, d_out_ids_,
            d_out_classes_, d_out_det_idx_, d_out_count_);

        // 7. D2H copy + sync — destinations are pinned (h_out_*_pinned_), so
        // the driver does not allocate per-call staging buffers.
        cudaMemcpyAsync(h_out_count_pinned_, d_out_count_, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_);
        int actual = std::max(0, n_out);
        if (actual > 0) {
            cudaMemcpyAsync(h_ids,    d_out_ids_,    actual * sizeof(int),
                            cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(h_boxes,  d_out_boxes_,  actual * 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(h_scores, d_out_scores_, actual * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_);
        }
        cudaStreamSynchronize(stream_);

        // 8. Build FrameResult (post-tracker FP hard filter applied per-track)
        FrameResult res;
        res.frame_id = fi + 1;  // 1-indexed
        for (int k = 0; k < actual; ++k) {
            float score = h_scores[k];
            if (cfg.fp_hard_filter) {
                float bw = h_boxes[k*4+2] - h_boxes[k*4+0];
                float bh = h_boxes[k*4+3] - h_boxes[k*4+1];
                float area = bw * bh;
                if (score < cfg.fp_min_score) continue;
                if (area > cfg.fp_max_area && score < cfg.fp_max_susp_score) continue;
            }
            res.track_ids.push_back(h_ids[k]);
            res.boxes.push_back({h_boxes[k*4+0], h_boxes[k*4+1],
                                 h_boxes[k*4+2], h_boxes[k*4+3]});
            res.scores.push_back(score);
        }
        results.push_back(std::move(res));
    }

    return results;
}

} // namespace saccade
