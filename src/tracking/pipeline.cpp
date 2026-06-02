#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace saccade {

namespace {

template <typename Fn>
void measure_gpu_stage(cudaStream_t stream, double& out_ms, Fn&& fn) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    fn();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    out_ms = static_cast<double>(ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace

PerceptionPipeline::PerceptionPipeline(FeatureExtractor* reid, Cropper* cropper, Config cfg)
    : reid_(reid), cropper_(cropper), cfg_(cfg) {
    if (cfg_.max_detections > 0)
        ensure_scratch(cfg_.max_detections, nullptr);
}

PerceptionPipeline::~PerceptionPipeline() {
    if (d_filter_keep_indices_) cudaFree(d_filter_keep_indices_);
    if (d_filter_suspect_flags_) cudaFree(d_filter_suspect_flags_);
    if (d_filter_count_) cudaFree(d_filter_count_);
    if (d_nms_order_) cudaFree(d_nms_order_);
    if (d_nms_keep_) cudaFree(d_nms_keep_);
    if (d_nms_suppression_) cudaFree(d_nms_suppression_);
    if (d_nms_remv_) cudaFree(d_nms_remv_);
    if (d_nms_count_) cudaFree(d_nms_count_);
    if (d_nms_immunity_mask_) cudaFree(d_nms_immunity_mask_);
    if (d_crop_buf_) cudaFree(d_crop_buf_);
    if (d_compact_boxes_)   cudaFree(d_compact_boxes_);
    if (d_compact_scores_)  cudaFree(d_compact_scores_);
    if (d_compact_classes_) cudaFree(d_compact_classes_);
    if (d_compact_suspect_) cudaFree(d_compact_suspect_);
    if (d_sort_keys_in_)    cudaFree(d_sort_keys_in_);
    if (d_sort_keys_out_)   cudaFree(d_sort_keys_out_);
    if (d_cub_sort_tmp_)    cudaFree(d_cub_sort_tmp_);
}

void PerceptionPipeline::ensure_scratch(int n_dets, cudaStream_t /*stream*/) {
    if (n_dets <= scratch_capacity_) return;
    int cap = std::max(n_dets, 256);
    if (d_filter_keep_indices_) cudaFree(d_filter_keep_indices_);
    if (d_filter_suspect_flags_) cudaFree(d_filter_suspect_flags_);
    if (d_nms_order_) cudaFree(d_nms_order_);
    if (d_nms_keep_) cudaFree(d_nms_keep_);

    int col_blocks = (cap + 63) / 64;
    if (d_nms_suppression_) cudaFree(d_nms_suppression_);
    if (d_nms_remv_) cudaFree(d_nms_remv_);
    if (d_nms_immunity_mask_) cudaFree(d_nms_immunity_mask_);

    cudaMalloc(&d_filter_keep_indices_, cap * sizeof(int));
    cudaMalloc(&d_filter_suspect_flags_, cap * sizeof(bool));
    cudaMalloc(&d_nms_order_, cap * sizeof(int64_t));
    cudaMalloc(&d_nms_keep_, cap * sizeof(int));
    cudaMalloc(&d_nms_suppression_, (size_t)cap * col_blocks * sizeof(uint64_t));
    cudaMalloc(&d_nms_remv_, col_blocks * sizeof(uint64_t));
    cudaMalloc(&d_nms_immunity_mask_, cap * sizeof(bool));

    if (!d_filter_count_) cudaMalloc(&d_filter_count_, sizeof(int));
    if (!d_nms_count_) cudaMalloc(&d_nms_count_, sizeof(int));

    // M1: GPU compaction scratch
    if (d_compact_boxes_)   cudaFree(d_compact_boxes_);
    if (d_compact_scores_)  cudaFree(d_compact_scores_);
    if (d_compact_classes_) cudaFree(d_compact_classes_);
    if (d_compact_suspect_) cudaFree(d_compact_suspect_);
    if (d_sort_keys_in_)    cudaFree(d_sort_keys_in_);
    if (d_sort_keys_out_)   cudaFree(d_sort_keys_out_);
    cudaMalloc(&d_compact_boxes_,   cap * 4 * sizeof(float));
    cudaMalloc(&d_compact_scores_,  cap * sizeof(float));
    cudaMalloc(&d_compact_classes_, cap * sizeof(int));
    cudaMalloc(&d_compact_suspect_, cap * sizeof(bool));
    // sort keys buffers
    cudaMalloc(&d_sort_keys_in_,    cap * sizeof(uint64_t));
    cudaMalloc(&d_sort_keys_out_,   cap * sizeof(uint64_t));

    size_t new_tmp = argsort_scores_descending_bytes(cap);
    if (new_tmp > cub_sort_tmp_bytes_) {
        if (d_cub_sort_tmp_) cudaFree(d_cub_sort_tmp_);
        cudaMalloc(&d_cub_sort_tmp_, new_tmp);
        cub_sort_tmp_bytes_ = new_tmp;
    }

    scratch_capacity_ = cap;
}

void PerceptionPipeline::ensure_crop_buf(int n_boxes) {
    if (!reid_ || !cropper_) return;
    auto [crop_h, crop_w] = reid_->get_input_hw();
    int needed = n_boxes * 3 * crop_h * crop_w;
    if (needed <= crop_buf_capacity_) return;
    if (d_crop_buf_) cudaFree(d_crop_buf_);
    cudaMalloc(&d_crop_buf_, needed * sizeof(float));
    crop_buf_capacity_ = needed;
}

int PerceptionPipeline::process_detections(
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
    cudaStream_t stream)
{
    if (n_in <= 0) return 0;
    int* d_out_count = nullptr;
    cudaMalloc(&d_out_count, sizeof(int));
    process_detections_into(
        boxes_ptr, scores_ptr, classes_ptr, n_in,
        frame_w, frame_h, is_tiled,
        out_boxes, out_scores, out_classes, out_suspect,
        d_out_count, nullptr, nullptr, 0, 0.5f, stream);
    int n_out = 0;
    cudaMemcpyAsync(&n_out, d_out_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_out_count);
    return n_out;
}

void PerceptionPipeline::process_detections_into(
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
    cudaStream_t stream)
{
    if (n_in <= 0) {
        cudaMemsetAsync(out_count, 0, sizeof(int), stream);
        if (postprocess_profiling_enabled_) {
            last_postprocess_profile_stats_.input_boxes = 0;
            last_postprocess_profile_stats_.filtered_boxes = 0;
            last_postprocess_profile_stats_.output_boxes = 0;
        }
        return;
    }
    ensure_scratch(n_in, stream);

    using Clock = std::chrono::steady_clock;
    const bool profile_post = postprocess_profiling_enabled_;
    const Clock::time_point filter_start = profile_post ? Clock::now() : Clock::time_point{};

    cudaMemsetAsync(d_filter_count_, 0, sizeof(int), stream);
    auto launch_filter_kernel = [&] {
        filter_detections_cuda(
            boxes_ptr, scores_ptr, classes_ptr, n_in,
            d_filter_keep_indices_, d_filter_suspect_flags_, nullptr, d_filter_count_,
            cfg_.score_threshold,
            cfg_.person_only, cfg_.person_class,
            is_tiled, frame_w, frame_h,
            cfg_.person_geometry_prior, cfg_.geometry_suspect_support,
            cfg_.person_min_height_ratio,
            cfg_.person_min_aspect, cfg_.person_max_aspect,
            cfg_.person_min_area_ratio, cfg_.person_max_area_ratio,
            stream);
    };
    auto launch_gather_compact3 = [&] {
        gather_compact3_counted_cuda(
            boxes_ptr, scores_ptr, classes_ptr,
            out_boxes, out_scores, out_classes,
            d_filter_keep_indices_, d_filter_count_, n_in, stream);
    };
    auto launch_copy_suspect = [&] {
        copy_bool_counted_cuda(
            d_filter_suspect_flags_, out_suspect, d_filter_count_, n_in, stream
        );
    };
    if (profile_post) {
        measure_gpu_stage(
            stream,
            last_postprocess_profile_stats_.native_filter_kernel_ms,
            launch_filter_kernel
        );
    } else {
        launch_filter_kernel();
    }

    int filter_count = 0;
    const Clock::time_point filter_count_sync_start = profile_post ? Clock::now() : Clock::time_point{};
    cudaMemcpyAsync(&filter_count, d_filter_count_, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (profile_post) {
        last_postprocess_profile_stats_.filter_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - filter_start).count();
        last_postprocess_profile_stats_.native_filter_count_sync_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - filter_count_sync_start).count();
        last_postprocess_profile_stats_.filtered_boxes = filter_count;
    }

    if (filter_count <= 1) {
        checkCuda(cudaMemcpyAsync(
            out_count,
            &filter_count,
            sizeof(int),
            cudaMemcpyHostToDevice,
            stream
        ));
        if (
            cfg_.geometry_suspect_support
            && cfg_.geometry_suspect_support_score > 0.0f
            && filter_count > 0
        ) {
            auto launch_suspect_penalty = [&] {
                penalize_suspect_scores_cuda(
                    out_scores,
                    out_suspect,
                    out_count,
                    cfg_.geometry_suspect_support_score,
                    n_in,
                    stream
                );
            };
            if (profile_post) {
                measure_gpu_stage(
                    stream,
                    last_postprocess_profile_stats_.native_suspect_penalty_ms,
                    launch_suspect_penalty
                );
            } else {
                launch_suspect_penalty();
            }
        }
        if (profile_post) {
            cudaStreamSynchronize(stream);
            last_postprocess_profile_stats_.nms_ms = 0.0;
            last_postprocess_profile_stats_.output_boxes = filter_count;
        }
        return;
    }

    const Clock::time_point nms_start = profile_post ? Clock::now() : Clock::time_point{};

    if (filter_count <= 64) {
        // Small path: skip gather/copy — compact_grid_nms reads raw data
        // via keep_indices and writes final output in one merged kernel.
        if (profile_post) {
            last_postprocess_profile_stats_.native_gather_compact3_ms = 0.0;
            last_postprocess_profile_stats_.native_copy_suspect_ms = 0.0;
            last_postprocess_profile_stats_.native_filter_gather_ms =
                last_postprocess_profile_stats_.native_filter_kernel_ms;
        }
        auto launch_small_nms = [&] {
            compact_grid_nms_cuda(
                boxes_ptr, scores_ptr, classes_ptr,
                n_in,
                d_filter_keep_indices_, filter_count,
                out_boxes, out_scores, out_classes, out_suspect, out_count,
                cfg_.nms_threshold,
                stream);
        };
        if (profile_post) {
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_small_nms_ms,
                launch_small_nms
            );
        } else {
            launch_small_nms();
        }
    } else {
        // Large path: gather+copied arrays needed for CUB sort + bitmask NMS.
        if (profile_post) {
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_gather_compact3_ms,
                launch_gather_compact3
            );
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_copy_suspect_ms,
                launch_copy_suspect
            );
            last_postprocess_profile_stats_.native_filter_gather_ms =
                last_postprocess_profile_stats_.native_filter_kernel_ms
                + last_postprocess_profile_stats_.native_gather_compact3_ms
                + last_postprocess_profile_stats_.native_copy_suspect_ms;
        } else {
            launch_gather_compact3();
            launch_copy_suspect();
        }
        const int col_blocks = (n_in + 63) / 64;
        cudaMemsetAsync(d_nms_suppression_, 0, (size_t)n_in * col_blocks * sizeof(uint64_t), stream);
        cudaMemsetAsync(d_nms_remv_, 0, col_blocks * sizeof(uint64_t), stream);
        cudaMemsetAsync(d_nms_count_, 0, sizeof(int), stream);

        auto launch_large_argsort = [&] {
            argsort_scores_descending_cuda(
                out_scores, n_in,
                d_nms_order_, d_sort_keys_in_, d_sort_keys_out_,
                d_cub_sort_tmp_, cub_sort_tmp_bytes_, stream);
        };
        auto launch_large_nms = [&] {

            nms_counted_cuda(
                out_boxes, out_scores, out_classes, d_nms_order_,
                n_in, d_filter_count_, d_nms_keep_, d_nms_suppression_, d_nms_remv_,
                d_nms_count_, cfg_.nms_threshold, false,
                priors_ptr, prior_classes_ptr, num_priors, prior_iou_threshold,
                d_nms_immunity_mask_,
                stream);
        };
        if (profile_post) {
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_large_argsort_ms,
                launch_large_argsort
            );
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_large_nms_ms,
                launch_large_nms
            );
            last_postprocess_profile_stats_.native_large_sort_nms_ms =
                last_postprocess_profile_stats_.native_large_argsort_ms
                + last_postprocess_profile_stats_.native_large_nms_ms;
        } else {
            launch_large_argsort();
            launch_large_nms();
        }

        auto launch_large_gather4 = [&] {
            gather_compact4_counted_cuda(
                out_boxes, out_scores, out_classes, out_suspect,
                d_compact_boxes_, d_compact_scores_, d_compact_classes_, d_compact_suspect_,
                d_nms_keep_, d_nms_count_, n_in, stream);
        };
        auto launch_large_copyback = [&] {
            cudaMemcpyAsync(out_boxes,   d_compact_boxes_,   n_in * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_scores,  d_compact_scores_,  n_in *     sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_classes, d_compact_classes_, n_in *     sizeof(int),   cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_suspect, d_compact_suspect_, n_in *     sizeof(bool),  cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_count, d_nms_count_, sizeof(int), cudaMemcpyDeviceToDevice, stream);
        };
        if (profile_post) {
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_large_gather4_ms,
                launch_large_gather4
            );
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_large_copyback_ms,
                launch_large_copyback
            );
            last_postprocess_profile_stats_.native_compact_copy_ms =
                last_postprocess_profile_stats_.native_large_gather4_ms
                + last_postprocess_profile_stats_.native_large_copyback_ms;
        } else {
            launch_large_gather4();
            launch_large_copyback();
        }
    }

    if (cfg_.geometry_suspect_support && cfg_.geometry_suspect_support_score > 0.0f) {
        auto launch_suspect_penalty = [&] {
            penalize_suspect_scores_cuda(
                out_scores,
                out_suspect,
                out_count,
                cfg_.geometry_suspect_support_score,
                n_in,
                stream
            );
        };
        if (profile_post) {
            measure_gpu_stage(
                stream,
                last_postprocess_profile_stats_.native_suspect_penalty_ms,
                launch_suspect_penalty
            );
        } else {
            launch_suspect_penalty();
        }
    }

    if (profile_post) {
        cudaStreamSynchronize(stream);
        last_postprocess_profile_stats_.nms_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - nms_start).count();
    }
}

int PerceptionPipeline::process_detections_n(
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
    cudaStream_t stream)
{
    if (n_in <= 0) return 0;

    using Clock = std::chrono::steady_clock;
    const bool profile_post = postprocess_profiling_enabled_;
    if (profile_post) {
        reset_postprocess_profile_stats();
        last_postprocess_profile_stats_.input_boxes = n_in;
    }
    const Clock::time_point total_start = profile_post ? Clock::now() : Clock::time_point{};

    // Reuse d_filter_count_ as a pinned staging slot for the final count.
    // process_detections_into writes d_nms_count_ → out_count (D2D).
    // We'll read d_filter_count_ after the stream sync instead.
    int* d_count_staging = d_filter_count_;  // safe to reuse after filter stage
    cudaMemsetAsync(d_count_staging, 0, sizeof(int), stream);

    process_detections_into(
        boxes_ptr, scores_ptr, classes_ptr, n_in,
        frame_w, frame_h, is_tiled,
        out_boxes, out_scores, out_classes, out_suspect,
        d_count_staging,        // out_count written D2D here
        priors_ptr, prior_classes_ptr, num_priors, prior_iou_threshold,
        stream);

    // Sync stream (GIL already released in Python binding) then read count.
    int n_post = 0;
    const Clock::time_point count_start = profile_post ? Clock::now() : Clock::time_point{};
    cudaMemcpyAsync(&n_post, d_count_staging, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (profile_post) {
        last_postprocess_profile_stats_.count_d2h_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - count_start).count();
        last_postprocess_profile_stats_.total_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - total_start).count();
        last_postprocess_profile_stats_.output_boxes = n_post;
    }
    return n_post;
}

int PerceptionPipeline::process_detections_n_fixed(
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
    cudaStream_t stream)
{
    if (n_in <= 0) return 0;

    // Run the same filter+NMS pipeline as process_detections_into, which includes
    // the filter_count D2H sync used for small/large path selection and NMS efficiency.
    // The gather_compact*_counted kernels already zero-pad output beyond the actual
    // post-NMS count, so no additional work is needed.
    // The ONLY difference from process_detections_n: we skip the final n_post D2H read
    // and return n_in (fixed size) instead of the actual post-NMS count.
    int* d_count_staging = d_filter_count_;
    cudaMemsetAsync(d_count_staging, 0, sizeof(int), stream);
    process_detections_into(
        boxes_ptr, scores_ptr, classes_ptr, n_in,
        frame_w, frame_h, is_tiled,
        out_boxes, out_scores, out_classes, out_suspect,
        d_count_staging,
        priors_ptr, prior_classes_ptr, num_priors, prior_iou_threshold,
        stream);

    // Output layout: [0..n_nms_survived] valid, [n_nms_survived+1..n_in-1] zero-padded.
    // Caller feeds this directly to GraphedTrackerUpdate; zero-score slots are gated
    // by track_thresh inside the tracker kernels without any CPU synchronisation.
    return n_in;
}

void PerceptionPipeline::extract_reid(
    const float* frame_ptr, int frame_h, int frame_w,
    const float* boxes_ptr, int n_boxes,
    float* out_embeds,
    cudaStream_t stream)
{
    if (!reid_ || !cropper_ || n_boxes <= 0) return;
    if (reid_profiling_enabled_) {
        reset_reid_profile_stats();
        last_reid_profile_stats_.images = n_boxes;
    }
    ensure_crop_buf(n_boxes);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (reid_profiling_enabled_) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    if (reid_profiling_enabled_) cudaEventRecord(start, stream);
    cropper_->process_gpu(
        const_cast<void*>(reinterpret_cast<const void*>(frame_ptr)),
        frame_w, frame_h,
        const_cast<float*>(boxes_ptr), n_boxes,
        d_crop_buf_, stream);
    if (reid_profiling_enabled_) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        last_reid_profile_stats_.crop_ms += ms;
    }
    reid_->extract(d_crop_buf_, n_boxes, out_embeds, stream);
    if (reid_profiling_enabled_) {
        const auto feature_stats = reid_->get_profile_stats();
        last_reid_profile_stats_.extract_pre_normalize_ms = feature_stats.pre_normalize_ms;
        last_reid_profile_stats_.extract_trt_enqueue_ms = feature_stats.trt_enqueue_ms;
        last_reid_profile_stats_.extract_l2_normalize_ms = feature_stats.l2_normalize_ms;
        last_reid_profile_stats_.extract_total_ms = feature_stats.total_ms;
        last_reid_profile_stats_.chunks = feature_stats.chunks;
        last_reid_profile_stats_.total_ms =
            last_reid_profile_stats_.crop_ms + last_reid_profile_stats_.extract_total_ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int PerceptionPipeline::get_embed_dim() const {
    return reid_ ? reid_->get_feature_dim() : 0;
}

void PerceptionPipeline::set_reid_profiling_enabled(bool enabled) {
    reid_profiling_enabled_ = enabled;
    if (reid_) {
        reid_->set_profiling_enabled(enabled);
    }
}

void PerceptionPipeline::reset_reid_profile_stats() {
    last_reid_profile_stats_ = ReIDProfileStats{};
    if (reid_) {
        reid_->reset_profile_stats();
    }
}

PerceptionPipeline::ReIDProfileStats PerceptionPipeline::get_reid_profile_stats() const {
    return last_reid_profile_stats_;
}

void PerceptionPipeline::set_postprocess_profiling_enabled(bool enabled) {
    postprocess_profiling_enabled_ = enabled;
}

void PerceptionPipeline::reset_postprocess_profile_stats() {
    last_postprocess_profile_stats_ = PostprocessProfileStats{};
}

PerceptionPipeline::PostprocessProfileStats PerceptionPipeline::get_postprocess_profile_stats() const {
    return last_postprocess_profile_stats_;
}

} // namespace saccade
