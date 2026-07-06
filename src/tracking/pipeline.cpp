#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"
#include "tracking/copy_pad.cuh"
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
    if (d_private_nms_keep_) cudaFree(d_private_nms_keep_);
    if (d_private_nms_count_) cudaFree(d_private_nms_count_);
    if (d_private_added_count_) cudaFree(d_private_added_count_);
    if (d_private_baseline_mask_) cudaFree(d_private_baseline_mask_);
    delete crop_pool_; crop_pool_ = nullptr;
    delete crop_ring_; crop_ring_ = nullptr;
    if (d_stash_scratch_) cudaFree(d_stash_scratch_);
    if (d_requery_crops_) cudaFree(d_requery_crops_);
    if (d_requery_embeds_) cudaFree(d_requery_embeds_);
    if (requery_stream_) cudaStreamDestroy(requery_stream_);
    if (d_batch_buf_) cudaFree(d_batch_buf_);
    if (crop_stream_) cudaStreamDestroy(crop_stream_);
    if (reid_stream_) cudaStreamDestroy(reid_stream_);
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
    if (d_private_nms_keep_) cudaFree(d_private_nms_keep_);
    if (d_private_baseline_mask_) cudaFree(d_private_baseline_mask_);

    int col_blocks = (cap + 63) / 64;
    if (d_nms_suppression_) cudaFree(d_nms_suppression_);
    if (d_nms_remv_) cudaFree(d_nms_remv_);
    if (d_nms_immunity_mask_) cudaFree(d_nms_immunity_mask_);

    cudaMalloc(&d_filter_keep_indices_, cap * sizeof(int));
    cudaMalloc(&d_filter_suspect_flags_, cap * sizeof(bool));
    cudaMalloc(&d_nms_order_, cap * sizeof(int64_t));
    cudaMalloc(&d_nms_keep_, cap * sizeof(int));
    cudaMalloc(&d_private_nms_keep_, cap * sizeof(int));
    cudaMalloc(&d_private_baseline_mask_, cap * sizeof(bool));
    cudaMalloc(&d_nms_suppression_, (size_t)cap * col_blocks * sizeof(uint64_t));
    cudaMalloc(&d_nms_remv_, col_blocks * sizeof(uint64_t));
    cudaMalloc(&d_nms_immunity_mask_, cap * sizeof(bool));

    if (!d_filter_count_) cudaMalloc(&d_filter_count_, sizeof(int));
    if (!d_nms_count_) cudaMalloc(&d_nms_count_, sizeof(int));
    if (!d_private_nms_count_) cudaMalloc(&d_private_nms_count_, sizeof(int));
    if (!d_private_added_count_) cudaMalloc(&d_private_added_count_, sizeof(int));

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

void PerceptionPipeline::ensure_crop_pool() {
    if (!reid_ || !cropper_) return;
    auto [crop_h, crop_w] = reid_->get_input_hw();
    int desired_cap = crop_pool_capacity_ > 0 ? crop_pool_capacity_ : 256;
    if (crop_pool_ && crop_pool_->capacity() >= desired_cap
        && crop_pool_->crop_h() == crop_h && crop_pool_->crop_w() == crop_w) {
        // Pool already sized — just ensure streams + batch buffer exist.
    } else {
        delete crop_pool_;
        crop_pool_ = new CropPool(desired_cap, crop_h, crop_w);
    }
    // Create async streams (idempotent — only once).
    if (!crop_stream_) cudaStreamCreateWithFlags(&crop_stream_, cudaStreamNonBlocking);
    if (!reid_stream_) cudaStreamCreateWithFlags(&reid_stream_, cudaStreamNonBlocking);
    // Allocate / resize batch buffer.
    int max_batch = reid_->get_max_batch();
    int needed = max_batch * 3 * crop_h * crop_w;
    if (needed > batch_buf_capacity_ || crop_h != batch_buf_crop_h_ || crop_w != batch_buf_crop_w_) {
        if (d_batch_buf_) cudaFree(d_batch_buf_);
        cudaMalloc(&d_batch_buf_, needed * sizeof(float));
        batch_buf_capacity_ = needed;
        batch_buf_crop_h_ = crop_h;
        batch_buf_crop_w_ = crop_w;
    }
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
    // Reuse the persistent d_nms_count_ scratch (allocated in ensure_scratch)
    // instead of malloc/free-ing a 4-byte staging slot on every call.
    process_detections_into(
        boxes_ptr, scores_ptr, classes_ptr, n_in,
        frame_w, frame_h, is_tiled,
        out_boxes, out_scores, out_classes, out_suspect,
        d_nms_count_, nullptr, nullptr, 0, 0.5f, stream);
    int n_out = 0;
    cudaMemcpyAsync(&n_out, d_nms_count_, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
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
    cudaStream_t stream,
    const float* private_priors_ptr,
    int num_private_priors)
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
    if (cfg_.private_continuation_enabled && d_private_added_count_ != nullptr) {
        cudaMemsetAsync(d_private_added_count_, 0, sizeof(int), stream);
    }

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

    if (filter_count <= 64 && !cfg_.private_continuation_enabled) {
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
        const bool nms_class_aware = cfg_.private_continuation_enabled
            ? !cfg_.person_only
            : false;
        auto launch_large_nms = [&] {

            nms_counted_cuda(
                out_boxes, out_scores, out_classes, d_nms_order_,
                n_in, d_filter_count_, d_nms_keep_, d_nms_suppression_, d_nms_remv_,
                d_nms_count_, cfg_.nms_threshold, nms_class_aware,
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
        const float private_score_ceiling = cfg_.private_low_stage_only
            ? std::min(
                cfg_.private_new_track_thresh - cfg_.private_score_eps,
                cfg_.private_mid_thresh - cfg_.private_score_eps
            )
            : cfg_.private_new_track_thresh - cfg_.private_score_eps;
        const float private_score_floor = std::max(
            cfg_.private_min_score,
            cfg_.private_track_thresh
        );
        const bool run_private = (
            cfg_.private_continuation_enabled
            && cfg_.private_candidate_nms_iou > cfg_.nms_threshold
            && cfg_.private_new_track_thresh - cfg_.private_score_eps > cfg_.private_track_thresh
            && private_score_ceiling > cfg_.private_track_thresh
        );
        auto launch_private_candidate_nms = [&] {
            cudaMemsetAsync(
                d_nms_suppression_,
                0,
                (size_t)n_in * col_blocks * sizeof(uint64_t),
                stream
            );
            cudaMemsetAsync(d_nms_remv_, 0, col_blocks * sizeof(uint64_t), stream);
            cudaMemsetAsync(d_private_nms_count_, 0, sizeof(int), stream);
            nms_counted_cuda(
                out_boxes, out_scores, out_classes, d_nms_order_,
                n_in, d_filter_count_, d_private_nms_keep_,
                d_nms_suppression_, d_nms_remv_, d_private_nms_count_,
                cfg_.private_candidate_nms_iou, nms_class_aware,
                priors_ptr, prior_classes_ptr, num_priors, prior_iou_threshold,
                d_nms_immunity_mask_,
                stream);
        };
        auto launch_private_append = [&] {
            append_private_continuation_cuda(
                out_boxes,
                out_scores,
                out_classes,
                out_suspect,
                d_compact_boxes_,
                d_compact_scores_,
                d_compact_classes_,
                d_compact_suspect_,
                d_nms_count_,
                n_in,
                d_nms_keep_,
                d_nms_count_,
                d_private_baseline_mask_,
                d_private_nms_keep_,
                d_private_nms_count_,
                private_priors_ptr,
                num_private_priors,
                cfg_.private_prior_iou_threshold,
                cfg_.private_prior_center_threshold,
                private_score_floor,
                private_score_ceiling,
                cfg_.private_max_candidates,
                d_private_added_count_,
                stream);
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
            if (run_private) {
                measure_gpu_stage(
                    stream,
                    last_postprocess_profile_stats_.native_private_candidate_nms_ms,
                    launch_private_candidate_nms
                );
                measure_gpu_stage(
                    stream,
                    last_postprocess_profile_stats_.native_private_append_ms,
                    launch_private_append
                );
            }
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
            if (run_private) {
                launch_private_candidate_nms();
                launch_private_append();
            }
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
    cudaStream_t stream,
    const float* private_priors_ptr,
    int num_private_priors)
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
        stream,
        private_priors_ptr, num_private_priors);

    // Sync stream (GIL already released in Python binding) then read count.
    int n_post = 0;
    const Clock::time_point count_start = profile_post ? Clock::now() : Clock::time_point{};
    cudaMemcpyAsync(&n_post, d_count_staging, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (profile_post) {
        if (cfg_.private_continuation_enabled && d_private_added_count_ != nullptr) {
            int private_boxes = 0;
            cudaMemcpy(&private_boxes, d_private_added_count_, sizeof(int), cudaMemcpyDeviceToHost);
            last_postprocess_profile_stats_.private_boxes = private_boxes;
        }
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

void PerceptionPipeline::process_detections_graph(
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
    if (n_in <= 0) { cudaMemsetAsync(out_count, 0, sizeof(int), stream); return; }
    ensure_scratch(n_in, stream);

    cudaMemsetAsync(d_filter_count_, 0, sizeof(int), stream);
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

    gather_compact3_counted_cuda(
        boxes_ptr, scores_ptr, classes_ptr,
        out_boxes, out_scores, out_classes,
        d_filter_keep_indices_, d_filter_count_, n_in, stream);

    copy_bool_counted_cuda(
        d_filter_suspect_flags_, out_suspect, d_filter_count_, n_in, stream);

    const int col_blocks = (n_in + 63) / 64;
    cudaMemsetAsync(d_nms_suppression_, 0, (size_t)n_in * col_blocks * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_nms_remv_, 0, col_blocks * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_nms_count_, 0, sizeof(int), stream);

    argsort_scores_descending_cuda(
        out_scores, n_in,
        d_nms_order_, d_sort_keys_in_, d_sort_keys_out_,
        d_cub_sort_tmp_, cub_sort_tmp_bytes_, stream);

    nms_counted_cuda(
        out_boxes, out_scores, out_classes, d_nms_order_,
        n_in, d_filter_count_, d_nms_keep_, d_nms_suppression_, d_nms_remv_,
        d_nms_count_, cfg_.nms_threshold, false,
        priors_ptr, prior_classes_ptr, num_priors, prior_iou_threshold,
        d_nms_immunity_mask_,
        stream);

    gather_compact4_counted_cuda(
        out_boxes, out_scores, out_classes, out_suspect,
        d_compact_boxes_, d_compact_scores_, d_compact_classes_, d_compact_suspect_,
        d_nms_keep_, d_nms_count_, n_in, stream);

    cudaMemcpyAsync(out_boxes,   d_compact_boxes_,   n_in * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_scores,  d_compact_scores_,  n_in *     sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_classes, d_compact_classes_, n_in *     sizeof(int),   cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_suspect, d_compact_suspect_, n_in *     sizeof(bool),  cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_count, d_nms_count_, sizeof(int), cudaMemcpyDeviceToDevice, stream);

    if (cfg_.geometry_suspect_support && cfg_.geometry_suspect_support_score > 0.0f) {
        penalize_suspect_scores_cuda(
            out_scores, out_suspect, out_count,
            cfg_.geometry_suspect_support_score, n_in, stream);
    }
}

void PerceptionPipeline::process_detections_interleaved_graph(
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
    cudaStream_t stream)
{
    interleaved_to_split(det_6d, n_in, split_boxes, split_scores, split_classes, stream);
    process_detections_graph(
        split_boxes, split_scores, split_classes, n_in,
        frame_w, frame_h, is_tiled,
        out_boxes, out_scores, out_classes, out_suspect, out_count,
        nullptr, nullptr, 0, 0.0f, stream);
}

void PerceptionPipeline::crop_into_pool(
    const float* frame_ptr, int frame_h, int frame_w,
    const float* boxes_ptr, int n_boxes,
    int* out_slot,
    cudaStream_t stream)
{
    *out_slot = -1;
    if (!reid_ || !cropper_ || n_boxes <= 0) return;
    ensure_crop_pool();
    int slot = crop_pool_->acquire(n_boxes);
    if (slot < 0) return;
    *out_slot = slot;
    float* crop_ptr = crop_pool_->slot_ptr(slot);
    if (reid_profiling_enabled_) {
        last_reid_profile_stats_.images = n_boxes;
    }
    if (reid_profiling_enabled_) {
        cudaEvent_t start = nullptr, stop = nullptr;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        cropper_->process_gpu(
            const_cast<void*>(reinterpret_cast<const void*>(frame_ptr)),
            frame_w, frame_h,
            const_cast<float*>(boxes_ptr), n_boxes,
            crop_ptr, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        last_reid_profile_stats_.crop_ms += ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        cropper_->process_gpu(
            const_cast<void*>(reinterpret_cast<const void*>(frame_ptr)),
            frame_w, frame_h,
            const_cast<float*>(boxes_ptr), n_boxes,
            crop_ptr, stream);
    }
}

void PerceptionPipeline::extract_from_pool(
    int slot, int n_boxes,
    float* out_embeds,
    cudaStream_t stream)
{
    if (!reid_ || !crop_pool_ || slot < 0 || n_boxes <= 0) return;
    float* crop_ptr = crop_pool_->slot_ptr(slot);
    reid_->extract(crop_ptr, n_boxes, out_embeds, stream);
    if (reid_profiling_enabled_) {
        const auto feature_stats = reid_->get_profile_stats();
        last_reid_profile_stats_.extract_pre_normalize_ms = feature_stats.pre_normalize_ms;
        last_reid_profile_stats_.extract_trt_enqueue_ms = feature_stats.trt_enqueue_ms;
        last_reid_profile_stats_.extract_l2_normalize_ms = feature_stats.l2_normalize_ms;
        last_reid_profile_stats_.extract_total_ms = feature_stats.total_ms;
        last_reid_profile_stats_.chunks = feature_stats.chunks;
        last_reid_profile_stats_.total_ms =
            last_reid_profile_stats_.crop_ms + last_reid_profile_stats_.extract_total_ms;
    }
    crop_pool_->release(slot, n_boxes);
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
    int slot = -1;
    crop_into_pool(frame_ptr, frame_h, frame_w, boxes_ptr, n_boxes, &slot, stream);
    if (slot < 0) return;
    extract_from_pool(slot, n_boxes, out_embeds, stream);
}

// ── Borderline re-query crop ring (ReidCropStore) ──────────────────────────
void PerceptionPipeline::enable_crop_ring(int capacity, int depth) {
    if (!reid_) return;
    auto [crop_h, crop_w] = reid_->get_input_hw();
    delete crop_ring_;
    crop_ring_ = new CropRingStore(capacity, depth, crop_h, crop_w);
    if (!requery_stream_) cudaStreamCreate(&requery_stream_);
}

void PerceptionPipeline::ensure_stash_scratch(int n_boxes) {
    if (!crop_ring_ || n_boxes <= 0) return;
    if (d_stash_scratch_ && stash_scratch_n_ >= n_boxes) return;
    if (d_stash_scratch_) cudaFree(d_stash_scratch_);
    stash_scratch_n_ = n_boxes;
    size_t bytes =
        static_cast<size_t>(n_boxes) * crop_ring_->crop_elem_count() * sizeof(float);
    cudaMalloc(&d_stash_scratch_, bytes);
}

void PerceptionPipeline::ensure_requery_scratch(int n_rows) {
    if (!crop_ring_ || !reid_ || n_rows <= 0) return;
    if (d_requery_crops_ && requery_scratch_n_ >= n_rows) return;
    if (d_requery_crops_) cudaFree(d_requery_crops_);
    if (d_requery_embeds_) cudaFree(d_requery_embeds_);
    requery_scratch_n_ = n_rows;
    cudaMalloc(&d_requery_crops_,
               static_cast<size_t>(n_rows) * crop_ring_->crop_elem_count()
                   * sizeof(float));
    cudaMalloc(&d_requery_embeds_,
               static_cast<size_t>(n_rows) * reid_->get_feature_dim()
                   * sizeof(float));
}

void PerceptionPipeline::stash_crops(
    const uint64_t* uids, const int* frames,
    const float* frame_ptr, int frame_h, int frame_w,
    const float* boxes_ptr, int n_boxes,
    const bool* clean,
    cudaStream_t stream)
{
    if (!crop_ring_ || !cropper_ || n_boxes <= 0 || !uids || !boxes_ptr) return;
    ensure_stash_scratch(n_boxes);
    if (!d_stash_scratch_) return;
    // Crop all boxes contiguously into the dedicated stash scratch (not the
    // hot-path crop_pool_, so there is no cross-stream slot-reuse hazard), then
    // copy each crop into its uid's ring slot — all on @p stream, in order.
    cropper_->process_gpu(
        const_cast<void*>(reinterpret_cast<const void*>(frame_ptr)),
        frame_w, frame_h,
        const_cast<float*>(boxes_ptr), n_boxes,
        d_stash_scratch_, stream);
    // One batched scatter instead of per-crop stash: the per-frame stash of
    // every confirmed track is launch-bound (3 CUDA API calls per crop).
    crop_ring_->stash_batch(uids, frames, d_stash_scratch_, clean, n_boxes,
                            stream);
}

int PerceptionPipeline::embed_dim() const {
    return reid_ ? reid_->get_feature_dim() : 0;
}

int PerceptionPipeline::ring_depth() const {
    return crop_ring_ ? crop_ring_->depth() : 0;
}

int PerceptionPipeline::requery_extract(const uint64_t* uids, int n_uids,
                                        float* out_embeds, int* out_counts) {
    if (!crop_ring_ || !reid_ || n_uids <= 0 || !uids || !out_embeds) return 0;
    if (!requery_stream_) cudaStreamCreate(&requery_stream_);
    const int max_rows = crop_ring_->depth() * n_uids;
    ensure_requery_scratch(max_rows);
    if (!d_requery_crops_ || !d_requery_embeds_) return 0;
    // Clean crops only: the validated dense recent-tail substrate excludes
    // occluded crops (mnv4 is pollution-sensitive). A uid with no clean crops
    // gets count 0 and keeps its sparse bank in the rescore.
    int total = crop_ring_->gather_many(uids, n_uids, d_requery_crops_,
                                        out_counts, requery_stream_,
                                        /*clean_only=*/true);
    if (total <= 0) return 0;
    reid_->extract(d_requery_crops_, total, d_requery_embeds_, requery_stream_);
    cudaStreamSynchronize(requery_stream_);
    const int dim = reid_->get_feature_dim();
    cudaMemcpy(out_embeds, d_requery_embeds_,
               static_cast<size_t>(total) * dim * sizeof(float),
               cudaMemcpyDeviceToHost);
    return total;
}

void PerceptionPipeline::evict(uint64_t uid) {
    if (crop_ring_) crop_ring_->evict(uid);
}

int PerceptionPipeline::gather_crops_framed(const uint64_t* uids,
                                            const int* frames, int n,
                                            float* batch, cudaStream_t stream,
                                            bool clean_only) {
    if (!crop_ring_) return 0;
    return crop_ring_->gather_framed(uids, frames, n, batch, stream, clean_only);
}

bool PerceptionPipeline::has_crop(uint64_t uid, int frame, bool clean_only) const {
    if (!crop_ring_) return false;
    return crop_ring_->has_crop(uid, frame, clean_only);
}

void PerceptionPipeline::crop_into_pool_async(
    const float* frame_ptr, int frame_h, int frame_w,
    const float* boxes_ptr, int n_boxes,
    int* out_slot,
    cudaEvent_t* out_event)
{
    *out_slot = -1;
    *out_event = nullptr;
    if (!reid_ || !cropper_ || n_boxes <= 0) return;
    ensure_crop_pool();
    int slot = crop_pool_->acquire(n_boxes);
    if (slot < 0) return;
    *out_slot = slot;
    // Create event for crop completion signaling.
    cudaEvent_t evt;
    cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
    *out_event = evt;
    float* crop_ptr = crop_pool_->slot_ptr(slot);
    // The frame tensor was allocated on the caller's (default) stream.
    // Before reading it on crop_stream_, wait for the default stream to
    // finish any pending writes to the frame.
    cudaEvent_t frame_ready;
    cudaEventCreateWithFlags(&frame_ready, cudaEventDisableTiming);
    cudaEventRecord(frame_ready, 0);  // record on default stream
    cudaStreamWaitEvent(crop_stream_, frame_ready);
    cudaEventDestroy(frame_ready);
    cropper_->process_gpu(
        const_cast<void*>(reinterpret_cast<const void*>(frame_ptr)),
        frame_w, frame_h,
        const_cast<float*>(boxes_ptr), n_boxes,
        crop_ptr, crop_stream_);
    cudaEventRecord(evt, crop_stream_);
}

int PerceptionPipeline::extract_batch_from_pool(
    const ReIDCropJob* jobs, int n_jobs,
    float* out_embeds,
    ReIDResult* out_results)
{
    if (!reid_ || !crop_pool_ || n_jobs <= 0) return 0;
    ensure_crop_pool();
    int max_batch = reid_->get_max_batch();
    auto [crop_h, crop_w] = reid_->get_input_hw();
    int crop_elem = 3 * crop_h * crop_w;
    int total_crops = 0;
    for (int j = 0; j < n_jobs; ++j) {
        total_crops += jobs[j].n_crops;
    }
    if (total_crops <= 0) return 0;
    if (total_crops > max_batch) {
        // Caller should chunk — for now, process up to max_batch.
        total_crops = max_batch;
    }

    // Phase 2: gather crops from pool slots to contiguous batch buffer.
    // For each job, wait on its crop_ready event, then copy crops to batch_buf.
    int batch_offset = 0;
    int results_idx = 0;
    for (int j = 0; j < n_jobs && batch_offset < max_batch; ++j) {
        const auto& job = jobs[j];
        if (job.crop_slot < 0 || job.n_crops <= 0) continue;
        // Wait for crop to complete on crop_stream before reading on reid_stream.
        if (job.crop_ready) {
            cudaStreamWaitEvent(reid_stream_, job.crop_ready);
        }
        float* src = crop_pool_->slot_ptr(job.crop_slot);
        float* dst = d_batch_buf_ + static_cast<size_t>(batch_offset) * crop_elem;
        int n_copy = job.n_crops;
        if (batch_offset + n_copy > max_batch) {
            n_copy = max_batch - batch_offset;
        }
        cudaMemcpyAsync(dst, src,
                        static_cast<size_t>(n_copy) * crop_elem * sizeof(float),
                        cudaMemcpyDeviceToDevice, reid_stream_);
        // Record result metadata.
        if (out_results) {
            out_results[results_idx].frame_idx = job.frame_idx;
            out_results[results_idx].track_uid = job.track_uid;
            out_results[results_idx].generation = job.generation;
            out_results[results_idx].reason = job.reason;
            out_results[results_idx].embed_offset = batch_offset;
            out_results[results_idx].n_crops = n_copy;
        }
        batch_offset += n_copy;
        ++results_idx;
    }
    int n_to_extract = batch_offset;
    if (n_to_extract <= 0) return 0;

    // Run TensorRT inference on the contiguous batch buffer.
    reid_->extract(d_batch_buf_, n_to_extract, out_embeds, reid_stream_);

    // Release pool slots (after the copy is done — but we need to sync to
    // ensure the D2D copy has completed before releasing.  Since extract
    // also runs on reid_stream_ and is ordered after the copies, we can
    // release after the extract completes.  Use a sync for Phase 2; Phase 3
    // can use an event for deferred release.)
    cudaStreamSynchronize(reid_stream_);

    for (int j = 0; j < n_jobs; ++j) {
        const auto& job = jobs[j];
        if (job.crop_slot >= 0 && job.n_crops > 0) {
            crop_pool_->release(job.crop_slot, job.n_crops);
        }
        if (job.crop_ready) {
            cudaEventDestroy(job.crop_ready);
        }
    }

    return n_to_extract;
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
