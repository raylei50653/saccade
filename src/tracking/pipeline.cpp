#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace saccade {

PerceptionPipeline::PerceptionPipeline(FeatureExtractor* reid, Cropper* cropper, Config cfg)
    : reid_(reid), cropper_(cropper), cfg_(cfg) {}

PerceptionPipeline::~PerceptionPipeline() {
    if (d_filter_keep_indices_) cudaFree(d_filter_keep_indices_);
    if (d_filter_suspect_flags_) cudaFree(d_filter_suspect_flags_);
    if (d_filter_count_) cudaFree(d_filter_count_);
    if (d_nms_order_) cudaFree(d_nms_order_);
    if (d_nms_keep_) cudaFree(d_nms_keep_);
    if (d_nms_suppression_) cudaFree(d_nms_suppression_);
    if (d_nms_remv_) cudaFree(d_nms_remv_);
    if (d_nms_count_) cudaFree(d_nms_count_);
    if (d_crop_buf_) cudaFree(d_crop_buf_);
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

    cudaMalloc(&d_filter_keep_indices_, cap * sizeof(int));
    cudaMalloc(&d_filter_suspect_flags_, cap * sizeof(bool));
    cudaMalloc(&d_nms_order_, cap * sizeof(int64_t));
    cudaMalloc(&d_nms_keep_, cap * sizeof(int));
    cudaMalloc(&d_nms_suppression_, (size_t)cap * col_blocks * sizeof(uint64_t));
    cudaMalloc(&d_nms_remv_, col_blocks * sizeof(uint64_t));

    if (!d_filter_count_) cudaMalloc(&d_filter_count_, sizeof(int));
    if (!d_nms_count_) cudaMalloc(&d_nms_count_, sizeof(int));
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

// Helper: argsort scores descending on CPU using pinned host copy
static void fill_order_descending(
    const float* d_scores, int n, int64_t* d_order, cudaStream_t stream)
{
    std::vector<float> h_scores(n);
    cudaMemcpyAsync(h_scores.data(), d_scores, n * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<int64_t> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int64_t a, int64_t b){ return h_scores[a] > h_scores[b]; });
    cudaMemcpyAsync(d_order, order.data(), n * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
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
    ensure_scratch(n_in, stream);

    // Stage 1: filter (score threshold + geometry prior)
    cudaMemsetAsync(d_filter_count_, 0, sizeof(int), stream);
    filter_detections_cuda(
        boxes_ptr, scores_ptr, classes_ptr, n_in,
        d_filter_keep_indices_, d_filter_suspect_flags_, d_filter_count_,
        cfg_.score_threshold,
        cfg_.person_only, cfg_.person_class,
        is_tiled, frame_w, frame_h,
        cfg_.person_geometry_prior, cfg_.geometry_suspect_support,
        cfg_.person_min_height_ratio,
        cfg_.person_min_aspect, cfg_.person_max_aspect,
        cfg_.person_min_area_ratio, cfg_.person_max_area_ratio,
        stream);
    int n_filtered = 0;
    cudaMemcpyAsync(&n_filtered, d_filter_count_, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (n_filtered <= 0) return 0;

    // Compact filtered arrays into out_* buffers
    // We use host-side compaction (already synced) for simplicity
    {
        std::vector<int>   h_keep(n_filtered);
        std::vector<uint8_t>  h_suspect(n_filtered);
        cudaMemcpy(h_keep.data(), d_filter_keep_indices_, n_filtered * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_suspect.data(), d_filter_suspect_flags_, n_filtered * sizeof(bool), cudaMemcpyDeviceToHost);

        std::vector<float> h_boxes_in(n_in * 4);
        std::vector<float> h_scores_in(n_in);
        std::vector<int>   h_classes_in(n_in);
        cudaMemcpy(h_boxes_in.data(), boxes_ptr, n_in * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_scores_in.data(), scores_ptr, n_in * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_classes_in.data(), classes_ptr, n_in * sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<float> h_boxes_f(n_filtered * 4);
        std::vector<float> h_scores_f(n_filtered);
        std::vector<int>   h_classes_f(n_filtered);
        for (int i = 0; i < n_filtered; ++i) {
            int k = h_keep[i];
            for (int j = 0; j < 4; ++j) h_boxes_f[i * 4 + j] = h_boxes_in[k * 4 + j];
            h_scores_f[i] = h_scores_in[k];
            h_classes_f[i] = h_classes_in[k];
        }
        cudaMemcpyAsync(out_boxes, h_boxes_f.data(), n_filtered * 4 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out_scores, h_scores_f.data(), n_filtered * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out_classes, h_classes_f.data(), n_filtered * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out_suspect, h_suspect.data(), n_filtered * sizeof(bool), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    // Stage 2: NMS
    ensure_scratch(n_filtered, stream);
    fill_order_descending(out_scores, n_filtered, d_nms_order_, stream);
    cudaMemsetAsync(d_nms_count_, 0, sizeof(int), stream);

    int col_blocks = (n_filtered + 63) / 64;
    cudaMemsetAsync(d_nms_suppression_, 0, (size_t)n_filtered * col_blocks * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_nms_remv_, 0, col_blocks * sizeof(uint64_t), stream);

    nms_cuda(
        out_boxes, out_scores, out_classes, d_nms_order_,
        n_filtered, d_nms_keep_, d_nms_suppression_, d_nms_remv_,
        d_nms_count_, cfg_.nms_threshold, false, stream);

    int n_nms = 0;
    cudaMemcpy(&n_nms, d_nms_count_, sizeof(int), cudaMemcpyDeviceToHost);
    if (n_nms <= 0) return 0;
    if (n_nms == n_filtered) return n_nms;

    // Compact NMS result back into out_* in-place (host-side)
    {
        std::vector<int> h_nms_keep(n_nms);
        cudaMemcpy(h_nms_keep.data(), d_nms_keep_, n_nms * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<float> h_boxes(n_filtered * 4), h_scores(n_filtered);
        std::vector<int>   h_classes(n_filtered);
        std::vector<uint8_t>  h_suspect(n_filtered);
        cudaMemcpy(h_boxes.data(), out_boxes, n_filtered * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_scores.data(), out_scores, n_filtered * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_classes.data(), out_classes, n_filtered * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_suspect.data(), out_suspect, n_filtered * sizeof(bool), cudaMemcpyDeviceToHost);

        std::vector<float>   h_boxes_out(n_nms * 4), h_scores_out(n_nms);
        std::vector<int>     h_classes_out(n_nms);
        std::vector<uint8_t> h_suspect_out(n_nms);
        for (int i = 0; i < n_nms; ++i) {
            int k = h_nms_keep[i];
            for (int j = 0; j < 4; ++j) h_boxes_out[i * 4 + j] = h_boxes[k * 4 + j];
            h_scores_out[i] = h_scores[k];
            h_classes_out[i] = h_classes[k];
            h_suspect_out[i] = h_suspect[k];
        }
        cudaMemcpyAsync(out_boxes, h_boxes_out.data(), n_nms * 4 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out_scores, h_scores_out.data(), n_nms * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out_classes, h_classes_out.data(), n_nms * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out_suspect, h_suspect_out.data(), n_nms * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    return n_nms;
}

void PerceptionPipeline::extract_reid(
    const float* frame_ptr, int frame_h, int frame_w,
    const float* boxes_ptr, int n_boxes,
    float* out_embeds,
    cudaStream_t stream)
{
    if (!reid_ || !cropper_ || n_boxes <= 0) return;
    ensure_crop_buf(n_boxes);
    cropper_->process_gpu(
        const_cast<void*>(reinterpret_cast<const void*>(frame_ptr)),
        frame_w, frame_h,
        const_cast<float*>(boxes_ptr), n_boxes,
        d_crop_buf_, stream);
    reid_->extract(d_crop_buf_, n_boxes, out_embeds, stream);
}

int PerceptionPipeline::get_embed_dim() const {
    return reid_ ? reid_->get_feature_dim() : 0;
}

} // namespace saccade
