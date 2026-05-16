#include "tracking/workbench.hpp"
#include <stdexcept>

namespace saccade {

Workbench::Workbench(PerceptionPipeline* pipeline,
                     GPUByteTracker*    tracker,
                     cudaStream_t       stream,
                     int max_dets,
                     int max_tracks)
    : pipeline_(pipeline),
      tracker_(tracker),
      stream_(stream),
      max_dets_(max_dets),
      max_tracks_(max_tracks)
{
    if (pipeline == nullptr || tracker == nullptr) {
        throw std::invalid_argument(
            "Workbench: pipeline and tracker pointers must be non-null");
    }
    cudaMalloc(&d_post_boxes_,   sizeof(float) * max_dets * 4);
    cudaMalloc(&d_post_scores_,  sizeof(float) * max_dets);
    cudaMalloc(&d_post_classes_, sizeof(int)   * max_dets);
    cudaMalloc(&d_post_suspect_, sizeof(bool)  * max_dets);
    cudaMalloc(&d_post_count_,   sizeof(int));
}

Workbench::~Workbench() {
    cudaFree(d_post_boxes_);
    cudaFree(d_post_scores_);
    cudaFree(d_post_classes_);
    cudaFree(d_post_suspect_);
    cudaFree(d_post_count_);
}

int Workbench::process_frame_postyolo(
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
    const float* embeddings_ptr,
    const float* gmc_ptr,
    float        light_factor,
    float        mid_thresh_scale,
    float* out_boxes,
    float* out_scores,
    int*   out_ids,
    int*   out_classes,
    int*   out_det_idx,
    int*   out_count)
{
    // 1. Filter + NMS + optional geometry/ONMS, writing into per-workbench
    //    post-NMS scratch on stream_.
    pipeline_->process_detections_into(
        raw_boxes, raw_scores, raw_classes, n_in,
        frame_w, frame_h, is_tiled,
        d_post_boxes_, d_post_scores_, d_post_classes_, d_post_suspect_,
        d_post_count_,
        priors_ptr, prior_classes_ptr, num_priors, prior_iou_threshold,
        stream_);

    // 2. Pull post-NMS count to host — tracker.update_into needs num_dets.
    int n_post = 0;
    cudaMemcpyAsync(&n_post, d_post_count_, sizeof(int),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    

    if (n_post < 0 || n_post > max_dets_) {
        // This should never happen if pipeline.cpp is correct
        return 0; 
    }

    // 2.5 Optional: apply geometry suspect penalty
    const auto& cfg = pipeline_->get_config();
    if (cfg.geometry_suspect_support && n_post > 0) {
        penalize_suspect_scores_cuda(
            d_post_scores_,
            d_post_suspect_,
            d_post_count_,
            cfg.geometry_suspect_support_score,
            max_dets_,
            stream_);
    }

    // 3. Tracker update_into → caller-provided output buffers, still on stream_.
    //    update_into's signature is non-const for legacy reasons; it does not
    //    write through embeddings/gmc.
    try {
        tracker_->update_into(
            d_post_boxes_, d_post_scores_, d_post_classes_, n_post,
            stream_,
            out_boxes, out_scores, out_ids, out_classes,
            out_det_idx, out_count,
            const_cast<float*>(embeddings_ptr),
            const_cast<float*>(gmc_ptr),
            light_factor, mid_thresh_scale);
    } catch (const std::exception& e) {
        return 0;
    }

    // 4. Pull final track count and sync once so caller sees finished frame.
    int n_tracks = 0;
    cudaMemcpyAsync(&n_tracks, out_count, sizeof(int),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return n_tracks;
}

} // namespace saccade
