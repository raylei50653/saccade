// Native post-detector host (#465 Phase B PR-5, U3a).
// See saccade_shipping/post_detector_host.hpp.
#include "saccade_shipping/post_detector_host.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "saccade_shipping/native_build.hpp"
#include "tracking/copy_pad.cuh"
#include "tracking/gmc.hpp"
#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"

namespace saccade::shipping {
namespace {

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("post-detector host: ") + what + ": " +
                                 cudaGetErrorString(e));
    }
}

template <class T>
T* device_alloc(std::size_t count, const char* what) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, count * sizeof(T)), what);
    cuda_check(cudaMemset(p, 0, count * sizeof(T)), what);
    return static_cast<T*>(p);
}

// One captured stream segment (cudaStreamBeginCapture, thread-local mode).
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
    }
    bool captured() const { return exec != nullptr; }
    template <class F>
    void capture(cudaStream_t stream, const char* what, const F& body) {
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), what);
        try {
            body();
        } catch (...) {
            cudaGraph_t partial = nullptr;
            if (cudaStreamEndCapture(stream, &partial) == cudaSuccess && partial != nullptr) {
                cudaGraphDestroy(partial);
            }
            throw;
        }
        cuda_check(cudaStreamEndCapture(stream, &graph), what);
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), what);
    }
    void launch(cudaStream_t stream, const char* what) const {
        cuda_check(cudaGraphLaunch(exec, stream), what);
    }
};

template <class T>
void d2h(std::vector<T>& dst, const T* src, std::size_t count, cudaStream_t stream) {
    dst.resize(count);
    if (count > 0) {
        cuda_check(cudaMemcpyAsync(dst.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
              "D2H");
    }
}

}  // namespace

// Device buffers, sized and laid out as the oracle's (pipeline.py: main NMS
// input and post buffers of nms_fixed_n rows, the tracker wrapper's prior
// buffers of max_objects rows, GraphedTrackerUpdate's max_assoc inputs and
// max_objects outputs, the shared GMC warp). Zero-initialized.
struct PostDetectorHost::DeviceBuffers {
    std::vector<void*> owned;
    template <class T>
    T* alloc(std::size_t count, const char* what) {
        T* p = device_alloc<T>(count, what);
        owned.push_back(p);
        return p;
    }
    ~DeviceBuffers() {
        for (void* p : owned) cudaFree(p);
    }

    float *nms_in_boxes, *nms_in_scores;
    std::int32_t* nms_in_classes;
    float *post_boxes, *post_scores;
    std::int32_t* post_classes;
    bool* post_suspect;
    std::int32_t* post_count;
    float* prior_boxes;
    std::int32_t* prior_classes;
    float* gmc_warp;  // GMC output (`_shared_gmc_warp`)
    float *trk_boxes, *trk_scores;
    std::int32_t* trk_classes;
    float* trk_gmc;  // GraphedTrackerUpdate.d_gmc: identity until a warp is copied in
    float *out_boxes, *out_scores;
    std::int32_t *out_ids, *out_classes, *out_det_idx, *out_count;
    float* gmc_frame;  // GraphMode::Captured: the GMC graph's input (`_gmc_frame_buf`)
};

struct PostDetectorHost::Graphs {
    Graph nms, gmc, tracker;
};

PostDetectorHost::PostDetectorHost(const ResolvedShippingConfig& cfg, SequenceGeometry geometry,
                                   PerceptionPipeline& pipeline, cudaStream_t stream,
                                   GraphMode graphs)
    : plan_(plan_post_detector(cfg)),
      geometry_(geometry),
      pipeline_(pipeline),
      stream_(stream),
      tracker_(build_tracker(cfg, geometry)),
      gmc_(plan_.gmc ? build_gmc(cfg) : nullptr),
      buf_(std::make_unique<DeviceBuffers>()),
      graphs_(graphs),
      g_(std::make_unique<Graphs>()) {
    const auto n = static_cast<std::size_t>(plan_.nms_fixed_n);
    const auto a = static_cast<std::size_t>(plan_.max_assoc);
    const auto o = static_cast<std::size_t>(plan_.max_objects);
    auto& b = *buf_;
    b.nms_in_boxes = b.alloc<float>(n * 4, "nms input");
    b.nms_in_scores = b.alloc<float>(n, "nms input");
    b.nms_in_classes = b.alloc<std::int32_t>(n, "nms input");
    b.post_boxes = b.alloc<float>(n * 4, "post buffers");
    b.post_scores = b.alloc<float>(n, "post buffers");
    b.post_classes = b.alloc<std::int32_t>(n, "post buffers");
    b.post_suspect = b.alloc<bool>(n, "post buffers");
    b.post_count = b.alloc<std::int32_t>(1, "post count");
    b.prior_boxes = b.alloc<float>(o * 4, "prior buffers");
    b.prior_classes = b.alloc<std::int32_t>(o, "prior buffers");
    b.gmc_warp = b.alloc<float>(6, "gmc warp");
    b.trk_boxes = b.alloc<float>(a * 4, "tracker input");
    b.trk_scores = b.alloc<float>(a, "tracker input");
    b.trk_classes = b.alloc<std::int32_t>(a, "tracker input");
    b.trk_gmc = b.alloc<float>(6, "tracker gmc");
    b.out_boxes = b.alloc<float>(o * 4, "tracker output");
    b.out_scores = b.alloc<float>(o, "tracker output");
    b.out_ids = b.alloc<std::int32_t>(o, "tracker output");
    b.out_classes = b.alloc<std::int32_t>(o, "tracker output");
    b.out_det_idx = b.alloc<std::int32_t>(o, "tracker output");
    b.out_count = b.alloc<std::int32_t>(1, "tracker output");
    // torch.zeros(3, h, w) when the direct GMC is in use.
    b.gmc_frame = graphs_ == GraphMode::Captured && gmc_
                      ? b.alloc<float>(3 * static_cast<std::size_t>(geometry_.im_width) *
                                           static_cast<std::size_t>(geometry_.im_height),
                                       "gmc frame")
                      : nullptr;
    const float identity[6] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};  // torch.eye(2, 3)
    cuda_check(cudaMemcpy(b.trk_gmc, identity, sizeof(identity), cudaMemcpyHostToDevice),
          "tracker gmc");
}

PostDetectorHost::~PostDetectorHost() {
    // Replays may still be queued on the stream when the graphs and buffers go.
    cudaStreamSynchronize(stream_);
}

void PostDetectorHost::set_pre_roll_for_measurement(int updates) {
    if (pre_rolled_ || updates < 0) {
        throw std::logic_error("set_pre_roll_for_measurement: before the first update, >= 0");
    }
    plan_.tracker_pre_roll = updates;
}

void PostDetectorHost::tracker_update(int num_dets) {
    auto& b = *buf_;
    tracker_->update_into(b.trk_boxes, b.trk_scores, b.trk_classes, num_dets, stream_,
                          b.out_boxes, b.out_scores, b.out_ids, b.out_classes, b.out_det_idx,
                          b.out_count, /*embeddings_ptr=*/nullptr, b.trk_gmc,
                          /*light_factor=*/0.0f, /*mid_thresh_scale=*/1.0f, plan_.max_objects);
}

void PostDetectorHost::run_pre_roll() {
    // GraphedTrackerUpdate warm-up: its scratch inputs are still zero and its
    // d_gmc identity, because capture happens before the first copy_inputs.
    for (int i = 0; i < plan_.tracker_pre_roll; ++i) {
        tracker_update(plan_.max_assoc);
        ++pre_roll_run_;
    }
    cuda_check(cudaStreamSynchronize(stream_), "pre-roll");
    if (graphs_ == GraphMode::Captured) {
        g_->tracker.capture(stream_, "tracker capture", [&] { tracker_update(plan_.max_assoc); });
        ++graph_stats_.tracker_captures;
    }
    pre_rolled_ = true;
}

void PostDetectorHost::main_nms(int nf, int w, int h, bool is_tiled) {
    auto& b = *buf_;
    auto call = [&] {
        pipeline_.process_detections_main_nms_graph_nocopyback(
            b.nms_in_boxes, b.nms_in_scores, b.nms_in_classes, nf, w, h, is_tiled, b.post_boxes,
            b.post_scores, b.post_classes, b.post_suspect, b.post_count, nullptr, nullptr, 0, 0.0f,
            stream_);
    };
    if (graphs_ == GraphMode::Eager) {
        call();
        return;
    }
    if (!g_->nms.captured()) {
        call();  // warm-up, eager, on the real (padded) input
        cuda_check(cudaStreamSynchronize(stream_), "main NMS warm-up");
        g_->nms.capture(stream_, "main NMS capture", call);
        ++graph_stats_.nms_captures;
    }
    g_->nms.launch(stream_, "main NMS replay");
    ++graph_stats_.nms_replays;
}

void PostDetectorHost::gmc(const float* frame_chw, int w, int h) {
    auto& b = *buf_;
    if (graphs_ == GraphMode::Eager) {
        gmc_->estimate_into_direct(frame_chw, w, h, stream_, b.gmc_warp);
        return;
    }
    const std::size_t bytes = 3 * static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * sizeof(float);
    if (!g_->gmc.captured()) {
        cuda_check(cudaMemcpyAsync(b.gmc_frame, frame_chw, bytes, cudaMemcpyDeviceToDevice, stream_),
                   "gmc frame");
        gmc_->estimate_into_direct(b.gmc_frame, w, h, stream_, b.gmc_warp);
        cuda_check(cudaStreamSynchronize(stream_), "gmc warm-up");
        g_->gmc.capture(stream_, "gmc capture",
                        [&] { gmc_->estimate_into_direct(b.gmc_frame, w, h, stream_, b.gmc_warp); });
        ++graph_stats_.gmc_captures;
        return;  // the capture branch does not replay
    }
    if (!stale_gmc_input_) {
        cuda_check(cudaMemcpyAsync(b.gmc_frame, frame_chw, bytes, cudaMemcpyDeviceToDevice, stream_),
                   "gmc frame");
    }
    g_->gmc.launch(stream_, "gmc replay");
    ++graph_stats_.gmc_replays;
}

FrameResult PostDetectorHost::process(const DeviceDetections& det, const float* frame_chw) {
    FrameResult r;
    if (det.n <= 0) return r;  // evaluator: `if fused_boxes.numel() == 0: ... return True`
    auto& b = *buf_;
    const int w = geometry_.im_width, h = geometry_.im_height;

    // _run_native_tensor_prep: private priors from the tracker's current state.
    const float* private_priors = nullptr;
    int n_private = 0;
    if (plan_.private_priors) {
        const int count = tracker_->build_track_priors_gpu(
            b.prior_boxes, b.prior_classes, /*min_track_age=*/0, plan_.private_prior_max_age,
            /*min_track_score=*/0.0f, stream_);
        if (count > 0) {
            private_priors = b.prior_boxes;
            n_private = count;
        }
    }

    // _run_nms, graphed main NMS path (SACCADE_MAIN_NMS_GRAPHED; ONMS off).
    const int nf = plan_.nms_fixed_n;
    copy_pad_detections(det.boxes, det.scores, det.classes, std::min(det.n, nf), b.nms_in_boxes,
                        b.nms_in_scores, b.nms_in_classes, nf, stream_);
    main_nms(nf, w, h, det.is_tiled);
    const int n_post = pipeline_.process_detections_split_pipeline_graphed(
        b.post_boxes, b.post_scores, b.post_classes, b.post_suspect, b.post_count, nf,
        private_priors, n_private, stream_);
    if (n_post < 0 || n_post > nf) {
        throw std::runtime_error("post-detector host: NMS count out of range");
    }

    // _run_post_nms_finalize slices the post buffers; the filters run on host.
    d2h(r.post_nms.boxes, b.post_boxes, static_cast<std::size_t>(n_post) * 4, stream_);
    d2h(r.post_nms.scores, b.post_scores, static_cast<std::size_t>(n_post), stream_);
    d2h(r.post_nms.classes, b.post_classes, static_cast<std::size_t>(n_post), stream_);
    cuda_check(cudaStreamSynchronize(stream_), "post-NMS readback");

    // _run_detection_filters: external FP rule filter, then FP hard filter.
    r.tracker_input = plan_.external_fp ? apply_external_fp_rule(r.post_nms, plan_.external_fp_rule)
                                        : r.post_nms;
    if (plan_.fp_hard) apply_fp_hard_filter(r.tracker_input, plan_.fp_hard_filter);

    // _run_reid_and_gmc (ReID off): GMC on the frame.
    if (gmc_) gmc(frame_chw, w, h);

    // _run_track via GraphedTrackerUpdate: pre-roll at the sequence's first
    // update, then copy_inputs (rows, zero tail, warp) and the update.
    if (!pre_rolled_) run_pre_roll();
    const int n_in = static_cast<int>(std::min<std::size_t>(r.tracker_input.size(),
                                                            static_cast<std::size_t>(plan_.max_assoc)));
    const auto a = static_cast<std::size_t>(plan_.max_assoc);
    cuda_check(cudaMemsetAsync(b.trk_boxes, 0, a * 4 * sizeof(float), stream_), "tracker input");
    cuda_check(cudaMemsetAsync(b.trk_scores, 0, a * sizeof(float), stream_), "tracker input");
    cuda_check(cudaMemsetAsync(b.trk_classes, 0, a * sizeof(std::int32_t), stream_), "tracker input");
    if (n_in > 0) {
        cuda_check(cudaMemcpyAsync(b.trk_boxes, r.tracker_input.boxes.data(),
                              static_cast<std::size_t>(n_in) * 4 * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "tracker input");
        cuda_check(cudaMemcpyAsync(b.trk_scores, r.tracker_input.scores.data(),
                              static_cast<std::size_t>(n_in) * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "tracker input");
        cuda_check(cudaMemcpyAsync(b.trk_classes, r.tracker_input.classes.data(),
                              static_cast<std::size_t>(n_in) * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice, stream_),
              "tracker input");
    }
    if (gmc_) {
        cuda_check(cudaMemcpyAsync(b.trk_gmc, b.gmc_warp, 6 * sizeof(float), cudaMemcpyDeviceToDevice,
                              stream_),
              "tracker gmc");
    }
    if (graphs_ == GraphMode::Captured) {
        g_->tracker.launch(stream_, "tracker replay");
        ++graph_stats_.tracker_replays;
    } else {
        tracker_update(plan_.max_assoc);
    }

    std::int32_t count = 0;
    cuda_check(cudaMemcpyAsync(&count, b.out_count, sizeof(count), cudaMemcpyDeviceToHost, stream_),
          "tracker count");
    cuda_check(cudaMemcpyAsync(r.gmc_warp.data(), b.trk_gmc, 6 * sizeof(float), cudaMemcpyDeviceToHost,
                          stream_),
          "tracker gmc");
    cuda_check(cudaStreamSynchronize(stream_), "tracker update");
    if (count < 0 || count > plan_.max_objects) {
        throw std::runtime_error("post-detector host: tracker count out of range");
    }
    const auto c = static_cast<std::size_t>(count);
    d2h(r.tracker_output.boxes, b.out_boxes, c * 4, stream_);
    d2h(r.tracker_output.scores, b.out_scores, c, stream_);
    d2h(r.tracker_output.ids, b.out_ids, c, stream_);
    d2h(r.tracker_output.classes, b.out_classes, c, stream_);
    cuda_check(cudaStreamSynchronize(stream_), "tracker readback");
    r.updated = true;
    return r;
}

}  // namespace saccade::shipping
