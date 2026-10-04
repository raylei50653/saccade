// Native post-detector host (#465 Phase B PR-5, U3a): serial, eager.
//
// One host per sequence, as the oracle builds a GMC estimator, a tracker and a
// GraphedTrackerUpdate per sequence (pipeline.py EvalPipeline); the
// PerceptionPipeline is shared across sequences, as run_eval builds it once.
// `process` runs one frame's stages in the oracle's order (evaluator.py
// `_run_frame`, serial configuration) with the values of
// `plan_post_detector`; it reads no environment and no value outside the
// resolved config.
//
// Per frame, given the detector output on device:
//   empty detector output -> nothing runs (the oracle returns before NMS, GMC
//     and the tracker update; track ages and GMC's previous frame stay put);
//   private priors from the tracker -> copy_pad to nms_fixed_n -> main NMS
//     (no copyback) -> private-continuation append + count sync;
//   host-side external FP rule filter -> FP hard filter;
//   GMC on the frame (warp stays identity when GMC is off);
//   tracker update on the rows zero-padded to max_assoc, as the graphed update
//     replays it (num_dets = max_assoc, no embeddings, light_factor 0,
//     mid_thresh_scale 1), after kGraphedTrackerUpdatePreRoll empty updates
//     before the sequence's first real update.
// GraphMode::Eager (U3a, PR-5) runs what the oracle captures in CUDA graphs
// as the same calls. GraphMode::Captured (U5, PR-10) captures them as the
// oracle does, once per host (= per sequence), with cudaStreamBeginCapture in
// thread-local mode:
//   main NMS     at the first update: copy_pad, one eager nocopyback call, a
//                sync, the capture, then the replay (stages.py
//                `_capture_main_nms_graph_nocopyback`); later frames copy_pad
//                and replay; the private-continuation append stays eager;
//   GMC          at the first update: the frame copied into the host's GMC
//                frame buffer, one eager estimate on it, a sync, the capture
//                and no replay (stages.py `_run_reid_and_gmc`); later frames
//                copy the frame in and replay;
//   tracker      the pre-roll as above, then the capture of update_into on
//                the zero scratch inputs and identity warp (copy_inputs
//                captures before it writes); every update then copies its
//                inputs and replays.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "saccade_shipping/native_config.hpp"
#include "saccade_shipping/post_detector_plan.hpp"

namespace saccade {
class GPUByteTracker;
class GMC;
class PerceptionPipeline;
}  // namespace saccade

namespace saccade::shipping {

// The detector output of one frame, on device, as `_run_native_tensor_prep`
// hands it to native NMS: float32 xyxy boxes [n*4], float32 scores [n],
// int32 classes [n].
struct DeviceDetections {
    const float* boxes = nullptr;
    const float* scores = nullptr;
    const std::int32_t* classes = nullptr;
    int n = 0;
    bool is_tiled = false;
};

// The tracker's output rows for one frame (what the oracle materializes).
struct TrackerRows {
    std::vector<float> boxes;  // [count*4]
    std::vector<float> scores;
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> classes;
    std::size_t size() const { return ids.size(); }
};

enum class GraphMode { Eager, Captured };

// Graph captures and launches so far (the report and the tests).
struct PostGraphStats {
    int nms_captures = 0, nms_replays = 0;
    int gmc_captures = 0, gmc_replays = 0;
    int tracker_captures = 0, tracker_replays = 0;
};

struct FrameResult {
    bool updated = false;  // false: empty detector output, nothing ran
    DetectionRows post_nms;      // main NMS + private candidates
    DetectionRows tracker_input; // after the detection filters
    std::array<float, 6> gmc_warp{};
    TrackerRows tracker_output;
};

class PostDetectorHost {
public:
    // Builds the sequence's tracker and GMC from the resolved config
    // (native_build.hpp, readback-checked). `pipeline` must outlive the host.
    PostDetectorHost(const ResolvedShippingConfig& cfg, SequenceGeometry geometry,
                     PerceptionPipeline& pipeline, cudaStream_t stream,
                     GraphMode graphs = GraphMode::Eager);
    ~PostDetectorHost();
    PostDetectorHost(const PostDetectorHost&) = delete;
    PostDetectorHost& operator=(const PostDetectorHost&) = delete;

    // `frame_chw` is the GMC input on device: float32 [3, height, width] in
    // [0, 1] (the oracle's pool.frame_buffer). Synchronizes `stream`.
    FrameResult process(const DeviceDetections& detections, const float* frame_chw);

    const PostDetectorPlan& plan() const { return plan_; }
    // Empty tracker updates run so far (the pre-roll); for tests.
    int pre_roll_updates_run() const { return pre_roll_run_; }
    // Override the pre-roll count (developer measurement only; the shipping
    // value is the plan's). Must be called before the first process().
    void set_pre_roll_for_measurement(int updates);

    GraphMode graph_mode() const { return graphs_; }
    const PostGraphStats& graph_stats() const { return graph_stats_; }
    // GraphMode::Captured only (developer measurement: a negative control):
    // GMC replays without copying the frame into its captured input buffer.
    void set_stale_gmc_input_for_measurement(bool on) { stale_gmc_input_ = on; }

private:
    struct DeviceBuffers;
    struct Graphs;

    void run_pre_roll();
    void tracker_update(int num_dets);
    void main_nms(int nf, int w, int h, bool is_tiled);
    void gmc(const float* frame_chw, int w, int h);

    PostDetectorPlan plan_;
    SequenceGeometry geometry_;
    PerceptionPipeline& pipeline_;
    cudaStream_t stream_;
    std::unique_ptr<GPUByteTracker> tracker_;
    std::unique_ptr<GMC> gmc_;
    std::unique_ptr<DeviceBuffers> buf_;
    GraphMode graphs_;
    std::unique_ptr<Graphs> g_;
    PostGraphStats graph_stats_;
    bool stale_gmc_input_ = false;
    bool pre_rolled_ = false;
    int pre_roll_run_ = 0;
};

}  // namespace saccade::shipping
