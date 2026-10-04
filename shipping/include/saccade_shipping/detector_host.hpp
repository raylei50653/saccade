// Native detector host (#465 Phase B PR-8, U3b-2): resize -> TRT backbone ->
// LibTorch head (PR-1L artifact) -> S2, serial and eager. See
// detector_plan.hpp for the oracle path it reproduces.
//
// Loading is fail-closed, in this order, before anything is executed:
//   1. sha256 of the operator library, the TorchScript artifact and the
//      backbone engine == the plan's bindings (frozen lineage, or the
//      realization attestation for the operator library);
//   2. the operator library is dlopen'ed (it registers
//      `saccade_native::selective_scan_fwd` with the dispatcher);
//   3. the artifact's runtime requirements are set and read back (graph
//      executor optimize off, cuDNN benchmark off, cuDNN TF32 on, matmul TF32
//      off), then `torch::jit::load(path)` without a device map: parameters
//      must land on cuda:0 and every tensor constant must stay on the CPU (a
//      CUDA constant in the traced shape arithmetic turns aten::Int into a
//      device sync; PR-2L amendment A1);
//   4. the inlined forward graph has no prim::PythonOp, never calls the Python
//      op `saccade::selective_scan_fwd`, and calls the native op exactly the
//      lineage's number of times;
//   5. the backbone engine's I/O is one input [1, 3, img, img] and three
//      outputs of the planned feature shapes, in that order;
//   6. no libpython / libtorch_python is mapped into the process.
//
// Per frame (`detect`): ATen's upsample_bilinear2d (align_corners false; the
// kernel F.interpolate dispatches to) into [1, 3, img, img]; the engine on the
// host's stream; the head's forward; S2 (detector_s2.hpp, ATen topk); a
// device -> host copy of the rows, then a stream synchronize. Every value
// comes from the plan; nothing reads the environment.
//
// Whole-detect CUDA graph (`detect_graphed`, PR-10 / U5): the oracle's
// `MambaGatedDetector._forward_whole_graph` under the double-buffer schedule.
// The same resize -> engine -> head -> S2 is captured once per key (frame
// shape, image dims) with LibTorch's graph capture (private memory pool,
// thread-local capture mode, as cuda_capture.graphed_callables) and replayed
// on the host's stream; the frame is copied into the graph's static input
// first (make_graphed_callables). On a key miss: one warm-up run if the host
// is not warm (`_whole_graph_warmup`), the three warm-up iterations of
// make_graphed_callables, then the capture; a cache that already holds ten
// graphs is cleared first. `set_image_dims` with new dims clears the cache and
// the warm flag (`set_whole_graph_img_dims`); the same dims keep both. After
// the replay the rows are copied out of the graph's static output into the
// caller's buffers on the same stream -- the oracle's output clones, plus
// `_run_native_tensor_prep`'s float -> int32 class cast.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/post_detector_plan.hpp"

namespace saccade::shipping {

// What the loader observed (recorded by the probe).
struct HeadLoadReport {
    std::string op_library_sha256, head_artifact_sha256, backbone_engine_sha256;
    std::vector<std::string> param_devices;     // distinct, sorted
    std::vector<std::string> constant_devices;  // one per tensor constant
    int native_scan_calls = 0;
    HeadRuntimeRequirements runtime_readback;
    std::vector<std::string> engine_io;  // "name:[dims]" in engine order
    int trt_version = 0;                 // getInferLibVersion()
    std::string torch_version;
};

// Developer measurement only (the parity harness's negative controls): each
// mutates one stage on purpose; never set by a shipping caller.
enum class DetectorMutation {
    None,
    BackboneUlp,  // p3 element 0 moved by one ulp after the engine
    HeadUlp,      // cls_p3 element 0 moved by one ulp after the head
    S2Threshold,  // rows with score < 0.05 dropped (a threshold the fixed S2 does not have)
    S2TopK,       // top-(max_det - 1) instead of top-max_det
    S2Order,      // rows 0 and 1 swapped
    BoxUlp,       // x1 of row 0 moved by one ulp after the coordinate scaling
};
const char* detector_mutation_name(DetectorMutation m);
DetectorMutation parse_detector_mutation(const std::string& name);

// Device destination of one frame's rows: boxes [n*4], scores [n], int32
// classes [n], capacity >= max_det.
struct DeviceRowsOut {
    float* boxes = nullptr;
    float* scores = nullptr;
    std::int32_t* classes = nullptr;
    int capacity = 0;
};

// What the whole-detect graph did so far (the report and the tests).
struct WholeGraphStats {
    int captures = 0;
    int warmup_runs = 0;    // eager runs before captures (1 when not warm, + 3)
    int replays = 0;
    int cache_clears = 0;   // by set_image_dims with new dims, or a full cache
};

// Device views of one frame's stages, valid until the next `detect`.
struct DetectorStages {
    const float* resized = nullptr;   // [1, 3, img, img]
    const float* features[3] = {};    // p3, p4, p5 (plan.feature_shapes)
    const float* head[6] = {};        // cls_p3..p5 [1, nc, s, s], reg_p3..p5 [1, 4, s, s]
    const float* s2_raw = nullptr;    // [rows, 6] in img_size space
    const float* s2_scaled = nullptr; // [rows, 6] after the coordinate scaling
    int rows = 0;
};

class DetectorHost {
public:
    // Paths in the plan are resolved against `model_root`.
    DetectorHost(const DetectorPlan& plan, const std::string& model_root, cudaStream_t stream);
    ~DetectorHost();
    DetectorHost(const DetectorHost&) = delete;
    DetectorHost& operator=(const DetectorHost&) = delete;

    const HeadLoadReport& load_report() const;

    // set_whole_graph_img_dims(h, w): the per-sequence coordinate scales
    // (seqinfo.ini's imHeight / imWidth), and nothing else.
    void set_image_dims(int height, int width);

    // One frame: `frame_chw` is device float32 [3, height, width] (the ingest
    // frame buffer; as in the oracle, the frame carries its own size, separate
    // from the scales above); returns the detector rows (the PR-5 boundary)
    // after the stream has been synchronized.
    DetectionRows detect(const float* frame_chw, int height, int width);

    const DetectorStages& stages() const;

    // One frame through the whole-detect graph (see above), enqueued on the
    // host's stream; does not synchronize. `frame_chw` must stay unchanged
    // until the stream has passed this call. Returns the row count (max_det).
    // Mutations are eager-only: refused here. `refresh_input` false skips the
    // static-input copy (developer measurement only: a negative control).
    int detect_graphed(const float* frame_chw, int height, int width, const DeviceRowsOut& out,
                       bool refresh_input = true);
    const WholeGraphStats& graph_stats() const;

    void set_mutation_for_measurement(DetectorMutation m);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Libraries mapped into this process whose file name starts with libpython or
// libtorch_python (from /proc/self/maps).
std::vector<std::string> mapped_python_libraries();

}  // namespace saccade::shipping
