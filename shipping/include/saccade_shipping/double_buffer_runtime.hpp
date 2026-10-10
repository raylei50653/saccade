// Native end-to-end runtime, double-buffer schedule with CUDA graphs (#465
// Phase B PR-10, U5): the oracle's own schedule (`mot17.py --double-buffer`,
// evaluator.py `run_eval` frame loop, stages.py `_launch_double_buffer_detect`).
//
// Same hosts and the same per-run / per-sequence ownership as SerialRuntime
// (serial_runtime.hpp, PR-9). What changes is when work runs and on which
// stream, and that the four graphs the oracle captures are captured
// (schedule_plan.hpp, post_detector_host.hpp, detector_host.hpp):
//
//   streams       the main stream (post-detector host: NMS, filters, GMC,
//                 tracker) and the detect stream (the oracle's
//                 double_buffer_stream: normalize + whole-detect graph); per run
//   pools         two frame pools per sequence (IngestHost with 2 pools) and
//                 two device detection buffers per run, selected by frame
//                 parity (k - 1) % 2, as `double_buffer_pools` and the output
//                 clones of `PreparedDetection`
//   launch(k)     decode k into its pool's decode buffer (host thread; the
//                 decode has completed on return), record input_ready on the
//                 main stream, make the detect stream wait on it, then on the
//                 detect stream: normalize into the pool's frame buffer, the
//                 whole-detect graph, the rows out of the graph's static output
//                 into the parity's detection buffer, record ready
//   frame loop    launch(1); for k = 1..frame_end: launch(k + 1) when
//                 k < frame_end, then the main stream waits on ready(k) and runs
//                 the post-detector host on the parity's detections and frame
//                 buffer, and the MOT emit. One detection in flight: detect
//                 (k + 1) overlaps tracker(k).
//
// Why each buffer is free when it is written (the event barrier): launch(k+1)
// reuses frame k-1's frame buffer and detection buffer, whose last readers
// (GMC's frame copy, copy_pad) were enqueued on the main stream before
// input_ready(k+1) was recorded there; the detect stream waits on it. Frame
// k+1's decode buffer was last read by normalize(k-1); the host waits on that
// normalize's event before decoding into it (the oracle's allocator and
// record_stream play this part).
//
// Reads no environment; every value comes from the plans. Track ids are per
// sequence (boundary §5 B3).
#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <string>

#include "saccade_shipping/schedule_plan.hpp"
#include "saccade_shipping/serial_runtime.hpp"

namespace saccade::shipping {

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
// Developer measurement only (the parity harness's negative controls): each
// breaks one graph-ownership or parity rule above on purpose. Measurement
// variant only (SACCADE_SHIPPING_MEASUREMENT_HOOKS, #465 PR-C2).
enum class DoubleBufferMutation {
    None,
    StaleDetectorInput,      // the whole-detect graph replays without the frame copied in
    StaleGmcInput,           // the GMC graph replays without the frame copied in
    SwappedDetectionParity,  // tracker(k) reads the other parity's detection buffer (frame
                             // k+1's after waiting on it; frame k-1's on the last frame)
};
const char* double_buffer_mutation_name(DoubleBufferMutation m);
DoubleBufferMutation parse_double_buffer_mutation(const std::string& name);
#endif

class DoubleBufferRuntime {
public:
    // Plans everything (fail closed: ConfigError, including a config whose
    // schedule is not the double buffer) and loads the detector before any
    // sequence runs. Model paths are resolved against `model_root`.
    DoubleBufferRuntime(const ResolvedShippingConfig& cfg, const DetectorInputs& detector_inputs,
                        const std::string& model_root);
    // The same, from a detector plan already made from the lineage and the
    // attestation (Gate A, preflight.hpp): the files Gate A hashed are the
    // ones loaded, without reading the lineage again.
    DoubleBufferRuntime(const ResolvedShippingConfig& cfg, DetectorPlan detector_plan,
                        const std::string& model_root);
    ~DoubleBufferRuntime();
    DoubleBufferRuntime(const DoubleBufferRuntime&) = delete;
    DoubleBufferRuntime& operator=(const DoubleBufferRuntime&) = delete;

    // As SerialRuntime::run_sequence. An observer gets each frame's detector
    // rows read back from the parity's detection buffer (an extra main-stream
    // sync per frame; saccade_track --trace).
    SequenceRunResult run_sequence(const std::filesystem::path& sequence_dir, int max_frames = 0,
                                   FrameObserver* observer = nullptr);

    const DetectorPlan& detector_plan() const { return detector_plan_; }
    const SchedulePlan& schedule_plan() const { return schedule_plan_; }
    const HeadLoadReport& load_report() const;
    const NvjpegLibraryInfo& nvjpeg() const;
    const WholeGraphStats& detector_graph_stats() const;
    int sequences_run() const { return sequences_run_; }

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    void set_mutation_for_measurement(DoubleBufferMutation m) { mutation_ = m; }
#endif

private:
    struct Stream {
        cudaStream_t s = nullptr;
        Stream();
        ~Stream();
    };
    struct Events;
    struct DetectionBuffers;

    ResolvedShippingConfig cfg_;
    IngestPlan ingest_plan_;
    DetectorPlan detector_plan_;
    SequenceOutputPlan output_plan_;
    SchedulePlan schedule_plan_;
    // Declaration order is destruction order reversed: the streams outlive
    // every host that enqueues on them.
    Stream main_, detect_;
    std::unique_ptr<Events> events_;
    std::unique_ptr<JpegDecoder> decoder_;
    std::unique_ptr<DetectorHost> detector_;
    std::unique_ptr<PerceptionPipeline> pipeline_;
    std::unique_ptr<DetectionBuffers> det_[2];
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    DoubleBufferMutation mutation_ = DoubleBufferMutation::None;
#endif
    int sequences_run_ = 0;
};

}  // namespace saccade::shipping
