// Native end-to-end serial runtime (#465 Phase B PR-9, U3b-3):
// ingest -> detect -> post-detector -> MOT emit and tail, in one process.
//
// It adds no stage of its own: every stage is an already-measured host
// (ingest_host.hpp PR-7, detector_host.hpp PR-8, post_detector_host.hpp PR-5,
// sequence_output.hpp PR-6). What it owns is the wiring, and the wiring
// follows the oracle's serial configuration (`mot17.py` without
// `--double-buffer`, evaluator.py `run_eval` / pipeline.py `EvalPipeline`):
//
//   per run       the resolved config and the plans; one CUDA stream; one
//                 JpegDecoder (torchvision keeps one per process); one
//                 DetectorHost (the head is loaded once per run); one
//                 PerceptionPipeline (run_eval builds it once);
//   per sequence  the sequence input (seqinfo.ini + img1 listing); the
//                 detector's coordinate scales (set_whole_graph_img_dims);
//                 one IngestHost (the frame pool, sized from seqinfo.ini);
//                 one PostDetectorHost (tracker, GMC, graphed-update pre-roll);
//                 one SequenceOutput (per-sequence ids, emit, tail);
//   per frame     k = 1..frame_end: ingest(k) -> detect(frame buffer) -> the
//                 rows copied into the run's device detection buffers ->
//                 process(rows, the same frame buffer) -> add_frame(tracker
//                 rows). Each host synchronizes the stream before returning,
//                 so the frame buffer is not overwritten while a stage reads it.
//
// Serial and eager. The oracle's double-buffer schedule with its CUDA graphs
// is DoubleBufferRuntime (double_buffer_runtime.hpp, PR-10); this runtime stays
// the PR-9-measured serial reference (`saccade_track_measurement --schedule
// serial`). Track
// ids are per sequence (boundary §5 B3: no run-global ids in shipping). Reads
// no environment; every value comes from the plans.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "saccade_shipping/detector_host.hpp"
#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/ingest_host.hpp"
#include "saccade_shipping/ingest_plan.hpp"
#include "saccade_shipping/mot_output.hpp"
#include "saccade_shipping/post_detector_host.hpp"
#include "saccade_shipping/resolved_config.hpp"
#include "saccade_shipping/sequence_output.hpp"

namespace saccade {
class PerceptionPipeline;
}

namespace saccade::shipping {

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
// Developer measurement only (the parity harness's wiring negative controls):
// each breaks one ownership / lifetime rule above on purpose. Measurement
// variant only (SACCADE_SHIPPING_MEASUREMENT_HOOKS, #465 PR-C2).
enum class RuntimeMutation {
    None,
    SharedPostHost,    // the previous sequence's PostDetectorHost (tracker, GMC, pre-roll) is
                       // carried over when its geometry is the same; a sequence of another
                       // geometry gets a fresh host (GMC sized for one frame size must not read
                       // another's buffer)
    StaleImageDims,    // set_image_dims only for the first sequence
    GmcPreviousFrame,  // GMC reads the previous frame's buffer (the first frame's for frame 1)
};
const char* runtime_mutation_name(RuntimeMutation m);
RuntimeMutation parse_runtime_mutation(const std::string& name);
#endif

// Per-frame view for a developer observer (trace); valid during the call.
struct FrameTrace {
    int frame = 0;
    DecodePath decode_path = DecodePath::HardwareBatched;
    const DetectionRows* detections = nullptr;  // the detector boundary (PR-5 detector.bin)
    const FrameResult* result = nullptr;
};

class FrameObserver {
public:
    virtual ~FrameObserver() = default;
    virtual void on_frame(const FrameTrace& t) = 0;
};

// Graph captures / replays during one sequence (zero in the eager serial
// runtime), and the frame loop's wall time.
struct ScheduleStats {
    int detector_captures = 0, detector_warmup_runs = 0, detector_replays = 0;
    PostGraphStats post;
    double loop_seconds = 0.0;  // frames 1..frame_end, steady clock
};

struct SequenceRunStats {
    std::string name;  // the sequence directory's name
    int im_width = 0, im_height = 0, seq_length = 0;
    int frames = 0;           // frames ingested (frame_end)
    int tracker_updates = 0;  // frames with a non-empty detector output
    int skipped_empty = 0;
    int pre_roll_updates = 0;
    int hardware_decodes = 0, decoupled_decodes = 0;
    std::size_t track_ids = 0;
    std::size_t lines = 0;
    InterpolationStats interpolation;
    ScheduleStats schedule;
};

struct SequenceRunResult {
    SequenceRunStats stats;
    std::vector<std::string> lines;  // the sequence's MOT lines after the tail
};

class SerialRuntime {
public:
    // Plans everything (fail closed: ConfigError) and loads the detector
    // (detector_host.hpp's load checks) before any sequence runs. Model paths
    // in the detector plan are resolved against `model_root`.
    SerialRuntime(const ResolvedShippingConfig& cfg, const DetectorInputs& detector_inputs,
                  const std::string& model_root);
    // The same, from a detector plan already made from the lineage and the
    // attestation (Gate A, preflight.hpp): the files Gate A hashed are the
    // ones loaded, without reading the lineage again.
    SerialRuntime(const ResolvedShippingConfig& cfg, DetectorPlan detector_plan,
                  const std::string& model_root);
    ~SerialRuntime();
    SerialRuntime(const SerialRuntime&) = delete;
    SerialRuntime& operator=(const SerialRuntime&) = delete;

    // One sequence directory (seqinfo.ini + img1/), frames 1..frame_end
    // (frame_end = seqLength, or min(max_frames, seqLength) when max_frames >
    // 0, the oracle's --max-frames). Throws InputError on input the oracle
    // would not ingest as-is (ingest_plan.hpp, ingest_host.hpp).
    SequenceRunResult run_sequence(const std::filesystem::path& sequence_dir, int max_frames = 0,
                                   FrameObserver* observer = nullptr);

    const DetectorPlan& detector_plan() const { return detector_plan_; }
    const HeadLoadReport& load_report() const;
    const NvjpegLibraryInfo& nvjpeg() const;
    int sequences_run() const { return sequences_run_; }

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    void set_mutation_for_measurement(RuntimeMutation m) { mutation_ = m; }
#endif

private:
    struct Stream {
        cudaStream_t s = nullptr;
        Stream();
        ~Stream();
    };
    struct DeviceDetections;

    ResolvedShippingConfig cfg_;
    IngestPlan ingest_plan_;
    DetectorPlan detector_plan_;
    SequenceOutputPlan output_plan_;
    // Declaration order is destruction order reversed: the stream outlives
    // every host that enqueues on it.
    Stream stream_;
    std::unique_ptr<JpegDecoder> decoder_;
    std::unique_ptr<DetectorHost> detector_;
    std::unique_ptr<PerceptionPipeline> pipeline_;
    std::unique_ptr<DeviceDetections> det_;
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    std::unique_ptr<PostDetectorHost> shared_post_;  // RuntimeMutation::SharedPostHost only
    SequenceGeometry shared_geometry_{};
    RuntimeMutation mutation_ = RuntimeMutation::None;
#endif
    int sequences_run_ = 0;
};

}  // namespace saccade::shipping
