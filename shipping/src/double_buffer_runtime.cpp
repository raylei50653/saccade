// Native end-to-end runtime, double-buffer schedule (#465 Phase B PR-10);
// see double_buffer_runtime.hpp.
#include "saccade_shipping/double_buffer_runtime.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/native_build.hpp"
#include "tracking/pipeline.hpp"

namespace saccade::shipping {

namespace {

[[noreturn]] void run_error(const std::string& what) {
    throw std::runtime_error("double-buffer runtime: " + what);
}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) run_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

const char* double_buffer_mutation_name(DoubleBufferMutation m) {
    switch (m) {
        case DoubleBufferMutation::None: return "none";
        case DoubleBufferMutation::StaleDetectorInput: return "stale_detector_input";
        case DoubleBufferMutation::StaleGmcInput: return "stale_gmc_input";
        case DoubleBufferMutation::SwappedDetectionParity: return "swapped_detection_parity";
    }
    return "?";
}

DoubleBufferMutation parse_double_buffer_mutation(const std::string& name) {
    for (DoubleBufferMutation m :
         {DoubleBufferMutation::None, DoubleBufferMutation::StaleDetectorInput,
          DoubleBufferMutation::StaleGmcInput, DoubleBufferMutation::SwappedDetectionParity}) {
        if (name == double_buffer_mutation_name(m)) return m;
    }
    throw std::invalid_argument("unknown double-buffer mutation " + name);
}

DoubleBufferRuntime::Stream::Stream() {
    cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
}

DoubleBufferRuntime::Stream::~Stream() {
    if (s != nullptr) cudaStreamDestroy(s);
}

// Per parity: input_ready (main stream -> detect stream), ready (detect
// stream -> main stream) and normalized (the decode buffer is free again).
struct DoubleBufferRuntime::Events {
    cudaEvent_t input_ready[2] = {}, ready[2] = {}, normalized[2] = {};
    bool normalize_pending[2] = {false, false};

    Events() {
        for (int p = 0; p < 2; ++p) {
            for (cudaEvent_t* e : {&input_ready[p], &ready[p], &normalized[p]}) {
                cuda_check(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "event");
            }
        }
    }
    ~Events() {
        for (int p = 0; p < 2; ++p) {
            for (cudaEvent_t e : {input_ready[p], ready[p], normalized[p]}) {
                if (e != nullptr) cudaEventDestroy(e);
            }
        }
    }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
};

// One parity's detector rows on device, as the oracle's output clones hand
// them to `_run_native_tensor_prep`: boxes [n*4], scores [n], int32 classes [n].
struct DoubleBufferRuntime::DetectionBuffers {
    float* boxes = nullptr;
    float* scores = nullptr;
    std::int32_t* classes = nullptr;
    int capacity = 0;

    explicit DetectionBuffers(int n) : capacity(n) {
        const auto c = static_cast<std::size_t>(n);
        cuda_check(cudaMalloc(&boxes, c * 4 * sizeof(float)), "detection buffers");
        cuda_check(cudaMalloc(&scores, c * sizeof(float)), "detection buffers");
        cuda_check(cudaMalloc(&classes, c * sizeof(std::int32_t)), "detection buffers");
    }
    ~DetectionBuffers() {
        cudaFree(boxes);
        cudaFree(scores);
        cudaFree(classes);
    }
    DetectionBuffers(const DetectionBuffers&) = delete;
    DetectionBuffers& operator=(const DetectionBuffers&) = delete;

    DeviceRowsOut out() const { return DeviceRowsOut{boxes, scores, classes, capacity}; }
    shipping::DeviceDetections view(int n) const {
        // The whole-graph path never tiles (detector_plan.hpp).
        return shipping::DeviceDetections{boxes, scores, classes, n, false};
    }
    DetectionRows read(int n, cudaStream_t stream) const {
        DetectionRows r;
        const auto c = static_cast<std::size_t>(n);
        r.boxes.resize(c * 4);
        r.scores.resize(c);
        r.classes.resize(c);
        if (n > 0) {
            cuda_check(cudaMemcpyAsync(r.boxes.data(), boxes, c * 4 * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream), "trace rows");
            cuda_check(cudaMemcpyAsync(r.scores.data(), scores, c * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream), "trace rows");
            cuda_check(cudaMemcpyAsync(r.classes.data(), classes, c * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream), "trace rows");
            cuda_check(cudaStreamSynchronize(stream), "trace rows");
        }
        return r;
    }
};

DoubleBufferRuntime::DoubleBufferRuntime(const ResolvedShippingConfig& cfg,
                                         const DetectorInputs& detector_inputs,
                                         const std::string& model_root)
    : cfg_(cfg),
      ingest_plan_(plan_ingest(cfg)),
      detector_plan_(plan_detector_files(cfg, detector_inputs)),
      output_plan_(plan_sequence_output(cfg)),
      schedule_plan_(plan_schedule(cfg)) {
    if (!output_plan_.write_output) {
        throw ConfigError("the config writes no MOT output (steps.tail.write_output)");
    }
    if (!schedule_plan_.double_buffer) {
        throw ConfigError("the config's schedule is not the double buffer (steps.schedule.double_buffer)");
    }
    events_ = std::make_unique<Events>();
    decoder_ = std::make_unique<JpegDecoder>();
    detector_ = std::make_unique<DetectorHost>(detector_plan_, model_root, detect_.s);
    pipeline_ = build_perception_pipeline(cfg_);
    for (auto& d : det_) d = std::make_unique<DetectionBuffers>(detector_plan_.max_det);
}

DoubleBufferRuntime::~DoubleBufferRuntime() {
    cudaStreamSynchronize(detect_.s);
    cudaStreamSynchronize(main_.s);
}

const HeadLoadReport& DoubleBufferRuntime::load_report() const { return detector_->load_report(); }

const NvjpegLibraryInfo& DoubleBufferRuntime::nvjpeg() const { return decoder_->library(); }

const WholeGraphStats& DoubleBufferRuntime::detector_graph_stats() const {
    return detector_->graph_stats();
}

SequenceRunResult DoubleBufferRuntime::run_sequence(const std::filesystem::path& sequence_dir,
                                                    int max_frames, FrameObserver* observer) {
    const SequenceInput input = read_sequence_input(sequence_dir, max_frames);
    const cudaStream_t main = main_.s, detect = detect_.s;
    Events& ev = *events_;

    // Nothing of the previous sequence is still queued (run_sequence ends
    // with both streams synchronized), so its buffers and graphs may go.
    detector_->set_image_dims(input.im_height, input.im_width);
    const WholeGraphStats det_before = detector_->graph_stats();
    IngestHost ingest(ingest_plan_, input, *decoder_, main, /*pools=*/2);
    const SequenceGeometry geometry{input.im_width, input.im_height};
    PostDetectorHost post(cfg_, geometry, *pipeline_, main, GraphMode::Captured);
    post.set_stale_gmc_input_for_measurement(mutation_ == DoubleBufferMutation::StaleGmcInput);
    SequenceOutput output(output_plan_);
    for (bool& b : ev.normalize_pending) b = false;

    SequenceRunResult out;
    SequenceRunStats& st = out.stats;
    std::filesystem::path name = sequence_dir.lexically_normal();
    if (name.filename().empty()) name = name.parent_path();  // "MOT17-09-SDP/"
    st.name = name.filename().string();
    st.im_width = input.im_width;
    st.im_height = input.im_height;
    st.seq_length = input.seq_length;

    struct Pending {
        int frame = 0, parity = 0, rows = 0;
        DecodePath path = DecodePath::HardwareBatched;
    };
    // _launch_double_buffer_detect for frame k.
    auto launch = [&](int k) {
        Pending pd;
        pd.frame = k;
        pd.parity = (k - 1) % 2;
        const int p = pd.parity;
        if (ev.normalize_pending[p]) {
            cuda_check(cudaEventSynchronize(ev.normalized[p]), "decode buffer");
            ev.normalize_pending[p] = false;
        }
        pd.path = ingest.decode(k, p).path;
        cuda_check(cudaEventRecord(ev.input_ready[p], main), "input_ready");
        cuda_check(cudaStreamWaitEvent(detect, ev.input_ready[p], 0), "input_ready");
        ingest.normalize(p, detect);
        cuda_check(cudaEventRecord(ev.normalized[p], detect), "normalized");
        ev.normalize_pending[p] = true;
        pd.rows = detector_->detect_graphed(ingest.frame_chw(p), ingest.height(), ingest.width(),
                                            det_[p]->out(),
                                            mutation_ != DoubleBufferMutation::StaleDetectorInput);
        cuda_check(cudaEventRecord(ev.ready[p], detect), "ready");
        return pd;
    };

    const int frame_end = static_cast<int>(ingest.frame_count());
    const auto loop_start = std::chrono::steady_clock::now();
    try {
        std::optional<Pending> pending;
        if (frame_end >= 1) pending = launch(1);
        for (int k = 1; k <= frame_end; ++k) {
            std::optional<Pending> next;
            if (k < frame_end) next = launch(k + 1);
            const Pending& pd = *pending;
            int read_parity = pd.parity;
            int rows = pd.rows;
            cuda_check(cudaStreamWaitEvent(main, ev.ready[pd.parity], 0), "ready");
            if (mutation_ == DoubleBufferMutation::SwappedDetectionParity) {
                read_parity = 1 - pd.parity;
                if (next) {
                    cuda_check(cudaStreamWaitEvent(main, ev.ready[read_parity], 0), "ready");
                    rows = next->rows;
                }
            }
            (pd.path == DecodePath::HardwareBatched ? st.hardware_decodes : st.decoupled_decodes)++;
            const FrameResult r =
                post.process(det_[read_parity]->view(rows), ingest.frame_chw(pd.parity));
            if (r.updated) {
                ++st.tracker_updates;
                output.add_frame(k, r.tracker_output.boxes.data(), r.tracker_output.scores.data(),
                                 r.tracker_output.ids.data(), r.tracker_output.size());
            } else {
                ++st.skipped_empty;
            }
            if (observer != nullptr) {
                const DetectionRows rows_host = det_[read_parity]->read(rows, main);
                observer->on_frame(FrameTrace{k, pd.path, &rows_host, &r});
            }
            ++st.frames;
            pending = next;
        }
        cuda_check(cudaStreamSynchronize(detect), "end of sequence");
        cuda_check(cudaStreamSynchronize(main), "end of sequence");
    } catch (...) {
        cudaStreamSynchronize(detect);
        cudaStreamSynchronize(main);
        throw;
    }
    st.schedule.loop_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();

    const WholeGraphStats& det_after = detector_->graph_stats();
    st.schedule.detector_captures = det_after.captures - det_before.captures;
    st.schedule.detector_warmup_runs = det_after.warmup_runs - det_before.warmup_runs;
    st.schedule.detector_replays = det_after.replays - det_before.replays;
    st.schedule.post = post.graph_stats();
    st.pre_roll_updates = post.pre_roll_updates_run();
    st.track_ids = output.track_ids();
    out.lines = output.finish(&st.interpolation);
    st.lines = out.lines.size();
    ++sequences_run_;
    return out;
}

}  // namespace saccade::shipping
