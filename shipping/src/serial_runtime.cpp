// Native end-to-end serial runtime (#465 Phase B PR-9); see serial_runtime.hpp.
#include "saccade_shipping/serial_runtime.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

#include "saccade_shipping/native_build.hpp"
#include "tracking/pipeline.hpp"

namespace saccade::shipping {

namespace {

[[noreturn]] void run_error(const std::string& what) {
    throw std::runtime_error("serial runtime: " + what);
}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) run_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

const char* runtime_mutation_name(RuntimeMutation m) {
    switch (m) {
        case RuntimeMutation::None: return "none";
        case RuntimeMutation::SharedPostHost: return "shared_post_host";
        case RuntimeMutation::StaleImageDims: return "stale_image_dims";
        case RuntimeMutation::GmcPreviousFrame: return "gmc_previous_frame";
    }
    return "?";
}

RuntimeMutation parse_runtime_mutation(const std::string& name) {
    for (RuntimeMutation m : {RuntimeMutation::None, RuntimeMutation::SharedPostHost,
                              RuntimeMutation::StaleImageDims, RuntimeMutation::GmcPreviousFrame}) {
        if (name == runtime_mutation_name(m)) return m;
    }
    throw std::invalid_argument("unknown runtime mutation " + name);
}

SerialRuntime::Stream::Stream() {
    cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
}

SerialRuntime::Stream::~Stream() {
    if (s != nullptr) cudaStreamDestroy(s);
}

// The detector rows of the current frame on device, as _run_native_tensor_prep
// hands them to native NMS: boxes [n*4], scores [n], int32 classes [n].
// Capacity max_det (the S2 row count).
struct SerialRuntime::DeviceDetections {
    float* boxes = nullptr;
    float* scores = nullptr;
    std::int32_t* classes = nullptr;
    std::size_t capacity = 0;

    explicit DeviceDetections(std::size_t n) : capacity(n) {
        cuda_check(cudaMalloc(&boxes, n * 4 * sizeof(float)), "detection buffers");
        cuda_check(cudaMalloc(&scores, n * sizeof(float)), "detection buffers");
        cuda_check(cudaMalloc(&classes, n * sizeof(std::int32_t)), "detection buffers");
    }
    ~DeviceDetections() {
        cudaFree(boxes);
        cudaFree(scores);
        cudaFree(classes);
    }

    DeviceDetections(const DeviceDetections&) = delete;
    DeviceDetections& operator=(const DeviceDetections&) = delete;

    shipping::DeviceDetections upload(const DetectionRows& rows, cudaStream_t stream) const {
        const std::size_t n = rows.size();
        if (n > capacity) run_error("detector returned more rows than max_det");
        if (rows.boxes.size() != n * 4 || rows.classes.size() != n) run_error("ragged detector rows");
        if (n > 0) {
            cuda_check(cudaMemcpyAsync(boxes, rows.boxes.data(), n * 4 * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "detections");
            cuda_check(cudaMemcpyAsync(scores, rows.scores.data(), n * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "detections");
            cuda_check(cudaMemcpyAsync(classes, rows.classes.data(), n * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream), "detections");
            // The host vectors are the caller's; finish before they can go away.
            cuda_check(cudaStreamSynchronize(stream), "detections");
        }
        // The whole-graph path never tiles (detector_plan.hpp).
        return shipping::DeviceDetections{boxes, scores, classes, static_cast<int>(n), false};
    }
};

SerialRuntime::SerialRuntime(const ResolvedShippingConfig& cfg,
                             const DetectorInputs& detector_inputs, const std::string& model_root)
    : cfg_(cfg),
      ingest_plan_(plan_ingest(cfg)),
      detector_plan_(plan_detector_files(cfg, detector_inputs)),
      output_plan_(plan_sequence_output(cfg)) {
    if (!output_plan_.write_output) {
        throw ConfigError("the config writes no MOT output (steps.tail.write_output)");
    }
    decoder_ = std::make_unique<JpegDecoder>();
    detector_ = std::make_unique<DetectorHost>(detector_plan_, model_root, stream_.s);
    pipeline_ = build_perception_pipeline(cfg_);
    det_ = std::make_unique<DeviceDetections>(static_cast<std::size_t>(detector_plan_.max_det));
}

SerialRuntime::~SerialRuntime() = default;

const HeadLoadReport& SerialRuntime::load_report() const { return detector_->load_report(); }

const NvjpegLibraryInfo& SerialRuntime::nvjpeg() const { return decoder_->library(); }

SequenceRunResult SerialRuntime::run_sequence(const std::filesystem::path& sequence_dir,
                                              int max_frames, FrameObserver* observer) {
    const SequenceInput input = read_sequence_input(sequence_dir, max_frames);
    const cudaStream_t stream = stream_.s;

    if (mutation_ != RuntimeMutation::StaleImageDims || sequences_run_ == 0) {
        detector_->set_image_dims(input.im_height, input.im_width);
    }
    IngestHost ingest(ingest_plan_, input, *decoder_, stream);
    std::unique_ptr<PostDetectorHost> own_post;
    PostDetectorHost* post = nullptr;
    const SequenceGeometry geometry{input.im_width, input.im_height};
    if (mutation_ == RuntimeMutation::SharedPostHost) {
        if (!shared_post_ || shared_geometry_.im_width != geometry.im_width ||
            shared_geometry_.im_height != geometry.im_height) {
            shared_post_ = std::make_unique<PostDetectorHost>(cfg_, geometry, *pipeline_, stream);
            shared_geometry_ = geometry;
        }
        post = shared_post_.get();
    } else {
        own_post = std::make_unique<PostDetectorHost>(cfg_, geometry, *pipeline_, stream);
        post = own_post.get();
    }
    SequenceOutput output(output_plan_);

    // GmcPreviousFrame only: a copy of the previous frame buffer.
    float* previous = nullptr;
    const std::size_t frame_floats =
        3 * static_cast<std::size_t>(input.im_width) * static_cast<std::size_t>(input.im_height);
    if (mutation_ == RuntimeMutation::GmcPreviousFrame) {
        cuda_check(cudaMalloc(&previous, frame_floats * sizeof(float)), "previous frame");
    }

    SequenceRunResult out;
    SequenceRunStats& st = out.stats;
    std::filesystem::path name = sequence_dir.lexically_normal();
    if (name.filename().empty()) name = name.parent_path();  // "MOT17-09-SDP/"
    st.name = name.filename().string();
    st.im_width = input.im_width;
    st.im_height = input.im_height;
    st.seq_length = input.seq_length;
    const auto loop_start = std::chrono::steady_clock::now();
    try {
        for (int k = 1; k <= static_cast<int>(ingest.frame_count()); ++k) {
            const IngestFrame f = ingest.ingest(k);
            (f.path == DecodePath::HardwareBatched ? st.hardware_decodes : st.decoupled_decodes)++;
            const DetectionRows rows = detector_->detect(ingest.frame_chw(), ingest.height(), ingest.width());
            const float* gmc_frame = ingest.frame_chw();
            if (previous != nullptr) {
                if (k == 1) {
                    cuda_check(cudaMemcpyAsync(previous, ingest.frame_chw(), frame_floats * sizeof(float),
                                               cudaMemcpyDeviceToDevice, stream), "previous frame");
                }
                gmc_frame = previous;
            }
            const FrameResult r = post->process(det_->upload(rows, stream), gmc_frame);
            if (previous != nullptr) {
                cuda_check(cudaMemcpyAsync(previous, ingest.frame_chw(), frame_floats * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream), "previous frame");
                cuda_check(cudaStreamSynchronize(stream), "previous frame");
            }
            if (r.updated) {
                ++st.tracker_updates;
                output.add_frame(k, r.tracker_output.boxes.data(), r.tracker_output.scores.data(),
                                 r.tracker_output.ids.data(), r.tracker_output.size());
            } else {
                ++st.skipped_empty;
            }
            if (observer != nullptr) observer->on_frame(FrameTrace{k, f.path, &rows, &r});
            ++st.frames;
        }
    } catch (...) {
        cudaFree(previous);
        throw;
    }
    cudaFree(previous);
    st.schedule.loop_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();

    st.pre_roll_updates = post->pre_roll_updates_run();
    st.track_ids = output.track_ids();
    out.lines = output.finish(&st.interpolation);
    st.lines = out.lines.size();
    ++sequences_run_;
    return out;
}

}  // namespace saccade::shipping
