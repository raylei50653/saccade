// Native ingest host (#465 Phase B PR-7). See saccade_shipping/ingest_host.hpp.
#include "saccade_shipping/ingest_host.hpp"

#include <nvjpeg.h>

#include <stdexcept>
#include <string>

namespace saccade::shipping {
namespace {

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("shipping ingest: ") + what + ": " + cudaGetErrorString(e));
    }
}

void nvjpeg_check(nvjpegStatus_t s, const char* what) {
    if (s != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("shipping ingest: ") + what + " failed: nvjpeg status " +
                                 std::to_string(static_cast<int>(s)));
    }
}

}  // namespace

const char* decode_path_name(DecodePath p) {
    return p == DecodePath::HardwareBatched ? "hardware_batched" : "decoupled";
}

// The decoder's nvJPEG objects, as CUDAJpegDecoder holds them.
struct JpegDecoder::State {
    cudaStream_t stream = nullptr;
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t state = nullptr;            // batched (hardware) decode
    nvjpegJpegState_t decoupled_state = nullptr;  // decoupled decode
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegBufferPinned_t pinned[2] = {nullptr, nullptr};
    nvjpegBufferDevice_t device_buffer = nullptr;
    nvjpegJpegStream_t streams[2] = {nullptr, nullptr};
    nvjpegDecodeParams_t params = nullptr;

    ~State() {
        if (params) nvjpegDecodeParamsDestroy(params);
        for (auto* s : streams) if (s) nvjpegJpegStreamDestroy(s);
        if (device_buffer) nvjpegBufferDeviceDestroy(device_buffer);
        for (auto* p : pinned) if (p) nvjpegBufferPinnedDestroy(p);
        if (decoupled_state) nvjpegJpegStateDestroy(decoupled_state);
        if (decoder) nvjpegDecoderDestroy(decoder);
        if (state) nvjpegJpegStateDestroy(state);
        if (handle) nvjpegDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

JpegDecoder::JpegDecoder() : s_(std::make_unique<State>()) {
    nvjpeg_check(nvjpegGetProperty(MAJOR_VERSION, &library_.major), "nvjpegGetProperty");
    nvjpeg_check(nvjpegGetProperty(MINOR_VERSION, &library_.minor), "nvjpegGetProperty");
    nvjpeg_check(nvjpegGetProperty(PATCH_LEVEL, &library_.patch), "nvjpegGetProperty");

    // at::cuda::getStreamFromPool(false): a non-blocking stream of default priority.
    cuda_check(cudaStreamCreateWithFlags(&s_->stream, cudaStreamNonBlocking), "decoder stream");

    library_.hardware_decode = true;
    nvjpegStatus_t st = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr,
                                       NVJPEG_FLAGS_DEFAULT, &s_->handle);
    if (st == NVJPEG_STATUS_ARCH_MISMATCH) {
        nvjpeg_check(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, NVJPEG_FLAGS_DEFAULT,
                                    &s_->handle),
                     "nvjpegCreateEx(default backend)");
        library_.hardware_decode = false;
    } else {
        nvjpeg_check(st, "nvjpegCreateEx(hardware backend)");
    }
    nvjpeg_check(nvjpegJpegStateCreate(s_->handle, &s_->state), "nvjpegJpegStateCreate");
    nvjpeg_check(nvjpegDecoderCreate(s_->handle, NVJPEG_BACKEND_DEFAULT, &s_->decoder),
                 "nvjpegDecoderCreate");
    nvjpeg_check(nvjpegDecoderStateCreate(s_->handle, s_->decoder, &s_->decoupled_state),
                 "nvjpegDecoderStateCreate");
    for (auto& p : s_->pinned) {
        nvjpeg_check(nvjpegBufferPinnedCreate(s_->handle, nullptr, &p), "nvjpegBufferPinnedCreate");
    }
    nvjpeg_check(nvjpegBufferDeviceCreate(s_->handle, nullptr, &s_->device_buffer),
                 "nvjpegBufferDeviceCreate");
    for (auto& js : s_->streams) {
        nvjpeg_check(nvjpegJpegStreamCreate(s_->handle, &js), "nvjpegJpegStreamCreate");
    }
    nvjpeg_check(nvjpegDecodeParamsCreate(s_->handle, &s_->params), "nvjpegDecodeParamsCreate");
}

JpegDecoder::~JpegDecoder() = default;

JpegImageInfo JpegDecoder::image_info(const std::vector<std::uint8_t>& bitstream) const {
    int widths[NVJPEG_MAX_COMPONENT] = {};
    int heights[NVJPEG_MAX_COMPONENT] = {};
    int components = 0;
    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
    const nvjpegStatus_t st = nvjpegGetImageInfo(s_->handle, bitstream.data(), bitstream.size(),
                                                 &components, &subsampling, widths, heights);
    if (st != NVJPEG_STATUS_SUCCESS) {
        throw InputError("shipping ingest: nvjpegGetImageInfo failed: nvjpeg status " +
                         std::to_string(static_cast<int>(st)));
    }
    if (subsampling == NVJPEG_CSS_UNKNOWN) throw InputError("shipping ingest: unknown chroma subsampling");
    return JpegImageInfo{widths[0], heights[0], components, static_cast<int>(subsampling)};
}

DecodePath JpegDecoder::decode_rgb(const std::vector<std::uint8_t>& bitstream,
                                   const JpegImageInfo& info, std::uint8_t* rgb_planar) {
    State& s = *s_;
    nvjpegImage_t out{};
    const std::size_t plane = static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height);
    for (int c = 0; c < 3; ++c) {
        out.channel[c] = rgb_planar + c * plane;
        out.pitch[c] = static_cast<std::size_t>(info.width);
    }
    cuda_check(cudaStreamSynchronize(s.stream), "decoder stream sync");

    DecodePath path = DecodePath::Decoupled;
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    if (library_.hardware_decode && !force_decoupled_) {
#else
    if (library_.hardware_decode) {
#endif
        // Statuses unchecked, as in torchvision: a failed probe leaves
        // is_supported at -1 and sends the bitstream down the decoupled path.
        nvjpegJpegStreamParseHeader(s.handle, bitstream.data(), bitstream.size(), s.streams[0]);
        int is_supported = -1;
        nvjpegDecodeBatchedSupported(s.handle, s.streams[0], &is_supported);
        if (is_supported == 0) path = DecodePath::HardwareBatched;
    }

    if (path == DecodePath::HardwareBatched) {
        nvjpeg_check(nvjpegDecodeBatchedInitialize(s.handle, s.state, 1, 1, NVJPEG_OUTPUT_RGB),
                     "nvjpegDecodeBatchedInitialize");
        const unsigned char* data = bitstream.data();
        const std::size_t length = bitstream.size();
        nvjpeg_check(nvjpegDecodeBatched(s.handle, s.state, &data, &length, &out, s.stream),
                     "nvjpegDecodeBatched");
    } else {
        nvjpeg_check(nvjpegStateAttachDeviceBuffer(s.decoupled_state, s.device_buffer),
                     "nvjpegStateAttachDeviceBuffer");
        int buffer_index = 0;
        nvjpeg_check(nvjpegDecodeParamsSetOutputFormat(s.params, NVJPEG_OUTPUT_RGB),
                     "nvjpegDecodeParamsSetOutputFormat");
        nvjpeg_check(nvjpegJpegStreamParse(s.handle, bitstream.data(), bitstream.size(), 0, 0,
                                           s.streams[buffer_index]),
                     "nvjpegJpegStreamParse");
        nvjpeg_check(nvjpegStateAttachPinnedBuffer(s.decoupled_state, s.pinned[buffer_index]),
                     "nvjpegStateAttachPinnedBuffer");
        nvjpeg_check(nvjpegDecodeJpegHost(s.handle, s.decoder, s.decoupled_state, s.params,
                                          s.streams[buffer_index]),
                     "nvjpegDecodeJpegHost");
        cuda_check(cudaStreamSynchronize(s.stream), "decoder stream sync");
        nvjpeg_check(nvjpegDecodeJpegTransferToDevice(s.handle, s.decoder, s.decoupled_state,
                                                      s.streams[buffer_index], s.stream),
                     "nvjpegDecodeJpegTransferToDevice");
        nvjpeg_check(nvjpegDecodeJpegDevice(s.handle, s.decoder, s.decoupled_state, &out, s.stream),
                     "nvjpegDecodeJpegDevice");
    }
    cuda_check(cudaStreamSynchronize(s.stream), "decoder stream sync");
    return path;
}

IngestHost::IngestHost(const IngestPlan& plan, const SequenceInput& input, JpegDecoder& decoder,
                       cudaStream_t stream, int pools)
    : plan_(plan), input_(input), decoder_(decoder), stream_(stream) {
    if (input_.im_width <= 0 || input_.im_height <= 0) {
        throw InputError("shipping ingest: sequence geometry must be positive");
    }
    if (pools != 1 && pools != 2) throw std::invalid_argument("shipping ingest: pools must be 1 or 2");
    const std::size_t n = frame_floats();
    try {
        for (int p = 0; p < pools; ++p) {
            rgb_.push_back(nullptr);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&rgb_.back()), n), "decode buffer");
            // AdaptiveFramePool.frame_buffer: torch.zeros((3, h, w), float32).
            frame_.push_back(nullptr);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&frame_.back()), n * sizeof(float)),
                       "frame buffer");
            cuda_check(cudaMemsetAsync(frame_.back(), 0, n * sizeof(float), stream_), "frame buffer");
        }
        cuda_check(cudaStreamSynchronize(stream_), "frame buffer");
    } catch (...) {
        for (float* f : frame_) cudaFree(f);
        for (std::uint8_t* r : rgb_) cudaFree(r);
        throw;
    }
}

IngestHost::~IngestHost() {
    for (float* f : frame_) cudaFree(f);
    for (std::uint8_t* r : rgb_) cudaFree(r);
}

std::size_t IngestHost::frame_floats() const {
    return 3 * static_cast<std::size_t>(input_.im_width) * static_cast<std::size_t>(input_.im_height);
}

IngestFrame IngestHost::decode(int frame, int pool) {
    if (frame < 1 || static_cast<std::size_t>(frame) > input_.frames.size()) {
        throw std::out_of_range("shipping ingest: frame " + std::to_string(frame) + " out of range");
    }
    std::uint8_t* rgb = rgb_.at(static_cast<std::size_t>(pool));
    const std::string& name = input_.frames[static_cast<std::size_t>(frame - 1)];
    const std::vector<std::uint8_t> bytes = read_frame_file(input_.img_dir / name);
    const JpegImageInfo info = decoder_.image_info(bytes);
    // The oracle writes the decoded frame into a buffer sized from seqinfo.ini
    // with out=; a different size would resize that buffer, not fill it.
    if (info.width != input_.im_width || info.height != input_.im_height) {
        throw InputError("shipping ingest: " + name + " decodes to " + std::to_string(info.width) + "x" +
                         std::to_string(info.height) + ", seqinfo.ini says " +
                         std::to_string(input_.im_width) + "x" + std::to_string(input_.im_height));
    }
    return IngestFrame{name, decoder_.decode_rgb(bytes, info, rgb)};
}

void IngestHost::normalize(int pool, cudaStream_t stream) {
    const auto p = static_cast<std::size_t>(pool);
    normalize_rgb_u8(rgb_.at(p), frame_.at(p), frame_floats(), plan_.normalize_scale, stream);
}

IngestFrame IngestHost::ingest(int frame) {
    const IngestFrame f = decode(frame, 0);
    normalize(0, stream_);
    cuda_check(cudaStreamSynchronize(stream_), "ingest stream sync");
    return f;
}

}  // namespace saccade::shipping
