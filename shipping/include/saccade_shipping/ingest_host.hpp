// Native ingest host (#465 Phase B PR-7, U3b-1): nvJPEG decode + normalize.
//
// `JpegDecoder` is a twin of torchvision 0.26's CUDAJpegDecoder
// (torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp), the decoder the
// oracle's decode_jpeg(device="cuda", mode=RGB) runs, called the way the
// oracle calls it -- one bitstream per call:
//   * handle: nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE); on
//     NVJPEG_STATUS_ARCH_MISMATCH, nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT) and
//     no hardware decode. A decoupled decoder (NVJPEG_BACKEND_DEFAULT) with
//     two pinned buffers, one device buffer and two jpeg streams;
//   * output: nvjpegGetImageInfo; NVJPEG_OUTPUT_RGB into planar uint8
//     [3, height[0], width[0]], pitch width[0];
//   * path: with hardware decode, a bitstream nvjpegDecodeBatchedSupported
//     accepts (status unchecked, as there) is decoded by
//     nvjpegDecodeBatchedInitialize(batch 1, 1 CPU thread) +
//     nvjpegDecodeBatched; any other goes through the decoupled
//     host / transfer / device phases;
//   * the decoder's own non-blocking stream, synchronized before and after.
// The library is the oracle's own nvJPEG build: shipping/CMakeLists.txt links
// a copy that must be byte-identical to torchvision's bundled libnvjpeg.
//
// `normalize_rgb_u8` is the oracle's ingest op (ingest_plan.hpp):
// out[i] = float(in[i]) * scale, rounded to nearest, over the planar buffer;
// the [3, H, W] layout is the oracle's (CHW decode -> HWC view -> CHW view).
//
// `IngestHost` runs one sequence: read the file, decode, check the decoded
// size against seqinfo.ini, normalize into the float32 frame buffer -- the
// oracle's pool.frame_buffer, i.e. the `frame_chw` PostDetectorHost::process
// takes. `ingest` is the serial path (decode + normalize + synchronize, pool
// 0). The double-buffer schedule (PR-10, U5) holds two pools, as the oracle
// holds two AdaptiveFramePools, and splits the two halves: `decode` into a
// pool's decode buffer on the host thread (the oracle's decode worker; the
// decode has completed when it returns), `normalize` into that pool's frame
// buffer on the caller's (detect) stream, as `_run_detect`'s ingest_preprocess
// runs on the double-buffer stream. Reads no environment and no value outside
// the plan and the sequence input.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "saccade_shipping/ingest_plan.hpp"

namespace saccade::shipping {

enum class DecodePath { HardwareBatched, Decoupled };
const char* decode_path_name(DecodePath p);

struct NvjpegLibraryInfo {
    int major = 0, minor = 0, patch = 0;
    bool hardware_decode = false;  // the handle came up with NVJPEG_BACKEND_HARDWARE
};

struct JpegImageInfo {
    int width = 0, height = 0;  // component 0, as torchvision sizes its output
    int components = 0;
    int subsampling = 0;  // nvjpegChromaSubsampling_t
};

class JpegDecoder {
public:
    JpegDecoder();
    ~JpegDecoder();
    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;

    const NvjpegLibraryInfo& library() const { return library_; }

    // nvjpegGetImageInfo; throws on failure or unknown chroma subsampling.
    JpegImageInfo image_info(const std::vector<std::uint8_t>& bitstream) const;

    // Decodes into `rgb_planar` (device, 3 * info.height * info.width bytes);
    // returns when the decode has completed.
    DecodePath decode_rgb(const std::vector<std::uint8_t>& bitstream, const JpegImageInfo& info,
                          std::uint8_t* rgb_planar);

    // Send every bitstream down the decoupled path (developer measurement
    // only: the parity harness's negative control; the shipping choice is
    // torchvision's, above).
    void force_decoupled_for_measurement(bool on) { force_decoupled_ = on; }

private:
    struct State;
    std::unique_ptr<State> s_;
    NvjpegLibraryInfo library_;
    bool force_decoupled_ = false;
};

// out[i] = float(in[i]) * scale (round to nearest) for i < count, on `stream`.
void normalize_rgb_u8(const std::uint8_t* in, float* out, std::size_t count, float scale,
                      cudaStream_t stream);

struct IngestFrame {
    std::string file;
    DecodePath path;
};

class IngestHost {
public:
    // `decoder` (one per process, as torchvision keeps one) must outlive the host.
    // `pools`: 1 (serial) or 2 (double buffer); each is zero-initialized.
    IngestHost(const IngestPlan& plan, const SequenceInput& input, JpegDecoder& decoder,
               cudaStream_t stream, int pools = 1);
    ~IngestHost();
    IngestHost(const IngestHost&) = delete;
    IngestHost& operator=(const IngestHost&) = delete;

    std::size_t frame_count() const { return input_.frames.size(); }

    // Ingests frame k (1-based) into the buffers below; synchronizes `stream`.
    // Throws InputError when the file is not one the oracle would ingest into
    // this sequence's frame buffer (unreadable, empty, or decoded size !=
    // imWidth x imHeight).
    IngestFrame ingest(int frame);

    // Decodes frame k (1-based) into `pool`'s decode buffer; the same checks
    // as `ingest`. Returns when the decode has completed. The caller orders it
    // after every read of that decode buffer.
    IngestFrame decode(int frame, int pool);
    // Enqueues the normalize of `pool`'s decode buffer into its frame buffer
    // on `stream`; does not synchronize.
    void normalize(int pool, cudaStream_t stream);

    int pools() const { return static_cast<int>(rgb_.size()); }
    std::size_t frame_floats() const;
    int width() const { return input_.im_width; }
    int height() const { return input_.im_height; }
    // Device, planar [3, height, width]: the decoder output and the frame buffer.
    const std::uint8_t* decoded_rgb(int pool = 0) const { return rgb_.at(static_cast<std::size_t>(pool)); }
    const float* frame_chw(int pool = 0) const { return frame_.at(static_cast<std::size_t>(pool)); }

private:
    IngestPlan plan_;
    SequenceInput input_;
    JpegDecoder& decoder_;
    cudaStream_t stream_;
    std::vector<std::uint8_t*> rgb_;
    std::vector<float*> frame_;
};

}  // namespace saccade::shipping
