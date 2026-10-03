// Native ingest plan and sequence input (#465 Phase B PR-7, U3b-1). CUDA-free.
//
// The oracle's ingest (§2 oracle, serial or double-buffer alike) is, per frame:
//
//   file    the k-th entry of sorted(str(p.absolute()) for p in
//           (seq / "img1").glob("*.jpg")), k = 1..min(max_frames, seqLength)
//           (pipeline.py, streaming.py TorchvisionGpuStreamer, evaluator.py);
//   decode  torchvision.io.decode_jpeg(read_file(f), device="cuda",
//           mode=ImageReadMode.RGB): nvJPEG, planar RGB uint8 [3, H, W]
//           (ingest_host.hpp mirrors torchvision's decoder);
//   ingest  torch.div(frame_hwc.permute(2, 0, 1), 255.0, out=pool.frame_buffer)
//           (stages.py _run_detect), into a float32 [3, imHeight, imWidth]
//           buffer sized from seqinfo.ini. On CUDA torch computes a division
//           by a Python scalar as a multiply by the float32 reciprocal, so the
//           value is float(x) * (1.0f / 255.0f) -- not x / 255.0f, which
//           differs on 126 of the 256 inputs (and is what CPU torch computes).
//
// `plan_ingest` turns the resolved config into that ingest and fails closed
// (ConfigError) on every oracle gate on the path the native ingest does not
// implement. Those facts that are code literals in the oracle (the file glob,
// the RGB mode, the 255.0 divisor) are pinned to the oracle source by
// tests/unit/test_native_ingest_oracle_pins.py, not read from the config.
// Nothing here reads the process environment.
#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/resolved_config.hpp"

namespace saccade::shipping {

// The input does not have the form the shipping entrypoint accepts
// (boundary §2), or is a form the oracle would fail on.
class InputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// float32 1/255, as torch's CUDA division kernel forms it.
inline constexpr float kIngestNormalizeScale = 1.0f / 255.0f;

// The oracle's u8 -> float32 ingest of one value (host twin of the kernel in
// ingest_host.hpp). tests/native/fixtures/shipping_ingest.json pins all 256
// values to torch on CUDA.
inline float ingest_normalize(std::uint8_t x) {
    return static_cast<float>(x) * kIngestNormalizeScale;
}

struct IngestPlan {
    // Planar RGB uint8 decode, then the float32 frame buffer [3, H, W]; the
    // only ingest the plan admits. Kept as a value so hosts take a plan, not
    // a config they would have to re-check.
    float normalize_scale = kIngestNormalizeScale;
};

IngestPlan plan_ingest(const ResolvedShippingConfig& cfg);

// One sequence's input as the oracle reads it.
struct SequenceInput {
    std::filesystem::path img_dir;  // <sequence>/img1
    int im_width = 0;               // seqinfo.ini [Sequence] imWidth
    int im_height = 0;              // seqinfo.ini [Sequence] imHeight
    int seq_length = 0;             // seqinfo.ini [Sequence] seqLength
    // Every img1 entry the oracle's glob lists, in its order (file names).
    std::vector<std::string> listed;
    // The frames the oracle consumes: listed[0 .. frame_end), frame k = k-th.
    std::vector<std::string> frames;
};

// Reads <sequence>/seqinfo.ini and lists <sequence>/img1. `max_frames` <= 0
// means no bound (the oracle's `max_frames or int(1e9)`).
//
// seqinfo.ini is read as configparser reads it for these keys (keys are
// case-insensitive, `=` or `:`, `#`/`;` comment lines, surrounding whitespace
// and CR stripped), restricted to the plain form: indented lines (configparser
// continuation syntax), duplicate sections or keys, or an integer that is
// not [+-]digits (which also refuses a `%` interpolation) are refused rather
// than interpreted. Keys are
// read from [Sequence] only; configparser's fallback to [DEFAULT] is not
// implemented, so a key found only there is refused as missing. The listing matches every directory entry
// whose name ends in ".jpg" (pathlib's glob: dot files, directories and
// symlinks included, case-sensitive) and sorts by path bytes, which is
// Python's string order for ASCII names; a non-ASCII name is refused. Fewer
// listed entries than frames to consume is refused: the oracle's frame loop
// stops at the end of the listing and writes a truncated sequence, which the
// shipping entrypoint does not reproduce. Extra entries are listed and not
// consumed, as in the oracle.
SequenceInput read_sequence_input(const std::filesystem::path& sequence_dir, int max_frames = 0);

// The seqinfo.ini reader on its own (text of the file).
struct SeqInfo {
    int im_width = 0, im_height = 0, seq_length = 0;
};
SeqInfo parse_seqinfo(const std::string& text);

// The bytes the oracle hands nvJPEG for one frame (torchvision read_file).
// Refuses what decode_jpeg would refuse before decoding: a path that is not a
// regular file (after symlinks) or an empty file.
std::vector<std::uint8_t> read_frame_file(const std::filesystem::path& path);

}  // namespace saccade::shipping
