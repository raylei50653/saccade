// saccade_ingest_probe: run the native ingest (#465 Phase B PR-7, U3b-1) over
// one sequence and stream what it produced, for the parity harness
// scripts/eval/diagnostics/native_ingest_parity.py.
//
// Developer tool (developer_build_debug). It runs exactly the shipping ingest
// (saccade_shipping/ingest_plan.hpp, ingest_host.hpp): plan from the resolved
// config, sequence input from seqinfo.ini + img1/, one JpegDecoder, one
// IngestHost; it compares nothing itself. Output on stdout, little-endian,
// format saccade.native_ingest_stream/v1 -- a sequence of records, each a
// uint64 byte length, that many bytes of JSON, then the binary payloads the
// JSON announces:
//
//   preamble  {"format", "nvjpeg": {"version": [major, minor, patch],
//             "hardware_decode"}, "force_decoupled", "sequence", "im_width", "im_height",
//             "seq_length", "listed": [names], "frames": [names],
//             "normalize_lut_f32_hex": [256 x 8 hex digits, the kernel's
//             output for 0..255, bytes in memory order]}
//   frame     {"frame": k, "file", "path": "hardware_batched"|"decoupled",
//             "u8_bytes", "f32_bytes"} + decoded planar RGB uint8 [3, H, W]
//             + frame buffer float32 [3, H, W]
//   end       {"end": true, "frames": n}
//
// Exit 0 when every frame was ingested; 2 on any error (message on stderr).
//
// Usage:
//   saccade_ingest_probe --config configs/shipping/mamba_whole_graph.resolved.json
//       --sequence datasets/MOT17/train/MOT17-09-SDP [--max-frames N]
//       [--force-decoupled]  (developer measurement: decode every frame on the
//                             decoupled path; recorded in the preamble)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/ingest_host.hpp"
#include "saccade_shipping/ingest_plan.hpp"
#include "saccade_shipping/resolved_config.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace sh = saccade::shipping;
using sh::JsonValue;

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

void write_bytes(const void* data, std::size_t n) {
    if (n != 0 && std::fwrite(data, 1, n, stdout) != n) fail("write to stdout failed");
}

void write_record(const JsonValue& v) {
    const std::string text = sh::dump_python_json(v);
    const std::uint64_t n = text.size();
    write_bytes(&n, sizeof n);  // x86-64: little-endian
    write_bytes(text.data(), text.size());
}

JsonValue string_array(const std::vector<std::string>& items) {
    std::vector<JsonValue> out;
    for (const auto& s : items) out.push_back(JsonValue::make_string(s));
    return JsonValue::make_array(std::move(out));
}

std::string hex_bytes(const void* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(digits[b[i] >> 4]);
        out.push_back(digits[b[i] & 15]);
    }
    return out;
}

// The kernel's output for every input value.
JsonValue normalize_lut(const sh::IngestPlan& plan, cudaStream_t stream) {
    std::vector<std::uint8_t> in(256);
    for (int i = 0; i < 256; ++i) in[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
    std::uint8_t* d_in = nullptr;
    float* d_out = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_in), 256), "lut");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_out), 256 * sizeof(float)), "lut");
    cuda_check(cudaMemcpyAsync(d_in, in.data(), 256, cudaMemcpyHostToDevice, stream), "lut");
    sh::normalize_rgb_u8(d_in, d_out, 256, plan.normalize_scale, stream);
    std::vector<float> out(256);
    cuda_check(cudaMemcpyAsync(out.data(), d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost, stream),
               "lut");
    cuda_check(cudaStreamSynchronize(stream), "lut");
    cudaFree(d_in);
    cudaFree(d_out);
    std::vector<JsonValue> hex;
    for (float v : out) hex.push_back(JsonValue::make_string(hex_bytes(&v, sizeof v)));
    return JsonValue::make_array(std::move(hex));
}

struct Options {
    std::string config, sequence;
    int max_frames = 0;
    bool force_decoupled = false;
};

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail(a + " needs a value");
            return argv[++i];
        };
        if (a == "--config") o.config = next();
        else if (a == "--sequence") o.sequence = next();
        else if (a == "--max-frames") o.max_frames = std::stoi(next());
        else if (a == "--force-decoupled") o.force_decoupled = true;
        else fail("unknown argument " + a);
    }
    if (o.config.empty() || o.sequence.empty() || o.max_frames < 0) {
        fail("usage: saccade_ingest_probe --config JSON --sequence DIR [--max-frames N]");
    }
    return o;
}

int run(const Options& opt) {
    const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config_file(opt.config);
    const sh::IngestPlan plan = sh::plan_ingest(cfg);
    const std::filesystem::path seq_dir(opt.sequence);
    const sh::SequenceInput input = sh::read_sequence_input(seq_dir, opt.max_frames);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    sh::JpegDecoder decoder;
    decoder.force_decoupled_for_measurement(opt.force_decoupled);
    sh::IngestHost host(plan, input, decoder, stream);

    JsonValue pre = JsonValue::make_object();
    pre.set("format", JsonValue::make_string("saccade.native_ingest_stream/v1"));
    JsonValue nv = JsonValue::make_object();
    const auto& lib = decoder.library();
    nv.set("version", JsonValue::make_array({JsonValue::make_int(lib.major), JsonValue::make_int(lib.minor),
                                             JsonValue::make_int(lib.patch)}));
    nv.set("hardware_decode", JsonValue::make_bool(lib.hardware_decode));
    pre.set("nvjpeg", std::move(nv));
    pre.set("force_decoupled", JsonValue::make_bool(opt.force_decoupled));
    pre.set("sequence", JsonValue::make_string(seq_dir.filename().string()));
    pre.set("im_width", JsonValue::make_int(input.im_width));
    pre.set("im_height", JsonValue::make_int(input.im_height));
    pre.set("seq_length", JsonValue::make_int(input.seq_length));
    pre.set("listed", string_array(input.listed));
    pre.set("frames", string_array(input.frames));
    pre.set("normalize_lut_f32_hex", normalize_lut(plan, stream));
    write_record(pre);

    const std::size_t n = 3 * static_cast<std::size_t>(host.width()) * static_cast<std::size_t>(host.height());
    std::vector<std::uint8_t> rgb(n);
    std::vector<float> frame(n);
    for (int k = 1; k <= static_cast<int>(host.frame_count()); ++k) {
        const sh::IngestFrame f = host.ingest(k);
        cuda_check(cudaMemcpy(rgb.data(), host.decoded_rgb(), n, cudaMemcpyDeviceToHost), "D2H");
        cuda_check(cudaMemcpy(frame.data(), host.frame_chw(), n * sizeof(float), cudaMemcpyDeviceToHost),
                   "D2H");
        JsonValue rec = JsonValue::make_object();
        rec.set("frame", JsonValue::make_int(k));
        rec.set("file", JsonValue::make_string(f.file));
        rec.set("path", JsonValue::make_string(sh::decode_path_name(f.path)));
        rec.set("u8_bytes", JsonValue::make_int(static_cast<std::int64_t>(n)));
        rec.set("f32_bytes", JsonValue::make_int(static_cast<std::int64_t>(n * sizeof(float))));
        write_record(rec);
        write_bytes(rgb.data(), n);
        write_bytes(frame.data(), n * sizeof(float));
    }
    JsonValue end = JsonValue::make_object();
    end.set("end", JsonValue::make_bool(true));
    end.set("frames", JsonValue::make_int(static_cast<std::int64_t>(host.frame_count())));
    write_record(end);
    if (std::fflush(stdout) != 0) fail("flush stdout failed");
    cudaStreamDestroy(stream);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "saccade_ingest_probe: %s\n", e.what());
        return 2;
    }
}
