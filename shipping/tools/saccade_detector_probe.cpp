// saccade_detector_probe: run the native ingest (PR-7) and the native detector
// (#465 Phase B PR-8, U3b-2) over one sequence and stream what they produced,
// for the parity harness scripts/eval/diagnostics/native_detector_parity.py.
//
// Developer tool (developer_build_debug). It runs exactly the shipping path
// (saccade_shipping/ingest_*.hpp, detector_*.hpp): plans from the resolved
// config, the frozen PR-1L lineage and (optionally) the realization
// attestation; one JpegDecoder, one IngestHost, one DetectorHost; it compares
// nothing itself. No Python and no libtorch_python: the detector refuses to
// load if either is mapped, and the preamble records the check.
//
// Output on stdout, little-endian, format saccade.native_detector_stream/v1 --
// a sequence of records, each a uint64 byte length, that many bytes of JSON,
// then the binary payloads the JSON announces, in order:
//
//   preamble  {"format", "mode", "mutation", "sequence", "im_width",
//             "im_height", "seq_length", "frames": [names], "plan": {...},
//             "load": {...}, "nvjpeg": {...}, "python_libraries_mapped": []}
//   frame     {"frame": k, "file", "decode_path", "rows",
//             "payloads": [{"name", "dtype": "u8"|"f32"|"i32", "shape", "bytes"}]}
//             + the payloads. mode "stages": decoded_u8 [3, H, W], frame
//             [3, H, W], resized [1, 3, S, S], p3, p4, p5, cls_p3..p5,
//             reg_p3..p5, s2_raw [rows, 6], s2_scaled [rows, 6], boxes
//             [rows, 4], scores [rows], classes [rows]; mode "rows": the
//             last three only (the PR-5 detector boundary)
//   end       {"end": true, "frames": n}
//
// Exit 0 when every frame was processed; 2 on any error (message on stderr).
//
// Usage:
//   saccade_detector_probe --config configs/shipping/mamba_whole_graph.resolved.json
//       --lineage models/yolo/<stem>.lineage.json
//       [--attestation configs/shipping/mamba_head_realization.attestation.json]
//       --sequence datasets/MOT17/train/MOT17-09-SDP [--model-root .]
//       [--max-frames N] [--mode stages|rows]
//       [--mutation none|backbone_ulp|head_ulp|s2_threshold|s2_topk|s2_order|box_ulp]
//         (developer measurement: the harness's negative controls)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/detector_host.hpp"
#include "saccade_shipping/detector_plan.hpp"
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

JsonValue strings(const std::vector<std::string>& items) {
    std::vector<JsonValue> out;
    for (const auto& s : items) out.push_back(JsonValue::make_string(s));
    return JsonValue::make_array(std::move(out));
}

JsonValue ints(const std::vector<std::int64_t>& items) {
    std::vector<JsonValue> out;
    for (auto v : items) out.push_back(JsonValue::make_int(v));
    return JsonValue::make_array(std::move(out));
}

JsonValue binding(const sh::FileBinding& b) {
    JsonValue o = JsonValue::make_object();
    o.set("path", JsonValue::make_string(b.path));
    o.set("sha256", JsonValue::make_string(b.sha256));
    return o;
}

JsonValue runtime_json(const sh::HeadRuntimeRequirements& r) {
    JsonValue o = JsonValue::make_object();
    o.set("graph_executor_optimize", JsonValue::make_bool(r.graph_executor_optimize));
    o.set("cudnn_benchmark", JsonValue::make_bool(r.cudnn_benchmark));
    o.set("cudnn_allow_tf32", JsonValue::make_bool(r.cudnn_allow_tf32));
    o.set("matmul_allow_tf32", JsonValue::make_bool(r.matmul_allow_tf32));
    return o;
}

JsonValue plan_json(const sh::DetectorPlan& p) {
    JsonValue o = JsonValue::make_object();
    o.set("img_size", JsonValue::make_int(p.img_size));
    o.set("max_det", JsonValue::make_int(p.max_det));
    o.set("in_channels", ints({p.in_channels[0], p.in_channels[1], p.in_channels[2]}));
    o.set("num_classes", JsonValue::make_int(p.num_classes));
    o.set("anchors", JsonValue::make_int(p.anchors));
    o.set("backbone_engine", binding(p.backbone_engine));
    o.set("head_artifact", binding(p.head_artifact));
    JsonValue op = binding(p.op_library);
    op.set("lineage_sha256", JsonValue::make_string(p.op_library_lineage_sha256));
    op.set("from_attestation", JsonValue::make_bool(p.op_library_from_attestation));
    o.set("op_library", std::move(op));
    o.set("native_scan_calls", JsonValue::make_int(p.native_scan_calls));
    o.set("runtime_requirements", runtime_json(p.runtime));
    o.set("conf_thr_unused", JsonValue::make_float(p.conf_thr_unused));
    return o;
}

JsonValue load_json(const sh::HeadLoadReport& r) {
    JsonValue o = JsonValue::make_object();
    o.set("op_library_sha256", JsonValue::make_string(r.op_library_sha256));
    o.set("head_artifact_sha256", JsonValue::make_string(r.head_artifact_sha256));
    o.set("backbone_engine_sha256", JsonValue::make_string(r.backbone_engine_sha256));
    o.set("param_devices", strings(r.param_devices));
    o.set("constant_devices", strings(r.constant_devices));
    o.set("native_scan_calls", JsonValue::make_int(r.native_scan_calls));
    o.set("runtime_readback", runtime_json(r.runtime_readback));
    o.set("engine_io", strings(r.engine_io));
    o.set("trt_version", JsonValue::make_int(r.trt_version));
    o.set("torch_version", JsonValue::make_string(r.torch_version));
    return o;
}

struct Payload {
    std::string name, dtype;
    std::vector<std::int64_t> shape;
    const void* host = nullptr;    // either a host buffer ...
    const void* device = nullptr;  // ... or a device one (copied here)
    std::size_t bytes = 0;
};

struct Options {
    std::string config, lineage, attestation, sequence, model_root = ".", mode = "stages",
                                                                       mutation = "none";
    int max_frames = 0;
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
        else if (a == "--lineage") o.lineage = next();
        else if (a == "--attestation") o.attestation = next();
        else if (a == "--sequence") o.sequence = next();
        else if (a == "--model-root") o.model_root = next();
        else if (a == "--max-frames") o.max_frames = std::stoi(next());
        else if (a == "--mode") o.mode = next();
        else if (a == "--mutation") o.mutation = next();
        else fail("unknown argument " + a);
    }
    if (o.config.empty() || o.lineage.empty() || o.sequence.empty() || o.max_frames < 0 ||
        (o.mode != "stages" && o.mode != "rows")) {
        fail("usage: saccade_detector_probe --config JSON --lineage JSON [--attestation JSON] "
             "--sequence DIR [--model-root DIR] [--max-frames N] [--mode stages|rows] [--mutation M]");
    }
    return o;
}

int run(const Options& opt) {
    const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config_file(opt.config);
    const sh::IngestPlan ingest_plan = sh::plan_ingest(cfg);
    const sh::DetectorPlan plan = sh::plan_detector_files(cfg, {opt.lineage, opt.attestation});
    const sh::DetectorMutation mutation = sh::parse_detector_mutation(opt.mutation);
    const std::filesystem::path seq_dir(opt.sequence);
    const sh::SequenceInput input = sh::read_sequence_input(seq_dir, opt.max_frames);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    sh::JpegDecoder decoder;
    sh::IngestHost ingest(ingest_plan, input, decoder, stream);
    sh::DetectorHost detector(plan, opt.model_root, stream);
    detector.set_mutation_for_measurement(mutation);
    // set_whole_graph_img_dims(imHeight, imWidth) from seqinfo.ini.
    detector.set_image_dims(input.im_height, input.im_width);

    JsonValue pre = JsonValue::make_object();
    pre.set("format", JsonValue::make_string("saccade.native_detector_stream/v1"));
    pre.set("mode", JsonValue::make_string(opt.mode));
    pre.set("mutation", JsonValue::make_string(sh::detector_mutation_name(mutation)));
    pre.set("sequence", JsonValue::make_string(seq_dir.filename().string()));
    pre.set("im_width", JsonValue::make_int(input.im_width));
    pre.set("im_height", JsonValue::make_int(input.im_height));
    pre.set("seq_length", JsonValue::make_int(input.seq_length));
    pre.set("frames", strings(input.frames));
    pre.set("plan", plan_json(plan));
    pre.set("load", load_json(detector.load_report()));
    JsonValue nv = JsonValue::make_object();
    const auto& lib = decoder.library();
    nv.set("version", ints({lib.major, lib.minor, lib.patch}));
    nv.set("hardware_decode", JsonValue::make_bool(lib.hardware_decode));
    pre.set("nvjpeg", std::move(nv));
    pre.set("python_libraries_mapped", strings(sh::mapped_python_libraries()));
    write_record(pre);

    const std::int64_t H = input.im_height, W = input.im_width, S = plan.img_size;
    const std::size_t n_px = 3 * static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    std::vector<std::uint8_t> host;
    for (int k = 1; k <= static_cast<int>(ingest.frame_count()); ++k) {
        const sh::IngestFrame f = ingest.ingest(k);
        const sh::DetectionRows rows = detector.detect(ingest.frame_chw(), ingest.height(), ingest.width());
        const sh::DetectorStages& st = detector.stages();
        const std::int64_t n = static_cast<std::int64_t>(rows.size());
        std::vector<Payload> payloads;
        if (opt.mode == "stages") {
            payloads.push_back({"decoded_u8", "u8", {3, H, W}, nullptr, ingest.decoded_rgb(), n_px});
            payloads.push_back({"frame", "f32", {3, H, W}, nullptr, ingest.frame_chw(), n_px * 4});
            payloads.push_back({"resized", "f32", {1, 3, S, S}, nullptr, st.resized,
                                static_cast<std::size_t>(3 * S * S * 4)});
            const char* fn[3] = {"p3", "p4", "p5"};
            for (int i = 0; i < 3; ++i) {
                const auto& fs = plan.feature_shapes[static_cast<std::size_t>(i)];
                payloads.push_back({fn[i], "f32", {fs[0], fs[1], fs[2], fs[3]}, nullptr, st.features[i],
                                    static_cast<std::size_t>(fs[1]) * fs[2] * fs[3] * 4});
            }
            const char* hn[6] = {"cls_p3", "cls_p4", "cls_p5", "reg_p3", "reg_p4", "reg_p5"};
            for (int i = 0; i < 6; ++i) {
                const auto& fs = plan.feature_shapes[static_cast<std::size_t>(i % 3)];
                const std::int64_t c = i < 3 ? plan.num_classes : plan.reg_channels;
                payloads.push_back({hn[i], "f32", {1, c, fs[2], fs[3]}, nullptr, st.head[i],
                                    static_cast<std::size_t>(c) * fs[2] * fs[3] * 4});
            }
            payloads.push_back({"s2_raw", "f32", {st.rows, 6}, nullptr, st.s2_raw,
                                static_cast<std::size_t>(st.rows) * 6 * 4});
            payloads.push_back({"s2_scaled", "f32", {st.rows, 6}, nullptr, st.s2_scaled,
                                static_cast<std::size_t>(st.rows) * 6 * 4});
        }
        payloads.push_back({"boxes", "f32", {n, 4}, rows.boxes.data(), nullptr, rows.boxes.size() * 4});
        payloads.push_back({"scores", "f32", {n}, rows.scores.data(), nullptr, rows.scores.size() * 4});
        payloads.push_back({"classes", "i32", {n}, rows.classes.data(), nullptr, rows.classes.size() * 4});

        JsonValue rec = JsonValue::make_object();
        rec.set("frame", JsonValue::make_int(k));
        rec.set("file", JsonValue::make_string(f.file));
        rec.set("decode_path", JsonValue::make_string(sh::decode_path_name(f.path)));
        rec.set("rows", JsonValue::make_int(n));
        std::vector<JsonValue> pj;
        for (const Payload& p : payloads) {
            JsonValue o = JsonValue::make_object();
            o.set("name", JsonValue::make_string(p.name));
            o.set("dtype", JsonValue::make_string(p.dtype));
            o.set("shape", ints(p.shape));
            o.set("bytes", JsonValue::make_int(static_cast<std::int64_t>(p.bytes)));
            pj.push_back(std::move(o));
        }
        rec.set("payloads", JsonValue::make_array(std::move(pj)));
        write_record(rec);
        for (const Payload& p : payloads) {
            if (p.device != nullptr) {
                host.resize(p.bytes);
                cuda_check(cudaMemcpy(host.data(), p.device, p.bytes, cudaMemcpyDeviceToHost), "D2H");
                write_bytes(host.data(), p.bytes);
            } else {
                write_bytes(p.host, p.bytes);
            }
        }
    }
    JsonValue end = JsonValue::make_object();
    end.set("end", JsonValue::make_bool(true));
    end.set("frames", JsonValue::make_int(static_cast<std::int64_t>(ingest.frame_count())));
    write_record(end);
    if (std::fflush(stdout) != 0) fail("flush stdout failed");
    cudaStreamDestroy(stream);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // stdout carries the binary stream; anything a library prints through
    // std::cout (TRTEngine announces its load there) goes to stderr instead.
    std::cout.rdbuf(std::cerr.rdbuf());
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "saccade_detector_probe: %s\n", e.what());
        return 2;
    }
}
