// The sequence loop, trace and run report shared by the shipping entrypoint
// saccade_track (saccade_track.cpp) and the developer build
// saccade_track_measurement (saccade_track_measurement.cpp) (#465 Phase C
// PR-C2). Holds no measurement option and nothing that depends on
// SACCADE_SHIPPING_MEASUREMENT_HOOKS: what the developer build adds is in its
// own main and in the measurement variant of the runtime libraries.
//
// Interface options both binaries accept (shipping interface):
//   --config JSON --lineage JSON [--attestation JSON] [--model-root DIR]
//   --out DIR SEQUENCE_DIR...
//   --report JSON   what ran: run id, identity, plan bindings, load report,
//                   per-sequence counts (format saccade.native_track_report/v3)
//   --trace DIR     per sequence, DIR/<sequence>/detector.bin: each frame's
//                   detector rows in the PR-5 detector.bin record format
//                   (int32 frame, n, is_tiled; float32 boxes [n, 4]; float32
//                   scores [n]; int32 classes [n])
// Both write outputs only; neither changes what is computed.
//
// Completion (#536 CC-536-01-01, saccade_shipping/run_completion.hpp): after
// the arguments are parsed, the run id is stderr's first line; the run then
// takes <out> (exclusive lock; held by another run: exit 2, nothing changed),
// writes <out>/saccade_track.journal.json and removes this run's earlier
// report, <out>/<sequence>.txt and trace files before it reads the config.
// Each txt / trace is written to a temp file and renamed; the journal's
// state=complete, written after the report, is the only commit point. A rerun
// that fails therefore leaves no earlier output of its sequences: keep those
// with another --out. --report / --trace outside <out> are not under the lock:
// a report or trace counts only when the journal records its sha256.
//
// Gate A (#536 CC-536-01-02 N-T2, saccade_shipping/preflight.hpp): after
// <out> is taken and before any runtime is built (the first CUDA API call),
// the config and its plans, the lineage and attestation, every sequence's
// seqinfo.ini and img1 listing, the output directories and the three model
// files' sha256 are checked; a failure exits 2 with the journal failed and
// no CUDA call. Passing prints "<entrypoint>: preflight passed" to stderr.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/preflight.hpp"
#include "saccade_shipping/run_completion.hpp"
#include "saccade_shipping/serial_runtime.hpp"
#include "saccade_shipping/sha256.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace saccade::shipping::track {

// v3 (#536): v2 plus run_id and identity; v2 reports keep their meaning.
inline constexpr const char* kReportFormat = "saccade.native_track_report/v3";

[[noreturn]] inline void fail(const std::string& what) { throw std::runtime_error(what); }

struct Options {
    std::string config, lineage, attestation, model_root = ".", out, report, trace;
    std::vector<std::string> sequences;
};

// One argument of the shipping interface: true when `a` was consumed (`next`
// returns the option's value), false when it is not one of them.
inline bool parse_interface_arg(const std::string& a, const std::function<std::string()>& next,
                                Options& o) {
    if (a == "--config") o.config = next();
    else if (a == "--lineage") o.lineage = next();
    else if (a == "--attestation") o.attestation = next();
    else if (a == "--model-root") o.model_root = next();
    else if (a == "--out") o.out = next();
    else if (a == "--report") o.report = next();
    else if (a == "--trace") o.trace = next();
    else if (a.rfind("--", 0) == 0) return false;
    else o.sequences.push_back(a);
    return true;
}

inline bool interface_complete(const Options& o) {
    return !o.config.empty() && !o.lineage.empty() && !o.out.empty() && !o.sequences.empty();
}

// Sequence name = the directory's name, as the oracle names <seq>.txt.
inline std::string sequence_name(const std::string& dir) {
    std::filesystem::path p = std::filesystem::path(dir).lexically_normal();
    if (p.filename().empty()) p = p.parent_path();  // "MOT17-09-SDP/"
    return p.filename().string();
}

inline void require_distinct_sequences(const Options& o) {
    std::set<std::string> names;
    for (const std::string& s : o.sequences) {
        if (!names.insert(sequence_name(s)).second) fail("sequence " + sequence_name(s) + " given twice");
    }
}

// Right after the arguments are parsed: the run id (stderr's first line),
// then the argv checks, then <out> is taken (RunCompletion). `completion`
// stays empty when this throws, so the caller's error path writes no journal.
inline void begin_run(const Options& o, const char* entrypoint, std::optional<RunCompletion>& completion) {
    const std::string run_id = new_run_id();
    std::cerr << entrypoint << ": run_id " << run_id << std::endl;
    require_distinct_sequences(o);
    RunOutputs outputs{o.out, o.report, o.trace, {}};
    for (const std::string& s : o.sequences) outputs.sequences.push_back(sequence_name(s));
    completion.emplace(run_id, entrypoint, std::move(outputs));
}

// Gate A, right after begin_run; the runtime is built from what it returns.
// `max_frames` and `serial_requested` are the developer build's (0 / false in
// saccade_track).
inline PreflightResult preflight(const Options& o, const char* entrypoint, const RunCompletion& completion,
                                 int max_frames, bool serial_requested) {
    PreflightResult p = run_preflight(
        PreflightInputs{o.config, o.lineage, o.attestation, o.model_root, o.sequences, max_frames,
                        serial_requested},
        completion);
    std::cerr << entrypoint << ": preflight passed" << std::endl;
    return p;
}

class DetectorTrace : public FrameObserver {
public:
    explicit DetectorTrace(const std::filesystem::path& file) : f_(file, std::ios::binary) {
        if (!f_) fail("cannot open " + file.string());
    }
    void on_frame(const FrameTrace& t) override {
        const DetectionRows& r = *t.detections;
        const std::int32_t head[3] = {t.frame, static_cast<std::int32_t>(r.size()), 0};
        put(head, sizeof head);
        put(r.boxes.data(), r.boxes.size() * sizeof(float));
        put(r.scores.data(), r.scores.size() * sizeof(float));
        put(r.classes.data(), r.classes.size() * sizeof(std::int32_t));
        ++records_;
    }
    void close() {
        f_.close();
        if (!f_) fail("trace write failed");
    }
    std::int64_t records() const { return records_; }

private:
    void put(const void* p, std::size_t n) {
        if (n > 0) f_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    }
    std::ofstream f_;
    std::int64_t records_ = 0;
};

inline JsonValue strings(const std::vector<std::string>& items) {
    std::vector<JsonValue> out;
    for (const auto& s : items) out.push_back(JsonValue::make_string(s));
    return JsonValue::make_array(std::move(out));
}

inline JsonValue binding(const FileBinding& b) {
    JsonValue o = JsonValue::make_object();
    o.set("path", JsonValue::make_string(b.path));
    o.set("sha256", JsonValue::make_string(b.sha256));
    return o;
}

inline JsonValue runtime_json(const HeadRuntimeRequirements& r) {
    JsonValue o = JsonValue::make_object();
    o.set("graph_executor_optimize", JsonValue::make_bool(r.graph_executor_optimize));
    o.set("cudnn_benchmark", JsonValue::make_bool(r.cudnn_benchmark));
    o.set("cudnn_allow_tf32", JsonValue::make_bool(r.cudnn_allow_tf32));
    o.set("matmul_allow_tf32", JsonValue::make_bool(r.matmul_allow_tf32));
    return o;
}

template <class Runtime>
JsonValue load_json(const Runtime& rt) {
    const DetectorPlan& p = rt.detector_plan();
    const HeadLoadReport& r = rt.load_report();
    JsonValue plan = JsonValue::make_object();
    plan.set("backbone_engine", binding(p.backbone_engine));
    plan.set("head_artifact", binding(p.head_artifact));
    JsonValue op = binding(p.op_library);
    op.set("lineage_sha256", JsonValue::make_string(p.op_library_lineage_sha256));
    op.set("from_attestation", JsonValue::make_bool(p.op_library_from_attestation));
    plan.set("op_library", std::move(op));
    JsonValue load = JsonValue::make_object();
    load.set("op_library_sha256", JsonValue::make_string(r.op_library_sha256));
    load.set("head_artifact_sha256", JsonValue::make_string(r.head_artifact_sha256));
    load.set("backbone_engine_sha256", JsonValue::make_string(r.backbone_engine_sha256));
    load.set("param_devices", strings(r.param_devices));
    load.set("constant_devices", strings(r.constant_devices));
    load.set("native_scan_calls", JsonValue::make_int(r.native_scan_calls));
    load.set("runtime_readback", runtime_json(r.runtime_readback));
    load.set("engine_io", strings(r.engine_io));
    load.set("trt_version", JsonValue::make_int(r.trt_version));
    load.set("torch_version", JsonValue::make_string(r.torch_version));
    JsonValue o = JsonValue::make_object();
    o.set("plan", std::move(plan));
    o.set("load", std::move(load));
    return o;
}

inline JsonValue stats_json(const SequenceRunStats& s, const std::string& txt_sha256,
                            const std::string& txt_path, std::int64_t trace_records) {
    JsonValue o = JsonValue::make_object();
    o.set("im_width", JsonValue::make_int(s.im_width));
    o.set("im_height", JsonValue::make_int(s.im_height));
    o.set("seq_length", JsonValue::make_int(s.seq_length));
    o.set("frames", JsonValue::make_int(s.frames));
    o.set("tracker_updates", JsonValue::make_int(s.tracker_updates));
    o.set("skipped_empty_frames", JsonValue::make_int(s.skipped_empty));
    o.set("pre_roll_updates", JsonValue::make_int(s.pre_roll_updates));
    o.set("hardware_decodes", JsonValue::make_int(s.hardware_decodes));
    o.set("decoupled_decodes", JsonValue::make_int(s.decoupled_decodes));
    o.set("track_ids", JsonValue::make_int(static_cast<std::int64_t>(s.track_ids)));
    o.set("lines", JsonValue::make_int(static_cast<std::int64_t>(s.lines)));
    JsonValue interp = JsonValue::make_object();
    interp.set("tracks_interpolated", JsonValue::make_int(s.interpolation.tracks_interpolated));
    interp.set("gaps_filled", JsonValue::make_int(s.interpolation.gaps_filled));
    interp.set("frames_added", JsonValue::make_int(s.interpolation.frames_added));
    o.set("interpolation", std::move(interp));
    o.set("txt", JsonValue::make_string(txt_path));
    o.set("txt_sha256", JsonValue::make_string(txt_sha256));
    o.set("trace_records", trace_records < 0 ? JsonValue::make_null() : JsonValue::make_int(trace_records));
    const ScheduleStats& g = s.schedule;
    JsonValue graphs = JsonValue::make_object();
    graphs.set("detector_captures", JsonValue::make_int(g.detector_captures));
    graphs.set("detector_warmup_runs", JsonValue::make_int(g.detector_warmup_runs));
    graphs.set("detector_replays", JsonValue::make_int(g.detector_replays));
    graphs.set("nms_captures", JsonValue::make_int(g.post.nms_captures));
    graphs.set("nms_replays", JsonValue::make_int(g.post.nms_replays));
    graphs.set("gmc_captures", JsonValue::make_int(g.post.gmc_captures));
    graphs.set("gmc_replays", JsonValue::make_int(g.post.gmc_replays));
    graphs.set("tracker_captures", JsonValue::make_int(g.post.tracker_captures));
    graphs.set("tracker_replays", JsonValue::make_int(g.post.tracker_replays));
    o.set("graphs", std::move(graphs));
    o.set("loop_seconds", JsonValue::make_float(g.loop_seconds));
    return o;
}

// Every sequence in order -> <out>/<sequence>.txt (and the trace); then the
// report; each committed through `completion` (begin_run). `entrypoint`
// names the binary in the report; `measurement` (the developer build only) is
// added to it as is.
template <class Runtime>
int run_sequences(Runtime& rt, const Options& opt, RunCompletion& completion, const char* entrypoint,
                  const char* schedule, int max_frames, const JsonValue* measurement) {
    JsonValue seqs = JsonValue::make_object();
    std::vector<std::string> order;
    auto sequence = [&](std::size_t i, const std::filesystem::path& trace_tmp) {
        const std::string& dir = opt.sequences[i];
        const std::string name = sequence_name(dir);
        std::unique_ptr<DetectorTrace> trace;
        if (!trace_tmp.empty()) trace = std::make_unique<DetectorTrace>(trace_tmp);
        const SequenceRunResult res = rt.run_sequence(dir, max_frames, trace.get());
        std::string text = join_mot_lines(res.lines);
        const std::filesystem::path txt = std::filesystem::path(opt.out) / (name + ".txt");
        std::int64_t records = -1;
        if (trace) {
            trace->close();
            records = trace->records();
        }
        seqs.set(name, stats_json(res.stats, sha256_hex(text.data(), text.size()), txt.string(), records));
        order.push_back(name);
        std::cerr << "[" << entrypoint << "] " << name << ": " << res.stats.frames << " frames, "
                  << res.stats.lines << " lines, " << res.stats.track_ids << " ids\n";
        return text;
    };
    auto report = [&]() {
        JsonValue rep = JsonValue::make_object();
        rep.set("format", JsonValue::make_string(kReportFormat));
        rep.set("run_id", JsonValue::make_string(completion.run_id()));
        rep.set("identity", unverified_identity());
        rep.set("entrypoint", JsonValue::make_string(entrypoint));
        rep.set("config", JsonValue::make_string(opt.config));
        rep.set("lineage", JsonValue::make_string(opt.lineage));
        rep.set("attestation", JsonValue::make_string(opt.attestation));
        rep.set("schedule", JsonValue::make_string(schedule));
        if (measurement != nullptr) rep.set("measurement", *measurement);
        rep.set("detector", load_json(rt));
        JsonValue nv = JsonValue::make_object();
        const NvjpegLibraryInfo& lib = rt.nvjpeg();
        nv.set("version", JsonValue::make_string(std::to_string(lib.major) + "." +
                                                 std::to_string(lib.minor) + "." +
                                                 std::to_string(lib.patch)));
        nv.set("hardware_decode", JsonValue::make_bool(lib.hardware_decode));
        rep.set("nvjpeg", std::move(nv));
        rep.set("python_libraries_mapped", strings(mapped_python_libraries()));
        rep.set("sequence_order", strings(order));
        rep.set("sequences", std::move(seqs));
        return dump_python_json(rep) + "\n";
    };
    completion.run(sequence, report);
    return 0;
}

}  // namespace saccade::shipping::track
