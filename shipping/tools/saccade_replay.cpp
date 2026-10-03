// saccade_replay: replay a post-detector dump through the native U3a host and
// compare every stage with the Python serial reference (#465 Phase B PR-5),
// then turn the host's tracker output into the sequence's MOT txt (PR-6).
//
// Developer tool (developer_build_debug); the host it drives is
// saccade_shipping/post_detector_host.hpp. Input is a dump written by
// scripts/eval/diagnostics/dump_post_detector_replay.py (format
// saccade.post_detector_replay/v1, described there). For each sequence the
// host consumes only the dumped detector output and GMC input frames; the
// other streams are the reference its stages are compared with, frame by
// frame and bit for bit:
//
//   post_nms        main NMS + private candidates  vs post_nms.bin
//   tracker_input   rows after the detection filters vs tracker_in.bin rows
//   gmc             warp the tracker receives       vs tracker_in.bin warp
//   tracker_output  tracker rows (boxes, scores, local ids, classes)
//                                                   vs tracker_out.bin
//
//   mot_txt         the sequence's MOT lines (sequence_output.hpp: per-sequence
//                   ids, fast-emit lines, interpolation) vs the oracle's
//                   eval/<seq>.txt, byte for byte after relabeling: the oracle
//                   numbers ids run-globally (GlobalTrackIdMapper), so its ids
//                   for a sequence are the per-sequence ids plus the number of
//                   ids earlier sequences used; the offset is read from its
//                   _global_id_map.txt, which must list this sequence's ids as
//                   one contiguous block. (The reference files' integrity is
//                   `dump_post_detector_replay.py --verify`'s, against the
//                   manifest's mot_reference hashes, as for the streams.)
//
// Every per-update stream (post_nms, tracker_in, tracker_out, and gmc when GMC
// runs) must have exactly one record per replayed tracker update.
//
// The run is chained (the host's own outputs feed its next stages), so after a
// first divergence later frames may differ for that reason alone; the report
// keeps each stage's first divergent frame. Exit 0: every stage equal on every
// frame and every MOT txt identical; 1: something differs; 2: error.
//
// Usage:
//   saccade_replay --config configs/shipping/mamba_whole_graph.resolved.json
//       --dump <dump dir> --report <report.json> [--sequences A,B]
//       [--pre-roll N]   (developer measurement: override the tracker pre-roll)
//       [--mot-out DIR]  (write the native <seq>.txt files there; created if absent)
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/native_build.hpp"
#include "saccade_shipping/post_detector_host.hpp"
#include "saccade_shipping/resolved_config.hpp"
#include "saccade_shipping/sequence_output.hpp"
#include "saccade_shipping/strict_json.hpp"
#include "tracking/pipeline.hpp"

namespace sh = saccade::shipping;

namespace {

constexpr const char* kFormat = "saccade.post_detector_replay/v1";

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

const sh::JsonValue& field(const sh::JsonValue& obj, const char* key) {
    const sh::JsonValue* v = obj.find(key);
    if (v == nullptr) fail(std::string("dump metadata lacks \"") + key + "\"");
    return *v;
}

// ─── dump v1 reader ──────────────────────────────────────────────────────

class Reader {
public:
    explicit Reader(const std::string& path) : data_(read_file(path)), path_(path) {}
    bool done() const { return pos_ == data_.size(); }
    std::int32_t i32() {
        std::int32_t v;
        take(&v, sizeof v);
        return v;
    }
    template <class T>
    std::vector<T> array(std::size_t n) {
        std::vector<T> v(n);
        if (n > 0) take(v.data(), n * sizeof(T));
        return v;
    }

private:
    void take(void* dst, std::size_t bytes) {
        if (data_.size() - pos_ < bytes) fail(path_ + ": truncated record");
        std::memcpy(dst, data_.data() + pos_, bytes);
        pos_ += bytes;
    }
    std::string data_;
    std::string path_;
    std::size_t pos_ = 0;
};

sh::DetectionRows read_rows(Reader& r, std::size_t n) {
    sh::DetectionRows rows;
    rows.boxes = r.array<float>(n * 4);
    rows.scores = r.array<float>(n);
    rows.classes = r.array<std::int32_t>(n);
    return rows;
}

struct DetFrame {
    int frame;
    bool is_tiled;
    sh::DetectionRows rows;
};
struct RowsFrame {
    int frame;
    sh::DetectionRows rows;
};
struct GmcFrame {
    int frame, frame_index;
    std::array<float, 6> warp;
};
struct TrackerInFrame {
    int frame;
    std::array<float, 6> gmc;
    sh::DetectionRows rows;
};
struct TrackerOutFrame {
    int frame;
    sh::TrackerRows rows;
};

struct SequenceDump {
    std::string name, dir;
    int width = 0, height = 0;
    bool frames_stored = false;
    std::vector<DetFrame> detector;
    std::vector<RowsFrame> post_nms;
    std::vector<GmcFrame> gmc;
    std::vector<TrackerInFrame> tracker_in;
    std::vector<TrackerOutFrame> tracker_out;
};

std::array<float, 6> read6(Reader& r) {
    auto v = r.array<float>(6);
    std::array<float, 6> a{};
    std::copy(v.begin(), v.end(), a.begin());
    return a;
}

SequenceDump load_sequence(const std::string& dir, const std::string& name) {
    SequenceDump d;
    d.name = name;
    d.dir = dir + "/" + name;
    const sh::JsonValue meta = sh::parse_strict_json(read_file(d.dir + "/meta.json"));
    if (field(meta, "format").string != kFormat) fail(name + ": unknown dump format");
    d.width = static_cast<int>(field(meta, "width").integer);
    d.height = static_cast<int>(field(meta, "height").integer);
    d.frames_stored = field(field(meta, "frames_u8"), "stored").boolean;
    const sh::JsonValue& files = field(meta, "files");

    Reader det(d.dir + "/detector.bin");
    while (!det.done()) {
        DetFrame f;
        f.frame = det.i32();
        const int n = det.i32();
        f.is_tiled = det.i32() != 0;
        f.rows = read_rows(det, static_cast<std::size_t>(n));
        d.detector.push_back(std::move(f));
    }
    Reader post(d.dir + "/post_nms.bin");
    while (!post.done()) {
        RowsFrame f;
        f.frame = post.i32();
        f.rows = read_rows(post, static_cast<std::size_t>(post.i32()));
        d.post_nms.push_back(std::move(f));
    }
    Reader gmc(d.dir + "/gmc.bin");
    while (!gmc.done()) {
        GmcFrame f;
        f.frame = gmc.i32();
        f.frame_index = gmc.i32();
        if (gmc.i32() == 0) fail(name + ": GMC returned no warp");
        f.warp = read6(gmc);
        d.gmc.push_back(f);
    }
    Reader tin(d.dir + "/tracker_in.bin");
    while (!tin.done()) {
        TrackerInFrame f;
        f.frame = tin.i32();
        const int n = tin.i32();
        tin.i32();  // has_gmc
        f.gmc = read6(tin);
        f.rows = read_rows(tin, static_cast<std::size_t>(n));
        d.tracker_in.push_back(std::move(f));
    }
    Reader tout(d.dir + "/tracker_out.bin");
    while (!tout.done()) {
        TrackerOutFrame f;
        f.frame = tout.i32();
        const auto c = static_cast<std::size_t>(tout.i32());
        f.rows.boxes = tout.array<float>(c * 4);
        f.rows.scores = tout.array<float>(c);
        f.rows.ids = tout.array<std::int32_t>(c);
        f.rows.classes = tout.array<std::int32_t>(c);
        d.tracker_out.push_back(std::move(f));
    }
    const std::pair<const char*, std::size_t> counts[] = {
        {"detector.bin", d.detector.size()},     {"post_nms.bin", d.post_nms.size()},
        {"gmc.bin", d.gmc.size()},               {"tracker_in.bin", d.tracker_in.size()},
        {"tracker_out.bin", d.tracker_out.size()}};
    for (const auto& [file, n] : counts) {
        if (static_cast<std::size_t>(field(field(files, file), "records").integer) != n) {
            fail(name + "/" + file + ": record count differs from meta.json");
        }
    }
    return d;
}

// ─── comparison ──────────────────────────────────────────────────────────

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return m;
}

struct StageStats {
    int compared = 0, equal = 0;
    int first_divergent_frame = -1;
    std::string first_divergence;
    int count_mismatch_frames = 0;
    double max_abs_diff = 0.0;  // over frames with equal counts (report-only)

    void add(int frame, bool eq, bool count_eq, double diff, const std::string& detail) {
        ++compared;
        if (eq) {
            ++equal;
            return;
        }
        if (!count_eq) ++count_mismatch_frames;
        if (count_eq) max_abs_diff = std::max(max_abs_diff, diff);
        if (first_divergent_frame < 0) {
            first_divergent_frame = frame;
            first_divergence = detail;
        }
    }
    bool exact() const { return compared == equal; }
    sh::JsonValue json() const {
        auto o = sh::JsonValue::make_object();
        o.set("frames_compared", sh::JsonValue::make_int(compared));
        o.set("frames_equal", sh::JsonValue::make_int(equal));
        o.set("count_mismatch_frames", sh::JsonValue::make_int(count_mismatch_frames));
        o.set("max_abs_diff_equal_count", sh::JsonValue::make_float(max_abs_diff));
        o.set("first_divergent_frame", first_divergent_frame < 0
                                            ? sh::JsonValue::make_null()
                                            : sh::JsonValue::make_int(first_divergent_frame));
        o.set("first_divergence", first_divergence.empty()
                                      ? sh::JsonValue::make_null()
                                      : sh::JsonValue::make_string(first_divergence));
        return o;
    }
};

void compare_rows(StageStats& s, int frame, const sh::DetectionRows& got,
                  const sh::DetectionRows& want) {
    const bool count_eq = got.size() == want.size();
    const bool eq = count_eq && same_bits(got.boxes, want.boxes) &&
                    same_bits(got.scores, want.scores) && same_bits(got.classes, want.classes);
    std::string detail;
    if (!eq) {
        detail = "rows " + std::to_string(got.size()) + " vs " + std::to_string(want.size());
    }
    const double diff = count_eq ? std::max(max_abs_diff(got.boxes, want.boxes),
                                            max_abs_diff(got.scores, want.scores))
                                 : 0.0;
    s.add(frame, eq, count_eq, diff, detail);
}

void compare_tracker(StageStats& s, int frame, const sh::TrackerRows& got,
                     const sh::TrackerRows& want) {
    const bool count_eq = got.size() == want.size();
    const bool eq = count_eq && same_bits(got.boxes, want.boxes) &&
                    same_bits(got.scores, want.scores) && same_bits(got.ids, want.ids) &&
                    same_bits(got.classes, want.classes);
    std::string detail;
    if (!eq) {
        detail = "rows " + std::to_string(got.size()) + " vs " + std::to_string(want.size()) +
                 (count_eq && !same_bits(got.ids, want.ids) ? ", ids differ" : "");
    }
    const double diff = count_eq ? std::max(max_abs_diff(got.boxes, want.boxes),
                                            max_abs_diff(got.scores, want.scores))
                                 : 0.0;
    s.add(frame, eq, count_eq, diff, detail);
}

// u8 -> float32 as torch.div(u8, 255.0) computes it (a multiply by the
// float32 reciprocal on CUDA); checked against the dump's 256-entry table.
std::vector<float> frame_lut(const sh::JsonValue& manifest) {
    const sh::JsonValue& hex = field(manifest, "frame_lut_f32_hex");
    if (hex.array.size() != 256) fail("manifest frame_lut_f32_hex must have 256 entries");
    const float inv = 1.0f / 255.0f;
    std::vector<float> lut(256);
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<float>(i) * inv;
        std::uint32_t want = 0;
        const std::string& h = hex.array[i].string;  // little-endian bytes
        for (int k = 3; k >= 0; --k) {
            want = (want << 8) | static_cast<std::uint32_t>(std::stoul(h.substr(k * 2, 2), nullptr, 16));
        }
        std::uint32_t got;
        std::memcpy(&got, &lut[i], sizeof got);
        if (got != want) fail("u8 -> float32 conversion differs from torch at " + std::to_string(i));
    }
    return lut;
}

struct Options {
    std::string config, dump, report, mot_out;
    std::vector<std::string> sequences;
    std::optional<int> pre_roll;
};

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) fail(a + " needs a value");
            return argv[++i];
        };
        if (a == "--config") o.config = next();
        else if (a == "--dump") o.dump = next();
        else if (a == "--report") o.report = next();
        else if (a == "--sequences") o.sequences = split_csv(next());
        else if (a == "--pre-roll") o.pre_roll = std::stoi(next());
        else if (a == "--mot-out") o.mot_out = next();
        else fail("unknown argument " + a);
    }
    if (o.config.empty() || o.dump.empty() || o.report.empty()) {
        fail("usage: saccade_replay --config JSON --dump DIR --report JSON "
             "[--sequences A,B] [--pre-roll N] [--mot-out DIR]");
    }
    return o;
}

// ─── MOT txt (PR-6) ──────────────────────────────────────────────────────

struct MotReference {
    std::string text;         // the oracle's eval/<seq>.txt
    std::int64_t offset = 0;  // ids the oracle gave earlier sequences
    std::size_t ids = 0;      // this sequence's ids in _global_id_map.txt
};

MotReference load_mot_reference(const std::string& dump, const sh::JsonValue& manifest,
                                const std::string& seq) {
    const sh::JsonValue* ref = manifest.find("mot_reference");
    if (ref == nullptr) fail("dump manifest has no mot_reference (dumped before PR-6); re-dump");
    const sh::JsonValue* entry = field(*ref, "sequences").find(seq);
    if (entry == nullptr) fail(seq + ": no MOT reference in the dump manifest");
    MotReference r;
    r.text = read_file(dump + "/" + field(*entry, "path").string);

    // "<seq>\tlocal_id=<L>\tglobal_id=<G>" (GlobalTrackIdMapper.dump_lines)
    std::istringstream map(read_file(dump + "/" + field(field(*ref, "global_id_map"), "path").string));
    const std::string prefix = seq + "\tlocal_id=";
    const std::string key = "\tglobal_id=";
    std::vector<std::int64_t> ids;
    std::string line;
    while (std::getline(map, line)) {
        if (line.rfind(prefix, 0) != 0) continue;
        const std::size_t g = line.find(key);
        if (g == std::string::npos) fail("_global_id_map.txt: malformed line \"" + line + "\"");
        ids.push_back(std::stoll(line.substr(g + key.size())));
    }
    std::sort(ids.begin(), ids.end());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != ids.front() + static_cast<std::int64_t>(i)) {
            fail(seq + ": its run-global ids are not one contiguous block");
        }
    }
    r.ids = ids.size();
    r.offset = ids.empty() ? 0 : ids.front() - 1;
    return r;
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> out;
    if (text.empty()) return out;
    std::size_t start = 0;
    while (true) {
        const std::size_t end = text.find('\n', start);
        out.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

// The oracle's lines with the id field (2nd) shifted back by `offset`.
std::vector<std::string> relabel(const std::vector<std::string>& lines, std::int64_t offset) {
    std::vector<std::string> out;
    out.reserve(lines.size());
    for (const std::string& l : lines) {
        const std::size_t c1 = l.find(',');
        const std::size_t c2 = c1 == std::string::npos ? c1 : l.find(',', c1 + 1);
        if (c2 == std::string::npos) fail("reference MOT line without an id field: \"" + l + "\"");
        const std::int64_t id = std::stoll(l.substr(c1 + 1, c2 - c1 - 1));
        out.push_back(l.substr(0, c1 + 1) + std::to_string(id - offset) + l.substr(c2));
    }
    return out;
}

sh::JsonValue replay_sequence(const sh::ResolvedShippingConfig& cfg, const SequenceDump& d,
                              const std::vector<float>& lut, saccade::PerceptionPipeline& pipeline,
                              cudaStream_t stream, std::optional<int> pre_roll,
                              const MotReference& mot_ref, const std::string& mot_out,
                              bool& exact) {
    sh::PostDetectorHost host(cfg, sh::SequenceGeometry{d.width, d.height}, pipeline, stream);
    sh::SequenceOutput output(sh::plan_sequence_output(cfg));
    if (!output.plan().write_output) fail("the config writes no MOT output (steps.tail.write_output)");
    if (pre_roll) host.set_pre_roll_for_measurement(*pre_roll);
    if (host.plan().gmc && !d.frames_stored) fail(d.name + ": GMC needs stored frames");

    const std::size_t plane = static_cast<std::size_t>(d.width) * d.height;
    std::ifstream frames;
    if (d.frames_stored) {
        frames.open(d.dir + "/frames.u8", std::ios::binary);
        if (!frames) fail(d.name + ": cannot open frames.u8");
    }
    std::vector<std::uint8_t> u8(plane * 3);
    std::vector<float> f32(plane * 3);
    float* d_frame = nullptr;
    float *d_boxes = nullptr, *d_scores = nullptr;
    std::int32_t* d_classes = nullptr;
    std::size_t det_cap = 0;
    cuda_check(cudaMalloc(&d_frame, f32.size() * sizeof(float)), "frame buffer");

    StageStats post_nms, tracker_input, gmc, tracker_output;
    std::size_t k = 0;  // index into the per-update streams
    int updates = 0, skipped = 0;
    for (const DetFrame& df : d.detector) {
        const std::size_t n = df.rows.size();
        if (n > det_cap) {
            cudaFree(d_boxes);
            cudaFree(d_scores);
            cudaFree(d_classes);
            det_cap = n;
            cuda_check(cudaMalloc(&d_boxes, n * 4 * sizeof(float)), "detections");
            cuda_check(cudaMalloc(&d_scores, n * sizeof(float)), "detections");
            cuda_check(cudaMalloc(&d_classes, n * sizeof(std::int32_t)), "detections");
        }
        if (n > 0) {
            cuda_check(cudaMemcpy(d_boxes, df.rows.boxes.data(), n * 4 * sizeof(float),
                             cudaMemcpyHostToDevice), "detections");
            cuda_check(cudaMemcpy(d_scores, df.rows.scores.data(), n * sizeof(float),
                             cudaMemcpyHostToDevice), "detections");
            cuda_check(cudaMemcpy(d_classes, df.rows.classes.data(), n * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice), "detections");
        }
        const float* frame_ptr = nullptr;
        if (n > 0 && host.plan().gmc) {
            if (k >= d.gmc.size() || d.gmc[k].frame != df.frame) {
                fail(d.name + ": gmc.bin out of step with detector.bin at frame " +
                     std::to_string(df.frame));
            }
            frames.seekg(static_cast<std::streamoff>(d.gmc[k].frame_index) *
                         static_cast<std::streamoff>(u8.size()));
            frames.read(reinterpret_cast<char*>(u8.data()), static_cast<std::streamsize>(u8.size()));
            if (!frames) fail(d.name + ": frames.u8 truncated");
            for (std::size_t i = 0; i < u8.size(); ++i) f32[i] = lut[u8[i]];
            cuda_check(cudaMemcpy(d_frame, f32.data(), f32.size() * sizeof(float),
                             cudaMemcpyHostToDevice), "frame");
            frame_ptr = d_frame;
        }
        const sh::FrameResult r = host.process(
            sh::DeviceDetections{d_boxes, d_scores, d_classes, static_cast<int>(n), df.is_tiled},
            frame_ptr);
        if (!r.updated) {
            ++skipped;
            continue;
        }
        if (k >= d.post_nms.size() || k >= d.tracker_in.size() || k >= d.tracker_out.size() ||
            d.post_nms[k].frame != df.frame || d.tracker_in[k].frame != df.frame ||
            d.tracker_out[k].frame != df.frame) {
            fail(d.name + ": reference streams out of step at frame " + std::to_string(df.frame));
        }
        compare_rows(post_nms, df.frame, r.post_nms, d.post_nms[k].rows);
        compare_rows(tracker_input, df.frame, r.tracker_input, d.tracker_in[k].rows);
        {
            const std::vector<float> got(r.gmc_warp.begin(), r.gmc_warp.end());
            const std::vector<float> want(d.tracker_in[k].gmc.begin(), d.tracker_in[k].gmc.end());
            gmc.add(df.frame, same_bits(got, want), true, max_abs_diff(got, want),
                    same_bits(got, want) ? "" : "warp differs");
        }
        compare_tracker(tracker_output, df.frame, r.tracker_output, d.tracker_out[k].rows);
        output.add_frame(df.frame, r.tracker_output.boxes.data(), r.tracker_output.scores.data(),
                         r.tracker_output.ids.data(), r.tracker_output.size());
        ++k;
        ++updates;
    }
    // One record per replayed tracker update in every per-update stream.
    const std::size_t want_gmc = host.plan().gmc ? k : 0;
    if (d.post_nms.size() != k || d.tracker_in.size() != k || d.tracker_out.size() != k ||
        d.gmc.size() != want_gmc) {
        fail(d.name + ": reference streams (post_nms " + std::to_string(d.post_nms.size()) +
             ", tracker_in " + std::to_string(d.tracker_in.size()) + ", tracker_out " +
             std::to_string(d.tracker_out.size()) + ", gmc " + std::to_string(d.gmc.size()) +
             ") do not match " + std::to_string(k) + " replayed tracker updates");
    }
    cudaFree(d_frame);
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_classes);

    // MOT txt: the sequence tail, then the oracle's lines relabeled.
    const std::size_t track_ids = output.track_ids();
    sh::InterpolationStats interp;
    const std::vector<std::string> lines = output.finish(&interp);
    const std::string text = sh::join_mot_lines(lines);
    std::string written;
    if (!mot_out.empty()) {
        written = mot_out + "/" + d.name + ".txt";
        std::ofstream f(written, std::ios::binary);
        f << text;
        if (!f) fail("cannot write " + written);
    }
    const std::vector<std::string> want = relabel(split_lines(mot_ref.text), mot_ref.offset);
    const bool mot_equal = sh::join_mot_lines(want) == text;
    std::optional<std::size_t> first_diff;
    if (!mot_equal) {
        std::size_t i = 0;
        while (i < lines.size() && i < want.size() && lines[i] == want[i]) ++i;
        first_diff = i;
    }
    auto mot = sh::JsonValue::make_object();
    mot.set("lines", sh::JsonValue::make_int(static_cast<std::int64_t>(lines.size())));
    mot.set("reference_lines", sh::JsonValue::make_int(static_cast<std::int64_t>(want.size())));
    mot.set("bytes", sh::JsonValue::make_int(static_cast<std::int64_t>(text.size())));
    mot.set("track_ids", sh::JsonValue::make_int(static_cast<std::int64_t>(track_ids)));
    mot.set("reference_track_ids", sh::JsonValue::make_int(static_cast<std::int64_t>(mot_ref.ids)));
    mot.set("reference_id_offset", sh::JsonValue::make_int(mot_ref.offset));
    auto ist = sh::JsonValue::make_object();
    ist.set("tracks_interpolated", sh::JsonValue::make_int(interp.tracks_interpolated));
    ist.set("gaps_filled", sh::JsonValue::make_int(interp.gaps_filled));
    ist.set("frames_added", sh::JsonValue::make_int(interp.frames_added));
    mot.set("interpolation", ist);
    mot.set("byte_identical_after_relabel", sh::JsonValue::make_bool(mot_equal));
    mot.set("first_differing_line", first_diff ? sh::JsonValue::make_int(static_cast<std::int64_t>(*first_diff))
                                                : sh::JsonValue::make_null());
    mot.set("written", written.empty() ? sh::JsonValue::make_null() : sh::JsonValue::make_string(written));

    exact = post_nms.exact() && tracker_input.exact() && gmc.exact() && tracker_output.exact() &&
            mot_equal && track_ids == mot_ref.ids;
    auto o = sh::JsonValue::make_object();
    o.set("frames", sh::JsonValue::make_int(static_cast<std::int64_t>(d.detector.size())));
    o.set("tracker_updates", sh::JsonValue::make_int(updates));
    o.set("skipped_empty_frames", sh::JsonValue::make_int(skipped));
    o.set("pre_roll_updates", sh::JsonValue::make_int(host.pre_roll_updates_run()));
    auto stages = sh::JsonValue::make_object();
    stages.set("post_nms", post_nms.json());
    stages.set("tracker_input", tracker_input.json());
    stages.set("gmc", gmc.json());
    stages.set("tracker_output", tracker_output.json());
    o.set("stages", stages);
    o.set("mot_txt", mot);
    o.set("exact", sh::JsonValue::make_bool(exact));
    return o;
}

int run(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config_file(opt.config);
    if (!opt.mot_out.empty()) std::filesystem::create_directories(opt.mot_out);
    const sh::JsonValue manifest = sh::parse_strict_json(read_file(opt.dump + "/manifest.json"));
    if (field(manifest, "format").string != kFormat) fail("unknown dump format");
    const std::vector<float> lut = frame_lut(manifest);

    std::vector<std::string> names = opt.sequences;
    if (names.empty()) {
        for (const auto& [name, _] : field(manifest, "sequences").object) names.push_back(name);
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    // One PerceptionPipeline per run, as run_eval builds it.
    std::unique_ptr<saccade::PerceptionPipeline> pipeline = sh::build_perception_pipeline(cfg);

    auto seqs = sh::JsonValue::make_object();
    bool all_exact = true;
    for (const std::string& name : names) {
        const SequenceDump d = load_sequence(opt.dump, name);
        const MotReference mot_ref = load_mot_reference(opt.dump, manifest, name);
        bool exact = false;
        seqs.set(name, replay_sequence(cfg, d, lut, *pipeline, stream, opt.pre_roll, mot_ref,
                                       opt.mot_out, exact));
        all_exact = all_exact && exact;
        std::cerr << "[saccade_replay] " << name << ": " << (exact ? "EXACT" : "DIFFERS") << "\n";
    }
    pipeline.reset();
    cudaStreamDestroy(stream);

    auto report = sh::JsonValue::make_object();
    report.set("format", sh::JsonValue::make_string("saccade.post_detector_replay_report/v2"));
    report.set("config", sh::JsonValue::make_string(opt.config));
    report.set("dump", sh::JsonValue::make_string(opt.dump));
    report.set("pre_roll_override", opt.pre_roll ? sh::JsonValue::make_int(*opt.pre_roll)
                                                 : sh::JsonValue::make_null());
    report.set("sequences", seqs);
    report.set("verdict", sh::JsonValue::make_string(all_exact ? "EXACT" : "DIFFERS"));
    std::ofstream out(opt.report);
    out << sh::dump_python_json(report) << "\n";
    if (!out) fail("cannot write " + opt.report);
    return all_exact ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "saccade_replay: " << e.what() << "\n";
        return 2;
    }
}
