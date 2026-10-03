// Native post-detector host lifecycle on the real GPU objects (#465 Phase B
// PR-5, U3a). Needs a CUDA device; built with ENABLE_NATIVE_TESTS from the root
// build.
//
// Usage: saccade_shipping_post_detector_host_test <path to mamba_whole_graph.resolved.json>
//
// Stage parity against the Python oracle is measured end to end by
// saccade_replay on a dump (results/465_pr5_replay/). This test pins the
// lifecycle the host hard-codes, on synthetic input:
//   * an empty detector output runs nothing (no NMS, GMC or tracker update)
//     and does not trigger the pre-roll;
//   * the pre-roll (kGraphedTrackerUpdatePreRoll empty updates) runs once,
//     lazily, at the sequence's first real update;
//   * the pre-roll override is refused after the first update;
//   * a config the plan refuses makes the host constructor fail closed;
//   * two hosts fed the same frames give bit-identical results, and an
//     FP-hard-rejected row reaches the tracker with the reject score.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/native_build.hpp"
#include "saccade_shipping/post_detector_host.hpp"
#include "tracking/pipeline.hpp"

namespace sh = saccade::shipping;
using sh::JsonValue;

namespace {

int g_failures = 0;
int g_checks = 0;

#undef CHECK  // torch's c10 logging macro, via tracking/pipeline.hpp
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failures;                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kFrames = 6;

// Device copy of one frame's detections.
struct DeviceRows {
    float* boxes = nullptr;
    float* scores = nullptr;
    std::int32_t* classes = nullptr;
    int n = 0;
    DeviceRows(const std::vector<float>& b, const std::vector<float>& s) : n(static_cast<int>(s.size())) {
        cuda_check(cudaMalloc(&boxes, b.size() * sizeof(float)), "boxes");
        cuda_check(cudaMalloc(&scores, s.size() * sizeof(float)), "scores");
        cuda_check(cudaMalloc(&classes, s.size() * sizeof(std::int32_t)), "classes");
        const std::vector<std::int32_t> c(s.size(), 0);
        cuda_check(cudaMemcpy(boxes, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "boxes");
        cuda_check(cudaMemcpy(scores, s.data(), s.size() * sizeof(float), cudaMemcpyHostToDevice), "scores");
        cuda_check(cudaMemcpy(classes, c.data(), c.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                   "classes");
    }
    ~DeviceRows() {
        cudaFree(boxes);
        cudaFree(scores);
        cudaFree(classes);
    }
    DeviceRows(const DeviceRows&) = delete;
    DeviceRows& operator=(const DeviceRows&) = delete;
    sh::DeviceDetections view() const { return {boxes, scores, classes, n, false}; }
};

// Three walkers moving right, plus one large low-score box the FP hard filter
// rejects (score below fp_hard_filter_max_suspicious_score, area above the cap).
void frame_rows(int t, std::vector<float>& boxes, std::vector<float>& scores) {
    boxes.clear();
    scores.clear();
    for (int k = 0; k < 3; ++k) {
        const float x = 60.0f + 150.0f * static_cast<float>(k) + 4.0f * static_cast<float>(t);
        const float y = 120.0f + 20.0f * static_cast<float>(k);
        boxes.insert(boxes.end(), {x, y, x + 50.0f, y + 140.0f});
        scores.push_back(0.9f - 0.05f * static_cast<float>(k));
    }
    boxes.insert(boxes.end(), {10.0f, 10.0f, 410.0f, 460.0f});
    scores.push_back(0.3f);
}

// A deterministic textured frame shifted by `t` px (so GMC has structure).
std::vector<float> frame_pixels(int t) {
    std::vector<float> px(3 * static_cast<std::size_t>(kWidth) * kHeight);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const int u = x + t;
                const float v = 0.5f + 0.25f * std::sin(0.11f * static_cast<float>(u)) *
                                           std::cos(0.07f * static_cast<float>(y + 13 * c));
                px[(static_cast<std::size_t>(c) * kHeight + y) * kWidth + x] = v;
            }
        }
    }
    return px;
}

bool same_rows(const sh::DetectionRows& a, const sh::DetectionRows& b) {
    return a.boxes.size() == b.boxes.size() && a.scores.size() == b.scores.size() &&
           a.classes == b.classes &&
           std::memcmp(a.boxes.data(), b.boxes.data(), a.boxes.size() * sizeof(float)) == 0 &&
           std::memcmp(a.scores.data(), b.scores.data(), a.scores.size() * sizeof(float)) == 0;
}

bool same_result(const sh::FrameResult& a, const sh::FrameResult& b) {
    const auto& x = a.tracker_output;
    const auto& y = b.tracker_output;
    return a.updated == b.updated && same_rows(a.post_nms, b.post_nms) &&
           same_rows(a.tracker_input, b.tracker_input) &&
           std::memcmp(a.gmc_warp.data(), b.gmc_warp.data(), sizeof(float) * 6) == 0 &&
           x.ids == y.ids && x.classes == y.classes && x.boxes.size() == y.boxes.size() &&
           std::memcmp(x.boxes.data(), y.boxes.data(), x.boxes.size() * sizeof(float)) == 0 &&
           x.scores.size() == y.scores.size() &&
           std::memcmp(x.scores.data(), y.scores.data(), x.scores.size() * sizeof(float)) == 0;
}

std::vector<sh::FrameResult> run_sequence(const sh::ResolvedShippingConfig& cfg,
                                          saccade::PerceptionPipeline& pipeline, cudaStream_t stream,
                                          float* d_frame) {
    sh::PostDetectorHost host(cfg, sh::SequenceGeometry{kWidth, kHeight}, pipeline, stream);
    std::vector<sh::FrameResult> out;

    // An empty first frame: nothing runs, no pre-roll yet.
    const DeviceRows empty({}, {});
    const sh::FrameResult r0 = host.process(empty.view(), d_frame);
    CHECK(!r0.updated);
    CHECK(r0.post_nms.size() == 0 && r0.tracker_output.size() == 0);
    CHECK(host.pre_roll_updates_run() == 0);

    std::vector<float> boxes, scores;
    for (int t = 0; t < kFrames; ++t) {
        const std::vector<float> px = frame_pixels(t);
        cuda_check(cudaMemcpy(d_frame, px.data(), px.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "frame");
        frame_rows(t, boxes, scores);
        const DeviceRows rows(boxes, scores);
        out.push_back(host.process(rows.view(), d_frame));
        CHECK(out.back().updated);
        // The pre-roll ran exactly once, at the first real update.
        CHECK(host.pre_roll_updates_run() == sh::kGraphedTrackerUpdatePreRoll);
    }
    bool refused = false;
    try {
        host.set_pre_roll_for_measurement(1);
    } catch (const std::logic_error&) {
        refused = true;
    }
    CHECK(refused);
    return out;
}

void check_constructor_fails_closed(const std::string& config_text, saccade::PerceptionPipeline& pipeline,
                                    cudaStream_t stream) {
    JsonValue doc = sh::parse_strict_json(config_text);
    *doc.find("host_params")->find("steps")->find("filter.stage2_quality_gate") = JsonValue::make_bool(true);
    const sh::ResolvedShippingConfig bad = sh::load_resolved_shipping_config(doc);
    bool refused = false;
    try {
        sh::PostDetectorHost host(bad, sh::SequenceGeometry{kWidth, kHeight}, pipeline, stream);
    } catch (const sh::ConfigError&) {
        refused = true;
    }
    CHECK(refused);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <resolved config>\n", argv[0]);
        return 2;
    }
    try {
        const std::string config_text = read_file(argv[1]);
        const sh::ResolvedShippingConfig cfg = sh::parse_resolved_shipping_config(config_text);
        const sh::PostDetectorPlan plan = sh::plan_post_detector(cfg);
        CHECK(plan.fp_hard && plan.external_fp && plan.gmc);

        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        float* d_frame = nullptr;
        cuda_check(cudaMalloc(&d_frame, 3 * static_cast<std::size_t>(kWidth) * kHeight * sizeof(float)),
                   "frame");
        std::unique_ptr<saccade::PerceptionPipeline> pipeline = sh::build_perception_pipeline(cfg);

        const auto a = run_sequence(cfg, *pipeline, stream, d_frame);
        const auto b = run_sequence(cfg, *pipeline, stream, d_frame);
        CHECK(a.size() == b.size());
        int differing = 0;
        for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
            if (!same_result(a[i], b[i])) ++differing;
        }
        CHECK(differing == 0);

        // The large low-score box is the last NMS survivor class-0 row with
        // score 0.3; after the FP hard filter it carries the reject score, and
        // no track carries it.
        int rejected_rows = 0;
        for (const auto& r : a) {
            for (std::size_t i = 0; i < r.tracker_input.size(); ++i) {
                if (r.tracker_input.scores[i] == plan.fp_hard_filter.reject_score) ++rejected_rows;
            }
            for (float s : r.tracker_output.scores) CHECK(s > 0.0f);
        }
        CHECK(rejected_rows == kFrames);
        CHECK(!a.empty() && a.back().tracker_output.size() == 3);

        check_constructor_fails_closed(config_text, *pipeline, stream);

        pipeline.reset();
        cudaFree(d_frame);
        cudaStreamDestroy(stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
