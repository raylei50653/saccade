// Shipping builders on the real GPU objects (#465 Phase B PR-4b, U2b-b).
// Needs a CUDA device; built with ENABLE_NATIVE_TESTS from the root build.
//
// Usage: saccade_shipping_native_build_test <path to mamba_whole_graph.resolved.json>
//
// Pins:
//   * build_tracker / build_gmc / build_perception_pipeline succeed on the
//     committed config, and each object's snapshot equals the CUDA-free plan
//     (the readback the builders enforce is against the JSON itself);
//   * setting every SACCADE_* hatch variable does not change any built
//     object (shipping reads only the JSON);
//   * the first update_into inside a CUDA stream capture freezes the tracker:
//     every runtime-semantic setter then throws and the parameters stay put;
//   * a value a native setter canonicalizes makes build_tracker fail closed;
//   * the legacy front-end path (tracking/legacy_env.hpp) still lands its
//     env values in the same parameter state.

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/native_build.hpp"
#include "tracking/gmc.hpp"
#include "tracking/legacy_env.hpp"
#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"

namespace sh = saccade::shipping;
using saccade::FilterCompactionMode;
using saccade::GPUByteTracker;
using saccade::TrackerParams;
using sh::ConfigError;
using sh::JsonValue;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failures;                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

const sh::SequenceGeometry kGeometry{1920, 1080};

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

std::string repr(bool v) { return v ? "true" : "false"; }
std::string repr(int v) { return std::to_string(v); }
std::string repr(std::uintptr_t v) { return std::to_string(static_cast<unsigned long long>(v)); }
std::string repr(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    char buf[32];
    std::snprintf(buf, sizeof buf, "f32:%08x", u);
    return buf;
}
std::string repr(const std::optional<std::array<float, 9>>& v) { return v ? "matrix" : "null"; }
std::string repr(FilterCompactionMode v) { return saccade::to_string(v); }

using Flat = std::map<std::string, std::string>;
template <class Snapshot> Flat flatten(const Snapshot& s) {
    Flat out;
    s.visit([&](const auto& key, const auto& value) { out[std::string(key)] = repr(value); });
    return out;
}

const std::vector<std::string>& hatch_variables() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        for (const auto& c : sh::native_env_consumers()) v.emplace_back(c.key);
        return v;
    }();
    return names;
}

bool throws_logic_error(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

// ─── tests ───────────────────────────────────────────────────────────────

struct Built {
    Flat tracker, gmc, pipeline;
};

Built build_all(const sh::ResolvedShippingConfig& cfg) {
    Built b;
    b.tracker = flatten(sh::build_tracker(cfg, kGeometry)->snapshot());
    b.gmc = flatten(sh::build_gmc(cfg)->snapshot());
    b.pipeline = flatten(sh::build_perception_pipeline(cfg)->snapshot());
    return b;
}

void test_builders_read_back(const sh::ResolvedShippingConfig& cfg, Built& reference) {
    auto tracker = sh::build_tracker(cfg, kGeometry);  // throws on readback mismatch
    const saccade::TrackerSnapshot snap = tracker->snapshot();
    CHECK(flatten(snap.params) == flatten(sh::planned_tracker_params(cfg, kGeometry)));
    CHECK(snap.params.oao.score_w == -1.0f);
    CHECK(snap.max_objects == 2048 && snap.embedding_dim == 768 && snap.max_assoc == 1024);
    CHECK(!snap.config_frozen);
    CHECK(flatten(sh::build_gmc(cfg)->snapshot()) == flatten(sh::planned_gmc_snapshot(cfg)));
    CHECK(flatten(sh::build_perception_pipeline(cfg)->snapshot()) ==
          flatten(sh::planned_pipeline_snapshot(cfg)));
    reference = build_all(cfg);
}

void test_environment_is_not_read(const sh::ResolvedShippingConfig& cfg, const Built& reference) {
    for (const char* value : {"0", "1", "3.5", "nonsense"}) {
        for (const auto& name : hatch_variables()) setenv(name.c_str(), value, 1);
        const Built again = build_all(cfg);
        CHECK(again.tracker == reference.tracker);
        CHECK(again.gmc == reference.gmc);
        CHECK(again.pipeline == reference.pipeline);
    }
    for (const auto& name : hatch_variables()) unsetenv(name.c_str());
}

void test_capture_freezes_configuration(const sh::ResolvedShippingConfig& cfg) {
    auto tracker = sh::build_tracker(cfg, kGeometry);
    const saccade::TrackerSnapshot before = tracker->snapshot();
    const int max_assoc = before.max_assoc;
    const int max_objs = before.max_objects;

    float *boxes, *scores, *gmc, *out_boxes, *out_scores;
    int *classes, *out_ids, *out_classes, *out_det_idx, *out_count;
    cuda_ok(cudaMalloc(&boxes, max_assoc * 4 * sizeof(float)), "malloc");
    cuda_ok(cudaMalloc(&scores, max_assoc * sizeof(float)), "malloc");
    cuda_ok(cudaMalloc(&classes, max_assoc * sizeof(int)), "malloc");
    cuda_ok(cudaMalloc(&gmc, 6 * sizeof(float)), "malloc");
    cuda_ok(cudaMalloc(&out_boxes, max_objs * 4 * sizeof(float)), "malloc");
    cuda_ok(cudaMalloc(&out_scores, max_objs * sizeof(float)), "malloc");
    cuda_ok(cudaMalloc(&out_ids, max_objs * sizeof(int)), "malloc");
    cuda_ok(cudaMalloc(&out_classes, max_objs * sizeof(int)), "malloc");
    cuda_ok(cudaMalloc(&out_det_idx, max_objs * sizeof(int)), "malloc");
    cuda_ok(cudaMalloc(&out_count, sizeof(int)), "malloc");
    cuda_ok(cudaMemset(boxes, 0, max_assoc * 4 * sizeof(float)), "memset");
    cuda_ok(cudaMemset(scores, 0, max_assoc * sizeof(float)), "memset");
    cuda_ok(cudaMemset(classes, 0, max_assoc * sizeof(int)), "memset");
    const float identity[6] = {1, 0, 0, 0, 1, 0};
    cuda_ok(cudaMemcpy(gmc, identity, sizeof identity, cudaMemcpyHostToDevice), "memcpy");

    cudaStream_t stream;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    auto update = [&] {
        tracker->update_into(boxes, scores, classes, max_assoc, stream, out_boxes, out_scores,
                             out_ids, out_classes, out_det_idx, out_count, nullptr, gmc, 0.0f,
                             1.0f, max_objs);
    };

    // Uncaptured updates (the graph-capture warm-up) leave it mutable.
    update();
    update();
    cuda_ok(cudaStreamSynchronize(stream), "warmup");
    CHECK(!tracker->snapshot().config_frozen);
    tracker->set_frame_size(kGeometry.im_width, kGeometry.im_height);  // still allowed

    cudaGraph_t graph = nullptr;
    cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    update();
    cuda_ok(cudaStreamEndCapture(stream, &graph), "end capture");
    CHECK(tracker->snapshot().config_frozen);

    cudaGraphExec_t exec = nullptr;
    cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cuda_ok(cudaGraphLaunch(exec, stream), "launch");
    cuda_ok(cudaStreamSynchronize(stream), "replay");

    GPUByteTracker& t = *tracker;
    const std::vector<std::pair<const char*, std::function<void()>>> setters = {
        {"set_params", [&] { t.set_params(0.1f, 0.5f, 0.8f, 30); }},
        {"set_reid_params", [&] { t.set_reid_params(0.9f, 0.3f, 0.6f, 0.4f); }},
        {"set_reid_min_candidates", [&] { t.set_reid_min_candidates(1); }},
        {"set_relink_params", [&] { t.set_relink_params(false, 256, 0.6f, 2.5f, 4.0f, 300); }},
        {"set_unified_score_params", [&] { t.set_unified_score_params({}); }},
        {"set_frame_size", [&] { t.set_frame_size(640, 480); }},
        {"set_quality_params", [&] { t.set_quality_params(true); }},
        {"set_oao_params", [&] { t.set_oao_params(0.1f); }},
        {"set_occ_params", [&] { t.set_occ_params(false, 0.45f, 0.15f, 4, 0.5f); }},
        {"set_multiplicative_cost", [&] { t.set_multiplicative_cost(false); }},
        {"set_sinkhorn_lambda", [&] { t.set_sinkhorn_lambda(30.0f); }},
        {"set_stability_cost_w", [&] { t.set_stability_cost_w(0.0f); }},
        {"set_association_energy_params", [&] { t.set_association_energy_params(true, 0.1f, 0.1f); }},
        {"set_homography", [&] { t.set_homography(nullptr); }},
        {"set_hatch_params", [&] { t.set_hatch_params(TrackerParams::Hatch{}); }},
        {"set_research_portable_or_tail", [&] { t.set_research_portable_or_tail(false, {}); }},
        {"set_research_bridge_shadow", [&] { t.set_research_bridge_shadow(true); }},
        {"set_research_bridge_fidelity_audit", [&] { t.set_research_bridge_fidelity_audit(true); }},
        {"set_research_h0_bridge_trace", [&] { t.set_research_h0_bridge_trace(true); }},
    };
    for (const auto& [name, call] : setters) {
        const bool threw = throws_logic_error(call);
        if (!threw) std::fprintf(stderr, "  %s did not throw after capture\n", name);
        CHECK(threw);
    }
    saccade::TrackerSnapshot after = tracker->snapshot();
    CHECK(after.config_frozen);
    after.config_frozen = false;
    CHECK(flatten(after) == flatten(before));

    cuda_ok(cudaGraphExecDestroy(exec), "exec destroy");
    cuda_ok(cudaGraphDestroy(graph), "graph destroy");
    cuda_ok(cudaStreamDestroy(stream), "stream destroy");
    for (void* p : std::initializer_list<void*>{boxes, scores, classes, gmc, out_boxes, out_scores,
                                                out_ids, out_classes, out_det_idx, out_count}) {
        cuda_ok(cudaFree(p), "free");
    }
}

void test_canonicalized_value_fails_closed(const JsonValue& golden) {
    JsonValue doc = golden;
    for (JsonValue& call : doc.find("native_params")->find("GPUByteTracker")->find("calls")->array) {
        if (call.find("method")->string == "set_params") {
            *call.find("args")->find("confirm_streak") = JsonValue::make_int(0);
        }
    }
    const auto cfg = sh::load_resolved_shipping_config(doc);  // the PR-4a guard admits 0
    std::string error;
    try {
        sh::build_tracker(cfg, kGeometry);
    } catch (const ConfigError& e) {
        error = e.what();
    }
    CHECK(error.find("set_params.confirm_streak: native 1 != resolved 0") != std::string::npos);
    if (!error.empty()) std::printf("fail-closed as expected:\n%s\n", error.c_str());
}

void test_legacy_env_reaches_the_same_state() {
    setenv("SACCADE_STABILITY_W", "0", 1);
    setenv("SACCADE_ENABLE_DDA", "0", 1);
    setenv("SACCADE_GMC_PCR_THRESH", "3", 1);
    setenv("SACCADE_DETERMINISTIC_FILTER_COMPACTION", "1", 1);
    setenv("SACCADE_KALMAN_ADAPT_MODE", "2", 1);

    GPUByteTracker tracker(64, 8, 64);
    saccade::legacy_env::apply(tracker);
    const auto hatch = tracker.snapshot().params.hatch;
    CHECK(hatch.stability_w == 0.0f && !hatch.enable_dda);
    CHECK(saccade::legacy_env::kalman_adapt_mode_override() == std::optional<int>(2));

    saccade::GMC gmc(4);
    CHECK(gmc.snapshot().pcr_thresh == 5.0f);  // the GMC itself never reads env
    saccade::legacy_env::apply(gmc);
    CHECK(gmc.snapshot().pcr_thresh == 3.0f);

    saccade::PerceptionPipeline pipeline(nullptr, nullptr, saccade::PerceptionPipelineConfig{});
    CHECK(pipeline.snapshot().filter_compaction == FilterCompactionMode::kStableScan);
    saccade::legacy_env::apply(pipeline);
    CHECK(pipeline.snapshot().filter_compaction == FilterCompactionMode::kSerialStable);

    for (const char* name : {"SACCADE_STABILITY_W", "SACCADE_ENABLE_DDA", "SACCADE_GMC_PCR_THRESH",
                             "SACCADE_DETERMINISTIC_FILTER_COMPACTION", "SACCADE_KALMAN_ADAPT_MODE"}) {
        unsetenv(name);
    }
    GPUByteTracker plain(64, 8, 64);
    saccade::legacy_env::apply(plain);
    CHECK(plain.snapshot().params.hatch.stability_w == 0.1f);
    CHECK(!saccade::legacy_env::kalman_adapt_mode_override().has_value());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <resolved config json>\n", argv[0]);
        return 2;
    }
    const std::string text = read_file(argv[1]);
    if (text.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    try {
        const JsonValue golden = sh::parse_strict_json(text);
        const auto cfg = sh::load_resolved_shipping_config(golden);
        Built reference;
        test_builders_read_back(cfg, reference);
        test_environment_is_not_read(cfg, reference);
        test_capture_freezes_configuration(cfg);
        test_canonicalized_value_fails_closed(golden);
        test_legacy_env_reaches_the_same_state();
    } catch (const ConfigError& e) {
        std::fprintf(stderr, "unexpected ConfigError: %s\n", e.what());
        return 1;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
