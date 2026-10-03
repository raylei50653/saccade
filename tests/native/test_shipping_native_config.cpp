// Resolved JSON -> native parameter state, and readback (#465 Phase B PR-4b).
// CUDA-free: runs in the CI job `shipping-config-loader`.
//
// Usage: saccade_shipping_native_config_test <path to mamba_whole_graph.resolved.json>
//
// Pins, on the tracker's own parameter store (TrackerParams, the code the GPU
// tracker runs) and on the planned GMC/PerceptionPipeline snapshots:
//   * the committed config reads back exactly (every native key has a JSON
//     value, every JSON value has a native key, floats bit-exact), including
//     set_oao_params.score_w = -1;
//   * the native_env partition and the native-only expectations are exactly
//     the documented ones;
//   * the tracker setters run in the oracle's call order;
//   * a value a native setter would canonicalize fails readback;
//   * every resolved value reaches exactly one native field;
//   * the process environment does not change any planned value.
// The GPU objects are covered by test_shipping_native_build.cpp.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "saccade_shipping/native_config.hpp"
#include "saccade_shipping/resolved_config.hpp"

namespace sh = saccade::shipping;
using saccade::FilterCompactionMode;
using saccade::GmcSnapshot;
using saccade::PerceptionPipelineSnapshot;
using saccade::TrackerParams;
using saccade::TrackerSnapshot;
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

const sh::SequenceGeometry kGeometry{1920, 1080};

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// ─── snapshot flattening ─────────────────────────────────────────────────

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
std::string repr(const std::optional<std::array<float, 9>>& v) {
    if (!v) return "null";
    std::string s;
    for (float x : *v) s += repr(x) + ",";
    return s;
}
std::string repr(FilterCompactionMode v) { return saccade::to_string(v); }

using Flat = std::map<std::string, std::string>;

struct FlattenInto {
    Flat& out;
    std::string prefix;
    template <class K, class T> void operator()(const K& key, const T& value) const {
        out[prefix + std::string(key)] = repr(value);
    }
};

TrackerSnapshot planned_tracker_snapshot(const sh::ResolvedShippingConfig& cfg) {
    const auto& k = cfg.native_params.tracker.constructor;
    TrackerSnapshot s;
    s.max_objects = static_cast<int>(k.max_objects);
    s.embedding_dim = static_cast<int>(k.embedding_dim);
    s.max_assoc = static_cast<int>(std::max<std::int64_t>(1, k.max_assoc));
    s.params = sh::planned_tracker_params(cfg, kGeometry);
    return s;
}

Flat flatten_plan(const sh::ResolvedShippingConfig& cfg) {
    Flat out;
    planned_tracker_snapshot(cfg).visit(FlattenInto{out, "GPUByteTracker:"});
    sh::planned_gmc_snapshot(cfg).visit(FlattenInto{out, "GMC:"});
    sh::planned_pipeline_snapshot(cfg).visit(FlattenInto{out, "PerceptionPipeline:"});
    return out;
}

void print_mismatches(const char* what, const std::vector<std::string>& m) {
    for (const auto& line : m) std::fprintf(stderr, "  %s: %s\n", what, line.c_str());
}

// ─── JSON editing ────────────────────────────────────────────────────────

JsonValue& child(JsonValue& node, const std::string& key) {
    JsonValue* v = node.find(key);
    if (v == nullptr) {
        std::fprintf(stderr, "missing key %s\n", key.c_str());
        std::abort();
    }
    return *v;
}

JsonValue& call_args(JsonValue& doc, const std::string& object, const std::string& method) {
    for (JsonValue& call : child(child(child(doc, "native_params"), object), "calls").array) {
        if (child(call, "method").string == method) return child(call, "args");
    }
    std::fprintf(stderr, "no call %s.%s\n", object.c_str(), method.c_str());
    std::abort();
}

// A leaf of the resolved JSON and the native snapshot key it must reach.
struct Leaf {
    std::vector<std::string> path;  // keys from the document root; "#method" selects a call
    std::string native_key;         // "<Object>:<snapshot key>"
};

JsonValue& at(JsonValue& doc, const std::vector<std::string>& path) {
    JsonValue* node = &doc;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const std::string& step = path[i];
        if (!step.empty() && step[0] == '#') {
            // path = native_params, <Object>, #method, <arg...>
            node = &call_args(doc, path[1], step.substr(1));
        } else {
            node = &child(*node, step);
        }
    }
    return *node;
}

void args_leaves(const JsonValue& args, std::vector<std::string> base, const std::string& key_prefix,
                 const std::string& object, std::vector<Leaf>& out) {
    for (const auto& [k, v] : args.object) {
        auto path = base;
        path.push_back(k);
        const bool per_sequence = v.kind == JsonValue::Kind::Object && v.find("per_sequence");
        if (v.kind == JsonValue::Kind::Object && !per_sequence) {
            args_leaves(v, path, key_prefix + k + ".", object, out);
        } else if (!per_sequence && v.kind != JsonValue::Kind::Null) {
            out.push_back({path, object + ":" + key_prefix + k});
        }
    }
}

std::vector<Leaf> resolved_leaves(JsonValue& doc) {
    std::vector<Leaf> out;
    JsonValue& native = child(doc, "native_params");
    for (const char* object : {"GPUByteTracker", "GMC"}) {
        JsonValue& o = child(native, object);
        if (std::string(object) == "GMC") {
            args_leaves(child(o, "constructor"), {"native_params", object, "constructor"},
                        "constructor.", object, out);
        }
        for (JsonValue& call : child(o, "calls").array) {
            const std::string method = child(call, "method").string;
            args_leaves(child(call, "args"), {"native_params", object, "#" + method},
                        method + ".", object, out);
        }
    }
    args_leaves(child(child(native, "PerceptionPipelineConfig"), "fields"),
                {"native_params", "PerceptionPipelineConfig", "fields"}, "constructor.config.",
                "PerceptionPipeline", out);
    args_leaves(child(child(native, "PerceptionPipeline"), "constructor"),
                {"native_params", "PerceptionPipeline", "constructor"}, "constructor.",
                "PerceptionPipeline", out);
    for (JsonValue& call : child(child(native, "PerceptionPipeline"), "calls").array) {
        const std::string method = child(call, "method").string;
        args_leaves(child(call, "args"), {"native_params", "PerceptionPipeline", "#" + method},
                    method + ".", "PerceptionPipeline", out);
    }
    // The pipeline constructor's config reference is structure, not a value.
    std::vector<Leaf> kept;
    for (auto& leaf : out) {
        if (leaf.native_key != "PerceptionPipeline:constructor.config.ref") kept.push_back(leaf);
    }
    out.swap(kept);
    for (const auto& [k, v] : child(doc, "native_env").object) {
        if (v.kind == JsonValue::Kind::Object) continue;  // {"unset_effect"}: no value
        std::string consumer;
        for (const auto& c : sh::native_env_consumers()) {
            if (k == c.key) consumer = c.consumer;
        }
        std::string key = consumer + ":native_env." + k;
        if (k == "SACCADE_DETERMINISTIC_FILTER_COMPACTION" || k == "SACCADE_ATOMIC_FILTER_BASELINE") {
            key = "PerceptionPipeline:filter_compaction";
        }
        out.push_back({{"native_env", k}, key});
    }
    return out;
}

std::vector<JsonValue> perturbations(const JsonValue& v) {
    switch (v.kind) {
        case JsonValue::Kind::Bool:
            return {JsonValue::make_bool(!v.boolean)};
        case JsonValue::Kind::Int:
            return {JsonValue::make_int(v.integer + 1), JsonValue::make_int(v.integer - 1)};
        case JsonValue::Kind::Float:
            return {JsonValue::make_float(v.number + 0.125), JsonValue::make_float(v.number - 0.125),
                    JsonValue::make_float(v.number * 0.5), JsonValue::make_float(0.5),
                    JsonValue::make_float(0.25), JsonValue::make_float(2.0)};
        default:
            return {};
    }
}

bool loads(const JsonValue& doc, sh::ResolvedShippingConfig* out = nullptr) {
    try {
        auto cfg = sh::load_resolved_shipping_config(doc);
        if (out) *out = std::move(cfg);
        return true;
    } catch (const ConfigError&) {
        return false;
    }
}

// ─── tests ───────────────────────────────────────────────────────────────

void test_committed_config_reads_back(const sh::ResolvedShippingConfig& cfg) {
    const auto tracker = sh::readback_mismatches(sh::expected_tracker_snapshot(cfg, kGeometry),
                                                 planned_tracker_snapshot(cfg));
    print_mismatches("GPUByteTracker", tracker);
    CHECK(tracker.empty());
    const auto gmc =
        sh::readback_mismatches(sh::expected_gmc_snapshot(cfg), sh::planned_gmc_snapshot(cfg));
    print_mismatches("GMC", gmc);
    CHECK(gmc.empty());
    const auto pipeline = sh::readback_mismatches(sh::expected_pipeline_snapshot(cfg),
                                                  sh::planned_pipeline_snapshot(cfg));
    print_mismatches("PerceptionPipeline", pipeline);
    CHECK(pipeline.empty());

    const TrackerParams p = sh::planned_tracker_params(cfg, kGeometry);
    CHECK(p.oao.score_w == -1.0f);  // kept as resolved, not canonicalized to 0
    CHECK(p.oao.contest_thresh == -1.0f);
    CHECK(p.relink.bridge_app_veto == -1.0f);
    CHECK(p.core.new_track_thresh == 0.28f);
    CHECK(p.relink.bidirectional && !p.relink.enabled);
    CHECK(p.hatch.enable_dda && p.hatch.stability_w == 0.1f && p.hatch.freshness_w == 0.0f);
    CHECK(p.frame.w == 1920 && p.frame.h == 1080);
    CHECK(!p.homography.has_value());
    CHECK(sh::planned_gmc_snapshot(cfg).pcr_thresh == 5.0f);
    CHECK(sh::planned_pipeline_snapshot(cfg).filter_compaction == FilterCompactionMode::kStableScan);

    const auto tracker_keys = sh::expected_tracker_snapshot(cfg, kGeometry);
    // 3 constructor dims + 92 TrackerParams fields + 5 hooks/diagnostic + config_frozen
    CHECK(tracker_keys.size() == 101);
}

void test_partitions_are_the_documented_ones(const sh::ResolvedShippingConfig& cfg) {
    const JsonValue doc = sh::to_json(cfg);
    std::set<std::string> json_env;
    for (const auto& [k, v] : doc.find("native_env")->object) json_env.insert(k);
    std::set<std::string> consumed;
    for (const auto& c : sh::native_env_consumers()) {
        CHECK(consumed.insert(c.key).second);
        const std::string consumer = c.consumer;
        CHECK(consumer == "GPUByteTracker" || consumer == "GMC" || consumer == "PerceptionPipeline" ||
              (consumer.empty() && std::strlen(c.reason) > 0));
    }
    CHECK(consumed == json_env);

    std::set<std::string> native_only;
    for (const auto& row : sh::tracker_native_only_expectations()) {
        native_only.insert(row.key);
        CHECK(!row.reason.empty());
    }
    CHECK(native_only == (std::set<std::string>{
                             "set_reid_min_candidates.min_candidates",
                             "research.portable_or_tail",
                             "research.bridge_shadow",
                             "research.bridge_fidelity_audit",
                             "research.h0_bridge_trace",
                             "config_frozen",
                         }));
}

struct RecordingTracker {
    std::vector<std::string> calls;
    template <class... A> void record(const char* m, const A&...) { calls.emplace_back(m); }
    template <class... A> void set_hatch_params(const A&... a) { record("set_hatch_params", a...); }
    template <class... A> void set_homography(const A&... a) { record("set_homography", a...); }
    template <class... A> void set_reid_params(const A&... a) { record("set_reid_params", a...); }
    template <class... A> void set_relink_params(const A&... a) { record("set_relink_params", a...); }
    template <class... A> void set_unified_score_params(const A&... a) {
        record("set_unified_score_params", a...);
    }
    template <class... A> void set_frame_size(const A&... a) { record("set_frame_size", a...); }
    template <class... A> void set_quality_params(const A&... a) { record("set_quality_params", a...); }
    template <class... A> void set_params(const A&... a) { record("set_params", a...); }
    template <class... A> void set_oao_params(const A&... a) { record("set_oao_params", a...); }
    template <class... A> void set_occ_params(const A&... a) { record("set_occ_params", a...); }
    template <class... A> void set_multiplicative_cost(const A&... a) {
        record("set_multiplicative_cost", a...);
    }
    template <class... A> void set_sinkhorn_lambda(const A&... a) { record("set_sinkhorn_lambda", a...); }
    template <class... A> void set_stability_cost_w(const A&... a) {
        record("set_stability_cost_w", a...);
    }
    template <class... A> void set_association_energy_params(const A&... a) {
        record("set_association_energy_params", a...);
    }
};

void test_setters_run_in_oracle_order(const sh::ResolvedShippingConfig& cfg) {
    RecordingTracker rec;
    sh::apply_tracker_config(rec, cfg, kGeometry);
    std::vector<std::string> oracle = {"set_hatch_params"};
    const JsonValue doc = sh::to_json(cfg);
    for (const JsonValue& call : doc.find("native_params")->find("GPUByteTracker")->find("calls")->array) {
        oracle.push_back(call.find("method")->string);
    }
    CHECK(rec.calls == oracle);
}

void test_canonicalized_values_fail_readback(const JsonValue& golden) {
    struct Case {
        std::vector<std::string> path;
        JsonValue value;
        std::string key;
    };
    const std::vector<Case> cases = {
        {{"native_params", "GPUByteTracker", "#set_params", "confirm_streak"}, JsonValue::make_int(0),
         "set_params.confirm_streak"},
        {{"native_params", "GPUByteTracker", "#set_oao_params", "score_w"}, JsonValue::make_float(1.5),
         "set_oao_params.score_w"},
        {{"native_params", "GPUByteTracker", "#set_occ_params", "ttl"}, JsonValue::make_int(0),
         "set_occ_params.ttl"},
        {{"native_params", "GPUByteTracker", "#set_relink_params", "occ_gap_min"},
         JsonValue::make_int(0), "set_relink_params.occ_gap_min"},
        {{"native_params", "GPUByteTracker", "#set_relink_params", "bridge_ttl"}, JsonValue::make_int(0),
         "set_relink_params.bridge_ttl"},
        {{"native_params", "GPUByteTracker", "#set_oao_params", "tau"}, JsonValue::make_float(1.25),
         "set_oao_params.tau"},
        {{"native_env", "SACCADE_COAST_MAX_AGE"}, JsonValue::make_float(2.5),
         "native_env.SACCADE_COAST_MAX_AGE"},
    };
    for (const Case& c : cases) {
        JsonValue doc = golden;
        at(doc, c.path) = c.value;
        sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config(golden);
        const bool admitted = loads(doc, &cfg);  // the PR-4a guard admits each value
        CHECK(admitted);
        if (!admitted) continue;
        const auto m = sh::readback_mismatches(sh::expected_tracker_snapshot(cfg, kGeometry),
                                               planned_tracker_snapshot(cfg));
        CHECK(m.size() == 1);
        CHECK(!m.empty() && m.front().rfind(c.key + ":", 0) == 0);
        if (m.size() != 1) print_mismatches(c.key.c_str(), m);
        bool threw = false;
        try {
            sh::require_readback("GPUByteTracker", m);
        } catch (const ConfigError& e) {
            threw = std::string(e.what()).find(c.key) != std::string::npos;
        }
        CHECK(threw);
    }
}

void test_every_resolved_value_reaches_one_native_field(const JsonValue& golden) {
    JsonValue doc = golden;
    const Flat base = flatten_plan(sh::load_resolved_shipping_config(golden));
    std::set<std::string> pinned;
    int reached = 0;
    for (const Leaf& leaf : resolved_leaves(doc)) {
        bool done = false;
        for (const JsonValue& candidate : perturbations(at(doc, leaf.path))) {
            JsonValue mutated = golden;
            at(mutated, leaf.path) = candidate;
            sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config(golden);
            if (!loads(mutated, &cfg)) continue;
            const Flat after = flatten_plan(cfg);
            std::set<std::string> changed;
            for (const auto& [k, v] : after) {
                if (base.at(k) != v) changed.insert(k);
            }
            if (changed.empty()) continue;  // canonicalized back; try another value
            ++g_checks;
            if (changed != std::set<std::string>{leaf.native_key}) {
                ++g_failures;
                std::fprintf(stderr, "resolved %s reached:", leaf.native_key.c_str());
                for (const auto& k : changed) std::fprintf(stderr, " %s", k.c_str());
                std::fprintf(stderr, "\n");
            }
            done = true;
            ++reached;
            break;
        }
        if (!done) pinned.insert(leaf.native_key);
    }
    // Values the loader pins to a single admissible value: no perturbation loads.
    CHECK(pinned == (std::set<std::string>{
                        "PerceptionPipeline:constructor.reid_ptr",
                        "PerceptionPipeline:constructor.cropper_ptr",
                    }));
    for (const auto& k : pinned) std::fprintf(stderr, "  pinned: %s\n", k.c_str());
    // tracker call args 77 + GMC 7 + pipeline config 24 + pipeline call 1 + native_env values 14
    CHECK(reached == 77 + 7 + 24 + 1 + 14);
    std::printf("resolved values reaching native fields: %d (pinned by the loader: %zu)\n", reached,
                pinned.size());
}

void test_environment_does_not_change_the_plan(const sh::ResolvedShippingConfig& cfg) {
    const Flat before = flatten_plan(cfg);
    std::vector<std::string> names;
    for (const auto& c : sh::native_env_consumers()) names.emplace_back(c.key);
    for (const char* value : {"0", "1", "0.5", "false", "true", "7", "nonsense", ""}) {
        for (const auto& name : names) setenv(name.c_str(), value, 1);
        CHECK(flatten_plan(cfg) == before);
        CHECK(sh::readback_mismatches(sh::expected_tracker_snapshot(cfg, kGeometry),
                                      planned_tracker_snapshot(cfg))
                  .empty());
    }
    for (const auto& name : names) unsetenv(name.c_str());
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
    try {
        const JsonValue golden = sh::parse_strict_json(text);
        const auto cfg = sh::load_resolved_shipping_config(golden);
        test_committed_config_reads_back(cfg);
        test_partitions_are_the_documented_ones(cfg);
        test_setters_run_in_oracle_order(cfg);
        test_canonicalized_values_fail_readback(golden);
        test_every_resolved_value_reaches_one_native_field(golden);
        test_environment_does_not_change_the_plan(cfg);
    } catch (const ConfigError& e) {
        std::fprintf(stderr, "unexpected ConfigError: %s\n", e.what());
        return 1;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
