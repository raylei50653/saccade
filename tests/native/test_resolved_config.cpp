// Strict native loader for the resolved shipping config (#465 Phase B PR-4a).
//
// Usage: saccade_resolved_config_test <path to mamba_whole_graph.resolved.json>
//
// Pins: the committed JSON parses and its canonical snapshot is byte-identical
// to the file; removing any field, adding an unknown field or changing any
// value's JSON type fails; enum/range/NaN/Inf violations fail; process env does
// not affect the result; the writer matches Python's json.dumps/repr.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "saccade_shipping/resolved_config.hpp"

using saccade::shipping::ConfigError;
using saccade::shipping::JsonValue;
namespace sh = saccade::shipping;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                               \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(cond)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                         \
    } while (0)

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// Returns the ConfigError message, or "" when the load succeeded.
std::string load_error(const JsonValue& doc) {
    try {
        sh::load_resolved_shipping_config(doc);
    } catch (const ConfigError& e) {
        return e.what();
    }
    return "";
}

std::string parse_error(const std::string& text) {
    try {
        sh::parse_resolved_shipping_config(text);
    } catch (const ConfigError& e) {
        return e.what();
    }
    return "";
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

void expect_rejected(const std::string& err, const std::string& fragment, const std::string& what) {
    ++g_checks;
    if (err.empty() || !contains(err, fragment)) {
        ++g_failures;
        std::fprintf(stderr, "expected rejection [%s] containing \"%s\", got \"%s\"\n", what.c_str(),
                     fragment.c_str(), err.c_str());
    }
}

// "a/b/3/c": object keys and array indices.
JsonValue& at(JsonValue& root, const std::string& path) {
    JsonValue* node = &root;
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t slash = path.find('/', start);
        const std::string step = path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (node->kind == JsonValue::Kind::Array) {
            node = &node->array.at(std::stoul(step));
        } else {
            node = node->find(step);
            if (node == nullptr) {
                std::fprintf(stderr, "bad test path %s at %s\n", path.c_str(), step.c_str());
                std::abort();
            }
        }
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    return *node;
}

// Visits every node with a printable path; `fn(node, path)`.
void walk(JsonValue& node, const std::string& path, const std::function<void(JsonValue&, const std::string&)>& fn) {
    fn(node, path);
    if (node.kind == JsonValue::Kind::Object) {
        for (auto& [k, v] : node.object) walk(v, path + "/" + k, fn);
    } else if (node.kind == JsonValue::Kind::Array) {
        for (std::size_t i = 0; i < node.array.size(); ++i) walk(node.array[i], path + "/" + std::to_string(i), fn);
    }
}

// Collect paths first (mutating while walking would invalidate references).
std::vector<std::string> paths_where(JsonValue& doc, const std::function<bool(const JsonValue&)>& pred) {
    std::vector<std::string> out;
    walk(doc, "", [&](JsonValue& n, const std::string& p) {
        if (pred(n)) out.push_back(p);
    });
    return out;
}

JsonValue& at_walk_path(JsonValue& doc, const std::string& walk_path) {
    return walk_path.empty() ? doc : at(doc, walk_path.substr(1));
}

// ─── tests ────────────────────────────────────────────────────────────────

void test_golden_round_trip(const std::string& golden_text) {
    const auto config = sh::parse_resolved_shipping_config(golden_text);
    const std::string snapshot = sh::canonical_snapshot(config);
    CHECK(snapshot == golden_text);  // JSON -> typed -> snapshot is the committed file
    const auto again = sh::parse_resolved_shipping_config(snapshot);
    CHECK(sh::canonical_snapshot(again) == snapshot);
    CHECK(sh::to_json(config) == sh::parse_strict_json(golden_text));

    // Typed values carry the JSON semantics (spot checks across sections).
    CHECK(config.schema == sh::kResolvedConfigSchema);
    CHECK(config.native_params.tracker.constructor.max_objects == 2048);
    CHECK(config.native_params.tracker.calls.set_params.track_thresh == 0.05);
    CHECK(config.native_params.tracker.calls.set_params.kalman_adapt_mode == 0);
    CHECK(config.native_params.tracker.calls.set_relink_params.bridge_anchor == 2);
    CHECK(config.native_params.tracker.calls.set_frame_size.w == sh::PerSequenceValue::ImWidth);
    CHECK(config.native_params.gmc.constructor.downscale == 4);
    CHECK(config.native_params.perception_pipeline_config.fields.geometry_suspect_support_score ==
          0.07500000000000001);
    CHECK(config.native_env.enable_dda == true);
    CHECK(config.native_env.stability_w == 0.1);
    CHECK(config.native_env.gmc_pcr_thresh == 5.0);
    CHECK(config.native_env.kalman_adapt_mode.unset_effect ==
          "no override; set_params kalman_adapt_mode stands");
    CHECK(config.host_params.steps.tail_interpolation == true);
    CHECK(config.host_params.steps.filter_external_fp == true);
    CHECK(config.host_params.external_fp_rule_config.min_height == 72.0);
    CHECK(config.host_params.stage_order.front() == "_run_detect");
    CHECK(config.host_params.env.detect_barrier.value_or("") == "event");
    CHECK(!config.host_params.env.build_path.has_value());
    CHECK(config.host_params.cfg.interpolate_max_gap == 35);
    CHECK(config.host_params.cfg.kwargs_fpn_backbone_engine == "models/yolo/yolo26s_backbone_640_best.engine");
    CHECK((config.host_params.cfg.crop_hw == sh::IntList{224, 224}));
    CHECK(config.host_params.cfg.preprocess_modes.empty());
}

void test_every_field_is_required(const JsonValue& golden) {
    JsonValue scratch = golden;
    const auto objects = paths_where(scratch, [](const JsonValue& n) { return n.kind == JsonValue::Kind::Object; });
    int removed = 0;
    for (const auto& obj_path : objects) {
        JsonValue probe = golden;
        const auto keys = at_walk_path(probe, obj_path).object;  // copy of keys
        for (const auto& [key, _] : keys) {
            JsonValue doc = golden;
            at_walk_path(doc, obj_path).erase(key);
            expect_rejected(load_error(doc), "missing", "delete " + obj_path + "/" + key);
            ++removed;
        }
    }
    std::printf("  deleted fields: %d (each rejected)\n", removed);
    CHECK(removed > 700);
}

void test_unknown_fields_rejected(const JsonValue& golden) {
    JsonValue scratch = golden;
    const auto objects = paths_where(scratch, [](const JsonValue& n) { return n.kind == JsonValue::Kind::Object; });
    for (const auto& obj_path : objects) {
        JsonValue doc = golden;
        at_walk_path(doc, obj_path).set("__unknown_field__", JsonValue::make_int(0));
        expect_rejected(load_error(doc), "unknown field", "insert under " + obj_path);
    }
    std::printf("  objects probed with an unknown field: %zu (each rejected)\n", objects.size());
}

// Each scalar is swapped for values of other JSON kinds; every swap must fail.
// The replacements are chosen so no declared type accepts them (string fields
// that allow null are not offered null).
std::vector<JsonValue> wrong_kinds(const JsonValue& v) {
    switch (v.kind) {
        case JsonValue::Kind::Null:
            return {JsonValue::make_bool(false), JsonValue::make_int(0)};
        case JsonValue::Kind::Bool:
            return {JsonValue::make_int(v.boolean ? 1 : 0), JsonValue::make_string("true")};
        case JsonValue::Kind::Int:
            return {JsonValue::make_float(static_cast<double>(v.integer)),  // int written as 35.0
                    JsonValue::make_string(std::to_string(v.integer)), JsonValue::make_bool(true)};
        case JsonValue::Kind::Float:
            return {JsonValue::make_int(static_cast<std::int64_t>(v.number)),  // float written as an int literal
                    JsonValue::make_string("0.5"), JsonValue::make_bool(false)};
        case JsonValue::Kind::String:
            return {JsonValue::make_int(0), JsonValue::make_bool(true)};
        case JsonValue::Kind::Array:
            return {JsonValue::make_object()};
        case JsonValue::Kind::Object:
            return {JsonValue::make_array()};
    }
    return {};
}

void test_type_mismatch_rejected(const JsonValue& golden) {
    JsonValue scratch = golden;
    const auto nodes = paths_where(scratch, [](const JsonValue&) { return true; });
    int mutated = 0;
    for (const auto& p : nodes) {
        if (p.empty()) continue;
        const JsonValue original = at_walk_path(scratch, p);
        for (const auto& replacement : wrong_kinds(original)) {
            JsonValue doc = golden;
            at_walk_path(doc, p) = replacement;
            const std::string err = load_error(doc);
            expect_rejected(err, "expected", "retype " + p + " from " + sh::kind_name(original.kind) +
                                                 " to " + sh::kind_name(replacement.kind));
            ++mutated;
        }
    }
    std::printf("  type mutations: %d (each rejected)\n", mutated);
}

void test_non_finite_and_malformed_numbers(const std::string& golden_text) {
    const std::string anchor = "\"track_thresh\": 0.05,";
    CHECK(contains(golden_text, anchor));
    for (const char* bad : {"NaN", "-NaN", "Infinity", "-Infinity", "1e999", "-1e999", "inf"}) {
        std::string text = golden_text;
        text.replace(text.find(anchor), anchor.size(), std::string("\"track_thresh\": ") + bad + ",");
        expect_rejected(parse_error(text), "json syntax", std::string("track_thresh=") + bad);
    }
    const std::string int_anchor = "\"max_objects\": 2048,";
    for (const char* bad : {"9223372036854775808", "-9223372036854775809", "02048", "+2048"}) {
        std::string text = golden_text;
        text.replace(text.find(int_anchor), int_anchor.size(), std::string("\"max_objects\": ") + bad + ",");
        expect_rejected(parse_error(text), "json syntax", std::string("max_objects=") + bad);
    }
}

void test_duplicate_key_rejected(const std::string& golden_text) {
    const std::string anchor = "\"track_thresh\": 0.05,";
    std::string text = golden_text;
    text.replace(text.find(anchor), anchor.size(), anchor + " \"track_thresh\": 0.05,");
    expect_rejected(parse_error(text), "duplicate object key", "duplicate track_thresh");
}

struct Case {
    std::string path;
    std::function<void(JsonValue&)> mutate;
    std::string fragment;
};

std::function<void(JsonValue&)> set_to(JsonValue v) {
    return [v](JsonValue& n) { n = v; };
}

JsonValue per_seq(const char* source) {
    JsonValue o = JsonValue::make_object();
    o.set("per_sequence", JsonValue::make_string(source));
    return o;
}

void test_enum_range_and_semantic_checks(const JsonValue& golden) {
    const std::string tr = "native_params/GPUByteTracker/";
    const std::string calls = tr + "calls/";
    const std::vector<Case> cases = {
        {"schema", set_to(JsonValue::make_string("saccade.resolved_shipping_config/v2")), "unsupported schema"},
        {"schema", set_to(JsonValue::make_string("")), "unsupported schema"},
        {calls + "6/args/kalman_adapt_mode", set_to(JsonValue::make_int(5)), "not one of"},
        {calls + "6/args/kalman_adapt_mode", set_to(JsonValue::make_int(-1)), "not one of"},
        {calls + "2/args/bridge_anchor", set_to(JsonValue::make_int(3)), "not one of"},
        {calls + "7/args/occ_mode", set_to(JsonValue::make_int(2)), "not one of"},
        {calls + "6/args/track_thresh", set_to(JsonValue::make_float(1.5)), "outside"},
        {calls + "6/args/track_thresh", set_to(JsonValue::make_float(-0.1)), "outside"},
        {calls + "6/args/r_scale", set_to(JsonValue::make_float(0.0)), "outside"},
        {calls + "1/args/cos_threshold", set_to(JsonValue::make_float(-1.5)), "outside"},
        {calls + "10/args/lambda", set_to(JsonValue::make_float(0.5)), "outside"},
        {calls + "2/args/bridge_app_veto", set_to(JsonValue::make_float(-2.0)), "outside"},
        {tr + "constructor/max_objects", set_to(JsonValue::make_int(0)), "outside"},
        {tr + "constructor/max_objects", set_to(JsonValue::make_int(2147483648LL)), "outside"},
        {tr + "constructor/max_assoc", set_to(JsonValue::make_int(-4)), "outside"},
        {"native_params/GMC/constructor/quality_level", set_to(JsonValue::make_float(0.0)), "outside"},
        {"native_params/PerceptionPipelineConfig/fields/nms_threshold", set_to(JsonValue::make_float(1.01)), "outside"},
        {"native_params/PerceptionPipeline/constructor/reid_ptr", set_to(JsonValue::make_int(1)), "not one of"},
        {"native_params/PerceptionPipeline/constructor/config/ref", set_to(JsonValue::make_string("GMC")), "not one of"},
        {calls + "0/args/h", set_to(JsonValue::make_array({JsonValue::make_float(1.0)})), "expected null"},
        {calls + "4/args/w", set_to(per_seq("seqinfo.ini:Sequence.imHeight")), "must be the per-sequence imWidth"},
        {calls + "4/args/h", set_to(per_seq("seqinfo.ini:Sequence.frameRate")), "unknown per-sequence source"},
        {"source/per_sequence_values/0", set_to(per_seq("seqinfo.ini:Sequence.imHeight")), "per-sequence values"},
        {"host_params/detector/calls/0/args", [](JsonValue& n) { std::swap(n.array[0], n.array[1]); }, "per-sequence values"},
        // call order and count are schema
        {calls + "6/method", set_to(JsonValue::make_string("set_oao_params")), "call order is schema"},
        {"native_params/GPUByteTracker/calls", [](JsonValue& n) { std::swap(n.array[5], n.array[6]); }, "call order is schema"},
        {"native_params/GPUByteTracker/calls", [](JsonValue& n) { n.array.pop_back(); }, "missing call set_association_energy_params"},
        {"native_params/GPUByteTracker/calls", [](JsonValue& n) { n.array.push_back(n.array.back()); }, "unexpected extra call"},
        {"native_params/GMC/calls", [](JsonValue& n) { n.array.clear(); }, "missing call set_profiling_enabled"},
        // stage order: known stages, each exactly once
        {"host_params/stage_order/3", set_to(JsonValue::make_string("_run_bogus")), "unknown entry"},
        {"host_params/stage_order/1", set_to(JsonValue::make_string("_run_detect")), "duplicate entry"},
        {"host_params/stage_order", [](JsonValue& n) { n.array.pop_back(); }, "exactly once"},
        // boundary §5 B2/B4: tail steps without a shipping implementation stay off
        {"host_params/steps/tail.cheb_gr_or_occ_audit", set_to(JsonValue::make_bool(true)), "must be false"},
        {"host_params/steps/tail.post_lifecycle_merge", set_to(JsonValue::make_bool(true)), "must be false"},
        {"host_params/steps/tail.deferred_alias", set_to(JsonValue::make_bool(true)), "must be false"},
        {"host_params/steps/tail.tracklet_quality_filter", set_to(JsonValue::make_bool(true)), "must be false"},
        {"host_params/env/SACCADE_DETECT_BARRIER", set_to(JsonValue::make_string("bogus")), "not one of"},
        {"host_params/detector/contract/box_format", set_to(JsonValue::make_string("xywh")), "not one of"},
        {"host_params/detector/build/device", set_to(JsonValue::make_string("cpu")), "not one of"},
        {"host_params/detector/build/temporal_T_override", set_to(JsonValue::make_int(3)), "expected null"},
        {"host_params/detector/head_calls/set_head_compile", [](JsonValue& n) { n.array.push_back(n.array[0]); }, "exactly 1"},
        {"host_params/detector/detect_fn", set_to(JsonValue::make_string("")), "must not be empty"},
        {"host_params/initial_tracker_thresholds", [](JsonValue& n) { n.array.pop_back(); }, "exactly 3"},
        {"host_params/initial_tracker_thresholds/1", set_to(JsonValue::make_float(1.5)), "outside"},
        {"host_params/fp_hard_reject_score", set_to(JsonValue::make_float(-1.5)), "outside"},
        {"host_params/external_fp_rule_config/min_score", set_to(JsonValue::make_float(2.0)), "outside"},
        {"host_params/capacities/nms_fixed_n", set_to(JsonValue::make_int(0)), "outside"},
        {"source/preset_sha256", set_to(JsonValue::make_string("093B66ED124063F035AE9CF2A76E4F5426743CD819FB66E3E54994C97EA42CD1")), "hex"},
        {"source/preset_sha256", set_to(JsonValue::make_string("093b66ed")), "hex"},
        // native env: a defaulted getenv must carry its value; an unset one its effect
        {"native_env/SACCADE_ENABLE_DDA", set_to(per_seq("x")), "expected bool"},
        {"native_env/SACCADE_ASSOC_DUMP", set_to(JsonValue::make_bool(true)), "expected object"},
        {"native_env/SACCADE_COAST_SCORE_DECAY", set_to(JsonValue::make_float(1.5)), "outside"},
        {"native_env/SACCADE_GATE_ADAPT_R_MULT", set_to(JsonValue::make_float(0.0)), "outside"},
    };
    int n = 0;
    for (const auto& c : cases) {
        JsonValue doc = golden;
        c.mutate(at(doc, c.path));
        expect_rejected(load_error(doc), c.fragment, c.path);
        ++n;
    }
    std::printf("  enum/range/semantic cases: %d (each rejected)\n", n);
}

void test_env_does_not_affect_parse(const std::string& golden_text, const JsonValue& golden) {
    const std::string before = sh::canonical_snapshot(sh::parse_resolved_shipping_config(golden_text));
    std::vector<std::string> names;
    for (const auto& [k, _] : golden.find("native_env")->object) names.push_back(k);
    for (const auto& [k, _] : golden.find("host_params")->find("env")->object) names.push_back(k);
    for (const char* extra : {"SACCADE_KALMAN_ADAPT_MODE", "SACCADE_STABILITY_W", "SACCADE_PRESET",
                              "SACCADE_RESOLVED_CONFIG", "SACCADE_SHIPPING_CONFIG"}) {
        names.push_back(extra);
    }
    int set = 0;
    for (const char* value : {"0", "1", "garbage", "-1e9", ""}) {
        for (const auto& name : names) setenv(name.c_str(), value, 1);
        set += static_cast<int>(names.size());
        const std::string after = sh::canonical_snapshot(sh::parse_resolved_shipping_config(golden_text));
        CHECK(after == before);
        CHECK(after == golden_text);
    }
    for (const auto& name : names) unsetenv(name.c_str());
    std::printf("  env assignments with snapshot unchanged: %d\n", set);
}

void test_python_float_repr() {
    const struct {
        double v;
        const char* repr;
    } cases[] = {
        {0.0, "0.0"}, {-0.0, "-0.0"}, {1.0, "1.0"}, {0.1, "0.1"}, {6e-05, "6e-05"},
        {0.0001, "0.0001"}, {0.001, "0.001"}, {1e+16, "1e+16"},
        {1000000000000000.0, "1000000000000000.0"},
        {1.2345678901234568e+17, "1.2345678901234568e+17"},
        {0.07500000000000001, "0.07500000000000001"}, {2.5e-310, "2.5e-310"},
        {1.7976931348623157e+308, "1.7976931348623157e+308"}, {5e-324, "5e-324"},
        {100000.0, "100000.0"}, {1.5, "1.5"}, {-2.8, "-2.8"}, {1234.5678, "1234.5678"},
        {9999999999999998.0, "9999999999999998.0"}, {1e-05, "1e-05"}, {-1e-07, "-1e-07"},
        {0.30000000000000004, "0.30000000000000004"}, {1.152921504606847e+18, "1.152921504606847e+18"},
        {1e+22, "1e+22"}, {123.0, "123.0"},
    };
    for (const auto& c : cases) {
        const std::string got = sh::python_float_repr(c.v);
        if (got != c.repr) std::fprintf(stderr, "repr mismatch: want %s got %s\n", c.repr, got.c_str());
        CHECK(got == c.repr);
    }
    bool threw = false;
    try {
        sh::python_float_repr(std::nan(""));
    } catch (const ConfigError&) {
        threw = true;
    }
    CHECK(threw);
}

void test_python_json_layout() {
    // Python: json.dumps("a\"b\\c\n\r\t\b\f\x01\x7f é 😀 /")
    const auto s = JsonValue::make_string("a\"b\\c\n\r\t\b\f\x01\x7f \xc3\xa9 \xf0\x9f\x98\x80 /");
    CHECK(sh::dump_python_json(s) ==
          "\"a\\\"b\\\\c\\n\\r\\t\\b\\f\\u0001\\u007f \\u00e9 \\ud83d\\ude00 /\"");
    // Python: json.dumps({"a":[],"b":{},"c":[1,{"d":None}],"e":"x"}, indent=2)
    const auto doc = sh::parse_strict_json(R"({"a":[],"b":{},"c":[1,{"d":null}],"e":"x"})");
    CHECK(sh::dump_python_json(doc) ==
          "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    {\n      \"d\": null\n    }\n  ],\n  \"e\": \"x\"\n}");
    // Escapes round-trip through the reader.
    CHECK(sh::parse_strict_json(sh::dump_python_json(s)) == s);
}

void test_strict_syntax() {
    const char* bad[] = {
        "",
        "{} {}",
        "{\"a\": 1,}",
        "[1, 2,]",
        "{'a': 1}",
        "{\"a\": 1 // c\n}",
        "\xef\xbb\xbf{}",              // BOM
        "{\"a\": \"x\ny\"}",           // raw control character
        "{\"a\": \"\\ud800\"}",        // lone high surrogate
        "{\"a\": \"\\udc00\"}",        // lone low surrogate
        "{\"a\": \"\\x41\"}",          // invalid escape
        "{\"a\": \"\xff\"}",           // invalid UTF-8
        "{\"a\": \"\xc0\xaf\"}",       // overlong UTF-8
        "{\"a\": 1.}",
        "{\"a\": .5}",
        "{\"a\": 1e}",
        "{\"a\": tru}",
        "{\"a\" 1}",
        "{\"a\": [1 2]}",
        "\"unterminated",
    };
    for (const char* text : bad) {
        bool threw = false;
        try {
            sh::parse_strict_json(text);
        } catch (const ConfigError&) {
            threw = true;
        }
        if (!threw) std::fprintf(stderr, "accepted malformed JSON: %s\n", text);
        CHECK(threw);
    }
    const auto v = sh::parse_strict_json("{\"i\": -0, \"f\": 1E2, \"g\": 2.5e-3, \"u\": \"\\u00e9\\/\"}");
    CHECK(v.find("i")->kind == JsonValue::Kind::Int);
    CHECK(v.find("f")->kind == JsonValue::Kind::Float && v.find("f")->number == 100.0);
    CHECK(v.find("g")->kind == JsonValue::Kind::Float && v.find("g")->number == 0.0025);
    CHECK(v.find("u")->string == "\xc3\xa9/");
}

void test_missing_file() {
    bool threw = false;
    try {
        sh::load_resolved_shipping_config_file("/nonexistent/resolved.json");
    } catch (const ConfigError&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

static void run_all(const char* path, const std::string& golden_text) {
    const JsonValue golden = sh::parse_strict_json(golden_text);
    test_golden_round_trip(golden_text);
    test_every_field_is_required(golden);
    test_unknown_fields_rejected(golden);
    test_type_mismatch_rejected(golden);
    test_non_finite_and_malformed_numbers(golden_text);
    test_duplicate_key_rejected(golden_text);
    test_enum_range_and_semantic_checks(golden);
    test_env_does_not_affect_parse(golden_text, golden);
    test_python_float_repr();
    test_python_json_layout();
    test_strict_syntax();
    test_missing_file();
    CHECK(sh::canonical_snapshot(sh::load_resolved_shipping_config_file(path)) == golden_text);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <resolved config json>\n", argv[0]);
        return 2;
    }
    const std::string golden_text = read_file(argv[1]);
    if (golden_text.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    try {
        run_all(argv[1], golden_text);
    } catch (const ConfigError& e) {
        std::fprintf(stderr, "unexpected ConfigError: %s\n", e.what());
        return 1;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
