// Native MOT output path vs the Python oracle (#465 Phase B PR-6, U4).
// CUDA-free: runs in the CI job `shipping-config-loader`.
//
// Usage: saccade_shipping_mot_output_test <resolved config> <fixture>
//
// The fixture (tests/native/fixtures/shipping_mot_output.json) is the output of
// the Python functions themselves (scripts/model/
// render_shipping_mot_output_fixture.py; tests/unit/
// test_shipping_mot_output_oracle_pins.py keeps it fresh). Pins:
//   * SequenceIdMapper + emit_mot_lines give helpers.fast_emit_mot_lines'
//     lines (with a one-sequence GlobalTrackIdMapper) byte for byte;
//   * interpolate_tracklets gives post_merge.interpolate_tracklets' lines and
//     stats byte for byte;
//   * plan_sequence_output derives the fixture's headline interpolation
//     parameters from the committed config, and fails closed on any emit or
//     tail branch the output path does not implement;
//   * SequenceOutput's single-use lifecycle and the parser's refusal of
//     malformed lines.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/mot_output.hpp"
#include "saccade_shipping/post_detector_plan.hpp"
#include "saccade_shipping/resolved_config.hpp"
#include "saccade_shipping/sequence_output.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace sh = saccade::shipping;
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

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

const JsonValue& at(const JsonValue& obj, const char* key) {
    const JsonValue* v = obj.find(key);
    if (v == nullptr) throw std::runtime_error(std::string("fixture: missing key ") + key);
    return *v;
}

std::vector<float> floats(const JsonValue& arr) {
    std::vector<float> out;
    for (const JsonValue& v : arr.array) {
        const auto u = static_cast<std::uint32_t>(v.integer);
        float f;
        std::memcpy(&f, &u, sizeof f);
        out.push_back(f);
    }
    return out;
}

std::vector<std::string> strings(const JsonValue& arr) {
    std::vector<std::string> out;
    for (const JsonValue& v : arr.array) out.push_back(v.string);
    return out;
}

double number(const JsonValue& v) {
    return v.kind == JsonValue::Kind::Int ? static_cast<double>(v.integer) : v.number;
}

// Reports the first differing line of a case.
bool same_lines(const char* name, const std::vector<std::string>& got,
                const std::vector<std::string>& want) {
    if (got == want) return true;
    std::fprintf(stderr, "%s: %zu lines vs %zu expected\n", name, got.size(), want.size());
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        if (got[i] != want[i]) {
            std::fprintf(stderr, "  first difference at %zu:\n    got  %s\n    want %s\n", i,
                         got[i].c_str(), want[i].c_str());
            break;
        }
    }
    return false;
}

void check_emit_case(const JsonValue& c) {
    sh::SequenceIdMapper ids;
    std::vector<std::string> lines;
    for (const JsonValue& f : at(c, "frames").array) {
        const std::vector<float> boxes = floats(at(f, "boxes_bits"));
        const std::vector<float> scores = floats(at(f, "scores_bits"));
        std::vector<std::int32_t> local;
        for (const JsonValue& v : at(f, "ids").array) local.push_back(static_cast<std::int32_t>(v.integer));
        CHECK(boxes.size() == local.size() * 4 && scores.size() == local.size());
        sh::emit_mot_lines(lines, ids, at(f, "frame").integer, boxes.data(), scores.data(),
                           local.data(), local.size());
    }
    CHECK(same_lines(at(c, "name").string.c_str(), lines, strings(at(c, "expected_lines"))));
}

sh::InterpolationParams params_of(const JsonValue& p) {
    sh::InterpolationParams out;
    out.max_gap = at(p, "max_gap").integer;
    out.min_track_len = at(p, "min_track_len").integer;
    out.min_h = number(at(p, "min_h"));
    return out;
}

void check_interpolation_case(const JsonValue& c) {
    const std::string& name = at(c, "name").string;
    sh::InterpolationStats st;
    const std::vector<std::string> got =
        sh::interpolate_tracklets(strings(at(c, "lines")), params_of(at(c, "params")), &st);
    CHECK(same_lines(name.c_str(), got, strings(at(c, "expected_lines"))));
    const JsonValue& want = at(c, "expected_stats");
    const bool stats_eq = st.tracks_interpolated == at(want, "tracks_interpolated").integer &&
                          st.gaps_filled == at(want, "gaps_filled").integer &&
                          st.frames_added == at(want, "frames_added").integer;
    if (!stats_eq) std::fprintf(stderr, "%s: stats differ\n", name.c_str());
    CHECK(stats_eq);
}

void check_plan(const sh::ResolvedShippingConfig& cfg, const JsonValue& headline) {
    const sh::SequenceOutputPlan p = sh::plan_sequence_output(cfg);
    const sh::InterpolationParams want = params_of(headline);
    CHECK(p.interpolate);
    CHECK(p.write_output);
    CHECK(p.interpolation.max_gap == want.max_gap);
    CHECK(p.interpolation.min_track_len == want.min_track_len);
    CHECK(p.interpolation.min_h == want.min_h);
}

// Each mutation turns on an emit or tail branch the output path does not
// implement, or makes a step disagree with the cfg value it was evaluated from.
void check_fail_closed(const std::string& config_text) {
    struct Mutation {
        const char* section;
        const char* key;
        JsonValue value;
    };
    const std::vector<Mutation> mutations = {
        {"steps", "relink.semantic_relinker", JsonValue::make_bool(true)},
        {"steps", "emit.pipeline_relink", JsonValue::make_bool(true)},
        {"steps", "emit.id_stability_filter", JsonValue::make_bool(true)},
        {"steps", "emit.appearance_bank", JsonValue::make_bool(true)},
        {"steps", "emit.dynamic_reid", JsonValue::make_bool(true)},
        {"steps", "emit.fast_emit_reid_mode", JsonValue::make_bool(false)},
        {"steps", "emit.id_stability_kwarg", JsonValue::make_bool(true)},
        {"steps", "track.workbench", JsonValue::make_bool(true)},
        {"steps", "tail.interpolation", JsonValue::make_bool(false)},
        {"steps", "tail.write_output", JsonValue::make_bool(false)},
        {"cfg", "interpolate_tracklets", JsonValue::make_bool(false)},
        {"cfg", "latency_only", JsonValue::make_bool(true)},
    };
    for (const Mutation& m : mutations) {
        JsonValue doc = sh::parse_strict_json(config_text);
        JsonValue* section = doc.find("host_params")->find(m.section);
        JsonValue* slot = section == nullptr ? nullptr : section->find(m.key);
        CHECK(slot != nullptr);
        if (slot == nullptr) continue;
        *slot = m.value;
        bool refused = false;
        try {
            sh::plan_sequence_output(sh::load_resolved_shipping_config(doc));
        } catch (const sh::ConfigError&) {
            refused = true;
        }
        if (!refused) std::fprintf(stderr, "not refused: %s.%s\n", m.section, m.key);
        CHECK(refused);
    }
    // The workbench replaces the tracker path: the post-detector plan refuses it too.
    JsonValue doc = sh::parse_strict_json(config_text);
    *doc.find("host_params")->find("steps")->find("track.workbench") = JsonValue::make_bool(true);
    bool refused = false;
    try {
        sh::plan_post_detector(sh::load_resolved_shipping_config(doc));
    } catch (const sh::ConfigError&) {
        refused = true;
    }
    CHECK(refused);
}

bool throws(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void check_lifecycle_and_parser(const sh::ResolvedShippingConfig& cfg) {
    sh::SequenceOutput out(sh::plan_sequence_output(cfg));
    const float boxes[8] = {1.0f, 2.0f, 11.0f, 22.0f, 5.0f, 5.0f, 6.0f, 7.0f};
    const float scores[2] = {0.5f, 0.25f};
    const std::int32_t local[2] = {42, 7};
    out.add_frame(3, boxes, scores, local, 2);
    out.add_frame(4, boxes, scores, local + 1, 1);  // id 7 reappears: same output id
    CHECK(out.emitted_lines() == 3);
    CHECK(out.track_ids() == 2);
    const std::vector<std::string> lines = out.finish();
    CHECK(lines.size() == 3);
    CHECK(lines.at(0) == "3,1,1.00,2.00,10.00,20.00,0.5000,-1,-1,-1");
    CHECK(lines.at(2) == "4,2,1.00,2.00,10.00,20.00,0.5000,-1,-1,-1");
    CHECK(throws([&] { out.finish(); }));
    CHECK(throws([&] { out.add_frame(5, boxes, scores, local, 1); }));
    CHECK(sh::join_mot_lines(lines) == lines[0] + "\n" + lines[1] + "\n" + lines[2]);
    CHECK(sh::join_mot_lines({}).empty());

    const sh::InterpolationParams p{35, 1, 0.0};
    for (const char* bad : {"1,2,3.0,4.0,5.0,6.0", "1.0,2,3,4,5,6,0.5,-1,-1,-1",
                            "1,2,3,4,x,6,0.5,-1,-1,-1", "1,2,3,4,5,6,0.5 ,-1,-1,-1", ""}) {
        CHECK(throws([&] { sh::interpolate_tracklets({bad, "9,2,1,1,1,1,1,-1,-1,-1"}, p); }));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <resolved config> <fixture>\n", argv[0]);
        return 2;
    }
    try {
        const std::string config_text = read_file(argv[1]);
        const sh::ResolvedShippingConfig cfg = sh::parse_resolved_shipping_config(config_text);
        const JsonValue fixture = sh::parse_strict_json(read_file(argv[2]));
        CHECK(at(fixture, "format").string == "saccade.shipping_mot_output_fixture/v1");
        const JsonValue& emit = at(fixture, "emit_cases");
        const JsonValue& interp = at(fixture, "interpolation_cases");
        CHECK(emit.array.size() == 2);
        CHECK(interp.array.size() == 10);
        for (const JsonValue& c : emit.array) check_emit_case(c);
        for (const JsonValue& c : interp.array) check_interpolation_case(c);
        check_plan(cfg, at(fixture, "headline_interpolation"));
        check_fail_closed(config_text);
        check_lifecycle_and_parser(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
