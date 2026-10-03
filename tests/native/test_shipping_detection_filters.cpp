// Host-side detection filters of the native post-detector host vs the Python
// oracle (#465 Phase B PR-5, U3a). CUDA-free: runs in the CI job
// `shipping-config-loader`.
//
// Usage: saccade_shipping_detection_filters_test <resolved config> <fixture>
//
// The fixture (tests/native/fixtures/shipping_detection_filters.json) is the
// output of the Python functions themselves (scripts/model/
// render_shipping_detection_filters_fixture.py; tests/unit/
// test_post_detector_host_oracle_pins.py keeps it fresh). Pins:
//   * apply_external_fp_rule keeps exactly the rows _apply_external_fp_filter
//     (mode "rule", penalty off) keeps, in order, scores unchanged;
//   * fp_hard_reject_mask / apply_fp_hard_filter give the oracle's mask and
//     masked_fill scores bit for bit;
//   * plan_post_detector derives from the committed config exactly the float32
//     thresholds of the fixture's `headline` case;
//   * plan_post_detector fails closed on a gate the host does not implement.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "saccade_shipping/post_detector_plan.hpp"
#include "saccade_shipping/resolved_config.hpp"
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

float from_bits(std::int64_t bits) {
    const auto u = static_cast<std::uint32_t>(bits);
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::uint32_t to_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

std::vector<float> floats(const JsonValue& arr) {
    std::vector<float> out;
    for (const JsonValue& v : arr.array) out.push_back(from_bits(v.integer));
    return out;
}

float f(const JsonValue& group, const char* key) { return from_bits(at(group, key).integer); }

sh::ExternalFpRule rule_of(const JsonValue& t) {
    const JsonValue& e = at(t, "external_fp");
    return {f(e, "max_score"),  f(e, "min_score"),     f(e, "low_score"), f(e, "medium_score"),
            f(e, "min_height"), f(e, "medium_height"), f(e, "min_aspect")};
}

sh::FpHardFilter hard_of(const JsonValue& t) {
    const JsonValue& h = at(t, "fp_hard");
    return {f(h, "min_score"), f(h, "max_suspicious_area"), f(h, "max_suspicious_score"),
            f(h, "reject_score")};
}

bool same_bits(float a, float b) { return to_bits(a) == to_bits(b); }

void check_case(const JsonValue& c) {
    const std::string name = at(c, "name").string;
    const JsonValue& t = at(c, "thresholds_bits");
    const JsonValue& exp = at(c, "expected");

    sh::DetectionRows rows;
    rows.boxes = floats(at(c, "boxes_bits"));
    rows.scores = floats(at(c, "scores_bits"));
    for (std::size_t i = 0; i < rows.scores.size(); ++i) {
        rows.classes.push_back(static_cast<std::int32_t>(i));  // row index, as the generator
    }
    CHECK(rows.boxes.size() == 4 * rows.scores.size());

    // External FP rule filter: survivors and their order.
    const sh::DetectionRows kept = sh::apply_external_fp_rule(rows, rule_of(t));
    const JsonValue& want_kept = at(exp, "external_fp_kept_rows");
    CHECK(kept.size() == want_kept.array.size());
    int kept_mismatch = 0;
    for (std::size_t i = 0; i < kept.size() && i < want_kept.array.size(); ++i) {
        const auto row = static_cast<std::size_t>(want_kept.array[i].integer);
        if (kept.classes[i] != static_cast<std::int32_t>(row) ||
            !same_bits(kept.scores[i], rows.scores[row]) ||
            std::memcmp(&kept.boxes[i * 4], &rows.boxes[row * 4], 4 * sizeof(float)) != 0) {
            ++kept_mismatch;
        }
    }
    CHECK(kept_mismatch == 0);

    // FP hard filter: mask and masked_fill scores.
    const sh::FpHardFilter hard = hard_of(t);
    const std::vector<bool> reject = sh::fp_hard_reject_mask(rows, hard);
    const JsonValue& want_reject = at(exp, "fp_hard_reject");
    CHECK(reject.size() == want_reject.array.size());
    int reject_mismatch = 0;
    for (std::size_t i = 0; i < reject.size() && i < want_reject.array.size(); ++i) {
        if (reject[i] != want_reject.array[i].boolean) ++reject_mismatch;
    }
    CHECK(reject_mismatch == 0);

    sh::DetectionRows filled = rows;
    sh::apply_fp_hard_filter(filled, hard);
    const std::vector<float> want_scores = floats(at(exp, "fp_hard_scores_bits"));
    CHECK(filled.scores.size() == want_scores.size());
    int score_mismatch = 0;
    for (std::size_t i = 0; i < filled.scores.size() && i < want_scores.size(); ++i) {
        if (!same_bits(filled.scores[i], want_scores[i])) ++score_mismatch;
    }
    CHECK(score_mismatch == 0);

    std::printf("[%s] %zu rows: external FP kept %zu (mismatch %d), fp_hard rejected %zu "
                "(mask mismatch %d, score mismatch %d)\n",
                name.c_str(), rows.size(), kept.size(), kept_mismatch,
                static_cast<std::size_t>(std::count(reject.begin(), reject.end(), true)),
                reject_mismatch, score_mismatch);
}

void check_plan_thresholds(const sh::ResolvedShippingConfig& cfg, const JsonValue& headline) {
    const sh::PostDetectorPlan p = sh::plan_post_detector(cfg);
    const JsonValue& t = at(headline, "thresholds_bits");
    const sh::ExternalFpRule r = rule_of(t);
    CHECK(p.external_fp);
    CHECK(same_bits(p.external_fp_rule.max_score, r.max_score));
    CHECK(same_bits(p.external_fp_rule.min_score, r.min_score));
    CHECK(same_bits(p.external_fp_rule.low_score, r.low_score));
    CHECK(same_bits(p.external_fp_rule.medium_score, r.medium_score));
    CHECK(same_bits(p.external_fp_rule.min_height, r.min_height));
    CHECK(same_bits(p.external_fp_rule.medium_height, r.medium_height));
    CHECK(same_bits(p.external_fp_rule.min_aspect, r.min_aspect));
    const sh::FpHardFilter h = hard_of(t);
    CHECK(p.fp_hard);
    CHECK(same_bits(p.fp_hard_filter.min_score, h.min_score));
    CHECK(same_bits(p.fp_hard_filter.max_suspicious_area, h.max_suspicious_area));
    CHECK(same_bits(p.fp_hard_filter.max_suspicious_score, h.max_suspicious_score));
    CHECK(same_bits(p.fp_hard_filter.reject_score, h.reject_score));
    CHECK(p.tracker_pre_roll == sh::kGraphedTrackerUpdatePreRoll);
    CHECK(p.nms_fixed_n == p.max_assoc);
}

// Each mutation turns on an oracle branch the host does not implement.
void check_fail_closed(const std::string& config_text) {
    struct Mutation {
        const char* section;
        const char* key;
        JsonValue value;
    };
    const std::vector<Mutation> mutations = {
        {"cfg", "stage2_quality_gate", JsonValue::make_bool(true)},
        {"cfg", "external_fp_filter_mode", JsonValue::make_string("logistic")},
        {"cfg", "external_fp_penalty", JsonValue::make_float(0.5)},
        {"cfg", "narrow_person_score_bonus", JsonValue::make_float(0.05)},
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
            sh::plan_post_detector(sh::load_resolved_shipping_config(doc));
        } catch (const sh::ConfigError&) {
            refused = true;
        }
        if (!refused) std::fprintf(stderr, "not refused: %s.%s\n", m.section, m.key);
        CHECK(refused);
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
        CHECK(at(fixture, "format").string == "saccade.shipping_detection_filters_fixture/v1");
        const JsonValue& cases = at(fixture, "cases");
        CHECK(cases.array.size() == 2);
        for (const JsonValue& c : cases.array) check_case(c);
        check_plan_thresholds(cfg, cases.array.at(0));
        check_fail_closed(config_text);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
