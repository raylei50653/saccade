#include "tracking/cheb_gr_online.hpp"
#include "tracking/cheb_gr_kreciprocal.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << msg << std::endl;
    std::exit(1);
}

void expect_true(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expect_eq(int a, int b, const std::string& msg) {
    if (a != b) {
        std::ostringstream oss;
        oss << msg << ": expected " << b << " got " << a;
        fail(oss.str());
    }
}

std::vector<float> normed(std::mt19937& rng, int n, int d,
                          const std::vector<float>& center) {
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::vector<float> data(n * d);
    for (int i = 0; i < n * d; ++i) {
        data[i] = center[i % d] + dist(rng);
    }
    for (int i = 0; i < n; ++i) {
        float norm = 0.0f;
        for (int j = 0; j < d; ++j) norm += data[i * d + j] * data[i * d + j];
        norm = std::sqrt(norm);
        if (norm > 1e-12f)
            for (int j = 0; j < d; ++j) data[i * d + j] /= norm;
    }
    return data;
}

void track_lines(std::vector<std::string>& lines, int tid, int start_fr,
                 int end_fr, float x = 10.0f) {
    for (int fr = start_fr; fr <= end_fr; ++fr) {
        std::ostringstream oss;
        oss << fr << "," << tid << "," << x << ",10,20,40,0.9,-1,-1,-1";
        lines.push_back(oss.str());
    }
}

std::unordered_set<int> extract_ids(
    const std::vector<std::string>& lines) {
    std::unordered_set<int> ids;
    for (const auto& ln : lines) {
        size_t pos = ln.find(',');
        if (pos == std::string::npos) continue;
        size_t pos2 = ln.find(',', pos + 1);
        if (pos2 == std::string::npos) continue;
        ids.insert(std::stoi(ln.substr(pos + 1, pos2 - pos - 1)));
    }
    return ids;
}

void test_parse_and_format_roundtrip() {
    std::vector<std::string> lines = {
        "1,5,10.0,20.0,30.0,40.0,0.9000,-1,-1,-1",
        "2,5,11.0,21.0,30.0,40.0,0.8500,-1,-1,-1"};
    auto records = saccade::parse_mot_lines(lines);
    expect_eq(static_cast<int>(records.size()), 2, "parse count");
    expect_eq(records[0].frame, 1, "frame");
    expect_eq(records[0].track_id, 5, "track_id");

    auto formatted = saccade::format_mot_records(records);
    expect_true(formatted.size() == 2, "format count");
    std::cout << "  [PASS] test_parse_and_format_roundtrip" << std::endl;
}

void test_disabled_is_passthrough() {
    std::vector<std::string> lines = {
        "1,1,10,10,20,40,0.9,-1,-1,-1"};
    saccade::HandoverParams params;
    params.enabled = false;
    auto [out, stats] = saccade::causal_handover_lines(
        lines, {}, {}, 32, params);
    expect_true(out == lines, "output unchanged when disabled");
    expect_eq(stats.handovers, 0, "no handovers when disabled");
    std::cout << "  [PASS] test_disabled_is_passthrough" << std::endl;
}

void test_build_tracklets() {
    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 5);
    track_lines(lines, 2, 3, 8);
    auto records = saccade::parse_mot_lines(lines);
    auto tracklets = saccade::build_output_tracklets(records, 5);
    expect_eq(static_cast<int>(tracklets.size()), 2, "tracklet count");
    std::sort(tracklets.begin(), tracklets.end(),
              [](const auto& a, const auto& b) {
                  return a.track_id < b.track_id;
              });
    expect_eq(tracklets[0].track_id, 1, "tracklet 0 id");
    expect_eq(tracklets[0].start, 1, "tracklet 0 start");
    expect_eq(tracklets[0].end, 5, "tracklet 0 end");
    expect_eq(tracklets[1].track_id, 2, "tracklet 1 id");
    expect_eq(tracklets[1].start, 3, "tracklet 1 start");
    expect_eq(tracklets[1].end, 8, "tracklet 1 end");
    std::cout << "  [PASS] test_build_tracklets" << std::endl;
}

void test_simple_handover_relabels_newborn() {
    int d = 32;
    std::mt19937 rng(0);
    std::vector<float> c0(d), c1(d);
    std::normal_distribution<float> init_dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) { c0[i] = init_dist(rng); c1[i] = init_dist(rng); }

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 2, 16, 30);
    track_lines(lines, 3, 1, 30, 200.0f);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c0);
    head[3] = normed(rng, 3, d, c1);

    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);
    bank[3] = normed(rng, 8, d, c1);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.max_fwd = 0;

    auto [out, stats] =
        saccade::causal_handover_lines(lines, head, bank, d, params);
    expect_eq(stats.handovers, 1, "should have 1 handover");
    expect_eq(stats.ids_after, 2, "should have 2 identities after");
    auto ids = extract_ids(out);
    expect_true(ids.count(1) && ids.count(3) && ids.size() == 2,
                "ids should be {1, 3}");
    std::cout << "  [PASS] test_simple_handover_relabels_newborn" << std::endl;
}

void test_gap_gate_blocks_far_and_overlapping() {
    int d = 32;
    std::mt19937 rng(1);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c0);
    head[4] = normed(rng, 3, d, c0);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);
    bank[3] = normed(rng, 8, d, c0);
    bank[2] = normed(rng, 8, d, c0);
    bank[4] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.max_gap = 60;
    params.max_fwd = 0;

    {
        std::vector<std::string> lines;
        track_lines(lines, 1, 1, 10);
        track_lines(lines, 2, 81, 90);
        auto [out, stats] =
            saccade::causal_handover_lines(lines, head, bank, d, params);
        expect_eq(stats.handovers, 0, "gap beyond max_gap should block");
    }

    {
        std::vector<std::string> lines;
        track_lines(lines, 3, 1, 20);
        track_lines(lines, 4, 19, 30);
        auto [out, stats] =
            saccade::causal_handover_lines(lines, head, bank, d, params);
        expect_eq(stats.handovers, 0, "overlap should block");
    }
    std::cout << "  [PASS] test_gap_gate_blocks_far_and_overlapping"
              << std::endl;
}

void test_identity_revived_at_most_once() {
    int d = 32;
    std::mt19937 rng(2);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 2, 16, 26);
    track_lines(lines, 3, 20, 30);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c0);
    head[3] = normed(rng, 3, d, c0);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);
    bank[2] = normed(rng, 8, d, c0);
    bank[3] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.max_fwd = 0;

    auto [out, stats] =
        saccade::causal_handover_lines(lines, head, bank, d, params);
    expect_eq(stats.handovers, 1, "should revive at most once");
    auto ids = extract_ids(out);
    expect_true(ids.count(1) && ids.size() == 2,
                "id 1 should be present, total 2");
    std::cout << "  [PASS] test_identity_revived_at_most_once" << std::endl;
}

void test_chain_handover_follows_labels() {
    int d = 32;
    std::mt19937 rng(3);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 2, 15, 25);
    track_lines(lines, 3, 30, 40);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c0);
    head[3] = normed(rng, 3, d, c0);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);
    bank[2] = normed(rng, 8, d, c0);
    bank[3] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.max_fwd = 0;

    auto [out, stats] =
        saccade::causal_handover_lines(lines, head, bank, d, params);
    expect_eq(stats.handovers, 2, "chain should have 2 handovers");
    auto ids = extract_ids(out);
    expect_eq(static_cast<int>(ids.size()), 1, "all merged to 1 identity");
    std::cout << "  [PASS] test_chain_handover_follows_labels" << std::endl;
}

void test_causality_no_future_candidates() {
    int d = 32;
    std::mt19937 rng(4);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 20);
    track_lines(lines, 2, 10, 30, 200.0f);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c0);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.99f;
    params.max_fwd = 0;

    auto [out, stats] =
        saccade::causal_handover_lines(lines, head, bank, d, params);
    expect_eq(stats.handovers, 0,
              "track dying after birth should not be candidate");
    std::cout << "  [PASS] test_causality_no_future_candidates" << std::endl;
}

void test_cost_gate_rejects_different_identity() {
    int d = 32;
    std::mt19937 rng(5);
    std::vector<float> c0(d), c1(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) { c0[i] = dist(rng); c1[i] = dist(rng); }

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 2, 16, 26);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c1);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.3f;
    params.max_fwd = 0;

    auto [out, stats] =
        saccade::causal_handover_lines(lines, head, bank, d, params);
    expect_eq(stats.handovers, 0, "cost gate should reject different id");
    auto ids = extract_ids(out);
    expect_true(ids.size() == 2, "should keep both ids");
    std::cout << "  [PASS] test_cost_gate_rejects_different_identity"
              << std::endl;
}

void test_newborn_without_head_samples_keeps_id() {
    int d = 32;
    std::mt19937 rng(6);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 2, 16, 26);

    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.max_fwd = 0;

    auto [out, stats] =
        saccade::causal_handover_lines(lines, {}, bank, d, params);
    expect_eq(stats.handovers, 0, "no head samples should block handover");
    auto ids = extract_ids(out);
    expect_true(ids.count(1) && ids.count(2), "should keep both ids");
    std::cout << "  [PASS] test_newborn_without_head_samples_keeps_id"
              << std::endl;
}

void test_min_head_gate_blocks_single_head_sample() {
    int d = 32;
    std::mt19937 rng(7);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 2, 16, 26);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 1, d, c0);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.min_head_samples = 2;
    params.max_fwd = 0;

    auto [out, stats] =
        saccade::causal_handover_lines(lines, head, bank, d, params);
    expect_eq(stats.handovers, 0, "min_head should block with only 1 sample");
    expect_eq(stats.reject_min_head, 1, "should record min_head reject");
    std::cout << "  [PASS] test_min_head_gate_blocks_single_head_sample"
              << std::endl;
}

void test_margin_gate_rejects_ambiguous() {
    int d = 32;
    std::mt19937 rng(8);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 3, 3, 13);
    track_lines(lines, 2, 16, 26);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c0);
    auto shared_bank = normed(rng, 8, d, c0);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = shared_bank;
    bank[3] = shared_bank;

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.margin = 0.0f;
    params.max_fwd = 0;

    std::vector<saccade::HandoverDecision> decision_log;
    auto [out_loose, stats_loose] = saccade::causal_handover_lines(
        lines, head, bank, d, params, &decision_log);
    float obs = decision_log.empty() ? 0.0f : decision_log[0].margin;

    params.margin = std::max(obs * 1.1f, 1e-6f);
    auto [out, stats] =
        saccade::causal_handover_lines(lines, head, bank, d, params);
    expect_eq(stats.handovers, 0, "margin gate should reject ambiguous");
    expect_eq(stats.reject_margin, 1, "should record margin reject");
    auto ids = extract_ids(out);
    expect_eq(static_cast<int>(ids.size()), 3, "should keep all 3 ids");

    std::cout << "  [PASS] test_margin_gate_rejects_ambiguous" << std::endl;
}

void test_decision_log() {
    int d = 32;
    std::mt19937 rng(9);
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);

    std::vector<std::string> lines;
    track_lines(lines, 1, 1, 10);
    track_lines(lines, 2, 16, 26);

    std::unordered_map<int, std::vector<float>> head;
    head[2] = normed(rng, 3, d, c0);
    std::unordered_map<int, std::vector<float>> bank;
    bank[1] = normed(rng, 8, d, c0);

    saccade::HandoverParams params;
    params.enabled = true;
    params.max_cost = 0.9f;
    params.max_fwd = 0;

    std::vector<saccade::HandoverDecision> decision_log;
    auto [out, stats] = saccade::causal_handover_lines(
        lines, head, bank, d, params, &decision_log);
    expect_eq(stats.handovers, 1, "should accept");
    expect_eq(stats.decisions_logged, 1, "should log 1 decision");
    expect_eq(decision_log[0].newborn_id, 2, "log newborn_id");
    expect_eq(decision_log[0].candidate_id, 1, "log candidate_id");
    expect_true(decision_log[0].accepted, "log accepted");
    expect_true(decision_log[0].reason == "accepted", "log reason");
    std::cout << "  [PASS] test_decision_log" << std::endl;
}

void test_cheb_gr_kreciprocal_shape() {
    std::mt19937 rng(42);
    int d = 16, n = 7;
    std::vector<float> c0(d);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < d; ++i) c0[i] = dist(rng);
    auto emb = normed(rng, n, d, c0);

    Eigen::Map<Eigen::MatrixXf> feats(emb.data(), n, d);
    auto result = saccade::cheb_gr_kreciprocal(feats, feats, 2.0f, 4, 5, 0.3f);
    expect_eq(static_cast<int>(result.rows()), n, "krecip rows");
    expect_eq(static_cast<int>(result.cols()), n, "krecip cols");
    expect_true(result.array().isFinite().all(), "krecip all finite");
    std::cout << "  [PASS] test_cheb_gr_kreciprocal_shape" << std::endl;
}

}  // namespace

int main() {
    std::cout << "test_cheb_gr_online:" << std::endl;
    test_parse_and_format_roundtrip();
    test_build_tracklets();
    test_disabled_is_passthrough();
    test_simple_handover_relabels_newborn();
    test_gap_gate_blocks_far_and_overlapping();
    test_identity_revived_at_most_once();
    test_chain_handover_follows_labels();
    test_causality_no_future_candidates();
    test_cost_gate_rejects_different_identity();
    test_newborn_without_head_samples_keeps_id();
    test_min_head_gate_blocks_single_head_sample();
    test_margin_gate_rejects_ambiguous();
    test_decision_log();
    test_cheb_gr_kreciprocal_shape();
    std::cout << "All tests passed." << std::endl;
    return 0;
}
