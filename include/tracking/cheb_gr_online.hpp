#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace saccade {

struct MotRecord {
    int frame = 0;
    int track_id = 0;
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float score = 0.0f;
    std::vector<std::string> tail;
};

struct OutputTracklet {
    int track_id = 0;
    std::vector<MotRecord> records;
    int start = 0;
    int end = 0;
    float start_box[4] = {0, 0, 0, 0};
    float end_box[4] = {0, 0, 0, 0};
    float start_velocity[2] = {0, 0};
    float end_velocity[2] = {0, 0};
    float mean_score = 0.0f;
};

struct HandoverStats {
    int events = 0;
    int events_with_candidates = 0;
    int handovers = 0;
    int reject_no_head = 0;
    int reject_min_head = 0;
    int reject_cost = 0;
    int reject_margin = 0;
    int decisions_logged = 0;
    int ids_before = 0;
    int ids_after = 0;
};

struct HandoverParams {
    bool enabled = false;
    float max_cost = 0.45f;
    int max_gap = 60;
    int decide_n = 5;
    int min_head_samples = 1;
    float margin = 0.0f;
    float pool_frac = 0.3f;
    float cheb_lambda = 2.0f;
    int k2 = 6;
    int max_fwd = 50;
    float fuse_lambda = 0.3f;
};

struct HandoverDecision {
    int newborn_id = 0;
    int newborn_start = 0;
    int newborn_end = 0;
    int candidate_id = 0;
    int candidate_label = 0;
    int candidate_start = 0;
    int candidate_end = 0;
    int gap = 0;
    int head_n = 0;
    int bank_n = 0;
    int candidate_count = 0;
    float best_cost = 0.0f;
    float second_cost = 0.0f;
    float margin = 0.0f;
    float required_margin = 0.0f;
    float max_cost = 0.0f;
    bool accepted = false;
    std::string reason;
};

std::vector<MotRecord> parse_mot_lines(const std::vector<std::string>& lines);

std::vector<std::string> format_mot_records(const std::vector<MotRecord>& records);

std::vector<OutputTracklet> build_output_tracklets(
    const std::vector<MotRecord>& records, int velocity_samples);

std::pair<std::vector<std::string>, HandoverStats> causal_handover_lines(
    const std::vector<std::string>& results_lines,
    const std::unordered_map<int, std::vector<float>>& head_embs,
    const std::unordered_map<int, std::vector<float>>& bank_embs,
    int embedding_dim,
    const HandoverParams& params,
    std::vector<HandoverDecision>* decision_log = nullptr);

}  // namespace saccade
