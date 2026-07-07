#include "tracking/cheb_gr_online.hpp"
#include "tracking/cheb_gr_kreciprocal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace saccade {

namespace {

float get_float(const std::string& s) {
    return std::strtof(s.c_str(), nullptr);
}

int get_int(const std::string& s) {
    return static_cast<int>(std::strtof(s.c_str(), nullptr));
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

void mot_box(const MotRecord& r, float out[4]) {
    out[0] = r.x;
    out[1] = r.y;
    out[2] = r.x + r.w;
    out[3] = r.y + r.h;
}

void box_center(const float box[4], float out[2]) {
    out[0] = (box[0] + box[2]) * 0.5f;
    out[1] = (box[1] + box[3]) * 0.5f;
}

void tracklet_velocity(const std::vector<MotRecord>& records,
                       bool from_start,
                       int samples,
                       float out[2]) {
    out[0] = 0.0f;
    out[1] = 0.0f;
    int n = static_cast<int>(records.size());
    if (n < 2) return;
    int start_idx = from_start ? 0 : std::max(0, n - samples);
    int end_idx = from_start ? std::min(samples, n) : n;
    if (end_idx - start_idx < 2) return;
    const auto& first = records[start_idx];
    const auto& last = records[end_idx - 1];
    int dt = std::max(last.frame - first.frame, 1);
    float fc[4], lc[4], fc_c[2], lc_c[2];
    mot_box(first, fc);
    mot_box(last, lc);
    box_center(fc, fc_c);
    box_center(lc, lc_c);
    out[0] = (lc_c[0] - fc_c[0]) / static_cast<float>(dt);
    out[1] = (lc_c[1] - fc_c[1]) / static_cast<float>(dt);
}

}  // namespace

std::vector<MotRecord> parse_mot_lines(const std::vector<std::string>& lines) {
    std::vector<MotRecord> records;
    for (const auto& line : lines) {
        if (line.empty()) continue;
        auto parts = split(line, ',');
        if (parts.size() < 7) continue;
        MotRecord r;
        r.frame = get_int(parts[0]);
        r.track_id = get_int(parts[1]);
        r.x = get_float(parts[2]);
        r.y = get_float(parts[3]);
        r.w = get_float(parts[4]);
        r.h = get_float(parts[5]);
        r.score = get_float(parts[6]);
        for (size_t i = 7; i < parts.size(); ++i) {
            r.tail.push_back(parts[i]);
        }
        records.push_back(r);
    }
    return records;
}

std::vector<std::string> format_mot_records(
    const std::vector<MotRecord>& records) {
    std::vector<MotRecord> sorted = records;
    std::sort(sorted.begin(), sorted.end(),
              [](const MotRecord& a, const MotRecord& b) {
                  if (a.frame != b.frame) return a.frame < b.frame;
                  return a.track_id < b.track_id;
              });
    std::vector<std::string> lines;
    for (const auto& r : sorted) {
        std::ostringstream oss;
        oss << r.frame << "," << r.track_id << ",";
        oss.precision(2);
        oss << std::fixed << r.x << "," << r.y << "," << r.w << "," << r.h
            << ",";
        oss.precision(4);
        oss << r.score;
        if (r.tail.empty()) {
            oss << ",-1,-1,-1";
        } else {
            for (const auto& t : r.tail) oss << "," << t;
        }
        lines.push_back(oss.str());
    }
    return lines;
}

std::vector<OutputTracklet> build_output_tracklets(
    const std::vector<MotRecord>& records,
    int velocity_samples) {
    std::unordered_map<int, std::vector<MotRecord>> by_id;
    for (const auto& r : records) {
        by_id[r.track_id].push_back(r);
    }
    std::vector<OutputTracklet> tracklets;
    for (auto& [tid, items] : by_id) {
        std::sort(items.begin(), items.end(),
                  [](const MotRecord& a, const MotRecord& b) {
                      return a.frame < b.frame;
                  });
        OutputTracklet t;
        t.track_id = tid;
        t.records = items;
        t.start = items.front().frame;
        t.end = items.back().frame;
        mot_box(items.front(), t.start_box);
        mot_box(items.back(), t.end_box);
        tracklet_velocity(items, true, velocity_samples, t.start_velocity);
        tracklet_velocity(items, false, velocity_samples, t.end_velocity);
        float sum = 0.0f;
        for (const auto& r : items) sum += r.score;
        t.mean_score = sum / static_cast<float>(items.size());
        tracklets.push_back(t);
    }
    return tracklets;
}

std::pair<std::vector<std::string>, HandoverStats>
causal_handover_lines(
    const std::vector<std::string>& results_lines,
    const std::unordered_map<int, std::vector<float>>& head_embs,
    const std::unordered_map<int, std::vector<float>>& bank_embs,
    int embedding_dim,
    const HandoverParams& params,
    std::vector<HandoverDecision>* decision_log) {

    HandoverStats stats;

    if (!params.enabled || results_lines.empty()) {
        return {results_lines, stats};
    }

    auto records = parse_mot_lines(results_lines);
    auto tracklets = build_output_tracklets(records, 5);
    stats.ids_before = static_cast<int>(tracklets.size());

    if (tracklets.size() <= 1) {
        stats.ids_after = stats.ids_before;
        return {results_lines, stats};
    }

    std::unordered_map<int, int> label;
    for (const auto& t : tracklets) label[t.track_id] = t.track_id;

    auto deaths = tracklets;
    auto decisions = tracklets;
    std::sort(deaths.begin(), deaths.end(),
              [](const OutputTracklet& a, const OutputTracklet& b) {
                  if (a.end != b.end) return a.end < b.end;
                  if (a.start != b.start) return a.start < b.start;
                  return a.track_id < b.track_id;
              });
    std::sort(decisions.begin(), decisions.end(),
              [&params](const OutputTracklet& a, const OutputTracklet& b) {
                  int da = a.start + params.decide_n;
                  int db = b.start + params.decide_n;
                  if (da != db) return da < db;
                  if (a.start != b.start) return a.start < b.start;
                  return a.track_id < b.track_id;
              });

    std::unordered_map<int, const OutputTracklet*> archive;
    int di = 0;
    int nd = static_cast<int>(deaths.size());

    for (const auto& tb : decisions) {
        int t_decide = tb.start + params.decide_n;

        while (di < nd && deaths[di].end < t_decide) {
            archive[deaths[di].track_id] = &deaths[di];
            ++di;
        }

        {
            std::vector<int> expired;
            for (const auto& [tid, ta] : archive) {
                if (ta->end < tb.start - params.max_gap) {
                    expired.push_back(tid);
                }
            }
            for (int tid : expired) archive.erase(tid);
        }

        stats.events++;

        auto head_it = head_embs.find(tb.track_id);
        if (head_it == head_embs.end()) {
            stats.reject_no_head++;
            continue;
        }
        int head_n = static_cast<int>(head_it->second.size()) / embedding_dim;
        if (head_n == 0) {
            stats.reject_no_head++;
            continue;
        }
        if (head_n < params.min_head_samples) {
            stats.reject_min_head++;
            continue;
        }

        std::vector<const OutputTracklet*> in_graph;
        for (const auto& [tid, ta] : archive) {
            auto bank_it = bank_embs.find(tid);
            if (bank_it != bank_embs.end() &&
                !bank_it->second.empty()) {
                in_graph.push_back(ta);
            }
        }

        std::vector<const OutputTracklet*> cands;
        for (auto* ta : in_graph) {
            int gap = tb.start - ta->end;
            if (gap >= 1 && gap <= params.max_gap) {
                cands.push_back(ta);
            }
        }
        if (cands.empty()) continue;
        stats.events_with_candidates++;

        std::vector<int> graph_tids;
        graph_tids.push_back(tb.track_id);
        for (auto* ta : in_graph) graph_tids.push_back(ta->track_id);

        int total_rows = head_n;
        std::unordered_map<int, std::pair<int,int>> span_by_tid;
        for (auto* ta : in_graph) {
            auto it = bank_embs.find(ta->track_id);
            if (it != bank_embs.end()) {
                int n = static_cast<int>(it->second.size()) / embedding_dim;
                span_by_tid[ta->track_id] = {total_rows, total_rows + n};
                total_rows += n;
            }
        }

        float* full_data = new float[total_rows * embedding_dim];
        std::memcpy(full_data, head_it->second.data(),
                    head_n * embedding_dim * sizeof(float));
        for (auto* ta : in_graph) {
            auto it = bank_embs.find(ta->track_id);
            if (it != bank_embs.end()) {
                auto span = span_by_tid[ta->track_id];
                int n = span.second - span.first;
                std::memcpy(full_data + span.first * embedding_dim,
                            it->second.data(),
                            n * embedding_dim * sizeof(float));
            }
        }

        // full_data is row-major (one sample per contiguous embedding_dim
        // block); Eigen::MatrixXf defaults to column-major, so map row-major.
        Eigen::MatrixXf feats =
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>>(
                full_data, total_rows, embedding_dim);
        auto sdist = cheb_gr_kreciprocal_self(
            feats, params.cheb_lambda, params.k2, params.max_fwd,
            params.fuse_lambda);

        std::vector<std::pair<float, const OutputTracklet*>> scored;
        auto head_rows = sdist.topRows(head_n);
        for (auto* ta : cands) {
            auto it = span_by_tid.find(ta->track_id);
            if (it == span_by_tid.end()) continue;
            int lo = it->second.first;
            int hi = it->second.second;
            std::vector<float> block;
            for (int r = 0; r < head_n; ++r) {
                for (int c = lo; c < hi; ++c) {
                    block.push_back(head_rows(r, c));
                }
            }
            int k = std::max(1, static_cast<int>(
                std::round(params.pool_frac * static_cast<float>(block.size()))));
            std::nth_element(block.begin(), block.begin() + k, block.end());
            float cost = 0.0f;
            for (int i = 0; i < k; ++i) cost += block[i];
            cost /= static_cast<float>(k);
            scored.push_back({cost, ta});
        }
        delete[] full_data;

        if (scored.empty()) continue;

        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) return a.first < b.first;
                      return a.second->track_id < b.second->track_id;
                  });

        float best_cost = scored[0].first;
        auto* best = scored[0].second;
        float second_cost = scored.size() > 1
                                ? scored[1].first
                                : std::numeric_limits<float>::infinity();
        float observed_margin = second_cost - best_cost;

        std::string reason = "accepted";
        bool accepted = true;
        if (best_cost > params.max_cost) {
            reason = "cost";
            accepted = false;
            stats.reject_cost++;
        } else if (params.margin > 0.0f && observed_margin < params.margin) {
            reason = "margin";
            accepted = false;
            stats.reject_margin++;
        }

        if (decision_log) {
            HandoverDecision d;
            d.newborn_id = tb.track_id;
            d.newborn_start = tb.start;
            d.newborn_end = tb.end;
            d.candidate_id = best->track_id;
            auto lit = label.find(best->track_id);
            d.candidate_label =
                lit != label.end() ? lit->second : best->track_id;
            d.candidate_start = best->start;
            d.candidate_end = best->end;
            d.gap = tb.start - best->end;
            d.head_n = head_n;
            auto bank_it = bank_embs.find(best->track_id);
            d.bank_n = bank_it != bank_embs.end()
                           ? static_cast<int>(bank_it->second.size()) /
                                 embedding_dim
                           : 0;
            d.candidate_count = static_cast<int>(scored.size());
            d.best_cost = best_cost;
            d.second_cost =
                std::isfinite(second_cost) ? second_cost : -1.0f;
            d.margin = std::isfinite(observed_margin) ? observed_margin
                                                       : 999.0f;
            d.required_margin = params.margin;
            d.max_cost = params.max_cost;
            d.accepted = accepted;
            d.reason = reason;
            decision_log->push_back(d);
            stats.decisions_logged++;
        }

        if (!accepted) continue;

        label[tb.track_id] = label[best->track_id];
        archive.erase(best->track_id);
        stats.handovers++;
    }

    if (stats.handovers == 0) {
        stats.ids_after = stats.ids_before;
        return {results_lines, stats};
    }

    for (auto& r : records) {
        auto it = label.find(r.track_id);
        if (it != label.end()) r.track_id = it->second;
    }

    std::unordered_set<int> unique_labels;
    for (const auto& [_, v] : label) unique_labels.insert(v);
    stats.ids_after = static_cast<int>(unique_labels.size());

    return {format_mot_records(records), stats};
}

}  // namespace saccade
