#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "tracking/tracker_gpu.hpp"
#include "tracking/gmc.hpp"
#include "tracking/pipeline.hpp"
#include "perception/feature_extractor.hpp"
#include "perception/preprocessor.hpp"
#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace saccade;

namespace {

float compute_iou(const float* a, const float* b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-6f);
}

py::tuple merge_cross_tile_duplicates_cpu(
    py::array_t<float, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    py::array_t<int, py::array::c_style | py::array::forcecast> classes,
    float iou_threshold,
    float center_threshold,
    float area_ratio_threshold
) {
    if (boxes.ndim() != 2 || boxes.shape(1) != 4) {
        throw std::invalid_argument("boxes must have shape [N, 4]");
    }
    if (scores.ndim() != 1 || classes.ndim() != 1) {
        throw std::invalid_argument("scores and classes must have shape [N]");
    }
    const ssize_t num = boxes.shape(0);
    if (scores.shape(0) != num || classes.shape(0) != num) {
        throw std::invalid_argument("boxes, scores, and classes must have the same length");
    }

    auto boxes_in = boxes.unchecked<2>();
    auto scores_in = scores.unchecked<1>();
    auto classes_in = classes.unchecked<1>();

    if (num <= 1) {
        return py::make_tuple(boxes, scores, classes);
    }

    std::vector<int> order(static_cast<size_t>(num));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores_in(lhs) > scores_in(rhs);
    });

    std::vector<unsigned char> consumed(static_cast<size_t>(num), 0);
    std::vector<float> out_boxes;
    std::vector<float> out_scores;
    std::vector<int> out_classes;
    out_boxes.reserve(static_cast<size_t>(num) * 4);
    out_scores.reserve(static_cast<size_t>(num));
    out_classes.reserve(static_cast<size_t>(num));

    for (int anchor_idx : order) {
        if (consumed[static_cast<size_t>(anchor_idx)]) {
            continue;
        }

        const int anchor_class = classes_in(anchor_idx);
        const float anchor_box[4] = {
            boxes_in(anchor_idx, 0),
            boxes_in(anchor_idx, 1),
            boxes_in(anchor_idx, 2),
            boxes_in(anchor_idx, 3),
        };
        const float anchor_w = std::max(1e-6f, anchor_box[2] - anchor_box[0]);
        const float anchor_h = std::max(1e-6f, anchor_box[3] - anchor_box[1]);
        const float anchor_area = anchor_w * anchor_h;
        const float anchor_cx = 0.5f * (anchor_box[0] + anchor_box[2]);
        const float anchor_cy = 0.5f * (anchor_box[1] + anchor_box[3]);

        std::vector<int> cluster_indices;
        cluster_indices.reserve(4);
        for (int candidate_idx : order) {
            if (consumed[static_cast<size_t>(candidate_idx)]) {
                continue;
            }
            if (classes_in(candidate_idx) != anchor_class) {
                continue;
            }

            const float candidate_box[4] = {
                boxes_in(candidate_idx, 0),
                boxes_in(candidate_idx, 1),
                boxes_in(candidate_idx, 2),
                boxes_in(candidate_idx, 3),
            };
            const float iou = compute_iou(anchor_box, candidate_box);
            const float candidate_w = std::max(1e-6f, candidate_box[2] - candidate_box[0]);
            const float candidate_h = std::max(1e-6f, candidate_box[3] - candidate_box[1]);
            const float candidate_area = candidate_w * candidate_h;
            const float min_w = std::min(anchor_w, candidate_w);
            const float min_h = std::min(anchor_h, candidate_h);
            const float center_gate = std::sqrt(min_w * min_w + min_h * min_h) * center_threshold;
            const float candidate_cx = 0.5f * (candidate_box[0] + candidate_box[2]);
            const float candidate_cy = 0.5f * (candidate_box[1] + candidate_box[3]);
            const float center_dx = candidate_cx - anchor_cx;
            const float center_dy = candidate_cy - anchor_cy;
            const float center_dist = std::sqrt(center_dx * center_dx + center_dy * center_dy);
            const float area_ratio = std::min(
                candidate_area / std::max(anchor_area, 1e-6f),
                anchor_area / std::max(candidate_area, 1e-6f)
            );

            if (iou >= iou_threshold || (center_dist <= center_gate && area_ratio >= area_ratio_threshold)) {
                cluster_indices.push_back(candidate_idx);
            }
        }

        float weight_sum = 0.0f;
        float fused_box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float fused_score = 0.0f;
        for (int idx : cluster_indices) {
            const float score = scores_in(idx);
            weight_sum += score;
            fused_score = std::max(fused_score, score);
            for (int k = 0; k < 4; ++k) {
                fused_box[k] += boxes_in(idx, k) * score;
            }
            consumed[static_cast<size_t>(idx)] = 1;
        }

        const float inv_weight_sum = 1.0f / std::max(weight_sum, 1e-6f);
        for (float& coord : fused_box) {
            coord *= inv_weight_sum;
            out_boxes.push_back(coord);
        }
        out_scores.push_back(fused_score);
        out_classes.push_back(anchor_class);
    }

    const ssize_t out_num = static_cast<ssize_t>(out_scores.size());
    py::array_t<float> out_boxes_arr({out_num, static_cast<ssize_t>(4)});
    py::array_t<float> out_scores_arr({out_num});
    py::array_t<int> out_classes_arr({out_num});

    auto out_boxes_mut = out_boxes_arr.mutable_unchecked<2>();
    auto out_scores_mut = out_scores_arr.mutable_unchecked<1>();
    auto out_classes_mut = out_classes_arr.mutable_unchecked<1>();
    for (ssize_t i = 0; i < out_num; ++i) {
        for (ssize_t k = 0; k < 4; ++k) {
            out_boxes_mut(i, k) = out_boxes[static_cast<size_t>(i) * 4 + static_cast<size_t>(k)];
        }
        out_scores_mut(i) = out_scores[static_cast<size_t>(i)];
        out_classes_mut(i) = out_classes[static_cast<size_t>(i)];
    }

    return py::make_tuple(out_boxes_arr, out_scores_arr, out_classes_arr);
}

py::tuple filter_detections_cpu(
    py::array_t<float, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    py::array_t<int, py::array::c_style | py::array::forcecast> classes,
    float score_threshold,
    bool track_person_only,
    int person_class,
    bool is_tiled,
    int frame_w,
    int frame_h,
    bool person_geometry_prior,
    bool geometry_suspect_support,
    float person_min_height_ratio,
    float person_min_aspect,
    float person_max_aspect,
    float person_min_area_ratio,
    float person_max_area_ratio
) {
    if (boxes.ndim() != 2 || boxes.shape(1) != 4) {
        throw std::invalid_argument("boxes must have shape [N, 4]");
    }
    if (scores.ndim() != 1 || classes.ndim() != 1) {
        throw std::invalid_argument("scores and classes must have shape [N]");
    }
    const ssize_t num = boxes.shape(0);
    if (scores.shape(0) != num || classes.shape(0) != num) {
        throw std::invalid_argument("boxes, scores, and classes must have the same length");
    }

    auto boxes_in = boxes.unchecked<2>();
    auto scores_in = scores.unchecked<1>();
    auto classes_in = classes.unchecked<1>();
    std::vector<int> keep_indices;
    std::vector<unsigned char> suspect_flags;
    keep_indices.reserve(static_cast<size_t>(num));
    suspect_flags.reserve(static_cast<size_t>(num));

    const float frame_area = std::max(static_cast<float>(frame_w * frame_h), 1.0f);
    for (ssize_t i = 0; i < num; ++i) {
        bool keep = scores_in(i) > score_threshold;
        if (track_person_only) {
            keep = keep && classes_in(i) == person_class;
        }
        if (is_tiled) {
            const float cx = (boxes_in(i, 0) + boxes_in(i, 2)) * 0.5f;
            const float cy = (boxes_in(i, 1) + boxes_in(i, 3)) * 0.5f;
            keep = keep && cx >= 0.0f && cx < frame_w && cy >= 0.0f && cy < frame_h;
        }

        bool geometry_clean = true;
        if (person_geometry_prior) {
            const float box_w = std::max(boxes_in(i, 2) - boxes_in(i, 0), 1e-6f);
            const float box_h = std::max(boxes_in(i, 3) - boxes_in(i, 1), 1e-6f);
            const float aspect = box_h / box_w;
            const float area_ratio = (box_w * box_h) / frame_area;
            if (person_min_height_ratio > 0.0f) {
                geometry_clean = geometry_clean && box_h >= frame_h * person_min_height_ratio;
            }
            if (person_min_aspect > 0.0f) {
                geometry_clean = geometry_clean && aspect >= person_min_aspect;
            }
            if (person_max_aspect > 0.0f) {
                geometry_clean = geometry_clean && aspect <= person_max_aspect;
            }
            if (person_min_area_ratio > 0.0f) {
                geometry_clean = geometry_clean && area_ratio >= person_min_area_ratio;
            }
            if (person_max_area_ratio > 0.0f) {
                geometry_clean = geometry_clean && area_ratio <= person_max_area_ratio;
            }
            if (!geometry_suspect_support) {
                keep = keep && geometry_clean;
            }
        }

        if (keep) {
            keep_indices.push_back(static_cast<int>(i));
            suspect_flags.push_back(static_cast<unsigned char>(
                person_geometry_prior && geometry_suspect_support && !geometry_clean
            ));
        }
    }

    const ssize_t out_num = static_cast<ssize_t>(keep_indices.size());
    py::array_t<int> keep_arr({out_num});
    py::array_t<bool> suspect_arr({out_num});
    auto keep_mut = keep_arr.mutable_unchecked<1>();
    auto suspect_mut = suspect_arr.mutable_unchecked<1>();
    for (ssize_t i = 0; i < out_num; ++i) {
        keep_mut(i) = keep_indices[static_cast<size_t>(i)];
        suspect_mut(i) = static_cast<bool>(suspect_flags[static_cast<size_t>(i)]);
    }
    return py::make_tuple(keep_arr, suspect_arr);
}

} // namespace

namespace {

struct RelinkBox {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct RelinkMotionSnapshot {
    std::array<float, 4> state{};
    std::array<float, 16> covariance{};
};

struct RelinkStats {
    int attempts = 0;
    int accepted = 0;
    int reject_age = 0;
    int reject_assigned = 0;
    int reject_spatial = 0;
    int reject_mahalanobis = 0;
    int reject_similarity = 0;
    int reject_consistency = 0;
    int reject_margin = 0;
    int new_ids = 0;
};

class SemanticRelinkerCpp {
public:
    SemanticRelinkerCpp(
        float sim_threshold,
        int ttl,
        float ema_beta,
        float spatial_gate,
        int min_lost_frames,
        float min_iou,
        float mahalanobis_threshold,
        int buffer_size,
        float min_consistency,
        std::string rerank_mode,
        float reciprocal_margin,
        bool debug
    )
        : sim_threshold_(sim_threshold),
          ttl_(ttl),
          ema_beta_(std::clamp(ema_beta, 0.0f, 1.0f)),
          spatial_gate_(spatial_gate),
          min_lost_frames_(min_lost_frames),
          min_iou_(min_iou),
          mahalanobis_threshold_(mahalanobis_threshold),
          buffer_size_(std::max(1, buffer_size)),
          min_consistency_(min_consistency),
          rerank_mode_(std::move(rerank_mode)),
          reciprocal_margin_(std::max(0.0f, reciprocal_margin)),
          debug_(debug) {}

    void update_motion_snapshots(const std::vector<TrackStateSnapshot>& snapshots) {
        for (const auto& snap : snapshots) {
            const int canonical = alias_.count(snap.obj_id) ? alias_.at(snap.obj_id) : snap.obj_id;
            RelinkMotionSnapshot out;
            for (size_t i = 0; i < out.state.size() && i < snap.state.size(); ++i) {
                out.state[i] = snap.state[i];
            }
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    const size_t src = static_cast<size_t>(r * 8 + c);
                    const size_t dst = static_cast<size_t>(r * 4 + c);
                    if (src < snap.covariance.size()) {
                        out.covariance[dst] = snap.covariance[src];
                    }
                }
            }
            motion_[canonical] = out;
        }
    }

    std::vector<int> motion_candidate_ids(int frame_id = -1) const {
        if (mahalanobis_threshold_ <= 0.0f) {
            return {};
        }
        std::vector<int> ids;
        ids.reserve(feature_order_.size());
        for (int cid : feature_order_) {
            const auto feature_it = features_.find(cid);
            if (feature_it == features_.end()) {
                continue;
            }
            if (frame_id >= 0) {
                const auto seen_it = last_seen_.find(cid);
                if (seen_it == last_seen_.end()) {
                    continue;
                }
                const int age = frame_id - seen_it->second;
                if (age < min_lost_frames_ || age > ttl_) {
                    continue;
                }
            }
            ids.push_back(cid);
        }
        return ids;
    }

    void inject_reference(int canonical_id, py::object embedding) {
        std::vector<float> emb = normalize(extract_embedding(embedding));
        features_[canonical_id] = emb;
        if (buffer_size_ > 1) {
            auto& buf = buffers_[canonical_id];
            buf.push_back(emb);
            if (static_cast<int>(buf.size()) > buffer_size_) {
                buf.erase(buf.begin());
            }
        }
    }

    void inject_references_many(py::iterable references) {
        for (py::handle item : references) {
            py::tuple pair = py::reinterpret_borrow<py::tuple>(item);
            if (pair.size() != 2) {
                throw std::runtime_error("inject_references_many expects (canonical_id, embedding) pairs");
            }
            inject_reference(pair[0].cast<int>(), py::reinterpret_borrow<py::object>(pair[1]));
        }
    }

    int canonical_id(int raw_id) const {
        auto it = alias_.find(raw_id);
        return it == alias_.end() ? raw_id : it->second;
    }

    bool has_feature(int canonical_id) const {
        return features_.find(canonical_id) != features_.end();
    }

    int resolve(
        int raw_id,
        py::object embedding,
        py::sequence box_obj,
        float score,
        int frame_id,
        int frame_w,
        int frame_h,
        py::set assigned
    ) {
        (void)score;
        if (embedding.is_none()) {
            auto it = alias_.find(raw_id);
            return it == alias_.end() ? raw_id : it->second;
        }

        std::vector<float> emb = normalize(extract_embedding(embedding));
        if (!alias_.count(raw_id)) {
            stats_.attempts += 1;
            int best_id = -1;
            float best_sim = sim_threshold_;
            float second_best_sim = sim_threshold_ - 1.0f;
            float best_iou = 0.0f;
            float best_center = 0.0f;
            float best_maha = 0.0f;
            const RelinkBox box = parse_box(box_obj);

            for (int cid : feature_order_) {
                const auto feature_it = features_.find(cid);
                if (feature_it == features_.end()) {
                    continue;
                }
                py::int_ py_cid(cid);
                if (PySet_Contains(assigned.ptr(), py_cid.ptr()) == 1) {
                    stats_.reject_assigned += 1;
                    continue;
                }
                const int age = frame_id - last_seen_.at(cid);
                if (age < min_lost_frames_ || age > ttl_) {
                    stats_.reject_age += 1;
                    continue;
                }
                auto [center_norm, iou] = spatial_metrics(box, last_boxes_.at(cid), frame_w, frame_h);
                if (center_norm > spatial_gate_ || iou < min_iou_) {
                    stats_.reject_spatial += 1;
                    continue;
                }
                float maha = 0.0f;
                if (mahalanobis_threshold_ > 0.0f) {
                    const auto motion_it = motion_.find(cid);
                    if (motion_it == motion_.end()) {
                        stats_.reject_mahalanobis += 1;
                        continue;
                    }
                    maha = mahalanobis(box, motion_it->second);
                    if (maha > mahalanobis_threshold_) {
                        stats_.reject_mahalanobis += 1;
                        continue;
                    }
                }
                if (min_consistency_ > 0.0f && buffer_size_ > 1) {
                    const float consistency = buffer_consistency(cid);
                    if (consistency < min_consistency_) {
                        stats_.reject_consistency += 1;
                        continue;
                    }
                }

                std::vector<float> ref = buffer_size_ > 1 ? buffer_mean(cid) : feature_it->second;
                if (ref.empty()) {
                    ref = feature_it->second;
                }
                const float sim = dot(emb, ref);
                if (sim > best_sim) {
                    if (best_id >= 0) {
                        second_best_sim = best_sim;
                    }
                    best_id = cid;
                    best_sim = sim;
                    best_iou = iou;
                    best_center = center_norm;
                    best_maha = maha;
                } else {
                    if (sim > second_best_sim) {
                        second_best_sim = sim;
                    }
                    stats_.reject_similarity += 1;
                }
            }

            if (best_id >= 0 && reciprocal_margin_ > 0.0f) {
                if (best_sim - second_best_sim < reciprocal_margin_) {
                    stats_.reject_margin += 1;
                    best_id = -1;
                }
            }

            if (best_id >= 0) {
                stats_.accepted += 1;
                accept_sims_.push_back(best_sim);
                accept_ious_.push_back(best_iou);
                accept_center_dists_.push_back(best_center);
                accept_mahas_.push_back(best_maha);
                alias_[raw_id] = best_id;
            } else {
                stats_.new_ids += 1;
                alias_[raw_id] = raw_id;
            }
        }

        const int canonical = alias_.at(raw_id);
        if (buffer_size_ > 1) {
            auto& buf = buffers_[canonical];
            buf.push_back(emb);
            if (static_cast<int>(buf.size()) > buffer_size_) {
                buf.erase(buf.begin());
            }
            features_[canonical] = buffer_mean(canonical);
        } else {
            auto old = features_.find(canonical);
            if (old == features_.end()) {
                features_[canonical] = emb;
            } else {
                std::vector<float> updated(emb.size(), 0.0f);
                for (size_t i = 0; i < emb.size(); ++i) {
                    updated[i] = ema_beta_ * old->second[i] + (1.0f - ema_beta_) * emb[i];
                }
                features_[canonical] = normalize(updated);
            }
        }
        if (std::find(feature_order_.begin(), feature_order_.end(), canonical) == feature_order_.end()) {
            feature_order_.push_back(canonical);
        }
        last_seen_[canonical] = frame_id;
        last_boxes_[canonical] = parse_box(box_obj);
        assigned.add(py::int_(canonical));
        return canonical;
    }

    py::list resolve_many(
        py::iterable candidates,
        int frame_id,
        int frame_w,
        int frame_h
    ) {
        py::set assigned;
        py::list out;
        for (py::handle item : candidates) {
            py::tuple candidate = py::reinterpret_borrow<py::tuple>(item);
            if (candidate.size() != 4) {
                throw std::runtime_error("resolve_many expects (raw_id, embedding, box, score) tuples");
            }
            out.append(resolve(
                candidate[0].cast<int>(),
                py::reinterpret_borrow<py::object>(candidate[1]),
                candidate[2].cast<py::sequence>(),
                candidate[3].cast<float>(),
                frame_id,
                frame_w,
                frame_h,
                assigned
            ));
        }
        return out;
    }

    py::list resolve_many_packed(
        py::sequence raw_ids,
        py::sequence embeddings,
        py::sequence boxes,
        py::sequence scores,
        int frame_id,
        int frame_w,
        int frame_h
    ) {
        const py::ssize_t n = py::len(raw_ids);
        if (py::len(embeddings) != n || py::len(boxes) != n || py::len(scores) != n) {
            throw std::runtime_error("resolve_many_packed expects equally sized raw_ids/embeddings/boxes/scores");
        }
        py::set assigned;
        py::list out;
        for (py::ssize_t i = 0; i < n; ++i) {
            out.append(resolve(
                raw_ids[i].cast<int>(),
                py::reinterpret_borrow<py::object>(embeddings[i]),
                boxes[i].cast<py::sequence>(),
                scores[i].cast<float>(),
                frame_id,
                frame_w,
                frame_h,
                assigned
            ));
        }
        return out;
    }

    py::dict get_alias() const {
        py::dict out;
        for (const auto& [k, v] : alias_) {
            out[py::int_(k)] = py::int_(v);
        }
        return out;
    }

    py::dict get_features() const {
        py::dict out;
        for (const auto& [k, v] : features_) {
            out[py::int_(k)] = py::int_(k);
        }
        return out;
    }

    py::dict stats() const {
        py::dict out;
        out["attempts"] = stats_.attempts;
        out["accepted"] = stats_.accepted;
        out["reject_age"] = stats_.reject_age;
        out["reject_assigned"] = stats_.reject_assigned;
        out["reject_spatial"] = stats_.reject_spatial;
        out["reject_mahalanobis"] = stats_.reject_mahalanobis;
        out["reject_similarity"] = stats_.reject_similarity;
        out["reject_consistency"] = stats_.reject_consistency;
        out["reject_margin"] = stats_.reject_margin;
        out["new_ids"] = stats_.new_ids;
        return out;
    }

    void report() const {
        py::print("🔁 Semantic Relink Report:");
        py::print(
            "  attempts=" + std::to_string(stats_.attempts) +
            " accepted=" + std::to_string(stats_.accepted) +
            " new_ids=" + std::to_string(stats_.new_ids) +
            " reject_age=" + std::to_string(stats_.reject_age) +
            " reject_assigned=" + std::to_string(stats_.reject_assigned) +
            " reject_spatial=" + std::to_string(stats_.reject_spatial) +
            " reject_mahalanobis=" + std::to_string(stats_.reject_mahalanobis) +
            " reject_similarity=" + std::to_string(stats_.reject_similarity) +
            " reject_margin=" + std::to_string(stats_.reject_margin)
        );
        if (!accept_sims_.empty()) {
            py::print(
                "  accepted mean_sim=" + format3(mean(accept_sims_)) +
                " mean_iou=" + format3(mean(accept_ious_)) +
                " mean_center_norm=" + format3(mean(accept_center_dists_)) +
                " mean_maha=" + format3(mean(accept_mahas_))
            );
        }
        if (buffer_size_ > 1) {
            py::print(
                "  buffer_size=" + std::to_string(buffer_size_) +
                " reject_consistency=" + std::to_string(stats_.reject_consistency)
            );
        }
    }

    // Internal C++ resolve — takes pre-normalized embedding and C++ box.
    // Called by IdentityResolverCpp to avoid re-parsing Python objects between stages.
    int resolve_cpp(
        int raw_id,
        const std::vector<float>& emb,
        bool has_emb,
        const RelinkBox& box,
        float /*score*/,
        int frame_id,
        int frame_w,
        int frame_h,
        std::unordered_set<int>& assigned
    ) {
        if (!has_emb) {
            auto it = alias_.find(raw_id);
            return it == alias_.end() ? raw_id : it->second;
        }

        if (!alias_.count(raw_id)) {
            stats_.attempts += 1;
            int best_id = -1;
            float best_sim = sim_threshold_;
            float second_best_sim = sim_threshold_ - 1.0f;
            float best_iou = 0.0f, best_center = 0.0f, best_maha = 0.0f;

            for (int cid : feature_order_) {
                const auto feature_it = features_.find(cid);
                if (feature_it == features_.end()) continue;
                if (assigned.count(cid)) { stats_.reject_assigned += 1; continue; }
                const int age = frame_id - last_seen_.at(cid);
                if (age < min_lost_frames_ || age > ttl_) { stats_.reject_age += 1; continue; }
                auto [center_norm, iou] = spatial_metrics(box, last_boxes_.at(cid), frame_w, frame_h);
                if (center_norm > spatial_gate_ || iou < min_iou_) { stats_.reject_spatial += 1; continue; }
                float maha = 0.0f;
                if (mahalanobis_threshold_ > 0.0f) {
                    const auto motion_it = motion_.find(cid);
                    if (motion_it == motion_.end()) { stats_.reject_mahalanobis += 1; continue; }
                    maha = mahalanobis(box, motion_it->second);
                    if (maha > mahalanobis_threshold_) { stats_.reject_mahalanobis += 1; continue; }
                }
                if (min_consistency_ > 0.0f && buffer_size_ > 1) {
                    if (buffer_consistency(cid) < min_consistency_) { stats_.reject_consistency += 1; continue; }
                }
                std::vector<float> ref = buffer_size_ > 1 ? buffer_mean(cid) : feature_it->second;
                if (ref.empty()) ref = feature_it->second;
                const float sim = dot(emb, ref);
                if (sim > best_sim) {
                    if (best_id >= 0) second_best_sim = best_sim;
                    best_id = cid; best_sim = sim;
                    best_iou = iou; best_center = center_norm; best_maha = maha;
                } else {
                    if (sim > second_best_sim) second_best_sim = sim;
                    stats_.reject_similarity += 1;
                }
            }

            if (best_id >= 0 && reciprocal_margin_ > 0.0f) {
                if (best_sim - second_best_sim < reciprocal_margin_) {
                    stats_.reject_margin += 1; best_id = -1;
                }
            }

            if (best_id >= 0) {
                stats_.accepted += 1;
                accept_sims_.push_back(best_sim);
                accept_ious_.push_back(best_iou);
                accept_center_dists_.push_back(best_center);
                accept_mahas_.push_back(best_maha);
                alias_[raw_id] = best_id;
            } else {
                stats_.new_ids += 1;
                alias_[raw_id] = raw_id;
            }
        }

        const int canonical = alias_.at(raw_id);
        if (buffer_size_ > 1) {
            auto& buf = buffers_[canonical];
            buf.push_back(emb);
            if (static_cast<int>(buf.size()) > buffer_size_) buf.erase(buf.begin());
            features_[canonical] = buffer_mean(canonical);
        } else {
            auto old_it = features_.find(canonical);
            if (old_it == features_.end()) {
                features_[canonical] = emb;
            } else {
                std::vector<float> updated(emb.size(), 0.0f);
                for (size_t i = 0; i < emb.size(); ++i)
                    updated[i] = ema_beta_ * old_it->second[i] + (1.0f - ema_beta_) * emb[i];
                features_[canonical] = normalize(updated);
            }
        }
        if (std::find(feature_order_.begin(), feature_order_.end(), canonical) == feature_order_.end())
            feature_order_.push_back(canonical);
        last_seen_[canonical] = frame_id;
        last_boxes_[canonical] = box;
        assigned.insert(canonical);
        return canonical;
    }

private:
    static std::string format3(float value) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", value);
        return std::string(buf);
    }

    static float mean(const std::vector<float>& values) {
        if (values.empty()) {
            return 0.0f;
        }
        return std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
    }

    static std::vector<float> extract_embedding(py::object embedding) {
        py::object numpy_obj = embedding.attr("detach")().attr("float")().attr("cpu")().attr("numpy")();
        py::array_t<float, py::array::c_style | py::array::forcecast> arr(numpy_obj);
        if (arr.ndim() != 1) {
            throw std::invalid_argument("SemanticRelinker embedding must be a 1D tensor");
        }
        auto view = arr.unchecked<1>();
        std::vector<float> out(static_cast<size_t>(arr.shape(0)));
        for (ssize_t i = 0; i < arr.shape(0); ++i) {
            out[static_cast<size_t>(i)] = view(i);
        }
        return out;
    }

    static RelinkBox parse_box(py::sequence seq) {
        if (py::len(seq) != 4) {
            throw std::invalid_argument("box must have four elements");
        }
        return {
            seq[0].cast<float>(),
            seq[1].cast<float>(),
            seq[2].cast<float>(),
            seq[3].cast<float>(),
        };
    }

    static std::vector<float> normalize(const std::vector<float>& input) {
        float norm_sq = 0.0f;
        for (float value : input) {
            norm_sq += value * value;
        }
        const float inv_norm = 1.0f / std::max(std::sqrt(norm_sq), 1e-12f);
        std::vector<float> out(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            out[i] = input[i] * inv_norm;
        }
        return out;
    }

    static float dot(const std::vector<float>& a, const std::vector<float>& b) {
        const size_t n = std::min(a.size(), b.size());
        float out = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            out += a[i] * b[i];
        }
        return out;
    }

    static std::pair<float, float> spatial_metrics(
        const RelinkBox& box,
        const RelinkBox& old_box,
        int frame_w,
        int frame_h
    ) {
        const float cx = (box.x1 + box.x2) * 0.5f;
        const float cy = (box.y1 + box.y2) * 0.5f;
        const float ocx = (old_box.x1 + old_box.x2) * 0.5f;
        const float ocy = (old_box.y1 + old_box.y2) * 0.5f;
        const float dx = cx - ocx;
        const float dy = cy - ocy;
        const float center_norm = std::sqrt(dx * dx + dy * dy) / std::max({frame_w, frame_h, 1});

        const float ix1 = std::max(box.x1, old_box.x1);
        const float iy1 = std::max(box.y1, old_box.y1);
        const float ix2 = std::min(box.x2, old_box.x2);
        const float iy2 = std::min(box.y2, old_box.y2);
        const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
        const float area = std::max(0.0f, box.x2 - box.x1) * std::max(0.0f, box.y2 - box.y1);
        const float old_area = std::max(0.0f, old_box.x2 - old_box.x1) * std::max(0.0f, old_box.y2 - old_box.y1);
        const float iou = inter / std::max(area + old_area - inter, 1e-6f);
        return {center_norm, iou};
    }

    static float mahalanobis(const RelinkBox& box, const RelinkMotionSnapshot& snap) {
        const float bw = std::max(1e-6f, box.x2 - box.x1);
        const float bh = std::max(1e-6f, box.y2 - box.y1);
        const std::array<float, 4> measurement = {
            (box.x1 + box.x2) * 0.5f,
            (box.y1 + box.y2) * 0.5f,
            bw / bh,
            bh,
        };
        std::array<float, 4> residual{};
        for (size_t i = 0; i < residual.size(); ++i) {
            residual[i] = measurement[i] - snap.state[i];
        }

        std::array<std::array<float, 5>, 4> aug{};
        const float h = std::max(snap.state[3], 1e-6f);
        const float pos_std = h / 20.0f;
        const float r_diag[4] = {pos_std * pos_std, pos_std * pos_std, 1e-2f, pos_std * pos_std};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                aug[r][c] = snap.covariance[static_cast<size_t>(r * 4 + c)] + (r == c ? r_diag[r] : 0.0f);
            }
            aug[r][4] = residual[static_cast<size_t>(r)];
        }

        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            for (int r = col + 1; r < 4; ++r) {
                if (std::fabs(aug[r][col]) > std::fabs(aug[pivot][col])) {
                    pivot = r;
                }
            }
            if (std::fabs(aug[pivot][col]) < 1e-8f) {
                return std::numeric_limits<float>::infinity();
            }
            if (pivot != col) {
                std::swap(aug[pivot], aug[col]);
            }
            const float denom = aug[col][col];
            for (int c = col; c < 5; ++c) {
                aug[col][c] /= denom;
            }
            for (int r = 0; r < 4; ++r) {
                if (r == col) {
                    continue;
                }
                const float factor = aug[r][col];
                for (int c = col; c < 5; ++c) {
                    aug[r][c] -= factor * aug[col][c];
                }
            }
        }

        float out = 0.0f;
        for (int i = 0; i < 4; ++i) {
            out += residual[static_cast<size_t>(i)] * aug[i][4];
        }
        return out;
    }

    std::vector<float> buffer_mean(int cid) const {
        auto it = buffers_.find(cid);
        if (it == buffers_.end() || it->second.empty()) {
            return {};
        }
        std::vector<float> out(it->second.front().size(), 0.0f);
        for (const auto& emb : it->second) {
            for (size_t i = 0; i < emb.size(); ++i) {
                out[i] += emb[i];
            }
        }
        const float inv = 1.0f / static_cast<float>(it->second.size());
        for (float& value : out) {
            value *= inv;
        }
        return normalize(out);
    }

    float buffer_consistency(int cid) const {
        auto it = buffers_.find(cid);
        if (it == buffers_.end() || it->second.size() < 2) {
            return 1.0f;
        }
        float sum = 0.0f;
        const int n = static_cast<int>(it->second.size());
        for (const auto& a : it->second) {
            for (const auto& b : it->second) {
                sum += dot(a, b);
            }
        }
        return (sum - static_cast<float>(n)) / static_cast<float>(n * (n - 1));
    }

    float sim_threshold_;
    int ttl_;
    float ema_beta_;
    float spatial_gate_;
    int min_lost_frames_;
    float min_iou_;
    float mahalanobis_threshold_;
    int buffer_size_;
    float min_consistency_;
    std::string rerank_mode_;
    float reciprocal_margin_;
    bool debug_;

    std::unordered_map<int, int> alias_;
    std::unordered_map<int, std::vector<float>> features_;
    std::unordered_map<int, std::vector<std::vector<float>>> buffers_;
    std::unordered_map<int, int> last_seen_;
    std::unordered_map<int, RelinkBox> last_boxes_;
    std::unordered_map<int, RelinkMotionSnapshot> motion_;
    std::vector<int> feature_order_;
    RelinkStats stats_;
    std::vector<float> accept_sims_;
    std::vector<float> accept_ious_;
    std::vector<float> accept_center_dists_;
    std::vector<float> accept_mahas_;
};

// ============================================================
// TrackletLifecycleMerger (C++ port of Python class in runner.py)
// ============================================================

struct LifecycleState {
    int output_id = 0;
    int last_frame_id = 0;
    RelinkBox box{};
    float score = 0.0f;
    std::vector<float> embedding;  // normalized; empty == no embedding
};

struct LifecycleStats {
    int attempts = 0, accepted = 0, new_ids = 0;
    int reject_age = 0, reject_assigned = 0, reject_spatial = 0, reject_similarity = 0;
};

class TrackletLifecycleMergerCpp {
public:
    TrackletLifecycleMergerCpp(
        bool enabled, int ttl, int min_gap,
        float spatial_gate, float min_iou, float sim_threshold,
        bool require_embedding, float ema
    )
        : enabled_(enabled),
          ttl_(std::max(1, ttl)),
          min_gap_(std::max(0, min_gap)),
          spatial_gate_(spatial_gate),
          min_iou_(min_iou),
          sim_threshold_(sim_threshold),
          require_embedding_(require_embedding),
          ema_(std::clamp(ema, 0.0f, 1.0f)) {}

    // Internal C++ resolve — used by IdentityResolverCpp.
    int resolve_cpp(
        int local_id,
        const RelinkBox& box,
        float score,
        int frame_id,
        int frame_w,
        int frame_h,
        const std::vector<float>& emb,
        bool has_emb,
        std::unordered_set<int>& assigned_outputs
    ) {
        if (!enabled_) return local_id;

        auto alias_it = alias_.find(local_id);
        int output_id = (alias_it != alias_.end()) ? alias_it->second : -1;

        if (output_id < 0) {
            stats_.attempts += 1;
            int best_id = -1;
            float best_score = -1.0f;

            for (auto& [candidate_id, state] : states_) {
                if (assigned_outputs.count(candidate_id)) {
                    stats_.reject_assigned += 1; continue;
                }
                const int age = frame_id - state.last_frame_id;
                if (age < min_gap_ || age > ttl_) {
                    stats_.reject_age += 1; continue;
                }
                auto [center_norm, iou] = spatial_metrics_lc(box, state.box, frame_w, frame_h);
                if (center_norm > spatial_gate_ || iou < min_iou_) {
                    stats_.reject_spatial += 1; continue;
                }
                float sim = 0.0f;
                if (has_emb && !state.embedding.empty()) {
                    sim = dot_lc(emb, state.embedding);
                    if (sim < sim_threshold_) {
                        stats_.reject_similarity += 1; continue;
                    }
                } else if (require_embedding_) {
                    stats_.reject_similarity += 1; continue;
                }
                const float candidate_score =
                    sim + std::max(0.0f, spatial_gate_ - center_norm) + iou;
                if (candidate_score > best_score) {
                    best_score = candidate_score;
                    best_id = candidate_id;
                }
            }

            if (best_id < 0) {
                stats_.new_ids += 1;
                output_id = local_id;
            } else {
                stats_.accepted += 1;
                output_id = best_id;
            }
            alias_[local_id] = output_id;
        }

        // EMA update of state embedding
        auto old_it = states_.find(output_id);
        std::vector<float> updated_emb;
        if (has_emb) {
            updated_emb = emb;
            if (old_it != states_.end() && !old_it->second.embedding.empty()) {
                updated_emb = normalize_lc(ema_mix_lc(old_it->second.embedding, emb, ema_));
            }
        } else if (old_it != states_.end()) {
            updated_emb = old_it->second.embedding;
        }

        states_[output_id] = LifecycleState{output_id, frame_id, box, score, std::move(updated_emb)};
        assigned_outputs.insert(output_id);
        return output_id;
    }

    // Python-facing API (same keyword arg names as Python class)
    py::list resolve_many_packed(
        py::sequence local_ids,
        py::sequence boxes,
        py::sequence scores,
        py::sequence embeddings,
        int frame_id,
        int frame_w,
        int frame_h
    ) {
        const py::ssize_t n = py::len(local_ids);
        std::unordered_set<int> assigned;
        py::list out;
        for (py::ssize_t i = 0; i < n; ++i) {
            auto [emb, has_emb] = parse_emb_lc(py::reinterpret_borrow<py::object>(embeddings[i]));
            out.append(resolve_cpp(
                local_ids[i].cast<int>(),
                parse_box_lc(boxes[i]),
                scores[i].cast<float>(),
                frame_id, frame_w, frame_h,
                emb, has_emb, assigned
            ));
        }
        return out;
    }

    py::list resolve_many(
        py::iterable candidates,
        int frame_id,
        int frame_w,
        int frame_h
    ) {
        std::unordered_set<int> assigned;
        py::list out;
        for (py::handle item : candidates) {
            py::tuple t = py::reinterpret_borrow<py::tuple>(item);
            if (t.size() != 4)
                throw std::runtime_error(
                    "resolve_many expects (local_id, box, score, embedding) tuples");
            auto [emb, has_emb] = parse_emb_lc(py::reinterpret_borrow<py::object>(t[3]));
            out.append(resolve_cpp(
                t[0].cast<int>(),
                parse_box_lc(t[1]),
                t[2].cast<float>(),
                frame_id, frame_w, frame_h,
                emb, has_emb, assigned
            ));
        }
        return out;
    }

    void prune(int frame_id) {
        std::vector<int> stale;
        for (const auto& [oid, state] : states_) {
            if (frame_id - state.last_frame_id > ttl_) stale.push_back(oid);
        }
        for (int oid : stale) states_.erase(oid);
    }

    py::dict get_alias() const {
        py::dict out;
        for (const auto& [k, v] : alias_) out[py::int_(k)] = py::int_(v);
        return out;
    }

    py::dict stats_dict() const {
        py::dict out;
        out["attempts"]          = stats_.attempts;
        out["accepted"]          = stats_.accepted;
        out["new_ids"]           = stats_.new_ids;
        out["reject_age"]        = stats_.reject_age;
        out["reject_assigned"]   = stats_.reject_assigned;
        out["reject_spatial"]    = stats_.reject_spatial;
        out["reject_similarity"] = stats_.reject_similarity;
        return out;
    }

    void report() const {
        if (!enabled_) return;
        py::print("🔗 Tracklet Lifecycle Report:");
        py::print(
            "  attempts=" + std::to_string(stats_.attempts) +
            " accepted=" + std::to_string(stats_.accepted) +
            " new_ids=" + std::to_string(stats_.new_ids) +
            " reject_age=" + std::to_string(stats_.reject_age) +
            " reject_assigned=" + std::to_string(stats_.reject_assigned) +
            " reject_spatial=" + std::to_string(stats_.reject_spatial) +
            " reject_similarity=" + std::to_string(stats_.reject_similarity)
        );
    }

private:
    static std::pair<std::vector<float>, bool> parse_emb_lc(py::object emb_obj) {
        if (emb_obj.is_none()) return {{}, false};
        py::object np = emb_obj.attr("detach")().attr("float")().attr("cpu")().attr("numpy")();
        py::array_t<float, py::array::c_style | py::array::forcecast> arr(np);
        if (arr.ndim() != 1)
            throw std::invalid_argument("TrackletLifecycleMerger embedding must be 1D");
        auto view = arr.unchecked<1>();
        std::vector<float> out(static_cast<size_t>(arr.shape(0)));
        for (ssize_t i = 0; i < arr.shape(0); ++i) out[static_cast<size_t>(i)] = view(i);
        return {normalize_lc(out), true};
    }

    static RelinkBox parse_box_lc(py::handle h) {
        py::sequence seq = h.cast<py::sequence>();
        if (py::len(seq) != 4) throw std::invalid_argument("box must have four elements");
        return {seq[0].cast<float>(), seq[1].cast<float>(),
                seq[2].cast<float>(), seq[3].cast<float>()};
    }

    static std::vector<float> normalize_lc(const std::vector<float>& v) {
        float norm_sq = 0.0f;
        for (float x : v) norm_sq += x * x;
        const float inv = 1.0f / std::max(std::sqrt(norm_sq), 1e-12f);
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] * inv;
        return out;
    }

    static float dot_lc(const std::vector<float>& a, const std::vector<float>& b) {
        float s = 0.0f;
        const size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    }

    static std::vector<float> ema_mix_lc(
        const std::vector<float>& old_v, const std::vector<float>& new_v, float alpha
    ) {
        const size_t n = std::min(old_v.size(), new_v.size());
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = alpha * old_v[i] + (1.0f - alpha) * new_v[i];
        return out;
    }

    static std::pair<float, float> spatial_metrics_lc(
        const RelinkBox& box, const RelinkBox& old_box, int w, int h
    ) {
        const float cx  = (box.x1 + box.x2) * 0.5f,  cy  = (box.y1 + box.y2) * 0.5f;
        const float ocx = (old_box.x1 + old_box.x2) * 0.5f, ocy = (old_box.y1 + old_box.y2) * 0.5f;
        const float center_norm =
            std::sqrt((cx - ocx) * (cx - ocx) + (cy - ocy) * (cy - ocy))
            / static_cast<float>(std::max({w, h, 1}));
        const float ix1 = std::max(box.x1, old_box.x1), iy1 = std::max(box.y1, old_box.y1);
        const float ix2 = std::min(box.x2, old_box.x2), iy2 = std::min(box.y2, old_box.y2);
        const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
        const float area_a = std::max(0.0f, box.x2 - box.x1) * std::max(0.0f, box.y2 - box.y1);
        const float area_b =
            std::max(0.0f, old_box.x2 - old_box.x1) * std::max(0.0f, old_box.y2 - old_box.y1);
        return {center_norm, inter / std::max(area_a + area_b - inter, 1e-6f)};
    }

    bool enabled_;
    int ttl_, min_gap_;
    float spatial_gate_, min_iou_, sim_threshold_, ema_;
    bool require_embedding_;
    std::unordered_map<int, int> alias_;
    std::unordered_map<int, LifecycleState> states_;
    LifecycleStats stats_;
};

// ============================================================
// IdentityResolver — single-boundary facade over relink + lifecycle
// ============================================================

class IdentityResolverCpp {
public:
    IdentityResolverCpp(py::object relinker_obj, py::object lifecycle_obj)
        : relinker_obj_(std::move(relinker_obj)),
          lifecycle_obj_(std::move(lifecycle_obj)),
          relinker_(relinker_obj_.cast<SemanticRelinkerCpp*>()),
          lifecycle_(lifecycle_obj_.cast<TrackletLifecycleMergerCpp*>()) {}

    py::list resolve_pass(
        py::sequence local_ids,
        py::sequence embeddings,
        py::sequence boxes,
        py::sequence scores,
        int frame_id,
        int frame_w,
        int frame_h
    ) {
        const py::ssize_t n = py::len(local_ids);
        if (n == 0) return py::list{};

        // Parse all inputs once into C++ vectors
        std::vector<int> ids_v(static_cast<size_t>(n));
        std::vector<std::vector<float>> embs_v(static_cast<size_t>(n));
        std::vector<bool> has_emb_v(static_cast<size_t>(n), false);
        std::vector<RelinkBox> boxes_v(static_cast<size_t>(n));
        std::vector<float> scores_v(static_cast<size_t>(n));

        for (py::ssize_t i = 0; i < n; ++i) {
            const size_t idx = static_cast<size_t>(i);
            ids_v[idx] = local_ids[i].cast<int>();
            boxes_v[idx] = parse_box_ip(boxes[i]);
            scores_v[idx] = scores[i].cast<float>();
            py::object emb_obj = py::reinterpret_borrow<py::object>(embeddings[i]);
            if (!emb_obj.is_none()) {
                embs_v[idx] = normalize_ip(extract_emb_ip(emb_obj));
                has_emb_v[idx] = true;
            }
        }

        // Stage 1: semantic relink (all candidates share one assigned set)
        std::vector<int> relinked(static_cast<size_t>(n));
        {
            std::unordered_set<int> assigned;
            for (py::ssize_t i = 0; i < n; ++i) {
                const size_t idx = static_cast<size_t>(i);
                relinked[idx] = relinker_->resolve_cpp(
                    ids_v[idx], embs_v[idx], has_emb_v[idx],
                    boxes_v[idx], scores_v[idx],
                    frame_id, frame_w, frame_h, assigned
                );
            }
        }

        // Stage 2: lifecycle merge
        py::list out;
        {
            std::unordered_set<int> assigned;
            for (py::ssize_t i = 0; i < n; ++i) {
                const size_t idx = static_cast<size_t>(i);
                out.append(lifecycle_->resolve_cpp(
                    relinked[idx],
                    boxes_v[idx], scores_v[idx],
                    frame_id, frame_w, frame_h,
                    embs_v[idx], has_emb_v[idx], assigned
                ));
            }
        }
        return out;
    }

private:
    static std::vector<float> extract_emb_ip(py::object emb_obj) {
        py::object np = emb_obj.attr("detach")().attr("float")().attr("cpu")().attr("numpy")();
        py::array_t<float, py::array::c_style | py::array::forcecast> arr(np);
        if (arr.ndim() != 1)
            throw std::invalid_argument("IdentityResolver embedding must be 1D");
        auto view = arr.unchecked<1>();
        std::vector<float> out(static_cast<size_t>(arr.shape(0)));
        for (ssize_t i = 0; i < arr.shape(0); ++i) out[static_cast<size_t>(i)] = view(i);
        return out;
    }

    static RelinkBox parse_box_ip(py::handle h) {
        py::sequence seq = h.cast<py::sequence>();
        if (py::len(seq) != 4) throw std::invalid_argument("box must have four elements");
        return {seq[0].cast<float>(), seq[1].cast<float>(),
                seq[2].cast<float>(), seq[3].cast<float>()};
    }

    static std::vector<float> normalize_ip(const std::vector<float>& v) {
        float norm_sq = 0.0f;
        for (float x : v) norm_sq += x * x;
        const float inv = 1.0f / std::max(std::sqrt(norm_sq), 1e-12f);
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] * inv;
        return out;
    }

    py::object relinker_obj_;   // keeps relinker alive
    py::object lifecycle_obj_;  // keeps lifecycle alive
    SemanticRelinkerCpp* relinker_;
    TrackletLifecycleMergerCpp* lifecycle_;
};

} // namespace

PYBIND11_MODULE(saccade_tracking_ext, m) {
    m.doc() = "Saccade GPU Tracker (Python Bindings)";

    py::class_<TrackResult>(m, "TrackResult")
        .def_readonly("x1", &TrackResult::x1)
        .def_readonly("y1", &TrackResult::y1)
        .def_readonly("x2", &TrackResult::x2)
        .def_readonly("y2", &TrackResult::y2)
        .def_readonly("obj_id", &TrackResult::obj_id)
        .def_readonly("score", &TrackResult::score)
        .def_readonly("class_id", &TrackResult::class_id)
        .def_readonly("det_idx", &TrackResult::det_idx);

    py::class_<TrackStateSnapshot>(m, "TrackStateSnapshot")
        .def_readonly("obj_id", &TrackStateSnapshot::obj_id)
        .def_readonly("class_id", &TrackStateSnapshot::class_id)
        .def_readonly("age", &TrackStateSnapshot::age)
        .def_readonly("score", &TrackStateSnapshot::score)
        .def_readonly("state", &TrackStateSnapshot::state)
        .def_readonly("covariance", &TrackStateSnapshot::covariance);

    py::class_<TrackCandidateSnapshot>(m, "TrackCandidateSnapshot")
        .def_readonly("obj_id", &TrackCandidateSnapshot::obj_id)
        .def_readonly("class_id", &TrackCandidateSnapshot::class_id)
        .def_readonly("age", &TrackCandidateSnapshot::age)
        .def_readonly("hit_streak", &TrackCandidateSnapshot::hit_streak)
        .def_readonly("required_confirm_streak", &TrackCandidateSnapshot::required_confirm_streak)
        .def_readonly("score", &TrackCandidateSnapshot::score)
        .def_readonly("x1", &TrackCandidateSnapshot::x1)
        .def_readonly("y1", &TrackCandidateSnapshot::y1)
        .def_readonly("x2", &TrackCandidateSnapshot::x2)
        .def_readonly("y2", &TrackCandidateSnapshot::y2);

    py::class_<GPUByteTracker>(m, "GPUByteTracker")
        .def(py::init<int, int>(), py::arg("max_objects") = 2048, py::arg("embedding_dim") = 768)
        .def("set_params", &GPUByteTracker::set_params,
             py::arg("track_thresh"),
             py::arg("high_thresh"),
             py::arg("match_thresh"),
             py::arg("track_buffer"),
             py::arg("mid_thresh") = 0.40f,
             py::arg("confirm_streak") = 3,
             py::arg("confirm_score_thresh") = 0.50f,
             py::arg("adaptive_confirmation") = false,
             py::arg("new_track_thresh") = -1.0f,
             py::arg("nsa_kalman") = false)
        .def("set_reid_params", &GPUByteTracker::set_reid_params,
             py::arg("cos_threshold"), py::arg("iou_low"), py::arg("iou_high"), py::arg("weight"))
        .def("update_reference_features", [](GPUByteTracker& self, uintptr_t ids_ptr, uintptr_t features_ptr, int num, uintptr_t stream_ptr) {
            self.update_reference_features(
                reinterpret_cast<int*>(ids_ptr),
                reinterpret_cast<float*>(features_ptr),
                num,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        }, py::arg("ids_ptr"), py::arg("features_ptr"), py::arg("num"), py::arg("stream_ptr"))
        .def("update", [](GPUByteTracker& self, uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr, int num_dets, uintptr_t stream_ptr,
                          uintptr_t embeddings_ptr, uintptr_t gmc_ptr, float light_factor, float mid_thresh_scale) {
            return self.update(
                reinterpret_cast<float*>(boxes_ptr),
                reinterpret_cast<float*>(scores_ptr),
                reinterpret_cast<int*>(classes_ptr),
                num_dets,
                reinterpret_cast<cudaStream_t>(stream_ptr),
                embeddings_ptr ? reinterpret_cast<float*>(embeddings_ptr) : nullptr,
                gmc_ptr ? reinterpret_cast<float*>(gmc_ptr) : nullptr,
                light_factor,
                mid_thresh_scale
            );
        }, 
        py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"), py::arg("num_dets"), py::arg("stream_ptr"),
        py::arg("embeddings_ptr") = 0, py::arg("gmc_ptr") = 0, py::arg("light_factor") = 0.0f, py::arg("mid_thresh_scale") = 1.0f,
        "Update tracker with raw GPU pointers and stream")
        .def("update_into", [](GPUByteTracker& self,
                               uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr, int num_dets, uintptr_t stream_ptr,
                               uintptr_t out_boxes_ptr, uintptr_t out_scores_ptr, uintptr_t out_ids_ptr, uintptr_t out_classes_ptr,
                               uintptr_t out_det_idx_ptr, uintptr_t out_count_ptr,
                               uintptr_t embeddings_ptr, uintptr_t gmc_ptr, float light_factor, float mid_thresh_scale) {
            self.update_into(
                reinterpret_cast<float*>(boxes_ptr),
                reinterpret_cast<float*>(scores_ptr),
                reinterpret_cast<int*>(classes_ptr),
                num_dets,
                reinterpret_cast<cudaStream_t>(stream_ptr),
                reinterpret_cast<float*>(out_boxes_ptr),
                reinterpret_cast<float*>(out_scores_ptr),
                reinterpret_cast<int*>(out_ids_ptr),
                reinterpret_cast<int*>(out_classes_ptr),
                reinterpret_cast<int*>(out_det_idx_ptr),
                reinterpret_cast<int*>(out_count_ptr),
                embeddings_ptr ? reinterpret_cast<float*>(embeddings_ptr) : nullptr,
                gmc_ptr ? reinterpret_cast<float*>(gmc_ptr) : nullptr,
                light_factor,
                mid_thresh_scale
            );
        },
        py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"), py::arg("num_dets"), py::arg("stream_ptr"),
        py::arg("out_boxes_ptr"), py::arg("out_scores_ptr"), py::arg("out_ids_ptr"), py::arg("out_classes_ptr"),
        py::arg("out_det_idx_ptr"), py::arg("out_count_ptr"),
        py::arg("embeddings_ptr") = 0, py::arg("gmc_ptr") = 0, py::arg("light_factor") = 0.0f, py::arg("mid_thresh_scale") = 1.0f,
        "Update tracker and write compact results into caller-provided GPU buffers")
        .def("get_state_snapshots", [](GPUByteTracker& self, uintptr_t stream_ptr) {
            return self.get_state_snapshots(reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("stream_ptr"),
        "Return active Kalman state and covariance snapshots")
        .def("get_motion_snapshots_for_track_ids",
             [](GPUByteTracker& self, const std::vector<int>& track_ids, uintptr_t stream_ptr) {
                 return self.get_motion_snapshots_for_track_ids(
                     track_ids,
                     reinterpret_cast<cudaStream_t>(stream_ptr)
                 );
             },
             py::arg("track_ids"),
             py::arg("stream_ptr"),
             "Return Kalman motion snapshots only for the requested track IDs")
        .def("get_tentative_candidates", [](GPUByteTracker& self, uintptr_t stream_ptr) {
            return self.get_tentative_candidates(reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("stream_ptr"),
        "Return active tentative tracks that are candidates for lazy ReID arbitration")
        .def("set_clean_embedding_flags", [](GPUByteTracker& self,
                                              uintptr_t ids_ptr, uintptr_t flags_ptr,
                                              int n, uintptr_t stream_ptr) {
            self.set_clean_embedding_flags(
                reinterpret_cast<int*>(ids_ptr),
                reinterpret_cast<bool*>(flags_ptr),
                n,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        },
        py::arg("ids_ptr"), py::arg("flags_ptr"), py::arg("n"), py::arg("stream_ptr"),
        "Sync per-track clean-embedding flags from Python bank state to the CUDA tracker");

    py::class_<SemanticRelinkerCpp>(m, "SemanticRelinker")
        .def(py::init<float, int, float, float, int, float, float, int, float, std::string, float, bool>(),
             py::arg("sim_threshold") = 0.985f,
             py::arg("ttl") = 45,
             py::arg("ema_beta") = 0.83f,
             py::arg("spatial_gate") = 0.11f,
             py::arg("min_lost_frames") = 2,
             py::arg("min_iou") = 0.0f,
             py::arg("mahalanobis_threshold") = 6.6f,
             py::arg("buffer_size") = 1,
             py::arg("min_consistency") = 0.0f,
             py::arg("rerank_mode") = "mean",
             py::arg("reciprocal_margin") = 0.0f,
             py::arg("debug") = false)
        .def("update_motion_snapshots", &SemanticRelinkerCpp::update_motion_snapshots, py::arg("snapshots"))
        .def("motion_candidate_ids", &SemanticRelinkerCpp::motion_candidate_ids, py::arg("frame_id") = -1)
        .def("inject_reference", &SemanticRelinkerCpp::inject_reference,
             py::arg("canonical_id"), py::arg("embedding"))
        .def("inject_references_many", &SemanticRelinkerCpp::inject_references_many,
             py::arg("references"))
        .def("canonical_id", &SemanticRelinkerCpp::canonical_id, py::arg("raw_id"))
        .def("has_feature", &SemanticRelinkerCpp::has_feature, py::arg("canonical_id"))
        .def("resolve", &SemanticRelinkerCpp::resolve,
             py::arg("raw_id"),
             py::arg("embedding"),
             py::arg("box"),
             py::arg("score"),
             py::arg("frame_id"),
             py::arg("w"),
             py::arg("h"),
             py::arg("assigned"))
        .def("resolve_many", &SemanticRelinkerCpp::resolve_many,
             py::arg("candidates"),
             py::arg("frame_id"),
             py::arg("w"),
             py::arg("h"))
        .def("resolve_many_packed", &SemanticRelinkerCpp::resolve_many_packed,
             py::arg("raw_ids"),
             py::arg("embeddings"),
             py::arg("boxes"),
             py::arg("scores"),
             py::arg("frame_id"),
             py::arg("w"),
             py::arg("h"))
        .def_property_readonly("alias", &SemanticRelinkerCpp::get_alias)
        .def_property_readonly("features", &SemanticRelinkerCpp::get_features)
        .def_property_readonly("stats", &SemanticRelinkerCpp::stats)
        .def("report", &SemanticRelinkerCpp::report);

    py::class_<TrackletLifecycleMergerCpp>(m, "TrackletLifecycleMerger")
        .def(py::init<bool, int, int, float, float, float, bool, float>(),
             py::arg("enabled")            = true,
             py::arg("ttl")                = 30,
             py::arg("min_gap")            = 1,
             py::arg("spatial_gate")       = 0.3f,
             py::arg("min_iou")            = 0.1f,
             py::arg("sim_threshold")      = 0.5f,
             py::arg("require_embedding")  = false,
             py::arg("ema")                = 0.7f)
        .def("resolve_many_packed", &TrackletLifecycleMergerCpp::resolve_many_packed,
             py::arg("local_ids"), py::arg("boxes"), py::arg("scores"), py::arg("embeddings"),
             py::arg("frame_id"), py::arg("frame_w"), py::arg("frame_h"))
        .def("resolve_many", &TrackletLifecycleMergerCpp::resolve_many,
             py::arg("candidates"), py::arg("frame_id"), py::arg("frame_w"), py::arg("frame_h"))
        .def("prune",    &TrackletLifecycleMergerCpp::prune,    py::arg("frame_id"))
        .def("report",   &TrackletLifecycleMergerCpp::report)
        .def_property_readonly("alias", &TrackletLifecycleMergerCpp::get_alias)
        .def_property_readonly("stats", &TrackletLifecycleMergerCpp::stats_dict);

    py::class_<IdentityResolverCpp>(m, "IdentityResolver")
        .def(py::init<py::object, py::object>(),
             py::arg("relinker"), py::arg("lifecycle_merger"))
        .def("resolve_pass", &IdentityResolverCpp::resolve_pass,
             py::arg("local_ids"), py::arg("embeddings"), py::arg("boxes"), py::arg("scores"),
             py::arg("frame_id"), py::arg("frame_w"), py::arg("frame_h"));

    py::class_<GMC>(m, "GMC")
        .def(py::init<int, int, float, float, int, float>(),
             py::arg("downscale") = 8,
             py::arg("max_corners") = 100,
             py::arg("quality_level") = 0.01f,
             py::arg("min_distance") = 10.0f,
             py::arg("min_inliers") = 8,
             py::arg("ransac_threshold") = 3.0f)
        .def("estimate", [](GMC& self, uintptr_t frame_ptr, int width, int height, uintptr_t stream_ptr) {
            auto warp = self.estimate(reinterpret_cast<const float*>(frame_ptr), width, height, reinterpret_cast<cudaStream_t>(stream_ptr));
            if (warp.empty()) return py::none().cast<py::object>();
            return py::cast(warp);
        }, py::arg("frame_ptr"), py::arg("width"), py::arg("height"), py::arg("stream_ptr"))
        .def("estimate_mat", [](GMC& self, py::array_t<uint8_t> frame, int downscale) {
            py::buffer_info info = frame.request();
            if (info.ndim != 2 && info.ndim != 3) {
                throw std::runtime_error("Frame must be 2D (gray) or 3D (BGR/RGB)");
            }
            int h = static_cast<int>(info.shape[0]);
            int w = static_cast<int>(info.shape[1]);
            int channels = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
            int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
            cv::Mat mat(h, w, type, info.ptr);
            auto warp = self.estimate_mat(mat, downscale);
            if (warp.empty()) return py::none().cast<py::object>();
            return py::cast(warp);
        }, py::arg("frame"), py::arg("downscale") = -1)
        .def("reset", &GMC::reset);

    m.def(
        "merge_cross_tile_duplicates",
        &merge_cross_tile_duplicates_cpu,
        py::arg("boxes"),
        py::arg("scores"),
        py::arg("classes"),
        py::arg("iou_threshold") = 0.45f,
        py::arg("center_threshold") = 0.18f,
        py::arg("area_ratio_threshold") = 0.6f,
        "Merge duplicate detections across overlapping tiles on CPU."
    );

    m.def(
        "filter_detections",
        &filter_detections_cpu,
        py::arg("boxes"),
        py::arg("scores"),
        py::arg("classes"),
        py::arg("score_threshold"),
        py::arg("track_person_only"),
        py::arg("person_class"),
        py::arg("is_tiled"),
        py::arg("frame_w"),
        py::arg("frame_h"),
        py::arg("person_geometry_prior"),
        py::arg("geometry_suspect_support"),
        py::arg("person_min_height_ratio"),
        py::arg("person_min_aspect"),
        py::arg("person_max_aspect"),
        py::arg("person_min_area_ratio"),
        py::arg("person_max_area_ratio"),
        "Return kept detection indices and geometry-suspect flags."
    );

    m.def(
        "filter_detections_cuda",
        [](
            uintptr_t boxes_ptr,
            uintptr_t scores_ptr,
            uintptr_t classes_ptr,
            int num_dets,
            uintptr_t keep_indices_ptr,
            uintptr_t suspect_flags_ptr,
            uintptr_t out_count_ptr,
            float score_threshold,
            bool track_person_only,
            int person_class,
            bool is_tiled,
            int frame_w,
            int frame_h,
            bool person_geometry_prior,
            bool geometry_suspect_support,
            float person_min_height_ratio,
            float person_min_aspect,
            float person_max_aspect,
            float person_min_area_ratio,
            float person_max_area_ratio,
            uintptr_t stream_ptr
        ) {
            filter_detections_cuda(
                reinterpret_cast<const float*>(boxes_ptr),
                reinterpret_cast<const float*>(scores_ptr),
                reinterpret_cast<const int*>(classes_ptr),
                num_dets,
                reinterpret_cast<int*>(keep_indices_ptr),
                reinterpret_cast<bool*>(suspect_flags_ptr),
                reinterpret_cast<int*>(out_count_ptr),
                score_threshold,
                track_person_only,
                person_class,
                is_tiled,
                frame_w,
                frame_h,
                person_geometry_prior,
                geometry_suspect_support,
                person_min_height_ratio,
                person_min_aspect,
                person_max_aspect,
                person_min_area_ratio,
                person_max_area_ratio,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        },
        py::arg("boxes_ptr"),
        py::arg("scores_ptr"),
        py::arg("classes_ptr"),
        py::arg("num_dets"),
        py::arg("keep_indices_ptr"),
        py::arg("suspect_flags_ptr"),
        py::arg("out_count_ptr"),
        py::arg("score_threshold"),
        py::arg("track_person_only"),
        py::arg("person_class"),
        py::arg("is_tiled"),
        py::arg("frame_w"),
        py::arg("frame_h"),
        py::arg("person_geometry_prior"),
        py::arg("geometry_suspect_support"),
        py::arg("person_min_height_ratio"),
        py::arg("person_min_aspect"),
        py::arg("person_max_aspect"),
        py::arg("person_min_area_ratio"),
        py::arg("person_max_area_ratio"),
        py::arg("stream_ptr"),
        "Return kept detection indices and geometry-suspect flags on the caller's CUDA stream."
    );

    m.def(
        "nms_cuda",
        [](
            uintptr_t boxes_ptr,
            uintptr_t scores_ptr,
            uintptr_t classes_ptr,
            uintptr_t order_indices_ptr,
            int num_dets,
            uintptr_t keep_indices_ptr,
            uintptr_t suppression_masks_ptr,
            uintptr_t remv_ptr,
            uintptr_t out_count_ptr,
            float iou_threshold,
            bool class_aware,
            uintptr_t stream_ptr
        ) {
            nms_cuda(
                reinterpret_cast<const float*>(boxes_ptr),
                reinterpret_cast<const float*>(scores_ptr),
                reinterpret_cast<const int*>(classes_ptr),
                reinterpret_cast<const int64_t*>(order_indices_ptr),
                num_dets,
                reinterpret_cast<int*>(keep_indices_ptr),
                reinterpret_cast<uint64_t*>(suppression_masks_ptr),
                reinterpret_cast<uint64_t*>(remv_ptr),
                reinterpret_cast<int*>(out_count_ptr),
                iou_threshold,
                class_aware,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        },
        py::arg("boxes_ptr"),
        py::arg("scores_ptr"),
        py::arg("classes_ptr"),
        py::arg("order_indices_ptr"),
        py::arg("num_dets"),
        py::arg("keep_indices_ptr"),
        py::arg("suppression_masks_ptr"),
        py::arg("remv_ptr"),
        py::arg("out_count_ptr"),
        py::arg("iou_threshold"),
        py::arg("class_aware"),
        py::arg("stream_ptr"),
        "Run parallel bitmask greedy NMS on the caller's CUDA stream."
    );

    m.def(
        "merge_cross_tile_duplicates_cuda",
        [](
            uintptr_t boxes_ptr,
            uintptr_t scores_ptr,
            uintptr_t classes_ptr,
            int num_dets,
            uintptr_t anchor_indices_ptr,
            uintptr_t box_sums_ptr,
            uintptr_t score_sums_ptr,
            uintptr_t score_bits_max_ptr,
            uintptr_t cluster_counts_ptr,
            uintptr_t out_boxes_ptr,
            uintptr_t out_scores_ptr,
            uintptr_t out_classes_ptr,
            uintptr_t out_count_ptr,
            float iou_threshold,
            float center_threshold,
            float area_ratio_threshold,
            uintptr_t stream_ptr
        ) {
            merge_cross_tile_duplicates_cuda(
                reinterpret_cast<const float*>(boxes_ptr),
                reinterpret_cast<const float*>(scores_ptr),
                reinterpret_cast<const int*>(classes_ptr),
                num_dets,
                reinterpret_cast<int*>(anchor_indices_ptr),
                reinterpret_cast<float*>(box_sums_ptr),
                reinterpret_cast<float*>(score_sums_ptr),
                reinterpret_cast<int*>(score_bits_max_ptr),
                reinterpret_cast<int*>(cluster_counts_ptr),
                reinterpret_cast<float*>(out_boxes_ptr),
                reinterpret_cast<float*>(out_scores_ptr),
                reinterpret_cast<int*>(out_classes_ptr),
                reinterpret_cast<int*>(out_count_ptr),
                iou_threshold,
                center_threshold,
                area_ratio_threshold,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        },
        py::arg("boxes_ptr"),
        py::arg("scores_ptr"),
        py::arg("classes_ptr"),
        py::arg("num_dets"),
        py::arg("anchor_indices_ptr"),
        py::arg("box_sums_ptr"),
        py::arg("score_sums_ptr"),
        py::arg("score_bits_max_ptr"),
        py::arg("cluster_counts_ptr"),
        py::arg("out_boxes_ptr"),
        py::arg("out_scores_ptr"),
        py::arg("out_classes_ptr"),
        py::arg("out_count_ptr"),
        py::arg("iou_threshold") = 0.45f,
        py::arg("center_threshold") = 0.18f,
        py::arg("area_ratio_threshold") = 0.6f,
        py::arg("stream_ptr"),
        "Merge duplicate detections across overlapping tiles on the caller's CUDA stream."
    );

    // PerceptionPipeline — C++ facade for filter+NMS+reid hot path
    py::class_<PerceptionPipeline::Config>(m, "PerceptionPipelineConfig")
        .def(py::init<>())
        .def_readwrite("score_threshold",           &PerceptionPipeline::Config::score_threshold)
        .def_readwrite("person_class",              &PerceptionPipeline::Config::person_class)
        .def_readwrite("person_only",               &PerceptionPipeline::Config::person_only)
        .def_readwrite("nms_threshold",             &PerceptionPipeline::Config::nms_threshold)
        .def_readwrite("person_geometry_prior",     &PerceptionPipeline::Config::person_geometry_prior)
        .def_readwrite("geometry_suspect_support",  &PerceptionPipeline::Config::geometry_suspect_support)
        .def_readwrite("person_min_height_ratio",   &PerceptionPipeline::Config::person_min_height_ratio)
        .def_readwrite("person_min_aspect",         &PerceptionPipeline::Config::person_min_aspect)
        .def_readwrite("person_max_aspect",         &PerceptionPipeline::Config::person_max_aspect)
        .def_readwrite("person_min_area_ratio",     &PerceptionPipeline::Config::person_min_area_ratio)
        .def_readwrite("person_max_area_ratio",     &PerceptionPipeline::Config::person_max_area_ratio)
        .def_readwrite("max_detections",            &PerceptionPipeline::Config::max_detections);

    py::class_<PerceptionPipeline>(m, "PerceptionPipeline")
        .def(py::init([](uintptr_t reid_ptr, uintptr_t cropper_ptr,
                         const PerceptionPipeline::Config& cfg) {
            return new PerceptionPipeline(
                reinterpret_cast<FeatureExtractor*>(reid_ptr),
                reinterpret_cast<Cropper*>(cropper_ptr),
                cfg);
        }), py::arg("reid_ptr"), py::arg("cropper_ptr"), py::arg("config"))
        .def("process_detections",
            [](PerceptionPipeline& self,
               uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr,
               int n_in, int frame_w, int frame_h, bool is_tiled,
               uintptr_t out_boxes, uintptr_t out_scores, uintptr_t out_classes,
               uintptr_t out_suspect, uintptr_t stream_ptr) -> int {
                return self.process_detections(
                    reinterpret_cast<const float*>(boxes_ptr),
                    reinterpret_cast<const float*>(scores_ptr),
                    reinterpret_cast<const int*>(classes_ptr),
                    n_in, frame_w, frame_h, is_tiled,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<bool*>(out_suspect),
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"), py::arg("stream_ptr"))
        .def("process_detections_into",
            [](PerceptionPipeline& self,
               uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr,
               int n_in, int frame_w, int frame_h, bool is_tiled,
               uintptr_t out_boxes, uintptr_t out_scores, uintptr_t out_classes,
               uintptr_t out_suspect, uintptr_t out_count, uintptr_t stream_ptr) {
                self.process_detections_into(
                    reinterpret_cast<const float*>(boxes_ptr),
                    reinterpret_cast<const float*>(scores_ptr),
                    reinterpret_cast<const int*>(classes_ptr),
                    n_in, frame_w, frame_h, is_tiled,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<bool*>(out_suspect),
                    reinterpret_cast<int*>(out_count),
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"), py::arg("out_count"), py::arg("stream_ptr"))
        .def("extract_reid",
            [](PerceptionPipeline& self,
               uintptr_t frame_ptr, int frame_h, int frame_w,
               uintptr_t boxes_ptr, int n_boxes,
               uintptr_t out_embeds, uintptr_t stream_ptr) {
                self.extract_reid(
                    reinterpret_cast<const float*>(frame_ptr),
                    frame_h, frame_w,
                    reinterpret_cast<const float*>(boxes_ptr),
                    n_boxes,
                    reinterpret_cast<float*>(out_embeds),
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("frame_ptr"), py::arg("frame_h"), py::arg("frame_w"),
            py::arg("boxes_ptr"), py::arg("n_boxes"),
            py::arg("out_embeds"), py::arg("stream_ptr"))
        .def("set_reid_profiling_enabled", &PerceptionPipeline::set_reid_profiling_enabled, py::arg("enabled"))
        .def("reset_reid_profile_stats", &PerceptionPipeline::reset_reid_profile_stats)
        .def("get_reid_profile_stats",
            [](const PerceptionPipeline& self) {
                const auto stats = self.get_reid_profile_stats();
                py::dict out;
                out["crop_ms"] = stats.crop_ms;
                out["extract_pre_normalize_ms"] = stats.extract_pre_normalize_ms;
                out["extract_trt_enqueue_ms"] = stats.extract_trt_enqueue_ms;
                out["extract_l2_normalize_ms"] = stats.extract_l2_normalize_ms;
                out["extract_total_ms"] = stats.extract_total_ms;
                out["total_ms"] = stats.total_ms;
                out["chunks"] = stats.chunks;
                out["images"] = stats.images;
                return out;
            })
        .def_property_readonly("embed_dim", &PerceptionPipeline::get_embed_dim);
}
