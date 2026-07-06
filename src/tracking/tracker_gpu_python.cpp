#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "tracking/auction.hpp"
#include "tracking/box_ops.hpp"
#include "tracking/tracker_gpu.hpp"
#include "tracking/gmc.hpp"
#include "tracking/cheb_gr_kreciprocal.hpp"
#include "tracking/dynamic_reid_controller.hpp"
#include <Eigen/Dense>
#include "tracking/pipeline.hpp"
#include "tracking/workbench.hpp"
#include "tracking/mamba_scan.cuh"
#include "tracking/quality_filter.cuh"
#include "tracking/copy_pad.cuh"
#include "tracking/relink_gate.hpp"
#include "tracking/kalman_gpu.cuh"
#include "perception/feature_extractor.hpp"
#include "perception/preprocessor.hpp"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace saccade;

// Forward declaration from letterbox_kernel.cu (saccade_perception)
namespace saccade {
void launch_letterbox_gpu(
    const float* src, int src_w, int src_h,
    float* dst, int dst_size,
    int x_off, int y_off, int w_new, int h_new,
    float pad_val, cudaStream_t stream);
}

namespace {

bool is_box_near_tiled_seam_cpu(
    const float* box,
    int tiling_mode,
    int frame_w,
    int frame_h,
    float seam_margin_canvas_px
) {
    return tracking::is_box_near_tiled_seam(
        tracking::load_box4(box), tiling_mode, frame_w, frame_h, seam_margin_canvas_px
    );
}

py::tuple merge_cross_tile_duplicates_cpu(
    py::array_t<float, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    py::array_t<int, py::array::c_style | py::array::forcecast> classes,
    float iou_threshold,
    float center_threshold,
    float area_ratio_threshold,
    int tiling_mode,
    int frame_w,
    int frame_h,
    float seam_margin_canvas_px,
    float seam_center_scale,
    float seam_area_ratio_threshold,
    float seam_min_overlap_ratio
) {
    constexpr float kTiledSeamCoordWeight = 0.35f;
    constexpr float kTiledBestBlend = 0.25f;
    const tracking::DuplicateMergeParams params{
        iou_threshold,
        center_threshold,
        area_ratio_threshold,
        tiling_mode,
        frame_w,
        frame_h,
        seam_margin_canvas_px,
        seam_center_scale,
        seam_area_ratio_threshold,
        seam_min_overlap_ratio,
    };
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
        const tracking::Box4f anchor = tracking::load_box4(anchor_box);

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
            if (tracking::duplicate_match(anchor, tracking::load_box4(candidate_box), params)) {
                cluster_indices.push_back(candidate_idx);
            }
        }

        float fused_box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float fused_score = 0.0f;
        if (tiling_mode > 0) {
            int best_idx = cluster_indices.front();
            bool best_is_seam = true;
            float best_score = -1.0f;
            float coord_weight_sum = 0.0f;
            for (int idx : cluster_indices) {
                const float candidate_box[4] = {
                    boxes_in(idx, 0), boxes_in(idx, 1), boxes_in(idx, 2), boxes_in(idx, 3),
                };
                const bool is_seam = is_box_near_tiled_seam_cpu(
                    candidate_box, tiling_mode, frame_w, frame_h, seam_margin_canvas_px
                );
                const float score = scores_in(idx);
                if ((best_is_seam && !is_seam) || (best_is_seam == is_seam && score > best_score)) {
                    best_idx = idx;
                    best_is_seam = is_seam;
                    best_score = score;
                }
                const float coord_weight =
                    score * (is_seam ? kTiledSeamCoordWeight : 1.0f);
                coord_weight_sum += coord_weight;
                for (int k = 0; k < 4; ++k) {
                    fused_box[k] += boxes_in(idx, k) * coord_weight;
                }
            }
            const float inv_coord_weight_sum = 1.0f / std::max(coord_weight_sum, 1e-6f);
            for (float& coord : fused_box) coord *= inv_coord_weight_sum;
            if (cluster_indices.size() > 1) {
                for (int k = 0; k < 4; ++k) {
                    fused_box[k] =
                        fused_box[k] * (1.0f - kTiledBestBlend)
                        + boxes_in(best_idx, k) * kTiledBestBlend;
                }
            } else {
                for (int k = 0; k < 4; ++k) fused_box[k] = boxes_in(best_idx, k);
            }
            fused_score = best_score;
        } else {
            float weight_sum = 0.0f;
            for (int idx : cluster_indices) {
                const float score = scores_in(idx);
                weight_sum += score;
                fused_score = std::max(fused_score, score);
                for (int k = 0; k < 4; ++k) {
                    fused_box[k] += boxes_in(idx, k) * score;
                }
            }
            const float inv_weight_sum = 1.0f / std::max(weight_sum, 1e-6f);
            for (float& coord : fused_box) coord *= inv_weight_sum;
        }
        for (int idx : cluster_indices) {
            consumed[static_cast<size_t>(idx)] = 1;
        }

        for (float coord : fused_box) out_boxes.push_back(coord);
        out_scores.push_back(fused_score);
        out_classes.push_back(anchor_class);
    }

    const ssize_t out_num = static_cast<ssize_t>(out_scores.size());
    py::array_t<float> out_boxes_arr({out_num, static_cast<ssize_t>(4)});
    py::array_t<float> out_scores_arr(std::vector<ssize_t>{out_num});
    py::array_t<int> out_classes_arr(std::vector<ssize_t>{out_num});

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
    const tracking::DetectionFilterParams params{
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
    };

    auto boxes_in = boxes.unchecked<2>();
    auto scores_in = scores.unchecked<1>();
    auto classes_in = classes.unchecked<1>();
    std::vector<int> keep_indices;
    std::vector<unsigned char> suspect_flags;
    keep_indices.reserve(static_cast<size_t>(num));
    suspect_flags.reserve(static_cast<size_t>(num));

    for (ssize_t i = 0; i < num; ++i) {
        bool geometry_clean = true;
        const float raw_box[4] = {
            boxes_in(i, 0), boxes_in(i, 1), boxes_in(i, 2), boxes_in(i, 3),
        };
        const bool keep = tracking::detection_keep(
            tracking::load_box4(raw_box), scores_in(i), classes_in(i), params, geometry_clean
        );

        if (keep) {
            keep_indices.push_back(static_cast<int>(i));
            suspect_flags.push_back(static_cast<unsigned char>(
                person_geometry_prior && geometry_suspect_support && !geometry_clean
            ));
        }
    }

    const ssize_t out_num = static_cast<ssize_t>(keep_indices.size());
    py::array_t<int> keep_arr(std::vector<ssize_t>{out_num});
    py::array_t<bool> suspect_arr(std::vector<ssize_t>{out_num});
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

tracking::Box4f to_box4(const RelinkBox& box) {
    return {box.x1, box.y1, box.x2, box.y2};
}

struct RelinkMotionSnapshot {
    std::array<float, 8> state{};       // [cx, cy, a, h, vx, vy, va, vh]
    std::array<float, 64> covariance{}; // row-major 8x8 Kalman covariance
    int frame = -1;                     // frame the snapshot was captured at
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
    int reject_quality = 0;
    int reject_kalman = 0;
    int reject_direction = 0;
    int reject_speed = 0;
    int delayed_claim_pending = 0;
    int delayed_claim_ready = 0;
    int delayed_claim_accepted = 0;
    int cheb_gr_claim_attempts = 0;
    int cheb_gr_claim_accepted = 0;
    int new_ids = 0;
    int reject_backward = 0;
    int accept_bidir = 0;
    int relink_split_collision = 0;
};

struct RelinkCandidateGates {
    int cid = -1;
    int age = 0;
    float iou = 0.0f;
    float center_norm = 0.0f;
    float maha = 0.0f;
    float kalman_d2 = -1.0f;  // <0 when Kalman gate inactive for this candidate
};

struct PendingClaim {
    int first = -1;
    int last = -1;
    int hits = 0;
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
        bool debug,
        float clean_score_threshold = 0.0f,
        float clean_margin_ratio = 0.0f,
        float clean_min_aspect = 0.0f,
        float clean_max_aspect = 99.0f,
        float strict_sim_threshold = 0.0f,
        float w_sim_base = 0.0f,
        float w_iou_base = 0.0f,
        float w_maha_base = 0.0f,
        float shift_ambiguity = 0.0f,
        float shift_lost_age = 0.0f,
        float iou_weight = 0.0f,
        float mahalanobis_weight = 0.0f,
        float dynamic_margin_crowd = 0.0f,
        float dynamic_margin_age = 0.0f,
        bool kalman_gate = false,
        float kalman_chi2 = 9.4877f,
        float kalman_penalty_weight = 0.0f,
        float kalman_dir_min_cos = -1.0f,
        float kalman_dir_min_speed = 1.0f,
        float kalman_person_height_m = 0.0f,
        float kalman_accel_long = 2.0f,
        float kalman_accel_lat = 1.0f,
        float kalman_fps = 30.0f,
        float kalman_max_speed_mps = 0.0f,
        bool delayed_claim = false,
        int claim_warmup_frames = 3,
        bool bidirectional = false,
        float bridge_px = 1.5f,
        float bridge_h_lo = 0.0f,
        float bridge_h_hi = 0.0f,
        bool exp_density_gating = false,
        float exp_density_k = 2.0f,
        float exp_density_eta = 0.15f,
        bool cheb_gr_claim = false,
        float cheb_gr_max_cost = 0.45f,
        float cheb_gr_margin = 0.05f,
        int cheb_gr_min_head = 2,
        float cheb_gr_pool_frac = 0.3f,
        float cheb_gr_min_sim = 0.0f,
        float cheb_gr_lambda = 2.0f,
        int cheb_gr_k2 = 6,
        int cheb_gr_max_fwd = 50,
        float cheb_gr_fuse_lambda = 0.3f
    )
        : sim_threshold_(sim_threshold),
          ttl_(ttl),
          ema_beta_(std::clamp(ema_beta, 0.0f, 1.0f)),
          spatial_gate_(spatial_gate),
          min_lost_frames_(min_lost_frames),
          min_iou_(min_iou),
          mahalanobis_threshold_(mahalanobis_threshold),
          exp_density_gating_(exp_density_gating),
          exp_density_k_(exp_density_k),
          exp_density_eta_(exp_density_eta),
          buffer_size_(std::max(1, buffer_size)),
          min_consistency_(min_consistency),
          rerank_mode_(std::move(rerank_mode)),
          reciprocal_margin_(std::max(0.0f, reciprocal_margin)),
          debug_(debug),
          clean_score_threshold_(clean_score_threshold),
          clean_margin_ratio_(clean_margin_ratio),
          clean_min_aspect_(clean_min_aspect),
          clean_max_aspect_(clean_max_aspect),
          strict_sim_threshold_(strict_sim_threshold > 0.0f ? strict_sim_threshold : sim_threshold),
          w_sim_base_(std::max(0.0f, w_sim_base)),
          w_iou_base_(std::max(0.0f, w_iou_base)),
          w_maha_base_(std::max(0.0f, w_maha_base)),
          shift_ambiguity_(shift_ambiguity),
          shift_lost_age_(shift_lost_age),
          iou_weight_(std::max(0.0f, iou_weight)),
          mahalanobis_weight_(std::max(0.0f, mahalanobis_weight)),
          dynamic_margin_crowd_(std::max(0.0f, dynamic_margin_crowd)),
          dynamic_margin_age_(std::max(0.0f, dynamic_margin_age)),
          kalman_gate_(kalman_gate),
          kalman_chi2_(kalman_chi2),
          kalman_penalty_weight_(std::max(0.0f, kalman_penalty_weight)),
          kalman_dir_min_cos_(kalman_dir_min_cos),
          kalman_dir_min_speed_(std::max(0.0f, kalman_dir_min_speed)),
          kalman_person_height_m_(std::max(0.0f, kalman_person_height_m)),
          kalman_accel_long_(std::max(0.0f, kalman_accel_long)),
          kalman_accel_lat_(std::max(0.0f, kalman_accel_lat)),
          kalman_fps_(kalman_fps > 0.0f ? kalman_fps : 30.0f),
          kalman_max_speed_mps_(std::max(0.0f, kalman_max_speed_mps)),
          delayed_claim_(delayed_claim),
          claim_warmup_frames_(std::max(1, claim_warmup_frames)),
          bidirectional_(bidirectional),
          bridge_px_(bridge_px),
          bridge_h_lo_(std::max(0.0f, bridge_h_lo)),
          bridge_h_hi_(std::max(0.0f, bridge_h_hi)),
          cheb_gr_claim_(cheb_gr_claim),
          cheb_gr_max_cost_(cheb_gr_max_cost),
          cheb_gr_margin_(std::max(0.0f, cheb_gr_margin)),
          cheb_gr_min_head_(std::max(1, std::min(cheb_gr_min_head, cheb_gr_min_head))),
          cheb_gr_pool_frac_(std::max(0.0f, cheb_gr_pool_frac)),
          cheb_gr_min_sim_(cheb_gr_min_sim),
          cheb_gr_lambda_(cheb_gr_lambda),
          cheb_gr_k2_(std::max(1, cheb_gr_k2)),
          cheb_gr_max_fwd_(cheb_gr_max_fwd),
          cheb_gr_fuse_lambda_(cheb_gr_fuse_lambda),
          split_counter_(0) {}

    void update_motion_snapshots(const std::vector<TrackStateSnapshot>& snapshots, int frame_id = -1) {
        for (const auto& snap : snapshots) {
            const int canonical = alias_.count(snap.obj_id) ? alias_.at(snap.obj_id) : snap.obj_id;
            RelinkMotionSnapshot out;
            for (size_t i = 0; i < out.state.size() && i < snap.state.size(); ++i) {
                out.state[i] = snap.state[i];
            }
            for (size_t i = 0; i < out.covariance.size() && i < snap.covariance.size(); ++i) {
                out.covariance[i] = snap.covariance[i];
            }
            out.frame = frame_id;
            motion_[canonical] = out;
        }
    }

    std::vector<int> motion_candidate_ids(int frame_id = -1) const {
        if (mahalanobis_threshold_ <= 0.0f && !kalman_gate_) {
            return {};
        }
        // Kalman gate needs the freshest snapshot of still-active lost tracks so it
        // can capture a clean "farewell" state before the tracker deactivates them;
        // include age>=0 candidates (the tracker only returns the active subset).
        const int min_age = kalman_gate_ ? 0 : min_lost_frames_;
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
                if (age < min_age || age > ttl_) {
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

    void inject_references_batch(
        const std::vector<int>& canonical_ids,
        py::object embeddings_tensor
    ) {
        if (canonical_ids.empty()) return;
        py::array_t<float, py::array::c_style | py::array::forcecast> arr(
            embeddings_tensor.attr("detach")().attr("float")().attr("cpu")().attr("numpy")()
        );
        if (arr.ndim() != 2 || static_cast<int>(arr.shape(0)) != static_cast<int>(canonical_ids.size()))
            throw std::invalid_argument("inject_references_batch: embeddings shape must be [N, D] with N == len(canonical_ids)");
        const int d = static_cast<int>(arr.shape(1));
        for (int i = 0; i < static_cast<int>(canonical_ids.size()); ++i) {
            const float* row = arr.data(i, 0);
            std::vector<float> emb = normalize(std::vector<float>(row, row + d));
            const int cid = canonical_ids[i];
            features_[cid] = emb;
            if (buffer_size_ > 1) {
                auto& buf = buffers_[cid];
                buf.push_back(emb);
                if (static_cast<int>(buf.size()) > buffer_size_) buf.erase(buf.begin());
            }
            if (std::find(feature_order_.begin(), feature_order_.end(), cid) == feature_order_.end())
                feature_order_.push_back(cid);
        }
    }

    int canonical_id(int raw_id) const {
        auto it = alias_.find(raw_id);
        return it == alias_.end() ? raw_id : it->second;
    }

    bool has_feature(int canonical_id) const {
        return features_.find(canonical_id) != features_.end();
    }

    bool is_bidirectional() const {
        return bidirectional_;
    }

    // GPU-accelerated gate table: pre-compute all per-pair gate quantities
    // in one CUDA launch. Called once per frame before the candidate loop.
    // Returns the number of query-candidate pairs computed.
    int build_gate_table(py::sequence raw_ids, py::sequence boxes,
                          int frame_id, int frame_w, int frame_h,
                          uintptr_t tracker_states = 0, uintptr_t tracker_covs = 0,
                          uintptr_t tracker_tids = 0, int tracker_max_objs = 0,
                          py::object query_embs = py::none()) {
        gate_tbl_.clear();
        gate_row_.clear();
        gate_col_.clear();
        gate_n_query_ = gate_n_cand_ = 0;

        const int n_query = static_cast<int>(py::len(raw_ids));
        if (n_query == 0) return 0;

        std::vector<int> cand_ids;
        for (int cid : feature_order_) {
            if (features_.find(cid) != features_.end()) cand_ids.push_back(cid);
        }
        const int n_cand = static_cast<int>(cand_ids.size());
        if (n_cand == 0) return 0;

        gate_n_query_ = n_query;
        gate_n_cand_ = n_cand;

        // Build query data on CPU
        std::vector<float> q_box(n_query * 4, 0.0f);
        std::vector<float> q_foot(n_query * 8, 0.0f);
        std::vector<int>   q_footn(n_query, 0);
        std::vector<float> q_emah(n_query, 1.0f);
        for (int i = 0; i < n_query; ++i) {
            int rid = raw_ids[i].cast<int>();
            gate_row_[rid] = i;
            py::sequence b = boxes[i].cast<py::sequence>();
            q_box[i * 4 + 0] = b[0].cast<float>();
            q_box[i * 4 + 1] = b[1].cast<float>();
            q_box[i * 4 + 2] = b[2].cast<float>();
            q_box[i * 4 + 3] = b[3].cast<float>();
            auto eh = ema_h_.find(rid);
            q_emah[i] = (eh != ema_h_.end()) ? eh->second : 1.0f;
            float cx = (q_box[i * 4 + 0] + q_box[i * 4 + 2]) * 0.5f;
            float cy = (q_box[i * 4 + 1] + q_box[i * 4 + 3]) * 0.5f;
            auto fh = foot_history_.find(rid);
            int k = 0;
            if (fh != foot_history_.end()) {
                for (const auto& p : fh->second) {
                    if (k >= 4) break;
                    q_foot[i * 8 + k * 2] = p.first;
                    q_foot[i * 8 + k * 2 + 1] = p.second;
                    ++k;
                }
            }
            if (k < 4) {
                q_foot[i * 8 + k * 2] = cx;
                q_foot[i * 8 + k * 2 + 1] = cy;
                ++k;
            }
            q_footn[i] = k;
        }

        // Build candidate data on CPU
        std::vector<float> c_last(n_cand * 4, 0.0f);
        std::vector<float> c_mean(n_cand * 6, 0.0f);
        std::vector<float> c_cov(n_cand * 10, 0.0f);
        std::vector<float> c_foot(n_cand * 8, 0.0f);
        std::vector<int>   c_footn(n_cand, 0);
        std::vector<float> c_emah(n_cand, 1.0f);
        std::vector<int>   c_gap(n_cand, 0);
        std::vector<int>   c_delta(n_cand, 0);
        std::vector<int>   c_has(n_cand, 0);
        for (int j = 0; j < n_cand; ++j) {
            int cid = cand_ids[j];
            gate_col_[cid] = j;
            auto lb = last_boxes_.find(cid);
            if (lb != last_boxes_.end()) {
                c_last[j * 4 + 0] = lb->second.x1;
                c_last[j * 4 + 1] = lb->second.y1;
                c_last[j * 4 + 2] = lb->second.x2;
                c_last[j * 4 + 3] = lb->second.y2;
            }
            auto eh = ema_h_.find(cid);
            c_emah[j] = (eh != ema_h_.end()) ? eh->second : 1.0f;
            auto ls = last_seen_.find(cid);
            c_gap[j] = (ls != last_seen_.end()) ? (frame_id - ls->second) : 0;
            auto fh = foot_history_.find(cid);
            int k = 0;
            if (fh != foot_history_.end()) {
                int total = (int)fh->second.size();
                for (int p = std::max(0, total - 4); p < total && k < 4; ++p) {
                    c_foot[j * 8 + k * 2] = fh->second[p].first;
                    c_foot[j * 8 + k * 2 + 1] = fh->second[p].second;
                    ++k;
                }
            }
            c_footn[j] = k;
            auto ms = motion_.find(cid);
            if (ms != motion_.end()) {
                c_has[j] = 1;
                c_delta[j] = (ms->second.frame >= 0) ? (frame_id - ms->second.frame) : 0;
                const auto& st = ms->second.state;
                c_mean[j * 6 + 0] = st[0];
                c_mean[j * 6 + 1] = st[1];
                c_mean[j * 6 + 2] = st[3];
                c_mean[j * 6 + 3] = st[4];
                c_mean[j * 6 + 4] = st[5];
                c_mean[j * 6 + 5] = st[7];
                const auto& P = ms->second.covariance;
                c_cov[j * 10 + 0] = P[0];  c_cov[j * 10 + 1] = P[1];
                c_cov[j * 10 + 2] = P[4];  c_cov[j * 10 + 3] = P[5];
                c_cov[j * 10 + 4] = P[9];  c_cov[j * 10 + 5] = P[12];
                c_cov[j * 10 + 6] = P[13]; c_cov[j * 10 + 7] = P[36];
                c_cov[j * 10 + 8] = P[37]; c_cov[j * 10 + 9] = P[45];
            }
        }

        // GPU allocation + launch
        float *d_table = nullptr, *d_qbox = nullptr, *d_qfoot = nullptr, *d_qemah = nullptr;
        float *d_clast = nullptr, *d_cmean = nullptr, *d_ccov = nullptr, *d_cfoot = nullptr;
        float *d_cemah = nullptr;
        int *d_qfootn = nullptr, *d_cfootn = nullptr, *d_cgap = nullptr, *d_cdelta = nullptr, *d_chas = nullptr;

        const size_t tsz = static_cast<size_t>(n_query) * n_cand * 6 * sizeof(float);
        cudaMalloc(&d_table, tsz);
        cudaMalloc(&d_qbox,  n_query * 4 * sizeof(float));
        cudaMalloc(&d_qfoot, n_query * 8 * sizeof(float));
        cudaMalloc(&d_qemah, n_query * sizeof(float));
        cudaMalloc(&d_qfootn, n_query * sizeof(int));
        cudaMalloc(&d_clast,  n_cand * 4 * sizeof(float));
        cudaMalloc(&d_cmean,  n_cand * 6 * sizeof(float));
        cudaMalloc(&d_ccov,   n_cand * 10 * sizeof(float));
        cudaMalloc(&d_cfoot,  n_cand * 8 * sizeof(float));
        cudaMalloc(&d_cemah,  n_cand * sizeof(float));
        cudaMalloc(&d_cfootn, n_cand * sizeof(int));
        cudaMalloc(&d_cgap,   n_cand * sizeof(int));
        cudaMalloc(&d_cdelta, n_cand * sizeof(int));
        cudaMalloc(&d_chas,   n_cand * sizeof(int));

        // H2D copies for query data and non-state candidate data
        cudaMemcpy(d_qbox,  q_box.data(),  n_query * 4 * sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_qfoot, q_foot.data(), n_query * 8 * sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_qemah, q_emah.data(), n_query * sizeof(float),      cudaMemcpyHostToDevice);
        cudaMemcpy(d_qfootn,q_footn.data(),n_query * sizeof(int),        cudaMemcpyHostToDevice);
        cudaMemcpy(d_clast, c_last.data(), n_cand * 4 * sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cfoot, c_foot.data(), n_cand * 8 * sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cemah, c_emah.data(), n_cand * sizeof(float),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_cfootn,c_footn.data(),n_cand * sizeof(int),         cudaMemcpyHostToDevice);
        cudaMemcpy(d_cgap,  c_gap.data(),  n_cand * sizeof(int),         cudaMemcpyHostToDevice);

        // Baseline candidate Kalman state from the CPU-sourced motion snapshot.
        // This must happen unconditionally so candidates that are NOT present in
        // the tracker GPU buffer (slot=-1) still have valid mean/cov; the D2D
        // gather below only overwrites the in-buffer slots and the gather kernel
        // early-returns for slot<0, leaving d_cmean/d_ccov untouched there.
        cudaMemcpy(d_cmean, c_mean.data(), n_cand * 6 * sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_ccov,  c_cov.data(),  n_cand * 10 * sizeof(float),  cudaMemcpyHostToDevice);

        // Candidate Kalman state: use D2D gather from tracker GPU buffers when
        // available (avoid H2D copy), otherwise fall back to CPU-sourced H2D.
        bool used_tracker_state = false;
        if (tracker_states && tracker_covs && tracker_tids && tracker_max_objs > 0) {
            std::vector<int> h_tids(tracker_max_objs);
            cudaMemcpy(h_tids.data(), reinterpret_cast<const int*>(tracker_tids),
                       tracker_max_objs * sizeof(int), cudaMemcpyDeviceToHost);
            std::unordered_map<int, int> tid_to_slot;
            for (int s = 0; s < tracker_max_objs; ++s)
                if (h_tids[s] != 0) tid_to_slot[h_tids[s]] = s;

            std::vector<int> slots(n_cand, -1);
            std::vector<int> h_delta(n_cand, 0);
            std::vector<int> h_has(n_cand, 0);
            int n_in_buffer = 0;
            for (int j = 0; j < n_cand; ++j) {
                auto it = tid_to_slot.find(cand_ids[j]);
                if (it != tid_to_slot.end()) {
                    slots[j] = it->second;
                    h_delta[j] = 0;
                    h_has[j] = 1;
                    ++n_in_buffer;
                } else {
                    h_delta[j] = c_delta[j];
                    h_has[j] = c_has[j];
                }
            }
            if (n_in_buffer > 0) {
                int* d_slots;
                cudaMalloc(&d_slots, n_cand * sizeof(int));
                cudaMemcpy(d_slots, slots.data(), n_cand * sizeof(int), cudaMemcpyHostToDevice);
                saccade::relink_gate::gather_tracker_state(
                    reinterpret_cast<const float*>(tracker_states),
                    reinterpret_cast<const float*>(tracker_covs),
                    d_slots, n_cand, d_cmean, d_ccov, nullptr);
                cudaDeviceSynchronize();
                cudaFree(d_slots);
                used_tracker_state = true;
            }
            // Update delta/has from the mixed CPU-tracker arrays
            cudaMemcpy(d_cdelta, h_delta.data(), n_cand * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_chas,   h_has.data(),   n_cand * sizeof(int), cudaMemcpyHostToDevice);
        }
        if (!used_tracker_state) {
            // No tracker buffer in play: delta/has come straight from the CPU
            // snapshot (mean/cov already uploaded above).
            cudaMemcpy(d_cdelta,c_delta.data(),n_cand * sizeof(int),         cudaMemcpyHostToDevice);
            cudaMemcpy(d_chas,  c_has.data(),  n_cand * sizeof(int),         cudaMemcpyHostToDevice);
        }

        const int dims = bidirectional_ ? 2 : 4;
        saccade::relink_gate::GateParams p{n_query, n_cand, frame_w, frame_h, dims,
            kalman_fps_, kalman_person_height_m_, kalman_max_speed_mps_,
            kalman_accel_long_, kalman_accel_lat_, kalman_dir_min_cos_, kalman_dir_min_speed_};
        saccade::relink_gate::launch(p, d_qbox, d_qfoot, d_qfootn, d_qemah,
            d_clast, d_cmean, d_ccov, d_cfoot, d_cfootn, d_cemah,
            d_cgap, d_cdelta, d_chas, d_table, nullptr);

        cudaDeviceSynchronize();

        // GPU batch dot + scoring (runs when query embeddings are available)
        float* d_sim = nullptr;
        sim_tbl_.clear();
        scoring_ids_.clear(); scoring_scores_.clear(); scoring_second_.clear();
        if (!query_embs.is_none()) {
            py::array_t<float, py::array::c_style | py::array::forcecast> qarr(
                query_embs.attr("detach")().attr("float")().attr("cpu")().attr("numpy")());
            int dim = qarr.ndim() >= 2 ? (int)qarr.shape(1) : (int)qarr.shape(0);
            int n_q = qarr.ndim() >= 2 ? (int)qarr.shape(0) : 1;
            if (n_q == n_query && dim > 0) {
                int feat_dim = 0;
                for (int cid : cand_ids) {
                    auto it = features_.find(cid);
                    if (it != features_.end() && (int)it->second.size() > feat_dim)
                        feat_dim = (int)it->second.size();
                }
                if (feat_dim > 0 && feat_dim == dim) {
                    std::vector<float> cand_feats(n_cand * dim, 0.0f);
                    for (int j = 0; j < n_cand; ++j) {
                        auto it = features_.find(cand_ids[j]);
                        if (it != features_.end()) {
                            // feat_dim is the MAX feature size; a candidate
                            // registered with a shorter placeholder vector must
                            // not be read past its own length (heap OOB). The
                            // remaining entries stay 0 -> near-zero similarity.
                            const int csz = std::min(dim, (int)it->second.size());
                            for (int k = 0; k < csz; ++k)
                                cand_feats[j * dim + k] = it->second[k];
                        }
                    }
                    float *d_qembs = nullptr, *d_cfeats = nullptr;
                    cudaMalloc(&d_qembs, n_query * dim * sizeof(float));
                    cudaMalloc(&d_cfeats, n_cand * dim * sizeof(float));
                    cudaMalloc(&d_sim,   n_query * n_cand * sizeof(float));
                    cudaMemcpy(d_qembs, qarr.data(), n_query * dim * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_cfeats, cand_feats.data(), n_cand * dim * sizeof(float), cudaMemcpyHostToDevice);
                    saccade::relink_gate::batch_dot(d_qembs, d_cfeats, n_query, n_cand, dim, d_sim, nullptr);
                    cudaDeviceSynchronize();
                    sim_tbl_.resize(static_cast<size_t>(n_query) * n_cand);
                    cudaMemcpy(sim_tbl_.data(), d_sim, sim_tbl_.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaFree(d_qembs); cudaFree(d_cfeats);

                    // Scoring kernel: uses gate table + sim table on GPU
                    std::vector<int> cand_ages_vec(n_cand);
                    std::vector<float> cand_maha_vec(n_cand, 0.0f);
                    for (int j = 0; j < n_cand; ++j) {
                        auto ls = last_seen_.find(cand_ids[j]);
                        cand_ages_vec[j] = (ls != last_seen_.end()) ? (frame_id - ls->second) : 0;
                        auto ms = motion_.find(cand_ids[j]);
                        if (ms != motion_.end()) {
                            auto lb = last_boxes_.find(cand_ids[j]);
                            if (lb != last_boxes_.end())
                                cand_maha_vec[j] = mahalanobis(lb->second, ms->second);
                        }
                    }
                    int *d_cids = nullptr, *d_cages = nullptr, *d_best = nullptr;
                    float *d_cmaha = nullptr, *d_bscore = nullptr, *d_sscore = nullptr;
                    cudaMalloc(&d_cids, n_cand * sizeof(int));
                    cudaMalloc(&d_cages, n_cand * sizeof(int));
                    cudaMalloc(&d_cmaha, n_cand * sizeof(float));
                    cudaMalloc(&d_best, n_query * sizeof(int));
                    cudaMalloc(&d_bscore, n_query * sizeof(float));
                    cudaMalloc(&d_sscore, n_query * sizeof(float));
                    cudaMemcpy(d_cids, cand_ids.data(), n_cand * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_cages, cand_ages_vec.data(), n_cand * sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_cmaha, cand_maha_vec.data(), n_cand * sizeof(float), cudaMemcpyHostToDevice);

                    saccade::relink_gate::ScoringParams sp;
                    sp.n_cand = n_cand; sp.ttl = ttl_;
                    sp.sim_threshold = sim_threshold_;
                    sp.w_sim_base = w_sim_base_; sp.w_iou_base = w_iou_base_; sp.w_maha_base = w_maha_base_;
                    sp.shift_ambiguity = shift_ambiguity_; sp.shift_lost_age = shift_lost_age_;
                    sp.mahalanobis_threshold = mahalanobis_threshold_;
                    sp.kalman_penalty_weight = kalman_penalty_weight_;
                    sp.reciprocal_margin = reciprocal_margin_;
                    sp.dynamic_margin_crowd = dynamic_margin_crowd_; sp.dynamic_margin_age = dynamic_margin_age_;
                    sp.iou_weight = iou_weight_; sp.mahalanobis_weight = mahalanobis_weight_;

                    saccade::relink_gate::launch_relink_scoring(
                        d_table, d_sim, d_cids, d_cages, d_cmaha,
                        n_query, sp, d_best, d_bscore, d_sscore, nullptr);
                    cudaDeviceSynchronize();

                    scoring_ids_.resize(n_query);
                    scoring_scores_.resize(n_query);
                    scoring_second_.resize(n_query);
                    cudaMemcpy(scoring_ids_.data(), d_best, n_query * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(scoring_scores_.data(), d_bscore, n_query * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(scoring_second_.data(), d_sscore, n_query * sizeof(float), cudaMemcpyDeviceToHost);

                    cudaFree(d_cids); cudaFree(d_cages); cudaFree(d_cmaha);
                    cudaFree(d_best); cudaFree(d_bscore); cudaFree(d_sscore);
                    cudaFree(d_sim); d_sim = nullptr;
                }
            }
        }

        gate_tbl_.resize(static_cast<size_t>(n_query) * n_cand * 6);
        cudaMemcpy(gate_tbl_.data(), d_table, tsz, cudaMemcpyDeviceToHost);
        if (d_sim) cudaFree(d_sim);

        cudaFree(d_table); cudaFree(d_qbox); cudaFree(d_qfoot); cudaFree(d_qemah);
        cudaFree(d_qfootn); cudaFree(d_clast); cudaFree(d_cmean); cudaFree(d_ccov);
        cudaFree(d_cfoot); cudaFree(d_cemah); cudaFree(d_cfootn);
        cudaFree(d_cgap); cudaFree(d_cdelta); cudaFree(d_chas);

        return n_query * n_cand;
    }

    void clear_gate_table() {
        gate_tbl_.clear();
        sim_tbl_.clear();
        scoring_ids_.clear();
        scoring_scores_.clear();
        scoring_second_.clear();
        gate_row_.clear();
        gate_col_.clear();
        gate_n_query_ = gate_n_cand_ = 0;
    }

    const float* gate_lookup(int raw_id, int cid) const {
        if (gate_tbl_.empty()) return nullptr;
        auto ri = gate_row_.find(raw_id);
        auto ci = gate_col_.find(cid);
        if (ri == gate_row_.end() || ci == gate_col_.end()) return nullptr;
        int idx = (ri->second * gate_n_cand_ + ci->second) * 6;
        return &gate_tbl_[idx];
    }

    float sim_lookup(int raw_id, int cid) const {
        if (sim_tbl_.empty()) return -2.0f;
        auto ri = gate_row_.find(raw_id);
        auto ci = gate_col_.find(cid);
        if (ri == gate_row_.end() || ci == gate_col_.end()) return -2.0f;
        int idx = ri->second * gate_n_cand_ + ci->second;
        return sim_tbl_[idx];
    }

    // Verify a GPU-scored winner against the SAME hard gates the inline
    // candidate loop enforces. The relink_scoring_kernel only filters on the
    // lenient sim_threshold and folds Kalman distance into a soft penalty, so
    // it can return a candidate the CPU path would hard-reject (age window,
    // physical speed, spatial reach, Kalman chi-square, direction-behind,
    // bidirectional bridge, Mahalanobis, buffer consistency, or strict
    // similarity for non-clean boxes). Returns true only when the candidate
    // would also be admitted by the inline loop; on false the caller leaves
    // best_* pristine and falls through to that loop. Because the GPU winner is
    // a global argmax, a passing winner equals the inline-loop pick, so the
    // fast-path stays consistent with the CPU path.
    bool gpu_winner_passes_gates(int raw_id, int cid, const RelinkBox& box,
                                 int frame_id, float current_sim_thresh,
                                 float best_sim_raw) {
        const auto ls = last_seen_.find(cid);
        if (ls == last_seen_.end()) return false;

        const float* gate = gate_lookup(raw_id, cid);
        if (!gate) return false;  // no precomputed gate row -> defer to inline loop

        RelinkCandidateGates gates;
        if (!evaluate_candidate_gates(
                raw_id, cid, box, frame_id, 0, 0, gate, gates,
                false, false)) {
            return false;
        }
        if (best_sim_raw < current_sim_thresh) return false;  // strict/lenient sim
        return true;
    }

    py::dict get_deferred_alias() const {
        py::dict out;
        for (const auto& [k, v] : deferred_alias_) {
            out[py::int_(k)] = py::int_(v);
        }
        return out;
    }

    bool mark_pending_claim(int raw_id, int frame_id) {
        if (!delayed_claim_) return false;
        auto& pending = pending_claims_[raw_id];
        if (pending.hits == 0) {
            pending.first = frame_id;
            pending.last = frame_id;
            pending.hits = 1;
        } else if (pending.last != frame_id) {
            pending.last = frame_id;
            pending.hits += 1;
        }
        const bool ready = pending.hits >= claim_warmup_frames_;
        if (ready) {
            stats_.delayed_claim_ready += 1;
        } else {
            stats_.delayed_claim_pending += 1;
        }
        return !ready;
    }

    bool pending_ready(int raw_id) const {
        auto it = pending_claims_.find(raw_id);
        return it != pending_claims_.end() && it->second.hits >= claim_warmup_frames_;
    }

    void remember_pending_head(int raw_id, const std::vector<float>& emb, bool is_clean) {
        if (!delayed_claim_ || !is_clean || emb.empty()) return;
        if (!pending_heads_.count(raw_id)) {
            pending_heads_[raw_id] = {};
        }
        pending_heads_[raw_id].push_back(emb);
    }

    int cheb_gr_claim_best(
        int raw_id,
        const std::vector<std::vector<float>>& candidates,
        const std::vector<int>& candidate_ids,
        int emb_dim
    ) {
        auto head_it = pending_heads_.find(raw_id);
        if (head_it == pending_heads_.end()) return -1;
        const auto& head = head_it->second;
        int H = static_cast<int>(head.size());
        if (H < cheb_gr_min_head_) return -1;

        int N = H + static_cast<int>(candidates.size());
        std::vector<float> feats_data(N * emb_dim);
        int pos = 0;
        for (const auto& h : head) {
            std::copy(h.begin(), h.end(), feats_data.begin() + pos * emb_dim);
            pos++;
        }
        std::unordered_map<int, std::pair<int,int>> spans;
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            std::copy(candidates[ci].begin(), candidates[ci].end(),
                      feats_data.begin() + pos * emb_dim);
            pos++;
            spans[candidate_ids[ci]] = {pos - 1, pos};
        }
        // feats_data is row-major (one sample per contiguous emb_dim block);
        // Eigen::MatrixXf defaults to column-major, so map row-major first.
        Eigen::MatrixXf feats =
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>>(
                feats_data.data(), N, emb_dim);
        auto sdist = saccade::cheb_gr_kreciprocal_self(
            feats, cheb_gr_lambda_, cheb_gr_k2_, cheb_gr_max_fwd_,
            cheb_gr_fuse_lambda_);

        float best_cost = cheb_gr_max_cost_ + 1.0f;
        float second_cost = std::numeric_limits<float>::infinity();
        int best_id = -1;

        auto head_rows = sdist.topRows(H);
        for (size_t ci = 0; ci < candidate_ids.size(); ++ci) {
            int cid = candidate_ids[ci];
            auto it = spans.find(cid);
            if (it == spans.end()) continue;
            int lo = it->second.first, hi = it->second.second;
            std::vector<float> block;
            for (int r = 0; r < H; ++r)
                for (int c = lo; c < hi; ++c)
                    block.push_back(head_rows(r, c));
            int k = std::max(1, static_cast<int>(
                std::round(cheb_gr_pool_frac_ * static_cast<float>(block.size()))));
            std::nth_element(block.begin(), block.begin() + k, block.end());
            float cost = 0.0f;
            for (int i = 0; i < k; ++i) cost += block[i];
            cost /= static_cast<float>(k);

            if (cost < best_cost) {
                second_cost = best_cost;
                best_cost = cost;
                best_id = cid;
            } else if (cost < second_cost) {
                second_cost = cost;
            }
        }

        if (best_cost > cheb_gr_max_cost_) return -1;
        float margin = second_cost - best_cost;
        if (cheb_gr_margin_ > 0.0f && margin < cheb_gr_margin_) return -1;
        return best_id;
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
        std::vector<float> emb;
        bool has_emb = !embedding.is_none();
        if (has_emb) {
            emb = normalize(extract_embedding(embedding));
        }

        const RelinkBox box = parse_box(box_obj);
        const bool is_clean = is_clean_observation(box, score, frame_w, frame_h);
        const float current_sim_thresh = is_clean ? sim_threshold_ : strict_sim_threshold_;

        if (delayed_claim_) {
            remember_pending_head(raw_id, emb, is_clean);
        }

        if (delayed_claim_ && pending_claims_.count(raw_id) && canonical_id(raw_id) == raw_id) {
            mark_pending_claim(raw_id, frame_id);
        }
        const bool claim_ready = pending_ready(raw_id);
        const bool first_claim = !alias_.count(raw_id);
        const bool delayed_self_claim = delayed_claim_ && claim_ready && canonical_id(raw_id) == raw_id;

        if (first_claim && mark_pending_claim(raw_id, frame_id)) {
            alias_[raw_id] = raw_id;
        } else if (first_claim || delayed_self_claim) {
            stats_.attempts += 1;
            int best_id = -1;
            float best_sim = current_sim_thresh;
            float second_best_sim = current_sim_thresh - 1.0f;
            float best_iou = 0.0f;
            float best_center = 0.0f;
            float best_maha = 0.0f;

            for (int cid : feature_order_) {
                if (delayed_claim_ && cid == raw_id) {
                    continue;
                }
                const auto feature_it = features_.find(cid);
                if (feature_it == features_.end()) {
                    continue;
                }
                py::int_ py_cid(cid);
                if (PySet_Contains(assigned.ptr(), py_cid.ptr()) == 1) {
                    stats_.reject_assigned += 1;
                    continue;
                }
                RelinkCandidateGates gates;
                if (!evaluate_candidate_gates(raw_id, cid, box, frame_id, frame_w, frame_h, nullptr, gates)) {
                    continue;
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
                    best_iou = gates.iou;
                    best_center = gates.center_norm;
                    best_maha = gates.maha;
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
                record_relink_accept(raw_id, best_id, best_sim, best_iou, best_center, best_maha);
            } else {
                record_new_identity(raw_id, claim_ready);
            }
        }

        int canonical = alias_.at(raw_id);
        commit_reference_state(canonical, box, emb, has_emb, is_clean, frame_id);

        if (ho_enabled_ && has_emb && is_clean) {
            feed_newborn_head(raw_id, emb, frame_id);
        }
        expire_dead_archive(frame_id);

        // Per-frame uniqueness guard
        py::int_ py_canonical(canonical);
        if (PySet_Contains(assigned.ptr(), py_canonical.ptr()) == 1) {
            canonical = split_on_collision(raw_id);
        }
        assigned.add(py::int_(canonical));
        return canonical;
    }

    py::list resolve_many(
        py::iterable candidates,
        int frame_id,
        int frame_w,
        int frame_h
    ) {
        struct C { int raw_id; py::object emb; py::sequence box; float score; };
        std::vector<C> cands;
        for (py::handle item : candidates) {
            py::tuple t = py::reinterpret_borrow<py::tuple>(item);
            if (t.size() != 4)
                throw std::runtime_error("resolve_many expects (raw_id, embedding, box, score) tuples");
            cands.push_back({t[0].cast<int>(), py::reinterpret_borrow<py::object>(t[1]),
                             t[2].cast<py::sequence>(), t[3].cast<float>()});
        }
        size_t n = cands.size();
        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;
        if (bidirectional_ && n > 0) {
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return cands[a].score > cands[b].score;
            });
        }
        py::set assigned;
        std::vector<int> results(n);
        for (size_t i : order) {
            results[i] = resolve(cands[i].raw_id, cands[i].emb, cands[i].box,
                                 cands[i].score, frame_id, frame_w, frame_h, assigned);
        }
        py::list out;
        for (size_t i = 0; i < n; ++i) out.append(results[i]);
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
        const size_t n = static_cast<size_t>(py::len(raw_ids));
        if (py::len(embeddings) != n || py::len(boxes) != n || py::len(scores) != n) {
            throw std::runtime_error("resolve_many_packed expects equally sized raw_ids/embeddings/boxes/scores");
        }
        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;
        if (bidirectional_ && n > 0) {
            std::vector<float> s(n);
            for (size_t i = 0; i < n; ++i) s[i] = scores[i].cast<float>();
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return s[a] > s[b];
            });
        }
        py::set assigned;
        std::vector<int> results(n);
        for (size_t i : order) {
            results[i] = resolve(
                raw_ids[i].cast<int>(),
                py::reinterpret_borrow<py::object>(embeddings[i]),
                boxes[i].cast<py::sequence>(),
                scores[i].cast<float>(),
                frame_id,
                frame_w,
                frame_h,
                assigned
            );
        }
        py::list out;
        for (size_t i = 0; i < n; ++i) out.append(results[i]);
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
        out["reject_quality"] = stats_.reject_quality;
        out["reject_kalman"] = stats_.reject_kalman;
        out["reject_direction"] = stats_.reject_direction;
        out["reject_speed"] = stats_.reject_speed;
        out["delayed_claim_pending"] = stats_.delayed_claim_pending;
        out["delayed_claim_ready"] = stats_.delayed_claim_ready;
        out["delayed_claim_accepted"] = stats_.delayed_claim_accepted;
        out["new_ids"] = stats_.new_ids;
        out["reject_backward"] = stats_.reject_backward;
        out["accept_bidir"] = stats_.accept_bidir;
        out["relink_split_collision"] = stats_.relink_split_collision;
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
            " reject_margin=" + std::to_string(stats_.reject_margin) +
            " reject_quality=" + std::to_string(stats_.reject_quality) +
            " reject_kalman=" + std::to_string(stats_.reject_kalman) +
            " reject_direction=" + std::to_string(stats_.reject_direction) +
            " reject_speed=" + std::to_string(stats_.reject_speed) +
            " reject_backward=" + std::to_string(stats_.reject_backward) +
            " accept_bidir=" + std::to_string(stats_.accept_bidir) +
            " split_collision=" + std::to_string(stats_.relink_split_collision)
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
        if (delayed_claim_) {
            py::print(
                "  delayed_claim pending_checks=" + std::to_string(stats_.delayed_claim_pending) +
                " ready_checks=" + std::to_string(stats_.delayed_claim_ready) +
                " accepted=" + std::to_string(stats_.delayed_claim_accepted) +
                " deferred_aliases=" + std::to_string(deferred_alias_.size())
            );
        }
    }

    // Closed-form linear regression velocity from 4 equally-spaced foot positions.
    // For frames t ∈ {0,1,2,3} with centres (x_i, y_i):
    //   v_x = (3·x₃ + x₂ − x₁ − 3·x₀) / 10
    //   v_y = (3·y₃ + y₂ − y₁ − 3·y₀) / 10
    static std::pair<float, float> regress_velocity_4(
        const std::vector<std::pair<float, float>>& positions) {
        if (positions.size() < 4) return {0.0f, 0.0f};
        float x0 = positions[0].first, y0 = positions[0].second;
        float x1 = positions[1].first, y1 = positions[1].second;
        float x2 = positions[2].first, y2 = positions[2].second;
        float x3 = positions[3].first, y3 = positions[3].second;
        return {(3.0f * x3 + x2 - x1 - 3.0f * x0) / 10.0f,
                (3.0f * y3 + y2 - y1 - 3.0f * y0) / 10.0f};
    }

    // Scale gate (disabled when bridge_h_hi<=0): lost/cand EMA-height ratio must
    // stay inside [h_lo, h_hi]; large size jumps across the gap are bogus bridges.
    bool bridge_scale_gate_ok(int lost_id, int cand_id) const {
        if (bridge_h_hi_ <= 0.0f) return true;
        auto hl = ema_h_.find(lost_id);
        auto hc = ema_h_.find(cand_id);
        if (hl == ema_h_.end() || hc == ema_h_.end()) return true;
        float hr = std::max(hl->second, 1e-3f) / std::max(hc->second, 1e-3f);
        return hr >= bridge_h_lo_ && hr <= bridge_h_hi_;
    }

    // Bidirectional midpoint bridge distance. Propagates both lost track and
    // candidate through half the gap using regressed velocities, then measures
    // Euclidean distance between the two midpoints, normalized by average height.
    float midpoint_bridge_dist(
        int lost_id, int cand_id, int gap, float cand_cx, float cand_cy) const {
        auto lost_it = foot_history_.find(lost_id);
        auto cand_it = foot_history_.find(cand_id);
        if (lost_it == foot_history_.end()) return 1e9f;
        const auto& lost_hist = lost_it->second;
        if (lost_hist.empty()) return 1e9f;
        std::vector<std::pair<float, float>> cand_hist;
        if (cand_it != foot_history_.end()) cand_hist = cand_it->second;
        cand_hist.push_back({cand_cx, cand_cy});
        auto [vx_l, vy_l] = regress_velocity_4(
            std::vector<std::pair<float, float>>(
                lost_hist.end() - std::min((int)lost_hist.size(), 4), lost_hist.end()));
        auto [vx_c, vy_c] = regress_velocity_4(
            std::vector<std::pair<float, float>>(
                cand_hist.begin(), cand_hist.begin() + std::min((int)cand_hist.size(), 4)));
        float h_lost = 1.0f;
        auto hl = ema_h_.find(lost_id);
        if (hl != ema_h_.end()) h_lost = hl->second;
        float h_ref = std::max(h_lost, 1.0f);
        float lx = lost_hist.back().first, ly = lost_hist.back().second;
        float cxf = cand_hist[0].first, cyf = cand_hist[0].second;
        // Speed-weighted foot-bridge score: symmetric full extrapolation 0.5*(fwd+bwd)
        // blended with spatial proximity dist_h, velocity weighted by exit speed.
        // See docs/modules/semantic/research/offline_relink_candidate_analysis.md §6c-d.
        float g = static_cast<float>(gap);
        float fwd_r = std::hypot(lx + vx_l * g - cxf, ly + vy_l * g - cyf) / h_ref;
        float bwd_r = std::hypot(cxf - vx_c * g - lx, cyf - vy_c * g - ly) / h_ref;
        float dist_h = std::hypot(lx - cxf, ly - cyf) / h_ref;
        float s_lost = std::hypot(vx_l, vy_l) / h_ref;
        float w = std::sqrt(std::clamp(s_lost / 0.12f, 0.0f, 1.0f));
        return w * 0.5f * (fwd_r + bwd_r) + (1.0f - w) * dist_h;
    }

    // Squared Mahalanobis distance for 2-DoF (center-only), used by the
    // bidirectional gate where only the positional sub-state is gated.
    static float kalman_gate_dist_2d(
        const RelinkBox& box, const RelinkMotionSnapshot& snap, int delta,
        float person_height_m, float accel_long, float accel_lat, float fps) {
        float x[8];
        float P[64];
        for (int i = 0; i < 8; ++i) x[i] = snap.state[i];
        for (int i = 0; i < 64; ++i) P[i] = snap.covariance[i];
        const int steps = std::max(0, delta);
        if (steps > 0) {
            if (person_height_m > 0.0f)
                predict_phys_delta(x, P, steps, person_height_m, accel_long, accel_lat, fps);
            else
                predict_delta(x, P, steps);
        }
        float h = std::max(x[3], 1e-6f);
        float pos_sq = (h / 20.0f) * (h / 20.0f);
        Eigen::Matrix2f S;
        S(0, 0) = P[0] + pos_sq;   S(0, 1) = P[1];
        S(1, 0) = P[8];            S(1, 1) = P[9] + pos_sq;
        float zx = (box.x1 + box.x2) * 0.5f;
        float zy = (box.y1 + box.y2) * 0.5f;
        Eigen::Vector2f y(zx - x[0], zy - x[1]);
        Eigen::LLT<Eigen::Matrix2f> llt(S);
        if (llt.info() != Eigen::Success) return 1e9f;
        Eigen::Vector2f w = llt.matrixL().solve(y);
        return w.squaredNorm();
    }

    // Per-frame uniqueness guard: split a colliding raw_id onto a fresh surrogate.
    int split_on_collision(int raw_id) {
        split_counter_ += 1;
        int surrogate = 1000000 + split_counter_;
        alias_[raw_id] = surrogate;
        stats_.relink_split_collision += 1;
        return surrogate;
    }

    // Internal C++ resolve — takes pre-normalized embedding and C++ box.
    // Called by IdentityResolverCpp to avoid re-parsing Python objects between stages.
    int resolve_cpp(
        int raw_id,
        const std::vector<float>& emb,
        bool has_emb,
        const RelinkBox& box,
        float score,
        int frame_id,
        int frame_w,
        int frame_h,
        std::unordered_set<int>& assigned
    ) {
        const bool is_clean = is_clean_observation(box, score, frame_w, frame_h);
        const float current_sim_thresh = is_clean ? sim_threshold_ : strict_sim_threshold_;

        if (delayed_claim_ && pending_claims_.count(raw_id) && canonical_id(raw_id) == raw_id) {
            mark_pending_claim(raw_id, frame_id);
        }
        const bool claim_ready = pending_ready(raw_id);
        const bool first_claim = !alias_.count(raw_id);
        const bool delayed_self_claim = delayed_claim_ && claim_ready && canonical_id(raw_id) == raw_id;

        if (first_claim && mark_pending_claim(raw_id, frame_id)) {
            alias_[raw_id] = raw_id;
        } else if (first_claim || delayed_self_claim) {
            stats_.attempts += 1;

            // GPU scoring fast-path: use pre-computed best from the scoring
            // kernel (runs once per frame after build_gate_table).
            int best_id = -1;
            float best_joint = -1.0f;
            float best_sim_raw = 0.0f;
            float second_best_joint = -2.0f;
            float best_iou = 0.0f, best_center = 0.0f, best_maha = 0.0f;
            bool gpu_scored = false;

            if (!scoring_ids_.empty()) {
                auto ri = gate_row_.find(raw_id);
                if (ri != gate_row_.end() && ri->second < (int)scoring_ids_.size()) {
                    int q = ri->second;
                    int gpu_best = scoring_ids_[q];
                    // Check assigned — the GPU doesn't handle per-frame uniqueness
                    if (gpu_best >= 0 && !assigned.count(gpu_best)) {
                        float gpu_joint = scoring_scores_[q];
                        bool accept = true;
                        // Apply dynamic_margin_age (GPU only applies static+crowd)
                        if (dynamic_margin_age_ > 0.0f) {
                            auto ls = last_seen_.find(gpu_best);
                            int lost_frames = (ls != last_seen_.end()) ? (frame_id - ls->second) : 0;
                            float age_factor = std::min(1.0f, static_cast<float>(lost_frames) / std::max(1, ttl_));
                            float extra_margin = dynamic_margin_age_ * age_factor;
                            float second = scoring_second_[q];
                            if (gpu_joint - second < extra_margin) accept = false;
                        }
                        if (accept) {
                            float sim_raw = sim_lookup(raw_id, gpu_best);
                            if (sim_raw < -1.5f) sim_raw = 0.0f;
                            // The scoring kernel only filtered on the lenient
                            // sim_threshold; verify the winner against the full
                            // hard-gate set (age/speed/spatial/Kalman-chi2/
                            // direction/bridge/Mahalanobis/consistency/strict-sim)
                            // before trusting it. On failure leave best_* pristine
                            // and fall through to the authoritative inline loop.
                            if (gpu_winner_passes_gates(raw_id, gpu_best, box, frame_id,
                                                        current_sim_thresh, sim_raw)) {
                                best_id = gpu_best;
                                best_joint = gpu_joint;
                                best_sim_raw = sim_raw;
                                gpu_scored = true;
                            }
                        }
                    }
                }
            }

            if (gpu_scored) {
                // Use fast-path: skip the O(n_cand) loop
                record_relink_accept(raw_id, best_id, best_sim_raw, best_iou, best_center, best_maha);
            } else {
            // Fall-through to inline candidate loop

            std::vector<RelinkCandidateGates> candidates_to_score;

            for (int cid : feature_order_) {
                if (delayed_claim_ && cid == raw_id) {
                    continue;
                }
                const auto feature_it = features_.find(cid);
                if (feature_it == features_.end()) continue;
                if (assigned.count(cid)) { stats_.reject_assigned += 1; continue; }

                const float* gate = gate_lookup(raw_id, cid);
                RelinkCandidateGates gates;
                if (!evaluate_candidate_gates(raw_id, cid, box, frame_id, frame_w, frame_h, gate, gates)) {
                    continue;
                }
                candidates_to_score.push_back(gates);
            }

            int n_gate_passed = static_cast<int>(candidates_to_score.size());

            bool cheb_gr_scored = false;
            int cheb_gr_dim = 0;
            for (const auto& [cid, f] : features_) { cheb_gr_dim = static_cast<int>(f.size()); break; }

            if (cheb_gr_claim_ && delayed_claim_ && claim_ready && has_emb
                && cheb_gr_dim > 0 && !candidates_to_score.empty()) {
                int H = 0;
                auto head_it = pending_heads_.find(raw_id);
                if (head_it != pending_heads_.end()) H = static_cast<int>(head_it->second.size());
                if (H >= cheb_gr_min_head_) {
                    std::vector<std::vector<float>> cand_embs;
                    std::vector<int> cand_ids;
                    for (const auto& gates : candidates_to_score) {
                        int cid = gates.cid;
                        auto fit = features_.find(cid);
                        if (fit == features_.end()) continue;
                        if (cheb_gr_min_sim_ > 0.0f) {
                            float sim_val = sim_lookup(raw_id, cid);
                            if (sim_val <= -2.0f) {
                                std::vector<float> ref = buffer_size_ > 1 ? buffer_mean(cid) : fit->second;
                                if (ref.empty()) ref = fit->second;
                                sim_val = dot(emb, ref);
                            }
                            if (sim_val < cheb_gr_min_sim_) continue;
                        }
                        std::vector<std::vector<float>> bank;
                        if (buffer_size_ > 1) {
                            auto bit = buffers_.find(cid);
                            if (bit != buffers_.end() && !bit->second.empty()) {
                                bank = bit->second;
                            }
                        }
                        if (bank.empty()) bank.push_back(fit->second);
                        cand_embs.push_back(bank.front());
                        cand_ids.push_back(cid);
                    }
                    if (!cand_ids.empty()) {
                        int gbest = cheb_gr_claim_best(raw_id, cand_embs, cand_ids, cheb_gr_dim);
                        if (gbest >= 0) {
                            best_id = gbest;
                            best_joint = 1.0f;
                            best_sim_raw = sim_lookup(raw_id, gbest);
                            if (best_sim_raw <= -2.0f) {
                                auto fit = features_.find(gbest);
                                if (fit != features_.end())
                                    best_sim_raw = dot(emb, fit->second);
                            }
                            stats_.cheb_gr_claim_attempts += 1;
                            stats_.cheb_gr_claim_accepted += 1;
                            cheb_gr_scored = true;
                        } else {
                            stats_.cheb_gr_claim_attempts += 1;
                        }
                    }
                }
            }

            if (cheb_gr_scored) {
                record_relink_accept(raw_id, best_id, best_sim_raw, best_iou, best_center, best_maha);
            } else {
            bool _use_legacy_joint = iou_weight_ > 0.0f || mahalanobis_weight_ > 0.0f;
            bool _use_unified_score = w_sim_base_ > 0.0f || w_iou_base_ > 0.0f || w_maha_base_ > 0.0f;

            if (!_use_unified_score && !_use_legacy_joint) {
                best_joint = current_sim_thresh;
                second_best_joint = current_sim_thresh - 1.0f;
            }

            for (const auto& cand : candidates_to_score) {
                int cid = cand.cid;
                float sim;
                if (has_emb) {
                    float pre_sim = sim_lookup(raw_id, cid);
                    if (pre_sim > -2.0f) {
                        sim = pre_sim;
                    } else {
                        const auto feature_it = features_.find(cid);
                        std::vector<float> ref = buffer_size_ > 1 ? buffer_mean(cid) : feature_it->second;
                        if (ref.empty()) ref = feature_it->second;
                        sim = dot(emb, ref);
                    }
                } else {
                    sim = 0.0f;
                }

                if (has_emb && sim < current_sim_thresh) {
                    stats_.reject_similarity += 1;
                    continue;
                }

                float maha_score = 0.0f;
                float dynamic_thresh = get_dynamic_mahalanobis_threshold(cand.cid);
                if (dynamic_thresh > 0.0f && cand.maha > 0.0f) {
                    maha_score = std::max(0.0f, 1.0f - cand.maha / dynamic_thresh);
                }

                float joint;
                if (_use_unified_score) {
                    float w_sim = w_sim_base_;
                    float w_iou = w_iou_base_;
                    float w_maha = w_maha_base_;

                    if (n_gate_passed > 1) {
                        float ambiguity_factor = std::min(1.0f, (n_gate_passed - 1) / 8.0f);
                        w_sim += shift_ambiguity_ * ambiguity_factor;
                        w_iou -= shift_ambiguity_ * ambiguity_factor;
                    }

                    float lost_factor = std::min(1.0f, static_cast<float>(cand.age) / std::max(1, ttl_));
                    w_sim += shift_lost_age_ * lost_factor;
                    w_iou -= shift_lost_age_ * lost_factor;

                    w_sim = std::max(0.0f, w_sim);
                    w_iou = std::max(0.0f, w_iou);
                    w_maha = std::max(0.0f, w_maha);
                    float sum_w = w_sim + w_iou + w_maha;
                    if (sum_w > 0.0f) {
                        w_sim /= sum_w;
                        w_iou /= sum_w;
                        w_maha /= sum_w;
                    }

                    joint = w_sim * sim + w_iou * cand.iou + w_maha * maha_score;
                } else if (_use_legacy_joint) {
                    joint = sim + iou_weight_ * cand.iou + mahalanobis_weight_ * maha_score;
                } else {
                    joint = sim;
                }

                // Spatial probability penalty: closer to the predicted cloud center
                // (smaller D^2) costs less. cost = 1 - exp(-D^2/2).
                if (kalman_penalty_weight_ > 0.0f && cand.kalman_d2 >= 0.0f) {
                    joint -= kalman_penalty_weight_ * (1.0f - std::exp(-0.5f * cand.kalman_d2));
                }

                if (joint > best_joint) {
                    if (best_id >= 0) second_best_joint = best_joint;
                    best_id = cid;
                    best_joint = joint;
                    best_sim_raw = sim;
                    best_iou = cand.iou;
                    best_center = cand.center_norm;
                    best_maha = cand.maha;
                } else {
                    if (joint > second_best_joint) second_best_joint = joint;
                    stats_.reject_similarity += 1;
                }
            }

            float effective_margin = reciprocal_margin_;
            if (best_id >= 0) {
                if (dynamic_margin_crowd_ > 0.0f && n_gate_passed > 1) {
                    float crowd_factor = std::min(1.0f, (n_gate_passed - 1) / 8.0f);
                    effective_margin += dynamic_margin_crowd_ * crowd_factor;
                }
                if (dynamic_margin_age_ > 0.0f) {
                    int lost_frames = frame_id - last_seen_.at(best_id);
                    float age_factor = std::min(1.0f, static_cast<float>(lost_frames) / std::max(1, ttl_));
                    effective_margin += dynamic_margin_age_ * age_factor;
                }
            }

            if (best_id >= 0 && effective_margin > 0.0f) {
                if (best_joint - second_best_joint < effective_margin) {
                    stats_.reject_margin += 1; best_id = -1;
                }
            }

            if (best_id >= 0) {
                record_relink_accept(raw_id, best_id, best_sim_raw, best_iou, best_center, best_maha);
            } else {
                record_new_identity(raw_id, claim_ready);
            }
            } // closes cheb_gr_scored else { standard scoring }
            } // closes GPU-scored else { fall-through to inline loop }
        }

        int canonical = alias_.at(raw_id);
        commit_reference_state(canonical, box, emb, has_emb, is_clean, frame_id);
        if (assigned.count(canonical)) {
            canonical = split_on_collision(raw_id);
        }
        assigned.insert(canonical);

        if (ho_enabled_ && has_emb && is_clean) {
            feed_newborn_head(raw_id, emb, frame_id);
        }
        expire_dead_archive(frame_id);

        return canonical;
    }

    void resolve_batch_from_host(
        int n_tracks,
        const float* boxes,
        const float* scores,
        int* ids,
        const float* embeddings,
        int embedding_dim,
        int frame_id,
        int frame_w,
        int frame_h
    ) {
        std::unordered_set<int> assigned;
        for (int i = 0; i < n_tracks; ++i) {
            RelinkBox box{
                boxes[i*4], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]};
            float score = scores[i];
            int raw_id = ids[i];
            std::vector<float> emb;
            bool has_emb = (embeddings != nullptr);
            if (has_emb) {
                const float* e = embeddings + i * embedding_dim;
                emb.assign(e, e + embedding_dim);
            }
            int canonical = resolve_cpp(raw_id, emb, has_emb, box, score,
                                        frame_id, frame_w, frame_h, assigned);
            ids[i] = canonical;
        }
    }

    struct HandoverSnapshot {
        int raw_id;
        int frame_id;
        int birth_frame;
        std::vector<std::vector<float>> head_samples;
        // Every in-window archive bank joins the Cheb-GR graph; only entries
        // with gap in [1, max_gap] are scoreable candidates, the rest are
        // context (keeps the re-ranked distance scale close to the offline
        // graph the max_cost operating point came from — Python parity).
        struct ArchiveEntry {
            int tid;
            int canonical;
            bool is_candidate;
            std::vector<std::vector<float>> bank;  // deep copy
        };
        std::vector<ArchiveEntry> archive;
    };

    struct HandoverScoreResult {
        int best_id = -1;
        float best_cost = std::numeric_limits<float>::infinity();
        float second_cost = std::numeric_limits<float>::infinity();
    };

    void set_handover_params(bool enabled, float max_cost, float margin,
                              int max_gap, int decide_n, int min_head,
                              float pool_frac, float cheb_lambda,
                              int k2, int max_fwd, float fuse_lambda) {
        ho_enabled_ = enabled;
        ho_max_cost_ = max_cost;
        ho_margin_ = margin;
        ho_max_gap_ = max_gap;
        ho_decide_n_ = decide_n;
        ho_min_head_ = min_head;
        ho_pool_frac_ = pool_frac;
        ho_cheb_lambda_ = cheb_lambda;
        ho_k2_ = k2;
        ho_max_fwd_ = max_fwd;
        ho_fuse_lambda_ = fuse_lambda;
    }

    /// Attach the crop store used for borderline re-query (nullptr disables).
    void set_crop_store(ReidCropStore* store) { ho_crop_store_ = store; }

    /**
     * @brief Configure borderline re-query.
     * @param band Half-width around the cost/margin gate that triggers re-query
     *             (0 disables).
     * @param top  Densify only the top-``top`` sparse-ranked candidates
     *             (0 = every in-graph candidate).
     */
    void set_handover_requery(float band, int top) {
        ho_requery_band_ = std::max(0.0f, band);
        ho_requery_top_ = std::max(0, top);
    }

    static bool ho_usable_emb(const std::vector<float>& emb) {
        // Budgeted extraction leaves all-zero rows for tracks without a fresh
        // embedding; zero vectors are mutual nearest neighbours in the
        // k-reciprocal graph and poison head/bank evidence.
        float norm_sq = 0.0f;
        for (float v : emb) norm_sq += v * v;
        return norm_sq >= 1e-8f;
    }

    void feed_newborn_head(int raw_id, const std::vector<float>& emb,
                           int frame_id) {
        if (!ho_enabled_ || !ho_usable_emb(emb)) return;
        auto [bit, born] = ho_track_birth_.try_emplace(raw_id, frame_id);
        // Head window only (Python parity: r.frame < birth + decide_n), one
        // sample per frame — resolve() and feed_frame_embeddings() may both
        // fire for the same track in the same frame.
        if (frame_id - bit->second >= ho_decide_n_) return;
        auto [fit, fresh] = ho_head_last_frame_.try_emplace(raw_id, frame_id);
        if (!fresh && fit->second == frame_id) return;
        fit->second = frame_id;
        ho_newborn_heads_[raw_id].push_back(emb);
    }

    void feed_life_bank(int canonical, const std::vector<float>& emb,
                        int frame_id) {
        if (!ho_enabled_ || !ho_usable_emb(emb)) return;
        auto& b = ho_life_bank_[canonical];
        if (b.last_frame == frame_id) return;
        b.last_frame = frame_id;
        if (b.skip > 0) { b.skip--; return; }
        b.skip = b.stride - 1;
        b.samples.push_back(emb);
        if (static_cast<int>(b.samples.size()) >= 2 * kHoBankCap) {
            std::vector<std::vector<float>> kept;
            kept.reserve(kHoBankCap);
            for (size_t i = 0; i < b.samples.size(); i += 2)
                kept.push_back(std::move(b.samples[i]));
            b.samples.swap(kept);
            b.stride *= 2;
        }
    }

    void archive_dead_track(int track_id, int frame_id) {
        if (!ho_enabled_) return;
        int canonical = canonical_id(track_id);
        std::vector<std::vector<float>> bank;
        auto bit = ho_life_bank_.find(canonical);
        if (bit != ho_life_bank_.end()) {
            bank = std::move(bit->second.samples);
            ho_life_bank_.erase(bit);
        }
        std::vector<float> embedding;
        auto fit = features_.find(canonical);
        if (fit != features_.end()) embedding = fit->second;
        // features_ holds a {0.0f} placeholder for tracks that never had an
        // embedding — size 1 never matches emb_dim, so it is unusable.
        if (bank.empty() && embedding.size() <= 1) return;
        auto& entry = ho_dead_archive_[track_id];
        entry.embedding = std::move(embedding);
        entry.bank = std::move(bank);
        // Death = last frame the track was actually emitted (Python parity:
        // tracklet end), not the frame we noticed it missing.
        auto la = ho_last_active_.find(track_id);
        entry.death_frame = (la != ho_last_active_.end()) ? la->second
                                                          : frame_id - 1;
        entry.canonical_label = canonical;
        ho_last_active_.erase(track_id);
    }

    int try_handover(int raw_id, int frame_id, int emb_dim) {
        if (!ho_enabled_) return -1;

        const int debug_level = []() {
            const char* val = std::getenv("SACCADE_HO_DEBUG_LEVEL");
            return val ? std::atoi(val) : 999;
        }();

        HandoverSnapshot snap;
        if (!build_handover_snapshot(raw_id, frame_id, emb_dim, snap))
            return -1;

        // --- debug gating ---
        if (debug_level <= 0) return -1;
        if (debug_level <= 1) {
            dump_handover_replay(snap);
            return -1;
        }
        if (debug_level <= 2) {
            dump_handover_replay(snap);
            return -1;
        }

        HandoverScoreResult best;
        if (!score_handover_candidates(snap, emb_dim, best))
            return -1;
        if (debug_level <= 3) {
            dump_handover_replay(snap);
            return -1;
        }
        if (debug_level <= 4) {
            dump_handover_replay(snap);
            return -1;
        }
        if (debug_level == 5) {
            int death = -1;
            auto it = ho_dead_archive_.find(best.best_id);
            if (it != ho_dead_archive_.end()) death = it->second.death_frame;
            std::fprintf(stderr,
                         "[ho] frame=%d raw=%d claims tid=%d cost=%.4f "
                         "second=%.4f H=%zu gap=%d\n",
                         frame_id, raw_id, best.best_id, best.best_cost,
                         best.second_cost, snap.head_samples.size(),
                         snap.birth_frame - death);
        }

        return apply_handover(snap, best);
    }

    bool build_handover_snapshot(int raw_id, int frame_id, int emb_dim,
                                 HandoverSnapshot& snap) {
        auto birth_it = ho_track_birth_.find(raw_id);
        if (birth_it == ho_track_birth_.end()) return false;
        int birth = birth_it->second;
        if (frame_id - birth < ho_decide_n_) return false;

        auto head_it = ho_newborn_heads_.find(raw_id);
        if (head_it == ho_newborn_heads_.end()) return false;
        const auto& head = head_it->second;
        if (static_cast<int>(head.size()) < ho_min_head_) return false;

        snap.raw_id = raw_id;
        snap.frame_id = frame_id;
        snap.birth_frame = birth;

        snap.head_samples.reserve(head.size());
        for (const auto& h : head) {
            if (static_cast<int>(h.size()) != emb_dim) continue;
            snap.head_samples.push_back(h);
        }
        if (snap.head_samples.empty()) return false;

        bool any_candidate = false;
        for (auto& [tid, entry] : ho_dead_archive_) {
            int gap = birth - entry.death_frame;
            if (gap > ho_max_gap_) continue;  // expired relative to this newborn
            HandoverSnapshot::ArchiveEntry ae;
            ae.tid = tid;
            ae.canonical = entry.canonical_label;
            ae.is_candidate = (gap >= 1);
            for (const auto& s : entry.bank) {
                if (static_cast<int>(s.size()) == emb_dim)
                    ae.bank.push_back(s);  // deep copy
            }
            if (ae.bank.empty() &&
                static_cast<int>(entry.embedding.size()) == emb_dim &&
                ho_usable_emb(entry.embedding)) {
                ae.bank.push_back(entry.embedding);
            }
            if (ae.bank.empty()) continue;
            any_candidate = any_candidate || ae.is_candidate;
            snap.archive.push_back(std::move(ae));
        }

        return any_candidate;
    }

    bool score_handover_candidates(const HandoverSnapshot& snap,
                                   int emb_dim,
                                   HandoverScoreResult& result,
                                   bool allow_requery = true) {
        int H = static_cast<int>(snap.head_samples.size());
        if (H <= 0 || emb_dim <= 0) return false;
        int total_rows = H;
        for (const auto& ae : snap.archive)
            total_rows += static_cast<int>(ae.bank.size());
        if (total_rows <= H) return false;

        std::vector<float> feats_data(
            static_cast<size_t>(total_rows) * static_cast<size_t>(emb_dim));
        int pos = 0;
        for (const auto& h : snap.head_samples) {
            std::copy(h.begin(), h.end(), feats_data.begin() + static_cast<size_t>(pos) * emb_dim);
            pos++;
        }
        std::vector<std::pair<int, int>> spans(snap.archive.size());
        for (size_t ci = 0; ci < snap.archive.size(); ++ci) {
            int lo = pos;
            for (const auto& s : snap.archive[ci].bank) {
                std::copy(s.begin(), s.end(), feats_data.begin() + static_cast<size_t>(pos) * emb_dim);
                pos++;
            }
            spans[ci] = {lo, pos};
        }
        if (pos != total_rows) return false;

        // feats_data is row-major (one sample per contiguous emb_dim block);
        // Eigen::MatrixXf defaults to column-major, so map row-major first.
        Eigen::MatrixXf feats =
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>>(
                feats_data.data(), total_rows, emb_dim);
        auto sdist = saccade::cheb_gr_kreciprocal_self(
            feats, ho_cheb_lambda_, ho_k2_, ho_max_fwd_, ho_fuse_lambda_);

        float best_cost = std::numeric_limits<float>::infinity();
        float second_cost = std::numeric_limits<float>::infinity();
        int best_id = -1;
        std::vector<std::pair<float, int>> cand_costs;  // (cost, tid) for re-query

        auto head_rows = sdist.topRows(H);
        for (size_t ci = 0; ci < snap.archive.size(); ++ci) {
            if (!snap.archive[ci].is_candidate) continue;
            int cand_id = snap.archive[ci].tid;
            auto [lo, hi] = spans[ci];
            std::vector<float> block;
            block.reserve(static_cast<size_t>(H) * (hi - lo));
            for (int r = 0; r < H; ++r)
                for (int c = lo; c < hi; ++c)
                    block.push_back(head_rows(r, c));
            int k = std::max(1, static_cast<int>(
                std::round(ho_pool_frac_ * static_cast<float>(block.size()))));
            k = std::min(k, static_cast<int>(block.size()));
            std::nth_element(block.begin(), block.begin() + (k - 1), block.end());
            float cost = 0.0f;
            for (int i = 0; i < k; ++i) cost += block[i];
            cost /= static_cast<float>(k);
            cand_costs.emplace_back(cost, cand_id);
            bool wins = cost < best_cost ||
                        (cost == best_cost && best_id >= 0 && cand_id < best_id);
            if (wins) {
                second_cost = std::min(second_cost, best_cost);
                best_cost = cost;
                best_id = cand_id;
            } else if (cost < second_cost) {
                second_cost = cost;
            }
        }

        // Borderline re-query: if a band-sized perturbation could flip a gate,
        // re-extract dense recent-tail banks for the top candidates and rescore
        // by recursing through this same path (reusing the validated Eigen
        // row-major mapping — see project_eigen_rowmajor_chebgr_bug). Default-off
        // and one-shot (allow_requery guards against re-entry).
        if (allow_requery && ho_crop_store_ != nullptr && ho_requery_band_ > 0.0f
            && best_id >= 0) {
            float obs_margin0 = second_cost - best_cost;
            bool cost_borderline =
                std::fabs(best_cost - ho_max_cost_) <= ho_requery_band_;
            bool margin_borderline =
                ho_margin_ > 0.0f
                && best_cost <= ho_max_cost_ + ho_requery_band_
                && std::fabs(obs_margin0 - ho_margin_) <= ho_requery_band_;
            if (cost_borderline || margin_borderline) {
                HandoverSnapshot dense;
                if (requery_densify(snap, cand_costs, emb_dim, dense)) {
                    return score_handover_candidates(dense, emb_dim, result,
                                                     /*allow_requery=*/false);
                }
            }
        }

        if (best_id < 0 || best_cost > ho_max_cost_) return false;
        float obs_margin = second_cost - best_cost;
        if (ho_margin_ > 0.0f && obs_margin < ho_margin_) return false;

        result.best_id = best_id;
        result.best_cost = best_cost;
        result.second_cost = second_cost;
        return true;
    }

    /**
     * @brief Build a snapshot with dense re-query banks for the top candidates.
     *
     * Selects the top-``ho_requery_top_`` candidates by sparse cost (0 = all),
     * asks the crop store to re-extract their dense recent-tail embeddings, and
     * returns a copy of @p snap with those candidates' banks replaced. Other
     * entries keep their sparse banks as graph context. Returns false (no
     * rescore) if nothing could be densified.
     */
    bool requery_densify(const HandoverSnapshot& snap,
                         std::vector<std::pair<float, int>> cand_costs,
                         int emb_dim, HandoverSnapshot& out) {
        if (ho_crop_store_ == nullptr || cand_costs.empty()) return false;
        std::sort(cand_costs.begin(), cand_costs.end());
        int n_top = ho_requery_top_ > 0
                        ? std::min(ho_requery_top_,
                                   static_cast<int>(cand_costs.size()))
                        : static_cast<int>(cand_costs.size());
        std::vector<uint64_t> uids;
        uids.reserve(n_top);
        for (int i = 0; i < n_top; ++i)
            uids.push_back(static_cast<uint64_t>(cand_costs[i].second));

        const int depth = ho_crop_store_->ring_depth();
        const int dim = ho_crop_store_->embed_dim();
        if (depth <= 0 || dim != emb_dim) return false;
        std::vector<float> embeds(
            static_cast<size_t>(depth) * uids.size() * emb_dim);
        std::vector<int> counts(uids.size(), 0);
        int total = ho_crop_store_->requery_extract(
            uids.data(), static_cast<int>(uids.size()), embeds.data(),
            counts.data());
        if (total <= 0) return false;

        // Map tid -> dense bank rows.
        std::unordered_map<int, std::vector<std::vector<float>>> dense_banks;
        size_t row = 0;
        for (size_t i = 0; i < uids.size(); ++i) {
            std::vector<std::vector<float>> bank;
            for (int r = 0; r < counts[i]; ++r) {
                const float* src = embeds.data() + (row + r) * emb_dim;
                bank.emplace_back(src, src + emb_dim);
            }
            row += counts[i];
            if (!bank.empty())
                dense_banks[static_cast<int>(cand_costs[i].second)] =
                    std::move(bank);
        }
        if (dense_banks.empty()) return false;

        out = snap;  // deep copy
        bool changed = false;
        for (auto& ae : out.archive) {
            auto it = dense_banks.find(ae.tid);
            if (it != dense_banks.end()) {
                ae.bank = std::move(it->second);
                changed = true;
            }
        }
        return changed;
    }

    int apply_handover(const HandoverSnapshot& snap, const HandoverScoreResult& result) {
        int best_id = result.best_id;
        auto arch_it = ho_dead_archive_.find(best_id);
        if (arch_it == ho_dead_archive_.end()) return -1;
        int canon = arch_it->second.canonical_label;

        alias_[snap.raw_id] = canon;
        ho_dead_archive_.erase(best_id);  // an identity is revived at most once
        // The consumed identity can never be scored again — free its ring crops.
        if (ho_crop_store_) ho_crop_store_->evict(static_cast<uint64_t>(best_id));
        // Keep ho_track_birth_: erasing it would let feed_newborn_head re-open
        // the head window for this (already decided) track.
        ho_newborn_heads_.erase(snap.raw_id);
        ho_head_last_frame_.erase(snap.raw_id);
        // The newborn's provisional identity is absorbed by canon: migrate its
        // clean bank samples and drop the stale identity-keyed state so it can
        // never be archived as a separate (living) identity later.
        if (snap.raw_id != canon) {
            auto bit = ho_life_bank_.find(snap.raw_id);
            if (bit != ho_life_bank_.end()) {
                auto& dst = ho_life_bank_[canon];
                for (auto& s : bit->second.samples)
                    dst.samples.push_back(std::move(s));
                if (static_cast<int>(dst.samples.size()) >= 2 * kHoBankCap) {
                    std::vector<std::vector<float>> kept;
                    kept.reserve(kHoBankCap);
                    for (size_t i = 0; i < dst.samples.size(); i += 2)
                        kept.push_back(std::move(dst.samples[i]));
                    dst.samples.swap(kept);
                    dst.stride *= 2;
                }
                ho_life_bank_.erase(snap.raw_id);
            }
            features_.erase(snap.raw_id);
            buffers_.erase(snap.raw_id);
            ho_last_active_.erase(snap.raw_id);
        }
        ho_handover_count_++;
        return canon;
    }

    void dump_handover_replay(const HandoverSnapshot& snap) {
        static int dump_seq = 0;
        std::string path = "/tmp/saccade_ho_frame" +
            std::to_string(snap.frame_id) + "_" + std::to_string(dump_seq++) + ".txt";
        std::ofstream f(path);
        if (!f) return;
        f << "frame " << snap.frame_id << "\n";
        f << "raw_id " << snap.raw_id << "\n";
        f << "birth " << snap.birth_frame << "\n";
        f << "H " << snap.head_samples.size() << " archive " << snap.archive.size() << "\n";
        for (const auto& h : snap.head_samples) {
            f << "head";
            for (float v : h) f << " " << v;
            f << "\n";
        }
        for (const auto& ae : snap.archive) {
            for (const auto& s : ae.bank) {
                f << "bank " << ae.tid << " " << (ae.is_candidate ? 1 : 0);
                for (float v : s) f << " " << v;
                f << "\n";
            }
        }
    }

    void expire_dead_archive(int frame_id) {
        if (!ho_enabled_) return;
        // decide_n slack: a newborn born at B decides at B+decide_n and may
        // still claim a candidate with gap == max_gap (death == B - max_gap).
        std::vector<int> expired;
        for (auto& [tid, entry] : ho_dead_archive_)
            if (frame_id - entry.death_frame > ho_max_gap_ + ho_decide_n_)
                expired.push_back(tid);
        for (int tid : expired) {
            ho_dead_archive_.erase(tid);
            // Beyond the claim window the crops are dead weight; freeing them
            // keeps LRU pressure off live tracks' recent tails.
            if (ho_crop_store_) ho_crop_store_->evict(static_cast<uint64_t>(tid));
        }
    }

    int handover_count() const { return ho_handover_count_; }

    void feed_frame_embeddings(const std::vector<float>& emb_flat,
                                int emb_dim, int frame_id,
                                const std::vector<int>& track_ids,
                                const std::vector<float>& scores,
                                const std::vector<int>& clean_flags = {}) {
        if (!ho_enabled_) return;
        int n = static_cast<int>(track_ids.size());
        // Birth = first frame the track is *emitted*, independent of when its
        // first clean embedding arrives (budgeted extraction can lag several
        // frames). A late birth would shift the [birth-max_gap, birth-1]
        // candidate window onto tracks that co-existed with this newborn,
        // breaking the causal disjointness guarantee (gap >= 1).
        for (int i = 0; i < n; ++i)
            ho_track_birth_.try_emplace(track_ids[i], frame_id);
        for (int i = 0; i < n; ++i) {
            if (static_cast<int>(emb_flat.size()) < (i + 1) * emb_dim) break;
            int raw_id = track_ids[i];
            const float* e = emb_flat.data() + i * emb_dim;
            // Budgeted extraction leaves zero rows for tracks without a fresh
            // embedding this frame — zero vectors are mutual nearest neighbours
            // in the k-reciprocal graph and must never enter head or bank.
            float norm_sq = 0.0f;
            for (int d = 0; d < emb_dim; ++d) norm_sq += e[d] * e[d];
            if (norm_sq < 1e-8f) continue;
            if (i < static_cast<int>(clean_flags.size()) && !clean_flags[i])
                continue;
            std::vector<float> emb(e, e + emb_dim);
            feed_newborn_head(raw_id, emb, frame_id);
            feed_life_bank(canonical_id(raw_id), emb, frame_id);
        }
        // One-shot decision (Python parity): each newborn decides exactly once,
        // decide_n frames after birth. The event is consumed whether or not a
        // handover happened.
        for (int i = 0; i < n; ++i) {
            int raw_id = track_ids[i];
            auto bit = ho_track_birth_.find(raw_id);
            if (bit == ho_track_birth_.end()) continue;
            if (frame_id - bit->second < ho_decide_n_) continue;
            if (!ho_decided_.insert(raw_id).second) continue;
            try_handover(raw_id, frame_id, emb_dim);
        }
        expire_dead_archive(frame_id);
    }

    void prune_and_archive(const std::vector<int>& active_ids, int frame_id) {
        if (!ho_enabled_) return;
        std::unordered_set<int> active_set(active_ids.begin(), active_ids.end());
        // A track that reappears was never dead — its identity must not be
        // claimable by newborns (temporary occlusion is not a death).
        for (int tid : active_ids) {
            ho_dead_archive_.erase(tid);
            ho_last_active_[tid] = frame_id;
        }
        // active_ids are *emitted* (canonical) ids. A raw/stale key whose
        // canonical identity is still emitted is alive — archiving it would
        // put a living identity back into the claimable pool.
        auto is_active = [&](int tid) {
            return active_set.count(tid) || active_set.count(canonical_id(tid));
        };
        std::vector<int> inactive;
        for (auto& [tid, _] : features_) {
            if (!is_active(tid)) inactive.push_back(tid);
        }
        // Tracks known only through the feed path (tracker-mode ReID without
        // semantic relink) live in ho_life_bank_ but never enter features_.
        for (auto& [tid, _] : ho_life_bank_) {
            if (!is_active(tid) && !features_.count(tid))
                inactive.push_back(tid);
        }
        for (int tid : inactive) {
            archive_dead_track(tid, frame_id);
            features_.erase(tid);
            last_seen_.erase(tid);
            last_boxes_.erase(tid);
            buffers_.erase(tid);
            ema_h_.erase(tid);
            foot_history_.erase(tid);
            alias_.erase(tid);
            pending_claims_.erase(tid);
            pending_heads_.erase(tid);
            ho_newborn_heads_.erase(tid);
            ho_track_birth_.erase(tid);
            ho_head_last_frame_.erase(tid);
            ho_decided_.erase(tid);
        }
    }

    // --- end handover methods ---

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
        const float* data = arr.data();
        return std::vector<float>(data, data + static_cast<size_t>(arr.shape(0)));
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

    bool is_clean_observation(const RelinkBox& box, float score, int frame_w, int frame_h) const {
        if (clean_score_threshold_ <= 0.0f && clean_margin_ratio_ <= 0.0f) {
            return true;
        }
        const float bw = box.x2 - box.x1;
        const float bh = box.y2 - box.y1;
        const float aspect = bw > 0.0f ? bh / bw : 0.0f;
        const float margin_w = frame_w * clean_margin_ratio_;
        const float margin_h = frame_h * clean_margin_ratio_;
        return score >= clean_score_threshold_ &&
               box.x1 >= margin_w &&
               box.y1 >= margin_h &&
               box.x2 <= frame_w - margin_w &&
               box.y2 <= frame_h - margin_h &&
               aspect >= clean_min_aspect_ &&
               aspect <= clean_max_aspect_;
    }

    void record_relink_accept(
        int raw_id,
        int best_id,
        float sim,
        float iou,
        float center_dist,
        float maha
    ) {
        stats_.accepted += 1;
        if (delayed_claim_ && raw_id != best_id) {
            stats_.delayed_claim_accepted += 1;
            deferred_alias_[raw_id] = best_id;
            pending_claims_.erase(raw_id);
        }
        accept_sims_.push_back(sim);
        accept_ious_.push_back(iou);
        accept_center_dists_.push_back(center_dist);
        accept_mahas_.push_back(maha);
        alias_[raw_id] = best_id;
    }

    void record_new_identity(int raw_id, bool claim_ready) {
        stats_.new_ids += 1;
        alias_[raw_id] = raw_id;
        if (delayed_claim_ && claim_ready) {
            pending_claims_.erase(raw_id);
        }
    }

    bool evaluate_candidate_gates(
        int raw_id,
        int cid,
        const RelinkBox& box,
        int frame_id,
        int frame_w,
        int frame_h,
        const float* gate,
        RelinkCandidateGates& out,
        bool record_stats = true,
        bool record_bidir_accept = true
    ) {
        const int age = frame_id - last_seen_.at(cid);
        if (age < min_lost_frames_ || age > ttl_) {
            if (record_stats) stats_.reject_age += 1;
            return false;
        }

        const bool speed_exceeds = gate ? (gate[4] > 0.5f)
            : exceeds_max_speed(
                  box, last_boxes_.at(cid), age, kalman_person_height_m_,
                  kalman_fps_, kalman_max_speed_mps_);
        if (speed_exceeds) {
            if (record_stats) stats_.reject_speed += 1;
            return false;
        }

        float center_norm = 0.0f;
        float iou = 0.0f;
        if (gate) {
            center_norm = gate[2];
            iou = gate[3];
        } else {
            auto sp = spatial_metrics(box, last_boxes_.at(cid), frame_w, frame_h);
            center_norm = sp.first;
            iou = sp.second;
        }

        if (center_norm > 1.0f) {
            if (record_stats) stats_.reject_spatial += 1;
            return false;
        }

        float kalman_d2 = -1.0f;
        bool kalman_gated = false;
        if (kalman_gate_) {
            if (gate && gate[0] >= 0.0f) {
                if (gate[5] > 0.5f) {
                    if (record_stats) stats_.reject_direction += 1;
                    return false;
                }
                kalman_d2 = gate[0];
                if (kalman_d2 > kalman_chi2_) {
                    if (record_stats) stats_.reject_kalman += 1;
                    return false;
                }
                kalman_gated = true;
            } else {
                const auto motion_it = motion_.find(cid);
                if (motion_it != motion_.end()) {
                    if (direction_behind(box, motion_it->second, kalman_dir_min_cos_, kalman_dir_min_speed_)) {
                        if (record_stats) stats_.reject_direction += 1;
                        return false;
                    }
                    const int delta = motion_it->second.frame >= 0
                                          ? (frame_id - motion_it->second.frame)
                                          : 0;
                    const int dims = bidirectional_ ? 2 : 4;
                    kalman_d2 = (dims <= 2)
                        ? kalman_gate_dist_2d(
                              box, motion_it->second, delta, kalman_person_height_m_,
                              kalman_accel_long_, kalman_accel_lat_, kalman_fps_)
                        : kalman_gate_dist(
                              box, motion_it->second, delta, kalman_person_height_m_,
                              kalman_accel_long_, kalman_accel_lat_, kalman_fps_);
                    if (kalman_d2 > kalman_chi2_) {
                        if (record_stats) stats_.reject_kalman += 1;
                        return false;
                    }
                    kalman_gated = true;
                }
            }
        }

        if (bidirectional_) {
            if (!bridge_scale_gate_ok(cid, raw_id)) {
                if (record_stats) stats_.reject_backward += 1;
                return false;
            }
            if (gate && gate[1] >= 0.0f) {
                if (gate[1] > bridge_px_) {
                    if (record_stats) stats_.reject_backward += 1;
                    return false;
                }
                if (record_bidir_accept) stats_.accept_bidir += 1;
            } else {
                auto fh = foot_history_.find(cid);
                if (fh != foot_history_.end()) {
                    const float b_cx = (box.x1 + box.x2) * 0.5f;
                    const float b_cy = (box.y1 + box.y2) * 0.5f;
                    const int gap = frame_id - last_seen_.at(cid);
                    const float dist = midpoint_bridge_dist(cid, raw_id, gap, b_cx, b_cy);
                    if (dist > bridge_px_) {
                        if (record_stats) stats_.reject_backward += 1;
                        return false;
                    }
                    if (record_bidir_accept) stats_.accept_bidir += 1;
                }
            }
        }

        if (!kalman_gated && (center_norm > spatial_gate_ || iou < min_iou_)) {
            if (record_stats) stats_.reject_spatial += 1;
            return false;
        }

        float maha = 0.0f;
        if (mahalanobis_threshold_ > 0.0f) {
            const auto motion_it = motion_.find(cid);
            if (motion_it == motion_.end()) {
                if (record_stats) stats_.reject_mahalanobis += 1;
                return false;
            }
            maha = mahalanobis(box, motion_it->second);
            const float dynamic_thresh = get_dynamic_mahalanobis_threshold(cid);
            if (maha > dynamic_thresh) {
                if (record_stats) stats_.reject_mahalanobis += 1;
                return false;
            }
        }

        if (min_consistency_ > 0.0f && buffer_size_ > 1) {
            if (buffer_consistency(cid) < min_consistency_) {
                if (record_stats) stats_.reject_consistency += 1;
                return false;
            }
        }

        out = {cid, age, iou, center_norm, maha, kalman_gated ? kalman_d2 : -1.0f};
        return true;
    }

    void commit_reference_state(
        int canonical,
        const RelinkBox& box,
        const std::vector<float>& emb,
        bool has_emb,
        bool is_clean,
        int frame_id
    ) {
        if (!is_clean) {
            stats_.reject_quality += 1;
        } else if (has_emb) {
            if (buffer_size_ > 1) {
                auto& buf = buffers_[canonical];
                buf.push_back(emb);
                if (static_cast<int>(buf.size()) > buffer_size_) {
                    buf.erase(buf.begin());
                }
                features_[canonical] = buffer_mean(canonical);
            } else {
                auto old_it = features_.find(canonical);
                if (old_it == features_.end()) {
                    features_[canonical] = emb;
                } else {
                    std::vector<float> updated(emb.size(), 0.0f);
                    for (size_t i = 0; i < emb.size(); ++i) {
                        updated[i] = ema_beta_ * old_it->second[i] + (1.0f - ema_beta_) * emb[i];
                    }
                    features_[canonical] = normalize(updated);
                }
            }
        } else if (features_.find(canonical) == features_.end()) {
            features_[canonical] = std::vector<float>{0.0f};
        }
        if (ho_enabled_ && has_emb && is_clean) {
            feed_life_bank(canonical, emb, frame_id);
        }

        if (std::find(feature_order_.begin(), feature_order_.end(), canonical) == feature_order_.end()) {
            feature_order_.push_back(canonical);
        }
        last_seen_[canonical] = frame_id;
        last_boxes_[canonical] = box;
        const float bh = box.y2 - box.y1;
        auto old_h = ema_h_.find(canonical);
        ema_h_[canonical] = (old_h == ema_h_.end()) ? bh : 0.95f * old_h->second + 0.05f * bh;
        const float enc_cx = (box.x1 + box.x2) * 0.5f;
        const float enc_cy = (box.y1 + box.y2) * 0.5f;
        auto& hist = foot_history_[canonical];
        hist.push_back({enc_cx, enc_cy});
        if (hist.size() > 8) {
            hist.erase(hist.begin());
        }
    }

    static std::pair<float, float> spatial_metrics(
        const RelinkBox& box,
        const RelinkBox& old_box,
        int frame_w,
        int frame_h
    ) {
        const tracking::SpatialMetrics metrics = tracking::spatial_metrics(
            to_box4(box), to_box4(old_box), frame_w, frame_h
        );
        return {metrics.center_norm, metrics.iou};
    }

    float get_dynamic_mahalanobis_threshold(int cid) const {
        if (!exp_density_gating_ || mahalanobis_threshold_ <= 0.0f) {
            return mahalanobis_threshold_;
        }
        auto it = last_boxes_.find(cid);
        if (it == last_boxes_.end()) return mahalanobis_threshold_;
        const RelinkBox& box_t = it->second;
        const float cx_t = (box_t.x1 + box_t.x2) * 0.5f;
        const float cy_t = (box_t.y1 + box_t.y2) * 0.5f;
        const float h_t = box_t.y2 - box_t.y1;
        if (h_t <= 0.0f) return mahalanobis_threshold_;

        int density = 0;
        for (const auto& [other_id, box_j] : last_boxes_) {
            if (other_id == cid) continue;
            const float cx_j = (box_j.x1 + box_j.x2) * 0.5f;
            const float cy_j = (box_j.y1 + box_j.y2) * 0.5f;
            const float dist = std::sqrt((cx_t - cx_j) * (cx_t - cx_j) + (cy_t - cy_j) * (cy_t - cy_j));
            if (dist < exp_density_k_ * h_t) {
                density++;
            }
        }
        return mahalanobis_threshold_ * std::exp(-exp_density_eta_ * static_cast<float>(density));
    }

    static float mahalanobis(const RelinkBox& box, const RelinkMotionSnapshot& snap) {
        const float bw = std::max(1e-6f, box.x2 - box.x1);
        const float bh = std::max(1e-6f, box.y2 - box.y1);
        Eigen::Vector4f z(
            (box.x1 + box.x2) * 0.5f, (box.y1 + box.y2) * 0.5f, bw / bh, bh);
        Eigen::Vector4f x(
            snap.state[0], snap.state[1], snap.state[2], snap.state[3]);
        Eigen::Vector4f residual = z - x;
        const float h = std::max(snap.state[3], 1e-6f);
        const float pos_std = h / 20.0f;
        Eigen::Map<const Eigen::Matrix<float, 8, 8, Eigen::RowMajor>> P_full(snap.covariance.data());
        Eigen::Matrix4f S = P_full.topLeftCorner<4, 4>();
        S(0, 0) += pos_std * pos_std; S(1, 1) += pos_std * pos_std;
        S(2, 2) += 1e-2f;            S(3, 3) += pos_std * pos_std;
        Eigen::LLT<Eigen::Matrix4f> llt(S);
        if (llt.info() != Eigen::Success)
            return std::numeric_limits<float>::infinity();
        Eigen::Vector4f w = llt.matrixL().solve(residual);
        return w.squaredNorm();
    }

    // Physical reachability gate: returns true if relinking the candidate to the
    // lost track would require an implied average speed above human limits, so the
    // candidate must be a different (new) person. Snapshot-independent — uses the
    // lost track's last box + age only — so it also covers the case where no Kalman
    // motion snapshot is available (where the cloud gate would otherwise fall back
    // to the loose static spatial gate). dist/time gives the average velocity, the
    // physical meaning the positional cloud alone lacks.
    static bool exceeds_max_speed(
        const RelinkBox& box, const RelinkBox& last_box, int age,
        float person_height_m, float fps, float max_speed_mps) {
        if (max_speed_mps <= 0.0f || person_height_m <= 0.0f) return false;
        const float lcx = (last_box.x1 + last_box.x2) * 0.5f;
        const float lcy = (last_box.y1 + last_box.y2) * 0.5f;
        const float lh = last_box.y2 - last_box.y1;
        const float zcx = (box.x1 + box.x2) * 0.5f;
        const float zcy = (box.y1 + box.y2) * 0.5f;
        const float zh = box.y2 - box.y1;
        const float dpx = std::sqrt((zcx - lcx) * (zcx - lcx) + (zcy - lcy) * (zcy - lcy));
        const float dt_s = std::max(age, 1) / std::max(fps, 1e-6f);
        // px/m scale averaged over both endpoints (perspective robustness)
        const float px_per_m = 0.5f * (std::max(lh, 1e-3f) + std::max(zh, 1e-3f)) / person_height_m;
        const float speed_mps = dpx / dt_s / std::max(px_per_m, 1e-6f);
        return speed_mps > max_speed_mps;
    }

    // Velocity-direction gate: returns true if the candidate detection lies
    // "behind" the lost track's motion direction and should be rejected. Covariance
    // inflation grows the gating cloud symmetrically, so a person who just left the
    // frame moving forward can still match someone entering behind them; this culls
    // candidates whose displacement from the last position opposes the velocity.
    // Skipped for near-stationary tracks (speed < min_speed) where direction is noise.
    static bool direction_behind(
        const RelinkBox& box, const RelinkMotionSnapshot& snap, float min_cos, float min_speed) {
        if (min_cos <= -1.0f) return false;  // gate disabled
        const float vx = snap.state[4];
        const float vy = snap.state[5];
        const float speed = std::sqrt(vx * vx + vy * vy);
        if (speed < min_speed) return false;
        const float zcx = (box.x1 + box.x2) * 0.5f;
        const float zcy = (box.y1 + box.y2) * 0.5f;
        const float dx = zcx - snap.state[0];
        const float dy = zcy - snap.state[1];
        const float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 1e-3f) return false;
        const float cosang = (dx * vx + dy * vy) / (speed * dist);
        return cosang < min_cos;
    }

    // Physically-grounded, inertia-aware predict step (white-noise-acceleration).
    // The mean advances at constant velocity (momentum), and the process noise is
    // derived from an assumed real-world person height + bounded human acceleration:
    //   px_per_m = h_px / person_height_m   (monocular metric scale from box height)
    //   sigma_a  = max_accel[m/s^2] * px_per_m / fps^2   (px/frame^2)
    // The acceleration covariance is anisotropic, decomposed along the velocity
    // direction: longitudinal (speed up / slow down) vs lateral (turning), with
    // accel_long >= accel_lat so the cloud stretches *along* the motion (velocity
    // inertia) and stays tight sideways. This replaces the unbounded h/160 velocity
    // noise that implied implausible accelerations / reversals.
    // Eigen-based constant-velocity Kalman predict (F*P*F^T + standard Q).
    // Uses block structure of F = [[I, I], [0, I]] to avoid full 8x8 multiply.
    static void predict_eigen(float x[8], float P[64]) {
        for (int i = 0; i < 4; ++i) x[i] += x[i + 4];
        Eigen::Map<Eigen::Matrix<float, 8, 8, Eigen::RowMajor>> Pm(P);
        Eigen::Matrix4f TL = Pm.topLeftCorner<4, 4>();
        Eigen::Matrix4f TR = Pm.topRightCorner<4, 4>();
        Eigen::Matrix4f BL = Pm.bottomLeftCorner<4, 4>();
        const auto& BR = Pm.bottomRightCorner<4, 4>();
        Pm.topLeftCorner<4, 4>()     = TL + TR + BL + BR;
        Pm.topRightCorner<4, 4>()    = TR + BR;
        Pm.bottomLeftCorner<4, 4>()  = BL + BR;
        // bottomRight stays BR
        float h = std::max(x[3], 1e-6f);
        float Q[64];
        saccade::kf_gpu::get_Q(h, Q);
        for (int i = 0; i < 64; ++i) Pm.data()[i] += Q[i];
    }

    // One-shot k-step constant-velocity propagation.
    // P_k = F^k * P_0 * (F^k)^T + sum_{i=0}^{k-1} F^i * Q * (F^i)^T.
    // Uses the block structure of F^k = [[I, k*I], [0, I]] and the closed-form
    // sums of i (k(k-1)/2) and i² (k(k-1)(2k-1)/6) to compute Q accumulation.
    static void predict_delta(float x[8], float P[64], int k) {
        if (k <= 0) return;
        float fk = static_cast<float>(k);
        for (int i = 0; i < 4; ++i) x[i] += fk * x[i + 4];

        Eigen::Map<Eigen::Matrix<float, 8, 8, Eigen::RowMajor>> Pm(P);
        Eigen::Matrix4f TL = Pm.topLeftCorner<4, 4>();
        Eigen::Matrix4f TR = Pm.topRightCorner<4, 4>();
        Eigen::Matrix4f BL = Pm.bottomLeftCorner<4, 4>();
        Eigen::Matrix4f BR = Pm.bottomRightCorner<4, 4>();

        // F^k * P * (F^k)^T
        float fk2 = fk * fk;
        Pm.topLeftCorner<4, 4>()     = TL + fk * (TR + BL) + fk2 * BR;
        Pm.topRightCorner<4, 4>()    = TR + fk * BR;
        Pm.bottomLeftCorner<4, 4>()  = BL + fk * BR;

        // Q accumulation: sum_{i=0}^{k-1} F^i * Q * (F^i)^T
        // For diagonal Q = [Pq, Pq, aq, Pq, Vq, Vq, vq, Vq]:
        //   Pq_acc = k*Pq + k(k-1)(2k-1)/6 * Vq
        //   Vq_acc = k * Vq
        //   XV_acc  = k(k-1)/2 * Vq  (pos-vel cross term)
        float h = std::max(x[3], 1e-6f);
        float pos_sq = (h / 20.0f) * (h / 20.0f);
        float vel_sq = (h / 160.0f) * (h / 160.0f);
        float s1 = fk * (fk - 1.0f) / 2.0f;                       // sum i
        float s2 = fk * (fk - 1.0f) * (2.0f * fk - 1.0f) / 6.0f;  // sum i²
        float pos_acc = fk * pos_sq + s2 * vel_sq;
        float vel_acc = fk * vel_sq;
        float xv_acc  = s1 * vel_sq;

        Pm(0, 0) += pos_acc; Pm(0, 4) += xv_acc; Pm(4, 0) += xv_acc; Pm(4, 4) += vel_acc;
        Pm(1, 1) += pos_acc; Pm(1, 5) += xv_acc; Pm(5, 1) += xv_acc; Pm(5, 5) += vel_acc;
        Pm(2, 2) += fk * 1e-4f;   Pm(6, 6) += fk * 1e-10f;
        Pm(3, 3) += pos_acc;      Pm(3, 7) += xv_acc; Pm(7, 3) += xv_acc; Pm(7, 7) += vel_acc;
    }

    // One-shot k-step physically-grounded propagation with anisotropic
    // acceleration noise. Mean advances at constant velocity; covariance uses
    // closed-form sums of the DWNA pos-vel blocks: pos~k(4k²-1)/12, xv~k²/2,
    // vel~k. Aspect/height noise uses the standard Q accumulation.
    static void predict_phys_delta(
        float x[8], float P[64], int k,
        float person_height_m, float accel_long, float accel_lat, float fps) {
        if (k <= 0) return;
        float fk = static_cast<float>(k);
        for (int i = 0; i < 4; ++i) x[i] += fk * x[i + 4];

        Eigen::Map<Eigen::Matrix<float, 8, 8, Eigen::RowMajor>> Pm(P);
        Eigen::Matrix4f TL = Pm.topLeftCorner<4, 4>();
        Eigen::Matrix4f TR = Pm.topRightCorner<4, 4>();
        Eigen::Matrix4f BL = Pm.bottomLeftCorner<4, 4>();
        Eigen::Matrix4f BR = Pm.bottomRightCorner<4, 4>();
        float fk2 = fk * fk;
        Pm.topLeftCorner<4, 4>()     = TL + fk * (TR + BL) + fk2 * BR;
        Pm.topRightCorner<4, 4>()    = TR + fk * BR;
        Pm.bottomLeftCorner<4, 4>()  = BL + fk * BR;

        // Anisotropic acceleration noise (single-step Sa, accumulated in closed form)
        const float h = std::max(x[3], 1e-3f);
        const float px_per_m = h / std::max(person_height_m, 1e-3f);
        const float inv_fps2 = 1.0f / std::max(fps * fps, 1e-6f);
        const float sl = accel_long * px_per_m * inv_fps2;
        const float st = accel_lat * px_per_m * inv_fps2;
        float sl2 = sl * sl, st2 = st * st;
        const float vx = x[4], vy = x[5];
        const float speed = std::sqrt(vx * vx + vy * vy);
        float ux = 1.0f, uy = 0.0f;
        if (speed > 1e-6f) { ux = vx / speed; uy = vy / speed; }
        else { sl2 = st2; }
        const float Saa00 = sl2 * ux * ux + st2 * uy * uy;
        const float Saa01 = (sl2 - st2) * ux * uy;
        const float Saa11 = sl2 * uy * uy + st2 * ux * ux;
        Eigen::Matrix2f Sa;
        Sa << Saa00, Saa01, Saa01, Saa11;

        // Closed-form DWNA accumulation: pos~k(4k²-1)/12, xv~k²/2, vel~k
        float pos_f = fk * (4.0f * fk2 - 1.0f) / 12.0f;
        float xv_f  = 0.5f * fk2;
        float vel_f = fk;
        Pm.block<2, 2>(0, 0) += pos_f * Sa;
        Pm.block<2, 2>(0, 4) += xv_f  * Sa;
        Pm.block<2, 2>(4, 0) += xv_f  * Sa;
        Pm.block<2, 2>(4, 4) += vel_f * Sa;

        // Aspect/height Q noise (standard accumulation)
        float Q[64];
        saccade::kf_gpu::get_Q(x[3], Q);
        float s1 = fk * (fk - 1.0f) / 2.0f;
        float s2 = fk * (fk - 1.0f) * (2.0f * fk - 1.0f) / 6.0f;
        float pos_sq = (h / 20.0f) * (h / 20.0f);
        float vel_sq = (h / 160.0f) * (h / 160.0f);
        Pm(2, 2) += Q[2 * 8 + 2] * fk;  Pm(6, 6) += Q[6 * 8 + 6] * fk;
        Pm(3, 3) += fk * pos_sq + s2 * vel_sq;
        Pm(3, 7) += s1 * vel_sq; Pm(7, 3) += s1 * vel_sq;
        Pm(7, 7) += fk * vel_sq;
    }

    static void predict_phys(
        float x[8], float P[64], float person_height_m, float accel_long, float accel_lat, float fps) {
        // 1. constant-velocity mean (inertia)
        for (int i = 0; i < 4; ++i) x[i] += x[i + 4];

        // 2. P = F P F^T  (F = [[I, I], [0, I]]) — block-structured to avoid 8x8 multiply
        Eigen::Map<Eigen::Matrix<float, 8, 8, Eigen::RowMajor>> Pm(P);
        Eigen::Matrix4f TL = Pm.topLeftCorner<4, 4>();
        Eigen::Matrix4f TR = Pm.topRightCorner<4, 4>();
        Eigen::Matrix4f BL = Pm.bottomLeftCorner<4, 4>();
        const auto& BR = Pm.bottomRightCorner<4, 4>();
        Pm.topLeftCorner<4, 4>()     = TL + TR + BL + BR;
        Pm.topRightCorner<4, 4>()    = TR + BR;
        Pm.bottomLeftCorner<4, 4>()  = BL + BR;

        // 3. anisotropic acceleration noise on (cx,cy,vx,vy) sub-state
        const float px_per_m = std::max(x[3], 1e-3f) / std::max(person_height_m, 1e-3f);
        const float inv_fps2 = 1.0f / std::max(fps * fps, 1e-6f);
        const float sl = accel_long * px_per_m * inv_fps2;
        const float st = accel_lat * px_per_m * inv_fps2;
        float sl2 = sl * sl, st2 = st * st;
        const float vx = x[4], vy = x[5];
        const float speed = std::sqrt(vx * vx + vy * vy);
        float ux = 1.0f, uy = 0.0f;
        if (speed > 1e-6f) { ux = vx / speed; uy = vy / speed; }
        else { sl2 = st2; }
        const float Saa00 = sl2 * ux * ux + st2 * uy * uy;
        const float Saa01 = (sl2 - st2) * ux * uy;
        const float Saa11 = sl2 * uy * uy + st2 * ux * ux;
        Eigen::Matrix2f Sa;
        Sa << Saa00, Saa01, Saa01, Saa11;
        Pm.block<2, 2>(0, 0) += 0.25f * Sa;
        Pm.block<2, 2>(0, 4) += 0.5f * Sa;
        Pm.block<2, 2>(4, 0) += 0.5f * Sa;
        Pm.block<2, 2>(4, 4) += Sa;

        // 4. keep modest noise on aspect & height so S stays well-conditioned
        float Q[64];
        saccade::kf_gpu::get_Q(x[3], Q);
        Pm(2, 2) += Q[2 * 8 + 2]; Pm(6, 6) += Q[6 * 8 + 6];
        Pm(3, 3) += Q[3 * 8 + 3]; Pm(7, 7) += Q[7 * 8 + 7];
    }

    // Squared Mahalanobis distance of a detection box against a lost track's
    // Kalman distribution, self-extrapolated `delta` frames forward. When
    // person_height_m > 0 the diffusion uses the physically-grounded, inertia-aware
    // model (predict_phys); otherwise it falls back to the tracker's kf_gpu Q.
    static float kalman_gate_dist(
        const RelinkBox& box, const RelinkMotionSnapshot& snap, int delta,
        float person_height_m, float accel_long, float accel_lat, float fps) {
        float x[8];
        float P[64];
        for (int i = 0; i < 8; ++i) x[i] = snap.state[static_cast<size_t>(i)];
        for (int i = 0; i < 64; ++i) P[i] = snap.covariance[static_cast<size_t>(i)];
        const int steps = std::max(0, delta);
        if (steps > 0) {
            if (person_height_m > 0.0f)
                predict_phys_delta(x, P, steps, person_height_m, accel_long, accel_lat, fps);
            else
                predict_delta(x, P, steps);
        }
        float R[16];
        saccade::kf_gpu::get_R(x[3], R);
        Eigen::Map<const Eigen::Matrix<float, 8, 8, Eigen::RowMajor>> Pm_full(P);
        Eigen::Matrix4f S = Pm_full.topLeftCorner<4, 4>();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                S(i, j) += R[i * 4 + j];
        const float bw = std::max(1e-6f, box.x2 - box.x1);
        const float bh = std::max(1e-6f, box.y2 - box.y1);
        Eigen::Vector4f z(
            (box.x1 + box.x2) * 0.5f, (box.y1 + box.y2) * 0.5f, bw / bh, bh);
        Eigen::Vector4f y = z - Eigen::Vector4f(x[0], x[1], x[2], x[3]);
        Eigen::LLT<Eigen::Matrix4f> llt(S);
        if (llt.info() != Eigen::Success) return 1e9f;
        Eigen::Vector4f w = llt.matrixL().solve(y);
        return w.squaredNorm();
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
    bool exp_density_gating_ = false;
    float exp_density_k_ = 2.0f;
    float exp_density_eta_ = 0.15f;
    int buffer_size_;
    float min_consistency_;
    std::string rerank_mode_;
    float reciprocal_margin_;
    [[maybe_unused]] bool debug_;
    float clean_score_threshold_;
    float clean_margin_ratio_;
    float clean_min_aspect_;
    float clean_max_aspect_;
    float strict_sim_threshold_;
    float w_sim_base_;
    float w_iou_base_;
    float w_maha_base_;
    float shift_ambiguity_;
    float shift_lost_age_;
    float iou_weight_;
    float mahalanobis_weight_;
    float dynamic_margin_crowd_;
    float dynamic_margin_age_;
    bool kalman_gate_;
    float kalman_chi2_;
    float kalman_penalty_weight_;
    float kalman_dir_min_cos_;
    float kalman_dir_min_speed_;
    float kalman_person_height_m_;
    float kalman_accel_long_;
    float kalman_accel_lat_;
    float kalman_fps_;
    float kalman_max_speed_mps_;
    bool delayed_claim_;
    int claim_warmup_frames_;
    bool bidirectional_;
    float bridge_px_;
    float bridge_h_lo_;
    float bridge_h_hi_;
    bool cheb_gr_claim_ = false;
    float cheb_gr_max_cost_ = 0.45f;
    float cheb_gr_margin_ = 0.05f;
    int cheb_gr_min_head_ = 2;
    float cheb_gr_pool_frac_ = 0.3f;
    float cheb_gr_min_sim_ = 0.0f;
    float cheb_gr_lambda_ = 2.0f;
    int cheb_gr_k2_ = 6;
    int cheb_gr_max_fwd_ = 50;
    float cheb_gr_fuse_lambda_ = 0.3f;
    int split_counter_;

    std::unordered_map<int, int> alias_;
    std::unordered_map<int, PendingClaim> pending_claims_;
    std::unordered_map<int, std::vector<std::vector<float>>> pending_heads_;
    std::unordered_map<int, int> deferred_alias_;
    std::unordered_map<int, std::vector<float>> features_;
    std::unordered_map<int, std::vector<std::vector<float>>> buffers_;
    std::unordered_map<int, int> last_seen_;
    std::unordered_map<int, RelinkBox> last_boxes_;
    std::unordered_map<int, RelinkMotionSnapshot> motion_;
    std::unordered_map<int, std::vector<std::pair<float, float>>> foot_history_;
    std::unordered_map<int, float> ema_h_;
    std::vector<int> feature_order_;
    RelinkStats stats_;
    std::vector<float> accept_sims_;
    std::vector<float> accept_ious_;
    std::vector<float> accept_center_dists_;
    std::vector<float> accept_mahas_;

    // GPU relink gate table: pre-computed per-pair gate quantities
    std::vector<float> gate_tbl_;
    int gate_n_query_ = 0;
    int gate_n_cand_ = 0;
    std::unordered_map<int, int> gate_row_;
    std::unordered_map<int, int> gate_col_;
    // GPU batch dot table: per-pair cosine similarities
    std::vector<float> sim_tbl_;
    // GPU scoring results: per-query best candidate
    std::vector<int> scoring_ids_;
    std::vector<float> scoring_scores_;
    std::vector<float> scoring_second_;

    // --- Online Cheb-GR handover ---
    bool ho_enabled_ = false;
    float ho_max_cost_ = 0.45f;
    float ho_margin_ = 0.0f;
    int ho_max_gap_ = 60;
    int ho_decide_n_ = 5;
    int ho_min_head_ = 1;
    float ho_pool_frac_ = 0.3f;
    float ho_cheb_lambda_ = 2.0f;
    int ho_k2_ = 6;
    int ho_max_fwd_ = 50;
    float ho_fuse_lambda_ = 0.3f;
    // Borderline re-query (default-off: ho_requery_band_ == 0). At a flippable
    // decision, dense recent-tail banks for the top candidates are re-extracted
    // from the crop store and the event is rescored, removing sparse-bank phase
    // jitter exactly where flips live. The store is keyed by the same id the
    // handover archive carries (ArchiveEntry::tid) — the stash side must match.
    ReidCropStore* ho_crop_store_ = nullptr;
    float ho_requery_band_ = 0.0f;
    int ho_requery_top_ = 0;

    struct DeadEntry {
        std::vector<float> embedding;           // last reference feature (fallback)
        std::vector<std::vector<float>> bank;   // full-life clean sample bank
        int death_frame;
        int canonical_label;
    };
    // Full-life clean sample bank (Python parity: bank_embs, n_samples=50).
    // Thinning keeps samples temporally distributed with bounded memory: once
    // 2*cap samples accumulate, every other one is dropped and the sampling
    // stride doubles.
    struct HoLifeBank {
        std::vector<std::vector<float>> samples;
        int stride = 1;
        int skip = 0;
        int last_frame = -1;
    };
    static constexpr int kHoBankCap = 50;
    std::unordered_map<int, DeadEntry> ho_dead_archive_;
    std::unordered_map<int, int> ho_track_birth_;  // raw_id → birth frame
    std::unordered_map<int, std::vector<std::vector<float>>> ho_newborn_heads_;
    std::unordered_map<int, int> ho_head_last_frame_;   // raw_id → last fed frame
    std::unordered_map<int, HoLifeBank> ho_life_bank_;  // canonical → bank
    std::unordered_map<int, int> ho_last_active_;       // tid → last emitted frame
    std::unordered_set<int> ho_decided_;                // raw ids with a consumed decision
    int ho_handover_count_ = 0;
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
        const float* data = arr.data();
        return {normalize_lc(std::vector<float>(data, data + static_cast<size_t>(arr.shape(0)))), true};
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
        const tracking::SpatialMetrics metrics = tracking::spatial_metrics(
            to_box4(box), to_box4(old_box), w, h
        );
        return {metrics.center_norm, metrics.iou};
    }

    bool enabled_;
    int ttl_, min_gap_;
    float spatial_gate_, min_iou_, sim_threshold_;
    bool require_embedding_;
    float ema_;
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
            std::vector<size_t> order(static_cast<size_t>(n));
            for (size_t i = 0; i < static_cast<size_t>(n); ++i) order[i] = i;
            if (relinker_->is_bidirectional() && n > 0) {
                std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return scores_v[a] > scores_v[b];
        });

            }
            std::unordered_set<int> assigned;
            for (size_t idx : order) {
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
        .def_readonly("track_uid", &TrackStateSnapshot::track_uid)
        .def_readonly("generation", &TrackStateSnapshot::generation)
        .def_readonly("class_id", &TrackStateSnapshot::class_id)
        .def_readonly("age", &TrackStateSnapshot::age)
        .def_readonly("score", &TrackStateSnapshot::score)
        .def_readonly("state", &TrackStateSnapshot::state)
        .def_readonly("covariance", &TrackStateSnapshot::covariance);

    py::class_<TrackCandidateSnapshot>(m, "TrackCandidateSnapshot")
        .def_readonly("obj_id", &TrackCandidateSnapshot::obj_id)
        .def_readonly("track_uid", &TrackCandidateSnapshot::track_uid)
        .def_readonly("generation", &TrackCandidateSnapshot::generation)
        .def_readonly("class_id", &TrackCandidateSnapshot::class_id)
        .def_readonly("age", &TrackCandidateSnapshot::age)
        .def_readonly("hit_streak", &TrackCandidateSnapshot::hit_streak)
        .def_readonly("required_confirm_streak", &TrackCandidateSnapshot::required_confirm_streak)
        .def_readonly("score", &TrackCandidateSnapshot::score)
        .def_readonly("x1", &TrackCandidateSnapshot::x1)
        .def_readonly("y1", &TrackCandidateSnapshot::y1)
        .def_readonly("x2", &TrackCandidateSnapshot::x2)
        .def_readonly("y2", &TrackCandidateSnapshot::y2);

    py::class_<UnifiedScoreParams>(m, "UnifiedScoreParams")
        .def(py::init<>())
        .def_readwrite("w_sim_base", &UnifiedScoreParams::w_sim_base)
        .def_readwrite("w_iou_base", &UnifiedScoreParams::w_iou_base)
        .def_readwrite("w_maha_base", &UnifiedScoreParams::w_maha_base)
        .def_readwrite("shift_ambiguity", &UnifiedScoreParams::shift_ambiguity)
        .def_readwrite("shift_lost_age", &UnifiedScoreParams::shift_lost_age);

    py::class_<GPUByteTracker>(m, "GPUByteTracker")
        .def(py::init<int, int, int>(),
             py::arg("max_objects") = 2048,
             py::arg("embedding_dim") = 768,
             py::arg("max_assoc") = 1024)
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
             py::arg("kalman_adapt_mode") = 0,
             py::arg("r_scale") = 1.0f,
             py::arg("vel_dir_weight") = 0.0f,
             py::arg("fuse_score_weight") = 0.0f,
             py::arg("stage2_match_thresh") = 0.5f,
             py::arg("birth_low_score_thresh") = 0.0f,
             py::arg("birth_prox_norm_thresh") = 0.0f)
        .def("set_reid_params", &GPUByteTracker::set_reid_params,
             py::arg("cos_threshold"), py::arg("iou_low"), py::arg("iou_high"), py::arg("weight"),
             py::arg("cost_cos_w") = 0.55f, py::arg("cost_iou_w") = 0.30f, py::arg("cost_score_w") = 0.15f)
        .def("set_relink_params", &GPUByteTracker::set_relink_params,
             py::arg("enabled"), py::arg("bank_cap") = 256, py::arg("sim_thresh") = 0.6f,
             py::arg("cheb_lambda") = 2.5f, py::arg("spatial_gate") = 4.0f, py::arg("max_age") = 300,
             py::arg("bidirectional") = false, py::arg("bridge_px") = 0.25f,
             py::arg("bridge_at") = 4, py::arg("bridge_min_lost") = 2, py::arg("bridge_ttl") = 120,
             py::arg("bridge_max_speed") = 0.0f, py::arg("bridge_person_height") = 1.65f,
             py::arg("bridge_fps") = 30.0f, py::arg("bridge_margin") = 0.0f,
             py::arg("bridge_spatial_gate") = 0.0f, py::arg("bridge_anchor") = 0,
             py::arg("bridge_anchor_rate") = 0.0f,
              py::arg("bridge_h_lo") = 0.0f, py::arg("bridge_h_hi") = 0.0f,
              py::arg("bridge_dir_bonus") = 0.0f,
             py::arg("occ_gate_cover") = 0.0f, py::arg("occ_gap_min") = 30,
             py::arg("occ_expand_px") = 0.0f, py::arg("occ_expand_cover") = 0.9f,
             py::arg("bridge_app_veto") = -1.0f,
             "Birth-time lost-bank ReID relink: revive a lost identity at spawn instead "
             "of minting a new id. Precision-first (high sim threshold + spatial gate). "
             "The bridge_* args enable the Phase-4 Kalman-free bidirectional foot-bridge "
             "(adopt a still-live lost id when a young track's foot path bridges to it).")
        .def("get_relink_debug", &GPUByteTracker::get_relink_debug,
             "Returns (archived, births, revived, bridge_attempts, bridge_accepts) counters.")
        .def("set_oao_params", &GPUByteTracker::set_oao_params,
             py::arg("tau"), py::arg("contest_thresh") = -1.0f, py::arg("score_w") = -1.0f,
             py::arg("occ_mode") = 0, py::arg("crowd_radius") = 0.0f, py::arg("height_gate") = 0.0f,
             py::arg("foot_gate") = 0.0f, py::arg("ramp_frames") = 0.0f,
             "OA-SORT OAO penalty weight [0, 1]. 0 = disabled. "
             "Tracks occluded by other tracks receive cost += tau * overlap_iou. "
             "contest_thresh < 0 = plain OAO; >= 0 = only penalise detections also "
             "claimed by t's max-overlap partner (partner-pred IoU >= thresh). "
             "score_w <= 0 = off; > 0 = scale the penalty by (1 - score_w * det_score) "
             "so confident detections get a reduced (not cut) penalty. "
             "occ_mode 0 = max single inter-track IoU (default); 1 = union coverage "
             "(fraction of the track covered by the union of other boxes). "
             "crowd_radius <= 0 = off; > 0 = scale penalty by (1 - 1/N), N = tracks "
             "within crowd_radius*h of t (incl. self): sparse overlaps damped, crowds full. "
             "height_gate <= 0 = off; > 0 = only same-depth partners (|h_t-h_j| <= "
             "gate*max(h)) contribute, sparing near/far projection overlaps. "
             "foot_gate <= 0 = off; > 0 = only same-foot-line partners (|footy_t-"
             "footy_j| <= gate*h_ref) contribute (truer depth proxy than height). "
             "ramp_frames <= 0 = off; > 0 = scale penalty by min(1, overlap_frames/"
             "ramp_frames): transient crossings damped, persistent crowds reach full.")
        .def("set_occ_params", &GPUByteTracker::set_occ_params,
             py::arg("enabled") = true, py::arg("iou_thresh") = 0.45f,
             py::arg("foot_gap") = 0.15f, py::arg("ttl") = 4,
             py::arg("cost_weight") = 0.50f,
             "Depth-gated occlusion-state machine (default off → bit-identical). "
             "Holds an occludee behind its occluder and depth-biases re-acquisition "
             "to resolve occlusion crossing-swaps.")
        .def("get_occ_front_info", &GPUByteTracker::get_occ_front_info,
             "Read back front-ttl and partner-slot arrays (env-gated diagnostic). "
             "Returns [max_objs] ttl values followed by [max_objs] partner slots.")
        .def("set_quality_params", &GPUByteTracker::set_quality_params,
             py::arg("enabled"), py::arg("w_aspect") = 0.50f, py::arg("w_center") = 0.30f, py::arg("w_area") = 0.20f)
        .def("set_multiplicative_cost", &GPUByteTracker::set_multiplicative_cost,
             py::arg("enabled"),
             "Enable log-linear cost: cost = 1 - IoU * exp(-Σ penalty).")
        .def("set_stability_cost_w", &GPUByteTracker::set_stability_cost_w,
             py::arg("w"),
             "Stability reward weight for multiplicative cost form.")
        .def("set_association_energy_params",
             &GPUByteTracker::set_association_energy_params,
             py::arg("enabled"), py::arg("score_cost_w") = 0.0f,
             py::arg("height_cost_w") = 0.0f,
             "Optional score and height-mismatch energy terms for association scoring.")
        .def("set_sinkhorn_lambda", &GPUByteTracker::set_sinkhorn_lambda,
             py::arg("lambda"),
             "Sinkhorn exponential lambda (cost→prob, default 30).")
        .def("set_frame_size", &GPUByteTracker::set_frame_size,
             py::arg("w"), py::arg("h"))
        .def("set_homography", [](GPUByteTracker& self, py::object h_obj) {
            if (h_obj.is_none()) {
                self.set_homography(nullptr);
            } else {
                py::array_t<float, py::array::c_style | py::array::forcecast> h(h_obj);
                if (h.size() != 9) throw std::invalid_argument("Homography must have 9 elements");
                self.set_homography(h.data());
            }
        }, py::arg("h"))
        .def("set_unified_score_params", [](GPUByteTracker& self, float w_sim_base, float w_iou_base, float w_maha_base, float shift_ambiguity, float shift_lost_age) {
            UnifiedScoreParams p;
            p.w_sim_base = w_sim_base;
            p.w_iou_base = w_iou_base;
            p.w_maha_base = w_maha_base;
            p.shift_ambiguity = shift_ambiguity;
            p.shift_lost_age = shift_lost_age;
            self.set_unified_score_params(p);
        }, py::arg("w_sim_base"), py::arg("w_iou_base"), py::arg("w_maha_base"), py::arg("shift_ambiguity"), py::arg("shift_lost_age"))
        .def("set_unified_score_params", &GPUByteTracker::set_unified_score_params, py::arg("params"))
        .def("update_reference_features", [](GPUByteTracker& self, uintptr_t ids_ptr, uintptr_t features_ptr, int num, uintptr_t stream_ptr) {
            self.update_reference_features(
                reinterpret_cast<int*>(ids_ptr),
                reinterpret_cast<float*>(features_ptr),
                num,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        }, py::arg("ids_ptr"), py::arg("features_ptr"), py::arg("num"), py::arg("stream_ptr"))
        .def("bind_features_buffer", [](GPUByteTracker& self, uintptr_t ptr) {
            self.bind_features_buffer(reinterpret_cast<float*>(ptr));
        }, py::arg("ptr"),
        "Bind an externally-owned CUDA float buffer (shape [max_objects, embed_dim]) as d_features_. "
        "Python owns the lifetime; C++ will not free it.")
        .def("get_active_tid_slot_pairs", [](GPUByteTracker& self) {
            return self.get_active_tid_slot_pairs();
        }, "Return list of (track_id, slot_index) for all active tracks. "
           "Shares the lazy-sync cache with update_reference_features / set_clean_embedding_flags.")
        .def("get_gpu_buffers", [](GPUByteTracker& self) {
            auto buf = self.get_gpu_buffers();
            return py::make_tuple(buf.states, buf.covs, buf.track_ids, buf.track_uids, buf.max_objs);
        }, "Return (states_ptr, covs_ptr, track_ids_ptr, track_uids_ptr, max_objs) device pointers.")
        .def_property_readonly("max_objects", &GPUByteTracker::max_objects)
        .def_property_readonly("max_assoc", &GPUByteTracker::max_assoc)
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
                               uintptr_t embeddings_ptr, uintptr_t gmc_ptr, float light_factor, float mid_thresh_scale,
                               int out_capacity) {
            // All args are raw pointers / primitives — no Python objects accessed.
            // Releasing the GIL lets sibling worker threads make Python progress
            // while this tracker's C++ work (1–3 ms/frame) runs on its stream.
            py::gil_scoped_release release;
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
                mid_thresh_scale,
                out_capacity
            );
        },
        py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"), py::arg("num_dets"), py::arg("stream_ptr"),
        py::arg("out_boxes_ptr"), py::arg("out_scores_ptr"), py::arg("out_ids_ptr"), py::arg("out_classes_ptr"),
        py::arg("out_det_idx_ptr"), py::arg("out_count_ptr"),
        py::arg("embeddings_ptr") = 0, py::arg("gmc_ptr") = 0,
        py::arg("light_factor") = 0.0f, py::arg("mid_thresh_scale") = 1.0f,
        py::arg("out_capacity") = -1,
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
        "Sync per-track clean-embedding flags from Python bank state to the CUDA tracker")
        .def("set_clean_embedding_flags_host", [](GPUByteTracker& self,
                                                   uintptr_t ids_ptr, uintptr_t flags_ptr,
                                                   int n, uintptr_t stream_ptr) {
            self.set_clean_embedding_flags_host(
                reinterpret_cast<int*>(ids_ptr),
                reinterpret_cast<bool*>(flags_ptr),
                n,
                reinterpret_cast<cudaStream_t>(stream_ptr)
            );
        },
        py::arg("ids_ptr"), py::arg("flags_ptr"), py::arg("n"), py::arg("stream_ptr"),
        "Same as set_clean_embedding_flags but takes host (CPU) pointers — skips the D2H round-trip for IDs")
        .def_property_readonly("cpp_ptr", [](GPUByteTracker& self) {
            return reinterpret_cast<uintptr_t>(&self);
        }, "Raw C++ pointer to this GPUByteTracker (for Workbench construction)")
        .def("compact_output_to_host",
             [](GPUByteTracker& self, uintptr_t stream_ptr,
                int capacity) -> py::tuple {
                 std::vector<float> boxes(capacity * 4);
                 std::vector<float> scores(capacity);
                 std::vector<int> ids(capacity);
                 std::vector<int> classes(capacity);
                 int n = self.compact_output_to_host(
                     boxes.data(), scores.data(), ids.data(), classes.data(),
                     capacity, reinterpret_cast<cudaStream_t>(stream_ptr));
                 boxes.resize(n * 4);
                 scores.resize(n);
                 ids.resize(n);
                 classes.resize(n);
                 return py::make_tuple(n, boxes, scores, ids, classes);
             },
             py::arg("stream_ptr"), py::arg("capacity") = -1,
             "Read compact tracker output directly into host memory in one batch. "
             "Returns (count, boxes[n*4], scores[n], ids[n], classes[n]).");

    py::class_<SemanticRelinkerCpp>(m, "SemanticRelinker")
        .def(py::init<float, int, float, float, int, float, float, int, float, std::string, float, bool, float, float, float, float, float, float, float, float, float, float, float, float, float, float, bool, float, float, float, float, float, float, float, float, float, bool, int, bool, float, float, float, bool, float, float, bool, float, float, int, float, float, float, int, int, float>(),
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
             py::arg("debug") = false,
             py::arg("clean_score_threshold") = 0.0f,
             py::arg("clean_margin_ratio") = 0.0f,
             py::arg("clean_min_aspect") = 0.0f,
             py::arg("clean_max_aspect") = 99.0f,
             py::arg("strict_sim_threshold") = 0.0f,
             py::arg("w_sim_base") = 0.0f,
             py::arg("w_iou_base") = 0.0f,
             py::arg("w_maha_base") = 0.0f,
             py::arg("shift_ambiguity") = 0.0f,
             py::arg("shift_lost_age") = 0.0f,
             py::arg("iou_weight") = 0.0f,
             py::arg("mahalanobis_weight") = 0.0f,
             py::arg("dynamic_margin_crowd") = 0.0f,
             py::arg("dynamic_margin_age") = 0.0f,
             py::arg("kalman_gate") = false,
             py::arg("kalman_chi2") = 9.4877f,
             py::arg("kalman_penalty_weight") = 0.0f,
             py::arg("kalman_dir_min_cos") = -1.0f,
             py::arg("kalman_dir_min_speed") = 1.0f,
             py::arg("kalman_person_height_m") = 0.0f,
             py::arg("kalman_accel_long") = 2.0f,
             py::arg("kalman_accel_lat") = 1.0f,
             py::arg("kalman_fps") = 30.0f,
             py::arg("kalman_max_speed_mps") = 0.0f,
             py::arg("delayed_claim") = false,
             py::arg("claim_warmup_frames") = 3,
             py::arg("bidirectional") = false,
             py::arg("bridge_px") = 1.5f,
             py::arg("bridge_h_lo") = 0.0f,
             py::arg("bridge_h_hi") = 0.0f,
             py::arg("exp_density_gating") = false,
             py::arg("exp_density_k") = 2.0f,
              py::arg("exp_density_eta") = 0.15f,
              py::arg("cheb_gr_claim") = false,
              py::arg("cheb_gr_max_cost") = 0.45f,
              py::arg("cheb_gr_margin") = 0.05f,
              py::arg("cheb_gr_min_head") = 2,
              py::arg("cheb_gr_pool_frac") = 0.3f,
              py::arg("cheb_gr_min_sim") = 0.0f,
              py::arg("cheb_gr_lambda") = 2.0f,
              py::arg("cheb_gr_k2") = 6,
              py::arg("cheb_gr_max_fwd") = 50,
              py::arg("cheb_gr_fuse_lambda") = 0.3f)
        .def("update_motion_snapshots", &SemanticRelinkerCpp::update_motion_snapshots,
             py::arg("snapshots"), py::arg("frame_id") = -1)
        .def("motion_candidate_ids", &SemanticRelinkerCpp::motion_candidate_ids, py::arg("frame_id") = -1)
        .def("inject_reference", &SemanticRelinkerCpp::inject_reference,
             py::arg("canonical_id"), py::arg("embedding"))
        .def("inject_references_many", &SemanticRelinkerCpp::inject_references_many,
             py::arg("references"))
        .def("inject_references_batch", &SemanticRelinkerCpp::inject_references_batch,
             py::arg("canonical_ids"), py::arg("embeddings_tensor"),
             "Batch-inject N reference embeddings from a single [N, D] CPU tensor")
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
        .def_property_readonly("deferred_alias", &SemanticRelinkerCpp::get_deferred_alias)
        .def_property_readonly("features", &SemanticRelinkerCpp::get_features)
        .def_property_readonly("stats", &SemanticRelinkerCpp::stats)
        .def("report", &SemanticRelinkerCpp::report)
        .def("build_gate_table", &SemanticRelinkerCpp::build_gate_table,
             py::arg("raw_ids"), py::arg("boxes"), py::arg("frame_id"), py::arg("w"), py::arg("h"),
             py::arg("tracker_states") = 0, py::arg("tracker_covs") = 0,
             py::arg("tracker_tids") = 0, py::arg("tracker_max_objs") = 0,
             py::arg("query_embs") = py::none(),
             "Build GPU-accelerated per-pair gate table for this frame. "
             "When tracker GPU pointers are provided, candidate Kalman state is "
             "copied device-to-device instead of H2D. "
             "When query_embs is a [N, D] tensor, batch cosine similarity is "
             "computed on GPU and used to skip CPU dot products.")
        .def("clear_gate_table", &SemanticRelinkerCpp::clear_gate_table,
             "Release the GPU gate table, reverting to inline gate computation.")
        .def("resolve_batch_from_host",
             &SemanticRelinkerCpp::resolve_batch_from_host,
             py::arg("n_tracks"), py::arg("boxes"), py::arg("scores"),
             py::arg("ids"), py::arg("embeddings"), py::arg("embedding_dim"),
             py::arg("frame_id"), py::arg("frame_w"), py::arg("frame_h"),
             "Resolve all tracker output IDs in-place from host-side data.")
        .def("set_handover_params", &SemanticRelinkerCpp::set_handover_params,
             py::arg("enabled"), py::arg("max_cost") = 0.45f,
             py::arg("margin") = 0.0f, py::arg("max_gap") = 60,
             py::arg("decide_n") = 5, py::arg("min_head") = 1,
             py::arg("pool_frac") = 0.3f, py::arg("cheb_lambda") = 2.0f,
             py::arg("k2") = 6, py::arg("max_fwd") = 50,
             py::arg("fuse_lambda") = 0.3f,
             "Configure online Cheb-GR handover parameters.")
        .def("set_crop_store",
             [](SemanticRelinkerCpp& self, PerceptionPipeline& pipe) {
                 self.set_crop_store(&pipe);
             },
             py::arg("pipeline"),
             "Attach a PerceptionPipeline crop ring as the borderline re-query "
             "store (its ReidCropStore interface).")
        .def("clear_crop_store",
             [](SemanticRelinkerCpp& self) { self.set_crop_store(nullptr); },
             "Detach the borderline re-query store.")
        .def("set_handover_requery", &SemanticRelinkerCpp::set_handover_requery,
             py::arg("band") = 0.0f, py::arg("top") = 0,
             "Configure borderline re-query (band=0 disables).")
        .def("prune_and_archive", &SemanticRelinkerCpp::prune_and_archive,
             py::arg("active_ids"), py::arg("frame_id"),
             "Archive dead tracks and prune inactive features.")
        .def_property_readonly("handover_count",
             &SemanticRelinkerCpp::handover_count)
        .def("feed_frame_embeddings",
             &SemanticRelinkerCpp::feed_frame_embeddings,
             py::arg("emb_flat"), py::arg("emb_dim"), py::arg("frame_id"),
             py::arg("track_ids"), py::arg("scores"),
             py::arg("clean_flags") = std::vector<int>{},
             "Feed per-frame track embeddings for online handover. "
             "clean_flags[i]=0 marks an occluded crop (skipped); zero "
             "embedding rows are skipped automatically.");

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

    py::class_<saccade::ReIDTrackObservation>(m, "ReIDTrackObservation")
        .def(py::init<>())
        .def(py::init<float, float, float, float, float>(),
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
             py::arg("det_score"))
        .def_readwrite("x1", &saccade::ReIDTrackObservation::x1)
        .def_readwrite("y1", &saccade::ReIDTrackObservation::y1)
        .def_readwrite("x2", &saccade::ReIDTrackObservation::x2)
        .def_readwrite("y2", &saccade::ReIDTrackObservation::y2)
        .def_readwrite("det_score", &saccade::ReIDTrackObservation::det_score);

    py::class_<saccade::DynamicReIDController>(m, "DynamicReIDController")
        .def(py::init<int, std::string, float, float, int, float, float, float, float, float, float, float, float, float, float, float, int, float, float, float, int, int>(),
             py::arg("history_size") = 5,
             py::arg("mode") = "event_any",
             py::arg("unstable_iou") = 0.50f,
             py::arg("unstable_center_shift") = 0.30f,
             py::arg("crowd_threshold") = 8,
             py::arg("long_memory_decay") = 0.80f,
             py::arg("long_memory_trigger") = 1.25f,
             py::arg("score_decay") = 0.80f,
             py::arg("score_threshold") = 2.0f,
             py::arg("score_threshold_low") = 0.0f,
             py::arg("weight_new") = 1.0f,
             py::arg("weight_lost") = 1.4f,
             py::arg("weight_geom") = 0.5f,
             py::arg("weight_conf") = 0.5f,
             py::arg("birth_death_boost") = 1.0f,
             py::arg("birth_death_lost_min") = 0.0f,
             py::arg("lost_age_cap") = 30,
             py::arg("unstable_shift_weight") = 1.0f,
             py::arg("unstable_iou_weight") = 1.0f,
             py::arg("conf_jitter_gate") = 0.10f,
             py::arg("trigger_persist_frames") = 1,
             py::arg("cooldown_frames") = 0)
        .def("observe",
             [](saccade::DynamicReIDController& self,
                const std::unordered_map<int, saccade::ReIDTrackObservation>& tracks,
                const std::vector<float>& gmc) {
                 self.observe(tracks, gmc);
             },
             py::arg("tracks"), py::arg("gmc") = std::vector<float>{})
        .def("should_reid", &saccade::DynamicReIDController::should_reid,
             py::arg("det_count"))
        .def("get_priorities", &saccade::DynamicReIDController::get_priorities);

    py::class_<GMC>(m, "GMC")
        .def(py::init<int, int, float, float, int, float>(),
             py::arg("downscale") = 8,
             py::arg("max_corners") = 100,
             py::arg("quality_level") = 0.01f,
             py::arg("min_distance") = 10.0f,
             py::arg("min_inliers") = 8,
             py::arg("ransac_threshold") = 3.0f)
        .def("estimate", [](GMC& self, uintptr_t frame_ptr, int width, int height, uintptr_t stream_ptr, bool use_gpu_phase_corr) {
            auto warp = self.estimate(reinterpret_cast<const float*>(frame_ptr), width, height, reinterpret_cast<cudaStream_t>(stream_ptr), use_gpu_phase_corr);
            if (warp.empty()) return py::none().cast<py::object>();
            return py::cast(warp);
        }, py::arg("frame_ptr"), py::arg("width"), py::arg("height"), py::arg("stream_ptr"), py::arg("use_gpu_phase_corr") = true)
        .def("estimate_into", [](GMC& self, uintptr_t frame_ptr, int width, int height,
                                  uintptr_t stream_ptr, uintptr_t out_warp_ptr,
                                  bool use_gpu_phase_corr, bool sync_caller_stream) {
            py::gil_scoped_release release;
            self.estimate_into(
                reinterpret_cast<const float*>(frame_ptr),
                width, height,
                reinterpret_cast<cudaStream_t>(stream_ptr),
                reinterpret_cast<float*>(out_warp_ptr),
                use_gpu_phase_corr, sync_caller_stream);
        }, py::arg("frame_ptr"), py::arg("width"), py::arg("height"),
           py::arg("stream_ptr"), py::arg("out_warp_ptr"),
           py::arg("use_gpu_phase_corr") = true, py::arg("sync_caller_stream") = true)
        .def("estimate_into_direct", [](GMC& self, uintptr_t frame_ptr, int width, int height,
                                         uintptr_t stream_ptr, uintptr_t out_warp_ptr) {
            py::gil_scoped_release release;
            self.estimate_into_direct(
                reinterpret_cast<const float*>(frame_ptr),
                width, height,
                reinterpret_cast<cudaStream_t>(stream_ptr),
                reinterpret_cast<float*>(out_warp_ptr));
        }, py::arg("frame_ptr"), py::arg("width"), py::arg("height"),
           py::arg("stream_ptr"), py::arg("out_warp_ptr"))
        .def("sync_to_stream", [](GMC& self, uintptr_t stream_ptr) {
            self.sync_to_stream(reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("stream_ptr"))
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
        .def("set_profiling_enabled", &GMC::set_profiling_enabled, py::arg("enabled"))
        .def("reset_profile_stats", &GMC::reset_profile_stats)
        .def("get_profile_stats",
            [](const GMC& self) {
                const auto stats = self.get_profile_stats();
                py::dict out;
                out["gray_downscale_ms"] = stats.gray_downscale_ms;
                out["fg_mask_ms"] = stats.fg_mask_ms;
                out["phase_corr_ms"] = stats.phase_corr_ms;
                out["fft_ms"] = stats.fft_ms;
                out["cross_power_ms"] = stats.cross_power_ms;
                out["ifft_ms"] = stats.ifft_ms;
                out["peak_find_ms"] = stats.peak_find_ms;
                out["handoff_ms"] = stats.handoff_ms;
                out["total_ms"] = stats.total_ms;
                out["frames"] = stats.frames;
                return out;
            })
        .def("reset", &GMC::reset)
        .def("pcr_score", &GMC::pcr_score,
             "PCR (peak-to-RMS ratio) from the most recent GPU phase correlation. "
             "0.0 before first estimate; low values indicate unreliable motion estimate.")
        .def("set_fg_mask_boxes", [](GMC& self, py::list boxes_flat) {
            std::vector<float> v;
            v.reserve(py::len(boxes_flat));
            for (auto item : boxes_flat) v.push_back(item.cast<float>());
            self.set_fg_mask_boxes(v);
        }, py::arg("boxes_flat"),
           "Set foreground boxes [x1,y1,x2,y2,...] to zero before phase correlation. "
           "Call once per frame before estimate().")
        .def("set_fg_mask_boxes_gpu", [](GMC& self, uintptr_t boxes_ptr, int n_boxes, uintptr_t stream_ptr) {
            py::gil_scoped_release release;
            self.set_fg_mask_boxes_gpu(
                reinterpret_cast<const float*>(boxes_ptr),
                n_boxes,
                reinterpret_cast<cudaStream_t>(stream_ptr));
        }, py::arg("boxes_ptr"), py::arg("n_boxes"), py::arg("stream_ptr") = 0,
           "Set foreground boxes from GPU memory. No D2H roundtrip.");

    m.def(
        "merge_cross_tile_duplicates",
        &merge_cross_tile_duplicates_cpu,
        py::arg("boxes"),
        py::arg("scores"),
        py::arg("classes"),
        py::arg("iou_threshold") = 0.45f,
        py::arg("center_threshold") = 0.18f,
        py::arg("area_ratio_threshold") = 0.6f,
        py::arg("tiling_mode") = 0,
        py::arg("frame_w") = 0,
        py::arg("frame_h") = 0,
        py::arg("seam_margin_canvas_px") = 24.0f,
        py::arg("seam_center_scale") = 1.8f,
        py::arg("seam_area_ratio_threshold") = 0.30f,
        py::arg("seam_min_overlap_ratio") = 0.45f,
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

    m.def("emit_tracks_unified",
          [](GPUByteTracker& tracker,
             py::object relinker_obj,
             uintptr_t stream_ptr, int frame_id, int frame_w, int frame_h,
             int capacity,
             py::object embeddings_obj) -> py::tuple {
              int count, emb_dim = 0;
              std::vector<float> emb_data;
              std::vector<float> boxes(capacity * 4);
              std::vector<float> scores(capacity);
              std::vector<int> ids(capacity);
              std::vector<int> classes(capacity);

              if (!embeddings_obj.is_none()) {
                  py::array arr = embeddings_obj.attr("detach")().attr("cpu")().attr("numpy")();
                  py::array_t<float, py::array::c_style> arr_flat(arr.attr("ravel")());
                  emb_data.assign(arr_flat.data(), arr_flat.data() + arr_flat.size());
              }

              count = tracker.compact_output_to_host(
                  boxes.data(), scores.data(), ids.data(), classes.data(),
                  capacity, reinterpret_cast<cudaStream_t>(stream_ptr));

              if (count > 0 && !emb_data.empty()) {
                  emb_dim = static_cast<int>(emb_data.size()) / count;
              }
              if (count > 0 && !relinker_obj.is_none()) {
                  auto* relinker = relinker_obj.cast<SemanticRelinkerCpp*>();
                  relinker->resolve_batch_from_host(
                      count, boxes.data(), scores.data(), ids.data(),
                      emb_data.empty() ? nullptr : emb_data.data(),
                      emb_dim, frame_id, frame_w, frame_h);
              }
              boxes.resize(std::max(0, count) * 4);
              scores.resize(count);
              ids.resize(count);
              classes.resize(count);
              return py::make_tuple(count, boxes, scores, ids, classes);
          },
          py::arg("tracker"), py::arg("relinker") = py::none(),
          py::arg("stream_ptr"), py::arg("frame_id"),
          py::arg("frame_w"), py::arg("frame_h"),
          py::arg("capacity") = -1,
          py::arg("embeddings") = py::none(),
          "Unified emit: D2H + resolve canonical IDs in one call.");

    m.def(
        "filter_detections_cuda",
        [](
            uintptr_t boxes_ptr,
            uintptr_t scores_ptr,
            uintptr_t classes_ptr,
            int num_dets,
            uintptr_t keep_indices_ptr,
            uintptr_t suspect_flags_ptr,
            uintptr_t quality_scores_ptr,
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
                reinterpret_cast<float*>(quality_scores_ptr),
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
        py::arg("quality_scores_ptr"),
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
            uintptr_t best_boxes_ptr,
            uintptr_t best_key_bits_ptr,
            uintptr_t cluster_counts_ptr,
            uintptr_t out_boxes_ptr,
            uintptr_t out_scores_ptr,
            uintptr_t out_classes_ptr,
            uintptr_t out_count_ptr,
            float iou_threshold,
            float center_threshold,
            float area_ratio_threshold,
            int tiling_mode,
            int frame_w,
            int frame_h,
            float seam_margin_canvas_px,
            float seam_center_scale,
            float seam_area_ratio_threshold,
            float seam_min_overlap_ratio,
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
                reinterpret_cast<float*>(best_boxes_ptr),
                reinterpret_cast<int*>(best_key_bits_ptr),
                reinterpret_cast<int*>(cluster_counts_ptr),
                reinterpret_cast<float*>(out_boxes_ptr),
                reinterpret_cast<float*>(out_scores_ptr),
                reinterpret_cast<int*>(out_classes_ptr),
                reinterpret_cast<int*>(out_count_ptr),
                iou_threshold,
                center_threshold,
                area_ratio_threshold,
                tiling_mode,
                frame_w,
                frame_h,
                seam_margin_canvas_px,
                seam_center_scale,
                seam_area_ratio_threshold,
                seam_min_overlap_ratio,
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
        py::arg("best_boxes_ptr"),
        py::arg("best_key_bits_ptr"),
        py::arg("cluster_counts_ptr"),
        py::arg("out_boxes_ptr"),
        py::arg("out_scores_ptr"),
        py::arg("out_classes_ptr"),
        py::arg("out_count_ptr"),
        py::arg("iou_threshold") = 0.45f,
        py::arg("center_threshold") = 0.18f,
        py::arg("area_ratio_threshold") = 0.6f,
        py::arg("tiling_mode") = 0,
        py::arg("frame_w") = 0,
        py::arg("frame_h") = 0,
        py::arg("seam_margin_canvas_px") = 24.0f,
        py::arg("seam_center_scale") = 1.8f,
        py::arg("seam_area_ratio_threshold") = 0.30f,
        py::arg("seam_min_overlap_ratio") = 0.45f,
        py::arg("stream_ptr"),
        "Merge duplicate detections across overlapping tiles on the caller's CUDA stream."
    );

    // PerceptionPipeline — C++ facade for filter+NMS+reid hot path
    py::class_<ReIDQueue>(m, "ReIDQueue")
        .def(py::init<>())
        .def("size", &ReIDQueue::size)
        .def("shutdown", &ReIDQueue::shutdown);

    py::class_<PerceptionPipeline::Config>(m, "PerceptionPipelineConfig")
        .def(py::init<>())
        .def_readwrite("score_threshold",           &PerceptionPipeline::Config::score_threshold)
        .def_readwrite("person_class",              &PerceptionPipeline::Config::person_class)
        .def_readwrite("person_only",               &PerceptionPipeline::Config::person_only)
        .def_readwrite("nms_threshold",             &PerceptionPipeline::Config::nms_threshold)
        .def_readwrite("person_geometry_prior",     &PerceptionPipeline::Config::person_geometry_prior)
        .def_readwrite("geometry_suspect_support",  &PerceptionPipeline::Config::geometry_suspect_support)
        .def_readwrite("geometry_suspect_support_score", &PerceptionPipeline::Config::geometry_suspect_support_score)
        .def_readwrite("person_min_height_ratio",   &PerceptionPipeline::Config::person_min_height_ratio)
        .def_readwrite("person_min_aspect",         &PerceptionPipeline::Config::person_min_aspect)
        .def_readwrite("person_max_aspect",         &PerceptionPipeline::Config::person_max_aspect)
        .def_readwrite("person_min_area_ratio",     &PerceptionPipeline::Config::person_min_area_ratio)
        .def_readwrite("person_max_area_ratio",     &PerceptionPipeline::Config::person_max_area_ratio)
        .def_readwrite("max_detections",            &PerceptionPipeline::Config::max_detections)
        .def_readwrite("private_continuation_enabled", &PerceptionPipeline::Config::private_continuation_enabled)
        .def_readwrite("private_candidate_nms_iou", &PerceptionPipeline::Config::private_candidate_nms_iou)
        .def_readwrite("private_min_score",         &PerceptionPipeline::Config::private_min_score)
        .def_readwrite("private_max_candidates",    &PerceptionPipeline::Config::private_max_candidates)
        .def_readwrite("private_prior_iou_threshold", &PerceptionPipeline::Config::private_prior_iou_threshold)
        .def_readwrite("private_prior_center_threshold", &PerceptionPipeline::Config::private_prior_center_threshold)
        .def_readwrite("private_low_stage_only",    &PerceptionPipeline::Config::private_low_stage_only)
        .def_readwrite("private_track_thresh",      &PerceptionPipeline::Config::private_track_thresh)
        .def_readwrite("private_mid_thresh",        &PerceptionPipeline::Config::private_mid_thresh)
        .def_readwrite("private_new_track_thresh",  &PerceptionPipeline::Config::private_new_track_thresh)
        .def_readwrite("private_score_eps",         &PerceptionPipeline::Config::private_score_eps);

    py::class_<PerceptionPipeline>(m, "PerceptionPipeline")
        .def(py::init([](uintptr_t reid_ptr, uintptr_t cropper_ptr,
                         const PerceptionPipeline::Config& cfg) {
            return new PerceptionPipeline(
                reinterpret_cast<FeatureExtractor*>(reid_ptr),
                reinterpret_cast<Cropper*>(cropper_ptr),
                cfg);
        }), py::arg("reid_ptr"), py::arg("cropper_ptr"), py::arg("config"))
        .def("enable_crop_ring", &PerceptionPipeline::enable_crop_ring,
             py::arg("capacity"), py::arg("depth"),
             "Enable the per-track_uid recent-crop ring for borderline re-query.")
        .def("crop_ring_enabled", &PerceptionPipeline::crop_ring_enabled)
        .def("ring_depth", &PerceptionPipeline::ring_depth)
        .def("stash_crops",
            [](PerceptionPipeline& self,
               uintptr_t uids_ptr, uintptr_t frames_ptr,
               uintptr_t frame_ptr, int frame_h, int frame_w,
               uintptr_t boxes_ptr, int n_boxes,
               uintptr_t clean_ptr, uintptr_t stream_ptr) {
                self.stash_crops(
                    reinterpret_cast<const uint64_t*>(uids_ptr),
                    reinterpret_cast<const int*>(frames_ptr),
                    reinterpret_cast<const float*>(frame_ptr),
                    frame_h, frame_w,
                    reinterpret_cast<const float*>(boxes_ptr), n_boxes,
                    reinterpret_cast<const bool*>(clean_ptr),
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("uids_ptr"), py::arg("frames_ptr"),
            py::arg("frame_ptr"), py::arg("frame_h"), py::arg("frame_w"),
            py::arg("boxes_ptr"), py::arg("n_boxes"),
            py::arg("clean_ptr"), py::arg("stream_ptr"),
            "Stash n_boxes crops keyed by track_uid (clean_ptr may be 0).")
        .def("evict_crop_uid",
            [](PerceptionPipeline& self, uint64_t uid) { self.evict(uid); },
            py::arg("uid"), "Drop a track_uid's stashed crops.")
        .def("gather_crops_framed",
            [](PerceptionPipeline& self,
               uintptr_t uids_ptr, uintptr_t frames_ptr, int n,
               uintptr_t batch_ptr, uintptr_t stream_ptr,
               bool clean_only) -> int {
                return self.gather_crops_framed(
                    reinterpret_cast<const uint64_t*>(uids_ptr),
                    reinterpret_cast<const int*>(frames_ptr),
                    n,
                    reinterpret_cast<float*>(batch_ptr),
                    reinterpret_cast<cudaStream_t>(stream_ptr),
                    clean_only);
            },
            py::arg("uids_ptr"), py::arg("frames_ptr"), py::arg("n"),
            py::arg("batch_ptr"), py::arg("stream_ptr"),
            py::arg("clean_only") = false,
            "Gather one raw crop per (uid, frame) pair into batch.")
        .def("has_crop", &PerceptionPipeline::has_crop,
             py::arg("uid"), py::arg("frame"), py::arg("clean_only") = false)
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
               uintptr_t out_suspect, uintptr_t out_count,
               uintptr_t priors_ptr, uintptr_t prior_classes_ptr,
               int num_priors, float prior_iou_threshold,
               uintptr_t stream_ptr) {
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
                    reinterpret_cast<const float*>(priors_ptr),
                    reinterpret_cast<const int*>(prior_classes_ptr),
                    num_priors,
                    prior_iou_threshold,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"), py::arg("out_count"),
            py::arg("priors_ptr") = 0, py::arg("prior_classes_ptr") = 0,
            py::arg("num_priors") = 0, py::arg("prior_iou_threshold") = 0.50f,
            py::arg("stream_ptr"))
        // GIL-free synchronous variant: releases GIL for the entire filter+NMS+sync
        // sequence so concurrent Python threads can make progress during GPU execution.
        .def("process_detections_n",
            [](PerceptionPipeline& self,
               uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr,
               int n_in, int frame_w, int frame_h, bool is_tiled,
               uintptr_t out_boxes, uintptr_t out_scores, uintptr_t out_classes,
               uintptr_t out_suspect,
               uintptr_t priors_ptr, uintptr_t prior_classes_ptr,
               int num_priors, float prior_iou_threshold,
               uintptr_t stream_ptr) -> int {
                py::gil_scoped_release release;
                return self.process_detections_n(
                    reinterpret_cast<const float*>(boxes_ptr),
                    reinterpret_cast<const float*>(scores_ptr),
                    reinterpret_cast<const int*>(classes_ptr),
                    n_in, frame_w, frame_h, is_tiled,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<bool*>(out_suspect),
                    reinterpret_cast<const float*>(priors_ptr),
                    reinterpret_cast<const int*>(prior_classes_ptr),
                    num_priors, prior_iou_threshold,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"),
            py::arg("priors_ptr") = 0, py::arg("prior_classes_ptr") = 0,
            py::arg("num_priors") = 0, py::arg("prior_iou_threshold") = 0.50f,
            py::arg("stream_ptr"))
        .def("process_detections_n_private",
            [](PerceptionPipeline& self,
               uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr,
               int n_in, int frame_w, int frame_h, bool is_tiled,
               uintptr_t out_boxes, uintptr_t out_scores, uintptr_t out_classes,
               uintptr_t out_suspect,
               uintptr_t priors_ptr, uintptr_t prior_classes_ptr,
               int num_priors, float prior_iou_threshold,
               uintptr_t private_priors_ptr, int num_private_priors,
               uintptr_t stream_ptr) -> int {
                py::gil_scoped_release release;
                return self.process_detections_n(
                    reinterpret_cast<const float*>(boxes_ptr),
                    reinterpret_cast<const float*>(scores_ptr),
                    reinterpret_cast<const int*>(classes_ptr),
                    n_in, frame_w, frame_h, is_tiled,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<bool*>(out_suspect),
                    reinterpret_cast<const float*>(priors_ptr),
                    reinterpret_cast<const int*>(prior_classes_ptr),
                    num_priors,
                    prior_iou_threshold,
                    reinterpret_cast<cudaStream_t>(stream_ptr),
                    reinterpret_cast<const float*>(private_priors_ptr),
                    num_private_priors);
            },
            py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"),
            py::arg("priors_ptr") = 0, py::arg("prior_classes_ptr") = 0,
            py::arg("num_priors") = 0, py::arg("prior_iou_threshold") = 0.50f,
            py::arg("private_priors_ptr") = 0, py::arg("num_private_priors") = 0,
            py::arg("stream_ptr"))
        .def("process_detections_n_fixed",
            [](PerceptionPipeline& self,
               uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr,
               int n_in, int frame_w, int frame_h, bool is_tiled,
               uintptr_t out_boxes, uintptr_t out_scores, uintptr_t out_classes,
               uintptr_t out_suspect,
               uintptr_t priors_ptr, uintptr_t prior_classes_ptr,
               int num_priors, float prior_iou_threshold,
               uintptr_t stream_ptr) -> int {
                // GIL released: all work is async GPU + D2D, no CPU sync.
                py::gil_scoped_release release;
                return self.process_detections_n_fixed(
                    reinterpret_cast<const float*>(boxes_ptr),
                    reinterpret_cast<const float*>(scores_ptr),
                    reinterpret_cast<const int*>(classes_ptr),
                    n_in, frame_w, frame_h, is_tiled,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<bool*>(out_suspect),
                    reinterpret_cast<const float*>(priors_ptr),
                    reinterpret_cast<const int*>(prior_classes_ptr),
                    num_priors, prior_iou_threshold,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"),
            py::arg("priors_ptr") = 0, py::arg("prior_classes_ptr") = 0,
            py::arg("num_priors") = 0, py::arg("prior_iou_threshold") = 0.50f,
            py::arg("stream_ptr"))
        .def("process_detections_graph",
            [](PerceptionPipeline& self,
               uintptr_t boxes_ptr, uintptr_t scores_ptr, uintptr_t classes_ptr,
               int n_in, int frame_w, int frame_h, bool is_tiled,
               uintptr_t out_boxes, uintptr_t out_scores, uintptr_t out_classes,
               uintptr_t out_suspect, uintptr_t out_count,
               uintptr_t priors_ptr, uintptr_t prior_classes_ptr,
               int num_priors, float prior_iou_threshold,
               uintptr_t stream_ptr) {
                py::gil_scoped_release release;
                self.process_detections_graph(
                    reinterpret_cast<const float*>(boxes_ptr),
                    reinterpret_cast<const float*>(scores_ptr),
                    reinterpret_cast<const int*>(classes_ptr),
                    n_in, frame_w, frame_h, is_tiled,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<bool*>(out_suspect),
                    reinterpret_cast<int*>(out_count),
                    reinterpret_cast<const float*>(priors_ptr),
                    reinterpret_cast<const int*>(prior_classes_ptr),
                    num_priors, prior_iou_threshold,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("boxes_ptr"), py::arg("scores_ptr"), py::arg("classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"), py::arg("out_count"),
            py::arg("priors_ptr") = 0, py::arg("prior_classes_ptr") = 0,
            py::arg("num_priors") = 0, py::arg("prior_iou_threshold") = 0.50f,
            py::arg("stream_ptr"))
        .def("process_detections_interleaved_graph",
            [](PerceptionPipeline& self,
               uintptr_t det_6d,
               int n_in,
               uintptr_t split_boxes, uintptr_t split_scores, uintptr_t split_classes,
               int frame_w, int frame_h, bool is_tiled,
               uintptr_t out_boxes, uintptr_t out_scores, uintptr_t out_classes,
               uintptr_t out_suspect, uintptr_t out_count,
               uintptr_t stream_ptr) {
                py::gil_scoped_release release;
                self.process_detections_interleaved_graph(
                    reinterpret_cast<const float*>(det_6d),
                    n_in,
                    reinterpret_cast<float*>(split_boxes),
                    reinterpret_cast<float*>(split_scores),
                    reinterpret_cast<int*>(split_classes),
                    frame_w, frame_h, is_tiled,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<bool*>(out_suspect),
                    reinterpret_cast<int*>(out_count),
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("det_6d"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"),
            py::arg("is_tiled"),
            py::arg("split_boxes"), py::arg("split_scores"), py::arg("split_classes"),
            py::arg("out_boxes"), py::arg("out_scores"), py::arg("out_classes"),
            py::arg("out_suspect"), py::arg("out_count"),
            py::arg("stream_ptr"))
        .def("extract_reid",
            [](PerceptionPipeline& self,
               uintptr_t frame_ptr, int frame_h, int frame_w,
               uintptr_t boxes_ptr, int n_boxes,
               uintptr_t out_embeds, uintptr_t stream_ptr) {
                // All args are raw pointers / primitives — safe to release GIL.
                py::gil_scoped_release release;
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
        .def("crop_into_pool",
            [](PerceptionPipeline& self,
               uintptr_t frame_ptr, int frame_h, int frame_w,
               uintptr_t boxes_ptr, int n_boxes,
               uintptr_t stream_ptr) -> int {
                py::gil_scoped_release release;
                int slot = -1;
                self.crop_into_pool(
                    reinterpret_cast<const float*>(frame_ptr),
                    frame_h, frame_w,
                    reinterpret_cast<const float*>(boxes_ptr),
                    n_boxes,
                    &slot,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
                return slot;
            },
            py::arg("frame_ptr"), py::arg("frame_h"), py::arg("frame_w"),
            py::arg("boxes_ptr"), py::arg("n_boxes"),
            py::arg("stream_ptr"),
            "Crop boxes into the CropPool. Returns the starting slot index (-1 on failure).")
        .def("extract_from_pool",
            [](PerceptionPipeline& self,
               int slot, int n_boxes,
               uintptr_t out_embeds, uintptr_t stream_ptr) {
                py::gil_scoped_release release;
                self.extract_from_pool(
                    slot, n_boxes,
                    reinterpret_cast<float*>(out_embeds),
                    reinterpret_cast<cudaStream_t>(stream_ptr));
            },
            py::arg("slot"), py::arg("n_boxes"),
            py::arg("out_embeds"), py::arg("stream_ptr"),
            "Extract ReID embeddings from crops in the pool, then release the slots.")
        .def("crop_into_pool_async",
            [](PerceptionPipeline& self,
               uintptr_t frame_ptr, int frame_h, int frame_w,
               uintptr_t boxes_ptr, int n_boxes) -> py::tuple {
                int slot = -1;
                cudaEvent_t evt = nullptr;
                {
                    py::gil_scoped_release release;
                    self.crop_into_pool_async(
                        reinterpret_cast<const float*>(frame_ptr),
                        frame_h, frame_w,
                        reinterpret_cast<const float*>(boxes_ptr),
                        n_boxes,
                        &slot, &evt);
                }
                return py::make_tuple(slot, reinterpret_cast<uintptr_t>(evt));
            },
            py::arg("frame_ptr"), py::arg("frame_h"), py::arg("frame_w"),
            py::arg("boxes_ptr"), py::arg("n_boxes"),
            "Async crop into pool on crop_stream. Returns (slot, event_ptr).")
        .def("extract_batch_from_pool",
            [](PerceptionPipeline& self,
               const py::list& jobs,
               uintptr_t out_embeds) -> py::tuple {
                // Convert Python list of dicts to ReIDCropJob array.
                std::vector<ReIDCropJob> cpp_jobs;
                cpp_jobs.reserve(jobs.size());
                for (auto& item : jobs) {
                    auto d = item.cast<py::dict>();
                    ReIDCropJob job;
                    job.crop_slot = d["crop_slot"].cast<int>();
                    job.n_crops = d["n_crops"].cast<int>();
                    job.frame_idx = d["frame_idx"].cast<int>();
                    job.track_uid = d["track_uid"].cast<uint64_t>();
                    job.generation = d["generation"].cast<int>();
                    job.det_score = d["det_score"].cast<float>();
                    job.quality = d["quality"].cast<float>();
                    job.reason = d["reason"].cast<int>();
                    uintptr_t evt_ptr = d["crop_ready"].cast<uintptr_t>();
                    job.crop_ready = reinterpret_cast<cudaEvent_t>(evt_ptr);
                    cpp_jobs.push_back(job);
                }
                int n_jobs = static_cast<int>(cpp_jobs.size());
                std::vector<ReIDResult> results(n_jobs);
                int n_extracted;
                {
                    py::gil_scoped_release release;
                    n_extracted = self.extract_batch_from_pool(
                        cpp_jobs.data(), n_jobs,
                        reinterpret_cast<float*>(out_embeds),
                        results.data());
                }
                // Build Python result list with GIL held.
                py::list result_list;
                for (int i = 0; i < n_jobs; ++i) {
                    py::dict r;
                    r["frame_idx"] = results[i].frame_idx;
                    r["track_uid"] = results[i].track_uid;
                    r["generation"] = results[i].generation;
                    r["reason"] = results[i].reason;
                    r["embed_offset"] = results[i].embed_offset;
                    r["n_crops"] = results[i].n_crops;
                    result_list.append(r);
                }
                return py::make_tuple(n_extracted, result_list);
            },
            py::arg("jobs"), py::arg("out_embeds"),
            "Batch extract: wait on crop events, gather, infer, release slots. "
            "Returns (n_extracted, results_list).")
        .def("crop_stream",
            [](PerceptionPipeline& self) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(self.crop_stream());
            },
            "Return the internal crop CUDA stream pointer.")
        .def("reid_stream",
            [](PerceptionPipeline& self) -> uintptr_t {
                return reinterpret_cast<uintptr_t>(self.reid_stream());
            },
            "Return the internal reid CUDA stream pointer.")
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
        .def("set_postprocess_profiling_enabled", &PerceptionPipeline::set_postprocess_profiling_enabled, py::arg("enabled"))
        .def("reset_postprocess_profile_stats", &PerceptionPipeline::reset_postprocess_profile_stats)
        .def("get_postprocess_profile_stats",
            [](const PerceptionPipeline& self) {
                const auto stats = self.get_postprocess_profile_stats();
                py::dict out;
                out["filter_ms"] = stats.filter_ms;
                out["nms_ms"] = stats.nms_ms;
                out["count_d2h_ms"] = stats.count_d2h_ms;
                out["total_ms"] = stats.total_ms;
                out["native_filter_gather_ms"] = stats.native_filter_gather_ms;
                out["native_filter_kernel_ms"] = stats.native_filter_kernel_ms;
                out["native_gather_compact3_ms"] = stats.native_gather_compact3_ms;
                out["native_copy_suspect_ms"] = stats.native_copy_suspect_ms;
                out["native_filter_count_sync_ms"] = stats.native_filter_count_sync_ms;
                out["native_small_nms_ms"] = stats.native_small_nms_ms;
                out["native_suspect_penalty_ms"] = stats.native_suspect_penalty_ms;
                out["native_large_sort_nms_ms"] = stats.native_large_sort_nms_ms;
                out["native_large_argsort_ms"] = stats.native_large_argsort_ms;
                out["native_large_nms_ms"] = stats.native_large_nms_ms;
                out["native_compact_copy_ms"] = stats.native_compact_copy_ms;
                out["native_large_gather4_ms"] = stats.native_large_gather4_ms;
                out["native_large_copyback_ms"] = stats.native_large_copyback_ms;
                out["native_private_candidate_nms_ms"] = stats.native_private_candidate_nms_ms;
                out["native_private_append_ms"] = stats.native_private_append_ms;
                out["input_boxes"] = stats.input_boxes;
                out["filtered_boxes"] = stats.filtered_boxes;
                out["output_boxes"] = stats.output_boxes;
                out["private_boxes"] = stats.private_boxes;
                return out;
            })
        .def_property_readonly("embed_dim", &PerceptionPipeline::get_embed_dim)
        .def_property_readonly("cpp_ptr", [](PerceptionPipeline& self) {
            return reinterpret_cast<uintptr_t>(&self);
        }, "Raw C++ pointer to this PerceptionPipeline (for Workbench construction)");

    // Workbench: per-thread isolated workspace for the post-YOLO hot path.
    // See include/tracking/workbench.hpp for the full architectural intent.
    py::class_<Workbench>(m, "Workbench")
        .def(py::init([](uintptr_t pipeline_ptr, uintptr_t tracker_ptr,
                         uintptr_t stream_ptr, int max_dets, int max_tracks,
                         int output_capacity) {
                return new Workbench(
                    reinterpret_cast<PerceptionPipeline*>(pipeline_ptr),
                    reinterpret_cast<GPUByteTracker*>(tracker_ptr),
                    reinterpret_cast<cudaStream_t>(stream_ptr),
                    max_dets, max_tracks, output_capacity);
             }),
             py::arg("pipeline_ptr"), py::arg("tracker_ptr"), py::arg("stream_ptr"),
             py::arg("max_dets") = 2048, py::arg("max_tracks") = 256,
             py::arg("output_capacity") = -1,
             "Borrow pipeline + tracker (must be per-workbench instances, not shared) "
             "and a CUDA stream. Allocates per-workbench post-NMS scratch.")
        .def("process_frame_postyolo",
            [](Workbench& self,
               uintptr_t raw_boxes, uintptr_t raw_scores, uintptr_t raw_classes,
               int n_in, int frame_w, int frame_h, bool is_tiled,
               uintptr_t priors_ptr, uintptr_t prior_classes_ptr,
               int num_priors, float prior_iou_threshold,
               uintptr_t embeddings_ptr, uintptr_t gmc_ptr,
               float light_factor, float mid_thresh_scale,
               uintptr_t out_boxes, uintptr_t out_scores,
               uintptr_t out_ids, uintptr_t out_classes,
               uintptr_t out_det_idx, uintptr_t out_count) -> int {
                // Single GIL release for the entire ~3-6 ms hot path. All
                // inputs are raw GPU pointers / primitives; nothing accesses
                // Python state inside, so sibling worker threads can make
                // Python progress (incl. their own next-frame submission)
                // while this thread's C++ kernels run on its CUDA stream.
                py::gil_scoped_release release;
                return self.process_frame_postyolo(
                    reinterpret_cast<const float*>(raw_boxes),
                    reinterpret_cast<const float*>(raw_scores),
                    reinterpret_cast<const int*>(raw_classes),
                    n_in, frame_w, frame_h, is_tiled,
                    reinterpret_cast<const float*>(priors_ptr),
                    reinterpret_cast<const int*>(prior_classes_ptr),
                    num_priors, prior_iou_threshold,
                    reinterpret_cast<const float*>(embeddings_ptr),
                    reinterpret_cast<const float*>(gmc_ptr),
                    light_factor, mid_thresh_scale,
                    reinterpret_cast<float*>(out_boxes),
                    reinterpret_cast<float*>(out_scores),
                    reinterpret_cast<int*>(out_ids),
                    reinterpret_cast<int*>(out_classes),
                    reinterpret_cast<int*>(out_det_idx),
                    reinterpret_cast<int*>(out_count));
            },
            py::arg("raw_boxes_ptr"), py::arg("raw_scores_ptr"), py::arg("raw_classes_ptr"),
            py::arg("n_in"), py::arg("frame_w"), py::arg("frame_h"), py::arg("is_tiled"),
            py::arg("priors_ptr") = 0, py::arg("prior_classes_ptr") = 0,
            py::arg("num_priors") = 0, py::arg("prior_iou_threshold") = 0.5f,
            py::arg("embeddings_ptr") = 0, py::arg("gmc_ptr") = 0,
            py::arg("light_factor") = 0.0f, py::arg("mid_thresh_scale") = 1.0f,
            py::arg("out_boxes_ptr"), py::arg("out_scores_ptr"),
            py::arg("out_ids_ptr"), py::arg("out_classes_ptr"),
            py::arg("out_det_idx_ptr"), py::arg("out_count_ptr"));

    // Fused letterbox: bilinear resize + pad in one CUDA kernel.
    // Replaces: interpolate → fill_ → copy_ (3 ops) with a single kernel launch.
    m.def("letterbox_gpu", [](
        uintptr_t src_ptr, int src_w, int src_h,
        uintptr_t dst_ptr, int dst_size,
        int x_off, int y_off, int w_new, int h_new,
        float pad_val, uintptr_t stream_ptr)
    {
        launch_letterbox_gpu(
            reinterpret_cast<const float*>(src_ptr), src_w, src_h,
            reinterpret_cast<float*>(dst_ptr), dst_size,
            x_off, y_off, w_new, h_new,
            pad_val, reinterpret_cast<cudaStream_t>(stream_ptr));
    },
    py::arg("src_ptr"), py::arg("src_w"), py::arg("src_h"),
    py::arg("dst_ptr"), py::arg("dst_size"),
    py::arg("x_off"), py::arg("y_off"), py::arg("w_new"), py::arg("h_new"),
    py::arg("pad_val") = 114.0f / 255.0f,
    py::arg("stream_ptr") = 0,
    "Fused letterbox: bilinear resize + constant pad into a square canvas in one kernel.");

    // ── GPU Quality Filter ops (ports of quality.py / workbench.py) ──────────

    m.def("quality_scale_scores",
        [](uintptr_t scores_ptr, uintptr_t boxes_ptr, int n,
           int frame_w, int frame_h,
           float w_aspect, float w_center, float w_area,
           uintptr_t stream_ptr) {
            py::gil_scoped_release release;
            quality_scale_scores(
                reinterpret_cast<float*>(scores_ptr),
                reinterpret_cast<const float*>(boxes_ptr),
                n, frame_w, frame_h, w_aspect, w_center, w_area,
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("scores_ptr"), py::arg("boxes_ptr"), py::arg("n"),
        py::arg("frame_w"), py::arg("frame_h"),
        py::arg("w_aspect") = 0.50f, py::arg("w_center") = 0.30f, py::arg("w_area") = 0.20f,
        py::arg("stream_ptr") = 0,
        "Multiply scores in-place by detection geometry quality (aspect/center/area). "
        "GPU equivalent of compute_detection_quality_batch().");

    m.def("narrow_bonus_scores",
        [](uintptr_t scores_ptr, uintptr_t boxes_ptr, uintptr_t classes_ptr, int n,
           float bonus, int person_class,
           float narrow_aspect_thresh, float narrow_height_thresh,
           int frame_h, uintptr_t stream_ptr) {
            py::gil_scoped_release release;
            narrow_bonus_scores(
                reinterpret_cast<float*>(scores_ptr),
                reinterpret_cast<const float*>(boxes_ptr),
                reinterpret_cast<const int*>(classes_ptr),
                n, bonus, person_class,
                narrow_aspect_thresh, narrow_height_thresh, frame_h,
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("scores_ptr"), py::arg("boxes_ptr"), py::arg("classes_ptr"), py::arg("n"),
        py::arg("bonus"), py::arg("person_class") = 0,
        py::arg("narrow_aspect_thresh") = 2.1f, py::arg("narrow_height_thresh") = 0.5f,
        py::arg("frame_h") = 1080, py::arg("stream_ptr") = 0,
        "Add score bonus to narrow/tall person detections in-place. "
        "GPU equivalent of _apply_narrow_bonus().");

    m.def("fp_hard_filter",
        [](uintptr_t boxes_in, uintptr_t scores_in, uintptr_t classes_in, int n,
           float min_score, float max_area, float max_suspicious_score,
           uintptr_t boxes_out, uintptr_t scores_out, uintptr_t classes_out,
           uintptr_t stream_ptr) -> int {
            py::gil_scoped_release release;
            return fp_hard_filter(
                reinterpret_cast<const float*>(boxes_in),
                reinterpret_cast<const float*>(scores_in),
                reinterpret_cast<const int*>(classes_in),
                n, min_score, max_area, max_suspicious_score,
                reinterpret_cast<float*>(boxes_out),
                reinterpret_cast<float*>(scores_out),
                reinterpret_cast<int*>(classes_out),
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("boxes_in"), py::arg("scores_in"), py::arg("classes_in"), py::arg("n"),
        py::arg("min_score") = 0.25f, py::arg("max_area") = 10000.0f,
        py::arg("max_suspicious_score") = 0.45f,
        py::arg("boxes_out"), py::arg("scores_out"), py::arg("classes_out"),
        py::arg("stream_ptr") = 0,
        "Compact-filter detections removing very-low-score and large+uncertain boxes. "
        "Returns count of kept detections. GPU equivalent of _apply_fp_hard_filter().");

    m.def("relink_gate_batch",
        [](int n_query, int n_cand, int w, int h, int dims,
           float fps, float person_height_m, float max_speed_mps,
           float accel_long, float accel_lat, float dir_min_cos, float dir_min_speed,
           uintptr_t query_boxes, uintptr_t query_foot, uintptr_t query_foot_n,
           uintptr_t query_emah,
           uintptr_t cand_last_box, uintptr_t cand_mean, uintptr_t cand_cov,
           uintptr_t cand_foot, uintptr_t cand_foot_n, uintptr_t cand_emah,
           uintptr_t cand_gap, uintptr_t cand_delta, uintptr_t cand_has_snap,
           uintptr_t table, uintptr_t stream_ptr) {
            py::gil_scoped_release release;
            saccade::relink_gate::GateParams p{
                n_query, n_cand, w, h, dims, fps, person_height_m, max_speed_mps,
                accel_long, accel_lat, dir_min_cos, dir_min_speed};
            saccade::relink_gate::launch(
                p,
                reinterpret_cast<const float*>(query_boxes),
                reinterpret_cast<const float*>(query_foot),
                reinterpret_cast<const int*>(query_foot_n),
                reinterpret_cast<const float*>(query_emah),
                reinterpret_cast<const float*>(cand_last_box),
                reinterpret_cast<const float*>(cand_mean),
                reinterpret_cast<const float*>(cand_cov),
                reinterpret_cast<const float*>(cand_foot),
                reinterpret_cast<const int*>(cand_foot_n),
                reinterpret_cast<const float*>(cand_emah),
                reinterpret_cast<const int*>(cand_gap),
                reinterpret_cast<const int*>(cand_delta),
                reinterpret_cast<const int*>(cand_has_snap),
                reinterpret_cast<float*>(table),
                reinterpret_cast<void*>(stream_ptr));
        },
        py::arg("n_query"), py::arg("n_cand"), py::arg("w"), py::arg("h"), py::arg("dims"),
        py::arg("fps"), py::arg("person_height_m"), py::arg("max_speed_mps"),
        py::arg("accel_long"), py::arg("accel_lat"),
        py::arg("dir_min_cos"), py::arg("dir_min_speed"),
        py::arg("query_boxes"), py::arg("query_foot"), py::arg("query_foot_n"),
        py::arg("query_emah"),
        py::arg("cand_last_box"), py::arg("cand_mean"), py::arg("cand_cov"),
        py::arg("cand_foot"), py::arg("cand_foot_n"), py::arg("cand_emah"),
        py::arg("cand_gap"), py::arg("cand_delta"), py::arg("cand_has_snap"),
        py::arg("table"), py::arg("stream_ptr") = 0,
        "Batched per-(query,candidate) relink gate table. Fills table "
        "[n_query*n_cand*6] = {kalman_d2,bridge_dist,center_norm,iou,speed_exceeds,"
        "dir_behind}. All device pointers. Thresholds stay Python-side.");

    m.def("copy_pad_detections",
        [](uintptr_t src_boxes, uintptr_t src_scores, uintptr_t src_classes,
           int n_copy,
           uintptr_t dst_boxes, uintptr_t dst_scores, uintptr_t dst_classes,
           int padded_n, uintptr_t stream_ptr) {
            py::gil_scoped_release release;
            copy_pad_detections(
                reinterpret_cast<const float*>(src_boxes),
                reinterpret_cast<const float*>(src_scores),
                reinterpret_cast<const int*>(src_classes),
                n_copy,
                reinterpret_cast<float*>(dst_boxes),
                reinterpret_cast<float*>(dst_scores),
                reinterpret_cast<int*>(dst_classes),
                padded_n,
                reinterpret_cast<cudaStream_t>(stream_ptr));
        },
        py::arg("src_boxes"), py::arg("src_scores"), py::arg("src_classes"),
        py::arg("n_copy"),
        py::arg("dst_boxes"), py::arg("dst_scores"), py::arg("dst_classes"),
        py::arg("padded_n"), py::arg("stream_ptr") = 0,
        "Copy N detections to padded output, zero-fill tail. "
        "Single kernel replaces copy+zero_ pairs for graph-NMS input prep.");

    m.def("auction_solve_cpp",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> cost_matrix, float epsilon) {
            if (cost_matrix.ndim() != 2) {
                throw std::invalid_argument("cost_matrix must be 2D");
            }
            int n_bidders = static_cast<int>(cost_matrix.shape(0));
            int n_items = static_cast<int>(cost_matrix.shape(1));
            
            auto buf = cost_matrix.unchecked<2>();
            float max_cost = 0.0f;
            for (int i = 0; i < n_bidders; ++i) {
                for (int j = 0; j < n_items; ++j) {
                    max_cost = std::max(max_cost, buf(i, j));
                }
            }

            std::vector<std::vector<float>> profit_matrix(n_bidders, std::vector<float>(n_items, 0.0f));
            for (int i = 0; i < n_bidders; ++i) {
                for (int j = 0; j < n_items; ++j) {
                    profit_matrix[i][j] = max_cost - buf(i, j) + 1.0f;
                }
            }
            
            std::vector<int> assignment;
            // release GIL for auction algorithm execution
            {
                py::gil_scoped_release release;
                saccade::AuctionAlgorithm::Solve(profit_matrix, assignment, epsilon);
            }
            
            std::vector<int> row_ind;
            std::vector<int> col_ind;
            for (int i = 0; i < n_bidders; ++i) {
                if (assignment[i] != -1) {
                    row_ind.push_back(i);
                    col_ind.push_back(assignment[i]);
                }
            }
            
            return py::make_tuple(row_ind, col_ind);
        },
        py::arg("cost_matrix"), py::arg("epsilon") = 0.01f,
        "Solve linear assignment problem using C++ Auction Algorithm (minimizing cost_matrix).");

    m.def("selective_scan_fwd",
        [](
            uintptr_t u_ptr, uintptr_t delta_ptr, uintptr_t A_ptr,
            uintptr_t B_ptr, uintptr_t C_ptr, uintptr_t D_ptr, uintptr_t y_ptr,
            int B_dim, int L_dim, int D_dim, int N_dim, int has_D,
            int a_per_channel, bool is_half, uintptr_t stream_ptr
        ) {
            if (N_dim <= 0 || N_dim > 32 || (N_dim & (N_dim - 1)) != 0) {
                throw std::invalid_argument(
                    "selective_scan_fwd requires power-of-two N_dim in [1, 32]");
            }
            SelectiveScanParams params;
            params.B = B_dim;
            params.L = L_dim;
            params.D = D_dim;
            params.N = N_dim;
            params.has_D = has_D;
            params.a_per_channel = a_per_channel;

            // Launch on the caller's CUDA stream (PyTorch's current stream).
            // Defaulting to the legacy default stream (stream 0) makes the
            // kernel invisible to CUDA-graph capture, which never records work
            // on the capture stream — replay then leaves y unfilled and the
            // head output saturates. See mamba_head_cuda_graph_eval_bug doc.
            void* stream = reinterpret_cast<void*>(stream_ptr);

            {
                py::gil_scoped_release release;
                if (is_half) {
                    selective_scan_fwd_half(
                        reinterpret_cast<const void*>(u_ptr),
                        reinterpret_cast<const void*>(delta_ptr),
                        reinterpret_cast<const void*>(A_ptr),
                        reinterpret_cast<const void*>(B_ptr),
                        reinterpret_cast<const void*>(C_ptr),
                        has_D ? reinterpret_cast<const void*>(D_ptr) : nullptr,
                        reinterpret_cast<void*>(y_ptr),
                        params,
                        stream
                    );
                } else {
                    selective_scan_fwd(
                        reinterpret_cast<const float*>(u_ptr),
                        reinterpret_cast<const float*>(delta_ptr),
                        reinterpret_cast<const float*>(A_ptr),
                        reinterpret_cast<const float*>(B_ptr),
                        reinterpret_cast<const float*>(C_ptr),
                        has_D ? reinterpret_cast<const float*>(D_ptr) : nullptr,
                        reinterpret_cast<float*>(y_ptr),
                        params,
                        stream
                    );
                }
            }
        },
        py::arg("u_ptr"), py::arg("delta_ptr"), py::arg("A_ptr"),
        py::arg("B_ptr"), py::arg("C_ptr"), py::arg("D_ptr"), py::arg("y_ptr"),
        py::arg("B_dim"), py::arg("L_dim"), py::arg("D_dim"),
        py::arg("N_dim"), py::arg("has_D"),
        py::arg("a_per_channel") = 0, py::arg("is_half") = false,
        py::arg("stream_ptr") = 0,
        "CUDA selective scan (Mamba SSM kernel) supporting both float and half.");

    m.def("selective_scan_bwd",
        [](
            uintptr_t grad_y_ptr, uintptr_t u_ptr, uintptr_t delta_ptr,
            uintptr_t A_ptr, uintptr_t B_ptr, uintptr_t C_ptr, uintptr_t D_ptr,
            uintptr_t h_buf_ptr, uintptr_t du_ptr, uintptr_t ddelta_ptr,
            uintptr_t dA_ptr, uintptr_t dB_ptr, uintptr_t dC_ptr, uintptr_t dD_ptr,
            int B_dim, int L_dim, int D_dim, int N_dim, int has_D,
            int a_per_channel, uintptr_t stream_ptr
        ) {
            if (N_dim <= 0 || N_dim > 32 || (N_dim & (N_dim - 1)) != 0) {
                throw std::invalid_argument(
                    "selective_scan_bwd requires power-of-two N_dim in [1, 32]");
            }
            SelectiveScanParams params;
            params.B = B_dim;
            params.L = L_dim;
            params.D = D_dim;
            params.N = N_dim;
            params.has_D = has_D;
            params.a_per_channel = a_per_channel;

            void* stream = reinterpret_cast<void*>(stream_ptr);
            {
                py::gil_scoped_release release;
                selective_scan_bwd(
                    reinterpret_cast<const float*>(grad_y_ptr),
                    reinterpret_cast<const float*>(u_ptr),
                    reinterpret_cast<const float*>(delta_ptr),
                    reinterpret_cast<const float*>(A_ptr),
                    reinterpret_cast<const float*>(B_ptr),
                    reinterpret_cast<const float*>(C_ptr),
                    has_D ? reinterpret_cast<const float*>(D_ptr) : nullptr,
                    reinterpret_cast<float*>(h_buf_ptr),
                    reinterpret_cast<float*>(du_ptr),
                    reinterpret_cast<float*>(ddelta_ptr),
                    reinterpret_cast<float*>(dA_ptr),
                    reinterpret_cast<float*>(dB_ptr),
                    reinterpret_cast<float*>(dC_ptr),
                    has_D ? reinterpret_cast<float*>(dD_ptr) : nullptr,
                    params,
                    stream
                );
            }
        },
        py::arg("grad_y_ptr"), py::arg("u_ptr"), py::arg("delta_ptr"),
        py::arg("A_ptr"), py::arg("B_ptr"), py::arg("C_ptr"), py::arg("D_ptr"),
        py::arg("h_buf_ptr"), py::arg("du_ptr"), py::arg("ddelta_ptr"),
        py::arg("dA_ptr"), py::arg("dB_ptr"), py::arg("dC_ptr"), py::arg("dD_ptr"),
        py::arg("B_dim"), py::arg("L_dim"), py::arg("D_dim"), py::arg("N_dim"),
        py::arg("has_D"), py::arg("a_per_channel") = 0, py::arg("stream_ptr") = 0,
        "CUDA backward of the selective scan (fp32 training path).");
}
