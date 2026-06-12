// GPU batched relink-gate evaluation.
//
// Offloads the O(n_query x n_cand) per-pair gate computation of the Python
// SemanticRelinker (see report_data/algorithms.md sec 6.4 / 6.6 and
// src/saccade/perception/eval/relink.py) onto the GPU.  The Python side keeps
// all stateful decision logic (frame ordering, the dynamic `assigned` set,
// alias/split-collision, reciprocal margin) and only reads the per-pair gate
// quantities from the table this kernel fills — so the result is bit-faithful
// to the pure-Python path while removing the relink worker's hot loop.
//
// Candidate state is marshalled from the Python relinker each frame (approach
// "GPU gate, Python decide"): the candidate set stays identical to relink.py so
// the +1.2 IDF1 / id-uniqueness win is preserved.
#pragma once

namespace saccade {
namespace relink_gate {

// Per-pair output table layout: row-major [n_query * n_cand * GATE_COLS].
// Columns:
//   0 kalman_d2     squared Mahalanobis (dims=2), -1 if no snapshot
//   1 bridge_dist   midpoint-bridge distance / h_ref, -1 if no foot history
//   2 center_norm   ||c_q - c_cand|| / max(w,h)
//   3 iou           IoU(box_q, last_box_cand)
//   4 speed_exceeds 1.0 if above-human implied speed else 0.0
//   5 dir_behind    1.0 if query is behind the candidate's velocity else 0.0
constexpr int GATE_COLS = 6;

// All geometry is in pixel space; angles/cosines unitless.  `dims` selects the
// Kalman gate dimensionality (2 for bidirectional, 4 otherwise).  When
// person_height_m > 0 the physical (inertia-aware) predict is used, matching
// relink.py `_predict_phys_np`; otherwise the constant-velocity predict.
struct GateParams {
    int   n_query;
    int   n_cand;
    int   w;
    int   h;
    int   dims;            // 2 (bidirectional) or 4
    float fps;
    float person_height_m; // <=0 -> constant-velocity predict, speed gate off
    float max_speed_mps;   // <=0 -> speed gate off
    float accel_long;      // kalman_accel_long
    float accel_lat;       // kalman_accel_lat
    float dir_min_cos;     // <=-1 -> direction gate off
    float dir_min_speed;   // near-stationary cutoff for the direction gate
};

// Inputs (host or device pointers depending on the launcher; the pybind wrapper
// accepts CUDA tensors).  Sizes:
//   query_boxes   [n_query * 4]   (x1,y1,x2,y2)
//   query_foot    [n_query * 8]   first-4 foot centres (x,y) oldest->newest
//   query_foot_n  [n_query]       valid foot count (0..4)
//   query_emah    [n_query]       query track ema height (h_cand in the bridge)
//   cand_last_box [n_cand * 4]
//   cand_mean     [n_cand * 6]    (x0,x1,x3,x4,x5,x7) reduced Kalman mean
//   cand_cov      [n_cand * 10]   symmetric 4x4 over {0,1,4,5}: 00,01,04,05,11,14,15,44,45,55
//   cand_foot     [n_cand * 8]    last-4 foot centres (x,y)
//   cand_foot_n   [n_cand]
//   cand_emah     [n_cand]
//   cand_gap      [n_cand]        frame_id - last_seen   (speed dt + bridge half)
//   cand_delta    [n_cand]        frame_id - motion_frame (Kalman predict steps)
//   cand_has_snap [n_cand]        1 if a Kalman snapshot exists
// Output:
//   table         [n_query * n_cand * GATE_COLS]
void launch(const GateParams& p,
            const float* query_boxes, const float* query_foot, const int* query_foot_n,
            const float* query_emah,
            const float* cand_last_box, const float* cand_mean, const float* cand_cov,
            const float* cand_foot, const int* cand_foot_n, const float* cand_emah,
            const int* cand_gap, const int* cand_delta, const int* cand_has_snap,
            float* table, void* stream);

// Gather tracker Kalman state into relink gate format (GPU→GPU).
// Extracts 6-element mean {cx,cy,h,vx,vy,vh} and 10-element covariance
// (symmetric 4x4 over {cx,cy,vx,vy}) from the tracker's dense buffers.
void gather_tracker_state(const float* src_states, const float* src_covs,
                          const int* slots, int n_cand,
                          float* dst_mean, float* dst_cov, void* stream);

// Batch dot product: computes dot(queries[i], candidates[j]) for all pairs.
void batch_dot(const float* queries, const float* candidates,
               int n_query, int n_cand, int dim,
               float* out, void* stream);

// Scoring parameters for the relink joint-score kernel.
struct ScoringParams {
    int   n_cand = 0;
    int   ttl = 45;
    float sim_threshold = 0.85f;
    float w_sim_base = 0.0f, w_iou_base = 0.0f, w_maha_base = 0.0f;
    float shift_ambiguity = 0.0f, shift_lost_age = 0.0f;
    float mahalanobis_threshold = 0.0f;
    float kalman_penalty_weight = 0.0f;
    float reciprocal_margin = 0.0f;
    float dynamic_margin_crowd = 0.0f, dynamic_margin_age = 0.0f;
    float iou_weight = 0.0f, mahalanobis_weight = 0.0f;
};

// Batch joint-score computation: one block per query, computes dynamic
// weighted scores for all candidates, reduces to top 2 per query.
void launch_relink_scoring(
    const float* gate_tbl, const float* sim_tbl,
    const int* cand_ids, const int* cand_ages, const float* cand_maha,
    int n_query, ScoringParams params,
    int* best_ids, float* best_scores, float* second_scores, void* stream);

}  // namespace relink_gate
}  // namespace saccade
