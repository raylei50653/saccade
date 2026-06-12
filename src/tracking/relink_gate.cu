// GPU batched relink-gate evaluation — see include/tracking/relink_gate.hpp.
//
// Faithfully mirrors the per-pair gate math of src/saccade/perception/eval/
// relink.py (_kalman_gate_dist dims=2, _midpoint_bridge_dist, _spatial_metrics,
// _exceeds_max_speed, _direction_behind) and report_data/algorithms.md 6.4/6.6.
// Only raw quantities are emitted; all thresholds remain on the Python side.
#include "tracking/relink_gate.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace saccade {
namespace relink_gate {

// 4-frame closed-form regression velocity (algorithms.md 6.6).  Matches
// relink.py _regress_velocity_4: returns (0,0) when fewer than 4 points.
__device__ __forceinline__ void regress4(const float* fxy, int n, float& vx, float& vy) {
    if (n < 4) { vx = 0.0f; vy = 0.0f; return; }
    float x0 = fxy[0], y0 = fxy[1];
    float x1 = fxy[2], y1 = fxy[3];
    float x2 = fxy[4], y2 = fxy[5];
    float x3 = fxy[6], y3 = fxy[7];
    vx = (3.0f * x3 + x2 - x1 - 3.0f * x0) / 10.0f;
    vy = (3.0f * y3 + y2 - y1 - 3.0f * y0) / 10.0f;
}

// 8-frame OLS regression velocity.  t=0..7, t̄=3.5, Stt=42.
// Falls back to regress4 when n < 8.  Buffer fxy must have 16 floats (stride 16).
__device__ __forceinline__ void regress8(const float* fxy, int n, float& vx, float& vy) {
    if (n < 8) { regress4(fxy, n, vx, vy); return; }
    float x0=fxy[0], y0=fxy[1], x1=fxy[2], y1=fxy[3];
    float x2=fxy[4], y2=fxy[5], x3=fxy[6], y3=fxy[7];
    float x4=fxy[8], y4=fxy[9], x5=fxy[10],y5=fxy[11];
    float x6=fxy[12],y6=fxy[13],x7=fxy[14],y7=fxy[15];
    vx = (-7.f*x0-5.f*x1-3.f*x2-x3+x4+3.f*x5+5.f*x6+7.f*x7)/84.f;
    vy = (-7.f*y0-5.f*y1-3.f*y2-y3+y4+3.f*y5+5.f*y6+7.f*y7)/84.f;
}

// One physical (inertia-aware) predict step on the reduced {0,1,4,5} system.
// mean: (x0,x1,x3,x4,x5,x7) packed; cov: symmetric 4x4 over {0,1,4,5} packed as
//   00,01,04,05, 11,14,15, 44,45, 55  (10 entries).
// Mirrors relink.py _predict_phys_np restricted to the position/velocity block
// (height x3 is carried only to scale the anisotropic acceleration noise).
__device__ void predict_phys_step(float* m, float* c,
                                  float person_h, float fps,
                                  float accel_long, float accel_lat) {
    // local cov indices over states {0,1,4,5} -> [0,1,2,3]
    // packed: c[0]=P00 c[1]=P01 c[2]=P02 c[3]=P03 c[4]=P11 c[5]=P12 c[6]=P13
    //         c[7]=P22 c[8]=P23 c[9]=P33   (symmetric)
    float P00 = c[0], P01 = c[1], P02 = c[2], P03 = c[3];
    float P11 = c[4], P12 = c[5], P13 = c[6];
    float P22 = c[7], P23 = c[8];
    float P33 = c[9];

    // mean advance: x0+=x4, x1+=x5, x3+=x7
    m[0] += m[3];  // x0 += x4
    m[1] += m[4];  // x1 += x5
    m[2] += m[5];  // x3 += x7

    // F (local) couples row0+=row2, row1+=row3; rows2,3 unchanged.
    // P' = F P F^T.  With F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]:
    float n00 = P00 + P02 + P02 + P22;        // P00 + 2 P02 + P22
    float n01 = P01 + P03 + P12 + P23;        // P01 + P03 + P12 + P23
    float n02 = P02 + P22;
    float n03 = P03 + P23;
    float n11 = P11 + P13 + P13 + P33;        // P11 + 2 P13 + P33
    float n12 = P12 + P23;
    float n13 = P13 + P33;
    float n22 = P22, n23 = P23, n33 = P33;

    // anisotropic acceleration noise (decomposed along velocity direction)
    float px_per_m = fmaxf(m[2], 1e-3f) / fmaxf(person_h, 1e-3f);
    float inv_fps2 = 1.0f / fmaxf(fps * fps, 1e-6f);
    float sl = accel_long * px_per_m * inv_fps2;
    float st = accel_lat * px_per_m * inv_fps2;
    float sl2 = sl * sl, st2 = st * st;
    float vx = m[3], vy = m[4];
    float speed = sqrtf(vx * vx + vy * vy);
    float ux, uy;
    if (speed > 1e-6f) { ux = vx / speed; uy = vy / speed; }
    else               { ux = 1.0f; uy = 0.0f; sl2 = st2; }  // isotropic
    float sa00 = sl2 * ux * ux + st2 * uy * uy;
    float sa01 = (sl2 - st2) * ux * uy;
    float sa11 = sl2 * uy * uy + st2 * ux * ux;

    // add to pos(0,1)/vel(2,3) sub-blocks (0.25 pp, 0.5 pv/vp, 1.0 vv)
    n00 += 0.25f * sa00; n01 += 0.25f * sa01; n11 += 0.25f * sa11;
    n02 += 0.5f * sa00;  n03 += 0.5f * sa01;
    n12 += 0.5f * sa01;  n13 += 0.5f * sa11;          // P[1,4]+=0.5*sa[1,0]=sa01
    n22 += sa00;         n23 += sa01;  n33 += sa11;

    c[0] = n00; c[1] = n01; c[2] = n02; c[3] = n03;
    c[4] = n11; c[5] = n12; c[6] = n13;
    c[7] = n22; c[8] = n23; c[9] = n33;
}

__global__ void gate_kernel(GateParams p,
                            const float* __restrict__ qbox, const float* __restrict__ qfoot,
                            const int* __restrict__ qfoot_n,
                            const float* __restrict__ clast, const float* __restrict__ cmean,
                            const float* __restrict__ ccov, const float* __restrict__ cfoot,
                            const int* __restrict__ cfoot_n, const float* __restrict__ cemah,
                            const int* __restrict__ cgap, const int* __restrict__ cdelta,
                            const int* __restrict__ chas,
                            const float* __restrict__ qemah,
                            float* __restrict__ table) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;  // query (detection)
    int c = blockIdx.y * blockDim.y + threadIdx.y;  // candidate (lost track)
    if (q >= p.n_query || c >= p.n_cand) return;

    const float* qb = qbox + q * 4;
    float qcx = (qb[0] + qb[2]) * 0.5f;
    float qcy = (qb[1] + qb[3]) * 0.5f;
    float qw  = fmaxf(qb[2] - qb[0], 0.0f);
    float qh  = fmaxf(qb[3] - qb[1], 0.0f);

    const float* lb = clast + c * 4;
    float lcx = (lb[0] + lb[2]) * 0.5f;
    float lcy = (lb[1] + lb[3]) * 0.5f;
    int   gap = cgap[c];

    // --- spatial metrics (center_norm, iou) ---
    float dist = sqrtf((qcx - lcx) * (qcx - lcx) + (qcy - lcy) * (qcy - lcy));
    float center_norm = dist / (float)max(p.w, p.h);
    float ix1 = fmaxf(qb[0], lb[0]), iy1 = fmaxf(qb[1], lb[1]);
    float ix2 = fminf(qb[2], lb[2]), iy2 = fminf(qb[3], lb[3]);
    float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
    float area  = qw * qh;
    float larea = fmaxf(0.0f, lb[2] - lb[0]) * fmaxf(0.0f, lb[3] - lb[1]);
    float iou = inter / (area + larea - inter + 1e-6f);

    // --- physical speed gate (snapshot-independent) ---
    float speed_exceeds = 0.0f;
    if (p.max_speed_mps > 0.0f && p.person_height_m > 0.0f) {
        float lh = fmaxf(lb[3] - lb[1], 1e-3f);
        float zh = fmaxf(qh, 1e-3f);
        float dpx = dist;
        float dt_s = (float)max(gap, 1) / fmaxf(p.fps, 1e-6f);
        float px_per_m = 0.5f * (lh + zh) / p.person_height_m;
        float speed_mps = dpx / dt_s / fmaxf(px_per_m, 1e-6f);
        speed_exceeds = (speed_mps > p.max_speed_mps) ? 1.0f : 0.0f;
    }

    // --- direction-behind gate (needs snapshot velocity) ---
    float dir_behind = 0.0f;
    float kalman_d2 = -1.0f;
    if (chas[c]) {
        const float* m0 = cmean + c * 6;  // x0,x1,x3,x4,x5,x7
        float vx = m0[3], vy = m0[4];
        if (p.dir_min_cos > -1.0f) {
            float spd = sqrtf(vx * vx + vy * vy);
            if (spd >= p.dir_min_speed) {
                float dx = qcx - m0[0], dy = qcy - m0[1];
                float dd = sqrtf(dx * dx + dy * dy);
                if (dd >= 1e-3f) {
                    float cosang = (dx * vx + dy * vy) / (spd * dd);
                    if (cosang < p.dir_min_cos) dir_behind = 1.0f;
                }
            }
        }

        // --- Kalman chi-square gate (dims=2): extrapolate snapshot `delta` steps ---
        if (p.dims <= 2) {
            float m[6]; for (int i = 0; i < 6; ++i) m[i] = m0[i];
            float cc[10]; for (int i = 0; i < 10; ++i) cc[i] = ccov[c * 10 + i];
            int delta = cdelta[c];
            int steps = delta > 0 ? delta : 0;
            for (int s = 0; s < steps; ++s)
                predict_phys_step(m, cc, p.person_height_m, p.fps, p.accel_long, p.accel_lat);
            float hh = fmaxf(m[2], 1e-6f);
            float r2 = (hh / 20.0f) * (hh / 20.0f);
            float S00 = cc[0] + r2;     // P00 + r
            float S01 = cc[1];          // P01
            float S11 = cc[4] + r2;     // P11 + r
            float rx = qcx - m[0];
            float ry = qcy - m[1];

            // Cholesky whitewashing for S = L L^T.  Forward-substitute L w = y,
            // then D² = w^T w.  Default to 1e9f (invalid) with guard clauses
            // to keep the warp flat and avoid NaN from degenerates.
            kalman_d2 = 1e9f;
            float l00 = sqrtf(S00);
            if (l00 >= 1e-5f) {
                float l10 = S01 / l00;
                float inner = S11 - l10 * l10;
                if (inner >= 1e-5f) {
                    float l11 = sqrtf(inner);
                    float w0 = rx / l00;
                    float w1 = (ry - l10 * w0) / l11;
                    kalman_d2 = w0 * w0 + w1 * w1;
                }
            }
        }
    }

    // --- bidirectional midpoint bridge ---
    float bridge_dist = -1.0f;
    if (cfoot_n[c] > 0) {
        float vxl, vyl, vxc, vyc;
        regress4(cfoot + c * 8, cfoot_n[c], vxl, vyl);  // lost side (last-4)
        regress4(qfoot + q * 8, qfoot_n[q], vxc, vyc);  // query side (first-4)
        int ln = cfoot_n[c];
        float lx = cfoot[c * 8 + (ln - 1) * 2 + 0];     // lost_hist[-1]
        float ly = cfoot[c * 8 + (ln - 1) * 2 + 1];
        float cx0 = qfoot_n[q] > 0 ? qfoot[q * 8 + 0] : 0.0f;  // cand_hist[0]
        float cy0 = qfoot_n[q] > 0 ? qfoot[q * 8 + 1] : 0.0f;
        float half = gap * 0.5f;
        float xl = lx + vxl * half, yl = ly + vyl * half;
        float xc = cx0 - vxc * half, yc = cy0 - vyc * half;
        float dpx = sqrtf((xl - xc) * (xl - xc) + (yl - yc) * (yl - yc));
        float h_ref = fmaxf((cemah[c] + qemah[q]) * 0.5f, 1.0f);
        bridge_dist = dpx / h_ref;
    }

    float* out = table + ((size_t)q * p.n_cand + c) * GATE_COLS;
    out[0] = kalman_d2;
    out[1] = bridge_dist;
    out[2] = center_norm;
    out[3] = iou;
    out[4] = speed_exceeds;
    out[5] = dir_behind;
}

void launch(const GateParams& p,
            const float* query_boxes, const float* query_foot, const int* query_foot_n,
            const float* query_emah,
            const float* cand_last_box, const float* cand_mean, const float* cand_cov,
            const float* cand_foot, const int* cand_foot_n, const float* cand_emah,
            const int* cand_gap, const int* cand_delta, const int* cand_has_snap,
            float* table, void* stream) {
    if (p.n_query == 0 || p.n_cand == 0) return;
    dim3 block(16, 16);
    dim3 grid((p.n_query + block.x - 1) / block.x, (p.n_cand + block.y - 1) / block.y);
    gate_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        p, query_boxes, query_foot, query_foot_n,
        cand_last_box, cand_mean, cand_cov, cand_foot, cand_foot_n, cand_emah,
        cand_gap, cand_delta, cand_has_snap, query_emah, table);
}

// Gather tracker Kalman state into relink gate table format.
// Extracts {x0,x1,x3,x4,x5,x7} from each 8D state and the symmetric 4x4
// covariance sub-block {0,1,4,5} from each 8x8 row-major covariance.
__global__ void gather_tracker_state_kernel(
    const float* src_states, const float* src_covs,
    const int* slots, int n_cand,
    float* dst_mean, float* dst_cov) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cand) return;
    int slot = slots[j];
    if (slot < 0) return;
    int src_s = slot * 8;
    int src_c = slot * 64;
    int dst_m = j * 6;
    dst_mean[dst_m + 0] = src_states[src_s + 0];
    dst_mean[dst_m + 1] = src_states[src_s + 1];
    dst_mean[dst_m + 2] = src_states[src_s + 3];
    dst_mean[dst_m + 3] = src_states[src_s + 4];
    dst_mean[dst_m + 4] = src_states[src_s + 5];
    dst_mean[dst_m + 5] = src_states[src_s + 7];
    int dst_c = j * 10;
    dst_cov[dst_c + 0] = src_covs[src_c + 0];
    dst_cov[dst_c + 1] = src_covs[src_c + 1];
    dst_cov[dst_c + 2] = src_covs[src_c + 4];
    dst_cov[dst_c + 3] = src_covs[src_c + 5];
    dst_cov[dst_c + 4] = src_covs[src_c + 9];
    dst_cov[dst_c + 5] = src_covs[src_c + 12];
    dst_cov[dst_c + 6] = src_covs[src_c + 13];
    dst_cov[dst_c + 7] = src_covs[src_c + 36];
    dst_cov[dst_c + 8] = src_covs[src_c + 37];
    dst_cov[dst_c + 9] = src_covs[src_c + 45];
}

void gather_tracker_state(const float* src_states, const float* src_covs,
                           const int* slots, int n_cand,
                           float* dst_mean, float* dst_cov, void* stream) {
    if (n_cand <= 0) return;
    int block = 256;
    int grid = (n_cand + block - 1) / block;
    gather_tracker_state_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        src_states, src_covs, slots, n_cand, dst_mean, dst_cov);
}

// Batch cosine similarity: computes dot(q_i, c_j) for all query-candidate pairs.
__global__ void batch_dot_kernel(
    const float* queries, const float* candidates,
    int n_query, int n_cand, int dim,
    float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * n_cand;
    if (idx >= total) return;
    int i = idx / n_cand;
    int j = idx % n_cand;
    const float* q = queries + i * dim;
    const float* c = candidates + j * dim;
    float dot = 0.0f;
    for (int k = 0; k < dim; ++k) dot += q[k] * c[k];
    out[idx] = dot;
}

void batch_dot(const float* queries, const float* candidates,
               int n_query, int n_cand, int dim,
               float* out, void* stream) {
    if (n_query <= 0 || n_cand <= 0) return;
    int total = n_query * n_cand;
    int block = 256;
    int grid = (total + block - 1) / block;
    batch_dot_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        queries, candidates, n_query, n_cand, dim, out);
}

// ── relink scoring kernel ──────────────────────────────────────────
// One block per query.  Phase 1 counts gate-passing candidates.
// Phase 2 computes per-candidate joint score with dynamic weights
// derived from the gate count and candidate age, then reduces to the
// best two.  The final margin check is done on the GPU so the CPU
// path can read the answer directly.

__global__ void relink_scoring_kernel(
    const float* __restrict__ gate_tbl,    // [n_query, n_cand, 6]
    const float* __restrict__ sim_tbl,     // [n_query, n_cand]
    const int*   __restrict__ cand_ids,    // [n_cand]
    const int*   __restrict__ cand_ages,   // [n_cand]
    const float* __restrict__ cand_maha,   // [n_cand]
    ScoringParams p,
    int*   __restrict__ best_ids,          // [n_query]  output, -1 = none
    float* __restrict__ best_scores,       // [n_query]
    float* __restrict__ second_scores) {   // [n_query]

    int q = blockIdx.x;
    int tid = threadIdx.x;
    int n = p.n_cand;

    const float* gq = gate_tbl + static_cast<size_t>(q) * n * 6;
    const float* sq = sim_tbl  + static_cast<size_t>(q) * n;

    __shared__ int s_cnt;
    if (tid == 0) s_cnt = 0;
    __syncthreads();

    // Phase 1: count candidates that pass the sim gate
    for (int c = tid; c < n; c += blockDim.x) {
        if (sq[c] >= p.sim_threshold)
            atomicAdd(&s_cnt, 1);
    }
    __syncthreads();
    int n_passed = s_cnt;

    // Dynamic weight shifts (same for all candidates of this query)
    bool use_unified = p.w_sim_base > 0.0f || p.w_iou_base > 0.0f || p.w_maha_base > 0.0f;
    bool use_legacy = p.iou_weight > 0.0f || p.mahalanobis_weight > 0.0f;
    float amb_factor = (n_passed > 1) ? fminf(1.0f, float(n_passed - 1) / 8.0f) : 0.0f;
    float w_sim_dyn = p.w_sim_base + p.shift_ambiguity * amb_factor;
    float w_iou_dyn = p.w_iou_base - p.shift_ambiguity * amb_factor;

    float local_best = -1e9f, local_second = -1e9f;
    int   local_best_id = -1;

    for (int c = tid; c < n; c += blockDim.x) {
        float sim = sq[c];
        if (sim < p.sim_threshold) continue;

        float kalman_d2 = gq[c * 6 + 0];
        float iou       = gq[c * 6 + 3];

        float maha = cand_maha[c];
        float maha_score = 0.0f;
        if (p.mahalanobis_threshold > 0.0f && maha > 0.0f)
            maha_score = fmaxf(0.0f, 1.0f - maha / p.mahalanobis_threshold);

        float joint;
        if (use_unified) {
            float lost_factor = fminf(1.0f, float(cand_ages[c]) / fmaxf(1, p.ttl));
            float ws = w_sim_dyn + p.shift_lost_age * lost_factor;
            float wi = w_iou_dyn - p.shift_lost_age * lost_factor;
            float wm = p.w_maha_base;
            ws = fmaxf(0.0f, ws); wi = fmaxf(0.0f, wi); wm = fmaxf(0.0f, wm);
            float sw = ws + wi + wm;
            if (sw > 0.0f) { ws /= sw; wi /= sw; wm /= sw; }
            joint = ws * sim + wi * iou + wm * maha_score;
        } else if (use_legacy) {
            joint = sim + p.iou_weight * iou + p.mahalanobis_weight * maha_score;
        } else {
            joint = sim;
        }

        if (p.kalman_penalty_weight > 0.0f && kalman_d2 >= 0.0f)
            joint -= p.kalman_penalty_weight * (1.0f - expf(-0.5f * kalman_d2));

        if (joint > local_best) {
            local_second = local_best;
            local_best = joint;
            local_best_id = cand_ids[c];
        } else if (joint > local_second) {
            local_second = joint;
        }
    }

    // Tree reduction to find block-wide best & second best
    __shared__ float s_j[256], s_s[256];
    __shared__ int   s_i[256];
    s_j[tid] = local_best; s_s[tid] = local_second; s_i[tid] = local_best_id;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int o = tid + stride;
            if (s_j[o] > s_j[tid]) {
                s_s[tid] = fmaxf(s_j[tid], s_s[o]);
                s_j[tid] = s_j[o];
                s_i[tid] = s_i[o];
            } else {
                if (s_j[o] > s_s[tid]) s_s[tid] = s_j[o];
                if (s_s[o] > s_s[tid]) s_s[tid] = s_s[o];
            }
        }
        __syncthreads();
    }

    // Thread 0: margin check, dynamic margin, write output
    if (tid == 0) {
        if (s_i[0] >= 0) {
            float margin = p.reciprocal_margin;
            if (p.dynamic_margin_crowd > 0.0f && n_passed > 1)
                margin += p.dynamic_margin_crowd * fminf(1.0f, float(n_passed - 1) / 8.0f);
            // dynamic_margin_age needs the best candidate's age — we look it up
            // after the kernel on CPU, so only apply static+crowd margin on GPU
            if (margin > 0.0f && s_j[0] - s_s[0] < margin) {
                s_i[0] = -1;
                s_j[0] = p.sim_threshold;
            }
        }
        best_ids[q]    = s_i[0];
        best_scores[q] = s_j[0];
        second_scores[q] = s_s[0];
    }
}

void launch_relink_scoring(
    const float* gate_tbl, const float* sim_tbl,
    const int* cand_ids, const int* cand_ages, const float* cand_maha,
    int n_query, ScoringParams params,
    int* best_ids, float* best_scores, float* second_scores, void* stream) {
    if (n_query <= 0 || params.n_cand <= 0) return;
    relink_scoring_kernel<<<n_query, 256, 0, (cudaStream_t)stream>>>(
        gate_tbl, sim_tbl, cand_ids, cand_ages, cand_maha,
        params, best_ids, best_scores, second_scores);
}

}  // namespace relink_gate
}  // namespace saccade
