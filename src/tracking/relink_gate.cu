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
            float det = S00 * S11 - S01 * S01;
            // residual^T S^-1 residual
            kalman_d2 = (rx * rx * S11 - 2.0f * rx * ry * S01 + ry * ry * S00)
                        / (det != 0.0f ? det : 1e-12f);
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

}  // namespace relink_gate
}  // namespace saccade
