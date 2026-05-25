#include "tracking/tracker_gpu.hpp"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "tracking/sinkhorn.hpp"
#include "tracking/kalman_gpu.cuh"

#define checkCuda(status) do {                                   \
    if (status != cudaSuccess) {                                 \
      std::stringstream _ss;                                     \
      _ss << "CUDA Error: " << cudaGetErrorString(status)        \
          << " at " << __FILE__ << ":" << __LINE__;              \
      throw std::runtime_error(_ss.str());                       \
    }                                                            \
} while (0)

namespace saccade {

namespace {
constexpr int TRACK_EMPTY = 0;
constexpr int TRACK_TENTATIVE = 1;

bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) return default_value;
    return !(
        std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "False") == 0 ||
        std::strcmp(value, "FALSE") == 0
    );
}

float env_float_value(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) return default_value;
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value) return default_value;
    return parsed;
}
}

// --- CUDA Kernels ---

__global__ void predict_kernel(float* states, float* covs, bool* active, int* age, int max_objs, int max_age) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_objs) return;
    if (active[idx]) {
        kf_gpu::predict(states + idx * 8, covs + idx * 64);
        age[idx]++;
        if (age[idx] >= max_age) active[idx] = false;
    }
}

__global__ void gmc_kernel(float* states, float* covs, bool* active, const float* gmc, int max_objs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_objs) return;
    if (active[idx]) {
        float* x = states + idx * 8;
        float* P = covs + idx * 64;
        float H00 = gmc[0], H01 = gmc[1], H02 = gmc[2], H10 = gmc[3], H11 = gmc[4], H12 = gmc[5];
        float old_cx = x[0], old_cy = x[1];
        x[0] = H00 * old_cx + H01 * old_cy + H02;
        x[1] = H10 * old_cx + H11 * old_cy + H12;
        float old_vx = x[4], old_vy = x[5];
        x[4] = H00 * old_vx + H01 * old_vy;
        x[5] = H10 * old_vx + H11 * old_vy;
        auto rotate_cov = [&](float* p_block) {
            float p00 = p_block[0], p01 = p_block[1], p10 = p_block[8], p11 = p_block[9];
            float mp00 = H00 * p00 + H01 * p10, mp01 = H00 * p01 + H01 * p11;
            float mp10 = H10 * p00 + H11 * p10, mp11 = H10 * p01 + H11 * p11;
            p_block[0] = mp00 * H00 + mp01 * H01; p_block[1] = mp00 * H10 + mp01 * H11;
            p_block[8] = mp10 * H00 + mp11 * H01; p_block[9] = mp10 * H10 + mp11 * H11;
        };
        rotate_cov(P); rotate_cov(P + 36);
    }
}

// Kalman update for matched (track, detection) pairs — keeps d_covs_ on GPU.
// matched_pairs: [n_matched × 2] interleaved (track_slot, det_box_idx).
__global__ void kalman_update_kernel(
    float* states, float* covs, const float* det_boxes,
    const int* matched_pairs, int n_matched, float light_factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_matched) return;
    int t = matched_pairs[i * 2];
    int d = matched_pairs[i * 2 + 1];
    const float* box = det_boxes + d * 4;
    float z[4] = {
        (box[0] + box[2]) * 0.5f,
        (box[1] + box[3]) * 0.5f,
        (box[2] - box[0]) / fmaxf(box[3] - box[1], 1e-6f),
        box[3] - box[1]
    };
    kf_gpu::update(states + t * 8, covs + t * 64, z, light_factor);
}

// Initialize covariance for newly spawned tracks.
__global__ void init_covariance_kernel(float* covs, const int* new_slots, int n_new) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_new) return;
    kf_gpu::init_covariance(covs + new_slots[i] * 64);
}

__global__ void apply_detection_quality_scaling_kernel(
    float* d_scores, const float* d_boxes, int num_dets,
    int frame_w, int frame_h,
    float w_aspect, float w_center, float w_area)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_dets) return;

    float x1 = d_boxes[i * 4 + 0];
    float y1 = d_boxes[i * 4 + 1];
    float x2 = d_boxes[i * 4 + 2];
    float y2 = d_boxes[i * 4 + 3];

    float bw = fmaxf(x2 - x1, 1e-6f);
    float bh = fmaxf(y2 - y1, 1e-6f);

    // Aspect ratio: Gaussian peak at 2.5
    float aspect = bh / bw;
    float aspect_q = expf(-0.5f * powf((aspect - 2.5f) / 1.2f, 2.0f));

    // Center bias
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float cx_norm = cx / fmaxf((float)frame_w, 1.0f);
    float cy_norm = cy / fmaxf((float)frame_h, 1.0f);
    
    float center_q = fminf(fminf(cx_norm, 1.0f - cx_norm), fminf(cy_norm, 1.0f - cy_norm)) * 4.0f;
    center_q = fmaxf(0.0f, fminf(1.0f, center_q));

    // Area ratio: Gaussian peak at 0.01
    float area_ratio = (bw * bh) / fmaxf((float)(frame_w * frame_h), 1.0f);
    float area_q = expf(-0.5f * powf((area_ratio - 0.01f) / 0.01f, 2.0f));

    float quality = w_aspect * aspect_q + w_center * center_q + w_area * area_q;
    d_scores[i] *= quality;
}

namespace kernel {

// Compute per-track S^-1 (innovation covariance inverse) after predict+GMC, before association.
// Stores 16 floats (row-major 4x4) per track in s_inv_out.
__global__ void compute_innovation_sinv_kernel(
    const float* states, const float* covs, const bool* active,
    float* s_inv_out, int max_objs)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= max_objs) return;
    float* s_inv = s_inv_out + t * 16;
    if (!active[t]) {
        for (int i = 0; i < 16; ++i) s_inv[i] = 0.0f;
        return;
    }
    kf_gpu::compute_S_inv(states + t * 8, covs + t * 64, s_inv);
}

// Returns Mahalanobis^2 between a detection (x1,y1,x2,y2) and a track predicted state.
// If homography is provided, projects bottom-center to ground plane first (2D MMD).
__device__ __forceinline__ float mahal_sq_det(
    const float* trk_state, const float* det_box, const float* s_inv, const float* homography)
{
    float det_h  = fmaxf(det_box[3] - det_box[1], 1e-6f);
    float det_cx = (det_box[0] + det_box[2]) * 0.5f;
    float det_cy = (det_box[1] + det_box[3]) * 0.5f;
    
    if (homography) {
        // 2D MMD: Project bottom center to ground plane
        auto project = [&](float x, float y, float& ox, float& oy) {
            float z = homography[6] * x + homography[7] * y + homography[8];
            float inv_z = 1.0f / (fmaxf(std::abs(z), 1e-6f));
            ox = (homography[0] * x + homography[1] * y + homography[2]) * inv_z;
            oy = (homography[3] * x + homography[4] * y + homography[5]) * inv_z;
        };
        
        float t_gx, t_gy, d_gx, d_gy;
        project(trk_state[0], trk_state[1] + trk_state[3] * 0.5f, t_gx, t_gy);
        project(det_cx, det_box[3], d_gx, d_gy);
        
        // Simple Euclidean distance on ground plane as fallback/complement for gating
        // In a full MMD, we'd need ground-plane KF. For now, we use a hybrid approach:
        // If homography is provided, we use the ground-plane L2 distance for the center part of Mahalanobis.
        float dx = t_gx - d_gx;
        float dy = t_gy - d_gy;
        return (dx * dx + dy * dy) * 0.01f; // Scaled L2 for gating
    }

    float det_ar = (det_box[2] - det_box[0]) / det_h;
    float innov[4] = {det_cx - trk_state[0], det_cy - trk_state[1],
                      det_ar - trk_state[2], det_h  - trk_state[3]};
    float d2 = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float tmp = 0.0f;
        for (int j = 0; j < 4; ++j) tmp += s_inv[i*4+j] * innov[j];
        d2 += tmp * innov[i];
    }
    return d2;
}

// Counts per-track how many detections pass the Stage 1 gate:
//   IoU > iou_gate  OR  Mahalanobis^2 < maha_gate
__global__ void count_stage1_candidates_kernel(
    const float* trk_states, const float* det_boxes,
    const bool* trk_active, int* candidate_count,
    const float* trk_s_inv, const float* homography,
    int n_trk, int n_det, float iou_gate, float maha_gate)
{
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_trk || d >= n_det || !trk_active[t]) return;

    const float* st = trk_states + t * 8;
    float tw = st[2] * st[3], th = st[3];
    float b1_x1 = st[0] - tw * 0.5f, b1_y1 = st[1] - th * 0.5f;
    float b1_x2 = st[0] + tw * 0.5f, b1_y2 = st[1] + th * 0.5f;
    const float* b2 = det_boxes + d * 4;

    float ix1 = fmaxf(b1_x1, b2[0]), iy1 = fmaxf(b1_y1, b2[1]);
    float ix2 = fminf(b1_x2, b2[2]), iy2 = fminf(b1_y2, b2[3]);
    float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
    float area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);
    float area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    float iou = inter / (area1 + area2 - inter + 1e-6f);

    bool pass = (iou > iou_gate);
    if (!pass && trk_s_inv) {
        float d2 = mahal_sq_det(st, b2, trk_s_inv + t * 16, homography);
        pass = (d2 < maha_gate);
    }
    if (pass) atomicAdd(&candidate_count[t], 1);
}

// OA-SORT: per-track occlusion coefficient = max IoU of predicted box with all other active
// track predicted boxes. Cheap O(n²) kernel; n_active is typically ≤ 128.
__global__ void compute_track_occlusion_kernel(
    const float* states, const bool* active, float* occ_coeff, int max_objs)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= max_objs) return;
    if (!active[t]) { occ_coeff[t] = 0.0f; return; }

    const float* st = states + t * 8;
    float tw = st[2] * st[3], th = st[3];
    float tx1 = st[0] - tw * 0.5f, ty1 = st[1] - th * 0.5f;
    float tx2 = st[0] + tw * 0.5f, ty2 = st[1] + th * 0.5f;
    float t_area = (tx2 - tx1) * (ty2 - ty1);

    float max_occ = 0.0f;
    for (int j = 0; j < max_objs; ++j) {
        if (j == t || !active[j]) continue;
        const float* sj = states + j * 8;
        float jw = sj[2] * sj[3], jh = sj[3];
        float jx1 = sj[0] - jw * 0.5f, jy1 = sj[1] - jh * 0.5f;
        float jx2 = sj[0] + jw * 0.5f, jy2 = sj[1] + jh * 0.5f;
        float j_area = (jx2 - jx1) * (jy2 - jy1);
        float ix1 = fmaxf(tx1, jx1), iy1 = fmaxf(ty1, jy1);
        float ix2 = fminf(tx2, jx2), iy2 = fminf(ty2, jy2);
        float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
        if (inter <= 0.0f) continue;
        float iou = inter / (t_area + j_area - inter + 1e-6f);
        max_occ = fmaxf(max_occ, iou);
    }
    occ_coeff[t] = max_occ;
}

// Two-stage conditional cost matrix.
// Stage 1 gate: IoU > iou_gate OR Mahalanobis^2 < maha_gate; else hard reject (cost=1).
// Stage 2: if candidate_count[t] >= 2 AND has_clean_embedding[t] AND det embedding valid (norm>0.25):
//            cost = 1 - (0.55*CosSim + 0.30*IoU + 0.15*det_score)
//          else:
//            cost = 1 - IoU  (stable IoU-only fallback — also used for budget-zero detections)
// OC-SORT: optional vel_dir_weight penalises matches where detection direction opposes track velocity.
// OA-SORT: optional oao_tau adds occlusion penalty (cost += tau * occ_coeff[t]).
__global__ void compute_conditional_cost_kernel(
    const float* trk_states, const float* det_boxes,
    const float* trk_embeds, const float* det_embeds,
    const float* det_scores,
    const int* candidate_count, const bool* has_clean_embedding,
    const float* trk_s_inv, const float* homography,
    float* cost_matrix,
    int n_trk, int n_det, int embed_dim, float iou_gate, float maha_gate,
    float vel_dir_weight, float fuse_score_weight,
    float cost_cos_w, float cost_iou_w, float cost_score_w,
    const float* d_occ_coeff, float oao_tau)
{
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_trk || d >= n_det) return;

    const float* st = trk_states + t * 8;
    float tw = st[2] * st[3], th = st[3];
    float b1_x1 = st[0] - tw * 0.5f, b1_y1 = st[1] - th * 0.5f;
    float b1_x2 = st[0] + tw * 0.5f, b1_y2 = st[1] + th * 0.5f;
    const float* b2 = det_boxes + d * 4;

    float ix1 = fmaxf(b1_x1, b2[0]), iy1 = fmaxf(b1_y1, b2[1]);
    float ix2 = fminf(b1_x2, b2[2]), iy2 = fminf(b1_y2, b2[3]);
    float inter = fmaxf(0.0f, ix2 - ix1) * fmaxf(0.0f, iy2 - iy1);
    float area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);
    float area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    float iou = inter / (area1 + area2 - inter + 1e-6f);

    bool pass_iou = (iou > iou_gate);
    if (!pass_iou) {
        bool pass_maha = trk_s_inv && (mahal_sq_det(st, b2, trk_s_inv + t * 16, homography) < maha_gate);
        if (!pass_maha) {
            cost_matrix[t * n_det + d] = 1.0f;
            return;
        }
    }

    float ds = det_scores ? det_scores[d] : 0.5f;
    float fused_iou = iou * (1.0f - fuse_score_weight * ds);
    float cost;
    bool try_appearance = (candidate_count[t] >= 2 && has_clean_embedding[t] && trk_embeds && det_embeds);
    if (try_appearance) {
        const float* e1 = trk_embeds + t * embed_dim;
        const float* e2 = det_embeds + d * embed_dim;
        float cos_sim = 0.0f, norm_sq = 0.0f;
        for (int k = 0; k < embed_dim; ++k) {
            cos_sim += e1[k] * e2[k];
            norm_sq += e2[k] * e2[k];
        }
        if (norm_sq > 0.0625f) {  // det embedding valid (non-zero budget slot)
            cos_sim = fmaxf(0.0f, cos_sim);
            cost = 1.0f - (cost_cos_w * cos_sim + cost_iou_w * fused_iou + cost_score_w * ds);
        } else {
            cost = 1.0f - fused_iou;
        }
    } else {
        cost = 1.0f - fused_iou;
    }

    // OC-SORT velocity direction penalty
    if (vel_dir_weight > 0.0f) {
        float vx = st[4], vy = st[5];
        float vel_sq = vx * vx + vy * vy;
        if (vel_sq > 1.0f) {
            float det_cx = (b2[0] + b2[2]) * 0.5f, det_cy = (b2[1] + b2[3]) * 0.5f;
            float dx = det_cx - st[0], dy = det_cy - st[1];
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq > 1e-6f) {
                float cos_dir = (vx * dx + vy * dy) / sqrtf(vel_sq * dist_sq);
                cost += vel_dir_weight * fmaxf(0.0f, -cos_dir);
            }
        }
    }

    // OA-SORT OAO: tracks occluded by other tracks get a cost penalty to prevent cost confusion
    if (oao_tau > 0.0f && d_occ_coeff) {
        cost = fminf(1.0f, cost + oao_tau * d_occ_coeff[t]);
    }

    cost_matrix[t * n_det + d] = fminf(1.0f, fmaxf(0.0f, cost));
}

__device__ __forceinline__ void merge_top3_fixed(float* a_v, int* a_idx, const float* b_v, const int* b_idx) {
    float res_v[3]; int res_idx[3];
    int i = 0, j = 0;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        if (a_v[i] >= b_v[j]) { res_v[k] = a_v[i]; res_idx[k] = a_idx[i]; i++; }
        else { res_v[k] = b_v[j]; res_idx[k] = b_idx[j]; j++; }
    }
    #pragma unroll
    for (int k = 0; k < 3; ++k) { a_v[k] = res_v[k]; a_idx[k] = res_idx[k]; }
}

__global__ void compute_cost_matrix_kernel(
    const float* trk_states, const float* det_boxes,
    const float* trk_embeds, const float* det_embeds,
    float* cost_matrix, int n_trk, int n_det, int embed_dim,
    float reid_weight, float cos_threshold, float gate_threshold)
{
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_trk || d >= n_det) return;

    const float* st = trk_states + t * 8;
    float tw = st[2] * st[3], th = st[3];
    float b1[4] = {st[0] - tw/2.0f, st[1] - th/2.0f, st[0] + tw/2.0f, st[1] + th/2.0f};
    const float* b2 = det_boxes + d * 4;

    float dx = st[0] - (b2[0] + b2[2]) * 0.5f;
    float dy = st[1] - (b2[1] + b2[3]) * 0.5f;
    float dist_sq = dx*dx + dy*dy;
    float gate_sq = gate_threshold * gate_threshold;

    if (dist_sq > gate_sq) {
        cost_matrix[t * n_det + d] = 1.0f; return;
    }

    float x1 = fmaxf(b1[0], b2[0]), y1 = fmaxf(b1[1], b2[1]), x2 = fminf(b1[2], b2[2]), y2 = fminf(b1[3], b2[3]);
    float inter = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]), area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    float iou = inter / (area1 + area2 - inter + 1e-6f);

    float cos_sim = 0.0f;
    if (trk_embeds && det_embeds) {
        const float* e1 = trk_embeds + t * embed_dim;
        const float* e2 = det_embeds + d * embed_dim;
        for (int k = 0; k < embed_dim; ++k) cos_sim += e1[k] * e2[k];
    }

    float cost = 1.0f - iou;
    if (cos_sim > cos_threshold) {
        // Distance decay: similarity decreases as distance increases towards gate_threshold
        float dist_decay = expf(-2.0f * (dist_sq / gate_sq));
        float decayed_sim = cos_sim * dist_decay;
        cost = (1.0f - reid_weight) * cost + reid_weight * (1.0f - decayed_sim);
    }
    cost_matrix[t * n_det + d] = cost;
}

// 128-thread TopK kernel with shared-memory tree reduction.
// Replaces the 1-warp design to improve GPU occupancy and cut scheduling jitter.
// Shared memory layout: s_vals[3][128] + s_idxs[3][128] = 3 KiB per block.
__global__ void fused_sinkhorn_topk_kernel(
    const float* cost_matrix, const float* det_scores, const float* det_boxes,
    const int* trk_states, const bool* trk_active, const int* trk_to_det,
    int n_trk, int n_det, float lambda, float max_cost,
    float min_det_score, float max_det_score, int required_trk_state,
    const int* det_to_trk,
    int* out_indices, float* out_probs)
{
    __shared__ float s_vals[3][128];
    __shared__ int   s_idxs[3][128];

    int t = blockIdx.x; if (t >= n_trk) return;
    int tid = threadIdx.x;

    float lv0 = -1e9f, lv1 = -1e9f, lv2 = -1e9f;
    int   li0 = -1,    li1 = -1,    li2 = -1;

    bool valid_trk = trk_active[t] && (trk_to_det[t] == -1);
    if (required_trk_state != -1 && trk_states[t] != required_trk_state) valid_trk = false;

    if (valid_trk) {
        for (int d = tid; d < n_det; d += 128) {
            if (det_to_trk && det_to_trk[d] != -1) continue;
            float score = det_scores[d];
            if (score < min_det_score || score >= max_det_score) continue;
            
            float cost = cost_matrix[t * n_det + d];
            if (cost > max_cost) continue;

            // ADR 017: Quality-Aware Sinkhorn Prior (Winning Strategy: v2_aspect_only_soft)
            float aspect_penalty = 1.0f;
            if (det_boxes) {
                const float* b2 = det_boxes + d * 4;
                float aspect = (b2[2] - b2[0]) / (b2[3] - b2[1] + 1e-6f);
                // Pedestrian aspect ratio penalty: penalize abnormal shapes (e.g., highly occluded / truncated)
                // Typical pedestrian is around 0.3~0.5.
                if (aspect > 0.8f) aspect_penalty = fmaxf(0.5f, 1.0f - (aspect - 0.8f));
                else if (aspect < 0.15f) aspect_penalty = fmaxf(0.5f, 1.0f - (0.15f - aspect) * 5.0f);
            }

            float p = expf(-lambda * cost) * aspect_penalty;
            if (p > lv0) {
                lv2 = lv1; li2 = li1; lv1 = lv0; li1 = li0; lv0 = p; li0 = d;
            } else if (p > lv1) {
                lv2 = lv1; li2 = li1; lv1 = p; li1 = d;
            } else if (p > lv2) {
                lv2 = p; li2 = d;
            }
        }
    }

    s_vals[0][tid] = lv0; s_idxs[0][tid] = li0;
    s_vals[1][tid] = lv1; s_idxs[1][tid] = li1;
    s_vals[2][tid] = lv2; s_idxs[2][tid] = li2;
    __syncthreads();

    // Tree reduction: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    #pragma unroll
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float av[3], bv[3]; int ai[3], bi[3];
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                av[i] = s_vals[i][tid];          ai[i] = s_idxs[i][tid];
                bv[i] = s_vals[i][tid + stride]; bi[i] = s_idxs[i][tid + stride];
            }
            merge_top3_fixed(av, ai, bv, bi);
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                s_vals[i][tid] = av[i]; s_idxs[i][tid] = ai[i];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            out_indices[t * 3 + i] = s_idxs[i][0];
            out_probs[t * 3 + i]   = s_vals[i][0];
        }
    }
}

// Two-level auction with shared-memory price cache.
// Level 1: intra-block bids resolved via shared-memory atomicMax (no L2 traffic).
// Level 2: only the block winner escalates to global atomicMax (far fewer atomics).
// Dynamic shared memory: n_det floats (caller passes n_det * sizeof(float)).
// Prices are always non-negative so reinterpreting as int preserves ordering.
__global__ void parallel_auction_shmem_kernel(
    const int* topk_indices, const float* topk_probs,
    uint64_t* g_prices, int* trk_to_det, int* det_to_trk,
    int n_trk, int n_det, int K, float epsilon)
{
    extern __shared__ uint64_t s_prices_u64[];

    for (int d = threadIdx.x; d < n_det; d += blockDim.x)
        s_prices_u64[d] = g_prices[d];
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (t < n_trk) && (trk_to_det[t] == -1);

    int   best_det = -1;
    float bid      = -1e9f;

    if (active) {
        float best_val = -1e9f, second_best_val = -1e9f;
        for (int k = 0; k < K; ++k) {
            int d = topk_indices[t * K + k];
            if (d < 0 || d >= n_det) continue;
            float current_price = __int_as_float((int)(s_prices_u64[d] >> 32));
            float val = topk_probs[t * K + k] - current_price;
            if (val > best_val) {
                second_best_val = best_val; best_val = val; best_det = d;
            } else if (val > second_best_val) {
                second_best_val = val;
            }
        }
        if (best_det >= 0) {
            float inc = (second_best_val <= -1e8f) ? epsilon
                                                    : (best_val - second_best_val + epsilon);
            float current_price = __int_as_float((int)(s_prices_u64[best_det] >> 32));
            bid = current_price + inc;
            
            uint32_t bid_float_bits = __float_as_uint(bid);
            uint32_t tie_breaker = n_trk - t;
            uint64_t bid_u64 = ((uint64_t)bid_float_bits << 32) | (uint64_t)tie_breaker;

            // Level 1: intra-block conflict resolution
            atomicMax((unsigned long long*)&s_prices_u64[best_det], (unsigned long long)bid_u64);
        }
    }
    __syncthreads();

    // Level 2: block winner commits to global memory
    if (best_det >= 0) {
        uint32_t bid_float_bits = __float_as_uint(bid);
        uint32_t tie_breaker = n_trk - t;
        uint64_t bid_u64 = ((uint64_t)bid_float_bits << 32) | (uint64_t)tie_breaker;

        if (s_prices_u64[best_det] == bid_u64) {
            unsigned long long prev = atomicMax((unsigned long long*)&g_prices[best_det], (unsigned long long)bid_u64);
            if (bid_u64 > prev) {
                trk_to_det[t] = best_det;
                det_to_trk[best_det] = t;
            }
        }
    }
}


__global__ void track_state_update_pre_kernel(
    bool* active, int* state, int* hit_streak, int* req_confirm_streak, float* score_sum, int max_objs) 
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= max_objs) return;
    if (!active[t]) {
        state[t] = 0; // TRACK_EMPTY
        hit_streak[t] = 0;
        req_confirm_streak[t] = 0;
        score_sum[t] = 0.0f;
    } else if (state[t] == 0) {
        state[t] = 2; // TRACK_CONFIRMED
    }
}

__global__ void track_state_update_post_kernel(
    bool* active, int* state, int* age, float* trk_scores, int* trk_classes,
    int* hit_streak, int* req_confirm_streak, float* score_sum,
    const int* trk_to_det, const float* det_scores, const int* det_classes,
    int confirm_streak, float confirm_score_thresh, int max_objs) 
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= max_objs || !active[t]) return;

    int d = trk_to_det[t];
    if (d >= 0) {
        age[t] = 0;
        trk_scores[t] = det_scores[d];
        trk_classes[t] = det_classes[d];
        hit_streak[t] += 1;
        score_sum[t] += det_scores[d];
        
        int req = req_confirm_streak[t] > 0 ? req_confirm_streak[t] : confirm_streak;
        if (state[t] == 1 && hit_streak[t] >= req && (score_sum[t] / max(hit_streak[t], 1)) >= confirm_score_thresh) {
            state[t] = 2;
        }
    } else {
        if (state[t] == 1) { // TRACK_TENTATIVE
            active[t] = false;
            state[t] = 0; // TRACK_EMPTY
            hit_streak[t] = 0;
            req_confirm_streak[t] = 0;
            score_sum[t] = 0.0f;
        } else {
            hit_streak[t] = 0;
            score_sum[t] = 0.0f;
        }
    }
}

__global__ void inline_kalman_update_kernel(
    float* states, float* covs, const float* det_boxes,
    const int* trk_to_det, const bool* active, int max_objs, float light_factor,
    const float* det_scores, bool nsa_kalman, float r_scale)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= max_objs || !active[t]) return;
    int d = trk_to_det[t];
    if (d < 0) return;

    const float* box = det_boxes + d * 4;
    float bh = fmaxf(box[3] - box[1], 1e-6f);
    float z[4] = {
        (box[0] + box[2]) / 2.0f,
        (box[1] + box[3]) / 2.0f,
        (box[2] - box[0]) / bh,
        bh
    };
    float nsa_mult = 1.0f;
    if (nsa_kalman && det_scores) {
        float s = det_scores[d];
        float q = 1.0f - s;
        nsa_mult = fmaxf(0.05f, q * q);
    }
    kf_gpu::update(states + t * 8, covs + t * 64, z, light_factor, nsa_mult, r_scale);
}
} // namespace kernel

// ── M2 GPU lifecycle helpers ───────────────────────────────────────────────

__global__ void collect_free_slots_kernel(
    const bool* active, int max_objs, int* free_slots, int* n_free)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int count = 0;
    for (int i = 0; i < max_objs; ++i) {
        if (!active[i]) {
            free_slots[count++] = i;
        }
    }
    *n_free = count;
}

__global__ void spawn_new_tracks_kernel(
    const int*   d_det_to_trk,
    const float* d_det_boxes,
    const float* d_det_scores,
    const int*   d_det_classes,
    int n_det, float new_track_thresh,
    const int* d_free_slots, const int* d_n_free,
    bool*  d_active,  float* d_states, int* d_state,
    int*   d_track_ids, int* d_age,
    float* d_trk_scores, int* d_classes,
    int*   d_hit_streak, int* d_confirm_req, float* d_score_sum,
    int* d_track_id_ctr, int* d_slot_cursor,
    int confirm_streak, float birth_low_score_thresh,
    int max_objs, float birth_prox_norm_thresh)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int cursor = *d_slot_cursor;
    int n_free = *d_n_free;
    int id_ctr = *d_track_id_ctr;

    for (int d = 0; d < n_det; ++d) {
        if (d_det_to_trk[d] >= 0) continue;
        if (d_det_scores[d] < new_track_thresh) continue;

        // Proximity birth gate: suppress if center is within birth_prox_norm_thresh
        // box-heights of any confirmed track. Targets ghost tracks (shadows) that
        // are spatially close to a real person but have IoU < NMS threshold.
        if (birth_prox_norm_thresh > 0.0f) {
            const float* det_box = d_det_boxes + d * 4;
            float det_cx = (det_box[0] + det_box[2]) * 0.5f;
            float det_cy = (det_box[1] + det_box[3]) * 0.5f;
            float det_h  = det_box[3] - det_box[1];
            bool too_close = false;
            for (int t = 0; t < max_objs && !too_close; ++t) {
                if (!d_active[t] || d_state[t] != 2) continue;  // CONFIRMED only
                float trk_cx = d_states[t * 8 + 0];
                float trk_cy = d_states[t * 8 + 1];
                float trk_h  = d_states[t * 8 + 3];
                float dx = det_cx - trk_cx;
                float dy = det_cy - trk_cy;
                float dist = sqrtf(dx * dx + dy * dy);
                float ref_h = fmaxf(fmaxf(det_h, trk_h), 1.0f);
                too_close = (dist < birth_prox_norm_thresh * ref_h);
            }
            if (too_close) continue;
        }

        if (cursor >= n_free) break;
        int slot = d_free_slots[cursor++];

        const float* box = d_det_boxes + d * 4;
        float bh = fmaxf(box[3] - box[1], 1e-6f);
        d_states[slot*8+0] = (box[0] + box[2]) * 0.5f;
        d_states[slot*8+1] = (box[1] + box[3]) * 0.5f;
        d_states[slot*8+2] = (box[2] - box[0]) / bh;
        d_states[slot*8+3] = bh;
        d_states[slot*8+4] = 0.0f; d_states[slot*8+5] = 0.0f; d_states[slot*8+6] = 0.0f; d_states[slot*8+7] = 0.0f;

        d_track_ids[slot]   = id_ctr++;
        d_age[slot]         = 0;
        d_state[slot]       = 1; // TRACK_TENTATIVE
        d_hit_streak[slot]  = 1;
        d_confirm_req[slot] = (birth_low_score_thresh > 0.0f && d_det_scores[d] < birth_low_score_thresh)
            ? (confirm_streak + 1) : 0;
        d_score_sum[slot]   = d_det_scores[d];
        d_trk_scores[slot]  = d_det_scores[d];
        d_classes[slot]     = d_det_classes[d];
        d_active[slot]      = true;
    }

    *d_slot_cursor = cursor;
    *d_track_id_ctr = id_ctr;
}

// Initialise covariance for every slot that is active+tentative+hit_streak==1 (freshly spawned).
__global__ void init_covariance_if_new_kernel(
    const bool* active, const int* state, const int* hit_streak,
    float* covs, int max_objs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_objs) return;
    if (!active[i] || state[i] != 1 || hit_streak[i] != 1) return;
    kf_gpu::init_covariance(covs + i * 64);
}

__global__ void compact_results_kernel(
    const bool*  d_active,
    const int*   d_state,
    const int*   d_age,
    const float* d_states,
    const int*   d_track_ids,
    const float* d_scores,
    const int*   d_classes,
    const int*   d_trk_to_det,
    int max_objs,
    float* out_boxes, float* out_scores,
    int* out_ids, int* out_classes, int* out_det_idx,
    int* out_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int k = 0;
    for (int i = 0; i < max_objs; ++i) {
        if (!d_active[i] || d_state[i] != 2 || d_age[i] != 0) continue;
        float cx = d_states[i*8+0], cy = d_states[i*8+1];
        float a  = d_states[i*8+2], h  = d_states[i*8+3], w = a * h;
        out_boxes[k*4+0] = cx - w * 0.5f;
        out_boxes[k*4+1] = cy - h * 0.5f;
        out_boxes[k*4+2] = cx + w * 0.5f;
        out_boxes[k*4+3] = cy + h * 0.5f;
        out_scores[k]   = d_scores[i];
        out_ids[k]      = d_track_ids[i];
        out_classes[k]  = d_classes[i];
        out_det_idx[k]  = d_trk_to_det[i];
        k++;
    }
    *out_count = k;
}

// ── M1 GPU compaction helpers ──────────────────────────────────────────────

// Stable argsort keys: upper 32 bits = raw score bits (positive float → monotone uint32),
// lower 32 bits = (n-1-i) so equal-score ties preserve original order (lower index first).
// SortKeysDescending → larger key first → higher score first, lower index for ties.
__global__ void build_sort_keys_kernel(const float* scores, int n, uint64_t* keys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t sb;
    memcpy(&sb, &scores[i], sizeof(float));
    keys[i] = ((uint64_t)sb << 32) | (uint64_t)(uint32_t)(n - 1 - i);
}

// Decode original index from sorted compound key: index = n - 1 - lower32(key).
__global__ void decode_sort_order_kernel(const uint64_t* sorted_keys, int64_t* order, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    order[i] = (int64_t)((n - 1) - (int)(uint32_t)sorted_keys[i]);
}

__global__ void gather_compact3_kernel(
    const float* __restrict__ src_boxes,
    const float* __restrict__ src_scores,
    const int*   __restrict__ src_classes,
    float* dst_boxes, float* dst_scores, int* dst_classes,
    const int* __restrict__ indices, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = indices[i];
    dst_boxes[i*4+0] = src_boxes[k*4+0];
    dst_boxes[i*4+1] = src_boxes[k*4+1];
    dst_boxes[i*4+2] = src_boxes[k*4+2];
    dst_boxes[i*4+3] = src_boxes[k*4+3];
    dst_scores[i]    = src_scores[k];
    dst_classes[i]   = src_classes[k];
}

__global__ void gather_compact3_counted_kernel(
    const float* __restrict__ src_boxes,
    const float* __restrict__ src_scores,
    const int*   __restrict__ src_classes,
    float* dst_boxes, float* dst_scores, int* dst_classes,
    const int* __restrict__ indices, const int* __restrict__ count_ptr, int max_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_n) return;
    const int n = *count_ptr;
    if (i < n) {
        int k = indices[i];
        dst_boxes[i*4+0] = src_boxes[k*4+0];
        dst_boxes[i*4+1] = src_boxes[k*4+1];
        dst_boxes[i*4+2] = src_boxes[k*4+2];
        dst_boxes[i*4+3] = src_boxes[k*4+3];
        dst_scores[i]    = src_scores[k];
        dst_classes[i]   = src_classes[k];
    } else {
        dst_boxes[i*4+0] = 0.0f;
        dst_boxes[i*4+1] = 0.0f;
        dst_boxes[i*4+2] = 0.0f;
        dst_boxes[i*4+3] = 0.0f;
        dst_scores[i]    = 0.0f;
        dst_classes[i]   = -1;
    }
}

__global__ void gather_compact4_kernel(
    const float* __restrict__ src_boxes,
    const float* __restrict__ src_scores,
    const int*   __restrict__ src_classes,
    const bool*  __restrict__ src_suspect,
    float* dst_boxes, float* dst_scores, int* dst_classes, bool* dst_suspect,
    const int* __restrict__ indices, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = indices[i];
    dst_boxes[i*4+0] = src_boxes[k*4+0];
    dst_boxes[i*4+1] = src_boxes[k*4+1];
    dst_boxes[i*4+2] = src_boxes[k*4+2];
    dst_boxes[i*4+3] = src_boxes[k*4+3];
    dst_scores[i]    = src_scores[k];
    dst_classes[i]   = src_classes[k];
    dst_suspect[i]   = src_suspect[k];
}

__global__ void gather_compact4_counted_kernel(
    const float* __restrict__ src_boxes,
    const float* __restrict__ src_scores,
    const int*   __restrict__ src_classes,
    const bool*  __restrict__ src_suspect,
    float* dst_boxes, float* dst_scores, int* dst_classes, bool* dst_suspect,
    const int* __restrict__ indices, const int* __restrict__ count_ptr, int max_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_n) return;
    const int n = *count_ptr;
    if (i < n) {
        int k = indices[i];
        dst_boxes[i*4+0] = src_boxes[k*4+0];
        dst_boxes[i*4+1] = src_boxes[k*4+1];
        dst_boxes[i*4+2] = src_boxes[k*4+2];
        dst_boxes[i*4+3] = src_boxes[k*4+3];
        dst_scores[i]    = src_scores[k];
        dst_classes[i]   = src_classes[k];
        dst_suspect[i]   = src_suspect[k];
    } else {
        dst_boxes[i*4+0] = 0.0f;
        dst_boxes[i*4+1] = 0.0f;
        dst_boxes[i*4+2] = 0.0f;
        dst_boxes[i*4+3] = 0.0f;
        dst_scores[i]    = 0.0f;
        dst_classes[i]   = -1;
        dst_suspect[i]   = false;
    }
}

__global__ void copy_bool_counted_kernel(
    const bool* __restrict__ src,
    bool* dst,
    const int* __restrict__ count_ptr,
    int max_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_n) return;
    const int n = *count_ptr;
    dst[i] = (i < n) ? src[i] : false;
}

void gather_compact3_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes,
    float* dst_boxes, float* dst_scores, int* dst_classes,
    const int* indices, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    const int thr = 256, blk = (n + thr - 1) / thr;
    gather_compact3_kernel<<<blk, thr, 0, stream>>>(
        src_boxes, src_scores, src_classes,
        dst_boxes, dst_scores, dst_classes,
        indices, n);
}

void gather_compact3_counted_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes,
    float* dst_boxes, float* dst_scores, int* dst_classes,
    const int* indices, const int* count_ptr, int max_n, cudaStream_t stream)
{
    if (max_n <= 0) return;
    const int thr = 256, blk = (max_n + thr - 1) / thr;
    gather_compact3_counted_kernel<<<blk, thr, 0, stream>>>(
        src_boxes, src_scores, src_classes,
        dst_boxes, dst_scores, dst_classes,
        indices, count_ptr, max_n);
}

void gather_compact4_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes, const bool* src_suspect,
    float* dst_boxes, float* dst_scores, int* dst_classes, bool* dst_suspect,
    const int* indices, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    const int thr = 256, blk = (n + thr - 1) / thr;
    gather_compact4_kernel<<<blk, thr, 0, stream>>>(
        src_boxes, src_scores, src_classes, src_suspect,
        dst_boxes, dst_scores, dst_classes, dst_suspect,
        indices, n);
}

void gather_compact4_counted_cuda(
    const float* src_boxes, const float* src_scores, const int* src_classes, const bool* src_suspect,
    float* dst_boxes, float* dst_scores, int* dst_classes, bool* dst_suspect,
    const int* indices, const int* count_ptr, int max_n, cudaStream_t stream)
{
    if (max_n <= 0) return;
    const int thr = 256, blk = (max_n + thr - 1) / thr;
    gather_compact4_counted_kernel<<<blk, thr, 0, stream>>>(
        src_boxes, src_scores, src_classes, src_suspect,
        dst_boxes, dst_scores, dst_classes, dst_suspect,
        indices, count_ptr, max_n);
}


__global__ void penalize_suspect_scores_kernel(
    float* scores, const bool* suspect, const int* count_ptr, float penalty_score, int max_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int count = *count_ptr;
    if (i < count && i < max_n) {
        if (suspect[i] && scores[i] > penalty_score) {
            scores[i] = penalty_score;
        }
    }
}

void penalize_suspect_scores_cuda(
    float* scores, const bool* suspect, const int* count_ptr, float penalty_score, int max_n, cudaStream_t stream)
{
    if (max_n <= 0) return;
    const int thr = 256, blk = (max_n + thr - 1) / thr;
    penalize_suspect_scores_kernel<<<blk, thr, 0, stream>>>(scores, suspect, count_ptr, penalty_score, max_n);
}

void copy_bool_counted_cuda(
    const bool* src, bool* dst, const int* count_ptr, int max_n, cudaStream_t stream)
{
    if (max_n <= 0) return;
    const int thr = 256, blk = (max_n + thr - 1) / thr;
    copy_bool_counted_kernel<<<blk, thr, 0, stream>>>(src, dst, count_ptr, max_n);
}

size_t argsort_scores_descending_bytes(int n)
{
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeysDescending(
        nullptr, bytes,
        static_cast<const uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr), n);
    return bytes;
}

void argsort_scores_descending_cuda(
    const float* d_scores, int n,
    int64_t* d_order_out, uint64_t* d_keys_in, uint64_t* d_keys_out,
    void* d_cub_tmp, size_t cub_tmp_bytes, cudaStream_t stream)
{
    if (n <= 0) return;
    const int thr = 256, blk = (n + thr - 1) / thr;
    build_sort_keys_kernel<<<blk, thr, 0, stream>>>(d_scores, n, d_keys_in);
    size_t bytes = cub_tmp_bytes;
    cub::DeviceRadixSort::SortKeysDescending(
        d_cub_tmp, bytes, d_keys_in, d_keys_out, n, 0, 64, stream);
    decode_sort_order_kernel<<<blk, thr, 0, stream>>>(d_keys_out, d_order_out, n);
}

class GPUByteTracker::Impl {
public:
    Impl(int max_objects, int embedding_dim)
        : max_objs_(max_objects), embed_dim_(embedding_dim) {
        enable_quality_scaling_ = false;
        q_w_aspect_ = 0.50f;
        q_w_center_ = 0.30f;
        q_w_area_ = 0.20f;
        frame_w_ = 1920;
        frame_h_ = 1080;

        checkCuda(cudaMalloc(&d_states_, max_objs_ * 8 * sizeof(float)));
        checkCuda(cudaMalloc(&d_covs_, max_objs_ * 64 * sizeof(float)));
        checkCuda(cudaMalloc(&d_active_, max_objs_ * sizeof(bool)));
        checkCuda(cudaMalloc(&d_age_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_scores_, max_objs_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_classes_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_track_ids_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_features_, max_objs_ * embed_dim_ * sizeof(float)));
        d_features_owned_ = true;
        
        max_assoc_ = 1024;
        checkCuda(cudaMalloc(&d_cost_matrix_, max_objs_ * max_assoc_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_sinkhorn_v_, max_assoc_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_topk_indices_, max_objs_ * 3 * sizeof(int)));
        checkCuda(cudaMalloc(&d_topk_probs_, max_objs_ * 3 * sizeof(float)));
        checkCuda(cudaMalloc(&d_auction_prices_, max_assoc_ * sizeof(uint64_t)));
        checkCuda(cudaMalloc(&d_trk_to_det_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_det_to_trk_, max_assoc_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_matched_pairs_, max_objs_ * 2 * sizeof(int)));
        checkCuda(cudaMalloc(&d_new_slots_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_state_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_hit_streak_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_confirm_streak_required_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_score_sum_, max_objs_ * sizeof(float)));

        checkCuda(cudaMalloc(&d_has_clean_embedding_, max_objs_ * sizeof(bool)));
        checkCuda(cudaMalloc(&d_candidate_count_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_s_inv_, max_objs_ * 16 * sizeof(float)));
        checkCuda(cudaMalloc(&d_homography_, 9 * sizeof(float)));
        checkCuda(cudaMemset(d_homography_, 0, 9 * sizeof(float)));
        checkCuda(cudaMalloc(&d_occ_coeff_, max_objs_ * sizeof(float)));
        checkCuda(cudaMemset(d_occ_coeff_, 0, max_objs_ * sizeof(float)));

        // M2: GPU spawn + compact result buffers
        checkCuda(cudaMalloc(&d_free_slots_,    max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_n_free_,        sizeof(int)));
        checkCuda(cudaMalloc(&d_slot_cursor_,   sizeof(int)));
        checkCuda(cudaMalloc(&d_track_id_ctr_,  sizeof(int)));
        checkCuda(cudaMalloc(&d_res_boxes_,     max_objs_ * 4 * sizeof(float)));
        checkCuda(cudaMalloc(&d_res_scores_,    max_objs_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_res_ids_,       max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_res_classes_,   max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_res_det_idx_,   max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_res_count_,     sizeof(int)));
        { int init_id = 1; checkCuda(cudaMemcpy(d_track_id_ctr_, &init_id, sizeof(int), cudaMemcpyHostToDevice)); }

        checkCuda(cudaMemset(d_active_, 0, max_objs_ * sizeof(bool)));
        checkCuda(cudaMemset(d_states_, 0, max_objs_ * 8 * sizeof(float)));
        checkCuda(cudaMemset(d_covs_, 0, max_objs_ * 64 * sizeof(float)));
        checkCuda(cudaMemset(d_age_, 0, max_objs_ * sizeof(int)));
        checkCuda(cudaMemset(d_features_, 0, max_objs_ * embed_dim_ * sizeof(float)));
        checkCuda(cudaMemset(d_state_, 0, max_objs_ * sizeof(int)));
        checkCuda(cudaMemset(d_hit_streak_, 0, max_objs_ * sizeof(int)));
        checkCuda(cudaMemset(d_confirm_streak_required_, 0, max_objs_ * sizeof(int)));
        checkCuda(cudaMemset(d_score_sum_, 0, max_objs_ * sizeof(float)));
        checkCuda(cudaMemset(d_has_clean_embedding_, 0, max_objs_ * sizeof(bool)));
        checkCuda(cudaMemset(d_candidate_count_, 0, max_objs_ * sizeof(int)));
        checkCuda(cudaMemset(d_s_inv_, 0, max_objs_ * 16 * sizeof(float)));

        h_active_raw_.resize(max_objs_, 0);
        h_has_clean_embedding_.resize(max_objs_, 0);
        h_states_.resize(max_objs_ * 8, 0.0f);
        h_covs_.resize(max_objs_ * 64, 0.0f);
        h_track_ids_.resize(max_objs_, 0);
        h_scores_.resize(max_objs_, 0.0f);
        h_classes_.resize(max_objs_, 0);
        h_age_.resize(max_objs_, 0);
        h_state_.resize(max_objs_, TRACK_EMPTY);
        h_hit_streak_.resize(max_objs_, 0);
        h_confirm_streak_required_.resize(max_objs_, 0);
        h_score_sum_.resize(max_objs_, 0.0f);
    }

    ~Impl() {
        cudaFree(d_states_); cudaFree(d_covs_); cudaFree(d_active_);
        cudaFree(d_age_); cudaFree(d_scores_); cudaFree(d_classes_);
        cudaFree(d_track_ids_);
        if (d_features_owned_) cudaFree(d_features_);
        cudaFree(d_cost_matrix_); cudaFree(d_sinkhorn_v_);
        cudaFree(d_topk_indices_); cudaFree(d_topk_probs_);
        cudaFree(d_auction_prices_); cudaFree(d_trk_to_det_); cudaFree(d_det_to_trk_);
        cudaFree(d_matched_pairs_); cudaFree(d_new_slots_);
        cudaFree(d_state_); cudaFree(d_hit_streak_); cudaFree(d_confirm_streak_required_);
        cudaFree(d_score_sum_);
        cudaFree(d_has_clean_embedding_); cudaFree(d_candidate_count_);
        cudaFree(d_s_inv_);
        cudaFree(d_homography_);
        cudaFree(d_occ_coeff_);
        // M2
        cudaFree(d_free_slots_); cudaFree(d_n_free_); cudaFree(d_slot_cursor_);
        cudaFree(d_track_id_ctr_);
        cudaFree(d_res_boxes_); cudaFree(d_res_scores_); cudaFree(d_res_ids_);
        cudaFree(d_res_classes_); cudaFree(d_res_det_idx_); cudaFree(d_res_count_);
    }

    std::vector<TrackResult> update(float* d_boxes, float* d_scores, int* d_classes, int num_dets,
                                   cudaStream_t stream, float* d_embeddings, float* d_gmc,
                                   float light_factor, float mid_thresh_scale) {
        nvtxRangePushA("Tracker::Update");
        run_update_device(
            d_boxes, d_scores, d_classes, num_dets, stream, d_embeddings, d_gmc,
            light_factor, mid_thresh_scale);

        // Single blocking D2H: waits for all prior stream work
        int n_res = 0;
        checkCuda(cudaMemcpy(&n_res, d_res_count_, sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<TrackResult> results(static_cast<size_t>(n_res));
        if (n_res > 0) {
            std::vector<float> hb(n_res * 4);
            std::vector<float> hs(n_res);
            std::vector<int>   hi(n_res), hc(n_res), hd(n_res);
            checkCuda(cudaMemcpy(hb.data(), d_res_boxes_,   n_res * 4 * sizeof(float), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(hs.data(), d_res_scores_,  n_res *     sizeof(float), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(hi.data(), d_res_ids_,     n_res *     sizeof(int),   cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(hc.data(), d_res_classes_, n_res *     sizeof(int),   cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(hd.data(), d_res_det_idx_, n_res *     sizeof(int),   cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_res; ++i)
                results[i] = {hb[i*4], hb[i*4+1], hb[i*4+2], hb[i*4+3],
                               hi[i], hs[i], hc[i], hd[i]};
        }
        nvtxRangePop();
        return results;
    }

    void update_into(
        float* d_boxes, float* d_scores, int* d_classes, int num_dets,
        cudaStream_t stream,
        float* out_boxes, float* out_scores, int* out_ids, int* out_classes, int* out_det_idx, int* out_count,
        float* d_embeddings, float* d_gmc,
        float light_factor, float mid_thresh_scale) {
        nvtxRangePushA("Tracker::UpdateInto");
        run_update_device(
            d_boxes, d_scores, d_classes, num_dets, stream, d_embeddings, d_gmc,
            light_factor, mid_thresh_scale);
        checkCuda(cudaMemcpyAsync(out_boxes,   d_res_boxes_,   max_objs_ * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        checkCuda(cudaMemcpyAsync(out_scores,  d_res_scores_,  max_objs_ *     sizeof(float), cudaMemcpyDeviceToDevice, stream));
        checkCuda(cudaMemcpyAsync(out_ids,     d_res_ids_,     max_objs_ *     sizeof(int),   cudaMemcpyDeviceToDevice, stream));
        checkCuda(cudaMemcpyAsync(out_classes, d_res_classes_, max_objs_ *     sizeof(int),   cudaMemcpyDeviceToDevice, stream));
        checkCuda(cudaMemcpyAsync(out_det_idx, d_res_det_idx_, max_objs_ *     sizeof(int),   cudaMemcpyDeviceToDevice, stream));
        checkCuda(cudaMemcpyAsync(out_count,   d_res_count_,   sizeof(int),                    cudaMemcpyDeviceToDevice, stream));
        nvtxRangePop();
    }
    void run_update_device(
        float* d_boxes, float* d_scores, int* d_classes, int num_dets,
        cudaStream_t stream, float* d_embeddings, float* d_gmc,
        float light_factor, float mid_thresh_scale) {
        
        if (num_dets > 0 && enable_quality_scaling_) {
            int threads = 256;
            int blocks = (num_dets + threads - 1) / threads;
            apply_detection_quality_scaling_kernel<<<blocks, threads, 0, stream>>>(
                d_scores, d_boxes, num_dets, frame_w_, frame_h_,
                q_w_aspect_, q_w_center_, q_w_area_);
        }

        int threads = 256;
        int blocks = (max_objs_ + threads - 1) / threads;
        predict_kernel<<<blocks, threads, 0, stream>>>(d_states_, d_covs_, d_active_, d_age_, max_objs_, max_age_);
        if (d_gmc) gmc_kernel<<<blocks, threads, 0, stream>>>(d_states_, d_covs_, d_active_, d_gmc, max_objs_);

        // Phase 2: compute per-track S^-1 from post-predict covariance for Mahalanobis gating
        kernel::compute_innovation_sinv_kernel<<<blocks, threads, 0, stream>>>(
            d_states_, d_covs_, d_active_, d_s_inv_, max_objs_);

        if (num_dets > 0) {
            nvtxRangePushA("Association");
            dim3 b_size(16, 16);
            dim3 g_size((num_dets + 15) / 16, (max_objs_ + 15) / 16);

            // OA-SORT OAO: compute per-track occlusion coefficient from predicted positions
            if (oao_tau_ > 0.0f) {
                kernel::compute_track_occlusion_kernel<<<blocks, threads, 0, stream>>>(
                    d_states_, d_active_, d_occ_coeff_, max_objs_);
            }

            // Stage 1: count candidates passing IoU gate OR Mahalanobis gate
            nvtxRangePushA("Assoc/CostMatrix");
            checkCuda(cudaMemsetAsync(d_candidate_count_, 0, max_objs_ * sizeof(int), stream));
            kernel::count_stage1_candidates_kernel<<<g_size, b_size, 0, stream>>>(
                d_states_, d_boxes, d_active_, d_candidate_count_,
                d_s_inv_, d_homography_, max_objs_, num_dets, iou_stage1_gate_, maha_gate_);

            // Conditional cost: IoU-only fallback, appearance only for ambiguous + clean tracks
            kernel::compute_conditional_cost_kernel<<<g_size, b_size, 0, stream>>>(
                d_states_, d_boxes, d_features_, d_embeddings, d_scores,
                d_candidate_count_, d_has_clean_embedding_,
                d_s_inv_, d_homography_, d_cost_matrix_, max_objs_, num_dets, embed_dim_, iou_stage1_gate_, maha_gate_,
                vel_dir_weight_, fuse_score_weight_,
                reid_cost_cos_w_, reid_cost_iou_w_, reid_cost_score_w_,
                oao_tau_ > 0.0f ? d_occ_coeff_ : nullptr, oao_tau_);
            nvtxRangePop(); // Assoc/CostMatrix

            kernel::track_state_update_pre_kernel<<<blocks, threads, 0, stream>>>(
                d_active_, d_state_, d_hit_streak_, d_confirm_streak_required_, d_score_sum_, max_objs_);

            checkCuda(cudaMemsetAsync(d_det_to_trk_, -1, num_dets * sizeof(int), stream));
            checkCuda(cudaMemsetAsync(d_trk_to_det_, -1, max_objs_ * sizeof(int), stream));
            const int shmem_auction = num_dets * static_cast<int>(sizeof(uint64_t));
            dim3 auc_b(32); dim3 auc_g((max_objs_ + 31) / 32);

            const float effective_mid_thresh = std::clamp(
                mid_thresh_ * std::max(mid_thresh_scale, 0.01f),
                track_thresh_,
                high_thresh_
            );
            const float effective_new_track_thresh = std::clamp(
                new_track_thresh_ * std::max(mid_thresh_scale, 0.01f),
                track_thresh_,
                high_thresh_
            );
            static const bool enable_dda = env_flag_enabled("SACCADE_ENABLE_DDA", true);
            static const float dda_max_cost = env_float_value("SACCADE_DDA_MAX_COST", 0.12f);
            const int* matched_det_mask = nullptr;

            if (enable_dda) {
                // Stage 0: Decomposed Data Association (DDA) - Unambiguous Matching
                // Match isolated/obvious targets first using a very strict cost threshold.
                nvtxRangePushA("Assoc/S0_Unambiguous");
                kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                    d_cost_matrix_, d_scores, d_boxes, d_state_, d_active_, d_trk_to_det_,
                    max_objs_, num_dets, 30.0f, dda_max_cost,
                    high_thresh_, 1.1f, 2,
                    nullptr,
                    d_topk_indices_, d_topk_probs_
                );
                checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(uint64_t), stream));
                kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                    d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                    max_objs_, num_dets, 3, 0.01f);
                nvtxRangePop(); // Assoc/S0_Unambiguous
                matched_det_mask = d_det_to_trk_;
            }

            // Stage 1: High-conf dets -> Confirmed tracks first (D2-C)
            nvtxRangePushA("Assoc/S1_HiConf");
            kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                d_cost_matrix_, d_scores, d_boxes, d_state_, d_active_, d_trk_to_det_,
                max_objs_, num_dets, 30.0f, match_thresh_,
                high_thresh_, 1.1f, 2, // TRACK_CONFIRMED only
                matched_det_mask,
                d_topk_indices_, d_topk_probs_
            );
            checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(uint64_t), stream));
            kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                max_objs_, num_dets, 3, 0.01f);
            nvtxRangePop(); // Assoc/S1_HiConf

            // Stage 1b: Mid-conf dets -> Unmatched confirmed tracks (D2-C)
            nvtxRangePushA("Assoc/S1b_MidConf");
            kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                d_cost_matrix_, d_scores, d_boxes, d_state_, d_active_, d_trk_to_det_,
                max_objs_, num_dets, 30.0f, match_thresh_,
                effective_mid_thresh, high_thresh_, 2, // TRACK_CONFIRMED only
                matched_det_mask,
                d_topk_indices_, d_topk_probs_
            );
            checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(uint64_t), stream));
            kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                max_objs_, num_dets, 3, 0.01f);
            nvtxRangePop(); // Assoc/S1b_MidConf

            // Stage 1c: Unmatched high+mid-conf dets -> Tentative tracks (D2-C isolation)
            nvtxRangePushA("Assoc/S1c_Tentative");
            kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                d_cost_matrix_, d_scores, d_boxes, d_state_, d_active_, d_trk_to_det_,
                max_objs_, num_dets, 30.0f, match_thresh_,
                effective_mid_thresh, 1.1f, 1, // TRACK_TENTATIVE only
                matched_det_mask,
                d_topk_indices_, d_topk_probs_
            );
            checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(uint64_t), stream));
            kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                max_objs_, num_dets, 3, 0.01f);
            nvtxRangePop(); // Assoc/S1c_Tentative

            // Stage 2: Low-conf dets -> Unmatched confirmed tracks only
            nvtxRangePushA("Assoc/S2_LoConf");
            kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                d_cost_matrix_, d_scores, d_boxes, d_state_, d_active_, d_trk_to_det_,
                max_objs_, num_dets, 30.0f, stage2_match_thresh_,
                track_thresh_, effective_mid_thresh, 2, // TRACK_CONFIRMED = 2
                nullptr,
                d_topk_indices_, d_topk_probs_
            );
            checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(uint64_t), stream));
            kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                max_objs_, num_dets, 3, 0.01f);
            nvtxRangePop(); // Assoc/S2_LoConf

            nvtxRangePushA("Assoc/StateUpdate");
            kernel::track_state_update_post_kernel<<<blocks, threads, 0, stream>>>(
                d_active_, d_state_, d_age_, d_scores_, d_classes_,
                d_hit_streak_, d_confirm_streak_required_, d_score_sum_,
                d_trk_to_det_, d_scores, d_classes,
                confirm_streak_, confirm_score_thresh_, max_objs_
            );

            kernel::inline_kalman_update_kernel<<<blocks, threads, 0, stream>>>(
                d_states_, d_covs_, d_boxes, d_trk_to_det_, d_active_, max_objs_, light_factor,
                d_scores, nsa_kalman_, r_scale_
            );
            nvtxRangePop(); // Assoc/StateUpdate

            nvtxRangePop();
        }

        // M2: GPU spawn new tracks
        if (num_dets > 0) {
            const float effective_new_track_thresh = std::clamp(
                new_track_thresh_ * std::max(mid_thresh_scale, 0.01f),
                track_thresh_, high_thresh_);
            cudaMemsetAsync(d_n_free_,      0, sizeof(int), stream);
            cudaMemsetAsync(d_slot_cursor_, 0, sizeof(int), stream);
            collect_free_slots_kernel<<<(max_objs_ + 255) / 256, 256, 0, stream>>>(
                d_active_, max_objs_, d_free_slots_, d_n_free_);
            spawn_new_tracks_kernel<<<(num_dets + 255) / 256, 256, 0, stream>>>(
                d_det_to_trk_, d_boxes, d_scores, d_classes, num_dets,
                effective_new_track_thresh,
                d_free_slots_, d_n_free_,
                d_active_, d_states_, d_state_,
                d_track_ids_, d_age_, d_scores_, d_classes_,
                d_hit_streak_, d_confirm_streak_required_, d_score_sum_,
                d_track_id_ctr_, d_slot_cursor_,
                confirm_streak_, birth_low_score_thresh_,
                max_objs_, birth_prox_norm_thresh_);
            init_covariance_if_new_kernel<<<(max_objs_ + 255) / 256, 256, 0, stream>>>(
                d_active_, d_state_, d_hit_streak_, d_covs_, max_objs_);
        }

        // M2: GPU compact results — write only confirmed+updated tracks
        cudaMemsetAsync(d_res_count_, 0, sizeof(int), stream);
        compact_results_kernel<<<(max_objs_ + 255) / 256, 256, 0, stream>>>(
            d_active_, d_state_, d_age_,
            d_states_, d_track_ids_, d_scores_, d_classes_,
            d_trk_to_det_, max_objs_,
            d_res_boxes_, d_res_scores_, d_res_ids_, d_res_classes_, d_res_det_idx_,
            d_res_count_);
        h_dirty_ = true;           // host arrays are stale until next lazy sync
        h_slot_map_dirty_ = true;  // tid→slot map must be rebuilt before next scatter
    }

    void set_params(float track_thresh, float high_thresh, float match_thresh, int track_buffer,
                    float mid_thresh, int confirm_streak, float confirm_score_thresh,
                    bool adaptive_confirmation, float new_track_thresh, bool nsa_kalman,
                    float r_scale = 1.0f, float vel_dir_weight = 0.0f, float fuse_score_weight = 0.0f,
                    float stage2_match_thresh = 0.5f, float birth_low_score_thresh = 0.0f,
                    float birth_prox_norm_thresh = 0.0f) {
        track_thresh_ = track_thresh; high_thresh_ = high_thresh; match_thresh_ = match_thresh; max_age_ = track_buffer;
        mid_thresh_ = mid_thresh;
        new_track_thresh_ = new_track_thresh >= 0.0f ? new_track_thresh : mid_thresh;
        confirm_streak_ = std::max(confirm_streak, 1);
        confirm_score_thresh_ = confirm_score_thresh;
        adaptive_confirmation_ = adaptive_confirmation;
        nsa_kalman_ = nsa_kalman;
        r_scale_ = std::max(0.01f, r_scale);
        vel_dir_weight_ = fmaxf(0.0f, vel_dir_weight);
        fuse_score_weight_ = std::clamp(fuse_score_weight, 0.0f, 1.0f);
        stage2_match_thresh_ = std::clamp(stage2_match_thresh, 0.0f, 1.0f);
        birth_low_score_thresh_ = fmaxf(0.0f, birth_low_score_thresh);
        birth_prox_norm_thresh_ = fmaxf(0.0f, birth_prox_norm_thresh);
    }
    void set_reid_params(float cos_threshold, float iou_low, float iou_high, float weight,
                         float cost_cos_w = 0.55f, float cost_iou_w = 0.30f, float cost_score_w = 0.15f) {
        reid_cos_threshold_ = cos_threshold; reid_iou_low_ = iou_low; reid_iou_high_ = iou_high; reid_weight_ = weight;
        reid_cost_cos_w_ = cost_cos_w; reid_cost_iou_w_ = cost_iou_w; reid_cost_score_w_ = cost_score_w;
    }
    void set_oao_params(float tau) {
        oao_tau_ = std::clamp(tau, 0.0f, 1.0f);
    }
    void set_quality_params(bool enabled, float w_aspect, float w_center, float w_area) {
        enable_quality_scaling_ = enabled;
        q_w_aspect_ = w_aspect;
        q_w_center_ = w_center;
        q_w_area_ = w_area;
    }
    void set_frame_size(int w, int h) {
        frame_w_ = w;
        frame_h_ = h;
    }
    void set_homography(const float* h) {
        if (h) {
            checkCuda(cudaMemcpy(d_homography_, h, 9 * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            checkCuda(cudaMemset(d_homography_, 0, 9 * sizeof(float)));
        }
    }
    void set_unified_score_params(const UnifiedScoreParams& /*params*/) {
        // Unified score params are reserved for future use in the C++ tracker.
        // The Python layer applies them directly during semantic reranking.
    }

    // Rebuild h_tid_to_slot_ from host arrays.  D2H only if h_dirty_; map rebuild only if
    // h_slot_map_dirty_.  Both update_reference_features_impl and set_clean_embedding_flags
    // call this, so the D2H + rebuild happen at most once per frame.
    void ensure_slot_map() {
        if (!h_slot_map_dirty_) return;
        if (h_dirty_) {
            checkCuda(cudaMemcpy(h_active_raw_.data(), d_active_,    max_objs_ * sizeof(bool), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(h_track_ids_.data(),  d_track_ids_, max_objs_ * sizeof(int),  cudaMemcpyDeviceToHost));
            h_dirty_ = false;
        }
        h_tid_to_slot_.clear();
        for (int slot = 0; slot < max_objs_; ++slot) {
            if (h_active_raw_[slot]) {
                h_tid_to_slot_[h_track_ids_[slot]] = slot;
            }
        }
        h_slot_map_dirty_ = false;
    }

    // Bind an externally-owned GPU float buffer as d_features_.
    // The caller (Python) owns the lifetime; C++ will not cudaFree it.
    void bind_external_features_buffer(float* ptr) {
        if (d_features_owned_) {
            cudaFree(d_features_);
            d_features_owned_ = false;
        }
        d_features_ = ptr;
        // Zero the external buffer so stale data doesn't corrupt association.
        cudaMemset(d_features_, 0, max_objs_ * embed_dim_ * sizeof(float));
    }

    // Return the current tid→slot mapping as a flat list of (tid, slot) pairs.
    // Shares the ensure_slot_map() cache — free if called in the same frame as
    // update_reference_features_impl or set_clean_embedding_flags.
    std::vector<std::pair<int,int>> get_active_tid_slot_pairs() {
        ensure_slot_map();
        std::vector<std::pair<int,int>> result;
        result.reserve(h_tid_to_slot_.size());
        for (const auto& kv : h_tid_to_slot_) {
            result.emplace_back(kv.first, kv.second);
        }
        return result;
    }

    // Scatter bank representative embeddings into d_features_ at the correct slots.
    // d_track_ids_gpu and d_features_src are GPU pointers; n features each of embed_dim_ floats.
    void update_reference_features_impl(int* d_track_ids_gpu, float* d_features_src, int num, cudaStream_t stream) {
        if (num <= 0) return;
        ensure_slot_map();
        std::vector<int> h_tids(num);
        checkCuda(cudaMemcpy(h_tids.data(), d_track_ids_gpu, num * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num; ++i) {
            auto it = h_tid_to_slot_.find(h_tids[i]);
            if (it == h_tid_to_slot_.end()) continue;
            checkCuda(cudaMemcpyAsync(
                d_features_ + it->second * embed_dim_,
                d_features_src + i * embed_dim_,
                embed_dim_ * sizeof(float),
                cudaMemcpyDeviceToDevice, stream
            ));
        }
    }

    // Update d_has_clean_embedding_ from Python bank's clean_ids.
    // d_track_ids_in and d_flags_in are GPU pointers (int32 and bool/uint8).
    void set_clean_embedding_flags(int* d_track_ids_in, bool* d_flags_in, int n, cudaStream_t stream) {
        checkCuda(cudaMemsetAsync(d_has_clean_embedding_, 0, max_objs_ * sizeof(bool), stream));
        std::fill(h_has_clean_embedding_.begin(), h_has_clean_embedding_.end(), static_cast<uint8_t>(0));
        if (n == 0) return;
        ensure_slot_map();
        std::vector<int> h_tids(n);
        std::vector<uint8_t> h_flags(n);
        checkCuda(cudaMemcpy(h_tids.data(), d_track_ids_in, n * sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(h_flags.data(), d_flags_in, n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            if (!h_flags[i]) continue;
            auto it = h_tid_to_slot_.find(h_tids[i]);
            if (it != h_tid_to_slot_.end()) {
                h_has_clean_embedding_[it->second] = 1;
            }
        }
        checkCuda(cudaMemcpyAsync(d_has_clean_embedding_, h_has_clean_embedding_.data(),
                                   max_objs_ * sizeof(bool), cudaMemcpyHostToDevice, stream));
    }

    void set_clean_embedding_flags_host(int* h_tids, bool* h_flags, int n, cudaStream_t stream) {
        checkCuda(cudaMemsetAsync(d_has_clean_embedding_, 0, max_objs_ * sizeof(bool), stream));
        std::fill(h_has_clean_embedding_.begin(), h_has_clean_embedding_.end(), static_cast<uint8_t>(0));
        if (n == 0) return;
        ensure_slot_map();
        for (int i = 0; i < n; ++i) {
            if (!h_flags[i]) continue;
            auto it = h_tid_to_slot_.find(h_tids[i]);
            if (it != h_tid_to_slot_.end()) {
                h_has_clean_embedding_[it->second] = 1;
            }
        }
        checkCuda(cudaMemcpyAsync(d_has_clean_embedding_, h_has_clean_embedding_.data(),
                                   max_objs_ * sizeof(bool), cudaMemcpyHostToDevice, stream));
    }

    std::vector<TrackStateSnapshot> get_state_snapshots(cudaStream_t stream) {
        checkCuda(cudaMemcpyAsync(h_active_raw_.data(), d_active_, max_objs_ * sizeof(bool), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_states_.data(), d_states_, max_objs_ * 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_covs_.data(), d_covs_, max_objs_ * 64 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_age_.data(), d_age_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_scores_.data(), d_scores_, max_objs_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_classes_.data(), d_classes_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_track_ids_.data(), d_track_ids_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        h_dirty_ = false;

        std::vector<TrackStateSnapshot> snapshots;
        snapshots.reserve(max_objs_);
        for (int i = 0; i < max_objs_; ++i) {
            if (!h_active_raw_[i]) continue;
            TrackStateSnapshot snap;
            snap.obj_id = h_track_ids_[i];
            snap.class_id = h_classes_[i];
            snap.age = h_age_[i];
            snap.score = h_scores_[i];
            snap.state.assign(
                h_states_.begin() + static_cast<size_t>(i) * 8,
                h_states_.begin() + static_cast<size_t>(i + 1) * 8
            );
            snap.covariance.assign(
                h_covs_.begin() + static_cast<size_t>(i) * 64,
                h_covs_.begin() + static_cast<size_t>(i + 1) * 64
            );
            snapshots.push_back(std::move(snap));
        }
        return snapshots;
    }

    std::vector<TrackStateSnapshot> get_motion_snapshots_for_track_ids(
        const std::vector<int>& track_ids,
        cudaStream_t stream
    ) {
        if (track_ids.empty()) {
            return {};
        }
        checkCuda(cudaMemcpyAsync(h_active_raw_.data(), d_active_, max_objs_ * sizeof(bool), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_track_ids_.data(), d_track_ids_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        h_dirty_ = false;

        std::unordered_set<int> wanted(track_ids.begin(), track_ids.end());
        std::vector<int> matched_slots;
        matched_slots.reserve(track_ids.size());
        for (int slot = 0; slot < max_objs_; ++slot) {
            if (!h_active_raw_[slot]) continue;
            if (wanted.find(h_track_ids_[slot]) != wanted.end()) {
                matched_slots.push_back(slot);
            }
        }
        if (matched_slots.empty()) {
            return {};
        }

        std::vector<TrackStateSnapshot> snapshots(matched_slots.size());
        std::vector<float> matched_states(matched_slots.size() * 8, 0.0f);
        std::vector<float> matched_covs(matched_slots.size() * 64, 0.0f);
        std::vector<int> matched_classes(matched_slots.size(), 0);
        std::vector<int> matched_ages(matched_slots.size(), 0);
        std::vector<float> matched_scores(matched_slots.size(), 0.0f);

        for (size_t i = 0; i < matched_slots.size(); ++i) {
            const int slot = matched_slots[i];
            snapshots[i].obj_id = h_track_ids_[slot];
            checkCuda(cudaMemcpyAsync(
                matched_states.data() + i * 8,
                d_states_ + static_cast<size_t>(slot) * 8,
                8 * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            ));
            checkCuda(cudaMemcpyAsync(
                matched_covs.data() + i * 64,
                d_covs_ + static_cast<size_t>(slot) * 64,
                64 * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            ));
            checkCuda(cudaMemcpyAsync(
                matched_classes.data() + i,
                d_classes_ + slot,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream
            ));
            checkCuda(cudaMemcpyAsync(
                matched_ages.data() + i,
                d_age_ + slot,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream
            ));
            checkCuda(cudaMemcpyAsync(
                matched_scores.data() + i,
                d_scores_ + slot,
                sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            ));
        }
        cudaStreamSynchronize(stream);

        for (size_t i = 0; i < matched_slots.size(); ++i) {
            snapshots[i].class_id = matched_classes[i];
            snapshots[i].age = matched_ages[i];
            snapshots[i].score = matched_scores[i];
            snapshots[i].state.assign(
                matched_states.begin() + static_cast<ptrdiff_t>(i * 8),
                matched_states.begin() + static_cast<ptrdiff_t>((i + 1) * 8)
            );
            snapshots[i].covariance.assign(
                matched_covs.begin() + static_cast<ptrdiff_t>(i * 64),
                matched_covs.begin() + static_cast<ptrdiff_t>((i + 1) * 64)
            );
        }
        return snapshots;
    }

    std::vector<TrackCandidateSnapshot> get_tentative_candidates(cudaStream_t stream) {
        checkCuda(cudaMemcpyAsync(h_active_raw_.data(),            d_active_,                    max_objs_ *     sizeof(bool),  cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_states_.data(),                d_states_,                    max_objs_ * 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_age_.data(),                   d_age_,                       max_objs_ *     sizeof(int),   cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_scores_.data(),                d_scores_,                    max_objs_ *     sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_classes_.data(),               d_classes_,                   max_objs_ *     sizeof(int),   cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_track_ids_.data(),             d_track_ids_,                 max_objs_ *     sizeof(int),   cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_state_.data(),                 d_state_,                     max_objs_ *     sizeof(int),   cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_hit_streak_.data(),            d_hit_streak_,                max_objs_ *     sizeof(int),   cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_confirm_streak_required_.data(), d_confirm_streak_required_, max_objs_ *     sizeof(int),   cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        h_dirty_ = false;

        std::vector<TrackCandidateSnapshot> candidates;
        for (int i = 0; i < max_objs_; ++i) {
            if (!h_active_raw_[i] || h_state_[i] != TRACK_TENTATIVE) continue;
            float cx = h_states_[i * 8], cy = h_states_[i * 8 + 1];
            float a = h_states_[i * 8 + 2], h = h_states_[i * 8 + 3], w = a * h;
            candidates.push_back({
                h_track_ids_[i],
                h_classes_[i],
                h_age_[i],
                h_hit_streak_[i],
                h_confirm_streak_required_[i] > 0 ? h_confirm_streak_required_[i] : confirm_streak_,
                h_scores_[i],
                cx - w / 2.0f,
                cy - h / 2.0f,
                cx + w / 2.0f,
                cy + h / 2.0f,
            });
        }
        return candidates;
    }

private:
    int required_confirm_streak_for_detection(float score, float mid_thresh_scale) const {
        if (!adaptive_confirmation_) return confirm_streak_;
        if (score >= high_thresh_) return confirm_streak_;
        if (mid_thresh_scale > 1.05f) return confirm_streak_ + 2;
        if (mid_thresh_scale < 0.95f) return confirm_streak_;
        return confirm_streak_ + 1;
    }

    int max_objs_, embed_dim_, max_assoc_;
    float track_thresh_ = 0.1f, high_thresh_ = 0.5f, match_thresh_ = 0.8f, mid_thresh_ = 0.40f, new_track_thresh_ = 0.40f;
    float reid_cos_threshold_ = 0.90f, reid_iou_low_ = 0.3f, reid_iou_high_ = 0.6f, reid_weight_ = 0.4f;
    float reid_cost_cos_w_ = 0.55f, reid_cost_iou_w_ = 0.30f, reid_cost_score_w_ = 0.15f;
    
    bool enable_quality_scaling_ = false;
    float q_w_aspect_ = 0.50f;
    float q_w_center_ = 0.30f;
    float q_w_area_ = 0.20f;
    int frame_w_ = 1920;
    int frame_h_ = 1080;

    float iou_stage1_gate_ = 0.30f;
    float maha_gate_ = 9.4877f;
    int max_age_ = 30, confirm_streak_ = 3;
    float confirm_score_thresh_ = 0.50f;
    bool adaptive_confirmation_ = false;
    bool nsa_kalman_ = false;
    float r_scale_ = 1.0f;
    float vel_dir_weight_ = 0.0f;
    float fuse_score_weight_ = 0.0f;
    float stage2_match_thresh_ = 0.5f;
    float birth_low_score_thresh_ = 0.0f;
    float birth_prox_norm_thresh_ = 0.0f;
    float oao_tau_ = 0.0f;
    float *d_states_, *d_covs_, *d_scores_, *d_features_;
    float* d_occ_coeff_ = nullptr;
    float *d_cost_matrix_, *d_sinkhorn_v_, *d_topk_probs_;
    uint64_t *d_auction_prices_;
    int *d_topk_indices_, *d_trk_to_det_, *d_det_to_trk_;
    int *d_matched_pairs_, *d_new_slots_;
    int *d_state_, *d_hit_streak_, *d_confirm_streak_required_;
    float *d_score_sum_;
    bool* d_active_;
    bool* d_has_clean_embedding_;
    int* d_candidate_count_;
    float* d_s_inv_;
    float* d_homography_;
    int *d_age_, *d_classes_, *d_track_ids_;
    // M2: GPU spawn
    int *d_free_slots_  = nullptr;
    int *d_n_free_      = nullptr;
    int *d_slot_cursor_ = nullptr;
    int *d_track_id_ctr_= nullptr;
    // M2: compact result outputs
    float *d_res_boxes_   = nullptr;
    float *d_res_scores_  = nullptr;
    int   *d_res_ids_     = nullptr;
    int   *d_res_classes_ = nullptr;
    int   *d_res_det_idx_ = nullptr;
    int   *d_res_count_   = nullptr;
    // Host caches — valid only after a lazy sync (set_clean_embedding_flags /
    // update_reference_features_impl / get_tentative_candidates).
    // h_dirty_ is set true by update() and cleared by each lazy-sync function.
    // Any read of h_* arrays while h_dirty_==true is a stale-read bug.
    bool h_dirty_ = false;
    // h_slot_map_dirty_: set true alongside h_dirty_; cleared when h_tid_to_slot_ is rebuilt.
    // ensure_slot_map() rebuilds only once per frame, shared by update_reference_features_impl
    // and set_clean_embedding_flags.
    bool h_slot_map_dirty_ = true;
    std::unordered_map<int,int> h_tid_to_slot_;
    bool d_features_owned_ = true;  // false when bound to an external Python tensor
    std::vector<float>    h_states_, h_covs_, h_scores_;
    std::vector<uint8_t>  h_active_raw_;
    std::vector<uint8_t>  h_has_clean_embedding_;
    std::vector<int>      h_age_, h_classes_, h_track_ids_;
    std::vector<int>      h_state_, h_hit_streak_, h_confirm_streak_required_;
    std::vector<float>    h_score_sum_;
};

GPUByteTracker::GPUByteTracker(int max_objs, int embedding_dim) : pimpl_(std::make_unique<Impl>(max_objs, embedding_dim)) {}
GPUByteTracker::~GPUByteTracker() = default;
void GPUByteTracker::set_params(float track_thresh, float high_thresh, float match_thresh, int track_buffer,
                                float mid_thresh, int confirm_streak, float confirm_score_thresh,
                                bool adaptive_confirmation, float new_track_thresh, bool nsa_kalman,
                                float r_scale, float vel_dir_weight, float fuse_score_weight,
                                float stage2_match_thresh, float birth_low_score_thresh,
                                float birth_prox_norm_thresh) {
    pimpl_->set_params(track_thresh, high_thresh, match_thresh, track_buffer, mid_thresh, confirm_streak, confirm_score_thresh, adaptive_confirmation, new_track_thresh, nsa_kalman, r_scale, vel_dir_weight, fuse_score_weight, stage2_match_thresh, birth_low_score_thresh, birth_prox_norm_thresh);
}
void GPUByteTracker::set_reid_params(float cos_threshold, float iou_low, float iou_high, float weight,
                                     float cost_cos_w, float cost_iou_w, float cost_score_w) {
    pimpl_->set_reid_params(cos_threshold, iou_low, iou_high, weight, cost_cos_w, cost_iou_w, cost_score_w);
}
void GPUByteTracker::set_oao_params(float tau) {
    pimpl_->set_oao_params(tau);
}

void GPUByteTracker::set_quality_params(bool enabled, float w_aspect, float w_center, float w_area) {
    pimpl_->set_quality_params(enabled, w_aspect, w_center, w_area);
}

void GPUByteTracker::set_frame_size(int w, int h) {
    pimpl_->set_frame_size(w, h);
}

void GPUByteTracker::set_homography(const float* h) { pimpl_->set_homography(h); }
void GPUByteTracker::set_unified_score_params(const UnifiedScoreParams& params) { pimpl_->set_unified_score_params(params); }
void GPUByteTracker::update_reference_features(int* track_ids, float* features, int num, cudaStream_t stream) { pimpl_->update_reference_features_impl(track_ids, features, num, stream); }
void GPUByteTracker::set_clean_embedding_flags(int* track_ids, bool* flags, int n, cudaStream_t stream) { pimpl_->set_clean_embedding_flags(track_ids, flags, n, stream); }
void GPUByteTracker::set_clean_embedding_flags_host(int* h_tids, bool* h_flags, int n, cudaStream_t stream) { pimpl_->set_clean_embedding_flags_host(h_tids, h_flags, n, stream); }
void GPUByteTracker::bind_features_buffer(float* ptr) { pimpl_->bind_external_features_buffer(ptr); }
std::vector<std::pair<int,int>> GPUByteTracker::get_active_tid_slot_pairs() { return pimpl_->get_active_tid_slot_pairs(); }
void GPUByteTracker::update_into(
    float* b, float* s, int* c, int n, cudaStream_t stream,
    float* out_boxes, float* out_scores, int* out_ids, int* out_classes, int* out_det_idx, int* out_count,
    float* e, float* g, float l, float m) {
    pimpl_->update_into(
        b, s, c, n, stream,
        out_boxes, out_scores, out_ids, out_classes, out_det_idx, out_count,
        e, g, l, m);
}
std::vector<TrackResult> GPUByteTracker::update(float* b, float* s, int* c, int n, cudaStream_t stream, float* e, float* g, float l, float m) {
    return pimpl_->update(b, s, c, n, stream, e, g, l, m);
}
std::vector<TrackStateSnapshot> GPUByteTracker::get_state_snapshots(cudaStream_t stream) { return pimpl_->get_state_snapshots(stream); }
std::vector<TrackStateSnapshot> GPUByteTracker::get_motion_snapshots_for_track_ids(const std::vector<int>& track_ids, cudaStream_t stream) {
    return pimpl_->get_motion_snapshots_for_track_ids(track_ids, stream);
}
std::vector<TrackCandidateSnapshot> GPUByteTracker::get_tentative_candidates(cudaStream_t stream) { return pimpl_->get_tentative_candidates(stream); }

__device__ float get_iou_device(const float* b1, const float* b2) {
    const float x1 = fmaxf(b1[0], b2[0]);
    const float y1 = fmaxf(b1[1], b2[1]);
    const float x2 = fminf(b1[2], b2[2]);
    const float y2 = fminf(b1[3], b2[3]);
    const float inter = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    const float area1 = fmaxf(0.0f, b1[2] - b1[0]) * fmaxf(0.0f, b1[3] - b1[1]);
    const float area2 = fmaxf(0.0f, b2[2] - b2[0]) * fmaxf(0.0f, b2[3] - b2[1]);
    return inter / (area1 + area2 - inter + 1e-6f);
}

__device__ bool is_box_near_tiled_seam_device(
    const float* box,
    int tiling_mode,
    int frame_w,
    int frame_h,
    float seam_margin_canvas_px
) {
    if (tiling_mode <= 0 || frame_w <= 0 || frame_h <= 0) return false;
    const float r = 960.0f / fmaxf((float)frame_h, (float)frame_w);
    const int h_new = (int)((float)frame_h * r);
    const int w_new = (int)((float)frame_w * r);
    const float y_off = (float)((960 - h_new) / 2);
    const float x_off = (float)((960 - w_new) / 2);
    const float seam_margin_orig = seam_margin_canvas_px / fmaxf(r, 1e-6f);
    const float cx = 0.5f * (box[0] + box[2]);
    const float cy = 0.5f * (box[1] + box[3]);
    const float seam_xs[4] = {160.0f, 320.0f, 640.0f, 800.0f};
    const int seam_x_count = (tiling_mode == 2) ? 4 : 2;
    const int seam_x_start = (tiling_mode == 2) ? 0 : 1;
    for (int i = seam_x_start; i < seam_x_start + seam_x_count; ++i) {
        const float sx_canvas = seam_xs[i];
        if (!(x_off < sx_canvas && sx_canvas < x_off + (float)w_new)) continue;
        const float sx = (sx_canvas - x_off) / r;
        if ((box[0] <= sx && box[2] >= sx) || fabsf(cx - sx) <= seam_margin_orig) return true;
    }
    const float seam_ys[2] = {320.0f, 640.0f};
    for (int i = 0; i < 2; ++i) {
        const float sy_canvas = seam_ys[i];
        if (!(y_off < sy_canvas && sy_canvas < y_off + (float)h_new)) continue;
        const float sy = (sy_canvas - y_off) / r;
        if ((box[1] <= sy && box[3] >= sy) || fabsf(cy - sy) <= seam_margin_orig) return true;
    }
    return false;
}

__global__ void assign_duplicate_anchor_kernel(
    const float* boxes,
    const int* classes,
    int num_dets,
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
    int* anchor_indices
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) return;
    const float* candidate = boxes + idx * 4;
    const int candidate_class = classes[idx];
    int anchor = idx;
    const float candidate_w = fmaxf(candidate[2] - candidate[0], 1e-6f);
    const float candidate_h = fmaxf(candidate[3] - candidate[1], 1e-6f);
    const float candidate_area = candidate_w * candidate_h;
    const float candidate_cx = 0.5f * (candidate[0] + candidate[2]);
    const float candidate_cy = 0.5f * (candidate[1] + candidate[3]);
    const bool candidate_is_seam = is_box_near_tiled_seam_device(
        candidate, tiling_mode, frame_w, frame_h, seam_margin_canvas_px
    );
    for (int prev = 0; prev < idx; ++prev) {
        if (classes[prev] != candidate_class) continue;
        const float* other = boxes + prev * 4;
        const float iou = get_iou_device(candidate, other);
        const float other_w = fmaxf(other[2] - other[0], 1e-6f), other_h = fmaxf(other[3] - other[1], 1e-6f);
        const float other_area = other_w * other_h;
        const float min_w = fminf(candidate_w, other_w), min_h = fminf(candidate_h, other_h);
        const float center_gate = sqrtf(min_w * min_w + min_h * min_h) * center_threshold;
        const float other_cx = 0.5f * (other[0] + other[2]), other_cy = 0.5f * (other[1] + other[3]);
        const float center_dx = other_cx - candidate_cx, center_dy = other_cy - candidate_cy;
        const float center_dist = sqrtf(center_dx * center_dx + center_dy * center_dy);
        const float area_ratio = fminf(candidate_area / fmaxf(other_area, 1e-6f), other_area / fmaxf(candidate_area, 1e-6f));
        const float x1 = fmaxf(candidate[0], other[0]);
        const float y1 = fmaxf(candidate[1], other[1]);
        const float x2 = fminf(candidate[2], other[2]);
        const float y2 = fminf(candidate[3], other[3]);
        const float inter_w = fmaxf(0.0f, x2 - x1);
        const float inter_h = fmaxf(0.0f, y2 - y1);
        const float overlap_ratio_x = inter_w / fmaxf(min_w, 1e-6f);
        const float overlap_ratio_y = inter_h / fmaxf(min_h, 1e-6f);
        const bool other_is_seam = is_box_near_tiled_seam_device(
            other, tiling_mode, frame_w, frame_h, seam_margin_canvas_px
        );
        const bool seam_pair = candidate_is_seam || other_is_seam;
        const bool seam_duplicate =
            seam_pair &&
            center_dist <= center_gate * seam_center_scale &&
            area_ratio >= seam_area_ratio_threshold &&
            overlap_ratio_x >= seam_min_overlap_ratio &&
            overlap_ratio_y >= seam_min_overlap_ratio;
        if (iou >= iou_threshold || (center_dist <= center_gate && area_ratio >= area_ratio_threshold) || seam_duplicate) {
            anchor = prev; break;
        }
    }
    anchor_indices[idx] = anchor;
}

__global__ void aggregate_duplicate_clusters_kernel(
    const float* boxes,
    const float* scores,
    const int* classes,
    const int* anchor_indices,
    int num_dets,
    int tiling_mode,
    int frame_w,
    int frame_h,
    float seam_margin_canvas_px,
    float* box_sums,
    float* score_sums,
    int* score_bits_max,
    float* best_boxes,
    int* best_key_bits,
    int* cluster_counts
) {
    constexpr float kTiledSeamCoordWeight = 0.35f;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) return;
    const int anchor = anchor_indices[idx];
    const float score = scores[idx];
    const bool is_seam = is_box_near_tiled_seam_device(
        boxes + idx * 4, tiling_mode, frame_w, frame_h, seam_margin_canvas_px
    );
    const float coord_weight =
        score * ((tiling_mode > 0 && is_seam) ? kTiledSeamCoordWeight : 1.0f);
    atomicAdd(score_sums + anchor, coord_weight);
    atomicAdd(cluster_counts + anchor, 1);
    atomicMax(score_bits_max + anchor, __float_as_int(score));
    for (int k = 0; k < 4; ++k) {
        atomicAdd(box_sums + anchor * 4 + k, boxes[idx * 4 + k] * coord_weight);
    }
    const int key_bonus = (!is_seam && tiling_mode > 0) ? 0x40000000 : 0;
    const int key = __float_as_int(score) + key_bonus;
    const int prev_key = atomicMax(best_key_bits + anchor, key);
    if (key > prev_key) {
        for (int k = 0; k < 4; ++k) best_boxes[anchor * 4 + k] = boxes[idx * 4 + k];
    }
}

__global__ void compact_duplicate_clusters_kernel(
    const float* box_sums,
    const float* score_sums,
    const int* score_bits_max,
    const float* best_boxes,
    const int* best_key_bits,
    const int* cluster_counts,
    const int* classes,
    int num_dets,
    int tiling_mode,
    float* out_boxes,
    float* out_scores,
    int* out_classes,
    int* out_count
) {
    constexpr float kTiledBestBlend = 0.25f;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int out_idx = 0;
    for (int idx = 0; idx < num_dets; ++idx) {
        if (cluster_counts[idx] <= 0) continue;
        if (tiling_mode > 0) {
            const float inv_coord_sum = 1.0f / fmaxf(score_sums[idx], 1e-6f);
            const float blend_best = cluster_counts[idx] > 1 ? kTiledBestBlend : 1.0f;
            const float blend_avg = 1.0f - blend_best;
            for (int k = 0; k < 4; ++k) {
                const float avg_box = box_sums[idx * 4 + k] * inv_coord_sum;
                const float best_box = best_boxes[idx * 4 + k];
                out_boxes[out_idx * 4 + k] =
                    (cluster_counts[idx] > 1) ? (avg_box * blend_avg + best_box * blend_best) : best_box;
            }
            int key = best_key_bits[idx];
            if (key >= 0x40000000) key -= 0x40000000;
            out_scores[out_idx] = __int_as_float(key);
        } else {
            const float inv_score_sum = 1.0f / fmaxf(score_sums[idx], 1e-6f);
            for (int k = 0; k < 4; ++k) out_boxes[out_idx * 4 + k] = box_sums[idx * 4 + k] * inv_score_sum;
            out_scores[out_idx] = __int_as_float(score_bits_max[idx]);
        }
        out_classes[out_idx] = classes[idx];
        ++out_idx;
    }
    *out_count = out_idx;
}

__global__ void filter_detections_kernel(
    const float* boxes,
    const float* scores,
    const int* classes,
    int num_dets,
    int* keep_indices,
    bool* suspect_flags,
    float* quality_scores,
    int* out_count,
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) return;

    const float* box = boxes + idx * 4;
    const float score = scores[idx];
    bool keep = score > score_threshold;
    if (track_person_only) {
        keep = keep && classes[idx] == person_class;
    }
    const float cx = (box[0] + box[2]) * 0.5f;
    const float cy = (box[1] + box[3]) * 0.5f;
    if (is_tiled) {
        keep = keep && cx >= 0.0f && cx < static_cast<float>(frame_w) && cy >= 0.0f && cy < static_cast<float>(frame_h);
    }

    const float box_w = fmaxf(box[2] - box[0], 1e-6f);
    const float box_h = fmaxf(box[3] - box[1], 1e-6f);
    const float aspect = box_h / box_w;
    const float frame_area = fmaxf(static_cast<float>(frame_w) * static_cast<float>(frame_h), 1.0f);
    const float area_ratio = (box_w * box_h) / frame_area;

    bool geometry_clean = true;
    if (person_geometry_prior) {
        if (person_min_height_ratio > 0.0f) {
            geometry_clean = geometry_clean && box_h >= static_cast<float>(frame_h) * person_min_height_ratio;
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
        const int out_idx = atomicAdd(out_count, 1);
        keep_indices[out_idx] = idx;
        suspect_flags[out_idx] = person_geometry_prior && geometry_suspect_support && !geometry_clean;
        
        // A6: Continuous Quality Score (Aspect + Center + Area)
        // 1. Aspect: Gaussian peak at 2.5
        float aspect_q = expf(-0.5f * powf((aspect - 2.5f) / 1.2f, 2.0f));
        
        // 2. Center: 1.0 at center, 0.0 at boundary (proxy for truncation)
        float cx_norm = cx / fmaxf(static_cast<float>(frame_w), 1.0f);
        float cy_norm = cy / fmaxf(static_cast<float>(frame_h), 1.0f);
        float center_q = fminf(fminf(cx_norm, 1.0f - cx_norm), fminf(cy_norm, 1.0f - cy_norm)) * 4.0f;
        center_q = fmaxf(0.0f, fminf(1.0f, center_q));
        
        // 3. Area: Gaussian peak at 0.01
        float area_q = expf(-0.5f * powf((area_ratio - 0.01f) / 0.01f, 2.0f));
        
        // Combined quality (geometrical only); weights must sum to 1.0
        if (quality_scores) {
            quality_scores[out_idx] = 0.50f * aspect_q + 0.30f * center_q + 0.20f * area_q;
        }
    }
}

constexpr int NMS_BLOCK_SIZE = 64;

// ============================================================================
// 5 Optimizations merged into one kernel:
//   #1 Early IoU Culling: Grid spatial indexing skips non-overlapping IoU
//   #2 Two-stage NMS: Grid pre-filter then exact IoU
//   #3 Filter-NMS Overlap: Single kernel replaces filter+gather+sort+NMS
//   #5 Remove immunity: No dead-code branch in compact path
// ============================================================================

__device__ __forceinline__ float compute_iou_inline(const float* b1, const float* b2) {
    const float x1 = fmaxf(b1[0], b2[0]);
    const float y1 = fmaxf(b1[1], b2[1]);
    const float x2 = fminf(b1[2], b2[2]);
    const float y2 = fminf(b1[3], b2[3]);
    const float inter = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    const float a1 = fmaxf(0.0f, b1[2] - b1[0]) * fmaxf(0.0f, b1[3] - b1[1]);
    const float a2 = fmaxf(0.0f, b2[2] - b2[0]) * fmaxf(0.0f, b2[3] - b2[1]);
    return inter / (a1 + a2 - inter + 1e-6f);
}

__global__ void compact_grid_nms_kernel(
    const float* boxes, const float* scores, const int* classes,
    int num_dets, const int* keep_indices, int valid_count,
    float* out_boxes, float* out_scores, int* out_classes,
    bool* out_suspect, int* out_count,
    float iou_threshold, int grid_w, int grid_h)
{
    extern __shared__ char s_mem[];
    // Shared memory layout (byte offsets from s_mem):
    // s_boxes:   byte 0
    // s_scores:  byte 16*vc
    // s_classes: byte 20*vc
    // s_suspect: byte 24*vc (bool array, vc bytes)
    // s_grid_idx:byte 24*vc + pad (int array, 4*vc bytes)
    // Total: 28*vc + pad_suspect + suppressed[256]
    const int vc = valid_count;
    const int pad_suspect = ((vc + 3) & ~3); // round up to 4-byte boundary
    float* s_boxes    = reinterpret_cast<float*>(s_mem);
    float* s_scores   = s_boxes + vc * 4;  // byte offset 16*vc
    int*   s_classes  = reinterpret_cast<int*>(s_scores + vc);  // byte offset 20*vc
    bool*  s_suspect  = reinterpret_cast<bool*>(s_classes + vc); // byte offset 24*vc
    int*   s_grid_idx = reinterpret_cast<int*>(static_cast<char*>(s_mem) + 24*vc + pad_suspect);

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    // Phase 1: Compact filtered boxes into shared memory
    if (idx < vc) {
        const int orig = keep_indices[idx];
        s_boxes[idx*4+0] = boxes[orig*4+0];
        s_boxes[idx*4+1] = boxes[orig*4+1];
        s_boxes[idx*4+2] = boxes[orig*4+2];
        s_boxes[idx*4+3] = boxes[orig*4+3];
        s_scores[idx] = scores[orig];
        s_classes[idx] = classes[orig];
        s_suspect[idx] = false;
        // Compute grid cell from center
        float cx = (s_boxes[idx*4+0] + s_boxes[idx*4+2]) * 0.5f;
        float cy = (s_boxes[idx*4+1] + s_boxes[idx*4+3]) * 0.5f;
        int gx = min(static_cast<int>(cx * grid_w), grid_w - 1);
        int gy = min(static_cast<int>(cy * grid_h), grid_h - 1);
        gx = max(0, gx); gy = max(0, gy);
        s_grid_idx[idx] = gy * grid_w + gx;
    }
    __syncthreads();

    // Phase 2: Insertion sort by score descending (efficient for n<=256)
    for (int i = 1; i < vc; ++i) {
        float    ks = s_scores[i];
        int      kc = s_classes[i];
        int      gi = s_grid_idx[i];
        float    kx1 = s_boxes[i*4+0], ky1 = s_boxes[i*4+1];
        float    kx2 = s_boxes[i*4+2], ky2 = s_boxes[i*4+3];
        bool     ksus = s_suspect[i];
        int j = i - 1;
        while (j >= 0 && s_scores[j] < ks) {
            s_scores[j+1] = s_scores[j];
            s_classes[j+1] = s_classes[j];
            s_grid_idx[j+1] = s_grid_idx[j];
            s_boxes[(j+1)*4+0] = s_boxes[j*4+0];
            s_boxes[(j+1)*4+1] = s_boxes[j*4+1];
            s_boxes[(j+1)*4+2] = s_boxes[j*4+2];
            s_boxes[(j+1)*4+3] = s_boxes[j*4+3];
            s_suspect[j+1] = s_suspect[j];
            --j;
        }
        s_scores[j+1] = ks; s_classes[j+1] = kc;
        s_grid_idx[j+1] = gi;
        s_boxes[(j+1)*4+0] = kx1; s_boxes[(j+1)*4+1] = ky1;
        s_boxes[(j+1)*4+2] = kx2; s_boxes[(j+1)*4+3] = ky2;
        s_suspect[j+1] = ksus;
    }
    __syncthreads();

    // Phase 3: Grid-based NMS with early IoU culling
    // Grid: image [0,1]x[0,1] -> grid_w x grid_h cells
    // Strategy (#1, #2): For each top box, scan lower-score boxes but skip via:
    //   (a) Grid Manhattan distance > 2 => far apart, skip IoU entirely
    //   (b) Grid AABB pre-check (quick bounding-box overlap)
    //   (c) Exact IoU only for candidates that pass both pre-checks
    // (#5) No immunity_mask check - clean path, no dead-code branch
    __shared__ int suppressed[256];
    for (int i = tid; i < vc; i += blockDim.x) {
        suppressed[i] = 0;
    }
    __syncthreads();

    int keep_count = 0;
    for (int i = 0; i < vc; ++i) {
        if (suppressed[i]) continue;
        const float* rb = s_boxes + i * 4;
        const int rc = s_classes[i];
        const int rcell = s_grid_idx[i];
        const int rgx = rcell % grid_w;
        const int rgy = rcell / grid_w;

        for (int j = i + 1; j < vc; ++j) {
            if (suppressed[j]) continue;
            if (rc != s_classes[j]) continue;  // class-aware NMS

            // Grid early culling (#1): Manhattan distance between center cells
            const int gj = s_grid_idx[j];
            const int gjx = gj % grid_w;
            const int gjy = gj / grid_w;
            const int ddx = (rgx > gjx) ? (rgx - gjx) : (gjx - rgx);
            const int ddy = (rgy > gjy) ? (rgy - gjy) : (gjy - rgy);
            if (ddx + ddy > 2) continue;  // Far apart -> skip IoU

            // Grid AABB pre-check (#2): quick bounding-box overlap test
            const float* cb = s_boxes + j * 4;
            const float gx1 = fmaxf(rb[0], cb[0]);
            const float gy1 = fmaxf(rb[1], cb[1]);
            const float gx2 = fminf(rb[2], cb[2]);
            const float gy2 = fminf(rb[3], cb[3]);
            if (gx1 >= gx2 || gy1 >= gy2) continue;  // No spatial overlap

            // (#5) No immunity_mask - clean exact IoU computation
            if (compute_iou_inline(rb, cb) > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }
    __syncthreads();

    // Phase 4: Write output
    if (tid == 0) {
        for (int i = 0; i < vc; ++i) {
            if (!suppressed[i]) {
                int k = keep_count++;
                out_boxes[k*4+0] = s_boxes[i*4+0];
                out_boxes[k*4+1] = s_boxes[i*4+1];
                out_boxes[k*4+2] = s_boxes[i*4+2];
                out_boxes[k*4+3] = s_boxes[i*4+3];
                out_scores[k] = s_scores[i];
                out_classes[k] = s_classes[i];
                out_suspect[k] = s_suspect[i];
            }
        }
        *out_count = keep_count;
    }
}

void compact_grid_nms_cuda(
    const float* boxes_ptr, const float* scores_ptr, const int* classes_ptr,
    int num_dets, const int* keep_indices, int valid_count,
    float* out_boxes, float* out_scores, int* out_classes,
    bool* out_suspect, int* out_count,
    float iou_threshold,
    cudaStream_t stream)
{
    checkCuda(cudaMemsetAsync(out_count, 0, sizeof(int), stream));
    const int threads = std::min(valid_count, 256);
    const int blocks = (valid_count + threads - 1) / threads;
    // Shared memory: boxes(4*vc*4) + scores(4*vc) + classes(4*vc)
    //             + suspect(1*vc padded to 4-byte) + grid_idx(4*vc)
    //             + suppressed[256](1024 bytes)
    // For vc=39: 780+156+156+39+156+1024 = ~2311 bytes
    const size_t smem = 16 * valid_count + ((valid_count + 3) & ~3) + 1024 + 576;
    compact_grid_nms_kernel<<<blocks, threads, static_cast<unsigned int>(smem), stream>>>(
        boxes_ptr, scores_ptr, classes_ptr, num_dets,
        keep_indices, valid_count,
        out_boxes, out_scores, out_classes, out_suspect, out_count,
        iou_threshold, 16, 9);
    checkCuda(cudaGetLastError());
}

__global__ void nms_bitmask_kernel(
    const float* boxes,
    const int* classes,
    const int64_t* order_indices,
    int num_dets,
    int col_blocks,
    uint64_t* suppression_masks,
    float iou_threshold,
    bool class_aware
) {
    const int col_block = blockIdx.x;
    const int row_block = blockIdx.y;
    const int row_offset = threadIdx.x;
    const int row_pos = row_block * NMS_BLOCK_SIZE + row_offset;
    if (row_pos >= num_dets || row_block > col_block) {
        return;
    }

    const int row_idx = static_cast<int>(order_indices[row_pos]);
    const float* row_box = boxes + row_idx * 4;
    const int row_class = classes[row_idx];
    const int col_start = col_block == row_block ? row_offset + 1 : 0;
    uint64_t mask = 0ULL;
    for (int col_offset = col_start; col_offset < NMS_BLOCK_SIZE; ++col_offset) {
        const int col_pos = col_block * NMS_BLOCK_SIZE + col_offset;
        if (col_pos >= num_dets) {
            break;
        }
        const int col_idx = static_cast<int>(order_indices[col_pos]);
        if (class_aware && classes[col_idx] != row_class) {
            continue;
        }
        const float iou = get_iou_device(row_box, boxes + col_idx * 4);
        if (iou > iou_threshold) {
            mask |= 1ULL << col_offset;
        }
    }
    suppression_masks[row_pos * col_blocks + col_block] = mask;
}

__global__ void nms_select_kernel(
    const uint64_t* suppression_masks,
    const int64_t* order_indices,
    int num_dets,
    int col_blocks,
    int* keep_indices,
    uint64_t* remv,
    int* out_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int keep_count = 0;
    for (int order_pos = 0; order_pos < num_dets; ++order_pos) {
        const int block = order_pos / NMS_BLOCK_SIZE;
        const int offset = order_pos % NMS_BLOCK_SIZE;
        if (remv[block] & (1ULL << offset)) {
            continue;
        }
        keep_indices[keep_count++] = static_cast<int>(order_indices[order_pos]);
        const uint64_t* row_masks = suppression_masks + order_pos * col_blocks;
        for (int col = block; col < col_blocks; ++col) {
            remv[col] |= row_masks[col];
        }
    }
    *out_count = keep_count;
}

__global__ void compute_prior_immunity_kernel(
    const float* boxes,
    const int* classes,
    const int64_t* order_indices,
    const int* valid_count_ptr,
    int max_dets,
    const float* priors,
    const int* prior_classes,
    int num_priors,
    float prior_iou_threshold,
    bool class_aware,
    bool* immunity_mask
) {
    const int valid_count = min(*valid_count_ptr, max_dets);
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= valid_count) return;

    const int idx = static_cast<int>(order_indices[pos]);
    const float* box = boxes + idx * 4;
    const int box_class = classes[idx];

    bool immune = false;
    for (int i = 0; i < num_priors; ++i) {
        if (class_aware && prior_classes && box_class != prior_classes[i]) continue;
        if (get_iou_device(box, priors + i * 4) > prior_iou_threshold) {
            immune = true;
            break;
        }
    }
    immunity_mask[pos] = immune;
}

__global__ void nms_bitmask_counted_kernel(
    const float* boxes,
    const int* classes,
    const int64_t* order_indices,
    const int* valid_count_ptr,
    int max_dets,
    int col_blocks,
    uint64_t* suppression_masks,
    float iou_threshold,
    bool class_aware,
    const bool* immunity_mask
) {
    const int valid_count = min(*valid_count_ptr, max_dets);
    const int col_block = blockIdx.x;
    const int row_block = blockIdx.y;
    const int row_offset = threadIdx.x;
    const int row_pos = row_block * NMS_BLOCK_SIZE + row_offset;
    if (row_pos >= valid_count || row_block > col_block) {
        return;
    }

    const int row_idx = static_cast<int>(order_indices[row_pos]);
    const float* row_box = boxes + row_idx * 4;
    const int row_class = classes[row_idx];
    const int col_start = col_block == row_block ? row_offset + 1 : 0;
    uint64_t mask = 0ULL;
    for (int col_offset = col_start; col_offset < NMS_BLOCK_SIZE; ++col_offset) {
        const int col_pos = col_block * NMS_BLOCK_SIZE + col_offset;
        if (col_pos >= valid_count) {
            break;
        }
        const int col_idx = static_cast<int>(order_indices[col_pos]);
        if (class_aware && classes[col_idx] != row_class) {
            continue;
        }
        const float iou = get_iou_device(row_box, boxes + col_idx * 4);
        if (iou > iou_threshold) {
            if (!immunity_mask || !immunity_mask[col_pos]) {
                mask |= 1ULL << col_offset;
            }
        }
    }
    suppression_masks[row_pos * col_blocks + col_block] = mask;
}

__global__ void nms_select_counted_kernel(
    const uint64_t* suppression_masks,
    const int64_t* order_indices,
    const int* valid_count_ptr,
    int max_dets,
    int col_blocks,
    int* keep_indices,
    uint64_t* remv,
    int* out_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int valid_count = min(*valid_count_ptr, max_dets);
    int keep_count = 0;
    for (int order_pos = 0; order_pos < valid_count; ++order_pos) {
        const int block = order_pos / NMS_BLOCK_SIZE;
        const int offset = order_pos % NMS_BLOCK_SIZE;
        if (remv[block] & (1ULL << offset)) {
            continue;
        }
        keep_indices[keep_count++] = static_cast<int>(order_indices[order_pos]);
        const uint64_t* row_masks = suppression_masks + order_pos * col_blocks;
        for (int col = block; col < col_blocks; ++col) {
            remv[col] |= row_masks[col];
        }
    }
    *out_count = keep_count;
}

void merge_cross_tile_duplicates_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    int num_dets,
    int* anchor_indices_ptr,
    float* box_sums_ptr,
    float* score_sums_ptr,
    int* score_bits_max_ptr,
    float* best_boxes_ptr,
    int* best_key_bits_ptr,
    int* cluster_counts_ptr,
    float* out_boxes_ptr,
    float* out_scores_ptr,
    int* out_classes_ptr,
    int* out_count_ptr,
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
    cudaStream_t stream
) {
    if (num_dets <= 0) { checkCuda(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), stream)); return; }
    checkCuda(cudaMemsetAsync(box_sums_ptr, 0, num_dets * 4 * sizeof(float), stream));
    checkCuda(cudaMemsetAsync(score_sums_ptr, 0, num_dets * sizeof(float), stream));
    checkCuda(cudaMemsetAsync(score_bits_max_ptr, 0, num_dets * sizeof(int), stream));
    checkCuda(cudaMemsetAsync(best_boxes_ptr, 0, num_dets * 4 * sizeof(float), stream));
    checkCuda(cudaMemsetAsync(best_key_bits_ptr, 0, num_dets * sizeof(int), stream));
    checkCuda(cudaMemsetAsync(cluster_counts_ptr, 0, num_dets * sizeof(int), stream));
    const int threads = 256; const int blocks = (num_dets + threads - 1) / threads;
    assign_duplicate_anchor_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr, classes_ptr, num_dets, iou_threshold, center_threshold, area_ratio_threshold,
        tiling_mode, frame_w, frame_h, seam_margin_canvas_px, seam_center_scale,
        seam_area_ratio_threshold, seam_min_overlap_ratio, anchor_indices_ptr
    );
    aggregate_duplicate_clusters_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr, scores_ptr, classes_ptr, anchor_indices_ptr, num_dets,
        tiling_mode, frame_w, frame_h, seam_margin_canvas_px,
        box_sums_ptr, score_sums_ptr, score_bits_max_ptr, best_boxes_ptr, best_key_bits_ptr, cluster_counts_ptr
    );
    compact_duplicate_clusters_kernel<<<1, 1, 0, stream>>>(
        box_sums_ptr, score_sums_ptr, score_bits_max_ptr, best_boxes_ptr, best_key_bits_ptr,
        cluster_counts_ptr, classes_ptr, num_dets, tiling_mode,
        out_boxes_ptr, out_scores_ptr, out_classes_ptr, out_count_ptr
    );
}

void filter_detections_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    int num_dets,
    int* keep_indices_ptr,
    bool* suspect_flags_ptr,
    float* quality_scores_ptr,
    int* out_count_ptr,
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
    cudaStream_t stream
) {
    checkCuda(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), stream));
    if (num_dets <= 0) {
        return;
    }
    // Clear any stale error before kernel launch so cudaGetLastError below
    // reports only errors from THIS kernel.
    cudaGetLastError();
    const int threads = 256;
    const int blocks = (num_dets + threads - 1) / threads;
    filter_detections_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr,
        scores_ptr,
        classes_ptr,
        num_dets,
        keep_indices_ptr,
        suspect_flags_ptr,
        quality_scores_ptr,
        out_count_ptr,
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
        person_max_area_ratio
    );
    {
        cudaError_t _err = cudaGetLastError();
        if (_err != cudaSuccess) {
            std::stringstream _ss;
            _ss << "CUDA Error: " << cudaGetErrorString(_err)
                << " (code=" << static_cast<int>(_err) << ")"
                << " at " << __FILE__ << ":" << __LINE__;
            throw std::runtime_error(_ss.str());
        }
    }
}

void nms_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    const int64_t* order_indices_ptr,
    int num_dets,
    int* keep_indices_ptr,
    uint64_t* suppression_masks_ptr,
    uint64_t* remv_ptr,
    int* out_count_ptr,
    float iou_threshold,
    bool class_aware,
    cudaStream_t stream
) {
    (void)scores_ptr;
    checkCuda(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), stream));
    if (num_dets <= 0) {
        return;
    }
    const int col_blocks = (num_dets + NMS_BLOCK_SIZE - 1) / NMS_BLOCK_SIZE;
    checkCuda(cudaMemsetAsync(suppression_masks_ptr, 0, static_cast<size_t>(num_dets) * col_blocks * sizeof(uint64_t), stream));
    checkCuda(cudaMemsetAsync(remv_ptr, 0, col_blocks * sizeof(uint64_t), stream));
    const dim3 blocks(col_blocks, col_blocks);
    nms_bitmask_kernel<<<blocks, NMS_BLOCK_SIZE, 0, stream>>>(
        boxes_ptr,
        classes_ptr,
        order_indices_ptr,
        num_dets,
        col_blocks,
        suppression_masks_ptr,
        iou_threshold,
        class_aware
    );
    nms_select_kernel<<<1, 1, 0, stream>>>(
        suppression_masks_ptr,
        order_indices_ptr,
        num_dets,
        col_blocks,
        keep_indices_ptr,
        remv_ptr,
        out_count_ptr
    );
    checkCuda(cudaGetLastError());
}

void nms_counted_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    const int64_t* order_indices_ptr,
    int max_dets,
    const int* valid_count_ptr,
    int* keep_indices_ptr,
    uint64_t* suppression_masks_ptr,
    uint64_t* remv_ptr,
    int* out_count_ptr,
    float iou_threshold,
    bool class_aware,
    const float* priors_ptr,
    const int* prior_classes_ptr,
    int num_priors,
    float prior_iou_threshold,
    bool* immunity_mask_ptr,
    cudaStream_t stream
) {
    (void)scores_ptr;
    checkCuda(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), stream));
    if (max_dets <= 0) {
        return;
    }

    if (num_priors > 0 && priors_ptr != nullptr && immunity_mask_ptr != nullptr) {
        const int threads = 256;
        const int blocks = (max_dets + threads - 1) / threads;
        checkCuda(cudaMemsetAsync(immunity_mask_ptr, 0, max_dets * sizeof(bool), stream));
        compute_prior_immunity_kernel<<<blocks, threads, 0, stream>>>(
            boxes_ptr,
            classes_ptr,
            order_indices_ptr,
            valid_count_ptr,
            max_dets,
            priors_ptr,
            prior_classes_ptr,
            num_priors,
            prior_iou_threshold,
            class_aware,
            immunity_mask_ptr
        );
    } else if (immunity_mask_ptr != nullptr) {
        checkCuda(cudaMemsetAsync(immunity_mask_ptr, 0, max_dets * sizeof(bool), stream));
    }

    const int col_blocks = (max_dets + NMS_BLOCK_SIZE - 1) / NMS_BLOCK_SIZE;
    checkCuda(cudaMemsetAsync(suppression_masks_ptr, 0, static_cast<size_t>(max_dets) * col_blocks * sizeof(uint64_t), stream));
    checkCuda(cudaMemsetAsync(remv_ptr, 0, col_blocks * sizeof(uint64_t), stream));
    const dim3 blocks(col_blocks, col_blocks);
    nms_bitmask_counted_kernel<<<blocks, NMS_BLOCK_SIZE, 0, stream>>>(
        boxes_ptr,
        classes_ptr,
        order_indices_ptr,
        valid_count_ptr,
        max_dets,
        col_blocks,
        suppression_masks_ptr,
        iou_threshold,
        class_aware,
        num_priors > 0 ? immunity_mask_ptr : nullptr
    );
    nms_select_counted_kernel<<<1, 1, 0, stream>>>(
        suppression_masks_ptr,
        order_indices_ptr,
        valid_count_ptr,
        max_dets,
        col_blocks,
        keep_indices_ptr,
        remv_ptr,
        out_count_ptr
    );
    checkCuda(cudaGetLastError());
}

} // namespace saccade
