#include "tracking/tracker_gpu.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <unordered_map>
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
__device__ __forceinline__ float mahal_sq_det(
    const float* trk_state, const float* det_box, const float* s_inv)
{
    float det_h  = fmaxf(det_box[3] - det_box[1], 1e-6f);
    float det_cx = (det_box[0] + det_box[2]) * 0.5f;
    float det_cy = (det_box[1] + det_box[3]) * 0.5f;
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
    const float* trk_s_inv,
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
        float d2 = mahal_sq_det(st, b2, trk_s_inv + t * 16);
        pass = (d2 < maha_gate);
    }
    if (pass) atomicAdd(&candidate_count[t], 1);
}

// Two-stage conditional cost matrix.
// Stage 1 gate: IoU > iou_gate OR Mahalanobis^2 < maha_gate; else hard reject (cost=1).
// Stage 2: if candidate_count[t] >= 2 AND has_clean_embedding[t]:
//            cost = 1 - (0.55*CosSim + 0.30*IoU + 0.15*det_score)
//          else:
//            cost = 1 - IoU  (stable IoU-only fallback)
__global__ void compute_conditional_cost_kernel(
    const float* trk_states, const float* det_boxes,
    const float* trk_embeds, const float* det_embeds,
    const float* det_scores,
    const int* candidate_count, const bool* has_clean_embedding,
    const float* trk_s_inv,
    float* cost_matrix,
    int n_trk, int n_det, int embed_dim, float iou_gate, float maha_gate)
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
        bool pass_maha = trk_s_inv && (mahal_sq_det(st, b2, trk_s_inv + t * 16) < maha_gate);
        if (!pass_maha) {
            cost_matrix[t * n_det + d] = 1.0f;
            return;
        }
    }

    float cost;
    if (candidate_count[t] >= 2 && has_clean_embedding[t] && trk_embeds && det_embeds) {
        float cos_sim = 0.0f;
        const float* e1 = trk_embeds + t * embed_dim;
        const float* e2 = det_embeds + d * embed_dim;
        for (int k = 0; k < embed_dim; ++k) cos_sim += e1[k] * e2[k];
        cos_sim = fmaxf(0.0f, cos_sim);
        float ds = det_scores ? det_scores[d] : 0.5f;
        cost = 1.0f - (0.55f * cos_sim + 0.30f * iou + 0.15f * ds);
    } else {
        cost = 1.0f - iou;
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
    const float* cost_matrix, const float* det_scores, const int* trk_states,
    const bool* trk_active, const int* trk_to_det,
    int n_trk, int n_det, float lambda, float max_cost,
    float min_det_score, float max_det_score, int required_trk_state,
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
            float score = det_scores[d];
            if (score < min_det_score || score >= max_det_score) continue;
            
            float cost = cost_matrix[t * n_det + d];
            if (cost > max_cost) continue;

            float p = expf(-lambda * cost);
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
    float* g_prices, int* trk_to_det, int* det_to_trk,
    int n_trk, int n_det, int K, float epsilon)
{
    extern __shared__ float s_prices[];

    for (int d = threadIdx.x; d < n_det; d += blockDim.x)
        s_prices[d] = g_prices[d];
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
            float val = topk_probs[t * K + k] - s_prices[d];
            if (val > best_val) {
                second_best_val = best_val; best_val = val; best_det = d;
            } else if (val > second_best_val) {
                second_best_val = val;
            }
        }
        if (best_det >= 0) {
            float inc = (second_best_val <= -1e8f) ? epsilon
                                                    : (best_val - second_best_val + epsilon);
            bid = s_prices[best_det] + inc;
            // Level 1: intra-block conflict resolution
            atomicMax((int*)&s_prices[best_det], __float_as_int(bid));
        }
    }
    __syncthreads();

    // Level 2: block winner commits to global memory
    if (best_det >= 0 && __float_as_int(s_prices[best_det]) == __float_as_int(bid)) {
        int prev = atomicMax((int*)&g_prices[best_det], __float_as_int(bid));
        if (__float_as_int(bid) > prev) {
            trk_to_det[t] = best_det;
            atomicExch((int*)&det_to_trk[best_det], t);
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
    const float* det_scores, bool nsa_kalman)
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
    kf_gpu::update(states + t * 8, covs + t * 64, z, light_factor, nsa_mult);
}
} // namespace kernel

class GPUByteTracker::Impl {
public:
    Impl(int max_objects, int embedding_dim) 
        : max_objs_(max_objects), embed_dim_(embedding_dim), track_id_counter_(1) {
        checkCuda(cudaMalloc(&d_states_, max_objs_ * 8 * sizeof(float)));
        checkCuda(cudaMalloc(&d_covs_, max_objs_ * 64 * sizeof(float)));
        checkCuda(cudaMalloc(&d_active_, max_objs_ * sizeof(bool)));
        checkCuda(cudaMalloc(&d_age_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_scores_, max_objs_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_classes_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_track_ids_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_features_, max_objs_ * embed_dim_ * sizeof(float)));
        
        max_assoc_ = 1024;
        checkCuda(cudaMalloc(&d_cost_matrix_, max_objs_ * max_assoc_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_sinkhorn_v_, max_assoc_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_topk_indices_, max_objs_ * 3 * sizeof(int)));
        checkCuda(cudaMalloc(&d_topk_probs_, max_objs_ * 3 * sizeof(float)));
        checkCuda(cudaMalloc(&d_auction_prices_, max_assoc_ * sizeof(float)));
        checkCuda(cudaMalloc(&d_trk_to_det_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_det_to_trk_, max_assoc_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_matched_pairs_, max_objs_ * 2 * sizeof(int)));
        checkCuda(cudaMalloc(&d_new_slots_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_state_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_hit_streak_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_confirm_streak_required_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_score_sum_, max_objs_ * sizeof(float)));
        checkCuda(cudaMallocHost(&d_det_to_trk_h_, max_assoc_ * sizeof(int)));

        checkCuda(cudaMalloc(&d_has_clean_embedding_, max_objs_ * sizeof(bool)));
        checkCuda(cudaMalloc(&d_candidate_count_, max_objs_ * sizeof(int)));
        checkCuda(cudaMalloc(&d_s_inv_, max_objs_ * 16 * sizeof(float)));

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
        h_track_ids_.resize(max_objs_, 0);
        h_scores_.resize(max_objs_, 0.0f);
        h_classes_.resize(max_objs_, 0);
        h_age_.resize(max_objs_, 0);
        h_state_.resize(max_objs_, TRACK_EMPTY);
        h_hit_streak_.resize(max_objs_, 0);
        h_confirm_streak_required_.resize(max_objs_, 0);
        h_score_sum_.resize(max_objs_, 0.0f);
        h_matched_pairs_.resize(max_objs_ * 2);
        h_new_slots_.resize(max_objs_);
        h_det_to_trk_.resize(max_assoc_, -1);
        h_topk_indices_.resize(max_objs_ * 3, -1);
    }

    ~Impl() {
        cudaFree(d_states_); cudaFree(d_covs_); cudaFree(d_active_);
        cudaFree(d_age_); cudaFree(d_scores_); cudaFree(d_classes_);
        cudaFree(d_track_ids_); cudaFree(d_features_);
        cudaFree(d_cost_matrix_); cudaFree(d_sinkhorn_v_);
        cudaFree(d_topk_indices_); cudaFree(d_topk_probs_);
        cudaFree(d_auction_prices_); cudaFree(d_trk_to_det_); cudaFree(d_det_to_trk_);
        cudaFree(d_matched_pairs_); cudaFree(d_new_slots_);
        cudaFree(d_state_); cudaFree(d_hit_streak_); cudaFree(d_confirm_streak_required_);
        cudaFree(d_score_sum_); cudaFreeHost(d_det_to_trk_h_);
        cudaFree(d_has_clean_embedding_); cudaFree(d_candidate_count_);
        cudaFree(d_s_inv_);
    }

    std::vector<TrackResult> update(float* d_boxes, float* d_scores, int* d_classes, int num_dets,
                                   cudaStream_t stream, float* d_embeddings, float* d_gmc,
                                   float light_factor, float mid_thresh_scale) {
        nvtxRangePushA("Tracker::Update");
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

            // Stage 1: count candidates passing IoU gate OR Mahalanobis gate
            checkCuda(cudaMemsetAsync(d_candidate_count_, 0, max_objs_ * sizeof(int), stream));
            kernel::count_stage1_candidates_kernel<<<g_size, b_size, 0, stream>>>(
                d_states_, d_boxes, d_active_, d_candidate_count_,
                d_s_inv_, max_objs_, num_dets, iou_stage1_gate_, maha_gate_);

            // Conditional cost: IoU-only fallback, appearance only for ambiguous + clean tracks
            kernel::compute_conditional_cost_kernel<<<g_size, b_size, 0, stream>>>(
                d_states_, d_boxes, d_features_, d_embeddings, d_scores,
                d_candidate_count_, d_has_clean_embedding_,
                d_s_inv_, d_cost_matrix_, max_objs_, num_dets, embed_dim_, iou_stage1_gate_, maha_gate_);
            
            kernel::track_state_update_pre_kernel<<<blocks, threads, 0, stream>>>(
                d_active_, d_state_, d_hit_streak_, d_confirm_streak_required_, d_score_sum_, max_objs_);

            checkCuda(cudaMemsetAsync(d_det_to_trk_, -1, num_dets * sizeof(int), stream));
            checkCuda(cudaMemsetAsync(d_trk_to_det_, -1, max_objs_ * sizeof(int), stream));
            const int shmem_auction = num_dets * static_cast<int>(sizeof(float));
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

            // Stage 1: High-conf dets -> All active tracks
            kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                d_cost_matrix_, d_scores, d_state_, d_active_, d_trk_to_det_,
                max_objs_, num_dets, 30.0f, match_thresh_,
                high_thresh_, 1.1f, -1,
                d_topk_indices_, d_topk_probs_
            );
            checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(float), stream));
            kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                max_objs_, num_dets, 3, 0.01f);

            // Stage 1b: Mid-conf dets -> Unmatched active tracks
            kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                d_cost_matrix_, d_scores, d_state_, d_active_, d_trk_to_det_,
                max_objs_, num_dets, 30.0f, match_thresh_,
                effective_mid_thresh, high_thresh_, -1,
                d_topk_indices_, d_topk_probs_
            );
            checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(float), stream));
            kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                max_objs_, num_dets, 3, 0.01f);

            // Stage 2: Low-conf dets -> Unmatched confirmed tracks only
            kernel::fused_sinkhorn_topk_kernel<<<max_objs_, 128, 0, stream>>>(
                d_cost_matrix_, d_scores, d_state_, d_active_, d_trk_to_det_,
                max_objs_, num_dets, 30.0f, 0.5f,
                track_thresh_, effective_mid_thresh, 2, // TRACK_CONFIRMED = 2
                d_topk_indices_, d_topk_probs_
            );
            checkCuda(cudaMemsetAsync(d_auction_prices_, 0, num_dets * sizeof(float), stream));
            kernel::parallel_auction_shmem_kernel<<<auc_g, auc_b, shmem_auction, stream>>>(
                d_topk_indices_, d_topk_probs_, d_auction_prices_, d_trk_to_det_, d_det_to_trk_,
                max_objs_, num_dets, 3, 0.01f);
            
            kernel::track_state_update_post_kernel<<<blocks, threads, 0, stream>>>(
                d_active_, d_state_, d_age_, d_scores_, d_classes_,
                d_hit_streak_, d_confirm_streak_required_, d_score_sum_,
                d_trk_to_det_, d_scores, d_classes,
                confirm_streak_, confirm_score_thresh_, max_objs_
            );

            kernel::inline_kalman_update_kernel<<<blocks, threads, 0, stream>>>(
                d_states_, d_covs_, d_boxes, d_trk_to_det_, d_active_, max_objs_, light_factor,
                d_scores, nsa_kalman_
            );

            nvtxRangePop();
        }

        checkCuda(cudaMemcpyAsync(h_active_raw_.data(), d_active_, max_objs_ * sizeof(bool), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_state_.data(), d_state_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_states_.data(), d_states_, max_objs_ * 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_track_ids_.data(), d_track_ids_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_age_.data(), d_age_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_scores_.data(), d_scores_, max_objs_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_classes_.data(), d_classes_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));

        std::vector<float> h_det_boxes;
        std::vector<float> h_det_scores_inp;
        std::vector<int> h_det_classes_inp;
        if (num_dets > 0) {
            h_det_boxes.resize(num_dets * 4);
            h_det_scores_inp.resize(num_dets);
            h_det_classes_inp.resize(num_dets);
            checkCuda(cudaMemcpyAsync(h_det_boxes.data(), d_boxes, num_dets * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaMemcpyAsync(h_det_scores_inp.data(), d_scores, num_dets * sizeof(float), cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaMemcpyAsync(h_det_classes_inp.data(), d_classes, num_dets * sizeof(int), cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaMemcpyAsync(d_det_to_trk_h_, d_det_to_trk_, num_dets * sizeof(int), cudaMemcpyDeviceToHost, stream));
        }

        cudaStreamSynchronize(stream);

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

        if (num_dets > 0) {
            int n_new = 0;
            for (int d = 0; d < num_dets; ++d) {
                if (d_det_to_trk_h_[d] >= 0 || h_det_scores_inp[d] < effective_new_track_thresh) continue;
                int slot = -1;
                for (int i = 0; i < max_objs_; ++i) {
                    if (!h_active_raw_[i]) { slot = i; break; }
                }
                if (slot == -1) break;

                const float* box = h_det_boxes.data() + d * 4;
                float cx = (box[0] + box[2]) * 0.5f;
                float cy = (box[1] + box[3]) * 0.5f;
                float bh = std::max(box[3] - box[1], 1e-6f);
                float cls = h_det_classes_inp[d];
                float score = h_det_scores_inp[d];
                h_active_raw_[slot] = 1;
                h_states_[slot * 8 + 0] = cx;
                h_states_[slot * 8 + 1] = cy;
                h_states_[slot * 8 + 2] = (box[2] - box[0]) / bh;
                h_states_[slot * 8 + 3] = bh;
                for (int k = 4; k < 8; ++k) h_states_[slot * 8 + k] = 0.0f;
                
                h_track_ids_[slot] = track_id_counter_++;
                h_age_[slot] = 0;
                h_state_[slot] = 1; // TRACK_TENTATIVE
                h_hit_streak_[slot] = 1;
                h_confirm_streak_required_[slot] = 0;
                h_score_sum_[slot] = score;
                h_scores_[slot] = score;
                h_classes_[slot] = cls;
                
                h_new_slots_[n_new++] = slot;
            }
            if (n_new > 0) {
                checkCuda(cudaMemcpyAsync(d_active_, h_active_raw_.data(), max_objs_ * sizeof(bool), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_state_, h_state_.data(), max_objs_ * sizeof(int), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_hit_streak_, h_hit_streak_.data(), max_objs_ * sizeof(int), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_score_sum_, h_score_sum_.data(), max_objs_ * sizeof(float), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_confirm_streak_required_, h_confirm_streak_required_.data(), max_objs_ * sizeof(int), cudaMemcpyHostToDevice, stream));
                
                checkCuda(cudaMemcpyAsync(d_states_, h_states_.data(), max_objs_ * 8 * sizeof(float), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_track_ids_, h_track_ids_.data(), max_objs_ * sizeof(int), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_age_, h_age_.data(), max_objs_ * sizeof(int), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_scores_, h_scores_.data(), max_objs_ * sizeof(float), cudaMemcpyHostToDevice, stream));
                checkCuda(cudaMemcpyAsync(d_classes_, h_classes_.data(), max_objs_ * sizeof(int), cudaMemcpyHostToDevice, stream));

                checkCuda(cudaMemcpyAsync(d_new_slots_, h_new_slots_.data(), n_new * sizeof(int), cudaMemcpyHostToDevice, stream));
                int threads = 128;
                int blocks = (n_new + threads - 1) / threads;
                init_covariance_kernel<<<blocks, threads, 0, stream>>>(d_covs_, d_new_slots_, n_new);
                cudaStreamSynchronize(stream);
            }
        }

        // Build trk_slot → det_idx reverse mapping from pinned det→slot map
        std::vector<int> h_trk_to_det(max_objs_, -1);
        if (num_dets > 0) {
            for (int d = 0; d < num_dets; ++d) {
                int slot = d_det_to_trk_h_[d];
                if (slot >= 0 && slot < max_objs_)
                    h_trk_to_det[slot] = d;
            }
        }

        std::vector<TrackResult> results;
        for (int i = 0; i < max_objs_; ++i) {
            if (h_active_raw_[i] && h_state_[i] == 2 && h_age_[i] == 0) {
                float cx = h_states_[i * 8], cy = h_states_[i * 8 + 1], a = h_states_[i * 8 + 2], h = h_states_[i * 8 + 3], w = a * h;
                results.push_back({cx - w/2.0f, cy - h/2.0f, cx + w/2.0f, cy + h/2.0f, h_track_ids_[i], h_scores_[i], h_classes_[i], h_trk_to_det[i]});
            }
        }
        nvtxRangePop();
        return results;
    }

    void set_params(float track_thresh, float high_thresh, float match_thresh, int track_buffer,
                    float mid_thresh, int confirm_streak, float confirm_score_thresh,
                    bool adaptive_confirmation, float new_track_thresh, bool nsa_kalman) {
        track_thresh_ = track_thresh; high_thresh_ = high_thresh; match_thresh_ = match_thresh; max_age_ = track_buffer;
        mid_thresh_ = mid_thresh;
        new_track_thresh_ = new_track_thresh >= 0.0f ? new_track_thresh : mid_thresh;
        confirm_streak_ = std::max(confirm_streak, 1);
        confirm_score_thresh_ = confirm_score_thresh;
        adaptive_confirmation_ = adaptive_confirmation;
        nsa_kalman_ = nsa_kalman;
    }
    void set_reid_params(float cos_threshold, float iou_low, float iou_high, float weight) {
        reid_cos_threshold_ = cos_threshold; reid_iou_low_ = iou_low; reid_iou_high_ = iou_high; reid_weight_ = weight;
    }

    // Scatter bank representative embeddings into d_features_ at the correct slots.
    // d_track_ids_gpu and d_features_src are GPU pointers; n features each of embed_dim_ floats.
    void update_reference_features_impl(int* d_track_ids_gpu, float* d_features_src, int num, cudaStream_t stream) {
        if (num <= 0) return;
        std::vector<int> h_tids(num);
        checkCuda(cudaMemcpy(h_tids.data(), d_track_ids_gpu, num * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < num; ++i) {
            const int tid = h_tids[i];
            for (int slot = 0; slot < max_objs_; ++slot) {
                if (h_active_raw_[slot] && h_track_ids_[slot] == tid) {
                    checkCuda(cudaMemcpyAsync(
                        d_features_ + slot * embed_dim_,
                        d_features_src + i * embed_dim_,
                        embed_dim_ * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream
                    ));
                    break;
                }
            }
        }
    }

    // Update d_has_clean_embedding_ from Python bank's clean_ids.
    // d_track_ids_in and d_flags_in are GPU pointers (int32 and bool/uint8).
    void set_clean_embedding_flags(int* d_track_ids_in, bool* d_flags_in, int n, cudaStream_t stream) {
        checkCuda(cudaMemsetAsync(d_has_clean_embedding_, 0, max_objs_ * sizeof(bool), stream));
        std::fill(h_has_clean_embedding_.begin(), h_has_clean_embedding_.end(), static_cast<uint8_t>(0));
        if (n == 0) return;
        std::vector<int> h_tids(n);
        std::vector<uint8_t> h_flags(n);
        checkCuda(cudaMemcpy(h_tids.data(), d_track_ids_in, n * sizeof(int), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(h_flags.data(), d_flags_in, n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            if (!h_flags[i]) continue;
            const int tid = h_tids[i];
            for (int slot = 0; slot < max_objs_; ++slot) {
                if (h_active_raw_[slot] && h_track_ids_[slot] == tid) {
                    h_has_clean_embedding_[slot] = 1;
                    break;
                }
            }
        }
        checkCuda(cudaMemcpyAsync(d_has_clean_embedding_, h_has_clean_embedding_.data(),
                                   max_objs_ * sizeof(bool), cudaMemcpyHostToDevice, stream));
    }

    std::vector<TrackCandidateSnapshot> get_tentative_candidates(cudaStream_t stream) {
        checkCuda(cudaMemcpyAsync(h_active_raw_.data(), d_active_, max_objs_ * sizeof(bool), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_states_.data(), d_states_, max_objs_ * 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_age_.data(), d_age_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_scores_.data(), d_scores_, max_objs_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_classes_.data(), d_classes_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaMemcpyAsync(h_track_ids_.data(), d_track_ids_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

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

    int max_objs_, embed_dim_, track_id_counter_, max_assoc_;
    float track_thresh_ = 0.1f, high_thresh_ = 0.5f, match_thresh_ = 0.8f, mid_thresh_ = 0.40f, new_track_thresh_ = 0.40f;
    float reid_cos_threshold_ = 0.90f, reid_iou_low_ = 0.3f, reid_iou_high_ = 0.6f, reid_weight_ = 0.4f;
    float iou_stage1_gate_ = 0.30f;
    float maha_gate_ = 9.4877f;
    int max_age_ = 30, confirm_streak_ = 3;
    float confirm_score_thresh_ = 0.50f;
    bool adaptive_confirmation_ = false;
    bool nsa_kalman_ = false;
    float *d_states_, *d_covs_, *d_scores_, *d_features_;
    float *d_cost_matrix_, *d_sinkhorn_v_, *d_topk_probs_, *d_auction_prices_;
    int *d_topk_indices_, *d_trk_to_det_, *d_det_to_trk_;
    int *d_matched_pairs_, *d_new_slots_;
    int *d_state_, *d_hit_streak_, *d_confirm_streak_required_;
    float *d_score_sum_;
    int *d_det_to_trk_h_;
    bool* d_active_;
    bool* d_has_clean_embedding_;
    int* d_candidate_count_;
    float* d_s_inv_;
    int *d_age_, *d_classes_, *d_track_ids_;
    std::vector<float> h_states_, h_scores_;
    std::vector<uint8_t> h_active_raw_;
    std::vector<uint8_t> h_has_clean_embedding_;
    std::vector<int> h_age_, h_classes_, h_track_ids_;
    std::vector<int> h_state_, h_hit_streak_, h_confirm_streak_required_;
    std::vector<float> h_score_sum_;
    std::vector<int> h_matched_pairs_, h_new_slots_, h_det_to_trk_, h_topk_indices_;
};

GPUByteTracker::GPUByteTracker(int max_objs, int embedding_dim) : pimpl_(std::make_unique<Impl>(max_objs, embedding_dim)) {}
GPUByteTracker::~GPUByteTracker() = default;
void GPUByteTracker::set_params(float track_thresh, float high_thresh, float match_thresh, int track_buffer,
                                float mid_thresh, int confirm_streak, float confirm_score_thresh,
                                bool adaptive_confirmation, float new_track_thresh, bool nsa_kalman) {
    pimpl_->set_params(track_thresh, high_thresh, match_thresh, track_buffer, mid_thresh, confirm_streak, confirm_score_thresh, adaptive_confirmation, new_track_thresh, nsa_kalman);
}
void GPUByteTracker::set_reid_params(float cos_threshold, float iou_low, float iou_high, float weight) { pimpl_->set_reid_params(cos_threshold, iou_low, iou_high, weight); }
void GPUByteTracker::update_reference_features(int* track_ids, float* features, int num, cudaStream_t stream) { pimpl_->update_reference_features_impl(track_ids, features, num, stream); }
void GPUByteTracker::set_clean_embedding_flags(int* track_ids, bool* flags, int n, cudaStream_t stream) { pimpl_->set_clean_embedding_flags(track_ids, flags, n, stream); }
std::vector<TrackResult> GPUByteTracker::update(float* b, float* s, int* c, int n, cudaStream_t stream, float* e, float* g, float l, float m) {
    return pimpl_->update(b, s, c, n, stream, e, g, l, m);
}
std::vector<TrackStateSnapshot> GPUByteTracker::get_state_snapshots(cudaStream_t stream) { return {}; }
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

__global__ void assign_duplicate_anchor_kernel(const float* boxes, const int* classes, int num_dets, float iou_threshold, float center_threshold, float area_ratio_threshold, int* anchor_indices) {
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
        if (iou >= iou_threshold || (center_dist <= center_gate && area_ratio >= area_ratio_threshold)) { anchor = prev; break; }
    }
    anchor_indices[idx] = anchor;
}

__global__ void aggregate_duplicate_clusters_kernel(const float* boxes, const float* scores, const int* classes, const int* anchor_indices, int num_dets, float* box_sums, float* score_sums, int* score_bits_max, int* cluster_counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) return;
    const int anchor = anchor_indices[idx];
    const float score = scores[idx];
    atomicAdd(score_sums + anchor, score);
    atomicAdd(cluster_counts + anchor, 1);
    atomicMax(score_bits_max + anchor, __float_as_int(score));
    for (int k = 0; k < 4; ++k) atomicAdd(box_sums + anchor * 4 + k, boxes[idx * 4 + k] * score);
}

__global__ void compact_duplicate_clusters_kernel(const float* box_sums, const float* score_sums, const int* score_bits_max, const int* cluster_counts, const int* classes, int num_dets, float* out_boxes, float* out_scores, int* out_classes, int* out_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int out_idx = 0;
    for (int idx = 0; idx < num_dets; ++idx) {
        if (cluster_counts[idx] <= 0) continue;
        const float inv_score_sum = 1.0f / fmaxf(score_sums[idx], 1e-6f);
        for (int k = 0; k < 4; ++k) out_boxes[out_idx * 4 + k] = box_sums[idx * 4 + k] * inv_score_sum;
        out_scores[out_idx] = __int_as_float(score_bits_max[idx]);
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
    bool keep = scores[idx] > score_threshold;
    if (track_person_only) {
        keep = keep && classes[idx] == person_class;
    }
    if (is_tiled) {
        const float cx = (box[0] + box[2]) * 0.5f;
        const float cy = (box[1] + box[3]) * 0.5f;
        keep = keep && cx >= 0.0f && cx < static_cast<float>(frame_w) && cy >= 0.0f && cy < static_cast<float>(frame_h);
    }

    bool geometry_clean = true;
    if (person_geometry_prior) {
        const float box_w = fmaxf(box[2] - box[0], 1e-6f);
        const float box_h = fmaxf(box[3] - box[1], 1e-6f);
        const float aspect = box_h / box_w;
        const float frame_area = fmaxf(static_cast<float>(frame_w) * static_cast<float>(frame_h), 1.0f);
        const float area_ratio = (box_w * box_h) / frame_area;
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
    }
}

constexpr int NMS_BLOCK_SIZE = 64;

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

void merge_cross_tile_duplicates_cuda(const float* boxes_ptr, const float* scores_ptr, const int* classes_ptr, int num_dets, int* anchor_indices_ptr, float* box_sums_ptr, float* score_sums_ptr, int* score_bits_max_ptr, int* cluster_counts_ptr, float* out_boxes_ptr, float* out_scores_ptr, int* out_classes_ptr, int* out_count_ptr, float iou_threshold, float center_threshold, float area_ratio_threshold, cudaStream_t stream) {
    if (num_dets <= 0) { checkCuda(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), stream)); return; }
    checkCuda(cudaMemsetAsync(box_sums_ptr, 0, num_dets * 4 * sizeof(float), stream));
    checkCuda(cudaMemsetAsync(score_sums_ptr, 0, num_dets * sizeof(float), stream));
    checkCuda(cudaMemsetAsync(score_bits_max_ptr, 0, num_dets * sizeof(int), stream));
    checkCuda(cudaMemsetAsync(cluster_counts_ptr, 0, num_dets * sizeof(int), stream));
    const int threads = 256; const int blocks = (num_dets + threads - 1) / threads;
    assign_duplicate_anchor_kernel<<<blocks, threads, 0, stream>>>(boxes_ptr, classes_ptr, num_dets, iou_threshold, center_threshold, area_ratio_threshold, anchor_indices_ptr);
    aggregate_duplicate_clusters_kernel<<<blocks, threads, 0, stream>>>(boxes_ptr, scores_ptr, classes_ptr, anchor_indices_ptr, num_dets, box_sums_ptr, score_sums_ptr, score_bits_max_ptr, cluster_counts_ptr);
    compact_duplicate_clusters_kernel<<<1, 1, 0, stream>>>(box_sums_ptr, score_sums_ptr, score_bits_max_ptr, cluster_counts_ptr, classes_ptr, num_dets, out_boxes_ptr, out_scores_ptr, out_classes_ptr, out_count_ptr);
}

void filter_detections_cuda(
    const float* boxes_ptr,
    const float* scores_ptr,
    const int* classes_ptr,
    int num_dets,
    int* keep_indices_ptr,
    bool* suspect_flags_ptr,
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
    const int threads = 256;
    const int blocks = (num_dets + threads - 1) / threads;
    filter_detections_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr,
        scores_ptr,
        classes_ptr,
        num_dets,
        keep_indices_ptr,
        suspect_flags_ptr,
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
    checkCuda(cudaGetLastError());
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

} // namespace saccade
