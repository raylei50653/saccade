#include "tracking/tracker_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <cstdint>
#include <cmath>
#include "tracking/sinkhorn.hpp"
#include "tracking/kalman_gpu.cuh"

namespace saccade {

// --- CUDA Kernels ---

__global__ void predict_kernel(float* states, float* covs, bool* active, int* age, int max_objs, int max_age) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_objs) return;

    if (active[idx]) {
        kf_gpu::predict(states + idx * 8, covs + idx * 64);
        age[idx]++;
        if (age[idx] >= max_age) {
            active[idx] = false;
        }
    }
}

__global__ void gmc_kernel(float* states, float* covs, bool* active, const float* gmc, int max_objs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_objs) return;

    if (active[idx]) {
        float* x = states + idx * 8;
        float* P = covs + idx * 64;

        // Affine H: [H00 H01 H02; H10 H11 H12]
        float H00 = gmc[0], H01 = gmc[1], H02 = gmc[2];
        float H10 = gmc[3], H11 = gmc[4], H12 = gmc[5];

        float old_cx = x[0], old_cy = x[1];
        x[0] = H00 * old_cx + H01 * old_cy + H02;
        x[1] = H10 * old_cx + H11 * old_cy + H12;

        // Rotate/Scale velocity
        float old_vx = x[4], old_vy = x[5];
        x[4] = H00 * old_vx + H01 * old_vy;
        x[5] = H10 * old_vx + H11 * old_vy;

        // Covariance rotation P' = M P M^T
        auto rotate_cov = [&](float* p_block) {
            float p00 = p_block[0], p01 = p_block[1];
            float p10 = p_block[8], p11 = p_block[9];

            // M * P
            float mp00 = H00 * p00 + H01 * p10;
            float mp01 = H00 * p01 + H01 * p11;
            float mp10 = H10 * p00 + H11 * p10;
            float mp11 = H10 * p01 + H11 * p11;

            // (M * P) * M^T
            p_block[0] = mp00 * H00 + mp01 * H01;
            p_block[1] = mp00 * H10 + mp01 * H11;
            p_block[8] = mp10 * H00 + mp11 * H01;
            p_block[9] = mp10 * H10 + mp11 * H11;
        };
        
        rotate_cov(P);      // Position (0,1)
        rotate_cov(P + 36); // Velocity (4,5)
    }
}

__global__ void update_kernel(float* states, float* covs, int* age, float* scores, 
                              int* trk_indices, float* det_boxes, float* det_scores, 
                              float light_factor, int num_updates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_updates) return;

    int trk_id = trk_indices[idx];
    
    float x1 = det_boxes[idx * 4 + 0];
    float y1 = det_boxes[idx * 4 + 1];
    float x2 = det_boxes[idx * 4 + 2];
    float y2 = det_boxes[idx * 4 + 3];
    
    float w = x2 - x1;
    float h = y2 - y1;
    float cx = x1 + w / 2.0f;
    float cy = y1 + h / 2.0f;
    float a = w / fmaxf(h, 1e-6f);
    
    float z[4] = {cx, cy, a, h};
    
    kf_gpu::update(states + trk_id * 8, covs + trk_id * 64, z, light_factor);
    
    age[trk_id] = 0;
    scores[trk_id] = det_scores[idx];
}

__global__ void init_kernel(float* states, float* covs, bool* active, int* age, float* scores, int* classes, int* track_ids,
                            int* free_indices, float* det_boxes, float* det_scores, int* det_classes, int* det_track_ids, int num_inits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inits) return;

    int trk_id = free_indices[idx];
    
    float x1 = det_boxes[idx * 4 + 0];
    float y1 = det_boxes[idx * 4 + 1];
    float x2 = det_boxes[idx * 4 + 2];
    float y2 = det_boxes[idx * 4 + 3];
    
    float w = x2 - x1;
    float h = y2 - y1;
    
    float* st = states + trk_id * 8;
    st[0] = x1 + w / 2.0f;
    st[1] = y1 + h / 2.0f;
    st[2] = w / fmaxf(h, 1e-6f);
    st[3] = h;
    st[4] = 0.0f;
    st[5] = 0.0f;
    st[6] = 0.0f;
    st[7] = 0.0f;
    
    kf_gpu::init_covariance(covs + trk_id * 64);
    
    active[trk_id] = true;
    age[trk_id] = 0;
    scores[trk_id] = det_scores[idx];
    classes[trk_id] = det_classes[idx];
    track_ids[trk_id] = det_track_ids[idx];
}

// 簡單的 IoU 計算函數 (CPU 端)
float get_iou_cpu(const float* b1, const float* b2) {
    float x1 = std::max(b1[0], b2[0]);
    float y1 = std::max(b1[1], b2[1]);
    float x2 = std::min(b1[2], b2[2]);
    float y2 = std::min(b1[3], b2[3]);
    
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    float area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    return inter / (area1 + area2 - inter + 1e-6f);
}

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

__global__ void assign_duplicate_anchor_kernel(
    const float* boxes,
    const int* classes,
    int num_dets,
    float iou_threshold,
    float center_threshold,
    float area_ratio_threshold,
    int* anchor_indices
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) {
        return;
    }

    const float* candidate = boxes + idx * 4;
    const int candidate_class = classes[idx];
    int anchor = idx;

    const float candidate_w = fmaxf(candidate[2] - candidate[0], 1e-6f);
    const float candidate_h = fmaxf(candidate[3] - candidate[1], 1e-6f);
    const float candidate_area = candidate_w * candidate_h;
    const float candidate_cx = 0.5f * (candidate[0] + candidate[2]);
    const float candidate_cy = 0.5f * (candidate[1] + candidate[3]);

    for (int prev = 0; prev < idx; ++prev) {
        if (classes[prev] != candidate_class) {
            continue;
        }

        const float* other = boxes + prev * 4;
        const float iou = get_iou_device(candidate, other);
        const float other_w = fmaxf(other[2] - other[0], 1e-6f);
        const float other_h = fmaxf(other[3] - other[1], 1e-6f);
        const float other_area = other_w * other_h;
        const float min_w = fminf(candidate_w, other_w);
        const float min_h = fminf(candidate_h, other_h);
        const float center_gate = sqrtf(min_w * min_w + min_h * min_h) * center_threshold;
        const float other_cx = 0.5f * (other[0] + other[2]);
        const float other_cy = 0.5f * (other[1] + other[3]);
        const float center_dx = other_cx - candidate_cx;
        const float center_dy = other_cy - candidate_cy;
        const float center_dist = sqrtf(center_dx * center_dx + center_dy * center_dy);
        const float area_ratio = fminf(
            candidate_area / fmaxf(other_area, 1e-6f),
            other_area / fmaxf(candidate_area, 1e-6f)
        );

        if (iou >= iou_threshold || (center_dist <= center_gate && area_ratio >= area_ratio_threshold)) {
            anchor = prev;
            break;
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
    float* box_sums,
    float* score_sums,
    int* score_bits_max,
    int* cluster_counts
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) {
        return;
    }

    const int anchor = anchor_indices[idx];
    const float score = scores[idx];
    atomicAdd(score_sums + anchor, score);
    atomicAdd(cluster_counts + anchor, 1);
    atomicMax(score_bits_max + anchor, __float_as_int(score));
    for (int k = 0; k < 4; ++k) {
        atomicAdd(box_sums + anchor * 4 + k, boxes[idx * 4 + k] * score);
    }
}

__global__ void compact_duplicate_clusters_kernel(
    const float* box_sums,
    const float* score_sums,
    const int* score_bits_max,
    const int* cluster_counts,
    const int* classes,
    int num_dets,
    float* out_boxes,
    float* out_scores,
    int* out_classes,
    int* out_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int out_idx = 0;
    for (int idx = 0; idx < num_dets; ++idx) {
        const int count = cluster_counts[idx];
        if (count <= 0) {
            continue;
        }

        const float inv_score_sum = 1.0f / fmaxf(score_sums[idx], 1e-6f);
        for (int k = 0; k < 4; ++k) {
            out_boxes[out_idx * 4 + k] = box_sums[idx * 4 + k] * inv_score_sum;
        }
        out_scores[out_idx] = __int_as_float(score_bits_max[idx]);
        out_classes[out_idx] = classes[idx];
        ++out_idx;
    }

    *out_count = out_idx;
}

// --- GPUByteTracker Impl ---

class GPUByteTracker::Impl {
public:
    Impl(int max_objects, int embedding_dim) 
        : max_objs_(max_objects), embed_dim_(embedding_dim), track_id_counter_(1) {
        cudaMalloc(&d_states_, max_objs_ * 8 * sizeof(float));
        cudaMalloc(&d_covs_, max_objs_ * 64 * sizeof(float));
        cudaMalloc(&d_active_, max_objs_ * sizeof(bool));
        cudaMalloc(&d_age_, max_objs_ * sizeof(int));
        cudaMalloc(&d_scores_, max_objs_ * sizeof(float));
        cudaMalloc(&d_classes_, max_objs_ * sizeof(int));
        cudaMalloc(&d_track_ids_, max_objs_ * sizeof(int));
        cudaMalloc(&d_features_, max_objs_ * embed_dim_ * sizeof(float));
        
        cudaMemset(d_active_, 0, max_objs_ * sizeof(bool));
        cudaMemset(d_states_, 0, max_objs_ * 8 * sizeof(float));
        cudaMemset(d_age_, 0, max_objs_ * sizeof(int));
        cudaMemset(d_features_, 0, max_objs_ * embed_dim_ * sizeof(float));
        
        // Host buffers
        h_states_.resize(max_objs_ * 8);
        h_covs_.resize(max_objs_ * 64);
        h_active_.resize(max_objs_);
        h_age_.resize(max_objs_);
        h_scores_.resize(max_objs_);
        h_classes_.resize(max_objs_);
        h_track_ids_.resize(max_objs_);
        h_features_.resize(max_objs_ * embed_dim_);
    }

    ~Impl() {
        cudaFree(d_states_);
        cudaFree(d_covs_);
        cudaFree(d_active_);
        cudaFree(d_age_);
        cudaFree(d_scores_);
        cudaFree(d_classes_);
        cudaFree(d_track_ids_);
        cudaFree(d_features_);
    }

    void update_reference_features(int* track_ids, float* features, int num, cudaStream_t stream) {
        if (num <= 0) return;
        std::vector<int> h_tids(num);
        cudaMemcpyAsync(h_tids.data(), track_ids, num * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int i = 0; i < num; ++i) {
            int tid = h_tids[i];
            for (int j = 0; j < max_objs_; ++j) {
                if (h_active_[j] && h_track_ids_[j] == tid) {
                    cudaMemcpyAsync(d_features_ + j * embed_dim_, features + i * embed_dim_, 
                                   embed_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                    break;
                }
            }
        }
    }

    std::vector<TrackResult> update(float* d_boxes, float* d_scores, int* d_classes, int num_dets, 
                                   cudaStream_t stream, float* d_embeddings, float* d_gmc, float light_factor) {
        int threads = 256;
        int blocks = (max_objs_ + threads - 1) / threads;
        predict_kernel<<<blocks, threads, 0, stream>>>(d_states_, d_covs_, d_active_, d_age_, max_objs_, max_age_);
        
        if (d_gmc) {
            gmc_kernel<<<blocks, threads, 0, stream>>>(d_states_, d_covs_, d_active_, d_gmc, max_objs_);
        }

        std::vector<float> h_det_boxes(num_dets * 4);
        std::vector<float> h_det_scores(num_dets);
        std::vector<int> h_det_classes(num_dets);
        std::vector<float> h_det_embeds;
        if (d_embeddings && num_dets > 0) h_det_embeds.resize(num_dets * embed_dim_);
        
        if (num_dets > 0) {
            cudaMemcpyAsync(h_det_boxes.data(), d_boxes, num_dets * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_det_scores.data(), d_scores, num_dets * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_det_classes.data(), d_classes, num_dets * sizeof(int), cudaMemcpyDeviceToHost, stream);
            if (d_embeddings) {
                cudaMemcpyAsync(h_det_embeds.data(), d_embeddings, num_dets * embed_dim_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
            }
        }
        
        cudaMemcpyAsync(h_states_.data(), d_states_, max_objs_ * 8 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_active_.data(), d_active_, max_objs_ * sizeof(bool), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_age_.data(), d_age_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_classes_.data(), d_classes_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_track_ids_.data(), d_track_ids_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_scores_.data(), d_scores_, max_objs_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_features_.data(), d_features_, max_objs_ * embed_dim_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream); 
        
        std::vector<int> active_tracks;
        std::vector<std::vector<float>> track_boxes;
        
        for (int i = 0; i < max_objs_; ++i) {
            if (h_active_[i]) {
                active_tracks.push_back(i);
                float cx = h_states_[i * 8 + 0], cy = h_states_[i * 8 + 1];
                float a = h_states_[i * 8 + 2], h = h_states_[i * 8 + 3], w = a * h;
                track_boxes.push_back({cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f});
            }
        }

        std::vector<bool> det_matched(num_dets, false);
        std::vector<bool> trk_matched(active_tracks.size(), false);
        std::vector<int> det_high, det_low;
        
        for (int d = 0; d < num_dets; ++d) {
            if (h_det_scores[d] >= high_thresh_) det_high.push_back(d);
            else if (h_det_scores[d] >= track_thresh_) det_low.push_back(d);
        }
        
        auto match_sinkhorn = [&](const std::vector<int>& det_indices, const std::vector<int>& trk_indices, float iou_thresh, std::vector<std::pair<int, int>>& matched, bool use_reid = false) {
            if (det_indices.empty() || trk_indices.empty()) return;
            int n = det_indices.size(), m = trk_indices.size();
            std::vector<std::vector<float>> cost_matrix(n, std::vector<float>(m, 1.0f));

            for (int d = 0; d < n; ++d) {
                int d_idx = det_indices[d];
                for (int t = 0; t < m; ++t) {
                    int t_idx = trk_indices[t], trk_id = active_tracks[t_idx];
                    if (h_det_classes[d_idx] == h_classes_[trk_id]) {
                        float iou = get_iou_cpu(&h_det_boxes[d_idx * 4], track_boxes[t_idx].data());
                        if (iou >= iou_thresh) {
                            cost_matrix[d][t] = 1.0f - iou;
                        }
                    }
                }
            }
            std::vector<int> assignment;
            SinkhornAlgorithm::Solve(cost_matrix, assignment, 30.0f, 50);
            for (int d = 0; d < n; ++d) {
                int t = assignment[d];
                if (t != -1 && cost_matrix[d][t] < 1.0f) matched.push_back({det_indices[d], trk_indices[t]});
            }
        };

        std::vector<int> trk_stage1;
        for (size_t t = 0; t < active_tracks.size(); ++t) trk_stage1.push_back(t);
        std::vector<std::pair<int, int>> matched_stage1;
        match_sinkhorn(det_high, trk_stage1, 0.2f, matched_stage1, true);
        for (const auto& match : matched_stage1) { det_matched[match.first] = true; trk_matched[match.second] = true; }
        
        std::vector<int> trk_stage2;
        for (size_t t = 0; t < active_tracks.size(); ++t) {
            if (!trk_matched[t] && h_age_[active_tracks[t]] <= 1) trk_stage2.push_back(t);
        }
        std::vector<std::pair<int, int>> matched_stage2;
        match_sinkhorn(det_low, trk_stage2, 0.4f, matched_stage2, false);
        for (const auto& match : matched_stage2) { det_matched[match.first] = true; trk_matched[match.second] = true; }
        
        std::vector<int> u_trk_indices;
        std::vector<float> u_det_boxes, u_det_scores, u_det_embeds;
        
        auto collect_update = [&](int det_idx, int trk_idx) {
            int trk_id = active_tracks[trk_idx];
            u_trk_indices.push_back(trk_id);
            u_det_boxes.push_back(h_det_boxes[det_idx * 4 + 0]); u_det_boxes.push_back(h_det_boxes[det_idx * 4 + 1]);
            u_det_boxes.push_back(h_det_boxes[det_idx * 4 + 2]); u_det_boxes.push_back(h_det_boxes[det_idx * 4 + 3]);
            u_det_scores.push_back(h_det_scores[det_idx]);
            if (d_embeddings) for (int k = 0; k < embed_dim_; ++k) u_det_embeds.push_back(h_det_embeds[det_idx * embed_dim_ + k]);
            h_age_[trk_id] = 0; h_scores_[trk_id] = h_det_scores[det_idx];
        };
        for (const auto& match : matched_stage1) collect_update(match.first, match.second);
        for (const auto& match : matched_stage2) collect_update(match.first, match.second);
        
        if (!u_trk_indices.empty()) {
            int num_upd = u_trk_indices.size();
            int* d_u_trk; float *d_u_box, *d_u_score;
            cudaMallocAsync(&d_u_trk, num_upd * sizeof(int), stream);
            cudaMallocAsync(&d_u_box, num_upd * 4 * sizeof(float), stream);
            cudaMallocAsync(&d_u_score, num_upd * sizeof(float), stream);
            cudaMemcpyAsync(d_u_trk, u_trk_indices.data(), num_upd * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_u_box, u_det_boxes.data(), num_upd * 4 * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_u_score, u_det_scores.data(), num_upd * sizeof(float), cudaMemcpyHostToDevice, stream);
            int b_upd = (num_upd + threads - 1) / threads;
            update_kernel<<<b_upd, threads, 0, stream>>>(d_states_, d_covs_, d_age_, d_scores_, d_u_trk, d_u_box, d_u_score, light_factor, num_upd);
            if (d_embeddings) {
                for (int i = 0; i < num_upd; ++i) {
                    cudaMemcpyAsync(d_features_ + u_trk_indices[i] * embed_dim_, u_det_embeds.data() + i * embed_dim_, embed_dim_ * sizeof(float), cudaMemcpyHostToDevice, stream);
                }
            }
            cudaFreeAsync(d_u_trk, stream); cudaFreeAsync(d_u_box, stream); cudaFreeAsync(d_u_score, stream);
        }
        
        std::vector<int> init_free_indices, init_classes, init_tids;
        std::vector<float> init_boxes, init_scores, init_embeds;
        for (int d : det_high) {
            if (!det_matched[d] && h_det_scores[d] > 0.6f) {
                for (int i = 0; i < max_objs_; ++i) {
                    if (!h_active_[i]) {
                        h_active_[i] = true; init_free_indices.push_back(i);
                        init_boxes.push_back(h_det_boxes[d * 4 + 0]); init_boxes.push_back(h_det_boxes[d * 4 + 1]);
                        init_boxes.push_back(h_det_boxes[d * 4 + 2]); init_boxes.push_back(h_det_boxes[d * 4 + 3]);
                        init_scores.push_back(h_det_scores[d]); init_classes.push_back(h_det_classes[d]);
                        init_tids.push_back(track_id_counter_++);
                        if (d_embeddings) for (int k = 0; k < embed_dim_; ++k) init_embeds.push_back(h_det_embeds[d * embed_dim_ + k]);
                        h_track_ids_[i] = init_tids.back(); h_classes_[i] = init_classes.back();
                        h_scores_[i] = init_scores.back(); h_age_[i] = 0; break;
                    }
                }
            }
        }
        
        if (!init_free_indices.empty()) {
            int num_init = init_free_indices.size();
            int *d_i_free, *d_i_cls, *d_i_tid; float *d_i_box, *d_i_score;
            cudaMallocAsync(&d_i_free, num_init * sizeof(int), stream);
            cudaMallocAsync(&d_i_box, num_init * 4 * sizeof(float), stream);
            cudaMallocAsync(&d_i_score, num_init * sizeof(float), stream);
            cudaMallocAsync(&d_i_cls, num_init * sizeof(int), stream);
            cudaMallocAsync(&d_i_tid, num_init * sizeof(int), stream);
            cudaMemcpyAsync(d_i_free, init_free_indices.data(), num_init * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_i_box, init_boxes.data(), num_init * 4 * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_i_score, init_scores.data(), num_init * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_i_cls, init_classes.data(), num_init * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_i_tid, init_tids.data(), num_init * sizeof(int), cudaMemcpyHostToDevice, stream);
            int b_init = (num_init + threads - 1) / threads;
            init_kernel<<<b_init, threads, 0, stream>>>(d_states_, d_covs_, d_active_, d_age_, d_scores_, d_classes_, d_track_ids_, d_i_free, d_i_box, d_i_score, d_i_cls, d_i_tid, num_init);
            if (d_embeddings) {
                for (int i = 0; i < num_init; ++i) {
                    cudaMemcpyAsync(d_features_ + init_free_indices[i] * embed_dim_, init_embeds.data() + i * embed_dim_, embed_dim_ * sizeof(float), cudaMemcpyHostToDevice, stream);
                }
            }
            cudaFreeAsync(d_i_free, stream); cudaFreeAsync(d_i_box, stream); cudaFreeAsync(d_i_score, stream);
            cudaFreeAsync(d_i_cls, stream); cudaFreeAsync(d_i_tid, stream);
        }

        cudaMemcpyAsync(h_states_.data(), d_states_, max_objs_ * 8 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_covs_.data(), d_covs_, max_objs_ * 64 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<TrackResult> results;
        for (int i = 0; i < max_objs_; ++i) {
            if (h_active_[i] && h_age_[i] == 0) {
                float cx = h_states_[i * 8 + 0], cy = h_states_[i * 8 + 1], a = h_states_[i * 8 + 2], h = h_states_[i * 8 + 3], w = a * h;
                results.push_back({cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f, h_track_ids_[i], h_scores_[i], h_classes_[i]});
            }
        }
        return results;
    }

    void set_params(float track_thresh, float high_thresh, float match_thresh, int track_buffer) {
        track_thresh_ = track_thresh; high_thresh_ = high_thresh; match_thresh_ = match_thresh; max_age_ = track_buffer;
    }

    std::vector<TrackStateSnapshot> get_state_snapshots(cudaStream_t stream) {
        cudaMemcpyAsync(h_states_.data(), d_states_, max_objs_ * 8 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_covs_.data(), d_covs_, max_objs_ * 64 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_active_.data(), d_active_, max_objs_ * sizeof(bool), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_age_.data(), d_age_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_classes_.data(), d_classes_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_track_ids_.data(), d_track_ids_, max_objs_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_scores_.data(), d_scores_, max_objs_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<TrackStateSnapshot> snapshots;
        for (int i = 0; i < max_objs_; ++i) {
            if (!h_active_[i]) continue;
            TrackStateSnapshot snap;
            snap.obj_id = h_track_ids_[i]; snap.class_id = h_classes_[i];
            snap.age = h_age_[i]; snap.score = h_scores_[i];
            snap.state.assign(h_states_.begin() + i * 8, h_states_.begin() + (i + 1) * 8);
            snap.covariance.assign(h_covs_.begin() + i * 64, h_covs_.begin() + (i + 1) * 64);
            snapshots.push_back(std::move(snap));
        }
        return snapshots;
    }

private:
    int max_objs_, embed_dim_, track_id_counter_;
    float track_thresh_ = 0.1f, high_thresh_ = 0.5f, match_thresh_ = 0.8f;
    int max_age_ = 30;
    float *d_states_, *d_covs_, *d_scores_, *d_features_;
    bool* d_active_;
    int *d_age_, *d_classes_, *d_track_ids_;
    std::vector<float> h_states_, h_covs_, h_scores_, h_features_;
    std::vector<uint8_t> h_active_;
    std::vector<int> h_age_, h_classes_, h_track_ids_;
};

GPUByteTracker::GPUByteTracker(int max_objs, int embedding_dim) : pimpl_(std::make_unique<Impl>(max_objs, embedding_dim)) {}
GPUByteTracker::~GPUByteTracker() = default;

void GPUByteTracker::set_params(float track_thresh, float high_thresh, float match_thresh, int track_buffer) {
    pimpl_->set_params(track_thresh, high_thresh, match_thresh, track_buffer);
}

void GPUByteTracker::update_reference_features(int* track_ids, float* features, int num, cudaStream_t stream) {
    pimpl_->update_reference_features(track_ids, features, num, stream);
}

std::vector<TrackResult> GPUByteTracker::update(float* b, float* s, int* c, int n, cudaStream_t stream, float* e, float* g, float l) {
    return pimpl_->update(b, s, c, n, stream, e, g, l);
}

std::vector<TrackStateSnapshot> GPUByteTracker::get_state_snapshots(cudaStream_t stream) {
    return pimpl_->get_state_snapshots(stream);
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
    int* cluster_counts_ptr,
    float* out_boxes_ptr,
    float* out_scores_ptr,
    int* out_classes_ptr,
    int* out_count_ptr,
    float iou_threshold,
    float center_threshold,
    float area_ratio_threshold,
    cudaStream_t stream
) {
    if (num_dets <= 0) {
        checkCuda(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), stream));
        return;
    }

    checkCuda(cudaMemsetAsync(box_sums_ptr, 0, num_dets * 4 * sizeof(float), stream));
    checkCuda(cudaMemsetAsync(score_sums_ptr, 0, num_dets * sizeof(float), stream));
    checkCuda(cudaMemsetAsync(score_bits_max_ptr, 0, num_dets * sizeof(int), stream));
    checkCuda(cudaMemsetAsync(cluster_counts_ptr, 0, num_dets * sizeof(int), stream));
    checkCuda(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), stream));

    const int threads = 256;
    const int blocks = (num_dets + threads - 1) / threads;
    assign_duplicate_anchor_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr,
        classes_ptr,
        num_dets,
        iou_threshold,
        center_threshold,
        area_ratio_threshold,
        anchor_indices_ptr
    );
    aggregate_duplicate_clusters_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr,
        scores_ptr,
        classes_ptr,
        anchor_indices_ptr,
        num_dets,
        box_sums_ptr,
        score_sums_ptr,
        score_bits_max_ptr,
        cluster_counts_ptr
    );
    compact_duplicate_clusters_kernel<<<1, 1, 0, stream>>>(
        box_sums_ptr,
        score_sums_ptr,
        score_bits_max_ptr,
        cluster_counts_ptr,
        classes_ptr,
        num_dets,
        out_boxes_ptr,
        out_scores_ptr,
        out_classes_ptr,
        out_count_ptr
    );

    checkCuda(cudaGetLastError());
}

} // namespace saccade
