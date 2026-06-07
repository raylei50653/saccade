#include "tracking/multi_signal_birth_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>

namespace saccade {

namespace {

__device__ inline float box_iou(const float* a, const float* b) {
    const float x1 = fmaxf(a[0], b[0]);
    const float y1 = fmaxf(a[1], b[1]);
    const float x2 = fminf(a[2], b[2]);
    const float y2 = fminf(a[3], b[3]);
    const float w = fmaxf(0.0f, x2 - x1);
    const float h = fmaxf(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = fmaxf(0.0f, a[2] - a[0]) * fmaxf(0.0f, a[3] - a[1]);
    const float area_b = fmaxf(0.0f, b[2] - b[0]) * fmaxf(0.0f, b[3] - b[1]);
    return inter / fmaxf(area_a + area_b - inter, 1e-6f);
}

__global__ void multi_signal_iou_match_kernel(
    const float* __restrict__ boxes,
    const float* __restrict__ scores,
    int num_dets,
    const float* __restrict__ cand_boxes,
    int num_candidates,
    int* __restrict__ compact_idx,
    int* __restrict__ compact_count,
    float min_score,
    float new_track_thresh,
    float iou_match) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) return;

    const float score = scores[idx];
    if (score >= new_track_thresh || score < min_score) return;

    const float* box = boxes + static_cast<size_t>(idx) * 4;

    if (num_candidates == 0) {
        const int out = atomicAdd(compact_count, 1);
        compact_idx[out] = idx;
        return;
    }

    int best_c = -1;
    float best_i = -1.0f;

    for (int c = 0; c < num_candidates; ++c) {
        const float* cand_box = cand_boxes + static_cast<size_t>(c) * 4;
        const float iou = box_iou(box, cand_box);
        if (iou > best_i) {
            best_i = iou;
            best_c = c;
        }
    }

    if (best_i >= iou_match && best_c >= 0) {
        // Store: det_idx + best_cand_idx packed. 
        // compact_idx encodes det_idx in high bits, cand_idx in low bits.
        const int packed = (idx << 16) | (best_c & 0xFFFF);
        const int out = atomicAdd(compact_count, 1);
        compact_idx[out] = packed;
    } else {
        // No match — flag as new candidate. 
        // compact_idx encodes det_idx in high bits, -1 in low bits.
        const int packed = (idx << 16) | 0xFFFF;
        const int out = atomicAdd(compact_count, 1);
        compact_idx[out] = packed;
    }
}

static inline float geometry_quality(const float* box, float min_aspect, int max_area_px) {
    const float w = fmaxf(box[2] - box[0], 1.0f);
    const float h = fmaxf(box[3] - box[1], 1.0f);
    const float aspect = h / w;
    if (min_aspect > 0.0f && aspect < min_aspect) return -1.0f;
    if (max_area_px > 0) {
        const float area = w * h;
        if (area > static_cast<float>(max_area_px)) return -1.0f;
    }
    if (aspect < 1.0f) return 0.0f;
    if (aspect < 2.0f) return aspect - 1.0f;
    if (aspect <= 4.0f) return 1.0f;
    return fmaxf(0.0f, 1.0f - (aspect - 4.0f) / 3.0f);
}

static inline float compute_evidence(
    const MultiSignalBirthPool::Candidate& cand,
    const MultiSignalBirthPool::Config& cfg) {

    const int n = static_cast<int>(cand.history_scores.size());
    if (n < cfg.min_frames) return 0.0f;

    const float streak = fminf(1.0f, static_cast<float>(n - 1) / fmaxf(static_cast<float>(cfg.min_frames - 1), 1.0f));

    float best_score = 0.0f;
    for (int i = 0; i < n; ++i) {
        best_score = fmaxf(best_score, cand.history_scores[i]);
    }
    const float score_range = fmaxf(cfg.new_track_thresh - cfg.min_score, 0.01f);
    const float score_norm = fminf(1.0f, fmaxf(0.0f, (best_score - cfg.min_score) / score_range));

    const int last = n - 1;
    const float gq = geometry_quality(
        cand.history_boxes.data() + last * 4,
        cfg.min_aspect, cfg.max_area_px);
    if (gq < 0.0f) return 0.0f;

    float motion_norm = 0.0f;
    if (n >= 2) {
        float total = 0.0f;
        for (int i = 1; i < n; ++i) {
            const float* b0 = cand.history_boxes.data() + (i - 1) * 4;
            const float* b1 = cand.history_boxes.data() + i * 4;
            const float cx0 = 0.5f * (b0[0] + b0[2]);
            const float cy0 = 0.5f * (b0[1] + b0[3]);
            const float cx1 = 0.5f * (b1[0] + b1[2]);
            const float cy1 = 0.5f * (b1[1] + b1[3]);
            total += sqrtf((cx1 - cx0) * (cx1 - cx0) + (cy1 - cy0) * (cy1 - cy0));
        }
        motion_norm = fminf(1.0f, (total / static_cast<float>(n - 1)) / fmaxf(cfg.target_motion_px, 1.0f));
    }

    return cfg.w_score * score_norm
         + cfg.w_motion * motion_norm
         + cfg.w_quality * gq
         + cfg.w_streak * streak;
}

} // namespace

MultiSignalBirthPool::MultiSignalBirthPool(const Config& cfg)
    : cfg_(cfg) {
    ensure_capacity(std::max(cfg_.capacity, 1));
    reset();
}

MultiSignalBirthPool::~MultiSignalBirthPool() {
    if (d_cand_last_boxes_) cudaFree(d_cand_last_boxes_);
    if (d_compact_idx_) cudaFree(d_compact_idx_);
    if (d_compact_count_) cudaFree(d_compact_count_);
}

void MultiSignalBirthPool::ensure_capacity(int num_dets) {
    if (num_dets <= capacity_ && cfg_.max_candidates <= cand_capacity_) {
        return;
    }
    capacity_ = std::max(num_dets, cfg_.capacity);
    cand_capacity_ = std::max(cfg_.max_candidates, 16);

    if (d_cand_last_boxes_) cudaFree(d_cand_last_boxes_);
    if (d_compact_idx_) cudaFree(d_compact_idx_);
    if (d_compact_count_) cudaFree(d_compact_count_);

    cudaMalloc(&d_cand_last_boxes_, static_cast<size_t>(cand_capacity_) * 4 * sizeof(float));
    cudaMalloc(&d_compact_idx_, static_cast<size_t>(capacity_) * sizeof(int));
    cudaMalloc(&d_compact_count_, sizeof(int));
}

void MultiSignalBirthPool::reset() {
    candidates_.clear();
    active_cand_count_ = 0;
}

void MultiSignalBirthPool::update_and_apply(
    int frame_id,
    const float* boxes_ptr,
    const float* scores_ptr,
    int num_dets,
    bool* promote_mask_ptr,
    bool* replace_mask_ptr,
    cudaStream_t stream) {

    if (!boxes_ptr || !scores_ptr || !promote_mask_ptr || !replace_mask_ptr || num_dets <= 0) {
        return;
    }
    ensure_capacity(num_dets);

    const int threads = 256;
    const int blocks = (num_dets + threads - 1) / threads;

    cudaMemsetAsync(d_compact_count_, 0, sizeof(int), stream);

    multi_signal_iou_match_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr,
        scores_ptr,
        num_dets,
        d_cand_last_boxes_,
        active_cand_count_,
        d_compact_idx_,
        d_compact_count_,
        cfg_.min_score,
        cfg_.new_track_thresh,
        cfg_.iou_match);

    int compact_count_host = 0;
    cudaMemcpyAsync(&compact_count_host, d_compact_count_, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<int> compact_packed(compact_count_host);

    if (compact_count_host > 0) {
        cudaMemcpy(compact_packed.data(), d_compact_idx_,
                   static_cast<size_t>(compact_count_host) * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    std::vector<float> host_scores(num_dets);
    std::vector<float> host_boxes(static_cast<size_t>(num_dets) * 4);

    cudaMemcpy(host_scores.data(), scores_ptr, num_dets * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_boxes.data(), boxes_ptr,
               static_cast<size_t>(num_dets) * 4 * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::set<int> matched_cand_set;
    std::vector<int> to_remove;
    std::vector<uint8_t> promote_mask(num_dets, 0);
    std::vector<uint8_t> replace_mask(num_dets, 0);

    for (int k = 0; k < compact_count_host; ++k) {
        const int packed = compact_packed[k];
        const int det_idx = packed >> 16;
        const int cand_idx = packed & 0xFFFF;
        const bool is_new = (cand_idx == 0xFFFF);

        if (det_idx < 0 || det_idx >= num_dets) continue;

        const float score = host_scores[det_idx];
        const int off = det_idx * 4;
        const float box[4] = {
            host_boxes[off + 0], host_boxes[off + 1],
            host_boxes[off + 2], host_boxes[off + 3]};

        if (is_new) {
            Candidate cand;
            cand.history_boxes = {box[0], box[1], box[2], box[3]};
            cand.history_scores = {score};
            cand.history_frames = {frame_id};
            cand.last_frame = frame_id;
            candidates_.push_back(cand);
            continue;
        }

        if (cand_idx >= static_cast<int>(candidates_.size())) continue;
        if (matched_cand_set.find(cand_idx) != matched_cand_set.end()) {
            // Another detection already claimed this candidate
            Candidate cand;
            cand.history_boxes = {box[0], box[1], box[2], box[3]};
            cand.history_scores = {score};
            cand.history_frames = {frame_id};
            cand.last_frame = frame_id;
            candidates_.push_back(cand);
            continue;
        }

        matched_cand_set.insert(cand_idx);
        auto& cand = candidates_[cand_idx];

        cand.history_boxes.insert(cand.history_boxes.end(), {box[0], box[1], box[2], box[3]});
        cand.history_scores.push_back(score);
        cand.history_frames.push_back(frame_id);
        cand.last_frame = frame_id;

        const float ev = compute_evidence(cand, cfg_);
        if (cfg_.replace_mode && ev >= cfg_.replace_evidence_threshold) {
            promote_mask[det_idx] = 1;
            replace_mask[det_idx] = 1;
            to_remove.push_back(cand_idx);
        } else if (ev >= cfg_.evidence_threshold) {
            promote_mask[det_idx] = 1;
            to_remove.push_back(cand_idx);
        }
    }

    // Remove promoted candidates
    std::sort(to_remove.begin(), to_remove.end());
    for (auto rit = to_remove.rbegin(); rit != to_remove.rend(); ++rit) {
        candidates_.erase(candidates_.begin() + *rit);
    }

    // Expire old candidates
    candidates_.erase(
        std::remove_if(candidates_.begin(), candidates_.end(),
            [this, frame_id](const Candidate& c) {
                return frame_id - c.last_frame > cfg_.ttl_frames;
            }),
        candidates_.end());

    // Build last-box buffer for next frame
    active_cand_count_ = static_cast<int>(candidates_.size());
    if (active_cand_count_ > 0 && active_cand_count_ <= cand_capacity_) {
        std::vector<float> host_last_boxes(static_cast<size_t>(active_cand_count_) * 4);
        for (int c = 0; c < active_cand_count_; ++c) {
            const auto& cand = candidates_[c];
            const int last = static_cast<int>(cand.history_boxes.size()) / 4 - 1;
            const float* src = cand.history_boxes.data() + last * 4;
            host_last_boxes[static_cast<size_t>(c) * 4 + 0] = src[0];
            host_last_boxes[static_cast<size_t>(c) * 4 + 1] = src[1];
            host_last_boxes[static_cast<size_t>(c) * 4 + 2] = src[2];
            host_last_boxes[static_cast<size_t>(c) * 4 + 3] = src[3];
        }
        cudaMemcpyAsync(d_cand_last_boxes_, host_last_boxes.data(),
                        static_cast<size_t>(active_cand_count_) * 4 * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    cudaMemcpyAsync(promote_mask_ptr, promote_mask.data(),
                    static_cast<size_t>(num_dets) * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(replace_mask_ptr, replace_mask.data(),
                    static_cast<size_t>(num_dets) * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);

    // These async H2D copies source from host_last_boxes / promote_mask /
    // replace_mask, which are stack-local and destroyed on return. Synchronize
    // so the copies complete before the source buffers are freed (and before
    // the caller consumes the masks downstream).
    cudaStreamSynchronize(stream);
}

} // namespace saccade
