#include "tracking/temporal_birth_pool.hpp"

#include <algorithm>
#include <cmath>

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

__global__ void apply_temporal_birth_boost_kernel(
    const float* boxes,
    float* scores,
    int num_dets,
    const float* ring_boxes,
    const int* ring_counts,
    int capacity,
    int window_capacity,
    int filled,
    int next_slot,
    int required_window,
    float iou_thresh,
    float boost,
    float min_score,
    float min_motion_px,
    float new_track_thresh,
    float high_thresh) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) {
        return;
    }
    const float score = scores[idx];
    if (score >= new_track_thresh || score < min_score) {
        return;
    }
    if (required_window <= 0) {
        scores[idx] = fminf(score + boost, high_thresh);
        return;
    }
    if (filled < required_window) {
        return;
    }

    const float* box = boxes + static_cast<size_t>(idx) * 4;
    float oldest_cx = 0.0f;
    float oldest_cy = 0.0f;

    for (int frame_idx = 0; frame_idx < required_window; ++frame_idx) {
        const int slot = (next_slot - required_window + frame_idx + window_capacity) % window_capacity;
        const int prev_count = ring_counts[slot];
        if (prev_count <= 0) {
            return;
        }
        const float* prev_boxes = ring_boxes + static_cast<size_t>(slot) * capacity * 4;
        float best_iou = -1.0f;
        int best_idx = -1;
        for (int j = 0; j < prev_count; ++j) {
            const float* prev_box = prev_boxes + static_cast<size_t>(j) * 4;
            const float iou = box_iou(box, prev_box);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = j;
            }
        }
        if (best_iou < iou_thresh || best_idx < 0) {
            return;
        }
        if (frame_idx == 0 && min_motion_px > 0.0f) {
            const float* oldest_box = prev_boxes + static_cast<size_t>(best_idx) * 4;
            oldest_cx = 0.5f * (oldest_box[0] + oldest_box[2]);
            oldest_cy = 0.5f * (oldest_box[1] + oldest_box[3]);
        }
    }

    if (min_motion_px > 0.0f) {
        const float curr_cx = 0.5f * (box[0] + box[2]);
        const float curr_cy = 0.5f * (box[1] + box[3]);
        const float dx = curr_cx - oldest_cx;
        const float dy = curr_cy - oldest_cy;
        if (sqrtf(dx * dx + dy * dy) < min_motion_px) {
            return;
        }
    }

    scores[idx] = fminf(score + boost, high_thresh);
}

__global__ void store_subthreshold_boxes_kernel(
    const float* boxes,
    const float* scores,
    int num_dets,
    float* slot_boxes,
    int* slot_count,
    int capacity,
    float new_track_thresh) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets || scores[idx] >= new_track_thresh) {
        return;
    }
    const int out_idx = atomicAdd(slot_count, 1);
    if (out_idx >= capacity) {
        return;
    }
    const float* src = boxes + static_cast<size_t>(idx) * 4;
    float* dst = slot_boxes + static_cast<size_t>(out_idx) * 4;
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
}

} // namespace

TemporalBirthPool::TemporalBirthPool(const Config& cfg)
    : cfg_(cfg) {
    required_window_ = std::max(cfg_.frames - 1, 0);
    window_capacity_ = std::max(required_window_, 1);
    ensure_capacity(std::max(cfg_.capacity, 1));
    reset();
}

TemporalBirthPool::~TemporalBirthPool() {
    if (d_ring_boxes_) cudaFree(d_ring_boxes_);
    if (d_ring_counts_) cudaFree(d_ring_counts_);
}

void TemporalBirthPool::ensure_capacity(int num_dets) {
    if (num_dets <= capacity_) {
        return;
    }
    const int new_capacity = std::max(num_dets, cfg_.capacity);
    if (d_ring_boxes_) cudaFree(d_ring_boxes_);
    if (d_ring_counts_) cudaFree(d_ring_counts_);
    cudaMalloc(&d_ring_boxes_, static_cast<size_t>(window_capacity_) * new_capacity * 4 * sizeof(float));
    cudaMalloc(&d_ring_counts_, static_cast<size_t>(window_capacity_) * sizeof(int));
    capacity_ = new_capacity;
    next_slot_ = 0;
    filled_ = 0;
    cudaMemset(d_ring_counts_, 0, static_cast<size_t>(window_capacity_) * sizeof(int));
}

void TemporalBirthPool::reset() {
    next_slot_ = 0;
    filled_ = 0;
    if (d_ring_counts_) {
        cudaMemset(d_ring_counts_, 0, static_cast<size_t>(window_capacity_) * sizeof(int));
    }
}

void TemporalBirthPool::update_and_apply(
    const float* boxes_ptr,
    float* scores_ptr,
    int num_dets,
    cudaStream_t stream) {
    if (!boxes_ptr || !scores_ptr || num_dets <= 0) {
        return;
    }
    ensure_capacity(num_dets);

    const int threads = 256;
    const int blocks = (num_dets + threads - 1) / threads;
    if (cfg_.boost > 0.0f) {
        apply_temporal_birth_boost_kernel<<<blocks, threads, 0, stream>>>(
            boxes_ptr,
            scores_ptr,
            num_dets,
            d_ring_boxes_,
            d_ring_counts_,
            capacity_,
            window_capacity_,
            filled_,
            next_slot_,
            required_window_,
            cfg_.iou_thresh,
            cfg_.boost,
            cfg_.min_score,
            cfg_.min_motion_px,
            cfg_.new_track_thresh,
            cfg_.high_thresh);
    }

    cudaMemsetAsync(d_ring_counts_ + next_slot_, 0, sizeof(int), stream);
    float* slot_boxes = d_ring_boxes_ + static_cast<size_t>(next_slot_) * capacity_ * 4;
    store_subthreshold_boxes_kernel<<<blocks, threads, 0, stream>>>(
        boxes_ptr,
        scores_ptr,
        num_dets,
        slot_boxes,
        d_ring_counts_ + next_slot_,
        capacity_,
        cfg_.new_track_thresh);

    next_slot_ = (next_slot_ + 1) % window_capacity_;
    if (filled_ < window_capacity_) {
        ++filled_;
    }
}

} // namespace saccade
