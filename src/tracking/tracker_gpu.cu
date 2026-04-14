#include "tracking/tracker_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

namespace saccade {

// --- CUDA Kernels ---

__global__ void predict_kernel(float* states, bool* active, int* age, int max_objs, int max_age) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_objs) return;

    if (active[idx]) {
        // states: [x, y, a, h, dx, dy, da, dh]
        states[idx * 8 + 0] += states[idx * 8 + 4]; // x
        states[idx * 8 + 1] += states[idx * 8 + 5]; // y
        states[idx * 8 + 2] += states[idx * 8 + 6]; // a
        states[idx * 8 + 3] += states[idx * 8 + 7]; // h
        
        age[idx]++;
        if (age[idx] >= max_age) {
            active[idx] = false;
        }
    }
}

// 簡單的 IoU 計算 Kernel (一對多)
__device__ float get_iou(const float* b1, const float* b2) {
    float x1 = max(b1[0], b2[0]);
    float y1 = max(b1[1], b2[1]);
    float x2 = min(b1[2], b2[2]);
    float y2 = min(b1[3], b2[3]);
    
    float inter = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    float area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    return inter / (area1 + area2 - inter + 1e-6f);
}

// --- GPUByteTracker Impl ---

class GPUByteTracker::Impl {
public:
    Impl(int max_objects) : max_objs_(max_objects) {
        cudaMalloc(&d_states_, max_objs_ * 8 * sizeof(float));
        cudaMalloc(&d_active_, max_objs_ * sizeof(bool));
        cudaMalloc(&d_age_, max_objs_ * sizeof(int));
        cudaMalloc(&d_scores_, max_objs_ * sizeof(float));
        cudaMalloc(&d_classes_, max_objs_ * sizeof(int));
        
        cudaMemset(d_active_, 0, max_objs_ * sizeof(bool));
        cudaMemset(d_states_, 0, max_objs_ * 8 * sizeof(float));
    }

    ~Impl() {
        cudaFree(d_states_);
        cudaFree(d_active_);
        cudaFree(d_age_);
        cudaFree(d_scores_);
        cudaFree(d_classes_);
    }

    std::vector<TrackResult> update(float* d_boxes, float* d_scores, int* d_classes, int num_dets, cudaStream_t stream) {
        if (num_dets == 0) return {};

        // 1. 預測
        int threads = 256;
        int blocks = (max_objs_ + threads - 1) / threads;
        predict_kernel<<<blocks, threads, 0, stream>>>(d_states_, d_active_, d_age_, max_objs_, 30);

        // 2. 匹配邏輯 (此處為了穩定性，將 IoU 矩陣搬回 CPU 做 Greedy Match)
        // 雖然不是完全 Zero-Sync，但對於 num_dets < 100 來說，同步開銷極小。
        // 未來可在 Phase 6 實作全 CUDA 匹配。
        
        // 暫時返回空以通過編譯，待後續細化匹配邏輯
        return {};
    }

private:
    int max_objs_;
    float* d_states_;
    bool* d_active_;
    int* d_age_;
    float* d_scores_;
    int* d_classes_;
};

GPUByteTracker::GPUByteTracker(int max_objs) : pimpl_(std::make_unique<Impl>(max_objs)) {}
GPUByteTracker::~GPUByteTracker() = default;
std::vector<TrackResult> GPUByteTracker::update(float* b, float* s, int* c, int n, cudaStream_t stream) {
    return pimpl_->update(b, s, c, n, stream);
}

} // namespace saccade
