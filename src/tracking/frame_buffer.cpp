#include "tracking/frame_buffer.hpp"
#include <cuda_runtime.h>
#include <cstring>

namespace saccade {

FrameBufferPool::FrameBufferPool(int capacity, int frame_h, int frame_w, int channels)
    : capacity_(capacity), frame_h_(frame_h), frame_w_(frame_w), channels_(channels) {
    if (capacity_ <= 0 || frame_h_ <= 0 || frame_w_ <= 0) return;
    size_t bytes = static_cast<size_t>(capacity_) * channels_ * frame_h_ * frame_w_ * sizeof(float);
    cudaMalloc(&d_buffer_, bytes);
    cudaMemset(d_buffer_, 0, bytes);
    free_list_.reserve(capacity_);
    for (int i = 0; i < capacity_; ++i) {
        free_list_.push_back(d_buffer_ + static_cast<size_t>(i) * channels_ * frame_h_ * frame_w_);
    }
}

FrameBufferPool::~FrameBufferPool() {
    for (auto& pr : pending_) {
        cudaEventSynchronize(pr.event);
        cudaEventDestroy(pr.event);
    }
    if (d_buffer_) cudaFree(d_buffer_);
}

float* FrameBufferPool::acquire() {
    poll_releases();
    if (free_list_.empty()) return nullptr;
    float* buf = free_list_.back();
    free_list_.pop_back();
    return buf;
}

void FrameBufferPool::release(float* buf) {
    if (!buf) return;
    free_list_.push_back(buf);
}

void FrameBufferPool::release_after(float* buf, cudaEvent_t event) {
    if (!buf) return;
    if (!event) {
        release(buf);
        return;
    }
    pending_.push_back({buf, event});
}

void FrameBufferPool::poll_releases() {
    for (auto it = pending_.begin(); it != pending_.end();) {
        cudaError_t err = cudaEventQuery(it->event);
        if (err == cudaSuccess) {
            free_list_.push_back(it->buf);
            cudaEventDestroy(it->event);
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace saccade
