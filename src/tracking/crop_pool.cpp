#include "tracking/crop_pool.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace saccade {

CropPool::CropPool(int capacity, int crop_h, int crop_w)
    : capacity_(capacity), crop_h_(crop_h), crop_w_(crop_w) {
    if (capacity_ <= 0 || crop_h_ <= 0 || crop_w_ <= 0) return;
    size_t bytes = static_cast<size_t>(capacity_) * crop_elem_count() * sizeof(float);
    cudaMalloc(&d_buffer_, bytes);
    cudaMemset(d_buffer_, 0, bytes);
    free_slots_.reserve(capacity_);
    for (int i = 0; i < capacity_; ++i) {
        free_slots_.push_back(i);
    }
}

CropPool::~CropPool() {
    if (d_buffer_) cudaFree(d_buffer_);
}

int CropPool::acquire(int n) {
    if (n <= 0) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    // Find n contiguous free slots.
    if (static_cast<int>(free_slots_.size()) < n) return -1;
    // Search for a contiguous run in the free list.
    // The free list is not sorted, so we search by sorting a copy.
    // For small pools (256-512), this is fast enough.
    std::vector<int> sorted_free(free_slots_);
    std::sort(sorted_free.begin(), sorted_free.end());
    int run_start = -1;
    int run_len = 1;
    for (int i = 1; i < static_cast<int>(sorted_free.size()); ++i) {
        if (sorted_free[i] == sorted_free[i - 1] + 1) {
            ++run_len;
            if (run_len >= n) {
                run_start = sorted_free[i] - n + 1;
                break;
            }
        } else {
            run_len = 1;
        }
    }
    if (n == 1) run_start = sorted_free[0];
    if (run_start < 0) return -1;

    // Remove the acquired slots from free_slots_.
    for (int s = run_start; s < run_start + n; ++s) {
        auto it = std::find(free_slots_.begin(), free_slots_.end(), s);
        if (it != free_slots_.end()) free_slots_.erase(it);
    }
    return run_start;
}

void CropPool::release(int start, int n) {
    if (n <= 0 || start < 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int s = start; s < start + n; ++s) {
        if (s >= 0 && s < capacity_) {
            free_slots_.push_back(s);
        }
    }
}

float* CropPool::slot_ptr(int slot) {
    if (!d_buffer_ || slot < 0 || slot >= capacity_) return nullptr;
    return d_buffer_ + static_cast<size_t>(slot) * crop_elem_count();
}

} // namespace saccade
