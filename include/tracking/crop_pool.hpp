#pragma once

#include "saccade/common.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <vector>

namespace saccade {

/**
 * @brief CropPool — pre-allocated GPU ring of ReID crop tensors.
 *
 * Each slot is a [3, crop_h, crop_w] float32 tensor (CHW, [0,1], same format
 * as FeatureExtractor::extract input).  acquire(n) reserves n contiguous
 * slots and returns the starting index; slot_ptr(start) gives a device
 * pointer to the [n, 3, crop_h, crop_w] contiguous run that can be passed
 * directly to FeatureExtractor::extract.
 *
 * Phase 2: per-slot free-list.  acquire(n) finds n contiguous free slots;
 * release(slot, n) returns them individually.  Thread-safe via mutex
 * (main thread acquires on crop_stream, ReID worker releases on reid_stream).
 */
class SACCADE_TRACKING_API CropPool {
public:
    CropPool(int capacity, int crop_h, int crop_w);
    ~CropPool();

    CropPool(const CropPool&) = delete;
    CropPool& operator=(const CropPool&) = delete;
    CropPool(CropPool&&) = delete;
    CropPool& operator=(CropPool&&) = delete;

    /**
     * @brief Reserve n contiguous slots.  Returns starting slot index, or -1
     *        if the pool cannot satisfy the request.
     */
    int acquire(int n);

    /**
     * @brief Return n slots starting at @p start to the pool.
     * Supports per-slot release — slots are returned to the free-list
     * individually and can be reused by later acquire() calls.
     */
    void release(int start, int n);

    /// Device pointer to slot @p slot (or the start of a contiguous run).
    float* slot_ptr(int slot);

    /// Number of slots currently in use.
    int in_use() const { return capacity_ - static_cast<int>(free_slots_.size()); }

    /// Number of slots available.
    int available() const { return static_cast<int>(free_slots_.size()); }

    int capacity() const { return capacity_; }
    int crop_h() const { return crop_h_; }
    int crop_w() const { return crop_w_; }
    int crop_elem_count() const { return 3 * crop_h_ * crop_w_; }

private:
    float* d_buffer_ = nullptr;   // [capacity, 3, crop_h, crop_w] float32 GPU
    int    capacity_  = 0;
    int    crop_h_    = 0;
    int    crop_w_    = 0;
    std::vector<int>  free_slots_;  // indices of available slots
    std::mutex        mutex_;
};

} // namespace saccade
