#pragma once

#include "saccade/common.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace saccade {

/**
 * @brief FrameBufferPool — pre-allocated GPU frame tensors with event-based
 *        release semantics.
 *
 * Each buffer is a [3, frame_h, frame_w] float32 tensor (CHW, [0,1] — the
 * same format PerceptionPipeline::extract_reid / crop_into_pool expects).
 *
 * acquire() returns a device pointer or nullptr if all buffers are in use.
 * release(buf) returns a buffer immediately (sync path).
 * release_after(buf, event) defers the return until the CUDA event is
 * reached — this is the key primitive that lets the crop_stream release the
 * frame before the reid_stream finishes inference (Phase 2).
 *
 * Phase 1 (sync fallback): callers use release() immediately after
 * crop_into_pool, so the pool behaves as a simple free-list.  The
 * release_after path is exercised in Phase 2 when crop_stream and
 * reid_stream are separate.
 */
class SACCADE_TRACKING_API FrameBufferPool {
public:
    FrameBufferPool(int capacity, int frame_h, int frame_w, int channels = 3);
    ~FrameBufferPool();

    FrameBufferPool(const FrameBufferPool&) = delete;
    FrameBufferPool& operator=(const FrameBufferPool&) = delete;
    FrameBufferPool(FrameBufferPool&&) = delete;
    FrameBufferPool& operator=(FrameBufferPool&&) = delete;

    /// Acquire a frame buffer, or nullptr if all in use.
    float* acquire();

    /// Return a buffer to the pool immediately (sync path).
    void release(float* buf);

    /**
     * @brief Return a buffer to the pool after @p event has been reached.
     *
     * The buffer is NOT reused until cudaEventQuery(event) returns
     * cudaSuccess.  This is typically called from the main thread right
     * after launching the crop kernel on crop_stream and recording
     * crop_done_event — the frame can be recycled before ReID inference
     * finishes because the crop has already copied the pixels into a
     * CropPool slot.
     */
    void release_after(float* buf, cudaEvent_t event);

    /// Process pending release_after completions (call periodically).
    void poll_releases();

    int capacity() const { return capacity_; }
    int available() const { return static_cast<int>(free_list_.size()); }
    int frame_h() const { return frame_h_; }
    int frame_w() const { return frame_w_; }

private:
    float* d_buffer_ = nullptr;   // [capacity, channels, frame_h, frame_w] GPU
    int    capacity_  = 0;
    int    frame_h_   = 0;
    int    frame_w_   = 0;
    int    channels_  = 3;

    std::vector<float*> free_list_;
    struct PendingRelease {
        float* buf;
        cudaEvent_t event;
    };
    std::vector<PendingRelease> pending_;
};

} // namespace saccade
