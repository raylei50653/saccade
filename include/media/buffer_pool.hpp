#pragma once

#include "saccade/common.hpp"
#include <atomic>
#include <cuda_runtime.h>
#include <cstddef>
#include <mutex>

namespace saccade {

/**
 * @brief 緩衝區狀態列舉 (State Machine)
 *
 *   EMPTY       C++ 可寫入 (空閒,可被 acquire)
 *   WRITING     C++ 正在配置/搬運 (H2D 已排隊但未完成)
 *   READY       搬運指令已發出,等待 consumer sync 後讀取
 *   PROCESSING  consumer 正持有此 Buffer 進行計算
 */
enum class BufferStatus {
    EMPTY = 0,
    WRITING = 1,
    READY = 2,
    PROCESSING = 3
};

/**
 * @brief 工業級 GPU 緩衝池 — GStreamer 解碼 H2D 的無 race 後端。
 *
 * 設計目標 (對應 ADR-009):
 *   - POOL_SIZE 個獨立 cudaStream_t,允許影格 N 搬運與影格 N-1 計算並行
 *   - 以 CAS 做狀態轉移,消除 check-then-act race (S5/S7)
 *   - 固定 C-array 存放 device pointer,避免 vector clear() 造成 dangling (S2)
 *   - ensure_grow 只成長 EMPTY 槽,不釋放 READY/PROCESSING 槽 (S2/S6)
 *   - 解構子先 sync 全部 stream 再 free,保證 in-flight memcpy 完成 (S3/S5)
 *
 * Thread model:
 *   - acquire_empty_slot / ensure_grow: 串行化於 pool_mutex_ (streaming thread)
 *   - mark_processing / release:       lock-free CAS (consumer thread)
 *   - sync_slot:                        cudaStreamSynchronize 本身 thread-safe
 *   - device_ptr / stream:              固定 array,非 EMPTY 槽的指標穩定不變
 *
 * 不依賴 GStreamer — 可獨立單元測試 (見 tests/native/test_gst_buffer_pool.cpp)。
 */
class SACCADE_MEDIA_API BufferPool {
public:
    static constexpr size_t POOL_SIZE = 5;

    /**
     * @param initial_bytes 每槽初始顯存大小 (0 = 延遲到 acquire 時配置)
     */
    explicit BufferPool(size_t initial_bytes = 0);
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /**
     * @brief 原子取得一個 EMPTY 槽並轉為 WRITING。
     *
     * 若 min_bytes > 0 且該槽現有顯存不足,會在持有 pool_mutex_ 下
     * cudaFree + cudaMalloc 該槽 (只影響 EMPTY 槽,不動 READY/PROCESSING)。
     *
     * @return 槽索引,或 -1 表示池耗盡 (全在 READY/PROCESSING/WRITING) → drop frame
     */
    int acquire_empty_slot(size_t min_bytes = 0);

    /**
     * @brief 在該槽專屬 stream 上排隊 H2D,完成後將狀態設為 READY。
     *
     * 呼叫者必須持有該槽 (state == WRITING)。排隊後立即返回,
     * consumer 讀取 device_ptr 前必須先 sync_slot(idx) 等 H2D 完成 (修 S1)。
     *
     * @return true 成功;false 表示 idx 逾界或狀態非 WRITING
     */
    bool submit_h2d(int idx, const void* host_ptr, size_t bytes);

    /**
     * @brief 等待該槽 H2D 完成 (cudaStreamSynchronize)。
     *
     * consumer 在讀取 device_ptr 前必須呼叫,以避免讀到未完成 copy。
     */
    void sync_slot(int idx);

    /**
     * @brief READY → PROCESSING (CAS) — consumer 取得 buffer 所有權。
     *
     * 對應 Python FrameData.__enter__ / mark_processing。
     * @return true 成功;false 表示狀態非 READY (可能已被 release 或仍 WRITING)
     */
    bool mark_processing(int idx);

    /**
     * @brief 釋放槽位回 EMPTY (CAS) — consumer 歸還 buffer。
     *
     * 接受 PROCESSING 或 READY → EMPTY。對應 Python FrameData.__exit__ / release。
     * WRITING / EMPTY 不動 (避免 clobber 進行中的搬運)。
     */
    void release(int idx);

    /** @brief 該槽 device pointer (非 EMPTY 槽穩定可讀)。 */
    void* device_ptr(int idx) const;

    /** @brief 該槽專屬 CUDA stream (固定,解構前不變)。 */
    cudaStream_t stream(int idx) const;

    /** @brief 該槽目前配置的顯存大小 (bytes)。 */
    size_t slot_bytes(int idx) const;

    /** @brief 該槽目前狀態 (測試 / 觀測用)。 */
    BufferStatus state(int idx) const;

private:
    void alloc_slot_locked(int idx, size_t bytes);
    void free_slot_locked(int idx);

    void*          d_buffers_[POOL_SIZE]   = {};
    cudaStream_t   streams_[POOL_SIZE]     = {};
    std::atomic<int>    states_[POOL_SIZE] = {};
    size_t         slot_bytes_[POOL_SIZE]  = {};
    std::mutex     pool_mutex_;
    std::atomic<size_t> write_hint_{0};
};

} // namespace saccade
