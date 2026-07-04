#include "media/buffer_pool.hpp"
#include <iostream>

namespace saccade {

BufferPool::BufferPool(size_t initial_bytes) {
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        streams_[i] = nullptr;
        d_buffers_[i] = nullptr;
        slot_bytes_[i] = 0;
        states_[i].store(static_cast<int>(BufferStatus::EMPTY));
        if (initial_bytes > 0) {
            alloc_slot_locked(static_cast<int>(i), initial_bytes);
        }
    }
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        if (streams_[i] == nullptr) {
            checkCuda(cudaStreamCreate(&streams_[i]));
        }
    }
}

BufferPool::~BufferPool() {
    // 先 sync 全部 stream,確保所有 in-flight H2D 完成 (修 S3/S5),
    // 再 free 顯存與銷毀 stream。
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        if (streams_[i]) {
            cudaStreamSynchronize(streams_[i]);
        }
    }
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        free_slot_locked(static_cast<int>(i));
    }
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        if (streams_[i]) {
            cudaStreamDestroy(streams_[i]);
            streams_[i] = nullptr;
        }
    }
}

void BufferPool::alloc_slot_locked(int idx, size_t bytes) {
    if (d_buffers_[idx]) {
        cudaFree(d_buffers_[idx]);
        d_buffers_[idx] = nullptr;
        slot_bytes_[idx] = 0;
    }
    if (bytes > 0) {
        checkCuda(cudaMalloc(&d_buffers_[idx], bytes));
        slot_bytes_[idx] = bytes;
    }
}

void BufferPool::free_slot_locked(int idx) {
    if (d_buffers_[idx]) {
        cudaFree(d_buffers_[idx]);
        d_buffers_[idx] = nullptr;
        slot_bytes_[idx] = 0;
    }
}

int BufferPool::acquire_empty_slot(size_t min_bytes) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    const size_t hint = write_hint_.load(std::memory_order_relaxed);
    int target = -1;
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        const size_t check_idx = (hint + i) % POOL_SIZE;
        if (states_[check_idx].load(std::memory_order_acquire) ==
            static_cast<int>(BufferStatus::EMPTY)) {
            target = static_cast<int>(check_idx);
            break;
        }
    }
    if (target < 0) {
        return -1; // pool 耗盡 → drop frame
    }

    // 只成長不釋放 in-use slot:僅 EMPTY 槽可重配 (修 S2/S6)。
    if (min_bytes > 0 && slot_bytes_[target] < min_bytes) {
        alloc_slot_locked(target, min_bytes);
    }

    // EMPTY → WRITING (CAS,在 mutex 內仍用 CAS 以保證狀態不變式)。
    int expected = static_cast<int>(BufferStatus::EMPTY);
    if (!states_[target].compare_exchange_strong(
            expected, static_cast<int>(BufferStatus::WRITING),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        // 極罕見:被人搶走 → drop 此幀。
        return -1;
    }

    write_hint_.store((static_cast<size_t>(target) + 1) % POOL_SIZE,
                      std::memory_order_relaxed);
    return target;
}

bool BufferPool::submit_h2d(int idx, const void* host_ptr, size_t bytes) {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return false;
    }
    if (states_[idx].load(std::memory_order_acquire) !=
        static_cast<int>(BufferStatus::WRITING)) {
        return false;
    }
    if (bytes > slot_bytes_[idx]) {
        return false; // 呼叫者應先以 acquire_empty_slot(bytes) 確保空間
    }
    if (d_buffers_[idx] == nullptr || streams_[idx] == nullptr) {
        return false;
    }
    checkCuda(cudaMemcpyAsync(d_buffers_[idx], host_ptr, bytes,
                              cudaMemcpyHostToDevice, streams_[idx]));
    // 搬運已排隊 → READY (consumer 讀取前必須 sync_slot)。
    states_[idx].store(static_cast<int>(BufferStatus::READY),
                       std::memory_order_release);
    return true;
}

void BufferPool::sync_slot(int idx) {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return;
    }
    if (streams_[idx]) {
        cudaStreamSynchronize(streams_[idx]);
    }
}

bool BufferPool::mark_processing(int idx) {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return false;
    }
    int expected = static_cast<int>(BufferStatus::READY);
    return states_[idx].compare_exchange_strong(
        expected, static_cast<int>(BufferStatus::PROCESSING),
        std::memory_order_acq_rel, std::memory_order_acquire);
}

void BufferPool::release(int idx) {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return;
    }
    // CAS loop:接受 PROCESSING 或 READY → EMPTY (修 S4/S11)。
    // 不動 WRITING / EMPTY,避免 clobber 進行中的搬運或重複釋放。
    while (true) {
        int cur = states_[idx].load(std::memory_order_acquire);
        if (cur == static_cast<int>(BufferStatus::PROCESSING) ||
            cur == static_cast<int>(BufferStatus::READY)) {
            if (states_[idx].compare_exchange_strong(
                    cur, static_cast<int>(BufferStatus::EMPTY),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
            continue; // 被別人改了,重讀
        }
        return; // EMPTY 或 WRITING:不動
    }
}

void* BufferPool::device_ptr(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return nullptr;
    }
    return d_buffers_[idx];
}

cudaStream_t BufferPool::stream(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return nullptr;
    }
    return streams_[idx];
}

size_t BufferPool::slot_bytes(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return 0;
    }
    return slot_bytes_[idx];
}

BufferStatus BufferPool::state(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(POOL_SIZE)) {
        return BufferStatus::EMPTY;
    }
    return static_cast<BufferStatus>(states_[idx].load(std::memory_order_acquire));
}

} // namespace saccade
