#include "tracking/crop_ring_store.hpp"

#include <algorithm>

namespace saccade {

CropRingStore::CropRingStore(int capacity, int depth, int crop_h, int crop_w)
    : pool_(std::max(capacity, std::max(depth, 1)), crop_h, crop_w),
      depth_(std::max(depth, 1)),
      slot_meta_(static_cast<size_t>(std::max(capacity, std::max(depth, 1)))) {
    write_ev_.resize(slot_meta_.size(), nullptr);
    for (auto& ev : write_ev_) {
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    }
}

CropRingStore::~CropRingStore() {
    for (auto& ev : write_ev_) {
        if (ev) cudaEventDestroy(ev);
    }
}

void CropRingStore::touch_lru_(int slot) {
    auto it = lru_pos_.find(slot);
    if (it != lru_pos_.end()) {
        lru_.erase(it->second);
    }
    lru_.push_back(slot);
    lru_pos_[slot] = std::prev(lru_.end());
}

void CropRingStore::drop_lru_(int slot) {
    auto it = lru_pos_.find(slot);
    if (it != lru_pos_.end()) {
        lru_.erase(it->second);
        lru_pos_.erase(it);
    }
}

void CropRingStore::free_slot_(int slot) {
    if (slot < 0 || slot >= static_cast<int>(slot_meta_.size())) return;
    slot_meta_[slot].live = false;
    drop_lru_(slot);
    pool_.release(slot, 1);
}

int CropRingStore::acquire_slot_() {
    if (pool_.available() <= 0) {
        // Evict the globally least-recently-stashed slot.
        if (lru_.empty()) return -1;
        int victim = lru_.front();
        const SlotMeta& m = slot_meta_[victim];
        auto uit = uid_slots_.find(m.uid);
        if (uit != uid_slots_.end()) {
            auto& dq = uit->second;
            dq.erase(std::remove(dq.begin(), dq.end(), victim), dq.end());
            if (dq.empty()) uid_slots_.erase(uit);
        }
        free_slot_(victim);
    }
    return pool_.acquire(1);
}

int CropRingStore::stash(uint64_t uid, int frame, const float* crop_dev,
                         bool clean, cudaStream_t stream) {
    if (crop_dev == nullptr) return -1;
    std::lock_guard<std::mutex> lock(mutex_);

    auto uit = uid_slots_.find(uid);
    if (uit != uid_slots_.end() &&
        static_cast<int>(uit->second.size()) >= depth_) {
        int old = uit->second.front();
        uit->second.pop_front();
        free_slot_(old);  // drop this uid's oldest crop
    }

    // acquire_slot_ can LRU-evict this same uid's last slot and erase its map
    // entry, so the deque is looked up again only after acquisition.
    int slot = acquire_slot_();
    if (slot < 0) {
        uit = uid_slots_.find(uid);
        if (uit != uid_slots_.end() && uit->second.empty()) uid_slots_.erase(uit);
        return -1;
    }

    float* dst = pool_.slot_ptr(slot);
    const size_t bytes =
        static_cast<size_t>(pool_.crop_elem_count()) * sizeof(float);
    // A recycled slot may carry a still-pending write from another stream;
    // order behind it, then publish this write for cross-stream readers.
    cudaStreamWaitEvent(stream, write_ev_[slot], 0);
    cudaMemcpyAsync(dst, crop_dev, bytes, cudaMemcpyDeviceToDevice, stream);
    cudaEventRecord(write_ev_[slot], stream);

    slot_meta_[slot] = SlotMeta{uid, frame, clean, true};
    uid_slots_[uid].push_back(slot);
    touch_lru_(slot);
    return slot;
}

int CropRingStore::count(uint64_t uid) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = uid_slots_.find(uid);
    return it == uid_slots_.end() ? 0 : static_cast<int>(it->second.size());
}

bool CropRingStore::has(uint64_t uid) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = uid_slots_.find(uid);
    return it != uid_slots_.end() && !it->second.empty();
}

int CropRingStore::gather(uint64_t uid, float* batch, cudaStream_t stream,
                          bool clean_only) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = uid_slots_.find(uid);
    if (it == uid_slots_.end() || batch == nullptr) return 0;
    const int elem = pool_.crop_elem_count();
    const size_t bytes = static_cast<size_t>(elem) * sizeof(float);
    int written = 0;
    for (int slot : it->second) {  // oldest → newest
        if (clean_only && !slot_meta_[slot].clean) continue;
        cudaStreamWaitEvent(stream, write_ev_[slot], 0);
        const float* src = pool_.slot_ptr(slot);
        cudaMemcpyAsync(batch + static_cast<size_t>(written) * elem, src, bytes,
                        cudaMemcpyDeviceToDevice, stream);
        ++written;
    }
    // Reads must complete before the lock is released: a subsequent stash may
    // recycle any of these slots and would overwrite an in-flight read (the
    // mutex orders host bookkeeping, not GPU execution).
    if (written > 0) cudaStreamSynchronize(stream);
    return written;
}

int CropRingStore::gather_many(const uint64_t* uids, int n_uids, float* batch,
                               int* out_counts, cudaStream_t stream,
                               bool clean_only) {
    if (uids == nullptr || batch == nullptr || n_uids <= 0) return 0;
    const int elem = pool_.crop_elem_count();
    int total = 0;
    for (int i = 0; i < n_uids; ++i) {
        int c = gather(uids[i], batch + static_cast<size_t>(total) * elem,
                       stream, clean_only);
        if (out_counts != nullptr) out_counts[i] = c;
        total += c;
    }
    return total;
}

bool CropRingStore::has_crop(uint64_t uid, int frame, bool clean_only) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = uid_slots_.find(uid);
    if (it == uid_slots_.end()) return false;
    for (int slot : it->second) {
        if (slot_meta_[slot].frame == frame) {
            if (!clean_only || slot_meta_[slot].clean) return true;
        }
    }
    return false;
}

const float* CropRingStore::find_crop_ptr(uint64_t uid, int frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = uid_slots_.find(uid);
    if (it == uid_slots_.end()) return nullptr;
    for (int slot : it->second) {
        if (slot_meta_[slot].frame == frame) {
            return pool_.slot_ptr(slot);
        }
    }
    return nullptr;
}

int CropRingStore::gather_framed(const uint64_t* uids, const int* frames,
                                 int n, float* batch, cudaStream_t stream,
                                 bool clean_only) {
    if (uids == nullptr || frames == nullptr || batch == nullptr || n <= 0)
        return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    const int elem = pool_.crop_elem_count();
    const size_t bytes = static_cast<size_t>(elem) * sizeof(float);
    int written = 0;
    for (int i = 0; i < n; ++i) {
        auto it = uid_slots_.find(uids[i]);
        if (it == uid_slots_.end()) continue;
        for (int slot : it->second) {
            if (slot_meta_[slot].frame == frames[i]) {
                if (clean_only && !slot_meta_[slot].clean) break;
                cudaStreamWaitEvent(stream, write_ev_[slot], 0);
                const float* src = pool_.slot_ptr(slot);
                cudaMemcpyAsync(batch + static_cast<size_t>(written) * elem,
                                src, bytes, cudaMemcpyDeviceToDevice, stream);
                ++written;
                break;
            }
        }
    }
    // See gather(): reads must land before a later stash can recycle slots.
    if (written > 0) cudaStreamSynchronize(stream);
    return written;
}

void CropRingStore::evict(uint64_t uid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = uid_slots_.find(uid);
    if (it == uid_slots_.end()) return;
    for (int slot : it->second) {
        free_slot_(slot);
    }
    uid_slots_.erase(it);
}

}  // namespace saccade
