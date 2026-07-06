#pragma once

#include "saccade/common.hpp"
#include "tracking/crop_pool.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace saccade {

/**
 * @brief CropRingStore — per-``track_uid`` ring of recent ReID crops.
 *
 * Backs borderline re-query in the online handover: each never-reused
 * ``track_uid`` keeps its most recent ``depth`` raw crops so a flippable
 * decision can re-extract a dense recent-tail bank for the top candidates on
 * demand, instead of re-reading crops from disk.
 *
 * GPU pixels live in an owned :class:`CropPool` (one contiguous
 * ``[capacity, 3, crop_h, crop_w]`` buffer); the ``uid -> slots`` map and the
 * global LRU are host-side bookkeeping. When the pool is full the globally
 * least-recently-stashed slot is recycled, so total GPU memory is bounded
 * independent of how many tracks a sequence spawns.
 *
 * The class depends on neither TensorRT nor the crop kernel — callers hand it
 * crops that are already device-resident ``[3, crop_h, crop_w]`` tensors — so
 * it is unit-testable with raw device buffers. :class:`PerceptionPipeline`
 * composes it with the cropper (stash side) and the ReID extractor (re-query
 * side) to satisfy the :class:`ReidCropStore` interface.
 */
class SACCADE_TRACKING_API CropRingStore {
public:
    CropRingStore(int capacity, int depth, int crop_h, int crop_w);
    ~CropRingStore();

    CropRingStore(const CropRingStore&) = delete;
    CropRingStore& operator=(const CropRingStore&) = delete;
    CropRingStore(CropRingStore&&) = delete;
    CropRingStore& operator=(CropRingStore&&) = delete;

    /**
     * @brief Stash one ``[3, crop_h, crop_w]`` device crop for @p uid.
     *
     * Copies @p crop_dev into a ring slot on @p stream. Enforces the per-uid
     * depth cap (dropping the uid's oldest crop) and, when the pool is full,
     * recycles the globally least-recently-stashed slot. Returns the slot
     * index, or -1 on failure.
     */
    int stash(uint64_t uid, int frame, const float* crop_dev, bool clean,
              cudaStream_t stream);

    /// Number of crops currently held for @p uid.
    int count(uint64_t uid) const;

    /**
     * @brief Gather @p uid's crops (oldest→newest) contiguously into @p batch.
     *
     * @param batch      Device buffer sized >= ``depth() * crop_elem_count()``.
     * @param clean_only When true, only crops flagged clean are emitted.
     * @return Number of crop rows written (0 if the uid has none).
     */
    int gather(uint64_t uid, float* batch, cudaStream_t stream,
               bool clean_only = false);

    /**
     * @brief Gather several uids contiguously into @p batch.
     *
     * @param out_counts ``[n_uids]`` host — rows written for ``uids[i]``.
     * @param batch      Device buffer sized >=
     *                   ``depth() * n_uids * crop_elem_count()``.
     * @return Total crop rows written across all uids.
     */
    int gather_many(const uint64_t* uids, int n_uids, float* batch,
                    int* out_counts, cudaStream_t stream,
                    bool clean_only = false);

    /// Drop all of @p uid's stashed crops.
    void evict(uint64_t uid);

    bool has(uint64_t uid) const;

    /// Device pointer to the crop for @p uid at @p frame, or nullptr.
    const float* find_crop_ptr(uint64_t uid, int frame) const;

    /// True if a crop exists at (uid, frame), optionally gated by clean flag.
    bool has_crop(uint64_t uid, int frame, bool clean_only = false) const;

    /// Gather one crop per (uid, frame) pair into contiguous @p batch,
    /// optionally gated by the clean flag (mirrors ``has_crop``).
    /// Returns total rows written.
    int gather_framed(const uint64_t* uids, const int* frames, int n,
                      float* batch, cudaStream_t stream,
                      bool clean_only = false);

    int depth() const { return depth_; }
    int crop_h() const { return pool_.crop_h(); }
    int crop_w() const { return pool_.crop_w(); }
    int crop_elem_count() const { return pool_.crop_elem_count(); }
    int capacity() const { return pool_.capacity(); }
    int in_use() const { return pool_.in_use(); }

private:
    struct SlotMeta {
        uint64_t uid = 0;
        int frame = -1;
        bool clean = false;
        bool live = false;
    };

    int acquire_slot_();          // a free slot, evicting the global LRU if full
    void touch_lru_(int slot);    // mark slot most-recently used
    void drop_lru_(int slot);     // remove slot from the LRU list
    void free_slot_(int slot);    // clear meta + return slot to the pool

    CropPool pool_;
    int depth_;
    // Per-slot completion event of the last stash write. Stash and gather run
    // on different streams (tracker vs bg-handover requery); the mutex orders
    // host bookkeeping only, so pixel access is ordered via these events:
    // stash waits on the slot's previous write before overwriting a recycled
    // slot and records after its copy; gathers wait on it before reading and
    // synchronize their stream under the lock so a later stash cannot recycle
    // a slot whose read is still in flight.
    std::vector<cudaEvent_t> write_ev_;
    std::unordered_map<uint64_t, std::deque<int>> uid_slots_;
    std::vector<SlotMeta> slot_meta_;
    std::list<int> lru_;  // front = least recently stashed
    std::unordered_map<int, std::list<int>::iterator> lru_pos_;
    mutable std::mutex mutex_;
};

}  // namespace saccade
