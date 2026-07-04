#pragma once

#include "saccade/common.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>

namespace saccade {

/// Reason codes for why a crop was triggered (mirrors DynamicReIDController logic).
enum ReIDReason : int {
    REID_NONE      = 0,
    REID_NEWBORN   = 1,
    REID_HANDOVER  = 2,
    REID_RECOVERY  = 3,
    REID_CONFIRM   = 4,
    REID_CLEAN     = 5,
    REID_ARCHIVE   = 6,
};

/**
 * @brief ReIDCropJob — metadata carried with each crop through the async ReID pipeline.
 *
 * Created on the main thread right after crop_into_pool_async().  The crop
 * tensor lives in the CropPool at @p crop_slot; the @p crop_ready event is
 * recorded on crop_stream and must be waited on before reading the crop from
 * reid_stream.
 *
 * The (track_uid, generation) pair is checked when the ReID result returns —
 * if the tracker has advanced the generation or retired the uid, the result
 * is discarded (prevents "舊 crop 修新 ID").
 */
struct ReIDCropJob {
    int       crop_slot    = -1;   ///< Starting slot in CropPool
    int       n_crops      = 0;    ///< Number of contiguous crops in this job
    int       frame_idx    = 0;
    uint64_t  track_uid    = 0;    ///< Immutable track identity (Phase 0)
    int       generation   = 0;    ///< Generation at crop time (Phase 0)
    float     x1 = 0, y1 = 0, x2 = 0, y2 = 0;  ///< Representative bbox
    float     det_score    = 0.0f;
    float     quality      = 0.0f;
    int       reason       = REID_NONE;
    cudaEvent_t crop_ready = nullptr;  ///< Signaled when crop kernel completes on crop_stream

    ReIDCropJob() = default;
    ReIDCropJob(int slot, int n, int frame, uint64_t uid, int gen,
                float bx1, float by1, float bx2, float by2,
                float score, float qual, int why, cudaEvent_t evt)
        : crop_slot(slot), n_crops(n), frame_idx(frame),
          track_uid(uid), generation(gen),
          x1(bx1), y1(by1), x2(bx2), y2(by2),
          det_score(score), quality(qual), reason(why), crop_ready(evt) {}
};

/**
 * @brief ReIDResult — embedding + metadata returned from the ReID worker.
 *
 * The caller checks (track_uid, generation) against the current tracker state
 * before applying the embedding to avoid stale-result contamination.
 */
struct ReIDResult {
    int       frame_idx    = 0;
    uint64_t  track_uid    = 0;
    int       generation   = 0;
    int       reason       = REID_NONE;
    int       embed_offset = 0;    ///< Row index into the batch embedding output
    int       n_crops      = 0;
};

/**
 * @brief ReIDQueue — thread-safe MPSC queue for ReIDCropJob.
 *
 * Main thread pushes jobs (after crop_into_pool_async).  ReID worker thread
 * pops jobs in FIFO order for batched inference.  Blocking pop with timeout
 * supports batching up to N_opt jobs before triggering inference.
 */
class SACCADE_TRACKING_API ReIDQueue {
public:
    ReIDQueue() = default;
    ~ReIDQueue();

    ReIDQueue(const ReIDQueue&) = delete;
    ReIDQueue& operator=(const ReIDQueue&) = delete;

    /// Push a job onto the queue (main thread).
    void push(ReIDCropJob&& job);

    /**
     * @brief Pop up to @p max_batch jobs without blocking.
     * @return Number of jobs popped (0 if queue is empty).
     */
    int try_pop_batch(ReIDCropJob* out, int max_batch);

    /**
     * @brief Pop up to @p max_batch jobs, waiting up to @p timeout_ms for
     *        at least one job to arrive.
     * @return Number of jobs popped (0 on timeout).
     */
    int wait_pop_batch(ReIDCropJob* out, int max_batch, int timeout_ms);

    /// Number of jobs currently in the queue.
    int size();

    /// Signal shutdown (unblocks wait_pop_batch).
    void shutdown();

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::deque<ReIDCropJob> queue_;
    std::atomic<bool>       shutdown_{false};
};

} // namespace saccade
