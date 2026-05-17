#pragma once
#include "saccade/common.hpp"
#include "tracking/seq_runner.hpp"
#include "tracking/pipeline.hpp"
#include "perception/trt_engine.hpp"
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace saccade {

/**
 * Thread pool for concurrent multi-sequence evaluation.
 *
 * Creates n_threads SequenceRunner instances (each with its own TRT context,
 * stream, tracker, and pipeline). Sequences are distributed via a work-stealing
 * queue so the pool handles any number of sequences (including more than n_threads).
 *
 * Usage:
 *   EvaluatorPool pool(engine, pipe_cfg, n_threads=4);
 *   auto results = pool.run_sequences({cfg0, cfg1, cfg2, cfg3, ...});
 *   // results: map<seq_name, vector<FrameResult>>
 */
class SACCADE_TRACKING_API EvaluatorPool {
public:
    EvaluatorPool(TRTEngine*                        detect_engine,
                  const PerceptionPipeline::Config& pipe_cfg,
                  int                               n_threads   = 4,
                  int                               max_dets    = 2048,
                  int                               max_tracks  = 256,
                  int                               device_id   = 0);
    ~EvaluatorPool() = default;

    EvaluatorPool(const EvaluatorPool&) = delete;
    EvaluatorPool& operator=(const EvaluatorPool&) = delete;

    /**
     * Run all sequences in parallel. Blocks until all are done.
     * @return map from sequence name to per-frame results.
     */
    std::map<std::string, std::vector<FrameResult>>
        run_sequences(const std::vector<SequenceConfig>& seqs);

private:
    // Worker function: pulls sequences from queue and runs them
    void worker_fn(int worker_idx,
                   std::queue<const SequenceConfig*>& work_queue,
                   std::mutex& queue_mtx,
                   std::map<std::string, std::vector<FrameResult>>& results,
                   std::mutex& results_mtx);

    TRTEngine*                  detect_engine_;
    PerceptionPipeline::Config  pipe_cfg_;
    int                         n_threads_;
    int                         max_dets_;
    int                         max_tracks_;
    int                         device_id_;
};

} // namespace saccade
