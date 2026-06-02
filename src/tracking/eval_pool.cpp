#include "tracking/eval_pool.hpp"
#include <stdexcept>

namespace saccade {

EvaluatorPool::EvaluatorPool(BaseDetector*                      detect_detector,
                              const PerceptionPipeline::Config& pipe_cfg,
                              int                               n_threads,
                              int                               max_dets,
                              int                               max_tracks,
                              int                               device_id)
    : detect_detector_(detect_detector)
    , pipe_cfg_(pipe_cfg)
    , n_threads_(std::max(1, n_threads))
    , max_dets_(max_dets)
    , max_tracks_(max_tracks)
    , device_id_(device_id)
{}

void EvaluatorPool::worker_fn(
    int                                              worker_idx,
    std::queue<const SequenceConfig*>&               work_queue,
    std::mutex&                                      queue_mtx,
    std::map<std::string, std::vector<FrameResult>>& results,
    std::mutex&                                      results_mtx)
{
    // Each worker creates its own SequenceRunner (per-thread TRT context + stream)
    std::unique_ptr<SequenceRunner> runner_ptr;
    try {
        runner_ptr = std::make_unique<SequenceRunner>(
            detect_detector_, pipe_cfg_, max_dets_, max_tracks_, device_id_);
    } catch (const std::exception& e) {
        fprintf(stderr, "[EvaluatorPool] Worker %d: failed to create SequenceRunner: %s\n",
                worker_idx, e.what());
        return;
    }
    SequenceRunner& runner = *runner_ptr;

    while (true) {
        const SequenceConfig* cfg = nullptr;
        {
            std::lock_guard<std::mutex> lock(queue_mtx);
            if (work_queue.empty()) break;
            cfg = work_queue.front();
            work_queue.pop();
        }

        try {
            auto frame_results = runner.run(*cfg);
            std::lock_guard<std::mutex> lock(results_mtx);
            results[cfg->name] = std::move(frame_results);
        } catch (const std::exception& e) {
            // Log error and store empty result so the sequence appears in the output
            fprintf(stderr, "[EvaluatorPool] ERROR in sequence %s: %s\n",
                    cfg->name.c_str(), e.what());
            std::lock_guard<std::mutex> lock(results_mtx);
            results[cfg->name] = {};  // empty FrameResults
        }
    }
}

std::map<std::string, std::vector<FrameResult>>
EvaluatorPool::run_sequences(const std::vector<SequenceConfig>& seqs) {
    // Build work queue
    std::queue<const SequenceConfig*> work_queue;
    for (const auto& s : seqs)
        work_queue.push(&s);

    std::mutex queue_mtx;
    std::map<std::string, std::vector<FrameResult>> results;
    std::mutex results_mtx;

    // Launch worker threads
    int n_workers = std::min(n_threads_, static_cast<int>(seqs.size()));
    std::vector<std::thread> threads;
    threads.reserve(n_workers);

    for (int i = 0; i < n_workers; ++i) {
        threads.emplace_back(&EvaluatorPool::worker_fn, this,
                             i, std::ref(work_queue), std::ref(queue_mtx),
                             std::ref(results), std::ref(results_mtx));
    }

    for (auto& t : threads)
        t.join();

    return results;
}

} // namespace saccade
