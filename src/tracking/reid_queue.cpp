#include "tracking/reid_queue.hpp"

namespace saccade {

ReIDQueue::~ReIDQueue() {
    shutdown();
    // Drain remaining jobs to destroy their events.
    ReIDCropJob job;
    while (try_pop_batch(&job, 1) > 0) {
        if (job.crop_ready) {
            cudaEventSynchronize(job.crop_ready);
            cudaEventDestroy(job.crop_ready);
        }
    }
}

void ReIDQueue::push(ReIDCropJob&& job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(job));
    }
    cv_.notify_one();
}

int ReIDQueue::try_pop_batch(ReIDCropJob* out, int max_batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    int n = 0;
    while (n < max_batch && !queue_.empty()) {
        out[n] = std::move(queue_.front());
        queue_.pop_front();
        ++n;
    }
    return n;
}

int ReIDQueue::wait_pop_batch(ReIDCropJob* out, int max_batch, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty() && !shutdown_) {
        cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                     [this] { return !queue_.empty() || shutdown_; });
    }
    if (shutdown_ && queue_.empty()) return 0;
    int n = 0;
    while (n < max_batch && !queue_.empty()) {
        out[n] = std::move(queue_.front());
        queue_.pop_front();
        ++n;
    }
    return n;
}

int ReIDQueue::size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(queue_.size());
}

void ReIDQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

} // namespace saccade
