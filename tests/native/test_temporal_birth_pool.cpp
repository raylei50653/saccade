// Native tests for TemporalBirthPool — covers the OOB clamp fix and basic
// ring-window boost behaviour. Compiled into saccade_tracking (the pool's
// .cu is now in the static lib) so no extra sources are needed on the link line.
#include "tracking/temporal_birth_pool.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace saccade;

void fail(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void expect_near(float actual, float expected, float tolerance, const std::string& label) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream oss;
        oss << label << ": expected " << expected << " got " << actual
            << " (tol " << tolerance << ")";
        fail(oss.str());
    }
}

// Round-trip a device float array to host.
std::vector<float> d2h(const float* d, int n, cudaStream_t stream) {
    std::vector<float> h(n);
    cudaMemcpyAsync(h.data(), d, (size_t)n * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

} // namespace

// ── config sanity ────────────────────────────────────────────────────────
static void test_config_defaults() {
    TemporalBirthPool::Config cfg;
    expect_true(cfg.frames == 3, "default frames");
    expect_true(cfg.iou_thresh == 0.50f, "default iou_thresh");
    expect_true(cfg.boost == 0.25f, "default boost");
    expect_true(cfg.capacity == 2048, "default capacity");
    expect_true(cfg.high_thresh == 0.80f, "default high_thresh");
}

// ── boost disabled: scores unchanged for sub-threshold detections ────────
static void test_boost_disabled() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    TemporalBirthPool::Config cfg;
    cfg.boost = 0.0f;            // no boost
    cfg.frames = 1;              // minimal window
    cfg.capacity = 16;
    TemporalBirthPool pool(cfg);

    const int N = 4;
    std::vector<float> boxes(N * 4, 0.0f);
    std::vector<float> scores = {0.30f, 0.35f, 0.40f, 0.10f};
    float* d_boxes;   cudaMalloc(&d_boxes, N * 4 * sizeof(float));
    float* d_scores;  cudaMalloc(&d_scores, N * sizeof(float));
    cudaMemcpyAsync(d_boxes, boxes.data(), N * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scores, scores.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    pool.update_and_apply(d_boxes, d_scores, N, stream);
    auto out = d2h(d_scores, N, stream);

    // boost == 0 → kernel skipped, scores untouched
    for (int i = 0; i < N; ++i) {
        expect_near(out[i], scores[i], 1e-6f, "boost_disabled score unchanged");
    }

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaStreamDestroy(stream);
}

// ── ring window requires `frames` frames before boosting ─────────────────
static void test_window_not_filled() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    TemporalBirthPool::Config cfg;
    cfg.frames = 3;              // need 3 stored frames before boost
    cfg.boost = 0.25f;
    cfg.iou_thresh = 0.10f;      // loose so any overlap counts
    cfg.min_score = 0.05f;
    cfg.new_track_thresh = 0.70f;
    cfg.high_thresh = 0.80f;
    cfg.capacity = 16;
    TemporalBirthPool pool(cfg);

    const int N = 1;
    float boxes[4] = {0.0f, 0.0f, 10.0f, 10.0f};
    float score = 0.30f;
    float* d_boxes;  cudaMalloc(&d_boxes, N * 4 * sizeof(float));
    float* d_scores; cudaMalloc(&d_scores, N * sizeof(float));

    // Frame 0: store only (filled < required_window → no boost)
    cudaMemcpyAsync(d_boxes, boxes, N * 4 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scores, &score, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);
    auto out = d2h(d_scores, N, stream);
    expect_near(out[0], 0.30f, 1e-6f, "frame 0 not boosted (window not filled)");

    // Frame 1: still not enough (filled == 1, required 3)
    cudaMemcpyAsync(d_scores, &score, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);
    out = d2h(d_scores, N, stream);
    expect_near(out[0], 0.30f, 1e-6f, "frame 1 not boosted (window not filled)");

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaStreamDestroy(stream);
}

// ── OOB clamp: overfill the ring slot past capacity and verify the next
//    boost read does not crash / corrupt (the clamp prevents OOB reads). ───
static void test_oob_clamp_no_crash() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    TemporalBirthPool::Config cfg;
    cfg.frames = 2;
    cfg.boost = 0.25f;
    cfg.iou_thresh = 0.10f;
    cfg.min_score = 0.05f;
    cfg.new_track_thresh = 0.70f;
    cfg.high_thresh = 0.80f;
    cfg.capacity = 4;            // tiny capacity to force overflow easily
    TemporalBirthPool pool(cfg);

    // Feed a frame with many sub-threshold detections so store_subthreshold
    // bumps slot_count well past `capacity` (writer drops out-of-range stores).
    const int N = 32;            // >> capacity(4)
    std::vector<float> boxes(N * 4, 0.0f);
    for (int i = 0; i < N; ++i) {
        boxes[i * 4 + 0] = 0.0f;
        boxes[i * 4 + 1] = 0.0f;
        boxes[i * 4 + 2] = 10.0f;
        boxes[i * 4 + 3] = 10.0f;
    }
    std::vector<float> scores(N, 0.30f);  // all sub-threshold → all stored
    float* d_boxes;  cudaMalloc(&d_boxes, N * 4 * sizeof(float));
    float* d_scores; cudaMalloc(&d_scores, N * sizeof(float));
    cudaMemcpyAsync(d_boxes, boxes.data(), N * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Frame 0: overfill the ring.
    cudaMemcpyAsync(d_scores, scores.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);

    // Frame 1: now the ring has a slot whose stored count exceeds capacity.
    // The reader (apply_temporal_birth_boost_kernel) must clamp to capacity
    // and read only the valid prefix — no OOB, no crash. We don't assert the
    // exact boost here (filling still < required_window=2 so no boost yet),
    // only that the call completes without fault.
    cudaMemcpyAsync(d_scores, scores.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);
    auto out = d2h(d_scores, N, stream);
    // No boost yet (window just filled at frame 1); verify no corruption.
    for (int i = 0; i < N; ++i) {
        expect_true(out[i] >= 0.0f && out[i] <= 1.0f, "score in valid range after overfill");
    }

    // Frame 2: window now filled (>=2). Boost kernel reads the overfull slot;
    // the clamp must keep it safe. Boxes overlap perfectly → boost applies.
    cudaMemcpyAsync(d_scores, scores.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);
    out = d2h(d_scores, N, stream);
    // At least some sub-threshold scores should have been boosted upward.
    bool any_boosted = false;
    for (int i = 0; i < N; ++i) {
        if (out[i] > 0.30f + 1e-5f) { any_boosted = true; break; }
    }
    expect_true(any_boosted, "boost applied after window filled despite overfull slot");

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaStreamDestroy(stream);
}

// ── reset clears ring state ──────────────────────────────────────────────
static void test_reset_clears_window() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    TemporalBirthPool::Config cfg;
    // frames=2 → required_window=1: boost needs at least 1 stored frame.
    // This lets us observe that reset clears the window (filled drops to 0).
    cfg.frames = 2;
    cfg.boost = 0.25f;
    cfg.iou_thresh = 0.10f;
    cfg.min_score = 0.05f;
    cfg.new_track_thresh = 0.70f;
    cfg.high_thresh = 0.80f;
    cfg.capacity = 16;
    TemporalBirthPool pool(cfg);

    const int N = 2;
    std::vector<float> boxes(N * 4, 0.0f);
    for (int i = 0; i < N; ++i) {
        boxes[i * 4 + 2] = 10.0f;
        boxes[i * 4 + 3] = 10.0f;
    }
    std::vector<float> scores = {0.30f, 0.35f};
    float* d_boxes;  cudaMalloc(&d_boxes, N * 4 * sizeof(float));
    float* d_scores; cudaMalloc(&d_scores, N * sizeof(float));
    cudaMemcpyAsync(d_boxes, boxes.data(), N * 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Frame 0: store only (filled=0 < required_window=1 → no boost)
    cudaMemcpyAsync(d_scores, scores.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);
    auto out = d2h(d_scores, N, stream);
    for (int i = 0; i < N; ++i) {
        expect_near(out[i], scores[i], 1e-6f, "frame 0 no boost (window empty)");
    }

    // Frame 1 without reset: window now filled → boost applies
    cudaMemcpyAsync(d_scores, scores.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);
    out = d2h(d_scores, N, stream);
    bool any_boosted = false;
    for (int i = 0; i < N; ++i) {
        if (out[i] > scores[i] + 1e-5f) { any_boosted = true; break; }
    }
    expect_true(any_boosted, "frame 1 boosted (window filled)");

    // Reset → window cleared.  Next frame should NOT boost (filled=0 < 1).
    pool.reset();
    cudaMemcpyAsync(d_scores, scores.data(), N * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    pool.update_and_apply(d_boxes, d_scores, N, stream);
    out = d2h(d_scores, N, stream);
    for (int i = 0; i < N; ++i) {
        expect_near(out[i], scores[i], 1e-6f, "no boost right after reset");
    }

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaStreamDestroy(stream);
}

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    test_config_defaults();
    test_boost_disabled();
    test_window_not_filled();
    test_oob_clamp_no_crash();
    test_reset_clears_window();
    std::cout << "temporal_birth_pool tests passed" << std::endl;
    return 0;
}
