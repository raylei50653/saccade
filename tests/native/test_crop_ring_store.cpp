#include "tracking/crop_ring_store.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using saccade::CropRingStore;

namespace {

void fail(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

void expect_true(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void check_cuda(cudaError_t status, const std::string& label) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << label << ": " << cudaGetErrorString(status);
        fail(oss.str());
    }
}

// A device crop of `elem` floats all equal to `value`.
float* make_crop(int elem, float value) {
    float* d = nullptr;
    check_cuda(cudaMalloc(&d, elem * sizeof(float)), "malloc crop");
    std::vector<float> h(elem, value);
    check_cuda(cudaMemcpy(d, h.data(), elem * sizeof(float),
                          cudaMemcpyHostToDevice),
               "upload crop");
    return d;
}

// First element of gathered row `row` in a device batch buffer.
float row_value(const float* d_batch, int row, int elem) {
    float v = 0.0f;
    check_cuda(cudaMemcpy(&v, d_batch + static_cast<size_t>(row) * elem,
                          sizeof(float), cudaMemcpyDeviceToHost),
               "read row");
    return v;
}

void test_stash_gather_roundtrip() {
    const int ch = 2, cw = 2, elem = 3 * ch * cw;
    CropRingStore ring(/*capacity=*/8, /*depth=*/4, ch, cw);
    expect_true(ring.depth() == 4, "depth");
    expect_true(ring.crop_elem_count() == elem, "elem count");
    expect_true(ring.in_use() == 0, "starts empty");

    for (int f = 0; f < 3; ++f) {
        float* c = make_crop(elem, static_cast<float>(f));
        ring.stash(7, f, c, /*clean=*/true, nullptr);
        check_cuda(cudaFree(c), "free crop");
    }
    check_cuda(cudaStreamSynchronize(nullptr), "sync after stash");
    expect_true(ring.count(7) == 3, "count after 3 stashes");
    expect_true(ring.in_use() == 3, "in_use after 3 stashes");

    float* batch = nullptr;
    check_cuda(cudaMalloc(&batch, ring.depth() * elem * sizeof(float)),
               "malloc batch");
    int n = ring.gather(7, batch, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync after gather");
    expect_true(n == 3, "gather returns 3 rows");
    // oldest → newest ordering
    expect_true(std::fabs(row_value(batch, 0, elem) - 0.0f) < 1e-6f, "row0 oldest");
    expect_true(std::fabs(row_value(batch, 2, elem) - 2.0f) < 1e-6f, "row2 newest");
    check_cuda(cudaFree(batch), "free batch");

    std::cout << "stash_gather_roundtrip passed" << std::endl;
}

void test_per_uid_depth_caps_tail() {
    const int ch = 2, cw = 2, elem = 3 * ch * cw;
    CropRingStore ring(/*capacity=*/16, /*depth=*/3, ch, cw);
    for (int f = 0; f < 6; ++f) {
        float* c = make_crop(elem, static_cast<float>(f));
        ring.stash(1, f, c, true, nullptr);
        check_cuda(cudaFree(c), "free crop");
    }
    check_cuda(cudaStreamSynchronize(nullptr), "sync");
    expect_true(ring.count(1) == 3, "depth cap holds only 3");
    expect_true(ring.in_use() == 3, "only 3 slots in use");

    float* batch = nullptr;
    check_cuda(cudaMalloc(&batch, ring.depth() * elem * sizeof(float)), "malloc");
    int n = ring.gather(1, batch, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync gather");
    expect_true(n == 3, "gather 3");
    // most recent `depth` survive: frames 3,4,5
    expect_true(std::fabs(row_value(batch, 0, elem) - 3.0f) < 1e-6f, "oldest kept = 3");
    expect_true(std::fabs(row_value(batch, 2, elem) - 5.0f) < 1e-6f, "newest = 5");
    check_cuda(cudaFree(batch), "free batch");

    std::cout << "per_uid_depth_caps_tail passed" << std::endl;
}

void test_global_lru_eviction_bounds_memory() {
    const int ch = 2, cw = 2, elem = 3 * ch * cw;
    CropRingStore ring(/*capacity=*/3, /*depth=*/3, ch, cw);
    for (int u = 1; u <= 3; ++u) {
        float* c = make_crop(elem, static_cast<float>(u));
        ring.stash(u, 0, c, true, nullptr);
        check_cuda(cudaFree(c), "free");
    }
    check_cuda(cudaStreamSynchronize(nullptr), "sync");
    expect_true(ring.in_use() == 3, "pool full");

    // capacity full; a new uid evicts the least-recently-stashed (uid 1).
    float* c4 = make_crop(elem, 4.0f);
    ring.stash(4, 1, c4, true, nullptr);
    check_cuda(cudaFree(c4), "free");
    check_cuda(cudaStreamSynchronize(nullptr), "sync");
    expect_true(ring.count(1) == 0, "uid 1 evicted by global LRU");
    expect_true(ring.has(4), "uid 4 stashed");
    expect_true(ring.in_use() == 3, "memory stays bounded");

    std::cout << "global_lru_eviction_bounds_memory passed" << std::endl;
}

void test_self_lru_eviction_on_stash() {
    const int ch = 2, cw = 2, elem = 3 * ch * cw;
    CropRingStore ring(/*capacity=*/3, /*depth=*/3, ch, cw);
    float* a = make_crop(elem, 1.0f);
    float* b = make_crop(elem, 2.0f);
    ring.stash(1, 0, a, true, nullptr);
    ring.stash(2, 0, b, true, nullptr);
    ring.stash(3, 0, b, true, nullptr);  // pool now full
    // uid 1 is below depth and its only slot is the global LRU victim of its
    // own stash: acquire must not leave the uid's deque dangling (regression:
    // heap-use-after-free when the eviction erased the map entry in use).
    ring.stash(1, 5, a, true, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync");
    check_cuda(cudaFree(a), "free a");
    check_cuda(cudaFree(b), "free b");
    expect_true(ring.count(1) == 1, "uid 1 keeps exactly the new crop");
    expect_true(ring.has_crop(1, 5), "new crop present");
    expect_true(!ring.has_crop(1, 0), "old crop evicted");
    expect_true(ring.in_use() == 3, "memory stays bounded");

    std::cout << "self_lru_eviction_on_stash passed" << std::endl;
}

void test_evict_frees_slots() {
    const int ch = 2, cw = 2, elem = 3 * ch * cw;
    CropRingStore ring(/*capacity=*/4, /*depth=*/4, ch, cw);
    float* a = make_crop(elem, 1.0f);
    float* b = make_crop(elem, 2.0f);
    ring.stash(5, 0, a, true, nullptr);
    ring.stash(5, 1, b, true, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync");
    check_cuda(cudaFree(a), "free a");
    check_cuda(cudaFree(b), "free b");
    expect_true(ring.in_use() == 2, "2 in use");

    ring.evict(5);
    expect_true(ring.count(5) == 0, "evicted");
    expect_true(ring.in_use() == 0, "slots freed");

    // slots are reusable after eviction
    float* c = make_crop(elem, 9.0f);
    ring.stash(6, 0, c, true, nullptr);
    check_cuda(cudaFree(c), "free c");
    expect_true(ring.has(6), "reused after evict");

    std::cout << "evict_frees_slots passed" << std::endl;
}

void test_clean_only_filter() {
    const int ch = 2, cw = 2, elem = 3 * ch * cw;
    CropRingStore ring(/*capacity=*/8, /*depth=*/4, ch, cw);
    float* a = make_crop(elem, 1.0f);
    float* b = make_crop(elem, 2.0f);
    float* c = make_crop(elem, 3.0f);
    ring.stash(1, 0, a, /*clean=*/true, nullptr);
    ring.stash(1, 1, b, /*clean=*/false, nullptr);
    ring.stash(1, 2, c, /*clean=*/true, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync");
    check_cuda(cudaFree(a), "free");
    check_cuda(cudaFree(b), "free");
    check_cuda(cudaFree(c), "free");

    float* batch = nullptr;
    check_cuda(cudaMalloc(&batch, ring.depth() * elem * sizeof(float)), "malloc");
    int n = ring.gather(1, batch, nullptr, /*clean_only=*/true);
    check_cuda(cudaStreamSynchronize(nullptr), "sync gather");
    expect_true(n == 2, "only 2 clean crops gathered");
    expect_true(std::fabs(row_value(batch, 0, elem) - 1.0f) < 1e-6f, "clean row0=1");
    expect_true(std::fabs(row_value(batch, 1, elem) - 3.0f) < 1e-6f, "clean row1=3");
    check_cuda(cudaFree(batch), "free batch");

    std::cout << "clean_only_filter passed" << std::endl;
}

void test_gather_many_contiguous() {
    const int ch = 2, cw = 2, elem = 3 * ch * cw;
    CropRingStore ring(/*capacity=*/16, /*depth=*/4, ch, cw);
    float* a = make_crop(elem, 1.0f);
    float* b = make_crop(elem, 2.0f);
    ring.stash(10, 0, a, true, nullptr);
    ring.stash(10, 1, a, true, nullptr);
    ring.stash(20, 0, b, true, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync");
    check_cuda(cudaFree(a), "free");
    check_cuda(cudaFree(b), "free");

    uint64_t uids[2] = {10, 20};
    int counts[2] = {0, 0};
    float* batch = nullptr;
    check_cuda(cudaMalloc(&batch, ring.depth() * 2 * elem * sizeof(float)),
               "malloc");
    int total = ring.gather_many(uids, 2, batch, counts, nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync gather_many");
    expect_true(total == 3, "total 3 rows");
    expect_true(counts[0] == 2 && counts[1] == 1, "per-uid counts");
    expect_true(std::fabs(row_value(batch, 0, elem) - 1.0f) < 1e-6f, "uid10 row0");
    expect_true(std::fabs(row_value(batch, 2, elem) - 2.0f) < 1e-6f, "uid20 row0");
    check_cuda(cudaFree(batch), "free batch");

    std::cout << "gather_many_contiguous passed" << std::endl;
}

}  // namespace

int main() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fail("No CUDA device available for crop_ring_store_test");
    }

    test_stash_gather_roundtrip();
    test_per_uid_depth_caps_tail();
    test_global_lru_eviction_bounds_memory();
    test_self_lru_eviction_on_stash();
    test_evict_frees_slots();
    test_clean_only_filter();
    test_gather_many_contiguous();

    std::cout << "crop ring store tests passed" << std::endl;
    return 0;
}
