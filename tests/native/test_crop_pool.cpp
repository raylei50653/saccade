#include "tracking/crop_pool.hpp"
#include "tracking/frame_buffer.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using saccade::CropPool;
using saccade::FrameBufferPool;

namespace {

void fail(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void check_cuda(cudaError_t status, const std::string& label) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << label << ": " << cudaGetErrorString(status);
        fail(oss.str());
    }
}

void test_crop_pool_acquire_release() {
    CropPool pool(256, 224, 112);
    expect_true(pool.capacity() == 256, "capacity");
    expect_true(pool.crop_h() == 224, "crop_h");
    expect_true(pool.crop_w() == 112, "crop_w");
    expect_true(pool.crop_elem_count() == 3 * 224 * 112, "crop_elem_count");
    expect_true(pool.in_use() == 0, "in_use starts at 0");

    int slot = pool.acquire(8);
    expect_true(slot == 0, "first acquire returns slot 0");
    expect_true(pool.in_use() == 8, "in_use after acquire(8)");

    int slot2 = pool.acquire(4);
    expect_true(slot2 == 8, "second acquire returns slot 8");
    expect_true(pool.in_use() == 12, "in_use after acquire(4)");

    pool.release(0, 12);
    expect_true(pool.in_use() == 0, "in_use after release");

    // Overflow: acquire more than capacity
    int slot3 = pool.acquire(257);
    expect_true(slot3 == -1, "acquire over capacity returns -1");
    expect_true(pool.in_use() == 0, "in_use unchanged after failed acquire");

    // Acquire exactly capacity
    int slot4 = pool.acquire(256);
    expect_true(slot4 == 0, "acquire exactly capacity returns 0");
    expect_true(pool.in_use() == 256, "in_use at capacity");
    pool.release(0, 256);
    expect_true(pool.in_use() == 0, "in_use after full release");

    // slot_ptr should be non-null for valid slots, null for invalid
    expect_true(pool.slot_ptr(0) != nullptr, "slot_ptr(0) non-null");
    expect_true(pool.slot_ptr(255) != nullptr, "slot_ptr(255) non-null");
    expect_true(pool.slot_ptr(256) == nullptr, "slot_ptr(256) null (out of range)");
    expect_true(pool.slot_ptr(-1) == nullptr, "slot_ptr(-1) null");

    // Verify slot_ptr arithmetic: slot_ptr(1) - slot_ptr(0) should equal crop_elem_count()
    float* p0 = pool.slot_ptr(0);
    float* p1 = pool.slot_ptr(1);
    ptrdiff_t diff = p1 - p0;
    expect_true(diff == pool.crop_elem_count(), "slot_ptr arithmetic correct");

    std::cout << "crop_pool_acquire_release passed" << std::endl;
}

void test_crop_pool_per_slot_release() {
    // Phase 2: per-slot free-list — release individual slots and re-acquire.
    CropPool pool(8, 224, 112);
    expect_true(pool.available() == 8, "all available");

    // Acquire 4 contiguous slots.
    int slot = pool.acquire(4);
    expect_true(slot == 0, "acquire 4 → slot 0");
    expect_true(pool.available() == 4, "4 remaining");

    // Release slots 2 and 3 (partial release).
    pool.release(2, 2);
    expect_true(pool.available() == 6, "6 available after partial release");

    // Acquire 2 — should get slots 2 and 3 back (contiguous run).
    int slot2 = pool.acquire(2);
    expect_true(slot2 == 2, "re-acquire released slots");
    expect_true(pool.available() == 4, "4 available after re-acquire");

    // Release everything.
    pool.release(0, 2);
    pool.release(2, 2);
    expect_true(pool.available() == 8, "all available after full release");

    // Interleaved acquire/release pattern (simulates async crop + reid).
    int s1 = pool.acquire(2);  // slots 0,1
    int s2 = pool.acquire(2);  // slots 2,3
    pool.release(s1, 2);       // return 0,1
    int s3 = pool.acquire(1);  // should get slot 0 or 1
    expect_true(s3 >= 0, "acquire after interleaved release");
    pool.release(s2, 2);       // return 2,3
    pool.release(s3, 1);       // return s3
    expect_true(pool.available() == 8, "all available after interleaved test");

    std::cout << "crop_pool_per_slot_release passed" << std::endl;
}

void test_crop_pool_data_roundtrip() {
    // Write known values into pool slots via slot_ptr, read them back.
    const int cap = 4;
    const int ch = 224;
    const int cw = 112;
    CropPool pool(cap, ch, cw);

    int slot = pool.acquire(2);
    expect_true(slot == 0, "acquire 2 slots");

    // Write a pattern into slot 0
    float* p0 = pool.slot_ptr(0);
    int elem = 3 * ch * cw;
    std::vector<float> h_data(elem);
    for (int i = 0; i < elem; ++i) h_data[i] = static_cast<float>(i) * 0.001f;
    check_cuda(cudaMemcpy(p0, h_data.data(), elem * sizeof(float),
                          cudaMemcpyHostToDevice), "write slot 0");

    // Write a different pattern into slot 1
    float* p1 = pool.slot_ptr(1);
    std::vector<float> h_data2(elem, 0.5f);
    check_cuda(cudaMemcpy(p1, h_data2.data(), elem * sizeof(float),
                          cudaMemcpyHostToDevice), "write slot 1");

    // Read back slot 0
    std::vector<float> h_readback(elem);
    check_cuda(cudaMemcpy(h_readback.data(), p0, elem * sizeof(float),
                          cudaMemcpyDeviceToHost), "read slot 0");
    for (int i = 0; i < elem; ++i) {
        if (std::fabs(h_readback[i] - h_data[i]) > 1e-6f) {
            fail("slot 0 data mismatch");
        }
    }

    // Read back slot 1
    check_cuda(cudaMemcpy(h_readback.data(), p1, elem * sizeof(float),
                          cudaMemcpyDeviceToHost), "read slot 1");
    for (int i = 0; i < elem; ++i) {
        if (std::fabs(h_readback[i] - 0.5f) > 1e-6f) {
            fail("slot 1 data mismatch");
        }
    }

    pool.release(0, 2);
    std::cout << "crop_pool_data_roundtrip passed" << std::endl;
}

void test_frame_buffer_pool_acquire_release() {
    FrameBufferPool pool(4, 480, 640);
    expect_true(pool.capacity() == 4, "frame buffer capacity");
    expect_true(pool.available() == 4, "all available initially");

    float* buf0 = pool.acquire();
    expect_true(buf0 != nullptr, "acquire returns non-null");
    expect_true(pool.available() == 3, "one less available");

    float* buf1 = pool.acquire();
    expect_true(buf1 != nullptr, "second acquire non-null");
    expect_true(buf0 != buf1, "different buffers");

    pool.release(buf0);
    expect_true(pool.available() == 3, "available after release");

    pool.release(buf1);
    expect_true(pool.available() == 4, "all available after full release");

    // Exhaust the pool
    float* bufs[4];
    for (int i = 0; i < 4; ++i) {
        bufs[i] = pool.acquire();
        expect_true(bufs[i] != nullptr, "acquire all");
    }
    expect_true(pool.available() == 0, "pool exhausted");
    float* null_buf = pool.acquire();
    expect_true(null_buf == nullptr, "acquire on empty pool returns null");
    for (int i = 0; i < 4; ++i) pool.release(bufs[i]);
    expect_true(pool.available() == 4, "all returned");

    std::cout << "frame_buffer_pool_acquire_release passed" << std::endl;
}

void test_frame_buffer_pool_release_after() {
    FrameBufferPool pool(2, 480, 640);
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "stream create");

    float* buf = pool.acquire();
    expect_true(buf != nullptr, "acquire for release_after test");
    expect_true(pool.available() == 1, "one in use");

    // Record an event on the stream and sync — event is complete.
    cudaEvent_t event;
    check_cuda(cudaEventCreate(&event), "event create");
    check_cuda(cudaEventRecord(event, stream), "event record");
    check_cuda(cudaStreamSynchronize(stream), "stream sync");

    // release_after — event is complete, so poll should return it
    pool.release_after(buf, event);
    expect_true(pool.available() == 1, "not returned until poll");

    pool.poll_releases();
    expect_true(pool.available() == 2, "returned after poll (event complete)");

    // Test release_after with an event that gets completed later.
    // Create a fresh event, record it, but don't sync yet.
    float* buf2 = pool.acquire();
    expect_true(buf2 != nullptr, "acquire second");
    expect_true(pool.available() == 1, "one in use again");

    cudaEvent_t event2;
    check_cuda(cudaEventCreate(&event2), "event2 create");
    check_cuda(cudaEventRecord(event2, stream), "event2 record");
    // Don't sync — event2 may or may not be complete depending on timing.
    pool.release_after(buf2, event2);

    // Force completion by syncing, then poll.
    check_cuda(cudaStreamSynchronize(stream), "stream sync for event2");
    pool.poll_releases();
    expect_true(pool.available() == 2, "returned after event2 sync + poll");

    check_cuda(cudaStreamDestroy(stream), "stream destroy");
    std::cout << "frame_buffer_pool_release_after passed" << std::endl;
}

}  // namespace

int main() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fail("No CUDA device available for crop_pool_test");
    }

    test_crop_pool_acquire_release();
    test_crop_pool_per_slot_release();
    test_crop_pool_data_roundtrip();
    test_frame_buffer_pool_acquire_release();
    test_frame_buffer_pool_release_after();

    std::cout << "crop pool / frame buffer tests passed" << std::endl;
    return 0;
}
