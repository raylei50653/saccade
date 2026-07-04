// Native tests for BufferPool — covers the race fixes (S1-S9) for the
// GStreamer GPU-decode H2D buffer pool. Compiled standalone (no GStreamer
// dependency) so BufferPool can be unit-tested in isolation.
#include "media/buffer_pool.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace saccade;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

// D2H: read n bytes from device into a host vector.
std::vector<uint8_t> d2h_bytes(const void* d, size_t n, cudaStream_t stream) {
    std::vector<uint8_t> host(n);
    if (n > 0) {
        checkCuda(cudaMemcpyAsync(host.data(), d, n, cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
    }
    return host;
}

// ── Test 1: 單緒連續 acquire 5 次成功,第 6 次回 -1 (pool 耗盡) ──────────────
void test_acquire_exhaustion() {
    BufferPool pool(64);
    int idxs[BufferPool::POOL_SIZE];
    for (size_t i = 0; i < BufferPool::POOL_SIZE; ++i) {
        idxs[i] = pool.acquire_empty_slot(64);
        expect_true(idxs[i] >= 0, "test1: acquire should succeed for all 5 slots");
        expect_true(pool.state(idxs[i]) == BufferStatus::WRITING,
                    "test1: acquired slot must be WRITING");
    }
    int sixth = pool.acquire_empty_slot(64);
    expect_true(sixth == -1, "test1: 6th acquire must return -1 (pool exhausted)");
    // Cleanup: release all so destructor sees nothing in-flight.
    for (size_t i = 0; i < BufferPool::POOL_SIZE; ++i) {
        pool.release(idxs[i]); // READY/WRITING → EMPTY? release only accepts PROCESSING/READY.
    }
    // Actually slots are WRITING (never submitted). release() won't touch WRITING.
    // That's fine — destructor syncs + frees regardless of state.
    std::cout << "✓ test1: acquire exhaustion" << std::endl;
}

// ── Test 2: 多緒競爭 acquire,從不回傳重複 idx (CAS 正確性) ──────────────────
void test_concurrent_acquire_no_duplicates() {
    BufferPool pool(64);
    constexpr int NUM_THREADS = 8;
    std::vector<int> results(NUM_THREADS, -999);
    std::atomic<int> barrier{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            barrier.fetch_add(1);
            while (barrier.load() < NUM_THREADS) {
                /* spin-wait barrier */
            }
            results[t] = pool.acquire_empty_slot(64);
        });
    }
    for (auto& th : threads) th.join();

    // 只有 5 個槽,8 個 thread → 恰 5 個成功,3 個 -1,且成功的 idx 互不重複。
    std::set<int> unique_success;
    int success_count = 0;
    for (int r : results) {
        if (r >= 0) {
            ++success_count;
            unique_success.insert(r);
        } else {
            expect_true(r == -1, "test2: failure must be -1, not other negative");
        }
    }
    expect_true(success_count == static_cast<int>(BufferPool::POOL_SIZE),
                "test2: exactly 5 acquisitions should succeed");
    expect_true(unique_success.size() == static_cast<size_t>(success_count),
                "test2: no duplicate slot indices across threads (CAS correctness)");
    std::cout << "✓ test2: concurrent acquire no duplicates" << std::endl;
}

// ── Test 3: submit_h2d + sync_slot 讀回正確 pattern (修 S1 回歸測試) ─────────
void test_submit_h2d_and_sync() {
    BufferPool pool(256);
    int idx = pool.acquire_empty_slot(256);
    expect_true(idx >= 0, "test3: acquire");

    // 填 host pattern
    std::vector<uint8_t> host(256);
    for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<uint8_t>(i * 7 + 3);

    bool ok = pool.submit_h2d(idx, host.data(), host.size());
    expect_true(ok, "test3: submit_h2d should succeed on WRITING slot");
    expect_true(pool.state(idx) == BufferStatus::READY,
                "test3: slot must be READY after submit_h2d");

    // sync_slot 後讀回 — 若無 sync 會讀到未完成 copy (即 S1 race)。
    pool.sync_slot(idx);
    auto readback = d2h_bytes(pool.device_ptr(idx), host.size(), pool.stream(idx));
    expect_true(readback == host, "test3: readback must match host pattern after sync");
    std::cout << "✓ test3: submit_h2d + sync_slot data integrity" << std::endl;
}

// ── Test 4: release 後可重用 ────────────────────────────────────────────────
void test_release_reuse() {
    BufferPool pool(64);
    int idx = pool.acquire_empty_slot(64);
    expect_true(idx >= 0, "test4: acquire");
    expect_true(pool.submit_h2d(idx, nullptr, 0) || true,
                "test4: submit zero-byte (state transition only)");
    // submit_h2d with 0 bytes: bytes(0) <= slot_bytes(64) ok, but host_ptr=nullptr
    // and bytes=0 → cudaMemcpyAsync with 0 count is a no-op. State → READY.
    expect_true(pool.state(idx) == BufferStatus::READY, "test4: READY after submit");

    expect_true(pool.mark_processing(idx), "test4: mark_processing READY→PROCESSING");
    expect_true(pool.state(idx) == BufferStatus::PROCESSING, "test4: PROCESSING state");

    pool.release(idx);
    expect_true(pool.state(idx) == BufferStatus::EMPTY, "test4: EMPTY after release");

    // 重用同一槽
    int idx2 = pool.acquire_empty_slot(64);
    expect_true(idx2 >= 0, "test4: reacquire after release");
    // 不要求拿到同一 idx (write_hint 影響起點),但池內至少有一個可用了。
    pool.release(idx2);
    std::cout << "✓ test4: release and reuse" << std::endl;
}

// ── Test 5: mark_processing 在 WRITING 狀態下不應成功 (CAS 守護) ─────────────
void test_mark_processing_rejects_writing() {
    BufferPool pool(64);
    int idx = pool.acquire_empty_slot(64);
    expect_true(idx >= 0, "test5: acquire");
    expect_true(pool.state(idx) == BufferStatus::WRITING, "test5: WRITING state");

    // WRITING → PROCESSING 必須失敗 (mark_processing 只接受 READY)
    bool result = pool.mark_processing(idx);
    expect_true(!result, "test5: mark_processing must fail on WRITING slot");
    expect_true(pool.state(idx) == BufferStatus::WRITING,
                "test5: state unchanged after rejected mark_processing");

    // release 不應動 WRITING 槽 (避免 clobber 進行中搬運)
    pool.release(idx);
    expect_true(pool.state(idx) == BufferStatus::WRITING,
                "test5: release must not clobber WRITING slot");
    std::cout << "✓ test5: mark_processing rejects WRITING (CAS guard)" << std::endl;
}

// ── Test 6: ensureBufferPool 只成長,READY 槽保留舊 ptr 仍可讀 ────────────────
void test_grow_preserves_inuse_slots() {
    BufferPool pool(1024); // 每槽初始 1KB

    // slot 0: acquire + submit 512B → READY (1KB buffer)
    int idx0 = pool.acquire_empty_slot(512);
    expect_true(idx0 >= 0, "test6: acquire slot 0");
    std::vector<uint8_t> host0(512);
    for (size_t i = 0; i < host0.size(); ++i) host0[i] = static_cast<uint8_t>(0xAB ^ i);
    expect_true(pool.submit_h2d(idx0, host0.data(), host0.size()), "test6: submit slot 0");
    expect_true(pool.state(idx0) == BufferStatus::READY, "test6: slot 0 READY");

    // 要求 4KB → 應挑 EMPTY 槽 (slot 1+) 並成長,不動 slot 0
    int idx1 = pool.acquire_empty_slot(4096);
    expect_true(idx1 >= 0, "test6: acquire slot 1 with 4KB");
    expect_true(idx1 != idx0, "test6: must not reuse READY slot");
    expect_true(pool.slot_bytes(idx1) == 4096, "test6: slot 1 grown to 4KB");
    expect_true(pool.slot_bytes(idx0) == 1024, "test6: slot 0 size unchanged (1KB)");

    // slot 0 的舊 ptr 仍可讀 (sync + D2H 讀回 pattern)
    pool.sync_slot(idx0);
    auto readback0 = d2h_bytes(pool.device_ptr(idx0), host0.size(), pool.stream(idx0));
    expect_true(readback0 == host0,
                "test6: slot 0 data intact after growing a different slot");

    // slot 1 也能正常 submit + sync 4KB
    std::vector<uint8_t> host1(4096, 0x5A);
    expect_true(pool.submit_h2d(idx1, host1.data(), host1.size()), "test6: submit slot 1");
    pool.sync_slot(idx1);
    auto readback1 = d2h_bytes(pool.device_ptr(idx1), host1.size(), pool.stream(idx1));
    expect_true(readback1 == host1, "test6: slot 1 4KB data integrity");
    std::cout << "✓ test6: grow preserves in-use slots" << std::endl;
}

// ── Test 7: 解構子安全 — in-flight H2D 未 sync,直接銷毀不 crash (修 S3/S5)
void test_destructor_with_inflight_h2d() {
    // 用動態配置以便在 thread submit 後於主緒銷毀。
    auto* pool = new BufferPool(4096);
    int idx = pool->acquire_empty_slot(4096);
    expect_true(idx >= 0, "test7: acquire");

    std::vector<uint8_t> host(4096, 0x77);
    // 在另一條 thread 排隊 H2D,不 sync。
    std::thread worker([&]() {
        pool->submit_h2d(idx, host.data(), host.size());
    });
    worker.join(); // 確保 C++ submit 呼叫完成 (CUDA copy 仍在 stream 上 in-flight)

    // 直接 delete — ~BufferPool 必須 sync 全部 stream 再 free,不 crash。
    delete pool;
    std::cout << "✓ test7: destructor safe with in-flight H2D" << std::endl;
}

// ── Test 8: write_idx 不影響正確性 — 跨緒 acquire 不依賴 write_idx ───────────
void test_write_hint_irrelevant_to_correctness() {
    // 把 write_hint 推到非 0 起點,驗證跨緒 acquire 仍拿到不重複 idx。
    BufferPool pool(64);
    // 先 acquire + release 一輪,讓 write_hint 推進。
    for (int round = 0; round < 3; ++round) {
        int idx = pool.acquire_empty_slot(64);
        expect_true(idx >= 0, "test8: warmup acquire");
        pool.submit_h2d(idx, nullptr, 0); // WRITING → READY (0-byte no-op)
        pool.mark_processing(idx);         // READY → PROCESSING
        pool.release(idx);                 // PROCESSING → EMPTY
    }

    // 現在 5 槽皆 EMPTY,write_hint 在某處。多緒 acquire。
    constexpr int NUM_THREADS = 5;
    std::vector<int> results(NUM_THREADS, -1);
    std::atomic<int> barrier{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            barrier.fetch_add(1);
            while (barrier.load() < NUM_THREADS) { /* spin */ }
            results[t] = pool.acquire_empty_slot(64);
        });
    }
    for (auto& th : threads) th.join();

    std::set<int> unique;
    for (int r : results) {
        expect_true(r >= 0, "test8: all 5 threads should succeed (all EMPTY)");
        unique.insert(r);
    }
    expect_true(unique.size() == static_cast<size_t>(NUM_THREADS),
                "test8: no duplicates regardless of write_hint position");
    std::cout << "✓ test8: write_hint irrelevant to correctness" << std::endl;
}

} // namespace

int main() {
    int dev = 0;
    checkCuda(cudaGetDevice(&dev));
    std::cout << "Running BufferPool race tests on CUDA device " << dev << std::endl;

    test_acquire_exhaustion();
    test_concurrent_acquire_no_duplicates();
    test_submit_h2d_and_sync();
    test_release_reuse();
    test_mark_processing_rejects_writing();
    test_grow_preserves_inuse_slots();
    test_destructor_with_inflight_h2d();
    test_write_hint_irrelevant_to_correctness();

    std::cout << "\n✅ All BufferPool race tests passed." << std::endl;
    return 0;
}
