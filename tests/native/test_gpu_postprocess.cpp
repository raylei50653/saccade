#include "tracking/tracker_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

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

void expect_near(float actual, float expected, float tolerance, const std::string& label) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream oss;
        oss << label << " expected " << expected << " got " << actual;
        fail(oss.str());
    }
}

void check_cuda(cudaError_t status, const std::string& label) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << label << ": " << cudaGetErrorString(status);
        fail(oss.str());
    }
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;

    explicit DeviceBuffer(size_t count) {
        check_cuda(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
void copy_to_device(T* dst, const std::vector<T>& src) {
    check_cuda(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

template <typename T>
std::vector<T> copy_to_host(const T* src, size_t count) {
    std::vector<T> out(count);
    check_cuda(cudaMemcpy(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    return out;
}

std::vector<unsigned char> copy_bool_bytes_to_host(const bool* src, size_t count) {
    std::vector<unsigned char> out(count);
    check_cuda(cudaMemcpy(out.data(), src, count * sizeof(bool), cudaMemcpyDeviceToHost), "cudaMemcpy bool D2H");
    return out;
}

void test_filter_detections_cuda() {
    const std::vector<float> boxes{
        10.0f, 10.0f, 20.0f, 40.0f,
        10.0f, 10.0f, 20.0f, 40.0f,
        15.0f, 15.0f, 25.0f, 45.0f,
        120.0f, 10.0f, 130.0f, 45.0f,
        30.0f, 10.0f, 80.0f, 20.0f,
    };
    const std::vector<float> scores{0.95f, 0.90f, 0.01f, 0.80f, 0.85f};
    const std::vector<int> classes{0, 1, 0, 0, 0};
    const int num_dets = 5;

    DeviceBuffer<float> d_boxes(boxes.size());
    DeviceBuffer<float> d_scores(scores.size());
    DeviceBuffer<int> d_classes(classes.size());
    DeviceBuffer<int> d_keep(num_dets);
    DeviceBuffer<bool> d_suspect(num_dets);
    DeviceBuffer<int> d_count(1);

    copy_to_device(d_boxes.ptr, boxes);
    copy_to_device(d_scores.ptr, scores);
    copy_to_device(d_classes.ptr, classes);

    saccade::filter_detections_cuda(
        d_boxes.ptr,
        d_scores.ptr,
        d_classes.ptr,
        num_dets,
        d_keep.ptr,
        d_suspect.ptr,
        d_count.ptr,
        0.05f,
        true,
        0,
        true,
        100,
        100,
        true,
        true,
        0.018f,
        1.0f,
        5.5f,
        0.00006f,
        0.0f,
        nullptr
    );
    check_cuda(cudaDeviceSynchronize(), "filter_detections_cuda sync");

    const int count = copy_to_host(d_count.ptr, 1)[0];
    expect_true(count == 2, "filter_detections_cuda expected 2 kept detections");

    const std::vector<int> keep = copy_to_host(d_keep.ptr, count);
    const std::vector<unsigned char> suspect = copy_bool_bytes_to_host(d_suspect.ptr, count);
    std::map<int, bool> keep_map;
    for (int i = 0; i < count; ++i) {
        keep_map[keep[static_cast<size_t>(i)]] = suspect[static_cast<size_t>(i)];
    }

    expect_true(keep_map.size() == 2, "filter_detections_cuda expected 2 unique keep indices");
    expect_true(keep_map.count(0) == 1, "filter_detections_cuda missing clean person detection");
    expect_true(keep_map.count(4) == 1, "filter_detections_cuda missing geometry suspect detection");
    expect_true(keep_map[0] == false, "filter_detections_cuda clean detection marked suspect");
    expect_true(keep_map[4] == true, "filter_detections_cuda geometry suspect not marked");
}

void test_nms_cuda() {
    const std::vector<float> boxes{
        0.0f, 0.0f, 10.0f, 10.0f,
        1.0f, 1.0f, 11.0f, 11.0f,
        30.0f, 30.0f, 40.0f, 40.0f,
    };
    const std::vector<float> scores{0.95f, 0.90f, 0.80f};
    const std::vector<int> classes{0, 1, 1};
    const std::vector<int64_t> order{0, 1, 2};
    const int num_dets = 3;
    const int col_blocks = 1;

    DeviceBuffer<float> d_boxes(boxes.size());
    DeviceBuffer<float> d_scores(scores.size());
    DeviceBuffer<int> d_classes(classes.size());
    DeviceBuffer<int64_t> d_order(order.size());
    DeviceBuffer<int> d_keep(num_dets);
    DeviceBuffer<uint64_t> d_masks(static_cast<size_t>(num_dets) * col_blocks);
    DeviceBuffer<uint64_t> d_remv(col_blocks);
    DeviceBuffer<int> d_count(1);

    copy_to_device(d_boxes.ptr, boxes);
    copy_to_device(d_scores.ptr, scores);
    copy_to_device(d_classes.ptr, classes);
    copy_to_device(d_order.ptr, order);

    saccade::nms_cuda(
        d_boxes.ptr,
        d_scores.ptr,
        d_classes.ptr,
        d_order.ptr,
        num_dets,
        d_keep.ptr,
        d_masks.ptr,
        d_remv.ptr,
        d_count.ptr,
        0.5f,
        false,
        nullptr
    );
    check_cuda(cudaDeviceSynchronize(), "nms_cuda sync");

    const int count = copy_to_host(d_count.ptr, 1)[0];
    expect_true(count == 2, "nms_cuda expected 2 kept detections");
    const std::vector<int> keep = copy_to_host(d_keep.ptr, count);
    expect_true(keep == std::vector<int>({0, 2}), "nms_cuda unexpected keep order");

    saccade::nms_cuda(
        d_boxes.ptr,
        d_scores.ptr,
        d_classes.ptr,
        d_order.ptr,
        num_dets,
        d_keep.ptr,
        d_masks.ptr,
        d_remv.ptr,
        d_count.ptr,
        0.5f,
        true,
        nullptr
    );
    check_cuda(cudaDeviceSynchronize(), "nms_cuda class-aware sync");

    const int class_aware_count = copy_to_host(d_count.ptr, 1)[0];
    expect_true(class_aware_count == 3, "nms_cuda class-aware expected 3 kept detections");
}

void test_merge_cross_tile_duplicates_cuda() {
    const std::vector<float> boxes{
        0.0f, 0.0f, 10.0f, 10.0f,
        1.0f, 1.0f, 11.0f, 11.0f,
        50.0f, 50.0f, 60.0f, 60.0f,
    };
    const std::vector<float> scores{0.90f, 0.60f, 0.80f};
    const std::vector<int> classes{0, 0, 0};
    const int num_dets = 3;

    DeviceBuffer<float> d_boxes(boxes.size());
    DeviceBuffer<float> d_scores(scores.size());
    DeviceBuffer<int> d_classes(classes.size());
    DeviceBuffer<int> d_anchor(num_dets);
    DeviceBuffer<float> d_box_sums(boxes.size());
    DeviceBuffer<float> d_score_sums(scores.size());
    DeviceBuffer<int> d_score_bits_max(num_dets);
    DeviceBuffer<int> d_cluster_counts(num_dets);
    DeviceBuffer<float> d_out_boxes(boxes.size());
    DeviceBuffer<float> d_out_scores(scores.size());
    DeviceBuffer<int> d_out_classes(classes.size());
    DeviceBuffer<int> d_out_count(1);

    copy_to_device(d_boxes.ptr, boxes);
    copy_to_device(d_scores.ptr, scores);
    copy_to_device(d_classes.ptr, classes);

    saccade::merge_cross_tile_duplicates_cuda(
        d_boxes.ptr,
        d_scores.ptr,
        d_classes.ptr,
        num_dets,
        d_anchor.ptr,
        d_box_sums.ptr,
        d_score_sums.ptr,
        d_score_bits_max.ptr,
        d_cluster_counts.ptr,
        d_out_boxes.ptr,
        d_out_scores.ptr,
        d_out_classes.ptr,
        d_out_count.ptr,
        0.45f,
        0.18f,
        0.6f,
        nullptr
    );
    check_cuda(cudaDeviceSynchronize(), "merge_cross_tile_duplicates_cuda sync");

    const int out_count = copy_to_host(d_out_count.ptr, 1)[0];
    expect_true(out_count == 2, "merge_cross_tile_duplicates_cuda expected 2 merged clusters");

    const std::vector<float> out_boxes = copy_to_host(d_out_boxes.ptr, static_cast<size_t>(out_count) * 4);
    const std::vector<float> out_scores = copy_to_host(d_out_scores.ptr, out_count);
    const std::vector<int> out_classes = copy_to_host(d_out_classes.ptr, out_count);

    expect_near(out_boxes[0], 0.4f, 1e-3f, "merged box x1");
    expect_near(out_boxes[1], 0.4f, 1e-3f, "merged box y1");
    expect_near(out_boxes[2], 10.4f, 1e-3f, "merged box x2");
    expect_near(out_boxes[3], 10.4f, 1e-3f, "merged box y2");
    expect_near(out_scores[0], 0.90f, 1e-6f, "merged score max");
    expect_true(out_classes[0] == 0, "merged class mismatch");

    expect_near(out_boxes[4], 50.0f, 1e-6f, "standalone box x1");
    expect_near(out_boxes[7], 60.0f, 1e-6f, "standalone box y2");
    expect_near(out_scores[1], 0.80f, 1e-6f, "standalone score");
}

}  // namespace

int main() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fail("No CUDA device available for saccade_gpu_postprocess_test");
    }

    test_filter_detections_cuda();
    test_nms_cuda();
    test_merge_cross_tile_duplicates_cuda();

    std::cout << "gpu postprocess tests passed" << std::endl;
    return 0;
}
