// Native tests for the FPN ReID CUDA launchers (fpn_reid_cuda.cu) — verifies
// centre_pool indexing, conv1x1, linear, bn1d, and l2_normalise against host
// references.  The launchers are C-linkage (fpn_reid_launchers.cuh); the .cu
// is compiled into saccade_tracking.
#include "tracking/fpn_reid_launchers.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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
        oss << label << ": expected " << expected << " got " << actual
            << " (tol " << tolerance << ")";
        fail(oss.str());
    }
}

std::vector<float> d2h(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

} // namespace

// ── centre_pool: pick the feature channel at the box-centre spatial cell ─
static void test_centre_pool() {
    // feat: [C=3, H=2, W=2], values = channel index so we can verify indexing.
    // layout CHW: feat[c*H*W + y*W + x]
    std::vector<float> feat = {
        0,0, 0,0,   // channel 0
        1,1, 1,1,   // channel 1
        2,2, 2,2,   // channel 2
    };
    const int C = 3, H = 2, W = 2, img_size = 2;
    // 2 boxes: centre of box0 at (0.5,0.5)*img -> cell (0,0); box1 centre (1.5,1.5)->(1,1)
    std::vector<float> boxes = {
        0.0f, 0.0f, 1.0f, 1.0f,   // centre (0.5,0.5) -> cx_idx=0, cy_idx=0
        1.0f, 1.0f, 2.0f, 2.0f,   // centre (1.5,1.5) -> cx_idx=1, cy_idx=1
    };
    const int N = 2;
    std::vector<float> pooled_ref = {
        0.0f, 1.0f, 2.0f,   // box0 at (0,0): channels 0,1,2
        0.0f, 1.0f, 2.0f,   // box1 at (1,1): all channels constant per-channel
    };

    float *d_feat, *d_boxes, *d_out;
    cudaMalloc(&d_feat, feat.size() * sizeof(float));
    cudaMalloc(&d_boxes, boxes.size() * sizeof(float));
    cudaMalloc(&d_out, pooled_ref.size() * sizeof(float));
    cudaMemcpy(d_feat, feat.data(), feat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boxes, boxes.data(), boxes.size() * sizeof(float), cudaMemcpyHostToDevice);

    fpn_centre_pool(d_feat, C, H, W, d_boxes, N, d_out, img_size, nullptr);
    auto got = d2h(d_out, pooled_ref.size());
    for (size_t i = 0; i < pooled_ref.size(); ++i) {
        expect_near(got[i], pooled_ref[i], 1e-6f, "centre_pool value");
    }

    cudaFree(d_feat); cudaFree(d_boxes); cudaFree(d_out);
}

// ── conv1x1: out = weight @ pooled (one row per output channel) ──────────
static void test_conv1x1() {
    const int N = 2, C = 3, O = 2;
    std::vector<float> pooled = {1,2,3, 4,5,6};              // [N,C]
    std::vector<float> weight = {1,0,0, 0,1,0};              // [O,C] -> identity-ish
    std::vector<float> ref = {1,2, 4,5};                     // row0=ch0, row1=ch1
    float *d_in, *d_w, *d_out;
    cudaMalloc(&d_in, pooled.size() * sizeof(float));
    cudaMalloc(&d_w, weight.size() * sizeof(float));
    cudaMalloc(&d_out, ref.size() * sizeof(float));
    cudaMemcpy(d_in, pooled.data(), pooled.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    fpn_conv1x1(d_in, C, d_w, O, d_out, N, nullptr);
    auto got = d2h(d_out, ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        expect_near(got[i], ref[i], 1e-5f, "conv1x1 value");
    }
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
}

// ── linear: same math as conv1x1 but on a concatenated (mid_dim) input ────
static void test_linear() {
    const int N = 1, D = 4, O = 2;
    std::vector<float> data = {1,2,3,4};
    std::vector<float> weight = {1,1,0,0, 0,0,1,1};          // [O,D]
    std::vector<float> ref = {3, 7};                          // 1+2, 3+4
    float *d_in, *d_w, *d_out;
    cudaMalloc(&d_in, data.size() * sizeof(float));
    cudaMalloc(&d_w, weight.size() * sizeof(float));
    cudaMalloc(&d_out, ref.size() * sizeof(float));
    cudaMemcpy(d_in, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    fpn_linear(d_in, D, d_w, O, d_out, N, nullptr);
    auto got = d2h(d_out, ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        expect_near(got[i], ref[i], 1e-5f, "linear value");
    }
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
}

// ── bn1d eval: (x - mean)/sqrt(var + eps) per channel ────────────────────
static void test_bn1d() {
    const int N = 2, D = 2;
    std::vector<float> data = {2,4, 4,8};                    // [N,D]
    std::vector<float> mean = {0,4};
    std::vector<float> var  = {1,4};
    float eps = 1e-5f;
    // row0: (2-0)/1=2, (4-4)/2=0 ; row1: (4-0)/1=4, (8-4)/2=2
    std::vector<float> ref = {2,0, 4,2};
    float *d_in, *d_m, *d_v, *d_out;
    cudaMalloc(&d_in, data.size() * sizeof(float));
    cudaMalloc(&d_m, mean.size() * sizeof(float));
    cudaMalloc(&d_v, var.size() * sizeof(float));
    cudaMalloc(&d_out, ref.size() * sizeof(float));
    cudaMemcpy(d_in, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, mean.data(), mean.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, var.data(), var.size() * sizeof(float), cudaMemcpyHostToDevice);
    fpn_bn1d(d_in, D, N, d_m, d_v, eps, nullptr);
    auto got = d2h(d_in, ref.size());  // in-place
    for (size_t i = 0; i < ref.size(); ++i) {
        expect_near(got[i], ref[i], 1e-3f, "bn1d value");
    }
    cudaFree(d_in); cudaFree(d_m); cudaFree(d_v); cudaFree(d_out);
}

// ── l2_normalise: each row scaled to unit L2 norm ────────────────────────
static void test_l2_normalise() {
    const int N = 2, D = 3;
    std::vector<float> data = {3,4,0, 0,0,5};                // norms 5, 5
    float *d_in;
    cudaMalloc(&d_in, data.size() * sizeof(float));
    cudaMemcpy(d_in, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    fpn_l2_normalise(d_in, D, N, 1e-8f, nullptr);
    auto got = d2h(d_in, data.size());
    // row0: 3/5,4/5,0 ; row1: 0,0,1
    expect_near(got[0], 0.6f, 1e-5f, "l2 row0 x");
    expect_near(got[1], 0.8f, 1e-5f, "l2 row0 y");
    expect_near(got[2], 0.0f, 1e-5f, "l2 row0 z");
    expect_near(got[5], 1.0f, 1e-5f, "l2 row1 z");
    // verify norms == 1
    for (int i = 0; i < N; ++i) {
        float n = 0;
        for (int j = 0; j < D; ++j) n += got[i * D + j] * got[i * D + j];
        expect_near(std::sqrt(n), 1.0f, 1e-4f, "l2 unit norm");
    }
    cudaFree(d_in);
}

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    test_centre_pool();
    test_conv1x1();
    test_linear();
    test_bn1d();
    test_l2_normalise();
    std::cout << "fpn_reid launcher tests passed" << std::endl;
    return 0;
}
