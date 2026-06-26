// Native test for the GMC grayscale-downscale kernel's Hanning window
// divide-by-zero guard (gmc_kernel.cu).  The guard makes dst_w/dst_h == 1
// fall back to a no-window (1.0) multiplier instead of dividing by (dim-1)=0.
//
// launch_grayscale_downscale is in namespace saccade (exported from
// saccade_tracking); declared as an extern here since it has no public header.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace saccade {
void launch_grayscale_downscale(const float* src, float* dst,
                                int src_w, int src_h, int dst_w, int dst_h,
                                cudaStream_t stream);
}

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

} // namespace

// ── dst_w == 1 must not fault; output should be the grayscaled source ────
static void test_hanning_dst1_no_fault() {
    // src: 2x2 CHW, all ones → gray = 0.299+0.587+0.114 = 1.0 per pixel.
    const int src_w = 2, src_h = 2;
    std::vector<float> src((size_t)3 * src_w * src_h, 1.0f);
    float* d_src; cudaMalloc(&d_src, src.size() * sizeof(float));
    cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Downscale to 1x1 — exercises the (dst_w-1)==0 and (dst_h-1)==0 guard.
    const int dst_w = 1, dst_h = 1;
    float* d_dst; cudaMalloc(&d_dst, (size_t)dst_w * dst_h * sizeof(float));

    // Must not crash (divide by zero) and must enqueue successfully.
    saccade::launch_grayscale_downscale(d_src, d_dst, src_w, src_h, dst_w, dst_h, nullptr);
    cudaError_t err = cudaDeviceSynchronize();
    expect_true(err == cudaSuccess, "dst=1 downscale did not fault");

    float got = 0.0f;
    cudaMemcpy(&got, d_dst, sizeof(float), cudaMemcpyDeviceToHost);
    // Grayscale of (1,1,1) = 1.0; window multiplier is 1.0 (guard) → 1.0.
    expect_true(std::fabs(got - 1.0f) < 1e-5f, "dst=1 grayscale value preserved");

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    test_hanning_dst1_no_fault();
    std::cout << "gmc hanning guard test passed" << std::endl;
    return 0;
}
