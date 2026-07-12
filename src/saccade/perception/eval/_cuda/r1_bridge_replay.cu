// Research-only: bit-exact Consumer-A bridge_anchor4 / bridge_vel4 on device.
// Compiled as a small shared library and called from the host R0 replay path.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__device__ __forceinline__ float bridge_vel4(const float* y) {
    return (3.0f * y[3] + y[2] - y[1] - 3.0f * y[0]) / 10.0f;
}

__device__ __forceinline__ float bridge_linres4(const float* y) {
    float ybar = 0.25f * (y[0] + y[1] + y[2] + y[3]);
    float sxy = -1.5f * (y[0] - ybar) - 0.5f * (y[1] - ybar)
                + 0.5f * (y[2] - ybar) + 1.5f * (y[3] - ybar);
    float slope = sxy / 5.0f;
    float res = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float fit = ybar + slope * ((float)i - 1.5f);
        float d = y[i] - fit;
        res += d * d;
    }
    return res;
}

// Line-for-line with tracker_gpu.cu::bridge_anchor4.
__device__ __forceinline__ void bridge_anchor4(
    const float* p, int anchor_mode, float rate_gate, int endpoint_idx,
    float& ax, float& ay, float& vx, float& vy)
{
    float cx[4], cy[4], yt[4], yb[4], hbar = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float x = p[i * 3 + 0], c = p[i * 3 + 1], h = p[i * 3 + 2];
        cx[i] = x; cy[i] = c; yt[i] = c - 0.5f * h; yb[i] = c + 0.5f * h;
        hbar += 0.25f * h;
    }
    vx = bridge_vel4(cx);
    ax = cx[endpoint_idx];
    bool use_edges = (anchor_mode == 2);
    if (use_edges && rate_gate > 0.0f) {
        float dh = (fabsf(p[1 * 3 + 2] - p[0 * 3 + 2])
                    + fabsf(p[2 * 3 + 2] - p[1 * 3 + 2])
                    + fabsf(p[3 * 3 + 2] - p[2 * 3 + 2])) / 3.0f;
        if (dh / (hbar + 1e-3f) <= rate_gate) use_edges = false;
    }
    if (anchor_mode == 1) {
        vy = bridge_vel4(yb);
        ay = yb[endpoint_idx];
    } else if (use_edges) {
        float hn = hbar * hbar + 1e-3f;
        float wt = 1.0f / (bridge_linres4(yt) / hn + 0.01f);
        float wb = 1.0f / (bridge_linres4(yb) / hn + 0.01f);
        float ws = wt + wb;
        vy = (wt * bridge_vel4(yt) + wb * bridge_vel4(yb)) / ws;
        ay = (wt * yt[endpoint_idx] + wb * yb[endpoint_idx]) / ws;
    } else {
        vy = bridge_vel4(cy);
        ay = cy[endpoint_idx];
    }
}

__global__ void anchor4_kernel(
    const float* rings, const int* modes, const float* rates, const int* endpoints,
    float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ax, ay, vx, vy;
    bridge_anchor4(rings + i * 12, modes[i], rates[i], endpoints[i], ax, ay, vx, vy);
    out[i * 4 + 0] = ax;
    out[i * 4 + 1] = ay;
    out[i * 4 + 2] = vx;
    out[i * 4 + 3] = vy;
}

__global__ void vel4_kernel(const float* samples, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bridge_vel4(samples + i * 4);
}

extern "C" {

// Batch bridge_anchor4. rings: n*12 floats, out: n*4 floats (ax,ay,vx,vy).
int r1_bridge_anchor4_batch(
    const float* rings_h, const int* modes_h, const float* rates_h,
    const int* endpoints_h, float* out_h, int n)
{
    if (n <= 0) return 0;
    float *d_rings = nullptr, *d_rates = nullptr, *d_out = nullptr;
    int *d_modes = nullptr, *d_eps = nullptr;
    cudaError_t err;
    auto fail = [&](cudaError_t e) -> int {
        if (d_rings) cudaFree(d_rings);
        if (d_modes) cudaFree(d_modes);
        if (d_rates) cudaFree(d_rates);
        if (d_eps) cudaFree(d_eps);
        if (d_out) cudaFree(d_out);
        return int(e);
    };
    if ((err = cudaMalloc(&d_rings, sizeof(float) * 12 * n))) return fail(err);
    if ((err = cudaMalloc(&d_modes, sizeof(int) * n))) return fail(err);
    if ((err = cudaMalloc(&d_rates, sizeof(float) * n))) return fail(err);
    if ((err = cudaMalloc(&d_eps, sizeof(int) * n))) return fail(err);
    if ((err = cudaMalloc(&d_out, sizeof(float) * 4 * n))) return fail(err);
    if ((err = cudaMemcpy(d_rings, rings_h, sizeof(float) * 12 * n, cudaMemcpyHostToDevice))) return fail(err);
    if ((err = cudaMemcpy(d_modes, modes_h, sizeof(int) * n, cudaMemcpyHostToDevice))) return fail(err);
    if ((err = cudaMemcpy(d_rates, rates_h, sizeof(float) * n, cudaMemcpyHostToDevice))) return fail(err);
    if ((err = cudaMemcpy(d_eps, endpoints_h, sizeof(int) * n, cudaMemcpyHostToDevice))) return fail(err);
    int block = 128;
    int grid = (n + block - 1) / block;
    anchor4_kernel<<<grid, block>>>(d_rings, d_modes, d_rates, d_eps, d_out, n);
    if ((err = cudaGetLastError())) return fail(err);
    if ((err = cudaDeviceSynchronize())) return fail(err);
    if ((err = cudaMemcpy(out_h, d_out, sizeof(float) * 4 * n, cudaMemcpyDeviceToHost))) return fail(err);
    fail(cudaSuccess);
    return 0;
}

int r1_bridge_vel4_batch(const float* samples_h, float* out_h, int n) {
    if (n <= 0) return 0;
    float *d_in = nullptr, *d_out = nullptr;
    cudaError_t err;
    if ((err = cudaMalloc(&d_in, sizeof(float) * 4 * n))) return int(err);
    if ((err = cudaMalloc(&d_out, sizeof(float) * n))) { cudaFree(d_in); return int(err); }
    if ((err = cudaMemcpy(d_in, samples_h, sizeof(float) * 4 * n, cudaMemcpyHostToDevice))) {
        cudaFree(d_in); cudaFree(d_out); return int(err);
    }
    int block = 128;
    int grid = (n + block - 1) / block;
    vel4_kernel<<<grid, block>>>(d_in, d_out, n);
    if ((err = cudaGetLastError())) { cudaFree(d_in); cudaFree(d_out); return int(err); }
    if ((err = cudaDeviceSynchronize())) { cudaFree(d_in); cudaFree(d_out); return int(err); }
    if ((err = cudaMemcpy(out_h, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost))) {
        cudaFree(d_in); cudaFree(d_out); return int(err);
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

}  // extern "C"
