// Bounded persistent residency pressure for #419; NOT an SM partition.
// Build: nvcc -shared -Xcompiler -fPIC -O2 -arch=sm_120 blocker.cu -o blocker.so
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

struct Sample { unsigned sm; unsigned long long begin, end; };
static cudaStream_t stream;
static Sample *device_samples;
static int blocks, shared_bytes, sm_count, occupancy;
static std::string error, result;
static std::vector<Sample> samples;

static int check(cudaError_t e) {
    if (e == cudaSuccess) return 0;
    error = cudaGetErrorString(e);
    return int(e);
}
extern "C" const char* blocker_error() { return error.c_str(); }
__device__ unsigned long long timer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__global__ void pressure(Sample* out, unsigned long long duration_ns) {
    extern __shared__ volatile unsigned char residency[];
    residency[0] = 1;
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    auto begin = timer();
    while (timer() - begin < duration_ns) __nanosleep(100000);
    out[blockIdx.x] = {sm, begin, timer()};
}
extern "C" int blocker_init(int k) {
    cudaDeviceProp prop{};
    if (check(cudaGetDeviceProperties(&prop, 0))) return 1;
    if (k < 1 || k >= prop.multiProcessorCount) { error = "K must be in [1, SM count)"; return 1; }
    blocks = k;
    sm_count = prop.multiProcessorCount;
    // > half the SM's shared memory: occupancy must independently confirm 1.
    shared_bytes = int(prop.sharedMemPerMultiprocessor / 2 + 1024);
    if (shared_bytes > prop.sharedMemPerBlockOptin) { error = "insufficient opt-in shared memory"; return 1; }
    if (check(cudaFuncSetAttribute(pressure, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes))) return 1;
    if (check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, pressure, 1, shared_bytes))) return 1;
    if (occupancy != 1) { error = "blocker occupancy is not one block per SM"; return 1; }
    if (check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) return 1;
    if (check(cudaMalloc(&device_samples, blocks * sizeof(Sample)))) return 1;
    samples.resize(blocks);
    return 0;
}
extern "C" const char* blocker_info() {
    result = "{\"sm_count\":" + std::to_string(sm_count) + ",\"shared_bytes\":" + std::to_string(shared_bytes)
      + ",\"max_blocks_per_sm\":" + std::to_string(occupancy) + "}";
    return result.c_str();
}
extern "C" const char* blocker_pulse(int microseconds) {
    if (microseconds < 100 || microseconds > 100000) { error = "pulse must be 100..100000 us"; return nullptr; }
    if (check(cudaSetDevice(0))) return nullptr;
    pressure<<<blocks, 1, shared_bytes, stream>>>(device_samples, (unsigned long long)microseconds * 1000);
    if (check(cudaGetLastError())) return nullptr;
    if (check(cudaMemcpyAsync(samples.data(), device_samples, blocks*sizeof(Sample), cudaMemcpyDeviceToHost, stream))) return nullptr;
    if (check(cudaStreamSynchronize(stream))) return nullptr;
    std::ostringstream s;
    s << "[";
    for (int i=0; i<blocks; ++i) {
        if (i) s << ",";
        s << "[" << samples[i].sm << "," << samples[i].begin << "," << samples[i].end << "]";
    }
    s << "]";
    result = s.str();
    return result.c_str();
}
extern "C" int blocker_close() {
    int a = check(cudaStreamDestroy(stream));
    int b = check(cudaFree(device_samples));
    return a ? a : b;
}
