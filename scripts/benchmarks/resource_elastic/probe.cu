// Bounded synthetic co-load probe. Explicit streams; no device-wide wait.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <vector>

struct Stamp { unsigned long long begin, end; unsigned sm; };
struct State { Stamp *stable, *burst; float *result, *memory; int chunks; };
constexpr int stable_blocks = 128;
constexpr int burst_blocks = 1024;
constexpr int memory_items = 32 * 1024 * 1024; // 128 MiB, separate from stable data
static thread_local const char *error = "";
static bool ok(cudaError_t e) {
    if (e == cudaSuccess) return true;
    error = cudaGetErrorString(e); return false;
}
__device__ unsigned long long timer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
template<bool memory> __global__ void work(Stamp *stamps, float *out,
                                         float *data, int iterations) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    unsigned long long begin = timer();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0.5f;
    if constexpr (memory) {
        volatile float *v = data;
        for (int r = 0; r < iterations; ++r)
            for (int i = idx; i < memory_items; i += gridDim.x * blockDim.x)
                v[i] = v[i] + 1.0f;
    } else {
        #pragma unroll 1
        for (int r = 0; r < iterations; ++r) x = fmaf(x, 0.99999f, 0.00001f);
        if (threadIdx.x == 0) out[blockIdx.x] = x;
    }
    __syncthreads();
    if (threadIdx.x == 0) stamps[blockIdx.x] = {begin, timer(), sm};
}
extern "C" const char *probe_error() { return error; }
extern "C" void *probe_create(int chunks) {
    State *s = new State{};
    s->chunks = chunks;
    if (!ok(cudaMalloc(&s->stable, stable_blocks * sizeof(Stamp))) ||
        !ok(cudaMalloc(&s->burst, burst_blocks * chunks * sizeof(Stamp))) ||
        !ok(cudaMalloc(&s->result, (stable_blocks + burst_blocks) * sizeof(float))) ||
        !ok(cudaMalloc(&s->memory, memory_items * sizeof(float)))) {
        cudaFree(s->stable); cudaFree(s->burst); cudaFree(s->result);
        cudaFree(s->memory); delete s; return nullptr;
    }
    if (!ok(cudaMemset(s->memory, 0, memory_items * sizeof(float)))) {
        cudaFree(s->stable); cudaFree(s->burst); cudaFree(s->result);
        cudaFree(s->memory); delete s; return nullptr;
    }
    return s;
}
extern "C" void probe_destroy(void *p) {
    State *s = static_cast<State *>(p);
    cudaFree(s->stable); cudaFree(s->burst); cudaFree(s->result);
    cudaFree(s->memory); delete s;
}
// kind: 0 = solo, 1 = compute burst, 2 = memory burst.
// metrics: stable host response ms, stable GPU envelope ms, burst GPU
// envelope ms, envelope overlap ms, stable result min/max, stable first-start
// offset from burst first-start ms, stable completion offset from burst
// first-start ms, fraction of stable blocks overlapping a burst block.
// SM lists include EVERY measured block; caller validates against pool probes.
extern "C" int probe_run(void *p, void *a, void *b, int kind,
                         double *metrics, unsigned *sms) {
    State *s = static_cast<State *>(p);
    auto stable = static_cast<cudaStream_t>(a);
    auto burst = static_cast<cudaStream_t>(b);
    if (kind == 1) for (int i=0; i<s->chunks; ++i)
        work<false><<<burst_blocks, 256, 0, burst>>>(s->burst + i*burst_blocks,
            s->result + stable_blocks, s->memory, 32768/s->chunks);
    if (kind == 2)
        work<true><<<burst_blocks, 256, 0, burst>>>(s->burst,
            s->result + stable_blocks, s->memory, 4);
    if (!ok(cudaGetLastError())) return 1;
    auto t0 = std::chrono::steady_clock::now();
    work<false><<<stable_blocks, 256, 0, stable>>>(s->stable, s->result,
        s->memory, 4096);
    if (!ok(cudaGetLastError()) || !ok(cudaStreamSynchronize(stable))) return 1;
    metrics[0] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (!ok(cudaStreamSynchronize(burst))) return 1;
    int nburst = kind == 1 ? burst_blocks*s->chunks : burst_blocks;
    std::vector<Stamp> x(stable_blocks), y(nburst);
    std::vector<float> values(stable_blocks);
    if (!ok(cudaMemcpy(x.data(), s->stable, x.size()*sizeof(Stamp), cudaMemcpyDeviceToHost)) ||
        !ok(cudaMemcpy(values.data(), s->result, values.size()*sizeof(float), cudaMemcpyDeviceToHost))) return 1;
    if (kind && !ok(cudaMemcpy(y.data(), s->burst, y.size()*sizeof(Stamp), cudaMemcpyDeviceToHost))) return 1;
    auto bounds = [](const std::vector<Stamp>& stamps) {
        unsigned long long lo = ~0ULL, hi = 0;
        for (const auto& v : stamps) { lo = std::min(lo, v.begin); hi = std::max(hi, v.end); }
        return std::make_pair(lo, hi);
    };
    auto sx = bounds(x);
    auto sy = kind ? bounds(y) : std::make_pair(0ULL, 0ULL);
    metrics[1] = (sx.second-sx.first)/1e6;
    metrics[2] = (sy.second-sy.first)/1e6;
    auto lo = std::max(sx.first, sy.first), hi = std::min(sx.second, sy.second);
    metrics[3] = hi > lo ? (hi-lo)/1e6 : 0;
    metrics[4] = *std::min_element(values.begin(), values.end());
    metrics[5] = *std::max_element(values.begin(), values.end());
    metrics[6] = kind ? (static_cast<double>(sx.first)-sy.first)/1e6 : 0;
    metrics[7] = kind ? (static_cast<double>(sx.second)-sy.first)/1e6 : 0;
    int overlapping = 0;
    if (kind) for (const auto& v : x) {
        for (const auto& w : y) {
            if (std::max(v.begin, w.begin) < std::min(v.end, w.end)) {
                ++overlapping; break;
            }
        }
    }
    metrics[8] = static_cast<double>(overlapping)/stable_blocks;
    for (int i=0; i<stable_blocks; ++i) sms[i] = x[i].sm;
    for (int i=0; i<burst_blocks*s->chunks; ++i)
        sms[stable_blocks+i] = kind && i<nburst ? y[i].sm : ~0U;
    return 0;
}
