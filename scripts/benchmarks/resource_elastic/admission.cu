// Synthetic arrival/admission probe. Host polling is part of the measured policy.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

using U = unsigned long long;
using Clock = std::chrono::steady_clock;
using Atomic = cuda::atomic_ref<U, cuda::thread_scope_system>;
struct Stamp { U begin, end, sm; };
struct alignas(64) Signal { U value; };
struct State {
    Stamp *stamps = nullptr;
    float *values = nullptr;
    Signal *host = nullptr, *device = nullptr;
    cudaEvent_t events[17]{};
};
static thread_local std::string error;
static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static U now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
__device__ U timer() { U t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void work(Stamp *stamps, float *out, Signal *signal, int iterations) {
    unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    U begin = timer();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        Atomic(signal->value).store(begin, cuda::memory_order_release);
    float x = 0.5f;
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) x = fmaf(x, 0.99999f, 0.00001f);
    __syncthreads();
    if (threadIdx.x == 0) { out[blockIdx.x] = x; stamps[blockIdx.x] = {begin, timer(), sm}; }
}
extern "C" const char *admission_error() { return error.c_str(); }
extern "C" void admission_destroy(void *ptr) {
    auto s = static_cast<State*>(ptr); if (!s) return;
    for (auto e : s->events) if (e) cudaEventDestroy(e);
    if (s->stamps) cudaFree(s->stamps);
    if (s->values) cudaFree(s->values);
    if (s->host) cudaFreeHost(s->host);
    delete s;
}
extern "C" void *admission_create() {
    auto s = new State{};
    try {
        check(cudaMalloc(&s->stamps, (128 + 16*1024)*sizeof(Stamp)));
        check(cudaMalloc(&s->values, (128 + 16*1024)*sizeof(float)));
        check(cudaHostAlloc(&s->host, 17*sizeof(Signal), cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&s->device, s->host, 0));
        for (auto &e : s->events) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        return s;
    } catch (const std::exception &e) { error = e.what(); admission_destroy(s); return nullptr; }
}
// Host fields (ns): origin, burst-start observation, target arrival, stable
// enqueue begin/end, stable block0 last-zero/first-nonzero polls, stable done,
// admitted drain observed, all work done; then admitted/completed at arrival,
// max unretired, stable block0 GPU signal, first burst block0 GPU signal.
// Per chunk host fields: enqueue begin/end, completion observed (ns).
extern "C" int admission_run(void *ptr, void *stable_ptr, void *burst_ptr,
    int chunks, int window, U delay_ns, U *host, U *chunk_host, U *raw, float *values) {
    auto s = static_cast<State*>(ptr);
    auto stable = static_cast<cudaStream_t>(stable_ptr), burst = static_cast<cudaStream_t>(burst_ptr);
    try {
        if (chunks < 1 || chunks > 16 || 32768%chunks || window < 1 || window > chunks)
            throw std::runtime_error("invalid chunk/window");
        for (int i=0; i<17; ++i) Atomic(s->host[i].value).store(0, cuda::memory_order_relaxed);
        std::fill(host, host+15, 0); std::fill(chunk_host, chunk_host+48, 0);
        int admitted=0, completed=0, max_unretired=0;
        host[0]=now();
        auto timeout = [&] { if (now()-host[0] > 5000000000ULL) throw std::runtime_error("5s probe timeout"); };
        auto retire = [&] {
            while (completed < admitted) {
                auto e = cudaEventQuery(s->events[completed]);
                if (e == cudaErrorNotReady) break;
                check(e); chunk_host[3*completed+2]=now(); ++completed;
            }
        };
        auto launch = [&] {
            int i=admitted;
            chunk_host[3*i]=now();
            work<<<1024,256,0,burst>>>(s->stamps+128+i*1024, s->values+128+i*1024,
                                     s->device+i, 32768/chunks);
            check(cudaGetLastError());
            check(cudaEventRecord(s->events[i], burst));
            chunk_host[3*i+1]=now(); ++admitted;
            max_unretired=std::max(max_unretired,admitted-completed);
        };
        // Fill the initial window, then require an in-kernel start signal.
        while (admitted < window) launch();
        U burst_signal=0;
        while (!(burst_signal=Atomic(s->host[0].value).load(cuda::memory_order_acquire))) timeout();
        host[1]=now(); host[2]=host[1]+delay_ns;
        // Rolling admission before arrival. One dispatch per iteration limits
        // host overshoot. GPU-complete events, not signals, release queue slots.
        while (now() < host[2]) {
            retire();
            if (admitted < chunks && admitted-completed < window) launch();
            timeout();
        }
        retire(); host[10]=admitted; host[11]=completed;
        host[3]=now(); host[5]=host[3];
        work<<<128,256,0,stable>>>(s->stamps, s->values, s->device+16, 4096);
        check(cudaGetLastError()); check(cudaEventRecord(s->events[16],stable));
        host[4]=now();
        // Freeze admission at arrival, observe stable completion and drain of
        // the already admitted prefix independently, then resume remaining work.
        while (!host[7] || !host[8]) {
            U before=now();
            U sig=Atomic(s->host[16].value).load(cuda::memory_order_acquire);
            if (!host[6]) {
                if (sig) { host[6]=now(); host[13]=sig; }
                else host[5]=before;
            }
            if (!host[7]) {
                auto e=cudaEventQuery(s->events[16]);
                if (e != cudaErrorNotReady) { check(e); host[7]=now(); }
            }
            retire(); if (!host[8] && completed == admitted) host[8]=now();
            timeout();
        }
        while (completed < chunks) {
            retire();
            if (admitted < chunks && admitted-completed < window) launch();
            timeout();
        }
        host[9]=now(); host[12]=max_unretired; host[14]=burst_signal;
        check(cudaMemcpy(raw,s->stamps,(128+chunks*1024)*sizeof(Stamp),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(values,s->values,(128+chunks*1024)*sizeof(float),cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception &e) { error=e.what(); return 1; }
}
