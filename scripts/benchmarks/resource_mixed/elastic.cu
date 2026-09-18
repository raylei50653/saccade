// Diagnostic bounded elastic queues; no production runtime changes.
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using U = unsigned long long;
static U now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Stamp { U begin, end; unsigned sm; float value; };
struct Record { U burst, lane, enqueue, observed; };
__global__ void mixed_elastic(Stamp *out, int iterations) {
    U begin, end; unsigned sm;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    float value=1.f+threadIdx.x*0.001f;
    for(int i=0;i<iterations;++i) value=__fmaf_rn(value,1.000001f,0.000001f);
    // All threads contribute observable work; leader records a reproducible sum.
    __shared__ float sums[256]; sums[threadIdx.x]=value; __syncthreads();
    for(int d=128;d;d/=2) { if(threadIdx.x<d) sums[threadIdx.x]+=sums[threadIdx.x+d]; __syncthreads(); }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
    if(threadIdx.x==0) out[blockIdx.x]={begin,end,sm,sums[0]};
}
// Up to MAX_LANES admission lanes; `borrowed` names the lane that is frozen
// while the stable owner is busy (-1: no borrowed lane). Lane 0's context owns
// the device evidence buffer.
static const int MAX_LANES=4, MAX_WINDOW=4;
struct State {
    cudaStream_t streams[MAX_LANES]; CUcontext contexts[MAX_LANES]; int lanes, borrowed;
    int window, iterations, bursts, units; U period;
    std::atomic<bool> busy{true}, done{false}; std::mutex mutex;
    std::string error; Stamp *device=nullptr; std::vector<Stamp> stamps; std::vector<Record> records;
    U origin=0, end=0; cudaEvent_t events[MAX_LANES*MAX_WINDOW]{}; int active[MAX_LANES*MAX_WINDOW];
};
static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void context(CUcontext c) { if(cuCtxSetCurrent(c)!=CUDA_SUCCESS) throw std::runtime_error("set context failed"); }
extern "C" void *mixed_create_lanes(void **streams, void **contexts, int lanes, int borrowed, int window, int iterations, int bursts, int units, U period) {
    auto s=new State; s->lanes=lanes; s->borrowed=borrowed;
    s->window=window; s->iterations=iterations; s->bursts=bursts; s->units=units; s->period=period;
    try {
        if(lanes<1 || lanes>MAX_LANES || borrowed<-1 || borrowed>=lanes) throw std::runtime_error("invalid lanes");
        for(int i=0;i<lanes;++i) { s->streams[i]=(cudaStream_t)streams[i]; s->contexts[i]=(CUcontext)contexts[i]; }
        if(window<1 || window>MAX_WINDOW || iterations<1 || bursts<1 || units<1 || (long long)bursts*units>100000) throw std::runtime_error("invalid workload");
        s->stamps.resize((size_t)bursts*units*256); s->records.reserve(bursts*units);
        context(s->contexts[0]); check(cudaMalloc(&s->device,s->stamps.size()*sizeof(Stamp)));
        for(int i=0;i<lanes*window;++i) { context(s->contexts[i/window]); check(cudaEventCreateWithFlags(&s->events[i],cudaEventDisableTiming)); s->active[i]=-1; }
    } catch(const std::exception &e) { s->error=e.what(); }
    return s;
}
// Two-lane entry used by the serial mixed harness: lane 1 is borrowed when dynamic.
extern "C" void *mixed_create(void *s0, void *s1, void *c0, void *c1, int window, int iterations, int bursts, int units, U period, int dynamic) {
    void *streams[2]={s0,s1}, *contexts[2]={c0,c1};
    return mixed_create_lanes(streams,contexts,2,dynamic?1:-1,window,iterations,bursts,units,period);
}
extern "C" const char *mixed_error(void *p) { return ((State*)p)->error.c_str(); }
extern "C" U mixed_now() { return now(); }
extern "C" int mixed_busy(void *p, int busy) {
    auto s=(State*)p;
    try {
        {std::lock_guard<std::mutex> lock(s->mutex); s->busy=busy;}
        if(busy && s->borrowed>=0) {context(s->contexts[s->borrowed]); check(cudaStreamSynchronize(s->streams[s->borrowed]));}
        return 0;
    } catch(const std::exception &e) { s->error=e.what(); return 1; }
}
extern "C" int mixed_run(void *p, U origin) {
    auto s=(State*)p; s->origin=origin;
    try {
        int submitted=0, completed=0, total=s->bursts*s->units;
        while(completed<total) {
            if(now()-origin>120000000000ULL) throw std::runtime_error("elastic 120s timeout");
            U elapsed=now()>=origin?now()-origin:0;
            int released=now()>=origin?std::min(s->bursts,(int)(elapsed/s->period)+1)*s->units:0;
            for(int slot=0;slot<s->lanes*s->window;++slot) {
                int lane=slot/s->window; context(s->contexts[lane]);
                if(s->active[slot]>=0) {
                    auto e=cudaEventQuery(s->events[slot]);
                    if(e!=cudaErrorNotReady) {check(e); s->records[s->active[slot]].observed=now(); s->active[slot]=-1; ++completed;}
                }
                if(s->active[slot]<0 && submitted<released) {
                    std::lock_guard<std::mutex> lock(s->mutex);
                    if(lane==s->borrowed && s->busy) continue;
                    s->records.push_back({(U)(submitted/s->units),(U)lane,now(),0});
                    mixed_elastic<<<256,256,0,s->streams[lane]>>>(s->device+(size_t)submitted*256,s->iterations);
                    check(cudaGetLastError()); check(cudaEventRecord(s->events[slot],s->streams[lane]));
                    s->active[slot]=submitted++;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        s->end=now(); context(s->contexts[0]);
        s->done=true; return 0;
    } catch(const std::exception &e) { s->error=e.what(); s->done=true; return 1; }
}
extern "C" int mixed_save(void *p, const char *records_path, const char *stamps_path) {
    auto s=(State*)p;
    // Evidence transfer is deliberately after all stable frames, never at
    // elastic completion inside the measured stable-service horizon.
    try { context(s->contexts[0]); check(cudaMemcpy(s->stamps.data(),s->device,s->stamps.size()*sizeof(Stamp),cudaMemcpyDeviceToHost)); }
    catch(const std::exception &e) { s->error=e.what(); return 1; }
    FILE *f=fopen(records_path,"wb"); if(!f) return 1;
    fwrite(s->records.data(),sizeof(Record),s->records.size(),f); fclose(f);
    f=fopen(stamps_path,"wb"); if(!f) return 1;
    fwrite(s->stamps.data(),sizeof(Stamp),s->stamps.size(),f); fclose(f); return 0;
}
extern "C" float mixed_reference(int iterations) {
    float values[256]; for(int j=0;j<256;++j) { float v=1.f+j*0.001f; for(int i=0;i<iterations;++i) v=std::fma(v,1.000001f,0.000001f); values[j]=v; }
    for(int d=128;d;d/=2) for(int j=0;j<d;++j) values[j]+=values[j+d]; return values[0];
}
