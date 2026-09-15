// Reuse the independently bounded kernel and mapped start publication.
#include "admission.cu"
struct ElasticState {
    Stamp *stamps=nullptr; float *values=nullptr;
    Signal *host=nullptr, *device=nullptr;
    cudaEvent_t events[260]{};
};
extern "C" void budget_destroy(void *ptr) {
    auto s=static_cast<ElasticState*>(ptr); if (!s) return;
    for (auto e:s->events) if(e) cudaEventDestroy(e);
    if(s->stamps) cudaFree(s->stamps); if(s->values) cudaFree(s->values);
    if(s->host) cudaFreeHost(s->host); delete s;
}
extern "C" void *budget_create() {
    auto s=new ElasticState{};
    try {
        check(cudaMalloc(&s->stamps,(4*128+256*1024)*sizeof(Stamp)));
        check(cudaMalloc(&s->values,(4*128+256*1024)*sizeof(float)));
        check(cudaHostAlloc(&s->host,260*sizeof(Signal),cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&s->device,s->host,0));
        for(auto &e:s->events) check(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
        return s;
    } catch(const std::exception &e) {error=e.what(); budget_destroy(s); return nullptr;}
}
// ch: enqueue begin/end, retirement observation, frozen estimate, ledger after admission.
// arrivals: target, submit begin/end, stable done, drain observed, admitted,
// retired at freeze, ledger at freeze, resume. host: origin, all done, start observed.
extern "C" int budget_run(void *ptr, void *sp, void *bp, int n, int m,
    int window, U budget, U period, int *iterations, U *estimates,
    U *host, U *ch, U *arrivals, U *raw, float *values) {
    auto s=static_cast<ElasticState*>(ptr);
    auto stable=static_cast<cudaStream_t>(sp), burst=static_cast<cudaStream_t>(bp);
    try {
        if(n<1 || n>256 || m<0 || m>4 || window<0 || (window==0 && budget==0))
            throw std::runtime_error("invalid controller arguments");
        for(int i=0;i<n;++i) if(iterations[i]<1 || estimates[i]==0 || (!window && estimates[i]>budget))
            throw std::runtime_error("unit exceeds budget or invalid estimate");
        for(int i=0;i<260;++i) Atomic(s->host[i].value).store(0,cuda::memory_order_relaxed);
        std::fill(ch,ch+5*n,0); std::fill(arrivals,arrivals+9*m,0);
        int admitted=0, retired=0, arrival=0; U ledger=0;
        host[0]=now(); host[2]=0;
        auto timeout=[&] {if(now()-host[0]>10000000000ULL) throw std::runtime_error("10s timeout");};
        auto retire=[&] {
            while(retired<admitted) {
                auto e=cudaEventQuery(s->events[retired]); if(e==cudaErrorNotReady) break;
                check(e); ch[5*retired+2]=now(); ledger-=estimates[retired]; ++retired;
            }
        };
        auto launch=[&] {
            int i=admitted; ch[5*i]=now();
            work<<<1024,256,0,burst>>>(s->stamps+m*128+i*1024,s->values+m*128+i*1024,s->device+i,iterations[i]);
            check(cudaGetLastError()); check(cudaEventRecord(s->events[i],burst));
            ch[5*i+1]=now(); ledger+=estimates[i]; ch[5*i+3]=estimates[i]; ch[5*i+4]=ledger; ++admitted;
        };
        launch();
        while(!Atomic(s->host[0].value).load(cuda::memory_order_acquire)) timeout();
        host[2]=now();
        while(retired<n || arrival<m) {
            retire();
            if(arrival<m && now()>=host[2]+period*(arrival+1)) {
                U *a=arrivals+9*arrival;
                a[0]=host[2]+period*(arrival+1); a[5]=admitted; a[6]=retired; a[7]=ledger;
                a[1]=now();
                work<<<128,256,0,stable>>>(s->stamps+arrival*128,s->values+arrival*128,s->device+256+arrival,4096);
                check(cudaGetLastError()); check(cudaEventRecord(s->events[256+arrival],stable)); a[2]=now();
                while(!a[3] || !a[4]) {
                    if(!a[3]) {auto e=cudaEventQuery(s->events[256+arrival]); if(e!=cudaErrorNotReady) {check(e); a[3]=now();}}
                    retire(); if(!a[4] && retired==admitted) a[4]=now(); timeout();
                }
                a[8]=now(); ++arrival;
            } else if(admitted<n && (window ? admitted-retired<window : ledger+estimates[admitted]<=budget)) launch();
            timeout();
        }
        host[1]=now();
        check(cudaMemcpy(raw,s->stamps,(m*128+n*1024)*sizeof(Stamp),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(values,s->values,(m*128+n*1024)*sizeof(float),cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception &e) {error=e.what(); return 1;}
}
