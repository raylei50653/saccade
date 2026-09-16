// Reuse the audited synthetic work kernel; all routing is explicit per launch.
#include "admission.cu"
constexpr int MAX_WORK=4096, BLOCKS=256, ARRIVALS=16;
struct RoutingState {
    Stamp *stamps=nullptr; float *values=nullptr;
    Signal *host=nullptr, *device=nullptr;
    cudaEvent_t events[MAX_WORK]{};
};
extern "C" void routing_destroy(void *ptr) {
    auto s=static_cast<RoutingState*>(ptr); if(!s) return;
    for(auto e:s->events) if(e) cudaEventDestroy(e);
    if(s->stamps) cudaFree(s->stamps); if(s->values) cudaFree(s->values);
    if(s->host) cudaFreeHost(s->host); delete s;
}
extern "C" void *routing_create() {
    auto s=new RoutingState{};
    try {
        check(cudaMalloc(&s->stamps,MAX_WORK*BLOCKS*sizeof(Stamp)));
        check(cudaMalloc(&s->values,MAX_WORK*BLOCKS*sizeof(float)));
        check(cudaHostAlloc(&s->host,MAX_WORK*sizeof(Signal),cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&s->device,s->host,0));
        for(auto &e:s->events) check(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
        return s;
    } catch(const std::exception &e) {error=e.what(); routing_destroy(s); return nullptr;}
}
// policy: 0 fixed stable A+C; 1 static elastic B+C; 2 borrow C for all idle;
// 3 free headroom C, use on high demand; 4 dynamic C on high demand;
// 5 shared matched budget; 6 full-device shared. Shared policies use A/B only.
// Records: kind (0 stable/1 elastic), logical lane, job, enqueue begin/end,
// completion observation. Arrivals: target, dispatch, reclaim request,
// drain observed, first C enqueue, stable completion, C release.
extern "C" int routing_run(void *ptr, void *ap, void *bp, void *cp,
    int policy, int window, int iterations, U period, U *host, U *records,
    U *arrivals, U *raw, float *values) {
    auto s=static_cast<RoutingState*>(ptr);
    cudaStream_t streams[]={static_cast<cudaStream_t>(ap),static_cast<cudaStream_t>(bp),static_cast<cudaStream_t>(cp)};
    try {
        if(policy<0 || policy>6 || window<1 || window>4 || iterations<1 || period<1)
            throw std::runtime_error("invalid routing arguments");
        std::fill(records,records+MAX_WORK*6,0); std::fill(arrivals,arrivals+ARRIVALS*7,0);
        for(int i=0;i<MAX_WORK;++i) Atomic(s->host[i].value).store(0,cuda::memory_order_relaxed);
        int n=0, active=-1, finished=0, submitted=0, stable_done=0;
        int inflight[3]={0,0,0}; bool use_c=false, c_ready=false;
        host[0]=now(); host[1]=0;
        // Fixed wall-clock arrivals; low, low, high, high demand repeats.
        for(int j=0;j<ARRIVALS;++j) arrivals[j*7]=host[0]+period*(j+1);
        auto launch=[&](int kind,int lane,int job) {
            if(n==MAX_WORK) throw std::runtime_error("work record capacity exhausted");
            U *r=records+n*6; r[0]=kind; r[1]=lane; r[2]=job; r[3]=now();
            work<<<BLOCKS,256,0,streams[lane]>>>(s->stamps+n*BLOCKS,s->values+n*BLOCKS,s->device+n,kind?iterations:4096);
            check(cudaGetLastError()); check(cudaEventRecord(s->events[n],streams[lane]));
            r[4]=now(); ++inflight[lane]; ++n;
        };
        while(finished<ARRIVALS) {
            if(now()-host[0]>10000000000ULL) throw std::runtime_error("10s routing timeout");
            for(int i=0;i<n;++i) {
                U *r=records+i*6; if(r[5]) continue;
                auto e=cudaEventQuery(s->events[i]); if(e==cudaErrorNotReady) continue;
                check(e); r[5]=now(); --inflight[r[1]];
                if(r[0]==0) ++stable_done;
            }
            if(active>=0) {
                int units=(active%4<2)?1:8; U *a=arrivals+active*7;
                if(use_c && !c_ready && inflight[2]==0) {a[3]=now(); c_ready=true;}
                if(submitted<units && inflight[0]==0) {launch(0,0,active); ++submitted;}
                if(submitted<units && use_c && c_ready && inflight[2]==0) {
                    launch(0,2,active); if(!a[4]) a[4]=records[(n-1)*6+3]; ++submitted;
                }
                if(stable_done==units && !a[5]) a[5]=now();
                if(stable_done==units && (!use_c || c_ready)) {
                    if(use_c) a[6]=now(); ++finished; active=-1;
                }
            }
            if(active<0 && finished<ARRIVALS && now()>=arrivals[finished*7]) {
                active=finished; submitted=0; stable_done=0;
                bool high=(active%4>=2);
                use_c=(policy==0 || policy==2 || ((policy==3 || policy==4)&&high));
                c_ready=false; U *a=arrivals+active*7; a[1]=now();
                if(use_c) a[2]=now();
                continue;
            }
            if(finished==ARRIVALS) break;
            if(inflight[1]<window) launch(1,1,ARRIVALS);
            bool borrow=(policy==1 || policy==2 || policy==4);
            if(borrow && !(active>=0 && use_c) && inflight[2]<window) launch(1,2,ARRIVALS);
        }
        host[1]=now(); // stop new elastic admission; drain is included separately
        for(int i=0;i<n;++i) if(!records[i*6+5]) {
            check(cudaEventSynchronize(s->events[i])); records[i*6+5]=now();
        }
        host[2]=now(); host[3]=n;
        check(cudaMemcpy(raw,s->stamps,n*BLOCKS*sizeof(Stamp),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(values,s->values,n*BLOCKS*sizeof(float),cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception &e) {error=e.what(); return 1;}
}
