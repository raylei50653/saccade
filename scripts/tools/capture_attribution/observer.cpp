// Diagnostic observer only. Never call CUDA, synchronize, or clear CUDA errors
// from a callback. Flags come from observed successful creates/GetFlags calls.
#include <cupti.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <execinfo.h>
#include <map>
#include <mutex>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static FILE *trace_file = nullptr;
static CUpti_SubscriberHandle subscriber;
static std::mutex output_lock;
static thread_local uint64_t site_id = 0;
static uint64_t sequence = 0;
static std::map<std::pair<uintptr_t, uintptr_t>, std::pair<int, const char *>> flags;

struct Decoded {
    bool selected = false, has_stream = false, has_capture_id = false;
    uintptr_t stream = 0, event = 0;
    uint64_t capture_id = 0;
    int mode = -1, flags = -1, status = -1, event_flags = -1;
    const char *flag_source = "unknown";
};

static void CUPTIAPI callback(void *, CUpti_CallbackDomain domain,
                             CUpti_CallbackId id, const void *raw) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const auto *data = static_cast<const CUpti_CallbackData *>(raw);
    const bool entering = data->callbackSite == CUPTI_API_ENTER;
    const int rc = !entering && data->functionReturnValue ?
        *static_cast<const int *>(data->functionReturnValue) : -1;
    const bool ok = !entering && rc == 0;
    Decoded e;
    const char *name = data->functionName ? data->functionName : "unknown";
    // Retain even obsolete/new stream API variants without typed metadata.
    // They have has_stream=false and cannot pass the attribution coverage gate.
    e.selected = strstr(name, "EventRecord") || strstr(name, "EventDestroy") || (strstr(name, "Stream") && (
        strstr(name, "Capture") || strstr(name, "Capturing") ||
        strstr(name, "Create") || strstr(name, "Destroy") ||
        strstr(name, "GetFlags") || strstr(name, "WaitEvent") ||
        strstr(name, "Synchronize")));
#include "decode.inc"
    // Every nonzero API return is retained, including errors outside stream APIs.
    if (!e.selected && (entering || rc == 0)) return;
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    const auto context = reinterpret_cast<uintptr_t>(data->context);
    std::lock_guard<std::mutex> lock(output_lock);
    if (!trace_file) return;
    const auto key = std::make_pair(context, e.stream);
    if (e.has_stream) {
        if (e.flags >= 0) flags[key] = {e.flags, e.flag_source};
        else if (const auto it = flags.find(key); it != flags.end()) {
            e.flags = it->second.first;
            e.flag_source = it->second.second;
        }
    }
    fprintf(trace_file,
        "{\"seq\":%llu,\"ns\":%llu,\"pid\":%d,\"tid\":%ld,\"domain\":%u,"
        "\"cbid\":%u,\"correlation\":%u,\"phase\":\"%s\",\"api\":\"%s\","
        "\"context\":%llu,\"context_uid\":%u,\"site_id\":%llu,\"selected\":%s,\"has_stream\":%s,"
        "\"stream\":%llu,\"flags\":%d,\"flags_source\":\"%s\",\"mode\":%d,"
        "\"status\":%d,\"event\":%llu,\"event_flags\":%d,"
        "\"has_capture_id\":%s,\"capture_id\":%llu,\"rc\":%d,\"native_stack\":[",
        static_cast<unsigned long long>(++sequence),
        static_cast<unsigned long long>(now.tv_sec) * 1000000000ULL + now.tv_nsec,
        getpid(), syscall(SYS_gettid), domain, id, data->correlationId,
        entering ? "enter" : "exit", name, static_cast<unsigned long long>(context),
        data->contextUid, static_cast<unsigned long long>(site_id),
        e.selected ? "true" : "false",
        e.has_stream ? "true" : "false", static_cast<unsigned long long>(e.stream),
        e.flags, e.flag_source, e.mode, e.status,
        static_cast<unsigned long long>(e.event), e.event_flags,
        e.has_capture_id ? "true" : "false", static_cast<unsigned long long>(e.capture_id), rc);
    if ((entering && (strstr(name, "Begin") || strstr(name, "StreamCreate") ||
         (site_id && (strstr(name, "EventRecord") || strstr(name, "StreamWaitEvent"))))) ||
        (!entering && rc != 0)) {
        void *stack[32];
        const int count = backtrace(stack, 32);
        for (int i = 0; i < count; ++i)
            fprintf(trace_file, "%s%llu", i ? "," : "",
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stack[i])));
    }
    fprintf(trace_file, "]}\n");
    if (fflush(trace_file) != 0 || ferror(trace_file)) _exit(74);
    if (ok && e.has_stream && strstr(name, "StreamDestroy")) flags.erase(key);
}

extern "C" int attribution_start(const char *path) {
    if (trace_file) return -2;
    trace_file = fopen(path, "wx");
    if (!trace_file) return -1;
    CUptiResult rc = cuptiSubscribe(&subscriber, callback, nullptr);
    if (rc == CUPTI_SUCCESS) rc = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    if (rc == CUPTI_SUCCESS) rc = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    return static_cast<int>(rc);
}

extern "C" void attribution_site(uint64_t id) { site_id = id; }

extern "C" int attribution_stop() {
    const auto rc = cuptiUnsubscribe(subscriber);
    std::lock_guard<std::mutex> lock(output_lock);
    if (trace_file) { fclose(trace_file); trace_file = nullptr; }
    return static_cast<int>(rc);
}
