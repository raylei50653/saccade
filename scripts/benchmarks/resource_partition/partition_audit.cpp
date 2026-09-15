// CUPTI activity tracer for #419 phase B. Records every GPU kernel, memcpy and
// memset with the CUPTI context ID it executed in, plus context records, so a
// reporter can prove which context (primary or a Green Context) ran each launch.
// Diagnostic only. Never calls CUDA from a CUPTI buffer callback.
//
// Line formats (space separated):
//   N <name_id> <kernel name>
//   K <start> <end> <ctx> <stream> <graph> <correlation> <name_id> <device>
//   M <start> <end> <ctx> <stream> <graph> <correlation> <copy_kind> <bytes>
//   S <start> <end> <ctx> <stream> <graph> <correlation> 0 <bytes>
//   C <ctx> <device> <null_stream> <parent_ctx> <is_green> <sm_count>
//   G <ctx> <parent_ctx> <device> <tpcs> <sm_count> [tpc mask words]
//   T <ctx> <stream> <flags> <correlation>
//   L <cupti_timestamp> <label>
//   A <attribute> <value>          D <dropped records>          E dropped=<total>
#include <cupti.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

static FILE *out = nullptr;
static std::mutex lock;
static std::unordered_map<std::string, uint32_t> names;
static std::string last_error;
static uint64_t dropped = 0;

static const char *err(CUptiResult r) {
    const char *s = nullptr;
    cuptiGetResultString(r, &s);
    return s ? s : "?";
}

#define CHECK(call) do { CUptiResult _r = (call); if (_r != CUPTI_SUCCESS) { last_error = std::string(#call) + ": " + err(_r); return 1; } } while (0)

static uint32_t name_id(const char *name) {
    auto it = names.find(name);
    if (it != names.end()) return it->second;
    uint32_t id = static_cast<uint32_t>(names.size());
    names.emplace(name, id);
    fprintf(out, "N %u %s\n", id, name);
    return id;
}

static void CUPTIAPI requested(uint8_t **buffer, size_t *size, size_t *max) {
    *size = 8 << 20;
    *buffer = static_cast<uint8_t *>(malloc(*size + 8));
    *max = 0;
}

static void CUPTIAPI completed(CUcontext, uint32_t, uint8_t *buffer, size_t size, size_t valid) {
    CUpti_Activity *rec = nullptr;
    std::lock_guard<std::mutex> g(lock);
    if (out) {
        while (cuptiActivityGetNextRecord(buffer, valid, &rec) == CUPTI_SUCCESS) {
            switch (rec->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                const auto *k = reinterpret_cast<const CUpti_ActivityKernel13 *>(rec);
                fprintf(out, "K %llu %llu %u %u %u %u %u %u\n",
                        (unsigned long long)k->start, (unsigned long long)k->end,
                        k->contextId, k->streamId, k->graphId, k->correlationId,
                        name_id(k->name ? k->name : "?"), k->deviceId);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                const auto *m = reinterpret_cast<const CUpti_ActivityMemcpy7 *>(rec);
                fprintf(out, "M %llu %llu %u %u %u %u %u %llu\n",
                        (unsigned long long)m->start, (unsigned long long)m->end,
                        m->contextId, m->streamId, m->graphId, m->correlationId,
                        (unsigned)m->copyKind, (unsigned long long)m->bytes);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                const auto *m = reinterpret_cast<const CUpti_ActivityMemset5 *>(rec);
                fprintf(out, "S %llu %llu %u %u %u %u 0 %llu\n",
                        (unsigned long long)m->start, (unsigned long long)m->end,
                        m->contextId, m->streamId, m->graphId, m->correlationId,
                        (unsigned long long)m->bytes);
                break;
            }
            case CUPTI_ACTIVITY_KIND_CONTEXT: {
                const auto *c = reinterpret_cast<const CUpti_ActivityContext4 *>(rec);
                fprintf(out, "C %u %u %u %u %u %u\n", c->contextId, c->deviceId,
                        (unsigned)c->nullStreamId, c->parentContextId,
                        (unsigned)c->isGreenContext, (unsigned)c->numMultiprocessors);
                break;
            }
            case CUPTI_ACTIVITY_KIND_GREEN_CONTEXT: {
                const auto *c = reinterpret_cast<const CUpti_ActivityGreenContext3 *>(rec);
                fprintf(out, "G %u %u %u %u %u", c->contextId, c->parentContextId,
                        c->deviceId, c->numTpcs, (unsigned)c->numMultiprocessors);
                for (unsigned i = 0; i < c->logicalTpcMaskSize && i < 32; ++i)
                    fprintf(out, " %u", c->logicalTpcMask[i]);
                fprintf(out, "\n");
                break;
            }
            case CUPTI_ACTIVITY_KIND_STREAM: {
                const auto *s = reinterpret_cast<const CUpti_ActivityStream *>(rec);
                fprintf(out, "T %u %u %u %u\n", s->contextId, s->streamId, (unsigned)s->flag, s->correlationId);
                break;
            }
            default:
                break;
            }
        }
        size_t nd = 0;
        if (cuptiActivityGetNumDroppedRecords(nullptr, 0, &nd) == CUPTI_SUCCESS && nd) {
            dropped += nd;
            fprintf(out, "D %zu\n", nd);
        }
        fflush(out);
    }
    (void)size;
    free(buffer);
}

extern "C" const char *audit_error() { return last_error.c_str(); }

extern "C" int audit_start(const char *path) {
    std::lock_guard<std::mutex> g(lock);
    out = fopen(path, "w");
    if (!out) { last_error = "cannot open output"; return 1; }
    size_t v = 32 << 20, vs = sizeof(size_t);
    CHECK(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &vs, &v));
    CHECK(cuptiActivityRegisterCallbacks(requested, completed));
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    { CUptiResult r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GREEN_CONTEXT); fprintf(out, "A green_context_kind %s\n", r == CUPTI_SUCCESS ? "enabled" : err(r)); }
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_STREAM));
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    return 0;
}

extern "C" int audit_flush() {
    CHECK(cuptiActivityFlushAll(1));
    return 0;
}

extern "C" int audit_mark(const char *label) {
    uint64_t ts = 0;
    CHECK(cuptiGetTimestamp(&ts));
    std::lock_guard<std::mutex> g(lock);
    if (out) { fprintf(out, "L %llu %s\n", (unsigned long long)ts, label); fflush(out); }
    return 0;
}

extern "C" uint64_t audit_timestamp() {
    uint64_t ts = 0;
    cuptiGetTimestamp(&ts);
    return ts;
}

extern "C" int audit_context_id(void *ctx, uint32_t *id) {
    CHECK(cuptiGetContextId(static_cast<CUcontext>(ctx), id));
    return 0;
}

extern "C" int audit_stream_id(void *ctx, void *stream, uint32_t *id) {
    CHECK(cuptiGetStreamIdEx(static_cast<CUcontext>(ctx), static_cast<CUstream>(stream), 0, id));
    return 0;
}

extern "C" int audit_stop() {
    CHECK(cuptiActivityFlushAll(1));
    CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_STREAM));
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_GREEN_CONTEXT);
    CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
    std::lock_guard<std::mutex> g(lock);
    if (out) { fprintf(out, "E dropped=%llu\n", (unsigned long long)dropped); fclose(out); out = nullptr; }
    return 0;
}
