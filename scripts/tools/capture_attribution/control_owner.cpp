// Synthetic-only native callers for validating stream-creation attribution.
// Keeping these calls in a named, debug-built module gives the observer an
// independently checkable module and symbol immediately above each CUDA API.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static void retain_frame(void *stream) {
#if defined(__GNUC__)
    asm volatile("" : : "r"(stream) : "memory");
#else
    (void)stream;
#endif
}

extern "C" NOINLINE int attribution_control_cuda_create_default(uintptr_t *out) {
    cudaStream_t stream = nullptr;
    const auto rc = cudaStreamCreate(&stream);
    if (rc == cudaSuccess) *out = reinterpret_cast<uintptr_t>(stream);
    retain_frame(stream);
    return static_cast<int>(rc);
}

extern "C" NOINLINE int attribution_control_cuda_create_flags(
    uintptr_t *out, unsigned int flags) {
    cudaStream_t stream = nullptr;
    const auto rc = cudaStreamCreateWithFlags(&stream, flags);
    if (rc == cudaSuccess) *out = reinterpret_cast<uintptr_t>(stream);
    retain_frame(stream);
    return static_cast<int>(rc);
}

extern "C" NOINLINE int attribution_control_cuda_create_priority(
    uintptr_t *out, unsigned int flags, int priority) {
    cudaStream_t stream = nullptr;
    const auto rc = cudaStreamCreateWithPriority(&stream, flags, priority);
    if (rc == cudaSuccess) *out = reinterpret_cast<uintptr_t>(stream);
    retain_frame(stream);
    return static_cast<int>(rc);
}

extern "C" NOINLINE int attribution_control_cuda_destroy(uintptr_t raw) {
    return static_cast<int>(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(raw)));
}

extern "C" NOINLINE int attribution_control_cu_create_flags(
    uintptr_t *out, unsigned int flags) {
    CUstream stream = nullptr;
    const auto rc = cuStreamCreate(&stream, flags);
    if (rc == CUDA_SUCCESS) *out = reinterpret_cast<uintptr_t>(stream);
    retain_frame(stream);
    return static_cast<int>(rc);
}

extern "C" NOINLINE int attribution_control_cu_create_priority(
    uintptr_t *out, unsigned int flags, int priority) {
    CUstream stream = nullptr;
    const auto rc = cuStreamCreateWithPriority(&stream, flags, priority);
    if (rc == CUDA_SUCCESS) *out = reinterpret_cast<uintptr_t>(stream);
    retain_frame(stream);
    return static_cast<int>(rc);
}

extern "C" NOINLINE int attribution_control_cu_destroy(uintptr_t raw) {
    return static_cast<int>(cuStreamDestroy(reinterpret_cast<CUstream>(raw)));
}
