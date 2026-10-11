#include "saccade_shipping/native_runtime_build.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <torch/version.h>

#include <string>

#ifndef SACCADE_LIBTORCH_BUILD_VERSION
#error "native build must bind the LibTorch package version, including its CUDA qualifier"
#endif

namespace saccade::shipping {

RuntimeBuildCompatibility native_runtime_build() {
    RuntimeBuildCompatibility build;
#if defined(__linux__) && defined(__x86_64__)
    build.platform = "linux-x86_64";
#endif
    build.cuda_runtime = std::to_string(CUDART_VERSION / 1000) + "." +
                         std::to_string((CUDART_VERSION % 1000) / 10);
    // The pinned OSS SDK headers can differ from the runtime wheel's patch
    // and build. Read the library actually linked by this native process;
    // these version APIs neither create a runtime nor initialize a device.
    const int trt_major = getInferLibMajorVersion();
    const int trt_minor = getInferLibMinorVersion();
    const int trt_patch = getInferLibPatchVersion();
    const int trt_build = getInferLibBuildVersion();
    if (trt_major > 0 && trt_minor >= 0 && trt_patch >= 0 && trt_build >= 0) {
        build.tensorrt = std::to_string(trt_major) + "." + std::to_string(trt_minor) + "." +
                         std::to_string(trt_patch) + "." + std::to_string(trt_build);
    }
    const std::string package = SACCADE_LIBTORCH_BUILD_VERSION;
    const std::string headers = TORCH_VERSION;
    // A package/header disagreement is unknown, so manifest Gate A refuses
    // it. The build qualifier is retained verbatim, never inferred from the
    // CUDA runtime headers (Torch and cudart have distinct build origins).
    if (package == headers || package.rfind(headers + "+", 0) == 0) build.libtorch = package;
    return build;
}

}  // namespace saccade::shipping
