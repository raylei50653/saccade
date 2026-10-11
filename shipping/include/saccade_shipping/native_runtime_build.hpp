// Build coordinates supplied by the native entrypoint to CUDA-free Gate A.
// No CLI/environment override and no device API call.
#pragma once

#include "saccade_shipping/model_compatibility.hpp"

namespace saccade::shipping {

RuntimeBuildCompatibility native_runtime_build();

}  // namespace saccade::shipping
