// Build the native tracking objects from the resolved shipping config
// (#465 Phase B PR-4b, U2b-b). Links saccade_tracking (CUDA).
//
// Each builder constructs the object from the JSON's constructor values,
// applies every remaining value explicitly (no native default is relied on,
// the process environment is never read), then reads the object's snapshot
// back and throws ConfigError unless it equals the resolved config
// (native_config.hpp). The returned object has not run an update yet, so its
// configuration is still mutable; the first update inside a CUDA graph
// capture freezes it.
#pragma once

#include <memory>

#include "saccade_shipping/native_config.hpp"
#include "saccade_shipping/resolved_config.hpp"

namespace saccade {
class GPUByteTracker;
class GMC;
class PerceptionPipeline;
}  // namespace saccade

namespace saccade::shipping {

// One tracker per sequence, as the oracle does (set_frame_size is per sequence).
std::unique_ptr<GPUByteTracker> build_tracker(const ResolvedShippingConfig& cfg,
                                              SequenceGeometry geometry);
std::unique_ptr<GMC> build_gmc(const ResolvedShippingConfig& cfg);
// Shipping has no ReID extractor and no cropper (the loader pins both
// constructor pointers to 0).
std::unique_ptr<PerceptionPipeline> build_perception_pipeline(const ResolvedShippingConfig& cfg);

}  // namespace saccade::shipping
