// Model-bundle compatibility (#549 S2-2). CUDA-free values and comparisons;
// native build coordinates are supplied by the compiled entrypoint, never CLI.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "saccade_shipping/resolved_config.hpp"

namespace saccade::shipping {

struct RuntimeBuildCompatibility {
    std::string platform, cuda_runtime, tensorrt, libtorch;
};

struct CompatibilityTarget {
    std::string platform, cuda_runtime, tensorrt, libtorch;
    std::vector<std::string> gpu_sm;
};

struct TensorContract {
    std::int64_t ordinal = 0;
    std::string name;
    std::vector<std::int64_t> shape;
    std::string dtype, device;
};

struct TensorIoContract {
    std::vector<TensorContract> inputs, outputs;
};

// After Gate A, targets contains only whole targets whose platform and all
// three build versions matched. Gate B must not combine fields from targets.
struct ManifestLoadContract {
    std::vector<CompatibilityTarget> targets;
    TensorIoContract backbone, head;
};

// A catchable Gate B hash / loader / compatibility / load-check refusal.
class DetectorLoadError : public ConfigError {
public:
    using ConfigError::ConfigError;
};

// Gate A: refuses unavailable build coordinates and a manifest without one
// whole matching target. The returned contract is the immutable handoff.
ManifestLoadContract select_manifest_build_targets(const ManifestLoadContract& manifest,
                                                  const RuntimeBuildCompatibility& build);

// Gate B: called before TensorRT engine deserialization; no CUDA calls here.
void require_manifest_gpu_sm(const ManifestLoadContract& contract, const std::string& actual_sm);

// Exact ordered I/O comparison. Observed TRT shapes must already be concrete
// context readbacks after binding the manifest's input shape; -1 is refused.
void require_tensor_contract(const TensorContract& expected, const TensorContract& observed,
                             const std::string& what, bool compare_name = true);
void require_tensor_io_contract(const TensorIoContract& expected, const TensorIoContract& observed,
                                const std::string& what, bool compare_names = true);

}  // namespace saccade::shipping
