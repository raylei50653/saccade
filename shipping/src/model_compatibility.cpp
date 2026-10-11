#include "saccade_shipping/model_compatibility.hpp"

#include <algorithm>
#include <sstream>

namespace saccade::shipping {
namespace {

std::string versions(const RuntimeBuildCompatibility& b) {
    return "platform=" + b.platform + ", CUDA=" + b.cuda_runtime + ", TensorRT=" + b.tensorrt +
           ", LibTorch=" + b.libtorch;
}

bool known(const std::string& value) {
    return !value.empty() && value != "unknown" && value != "unresolved";
}

std::string shape(const std::vector<std::int64_t>& dims) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < dims.size(); ++i) out << (i ? "," : "") << dims[i];
    out << ']';
    return out.str();
}

[[noreturn]] void mismatch(const std::string& what, const std::string& field,
                           const std::string& expected, const std::string& observed) {
    throw DetectorLoadError(what + " " + field + " mismatch: expected " + expected + ", observed " + observed);
}

}  // namespace

ManifestLoadContract select_manifest_build_targets(const ManifestLoadContract& manifest,
                                                  const RuntimeBuildCompatibility& build) {
    if (!known(build.platform) || !known(build.cuda_runtime) || !known(build.tensorrt) || !known(build.libtorch)) {
        throw ConfigError("model bundle compatibility: runtime build versions unavailable (" + versions(build) +
                          "); compatibility cannot be proven");
    }
    ManifestLoadContract matched = manifest;
    matched.targets.clear();
    for (const auto& t : manifest.targets) {
        if (t.platform == build.platform && t.cuda_runtime == build.cuda_runtime && t.tensorrt == build.tensorrt &&
            t.libtorch == build.libtorch) {
            matched.targets.push_back(t);
        }
    }
    if (matched.targets.empty()) {
        std::string declared;
        for (const auto& t : manifest.targets) {
            declared += (declared.empty() ? "" : "; ") + versions({t.platform, t.cuda_runtime, t.tensorrt, t.libtorch});
        }
        throw ConfigError("model bundle compatibility: no manifest target matches runtime build (" + versions(build) +
                          "); declared targets: " + declared);
    }
    return matched;
}

void require_manifest_gpu_sm(const ManifestLoadContract& contract, const std::string& actual_sm) {
    std::string allowed;
    for (const auto& t : contract.targets) {
        for (const auto& sm : t.gpu_sm) {
            if (!actual_sm.empty() && actual_sm == sm) return;
            allowed += (allowed.empty() ? "" : ",") + sm;
        }
    }
    throw DetectorLoadError("model bundle compatibility: GPU SM " +
                            (actual_sm.empty() ? std::string("unavailable") : actual_sm) +
                            " is unsupported by the version-matched targets (" + allowed +
                            "); refused before engine deserialization");
}

void require_tensor_contract(const TensorContract& expected, const TensorContract& observed,
                             const std::string& what, bool compare_name) {
    if (expected.ordinal != observed.ordinal)
        mismatch(what, "ordinal/order", std::to_string(expected.ordinal), std::to_string(observed.ordinal));
    if (compare_name && expected.name != observed.name) mismatch(what, "name/order", expected.name, observed.name);
    if (expected.dtype != observed.dtype) mismatch(what, "dtype", expected.dtype, observed.dtype);
    if (expected.device != observed.device) mismatch(what, "device", expected.device, observed.device);
    if (expected.shape != observed.shape) mismatch(what, "shape", shape(expected.shape), shape(observed.shape));
    if (std::any_of(observed.shape.begin(), observed.shape.end(), [](auto d) { return d <= 0; }))
        throw DetectorLoadError(what + " shape is not a proven concrete positive shape: " + shape(observed.shape));
}

void require_tensor_io_contract(const TensorIoContract& expected, const TensorIoContract& observed,
                                const std::string& what, bool compare_names) {
    auto require_list = [&](const auto& wanted, const auto& got, const char* direction) {
        const std::string label = what + " " + direction;
        if (wanted.size() != got.size())
            mismatch(label, "count", std::to_string(wanted.size()), std::to_string(got.size()));
        for (std::size_t i = 0; i < wanted.size(); ++i) {
            if (wanted[i].ordinal != static_cast<std::int64_t>(i))
                throw DetectorLoadError(label + " manifest ordinal/order is not contiguous at " + std::to_string(i));
            require_tensor_contract(wanted[i], got[i], label + "[" + std::to_string(i) + "]", compare_names);
        }
    };
    require_list(expected.inputs, observed.inputs, "inputs");
    require_list(expected.outputs, observed.outputs, "outputs");
}

}  // namespace saccade::shipping
