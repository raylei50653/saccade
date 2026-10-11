// CUDA-free comparison controls for #549 S2-2, MB-43..MB-46. Runtime
// coordinates and observed tensors here are explicit synthetic inputs;
// native loading and pre-deserialization ordering need the GPU test too.
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "saccade_shipping/model_bundle.hpp"
#include "saccade_shipping/model_compatibility.hpp"

namespace sh = saccade::shipping;

namespace {
int failures = 0, checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; \
    std::fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #condition); } } while (0)

template <class Error, class F>
void refused(F&& f, const char* reason) {
    try {
        f();
        CHECK(false);
    } catch (const Error& e) {
        CHECK(std::string(e.what()).find(reason) != std::string::npos);
    }
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    std::ostringstream bytes;
    bytes << input.rdbuf();
    const auto parsed = sh::parse_model_bundle(sh::parse_strict_json(bytes.str()));
    const auto& contract = parsed.load_contract;
    const auto& first = contract.targets.at(0);
    const sh::RuntimeBuildCompatibility build{first.platform, first.cuda_runtime, first.tensorrt, first.libtorch};
    const auto matched = sh::select_manifest_build_targets(contract, build);
    CHECK(matched.targets.size() == 1);
    CHECK(matched.backbone.inputs.at(0).name == "images");
    CHECK(matched.backbone.outputs.at(0).name == "p3_feat");
    CHECK(matched.head.outputs.at(3).name == "reg_p3");
    CHECK(matched.head.outputs.at(3).shape == std::vector<std::int64_t>({1, 4, 80, 80}));
    sh::require_manifest_gpu_sm(matched, first.gpu_sm.at(0));
    sh::require_tensor_io_contract(contract.backbone, contract.backbone, "engine");
    sh::require_tensor_io_contract(contract.head, contract.head, "head");

    // MB-43: every build coordinate needs a match in the same complete row.
    for (const auto change : {0, 1, 2, 3}) {
        auto bad = build;
        if (change == 0) bad.platform = "linux-aarch64";
        if (change == 1) bad.cuda_runtime = "99.0";
        if (change == 2) bad.tensorrt = "99.0.0.0";
        if (change == 3) bad.libtorch = "99.0.0+cu130";
        refused<sh::ConfigError>([&] { sh::select_manifest_build_targets(contract, bad); }, "no manifest target");
    }
    for (const auto unavailable : {0, 1, 2, 3}) {
        auto bad = build;
        if (unavailable == 0) bad.platform.clear();
        if (unavailable == 1) bad.cuda_runtime.clear();
        if (unavailable == 2) bad.tensorrt.clear();
        if (unavailable == 3) bad.libtorch.clear();
        refused<sh::ConfigError>([&] { sh::select_manifest_build_targets(contract, bad); }, "versions unavailable");
    }
    auto split = contract;
    split.targets = {first, first, first};
    split.targets[0].cuda_runtime = "99.0";
    split.targets[1].tensorrt = "99.0.0.0";
    split.targets[2].libtorch = "99.0.0+cu130";
    refused<sh::ConfigError>([&] { sh::select_manifest_build_targets(split, build); }, "no manifest target");

    // MB-44: SM belongs to a version-matched row, not any manifest row.
    auto multiple = contract;
    auto other = first;
    other.cuda_runtime = "99.0";
    other.gpu_sm = {"sm_86"};
    multiple.targets.push_back(other);
    auto restricted = sh::select_manifest_build_targets(multiple, build);
    CHECK(restricted.targets.size() == 1);
    refused<sh::DetectorLoadError>([&] { sh::require_manifest_gpu_sm(restricted, "sm_86"); }, "before engine deserialization");
    refused<sh::DetectorLoadError>([&] { sh::require_manifest_gpu_sm(restricted, ""); }, "SM unavailable");
    other = first;
    other.gpu_sm = {"sm_86"};
    multiple.targets.push_back(other);
    restricted = sh::select_manifest_build_targets(multiple, build);
    CHECK(restricted.targets.size() == 2);
    sh::require_manifest_gpu_sm(restricted, "sm_86");

    // MB-45 / MB-46: exact observed dtype, shape, name, ordinal and count.
    auto observed = contract.backbone;
    observed.outputs[0].dtype = "float16";
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(contract.backbone, observed, "engine"); }, "dtype mismatch");
    observed = contract.backbone;
    observed.outputs[0].shape[1] += 1;
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(contract.backbone, observed, "engine"); }, "shape mismatch");
    observed = contract.backbone;
    observed.inputs[0].shape[0] = -1;
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(contract.backbone, observed, "engine"); }, "shape mismatch");
    observed = contract.backbone;
    observed.outputs[0].name = "p4_feat";
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(contract.backbone, observed, "engine"); }, "name/order mismatch");
    observed = contract.backbone;
    observed.outputs[0].ordinal = 1;
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(contract.backbone, observed, "engine"); }, "ordinal/order mismatch");
    observed = contract.backbone;
    observed.outputs.pop_back();
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(contract.backbone, observed, "engine"); }, "count mismatch");
    observed = contract.backbone;
    observed.inputs[0].device = "cpu";
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(contract.backbone, observed, "engine"); }, "device mismatch");
    auto invalid = contract.backbone;
    invalid.outputs[0].ordinal = 1;
    refused<sh::DetectorLoadError>([&] { sh::require_tensor_io_contract(invalid, invalid, "engine"); }, "manifest ordinal/order");

    std::fprintf(stderr, "model compatibility: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
