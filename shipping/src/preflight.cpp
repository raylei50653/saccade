// Gate A of saccade_track (#536 CC-536-01-02). See saccade_shipping/preflight.hpp.
#include "saccade_shipping/preflight.hpp"

#include <exception>
#include <filesystem>
#include <system_error>
#include <utility>

#include "saccade_shipping/ingest_plan.hpp"
#include "saccade_shipping/native_config.hpp"
#include "saccade_shipping/post_detector_plan.hpp"
#include "saccade_shipping/sequence_output.hpp"
#include "saccade_shipping/sha256.hpp"

namespace saccade::shipping {
namespace {

[[noreturn]] void preflight_error(const std::string& what) { throw PreflightError("preflight: " + what); }

// One step of Gate A: any error it throws becomes a PreflightError.
template <class F>
auto step(const std::string& context, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const PreflightError&) {
        throw;
    } catch (const std::exception& e) {
        preflight_error(context + e.what());
    }
}

void check_artifact(const char* what, const std::string& model_root, const FileBinding& b) {
    const std::string path = resolve_model_path(model_root, b.path);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        preflight_error(std::string(what) + " " + path + " is missing (not a regular file)");
    }
    const std::string got = step("", [&] { return sha256_file_hex(path); });
    if (got != b.sha256) {
        preflight_error(std::string(what) + " " + path + " sha256 " + got + " != " + b.sha256);
    }
}

}  // namespace

PreflightResult run_preflight(const PreflightInputs& in, const RunCompletion& completion) {
    // 1. The config and every CUDA-free plan the runtime builds from it.
    const ResolvedShippingConfig cfg =
        step("", [&] { return load_resolved_shipping_config_file(in.config); });
    const Schedule schedule = step("", [&] {
        plan_ingest(cfg);
        if (!plan_sequence_output(cfg).write_output) {
            throw ConfigError("the config writes no MOT output (steps.tail.write_output)");
        }
        plan_post_detector(cfg);
        planned_pipeline_snapshot(cfg);
        return select_schedule(cfg, in.serial_requested);
    });
    DetectorPlan detector =
        step("", [&] { return plan_detector_files(cfg, DetectorInputs{in.lineage, in.attestation}); });

    // 2. Every sequence's input, as run_sequence will read it.
    for (const std::string& dir : in.sequences) {
        step("sequence " + dir + ": ", [&] { return read_sequence_input(dir, in.max_frames); });
    }

    // 3. Every directory the run publishes into.
    step("", [&] { completion.check_writable(); });

    // 4. The three files Gate B loads, by the plan's sha256.
    check_artifact("operator library", in.model_root, detector.op_library);
    check_artifact("head artifact", in.model_root, detector.head_artifact);
    check_artifact("backbone engine", in.model_root, detector.backbone_engine);
    return PreflightResult{cfg, schedule, std::move(detector)};
}

}  // namespace saccade::shipping
