// Legacy SACCADE_* hatch resolution for the native tracking objects
// (#465 Phase B PR-4b). See tracking/legacy_env.hpp. Each read below keeps the
// parsing and the unset default of the code it was moved out of
// (tracker_gpu.cu constructor/update path, gmc_kernel.cu, pipeline.cpp,
// filter_detections_cuda); scripts/model/export_resolved_shipping_config.py
// records these defaults as the resolved JSON's `native_env`.
#include "tracking/legacy_env.hpp"

#include <cstdlib>
#include <cstring>

#include "saccade/env_flag.hpp"
#include "tracking/gmc.hpp"
#include "tracking/perception_params.hpp"
#include "tracking/pipeline.hpp"
#include "tracking/tracker_gpu.hpp"

namespace saccade {
namespace legacy_env {

namespace {

// Unset/empty → default; 0/false/False/FALSE → false; anything else → true.
bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) return default_value;
    return !(
        std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "False") == 0 ||
        std::strcmp(value, "FALSE") == 0
    );
}

// Unset/empty/unparsable → default.
float env_float_value(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) return default_value;
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value) return default_value;
    return parsed;
}

}  // namespace

TrackerParams::Hatch tracker_hatch() {
    TrackerParams::Hatch h;
    h.enable_dda = env_flag_enabled("SACCADE_ENABLE_DDA", true);
    h.dda_max_cost = env_float_value("SACCADE_DDA_MAX_COST", 0.12f);
    h.gate_adapt_r_mult = env_float_value("SACCADE_GATE_ADAPT_R_MULT", 1.0f);
    h.occ_vel_damp = env_float_value("SACCADE_OCC_VEL_DAMP", 1.0f);
    h.occ_vel_occ_thresh = env_float_value("SACCADE_OCC_VEL_OCC_THRESH", 0.05f);
    h.output_measurement = env_flag_enabled("SACCADE_OUTPUT_MEASUREMENT", false);
    h.coast_max_age = static_cast<int>(env_float_value("SACCADE_COAST_MAX_AGE", 0.0f));
    h.coast_score_decay = env_float_value("SACCADE_COAST_SCORE_DECAY", 1.0f);
    h.coast_occ_thresh = env_float_value("SACCADE_COAST_OCC_THRESH", 0.0f);
    // The two auction bid weights were raw strtof reads (set-but-empty → 0).
    h.freshness_w = [] {
        const char* v = std::getenv("SACCADE_FRESHNESS_W");
        return v ? std::strtof(v, nullptr) : 0.0f;
    }();
    h.stability_w = [] {
        const char* v = std::getenv("SACCADE_STABILITY_W");
        return v ? std::strtof(v, nullptr) : 0.1f;
    }();
    return h;
}

std::optional<int> kalman_adapt_mode_override() {
    const char* v = std::getenv("SACCADE_KALMAN_ADAPT_MODE");
    if (v && *v) return std::atoi(v);
    return std::nullopt;
}

std::string assoc_dump_path() {
    const char* v = std::getenv("SACCADE_ASSOC_DUMP");
    return v ? std::string(v) : std::string();
}

float gmc_pcr_thresh() {
    const char* v = std::getenv("SACCADE_GMC_PCR_THRESH");
    return v ? std::strtof(v, nullptr) : 5.0f;
}

FilterCompactionMode filter_compaction_mode() {
    return saccade::filter_compaction_mode(
        env_flag_enabled("SACCADE_DETERMINISTIC_FILTER_COMPACTION", false),
        env_flag_enabled("SACCADE_ATOMIC_FILTER_BASELINE", false));
}

bool assoc_stats() {
    return env_diagnostic_on("SACCADE_ASSOC_STATS");
}

void apply(GPUByteTracker& tracker) {
    tracker.set_hatch_params(tracker_hatch());
    tracker.set_assoc_dump_path(assoc_dump_path());
}

void apply(GMC& gmc) {
    gmc.set_pcr_thresh(gmc_pcr_thresh());
}

void apply(PerceptionPipeline& pipeline) {
    pipeline.set_filter_compaction_mode(filter_compaction_mode());
    if (assoc_stats()) {
        pipeline.set_private_workload_stats_enabled(true);
    }
}

}  // namespace legacy_env
}  // namespace saccade
