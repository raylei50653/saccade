#pragma once

// Legacy SACCADE_* environment hatches for the native tracking objects
// (#465 Phase B PR-4b).
//
// The tracker, GMC and PerceptionPipeline no longer read the environment;
// they take every value as an explicit parameter. These functions are the
// *only* native code that turns SACCADE_* variables into those parameters, and
// only the legacy front-ends call them: the pybind bindings (eval harness) and
// seq_runner (`--cpp-threads`). The shipping runtime never includes this
// header; its values come from the resolved JSON's `native_env`.
//
// Each read keeps the exact parsing and default of the code it replaced, so a
// legacy run resolves the same values it did before PR-4b. When a variable is
// read moved: the tracker hatches and the GMC phase-correlation threshold were
// read at construction or once per process at the first update/estimate; they
// are now read when the front-end constructs the object.

#include <optional>
#include <string>

#include "tracking/tracker_params.hpp"

namespace saccade {

class GPUByteTracker;
class GMC;
class PerceptionPipeline;

enum class FilterCompactionMode;

namespace legacy_env {

// SACCADE_ENABLE_DDA, _DDA_MAX_COST, _GATE_ADAPT_R_MULT, _OCC_VEL_DAMP,
// _OCC_VEL_OCC_THRESH, _OUTPUT_MEASUREMENT, _COAST_MAX_AGE,
// _COAST_SCORE_DECAY, _COAST_OCC_THRESH, _FRESHNESS_W, _STABILITY_W.
TrackerParams::Hatch tracker_hatch();

// SACCADE_KALMAN_ADAPT_MODE: when set, replaces set_params' kalman_adapt_mode.
// Read on every set_params call, as before.
std::optional<int> kalman_adapt_mode_override();

// SACCADE_ASSOC_DUMP: association dump CSV path; empty = off.
std::string assoc_dump_path();

// SACCADE_GMC_PCR_THRESH.
float gmc_pcr_thresh();

// SACCADE_DETERMINISTIC_FILTER_COMPACTION / SACCADE_ATOMIC_FILTER_BASELINE.
FilterCompactionMode filter_compaction_mode();

// SACCADE_ASSOC_STATS.
bool assoc_stats();

// Apply every construction-time hatch above to a freshly constructed object.
void apply(GPUByteTracker& tracker);
void apply(GMC& gmc);
void apply(PerceptionPipeline& pipeline);

}  // namespace legacy_env
}  // namespace saccade
