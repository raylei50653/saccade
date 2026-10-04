// Frame schedule and graph capture plan (#465 Phase B PR-10).
// See saccade_shipping/schedule_plan.hpp.
#include "saccade_shipping/schedule_plan.hpp"

#include <string>

#include "saccade_shipping/native_config.hpp"

namespace saccade::shipping {
namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw ConfigError("shipping schedule: " + what);
}

bool env_is(const std::optional<std::string>& v, const char* want) {
    return v.has_value() && *v == want;
}

}  // namespace

SchedulePlan plan_schedule(const ResolvedShippingConfig& cfg) {
    const auto& hp = cfg.host_params;
    const auto& s = hp.steps;
    const auto& b = hp.detector.build;
    SchedulePlan p;
    // pipeline.py _double_buffer_eligible: the env switch, the event barrier
    // and a frame-independent detector. The exporter recorded the outcome; a
    // config whose inputs disagree with it is refused rather than reinterpreted.
    p.double_buffer = s.schedule_double_buffer;
    require(env_is(hp.env.double_buffer, "1") == p.double_buffer,
            "steps.schedule.double_buffer disagrees with SACCADE_DOUBLE_BUFFER");
    if (p.double_buffer) {
        require(env_is(hp.env.detect_barrier, "event"),
                "the double-buffer schedule needs SACCADE_DETECT_BARRIER \"event\"");
        require(b.use_whole_graph, "the double-buffer schedule needs a whole-graph detector");
    }
    // MambaGatedDetector.forward: the whole graph with the TRT backbone.
    p.whole_detect_graph = b.use_whole_graph && !b.trt_backbone_engine.empty();
    // stages.py _run_nms: the graphed main NMS, only without ONMS priors.
    p.main_nms_graph = env_is(hp.env.main_nms_graphed, "1") && !hp.onms.enabled && !s.post_onms_priors;
    // pipeline.py: the C++ cuFFT GMC (gmc_mode "gpu"), graphable without the
    // foreground mask.
    p.gmc_graph = s.gmc && hp.cfg.gmc_mode == "gpu" && !s.gmc_fg_mask;
    p.tracker_graph = s.track_graphed_update;
    require(p.whole_detect_graph, "the whole-detect graph must be on (use_whole_graph, TRT backbone)");
    require(p.main_nms_graph, "the graphed main NMS must be on (SACCADE_MAIN_NMS_GRAPHED, no ONMS)");
    require(!s.gmc || p.gmc_graph, "GMC must be the graphable direct GMC (gmc_mode \"gpu\", no fg mask)");
    require(p.tracker_graph, "steps.track.graphed_update must be true");
    return p;
}

}  // namespace saccade::shipping
