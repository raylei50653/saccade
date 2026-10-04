// Frame schedule and graph capture plan (#465 Phase B PR-10, U5). CUDA-free.
//
// Which schedule and which CUDA graphs the oracle runs for this config,
// derived only from the resolved config (fail closed: ConfigError):
//   * double_buffer: steps.schedule.double_buffer (the exporter's record of
//     `_double_buffer_eligible`); when true the oracle's own preconditions
//     must read the same way: SACCADE_DOUBLE_BUFFER "1" and
//     SACCADE_DETECT_BARRIER "event" (the only barrier mode that admits the
//     overlap), and a frame-independent detector (whole graph);
//   * the four graphs the oracle captures on this path: the whole-detect
//     graph (detector.build.use_whole_graph with the TRT backbone), the main
//     NMS graph (SACCADE_MAIN_NMS_GRAPHED "1", no ONMS priors), the direct GMC
//     graph (gmc_mode "gpu", no foreground mask) and the graphed tracker
//     update (steps.track.graphed_update). The native runtime implements the
//     path where all four are captured and refuses any other.
#pragma once

#include "saccade_shipping/resolved_config.hpp"

namespace saccade::shipping {

struct SchedulePlan {
    bool double_buffer = false;
    bool whole_detect_graph = false, main_nms_graph = false, gmc_graph = false,
         tracker_graph = false;
};

SchedulePlan plan_schedule(const ResolvedShippingConfig& cfg);

}  // namespace saccade::shipping
