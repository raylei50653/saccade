// Per-sequence output of the shipping host (#465 Phase B PR-6, U4). CUDA-free.
//
// `plan_sequence_output` reads, from the resolved config, everything the
// oracle's emit and sequence tail read in the headline configuration, and
// fails closed (ConfigError) on any oracle branch that would produce
// different lines:
//   * emit: `_run_emit` takes the fast emit (helpers.py fast_emit_mot_lines,
//     one line per tracker row) only with no semantic relinker, id-stability
//     filter, appearance bank or dynamic-ReID controller, a fast-emit
//     `reid_mode`, the `id_stability_filter` kwarg off, no pipelined relink and
//     no workbench (steps emit.*, relink.semantic_relinker, track.workbench);
//   * tail: interpolation is the only tail step with a native implementation
//     (boundary §5 B4); the loader already refuses the other tail steps on.
//     Its parameters come from host_params.cfg; whether the file is written
//     is steps.tail.write_output.
// `SequenceOutput` accumulates one sequence's lines frame by frame and applies
// the tail once at the end (mot_output.hpp holds the formatting rules).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "saccade_shipping/mot_output.hpp"
#include "saccade_shipping/resolved_config.hpp"

namespace saccade::shipping {

struct SequenceOutputPlan {
    bool interpolate = false;  // steps.tail.interpolation
    InterpolationParams interpolation;
    bool write_output = false;  // steps.tail.write_output
};

SequenceOutputPlan plan_sequence_output(const ResolvedShippingConfig& cfg);

class SequenceOutput {
public:
    explicit SequenceOutput(SequenceOutputPlan plan) : plan_(plan) {}

    // One tracker update's rows (xyxy boxes [count*4], scores, local ids), in
    // frame order. Frames without a tracker update add nothing.
    void add_frame(std::int64_t frame, const float* boxes_xyxy, const float* scores,
                   const std::int32_t* local_ids, std::size_t count);

    // Applies the tail and returns the sequence's final lines; once.
    std::vector<std::string> finish(InterpolationStats* stats = nullptr);

    const SequenceOutputPlan& plan() const { return plan_; }
    std::size_t emitted_lines() const { return lines_.size(); }
    std::size_t track_ids() const { return ids_.size(); }

private:
    SequenceOutputPlan plan_;
    SequenceIdMapper ids_;
    std::vector<std::string> lines_;
    bool finished_ = false;
};

}  // namespace saccade::shipping
