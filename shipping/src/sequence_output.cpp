// Per-sequence output of the shipping host (#465 Phase B PR-6).
// See saccade_shipping/sequence_output.hpp.
#include "saccade_shipping/sequence_output.hpp"

#include <stdexcept>
#include <string>

#include "saccade_shipping/native_config.hpp"

namespace saccade::shipping {
namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw ConfigError("shipping sequence output: " + what +
                          " (the U4 output path has no implementation for it)");
    }
}

}  // namespace

SequenceOutputPlan plan_sequence_output(const ResolvedShippingConfig& cfg) {
    const auto& s = cfg.host_params.steps;
    const auto& c = cfg.host_params.cfg;

    // Emit: the oracle's fast emit and nothing else.
    require(!s.relink_semantic_relinker, "steps.relink.semantic_relinker must be false");
    require(!s.emit_pipeline_relink, "steps.emit.pipeline_relink must be false");
    require(!s.emit_id_stability_filter, "steps.emit.id_stability_filter must be false");
    require(!s.emit_appearance_bank, "steps.emit.appearance_bank must be false");
    require(!s.emit_dynamic_reid, "steps.emit.dynamic_reid must be false");
    require(s.emit_fast_emit_reid_mode, "steps.emit.fast_emit_reid_mode must be true");
    require(!s.emit_id_stability_kwarg, "steps.emit.id_stability_kwarg must be false");
    require(!s.track_workbench, "steps.track.workbench must be false");

    // Tail: steps are the gates; the cfg values they were evaluated from must
    // agree, so a hand-edited file cannot turn a step on without parameters.
    require(s.tail_interpolation == c.interpolate_tracklets,
            "steps.tail.interpolation must equal cfg.interpolate_tracklets");
    require(s.tail_write_output == !c.latency_only,
            "steps.tail.write_output must equal not cfg.latency_only");

    SequenceOutputPlan p;
    p.interpolate = s.tail_interpolation;
    p.interpolation.max_gap = c.interpolate_max_gap;
    p.interpolation.min_track_len = c.interpolate_min_track_len;
    p.interpolation.min_h = static_cast<double>(c.interpolate_min_h);
    p.write_output = s.tail_write_output;
    return p;
}

void SequenceOutput::add_frame(std::int64_t frame, const float* boxes_xyxy, const float* scores,
                               const std::int32_t* local_ids, std::size_t count) {
    if (finished_) throw std::logic_error("SequenceOutput: add_frame after finish");
    emit_mot_lines(lines_, ids_, frame, boxes_xyxy, scores, local_ids, count);
}

std::vector<std::string> SequenceOutput::finish(InterpolationStats* stats) {
    if (finished_) throw std::logic_error("SequenceOutput: finish called twice");
    finished_ = true;
    if (stats != nullptr) *stats = InterpolationStats{};
    if (!plan_.interpolate) return std::move(lines_);
    return interpolate_tracklets(lines_, plan_.interpolation, stats);
}

}  // namespace saccade::shipping
