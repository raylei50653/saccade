// MOT output of one sequence (#465 Phase B PR-6, U4; boundary §5 B3, B4).
// CUDA-free, config-free.
//
// Native twins of the oracle's per-sequence output path in the headline
// configuration:
//   * B3 -- track ids and MOT lines. The oracle maps each tracker local id to a
//     run-global id in first-appearance order (tracking.py
//     GlobalTrackIdMapper) and formats one line per tracker row
//     (helpers.py fast_emit_mot_lines). Shipping keeps the first-appearance
//     rule per sequence, starting at 1; the run-global counter stays in the
//     eval harness. For one sequence the two are the same ids.
//   * B4 -- sequence-tail interpolation (post_merge.py interpolate_tracklets):
//     MOT lines in, MOT lines out, parsed from the text as the oracle parses
//     it (so interpolation sees the rounded values), in float64 with the
//     oracle's operation order, original lines kept verbatim.
// Numbers are formatted as Python's `format(float, ".Nf")` (correctly rounded,
// locale-independent) and parsed correctly rounded, as pandas' C parser does
// for these short decimals. tests/native/fixtures/shipping_mot_output.json pins
// both functions against the Python ones.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace saccade::shipping {

// Output ids of one sequence: 1, 2, ... in order of first appearance.
class SequenceIdMapper {
public:
    std::int64_t map(std::int32_t local_id);
    std::size_t size() const { return ids_.size(); }

private:
    std::unordered_map<std::int32_t, std::int64_t> ids_;
};

// "{frame},{id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{score:.4f},-1,-1,-1"
// with every float widened to double first, as the oracle's float(...) does.
std::string format_mot_line(std::int64_t frame, std::int64_t id, float x1, float y1, float x2,
                            float y2, float score);

// One frame of tracker rows (xyxy boxes [count*4], scores, local ids) to MOT
// lines, in row order, mapping ids through `ids`.
void emit_mot_lines(std::vector<std::string>& out, SequenceIdMapper& ids, std::int64_t frame,
                    const float* boxes_xyxy, const float* scores, const std::int32_t* local_ids,
                    std::size_t count);

struct InterpolationParams {
    std::int64_t max_gap = 0;
    std::int64_t min_track_len = 0;
    double min_h = 0.0;
};

struct InterpolationStats {
    std::int64_t tracks_interpolated = 0;
    std::int64_t gaps_filled = 0;
    std::int64_t frames_added = 0;
};

// interpolate_tracklets: for tracks with >= min_track_len lines, fill every
// gap of 1..max_gap missing frames (both ends' h >= min_h when min_h > 0)
// linearly; return the input lines plus the new ones, stably sorted by
// (frame, id). When nothing is filled the input is returned unchanged (and
// unsorted), as the oracle does. Throws std::invalid_argument on a line that is
// not `int,int,float,float,float,float,float,...`.
std::vector<std::string> interpolate_tracklets(const std::vector<std::string>& lines,
                                               const InterpolationParams& params,
                                               InterpolationStats* stats = nullptr);

// The output file's contents: "\n".join(lines) (no trailing newline).
std::string join_mot_lines(const std::vector<std::string>& lines);

}  // namespace saccade::shipping
