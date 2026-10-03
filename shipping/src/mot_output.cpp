// MOT output of one sequence (#465 Phase B PR-6). See saccade_shipping/mot_output.hpp.
//
// Built with -ffp-contract=off: the interpolation must round after the multiply
// and after the add, as numpy's separate ufuncs do.
#include "saccade_shipping/mot_output.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>

namespace saccade::shipping {
namespace {

// format(v, ".{precision}f"): to_chars in fixed notation is printf("%.*f") in
// the "C" locale, i.e. correctly rounded like Python's float formatting. Python
// prints every NaN as "nan"; to_chars would keep the sign bit.
void append_fixed(std::string& out, double v, int precision) {
    if (std::isnan(v)) {
        out += "nan";
        return;
    }
    char buf[400];  // |v| <= DBL_MAX: 309 integer digits + sign + point + 4
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed, precision);
    if (r.ec != std::errc()) throw std::logic_error("mot_output: float formatting failed");
    out.append(buf, r.ptr);
}

void append_int(std::string& out, std::int64_t v) {
    char buf[24];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    out.append(buf, r.ptr);
}

constexpr std::string_view kTail = ",-1,-1,-1";

[[noreturn]] void bad_line(const std::string& line, const char* what) {
    throw std::invalid_argument(std::string("mot_output: ") + what + ": \"" + line + "\"");
}

std::int64_t parse_int(std::string_view s, const std::string& line) {
    std::int64_t v = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc() || r.ptr != s.data() + s.size()) bad_line(line, "not an integer field");
    return v;
}

// Correctly rounded, locale-independent; accepts what Python's formatting
// writes (including "nan", "inf", "-inf").
double parse_float(std::string_view s, const std::string& line) {
    double v = 0.0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc() || r.ptr != s.data() + s.size()) bad_line(line, "not a float field");
    return v;
}

// The first `n` comma-separated fields of `line` (more may follow).
std::vector<std::string_view> fields(const std::string& line, std::size_t n) {
    std::vector<std::string_view> out;
    out.reserve(n);
    std::string_view rest(line);
    while (out.size() < n) {
        const std::size_t comma = rest.find(',');
        if (comma == std::string_view::npos) {
            out.push_back(rest);
            break;
        }
        out.push_back(rest.substr(0, comma));
        rest.remove_prefix(comma + 1);
    }
    if (out.size() < n) bad_line(line, "too few fields");
    return out;
}

// One parsed line as the oracle's float64 row: frame, tid, x, y, w, h, score.
struct Row {
    std::int64_t frame, tid;
    double v[7];
};

Row parse_row(const std::string& line) {
    const auto f = fields(line, 7);
    Row r{};
    r.frame = parse_int(f[0], line);
    r.tid = parse_int(f[1], line);
    r.v[0] = static_cast<double>(r.frame);
    r.v[1] = static_cast<double>(r.tid);
    for (int c = 2; c < 7; ++c) r.v[c] = parse_float(f[c], line);
    return r;
}

}  // namespace

std::int64_t SequenceIdMapper::map(std::int32_t local_id) {
    const auto next = static_cast<std::int64_t>(ids_.size()) + 1;
    return ids_.try_emplace(local_id, next).first->second;
}

std::string format_mot_line(std::int64_t frame, std::int64_t id, float x1, float y1, float x2,
                            float y2, float score) {
    const double dx1 = x1, dy1 = y1, dx2 = x2, dy2 = y2;
    std::string s;
    s.reserve(64);
    append_int(s, frame);
    s += ',';
    append_int(s, id);
    s += ',';
    append_fixed(s, dx1, 2);
    s += ',';
    append_fixed(s, dy1, 2);
    s += ',';
    append_fixed(s, dx2 - dx1, 2);
    s += ',';
    append_fixed(s, dy2 - dy1, 2);
    s += ',';
    append_fixed(s, static_cast<double>(score), 4);
    s += kTail;
    return s;
}

void emit_mot_lines(std::vector<std::string>& out, SequenceIdMapper& ids, std::int64_t frame,
                    const float* boxes_xyxy, const float* scores, const std::int32_t* local_ids,
                    std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const float* b = boxes_xyxy + i * 4;
        out.push_back(format_mot_line(frame, ids.map(local_ids[i]), b[0], b[1], b[2], b[3],
                                      scores[i]));
    }
}

std::vector<std::string> interpolate_tracklets(const std::vector<std::string>& lines,
                                               const InterpolationParams& p,
                                               InterpolationStats* stats) {
    InterpolationStats st;
    if (stats != nullptr) *stats = st;
    if (lines.empty() || p.max_gap <= 0) return lines;

    std::vector<Row> all;
    all.reserve(lines.size());
    for (const std::string& line : lines) all.push_back(parse_row(line));

    // Confirmed tracks only (`groupby("tid")["frame"].count() >= min_track_len`),
    // then `sort_values(["tid", "frame"])` -- a stable lexsort.
    std::unordered_map<std::int64_t, std::int64_t> sizes;
    for (const Row& r : all) ++sizes[r.tid];
    std::vector<const Row*> rows;
    for (const Row& r : all) {
        if (sizes[r.tid] >= p.min_track_len) rows.push_back(&r);
    }
    if (rows.empty()) return lines;
    std::stable_sort(rows.begin(), rows.end(), [](const Row* a, const Row* b) {
        return a->tid != b->tid ? a->tid < b->tid : a->frame < b->frame;
    });

    std::vector<std::size_t> gap_idx;
    for (std::size_t i = 0; i + 1 < rows.size(); ++i) {
        const double* r0 = rows[i]->v;
        const double* r1 = rows[i + 1]->v;
        const double gap = r0[1] == r1[1] ? r1[0] - r0[0] - 1.0 : 0.0;
        bool valid = gap >= 1.0 && gap <= static_cast<double>(p.max_gap);
        if (p.min_h > 0.0) valid = valid && r0[5] >= p.min_h && r1[5] >= p.min_h;
        if (valid) gap_idx.push_back(i);
    }
    st.gaps_filled = static_cast<std::int64_t>(gap_idx.size());
    if (gap_idx.empty()) {
        if (stats != nullptr) *stats = st;
        return lines;
    }

    // New lines only; the originals stay verbatim.
    std::vector<std::string> added;
    std::vector<std::pair<std::int64_t, std::int64_t>> added_keys;
    std::set<std::int64_t> tracks;
    for (const std::size_t i : gap_idx) {
        const Row& a = *rows[i];
        const Row& b = *rows[i + 1];
        const std::int64_t gap = b.frame - a.frame - 1;
        st.frames_added += gap;
        tracks.insert(a.tid);
        const double denom = static_cast<double>(gap + 1);
        for (std::int64_t k = 1; k <= gap; ++k) {
            // np.arange(1, gap + 1, dtype=float64) / (gap + 1), then
            // r0 + alphas[:, None] * (r1 - r0): subtract, multiply, add.
            const double alpha = static_cast<double>(k) / denom;
            double v[7];
            for (int c = 2; c < 7; ++c) {
                const double diff = b.v[c] - a.v[c];
                const double step = alpha * diff;
                v[c] = a.v[c] + step;
            }
            const std::int64_t frame = a.frame + k;
            std::string s;
            s.reserve(64);
            append_int(s, frame);
            s += ',';
            append_int(s, a.tid);
            for (int c = 2; c < 6; ++c) {
                s += ',';
                append_fixed(s, v[c], 2);
            }
            s += ',';
            append_fixed(s, v[6], 4);
            s += kTail;
            added.push_back(std::move(s));
            added_keys.emplace_back(frame, a.tid);
        }
    }
    st.tracks_interpolated = static_cast<std::int64_t>(tracks.size());

    // lines + added, stably sorted by (int(frame), int(tid)).
    std::vector<std::pair<std::int64_t, std::int64_t>> keys;
    keys.reserve(all.size() + added.size());
    for (const Row& r : all) keys.emplace_back(r.frame, r.tid);
    keys.insert(keys.end(), added_keys.begin(), added_keys.end());
    std::vector<std::size_t> order(keys.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t x, std::size_t y) { return keys[x] < keys[y]; });
    std::vector<std::string> out;
    out.reserve(order.size());
    for (const std::size_t j : order) {
        out.push_back(j < lines.size() ? lines[j] : added[j - lines.size()]);
    }
    if (stats != nullptr) *stats = st;
    return out;
}

std::string join_mot_lines(const std::vector<std::string>& lines) {
    std::string out;
    std::size_t n = lines.empty() ? 0 : lines.size() - 1;
    for (const std::string& l : lines) n += l.size();
    out.reserve(n);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) out += '\n';
        out += lines[i];
    }
    return out;
}

}  // namespace saccade::shipping
