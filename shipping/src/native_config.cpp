// Resolved shipping config -> native tracking state, and readback
// (#465 Phase B PR-4b). See saccade_shipping/native_config.hpp.
#include "saccade_shipping/native_config.hpp"

#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <sstream>

namespace saccade::shipping {

int native_int(std::int64_t value, const char* what) {
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw ConfigError(std::string("shipping: ") + what + " does not fit a native int");
    }
    return static_cast<int>(value);
}

TrackerParams::Hatch tracker_hatch(const NativeEnvParams& env) {
    TrackerParams::Hatch h;
    h.enable_dda = env.enable_dda;
    h.dda_max_cost = native_float(env.dda_max_cost);
    h.gate_adapt_r_mult = native_float(env.gate_adapt_r_mult);
    h.occ_vel_damp = native_float(env.occ_vel_damp);
    h.occ_vel_occ_thresh = native_float(env.occ_vel_occ_thresh);
    h.output_measurement = env.output_measurement;
    // The legacy read truncated the float env value to int; a non-integral
    // value fails readback instead of being truncated silently.
    h.coast_max_age = static_cast<int>(env.coast_max_age);
    h.coast_score_decay = native_float(env.coast_score_decay);
    h.coast_occ_thresh = native_float(env.coast_occ_thresh);
    h.freshness_w = native_float(env.freshness_w);
    h.stability_w = native_float(env.stability_w);
    return h;
}

TrackerParams planned_tracker_params(const ResolvedShippingConfig& cfg, SequenceGeometry g) {
    TrackerParams params;
    apply_tracker_config(params, cfg, g);
    return params;
}

PerceptionPipelineConfig planned_pipeline_config(const ResolvedShippingConfig& cfg) {
    const auto& f = cfg.native_params.perception_pipeline_config.fields;
    PerceptionPipelineConfig c;
    c.score_threshold = native_float(f.score_threshold);
    c.person_class = native_int(f.person_class, "person_class");
    c.person_only = f.person_only;
    c.nms_threshold = native_float(f.nms_threshold);
    c.person_geometry_prior = f.person_geometry_prior;
    c.geometry_suspect_support = f.geometry_suspect_support;
    c.geometry_suspect_support_score = native_float(f.geometry_suspect_support_score);
    c.person_min_height_ratio = native_float(f.person_min_height_ratio);
    c.person_min_aspect = native_float(f.person_min_aspect);
    c.person_max_aspect = native_float(f.person_max_aspect);
    c.person_min_area_ratio = native_float(f.person_min_area_ratio);
    c.person_max_area_ratio = native_float(f.person_max_area_ratio);
    c.max_detections = native_int(f.max_detections, "max_detections");
    c.private_continuation_enabled = f.private_continuation_enabled;
    c.private_candidate_nms_iou = native_float(f.private_candidate_nms_iou);
    c.private_min_score = native_float(f.private_min_score);
    c.private_max_candidates = native_int(f.private_max_candidates, "private_max_candidates");
    c.private_prior_iou_threshold = native_float(f.private_prior_iou_threshold);
    c.private_prior_center_threshold = native_float(f.private_prior_center_threshold);
    c.private_low_stage_only = f.private_low_stage_only;
    c.private_track_thresh = native_float(f.private_track_thresh);
    c.private_mid_thresh = native_float(f.private_mid_thresh);
    c.private_new_track_thresh = native_float(f.private_new_track_thresh);
    c.private_score_eps = native_float(f.private_score_eps);
    return c;
}

FilterCompactionMode planned_filter_compaction(const ResolvedShippingConfig& cfg) {
    return filter_compaction_mode(cfg.native_env.deterministic_filter_compaction,
                                  cfg.native_env.atomic_filter_baseline);
}

GmcSnapshot planned_gmc_snapshot(const ResolvedShippingConfig& cfg) {
    const auto& k = cfg.native_params.gmc.constructor;
    GmcSnapshot s;
    s.downscale = native_int(k.downscale, "downscale");
    s.max_corners = native_int(k.max_corners, "max_corners");
    s.quality_level = native_float(k.quality_level);
    s.min_distance = native_float(k.min_distance);
    s.min_inliers = native_int(k.min_inliers, "min_inliers");
    s.ransac_threshold = native_float(k.ransac_threshold);
    s.pcr_thresh = native_float(cfg.native_env.gmc_pcr_thresh);
    s.profiling_enabled = cfg.native_params.gmc.calls.set_profiling_enabled.enabled;
    return s;
}

PerceptionPipelineSnapshot planned_pipeline_snapshot(const ResolvedShippingConfig& cfg) {
    const auto& k = cfg.native_params.perception_pipeline.constructor;
    PerceptionPipelineSnapshot s;
    // The loader admits only 0 for both pointers: no ReID extractor, no cropper.
    s.reid_ptr = static_cast<std::uintptr_t>(k.reid_ptr);
    s.cropper_ptr = static_cast<std::uintptr_t>(k.cropper_ptr);
    s.config = planned_pipeline_config(cfg);
    s.postprocess_profiling_enabled =
        cfg.native_params.perception_pipeline.calls.set_postprocess_profiling_enabled.enabled;
    s.filter_compaction = planned_filter_compaction(cfg);
    s.private_workload_stats_enabled = false;  // native_env.SACCADE_ASSOC_STATS: unset
    return s;
}

// ─── expectations ────────────────────────────────────────────────────────

const std::vector<NativeEnvConsumer>& native_env_consumers() {
    static const std::vector<NativeEnvConsumer> consumers = {
        {"SACCADE_ASSOC_DUMP", "GPUByteTracker", ""},
        {"SACCADE_ASSOC_STATS", "PerceptionPipeline", ""},
        {"SACCADE_ATOMIC_FILTER_BASELINE", "PerceptionPipeline", ""},
        {"SACCADE_COAST_MAX_AGE", "GPUByteTracker", ""},
        {"SACCADE_COAST_OCC_THRESH", "GPUByteTracker", ""},
        {"SACCADE_COAST_SCORE_DECAY", "GPUByteTracker", ""},
        {"SACCADE_DDA_MAX_COST", "GPUByteTracker", ""},
        {"SACCADE_DETERMINISTIC_FILTER_COMPACTION", "PerceptionPipeline", ""},
        {"SACCADE_ENABLE_DDA", "GPUByteTracker", ""},
        {"SACCADE_FRESHNESS_W", "GPUByteTracker", ""},
        {"SACCADE_GATE_ADAPT_R_MULT", "GPUByteTracker", ""},
        {"SACCADE_GMC_PCR_THRESH", "GMC", ""},
        {"SACCADE_HO_DEBUG_LEVEL", "",
         "read only by the Cheb-GR handover relinker binding, which shipping does not "
         "construct (resolved JSON source.native_objects_not_shipped)"},
        {"SACCADE_KALMAN_ADAPT_MODE", "",
         "legacy-only override of set_params.kalman_adapt_mode applied by the pybind/"
         "seq_runner front-ends; unset means no override, and shipping applies "
         "set_params.kalman_adapt_mode from the JSON directly"},
        {"SACCADE_OCC_VEL_DAMP", "GPUByteTracker", ""},
        {"SACCADE_OCC_VEL_OCC_THRESH", "GPUByteTracker", ""},
        {"SACCADE_OUTPUT_MEASUREMENT", "GPUByteTracker", ""},
        {"SACCADE_STABILITY_W", "GPUByteTracker", ""},
    };
    return consumers;
}

const std::vector<NativeOnlyExpectation>& tracker_native_only_expectations() {
    static const std::vector<NativeOnlyExpectation> rows = {
        {"set_reid_min_candidates.min_candidates",
         JsonValue::make_int(TrackerParams{}.reid_min_candidates),
         "no pybind binding, so the oracle never sets it and the exporter has no value; "
         "read only on the embedding-association branch (update with a non-null "
         "embeddings pointer), which is unreachable on shipping trackers "
         "(embeddings_forbidden)"},
        {"embeddings_forbidden", JsonValue::make_bool(true),
         "shipping has no ReID: build_tracker calls forbid_embeddings(), so an update "
         "with embeddings throws instead of reaching the embedding branch"},
        {"research.portable_or_tail", JsonValue::make_bool(false),
         "research hook; never armed by the oracle"},
        {"research.bridge_shadow", JsonValue::make_bool(false),
         "research hook; never armed by the oracle"},
        {"research.bridge_fidelity_audit", JsonValue::make_bool(false),
         "research capture; never armed by the oracle"},
        {"research.h0_bridge_trace", JsonValue::make_bool(false),
         "research capture; never armed by the oracle"},
        {"config_frozen", JsonValue::make_bool(false),
         "configuration is applied before the first graph capture"},
    };
    return rows;
}

namespace {

const JsonValue& child(const JsonValue& node, std::string_view key) {
    const JsonValue* v = node.find(key);
    if (v == nullptr) {
        throw ConfigError("shipping: resolved config has no '" + std::string(key) + "'");
    }
    return *v;
}

void put(SnapshotExpectation& out, const std::string& key, JsonValue value) {
    if (!out.emplace(key, std::move(value)).second) {
        throw ConfigError("shipping: two resolved-config values map to native key " + key);
    }
}

bool is_unset_env(const JsonValue& v) {
    return v.kind == JsonValue::Kind::Object && v.object.size() == 1 &&
           v.object.front().first == "unset_effect";
}

// Args object -> `<prefix>.<arg>` (nested objects keep extending the key).
// `{"per_sequence": ...}` is a leaf resolved from the sequence geometry.
void flatten(const JsonValue& obj, const std::string& prefix, SnapshotExpectation& out,
             const std::optional<SequenceGeometry>& g) {
    for (const auto& [k, v] : obj.object) {
        const std::string key = prefix + "." + k;
        if (v.kind == JsonValue::Kind::Object && v.object.size() == 1 &&
            v.object.front().first == "per_sequence") {
            if (!g) throw ConfigError("shipping: per-sequence value " + key + " needs a geometry");
            const std::string& which = v.object.front().second.string;
            if (which == "seqinfo.ini:Sequence.imWidth") {
                put(out, key, JsonValue::make_int(g->im_width));
            } else if (which == "seqinfo.ini:Sequence.imHeight") {
                put(out, key, JsonValue::make_int(g->im_height));
            } else {
                throw ConfigError("shipping: unknown per-sequence value " + which);
            }
        } else if (v.kind == JsonValue::Kind::Object) {
            flatten(v, key, out, g);
        } else {
            put(out, key, v);
        }
    }
}

void flatten_object(const JsonValue& object, SnapshotExpectation& out,
                    const std::optional<SequenceGeometry>& g) {
    flatten(child(object, "constructor"), "constructor", out, g);
    for (const JsonValue& call : child(object, "calls").array) {
        flatten(child(call, "args"), child(call, "method").string, out, g);
    }
}

// native_env keys this object consumes, as `native_env.<KEY>`, except the
// diagnostics, which the caller maps onto their snapshot keys.
void put_native_env(const JsonValue& env, const char* consumer, SnapshotExpectation& out) {
    for (const NativeEnvConsumer& c : native_env_consumers()) {
        if (std::strcmp(c.consumer, consumer) != 0) continue;
        const JsonValue& v = child(env, c.key);
        if (is_unset_env(v)) continue;  // diagnostics: mapped by the caller
        put(out, std::string("native_env.") + c.key, v);
    }
}

}  // namespace

SnapshotExpectation expected_tracker_snapshot(const ResolvedShippingConfig& cfg, SequenceGeometry g) {
    const JsonValue doc = to_json(cfg);
    SnapshotExpectation out;
    flatten_object(child(child(doc, "native_params"), "GPUByteTracker"), out, g);
    const JsonValue& env = child(doc, "native_env");
    put_native_env(env, "GPUByteTracker", out);
    // SACCADE_ASSOC_DUMP: the loader requires {"unset_effect"} → dump off.
    if (!is_unset_env(child(env, "SACCADE_ASSOC_DUMP"))) {
        throw ConfigError("shipping: native_env.SACCADE_ASSOC_DUMP must be unset");
    }
    put(out, "diagnostic.assoc_dump", JsonValue::make_bool(false));
    for (const NativeOnlyExpectation& row : tracker_native_only_expectations()) {
        put(out, row.key, row.value);
    }
    return out;
}

SnapshotExpectation expected_gmc_snapshot(const ResolvedShippingConfig& cfg) {
    const JsonValue doc = to_json(cfg);
    SnapshotExpectation out;
    flatten_object(child(child(doc, "native_params"), "GMC"), out, std::nullopt);
    put_native_env(child(doc, "native_env"), "GMC", out);
    return out;
}

SnapshotExpectation expected_pipeline_snapshot(const ResolvedShippingConfig& cfg) {
    const JsonValue doc = to_json(cfg);
    const JsonValue& native = child(doc, "native_params");
    const JsonValue& pipeline = child(native, "PerceptionPipeline");
    SnapshotExpectation out;
    // constructor.config is a reference to the PerceptionPipelineConfig
    // object; expand it into that object's fields.
    for (const auto& [k, v] : child(pipeline, "constructor").object) {
        if (k == "config") {
            if (child(v, "ref").string != "PerceptionPipelineConfig") {
                throw ConfigError("shipping: PerceptionPipeline.constructor.config must reference "
                                  "PerceptionPipelineConfig");
            }
            const JsonValue& config = child(native, "PerceptionPipelineConfig");
            if (!child(config, "constructor").object.empty()) {
                throw ConfigError("shipping: PerceptionPipelineConfig takes no constructor args");
            }
            flatten(child(config, "fields"), "constructor.config", out, std::nullopt);
        } else {
            put(out, "constructor." + k, v);
        }
    }
    for (const JsonValue& call : child(pipeline, "calls").array) {
        flatten(child(call, "args"), child(call, "method").string, out, std::nullopt);
    }
    const JsonValue& env = child(doc, "native_env");
    put(out, "filter_compaction",
        JsonValue::make_string(to_string(planned_filter_compaction(cfg))));
    if (!is_unset_env(child(env, "SACCADE_ASSOC_STATS"))) {
        throw ConfigError("shipping: native_env.SACCADE_ASSOC_STATS must be unset");
    }
    put(out, "diagnostic.private_workload_stats", JsonValue::make_bool(false));
    return out;
}

// ─── readback ────────────────────────────────────────────────────────────

namespace {

std::uint32_t float_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

bool matches(const JsonValue& want, bool v) {
    return want.kind == JsonValue::Kind::Bool && want.boolean == v;
}
bool matches(const JsonValue& want, int v) {
    if (want.kind == JsonValue::Kind::Int) return want.integer == v;
    // SACCADE_COAST_MAX_AGE is a float in the JSON (the legacy read parsed a
    // float and truncated it); readback requires the exact integral value.
    return want.kind == JsonValue::Kind::Float && want.number == static_cast<double>(v);
}
bool matches(const JsonValue& want, std::uintptr_t v) {
    return want.kind == JsonValue::Kind::Int && want.integer >= 0 &&
           static_cast<std::uint64_t>(want.integer) == static_cast<std::uint64_t>(v);
}
bool matches(const JsonValue& want, float v) {
    return want.kind == JsonValue::Kind::Float &&
           float_bits(static_cast<float>(want.number)) == float_bits(v);
}
bool matches(const JsonValue& want, const std::optional<std::array<float, 9>>& v) {
    return want.kind == JsonValue::Kind::Null && !v.has_value();
}
bool matches(const JsonValue& want, FilterCompactionMode v) {
    return want.kind == JsonValue::Kind::String && want.string == to_string(v);
}

std::string show(bool v) { return v ? "true" : "false"; }
std::string show(int v) { return std::to_string(v); }
std::string show(std::uintptr_t v) { return std::to_string(static_cast<std::uint64_t>(v)); }
std::string show(float v) { return python_float_repr(static_cast<double>(v)); }
std::string show(const std::optional<std::array<float, 9>>& v) {
    return v ? "<3x3 matrix>" : "null";
}
std::string show(FilterCompactionMode v) { return std::string("\"") + to_string(v) + "\""; }

struct Comparator {
    const SnapshotExpectation& expected;
    std::set<std::string>& seen;
    std::vector<std::string>& mismatches;

    template <class K, class T> void operator()(const K& key_like, const T& value) const {
        const std::string key(key_like);
        if (!seen.insert(key).second) {
            mismatches.push_back(key + ": native snapshot visits this key twice");
            return;
        }
        const auto it = expected.find(key);
        if (it == expected.end()) {
            mismatches.push_back(key + ": native field has no resolved-config value");
            return;
        }
        if (!matches(it->second, value)) {
            mismatches.push_back(key + ": native " + show(value) + " != resolved " +
                                 dump_python_json(it->second));
        }
    }
};

template <class Snapshot>
std::vector<std::string> compare(const SnapshotExpectation& expected, const Snapshot& snapshot) {
    std::set<std::string> seen;
    std::vector<std::string> mismatches;
    snapshot.visit(Comparator{expected, seen, mismatches});
    for (const auto& [key, value] : expected) {
        if (seen.count(key) == 0) {
            mismatches.push_back(key + ": resolved-config value " + dump_python_json(value) +
                                 " has no native field");
        }
    }
    return mismatches;
}

}  // namespace

std::vector<std::string> readback_mismatches(const SnapshotExpectation& expected,
                                             const TrackerSnapshot& snapshot) {
    return compare(expected, snapshot);
}
std::vector<std::string> readback_mismatches(const SnapshotExpectation& expected,
                                             const GmcSnapshot& snapshot) {
    return compare(expected, snapshot);
}
std::vector<std::string> readback_mismatches(const SnapshotExpectation& expected,
                                             const PerceptionPipelineSnapshot& snapshot) {
    return compare(expected, snapshot);
}

void require_readback(const char* object, const std::vector<std::string>& mismatches) {
    if (mismatches.empty()) return;
    std::ostringstream msg;
    msg << "shipping: " << object << " readback != resolved config (" << mismatches.size()
        << " mismatch" << (mismatches.size() == 1 ? "" : "es") << ")";
    for (const std::string& m : mismatches) msg << "\n  " << m;
    throw ConfigError(msg.str());
}

}  // namespace saccade::shipping
