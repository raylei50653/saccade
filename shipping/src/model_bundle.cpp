// Model bundle manifest and allowlist (#549 S2-1). See
// saccade_shipping/model_bundle.hpp.
#include "saccade_shipping/model_bundle.hpp"

#include <fcntl.h>
#include <linux/openat2.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <set>
#include <utility>
#include <vector>

#include "saccade_shipping/sha256.hpp"

namespace saccade::shipping {
namespace {

// ── schema keywords ───────────────────────────────────────────────────────────

const std::regex& re(const char* pattern) {
    static std::map<std::string, std::regex> cache;
    auto it = cache.find(pattern);
    if (it == cache.end()) it = cache.emplace(pattern, std::regex(pattern, std::regex::ECMAScript)).first;
    return it->second;
}

// $defs of the two schemas.
constexpr const char* kSha256 = "^[0-9a-f]{64}$";
constexpr const char* kGitCommit = "^[0-9a-f]{40}$";
constexpr const char* kMemberId = "^[a-z][a-z0-9_]{0,31}$";
constexpr const char* kRepoPath = "^(?!.*(?:^|/)\\.{1,2}(?:/|$))[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$";
constexpr const char* kBundleName = "^[a-z0-9][a-z0-9._-]{0,63}$";
constexpr const char* kSemver = "^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$";
constexpr const char* kDecisionRef =
    "^https://github\\.com/raylei50653/saccade/(issues|pull)/[0-9]+#issuecomment-[0-9]+$";
constexpr const char* kDate = "^[0-9]{4}-[0-9]{2}-[0-9]{2}$";

constexpr const char* kProvenanceReading =
    "Producer-recorded provenance. Not authenticated by the runtime; trusted only through the "
    "release-side allowlist entry for this manifest's sha256.";
constexpr const char* kRightsReading =
    "Release-review input owned by #547. The runtime never reads this block; nothing here asserts "
    "legal compliance or permission to redistribute.";
constexpr const char* kAllowlistReading =
    "Approved model-bundle manifests for this runtime build, by exact manifest file sha256. Engineering "
    "identity only: not publisher authentication of the bundle, not a rights grant, not a benchmark claim.";

using Kind = JsonValue::Kind;
using Names = std::vector<const char*>;

std::size_t code_points(const std::string& s) {
    std::size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

// JSON Schema "integer": an integral number of any magnitude (1.0 and 1e300
// count, as they do there). `out` is clamped to int64; callers compare Float
// values as doubles (Checker::integer).
bool integral(const JsonValue& v, std::int64_t* out = nullptr) {
    if (v.kind == Kind::Int) {
        if (out) *out = v.integer;
        return true;
    }
    if (v.kind == Kind::Float && std::isfinite(v.number) && std::floor(v.number) == v.number) {
        if (out) {
            *out = v.number >= 9223372036854775808.0    ? INT64_MAX
                   : v.number < -9223372036854775808.0 ? INT64_MIN
                                                        : static_cast<std::int64_t>(v.number);
        }
        return true;
    }
    return false;
}

bool is_number(const JsonValue& v) { return v.kind == Kind::Int || v.kind == Kind::Float; }

double number_of(const JsonValue& v) {
    return v.kind == Kind::Int ? static_cast<double>(v.integer) : v.number;
}

// Instance equality as JSON Schema has it for const / enum / uniqueItems:
// numbers compare by value (1 == 1.0); everything else structurally.
bool json_equal(const JsonValue& a, const JsonValue& b) {
    if (is_number(a) && is_number(b)) return number_of(a) == number_of(b);
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case Kind::Array:
            if (a.array.size() != b.array.size()) return false;
            for (std::size_t i = 0; i < a.array.size(); ++i) {
                if (!json_equal(a.array[i], b.array[i])) return false;
            }
            return true;
        case Kind::Object: {
            if (a.object.size() != b.object.size()) return false;
            for (const auto& [k, v] : a.object) {
                const JsonValue* w = b.find(k);
                if (w == nullptr || !json_equal(v, *w)) return false;
            }
            return true;
        }
        default: return a == b;
    }
}

std::string at(const std::string& path, const std::string& key) { return path + "/" + key; }
std::string at(const std::string& path, std::size_t i) { return path + "/" + std::to_string(i); }

// Collects issues; each check validates one schema keyword set at one
// instance path, the way the published schemas state it.
class Checker {
public:
    std::vector<BundleIssue> issues;

    void add(const std::string& path, const char* keyword, std::string message) {
        issues.push_back(BundleIssue{path, keyword, "", std::move(message)});
    }

    bool type(const JsonValue& v, const std::string& path, const char* type) {
        const std::string t = type;
        const bool ok = (t == "object" && v.kind == Kind::Object) || (t == "array" && v.kind == Kind::Array) ||
                        (t == "string" && v.kind == Kind::String) || (t == "integer" && integral(v)) ||
                        (t == "null" && v.kind == Kind::Null);
        if (!ok) add(path, "type", std::string(kind_name(v.kind)) + " is not of type " + t);
        return ok;
    }

    // type object + required + additionalProperties: false. True when `v` is
    // an object (its properties are then checked by the caller).
    bool object(const JsonValue& v, const std::string& path, const Names& required, const Names& allowed) {
        if (!type(v, path, "object")) return false;
        for (const char* r : required) {
            if (v.find(r) == nullptr) add(path, "required", std::string("'") + r + "' is a required property");
        }
        std::string extra;
        for (const auto& [k, value] : v.object) {
            bool known = false;
            for (const char* a : allowed) known = known || k == a;
            if (!known) extra += (extra.empty() ? "'" : ", '") + k + "'";
        }
        if (!extra.empty()) add(path, "additionalProperties", "additional properties are not allowed (" + extra + ")");
        return true;
    }

    void string(const JsonValue& v, const std::string& path, const char* pattern, std::size_t min_length = 0,
                std::size_t max_length = 0) {
        if (!type(v, path, "string")) return;
        if (max_length > 0 && code_points(v.string) > max_length) {
            add(path, "maxLength", "'" + v.string + "' is longer than " + std::to_string(max_length));
        }
        if (min_length > 0 && code_points(v.string) < min_length) {
            add(path, "minLength", "'" + v.string + "' is shorter than " + std::to_string(min_length));
        }
        if (pattern != nullptr && !std::regex_search(v.string, re(pattern))) {
            add(path, "pattern", "'" + v.string + "' does not match '" + pattern + "'");
        }
    }

    void string_const(const JsonValue& v, const std::string& path, const char* value) {
        if (!json_equal(v, JsonValue::make_string(value))) add(path, "const", std::string("must be '") + value + "'");
    }

    void string_enum(const JsonValue& v, const std::string& path, const Names& values) {
        for (const char* s : values) {
            if (json_equal(v, JsonValue::make_string(s))) return;
        }
        std::string all;
        for (const char* s : values) all += (all.empty() ? "'" : ", '") + std::string(s) + "'";
        add(path, "enum", "is not one of [" + all + "]");
    }

    void integer(const JsonValue& v, const std::string& path, std::int64_t minimum, std::int64_t multiple_of = 0) {
        if (!type(v, path, "integer")) return;
        // An integral Float is compared as a double: no int64 clamping.
        const bool below = v.kind == Kind::Int ? v.integer < minimum : v.number < static_cast<double>(minimum);
        const bool off = multiple_of > 0 && (v.kind == Kind::Int ? v.integer % multiple_of != 0
                                                                 : std::fmod(v.number, static_cast<double>(multiple_of)) != 0.0);
        const std::string shown = v.kind == Kind::Int ? std::to_string(v.integer) : python_float_repr(v.number);
        if (below) add(path, "minimum", shown + " is less than the minimum of " + std::to_string(minimum));
        if (off) add(path, "multipleOf", shown + " is not a multiple of " + std::to_string(multiple_of));
    }

    // type array + minItems / maxItems / uniqueItems. True when an array.
    bool array(const JsonValue& v, const std::string& path, std::size_t min_items = 0, std::size_t max_items = 0,
               bool unique = false) {
        if (!type(v, path, "array")) return false;
        if (v.array.size() < min_items) add(path, "minItems", "should be non-empty / has fewer than " + std::to_string(min_items) + " items");
        if (max_items > 0 && v.array.size() > max_items) add(path, "maxItems", "has more than " + std::to_string(max_items) + " items");
        if (unique) {
            for (std::size_t i = 0; i < v.array.size(); ++i) {
                for (std::size_t j = i + 1; j < v.array.size(); ++j) {
                    if (json_equal(v.array[i], v.array[j])) {
                        add(path, "uniqueItems", "has non-unique elements");
                        return true;
                    }
                }
            }
        }
        return true;
    }

    // oneOf: [{"type": "null"}, <alt>]: exactly one may match; null never
    // matches <alt> in these schemas, so this is "null or a valid <alt>".
    template <class F>
    void null_or(const JsonValue& v, const std::string& path, F&& alt) {
        if (v.kind == Kind::Null) return;
        Checker inner;
        alt(inner, v, path);
        if (!inner.issues.empty()) add(path, "oneOf", "is not valid under any of the given schemas");
    }
};

const JsonValue* field(const JsonValue& o, const char* key) { return o.kind == Kind::Object ? o.find(key) : nullptr; }

// ── saccade.model_bundle/v1 ───────────────────────────────────────────────────

void repo_path(Checker& c, const JsonValue& v, const std::string& p) { c.string(v, p, kRepoPath, 0, 255); }

void tensor(Checker& c, const JsonValue& t, const std::string& p) {
    if (!c.object(t, p, {"ordinal", "name", "shape", "dtype", "device"},
                  {"ordinal", "name", "shape", "dtype", "device"}))
        return;
    if (auto* v = field(t, "ordinal")) c.integer(*v, at(p, "ordinal"), 0);
    if (auto* v = field(t, "name")) c.string(*v, at(p, "name"), nullptr, 1);
    if (auto* v = field(t, "shape")) {
        if (c.array(*v, at(p, "shape"), 1)) {
            for (std::size_t i = 0; i < v->array.size(); ++i) c.integer(v->array[i], at(at(p, "shape"), i), 1);
        }
    }
    if (auto* v = field(t, "dtype")) {
        c.string_enum(*v, at(p, "dtype"), {"float32", "float16", "bfloat16", "int8", "int32", "int64", "uint8", "bool"});
    }
    if (auto* v = field(t, "device")) c.string_enum(*v, at(p, "device"), {"cuda"});
}

void io_spec(Checker& c, const JsonValue& s, const std::string& p) {
    if (!c.object(s, p, {"inputs", "outputs"}, {"inputs", "outputs"})) return;
    for (const char* k : {"inputs", "outputs"}) {
        const JsonValue* v = field(s, k);
        if (v == nullptr || !c.array(*v, at(p, k), 1)) continue;
        for (std::size_t i = 0; i < v->array.size(); ++i) tensor(c, v->array[i], at(at(p, k), i));
    }
}

void member(Checker& c, const JsonValue& m, const std::string& p) {
    if (!c.object(m, p, {"id", "role", "carried_by", "path", "bytes", "sha256", "inventory_id", "rights_audit_item"},
                  {"id", "role", "carried_by", "path", "bytes", "sha256", "json_schema", "inventory_id",
                   "rights_audit_item"}))
        return;
    if (auto* v = field(m, "id")) c.string(*v, at(p, "id"), kMemberId);
    if (auto* v = field(m, "role")) {
        c.string_enum(*v, at(p, "role"), {"backbone_engine", "head_torchscript", "scan_operator", "resolved_config",
                                          "head_lineage", "realization_attestation"});
    }
    if (auto* v = field(m, "carried_by")) c.string_enum(*v, at(p, "carried_by"), {"model_bundle", "runtime_package"});
    if (auto* v = field(m, "path")) repo_path(c, *v, at(p, "path"));
    if (auto* v = field(m, "bytes")) c.integer(*v, at(p, "bytes"), 1);
    if (auto* v = field(m, "sha256")) c.string(*v, at(p, "sha256"), kSha256);
    if (auto* v = field(m, "json_schema")) {
        c.string_enum(*v, at(p, "json_schema"), {"saccade.resolved_shipping_config/v1",
                                                 "saccade.head_artifact_lineage_torchscript/v1",
                                                 "saccade.head_realization_attestation/v1"});
    }
    if (auto* v = field(m, "inventory_id")) {
        c.null_or(*v, at(p, "inventory_id"),
                  [](Checker& k, const JsonValue& x, const std::string& q) { k.string(x, q, "^N0[1-9]$"); });
    }
    if (auto* v = field(m, "rights_audit_item")) {
        c.null_or(*v, at(p, "rights_audit_item"),
                  [](Checker& k, const JsonValue& x, const std::string& q) { k.string(x, q, "^[LM]-[0-9]+$"); });
    }
}

void target(Checker& c, const JsonValue& t, const std::string& p) {
    const Names keys = {"platform", "gpu_sm", "cuda_runtime", "tensorrt", "libtorch", "cudnn",
                        "glibc", "engine_precision", "qualification", "evidence"};
    if (!c.object(t, p, keys, keys)) return;
    if (auto* v = field(t, "platform")) c.string_enum(*v, at(p, "platform"), {"linux-x86_64"});
    if (auto* v = field(t, "gpu_sm")) {
        if (c.array(*v, at(p, "gpu_sm"), 1, 0, true)) {
            for (std::size_t i = 0; i < v->array.size(); ++i) c.string(v->array[i], at(at(p, "gpu_sm"), i), "^sm_[0-9]{2,3}$");
        }
    }
    if (auto* v = field(t, "cuda_runtime")) c.string(*v, at(p, "cuda_runtime"), "^[0-9]+\\.[0-9]+$");
    if (auto* v = field(t, "tensorrt")) c.string(*v, at(p, "tensorrt"), "^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$");
    if (auto* v = field(t, "libtorch")) c.string(*v, at(p, "libtorch"), nullptr, 1);
    if (auto* v = field(t, "cudnn")) c.string(*v, at(p, "cudnn"), "^[0-9]+$");
    if (auto* v = field(t, "glibc")) c.string(*v, at(p, "glibc"), "^[0-9]+\\.[0-9]+$");
    if (auto* v = field(t, "engine_precision")) {
        c.string_enum(*v, at(p, "engine_precision"), {"fp32", "fp16", "int8", "mixed", "unresolved"});
    }
    if (auto* v = field(t, "qualification")) {
        c.string_enum(*v, at(p, "qualification"), {"parity_recorded", "loadable_unqualified"});
    }
    const JsonValue* evidence = field(t, "evidence");
    if (evidence != nullptr && c.array(*evidence, at(p, "evidence"))) {
        for (std::size_t i = 0; i < evidence->array.size(); ++i) repo_path(c, evidence->array[i], at(at(p, "evidence"), i));
    }
    // if qualification == parity_recorded (an absent property satisfies the
    // `if`) then evidence minItems 1.
    const JsonValue* q = field(t, "qualification");
    if ((q == nullptr || json_equal(*q, JsonValue::make_string("parity_recorded"))) && evidence != nullptr &&
        evidence->kind == Kind::Array && evidence->array.empty()) {
        c.add(at(p, "evidence"), "minItems", "should be non-empty");
    }
}

void channel_decision(Checker& c, const JsonValue& d, const std::string& p) {
    if (!c.object(d, p, {"state", "decision_ref"}, {"state", "decision_ref"})) return;
    const JsonValue* state = field(d, "state");
    const JsonValue* ref = field(d, "decision_ref");
    if (state != nullptr) c.string_enum(*state, at(p, "state"), {"not_decided", "approved", "denied"});
    if (ref != nullptr) {
        c.null_or(*ref, at(p, "decision_ref"),
                  [](Checker& k, const JsonValue& x, const std::string& q) { k.string(x, q, kDecisionRef); });
    }
    const bool decided = state == nullptr || json_equal(*state, JsonValue::make_string("approved")) ||
                         json_equal(*state, JsonValue::make_string("denied"));
    if (ref != nullptr) c.type(*ref, at(p, "decision_ref"), decided ? "string" : "null");
}

void manifest_schema(Checker& c, const JsonValue& m) {
    const Names top = {"schema",        "bundle",         "runtime_contract", "members",
                       "pairing",       "io",             "preprocessing",    "postprocessing",
                       "compatibility", "provenance",     "rights",           "migration"};
    if (!c.object(m, "", top, top)) return;
    if (auto* v = field(m, "schema")) c.string_const(*v, "/schema", kModelBundleSchema);

    if (auto* b = field(m, "bundle"); b && c.object(*b, "/bundle", {"name", "version", "producer"},
                                                    {"name", "version", "producer"})) {
        if (auto* v = field(*b, "name")) c.string(*v, "/bundle/name", kBundleName);
        if (auto* v = field(*b, "version")) c.string(*v, "/bundle/version", kSemver);
        if (auto* pr = field(*b, "producer");
            pr && c.object(*pr, "/bundle/producer", {"tool", "tool_commit"}, {"tool", "tool_commit"})) {
            if (auto* v = field(*pr, "tool")) repo_path(c, *v, "/bundle/producer/tool");
            if (auto* v = field(*pr, "tool_commit")) c.string(*v, "/bundle/producer/tool_commit", kGitCommit);
        }
    }

    if (auto* rc = field(m, "runtime_contract");
        rc && c.object(*rc, "/runtime_contract", {"detector_contract", "requires_operator"},
                       {"detector_contract", "requires_operator"})) {
        if (auto* v = field(*rc, "detector_contract")) {
            c.string_enum(*v, "/runtime_contract/detector_contract", {kNativeDetectorContract});
        }
        if (auto* ro = field(*rc, "requires_operator");
            ro && c.object(*ro, "/runtime_contract/requires_operator", {"interface", "member"}, {"interface", "member"})) {
            if (auto* v = field(*ro, "interface")) {
                c.string_const(*v, "/runtime_contract/requires_operator/interface", "saccade_native::selective_scan_fwd");
            }
            if (auto* v = field(*ro, "member")) c.string(*v, "/runtime_contract/requires_operator/member", kMemberId);
        }
    }

    if (auto* ms = field(m, "members"); ms && c.array(*ms, "/members", 5, 16)) {
        for (std::size_t i = 0; i < ms->array.size(); ++i) member(c, ms->array[i], at("/members", i));
    }

    const Names slots = {"backbone_engine", "head", "operator", "config", "lineage", "attestation"};
    if (auto* pa = field(m, "pairing"); pa && c.object(*pa, "/pairing", slots, slots)) {
        for (const char* s : slots) {
            if (auto* v = field(*pa, s)) c.string(*v, at("/pairing", s), kMemberId);
        }
    }

    if (auto* io = field(m, "io"); io && c.object(*io, "/io", {"backbone_engine", "head"}, {"backbone_engine", "head"})) {
        if (auto* v = field(*io, "backbone_engine")) io_spec(c, *v, "/io/backbone_engine");
        if (auto* v = field(*io, "head")) io_spec(c, *v, "/io/head");
    }

    const Names pre = {"decode", "color_order", "value_scale", "resize", "layout", "normalization"};
    if (auto* pp = field(m, "preprocessing"); pp && c.object(*pp, "/preprocessing", pre, pre)) {
        if (auto* v = field(*pp, "decode")) c.string_enum(*v, "/preprocessing/decode", {"nvjpeg_rgb8"});
        if (auto* v = field(*pp, "color_order")) c.string_enum(*v, "/preprocessing/color_order", {"RGB"});
        if (auto* v = field(*pp, "value_scale")) c.string_enum(*v, "/preprocessing/value_scale", {"uint8_div_255"});
        const Names rk = {"mode", "width", "height", "interpolation", "align_corners"};
        if (auto* r = field(*pp, "resize"); r && c.object(*r, "/preprocessing/resize", rk, rk)) {
            if (auto* v = field(*r, "mode")) c.string_enum(*v, "/preprocessing/resize/mode", {"stretch"});
            if (auto* v = field(*r, "width")) c.integer(*v, "/preprocessing/resize/width", 32, 32);
            if (auto* v = field(*r, "height")) c.integer(*v, "/preprocessing/resize/height", 32, 32);
            if (auto* v = field(*r, "interpolation")) c.string_enum(*v, "/preprocessing/resize/interpolation", {"bilinear"});
            if (auto* v = field(*r, "align_corners"); v && !json_equal(*v, JsonValue::make_bool(false))) {
                c.add("/preprocessing/resize/align_corners", "const", "must be false");
            }
        }
        if (auto* v = field(*pp, "layout")) c.string_enum(*v, "/preprocessing/layout", {"NCHW"});
        if (auto* v = field(*pp, "normalization")) c.type(*v, "/preprocessing/normalization", "null");
    }

    const Names post = {"score_activation", "class_reduce", "box_encoding", "box_format", "num_classes", "box_channels"};
    if (auto* po = field(m, "postprocessing"); po && c.object(*po, "/postprocessing", post, post)) {
        if (auto* v = field(*po, "score_activation")) c.string_enum(*v, "/postprocessing/score_activation", {"sigmoid"});
        if (auto* v = field(*po, "class_reduce")) c.string_enum(*v, "/postprocessing/class_reduce", {"max"});
        if (auto* v = field(*po, "box_encoding")) c.string_enum(*v, "/postprocessing/box_encoding", {"ltrb_anchor_decode"});
        if (auto* v = field(*po, "box_format")) c.string_enum(*v, "/postprocessing/box_format", {"xyxy"});
        if (auto* v = field(*po, "num_classes")) c.integer(*v, "/postprocessing/num_classes", 1);
        if (auto* v = field(*po, "box_channels"); v && !json_equal(*v, JsonValue::make_int(4))) {
            c.add("/postprocessing/box_channels", "const", "must be 4");
        }
    }

    if (auto* co = field(m, "compatibility"); co && c.object(*co, "/compatibility", {"targets"}, {"targets"})) {
        if (auto* ts = field(*co, "targets"); ts && c.array(*ts, "/compatibility/targets", 1)) {
            for (std::size_t i = 0; i < ts->array.size(); ++i) target(c, ts->array[i], at("/compatibility/targets", i));
        }
    }

    const Names prov = {"reading", "inventory_ref", "training_lineage_ref", "sources"};
    if (auto* pv = field(m, "provenance"); pv && c.object(*pv, "/provenance", prov, prov)) {
        if (auto* v = field(*pv, "reading")) c.string_const(*v, "/provenance/reading", kProvenanceReading);
        if (auto* v = field(*pv, "inventory_ref")) repo_path(c, *v, "/provenance/inventory_ref");
        if (auto* v = field(*pv, "training_lineage_ref")) repo_path(c, *v, "/provenance/training_lineage_ref");
        if (auto* ss = field(*pv, "sources"); ss && c.array(*ss, "/provenance/sources")) {
            for (std::size_t i = 0; i < ss->array.size(); ++i) {
                const JsonValue& s = ss->array[i];
                const std::string p = at("/provenance/sources", i);
                if (!c.object(s, p, {"kind", "value", "status"}, {"kind", "value", "status"})) continue;
                if (auto* v = field(s, "kind")) {
                    c.string_enum(*v, at(p, "kind"), {"git_commit", "checkpoint_sha256", "upstream_model", "dataset"});
                }
                if (auto* v = field(s, "value")) c.string(*v, at(p, "value"), nullptr, 1);
                if (auto* v = field(s, "status")) c.string_enum(*v, at(p, "status"), {"recorded", "unresolved"});
            }
        }
    }

    const Names rights = {"reading", "decision_owner", "review_status", "channels"};
    if (auto* rt = field(m, "rights"); rt && c.object(*rt, "/rights", rights, rights)) {
        if (auto* v = field(*rt, "reading")) c.string_const(*v, "/rights/reading", kRightsReading);
        if (auto* v = field(*rt, "decision_owner")) c.string_const(*v, "/rights/decision_owner", "#547 release owner");
        if (auto* v = field(*rt, "review_status")) {
            c.string_enum(*v, "/rights/review_status", {"unreviewed", "open", "owner_decided"});
        }
        if (auto* ch = field(*rt, "channels");
            ch && c.object(*ch, "/rights/channels", {"private", "public"}, {"private", "public"})) {
            if (auto* v = field(*ch, "private")) channel_decision(c, *v, "/rights/channels/private");
            if (auto* v = field(*ch, "public")) channel_decision(c, *v, "/rights/channels/public");
        }
    }

    if (auto* mg = field(m, "migration");
        mg && c.object(*mg, "/migration", {"supersedes", "rollback_targets"}, {"supersedes", "rollback_targets"})) {
        if (auto* v = field(*mg, "supersedes")) {
            c.null_or(*v, "/migration/supersedes",
                      [](Checker& k, const JsonValue& x, const std::string& q) { k.string(x, q, kSha256); });
        }
        if (auto* rb = field(*mg, "rollback_targets"); rb && c.array(*rb, "/migration/rollback_targets", 0, 0, true)) {
            for (std::size_t i = 0; i < rb->array.size(); ++i) {
                c.string(rb->array[i], at("/migration/rollback_targets", i), kSha256);
            }
        }
    }
}

// ── R-01..R-08 ────────────────────────────────────────────────────────────────

struct PairingRole {
    const char* slot;
    const char* role;
    RootKind carrier;
};
// The pairing slot of each v1 role, and (R-07, ADR 028 D3) who carries it.
constexpr PairingRole kPairing[] = {
    {"backbone_engine", "backbone_engine", RootKind::ModelBundle},
    {"head", "head_torchscript", RootKind::ModelBundle},
    {"operator", "scan_operator", RootKind::RuntimePackage},
    {"config", "resolved_config", RootKind::ModelBundle},
    {"lineage", "head_lineage", RootKind::ModelBundle},
    {"attestation", "realization_attestation", RootKind::ModelBundle},
};

const char* json_schema_of(const std::string& role) {
    if (role == "resolved_config") return "saccade.resolved_shipping_config/v1";
    if (role == "head_lineage") return "saccade.head_artifact_lineage_torchscript/v1";
    if (role == "realization_attestation") return "saccade.head_realization_attestation/v1";
    return nullptr;
}

const RootKind* carrier_of(const std::string& role) {
    for (const PairingRole& r : kPairing) {
        if (role == r.role) return &r.carrier;
    }
    return nullptr;
}

std::vector<std::int64_t> ints(const JsonValue& shape) {
    std::vector<std::int64_t> out;
    for (const JsonValue& d : shape.array) {
        std::int64_t n = 0;
        integral(d, &n);
        out.push_back(n);
    }
    return out;
}

void rule(std::vector<BundleIssue>& out, const char* id, std::string message) {
    out.push_back(BundleIssue{"", "", id, std::move(message)});
}

std::string lower_ascii(std::string s) {
    for (char& ch : s) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return s;
}

[[noreturn]] void refuse(const char* what, const std::vector<BundleIssue>& issues) {
    std::string all;
    for (const BundleIssue& i : issues) all += (all.empty() ? "" : "; ") + describe(i);
    throw ConfigError(std::string(what) + ": " + all);
}

}  // namespace

std::string describe(const BundleIssue& i) {
    if (!i.rule.empty()) return i.rule + " " + i.message;
    return (i.instance_path.empty() ? std::string("/") : i.instance_path) + ": " + i.keyword + ": " + i.message;
}

std::vector<BundleIssue> model_bundle_schema_issues(const JsonValue& manifest) {
    Checker c;
    manifest_schema(c, manifest);
    return c.issues;
}

std::vector<BundleIssue> model_bundle_rule_issues(const JsonValue& m) {
    if (!model_bundle_schema_issues(m).empty()) throw std::logic_error("model_bundle_rule_issues: schema-invalid manifest");
    std::vector<BundleIssue> out;
    const auto& members = m.find("members")->array;
    std::map<std::string, const JsonValue*> by_id;
    for (const JsonValue& x : members) by_id.emplace(x.find("id")->string, &x);
    if (by_id.size() != members.size()) rule(out, "R-01", "duplicate member id");
    for (const char* carrier : {"model_bundle", "runtime_package"}) {
        std::set<std::string> folded;
        std::size_t n = 0;
        for (const JsonValue& x : members) {
            if (x.find("carried_by")->string != carrier) continue;
            folded.insert(lower_ascii(x.find("path")->string));
            ++n;
        }
        if (folded.size() != n) rule(out, "R-02", std::string("member paths collide under ") + carrier + " (case-folded)");
    }
    const JsonValue& pairing = *m.find("pairing");
    for (const PairingRole& r : kPairing) {
        auto it = by_id.find(pairing.find(r.slot)->string);
        if (it == by_id.end() || it->second->find("role")->string != r.role) {
            rule(out, "R-03", std::string("pairing.") + r.slot + " does not name a " + r.role + " member");
        }
    }
    std::multiset<std::string> roles, want;
    for (const JsonValue& x : members) roles.insert(x.find("role")->string);
    for (const PairingRole& r : kPairing) want.insert(r.role);
    if (roles != want) rule(out, "R-04", "v1 needs each role exactly once");
    if (m.find("runtime_contract")->find("requires_operator")->find("member")->string != pairing.find("operator")->string) {
        rule(out, "R-05", "requires_operator.member is not pairing.operator");
    }
    for (const JsonValue& x : members) {
        const std::string& role = x.find("role")->string;
        const char* want_schema = json_schema_of(role);
        const JsonValue* got = x.find("json_schema");
        if ((want_schema == nullptr) != (got == nullptr) || (got != nullptr && got->string != want_schema)) {
            rule(out, "R-06", "member " + x.find("id")->string + " json_schema does not match its role");
        }
        const RootKind* carrier = carrier_of(role);
        if (carrier != nullptr && x.find("carried_by")->string != root_kind_name(*carrier)) {
            rule(out, "R-07", role + " must be carried by " + root_kind_name(*carrier));
        }
    }
    const JsonValue& bb = *m.find("io")->find("backbone_engine");
    const JsonValue& head = *m.find("io")->find("head");
    const std::pair<const char*, const JsonValue*> lists[] = {{"backbone inputs", bb.find("inputs")},
                                                              {"backbone outputs", bb.find("outputs")},
                                                              {"head inputs", head.find("inputs")},
                                                              {"head outputs", head.find("outputs")}};
    for (const auto& [name, tensors] : lists) {
        for (std::size_t i = 0; i < tensors->array.size(); ++i) {
            std::int64_t ordinal = -1;
            integral(*tensors->array[i].find("ordinal"), &ordinal);
            if (ordinal != static_cast<std::int64_t>(i)) {
                rule(out, "R-08", std::string(name) + " ordinals are not 0..n-1");
                break;
            }
        }
    }
    const auto& head_in = head.find("inputs")->array;
    const auto& bb_out = bb.find("outputs")->array;
    bool same = head_in.size() == bb_out.size();
    for (std::size_t i = 0; same && i < head_in.size(); ++i) {
        same = ints(*head_in[i].find("shape")) == ints(*bb_out[i].find("shape")) &&
               head_in[i].find("dtype")->string == bb_out[i].find("dtype")->string;
    }
    if (!same) rule(out, "R-08", "head inputs do not match backbone outputs (shape, dtype)");
    const auto& head_out = head.find("outputs")->array;
    const std::size_t levels = head_in.size();
    if (head_out.size() != 2 * levels) {
        rule(out, "R-08", "head must emit one cls and one reg tensor per level");
    } else {
        const JsonValue& post = *m.find("postprocessing");
        std::int64_t num_classes = 0, box_channels = 0;
        integral(*post.find("num_classes"), &num_classes);
        integral(*post.find("box_channels"), &box_channels);
        for (std::size_t k = 0; k < head_out.size(); ++k) {
            const std::vector<std::int64_t> level = ints(*head_in[k % levels].find("shape"));
            std::vector<std::int64_t> want_shape{level[0], k < levels ? num_classes : box_channels};
            for (std::size_t d = 2; d < level.size(); ++d) want_shape.push_back(level[d]);
            if (ints(*head_out[k].find("shape")) != want_shape) {
                rule(out, "R-08", "head output " + head_out[k].find("name")->string + " shape is not the level's");
            }
        }
    }
    const JsonValue& resize = *m.find("preprocessing")->find("resize");
    const std::vector<std::int64_t> bb_in = ints(*bb.find("inputs")->array[0].find("shape"));
    std::int64_t height = 0, width = 0;
    integral(*resize.find("height"), &height);
    integral(*resize.find("width"), &width);
    if (std::vector<std::int64_t>(bb_in.begin() + std::min<std::size_t>(2, bb_in.size()), bb_in.end()) !=
        std::vector<std::int64_t>{height, width}) {
        rule(out, "R-08", "resize does not produce the backbone input H/W");
    }
    return out;
}

std::vector<BundleIssue> model_bundle_issues(const JsonValue& manifest) {
    std::vector<BundleIssue> issues = model_bundle_schema_issues(manifest);
    return issues.empty() ? model_bundle_rule_issues(manifest) : issues;
}

std::vector<BundleIssue> trusted_bundles_schema_issues(const JsonValue& a) {
    Checker c;
    const Names top = {"schema", "reading", "detector_contract", "entries"};
    if (!c.object(a, "", top, top)) return c.issues;
    if (auto* v = field(a, "schema")) c.string_const(*v, "/schema", kTrustedModelBundlesSchema);
    if (auto* v = field(a, "reading")) c.string_const(*v, "/reading", kAllowlistReading);
    if (auto* v = field(a, "detector_contract")) c.string_enum(*v, "/detector_contract", {kNativeDetectorContract});
    const JsonValue* entries = field(a, "entries");
    if (entries == nullptr || !c.array(*entries, "/entries")) return c.issues;
    const Names keys = {"manifest_sha256", "bundle_name", "bundle_version", "state", "approval", "revocation"};
    for (std::size_t i = 0; i < entries->array.size(); ++i) {
        const JsonValue& e = entries->array[i];
        const std::string p = at("/entries", i);
        if (!c.object(e, p, keys, keys)) continue;
        if (auto* v = field(e, "manifest_sha256")) c.string(*v, at(p, "manifest_sha256"), kSha256);
        if (auto* v = field(e, "bundle_name")) c.string(*v, at(p, "bundle_name"), kBundleName);
        if (auto* v = field(e, "bundle_version")) c.string(*v, at(p, "bundle_version"), nullptr, 1);
        const JsonValue* state = field(e, "state");
        if (state != nullptr) c.string_enum(*state, at(p, "state"), {"approved", "revoked", "example"});
        const JsonValue* approval = field(e, "approval");
        if (approval != nullptr) {
            c.null_or(*approval, at(p, "approval"), [](Checker& k, const JsonValue& x, const std::string& q) {
                if (!k.object(x, q, {"decision_ref", "date"}, {"decision_ref", "date"})) return;
                if (auto* v = field(x, "decision_ref")) k.string(*v, at(q, "decision_ref"), kDecisionRef);
                if (auto* v = field(x, "date")) k.string(*v, at(q, "date"), kDate);
            });
        }
        const JsonValue* revocation = field(e, "revocation");
        if (revocation != nullptr) {
            c.null_or(*revocation, at(p, "revocation"), [](Checker& k, const JsonValue& x, const std::string& q) {
                if (!k.object(x, q, {"decision_ref", "date", "reason"}, {"decision_ref", "date", "reason"})) return;
                if (auto* v = field(x, "decision_ref")) k.string(*v, at(q, "decision_ref"), kDecisionRef);
                if (auto* v = field(x, "date")) k.string(*v, at(q, "date"), kDate);
                if (auto* v = field(x, "reason")) k.string(*v, at(q, "reason"), nullptr, 1);
            });
        }
        // allOf: if state == X (an absent state satisfies every `if`) then
        // approval / revocation must have the listed types.
        struct When {
            const char* state;
            const char* approval;
            const char* revocation;
        };
        for (const When& w : {When{"approved", "object", "null"}, When{"revoked", "object", "object"},
                              When{"example", "null", "null"}}) {
            if (state != nullptr && !json_equal(*state, JsonValue::make_string(w.state))) continue;
            if (approval != nullptr) c.type(*approval, at(p, "approval"), w.approval);
            if (revocation != nullptr) c.type(*revocation, at(p, "revocation"), w.revocation);
        }
    }
    return c.issues;
}

std::vector<BundleIssue> trusted_bundles_rule_issues(const JsonValue& a) {
    if (!trusted_bundles_schema_issues(a).empty()) {
        throw std::logic_error("trusted_bundles_rule_issues: schema-invalid allowlist");
    }
    std::set<std::string> seen;
    std::vector<BundleIssue> out;
    for (const JsonValue& e : a.find("entries")->array) {
        if (!seen.insert(e.find("manifest_sha256")->string).second) {
            rule(out, "R-09", "duplicate manifest_sha256");
            break;
        }
    }
    return out;
}

std::vector<BundleIssue> trusted_bundles_issues(const JsonValue& allowlist) {
    std::vector<BundleIssue> issues = trusted_bundles_schema_issues(allowlist);
    return issues.empty() ? trusted_bundles_rule_issues(allowlist) : issues;
}

const char* root_kind_name(RootKind k) { return k == RootKind::ModelBundle ? "model_bundle" : "runtime_package"; }

ModelBundleManifest parse_model_bundle(const JsonValue& m) {
    const std::vector<BundleIssue> issues = model_bundle_issues(m);
    if (!issues.empty()) refuse("model bundle manifest", issues);
    ModelBundleManifest out;
    out.bundle_name = m.find("bundle")->find("name")->string;
    out.bundle_version = m.find("bundle")->find("version")->string;
    out.detector_contract = m.find("runtime_contract")->find("detector_contract")->string;
    std::map<std::string, std::size_t> by_id;
    const auto& members = m.find("members")->array;
    for (std::size_t i = 0; i < members.size(); ++i) {
        const JsonValue& x = members[i];
        BundleMember b;
        b.id = x.find("id")->string;
        b.role = x.find("role")->string;
        b.path = x.find("path")->string;
        b.sha256 = x.find("sha256")->string;
        if (const JsonValue* s = x.find("json_schema")) b.json_schema = s->string;
        b.carried_by = x.find("carried_by")->string == "model_bundle" ? RootKind::ModelBundle : RootKind::RuntimePackage;
        integral(*x.find("bytes"), &b.bytes);
        b.index = i;
        by_id[b.id] = i;
        out.members.push_back(std::move(b));
    }
    const JsonValue& pairing = *m.find("pairing");
    out.backbone_engine = by_id.at(pairing.find("backbone_engine")->string);
    out.head = by_id.at(pairing.find("head")->string);
    out.op_library = by_id.at(pairing.find("operator")->string);
    out.config = by_id.at(pairing.find("config")->string);
    out.lineage = by_id.at(pairing.find("lineage")->string);
    out.attestation = by_id.at(pairing.find("attestation")->string);
    return out;
}

const char* allowlist_state_name(AllowlistState s) {
    switch (s) {
        case AllowlistState::Absent: return "absent";
        case AllowlistState::Example: return "example";
        case AllowlistState::Revoked: return "revoked";
        case AllowlistState::Approved: return "approved";
    }
    return "?";
}

TrustedModelBundles parse_trusted_bundles(const JsonValue& a) {
    const std::vector<BundleIssue> issues = trusted_bundles_issues(a);
    if (!issues.empty()) refuse("trusted model bundles", issues);
    TrustedModelBundles out;
    out.detector_contract = a.find("detector_contract")->string;
    const auto& entries = a.find("entries")->array;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        out.entries.push_back(
            TrustedBundleEntry{entries[i].find("manifest_sha256")->string, entries[i].find("state")->string, i});
    }
    return out;
}

AllowlistState allowlist_state(const TrustedModelBundles& a, const std::string& manifest_sha256, std::size_t* index) {
    for (const TrustedBundleEntry& e : a.entries) {
        if (e.manifest_sha256 != manifest_sha256) continue;
        if (index) *index = e.index;
        if (e.state == "approved") return AllowlistState::Approved;
        if (e.state == "revoked") return AllowlistState::Revoked;
        return AllowlistState::Example;
    }
    return AllowlistState::Absent;
}

// ── VL1 ───────────────────────────────────────────────────────────────────────

UniqueFd::~UniqueFd() {
    if (fd_ >= 0) ::close(fd_);
}

UniqueFd& UniqueFd::operator=(UniqueFd&& o) noexcept {
    if (this != &o) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = o.release();
    }
    return *this;
}

BundleRoot BundleRoot::open(const std::string& dir, RootKind kind, const std::string& what) {
    if (dir.empty()) throw ConfigError(what + " is not given");
    char* real = ::realpath(dir.c_str(), nullptr);
    if (real == nullptr) {
        throw ConfigError(what + " " + dir + " cannot be resolved: " + std::strerror(errno));
    }
    BundleRoot r;
    r.real_ = real;
    std::free(real);
    r.kind_ = kind;
    r.fd_ = UniqueFd(::open(r.real_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (r.fd_.get() < 0) {
        throw ConfigError(what + " " + r.real_ + " is not an openable directory: " + std::strerror(errno));
    }
    return r;
}

std::string BundleRoot::absolute(const std::string& rel) const {
    return real_ == "/" ? "/" + rel : real_ + "/" + rel;
}

BeneathOpen open_beneath(const BundleRoot& root, const std::string& rel) {
    BeneathOpen out;
    open_how how{};
    how.flags = O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC;
    how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
    long fd = -1;
    for (int attempt = 0; attempt < 3; ++attempt) {
        fd = ::syscall(SYS_openat2, root.fd(), rel.c_str(), &how, sizeof how);
        if (fd >= 0 || errno != EAGAIN) break;  // EAGAIN: a concurrent rename; retry
    }
    if (fd < 0) {
        const int e = errno;
        const std::string where = root.absolute(rel);
        switch (e) {
            case ENOSYS:
            case E2BIG:
            case EPERM:
                throw SecureOpenUnsupported("secure open (openat2 RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS) of " + where +
                                            " is not supported here: " + std::strerror(e) +
                                            "; manifest mode has no weaker fallback");
            case ENOENT:
            case ENOTDIR:
                out.result = BeneathOpen::Result::Missing;
                out.reason = where + " does not exist";
                return out;
            case ELOOP:
                out.result = BeneathOpen::Result::Unsafe;
                out.reason = where + " is or passes through a symbolic link";
                return out;
            case EXDEV:
                out.result = BeneathOpen::Result::Unsafe;
                out.reason = where + " resolves outside its root";
                return out;
            default:
                throw ConfigError("cannot open " + where + ": " + std::strerror(e));
        }
    }
    out.fd = UniqueFd(static_cast<int>(fd));
    struct stat st {};
    if (::fstat(out.fd.get(), &st) != 0) throw ConfigError("cannot stat " + root.absolute(rel) + ": " + std::strerror(errno));
    if (!S_ISREG(st.st_mode)) {
        out.result = BeneathOpen::Result::NotRegular;
        out.reason = root.absolute(rel) + " is not a regular file";
        out.fd = UniqueFd();
        return out;
    }
    out.result = BeneathOpen::Result::Opened;
    out.size = static_cast<std::int64_t>(st.st_size);
    return out;
}

namespace {

// Reads `fd` from offset 0 to EOF, handing each chunk to `sink`; returns the
// byte count.
template <class Sink>
std::int64_t read_chunks(int fd, const std::string& what, Sink&& sink) {
    std::vector<char> buf(1 << 20);
    std::int64_t total = 0;
    off_t off = 0;
    for (;;) {
        const ssize_t n = ::pread(fd, buf.data(), buf.size(), off);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw ConfigError("cannot read " + what + ": " + std::strerror(errno));
        }
        if (n == 0) return total;
        sink(buf.data(), static_cast<std::size_t>(n));
        total += n;
        off += n;
    }
}

void require_size(std::int64_t got, std::int64_t want, const std::string& what) {
    if (got != want) {
        throw ConfigError(what + " changed while it was read (" + std::to_string(got) + " bytes, " +
                          std::to_string(want) + " at open)");
    }
}

}  // namespace

std::string read_all(int fd, std::int64_t expected_size, const std::string& what) {
    std::string bytes;
    const std::int64_t n = read_chunks(fd, what, [&](const char* p, std::size_t k) { bytes.append(p, k); });
    require_size(n, expected_size, what);
    return bytes;
}

std::string sha256_fd(int fd, std::int64_t expected_size, const std::string& what) {
    Sha256 h;
    const std::int64_t n = read_chunks(fd, what, [&](const char* p, std::size_t k) { h.update(p, k); });
    require_size(n, expected_size, what);
    return h.hex_digest();
}

}  // namespace saccade::shipping
