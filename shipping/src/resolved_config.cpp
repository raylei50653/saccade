#include "saccade_shipping/resolved_config.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <type_traits>

namespace saccade::shipping {

namespace {

constexpr std::string_view kPerSequenceKey = "per_sequence";
constexpr std::string_view kImWidth = "seqinfo.ini:Sequence.imWidth";
constexpr std::string_view kImHeight = "seqinfo.ini:Sequence.imHeight";

std::string join(const std::string& path, std::string_view key) {
    return path.empty() ? std::string(key) : path + "." + std::string(key);
}

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw ConfigError(path + ": " + what);
}

void expect_kind(const JsonValue& v, JsonValue::Kind kind, const std::string& path) {
    if (v.kind != kind) {
        fail(path, std::string("expected ") + kind_name(kind) + ", got " + kind_name(v.kind));
    }
}

const char* per_sequence_name(PerSequenceValue v) {
    return v == PerSequenceValue::ImWidth ? "imWidth" : "imHeight";
}

// ─── leaf reads ───────────────────────────────────────────────────────────

void read_value(const JsonValue& v, bool& out, const std::string& path) {
    expect_kind(v, JsonValue::Kind::Bool, path);
    out = v.boolean;
}

void read_value(const JsonValue& v, std::int64_t& out, const std::string& path) {
    expect_kind(v, JsonValue::Kind::Int, path);
    out = v.integer;
}

void read_value(const JsonValue& v, double& out, const std::string& path) {
    expect_kind(v, JsonValue::Kind::Float, path);  // an int literal is not a float
    out = v.number;
}

void read_value(const JsonValue& v, std::string& out, const std::string& path) {
    expect_kind(v, JsonValue::Kind::String, path);
    out = v.string;
}

void read_value(const JsonValue& v, NullValue&, const std::string& path) {
    expect_kind(v, JsonValue::Kind::Null, path);
}

void read_value(const JsonValue& v, std::optional<std::string>& out, const std::string& path) {
    if (v.kind == JsonValue::Kind::Null) {
        out.reset();
        return;
    }
    if (v.kind != JsonValue::Kind::String) {
        fail(path, std::string("expected string or null, got ") + kind_name(v.kind));
    }
    out = v.string;
}

void read_value(const JsonValue& v, PerSequenceValue& out, const std::string& path) {
    expect_kind(v, JsonValue::Kind::Object, path);
    for (const auto& [key, _] : v.object) {
        if (key != kPerSequenceKey) fail(join(path, key), "unknown field");
    }
    const JsonValue* ref = v.find(kPerSequenceKey);
    const std::string ref_path = join(path, kPerSequenceKey);
    if (ref == nullptr) fail(ref_path, "missing required field");
    expect_kind(*ref, JsonValue::Kind::String, ref_path);
    if (ref->string == kImWidth) {
        out = PerSequenceValue::ImWidth;
    } else if (ref->string == kImHeight) {
        out = PerSequenceValue::ImHeight;
    } else {
        fail(ref_path, "unknown per-sequence source \"" + ref->string + "\"");
    }
}

template <class E>
void read_list(const JsonValue& v, std::vector<E>& out, const std::string& path) {
    expect_kind(v, JsonValue::Kind::Array, path);
    out.clear();
    for (std::size_t k = 0; k < v.array.size(); ++k) {
        E element{};
        read_value(v.array[k], element, path + "[" + std::to_string(k) + "]");
        out.push_back(element);
    }
}

void read_value(const JsonValue& v, IntList& out, const std::string& path) { read_list(v, out, path); }
void read_value(const JsonValue& v, FloatList& out, const std::string& path) { read_list(v, out, path); }
void read_value(const JsonValue& v, StringList& out, const std::string& path) { read_list(v, out, path); }
void read_value(const JsonValue& v, PerSequenceList& out, const std::string& path) {
    read_list(v, out, path);
}
void read_value(const JsonValue& v, BoolList& out, const std::string& path) {
    expect_kind(v, JsonValue::Kind::Array, path);
    out.clear();
    for (std::size_t k = 0; k < v.array.size(); ++k) {
        bool element = false;
        read_value(v.array[k], element, path + "[" + std::to_string(k) + "]");
        out.push_back(element);
    }
}

template <class S> void read_value(const JsonValue& v, S& out, const std::string& path);

// ─── value checks ─────────────────────────────────────────────────────────

template <class T> void apply_check(const T&, const check::Any&, const std::string&) {}

void apply_check(double x, const check::FloatRange& r, const std::string& path) {
    const bool lo_ok = r.lo_open ? x > r.lo : x >= r.lo;
    const bool hi_ok = r.hi_open ? x < r.hi : x <= r.hi;
    if (!lo_ok || !hi_ok) {
        const std::string hi = r.hi == kInf ? "inf" : python_float_repr(r.hi);
        fail(path, python_float_repr(x) + " outside " + (r.lo_open ? "(" : "[") +
                       python_float_repr(r.lo) + ", " + hi + (r.hi_open || r.hi == kInf ? ")" : "]"));
    }
}

void apply_check(std::int64_t x, const check::IntRange& r, const std::string& path) {
    if (x < r.lo || x > r.hi) {
        fail(path, std::to_string(x) + " outside [" + std::to_string(r.lo) + ", " +
                       std::to_string(r.hi) + "]");
    }
}

void apply_check(std::int64_t x, const check::IntOneOf& c, const std::string& path) {
    if (std::find(c.values.begin(), c.values.end(), x) == c.values.end()) {
        std::string allowed;
        for (auto a : c.values) allowed += (allowed.empty() ? "" : ", ") + std::to_string(a);
        fail(path, std::to_string(x) + " is not one of {" + allowed + "}");
    }
}

void apply_check(const std::string& x, const check::StringOneOf& c, const std::string& path) {
    if (std::find(c.values.begin(), c.values.end(), x) == c.values.end()) {
        std::string allowed;
        for (const auto& a : c.values) allowed += (allowed.empty() ? "\"" : ", \"") + a + "\"";
        fail(path, "\"" + x + "\" is not one of {" + allowed + "}");
    }
}

void apply_check(const std::optional<std::string>& x, const check::StringOneOf& c,
                 const std::string& path) {
    if (x) apply_check(*x, c, path);
}

void apply_check(const std::string& x, const check::NonEmpty&, const std::string& path) {
    if (x.empty()) fail(path, "must not be empty");
}

void apply_check(const std::string& x, const check::Sha256Hex&, const std::string& path) {
    const bool ok = x.size() == 64 && std::all_of(x.begin(), x.end(), [](char c) {
                        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                    });
    if (!ok) fail(path, "must be 64 lowercase hex digits");
}

void apply_check(bool x, const check::BoolIs& c, const std::string& path) {
    if (x != c.value) {
        fail(path, std::string("must be ") + (c.value ? "true" : "false") +
                       " (no shipping implementation for the other value)");
    }
}

void apply_check(PerSequenceValue x, const check::PerSequenceIs& c, const std::string& path) {
    if (x != c.value) {
        fail(path, std::string("must be the per-sequence ") + per_sequence_name(c.value) + ", got " +
                       per_sequence_name(x));
    }
}

void apply_check(const PerSequenceList& x, const check::PerSequenceListIs& c, const std::string& path) {
    if (x != c.values) {
        std::string want;
        for (auto e : c.values) want += (want.empty() ? "" : ", ") + std::string(per_sequence_name(e));
        fail(path, "must be the per-sequence values [" + want + "] in that order");
    }
}

void apply_check(const StringList& x, const check::PermutationOf& c, const std::string& path) {
    std::vector<std::string> seen;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const std::string p = path + "[" + std::to_string(k) + "]";
        if (std::find(c.values.begin(), c.values.end(), x[k]) == c.values.end()) {
            fail(p, "unknown entry \"" + x[k] + "\"");
        }
        if (std::find(seen.begin(), seen.end(), x[k]) != seen.end()) {
            fail(p, "duplicate entry \"" + x[k] + "\"");
        }
        seen.push_back(x[k]);
    }
    if (x.size() != c.values.size()) {
        fail(path, "must list all " + std::to_string(c.values.size()) + " entries exactly once");
    }
}

void apply_check(const FloatList& x, const check::FloatListOf& c, const std::string& path) {
    if (x.size() != c.size) fail(path, "must have exactly " + std::to_string(c.size) + " elements");
    for (std::size_t k = 0; k < x.size(); ++k) {
        apply_check(x[k], c.element, path + "[" + std::to_string(k) + "]");
    }
}

void apply_check(const BoolList& x, const check::BoolListSize& c, const std::string& path) {
    if (x.size() != c.size) fail(path, "must have exactly " + std::to_string(c.size) + " elements");
}

// ─── reader visitors ──────────────────────────────────────────────────────

class ObjectReader {
public:
    ObjectReader(const JsonValue& obj, std::string path) : obj_(obj), path_(std::move(path)) {
        expect_kind(obj_, JsonValue::Kind::Object, path_.empty() ? "<document>" : path_);
        used_.assign(obj_.object.size(), false);
    }

    template <class T, class C = check::Any>
    void field(std::string_view key, T& out, const C& c = {}) {
        const std::string path = join(path_, key);
        for (std::size_t k = 0; k < obj_.object.size(); ++k) {
            if (obj_.object[k].first == key) {
                used_[k] = true;
                read_value(obj_.object[k].second, out, path);
                apply_check(out, c, path);
                return;
            }
        }
        fail(path, "missing required field");
    }

    void finish() const {
        for (std::size_t k = 0; k < obj_.object.size(); ++k) {
            if (!used_[k]) fail(join(path_, obj_.object[k].first), "unknown field");
        }
    }

private:
    const JsonValue& obj_;
    std::string path_;
    std::vector<bool> used_;
};

class CallReader {
public:
    CallReader(const JsonValue& arr, std::string path) : arr_(arr), path_(std::move(path)) {
        expect_kind(arr_, JsonValue::Kind::Array, path_);
    }

    template <class A, class C = check::Any>
    void call(std::string_view method, A& args, const C& c = {}) {
        const std::string path = path_ + "[" + std::to_string(next_) + "]";
        if (next_ >= arr_.array.size()) fail(path, "missing call " + std::string(method));
        const JsonValue& record = arr_.array[next_++];
        expect_kind(record, JsonValue::Kind::Object, path);
        for (const auto& [key, _] : record.object) {
            if (key != "method" && key != "args") fail(join(path, key), "unknown field");
        }
        const JsonValue* name = record.find("method");
        if (name == nullptr) fail(join(path, "method"), "missing required field");
        expect_kind(*name, JsonValue::Kind::String, join(path, "method"));
        if (name->string != method) {
            fail(join(path, "method"), "expected \"" + std::string(method) + "\", got \"" +
                                           name->string + "\" (call order is schema)");
        }
        const JsonValue* raw_args = record.find("args");
        const std::string args_path = path + "(" + std::string(method) + ").args";
        if (raw_args == nullptr) fail(args_path, "missing required field");
        read_value(*raw_args, args, args_path);
        apply_check(args, c, args_path);
    }

    void finish() const {
        if (next_ != arr_.array.size()) {
            fail(path_ + "[" + std::to_string(next_) + "]", "unexpected extra call");
        }
    }

private:
    const JsonValue& arr_;
    std::string path_;
    std::size_t next_ = 0;
};

template <class S> void read_value(const JsonValue& v, S& out, const std::string& path) {
    if constexpr (std::is_base_of_v<CallSequence, S>) {
        CallReader reader(v, path);
        out.visit(reader);
        reader.finish();
    } else {
        ObjectReader reader(v, path);
        out.visit(reader);
        reader.finish();
    }
}

// ─── writers ──────────────────────────────────────────────────────────────

JsonValue write_value(bool v) { return JsonValue::make_bool(v); }
JsonValue write_value(std::int64_t v) { return JsonValue::make_int(v); }
JsonValue write_value(double v) { return JsonValue::make_float(v); }
JsonValue write_value(const std::string& v) { return JsonValue::make_string(v); }
JsonValue write_value(const NullValue&) { return JsonValue::make_null(); }
JsonValue write_value(const std::optional<std::string>& v) {
    return v ? JsonValue::make_string(*v) : JsonValue::make_null();
}
JsonValue write_value(PerSequenceValue v) {
    JsonValue obj = JsonValue::make_object();
    obj.set(std::string(kPerSequenceKey),
            JsonValue::make_string(std::string(v == PerSequenceValue::ImWidth ? kImWidth : kImHeight)));
    return obj;
}
template <class E> JsonValue write_list(const std::vector<E>& v) {
    JsonValue arr = JsonValue::make_array();
    for (const auto& e : v) arr.array.push_back(write_value(e));
    return arr;
}
JsonValue write_value(const IntList& v) { return write_list(v); }
JsonValue write_value(const FloatList& v) { return write_list(v); }
JsonValue write_value(const StringList& v) { return write_list(v); }
JsonValue write_value(const PerSequenceList& v) { return write_list(v); }
JsonValue write_value(const BoolList& v) {
    JsonValue arr = JsonValue::make_array();
    for (bool e : v) arr.array.push_back(JsonValue::make_bool(e));
    return arr;
}

template <class S> JsonValue write_value(const S& s);

class ObjectWriter {
public:
    template <class T, class C = check::Any>
    void field(std::string_view key, T& value, const C& = {}) {
        out.set(std::string(key), write_value(static_cast<const T&>(value)));
    }
    JsonValue out = JsonValue::make_object();
};

class CallWriter {
public:
    template <class A, class C = check::Any>
    void call(std::string_view method, A& args, const C& = {}) {
        JsonValue record = JsonValue::make_object();
        record.set("method", JsonValue::make_string(std::string(method)));
        record.set("args", write_value(static_cast<const A&>(args)));
        out.array.push_back(std::move(record));
    }
    JsonValue out = JsonValue::make_array();
};

template <class S> JsonValue write_value(const S& s) {
    // visit() is non-const because the reader shares it; writers only read.
    S& walk = const_cast<S&>(s);
    if constexpr (std::is_base_of_v<CallSequence, S>) {
        CallWriter writer;
        walk.visit(writer);
        return std::move(writer.out);
    } else {
        ObjectWriter writer;
        walk.visit(writer);
        return std::move(writer.out);
    }
}

}  // namespace

ResolvedShippingConfig load_resolved_shipping_config(const JsonValue& document) {
    // Check the version before anything else so a different schema fails as
    // such rather than as a field-level mismatch.
    expect_kind(document, JsonValue::Kind::Object, "<document>");
    const JsonValue* schema = document.find("schema");
    if (schema == nullptr) fail("schema", "missing required field");
    expect_kind(*schema, JsonValue::Kind::String, "schema");
    if (schema->string != kResolvedConfigSchema) {
        fail("schema", "unsupported schema \"" + schema->string + "\" (this loader reads exactly \"" +
                           std::string(kResolvedConfigSchema) + "\")");
    }
    ResolvedShippingConfig config;
    ObjectReader reader(document, "");
    config.visit(reader);
    reader.finish();
    return config;
}

ResolvedShippingConfig parse_resolved_shipping_config(std::string_view json_text) {
    return load_resolved_shipping_config(parse_strict_json(json_text));
}

ResolvedShippingConfig load_resolved_shipping_config_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError("cannot open resolved config " + path);
    std::ostringstream text;
    text << in.rdbuf();
    if (!in.good() && !in.eof()) throw ConfigError("cannot read resolved config " + path);
    return parse_resolved_shipping_config(text.str());
}

JsonValue to_json(const ResolvedShippingConfig& config) { return write_value(config); }

std::string canonical_snapshot(const ResolvedShippingConfig& config) {
    return dump_python_json(to_json(config)) + "\n";
}

}  // namespace saccade::shipping
