// Strict JSON reader/writer for the shipping resolved config (#465 PR-4a).
//
// Reader: RFC 8259 JSON with no extensions, plus the rules the resolved config
// needs to stay unambiguous -- duplicate object keys are rejected, NaN/Infinity
// tokens and numbers that overflow a double are rejected, and a number literal
// keeps its kind (no '.', 'e' or 'E' => Int, else Float), which is how the
// exporter's `json.dumps` distinguishes Python int from float.
//
// Writer: byte-identical to Python `json.dumps(v, indent=2, ensure_ascii=True,
// allow_nan=False)` for the values the reader produces (object key order is
// kept; floats use Python `repr`).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace saccade::shipping {

// Every failure of the reader or the typed loader. `what()` names the JSON
// path (or byte offset for syntax errors) and the violated rule.
class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct JsonValue {
    enum class Kind { Null, Bool, Int, Float, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    std::int64_t integer = 0;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;  // file order

    static JsonValue make_null() { return {}; }
    static JsonValue make_bool(bool v);
    static JsonValue make_int(std::int64_t v);
    static JsonValue make_float(double v);
    static JsonValue make_string(std::string v);
    static JsonValue make_array(std::vector<JsonValue> v = {});
    static JsonValue make_object();

    // Object helpers. `find` returns nullptr when the key is absent.
    const JsonValue* find(std::string_view key) const;
    JsonValue* find(std::string_view key);
    void set(std::string key, JsonValue value);  // append; caller ensures uniqueness
    bool erase(std::string_view key);

    bool operator==(const JsonValue& other) const;
    bool operator!=(const JsonValue& other) const { return !(*this == other); }
};

const char* kind_name(JsonValue::Kind kind);

JsonValue parse_strict_json(std::string_view text);

// `json.dumps(v, indent=2, ensure_ascii=True, allow_nan=False)` without the
// trailing newline the exporter appends.
std::string dump_python_json(const JsonValue& value);

// Python `repr(float)`; throws ConfigError for NaN/Inf.
std::string python_float_repr(double value);

}  // namespace saccade::shipping
