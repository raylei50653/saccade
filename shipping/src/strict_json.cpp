#include "saccade_shipping/strict_json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <system_error>

namespace saccade::shipping {

JsonValue JsonValue::make_bool(bool v) {
    JsonValue j;
    j.kind = Kind::Bool;
    j.boolean = v;
    return j;
}

JsonValue JsonValue::make_int(std::int64_t v) {
    JsonValue j;
    j.kind = Kind::Int;
    j.integer = v;
    return j;
}

JsonValue JsonValue::make_float(double v) {
    JsonValue j;
    j.kind = Kind::Float;
    j.number = v;
    return j;
}

JsonValue JsonValue::make_string(std::string v) {
    JsonValue j;
    j.kind = Kind::String;
    j.string = std::move(v);
    return j;
}

JsonValue JsonValue::make_array(std::vector<JsonValue> v) {
    JsonValue j;
    j.kind = Kind::Array;
    j.array = std::move(v);
    return j;
}

JsonValue JsonValue::make_object() {
    JsonValue j;
    j.kind = Kind::Object;
    return j;
}

const JsonValue* JsonValue::find(std::string_view key) const {
    for (const auto& [k, v] : object) {
        if (k == key) return &v;
    }
    return nullptr;
}

JsonValue* JsonValue::find(std::string_view key) {
    for (auto& [k, v] : object) {
        if (k == key) return &v;
    }
    return nullptr;
}

void JsonValue::set(std::string key, JsonValue value) {
    object.emplace_back(std::move(key), std::move(value));
}

bool JsonValue::erase(std::string_view key) {
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (it->first == key) {
            object.erase(it);
            return true;
        }
    }
    return false;
}

bool JsonValue::operator==(const JsonValue& other) const {
    if (kind != other.kind) return false;
    switch (kind) {
        case Kind::Null: return true;
        case Kind::Bool: return boolean == other.boolean;
        case Kind::Int: return integer == other.integer;
        case Kind::Float:
            return number == other.number && std::signbit(number) == std::signbit(other.number);
        case Kind::String: return string == other.string;
        case Kind::Array: return array == other.array;
        case Kind::Object: return object == other.object;
    }
    return false;
}

const char* kind_name(JsonValue::Kind kind) {
    switch (kind) {
        case JsonValue::Kind::Null: return "null";
        case JsonValue::Kind::Bool: return "bool";
        case JsonValue::Kind::Int: return "int";
        case JsonValue::Kind::Float: return "float";
        case JsonValue::Kind::String: return "string";
        case JsonValue::Kind::Array: return "array";
        case JsonValue::Kind::Object: return "object";
    }
    return "?";
}

// ─── reader ───────────────────────────────────────────────────────────────

namespace {

constexpr int kMaxDepth = 64;

class Reader {
public:
    explicit Reader(std::string_view text) : s_(text) {}

    JsonValue document() {
        skip_ws();
        JsonValue v = value(0);
        skip_ws();
        if (i_ != s_.size()) fail("trailing content after the JSON document");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw ConfigError("json syntax at byte " + std::to_string(i_) + ": " + what);
    }

    bool at_end() const { return i_ >= s_.size(); }
    char peek() const { return at_end() ? '\0' : s_[i_]; }

    void skip_ws() {
        while (!at_end() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) {
            ++i_;
        }
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++i_;
    }

    void literal(std::string_view word) {
        if (s_.substr(i_, word.size()) != word) fail("invalid literal");
        i_ += word.size();
    }

    JsonValue value(int depth) {
        if (depth > kMaxDepth) fail("nesting deeper than " + std::to_string(kMaxDepth));
        switch (peek()) {
            case '{': return object(depth);
            case '[': return array(depth);
            case '"': return JsonValue::make_string(string());
            case 't': literal("true"); return JsonValue::make_bool(true);
            case 'f': literal("false"); return JsonValue::make_bool(false);
            case 'n': literal("null"); return JsonValue::make_null();
            default:
                if (peek() == '-' || (peek() >= '0' && peek() <= '9')) return number();
                fail("unexpected character (NaN/Infinity are not JSON)");
        }
    }

    JsonValue object(int depth) {
        expect('{');
        JsonValue obj = JsonValue::make_object();
        skip_ws();
        if (peek() == '}') {
            ++i_;
            return obj;
        }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("expected an object key string");
            std::string key = string();
            if (obj.find(key) != nullptr) fail("duplicate object key \"" + key + "\"");
            skip_ws();
            expect(':');
            skip_ws();
            obj.set(std::move(key), value(depth + 1));
            skip_ws();
            if (peek() == ',') {
                ++i_;
                continue;
            }
            expect('}');
            return obj;
        }
    }

    JsonValue array(int depth) {
        expect('[');
        JsonValue arr = JsonValue::make_array();
        skip_ws();
        if (peek() == ']') {
            ++i_;
            return arr;
        }
        while (true) {
            skip_ws();
            arr.array.push_back(value(depth + 1));
            skip_ws();
            if (peek() == ',') {
                ++i_;
                continue;
            }
            expect(']');
            return arr;
        }
    }

    JsonValue number() {
        const std::size_t start = i_;
        bool is_float = false;
        if (peek() == '-') ++i_;
        if (peek() == '0') {
            ++i_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++i_;
        } else {
            fail("invalid number");
        }
        if (peek() == '.') {
            is_float = true;
            ++i_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number fraction");
            while (peek() >= '0' && peek() <= '9') ++i_;
        }
        if (peek() == 'e' || peek() == 'E') {
            is_float = true;
            ++i_;
            if (peek() == '+' || peek() == '-') ++i_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number exponent");
            while (peek() >= '0' && peek() <= '9') ++i_;
        }
        const char* first = s_.data() + start;
        const char* last = s_.data() + i_;
        if (is_float) {
            double v = 0.0;
            auto [ptr, ec] = std::from_chars(first, last, v);
            if (ec != std::errc() || ptr != last || !std::isfinite(v)) {
                fail("number " + std::string(first, last) + " is not a finite double");
            }
            return JsonValue::make_float(v);
        }
        std::int64_t v = 0;
        auto [ptr, ec] = std::from_chars(first, last, v);
        if (ec != std::errc() || ptr != last) {
            fail("integer " + std::string(first, last) + " does not fit int64");
        }
        return JsonValue::make_int(v);
    }

    unsigned hex4() {
        if (i_ + 4 > s_.size()) fail("truncated \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    // Copies one raw UTF-8 sequence, rejecting malformed/overlong/surrogate forms.
    void raw_utf8(std::string& out) {
        const auto b0 = static_cast<unsigned char>(s_[i_]);
        int extra = 0;
        unsigned cp = 0;
        if (b0 < 0x80) {
            out += s_[i_++];
            return;
        } else if ((b0 & 0xE0) == 0xC0) {
            extra = 1;
            cp = b0 & 0x1F;
        } else if ((b0 & 0xF0) == 0xE0) {
            extra = 2;
            cp = b0 & 0x0F;
        } else if ((b0 & 0xF8) == 0xF0) {
            extra = 3;
            cp = b0 & 0x07;
        } else {
            fail("invalid UTF-8 lead byte");
        }
        if (i_ + 1 + static_cast<std::size_t>(extra) > s_.size()) fail("truncated UTF-8 sequence");
        for (int k = 1; k <= extra; ++k) {
            const auto b = static_cast<unsigned char>(s_[i_ + static_cast<std::size_t>(k)]);
            if ((b & 0xC0) != 0x80) fail("invalid UTF-8 continuation byte");
            cp = (cp << 6) | (b & 0x3F);
        }
        static constexpr unsigned kMin[] = {0, 0x80, 0x800, 0x10000};
        if (cp < kMin[extra] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            fail("invalid UTF-8 code point");
        }
        out.append(s_.substr(i_, 1 + static_cast<std::size_t>(extra)));
        i_ += 1 + static_cast<std::size_t>(extra);
    }

    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (at_end()) fail("unterminated string");
            const char c = s_[i_];
            if (c == '"') {
                ++i_;
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) fail("unescaped control character in string");
            if (c != '\\') {
                raw_utf8(out);
                continue;
            }
            ++i_;
            if (at_end()) fail("unterminated escape");
            const char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (s_.substr(i_, 2) != "\\u") fail("unpaired high surrogate");
                        i_ += 2;
                        const unsigned lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("unpaired high surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unpaired low surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("invalid escape");
            }
        }
    }

    std::string_view s_;
    std::size_t i_ = 0;
};

}  // namespace

JsonValue parse_strict_json(std::string_view text) { return Reader(text).document(); }

// ─── writer ───────────────────────────────────────────────────────────────

std::string python_float_repr(double value) {
    if (!std::isfinite(value)) throw ConfigError("cannot serialize a non-finite float");
    // Shortest round-trip digits, then Python's float_repr_style='short' layout:
    // exponent form iff decpt <= -4 or decpt > 16, ".0" appended to integers.
    char buf[64];
    auto [end, ec] = std::to_chars(buf, buf + sizeof buf, value, std::chars_format::scientific);
    if (ec != std::errc()) throw ConfigError("float formatting failed");
    std::string sci(buf, end);
    std::string sign;
    if (!sci.empty() && sci[0] == '-') {
        sign = "-";
        sci.erase(0, 1);
    }
    const std::size_t epos = sci.find('e');
    std::string digits;
    for (std::size_t k = 0; k < epos; ++k) {
        if (sci[k] != '.') digits += sci[k];
    }
    const int exp10 = std::stoi(sci.substr(epos + 1));
    const int decpt = exp10 + 1;
    const int nd = static_cast<int>(digits.size());
    std::string out = sign;
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (nd > 1) out += "." + digits.substr(1);
        char ebuf[16];
        std::snprintf(ebuf, sizeof ebuf, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
        out += ebuf;
    } else if (decpt <= 0) {
        out += "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    } else if (decpt >= nd) {
        out += digits + std::string(static_cast<std::size_t>(decpt - nd), '0') + ".0";
    } else {
        out += digits.substr(0, static_cast<std::size_t>(decpt)) + "." +
               digits.substr(static_cast<std::size_t>(decpt));
    }
    return out;
}

namespace {

void write_escaped_u(std::string& out, unsigned unit) {
    char buf[8];
    std::snprintf(buf, sizeof buf, "\\u%04x", unit);
    out += buf;
}

// ensure_ascii=True: everything outside ' '..'~' is escaped, named escapes for
// \b \f \n \r \t, lowercase \uXXXX otherwise, surrogate pairs above the BMP.
void write_string(std::string& out, const std::string& s) {
    out += '"';
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        unsigned cp = b0;
        std::size_t len = 1;
        if (b0 >= 0xF0) {
            len = 4;
            cp = b0 & 0x07;
        } else if (b0 >= 0xE0) {
            len = 3;
            cp = b0 & 0x0F;
        } else if (b0 >= 0xC0) {
            len = 2;
            cp = b0 & 0x1F;
        }
        for (std::size_t k = 1; k < len; ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        }
        i += len;
        switch (cp) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            default: break;
        }
        if (cp >= 0x20 && cp <= 0x7E) {
            out += static_cast<char>(cp);
        } else if (cp < 0x10000) {
            write_escaped_u(out, cp);
        } else {
            const unsigned v = cp - 0x10000;
            write_escaped_u(out, 0xD800 | (v >> 10));
            write_escaped_u(out, 0xDC00 | (v & 0x3FF));
        }
    }
    out += '"';
}

void write_value(std::string& out, const JsonValue& v, int indent) {
    const auto newline = [&](int level) {
        out += '\n';
        out.append(static_cast<std::size_t>(level) * 2, ' ');
    };
    switch (v.kind) {
        case JsonValue::Kind::Null: out += "null"; return;
        case JsonValue::Kind::Bool: out += v.boolean ? "true" : "false"; return;
        case JsonValue::Kind::Int: out += std::to_string(v.integer); return;
        case JsonValue::Kind::Float: out += python_float_repr(v.number); return;
        case JsonValue::Kind::String: write_string(out, v.string); return;
        case JsonValue::Kind::Array:
            if (v.array.empty()) {
                out += "[]";
                return;
            }
            out += '[';
            for (std::size_t k = 0; k < v.array.size(); ++k) {
                if (k) out += ',';
                newline(indent + 1);
                write_value(out, v.array[k], indent + 1);
            }
            newline(indent);
            out += ']';
            return;
        case JsonValue::Kind::Object:
            if (v.object.empty()) {
                out += "{}";
                return;
            }
            out += '{';
            for (std::size_t k = 0; k < v.object.size(); ++k) {
                if (k) out += ',';
                newline(indent + 1);
                write_string(out, v.object[k].first);
                out += ": ";
                write_value(out, v.object[k].second, indent + 1);
            }
            newline(indent);
            out += '}';
            return;
    }
}

}  // namespace

std::string dump_python_json(const JsonValue& value) {
    std::string out;
    write_value(out, value, 0);
    return out;
}

}  // namespace saccade::shipping
