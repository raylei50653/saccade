// The production N03 pin as compiled (#549 S2-2, P-N03 rev 3d): the
// generated saccade_shipping_n03_pin object equals shipping/n03_operator_pin.json,
// the record review accepts.
//
// Usage: saccade_shipping_n03_pin_test <shipping/n03_operator_pin.json>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "saccade_shipping/operator_loader.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace sh = saccade::shipping;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <n03_operator_pin.json>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    const sh::JsonValue pin = sh::parse_strict_json(text.str());
    const sh::JsonValue* sha = pin.find("sha256");
    const sh::JsonValue* bytes = pin.find("bytes");
    const sh::N03OperatorPin& compiled = sh::n03_operator_pin();
    int failures = 0;
    if (!sha || sha->kind != sh::JsonValue::Kind::String || sha->string != compiled.sha256) {
        std::fprintf(stderr, "compiled pin sha256 %s is not the record's\n", compiled.sha256);
        ++failures;
    }
    if (!bytes || bytes->kind != sh::JsonValue::Kind::Int || bytes->integer != compiled.bytes) {
        std::fprintf(stderr, "compiled pin bytes %lld are not the record's\n", static_cast<long long>(compiled.bytes));
        ++failures;
    }
    std::printf("2 checks, %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
