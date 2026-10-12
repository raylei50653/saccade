// N03 admission tool (#549 S2-2, P-N03 rev 3d): reads a candidate operator
// library once into an owned buffer and admits it against the pin
// (saccade_shipping/n03_admission.hpp). CPU only; not installed and linked by
// no runtime target.
//
// Usage: saccade_n03_admission --pin <n03_operator_pin.json> [--skip-if-missing] <operator library>
//   exit 0: admitted, the record (JSON) on stdout;
//   exit 2: refused or unreadable, the reason on stderr, no record;
//   exit 77: --skip-if-missing and the library does not exist (ctest SKIP).
#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "saccade_shipping/n03_admission.hpp"

#ifndef SACCADE_N03_ADMISSION_SOURCE_SHA256
#error "SACCADE_N03_ADMISSION_SOURCE_SHA256 must name the sha256 of src/n03_admission.cpp"
#endif

namespace {

bool read_whole(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream o;
    o << in.rdbuf();
    if (in.bad()) return false;
    out = o.str();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string pin_path, library;
    bool skip_if_missing = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pin" && i + 1 < argc) {
            pin_path = argv[++i];
        } else if (a == "--skip-if-missing") {
            skip_if_missing = true;
        } else if (library.empty() && !a.empty() && a[0] != '-') {
            library = a;
        } else {
            library.clear();
            break;
        }
    }
    if (pin_path.empty() || library.empty()) {
        std::fprintf(stderr, "usage: %s --pin <n03_operator_pin.json> [--skip-if-missing] <operator library>\n",
                     argv[0]);
        return 2;
    }
    struct stat st {};
    if (skip_if_missing && ::stat(library.c_str(), &st) != 0 && errno == ENOENT) {
        std::printf("SKIP: %s does not exist\n", library.c_str());
        return 77;
    }
    std::string pin, bytes;
    if (!read_whole(pin_path, pin)) {
        std::fprintf(stderr, "n03 admission: cannot read the pin %s\n", pin_path.c_str());
        return 2;
    }
    if (!read_whole(library, bytes)) {
        std::fprintf(stderr, "n03 admission: cannot read %s\n", library.c_str());
        return 2;
    }
    try {
        std::cout << saccade::shipping::admit_n03_operator(bytes, library, pin, SACCADE_N03_ADMISSION_SOURCE_SHA256);
    } catch (const saccade::shipping::N03AdmissionError& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    return 0;
}
