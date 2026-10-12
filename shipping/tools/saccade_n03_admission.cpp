// N03 admission tool (#549 S2-2, P-N03 rev 3d): reads a candidate operator
// library once into an owned buffer (one descriptor, fstat's size, then end of
// file; any read error refuses) and admits it against the pin
// (saccade_shipping/n03_admission.hpp). CPU only; not installed and linked by
// no runtime target.
//
// Usage: saccade_n03_admission --pin <n03_operator_pin.json> [--skip-if-missing] <operator library>
//   exit 0: admitted, the record (JSON) on stdout;
//   exit 2: refused or unreadable, the reason on stderr, no record;
//   exit 77: --skip-if-missing and the library does not exist (ctest SKIP).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <iostream>
#include <string>

#include "saccade_shipping/n03_admission.hpp"

#ifndef SACCADE_N03_ADMISSION_SOURCE_SHA256
#error "SACCADE_N03_ADMISSION_SOURCE_SHA256 must name the sha256 of src/n03_admission.cpp"
#endif

namespace {

// The whole file, read once through one descriptor: exactly fstat's size,
// then end of file; false on any open, stat or read error.
bool read_whole(const std::string& path, std::string& out) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st {};
    bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0;
    if (ok) {
        out.assign(static_cast<std::size_t>(st.st_size), '\0');
        std::size_t done = 0;
        while (ok && done < out.size()) {
            const ssize_t n = ::read(fd, out.data() + done, out.size() - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) ok = false;
            else done += static_cast<std::size_t>(n);
        }
        char extra = 0;
        ssize_t n = 0;
        while (ok && (n = ::read(fd, &extra, 1)) < 0 && errno == EINTR) {
        }
        if (n != 0) ok = false;  // an error, or more bytes than fstat said
    }
    ::close(fd);
    return ok;
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
