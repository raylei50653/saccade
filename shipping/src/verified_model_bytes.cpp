#include "saccade_shipping/verified_model_bytes.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>

#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/sha256.hpp"

namespace saccade::shipping {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw DetectorLoadError("shipping detector load: " + message);
}

struct CloseFd {
    int fd;
    ~CloseFd() { ::close(fd); }
};

}  // namespace

VerifiedModelBytes VerifiedModelBytes::read(const std::string& what, const std::string& path,
                                            const std::string& expected_sha256,
                                            std::int64_t expected_size) {
    const std::string description = what + " " + path;
    // Manifest mode retains its symlink refusal. Legacy mode retains its
    // existing path acceptance; integrity still comes from this exact buffer.
    const int flags = O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOCTTY |
                      (expected_size >= 0 ? O_NOFOLLOW : 0);
    const int fd = ::open(path.c_str(), flags);
    if (fd < 0) fail("cannot open " + description + ": " + std::strerror(errno));
    const CloseFd guard{fd};
    struct stat st {};
    if (::fstat(fd, &st) != 0) fail("cannot stat " + description + ": " + std::strerror(errno));
    if (!S_ISREG(st.st_mode)) fail(description + " is not a regular file");
    if (st.st_size < 0 || static_cast<std::uint64_t>(st.st_size) > std::string().max_size()) {
        fail(description + " size is not representable");
    }
    if (expected_size >= 0 && st.st_size != expected_size) {
        fail(description + " bytes " + std::to_string(st.st_size) + " != " +
             std::to_string(expected_size));
    }
    std::string bytes(static_cast<std::size_t>(st.st_size), '\0');
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto remaining = bytes.size() - offset;
        const auto count = std::min(remaining, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t n = ::read(fd, bytes.data() + offset, count);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) fail("cannot read " + description + ": " + std::strerror(errno));
        if (n == 0) fail(description + " changed size while reading");
        offset += static_cast<std::size_t>(n);
    }
    char extra;
    ssize_t n;
    do { n = ::read(fd, &extra, 1); } while (n < 0 && errno == EINTR);
    if (n < 0) fail("cannot read " + description + ": " + std::strerror(errno));
    if (n != 0) fail(description + " changed size while reading");
    std::string got = sha256_hex(bytes.data(), bytes.size());
    if (got != expected_sha256) fail(description + " sha256 " + got + " != " + expected_sha256);
    return VerifiedModelBytes(std::move(bytes), std::move(got));
}

VerifiedModelStream::Buffer::Buffer(const VerifiedModelBytes& bytes) {
    // std::streambuf's get area takes char*, but this buffer is input-only;
    // neither the stream nor this class has a put area or writes its bytes.
    char* begin = const_cast<char*>(bytes.data());
    setg(begin, begin, begin + bytes.size());
}

VerifiedModelStream::Buffer::pos_type VerifiedModelStream::Buffer::seekoff(
    off_type off, std::ios_base::seekdir dir, std::ios_base::openmode mode) {
    if (!(mode & std::ios_base::in) || (mode & std::ios_base::out)) return pos_type(off_type(-1));
    const off_type size = egptr() - eback();
    off_type base = 0;
    if (dir == std::ios_base::cur) base = gptr() - eback();
    else if (dir == std::ios_base::end) base = size;
    else if (dir != std::ios_base::beg) return pos_type(off_type(-1));
    if (off < -base || off > size - base) return pos_type(off_type(-1));
    setg(eback(), eback() + base + off, egptr());
    return pos_type(base + off);
}

VerifiedModelStream::Buffer::pos_type VerifiedModelStream::Buffer::seekpos(
    pos_type pos, std::ios_base::openmode mode) {
    return seekoff(off_type(pos), std::ios_base::beg, mode);
}

VerifiedModelStream::VerifiedModelStream(const VerifiedModelBytes& bytes)
    : std::istream(nullptr), buffer_(bytes) {
    rdbuf(&buffer_);
}

}  // namespace saccade::shipping
