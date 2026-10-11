// Gate B model bytes (#549 S2-2). Read each engine/head exactly once, hash
// the owned immutable buffer, then load that buffer instead of reopening its
// path. This does not make the path-loaded operator a loaded_buffer object.
#pragma once

#include <cstdint>
#include <istream>
#include <streambuf>
#include <string>
#include <utility>

namespace saccade::shipping {

class VerifiedModelBytes {
public:
    // expected_size < 0 is legacy mode; manifest mode supplies Gate A's size.
    // Refuses nonregular files, read/size errors and a SHA-256 mismatch.
    static VerifiedModelBytes read(const std::string& what, const std::string& path,
                                   const std::string& expected_sha256,
                                   std::int64_t expected_size = -1);
    const char* data() const { return bytes_.data(); }
    std::size_t size() const { return bytes_.size(); }
    const std::string& sha256() const { return sha256_; }

private:
    VerifiedModelBytes(std::string bytes, std::string sha256)
        : bytes_(std::move(bytes)), sha256_(std::move(sha256)) {}
    std::string bytes_, sha256_;
};

// A seekable input view of the exact verified memory buffer, for
// torch::jit::load(std::istream&). Owns no copy and exposes no write API.
// The VerifiedModelBytes object must outlive the stream.
class VerifiedModelStream final : public std::istream {
public:
    explicit VerifiedModelStream(const VerifiedModelBytes& bytes);

private:
    class Buffer final : public std::streambuf {
    public:
        explicit Buffer(const VerifiedModelBytes& bytes);

    protected:
        pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                         std::ios_base::openmode mode) override;
        pos_type seekpos(pos_type pos, std::ios_base::openmode mode) override;
    } buffer_;
};

}  // namespace saccade::shipping
