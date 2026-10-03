// SHA-256 (FIPS 180-4) for the shipping lineage gate (#465 Phase B PR-8, B5).
// CUDA-free. The shipping detector verifies its model files by hash before it
// loads them (boundary §5 B5: "shipping 只做 hash 驗證"); this is that hash,
// with no dependency on a crypto library.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace saccade::shipping {

class Sha256 {
public:
    Sha256();
    void update(const void* data, std::size_t n);
    // Lowercase hex digest; the object must not be updated afterwards.
    std::string hex_digest();

private:
    void block(const std::uint8_t* p);
    std::uint32_t h_[8];
    std::uint8_t buf_[64];
    std::size_t buf_len_ = 0;
    std::uint64_t total_ = 0;
    bool done_ = false;
};

std::string sha256_hex(const void* data, std::size_t n);

// Hash of a whole file; throws ConfigError when it cannot be read.
std::string sha256_file_hex(const std::string& path);

}  // namespace saccade::shipping
