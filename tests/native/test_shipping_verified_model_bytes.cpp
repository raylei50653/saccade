// CPU controls for Gate B's same-buffer invariant (#549 S2-2, MB-41/MB-58).
// A path replaced after SHA verification cannot change the immutable bytes
// supplied to the seekable head stream or the engine's pointer/size API.
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/sha256.hpp"
#include "saccade_shipping/verified_model_bytes.hpp"

namespace fs = std::filesystem;
namespace sh = saccade::shipping;

namespace {
int failures = 0;
int checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; \
    std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition); } } while (0)

void write(const fs::path& path, const std::string& value) {
    std::ofstream out(path, std::ios::binary);
    out.write(value.data(), value.size());
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

std::string error(const fs::path& path, const std::string& sha, std::int64_t size = -1) {
    try { (void)sh::VerifiedModelBytes::read("test model", path.string(), sha, size); }
    catch (const sh::DetectorLoadError& e) { return e.what(); }
    return {};
}
}  // namespace

int main() {
    char temp[] = "/tmp/saccade_verified_model_bytes.XXXXXX";
    char* created = ::mkdtemp(temp);
    if (!created) return 2;
    const fs::path dir = created;
    try {
        const fs::path path = dir / "model";
        const std::string original("model\0\xff\x01\x7f\nbytes", 15);
        const std::string sha = sh::sha256_hex(original.data(), original.size());
        write(path, original);
        const sh::VerifiedModelBytes bytes = sh::VerifiedModelBytes::read("test model", path.string(), sha, original.size());
        CHECK(bytes.sha256() == sha);
        CHECK(std::string(bytes.data(), bytes.size()) == original);
        sh::VerifiedModelStream stream(bytes);
        stream.seekg(-5, std::ios::end);
        std::string tail(5, '\0');
        stream.read(tail.data(), tail.size());
        CHECK(tail == original.substr(original.size() - 5));

        // Deterministic TOCTOU: atomically replace the path after hashing,
        // then read the engine view and head stream, which retain the original.
        const std::string replacement(original.size(), '!');
        write(dir / "replacement", replacement);
        fs::rename(dir / "replacement", path);
        CHECK(sh::sha256_file_hex(path.string()) != bytes.sha256());
        CHECK(sh::sha256_hex(bytes.data(), bytes.size()) == sha);
        stream.seekg(0);
        std::string loaded(original.size(), '\0');
        stream.read(loaded.data(), loaded.size());
        CHECK(loaded == original);
        stream.seekg(-7, std::ios::cur);
        CHECK(stream.tellg() == static_cast<std::streamoff>(original.size() - 7));
        stream.seekg(-1, std::ios::beg);
        CHECK(stream.fail());
        stream.clear();
        stream.seekg(original.size() + 1, std::ios::beg);
        CHECK(stream.fail());

        const std::string changed = error(path, sha, original.size());
        CHECK(changed.find("shipping detector load: test model " + path.string() + " sha256 ") == 0);
        CHECK(error(path, sha, original.size() + 1).find(" bytes ") != std::string::npos);
        CHECK(error(dir / "absent", sha).find("cannot open") != std::string::npos);
        CHECK(error(dir, sha).find("not a regular file") != std::string::npos);
        fs::create_symlink(path, dir / "link");
        CHECK(error(dir / "link", sha).find("cannot open") != std::string::npos);
        CHECK(::mkfifo((dir / "fifo").c_str(), 0600) == 0);
        CHECK(error(dir / "fifo", sha).find("not a regular file") != std::string::npos);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        fs::remove_all(dir);
        return 2;
    }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
