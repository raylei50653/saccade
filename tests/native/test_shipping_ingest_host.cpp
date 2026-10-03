// Native ingest host on the GPU (#465 Phase B PR-7, U3b-1). Needs a CUDA
// device; built with ENABLE_NATIVE_TESTS from the root build.
//
// Usage: saccade_shipping_ingest_host_test <resolved config> <shipping_ingest.json> <jpeg dir>
//
// The fixture is the oracle's own output (scripts/model/render_shipping_ingest_fixture.py).
// Real-frame parity is measured by scripts/eval/diagnostics/native_ingest_parity.py;
// this test pins, on the committed inputs:
//   * the normalize kernel equals torch's CUDA ingest op for all 256 values,
//     also through the grid-stride loop of a buffer larger than one grid;
//   * the decoder gives torchvision's planar RGB bytes for every fixture JPEG
//     (4:2:0 / 4:2:2 / 4:4:4, odd size, progressive, grayscale), twice;
//   * IngestHost decodes frame k = k-th listed file, its frame buffer is the
//     normalize of its decoded bytes, and it refuses a decoded size other than
//     seqinfo.ini's, a bitstream nvJPEG cannot read, a directory and an
//     out-of-range frame.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "saccade_shipping/ingest_host.hpp"
#include "saccade_shipping/ingest_plan.hpp"
#include "saccade_shipping/strict_json.hpp"

namespace fs = std::filesystem;
namespace sh = saccade::shipping;
using sh::JsonValue;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failures;                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

const JsonValue& at(const JsonValue& v, const char* key) {
    const JsonValue* p = v.find(key);
    if (p == nullptr) throw std::runtime_error(std::string("fixture: missing ") + key);
    return *p;
}

std::vector<std::uint8_t> from_hex(const std::string& h) {
    std::vector<std::uint8_t> out(h.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<std::uint8_t>(std::stoul(h.substr(i * 2, 2), nullptr, 16));
    }
    return out;
}

std::vector<std::uint32_t> table_bits(const JsonValue& table) {
    std::vector<std::uint32_t> out;
    for (const JsonValue& v : table.array) {
        const std::vector<std::uint8_t> b = from_hex(v.string);
        std::uint32_t u;
        std::memcpy(&u, b.data(), sizeof u);  // bytes in memory order
        out.push_back(u);
    }
    return out;
}

template <class T>
struct DeviceBuffer {
    T* p = nullptr;
    explicit DeviceBuffer(std::size_t n) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "alloc"); }
    ~DeviceBuffer() { cudaFree(p); }
};

template <class T>
std::vector<T> d2h(const T* p, std::size_t n) {
    std::vector<T> out(n);
    cuda_check(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
    return out;
}

bool throws_input(const std::function<void()>& f) {
    try {
        f();
    } catch (const sh::InputError&) {
        return true;
    }
    return false;
}

void check_normalize(const std::vector<std::uint32_t>& want, cudaStream_t stream) {
    CHECK(want.size() == 256);
    // More elements than one grid of the launcher covers (65535 x 256).
    const std::size_t n = 65535ull * 256ull + 1000ull;
    std::vector<std::uint8_t> in(n);
    for (std::size_t i = 0; i < n; ++i) in[i] = static_cast<std::uint8_t>((i * 7) & 255);
    DeviceBuffer<std::uint8_t> d_in(n);
    DeviceBuffer<float> d_out(n);
    cuda_check(cudaMemcpy(d_in.p, in.data(), n, cudaMemcpyHostToDevice), "H2D");
    sh::normalize_rgb_u8(d_in.p, d_out.p, n, sh::kIngestNormalizeScale, stream);
    cuda_check(cudaStreamSynchronize(stream), "sync");
    const std::vector<float> out = d2h(d_out.p, n);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t u;
        std::memcpy(&u, &out[i], sizeof u);
        if (u != want[in[i]]) ++bad;
    }
    if (bad != 0) std::fprintf(stderr, "normalize kernel: %zu of %zu values differ from torch\n", bad, n);
    CHECK(bad == 0);
}

// Decoder vs torchvision on every fixture JPEG.
void check_decoder(sh::JpegDecoder& dec, const JsonValue& jpegs, const fs::path& dir) {
    CHECK(jpegs.array.size() == 6);
    for (const JsonValue& j : jpegs.array) {
        const std::string& file = at(j, "file").string;
        const std::string raw = read_file(dir / file);
        const std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
        const sh::JpegImageInfo info = dec.image_info(bytes);
        CHECK(info.width == at(j, "width").integer && info.height == at(j, "height").integer);
        const std::vector<std::uint8_t> want = from_hex(at(j, "rgb_planar_hex").string);
        const std::size_t n = 3 * static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height);
        CHECK(want.size() == n);
        DeviceBuffer<std::uint8_t> d(n);
        for (int rep = 0; rep < 2; ++rep) {
            cuda_check(cudaMemset(d.p, 0xA5, n), "memset");
            const sh::DecodePath path = dec.decode_rgb(bytes, info, d.p);
            const std::vector<std::uint8_t> got = d2h(d.p, n);
            std::size_t diff = 0;
            for (std::size_t i = 0; i < n && i < want.size(); ++i) diff += got[i] != want[i];
            std::printf("  %s %dx%d %s: %zu of %zu bytes differ from torchvision\n", file.c_str(),
                        info.width, info.height, sh::decode_path_name(path), diff, n);
            CHECK(diff == 0);
        }
    }
}

void check_host(sh::JpegDecoder& dec, const JsonValue& jpegs, const fs::path& dir,
                const std::vector<std::uint32_t>& table, const fs::path& root, cudaStream_t stream) {
    const sh::IngestPlan plan{};
    auto fixture_bytes = [&](const std::string& file) {
        for (const JsonValue& j : jpegs.array) {
            if (at(j, "file").string == file) return from_hex(at(j, "rgb_planar_hex").string);
        }
        throw std::runtime_error("fixture: no " + file);
    };

    // Two frames, listed out of creation order; frame 1 is the progressive one.
    const fs::path seq = root / "seq";
    fs::create_directories(seq / "img1");
    write_file(seq / "seqinfo.ini", "[Sequence]\nimWidth=48\nimHeight=32\nseqLength=2\n");
    write_file(seq / "img1" / "000002.jpg", read_file(dir / "baseline_420.jpg"));
    write_file(seq / "img1" / "000001.jpg", read_file(dir / "progressive_420.jpg"));
    write_file(seq / "img1" / "000003.jpg", "not consumed");
    const sh::SequenceInput in = sh::read_sequence_input(seq);
    sh::IngestHost host(plan, in, dec, stream);
    CHECK(host.frame_count() == 2);
    const char* expect[] = {"progressive_420.jpg", "baseline_420.jpg"};
    for (int k = 1; k <= 2; ++k) {
        const sh::IngestFrame f = host.ingest(k);
        CHECK(f.file == (k == 1 ? "000001.jpg" : "000002.jpg"));
        const std::size_t n = 3 * 48 * 32;
        const std::vector<std::uint8_t> rgb = d2h(host.decoded_rgb(), n);
        const std::vector<float> frame = d2h(host.frame_chw(), n);
        CHECK(rgb == fixture_bytes(expect[k - 1]));
        std::size_t bad = 0;
        for (std::size_t i = 0; i < n; ++i) {
            std::uint32_t u;
            std::memcpy(&u, &frame[i], sizeof u);
            if (u != table[rgb[i]]) ++bad;
        }
        CHECK(bad == 0);
    }
    for (int k : {0, 3}) {
        bool out_of_range = false;
        try {
            host.ingest(k);
        } catch (const std::out_of_range&) {
            out_of_range = true;
        }
        CHECK(out_of_range);
    }

    // seqinfo.ini says 47x31; the frame decodes to 48x32.
    const fs::path wrong = root / "wrong_size";
    fs::create_directories(wrong / "img1");
    write_file(wrong / "seqinfo.ini", "[Sequence]\nimWidth=47\nimHeight=31\nseqLength=1\n");
    write_file(wrong / "img1" / "000001.jpg", read_file(dir / "baseline_420.jpg"));
    sh::IngestHost wrong_host(plan, sh::read_sequence_input(wrong), dec, stream);
    CHECK(throws_input([&] { wrong_host.ingest(1); }));

    // Not a JPEG, and a directory, in frame position 1.
    const fs::path bad = root / "bad";
    fs::create_directories(bad / "img1" / "000002.jpg");
    write_file(bad / "seqinfo.ini", "[Sequence]\nimWidth=48\nimHeight=32\nseqLength=2\n");
    write_file(bad / "img1" / "000001.jpg", "not a jpeg bitstream");
    sh::IngestHost bad_host(plan, sh::read_sequence_input(bad), dec, stream);
    CHECK(throws_input([&] { bad_host.ingest(1); }));
    CHECK(throws_input([&] { bad_host.ingest(2); }));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <resolved config> <shipping_ingest.json> <jpeg dir>\n", argv[0]);
        return 2;
    }
    fs::path root;
    try {
        const sh::IngestPlan plan = sh::plan_ingest(sh::load_resolved_shipping_config_file(argv[1]));
        CHECK(plan.normalize_scale == sh::kIngestNormalizeScale);
        const JsonValue fixture = sh::parse_strict_json(read_file(argv[2]));
        const fs::path dir(argv[3]);
        const std::vector<std::uint32_t> table = table_bits(at(fixture, "normalize_f32_hex"));
        root = fs::temp_directory_path() /
               ("saccade_ingest_host_test_" + std::to_string(static_cast<long long>(::getpid())));
        fs::remove_all(root);
        fs::create_directories(root);

        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        {
            sh::JpegDecoder dec;
            const auto& lib = dec.library();
            std::printf("nvJPEG %d.%d.%d, hardware decode %s\n", lib.major, lib.minor, lib.patch,
                        lib.hardware_decode ? "available" : "unavailable");
            check_normalize(table, stream);
            check_decoder(dec, at(fixture, "jpegs"), dir);
            check_host(dec, at(fixture, "jpegs"), dir, table, root, stream);
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        if (!root.empty()) fs::remove_all(root);
        return 2;
    }
    fs::remove_all(root);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
