// Native S2 vs the oracle's compiled S2, and the detector loader's fail-closed
// hash checks (#465 Phase B PR-8, U3b-2). Needs a CUDA device.
//
// Usage: saccade_shipping_detector_s2_test <shipping_detector_s2.json>
//            [<resolved config> <lineage> <attestation> <model root>]
//
// The fixture is written by scripts/model/render_shipping_detector_s2_fixture.py
// from the oracle's own code (native_detector_parity.oracle_s2: the
// torch.compile'd _postprocess_mamba_fixed + _whole_graph_fn's coordinate
// scaling). For every case this test rebuilds the same float32 inputs from the
// case's generator spec (splitmix64, mirrored from the renderer), runs s2_run
// and requires the (rows, 6) output before and after scaling to be
// bit-identical. Cases where the eager S2 differs from the compiled one must
// exist, so a native S2 that computed eager numerics would fail here.
//
// With the optional arguments, and only when the model files exist under the
// model root, it also checks that DetectorHost refuses a wrong sha256 for each
// of the operator library, the head artifact and the backbone engine before
// loading anything (the operator library is not mapped afterwards), and that
// the real plan loads with the report the lineage predicts.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/detector_host.hpp"
#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/detector_s2.hpp"
#include "saccade_shipping/strict_json.hpp"

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

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

const JsonValue& at(const JsonValue& v, const char* key) {
    const JsonValue* p = v.find(key);
    if (p == nullptr) throw std::runtime_error(std::string("fixture: missing ") + key);
    return *p;
}

std::vector<std::uint8_t> b64(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<std::uint8_t> out;
    std::uint32_t acc = 0;
    int bits = 0;
    for (char c : s) {
        if (c == '=') break;
        const int v = val(c);
        if (v < 0) throw std::runtime_error("fixture: bad base64");
        acc = (acc << 6) | static_cast<std::uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// render_shipping_detector_s2_fixture.splitmix64 (uint64 arithmetic wraps).
std::uint64_t splitmix64(std::uint64_t x) {
    std::uint64_t z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float special(const std::string& name) {
    if (name == "nan") return std::numeric_limits<float>::quiet_NaN();
    if (name == "inf") return std::numeric_limits<float>::infinity();
    if (name == "-inf") return -std::numeric_limits<float>::infinity();
    throw std::runtime_error("fixture: bad special " + name);
}

// render_shipping_detector_s2_fixture.make_inputs.
std::vector<std::vector<float>> make_inputs(const JsonValue& c, int nc, const int sides[3]) {
    const std::uint64_t seed = static_cast<std::uint64_t>(at(c, "seed").integer);
    std::vector<std::vector<float>> t(6);
    for (int stream = 0; stream < 6; ++stream) {
        const bool cls = stream < 3;
        const int side = sides[stream % 3];
        const std::size_t n = static_cast<std::size_t>(cls ? nc : 4) * side * side;
        const JsonValue& spec = at(c, cls ? "cls" : "reg");
        t[static_cast<std::size_t>(stream)].resize(n);
        auto& v = t[static_cast<std::size_t>(stream)];
        if (at(spec, "mode").string == "fill") {
            std::fill(v.begin(), v.end(), special(at(spec, "value").string));
            continue;
        }
        const std::int64_t lo = at(spec, "lo").integer;
        const std::uint64_t span = static_cast<std::uint64_t>(at(spec, "span").integer);
        const int frac = static_cast<int>(at(spec, "frac").integer);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t u = splitmix64((seed << 32) | (static_cast<std::uint64_t>(stream) << 24) | i);
            const std::int64_t k = lo + static_cast<std::int64_t>((u >> 40) % span);
            v[i] = static_cast<float>(std::ldexp(static_cast<double>(k), -frac));
        }
    }
    const int bases[3] = {0, sides[0] * sides[0], sides[0] * sides[0] + sides[1] * sides[1]};
    auto locate = [&](int anchor, int& level, int& local) {
        level = anchor < bases[1] ? 0 : (anchor < bases[2] ? 1 : 2);
        local = anchor - bases[level];
    };
    if (const JsonValue* sw = c.find("sweep")) {
        const int count = static_cast<int>(at(*sw, "count").integer);
        const int stride = static_cast<int>(at(*sw, "anchor_stride").integer);
        const int ch = static_cast<int>(at(*sw, "cls_channel").integer);
        for (int j = 0; j < count; ++j) {
            int level = 0, local = 0;
            locate(j * stride, level, local);
            const double v =
                std::ldexp(static_cast<double>(at(*sw, "lo").integer + j * at(*sw, "step").integer),
                           -static_cast<int>(at(*sw, "frac").integer));
            t[static_cast<std::size_t>(level)][static_cast<std::size_t>(ch) * sides[level] * sides[level] +
                                               static_cast<std::size_t>(local)] = static_cast<float>(v);
        }
    }
    if (const JsonValue* sps = c.find("specials")) {
        for (const JsonValue& sp : sps->array) {
            int level = 0, local = 0;
            locate(static_cast<int>(at(sp, "anchor").integer), level, local);
            const bool cls = at(sp, "tensor").string == "cls";
            const std::size_t ch = static_cast<std::size_t>(at(sp, "channel").integer);
            t[static_cast<std::size_t>(cls ? level : 3 + level)]
             [ch * sides[level] * sides[level] + static_cast<std::size_t>(local)] =
                 special(at(sp, "value").string);
        }
    }
    return t;
}

void check_s2_fixture(const JsonValue& fx) {
    const int nc = static_cast<int>(at(fx, "num_classes").integer);
    const int k = static_cast<int>(at(fx, "max_det").integer);
    const int img = static_cast<int>(at(fx, "img_size").integer);
    int sides[3];
    for (int i = 0; i < 3; ++i) sides[i] = img / static_cast<int>(at(fx, "strides").array[static_cast<std::size_t>(i)].integer);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    int eager_differs = 0;
    for (const JsonValue& c : at(fx, "cases").array) {
        const std::string name = at(c, "name").string;
        const auto host = make_inputs(c, nc, sides);
        std::vector<float*> dev(6, nullptr);
        for (int i = 0; i < 6; ++i) {
            const std::size_t bytes = host[static_cast<std::size_t>(i)].size() * sizeof(float);
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&dev[static_cast<std::size_t>(i)]), bytes), "malloc");
            cuda_check(cudaMemcpy(dev[static_cast<std::size_t>(i)], host[static_cast<std::size_t>(i)].data(),
                                  bytes, cudaMemcpyHostToDevice),
                       "H2D");
        }
        sh::S2Level levels[3];
        for (int i = 0; i < 3; ++i) {
            levels[i] = {dev[static_cast<std::size_t>(i)], dev[static_cast<std::size_t>(3 + i)], sides[i],
                         static_cast<float>(at(fx, "strides").array[static_cast<std::size_t>(i)].integer)};
        }
        const float sx = sh::coordinate_scale(static_cast<int>(at(c, "width").integer), img);
        const float sy = sh::coordinate_scale(static_cast<int>(at(c, "height").integer), img);
        float *raw = nullptr, *scaled = nullptr;
        const std::size_t out_bytes = static_cast<std::size_t>(k) * 6 * sizeof(float);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&raw), out_bytes), "malloc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&scaled), out_bytes), "malloc");
        sh::s2_run(levels, nc, k, sx, sy, raw, scaled, stream);
        std::vector<std::uint8_t> got_raw(out_bytes), got_scaled(out_bytes);
        cuda_check(cudaMemcpyAsync(got_raw.data(), raw, out_bytes, cudaMemcpyDeviceToHost, stream), "D2H");
        cuda_check(cudaMemcpyAsync(got_scaled.data(), scaled, out_bytes, cudaMemcpyDeviceToHost, stream), "D2H");
        cuda_check(cudaStreamSynchronize(stream), "sync");
        const auto want_raw = b64(at(c, "raw_f32_b64").string);
        const auto want_scaled = b64(at(c, "scaled_f32_b64").string);
        const bool raw_ok = got_raw == want_raw;
        const bool scaled_ok = got_scaled == want_scaled;
        if (!raw_ok || !scaled_ok) {
            int first = -1, n_diff = 0;
            for (std::size_t i = 0; i + 4 <= std::min(got_scaled.size(), want_scaled.size()); i += 4) {
                if (std::memcmp(&got_scaled[i], &want_scaled[i], 4) != 0) {
                    if (first < 0) first = static_cast<int>(i / 4);
                    ++n_diff;
                }
            }
            std::fprintf(stderr, "case %s: raw %s scaled %s (first differing value %d, row %d col %d; %d values)\n",
                         name.c_str(), raw_ok ? "ok" : "DIFFERS", scaled_ok ? "ok" : "DIFFERS", first,
                         first / 6, first % 6, n_diff);
        }
        CHECK(raw_ok);
        CHECK(scaled_ok);
        if (!at(c, "eager_equal").boolean) ++eager_differs;
        for (float* p : dev) cudaFree(p);
        cudaFree(raw);
        cudaFree(scaled);
    }
    CHECK(eager_differs > 0);  // the fixture tells compiled from eager S2
    cudaStreamDestroy(stream);
}

bool mapped(const std::string& basename) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find(basename) != std::string::npos) return true;
    }
    return false;
}

bool refuses(const sh::DetectorPlan& plan, const std::string& root, const std::string& what) {
    try {
        sh::DetectorHost host(plan, root, nullptr);
    } catch (const sh::ConfigError& e) {
        if (std::string(e.what()).find(what) != std::string::npos) return true;
        std::fprintf(stderr, "refused for another reason: %s\n", e.what());
        return false;
    }
    std::fprintf(stderr, "not refused: %s\n", what.c_str());
    return false;
}

void check_loader(const std::string& config, const std::string& lineage, const std::string& attestation,
                  const std::string& root) {
    const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config_file(config);
    const sh::DetectorPlan plan = sh::plan_detector_files(cfg, {lineage, attestation});
    for (const std::string& rel : {plan.op_library.path, plan.head_artifact.path, plan.backbone_engine.path}) {
        if (!std::filesystem::exists(std::filesystem::path(root) / rel)) {
            std::printf("SKIP loader checks: %s not present\n", rel.c_str());
            return;
        }
    }
    const std::string zero(64, '0');
    sh::DetectorPlan p = plan;
    p.op_library.sha256 = zero;
    CHECK(refuses(p, root, "operator library"));
    p = plan;
    p.head_artifact.sha256 = zero;
    CHECK(refuses(p, root, "head artifact"));
    p = plan;
    p.backbone_engine.sha256 = zero;
    CHECK(refuses(p, root, "backbone engine"));
    // Refused on hashes before anything was loaded.
    CHECK(!mapped(std::filesystem::path(plan.op_library.path).filename().string()));

    sh::DetectorHost host(plan, root, nullptr);
    const sh::HeadLoadReport& r = host.load_report();
    CHECK(r.op_library_sha256 == plan.op_library.sha256);
    CHECK(r.param_devices == std::vector<std::string>{"cuda:0"});
    CHECK(!r.constant_devices.empty());
    for (const auto& d : r.constant_devices) CHECK(d == "cpu");
    CHECK(r.native_scan_calls == plan.native_scan_calls);
    CHECK(!r.runtime_readback.graph_executor_optimize && !r.runtime_readback.cudnn_benchmark &&
          r.runtime_readback.cudnn_allow_tf32 && !r.runtime_readback.matmul_allow_tf32);
    CHECK(r.engine_io.size() == 4);
    CHECK(sh::mapped_python_libraries().empty());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 6) {
        std::fprintf(stderr,
                     "usage: %s <shipping_detector_s2.json> [<config> <lineage> <attestation> <model root>]\n",
                     argv[0]);
        return 2;
    }
    try {
        check_s2_fixture(sh::parse_strict_json(read_file(argv[1])));
        if (argc == 6) check_loader(argv[2], argv[3], argv[4], argv[5]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
