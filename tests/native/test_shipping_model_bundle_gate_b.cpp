// Gate B in manifest mode (#549 S2-1): the detector load takes its paths only
// from Gate A's resolved bindings. Needs a CUDA device, the models and the
// attested operator library (ENABLE_NATIVE_TESTS); not a parity claim.
//
// Usage: saccade_shipping_model_bundle_gate_b_test <repo root>
//
// A real two-root bundle is laid out in a temp directory: the published
// example manifest over copies of N01 / N02 / N04 / N05 / N06 (bundle root)
// and the operator library plus a copy of the production allowlist (runtime
// root); Gate A (run_preflight, manifest mode, the allowlist's sha256 as the
// pin) gives the plan, then:
//   * cross_root:  DetectorHost(plan, "", stream) loads the three files; the
//                  load report's hashes are the manifest's and the operator
//                  library runs from the sealed memfd made from the runtime
//                  root's copy (by /proc/self/maps; neither the runtime root's
//                  nor the repository's path is mapped);
//   * foreign_operator (D-N03-5): a plan naming another operator file than its
//                  binding, or none, is refused before anything is read;
//   * replaced (MB-55): a member is replaced after Gate A (same size, one byte
//                  flipped): the load refuses it at the resolved path before
//                  loading anything (DetectorLoadError, sha256 mismatch);
//   * model_root:  a plan with resolved bindings and a model root is refused;
//   * not_the_plan: resolved bindings whose sha256 is not the plan's are refused.
// Skips (exit 0, "SKIP") when a model file is missing.
#include <cuda_runtime.h>
#include <torch/script.h>
#include <ftw.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/detector_host.hpp"
#include "perception/trt_engine.hpp"
#include "saccade_shipping/model_bundle.hpp"
#include "saccade_shipping/native_runtime_build.hpp"
#include "saccade_shipping/preflight.hpp"
#include "saccade_shipping/run_completion.hpp"
#include "saccade_shipping/sha256.hpp"
#include "saccade_shipping/strict_json.hpp"
#include "saccade_shipping/verified_model_bytes.hpp"

namespace fs = std::filesystem;
namespace sh = saccade::shipping;
using sh::JsonValue;

namespace {

int g_failures = 0;
int g_checks = 0;

#ifdef CHECK
#undef CHECK
#endif
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failures;                                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << s;
    if (!f) throw std::runtime_error("cannot write " + p.string());
}

struct Layout {
    fs::path bundle, runtime, out;
    std::vector<std::string> sequences;
    std::string pin;
    JsonValue manifest;
};

Layout make_layout(const fs::path& repo, const fs::path& dir) {
    Layout l;
    l.bundle = dir / "bundle";
    l.runtime = dir / "runtime";
    l.out = dir / "out";
    const fs::path example =
        repo / "docs/architecture/model_bundle_549/examples/headline_s_n01_n06.model_bundle.example.json";
    l.manifest = sh::parse_strict_json(read_file(example));
    for (const JsonValue& m : l.manifest.find("members")->array) {
        const std::string& rel = m.find("path")->string;
        const bool in_bundle = m.find("carried_by")->string == "model_bundle";
        const fs::path dst = (in_bundle ? l.bundle : l.runtime) / rel;
        fs::create_directories(dst.parent_path());
        fs::copy_file(repo / rel, dst, fs::copy_options::overwrite_existing);
    }
    fs::copy_file(example, l.bundle / sh::kModelBundleManifestName, fs::copy_options::overwrite_existing);
    const std::string allowlist = read_file(repo / "shipping/trusted_model_bundles.json");
    write_file(l.runtime / sh::kTrustedModelBundlesName, allowlist);
    l.pin = sh::sha256_hex(allowlist.data(), allowlist.size());
    for (const char* name : {"SEQ-A"}) {
        const fs::path seq = dir / "seqs" / name;
        write_file(seq / "seqinfo.ini", std::string("[Sequence]\nname=") + name +
                                            "\nimWidth=8\nimHeight=6\nseqLength=1\n");
        write_file(seq / "img1" / "000001.jpg", "not decoded");
        l.sequences.push_back(seq.string());
    }
    return l;
}

sh::DetectorPlan gate_a(const Layout& l) {
    sh::RunCompletion completion(sh::new_run_id(), "saccade_shipping_model_bundle_gate_b_test",
                                 sh::RunOutputs{l.out, {}, {}, {"SEQ-A"}}, true);
    sh::PreflightInputs in;
    in.model_root.clear();
    in.sequences = l.sequences;
    in.model_bundle = l.bundle.string();
    in.runtime_root = l.runtime.string();
    in.allowlist_sha256 = l.pin;
    in.runtime_build = sh::native_runtime_build();
    sh::PreflightResult r = sh::run_preflight(in, completion);
    CHECK(r.detector.resolved != nullptr);
    // D-N03-5: the caller of Gate B names the runtime package's operator.
    r.detector.runtime_operator_path = (l.runtime / "build/libsaccade_scan_torchop.so").string();
    return r.detector;
}

std::string load_error(const sh::DetectorPlan& plan, const std::string& model_root, cudaStream_t stream) {
    try {
        sh::DetectorHost host(plan, model_root, stream);
    } catch (const sh::DetectorLoadError& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("not a DetectorLoadError: ") + e.what();
    }
    return "";
}

// The temp tree, removed with nftw: in this executable (LibTorch linked)
// std::filesystem::remove_all binds to a libtorch symbol that crashes.
void remove_tree(const fs::path& dir) {
    ::nftw(
        dir.c_str(), [](const char* p, const struct stat*, int, struct FTW*) { return ::remove(p); }, 16,
        FTW_DEPTH | FTW_PHYS);
}

bool mapped(const std::string& path) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.size() >= path.size() && line.compare(line.size() - path.size(), path.size(), path) == 0) return true;
    }
    return false;
}

bool mapped_memfd() {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("/memfd:saccade-n03:src-fd=") != std::string::npos) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <repo root>\n", argv[0]);
        return 2;
    }
    const fs::path repo = argv[1];
    for (const char* rel : {"models/yolo/yolo26s_backbone_640_best.engine",
                            "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt",
                            "build/libsaccade_scan_torchop.so"}) {
        if (!fs::is_regular_file(repo / rel)) {
            std::printf("SKIP: %s not present\n", rel);
            return 0;
        }
    }
    char tmpl[] = "/tmp/saccade_model_bundle_gate_b.XXXXXX";
    const fs::path base = ::mkdtemp(tmpl);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate failed\n");
        return 2;
    }
    try {
        // MB-55 first, before the operator library is mapped into this process.
        {
            const Layout l = make_layout(repo, base / "replaced");
            const sh::DetectorPlan plan = gate_a(l);
            const fs::path head = plan.resolved->head_artifact.absolute_path;
            std::string bytes = read_file(head);
            bytes[bytes.size() / 2] = static_cast<char>(bytes[bytes.size() / 2] ^ 1);
            write_file(head, bytes);
            const std::string e = load_error(plan, "", stream);
            std::fprintf(stderr, "replaced: %s\n", e.c_str());
            CHECK(e.find("shipping detector load: head artifact " + head.string() + " sha256 ") == 0);
            CHECK(!mapped(plan.resolved->op_library.absolute_path));  // refused before dlopen
            CHECK(!mapped_memfd());
        }
        {
            const Layout l = make_layout(repo, base / "unsupported_sm");
            sh::DetectorPlan plan = gate_a(l);
            auto contract = std::make_shared<sh::ManifestLoadContract>(*plan.manifest_contract);
            for (auto& target : contract->targets) target.gpu_sm = {"sm_999"};
            plan.manifest_contract = contract;
            // Even an unreadable engine path cannot cause deserialization:
            // the actual device's unsupported SM must be refused first.
            fs::remove(plan.resolved->backbone_engine.absolute_path);
            const std::string e = load_error(plan, "", stream);
            std::fprintf(stderr, "unsupported_sm: %s\n", e.c_str());
            CHECK(e.find("GPU SM") != std::string::npos);
            CHECK(!mapped(plan.resolved->op_library.absolute_path));
            CHECK(!mapped_memfd());
        }
        {
            // D-N03-5: the runtime package's operator is the only one; a plan
            // naming another file (the repository's identical build) or none
            // is refused before anything is read or mapped.
            const Layout l = make_layout(repo, base / "foreign_operator");
            sh::DetectorPlan plan = gate_a(l);
            plan.runtime_operator_path = fs::canonical(repo / "build/libsaccade_scan_torchop.so").string();
            const std::string e = load_error(plan, "", stream);
            std::fprintf(stderr, "foreign_operator: %s\n", e.c_str());
            CHECK(e == "shipping detector load: operator library " + plan.resolved->op_library.absolute_path +
                           " is not the runtime package's " + plan.runtime_operator_path + " (D-N03-5)");
            plan.runtime_operator_path.clear();
            CHECK(load_error(plan, "", stream) ==
                  "shipping detector load: no runtime package operator library was named (D-N03-5)");
            CHECK(!mapped_memfd());
        }
        {
            const Layout l = make_layout(repo, base / "model_root");
            const std::string e = load_error(gate_a(l), repo.string(), stream);
            std::fprintf(stderr, "model_root: %s\n", e.c_str());
            CHECK(e == "shipping detector load: a plan with resolved bindings has no model root");
        }
        {
            const Layout l = make_layout(repo, base / "not_the_plan");
            sh::DetectorPlan plan = gate_a(l);
            auto other = std::make_shared<sh::ResolvedLoadBindings>(*plan.resolved);
            other->backbone_engine.sha256 = std::string(64, '0');
            plan.resolved = other;
            const std::string e = load_error(plan, "", stream);
            std::fprintf(stderr, "not_the_plan: %s\n", e.c_str());
            CHECK(e == "shipping detector load: the resolved bindings are not the plan's");
        }
        {
            const Layout l = make_layout(repo, base / "cross_root");
            const sh::DetectorPlan plan = gate_a(l);
            const sh::ResolvedLoadBindings& r = *plan.resolved;
            CHECK(r.op_library.root_kind == "runtime_package");
            CHECK(r.head_artifact.root_kind == "model_bundle");
            CHECK(r.backbone_engine.root_kind == "model_bundle");
            sh::DetectorHost host(plan, "", stream);
            const sh::HeadLoadReport& rep = host.load_report();
            CHECK(rep.op_library_sha256 == r.op_library.sha256);
            CHECK(rep.head_artifact_sha256 == r.head_artifact.sha256);
            CHECK(rep.backbone_engine_sha256 == r.backbone_engine.sha256);
            CHECK(rep.head_artifact_byte_scope == "loaded_buffer");
            CHECK(rep.backbone_engine_byte_scope == "loaded_buffer");
            CHECK(rep.op_library_byte_scope == "loaded_buffer");
            CHECK(!rep.gpu_sm.empty());
            // The operator runs from the sealed memfd made from the runtime
            // root's copy (#549 S2-2); neither path is mapped.
            CHECK(rep.op_library_mapped.rfind("/memfd:saccade-n03:src-fd=", 0) == 0);
            CHECK(mapped_memfd());
            CHECK(!mapped(r.op_library.absolute_path));
            CHECK(!mapped(fs::canonical(repo / "build/libsaccade_scan_torchop.so").string()));
            std::fprintf(stderr, "cross_root: loaded %s from %s\n", r.op_library.absolute_path.c_str(),
                         rep.op_library_mapped.c_str());

            // MB-41/MB-58: actual LibTorch and TensorRT deserialization after
            // deterministic post-hash replacement loads the original buffers.
            // The operator is already registered by the real DetectorHost.
            const sh::VerifiedModelBytes head_bytes = sh::VerifiedModelBytes::read(
                "head artifact", r.head_artifact.absolute_path, r.head_artifact.sha256, r.head_artifact.bytes);
            const sh::VerifiedModelBytes engine_bytes = sh::VerifiedModelBytes::read(
                "backbone engine", r.backbone_engine.absolute_path, r.backbone_engine.sha256, r.backbone_engine.bytes);
            write_file(r.head_artifact.absolute_path, std::string(head_bytes.size(), '\0'));
            write_file(r.backbone_engine.absolute_path, std::string(engine_bytes.size(), '\0'));
            CHECK(sh::sha256_file_hex(r.head_artifact.absolute_path) != head_bytes.sha256());
            CHECK(sh::sha256_file_hex(r.backbone_engine.absolute_path) != engine_bytes.sha256());
            {
                sh::VerifiedModelStream verified_stream(head_bytes);
                torch::jit::Module verified_head = torch::jit::load(verified_stream);
                for (const auto& parameter : verified_head.parameters()) CHECK(parameter.device().str() == "cuda:0");
                saccade::TRTEngine verified_engine(engine_bytes.data(), engine_bytes.size());
                CHECK(verified_engine.get_nb_tensors() == 4);
                CHECK(std::string(verified_engine.get_tensor_name(0)) == "images");
            }
            std::fprintf(stderr, "post_hash_replaced: actual head/engine loaded original verified buffers\n");
            write_file(r.head_artifact.absolute_path, std::string(head_bytes.data(), head_bytes.size()));
            write_file(r.backbone_engine.absolute_path, std::string(engine_bytes.data(), engine_bytes.size()));

            // These contracts came through Gate A, then deliberately change a
            // single handoff field. The real engine readback must refuse them.
            for (const char* kind : {"dtype", "shape", "name", "order", "head_dtype"}) {
                sh::DetectorPlan mismatch = plan;
                auto contract = std::make_shared<sh::ManifestLoadContract>(*plan.manifest_contract);
                const std::string mutation = kind;
                if (mutation == "dtype") contract->backbone.inputs[0].dtype = "float16";
                if (mutation == "shape") contract->backbone.inputs[0].shape[2] = 641;
                if (mutation == "name") contract->backbone.inputs[0].name = "unexpected";
                if (mutation == "order") std::swap(contract->backbone.outputs[0], contract->backbone.outputs[1]);
                if (mutation == "head_dtype") contract->head.inputs[0].dtype = "float16";
                mismatch.manifest_contract = contract;
                const std::string e = load_error(mismatch, "", stream);
                std::fprintf(stderr, "%s: %s\n", kind, e.c_str());
                CHECK(!e.empty());
                CHECK(e.find("not a DetectorLoadError") == std::string::npos);
                CHECK(e.find(mutation == "head_dtype" ? "head supported contract" : "backbone engine") != std::string::npos);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    cudaStreamDestroy(stream);
    remove_tree(base);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
