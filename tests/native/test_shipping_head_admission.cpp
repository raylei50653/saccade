// TorchScript head admission (#549 MB-41, F-1; saccade_shipping/head_admission.hpp).
// CPU only, LibTorch linked (ENABLE_NATIVE_TESTS). The N02 positive control
// needs the models and the attested operator library, and is skipped without
// them.
//
// Usage: saccade_shipping_head_admission_test <repo root>
//
//   * G-1 (fixtures/g1_heads.hpp): each head is first shown to read its
//     sidecar under plain torch::jit::load (the value follows the file). Then
//     the sidecar is removed, so any read would fail with another error, and:
//     - setstate and flatbuffer are refused before the load;
//     - forward and fork load without reading and are refused after it. For
//       fork, aten::from_file is found only in the prim::fork Subgraph.
//   * Single changes to an archive that is otherwise admitted:
//     - before the load: an extra record, a custom class, a class the code
//       does not define, another global (in data and in the debug ranges),
//       an opcode the unpickler does not read, protocol 3, a truncated
//       pickle, code outside a def or class;
//     - after the load: an operator off the list (aten::sin), control flow
//       (prim::If), a from_file method that forward never calls, a call
//       through a module interface (it does not resolve statically), and a
//       call into another compilation unit (it resolves, but not to one of
//       the head's own functions).
//   * Another LibTorch build refuses every head, before and after the load.
//   * Positive controls: a small head, the same head rewritten, and N02.
//     N02 uses exactly the reviewed list, so the list is no wider than the
//     reviewed head.
#include <caffe2/serialize/inline_container.h>
#include <dlfcn.h>
#include <ftw.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/script.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fixtures/g1_heads.hpp"
#include "saccade_shipping/head_admission.hpp"
#include "saccade_shipping/model_compatibility.hpp"
#include "saccade_shipping/sha256.hpp"
#include "saccade_shipping/verified_model_bytes.hpp"

namespace fs = std::filesystem;
namespace sh = saccade::shipping;

namespace {
int failures = 0;
int checks = 0;
#ifdef CHECK
#undef CHECK
#endif
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; \
    std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition); } } while (0)

const std::string kBuild = sh::kHeadAdmissionLibTorch;
const std::string kPrefix = "shipping detector load: head admission (F-1): ";

void write(const fs::path& path, const std::string& value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// The bytes as Gate B holds them: written, then read back and hashed.
sh::VerifiedModelBytes verified(const fs::path& dir, const std::string& bytes) {
    const fs::path p = dir / "head.pt";
    write(p, bytes);
    return sh::VerifiedModelBytes::read("head artifact", p.string(), sh::sha256_hex(bytes.data(), bytes.size()));
}

std::string archive_error(const fs::path& dir, const std::string& bytes, const std::string& build = kBuild) {
    try {
        sh::admit_head_archive(verified(dir, bytes), build);
    } catch (const sh::DetectorLoadError& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("not a DetectorLoadError: ") + e.what();
    }
    return "";
}

torch::jit::Module load(const std::string& bytes) {
    std::istringstream in(bytes);
    return torch::jit::load(in, torch::kCPU);
}

std::string module_error(const torch::jit::Module& m, const std::string& build = kBuild,
                         std::vector<std::string>* used = nullptr) {
    try {
        std::vector<std::string> u = sh::admit_head_module(m, build);
        if (used) *used = std::move(u);
    } catch (const sh::DetectorLoadError& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("not a DetectorLoadError: ") + e.what();
    }
    return "";
}

bool starts(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }
bool has(const std::string& s, const std::string& p) { return s.find(p) != std::string::npos; }

// A copy of a ZIP archive with records edited (name -> bytes; an empty name
// list entry removes nothing). The writer adds version, byteorder and the
// serialization id itself.
using Records = std::vector<std::pair<std::string, std::string>>;
std::string rewrite(const std::string& bytes, const std::function<void(Records&)>& edit) {
    std::istringstream in(bytes);
    caffe2::serialize::PyTorchStreamReader reader(&in);
    Records records;
    for (const std::string& name : reader.getAllRecords()) {
        if (name == "version" || name == ".data/version" || name == "byteorder" || name == ".data/serialization_id") {
            continue;
        }
        auto [data, size] = reader.getRecord(name);
        records.emplace_back(name, std::string(static_cast<const char*>(data.get()), size));
    }
    edit(records);
    std::string out;
    {
        caffe2::serialize::PyTorchStreamWriter writer([&](const void* p, size_t n) {
            out.append(static_cast<const char*>(p), n);
            return n;
        });
        writer.setMinVersion(reader.version());
        for (const auto& [name, data] : records) writer.writeRecord(name, data.data(), data.size());
        writer.writeEndOfFile();
    }
    return out;
}

std::string& record(Records& r, const std::string& name) {
    for (auto& [n, d] : r) {
        if (n == name) return d;
    }
    throw std::runtime_error("no record " + name);
}

void replace_once(std::string& s, const std::string& from, const std::string& to) {
    const auto at = s.find(from);
    if (at == std::string::npos || s.find(from, at + 1) != std::string::npos) {
        throw std::runtime_error("expected one occurrence of the pickle bytes to change");
    }
    s.replace(at, from.size(), to);
}

// A small benign head: forward adds a tensor attribute.
std::string benign(const std::string& extra_source = "", const std::string& forward = "torch.add(x, self.w)") {
    torch::jit::Module m("__torch__.Benign");
    m.register_attribute("path", c10::StringType::get(), std::string("/nonexistent"));
    m.register_attribute("w", c10::TensorType::get(), torch::ones({1}));
    m.define("def forward(self, x: Tensor) -> Tensor:\n    return " + forward + "\n" + extra_source);
    std::ostringstream out;
    m.save(out);
    return out.str();
}

// forward calls `one` through a module interface attribute: the call's target
// is not known until the attribute's value is, so it does not resolve.
std::string interface_call() {
    torch::jit::Module sub("__torch__.Sub");
    sub.define("def one(self, x: Tensor) -> Tensor:\n    return x\n");
    torch::jit::Module m("__torch__.Caller");
    const auto cu = m._ivalue()->compilation_unit();
    torch::jit::Parser p(std::make_shared<torch::jit::Source>(
        "class OneInterface(ModuleInterface):\n    def one(self, x: Tensor) -> Tensor:\n        pass\n"));
    const torch::jit::ClassDef def(p.parseClass());
    cu->define_interface(c10::QualifiedName("__torch__.OneInterface"), def, torch::jit::nativeResolver(),
                         /*is_module=*/true);
    m.register_attribute("sub", cu->get_interface("__torch__.OneInterface"), sub._ivalue());
    m.define("def forward(self, x: Tensor) -> Tensor:\n    return self.sub.one(x)\n");
    std::ostringstream out;
    m.save(out);
    return out.str();
}

// The temp tree, removed with nftw: in a LibTorch-linked executable
// std::filesystem::remove_all binds to a libtorch symbol that crashes.
void remove_tree(const fs::path& dir) {
    ::nftw(
        dir.c_str(), [](const char* p, const struct stat*, int, struct FTW*) { return ::remove(p); }, 16,
        FTW_DEPTH | FTW_PHYS);
}

// A walk of nested blocks only (what Gate B's forward check walked before F-1).
bool block_walk_sees(torch::jit::Block* b, const std::string& kind) {
    for (torch::jit::Node* n : b->nodes()) {
        if (n->kind().toQualString() == kind) return true;
        for (torch::jit::Block* sub : n->blocks()) {
            if (block_walk_sees(sub, kind)) return true;
        }
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
    // As Gate B sets it before the load (the lineage's runtime requirement).
    torch::jit::setGraphExecutorOptimize(false);
    char tmpl[] = "/tmp/saccade_head_admission.XXXXXX";
    const fs::path dir = ::mkdtemp(tmpl);
    try {
        // ── G-1 ──────────────────────────────────────────────────────────
        for (const std::string& kind : g1::kinds()) {
            const fs::path sidecar = dir / ("sidecar_" + kind + ".bin");
            const std::string bytes = g1::archive(kind, sidecar.string());
            // The control reads the file: its value follows the sidecar.
            g1::write_sidecar(sidecar.string(), 60.0f);
            const float first = g1::read_value(kind, bytes);
            g1::write_sidecar(sidecar.string(), 7.0f);
            const float second = g1::read_value(kind, bytes);
            std::fprintf(stderr, "%s: plain load reads the sidecar: %.1f then %.1f\n", kind.c_str(), first, second);
            CHECK(first == 60.0f && second == 7.0f);
            // Without the sidecar a read fails with LibTorch's own error.
            fs::remove(sidecar);
            std::string direct;
            try {
                (void)g1::read_value(kind, bytes);
            } catch (const std::exception& e) {
                direct = e.what();
            }
            CHECK(!direct.empty());

            const std::string pre = archive_error(dir, bytes);
            std::fprintf(stderr, "%s: before the load: %s\n", kind.c_str(), pre.empty() ? "admitted" : pre.c_str());
            if (kind == "setstate") {
                CHECK(pre == kPrefix + "class __torch__.G1setstate defines __setstate__, which the loader runs");
                continue;
            }
            if (kind == "flatbuffer") {
                CHECK(pre == kPrefix + "the head is not a ZIP TorchScript archive (FlatBuffer and other formats are "
                                       "refused)");
                continue;
            }
            // forward / fork: admitted before the load, and the load does not
            // read (the sidecar is gone and the load succeeds).
            CHECK(pre.empty());
            torch::jit::Module m = load(bytes);
            const std::string post = module_error(m);
            std::fprintf(stderr, "%s: after the load: %s\n", kind.c_str(), post.c_str());
            CHECK(starts(post, kPrefix + "function __torch__.G1" + kind + ".forward uses "));
            CHECK(has(post, "aten::from_file"));
            if (kind == "fork") {
                CHECK(has(post, "prim::fork"));
                auto graph = m.get_method("forward").graph()->copy();
                torch::jit::Inline(*graph);
                CHECK(!block_walk_sees(graph->block(), "aten::from_file"));  // only in the Subgraph
            }
            CHECK(!fs::exists(sidecar));
        }

        // ── positive: a small head, and the same head rewritten ─────────
        const std::string ok = benign();
        CHECK(archive_error(dir, ok).empty());
        std::vector<std::string> used;
        CHECK(module_error(load(ok), kBuild, &used).empty());
        CHECK((used == std::vector<std::string>{"aten::add.Tensor", "prim::Constant", "prim::GetAttr"}));
        const std::string same = rewrite(ok, [](Records&) {});
        CHECK(archive_error(dir, same).empty());
        CHECK(module_error(load(same)).empty());

        // ── before the load: one change each ─────────────────────────────
        const auto pre_case = [&](const char* name, const std::string& bytes, const std::string& want) {
            const std::string e = archive_error(dir, bytes);
            std::fprintf(stderr, "%s: %s\n", name, e.c_str());
            CHECK(starts(e, kPrefix));
            CHECK(has(e, want));
        };
        pre_case("extra_record",
                 rewrite(ok, [](Records& r) { r.emplace_back("extra/note.txt", "x"); }),
                 "archive record extra/note.txt is not one the head loader reads");
        pre_case("custom_class", rewrite(ok, [](Records& r) {
                     replace_once(record(r, "data.pkl"), "c__torch__\nBenign\n",
                                  "c__torch__.torch.classes.quantized\nLinearPackedParamsBase\n");
                 }),
                 "restores a custom class __torch__.torch.classes.quantized.LinearPackedParamsBase");
        pre_case("undefined_class", rewrite(ok, [](Records& r) {
                     replace_once(record(r, "data.pkl"), "c__torch__\nBenign\n", "c__torch__\nNoSuchHead\n");
                 }),
                 "restores __torch__.NoSuchHead, which the archive's code does not define");
        pre_case("other_global", rewrite(ok, [](Records& r) {
                     replace_once(record(r, "data.pkl"), "ctorch._utils\n_rebuild_tensor_v2\n",
                                  "ctorch._utils\n_rebuild_qtensor\n");
                 }),
                 "data.pkl uses the global torch._utils._rebuild_qtensor");
        pre_case("debug_global", rewrite(ok, [](Records& r) {
                     replace_once(record(r, "code/__torch__.py.debug_pkl"), "ctorch.jit._pickle\nbuild_intlist\n",
                                  "ctorch.jit._pickle\nbuild_tensorlist\n");
                 }),
                 "debug ranges code/__torch__.py.debug_pkl name torch.jit._pickle.build_tensorlist");
        pre_case("opcode", rewrite(ok, [](Records& r) { record(r, "data.pkl").insert(2, "I0\n"); }),
                 "is not one LibTorch's unpickler reads");
        pre_case("protocol", rewrite(ok, [](Records& r) { record(r, "data.pkl")[1] = 3; }), "protocol is not 2");
        pre_case("truncated", rewrite(ok, [](Records& r) {
                     std::string& d = record(r, "data.pkl");
                     d.resize(d.size() / 2);
                 }),
                 "pickle data.pkl at byte");
        pre_case("top_level", rewrite(ok, [](Records& r) { record(r, "code/__torch__.py") += "x = 1\n"; }),
                 "at top level");
        {
            const std::string e = archive_error(dir, ok, "2.10.0+cu130");
            CHECK(e == kPrefix + "the operator list was reviewed for LibTorch " + kBuild +
                           ", this runtime is built against 2.10.0+cu130");
            CHECK(starts(module_error(load(ok), "2.10.0+cu130"), kPrefix + "the operator list was reviewed"));
        }

        // ── after the load: one change each ──────────────────────────────
        const auto post_case = [&](const char* name, const std::string& bytes, const std::string& want) {
            CHECK(archive_error(dir, bytes).empty());
            const std::string e = module_error(load(bytes));
            std::fprintf(stderr, "%s: %s\n", name, e.c_str());
            CHECK(starts(e, kPrefix));
            CHECK(has(e, want));
        };
        post_case("off_list", benign("", "torch.sin(x)"), "uses aten::sin, not among the reviewed head operators");
        post_case("control_flow", benign("", "x if bool(x.sum() > 0) else self.w"), "prim::If");
        post_case("uncalled_method",
                  benign("def helper(self) -> Tensor:\n    return torch.from_file(self.path, False, 1, dtype=6)\n"),
                  "function __torch__.Benign.helper uses aten::from_file");
        post_case("interface_call", interface_call(),
                  "function __torch__.Caller.forward makes a call that does not resolve to the head's own functions: "
                  "prim::CallMethod one on __torch__.OneInterface");

        {
            // In memory, each Module has its own compilation unit: outer's
            // call resolves to inner's method, which is not outer's own.
            torch::jit::Module inner("__torch__.Inner");
            inner.define("def forward(self, x: Tensor) -> Tensor:\n    return x\n");
            torch::jit::Module outer("__torch__.Outer");
            outer.register_module("inner", inner);
            outer.define("def forward(self, x: Tensor) -> Tensor:\n    return self.inner.forward(x)\n");
            const std::string e = module_error(outer);
            std::fprintf(stderr, "foreign_function: %s\n", e.c_str());
            CHECK(e == kPrefix +
                           "function __torch__.Outer.forward makes a call that does not resolve to the head's own "
                           "functions: prim::CallMethod forward on __torch__.Inner");
        }

        // ── N02: admitted, and its operators are exactly the list ───────
        const fs::path n02 = repo / "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt";
        const fs::path op = repo / "build/libsaccade_scan_torchop.so";
        if (fs::is_regular_file(n02) && fs::is_regular_file(op)) {
            if (::dlopen(op.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) throw std::runtime_error(::dlerror());
            const std::string bytes = read(n02);
            CHECK(sh::sha256_hex(bytes.data(), bytes.size()).rfind("1663ec97", 0) == 0);
            CHECK(archive_error(dir, bytes).empty());
            std::vector<std::string> n02_used;
            CHECK(module_error(load(bytes), kBuild, &n02_used).empty());
            const std::vector<std::string> list(sh::kHeadAdmissionOperators,
                                                sh::kHeadAdmissionOperators + sh::kHeadAdmissionOperatorCount);
            CHECK(std::is_sorted(list.begin(), list.end()));
            CHECK(n02_used == list);
            std::fprintf(stderr, "n02: admitted, %zu operators (the list has %zu)\n", n02_used.size(), list.size());
        } else {
            std::printf("SKIP n02: model or operator library not present\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    remove_tree(dir);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
