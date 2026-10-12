// The G-1 reproductions (#549 MB-41, docs/reference/model_bundle_mb41_549.md)
// as TorchScript heads, built in-process with LibTorch, no Python. Each head
// reads `sidecar`, a float32 file outside the archive that was never hashed,
// with aten::from_file:
//   setstate:   a scripted __setstate__ reads it while the archive loads;
//   forward:    forward reads it on every call;
//   fork:       forward reads it through fork (the read sits in the
//               prim::fork node's Subgraph);
//   flatbuffer: the setstate head, saved as FlatBuffer instead of ZIP;
//   default:    a parameter default that the importer evaluates while the
//               archive loads (F-1 review round 1). The code record is
//               written by hand, as an archive's author could; a saved
//               module would carry the evaluated value instead.
// read_value(kind, bytes) returns what the head made of the sidecar under
// plain torch::jit::load. The tests use it to show each control really reads
// the file before they check that Gate B refuses it.
#pragma once

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/script.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace g1 {

inline const std::vector<std::string>& kinds() {
    static const std::vector<std::string> k = {"setstate", "forward", "fork", "flatbuffer", "default"};
    return k;
}

// A copy of a ZIP archive with its records edited. The writer adds version,
// byteorder and the serialization id itself.
using Records = std::vector<std::pair<std::string, std::string>>;
inline std::string rewrite(const std::string& bytes, const std::function<void(Records&)>& edit) {
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

inline std::string& record(Records& r, const std::string& name) {
    for (auto& [n, d] : r) {
        if (n == name) return d;
    }
    throw std::runtime_error("no record " + name);
}

inline void replace_once(std::string& s, const std::string& from, const std::string& to) {
    const auto at = s.find(from);
    if (at == std::string::npos || s.find(from, at + 1) != std::string::npos) {
        throw std::runtime_error("expected exactly one occurrence of: " + from);
    }
    s.replace(at, from.size(), to);
}

// C++ define() resolves only `torch`. The importer's environment also has
// `fork` (the special form that serialized TorchScript code calls).
struct ForkResolver : torch::jit::Resolver {
    std::shared_ptr<torch::jit::SugaredValue> resolveValue(const std::string& name, torch::jit::GraphFunction& m,
                                                           const torch::jit::SourceRange& loc) override {
        if (name == "fork") return torch::jit::SpecialFormValue::create(c10::prim::fork);
        return torch::jit::nativeResolver()->resolveValue(name, m, loc);
    }
    c10::TypePtr resolveType(const std::string&, const torch::jit::SourceRange&) override { return nullptr; }
};

inline torch::jit::Module module(const std::string& kind, const std::string& sidecar) {
    torch::jit::Module m("__torch__.G1" + kind);
    m.register_attribute("path", c10::StringType::get(), sidecar);
    m.register_attribute("w", c10::TensorType::get(), torch::zeros({1}));
    if (kind == "setstate" || kind == "flatbuffer") {
        m.define(R"(
def forward(self, x: Tensor) -> Tensor:
    return torch.add(x, self.w)

def __getstate__(self) -> Tuple[str]:
    return (self.path,)

def __setstate__(self, state: Tuple[str]) -> None:
    self.path = state[0]
    self.w = torch.from_file(self.path, False, 1, dtype=6)
)");
    } else if (kind == "forward") {
        m.define(R"(
def forward(self, x: Tensor) -> Tensor:
    return torch.add(x, torch.from_file(self.path, False, 1, dtype=6))
)");
    } else if (kind == "default") {
        m.define(R"(
def forward(self, x: Tensor, y: Tensor) -> Tensor:
    return torch.add(x, y)
)");
    } else if (kind == "fork") {
        m.define(R"(
def forward(self, x: Tensor) -> Tensor:
    path = self.path
    def __forked_function():
        _0 = torch.from_file(path, False, 1, dtype=6)
        return _0
    fut = fork(__forked_function)
    return torch.add(x, torch.wait(fut))
)",
                 std::make_shared<ForkResolver>());
    } else {
        throw std::runtime_error("unknown G-1 kind " + kind);
    }
    return m;
}

// The head's archive bytes (ZIP, or FlatBuffer for "flatbuffer").
inline std::string archive(const std::string& kind, const std::string& sidecar) {
    torch::jit::Module m = module(kind, sidecar);
    if (kind == "flatbuffer") {
        auto buf = torch::jit::save_jit_module_to_bytes(m);
        return std::string(reinterpret_cast<const char*>(buf->data()), buf->size());
    }
    std::ostringstream out;
    m.save(out);
    if (kind != "default") return out.str();
    // forward's `y` gets a default the importer evaluates. Its debug ranges
    // no longer match the text, so they go too.
    return rewrite(out.str(), [&](Records& r) {
        replace_once(record(r, "code/__torch__.py"), "y: Tensor) -> Tensor:",
                     "y: Tensor=torch.from_file(\"" + sidecar + "\", False, 1, dtype=6)) -> Tensor:");
        r.erase(std::remove_if(r.begin(), r.end(), [](const auto& e) { return e.first == "code/__torch__.py.debug_pkl"; }),
                r.end());
    });
}

// Plain torch::jit::load, then the value the head took from the sidecar
// (the state for setstate/flatbuffer, forward(0) for forward/fork/default).
inline float read_value(const std::string& kind, const std::string& bytes) {
    std::istringstream in(bytes);
    torch::jit::Module m = torch::jit::load(in, torch::kCPU);
    at::Tensor zero = torch::zeros({1});
    at::Tensor v = (kind == "setstate" || kind == "flatbuffer") ? m.attr("w").toTensor()
                                                               : m.forward({zero}).toTensor();
    return v.item<float>();
}

inline void write_sidecar(const std::string& path, float value) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&value), sizeof value);
    if (!f) throw std::runtime_error("cannot write " + path);
}

}  // namespace g1
