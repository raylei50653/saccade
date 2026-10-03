// Native detector plan vs the resolved config, the frozen PR-1L lineage and
// the realization attestation (#465 Phase B PR-8, U3b-2). CPU only; CI job
// `shipping-config-loader`.
//
// Usage: saccade_shipping_detector_plan_test <resolved config> <lineage> <attestation>
//
// The lineage is tests/native/fixtures/shipping_head_lineage.json, a byte copy
// of the frozen PR-1L lineage (tests/unit/test_native_detector_oracle_pins.py
// keeps it equal to models/yolo/ when that file exists); the attestation is the
// committed configs/shipping/mamba_head_realization.attestation.json. Checks:
//   * SHA-256 against the FIPS 180-4 vectors, one-shot and split at every
//     block-boundary offset;
//   * plan_detector accepts the headline config and reads every value from
//     the config / lineage / attestation (no fallback);
//   * every detector gate of §12.2 is refused when flipped;
//   * every lineage field the plan reads is refused when changed (checked
//     without the attestation, which would refuse any lineage byte change);
//   * every attestation field is refused when changed, and the attestation
//     refuses a lineage whose bytes changed;
//   * without an attestation the operator library is the lineage's build.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/sha256.hpp"
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

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

JsonValue* path(JsonValue& doc, std::initializer_list<const char*> keys) {
    JsonValue* v = &doc;
    for (const char* k : keys) {
        JsonValue* next = v->find(k);
        if (next == nullptr) throw std::runtime_error(std::string("no key ") + k);
        v = next;
    }
    return v;
}

struct Inputs {
    std::string config, lineage, attestation;
};

sh::DetectorPlan plan_of(const JsonValue& config, const std::string& lineage_text,
                         const JsonValue* attestation) {
    const sh::ResolvedShippingConfig cfg = sh::load_resolved_shipping_config(config);
    return sh::plan_detector(cfg, sh::parse_strict_json(lineage_text),
                             sh::sha256_hex(lineage_text.data(), lineage_text.size()), attestation);
}

bool refused(const std::function<void()>& f, const std::string& what) {
    try {
        f();
    } catch (const sh::ConfigError&) {
        return true;
    }
    std::fprintf(stderr, "not refused: %s\n", what.c_str());
    return false;
}

void check_sha256() {
    auto one = [](const std::string& s) { return sh::sha256_hex(s.data(), s.size()); };
    CHECK(one("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(one("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(one(two_blocks) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const std::string million(1000000, 'a');
    CHECK(one(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // Split updates across every offset of a 130-byte message.
    std::string msg;
    for (int i = 0; i < 130; ++i) msg.push_back(static_cast<char>(i * 7 + 3));
    const std::string want = one(msg);
    bool all = true;
    for (std::size_t cut = 0; cut <= msg.size(); ++cut) {
        sh::Sha256 h;
        h.update(msg.data(), cut);
        h.update(msg.data() + cut, msg.size() - cut);
        all = all && h.hex_digest() == want;
    }
    CHECK(all);
}

void check_accepts(const Inputs& in) {
    const JsonValue cfg = sh::parse_strict_json(in.config);
    const JsonValue att = sh::parse_strict_json(in.attestation);
    const sh::DetectorPlan p = plan_of(cfg, in.lineage, &att);
    CHECK(p.img_size == 640);
    CHECK(p.max_det == 300);
    CHECK(p.in_channels[0] == 128 && p.in_channels[1] == 256 && p.in_channels[2] == 512);
    CHECK(p.num_classes == 80);
    CHECK(p.reg_channels == 4);
    CHECK(p.anchors == 8400);
    CHECK(p.feature_shapes[0][3] == 80 && p.feature_shapes[1][3] == 40 && p.feature_shapes[2][3] == 20);
    CHECK(p.native_scan_calls == 3);
    CHECK(!p.runtime.graph_executor_optimize && !p.runtime.cudnn_benchmark && p.runtime.cudnn_allow_tf32 &&
          !p.runtime.matmul_allow_tf32);
    const JsonValue lin = sh::parse_strict_json(in.lineage);
    CHECK(p.head_artifact.path == lin.find("torchscript")->find("path")->string);
    CHECK(p.head_artifact.sha256 == lin.find("torchscript")->find("sha256")->string);
    CHECK(p.backbone_engine.path == "models/yolo/yolo26s_backbone_640_best.engine");
    CHECK(p.backbone_engine.sha256 ==
          lin.find("companions")->find("backbone_engine")->find("sha256")->string);
    CHECK(p.op_library_from_attestation);
    CHECK(p.op_library.sha256 == att.find("op_library")->find("sha256")->string);
    CHECK(p.op_library_lineage_sha256 == lin.find("op_library")->find("sha256")->string);
    CHECK(p.op_library.sha256 != p.op_library_lineage_sha256);
    CHECK(p.conf_thr_unused == 0.001);

    // Without an attestation: the lineage's own build.
    const sh::DetectorPlan q = plan_of(cfg, in.lineage, nullptr);
    CHECK(!q.op_library_from_attestation);
    CHECK(q.op_library.sha256 == q.op_library_lineage_sha256);

    // set_whole_graph_img_dims: float32(double(orig) / img_size).
    CHECK(sh::coordinate_scale(1920, 640) == 3.0f);
    CHECK(sh::coordinate_scale(1080, 640) == 1.6875f);
    CHECK(sh::coordinate_scale(480, 640) == 0.75f);
    CHECK(sh::coordinate_scale(563, 640) == static_cast<float>(563.0 / 640.0));
}

void check_config_gates(const Inputs& in) {
    const JsonValue att = sh::parse_strict_json(in.attestation);
    struct Flip {
        std::vector<const char*> keys;
        JsonValue value;
    };
    const std::vector<Flip> flips = {
        {{"host_params", "detector", "build", "use_whole_graph"}, JsonValue::make_bool(false)},
        {{"host_params", "detector", "build", "trt_head_engine"}, JsonValue::make_string("x.engine")},
        {{"host_params", "detector", "build", "small_p3_max_threshold"}, JsonValue::make_float(0.1)},
        {{"host_params", "detector", "build", "postprocess_compile"}, JsonValue::make_bool(false)},
        {{"host_params", "detector", "build", "img_size"}, JsonValue::make_int(600)},
        {{"host_params", "detector", "build", "mamba_ckpt"}, JsonValue::make_string("runs/other.ckpt")},
        {{"host_params", "detector", "build", "trt_backbone_engine"}, JsonValue::make_string("other.engine")},
        {{"host_params", "detector", "build", "yolo_pt_path"}, JsonValue::make_string("other.pt")},
        {{"host_params", "detector", "build", "teacher_ckpt"}, JsonValue::make_string("other.ckpt")},
        {{"host_params", "detector", "head_calls", "set_head_compile"},
         JsonValue::make_array({JsonValue::make_bool(false)})},
        {{"host_params", "detector", "head_calls", "set_block_compile"},
         JsonValue::make_array({JsonValue::make_bool(false)})},
        {{"host_params", "detector", "detect_fn"}, JsonValue::make_string("detect_native_960")},
        {{"host_params", "detector", "contract", "feature_dim"}, JsonValue::make_int(128)},
        {{"host_params", "detector", "contract", "fpn_reid_mode"}, JsonValue::make_bool(true)},
        {{"host_params", "detector", "contract", "box_format"}, JsonValue::make_string("cxcywh")},
        {{"host_params", "cfg", "detector_box_format"}, JsonValue::make_string("cxcywh")},
        {{"host_params", "cfg", "tiling"}, JsonValue::make_string("native_960")},
        {{"host_params", "cfg", "kwargs.tiling"}, JsonValue::make_string("native_960")},
        {{"host_params", "cfg", "tta"}, JsonValue::make_bool(true)},
        {{"host_params", "cfg", "preprocess_modes"},
         JsonValue::make_array({JsonValue::make_string("letterbox")})},
        {{"host_params", "steps", "ingest.nv12_buffer"}, JsonValue::make_bool(true)},
        {{"host_params", "steps", "track.workbench"}, JsonValue::make_bool(true)},
        {{"source", "preset_sha256"},
         JsonValue::make_string("0000000000000000000000000000000000000000000000000000000000000000")},
    };
    for (const Flip& f : flips) {
        JsonValue cfg = sh::parse_strict_json(in.config);
        JsonValue* slot = &cfg;
        std::string name;
        for (const char* k : f.keys) {
            slot = slot->find(k);
            if (slot == nullptr) throw std::runtime_error(std::string("no config key ") + k);
            name += std::string(".") + k;
        }
        *slot = f.value;
        CHECK(refused([&] { plan_of(cfg, in.lineage, &att); }, "config" + name));
    }
}

// Lineage fields, checked without the attestation (which pins the bytes).
void check_lineage_fields(const Inputs& in) {
    const JsonValue cfg = sh::parse_strict_json(in.config);
    using Mut = std::function<void(JsonValue&)>;
    const std::vector<std::pair<std::string, Mut>> muts = {
        {"schema", [](JsonValue& l) { *path(l, {"schema"}) = JsonValue::make_string("other/v1"); }},
        {"tool.git_dirty", [](JsonValue& l) { *path(l, {"tool", "git_dirty"}) = JsonValue::make_bool(true); }},
        {"preset.sha256",
         [](JsonValue& l) {
             *path(l, {"preset", "sha256"}) =
                 JsonValue::make_string("1111111111111111111111111111111111111111111111111111111111111111");
         }},
        {"preset.mamba_ckpt", [](JsonValue& l) { *path(l, {"preset", "mamba_ckpt"}) = JsonValue::make_string("x"); }},
        {"preset.fpn_backbone_engine",
         [](JsonValue& l) { *path(l, {"preset", "fpn_backbone_engine"}) = JsonValue::make_string("x"); }},
        {"inventory.ckpt_sha256_match",
         [](JsonValue& l) { *path(l, {"inventory", "ckpt_sha256_match"}) = JsonValue::make_bool(false); }},
        {"inventory.backbone_engine_sha256_match",
         [](JsonValue& l) { *path(l, {"inventory", "backbone_engine_sha256_match"}) = JsonValue::make_bool(false); }},
        {"source.mamba_args.use_detail_fusion",
         [](JsonValue& l) {
             *path(l, {"source", "mamba_args", "use_detail_fusion"}) = JsonValue::make_bool(true);
         }},
        {"source.mamba_args.reg_max",
         [](JsonValue& l) { path(l, {"source", "mamba_args"})->set("reg_max", JsonValue::make_int(16)); }},
        {"source.mamba_args.use_strip_detail",
         [](JsonValue& l) { path(l, {"source", "mamba_args"})->set("use_strip_detail", JsonValue::make_bool(true)); }},
        {"head_load.missing_keys",
         [](JsonValue& l) {
             *path(l, {"head_load", "missing_keys"}) = JsonValue::make_array({JsonValue::make_string("w")});
         }},
        {"head_load.use_detail_fusion",
         [](JsonValue& l) { *path(l, {"head_load", "use_detail_fusion"}) = JsonValue::make_bool(true); }},
        {"torchscript.inputs.p4",
         [](JsonValue& l) {
             path(l, {"torchscript", "inputs", "p4"})->array[2] = JsonValue::make_int(41);
         }},
        {"torchscript.outputs",
         [](JsonValue& l) {
             auto& a = path(l, {"torchscript", "outputs"})->array;
             std::swap(a[0], a[3]);
         }},
        {"torchscript.dtype", [](JsonValue& l) { *path(l, {"torchscript", "dtype"}) = JsonValue::make_string("float16"); }},
        {"torchscript.native_scan_calls",
         [](JsonValue& l) { *path(l, {"torchscript", "native_scan_calls"}) = JsonValue::make_int(0); }},
        {"torchscript.sha256", [](JsonValue& l) { *path(l, {"torchscript", "sha256"}) = JsonValue::make_string("abc"); }},
        {"structural_check.bitwise_equal_all",
         [](JsonValue& l) { *path(l, {"structural_check", "bitwise_equal_all"}) = JsonValue::make_bool(false); }},
        {"op_library.op",
         [](JsonValue& l) { *path(l, {"op_library", "op"}) = JsonValue::make_string("saccade::selective_scan_fwd"); }},
        {"op_library.needed",
         [](JsonValue& l) {
             path(l, {"op_library", "needed"})->array.push_back(JsonValue::make_string("libtorch_python.so"));
         }},
        {"runtime_requirements.graph_executor_optimize",
         [](JsonValue& l) {
             *path(l, {"runtime_requirements", "graph_executor_optimize"}) = JsonValue::make_bool(true);
         }},
        {"companions.backbone_engine.path",
         [](JsonValue& l) { *path(l, {"companions", "backbone_engine", "path"}) = JsonValue::make_string("x"); }},
    };
    for (const auto& [name, mut] : muts) {
        JsonValue lin = sh::parse_strict_json(in.lineage);
        mut(lin);
        const std::string text = sh::dump_python_json(lin);
        CHECK(refused([&] { plan_of(cfg, text, nullptr); }, "lineage." + name));
    }
    // With the attestation, any change of the lineage bytes is refused.
    const JsonValue att = sh::parse_strict_json(in.attestation);
    CHECK(refused([&] { plan_of(cfg, in.lineage + " ", &att); }, "lineage bytes vs attestation"));
}

void check_attestation_fields(const Inputs& in) {
    const JsonValue cfg = sh::parse_strict_json(in.config);
    const std::string other = "2222222222222222222222222222222222222222222222222222222222222222";
    using Mut = std::function<void(JsonValue&)>;
    const std::vector<std::pair<std::string, Mut>> muts = {
        {"schema", [](JsonValue& a) { *path(a, {"schema"}) = JsonValue::make_string("other/v1"); }},
        {"frozen_lineage.sha256", [&](JsonValue& a) { *path(a, {"frozen_lineage", "sha256"}) = JsonValue::make_string(other); }},
        {"frozen_lineage.torchscript_sha256",
         [&](JsonValue& a) { *path(a, {"frozen_lineage", "torchscript_sha256"}) = JsonValue::make_string(other); }},
        {"frozen_lineage.torchscript_content_sha256",
         [&](JsonValue& a) {
             *path(a, {"frozen_lineage", "torchscript_content_sha256"}) = JsonValue::make_string(other);
         }},
        {"frozen_lineage.op_library_sha256",
         [&](JsonValue& a) { *path(a, {"frozen_lineage", "op_library_sha256"}) = JsonValue::make_string(other); }},
        {"op_library.path", [](JsonValue& a) { *path(a, {"op_library", "path"}) = JsonValue::make_string("other.so"); }},
        {"op_library.sha256", [](JsonValue& a) { *path(a, {"op_library", "sha256"}) = JsonValue::make_string("xyz"); }},
        {"op_library.needed",
         [](JsonValue& a) {
             path(a, {"op_library", "needed"})->array.push_back(JsonValue::make_string("libpython3.12.so.1.0"));
         }},
        {"a_l_reproduction.identical",
         [](JsonValue& a) { *path(a, {"a_l_reproduction", "identical"}) = JsonValue::make_bool(false); }},
    };
    for (const auto& [name, mut] : muts) {
        JsonValue att = sh::parse_strict_json(in.attestation);
        mut(att);
        CHECK(refused([&] { plan_of(cfg, in.lineage, &att); }, "attestation." + name));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <resolved config> <lineage> <attestation>\n", argv[0]);
        return 2;
    }
    try {
        const Inputs in{read_file(argv[1]), read_file(argv[2]), read_file(argv[3])};
        check_sha256();
        check_accepts(in);
        check_config_gates(in);
        check_lineage_fields(in);
        check_attestation_fields(in);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
