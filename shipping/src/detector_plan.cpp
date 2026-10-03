// Native detector plan (#465 Phase B PR-8). See saccade_shipping/detector_plan.hpp.
#include "saccade_shipping/detector_plan.hpp"

#include <fstream>
#include <sstream>

#include "saccade_shipping/sha256.hpp"

namespace saccade::shipping {
namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw ConfigError("shipping detector: " + what +
                          " (the U3b-2 detector has no implementation for it)");
    }
}

[[noreturn]] void lineage_error(const std::string& what) {
    throw ConfigError("shipping detector lineage: " + what);
}

// Strict accessors over a parsed JSON document; `path` names the JSON path.
const JsonValue& at(const JsonValue& o, const std::string& key, const std::string& path) {
    if (o.kind != JsonValue::Kind::Object) lineage_error(path + " is not an object");
    const JsonValue* v = o.find(key);
    if (v == nullptr) lineage_error(path + "." + key + " is missing");
    return *v;
}

const std::string& str_at(const JsonValue& o, const std::string& key, const std::string& path) {
    const JsonValue& v = at(o, key, path);
    if (v.kind != JsonValue::Kind::String) lineage_error(path + "." + key + " is not a string");
    return v.string;
}

bool bool_at(const JsonValue& o, const std::string& key, const std::string& path) {
    const JsonValue& v = at(o, key, path);
    if (v.kind != JsonValue::Kind::Bool) lineage_error(path + "." + key + " is not a bool");
    return v.boolean;
}

std::int64_t int_at(const JsonValue& o, const std::string& key, const std::string& path) {
    const JsonValue& v = at(o, key, path);
    if (v.kind != JsonValue::Kind::Int) lineage_error(path + "." + key + " is not an int");
    return v.integer;
}

const std::vector<JsonValue>& array_at(const JsonValue& o, const std::string& key,
                                       const std::string& path) {
    const JsonValue& v = at(o, key, path);
    if (v.kind != JsonValue::Kind::Array) lineage_error(path + "." + key + " is not an array");
    return v.array;
}

std::vector<std::int64_t> ints_at(const JsonValue& o, const std::string& key, const std::string& path) {
    std::vector<std::int64_t> out;
    for (const JsonValue& e : array_at(o, key, path)) {
        if (e.kind != JsonValue::Kind::Int) lineage_error(path + "." + key + " holds a non-int");
        out.push_back(e.integer);
    }
    return out;
}

bool sha256_hex_ok(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

const std::string& sha_at(const JsonValue& o, const std::string& key, const std::string& path) {
    const std::string& s = str_at(o, key, path);
    if (!sha256_hex_ok(s)) lineage_error(path + "." + key + " is not a sha256 hex digest");
    return s;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError("cannot open " + path);
    std::ostringstream text;
    text << in.rdbuf();
    if (!in.good() && !in.eof()) throw ConfigError("cannot read " + path);
    return text.str();
}

}  // namespace

DetectorPlan plan_detector(const ResolvedShippingConfig& cfg, const JsonValue& lineage,
                           const std::string& lineage_sha256,
                           const JsonValue* realization_attestation) {
    const auto& hp = cfg.host_params;
    const auto& d = hp.detector;
    const auto& b = d.build;
    const auto& s = hp.steps;
    const auto& c = hp.cfg;

    // ── resolved config: the oracle's detector path, gate by gate ──────────
    // MambaGatedDetector: whole-graph TRT path (the only one with the fixed S2).
    require(b.use_whole_graph, "host_params.detector.build.use_whole_graph must be true");
    // The head slot is the accepted U1 form (A_L); a configured TRT head engine
    // would put a different head there.
    require(b.trt_head_engine.empty(), "host_params.detector.build.trt_head_engine must be empty");
    // _postprocess_mamba_fixed_eager: small_p3_max_threshold > 0 adds
    // _fuse_small_p3_scores.
    require(b.small_p3_max_threshold == 0.0,
            "host_params.detector.build.small_p3_max_threshold must be 0.0");
    // The native S2 is the twin of the torch.compile'd S2 (Inductor's sigmoid);
    // eager S2 computes the sigmoid differently.
    require(b.postprocess_compile, "host_params.detector.build.postprocess_compile must be true");
    // PR-2L accepted A_L for the headline configuration, whose head calls are
    // these; any other head setting is a configuration the acceptance did not cover.
    require(d.head_calls.set_head_compile == BoolList{true} &&
                d.head_calls.set_block_compile == BoolList{true},
            "host_params.detector.head_calls must be the headline's (set_head_compile [true], "
            "set_block_compile [true]) that the PR-2L acceptance covers");
    // evaluator.py: tiling "native_640" selects detect_native_640, whose
    // whole-graph branch (no letterbox, no NV12) returns the S2 rows as they are.
    require(d.detect_fn == "detect_native_640",
            "host_params.detector.detect_fn must be detect_native_640");
    require(c.tiling == "native_640" && c.kwargs_tiling == "native_640",
            "cfg.tiling / kwargs.tiling must be native_640");
    require(!c.tta, "cfg.tta must be false");
    require(c.preprocess_modes.empty(), "cfg.preprocess_modes must be empty (no letterbox)");
    require(!s.ingest_nv12_buffer, "steps.ingest.nv12_buffer must be false (preprocessed path)");
    require(!s.track_workbench, "steps.track.workbench must be false (its own detector call)");
    require(d.contract.feature_dim == 0 && !d.contract.fpn_reid_mode,
            "host_params.detector.contract must have feature_dim 0 and fpn_reid_mode false");
    require(d.contract.box_format == "xyxy" && c.detector_box_format == "xyxy",
            "the detector box format must be xyxy");
    require(b.img_size > 0 && b.img_size % 32 == 0,
            "host_params.detector.build.img_size must be a positive multiple of 32");
    require(b.max_det > 0, "host_params.detector.build.max_det must be positive");

    DetectorPlan plan;
    plan.img_size = static_cast<int>(b.img_size);
    plan.max_det = static_cast<int>(b.max_det);
    plan.conf_thr_unused = b.conf_thr;

    // ── frozen PR-1L lineage (B5): it must describe this config's model ────
    const std::string L = "lineage";
    if (str_at(lineage, "schema", L) != kHeadLineageSchema) {
        lineage_error("schema is not " + std::string(kHeadLineageSchema));
    }
    if (bool_at(at(lineage, "tool", L), "git_dirty", L + ".tool")) {
        lineage_error("tool.git_dirty is true");
    }
    const JsonValue& preset = at(lineage, "preset", L);
    if (str_at(preset, "path", L + ".preset") != cfg.source.preset ||
        str_at(preset, "sha256", L + ".preset") != cfg.source.preset_sha256) {
        lineage_error("preset path/sha256 differ from the resolved config's source");
    }
    if (str_at(preset, "mamba_ckpt", L + ".preset") != b.mamba_ckpt ||
        str_at(at(at(lineage, "source", L), "mamba_ckpt", L + ".source"), "path",
               L + ".source.mamba_ckpt") != b.mamba_ckpt) {
        lineage_error("mamba_ckpt differs from host_params.detector.build.mamba_ckpt");
    }
    if (str_at(preset, "fpn_backbone_engine", L + ".preset") != b.trt_backbone_engine) {
        lineage_error("preset.fpn_backbone_engine differs from build.trt_backbone_engine");
    }
    const JsonValue& inventory = at(lineage, "inventory", L);
    if (!bool_at(inventory, "ckpt_sha256_match", L + ".inventory") ||
        !bool_at(inventory, "backbone_engine_sha256_match", L + ".inventory")) {
        lineage_error("the inventory did not match the ckpt / backbone engine");
    }
    const JsonValue& builder = at(at(lineage, "source", L), "builder_inputs", L + ".source");
    if (str_at(at(builder, "yolo_weights", L), "path", L) != b.yolo_pt_path ||
        str_at(at(builder, "teacher_ckpt", L), "path", L) != b.teacher_ckpt) {
        lineage_error("builder inputs differ from build.yolo_pt_path / build.teacher_ckpt");
    }
    const JsonValue& args = at(at(lineage, "source", L), "mamba_args", L + ".source");
    if (bool_at(args, "use_detail_fusion", L + ".mamba_args")) {
        lineage_error("mamba_args.use_detail_fusion is true");
    }
    if (const JsonValue* rm = args.find("reg_max")) {
        // MambaGatedDetector: mamba_args.get("reg_max", 1).
        if (rm->kind != JsonValue::Kind::Int || rm->integer != 1) {
            lineage_error("mamba_args.reg_max is not 1 (DFL decode not implemented)");
        }
    }
    if (const JsonValue* sd = args.find("use_strip_detail")) {
        if (sd->kind != JsonValue::Kind::Bool || sd->boolean) {
            lineage_error("mamba_args.use_strip_detail is not false");
        }
    }
    const std::int64_t nc = int_at(args, "num_classes", L + ".mamba_args");
    if (nc <= 0) lineage_error("mamba_args.num_classes is not positive");
    plan.num_classes = static_cast<int>(nc);

    const JsonValue& load = at(lineage, "head_load", L);
    if (!array_at(load, "missing_keys", L).empty() || !array_at(load, "unexpected_keys", L).empty()) {
        lineage_error("head_load reports missing or unexpected keys");
    }
    if (bool_at(load, "use_detail_fusion", L + ".head_load")) {
        lineage_error("head_load.use_detail_fusion is true");
    }
    const auto ch = ints_at(load, "in_channels", L + ".head_load");
    if (ch.size() != 3) lineage_error("head_load.in_channels does not have 3 entries");

    const JsonValue& ts = at(lineage, "torchscript", L);
    const std::string T = L + ".torchscript";
    if (str_at(ts, "dtype", T) != "float32" || str_at(ts, "batch", T) != "static 1") {
        lineage_error("torchscript is not float32 / static batch 1");
    }
    const JsonValue& inputs = at(ts, "inputs", T);
    const char* in_names[3] = {"p3", "p4", "p5"};
    int anchors = 0;
    for (int i = 0; i < 3; ++i) {
        const int side = plan.img_size / kDetectorStrides[static_cast<std::size_t>(i)];
        const std::vector<std::int64_t> want = {1, ch[static_cast<std::size_t>(i)], side, side};
        if (ints_at(inputs, in_names[i], T + ".inputs") != want) {
            lineage_error(std::string("torchscript.inputs.") + in_names[i] +
                          " is not [1, in_channels, img_size/stride, img_size/stride]");
        }
        plan.in_channels[static_cast<std::size_t>(i)] = static_cast<int>(ch[static_cast<std::size_t>(i)]);
        plan.feature_shapes[static_cast<std::size_t>(i)] = {1, static_cast<int>(ch[static_cast<std::size_t>(i)]),
                                                            side, side};
        anchors += side * side;
    }
    plan.anchors = anchors;
    const std::vector<std::string> want_out = {"cls_p3", "cls_p4", "cls_p5", "reg_p3", "reg_p4", "reg_p5"};
    std::vector<std::string> outs;
    for (const JsonValue& e : array_at(ts, "outputs", T)) {
        if (e.kind != JsonValue::Kind::String) lineage_error("torchscript.outputs holds a non-string");
        outs.push_back(e.string);
    }
    if (outs != want_out) lineage_error("torchscript.outputs is not cls_p3..p5, reg_p3..p5");
    const std::int64_t calls = int_at(ts, "native_scan_calls", T);
    if (calls <= 0) lineage_error("torchscript.native_scan_calls is not positive");
    plan.native_scan_calls = static_cast<int>(calls);
    plan.head_artifact = {str_at(ts, "path", T), sha_at(ts, "sha256", T)};
    const std::string& content_sha = sha_at(ts, "content_sha256", T);

    const JsonValue& sc = at(lineage, "structural_check", L);
    if (!bool_at(sc, "bitwise_equal_all", L + ".structural_check")) {
        lineage_error("structural_check.bitwise_equal_all is false");
    }

    const JsonValue& op = at(lineage, "op_library", L);
    if (str_at(op, "op", L + ".op_library") != kNativeScanOp) {
        lineage_error("op_library.op is not " + std::string(kNativeScanOp));
    }
    for (const JsonValue& e : array_at(op, "needed", L + ".op_library")) {
        if (e.kind != JsonValue::Kind::String) lineage_error("op_library.needed holds a non-string");
        if (e.string.rfind("libpython", 0) == 0 || e.string.rfind("libtorch_python", 0) == 0) {
            lineage_error("op_library links " + e.string);
        }
    }
    plan.op_library = {str_at(op, "path", L + ".op_library"), sha_at(op, "sha256", L + ".op_library")};
    plan.op_library_lineage_sha256 = plan.op_library.sha256;

    const JsonValue& rt = at(lineage, "runtime_requirements", L);
    const std::string R = L + ".runtime_requirements";
    plan.runtime.graph_executor_optimize = bool_at(rt, "graph_executor_optimize", R);
    plan.runtime.cudnn_benchmark = bool_at(rt, "cudnn_benchmark", R);
    plan.runtime.cudnn_allow_tf32 = bool_at(rt, "cudnn_allow_tf32", R);
    plan.runtime.matmul_allow_tf32 = bool_at(rt, "matmul_allow_tf32", R);
    // PR-1L §1: the artifact's numerics are eager aten kernels only with the
    // graph executor's optimization off.
    if (plan.runtime.graph_executor_optimize) {
        lineage_error("runtime_requirements.graph_executor_optimize is true");
    }

    const JsonValue& backbone = at(at(lineage, "companions", L), "backbone_engine", L + ".companions");
    if (str_at(backbone, "path", L + ".companions.backbone_engine") != b.trt_backbone_engine) {
        lineage_error("companions.backbone_engine.path differs from build.trt_backbone_engine");
    }
    plan.backbone_engine = {b.trt_backbone_engine,
                            sha_at(backbone, "sha256", L + ".companions.backbone_engine")};

    // ── realization attestation: which build realizes the lineage's operator ─
    if (realization_attestation != nullptr) {
        const JsonValue& a = *realization_attestation;
        const std::string A = "realization attestation";
        if (str_at(a, "schema", A) != kHeadRealizationSchema) {
            lineage_error(A + ": schema is not " + std::string(kHeadRealizationSchema));
        }
        const JsonValue& fl = at(a, "frozen_lineage", A);
        if (sha_at(fl, "sha256", A + ".frozen_lineage") != lineage_sha256) {
            lineage_error(A + " is bound to a different lineage file (sha256)");
        }
        if (sha_at(fl, "torchscript_sha256", A + ".frozen_lineage") != plan.head_artifact.sha256 ||
            sha_at(fl, "torchscript_content_sha256", A + ".frozen_lineage") != content_sha ||
            sha_at(fl, "op_library_sha256", A + ".frozen_lineage") != plan.op_library_lineage_sha256) {
            lineage_error(A + ".frozen_lineage disagrees with the lineage");
        }
        const JsonValue& ol = at(a, "op_library", A);
        if (str_at(ol, "path", A + ".op_library") != plan.op_library.path) {
            lineage_error(A + ".op_library.path differs from the lineage's");
        }
        for (const JsonValue& e : array_at(ol, "needed", A + ".op_library")) {
            if (e.kind != JsonValue::Kind::String) lineage_error(A + ".op_library.needed holds a non-string");
            if (e.string.rfind("libpython", 0) == 0 || e.string.rfind("libtorch_python", 0) == 0) {
                lineage_error(A + ": the realized op library links " + e.string);
            }
        }
        const JsonValue& repro = at(a, "a_l_reproduction", A);
        if (!bool_at(repro, "identical", A + ".a_l_reproduction")) {
            lineage_error(A + ": the A_L reproduction is not byte-identical");
        }
        plan.op_library.sha256 = sha_at(ol, "sha256", A + ".op_library");
        plan.op_library_from_attestation = true;
    }
    return plan;
}

DetectorPlan plan_detector_files(const ResolvedShippingConfig& cfg, const DetectorInputs& in) {
    const std::string lineage_text = read_file(in.lineage_path);
    const JsonValue lineage = parse_strict_json(lineage_text);
    const std::string lineage_sha = sha256_hex(lineage_text.data(), lineage_text.size());
    if (in.attestation_path.empty()) return plan_detector(cfg, lineage, lineage_sha, nullptr);
    const JsonValue att = parse_strict_json(read_file(in.attestation_path));
    return plan_detector(cfg, lineage, lineage_sha, &att);
}

}  // namespace saccade::shipping
