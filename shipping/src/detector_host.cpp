// Native detector host (#465 Phase B PR-8). See saccade_shipping/detector_host.hpp.
#include "saccade_shipping/detector_host.hpp"

#include <NvInfer.h>
#include <dlfcn.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/jit/passes/inliner.h>
// Declares torch::jit::{get,set}GraphExecutorOptimize, which libtorch_cpu
// defines (the header's directory name notwithstanding; no libtorch_python).
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/script.h>
#include <torch/version.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <sstream>
#include <stdexcept>

#include "perception/trt_engine.hpp"
#include "saccade_shipping/detector_s2.hpp"
#include "saccade_shipping/sha256.hpp"

namespace saccade::shipping {
namespace {

[[noreturn]] void load_error(const std::string& what) {
    throw ConfigError("shipping detector load: " + what);
}

[[noreturn]] void run_error(const std::string& what) {
    throw std::runtime_error("shipping detector: " + what);
}

void check_sha(const std::string& what, const std::string& path, const std::string& want,
               std::string& got) {
    got = sha256_file_hex(path);
    if (got != want) load_error(what + " " + path + " sha256 " + got + " != " + want);
}

HeadRuntimeRequirements runtime_readback() {
    auto& ctx = at::globalContext();
    HeadRuntimeRequirements r;
    r.graph_executor_optimize = torch::jit::getGraphExecutorOptimize();
    r.cudnn_benchmark = ctx.benchmarkCuDNN();
    r.cudnn_allow_tf32 = ctx.allowTF32CuDNN();
    r.matmul_allow_tf32 = ctx.allowTF32CuBLAS();
    return r;
}

bool same(const HeadRuntimeRequirements& a, const HeadRuntimeRequirements& b) {
    return a.graph_executor_optimize == b.graph_executor_optimize &&
           a.cudnn_benchmark == b.cudnn_benchmark && a.cudnn_allow_tf32 == b.cudnn_allow_tf32 &&
           a.matmul_allow_tf32 == b.matmul_allow_tf32;
}

template <class F>
void walk(torch::jit::Block* block, const F& f) {
    for (torch::jit::Node* n : block->nodes()) {
        f(n);
        for (torch::jit::Block* sub : n->blocks()) walk(sub, f);
    }
}

std::string dims_text(const nvinfer1::Dims& d) {
    std::ostringstream o;
    o << "[";
    for (int i = 0; i < d.nbDims; ++i) o << (i ? "," : "") << d.d[i];
    o << "]";
    return o.str();
}

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
void next_ulp_inplace(at::Tensor t) {
    auto first = t.view({-1}).narrow(0, 0, 1);
    first.copy_(at::nextafter(first, at::full_like(first, INFINITY)));
}
#endif

}  // namespace

void s2_run(const S2Level levels[3], int num_classes, int k, float sx, float sy, float* raw,
            float* scaled, cudaStream_t stream) {
    if (k <= 0) throw std::invalid_argument("shipping detector S2: k <= 0");
    c10::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal(stream, 0));
    int anchors = 0;
    for (int i = 0; i < 3; ++i) anchors += levels[i].side * levels[i].side;
    const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    const at::Tensor scores_max = at::empty({anchors}, f32);
    const at::Tensor class_idx = at::empty({anchors}, f32.dtype(at::kLong));
    s2_score_max(levels, num_classes, scores_max.data_ptr<float>(), class_idx.data_ptr<std::int64_t>(),
                 stream);
    // _postprocess_mamba_fixed: scores_max[b].topk(max_det) -> aten.topk(x, k).
    const auto topk = at::topk(scores_max, k);
    const at::Tensor top_score = std::get<0>(topk).contiguous();
    const at::Tensor top_idx = std::get<1>(topk).contiguous();
    s2_decode_gather(levels, top_idx.data_ptr<std::int64_t>(), top_score.data_ptr<float>(),
                     class_idx.data_ptr<std::int64_t>(), k, sx, sy, raw, scaled, stream);
}

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
const char* detector_mutation_name(DetectorMutation m) {
    switch (m) {
        case DetectorMutation::None: return "none";
        case DetectorMutation::BackboneUlp: return "backbone_ulp";
        case DetectorMutation::HeadUlp: return "head_ulp";
        case DetectorMutation::S2Threshold: return "s2_threshold";
        case DetectorMutation::S2TopK: return "s2_topk";
        case DetectorMutation::S2Order: return "s2_order";
        case DetectorMutation::BoxUlp: return "box_ulp";
    }
    return "?";
}

DetectorMutation parse_detector_mutation(const std::string& name) {
    for (DetectorMutation m :
         {DetectorMutation::None, DetectorMutation::BackboneUlp, DetectorMutation::HeadUlp,
          DetectorMutation::S2Threshold, DetectorMutation::S2TopK, DetectorMutation::S2Order,
          DetectorMutation::BoxUlp}) {
        if (name == detector_mutation_name(m)) return m;
    }
    throw std::invalid_argument("unknown detector mutation " + name);
}
#endif

std::vector<std::string> mapped_python_libraries() {
    std::ifstream maps("/proc/self/maps");
    std::set<std::string> found;
    std::string line;
    while (std::getline(maps, line)) {
        const auto slash = line.find('/');
        if (slash == std::string::npos) continue;
        const std::string path = line.substr(slash);
        const std::string name = std::filesystem::path(path).filename().string();
        if (name.rfind("libpython", 0) == 0 || name.rfind("libtorch_python", 0) == 0) {
            found.insert(path);
        }
    }
    return {found.begin(), found.end()};
}

struct DetectorHost::Impl {
    DetectorPlan plan;
    HeadLoadReport report;
    cudaStream_t raw_stream;
    c10::cuda::CUDAStream stream;
    std::unique_ptr<saccade::TRTEngine> engine;
    std::string engine_input;
    std::vector<std::string> engine_outputs;
    torch::jit::Module head;
    void* op_handle = nullptr;
    bool scales_set = false;
    float sx = 0.0f, sy = 0.0f;
    at::Tensor resized, feats[3], head_out[6], s2_raw, s2_scaled;
    DetectorStages stages;
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    DetectorMutation mutation = DetectorMutation::None;
    bool stale_graph_input = false;
#endif

    // Whole-detect graphs, keyed (frame h, frame w, image h, image w).
    struct Captured {
        at::Tensor static_in;  // [1, 3, h, w]
        std::unique_ptr<at::cuda::CUDAGraph> graph;
    };
    using GraphKey = std::tuple<int, int, int, int>;
    int img_h = 0, img_w = 0;
    bool warm = false;
    std::map<GraphKey, Captured> graphs;
    WholeGraphStats graph_stats;

    Impl(const DetectorPlan& p, cudaStream_t s)
        : plan(p), raw_stream(s), stream(at::cuda::getStreamFromExternal(s, 0)) {}
};

DetectorHost::DetectorHost(const DetectorPlan& plan, const std::string& model_root,
                           cudaStream_t stream)
    : impl_(std::make_unique<Impl>(plan, stream)) {
    Impl& m = *impl_;
    const std::string op_path = resolve_model_path(model_root, plan.op_library.path);
    const std::string head_path = resolve_model_path(model_root, plan.head_artifact.path);
    const std::string engine_path = resolve_model_path(model_root, plan.backbone_engine.path);

    // 1. hashes before anything is loaded.
    check_sha("operator library", op_path, plan.op_library.sha256, m.report.op_library_sha256);
    check_sha("head artifact", head_path, plan.head_artifact.sha256, m.report.head_artifact_sha256);
    check_sha("backbone engine", engine_path, plan.backbone_engine.sha256,
              m.report.backbone_engine_sha256);

    // 2. the native scan operator.
    m.op_handle = dlopen(op_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (m.op_handle == nullptr) load_error(std::string("dlopen ") + op_path + ": " + dlerror());

    // 3. runtime requirements, then the artifact (no device map).
    torch::jit::setGraphExecutorOptimize(plan.runtime.graph_executor_optimize);
    auto& ctx = at::globalContext();
    ctx.setBenchmarkCuDNN(plan.runtime.cudnn_benchmark);
    ctx.setAllowTF32CuDNN(plan.runtime.cudnn_allow_tf32);
    ctx.setAllowTF32CuBLAS(plan.runtime.matmul_allow_tf32);
    m.report.runtime_readback = runtime_readback();
    if (!same(m.report.runtime_readback, plan.runtime)) {
        load_error("runtime requirements did not read back as set");
    }
    m.head = torch::jit::load(head_path);
    m.head.eval();
    std::set<std::string> param_devices;
    for (const at::Tensor& t : m.head.parameters(/*recurse=*/true)) param_devices.insert(t.device().str());
    for (const at::Tensor& t : m.head.buffers(/*recurse=*/true)) param_devices.insert(t.device().str());
    m.report.param_devices.assign(param_devices.begin(), param_devices.end());
    if (m.report.param_devices != std::vector<std::string>{"cuda:0"}) {
        load_error("head parameters/buffers are not all on cuda:0");
    }

    // 4. the inlined forward graph.
    std::shared_ptr<torch::jit::Graph> graph = m.head.get_method("forward").graph()->copy();
    torch::jit::Inline(*graph);
    int native_calls = 0;
    walk(graph->block(), [&](torch::jit::Node* n) {
        const std::string kind = n->kind().toQualString();
        if (kind == "prim::PythonOp") load_error("head graph contains prim::PythonOp");
        if (kind == kPythonScanOp) load_error(std::string("head graph calls ") + kPythonScanOp);
        if (kind == kNativeScanOp) ++native_calls;
        if (n->kind() == c10::prim::Constant && n->outputs().size() == 1 &&
            n->output()->type()->kind() == c10::TypeKind::TensorType) {
            const auto v = torch::jit::toIValue(n->output());
            if (!v || !v->isTensor()) load_error("unreadable tensor constant in the head graph");
            m.report.constant_devices.push_back(v->toTensor().device().str());
        }
    });
    m.report.native_scan_calls = native_calls;
    if (native_calls != plan.native_scan_calls) {
        load_error("head graph calls " + std::string(kNativeScanOp) + " " + std::to_string(native_calls) +
                   " times, lineage says " + std::to_string(plan.native_scan_calls));
    }
    for (const std::string& d : m.report.constant_devices) {
        if (d != "cpu") load_error("a head tensor constant is on " + d + ", required cpu");
    }

    // 5. the backbone engine (the existing TRTEngine) and its I/O.
    m.engine = std::make_unique<saccade::TRTEngine>(engine_path);
    m.report.trt_version = getInferLibVersion();
    const int n_io = m.engine->get_nb_tensors();
    if (n_io != 4) load_error("backbone engine has " + std::to_string(n_io) + " I/O tensors, expected 4");
    for (int i = 0; i < n_io; ++i) {
        const char* name = m.engine->get_tensor_name(i);
        m.report.engine_io.push_back(std::string(name) + ":" + dims_text(m.engine->getTensorDims(name)));
        // TRTYoloBackbone: tensor 0 is the input, the rest are p3, p4, p5 in order.
        if ((i == 0) != m.engine->is_input(name)) load_error("backbone engine I/O order");
        if (i == 0) m.engine_input = name;
        else m.engine_outputs.push_back(name);
    }
    if (!m.engine->set_input_shape(m.engine_input.c_str(), {1, 3, plan.img_size, plan.img_size})) {
        load_error("backbone engine rejects input [1, 3, img, img]");
    }
    const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    for (int i = 0; i < 3; ++i) {
        const auto& want = plan.feature_shapes[static_cast<std::size_t>(i)];
        const nvinfer1::Dims d = m.engine->getTensorDims(m.engine_outputs[static_cast<std::size_t>(i)].c_str());
        bool ok = d.nbDims == 4;
        for (int k = 0; ok && k < 4; ++k) {
            const auto dk = d.d[k] == -1 && k == 0 ? 1 : d.d[k];
            ok = dk == want[static_cast<std::size_t>(k)];
        }
        if (!ok) load_error("backbone output " + m.engine_outputs[static_cast<std::size_t>(i)] + " " +
                            dims_text(d) + " does not match the head's input");
        m.feats[i] = at::empty({want[0], want[1], want[2], want[3]}, f32);
    }
    m.report.torch_version = TORCH_VERSION;

    // 6. no Python in the process.
    const auto py = mapped_python_libraries();
    if (!py.empty()) load_error("Python library mapped into the detector process: " + py.front());

    m.s2_raw = at::empty({plan.max_det, 6}, f32);
    m.s2_scaled = at::empty({plan.max_det, 6}, f32);
}

DetectorHost::~DetectorHost() = default;  // the operator library stays loaded

const HeadLoadReport& DetectorHost::load_report() const { return impl_->report; }

const DetectorStages& DetectorHost::stages() const { return impl_->stages; }

#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
void DetectorHost::set_mutation_for_measurement(DetectorMutation m) { impl_->mutation = m; }

void DetectorHost::set_stale_graph_input_for_measurement(bool on) { impl_->stale_graph_input = on; }
#endif

void DetectorHost::set_image_dims(int height, int width) {
    if (height <= 0 || width <= 0) run_error("image dims must be positive");
    Impl& m = *impl_;
    m.scales_set = true;
    m.sx = coordinate_scale(width, m.plan.img_size);
    m.sy = coordinate_scale(height, m.plan.img_size);
    // set_whole_graph_img_dims: the same dims keep the graphs and the warm flag.
    if (height == m.img_h && width == m.img_w) return;
    m.img_h = height;
    m.img_w = width;
    if (!m.graphs.empty()) {
        // A replay may still be queued on the stream.
        if (cudaStreamSynchronize(m.raw_stream) != cudaSuccess) run_error("stream synchronize failed");
        m.graphs.clear();
        ++m.graph_stats.cache_clears;
    }
    m.warm = false;
}

namespace {

// _whole_graph_fn on the current (guarded) stream: resize -> engine -> head ->
// S2 into m.s2_raw / m.s2_scaled (k rows). The eager mutations of the engine
// and head outputs (measurement variant) apply only when `mutate` is set.
template <class Impl>
void whole_forward(Impl& m, const at::Tensor& frame, int k, [[maybe_unused]] bool mutate) {
    const DetectorPlan& p = m.plan;
    // resize: F.interpolate(frame[None], (img, img), mode="bilinear", align_corners=False).
    m.resized = at::upsample_bilinear2d(frame, at::IntArrayRef{p.img_size, p.img_size}, false,
                                        std::nullopt);
    if (!m.resized.is_contiguous()) run_error("resized frame is not contiguous");

    // backbone: TRTYoloBackbone.infer_graph on the current stream.
    if (!m.engine->set_input_shape(m.engine_input.c_str(), {1, 3, p.img_size, p.img_size}) ||
        !m.engine->set_tensor_address(m.engine_input.c_str(), m.resized.data_ptr())) {
        run_error("backbone input binding failed");
    }
    for (int i = 0; i < 3; ++i) {
        if (!m.engine->set_tensor_address(m.engine_outputs[static_cast<std::size_t>(i)].c_str(),
                                          m.feats[i].data_ptr())) {
            run_error("backbone output binding failed");
        }
    }
    if (!m.engine->enqueue_v3(m.raw_stream)) run_error("backbone enqueue failed");
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    if (mutate && m.mutation == DetectorMutation::BackboneUlp) next_ulp_inplace(m.feats[0]);
#endif

    // head: the PR-1L artifact.
    const auto out = m.head.forward({m.feats[0], m.feats[1], m.feats[2]});
    if (!out.isTuple() || out.toTupleRef().elements().size() != 6) run_error("head did not return 6 outputs");
    const auto& elems = out.toTupleRef().elements();
    for (int i = 0; i < 6; ++i) {
        const at::Tensor t = elems[static_cast<std::size_t>(i)].toTensor();
        const auto& fs = p.feature_shapes[static_cast<std::size_t>(i % 3)];
        const std::vector<int64_t> want = {1, i < 3 ? p.num_classes : p.reg_channels, fs[2], fs[3]};
        if (t.sizes().vec() != want || t.scalar_type() != at::kFloat || !t.is_cuda() || !t.is_contiguous()) {
            run_error("head output " + std::to_string(i) + " is not a contiguous CUDA float32 tensor of the "
                      "planned shape");
        }
        m.head_out[i] = t;
    }
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    if (mutate && m.mutation == DetectorMutation::HeadUlp) next_ulp_inplace(m.head_out[0]);
#endif

    // S2.
    S2Level levels[3];
    for (int i = 0; i < 3; ++i) {
        levels[i] = {m.head_out[i].template data_ptr<float>(), m.head_out[i + 3].template data_ptr<float>(),
                     p.feature_shapes[static_cast<std::size_t>(i)][2],
                     static_cast<float>(kDetectorStrides[static_cast<std::size_t>(i)])};
    }
    s2_run(levels, p.num_classes, k, m.sx, m.sy, m.s2_raw.template data_ptr<float>(),
           m.s2_scaled.template data_ptr<float>(), m.raw_stream);
}

}  // namespace

DetectionRows DetectorHost::detect(const float* frame_chw, int height, int width) {
    Impl& m = *impl_;
    const DetectorPlan& p = m.plan;
    if (!m.scales_set) run_error("set_image_dims was not called");
    if (height <= 0 || width <= 0) run_error("frame dims must be positive");
    if (!same(runtime_readback(), p.runtime)) run_error("runtime requirements changed after load");
    c10::cuda::CUDAStreamGuard guard(m.stream);
    torch::NoGradGuard no_grad;
    const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    const at::Tensor frame =
        at::from_blob(const_cast<float*>(frame_chw), {1, 3, height, width}, f32);
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    const int k = m.mutation == DetectorMutation::S2TopK ? p.max_det - 1 : p.max_det;
#else
    const int k = p.max_det;
#endif
    whole_forward(m, frame, k, /*mutate=*/true);

    std::vector<float> raw(static_cast<std::size_t>(k) * 6), scaled(raw.size());
    auto d2h = [&](std::vector<float>& dst, const at::Tensor& src) {
        if (cudaMemcpyAsync(dst.data(), src.data_ptr<float>(), dst.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, m.raw_stream) != cudaSuccess) {
            run_error("device -> host copy failed");
        }
    };
    d2h(raw, m.s2_raw);
    d2h(scaled, m.s2_scaled);
    if (cudaStreamSynchronize(m.raw_stream) != cudaSuccess) run_error("stream synchronize failed");

    int rows = k;
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    if (m.mutation != DetectorMutation::None && m.mutation != DetectorMutation::BackboneUlp &&
        m.mutation != DetectorMutation::HeadUlp && m.mutation != DetectorMutation::S2TopK) {
        if (m.mutation == DetectorMutation::S2Order && k >= 2) {
            for (int c = 0; c < 6; ++c) {
                std::swap(raw[static_cast<std::size_t>(c)], raw[6 + static_cast<std::size_t>(c)]);
                std::swap(scaled[static_cast<std::size_t>(c)], scaled[6 + static_cast<std::size_t>(c)]);
            }
        } else if (m.mutation == DetectorMutation::BoxUlp && k >= 1) {
            scaled[0] = std::nextafter(scaled[0], INFINITY);
        } else if (m.mutation == DetectorMutation::S2Threshold) {
            std::vector<float> r2, s2;
            for (int j = 0; j < k; ++j) {
                if (scaled[static_cast<std::size_t>(j) * 6 + 4] < 0.05f) continue;
                r2.insert(r2.end(), raw.begin() + j * 6, raw.begin() + j * 6 + 6);
                s2.insert(s2.end(), scaled.begin() + j * 6, scaled.begin() + j * 6 + 6);
            }
            raw.swap(r2);
            scaled.swap(s2);
            rows = static_cast<int>(scaled.size() / 6);
        }
        auto h2d = [&](at::Tensor& dst, const std::vector<float>& src) {
            if (!src.empty() && cudaMemcpy(dst.data_ptr<float>(), src.data(), src.size() * sizeof(float),
                                           cudaMemcpyHostToDevice) != cudaSuccess) {
                run_error("host -> device copy failed");
            }
        };
        h2d(m.s2_raw, raw);
        h2d(m.s2_scaled, scaled);
    }
#endif

    m.stages.resized = m.resized.data_ptr<float>();
    for (int i = 0; i < 3; ++i) m.stages.features[i] = m.feats[i].data_ptr<float>();
    for (int i = 0; i < 6; ++i) m.stages.head[i] = m.head_out[i].data_ptr<float>();
    m.stages.s2_raw = m.s2_raw.data_ptr<float>();
    m.stages.s2_scaled = m.s2_scaled.data_ptr<float>();
    m.stages.rows = rows;

    // detect_single_patch_640 -> _run_native_tensor_prep: boxes, scores,
    // classes (float class index -> int32).
    DetectionRows out_rows;
    for (int j = 0; j < rows; ++j) {
        const float* r = scaled.data() + static_cast<std::size_t>(j) * 6;
        out_rows.boxes.insert(out_rows.boxes.end(), r, r + 4);
        out_rows.scores.push_back(r[4]);
        out_rows.classes.push_back(static_cast<std::int32_t>(r[5]));
    }
    return out_rows;
}

const WholeGraphStats& DetectorHost::graph_stats() const { return impl_->graph_stats; }

int DetectorHost::detect_graphed(const float* frame_chw, int height, int width,
                                 const DeviceRowsOut& out) {
    Impl& m = *impl_;
    const DetectorPlan& p = m.plan;
    if (!m.scales_set) run_error("set_image_dims was not called");
    if (height <= 0 || width <= 0) run_error("frame dims must be positive");
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    if (m.mutation != DetectorMutation::None) run_error("detector mutations are eager-only");
#endif
    if (out.capacity < p.max_det) run_error("row buffers smaller than max_det");
    if (!same(runtime_readback(), p.runtime)) run_error("runtime requirements changed after load");
    c10::cuda::CUDAStreamGuard guard(m.stream);
    torch::NoGradGuard no_grad;
    const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    const at::Tensor frame =
        at::from_blob(const_cast<float*>(frame_chw), {1, 3, height, width}, f32);

    const Impl::GraphKey key{height, width, m.img_h, m.img_w};
    auto it = m.graphs.find(key);
    if (it == m.graphs.end()) {
        if (!m.warm) {  // _whole_graph_warmup: one run on a clone, then a device sync
            const at::Tensor warm = frame.clone();
            whole_forward(m, warm, p.max_det, false);
            if (cudaDeviceSynchronize() != cudaSuccess) run_error("warm-up synchronize failed");
            ++m.graph_stats.warmup_runs;
            m.warm = true;
        }
        if (m.graphs.size() >= 10) {
            m.graphs.clear();
            ++m.graph_stats.cache_clears;
        }
        // make_graphed_callables: sample = frame.clone() is the static input;
        // three warm-up iterations on it, then the capture.
        Impl::Captured c;
        c.static_in = frame.clone();
        for (int i = 0; i < 3; ++i) {
            whole_forward(m, c.static_in, p.max_det, false);
            ++m.graph_stats.warmup_runs;
        }
        if (cudaStreamSynchronize(m.raw_stream) != cudaSuccess) run_error("warm-up synchronize failed");
        c.graph = std::make_unique<at::cuda::CUDAGraph>();
        c.graph->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
        whole_forward(m, c.static_in, p.max_det, false);
        c.graph->capture_end();
        // The stage views pointed into the graph's pool; nothing reads them in
        // graph mode.
        m.resized = at::Tensor();
        for (auto& t : m.head_out) t = at::Tensor();
        ++m.graph_stats.captures;
        it = m.graphs.emplace(key, std::move(c)).first;
    }
    Impl::Captured& c = it->second;
#ifdef SACCADE_SHIPPING_MEASUREMENT_HOOKS
    if (!m.stale_graph_input) c.static_in.copy_(frame);
#else
    c.static_in.copy_(frame);
#endif
    c.graph->replay();
    ++m.graph_stats.replays;

    // The rows leave the static output before the next replay can overwrite it.
    const int n = p.max_det;
    const at::Tensor rows = m.s2_scaled.narrow(0, 0, n);
    at::from_blob(out.boxes, {n, 4}, f32).copy_(rows.narrow(1, 0, 4));
    at::from_blob(out.scores, {n}, f32).copy_(rows.select(1, 4));
    at::from_blob(out.classes, {n}, f32.dtype(at::kInt)).copy_(rows.select(1, 5));
    return n;
}

}  // namespace saccade::shipping
