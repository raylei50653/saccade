#include "tracking/mamba_scan_plugin.hpp"
#include "tracking/mamba_scan.cuh"

#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

namespace saccade::trt {

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// Core capability
// ---------------------------------------------------------------------------
class SelectiveScanPlugin::Core final : public IPluginV3OneCore {
public:
    Core(SelectiveScanPluginParams p) : params_(p) {}

    AsciiChar const* getPluginName() const noexcept override {
        return kSelectiveScanPluginName;
    }

    AsciiChar const* getPluginVersion() const noexcept override {
        return kSelectiveScanPluginVersion;
    }

    AsciiChar const* getPluginNamespace() const noexcept override {
        return kSelectiveScanNamespace;
    }

    SelectiveScanPluginParams params_;
};

// ---------------------------------------------------------------------------
// Build capability
// ---------------------------------------------------------------------------
class SelectiveScanPlugin::Build final : public IPluginV3OneBuild {
public:
    Build(SelectiveScanPluginParams p) : params_(p) {}

    int32_t configurePlugin(
        DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        (void)out;
        if (nbInputs < 5 || nbInputs > 7) return -1;
        if (nbOutputs != 1) return -1;

        auto const& u_desc = in[0].desc;
        params_.B = u_desc.dims.d[0];
        params_.L = u_desc.dims.d[1];
        params_.D = u_desc.dims.d[2];

        auto const& a_desc = in[2].desc;
        params_.N = a_desc.dims.d[1];

        params_.has_D = (nbInputs >= 6);

        return 0;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (nbOutputs != 1) return -1;
        if (nbInputs < 1) return -1;
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(
        DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        (void)shapeInputs; (void)nbShapeInputs;
        if (nbInputs < 1 || nbOutputs != 1) return -1;

        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];  // B
        outputs[0].d[1] = inputs[0].d[1];  // L
        outputs[0].d[2] = inputs[0].d[2];  // D
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        int32_t const total = nbInputs + nbOutputs;
        if (pos < 0 || pos >= total) return false;

        auto fmt = inOut[pos].desc.format;
        auto dtype = inOut[pos].desc.type;

        bool const ok_fmt = (fmt == TensorFormat::kLINEAR);
        bool const ok_type = (dtype == DataType::kFLOAT || dtype == DataType::kHALF);

        if (!ok_fmt || !ok_type) return false;

        if (pos == 0) return true;
        return inOut[pos].desc.type == inOut[0].desc.type;
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getWorkspaceSize(
        DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        (void)outputs; (void)nbOutputs;
        int const C_last_dim = (nbInputs > 4 && inputs[4].desc.dims.nbDims > 0)
            ? inputs[4].desc.dims.d[inputs[4].desc.dims.nbDims - 1]
            : params_.N;
        bool const need_broadcast = (C_last_dim < params_.N && C_last_dim == 1);
        if (!need_broadcast) return 0;
        // Reserve space for C broadcast: largest possible (max profile batch × L × N × elem)
        // Use max shape from profile, not desc.dims (desc.dims.d[0] may be -1 for dynamic dims).
        int32_t max_B = inputs[0].max.d[0];
        if (max_B <= 0) max_B = inputs[0].desc.dims.d[0];
        if (max_B <= 0) max_B = 4;
        size_t const L     = static_cast<size_t>(inputs[0].desc.dims.d[1]);
        size_t const N     = static_cast<size_t>(params_.N);
        size_t const elem  = (inputs[0].desc.type == DataType::kHALF) ? 2 : 4;
        return static_cast<size_t>(max_B) * L * N * elem;
    }

    SelectiveScanPluginParams params_;
};

// ---------------------------------------------------------------------------
// Runtime capability
// ---------------------------------------------------------------------------
class SelectiveScanPlugin::Runtime final : public IPluginV3OneRuntime {
public:
    Runtime(SelectiveScanPluginParams p) : params_(p) {}

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs,
        PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        if (nbInputs < 1 || nbOutputs != 1) return -1;
        auto const& dims = in[0].dims;
        params_.B = dims.d[0];
        params_.L = dims.d[1];
        params_.D = dims.d[2];

        auto const& a_dims = in[2].dims;
        params_.N = a_dims.d[a_dims.nbDims - 1];

        params_.a_per_channel = (a_dims.nbDims >= 2 && a_dims.d[0] > 1);
        params_.has_D = (nbInputs >= 6);
        return 0;
    }

    int32_t enqueue(
        PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override
    {

        SelectiveScanParams kernel_params;
        kernel_params.B = params_.B;
        kernel_params.L = params_.L;
        kernel_params.D = params_.D;
        kernel_params.N = params_.N;
        kernel_params.has_D = params_.has_D;
        kernel_params.a_per_channel = params_.a_per_channel;

        bool is_half = (inputDesc[0].type == DataType::kHALF);

        void const* C_ptr = inputs[4];
        int const C_last_dim = inputDesc[4].dims.d[inputDesc[4].dims.nbDims - 1];

        // If C is rank-1 in the last dim (c_rank=1 from legacy checkpoints),
        // broadcast to (B, L, N) so the kernel reads the same value for all N.
        bool const need_C_broadcast = (C_last_dim < kernel_params.N && C_last_dim == 1);
        void* C_alloc = nullptr;

        if (need_C_broadcast) {
            if (workspace == nullptr) return -1;  // workspace should have been allocated
            C_alloc = workspace;

            if (is_half) {
                broadcast_C_half(inputs[4], C_alloc, params_.B, params_.L, params_.N, stream);
            } else {
                broadcast_C_float(
                    static_cast<float const*>(inputs[4]),
                    static_cast<float*>(C_alloc),
                    params_.B, params_.L, params_.N, stream);
            }
            C_ptr = C_alloc;
        }

        if (is_half) {
            selective_scan_fwd_half(
                inputs[0], inputs[1], inputs[2], inputs[3], C_ptr,
                params_.has_D ? inputs[5] : nullptr,
                outputs[0], kernel_params, stream);
        } else {
            selective_scan_fwd(
                static_cast<float const*>(inputs[0]),
                static_cast<float const*>(inputs[1]),
                static_cast<float const*>(inputs[2]),
                static_cast<float const*>(inputs[3]),
                static_cast<float const*>(C_ptr),
                params_.has_D ? static_cast<float const*>(inputs[5]) : nullptr,
                static_cast<float*>(outputs[0]),
                kernel_params, stream);
        }

        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override {
        (void)context;
        return new SelectiveScanPlugin(params_);
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &empty_fields_;
    }

    SelectiveScanPluginParams params_;
    PluginFieldCollection empty_fields_{};
};

// ---------------------------------------------------------------------------
// SelectiveScanPlugin (IPluginV3 wrapper)
// ---------------------------------------------------------------------------
SelectiveScanPlugin::SelectiveScanPlugin(SelectiveScanPluginParams params)
    : params_(params)
{
    core_    = new Core(params_);
    build_   = new Build(params_);
    runtime_ = new Runtime(params_);
}

IPluginCapability*
SelectiveScanPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    switch (type) {
        case PluginCapabilityType::kCORE:    return core_;
        case PluginCapabilityType::kBUILD:   return build_;
        case PluginCapabilityType::kRUNTIME: return runtime_;
    }
    return nullptr;
}

IPluginV3* SelectiveScanPlugin::clone() noexcept {
    return new SelectiveScanPlugin(params_);
}

// ---------------------------------------------------------------------------
// SelectiveScanPluginCreator
// ---------------------------------------------------------------------------
IPluginV3* SelectiveScanPluginCreator::createPlugin(
    AsciiChar const* name, PluginFieldCollection const* fc,
    TensorRTPhase phase) noexcept
{
    (void)phase;
    (void)name;

    SelectiveScanPluginParams params;

    if (fc != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            auto const& f = fc->fields[i];
            if (f.name == nullptr) continue;
            if (std::strcmp(f.name, "has_D") == 0 && f.type == PluginFieldType::kINT32) {
                params.has_D = (*static_cast<int32_t const*>(f.data)) != 0;
            } else if (std::strcmp(f.name, "a_per_channel") == 0 && f.type == PluginFieldType::kINT32) {
                params.a_per_channel = (*static_cast<int32_t const*>(f.data)) != 0;
            } else if (std::strcmp(f.name, "B") == 0 && f.type == PluginFieldType::kINT32) {
                params.B = *static_cast<int32_t const*>(f.data);
            } else if (std::strcmp(f.name, "L") == 0 && f.type == PluginFieldType::kINT32) {
                params.L = *static_cast<int32_t const*>(f.data);
            } else if (std::strcmp(f.name, "D") == 0 && f.type == PluginFieldType::kINT32) {
                params.D = *static_cast<int32_t const*>(f.data);
            } else if (std::strcmp(f.name, "N") == 0 && f.type == PluginFieldType::kINT32) {
                params.N = *static_cast<int32_t const*>(f.data);
            }
        }
    }

    auto* plugin = new SelectiveScanPlugin(params);
    return plugin;
}

PluginFieldCollection const* SelectiveScanPluginCreator::getFieldNames() noexcept {
    static PluginField fields[] = {
        PluginField("has_D", nullptr, PluginFieldType::kINT32, 1),
        PluginField("a_per_channel", nullptr, PluginFieldType::kINT32, 1),
        PluginField("B", nullptr, PluginFieldType::kINT32, 1),
        PluginField("L", nullptr, PluginFieldType::kINT32, 1),
        PluginField("D", nullptr, PluginFieldType::kINT32, 1),
        PluginField("N", nullptr, PluginFieldType::kINT32, 1),
    };
    static int const n = sizeof(fields) / sizeof(fields[0]);
    field_collection_.nbFields = n;
    field_collection_.fields = fields;
    return &field_collection_;
}

AsciiChar const* SelectiveScanPluginCreator::getPluginName() const noexcept {
    return kSelectiveScanPluginName;
}

AsciiChar const* SelectiveScanPluginCreator::getPluginVersion() const noexcept {
    return kSelectiveScanPluginVersion;
}

AsciiChar const* SelectiveScanPluginCreator::getPluginNamespace() const noexcept {
    return kSelectiveScanNamespace;
}

} // namespace saccade::trt

// ---------------------------------------------------------------------------
// Plugin registration — called by TensorRT when loading this shared library
// via IPluginRegistry::loadLibrary().
//
// TensorRT 10.x expects the library to export:
//   getCreators(int32_t& nbCreators) → IPluginCreatorInterface* const*
//   setLoggerFinder(ILoggerFinder* finder)
// ---------------------------------------------------------------------------

static saccade::trt::SelectiveScanPluginCreator gSelectiveScanCreator;

extern "C" TENSORRTAPI bool initLibNvInferPlugins(
    void* logger, char const* libNamespace)
{
    (void)logger;
    nvinfer1::IPluginRegistry* registry = getPluginRegistry();
    if (registry != nullptr) {
        registry->registerCreator(gSelectiveScanCreator, libNamespace ? libNamespace : "");
    }
    return true;
}

extern "C" TENSORRTAPI nvinfer1::IPluginCreatorInterface* const* getCreators(
    int32_t& nbCreators)
{
    static nvinfer1::IPluginCreatorInterface* const creators[] = {&gSelectiveScanCreator};
    nbCreators = 1;
    return creators;
}

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    (void)finder;
}
