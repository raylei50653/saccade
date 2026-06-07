#pragma once

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// TensorRT Plugin: SelectiveScan
// Wraps the custom CUDA kernel selective_scan_fwd (mamba_scan.cu)
// for use inside a TensorRT engine built from ONNX.
//
// ONNX op signature:
//   domain = "saccade"
//   type   = "SelectiveScan"
//
// Inputs (6-7 tensors):
//   0. u     : (B, L, D)   float or half
//   1. delta : (B, L, D)   float or half
//   2. A     : (rows, N)   float or half  — rows=1 (shared) or D (per-channel)
//   3. B     : (B, L, N)   float or half
//   4. C     : (B, L, N)   float or half
//   5. D     : (D,)         float or half  — skip connection (optional)
//   6. params: (2) int64    a_per_channel, has_D  — shape tensor input
//
// Output:
//   0. y     : (B, L, D)   same dtype as u
// ---------------------------------------------------------------------------

namespace saccade::trt {

constexpr char const* kSelectiveScanPluginName    = "SelectiveScan";
constexpr char const* kSelectiveScanPluginVersion = "1";
constexpr char const* kSelectiveScanNamespace     = "saccade";

struct SelectiveScanPluginParams {
    int32_t B = 0;
    int32_t L = 0;
    int32_t D = 0;
    int32_t N = 0;
    bool has_D = false;
    bool a_per_channel = false;
};

class SelectiveScanPlugin final : public nvinfer1::IPluginV3 {
public:
    SelectiveScanPlugin(SelectiveScanPluginParams params);

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    nvinfer1::IPluginV3* clone() noexcept override;

private:
    SelectiveScanPluginParams params_;

    class Core;
    class Build;
    class Runtime;

    Core*    core_    = nullptr;
    Build*   build_   = nullptr;
    Runtime* runtime_ = nullptr;
};

class SelectiveScanPluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(
        nvinfer1::AsciiChar const* name,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection field_collection_{};
};

} // namespace saccade::trt
