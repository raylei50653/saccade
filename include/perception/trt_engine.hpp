#include "saccade/common.hpp"
#include <NvInfer.h>
#include <string>
#include <vector>

namespace saccade {

/**
 * @brief 感知引擎接口 (IPerceptionEngine)
 * 用於解耦不同的偵測/提取實作。
 */
class SACCADE_PERCEPTION_API IPerceptionEngine {
public:
    virtual ~IPerceptionEngine() = default;
    virtual bool infer(const std::vector<void*>& bindings, cudaStream_t stream) = 0;
    virtual nvinfer1::Dims getTensorDims(const char* name) const = 0;
};

/**
 * @brief TensorRT 核心引擎實作
 * 僅在 libperception.so 內部實作，外部透過導出宏使用。
 */
class SACCADE_PERCEPTION_API TRTEngine : public IPerceptionEngine {
public:
    TRTEngine(const std::string& model_path);
    ~TRTEngine();

    bool infer(const std::vector<void*>& bindings, cudaStream_t stream) override;
    nvinfer1::Dims getTensorDims(const char* name) const override;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace saccade
