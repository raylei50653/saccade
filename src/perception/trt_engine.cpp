#include "perception/trt_engine.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>

namespace saccade {

// 內部的日誌記錄器
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

// Pimpl 內部類別實作
class TRTEngine::Impl {
public:
    Impl(const std::string& model_path) {
        std::ifstream file(model_path, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("❌ [TRTEngine] Model not found: " + model_path);
        }

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> model_data(size);
        file.read(model_data.data(), size);
        file.close();

        // 建立 TensorRT 執行時組件 (RAII)
        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) throw std::runtime_error("❌ Failed to create TRT Runtime");

        engine_.reset(runtime_->deserializeCudaEngine(model_data.data(), size));
        if (!engine_) throw std::runtime_error("❌ Failed to deserialize engine");

        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("❌ Failed to create execution context");

        std::cout << "✅ [TRTEngine] Pimpl Loaded: " << model_path << std::endl;
    }

    bool infer(const std::vector<void*>& bindings, cudaStream_t stream) {
        // 配置 Tensor 位址 (TensorRT 8.6+ V3 API)
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            context_->setTensorAddress(name, bindings[i]);
        }
        return context_->enqueueV3(stream);
    }

    nvinfer1::Dims getTensorDims(const char* name) const {
        return engine_->getTensorShape(name);
    }

private:
    // 使用自定義 Deleter 確保 TensorRT 物件正確釋放
    struct TRTDeleter {
        template <typename T>
        void operator()(T* obj) const { if (obj) obj->destroy(); }
    };

    // 由於 TensorRT 版本在不同環境中可能不同，這裡使用智能指針管理物件生命週期
    // 註：TensorRT 10.0+ 建議使用 std::unique_ptr 配合預設 Deleter
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

// --- TRTEngine 公共接口轉發 ---

TRTEngine::TRTEngine(const std::string& model_path)
    : pimpl_(std::make_unique<Impl>(model_path)) {}

TRTEngine::~TRTEngine() = default;

bool TRTEngine::infer(const std::vector<void*>& bindings, cudaStream_t stream) {
    return pimpl_->infer(bindings, stream);
}

nvinfer1::Dims TRTEngine::getTensorDims(const char* name) const {
    return pimpl_->getTensorDims(name);
}

} // namespace saccade
