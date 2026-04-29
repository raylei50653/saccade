#include "perception/trt_engine.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>

namespace saccade {

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

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

        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) throw std::runtime_error("❌ Failed to create TRT Runtime");

        engine_.reset(runtime_->deserializeCudaEngine(model_data.data(), size));
        if (!engine_) throw std::runtime_error("❌ Failed to deserialize engine");

        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("❌ Failed to create execution context");

        std::cout << "✅ [TRTEngine] Pimpl Loaded: " << model_path << std::endl;
    }

    bool infer(const std::vector<void*>& bindings, cudaStream_t stream) {
        int nbTensors = engine_->getNbIOTensors();
        for (int i = 0; i < nbTensors; ++i) {
            if (i >= (int)bindings.size()) continue;
            const char* name = engine_->getIOTensorName(i);
            context_->setTensorAddress(name, bindings[i]);
        }
        return context_->enqueueV3(stream);
    }

    bool setTensorAddress(const char* name, void* ptr) {
        return context_->setTensorAddress(name, ptr);
    }

    bool enqueueV3(cudaStream_t stream) {
        return context_->enqueueV3(stream);
    }

    nvinfer1::Dims getTensorDims(const char* name) const {
        return engine_->getTensorShape(name);
    }

    nvinfer1::Dims getTensorProfileDims(
        const char* name,
        int profile_index,
        nvinfer1::OptProfileSelector selector
    ) const {
        return engine_->getProfileShape(name, profile_index, selector);
    }

    int getNbTensors() const {
        return engine_->getNbIOTensors();
    }

    const char* getTensorName(int index) const {
        return engine_->getIOTensorName(index);
    }

    bool isInput(const char* name) const {
        return engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    }

    bool setInputShape(const char* name, const std::vector<int64_t>& shape) {
        nvinfer1::Dims dims;
        dims.nbDims = shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            dims.d[i] = shape[i];
        }
        return context_->setInputShape(name, dims);
    }

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

// --- TRTEngine Forwarding ---

TRTEngine::TRTEngine(const std::string& model_path)
    : pimpl_(std::make_unique<Impl>(model_path)) {}

TRTEngine::~TRTEngine() = default;

bool TRTEngine::infer(const std::vector<void*>& bindings, cudaStream_t stream) {
    return pimpl_->infer(bindings, stream);
}

bool TRTEngine::set_tensor_address(const char* name, void* ptr) {
    return pimpl_->setTensorAddress(name, ptr);
}

bool TRTEngine::enqueue_v3(cudaStream_t stream) {
    return pimpl_->enqueueV3(stream);
}

nvinfer1::Dims TRTEngine::getTensorDims(const char* name) const {
    return pimpl_->getTensorDims(name);
}

nvinfer1::Dims TRTEngine::getTensorProfileDims(
    const char* name,
    int profile_index,
    nvinfer1::OptProfileSelector selector
) const {
    return pimpl_->getTensorProfileDims(name, profile_index, selector);
}

int TRTEngine::get_nb_tensors() const {
    return pimpl_->getNbTensors();
}

const char* TRTEngine::get_tensor_name(int index) const {
    return pimpl_->getTensorName(index);
}

bool TRTEngine::is_input(const char* name) const {
    return pimpl_->isInput(name);
}

bool TRTEngine::set_input_shape(const char* name, const std::vector<int64_t>& shape) {
    return pimpl_->setInputShape(name, shape);
}

} // namespace saccade
