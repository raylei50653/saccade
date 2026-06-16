#include "perception/trt_engine.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

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
    Impl(const std::string& model_path) : internal_stream_(nullptr) {
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

        cudaStreamCreateWithFlags(&internal_stream_, cudaStreamNonBlocking);

        try {
            runtime_.reset(nvinfer1::createInferRuntime(gLogger));
            if (!runtime_) throw std::runtime_error("❌ Failed to create TRT Runtime");

            engine_.reset(runtime_->deserializeCudaEngine(model_data.data(), size));
            if (!engine_) throw std::runtime_error("❌ Failed to deserialize engine");
        } catch (...) {
            cudaStreamDestroy(internal_stream_);
            internal_stream_ = nullptr;
            throw;
        }

        // context_ is created lazily on first infer()/enqueueV3() call so that
        // callers that only query metadata (tensor shapes, engine ptr) or use
        // infer_with_context() with their own context don't pay the VRAM cost.
        std::cout << "✅ [TRTEngine] Pimpl Loaded: " << model_path << std::endl;
    }

    ~Impl() {
        if (internal_stream_) cudaStreamDestroy(internal_stream_);
    }

    nvinfer1::IExecutionContext* ensure_context() {
        if (!context_) {
            context_.reset(engine_->createExecutionContext());
            if (!context_) throw std::runtime_error("❌ Failed to create execution context");
        }
        return context_.get();
    }

    bool infer(const std::vector<void*>& bindings, cudaStream_t stream) {
        cudaStream_t s = stream ? stream : internal_stream_;
        auto* ctx = ensure_context();
        int nbTensors = engine_->getNbIOTensors();
        for (int i = 0; i < nbTensors; ++i) {
            if (i >= (int)bindings.size()) continue;
            const char* name = engine_->getIOTensorName(i);
            ctx->setTensorAddress(name, bindings[i]);
        }
        return ctx->enqueueV3(s);
    }

    bool setTensorAddress(const char* name, void* ptr) {
        return ensure_context()->setTensorAddress(name, ptr);
    }

    bool enqueueV3(cudaStream_t stream) {
        return ensure_context()->enqueueV3(stream ? stream : internal_stream_);
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
        return ensure_context()->setInputShape(name, dims);
    }

    nvinfer1::IExecutionContext* create_context() const {
        return engine_->createExecutionContext();
    }

    bool infer_with_context(nvinfer1::IExecutionContext* ctx,
                            const std::vector<void*>& bindings,
                            cudaStream_t stream) {
        cudaStream_t s = stream ? stream : internal_stream_;
        int nbTensors = engine_->getNbIOTensors();
        for (int i = 0; i < nbTensors; ++i) {
            if (i >= (int)bindings.size()) continue;
            const char* name = engine_->getIOTensorName(i);
            ctx->setTensorAddress(name, bindings[i]);
        }
        return ctx->enqueueV3(s);
    }

private:
    cudaStream_t internal_stream_;
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

nvinfer1::IExecutionContext* TRTEngine::create_context() const {
    return pimpl_->create_context();
}

bool TRTEngine::infer_with_context(nvinfer1::IExecutionContext* ctx,
                                    const std::vector<void*>& bindings,
                                    cudaStream_t stream) {
    return pimpl_->infer_with_context(ctx, bindings, stream);
}

void TRTEngine::delete_context(nvinfer1::IExecutionContext* ctx) {
    delete ctx;
}

// ── BaseDetector interface implementations ──────────────────────────────────

torch::Tensor TRTEngine::forward(torch::Tensor d_input_img) {
    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto out_dims = getTensorDims("output0");
    std::vector<int64_t> out_shape;
    for (int i = 0; i < out_dims.nbDims; ++i) {
        out_shape.push_back(out_dims.d[i]);
    }
    if (out_shape.empty()) {
        out_shape = {1, 8400, 6}; // YOLO default fallback
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(d_input_img.device());
    auto y = torch::empty(out_shape, options);

    std::vector<void*> bindings = { d_input_img.data_ptr(), y.data_ptr() };
    infer(bindings, current_stream.stream());
    return y;
}

torch::Tensor TRTEngine::extract_fpn_embeddings(torch::Tensor d_boxes_xyxy) {
    // Standard TRT backbone has no appearance embedding branch; return zero dummy embeddings
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(d_boxes_xyxy.device());
    auto out = torch::zeros({d_boxes_xyxy.size(0), get_fpn_dim()}, options);
    return out / (out.norm(2, 1, true) + 1e-12f);
}

int TRTEngine::forward_ptr(uintptr_t d_input_img, uintptr_t d_out_dets) {
    auto current_stream = at::cuda::getCurrentCUDAStream();
    std::vector<void*> bindings = { reinterpret_cast<void*>(d_input_img), reinterpret_cast<void*>(d_out_dets) };
    infer(bindings, current_stream.stream());
    
    // Return max_raw_dets from output0 dims
    auto out_dims = getTensorDims("output0");
    if (out_dims.nbDims >= 2) {
        return out_dims.d[1];
    }
    return 8400; // YOLO standard default fallback
}

void TRTEngine::extract_fpn_embeddings_ptr(uintptr_t d_boxes_xyxy, int num_dets, uintptr_t d_out_embs) {
    if (num_dets == 0) return;
    auto current_stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(reinterpret_cast<void*>(d_out_embs), 0, num_dets * get_fpn_dim() * sizeof(float), current_stream.stream());
}

int TRTEngine::get_img_size() const {
    auto dims = getTensorDims("images");
    if (dims.nbDims >= 3) {
        return dims.d[2];
    }
    return 640; // standard default fallback
}

int TRTEngine::get_fpn_dim() const {
    return 128;
}

} // namespace saccade
