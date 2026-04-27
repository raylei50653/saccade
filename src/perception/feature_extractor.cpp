#include "perception/feature_extractor.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

namespace saccade {

// Extern kernels from preprocessor_gpu.cu
void launch_reid_pre_normalize(
    float* data, int num, int channels, int h, int w,
    const float* mean, const float* std, bool is_siglip,
    cudaStream_t stream);

void launch_l2_normalize(float* data, int n, int dim, cudaStream_t stream);

FeatureExtractor::FeatureExtractor(const std::string& model_path, ModelType type, int max_batch)
    : type_(type), max_batch_(max_batch) {
    
    engine_ = std::make_unique<TRTEngine>(model_path);
    
    // Auto-detect names and shapes
    input_name_ = (type == ModelType::SIGLIP2) ? "pixel_values" : "input";
    output_name_ = (type == ModelType::SIGLIP2) ? "image_embeds" : "output";
    
    // Validate primary input
    auto in_dims = engine_->getTensorDims(input_name_.c_str());
    if (in_dims.nbDims == -1) {
        // Fallback search
        for (int i = 0; i < engine_->get_nb_tensors(); ++i) {
            const char* name = engine_->get_tensor_name(i);
            if (engine_->is_input(name)) {
                input_name_ = name;
                in_dims = engine_->getTensorDims(name);
                break;
            }
        }
    }
    
    is_dynamic_ = (in_dims.d[0] == -1);
    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];
    
    // Validate primary output and discover others
    for (int i = 0; i < engine_->get_nb_tensors(); ++i) {
        const char* name = engine_->get_tensor_name(i);
        if (engine_->is_input(name)) continue;

        if (std::string(name) == output_name_) {
            auto out_dims = engine_->getTensorDims(name);
            feature_dim_ = out_dims.d[out_dims.nbDims - 1];
        } else {
            // Unused output tensor (like last_hidden_state in SigLIP2).
            // TensorRT 10+ requires ALL outputs to have an address.
            auto dims = engine_->getTensorDims(name);
            size_t size = 4; // float
            for (int j = 0; j < dims.nbDims; ++j) {
                int d = dims.d[j];
                size *= (d == -1 ? max_batch_ : d);
            }
            void* ptr = nullptr;
            cudaMalloc(&ptr, size);
            scratch_buffers_.push_back({name, ptr});
            std::cout << "🔍 [FeatureExtractor] Allocated scratch buffer for unused output: " << name << " (" << size << " bytes)" << std::endl;
        }
    }

    if (type_ != ModelType::SIGLIP2) {
        float h_mean[] = {0.485f, 0.456f, 0.406f};
        float h_std[] = {0.229f, 0.224f, 0.225f};
        cudaMalloc(&d_mean_, sizeof(h_mean));
        cudaMalloc(&d_std_, sizeof(h_std));
        cudaMemcpy(d_mean_, h_mean, sizeof(h_mean), cudaMemcpyHostToDevice);
        cudaMemcpy(d_std_, h_std, sizeof(h_std), cudaMemcpyHostToDevice);
    }
}

FeatureExtractor::~FeatureExtractor() {
    if (d_mean_) cudaFree(d_mean_);
    if (d_std_) cudaFree(d_std_);
    for (auto& pair : scratch_buffers_) {
        cudaFree(pair.second);
    }
}

void FeatureExtractor::extract(void* input_cuda_ptr, int num_images, void* output_cuda_ptr, cudaStream_t stream) {
    if (num_images <= 0) return;

    int processed = 0;
    while (processed < num_images) {
        int batch = std::min(num_images - processed, max_batch_);
        float* cur_input = (float*)input_cuda_ptr + processed * 3 * input_h_ * input_w_;
        float* cur_output = (float*)output_cuda_ptr + processed * feature_dim_;

        // 1. Pre-normalize
        launch_reid_pre_normalize(
            cur_input, batch, 3, input_h_, input_w_,
            (float*)d_mean_, (float*)d_std_, 
            type_ == ModelType::SIGLIP2, 
            stream
        );

        // 2. Set Input Shape if dynamic
        if (is_dynamic_) {
            engine_->set_input_shape(input_name_.c_str(), {batch, 3, input_h_, input_w_});
        }

        // 3. Bind all tensors (required for TensorRT 10+)
        engine_->set_tensor_address(input_name_.c_str(), cur_input);
        engine_->set_tensor_address(output_name_.c_str(), cur_output);
        for (auto& pair : scratch_buffers_) {
            engine_->set_tensor_address(pair.first.c_str(), pair.second);
        }

        // 4. Infer
        engine_->enqueue_v3(stream);

        // 5. L2 Normalize output
        launch_l2_normalize(cur_output, batch, feature_dim_, stream);

        processed += batch;
    }
}

int FeatureExtractor::get_feature_dim() const { return feature_dim_; }
int FeatureExtractor::get_max_batch() const { return max_batch_; }
std::pair<int, int> FeatureExtractor::get_input_hw() const { return {input_h_, input_w_}; }

} // namespace saccade
