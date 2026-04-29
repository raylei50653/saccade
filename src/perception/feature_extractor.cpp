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

void launch_parts_fuse(
    const float* parts_ptr, float* out_ptr,
    int num_dets, int feat_dim,
    float w0, float w1, float w2,
    cudaStream_t stream);

FeatureExtractor::FeatureExtractor(const std::string& model_path, ModelType type, int max_batch)
    : type_(type), max_batch_(max_batch) {
    
    engine_ = std::make_unique<TRTEngine>(model_path);
    
    // Auto-detect names and shapes
    bool is_siglip_family = (type == ModelType::SIGLIP2);
    input_name_ = is_siglip_family ? "pixel_values" : "input";
    output_name_ = is_siglip_family ? "image_embeds" : "output";
    
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
    if (is_dynamic_) {
        auto opt_dims = engine_->getTensorProfileDims(
            input_name_.c_str(), 0, nvinfer1::OptProfileSelector::kOPT
        );
        auto max_dims = engine_->getTensorProfileDims(
            input_name_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX
        );
        if (opt_dims.nbDims >= 4) {
            input_h_ = opt_dims.d[2];
            input_w_ = opt_dims.d[3];
        }
        if (max_dims.nbDims >= 1 && max_dims.d[0] > 0) {
            const int profile_max_batch = static_cast<int>(max_dims.d[0]);
            if (max_batch_ > 0) {
                max_batch_ = std::min(max_batch_, profile_max_batch);
            } else {
                max_batch_ = profile_max_batch;
            }
        }
    }
    max_batch_ = std::max(max_batch_, 1);
    
    // Validate primary output and discover others.
    // Some exported ReID engines use names like `image_embeds` or `features`
    // instead of the legacy `output`, so fall back to the first non-input tensor.
    bool found_primary_output = false;
    std::string first_output_name;
    for (int i = 0; i < engine_->get_nb_tensors(); ++i) {
        const char* name = engine_->get_tensor_name(i);
        if (engine_->is_input(name)) continue;

        if (first_output_name.empty()) {
            first_output_name = name;
        }

        if (std::string(name) == output_name_) {
            auto out_dims = engine_->getTensorDims(name);
            feature_dim_ = out_dims.d[out_dims.nbDims - 1];
            found_primary_output = true;
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

    if (!found_primary_output) {
        if (first_output_name.empty()) {
            throw std::runtime_error("FeatureExtractor: no output tensors found in engine");
        }

        output_name_ = first_output_name;
        auto out_dims = engine_->getTensorDims(output_name_.c_str());
        feature_dim_ = out_dims.d[out_dims.nbDims - 1];

        for (auto it = scratch_buffers_.begin(); it != scratch_buffers_.end(); ++it) {
            if (it->first == output_name_) {
                cudaFree(it->second);
                scratch_buffers_.erase(it);
                break;
            }
        }

        std::cout << "ℹ️ [FeatureExtractor] Falling back to primary output tensor: "
                  << output_name_ << " (dim=" << feature_dim_ << ")" << std::endl;
    }

    // OSNET and FastReID use the same ImageNet normalization as DINOv2/TransReID.
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

void FeatureExtractor::extract_parts_fused(
    void* input_cuda_ptr, int num_dets,
    void* output_cuda_ptr, cudaStream_t stream)
{
    if (num_dets <= 0) return;
    // input_cuda_ptr: [3*num_dets, 3, H, W] — 3 parts stacked in part-major order
    // Step 1: extract all 3*num_dets crops into a temporary buffer [3*num_dets, feat_dim]
    void* part_buf = nullptr;
    cudaMalloc(&part_buf, 3 * num_dets * feature_dim_ * sizeof(float));
    extract(input_cuda_ptr, 3 * num_dets, part_buf, stream);
    // Step 2: weighted sum [0.5, 0.3, 0.2] over parts → [num_dets, feat_dim]
    launch_parts_fuse(
        (const float*)part_buf, (float*)output_cuda_ptr,
        num_dets, feature_dim_,
        0.5f, 0.3f, 0.2f, stream);
    // Step 3: L2 normalize result
    launch_l2_normalize((float*)output_cuda_ptr, num_dets, feature_dim_, stream);
    cudaFree(part_buf);
}

int FeatureExtractor::get_feature_dim() const { return feature_dim_; }
int FeatureExtractor::get_max_batch() const { return max_batch_; }
std::pair<int, int> FeatureExtractor::get_input_hw() const { return {input_h_, input_w_}; }

} // namespace saccade
