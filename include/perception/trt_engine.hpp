#pragma once

#include "perception/base_detector.hpp"
#include <NvInfer.h>
#include <torch/torch.h>
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
    virtual bool set_tensor_address(const char* name, void* ptr) = 0;
    virtual bool enqueue_v3(cudaStream_t stream) = 0;
    virtual nvinfer1::Dims getTensorDims(const char* name) const = 0;
    virtual nvinfer1::Dims getTensorProfileDims(
        const char* name,
        int profile_index,
        nvinfer1::OptProfileSelector selector
    ) const = 0;
    virtual int get_nb_tensors() const = 0;
    virtual const char* get_tensor_name(int index) const = 0;
    virtual bool is_input(const char* name) const = 0;
    virtual bool set_input_shape(const char* name, const std::vector<int64_t>& shape) = 0;
};

/**
 * @brief TensorRT 核心引擎實作
 * 僅在 libperception.so 內部實作，外部透過導出宏使用。
 * 
 * 支援多執行緒：透過 create_context() 為每個執行緒建立獨立的
 * execution context，共享同一 engine_ 例項（只讀，線程安全）。
 */
class SACCADE_PERCEPTION_API TRTEngine : public IPerceptionEngine, public BaseDetector {
public:
    TRTEngine(const std::string& model_path);
    ~TRTEngine() override;

    // IPerceptionEngine interface
    bool infer(const std::vector<void*>& bindings, cudaStream_t stream) override;
    bool set_tensor_address(const char* name, void* ptr) override;
    bool enqueue_v3(cudaStream_t stream) override;
    nvinfer1::Dims getTensorDims(const char* name) const override;
    nvinfer1::Dims getTensorProfileDims(
        const char* name,
        int profile_index,
        nvinfer1::OptProfileSelector selector
    ) const override;
    int get_nb_tensors() const override;
    const char* get_tensor_name(int index) const override;
    bool is_input(const char* name) const override;
    bool set_input_shape(const char* name, const std::vector<int64_t>& shape) override;

    // Context factory for concurrent evaluation
    /**
     * @brief 建立獨立的執行上下文（用於多序列併發推論）
     * @return 新建立的 execution context 指標（呼叫端負責 delete_context()）
     */
    nvinfer1::IExecutionContext* create_context() const;

    /**
     * @brief 使用指定 context 執行推理（線程安全，每個 context 獨立）
     */
    bool infer_with_context(nvinfer1::IExecutionContext* ctx,
                            const std::vector<void*>& bindings,
                            cudaStream_t stream);

    /**
     * @brief 釋放由 create_context() 建立的 context
     */
    static void delete_context(nvinfer1::IExecutionContext* ctx);

    // BaseDetector interface
    torch::Tensor forward(torch::Tensor d_input_img) override;
    torch::Tensor extract_fpn_embeddings(torch::Tensor d_boxes_xyxy) override;
    int forward_ptr(uintptr_t d_input_img, uintptr_t d_out_dets) override;
    void extract_fpn_embeddings_ptr(uintptr_t d_boxes_xyxy, int num_dets, uintptr_t d_out_embs) override;
    int get_img_size() const override;
    int get_fpn_dim() const override;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace saccade
