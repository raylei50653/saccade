#pragma once

#include "saccade/common.hpp"
#include "perception/trt_engine.hpp"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

namespace saccade {

/**
 * @brief TRT backbone wrapper that supports batch-N inference.
 *
 * Exposes a single batch_infer() method:
 *   Input:  [N, 3, H, W] float32 GPU tensor
 *   Output: tuple of (p3[N,128,80,80], p4[N,256,40,40], p5[N,512,20,20])
 *
 * This class is intentionally minimal — it only runs the TRT backbone.
 * The Mamba SSM head (which uses a Python custom_op) stays in Python.
 */
class SACCADE_PERCEPTION_API BatchedBackbone {
public:
    /**
     * @param trt_engine_path  Path to batch-N TRT engine file (.engine)
     * @param max_batch        Maximum batch size the engine was built with
     * @param img_size         Input spatial resolution (default 640)
     */
    BatchedBackbone(
        const std::string& trt_engine_path,
        int max_batch = 4,
        int img_size  = 640
    );

    ~BatchedBackbone();

    BatchedBackbone(const BatchedBackbone&) = delete;
    BatchedBackbone& operator=(const BatchedBackbone&) = delete;

    /**
     * @brief Run TRT backbone for a batch of N frames.
     * @param batch_frames  GPU Tensor [N, 3, H, W], float32, values in [0,1]
     * @return              tuple (p3, p4, p5) each with leading dim N
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    batch_infer(torch::Tensor batch_frames);

    int get_img_size()  const { return img_size_; }
    int get_max_batch() const { return max_batch_; }

private:
    std::unique_ptr<TRTEngine>       engine_;
    nvinfer1::IExecutionContext*     ctx_     = nullptr;
    cudaStream_t                     stream_  = nullptr;

    int max_batch_;
    int img_size_;

    // Pre-allocated batch GPU buffers
    void*  d_batch_p3_ = nullptr;  // [max_batch, 128, 80, 80]
    void*  d_batch_p4_ = nullptr;  // [max_batch, 256, 40, 40]
    void*  d_batch_p5_ = nullptr;  // [max_batch, 512, 20, 20]

    static constexpr int p3_c = 128, p3_h = 80, p3_w = 80;
    static constexpr int p4_c = 256, p4_h = 40, p4_w = 40;
    static constexpr int p5_c = 512, p5_h = 20, p5_w = 20;
};

} // namespace saccade
