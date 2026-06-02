#include "perception/batched_mamba_detector.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            std::ostringstream _ss;                                               \
            _ss << "CUDA error " << cudaGetErrorString(_e)                       \
                << " at " << __FILE__ << ":" << __LINE__;                        \
            throw std::runtime_error(_ss.str());                                 \
        }                                                                         \
    } while (0)

namespace saccade {

BatchedBackbone::BatchedBackbone(const std::string& trt_engine_path,
                                 int max_batch, int img_size)
    : max_batch_(std::max(1, max_batch))
    , img_size_(img_size)
{
    engine_ = std::make_unique<TRTEngine>(trt_engine_path);
    ctx_    = engine_->create_context();
    if (!ctx_)
        throw std::runtime_error("BatchedBackbone: failed to create TRT context");
    CHECK_CUDA(cudaStreamCreate(&stream_));

    // Allocate batch-sized output buffers
    const int N = max_batch_;
    CHECK_CUDA(cudaMalloc(&d_batch_p3_,
        (size_t)N * p3_c * p3_h * p3_w * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_p4_,
        (size_t)N * p4_c * p4_h * p4_w * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_p5_,
        (size_t)N * p5_c * p5_h * p5_w * sizeof(float)));
}

BatchedBackbone::~BatchedBackbone() {
    if (d_batch_p3_) cudaFree(d_batch_p3_);
    if (d_batch_p4_) cudaFree(d_batch_p4_);
    if (d_batch_p5_) cudaFree(d_batch_p5_);
    if (stream_)     cudaStreamDestroy(stream_);
    if (ctx_)        TRTEngine::delete_context(ctx_);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BatchedBackbone::batch_infer(torch::Tensor batch_frames) {
    auto frame_cont = batch_frames.contiguous();
    const int N = (int)frame_cont.size(0);

    if (N < 1 || N > max_batch_)
        throw std::runtime_error("BatchedBackbone: batch size out of range");

    auto opts_f = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(torch::kCUDA);

    // Set dynamic batch shape
    nvinfer1::Dims4 in_dims{N, 3, img_size_, img_size_};
    ctx_->setInputShape("images", in_dims);

    // Run TRT backbone on dedicated stream
    engine_->infer_with_context(
        ctx_,
        {frame_cont.data_ptr<float>(), d_batch_p3_, d_batch_p4_, d_batch_p5_},
        stream_);

    CHECK_CUDA(cudaStreamSynchronize(stream_));

    // Wrap outputs as PyTorch tensors (cloned so caller owns the data)
    auto p3 = torch::from_blob(d_batch_p3_,
        {N, p3_c, p3_h, p3_w}, opts_f).clone();
    auto p4 = torch::from_blob(d_batch_p4_,
        {N, p4_c, p4_h, p4_w}, opts_f).clone();
    auto p5 = torch::from_blob(d_batch_p5_,
        {N, p5_c, p5_h, p5_w}, opts_f).clone();

    return {p3, p4, p5};
}

} // namespace saccade
