// FPN ReID torch binding — orchestrates tensor allocation and calls the
// C-linkage CUDA launchers in fpn_reid_cuda.cu.  Kept in a .cpp (host-compiler
// only, never parsed by nvcc's front-end) so torch 2.x headers compile cleanly.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "tracking/fpn_reid_launchers.cuh"

torch::Tensor fpn_reid_extract(
    std::vector<torch::Tensor> feats,
    std::vector<torch::Tensor> conv_weights,
    torch::Tensor boxes,
    int64_t img_size,
    torch::Tensor proj_weight,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    int N = boxes.size(0);
    int n_scales = static_cast<int>(feats.size());
    int out_dim = static_cast<int>(conv_weights[0].size(0));
    bool has_proj = proj_weight.numel() > 0;
    bool has_bn = running_mean.numel() > 0;

    // conv1x1_kernel / linear_kernel launch with out_dim threads per block;
    // CUDA caps threads/block at 1024. Guard against oversized ReID heads
    // (a silent launch failure would otherwise produce garbage embeddings).
    TORCH_CHECK(out_dim > 0 && out_dim <= 1024,
                "fpn_reid_extract: out_dim must be in (0, 1024], got ", out_dim);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(boxes.device());
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(boxes.get_device());
    cudaStream_t stream = cuda_stream.stream();

    std::vector<torch::Tensor> pooled;
    for (int s = 0; s < n_scales; s++) {
        int C = static_cast<int>(feats[s].size(1));
        int H = static_cast<int>(feats[s].size(2));
        int W = static_cast<int>(feats[s].size(3));
        auto pool = torch::empty({N, C}, options);
        fpn_centre_pool(
            feats[s].data_ptr<float>(), C, H, W,
            boxes.data_ptr<float>(), N,
            pool.data_ptr<float>(),
            static_cast<int>(img_size), stream);
        pooled.push_back(pool);
    }

    std::vector<torch::Tensor> conv_out;
    for (int s = 0; s < n_scales; s++) {
        int C = static_cast<int>(conv_weights[s].size(1));
        auto out = torch::empty({N, out_dim}, options);
        fpn_conv1x1(
            pooled[s].data_ptr<float>(), C,
            conv_weights[s].data_ptr<float>(), out_dim,
            out.data_ptr<float>(), N, stream);
        conv_out.push_back(out);
    }

    auto cat = torch::cat(conv_out, 1);
    torch::Tensor result;

    if (has_proj) {
        int mid_dim = static_cast<int>(cat.size(1));
        auto projected = torch::empty({N, out_dim}, options);
        fpn_linear(
            cat.data_ptr<float>(), mid_dim,
            proj_weight.data_ptr<float>(), out_dim,
            projected.data_ptr<float>(), N, stream);

        if (has_bn) {
            fpn_bn1d(
                projected.data_ptr<float>(), out_dim, N,
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                static_cast<float>(eps), stream);
        }
        result = projected;
    } else {
        result = cat;
    }

    int D = static_cast<int>(result.size(1));
    fpn_l2_normalise(result.data_ptr<float>(), D, N, 1e-8f, stream);

    return result;
}


PYBIND11_MODULE(saccade_fpn_reid_cuda, m) {
    m.def("fpn_reid_extract", &fpn_reid_extract, "FPN centre-pool + conv ReID");
}
