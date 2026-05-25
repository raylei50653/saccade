#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

__global__ void centre_pool_kernel(
    const float* feat, int C, int H, int W,
    const float* boxes, int N,
    float* pooled, int img_size
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float cx = (boxes[n*4+0] + boxes[n*4+2]) * 0.5f;
    float cy = (boxes[n*4+1] + boxes[n*4+3]) * 0.5f;
    int cx_idx = static_cast<int>(cx * W / img_size);
    int cy_idx = static_cast<int>(cy * H / img_size);
    if (cx_idx < 0) cx_idx = 0; if (cx_idx >= W) cx_idx = W - 1;
    if (cy_idx < 0) cy_idx = 0; if (cy_idx >= H) cy_idx = H - 1;

    for (int c = 0; c < C; c++) {
        pooled[n * C + c] = feat[c * H * W + cy_idx * W + cx_idx];
    }
}

__global__ void conv1x1_kernel(
    const float* centre_feat, int C,
    const float* weight, int O,
    float* out, int N
) {
    int n = blockIdx.x;
    int o = threadIdx.x;
    if (n >= N || o >= O) return;

    float sum = 0.0f;
    const float* cf = centre_feat + n * C;
    const float* w = weight + o * C;
    for (int c = 0; c < C; c++) {
        sum += cf[c] * w[c];
    }
    out[n * O + o] = sum;
}

__global__ void l2_normalise_kernel(float* data, int D, int N, float eps) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float* row = data + n * D;
    float norm = 0.0f;
    for (int d = 0; d < D; d++) norm += row[d] * row[d];
    norm = sqrtf(norm + eps);
    for (int d = 0; d < D; d++) row[d] /= norm;
}

__global__ void bn1d_eval_kernel(
    float* data, int D, int N,
    const float* running_mean,
    const float* running_var,
    float eps
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float* row = data + n * D;
    for (int d = 0; d < D; d++) {
        row[d] = (row[d] - running_mean[d]) / sqrtf(running_var[d] + eps);
    }
}

__global__ void linear_kernel(
    const float* data, int D,
    const float* weight, int O,
    float* out, int N
) {
    int n = blockIdx.x;
    int o = threadIdx.x;
    if (n >= N || o >= O) return;

    float sum = 0.0f;
    const float* d = data + n * D;
    const float* w = weight + o * D;
    for (int i = 0; i < D; i++) sum += d[i] * w[i];
    out[n * O + o] = sum;
}

} // anonymous namespace


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
    int out_dim = conv_weights[0].size(0);
    bool has_proj = proj_weight.numel() > 0;
    bool has_bn = running_mean.numel() > 0;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(boxes.device());
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(boxes.get_device());
    cudaStream_t stream = cuda_stream.stream();

    std::vector<torch::Tensor> pooled;
    for (int s = 0; s < n_scales; s++) {
        int C = static_cast<int>(feats[s].size(1));
        int H = static_cast<int>(feats[s].size(2));
        int W = static_cast<int>(feats[s].size(3));
        auto pool = torch::empty({N, C}, options);
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        centre_pool_kernel<<<blocks, threads, 0, stream>>>(
            feats[s].data_ptr<float>(), C, H, W,
            boxes.data_ptr<float>(), N,
            pool.data_ptr<float>(),
            static_cast<int>(img_size)
        );
        pooled.push_back(pool);
    }

    std::vector<torch::Tensor> conv_out;
    for (int s = 0; s < n_scales; s++) {
        int C = static_cast<int>(conv_weights[s].size(1));
        auto out = torch::empty({N, out_dim}, options);
        conv1x1_kernel<<<N, out_dim, 0, stream>>>(
            pooled[s].data_ptr<float>(), C,
            conv_weights[s].data_ptr<float>(), out_dim,
            out.data_ptr<float>(), N
        );
        conv_out.push_back(out);
    }

    auto cat = torch::cat(conv_out, 1);
    torch::Tensor result;

    if (has_proj) {
        int mid_dim = static_cast<int>(cat.size(1));
        auto projected = torch::empty({N, out_dim}, options);
        linear_kernel<<<N, out_dim, 0, stream>>>(
            cat.data_ptr<float>(), mid_dim,
            proj_weight.data_ptr<float>(), out_dim,
            projected.data_ptr<float>(), N
        );

        if (has_bn) {
            int bn_threads = 256;
            int bn_blocks = (N + bn_threads - 1) / bn_threads;
            bn1d_eval_kernel<<<bn_blocks, bn_threads, 0, stream>>>(
                projected.data_ptr<float>(), out_dim, N,
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                static_cast<float>(eps)
            );
        }
        result = projected;
    } else {
        result = cat;
    }

    int D = static_cast<int>(result.size(1));
    int l2_threads = 256;
    int l2_blocks = (N + l2_threads - 1) / l2_threads;
    l2_normalise_kernel<<<l2_blocks, l2_threads, 0, stream>>>(
        result.data_ptr<float>(), D, N, 1e-8f
    );

    return result;
}


PYBIND11_MODULE(saccade_fpn_reid_cuda, m) {
    m.def("fpn_reid_extract", &fpn_reid_extract, "FPN centre-pool + conv ReID");
}
