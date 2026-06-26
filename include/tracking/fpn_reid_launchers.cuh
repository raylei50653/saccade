#pragma once

// C-linkage launchers for the FPN ReID CUDA kernels (fpn_reid_cuda.cu).
// Declared here so both the torch binding (fpn_reid_binding.cpp) and native
// tests can include them without re-declaring.
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void fpn_centre_pool(const float* feat, int C, int H, int W,
                     const float* boxes, int N,
                     float* pooled, int img_size, cudaStream_t stream);
void fpn_conv1x1(const float* centre_feat, int C,
                 const float* weight, int O,
                 float* out, int N, cudaStream_t stream);
void fpn_l2_normalise(float* data, int D, int N, float eps, cudaStream_t stream);
void fpn_bn1d(float* data, int D, int N,
              const float* running_mean, const float* running_var,
              float eps, cudaStream_t stream);
void fpn_linear(const float* data, int D,
                const float* weight, int O,
                float* out, int N, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
