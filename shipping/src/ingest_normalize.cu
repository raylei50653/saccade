// The oracle's ingest op on device (#465 Phase B PR-7): float(x) * scale,
// round to nearest -- torch's CUDA kernel for torch.div(uint8, 255.0)
// multiplies by the float32 reciprocal. See saccade_shipping/ingest_host.hpp.
#include <stdexcept>
#include <string>

#include "saccade_shipping/ingest_host.hpp"

namespace saccade::shipping {
namespace {

__global__ void normalize_rgb_u8_kernel(const std::uint8_t* __restrict__ in, float* __restrict__ out,
                                        std::size_t count, float scale) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        out[i] = __fmul_rn(static_cast<float>(in[i]), scale);
    }
}

}  // namespace

void normalize_rgb_u8(const std::uint8_t* in, float* out, std::size_t count, float scale,
                      cudaStream_t stream) {
    if (count == 0) return;
    constexpr int kThreads = 256;
    const std::size_t blocks = (count + kThreads - 1) / kThreads;
    const unsigned grid = static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
    normalize_rgb_u8_kernel<<<grid, kThreads, 0, stream>>>(in, out, count, scale);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("shipping ingest: normalize launch: ") + cudaGetErrorString(e));
    }
}

}  // namespace saccade::shipping
