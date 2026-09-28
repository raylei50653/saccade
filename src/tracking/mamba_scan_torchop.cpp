// LibTorch operator for the Mamba selective scan (#465 Phase B PR-1L).
//
// Registers `saccade_native::selective_scan_fwd` in C++ so a TorchScript head
// artifact can run the scan without Python. It mirrors the Python custom op
// `saccade::selective_scan_fwd` (src/saccade/perception/temporal_yolo/
// mamba_head.py) argument for argument and launches the same
// `selective_scan_fwd` / `selective_scan_fwd_half` launchers from
// mamba_scan.cu on the current CUDA stream (so CUDA-graph capture records it).
//
// The namespace differs from the Python op's on purpose: a process that has
// both the Python op and this library loaded (the parity harness) must not
// get a duplicate-registration error, and a traced artifact must name exactly
// one implementation.
//
// Built as a self-contained shared library (libsaccade_scan_torchop.so, see
// CMakeLists.txt): it links libtorch/libc10 and cudart only -- never
// libtorch_python or libpython -- so a native runtime can dlopen it.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include "tracking/mamba_scan.cuh"

namespace {

bool valid_state_count(int64_t n) { return n >= 1 && n <= 32 && (n & (n - 1)) == 0; }

at::Tensor selective_scan_fwd_cuda(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    int64_t a_per_channel,
    bool is_half) {
    TORCH_CHECK(u.is_cuda(), "saccade_native::selective_scan_fwd: u must be CUDA");
    TORCH_CHECK(u.dim() == 3, "saccade_native::selective_scan_fwd: u must be (B, L, D)");
    const int64_t N = A.size(-1);
    TORCH_CHECK(valid_state_count(N),
                "saccade_native::selective_scan_fwd requires power-of-two N in [1, 32], got ", N);
    TORCH_CHECK(is_half == (u.scalar_type() == at::kHalf),
                "saccade_native::selective_scan_fwd: is_half does not match u dtype");
    TORCH_CHECK(is_half || u.scalar_type() == at::kFloat,
                "saccade_native::selective_scan_fwd: u must be float32 or float16");

    const at::Tensor u_c = u.contiguous();
    const at::Tensor delta_c = delta.contiguous();
    const at::Tensor A_c = A.contiguous();
    const at::Tensor B_c = B.contiguous();
    const at::Tensor C_c = C.contiguous();
    const bool has_D = D.numel() > 0;
    const at::Tensor D_c = has_D ? D.contiguous() : D;
    at::Tensor y = at::empty_like(u_c);

    SelectiveScanParams params;
    params.B = static_cast<int>(u_c.size(0));
    params.L = static_cast<int>(u_c.size(1));
    params.D = static_cast<int>(u_c.size(2));
    params.N = static_cast<int>(N);
    params.has_D = has_D;
    params.a_per_channel = a_per_channel != 0;

    void* stream = at::cuda::getCurrentCUDAStream(u_c.device().index()).stream();
    if (is_half) {
        selective_scan_fwd_half(u_c.data_ptr(), delta_c.data_ptr(), A_c.data_ptr(),
                                B_c.data_ptr(), C_c.data_ptr(),
                                has_D ? D_c.data_ptr() : nullptr, y.data_ptr(), params,
                                stream);
    } else {
        selective_scan_fwd(u_c.data_ptr<float>(), delta_c.data_ptr<float>(),
                           A_c.data_ptr<float>(), B_c.data_ptr<float>(),
                           C_c.data_ptr<float>(),
                           has_D ? D_c.data_ptr<float>() : nullptr, y.data_ptr<float>(),
                           params, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

}  // namespace

TORCH_LIBRARY(saccade_native, m) {
    m.def(
        "selective_scan_fwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, "
        "Tensor D, int a_per_channel, bool is_half) -> Tensor");
}

TORCH_LIBRARY_IMPL(saccade_native, CUDA, m) {
    m.impl("selective_scan_fwd", &selective_scan_fwd_cuda);
}
