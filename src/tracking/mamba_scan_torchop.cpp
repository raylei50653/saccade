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
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>

#include "tracking/mamba_scan.cuh"

namespace {

constexpr const char* kOp = "saccade_native::selective_scan_fwd: ";

bool valid_state_count(int64_t n) { return n >= 1 && n <= 32 && (n & (n - 1)) == 0; }

// Same device and dtype as u; the launcher reinterprets every pointer with
// u's element type on u's stream.
void check_like_u(const at::Tensor& t, const at::Tensor& u, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == u.device(), kOp, name,
                " must be on u's CUDA device ", u.device(), ", got ", t.device());
    TORCH_CHECK(t.scalar_type() == u.scalar_type(), kOp, name, " dtype ",
                t.scalar_type(), " != u dtype ", u.scalar_type());
}

// The operator boundary of the shipping runtime (shipping boundary G2): every
// input is validated against the contract of the Python op
// saccade::selective_scan_fwd as mamba_head._selective_scan_cuda calls it,
// before any raw pointer reaches the CUDA launcher, so a contract violation
// is a C++ error instead of an illegal access or a misread buffer.
//   u, delta : (B, L, D)                    B, L, D >= 1
//   A        : a_per_channel = 1 -> (D, N); 0 -> (N,) or (1, N)
//   B, C     : (B, L, N)   (C already broadcast to N by the caller)
//   D        : empty, or (D,)
//   N        : power of two in [1, 32];  a_per_channel in {0, 1}
//   dtype    : all float32 (is_half = false) or all float16 (is_half = true)
void check_contract(const at::Tensor& u, const at::Tensor& delta, const at::Tensor& A,
                    const at::Tensor& B, const at::Tensor& C, const at::Tensor& D,
                    int64_t a_per_channel, bool is_half) {
    TORCH_CHECK(u.is_cuda(), kOp, "u must be a CUDA tensor");
    TORCH_CHECK(u.dim() == 3, kOp, "u must be (B, L, D), got ", u.sizes());
    TORCH_CHECK(u.size(0) >= 1 && u.size(1) >= 1 && u.size(2) >= 1, kOp,
                "u must be non-empty, got ", u.sizes());
    TORCH_CHECK(u.numel() <= std::numeric_limits<int32_t>::max(), kOp,
                "u has too many elements for 32-bit kernel indexing");
    TORCH_CHECK(u.scalar_type() == at::kFloat || u.scalar_type() == at::kHalf, kOp,
                "u must be float32 or float16, got ", u.scalar_type());
    TORCH_CHECK(is_half == (u.scalar_type() == at::kHalf), kOp,
                "is_half does not match u dtype");
    TORCH_CHECK(a_per_channel == 0 || a_per_channel == 1, kOp,
                "a_per_channel must be 0 or 1, got ", a_per_channel);

    const int64_t Bn = u.size(0), L = u.size(1), Dd = u.size(2);
    check_like_u(delta, u, "delta");
    TORCH_CHECK(delta.sizes() == u.sizes(), kOp, "delta must match u ", u.sizes(),
                ", got ", delta.sizes());

    check_like_u(A, u, "A");
    TORCH_CHECK(A.dim() == 1 || A.dim() == 2, kOp, "A must be 1-D or 2-D, got ", A.sizes());
    const int64_t N = A.size(-1);
    TORCH_CHECK(valid_state_count(N), kOp, "requires power-of-two N in [1, 32], got ", N);
    if (a_per_channel == 1) {
        TORCH_CHECK(A.dim() == 2 && A.size(0) == Dd, kOp,
                    "per-channel A must be (D, N) = (", Dd, ", ", N, "), got ", A.sizes());
    } else {
        TORCH_CHECK(A.numel() == N, kOp, "shared A must be (N,) or (1, N), got ",
                    A.sizes());
    }

    check_like_u(B, u, "B");
    TORCH_CHECK(B.dim() == 3 && B.size(0) == Bn && B.size(1) == L && B.size(2) == N, kOp,
                "B must be (B, L, N) = (", Bn, ", ", L, ", ", N, "), got ", B.sizes());
    check_like_u(C, u, "C");
    TORCH_CHECK(C.dim() == 3 && C.size(0) == Bn && C.size(1) == L && C.size(2) == N, kOp,
                "C must be (B, L, N) = (", Bn, ", ", L, ", ", N, "), got ", C.sizes());

    if (D.numel() > 0) {
        check_like_u(D, u, "D");
        TORCH_CHECK(D.dim() == 1 && D.size(0) == Dd, kOp, "D must be empty or (D,) = (",
                    Dd, ",), got ", D.sizes());
    }
}

at::Tensor selective_scan_fwd_cuda(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    int64_t a_per_channel,
    bool is_half) {
    check_contract(u, delta, A, B, C, D, a_per_channel, is_half);
    const int64_t N = A.size(-1);
    const c10::cuda::CUDAGuard guard(u.device());

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
