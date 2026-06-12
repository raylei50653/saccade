#pragma once
#include <cstdint>

struct SelectiveScanParams {
    int B;              // batch size
    int L;              // sequence length
    int D;              // model dimension (channels)
    int N;              // state dimension
    bool has_D;         // whether skip connection D is provided
    bool a_per_channel; // A is (D, N) per-channel; else (1, N) shared
};

void selective_scan_fwd(
    const float* u,     // (B, L, D)
    const float* delta, // (B, L, D)
    const float* A,     // (N,) shared or (D, N) per-channel
    const float* B_ssm, // (B, L, N)
    const float* C_ssm, // (B, L, N)
    const float* D,     // (D,)  or nullptr
    float* y,           // (B, L, D)  output
    const SelectiveScanParams& params,
    void* stream = nullptr
);

void selective_scan_fwd_half(
    const void* u,     // (B, L, D)
    const void* delta, // (B, L, D)
    const void* A,     // (N,)
    const void* B_ssm, // (B, L, N)
    const void* C_ssm, // (B, L, N)
    const void* D,     // (D,)  or nullptr
    void* y,           // (B, L, D)  output
    const SelectiveScanParams& params,
    void* stream = nullptr
);

// Reverse-time gradient of the selective scan (fp32 only — training path).
//
// Given grad_y = dL/dy, computes gradients w.r.t. every input. delta is the
// RAW pre-softplus delta (softplus is applied internally, matching fwd); its
// gradient already folds in the softplus derivative. C must be the full
// (B, L, N) tensor (rank-1 C broadcast to N before calling).
//
// `h_buf` is caller-provided scratch of shape (B, L, D, N): the kernel writes
// the forward hidden states there in a recompute pass, then consumes them in
// the reverse pass.
//
// Accumulating outputs (dA, dB_ssm, dC_ssm, dD) are summed with atomics and so
// MUST be zero-initialised by the caller. Per-(b,t,d) outputs (du, ddelta) are
// fully written and need no pre-zeroing.
void selective_scan_bwd(
    const float* grad_y, // (B, L, D)   upstream gradient dL/dy
    const float* u,      // (B, L, D)
    const float* delta,  // (B, L, D)   raw (pre-softplus)
    const float* A,      // (N,) shared or (D, N) per-channel
    const float* B_ssm,  // (B, L, N)
    const float* C_ssm,  // (B, L, N)   full N (no rank-1 broadcast)
    const float* D,      // (D,) or nullptr
    float* h_buf,        // (B, L, D, N) scratch (forward hidden states)
    float* du,           // (B, L, D)   out
    float* ddelta,       // (B, L, D)   out (w.r.t. raw delta)
    float* dA,           // (N,) or (D, N) out — zero-init
    float* dB_ssm,       // (B, L, N)   out — zero-init
    float* dC_ssm,       // (B, L, N)   out — zero-init
    float* dD,           // (D,) or nullptr out — zero-init
    const SelectiveScanParams& params,
    void* stream = nullptr
);

