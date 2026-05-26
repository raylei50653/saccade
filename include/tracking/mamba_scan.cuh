#pragma once
#include <cstdint>

struct SelectiveScanParams {
    int B;       // batch size
    int L;       // sequence length
    int D;       // model dimension (channels)
    int N;       // state dimension
    bool has_D;  // whether skip connection D is provided
};

void selective_scan_fwd(
    const float* u,     // (B, L, D)
    const float* delta, // (B, L, D)
    const float* A,     // (N,)
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

