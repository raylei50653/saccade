"""Validate the CUDA selective-scan backward against the JIT autograd reference.

Runs the same inputs through `_SelectiveScanCudaFn` (CUDA fwd + analytic CUDA
bwd) and `_selective_scan_jit` (pure-PyTorch, exact autograd), and compares the
forward output plus gradients w.r.t. every input.
"""
# status: experiment

from __future__ import annotations

import torch

from saccade.perception.temporal_yolo.mamba_head import (
    _selective_scan_jit,
    _SelectiveScanCudaFn,
)


def _mk(B, L, D, N, c_rank, per_channel_a, has_D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = D if per_channel_a else 1
    base = dict(device="cuda", dtype=torch.float32)
    u = torch.randn(B, L, D, generator=g, **base)
    delta = torch.randn(B, L, D, generator=g, **base) * 0.5
    # A must be negative (real S4D): A = -exp(A_log)
    A = -torch.rand(rows, N, generator=g, **base) - 0.1
    Bm = torch.randn(B, L, N, generator=g, **base)
    C = torch.randn(B, L, c_rank, generator=g, **base)
    D = torch.randn(D, generator=g, **base) if has_D else None
    return u, delta, A, Bm, C, D


def _clone_leaf(t):
    if t is None:
        return None
    return t.detach().clone().requires_grad_(True)


def run_case(B, L, D, N, c_rank, per_channel_a, has_D):
    tag = (
        f"B={B} L={L} D={D} N={N} c_rank={c_rank} "
        f"per_ch_A={per_channel_a} has_D={has_D}"
    )
    u, delta, A, Bm, C, Dv = _mk(B, L, D, N, c_rank, per_channel_a, has_D)

    # Reference path (JIT autograd)
    ru, rd, rA, rB, rC, rD = (_clone_leaf(t) for t in (u, delta, A, Bm, C, Dv))
    y_ref = _selective_scan_jit(ru, rd, rA, rB, rC, rD)
    # Fixed upstream gradient for reproducibility
    go = torch.randn_like(y_ref)
    y_ref.backward(go)

    # CUDA path
    cu, cd, cA, cB, cC, cD = (_clone_leaf(t) for t in (u, delta, A, Bm, C, Dv))
    y_cuda = _SelectiveScanCudaFn.apply(cu, cd, cA, cB, cC, cD)
    y_cuda.backward(go.clone())

    def cmp(name, a, b):
        if a is None and b is None:
            return True, 0.0
        amax = (a - b).abs().max().item()
        denom = b.abs().max().item() + 1e-6
        rel = amax / denom
        ok = amax < 1e-3 or rel < 1e-3
        flag = "ok " if ok else "FAIL"
        print(f"    {flag} {name:8s} max_abs={amax:.3e} rel={rel:.3e}")
        return ok, amax

    print(f"[{tag}]")
    all_ok = True
    all_ok &= cmp("y", y_cuda, y_ref)[0]
    all_ok &= cmp("d_u", cu.grad, ru.grad)[0]
    all_ok &= cmp("d_delta", cd.grad, rd.grad)[0]
    all_ok &= cmp("d_A", cA.grad, rA.grad)[0]
    all_ok &= cmp("d_B", cB.grad, rB.grad)[0]
    all_ok &= cmp("d_C", cC.grad, rC.grad)[0]
    if has_D:
        all_ok &= cmp("d_D", cD.grad, rD.grad)[0]
    return all_ok


def main():
    torch.manual_seed(0)
    cases = [
        # B, L, D, N, c_rank, per_channel_a, has_D
        (2, 32, 8, 16, 1, False, True),  # production-like: rank-1 C, shared A, D
        (2, 32, 8, 16, 16, False, True),  # full-N C
        (2, 16, 4, 16, 1, True, True),  # per-channel A
        (1, 64, 8, 16, 1, False, False),  # no D
        (3, 33, 6, 16, 16, True, False),  # per-channel A, full C, no D, odd L
    ]
    ok = True
    for c in cases:
        ok &= run_case(*c)
        print()
    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
