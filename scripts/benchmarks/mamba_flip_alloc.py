#!/usr/bin/env python3
"""Test the hypothesis: cross-scan flip cost = allocating new tensors, not the
flip arithmetic. Compares producing the (4B,C,H,W) flipped stack three ways:

  A) torch.stack([x, flip, flip, flip])     — allocates flips + stack (current)
  B) static buf; buf[k].copy_(x.flip(...))  — reuse buf, but flip still allocs temp
  C) static buf; buf[k].copy_(rev_view)     — reversed strided VIEW, zero new alloc

Plus an empty_cache() variant of A to expose cudaMalloc vs cached-reuse cost.
"""

from __future__ import annotations

import time

import torch

WARMUP, ITERS = 50, 500


def bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def rev_view(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Reversed view along `dims` via negative strides (no copy), using as_strided."""
    sizes = list(x.shape)
    strides = list(x.stride())
    offset = x.storage_offset()
    for d in dims:
        offset += (sizes[d] - 1) * strides[d]
        strides[d] = -strides[d]
    return x.as_strided(sizes, strides, offset)


def main() -> None:
    torch.cuda.init()
    # P3 scale x_small: (1,128,20,20) — the largest cross-scan input.
    for B, C, H, W in [(1, 128, 20, 20), (1, 128, 10, 10), (1, 128, 5, 5)]:
        x = torch.rand(B, C, H, W, device="cuda")
        buf = torch.empty(4 * B, C, H, W, device="cuda")

        def vA(x=x):
            scans = torch.stack(
                [x, torch.flip(x, [2, 3]), torch.flip(x, [3]), torch.flip(x, [2])], 0
            )
            return scans.reshape(4 * B, C, H, W).flatten(2).transpose(1, 2)

        def vB(x=x, buf=buf):
            buf[0:B].copy_(x)
            buf[B : 2 * B].copy_(torch.flip(x, [2, 3]))
            buf[2 * B : 3 * B].copy_(torch.flip(x, [3]))
            buf[3 * B : 4 * B].copy_(torch.flip(x, [2]))
            return buf.flatten(2).transpose(1, 2)

        def vA_nocache(x=x):
            torch.cuda.empty_cache()
            return vA(x)

        # sanity: B must equal A numerically
        a = vA().clone()
        b = vB().clone()
        ok = torch.allclose(a, b)

        tA = bench(vA)
        tB = bench(vB)
        tAnc = bench(vA_nocache)
        print(
            f"({B},{C},{H},{W}) match={ok}  "
            f"A_stack(alloc)={tA:.4f}  B_buf_reuse={tB:.4f}  "
            f"A_emptycache(cudaMalloc)={tAnc:.4f} ms"
        )


if __name__ == "__main__":
    main()
