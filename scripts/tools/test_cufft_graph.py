#!/usr/bin/env python3
"""Minimal cuFFT CUDA graph capture test.

Verifies whether cuFFT R2C → C2R roundtrip survives graph capture/replay.
If this fails, cuFFT graph capture is fundamentally broken in current CUDA.
If this passes, the GMC graph-capture failure has a different root cause.

Usage:
    uv run scripts/tools/test_cufft_graph.py [--size W] [--size H]
"""

import argparse
import sys

import numpy as np
import torch


def test_cufft_graph_capture(h: int = 270, w: int = 480):
    torch.cuda.synchronize()

    rng = np.random.RandomState(42)
    data = rng.randn(h, w).astype(np.float32)

    x = torch.from_numpy(data).cuda()
    x = x.contiguous()

    out_ref = torch.fft.irfft2(torch.fft.rfft2(x), s=(h, w))
    torch.cuda.synchronize()

    x1 = x.clone()

    def rfft_r2c(inp):
        return torch.fft.rfft2(inp)

    def irfft_c2r(complex_in):
        return torch.fft.irfft2(complex_in, s=(h, w))

    complex_tmp = rfft_r2c(x1)
    _ = irfft_c2r(complex_tmp)
    torch.cuda.synchronize()

    x_graph = x.clone()

    def graph_fn(inp):
        c = torch.fft.rfft2(inp)
        return torch.fft.irfft2(c, s=(h, w))

    graphed = torch.cuda.make_graphed_callables(graph_fn, (x_graph.clone(),))
    torch.cuda.synchronize()

    result = graphed(x_graph)
    torch.cuda.synchronize()

    diff = (result - out_ref).abs().max().item()
    expected = out_ref.abs().max().item()
    rel_err = diff / max(expected, 1e-6)

    print(f"  Input shape:      ({h}, {w})")
    print(f"  Ref max:           {expected:.6f}")
    print(f"  Graph max diff:    {diff:.6f}")
    print(f"  Relative error:    {rel_err:.6e}")

    if rel_err < 1e-4:
        print("  ✅ PASS: torch.fft graph capture works correctly")
        return 0
    elif rel_err < 1e-2:
        print("  ⚠️  TOLERABLE: small numerical differences within 1%")
        return 0
    else:
        print(
            f"  ❌ FAIL: graph replay produces significantly different output (rel_err={rel_err:.4f})"
        )
        # Show a snippet
        ref_flat = out_ref.flatten()[:10]
        graph_flat = result.flatten()[:10]
        print(f"  Ref[:10]:   {ref_flat.cpu().tolist()}")
        print(f"  Graph[:10]: {graph_flat.cpu().tolist()}")
        return 1


def test_cufft_stream_capture(h: int = 270, w: int = 480):
    """Test cuFFT on a dedicated stream with graph capture."""
    import saccade_tracking_ext

    stream = torch.cuda.Stream()
    stream_ptr = stream.cuda_stream

    rng = np.random.RandomState(42)
    data = rng.randn(3, h, w).astype(np.float32)

    d_input = torch.from_numpy(data).cuda().contiguous()

    gmc = saccade_tracking_ext.GMC(downscale=1)

    # First do a regular (non-graph) call to measure baseline
    d_warp = torch.zeros(6, dtype=torch.float32, device="cuda")
    with torch.cuda.stream(stream):
        gmc.estimate_into_direct(
            d_input.data_ptr(), w, h, stream_ptr, d_warp.data_ptr()
        )
    torch.cuda.synchronize()

    print(f"  Non-graph warp: {d_warp.cpu().tolist()}")

    # Now try to capture a graph with cuFFT calls
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            with torch.cuda.graph(g):
                gmc.estimate_into_direct(
                    d_input.data_ptr(), w, h, stream_ptr, d_warp.data_ptr()
                )

        d_warp.zero_()
        with torch.cuda.stream(stream):
            g.replay()
        torch.cuda.synchronize()

        print(f"  Graph-replay warp: {d_warp.cpu().tolist()}")
        print("  ✅ CUDAGraph context manager works with estimate_into_direct")
        return 0
    except Exception as e:
        print(f"  ❌ CUDAGraph failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    args = parser.parse_args()

    w = args.width
    h = args.height

    print("=" * 60)
    print("Test 1: torch.fft graph capture")
    print("=" * 60)
    rc1 = test_cufft_graph_capture(h, w)

    print()
    print("=" * 60)
    print("Test 2: CUDAGraph context manager with estimate_into_direct")
    print("=" * 60)
    rc2 = test_cufft_stream_capture(h, w)

    print()
    if rc1 == 0 and rc2 == 0:
        print("✅ All tests passed")
    else:
        print("❌ Some tests failed")
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())
