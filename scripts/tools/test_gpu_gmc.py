"""Manual GPU GMC correctness/perf smoke test."""

# status: diagnostic
import torch
import numpy as np
import cv2
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

try:
    from saccade_tracking_ext import GMC as CppGMC
except ImportError:
    print("Cannot import saccade_tracking_ext. Build the project first.")
    sys.exit(1)


def make_frames(dx_true: float, dy_true: float, w: int = 1920, h: int = 1080):
    device = torch.device("cuda")
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), -1)
    M = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
    img2 = cv2.warpAffine(img, M, (w, h))
    f1 = torch.from_numpy(img).permute(2, 0, 1).float().to(device).contiguous()
    f2 = torch.from_numpy(img2).permute(2, 0, 1).float().to(device).contiguous()
    return f1, f2, w, h


def test_estimate_basic():
    """estimate() returns correct translation."""
    f1, f2, w, h = make_frames(16.0, -8.0)
    gmc = CppGMC(downscale=8)
    stream = torch.cuda.current_stream().cuda_stream
    gmc.estimate(f1.data_ptr(), w, h, stream, True)
    warp = gmc.estimate(f2.data_ptr(), w, h, stream, True)
    assert warp is not None, "estimate() returned None"
    dx_est, dy_est = warp[2], warp[5]
    print(f"  [estimate] true=(16,-8)  est=({dx_est:.3f},{dy_est:.3f})")
    assert abs(dx_est - 16.0) < 2.0, f"dx error too large: {dx_est}"
    assert abs(dy_est - (-8.0)) < 2.0, f"dy error too large: {dy_est}"
    print("  PASS")


def test_estimate_into_matches_estimate():
    """estimate_into() warp matches estimate() for same frames."""
    f1, f2, w, h = make_frames(16.0, -8.0)
    stream = torch.cuda.current_stream().cuda_stream

    # Reference via estimate()
    gmc_ref = CppGMC(downscale=8)
    gmc_ref.estimate(f1.data_ptr(), w, h, stream, True)
    warp_ref = gmc_ref.estimate(f2.data_ptr(), w, h, stream, True)
    assert warp_ref is not None

    # estimate_into()
    gmc_into = CppGMC(downscale=8)
    d_warp = torch.zeros(6, dtype=torch.float32, device="cuda")
    gmc_into.estimate(f1.data_ptr(), w, h, stream, True)
    gmc_into.estimate_into(f1.data_ptr(), w, h, stream, d_warp.data_ptr(), True)
    gmc_into.reset()
    gmc_into.estimate(f1.data_ptr(), w, h, stream, True)
    gmc_into.estimate_into(f2.data_ptr(), w, h, stream, d_warp.data_ptr(), True)
    torch.cuda.synchronize()

    warp_into = d_warp.cpu().tolist()
    dx_ref, dy_ref = warp_ref[2], warp_ref[5]
    dx_into, dy_into = warp_into[2], warp_into[5]
    print(f"  [estimate]      dx={dx_ref:.3f}  dy={dy_ref:.3f}")
    print(f"  [estimate_into] dx={dx_into:.3f}  dy={dy_into:.3f}")
    assert abs(dx_into - dx_ref) < 0.5, f"dx mismatch: {dx_into} vs {dx_ref}"
    assert abs(dy_into - dy_ref) < 0.5, f"dy mismatch: {dy_into} vs {dy_ref}"
    print("  PASS")


def test_profiling_sub_stages():
    """With profiling enabled, sub-stage fields are populated and sum to phase_corr_ms."""
    f1, f2, w, h = make_frames(16.0, -8.0)
    stream = torch.cuda.current_stream().cuda_stream
    d_warp = torch.zeros(6, dtype=torch.float32, device="cuda")

    gmc = CppGMC(downscale=8)
    gmc.set_profiling_enabled(True)

    # Warm-up
    gmc.estimate_into(f1.data_ptr(), w, h, stream, d_warp.data_ptr(), True)
    torch.cuda.synchronize()

    gmc.reset_profile_stats()
    gmc.estimate_into(f2.data_ptr(), w, h, stream, d_warp.data_ptr(), True)
    torch.cuda.synchronize()

    stats = gmc.get_profile_stats()
    print(f"  gray_downscale: {stats['gray_downscale_ms']:.3f} ms")
    print(f"  fft:            {stats['fft_ms']:.3f} ms")
    print(f"  cross_power:    {stats['cross_power_ms']:.3f} ms")
    print(f"  ifft:           {stats['ifft_ms']:.3f} ms")
    print(f"  peak_find:      {stats['peak_find_ms']:.3f} ms")
    print(f"  phase_corr:     {stats['phase_corr_ms']:.3f} ms  (sum of sub-stages)")
    print(f"  handoff:        {stats['handoff_ms']:.3f} ms")
    print(f"  total:          {stats['total_ms']:.3f} ms")

    sub_sum = (
        stats["fft_ms"]
        + stats["cross_power_ms"]
        + stats["ifft_ms"]
        + stats["peak_find_ms"]
    )
    assert abs(sub_sum - stats["phase_corr_ms"]) < 0.01, (
        f"sub-stage sum {sub_sum:.4f} != phase_corr_ms {stats['phase_corr_ms']:.4f}"
    )

    for key in ("fft_ms", "cross_power_ms", "ifft_ms", "peak_find_ms"):
        assert stats[key] > 0.0, f"{key} should be > 0 when profiling enabled"

    print("  PASS")


def test_identity_on_first_frame():
    """estimate_into() writes identity warp on the first call (no prev frame)."""
    f1, _, w, h = make_frames(0.0, 0.0)
    stream = torch.cuda.current_stream().cuda_stream
    d_warp = torch.zeros(6, dtype=torch.float32, device="cuda")

    gmc = CppGMC(downscale=8)
    gmc.estimate_into(f1.data_ptr(), w, h, stream, d_warp.data_ptr(), True)
    torch.cuda.synchronize()

    warp = d_warp.cpu().tolist()
    # First frame: prev is zeroed → PCR will be low → identity written
    # (tx/ty == 0 is sufficient here)
    assert abs(warp[2]) < 1.0 and abs(warp[5]) < 1.0, (
        f"First-frame warp should be near identity, got tx={warp[2]}, ty={warp[5]}"
    )
    print(f"  warp={[f'{v:.3f}' for v in warp]}")
    print("  PASS")


if __name__ == "__main__":
    tests = [
        ("estimate basic", test_estimate_basic),
        ("estimate_into vs estimate", test_estimate_into_matches_estimate),
        ("profiling sub-stages", test_profiling_sub_stages),
        ("identity on first frame", test_identity_on_first_frame),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(name)
    print(f"\n{'=' * 50}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
