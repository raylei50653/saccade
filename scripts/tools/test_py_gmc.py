#!/usr/bin/env python3
"""Pure-Python GMC with PyTorch FFT, designed for CUDA graph capture.

Replaces the C++ cuFFT-based GMC. All operations are PyTorch ops
that support CUDA graph capture (including torch.fft).

Usage test:
    uv run scripts/tools/test_py_gmc.py --frames 30
"""
# status: diagnostic

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))


class PyGraphedGMC:
    """GMC using torch.fft — graph-capturable.

    Usage:
        gmc = PyGraphedGMC(h_orig, w_orig, downscale=4)
        gmc.capture()  # one-time
        warp = gmc(frame_chw)  # per-frame, GPU graph replay
    """

    def __init__(self, h_orig: int, w_orig: int, downscale: int = 4):
        d = "cuda"
        self.h_ds = h_orig // downscale
        self.w_ds = w_orig // downscale
        self.downscale = float(downscale)
        self._ds_hw = (self.h_ds, self.w_ds)
        self._ds_tensor = torch.tensor(
            [float(downscale)], dtype=torch.float32, device=d
        )
        self.prev_gray = torch.zeros(
            self.h_ds, self.w_ds, dtype=torch.float32, device=d
        )

        hh = torch.hann_window(self.h_ds, device=d, dtype=torch.float32)
        hw = torch.hann_window(self.w_ds, device=d, dtype=torch.float32)
        self.hann = (hh.unsqueeze(1) * hw.unsqueeze(0)).contiguous()

        self.warp_out = torch.zeros(6, dtype=torch.float32, device=d)
        self._warp_id = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=d
        )
        self.update_indices = torch.tensor([2, 5], dtype=torch.int64, device=d)
        self.offsets_dy = torch.tensor(
            [-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.int64, device=d
        )
        self.offsets_dx = torch.tensor(
            [-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=torch.int64, device=d
        )
        self._graphed = None
        self._frame_buf = torch.zeros(3, h_orig, w_orig, dtype=torch.float32, device=d)
        self._captured = False

        # RGB→gray weights (CHW)
        self._gray_w = torch.tensor([0.299, 0.587, 0.114], device=d).view(3, 1, 1)

    def _ensure_captured(self):
        if self._captured:
            return
        hds, wds = self.h_ds, self.w_ds
        ds_t = self._ds_tensor
        hann = self.hann
        pg = self.prev_gray
        warp = self.warp_out
        warp_id = self._warp_id

        with torch.no_grad():
            # warmup
            fb = self._frame_buf
            gray_ds = F.interpolate(
                fb.unsqueeze(0) if fb.dim() == 3 else fb,
                size=(hds, wds),
                mode="nearest",
            ).squeeze(0)
            gray_ds = (gray_ds * self._gray_w).sum(dim=0) * hann
            fft_p = torch.fft.rfft2(pg)
            fft_c = torch.fft.rfft2(gray_ds)
            cross = fft_p.conj() * fft_c
            cross = cross / (cross.abs() + 1e-6)
            torch.fft.irfft2(cross, s=(hds, wds))  # warmup (result discarded)
            torch.cuda.synchronize()
            pg.zero_()

            # second warmup
            gray_ds2 = F.interpolate(
                fb.unsqueeze(0) if fb.dim() == 3 else fb,
                size=(hds, wds),
                mode="nearest",
            ).squeeze(0)
            gray_ds2 = (gray_ds2 * self._gray_w).sum(dim=0) * hann
            fft_p2 = torch.fft.rfft2(pg)
            fft_c2 = torch.fft.rfft2(gray_ds2)
            cross2 = fft_p2.conj() * fft_c2
            cross2 = cross2 / (cross2.abs() + 1e-6)
            torch.fft.irfft2(cross2, s=(hds, wds))  # warmup (result discarded)
            torch.cuda.synchronize()

            pg.zero_()

            def _graph_fn(frame_chw, pg_in, warp_out):
                gray_ds = F.interpolate(
                    frame_chw.unsqueeze(0), size=(hds, wds), mode="nearest"
                ).squeeze(0)
                gray_ds = (gray_ds * self._gray_w).sum(dim=0) * hann

                fft_p = torch.fft.rfft2(pg_in)
                fft_c = torch.fft.rfft2(gray_ds)
                cross = fft_p.conj() * fft_c
                cross = cross / (cross.abs() + 1e-6)
                corr_irfft = torch.fft.irfft2(cross, s=(hds, wds))

                # Find peak with sub-pixel accuracy and PCR check
                max_v, max_i = corr_irfft.view(-1).max(dim=0)
                py = max_i // wds
                px = max_i % wds

                n_pixels = hds * wds
                sum_sq = (corr_irfft * corr_irfft).sum()
                rms = torch.sqrt(sum_sq / n_pixels + 1e-12)
                ratio = max_v / (rms + 1e-12)

                # Neighbors coordinates (shape: 9)
                ny = (py + self.offsets_dy + hds) % hds
                nx = (px + self.offsets_dx + wds) % wds

                # 1D indices of neighbors (shape: 9)
                n_indices = ny * wds + nx

                # Gather neighbor values (shape: 9)
                v_all = torch.gather(corr_irfft.view(-1), 0, n_indices)
                v_all = torch.clamp(v_all, min=0.0)

                # Vectorized sum_x, sum_y
                px_all = px.float() + self.offsets_dx.float()
                py_all = py.float() + self.offsets_dy.float()

                sum_v = v_all.sum()
                sum_x = (v_all * px_all).sum()
                sum_y = (v_all * py_all).sum()

                peak_x = sum_x / (sum_v + 1e-6)
                peak_y = sum_y / (sum_v + 1e-6)

                valid = ratio >= 5.0

                px_f = peak_x - torch.where(peak_x > wds / 2.0, float(wds), 0.0)
                py_f = peak_y - torch.where(peak_y > hds / 2.0, float(hds), 0.0)

                warp_out.copy_(warp_id)
                tx = torch.where(valid, px_f * ds_t, 0.0)
                ty = torch.where(valid, py_f * ds_t, 0.0)

                updates = torch.cat([tx.view(1), ty.view(1)])
                warp_out.scatter_(0, self.update_indices, updates)

                pg_in.copy_(gray_ds)

                return warp_out, pg_in

            fb_clone = fb.clone()
            pg_clone = pg.clone()
            warp_clone = warp.clone()
            self._graphed = torch.cuda.make_graphed_callables(
                _graph_fn, (fb_clone, pg_clone, warp_clone)
            )
            pg.zero_()

        self._captured = True
        print(f"🕯️ [PyGMC] Captured graph for img=({hds}×{wds} ds={self.downscale:.0f})")

    def __call__(self, frame_chw: torch.Tensor) -> torch.Tensor:
        self._ensure_captured()
        self._frame_buf.copy_(frame_chw)
        warp_out, pg_out = self._graphed(self._frame_buf, self.prev_gray, self.warp_out)
        self.prev_gray.copy_(pg_out)
        return warp_out


def compare_with_cpp(seq_name: str = "MOT17-02-SDP", n_frames: int = 30, ds: int = 4):
    """Compare PyGraphedGMC with C++ GMC estimate_into_direct."""
    from saccade.perception.eval.evaluator import TorchvisionGpuStreamer
    from saccade_tracking_ext import GMC as CppGMC

    data_root = Path("datasets/MOT17")
    seq_path = data_root / "train" / seq_name
    if not (seq_path / "img1").exists():
        for d in (data_root / "train").iterdir():
            if d.is_dir() and d.name.endswith("-SDP"):
                seq_path = d
                break

    streamer = TorchvisionGpuStreamer(seq_path / "img1")

    frames_chw = []
    for _, frame_hwc in zip(range(n_frames), streamer):
        frames_chw.append(frame_hwc.permute(2, 0, 1).float() / 255.0)
    print(f"Loaded {len(frames_chw)} frames from {seq_path}")

    cpp_gmc = CppGMC(downscale=ds)
    py_gmc = PyGraphedGMC(frames_chw[0].shape[1], frames_chw[0].shape[2], downscale=ds)
    stream = torch.cuda.current_stream().cuda_stream
    d_warp_cpp = torch.zeros(6, device="cuda")
    d_warp_py = torch.zeros(6, device="cuda")

    dx_err_acc = 0.0
    n_valid = 0
    max_err = 0.0

    print(
        f"{'Frame':>6} {'cpp_dx':>10} {'cpp_dy':>10} "
        f"{'py_dx':>10} {'py_dy':>10} {'err':>10}"
    )
    print("-" * 65)

    for i, frame in enumerate(frames_chw):
        h, w = frame.shape[1], frame.shape[2]

        cpp_gmc.estimate_into_direct(
            frame.data_ptr(), w, h, stream, d_warp_cpp.data_ptr()
        )
        torch.cuda.synchronize()

        d_warp_py = py_gmc(frame)
        torch.cuda.synchronize()

        cpp_warp = d_warp_cpp.cpu().tolist()
        py_warp = d_warp_py.cpu().tolist()

        err = abs(cpp_warp[2] - py_warp[2]) + abs(cpp_warp[5] - py_warp[5])
        if i > 0:
            dx_err_acc += err
            n_valid += 1
            max_err = max(max_err, err)

        print(
            f"{i:>6} {cpp_warp[2]:>10.4f} {cpp_warp[5]:>10.4f} "
            f"{py_warp[2]:>10.4f} {py_warp[5]:>10.4f} {err:>10.4f}"
        )

    print("-" * 65)
    if n_valid > 0:
        print(
            f"Mean error: {dx_err_acc / n_valid:.6f}  Max: {max_err:.6f}  Frames: {n_valid}"
        )
    if max_err < 0.5:
        print("✅ PASS: PyGMC matches C++ GMC within tolerance")
        return 0
    else:
        print(f"❌ FAIL: max error {max_err:.4f} exceeds tolerance")
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--seq", default="MOT17-02-SDP")
    args = parser.parse_args()
    return compare_with_cpp(args.seq, args.frames, args.downscale)


if __name__ == "__main__":
    sys.exit(main())
