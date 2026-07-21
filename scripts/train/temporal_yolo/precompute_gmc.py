"""
Precompute GMC affine matrices for all MOT17 training frames using C++ GPU GMC.

Output: cache_dir/gmc_matrices_cpp.pt -> dict[str, torch.Tensor]
    {seq_name: tensor(N_frames-1, 2, 3)}  (prev->curr 2x3 affine warp)

Uses C++ GPU GMC (phase correlation FFT) which is far more accurate than
sparse optical flow — detects sub-pixel motion on every frame pair.

Usage:
    uv run scripts/train/temporal_yolo/precompute_gmc.py --data-root datasets/MOT17
"""
# status: diagnostic

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "build"))

import saccade_tracking_ext  # noqa — must be before torchvision (libtiff conflict)
from saccade_tracking_ext import GMC
import torchvision.io as tv_io  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="datasets/MOT17")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--cache-dir", default="runs/trt_feat_cache_v2")
    p.add_argument(
        "--detector", default="SDP", help="Only process sequences with this suffix"
    )
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.cache_dir) / "gmc_matrices_cpp.pt"
    img_size = args.img_size
    device = torch.device("cuda")
    stream = torch.cuda.current_stream().cuda_stream

    gmc = GMC(4)  # downscale=4 for sub-pixel accuracy
    out_warp = torch.empty(6, dtype=torch.float32, device=device)

    all_matrices: dict[str, torch.Tensor] = {}
    total_pairs = 0
    total_frames = 0

    seq_dirs = sorted(
        d
        for d in (data_root / "train").iterdir()
        if d.is_dir() and d.name.endswith(f"-{args.detector}")
    )

    for seq_dir in seq_dirs:
        seq = seq_dir.name
        frame_paths = sorted((seq_dir / "img1").glob("*.jpg"))
        if len(frame_paths) < 2:
            continue

        matrices: list[torch.Tensor] = []
        prev_gpu = None
        gmc.reset()

        for fp in frame_paths:
            img = tv_io.read_image(str(fp))
            img_f = (
                F.interpolate(
                    img.unsqueeze(0).float(),
                    size=(img_size, img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .div_(255.0)
                .to(device)
            )

            if prev_gpu is None:
                prev_gpu = img_f
                gmc.estimate_into(
                    prev_gpu.data_ptr(), img_size, img_size, stream, out_warp.data_ptr()
                )
                continue

            gmc.estimate_into(
                img_f.data_ptr(), img_size, img_size, stream, out_warp.data_ptr()
            )
            matrices.append(out_warp.clone().cpu().reshape(2, 3))

            total_pairs += 1
            total_frames += 1
            if total_pairs % 200 == 0:
                print(f"  {total_pairs} frame pairs processed")

        if matrices:
            all_matrices[seq] = torch.stack(matrices)
        print(f"  {seq}: {len(matrices)} pairs")

    torch.save(all_matrices, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved {total_pairs} GMC matrices → {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
