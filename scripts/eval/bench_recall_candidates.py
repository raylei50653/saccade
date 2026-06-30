#!/usr/bin/env python3
"""Head-only latency probes for recall-recovery architecture candidates.

This benchmark answers a narrow question before committing to training:

* can we change the reduction path to preserve more small-object phase/detail
  without paying the full-res Mamba scan cost?
* how expensive is the P3-only sr=2 compromise versus changing every scale?

It uses synthetic FPN tensors at the deploy shapes, so the numbers are latency
and parameter evidence only. Accuracy still requires training/eval.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from saccade.perception.temporal_yolo.mamba_head import (
    MambaDetectionHead,
    WaveletReduction,
)


class BlurConvReduction(nn.Module):
    """Fixed 3x3 low-pass prefilter followed by the normal strided conv."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]) / 16.0
        self.register_buffer(
            "kernel",
            kernel_2d.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1),
        )
        self.conv = nn.Conv2d(channels, channels, reduction, stride=reduction)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        x = F.conv2d(x, self.kernel.to(dtype=x.dtype), groups=x.shape[1])
        return self.conv(x)


class SpaceToDepthReduction(nn.Module):
    """Lossless phase packing followed by a 1x1 projection back to d_model."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.reduction = reduction
        self.proj = nn.Conv2d(channels * reduction * reduction, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.pixel_unshuffle(x, self.reduction))


class PixelShuffleUpsample(nn.Module):
    """Learned upsample with an explicit per-level factor."""

    def __init__(self, channels: int, factor: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * factor * factor, 3, padding=1),
            nn.PixelShuffle(factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _checkpoint_in_channels(checkpoint: dict) -> list[int]:
    margs = checkpoint["mamba_args"]
    if "in_channels" in margs:
        return [int(v) for v in margs["in_channels"]]
    sd = checkpoint["student"]
    return [int(sd[f"input_proj.{i}.weight"].shape[1]) for i in range(3)]


def build_head(
    mamba_args: dict,
    in_ch: list[int],
    device: torch.device,
    *,
    spatial_reduction: int | None = None,
) -> MambaDetectionHead:
    sr = int(spatial_reduction or mamba_args["spatial_reduction"])
    return MambaDetectionHead(
        in_channels=tuple(in_ch),
        d_model=int(mamba_args["d_model"]),
        d_state=int(mamba_args["d_state"]),
        num_blocks=int(mamba_args["num_blocks"]),
        num_classes=int(mamba_args.get("num_classes", 1)),
        reg_max=1,
        spatial_reduction=sr,
        use_pixel_shuffle=bool(mamba_args.get("use_pixel_shuffle", False)),
        use_cross_scan=bool(mamba_args.get("use_cross_scan", False)),
        use_temporal_mamba=bool(mamba_args.get("use_temporal_mamba", False)),
        use_temporal_attention=bool(mamba_args.get("use_temporal_attention", False)),
        per_channel_a=bool(mamba_args.get("per_channel_a", False)),
        scan_stop_grad=bool(mamba_args.get("scan_stop_grad", False)),
        legacy_n1_scan=bool(mamba_args.get("legacy_n1_scan", False)),
        reduction_variant=mamba_args.get("reduction_variant", "conv"),
    ).to(device)


def apply_variant(head: MambaDetectionHead, name: str) -> bool:
    """Mutate a freshly-built head. Return whether to bypass down/up."""

    d_model = int(head.d_model)
    sr = int(head.spatial_reduction)
    if name == "baseline_sr4":
        return False
    if name == "blur_conv_sr4":
        head.downsample = nn.ModuleList(
            [
                BlurConvReduction(d_model, sr).to(next(head.parameters()).device)
                for _ in range(head.nl)
            ]
        )
        return False
    if name == "space_to_depth_sr4":
        head.downsample = nn.ModuleList(
            [
                SpaceToDepthReduction(d_model, sr).to(next(head.parameters()).device)
                for _ in range(head.nl)
            ]
        )
        return False
    if name == "wavelet_sr4":
        head.downsample = nn.ModuleList(
            [
                WaveletReduction(d_model, sr).to(next(head.parameters()).device)
                for _ in range(head.nl)
            ]
        )
        return False
    if name == "p3_sr2":
        device = next(head.parameters()).device
        head.downsample[0] = nn.Conv2d(d_model, d_model, 2, stride=2).to(device)
        if head.use_pixel_shuffle and head.upsample is not None:
            head.upsample[0] = PixelShuffleUpsample(d_model, 2).to(device)
            head.upsample_loaded = True
        return False
    if name == "p3_space_to_depth_sr2":
        device = next(head.parameters()).device
        head.downsample[0] = SpaceToDepthReduction(d_model, 2).to(device)
        if head.use_pixel_shuffle and head.upsample is not None:
            head.upsample[0] = PixelShuffleUpsample(d_model, 2).to(device)
            head.upsample_loaded = True
        return False
    if name == "no_downup_full_scan":
        return True
    raise ValueError(f"unknown variant: {name}")


def count_params(head: MambaDetectionHead) -> int:
    return sum(p.numel() for p in head.parameters())


@torch.no_grad()
def scan_grids(
    head: MambaDetectionHead, feats: list[Tensor], *, bypass: bool
) -> list[list[int]]:
    grids: list[list[int]] = []
    for i, feat in enumerate(feats):
        x_proj = head.input_proj[i](feat)
        x_small = x_proj if bypass else head.downsample[i](x_proj)
        grids.append([int(x_small.shape[-2]), int(x_small.shape[-1])])
    return grids


@torch.no_grad()
def timed(
    head: MambaDetectionHead,
    feats: list[Tensor],
    *,
    bypass: bool,
    iters: int,
    warmup: int,
) -> float:
    head._bypass_reduction = bypass
    for _ in range(warmup):
        head._forward_eager(feats, T=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        head._forward_eager(feats, T=1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/mamba_gt_vgt_mamba_v14/best.ckpt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline_sr4",
            "blur_conv_sr4",
            "space_to_depth_sr4",
            "wavelet_sr4",
            "p3_sr2",
            "p3_space_to_depth_sr2",
            "all_sr2",
            "no_downup_full_scan",
        ],
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    margs = checkpoint["mamba_args"]
    in_ch = _checkpoint_in_channels(checkpoint)
    feats = [
        torch.randn(1, in_ch[0], args.img_size // 8, args.img_size // 8, device=device),
        torch.randn(
            1, in_ch[1], args.img_size // 16, args.img_size // 16, device=device
        ),
        torch.randn(
            1, in_ch[2], args.img_size // 32, args.img_size // 32, device=device
        ),
    ]

    print(
        f"ckpt={args.ckpt} d_model={margs['d_model']} "
        f"base_sr={margs['spatial_reduction']} in_ch={in_ch}"
    )
    print(f"feat sizes: {[tuple(f.shape) for f in feats]}\n")

    rows: list[dict] = []
    baseline_ms: float | None = None
    for variant in args.variants:
        if variant == "all_sr2":
            head = build_head(margs, in_ch, device, spatial_reduction=2).eval()
            bypass = False
        else:
            head = build_head(margs, in_ch, device).eval()
            bypass = apply_variant(head, variant)
        grids = scan_grids(head, feats, bypass=bypass)
        ms = timed(head, feats, bypass=bypass, iters=args.iters, warmup=args.warmup)
        if baseline_ms is None and variant == "baseline_sr4":
            baseline_ms = ms
        rel = ms / baseline_ms if baseline_ms else 1.0
        row = {
            "variant": variant,
            "ms": ms,
            "fps": 1000.0 / ms,
            "relative_to_baseline": rel,
            "params": count_params(head),
            "scan_grids": grids,
            "scan_tokens": [h * w for h, w in grids],
        }
        rows.append(row)
        print(
            f"{variant:<24} {ms:8.3f} ms  {1000.0 / ms:8.1f} FPS  "
            f"{rel:5.2f}x  params={row['params'] / 1e6:6.2f}M  grids={grids}"
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
