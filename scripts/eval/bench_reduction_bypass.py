#!/usr/bin/env python3
"""Forward-latency: current head (spatial_reduction down/up) vs "no down/up".

Answers "how much slower if we delete downsample+upsample and run the Mamba scan
at full FPN resolution". Builds the head from a checkpoint's mamba_args, feeds
synthetic FPN features at 640-input resolution (P3 80x80 / P4 40x40 / P5 20x20),
and times _forward_eager(T=1, deploy/single-frame) in both modes.

Absolute ms will NOT match the CUDA-graph production figure (this is eager fp32);
the RATIO bypass/production is the answer to the speed question. The bypass path
is gated by head._bypass_reduction (default off, bit-exact when unset).
"""
# status: diagnostic

from __future__ import annotations

import argparse
import time

import torch

from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead


def build_head(
    mamba_args: dict, in_ch: list[int], device: torch.device
) -> MambaDetectionHead:
    return MambaDetectionHead(
        in_channels=in_ch,
        d_model=mamba_args["d_model"],
        d_state=mamba_args["d_state"],
        num_blocks=mamba_args["num_blocks"],
        num_classes=mamba_args.get("num_classes", 1),
        reg_max=1,
        spatial_reduction=mamba_args["spatial_reduction"],
        use_pixel_shuffle=mamba_args["use_pixel_shuffle"],
        use_cross_scan=mamba_args.get("use_cross_scan", False),
        use_temporal_mamba=mamba_args.get("use_temporal_mamba", False),
        per_channel_a=mamba_args.get("per_channel_a", False),
        scan_stop_grad=mamba_args.get("scan_stop_grad", False),
        legacy_n1_scan=mamba_args.get("legacy_n1_scan", False),
        reduction_variant=mamba_args.get("reduction_variant", "conv"),
    ).to(device)


def count_params(head: MambaDetectionHead, *, exclude_updown: bool) -> int:
    total = 0
    for name, p in head.named_parameters():
        if exclude_updown and (
            name.startswith("upsample") or name.startswith("downsample")
        ):
            continue
        total += p.numel()
    return total


@torch.no_grad()
def timed(
    head: MambaDetectionHead, feats, *, bypass: bool, iters: int, warmup: int
) -> float:
    head._bypass_reduction = bypass
    for _ in range(warmup):
        head._forward_eager(feats, T=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        head._forward_eager(feats, T=1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mamba_gt_v14replica_t3_t1/best.ckpt")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    margs = ck["mamba_args"]
    if "in_channels" in margs:
        in_ch = list(margs["in_channels"])
    else:  # older ckpts: derive from input_proj weight shapes (d_model, in_ch, 1, 1)
        sd = ck["student"]
        in_ch = [int(sd[f"input_proj.{i}.weight"].shape[1]) for i in range(3)]
    head = build_head(margs, in_ch, device).eval()

    # FPN feature maps for a square img_size input at strides 8/16/32.
    s = args.img_size
    feats = [
        torch.randn(1, in_ch[0], s // 8, s // 8, device=device),
        torch.randn(1, in_ch[1], s // 16, s // 16, device=device),
        torch.randn(1, in_ch[2], s // 32, s // 32, device=device),
    ]

    print(
        f"ckpt={args.ckpt}  d_model={margs['d_model']}  "
        f"spatial_reduction={margs['spatial_reduction']}  in_ch={in_ch}"
    )
    print(f"feat sizes: {[tuple(f.shape) for f in feats]}\n")

    prod_ms = timed(head, feats, bypass=False, iters=args.iters, warmup=args.warmup)
    byp_ms = timed(head, feats, bypass=True, iters=args.iters, warmup=args.warmup)

    p_full = count_params(head, exclude_updown=False)
    p_lean = count_params(head, exclude_updown=True)

    def fps(ms: float) -> float:
        return 1e3 / ms

    print("                         head-only ms     head-only FPS     params")
    print(
        f"  production (down/up)   {prod_ms:9.3f}      {fps(prod_ms):9.1f}     "
        f"{p_full / 1e6:6.2f}M"
    )
    print(
        f"  bypass (no down/up)    {byp_ms:9.3f}      {fps(byp_ms):9.1f}     "
        f"{p_lean / 1e6:6.2f}M  (down/up deleted)"
    )
    print(
        f"\n  bypass / production  = {byp_ms / prod_ms:.2f}x  "
        f"(head-only; eager fp32 — ratio is the answer, not absolute)"
    )
    print(
        f"  params lean / full   = {p_lean / p_full:.2f}x  "
        f"({(p_full - p_lean) / 1e6:.2f}M removed)"
    )


if __name__ == "__main__":
    main()
