"""
Mamba head training on TRT backbone features.

Two phases (auto-selected):
    Phase 1: --from-scratch → MSE distillation (teacher Detect head targets), stable init
    Phase 2: --resume <ckpt> → v8DetectionLoss (GT supervision), detection fine-tuning

Architecture:
    TRT Backbone → P3/P4/P5 → MambaDetectionHead → cls/reg → loss

Usage:
    # Phase 1: distillation from scratch
    uv run scripts/train/temporal_yolo/train_mamba_trt.py \
        --data-root datasets/MOT17 --from-scratch \
        --batch-size 32 --lr 1e-3 --epochs 20

    # Phase 2: GT fine-tuning
    uv run scripts/train/temporal_yolo/train_mamba_trt.py \
        --data-root datasets/MOT17 \
        --resume runs/mamba_trt_scratch/best.ckpt \
        --batch-size 32 --lr 1e-5 --epochs 20
"""
# status: archive-candidate

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "build"))
import saccade_tracking_ext  # noqa

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader  # noqa
from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead  # noqa
from saccade.perception.temporal_yolo.mamba_gated_detector import TRTYoloBackbone  # noqa
from saccade.perception.temporal_yolo.yolo_gated_detector import (
    GatedDetConfig,
    build_gated_yolo_detector,
)  # noqa
from ultralytics.utils.loss import v8DetectionLoss  # noqa


# ── helpers ──
def _xyxy_to_cxcywh_norm(b, sz):
    if b.numel() == 0:
        return b.new_zeros((0, 4))
    cx, cy = (b[:, 0] + b[:, 2]) / 2 / sz, (b[:, 1] + b[:, 3]) / 2 / sz
    w, h = (b[:, 2] - b[:, 0]) / sz, (b[:, 3] - b[:, 1]) / sz
    return torch.stack([cx, cy, w, h], dim=1)


def _make_yolo_batch(gts, sz, dev):
    bi, cl, bb = [], [], []
    for i, bx in enumerate(gts):
        if bx.numel() == 0:
            continue
        n = bx.shape[0]
        bi.append(torch.full((n,), float(i)))
        cl.append(torch.zeros(n))
        bb.append(_xyxy_to_cxcywh_norm(bx, sz))
    if not bi:
        return {
            "batch_idx": torch.zeros(0, device=dev),
            "cls": torch.zeros(0, device=dev),
            "bboxes": torch.zeros(0, 4, device=dev),
        }
    return {
        "batch_idx": torch.cat(bi).to(dev),
        "cls": torch.cat(cl).to(dev),
        "bboxes": torch.cat(bb).to(dev),
    }


def _build_preds(c, r, f):
    return {
        "boxes": torch.cat([x.flatten(2) for x in r], dim=2),
        "scores": torch.cat([x.flatten(2) for x in c], dim=2),
        "feats": f,
    }


def _save(ck, d, e, best=False):
    d.mkdir(parents=True, exist_ok=True)
    torch.save(ck, d / "latest.ckpt")
    torch.save(ck, d / f"epoch_{e:04d}.ckpt")
    if best:
        torch.save(ck, d / "best.ckpt")
    print(f"  Saved epoch_{e:04d}.ckpt" + (" [BEST]" if best else ""))


def _strip(s):
    return {k.replace("._orig_mod.", "."): v for k, v in s.items()}


# ── main ──
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument(
        "--trt-engine", default="models/yolo/yolo26s_backbone_640_batch32.engine"
    )
    p.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    p.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    p.add_argument("--run-dir", default="runs/mamba_trt")
    p.add_argument("--resume", default="")
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--clip-len", type=int, default=4)
    p.add_argument("--spatial-reduction", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--seqs", default="")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--clip-grad", type=float, default=0.1)
    p.add_argument(
        "--accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── TRT backbone ──
    epath = _root / args.trt_engine
    if not epath.exists():
        raise FileNotFoundError(str(epath))
    print(f"TRT: {epath}")
    trunk = TRTYoloBackbone(str(epath))

    # ── teacher (for Detect head) ──
    teacher = build_gated_yolo_detector(
        str(_root / args.yolo_weights),
        cfg=GatedDetConfig(
            scales=("p3", "p4", "p5"), freeze_backbone=True, img_size=args.img_size
        ),
        device=dev,
        weights_path=str(_root / args.teacher_ckpt),
    )
    teacher.eval()
    for x in teacher.parameters():
        x.requires_grad_(False)
    nc = teacher.yolo_model.model[-1].nc
    detect = teacher.yolo_model.model[-1]

    # ── student ──
    if args.from_scratch or not args.resume:
        ma = {
            "d_model": args.d_model,
            "d_state": args.d_state,
            "num_blocks": args.num_blocks,
            "spatial_reduction": args.spatial_reduction,
            "reduction_variant": "conv",
            "num_classes": nc,
        }
        mamba = MambaDetectionHead(
            (128, 256, 512),
            ma["d_model"],
            ma["d_state"],
            ma["num_blocks"],
            nc,
            1,
            ma["spatial_reduction"],
            reduction_variant=ma.get("reduction_variant", "conv"),
        ).to(dev)
        phase1 = False
        print(f"[Phase 2 scratch] GT supervision  LR={args.lr}  B={args.batch_size}")
    else:
        ckpt = torch.load(_root / args.resume, map_location="cpu", weights_only=False)
        ma = ckpt.get("mamba_args", {})
        for k, v in {
            "d_model": args.d_model,
            "d_state": args.d_state,
            "num_blocks": args.num_blocks,
            "spatial_reduction": args.spatial_reduction,
            "reduction_variant": "conv",
            "num_classes": nc,
        }.items():
            ma.setdefault(k, v)
        mamba = MambaDetectionHead(
            (128, 256, 512),
            ma["d_model"],
            ma["d_state"],
            ma["num_blocks"],
            nc,
            1,
            ma["spatial_reduction"],
            reduction_variant=ma.get("reduction_variant", "conv"),
        ).to(dev)
        mamba.load_state_dict(_strip(ckpt["student"]), strict=False)
        phase1 = False
        print(f"[Phase 2] GT fine-tune  {args.resume}  LR={args.lr}")
    mamba.train()
    print(f"Mamba: {sum(p.numel() for p in mamba.parameters()):,} params")

    # ── Phase 2 loss (only for GT supervision) ──
    base_args = (
        dict(teacher.yolo_model.args)
        if isinstance(teacher.yolo_model.args, dict)
        else {}
    )
    base_args.update({"box": 7.5, "cls": 0.5, "dfl": 1.5})
    teacher.yolo_model.args = SimpleNamespace(**base_args)
    criterion = v8DetectionLoss(teacher.yolo_model)

    # ── optimizer ──
    opt = torch.optim.AdamW(mamba.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    wu: Any = None
    if args.warmup_epochs > 0:
        wu = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda e: (e + 1) / args.warmup_epochs if e < args.warmup_epochs else 1.0,
        )

    best = float("inf")
    seqs = args.seqs.split(",") if args.seqs else None
    loader = build_mot17_dataloader(
        _root / args.data_root,
        args.clip_len,
        args.img_size,
        args.batch_size,
        args.clip_len * 2,
        True,
        seqs,
    )
    N = len(loader)
    print(
        f"Batches: {N}  Mode: {'distil' if phase1 else 'GT'}  B={args.batch_size} T={args.clip_len} accum={args.accum_steps}\n"
    )

    # ── loop ──
    for ep in range(1, args.epochs + 1):
        total, st = 0.0, 0
        t0 = time.perf_counter()
        acc = 0
        opt.zero_grad()

        for i, batch in enumerate(loader):
            frames = batch["frames"].to(dev, dtype=torch.float32) / 255.0
            gt = batch["gt_boxes"]
            B, T_f = frames.shape[:2]

            # Only print every N batches for large batch counts
            verbose = ((i + 1) % max(1, min(50, N // 10)) == 0) or (i == N - 1)

            loss = frames.new_zeros(())
            for t in range(T_f):
                ft = frames[:, t]
                gt_t = [gt[b][t] for b in range(B)]
                p3, p4, p5 = trunk.infer(ft)
                feats = [p3, p4, p5]

                cls, reg = mamba(feats)

                if phase1:
                    with torch.no_grad():
                        t_cls = [detect.cv3[si](feats[si]) for si in range(len(feats))]
                        t_reg = [detect.cv2[si](feats[si]) for si in range(len(feats))]
                    for si in range(len(feats)):
                        loss = (
                            loss
                            + F.mse_loss(cls[si], t_cls[si])
                            + F.mse_loss(reg[si], t_reg[si])
                        )
                else:
                    preds = _build_preds(cls, reg, feats)
                    sl, _ = criterion(preds, _make_yolo_batch(gt_t, args.img_size, dev))
                    loss = loss + sl.sum()

            loss = loss / (T_f * B * args.accum_steps)
            loss.backward()
            acc += 1

            if acc == args.accum_steps:
                nn.utils.clip_grad_norm_(mamba.parameters(), args.clip_grad)
                opt.step()
                opt.zero_grad()
                acc = 0

            v = loss.item() * args.accum_steps
            if torch.isfinite(loss.detach()):
                total += v
                st += 1

            if verbose:
                eta = (N - i - 1) * (time.perf_counter() - t0) / max(i + 1, 1)
                print(
                    f"\r  e{ep:3d} [{i + 1:4d}/{N}] loss={v:10.4f}  ETA {eta:.0f}s",
                    end="",
                    flush=True,
                )

        if acc > 0:
            nn.utils.clip_grad_norm_(mamba.parameters(), args.clip_grad)
            opt.step()
            opt.zero_grad()

        sched.step()
        if wu is not None and ep <= args.warmup_epochs:
            wu.step()
        avg = total / max(st, 1)
        dt = time.perf_counter() - t0
        is_best = avg < best
        best = min(best, avg)
        lr = opt.param_groups[0]["lr"]
        print(
            f"\r  e{ep:3d} [{N}/{N}] loss={avg:10.4f}  lr={lr:.2e}  {dt / 60:.1f}min"
            + (" [BEST]" if is_best else "")
        )

        if ep % args.save_every == 0 or ep == args.epochs or is_best:
            _save(
                {
                    "epoch": ep,
                    "student": mamba.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_loss": best,
                    "args": vars(args),
                    "mamba_args": ma,
                },
                run_dir,
                ep,
                is_best,
            )

    print(f"Done. Best: {best:.4f}")


import torch.nn.functional as F  # noqa: E402

if __name__ == "__main__":
    main()
