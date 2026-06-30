"""
Train Mamba head from cached TRT backbone features (Phase 2 GT supervision).

Reads pre-cached FP16 features from cache_trt_feats.py output, avoids
TRT backbone inference entirely during training.

Usage:
    # Step 1: cache features (once)
    uv run scripts/train/temporal_yolo/cache_trt_feats.py --cache-dir runs/trt_feat_cache

    # Step 2: train from cached features
    uv run scripts/train/temporal_yolo/train_mamba_cached.py \
        --data-root datasets/MOT17 \
        --cache-dir runs/trt_feat_cache \
        --from-scratch --batch-size 8 --lr 1e-4 --epochs 30
"""

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

from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead  # noqa
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


# ── Cached dataset ──
class CachedFeatDataset(torch.utils.data.Dataset):
    """Returns (features_list, gt_boxes_list) for each clip.

    features_list: list of (p3, p4, p5) tuples, one per frame in the clip.
    gt_boxes_list: list of gt box tensors, one per frame.
    """

    def __init__(
        self,
        cache_dir: Path,
        data_root: Path,
        seq_names: list[str],
        clip_len: int = 4,
        stride: int = 8,
        img_size: int = 640,
    ):
        self.img_size = img_size
        self.clip_len = clip_len
        self.stride = stride

        print(f"Loading cached features from {cache_dir} ...")
        self._feat_cache: dict[str, dict[int, tuple]] = {}
        self._gt_cache: dict[str, dict[int, torch.Tensor]] = {}

        for sname in seq_names:
            cache_path = cache_dir / f"{sname}.pt"
            if not cache_path.exists():
                print(f"  [Warn] No cache for {sname}, skipping")
                continue
            self._feat_cache[sname] = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
            self._gt_cache[sname] = self._load_gt(
                data_root / "MOT17" / "train" / sname / "gt" / "gt.txt"
            )

        n_total = 0
        self._clips: list[tuple[str, int]] = []  # (seq_name, start_frame_idx)
        for sname in sorted(self._feat_cache.keys()):
            n_frames = len(self._feat_cache[sname])
            for start in range(0, n_frames - clip_len + 1, stride):
                self._clips.append((sname, start))
            n_total += n_frames
        print(
            f"  {len(self._clips)} clips from {n_total} frames ({len(self._feat_cache)} sequences)"
        )

    @staticmethod
    def _load_gt(gt_path: Path) -> dict[int, torch.Tensor]:
        by_frame: dict[int, list] = {}
        with open(gt_path) as f:
            for line in f:
                cols = line.strip().split(",")
                if len(cols) < 6:
                    continue
                conf = int(cols[6]) if len(cols) > 6 else 1
                cls_id = int(cols[7]) if len(cols) > 7 else 1
                if conf != 1 or cls_id != 1:
                    continue
                fid = int(cols[0])
                x, y, w, h = (
                    float(cols[2]),
                    float(cols[3]),
                    float(cols[4]),
                    float(cols[5]),
                )
                by_frame.setdefault(fid, []).append([x, y, x + w, y + h])
        return {
            fid: torch.tensor(boxes, dtype=torch.float32)
            for fid, boxes in by_frame.items()
        }

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        sname, start = self._clips[idx]
        feat_cache = self._feat_cache[sname]
        gt_cache = self._gt_cache[sname]

        # Map clip frame index → frame ID (frame IDs start at 1 in MOT)
        feats_list = []
        gt_list = []
        for t in range(self.clip_len):
            cache_idx = start + t
            frame_id = cache_idx + 1  # MOT uses 1-indexed frame IDs
            p3, p4, p5 = feat_cache[cache_idx]
            feats_list.append((p3, p4, p5))
            gt_list.append(gt_cache.get(frame_id, torch.zeros(0, 4)))

        return feats_list, gt_list


def _collate(batch):
    """Collate: each item is (feats_list, gt_list) for one clip."""
    # Group all features by time step
    T = len(batch[0][0])

    batched_feats = []
    batched_gt = []
    for t in range(T):
        p3s = torch.cat([b[0][t][0] for b in batch], dim=0)
        p4s = torch.cat([b[0][t][1] for b in batch], dim=0)
        p5s = torch.cat([b[0][t][2] for b in batch], dim=0)
        batched_feats.append((p3s, p4s, p5s))
        batched_gt.append([b[1][t] for b in batch])

    return batched_feats, batched_gt


# ── main ──
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--cache-dir", default="runs/trt_feat_cache")
    p.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    p.add_argument("--run-dir", default="runs/mamba_cached")
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument(
        "--mamba-ckpt",
        default="runs/mamba_gt_960_v2/best.ckpt",
        help="Pretrained checkpoint for fine-tuning (skipped if --from-scratch)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--clip-len", type=int, default=4)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--spatial-reduction", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--seqs", default="")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--clip-grad", type=float, default=0.1)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true", help="Use AMP in FP16 precision")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── teacher (for loss config only) ──
    teacher = build_gated_yolo_detector(
        str(_root / args.yolo_weights),
        cfg=GatedDetConfig(
            scales=("p3", "p4", "p5"), freeze_backbone=True, img_size=args.img_size
        ),
        device=dev,
    )
    teacher.eval()
    for x in teacher.parameters():
        x.requires_grad_(False)
    nc = teacher.yolo_model.model[-1].nc

    # ── student ──
    if args.from_scratch:
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
        print(f"[From scratch]  LR={args.lr}")
    else:
        ckpt = torch.load(
            _root / args.mamba_ckpt, map_location="cpu", weights_only=False
        )
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
        print(f"[Fine-tune {args.mamba_ckpt}]  LR={args.lr}")
    mamba.train()
    print(f"Mamba: {sum(p.numel() for p in mamba.parameters()):,} params")

    base_args = (
        dict(teacher.yolo_model.args)
        if isinstance(teacher.yolo_model.args, dict)
        else {}
    )
    base_args.update({"box": 7.5, "cls": 0.5, "dfl": 1.5})
    teacher.yolo_model.args = SimpleNamespace(**base_args)
    criterion = v8DetectionLoss(teacher.yolo_model)
    del teacher

    # ── data ──
    data_root = _root / args.data_root
    cache_dir = _root / args.cache_dir
    if args.seqs:
        seq_names = [s.strip() for s in args.seqs.split(",") if s.strip()]
    else:
        seq_names = sorted(p.stem for p in cache_dir.glob("*.pt"))

    dataset = CachedFeatDataset(
        cache_dir, data_root, seq_names, args.clip_len, args.stride, args.img_size
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )

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

    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    N = len(loader)
    best = float("inf")
    print(f"Batches: {N}  B={args.batch_size}  T={args.clip_len}  LR={args.lr}\n")

    # ── loop ──
    for ep in range(1, args.epochs + 1):
        total, st = 0.0, 0
        t0 = time.perf_counter()
        acc = 0
        opt.zero_grad()

        for i, (batched_feats, batched_gt) in enumerate(loader):
            T_f = len(batched_feats)
            B = len(batched_gt[0])

            loss = torch.zeros((), device=dev)
            for t in range(T_f):
                p3, p4, p5 = batched_feats[t]
                dtype = torch.float16 if args.fp16 else torch.float32
                feats = [
                    p3.to(dev, non_blocking=True).to(dtype),
                    p4.to(dev, non_blocking=True).to(dtype),
                    p5.to(dev, non_blocking=True).to(dtype),
                ]
                gt_t = [g.to(dev, non_blocking=True) for g in batched_gt[t]]

                with torch.amp.autocast("cuda", enabled=args.fp16, dtype=torch.float16):
                    cls, reg = mamba(feats)

                # Convert to float32 for stable YOLOv8 loss computation (prevents NaNs)
                cls_f32 = [c.float() for c in cls]
                reg_f32 = [r.float() for r in reg]
                feats_f32 = [f.float() for f in feats]

                preds = _build_preds(cls_f32, reg_f32, feats_f32)
                sl, _ = criterion(preds, _make_yolo_batch(gt_t, args.img_size, dev))
                loss = loss + sl.sum()

            loss = loss / (T_f * B * args.accum_steps)

            scaler.scale(loss).backward()
            acc += 1

            if acc == args.accum_steps:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(mamba.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                acc = 0

            v = loss.item() * args.accum_steps
            if torch.isfinite(loss.detach()):
                total += v
                st += 1

            if (i + 1) % max(1, min(N // 10, 50)) == 0 or i == N - 1:
                eta = (N - i - 1) * (time.perf_counter() - t0) / max(i + 1, 1)
                print(
                    f"\r  e{ep:3d} [{i + 1:4d}/{N}] loss={v:10.4f}  ETA {eta:.0f}s",
                    end="",
                    flush=True,
                )

        if acc > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(mamba.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()
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


if __name__ == "__main__":
    main()
