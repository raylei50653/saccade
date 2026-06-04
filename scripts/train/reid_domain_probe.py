#!/usr/bin/env python
"""MOT-domain ReID projection-head probe (leak-free).

Cheap test of the open question: can MOT-domain supervision lift the frozen
siglip2_reid embedding's identity discrimination on MOT17? We train a small
projection head (SupCon) on ID-labelled crops from MOT20 / DanceTrack /
SportsMOT, then evaluate the head's output on MOT17 — disjoint identities, so
no leakage. The raw (no-head) MOT17 numbers from reid_id_benchmark are the
baseline to beat, especially at gap >= 31 frames (where motion fails).

  PASS gate: MOT17 gap 31+ rank-1 clearly beats the raw frozen baseline
  (~60% / 33% / 13% for gap 31-60 / 61-120 / 121+). Then escalate to a full
  backbone fine-tune; otherwise frozen features are the ceiling.

Usage:
  uv run scripts/train/reid_domain_probe.py --cache-dir runs/reid_probe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
sys.path.insert(0, str(_ROOT.parent / "src"))

from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402
from scripts.eval.reid_id_benchmark import (  # noqa: E402
    GAP_BUCKETS,
    _benchmark,
    _extract_seq,
    _load_gt,
)

TRAIN_GLOBS = {
    "MOT20": "datasets/MOT20/MOT20/train/*/gt/gt.txt",
    "DanceTrack": "datasets/DanceTrack/train/*/gt/gt.txt",
    "SportsMOT": "datasets/SportsMOT/val/*/gt/gt.txt",
}


def _gather_train(extractor, per_id, crop_hw, max_seqs_per_ds, im_ext):
    """Returns (feats [N,D] frozen+L2, labels [N] global int)."""
    feats_list, labels_list = [], []
    next_label = 0
    for ds, glob in TRAIN_GLOBS.items():
        gts = sorted(Path(".").glob(glob))[:max_seqs_per_ds]
        for gtp in gts:
            seq_dir = gtp.parent.parent / "img1"
            gt = _load_gt(gtp)
            if not gt:
                continue
            # Some datasets ship gt but not all frames (e.g. DanceTrack images
            # un-extracted). Keep only gt entries whose image actually exists.
            have = {int(p.stem) for p in seq_dir.glob(f"*{im_ext}")}
            if not have:
                continue
            gt = {
                tid: [(fr, b) for fr, b in items if fr in have]
                for tid, items in gt.items()
            }
            gt = {tid: items for tid, items in gt.items() if items}
            if not gt:
                continue
            res = _extract_seq(seq_dir, gt, extractor, per_id, crop_hw, im_ext)
            if res is None:
                continue
            f, lab, *_ = res
            # remap this sequence's ids to a global contiguous range
            uniq = {
                int(t): next_label + i for i, t in enumerate(sorted(set(lab.tolist())))
            }
            glob_lab = torch.tensor([uniq[int(t)] for t in lab.tolist()])
            next_label += len(uniq)
            feats_list.append(f.cpu())
            labels_list.append(glob_lab)
            print(
                f"  [{ds}] {gtp.parent.parent.name}: {f.shape[0]} crops, {len(uniq)} ids"
            )
    feats = torch.cat(feats_list)
    labels = torch.cat(labels_list)
    print(f"train pool: {feats.shape[0]} crops, {int(labels.max()) + 1} identities")
    return feats, labels


class Head(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 512, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def _supcon(z: torch.Tensor, labels: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss over an L2-normalized batch."""
    sim = z @ z.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(sim)
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    denom = exp.masked_fill(eye, 0).sum(1)
    log_prob = sim - torch.log(denom + 1e-9)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    loss = -(log_prob * pos).sum(1)[valid] / pos_cnt[valid]
    return loss.mean()


def _pk_batch(labels_np, by_label, P, K, rng):
    classes = rng.choice(list(by_label), size=min(P, len(by_label)), replace=False)
    idx = []
    for c in classes:
        pool = by_label[c]
        take = rng.choice(pool, size=min(K, len(pool)), replace=len(pool) < K)
        idx.extend(take.tolist())
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/reid_probe")
    ap.add_argument("--model-type", default="siglip2_reid")
    ap.add_argument("--per-id", type=int, default=12)
    ap.add_argument("--max-seqs-per-ds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--P", type=int, default=32, help="classes per batch")
    ap.add_argument("--K", type=int, default=4, help="samples per class")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gt-root", default="datasets/MOT17/train")
    ap.add_argument("--im-ext", default=".jpg")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    crop_hw = (
        (256, 128)
        if args.model_type in {"transreid", "osnet", "fastreid"}
        else (224, 224)
    )
    device = "cuda"

    extractor = TRTFeatureExtractor(
        engine_path="", model_type=args.model_type, max_batch=64
    )

    # ── 1. Train-pool embeddings (cached) ──────────────────────────────────────
    tr_cache = cache / f"train_{args.model_type}.pt"
    if tr_cache.exists():
        d = torch.load(tr_cache)
        feats, labels = d["feats"], d["labels"]
        print(
            f"loaded train cache: {feats.shape[0]} crops, {int(labels.max()) + 1} ids"
        )
    else:
        feats, labels = _gather_train(
            extractor, args.per_id, crop_hw, args.max_seqs_per_ds, args.im_ext
        )
        torch.save({"feats": feats, "labels": labels}, tr_cache)

    # ── 2. MOT17 eval embeddings (cached, raw frozen) ──────────────────────────
    ev_cache = cache / f"mot17_{args.model_type}.pt"
    if ev_cache.exists():
        ev = torch.load(ev_cache)
    else:
        ev = {}
        for seq in sorted(
            d.name
            for d in Path(args.gt_root).iterdir()
            if d.is_dir() and d.name.endswith("-SDP")
        ):
            gt = _load_gt(Path(args.gt_root) / seq / "gt" / "gt.txt")
            res = _extract_seq(
                Path(args.gt_root) / seq / "img1",
                gt,
                extractor,
                20,
                crop_hw,
                args.im_ext,
            )
            if res:
                f, lab, fr, _ = res
                ev[seq] = (f.cpu(), lab, fr)
        torch.save(ev, ev_cache)

    def eval_mot17(head=None):
        gh, gt_ = {b: 0 for b in GAP_BUCKETS}, {b: 0 for b in GAP_BUCKETS}
        r1s, gaps = [], []
        for seq, (f, lab, fr) in ev.items():
            fe = f.to(device)
            if head is not None:
                head.eval()
                with torch.no_grad():
                    fe = head(fe)
            m = _benchmark(fe, lab, fr)
            r1s.append(m["rank1"])
            gaps.append(m["gap"])
            for b in GAP_BUCKETS:
                gh[b] += m["gap_hit"].get(b, 0)
                gt_[b] += m["gap_tot"].get(b, 0)
        print(f"  macro rank1={np.mean(r1s) * 100:.1f}%  mean gap={np.mean(gaps):.3f}")
        for lo, hi in GAP_BUCKETS:
            b = (lo, hi)
            if gt_[b]:
                name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
                print(
                    f"    gap {name:>8}: rank1={gh[b] / gt_[b] * 100:5.1f}% (n={gt_[b]})"
                )

    print("\n=== MOT17 raw frozen (baseline) ===")
    eval_mot17(None)

    # ── 3. Train projection head (SupCon, PK batches) ──────────────────────────
    feats = feats.to(device)
    labels_np = labels.numpy()
    by_label: dict[int, np.ndarray] = {}
    for c in np.unique(labels_np):
        by_label[int(c)] = np.where(labels_np == c)[0]
    head = Head(feats.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    iters = max(1, len(by_label) // args.P) * 4
    for ep in range(args.epochs):
        head.train()
        tot = 0.0
        for _ in range(iters):
            idx = _pk_batch(labels_np, by_label, args.P, args.K, rng)
            z = head(feats[idx])
            loss = _supcon(z, labels[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep + 1}/{args.epochs} supcon={tot / iters:.3f}")

    print("\n=== MOT17 with trained head ===")
    eval_mot17(head)
    torch.save(head.state_dict(), cache / f"head_{args.model_type}.pt")
    print(f"saved head → {cache / f'head_{args.model_type}.pt'}")


if __name__ == "__main__":
    main()
