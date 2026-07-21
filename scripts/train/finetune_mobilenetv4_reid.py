#!/usr/bin/env python
"""Full-backbone MobileNetV4 ReID fine-tune on Market1501 + MOT-domain crops.

Escalation of the passed offline gate (mobilenetv4 ImageNet weights already
beat the MOT17 appearance ceiling, see
docs/modules/reid/mobilenetv4_integration_options.md): fine-tune the whole
backbone with a standard BoT recipe (BNNeck + CE label-smoothing + batch-hard
triplet, PK sampling) on ID-labelled crops that are identity-disjoint from
MOT17 — Market1501 bounding_box_train + MOT20/DanceTrack/SportsMOT GT crops
(same leak-free protocol as reid_domain_probe.py). Evaluation is the exact
reid_id_benchmark protocol on MOT17 train 7xSDP (224x224 stretch, bicubic),
so numbers are directly comparable to the ImageNet baseline
(conv_small: gap 31-60 71.8% / 61-120 52.0% / 121+ 22.9%).

Usage:
  uv run scripts/train/finetune_mobilenetv4_reid.py --run-dir runs/reid_mnv4_ft
"""
# status: stable

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

_ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.eval.appearance.reid_id_benchmark import (  # noqa: E402
    GAP_BUCKETS,
    SIZE_BUCKETS,
    _benchmark,
    _extract_seq,
    _load_gt,
    _temporal_idx,
)

# Identity-disjoint from MOT17 (same sources as reid_domain_probe.py).
TRAIN_GLOBS = {
    "MOT20": "datasets/MOT20/MOT20/train/*/gt/gt.txt",
    "DanceTrack": "datasets/DanceTrack/train/*/gt/gt.txt",
    "SportsMOT": "datasets/SportsMOT/val/*/gt/gt.txt",
}
MARKET_TRAIN = "datasets/Market-1501-v15.09.15/bounding_box_train"


# ---------------------------------------------------------------------------
# Crop cache: MOT-domain GT crops pre-resized to disk
# ---------------------------------------------------------------------------


def _load_gt_full(path: Path):
    """All GT rows per frame (any class, for occluder geometry) + per-track
    pedestrian rows with visibility: ({frame: [(x,y,w,h)]}, {tid: [(fr, box, vis)]})."""
    by_frame: dict[int, list] = defaultdict(list)
    tracks: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        fr = int(float(p[0]))
        box = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        by_frame[fr].append(box)
        flag = float(p[6]) if len(p) > 6 else 1.0
        cls = int(float(p[7])) if len(p) > 7 else 1
        vis = float(p[8]) if len(p) > 8 else 1.0
        if flag >= 1 and cls == 1:
            tracks[int(float(p[1]))].append((fr, box, vis))
    return by_frame, tracks


def _occluded_behind(box, others, cov_thresh: float) -> bool:
    """True if some other box covers > cov_thresh of `box` and sits in front
    (lower foot = closer to camera in a pedestrian scene)."""
    x, y, w, h = box
    foot = y + h
    area = max(w * h, 1e-6)
    for ox, oy, ow, oh in others:
        if (ox, oy, ow, oh) == box:
            continue
        if oy + oh <= foot:  # not in front
            continue
        iw = min(x + w, ox + ow) - max(x, ox)
        ih = min(y + h, oy + oh) - max(y, oy)
        if iw > 0 and ih > 0 and (iw * ih) / area > cov_thresh:
            return True
    return False


def build_mot_crop_cache(
    cache_dir: Path,
    per_id: int,
    out_hw: tuple[int, int],
    min_h: int,
    vis_min: float,
    occ_cov: float,
    *,
    gpu_decode: bool = False,
) -> list[dict]:
    """Crop per-id temporally-spread GT boxes to cache_dir; returns index entries.

    Occlusion decontamination: drop boxes with annotated vis < vis_min (MOT20;
    DanceTrack/SportsMOT annotate vis=1 everywhere) and, geometrically, boxes
    covered > occ_cov by a box with a lower foot (occluder in front) — a crop
    whose pixels are mostly another person must not carry this identity label.
    """
    out_h, out_w = out_hw
    if gpu_decode:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-decode-cache requires CUDA")
        from torchvision.io import ImageReadMode, decode_jpeg, read_file

        def load_frame(path: Path) -> torch.Tensor:
            raw = read_file(str(path))
            return decode_jpeg(raw, mode=ImageReadMode.RGB, device="cuda")

        def save_crop(img_chw: torch.Tensor, box, rel: str) -> None:
            x1, y1, x2, y2 = box
            crop = img_chw[:, y1:y2, x1:x2].float().unsqueeze(0) / 255.0
            crop = F.interpolate(
                crop,
                size=(out_h, out_w),
                mode="bicubic",
                align_corners=False,
            )
            crop_u8 = (crop.squeeze(0).clamp(0, 1) * 255.0).round().to(torch.uint8)
            torch.save(crop_u8.cpu(), cache_dir / rel)

    else:
        load_frame = None
        save_crop = None

    entries: list[dict] = []
    next_label = 0
    n_vis_drop = n_occ_drop = 0
    for ds, glob in TRAIN_GLOBS.items():
        for gtp in sorted(_ROOT.glob(glob)):
            seq_dir = gtp.parent.parent / "img1"
            seq = gtp.parent.parent.name
            by_frame, gt = _load_gt_full(gtp)
            have = {int(p.stem): p for p in seq_dir.glob("*.jpg")}
            if not gt or not have:
                continue
            samples: list[tuple[int, int, tuple]] = []
            labels_here: set[int] = set()
            for tid, items in gt.items():
                kept = []
                for fr, b, vis in items:
                    if fr not in have or b[3] < min_h:
                        continue
                    if vis < vis_min:
                        n_vis_drop += 1
                        continue
                    if occ_cov < 1.0 and _occluded_behind(b, by_frame[fr], occ_cov):
                        n_occ_drop += 1
                        continue
                    kept.append((fr, b))
                items = sorted(kept, key=lambda r: r[0])
                if len(items) < 2:
                    continue
                for j in _temporal_idx(len(items), per_id):
                    samples.append((tid, *items[j]))
                    labels_here.add(tid)
            if not samples:
                continue
            tid_map = {t: next_label + i for i, t in enumerate(sorted(labels_here))}
            next_label += len(labels_here)
            by_frame: dict[int, list] = defaultdict(list)
            for tid, fr, box in samples:
                by_frame[fr].append((tid, box))
            for fr, lst in by_frame.items():
                if gpu_decode:
                    img_chw = load_frame(have[fr])  # type: ignore[misc]
                    _, fh, fw = img_chw.shape
                else:
                    img = Image.open(have[fr]).convert("RGB")
                    fw, fh = img.size
                for tid, (x, y, w, h) in lst:
                    box = (
                        max(0, int(x)),
                        max(0, int(y)),
                        min(fw, int(x + w)),
                        min(fh, int(y + h)),
                    )
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    if gpu_decode:
                        rel = f"{ds}_{seq}_{tid}_{fr}.pt"
                        save_crop(img_chw, box, rel)  # type: ignore[misc]
                    else:
                        crop = img.crop(box).resize((out_w, out_h), Image.BICUBIC)
                        rel = f"{ds}_{seq}_{tid}_{fr}.jpg"
                        crop.save(cache_dir / rel, quality=95)
                    entries.append(
                        {"path": rel, "label": tid_map[tid], "src_h": int(h)}
                    )
            print(f"  [{ds}] {seq}: {len(labels_here)} ids", flush=True)
    if vis_min > 0 or occ_cov < 1.0:
        print(
            f"occlusion decontamination: dropped {n_vis_drop} vis<{vis_min} "
            f"+ {n_occ_drop} geometric occluded-behind boxes"
        )
    return entries


def load_market_entries(label_offset: int) -> tuple[list[dict], int]:
    """Market1501 bounding_box_train crops; label from filename id prefix."""
    root = _ROOT / MARKET_TRAIN
    ids: dict[int, int] = {}
    entries = []
    for p in sorted(root.glob("*.jpg")):
        pid = int(p.name.split("_")[0])
        if pid == -1:
            continue
        if pid not in ids:
            ids[pid] = label_offset + len(ids)
        entries.append({"path": str(p), "label": ids[pid], "src_h": 128})
    return entries, len(ids)


def build_market_tensor_cache(
    cache_dir: Path, label_offset: int, out_hw
) -> tuple[list[dict], int]:
    """Decode Market1501 crop JPEGs with nvJPEG and store 224x224 uint8 tensors."""
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu-decode-cache requires CUDA")
    from torchvision.io import ImageReadMode, decode_jpeg, read_file

    out_h, out_w = out_hw
    root = _ROOT / MARKET_TRAIN
    ids: dict[int, int] = {}
    entries = []
    out_dir = cache_dir / "market1501"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(root.glob("*.jpg")):
        pid = int(p.name.split("_")[0])
        if pid == -1:
            continue
        if pid not in ids:
            ids[pid] = label_offset + len(ids)
        rel = f"market1501/{p.stem}.pt"
        out_path = cache_dir / rel
        if not out_path.exists():
            raw = read_file(str(p))
            img = decode_jpeg(raw, mode=ImageReadMode.RGB, device="cuda")
            crop = F.interpolate(
                img.float().unsqueeze(0) / 255.0,
                size=(out_h, out_w),
                mode="bicubic",
                align_corners=False,
            )
            crop_u8 = (crop.squeeze(0).clamp(0, 1) * 255.0).round().to(torch.uint8)
            torch.save(crop_u8.cpu(), out_path)
        entries.append({"path": rel, "label": ids[pid], "src_h": 128})
    return entries, len(ids)


# ---------------------------------------------------------------------------
# Dataset + PK sampler
# ---------------------------------------------------------------------------


class CropDataset(Dataset):
    def __init__(self, entries: list[dict], mot_cache: Path, out_hw, train: bool):
        self.entries = entries
        self.mot_cache = mot_cache
        self.out_hw = out_hw
        self.train = train
        from torchvision import transforms

        self._erase = transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0.5)
        self._jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        p = Path(e["path"])
        if not p.is_absolute():
            p = self.mot_cache / p
        out_h, out_w = self.out_hw

        if p.suffix == ".pt":
            t = torch.load(p, map_location="cpu", weights_only=True).float() / 255.0
            if self.train:
                if random.random() < 0.5:
                    t = torch.flip(t, dims=[2])
                # Keep the GPU-decode pixel domain for MOT crops: no PIL color
                # jitter/re-encode path, only tensor-space geometry/erasing.
                pad = 10
                t = F.pad(t, (pad, pad, pad, pad), value=0.5)
                ox, oy = random.randint(0, 2 * pad), random.randint(0, 2 * pad)
                t = t[:, oy : oy + out_h, ox : ox + out_w]
                t = self._erase(t)
            return t, e["label"]

        img = Image.open(p).convert("RGB")
        if self.train:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self._jitter(img)
            # pad + random crop (translation jitter) at target resolution
            img = img.resize((out_w, out_h), Image.BICUBIC)
            pad = 10
            canvas = Image.new("RGB", (out_w + 2 * pad, out_h + 2 * pad), (128,) * 3)
            canvas.paste(img, (pad, pad))
            ox, oy = random.randint(0, 2 * pad), random.randint(0, 2 * pad)
            img = canvas.crop((ox, oy, ox + out_w, oy + out_h))
        else:
            img = img.resize((out_w, out_h), Image.BICUBIC)
        t = (
            torch.from_numpy(
                np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()
            ).float()
            / 255.0
        )
        if self.train:
            t = self._erase(t)
        return t, e["label"]


class PKBatchSampler(Sampler):
    """P identities x K instances per batch."""

    def __init__(self, labels: list[int], P: int, K: int, iters: int, seed: int):
        self.by_label: dict[int, np.ndarray] = {}
        arr = np.asarray(labels)
        for c in np.unique(arr):
            self.by_label[int(c)] = np.where(arr == c)[0]
        self.P, self.K, self.iters = P, K, iters
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.iters

    def __iter__(self):
        classes = list(self.by_label)
        for _ in range(self.iters):
            picked = self.rng.choice(
                classes, size=min(self.P, len(classes)), replace=False
            )
            idx: list[int] = []
            for c in picked:
                pool = self.by_label[int(c)]
                take = self.rng.choice(pool, size=self.K, replace=len(pool) < self.K)
                idx.extend(take.tolist())
            yield idx


# ---------------------------------------------------------------------------
# Model + losses
# ---------------------------------------------------------------------------


class MNV4ReID(nn.Module):
    """timm backbone -> pooled feat -> BNNeck -> classifier (BoT)."""

    def __init__(self, arch: str, ckpt: Path, n_classes: int):
        import timm

        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        self.backbone.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.backbone.eval()
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.feat_dim = d
        self.bnneck = nn.BatchNorm1d(d)
        self.bnneck.bias.requires_grad_(False)  # BoT: no shift
        self.classifier = nn.Linear(d, n_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        feat = self.backbone(x)
        feat_bn = self.bnneck(feat)
        return feat, feat_bn, self.classifier(feat_bn)


def batch_hard_triplet(feat, labels, margin: float = 0.3) -> torch.Tensor:
    d = torch.cdist(feat, feat)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(feat), device=feat.device, dtype=torch.bool)
    pos = d.masked_fill(~same | eye, -1.0).max(1).values
    neg = d.masked_fill(same, float("inf")).min(1).values
    return F.relu(pos - neg + margin).mean()


# ---------------------------------------------------------------------------
# MOT17 eval (identical protocol to reid_id_benchmark)
# ---------------------------------------------------------------------------


class _FTExtractor:
    def __init__(self, model: MNV4ReID, mean, std, device: str):
        self.model = model
        self.device = device
        self.feature_dim = model.feat_dim
        self._mean = mean
        self._std = std

    @torch.no_grad()
    def extract(self, t: torch.Tensor) -> torch.Tensor:
        x = (t - self._mean) / self._std
        _, feat_bn, _ = self.model(x)
        return feat_bn.float()


def eval_mot17(
    model, mean, std, device, crop_hw, gt_root: Path, *, gpu_decode: bool
) -> dict:
    model.eval()
    ext = _FTExtractor(model, mean, std, device)
    gh: dict = defaultdict(int)
    gt_: dict = defaultdict(int)
    sh: dict = defaultdict(int)
    st: dict = defaultdict(int)
    r1s = []
    for seq in sorted(
        d.name for d in gt_root.iterdir() if d.is_dir() and d.name.endswith("-SDP")
    ):
        gt = _load_gt(gt_root / seq / "gt" / "gt.txt")
        res = _extract_seq(
            gt_root / seq / "img1",
            gt,
            ext,
            20,
            crop_hw,
            ".jpg",
            resample="bicubic",
            gpu_decode=gpu_decode,
        )
        if res is None:
            continue
        f, lab, fr, hgt = res
        m = _benchmark(f, lab, fr, hgt)
        r1s.append(m["rank1"])
        print(
            f"  [{seq}] rank1={m['rank1'] * 100:.1f}% mAP={m['mAP'] * 100:.1f}% "
            f"gap={m['gap']:.3f} d'={m['dprime']:.2f} AUC={m['auc'] * 100:.1f}%",
            flush=True,
        )
        for k in m["gap_tot"]:
            gh[k] += m["gap_hit"].get(k, 0)
            gt_[k] += m["gap_tot"][k]
        for k in m["size_tot"]:
            sh[k] += m["size_hit"].get(k, 0)
            st[k] += m["size_tot"][k]
    out = {"macro_rank1": float(np.mean(r1s))}
    for lo, hi in GAP_BUCKETS:
        k = (lo, hi)
        if gt_.get(k):
            name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            out[f"gap_{name}"] = gh[k] / gt_[k]
            print(f"    gap {name:>8}: rank1={gh[k] / gt_[k] * 100:5.1f}% (n={gt_[k]})")
    for lo, hi in SIZE_BUCKETS:
        k = (lo, hi)
        if st.get(k):
            name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            out[f"h_{name}"] = sh[k] / st[k]
            print(f"    h {name:>9}px: rank1={sh[k] / st[k] * 100:5.1f}% (n={st[k]})")
    # relink-regime aggregate used for model selection
    out["gap31plus"] = sum(gh[k] for k in gh if k[0] >= 31) / max(
        1, sum(gt_[k] for k in gt_ if k[0] >= 31)
    )
    print(f"    gap31+ aggregate: {out['gap31plus'] * 100:.1f}%", flush=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/reid_mnv4_ft")
    ap.add_argument("--arch", default="mobilenetv4_conv_small")
    ap.add_argument("--init", default="models/mobilenetv4/mobilenetv4_conv_small.pth")
    ap.add_argument("--input", type=int, default=224)
    ap.add_argument("--per-id", type=int, default=16, help="MOT crops per identity")
    ap.add_argument("--min-h", type=int, default=24, help="min GT box height (px)")
    ap.add_argument(
        "--vis-min",
        type=float,
        default=0.0,
        help="drop GT boxes with annotated visibility below this (MOT20)",
    )
    ap.add_argument(
        "--occ-cov",
        type=float,
        default=1.0,
        help="drop boxes covered more than this by a front (lower-foot) box; 1.0=off",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--P", type=int, default=24)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=3.5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--triplet-w", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gt-root", default="datasets/MOT17/train")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--gpu-decode-cache",
        action="store_true",
        help=(
            "Build MOT-domain crop cache from full frames with torchvision/nvJPEG "
            "GPU decode and tensor-space crop/resize, then train from cached .pt "
            "crops without PIL/JPEG re-decode."
        ),
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    run = _ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    out_hw = (args.input, args.input)

    # ── data ──
    tag = f"crops_{args.input}_pid{args.per_id}_minh{args.min_h}"
    if args.vis_min > 0 or args.occ_cov < 1.0:
        tag += f"_vis{args.vis_min}_cov{args.occ_cov}"
    if args.gpu_decode_cache:
        tag += "_gpu_decode"
    cache = run / tag
    if args.per_id == 0:  # Market1501-only arm
        mot_entries = []
    else:
        index_p = cache / "index.json"
        if index_p.exists():
            mot_entries = json.loads(index_p.read_text())
            print(f"loaded MOT crop cache: {len(mot_entries)} crops")
        else:
            cache.mkdir(parents=True, exist_ok=True)
            print("building MOT crop cache...", flush=True)
            mot_entries = build_mot_crop_cache(
                cache,
                args.per_id,
                out_hw,
                args.min_h,
                args.vis_min,
                args.occ_cov,
                gpu_decode=args.gpu_decode_cache,
            )
            index_p.write_text(json.dumps(mot_entries))
    n_mot_ids = 1 + max((e["label"] for e in mot_entries), default=-1)
    if args.gpu_decode_cache:
        market_index_p = cache / "market_index.json"
        if market_index_p.exists():
            market_entries = json.loads(market_index_p.read_text())
            n_market_ids = 1 + max(
                (e["label"] - n_mot_ids for e in market_entries), default=-1
            )
            print(f"loaded Market tensor cache: {len(market_entries)} crops")
        else:
            print("building Market tensor cache...", flush=True)
            market_entries, n_market_ids = build_market_tensor_cache(
                cache, n_mot_ids, out_hw
            )
            market_index_p.write_text(json.dumps(market_entries))
    else:
        market_entries, n_market_ids = load_market_entries(n_mot_ids)
    entries = mot_entries + market_entries
    n_classes = n_mot_ids + n_market_ids
    print(
        f"train pool: {len(entries)} crops, {n_classes} ids "
        f"(MOT {len(mot_entries)}/{n_mot_ids} + Market {len(market_entries)}/{n_market_ids})"
    )

    ds = CropDataset(entries, cache, out_hw, train=True)
    iters = max(1, n_classes // args.P)
    sampler = PKBatchSampler(
        [e["label"] for e in entries], args.P, args.K, iters, args.seed
    )
    dl = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── model / optim ──
    model = MNV4ReID(args.arch, _ROOT / args.init, n_classes).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    head_params = list(model.bnneck.parameters()) + list(model.classifier.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    total_iters = args.epochs * iters
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda it: (
            min(1.0, (it + 1) / args.warmup_iters)
            * 0.5
            * (1 + np.cos(np.pi * it / total_iters))
        ),
    )
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(
        "\n=== MOT17 baseline (ImageNet init, BNNeck untrained) skipped — "
        "reference: gap 31-60 71.8 / 61-120 52.0 / 121+ 22.9 (conv_small raw) ==="
    )

    best = {"gap31plus": 0.0}
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_ce = tot_tri = tot_acc = nb = 0.0
        for imgs, labels in dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = (imgs - mean) / std
                feat, _, logits = model(x)
                l_ce = ce(logits, labels)
                l_tri = batch_hard_triplet(feat.float(), labels, args.margin)
                loss = l_ce + args.triplet_w * l_tri
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_ce += float(l_ce)
            tot_tri += float(l_tri)
            tot_acc += float((logits.argmax(1) == labels).float().mean())
            nb += 1
        print(
            f"epoch {ep}/{args.epochs} ce={tot_ce / nb:.3f} tri={tot_tri / nb:.3f} "
            f"acc={tot_acc / nb * 100:.1f}% lr={sched.get_last_lr()[0]:.2e} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if ep % args.eval_every == 0 or ep == args.epochs:
            print(f"--- MOT17 eval @ epoch {ep} ---", flush=True)
            m = eval_mot17(
                model,
                mean,
                std,
                device,
                out_hw,
                _ROOT / args.gt_root,
                gpu_decode=args.gpu_decode_cache,
            )
            m["epoch"] = ep
            hist.append(m)
            ckpt = {
                "arch": args.arch,
                "input_hw": list(out_hw),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "n_classes": n_classes,
                "backbone": model.backbone.state_dict(),
                "bnneck": model.bnneck.state_dict(),
                "metrics": m,
                "args": vars(args),
            }
            torch.save(ckpt, run / "last.ckpt")
            if m["gap31plus"] > best["gap31plus"]:
                best = m
                torch.save(ckpt, run / "best.ckpt")
                print(f"  ** new best gap31+ {m['gap31plus'] * 100:.1f}% **")
            (run / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"\ndone. best gap31+ = {best['gap31plus'] * 100:.1f}%  → {run}/best.ckpt")


if __name__ == "__main__":
    main()
