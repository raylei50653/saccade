#!/usr/bin/env python
"""Spatial dilution probe for the mnv4 ReID embedding (LaSt-ViT direction gate).

Question: does the deployed mnv4 GAP embedding get polluted by occluder-person
cells in dirty crops, and — if so — is the pollution recoverable by selective
spatial pooling (oracle upper bound)?

Test A (dilution existence), per dirty crop (GT target box overlapped by a
GT pedestrian intruder with a lower foot = in front):
  - energy share of intruder-region cells on the 7x7 pre-GAP grid vs their
    area share (ratio > 1 → intruder cells punch above their area),
  - per-cell identity attribution: cos(cell, proto_intruder) - cos(cell,
    proto_target) inside vs outside the intruder region.

Test B (oracle recoverability), same crops:
  - masked GAP excluding intruder cells → Δmargin = (cosT - cosO) change,
    fraction of events whose margin improves, wrong→right flip rate,
  - control: random masks of the same cell count (must be beaten, otherwise
    "dropping cells" rather than "dropping intruder cells" explains the gain).

Cell embeddings and masked pools go through the *exact* deploy head: weighted
feature map → timm forward_head → BNNeck → L2 norm (weighting trick makes
avg-pool compute a masked mean). Full-mask parity with backbone(x) is asserted.

Usage:
  .venv/bin/python scripts/eval/appearance/mnv4_spatial_dilution_probe.py \
      [--ckpt runs/reid_mnv4_ft_visclean/best.ckpt] [--gt-root datasets/MOT17/train]
"""
# status: experiment

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# GT loading (mirrors finetune_mobilenetv4_reid conventions)
# ---------------------------------------------------------------------------


def load_gt(path: Path):
    """Per-frame pedestrian rows: {frame: [(tid, box, vis)]} with box=(x,y,w,h)."""
    by_frame: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        flag = float(p[6]) if len(p) > 6 else 1.0
        cls = int(float(p[7])) if len(p) > 7 else 1
        if flag < 1 or cls != 1:
            continue
        fr = int(float(p[0]))
        tid = int(float(p[1]))
        box = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        vis = float(p[8]) if len(p) > 8 else 1.0
        by_frame[fr].append((tid, box, vis))
    return by_frame


def _inter(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _iot(a, b) -> float:
    """Intersection area over area of a (coverage of the target box)."""
    r = _inter(a, b)
    return 0.0 if r is None else (r[2] * r[3]) / max(a[2] * a[3], 1e-6)


def _temporal_idx(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


# ---------------------------------------------------------------------------
# Model: backbone + BNNeck from the visclean ckpt, spatial-aware paths
# ---------------------------------------------------------------------------


class SpatialMNV4:
    def __init__(self, ckpt_path: Path, device: str):
        import timm

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.backbone = timm.create_model(ck["arch"], pretrained=False, num_classes=0)
        self.backbone.load_state_dict(ck["backbone"], strict=True)
        self.backbone.eval()
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.bnneck = torch.nn.BatchNorm1d(d)
        self.bnneck.load_state_dict(ck["bnneck"])
        self.backbone.to(device).eval()
        self.bnneck.to(device).eval()
        self.device = device
        self.mean = torch.tensor(ck["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ck["std"], device=device).view(1, 3, 1, 1)
        ih, iw = ck.get("input_hw", (224, 224))
        self.input_hw = (int(ih), int(iw))

        # parity: full-weight masked pool must equal the deploy embedding
        with torch.no_grad():
            x = torch.rand(2, 3, *self.input_hw, device=device)
            xn = (x - self.mean) / self.std
            ref = F.normalize(self.bnneck(self.backbone(xn)), dim=-1)
            fmap = self.backbone.forward_features(xn)
            ones = torch.ones(2, 1, *fmap.shape[-2:], device=device)
            got = self.pooled_embed(fmap, ones)
        assert torch.allclose(ref, got, atol=1e-5), "masked-pool parity failed"

    @torch.no_grad()
    def features(self, crops: torch.Tensor) -> torch.Tensor:
        """crops float [B,3,H,W] in [0,1] → pre-GAP fmap [B,C,gh,gw]."""
        x = (crops.to(self.device) - self.mean) / self.std
        return self.backbone.forward_features(x)  # type: ignore[no-any-return]

    @torch.no_grad()
    def pooled_embed(self, fmap: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Masked-mean pooling via weight trick, then deploy head.

        weights [B,1,gh,gw] ≥ 0; rescaled so avg-pool == Σ w·f / Σ w.
        """
        n_cells = fmap.shape[-2] * fmap.shape[-1]
        w = weights * (n_cells / weights.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6))
        feat = self.backbone.forward_head(fmap * w)
        return F.normalize(self.bnneck(feat), dim=-1)

    @torch.no_grad()
    def cell_embeds(self, fmap: torch.Tensor) -> torch.Tensor:
        """Each grid cell through the deploy head: [B,C,gh,gw] → [B,gh*gw,D]."""
        b, c, gh, gw = fmap.shape
        cells = fmap.permute(0, 2, 3, 1).reshape(b * gh * gw, c, 1, 1)
        feat = self.backbone.forward_head(cells)
        emb = F.normalize(self.bnneck(feat), dim=-1)
        return emb.view(b, gh * gw, -1)


# ---------------------------------------------------------------------------
# Crop + grid-mask helpers
# ---------------------------------------------------------------------------


def clip_box(box, fw, fh):
    x, y, w, h = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(fw, int(x + w)), min(fh, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def crop_tensor(img: Image.Image, box_xyxy, out_hw) -> torch.Tensor:
    crop = img.crop(box_xyxy).resize((out_hw[1], out_hw[0]), Image.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def jitter_rect(rect_xywh, jitter: float, rng: np.random.Generator):
    """Perturb an occluder rect to emulate tracker localization error.

    center shift ~ U[-j,j]*size on each axis, size scale ~ 1+U[-j,j] per axis.
    A tracker IoU of ~0.85 vs GT corresponds roughly to jitter ≈ 0.10.
    """
    x, y, w, h = rect_xywh
    cx, cy = x + w / 2, y + h / 2
    cx += rng.uniform(-jitter, jitter) * w
    cy += rng.uniform(-jitter, jitter) * h
    w *= 1.0 + rng.uniform(-jitter, jitter)
    h *= 1.0 + rng.uniform(-jitter, jitter)
    return (cx - w / 2, cy - h / 2, w, h)


def grid_overlap(target_xyxy, rect_xywh, gh: int, gw: int) -> np.ndarray:
    """Fraction of each grid cell (over the target crop) covered by rect."""
    tx0, ty0, tx1, ty1 = target_xyxy
    cw, ch = (tx1 - tx0) / gw, (ty1 - ty0) / gh
    rx0, ry0 = rect_xywh[0], rect_xywh[1]
    rx1, ry1 = rx0 + rect_xywh[2], ry0 + rect_xywh[3]
    out = np.zeros((gh, gw), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            cx0, cy0 = tx0 + j * cw, ty0 + i * ch
            iw = min(cx0 + cw, rx1) - max(cx0, rx0)
            ih = min(cy0 + ch, ry1) - max(cy0, ry0)
            if iw > 0 and ih > 0:
                out[i, j] = (iw * ih) / (cw * ch)
    return out


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def run_sequence(seq_dir: Path, model: SpatialMNV4, args, rng: np.random.Generator):
    by_frame = load_gt(seq_dir / "gt" / "gt.txt")
    img_dir = seq_dir / "img1"
    have = {int(p.stem): p for p in img_dir.glob("*.jpg")}
    if not by_frame or not have:
        return None

    # --- select prototype frames (clean) and dirty events -----------------
    proto_cand: dict[int, list] = defaultdict(list)  # tid -> [(fr, box)]
    events = []  # (fr, tid_t, box_t, tid_o, box_o, vis_t, iot)
    for fr, rows in sorted(by_frame.items()):
        if fr not in have:
            continue
        for tid, box, vis in rows:
            if box[3] < args.min_h:
                continue
            others = [(t2, b2, v2) for t2, b2, v2 in rows if t2 != tid]
            iots = [(_iot(box, b2), t2, b2) for t2, b2, _ in others]
            max_iot = max((r[0] for r in iots), default=0.0)
            if vis >= args.proto_vis and max_iot <= args.proto_max_iot:
                proto_cand[tid].append((fr, box))
            if vis > args.dirty_vis_max:
                continue
            # intruder: pedestrian in front (lower foot), substantial coverage
            best = None
            for iot, t2, b2 in iots:
                if iot < args.dirty_iot_min or iot > args.dirty_iot_max:
                    continue
                if b2[1] + b2[3] <= box[1] + box[3]:  # not in front
                    continue
                if best is None or iot > best[0]:
                    best = (iot, t2, b2)
            if best is not None:
                events.append((fr, tid, box, best[1], best[2], vis, best[0]))

    proto_ids = {t for t, lst in proto_cand.items() if len(lst) >= args.proto_min}
    events = [e for e in events if e[1] in proto_ids and e[3] in proto_ids]
    if not events:
        return None
    if len(events) > args.max_events:
        idx = np.linspace(0, len(events) - 1, args.max_events).round().astype(int)
        events = [events[i] for i in sorted(set(idx.tolist()))]

    # --- build prototypes ---------------------------------------------------
    need_ids = {e[1] for e in events} | {e[3] for e in events}
    proto_jobs: dict[int, list] = defaultdict(list)  # frame -> [(tid, box)]
    for tid in need_ids:
        lst = proto_cand[tid]
        for k in _temporal_idx(len(lst), args.proto_k):
            fr, box = lst[k]
            proto_jobs[fr].append((tid, box))

    proto_embs: dict[int, list] = defaultdict(list)
    batch, meta = [], []

    def flush_protos():
        nonlocal batch, meta
        if not batch:
            return
        fmap = model.features(torch.stack(batch))
        ones = torch.ones(len(batch), 1, *fmap.shape[-2:], device=model.device)
        emb = model.pooled_embed(fmap, ones)
        for k, tid in enumerate(meta):
            proto_embs[tid].append(emb[k])
        batch, meta = [], []

    for fr in sorted(proto_jobs):
        img = Image.open(have[fr]).convert("RGB")
        fw, fh = img.size
        for tid, box in proto_jobs[fr]:
            cb = clip_box(box, fw, fh)
            if cb is None:
                continue
            batch.append(crop_tensor(img, cb, model.input_hw))
            meta.append(tid)
            if len(batch) >= args.batch:
                flush_protos()
    flush_protos()
    protos = {
        t: F.normalize(torch.stack(v).mean(0), dim=-1)
        for t, v in proto_embs.items()
        if len(v) >= args.proto_min
    }
    events = [e for e in events if e[1] in protos and e[3] in protos]
    if not events:
        return None

    # --- process dirty events ----------------------------------------------
    rows_out = []
    ev_by_frame: dict[int, list] = defaultdict(list)
    for e in events:
        ev_by_frame[e[0]].append(e)

    ebatch: list[torch.Tensor] = []
    emeta: list[tuple] = []  # (tid_t, tid_o, mask_frac[gh,gw], vis, iot)

    def flush_events():
        nonlocal ebatch, emeta
        if not ebatch:
            return
        fmap = model.features(torch.stack(ebatch))
        b, c, gh, gw = fmap.shape
        n_cells = gh * gw
        cell_e = model.cell_embeds(fmap)  # [b, n, D]
        energy = fmap.pow(2).sum(1).sqrt().view(b, n_cells)  # [b, n]
        ones = torch.ones(b, 1, gh, gw, device=model.device)
        emb_full = model.pooled_embed(fmap, ones)

        masks = torch.stack([torch.from_numpy(m[2]) for m in emeta]).to(
            model.device
        )  # [b, gh, gw] fractional
        hard = (masks >= 0.5).float()
        keep = 1.0 - hard  # cells to keep in oracle pool
        # skip events with empty or full intruder mask
        valid = (hard.sum((1, 2)) >= 1) & (keep.sum((1, 2)) >= 1)

        emb_mask = model.pooled_embed(fmap, keep.unsqueeze(1))
        # random-drop control: same #cells dropped, uniform positions
        d_rand = []
        for _ in range(args.rand_draws):
            rm = torch.ones(b, n_cells, device=model.device)
            for k in range(b):
                nd = int(hard[k].sum().item())
                if 0 < nd < n_cells:
                    drop = torch.from_numpy(
                        rng.choice(n_cells, size=nd, replace=False)
                    ).to(model.device)
                    rm[k, drop] = 0.0
            d_rand.append(model.pooled_embed(fmap, rm.view(b, 1, gh, gw)))

        for k in range(b):
            if not bool(valid[k]):
                continue
            tid_t, tid_o = emeta[k][0], emeta[k][1]
            pt, po = protos[tid_t], protos[tid_o]
            hm = hard[k].view(-1).bool()
            cos_t_cells = cell_e[k] @ pt
            cos_o_cells = cell_e[k] @ po
            attr = (cos_o_cells - cos_t_cells).cpu().numpy()  # >0: looks like O
            en = energy[k]
            m_t, m_o = float(emb_full[k] @ pt), float(emb_full[k] @ po)
            x_t, x_o = float(emb_mask[k] @ pt), float(emb_mask[k] @ po)
            r_t = float(np.mean([float(dr[k] @ pt) for dr in d_rand]))
            r_o = float(np.mean([float(dr[k] @ po) for dr in d_rand]))
            rows_out.append(
                {
                    "seq": seq_dir.name,
                    "tid": tid_t,
                    "tid_o": tid_o,
                    "vis": emeta[k][3],
                    "iot": emeta[k][4],
                    "n_intruder_cells": int(hm.sum()),
                    "area_share": float(hm.float().mean()),
                    "energy_share": float(en[hm].sum() / en.sum().clamp(min=1e-6)),
                    "attr_intruder": float(attr[hm.cpu().numpy()].mean()),
                    "attr_target": float(attr[~hm.cpu().numpy()].mean()),
                    "cos_t_full": m_t,
                    "cos_o_full": m_o,
                    "cos_t_mask": x_t,
                    "cos_o_mask": x_o,
                    "cos_t_rand": r_t,
                    "cos_o_rand": r_o,
                }
            )
        ebatch, emeta = [], []

    for fr in sorted(ev_by_frame):
        img = Image.open(have[fr]).convert("RGB")
        fw, fh = img.size
        for _, tid, box, tid_o, box_o, vis, iot in ev_by_frame[fr]:
            cb = clip_box(box, fw, fh)
            if cb is None:
                continue
            rect = _inter((cb[0], cb[1], cb[2] - cb[0], cb[3] - cb[1]), box_o)
            if rect is None:
                continue
            if args.occ_jitter > 0:
                rect = jitter_rect(rect, args.occ_jitter, rng)
            gh = model.input_hw[0] // 32
            gw = model.input_hw[1] // 32
            mask = grid_overlap(cb, rect, gh, gw)
            ebatch.append(crop_tensor(img, cb, model.input_hw))
            emeta.append((tid, tid_o, mask, vis, iot))
            if len(ebatch) >= args.batch:
                flush_events()
    flush_events()
    return rows_out


def summarize(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"[{label}] no events")
        return
    g = lambda k: np.array([r[k] for r in rows], dtype=np.float64)  # noqa: E731
    area, energy = g("area_share"), g("energy_share")
    ratio = energy / np.clip(area, 1e-6, None)
    m_full = g("cos_t_full") - g("cos_o_full")
    m_mask = g("cos_t_mask") - g("cos_o_mask")
    m_rand = g("cos_t_rand") - g("cos_o_rand")
    wrong = m_full < 0
    flips = float(((m_full < 0) & (m_mask > 0)).sum())
    print(f"\n[{label}] n={len(rows)}")
    print(
        f"  A/dilution   energy/area ratio: median {np.median(ratio):.3f} "
        f"mean {ratio.mean():.3f}  (>1 → intruder cells over-contribute)"
    )
    print(
        f"  A/attribution  intruder cells (cosO-cosT): {g('attr_intruder').mean():+.4f}"
        f"   target cells: {g('attr_target').mean():+.4f}"
    )
    print(
        f"  B/margin  full {m_full.mean():+.4f} → oracle-mask {m_mask.mean():+.4f} "
        f"(Δ {np.mean(m_mask - m_full):+.4f})  rand-mask {m_rand.mean():+.4f} "
        f"(Δ {np.mean(m_rand - m_full):+.4f})"
    )
    print(
        f"  B/decisions  wrong@full {wrong.mean() * 100:.1f}%  "
        f"margin-improved {np.mean(m_mask > m_full) * 100:.1f}%  "
        f"wrong→right {flips / max(wrong.sum(), 1) * 100:.1f}% of wrong"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/reid_mnv4_ft_visclean/best.ckpt")
    ap.add_argument("--gt-root", default="datasets/MOT17/train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-h", type=int, default=64, dest="min_h")
    ap.add_argument("--proto-vis", type=float, default=0.9, dest="proto_vis")
    ap.add_argument("--proto-max-iot", type=float, default=0.1, dest="proto_max_iot")
    ap.add_argument("--proto-k", type=int, default=8, dest="proto_k")
    ap.add_argument("--proto-min", type=int, default=3, dest="proto_min")
    ap.add_argument("--dirty-vis-max", type=float, default=0.6, dest="dirty_vis_max")
    ap.add_argument("--dirty-iot-min", type=float, default=0.25, dest="dirty_iot_min")
    ap.add_argument("--dirty-iot-max", type=float, default=0.75, dest="dirty_iot_max")
    ap.add_argument("--max-events", type=int, default=400, dest="max_events")
    ap.add_argument("--rand-draws", type=int, default=3, dest="rand_draws")
    ap.add_argument(
        "--occ-jitter",
        type=float,
        default=0.0,
        dest="occ_jitter",
        help="perturb occluder box to emulate tracker localization error (0=GT oracle)",
    )
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", default="scripts/tools/out/mnv4_spatial_dilution_probe.csv"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = SpatialMNV4(_ROOT / args.ckpt, args.device)
    gt_root = _ROOT / args.gt_root

    all_rows: list[dict] = []
    for seq in sorted(
        d.name for d in gt_root.iterdir() if d.is_dir() and d.name.endswith("-SDP")
    ):
        rows = run_sequence(gt_root / seq, model, args, rng)
        if rows:
            summarize(rows, seq)
            all_rows.extend(rows)
    summarize(all_rows, "ALL")

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
