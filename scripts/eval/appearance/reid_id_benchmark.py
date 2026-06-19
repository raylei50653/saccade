#!/usr/bin/env python
"""Direct embedding identity-discrimination benchmark on MOT17 GT crops.

Answers the upstream question that decides whether *any* appearance method can
help: do the ReID embeddings actually separate MOT17 identities? We crop GT
boxes, extract embeddings (siglip2_reid by default), and measure intrinsic
discriminability — independent of the tracker / association logic.

Metrics (leave-one-out, same-frame matches excluded):
  * Rank-1 / mAP            — retrieval quality across a person's own samples.
  * intra vs inter cosine   — mean same-id vs different-id similarity + gap.
  * AUC / d'                 — pairwise same/different separability.
  * Rank-1 stratified by temporal gap — does discrimination survive long gaps
    (the regime where motion fails and relink would have to carry the load)?

Usage:
  uv run scripts/eval/appearance/reid_id_benchmark.py
  uv run scripts/eval/appearance/reid_id_benchmark.py --model-type siglip2_reid --per-id 20
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

GAP_BUCKETS = [(1, 10), (11, 30), (31, 60), (61, 120), (121, 10**9)]
SIZE_BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 10**9)]  # query box height (px)


def _load_gt(
    path: Path,
) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """{gt_id: [(frame, (x,y,w,h)), ...]} for pedestrian (class==1, flag==1)."""
    out: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        flag = float(p[6]) if len(p) > 6 else 1.0
        cls = int(float(p[7])) if len(p) > 7 else 1
        if flag < 1 or cls != 1:
            continue
        frame, tid = int(float(p[0])), int(float(p[1]))
        out[tid].append((frame, (float(p[2]), float(p[3]), float(p[4]), float(p[5]))))
    return out


def _temporal_idx(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [
        int((edges[b] + edges[b + 1] - 1) // 2)
        for b in range(k)
        if edges[b + 1] > edges[b]
    ]


def _extract_seq(
    seq_dir: Path,
    gt: dict,
    extractor,
    per_id: int,
    crop_hw,
    im_ext: str,
    resample: str = "bilinear",
):
    """Returns feats [M,D] (L2-normed), labels [M], frames [M], heights [M]."""
    from PIL import Image

    _RESAMPLE = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }[resample]

    # Resolve frame -> image path from the directory listing (robust to zero-pad
    # width: MOT17/MOT20 use 6 digits, DanceTrack 8). Skip frames with no image.
    frame_paths = {int(p.stem): p for p in Path(seq_dir).glob(f"*{im_ext}")}

    samples: list[tuple[int, int, tuple]] = []  # (label, frame, box)
    for tid, items in gt.items():
        items = sorted(items, key=lambda r: r[0])
        items = [it for it in items if it[0] in frame_paths]
        for j in _temporal_idx(len(items), per_id):
            frame, box = items[j]
            samples.append((tid, frame, box))
    if not samples:
        return None

    by_frame: dict[int, list[int]] = defaultdict(list)
    for si, (_, frame, _) in enumerate(samples):
        by_frame[frame].append(si)

    out_h, out_w = crop_hw
    crops: list[np.ndarray | None] = [None] * len(samples)
    for frame, si_list in by_frame.items():
        img = Image.open(frame_paths[frame]).convert("RGB")
        fw, fh = img.size
        for si in si_list:
            x, y, w, h = samples[si][2]
            box = (
                max(0, int(x)),
                max(0, int(y)),
                min(fw, int(x + w)),
                min(fh, int(y + h)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                box = (0, 0, fw, fh)
            c = img.crop(box).resize((out_w, out_h), _RESAMPLE)
            crops[si] = np.asarray(c, dtype=np.uint8).transpose(2, 0, 1)

    device = getattr(extractor, "device", "cuda")
    feats = torch.empty((len(samples), extractor.feature_dim), device=device)
    for s in range(0, len(samples), 256):
        arr = np.stack(crops[s : s + 256])
        t = torch.from_numpy(arr).to(device).float().div_(255.0)
        feats[s : s + t.shape[0]] = extractor.extract(t)
    feats = torch.nn.functional.normalize(feats, dim=1)
    labels = torch.tensor([s[0] for s in samples])
    frames = torch.tensor([s[1] for s in samples])
    heights = torch.tensor(
        [s[2][3] - s[2][1] for s in samples]
    )  # source box height (px)
    return feats, labels, frames, heights


def _benchmark(feats, labels, frames, heights=None):
    """Leave-one-out retrieval + separability metrics on one camera's samples."""
    n = feats.shape[0]
    sim = (feats @ feats.t()).cpu()  # [n,n] cosine
    labels = labels.cpu()
    frames = frames.cpu()
    heights = heights.cpu() if heights is not None else None
    same = labels[:, None] == labels[None, :]
    same_frame = frames[:, None] == frames[None, :]
    valid = ~same_frame
    valid.fill_diagonal_(False)

    # Rank-1 + AP per query (gallery = all valid)
    rank1 = 0
    aps = []
    n_query = 0
    gap_hit = defaultdict(int)
    gap_tot = defaultdict(int)
    size_hit = defaultdict(int)
    size_tot = defaultdict(int)
    for i in range(n):
        v = valid[i]
        if not bool(v.any()) or not bool(same[i][v].any()):
            continue
        n_query += 1
        s = sim[i].clone()
        s[~v] = -2.0
        order = torch.argsort(s, descending=True)
        order = order[v[order]]  # keep only valid gallery
        rel = same[i][order]
        if bool(rel[0]):
            rank1 += 1
        # AP
        hits = torch.cumsum(rel.float(), 0)
        ranks = torch.arange(1, len(rel) + 1)
        prec = hits / ranks
        ap = (prec * rel.float()).sum() / rel.float().sum()
        aps.append(float(ap))
        # rank-1 stratified by query box height (px) — tests resolution limit
        if heights is not None:
            qh = int(heights[i])
            for lo, hi in SIZE_BUCKETS:
                if lo <= qh < hi:
                    size_tot[(lo, hi)] += 1
                    if bool(rel[0]):
                        size_hit[(lo, hi)] += 1
                    break
        # gap of the top-1 match (temporal distance)
        g = int(abs(frames[order[0]] - frames[i]))
        for lo, hi in GAP_BUCKETS:
            if lo <= g <= hi:
                gap_tot[(lo, hi)] += 1
                if bool(rel[0]):
                    gap_hit[(lo, hi)] += 1
                break

    intra = sim[same & valid]
    inter = sim[(~same) & valid]
    intra_m, inter_m = float(intra.mean()), float(inter.mean())
    pooled_std = float(torch.sqrt((intra.var() + inter.var()) / 2) + 1e-9)
    dprime = (intra_m - inter_m) / pooled_std
    # AUC = P(intra_sim > inter_sim) via Mann-Whitney rank statistic (no [m,n] outer).
    if len(intra) and len(inter):
        combined = torch.cat([intra, inter])
        ranks = torch.argsort(torch.argsort(combined)).float() + 1.0
        r_intra = ranks[: len(intra)].sum()
        auc = float(
            (r_intra - len(intra) * (len(intra) + 1) / 2) / (len(intra) * len(inter))
        )
    else:
        auc = 0.0

    return {
        "n": n,
        "n_query": n_query,
        "rank1": rank1 / max(n_query, 1),
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "intra": intra_m,
        "inter": inter_m,
        "gap": intra_m - inter_m,
        "dprime": dprime,
        "auc": auc,
        "gap_hit": dict(gap_hit),
        "gap_tot": dict(gap_tot),
        "size_hit": dict(size_hit),
        "size_tot": dict(size_tot),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", default="datasets/MOT17/train")
    ap.add_argument("--sequences", default="")
    ap.add_argument("--model-type", default="siglip2_reid")
    ap.add_argument(
        "--per-id", type=int, default=20, help="Temporal samples per identity."
    )
    ap.add_argument("--im-ext", default=".jpg")
    ap.add_argument(
        "--resize", default="bilinear", choices=["bilinear", "bicubic", "lanczos"]
    )
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    seqs = (
        [s.strip() for s in args.sequences.split(",")]
        if args.sequences
        else sorted(
            d.name for d in gt_root.iterdir() if d.is_dir() and d.name.endswith("-SDP")
        )
    )
    crop_hw = (
        (256, 128)
        if args.model_type in {"transreid", "osnet", "fastreid"}
        else (224, 224)
    )
    extractor = TRTFeatureExtractor(
        engine_path="", model_type=args.model_type, max_batch=64
    )

    runs = []  # (seq, metrics)
    gh, gt_ = defaultdict(int), defaultdict(int)
    sh, st = defaultdict(int), defaultdict(int)
    for seq in seqs:
        gt = _load_gt(gt_root / seq / "gt" / "gt.txt")
        res = _extract_seq(
            gt_root / seq / "img1",
            gt,
            extractor,
            args.per_id,
            crop_hw,
            args.im_ext,
            resample=args.resize,
        )
        if res is None:
            continue
        f, lab, fr, hgt = res
        m = _benchmark(f, lab, fr, hgt)
        runs.append((seq, m))
        print(
            f"[{seq}] n={m['n']} rank1={m['rank1'] * 100:.1f}% mAP={m['mAP'] * 100:.1f}% "
            f"intra={m['intra']:.3f} inter={m['inter']:.3f} gap={m['gap']:.3f} "
            f"d'={m['dprime']:.2f} AUC={m['auc'] * 100:.1f}%"
        )
        for k in m["gap_tot"]:
            gh[k] += m["gap_hit"].get(k, 0)
            gt_[k] += m["gap_tot"][k]
        for k in m["size_tot"]:
            sh[k] += m["size_hit"].get(k, 0)
            st[k] += m["size_tot"][k]

    if not runs:
        print("no samples")
        return
    print(f"\n=== gap-stratified rank-1 (resize={args.resize}) ===")
    for lo, hi in GAP_BUCKETS:
        k = (lo, hi)
        if gt_.get(k):
            name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print(f"  gap {name:>8}: rank1={gh[k] / gt_[k] * 100:5.1f}%  (n={gt_[k]})")
    print("=== size-stratified rank-1 (query box height px) — SR-headroom probe ===")
    for lo, hi in SIZE_BUCKETS:
        k = (lo, hi)
        if st.get(k):
            name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print(f"  h {name:>9}px: rank1={sh[k] / st[k] * 100:5.1f}%  (n={st[k]})")


if __name__ == "__main__":
    main()
