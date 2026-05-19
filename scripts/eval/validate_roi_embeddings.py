#!/usr/bin/env python3
"""
Validate YOLO ROI pool feature discriminability for ReID.

For each GT detection in MOT17, extract ROI-pooled FPN embeddings.
Compute intra-ID vs inter-ID cosine similarity → tell us if features
are discriminative enough to use for lost-track re-association.

Usage:
    uv run scripts/eval/validate_roi_embeddings.py \
        --ckpt runs/gated_det_v1/best.ckpt \
        --sequences MOT17-09-SDP,MOT17-02-SDP
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    build_gated_yolo_detector,
)


# ---------------------------------------------------------------------------
# GT loader
# ---------------------------------------------------------------------------
def load_gt(seq_dir: Path, img_size: int) -> list[dict]:
    """Load GT annotations. Returns list of {frame_id, track_id, bbox_xyxy (img_size space)}."""
    gt_file = seq_dir / "gt" / "gt.txt"
    ini = seq_dir / "seqinfo.ini"
    orig_w, orig_h = 1920, 1080
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith("imWidth"):
                orig_w = int(line.split("=")[1])
            elif line.startswith("imHeight"):
                orig_h = int(line.split("=")[1])

    sx, sy = img_size / orig_w, img_size / orig_h
    entries = []
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        fid, tid = int(parts[0]), int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        vis = float(parts[8]) if len(parts) > 8 else 1.0
        cls = int(parts[7]) if len(parts) > 7 else 1
        if cls != 1 or vis < 0.3 or w < 10 or h < 10:
            continue
        # Scale to img_size space
        x1 = x * sx
        y1 = y * sy
        x2 = (x + w) * sx
        y2 = (y + h) * sy
        entries.append({"frame_id": fid, "track_id": tid, "bbox": [x1, y1, x2, y2]})
    return entries


def load_frame(seq_dir: Path, frame_id: int, img_size: int) -> torch.Tensor:
    import torchvision.io as tv_io

    path = seq_dir / "img1" / f"{frame_id:06d}.jpg"
    img = tv_io.read_image(str(path)).float() / 255.0
    return F.interpolate(
        img.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False
    )


# ---------------------------------------------------------------------------
# ROI embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(
    model,
    feat_cache: dict[str, torch.Tensor],
    boxes_xyxy: torch.Tensor,  # (N, 4) in img_size px space
    img_size: int,
    pool_size: int = 4,
) -> torch.Tensor:
    """ROI-pool P3/P4/P5 features at bbox positions → L2-normalised embedding (N, D)."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0, 1), device=boxes_xyxy.device)

    parts = []
    spatial_scales = {"p3": 1 / 8, "p4": 1 / 16, "p5": 1 / 32}

    # roi_align expects boxes as [batch_idx, x1, y1, x2, y2]
    batch_boxes = torch.cat(
        [torch.zeros(len(boxes_xyxy), 1, device=boxes_xyxy.device), boxes_xyxy], dim=1
    )  # (N, 5)

    for scale, feat in feat_cache.items():
        sp = spatial_scales[scale]
        pooled = roi_align(
            feat.float(),
            batch_boxes,
            output_size=pool_size,
            spatial_scale=sp,
            aligned=True,
        )  # (N, C, 4, 4)
        parts.append(pooled.flatten(1))  # (N, C*16)

    emb = torch.cat(parts, dim=1)  # (N, D_total)
    return F.normalize(emb, dim=1)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def validate_sequence(
    model, seq_dir: Path, img_size: int, device: torch.device, max_frames: int = 200
) -> dict:
    gt = load_gt(seq_dir, img_size)
    # Group by frame
    by_frame: dict[int, list] = defaultdict(list)
    for e in gt:
        by_frame[e["frame_id"]].append(e)

    # Per-track embedding bank {track_id: list[tensor (D,)]}
    track_embs: dict[int, list[torch.Tensor]] = defaultdict(list)

    model.eval()
    model.cache_feats = True

    frames_done = 0
    with torch.no_grad():
        for fid in sorted(by_frame.keys())[:max_frames]:
            anns = by_frame[fid]
            frame = load_frame(seq_dir, fid, img_size).to(device)
            model._feat_cache.clear()
            _ = model(frame)

            if not model._feat_cache:
                continue

            boxes = torch.tensor(
                [a["bbox"] for a in anns], dtype=torch.float32, device=device
            )
            embs = extract_embeddings(model, model._feat_cache, boxes, img_size)

            for i, ann in enumerate(anns):
                track_embs[ann["track_id"]].append(embs[i].cpu())

            frames_done += 1

    model.cache_feats = False

    # ── Compute intra-ID and inter-ID similarities ──
    # Intra: same ID, different frames
    intra_sims = []
    all_mean_embs = {}
    for tid, emb_list in track_embs.items():
        if len(emb_list) < 2:
            continue
        stacked = torch.stack(emb_list)  # (K, D)
        mean_emb = F.normalize(stacked.mean(0), dim=0)
        all_mean_embs[tid] = mean_emb
        # Pairwise intra-ID cosine sim (sample up to 50 pairs)
        n = min(len(emb_list), 20)
        sub = stacked[:n]
        sim_mat = sub @ sub.T  # (n, n)
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        intra_sims.extend(sim_mat[mask].tolist())

    # Inter: different IDs (compare mean embeddings)
    tids = list(all_mean_embs.keys())
    inter_sims = []
    for i in range(len(tids)):
        for j in range(i + 1, min(i + 20, len(tids))):
            s = float(all_mean_embs[tids[i]] @ all_mean_embs[tids[j]])
            inter_sims.append(s)

    return {
        "frames": frames_done,
        "tracks": len(track_embs),
        "emb_dim": embs.shape[1] if len(track_embs) > 0 else 0,
        "intra_mean": sum(intra_sims) / max(len(intra_sims), 1),
        "intra_std": torch.tensor(intra_sims).std().item() if intra_sims else 0,
        "inter_mean": sum(inter_sims) / max(len(inter_sims), 1),
        "inter_std": torch.tensor(inter_sims).std().item() if inter_sims else 0,
        "gap": (sum(intra_sims) / max(len(intra_sims), 1))
        - (sum(inter_sims) / max(len(inter_sims), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--sequences", default="MOT17-09-SDP,MOT17-02-SDP,MOT17-05-SDP")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = raw.get("args", {})
    scales = tuple(s.strip() for s in train_args.get("scales", "p3,p4,p5").split(","))
    cfg = GatedDetConfig(scales=scales)
    yolo_weights = project_root / train_args.get(
        "yolo_weights", "models/yolo/yolo26s.pt"
    )
    model = build_gated_yolo_detector(
        str(yolo_weights), cfg=cfg, device=device, weights_path=str(ckpt_path)
    )
    model.eval()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    print(f"Validating YOLO ROI embedding discriminability on {len(seqs)} sequences\n")
    print(
        f"{'Seq':<20} {'Frames':>6} {'Tracks':>6} {'Dim':>6} "
        f"{'Intra↑':>8} {'Inter↓':>8} {'Gap↑':>8}"
    )
    print("-" * 70)

    total_intra, total_inter, total_gap = [], [], []
    for seq in seqs:
        seq_dir = data_root / "train" / seq
        if not seq_dir.exists():
            print(f"  [SKIP] {seq}")
            continue
        r = validate_sequence(model, seq_dir, args.img_size, device, args.max_frames)
        print(
            f"{seq:<20} {r['frames']:>6} {r['tracks']:>6} {r['emb_dim']:>6} "
            f"{r['intra_mean']:>8.4f} {r['inter_mean']:>8.4f} {r['gap']:>8.4f}"
        )
        total_intra.append(r["intra_mean"])
        total_inter.append(r["inter_mean"])
        total_gap.append(r["gap"])

    if total_gap:
        print("-" * 70)
        print(
            f"{'Average':<20} {'':>6} {'':>6} {'':>6} "
            f"{sum(total_intra) / len(total_intra):>8.4f} "
            f"{sum(total_inter) / len(total_inter):>8.4f} "
            f"{sum(total_gap) / len(total_gap):>8.4f}"
        )
        gap = sum(total_gap) / len(total_gap)
        print(f"\nGap = intra_mean - inter_mean = {gap:.4f}")
        if gap > 0.15:
            print("✅ Features discriminative — suitable for ReID")
        elif gap > 0.05:
            print("⚠️  Marginally discriminative — may help but limited")
        else:
            print("❌ Not discriminative — need dedicated ReID model")


if __name__ == "__main__":
    main()
