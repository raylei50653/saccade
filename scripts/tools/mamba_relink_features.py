#!/usr/bin/env python3
"""Mamba-head feature probe for relink-candidate association matching.

Companion to osnet_relink_features.py: same candidate pairs, same
visibility-based endpoint sampling, but the embedding is the Mamba head's own
P3 feature (input to cls_head[0], 2*d_model channels) ROI-pooled over the box.

Motivation: the T3->T1 GT2 curriculum trains the spatial path for cross-frame
feature consistency (AssA +3.2 end-to-end). If that consistency is an
identity-bearing signal, the head features should separate true/false bridges
where OSNet (hard-pool AUC ~0.50) and geometry (~0.65) failed. Run once with
the plain replica checkpoint and once with the T3->T1 checkpoint — the paired
delta isolates the curriculum effect on the embedding quality.

Usage:
  .venv/bin/python scripts/tools/mamba_relink_features.py \
      --mamba-ckpt runs/mamba_gt_v14replica_t3_t1/best.ckpt \
      --col t3t1_cos --skip 3 --window 12 --topk 3 \
      --out scripts/tools/out/mamba_relink_t3t1.csv
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))
sys.path.insert(0, str(project_root / "scripts" / "tools"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import saccade_tracking_ext  # noqa: E402, F401

from color_relink_features import (  # noqa: E402
    _ap,
    _auc,
    load_boxes,
    pick_sample_frames,
)
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

P3_STRIDE = 8
IMG_SIZE = 640


def roi_pool_p3(feat: torch.Tensor, box, orig_hw) -> np.ndarray | None:
    """Mean-pool the P3 feature map over the (clipped) box region.

    feat: (C, Hp, Wp) P3 feature for one frame (640-stretch coords).
    box: (x, y, w, h) in original image coords.
    """
    H, W = orig_hw
    _, Hp, Wp = feat.shape
    x, y, w, h = box
    sx, sy = IMG_SIZE / W, IMG_SIZE / H
    x0 = int(np.floor(x * sx / P3_STRIDE))
    y0 = int(np.floor(y * sy / P3_STRIDE))
    x1 = int(np.ceil((x + w) * sx / P3_STRIDE))
    y1 = int(np.ceil((y + h) * sy / P3_STRIDE))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, x0 + 1), Wp), min(max(y1, y0 + 1), Hp)
    if x0 >= Wp or y0 >= Hp:
        return None
    v = feat[:, y0:y1, x0:x1].mean(dim=(1, 2))
    return v.float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--csv", type=Path, default=Path("scripts/tools/out/relink_candidates.csv")
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--mamba-ckpt", required=True)
    ap.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    ap.add_argument(
        "--teacher-ckpt", default="runs/gated_det_v14replica/epoch_0012.ckpt"
    )
    ap.add_argument(
        "--trt-backbone-engine",
        default="models/yolo/yolo26s_backbone_640_best.engine",
    )
    ap.add_argument("--col", default="mamba_cos", help="output column name")
    ap.add_argument(
        "--out", type=Path, default=Path("scripts/tools/out/mamba_relink_features.csv")
    )
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--hard-dist", type=float, default=1.0)
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke-test cap")
    args = ap.parse_args()

    device = torch.device("cuda")
    detector = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=IMG_SIZE,
        device="cuda",
        conf_thr=0.001,
        trt_backbone_engine=args.trt_backbone_engine,
        temporal_T_override=0,
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    detector.mamba_head.eval()

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs):
        captured["p3"] = inputs[0].detach()

    handle = detector.mamba_head.cls_head[0].register_forward_pre_hook(_hook)

    rows = list(csv.DictReader(open(args.csv)))
    if args.max_pairs:
        rows = rows[: args.max_pairs]
    need: dict[str, set] = defaultdict(set)
    for r in rows:
        need[r["seq"]].add((int(r["lost_id"]), "lost"))
        need[r["seq"]].add((int(r["cand_id"]), "cand"))

    embs: dict[tuple, np.ndarray] = {}
    for seq, endpoints in sorted(need.items()):
        by_id, by_frame = load_boxes(args.mot_dir / f"{seq}.txt")
        plan: dict[int, list] = defaultdict(list)
        for tid, side in endpoints:
            traj = by_id.get(tid)
            if not traj:
                continue
            for frm, box, _occ, vis in pick_sample_frames(
                traj, by_frame, side, args.window, args.topk, args.skip
            ):
                plan[frm].append(((seq, tid, side), box))
        img_dir = args.img_root / seq / "img1"
        acc: dict[tuple, list] = defaultdict(list)
        n_frames = 0
        for frm in sorted(plan):
            bgr = cv2.imread(str(img_dir / f"{frm:06d}.jpg"), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            orig_hw = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = (
                torch.from_numpy(rgb)
                .to(device=device, dtype=torch.float32)
                .permute(2, 0, 1)
                .div_(255.0)
                .unsqueeze(0)
            )
            t640 = F.interpolate(
                t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                detector.forward(t640)
            feat = captured["p3"][0]  # (C, 80, 80)
            for key, box in plan[frm]:
                v = roi_pool_p3(feat, box, orig_hw)
                if v is not None:
                    acc[key].append(v)
            n_frames += 1
        n0 = len(embs)
        for k, fs in acc.items():
            m = np.mean(fs, axis=0)
            embs[k] = m / max(np.linalg.norm(m), 1e-9)
        print(
            f"  {seq:<16} endpoints {len(embs) - n0}/{len(endpoints)} "
            f"({n_frames} frames)"
        )

    handle.remove()

    n_valid = 0
    for r in rows:
        seq = r["seq"]
        ea = embs.get((seq, int(r["lost_id"]), "lost"))
        eb = embs.get((seq, int(r["cand_id"]), "cand"))
        if ea is None or eb is None:
            r[args.col] = ""
            continue
        r[args.col] = f"{float(ea @ eb):.6f}"
        n_valid += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows ({n_valid} with embeddings) -> {args.out}")

    ok = [r for r in rows if r["gt_valid"] == "1" and r[args.col] != ""]
    y = np.array([int(r["gt_match"]) for r in ok])
    bd = np.array([float(r["bridge_dist"]) for r in ok])
    sc = np.array([float(r[args.col]) for r in ok])
    gap = np.array([int(r["gap"]) for r in ok])
    hard = bd <= args.hard_dist
    print(f"\nPool : {len(y)} pairs | {int(y.sum())} pos ({100 * y.mean():.1f}%)")
    print(f"Hard : {int(hard.sum())} pairs | {int(y[hard].sum())} pos")
    print(
        f"\n{'variant':<22} {'AUC full':>9} {'AUC hard':>9} {'AP full':>8} {'AP hard':>8}"
    )
    print(
        f"{'-bridge_dist (geom)':<22} {_auc(-bd, y):>9.4f} "
        f"{_auc(-bd[hard], y[hard]):>9.4f} "
        f"{_ap(-bd, y):>8.4f} {_ap(-bd[hard], y[hard]):>8.4f}"
    )
    print(
        f"{args.col:<22} {_auc(sc, y):>9.4f} {_auc(sc[hard], y[hard]):>9.4f} "
        f"{_ap(sc, y):>8.4f} {_ap(sc[hard], y[hard]):>8.4f}"
    )

    print(f"\n── {args.col}: hard-pool AUC by gap bucket ──")
    for glo, ghi in [(1, 10), (10, 30), (30, 80), (80, 301)]:
        m = hard & (gap >= glo) & (gap < ghi)
        if m.sum() and y[m].sum() and (1 - y[m]).sum():
            print(
                f"  gap [{glo:>3},{ghi:>3}): {int(m.sum()):>5} pairs, "
                f"{int(y[m].sum()):>4} pos, AUC {_auc(sc[m], y[m]):.4f}"
            )

    print(f"\n── {args.col}: hard-pool AUC per seq ──")
    seqs = np.array([r["seq"] for r in ok])
    for s in np.unique(seqs):
        m = hard & (seqs == s)
        if m.sum() and y[m].sum() and (1 - y[m]).sum():
            print(
                f"  {s}: n={int(m.sum()):>4} pos={int(y[m].sum()):>3} "
                f"{args.col} {_auc(sc[m], y[m]):.3f}  geom {_auc(-bd[m], y[m]):.3f}"
            )


if __name__ == "__main__":
    main()
