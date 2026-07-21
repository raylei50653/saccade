#!/usr/bin/env python3
"""Feasibility probe: can the Mamba HEAD ACTIVATION separate occluded from visible GT?

Follow-up to probe_occ_separability.py, which showed the head's scalar SCORE only
weakly separates occlusion (AUC 0.578 among detected GT). The score is a 1-D
bottleneck; the question here is whether the richer high-dimensional ACTIVATION the
head computes right before predicting class/score already encodes occlusion linearly.

What is probed
--------------
The activation hooked is ``x_cls`` -- the (d_model*2) feature map fed into each
scale's ``cls_head`` Sequential (i.e. ``torch.cat([x_cls_proj, x_up])`` in
MambaDetectionHead.forward, the post-SSM representation that directly produces the
classification logits). For every GT box the head detected, we roi_align-pool that
activation at the detection box across all 3 scales and concatenate -> one feature
vector per detected GT. A logistic-regression linear probe is then trained to
separate occluded (vis<occ_thresh) from visible GT, scored by GroupKFold CV with
groups = GT identity so the same person across frames never spans train/test (this
removes the temporal-duplication confound that would inflate AUC).

Reported AUCs (all on the SAME detected-GT population, apples-to-apples):
  * score_auc        : the 1-D detection score (reproduces probe_occ_separability)
  * activation_auc   : the linear probe on x_cls activation  <-- the question
  * activation_auc_shuffle : same probe with permuted labels (chance control ~0.5)

activation_auc >> score_auc and >> 0.5  => occlusion IS linearly decodable from the
head's activation => a trained visibility head is worth the budget.
activation_auc ~ score_auc ~ 0.5..0.6   => activation adds nothing the score lacks
=> NO-GO, occlusion is not in the head representation (it lives in tracker geometry).

Coordinates: frames are stretch-resized to 640 (no letterbox/tiling), so detection
boxes, GT boxes, and activation grids all share the 640 pixel space -- pooling is
exact via roi_align spatial_scale 1/stride.

Usage
-----
  .venv/bin/python scripts/eval/probe_occ_activation_separability.py \
      --output results/occ_separability/activation.json
"""
# status: stable

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "build"))

import saccade_tracking_ext  # noqa: E402, F401
import torchvision  # noqa: E402
from torchvision.ops import roi_align  # noqa: E402

from saccade.perception.temporal_yolo.data_pipeline import (  # noqa: E402
    resize_stretch_batch_gpu,
)
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

IMG_SIZE = 640
SPATIAL_SCALES = {0: 1 / 8, 1: 1 / 16, 2: 1 / 32}


def load_gt(gt_path: Path) -> dict[int, list[tuple]]:
    """{frame: [(gid, x, y, w, h, visibility), ...]} — keep ALL vis (incl. occluded)."""
    out: dict[int, list[tuple]] = {}
    for line in gt_path.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 9:
            continue
        fid, gid = int(p[0]), int(p[1])
        mark, cls_id, vis = int(p[6]), int(p[7]), float(p[8])
        if mark != 1 or cls_id != 1 or vis < 0.1:
            continue
        x, y, w, h = (float(v) for v in p[2:6])
        out.setdefault(fid, []).append((gid, x, y, w, h, vis))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mamba-ckpt", default="runs/mamba_gt_v14replica_t3_t1/best.ckpt")
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--sequences",
        default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,"
        "MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP",
    )
    ap.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    ap.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    ap.add_argument(
        "--trt-backbone-engine",
        default="models/yolo/yolo26s_backbone_640_best.engine",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--occ-thresh", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import cv2

    detector = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        mamba_ckpt=args.mamba_ckpt,
        img_size=IMG_SIZE,
        device=args.device,
        conf_thr=0.001,
        max_det=300,
        trt_backbone_engine=args.trt_backbone_engine,
        temporal_T_override=0,
        use_cuda_graph=False,
        use_whole_graph=False,
    )
    detector.eval()
    detector.mamba_head.set_head_compile(False)

    # Hook each scale's cls_head to capture BOTH its INPUT activation (x_cls) and
    # its OUTPUT cls logits (per-scale, pre-fusion) so we can compare the P3/P4/P5
    # score distributions during occlusion (the cross-pyramid-level signature).
    captured_in: dict[int, torch.Tensor] = {}
    captured_out: dict[int, torch.Tensor] = {}

    def make_hook(scale_idx: int):
        def hook(_module, inp, out):
            captured_in[scale_idx] = inp[0]
            captured_out[scale_idx] = out

        return hook

    handles = [
        detector.mamba_head.cls_head[i].register_forward_hook(make_hook(i))
        for i in range(len(detector.mamba_head.cls_head))
    ]

    feats: list[np.ndarray] = []
    scores: list[float] = []
    per_scale: list[np.ndarray] = []  # (M, n_scales) per-level person score
    geoms: list[list[float]] = []  # [h, footy, w, cx, cy, overlap_count]
    vis_labels: list[float] = []
    groups: list[str] = []  # seq:gid for GroupKFold

    for seq in (s.strip() for s in args.sequences.split(",") if s.strip()):
        seq_root = Path(args.data_root) / args.split / seq
        gts = load_gt(seq_root / "gt" / "gt.txt")
        img_dir = seq_root / "img1"
        frame_ids = sorted(gts.keys())
        if args.max_frames > 0:
            frame_ids = frame_ids[: args.max_frames]
        for n, fid in enumerate(frame_ids, start=1):
            img_path = img_dir / f"{fid:06d}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            sx, sy = w / IMG_SIZE, h / IMG_SIZE
            fb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(args.device)
            f640 = resize_stretch_batch_gpu(fb, IMG_SIZE, args.device)

            captured_in.clear()
            captured_out.clear()
            with torch.inference_mode():
                dets, _ = detector.forward(f640.float(), gate_input=None)
            if isinstance(dets, list):
                dets = dets[0]
            if dets.dim() == 3:
                dets = dets.squeeze(0)
            keep = dets[:, 4] > args.conf_thresh
            dets = dets[keep]
            if dets.numel() == 0 or len(captured_in) != len(SPATIAL_SCALES):
                continue
            det_boxes = dets[:, :4]  # 640 space
            det_scores = dets[:, 4]

            items = gts.get(fid, [])
            if not items:
                continue
            gt_boxes_640 = torch.tensor(
                [
                    [gx / sx, gy / sy, (gx + gw) / sx, (gy + gh) / sy]
                    for _, gx, gy, gw, gh, _v in items
                ],
                device=args.device,
                dtype=torch.float32,
            )
            ious = torchvision.ops.box_iou(gt_boxes_640, det_boxes)
            # Greedy GT->best det match.
            matched_det_idx = []
            matched_item_idx = []
            used = torch.zeros(det_boxes.shape[0], dtype=torch.bool, device=args.device)
            for gi in range(gt_boxes_640.shape[0]):
                row = ious[gi].clone()
                row[used] = -1.0
                biou, di = row.max(0)
                if float(biou) >= args.match_iou:
                    used[di] = True
                    matched_det_idx.append(int(di))
                    matched_item_idx.append(gi)
            if not matched_det_idx:
                continue

            mboxes = det_boxes[matched_det_idx]  # (M,4) 640 coords
            batch_boxes = torch.cat(
                [torch.zeros(len(mboxes), 1, device=args.device), mboxes], dim=1
            )
            # Pool x_cls activation (concat across scales) AND per-scale person
            # score (sigmoid of cls logits, max over classes) at matched boxes.
            parts = []
            scale_scores = []
            for i in range(len(SPATIAL_SCALES)):
                pooled = roi_align(
                    captured_in[i].float(),
                    batch_boxes,
                    output_size=1,
                    spatial_scale=SPATIAL_SCALES[i],
                    aligned=True,
                )
                parts.append(pooled.flatten(1))
                logits = captured_out[i].float()  # (1, num_classes, H, W)
                prob = logits.sigmoid().amax(
                    dim=1, keepdim=True
                )  # person/best-class prob
                s_pooled = roi_align(
                    prob,
                    batch_boxes,
                    output_size=1,
                    spatial_scale=SPATIAL_SCALES[i],
                    aligned=True,
                )  # max-pool would be ideal; roi_align avg over 1 cell ~ center
                scale_scores.append(s_pooled.flatten(1))
            fvec = torch.cat(parts, dim=1).cpu().numpy()  # (M, d_model*2*3)
            svec = torch.cat(scale_scores, dim=1).cpu().numpy()  # (M, n_scales)

            # Geometry baseline features per matched box (640 coords): height,
            # foot-y, width, cx, cy, and overlap_count = # of OTHER GT boxes with
            # IoU>0.1 (a purely geometric occlusion proxy the tracker already has).
            ov = (torchvision.ops.box_iou(mboxes, gt_boxes_640) > 0.1).sum(1) - 1
            mb = mboxes.cpu().numpy()
            ov_np = ov.cpu().numpy()

            for k, gi in enumerate(matched_item_idx):
                gid, _, _, _, _, vis = items[gi]
                x1, y1, x2, y2 = mb[k]
                feats.append(fvec[k])
                scores.append(float(det_scores[matched_det_idx[k]]))
                per_scale.append(svec[k])
                geoms.append(
                    [
                        y2 - y1,
                        y2,
                        x2 - x1,
                        0.5 * (x1 + x2),
                        0.5 * (y1 + y2),
                        float(ov_np[k]),
                    ]
                )
                vis_labels.append(float(vis))
                groups.append(f"{seq}:{gid}")
            if n % 200 == 0:
                print(f"{seq} [{n}/{len(frame_ids)}] matched={len(feats)}")

    for hd in handles:
        hd.remove()

    X = np.asarray(feats, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float64)
    PS = np.asarray(per_scale, dtype=np.float64)  # (n, n_scales) P3,P4,P5 person score
    G = np.asarray(geoms, dtype=np.float64)  # (n, 6) h,footy,w,cx,cy,overlap
    vis = np.asarray(vis_labels, dtype=np.float64)
    grp = np.asarray(groups)
    y = (vis < args.occ_thresh).astype(int)  # 1 = occluded (positive class)
    n_occ, n_vis = int(y.sum()), int((1 - y).sum())
    print(f"\nfeature dim={X.shape[1]}  n={len(y)}  occluded={n_occ}  visible={n_vis}")

    # Dump raw per-box records so any cross-scale combo / baseline is derivable
    # offline without re-running 5k frames of inference.
    npz_path = str(Path(args.output).with_suffix(".npz"))
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        PS=PS,
        G=G,
        score=sc,
        vis=vis,
        y=y,
        groups=grp,
        geom_cols=np.array(["h", "footy", "w", "cx", "cy", "overlap"]),
        scale_cols=np.array(["P3_s8", "P4_s16", "P5_s32"]),
    )
    print(f"dumped raw records -> {npz_path}")

    report: dict = {
        "n": int(len(y)),
        "feature_dim": int(X.shape[1]),
        "n_occluded": n_occ,
        "n_visible": n_vis,
        "occ_thresh": args.occ_thresh,
    }

    from sklearn.metrics import roc_auc_score

    # 1-D score AUC on this exact population (visible ranks above occluded).
    report["score_auc"] = float(roc_auc_score(1 - y, sc))

    # --- Per-scale (P3/P4/P5) score distribution during occlusion ---
    occ_m = y == 1
    vis_m = y == 0
    scale_names = ["P3_s8", "P4_s16", "P5_s32"]
    print(
        f"\n=== per-scale person-score distribution (occluded vis<{args.occ_thresh}) ==="
    )
    print(
        f"{'scale':10s}{'vis_mean':>10s}{'occ_mean':>10s}{'vis_p50':>10s}"
        f"{'occ_p50':>10s}{'AUC(vis>occ)':>14s}"
    )
    scale_rows = {}
    for i, nm in enumerate(scale_names):
        col = PS[:, i]
        vmean, omean = float(col[vis_m].mean()), float(col[occ_m].mean())
        vp50, op50 = float(np.median(col[vis_m])), float(np.median(col[occ_m]))
        auc_i = float(roc_auc_score(1 - y, col))  # visible>occluded on this scale
        scale_rows[nm] = {
            "vis_mean": vmean,
            "occ_mean": omean,
            "vis_p50": vp50,
            "occ_p50": op50,
            "auc": auc_i,
        }
        print(
            f"{nm:10s}{vmean:>10.3f}{omean:>10.3f}{vp50:>10.3f}{op50:>10.3f}{auc_i:>14.3f}"
        )
    report["per_scale_score"] = scale_rows

    # Cross-scale signature: which level fires strongest, and P3/P5 ratio.
    argmax_scale = PS.argmax(axis=1)
    report["argmax_scale_frac"] = {
        f"{scale_names[i]}_strongest": {
            "visible": float((argmax_scale[vis_m] == i).mean()),
            "occluded": float((argmax_scale[occ_m] == i).mean()),
        }
        for i in range(len(scale_names))
    }
    # --- Cross-scale DIFFERENCE signatures (the P3-P5 idea) ---
    # AUC convention: roc_auc_score(y, d) > 0.5 => occluded has LARGER d.
    print(f"\n=== cross-scale score DIFFERENCES (occluded vis<{args.occ_thresh}) ===")
    print(f"{'signature':14s}{'vis_mean':>10s}{'occ_mean':>10s}{'AUC(occ>·)':>12s}")
    diffs = {
        "P3_minus_P5": PS[:, 0] - PS[:, 2],
        "P3_minus_P4": PS[:, 0] - PS[:, 1],
        "P4_minus_P5": PS[:, 1] - PS[:, 2],
        # normalized: difference over total response (scale-invariant)
        "P3_P5_norm": (PS[:, 0] - PS[:, 2]) / (PS[:, 0] + PS[:, 2] + 1e-6),
    }
    diff_rows = {}
    for nm, d in diffs.items():
        a = float(roc_auc_score(y, d))
        diff_rows[nm] = {
            "vis_mean": float(d[vis_m].mean()),
            "occ_mean": float(d[occ_m].mean()),
            "auc_occ_larger": a,
        }
        print(f"{nm:14s}{d[vis_m].mean():>10.3f}{d[occ_m].mean():>10.3f}{a:>12.3f}")
    report["cross_scale_diffs"] = diff_rows
    ratio = PS[:, 0] / (PS[:, 2] + 1e-6)  # P3/P5 ratio (P5~0 makes this unstable)
    report["p3_over_p5_ratio_auc"] = float(roc_auc_score(y, ratio))
    grp_full = grp

    # Linear probe with GroupKFold by identity.
    n_splits = min(5, len(np.unique(grp)))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    gkf = GroupKFold(n_splits=n_splits)
    auc_cv = cross_val_score(clf, X, y, groups=grp, cv=gkf, scoring="roc_auc")
    report["activation_auc_cv"] = {
        "mean": float(auc_cv.mean()),
        "std": float(auc_cv.std()),
        "folds": [float(a) for a in auc_cv],
    }
    # 3-feature probe on just the per-scale (P3/P4/P5) scores: does the cross-scale
    # signature alone separate occlusion better than the single fused score?
    auc_ps = cross_val_score(clf, PS, y, groups=grp_full, cv=gkf, scoring="roc_auc")
    report["per_scale_probe_auc_cv"] = {
        "mean": float(auc_ps.mean()),
        "std": float(auc_ps.std()),
    }
    print(f"  per-scale (P3/P4/P5) 3-feature probe AUC = {auc_ps.mean():.3f}")
    # Shuffle control.
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    auc_shuf = cross_val_score(clf, X, y_shuf, groups=grp, cv=gkf, scoring="roc_auc")
    report["activation_auc_shuffle"] = float(auc_shuf.mean())

    # --- Confound control: is the activation signal NEW vs pure box geometry? ---
    # The tracker's occlusion gate already uses foot-gap / overlap geometry, so the
    # activation only matters if it beats a geometry-only baseline AND retains signal
    # after geometry is regressed out of it.
    auc_geom = cross_val_score(clf, G, y, groups=grp, cv=gkf, scoring="roc_auc")
    report["geometry_baseline_auc_cv"] = {
        "mean": float(auc_geom.mean()),
        "std": float(auc_geom.std()),
        "features": ["h", "footy", "w", "cx", "cy", "overlap"],
    }
    # Residualize activation against geometry (fit per-feature linear deps on G via
    # the full data; a leakage-free estimate would refit per fold, but for a
    # ceiling check the global residual is adequate and conservative-leaning).
    from sklearn.linear_model import LinearRegression

    Xr = X - LinearRegression().fit(G, X).predict(G)
    auc_resid = cross_val_score(clf, Xr, y, groups=grp, cv=gkf, scoring="roc_auc")
    report["activation_resid_geom_auc_cv"] = {
        "mean": float(auc_resid.mean()),
        "std": float(auc_resid.std()),
    }
    print(f"  geometry-only baseline AUC   = {auc_geom.mean():.3f}")
    print(f"  activation | geom residual AUC= {auc_resid.mean():.3f}")

    act = report["activation_auc_cv"]["mean"]
    print(f"\n=== separability (occluded vis<{args.occ_thresh}, detected GT) ===")
    print(f"  score_auc (1-D)              = {report['score_auc']:.3f}")
    print(
        f"  activation_auc (linear probe)= {act:.3f} "
        f"+/- {report['activation_auc_cv']['std']:.3f}  (GroupKFold by identity)"
    )
    print(f"  activation_auc shuffle ctrl  = {report['activation_auc_shuffle']:.3f}")
    geom = report["geometry_baseline_auc_cv"]["mean"]
    resid = report["activation_resid_geom_auc_cv"]["mean"]
    gain_score = act - report["score_auc"]
    gain_geom = act - geom
    # GO only if activation beats BOTH the score AND box geometry, and keeps signal
    # after geometry is residualized out (i.e. carries appearance cues the tracker
    # geometry does not already have).
    usable = act >= 0.70 and gain_geom >= 0.05 and resid >= 0.65
    if usable:
        report["verdict"] = (
            f"GO: activation decodes occlusion (AUC {act:.3f}); beats geometry "
            f"baseline {geom:.3f} (+{gain_geom:.3f}) and holds {resid:.3f} after "
            f"residualizing geometry -> carries non-geometric (appearance) occlusion cues"
        )
    elif act >= 0.70 and gain_geom < 0.05:
        report["verdict"] = (
            f"NO-GO (redundant): activation AUC {act:.3f} ~= geometry baseline "
            f"{geom:.3f}; signal is box-geometry the tracker already uses, not new"
        )
    else:
        report["verdict"] = (
            f"NO-GO: activation AUC {act:.3f} vs score {report['score_auc']:.3f} "
            f"(gain {gain_score:+.3f})"
        )
    print(f"\nVERDICT: {report['verdict']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
