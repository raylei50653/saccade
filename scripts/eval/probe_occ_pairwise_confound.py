#!/usr/bin/env python3
"""Follow-up to registry #46 (probe_occ_activation_separability.py): does the
"non-geometric" activation residual (AUC 0.793) survive once the geometry
baseline also includes PAIR-RELATIVE position to the nearest overlapping GT
box -- not just single-box features (h, footy, w, cx, cy, overlap_count)?

Motivation
----------
#46's confound control regressed activation against a geometry baseline of
[h, footy, w, cx, cy, overlap_count] -- all properties of the box ITSELF. It
never included the pairwise relative-position feature the C++ occlusion gate
(compute_track_occlusion_kernel, tracker_gpu.cu:439-440) actually uses and
that was independently proven GO (project_depth_ordering_crossing_swap):

    gap_h = (footy_t - footy_p) / h_ref      # h_ref = 0.5*(h_t + h_p)
    is_front = |gap_h| <= occ_foot_gap && footy_t > footy_p

If the 0.793 "non-geometric" residual is partly a proxy for this pairwise
signal (which the geometry baseline never saw), extending the baseline with
it should pull the residual AUC down toward the single-box baseline (0.670).
If 0.793 survives ~unchanged, the non-geometric claim in #46 is confirmed
more strongly (activation carries something beyond BOTH single-box geometry
AND nearest-partner relative position).

This script re-runs the same detected-GT population/probe as #46 but adds to
the geometry baseline: partner_iou (max IoU with any OTHER GT box in frame,
same threshold as overlap_count), gap_h (signed foot-line gap normalized by
h_ref, matching the C++ definition exactly), and dx_norm (signed horizontal
center offset normalized by width, for relative position beyond depth only).

Usage
-----
  .venv/bin/python scripts/eval/probe_occ_pairwise_confound.py \
      --output results/occ_separability/pairwise_confound.json
"""

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

from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

IMG_SIZE = 640
SPATIAL_SCALES = {0: 1 / 8, 1: 1 / 16, 2: 1 / 32}


def load_gt(gt_path: Path) -> dict[int, list[tuple]]:
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

    captured_in: dict[int, torch.Tensor] = {}

    def make_hook(scale_idx: int):
        def hook(_module, inp, out):
            captured_in[scale_idx] = inp[0]

        return hook

    handles = [
        detector.mamba_head.cls_head[i].register_forward_hook(make_hook(i))
        for i in range(len(detector.mamba_head.cls_head))
    ]

    feats: list[np.ndarray] = []
    geoms: list[list[float]] = []  # [h, footy, w, cx, cy, overlap_count]
    pair_geoms: list[list[float]] = []  # [partner_iou, gap_h, dx_norm]
    vis_labels: list[float] = []
    groups: list[str] = []

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
            det_boxes = dets[:, :4]

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

            mboxes = det_boxes[matched_det_idx]
            batch_boxes = torch.cat(
                [torch.zeros(len(mboxes), 1, device=args.device), mboxes], dim=1
            )
            parts = []
            for i in range(len(SPATIAL_SCALES)):
                pooled = roi_align(
                    captured_in[i].float(),
                    batch_boxes,
                    output_size=1,
                    spatial_scale=SPATIAL_SCALES[i],
                    aligned=True,
                )
                parts.append(pooled.flatten(1))
            fvec = torch.cat(parts, dim=1).cpu().numpy()

            # overlap_count (same threshold as #46) + pairwise nearest-partner
            # relative position, matching tracker_gpu.cu:439-440 exactly:
            #   gap_h = (footy_own - footy_partner) / (0.5*(h_own+h_partner))
            #   dx_norm = (cx_own - cx_partner) / (0.5*(w_own+w_partner))
            iou_mat = torchvision.ops.box_iou(mboxes, gt_boxes_640)  # (M, Ngt)
            mb = mboxes.cpu().numpy()
            gtb = gt_boxes_640.cpu().numpy()

            for k, gi in enumerate(matched_item_idx):
                gid, _, _, _, _, vis = items[gi]
                x1, y1, x2, y2 = mb[k]
                h_own, w_own = y2 - y1, x2 - x1
                footy_own, cx_own = y2, 0.5 * (x1 + x2)

                row = iou_mat[k].clone()
                row[gi] = -1.0
                partner_iou_t, partner_idx_t = row.max(0)
                partner_iou = float(partner_iou_t)
                overlap_count = float((row > 0.1).sum())

                if partner_iou > 0.1:
                    pidx = int(partner_idx_t)
                    px1, py1, px2, py2 = gtb[pidx]
                    h_p, w_p = py2 - py1, px2 - px1
                    footy_p, cx_p = py2, 0.5 * (px1 + px2)
                    h_ref = max(0.5 * (h_own + h_p), 1e-3)
                    w_ref = max(0.5 * (w_own + w_p), 1e-3)
                    gap_h = (footy_own - footy_p) / h_ref
                    dx_norm = (cx_own - cx_p) / w_ref
                else:
                    gap_h = 0.0
                    dx_norm = 0.0

                feats.append(fvec[k])
                geoms.append(
                    [h_own, footy_own, w_own, cx_own, 0.5 * (y1 + y2), overlap_count]
                )
                pair_geoms.append([partner_iou, gap_h, dx_norm])
                vis_labels.append(float(vis))
                groups.append(f"{seq}:{gid}")
            if n % 200 == 0:
                print(f"{seq} [{n}/{len(frame_ids)}] matched={len(feats)}")

    for hd in handles:
        hd.remove()

    X = np.asarray(feats, dtype=np.float32)
    G = np.asarray(geoms, dtype=np.float64)
    P = np.asarray(pair_geoms, dtype=np.float64)
    Gp = np.concatenate([G, P], axis=1)  # single-box geom + pairwise relative position
    vis = np.asarray(vis_labels, dtype=np.float64)
    grp = np.asarray(groups)
    y = (vis < args.occ_thresh).astype(int)
    n_occ, n_vis = int(y.sum()), int((1 - y).sum())
    print(f"\nn={len(y)}  occluded={n_occ}  visible={n_vis}")
    print(f"partner found (overlap>0.1): {int((P[:, 0] > 0.1).sum())} / {len(y)}")

    npz_path = str(Path(args.output).with_suffix(".npz"))
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        G=G,
        P=P,
        vis=vis,
        y=y,
        groups=grp,
        geom_cols=np.array(["h", "footy", "w", "cx", "cy", "overlap"]),
        pair_cols=np.array(["partner_iou", "gap_h", "dx_norm"]),
    )
    print(f"dumped raw records -> {npz_path}")

    n_splits = min(5, len(np.unique(grp)))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    gkf = GroupKFold(n_splits=n_splits)

    report: dict = {"n": int(len(y)), "n_occluded": n_occ, "n_visible": n_vis}

    auc_act = cross_val_score(clf, X, y, groups=grp, cv=gkf, scoring="roc_auc")
    report["activation_auc_cv"] = {
        "mean": float(auc_act.mean()),
        "std": float(auc_act.std()),
    }

    auc_geom_single = cross_val_score(clf, G, y, groups=grp, cv=gkf, scoring="roc_auc")
    report["geometry_single_box_auc_cv"] = {
        "mean": float(auc_geom_single.mean()),
        "std": float(auc_geom_single.std()),
        "features": ["h", "footy", "w", "cx", "cy", "overlap"],
    }

    auc_pair_only = cross_val_score(clf, P, y, groups=grp, cv=gkf, scoring="roc_auc")
    report["pairwise_position_only_auc_cv"] = {
        "mean": float(auc_pair_only.mean()),
        "std": float(auc_pair_only.std()),
        "features": ["partner_iou", "gap_h", "dx_norm"],
    }

    auc_geom_pair = cross_val_score(clf, Gp, y, groups=grp, cv=gkf, scoring="roc_auc")
    report["geometry_plus_pairwise_auc_cv"] = {
        "mean": float(auc_geom_pair.mean()),
        "std": float(auc_geom_pair.std()),
        "features": [
            "h",
            "footy",
            "w",
            "cx",
            "cy",
            "overlap",
            "partner_iou",
            "gap_h",
            "dx_norm",
        ],
    }

    Xr_single = X - LinearRegression().fit(G, X).predict(G)
    auc_resid_single = cross_val_score(
        clf, Xr_single, y, groups=grp, cv=gkf, scoring="roc_auc"
    )
    report["activation_resid_single_geom_auc_cv"] = {
        "mean": float(auc_resid_single.mean()),
        "std": float(auc_resid_single.std()),
    }

    Xr_pair = X - LinearRegression().fit(Gp, X).predict(Gp)
    auc_resid_pair = cross_val_score(
        clf, Xr_pair, y, groups=grp, cv=gkf, scoring="roc_auc"
    )
    report["activation_resid_geom_plus_pairwise_auc_cv"] = {
        "mean": float(auc_resid_pair.mean()),
        "std": float(auc_resid_pair.std()),
    }

    # Does pairwise position alone predict gap_h's sign relationship to occlusion,
    # for a sanity check against the C++ same-height/foot-gap gate semantics.
    has_partner = P[:, 0] > 0.1
    if has_partner.sum() > 10:
        report["gap_h_abs_auc_given_partner"] = float(
            roc_auc_score(y[has_partner], np.abs(P[has_partner, 1]))
        )

    print("\n=== #46 follow-up: pairwise relative-position confound ===")
    print(
        f"  activation_auc (linear probe)         = {report['activation_auc_cv']['mean']:.3f}"
    )
    print(
        f"  geometry (single-box, #46 baseline)   = {report['geometry_single_box_auc_cv']['mean']:.3f}"
    )
    print(
        f"  pairwise position ONLY (new)          = {report['pairwise_position_only_auc_cv']['mean']:.3f}"
    )
    print(
        f"  geometry + pairwise position           = {report['geometry_plus_pairwise_auc_cv']['mean']:.3f}"
    )
    print(
        f"  activation | single-box geom residual  = {report['activation_resid_single_geom_auc_cv']['mean']:.3f}  (#46's number)"
    )
    print(
        f"  activation | geom+pairwise residual     = {report['activation_resid_geom_plus_pairwise_auc_cv']['mean']:.3f}  (this test)"
    )

    resid_single = report["activation_resid_single_geom_auc_cv"]["mean"]
    resid_pair = report["activation_resid_geom_plus_pairwise_auc_cv"]["mean"]
    drop = resid_single - resid_pair
    if drop >= 0.05:
        verdict = (
            f"PARTIAL RETRACTION: residual drops {resid_single:.3f} -> {resid_pair:.3f} "
            f"(-{drop:.3f}) once pairwise relative position is included in the confound "
            "-> part of #46's 'non-geometric' signal was a pairwise-position proxy"
        )
    else:
        verdict = (
            f"CONFIRMED: residual holds {resid_single:.3f} -> {resid_pair:.3f} "
            f"({-drop:+.3f}) after also regressing out pairwise relative position "
            "-> activation carries signal beyond single-box geometry AND nearest-partner "
            "position (appearance-based, as #46 concluded)"
        )
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
