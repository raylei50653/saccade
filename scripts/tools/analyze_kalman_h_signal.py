#!/usr/bin/env python3
"""NSA/Kalman h-conditioned noise analysis + Phase-0 fits.

Follow-up to analyze_neutral_nogo_signals.py §A (NSA premise). Answers:

  1. In which score interval does the NSA premise hold? (fine bins, local rho,
     exposure)
  2. Do other per-det signals predict localisation noise after controlling
     score? (h_px, aspect, det_occ, gt_occ, gt_vis, edge_margin)
  3. Is NSA's (1-score)^2 just a proxy for an h-correction? (mutual partials,
     unfloored-shape comparison)
  4. Is the linear sigma∝h assumption (R pos_std=h/20, Q vel_std=h/160)
     supported by data? (sigma_px vs h, GT displacement / acceleration vs h)
  5. Phase-0 fits for the recalibration experiment:
     R:  pos_std = a + b*h   (fit on score>=0.94 subset, anchored so that
         a + b*h_med == h_med/20 -> kalman_r_scale calibration carries over)
     f:  mult = clamp(((1-s)/(1-s0))^2, 1, 30)  (s0 fit on g-corrected
         residuals)
     Q:  vel_std = a + b*h   (fit on GT frame-to-frame velocity change,
         anchored at h_med to h/160)

Inputs are the rescreen worktree artifacts (per-frame box dumps + MOT17 GT);
self-contained loaders, no saccade imports.

Usage:
    .venv/bin/python scripts/tools/analyze_kalman_h_signal.py \
        --dumps-dir ../saccade-rescreen/rescreen_logs/dumps \
        --gt-root ../saccade-rescreen/datasets/MOT17/train
"""
# status: diagnostic

from __future__ import annotations

import argparse
import configparser
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Rescreen artifacts live in a sibling checkout next to this repo. Resolve them
# relative to the repo root's parent so no absolute/home path is baked in.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESCREEN_ROOT = _REPO_ROOT.parent / "saccade-rescreen"

SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]

H_BINS = [(0, 40), (40, 60), (60, 80), (80, 120), (120, 200), (200, 10000)]
SCORE_BINS = [
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.85),
    (0.85, 0.9),
    (0.9, 0.94),
    (0.94, 0.97),
    (0.97, 0.99),
    (0.99, 1.01),
]


# ── loaders (standalone copies; same semantics as analyze_neutral_nogo_signals) ──
def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def load_gt(gt_root: Path, seq: str):
    """{frame: [(gid, x1, y1, x2, y2, vis)]} pedestrians with consider flag."""
    by_frame = defaultdict(list)
    with open(gt_root / seq / "gt" / "gt.txt") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            frm, gid = int(p[0]), int(p[1])
            x, y, w, h = (float(v) for v in p[2:6])
            consider, cls, vis = int(p[6]), int(p[7]), float(p[8])
            if consider != 1 or cls != 1:
                continue
            by_frame[frm].append((gid, x, y, x + w, y + h, vis))
    return dict(by_frame)


def load_dump(dumps_dir: Path, seq: str):
    """{frame: [(x1, y1, x2, y2, score)]} from the last dump stage per frame."""
    stage_rows: dict[tuple[int, str], list] = defaultdict(list)
    stages_seen: list[str] = []
    with open(dumps_dir / f"{seq}.csv") as f:
        for row in csv.DictReader(f):
            st = row["stage"]
            if st not in stages_seen:
                stages_seen.append(st)
            stage_rows[(int(row["frame"]), st)].append(
                (
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                    float(row["score"]),
                )
            )
    stage_rank = {st: i for i, st in enumerate(stages_seen)}
    best: dict[int, str] = {}
    for frm, st in stage_rows:
        if frm not in best or stage_rank[st] > stage_rank[best[frm]]:
            best[frm] = st
    return {frm: stage_rows[(frm, st)] for frm, st in best.items()}


def match_boxes_to_gt(boxes: list, gt_list: list, iou_thr: float = 0.5) -> list[int]:
    """Greedy IoU matching. Returns index into gt_list per box (-1 unmatched)."""
    out = [-1] * len(boxes)
    if not boxes or not gt_list:
        return out
    pairs = []
    for bi, b in enumerate(boxes):
        for gi, g in enumerate(gt_list):
            iou = box_iou(tuple(b[:4]), tuple(g[1:5]))
            if iou >= iou_thr:
                pairs.append((iou, bi, gi))
    pairs.sort(reverse=True)
    used_b: set[int] = set()
    used_g: set[int] = set()
    for _, bi, gi in pairs:
        if bi in used_b or gi in used_g:
            continue
        used_b.add(bi)
        used_g.add(gi)
        out[bi] = gi
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ca, cb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ca**2).sum() * (cb**2).sum()))
    return float((ca * cb).sum() / denom) if denom else float("nan")


def img_size(gt_root: Path, seq: str):
    cp = configparser.ConfigParser()
    cp.read(gt_root / seq / "seqinfo.ini")
    return int(cp["Sequence"]["imWidth"]), int(cp["Sequence"]["imHeight"])


def partial_rho(
    ctrl: np.ndarray, f: np.ndarray, e: np.ndarray, min_n: int = 200
) -> float:
    """Median Spearman(f, e) within deciles of the control variable."""
    dec = np.quantile(ctrl, np.linspace(0, 1, 11))
    parts = []
    for lo, hi in zip(dec[:-1], dec[1:]):
        m = (ctrl >= lo) & (ctrl < hi)
        if m.sum() >= min_n:
            parts.append(spearman(f[m], e[m]))
    return float(np.median(parts)) if parts else float("nan")


# ── data assembly ────────────────────────────────────────────────────────────
def collect(dumps_dir: Path, gt_root: Path):
    """Per matched det: (seq_idx, score, err_h, err_px, h, aspect, det_occ,
    gt_occ, gt_vis, edge_margin)."""
    rows = []
    for si, seq in enumerate(SEQS):
        W, H = img_size(gt_root, seq)
        gt = load_gt(gt_root, seq)
        dets = load_dump(dumps_dir, seq)
        for frm, boxes in dets.items():
            gt_list = gt.get(frm, [])
            if not gt_list:
                continue
            gis = match_boxes_to_gt(boxes, gt_list)
            for bi, (b, gi) in enumerate(zip(boxes, gis)):
                if gi < 0:
                    continue
                gid, gx1, gy1, gx2, gy2, gvis = gt_list[gi]
                gh = max(gy2 - gy1, 1.0)
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                err_px = math.hypot(cx - gcx, cy - gcy)
                bh = max(b[3] - b[1], 1e-6)
                bw = max(b[2] - b[0], 1e-6)
                det_occ = max(
                    (
                        box_iou(tuple(b[:4]), tuple(o[:4]))
                        for j, o in enumerate(boxes)
                        if j != bi
                    ),
                    default=0.0,
                )
                gt_occ = max(
                    (
                        box_iou((gx1, gy1, gx2, gy2), tuple(og[1:5]))
                        for og in gt_list
                        if og[0] != gid
                    ),
                    default=0.0,
                )
                edge = min(b[0] / W, b[1] / H, (W - b[2]) / W, (H - b[3]) / H)
                rows.append(
                    (
                        si,
                        b[4],
                        err_px / gh,
                        err_px,
                        bh,
                        bw / bh,
                        det_occ,
                        gt_occ,
                        gvis,
                        edge,
                    )
                )
    return np.array(rows)


def collect_gt_motion(gt_root: Path):
    """Per GT (track, frame): (h, |disp_px|, |dvel_px|) where dvel is the
    frame-to-frame velocity change (acceleration proxy — the quantity Q models)."""
    rows = []
    for seq in SEQS:
        gt = load_gt(gt_root, seq)
        traj: dict[int, dict[int, tuple]] = defaultdict(dict)
        for frm, gl in gt.items():
            for gid, x1, y1, x2, y2, _ in gl:
                traj[gid][frm] = ((x1 + x2) / 2, (y1 + y2) / 2, y2 - y1)
        for fr in traj.values():
            for f in fr:
                if f + 1 not in fr:
                    continue
                (cx0, cy0, h0), (cx1, cy1, _) = fr[f], fr[f + 1]
                disp = math.hypot(cx1 - cx0, cy1 - cy0)
                dvel = float("nan")
                if f + 2 in fr:
                    cx2, cy2, _ = fr[f + 2]
                    dvel = math.hypot(
                        (cx2 - cx1) - (cx1 - cx0), (cy2 - cy1) - (cy1 - cy0)
                    )
                rows.append((h0, disp, dvel))
    return np.array(rows)


# ── fits ─────────────────────────────────────────────────────────────────────
def fit_affine_binned(h: np.ndarray, y: np.ndarray, anchor_div: float):
    """Weighted LSQ of per-h-bin medians -> y = a + b*h, then rescale so that
    a + b*h_med == h_med/anchor_div (h_med = median of h)."""
    xs, ys, ws = [], [], []
    for lo, hi in H_BINS:
        m = (h >= lo) & (h < hi)
        if m.sum() < 100:
            continue
        xs.append(float(np.median(h[m])))
        ys.append(float(np.median(y[m])))
        ws.append(float(m.sum()))
    b, a = np.polyfit(np.array(xs), np.array(ys), 1, w=np.sqrt(np.array(ws)))
    h_med = float(np.median(h))
    k = (h_med / anchor_div) / max(a + b * h_med, 1e-9)
    return float(a * k), float(b * k), h_med, list(zip(xs, ys))


def fit_nsa_s0(score: np.ndarray, ratio_sq: np.ndarray) -> float:
    """Fit s0 in mult=((1-s)/(1-s0))^2 against per-score-bin variance ratios
    (only bins needing inflation >1)."""
    logs = []
    for (lo, hi), r in zip(SCORE_BINS, ratio_sq):
        if not np.isfinite(r) or r <= 1.0:
            continue
        s_mid = (lo + min(hi, 1.0)) / 2
        logs.append(math.log(1 - s_mid) - 0.5 * math.log(r))
    return 1.0 - math.exp(float(np.mean(logs))) if logs else float("nan")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dumps-dir",
        type=Path,
        default=_RESCREEN_ROOT / "rescreen_logs" / "dumps",
    )
    ap.add_argument(
        "--gt-root",
        type=Path,
        default=_RESCREEN_ROOT / "datasets" / "MOT17" / "train",
    )
    args = ap.parse_args()

    a = collect(args.dumps_dir, args.gt_root)
    seq_i, score, err, err_px, h = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    feat = {
        "h_px": a[:, 4],
        "aspect": a[:, 5],
        "det_occ": a[:, 6],
        "gt_occ": a[:, 7],
        "gt_vis": a[:, 8],
        "edge_margin": a[:, 9],
    }
    n = len(a)
    print(f"n={n} matched dets across {len(SEQS)} seqs\n")

    print("=== 1. exposure: score distribution of matched dets ===")
    for lo, hi in [
        (0.0, 0.3),
        (0.3, 0.5),
        (0.5, 0.7),
        (0.7, 0.85),
        (0.85, 0.94),
        (0.94, 1.01),
    ]:
        m = (score >= lo) & (score < hi)
        print(f"  [{lo:.2f},{hi:.2f})  n={m.sum():6d}  ({100 * m.mean():5.1f}%)")

    print("\n=== 2. fine score-bin residual curve (center_err_h) ===")
    ref = np.median(err[(score >= 0.94) & (score < 0.97)])
    print(
        f"  {'bin':>12} {'n':>6} {'med_err':>8} {'p90_err':>8}  var_ratio(vs [0.94,0.97))"
    )
    for lo, hi in SCORE_BINS:
        m = (score >= lo) & (score < hi)
        if m.sum() < 30:
            continue
        me = np.median(err[m])
        print(
            f"  [{lo:.2f},{hi:.2f}) {m.sum():6d} {me:8.4f} "
            f"{np.quantile(err[m], 0.9):8.4f}  {(me / ref) ** 2:8.1f}x"
        )

    print("\n=== 3. local Spearman(score, err) within score intervals ===")
    for lo, hi in [
        (0.3, 0.5),
        (0.5, 0.7),
        (0.7, 0.85),
        (0.85, 0.94),
        (0.94, 0.99),
        (0.99, 1.01),
        (0.85, 1.01),
        (0.3, 1.01),
    ]:
        m = (score >= lo) & (score < hi)
        if m.sum() < 100:
            continue
        print(
            f"  [{lo:.2f},{hi:.2f})  n={m.sum():6d}  rho={spearman(score[m], err[m]):+.3f}"
        )

    print("\n=== 4. alternative predictors of center_err_h ===")
    hi94 = score >= 0.94
    print(
        f"  {'feature':>12} {'rho_overall':>11} {'rho|s>=0.94':>11} {'partial(ctrl score)':>19}"
    )
    for name, f in feat.items():
        print(
            f"  {name:>12} {spearman(f, err):+11.3f} {spearman(f[hi94], err[hi94]):+11.3f}"
            f" {partial_rho(score, f, err):+19.3f}"
        )
    print(
        f"  {'score':>12} {spearman(score, err):+11.3f} {spearman(score[hi94], err[hi94]):+11.3f}"
        f" {partial_rho(feat['h_px'], score, err):+19.3f}  <- partial ctrl h"
    )

    print("\n=== 5. per-seq: h signal after score control (confound check) ===")
    for si, seq in enumerate(SEQS):
        m = seq_i == si
        print(
            f"  {seq:>14}  n={int(m.sum()):6d}  rho(score,err)={spearman(score[m], err[m]):+.3f}"
            f"  partial rho(h,err|score)={partial_rho(score[m], h[m], err[m], min_n=150):+.3f}"
        )

    print(
        "\n=== 6. NSA-as-h-proxy: implied vs needed multiplier per h bin (anchor h>=200) ==="
    )
    mref = h >= 200
    ref_err = np.median(err[mref])
    floor_ref = np.median(np.maximum(0.05, (1 - score[mref]) ** 2))
    nofloor_ref = np.median((1 - score[mref]) ** 2)
    print(
        f"  {'h bin':>14} {'n':>6} {'med_score':>9} {'needed':>8} {'nsa_floored':>11} {'nsa_unfloored':>13}"
    )
    for lo, hi in H_BINS:
        m = (h >= lo) & (h < hi)
        if m.sum() < 100:
            continue
        need = (np.median(err[m]) / ref_err) ** 2
        fl = np.median(np.maximum(0.05, (1 - score[m]) ** 2)) / floor_ref
        nf = np.median((1 - score[m]) ** 2) / nofloor_ref
        print(
            f"  [{lo:>5},{hi:<6}) {m.sum():6d} {np.median(score[m]):9.3f}"
            f" {need:7.1f}x {fl:10.1f}x {nf:12.1f}x"
        )

    print("\n=== 7. sigma_px(h) on score>=0.94 (R shape) + GT motion (Q shape) ===")
    print(
        f"  {'h bin':>14} {'sigma_px(s>=.94)':>16} {'h/20 (legacy)':>13} {'GT |disp|':>9} {'GT |dvel|':>9}"
    )
    g = collect_gt_motion(args.gt_root)
    gh, gdisp, gdvel = g[:, 0], g[:, 1], g[:, 2]
    for lo, hi in H_BINS:
        m94 = hi94 & (h >= lo) & (h < hi)
        mg = (gh >= lo) & (gh < hi)
        sig = np.median(err_px[m94]) if m94.sum() > 50 else float("nan")
        hm = (
            np.median(h[(h >= lo) & (h < hi)])
            if ((h >= lo) & (h < hi)).any()
            else float("nan")
        )
        dv = gdvel[mg]
        dv = dv[np.isfinite(dv)]
        print(
            f"  [{lo:>5},{hi:<6}) {sig:16.2f} {hm / 20:13.2f}"
            f" {np.median(gdisp[mg]) if mg.sum() > 100 else float('nan'):9.2f}"
            f" {np.median(dv) if len(dv) > 100 else float('nan'):9.2f}"
        )

    print("\n=== 8. Phase-0 fits ===")
    ra, rb, h_med, bins_r = fit_affine_binned(h[hi94], err_px[hi94], anchor_div=20.0)
    print(
        f"  R  pos_std = {ra:.4f} + {rb:.6f}*h   (anchored at h_med={h_med:.0f} to h/20)"
    )
    print(
        f"     fit bins (h, sigma_px): {[(round(x), round(y, 2)) for x, y in bins_r]}"
    )
    g_of = lambda hh: (ra + rb * hh) / (hh / 20.0)  # noqa: E731
    print(
        "     implied g(h) var mult: "
        + ", ".join(
            f"h={hh}:{g_of(hh) ** 2:.1f}x" for hh in (40, 60, 80, 120, 200, 300)
        )
    )

    sig_fit = ra + rb * h
    corr = err_px / np.maximum(sig_fit, 1e-6)
    ref_c = np.median(corr[(score >= 0.94) & (score < 0.97)])
    ratios = []
    for lo, hi in SCORE_BINS:
        m = (score >= lo) & (score < hi)
        ratios.append(
            (np.median(corr[m]) / ref_c) ** 2 if m.sum() >= 100 else float("nan")
        )
    s0 = fit_nsa_s0(score, np.array(ratios))
    print(
        f"  f  s0 = {s0:.4f}   mult = clamp(((1-s)/(1-s0))^2, 1, 30)  [fit on g-corrected residuals]"
    )
    print(
        "     per-bin g-corrected var ratios: "
        + ", ".join(
            f"[{lo},{hi}):{r:.1f}"
            for (lo, hi), r in zip(SCORE_BINS, ratios)
            if np.isfinite(r)
        )
    )

    mfin = np.isfinite(gdvel)
    qa, qb, qh_med, bins_q = fit_affine_binned(gh[mfin], gdvel[mfin], anchor_div=160.0)
    print(
        f"  Q  vel_std = {qa:.4f} + {qb:.6f}*h   (anchored at h_med={qh_med:.0f} to h/160)"
    )
    print(
        f"     fit bins (h, |dvel|_px): {[(round(x), round(y, 2)) for x, y in bins_q]}"
    )


if __name__ == "__main__":
    main()
