#!/usr/bin/env python3
"""Offline AUC test: color-histogram appearance features for relink candidates.

Consumes the pair table from build_relink_candidates.py plus the raw MOT txt
(full bboxes) and MOT17 frames, and measures whether cheap color features can
separate true from false (lost -> candidate) bridges where geometry cannot
(all motion features AUC ~0.55 in the gate-active region).

Per track endpoint (lost exit / candidate entry) we:
  1. scan the last/first --window frames and pick the frame with the highest
     geometric visibility (own box minus boxes of dets in front of it; "in
     front" = larger y2, i.e. feet lower in the image),
  2. crop, resize to 64x128, and build center-weighted HSV histograms
     (16 hue x 4 sat chromatic bins + 4 value bins for achromatic pixels).

Pair similarity = Bhattacharyya coefficient, four variants:
  full_nomask   whole-crop histogram, no occlusion mask
  split_nomask  torso band [0.20,0.55]h + legs band [0.55,0.90]h, averaged
  full_mask     whole-crop, occluder pixels zero-weighted
  split_mask    torso+legs, occluder pixels zero-weighted

AUC/AP are reported on the full pool and the hard pool (bridge_dist <= 1.0 h,
same definition as sweep_velocity_variants.py), against the geometric
bridge_dist baseline on identical rows.

Usage:
  .venv/bin/python scripts/tools/color_relink_features.py \
      --csv scripts/tools/out/relink_candidates.csv \
      --mot-dir results/MOT17_eval --img-root datasets/MOT17/train \
      --out scripts/tools/out/color_relink_features.csv
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CROP_W, CROP_H = 64, 128
H_BINS, S_BINS, V_BINS = 16, 4, 4  # 16x4 chromatic + 4 achromatic = 68 dims
ACHROMATIC_S = 0.15  # below this saturation hue is meaningless
TORSO = (0.20, 0.55)  # fraction of crop height
LEGS = (0.55, 0.90)
POOL_CX = (0.25, 0.75)  # hard center band (fraction of crop width)
BLUR_K = 9


# ── metrics (same as sweep_velocity_variants.py) ──────────────────────────────
def _auc(score: np.ndarray, y: np.ndarray) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    o = np.argsort(score, kind="mergesort")
    r = np.empty(len(score))
    ss = score[o]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def _ap(score: np.ndarray, y: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    o = np.argsort(-score, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / pos
    drec = np.diff(np.concatenate([[0], rec]))
    return float((prec * drec).sum())


# ── MOT loading (full bboxes, unlike build_relink_candidates) ─────────────────
def load_boxes(path: Path):
    """Return ({id: [(frame, x, y, w, h), ...]}, {frame: [(id, x, y, w, h), ...]})."""
    by_id: dict[int, list] = defaultdict(list)
    by_frame: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frm, tid = int(p[0]), int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            by_id[tid].append((frm, x, y, w, h))
            by_frame[frm].append((tid, x, y, w, h))
    for tid in by_id:
        by_id[tid].sort(key=lambda r: r[0])
    return dict(by_id), dict(by_frame)


# ── visibility ────────────────────────────────────────────────────────────────
def occluder_mask(box, others, grid_h=CROP_H, grid_w=CROP_W):
    """Boolean mask (grid_h, grid_w) in crop coords; True = covered by a det in
    front (larger y2).  Returns (mask, visibility)."""
    x, y, w, h = box
    y2 = y + h
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    if w <= 0 or h <= 0:
        return mask, 0.0
    for ox, oy, ow, oh in others:
        if oy + oh <= y2:  # not in front
            continue
        ix0, iy0 = max(x, ox), max(y, oy)
        ix1, iy1 = min(x + w, ox + ow), min(y + h, oy + oh)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        gx0 = int(np.floor((ix0 - x) / w * grid_w))
        gx1 = int(np.ceil((ix1 - x) / w * grid_w))
        gy0 = int(np.floor((iy0 - y) / h * grid_h))
        gy1 = int(np.ceil((iy1 - y) / h * grid_h))
        mask[max(gy0, 0) : min(gy1, grid_h), max(gx0, 0) : min(gx1, grid_w)] = True
    return mask, 1.0 - mask.mean()


def pick_sample_frames(
    traj, by_frame, side: str, window: int, top_k: int, skip: int = 0
):
    """Pick up to top_k frames (by visibility) from the terminal `window` frames,
    skipping the `skip` outermost ones (track death/birth frames are usually the
    most occluded — the occlusion is *why* the track died).

    side='lost' scans the last frames, side='cand' the first frames.  Returns
    [(frame, box, occ_mask, vis), ...] sorted by visibility desc."""
    if side == "lost":
        seg = traj[-(window + skip) : -skip] if skip else traj[-window:]
    else:
        seg = traj[skip : skip + window]
    if not seg:  # track shorter than skip — fall back to unskipped terminal frames
        seg = traj[-window:] if side == "lost" else traj[:window]
    scored = []
    for frm, x, y, w, h in seg:
        others = [
            (ox, oy, ow, oh)
            for tid, ox, oy, ow, oh in by_frame.get(frm, ())
            if (ox, oy, ow, oh) != (x, y, w, h)
        ]
        m, vis = occluder_mask((x, y, w, h), others)
        scored.append((vis, frm, (x, y, w, h), m))
    scored.sort(key=lambda t: -t[0])
    return [(frm, box, m, vis) for vis, frm, box, m in scored[:top_k]]


# ── histogram features ────────────────────────────────────────────────────────
def _center_weight():
    xs = (np.arange(CROP_W) - (CROP_W - 1) / 2.0) / (CROP_W / 4.0)
    return np.exp(-0.5 * xs**2)[None, :].repeat(CROP_H, axis=0)


CENTER_W = _center_weight()


NBINS = H_BINS * S_BINS + V_BINS


def _bin_idx(hsv):
    """Per-pixel histogram bin index (chromatic 16Hx4S, achromatic 4V)."""
    hue = hsv[..., 0].astype(np.int32) * H_BINS // 180
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.int32) * V_BINS // 256
    np.clip(hue, 0, H_BINS - 1, out=hue)
    np.clip(val, 0, V_BINS - 1, out=val)
    sbin = np.clip((sat * S_BINS).astype(np.int32), 0, S_BINS - 1)
    return np.where(sat >= ACHROMATIC_S, hue * S_BINS + sbin, H_BINS * S_BINS + val)


def hsv_hist(hsv, weight):
    """68-dim L1-normalised histogram: 16Hx4S chromatic + 4V achromatic."""
    feat = np.bincount(
        _bin_idx(hsv).ravel(), weights=weight.ravel(), minlength=NBINS
    ).astype(np.float64)
    s = feat.sum()
    return feat / s if s > 1e-9 else feat


def endpoint_features(img, box, occ):
    """Return dict of histograms for one endpoint sample, or None if degenerate."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x0, y0 = int(max(x, 0)), int(max(y, 0))
    x1, y1 = int(min(x + w, W)), int(min(y + h, H))
    if x1 - x0 < 8 or y1 - y0 < 16:
        return None
    crop = cv2.resize(img[y0:y1, x0:x1], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    vis_w = CENTER_W * (~occ)
    t0, t1 = int(TORSO[0] * CROP_H), int(TORSO[1] * CROP_H)
    l0, l1 = int(LEGS[0] * CROP_H), int(LEGS[1] * CROP_H)

    # blur -> hard center band -> pool (user-proposed variant): low-pass kills
    # texture/alignment noise, the hard x-crop excludes background columns.
    blur = cv2.GaussianBlur(crop, (BLUR_K, BLUR_K), 0)
    hsv_b = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab).astype(np.float32)
    xc0, xc1 = int(POOL_CX[0] * CROP_W), int(POOL_CX[1] * CROP_W)
    vis_f = (~occ).astype(np.float64)

    def _pool(y0, y1, w):
        ww = w[y0:y1, xc0:xc1]
        s = ww.sum()
        if s < 1e-6:
            return np.zeros(3, dtype=np.float64)
        return (lab[y0:y1, xc0:xc1] * ww[..., None]).sum((0, 1)) / s

    ones = np.ones((CROP_H, CROP_W))

    # background-ring discriminative weighting (CSR-DCF style): colors that also
    # appear just outside the box are background even inside the center band
    # (head sides, between legs, loose boxes).  w_fg(bin) = fg/(fg+bg).
    rx0, rx1 = int(max(x - 0.3 * w, 0)), int(min(x + 1.3 * w, W))
    ry0, ry1 = int(max(y - 0.1 * h, 0)), int(min(y + 1.1 * h, H))
    ring = img[ry0:ry1, rx0:rx1]
    ring_mask = np.ones(ring.shape[:2], dtype=bool)
    ring_mask[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = False
    bg = np.zeros(NBINS, dtype=np.float64)
    if ring_mask.any():
        hsv_r = cv2.cvtColor(
            cv2.GaussianBlur(ring, (BLUR_K, BLUR_K), 0), cv2.COLOR_BGR2HSV
        )
        bg = np.bincount(_bin_idx(hsv_r)[ring_mask], minlength=NBINS).astype(np.float64)
        bg /= max(bg.sum(), 1e-9)
    idx_b = _bin_idx(hsv_b)
    fg = np.bincount(idx_b[:, xc0:xc1].ravel(), minlength=NBINS).astype(np.float64)
    fg /= max(fg.sum(), 1e-9)
    w_fg = (fg / (fg + bg + 1e-12))[idx_b]
    ww = w_fg[:, xc0:xc1]
    vprof_bg = (lab[:, xc0:xc1] * ww[..., None]).sum(1) / np.maximum(ww.sum(1), 1e-6)[
        :, None
    ]

    return {
        "full_nomask": hsv_hist(hsv, CENTER_W),
        "torso_nomask": hsv_hist(hsv[t0:t1], CENTER_W[t0:t1]),
        "legs_nomask": hsv_hist(hsv[l0:l1], CENTER_W[l0:l1]),
        "full_mask": hsv_hist(hsv, vis_w),
        "torso_mask": hsv_hist(hsv[t0:t1], vis_w[t0:t1]),
        "legs_mask": hsv_hist(hsv[l0:l1], vis_w[l0:l1]),
        "histcb_torso": hsv_hist(hsv_b[t0:t1, xc0:xc1], CENTER_W[t0:t1, xc0:xc1]),
        "histcb_legs": hsv_hist(hsv_b[l0:l1, xc0:xc1], CENTER_W[l0:l1, xc0:xc1]),
        "pool_torso": _pool(t0, t1, ones),
        "pool_legs": _pool(l0, l1, ones),
        "pool_torso_mask": _pool(t0, t1, vis_f),
        "pool_legs_mask": _pool(l0, l1, vis_f),
        "pool_torso_bg": _pool(t0, t1, w_fg),
        "pool_legs_bg": _pool(l0, l1, w_fg),
        # blurred center-band vertical color profile (CROP_H, 3) — substrate for
        # translation-robust comparisons (FFT magnitude / best-shift alignment)
        "vprof": lab[:, xc0:xc1].mean(axis=1),
        "vprof_bg": vprof_bg,
    }


def bhatta(p, q):
    return float(np.sqrt(p * q).sum())


def _lab_sim(a, b, chroma_only=False):
    """Negative L2 distance in (blurred, pooled) Lab space — monotone for AUC."""
    d = a - b
    if chroma_only:
        d = d[1:]
    return -float(np.linalg.norm(d))


def _fft_sim(pa, pb, k=16):
    """Translation-invariant profile sim: -L2 between |rFFT| of the vertical
    Lab profiles, first k frequencies (incl. DC = mean color). Shift-invariant
    but phase-blind (loses which color is on top)."""
    fa = np.abs(np.fft.rfft(pa, axis=0))[:k]
    fb = np.abs(np.fft.rfft(pb, axis=0))[:k]
    return -float(np.linalg.norm(fa - fb))


def _shift_sim(pa, pb, max_shift=16):
    """Best-shift alignment sim: slide the profiles vertically, score each shift
    by mean per-row Lab distance over the overlap, keep the best. Keeps phase
    (color order), absorbs bbox jitter/truncation up to max_shift rows."""
    H = pa.shape[0]
    best = -1e18
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a, b = pa[s:], pb[: H - s]
        else:
            a, b = pa[: H + s], pb[-s:]
        d = float(np.linalg.norm(a - b, axis=1).mean())
        if -d > best:
            best = -d
    return best


def pair_sims(fa, fb):
    """Similarity columns for a (lost, cand) endpoint feature pair."""
    return {
        "color_fftmag": _fft_sim(fa["vprof"], fb["vprof"]),
        "color_shiftalign": _shift_sim(fa["vprof"], fb["vprof"]),
        "color_shiftalign_bgsub": _shift_sim(fa["vprof_bg"], fb["vprof_bg"]),
        "color_pool_bgsub": 0.5
        * (
            _lab_sim(fa["pool_torso_bg"], fb["pool_torso_bg"])
            + _lab_sim(fa["pool_legs_bg"], fb["pool_legs_bg"])
        ),
        "color_full_nomask": bhatta(fa["full_nomask"], fb["full_nomask"]),
        "color_split_nomask": 0.5
        * (
            bhatta(fa["torso_nomask"], fb["torso_nomask"])
            + bhatta(fa["legs_nomask"], fb["legs_nomask"])
        ),
        "color_full_mask": bhatta(fa["full_mask"], fb["full_mask"]),
        "color_split_mask": 0.5
        * (
            bhatta(fa["torso_mask"], fb["torso_mask"])
            + bhatta(fa["legs_mask"], fb["legs_mask"])
        ),
        "color_histcb_split": 0.5
        * (
            bhatta(fa["histcb_torso"], fb["histcb_torso"])
            + bhatta(fa["histcb_legs"], fb["histcb_legs"])
        ),
        "color_pool_center": 0.5
        * (
            _lab_sim(fa["pool_torso"], fb["pool_torso"])
            + _lab_sim(fa["pool_legs"], fb["pool_legs"])
        ),
        "color_pool_ab": 0.5
        * (
            _lab_sim(fa["pool_torso"], fb["pool_torso"], True)
            + _lab_sim(fa["pool_legs"], fb["pool_legs"], True)
        ),
        "color_pool_mask": 0.5
        * (
            _lab_sim(fa["pool_torso_mask"], fb["pool_torso_mask"])
            + _lab_sim(fa["pool_legs_mask"], fb["pool_legs_mask"])
        ),
    }


SIM_COLS = [
    "color_full_nomask",
    "color_split_nomask",
    "color_full_mask",
    "color_split_mask",
    "color_histcb_split",
    "color_pool_center",
    "color_pool_ab",
    "color_pool_mask",
    "color_fftmag",
    "color_shiftalign",
    "color_shiftalign_bgsub",
    "color_pool_bgsub",
]


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--csv", type=Path, default=Path("scripts/tools/out/relink_candidates.csv")
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument(
        "--out", type=Path, default=Path("scripts/tools/out/color_relink_features.csv")
    )
    ap.add_argument(
        "--window",
        type=int,
        default=8,
        help="terminal frames scanned per endpoint for the visibility pick",
    )
    ap.add_argument(
        "--skip",
        type=int,
        default=0,
        help="terminal frames skipped before the window (avoid the "
        "occlusion that killed/birthed the track)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=1,
        help="aggregate (mean) histograms over the top-k visible frames",
    )
    ap.add_argument(
        "--cand-skip",
        type=int,
        default=None,
        help="override --skip for the candidate side (delayed matching: "
        "wait N frames after birth before sampling)",
    )
    ap.add_argument(
        "--cand-window",
        type=int,
        default=None,
        help="override --window for the candidate side",
    )
    ap.add_argument(
        "--hard-dist",
        type=float,
        default=1.0,
        help="bridge_dist threshold defining the hard pool",
    )
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    print(f"{len(rows)} candidate pairs from {args.csv}")

    # endpoints needed per sequence: (id, side)
    need: dict[str, set] = defaultdict(set)
    for r in rows:
        need[r["seq"]].add((int(r["lost_id"]), "lost"))
        need[r["seq"]].add((int(r["cand_id"]), "cand"))

    feats: dict[tuple, dict] = {}  # (seq, id, side) -> hist dict
    meta: dict[tuple, tuple] = {}  # (seq, id, side) -> (vis, frame)
    for seq, endpoints in sorted(need.items()):
        by_id, by_frame = load_boxes(args.mot_dir / f"{seq}.txt")
        # choose sample frames first (boxes only), then load each image once
        plan: dict[int, list] = defaultdict(list)  # frame -> [(key, box, occ, vis)]
        c_skip = args.skip if args.cand_skip is None else args.cand_skip
        c_win = args.window if args.cand_window is None else args.cand_window
        for tid, side in endpoints:
            traj = by_id.get(tid)
            if not traj:
                continue
            win, skp = (c_win, c_skip) if side == "cand" else (args.window, args.skip)
            for frm, box, occ, vis in pick_sample_frames(
                traj, by_frame, side, win, args.topk, skp
            ):
                plan[frm].append(((seq, tid, side), box, occ, vis))
        img_dir = args.img_root / seq / "img1"
        samples: dict[tuple, list] = defaultdict(list)  # key -> [(hists, vis)]
        for frm in sorted(plan):
            img = cv2.imread(str(img_dir / f"{frm:06d}.jpg"))
            if img is None:
                continue
            for key, box, occ, vis in plan[frm]:
                f = endpoint_features(img, box, occ)
                if f is not None:
                    samples[key].append((f, vis))
        for key, lst in samples.items():
            feats[key] = {k: np.mean([f[k] for f, _ in lst], axis=0) for k in lst[0][0]}
            meta[key] = (max(v for _, v in lst), len(lst))
        print(f"  {seq:<16} endpoints {len(samples)}/{len(endpoints)}")

    # pair similarities
    n_valid = 0
    for r in rows:
        seq = r["seq"]
        ka, kb = (seq, int(r["lost_id"]), "lost"), (seq, int(r["cand_id"]), "cand")
        fa, fb = feats.get(ka), feats.get(kb)
        if fa is None or fb is None:
            for c in SIM_COLS:
                r[c] = ""
            r["vis_lost"] = r["vis_cand"] = ""
            continue
        r.update({k: f"{v:.6f}" for k, v in pair_sims(fa, fb).items()})
        r["vis_lost"] = f"{meta[ka][0]:.4f}"
        r["vis_cand"] = f"{meta[kb][0]:.4f}"
        n_valid += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows ({n_valid} with color features) -> {args.out}")

    # ── AUC report ────────────────────────────────────────────────────────────
    ok = [r for r in rows if r["gt_valid"] == "1" and r[SIM_COLS[0]] != ""]
    y = np.array([int(r["gt_match"]) for r in ok])
    bd = np.array([float(r["bridge_dist"]) for r in ok])
    hard = bd <= args.hard_dist
    print(f"\nPool : {len(y)} pairs | {int(y.sum())} pos ({100 * y.mean():.1f}%)")
    print(
        f"Hard (bridge_dist <= {args.hard_dist}h): {int(hard.sum())} pairs | "
        f"{int(y[hard].sum())} pos ({100 * y[hard].mean():.1f}%)"
    )

    print(
        f"\n{'variant':<22} {'AUC full':>9} {'AUC hard':>9} {'AP full':>8} {'AP hard':>8}"
    )
    print(
        f"{'-bridge_dist (geom)':<22} {_auc(-bd, y):>9.4f} {_auc(-bd[hard], y[hard]):>9.4f} "
        f"{_ap(-bd, y):>8.4f} {_ap(-bd[hard], y[hard]):>8.4f}"
    )
    for c in SIM_COLS:
        sc = np.array([float(r[c]) for r in ok])
        print(
            f"{c:<22} {_auc(sc, y):>9.4f} {_auc(sc[hard], y[hard]):>9.4f} "
            f"{_ap(sc, y):>8.4f} {_ap(sc[hard], y[hard]):>8.4f}"
        )

    # best color variant by hard AUC, stratified by gap
    best = max(
        SIM_COLS, key=lambda c: _auc(np.array([float(r[c]) for r in ok])[hard], y[hard])
    )
    sc = np.array([float(r[best]) for r in ok])
    gap = np.array([int(r["gap"]) for r in ok])
    print(f"\n── {best}: hard-pool AUC by gap bucket ──")
    for glo, ghi in [(1, 10), (10, 30), (30, 80), (80, 301)]:
        m = hard & (gap >= glo) & (gap < ghi)
        if m.sum() and y[m].sum() and (1 - y[m]).sum():
            print(
                f"  gap [{glo:>3},{ghi:>3}): {int(m.sum()):>5} pairs, "
                f"{int(y[m].sum()):>4} pos, AUC {_auc(sc[m], y[m]):.4f}"
            )
        else:
            print(f"  gap [{glo:>3},{ghi:>3}): {int(m.sum()):>5} pairs (degenerate)")

    # visibility-conditioned: both endpoints sampled clean
    vis_l = np.array([float(r["vis_lost"]) for r in ok])
    vis_c = np.array([float(r["vis_cand"]) for r in ok])
    print(f"\n── {best}: hard-pool AUC by endpoint visibility ──")
    for thr in (0.0, 0.7, 0.9):
        m = hard & (vis_l >= thr) & (vis_c >= thr)
        if m.sum() and y[m].sum() and (1 - y[m]).sum():
            print(
                f"  vis >= {thr:.1f}: {int(m.sum()):>5} pairs, {int(y[m].sum()):>4} pos, "
                f"AUC {_auc(sc[m], y[m]):.4f}"
            )


if __name__ == "__main__":
    main()
