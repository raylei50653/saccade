#!/usr/bin/env python3
"""Pre-loss motion statistics (GMC-compensated): turning magnitude + box-area change.

For every track that is lost mid-sequence (last frame < seqLength) in the
relink-OFF / interp-OFF MOT dump, measure — in a camera-motion-stabilized world
frame — (a) how sharply the object turns in its final W frames and (b) how its
box area changes over that window.  Camera pan/rotation/zoom is removed first by
composing the per-frame GMC affine (SparseOpticalFlowGMC, the same LK partial-
affine used by the tracker fallback) into a world->frame transform and back-
projecting each box.

Outputs charts (PNG) + printed percentiles.

Usage:
  .venv/bin/python scripts/tools/analyze_preloss_motion.py \
      --mot-dir results/MOT17_eval --img-root datasets/MOT17/train \
      --window 6 --out scripts/tools/out/preloss_motion
"""
# status: stable

from __future__ import annotations

import argparse
import configparser
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from saccade.perception.eval.gmc import SparseOpticalFlowGMC  # noqa: E402

DEFAULT_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]


def load_boxes(path: Path):
    """{id: [(frame, x, y, w, h), ...]} sorted by frame."""
    tr: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            tr[int(p[1])].append(
                (int(p[0]), float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            )
    for t in tr:
        tr[t].sort(key=lambda r: r[0])
    return dict(tr)


def compose(b, a):
    """Affine b∘a (apply a then b). Each is 2x3 [A|t]."""
    Aa, ta = a[:, :2], a[:, 2]
    Ab, tb = b[:, :2], b[:, 2]
    out = np.zeros((2, 3), np.float64)
    out[:, :2] = Ab @ Aa
    out[:, 2] = Ab @ ta + tb
    return out


def invert(m):
    A, t = m[:, :2], m[:, 2]
    Ai = np.linalg.inv(A)
    out = np.zeros((2, 3), np.float64)
    out[:, :2] = Ai
    out[:, 2] = -Ai @ t
    return out


def estimate_gmc_chain(img_dir: Path, n_frames: int, downscale: int):
    """Return {frame: C_f} world(frame1)->frame_f affine, plus identity for f=1."""
    gmc = SparseOpticalFlowGMC(downscale=downscale)
    C = {1: np.array([[1.0, 0, 0], [0, 1.0, 0]])}
    cur = C[1].copy()
    for f in range(1, n_frames + 1):
        img = cv2.imread(str(img_dir / f"{f:06d}.jpg"))
        if img is None:
            # keep chain going with identity step
            if f >= 2:
                C[f] = cur.copy()
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        H = gmc.estimate(t)  # prev->curr, full-res px; None on first/failure
        if f == 1:
            continue
        Hf = (
            H.cpu().numpy().astype(np.float64)
            if H is not None
            else np.array([[1.0, 0, 0], [0, 1.0, 0]])
        )
        cur = compose(Hf, cur)  # C_f = H_f ∘ C_{f-1}
        C[f] = cur.copy()
    return C


def world_point(C, frame, x, y):
    Ci = invert(C[frame])
    return Ci[:, :2] @ np.array([x, y]) + Ci[:, 2]


def world_area_scale(C, frame):
    return abs(np.linalg.det(C[frame][:, :2]))  # image_area = scale * world_area


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument("--img-root", type=Path, default=Path("datasets/MOT17/train"))
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument(
        "--window",
        type=int,
        default=8,
        help="pre-loss window: require this many CONSECUTIVE frames",
    )
    ap.add_argument("--downscale", type=int, default=8, help="GMC downscale")
    ap.add_argument(
        "--end-margin",
        type=int,
        default=2,
        help="track is 'lost' if last_frame <= seqLength - margin",
    )
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument(
        "--out", type=Path, default=Path("scripts/tools/out/preloss_motion")
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    W = args.window

    turn_net, turn_max, area_rate, area_total = [], [], [], []
    n_lost = 0
    for seq in args.seqs:
        mot = args.mot_dir / f"{seq}.txt"
        seqdir = args.img_root / seq
        if not mot.exists():
            print(f"{seq}: SKIP (no {mot})")
            continue
        ini = configparser.ConfigParser()
        ini.read(seqdir / "seqinfo.ini")
        n_frames = ini.getint("Sequence", "seqLength")
        print(f"{seq}: estimating GMC over {n_frames} frames ...", flush=True)
        C = estimate_gmc_chain(seqdir / "img1", n_frames, args.downscale)

        tracks = load_boxes(mot)
        seq_lost = 0
        for tid, tr in tracks.items():
            if len(tr) < args.min_len:
                continue
            last_f = tr[-1][0]
            if last_f > n_frames - args.end_margin:  # not lost; ran to end
                continue
            # require W CONSECUTIVE frames ending at loss (no gaps)
            want = list(range(last_f - W + 1, last_f + 1))
            by_f = {r[0]: r for r in tr}
            if not all(f in by_f and f in C for f in want):
                continue
            win = [by_f[f] for f in want]
            seq_lost += 1
            # world-frame foot points + world areas over window
            pts, areas = [], []
            for f, x, y, w, h in win:
                fx, fy = x + w / 2.0, y + h  # foot
                pts.append(world_point(C, f, fx, fy))
                areas.append((w * h) / max(world_area_scale(C, f), 1e-9))
            pts = np.array(pts)
            # per-step velocities + turning angles (deg)
            vels = np.diff(pts, axis=0)
            angs = []
            for i in range(1, len(vels)):
                a, b = vels[i - 1], vels[i]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-3 or nb < 1e-3:
                    continue
                cosv = np.clip(np.dot(a, b) / (na * nb), -1, 1)
                angs.append(np.degrees(np.arccos(cosv)))
            if angs:
                turn_max.append(max(angs))
                # net heading change: first vs last valid velocity
                v0, v1 = vels[0], vels[-1]
                if np.linalg.norm(v0) > 1e-3 and np.linalg.norm(v1) > 1e-3:
                    c = np.clip(
                        np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)),
                        -1,
                        1,
                    )
                    turn_net.append(np.degrees(np.arccos(c)))
            # area change RATE over the consecutive window: log-linear fit
            # slope = log2(area) per frame -> per-frame ratio = 2**slope
            a = np.clip(np.array(areas), 1e-9, None)
            idx = np.arange(len(a))
            slope = np.polyfit(idx, np.log2(a), 1)[0]
            area_rate.append((2.0**slope - 1.0) * 100.0)  # % per frame
            area_total.append(2.0 ** (slope * (len(a) - 1)))  # ratio over window
        n_lost += seq_lost
        print(f"  lost tracks analysed: {seq_lost}")

    turn_net = np.array(turn_net)
    turn_max = np.array(turn_max)
    area_rate = np.array(area_rate)
    area_total = np.array(area_total)

    def pct(name, a, fmt="{:.2f}"):
        if len(a) == 0:
            print(f"  {name}: (empty)")
            return
        qs = np.percentile(a, [10, 25, 50, 75, 90, 95])
        print(
            f"  {name:<22} n={len(a):>4}  "
            + " ".join(
                f"p{p}=" + fmt.format(v) for p, v in zip([10, 25, 50, 75, 90, 95], qs)
            )
        )

    print(
        f"\n=== Pre-loss motion (GMC-compensated, {W} consecutive frames) "
        f"over {n_lost} lost tracks ==="
    )
    pct("turn_net (deg)", turn_net)
    pct("turn_max/step (deg)", turn_max)
    pct("area rate (%/frame)", area_rate, fmt="{:+.2f}")
    pct(f"area total (x over {W}f)", area_total)
    print(f"  |area rate| median = {np.median(np.abs(area_rate)):.2f} %/frame")

    # ── charts ──
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].hist(turn_net, bins=36, range=(0, 180), color="#4C72B0")
    ax[0, 0].axvline(
        np.median(turn_net),
        color="r",
        ls="--",
        label=f"median {np.median(turn_net):.0f}°",
    )
    ax[0, 0].set(
        title="Net heading change before loss", xlabel="degrees", ylabel="lost tracks"
    )
    ax[0, 0].legend()

    ax[0, 1].hist(turn_max, bins=36, range=(0, 180), color="#DD8452")
    ax[0, 1].axvline(
        np.median(turn_max),
        color="r",
        ls="--",
        label=f"median {np.median(turn_max):.0f}°",
    )
    ax[0, 1].set(
        title="Max per-step turn before loss", xlabel="degrees", ylabel="lost tracks"
    )
    ax[0, 1].legend()

    rate_c = np.clip(area_rate, -20, 20)
    ax[1, 0].hist(rate_c, bins=41, range=(-20, 20), color="#55A868")
    ax[1, 0].axvline(0, color="k", lw=0.8)
    ax[1, 0].axvline(
        np.median(area_rate),
        color="r",
        ls="--",
        label=f"median {np.median(area_rate):+.1f} %/f",
    )
    ax[1, 0].set(
        title=f"Box area change RATE ({W} consecutive frames)",
        xlabel="% per frame  [+grow / -shrink]",
        ylabel="lost tracks",
    )
    ax[1, 0].legend()

    lt = np.log2(np.clip(area_total, 1e-3, 1e3))
    ax[1, 1].hist(lt, bins=40, range=(-2, 2), color="#C44E52")
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].set(
        title=f"Total area change over {W} frames (log2)",
        xlabel="log2(area_last / area_first)",
        ylabel="lost tracks",
    )

    fig.suptitle(
        f"Pre-loss motion, GMC-compensated (window={W}, n={n_lost} lost tracks)"
    )
    fig.tight_layout()
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"\nWrote chart -> {png}")


if __name__ == "__main__":
    main()
