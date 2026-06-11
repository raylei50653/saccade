#!/usr/bin/env python3
"""Gap-occupancy (exclusion) features for relink candidates.

Tests a constraint *orthogonal to pair geometry/appearance*: during the gap,
walk the straight bridge path between the lost exit foot and the candidate
entry foot, and measure how other tracked boxes occupy that path.

Two competing semantics — the data decides the sign:
  veto    : path occupied by another track => the candidate is someone else
  explain : path occupied => the disappearance is explained by occlusion,
            supporting the bridge (GT says true positives spend the gap at
            visibility ~0.18, i.e. behind someone)

Features per pair (sampled at <= --max-samples frames across the gap):
  occ_cover   fraction of samples where the path point lies inside >= 1 box
  occ_near    fraction of samples with a box foot within 0.5*h_ref
  occ_scale   fraction of samples covered by a box of similar height
              (0.6-1.6x h_ref) — a plausible occluder/person, not a far-off box

Reads the pair table produced by color_relink_features.py so combo tests
against geometry and color are possible; writes an augmented CSV and reports
AUC (full/hard), gap buckets, per-seq, and LOSO combo with -bridge_dist.

Usage:
  .venv/bin/python scripts/tools/gap_occupancy_features.py \
      --csv scripts/tools/out/color_relink_bgsub.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from color_relink_features import _ap, _auc, load_boxes  # noqa: E402

OCC_COLS = ["occ_cover", "occ_near", "occ_scale"]


def frame_arrays(by_frame):
    """{frame: (foot_x, foot_y, x0, y0, x1, y1, h) float arrays}."""
    out = {}
    for frm, lst in by_frame.items():
        a = np.array(
            [[x + w / 2.0, y + h, x, y, x + w, y + h, h] for _, x, y, w, h in lst],
            dtype=np.float64,
        )
        out[frm] = a
    return out


def pair_occupancy(row, frames, max_samples):
    ax, ay = float(row["lost_foot_x"]), float(row["lost_foot_y"])
    bx, by = float(row["cand_foot_x"]), float(row["cand_foot_y"])
    f0, f1 = int(float(row["lost_last_frame"])), int(float(row["cand_first_frame"]))
    h_ref = float(row["h_ref"])
    gap = f1 - f0
    ts = np.unique(
        np.linspace(f0 + 1, f1 - 1, min(max_samples, max(gap - 1, 1))).astype(int)
    )
    n_cov = n_near = n_scale = n = 0
    for t in ts:
        a = frames.get(int(t))
        if a is None:
            continue
        n += 1
        alpha = (t - f0) / gap
        px, py = ax + alpha * (bx - ax), ay + alpha * (by - ay)
        inside = (a[:, 2] <= px) & (px <= a[:, 4]) & (a[:, 3] <= py) & (py <= a[:, 5])
        if inside.any():
            n_cov += 1
            hr = a[inside, 6] / h_ref
            if ((hr >= 0.6) & (hr <= 1.6)).any():
                n_scale += 1
        d = np.hypot(a[:, 0] - px, a[:, 1] - py)
        if (d < 0.5 * h_ref).any():
            n_near += 1
    if n == 0:
        return None
    return {"occ_cover": n_cov / n, "occ_near": n_near / n, "occ_scale": n_scale / n}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--csv", type=Path, default=Path("scripts/tools/out/color_relink_bgsub.csv")
    )
    ap.add_argument("--mot-dir", type=Path, default=Path("results/MOT17_eval"))
    ap.add_argument(
        "--out", type=Path, default=Path("scripts/tools/out/gap_occupancy.csv")
    )
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--hard-dist", type=float, default=1.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    by_seq: dict[str, list] = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)

    for seq, seq_rows in sorted(by_seq.items()):
        _, by_frame = load_boxes(args.mot_dir / f"{seq}.txt")
        frames = frame_arrays(by_frame)
        n_ok = 0
        for r in seq_rows:
            occ = pair_occupancy(r, frames, args.max_samples)
            if occ is None:
                for c in OCC_COLS:
                    r[c] = ""
            else:
                r.update({k: f"{v:.4f}" for k, v in occ.items()})
                n_ok += 1
        print(f"  {seq:<16} {n_ok}/{len(seq_rows)} pairs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote -> {args.out}")

    ok = [r for r in rows if r["gt_valid"] == "1" and r.get("occ_cover", "") != ""]
    y = np.array([int(r["gt_match"]) for r in ok])
    bd = np.array([float(r["bridge_dist"]) for r in ok])
    gap = np.array([int(r["gap"]) for r in ok])
    seqs = np.array([r["seq"] for r in ok])
    hard = bd <= args.hard_dist
    print(
        f"\nPool {len(y)} ({int(y.sum())} pos) | hard {int(hard.sum())} "
        f"({int(y[hard].sum())} pos)"
    )
    print(
        f"\n{'feature':<14} {'AUC full':>9} {'AUC hard':>9} {'AP hard':>8} "
        f"{'pos μ':>7} {'neg μ':>7}  (hard pool means)"
    )
    print(
        f"{'-bridge_dist':<14} {_auc(-bd, y):>9.4f} {_auc(-bd[hard], y[hard]):>9.4f} "
        f"{_ap(-bd[hard], y[hard]):>8.4f}"
    )
    for c in OCC_COLS:
        s = np.array([float(r[c]) for r in ok])
        print(
            f"{c:<14} {_auc(s, y):>9.4f} {_auc(s[hard], y[hard]):>9.4f} "
            f"{_ap(s[hard], y[hard]):>8.4f} {s[hard & (y == 1)].mean():>7.3f} "
            f"{s[hard & (y == 0)].mean():>7.3f}"
        )

    best = max(
        OCC_COLS,
        key=lambda c: abs(
            _auc(np.array([float(r[c]) for r in ok])[hard], y[hard]) - 0.5
        ),
    )
    s = np.array([float(r[best]) for r in ok])
    print(f"\n── {best}: hard AUC by gap ──")
    for glo, ghi in [(1, 10), (10, 30), (30, 80), (80, 301)]:
        m = hard & (gap >= glo) & (gap < ghi)
        if y[m].sum() and (1 - y[m]).sum():
            print(
                f"  gap[{glo:>3},{ghi:>3}): {_auc(s[m], y[m]):.4f} "
                f"(n={int(m.sum())}, pos={int(y[m].sum())})"
            )

    print(f"\n── {best}: per-seq hard AUC ──")
    for sq in np.unique(seqs):
        m = hard & (seqs == sq)
        if y[m].sum() and (1 - y[m]).sum():
            print(f"  {sq}: {_auc(s[m], y[m]):.3f} (pos={int(y[m].sum())})")

    # independence: corr with bridge_dist among hard negatives + LOSO combo
    neg = hard & (y == 0)
    rho = np.corrcoef(s[neg], bd[neg])[0, 1]
    print(f"\nindependence: corr({best}, bridge_dist) on hard negatives = {rho:.3f}")
    zg = -(bd - bd.mean()) / bd.std()
    zs = (s - s.mean()) / s.std()
    ws = np.arange(-2, 2.01, 0.05)
    ha, hp, ba, bp = [], [], [], []
    for s_out in np.unique(seqs):
        tr, te = (seqs != s_out) & hard, (seqs == s_out) & hard
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        w = max(ws, key=lambda w_: _ap(zg[tr] + w_ * zs[tr], y[tr]))
        ha.append(_auc(zg[te] + w * zs[te], y[te]))
        hp.append(_ap(zg[te] + w * zs[te], y[te]))
        ba.append(_auc(zg[te], y[te]))
        bp.append(_ap(zg[te], y[te]))
    print(
        f"LOSO hard pool: combo AUC {np.mean(ha):.4f} AP {np.mean(hp):.4f} | "
        f"geom AUC {np.mean(ba):.4f} AP {np.mean(bp):.4f}"
    )


if __name__ == "__main__":
    main()
