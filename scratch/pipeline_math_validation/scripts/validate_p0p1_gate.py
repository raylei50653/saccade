"""Validate P0 (L_med height normalization) + P1 (scale gate) on the offline
relink candidate set before implementing them in the relinker.

Joins relink_candidates_mamba_nointerp.csv with per-track height series from
results/MOT17_mamba_whole_graph_nointerp/*.txt, then reports:
  - hard/full AUC for each normalization strategy (reproduce doc §5)
  - scale-gate recall / negative rejection / precision (doc §6)
  - combined gate + distance-threshold operating points (deployment preview)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "scripts/tools/out/relink_candidates_mamba_nointerp.csv"
RESULTS = ROOT / "results/MOT17_mamba_whole_graph_nointerp"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC; higher score = more positive."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s = pd.Series(scores)
    ranks = s.rank(method="average").to_numpy()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_tracks() -> dict[tuple[str, int], np.ndarray]:
    """(seq, tid) -> array[(frame, h)] sorted by frame."""
    out: dict[tuple[str, int], np.ndarray] = {}
    for txt in sorted(RESULTS.glob("MOT17-*.txt")):
        seq = txt.stem
        per_id: dict[int, list[tuple[int, float]]] = defaultdict(list)
        arr = np.loadtxt(txt, delimiter=",", usecols=(0, 1, 5))
        for fr, tid, h in arr:
            per_id[int(tid)].append((int(fr), float(h)))
        for tid, rows in per_id.items():
            rows.sort()
            out[(seq, tid)] = np.array(rows)
    return out


def median_skip(h: np.ndarray, skip_tail: int = 0, skip_head: int = 0) -> float:
    body = h[skip_head : len(h) - skip_tail] if len(h) > skip_head + skip_tail else h
    return float(np.median(body)) if len(body) else 1.0


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df.gt_valid == 1].reset_index(drop=True)
    tracks = load_tracks()

    h_lost_last = np.zeros(len(df))
    h_cand_first = np.zeros(len(df))
    med_l = np.zeros(len(df))
    med_c = np.zeros(len(df))
    missing = 0
    for i, r in df.iterrows():
        lk, ck = (r.seq, int(r.lost_id)), (r.seq, int(r.cand_id))
        if lk not in tracks or ck not in tracks:
            missing += 1
            continue
        lh = tracks[lk]
        ch = tracks[ck]
        lh = lh[lh[:, 0] <= r.lost_last_frame][:, 1]
        ch = ch[ch[:, 0] >= r.cand_first_frame][:, 1]
        if len(lh) == 0 or len(ch) == 0:
            missing += 1
            continue
        h_lost_last[i] = lh[-1]
        h_cand_first[i] = ch[0]
        med_l[i] = median_skip(lh, skip_tail=3)
        med_c[i] = median_skip(ch, skip_head=3)
    ok = h_lost_last > 0
    print(f"pairs={len(df)}  joined={ok.sum()}  missing={missing}")
    df = df[ok].reset_index(drop=True)
    h_lost_last, h_cand_first = h_lost_last[ok], h_cand_first[ok]
    med_l, med_c = med_l[ok], med_c[ok]

    y = df.gt_match.to_numpy()
    dist_px = np.hypot(df.lost_foot_x - df.cand_foot_x, df.lost_foot_y - df.cand_foot_y)
    hard = (df.bridge_dist <= 1.0).to_numpy()
    print(f"pos={int(y.sum())}  neg={int((1 - y).sum())}  hard pool={hard.sum()} ({int(y[hard].sum())} pos)")

    # --- normalization strategies (negate dist: lower dist = more positive) ---
    print("\n== normalization: full AUC / hard AUC ==")
    strategies = {
        "current csv dist_h": df.dist_h.to_numpy(),
        "avg(last,first)": dist_px / np.maximum((h_lost_last + h_cand_first) / 2, 1),
        "lost_last only": dist_px / np.maximum(h_lost_last, 1),
        "L_med skip3 (P0)": dist_px / np.maximum(med_l, 1),
    }
    for name, d in strategies.items():
        print(f"  {name:24s} {auc(-d, y):.4f} / {auc(-d[hard], y[hard]):.4f}")

    # --- scale gates ---
    print("\n== scale gates: recall / neg rejected / precision ==")
    ratio_old = h_lost_last / np.maximum(h_cand_first, 1)
    ratio_new = med_l / np.maximum(med_c, 1)
    base_prec = y.mean()
    for name, ratio, lo, hi in [
        ("old last/first [0.7,1.4]", ratio_old, 0.7, 1.4),
        ("old last/first [0.8,1.25]", ratio_old, 0.8, 1.25),
        ("new medL/medC [0.8,1.8]", ratio_new, 0.8, 1.8),
        ("new medL/medC [0.7,1.4]", ratio_new, 0.7, 1.4),
    ]:
        keep = (ratio >= lo) & (ratio <= hi)
        rec = y[keep].sum() / y.sum()
        rej = 1 - (keep & (y == 0)).sum() / (y == 0).sum()
        prec = y[keep].mean()
        print(f"  {name:28s} recall={rec:.3f}  neg_rej={rej:.3f}  prec {base_prec:.3f}->{prec:.3f}")

    # --- combined deployment preview: gate + L_med distance threshold ---
    print("\n== gate [0.7,1.4] (old) + L_med dist threshold: TP / FP ==")
    keep = (ratio_old >= 0.7) & (ratio_old <= 1.4)
    d_med = dist_px / np.maximum(med_l, 1)
    print(f"  currently accepted (deployed scorer): TP={int(df.accepted[y == 1].sum())} FP={int(df.accepted[y == 0].sum())}")
    for thr in (0.5, 1.0, 1.5, 2.0):
        acc = keep & (d_med <= thr)
        print(f"  thr={thr:4.1f}  TP={int(y[acc].sum()):3d}  FP={int((1 - y)[acc].sum()):5d}  prec={y[acc].mean() if acc.any() else 0:.3f}")
    print("\n  (no gate, L_med only)")
    for thr in (0.5, 1.0, 1.5, 2.0):
        acc = d_med <= thr
        print(f"  thr={thr:4.1f}  TP={int(y[acc].sum()):3d}  FP={int((1 - y)[acc].sum()):5d}  prec={y[acc].mean() if acc.any() else 0:.3f}")


if __name__ == "__main__":
    main()
