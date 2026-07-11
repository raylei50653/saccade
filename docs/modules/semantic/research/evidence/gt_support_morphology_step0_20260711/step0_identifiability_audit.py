"""Step-0 identifiability audit for Boolean risk-landscape morphology.

Read-only counting probe. NOT a study artifact; no gate, no rule search.

Question: with GT exposure = unique lost tracks (framework §8.1 unit),
how much of the k-atom Boolean hypercube has enough support for a
morphology verdict at all (user's class-6 gate)?

Declared choices (audit-only, best-case occupancy):
- pool          = gt_valid pairs, 7-seq (load_gt_valid_pool)
- atoms         = 8 mined signal families, oriented so z_i=1 == safer side
- binarization  = pool median per signal (balance-maximizing => occupancy
                  numbers are an UPPER bound vs any real sealed threshold)
- GT cell count = unique (seq, lost_id) among gt_match rows in the cell
- atom order    = mining AUC ranking (declared, nested prefixes for k<8)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ray/developer/ai/saccade")
sys.path.insert(0, str(REPO / "scripts/tools"))
import audit_relink_safe_reject as ar  # noqa: E402

PAIRS = REPO / "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"

pool = ar.load_gt_valid_pool(PAIRS)
if "resid_mean" not in pool:
    pool["resid_mean"] = 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
ar.ensure_prod_proxy_scores(pool)

# (signal, lower_is_better) — orientation: z=1 means "safer" side
ATOMS = [
    ("score_m_bridge", True),
    ("bridge_dist", True),
    ("dist_h", True),
    ("log_h_ratio", True),
    ("resid_mean", True),
    ("dir_cos", False),
    ("speed_mismatch", True),
    ("gap", True),
]

y = pool["gt_match"].astype(bool)
seq = np.asarray(pool["seq"])
lost = np.asarray(pool["lost_id"], dtype=object)
n_rows = y.size
track_key = np.array([f"{s}|{lid}" for s, lid in zip(seq, lost)], dtype=object)

print(f"pool rows (gt_valid): {n_rows}")
print(f"GT rows: {int(y.sum())}  FP rows: {int((~y).sum())}")
print(f"GT unique lost tracks: {len(set(track_key[y]))}")
print(f"pooled GT0 95% UCB (rule of 3, track unit): {3 / len(set(track_key[y])):.3f}")
for s in sorted(set(seq)):
    m = y & (seq == s)
    print(f"  {s}: GT rows {int(m.sum()):4d}  tracks {len(set(track_key[m])):4d}")

Z = np.zeros((n_rows, len(ATOMS)), dtype=np.int64)
print("\natom binarization (pool median):")
for j, (name, lib) in enumerate(ATOMS):
    v = np.asarray(pool[name], dtype=float)
    thr = float(np.nanmedian(v))
    z = (v <= thr) if lib else (v >= thr)
    Z[:, j] = z.astype(np.int64)
    gt_safe = float(z[y].mean())
    print(
        f"  {name:16s} thr={thr:10.4f}  P(z=1)={z.mean():.3f}  P(z=1|GT)={gt_safe:.3f}"
    )


def audit_k(k: int, seq_filter: str | None = None) -> None:
    mrow = np.ones(n_rows, dtype=bool)
    tag = "pooled 7-seq"
    if seq_filter is not None:
        mrow = seq == seq_filter
        tag = seq_filter
    code = np.zeros(n_rows, dtype=np.int64)
    for j in range(k):
        code |= Z[:, j] << j
    n_cells = 1 << k

    gt_tracks: dict[int, set[str]] = {}
    fp_rows = np.zeros(n_cells, dtype=np.int64)
    occupied = set()
    for i in np.nonzero(mrow)[0]:
        c = int(code[i])
        occupied.add(c)
        if y[i]:
            gt_tracks.setdefault(c, set()).add(track_key[i])
        else:
            fp_rows[c] += 1

    gt_counts = np.zeros(n_cells, dtype=np.int64)
    for c, s in gt_tracks.items():
        gt_counts[c] = len(s)

    n_occ = len(occupied)
    n_gt_cells = int((gt_counts > 0).sum())
    thresholds = [1, 5, 10, 20, 30, 59]
    marks = {t: int((gt_counts >= t).sum()) for t in thresholds}
    top = np.argsort(-gt_counts)[:5]
    print(f"\n[k={k}] {tag}  cells={n_cells}")
    print(f"  occupied cells (any row): {n_occ}/{n_cells} ({n_occ / n_cells:.1%})")
    print(f"  cells with >=1 GT track: {n_gt_cells}/{n_cells}")
    for t in thresholds[1:]:
        ucb = 3 / t
        print(
            f"  cells with >={t:3d} GT tracks: {marks[t]:4d}"
            f"   (GT0 cell-level UCB <= {ucb:.2f})"
        )
    tot = gt_counts.sum()
    if tot:
        top_share = gt_counts[top].sum() / tot
        print(
            f"  GT track-cell mass: total={tot} "
            f"(tracks span multiple cells) top5 cells share={top_share:.1%}"
        )
        for c in top:
            if gt_counts[c] == 0:
                break
            bits = format(c, f"0{k}b")[::-1]
            print(
                f"    cell z={bits}  GT tracks={gt_counts[c]:4d}  "
                f"FP rows={fp_rows[c]:6d}"
            )


for k in (4, 5, 6, 8):
    audit_k(k)

print("\n--- weakest folds, k=5 ---")
for s in ("MOT17-04-SDP", "MOT17-09-SDP"):
    audit_k(5, seq_filter=s)
