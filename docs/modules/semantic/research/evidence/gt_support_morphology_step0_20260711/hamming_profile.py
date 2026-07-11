"""Step-0 companion: GT-mass Hamming profile + off-corner violation profile.

Same declared choices as step0_identifiability_audit.py (pool-median
binarization, mining-AUC atom order, track = (seq, lost_id)).
Output recorded in hamming_profile.txt.
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
pool["resid_mean"] = 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
ar.ensure_prod_proxy_scores(pool)

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
tk = np.array([f"{s}|{lid}" for s, lid in zip(seq, lost)], dtype=object)
Z = np.zeros((y.size, len(ATOMS)), dtype=int)
for j, (name, lib) in enumerate(ATOMS):
    v = np.asarray(pool[name], float)
    thr = float(np.nanmedian(v))
    Z[:, j] = ((v <= thr) if lib else (v >= thr)).astype(int)

for k in (5, 8):
    ham = k - Z[:, :k].sum(1)
    d: dict[str, int] = {}
    for i in np.nonzero(y)[0]:
        d[tk[i]] = min(d.get(tk[i], k + 1), int(ham[i]))
    dist = np.array(sorted(d.values()))
    print(
        f"[k={k}] GT tracks by min Hamming distance to all-safe corner (n={len(dist)}):"
    )
    for dd in range(k + 1):
        n = int((dist == dd).sum())
        if n:
            print(f"   d={dd}: {n:4d} tracks  ({n / len(dist):.1%})")
    fp = ~y
    fpd = {dd: int((ham[fp] == dd).sum()) for dd in range(k + 1)}
    print("   FP rows by cell distance:", fpd)
    far = sum(v for kk, v in fpd.items() if kk >= 3)
    print(f"   FP rows at d>=3: {far} ({far / int(fp.sum()):.1%} of all FP)")

k = 8
ham = k - Z[:, :k].sum(1)
names = [a for a, _ in ATOMS]
best: dict[str, tuple[int, np.ndarray]] = {}
for i in np.nonzero(y)[0]:
    key = tk[i]
    if key not in best or ham[i] < best[key][0]:
        best[key] = (int(ham[i]), Z[i, :k].copy())
viol = np.zeros(k)
cnt = 0
for _, (dd, z) in best.items():
    if dd >= 3:
        cnt += 1
        viol += 1 - z
print(f"\n[k=8] GT tracks with min-distance>=3: {cnt}")
for j, atom_name in enumerate(names):
    print(f"   atom violated: {atom_name:16s} {int(viol[j]):3d}/{cnt}")
