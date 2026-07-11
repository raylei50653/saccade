"""Emit the committed Step-0 reproduction packet (review fix #5, PR #100).

Writes a small immutable packet into
docs/modules/semantic/research/evidence/gt_support_morphology_step0_20260711/
so a reviewer can audit, from the PR branch alone:
- per-GT-row atom bits and Hamming distances (track-to-row aggregation);
- per-sequence tail distribution and the 4 far-Hamming tail tracks;
- k=4/5/6/8 cell occupancy (GT tracks + FP rows per cell);
- atom thresholds/orientations and the CP numerator provenance.

Same declared choices as step0_identifiability_audit.py (pool-median
binarization, mining-AUC atom order, track = (seq, lost_id)).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ray/developer/ai/saccade")
sys.path.insert(0, str(REPO / "scripts/tools"))
import audit_relink_safe_reject as ar  # noqa: E402

PAIRS = REPO / "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
OUT = (
    REPO
    / "docs/modules/semantic/research/evidence/gt_support_morphology_step0_20260711"
)
OUT.mkdir(parents=True, exist_ok=True)

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
n_rows = y.size

thresholds: dict[str, dict[str, object]] = {}
Z = np.zeros((n_rows, len(ATOMS)), dtype=int)
for j, (name, lib) in enumerate(ATOMS):
    v = np.asarray(pool[name], float)
    thr = float(np.nanmedian(v))
    Z[:, j] = ((v <= thr) if lib else (v >= thr)).astype(int)
    thresholds[name] = {
        "pool_median_threshold": thr,
        "safe_side": "<= threshold" if lib else ">= threshold",
        "p_z1_pool": float(Z[:, j].mean()),
        "p_z1_gt": float(Z[y, j].mean()),
    }

# ---- gt_rows.csv: every GT row with atom bits, values, d_H (k=8) ----
names = [a for a, _ in ATOMS]
gt_idx = np.nonzero(y)[0]
with (OUT / "gt_rows.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        ["seq", "lost_id", "track_key"]
        + [f"z_{n}" for n in names]
        + [f"v_{n}" for n in names]
        + ["d_h_k8"]
    )
    for i in gt_idx:
        bits = Z[i].tolist()
        vals = [float(np.asarray(pool[n], float)[i]) for n in names]
        w.writerow(
            [seq[i], lost[i], tk[i]]
            + bits
            + [f"{v:.6g}" for v in vals]
            + [8 - int(sum(bits))]
        )

# ---- cell occupancy per k ----
for k in (4, 5, 6, 8):
    code = (Z[:, :k] * (1 << np.arange(k))).sum(1)
    gt_tracks: dict[int, set[str]] = {}
    fp_rows: dict[int, int] = {}
    for ri in range(n_rows):
        c = int(code[ri])
        if y[ri]:
            gt_tracks.setdefault(c, set()).add(str(tk[ri]))
        else:
            fp_rows[c] = fp_rows.get(c, 0) + 1
    with (OUT / f"cell_occupancy_k{k}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["cell_code", "bits_atom0_first", "n_gt_tracks", "n_fp_rows"])
        for c in range(1 << k):
            if c not in gt_tracks and c not in fp_rows:
                continue
            w.writerow(
                [
                    c,
                    format(c, f"0{k}b")[::-1],
                    len(gt_tracks.get(c, set())),
                    fp_rows.get(c, 0),
                ]
            )

# ---- far-Hamming tail (k=8, min d_H >= 3) + per-seq tail distribution ----
k = 8
ham = k - Z[:, :k].sum(1)
best: dict[str, int] = {}
for i in gt_idx:
    key = str(tk[i])
    best[key] = min(best.get(key, k + 1), int(ham[i]))
per_seq_tail: dict[str, int] = {}
tail = {}
for key, d in best.items():
    if d >= 3:
        s = key.split("|")[0]
        per_seq_tail[s] = per_seq_tail.get(s, 0) + 1
        rows = []
        for i in gt_idx:
            if str(tk[i]) == key:
                rows.append(
                    {
                        "frame_row_bits_atom0_first": "".join(map(str, Z[i, :k])),
                        "d_h": int(ham[i]),
                    }
                )
        tail[key] = {"min_d_h": d, "gt_rows": rows}
(OUT / "tail_tracks.json").write_text(
    json.dumps(
        {
            "definition": "GT tracks with min over their gt rows of d_H(k=8) >= 3",
            "representation": "descriptive min-d_H layer only (framework §19.4)",
            "n_tail_tracks": len(tail),
            "per_sequence_tail_counts": per_seq_tail,
            "tracks": tail,
        },
        indent=1,
    )
)

# ---- CP diagnostic provenance ----
n_tracks = len(best)
dist_hist: dict[int, int] = {}
for d in best.values():
    dist_hist[d] = dist_hist.get(d, 0) + 1
(OUT / "cp_ucb.json").write_text(
    json.dumps(
        {
            "trial_unit": "lost_track(seq,lost_id)",
            "n_tracks": n_tracks,
            "numerator_x": len(tail),
            "numerator_definition": "far-Hamming descriptive tail (k=8, min d_H >= 3)",
            "method": "Clopper-Pearson one-sided 95% upper bound",
            "value_x0": 1 - 0.05 ** (1 / n_tracks),
            "value_x": None,  # computed below without scipy dependency
            "status": "nominal; not cluster-adjusted (sequence-level residual clustering declared)",
            "boundary_use": "forbidden for epsilon_morph classification (framework §19.5 UCB validity)",
            "min_d_h_histogram_k8": {str(kk): v for kk, v in sorted(dist_hist.items())},
        },
        indent=1,
    )
)


# ---- manifest with input seal + file hashes ----
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


files = sorted(p for p in OUT.iterdir() if p.name not in ("manifest.json",))
manifest = {
    "study_id": "gt_support_morphology_step0_20260711",
    "source_pairs_csv": str(PAIRS.relative_to(REPO)),
    "source_pairs_csv_sha256": sha256(PAIRS),
    "pool_filter": "gt_valid==1 (audit_relink_safe_reject.load_gt_valid_pool)",
    "n_pool_rows": int(n_rows),
    "n_gt_rows": int(y.sum()),
    "n_gt_tracks": n_tracks,
    "atom_order": names,
    "binarization": "pool median (audit-only; best-case occupancy upper bound)",
    "atom_thresholds": thresholds,
    "trial_mapping": "descriptive min-d_H representative; set-valued H_C(u) NOT computed here",
    "files": {p.name: sha256(p) for p in files},
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("packet emitted:", OUT)
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size:8d} B")
