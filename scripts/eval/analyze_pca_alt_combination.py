#!/usr/bin/env python3
"""Offline follow-up to probe_private_continuation_assignment.py: the hand-picked
additive combination `cost + w*|gap| + w*|dx|` was monotonically NO-GO (best w=0,
n=2539). Before concluding gap_h/dx_norm have no place in the assignment
decision at all, try combination strategies the hand-picked formula couldn't
express:
  1. signed diffs (no abs()) -- maybe direction matters and abs() destroyed it
  2. data-fit logistic regression (lets weights/signs/scales be learned, not
     hand-set) on both signed-diff features and the full 6-raw-feature space
  3. per-feature standalone AUC (is there ANY separability in gap/dx alone,
     ignoring cost, that a combination could exploit?)

Reuses the already-collected results/occ_separability/pca_full7.npz -- no
re-run of detector inference needed.
"""
# status: experiment

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = np.load("/home/ray/developer/ai/saccade/results/occ_separability/pca_full7.npz")
labels = d["labels"]  # 1=A, 0=B
cost_a, cost_b = d["cost_a"], d["cost_b"]
gap_a, gap_b = d["gap_a"], d["gap_b"]
dx_a, dx_b = d["dx_a"], d["dx_b"]
n = len(labels)
print(f"n={n}  A={labels.sum()}  B={(1 - labels).sum()}")

# Baseline for reference (matches probe_private_continuation_assignment.py w=0).
baseline_acc = float(((cost_a < cost_b).astype(int) == labels).mean())
print(f"\nbaseline (argmin cost_iou) accuracy = {baseline_acc:.3f}")

cost_diff = cost_a - cost_b  # >0 means B is IoU-cheaper (favors B)
gap_diff = gap_a - gap_b
dx_diff = dx_a - dx_b
gap_absdiff = np.abs(gap_a) - np.abs(gap_b)
dx_absdiff = np.abs(dx_a) - np.abs(dx_b)

print("\n=== standalone AUC (does gap/dx alone separate A vs B, ignoring cost?) ===")
for name, feat in [
    (
        "cost_diff",
        -cost_diff,
    ),  # negate: more negative cost_a (favors A) -> should predict A=1
    ("gap_diff (signed)", -gap_diff),
    ("dx_diff (signed)", -dx_diff),
    ("gap_absdiff", -gap_absdiff),
    ("dx_absdiff", -dx_absdiff),
]:
    auc = roc_auc_score(labels, feat)
    print(f"  {name:22s} AUC={auc:.3f}")

print(
    "\n=== data-fit logistic regression (5-fold CV, no leakage-safe grouping -- optimistic upper bound) ==="
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

feature_sets = {
    "cost_diff only (baseline as classifier)": np.stack([cost_diff], axis=1),
    "cost_diff + signed gap_diff + dx_diff": np.stack(
        [cost_diff, gap_diff, dx_diff], axis=1
    ),
    "cost_diff + abs gap_diff + abs dx_diff": np.stack(
        [cost_diff, gap_absdiff, dx_absdiff], axis=1
    ),
    "all 6 raw (cost_a,b, gap_a,b, dx_a,b)": np.stack(
        [cost_a, cost_b, gap_a, gap_b, dx_a, dx_b], axis=1
    ),
}
for name, X in feature_sets.items():
    scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy")
    print(
        f"  {name:42s} acc={scores.mean():.3f} +/- {scores.std():.3f}  delta={scores.mean() - baseline_acc:+.3f}"
    )

# Fit on all data (not CV) just to inspect what sign/magnitude the model assigns
# to gap/dx once cost is already in the model -- diagnostic only.
clf_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
X_diag = np.stack([cost_diff, gap_diff, dx_diff], axis=1)
clf_full.fit(X_diag, labels)
coefs = clf_full.named_steps["logisticregression"].coef_[0]
print(f"\nfitted coefficients (standardized) [cost_diff, gap_diff, dx_diff] = {coefs}")
