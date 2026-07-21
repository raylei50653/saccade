#!/usr/bin/env python3
"""Energy transform separability audit (raw / log1p / sqrt / rank).

Critical trap
-------------
If a score is a single scalar used only for threshold / ranking,
AUC(energy) == AUC(log(energy)) for any strictly monotone transform.
Use AUC only for "is there ranking signal?", NOT for "raw vs log linear fit".

Three audit layers
------------------
1. Ranking: AUC, AP, KS, quantile gap, safe-negative-pruning curve
2. Linear margin: d', Fisher, Gaussian overlap, logistic logloss/Brier/ECE, coef stability
3. Diagnosis: rank_signal_only | raw_linear_good | log_linear_good | piecewise_needed | ...

Usage
-----
  uv run python scripts/tools/energy_transform_separability.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_energy_xform_<stamp> \\
    --all

Contract: docs/research/contracts/signal_table_schema.md §0.5
Ledger:   docs/research/eval/signal_analysis_ledger.md
"""
# status: diagnostic

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# ── pairs loader (reuse audit helpers) ───────────────────────────────────────
_AUDIT = Path(__file__).resolve().parent / "audit_relink_safe_reject.py"
_spec = importlib.util.spec_from_file_location("audit_relink_safe_reject", _AUDIT)
assert _spec and _spec.loader
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)

TRANSFORMS = ("raw", "log1p", "sqrt", "rank")
SLICES_GAP = [
    ("global", None),
    ("hard_pool", "hard"),
    ("gap_1-10", (1, 10)),
    ("gap_11-30", (11, 30)),
    ("gap_31-60", (31, 60)),
    ("gap_61-150", (61, 150)),
    ("gap_151-300", (151, 300)),
]


@dataclass(frozen=True)
class EnergySignal:
    signal_name: str
    """Raw energy array builder: pool -> nonnegative score (higher = more FP-like / worse for GT)."""
    extract: Callable[[dict[str, np.ndarray]], np.ndarray]
    """If True, lower raw energy is better for GT (distances). Ranking uses -energy for AUC."""
    lower_is_better: bool
    notes: str = ""


def default_catalog() -> list[EnergySignal]:
    def resid_mean(p: dict[str, np.ndarray]) -> np.ndarray:
        return 0.5 * (p["fwd_resid"] + p["bwd_resid"])

    return [
        EnergySignal(
            "score_m_bridge", lambda p: p["score_m_bridge"], True, "live-shaped bridge"
        ),
        EnergySignal(
            "bridge_dist", lambda p: p["bridge_dist"], True, "mid-point bridge"
        ),
        EnergySignal("dist_h", lambda p: p["dist_h"], True, "foot dist / h"),
        EnergySignal("resid_mean", resid_mean, True, "0.5*(fwd+bwd)"),
        EnergySignal(
            "speed_mismatch", lambda p: p["speed_mismatch"], True, "|exit-entry| speed"
        ),
        EnergySignal(
            "abs_log_h",
            lambda p: p["log_h_ratio"],
            True,
            "|log h_ratio| — already log-ish raw",
        ),
        EnergySignal(
            "abs_ratio_m1",
            lambda p: np.abs(p["h_ratio_lost_over_cand"] - 1.0),
            True,
            "|h_ratio - 1| linear deviation",
        ),
        EnergySignal(
            "neg_dir_cos",
            lambda p: -p["dir_cos"],  # higher = worse alignment for GT
            True,
            "−dir_cos so higher = more FP-like",
        ),
    ]


def apply_transform(
    energy: np.ndarray,
    transform: str,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Map energy to x for linear models. energy should be >=0 for log1p/sqrt."""
    e = np.asarray(energy, dtype=float)
    e = np.where(np.isfinite(e), e, 0.0)
    if transform == "raw":
        return e
    if transform == "log1p":
        # scale: typical median of positive mass for unit-ish log
        s = float(scale) if scale is not None and scale > 0 else 1.0
        return np.log1p(np.maximum(e, 0.0) / s)
    if transform == "sqrt":
        return np.sqrt(np.maximum(e, 0.0))
    if transform == "rank":
        # average ranks in [0,1]
        order = e.argsort(kind="mergesort")
        ranks = np.empty_like(e, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, num=e.size, endpoint=True)
        # ties: leave as-is (stable enough for audit)
        return ranks
    raise ValueError(f"unknown transform {transform}")


def _safe_auc(y: np.ndarray, score_higher_pos: np.ndarray) -> float | None:
    if y.sum() < 5 or (~y).sum() < 5:
        return None
    if float(np.nanstd(score_higher_pos)) < 1e-12:
        return None
    try:
        return float(roc_auc_score(y.astype(int), score_higher_pos))
    except ValueError:
        return None


def _safe_ap(y: np.ndarray, score_higher_pos: np.ndarray) -> float | None:
    if y.sum() < 5 or (~y).sum() < 5:
        return None
    try:
        return float(average_precision_score(y.astype(int), score_higher_pos))
    except ValueError:
        return None


def ks_distance(pos: np.ndarray, neg: np.ndarray) -> float:
    """Two-sample KS on empirical CDFs."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    xs = np.sort(np.unique(np.concatenate([pos, neg])))
    if xs.size == 0:
        return float("nan")
    # ECDFs
    cdf_p = np.searchsorted(np.sort(pos), xs, side="right") / pos.size
    cdf_n = np.searchsorted(np.sort(neg), xs, side="right") / neg.size
    return float(np.max(np.abs(cdf_p - cdf_n)))


def dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    """(mean_pos - mean_neg) / sqrt(0.5*(var_pos+var_neg)).

    For lower-is-better energy, pass x where GT should have LOWER mean than FP,
    so dprime is typically negative; we report signed and abs.
    """
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    vp, vn = float(pos.var()), float(neg.var())
    denom = math.sqrt(0.5 * (vp + vn) + 1e-12)
    return float((pos.mean() - neg.mean()) / denom)


def fisher_score(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    return float((pos.mean() - neg.mean()) ** 2 / (pos.var() + neg.var() + 1e-12))


def gaussian_overlap(pos: np.ndarray, neg: np.ndarray) -> float:
    """Approx overlap of two 1D Gaussians (0=sep, 1=identical). Bhattacharyya-based."""
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    m1, m2 = float(pos.mean()), float(neg.mean())
    v1, v2 = float(pos.var()) + 1e-12, float(neg.var()) + 1e-12
    # Bhattacharyya coefficient for univariate Gaussians
    bc = math.sqrt(2.0 * math.sqrt(v1 * v2) / (v1 + v2)) * math.exp(
        -0.25 * (m1 - m2) ** 2 / (v1 + v2)
    )
    return float(np.clip(bc, 0.0, 1.0))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = y.astype(float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def logistic_metrics(y: np.ndarray, x: np.ndarray) -> dict[str, float | None]:
    """1D logistic: P(pos) ~ sigmoid(a*x + b). Returns logloss, brier, ece, coef."""
    if y.sum() < 5 or (~y).sum() < 5:
        return {
            "logistic_logloss": None,
            "brier": None,
            "ece": None,
            "coef": None,
            "intercept": None,
        }
    X = x.reshape(-1, 1)
    # class weight balanced for rare pos
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
    )
    try:
        clf.fit(X, y.astype(int))
    except Exception:
        return {
            "logistic_logloss": None,
            "brier": None,
            "ece": None,
            "coef": None,
            "intercept": None,
        }
    proba = clf.predict_proba(X)[:, 1]
    # log_loss wants both classes
    try:
        ll = float(log_loss(y.astype(int), proba, labels=[0, 1]))
    except ValueError:
        ll = None
    try:
        br = float(brier_score_loss(y.astype(int), proba))
    except ValueError:
        br = None
    ece = expected_calibration_error(y.astype(int), proba)
    return {
        "logistic_logloss": ll,
        "brier": br,
        "ece": ece,
        "coef": float(clf.coef_.ravel()[0]),
        "intercept": float(clf.intercept_.ravel()[0]),
    }


def pruning_curve(
    y: np.ndarray,
    energy: np.ndarray,
    *,
    lower_is_better: bool,
    n_grid: int = 25,
) -> list[dict[str, float]]:
    """Safe-negative-pruning style: reject high energy (or low if flipped)."""
    # reject-score: higher => more reject
    rs = energy if lower_is_better else -energy
    # thr grid on reject-score quantiles
    qs = np.linspace(0.5, 0.99, n_grid)
    rows = []
    for q in qs:
        thr = float(np.quantile(rs, q))
        rej = rs >= thr
        gt = int(y.sum())
        fp = int((~y).sum())
        hurt = int((y & rej).sum())
        fprm = int((~y & rej).sum())
        rows.append(
            {
                "quantile": float(q),
                "thr": thr,
                "GT_hurt_rate": hurt / gt if gt else 0.0,
                "FP_removed_rate": fprm / fp if fp else 0.0,
            }
        )
    return rows


def slice_mask(
    pool: dict[str, np.ndarray], slice_id: str, gap_range: Any
) -> np.ndarray:
    n = pool["gt_match"].size
    if gap_range is None:
        return np.ones(n, dtype=bool)
    if gap_range == "hard":
        return pool["bridge_dist"] <= 1.0
    lo, hi = gap_range
    g = pool["gap"]
    return (g >= lo) & (g <= hi)


def diagnose_row(rows_for_signal_slice: list[dict[str, Any]]) -> str:
    """Pick best_transform + diagnosis from transform rows on one slice."""
    by_t = {r["transform"]: r for r in rows_for_signal_slice}
    aucs = [r["auc"] for r in rows_for_signal_slice if r["auc"] is not None]
    if not aucs or max(aucs) < 0.58:
        return "no_signal"

    # ranking strength from raw (same for monotone family on distances)
    auc_raw = by_t.get("raw", {}).get("auc")
    if auc_raw is None:
        auc_raw = max(aucs)

    def lin_quality(r: dict[str, Any]) -> float:
        # higher better linear margin: abs dprime + fisher - logloss
        dp = (
            abs(r["dprime"])
            if r["dprime"] is not None and np.isfinite(r["dprime"])
            else 0.0
        )
        fi = (
            r["fisher"] if r["fisher"] is not None and np.isfinite(r["fisher"]) else 0.0
        )
        ll = r["logistic_logloss"]
        ll_term = (1.0 / (1.0 + ll)) if ll is not None else 0.0
        return dp + fi + ll_term

    # among raw/log1p/sqrt (not rank)
    cand = [
        r for r in rows_for_signal_slice if r["transform"] in ("raw", "log1p", "sqrt")
    ]
    if not cand:
        return "no_signal"
    best = max(cand, key=lin_quality)
    rank_r = by_t.get("rank")
    best_dp = abs(best.get("dprime") or 0.0)

    # weak ranking + weak margin
    if auc_raw is not None and auc_raw < 0.65 and best_dp < 0.55:
        return "no_signal" if auc_raw < 0.60 else "rank_signal_only"

    # weak linear margin despite ranking
    if auc_raw is not None and auc_raw >= 0.70 and best_dp < 0.55:
        if (
            rank_r
            and rank_r.get("logistic_logloss") is not None
            and best.get("logistic_logloss")
        ):
            if rank_r["logistic_logloss"] + 0.05 < best["logistic_logloss"]:
                return "piecewise_needed"
        return "rank_signal_only"

    if (
        rank_r
        and auc_raw
        and auc_raw >= 0.70
        and best_dp < 0.8
        and abs(rank_r.get("dprime") or 0) > best_dp * 1.3
    ):
        return "piecewise_needed"

    # compare raw vs log (and sqrt as compressive family with log)
    raw_r = by_t.get("raw")
    log_r = by_t.get("log1p")
    sqrt_r = by_t.get("sqrt")
    if raw_r and log_r:
        dp_raw = abs(raw_r.get("dprime") or 0)
        dp_log = abs(log_r.get("dprime") or 0)
        dp_sqrt = abs(sqrt_r.get("dprime") or 0) if sqrt_r else 0.0
        fi_raw = raw_r.get("fisher") or 0
        fi_log = log_r.get("fisher") or 0
        # compressive wins
        if (
            max(dp_log, dp_sqrt) > dp_raw + 0.15
            or max(fi_log, (sqrt_r or {}).get("fisher") or 0) > fi_raw * 1.2
        ):
            return "log_linear_good"
        if dp_raw > max(dp_log, dp_sqrt) + 0.15 or fi_raw > fi_log * 1.2:
            return "raw_linear_good"

    if best["transform"] in ("log1p", "sqrt"):
        return "log_linear_good"
    if best["transform"] == "raw":
        return "raw_linear_good"
    return "rank_signal_only"


def audit_one(
    pool: dict[str, np.ndarray],
    sig: EnergySignal,
    *,
    mask: np.ndarray,
    slice_id: str,
) -> list[dict[str, Any]]:
    y = pool["gt_match"].astype(bool)[mask]
    energy = np.maximum(sig.extract(pool)[mask], 0.0)
    # scale for log1p: median of all energy on slice
    med = float(np.median(energy[energy > 0])) if np.any(energy > 0) else 1.0
    if med <= 0:
        med = 1.0

    rows: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        x = apply_transform(energy, transform, scale=med)
        pos, neg = x[y], x[~y]

        # ranking score: higher = more pos-like
        # lower_is_better energy => -energy for ranking on raw space;
        # for transforms of energy, still lower x better if monotone increasing transform
        if sig.lower_is_better:
            rank_score = -x
        else:
            rank_score = x

        auc = _safe_auc(y, rank_score)
        ap = _safe_ap(y, rank_score)
        dp = dprime(pos, neg)
        fi = fisher_score(pos, neg)
        ov = gaussian_overlap(pos, neg)
        ks = ks_distance(pos, neg)
        logm = logistic_metrics(y, x)

        # quantile gap on x: |med_pos - med_neg| / pooled iqr
        iqr_p = (
            float(np.percentile(pos, 75) - np.percentile(pos, 25)) if pos.size else 0.0
        )
        iqr_n = (
            float(np.percentile(neg, 75) - np.percentile(neg, 25)) if neg.size else 0.0
        )
        pooled_iqr = 0.5 * (iqr_p + iqr_n) + 1e-12
        qgap = (
            abs(float(np.median(pos)) - float(np.median(neg))) / pooled_iqr
            if pos.size and neg.size
            else float("nan")
        )

        coef_std = _coef_std_seq(pool, mask, x, y)

        row = {
            "signal_name": sig.signal_name,
            "transform": transform,
            "slice": slice_id,
            "n_gt": int(y.sum()),
            "n_fp": int((~y).sum()),
            "lower_is_better": sig.lower_is_better,
            "log1p_scale": med,
            # layer 1 ranking
            "auc": auc,
            "ap": ap,
            "ks": ks,
            "quantile_gap": qgap,
            # layer 2 linear
            "dprime": dp,
            "dprime_abs": abs(dp) if np.isfinite(dp) else None,
            "fisher": fi,
            "gaussian_overlap": ov,
            "mean_gt": float(pos.mean()) if pos.size else None,
            "mean_fp": float(neg.mean()) if neg.size else None,
            "std_gt": float(pos.std()) if pos.size else None,
            "std_fp": float(neg.std()) if neg.size else None,
            "median_gt": float(np.median(pos)) if pos.size else None,
            "median_fp": float(np.median(neg)) if neg.size else None,
            "iqr_gt": iqr_p,
            "iqr_fp": iqr_n,
            "logistic_logloss": logm["logistic_logloss"],
            "brier": logm["brier"],
            "ece": logm["ece"],
            "coef": logm["coef"],
            "intercept": logm["intercept"],
            "coef_std_across_seq": coef_std,
        }
        rows.append(row)

    # diagnosis on this slice
    diag = diagnose_row(rows)

    # best transform by linear quality among raw/log1p/sqrt
    def lin_q(r: dict[str, Any]) -> float:
        dp = r["dprime_abs"] or 0.0
        fi = r["fisher"] or 0.0
        ll = r["logistic_logloss"]
        return dp + fi + ((1.0 / (1.0 + ll)) if ll is not None else 0.0)

    best = max(
        (r for r in rows if r["transform"] in ("raw", "log1p", "sqrt")),
        key=lin_q,
        default=rows[0],
    )
    for r in rows:
        r["best_transform"] = best["transform"]
        r["diagnosis"] = diag

    return rows


def _coef_std_seq(
    pool: dict[str, np.ndarray],
    mask: np.ndarray,
    x_masked: np.ndarray,
    y_masked: np.ndarray,
) -> float | None:
    """x_masked / y_masked already restricted to mask; recover seq from pool[mask]."""
    seq = pool["seq"][mask]
    coefs = []
    for s in sorted(set(seq.tolist())):
        m = seq == s
        y = y_masked[m]
        if y.sum() < 5 or (~y).sum() < 5:
            continue
        met = logistic_metrics(y, x_masked[m])
        if met["coef"] is not None:
            coefs.append(float(met["coef"]))
    if len(coefs) < 2:
        return None
    return float(np.std(coefs))


def run_audit(
    pool: dict[str, np.ndarray],
    signals: list[EnergySignal],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _audit.ensure_prod_proxy_scores(pool)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"by_signal": {}}

    for sig in signals:
        sig_rows: list[dict[str, Any]] = []
        for slice_id, gap_range in SLICES_GAP:
            mask = slice_mask(pool, slice_id, gap_range)
            if mask.sum() < 20 or pool["gt_match"][mask].sum() < 5:
                continue
            rows = audit_one(pool, sig, mask=mask, slice_id=slice_id)
            sig_rows.extend(rows)
            all_rows.extend(rows)

        # global diagnosis
        glob = [
            r for r in sig_rows if r["slice"] == "global" and r["transform"] == "raw"
        ]
        glob_log = [
            r for r in sig_rows if r["slice"] == "global" and r["transform"] == "log1p"
        ]
        hard = [
            r for r in sig_rows if r["slice"] == "hard_pool" and r["transform"] == "raw"
        ]
        diag_g = next(
            (r["diagnosis"] for r in sig_rows if r["slice"] == "global"),
            "no_signal",
        )
        # slice_only: weak global but strong gap bin
        gap_strong = False
        for r in sig_rows:
            if r["slice"].startswith("gap_") and r["transform"] == "raw":
                if (
                    r["auc"] is not None
                    and r["auc"] >= 0.75
                    and (glob and (glob[0]["auc"] or 0) < 0.65)
                ):
                    gap_strong = True
        if gap_strong and diag_g in ("no_signal", "rank_signal_only"):
            diag_g = "slice_only_signal"
            for r in sig_rows:
                if r["slice"] == "global":
                    r["diagnosis"] = diag_g

        summary["by_signal"][sig.signal_name] = {
            "diagnosis_global": diag_g,
            "best_transform_global": next(
                (r["best_transform"] for r in sig_rows if r["slice"] == "global"),
                None,
            ),
            "auc_raw_global": glob[0]["auc"] if glob else None,
            "auc_raw_hard": hard[0]["auc"] if hard else None,
            "dprime_abs_raw_global": glob[0]["dprime_abs"] if glob else None,
            "dprime_abs_log_global": glob_log[0]["dprime_abs"] if glob_log else None,
            "fisher_raw_global": glob[0]["fisher"] if glob else None,
            "fisher_log_global": glob_log[0]["fisher"] if glob_log else None,
            "logloss_raw_global": glob[0]["logistic_logloss"] if glob else None,
            "logloss_log_global": glob_log[0]["logistic_logloss"] if glob_log else None,
            "notes": sig.notes,
        }

    return all_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # stable column order
    cols = [
        "signal_name",
        "transform",
        "slice",
        "n_gt",
        "n_fp",
        "auc",
        "ap",
        "ks",
        "quantile_gap",
        "dprime",
        "dprime_abs",
        "fisher",
        "gaussian_overlap",
        "mean_gt",
        "mean_fp",
        "std_gt",
        "std_fp",
        "median_gt",
        "median_fp",
        "iqr_gt",
        "iqr_fp",
        "logistic_logloss",
        "brier",
        "ece",
        "coef",
        "intercept",
        "coef_std_across_seq",
        "best_transform",
        "diagnosis",
        "lower_is_better",
        "log1p_scale",
    ]
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument("--signal", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    catalog = default_catalog()
    if args.list:
        for s in catalog:
            print(f"{s.signal_name:<20} lower_is_better={s.lower_is_better}  {s.notes}")
        return

    if args.signal:
        want = set(args.signal)
        catalog = [s for s in catalog if s.signal_name in want]
        miss = want - {s.signal_name for s in catalog}
        if miss:
            raise SystemExit(f"unknown signals: {sorted(miss)}")
    elif not args.all:
        raise SystemExit("pass --all or --signal NAME")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_energy_xform_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)

    rows, summary = run_audit(pool, catalog)
    write_csv(study / "energy_transform_separability.csv", rows)
    out = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "principle": {
            "auc_monotone_invariant": True,
            "use_auc_for": "ranking separability only",
            "use_dprime_fisher_logloss_for": "linear margin under transform",
        },
        "summary": summary,
        "n_rows": len(rows),
    }
    (study / "summary.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8"
    )
    (study / "rows.json").write_text(
        json.dumps(rows, indent=2, default=float) + "\n", encoding="utf-8"
    )

    # print global table
    print(f"STUDY={study}")
    print(
        f"{'signal':<18} {'diag':<20} {'best':<8} "
        f"{'AUC':>6} {"d'_raw":>7} {"d'_log":>7} {'F_raw':>7} {'F_log':>7} "
        f"{'ll_raw':>7} {'ll_log':>7}"
    )
    for name, s in summary["by_signal"].items():
        print(
            f"{name:<18} {s['diagnosis_global']:<20} {str(s['best_transform_global']):<8} "
            f"{s['auc_raw_global'] or float('nan'):6.3f} "
            f"{s['dprime_abs_raw_global'] or float('nan'):7.3f} "
            f"{s['dprime_abs_log_global'] or float('nan'):7.3f} "
            f"{s['fisher_raw_global'] or float('nan'):7.3f} "
            f"{s['fisher_log_global'] or float('nan'):7.3f} "
            f"{s['logloss_raw_global'] or float('nan'):7.3f} "
            f"{s['logloss_log_global'] or float('nan'):7.3f}"
        )
    print(f"\nWrote {study / 'energy_transform_separability.csv'}")
    print(f"Wrote {study / 'summary.json'}")


if __name__ == "__main__":
    main()
