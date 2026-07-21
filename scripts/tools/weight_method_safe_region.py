#!/usr/bin/env python3
"""Compare weighting methods by GT-safe *productive region* (not best FP).

Objective shift
---------------
  Old: maximize best_FP_removed under GT_hurt <= ε
  New: maximize productive_safe_area@80  (and robust / LOO / boundary)

Coordinate for thr / area
-------------------------
  Fused score s (reject-high). 1D domain in GT tail-mass of s:
    u = P_GT(s > thr) ∈ (0,1)
    thr(u) = quantile_GT(1 − u)
  safe length ratio on that unit interval (comparable across methods).

Five method families
--------------------
  1. GT-CDF / quantile linear evidence
  2. Soft-AND consensus (min / geometric / harmonic) on GT-CDF evidence
  3. Clipped log-z linear
  4. Sparse monotone logistic (nonneg + L1; pure numpy)
  5. Worst-seq / CVaR weight selection on GT-CDF linear

Avoided: unconstrained learned weights, raw sum, per-seq fit, MLP, FP-only Bayesian.

  uv run python scripts/tools/weight_method_safe_region.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_weight_safe_<stamp> \\
    --n-grid 40 --do-loo --jobs 0

CPU note: offline pairs.csv study — multi-process over packs (+ LOO folds
threaded). GPU stays idle by design; use --jobs to fill cores.
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

_tools = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_audit = _load("audit_relink_safe_reject", _tools / "audit_relink_safe_reject.py")

EPS_LADDER = (0.0, 0.001, 0.01)
CLIP_K = 3.0
LOGISTIC_L1 = 0.05
LOGISTIC_STEPS = 400
LOGISTIC_LR = 0.3

# Signal packs: pairs (primary) + one multi pack
DEFAULT_PACKS: list[tuple[str, ...]] = [
    ("score_m_bridge", "abs_log_h"),
    ("bridge_dist", "abs_log_h"),
    ("score_m_bridge", "abs_ratio_m1"),
    ("dist_h", "abs_log_h"),
    ("score_m_bridge", "abs_log_h", "neg_dir_cos"),
    ("score_m_bridge", "abs_log_h", "resid_mean"),
]


@dataclass(frozen=True)
class Axis:
    name: str
    extract: Callable[[dict[str, np.ndarray]], np.ndarray]


def axes() -> dict[str, Axis]:
    return {
        "score_m_bridge": Axis("score_m_bridge", lambda p: p["score_m_bridge"]),
        "bridge_dist": Axis("bridge_dist", lambda p: p["bridge_dist"]),
        "dist_h": Axis("dist_h", lambda p: p["dist_h"]),
        "resid_mean": Axis(
            "resid_mean", lambda p: 0.5 * (p["fwd_resid"] + p["bwd_resid"])
        ),
        "abs_log_h": Axis("abs_log_h", lambda p: p["log_h_ratio"]),
        "abs_ratio_m1": Axis(
            "abs_ratio_m1",
            lambda p: np.abs(p["h_ratio_lost_over_cand"] - 1.0),
        ),
        "neg_dir_cos": Axis("neg_dir_cos", lambda p: -p["dir_cos"]),
        "speed_mismatch": Axis("speed_mismatch", lambda p: p["speed_mismatch"]),
    }


# ── transforms ───────────────────────────────────────────────────────────────


def _finite(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x.astype(float), 0.0)


def extract_raw(
    pool: dict[str, np.ndarray], names: tuple[str, ...], ax: dict[str, Axis]
) -> np.ndarray:
    cols = [_finite(ax[n].extract(pool)) for n in names]
    return np.column_stack(cols)


def gt_cdf_evidence(raw: np.ndarray, y_pos_mask: np.ndarray) -> np.ndarray:
    """Per-feature empirical F_GT(x) = P_GT(S <= x) ∈ [0,1].

    Right-tail reject signals: high F_GT ⇒ more extreme than most GT
    (equiv. 1 − survival; reject-high friendly).

    Calibrated on *positive* rows only (GT matches).
    """
    n, d = raw.shape
    out = np.zeros((n, d), dtype=float)
    for j in range(d):
        pos = raw[y_pos_mask, j]
        if pos.size == 0:
            continue
        # searchsorted on sorted pos: rank / n_pos ≈ F
        sp = np.sort(pos)
        # right rank: fraction of GT <= x
        ranks = np.searchsorted(sp, raw[:, j], side="right")
        out[:, j] = ranks / max(len(sp), 1)
    return out


def clipped_log_z(
    raw: np.ndarray, fit_mask: np.ndarray | None = None, k: float = CLIP_K
) -> np.ndarray:
    """x' = clip(zscore(log1p(x / med)), -k, k) fit on fit_mask (default all)."""
    n, d = raw.shape
    out = np.zeros((n, d), dtype=float)
    m = np.ones(n, dtype=bool) if fit_mask is None else fit_mask
    for j in range(d):
        x = np.maximum(raw[:, j], 0.0)
        med = float(np.median(x[m])) if m.any() else 1.0
        med = max(med, 1e-9)
        zbase = np.log1p(x / med)
        mu = float(np.mean(zbase[m])) if m.any() else 0.0
        sd = float(np.std(zbase[m])) if m.any() else 1.0
        sd = max(sd, 1e-9)
        out[:, j] = np.clip((zbase - mu) / sd, -k, k)
    return out


def soft_and(e: np.ndarray, kind: str) -> np.ndarray:
    """Consensus of GT-CDF evidence columns. Higher = more joint suspicion."""
    e = np.clip(e, 1e-12, 1.0)
    if kind == "min":
        return e.min(axis=1)
    if kind == "geometric_mean":
        return np.exp(np.mean(np.log(e), axis=1))
    if kind == "harmonic_mean":
        return e.shape[1] / np.sum(1.0 / e, axis=1)
    raise ValueError(kind)


def linear_score(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = w / max(w.sum(), 1e-12)
    return e @ w


# ── sparse monotone logistic (nonneg + L1) ───────────────────────────────────


def fit_sparse_monotone_logistic(
    X: np.ndarray,
    y_fp: np.ndarray,
    *,
    l1: float = LOGISTIC_L1,
    steps: int = LOGISTIC_STEPS,
    lr: float = LOGISTIC_LR,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """logit P(FP) = b + X w, w >= 0, L1 on w. Returns (w, b).

    y_fp = 1 for negatives (FP candidates), 0 for GT.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # mild feature scale
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-9)
    Xs = (X - mu) / sd

    w = np.full(d, 0.1, dtype=float)
    b = 0.0
    y = y_fp.astype(float)
    # class balance weight
    n_pos = max(y.sum(), 1.0)
    n_neg = max((1 - y).sum(), 1.0)
    sw = np.where(y > 0.5, 0.5 * n / n_pos, 0.5 * n / n_neg)

    for t in range(steps):
        z = Xs @ w + b
        # stable sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = (p - y) * sw
        grad_w = (Xs.T @ err) / n + l1 * np.sign(w)
        grad_b = float(err.mean())
        # step decay
        step = lr / (1.0 + 0.002 * t)
        w = np.maximum(w - step * grad_w, 0.0)
        b = b - step * grad_b
        if t % 80 == 0 and t > 0:
            # tiny noise to escape plateaus
            w = np.maximum(w + rng.normal(0, 1e-4, size=d), 0.0)

    # map weights back to original X scale for reporting:
    # score = Xs @ w + b = X @ (w/sd) + (b - mu·w/sd)
    w_raw = w / sd
    b_raw = b - float(mu @ w_raw)
    return w_raw, b_raw


def logistic_score(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


# ── 1D safe-region metrics on fused score ────────────────────────────────────


def thr_from_gt_tail(pos: np.ndarray, u: float) -> float:
    u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
    return float(np.quantile(pos, 1.0 - u))


def per_seq_hurt_rates(
    y: np.ndarray, rej: np.ndarray, seq: np.ndarray
) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in np.unique(seq):
        m = seq == s
        ys = y[m]
        if ys.sum() == 0:
            continue
        out[str(s)] = float((ys & rej[m]).sum() / ys.sum())
    return out


def evaluate_1d_safe(
    score: np.ndarray,
    y: np.ndarray,
    seq: np.ndarray,
    *,
    n_grid: int = 40,
    eps_list: tuple[float, ...] = EPS_LADDER,
    alphas: tuple[float, ...] = (0.8, 0.9),
) -> dict[str, Any]:
    """1D fused-score metrics under GT_hurt ≤ ε.

    Important
    ---------
    In pure GT-tail-mass u of the *same* fused score, hurt_rate ≈ u by
    construction — so "safe_area_ratio in u-space" is nearly tautological and
    is reported only as a diagnostic (`safe_u_mass_diagnostic`).

    Primary differentiators across weighting methods:
      - frontier_FP_removed @ ε  (and rate)
      - productive_plateau_width_score_std  (thr band with FP≥α·best, hurt≤ε)
      - robust_FP = min over seq of FP at global thr that is per-seq safe
      - boundary_margin_score_std
      - per_seq_hurt at frontier thr
    """
    score = _finite(score)
    y = y.astype(bool)
    pos = score[y]
    neg = score[~y]
    if pos.size < 2:
        return {"error": "too_few_gt", "by_eps": {}}

    n_pos, n_neg = int(y.sum()), int((~y).sum())
    pos_std = float(np.std(pos))
    if pos_std < 1e-12:
        pos_std = float(np.std(score)) or 1.0

    # Thr grid: dense near the hard-safe frontier (above / around max GT),
    # plus mid-tail quantiles for ε>0. Include thr just above max(pos).
    max_pos = float(pos.max())
    qs = np.unique(
        np.concatenate(
            [
                np.linspace(0.50, 0.99, max(8, n_grid // 3)),
                np.linspace(0.99, 1.0, max(12, n_grid // 2)),
                np.array([1.0]),
            ]
        )
    )
    thrs = [float(np.quantile(pos, q)) for q in qs]
    # also thr slightly above max so ε=0 cell exists with FP_removed > 0 possible
    thrs.append(max_pos)  # score > max_pos ⇒ hurt 0 if no ties above
    thrs.append(max_pos + 1e-9 * max(abs(max_pos), 1.0))
    # unique sorted ascending (higher thr = stricter / fewer rejects for reject-high)
    thrs = sorted(set(thrs))

    grid: list[dict[str, Any]] = []
    for thr in thrs:
        rej = score > thr
        hurt = int((y & rej).sum())
        fprm = int((~y & rej).sum())
        # empirical GT tail mass at this thr
        u_emp = hurt / n_pos if n_pos else 0.0
        psh = per_seq_hurt_rates(y, rej, seq)
        grid.append(
            {
                "thr": thr,
                "u_emp": u_emp,
                "GT_hurt": hurt,
                "GT_hurt_rate": u_emp,
                "FP_removed": fprm,
                "FP_removed_rate": fprm / n_neg if n_neg else 0.0,
                "per_seq_hurt": psh,
            }
        )

    by_eps: dict[str, Any] = {}
    for eps in eps_list:
        safe = [g for g in grid if g["GT_hurt_rate"] <= eps + 1e-15]
        best_fp = max((g["FP_removed"] for g in safe), default=0)
        best = (
            max(safe, key=lambda g: (g["FP_removed"], -g["GT_hurt_rate"], g["thr"]))
            if safe
            else None
        )

        # Productive thr band: safe AND FP >= α * best
        prod_metrics: dict[str, float] = {}
        for alpha in alphas:
            thr_fp = alpha * best_fp if best_fp > 0 else float("inf")
            prod_cells = [g for g in safe if g["FP_removed"] >= thr_fp - 1e-9]
            if prod_cells:
                thr_lo = min(g["thr"] for g in prod_cells)  # most aggressive in band
                thr_hi = max(
                    g["thr"] for g in prod_cells
                )  # most strict still productive
                width = (thr_hi - thr_lo) / pos_std
            else:
                thr_lo = thr_hi = width = 0.0
            tag = int(alpha * 100)
            prod_metrics[f"productive_plateau_width_score_std@{tag}"] = float(width)
            # alias for table column naming continuity
            prod_metrics[f"productive_safe_area_ratio@{tag}"] = float(width)
            prod_metrics[f"productive_safe_area@{tag}"] = float(width)
            prod_metrics[f"productive_thr_lo@{tag}"] = float(thr_lo)
            prod_metrics[f"productive_thr_hi@{tag}"] = float(thr_hi)
            prod_metrics[f"n_prod_cells@{tag}"] = float(len(prod_cells))

        # Robust frontier: thr in safe that keeps ALL seqs ≤ ε, max FP
        robust_cells = [
            g
            for g in safe
            if g["per_seq_hurt"]
            and all(v <= eps + 1e-15 for v in g["per_seq_hurt"].values())
        ]
        robust_best = (
            max(robust_cells, key=lambda g: g["FP_removed"]) if robust_cells else None
        )
        robust_fp = int(robust_best["FP_removed"]) if robust_best else 0
        # robust "thickness": width of robust productive@80 band
        if robust_cells and robust_fp > 0:
            rob_prod = [g for g in robust_cells if g["FP_removed"] >= 0.8 * robust_fp]
            if rob_prod:
                robust_width = (
                    max(g["thr"] for g in rob_prod) - min(g["thr"] for g in rob_prod)
                ) / pos_std
            else:
                robust_width = 0.0
        else:
            robust_width = 0.0

        # boundary margin: from best thr down toward unsafe (lower thr = more aggressive)
        boundary_margin = None
        if best is not None:
            unsafe = [g for g in grid if g["GT_hurt_rate"] > eps + 1e-15]
            if unsafe:
                # nearest unsafe thr below best (more aggressive)
                below = [g for g in unsafe if g["thr"] < best["thr"]]
                if below:
                    nearest = max(below, key=lambda g: g["thr"])
                    boundary_margin = (best["thr"] - nearest["thr"]) / pos_std
                else:
                    boundary_margin = 0.0
            else:
                boundary_margin = float("inf")

        # per-seq FP at the global best thr (if best exists)
        per_seq_fp: dict[str, int] = {}
        per_seq_prod: list[float] = []
        if best is not None:
            thr_b = best["thr"]
            for s in sorted(set(seq.tolist())):
                sm = seq == s
                ys, ns = y[sm], ~y[sm]
                if not sm.any():
                    continue
                rej = score[sm] > thr_b
                per_seq_fp[str(s)] = int((ns & rej).sum())
                # local productive proxy: FP_rate on that seq at global thr if seq safe
                hurt_s = float((ys & rej).sum() / max(ys.sum(), 1))
                n_neg_s = int(ns.sum())
                fp_s = int((ns & rej).sum())
                if hurt_s <= eps + 1e-15 and n_neg_s > 0:
                    per_seq_prod.append(fp_s / n_neg_s)
                else:
                    per_seq_prod.append(0.0)

        # Diagnostic: fraction of *grid* with hurt≤ε (not comparable area claim)
        safe_u_diag = len(safe) / max(len(grid), 1)

        # Classification uses productive plateau width + robust FP fraction
        best_fp_rate = best_fp / n_neg if n_neg else 0.0
        robust_fp_rate = robust_fp / n_neg if n_neg else 0.0
        prod80_w = prod_metrics.get("productive_safe_area_ratio@80", 0.0)
        seq_unstable = bool(
            per_seq_prod
            and min(per_seq_prod) < 0.01 * max(max(per_seq_prod), 1e-9)
            and best_fp_rate > 0.05
        )
        cls = classify_1d(
            frontier_fp_rate=best_fp_rate,
            prod_plateau_w=prod80_w,
            robust_fp_rate=robust_fp_rate,
            boundary_margin=boundary_margin if boundary_margin != float("inf") else 1.0,
            seq_unstable=seq_unstable,
        )

        by_eps[str(eps)] = {
            "epsilon": eps,
            "coordinate_space": "fused_score_thr_grid",
            "n_grid": len(grid),
            "score_pos_std": pos_std,
            # primary
            "best_FP_removed": best_fp,
            "best_FP_removed_rate": best_fp_rate,
            "best_GT_hurt": best["GT_hurt"] if best else None,
            "best_GT_hurt_rate": best["GT_hurt_rate"] if best else None,
            "best_thr": best["thr"] if best else None,
            "best_u_emp": best["u_emp"] if best else None,
            **prod_metrics,
            "robust_FP_removed": robust_fp,
            "robust_FP_removed_rate": robust_fp_rate,
            "robust_plateau_width_score_std": robust_width,
            # aliases used by ranking / CSV flatten
            "robust_safe_area_ratio": robust_fp_rate,  # comparable rate, not area
            "safe_area_ratio": safe_u_diag,  # DIAGNOSTIC only
            "safe_u_mass_diagnostic": safe_u_diag,
            "best_point_boundary_distance": (
                None if boundary_margin == float("inf") else boundary_margin
            ),
            "plateau_width_min": prod80_w,
            "per_seq_safe_area_min": float(min(per_seq_prod)) if per_seq_prod else None,
            "per_seq_safe_area_std": float(np.std(per_seq_prod))
            if per_seq_prod
            else None,
            "per_seq_safe_area_mean": float(np.mean(per_seq_prod))
            if per_seq_prod
            else None,
            "per_seq_FP_at_best": per_seq_fp,
            "n_safe_cells": len(safe),
            "n_robust_cells": len(robust_cells),
            "classification": cls,
            "note": (
                "productive_*_ratio@α = plateau width in score_std units "
                "(NOT GT-tail area; 1D u-area is tautological)"
            ),
        }

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "by_eps": by_eps,
        "score_pos_median": float(np.median(pos)),
        "score_neg_median": float(np.median(neg)) if neg.size else None,
        "score_pos_std": pos_std,
    }


def classify_1d(
    *,
    frontier_fp_rate: float,
    prod_plateau_w: float,
    robust_fp_rate: float,
    boundary_margin: float | None,
    seq_unstable: bool,
) -> str:
    if seq_unstable:
        return "seq_unstable"
    if frontier_fp_rate < 0.01:
        return "isolated_sweet_spot" if frontier_fp_rate > 0 else "unsafe"
    if prod_plateau_w < 0.05 and frontier_fp_rate > 0:
        return "isolated_sweet_spot"
    if robust_fp_rate < 0.5 * frontier_fp_rate and frontier_fp_rate >= 0.05:
        return "seq_unstable"
    if (
        frontier_fp_rate >= 0.15
        and prod_plateau_w >= 0.2
        and (boundary_margin or 0) >= 0.05
    ):
        if robust_fp_rate >= 0.1:
            return "broad_safe_productive"
        return "usable_safe_region"
    if frontier_fp_rate >= 0.05 and prod_plateau_w >= 0.05:
        return "thin_but_promising"
    if frontier_fp_rate >= 0.05 and prod_plateau_w < 0.05:
        return "isolated_sweet_spot"
    return "thin_but_promising"


# ── weight grids / objectives ────────────────────────────────────────────────


def simplex_weights(d: int, n_steps: int = 9) -> list[np.ndarray]:
    """Uniform grid on simplex (including vertices / equal)."""
    if d == 1:
        return [np.array([1.0])]
    if d == 2:
        ts = np.linspace(0.0, 1.0, n_steps)
        return [np.array([t, 1.0 - t]) for t in ts]
    # d == 3: coarse barycentric
    out: list[np.ndarray] = []
    for i in range(n_steps):
        for j in range(n_steps - i):
            k = (n_steps - 1) - i - j
            if k < 0:
                continue
            w = np.array([i, j, k], dtype=float)
            if w.sum() <= 0:
                continue
            out.append(w / w.sum())
    # equal always included
    out.append(np.ones(d) / d)
    # unique
    uniq = {tuple(np.round(w, 6)): w for w in out}
    return list(uniq.values())


def objective_from_eval(
    ev: dict[str, Any],
    *,
    eps: float = 0.0,
    mode: str = "productive80",
) -> float:
    """Scalar for weight selection. Higher better.

    Prefer thick productive plateau + robust FP rate + frontier FP;
    not best_FP alone.
    """
    be = None
    for k, v in (ev.get("by_eps") or {}).items():
        if abs(float(k) - eps) < 1e-12:
            be = v
            break
    if not be:
        return -1.0
    p80 = float(be.get("productive_safe_area_ratio@80") or 0.0)  # plateau width / σ
    rob = float(
        be.get("robust_FP_removed_rate") or be.get("robust_safe_area_ratio") or 0.0
    )
    fp_r = float(be.get("best_FP_removed_rate") or 0.0)
    bdist = float(be.get("best_point_boundary_distance") or 0.0)
    sstd = float(be.get("per_seq_safe_area_std") or 0.0)
    smin = float(be.get("per_seq_safe_area_min") or 0.0)
    if mode == "productive80":
        # thickness first, then robust rate, then frontier capacity
        return 2.0 * p80 + 1.5 * rob + 1.0 * fp_r + 0.2 * bdist - 0.5 * sstd
    if mode == "cvar":
        return 2.0 * smin + 1.0 * p80 + 1.0 * rob + 0.5 * fp_r - 0.25 * sstd
    if mode == "robust":
        return 2.0 * rob + 1.0 * p80 + 0.5 * fp_r
    return p80 + fp_r


def per_seq_productive_proxy(
    score: np.ndarray,
    y: np.ndarray,
    seq: np.ndarray,
    *,
    eps: float,
    n_grid: int,
) -> list[float]:
    """Per-sequence productive@80 under that seq alone (for CVaR)."""
    vals: list[float] = []
    for s in np.unique(seq):
        m = seq == s
        if y[m].sum() < 2 or (~y[m]).sum() < 1:
            continue
        # local 1d eval
        ev = evaluate_1d_safe(score[m], y[m], seq[m], n_grid=max(15, n_grid // 2))
        be = ev["by_eps"].get(str(eps))
        if be:
            vals.append(float(be["productive_safe_area_ratio@80"]))
        else:
            vals.append(0.0)
    return vals


def cvar_mean_worst(vals: list[float], frac: float = 0.25) -> float:
    if not vals:
        return 0.0
    a = np.sort(np.asarray(vals, dtype=float))
    k = max(1, int(np.ceil(frac * len(a))))
    return float(a[:k].mean())


# ── method runners ───────────────────────────────────────────────────────────


def run_methods_for_pack(
    pool: dict[str, np.ndarray],
    names: tuple[str, ...],
    ax: dict[str, Axis],
    *,
    n_grid: int,
    weight_steps: int,
    do_loo: bool,
    loo_workers: int = 1,
) -> list[dict[str, Any]]:
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    raw = extract_raw(pool, names, ax)
    rows: list[dict[str, Any]] = []

    def pack_row(
        method: str,
        transform: str,
        weights: list[float] | None,
        score: np.ndarray,
        extra: dict[str, Any] | None = None,
        fit_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        # fit_mask unused for full-pool eval; kept for LOO callers
        _ = fit_mask
        ev = evaluate_1d_safe(score, y, seq, n_grid=n_grid)
        row: dict[str, Any] = {
            "pack": "+".join(names),
            "signals": list(names),
            "method": method,
            "transform": transform,
            "weights": weights,
            "n_signals": len(names),
            **{
                k: ev[k]
                for k in ("n_pos", "n_neg", "score_pos_median", "score_neg_median")
                if k in ev
            },
            "by_eps": ev.get("by_eps", {}),
        }
        if extra:
            row.update(extra)
        if do_loo:
            row["loo"] = loo_eval(
                pool,
                names,
                ax,
                method,
                transform,
                weights,
                n_grid=n_grid,
                extra=extra,
                loo_workers=loo_workers,
            )
        return row

    # ── 1. GT-CDF linear (equal + best productive + best cvar weights) ──
    e_gt = gt_cdf_evidence(raw, y)
    w_cands = simplex_weights(len(names), n_steps=weight_steps)
    best_prod: tuple[float, np.ndarray, dict[str, Any]] | None = None
    best_cvar: tuple[float, np.ndarray, dict[str, Any]] | None = None
    # weight search uses slightly coarser grid; full n_grid only on winners
    n_search = max(24, n_grid // 2)
    for w in w_cands:
        sc = linear_score(e_gt, w)
        ev = evaluate_1d_safe(sc, y, seq, n_grid=n_search)
        op = objective_from_eval(ev, eps=0.0, mode="productive80")
        # CVaR proxy: use per_seq_safe_area_min from full-pool eval (cheap)
        # rather than re-running evaluate per sequence (was O(n_seq) slower).
        be0 = None
        for k, v in (ev.get("by_eps") or {}).items():
            if abs(float(k) - 0.0) < 1e-12:
                be0 = v
                break
        smin = float(be0.get("per_seq_safe_area_min") or 0.0) if be0 else 0.0
        sstd = float(be0.get("per_seq_safe_area_std") or 0.0) if be0 else 0.0
        oc = 2.0 * smin + 0.5 * op - 0.25 * sstd
        if best_prod is None or op > best_prod[0]:
            best_prod = (op, w.copy(), ev)
        if best_cvar is None or oc > best_cvar[0]:
            best_cvar = (oc, w.copy(), ev)

    w_eq = np.ones(len(names)) / len(names)
    rows.append(
        pack_row(
            "gt_cdf_linear_equal",
            "GT_CDF_F",
            w_eq.tolist(),
            linear_score(e_gt, w_eq),
        )
    )
    if best_prod is not None:
        rows.append(
            pack_row(
                "gt_cdf_linear_max_prod80",
                "GT_CDF_F",
                best_prod[1].tolist(),
                linear_score(e_gt, best_prod[1]),
                extra={
                    "select_obj": "productive80+robust+bdist",
                    "select_score": best_prod[0],
                },
            )
        )
    if best_cvar is not None:
        rows.append(
            pack_row(
                "cvar_gt_cdf_linear",
                "GT_CDF_F",
                best_cvar[1].tolist(),
                linear_score(e_gt, best_cvar[1]),
                extra={
                    "select_obj": "CVaR30_per_seq_prod80 + 0.3*prod_obj",
                    "select_score": best_cvar[0],
                },
            )
        )

    # ── 2. Soft-AND ──
    for kind in ("min", "geometric_mean", "harmonic_mean"):
        rows.append(
            pack_row(
                f"soft_and_{kind}",
                "GT_CDF_F",
                None,
                soft_and(e_gt, kind),
            )
        )

    # ── 3. Clipped log-z linear ──
    e_lz = clipped_log_z(raw, fit_mask=None, k=CLIP_K)
    best_lz: tuple[float, np.ndarray] | None = None
    for w in w_cands:
        sc = linear_score(e_lz, w)
        ev = evaluate_1d_safe(sc, y, seq, n_grid=n_grid)
        op = objective_from_eval(ev, eps=0.0, mode="productive80")
        if best_lz is None or op > best_lz[0]:
            best_lz = (op, w.copy())
    rows.append(
        pack_row(
            "clipped_logz_linear_equal",
            f"clip_logz_k{CLIP_K}",
            w_eq.tolist(),
            linear_score(e_lz, w_eq),
        )
    )
    if best_lz is not None:
        rows.append(
            pack_row(
                "clipped_logz_linear_max_prod80",
                f"clip_logz_k{CLIP_K}",
                best_lz[1].tolist(),
                linear_score(e_lz, best_lz[1]),
                extra={"select_obj": "productive80", "select_score": best_lz[0]},
            )
        )

    # ── 4. Sparse monotone logistic on GT-CDF evidence ──
    y_fp = (~y).astype(float)
    w_log, b_log = fit_sparse_monotone_logistic(e_gt, y_fp)
    sc_log = logistic_score(e_gt, w_log, b_log)
    rows.append(
        pack_row(
            "sparse_monotone_logistic",
            "GT_CDF_F",
            w_log.tolist(),
            sc_log,
            extra={"logistic_bias": b_log, "logistic_l1": LOGISTIC_L1},
        )
    )
    # also on clipped log-z
    w_log2, b_log2 = fit_sparse_monotone_logistic(e_lz, y_fp)
    rows.append(
        pack_row(
            "sparse_monotone_logistic_logz",
            f"clip_logz_k{CLIP_K}",
            w_log2.tolist(),
            logistic_score(e_lz, w_log2, b_log2),
            extra={"logistic_bias": b_log2, "logistic_l1": LOGISTIC_L1},
        )
    )

    # ── baseline: raw equal (should be worse; for contrast only) ──
    # normalize raw per col to [0,1] by rank on all to avoid unit blow-up
    raw_rank = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        order = raw[:, j].argsort(kind="mergesort")
        ranks = np.empty(len(raw))
        ranks[order] = np.linspace(0, 1, len(raw))
        raw_rank[:, j] = ranks
    rows.append(
        pack_row(
            "raw_rank_linear_equal_BASELINE",
            "pool_rank",
            w_eq.tolist(),
            linear_score(raw_rank, w_eq),
        )
    )

    return rows


def _loo_one_fold(
    *,
    s: str,
    raw_all: np.ndarray,
    y: np.ndarray,
    seq: np.ndarray,
    method: str,
    transform: str,
    weights: list[float] | None,
    n_grid: int,
    extra: dict[str, Any] | None,
) -> dict[str, Any] | None:
    te = seq == s
    tr = ~te
    if y[tr].sum() < 5 or y[te].sum() < 1:
        return None
    raw_tr, raw_te = raw_all[tr], raw_all[te]
    y_tr, y_te = y[tr], y[te]
    seq_te = seq[te]

    sc_tr, sc_te, w_used = _scores_train_test(
        method, transform, weights, raw_tr, raw_te, y_tr, extra
    )
    if sc_tr is None:
        return None

    n_g = max(20, n_grid // 2)
    ev_tr = evaluate_1d_safe(sc_tr, y_tr, seq[tr], n_grid=n_g)
    ev_te = evaluate_1d_safe(sc_te, y_te, seq_te, n_grid=n_g)

    be_tr = ev_tr["by_eps"].get("0.0") or ev_tr["by_eps"].get("0")
    hurt_te = None
    fp_te = None
    fp_rate_te = None
    if be_tr and be_tr.get("best_thr") is not None:
        thr = float(be_tr["best_thr"])
        rej = sc_te > thr
        n_pos = max(int(y_te.sum()), 1)
        n_neg = max(int((~y_te).sum()), 1)
        hurt_te = float((y_te & rej).sum() / n_pos)
        fp_te = int((~y_te & rej).sum())
        fp_rate_te = fp_te / n_neg

    be_te0 = ev_te["by_eps"].get("0.0") or ev_te["by_eps"].get("0")
    plat = float(be_te0["productive_safe_area_ratio@80"]) if be_te0 else 0.0
    fpr = float(be_te0["best_FP_removed_rate"]) if be_te0 else 0.0
    return {
        "heldout": str(s),
        "weights": w_used,
        "heldout_plateau_w@eps0": plat,
        "heldout_frontier_FP_rate@eps0": fpr,
        "heldout_GT_hurt_rate@train_best_thr": hurt_te,
        "heldout_FP@train_best_thr": fp_te,
        "heldout_FP_rate@train_best_thr": fp_rate_te,
    }


def loo_eval(
    pool: dict[str, np.ndarray],
    names: tuple[str, ...],
    ax: dict[str, Axis],
    method: str,
    transform: str,
    weights: list[float] | None,
    *,
    n_grid: int,
    extra: dict[str, Any] | None = None,
    loo_workers: int = 1,
) -> dict[str, Any]:
    """Strict LOO: fit calibration/weights on train; eval 1d metrics on held-out.

    Fold-level ThreadPool when loo_workers > 1 (numpy-heavy; releases GIL often).
    """
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    raw_all = extract_raw(pool, names, ax)
    seqs = sorted(set(seq.tolist()))

    folds: list[dict[str, Any]] = []
    if loo_workers > 1 and len(seqs) > 1:
        with ThreadPoolExecutor(max_workers=loo_workers) as ex:
            futs = [
                ex.submit(
                    _loo_one_fold,
                    s=s,
                    raw_all=raw_all,
                    y=y,
                    seq=seq,
                    method=method,
                    transform=transform,
                    weights=weights,
                    n_grid=n_grid,
                    extra=extra,
                )
                for s in seqs
            ]
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    folds.append(r)
        folds.sort(key=lambda d: d["heldout"])
    else:
        for s in seqs:
            r = _loo_one_fold(
                s=s,
                raw_all=raw_all,
                y=y,
                seq=seq,
                method=method,
                transform=transform,
                weights=weights,
                n_grid=n_grid,
                extra=extra,
            )
            if r is not None:
                folds.append(r)

    heldout_plat = [float(f["heldout_plateau_w@eps0"]) for f in folds]
    heldout_fp_rate = [float(f["heldout_frontier_FP_rate@eps0"]) for f in folds]
    heldout_hurt = [
        float(f["heldout_GT_hurt_rate@train_best_thr"])
        for f in folds
        if f.get("heldout_GT_hurt_rate@train_best_thr") is not None
    ]
    heldout_fp = [
        float(f["heldout_FP@train_best_thr"])
        for f in folds
        if f.get("heldout_FP@train_best_thr") is not None
    ]

    return {
        "n_folds": len(folds),
        "LOO_safe_area_ratio_mean": float(np.mean(heldout_plat))
        if heldout_plat
        else None,
        "LOO_safe_area_ratio_min": float(np.min(heldout_plat))
        if heldout_plat
        else None,
        "LOO_productive80_mean": float(np.mean(heldout_plat)) if heldout_plat else None,
        "LOO_productive80_min": float(np.min(heldout_plat)) if heldout_plat else None,
        "LOO_frontier_FP_rate_mean": (
            float(np.mean(heldout_fp_rate)) if heldout_fp_rate else None
        ),
        "LOO_FP_mean": float(np.mean(heldout_fp)) if heldout_fp else None,
        "LOO_hurt_at_train_best_mean": (
            float(np.mean(heldout_hurt)) if heldout_hurt else None
        ),
        "LOO_hurt_at_train_best_max": (
            float(np.max(heldout_hurt)) if heldout_hurt else None
        ),
        "folds": folds,
    }


def _scores_train_test(
    method: str,
    transform: str,
    weights: list[float] | None,
    raw_tr: np.ndarray,
    raw_te: np.ndarray,
    y_tr: np.ndarray,
    extra: dict[str, Any] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, list[float] | None]:
    """Fit transform on train; produce train+test scores for method family."""
    d = raw_tr.shape[1]
    w_eq = np.ones(d) / d

    if (
        transform.startswith("GT_CDF")
        or method.startswith("soft_and")
        or method.startswith("gt_cdf")
        or method.startswith("cvar_gt")
        or method == "sparse_monotone_logistic"
    ):
        e_tr = gt_cdf_evidence(raw_tr, y_tr)
        # apply train CDF to test: use train positives only
        e_te = gt_cdf_evidence_apply(raw_te, raw_tr[y_tr])
    elif "logz" in transform or "logz" in method:
        e_tr = clipped_log_z(raw_tr, fit_mask=None)
        e_te = clipped_log_z_apply(raw_te, raw_tr)
    elif transform == "pool_rank":
        # rank within each split independently (baseline; not transferable scale)
        e_tr = _rank01(raw_tr)
        e_te = _rank01(raw_te)
    else:
        e_tr = gt_cdf_evidence(raw_tr, y_tr)
        e_te = gt_cdf_evidence_apply(raw_te, raw_tr[y_tr])

    if method.startswith("soft_and_"):
        kind = method.replace("soft_and_", "")
        return soft_and(e_tr, kind), soft_and(e_te, kind), None

    if method in (
        "gt_cdf_linear_equal",
        "clipped_logz_linear_equal",
        "raw_rank_linear_equal_BASELINE",
    ):
        return linear_score(e_tr, w_eq), linear_score(e_te, w_eq), w_eq.tolist()

    if method in (
        "gt_cdf_linear_max_prod80",
        "clipped_logz_linear_max_prod80",
        "cvar_gt_cdf_linear",
    ):
        # re-search weights on train
        mode = "cvar" if method.startswith("cvar") else "productive80"
        best_w = w_eq
        best_o = -1e9
        for w in simplex_weights(d, n_steps=7):
            sc = linear_score(e_tr, w)
            # cheap: only need obj at eps0 — full evaluate
            dummy_seq = np.array(["tr"] * len(y_tr), dtype=object)
            ev = evaluate_1d_safe(sc, y_tr, dummy_seq, n_grid=20)
            if mode == "cvar":
                # single train seq pool — fall back to productive
                o = objective_from_eval(ev, eps=0.0, mode="productive80")
            else:
                o = objective_from_eval(ev, eps=0.0, mode="productive80")
            if o > best_o:
                best_o = o
                best_w = w.copy()
        return (
            linear_score(e_tr, best_w),
            linear_score(e_te, best_w),
            best_w.tolist(),
        )

    if method.startswith("sparse_monotone_logistic"):
        y_fp = (~y_tr).astype(float)
        w, b = fit_sparse_monotone_logistic(e_tr, y_fp)
        return logistic_score(e_tr, w, b), logistic_score(e_te, w, b), w.tolist()

    # fallback: use provided weights
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        return linear_score(e_tr, w), linear_score(e_te, w), w.tolist()
    return None, None, None


def gt_cdf_evidence_apply(raw: np.ndarray, pos_train: np.ndarray) -> np.ndarray:
    n, d = raw.shape
    out = np.zeros((n, d), dtype=float)
    for j in range(d):
        sp = np.sort(pos_train[:, j])
        if sp.size == 0:
            continue
        ranks = np.searchsorted(sp, raw[:, j], side="right")
        out[:, j] = ranks / max(len(sp), 1)
    return out


def clipped_log_z_apply(
    raw: np.ndarray, raw_fit: np.ndarray, k: float = CLIP_K
) -> np.ndarray:
    n, d = raw.shape
    out = np.zeros((n, d), dtype=float)
    for j in range(d):
        xfit = np.maximum(raw_fit[:, j], 0.0)
        med = float(np.median(xfit))
        med = max(med, 1e-9)
        zfit = np.log1p(xfit / med)
        mu, sd = float(zfit.mean()), max(float(zfit.std()), 1e-9)
        x = np.maximum(raw[:, j], 0.0)
        z = np.log1p(x / med)
        out[:, j] = np.clip((z - mu) / sd, -k, k)
    return out


def _rank01(raw: np.ndarray) -> np.ndarray:
    out = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        order = raw[:, j].argsort(kind="mergesort")
        ranks = np.empty(len(raw))
        ranks[order] = np.linspace(0, 1, len(raw))
        out[:, j] = ranks
    return out


# ── flatten for CSV ──────────────────────────────────────────────────────────


def flatten_row(row: dict[str, Any], eps: float) -> dict[str, Any]:
    be = row.get("by_eps", {}).get(str(eps)) or {}
    loo = row.get("loo") or {}
    return {
        "pack": row["pack"],
        "method": row["method"],
        "transform": row["transform"],
        "weights": json.dumps(row.get("weights"))
        if row.get("weights") is not None
        else "",
        "epsilon": eps,
        "coordinate_space": "fused_score_thr_grid",
        "frontier_FP_rate": be.get("best_FP_removed_rate"),
        "best_FP_removed": be.get("best_FP_removed"),
        "best_GT_hurt": be.get("best_GT_hurt"),
        # productive_* = plateau width in score_std (not GT-tail area)
        "productive_plateau_w@80": be.get("productive_safe_area_ratio@80"),
        "productive_plateau_w@90": be.get("productive_safe_area_ratio@90"),
        "productive_safe_area@80": be.get("productive_safe_area_ratio@80"),
        "productive_safe_area@90": be.get("productive_safe_area_ratio@90"),
        "robust_FP_rate": be.get("robust_FP_removed_rate"),
        "robust_safe_area_ratio": be.get("robust_safe_area_ratio"),
        "robust_plateau_w": be.get("robust_plateau_width_score_std"),
        "safe_u_mass_diagnostic": be.get("safe_u_mass_diagnostic"),
        "safe_area_ratio": be.get("safe_area_ratio"),
        "LOO_safe_area_ratio_mean": loo.get("LOO_safe_area_ratio_mean"),
        "LOO_safe_area_ratio_min": loo.get("LOO_safe_area_ratio_min"),
        "LOO_productive80_mean": loo.get("LOO_productive80_mean"),
        "LOO_hurt_at_train_best_max": loo.get("LOO_hurt_at_train_best_max"),
        "LOO_FP_at_train_best_mean": loo.get("LOO_FP_mean"),
        "best_point_boundary_distance": be.get("best_point_boundary_distance"),
        "plateau_width_min": be.get("plateau_width_min"),
        "per_seq_safe_area_min": be.get("per_seq_safe_area_min"),
        "per_seq_safe_area_std": be.get("per_seq_safe_area_std"),
        "classification": be.get("classification"),
    }


# ── multiprocessing workers (pack-level) ─────────────────────────────────────

_WORKER_POOL: dict[str, np.ndarray] | None = None


def _mp_init(pool: dict[str, np.ndarray]) -> None:
    global _WORKER_POOL
    _WORKER_POOL = pool


def _mp_run_pack(
    payload: tuple[tuple[str, ...], int, int, bool, int],
) -> list[dict[str, Any]]:
    names, n_grid, weight_steps, do_loo, loo_workers = payload
    assert _WORKER_POOL is not None
    return run_methods_for_pack(
        _WORKER_POOL,
        names,
        axes(),
        n_grid=n_grid,
        weight_steps=weight_steps,
        do_loo=do_loo,
        loo_workers=loo_workers,
    )


def _print_pack_rows(pack_rows: list[dict[str, Any]]) -> None:
    for row in pack_rows:
        for eps in (0.0, 0.01):
            be = row["by_eps"].get(str(eps), {})
            loo = row.get("loo") or {}
            loo_h = loo.get("LOO_hurt_at_train_best_max")
            print(
                f"{row['method']:<34} {row['pack'][:26]:<26} {eps:5.3g} "
                f"{100 * float(be.get('best_FP_removed_rate') or 0):6.2f}% "
                f"{float(be.get('productive_safe_area_ratio@80') or 0):7.3f} "
                f"{100 * float(be.get('robust_FP_removed_rate') or 0):7.2f}% "
                f"{float(be.get('best_point_boundary_distance') or 0):6.3f} "
                f"{(loo_h if loo_h is not None else float('nan')):6.3f} "
                f"{be.get('classification', '')}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--weight-steps", type=int, default=9)
    ap.add_argument("--do-loo", action="store_true")
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="pack-level process workers; 0 = min(n_packs, cpu_count)",
    )
    ap.add_argument(
        "--loo-workers",
        type=int,
        default=0,
        help="thread workers per pack for LOO folds; 0 = min(7, max(1, jobs//n_packs))",
    )
    ap.add_argument(
        "--pack",
        action="append",
        default=None,
        help="comma-separated signal names; default=built-in packs",
    )
    args = ap.parse_args()

    ax = axes()
    packs: list[tuple[str, ...]] = []
    if args.pack:
        for p in args.pack:
            names = tuple(x.strip() for x in p.split(",") if x.strip())
            for n in names:
                if n not in ax:
                    raise SystemExit(f"unknown signal {n}; choose from {list(ax)}")
            packs.append(names)
    else:
        packs = list(DEFAULT_PACKS)

    n_cpu = os.cpu_count() or 4
    n_jobs = args.jobs if args.jobs > 0 else min(len(packs), n_cpu)
    n_jobs = max(1, min(n_jobs, len(packs), n_cpu))
    if args.loo_workers > 0:
        loo_workers = args.loo_workers
    else:
        # leave headroom: roughly n_cpu / n_jobs threads for LOO inside each pack
        loo_workers = max(1, min(8, n_cpu // max(n_jobs, 1)))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_weight_safe_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)

    all_rows: list[dict[str, Any]] = []
    print(f"STUDY={study}")
    print(
        f"CPU: n_cpu={n_cpu} pack_jobs={n_jobs} loo_workers/pack={loo_workers} "
        f"(GPU idle expected — pairs.csv offline)"
    )
    print(
        f"{'method':<34} {'pack':<26} {'eps':>5} {'FPr%':>7} {'plat80':>7} "
        f"{'robFPr%':>8} {'marg':>6} {'LOOh':>6} {'class'}"
    )

    payloads = [
        (names, args.n_grid, args.weight_steps, args.do_loo, loo_workers)
        for names in packs
    ]

    if n_jobs == 1 or len(packs) == 1:
        for names, n_grid, weight_steps, do_loo, lw in payloads:
            pack_rows = run_methods_for_pack(
                pool,
                names,
                ax,
                n_grid=n_grid,
                weight_steps=weight_steps,
                do_loo=do_loo,
                loo_workers=lw,
            )
            all_rows.extend(pack_rows)
            _print_pack_rows(pack_rows)
    else:
        # pack-level process pool; each pack rebuilds axes() in worker
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_mp_init,
            initargs=(pool,),
        ) as ex:
            futs = {ex.submit(_mp_run_pack, p): p[0] for p in payloads}
            # preserve pack order when collecting
            results: dict[tuple[str, ...], list[dict[str, Any]]] = {}
            for fut in as_completed(futs):
                names = futs[fut]
                pack_rows = fut.result()
                results[names] = pack_rows
                _print_pack_rows(pack_rows)
                print(f"  [done pack] {'+'.join(names)}", flush=True)
            for names in packs:
                all_rows.extend(results[names])

    # CSV flat table
    flat: list[dict[str, Any]] = []
    for row in all_rows:
        for eps in EPS_LADDER:
            flat.append(flatten_row(row, eps))

    # rank at eps=0: plateau thickness → robust FP rate → frontier FP → LOO (low hurt)
    def rank_key(r: dict[str, Any]) -> tuple:
        loo_hurt = r.get("LOO_hurt_at_train_best_max")
        loo_ok = 1.0 if loo_hurt is None else (1.0 if float(loo_hurt) <= 0.0 else 0.0)
        return (
            float(
                r.get("productive_plateau_w@80")
                or r.get("productive_safe_area@80")
                or 0
            ),
            float(r.get("robust_FP_rate") or r.get("robust_safe_area_ratio") or 0),
            float(r.get("frontier_FP_rate") or 0),
            loo_ok,
            float(r.get("best_point_boundary_distance") or 0),
        )

    flat0 = [r for r in flat if r["epsilon"] == 0.0]
    flat0_sorted = sorted(flat0, key=rank_key, reverse=True)

    cols = list(flat[0].keys()) if flat else []
    with (study / "weight_method_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(flat)

    with (study / "rank_eps0_by_productive_region.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(flat0_sorted)

    # slim json (drop huge fold dumps optionally keep)
    batch = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "objective": {
            "primary": "productive_plateau_width_score_std@80",
            "secondary": [
                "robust_FP_removed_rate",
                "frontier_FP_removed_rate",
                "LOO_hurt_at_train_best_max",
                "boundary_margin_score_std",
            ],
            "not": "best_FP_removed alone; not 1D GT-tail safe_area (tautological)",
        },
        "definition": {
            "evidence_GT_CDF": "F_GT(x)=P_GT(S<=x); high=more extreme than GT",
            "fused_metrics": (
                "thr grid on fused score; productive plateau = thr band width "
                "with hurt≤ε and FP≥0.8·best, measured in GT score std units"
            ),
            "soft_and": "min/geo/harm of F_GT evidence",
            "cvar": "max CVaR30 of per-seq productive proxy over weight simplex",
            "gpu": "this tool is CPU-only offline on pairs.csv",
        },
        "n_grid": args.n_grid,
        "do_loo": args.do_loo,
        "jobs": n_jobs,
        "loo_workers": loo_workers,
        "n_cpu": n_cpu,
        "packs": ["+".join(p) for p in packs],
        "rows": all_rows,
        "rank_eps0_top10": flat0_sorted[:10],
    }
    (study / "summary.json").write_text(
        json.dumps(batch, indent=2, default=float) + "\n", encoding="utf-8"
    )

    print(
        "\n=== TOP @ ε=0 by productive_plateau_w@80 "
        "(score_std thickness; then robust FP rate, frontier FP, LOO clean) ==="
    )
    for i, r in enumerate(flat0_sorted[:12], 1):
        print(
            f"{i:2d}. {r['method']:<34} {r['pack'][:22]:<22} "
            f"plat={float(r.get('productive_plateau_w@80') or 0):5.3f}σ "
            f"FPr={100 * float(r.get('frontier_FP_rate') or 0):5.1f}% "
            f"rob={100 * float(r.get('robust_FP_rate') or 0):5.1f}% "
            f"FP={r['best_FP_removed']} "
            f"LOOhmax={r.get('LOO_hurt_at_train_best_max')} "
            f"{r['classification']}"
        )
    print(f"\nWrote {study / 'weight_method_table.csv'}")
    print(f"Wrote {study / 'rank_eps0_by_productive_region.csv'}")
    print(
        "NOTE: 1D GT-tail safe_area is tautological — rank by plateau width + "
        "robust/frontier FP. Offline study is CPU (pairs.csv); GPU idle is expected."
    )


if __name__ == "__main__":
    main()
