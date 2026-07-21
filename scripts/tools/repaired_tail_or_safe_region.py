#!/usr/bin/env python3
"""Safe-region thickness for frozen all-tail OR candidate (not new policy search).

Candidate: m_b1_repaired_eps0_loo_pass — OR of singleton tail atoms, no zone/gap.

Question
--------
Around the freeze quantile q=0.85, is there a thick ε=0 *productive* region?

Coordinates
-----------
1) shared_pool_q  — thr_i = quantile(all rows of signal_i, q); same q for all atoms
   (matches how gate_rule_search tail_q atoms are fitted)

2) 2D free pair    — (q_a, q_b) free; other signals fixed at freeze q0.85 thr
   Area ratios on unit square of free (q_a,q_b) in [q_lo,q_hi]^2

3) LOO shared-q   — thr from train-6 at q; apply to held-out; map q → te hurt/FP

Metrics (ε ladder): safe_area_ratio, productive@80/90, boundary distance,
per-seq min, LOO productive length. Rank by thickness not best FP alone.

  uv run python scripts/tools/repaired_tail_or_safe_region.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --portable out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json \\
    --study-dir out/signal_study/m_repaired_tail_region_<stamp>
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_tools = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


grs = _load("gate_rule_search", _tools / "gate_rule_search.py")
_audit = _load("audit_relink_safe_reject", _tools / "audit_relink_safe_reject.py")

EPS_LADDER = (0.0, 0.001, 0.01)
FREEZE_Q = 0.85


def extract_named(
    pool: dict[str, np.ndarray], names: list[str]
) -> dict[str, np.ndarray]:
    sig = grs.extract_signals(pool)
    out = {}
    for n in names:
        x = np.asarray(sig[n], dtype=float)
        out[n] = np.where(np.isfinite(x), x, 0.0)
    return out


def thr_pool_q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def or_reject(feats: dict[str, np.ndarray], thrs: dict[str, float]) -> np.ndarray:
    n = next(iter(feats.values())).shape[0]
    rej = np.zeros(n, dtype=bool)
    for name, thr in thrs.items():
        rej |= feats[name] > thr
    return rej


def metrics(
    y: np.ndarray, rej: np.ndarray, seq: np.ndarray | None = None
) -> dict[str, Any]:
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    hurt = int((y & rej).sum())
    fprm = int((~y & rej).sum())
    out: dict[str, Any] = {
        "GT_hurt": hurt,
        "GT_hurt_rate": hurt / n_pos if n_pos else 0.0,
        "FP_removed": fprm,
        "FP_removed_rate": fprm / n_neg if n_neg else 0.0,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    if seq is not None:
        psh = {}
        for s in np.unique(seq):
            m = seq == s
            ys = y[m]
            if ys.sum() == 0:
                continue
            psh[str(s)] = float((ys & rej[m]).sum() / ys.sum())
        out["per_seq_hurt"] = psh
        rates = list(psh.values())
        out["seq_hurt_std"] = float(np.std(rates)) if len(rates) >= 2 else 0.0
    return out


def summarize_grid(
    cells: list[dict[str, Any]],
    *,
    coord_keys: tuple[str, ...],
    domain_measure: float,
    cell_area: float,
    eps_list: tuple[float, ...] = EPS_LADDER,
    freeze_point: dict[str, float] | None = None,
) -> dict[str, Any]:
    by_eps: dict[str, Any] = {}
    for eps in eps_list:
        safe = [c for c in cells if c["GT_hurt_rate"] <= eps + 1e-15]
        safe_area = len(safe) * cell_area
        safe_ratio = safe_area / domain_measure if domain_measure > 0 else 0.0
        best_fp = max((c["FP_removed"] for c in safe), default=0)
        best = (
            max(safe, key=lambda c: (c["FP_removed"], -c["GT_hurt_rate"]))
            if safe
            else None
        )

        prod = {}
        for alpha in (0.8, 0.9):
            thr_fp = alpha * best_fp if best_fp > 0 else float("inf")
            prod_cells = [c for c in safe if c["FP_removed"] >= thr_fp - 1e-9]
            pa = len(prod_cells) * cell_area
            tag = int(alpha * 100)
            prod[f"productive_safe_area@{tag}"] = pa
            prod[f"productive_safe_area_ratio@{tag}"] = (
                pa / domain_measure if domain_measure > 0 else 0.0
            )
            if prod_cells and len(coord_keys) == 1:
                k = coord_keys[0]
                prod[f"productive_q_lo@{tag}"] = float(min(c[k] for c in prod_cells))
                prod[f"productive_q_hi@{tag}"] = float(max(c[k] for c in prod_cells))
                prod[f"productive_width@{tag}"] = (
                    prod[f"productive_q_hi@{tag}"] - prod[f"productive_q_lo@{tag}"]
                )

        # robust: all seq ≤ eps
        robust = [
            c
            for c in safe
            if c.get("per_seq_hurt")
            and all(v <= eps + 1e-15 for v in c["per_seq_hurt"].values())
        ]
        robust_area = len(robust) * cell_area
        robust_ratio = robust_area / domain_measure if domain_measure > 0 else 0.0

        # boundary distance of best in coord space (L_inf)
        bdist = None
        if best is not None:
            unsafe = [c for c in cells if c["GT_hurt_rate"] > eps + 1e-15]
            if unsafe:
                dists = []
                for u in unsafe:
                    d = max(abs(u[k] - best[k]) for k in coord_keys)
                    dists.append(d)
                bdist = float(min(dists))
            else:
                bdist = float("inf")

        # freeze point membership
        freeze_safe = None
        freeze_fp = None
        freeze_near_best = None
        if freeze_point is not None:
            # nearest cell to freeze
            def _d(c: dict[str, Any]) -> float:
                return max(abs(c[k] - freeze_point[k]) for k in coord_keys)

            near = min(cells, key=_d)
            freeze_safe = near["GT_hurt_rate"] <= eps + 1e-15
            freeze_fp = near["FP_removed"]
            if best is not None:
                freeze_near_best = max(abs(near[k] - best[k]) for k in coord_keys)

        # per-seq safe ratio (each seq hurt ≤ eps)
        per_seq_ratios = []
        # only if cells have per_seq
        if cells and cells[0].get("per_seq_hurt"):
            seqs = sorted(cells[0]["per_seq_hurt"].keys())
            for s in seqs:
                n_ok = sum(
                    1
                    for c in cells
                    if (c.get("per_seq_hurt") or {}).get(s, 1.0) <= eps + 1e-15
                )
                per_seq_ratios.append(n_ok * cell_area / domain_measure)

        label = classify_region(
            safe_ratio=safe_ratio,
            prod80=prod.get("productive_safe_area_ratio@80", 0.0),
            robust_ratio=robust_ratio,
            bdist=None if bdist == float("inf") else bdist,
            ndim=len(coord_keys),
        )

        by_eps[str(eps)] = {
            "epsilon": eps,
            "safe_area": safe_area,
            "safe_area_ratio": safe_ratio,
            "n_safe_cells": len(safe),
            **prod,
            "robust_safe_area": robust_area,
            "robust_safe_area_ratio": robust_ratio,
            "best_FP_removed": best_fp,
            "best_GT_hurt": best["GT_hurt"] if best else None,
            "best_GT_hurt_rate": best["GT_hurt_rate"] if best else None,
            "best_coords": {k: best[k] for k in coord_keys} if best else None,
            "best_point_boundary_distance": (None if bdist == float("inf") else bdist),
            "per_seq_safe_area_min": (
                float(min(per_seq_ratios)) if per_seq_ratios else None
            ),
            "per_seq_safe_area_std": (
                float(np.std(per_seq_ratios)) if per_seq_ratios else None
            ),
            "freeze_point_safe": freeze_safe,
            "freeze_point_FP": freeze_fp,
            "freeze_to_best_coord_dist": freeze_near_best,
            "classification": label,
        }
    return by_eps


def classify_region(
    *,
    safe_ratio: float,
    prod80: float,
    robust_ratio: float,
    bdist: float | None,
    ndim: int,
) -> str:
    # 1D shared-q: widths ~0.05–0.2 are meaningful; 2D ratios like gt_safe_region
    if ndim == 1:
        if safe_ratio < 1e-12:
            return "empty_safe"
        if prod80 < 1e-12:
            return "safe_but_unproductive" if safe_ratio > 0 else "empty_safe"
        # prod80 here is length ratio of productive band / domain
        if prod80 >= 0.15 and (bdist or 0) >= 0.01:
            return "broad_safe_productive"
        if prod80 >= 0.05:
            return "usable_safe_region"
        if prod80 >= 0.02:
            return "thin_but_promising"
        return "isolated_sweet_spot"
    # 2D
    if safe_ratio < 0.01:
        return "isolated_sweet_spot"
    if prod80 < 1e-6 and safe_ratio >= 0.05:
        return "safe_but_unproductive"
    if safe_ratio < 0.05:
        return "thin_but_promising" if prod80 > 0 else "isolated_sweet_spot"
    if safe_ratio < 0.15:
        return "usable_safe_region" if prod80 >= 0.02 else "thin_but_promising"
    if prod80 >= 0.03 and robust_ratio >= 0.01:
        return "broad_safe_productive"
    return "usable_safe_region"


def shared_q_region(
    feats: dict[str, np.ndarray],
    y: np.ndarray,
    seq: np.ndarray,
    names: list[str],
    *,
    q_lo: float = 0.70,
    q_hi: float = 0.99,
    n_grid: int = 60,
    freeze_q: float = FREEZE_Q,
) -> dict[str, Any]:
    qs = np.linspace(q_lo, q_hi, n_grid)
    dq = float(qs[1] - qs[0]) if n_grid > 1 else 1.0
    domain = float(q_hi - q_lo)
    cells = []
    for q in qs:
        thrs = {n: thr_pool_q(feats[n], float(q)) for n in names}
        rej = or_reject(feats, thrs)
        m = metrics(y, rej, seq)
        cells.append({"q": float(q), "thrs": thrs, **m})

    by_eps = summarize_grid(
        cells,
        coord_keys=("q",),
        domain_measure=domain,
        cell_area=dq,
        freeze_point={"q": freeze_q},
    )
    # also store curve
    return {
        "mode": "shared_pool_q",
        "q_lo": q_lo,
        "q_hi": q_hi,
        "n_grid": n_grid,
        "freeze_q": freeze_q,
        "signals": names,
        "by_eps": by_eps,
        "curve": [
            {
                "q": c["q"],
                "GT_hurt": c["GT_hurt"],
                "GT_hurt_rate": c["GT_hurt_rate"],
                "FP_removed": c["FP_removed"],
                "FP_removed_rate": c["FP_removed_rate"],
            }
            for c in cells
        ],
    }


def pair_2d_region(
    feats: dict[str, np.ndarray],
    y: np.ndarray,
    seq: np.ndarray,
    name_a: str,
    name_b: str,
    names_all: list[str],
    *,
    freeze_q: float = FREEZE_Q,
    q_lo: float = 0.70,
    q_hi: float = 0.99,
    n_grid: int = 25,
) -> dict[str, Any]:
    # fixed thr for other signals at freeze_q
    fixed = {
        n: thr_pool_q(feats[n], freeze_q)
        for n in names_all
        if n not in (name_a, name_b)
    }
    qs = np.linspace(q_lo, q_hi, n_grid)
    dq = float(qs[1] - qs[0]) if n_grid > 1 else 1.0
    domain = (q_hi - q_lo) ** 2
    cell_area = dq * dq
    cells = []
    for qa in qs:
        for qb in qs:
            thrs = dict(fixed)
            thrs[name_a] = thr_pool_q(feats[name_a], float(qa))
            thrs[name_b] = thr_pool_q(feats[name_b], float(qb))
            rej = or_reject(feats, thrs)
            m = metrics(y, rej, seq)
            cells.append({"q_a": float(qa), "q_b": float(qb), **m})

    by_eps = summarize_grid(
        cells,
        coord_keys=("q_a", "q_b"),
        domain_measure=domain,
        cell_area=cell_area,
        freeze_point={"q_a": freeze_q, "q_b": freeze_q},
    )
    return {
        "mode": "2d_pair_others_fixed_q85",
        "pair": f"{name_a}__{name_b}",
        "name_a": name_a,
        "name_b": name_b,
        "fixed_signals": list(fixed.keys()),
        "q_lo": q_lo,
        "q_hi": q_hi,
        "n_grid": n_grid,
        "by_eps": by_eps,
    }


def loo_shared_q(
    pool: dict[str, np.ndarray],
    names: list[str],
    *,
    q_lo: float = 0.70,
    q_hi: float = 0.99,
    n_grid: int = 40,
    freeze_q: float = FREEZE_Q,
) -> dict[str, Any]:
    seq = pool["seq"]
    y = pool["gt_match"].astype(bool)
    feats_all = extract_named(pool, names)
    qs = np.linspace(q_lo, q_hi, n_grid)
    dq = float(qs[1] - qs[0]) if n_grid > 1 else 1.0
    domain = float(q_hi - q_lo)

    # aggregate: for each q, mean te hurt / sum hurt / mean te FP
    per_q: dict[float, list[dict[str, Any]]] = {float(q): [] for q in qs}
    folds = []

    for held in sorted({str(s) for s in seq.tolist()}):
        tr = seq != held
        te = seq == held
        y_te = y[te]
        feats_tr = {n: feats_all[n][tr] for n in names}
        feats_te = {n: feats_all[n][te] for n in names}
        seq_te = seq[te]
        fold_curve = []
        for q in qs:
            thrs = {n: thr_pool_q(feats_tr[n], float(q)) for n in names}
            rej = or_reject(feats_te, thrs)
            m = metrics(y_te, rej, seq_te)
            row = {
                "q": float(q),
                **{
                    k: m[k]
                    for k in (
                        "GT_hurt",
                        "GT_hurt_rate",
                        "FP_removed",
                        "FP_removed_rate",
                    )
                },
            }
            fold_curve.append(row)
            per_q[float(q)].append(row)
        folds.append({"heldout": held, "curve": fold_curve})

    # LOO-safe at each q: all folds GT_hurt==0
    loo_cells = []
    for q in qs:
        rows = per_q[float(q)]
        sum_hurt = int(sum(r["GT_hurt"] for r in rows))
        max_hurt = int(max(r["GT_hurt"] for r in rows))
        mean_fp = float(np.mean([r["FP_removed"] for r in rows]))
        n_gt0 = sum(1 for r in rows if r["GT_hurt"] == 0)
        loo_cells.append(
            {
                "q": float(q),
                "GT_hurt": sum_hurt,  # use sum as strict multi-fold
                "GT_hurt_rate": max_hurt,  # misuse rate field as max fold hurt count proxy
                "FP_removed": mean_fp,
                "FP_removed_rate": float(np.mean([r["FP_removed_rate"] for r in rows])),
                "n_folds_GT0": n_gt0,
                "max_fold_GT_hurt": max_hurt,
                "sum_fold_GT_hurt": sum_hurt,
                # for summarize: hurt_rate 0 only if all clean
                "per_seq_hurt": {
                    f["heldout"]: next(
                        r["GT_hurt_rate"]
                        for r in f["curve"]
                        if abs(r["q"] - float(q)) < 1e-12
                    )
                    for f in folds
                },
            }
        )

    # Build cells compatible with summarize: GT_hurt_rate = 0 iff all folds clean
    cells_for_sum = []
    for c in loo_cells:
        cells_for_sum.append(
            {
                "q": c["q"],
                "GT_hurt": c["sum_fold_GT_hurt"],
                "GT_hurt_rate": 0.0 if c["max_fold_GT_hurt"] == 0 else 1.0,
                "FP_removed": c["FP_removed"],
                "FP_removed_rate": c["FP_removed_rate"],
                "per_seq_hurt": {
                    k: (0.0 if v == 0 else 1.0) for k, v in c["per_seq_hurt"].items()
                },
                "n_folds_GT0": c["n_folds_GT0"],
                "max_fold_GT_hurt": c["max_fold_GT_hurt"],
            }
        )

    by_eps = summarize_grid(
        cells_for_sum,
        coord_keys=("q",),
        domain_measure=domain,
        cell_area=dq,
        freeze_point={"q": freeze_q},
        eps_list=(0.0,),  # LOO strict: only ε=0 (all folds clean)
    )
    # enrich with fold counts on freeze
    freeze_row = min(loo_cells, key=lambda c: abs(c["q"] - freeze_q))
    return {
        "mode": "loo_shared_pool_q",
        "q_lo": q_lo,
        "q_hi": q_hi,
        "n_grid": n_grid,
        "freeze_q": freeze_q,
        "by_eps": by_eps,
        "freeze_loo": freeze_row,
        "loo_curve": loo_cells,
        "folds": folds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--portable", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument("--q-lo", type=float, default=0.70)
    ap.add_argument("--q-hi", type=float, default=0.99)
    ap.add_argument("--n-grid-1d", type=int, default=60)
    ap.add_argument("--n-grid-2d", type=int, default=25)
    ap.add_argument("--freeze-q", type=float, default=FREEZE_Q)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_repaired_tail_region_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    portable = json.loads(args.portable.read_text(encoding="utf-8"))
    names = sorted({spec["signal"] for spec in portable["atom_specs"].values()})
    freeze_q = float(args.freeze_q)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    feats = extract_named(pool, names)

    print(f"STUDY={study}")
    print(f"signals={names} freeze_q={freeze_q}")

    # 1) shared-q
    shared = shared_q_region(
        feats,
        y,
        seq,
        names,
        q_lo=args.q_lo,
        q_hi=args.q_hi,
        n_grid=args.n_grid_1d,
        freeze_q=freeze_q,
    )
    print("\n=== shared_pool_q (all tails same q) ===")
    print(
        f"{'eps':>6} {'safe%':>7} {'p80%':>7} {'p80_w':>7} {'rob%':>7} "
        f"{'bestFP':>7} {'best_q':>6} {'fz_safe':>7} {'class'}"
    )
    for eps in EPS_LADDER:
        e = shared["by_eps"][str(eps)]
        bq = (e.get("best_coords") or {}).get("q")
        print(
            f"{eps:6.3g} {100 * e['safe_area_ratio']:6.2f}% "
            f"{100 * e['productive_safe_area_ratio@80']:6.2f}% "
            f"{e.get('productive_width@80') or 0:7.3f} "
            f"{100 * e['robust_safe_area_ratio']:6.2f}% "
            f"{e['best_FP_removed']:7.0f} "
            f"{(bq if bq is not None else float('nan')):6.3f} "
            f"{str(e['freeze_point_safe']):>7} "
            f"{e['classification']}"
        )

    # 2) 2D pairs (key combinations)
    pairs = [
        ("score_m_bridge", "abs_log_h"),
        ("score_m_bridge", "dist_h"),
        ("abs_log_h", "abs_ratio_m1"),
        ("dist_h", "abs_log_h"),
        ("score_m_bridge", "resid_mean"),
    ]
    pair_results = {}
    print("\n=== 2D pairs (others fixed @ q85) @ ε=0 ===")
    print(
        f"{'pair':<36} {'safe%':>7} {'p80%':>7} {'bdist':>6} {'best':>14} {'fz':>5} class"
    )
    for a, b in pairs:
        if a not in names or b not in names:
            continue
        pr = pair_2d_region(
            feats,
            y,
            seq,
            a,
            b,
            names,
            freeze_q=freeze_q,
            q_lo=args.q_lo,
            q_hi=args.q_hi,
            n_grid=args.n_grid_2d,
        )
        pair_results[pr["pair"]] = pr
        e = pr["by_eps"]["0.0"]
        bc = e.get("best_coords") or {}
        print(
            f"{pr['pair']:<36} {100 * e['safe_area_ratio']:6.2f}% "
            f"{100 * e['productive_safe_area_ratio@80']:6.2f}% "
            f"{float(e.get('best_point_boundary_distance') or 0):6.3f} "
            f"({bc.get('q_a', float('nan')):.2f},{bc.get('q_b', float('nan')):.2f}) "
            f"{str(e['freeze_point_safe']):>5} {e['classification']}"
        )

    # 3) LOO
    print("\n=== LOO shared_pool_q (thr fit train-6) ===")
    loo = loo_shared_q(
        pool,
        names,
        q_lo=args.q_lo,
        q_hi=args.q_hi,
        n_grid=min(40, args.n_grid_1d),
        freeze_q=freeze_q,
    )
    e0 = loo["by_eps"]["0.0"]
    print(
        f"LOO ε=0: safe%={100 * e0['safe_area_ratio']:.2f}% "
        f"p80%={100 * e0['productive_safe_area_ratio@80']:.2f}% "
        f"width={e0.get('productive_width@80')} "
        f"best_q={e0.get('best_coords')} "
        f"freeze_safe={e0['freeze_point_safe']} "
        f"freeze_n_GT0={loo['freeze_loo']['n_folds_GT0']}/7 "
        f"class={e0['classification']}"
    )
    print(
        f"  freeze q={freeze_q}: sum_hurt={loo['freeze_loo']['sum_fold_GT_hurt']} "
        f"mean_teFP={loo['freeze_loo']['FP_removed']:.1f}"
    )

    # upgrade decision
    sh0 = shared["by_eps"]["0.0"]
    loo_prod = e0.get("productive_safe_area_ratio@80") or 0.0
    sh_prod = sh0.get("productive_safe_area_ratio@80") or 0.0
    sh_w = sh0.get("productive_width@80") or 0.0
    freeze_ok = bool(sh0.get("freeze_point_safe")) and bool(e0.get("freeze_point_safe"))

    if freeze_ok and sh_prod >= 0.05 and loo_prod >= 0.02 and sh_w >= 0.02:
        upgrade = "LOO_pass_region_candidate"
    elif freeze_ok and (sh_prod > 0 or sh0["safe_area_ratio"] > 0):
        upgrade = "LOO_pass_point_candidate_thin_region"
    elif freeze_ok:
        upgrade = "LOO_pass_point_candidate_no_region"
    else:
        upgrade = "freeze_not_in_safe_region"

    summary = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "m_b1_repaired_eps0_loo_pass_20260709",
        "portable": str(args.portable.resolve()),
        "pairs_csv": str(args.pairs.resolve()),
        "exposure": _audit.exposure_summary(pool),
        "signals": names,
        "freeze_q": freeze_q,
        "question": (
            "On all-tail repaired OR, is there thick ε=0 productive plateau near q85?"
        ),
        "upgrade": upgrade,
        "shared_pool_q": shared,
        "pairs_2d": {
            k: {kk: vv for kk, vv in v.items() if kk != "_grid"}
            for k, v in pair_results.items()
        },
        "loo_shared_q": {
            k: v
            for k, v in loo.items()
            if k not in ("folds",)  # keep folds in separate file for size
        },
        "headline_metrics_eps0": {
            "shared_safe_area_ratio": sh0["safe_area_ratio"],
            "shared_productive@80": sh_prod,
            "shared_productive_width@80": sh_w,
            "shared_best_q": sh0.get("best_coords"),
            "shared_freeze_safe": sh0.get("freeze_point_safe"),
            "shared_classification": sh0["classification"],
            "loo_safe_area_ratio": e0["safe_area_ratio"],
            "loo_productive@80": loo_prod,
            "loo_productive_width@80": e0.get("productive_width@80"),
            "loo_best_q": e0.get("best_coords"),
            "loo_freeze_safe": e0.get("freeze_point_safe"),
            "loo_classification": e0["classification"],
        },
        "not_production": True,
    }

    (study / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    (study / "shared_q_curve.csv").parent.mkdir(parents=True, exist_ok=True)
    with (study / "shared_q_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(shared["curve"][0].keys()))
        w.writeheader()
        w.writerows(shared["curve"])
    with (study / "loo_shared_q_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        rows = loo["loo_curve"]
        cols = [
            "q",
            "n_folds_GT0",
            "max_fold_GT_hurt",
            "sum_fold_GT_hurt",
            "FP_removed",
            "FP_removed_rate",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (study / "loo_folds.json").write_text(
        json.dumps(loo["folds"], indent=2, default=float) + "\n", encoding="utf-8"
    )

    # pair table
    pair_rows = []
    for k, pr in pair_results.items():
        for eps in EPS_LADDER:
            e = pr["by_eps"][str(eps)]
            pair_rows.append(
                {
                    "pair": k,
                    "epsilon": eps,
                    "safe_area_ratio": e["safe_area_ratio"],
                    "productive_safe_area@80": e["productive_safe_area_ratio@80"],
                    "productive_safe_area@90": e["productive_safe_area_ratio@90"],
                    "robust_safe_area_ratio": e["robust_safe_area_ratio"],
                    "best_FP_removed": e["best_FP_removed"],
                    "best_coords": json.dumps(e.get("best_coords")),
                    "best_point_boundary_distance": e.get(
                        "best_point_boundary_distance"
                    ),
                    "freeze_point_safe": e.get("freeze_point_safe"),
                    "classification": e["classification"],
                }
            )
    with (study / "pairs_2d_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    print(f"\nUPGRADE={upgrade}")
    print(f"Wrote {study / 'summary.json'}")


if __name__ == "__main__":
    main()
