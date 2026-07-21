#!/usr/bin/env python3
"""GT-safe region area in GT-CDF / tail-mass coordinates (not raw thr).

Reject (right-tail AND):
  reject if A > thr_a AND B > thr_b

Coordinate (default):
  u = P_GT(score > thr)   # GT right-tail mass in [0,1]
  thr(u) = quantile_GT(1 - u)

Then search_domain is the unit square in (u_a, u_b). Area ratios are comparable
across signals with different raw units.

Metrics
-------
  safe_area_ratio(ε)
  productive_safe_area@α  (FP >= α * best_FP under ε)
  robust_safe_area_ratio  (∩_seq S_ε,s)
  best_point + distance_to_unsafe_boundary + plateau widths
  classification

  uv run python scripts/tools/gt_safe_region_area.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_gt_safe_area_<stamp> \\
    --all-default-pairs --n-grid 25
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
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
DEFAULT_PAIRS = [
    ("score_m_bridge", "abs_log_h"),
    ("bridge_dist", "abs_log_h"),
    ("score_m_bridge", "abs_ratio_m1"),
    ("resid_mean", "abs_log_h"),
    ("dist_h", "abs_log_h"),
    ("score_m_bridge", "neg_dir_cos"),
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


def thr_from_gt_tail(pos: np.ndarray, u: float) -> float:
    """thr such that P_GT(score > thr) ≈ u (right-tail mass).

    u→0 ⇒ thr high (strict); u→1 ⇒ thr low (aggressive).
    """
    u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
    # quantile at 1-u: fraction below thr is 1-u ⇒ fraction above ≈ u
    return float(np.quantile(pos, 1.0 - u))


def metrics_and(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    thr_a: float,
    thr_b: float,
    seq: np.ndarray | None = None,
) -> dict[str, Any]:
    rej = (a > thr_a) & (b > thr_b)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    hurt = int((y & rej).sum())
    fprm = int((~y & rej).sum())
    out = {
        "GT_hurt": hurt,
        "GT_hurt_rate": hurt / n_pos if n_pos else 0.0,
        "FP_removed": fprm,
        "FP_removed_rate": fprm / n_neg if n_neg else 0.0,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    if seq is not None:
        rates = []
        for s in np.unique(seq):
            m = seq == s
            ys = y[m]
            if ys.sum() == 0:
                continue
            rates.append(float((ys & rej[m]).sum() / ys.sum()))
        out["seq_hurt_std"] = float(np.std(rates)) if len(rates) >= 2 else 0.0
        out["per_seq_hurt"] = {
            str(s): float(
                (y[seq == s] & rej[seq == s]).sum() / max(y[seq == s].sum(), 1)
            )
            for s in np.unique(seq)
            if y[seq == s].sum() > 0
        }
    return out


def classify(
    *,
    safe_ratio: float,
    prod80: float,
    robust_ratio: float,
    boundary_dist: float | None,
    seq_unstable: bool,
) -> str:
    if seq_unstable:
        return "seq_unstable"
    if safe_ratio < 0.01:
        return "isolated_sweet_spot"
    if prod80 < 1e-6 and safe_ratio >= 0.05:
        return "safe_but_unproductive"
    if safe_ratio < 0.05:
        return "thin_but_promising" if prod80 > 0 else "isolated_sweet_spot"
    if safe_ratio < 0.15:
        if prod80 >= 0.02 and (boundary_dist or 0) >= 0.02:
            return "usable_safe_region"
        return "thin_but_promising"
    # broad
    if prod80 >= 0.03 and (boundary_dist or 0) >= 0.03 and robust_ratio >= 0.01:
        return "broad_safe_productive"
    if prod80 < 1e-6:
        return "safe_but_unproductive"
    return "usable_safe_region"


def analyze_pair_gt_cdf(
    pool: dict[str, np.ndarray],
    name_a: str,
    name_b: str,
    ax: dict[str, Axis],
    *,
    n_grid: int = 25,
    mode: str = "AND",
    alphas: tuple[float, ...] = (0.8, 0.9),
) -> dict[str, Any]:
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    a = np.asarray(ax[name_a].extract(pool), dtype=float)
    b = np.asarray(ax[name_b].extract(pool), dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    b = np.where(np.isfinite(b), b, 0.0)
    pos_a, pos_b = a[y], b[y]

    # uniform grid in GT tail-mass space
    us = np.linspace(0.02, 0.98, n_grid)
    du = float(us[1] - us[0]) if n_grid > 1 else 1.0
    cell_area = du * du  # uniform in u-space

    # precompute thr for each u
    thr_a = {float(u): thr_from_gt_tail(pos_a, float(u)) for u in us}
    thr_b = {float(u): thr_from_gt_tail(pos_b, float(u)) for u in us}

    grid: list[dict[str, Any]] = []
    for ua in us:
        for ub in us:
            ta, tb = thr_a[float(ua)], thr_b[float(ub)]
            if mode == "AND":
                m = metrics_and(y, a, b, ta, tb, seq=seq)
            else:
                # OR: reject if A>ta OR B>tb
                rej = (a > ta) | (b > tb)
                n_pos, n_neg = int(y.sum()), int((~y).sum())
                hurt = int((y & rej).sum())
                fprm = int((~y & rej).sum())
                m = {
                    "GT_hurt": hurt,
                    "GT_hurt_rate": hurt / n_pos if n_pos else 0.0,
                    "FP_removed": fprm,
                    "FP_removed_rate": fprm / n_neg if n_neg else 0.0,
                    "per_seq_hurt": {},
                    "seq_hurt_std": 0.0,
                }
            row = {
                "u_a": float(ua),
                "u_b": float(ub),
                "thr_a": ta,
                "thr_b": tb,
                "GT_hurt": m["GT_hurt"],
                "GT_hurt_rate": m["GT_hurt_rate"],
                "FP_removed": m["FP_removed"],
                "FP_removed_rate": m["FP_removed_rate"],
                "seq_hurt_std": m.get("seq_hurt_std", 0.0),
                "per_seq_hurt": m.get("per_seq_hurt") or {},
            }
            grid.append(row)

    n_cells = len(grid)

    # effective domain area of grid coverage in u-space
    u_lo, u_hi = float(us[0]), float(us[-1])
    domain_measure = (u_hi - u_lo) ** 2

    by_eps: dict[str, Any] = {}
    for eps in EPS_LADDER:
        safe = [g for g in grid if g["GT_hurt_rate"] <= eps + 1e-15]
        safe_area = len(safe) * cell_area
        safe_ratio = safe_area / domain_measure if domain_measure > 0 else 0.0

        best_fp = max((g["FP_removed"] for g in safe), default=0)
        best = None
        if safe:
            best = max(safe, key=lambda g: (g["FP_removed"], -g["GT_hurt_rate"]))

        prod = {}
        for alpha in alphas:
            thr_fp = alpha * best_fp if best_fp > 0 else float("inf")
            prod_cells = [g for g in safe if g["FP_removed"] >= thr_fp - 1e-9]
            pa = len(prod_cells) * cell_area
            prod[f"productive_safe_area@{int(alpha * 100)}"] = pa
            prod[f"productive_safe_area_ratio@{int(alpha * 100)}"] = (
                pa / domain_measure if domain_measure > 0 else 0.0
            )

        # robust: cell safe for ALL sequences (intersection of per-seq safe sets)
        robust_cells = []
        for g in safe:
            psh = g.get("per_seq_hurt") or {}
            if psh and all(v <= eps + 1e-15 for v in psh.values()):
                robust_cells.append(g)

        robust_area = len(robust_cells) * cell_area
        robust_ratio = robust_area / domain_measure if domain_measure > 0 else 0.0

        # boundary distance of best: min L_inf distance in u-space to unsafe cell
        boundary_dist = None
        plateau_wa = plateau_wb = 0.0
        if best is not None:
            unsafe = [g for g in grid if g["GT_hurt_rate"] > eps + 1e-15]
            if unsafe:
                dists = [
                    max(abs(g["u_a"] - best["u_a"]), abs(g["u_b"] - best["u_b"]))
                    for g in unsafe
                ]
                boundary_dist = float(min(dists))
            else:
                boundary_dist = float(
                    min(best["u_a"], best["u_b"], 1 - best["u_a"], 1 - best["u_b"])
                )
            # plateau: productive@80 neighborhood of best
            if best_fp > 0:
                near = [g for g in safe if g["FP_removed"] >= 0.8 * best_fp]
                if near:
                    plateau_wa = float(
                        max(g["u_a"] for g in near) - min(g["u_a"] for g in near)
                    )
                    plateau_wb = float(
                        max(g["u_b"] for g in near) - min(g["u_b"] for g in near)
                    )

        # per-seq safe area (each seq as if only that seq's GT_hurt)
        per_seq_ratios = []
        for s in sorted(set(seq.tolist())):
            sm = seq == s
            ys = y[sm]
            if ys.sum() < 1:
                continue
            # local safe: cells where this seq has hurt_rate <= eps
            n_safe_s = 0
            for g in grid:
                rej = (a > g["thr_a"]) & (b > g["thr_b"])
                hs = int((ys & rej[sm]).sum())
                if hs / ys.sum() <= eps + 1e-15:
                    n_safe_s += 1
            per_seq_ratios.append(n_safe_s * cell_area / domain_measure)

        seq_unstable = False
        if per_seq_ratios:
            if min(per_seq_ratios) < 0.01 and safe_ratio > 0.05:
                seq_unstable = True

        cls = classify(
            safe_ratio=safe_ratio,
            prod80=prod.get("productive_safe_area_ratio@80", 0.0),
            robust_ratio=robust_ratio,
            boundary_dist=boundary_dist,
            seq_unstable=seq_unstable,
        )

        by_eps[str(eps)] = {
            "epsilon": eps,
            "coordinate_space": "GT_tail_mass",
            "n_grid": n_grid,
            "n_cells": n_cells,
            "domain_u_lo": u_lo,
            "domain_u_hi": u_hi,
            "domain_measure": domain_measure,
            "cell_area": cell_area,
            "safe_area": safe_area,
            "safe_area_ratio": safe_ratio,
            "n_safe_cells": len(safe),
            **prod,
            "robust_safe_area": robust_area,
            "robust_safe_area_ratio": robust_ratio,
            "n_robust_cells": len(robust_cells),
            "best_FP_removed": best_fp,
            "best_GT_hurt": best["GT_hurt"] if best else None,
            "best_GT_hurt_rate": best["GT_hurt_rate"] if best else None,
            "best_u_a": best["u_a"] if best else None,
            "best_u_b": best["u_b"] if best else None,
            "best_thr_a": best["thr_a"] if best else None,
            "best_thr_b": best["thr_b"] if best else None,
            "best_point_boundary_distance": boundary_dist,
            "plateau_width_a": plateau_wa,
            "plateau_width_b": plateau_wb,
            "plateau_width_min": min(plateau_wa, plateau_wb),
            "per_seq_safe_area_min": float(min(per_seq_ratios))
            if per_seq_ratios
            else None,
            "per_seq_safe_area_std": float(np.std(per_seq_ratios))
            if per_seq_ratios
            else None,
            "per_seq_safe_area_mean": float(np.mean(per_seq_ratios))
            if per_seq_ratios
            else None,
            "classification": cls,
        }

    return {
        "pair": f"{name_a}__{name_b}",
        "rule_a": name_a,
        "rule_b": name_b,
        "mode": mode,
        "coordinate_space": "GT_tail_mass_u=P_GT(score>thr)",
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "by_eps": by_eps,
        # store grid only for eps=0 summary size control — optional thin dump
        "_grid_meta": {"n_grid": n_grid, "us": us.tolist()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument("--pair", action="append", default=None)
    ap.add_argument("--all-default-pairs", action="store_true")
    ap.add_argument("--n-grid", type=int, default=25)
    ap.add_argument("--mode", default="AND", choices=["AND", "OR"])
    args = ap.parse_args()

    ax = axes()
    pair_list: list[tuple[str, str]] = []
    if args.all_default_pairs:
        pair_list.extend(DEFAULT_PAIRS)
    if args.pair:
        for p in args.pair:
            a, b = p.split(",")
            pair_list.append((a.strip(), b.strip()))
    if not pair_list:
        raise SystemExit("pass --pair A,B and/or --all-default-pairs")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_gt_safe_area_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)

    batch: dict[str, Any] = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "exposure": _audit.exposure_summary(pool),
        "coordinate_space": "GT_tail_mass",
        "definition": {
            "u": "P_GT(score > thr)",
            "thr_u": "quantile_GT(1-u)",
            "S_eps": "GT_hurt_rate(theta) <= eps",
            "primary_metric": "productive_safe_area_ratio@80",
            "not_raw_thr_area": True,
        },
        "pairs": {},
    }

    summary_rows = []
    print(f"STUDY={study}")
    print(
        f"{'pair':<36} {'eps':>6} {'safe%':>7} {'p80%':>7} {'rob%':>7} "
        f"{'bestFP':>7} {'bdist':>6} {'class'}"
    )

    for na, nb in pair_list:
        res = analyze_pair_gt_cdf(pool, na, nb, ax, n_grid=args.n_grid, mode=args.mode)
        batch["pairs"][res["pair"]] = {k: res[k] for k in res if not k.startswith("_")}
        sub = study / "pairs" / res["pair"]
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps(batch["pairs"][res["pair"]], indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        for eps in EPS_LADDER:
            e = res["by_eps"][str(eps)]
            summary_rows.append(
                {
                    "rule_name": res["pair"],
                    "mode": args.mode,
                    "epsilon": eps,
                    "search_domain": f"u_tail∈[{e['domain_u_lo']:.2f},{e['domain_u_hi']:.2f}]^2",
                    "coordinate_space": "GT_tail_mass",
                    "safe_area_ratio": e["safe_area_ratio"],
                    "productive_safe_area@80": e["productive_safe_area_ratio@80"],
                    "productive_safe_area@90": e["productive_safe_area_ratio@90"],
                    "robust_safe_area_ratio": e["robust_safe_area_ratio"],
                    "best_FP_removed": e["best_FP_removed"],
                    "best_GT_hurt": e["best_GT_hurt"],
                    "best_point_boundary_distance": e["best_point_boundary_distance"],
                    "plateau_width_min": e["plateau_width_min"],
                    "per_seq_safe_area_min": e["per_seq_safe_area_min"],
                    "per_seq_safe_area_std": e["per_seq_safe_area_std"],
                    "classification": e["classification"],
                }
            )
            if eps in (0.0, 0.01):
                print(
                    f"{res['pair']:<36} {eps:6.3g} "
                    f"{100 * e['safe_area_ratio']:6.2f}% "
                    f"{100 * e['productive_safe_area_ratio@80']:6.2f}% "
                    f"{100 * e['robust_safe_area_ratio']:6.2f}% "
                    f"{e['best_FP_removed']:7d} "
                    f"{(e['best_point_boundary_distance'] or 0):6.3f} "
                    f"{e['classification']}"
                )

    cols = list(summary_rows[0].keys()) if summary_rows else []
    with (study / "safe_area_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)

    (study / "summary.json").write_text(
        json.dumps(batch, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {study / 'safe_area_table.csv'}")
    print(
        "NOTE: areas in GT_tail_mass unit square — not raw thr. "
        "Primary: productive_safe_area@80 + robust + boundary_distance."
    )


if __name__ == "__main__":
    main()
