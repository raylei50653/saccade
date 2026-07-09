#!/usr/bin/env python3
"""2D combination-gate surface + safe-region audit.

Why AND (intersection) can recover more thresholds
--------------------------------------------------
Single-signal safe thr is often thin or empty. Intersection:

  reject if A > ta AND B > tb

lets both ta and tb loosen while still protecting GT, because FP often fail
multiple signals together while GT usually fails only one.

Do NOT only report best (ta,tb). Production stability needs:

  safe_region = {(ta,tb) | GT_hurt(ta,tb) <= ε}
  safe_region_area, max/mean FP in region, plateau widths

Trap: thr on monotone transforms is equivalent for 1D; for AND of two
different physical quantities the surface is not a monotone reparam of either.

Usage
-----
  uv run python scripts/tools/combo_gate_safe_region.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_combo_safe_<stamp> \\
    --pair score_m_bridge,abs_log_h \\
    --pair bridge_dist,abs_log_h \\
    --all-default-pairs

Contract: signal_table_schema §0.4 safe-reject asymmetry (GT hard / FP soft)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

_AUDIT = Path(__file__).resolve().parent / "audit_relink_safe_reject.py"
_spec = importlib.util.spec_from_file_location("audit_relink_safe_reject", _AUDIT)
assert _spec and _spec.loader
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)

from saccade.perception.eval.signal_tables import (  # noqa: E402
    constrained_fp_prune_metrics,
)

EPSILONS = (0.0, 0.001, 0.01)


@dataclass(frozen=True)
class Axis:
    name: str
    extract: Callable[[dict[str, np.ndarray]], np.ndarray]
    """Higher => more reject-like (worse for GT association)."""
    notes: str = ""


def axis_catalog() -> dict[str, Axis]:
    return {
        "score_m_bridge": Axis(
            "score_m_bridge",
            lambda p: p["score_m_bridge"],
            "live-shaped bridge energy",
        ),
        "bridge_dist": Axis(
            "bridge_dist",
            lambda p: p["bridge_dist"],
            "mid-point bridge_dist",
        ),
        "dist_h": Axis("dist_h", lambda p: p["dist_h"], "foot dist/h"),
        "resid_mean": Axis(
            "resid_mean",
            lambda p: 0.5 * (p["fwd_resid"] + p["bwd_resid"]),
            "mean residual",
        ),
        "abs_log_h": Axis(
            "abs_log_h",
            lambda p: p["log_h_ratio"],
            "|log h_ratio|",
        ),
        "abs_ratio_m1": Axis(
            "abs_ratio_m1",
            lambda p: np.abs(p["h_ratio_lost_over_cand"] - 1.0),
            "|h_ratio-1|",
        ),
        "speed_mismatch": Axis(
            "speed_mismatch",
            lambda p: p["speed_mismatch"],
            "|exit-entry| speed",
        ),
        "neg_dir_cos": Axis(
            "neg_dir_cos",
            lambda p: -p["dir_cos"],
            "−dir_cos (higher = worse)",
        ),
    }


DEFAULT_PAIRS = [
    ("score_m_bridge", "abs_log_h"),
    ("bridge_dist", "abs_log_h"),
    ("score_m_bridge", "abs_ratio_m1"),
    ("resid_mean", "abs_log_h"),
    ("score_m_bridge", "neg_dir_cos"),
    ("abs_log_h", "speed_mismatch"),
    ("dist_h", "abs_log_h"),
]


def thr_grid_from_data(
    a: np.ndarray,
    *,
    n: int = 24,
    q_lo: float = 0.50,
    q_hi: float = 0.995,
) -> np.ndarray:
    """Quantile grid on score (higher = more reject). Covers mid→far tail."""
    qs = np.linspace(q_lo, q_hi, n)
    thr = np.unique(np.quantile(a, qs))
    # ensure sorted unique
    return np.sort(thr)


def metrics_at(
    y: np.ndarray,
    rej: np.ndarray,
    *,
    rule_name: str = "",
) -> dict[str, Any]:
    m = constrained_fp_prune_metrics(
        y, rej, rule_name=rule_name or "combo", coverage_bin="all"
    )
    return m


def boundary_mass(
    score: np.ndarray, thr: float, y: np.ndarray, band: float = 0.2
) -> float:
    """Fraction of GT with score in [thr*(1-band), thr*(1+band)]."""
    pos = score[y]
    if pos.size == 0 or thr <= 0:
        return float("nan")
    lo, hi = thr * (1.0 - band), thr * (1.0 + band)
    return float(np.mean((pos >= lo) & (pos <= hi)))


def single_axis_frontier(
    y: np.ndarray,
    score: np.ndarray,
    thr_grid: np.ndarray,
    epsilons: tuple[float, ...] = EPSILONS,
) -> dict[str, Any]:
    """Best single thr per ε and count of valid thr on grid."""
    out: dict[str, Any] = {"by_eps": {}, "grid_rows": []}
    for thr in thr_grid:
        rej = score > thr
        m = metrics_at(y, rej)
        out["grid_rows"].append(
            {
                "thr": float(thr),
                "GT_hurt": m["GT_hurt"],
                "GT_hurt_rate": m["GT_hurt_rate"],
                "FP_removed": m["FP_removed"],
                "FP_removed_rate": m["FP_removed_rate"],
                "safe_level": m["safe_level"],
            }
        )
    for eps in epsilons:
        valid = [r for r in out["grid_rows"] if r["GT_hurt_rate"] <= eps + 1e-15]
        if not valid:
            out["by_eps"][str(eps)] = {
                "feasible": False,
                "best_FP_removed": 0,
                "best_thr": None,
                "n_valid": 0,
                "valid_fraction": 0.0,
            }
            continue
        best = max(valid, key=lambda r: r["FP_removed"])
        out["by_eps"][str(eps)] = {
            "feasible": True,
            "best_FP_removed": best["FP_removed"],
            "best_FP_removed_rate": best["FP_removed_rate"],
            "best_thr": best["thr"],
            "best_GT_hurt": best["GT_hurt"],
            "best_GT_hurt_rate": best["GT_hurt_rate"],
            "n_valid": len(valid),
            "valid_fraction": len(valid) / max(len(out["grid_rows"]), 1),
            "thr_min_valid": min(r["thr"] for r in valid),
            "thr_max_valid": max(r["thr"] for r in valid),
            # width in thr units (higher thr = more reject = looser for lower-is-better energy?
            # score > thr: higher thr = fewer rejects = safer. So valid thr tend to be high.
            "plateau_width": max(r["thr"] for r in valid)
            - min(r["thr"] for r in valid),
        }
    return out


def combo_surface(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    thr_a: np.ndarray,
    thr_b: np.ndarray,
    *,
    mode: str = "AND",
    seq: np.ndarray | None = None,
    name_a: str = "A",
    name_b: str = "B",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full 2D grid + aggregate safe-region stats."""
    rows: list[dict[str, Any]] = []
    for ta in thr_a:
        for tb in thr_b:
            ra = score_a > ta
            rb = score_b > tb
            if mode == "AND":
                rej = ra & rb
            elif mode == "OR":
                rej = ra | rb
            else:
                raise ValueError(mode)
            m = metrics_at(y, rej, rule_name=f"{name_a}>{ta:.4g}&{name_b}>{tb:.4g}")
            # per-seq hurt rates for std
            seq_std = None
            if seq is not None:
                rates = []
                for s in np.unique(seq):
                    sm = seq == s
                    ys = y[sm]
                    if ys.sum() == 0:
                        continue
                    rates.append(float((ys & rej[sm]).sum() / ys.sum()))
                if len(rates) >= 2:
                    seq_std = float(np.std(rates))
            rows.append(
                {
                    "rule_a": name_a,
                    "rule_b": name_b,
                    "ta": float(ta),
                    "tb": float(tb),
                    "mode": mode,
                    "GT_hurt": m["GT_hurt"],
                    "GT_hurt_rate": m["GT_hurt_rate"],
                    "FP_removed": m["FP_removed"],
                    "FP_removed_rate": m["FP_removed_rate"],
                    "FP_removed_per_GT_hurt": m["FP_removed_per_GT_hurt"],
                    "safe_level": m["safe_level"],
                    "epsilon_pass_0": m["GT_hurt_rate"] <= 0.0 + 1e-15,
                    "epsilon_pass_0.001": m["GT_hurt_rate"] <= 0.001 + 1e-15,
                    "epsilon_pass_0.01": m["GT_hurt_rate"] <= 0.01 + 1e-15,
                    "seq_hurt_std": seq_std,
                    "boundary_mass_a": boundary_mass(score_a, float(ta), y),
                    "boundary_mass_b": boundary_mass(score_b, float(tb), y),
                    "n_reject": int(rej.sum()),
                }
            )

    agg: dict[str, Any] = {"mode": mode, "n_grid": len(rows), "by_eps": {}}
    n_grid = max(len(rows), 1)
    for eps in EPSILONS:
        valid = [r for r in rows if r["GT_hurt_rate"] <= eps + 1e-15]
        key = str(eps)
        if not valid:
            agg["by_eps"][key] = {
                "feasible": False,
                "n_valid": 0,
                "valid_area_ratio": 0.0,
                "safe_region_area": 0,
                "max_FP_in_safe_region": 0,
                "mean_FP_in_safe_region": 0.0,
                "best": None,
            }
            continue
        best = max(valid, key=lambda r: (r["FP_removed"], -r["GT_hurt"]))
        fps = [r["FP_removed"] for r in valid]
        # plateau: among points with FP >= 0.9 * best FP
        near = [r for r in valid if r["FP_removed"] >= 0.9 * best["FP_removed"]]
        ta_near = [r["ta"] for r in near]
        tb_near = [r["tb"] for r in near]
        # connected-ish area: count cells; also thr span of near-best plateau
        agg["by_eps"][key] = {
            "feasible": True,
            "n_valid": len(valid),
            "valid_area_ratio": len(valid) / n_grid,
            "safe_region_area": len(valid),  # cell count on discrete grid
            "max_FP_in_safe_region": int(max(fps)),
            "mean_FP_in_safe_region": float(np.mean(fps)),
            "median_FP_in_safe_region": float(np.median(fps)),
            "best": {
                "ta": best["ta"],
                "tb": best["tb"],
                "FP_removed": best["FP_removed"],
                "FP_removed_rate": best["FP_removed_rate"],
                "GT_hurt": best["GT_hurt"],
                "GT_hurt_rate": best["GT_hurt_rate"],
                "seq_hurt_std": best["seq_hurt_std"],
                "boundary_mass_a": best["boundary_mass_a"],
                "boundary_mass_b": best["boundary_mass_b"],
            },
            "plateau_near_best": {
                "n_cells": len(near),
                "fraction_of_safe": len(near) / len(valid),
                "ta_span": float(max(ta_near) - min(ta_near)) if near else 0.0,
                "tb_span": float(max(tb_near) - min(tb_near)) if near else 0.0,
                "isolated": len(near) <= 2,
            },
            # how many distinct ta / tb appear in safe region
            "n_unique_ta": len({r["ta"] for r in valid}),
            "n_unique_tb": len({r["tb"] for r in valid}),
            "mean_seq_hurt_std": float(
                np.nanmean(
                    [r["seq_hurt_std"] for r in valid if r["seq_hurt_std"] is not None]
                )
            )
            if any(r["seq_hurt_std"] is not None for r in valid)
            else None,
        }
    return rows, agg


def analyze_pair(
    pool: dict[str, np.ndarray],
    name_a: str,
    name_b: str,
    axes: dict[str, Axis],
    *,
    n_grid: int = 20,
    modes: tuple[str, ...] = ("AND",),
) -> dict[str, Any]:
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    sa = np.asarray(axes[name_a].extract(pool), dtype=float)
    sb = np.asarray(axes[name_b].extract(pool), dtype=float)
    sa = np.where(np.isfinite(sa), sa, 0.0)
    sb = np.where(np.isfinite(sb), sb, 0.0)

    thr_a = thr_grid_from_data(sa, n=n_grid)
    thr_b = thr_grid_from_data(sb, n=n_grid)

    single_a = single_axis_frontier(y, sa, thr_a)
    single_b = single_axis_frontier(y, sb, thr_b)

    result: dict[str, Any] = {
        "pair": f"{name_a}__{name_b}",
        "rule_a": name_a,
        "rule_b": name_b,
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "thr_a_grid": thr_a.tolist(),
        "thr_b_grid": thr_b.tolist(),
        "single_a": {k: v for k, v in single_a.items() if k != "grid_rows"},
        "single_b": {k: v for k, v in single_b.items() if k != "grid_rows"},
        "modes": {},
        "comparison": {},
    }

    all_surface_rows: list[dict[str, Any]] = []
    for mode in modes:
        rows, agg = combo_surface(
            y, sa, sb, thr_a, thr_b, mode=mode, seq=seq, name_a=name_a, name_b=name_b
        )
        result["modes"][mode] = agg
        all_surface_rows.extend(rows)

        # gain vs best single
        for eps in EPSILONS:
            ek = str(eps)
            and_e = agg["by_eps"][ek]
            ba = single_a["by_eps"][ek]
            bb = single_b["by_eps"][ek]
            best_single_fp = max(
                ba.get("best_FP_removed") or 0,
                bb.get("best_FP_removed") or 0,
            )
            best_and_fp = and_e.get("max_FP_in_safe_region") or 0
            gain = best_and_fp - best_single_fp
            # threshold recoverability: more valid cells than either single
            n_a = ba.get("n_valid") or 0
            n_b = bb.get("n_valid") or 0
            n_and = and_e.get("n_valid") or 0
            n_best_single = max(n_a, n_b)
            # area recovery: AND opens many more (ta,tb) than either 1D thr count
            area_ratio = (
                (n_and / max(n_best_single, 1)) if n_best_single else float(n_and)
            )
            fp_ratio = best_and_fp / max(best_single_fp, 1) if best_single_fp else 0.0
            plateau = and_e.get("plateau_near_best") or {}
            isolated = bool(plateau.get("isolated", True))

            # Two distinct gains (both matter for production):
            # 1) marginal_FP_gain: best AND FP > best single FP under ε
            # 2) threshold_recoverability: safe region area >> single valid set
            #    even when best FP is slightly lower
            recoverability = bool(
                and_e.get("feasible")
                and n_and >= 8
                and area_ratio >= 3.0
                and fp_ratio >= 0.50  # still keep half the best-single FP capacity
            )
            marginal = bool(
                and_e.get("feasible") and gain > 0 and n_and >= 4 and not isolated
            )
            recov = {
                "epsilon": eps,
                "best_single_A_FP": ba.get("best_FP_removed") or 0,
                "best_single_B_FP": bb.get("best_FP_removed") or 0,
                "best_single_FP": best_single_fp,
                "best_AND_FP": best_and_fp,
                "AND_gain_over_best_single": gain,
                "AND_FP_ratio_vs_best_single": fp_ratio,
                "single_A_n_valid": n_a,
                "single_B_n_valid": n_b,
                "AND_n_valid": n_and,
                "AND_valid_area_ratio": and_e.get("valid_area_ratio") or 0.0,
                "single_A_valid_fraction": ba.get("valid_fraction") or 0.0,
                "single_B_valid_fraction": bb.get("valid_fraction") or 0.0,
                "area_cell_ratio_vs_best_single": area_ratio,
                "area_gain_vs_best_single_frac": (
                    (and_e.get("valid_area_ratio") or 0)
                    - max(ba.get("valid_fraction") or 0, bb.get("valid_fraction") or 0)
                ),
                "plateau_isolated": isolated,
                "plateau_n_cells": plateau.get("n_cells"),
                "plateau_ta_span": plateau.get("ta_span"),
                "plateau_tb_span": plateau.get("tb_span"),
                "marginal_FP_gain": marginal,
                "threshold_recoverability": recoverability,
                "worth_keep": bool(marginal or recoverability),
                "worth_keep_soft": bool(
                    marginal
                    or recoverability
                    or (
                        and_e.get("feasible")
                        and n_and >= 2 * max(n_best_single, 1)
                        and fp_ratio >= 0.4
                    )
                ),
            }
            result["comparison"].setdefault(mode, {})[ek] = recov

    result["_surface_rows"] = all_surface_rows
    result["_single_a_rows"] = single_a["grid_rows"]
    result["_single_b_rows"] = single_b["grid_rows"]
    return result


def write_pair_csvs(study: Path, pair_result: dict[str, Any]) -> None:
    pair = pair_result["pair"]
    sub = study / "pairs" / pair
    sub.mkdir(parents=True, exist_ok=True)
    # surface
    rows = pair_result.pop("_surface_rows", [])
    if rows:
        cols = list(rows[0].keys())
        with (sub / "surface.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    # singles
    for tag, key in (("single_a", "_single_a_rows"), ("single_b", "_single_b_rows")):
        srows = pair_result.pop(key, [])
        if srows:
            with (sub / f"{tag}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
                w.writeheader()
                w.writerows(srows)
    # summary without huge grids
    slim = {k: v for k, v in pair_result.items() if not k.startswith("_")}
    # drop thr grids from json if huge — keep them, useful
    (sub / "summary.json").write_text(
        json.dumps(slim, indent=2, default=float) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, default=None)
    ap.add_argument(
        "--pair",
        action="append",
        default=None,
        help="name_a,name_b (repeatable)",
    )
    ap.add_argument(
        "--all-default-pairs",
        action="store_true",
        help="run DEFAULT_PAIRS",
    )
    ap.add_argument("--n-grid", type=int, default=20, help="quantile points per axis")
    ap.add_argument(
        "--modes",
        default="AND",
        help="comma list: AND,OR",
    )
    ap.add_argument("--list-axes", action="store_true")
    args = ap.parse_args()

    axes = axis_catalog()
    if args.list_axes:
        for k, ax in axes.items():
            print(f"{k:<18} {ax.notes}")
        return

    pair_list: list[tuple[str, str]] = []
    if args.all_default_pairs:
        pair_list.extend(DEFAULT_PAIRS)
    if args.pair:
        for p in args.pair:
            a, b = p.split(",")
            pair_list.append((a.strip(), b.strip()))
    if not pair_list:
        raise SystemExit("pass --pair A,B and/or --all-default-pairs")

    for a, b in pair_list:
        if a not in axes or b not in axes:
            raise SystemExit(f"unknown axis in {a},{b}; use --list-axes")

    modes = tuple(m.strip().upper() for m in args.modes.split(",") if m.strip())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_combo_safe_{stamp}")
    study.mkdir(parents=True, exist_ok=True)

    pool = _audit.load_gt_valid_pool(args.pairs)
    _audit.ensure_prod_proxy_scores(pool)

    batch: dict[str, Any] = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "n_grid_per_axis": args.n_grid,
        "modes": list(modes),
        "epsilons": list(EPSILONS),
        "principle": {
            "intersection_recovers_thresholds": True,
            "report_safe_region_not_only_best": True,
            "gains": ["marginal_FP_gain", "threshold_recoverability"],
        },
        "pairs": {},
    }

    print(f"STUDY={study}")
    print(
        f"{'pair':<36} {'eps':>5} {'1A_FP':>6} {'1B_FP':>6} {'AND_FP':>6} "
        f"{'dFP':>6} {'nvA':>4} {'nvB':>4} {'nv&':>5} {'&/1':>5} "
        f"{'recov':>5} {'mFP':>4} {'keep':>4}"
    )

    for name_a, name_b in pair_list:
        pr = analyze_pair(pool, name_a, name_b, axes, n_grid=args.n_grid, modes=modes)
        write_pair_csvs(study, pr)
        # print eps0 and eps1% AND
        for mode in modes:
            if mode != "AND":
                continue
            for eps in (0.0, 0.01):
                c = pr["comparison"][mode][str(eps)]
                print(
                    f"{pr['pair']:<36} {eps:5.3g} "
                    f"{c['best_single_A_FP']:6d} {c['best_single_B_FP']:6d} "
                    f"{c['best_AND_FP']:6d} {c['AND_gain_over_best_single']:6d} "
                    f"{c['single_A_n_valid']:4d} {c['single_B_n_valid']:4d} "
                    f"{c['AND_n_valid']:5d} "
                    f"{c['area_cell_ratio_vs_best_single']:5.1f} "
                    f"{'Y' if c.get('threshold_recoverability') else 'n':>5} "
                    f"{'Y' if c.get('marginal_FP_gain') else 'n':>4} "
                    f"{'Y' if c.get('worth_keep') else 'n':>4}"
                )
        batch["pairs"][pr["pair"]] = {
            "comparison": pr["comparison"],
            "modes": {
                m: {
                    "by_eps": pr["modes"][m]["by_eps"],
                    "n_grid": pr["modes"][m]["n_grid"],
                }
                for m in pr["modes"]
            },
            "single_a_by_eps": pr["single_a"]["by_eps"],
            "single_b_by_eps": pr["single_b"]["by_eps"],
        }

    (study / "summary.json").write_text(
        json.dumps(batch, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {study / 'summary.json'}")
    print(f"Per-pair surfaces under {study / 'pairs'}/")


if __name__ == "__main__":
    main()
