#!/usr/bin/env python3
"""Offline B1 safe-reject audit: max FP removal under GT_hurt <= ε.

This is **not** thr F1 tuning. Goal:

  maximize FP_removed          # soft upper bound
  s.t.     GT_hurt_rate <= ε   # hard loss
  ε ∈ {0, 0.1%, 1%}

Asymmetry (early gate):
  * Rejected **GT** pairs are usually *gone* — later assignment / scoring / NMS
    / metrics cannot resurrect that offline true-relink chance.
  * Surviving **FP** pairs may still be killed downstream — so FP_removed is a
    soft bound on negative-pool reduction, not an e2e FP guarantee.
  ⇒ Prefer ε=0 (safe_reject). Never trade hard GT loss for soft FP gains without
    an explicit fallback study.

Layering:
  * thr(gap) / coverage rules → calibration (protect GT)
  * context reject rules → safe / risky FP pruning

Usage:
  uv run python scripts/tools/audit_relink_safe_reject.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/<id> \\
    --write-study --by-gap --by-seq

Includes production-shaped m gates (px=0.4, h∈[0.6,1.7] offline proxy)
unless --no-prod-shaped. 7-seq MOT is cheap (~30s); re-run substrate freely.

Contract: docs/research/contracts/signal_table_schema.md §0.4
"""
# status: diagnostic

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from saccade.perception.eval.signal_tables import (
    SAFE_REJECT_AUDIT_COLS,
    SAFE_REJECT_AUDIT_FILENAME,
    SAFE_REJECT_SUMMARY_FILENAME,
    constrained_fp_prune_metrics,
    frontier_fp_removed_at_eps,
)

RuleFn = Callable[[dict[str, np.ndarray]], np.ndarray]


def _as_bool01(v: Any) -> int:
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no", ""):
        return 0
    return int(float(s))


def load_gt_valid_pool(pairs_csv: Path) -> dict[str, np.ndarray]:
    """Load builder CSV; keep gt_valid==1 only."""
    cols: dict[str, list[Any]] = {
        "gt_match": [],
        "gap": [],
        "bridge_dist": [],
        "dir_cos": [],
        "dist_h": [],
        "fwd_resid": [],
        "bwd_resid": [],
        "speed_h": [],
        "lost_exit_speed": [],
        "cand_entry_speed": [],
        "h_lost_raw": [],
        "h_cand_raw": [],
        "seq": [],
        "lost_id": [],
    }
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if _as_bool01(row.get("gt_valid", 0)) != 1:
                continue
            cols["gt_match"].append(_as_bool01(row["gt_match"]))
            cols["gap"].append(float(row["gap"]))
            cols["bridge_dist"].append(float(row["bridge_dist"]))
            cols["dir_cos"].append(float(row.get("dir_cos", 0.0) or 0.0))
            cols["dist_h"].append(float(row.get("dist_h", 0.0) or 0.0))
            cols["fwd_resid"].append(float(row.get("fwd_resid", 0.0) or 0.0))
            cols["bwd_resid"].append(float(row.get("bwd_resid", 0.0) or 0.0))
            cols["speed_h"].append(float(row.get("speed_h", 0.0) or 0.0))
            cols["lost_exit_speed"].append(
                float(row.get("lost_exit_speed", 0.0) or 0.0)
            )
            cols["cand_entry_speed"].append(
                float(row.get("cand_entry_speed", 0.0) or 0.0)
            )
            cols["h_lost_raw"].append(float(row.get("h_lost_raw", 1.0) or 1.0))
            cols["h_cand_raw"].append(float(row.get("h_cand_raw", 1.0) or 1.0))
            cols["seq"].append(row.get("seq", ""))
            cols["lost_id"].append(str(row.get("lost_id", "")))

    out: dict[str, np.ndarray] = {}
    for k, v in cols.items():
        if k in ("seq", "lost_id"):
            out[k] = np.asarray(v, dtype=object)
        elif k == "gt_match":
            out[k] = np.asarray(v, dtype=bool)
        else:
            out[k] = np.asarray(v, dtype=float)
    # derived
    hl = np.maximum(out["h_lost_raw"], 1e-6)
    hc = np.maximum(out["h_cand_raw"], 1e-6)
    out["log_h_ratio"] = np.abs(np.log(hc / hl))
    out["speed_mismatch"] = np.abs(out["lost_exit_speed"] - out["cand_entry_speed"])
    out["resid_max"] = np.maximum(out["fwd_resid"], out["bwd_resid"])
    return out


def exposure_summary(pool: dict[str, np.ndarray]) -> dict[str, Any]:
    """Independence-unit declaration for exposure counts (framework §8.1).

    Raw exposure unit is one relink pair row. Rows sharing a lost track
    (seq, lost_id) are correlated trials; the unique-cluster count is an
    upper bound on the number of independent trials, not automatically an
    effective sample size — tracks within a sequence still share scene,
    occlusion, and pipeline state. Cluster counts are declared only when
    every GT row carries a lost_id; partial metadata is reported as
    insufficient, never silently collapsed. Binomial-style bounds on
    non-zero hurt additionally require hurt outcomes aggregated to the
    same trial unit before using the cluster count as denominator.
    """
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"]
    n_gt = int(y.sum())
    out: dict[str, Any] = {
        "gt_exposure_unit_raw": "relink_pair_row",
        "n_gt_exposed": n_gt,
        "n_fp_exposed": int((~y).sum()),
        "declared_trial_unit": "lost_track(seq,lost_id)",
        "independence_assumption": (
            "clusters treated as independent trials; not verified"
        ),
        "remaining_clustering": "sequence",
    }
    lost = pool.get("lost_id")
    if lost is None:
        out["n_gt_exposed_clusters"] = None
        out["clustering"] = "unknown: pairs CSV lacks lost_id"
        return out
    lost_gt = np.asarray(lost, dtype=object)[y]
    seq_gt = seq[y]
    n_missing = int(sum(1 for x in lost_gt if not str(x).strip()))
    out["n_gt_rows_missing_lost_id"] = n_missing
    if n_missing > 0:
        out["n_gt_exposed_clusters"] = None
        out["clustering"] = (
            "unknown: pairs CSV lacks lost_id"
            if n_missing == n_gt
            else f"insufficient_metadata: {n_missing}/{n_gt} GT rows missing lost_id"
        )
        return out
    clusters = {(str(s), str(t)) for s, t in zip(seq_gt, lost_gt)}
    per_seq: dict[str, int] = {}
    for s, _ in clusters:
        per_seq[s] = per_seq.get(s, 0) + 1
    out["n_gt_exposed_clusters"] = len(clusters)
    out["per_seq_gt_exposed_clusters"] = dict(sorted(per_seq.items()))
    return out


# mamba_whole_graph_m production bridge knobs (configs/presets/mamba_whole_graph_m.yaml).
# Offline proxies on builder pairs — not bit-exact with live ema_h / foot-ring OLS.
M_PROD_BRIDGE_PX = 0.40
M_PROD_H_LO = 0.60
M_PROD_H_HI = 1.70
# s preset (for contrast; not active on m)
S_PROD_BRIDGE_PX = 0.25
S_PROD_H_LO = 0.75
S_PROD_H_HI = 1.33


def ensure_prod_proxy_scores(pool: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add offline proxies of live bridge score + height ratio.

    Live kernel (tracker_gpu.cu):
      w = sqrt(clip(s_lost / 0.12, 0, 1))
      bdist = w * 0.5*(fwd+bwd) + (1-w)*dist_h
      accept if bdist <= bridge_px and ema_lost/ema_cand in [h_lo, h_hi]

    Builder pairs store mid-point ``bridge_dist`` and residual terms; we rebuild
    the speed-weighted score as ``score_m_bridge`` for production-shaped gates.
    Height ratio uses raw endpoint heights (ema proxy).
    """
    if "score_m_bridge" not in pool:
        s = pool["lost_exit_speed"]
        w = np.sqrt(np.clip(s / 0.12, 0.0, 1.0))
        pool["score_m_bridge"] = (
            w * 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
            + (1.0 - w) * pool["dist_h"]
        )
    if "h_ratio_lost_over_cand" not in pool:
        hl = np.maximum(pool["h_lost_raw"], 1e-6)
        hc = np.maximum(pool["h_cand_raw"], 1e-6)
        pool["h_ratio_lost_over_cand"] = hl / hc
    return pool


def production_shaped_rules() -> list[tuple[str, str, RuleFn, str]]:
    """m (and contrast s) production-shaped reject masks = fail support gate.

    rule_class=baseline: coverage of what production already would cut,
    not a research safe_reject candidate library entry.
    """

    def r_m_px(p: dict[str, np.ndarray]) -> np.ndarray:
        ensure_prod_proxy_scores(p)
        return p["score_m_bridge"] > M_PROD_BRIDGE_PX

    def r_m_h(p: dict[str, np.ndarray]) -> np.ndarray:
        ensure_prod_proxy_scores(p)
        hr = p["h_ratio_lost_over_cand"]
        return (hr < M_PROD_H_LO) | (hr > M_PROD_H_HI)

    def r_m_combo(p: dict[str, np.ndarray]) -> np.ndarray:
        return r_m_px(p) | r_m_h(p)

    def r_m_midpoint_px(p: dict[str, np.ndarray]) -> np.ndarray:
        # mid-point bridge_dist (builder column) vs same px — shape contrast
        return p["bridge_dist"] > M_PROD_BRIDGE_PX

    def r_s_px(p: dict[str, np.ndarray]) -> np.ndarray:
        ensure_prod_proxy_scores(p)
        return p["score_m_bridge"] > S_PROD_BRIDGE_PX

    def r_s_h(p: dict[str, np.ndarray]) -> np.ndarray:
        ensure_prod_proxy_scores(p)
        hr = p["h_ratio_lost_over_cand"]
        return (hr < S_PROD_H_LO) | (hr > S_PROD_H_HI)

    def r_s_combo(p: dict[str, np.ndarray]) -> np.ndarray:
        return r_s_px(p) | r_s_h(p)

    return [
        (
            "prod_m_score_gt_px0.4",
            "baseline",
            r_m_px,
            f"m production-shaped: score_m_bridge > {M_PROD_BRIDGE_PX} "
            f"(offline proxy of relink_bridge_px)",
        ),
        (
            "prod_m_h_ratio_out_0.6_1.7",
            "baseline",
            r_m_h,
            f"m production-shaped: h_lost/h_cand ∉ [{M_PROD_H_LO},{M_PROD_H_HI}] "
            f"(raw-h proxy of ema scale gate)",
        ),
        (
            "prod_m_px_or_h_fail",
            "baseline",
            r_m_combo,
            "m production-shaped composite: px OR h-ratio fail (support gate coverage)",
        ),
        (
            "prod_m_midpoint_bridge_gt_0.4",
            "baseline",
            r_m_midpoint_px,
            "contrast: builder mid-point bridge_dist > 0.4 (not live score)",
        ),
        (
            "prod_s_score_gt_px0.25",
            "baseline",
            r_s_px,
            f"s-shaped contrast: score_m_bridge > {S_PROD_BRIDGE_PX} on m pairs",
        ),
        (
            "prod_s_h_ratio_out_0.75_1.33",
            "baseline",
            r_s_h,
            f"s-shaped contrast: h ratio ∉ [{S_PROD_H_LO},{S_PROD_H_HI}] on m pairs",
        ),
        (
            "prod_s_px_or_h_fail",
            "baseline",
            r_s_combo,
            "s-shaped composite on m pairs (tighter gates → more GT_hurt expected)",
        ),
    ]


def default_probe_rules() -> list[tuple[str, str, RuleFn, str]]:
    """Built-in probe reject rules (research only; not production).

    Returns list of (rule_name, rule_class_hint, fn, notes).
    fn(pool) -> reject_mask
    """

    def r_bridge_gt1(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["bridge_dist"] > 1.0

    def r_bridge_gt2(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["bridge_dist"] > 2.0

    def r_dir_neg(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["dir_cos"] < 0.0

    def r_dir_hard_neg(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["dir_cos"] < -0.5

    def r_long_gap_dir(p: dict[str, np.ndarray]) -> np.ndarray:
        return (p["gap"] > 60) & (p["dir_cos"] < 0.0)

    def r_mid_gap_geo_dir(p: dict[str, np.ndarray]) -> np.ndarray:
        return (p["gap"] > 30) & (p["bridge_dist"] > 2.0) & (p["dir_cos"] < 0.3)

    def r_long_far(p: dict[str, np.ndarray]) -> np.ndarray:
        return (p["gap"] > 60) & (p["bridge_dist"] > 3.0)

    def r_scale_jump(p: dict[str, np.ndarray]) -> np.ndarray:
        # |log h ratio| > log(1.5) ≈ 0.405
        return p["log_h_ratio"] > math.log(1.5)

    def r_speed_mismatch_hi(p: dict[str, np.ndarray]) -> np.ndarray:
        # relative to pool: above p90 of speed_mismatch among all
        thr = float(np.quantile(p["speed_mismatch"], 0.90))
        return p["speed_mismatch"] >= thr

    def r_long_gap_speed_dir(p: dict[str, np.ndarray]) -> np.ndarray:
        thr = float(np.quantile(p["speed_mismatch"], 0.80))
        return (p["gap"] > 60) & (p["dir_cos"] < 0.0) & (p["speed_mismatch"] >= thr)

    def r_long_gap_dir_far(p: dict[str, np.ndarray]) -> np.ndarray:
        return (p["gap"] > 60) & (p["dir_cos"] < 0.0) & (p["bridge_dist"] > 2.0)

    def r_bridge_strictly_above_all_gt(p: dict[str, np.ndarray]) -> np.ndarray:
        thr = float(p["bridge_dist"][p["gt_match"]].max())
        return p["bridge_dist"] > thr

    def r_scale_strictly_above_all_gt(p: dict[str, np.ndarray]) -> np.ndarray:
        thr = float(p["log_h_ratio"][p["gt_match"]].max())
        return p["log_h_ratio"] > thr

    def r_speed_strictly_above_all_gt(p: dict[str, np.ndarray]) -> np.ndarray:
        thr = float(p["speed_mismatch"][p["gt_match"]].max())
        return p["speed_mismatch"] > thr

    return [
        (
            "ceiling_bridge_dist_gt_all_gt",
            "baseline",
            r_bridge_strictly_above_all_gt,
            "oracle 1D ε=0 ceiling (score>max GT); headroom only — NOT a production candidate",
        ),
        (
            "ceiling_log_h_ratio_gt_all_gt",
            "baseline",
            r_scale_strictly_above_all_gt,
            "oracle 1D ε=0 ceiling (|log h|>max GT); headroom only — NOT a production candidate",
        ),
        (
            "ceiling_speed_mismatch_gt_all_gt",
            "baseline",
            r_speed_strictly_above_all_gt,
            "oracle 1D ε=0 ceiling (speed mismatch>max GT); headroom only — NOT a production candidate",
        ),
        (
            "baseline_bridge_dist_gt_1",
            "baseline",
            r_bridge_gt1,
            "1D thr baseline; expected to hurt long-gap GT",
        ),
        (
            "baseline_bridge_dist_gt_2",
            "baseline",
            r_bridge_gt2,
            "looser 1D thr baseline",
        ),
        (
            "dir_cos_lt_0",
            "risky_reject",
            r_dir_neg,
            "direction anti-aligned",
        ),
        (
            "dir_cos_lt_m0.5",
            "risky_reject",
            r_dir_hard_neg,
            "strong opposite direction",
        ),
        (
            "gap_gt60_and_dir_cos_lt_0",
            "safe_reject",
            r_long_gap_dir,
            "long gap + wrong direction (probe)",
        ),
        (
            "gap_gt30_bridge_gt2_dir_lt_0.3",
            "safe_reject",
            r_mid_gap_geo_dir,
            "mid+ gap + far + weak dir (probe)",
        ),
        (
            "gap_gt60_bridge_gt3",
            "risky_reject",
            r_long_far,
            "long gap + far geometry only",
        ),
        (
            "log_h_ratio_gt_log1.5",
            "risky_reject",
            r_scale_jump,
            "box scale change > 1.5x",
        ),
        (
            "speed_mismatch_ge_p90",
            "risky_reject",
            r_speed_mismatch_hi,
            "exit/entry speed |Δ| at pool p90",
        ),
        (
            "gap_gt60_dir_lt0_speed_ge_p80",
            "safe_reject",
            r_long_gap_speed_dir,
            "long gap + bad dir + speed mismatch",
        ),
        (
            "gap_gt60_dir_lt0_bridge_gt2",
            "safe_reject",
            r_long_gap_dir_far,
            "long gap + bad dir + far",
        ),
    ]


def all_audit_rules() -> list[tuple[str, str, RuleFn, str]]:
    """Production-shaped coverage first, then research probes."""
    return production_shaped_rules() + default_probe_rules()


def seq_bin_masks(seq: np.ndarray) -> list[tuple[str, np.ndarray]]:
    names = sorted({str(s) for s in seq.tolist() if str(s)})
    return [(f"seq_{n}", seq == n) for n in names]


def evaluate_rules(
    pool: dict[str, np.ndarray],
    rules: list[tuple[str, str, RuleFn, str]],
    *,
    coverage_bin: str = "all",
    bin_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    y = pool["gt_match"]
    if bin_mask is not None:
        y = y[bin_mask]
    rows: list[dict[str, Any]] = []
    for name, hint, fn, notes in rules:
        rej = fn(pool)
        if bin_mask is not None:
            rej = rej[bin_mask]
        m = constrained_fp_prune_metrics(
            y,
            rej,
            rule_name=name,
            coverage_bin=coverage_bin,
            rule_class=hint,
            notes=notes,
        )
        # Preserve baseline / calibration (incl. oracle ceiling_* probes).
        # Only reclassify optimistic safe_reject / empty hints by measured ε.
        if hint in ("baseline", "calibration"):
            m["rule_class"] = hint
        elif m["safe_level"] == "eps0" and m["FP_removed"] > 0:
            m["rule_class"] = "safe_reject"
        elif m["safe_level"] == "unsafe":
            m["rule_class"] = "risky_reject" if m["FP_removed"] > 0 else m["rule_class"]
        elif m["safe_level"] in ("eps0_1pct", "eps1pct"):
            m["rule_class"] = "risky_reject"
        rows.append(m)
    return rows


def gap_bin_masks(gap: np.ndarray) -> list[tuple[str, np.ndarray]]:
    bins = [
        ("gap_1-10", (gap >= 1) & (gap <= 10)),
        ("gap_11-30", (gap >= 11) & (gap <= 30)),
        ("gap_31-60", (gap >= 31) & (gap <= 60)),
        ("gap_61-150", (gap >= 61) & (gap <= 150)),
        ("gap_151-300", (gap >= 151) & (gap <= 300)),
    ]
    return [(n, m) for n, m in bins if bool(m.any())]


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(SAFE_REJECT_AUDIT_COLS), extrasaction="ignore"
        )
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in SAFE_REJECT_AUDIT_COLS}
            # stringify ratio
            v = out["FP_removed_per_GT_hurt"]
            if v == "safe":
                out["FP_removed_per_GT_hurt"] = "safe"
            elif isinstance(v, float):
                out["FP_removed_per_GT_hurt"] = f"{v:.6g}"
            w.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pairs", type=Path, required=True, help="builder pairs CSV")
    ap.add_argument(
        "--study-dir",
        type=Path,
        default=None,
        help="if set with --write-study, write audit artifacts here",
    )
    ap.add_argument(
        "--write-study",
        action="store_true",
        help=f"write {SAFE_REJECT_AUDIT_FILENAME} + {SAFE_REJECT_SUMMARY_FILENAME}",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="optional explicit audit CSV path",
    )
    ap.add_argument(
        "--by-gap",
        action="store_true",
        help="also evaluate each rule inside gap bins",
    )
    ap.add_argument(
        "--by-seq",
        action="store_true",
        help="also evaluate each rule per MOT sequence (7-seq coverage)",
    )
    ap.add_argument(
        "--no-prod-shaped",
        action="store_true",
        help="skip production-shaped m/s gates (probes only)",
    )
    args = ap.parse_args()

    pool = load_gt_valid_pool(args.pairs)
    ensure_prod_proxy_scores(pool)
    rules = default_probe_rules() if args.no_prod_shaped else all_audit_rules()
    rows = evaluate_rules(pool, rules, coverage_bin="all")

    if args.by_gap:
        for bname, mask in gap_bin_masks(pool["gap"]):
            rows.extend(evaluate_rules(pool, rules, coverage_bin=bname, bin_mask=mask))

    if args.by_seq:
        for bname, mask in seq_bin_masks(pool["seq"]):
            rows.extend(evaluate_rules(pool, rules, coverage_bin=bname, bin_mask=mask))

    # 1D frontiers on key scores (reject when score high = worse)
    frontiers = {
        "bridge_dist": frontier_fp_removed_at_eps(
            pool["gt_match"],
            pool["bridge_dist"],
            higher_means_more_reject=True,
        ),
        "score_m_bridge": frontier_fp_removed_at_eps(
            pool["gt_match"],
            pool["score_m_bridge"],
            higher_means_more_reject=True,
        ),
        "speed_mismatch": frontier_fp_removed_at_eps(
            pool["gt_match"],
            pool["speed_mismatch"],
            higher_means_more_reject=True,
        ),
        "log_h_ratio": frontier_fp_removed_at_eps(
            pool["gt_match"],
            pool["log_h_ratio"],
            higher_means_more_reject=True,
        ),
        # dir_cos: lower is worse → reject when score low
        "neg_dir_cos": frontier_fp_removed_at_eps(
            pool["gt_match"],
            -pool["dir_cos"],
            higher_means_more_reject=True,
        ),
    }

    def _print_rows(title: str, sel: list[dict[str, Any]], bin_w: int = 12) -> None:
        print(f"\n=== {title} ===")
        print(
            f"{'rule':<40} {'bin':<{bin_w}} {'class':<14} "
            f"{'GTh':>5} {'hurt%':>7} {'FPrm':>6} {'FPrm%':>7} {'ratio':>8} {'lvl':<10}"
        )
        for r in sel:
            ratio = r["FP_removed_per_GT_hurt"]
            ratio_s = ratio if ratio == "safe" else f"{float(ratio):.1f}"
            print(
                f"{r['rule_name']:<40} {r['coverage_bin']:<{bin_w}} {r['rule_class']:<14} "
                f"{r['GT_hurt']:5d} {r['GT_hurt_rate'] * 100:6.2f}% "
                f"{r['FP_removed']:6d} {r['FP_removed_rate'] * 100:6.2f}% "
                f"{ratio_s:>8} {r['safe_level']:<10}"
            )

    # production-shaped coverage (all-pool)
    prod_all = [
        r
        for r in rows
        if r["coverage_bin"] == "all" and r["rule_name"].startswith("prod_")
    ]
    _print_rows("Production-shaped gate coverage (all pool)", prod_all)

    # full all-pool table
    _print_rows(
        "All rules (all pool)",
        [r for r in rows if r["coverage_bin"] == "all"],
    )

    if args.by_seq:
        # highlight composite m gate per seq
        m_combo = [
            r
            for r in rows
            if r["rule_name"] == "prod_m_px_or_h_fail"
            and str(r["coverage_bin"]).startswith("seq_")
        ]
        _print_rows(
            "prod_m_px_or_h_fail by sequence",
            m_combo,
            bin_w=22,
        )

    print("\n=== 1D frontier: max FP_removed @ ε ===")
    for feat, fr in frontiers.items():
        print(f"  [{feat}]")
        for row in fr:
            thr = row.get("thr")
            thr_s = f"{thr:.4g}" if thr is not None else "n/a"
            print(
                f"    ε={row['epsilon']:<6} feasible={row.get('feasible')} "
                f"thr={thr_s:<10} FP_removed={row['FP_removed']:<6} "
                f"rate={row['FP_removed_rate'] * 100:5.2f}% "
                f"GT_hurt={row['GT_hurt']}"
            )

    # safe winners
    # Production-candidate safe rules only (exclude baseline / oracle ceilings).
    safe = [
        r
        for r in rows
        if r["coverage_bin"] == "all"
        and r["rule_class"] == "safe_reject"
        and r["safe_level"] == "eps0"
        and r["FP_removed"] > 0
    ]
    safe.sort(key=lambda r: -r["FP_removed"])
    print("\n=== Safe reject (ε=0, FP_removed>0; excludes prod_/ceiling baseline) ===")
    if not safe:
        print("  (none in probe set)")
    for r in safe:
        print(
            f"  {r['rule_name']}: FP_removed={r['FP_removed']} "
            f"({r['FP_removed_rate'] * 100:.2f}% of FP)"
        )

    seq_summary = {}
    if args.by_seq:
        for r in rows:
            if r["rule_name"] != "prod_m_px_or_h_fail":
                continue
            if not str(r["coverage_bin"]).startswith("seq_"):
                continue
            seq_summary[r["coverage_bin"]] = {
                "GT_total": r["GT_total"],
                "GT_hurt": r["GT_hurt"],
                "GT_hurt_rate": r["GT_hurt_rate"],
                "FP_removed": r["FP_removed"],
                "FP_removed_rate": r["FP_removed_rate"],
                "safe_level": r["safe_level"],
            }

    summary = {
        "pairs_csv": str(args.pairs.resolve()),
        "n_gt_valid": int(pool["gt_match"].size),
        "n_pos": int(pool["gt_match"].sum()),
        "n_neg": int((~pool["gt_match"]).sum()),
        "exposure": exposure_summary(pool),
        "goal": "maximize FP_removed s.t. GT_hurt_rate <= ε",
        "cost_asymmetry": {
            "GT_hurt": "hard — early reject of true pairs is usually irreversible",
            "FP_removed": (
                "soft upper bound — later assignment/scoring/NMS/e2e may still "
                "suppress surviving FPs; not an e2e FP delta guarantee"
            ),
            "preference": "ε=0 safe_reject over high FP_removed with GT_hurt>0",
        },
        "production_shaped": {
            "preset_ref": "mamba_whole_graph_m",
            "bridge_px": M_PROD_BRIDGE_PX,
            "h_lo": M_PROD_H_LO,
            "h_hi": M_PROD_H_HI,
            "score_proxy": (
                "score_m_bridge = w*0.5*(fwd+bwd)+(1-w)*dist_h; "
                "w=sqrt(clip(lost_exit_speed/0.12,0,1)); h_ratio=raw endpoint"
            ),
            "not_bit_exact": (
                "live uses ema_h + foot-ring OLS anchors; this is offline proxy "
                "on builder pairs for L0 coverage, not live fire counts"
            ),
        },
        "epsilons": {"eps0": 0.0, "eps0_1pct": 0.001, "eps1pct": 0.01},
        "audit_rows": rows,
        "frontiers_1d": frontiers,
        "safe_reject_all_pool": safe,
        "prod_m_combo_by_seq": seq_summary,
    }

    out_csv = args.out_csv
    if args.write_study:
        if args.study_dir is None:
            raise SystemExit("--write-study requires --study-dir")
        study = args.study_dir
        study.mkdir(parents=True, exist_ok=True)
        out_csv = study / SAFE_REJECT_AUDIT_FILENAME
        (study / SAFE_REJECT_SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
        )
        print(f"Wrote {study / SAFE_REJECT_SUMMARY_FILENAME}")

    if out_csv is not None:
        write_audit_csv(out_csv, rows)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
