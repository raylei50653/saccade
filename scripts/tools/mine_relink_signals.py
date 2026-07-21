#!/usr/bin/env python3
"""Batch deep-mine continuous relink signals on a B1 pairs CSV.

Automates the signal_analysis_ledger §2 checklist for every registered
offline signal on U_relink_pair. Numbers land in out/signal_study/<stamp>/;
human verdict one-liners are printed and written to summary.json.

  uv run python scripts/tools/mine_relink_signals.py \\
    --pairs out/signal_study/m_b1_smoke_*/pairs.csv \\
    --study-dir out/signal_study/m_b1_signal_mine_<stamp> \\
    --all

  # or one signal:
  uv run python scripts/tools/mine_relink_signals.py --pairs ... --signal m.score_m_bridge.px

Not covered here (need other universes / runs):
  - frame.iou_maha_cost  (U_cand dump)
  - live bridge fire counts
  - B2 reconnect (use reconnect_rate.py)
  - production GO / ledger promotion
"""
# status: stable

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from saccade.perception.eval.signal_tables import frontier_fp_removed_at_eps

# ── load audit helpers (prod proxy scores) ───────────────────────────────────
_AUDIT_PATH = Path(__file__).resolve().parent / "audit_relink_safe_reject.py"
_spec = importlib.util.spec_from_file_location("audit_relink_safe_reject", _AUDIT_PATH)
assert _spec and _spec.loader
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)

GAP_BINS = [
    ("1-10", 1, 10),
    ("11-30", 11, 30),
    ("31-60", 31, 60),
    ("61-150", 61, 150),
    ("151-300", 151, 300),
]


@dataclass(frozen=True)
class SignalSpec:
    signal_id: str
    """Column or derived score name after ensure."""
    score_key: str
    """If True, lower score is better for true pairs (geometry distances)."""
    lower_is_better: bool
    """Optional production-style reject: score fails gate when True."""
    reject_fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None
    reject_label: str
    thr_grid: list[float]
    notes: str


def _ensure(pool: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _audit.ensure_prod_proxy_scores(pool)
    # mean residual
    if "resid_mean" not in pool:
        pool["resid_mean"] = 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
    if "neg_dir_cos" not in pool:
        pool["neg_dir_cos"] = -pool["dir_cos"]
    return pool


def catalog() -> list[SignalSpec]:
    """Offline B1 signals we can fully auto-mine from pairs CSV."""

    def rej_m_px(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["score_m_bridge"] > _audit.M_PROD_BRIDGE_PX

    def rej_m_h(p: dict[str, np.ndarray]) -> np.ndarray:
        hr = p["h_ratio_lost_over_cand"]
        return (hr < _audit.M_PROD_H_LO) | (hr > _audit.M_PROD_H_HI)

    def rej_mid_1(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["bridge_dist"] > 1.0

    def rej_dir_neg(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["dir_cos"] < 0.0

    def rej_dir_hard(p: dict[str, np.ndarray]) -> np.ndarray:
        return p["dir_cos"] < -0.5

    return [
        SignalSpec(
            "m.score_m_bridge.px",
            "score_m_bridge",
            True,
            rej_m_px,
            f"score_m_bridge > {_audit.M_PROD_BRIDGE_PX} (m relink_bridge_px)",
            [0.15, 0.25, 0.40, 0.60, 1.0, 2.0, 5.0],
            "live-shaped speed-weighted bridge score",
        ),
        SignalSpec(
            "m.h_ratio.scale",
            "log_h_ratio",
            True,
            rej_m_h,
            f"h_ratio not in [{_audit.M_PROD_H_LO},{_audit.M_PROD_H_HI}]",
            [0.1, 0.2, 0.405, 0.5, 0.693, 0.91],
            "|log h_ratio|; production band is h-space not log-space thr",
        ),
        SignalSpec(
            "m.bridge_dist.midpoint",
            "bridge_dist",
            True,
            rej_mid_1,
            "bridge_dist > 1 (hard-pool edge / offline hub style)",
            [0.15, 0.30, 0.50, 1.0, 2.0, 5.0],
            "builder mid-point bridge_dist",
        ),
        SignalSpec(
            "m.dir_cos",
            "dir_cos",
            False,  # higher dir_cos better for pos
            rej_dir_neg,
            "dir_cos < 0 (anti-aligned)",
            [-0.5, 0.0, 0.3, 0.5, 0.8],
            "lost exit vel vs displacement cosine",
        ),
        SignalSpec(
            "m.speed_mismatch",
            "speed_mismatch",
            True,
            None,
            "",
            [0.02, 0.05, 0.10, 0.15, 0.20],
            "|exit-entry| speed in h/frame",
        ),
        SignalSpec(
            "m.fwd_bwd_resid",
            "resid_mean",
            True,
            None,
            "",
            [0.5, 1.0, 2.0, 3.0, 5.0],
            "0.5*(fwd_resid+bwd_resid)",
        ),
        SignalSpec(
            "m.dist_h",
            "dist_h",
            True,
            None,
            "",
            [0.5, 1.0, 2.0, 3.0, 5.0],
            "straight foot distance / h_ref",
        ),
        SignalSpec(
            "m.gap",
            "gap",
            True,  # shorter gap slightly more pos? weak prior
            None,
            "",
            [10, 30, 60, 150, 300],
            "time gap frames — context, not identity",
        ),
    ]


def dist_stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def gate_metrics(y: np.ndarray, rej: np.ndarray) -> dict[str, Any]:
    gt = int(y.sum())
    fp = int((~y).sum())
    hurt = int((y & rej).sum())
    fprm = int((~y & rej).sum())
    return {
        "n": int(y.size),
        "n_pos": gt,
        "n_neg": fp,
        "GT_hurt": hurt,
        "GT_hurt_rate": hurt / gt if gt else 0.0,
        "FP_removed": fprm,
        "FP_removed_rate": fprm / fp if fp else 0.0,
        "kept_pos": gt - hurt,
        "surviving_fp": fp - fprm,
        "surviving_fp_frac": (fp - fprm) / fp if fp else 0.0,
        "recall_pos_kept": (gt - hurt) / gt if gt else 0.0,
    }


def safe_auc(y: np.ndarray, score_for_rank: np.ndarray) -> float | None:
    """score_for_rank: higher = more pos-like."""
    if y.sum() < 5 or (~y).sum() < 5:
        return None
    # constant score → undefined
    if float(np.nanstd(score_for_rank)) < 1e-12:
        return None
    try:
        return float(roc_auc_score(y, score_for_rank))
    except ValueError:
        return None


def thr_table(
    y: np.ndarray,
    score: np.ndarray,
    thresholds: list[float],
    *,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    rows = []
    for thr in thresholds:
        if lower_is_better:
            # accept when score <= thr  → reject when score > thr
            rej = score > thr
            accept = score <= thr
        else:
            # accept when score >= thr
            rej = score < thr
            accept = score >= thr
        m = gate_metrics(y, rej)
        tp = int((y & accept).sum())
        fp = int((~y & accept).sum())
        fn = int((y & ~accept).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            {
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                **{
                    f"gate_{k}": v
                    for k, v in m.items()
                    if k.startswith("GT") or k.startswith("FP")
                },
            }
        )
    return rows


def mine_one(pool: dict[str, np.ndarray], spec: SignalSpec) -> dict[str, Any]:
    y = pool["gt_match"].astype(bool)
    score = pool[spec.score_key].astype(float)
    gap = pool["gap"]
    seq = pool["seq"]
    hard = pool["bridge_dist"] <= 1.0

    rank = -score if spec.lower_is_better else score
    auc_full = safe_auc(y, rank)
    auc_hard = safe_auc(y[hard], rank[hard]) if hard.any() else None

    report: dict[str, Any] = {
        "signal_id": spec.signal_id,
        "score_key": spec.score_key,
        "lower_is_better": spec.lower_is_better,
        "notes": spec.notes,
        "pool": {
            "full": {
                "n": int(y.size),
                "n_pos": int(y.sum()),
                "n_neg": int((~y).sum()),
                "base_rate": float(y.mean()),
            },
            "hard_bridge_dist_le_1": {
                "n": int(hard.sum()),
                "n_pos": int(y[hard].sum()),
                "n_neg": int((~y & hard).sum()),
            },
        },
        "signal_dist": {
            "pos": dist_stats(score[y]),
            "neg": dist_stats(score[~y]),
        },
        "auc": {
            "full": auc_full,
            "hard_bridge_dist_le_1": auc_hard,
            "rank_direction": "higher_more_pos_like",
            "score_transform": f"{'-' if spec.lower_is_better else ''}{spec.score_key}",
        },
    }

    # production / probe reject coverage
    if spec.reject_fn is not None:
        rej = spec.reject_fn(pool)
        report["reject_label"] = spec.reject_label
        report["gate_coverage"] = {
            "full": gate_metrics(y, rej),
            "hard_bridge_dist_le_1": gate_metrics(y[hard], rej[hard]),
        }
        hurt = y & rej
        kept = y & ~rej
        surv_fp = (~y) & ~rej
        report["hurt_gt_profile"] = {
            "n": int(hurt.sum()),
            "gap": dist_stats(gap[hurt]),
            "score": dist_stats(score[hurt]),
            "by_seq": {
                str(s): int((hurt & (seq == s)).sum())
                for s in sorted(set(seq.tolist()))
            },
        }
        report["kept_gt_profile"] = {
            "n": int(kept.sum()),
            "gap": dist_stats(gap[kept]),
            "score": dist_stats(score[kept]),
        }
        report["surviving_fp_profile"] = {
            "n": int(surv_fp.sum()),
            "frac_of_fp": float(surv_fp.sum() / max((~y).sum(), 1)),
            "gap": dist_stats(gap[surv_fp]),
            "score": dist_stats(score[surv_fp]),
        }

    # thr grid
    if spec.thr_grid:
        report["thr_table_full"] = thr_table(
            y, score, spec.thr_grid, lower_is_better=spec.lower_is_better
        )
        report["thr_table_hard"] = thr_table(
            y[hard],
            score[hard],
            spec.thr_grid,
            lower_is_better=spec.lower_is_better,
        )

    # 1D ε frontier (always on "higher = more reject-like")
    reject_score = score if spec.lower_is_better else -score
    report["frontier_1d"] = frontier_fp_removed_at_eps(
        y, reject_score, higher_means_more_reject=True
    )

    # by gap
    report["by_gap"] = {}
    for name, lo, hi in GAP_BINS:
        m = (gap >= lo) & (gap <= hi)
        if not m.any():
            continue
        entry: dict[str, Any] = {
            "auc": safe_auc(y[m], rank[m]),
            "pos_med": float(np.median(score[y & m])) if (y & m).any() else None,
            "neg_med": float(np.median(score[~y & m])) if (~y & m).any() else None,
            "n_pos": int(y[m].sum()),
            "n_neg": int((~y[m]).sum()),
        }
        if spec.reject_fn is not None:
            entry["gate"] = gate_metrics(y[m], spec.reject_fn(pool)[m])
        report["by_gap"][name] = entry

    # by seq
    report["by_seq"] = {}
    for s in sorted(set(seq.tolist())):
        m = seq == s
        entry = {
            "auc": safe_auc(y[m], rank[m]),
            "pos_med": float(np.median(score[y & m])) if (y & m).any() else None,
            "neg_med": float(np.median(score[~y & m])) if (~y & m).any() else None,
            "n_pos": int(y[m].sum()),
            "n_neg": int((~y[m]).sum()),
        }
        if spec.reject_fn is not None:
            entry["gate"] = gate_metrics(y[m], spec.reject_fn(pool)[m])
        report["by_seq"][str(s)] = entry

    report["auto_verdict"] = auto_verdict(report, spec)
    return report


def auto_verdict(report: dict[str, Any], spec: SignalSpec) -> str:
    """Deterministic one-liner for ledger; human may refine."""
    auc_f = report["auc"]["full"]
    auc_h = report["auc"]["hard_bridge_dist_le_1"]
    pos_m = report["signal_dist"]["pos"].get("median")
    neg_m = report["signal_dist"]["neg"].get("median")

    def auc_bucket(a: float | None) -> str:
        if a is None:
            return "n/a"
        if a >= 0.85:
            return "strong"
        if a >= 0.70:
            return "mid"
        if a >= 0.60:
            return "weak"
        return "near-random"

    parts = [
        f"AUC full={auc_f:.3f}({auc_bucket(auc_f)})"
        if auc_f is not None
        else "AUC full=n/a",
        f"hard={auc_h:.3f}({auc_bucket(auc_h)})" if auc_h is not None else "hard=n/a",
        f"pos_med={pos_m:.3g} neg_med={neg_m:.3g}"
        if pos_m is not None and neg_m is not None
        else "",
    ]
    if "gate_coverage" in report:
        g = report["gate_coverage"]["full"]
        parts.append(
            f"prod_gate GT_hurt={100 * g['GT_hurt_rate']:.1f}% "
            f"FP_rm={100 * g['FP_removed_rate']:.1f}% "
            f"surv_FP={100 * g['surviving_fp_frac']:.1f}%"
        )
    fr0 = next(
        (x for x in report.get("frontier_1d", []) if x.get("epsilon") == 0.0),
        None,
    )
    if fr0 and fr0.get("feasible"):
        parts.append(
            f"ε0_FPrm={100 * fr0['FP_removed_rate']:.1f}% (GT_hurt=0 headroom)"
        )
    # identity-solves? hard AUC + surviving
    if auc_h is not None and auc_h < 0.80:
        parts.append("hard not closed")
    elif auc_h is not None:
        parts.append("hard usable but check base-rate")
    return "; ".join(p for p in parts if p)


def write_ledger_stub(path: Path, rows: list[dict[str, Any]]) -> None:
    """Machine-readable ledger rows for merge into markdown by humans/tools."""
    path.write_text(json.dumps(rows, indent=2, default=float) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument(
        "--study-dir",
        type=Path,
        default=None,
        help="output study root (default out/signal_study/m_b1_signal_mine_<stamp>)",
    )
    ap.add_argument(
        "--signal",
        action="append",
        default=None,
        help="signal_id (repeatable); default --all registered offline signals",
    )
    ap.add_argument("--all", action="store_true", help="mine full offline catalog")
    ap.add_argument(
        "--list",
        action="store_true",
        help="list signal_ids and exit",
    )
    args = ap.parse_args()

    specs = catalog()
    if args.list:
        for s in specs:
            print(f"{s.signal_id:<28} score={s.score_key:<18} {s.notes}")
        return

    want = None
    if args.signal:
        want = set(args.signal)
    elif not args.all:
        raise SystemExit("pass --all or --signal <id> (use --list)")

    if want:
        specs = [s for s in specs if s.signal_id in want]
        missing = want - {s.signal_id for s in specs}
        if missing:
            raise SystemExit(f"unknown signal_id: {sorted(missing)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = args.study_dir or Path(f"out/signal_study/m_b1_signal_mine_{stamp}")
    study.mkdir(parents=True, exist_ok=True)
    sig_dir = study / "signals"
    sig_dir.mkdir(exist_ok=True)

    pool = _ensure(_audit.load_gt_valid_pool(args.pairs))
    ledger_rows: list[dict[str, Any]] = []
    batch: dict[str, Any] = {
        "study_id": study.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_csv": str(args.pairs.resolve()),
        "n_signals": len(specs),
        "signals": {},
    }

    print(f"STUDY={study}")
    print(
        f"pairs={args.pairs}  n_gt_valid={pool['gt_match'].size}  pos={int(pool['gt_match'].sum())}"
    )
    print()

    for spec in specs:
        rep = mine_one(pool, spec)
        out_p = sig_dir / f"{spec.signal_id.replace('.', '_')}.json"
        out_p.write_text(
            json.dumps(rep, indent=2, default=float) + "\n", encoding="utf-8"
        )
        batch["signals"][spec.signal_id] = {
            "file": str(out_p.relative_to(study)),
            "auc_full": rep["auc"]["full"],
            "auc_hard": rep["auc"]["hard_bridge_dist_le_1"],
            "auto_verdict": rep["auto_verdict"],
            "gate_full": rep.get("gate_coverage", {}).get("full"),
        }
        ledger_rows.append(
            {
                "signal_id": spec.signal_id,
                "status": "depth-done",
                "study": study.name,
                "file": str(out_p),
                "auto_verdict": rep["auto_verdict"],
                "auc_full": rep["auc"]["full"],
                "auc_hard": rep["auc"]["hard_bridge_dist_le_1"],
            }
        )
        af = rep["auc"]["full"]
        ah = rep["auc"]["hard_bridge_dist_le_1"]
        print(f"=== {spec.signal_id} ===")
        print(f"  AUC full={af}  hard={ah}")
        if "gate_coverage" in rep:
            g = rep["gate_coverage"]["full"]
            print(
                f"  gate: hurt={g['GT_hurt']}/{g['n_pos']} "
                f"({100 * g['GT_hurt_rate']:.1f}%)  "
                f"FPrm={g['FP_removed']} ({100 * g['FP_removed_rate']:.1f}%)"
            )
        print(f"  verdict: {rep['auto_verdict']}")
        print(f"  wrote {out_p}")
        print()

    (study / "summary.json").write_text(
        json.dumps(batch, indent=2, default=float) + "\n", encoding="utf-8"
    )
    write_ledger_stub(study / "ledger_rows.json", ledger_rows)

    # ranking table
    ranked = sorted(
        ledger_rows,
        key=lambda r: (r["auc_hard"] is not None, r["auc_hard"] or 0.0),
        reverse=True,
    )
    print("=== rank by hard AUC ===")
    for r in ranked:
        print(f"  {r['signal_id']:<28} full={r['auc_full']}  hard={r['auc_hard']}")
    print(f"\nWrote {study / 'summary.json'}")
    print(f"Wrote {study / 'ledger_rows.json'}")


if __name__ == "__main__":
    main()
