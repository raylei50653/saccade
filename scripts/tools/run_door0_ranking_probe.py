"""Door 0 — ambiguous-band ranking-power probe runner.

Implements the sealed declaration verbatim:
  docs/modules/semantic/research/
  ambiguous_band_ranking_power_probe_declaration_20260712.md
  (sealed via PR #135 merge = research-owner seal)

Read-only diagnostic probe (capability map; framework §20). Refuses to run
on substrate SHA mismatch (V1). Single authorized execution per §11.

Usage:
  .venv/bin/python scripts/tools/run_door0_ranking_probe.py \
      --study-dir out/signal_study/door0_ranking_probe_20260712
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/tools"))
import audit_relink_safe_reject as ar  # noqa: E402

PAIRS = REPO / "out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv"
PAIRS_SHA = "0ae3896791ec074fbe951198752c17385c4ee0770a7ec3831225d3ea56a69d17"

# Declared source columns for the NaN policy (declaration §2).
SOURCE_COLS = (
    "dist_h",
    "fwd_resid",
    "bwd_resid",
    "dir_cos",
    "gap",
    "lost_exit_speed",
    "cand_entry_speed",
    "h_lost_raw",
    "h_cand_raw",
)

# Atom family (declaration §6): (atom, tail) — "hi" fires at >= q85,
# "lo" fires at <= q15.
ATOMS: tuple[tuple[str, str], ...] = (
    ("dist_h", "hi"),
    ("log_h_ratio", "hi"),
    ("resid_mean", "hi"),
    ("speed_mismatch", "hi"),
    ("gap", "hi"),
    ("dir_cos", "lo"),
)

# Second-order ANDs (declaration §6, exact enumeration).
AND_PAIRS: tuple[tuple[str, str], ...] = (
    ("dist_h", "gap"),
    ("dist_h", "speed_mismatch"),
    ("gap", "speed_mismatch"),
    ("gap", "dir_cos"),
    ("dist_h", "log_h_ratio"),
    ("resid_mean", "gap"),
)

CAVEAT = (
    "study scope is the gate-retained band; this terminal establishes no "
    "claim inside the production-reachable set (s0 <= 0.4; 34 events, "
    "descriptive only); any step-4 decision must treat the decision surface "
    "(threshold/margin interplay), not assume in-place reranking behavior."
)

SEQS_ORDER: list[str] = []  # filled at load time (sorted unique)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool() -> dict[str, np.ndarray]:
    """Load per declaration §2: loader + prod proxy + resid_mean + cand_id."""
    pool = ar.load_gt_valid_pool(PAIRS)
    ar.ensure_prod_proxy_scores(pool)
    pool["resid_mean"] = 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
    # Parallel pass for cand_id (event key); loader omits it. Alignment is
    # asserted against seq / lost_id / gap of the loader output.
    cand_id: list[str] = []
    seq_chk: list[str] = []
    lost_chk: list[str] = []
    gap_chk: list[float] = []
    with PAIRS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("gt_valid", "0") not in ("1", "1.0", "True", "true"):
                continue
            cand_id.append(str(row.get("cand_id", "")))
            seq_chk.append(row.get("seq", ""))
            lost_chk.append(str(row.get("lost_id", "")))
            gap_chk.append(float(row["gap"]))
    n = pool["gt_match"].size
    if len(cand_id) != n:
        raise RuntimeError(f"cand_id pass rows {len(cand_id)} != pool rows {n}")
    if not (
        all(a == b for a, b in zip(seq_chk, pool["seq"]))
        and all(a == b for a, b in zip(lost_chk, pool["lost_id"]))
        and np.allclose(np.asarray(gap_chk), pool["gap"])
    ):
        raise RuntimeError("cand_id pass misaligned with loader output")
    pool["cand_id"] = np.asarray(cand_id, dtype=object)
    return pool


def fit_thresholds(
    pool: dict[str, np.ndarray], fit_mask: np.ndarray
) -> dict[str, float]:
    """q85 (hi) / q15 (lo) on band rows of the fitting set (§6, frozen)."""
    thr: dict[str, float] = {}
    for atom, tail in ATOMS:
        v = np.asarray(pool[atom], dtype=np.float64)[fit_mask]
        q = 0.85 if tail == "hi" else 0.15
        thr[atom] = float(np.quantile(v, q, method="linear"))
    return thr


def condition_masks(
    pool: dict[str, np.ndarray], thr: dict[str, float]
) -> dict[str, np.ndarray]:
    """All 12 candidate fire masks over the full pool (declaration §6)."""
    single: dict[str, np.ndarray] = {}
    for atom, tail in ATOMS:
        v = np.asarray(pool[atom], dtype=np.float64)
        single[atom] = v >= thr[atom] if tail == "hi" else v <= thr[atom]
    out: dict[str, np.ndarray] = {}
    for atom, _ in ATOMS:
        out[f"c_{atom}"] = single[atom]
    for a, b in AND_PAIRS:
        out[f"c_{a}_AND_{b}"] = single[a] & single[b]
    return out


def event_index(
    pool: dict[str, np.ndarray], mask: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    """Rankable events (>=1 GT, >=1 FP) -> row indices, on masked rows."""
    y = pool["gt_match"]
    groups: dict[tuple[str, str], list[int]] = {}
    for i in np.nonzero(mask)[0]:
        groups.setdefault((str(pool["seq"][i]), str(pool["cand_id"][i])), []).append(
            int(i)
        )
    events: dict[tuple[str, str], np.ndarray] = {}
    for key, idx in groups.items():
        arr = np.asarray(idx, dtype=np.int64)
        n_gt = int(y[arr].sum())
        if n_gt >= 1 and n_gt < arr.size:
            events[key] = arr
    return events


def event_metrics(
    y: np.ndarray, s0: np.ndarray, c: np.ndarray | None, idx: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Per-event (PWA, MRR, top1, n_good_frac, n_bad_frac) under key (c, s0).

    Baseline = c is None (equivalent to all-zero c). n_good / n_bad are the
    B6b pair counts vs the baseline key, returned as raw counts.
    Tie policy per declaration §5 (PWA tie = 0.5; ranks pessimistic).
    """
    gt = idx[y[idx]]
    fp = idx[~y[idx]]
    s_gt = s0[gt]
    s_fp = s0[fp]
    if c is None:
        c_gt = np.zeros(gt.size, dtype=np.int64)
        c_fp = np.zeros(fp.size, dtype=np.int64)
    else:
        c_gt = c[gt].astype(np.int64)
        c_fp = c[fp].astype(np.int64)
    # pairwise: GT strictly better / tied under lexicographic (c, s0)
    better = (c_gt[:, None] < c_fp[None, :]) | (
        (c_gt[:, None] == c_fp[None, :]) & (s_gt[:, None] < s_fp[None, :])
    )
    tied = (c_gt[:, None] == c_fp[None, :]) & (s_gt[:, None] == s_fp[None, :])
    contrib = better.astype(np.float64) + 0.5 * tied.astype(np.float64)
    pwa = float(contrib.mean())
    # baseline contributions for B6b flip decomposition
    b_better = s_gt[:, None] < s_fp[None, :]
    b_tied = s_gt[:, None] == s_fp[None, :]
    b_contrib = b_better.astype(np.float64) + 0.5 * b_tied.astype(np.float64)
    n_good = float((contrib > b_contrib).sum())
    n_bad = float((contrib < b_contrib).sum())
    # pessimistic ranks: rank = 1 + #{FP strictly better} + #{FP tied}
    fp_better = (~better) & (~tied)  # FP strictly better than GT
    ranks = 1 + fp_better.sum(axis=1) + tied.sum(axis=1)
    best = int(ranks.min())
    return pwa, 1.0 / best, 1.0 if best == 1 else 0.0, n_good, n_bad


def evaluate(
    pool: dict[str, np.ndarray],
    events: dict[tuple[str, str], np.ndarray],
    cond: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Per-event metric arrays over a fixed event set (insertion order)."""
    y = pool["gt_match"]
    s0 = np.asarray(pool["score_m_bridge"], dtype=np.float64)
    rows = [event_metrics(y, s0, cond, idx) for idx in events.values()]
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "pwa": arr[:, 0],
        "mrr": arr[:, 1],
        "top1": arr[:, 2],
        "n_good": arr[:, 3],
        "n_bad": arr[:, 4],
    }


def macro(v: np.ndarray) -> float:
    return float(v.mean()) if v.size else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    args = p.parse_args()
    study: Path = args.study_dir
    study.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    def say(line: str) -> None:
        report.append(line)
        print(line)

    # ---- V1: substrate identity (refuse on mismatch) ----
    actual = sha256(PAIRS)
    if actual != PAIRS_SHA:
        print(f"V1 FAIL: pairs SHA {actual} != declared {PAIRS_SHA}; refusing.")
        return 2
    say(f"V1 PASS  pairs.csv sha256={actual}")

    pool = load_pool()
    y = pool["gt_match"]
    n_rows = int(y.size)
    say(f"pool gt_valid rows={n_rows}  GT={int(y.sum())}  FP={int((~y).sum())}")

    # ---- NaN policy (§2) then coarse gate (§3) ----
    finite = np.ones(n_rows, dtype=bool)
    for col in SOURCE_COLS:
        finite &= np.isfinite(np.asarray(pool[col], dtype=np.float64))
    hr = pool["h_ratio_lost_over_cand"]
    band = finite & (hr >= 0.60) & (hr <= 1.70)
    n_band_pre = int(((hr >= 0.60) & (hr <= 1.70)).sum())
    nan_dropped = n_band_pre - int(band.sum())
    v5 = nan_dropped <= 0.05 * max(n_band_pre, 1)
    say(
        f"band rows={int(band.sum())}  (GT {int(y[band].sum())})  "
        f"nan_dropped={nan_dropped}  V5 {'PASS' if v5 else 'FAIL'}"
    )
    v4_frac = float(y[band].sum() / max(int(y.sum()), 1))
    v4 = v4_frac >= 0.90
    say(f"V4 {'PASS' if v4 else 'FAIL'}  GT-row retention={v4_frac:.3f}")

    seqs = sorted({str(s) for s in pool["seq"]})
    SEQS_ORDER.extend(seqs)

    events = event_index(pool, band)
    ev_keys = list(events.keys())
    ev_seq = np.asarray([k[0] for k in ev_keys], dtype=object)
    v2 = len(ev_keys) >= 150
    per_seq_n = {s: int((ev_seq == s).sum()) for s in seqs}
    v3 = sum(1 for s in seqs if per_seq_n[s] >= 10) >= 6
    say(f"V2 {'PASS' if v2 else 'FAIL'}  rankable events={len(ev_keys)}")
    say(f"V3 {'PASS' if v3 else 'FAIL'}  per-seq={per_seq_n}")

    if not (v2 and v3 and v4 and v5):
        say("TERMINAL: T0 UNRESOLVED / INVALID-STUDY (validity failure)")
        say(f"CAVEAT: {CAVEAT}")
        (study / "results.txt").write_text("\n".join(report) + "\n")
        return 1

    # ---- Baseline + headroom H (§7) ----
    base = evaluate(pool, events, None)
    base_pwa, base_mrr, base_top1 = (
        macro(base["pwa"]),
        macro(base["mrr"]),
        macro(base["top1"]),
    )
    say(f"baseline (s0): PWA={base_pwa:.4f}  MRR={base_mrr:.4f}  top1={base_top1:.4f}")
    h_hit = base_pwa >= 0.98 and base_top1 >= 0.98
    say(f"H: H1(PWA>=0.98)={base_pwa >= 0.98}  H2(top1>=0.98)={base_top1 >= 0.98}")

    # ---- Candidates: in-sample thresholds + masks ----
    thr_in = fit_thresholds(pool, band)
    say(f"in-sample thresholds: {json.dumps(thr_in, sort_keys=True)}")
    conds = condition_masks(pool, thr_in)

    # rankable-event rows for B6a fire rates
    ev_rows = np.concatenate(list(events.values()))
    ev_gt = ev_rows[y[ev_rows]]
    ev_fp = ev_rows[~y[ev_rows]]

    # P3 hard subset (baseline top-1 misses, in-sample)
    p3_mask = base["top1"] < 1.0
    say(f"P3 (baseline top-1 miss events): {int(p3_mask.sum())}/{len(ev_keys)}")

    # ---- LOO threshold sets ----
    loo_thr = {s: fit_thresholds(pool, band & (pool["seq"] != s)) for s in seqs}

    table: list[dict[str, object]] = []
    passers: list[str] = []
    for name, c in conds.items():
        cand = evaluate(pool, events, c)
        d_pwa = cand["pwa"] - base["pwa"]
        d1 = macro(d_pwa)
        per_seq_d = {s: macro(d_pwa[ev_seq == s]) for s in seqs}
        b1 = d1 >= 0.02
        nz = sum(1 for s in seqs if per_seq_d[s] >= 0)
        npos = sum(1 for s in seqs if per_seq_d[s] > 0)
        b2 = nz >= 5 and npos >= 4
        b3_val = min(macro(d_pwa[ev_seq != s]) for s in seqs)
        b3 = b3_val >= 0.01
        d_mrr = macro(cand["mrr"]) - base_mrr
        d_top1 = macro(cand["top1"]) - base_top1
        b4 = d_mrr >= 0 and d_top1 >= 0
        # B6 mechanical decomposition
        fire_fp = float(c[ev_fp].mean())
        fire_gt = float(c[ev_gt].mean())
        b6a = fire_fp > fire_gt
        n_good = float(cand["n_good"].sum())
        n_bad = float(cand["n_bad"].sum())
        b6b = n_good > n_bad
        b6 = b6a and b6b
        # B5 LOO (threshold refit per fold)
        loo_d: dict[str, float] = {}
        loo_all: list[np.ndarray] = []
        for s in seqs:
            c_fold = condition_masks(pool, loo_thr[s])[name]
            fold_events = {k: v for k, v in events.items() if k[0] == s}
            if not fold_events:
                loo_d[s] = float("nan")
                continue
            fold_base = evaluate(pool, fold_events, None)
            fold_cand = evaluate(pool, fold_events, c_fold)
            dd = fold_cand["pwa"] - fold_base["pwa"]
            loo_d[s] = macro(dd)
            loo_all.append(dd)
        loo_pooled = macro(np.concatenate(loo_all)) if loo_all else float("nan")
        loo_nz = sum(1 for s in seqs if loo_d[s] >= 0)
        b5 = loo_pooled >= 0.01 and loo_nz >= 5
        ok = b1 and b2 and b3 and b4 and b5 and b6
        if ok:
            passers.append(name)
        # P3 + reachable-slice descriptive
        p3_d_pwa = macro(d_pwa[p3_mask])
        p3_d_top1 = macro(cand["top1"][p3_mask]) - macro(base["top1"][p3_mask])
        table.append(
            {
                "candidate": name,
                "d_pwa": d1,
                "d_mrr": d_mrr,
                "d_top1": d_top1,
                "b3_min_excl": b3_val,
                "loo_pooled_d_pwa": loo_pooled,
                "loo_folds_ge0": loo_nz,
                "fire_fp": fire_fp,
                "fire_gt": fire_gt,
                "n_good": n_good,
                "n_bad": n_bad,
                "p3_d_pwa": p3_d_pwa,
                "p3_d_top1": p3_d_top1,
                **{f"d_pwa_{s}": per_seq_d[s] for s in seqs},
                **{f"loo_{s}": loo_d[s] for s in seqs},
                "B1": b1,
                "B2": b2,
                "B3": b3,
                "B4": b4,
                "B5": b5,
                "B6a": b6a,
                "B6b": b6b,
                "PASS": ok,
            }
        )
        say(
            f"{name:32s} dPWA={d1:+.4f} dMRR={d_mrr:+.4f} dtop1={d_top1:+.4f} "
            f"B3min={b3_val:+.4f} LOO={loo_pooled:+.4f}({loo_nz}/7) "
            f"fire FP/GT={fire_fp:.3f}/{fire_gt:.3f} good/bad={n_good:.0f}/"
            f"{n_bad:.0f} -> "
            f"{'PASS' if ok else 'fail'}"
            f"[{'1' if b1 else '-'}{'2' if b2 else '-'}{'3' if b3 else '-'}"
            f"{'4' if b4 else '-'}{'5' if b5 else '-'}{'6' if b6 else '-'}]"
        )

    # ---- P1 descriptive per-atom separability (no terminal force) ----
    p1_lines: list[str] = []
    for atom, tail in ATOMS:
        v = np.asarray(pool[atom], dtype=np.float64)
        v_gt, v_fp = v[ev_gt], v[ev_fp]
        # pooled rank AUC, oriented so higher = FP-like for "hi" tails
        allv = np.concatenate([v_gt, v_fp])
        order = allv.argsort(kind="mergesort").argsort().astype(np.float64) + 1.0
        r_gt = order[: v_gt.size]
        auc = float(
            (r_gt.sum() - v_gt.size * (v_gt.size + 1) / 2) / (v_gt.size * v_fp.size)
        )
        if tail == "hi":
            auc = 1.0 - auc  # P(FP ranks above GT on the unsafe side)
        p1_lines.append(f"P1 {atom:16s} tail={tail} pooled separability={auc:.4f}")
    for line in p1_lines:
        say(line)

    # ---- Reachable slice (descriptive only, §3) ----
    s0 = np.asarray(pool["score_m_bridge"], dtype=np.float64)
    reach = band & (s0 <= 0.40)
    reach_events = event_index(pool, reach)
    say(f"reachable slice: rankable events={len(reach_events)} (descriptive only)")
    if reach_events:
        rb = evaluate(pool, reach_events, None)
        say(f"  slice baseline PWA={macro(rb['pwa']):.4f} top1={macro(rb['top1']):.4f}")
        for name, c in conds.items():
            rc = evaluate(pool, reach_events, c)
            say(f"  slice {name:32s} dPWA={macro(rc['pwa'] - rb['pwa']):+.4f}")

    # ---- Terminal (§10; H precedence predeclared) ----
    if h_hit:
        terminal = "T3 NO_HEADROOM (H1 AND H2; candidates above are descriptive)"
    elif passers:
        terminal = f"T1 RANKING_SIGNAL_PRESENT (passers: {', '.join(passers)})"
    else:
        terminal = "T2 NO_USABLE_RANKING_POWER_IN_CLASS (12-member tested class)"
    say(f"TERMINAL: {terminal}")
    say(f"CAVEAT: {CAVEAT}")

    # ---- Outputs ----
    (study / "results.txt").write_text("\n".join(report) + "\n")
    with (study / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    (study / "thresholds.json").write_text(
        json.dumps(
            {"in_sample": thr_in, "loo": loo_thr, "pairs_sha256": actual},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    say(f"outputs written to {study}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
