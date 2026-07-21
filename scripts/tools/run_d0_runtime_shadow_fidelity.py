#!/usr/bin/env python3
"""D0 runtime shadow bridge fidelity — terminal verifier (Issue #112, v2).

Computes the terminal for the sealed declaration
``docs/modules/semantic/research/d0_runtime_shadow_fidelity_declaration_20260712.md``.

Fail-closed by construction:

* the four frozen input SHA256s must reproduce bit-for-bit (V6);
* the capture must be a **shadow** capture (a committing bridge rewrites track
  identity, so its ids cannot address a bridge-off cohort);
* fidelity is computed on the ``matched`` partition **only** -- ``cohort_gap``
  and ``unemitted`` are never admitted to an agreement denominator;
* the partition must conserve (matched + cohort_gap + unemitted == captured);
* the terminal is derived mechanically from the frozen, non-compensatory boxes.

This runner is **v2 only**. It never touches the v1 legacy sealed packet or its
runner; that packet stays frozen under its original semantics.
"""
# status: stable

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from saccade.perception.eval.consumer_a_bridge_fidelity import (
    PARTITION_COHORT_GAP,
    PARTITION_MATCHED,
    PARTITION_UNEMITTED,
    PARTITIONS,
    PRODUCTION_BRIDGE_PX,
    SPEED_WEIGHT_REF,
)

REPO = Path(__file__).resolve().parents[2]

# ── Frozen boxes (declaration §5; owner-confirmed 2026-07-12 before any metric
# was computed). Non-compensatory: threshold transfer (B1 ∧ B2) and rank
# transfer (B3) cannot substitute for one another.
B1_DECISION_AGREEMENT_MIN = 0.99
B2_ABSDELTA_Q95_MAX = 0.05
B3_SPEARMAN_RHO_MIN = 0.98

# Frozen estimator conventions. No implementer degrees of freedom.
THRESHOLD = PRODUCTION_BRIDGE_PX  # 0.4, inclusive (`<=`), matching the kernel
QUANTILE_METHOD = "linear"  # numpy type-7
TIE_POLICY = "average_ranks"  # scipy spearmanr default
V5_MIN_MATCHED = 1000
V5_MAX_NAN_FRACTION = 0.05

TERMINAL_FAITHFUL = "T1_PROXY_FAITHFUL"
TERMINAL_UNFAITHFUL = "T2_PROXY_UNFAITHFUL"
TERMINAL_NON_COVERING = "T3_FAITHFUL_BUT_NON_COVERING"
TERMINAL_UNRESOLVED = "T0_UNRESOLVED_INVALID_STUDY"


class ValidityFailure(RuntimeError):
    """A V-gate failed: the study is UNRESOLVED, not a fidelity finding."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: np.ndarray, p: float) -> float:
    return float(np.quantile(values, p, method=QUANTILE_METHOD))


def _col(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def offline_proxy_scores(pairs: Sequence[dict[str, Any]]) -> np.ndarray:
    """`s0 = score_m_bridge`, the production-shaped offline proxy.

    Mirrors ``audit_relink_safe_reject.ensure_prod_proxy_scores``. Never re-fit:
    this study measures the estimator that prior conclusions were built on.
    """
    speed = _col(pairs, "lost_exit_speed")
    w = np.sqrt(np.clip(speed / SPEED_WEIGHT_REF, 0.0, 1.0))
    return w * 0.5 * (_col(pairs, "fwd_resid") + _col(pairs, "bwd_resid")) + (
        1.0 - w
    ) * _col(pairs, "dist_h")


def verify_hashes(
    expected: dict[str, str], resolved: dict[str, Path]
) -> dict[str, str]:
    """V6 — the frozen inputs must reproduce bit-for-bit."""
    observed: dict[str, str] = {}
    mismatched: list[str] = []
    for name, want in expected.items():
        path = resolved[name]
        if not path.is_file():
            raise ValidityFailure(f"V6: frozen input missing: {path}")
        got = _sha256(path)
        observed[name] = got
        if got != want:
            mismatched.append(f"{name}: expected {want[:12]}…, got {got[:12]}…")
    if mismatched:
        raise ValidityFailure(f"V6: frozen inputs changed: {mismatched}")
    return observed


def run(
    study_dir: Path,
    substrate_dir: Path,
    expected_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    capture_csv = study_dir / "capture.csv.gz"
    pairs_csv = study_dir / "pairs.csv"
    manifest_path = study_dir / "capture.csv.gz.manifest.json"
    id_map = substrate_dir / "_global_id_map.txt"
    mot_files = sorted(substrate_dir.glob("MOT17-*.txt"))

    if expected_hashes:
        substrate_digest = hashlib.sha256()
        for path in mot_files:
            substrate_digest.update(path.read_bytes())
        resolved = {
            "pairs.csv": pairs_csv,
            "capture.csv.gz": capture_csv,
            "_global_id_map.txt": id_map,
        }
        hashes = verify_hashes(
            {k: v for k, v in expected_hashes.items() if k in resolved}, resolved
        )
        want_mot = expected_hashes.get("substrate_mot_concat")
        got_mot = substrate_digest.hexdigest()
        if want_mot and got_mot != want_mot:
            raise ValidityFailure(
                f"V6: substrate MOT changed: expected {want_mot[:12]}…, got {got_mot[:12]}…"
            )
        hashes["substrate_mot_concat"] = got_mot
    else:
        hashes = {
            "pairs.csv": _sha256(pairs_csv),
            "capture.csv.gz": _sha256(capture_csv),
            "_global_id_map.txt": _sha256(id_map),
        }

    export_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = export_manifest.get("provenance", {})

    # V1 — a committing bridge would have rewritten the ids we join on.
    if provenance.get("shadow") is not True:
        raise ValidityFailure(
            "V1: capture is not a shadow capture; a committing bridge mutates "
            "track identity and cannot be joined to a bridge-off cohort"
        )
    # V2 — no dropped events.
    if int(export_manifest.get("overflow_events", -1)) != 0:
        raise ValidityFailure("V2: capture overflowed")

    with gzip.open(capture_csv, "rt", encoding="utf-8") as stream:
        capture = list(csv.DictReader(stream))
    with pairs_csv.open(encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))

    parts = {p: [r for r in capture if r["partition"] == p] for p in PARTITIONS}
    matched, cohort_gap, unemitted = (
        parts[PARTITION_MATCHED],
        parts[PARTITION_COHORT_GAP],
        parts[PARTITION_UNEMITTED],
    )

    # V4 — the partition is exhaustive and mutually exclusive.
    if len(matched) + len(cohort_gap) + len(unemitted) != len(capture):
        raise ValidityFailure(
            f"V4: partition does not conserve: "
            f"{len(matched)}+{len(cohort_gap)}+{len(unemitted)} != {len(capture)}"
        )
    # V3 — no silent local-id fallback.
    leaked = [r for r in capture if r["event_key"] and int(r["lost_global_id"]) < 0]
    if leaked:
        raise ValidityFailure(f"V3: {len(leaked)} keyed rows carry an unresolved id")

    # ── join: matched rows carry global ids; the cohort is keyed on them ──────
    proxy_by_key = {
        (r["seq"], int(r["lost_id"]), int(r["cand_id"])): s
        for r, s in zip(pairs, offline_proxy_scores(pairs))
    }
    s0 = np.asarray(
        [
            proxy_by_key[(r["seq"], int(r["lost_global_id"]), int(r["cand_global_id"]))]
            for r in matched
        ],
        dtype=np.float64,
    )
    bdist = _col(matched, "bdist")

    # V5 — sample size and missingness.
    if len(matched) < V5_MIN_MATCHED:
        raise ValidityFailure(f"V5: matched N={len(matched)} < {V5_MIN_MATCHED}")
    n_nan = int(np.sum(~np.isfinite(s0)) + np.sum(~np.isfinite(bdist)))
    if n_nan:
        raise ValidityFailure(f"V5: {n_nan} non-finite values in required columns")

    # ── F1 threshold decision agreement (confusion never netted) ─────────────
    acc_runtime = bdist <= THRESHOLD
    acc_proxy = s0 <= THRESHOLD
    confusion = {
        "both_accept": int(np.sum(acc_runtime & acc_proxy)),
        "both_reject": int(np.sum(~acc_runtime & ~acc_proxy)),
        "proxy_accept_only": int(np.sum(acc_proxy & ~acc_runtime)),
        "runtime_accept_only": int(np.sum(acc_runtime & ~acc_proxy)),
    }
    f1 = float(np.mean(acc_runtime == acc_proxy))

    # ── F2 numeric error: q50/q90/q95, absolute and as a fraction of 0.4 ─────
    delta = s0 - bdist
    absdelta = np.abs(delta)
    f2 = {
        "delta_median": float(np.median(delta)),
        "delta_iqr": [_quantile(delta, 0.25), _quantile(delta, 0.75)],
        "absdelta_q50": _quantile(absdelta, 0.50),
        "absdelta_q90": _quantile(absdelta, 0.90),
        "absdelta_q95": _quantile(absdelta, 0.95),
    }
    f2["absdelta_q50_frac_threshold"] = f2["absdelta_q50"] / THRESHOLD
    f2["absdelta_q90_frac_threshold"] = f2["absdelta_q90"] / THRESHOLD
    f2["absdelta_q95_frac_threshold"] = f2["absdelta_q95"] / THRESHOLD

    # ── F3 rank agreement ────────────────────────────────────────────────────
    f3 = float(stats.spearmanr(s0, bdist).statistic)

    boxes = {
        "B1": bool(f1 >= B1_DECISION_AGREEMENT_MIN),
        "B2": bool(f2["absdelta_q95"] <= B2_ABSDELTA_Q95_MAX),
        "B3": bool(f3 >= B3_SPEARMAN_RHO_MIN),
    }

    # ── component attribution (diagnostic; cannot move the terminal) ─────────
    def offline_column(offline_key: str) -> np.ndarray:
        by_key = {
            (r["seq"], int(r["lost_id"]), int(r["cand_id"])): float(r[offline_key])
            for r in pairs
        }
        return np.asarray(
            [
                by_key[(r["seq"], int(r["lost_global_id"]), int(r["cand_global_id"]))]
                for r in matched
            ],
            dtype=np.float64,
        )

    components: dict[str, dict[str, float]] = {}
    absdiff: dict[str, np.ndarray] = {}
    for runtime_key, offline_key in (
        ("dist_h", "dist_h"),
        ("fwd_r", "fwd_resid"),
        ("bwd_r", "bwd_resid"),
    ):
        off = offline_column(offline_key)
        rt = _col(matched, runtime_key)
        d = np.abs(off - rt)
        absdiff[runtime_key] = d
        components[runtime_key] = {
            "absdiff_median": float(np.median(d)),
            "absdiff_q95": _quantile(d, 0.95),
            "spearman_rho": float(stats.spearmanr(off, rt).statistic),
        }

    # Horizon signature: the two reduction errors separate on the extrapolation
    # horizon. The scale operator (EMA vs raw height) contaminates even the
    # 0th-order dist_h term and is horizon-INDEPENDENT (a floor that does not
    # vanish as gap -> 0). The velocity operator's error is horizon-AMPLIFIED,
    # as `x + v*horizon` predicts.
    la = _col(matched, "la")
    horizon: dict[str, Any] = {
        "rho_la_vs_absdiff": {
            key: float(stats.spearmanr(la, absdiff[key]).statistic)
            for key in ("dist_h", "fwd_r", "bwd_r")
        },
        "bins": [],
    }
    for lo, hi in ((0, 6), (6, 11), (11, 16), (16, 22), (22, 10**9)):
        mask = (la >= lo) & (la < hi)
        if int(mask.sum()) == 0:
            continue
        horizon["bins"].append(
            {
                "la_lo": lo,
                "la_hi": None if hi == 10**9 else hi,
                "n": int(mask.sum()),
                "absdiff_dist_h_median": float(np.median(absdiff["dist_h"][mask])),
                "absdiff_fwd_r_median": float(np.median(absdiff["fwd_r"][mask])),
            }
        )

    # ── Issue #112 original reporting surface (diagnostic; the terminal is
    # already fixed by the frozen boxes above and cannot move here).
    #
    # The issue named three prior suspects; all three are confirmed by
    # `component_attribution` + `horizon_signature`:
    #   velocity estimator (anchor-4 foot-ring vs window-mean),
    #   extrapolation horizon (la vs offline pair gap),
    #   normalization (bilateral EMA h_ref vs raw endpoint-height mean).
    gt_by_key = {
        (r["seq"], int(r["lost_id"]), int(r["cand_id"])): (
            int(r.get("gt_match", 0) or 0),
            int(r.get("gt_valid", 0) or 0),
        )
        for r in pairs
    }
    gt_flags = np.asarray(
        [
            gt_by_key[(r["seq"], int(r["lost_global_id"]), int(r["cand_global_id"]))]
            for r in matched
        ],
        dtype=np.int64,
    )
    is_gt = (gt_flags[:, 0] == 1) & (gt_flags[:, 1] == 1)
    is_fp = (gt_flags[:, 0] == 0) & (gt_flags[:, 1] == 1)

    def _slice_report(mask: np.ndarray) -> dict[str, Any] | None:
        n = int(mask.sum())
        if n < 2:
            return None
        a, b = s0[mask], bdist[mask]
        rho = float(stats.spearmanr(a, b).statistic) if n >= 3 else float("nan")
        acc_a, acc_b = a <= THRESHOLD, b <= THRESHOLD
        # "offline-safe / online-unsafe": the proxy would accept, the kernel
        # rejects. The asymmetric error an offline gate conclusion would make.
        unsafe = int(np.sum(acc_a & ~acc_b))
        return {
            "n": n,
            "spearman_rho": rho,
            "decision_agreement": float(np.mean(acc_a == acc_b)),
            "absdelta_q50": _quantile(np.abs(a - b), 0.50),
            "absdelta_q95": _quantile(np.abs(a - b), 0.95),
            # q85 <-> q85 quantile-alignment error (distribution alignment, not
            # per-event error): |Q85(s0) - Q85(bdist)|.
            "q85_alignment_error": abs(_quantile(a, 0.85) - _quantile(b, 0.85)),
            "offline_safe_online_unsafe": unsafe,
            "offline_safe_online_unsafe_rate": unsafe / n,
        }

    offline_gap = np.asarray(
        [
            {
                (r["seq"], int(r["lost_id"]), int(r["cand_id"])): float(r["gap"])
                for r in pairs
            }[(r["seq"], int(r["lost_global_id"]), int(r["cand_global_id"]))]
            for r in matched
        ],
        dtype=np.float64,
    )
    issue_112: dict[str, Any] = {
        "terminal_vocabulary": "not_fidelity_aligned",  # issue's own wording for T2
        "prior_suspects_confirmed": [
            "velocity_estimator",
            "extrapolation_horizon",
            "normalization_ema_vs_raw_height",
        ],
        "overall": _slice_report(np.ones(len(matched), dtype=bool)),
        "gt_conditional": _slice_report(is_gt),
        "fp_conditional": _slice_report(is_fp),
        "by_gap_slice_S_A": {},
    }
    # Row-level reachable support S_A = {1 <= gap <= 26}, sliced.
    for lo, hi in ((1, 6), (6, 11), (11, 16), (16, 21), (21, 27)):
        rep = _slice_report((offline_gap >= lo) & (offline_gap < hi))
        if rep:
            issue_112["by_gap_slice_S_A"][f"[{lo},{hi})"] = rep

    # ── COVERAGE ─────────────────────────────────────────────────────────────
    mappable = len(matched) + len(cohort_gap)
    c1 = {
        "matched_share_all_captured": len(matched) / len(capture),
        "matched_share_mappable": len(matched) / mappable,
    }

    def _ks(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        res = stats.ks_2samp(a, b)
        return {"ks": float(res.statistic), "p": float(res.pvalue)}

    c2: dict[str, Any] = {"bdist": {}, "gap": {}, "per_sequence": {}}
    for label, group in (("cohort_gap", cohort_gap), ("unemitted", unemitted)):
        c2["bdist"][f"matched_vs_{label}"] = _ks(bdist, _col(group, "bdist"))
        c2["gap"][f"matched_vs_{label}"] = _ks(_col(matched, "gap"), _col(group, "gap"))
    c2["bdist"]["medians"] = {
        "matched": float(np.median(bdist)),
        "cohort_gap": float(np.median(_col(cohort_gap, "bdist"))),
        "unemitted": float(np.median(_col(unemitted, "bdist"))),
    }
    for seq in sorted({r["seq"] for r in capture}):
        c2["per_sequence"][seq] = {
            p: sum(1 for r in parts[p] if r["seq"] == seq) for p in PARTITIONS
        }

    # C3 — composition restricted to the region production actually acts in.
    accept_counts = {
        p: int(np.sum(_col(parts[p], "bdist") <= THRESHOLD)) if parts[p] else 0
        for p in PARTITIONS
    }
    total_accept = sum(accept_counts.values())
    c3: dict[str, Any] = {
        "threshold": THRESHOLD,
        "total_accept_region": total_accept,
        "counts": accept_counts,
        "shares": {},
    }
    coverage_pass = True
    for p in PARTITIONS:
        in_accept = accept_counts[p] / total_accept if total_accept else 0.0
        overall = len(parts[p]) / len(capture)
        c3["shares"][p] = {
            "accept_region": in_accept,
            "overall": overall,
            "delta_pp": 100.0 * (in_accept - overall),
        }
    # Coverage fails if the matched share in the accept region is materially
    # worse than overall (>= 10 pp shortfall) -- i.e. the cohort under-covers
    # exactly where production fires.
    coverage_pass = c3["shares"][PARTITION_MATCHED]["delta_pp"] > -10.0
    c3["coverage_pass"] = coverage_pass

    # ── Terminal (mechanical; boxes are non-compensatory) ────────────────────
    if all(boxes.values()):
        terminal = TERMINAL_FAITHFUL if coverage_pass else TERMINAL_NON_COVERING
    else:
        terminal = TERMINAL_UNFAITHFUL

    return {
        "study": "d0_runtime_shadow_fidelity_20260712",
        "terminal": terminal,
        "declaration": (
            "docs/modules/semantic/research/"
            "d0_runtime_shadow_fidelity_declaration_20260712.md"
        ),
        "validity": {
            "V1_shadow": True,
            "V2_overflow_events": 0,
            "V3_unresolved_keyed_rows": 0,
            "V4_partition_conserved": True,
            "V5_matched_n": len(matched),
            "V6_hashes": hashes,
        },
        "partition": {p: len(parts[p]) for p in PARTITIONS},
        "boxes": boxes,
        "box_bars": {
            "B1_decision_agreement_min": B1_DECISION_AGREEMENT_MIN,
            "B2_absdelta_q95_max": B2_ABSDELTA_Q95_MAX,
            "B3_spearman_rho_min": B3_SPEARMAN_RHO_MIN,
            "non_compensatory": True,
        },
        "conventions": {
            "threshold": THRESHOLD,
            "threshold_inclusive": "<=",
            "quantile_method": QUANTILE_METHOD,
            "tie_policy": TIE_POLICY,
        },
        "F1_decision_agreement": f1,
        "F1_confusion": confusion,
        "F2_numeric_error": f2,
        "F3_spearman_rho": f3,
        "component_attribution": components,
        "horizon_signature": horizon,
        "issue_112_reporting_surface": issue_112,
        "C1_share": c1,
        "C2_structural_bias": c2,
        "C3_accept_region": c3,
        "provenance": provenance,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--substrate-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-hashes",
        type=Path,
        help="JSON {name: sha256} of the frozen inputs; enables V6 (fail-closed)",
    )
    parser.add_argument("--output", type=Path, help="write metrics JSON here")
    args = parser.parse_args(argv)

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else REPO / p

    expected = (
        json.loads(_abs(args.expected_hashes).read_text(encoding="utf-8"))
        if args.expected_hashes
        else None
    )
    try:
        metrics = run(_abs(args.study_dir), _abs(args.substrate_dir), expected)
    except ValidityFailure as exc:
        print(
            json.dumps(
                {"terminal": TERMINAL_UNRESOLVED, "validity_failure": str(exc)},
                indent=2,
            )
        )
        return 1

    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        out = _abs(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
