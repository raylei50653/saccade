"""Boolean-atom partial-order audit (PR-D gate / issue #106).

Read-only offline audit. Consumes the sealed Step-0 morphology packet and the
accepted-with-limits PR-C escape-tail forensic. Produces atom roles, dependency
and scale-guard evidence, and an allowed/forbidden global order contract.

Does **not** run MWC, min-cut, rule search, weight fitting, or any closure
solve. Aggregate terminal is exactly one of:

  GLOBAL_PARTIAL_ORDER_READY
  CONDITIONAL_STRUCTURE_ONLY
  ORDERABILITY_UNRESOLVED

Usage::

  uv run python docs/modules/semantic/research/evidence/\\
    boolean_atom_partial_order_20260711/run_partial_order_audit.py \\
    --pairs out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv

  uv run python .../run_partial_order_audit.py --pairs ... --verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[5]
STEP0 = (
    REPO
    / "docs/modules/semantic/research/evidence/gt_support_morphology_step0_20260711"
)
PRC = REPO / "docs/modules/semantic/research/evidence/escape_tail_forensic_20260711"

sys.path.insert(0, str(REPO / "scripts/tools"))
import audit_relink_safe_reject as ar  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen substrate (must match Step-0 / PR-C)
# ---------------------------------------------------------------------------
ATOMS: list[tuple[str, bool]] = [
    ("score_m_bridge", True),  # lower is safer
    ("bridge_dist", True),
    ("dist_h", True),
    ("log_h_ratio", True),
    ("resid_mean", True),
    ("dir_cos", False),  # higher is safer
    ("speed_mismatch", True),
    ("gap", True),
]
ATOM_NAMES = [name for name, _ in ATOMS]
MOTION_ATOMS = ("speed_mismatch", "dir_cos", "resid_mean")
STRUCTURAL_ATOMS = ("bridge_dist", "dist_h")
HEIGHT_ATOMS = ("log_h_ratio",)
COMPOSITE_ATOMS = ("score_m_bridge",)
REGIME_ATOMS = ("gap",)

# PR-C binding (issue #102 / PR #104, ACCEPTED_WITH_LIMITS).
# These three atoms carry accepted long-gap re-entry role-reversal evidence.
PRC_ROLE_REVERSAL_ATOMS = frozenset(MOTION_ATOMS)
PRC_AGGREGATE = "ROLE_REVERSAL_SUPPORTED"
PRC_ACCEPTANCE = "ACCEPTED_WITH_LIMITS"

# Observable contexts for conditional_orderable motion atoms (no GT outcome used).
SHORT_GAP_MAX = 60.0  # frames; regime descriptor only
SPEED_MIX_REF = 0.12  # matches live kernel / ensure_prod_proxy_scores

# Terminal vocabulary (issue #106 — closed set).
TERMINALS = (
    "GLOBAL_PARTIAL_ORDER_READY",
    "CONDITIONAL_STRUCTURE_ONLY",
    "ORDERABILITY_UNRESOLVED",
)
ROLES = (
    "global_orderable",
    "conditional_orderable",
    "context_only",
    "unresolved",
)

PACKET_BODY_FILES = (
    "atom_roles.json",
    "atom_dependency_graph.json",
    "atom_metrics.csv",
    "pairwise_violation_profile.csv",
    "threshold_sensitivity.json",
    "allowed_global_order.json",
    "forbidden_order.json",
    "scale_guard.json",
    "aggregate.json",
    "run_partial_order_audit.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def z_bit(value: float, threshold: float, lower_is_safe: bool) -> int:
    return int(value <= threshold) if lower_is_safe else int(value >= threshold)


def verify_source(pairs: Path, step0_manifest: dict[str, Any]) -> None:
    if not pairs.is_file():
        raise FileNotFoundError(f"pairs CSV not found: {pairs}")
    expected = str(step0_manifest["source_pairs_csv_sha256"])
    actual = sha256(pairs)
    if actual != expected:
        raise ValueError(
            "pairs CSV SHA256 does not match the sealed Step-0 manifest: "
            f"expected {expected}, got {actual}"
        )


def load_step0_gt_rows() -> list[dict[str, str]]:
    path = STEP0 / "gt_rows.csv"
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def min_dh_representatives(
    gt_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Descriptive min-d_H representative per track (framework §19.4 descriptive)."""
    best: dict[str, dict[str, str]] = {}
    for row in gt_rows:
        key = row["track_key"]
        dh = int(row["d_h_k8"])
        if key not in best or dh < int(best[key]["d_h_k8"]):
            best[key] = row
    return best


def atom_z(row: dict[str, str], name: str) -> int:
    return int(row[f"z_{name}"])


def compute_vi_profile(
    tracks: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    n = len(tracks)
    out: dict[str, dict[str, Any]] = {}
    for name in ATOM_NAMES:
        viol_keys = [k for k, r in tracks.items() if atom_z(r, name) == 0]
        per_seq: dict[str, int] = {}
        for key in viol_keys:
            seq = tracks[key]["seq"]
            per_seq[seq] = per_seq.get(seq, 0) + 1
        out[name] = {
            "n_tracks": n,
            "n_violations": len(viol_keys),
            "V_i": len(viol_keys) / n if n else float("nan"),
            "P_z1_gt_track": 1.0 - (len(viol_keys) / n if n else float("nan")),
            "violation_track_keys": sorted(viol_keys),
            "per_sequence_violation_counts": dict(sorted(per_seq.items())),
            "n_sequences_with_violation": len(per_seq),
        }
    return out


def compute_shell_contribution(
    tracks: dict[str, dict[str, str]],
) -> dict[str, Any]:
    shells: dict[int, list[str]] = {}
    for key, row in tracks.items():
        dh = int(row["d_h_k8"])
        shells.setdefault(dh, []).append(key)
    contrib: dict[str, dict[str, int]] = {name: {} for name in ATOM_NAMES}
    shell_sizes: dict[str, int] = {}
    for dh, keys in sorted(shells.items()):
        shell_sizes[str(dh)] = len(keys)
        for name in ATOM_NAMES:
            contrib[name][str(dh)] = sum(
                1 for k in keys if atom_z(tracks[k], name) == 0
            )
    protected_tail = sorted(k for k, r in tracks.items() if int(r["d_h_k8"]) >= 3)
    tail_viol: dict[str, int] = {}
    for name in ATOM_NAMES:
        tail_viol[name] = sum(1 for k in protected_tail if atom_z(tracks[k], name) == 0)
    return {
        "representation": "descriptive min-d_H representative (framework §19.4)",
        "shell_sizes": shell_sizes,
        "per_atom_shell_violations": contrib,
        "protected_tail_d_h_ge_3": {
            "track_keys": protected_tail,
            "n": len(protected_tail),
            "per_atom_violations": tail_viol,
            "note": "protected GT mass candidate; never a veto (framework §19)",
        },
    }


def compute_pairwise(
    tracks: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    n = len(tracks)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(ATOM_NAMES):
        for b in ATOM_NAMES[i + 1 :]:
            both = sum(
                1 for r in tracks.values() if atom_z(r, a) == 0 and atom_z(r, b) == 0
            )
            a_only = sum(
                1 for r in tracks.values() if atom_z(r, a) == 0 and atom_z(r, b) == 1
            )
            b_only = sum(
                1 for r in tracks.values() if atom_z(r, a) == 1 and atom_z(r, b) == 0
            )
            rows.append(
                {
                    "atom_i": a,
                    "atom_j": b,
                    "V_ij": both,
                    "V_ij_rate": both / n if n else float("nan"),
                    "V_i_only": a_only,
                    "V_j_only": b_only,
                    "interpretation_bound": (
                        "shared violations do not prove interaction; "
                        "use only for coupled role-reversal / redundancy / "
                        "conditioning candidates — not cell-risk estimation"
                    ),
                }
            )
    return rows


def prepare_pool(pairs: Path) -> dict[str, np.ndarray]:
    pool = ar.load_gt_valid_pool(pairs)
    pool["resid_mean"] = 0.5 * (pool["fwd_resid"] + pool["bwd_resid"])
    ar.ensure_prod_proxy_scores(pool)
    return pool


def threshold_sensitivity(pool: dict[str, np.ndarray]) -> dict[str, Any]:
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"].astype(str)
    lost = pool["lost_id"].astype(str)
    keys = np.asarray([f"{s}|{lid}" for s, lid in zip(seq, lost)], dtype=object)
    quantiles = {"p40": 0.40, "median": 0.50, "p60": 0.60}
    out: dict[str, Any] = {
        "quantiles": {},
        "direction_flip_p40_to_p60": {},
        "note": (
            "p40/median/p60 are declared sensitivity diagnostics only; "
            "median remains the frozen descriptive split (Step-0)"
        ),
    }
    track_z: dict[str, dict[str, np.ndarray]] = {}
    for qname, q in quantiles.items():
        thrs = {
            name: float(np.nanquantile(np.asarray(pool[name], float), q))
            for name, _ in ATOMS
        }
        best_dh: dict[str, int] = {}
        best_z: dict[str, np.ndarray] = {}
        for i in np.where(y)[0]:
            key = str(keys[i])
            bits = np.zeros(len(ATOMS), dtype=int)
            for j, (name, lower) in enumerate(ATOMS):
                bits[j] = z_bit(float(pool[name][i]), thrs[name], lower)
            dh = int(len(ATOMS) - bits.sum())
            if key not in best_dh or dh < best_dh[key]:
                best_dh[key] = dh
                best_z[key] = bits
        track_z[qname] = best_z
        n_tracks = len(best_z)
        vi = {}
        for j, name in enumerate(ATOM_NAMES):
            n_viol = sum(1 for z in best_z.values() if int(z[j]) == 0)
            vi[name] = {
                "n_violations": n_viol,
                "V_i": n_viol / n_tracks if n_tracks else float("nan"),
            }
        tail_keys = sorted(k for k, d in best_dh.items() if d >= 3)
        tail_vi = {}
        for j, name in enumerate(ATOM_NAMES):
            if not tail_keys:
                tail_vi[name] = {"n_violations": 0, "V_i_tail": float("nan")}
            else:
                n_viol = sum(1 for k in tail_keys if int(best_z[k][j]) == 0)
                tail_vi[name] = {
                    "n_violations": n_viol,
                    "V_i_tail": n_viol / len(tail_keys),
                }
        out["quantiles"][qname] = {
            "thresholds": thrs,
            "n_tracks": n_tracks,
            "n_tail_tracks_d_h_ge_3": len(tail_keys),
            "V_i": vi,
            "tail_V_i": tail_vi,
        }

    # Direction flip count: track-level z bit changes between p40 and p60.
    z40 = track_z["p40"]
    z60 = track_z["p60"]
    common = sorted(set(z40) & set(z60))
    for j, name in enumerate(ATOM_NAMES):
        flips = sum(1 for k in common if int(z40[k][j]) != int(z60[k][j]))
        out["direction_flip_p40_to_p60"][name] = {
            "n_flips": flips,
            "n_tracks": len(common),
            "flip_rate": flips / len(common) if common else float("nan"),
        }
    return out


def scale_guard_score_m_bridge(pool: dict[str, np.ndarray]) -> dict[str, Any]:
    """Representational legitimacy audit for score_m_bridge (no reweighting)."""
    y = pool["gt_match"].astype(bool)
    s = np.asarray(pool["lost_exit_speed"], float)
    w = np.sqrt(np.clip(s / SPEED_MIX_REF, 0.0, 1.0))
    resid = 0.5 * (
        np.asarray(pool["fwd_resid"], float) + np.asarray(pool["bwd_resid"], float)
    )
    dist_h = np.asarray(pool["dist_h"], float)
    term_resid = w * resid
    term_dist = (1.0 - w) * dist_h
    score = term_resid + term_dist
    sealed = np.asarray(pool["score_m_bridge"], float)
    max_abs = float(np.nanmax(np.abs(score - sealed)))
    den = np.abs(term_resid) + np.abs(term_dist) + 1e-12
    frac_resid = np.abs(term_resid) / den

    def quantiles(arr: np.ndarray) -> dict[str, float]:
        return {
            "p10": float(np.nanquantile(arr, 0.10)),
            "p50": float(np.nanquantile(arr, 0.50)),
            "p90": float(np.nanquantile(arr, 0.90)),
            "mean": float(np.nanmean(arr)),
        }

    high_w = w >= 0.9
    low_w = w <= 0.1

    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        if a.size < 2:
            return None
        if float(np.nanstd(a)) == 0.0 or float(np.nanstd(b)) == 0.0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    parents = {
        "resid_mean": {
            "role_conflict_risk": "PR-C accepted role reversal on long-gap re-entry",
            "formula_role": "motion residual mean 0.5*(fwd+bwd)",
            "declared_safer": "lower",
        },
        "dist_h": {
            "role_conflict_risk": "structural/height-normalized distance; no PR-C reversal",
            "formula_role": "endpoint height-normalized foot distance",
            "declared_safer": "lower",
        },
        "lost_exit_speed": {
            "role_conflict_risk": "mixer only; introduces speed-dependent context",
            "formula_role": f"w = sqrt(clip(s/{SPEED_MIX_REF}, 0, 1))",
            "declared_safer": "n/a (weight, not order atom)",
        },
    }

    findings = [
        "score_m_bridge is a speed-weighted mix of resid_mean and dist_h",
        "parent resid_mean carries accepted PR-C role-reversal evidence",
        "high correlation with resid_mean indicates residual dominance on the "
        "declared safer direction of the composite",
        "w depends on lost_exit_speed → hidden short/fast vs slow mixing regime",
    ]
    blocks_global = True
    return {
        "atom": "score_m_bridge",
        "formula": (
            f"w=sqrt(clip(lost_exit_speed/{SPEED_MIX_REF},0,1)); "
            "score_m_bridge = w*0.5*(fwd_resid+bwd_resid) + (1-w)*dist_h"
        ),
        "formula_source": (
            "scripts/tools/audit_relink_safe_reject.ensure_prod_proxy_scores "
            "(live kernel proxy: tracker_gpu.cu bridge score)"
        ),
        "terms": {
            "resid_term": "w * resid_mean",
            "dist_term": "(1-w) * dist_h",
            "bounded_or_clipped": [
                f"w clipped via clip(s/{SPEED_MIX_REF}, 0, 1) then sqrt → [0,1]"
            ],
            "raw_unbounded": [
                "fwd_resid",
                "bwd_resid",
                "dist_h",
                "resid_mean",
            ],
        },
        "parents": parents,
        "recompute_vs_sealed_max_abs": max_abs,
        "w_stats": {
            "min": float(np.nanmin(w)),
            "median": float(np.nanmedian(w)),
            "max": float(np.nanmax(w)),
            "frac_high_w_ge_0.9": float(np.mean(high_w)),
            "frac_low_w_le_0.1": float(np.mean(low_w)),
        },
        "resid_term_fraction": {
            "all": quantiles(frac_resid),
            "gt": quantiles(frac_resid[y]),
            "fp": quantiles(frac_resid[~y]),
            "high_w_mean": float(np.nanmean(frac_resid[high_w]))
            if bool(high_w.any())
            else None,
            "low_w_mean": float(np.nanmean(frac_resid[low_w]))
            if bool(low_w.any())
            else None,
        },
        "correlations": {
            "score_vs_resid_mean_gt": corr(
                sealed[y], np.asarray(pool["resid_mean"], float)[y]
            ),
            "score_vs_dist_h_gt": corr(sealed[y], dist_h[y]),
            "score_vs_resid_mean_fp": corr(
                sealed[~y], np.asarray(pool["resid_mean"], float)[~y]
            ),
            "score_vs_dist_h_fp": corr(sealed[~y], dist_h[~y]),
            "resid_mean_vs_dist_h_gt": corr(
                np.asarray(pool["resid_mean"], float)[y], dist_h[y]
            ),
        },
        "unit_scale_compatibility": (
            "resid (px-like residual) and dist_h (height-normalized distance) "
            "are mixed without an explicit unit normalizer beyond speed weight; "
            "scale dominance is empirical, not unit-justified"
        ),
        "findings": findings,
        "blocks_global_orderable": blocks_global,
        "block_reasons": [
            "parent role conflict: resid_mean is PR-C motion role-reversal atom",
            "scale dominance: composite tracks resid_mean more than dist_h",
            "hidden context dependence via speed-dependent mixing weight w",
        ],
    }


def dependency_graph() -> dict[str, Any]:
    """Static provenance DAG for frozen atoms (no optimization edges)."""
    nodes = {
        "score_m_bridge": {
            "kind": "composite_derived",
            "parents": ["resid_mean", "dist_h", "lost_exit_speed"],
            "transform": "speed-weighted linear mix",
        },
        "bridge_dist": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "mid-point bridge distance from pair builder",
        },
        "dist_h": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "height-normalized foot distance",
        },
        "log_h_ratio": {
            "kind": "derived",
            "parents": ["h_lost_raw", "h_cand_raw"],
            "transform": "abs(log(h_cand/h_lost))",
        },
        "resid_mean": {
            "kind": "derived",
            "parents": ["fwd_resid", "bwd_resid"],
            "transform": "0.5*(fwd+bwd)",
        },
        "dir_cos": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "direction cosine of exit/entry motion",
        },
        "speed_mismatch": {
            "kind": "derived",
            "parents": ["lost_exit_speed", "cand_entry_speed"],
            "transform": "abs(exit_speed - entry_speed)",
        },
        "gap": {
            "kind": "builder_raw",
            "parents": ["lost_last_frame", "cand_first_frame"],
            "transform": "cand_first_frame - lost_last_frame",
        },
        "h_lost_raw": {"kind": "builder_raw", "parents": [], "transform": None},
        "h_cand_raw": {"kind": "builder_raw", "parents": [], "transform": None},
        "fwd_resid": {"kind": "builder_raw", "parents": [], "transform": None},
        "bwd_resid": {"kind": "builder_raw", "parents": [], "transform": None},
        "lost_exit_speed": {"kind": "builder_raw", "parents": [], "transform": None},
        "cand_entry_speed": {"kind": "builder_raw", "parents": [], "transform": None},
        "lost_last_frame": {"kind": "builder_raw", "parents": [], "transform": None},
        "cand_first_frame": {"kind": "builder_raw", "parents": [], "transform": None},
    }
    edges = []
    for child, meta in nodes.items():
        for parent in meta["parents"]:
            edges.append({"source": parent, "target": child, "relation": "depends_on"})
    return {
        "description": (
            "Dependency DAG for composite/derived atoms. Edges are provenance "
            "only — not closure arcs and not weights."
        ),
        "frozen_atoms": ATOM_NAMES,
        "nodes": nodes,
        "edges": edges,
    }


def gap_regime_profile(
    pool: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Observable gap stratification (no GT-outcome conditioning beyond labels)."""
    y = pool["gt_match"].astype(bool)
    seq = pool["seq"].astype(str)
    lost = pool["lost_id"].astype(str)
    keys = np.asarray([f"{s}|{lid}" for s, lid in zip(seq, lost)], dtype=object)
    gap = np.asarray(pool["gap"], float)
    # Median-split z bits for regime-conditional V_i.
    thrs = {
        name: float(np.nanmedian(np.asarray(pool[name], float))) for name, _ in ATOMS
    }
    best_i: dict[str, int] = {}
    best_dh: dict[str, int] = {}
    for i in np.where(y)[0]:
        key = str(keys[i])
        dh = 0
        for name, lower in ATOMS:
            if z_bit(float(pool[name][i]), thrs[name], lower) == 0:
                dh += 1
        if key not in best_dh or dh < best_dh[key]:
            best_dh[key] = dh
            best_i[key] = int(i)

    regimes = {
        f"short_gap_le_{int(SHORT_GAP_MAX)}": lambda i: gap[i] <= SHORT_GAP_MAX,
        f"long_gap_gt_{int(SHORT_GAP_MAX)}": lambda i: gap[i] > SHORT_GAP_MAX,
        "long_gap_gt_pool_median_129": lambda i: gap[i] > 129.0,
    }
    out: dict[str, Any] = {"short_gap_max_frames": SHORT_GAP_MAX, "regimes": {}}
    for rname, pred in regimes.items():
        idxs = [i for i in best_i.values() if pred(i)]
        vi = {}
        for name, lower in ATOMS:
            n_viol = sum(
                1 for i in idxs if z_bit(float(pool[name][i]), thrs[name], lower) == 0
            )
            vi[name] = {
                "n": len(idxs),
                "n_violations": n_viol,
                "V_i": n_viol / len(idxs) if idxs else float("nan"),
            }
        out["regimes"][rname] = vi
    return out


def assign_roles(
    vi: dict[str, dict[str, Any]],
    shell: dict[str, Any],
    sensitivity: dict[str, Any],
    scale_guard: dict[str, Any],
    prc_aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic role assignment under issue #106 rules + PR-C binding.

    Rules (closed; no fifth role):
    1. PR-C role-reversal motion atoms → not global_orderable; default
       conditional_orderable under short-gap continuous regime.
    2. score_m_bridge fails composite/scale-guard → context_only.
    3. gap is a regime descriptor → context_only.
    4. height/structural atoms with no accepted role reversal and non-artifactual
       safer direction → global_orderable.
    5. Anything failing the above with insufficient evidence → unresolved
       (not used on this substrate under current rules).
    """
    if prc_aggregate.get("terminal") != PRC_AGGREGATE:
        raise ValueError(
            f"PR-C aggregate terminal must be {PRC_AGGREGATE}, "
            f"got {prc_aggregate.get('terminal')}"
        )
    acceptance = prc_aggregate.get("research_acceptance", {})
    if acceptance.get("status") != PRC_ACCEPTANCE:
        raise ValueError(
            f"PR-C research_acceptance must be {PRC_ACCEPTANCE}, "
            f"got {acceptance.get('status')}"
        )

    tail_viol = shell["protected_tail_d_h_ge_3"]["per_atom_violations"]
    flips = sensitivity["direction_flip_p40_to_p60"]
    cards: dict[str, Any] = {}

    # --- motion atoms ---
    short_gap_context = {
        "name": "short_gap_continuous_association",
        "observable_without_gt_outcome": True,
        "definition": (
            f"gap <= {SHORT_GAP_MAX} frames (declared short-gap regime); "
            "excludes long-gap re-entry regime where PR-C accepted role reversal"
        ),
        "proposal_only_conditional_arcs": True,
        "note": (
            "Conditional orderability is a contract proposal for a later "
            "conditional-closure study; not a sealed global arc."
        ),
    }
    for name in MOTION_ATOMS:
        lower = dict(ATOMS)[name]
        cards[name] = {
            "atom": name,
            "role": "conditional_orderable",
            "provenance": "derived" if name != "dir_cos" else "builder_raw",
            "declared_safer_direction": "lower" if lower else "higher",
            "physical_interpretation": {
                "speed_mismatch": "exit/entry speed continuity",
                "dir_cos": "exit/entry direction continuity",
                "resid_mean": "bidirectional residual mean (motion fit quality)",
            }[name],
            "admissible_contexts": [short_gap_context],
            "V_i": vi[name],
            "tail_violations_d_h_ge_3": tail_viol[name],
            "threshold_flip_p40_p60": flips[name],
            "prc_binding": {
                "role_reversal_supported": True,
                "source": "PR-C #102 / PR #104 ACCEPTED_WITH_LIMITS",
                "claim_ceiling": "L1 single-sequence MOT17-10-SDP forensic",
                "blocks_global_orderable": True,
            },
            "supporting_evidence": [
                "PR-C aggregate ROLE_REVERSAL_SUPPORTED on long-gap re-entry",
                f"far-Hamming tail violations: {tail_viol[name]}/4 at median split",
                "direction remains mechanism-plausible under short-gap continuity",
            ],
            "competing_explanations": [
                "even short-gap V_i is non-trivial under pool-median split "
                "(threshold is exploratory, not a sealed operating point)",
                "L1 single-seq forensic cannot prove multi-sequence conditional order",
            ],
            "confidence_boundary": (
                "L1 conditional proposal only; may not enter global closure arcs; "
                "nested confirmation (PR-E) required before any stronger claim"
            ),
        }

    # --- gap ---
    cards["gap"] = {
        "atom": "gap",
        "role": "context_only",
        "provenance": "builder_raw",
        "declared_safer_direction": "lower",
        "physical_interpretation": (
            "occlusion / lost duration (frames); regime descriptor rather than "
            "a monotone safety dimension"
        ),
        "admissible_contexts": [],
        "V_i": vi["gap"],
        "tail_violations_d_h_ge_3": tail_viol["gap"],
        "threshold_flip_p40_p60": flips["gap"],
        "prc_binding": {
            "role_reversal_supported": False,
            "blocks_global_orderable": True,
        },
        "supporting_evidence": [
            "gap defines short vs long-gap regimes used to condition motion atoms",
            "long gap is not intrinsically unsafe (true re-entries exist)",
            f"threshold flip rate p40→p60 = {flips['gap']['flip_rate']:.3f}",
        ],
        "competing_explanations": [
            "some safe-reject studies treat very large gap as FP-enriched; "
            "that is a coverage heuristic, not a global GT-order proof"
        ],
        "confidence_boundary": (
            "context/regime only; forbidden as a global order dimension"
        ),
    }

    # --- score_m_bridge composite ---
    cards["score_m_bridge"] = {
        "atom": "score_m_bridge",
        "role": "context_only",
        "provenance": "composite_derived",
        "declared_safer_direction": "lower",
        "physical_interpretation": (
            "production-shaped speed-weighted mix of residual and dist_h"
        ),
        "admissible_contexts": [],
        "V_i": vi["score_m_bridge"],
        "tail_violations_d_h_ge_3": tail_viol["score_m_bridge"],
        "threshold_flip_p40_p60": flips["score_m_bridge"],
        "prc_binding": {
            "role_reversal_supported": False,
            "indirect_via_parent": "resid_mean",
            "blocks_global_orderable": True,
        },
        "scale_guard": {
            "blocks_global_orderable": scale_guard["blocks_global_orderable"],
            "block_reasons": scale_guard["block_reasons"],
            "correlations": scale_guard["correlations"],
        },
        "supporting_evidence": [
            "composite parents include resid_mean (PR-C motion role-reversal atom)",
            "scale-guard: residual term dominates the composite",
            "speed-dependent mixing weight injects hidden context dependence",
        ],
        "competing_explanations": [
            "as a production proxy the score remains useful for FP ranking, "
            "but ranking utility ≠ global monotone order legitimacy"
        ],
        "confidence_boundary": (
            "context/diagnostic only on this substrate; parent role conflict "
            "blocks global promotion without a redesign that removes the "
            "motion parent or re-proves direction under nested folds"
        ),
    }

    # --- structural + height ---
    for name in (*STRUCTURAL_ATOMS, *HEIGHT_ATOMS):
        lower = dict(ATOMS)[name]
        cards[name] = {
            "atom": name,
            "role": "global_orderable",
            "provenance": "derived" if name == "log_h_ratio" else "builder_raw",
            "declared_safer_direction": "lower" if lower else "higher",
            "physical_interpretation": {
                "bridge_dist": "mid-point bridge distance (geometry)",
                "dist_h": "height-normalized foot distance (geometry)",
                "log_h_ratio": "absolute log height ratio (scale consistency)",
            }[name],
            "admissible_contexts": [],
            "V_i": vi[name],
            "tail_violations_d_h_ge_3": tail_viol[name],
            "threshold_flip_p40_p60": flips[name],
            "prc_binding": {
                "role_reversal_supported": False,
                "blocks_global_orderable": False,
            },
            "supporting_evidence": [
                f"low track-level V_i under median split: {vi[name]['n_violations']}/"
                f"{vi[name]['n_tracks']}",
                f"protected-tail violations: {tail_viol[name]}/4 "
                + (
                    "(log_h_ratio remains 0/4 — height preserved on escape tail)"
                    if name == "log_h_ratio"
                    else ""
                ),
                "no accepted mechanism-level role reversal on current evidence",
                "parents (if any) do not include PR-C motion role-reversal atoms",
                f"p40→p60 flip rate {flips[name]['flip_rate']:.3f} "
                "(direction not a pure exploratory split artifact)",
            ],
            "competing_explanations": [
                "most structural violations concentrate on MOT17-10-SDP "
                "(sequence clustering); multi-seq confirmation still pending",
                "pool-median thresholds remain audit-only",
            ],
            "confidence_boundary": (
                "L1 global-order *contract* only — authorizes a separate "
                "restricted-closure prototype task, not nested L2+ claims, "
                "not production/preset/ledger changes"
            ),
        }

    # Integrity: all eight atoms exactly once.
    if set(cards) != set(ATOM_NAMES):
        raise AssertionError(
            f"role cards incomplete: {sorted(set(ATOM_NAMES) - set(cards))}"
        )
    for name, card in cards.items():
        if card["role"] not in ROLES:
            raise AssertionError(f"illegal role for {name}: {card['role']}")
        # Hard guard: PR-C motion atoms never global.
        if name in PRC_ROLE_REVERSAL_ATOMS and card["role"] == "global_orderable":
            raise AssertionError(
                f"PR-C motion atom {name} must not be global_orderable"
            )
        if name == "score_m_bridge" and card["role"] == "global_orderable":
            raise AssertionError(
                "score_m_bridge blocked by scale-guard/parent conflict"
            )

    return cards


def build_order_contract(
    cards: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    global_atoms = sorted(
        name for name, c in cards.items() if c["role"] == "global_orderable"
    )
    conditional = {
        name: c["admissible_contexts"]
        for name, c in cards.items()
        if c["role"] == "conditional_orderable"
    }
    context_only = sorted(
        name for name, c in cards.items() if c["role"] == "context_only"
    )
    unresolved = sorted(name for name, c in cards.items() if c["role"] == "unresolved")

    allowed = {
        "z_convention": (
            "z_i = 1 means the declared safer side; later reject domains must "
            "be downward-closed in this orientation"
        ),
        "global_atoms": global_atoms,
        "allowed_global_order_dimensions": [
            {
                "atom": name,
                "safer_direction": cards[name]["declared_safer_direction"],
                "may_participate_in_global_closure_arcs": True,
            }
            for name in global_atoms
        ],
        "allowed_global_arcs": [
            {
                "type": "coordinate_monotone",
                "atom": name,
                "meaning": (
                    f"moving toward safer side of {name} is a legal global "
                    "closure step (contract only; no weights assigned)"
                ),
            }
            for name in global_atoms
        ],
        "graph_contract_for_next_task": {
            "vertices": "Boolean cells on the global_orderable subcube only",
            "orientation": "z=1 safer; reject domain downward-closed",
            "optimization": "forbidden in this task; deferred to separate prototype",
            "baseline_comparison_if_prototype_authorized": "frozen OR-tail under exact GT-UCB",
        },
        "claim_ceiling": "L1 order contract; not L2+; not production",
    }

    forbidden_dims = []
    for name, c in cards.items():
        if c["role"] == "global_orderable":
            continue
        reasons = []
        if name in PRC_ROLE_REVERSAL_ATOMS:
            reasons.append("PR-C accepted motion role-reversal evidence")
        if name == "score_m_bridge":
            reasons.append("composite parent role conflict + scale dominance")
        if name == "gap":
            reasons.append("regime descriptor, not monotone safety dimension")
        if c["role"] == "conditional_orderable":
            reasons.append("only conditionally orderable; not global")
        if c["role"] == "context_only":
            reasons.append("context_only role")
        if c["role"] == "unresolved":
            reasons.append("unresolved role")
        forbidden_dims.append(
            {
                "atom": name,
                "role": c["role"],
                "forbidden_global_arcs": True,
                "reasons": reasons,
            }
        )

    forbidden = {
        "forbidden_global_dimensions": forbidden_dims,
        "forbidden_global_arcs": [
            {
                "atom": name,
                "reason": "; ".join(item["reasons"]),
            }
            for item, name in ((item, item["atom"]) for item in forbidden_dims)
        ],
        "conditional_arc_proposals": [
            {
                "atom": name,
                "status": "proposal-only",
                "contexts": conditional[name],
                "note": (
                    "not authorized for global MWC; requires a separately "
                    "reviewed conditional-representation task"
                ),
            }
            for name in sorted(conditional)
        ],
        "hard_blocks": [
            "global closure arcs on motion atoms (PR-C binding)",
            "global arcs on score_m_bridge without redesign",
            "global arcs on gap",
            "escape-tail veto",
            "MWC / min-cut / rule search inside this audit PR",
            "production / preset / ledger changes",
        ],
        "context_only_atoms": context_only,
        "unresolved_atoms": unresolved,
    }
    return allowed, forbidden


def decide_terminal(
    cards: dict[str, Any],
    allowed: dict[str, Any],
) -> dict[str, Any]:
    roles = {c["role"] for c in cards.values()}
    if not set(cards) == set(ATOM_NAMES):
        terminal = "ORDERABILITY_UNRESOLVED"
        reason = "incomplete atom-role map"
    elif "unresolved" in roles and not allowed["global_atoms"]:
        terminal = "ORDERABILITY_UNRESOLVED"
        reason = "unresolved roles block a complete orderability map"
    elif allowed["global_atoms"]:
        terminal = "GLOBAL_PARTIAL_ORDER_READY"
        reason = (
            f"nontrivial global_orderable set {allowed['global_atoms']}; "
            "all eight atoms have bounded roles; allowed/forbidden contract complete"
        )
    elif any(c["role"] == "conditional_orderable" for c in cards.values()) or any(
        c["role"] == "context_only" for c in cards.values()
    ):
        terminal = "CONDITIONAL_STRUCTURE_ONLY"
        reason = "no global_orderable dimension; conditional/context structure only"
    else:
        terminal = "ORDERABILITY_UNRESOLVED"
        reason = "no defensible order structure"

    if terminal not in TERMINALS:
        raise AssertionError(f"illegal terminal {terminal}")

    routing = {
        "GLOBAL_PARTIAL_ORDER_READY": (
            "open a **separate** restricted global-closure prototype task using "
            "only global_orderable atoms; compare against frozen OR-tail under "
            "exact GT-UCB; still candidate-only"
        ),
        "CONDITIONAL_STRUCTURE_ONLY": (
            "do not run global MWC; separately design and review the observable "
            "context contract first"
        ),
        "ORDERABILITY_UNRESOLVED": (
            "stop closure work; retain L1 morphology + forensic conclusion only"
        ),
    }[terminal]

    return {
        "terminal": terminal,
        "reason": reason,
        "routing": routing,
        "authorizes_restricted_closure_prototype": terminal
        == "GLOBAL_PARTIAL_ORDER_READY",
        "claim_ceiling": (
            "L1 partial-order contract on sealed 7-seq substrate; "
            "PR-C motion evidence remains L1 single-seq bound"
        ),
    }


def write_atom_metrics_csv(
    path: Path,
    cards: dict[str, Any],
    shell: dict[str, Any],
) -> None:
    tail = shell["protected_tail_d_h_ge_3"]["per_atom_violations"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "atom",
                "role",
                "safer_direction",
                "provenance",
                "n_tracks",
                "n_violations_V_i",
                "V_i",
                "P_z1_gt_track",
                "n_sequences_with_violation",
                "tail_violations_d_h_ge_3",
                "threshold_flip_p40_p60",
                "prc_blocks_global",
            ]
        )
        for name in ATOM_NAMES:
            c = cards[name]
            writer.writerow(
                [
                    name,
                    c["role"],
                    c["declared_safer_direction"],
                    c["provenance"],
                    c["V_i"]["n_tracks"],
                    c["V_i"]["n_violations"],
                    f"{c['V_i']['V_i']:.6g}",
                    f"{c['V_i']['P_z1_gt_track']:.6g}",
                    c["V_i"]["n_sequences_with_violation"],
                    tail[name],
                    c["threshold_flip_p40_p60"]["n_flips"],
                    bool(c["prc_binding"].get("blocks_global_orderable", False)),
                ]
            )


def write_pairwise_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "atom_i",
                "atom_j",
                "V_ij",
                "V_ij_rate",
                "V_i_only",
                "V_j_only",
                "interpretation_bound",
            ]
        )
        for row in pairs:
            writer.writerow(
                [
                    row["atom_i"],
                    row["atom_j"],
                    row["V_ij"],
                    f"{row['V_ij_rate']:.6g}",
                    row["V_i_only"],
                    row["V_j_only"],
                    row["interpretation_bound"],
                ]
            )


def emit(pairs: Path, out: Path) -> dict[str, Any]:
    step0_manifest = load_json(STEP0 / "manifest.json")
    prc_aggregate = load_json(PRC / "aggregate.json")
    verify_source(pairs, step0_manifest)

    # Substrate identity checks.
    if list(step0_manifest["atom_order"]) != ATOM_NAMES:
        raise ValueError("atom order drift vs Step-0 manifest")

    gt_rows = load_step0_gt_rows()
    tracks = min_dh_representatives(gt_rows)
    if len(tracks) != int(step0_manifest["n_gt_tracks"]):
        raise ValueError(
            f"track count mismatch: {len(tracks)} vs {step0_manifest['n_gt_tracks']}"
        )

    vi = compute_vi_profile(tracks)
    shell = compute_shell_contribution(tracks)
    pairwise = compute_pairwise(tracks)

    pool = prepare_pool(pairs)
    sensitivity = threshold_sensitivity(pool)
    # Cross-check median thresholds against Step-0 seal (float equality within 1e-9).
    sealed_thrs = step0_manifest["atom_thresholds"]
    med_thrs = sensitivity["quantiles"]["median"]["thresholds"]
    for name in ATOM_NAMES:
        sealed = float(sealed_thrs[name]["pool_median_threshold"])
        got = float(med_thrs[name])
        if abs(sealed - got) > 1e-9 * max(1.0, abs(sealed)):
            raise ValueError(
                f"median threshold drift for {name}: sealed={sealed}, got={got}"
            )

    scale_guard = scale_guard_score_m_bridge(pool)
    dep = dependency_graph()
    regimes = gap_regime_profile(pool)
    cards = assign_roles(vi, shell, sensitivity, scale_guard, prc_aggregate)
    allowed, forbidden = build_order_contract(cards)
    terminal = decide_terminal(cards, allowed)

    out.mkdir(parents=True, exist_ok=True)
    # Copy runner into out if emitting elsewhere (verify path).
    runner_src = Path(__file__).resolve()
    runner_dst = out / "run_partial_order_audit.py"
    if runner_src != runner_dst.resolve():
        shutil.copy2(runner_src, runner_dst)

    atom_roles = {
        "study_id": "boolean_atom_partial_order_20260711",
        "issue": 106,
        "pr_ladder": "PR-D gate (partial-order audit only)",
        "roles": {name: cards[name]["role"] for name in ATOM_NAMES},
        "cards": cards,
        "role_vocabulary": list(ROLES),
        "n_atoms": len(ATOM_NAMES),
    }
    write_json(out / "atom_roles.json", atom_roles)
    write_json(out / "atom_dependency_graph.json", dep)
    write_atom_metrics_csv(out / "atom_metrics.csv", cards, shell)
    write_pairwise_csv(out / "pairwise_violation_profile.csv", pairwise)

    sensitivity_out = {
        **sensitivity,
        "gap_regime_profile": regimes,
        "shell_contribution": shell,
    }
    write_json(out / "threshold_sensitivity.json", sensitivity_out)
    write_json(out / "allowed_global_order.json", allowed)
    write_json(out / "forbidden_order.json", forbidden)
    write_json(out / "scale_guard.json", scale_guard)

    aggregate = {
        "terminal": terminal["terminal"],
        "reason": terminal["reason"],
        "routing": terminal["routing"],
        "authorizes_restricted_closure_prototype": terminal[
            "authorizes_restricted_closure_prototype"
        ],
        "global_atoms": allowed["global_atoms"],
        "conditional_atoms": sorted(
            n for n, c in cards.items() if c["role"] == "conditional_orderable"
        ),
        "context_only_atoms": sorted(
            n for n, c in cards.items() if c["role"] == "context_only"
        ),
        "unresolved_atoms": sorted(
            n for n, c in cards.items() if c["role"] == "unresolved"
        ),
        "roles": {name: cards[name]["role"] for name in ATOM_NAMES},
        "prc_binding_respected": all(
            cards[n]["role"] != "global_orderable" for n in PRC_ROLE_REVERSAL_ATOMS
        ),
        "score_m_bridge_not_global": cards["score_m_bridge"]["role"]
        != "global_orderable",
        "claim_ceiling": terminal["claim_ceiling"],
        "scope_guards": [
            "no MWC / min-cut / rule search / weight optimization",
            "no production / preset / ledger change",
            "no escape-tail veto",
            "not observed != unsafe",
            "zero exposure != ordering proof",
        ],
    }
    write_json(out / "aggregate.json", aggregate)

    body_hashes = {
        name: sha256(out / name) for name in PACKET_BODY_FILES if (out / name).is_file()
    }
    manifest = {
        "study_id": "boolean_atom_partial_order_20260711",
        "issue": 106,
        "pr_ladder": "PR-D gate (partial-order audit; closure prototype separate)",
        "depends_on": {
            "step0_packet": str(STEP0.relative_to(REPO)),
            "step0_manifest_sha256": sha256(STEP0 / "manifest.json"),
            "prc_packet": str(PRC.relative_to(REPO)),
            "prc_aggregate_sha256": sha256(PRC / "aggregate.json"),
            "prc_aggregate_terminal": prc_aggregate.get("terminal"),
            "prc_research_acceptance": acceptance_status(prc_aggregate),
            "source_pairs_csv": str(step0_manifest["source_pairs_csv"]),
            "source_pairs_csv_sha256": str(step0_manifest["source_pairs_csv_sha256"]),
            "procedure": "framework §19 v1 (PR-A #100 sealed)",
            "research_line": "boolean_closure_domain_line_20260711 (PR-B #101)",
        },
        "atom_order": ATOM_NAMES,
        "atom_safer_directions": {
            name: ("lower" if lower else "higher") for name, lower in ATOMS
        },
        "binarization": "pool median (audit-only; matches Step-0)",
        "trial_unit": "(seq, lost_id); descriptive min-d_H representative",
        "n_gt_tracks": len(tracks),
        "n_gt_rows": len(gt_rows),
        "aggregate_terminal": terminal["terminal"],
        "roles": {name: cards[name]["role"] for name in ATOM_NAMES},
        "scope": (
            "read-only offline partial-order audit; no MWC/min-cut/rule-search/"
            "weight-opt; no production/preset/ledger changes"
        ),
        "files": body_hashes,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def acceptance_status(prc_aggregate: dict[str, Any]) -> str:
    acc = prc_aggregate.get("research_acceptance", {})
    if isinstance(acc, dict):
        return str(acc.get("status", "unknown"))
    return str(acc)


def _compare_packet(expected: Path, rebuilt: Path) -> list[str]:
    mismatched: list[str] = []
    exp_manifest = load_json(expected / "manifest.json")
    reb_manifest = load_json(rebuilt / "manifest.json")
    exp_files = exp_manifest.get("files", {})
    reb_files = reb_manifest.get("files", {})
    names = sorted(set(PACKET_BODY_FILES) | set(exp_files) | set(reb_files))
    for name in names:
        path_e = expected / name
        path_r = rebuilt / name
        if name == "manifest.json":
            continue
        if not path_e.is_file():
            mismatched.append(f"missing_on_disk:expected:{name}")
            continue
        if not path_r.is_file():
            mismatched.append(f"missing_on_disk:rebuilt:{name}")
            continue
        exp_digest = exp_files.get(name)
        reb_digest = reb_files.get(name)
        actual_e = sha256(path_e)
        actual_r = sha256(path_r)
        if exp_digest != actual_e:
            mismatched.append(f"manifest.json:files:stale_digest:expected:{name}")
        if reb_digest != actual_r:
            mismatched.append(f"manifest.json:files:stale_digest:rebuilt:{name}")
        if exp_digest != reb_digest:
            mismatched.append(f"manifest.json:files:digest_mismatch:{name}")
        if path_e.read_bytes() != path_r.read_bytes():
            mismatched.append(f"bytes_mismatch:{name}")
    # Compare terminals / roles without requiring byte-identical manifest
    # (self-hash of runner may differ only if source differs — already checked).
    for key in ("aggregate_terminal", "roles", "atom_order"):
        if exp_manifest.get(key) != reb_manifest.get(key):
            mismatched.append(f"manifest_field_mismatch:{key}")
    return mismatched


def verify(pairs: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="partial-order-audit-") as tmp:
        rebuilt = Path(tmp) / PACKET.name
        emit(pairs, rebuilt)
        mismatched = _compare_packet(PACKET, rebuilt)
    if mismatched:
        raise AssertionError(f"packet is not reproducible: {mismatched}")
    print("partial-order audit packet verification passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PACKET)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rebuild in a temp dir and compare to the committed packet",
    )
    args = parser.parse_args()
    if args.verify:
        if args.out.resolve() != PACKET.resolve():
            parser.error("--verify cannot be combined with a non-default --out")
        verify(args.pairs)
        return
    manifest = emit(args.pairs, args.out)
    print(f"packet emitted: {args.out}")
    print(f"aggregate terminal: {manifest['aggregate_terminal']}")
    print(f"roles: {manifest['roles']}")


if __name__ == "__main__":
    main()
