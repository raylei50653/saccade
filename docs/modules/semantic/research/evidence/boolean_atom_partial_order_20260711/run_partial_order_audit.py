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
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
# Pure structural / height leaves (no motion parents in the builder formula).
STRUCTURAL_LEAVES = ("dist_h",)
HEIGHT_ATOMS = ("log_h_ratio",)
# Motion-extrapolation composite: mid-point bridge uses endpoint velocities × gap.
MOTION_EXTRAPOLATION_COMPOSITES = frozenset({"bridge_dist"})
# Weighted / multi-parent composites with explicit scale-guard.
WEIGHTED_COMPOSITES = frozenset({"score_m_bridge"})
REGIME_ATOMS = frozenset({"gap"})

# PR-C binding (issue #102 / PR #104, ACCEPTED_WITH_LIMITS).
# These three atoms carry accepted long-gap re-entry role-reversal evidence.
PRC_ROLE_REVERSAL_ATOMS = frozenset(MOTION_ATOMS)
PRC_AGGREGATE = "ROLE_REVERSAL_SUPPORTED"
PRC_ACCEPTANCE = "ACCEPTED_WITH_LIMITS"

# Role assignment is research judgment; statistics are descriptive only.
# Executable guards below can only *block* global promotion, not invent it.
ROLE_ASSIGNMENT_MODE = (
    "research_judgment_with_executable_guards; "
    "V_i / shell / threshold metrics are descriptive and do not alone "
    "control role assignment"
)

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
    return cast(dict[str, np.ndarray], pool)


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
        "unit_scale_compatibility": {
            "compatible": True,
            "detail": (
                "All mixands are height-normalized dimensionless quantities "
                "from the pair builder: fwd_resid/h_ref, bwd_resid/h_ref, "
                "dist_h/h_ref (and bridge_dist/h_ref). There is no px-vs-h "
                "unit incompatibility in score_m_bridge. Residual dominance "
                "is an empirical scale/weight effect within a common unit, "
                "not a unit mismatch."
            ),
            "unit_incompatibility_is_block_reason": False,
        },
        "findings": findings,
        "blocks_global_orderable": blocks_global,
        "block_reasons": [
            "parent role conflict: resid_mean is PR-C motion role-reversal atom",
            "residual dominance: composite tracks resid_mean more than dist_h "
            "(empirical term-fraction / correlation, within a shared unit)",
            "hidden context dependence via speed-dependent mixing weight w",
        ],
    }


def dependency_graph() -> dict[str, Any]:
    """Static provenance DAG for frozen atoms (no optimization edges)."""
    # bridge_dist formula (pair builder / relink._midpoint_bridge_dist):
    #   m_l = x_l + v_l * gap/2
    #   m_c = x_c - v_c * gap/2
    #   bridge_dist = ||m_l - m_c|| / h_ref
    nodes: dict[str, dict[str, Any]] = {
        "score_m_bridge": {
            "kind": "weighted_composite",
            "parents": ["resid_mean", "dist_h", "lost_exit_speed"],
            "transform": "speed-weighted linear mix of height-normalized terms",
            "motion_derived": True,
        },
        "bridge_dist": {
            "kind": "motion_extrapolation_composite",
            "parents": [
                "lost_foot_xy",
                "cand_foot_xy",
                "lost_exit_velocity",
                "cand_entry_velocity",
                "gap",
                "h_ref",
            ],
            "transform": (
                "m_l=x_l+v_l*gap/2; m_c=x_c-v_c*gap/2; bridge_dist=||m_l-m_c||/h_ref"
            ),
            "formula_source": (
                "src/saccade/perception/eval/relink.py::_midpoint_bridge_dist "
                "(pair-builder column of the same form)"
            ),
            "motion_derived": True,
            "note": (
                "Not a parentless geometry leaf. Explicitly depends on endpoint "
                "velocities and gap; constant-velocity mid-gap extrapolation."
            ),
        },
        "dist_h": {
            "kind": "builder_raw",
            "parents": ["lost_foot_xy", "cand_foot_xy", "h_ref"],
            "transform": "||x_c - x_l|| / h_ref (endpoint geometry only)",
            "motion_derived": False,
        },
        "log_h_ratio": {
            "kind": "derived",
            "parents": ["h_lost_raw", "h_cand_raw"],
            "transform": "abs(log(h_cand/h_lost))",
            "motion_derived": False,
        },
        "resid_mean": {
            "kind": "derived",
            "parents": ["fwd_resid", "bwd_resid"],
            "transform": "0.5*(fwd+bwd) (each residual is /h_ref in the builder)",
            "motion_derived": True,
        },
        "dir_cos": {
            "kind": "builder_raw",
            "parents": ["lost_exit_velocity", "cand_entry_velocity"],
            "transform": "direction cosine of exit/entry motion",
            "motion_derived": True,
        },
        "speed_mismatch": {
            "kind": "derived",
            "parents": ["lost_exit_speed", "cand_entry_speed"],
            "transform": "abs(exit_speed - entry_speed) (speeds are /h_ref)",
            "motion_derived": True,
        },
        "gap": {
            "kind": "builder_raw",
            "parents": ["lost_last_frame", "cand_first_frame"],
            "transform": "cand_first_frame - lost_last_frame",
            "motion_derived": False,
            "regime_descriptor": True,
        },
        "h_lost_raw": {
            "kind": "builder_raw",
            "parents": [],
            "transform": None,
            "motion_derived": False,
        },
        "h_cand_raw": {
            "kind": "builder_raw",
            "parents": [],
            "transform": None,
            "motion_derived": False,
        },
        "h_ref": {
            "kind": "builder_raw",
            "parents": ["h_lost_raw", "h_cand_raw"],
            "transform": "max(0.5*(h_lost+h_cand), 1)",
            "motion_derived": False,
        },
        "fwd_resid": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "forward residual / h_ref (dimensionless)",
            "motion_derived": True,
        },
        "bwd_resid": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "backward residual / h_ref (dimensionless)",
            "motion_derived": True,
        },
        "lost_exit_speed": {
            "kind": "builder_raw",
            "parents": ["lost_exit_velocity", "h_ref"],
            "transform": "||v_l|| / h_ref",
            "motion_derived": True,
        },
        "cand_entry_speed": {
            "kind": "builder_raw",
            "parents": ["cand_entry_velocity", "h_ref"],
            "transform": "||v_c|| / h_ref",
            "motion_derived": True,
        },
        "lost_exit_velocity": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "endpoint velocity of lost track",
            "motion_derived": True,
        },
        "cand_entry_velocity": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "endpoint velocity of candidate track",
            "motion_derived": True,
        },
        "lost_foot_xy": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "lost endpoint foot position",
            "motion_derived": False,
        },
        "cand_foot_xy": {
            "kind": "builder_raw",
            "parents": [],
            "transform": "candidate endpoint foot position",
            "motion_derived": False,
        },
        "lost_last_frame": {
            "kind": "builder_raw",
            "parents": [],
            "transform": None,
            "motion_derived": False,
        },
        "cand_first_frame": {
            "kind": "builder_raw",
            "parents": [],
            "transform": None,
            "motion_derived": False,
        },
    }
    edges: list[dict[str, str]] = []
    for child, meta in nodes.items():
        parents = cast(list[str], meta["parents"])
        for parent in parents:
            edges.append({"source": parent, "target": child, "relation": "depends_on"})
    return {
        "description": (
            "Dependency DAG for composite/derived atoms. Edges are provenance "
            "only — not closure arcs and not weights."
        ),
        "frozen_atoms": ATOM_NAMES,
        "nodes": nodes,
        "edges": edges,
        "motion_derived_frozen_atoms": sorted(
            name
            for name in ATOM_NAMES
            if bool(nodes[name].get("motion_derived", False))
        ),
    }


def global_admissibility_check(atom: str, dep: dict[str, Any]) -> dict[str, Any]:
    """Executable guards that may only *block* global_orderable promotion.

    These checks do not assign roles by themselves. Research judgment still
    chooses among the non-blocked roles (conditional / context_only /
    unresolved / global when all guards pass).
    """
    nodes = cast(dict[str, dict[str, Any]], dep["nodes"])
    meta = nodes[atom]
    reasons: list[str] = []
    if atom in PRC_ROLE_REVERSAL_ATOMS:
        reasons.append("PR-C accepted motion role-reversal atom")
    if atom in REGIME_ATOMS:
        reasons.append("regime descriptor, not a monotone safety dimension")
    if atom in MOTION_EXTRAPOLATION_COMPOSITES:
        reasons.append(
            "motion-extrapolation composite (endpoint velocities × gap); "
            "not a pure structural leaf"
        )
    if atom in WEIGHTED_COMPOSITES:
        reasons.append(
            "weighted composite with motion parent and/or regime-dependent mix"
        )
    if bool(meta.get("motion_derived", False)) and atom not in (
        *STRUCTURAL_LEAVES,
        *HEIGHT_ATOMS,
    ):
        # Belt-and-suspenders: any motion-derived frozen atom is blocked unless
        # it is an explicitly enumerated pure structural/height leaf.
        if atom not in STRUCTURAL_LEAVES and atom not in HEIGHT_ATOMS:
            if "motion-extrapolation" not in " ".join(reasons):
                if (
                    atom not in PRC_ROLE_REVERSAL_ATOMS
                    and atom not in WEIGHTED_COMPOSITES
                ):
                    reasons.append("motion_derived=true in dependency DAG")
    # Parent-level conflict: direct parents that are PR-C motion atoms.
    parent_names = cast(list[str], meta.get("parents", []))
    prc_parents = sorted(set(parent_names) & set(PRC_ROLE_REVERSAL_ATOMS))
    if prc_parents:
        reasons.append(
            f"direct parent role conflict with PR-C motion atoms: {prc_parents}"
        )
    # Parent-level: any velocity/residual parents make silent global promotion illegal.
    motion_parent_markers = {
        "lost_exit_velocity",
        "cand_entry_velocity",
        "lost_exit_speed",
        "cand_entry_speed",
        "fwd_resid",
        "bwd_resid",
        "resid_mean",
        "dir_cos",
        "speed_mismatch",
    }
    motion_parents = sorted(set(parent_names) & motion_parent_markers)
    if motion_parents and atom not in STRUCTURAL_LEAVES and atom not in HEIGHT_ATOMS:
        tag = f"motion parents in formula: {motion_parents}"
        if tag not in reasons and not any("motion" in r for r in reasons):
            reasons.append(tag)
    admissible = len(reasons) == 0
    return {
        "atom": atom,
        "global_admissible": admissible,
        "block_reasons": reasons,
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

    def short_gap(i: int) -> bool:
        return bool(gap[i] <= SHORT_GAP_MAX)

    def long_gap(i: int) -> bool:
        return bool(gap[i] > SHORT_GAP_MAX)

    def long_gap_median(i: int) -> bool:
        return bool(gap[i] > 129.0)

    regimes: dict[str, Callable[[int], bool]] = {
        f"short_gap_le_{int(SHORT_GAP_MAX)}": short_gap,
        f"long_gap_gt_{int(SHORT_GAP_MAX)}": long_gap,
        "long_gap_gt_pool_median_129": long_gap_median,
    }
    out: dict[str, Any] = {"short_gap_max_frames": SHORT_GAP_MAX, "regimes": {}}
    for rname, pred in regimes.items():
        idxs = [i for i in best_i.values() if pred(i)]
        vi: dict[str, Any] = {}
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
    dep: dict[str, Any],
) -> dict[str, Any]:
    """Research-judgment role assignment under issue #106 + PR-C binding.

    Self-declaration
    ----------------
    Role assignment is **research judgment**, not a fitted classifier.
    Morphology statistics (V_i, shells, threshold flips) are **descriptive
    only** and do not alone control the role. Executable
    ``global_admissibility_check`` may only *block* ``global_orderable``.

    Closed role rules
    -----------------
    1. PR-C role-reversal motion atoms → not global; default
       ``conditional_orderable`` under short-gap continuous regime.
    2. ``score_m_bridge`` fails weighted-composite / parent-conflict guard →
       ``context_only`` (units are compatible; block is role/dominance/w).
    3. ``bridge_dist`` is a motion-extrapolation composite (velocities × gap)
       → not global; ``conditional_orderable`` under short-gap CV regime
       (no independent multi-seq mechanism evidence for global promotion).
    4. ``gap`` is a regime descriptor → ``context_only``.
    5. Pure height / structural leaves that pass admissibility
       (``log_h_ratio``, ``dist_h``) → ``global_orderable``.
    6. No fifth role.
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
    admissibility = {name: global_admissibility_check(name, dep) for name in ATOM_NAMES}
    cards: dict[str, Any] = {}

    short_gap_context = {
        "name": "short_gap_continuous_association",
        "observable_without_gt_outcome": True,
        "definition": (
            f"gap <= {SHORT_GAP_MAX} frames (declared short-gap regime); "
            "excludes long-gap re-entry regime where PR-C accepted role "
            "reversal and constant-velocity mid-gap extrapolation breaks"
        ),
        "proposal_only_conditional_arcs": True,
        "note": (
            "Conditional orderability is a contract proposal for a later "
            "conditional-closure study; not a sealed global arc."
        ),
    }

    # --- motion atoms (PR-C binding) ---
    for name in MOTION_ATOMS:
        lower = dict(ATOMS)[name]
        cards[name] = {
            "atom": name,
            "role": "conditional_orderable",
            "role_assignment": "research_judgment",
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
            "global_admissibility": admissibility[name],
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
        "role_assignment": "research_judgment",
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
        "global_admissibility": admissibility["gap"],
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

    # --- score_m_bridge weighted composite ---
    cards["score_m_bridge"] = {
        "atom": "score_m_bridge",
        "role": "context_only",
        "role_assignment": "research_judgment",
        "provenance": "weighted_composite",
        "declared_safer_direction": "lower",
        "physical_interpretation": (
            "production-shaped speed-weighted mix of height-normalized "
            "residual mean and dist_h"
        ),
        "admissible_contexts": [],
        "V_i": vi["score_m_bridge"],
        "tail_violations_d_h_ge_3": tail_viol["score_m_bridge"],
        "threshold_flip_p40_p60": flips["score_m_bridge"],
        "global_admissibility": admissibility["score_m_bridge"],
        "prc_binding": {
            "role_reversal_supported": False,
            "indirect_via_parent": "resid_mean",
            "blocks_global_orderable": True,
        },
        "scale_guard": {
            "blocks_global_orderable": scale_guard["blocks_global_orderable"],
            "block_reasons": scale_guard["block_reasons"],
            "correlations": scale_guard["correlations"],
            "unit_scale_compatibility": scale_guard["unit_scale_compatibility"],
        },
        "supporting_evidence": [
            "composite parents include resid_mean (PR-C motion role-reversal atom)",
            "residual dominance within a shared height-normalized unit "
            "(not a unit-mismatch claim)",
            "speed-dependent mixing weight injects hidden context dependence",
        ],
        "competing_explanations": [
            "as a production proxy the score remains useful for FP ranking, "
            "but ranking utility ≠ global monotone order legitimacy"
        ],
        "confidence_boundary": (
            "context/diagnostic only on this substrate; parent role conflict "
            "and residual dominance block global promotion without a redesign "
            "that removes the motion parent or re-proves direction under nested folds"
        ),
    }

    # --- bridge_dist: motion-extrapolation composite (NOT pure geometry) ---
    cards["bridge_dist"] = {
        "atom": "bridge_dist",
        "role": "conditional_orderable",
        "role_assignment": "research_judgment",
        "provenance": "motion_extrapolation_composite",
        "declared_safer_direction": "lower",
        "physical_interpretation": (
            "constant-velocity mid-gap extrapolation distance: "
            "m_l=x_l+v_l*gap/2, m_c=x_c-v_c*gap/2, bridge_dist=||m_l-m_c||/h_ref. "
            "Depends on endpoint velocities, gap, geometry, and height normalization "
            "— not a parentless structural leaf."
        ),
        "admissible_contexts": [short_gap_context],
        "V_i": vi["bridge_dist"],
        "tail_violations_d_h_ge_3": tail_viol["bridge_dist"],
        "threshold_flip_p40_p60": flips["bridge_dist"],
        "global_admissibility": admissibility["bridge_dist"],
        "prc_binding": {
            "role_reversal_supported": False,
            "related_to_prc_motion_reversal": True,
            "relation": (
                "Long-gap re-entry (PR-C TRUE_LONG_GAP_REENTRY) is exactly the "
                "regime where constant-velocity mid-gap extrapolation is least "
                "defensible; velocity parents share the motion substrate that "
                "carries accepted role-reversal evidence. No independent "
                "mechanism evidence authorizes global promotion of bridge_dist."
            ),
            "blocks_global_orderable": True,
        },
        "supporting_evidence": [
            "builder formula explicitly multiplies endpoint velocities by gap/2",
            "dependency DAG lists lost/cand exit-entry velocities + gap + h_ref",
            "global_admissibility_check blocks motion-extrapolation composites",
            "short-gap CV regime remains a plausible conditional context",
        ],
        "competing_explanations": [
            "low V_i alone cannot reclassify a motion-derived composite as "
            "global_orderable (statistics are descriptive only)",
            "tail co-violations with geometry do not prove pure structural role",
        ],
        "confidence_boundary": (
            "L1 conditional proposal only; forbidden as a global order dimension "
            "until independent multi-seq mechanism evidence is reviewed"
        ),
    }

    # --- pure structural / height leaves that pass global admissibility ---
    for name in (*STRUCTURAL_LEAVES, *HEIGHT_ATOMS):
        if not admissibility[name]["global_admissible"]:
            raise AssertionError(
                f"expected {name} to pass global admissibility, got "
                f"{admissibility[name]['block_reasons']}"
            )
        lower = dict(ATOMS)[name]
        cards[name] = {
            "atom": name,
            "role": "global_orderable",
            "role_assignment": "research_judgment",
            "provenance": "derived" if name == "log_h_ratio" else "builder_raw",
            "declared_safer_direction": "lower" if lower else "higher",
            "physical_interpretation": {
                "dist_h": "height-normalized endpoint foot distance (geometry only)",
                "log_h_ratio": "absolute log height ratio (scale consistency)",
            }[name],
            "admissible_contexts": [],
            "V_i": vi[name],
            "tail_violations_d_h_ge_3": tail_viol[name],
            "threshold_flip_p40_p60": flips[name],
            "global_admissibility": admissibility[name],
            "prc_binding": {
                "role_reversal_supported": False,
                "blocks_global_orderable": False,
            },
            "supporting_evidence": [
                f"global_admissibility_check passed for {name}",
                f"low track-level V_i under median split: {vi[name]['n_violations']}/"
                f"{vi[name]['n_tracks']}",
                f"protected-tail violations: {tail_viol[name]}/4 "
                + (
                    "(log_h_ratio remains 0/4 — height preserved on escape tail)"
                    if name == "log_h_ratio"
                    else ""
                ),
                "no accepted mechanism-level role reversal on current evidence",
                "no motion parents / not a motion-extrapolation composite",
                f"p40→p60 flip rate {flips[name]['flip_rate']:.3f} "
                "(direction not a pure exploratory split artifact)",
            ],
            "competing_explanations": [
                "sparse violations concentrate on MOT17-10-SDP "
                "(sequence clustering); multi-seq confirmation still pending",
                "pool-median thresholds remain audit-only",
            ],
            "confidence_boundary": (
                "L1 global-order *contract* only — authorizes a separate "
                "restricted-closure prototype task, not nested L2+ claims, "
                "not production/preset/ledger changes"
            ),
        }

    # Integrity: all eight atoms exactly once + executable guards.
    if set(cards) != set(ATOM_NAMES):
        raise AssertionError(
            f"role cards incomplete: {sorted(set(ATOM_NAMES) - set(cards))}"
        )
    for name, card in cards.items():
        if card["role"] not in ROLES:
            raise AssertionError(f"illegal role for {name}: {card['role']}")
        if (
            card["role"] == "global_orderable"
            and not admissibility[name]["global_admissible"]
        ):
            raise AssertionError(
                f"{name} marked global_orderable but admissibility blocked: "
                f"{admissibility[name]['block_reasons']}"
            )
        if name in PRC_ROLE_REVERSAL_ATOMS and card["role"] == "global_orderable":
            raise AssertionError(
                f"PR-C motion atom {name} must not be global_orderable"
            )
        if (
            name in MOTION_EXTRAPOLATION_COMPOSITES
            and card["role"] == "global_orderable"
        ):
            raise AssertionError(
                f"motion-extrapolation composite {name} must not be global_orderable"
            )
        if name in WEIGHTED_COMPOSITES and card["role"] == "global_orderable":
            raise AssertionError(
                f"weighted composite {name} must not be global_orderable"
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
        if name in MOTION_EXTRAPOLATION_COMPOSITES:
            reasons.append(
                "motion-extrapolation composite (endpoint velocities × gap); "
                "related to PR-C long-gap re-entry regime"
            )
        if name == "score_m_bridge":
            reasons.append(
                "weighted composite parent role conflict + residual dominance "
                "(units are height-normalized / compatible)"
            )
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
            "global arcs on bridge_dist (motion-extrapolation composite)",
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
            "operational only: after research acceptance, open a **separate** "
            "restricted global-closure prototype using only accepted "
            "global_orderable atoms; compare against frozen OR-tail under exact "
            "GT-UCB; still candidate-only. While research_acceptance is PENDING, "
            "restricted-closure remains BLOCKED."
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
    cards = assign_roles(vi, shell, sensitivity, scale_guard, prc_aggregate, dep)
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
        "role_assignment_mode": ROLE_ASSIGNMENT_MODE,
        "statistics_role": "descriptive_only",
        "roles": {name: cards[name]["role"] for name in ATOM_NAMES},
        "cards": cards,
        "role_vocabulary": list(ROLES),
        "n_atoms": len(ATOM_NAMES),
        "executable_global_guards": {
            name: cards[name]["global_admissibility"] for name in ATOM_NAMES
        },
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
        "terminal_status": "provisional_operational",
        "reason": terminal["reason"],
        "routing": terminal["routing"],
        # Operational true when terminal is GLOBAL_PARTIAL_ORDER_READY; research
        # acceptance may still leave restricted-closure BLOCKED until stamped.
        "authorizes_restricted_closure_prototype": terminal[
            "authorizes_restricted_closure_prototype"
        ],
        "research_acceptance": {
            "status": "PENDING",
            "pr": 107,
            "issue": 106,
            "note": (
                "Operational terminal is provisional. Research-owner review "
                "found initial bridge_dist provenance misclassification; "
                "revised map is not yet accepted. Restricted-closure remains "
                "BLOCKED until research acceptance is recorded on PR #107."
            ),
            "initial_operational_terminal": "GLOBAL_PARTIAL_ORDER_READY",
            "initial_global_atoms": [
                "bridge_dist",
                "dist_h",
                "log_h_ratio",
            ],
            "restricted_closure": "BLOCKED_until_research_acceptance",
        },
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
        "bridge_dist_not_global": cards["bridge_dist"]["role"] != "global_orderable",
        "role_assignment_mode": ROLE_ASSIGNMENT_MODE,
        "claim_ceiling": terminal["claim_ceiling"],
        "scope_guards": [
            "no MWC / min-cut / rule search / weight optimization",
            "no production / preset / ledger change",
            "no escape-tail veto",
            "not observed != unsafe",
            "zero exposure != ordering proof",
            "motion-derived composites cannot silent-global-promote",
            "provisional operational terminal != research acceptance",
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
