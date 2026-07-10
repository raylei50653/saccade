"""M-B1.5 Stage 2 Q4.5 — structured threshold-combination atlas.

Descriptive atlas of single-atom and pairwise AND/OR regions on the locked
selected∧resolved D_online decision cohort.

Does **not**:
  - pick a single best threshold as a rule
  - unrestricted Boolean mining / 3+ atom combos
  - learned classifiers
  - hook policy / e2e effect / production preset change
  - promote observed GT_hurt==0 points to safe rules
  - reopen Q4 thr-chase on inseparable absolute tails as policy

Q4 weak AUC only blocks promoting *singleton frozen tails* to production thr;
it does **not** forbid threshold/Boolean structure as a data-analysis method.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from saccade.perception.eval.d_online_stage2 import (
    CANDIDATE_UNIVERSE_ID,
    SUBSTRATE_ID,
    write_csv,
    write_json,
    write_parquet,
    _sha256_file,
)
from saccade.perception.eval.d_online_stage2_q4 import (
    _f,
    _i,
    load_d_online_events,
    lock_q4_cohort,
    reconcile_q4,
)
from saccade.perception.eval.portable_or_tail import ORDERED_SIGNALS

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

TAXONOMY_VERSION = "stage2_q45_atlas_v1"
Q1Q3_STUDY_ID = "m_b1_5_stage2_q1q3_20260710"
Q4_STUDY_ID = "m_b1_5_stage2_q4_20260710"
EXPECTED_PRIMARY_N = 87
EXPECTED_PRIMARY_NEG = 23
EXPECTED_PRIMARY_POS = 64

# Declared rank/quantile lattice for pairwise complete enumeration.
# Fractions of the empirical CDF on the primary pool (inclusive endpoints).
PAIRWISE_QUANTILE_LATTICE: tuple[float, ...] = tuple(
    i / 20.0 for i in range(0, 21)
)  # 0.00, 0.05, ..., 1.00

# Single-atom lattice: all observed unique values on primary (complete family).
SINGLE_LATTICE_KIND = "primary_unique_boundaries"
PAIRWISE_LATTICE_KIND = "primary_quantile_lattice_q05"

DIRECTIONS = ("high_tail", "low_tail")  # x >= t , x <= t
COMBINATORS = ("AND", "OR")

TERMINAL_A = "structured_region_supported"
TERMINAL_B = "isolated_safe_points_only"
TERMINAL_C = "conditional_enrichment_without_safe_region"
TERMINAL_D = "no_useful_conditional_structure"

# Enrichment: precision / base_rate; require min support for claim
MIN_SUPPORT_FOR_ENRICHMENT = 2
MIN_ENRICHMENT_USEFUL = 1.25  # 25% above base negative rate
MIN_SEQS_FOR_REGION = 2
MAX_SEQ_SHARE_NON_DOMINANT = 0.5


class Stage2Q45Error(ValueError):
    """Fail-closed Q4.5 atlas error."""


# ---------------------------------------------------------------------------
# Frame provenance check
# ---------------------------------------------------------------------------


def check_frame_column_provenance(
    events: Sequence[Mapping[str, Any]],
    *,
    stage1_study_dir: Path | None = None,
) -> dict[str, Any]:
    """Document exact semantics of the ``frame`` column on D_online events.

    Does **not** authorize temporal-family conclusions.
    """
    frames = [int(r.get("frame", -999)) for r in events]
    uniq = sorted(set(frames))
    # Consistency with event_id / join_key
    event_id_frames = []
    join_key_frames = []
    mismatches = []
    for r in events:
        f = int(r.get("frame", -999))
        eid = str(r.get("event_id", ""))
        jk = str(r.get("join_key", ""))
        # event_id pattern: SEQ:f{frame}:c...
        ef = None
        if ":f" in eid:
            try:
                ef = int(eid.split(":f", 1)[1].split(":", 1)[0])
            except ValueError:
                ef = None
        event_id_frames.append(ef)
        jf = None
        parts = jk.split("|")
        if len(parts) >= 2:
            try:
                jf = int(parts[1])
            except ValueError:
                jf = None
        join_key_frames.append(jf)
        if ef is not None and ef != f:
            mismatches.append(
                {"kind": "event_id", "event_id": eid, "frame": f, "parsed": ef}
            )
        if jf is not None and jf != f:
            mismatches.append(
                {"kind": "join_key", "join_key": jk, "frame": f, "parsed": jf}
            )

    # Cross-check: MOT birth frames for sample track ids (if A1 MOT available)
    mot_cross: list[dict[str, Any]] = []
    if stage1_study_dir is not None:
        a1 = Path(stage1_study_dir) / "e2e_A1_hook_off"
        if a1.is_dir():
            by_seq: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for r in events[:40]:
                by_seq[str(r["sequence"])].append(r)
            for seq, rows in list(by_seq.items())[:3]:
                mot_path = a1 / f"{seq}.txt"
                if not mot_path.is_file():
                    continue
                # load MOT: frame,id,...
                tid_frames: dict[int, list[int]] = defaultdict(list)
                with mot_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split(",")
                        if len(parts) < 2:
                            continue
                        try:
                            fr, tid = int(float(parts[0])), int(float(parts[1]))
                        except ValueError:
                            continue
                        tid_frames[tid].append(fr)
                for r in rows[:5]:
                    for role, key in (
                        ("cand", "cand_track_id"),
                        ("lost", "lost_track_id"),
                    ):
                        tid = _i(r.get(key, -1))
                        frs = tid_frames.get(tid, [])
                        mot_cross.append(
                            {
                                "sequence": seq,
                                "event_id": r.get("event_id"),
                                "audit_frame": int(r.get("frame", -1)),
                                "role": role,
                                "track_id": tid,
                                "mot_frame_min": min(frs) if frs else None,
                                "mot_frame_max": max(frs) if frs else None,
                                "mot_n_boxes": len(frs),
                                "audit_frame_equals_mot_min": (
                                    bool(frs) and int(r.get("frame", -1)) == min(frs)
                                ),
                            }
                        )

    n_mot_disagree = sum(
        1
        for m in mot_cross
        if m.get("mot_frame_min") is not None and m["audit_frame"] != m["mot_frame_min"]
    )

    # Source-code provenance (static)
    code_provenance = {
        "writer": (
            "src/tracking/tracker_gpu.cu relink_bidir_propose_kernel: "
            "ev.frame = frame_idx"
        ),
        "frame_idx_source": (
            "Host counter portable_audit_frame_++ passed into propose kernel "
            "(comment: 'Host frame counter for B-audit')"
        ),
        "reset_sites": [
            "ensure_portable_audit_ring_",
            "free_portable_audit_ring_",
            "clear_portable_or_tail_audit",
        ],
        "python_enrichment": (
            "portable_or_tail.enrich_online_audit_events copies e['frame'] into "
            "event_id (f{frame}) and join_key second field"
        ),
        "not_absolute_mot_frame": True,
        "not_gap_length": True,
        "not_fixed_probe_index_by_design": True,
        "observed_limitation": (
            "All Stage1 B-audit rows currently carry frame==4; counter is "
            "0-based host propose-invocation index, but observed constancy "
            "indicates the field is not a reliable multi-frame MOT timeline "
            "in this export (possible capture/staleness or single-write pattern). "
            "Cross-check vs A1 MOT track spans disagrees with absolute MOT frame."
        ),
    }

    is_absolute_mot = False  # established by code + MOT cross-check
    return {
        "check_id": "frame_column_provenance_q45",
        "column": "frame",
        "n_events": len(events),
        "unique_values": uniq,
        "n_unique": len(uniq),
        "all_equal_to_4": uniq == [4],
        "event_id_consistent": all(
            ef is None or ef == f for ef, f in zip(event_id_frames, frames)
        ),
        "join_key_consistent": all(
            jf is None or jf == f for jf, f in zip(join_key_frames, frames)
        ),
        "n_parse_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:10],
        "semantic": {
            "kind": "host_audit_propose_invocation_counter",
            "is_absolute_mot_frame": is_absolute_mot,
            "is_relative_event_frame": False,
            "is_gap_length": False,
            "is_fixed_probe_index": False,
            "is_export_limitation": True,
            "intended_meaning": (
                "0-based counter of bidirectional propose-kernel invocations "
                "since audit ring clear (portable_audit_frame_++)"
            ),
            "observed_meaning_limitation": (
                "Constant value 4 across all 244 D_online rows; "
                "must not be treated as absolute MOT frame index"
            ),
        },
        "code_provenance": code_provenance,
        "mot_crosscheck_sample": mot_cross[:15],
        "mot_crosscheck_n_disagree_with_mot_min": n_mot_disagree,
        "mot_crosscheck_n_compared": len(mot_cross),
        "conclusions": {
            "may_claim_absolute_mot_frame": False,
            "may_claim_temporal_information_unavailable_from_frame_alone": False,
            "observation_limitation_only": True,
            "affects_q45_threshold_atlas_mainline": False,
            "note": (
                "Do not write 'temporal information unavailable' solely because "
                "frame==4. The field is not absolute MOT frame. This is an "
                "instrumentation/export limitation for temporal studies; "
                "Q4.5 threshold-combination atlas does not depend on it."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Cohort lock
# ---------------------------------------------------------------------------


def lock_q45_cohort(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Same primary lock as Q4."""
    locked = lock_q4_cohort(rows)
    s = locked["summary"]
    if int(s["n_primary"]) != EXPECTED_PRIMARY_N:
        raise Stage2Q45Error(
            f"primary n={s['n_primary']} != locked {EXPECTED_PRIMARY_N}"
        )
    if int(s["n_primary_negative"]) != EXPECTED_PRIMARY_NEG:
        raise Stage2Q45Error(f"neg={s['n_primary_negative']} != {EXPECTED_PRIMARY_NEG}")
    if int(s["n_primary_positive_protect"]) != EXPECTED_PRIMARY_POS:
        raise Stage2Q45Error(
            f"pos={s['n_primary_positive_protect']} != {EXPECTED_PRIMARY_POS}"
        )
    return locked


# ---------------------------------------------------------------------------
# Threshold registry
# ---------------------------------------------------------------------------


def _sorted_unique(x: np.ndarray) -> np.ndarray:
    v = x[np.isfinite(x)]
    if len(v) == 0:
        return np.asarray([], dtype=float)
    return np.unique(v)


def _quantile_thresholds(x: np.ndarray, qs: Sequence[float]) -> list[float]:
    v = x[np.isfinite(x)]
    if len(v) == 0:
        return []
    out = []
    for q in qs:
        t = float(np.quantile(v, q, method="linear"))
        out.append(t)
    # dedupe preserving order
    seen: set[float] = set()
    deduped = []
    for t in out:
        # use round-trip key for float stability
        key = round(t, 12)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def build_threshold_registry(
    primary: Sequence[Mapping[str, Any]],
    *,
    signals: Sequence[str] = ORDERED_SIGNALS,
    secondary_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Register complete lattices for single-atom and pairwise atlases."""
    y = np.asarray([int(r["q4_y"]) for r in primary], dtype=int)
    seqs = np.asarray([str(r["sequence"]) for r in primary], dtype=object)
    n = len(primary)
    base_neg_rate = float(np.mean(y == 1)) if n else float("nan")

    all_feats = list(signals) + list(secondary_features or [])
    matrices: dict[str, np.ndarray] = {}
    for name in all_feats:
        matrices[name] = np.asarray([_f(r.get(name)) for r in primary], dtype=float)

    single_entries: list[dict[str, Any]] = []
    pairwise_entries: list[dict[str, Any]] = []

    for name in all_feats:
        x = matrices[name]
        uniq = _sorted_unique(x)
        q_thrs = _quantile_thresholds(x, PAIRWISE_QUANTILE_LATTICE)
        is_secondary = name not in signals
        for thr_idx, t in enumerate(uniq.tolist()):
            for direction in DIRECTIONS:
                single_entries.append(
                    {
                        "atom_id": f"S::{name}::{direction}::u{thr_idx}",
                        "scope": "single_atom",
                        "lattice_kind": SINGLE_LATTICE_KIND,
                        "feature": name,
                        "is_secondary_feature": int(is_secondary),
                        "direction": direction,
                        "thr_index": thr_idx,
                        "thr_value": float(t),
                        "n_unique_on_primary": int(len(uniq)),
                    }
                )
        for thr_idx, t in enumerate(q_thrs):
            for direction in DIRECTIONS:
                pairwise_entries.append(
                    {
                        "atom_id": f"P::{name}::{direction}::q{thr_idx}",
                        "scope": "pairwise_atom",
                        "lattice_kind": PAIRWISE_LATTICE_KIND,
                        "feature": name,
                        "is_secondary_feature": int(is_secondary),
                        "direction": direction,
                        "thr_index": thr_idx,
                        "thr_value": float(t),
                        "quantile_lattice_point": PAIRWISE_QUANTILE_LATTICE[
                            min(thr_idx, len(PAIRWISE_QUANTILE_LATTICE) - 1)
                        ]
                        if thr_idx < len(PAIRWISE_QUANTILE_LATTICE)
                        else None,
                        "n_quantile_levels_registered": len(q_thrs),
                    }
                )

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "cohort_n": n,
        "n_negative": int(np.sum(y == 1)),
        "n_positive_protect": int(np.sum(y == 0)),
        "base_negative_rate": base_neg_rate,
        "signals_primary": list(signals),
        "secondary_features": list(secondary_features or []),
        "single_lattice_kind": SINGLE_LATTICE_KIND,
        "pairwise_lattice_kind": PAIRWISE_LATTICE_KIND,
        "pairwise_quantile_lattice": list(PAIRWISE_QUANTILE_LATTICE),
        "directions": list(DIRECTIONS),
        "combinators": list(COMBINATORS),
        "n_single_atoms": len(single_entries),
        "n_pairwise_atoms": len(pairwise_entries),
        "single_atoms": single_entries,
        "pairwise_atoms": pairwise_entries,
        "feature_matrices_meta": {
            name: {
                "n_finite": int(np.sum(np.isfinite(matrices[name]))),
                "n_unique": int(len(_sorted_unique(matrices[name]))),
                "min": float(np.nanmin(matrices[name]))
                if np.isfinite(matrices[name]).any()
                else float("nan"),
                "max": float(np.nanmax(matrices[name]))
                if np.isfinite(matrices[name]).any()
                else float("nan"),
            }
            for name in all_feats
        },
        # keep arrays out of JSON registry; returned separately for eval
        "_matrices": matrices,
        "_y": y,
        "_sequences": seqs,
    }


def atom_mask(x: np.ndarray, direction: str, thr: float) -> np.ndarray:
    if direction == "high_tail":
        return np.isfinite(x) & (x >= thr)
    if direction == "low_tail":
        return np.isfinite(x) & (x <= thr)
    raise Stage2Q45Error(f"unknown direction {direction}")


# ---------------------------------------------------------------------------
# Region metrics
# ---------------------------------------------------------------------------


def region_metrics(
    mask: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
    *,
    base_neg_rate: float,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    n = len(y)
    support = int(mask.sum())
    n_neg = int(np.sum(mask & (y == 1)))
    n_gt = int(np.sum(mask & (y == 0)))
    precision = float(n_neg / support) if support > 0 else float("nan")
    enrichment = (
        float(precision / base_neg_rate)
        if support > 0 and base_neg_rate > 0
        else float("nan")
    )
    # per-sequence
    seq_neg: dict[str, int] = defaultdict(int)
    seq_gt: dict[str, int] = defaultdict(int)
    seq_sup: dict[str, int] = defaultdict(int)
    for i in np.where(mask)[0]:
        s = str(sequences[i])
        seq_sup[s] += 1
        if y[i] == 1:
            seq_neg[s] += 1
        else:
            seq_gt[s] += 1
    n_seq_with_support = len(seq_sup)
    n_seq_with_neg = sum(1 for v in seq_neg.values() if v > 0)
    max_seq_share = max(seq_sup.values()) / support if support > 0 else float("nan")
    max_neg_seq_share = max(seq_neg.values()) / n_neg if n_neg > 0 else float("nan")
    # LOO: drop each sequence present in cohort
    loo_rows = []
    all_seqs = sorted(set(str(s) for s in sequences))
    for hold in all_seqs:
        m = sequences != hold
        mm = mask[m]
        yy = y[m]
        sup = int(mm.sum())
        nn = int(np.sum(mm & (yy == 1)))
        ng = int(np.sum(mm & (yy == 0)))
        loo_rows.append(
            {
                "hold_out_sequence": hold,
                "support": sup,
                "n_neg": nn,
                "n_gt": ng,
                "gt_hurt": ng,
                "productive_safe": int(ng == 0 and nn > 0),
            }
        )
    loo_max_gt = max((r["gt_hurt"] for r in loo_rows), default=0)
    loo_all_safe = all(r["gt_hurt"] == 0 for r in loo_rows) if loo_rows else False
    loo_any_productive = any(r["productive_safe"] == 1 for r in loo_rows)

    observed_safe_point = support > 0 and n_gt == 0  # may be empty
    productive_safe_point = n_gt == 0 and n_neg > 0

    return {
        "support": support,
        "coverage": float(support / n) if n else float("nan"),
        "n_neg_captured": n_neg,
        "n_gt_captured": n_gt,
        "gt_hurt": n_gt,
        "gt_hurt_rate": float(n_gt / int(np.sum(y == 0)))
        if np.sum(y == 0)
        else float("nan"),
        "neg_capture_rate": float(n_neg / int(np.sum(y == 1)))
        if np.sum(y == 1)
        else float("nan"),
        "precision": precision,
        "enrichment": enrichment,
        "n_sequences_with_support": n_seq_with_support,
        "n_sequences_with_neg": n_seq_with_neg,
        "max_sequence_share": max_seq_share,
        "max_neg_sequence_share": max_neg_seq_share,
        "single_seq_neg_dominance": bool(
            n_neg > 0
            and math.isfinite(max_neg_seq_share)
            and max_neg_seq_share > MAX_SEQ_SHARE_NON_DOMINANT
        ),
        "observed_safe_point": bool(observed_safe_point and support > 0),
        "productive_safe_point": bool(productive_safe_point),
        # Explicit: not a rule
        "not_a_safe_rule": True,
        "loo_max_gt_hurt": int(loo_max_gt),
        "loo_all_gt_hurt_zero": bool(loo_all_safe),
        "loo_any_productive_safe": bool(loo_any_productive),
        "per_sequence_support": dict(seq_sup),
        "per_sequence_neg": dict(seq_neg),
        "per_sequence_gt": dict(seq_gt),
        "loo": loo_rows,
    }


# ---------------------------------------------------------------------------
# Single-atom atlas
# ---------------------------------------------------------------------------


def build_single_atom_atlas(
    registry: Mapping[str, Any],
    *,
    primary_signals_only: bool = False,
) -> list[dict[str, Any]]:
    matrices: dict[str, np.ndarray] = registry["_matrices"]  # type: ignore
    y: np.ndarray = registry["_y"]  # type: ignore
    sequences: np.ndarray = registry["_sequences"]  # type: ignore
    base = float(registry["base_negative_rate"])
    rows: list[dict[str, Any]] = []

    # Group atoms by (feature, direction) for neighbor continuity
    by_fd: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for atom in registry["single_atoms"]:
        if primary_signals_only and int(atom.get("is_secondary_feature", 0)):
            continue
        by_fd[(atom["feature"], atom["direction"])].append(atom)

    for (feature, direction), atoms in by_fd.items():
        atoms_sorted = sorted(atoms, key=lambda a: int(a["thr_index"]))
        x = matrices[feature]
        masks = []
        metrics_list = []
        for atom in atoms_sorted:
            m = atom_mask(x, direction, float(atom["thr_value"]))
            met = region_metrics(m, y, sequences, base_neg_rate=base)
            masks.append(m)
            metrics_list.append(met)

        for i, atom in enumerate(atoms_sorted):
            met = metrics_list[i]
            # neighbor continuity: adjacent thr_index
            neighbor_gt = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(atoms_sorted):
                    neighbor_gt.append(metrics_list[j]["gt_hurt"])
                    # subset relation
            # subset vs adjacent: for high_tail, higher thr ⇒ subset of lower thr
            subset_of_prev = None
            prev_subset_of_this = None
            if i > 0:
                subset_of_prev = bool(np.all(~masks[i] | masks[i - 1]))
                prev_subset_of_this = bool(np.all(~masks[i - 1] | masks[i]))
            subset_of_next = None
            next_subset_of_this = None
            if i + 1 < len(masks):
                subset_of_next = bool(np.all(~masks[i] | masks[i + 1]))
                next_subset_of_this = bool(np.all(~masks[i + 1] | masks[i]))

            # neighborhood GT continuity: neighbors also productive-safe?
            adj_productive_safe = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(metrics_list):
                    adj_productive_safe.append(
                        bool(metrics_list[j]["productive_safe_point"])
                    )

            rows.append(
                {
                    "atom_id": atom["atom_id"],
                    "feature": feature,
                    "is_secondary_feature": int(atom.get("is_secondary_feature", 0)),
                    "direction": direction,
                    "thr_index": int(atom["thr_index"]),
                    "thr_value": float(atom["thr_value"]),
                    "lattice_kind": atom["lattice_kind"],
                    "support": met["support"],
                    "coverage": met["coverage"],
                    "n_neg_captured": met["n_neg_captured"],
                    "n_gt_captured": met["n_gt_captured"],
                    "gt_hurt": met["gt_hurt"],
                    "gt_hurt_rate": met["gt_hurt_rate"],
                    "neg_capture_rate": met["neg_capture_rate"],
                    "precision": met["precision"],
                    "enrichment": met["enrichment"],
                    "n_sequences_with_support": met["n_sequences_with_support"],
                    "n_sequences_with_neg": met["n_sequences_with_neg"],
                    "max_sequence_share": met["max_sequence_share"],
                    "max_neg_sequence_share": met["max_neg_sequence_share"],
                    "single_seq_neg_dominance": int(met["single_seq_neg_dominance"]),
                    "observed_safe_point": int(met["observed_safe_point"]),
                    "productive_safe_point": int(met["productive_safe_point"]),
                    "not_a_safe_rule": 1,
                    "loo_max_gt_hurt": met["loo_max_gt_hurt"],
                    "loo_all_gt_hurt_zero": int(met["loo_all_gt_hurt_zero"]),
                    "loo_any_productive_safe": int(met["loo_any_productive_safe"]),
                    "n_adjacent_neighbors": len(adj_productive_safe),
                    "n_adjacent_also_productive_safe": sum(adj_productive_safe),
                    "subset_of_prev_thr": (
                        int(subset_of_prev) if subset_of_prev is not None else None
                    ),
                    "prev_subset_of_this": (
                        int(prev_subset_of_this)
                        if prev_subset_of_this is not None
                        else None
                    ),
                    "subset_of_next_thr": (
                        int(subset_of_next) if subset_of_next is not None else None
                    ),
                    "next_subset_of_this": (
                        int(next_subset_of_this)
                        if next_subset_of_this is not None
                        else None
                    ),
                    "neighbor_gt_hurts": "|".join(str(g) for g in neighbor_gt),
                    "necessity_neg_coverage": met["neg_capture_rate"],
                    "observed_sufficiency_gt_contamination": met["gt_hurt"],
                    "per_sequence_neg_json": json.dumps(
                        met["per_sequence_neg"], sort_keys=True
                    ),
                    "per_sequence_gt_json": json.dumps(
                        met["per_sequence_gt"], sort_keys=True
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Pairwise atlas
# ---------------------------------------------------------------------------


def build_pairwise_atlas(
    registry: Mapping[str, Any],
    *,
    combinator: str,
    primary_signals_only: bool = True,
) -> list[dict[str, Any]]:
    """Complete enumeration of registered pairwise lattice (AND or OR)."""
    if combinator not in COMBINATORS:
        raise Stage2Q45Error(f"bad combinator {combinator}")
    matrices: dict[str, np.ndarray] = registry["_matrices"]  # type: ignore
    y: np.ndarray = registry["_y"]  # type: ignore
    sequences: np.ndarray = registry["_sequences"]  # type: ignore
    base = float(registry["base_negative_rate"])

    atoms = [
        a
        for a in registry["pairwise_atoms"]
        if (not primary_signals_only) or not int(a.get("is_secondary_feature", 0))
    ]
    # group by feature for pairing across different features only
    by_feat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        by_feat[a["feature"]].append(a)

    features = sorted(by_feat.keys())
    # Precompute masks for each pairwise atom
    atom_masks: dict[str, np.ndarray] = {}
    for a in atoms:
        atom_masks[a["atom_id"]] = atom_mask(
            matrices[a["feature"]], a["direction"], float(a["thr_value"])
        )

    rows: list[dict[str, Any]] = []
    seen_sig: set[tuple[Any, ...]] = set()

    for f1, f2 in itertools.combinations(features, 2):
        for a in by_feat[f1]:
            for b in by_feat[f2]:
                m1 = atom_masks[a["atom_id"]]
                m2 = atom_masks[b["atom_id"]]
                if combinator == "AND":
                    m = m1 & m2
                else:
                    m = m1 | m2
                # signature for dedupe of empty/identical sets
                # use hash of bitmask
                sig = (combinator, m.tobytes())
                # Still keep all registered cells; mark semantic_duplicate
                is_dup = sig in seen_sig
                seen_sig.add(sig)

                met = region_metrics(m, y, sequences, base_neg_rate=base)
                rows.append(
                    {
                        "combo_id": (f"{combinator}::{a['atom_id']}::{b['atom_id']}"),
                        "combinator": combinator,
                        "atom_a_id": a["atom_id"],
                        "atom_b_id": b["atom_id"],
                        "feature_a": a["feature"],
                        "feature_b": b["feature"],
                        "direction_a": a["direction"],
                        "direction_b": b["direction"],
                        "thr_index_a": int(a["thr_index"]),
                        "thr_index_b": int(b["thr_index"]),
                        "thr_value_a": float(a["thr_value"]),
                        "thr_value_b": float(b["thr_value"]),
                        "lattice_kind": PAIRWISE_LATTICE_KIND,
                        "semantic_duplicate_mask": int(is_dup),
                        "empty_region": int(met["support"] == 0),
                        "support": met["support"],
                        "coverage": met["coverage"],
                        "n_neg_captured": met["n_neg_captured"],
                        "n_gt_captured": met["n_gt_captured"],
                        "gt_hurt": met["gt_hurt"],
                        "gt_hurt_rate": met["gt_hurt_rate"],
                        "neg_capture_rate": met["neg_capture_rate"],
                        "precision": met["precision"],
                        "enrichment": met["enrichment"],
                        "n_sequences_with_support": met["n_sequences_with_support"],
                        "n_sequences_with_neg": met["n_sequences_with_neg"],
                        "max_sequence_share": met["max_sequence_share"],
                        "max_neg_sequence_share": met["max_neg_sequence_share"],
                        "single_seq_neg_dominance": int(
                            met["single_seq_neg_dominance"]
                        ),
                        "observed_safe_point": int(met["observed_safe_point"]),
                        "productive_safe_point": int(met["productive_safe_point"]),
                        "not_a_safe_rule": 1,
                        "loo_max_gt_hurt": met["loo_max_gt_hurt"],
                        "loo_all_gt_hurt_zero": int(met["loo_all_gt_hurt_zero"]),
                        "loo_any_productive_safe": int(met["loo_any_productive_safe"]),
                        "necessity_neg_coverage": met["neg_capture_rate"],
                        "observed_sufficiency_gt_contamination": met["gt_hurt"],
                        "per_sequence_neg_json": json.dumps(
                            met["per_sequence_neg"], sort_keys=True
                        ),
                        "per_sequence_gt_json": json.dumps(
                            met["per_sequence_gt"], sort_keys=True
                        ),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Stability classification for productive-safe points
# ---------------------------------------------------------------------------


def classify_region_stability(
    atom_rows: Sequence[Mapping[str, Any]],
    pairwise_and: Sequence[Mapping[str, Any]],
    pairwise_or: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Classify observed productive-safe points by neighborhood + LOO."""
    out: list[dict[str, Any]] = []

    # Index single atoms by (feature, direction, thr_index)
    single_idx: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for r in atom_rows:
        if int(r.get("is_secondary_feature", 0)):
            continue
        single_idx[(str(r["feature"]), str(r["direction"]), int(r["thr_index"]))] = r

    def _stab_label(
        *,
        productive: bool,
        n_adj_also: int,
        n_adj: int,
        loo_all_safe: bool,
        loo_max_gt: int,
        n_seq_neg: int,
        single_dom: bool,
    ) -> str:
        if not productive:
            return "not_productive_safe"
        if n_seq_neg < MIN_SEQS_FOR_REGION or single_dom:
            # still classify geometric thickness
            if n_adj_also == 0:
                return "isolated_safe_point"
            if n_adj_also < n_adj:
                return "thin_safe_edge"
            if loo_all_safe and loo_max_gt == 0:
                return "loo_stable_region_but_seq_thin"
            return "locally_stable_region_but_seq_thin"
        if n_adj_also == 0:
            return "isolated_safe_point"
        if n_adj_also < max(n_adj, 1):
            # partial neighborhood
            if loo_all_safe:
                return "thin_safe_edge"
            return "thin_safe_edge"
        # full local neighborhood productive-safe
        if loo_all_safe and loo_max_gt == 0:
            return "loo_stable_region"
        return "locally_stable_region"

    for r in atom_rows:
        if not int(r.get("productive_safe_point", 0)):
            continue
        if int(r.get("is_secondary_feature", 0)):
            continue
        feat, direction, ti = (
            str(r["feature"]),
            str(r["direction"]),
            int(r["thr_index"]),
        )
        adj = []
        for dti in (-1, 1):
            nb = single_idx.get((feat, direction, ti + dti))
            if nb is not None:
                adj.append(nb)
        n_adj = len(adj)
        n_adj_also = sum(1 for a in adj if int(a.get("productive_safe_point", 0)))
        label = _stab_label(
            productive=True,
            n_adj_also=n_adj_also,
            n_adj=n_adj,
            loo_all_safe=bool(int(r.get("loo_all_gt_hurt_zero", 0))),
            loo_max_gt=int(r.get("loo_max_gt_hurt", 99)),
            n_seq_neg=int(r.get("n_sequences_with_neg", 0)),
            single_dom=bool(int(r.get("single_seq_neg_dominance", 0))),
        )
        out.append(
            {
                "region_id": r["atom_id"],
                "kind": "single_atom",
                "feature_a": feat,
                "feature_b": "",
                "direction_a": direction,
                "direction_b": "",
                "thr_index_a": ti,
                "thr_index_b": "",
                "thr_value_a": r["thr_value"],
                "thr_value_b": "",
                "support": r["support"],
                "n_neg_captured": r["n_neg_captured"],
                "gt_hurt": r["gt_hurt"],
                "n_sequences_with_neg": r["n_sequences_with_neg"],
                "max_neg_sequence_share": r["max_neg_sequence_share"],
                "loo_max_gt_hurt": r["loo_max_gt_hurt"],
                "loo_all_gt_hurt_zero": r["loo_all_gt_hurt_zero"],
                "n_adjacent_neighbors": n_adj,
                "n_adjacent_also_productive_safe": n_adj_also,
                "stability_class": label,
                "is_region_candidate": int(
                    label in ("loo_stable_region", "locally_stable_region")
                ),
                "not_a_safe_rule": 1,
            }
        )

    # Pairwise: neighborhood = ±1 thr_index on either atom (4-neighborhood in lattice)
    def _pair_index(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple, Mapping[str, Any]]:
        idx = {}
        for r in rows:
            key = (
                r["feature_a"],
                r["direction_a"],
                int(r["thr_index_a"]),
                r["feature_b"],
                r["direction_b"],
                int(r["thr_index_b"]),
            )
            idx[key] = r
        return idx

    for comb_name, rows in (("AND", pairwise_and), ("OR", pairwise_or)):
        idx = _pair_index(rows)
        for r in rows:
            if not int(r.get("productive_safe_point", 0)):
                continue
            if int(r.get("semantic_duplicate_mask", 0)) and comb_name == "AND":
                # still evaluate uniques; duplicates marked
                pass
            key = (
                r["feature_a"],
                r["direction_a"],
                int(r["thr_index_a"]),
                r["feature_b"],
                r["direction_b"],
                int(r["thr_index_b"]),
            )
            neighbors = []
            fa, da, ia, fb, db, ib = key
            for dia, dib in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = idx.get((fa, da, ia + dia, fb, db, ib + dib))
                if nb is not None:
                    neighbors.append(nb)
            n_adj = len(neighbors)
            n_adj_also = sum(
                1 for a in neighbors if int(a.get("productive_safe_point", 0))
            )
            label = _stab_label(
                productive=True,
                n_adj_also=n_adj_also,
                n_adj=n_adj,
                loo_all_safe=bool(int(r.get("loo_all_gt_hurt_zero", 0))),
                loo_max_gt=int(r.get("loo_max_gt_hurt", 99)),
                n_seq_neg=int(r.get("n_sequences_with_neg", 0)),
                single_dom=bool(int(r.get("single_seq_neg_dominance", 0))),
            )
            out.append(
                {
                    "region_id": r["combo_id"],
                    "kind": f"pairwise_{comb_name}",
                    "feature_a": r["feature_a"],
                    "feature_b": r["feature_b"],
                    "direction_a": r["direction_a"],
                    "direction_b": r["direction_b"],
                    "thr_index_a": r["thr_index_a"],
                    "thr_index_b": r["thr_index_b"],
                    "thr_value_a": r["thr_value_a"],
                    "thr_value_b": r["thr_value_b"],
                    "support": r["support"],
                    "n_neg_captured": r["n_neg_captured"],
                    "gt_hurt": r["gt_hurt"],
                    "n_sequences_with_neg": r["n_sequences_with_neg"],
                    "max_neg_sequence_share": r["max_neg_sequence_share"],
                    "loo_max_gt_hurt": r["loo_max_gt_hurt"],
                    "loo_all_gt_hurt_zero": r["loo_all_gt_hurt_zero"],
                    "n_adjacent_neighbors": n_adj,
                    "n_adjacent_also_productive_safe": n_adj_also,
                    "stability_class": label,
                    "is_region_candidate": int(
                        label in ("loo_stable_region", "locally_stable_region")
                    ),
                    "not_a_safe_rule": 1,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def build_pareto_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    """Pareto over multi-objective region quality (full atlas, not top-k only).

    Objectives (direction):
      gt_hurt ↓, n_neg_captured ↑, n_sequences_with_neg ↑,
      max_sequence_share ↓, loo_all_gt_hurt_zero ↑, loo_max_gt_hurt ↓

    Implementation: O(n log n) sort + linear scan after collapsing exact
    objective-ties, with a final O(f^2) pass on the candidate front (f ≪ n).
    """
    candidates: list[Mapping[str, Any]] = []
    for r in rows:
        if int(r.get("support", 0) or 0) <= 0:
            continue
        if int(r.get("semantic_duplicate_mask", 0)):
            continue
        candidates.append(r)

    if not candidates:
        return []

    def obj_key(r: Mapping[str, Any]) -> tuple:
        return (
            int(r["gt_hurt"]),
            -int(r["n_neg_captured"]),
            -int(r["n_sequences_with_neg"]),
            float(r.get("max_sequence_share") or 1.0),
            -int(r.get("loo_all_gt_hurt_zero", 0)),
            int(r.get("loo_max_gt_hurt", 99)),
        )

    def dominates_tuple(a: tuple, b: tuple) -> bool:
        """a dominates b on the obj_key encoding (lower is better for all)."""
        better_or_eq = all(x <= y for x, y in zip(a, b))
        strictly = any(x < y for x, y in zip(a, b))
        return better_or_eq and strictly

    # Keep one representative per identical objective vector
    best_by_obj: dict[tuple, Mapping[str, Any]] = {}
    for r in candidates:
        k = obj_key(r)
        if k not in best_by_obj:
            best_by_obj[k] = r
    uniq = list(best_by_obj.items())  # (obj_tuple, row)

    # Sort by first objective then others; greedy 1D front candidates
    uniq.sort(key=lambda t: t[0])
    prelim: list[tuple[tuple, Mapping[str, Any]]] = []
    for k, r in uniq:
        if any(dominates_tuple(pk, k) for pk, _ in prelim):
            continue
        # remove previously kept that this dominates
        prelim = [(pk, pr) for pk, pr in prelim if not dominates_tuple(k, pk)]
        prelim.append((k, r))

    # Exact front among prelim (small)
    frontier_rows: list[Mapping[str, Any]] = []
    for i, (ki, ri) in enumerate(prelim):
        if any(dominates_tuple(kj, ki) for j, (kj, _) in enumerate(prelim) if i != j):
            continue
        frontier_rows.append(ri)

    frontier_sorted = sorted(frontier_rows, key=obj_key)
    out = []
    for r in frontier_sorted:
        rid = r.get("atom_id") or r.get("combo_id") or ""
        out.append(
            {
                "kind": kind,
                "region_id": rid,
                "gt_hurt": r["gt_hurt"],
                "n_neg_captured": r["n_neg_captured"],
                "coverage": r["coverage"],
                "support": r["support"],
                "n_sequences_with_neg": r["n_sequences_with_neg"],
                "max_sequence_share": r["max_sequence_share"],
                "loo_all_gt_hurt_zero": r.get("loo_all_gt_hurt_zero"),
                "loo_max_gt_hurt": r.get("loo_max_gt_hurt"),
                "enrichment": r.get("enrichment"),
                "precision": r.get("precision"),
                "productive_safe_point": r.get("productive_safe_point"),
                "feature_a": r.get("feature") or r.get("feature_a"),
                "feature_b": r.get("feature_b", ""),
                "direction_a": r.get("direction") or r.get("direction_a"),
                "direction_b": r.get("direction_b", ""),
                "combinator": r.get("combinator", "single"),
                "not_a_safe_rule": 1,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Terminal classification
# ---------------------------------------------------------------------------


def classify_q45_terminal(
    *,
    stability_rows: Sequence[Mapping[str, Any]],
    atom_rows: Sequence[Mapping[str, Any]],
    pairwise_and: Sequence[Mapping[str, Any]],
    pairwise_or: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    blocked = [
        "threshold_as_production_rule_not_authorized",
        "boolean_as_hook_policy_not_authorized",
        "e2e_effect_claim_not_authorized",
        "production_promotion_blocked",
        "observed_safe_point_is_not_safe_rule",
        "unrestricted_boolean_mining_forbidden",
        "three_plus_atom_combos_forbidden",
    ]

    primary_atoms = [r for r in atom_rows if not int(r.get("is_secondary_feature", 0))]
    all_primary_regions = primary_atoms + list(pairwise_and) + list(pairwise_or)

    region_candidates = [
        r
        for r in stability_rows
        if int(r.get("is_region_candidate", 0))
        and r.get("stability_class")
        in (
            "loo_stable_region",
            "locally_stable_region",
        )
    ]
    loo_stable = [
        r for r in region_candidates if r.get("stability_class") == "loo_stable_region"
    ]

    # A: multi-seq, neighborhood thickness, LOO holds, low/zero GT
    if loo_stable:
        return {
            "stage2_q45_terminal": TERMINAL_A,
            "terminal_letter": "A",
            "reason": (
                f"{len(loo_stable)} LOO-stable productive-safe region(s) with "
                "local thickness; authorize formal restricted safe-region "
                "candidate validation (still not production rule)"
            ),
            "claims_blocked": blocked,
            "claims_allowed": [
                "formal_restricted_safe_region_candidate_validation_authorized",
            ],
            "next_authorized_step": (
                "formal restricted safe-region candidate validation on "
                "declared LOO-stable regions only"
            ),
            "n_region_candidates": len(region_candidates),
            "n_loo_stable_regions": len(loo_stable),
            "supporting_region_ids": [r["region_id"] for r in loo_stable[:20]],
            "production_preset": "unchanged",
        }

    isolated = [
        r
        for r in stability_rows
        if r.get("stability_class") in ("isolated_safe_point", "thin_safe_edge")
    ]
    any_productive_safe = any(
        int(r.get("productive_safe_point", 0)) for r in all_primary_regions
    )

    if isolated or (any_productive_safe and not region_candidates):
        n_prod = sum(
            1 for r in all_primary_regions if int(r.get("productive_safe_point", 0))
        )
        return {
            "stage2_q45_terminal": TERMINAL_B,
            "terminal_letter": "B",
            "reason": (
                f"observed productive-safe points exist (n_atlas_cells={n_prod}) "
                "but neighborhood/LOO thickness fails → isolated_safe_points_only; "
                "retain atlas, do not promote rules"
            ),
            "claims_blocked": blocked + ["formal_safe_region_not_authorized"],
            "claims_allowed": [
                "retain_atlas_for_followup",
                "report_isolated_observed_safe_points_descriptively",
            ],
            "next_authorized_step": (
                "retain full atlas; do not promote isolated points; "
                "optional deeper thickness diagnostics or ranking path"
            ),
            "n_productive_safe_cells": n_prod,
            "n_isolated_or_thin": len(isolated),
            "production_preset": "unchanged",
        }

    # C: enrichment without safe region
    enriching = [
        r
        for r in all_primary_regions
        if int(r.get("support", 0)) >= MIN_SUPPORT_FOR_ENRICHMENT
        and math.isfinite(float(r.get("enrichment") or 0))
        and float(r.get("enrichment") or 0) >= MIN_ENRICHMENT_USEFUL
        and int(r.get("n_neg_captured", 0)) > 0
        and int(r.get("gt_hurt", 0)) > 0
    ]
    if enriching:
        best = max(enriching, key=lambda r: float(r.get("enrichment") or 0))
        return {
            "stage2_q45_terminal": TERMINAL_C,
            "terminal_letter": "C",
            "reason": (
                f"conditional enrichment without zero-GT region "
                f"(n_enriching_cells={len(enriching)}; "
                f"best_enrichment={float(best.get('enrichment') or 0):.3f}); "
                "may study ranking/calibration, not reject rule"
            ),
            "claims_blocked": blocked
            + [
                "reject_rule_not_authorized",
                "formal_safe_region_not_authorized",
            ],
            "claims_allowed": [
                "report_conditional_enrichment_structure",
                "ranking_or_calibration_research_authorized",
            ],
            "next_authorized_step": (
                "ranking / calibration research on enriched cells; "
                "not reject-threshold promotion"
            ),
            "n_enriching_cells": len(enriching),
            "best_enrichment": float(best.get("enrichment") or 0),
            "production_preset": "unchanged",
        }

    # D: no useful structure
    return {
        "stage2_q45_terminal": TERMINAL_D,
        "terminal_letter": "D",
        "reason": (
            "complete restricted combination atlas shows no useful enrichment "
            "or productive-safe structure on locked cohort"
        ),
        "claims_blocked": blocked
        + [
            "reject_rule_not_authorized",
            "formal_safe_region_not_authorized",
        ],
        "claims_allowed": [
            "report_null_atlas_result",
            "new_signal_family_or_earlier_placement_authorized",
        ],
        "next_authorized_step": (
            "turn to new signal family or earlier placement "
            "(after full atlas null result)"
        ),
        "production_preset": "unchanged",
    }


# ---------------------------------------------------------------------------
# Secondary competition features (optional columns; not mainline)
# ---------------------------------------------------------------------------


def attach_secondary_competition_features(
    events: list[dict[str, Any]],
) -> list[str]:
    """Add a small set of competition-relative columns if reconstructable.

    Secondary only — does not replace frozen-signal main atlas.
    Missing relative fields stay NaN (no zero-fill).
    Returns list of secondary feature names attached.
    """
    from collections import defaultdict as dd

    groups: dict[tuple[Any, ...], list[int]] = dd(list)
    for i, r in enumerate(events):
        key = (str(r["sequence"]), int(r["frame"]), int(r.get("cand_slot", -1)))
        groups[key].append(i)

    secondary = [
        "sec_winner_runnerup_score_margin",
        "sec_delta_vs_ru_abs_log_h",
        "sec_competitor_count",
    ]
    for i, r in enumerate(events):
        for name in secondary:
            r[name] = float("nan")
        r["sec_competitor_count"] = float(_i(r.get("competitor_count", 0)))
        key = (str(r["sequence"]), int(r["frame"]), int(r.get("cand_slot", -1)))
        idxs = groups[key]
        ordered = sorted(
            idxs,
            key=lambda j: (
                _f(events[j].get("score_m_bridge")),
                int(events[j].get("lost_slot", -1)),
            ),
        )
        if (
            len(ordered) >= 2
            and _i(r.get("baseline_selected", r.get("baseline_accepted_candidate", 0)))
            == 1
        ):
            win = ordered[0]
            ru = ordered[1]
            # if this row is selected it should be ordered[0]
            if i == win or _i(r.get("baseline_rank", 99)) == 0:
                r["sec_winner_runnerup_score_margin"] = _f(
                    events[ru].get("score_m_bridge")
                ) - _f(events[win].get("score_m_bridge"))
                r["sec_delta_vs_ru_abs_log_h"] = _f(events[win].get("abs_log_h")) - _f(
                    events[ru].get("abs_log_h")
                )
    return secondary


# ---------------------------------------------------------------------------
# Full runner
# ---------------------------------------------------------------------------


def run_stage2_q45_atlas(
    *,
    q1q3_study_dir: Path,
    out_dir: Path,
    stage1_study_dir: Path | None = None,
    git_commit: str = "",
    study_id: str | None = None,
    include_secondary_competition: bool = True,
) -> dict[str, Any]:
    q1q3_study_dir = Path(q1q3_study_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_raw = load_d_online_events(q1q3_study_dir)
    source = q1q3_study_dir / "d_online_events.parquet"
    if not source.is_file():
        source = q1q3_study_dir / "d_online_events.csv"
    source_hash = _sha256_file(source) if source.is_file() else ""

    # Resolve stage1 dir for frame provenance MOT cross-check
    if stage1_study_dir is None:
        # try from events or default
        stage1_study_dir = Path(
            "out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close"
        )
    stage1_study_dir = Path(stage1_study_dir)

    frame_prov = check_frame_column_provenance(
        events_raw, stage1_study_dir=stage1_study_dir
    )

    events = [dict(r) for r in events_raw]
    secondary_names: list[str] = []
    if include_secondary_competition:
        secondary_names = attach_secondary_competition_features(events)

    locked = lock_q45_cohort(events)
    primary = locked["primary"]
    cohort_sum = locked["summary"]
    recon = reconcile_q4(n_d_online=len(events_raw), cohort=cohort_sum, primary=primary)
    if not recon["ok"]:
        raise Stage2Q45Error(f"recon FAIL: {recon.get('errors')}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sid = study_id or f"m_b1_5_stage2_q45_{stamp}"

    registry = build_threshold_registry(
        primary,
        signals=ORDERED_SIGNALS,
        secondary_features=secondary_names if include_secondary_competition else None,
    )
    # JSON-safe registry (drop arrays)
    registry_json = {k: v for k, v in registry.items() if not k.startswith("_")}

    atom_rows = build_single_atom_atlas(registry, primary_signals_only=False)
    # Pairwise only on primary frozen signals (mainline)
    pairwise_and = build_pairwise_atlas(
        registry, combinator="AND", primary_signals_only=True
    )
    pairwise_or = build_pairwise_atlas(
        registry, combinator="OR", primary_signals_only=True
    )

    # Secondary-only single-atom rows already mixed with is_secondary flag;
    # optional secondary pairwise skipped to keep mainline clean (user: secondary
    # can use same method — provide single-atom secondary in atom_atlas).

    stability = classify_region_stability(atom_rows, pairwise_and, pairwise_or)

    pareto_single = build_pareto_frontier(
        [r for r in atom_rows if not int(r.get("is_secondary_feature", 0))],
        kind="single_atom",
    )
    pareto_and = build_pareto_frontier(pairwise_and, kind="pairwise_AND")
    pareto_or = build_pareto_frontier(pairwise_or, kind="pairwise_OR")
    pareto_all = pareto_single + pareto_and + pareto_or

    terminal = classify_q45_terminal(
        stability_rows=stability,
        atom_rows=atom_rows,
        pairwise_and=pairwise_and,
        pairwise_or=pairwise_or,
    )

    # Per-sequence / LOO export (flattened from productive-safe + frontier)
    loo_rows = []
    per_seq_rows = []
    for r in atom_rows + pairwise_and + pairwise_or:
        rid = r.get("atom_id") or r.get("combo_id")
        if not int(r.get("productive_safe_point", 0)) and int(r.get("gt_hurt", 1)) > 0:
            # still keep high-enrichment cells' per-seq for atlas completeness?
            # Export only productive-safe + enrichment>=threshold to limit size
            enr = float(r.get("enrichment") or 0)
            if not (
                math.isfinite(enr)
                and enr >= MIN_ENRICHMENT_USEFUL
                and int(r.get("n_neg_captured", 0)) > 0
            ):
                continue
        # parse per-seq json
        try:
            pn = json.loads(r.get("per_sequence_neg_json") or "{}")
            pg = json.loads(r.get("per_sequence_gt_json") or "{}")
        except json.JSONDecodeError:
            pn, pg = {}, {}
        for s in sorted(set(list(pn) + list(pg))):
            per_seq_rows.append(
                {
                    "region_id": rid,
                    "sequence": s,
                    "n_neg": int(pn.get(s, 0)),
                    "n_gt": int(pg.get(s, 0)),
                    "support": int(pn.get(s, 0)) + int(pg.get(s, 0)),
                }
            )

    # LOO table from recompute on productive-safe primary regions only
    matrices = registry["_matrices"]
    y = registry["_y"]
    sequences = registry["_sequences"]
    base = float(registry["base_negative_rate"])
    for r in atom_rows:
        if not int(r.get("productive_safe_point", 0)):
            continue
        if int(r.get("is_secondary_feature", 0)):
            continue
        m = atom_mask(matrices[r["feature"]], r["direction"], float(r["thr_value"]))
        met = region_metrics(m, y, sequences, base_neg_rate=base)
        for lo in met["loo"]:
            loo_rows.append(
                {
                    "region_id": r["atom_id"],
                    "kind": "single_atom",
                    **lo,
                }
            )
    for r in pairwise_and + pairwise_or:
        if not int(r.get("productive_safe_point", 0)):
            continue
        m1 = atom_mask(
            matrices[r["feature_a"]],
            r["direction_a"],
            float(r["thr_value_a"]),
        )
        m2 = atom_mask(
            matrices[r["feature_b"]],
            r["direction_b"],
            float(r["thr_value_b"]),
        )
        m = (m1 & m2) if r["combinator"] == "AND" else (m1 | m2)
        met = region_metrics(m, y, sequences, base_neg_rate=base)
        for lo in met["loo"]:
            loo_rows.append(
                {
                    "region_id": r["combo_id"],
                    "kind": f"pairwise_{r['combinator']}",
                    **lo,
                }
            )

    # Counts for summary
    n_prod_single = sum(
        1
        for r in atom_rows
        if int(r.get("productive_safe_point", 0))
        and not int(r.get("is_secondary_feature", 0))
    )
    n_prod_and = sum(1 for r in pairwise_and if int(r.get("productive_safe_point", 0)))
    n_prod_or = sum(1 for r in pairwise_or if int(r.get("productive_safe_point", 0)))

    hashes: dict[str, str] = {}
    hashes["threshold_registry"] = write_json(
        out_dir / "threshold_registry.json", registry_json
    )
    hashes["frame_column_provenance"] = write_json(
        out_dir / "frame_column_provenance.json", frame_prov
    )
    hashes["atom_atlas"] = write_csv(out_dir / "atom_atlas.csv", atom_rows)
    write_parquet(out_dir / "atom_atlas.parquet", atom_rows)
    if (out_dir / "atom_atlas.parquet").is_file():
        hashes["atom_atlas_parquet"] = _sha256_file(out_dir / "atom_atlas.parquet")

    hashes["pairwise_and_atlas"] = write_csv(
        out_dir / "pairwise_and_atlas.csv", pairwise_and
    )
    write_parquet(out_dir / "pairwise_and_atlas.parquet", pairwise_and)
    if (out_dir / "pairwise_and_atlas.parquet").is_file():
        hashes["pairwise_and_atlas_parquet"] = _sha256_file(
            out_dir / "pairwise_and_atlas.parquet"
        )

    hashes["pairwise_or_atlas"] = write_csv(
        out_dir / "pairwise_or_atlas.csv", pairwise_or
    )
    write_parquet(out_dir / "pairwise_or_atlas.parquet", pairwise_or)
    if (out_dir / "pairwise_or_atlas.parquet").is_file():
        hashes["pairwise_or_atlas_parquet"] = _sha256_file(
            out_dir / "pairwise_or_atlas.parquet"
        )

    hashes["pareto_frontier"] = write_csv(out_dir / "pareto_frontier.csv", pareto_all)
    hashes["region_stability"] = write_csv(out_dir / "region_stability.csv", stability)
    hashes["per_sequence"] = write_csv(out_dir / "per_sequence.csv", per_seq_rows)
    hashes["loo"] = write_csv(out_dir / "loo.csv", loo_rows)
    hashes["cohort_summary"] = write_json(out_dir / "cohort_summary.json", cohort_sum)
    hashes["reconciliation"] = write_json(out_dir / "reconciliation.json", recon)

    summary = {
        "study_id": sid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_q1q3_study": str(q1q3_study_dir),
        "upstream_q4_study": Q4_STUDY_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "git_commit": git_commit,
        "D_online_total": len(events_raw),
        "n_primary": cohort_sum["n_primary"],
        "n_primary_negative": cohort_sum["n_primary_negative"],
        "n_primary_positive_protect": cohort_sum["n_primary_positive_protect"],
        "n_sequences_with_both_classes": cohort_sum["n_sequences_with_both_classes"],
        "cohort_definition": cohort_sum["cohort_definition"],
        "single_lattice_kind": SINGLE_LATTICE_KIND,
        "pairwise_lattice_kind": PAIRWISE_LATTICE_KIND,
        "n_single_atoms_registered": registry_json["n_single_atoms"],
        "n_pairwise_atoms_registered": registry_json["n_pairwise_atoms"],
        "n_atom_atlas_rows": len(atom_rows),
        "n_pairwise_and_rows": len(pairwise_and),
        "n_pairwise_or_rows": len(pairwise_or),
        "n_productive_safe_single": n_prod_single,
        "n_productive_safe_and": n_prod_and,
        "n_productive_safe_or": n_prod_or,
        "n_pareto_frontier": len(pareto_all),
        "n_stability_rows": len(stability),
        "secondary_competition_features": secondary_names,
        "frame_provenance_kind": frame_prov["semantic"]["kind"],
        "frame_is_absolute_mot": frame_prov["semantic"]["is_absolute_mot_frame"],
        "frame_observation_limitation_only": frame_prov["conclusions"][
            "observation_limitation_only"
        ],
        **terminal,
        "reconciliation_acceptance": recon["acceptance"],
        "forbidden_this_round": [
            "single_best_threshold_as_rule",
            "unrestricted_boolean_mining",
            "three_plus_atom_combos",
            "learned_classifier",
            "post_hoc_grammar_expansion",
            "hook_policy",
            "e2e_effect_claim",
            "production_preset_change",
            "sample_zero_gt_named_safe_rule",
            "q4_weak_auc_means_boolean_impossible",
            "new_signal_terminal_classification",
        ],
        "q4_interpretation": (
            "Q4 weak marginal AUC closes singleton frozen-tail thr promotion; "
            "does not forbid threshold/Boolean combination as analysis method."
        ),
    }
    hashes["summary"] = write_json(out_dir / "summary.json", summary)
    md = _render_md(summary, terminal, frame_prov, stability, pareto_all)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    hashes["summary_md"] = hashlib.sha256(md.encode("utf-8")).hexdigest()

    manifest = {
        "schema": "m_b1_5_stage2_q45_atlas_manifest_v1",
        "study_id": sid,
        "git_commit": git_commit,
        "source_q1q3_study": str(q1q3_study_dir),
        "source_event_table": str(source),
        "source_event_table_hash": source_hash,
        "upstream_q4_study": Q4_STUDY_ID,
        "candidate_universe_id": CANDIDATE_UNIVERSE_ID,
        "substrate_id": SUBSTRATE_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "cohort_definition": cohort_sum["cohort_definition"],
        "sequence_set": sorted({str(r["sequence"]) for r in primary}),
        "artifact_hashes": hashes,
        "created_utc": summary["created_utc"],
        "terminal_letter": terminal.get("terminal_letter"),
        "stage2_q45_terminal": terminal.get("stage2_q45_terminal"),
        "production_preset": "unchanged",
    }
    write_json(out_dir / "manifest.json", manifest)
    summary["manifest"] = manifest
    return summary


def _render_md(
    summary: Mapping[str, Any],
    terminal: Mapping[str, Any],
    frame_prov: Mapping[str, Any],
    stability: Sequence[Mapping[str, Any]],
    pareto: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# Stage 2 Q4.5 threshold-combination atlas — `{summary.get('study_id')}`",
        "",
        "<!-- doc-status: research-artifact -->",
        "",
        "## Terminal",
        "",
        "```text",
        f"stage2_q45_terminal: {terminal.get('stage2_q45_terminal')}",
        f"terminal_letter: {terminal.get('terminal_letter')}",
        f"n_primary: {summary.get('n_primary')} "
        f"(neg={summary.get('n_primary_negative')}, "
        f"pos={summary.get('n_primary_positive_protect')})",
        f"productive_safe cells: single={summary.get('n_productive_safe_single')} "
        f"AND={summary.get('n_productive_safe_and')} "
        f"OR={summary.get('n_productive_safe_or')}",
        f"next: {terminal.get('next_authorized_step')}",
        f"production_preset: {summary.get('production_preset')}",
        "```",
        "",
        f"**Reason:** {terminal.get('reason')}",
        "",
        "## Frame column provenance",
        "",
        f"- semantic: `{frame_prov['semantic']['kind']}`",
        f"- absolute MOT frame: **{frame_prov['semantic']['is_absolute_mot_frame']}**",
        f"- unique values on D_online: {frame_prov.get('unique_values')}",
        f"- observation limitation only: "
        f"{frame_prov['conclusions']['observation_limitation_only']}",
        f"- affects Q4.5 mainline: "
        f"{frame_prov['conclusions']['affects_q45_threshold_atlas_mainline']}",
        "",
        frame_prov["conclusions"]["note"],
        "",
        "## Atlas sizes",
        "",
        f"- single-atom rows: {summary.get('n_atom_atlas_rows')}",
        f"- pairwise AND rows: {summary.get('n_pairwise_and_rows')}",
        f"- pairwise OR rows: {summary.get('n_pairwise_or_rows')}",
        f"- pareto frontier rows: {summary.get('n_pareto_frontier')}",
        f"- stability rows (productive-safe): {summary.get('n_stability_rows')}",
        "",
        "## Stability class counts",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for r in stability:
        counts[str(r.get("stability_class"))] += 1
    for k in sorted(counts):
        lines.append(f"- `{k}`: {counts[k]}")
    lines += [
        "",
        "## Pareto head (first 12)",
        "",
        "| kind | gt_hurt | n_neg | cov | n_seq_neg | max_share | loo_safe |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for r in pareto[:12]:
        lines.append(
            f"| {r.get('kind')} | {r.get('gt_hurt')} | {r.get('n_neg_captured')} | "
            f"{float(r.get('coverage') or 0):.3f} | {r.get('n_sequences_with_neg')} | "
            f"{float(r.get('max_sequence_share') or 0):.2f} | "
            f"{r.get('loo_all_gt_hurt_zero')} |"
        )
    lines += [
        "",
        "## Claim firewall",
        "",
        "Blocked:",
        "",
    ]
    for b in terminal.get("claims_blocked", []):
        lines.append(f"- `{b}`")
    lines += ["", "Allowed:", ""]
    for a in terminal.get("claims_allowed", []) or ["(none)"]:
        lines.append(f"- `{a}`")
    lines += [
        "",
        "Observed productive-safe points are **not** safe rules.",
        "Q4 weak AUC ≠ Boolean combination impossible.",
        "",
    ]
    return "\n".join(lines) + "\n"
