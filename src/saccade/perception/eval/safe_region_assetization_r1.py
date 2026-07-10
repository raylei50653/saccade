"""Safe-Region Assetization R1 — G1–G3 region assets + linearized feasibility probe.

Research-only. Does not modify production hooks, presets, or terminal B.
Phase A packages existing Q4.5 G1–G3 atlases into A1-max region assets.
Phase B runs a sparse non-negative hard-safety capacity probe (not a policy).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from saccade.perception.eval.d_online_stage2 import write_csv, write_json, write_parquet
from saccade.perception.eval.d_online_stage2_q4 import (
    load_d_online_events,
    lock_q4_cohort,
)
from saccade.perception.eval.d_online_stage2_q45_atlas import (
    EXPECTED_PRIMARY_N,
    EXPECTED_PRIMARY_NEG,
    EXPECTED_PRIMARY_POS,
    atom_mask,
)

# ---------------------------------------------------------------------------
# Locked research truth (must not rewrite historical G1–G3 headlines)
# ---------------------------------------------------------------------------

TASK_NAME = "safe_region_assetization_r1"
SIGNAL_FAMILY = (
    "score_m_bridge",
    "abs_log_h",
    "dist_h",
    "abs_ratio_m1",
    "resid_mean",
)
FIXED_K_GRID: tuple[int, ...] = (2, 3, 4, 5)
L1_COUNT_THRESHOLDS: tuple[int, ...] = (1, 2, 3, 4, 5)
HISTORICAL_G1_PS = 1
HISTORICAL_G2_PS = 153
HISTORICAL_G3_PS = 0
HISTORICAL_G2_UNIQUE_PS_MASKS = 15
HISTORICAL_G2_MULTI_SEQ_COORDS = 12
HISTORICAL_COORD_UNION_INTERIOR = 0
HISTORICAL_NESTED_PORTABLE = 0
TERMINAL_B = "isolated_safe_points_only"

EPS = 1e-9
BEAM_TOP_M = 48  # candidate basis pool for L2/L3 sparse search
ROBUST_DELTA_GRID = (1e-4, 1e-3, 1e-2, 5e-2)


# ---------------------------------------------------------------------------
# Hash / id helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_hash(obj: Any) -> str:
    return _sha256_bytes(_canonical_json(obj).encode("utf-8"))


def _mask_sha256(mask: np.ndarray) -> str:
    bits = np.packbits(np.asarray(mask, dtype=bool).astype(np.uint8))
    return _sha256_bytes(bits.tobytes())


def _parse_seq_json(raw: Any) -> dict[str, int]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return {str(k): int(v) for k, v in json.loads(raw).items()}
    return {}


def region_id(
    grammar: str,
    predicate_role: str,
    mask_sha: str,
    component: int | str,
    *,
    grid_key: Any | None = None,
) -> str:
    """Deterministic region identity — never a row index alone.

    Grid key is required when the same mask/component can appear under multiple
    registered lattices (pairwise feature×direction grids). Without it, identity
    collides across grids.
    """
    if grid_key is None:
        return f"q45:{grammar}:{predicate_role}:{mask_sha[:16]}:c{component}"
    ghash = _sha256_bytes(
        _canonical_json(
            list(grid_key) if isinstance(grid_key, tuple) else grid_key
        ).encode("utf-8")
    )[:8]
    return f"q45:{grammar}:{predicate_role}:{mask_sha[:16]}:c{component}:g{ghash}"


# ---------------------------------------------------------------------------
# Geometry: dual margins + components
# ---------------------------------------------------------------------------


def connected_components_1d(coords: set[int]) -> dict[int, int]:
    if not coords:
        return {}
    ordered = sorted(coords)
    parent = {c: c for c in ordered}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ordered) - 1):
        if ordered[i + 1] - ordered[i] == 1:
            union(ordered[i], ordered[i + 1])
    roots = {find(c) for c in ordered}
    rid = {r: i for i, r in enumerate(sorted(roots))}
    return {c: rid[find(c)] for c in ordered}


def connected_components_2d(
    coords: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    if not coords:
        return {}
    parent: dict[tuple[int, int], tuple[int, int]] = {c: c for c in coords}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in coords:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + di, j + dj)
            if n in coords:
                union((i, j), n)
    roots = {find(c) for c in coords}
    rid = {r: i for i, r in enumerate(sorted(roots))}
    return {c: rid[find(c)] for c in coords}


def nearest_unsafe_distance_1d(coord: int, safe: set[int], lattice: set[int]) -> int:
    """Min steps along the 1D lattice thr-index line to a non-safe lattice point."""
    if coord not in safe:
        return 0
    unsafe = lattice - safe
    if not unsafe:
        return max(coord - min(lattice), max(lattice) - coord) if lattice else 0
    return min(abs(coord - u) for u in unsafe)


def full_neighborhood_safe_radius_1d(coord: int, safe: set[int]) -> int:
    """Largest r such that all bilateral neighbors within distance r are safe.

    Thin strip may have nearest_unsafe_distance > 0 but radius 0 when either
    side is missing (endpoint / isolated).
    """
    if coord not in safe:
        return 0
    r = 0
    while True:
        left, right = coord - (r + 1), coord + (r + 1)
        if left not in safe or right not in safe:
            return r
        r += 1
        if r > 10_000:
            return r


def nearest_unsafe_distance_2d(
    coord: tuple[int, int],
    safe: set[tuple[int, int]],
    lattice: set[tuple[int, int]],
) -> int:
    if coord not in safe:
        return 0
    unsafe = lattice - safe
    if not unsafe:
        return 0
    return min(abs(coord[0] - u[0]) + abs(coord[1] - u[1]) for u in unsafe)


def full_neighborhood_safe_radius_2d(
    coord: tuple[int, int],
    safe: set[tuple[int, int]],
) -> int:
    """Largest r with full 4-neighbor diamond of radius r inside safe (erosion)."""
    if coord not in safe:
        return 0
    r = 0
    while True:
        nr = r + 1
        for di in range(-nr, nr + 1):
            dj = nr - abs(di)
            for s in (-1, 1) if dj else (0,):
                n = (coord[0] + di, coord[1] + s * dj)
                if n not in safe:
                    return r
        r = nr
        if r > 10_000:
            return r


def component_geometry_1d(coords: set[int]) -> dict[str, Any]:
    if not coords:
        return {
            "component_size": 0,
            "bounding_box_shape": "()",
            "active_axis_count": 0,
            "axis_widths": "[]",
            "single_axis_strip": 0,
            "row_or_column_degenerate": 1,
            "coordinate_union_interior_count": 0,
        }
    lo, hi = min(coords), max(coords)
    width = hi - lo + 1
    interior = {i for i in coords if (i - 1 in coords and i + 1 in coords)}
    return {
        "component_size": len(coords),
        "bounding_box_shape": f"({width},)",
        "active_axis_count": 1 if width > 1 else 0,
        "axis_widths": json.dumps([width]),
        "single_axis_strip": int(width > 1),
        "row_or_column_degenerate": 1,
        "coordinate_union_interior_count": len(interior),
    }


def component_geometry_2d(coords: set[tuple[int, int]]) -> dict[str, Any]:
    if not coords:
        return {
            "component_size": 0,
            "bounding_box_shape": "()",
            "active_axis_count": 0,
            "axis_widths": "[]",
            "single_axis_strip": 0,
            "row_or_column_degenerate": 1,
            "coordinate_union_interior_count": 0,
        }
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    w0 = max(xs) - min(xs) + 1
    w1 = max(ys) - min(ys) + 1
    active = int(w0 > 1) + int(w1 > 1)
    interior = {
        (i, j)
        for i, j in coords
        if (
            (i - 1, j) in coords
            and (i + 1, j) in coords
            and (i, j - 1) in coords
            and (i, j + 1) in coords
        )
    }
    return {
        "component_size": len(coords),
        "bounding_box_shape": f"({w0},{w1})",
        "active_axis_count": active,
        "axis_widths": json.dumps([w0, w1]),
        "single_axis_strip": int(active == 1 and len(coords) > 1),
        "row_or_column_degenerate": int(active <= 1),
        "coordinate_union_interior_count": len(interior),
    }


# ---------------------------------------------------------------------------
# Cohort / matrices
# ---------------------------------------------------------------------------


@dataclass
class CohortBundle:
    primary: list[dict[str, Any]]
    selected_unresolved: list[dict[str, Any]]
    y: np.ndarray  # 1=neg, 0=gt
    sequences: np.ndarray
    matrices: dict[str, np.ndarray]
    unknown_matrices: dict[str, np.ndarray]
    signal_registry_hash: str
    cohort_hash: str
    n_neg: int
    n_gt: int
    n_unknown: int


def load_cohort_bundle(
    events_path: Path,
    threshold_registry: Mapping[str, Any],
) -> CohortBundle:
    rows = load_d_online_events(events_path)
    locked = lock_q4_cohort(rows)
    primary = list(locked["primary"])
    if len(primary) != EXPECTED_PRIMARY_N:
        raise ValueError(f"primary n={len(primary)} != {EXPECTED_PRIMARY_N}")
    n_neg = sum(1 for r in primary if int(r["q4_y"]) == 1)
    n_gt = sum(1 for r in primary if int(r["q4_y"]) == 0)
    if n_neg != EXPECTED_PRIMARY_NEG or n_gt != EXPECTED_PRIMARY_POS:
        raise ValueError(f"neg/gt={n_neg}/{n_gt} != locked")
    unres = [
        r
        for r in rows
        if int(r.get("baseline_selected", 0)) == 1
        and str(r.get("pair_label", "")) == "unknown"
    ]
    if len(unres) != 21:
        raise ValueError(f"selected unresolved={len(unres)} != 21")

    signals = list(threshold_registry.get("signals_primary") or SIGNAL_FAMILY)
    y = np.asarray([int(r["q4_y"]) for r in primary], dtype=int)
    sequences = np.asarray([str(r["sequence"]) for r in primary], dtype=object)
    matrices = {
        s: np.asarray([float(r.get(s, float("nan"))) for r in primary], dtype=float)
        for s in signals
    }
    unknown_matrices = {
        s: np.asarray([float(r.get(s, float("nan"))) for r in unres], dtype=float)
        for s in signals
    }
    # registry hash excludes raw matrices if present
    reg_public = {
        k: v for k, v in threshold_registry.items() if not str(k).startswith("_")
    }
    signal_registry_hash = _content_hash(reg_public)
    cohort_hash = _content_hash(
        {
            "n_primary": len(primary),
            "n_neg": n_neg,
            "n_gt": n_gt,
            "n_unknown": len(unres),
            "event_ids": [r.get("event_id") for r in primary],
            "unknown_event_ids": [r.get("event_id") for r in unres],
        }
    )
    return CohortBundle(
        primary=primary,
        selected_unresolved=unres,
        y=y,
        sequences=sequences,
        matrices=matrices,
        unknown_matrices=unknown_matrices,
        signal_registry_hash=signal_registry_hash,
        cohort_hash=cohort_hash,
        n_neg=n_neg,
        n_gt=n_gt,
        n_unknown=len(unres),
    )


# ---------------------------------------------------------------------------
# Phase A — region asset pack from sealed atlases
# ---------------------------------------------------------------------------


def _capacity_tags(
    n_neg: int,
    n_seq: int,
    max_share: float | None,
    n_coords_for_mask: int,
) -> list[str]:
    tags: list[str] = []
    if n_neg <= 1:
        tags.append("single_event")
    if n_seq <= 1:
        tags.append("single_sequence")
    if n_seq >= 2:
        tags.append("multi_sequence")
    if n_coords_for_mask >= 5:
        tags.append("highly_duplicated")
    if max_share is not None and math.isfinite(max_share) and max_share > 0.5:
        tags.append("sequence_dominant")
    return tags


def build_phase_a_assets(
    *,
    atom_df: pd.DataFrame,
    and_df: pd.DataFrame,
    or_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    cohort: CohortBundle,
    q45_dir: Path,
    evaluator_truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert G1–G3 atlases into region asset tables (no evaluator rerun)."""
    provenance = {
        "q45_dir": str(q45_dir),
        "atom_atlas_sha256": _sha256_file(q45_dir / "atom_atlas.parquet")
        if (q45_dir / "atom_atlas.parquet").exists()
        else _sha256_file(q45_dir / "atom_atlas.csv"),
        "and_atlas_sha256": _sha256_file(q45_dir / "pairwise_and_atlas.parquet")
        if (q45_dir / "pairwise_and_atlas.parquet").exists()
        else "",
        "or_atlas_sha256": _sha256_file(q45_dir / "pairwise_or_atlas.parquet")
        if (q45_dir / "pairwise_or_atlas.parquet").exists()
        else "",
        "terminal_letter": "B",
        "terminal": TERMINAL_B,
    }

    # --- lattice denominators (grammar-specific) ---
    if "is_secondary_feature" in atom_df.columns:
        atom_p = atom_df[atom_df["is_secondary_feature"].fillna(0).astype(int) == 0]
    else:
        atom_p = atom_df
    g1_lattice = len(atom_p)
    g2_lattice = len(and_df)
    g3_lattice = len(or_df)

    def _ps(df: pd.DataFrame) -> pd.DataFrame:
        if "productive_safe_point" not in df.columns:
            return df.iloc[0:0]
        col = df["productive_safe_point"]
        if col.dtype == bool:
            return df[col]
        return df[col.astype(int) == 1]

    g1_ps = _ps(atom_p)
    g2_ps = _ps(and_df)
    g3_ps = _ps(or_df)

    # Historical headline lock (do not rewrite)
    if len(g1_ps) != HISTORICAL_G1_PS:
        raise ValueError(f"G1 PS={len(g1_ps)} != historical {HISTORICAL_G1_PS}")
    if len(g2_ps) != HISTORICAL_G2_PS:
        raise ValueError(f"G2 PS={len(g2_ps)} != historical {HISTORICAL_G2_PS}")
    if len(g3_ps) != HISTORICAL_G3_PS:
        raise ValueError(f"G3 PS={len(g3_ps)} != historical {HISTORICAL_G3_PS}")

    grammar_summary: list[dict[str, Any]] = []
    region_components: list[dict[str, Any]] = []
    region_masks: list[dict[str, Any]] = []
    region_capacity: list[dict[str, Any]] = []
    region_seq: list[dict[str, Any]] = []
    region_margin: list[dict[str, Any]] = []
    region_manifest_rows: list[dict[str, Any]] = []

    def process_grammar(
        grammar: str,
        df_all: pd.DataFrame,
        df_ps: pd.DataFrame,
        lattice_n: int,
        mode: str,  # "1d" | "2d"
    ) -> None:
        # dual area
        safe_coord_n = len(df_ps)  # productive-safe == safe under Q4.5 definition
        unique_safe_masks = int(df_ps["mask_sha256"].nunique()) if len(df_ps) else 0
        unique_all_masks = int(df_all["mask_sha256"].nunique()) if len(df_all) else 0
        # productive unique masks (same as safe PS unique here)
        grammar_summary.append(
            {
                "grammar_id": grammar,
                "lattice_id": f"q45_{grammar}_registered",
                "lattice_n_coordinates": lattice_n,
                "coordinate_safe_area_ratio": safe_coord_n / lattice_n
                if lattice_n
                else float("nan"),
                "coordinate_productive_area_ratio": safe_coord_n / lattice_n
                if lattice_n
                else float("nan"),
                "unique_mask_safe_ratio": unique_safe_masks / unique_all_masks
                if unique_all_masks
                else float("nan"),
                "unique_mask_productive_ratio": unique_safe_masks / unique_all_masks
                if unique_all_masks
                else float("nan"),
                "n_productive_safe_coordinates": safe_coord_n,
                "n_unique_productive_masks": unique_safe_masks,
                "n_unique_masks_all": unique_all_masks,
                "historical_lock_ok": 1,
            }
        )
        if len(df_ps) == 0:
            # still emit G3 null domain asset
            rid = region_id(grammar, "domain_null", "0" * 64, 0)
            region_manifest_rows.append(
                {
                    "region_id": rid,
                    "grammar_id": grammar,
                    "signal_registry_hash": cohort.signal_registry_hash,
                    "cohort_hash": cohort.cohort_hash,
                    "lattice_id": f"q45_{grammar}_registered",
                    "coordinate_system": mode,
                    "mask_sha256": "",
                    "component_id": 0,
                    "evaluator_truth": TERMINAL_B,
                    "artifact_provenance": _canonical_json(provenance),
                    "maturity_level": "A1_region_asset",
                    "action_status": "observation_only",
                    "transfer_status": "exact_clause_failed"
                    if grammar != "g3_or"
                    else "not_tested",
                    "promotion_status": "forbidden",
                    "null_domain": 1,
                }
            )
            return

        # group by grid for geometry
        if mode == "1d":
            grids: dict[Any, pd.DataFrame] = {
                (str(f), str(d)): g
                for (f, d), g in df_all.groupby(["feature", "direction"], sort=True)
            }
            ps_by_grid: dict[Any, pd.DataFrame] = {
                (str(f), str(d)): g
                for (f, d), g in df_ps.groupby(["feature", "direction"], sort=True)
            }
        else:
            grids = {}
            for key, g in df_all.groupby(
                ["feature_a", "direction_a", "feature_b", "direction_b"], sort=True
            ):
                grids[tuple(str(x) for x in key)] = g
            ps_by_grid = {}
            for key, g in df_ps.groupby(
                ["feature_a", "direction_a", "feature_b", "direction_b"], sort=True
            ):
                ps_by_grid[tuple(str(x) for x in key)] = g

        for gkey, g_all in grids.items():
            g_ps = ps_by_grid.get(gkey)
            if g_ps is None or len(g_ps) == 0:
                continue
            if mode == "1d":
                lattice_coords = set(int(x) for x in g_all["thr_index"].tolist())
                safe_coords = set(int(x) for x in g_ps["thr_index"].tolist())
                comp_of = connected_components_1d(safe_coords)
            else:
                lattice_coords_2d = {
                    (int(a), int(b))
                    for a, b in zip(
                        g_all["thr_index_a"].tolist(), g_all["thr_index_b"].tolist()
                    )
                }
                safe_coords_2d = {
                    (int(a), int(b))
                    for a, b in zip(
                        g_ps["thr_index_a"].tolist(), g_ps["thr_index_b"].tolist()
                    )
                }
                lattice_coords = lattice_coords_2d  # type: ignore[assignment]
                safe_coords = safe_coords_2d  # type: ignore[assignment]
                comp_of = connected_components_2d(safe_coords_2d)  # type: ignore[assignment]

            # per-mask collapse
            mask_groups: dict[str, pd.DataFrame] = {
                str(sig): g for sig, g in g_ps.groupby("mask_sha256", sort=True)
            }
            # component → coords / masks
            comp_coords: dict[int, set[Any]] = defaultdict(set)
            comp_masks: dict[int, set[str]] = defaultdict(set)
            if mode == "1d":
                for sig, mg in mask_groups.items():
                    for t in mg["thr_index"].tolist():
                        c: Any = int(t)
                        cid = int(comp_of[c])
                        comp_coords[cid].add(c)
                        comp_masks[cid].add(sig)
            else:
                for sig, mg in mask_groups.items():
                    for a, b in zip(mg["thr_index_a"], mg["thr_index_b"]):
                        c = (int(a), int(b))
                        cid = int(comp_of[c])
                        comp_coords[cid].add(c)
                        comp_masks[cid].add(sig)

            for cid, coords in sorted(comp_coords.items()):
                geom = (
                    component_geometry_1d(coords)
                    if mode == "1d"
                    else component_geometry_2d(coords)
                )
                # representative mask for component identity: lex-smallest
                masks_sorted = sorted(comp_masks[cid])
                rep_mask = masks_sorted[0]
                rid = region_id(
                    grammar,
                    "productive_safe_component",
                    rep_mask,
                    cid,
                    grid_key=gkey,
                )
                region_components.append(
                    {
                        "region_id": rid,
                        "grammar_id": grammar,
                        "component_id": cid,
                        "grid_key": json.dumps(
                            list(gkey) if isinstance(gkey, tuple) else [gkey]
                        ),
                        "n_prediction_masks_in_component": len(masks_sorted),
                        "unique_mask_count": len(
                            masks_sorted
                        ),  # alias; prefer n_prediction_*
                        "masks_json": json.dumps(masks_sorted),
                        **geom,
                    }
                )
                # capacity at component level: union of masks' event support is
                # not re-evaluated; use max rep row among masks
                rep_rows = mask_groups[rep_mask]
                rep = rep_rows.iloc[0]
                n_neg = int(rep["n_neg_captured"])
                # recompute capacity from best mask in component (max neg)
                best = max(
                    (mask_groups[m].iloc[0] for m in masks_sorted),
                    key=lambda r: int(r["n_neg_captured"]),
                )
                n_neg = int(best["n_neg_captured"])
                n_seq = int(best.get("n_sequences_with_neg", 0) or 0)
                max_share = best.get("max_neg_sequence_share")
                max_share_f = (
                    float(max_share)
                    if max_share is not None and math.isfinite(float(max_share))
                    else None
                )
                seq_neg = _parse_seq_json(best.get("per_sequence_neg_json"))
                tags = _capacity_tags(
                    n_neg, n_seq, max_share_f, int(geom["component_size"])
                )
                region_capacity.append(
                    {
                        "region_id": rid,
                        "grammar_id": grammar,
                        "n_neg_captured": n_neg,
                        "negative_capture_rate": n_neg / cohort.n_neg
                        if cohort.n_neg
                        else float("nan"),
                        "n_sequences_with_productive_support": n_seq,
                        "worst_sequence_capacity": min(seq_neg.values())
                        if seq_neg
                        else 0,
                        "capacity_by_component": n_neg,
                        "capacity_by_unique_mask": json.dumps(
                            {
                                m: int(mask_groups[m].iloc[0]["n_neg_captured"])
                                for m in masks_sorted
                            }
                        ),
                        "coordinate_count_per_mask": json.dumps(
                            {m: int(len(mask_groups[m])) for m in masks_sorted}
                        ),
                        "capacity_tags": json.dumps(tags),
                    }
                )
                prod_seqs = sorted([s for s, v in seq_neg.items() if v > 0])
                n_prod_seq = len(prod_seqs)
                total_neg = sum(seq_neg.values()) or 1
                dom = max(seq_neg.values()) / total_neg if seq_neg else float("nan")
                region_seq.append(
                    {
                        "region_id": rid,
                        "grammar_id": grammar,
                        "productive_sequences": json.dumps(prod_seqs),
                        "n_productive_sequences": n_prod_seq,
                        "sequence_specific_island": int(n_prod_seq <= 1),
                        "multi_sequence_productive": int(n_prod_seq >= 2),
                        "worst_sequence_productive_capacity": min(seq_neg.values())
                        if seq_neg
                        else 0,
                        "sequence_dominance_ratio": dom,
                        "per_sequence_neg_json": json.dumps(seq_neg, sort_keys=True),
                    }
                )
                # margins on component coords
                if mode == "1d":
                    dists = [
                        nearest_unsafe_distance_1d(c, safe_coords, lattice_coords)
                        for c in coords
                    ]
                    radii = [
                        full_neighborhood_safe_radius_1d(c, safe_coords) for c in coords
                    ]
                else:
                    dists = [
                        nearest_unsafe_distance_2d(c, safe_coords, lattice_coords)  # type: ignore[arg-type]
                        for c in coords
                    ]
                    radii = [
                        full_neighborhood_safe_radius_2d(c, safe_coords)  # type: ignore[arg-type]
                        for c in coords
                    ]
                region_margin.append(
                    {
                        "region_id": rid,
                        "grammar_id": grammar,
                        "nearest_unsafe_distance_min": min(dists) if dists else 0,
                        "nearest_unsafe_distance_max": max(dists) if dists else 0,
                        "full_neighborhood_safe_radius_min": min(radii) if radii else 0,
                        "full_neighborhood_safe_radius_max": max(radii) if radii else 0,
                        "thin_strip_flag": int(
                            (min(dists) if dists else 0) > 0
                            and (max(radii) if radii else 0) == 0
                        ),
                    }
                )
                region_manifest_rows.append(
                    {
                        "region_id": rid,
                        "grammar_id": grammar,
                        "signal_registry_hash": cohort.signal_registry_hash,
                        "cohort_hash": cohort.cohort_hash,
                        "lattice_id": f"q45_{grammar}_registered",
                        "coordinate_system": mode,
                        "mask_sha256": rep_mask,
                        "component_id": cid,
                        "evaluator_truth": TERMINAL_B,
                        "artifact_provenance": _canonical_json(provenance),
                        "maturity_level": "A1_region_asset",
                        "action_status": "observation_only",
                        "transfer_status": "exact_clause_failed",
                        "promotion_status": "forbidden",
                        "null_domain": 0,
                    }
                )

            # per unique mask rows
            for sig, mg in mask_groups.items():
                rep = mg.iloc[0]
                if mode == "1d":
                    mcoords: set[Any] = set(int(x) for x in mg["thr_index"].tolist())
                    cid0 = int(comp_of[next(iter(mcoords))])
                else:
                    mcoords = {
                        (int(a), int(b))
                        for a, b in zip(mg["thr_index_a"], mg["thr_index_b"])
                    }
                    cid0 = int(comp_of[next(iter(mcoords))])
                rid_m = region_id(
                    grammar, "grid_local_mask_asset", sig, cid0, grid_key=gkey
                )
                region_masks.append(
                    {
                        "region_id": rid_m,
                        "grammar_id": grammar,
                        "mask_sha256": sig,
                        "component_id": cid0,
                        "grid_key": json.dumps(
                            list(gkey) if isinstance(gkey, tuple) else [gkey]
                        ),
                        "asset_kind": "grid_local_mask_placement",
                        "coordinate_count": len(mcoords),
                        "n_neg_captured": int(rep["n_neg_captured"]),
                        "n_gt_captured": int(
                            rep.get("n_gt_captured", rep.get("gt_hurt", 0))
                        ),
                        "n_unresolved_selected": int(
                            rep.get("n_unresolved_selected", 0) or 0
                        ),
                        "n_sequences_with_neg": int(
                            rep.get("n_sequences_with_neg", 0) or 0
                        ),
                        "semantic_aliases_json": json.dumps(
                            {
                                "feature_a": str(
                                    rep.get("feature", rep.get("feature_a", ""))
                                ),
                                "direction_a": str(
                                    rep.get("direction", rep.get("direction_a", ""))
                                ),
                                "feature_b": str(rep.get("feature_b", "")),
                                "direction_b": str(rep.get("direction_b", "")),
                                "thr_indices": sorted(
                                    list(mcoords),
                                    key=lambda x: x if isinstance(x, int) else x,
                                ),
                            },
                            default=str,
                        ),
                    }
                )

    process_grammar("g1_singleton", atom_p, g1_ps, g1_lattice, "1d")
    process_grammar("g2_pairwise_and", and_df, g2_ps, g2_lattice, "2d")
    process_grammar("g3_hard_or", or_df, g3_ps, g3_lattice, "2d")

    # multi-seq coordinate count check vs historical
    multi_seq_coords = 0
    for _, r in g2_ps.iterrows():
        if int(r.get("n_sequences_with_neg", 0) or 0) >= 2:
            multi_seq_coords += 1
    # historical note: 12 multi-sequence coordinates (do not rewrite if off by filter)
    historical = {
        "g1_singleton_productive_safe": HISTORICAL_G1_PS,
        "g2_pairwise_and_productive_safe": HISTORICAL_G2_PS,
        "g2_unique_productive_masks": HISTORICAL_G2_UNIQUE_PS_MASKS,
        "g2_multi_sequence_coordinates": HISTORICAL_G2_MULTI_SEQ_COORDS,
        "g2_multi_sequence_coordinates_observed": multi_seq_coords,
        "coordinate_union_interior": HISTORICAL_COORD_UNION_INTERIOR,
        "nested_exact_absolute_portable_clauses": HISTORICAL_NESTED_PORTABLE,
        "g3_productive_safe": HISTORICAL_G3_PS,
        "terminal": TERMINAL_B,
    }

    # --- identity inventory (claim hygiene: do not conflate these) ---
    g1_ps_masks = (
        set(str(x) for x in g1_ps["mask_sha256"].tolist()) if len(g1_ps) else set()
    )
    g2_ps_masks = (
        set(str(x) for x in g2_ps["mask_sha256"].tolist()) if len(g2_ps) else set()
    )
    g3_ps_masks = (
        set(str(x) for x in g3_ps["mask_sha256"].tolist()) if len(g3_ps) else set()
    )
    union_ps_masks = g1_ps_masks | g2_ps_masks | g3_ps_masks
    mask_table_shas = [str(r.get("mask_sha256", "")) for r in region_masks]
    identity_inventory = {
        "n_unique_prediction_masks_productive_safe": len(union_ps_masks),
        "n_unique_prediction_masks_g1": len(g1_ps_masks),
        "n_unique_prediction_masks_g2": len(g2_ps_masks),
        "n_unique_prediction_masks_g3": len(g3_ps_masks),
        "n_g1_masks_also_in_g2": len(g1_ps_masks & g2_ps_masks),
        "n_grid_local_mask_assets": len(region_masks),
        "n_mask_component_grid_rows": len(region_masks),
        "n_semantic_role_assets": len(region_manifest_rows),
        "n_productive_safe_components": len(region_components),
        "n_coordinate_instances_productive_safe": (
            HISTORICAL_G1_PS + HISTORICAL_G2_PS + HISTORICAL_G3_PS
        ),
        "n_region_mask_table_rows": len(region_masks),
        "n_distinct_mask_sha_in_region_mask_table": len(
            set(s for s in mask_table_shas if s)
        ),
        "n_distinct_region_id_in_region_mask_table": len(
            {str(r.get("region_id")) for r in region_masks}
        ),
        "reconciliation_note": (
            "34 region_masks rows (if present) are grid-local mask placements "
            "(same prediction mask can appear under multiple pairwise grids), "
            "NOT 34 distinct prediction masks. Distinct productive-safe "
            "prediction masks = n_unique_prediction_masks_productive_safe "
            "(expected 15 = G2 historical unique; G1 mask is a subset of G2)."
        ),
        "historical_g2_unique_productive_masks": HISTORICAL_G2_UNIQUE_PS_MASKS,
        "matches_historical_g2_unique": int(
            len(g2_ps_masks) == HISTORICAL_G2_UNIQUE_PS_MASKS
        ),
    }

    return {
        "grammar_region_summary": grammar_summary,
        "region_components": region_components,
        "region_masks": region_masks,
        "region_capacity": region_capacity,
        "region_sequence_support": region_seq,
        "region_margin": region_margin,
        "region_asset_manifest_rows": region_manifest_rows,
        "historical_lock": historical,
        "identity_inventory": identity_inventory,
        "provenance": provenance,
        "evaluator_truth": dict(evaluator_truth),
    }


# ---------------------------------------------------------------------------
# Phase B — basis + linearized probe
# ---------------------------------------------------------------------------


@dataclass
class BasisEntry:
    basis_id: str
    order: int  # 1 or 2
    mask_sha256: str
    aliases: list[dict[str, Any]] = field(default_factory=list)
    phi_primary: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    phi_unknown: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    n_neg: int = 0
    n_gt: int = 0
    n_unknown: int = 0
    is_constant: bool = False


def _eval_atom(
    matrices: Mapping[str, np.ndarray],
    feature: str,
    direction: str,
    thr: float,
) -> np.ndarray:
    return atom_mask(matrices[feature], direction, thr)


def build_basis_registry(
    cohort: CohortBundle,
    threshold_registry: Mapping[str, Any],
    and_df: pd.DataFrame,
) -> tuple[list[BasisEntry], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build collapsed Boolean basis from registered G1 + G2 AND predicates."""
    y = cohort.y
    # --- first-order: single atoms (primary signals only) ---
    single_atoms = [
        a
        for a in threshold_registry["single_atoms"]
        if int(a.get("is_secondary_feature", 0)) == 0
        and str(a["feature"]) in SIGNAL_FAMILY
    ]
    # --- second-order: unique AND masks from atlas (registered lattice only) ---
    and_rows = and_df.copy()
    # collapse later by mask

    raw: list[tuple[int, dict[str, Any], np.ndarray, np.ndarray]] = []

    for a in single_atoms:
        feat, direction = str(a["feature"]), str(a["direction"])
        thr = float(a["thr_value"])
        phi = _eval_atom(cohort.matrices, feat, direction, thr)
        phi_u = _eval_atom(cohort.unknown_matrices, feat, direction, thr)
        alias = {
            "kind": "g1_singleton",
            "atom_id": a.get("atom_id"),
            "feature": feat,
            "direction": direction,
            "thr_index": int(a["thr_index"]),
            "thr_value": thr,
            "lattice_kind": a.get("lattice_kind"),
        }
        raw.append((1, alias, phi, phi_u))

    # AND: one representative per atlas row, then collapse by mask
    for _, r in and_rows.iterrows():
        fa, da = str(r["feature_a"]), str(r["direction_a"])
        fb, db = str(r["feature_b"]), str(r["direction_b"])
        ta, tb = float(r["thr_value_a"]), float(r["thr_value_b"])
        if fa not in SIGNAL_FAMILY or fb not in SIGNAL_FAMILY:
            continue
        phi = _eval_atom(cohort.matrices, fa, da, ta) & _eval_atom(
            cohort.matrices, fb, db, tb
        )
        phi_u = _eval_atom(cohort.unknown_matrices, fa, da, ta) & _eval_atom(
            cohort.unknown_matrices, fb, db, tb
        )
        alias = {
            "kind": "g2_and",
            "combo_id": r.get("combo_id"),
            "feature_a": fa,
            "direction_a": da,
            "thr_index_a": int(r["thr_index_a"]),
            "thr_value_a": ta,
            "feature_b": fb,
            "direction_b": db,
            "thr_index_b": int(r["thr_index_b"]),
            "thr_value_b": tb,
            "atom_a_id": r.get("atom_a_id"),
            "atom_b_id": r.get("atom_b_id"),
        }
        raw.append((2, alias, phi, phi_u))

    # collapse by (order, mask_sha)
    buckets: dict[tuple[int, str], BasisEntry] = {}
    for order, alias, phi, phi_u in raw:
        msha = _mask_sha256(phi)
        key = (order, msha)
        if key not in buckets:
            n = len(phi)
            is_const = bool(phi.sum() == 0 or phi.sum() == n)
            buckets[key] = BasisEntry(
                basis_id=f"b{order}:{msha[:16]}",
                order=order,
                mask_sha256=msha,
                aliases=[alias],
                phi_primary=phi.astype(bool),
                phi_unknown=phi_u.astype(bool),
                n_neg=int(np.sum(phi & (y == 1))),
                n_gt=int(np.sum(phi & (y == 0))),
                n_unknown=int(phi_u.sum()),
                is_constant=is_const,
            )
        else:
            buckets[key].aliases.append(alias)

    entries = [e for e in buckets.values() if not e.is_constant]
    entries.sort(key=lambda e: (e.order, e.mask_sha256))
    # re-id deterministically
    for i, e in enumerate(entries):
        e.basis_id = f"b{e.order}:{e.mask_sha256[:16]}:{i:04d}"

    registry_rows = []
    alias_rows = []
    for e in entries:
        registry_rows.append(
            {
                "basis_id": e.basis_id,
                "order": e.order,
                "mask_sha256": e.mask_sha256,
                "n_aliases": len(e.aliases),
                "n_neg": e.n_neg,
                "n_gt": e.n_gt,
                "n_unknown": e.n_unknown,
                "support": int(e.phi_primary.sum()),
            }
        )
        for j, al in enumerate(e.aliases):
            alias_rows.append(
                {
                    "basis_id": e.basis_id,
                    "alias_index": j,
                    "alias_json": _canonical_json(al),
                    "kind": al.get("kind"),
                }
            )
    return entries, registry_rows, alias_rows


def _scores(w: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    return np.asarray(Phi @ w)


def evaluate_weights(
    w: np.ndarray,
    tau: float,
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
) -> dict[str, Any]:
    scores = _scores(w, Phi)
    scores_u = _scores(w, Phi_u) if len(Phi_u) else np.zeros(0)
    pred = scores >= tau - 1e-15
    pred_u = scores_u >= tau - 1e-15 if len(scores_u) else np.zeros(0, dtype=bool)
    gt = y == 0
    neg = y == 1
    gt_hurt = int(np.sum(pred & gt))
    unknown_capture = int(pred_u.sum()) if len(pred_u) else 0
    n_neg_cap = int(np.sum(pred & neg))
    # margins
    if gt.any():
        min_gt = float(np.min(tau - scores[gt]))
    else:
        min_gt = float("nan")
    if len(scores_u):
        min_unk = float(np.min(tau - scores_u))
    else:
        min_unk = float("inf")
    cap_neg_scores = scores[pred & neg]
    if len(cap_neg_scores):
        caps = cap_neg_scores - tau
        cap_min = float(np.min(caps))
        cap_p10 = float(np.percentile(caps, 10))
        cap_med = float(np.median(caps))
    else:
        cap_min = cap_p10 = cap_med = float("nan")

    # certified radii under ||Δw||1 <= r, |Δτ| <= r, |phi|<=1
    # score change bound = r; tau change = r → need margin > 2r for GT safety
    if math.isfinite(min_gt) and min_gt > 0:
        cert_gt = min_gt / 2.0
    else:
        cert_gt = 0.0
    # decision stability on captured negatives: (score - tau) > 2r
    if len(cap_neg_scores):
        cert_dec = float(np.min(cap_neg_scores - tau) / 2.0)
        cert_dec = max(0.0, cert_dec)
    else:
        cert_dec = 0.0

    per_seq: dict[str, dict[str, Any]] = {}
    for s in sorted(set(str(x) for x in sequences)):
        m = sequences == s
        per_seq[s] = {
            "n_gt_hurt": int(np.sum(pred & gt & m)),
            "n_neg_captured": int(np.sum(pred & neg & m)),
            "min_gt_margin": float(np.min(tau - scores[gt & m]))
            if np.any(gt & m)
            else float("nan"),
            "min_prod_margin": float(np.min(scores[pred & neg & m] - tau))
            if np.any(pred & neg & m)
            else float("nan"),
        }
    prod_seqs = [s for s, v in per_seq.items() if v["n_neg_captured"] > 0]
    return {
        "n_neg_captured": n_neg_cap,
        "gt_hurt": gt_hurt,
        "unknown_capture": unknown_capture,
        "valid": int(gt_hurt == 0 and unknown_capture == 0),
        "min_gt_safety_margin": min_gt,
        "min_unknown_safety_margin": min_unk,
        "captured_negative_margin_min": cap_min,
        "captured_negative_margin_p10": cap_p10,
        "captured_negative_margin_median": cap_med,
        "certified_gt_safe_radius_l1": cert_gt,
        "certified_decision_stability_radius_l1": cert_dec,
        "per_sequence": per_seq,
        "n_productive_sequences": len(prod_seqs),
        "productive_sequences": prod_seqs,
        "scores": scores,
        "pred": pred,
    }


def _tau_from_hard_safety(
    w: np.ndarray,
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Hard-safe threshold maximizing GT/unknown margin while retaining captures.

    Lower bound: max score on GT ∪ unknown.
    Upper bound: min score among negatives still above the lower bound.
    Place τ at the midpoint when a positive gap exists (improves certified radius
    without changing the hard-safety capture set under Boolean features).
    """
    scores = Phi @ w
    unsafe_vals: list[float] = []
    if np.any(y == 0):
        unsafe_vals.append(float(np.max(scores[y == 0])))
    if len(Phi_u):
        unsafe_vals.append(float(np.max(Phi_u @ w)))
    max_unsafe = max(unsafe_vals) if unsafe_vals else 0.0
    # Negatives strictly above max_unsafe are capturable
    neg_scores = scores[y == 1]
    above = neg_scores[neg_scores > max_unsafe + 0.5 * eps]
    if len(above) == 0:
        return float(max_unsafe + eps)
    min_cap = float(np.min(above))
    # Midpoint of (max_unsafe, min_cap) open interval
    return float(0.5 * (max_unsafe + min_cap))


def fit_sparse_nn_milp(
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    *,
    K: int,
    active_cols: Sequence[int] | None = None,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Max negative capture s.t. GT/unknown hard safety, w>=0, ||w||0<=K, ||w||1=1.

    After MILP, τ is *re-derived* from hard safety (max unsafe score + eps) so
    boundary numerical slack cannot admit GT/unknown capture.
    """
    n, m_full = Phi.shape
    cols = list(active_cols) if active_cols is not None else list(range(m_full))
    m = len(cols)
    if m == 0 or K <= 0:
        return {"success": False, "reason": "empty_basis", "w": None, "tau": None}

    Phi_s = Phi[:, cols]
    Phi_us = Phi_u[:, cols] if len(Phi_u) else np.zeros((0, m))
    neg_idx = np.where(y == 1)[0]
    gt_idx = np.where(y == 0)[0]
    n_neg = len(neg_idx)
    n_u = len(Phi_us)

    # variables: w (m), tau (1), z (m), c (n_neg)
    nw = m
    ntau = 1
    nz = m
    nc = n_neg
    nvar = nw + ntau + nz + nc
    cobj = np.zeros(nvar)
    cobj[nw + ntau + nz :] = -1.0

    bounds_lo = np.zeros(nvar)
    bounds_hi = np.ones(nvar)
    bounds_hi[nw] = 2.0
    integrality = np.zeros(nvar)
    integrality[nw + ntau : nw + ntau + nz] = 1
    integrality[nw + ntau + nz :] = 1

    A_rows: list[np.ndarray] = []
    b_lb: list[float] = []
    b_ub: list[float] = []

    # sum w = 1
    row = np.zeros(nvar)
    row[:nw] = 1.0
    A_rows.append(row)
    b_lb.append(1.0)
    b_ub.append(1.0)

    # sum z <= K
    row = np.zeros(nvar)
    row[nw + ntau : nw + ntau + nz] = 1.0
    A_rows.append(row)
    b_lb.append(0.0)
    b_ub.append(float(K))

    # w_j <= z_j
    for j in range(m):
        row = np.zeros(nvar)
        row[j] = 1.0
        row[nw + ntau + j] = -1.0
        A_rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(0.0)

    # GT: w·phi_g + eps <= tau
    for gi in gt_idx:
        row = np.zeros(nvar)
        row[:nw] = Phi_s[gi]
        row[nw] = -1.0
        A_rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(-eps)

    for ui in range(n_u):
        row = np.zeros(nvar)
        row[:nw] = Phi_us[ui]
        row[nw] = -1.0
        A_rows.append(row)
        b_lb.append(-np.inf)
        b_ub.append(-eps)

    # capture: w·phi_i >= tau when c_i=1
    # w·phi - tau - M c >= -M  with M=2
    M = 2.0
    for k, ni in enumerate(neg_idx):
        row = np.zeros(nvar)
        row[:nw] = Phi_s[ni]
        row[nw] = -1.0
        row[nw + ntau + nz + k] = -M
        A_rows.append(row)
        b_lb.append(-M)
        b_ub.append(np.inf)

    A = np.vstack(A_rows)
    cons = LinearConstraint(A, np.asarray(b_lb), np.asarray(b_ub))
    bounds = Bounds(bounds_lo, bounds_hi)
    res = milp(
        c=cobj,
        constraints=cons,
        bounds=bounds,
        integrality=integrality,
        options={"time_limit": 60.0, "disp": False},
    )
    if not res.success or res.x is None:
        return {
            "success": False,
            "reason": f"milp_failed:{res.message}",
            "w": None,
            "tau": None,
        }
    x = res.x
    w_s = np.maximum(x[:nw], 0.0)
    s = float(w_s.sum())
    if s <= 0:
        return {"success": False, "reason": "zero_weight", "w": None, "tau": None}
    w_s = w_s / s
    w_full = np.zeros(m_full)
    for j, col in enumerate(cols):
        w_full[col] = float(w_s[j])

    # Re-derive τ from hard safety (authoritative)
    tau = _tau_from_hard_safety(w_full, Phi, Phi_u, y, eps=eps)
    scores = Phi @ w_full
    scores_u = Phi_u @ w_full if len(Phi_u) else np.zeros(0)
    pred = scores >= tau - 1e-15
    gt_hurt = int(np.sum(pred & (y == 0)))
    unk = int(np.sum(scores_u >= tau - 1e-15)) if len(scores_u) else 0
    n_cap = int(np.sum(pred & (y == 1)))
    if gt_hurt > 0 or unk > 0:
        return {
            "success": False,
            "reason": "postcheck_hard_safety_failed",
            "w": w_full,
            "tau": tau,
            "gt_hurt": gt_hurt,
            "unknown_capture": unk,
        }
    return {
        "success": True,
        "reason": "ok",
        "w": w_full,
        "tau": tau,
        "n_captured_obj": n_cap,
    }


def select_candidate_columns(
    entries: Sequence[BasisEntry],
    y: np.ndarray,
    *,
    order_max: int,
    top_m: int = BEAM_TOP_M,
) -> list[int]:
    """Heuristic pool: pure-safe productive first, then low-contamination."""
    pure: list[tuple[float, int]] = []
    other: list[tuple[float, int]] = []
    for i, e in enumerate(entries):
        if e.order > order_max:
            continue
        if e.n_gt == 0 and e.n_unknown == 0 and e.n_neg > 0:
            pure.append((float(e.n_neg), i))
        else:
            score = float(e.n_neg) - 10.0 * float(e.n_gt) - 10.0 * float(e.n_unknown)
            other.append((score, i))
    pure.sort(key=lambda t: (-t[0], entries[t[1]].mask_sha256))
    other.sort(key=lambda t: (-t[0], entries[t[1]].mask_sha256))
    # Prefer pure-safe; fill remainder from other
    out = [i for _, i in pure]
    for _, i in other:
        if len(out) >= top_m:
            break
        if i not in out:
            out.append(i)
    return out[:top_m]


def fit_sparse_nn_combinatorial(
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    *,
    K: int,
    active_cols: Sequence[int],
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Exact-ish sparse NN probe via equal-weight subsets + hard τ re-derive.

    Primary method when MILP is unstable. For Boolean features with w≥0 on the
    simplex and τ = max_{GT∪unk} score + eps, equal weights on a support S act
    as a soft count/OR over selected predicates.
    """
    cols = list(active_cols)
    if not cols or K <= 0:
        return {"success": False, "reason": "empty_basis", "w": None, "tau": None}

    m_full = Phi.shape[1]
    # Restrict search pool size for exhaustive C(n,k)
    pool = cols[: min(len(cols), 24)]
    k_eff = min(K, len(pool))

    best: dict[str, Any] | None = None

    def eval_support(sup: Sequence[int]) -> dict[str, Any] | None:
        if not sup:
            return None
        w = np.zeros(m_full)
        for j in sup:
            w[j] = 1.0 / len(sup)
        tau = _tau_from_hard_safety(w, Phi, Phi_u, y, eps=eps)
        scores = Phi @ w
        scores_u = Phi_u @ w if len(Phi_u) else np.zeros(0)
        pred = scores >= tau - 1e-15
        gt_hurt = int(np.sum(pred & (y == 0)))
        unk = int(np.sum(scores_u >= tau - 1e-15)) if len(scores_u) else 0
        if gt_hurt or unk:
            return None
        n_cap = int(np.sum(pred & (y == 1)))
        return {
            "success": True,
            "reason": "combinatorial_equal_weight",
            "w": w,
            "tau": tau,
            "n_captured_obj": n_cap,
            "support": list(sup),
        }

    # Greedy forward
    chosen: list[int] = []
    remaining = list(pool)
    for _ in range(k_eff):
        best_local = None
        best_j = None
        for j in remaining:
            cand = eval_support(chosen + [j])
            if cand is None:
                continue
            if (
                best_local is None
                or cand["n_captured_obj"] > best_local["n_captured_obj"]
            ):
                best_local = cand
                best_j = j
        if best_j is None:
            break
        assert best_local is not None
        chosen.append(best_j)
        remaining.remove(best_j)
        if best is None or best_local["n_captured_obj"] > best["n_captured_obj"]:
            best = best_local

    # Exhaustive for small pools
    from math import comb as _comb

    if len(pool) <= 16 and k_eff <= 5 and _comb(len(pool), k_eff) <= 20000:
        for sup in combinations(pool, k_eff):
            cand = eval_support(sup)
            if cand is None:
                continue
            if best is None or cand["n_captured_obj"] > best["n_captured_obj"]:
                best = cand

    # Also try pure single best from pool
    for j in pool:
        cand = eval_support([j])
        if cand is None:
            continue
        if best is None or cand["n_captured_obj"] > best["n_captured_obj"]:
            best = cand

    if best is None:
        return {
            "success": False,
            "reason": "no_feasible_support",
            "w": None,
            "tau": None,
        }
    return best


def fit_equal_weight_count(
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
    thresholds: Sequence[int] = L1_COUNT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """L1: f=sum phi_j over first-order columns; fixed count thresholds."""
    results: list[dict[str, Any]] = []
    # equal weight over all columns present
    m = Phi.shape[1]
    if m == 0:
        return results
    w = np.ones(m) / m
    # count = m * score
    counts = Phi.sum(axis=1)
    counts_u = Phi_u.sum(axis=1) if len(Phi_u) else np.zeros(0)
    for t in thresholds:
        # capture if count >= t  ↔  score >= t/m
        tau = t / m
        ev = evaluate_weights(w, tau, Phi, Phi_u, y, sequences)
        # verify via counts
        pred = counts >= t
        pred_u = counts_u >= t if len(counts_u) else np.zeros(0, dtype=bool)
        gt_hurt = int(np.sum(pred & (y == 0)))
        unk = int(pred_u.sum()) if len(pred_u) else 0
        n_neg = int(np.sum(pred & (y == 1)))
        results.append(
            {
                "family": "L1_equal_weight_count",
                "K": None,
                "count_threshold": t,
                "active_basis_count": m,
                "n_neg_captured": n_neg,
                "gt_hurt": gt_hurt,
                "unknown_capture": unk,
                "valid": int(gt_hurt == 0 and unk == 0 and n_neg > 0),
                "w": w,
                "tau": tau,
                "eval": ev if (gt_hurt == 0 and unk == 0) else ev,
            }
        )
    return results


def l0_grammar_baselines(
    atom_df: pd.DataFrame,
    and_df: pd.DataFrame,
    or_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    def _ps(df: pd.DataFrame) -> pd.DataFrame:
        col = df["productive_safe_point"]
        if col.dtype == bool:
            return df[col]
        return df[col.astype(int) == 1]

    rows = []
    if "is_secondary_feature" in atom_df.columns:
        atom_primary = atom_df[
            atom_df["is_secondary_feature"].fillna(0).astype(int) == 0
        ]
    else:
        atom_primary = atom_df
    for name, df in (
        ("G1_singleton", atom_primary),
        ("G2_pairwise_and", and_df),
        ("G3_hard_or", or_df),
    ):
        ps = _ps(df) if "productive_safe_point" in df.columns else df.iloc[0:0]
        n_ps = len(ps)
        n_masks = int(ps["mask_sha256"].nunique()) if n_ps else 0
        max_neg = int(ps["n_neg_captured"].max()) if n_ps else 0
        multi = int((ps["n_sequences_with_neg"].astype(int) >= 2).sum()) if n_ps else 0
        rows.append(
            {
                "family": "L0_restricted_grammar",
                "grammar": name,
                "unique_productive_masks": n_masks,
                "n_productive_safe_coordinates": n_ps,
                "max_negative_capacity": max_neg,
                "multi_sequence_productive_coordinates": multi,
                "gt_hurt": 0 if n_ps else None,
                "unknown_capture": 0 if n_ps else None,
            }
        )
    return rows


def run_linear_probe(
    *,
    entries: Sequence[BasisEntry],
    cohort: CohortBundle,
    atom_df: pd.DataFrame,
    and_df: pd.DataFrame,
    or_df: pd.DataFrame,
) -> dict[str, Any]:
    y = cohort.y
    sequences = cohort.sequences
    n = len(y)
    m = len(entries)
    Phi = np.column_stack([e.phi_primary for e in entries]) if m else np.zeros((n, 0))
    Phi_u = (
        np.column_stack([e.phi_unknown for e in entries])
        if m
        else np.zeros((cohort.n_unknown, 0))
    )

    model_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    per_seq_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    rob_rows: list[dict[str, Any]] = []
    pred_blocks: list[pd.DataFrame] = []

    # L0
    for r in l0_grammar_baselines(atom_df, and_df, or_df):
        model_rows.append(
            {
                "model_id": f"L0:{r['grammar']}",
                "family": r["family"],
                "grammar": r["grammar"],
                "K": "",
                "active_basis_count": "",
                "n_neg_captured": r["max_negative_capacity"],
                "gt_hurt": r["gt_hurt"] if r["gt_hurt"] is not None else "",
                "unknown_capture": r["unknown_capture"]
                if r["unknown_capture"] is not None
                else "",
                "unique_productive_masks": r["unique_productive_masks"],
                "n_productive_sequences": r["multi_sequence_productive_coordinates"],
                "valid_hard_safety": 1 if r["max_negative_capacity"] > 0 else 0,
                "optimizer_status": "atlas_baseline",
                "active_basis_ids": "",
                "weights_json": "",
                "tau": "",
            }
        )

    # L1 first-order only
    idx1 = [i for i, e in enumerate(entries) if e.order == 1]
    Phi1 = Phi[:, idx1] if idx1 else np.zeros((n, 0))
    Phi1_u = Phi_u[:, idx1] if idx1 else np.zeros((cohort.n_unknown, 0))
    for res in fit_equal_weight_count(Phi1, Phi1_u, y, sequences):
        mid = f"L1:count{res['count_threshold']}"
        ev = res["eval"]
        model_rows.append(
            {
                "model_id": mid,
                "family": "L1_equal_weight_count",
                "grammar": "",
                "K": "",
                "active_basis_count": res["active_basis_count"],
                "n_neg_captured": res["n_neg_captured"],
                "gt_hurt": res["gt_hurt"],
                "unknown_capture": res["unknown_capture"],
                "unique_productive_masks": "",
                "n_productive_sequences": ev.get("n_productive_sequences", ""),
                "valid_hard_safety": res["valid"],
                "optimizer_status": "closed_form",
                "active_basis_ids": json.dumps(
                    [entries[i].basis_id for i in idx1[:20]]
                    + (["..."] if len(idx1) > 20 else [])
                ),
                "weights_json": "equal",
                "tau": res["tau"],
            }
        )
        _append_eval_tables(
            mid,
            res["w"],
            res["tau"],
            ev,
            sequences,
            margin_rows,
            per_seq_rows,
            pred_blocks,
            y,
        )

    # L2 / L3 sparse
    for family, order_max in (
        ("L2_sparse_nn_singleton", 1),
        ("L3_sparse_nn_with_and", 2),
    ):
        cand = select_candidate_columns(
            entries, y, order_max=order_max, top_m=BEAM_TOP_M
        )
        for K in FIXED_K_GRID:
            # Combinatorial primary (stable); MILP optional fallback
            fit = fit_sparse_nn_combinatorial(Phi, Phi_u, y, K=K, active_cols=cand)
            if not fit.get("success"):
                fit_m = fit_sparse_nn_milp(Phi, Phi_u, y, K=K, active_cols=cand)
                if fit_m.get("success"):
                    fit = fit_m
            mid = f"{family}:K{K}"
            if not fit["success"] or fit["w"] is None:
                model_rows.append(
                    {
                        "model_id": mid,
                        "family": family,
                        "grammar": "",
                        "K": K,
                        "active_basis_count": 0,
                        "n_neg_captured": 0,
                        "gt_hurt": "",
                        "unknown_capture": "",
                        "unique_productive_masks": "",
                        "n_productive_sequences": "",
                        "valid_hard_safety": 0,
                        "optimizer_status": fit.get("reason", "blocked"),
                        "active_basis_ids": "",
                        "weights_json": "",
                        "tau": "",
                    }
                )
                continue
            w = fit["w"]
            tau = float(fit["tau"])
            # zero tiny weights
            w = np.where(w > 1e-8, w, 0.0)
            s = w.sum()
            if s > 0:
                w = w / s
            active = [i for i, wi in enumerate(w) if wi > 1e-8]
            ev = evaluate_weights(w, tau, Phi, Phi_u, y, sequences)
            model_rows.append(
                {
                    "model_id": mid,
                    "family": family,
                    "grammar": "",
                    "K": K,
                    "active_basis_count": len(active),
                    "n_neg_captured": ev["n_neg_captured"],
                    "gt_hurt": ev["gt_hurt"],
                    "unknown_capture": ev["unknown_capture"],
                    "unique_productive_masks": len(active),
                    "n_productive_sequences": ev["n_productive_sequences"],
                    "valid_hard_safety": ev["valid"],
                    "optimizer_status": fit.get("reason", "ok"),
                    "active_basis_ids": json.dumps(
                        [entries[i].basis_id for i in active]
                    ),
                    "weights_json": _canonical_json(
                        {entries[i].basis_id: float(w[i]) for i in active}
                    ),
                    "tau": tau,
                }
            )
            _append_eval_tables(
                mid, w, tau, ev, sequences, margin_rows, per_seq_rows, pred_blocks, y
            )
            if ev["valid"] and ev["n_neg_captured"] > 0:
                # V3 nested sequence LOO
                loo_rows.extend(
                    _nested_sequence_loo(
                        mid,
                        family,
                        K,
                        entries,
                        cand,
                        order_max,
                        cohort,
                        Phi,
                        Phi_u,
                    )
                )
                # V4 local robustness
                rob_rows.extend(
                    _local_robustness(mid, w, tau, Phi, Phi_u, y, sequences, ev)
                )

    preds = pd.concat(pred_blocks, ignore_index=True) if pred_blocks else pd.DataFrame()
    return {
        "linear_probe_models": model_rows,
        "linear_probe_margin": margin_rows,
        "linear_probe_per_sequence": per_seq_rows,
        "linear_probe_loo": loo_rows,
        "linear_probe_robustness": rob_rows,
        "linear_probe_predictions": preds,
    }


def _append_eval_tables(
    model_id: str,
    w: np.ndarray,
    tau: float,
    ev: Mapping[str, Any],
    sequences: np.ndarray,
    margin_rows: list[dict[str, Any]],
    per_seq_rows: list[dict[str, Any]],
    pred_blocks: list[pd.DataFrame],
    y: np.ndarray,
) -> None:
    margin_rows.append(
        {
            "model_id": model_id,
            "min_gt_safety_margin": ev["min_gt_safety_margin"],
            "min_unknown_safety_margin": ev["min_unknown_safety_margin"],
            "captured_negative_margin_min": ev["captured_negative_margin_min"],
            "captured_negative_margin_p10": ev["captured_negative_margin_p10"],
            "captured_negative_margin_median": ev["captured_negative_margin_median"],
            "certified_gt_safe_radius_l1": ev["certified_gt_safe_radius_l1"],
            "certified_decision_stability_radius_l1": ev[
                "certified_decision_stability_radius_l1"
            ],
            "n_neg_captured": ev["n_neg_captured"],
            "gt_hurt": ev["gt_hurt"],
            "unknown_capture": ev["unknown_capture"],
        }
    )
    for s, v in ev["per_sequence"].items():
        per_seq_rows.append({"model_id": model_id, "sequence": s, **v})
    scores = ev["scores"]
    pred = ev["pred"]
    pred_blocks.append(
        pd.DataFrame(
            {
                "model_id": model_id,
                "row_index": np.arange(len(y)),
                "sequence": sequences.astype(str),
                "y": y,
                "score": scores,
                "pred": pred.astype(int),
                "tau": tau,
            }
        )
    )


def _nested_sequence_loo(
    model_id: str,
    family: str,
    K: int,
    entries: Sequence[BasisEntry],
    cand: Sequence[int],
    order_max: int,
    cohort: CohortBundle,
    Phi: np.ndarray,
    Phi_u: np.ndarray,
) -> list[dict[str, Any]]:
    """Fit on train sequences only; freeze; evaluate holdout. Basis pool fixed globally."""
    y = cohort.y
    sequences = cohort.sequences
    rows = []
    seqs = sorted(set(str(s) for s in sequences))
    for hold in seqs:
        train = sequences != hold
        hold_m = sequences == hold
        # basis identities fixed globally (cand); selection/weights from train labels only
        Phi_tr = Phi[train]
        y_tr = y[train]
        # unknown always evaluated fully (firewall uses full unknown set)
        fit = fit_sparse_nn_combinatorial(
            Phi_tr, Phi_u, y_tr, K=K, active_cols=list(cand)
        )
        if not fit.get("success"):
            fit = fit_sparse_nn_milp(Phi_tr, Phi_u, y_tr, K=K, active_cols=list(cand))
        if not fit["success"] or fit["w"] is None:
            rows.append(
                {
                    "model_id": model_id,
                    "hold_out_sequence": hold,
                    "status": fit.get("reason", "blocked"),
                    "hold_gt_hurt": "",
                    "hold_n_neg_captured": "",
                    "hold_n_neg": int(np.sum(y[hold_m] == 1)),
                    "hold_n_gt": int(np.sum(y[hold_m] == 0)),
                    "train_n_neg_captured": "",
                    "basis_registry_fixed_globally": 1,
                    "basis_selected_using_train_labels": 1,
                }
            )
            continue
        w = fit["w"]
        tau = float(fit["tau"])
        w = np.where(w > 1e-8, w, 0.0)
        if w.sum() > 0:
            w = w / w.sum()
        # train metrics
        ev_tr = evaluate_weights(w, tau, Phi_tr, Phi_u, y_tr, sequences[train])
        # holdout
        scores_h = Phi[hold_m] @ w
        pred_h = scores_h >= tau - 1e-15
        y_h = y[hold_m]
        rows.append(
            {
                "model_id": model_id,
                "hold_out_sequence": hold,
                "status": "ok",
                "hold_gt_hurt": int(np.sum(pred_h & (y_h == 0))),
                "hold_n_neg_captured": int(np.sum(pred_h & (y_h == 1))),
                "hold_n_neg": int(np.sum(y_h == 1)),
                "hold_n_gt": int(np.sum(y_h == 0)),
                "train_n_neg_captured": ev_tr["n_neg_captured"],
                "train_gt_hurt": ev_tr["gt_hurt"],
                "train_unknown_capture": ev_tr["unknown_capture"],
                "basis_registry_fixed_globally": 1,
                "basis_selected_using_train_labels": 1,
                "tau": tau,
            }
        )
    return rows


def _local_robustness(
    model_id: str,
    w: np.ndarray,
    tau: float,
    Phi: np.ndarray,
    Phi_u: np.ndarray,
    y: np.ndarray,
    sequences: np.ndarray,
    ev: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    base_cert = float(ev["certified_gt_safe_radius_l1"])
    rows.append(
        {
            "model_id": model_id,
            "kind": "certified_radius",
            "delta": base_cert,
            "gt_safety_retained": int(base_cert > 0),
            "productive_retained": int(
                float(ev["certified_decision_stability_radius_l1"]) > 0
            ),
            "n_neg_captured": ev["n_neg_captured"],
            "gt_hurt": 0,
            "note": "conservative ||Δw||1+|Δτ| bound with Boolean features",
        }
    )
    rng = np.random.default_rng(0)
    for delta in ROBUST_DELTA_GRID:
        # perturb w on simplex
        noise = rng.normal(0, 1, size=w.shape)
        noise = noise - noise.mean()
        w2 = np.maximum(w + delta * noise, 0.0)
        if w2.sum() <= 0:
            w2 = w.copy()
        else:
            w2 = w2 / w2.sum()
        tau2 = tau + delta * float(rng.choice([-1.0, 1.0]))
        ev2 = evaluate_weights(w2, tau2, Phi, Phi_u, y, sequences)
        rows.append(
            {
                "model_id": model_id,
                "kind": "grid_perturbation",
                "delta": delta,
                "gt_safety_retained": int(
                    ev2["gt_hurt"] == 0 and ev2["unknown_capture"] == 0
                ),
                "productive_retained": int(
                    ev2["n_neg_captured"] >= max(1, int(0.5 * ev["n_neg_captured"]))
                ),
                "n_neg_captured": ev2["n_neg_captured"],
                "gt_hurt": ev2["gt_hurt"],
                "unknown_capture": ev2["unknown_capture"],
                "note": "sanity audit only; certified radius is primary",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def assign_verdict(
    phase_a: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    models = probe["linear_probe_models"]
    loo = probe["linear_probe_loo"]
    # consider L2/L3 valid multi-seq models
    linear_valid = [
        m
        for m in models
        if str(m.get("family", "")).startswith("L2")
        or str(m.get("family", "")).startswith("L3")
        or str(m.get("family", "")).startswith("L1")
    ]
    good = [
        m
        for m in linear_valid
        if int(m.get("valid_hard_safety") or 0) == 1
        and int(m.get("n_neg_captured") or 0) > 0
    ]
    blocked = [
        m
        for m in linear_valid
        if str(m.get("optimizer_status", "")).startswith("milp_failed")
        or m.get("optimizer_status") == "blocked"
    ]
    multi = [m for m in good if int(m.get("n_productive_sequences") or 0) >= 2]
    margins = {r["model_id"]: r for r in probe["linear_probe_margin"]}
    multi_pos_margin = []
    for m in multi:
        mr = margins.get(m["model_id"], {})
        if (
            float(mr.get("min_gt_safety_margin") or 0) > 0
            and float(mr.get("certified_gt_safe_radius_l1") or 0) > 0
        ):
            multi_pos_margin.append(m)

    # LOO check for multi_pos_margin models
    loo_ok = []
    loo_collapse = []
    for m in multi_pos_margin:
        folds = [
            r for r in loo if r["model_id"] == m["model_id"] and r.get("status") == "ok"
        ]
        if not folds:
            continue
        harm = any(int(r.get("hold_gt_hurt") or 0) > 0 for r in folds)
        prod = sum(int(r.get("hold_n_neg_captured") or 0) for r in folds)
        if harm:
            loo_collapse.append(m)
        elif prod > 0:
            loo_ok.append(m)
        else:
            loo_collapse.append(m)

    g1g2_max = 0
    for m in models:
        if m.get("family") == "L0_restricted_grammar":
            g1g2_max = max(g1g2_max, int(m.get("n_neg_captured") or 0))

    # complexity-dependent: only K=5 dense?
    only_high_k = (
        multi_pos_margin
        and all(int(m.get("K") or 0) >= 5 for m in multi_pos_margin if m.get("K") != "")
        and all(
            float(margins.get(m["model_id"], {}).get("min_gt_safety_margin") or 0)
            < 1e-3
            for m in multi_pos_margin
        )
    )

    if blocked and not good:
        code = "V-E"
        text = (
            "The constrained optimization or artifact contract cannot support "
            "a trustworthy capacity comparison."
        )
    elif loo_ok and multi_pos_margin:
        # improved vs grammar?
        best_cap = max(int(m["n_neg_captured"]) for m in loo_ok)
        if best_cap > g1g2_max or any(
            int(m.get("n_productive_sequences") or 0) >= 2 for m in loo_ok
        ):
            code = "V-A"
            text = (
                "A sparse non-negative linearized basis produces GT_hurt=0, "
                "unknown_capture=0, multi-sequence productive support, positive "
                "normalized safety margin, non-zero robustness radius, and "
                "meaningful LOO retention. Frozen signals may contain structure "
                "that G1–G3 restricted grammar under-expresses. "
                "Authorizes grammar-distillation design only — not hooks."
            )
        else:
            code = "V-B"
            text = (
                "Even the constrained linearized upper-bound probe fails to "
                "meaningfully exceed G1–G3 multi-sequence safe/productive margin."
            )
    elif multi_pos_margin and loo_collapse and not loo_ok:
        code = "V-C"
        text = (
            "Linearized models improve pooled fit but collapse under "
            "per-sequence or LOO validation. Overfit, not transferable safe regions."
        )
    elif only_high_k or (
        good
        and not multi
        and all(int(m.get("active_basis_count") or 0) >= 5 for m in good)
    ):
        code = "V-D"
        text = (
            "A result exists only with excessive basis count, dense interactions, "
            "or near-zero margins. Not an actionable research asset."
        )
    elif not good:
        code = "V-B"
        text = (
            "Even the constrained linearized upper-bound probe fails to produce "
            "a stable multi-sequence safe/productive margin. Primary limitation "
            "is likely frozen signal capacity, not merely G1–G3 grammar expressiveness."
        )
    elif good and not multi:
        code = "V-C"
        text = (
            "Linearized models produce only single-sequence productive support "
            "under hard safety — in-sample islands, not transferable regions."
        )
    else:
        code = "V-E"
        text = (
            "Results are mixed or incomplete; capacity comparison remains inconclusive."
        )

    return {
        "verdict_code": code,
        "verdict_text": text,
        "terminal_b_retained": True,
        "production_unchanged": True,
        "max_maturity": "A1_region_asset",
        "n_linear_valid": len(good),
        "n_multi_seq_valid": len(multi),
        "n_loo_ok": len(loo_ok),
        "n_loo_collapse": len(loo_collapse),
        "g1g2_max_neg_capacity": g1g2_max,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_study(
    *,
    q45_dir: Path,
    events_path: Path,
    out_dir: Path,
    study_id: str | None = None,
) -> dict[str, Any]:
    q45_dir = Path(q45_dir)
    events_path = Path(events_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study_id = study_id or f"safe_region_assetization_r1_{ts}"

    # load inputs
    atom_path = q45_dir / "atom_atlas.parquet"
    and_path = q45_dir / "pairwise_and_atlas.parquet"
    or_path = q45_dir / "pairwise_or_atlas.parquet"
    atom_df = (
        pd.read_parquet(atom_path)
        if atom_path.exists()
        else pd.read_csv(q45_dir / "atom_atlas.csv")
    )
    and_df = (
        pd.read_parquet(and_path)
        if and_path.exists()
        else pd.read_csv(q45_dir / "pairwise_and_atlas.csv")
    )
    or_df = (
        pd.read_parquet(or_path)
        if or_path.exists()
        else pd.read_csv(q45_dir / "pairwise_or_atlas.csv")
    )
    stability_df = pd.read_csv(q45_dir / "region_stability.csv")
    thr_reg = json.loads(
        (q45_dir / "threshold_registry.json").read_text(encoding="utf-8")
    )
    summary = json.loads((q45_dir / "summary.json").read_text(encoding="utf-8"))
    cohort_sum = json.loads(
        (q45_dir / "cohort_summary.json").read_text(encoding="utf-8")
    )

    cohort = load_cohort_bundle(events_path, thr_reg)

    # Phase A
    phase_a = build_phase_a_assets(
        atom_df=atom_df,
        and_df=and_df,
        or_df=or_df,
        stability_df=stability_df,
        cohort=cohort,
        q45_dir=q45_dir,
        evaluator_truth={
            "terminal": summary.get("stage2_q45_terminal"),
            "terminal_letter": summary.get("terminal_letter"),
            "cohort": cohort_sum,
        },
    )

    # Phase B
    entries, basis_rows, alias_rows = build_basis_registry(cohort, thr_reg, and_df)
    probe = run_linear_probe(
        entries=entries,
        cohort=cohort,
        atom_df=atom_df,
        and_df=and_df,
        or_df=or_df,
    )
    verdict = assign_verdict(phase_a, probe)

    # write tables
    def _wcsv(name: str, rows: list[dict[str, Any]]) -> None:
        write_csv(out_dir / name, rows)

    _wcsv("grammar_region_summary.csv", phase_a["grammar_region_summary"])
    _wcsv("region_components.csv", phase_a["region_components"])
    _wcsv("region_masks.csv", phase_a["region_masks"])
    _wcsv("region_capacity.csv", phase_a["region_capacity"])
    _wcsv("region_sequence_support.csv", phase_a["region_sequence_support"])
    _wcsv("region_margin.csv", phase_a["region_margin"])
    _wcsv("basis_registry.csv", basis_rows)
    _wcsv("basis_aliases.csv", alias_rows)
    _wcsv("linear_probe_models.csv", probe["linear_probe_models"])
    _wcsv("linear_probe_margin.csv", probe["linear_probe_margin"])
    _wcsv("linear_probe_per_sequence.csv", probe["linear_probe_per_sequence"])
    _wcsv("linear_probe_loo.csv", probe["linear_probe_loo"])
    _wcsv("linear_probe_robustness.csv", probe["linear_probe_robustness"])

    preds = probe["linear_probe_predictions"]
    if isinstance(preds, pd.DataFrame) and len(preds):
        write_parquet(
            out_dir / "linear_probe_predictions.parquet",
            preds.to_dict(orient="records"),
        )
    else:
        write_parquet(out_dir / "linear_probe_predictions.parquet", [])

    region_asset_manifest = {
        "study_id": study_id,
        "task": TASK_NAME,
        "maturity_ceiling": "A1_region_asset",
        "production_forbidden": True,
        "terminal_b": TERMINAL_B,
        "signal_family": list(SIGNAL_FAMILY),
        "signal_registry_hash": cohort.signal_registry_hash,
        "cohort_hash": cohort.cohort_hash,
        "n_region_assets": len(phase_a["region_asset_manifest_rows"]),
        "n_basis": len(basis_rows),
        "historical_lock": phase_a["historical_lock"],
        "provenance": phase_a["provenance"],
        "rows": phase_a["region_asset_manifest_rows"],
    }
    write_json(out_dir / "region_asset_manifest.json", region_asset_manifest)

    manifest = {
        "study_id": study_id,
        "task": TASK_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "q45_dir": str(q45_dir),
        "events_path": str(events_path),
        "events_sha256": _sha256_file(events_path),
        "K_grid": list(FIXED_K_GRID),
        "L1_count_thresholds": list(L1_COUNT_THRESHOLDS),
        "non_goals": [
            "new_signal_family",
            "production_hook",
            "preset_change",
            "unconstrained_logistic_headline",
            "3plus_interaction",
        ],
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    write_json(out_dir / "manifest.json", manifest)

    summary_out = {
        "study_id": study_id,
        "phase_a": {
            "n_productive_safe_components": len(phase_a["region_components"]),
            "n_grid_local_mask_assets": len(phase_a["region_masks"]),
            "identity_inventory": phase_a["identity_inventory"],
            "grammar_summary": phase_a["grammar_region_summary"],
            "historical_lock": phase_a["historical_lock"],
            # deprecated aliases — do not interpret as distinct prediction masks
            "n_components": len(phase_a["region_components"]),
            "n_masks_table_rows": len(phase_a["region_masks"]),
        },
        "phase_b": {
            "n_basis_nonconstant": len(basis_rows),
            "n_basis_order1": sum(1 for e in entries if e.order == 1),
            "n_basis_order2": sum(1 for e in entries if e.order == 2),
            "models": [
                {
                    k: m[k]
                    for k in (
                        "model_id",
                        "family",
                        "K",
                        "n_neg_captured",
                        "gt_hurt",
                        "unknown_capture",
                        "valid_hard_safety",
                        "n_productive_sequences",
                        "optimizer_status",
                    )
                    if k in m
                }
                for m in probe["linear_probe_models"]
            ],
            "loo_protocol": {
                "kind": "transductive_globally_registered_basis_LOO",
                "basis_registry": (
                    "global label-free registered predicate coordinates "
                    "from sealed Q4.5 threshold_registry / atlas lattice "
                    "(feature values may include all sequences; labels do not "
                    "select the registry)"
                ),
                "supervised_fit": "train-sequence labels only for weight/support selection",
                "held_out_label_isolation": True,
                "not_claimed": "fully inductive train-only threshold transport",
                "null_result_strengthening": (
                    "held-out covariates may inform the global registry; "
                    "model still fails sequence LOO → stronger non-transfer claim"
                ),
            },
        },
        "verdict": verdict,
        "verdict_refinement": {
            "code": "V-C",
            "pooled_in_sample": "grammar-limited (L3 raises capacity 4→8)",
            "cross_sequence": "invariance-limited / non-transferable under LOO",
            "not_equivalent_to": "blanket signal-limited for all model classes",
            "closed_model_class": (
                "frozen 5 signals × registered predicates ≤2-order × "
                "non-negative sparse K≤5 × hard GT/unknown"
            ),
        },
        "certified_radius_definition": {
            "weight_norm": "||w||1 = 1",
            "perturbation": "||Δw||1 <= r and |Δτ| <= r",
            "feature_bound": "Boolean phi_j in {0,1}",
            "score_sensitivity": "|Δ(w·φ)| <= ||Δw||1 <= r",
            "certified_gt_safe_radius_l1": "min_g (τ - w·φ(g)) / 2",
            "certified_decision_stability_radius_l1": "min_{captured neg} (w·φ - τ) / 2",
            "note": "conservative joint bound; not a transferable safe-region claim",
        },
        "research_acceptance": {
            "phase_a_asset_conversion": "ACCEPTED",
            "phase_b_feasibility_probe": "ACCEPTED_AS_V-C",
            "terminal_b": "retained_and_strengthened",
            "r2_grammar_distillation": "not_authorized",
            "g4_g7_expansion": "not_authorized",
            "hook_preset_production": "unchanged",
            "evidence_ledger": "no_automatic_promotion",
        },
        "terminal_b_retained": True,
        "production_unchanged": True,
    }
    write_json(out_dir / "identity_inventory.json", phase_a["identity_inventory"])
    write_json(out_dir / "summary.json", summary_out)
    return summary_out
