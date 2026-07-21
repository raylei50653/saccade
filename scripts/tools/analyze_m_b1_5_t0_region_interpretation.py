#!/usr/bin/env python3
"""T0-B: Existing Atlas Region Interpretation Pack (read-only derivation).

Derives coordinate/mask area, productive capacity, cross-seq support geometry,
component shape, dual boundary margins, and a fail-closed G7 contract-gap report
from an existing Q4.5 atlas study. Never mutates inputs or reruns the evaluator.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FILES = {
    "atom_atlas": "atom_atlas.parquet",
    "pairwise_and_atlas": "pairwise_and_atlas.parquet",
    "pairwise_or_atlas": "pairwise_or_atlas.parquet",
    "region_stability": "region_stability.csv",
    "per_sequence": "per_sequence.csv",
    "threshold_registry": "threshold_registry.json",
    "summary": "summary.json",
    "manifest": "manifest.json",
}

MANIFEST_HASH_KEYS = {
    "atom_atlas.parquet": "atom_atlas_parquet",
    "pairwise_and_atlas.parquet": "pairwise_and_atlas_parquet",
    "pairwise_or_atlas.parquet": "pairwise_or_atlas_parquet",
    "region_stability.csv": "region_stability",
    "per_sequence.csv": "per_sequence",
    "threshold_registry.json": "threshold_registry",
    "summary.json": "summary",
}

HEADLINE = {
    "n_productive_safe_total": 154,
    "n_productive_safe_single": 1,
    "n_productive_safe_and": 153,
    "n_productive_safe_or": 0,
    "n_atom_rows": 870,
    "n_and_rows": 17640,
    "n_or_rows": 17640,
}

G1_THR = list(range(87))  # 0..86
G2_THR = list(range(21))  # 0..20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seq_json(raw: Any) -> dict[str, int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        obj = json.loads(raw)
        return {str(k): int(v) for k, v in obj.items()}
    fail(f"unsupported per_sequence json type: {type(raw)}")


@dataclass(frozen=True)
class GridKey:
    grammar: str  # G1_atom | G2_and | G3_or
    grid_id: str

    def as_dict(self) -> dict[str, str]:
        return {"grammar": self.grammar, "grid_id": self.grid_id}


def g1_grid_id(feature: str, direction: str) -> str:
    return f"S::{feature}::{direction}"


def g2_grid_id(
    feature_a: str, direction_a: str, feature_b: str, direction_b: str
) -> str:
    return f"P::{feature_a}::{direction_a}__{feature_b}::{direction_b}"


def validate_inputs(
    study: Path,
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths: dict[str, Path] = {}
    for key, name in REQUIRED_FILES.items():
        p = study / name
        if not p.is_file():
            fail(f"missing required input: {p}")
        paths[key] = p

    manifest = load_json(paths["manifest"])
    hashes = manifest.get("artifact_hashes") or {}
    computed: dict[str, str] = {}
    for fname, mkey in MANIFEST_HASH_KEYS.items():
        p = study / fname
        got = sha256_file(p)
        computed[fname] = got
        exp = hashes.get(mkey)
        if exp is None:
            fail(f"manifest missing hash key {mkey} for {fname}")
        if got != exp:
            fail(f"hash mismatch for {fname}: got={got} expected={exp}")
    # also hash manifest itself for provenance
    computed["manifest.json"] = sha256_file(paths["manifest"])
    return paths, computed, manifest


def reconcile_headlines(
    atom: pd.DataFrame,
    and_df: pd.DataFrame,
    or_df: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "n_atom_rows": (
            len(atom),
            HEADLINE["n_atom_rows"],
            summary.get("n_atom_atlas_rows"),
        ),
        "n_and_rows": (
            len(and_df),
            HEADLINE["n_and_rows"],
            summary.get("n_pairwise_and_rows"),
        ),
        "n_or_rows": (
            len(or_df),
            HEADLINE["n_or_rows"],
            summary.get("n_pairwise_or_rows"),
        ),
        "n_ps_single": (
            int(atom["productive_safe_point"].sum()),
            HEADLINE["n_productive_safe_single"],
            summary.get("n_productive_safe_single"),
        ),
        "n_ps_and": (
            int(and_df["productive_safe_point"].sum()),
            HEADLINE["n_productive_safe_and"],
            summary.get("n_productive_safe_and"),
        ),
        "n_ps_or": (
            int(or_df["productive_safe_point"].sum()),
            HEADLINE["n_productive_safe_or"],
            summary.get("n_productive_safe_or"),
        ),
    }
    total_ps = checks["n_ps_single"][0] + checks["n_ps_and"][0] + checks["n_ps_or"][0]
    checks["n_ps_total"] = (
        total_ps,
        HEADLINE["n_productive_safe_total"],
        summary.get("n_productive_safe_cells"),
    )

    errors: list[str] = []
    detail: dict[str, Any] = {}
    for name, (got, expected, summary_v) in checks.items():
        detail[name] = {"got": got, "expected": expected, "summary": summary_v}
        if got != expected:
            errors.append(f"{name}: got {got} expected {expected}")
        if summary_v is not None and int(summary_v) != got:
            errors.append(f"{name}: atlas {got} != summary {summary_v}")

    ok = not errors
    return {
        "ok": ok,
        "status": "PASS" if ok else "reconciliation_failed",
        "checks": detail,
        "errors": errors,
    }


def area_for_g1(atom: pd.DataFrame) -> tuple[list[dict], dict[str, Any]]:
    rows: list[dict] = []
    for (feature, direction), g in atom.groupby(["feature", "direction"], sort=True):
        denom = len(g)
        if denom != 87:
            fail(f"G1 grid incomplete {feature}/{direction}: {denom}")
        thr = set(int(x) for x in g["thr_index"])
        if thr != set(G1_THR):
            fail(f"G1 thr_index set incomplete for {feature}/{direction}")
        n_safe = int(g["observed_safe_point"].sum())
        n_ps = int(g["productive_safe_point"].sum())
        rows.append(
            {
                "grammar": "G1_atom",
                "grid_id": g1_grid_id(feature, direction),
                "feature_a": feature,
                "direction_a": direction,
                "feature_b": "",
                "direction_b": "",
                "n_registered_coords": denom,
                "n_observed_safe": n_safe,
                "n_productive_safe": n_ps,
                "coordinate_safe_area_ratio": n_safe / denom,
                "coordinate_productive_area_ratio": n_ps / denom,
            }
        )
    agg = _aggregate_area(rows, "G1_atom")
    return rows, agg


def area_for_pairwise(
    df: pd.DataFrame, grammar: str
) -> tuple[list[dict], dict[str, Any]]:
    rows: list[dict] = []
    keys = ["feature_a", "direction_a", "feature_b", "direction_b"]
    for key, g in df.groupby(keys, sort=True):
        fa, da, fb, db = key
        denom = len(g)
        if denom != 441:
            fail(f"{grammar} grid incomplete {key}: {denom}")
        thr_a = set(int(x) for x in g["thr_index_a"])
        thr_b = set(int(x) for x in g["thr_index_b"])
        if thr_a != set(G2_THR) or thr_b != set(G2_THR):
            fail(f"{grammar} thr grid incomplete {key}")
        n_safe = int(g["observed_safe_point"].sum())
        n_ps = int(g["productive_safe_point"].sum())
        rows.append(
            {
                "grammar": grammar,
                "grid_id": g2_grid_id(fa, da, fb, db),
                "feature_a": fa,
                "direction_a": da,
                "feature_b": fb,
                "direction_b": db,
                "n_registered_coords": denom,
                "n_observed_safe": n_safe,
                "n_productive_safe": n_ps,
                "coordinate_safe_area_ratio": n_safe / denom,
                "coordinate_productive_area_ratio": n_ps / denom,
            }
        )
    agg = _aggregate_area(rows, grammar)
    return rows, agg


def _aggregate_area(rows: list[dict], grammar: str) -> dict[str, Any]:
    n_coords = sum(r["n_registered_coords"] for r in rows)
    n_safe = sum(r["n_observed_safe"] for r in rows)
    n_ps = sum(r["n_productive_safe"] for r in rows)
    return {
        "grammar": grammar,
        "n_grids": len(rows),
        "n_registered_coords": n_coords,
        "n_observed_safe": n_safe,
        "n_productive_safe": n_ps,
        "coordinate_safe_area_ratio": (n_safe / n_coords) if n_coords else 0.0,
        "coordinate_productive_area_ratio": (n_ps / n_coords) if n_coords else 0.0,
        "note": "coordinate-weighted aggregate; do not compare G1 vs G2 raw counts without context",
    }


def unique_mask_for_g1(atom: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    grid_rows: list[dict] = []
    for (feature, direction), g in atom.groupby(["feature", "direction"], sort=True):
        grid_rows.append(_unique_mask_row("G1_atom", g1_grid_id(feature, direction), g))
    return grid_rows, _unique_mask_grammar_agg(grid_rows, "G1_atom")


def unique_mask_for_pairwise(
    df: pd.DataFrame, grammar: str
) -> tuple[list[dict], list[dict]]:
    grid_rows: list[dict] = []
    keys = ["feature_a", "direction_a", "feature_b", "direction_b"]
    for key, g in df.groupby(keys, sort=True):
        fa, da, fb, db = key
        grid_rows.append(_unique_mask_row(grammar, g2_grid_id(fa, da, fb, db), g))
    return grid_rows, _unique_mask_grammar_agg(grid_rows, grammar)


def _unique_mask_row(grammar: str, grid_id: str, g: pd.DataFrame) -> dict[str, Any]:
    all_masks = set(g["mask_sha256"].astype(str))
    safe_masks = set(g.loc[g["observed_safe_point"] == 1, "mask_sha256"].astype(str))
    ps_masks = set(g.loc[g["productive_safe_point"] == 1, "mask_sha256"].astype(str))
    n_all = len(all_masks)
    n_safe = len(safe_masks)
    n_ps = len(ps_masks)
    return {
        "grammar": grammar,
        "grid_id": grid_id,
        "scope": "per_registered_grid",
        "n_unique_masks_all": n_all,
        "n_unique_masks_observed_safe": n_safe,
        "n_unique_masks_productive_safe": n_ps,
        "unique_mask_safe_ratio": (n_safe / n_all) if n_all else 0.0,
        "unique_mask_productive_ratio": (n_ps / n_all) if n_all else 0.0,
        "n_coords_productive_safe": int(g["productive_safe_point"].sum()),
    }


def _unique_mask_grammar_agg(grid_rows: list[dict], grammar: str) -> list[dict]:
    # grid_scoped_micro: sum of per-grid denominators/numerators (no global dedupe)
    den = sum(r["n_unique_masks_all"] for r in grid_rows)
    safe = sum(r["n_unique_masks_observed_safe"] for r in grid_rows)
    ps = sum(r["n_unique_masks_productive_safe"] for r in grid_rows)
    return [
        {
            "grammar": grammar,
            "scope": "grid_scoped_micro",
            "n_grids": len(grid_rows),
            "n_unique_masks_all_sum": den,
            "n_unique_masks_observed_safe_sum": safe,
            "n_unique_masks_productive_safe_sum": ps,
            "unique_mask_safe_ratio_micro": (safe / den) if den else 0.0,
            "unique_mask_productive_ratio_micro": (ps / den) if den else 0.0,
            "note": "sum of per-grid unique-mask counts; NOT global mask-string dedupe",
        }
    ]


def bfs_components_1d(ps_coords: set[int]) -> list[list[int]]:
    remaining = set(ps_coords)
    comps: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        q = deque([start])
        comp = [start]
        while q:
            x = q.popleft()
            for nb in (x - 1, x + 1):
                if nb in remaining:
                    remaining.remove(nb)
                    q.append(nb)
                    comp.append(nb)
        comps.append(sorted(comp))
    return comps


def bfs_components_2d(ps_coords: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(ps_coords)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        remaining.remove(start)
        q = deque([start])
        comp = [start]
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in remaining:
                    remaining.remove(nb)
                    q.append(nb)
                    comp.append(nb)
        comps.append(sorted(comp))
    return comps


def distance_to_lattice_edge_1d(x: int, thr_min: int, thr_max: int) -> int:
    return min(x - thr_min, thr_max - x)


def distance_to_lattice_edge_2d(x: int, y: int, thr_min: int, thr_max: int) -> int:
    return min(x - thr_min, thr_max - x, y - thr_min, thr_max - y)


def nearest_unsafe_1d(
    x: int, ps: set[int], all_coords: set[int]
) -> tuple[int | None, bool]:
    """BFS to any registered non-PS coord. Returns (distance, edge_censored).

    edge_censored True if BFS exhausts without finding non-PS (only possible if
    entire connected registered component is PS — then nearest unsafe is undefined
    on-lattice and we report None with edge_censored if lattice finite).
    """
    if x not in all_coords:
        fail(f"coordinate {x} not on lattice")
    q = deque([(x, 0)])
    seen = {x}
    while q:
        cur, d = q.popleft()
        if d > 0 and cur not in ps:
            return d, False
        for nb in (cur - 1, cur + 1):
            if nb in all_coords and nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    # No non-PS found on lattice: entire grid PS or disconnected — mark censored
    return None, True


def nearest_unsafe_2d(
    coord: tuple[int, int],
    ps: set[tuple[int, int]],
    all_coords: set[tuple[int, int]],
) -> tuple[int | None, bool]:
    q = deque([(coord, 0)])
    seen = {coord}
    while q:
        (x, y), d = q.popleft()
        if d > 0 and (x, y) not in ps:
            return d, False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if nb in all_coords and nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None, True


def full_neighborhood_radius_1d(
    x: int, ps: set[int], thr_min: int, thr_max: int
) -> int:
    """Bilateral interval radius under conservative edge policy."""
    r = 0
    while True:
        nr = r + 1
        lo, hi = x - nr, x + nr
        if lo < thr_min or hi > thr_max:
            break  # missing off-lattice neighbor
        if lo not in ps or hi not in ps:
            break
        # all coordinates in [x-nr, x+nr] must be PS
        if not all(t in ps for t in range(lo, hi + 1)):
            break
        r = nr
    return r


def full_neighborhood_radius_2d(
    coord: tuple[int, int],
    ps: set[tuple[int, int]],
    thr_min: int,
    thr_max: int,
) -> int:
    """Manhattan ball / repeated 4-neigh erosion under conservative edge policy."""
    x0, y0 = coord
    r = 0
    while True:
        nr = r + 1
        ok = True
        for dx in range(-nr, nr + 1):
            for dy in range(-nr, nr + 1):
                if abs(dx) + abs(dy) > nr:
                    continue
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                if nx < thr_min or nx > thr_max or ny < thr_min or ny > thr_max:
                    ok = False
                    break
                if (nx, ny) not in ps:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
        r = nr
    return r


def classify_component_shape_1d(coords: list[int]) -> dict[str, Any]:
    span = (max(coords) - min(coords) + 1) if coords else 0
    size = len(coords)
    if size == 1:
        shape = "isolated_point"
    elif size == span:
        shape = "1d_interval"
    else:
        shape = "1d_gapped"  # should not happen for 4-neigh/bilateral components
    return {
        "shape_class": shape,
        "n_coords": size,
        "axis_span_a": span,
        "axis_span_b": 0,
        "bounding_box_a": span,
        "bounding_box_b": 1,
        "active_axis_count": 1 if size > 1 else 0,
        "is_single_cell_width_strip": False,
        "is_genuine_2d_thick": False,
    }


def classify_component_shape_2d(coords: list[tuple[int, int]]) -> dict[str, Any]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    span_a = max(xs) - min(xs) + 1
    span_b = max(ys) - min(ys) + 1
    size = len(coords)
    if size == 1:
        shape = "isolated_point"
    elif span_a == 1 and span_b > 1:
        shape = "column_strip"  # fixed thr_a, vary thr_b
    elif span_b == 1 and span_a > 1:
        shape = "row_strip"  # fixed thr_b, vary thr_a
    elif span_a > 1 and span_b > 1:
        # genuine 2D if bounding box filled enough that min(span)>1 and has interior candidate
        shape = "2d_region"
    else:
        shape = "isolated_point"

    single_cell_width = (span_a == 1 and span_b > 1) or (span_b == 1 and span_a > 1)
    # genuine 2d thick: bounding box has both spans >= 3 and at least one cell with 4-neigh radius>=1
    # (filled later with margin); for shape alone require both spans >= 2 and size >= 4
    genuine_2d = span_a >= 2 and span_b >= 2 and size >= 4 and not single_cell_width

    active = int(span_a > 1) + int(span_b > 1)
    return {
        "shape_class": shape,
        "n_coords": size,
        "axis_span_a": span_a,
        "axis_span_b": span_b,
        "bounding_box_a": span_a,
        "bounding_box_b": span_b,
        "active_axis_count": active,
        "is_single_cell_width_strip": single_cell_width,
        "is_genuine_2d_thick": genuine_2d,
    }


def synthetic_margin_checks() -> list[dict[str, Any]]:
    """Unit-style checks of dual-margin policy (no atlas I/O)."""
    results: list[dict[str, Any]] = []

    # isolated point on 0..20
    thr_min, thr_max = 0, 20
    all2 = {(i, j) for i in range(21) for j in range(21)}
    ps = {(10, 10)}
    nu, cens = nearest_unsafe_2d((10, 10), ps, all2)
    r = full_neighborhood_radius_2d((10, 10), ps, thr_min, thr_max)
    results.append(
        {
            "case": "isolated_point",
            "nearest_unsafe_distance": nu,
            "full_neighborhood_safe_radius": r,
            "pass": r == 0 and nu == 1,
        }
    )

    # one-cell-wide strip
    ps = {(10, j) for j in range(5, 16)}
    r_mid = full_neighborhood_radius_2d((10, 10), ps, thr_min, thr_max)
    nu_mid, _ = nearest_unsafe_2d((10, 10), ps, all2)
    results.append(
        {
            "case": "one_cell_wide_strip",
            "nearest_unsafe_distance": nu_mid,
            "full_neighborhood_safe_radius": r_mid,
            "pass": r_mid == 0 and (nu_mid is not None and nu_mid > 0),
        }
    )

    # 3x3 filled block center
    ps = {(x, y) for x in range(9, 12) for y in range(9, 12)}
    r_c = full_neighborhood_radius_2d((10, 10), ps, thr_min, thr_max)
    results.append(
        {
            "case": "3x3_block_center",
            "full_neighborhood_safe_radius": r_c,
            "pass": r_c >= 1,
        }
    )

    # diagonal-only disconnected
    ps = {(i, i) for i in range(5, 16)}
    comps = bfs_components_2d(ps)
    results.append(
        {
            "case": "diagonal_only_disconnected",
            "n_components": len(comps),
            "max_comp_size": max(len(c) for c in comps),
            "pass": len(comps) == len(ps) and max(len(c) for c in comps) == 1,
        }
    )

    # edge-touching strip: no artificial radius
    ps = {(0, j) for j in range(5, 16)}
    r_e = full_neighborhood_radius_2d((0, 10), ps, thr_min, thr_max)
    results.append(
        {
            "case": "edge_touching_strip_radius0",
            "full_neighborhood_safe_radius": r_e,
            "pass": r_e == 0,
        }
    )

    # nearest may be >0 while full radius = 0
    results.append(
        {
            "case": "nearest_gt0_radius0_on_strip",
            "pass": results[1]["pass"],
        }
    )

    if not all(r["pass"] for r in results):
        fail(f"synthetic margin checks failed: {results}")
    return results


def build_ps_records(
    atom: pd.DataFrame, and_df: pd.DataFrame, or_df: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, r in atom.loc[atom["productive_safe_point"] == 1].iterrows():
        records.append(
            {
                "grammar": "G1_atom",
                "cell_id": r["atom_id"],
                "grid_id": g1_grid_id(r["feature"], r["direction"]),
                "feature_a": r["feature"],
                "direction_a": r["direction"],
                "feature_b": "",
                "direction_b": "",
                "thr_index_a": int(r["thr_index"]),
                "thr_index_b": -1,
                "mask_sha256": str(r["mask_sha256"]),
                "n_neg_captured": int(r["n_neg_captured"]),
                "n_gt_captured": int(r["n_gt_captured"]),
                "neg_capture_rate": float(r["neg_capture_rate"]),
                "n_sequences_with_neg": int(r["n_sequences_with_neg"]),
                "max_neg_sequence_share": float(r["max_neg_sequence_share"])
                if pd.notna(r["max_neg_sequence_share"])
                else None,
                "single_seq_neg_dominance": int(r["single_seq_neg_dominance"]),
                "per_sequence_neg": parse_seq_json(r["per_sequence_neg_json"]),
                "per_sequence_gt": parse_seq_json(r["per_sequence_gt_json"]),
            }
        )
    for _, r in and_df.loc[and_df["productive_safe_point"] == 1].iterrows():
        records.append(
            {
                "grammar": "G2_and",
                "cell_id": r["combo_id"],
                "grid_id": g2_grid_id(
                    r["feature_a"], r["direction_a"], r["feature_b"], r["direction_b"]
                ),
                "feature_a": r["feature_a"],
                "direction_a": r["direction_a"],
                "feature_b": r["feature_b"],
                "direction_b": r["direction_b"],
                "thr_index_a": int(r["thr_index_a"]),
                "thr_index_b": int(r["thr_index_b"]),
                "mask_sha256": str(r["mask_sha256"]),
                "n_neg_captured": int(r["n_neg_captured"]),
                "n_gt_captured": int(r["n_gt_captured"]),
                "neg_capture_rate": float(r["neg_capture_rate"]),
                "n_sequences_with_neg": int(r["n_sequences_with_neg"]),
                "max_neg_sequence_share": float(r["max_neg_sequence_share"])
                if pd.notna(r["max_neg_sequence_share"])
                else None,
                "single_seq_neg_dominance": int(r["single_seq_neg_dominance"]),
                "per_sequence_neg": parse_seq_json(r["per_sequence_neg_json"]),
                "per_sequence_gt": parse_seq_json(r["per_sequence_gt_json"]),
            }
        )
    for _, r in or_df.loc[or_df["productive_safe_point"] == 1].iterrows():
        records.append(
            {
                "grammar": "G3_or",
                "cell_id": r["combo_id"],
                "grid_id": g2_grid_id(
                    r["feature_a"], r["direction_a"], r["feature_b"], r["direction_b"]
                ),
                "feature_a": r["feature_a"],
                "direction_a": r["direction_a"],
                "feature_b": r["feature_b"],
                "direction_b": r["direction_b"],
                "thr_index_a": int(r["thr_index_a"]),
                "thr_index_b": int(r["thr_index_b"]),
                "mask_sha256": str(r["mask_sha256"]),
                "n_neg_captured": int(r["n_neg_captured"]),
                "n_gt_captured": int(r["n_gt_captured"]),
                "neg_capture_rate": float(r["neg_capture_rate"]),
                "n_sequences_with_neg": int(r["n_sequences_with_neg"]),
                "max_neg_sequence_share": float(r["max_neg_sequence_share"])
                if pd.notna(r["max_neg_sequence_share"])
                else None,
                "single_seq_neg_dominance": int(r["single_seq_neg_dominance"]),
                "per_sequence_neg": parse_seq_json(r["per_sequence_neg_json"]),
                "per_sequence_gt": parse_seq_json(r["per_sequence_gt_json"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _positive_map(m: dict[str, int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in m.items() if int(v) > 0}


def cross_check_per_sequence(ps: pd.DataFrame, per: pd.DataFrame) -> dict[str, Any]:
    """Bidirectional fail-closed equality: atlas embedded maps ↔ per_sequence.csv."""
    errors: list[str] = []
    per_ps = per[per["region_id"].isin(set(ps["cell_id"]))].copy()
    checked_cells = 0
    checked_pos_entries = 0

    for _, row in ps.iterrows():
        cell = row["cell_id"]
        atlas_neg = {str(k): int(v) for k, v in (row["per_sequence_neg"] or {}).items()}
        atlas_gt = {str(k): int(v) for k, v in (row["per_sequence_gt"] or {}).items()}
        atlas_neg_pos = _positive_map(atlas_neg)
        atlas_gt_pos = _positive_map(atlas_gt)

        sub = per_ps.loc[per_ps["region_id"] == cell]
        per_neg_pos: dict[str, int] = {}
        per_gt_pos: dict[str, int] = {}
        for _, pr in sub.iterrows():
            seq = str(pr["sequence"])
            n_neg = int(pr["n_neg"])
            n_gt = int(pr["n_gt"])
            if n_neg > 0:
                per_neg_pos[seq] = n_neg
            if n_gt > 0:
                per_gt_pos[seq] = n_gt

        # missing / extra positive sequences (neg)
        for seq in sorted(set(atlas_neg_pos) - set(per_neg_pos)):
            errors.append(f"missing positive sequence (n_neg) {cell} {seq}")
        for seq in sorted(set(per_neg_pos) - set(atlas_neg_pos)):
            errors.append(f"extra positive sequence (n_neg) {cell} {seq}")

        # missing / extra positive sequences (gt)
        for seq in sorted(set(atlas_gt_pos) - set(per_gt_pos)):
            errors.append(f"missing positive sequence (n_gt) {cell} {seq}")
        for seq in sorted(set(per_gt_pos) - set(atlas_gt_pos)):
            errors.append(f"extra positive sequence (n_gt) {cell} {seq}")

        # per-seq value equality on union of positive keys
        for seq in sorted(set(atlas_neg_pos) | set(per_neg_pos)):
            a = atlas_neg_pos.get(seq)
            b = per_neg_pos.get(seq)
            if a is not None and b is not None and a != b:
                errors.append(
                    f"n_neg mismatch {cell} {seq}: atlas={a} per_sequence={b}"
                )
            checked_pos_entries += 1
        for seq in sorted(set(atlas_gt_pos) | set(per_gt_pos)):
            a = atlas_gt_pos.get(seq)
            b = per_gt_pos.get(seq)
            if a is not None and b is not None and a != b:
                errors.append(f"n_gt mismatch {cell} {seq}: atlas={a} per_sequence={b}")

        # sums vs cell-level captures (use atlas maps; must also match per_sequence sums)
        sum_atlas_neg = sum(atlas_neg.values())
        sum_atlas_gt = sum(atlas_gt.values())
        sum_per_neg = int(sub["n_neg"].sum()) if len(sub) else 0
        sum_per_gt = int(sub["n_gt"].sum()) if len(sub) else 0
        n_neg_cap = int(row["n_neg_captured"])
        n_gt_cap = int(row["n_gt_captured"])

        if sum_atlas_neg != n_neg_cap:
            errors.append(
                f"sum(seq n_neg) != n_neg_captured {cell}: sum_atlas={sum_atlas_neg} n_neg_captured={n_neg_cap}"
            )
        if sum_atlas_gt != n_gt_cap:
            errors.append(
                f"sum(seq n_gt) != n_gt_captured {cell}: sum_atlas={sum_atlas_gt} n_gt_captured={n_gt_cap}"
            )
        if sum_per_neg != n_neg_cap:
            errors.append(
                f"sum(per_sequence n_neg) != n_neg_captured {cell}: sum_per={sum_per_neg} n_neg_captured={n_neg_cap}"
            )
        if sum_per_gt != n_gt_cap:
            errors.append(
                f"sum(per_sequence n_gt) != n_gt_captured {cell}: sum_per={sum_per_gt} n_gt_captured={n_gt_cap}"
            )

        n_json = len(atlas_neg_pos)
        if n_json != int(row["n_sequences_with_neg"]):
            errors.append(
                f"n_sequences_with_neg mismatch {cell}: field={row['n_sequences_with_neg']} json_pos={n_json}"
            )
        if n_json != len(per_neg_pos):
            errors.append(
                f"n_sequences_with_neg vs per_sequence positive count {cell}: "
                f"field={row['n_sequences_with_neg']} per_pos={len(per_neg_pos)}"
            )

        checked_cells += 1

    return {
        "ok": not errors,
        "status": "PASS" if not errors else "reconciliation_failed",
        "mode": "bidirectional_equality",
        "n_ps_cells": len(ps),
        "n_cells_checked": checked_cells,
        "n_positive_neg_entries_checked": checked_pos_entries,
        "errors": errors[:80],
        "n_errors": len(errors),
    }


def assert_per_grid_mask_invariance(ps: pd.DataFrame) -> dict[str, Any]:
    """Fail closed if same (grammar, grid_id, mask) disagrees on capture / maps."""
    errors: list[str] = []
    keys = ["grammar", "grid_id", "mask_sha256"]
    n_groups = 0
    for key, g in ps.groupby(keys, sort=True):
        n_groups += 1
        if len(g) == 1:
            continue
        # compare all rows to the first
        ref = g.iloc[0]
        ref_neg = json.dumps(
            {str(k): int(v) for k, v in (ref["per_sequence_neg"] or {}).items()},
            sort_keys=True,
        )
        ref_gt = json.dumps(
            {str(k): int(v) for k, v in (ref["per_sequence_gt"] or {}).items()},
            sort_keys=True,
        )
        for _, row in g.iloc[1:].iterrows():
            if int(row["n_neg_captured"]) != int(ref["n_neg_captured"]):
                errors.append(f"mask invariance n_neg_captured {key}")
            if int(row["n_gt_captured"]) != int(ref["n_gt_captured"]):
                errors.append(f"mask invariance n_gt_captured {key}")
            if int(row["n_sequences_with_neg"]) != int(ref["n_sequences_with_neg"]):
                errors.append(f"mask invariance n_sequences_with_neg {key}")
            neg_j = json.dumps(
                {str(k): int(v) for k, v in (row["per_sequence_neg"] or {}).items()},
                sort_keys=True,
            )
            gt_j = json.dumps(
                {str(k): int(v) for k, v in (row["per_sequence_gt"] or {}).items()},
                sort_keys=True,
            )
            if neg_j != ref_neg:
                errors.append(f"mask invariance per_sequence_neg {key}")
            if gt_j != ref_gt:
                errors.append(f"mask invariance per_sequence_gt {key}")

    return {
        "ok": not errors,
        "status": "PASS" if not errors else "reconciliation_failed",
        "n_per_grid_mask_groups": n_groups,
        "n_errors": len(errors),
        "errors": errors[:40],
    }


def build_per_grid_mask_capacity(
    ps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per (grammar, grid_id, mask) after invariance is proven."""
    rows: list[dict[str, Any]] = []
    for (grammar, grid_id, mask), g in ps.groupby(
        ["grammar", "grid_id", "mask_sha256"], sort=True
    ):
        r0 = g.iloc[0]
        neg_pos = _positive_map(r0["per_sequence_neg"] or {})
        min_seq = min(neg_pos, key=neg_pos.get) if neg_pos else ""
        min_n = min(neg_pos.values()) if neg_pos else 0
        rows.append(
            {
                "grammar": grammar,
                "grid_id": grid_id,
                "mask_sha256": mask,
                "n_coords": len(g),
                "mask_n_neg": int(r0["n_neg_captured"]),
                "mask_n_gt": int(r0["n_gt_captured"]),
                "n_sequences_with_neg": int(r0["n_sequences_with_neg"]),
                "min_positive_sequence": min_seq,
                "min_positive_sequence_n_neg": min_n,
                "productive_sequences_json": json.dumps(neg_pos, sort_keys=True),
                "note": (
                    "mask_n_neg is per prediction mask unit "
                    "(not multiplied by coordinate plateau width)"
                ),
            }
        )
    mask_cap = pd.DataFrame(rows)
    # concentration on per-grid mask units using mask_n_neg (not coord-inflated sum)
    total = int(mask_cap["mask_n_neg"].sum()) if len(mask_cap) else 0
    ordered = mask_cap.sort_values(
        ["mask_n_neg", "grammar", "grid_id", "mask_sha256"],
        ascending=[False, True, True, True],
    )
    vals = ordered["mask_n_neg"].tolist()

    def top_share(k: int) -> float:
        if total <= 0:
            return 0.0
        return float(sum(vals[:k]) / total)

    dist = Counter(int(x) for x in mask_cap["mask_n_neg"].tolist())
    concentration = {
        "unit": "per_registered_grid_mask",
        "denominator": "sum of mask_n_neg over productive per-grid mask units",
        "n_productive_per_grid_mask_units": int(len(mask_cap)),
        "sum_mask_n_neg": total,
        "max_mask_n_neg": int(max(vals)) if vals else 0,
        "mask_n_neg_distribution": {str(k): int(v) for k, v in sorted(dist.items())},
        "top1_share_of_sum_mask_n_neg": top_share(1),
        "top3_share_of_sum_mask_n_neg": top_share(3),
        "top5_share_of_sum_mask_n_neg": top_share(5),
        "note": "global mask-string collapse is diagnostic only; primary unit is per-grid",
    }
    return mask_cap, concentration


def interpret(study: Path, out_dir: Path, script_path: Path) -> dict[str, Any]:
    synthetic = synthetic_margin_checks()

    paths, input_hashes, manifest = validate_inputs(study)
    summary = load_json(paths["summary"])
    registry = load_json(paths["threshold_registry"])

    atom = pd.read_parquet(paths["atom_atlas"])
    and_df = pd.read_parquet(paths["pairwise_and_atlas"])
    or_df = pd.read_parquet(paths["pairwise_or_atlas"])
    region = pd.read_csv(paths["region_stability"])
    per = pd.read_csv(paths["per_sequence"])

    recon = reconcile_headlines(atom, and_df, or_df, summary)
    if not recon["ok"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "artifact_reconciliation.json").write_text(
            json.dumps(recon, indent=2), encoding="utf-8"
        )
        fail(f"reconciliation_failed: {recon['errors']}")

    # --- 1 raw coordinate area ---
    g1_area, g1_agg = area_for_g1(atom)
    g2_area, g2_agg = area_for_pairwise(and_df, "G2_and")
    g3_area, g3_agg = area_for_pairwise(or_df, "G3_or")
    grammar_area = pd.DataFrame(g1_area + g2_area + g3_area)
    grammar_area_agg = pd.DataFrame([g1_agg, g2_agg, g3_agg])

    # --- 2 unique-mask area ---
    um_g1, um_g1_agg = unique_mask_for_g1(atom)
    um_g2, um_g2_agg = unique_mask_for_pairwise(and_df, "G2_and")
    um_g3, um_g3_agg = unique_mask_for_pairwise(or_df, "G3_or")
    unique_mask = pd.DataFrame(
        um_g1 + um_g2 + um_g3 + um_g1_agg + um_g2_agg + um_g3_agg
    )

    # --- PS records ---
    ps = build_ps_records(atom, and_df, or_df)
    if len(ps) != 154:
        fail(f"PS record count {len(ps)} != 154")

    pseq_check = cross_check_per_sequence(ps, per)
    if not pseq_check["ok"]:
        fail(f"per_sequence cross-check failed: {pseq_check['errors'][:5]}")

    mask_inv = assert_per_grid_mask_invariance(ps)
    if not mask_inv["ok"]:
        fail(f"per-grid mask invariance failed: {mask_inv['errors'][:5]}")

    # --- 3 productive capacity ---
    cap_rows: list[dict[str, Any]] = []
    for _, r in ps.iterrows():
        neg = r["per_sequence_neg"]
        seq_pos = _positive_map(neg or {})
        min_seq = min(seq_pos, key=seq_pos.get) if seq_pos else ""
        min_cap = min(seq_pos.values()) if seq_pos else 0
        cap_rows.append(
            {
                "grammar": r["grammar"],
                "cell_id": r["cell_id"],
                "grid_id": r["grid_id"],
                "mask_sha256": r["mask_sha256"],
                "n_neg_captured": r["n_neg_captured"],
                "n_gt_captured": r["n_gt_captured"],
                "neg_capture_rate": r["neg_capture_rate"],
                "n_sequences_with_neg": r["n_sequences_with_neg"],
                "max_neg_sequence_share": r["max_neg_sequence_share"],
                "single_seq_neg_dominance": r["single_seq_neg_dominance"],
                "min_positive_sequence": min_seq,
                "min_positive_sequence_n_neg": min_cap,
                "productive_sequences_json": json.dumps(seq_pos, sort_keys=True),
            }
        )
    capacity = pd.DataFrame(cap_rows)

    mask_cap, capacity_concentration = build_per_grid_mask_capacity(ps)

    # --- 4 cross-sequence ---
    xs_rows: list[dict[str, Any]] = []
    for _, r in capacity.iterrows():
        xs_rows.append(
            {
                "unit": "coordinate",
                "grammar": r["grammar"],
                "cell_id": r["cell_id"],
                "grid_id": r["grid_id"],
                "mask_sha256": r["mask_sha256"],
                "n_sequences_with_productive_support": r["n_sequences_with_neg"],
                "support_class": (
                    "multi_sequence"
                    if r["n_sequences_with_neg"] >= 2
                    else "single_sequence"
                    if r["n_sequences_with_neg"] == 1
                    else "none"
                ),
                "min_positive_sequence": r["min_positive_sequence"],
                "min_positive_sequence_n_neg": r["min_positive_sequence_n_neg"],
                "max_neg_sequence_share": r["max_neg_sequence_share"],
                "productive_sequences_json": r["productive_sequences_json"],
            }
        )
    # per-grid-mask units (maps already proven identical within group)
    for _, r in mask_cap.iterrows():
        xs_rows.append(
            {
                "unit": "per_grid_mask",
                "grammar": r["grammar"],
                "cell_id": "",
                "grid_id": r["grid_id"],
                "mask_sha256": r["mask_sha256"],
                "n_sequences_with_productive_support": r["n_sequences_with_neg"],
                "support_class": (
                    "multi_sequence"
                    if r["n_sequences_with_neg"] >= 2
                    else "single_sequence"
                    if r["n_sequences_with_neg"] == 1
                    else "none"
                ),
                "min_positive_sequence": r["min_positive_sequence"],
                "min_positive_sequence_n_neg": r["min_positive_sequence_n_neg"],
                "max_neg_sequence_share": None,
                "productive_sequences_json": r["productive_sequences_json"],
            }
        )
    cross_seq = pd.DataFrame(xs_rows)

    multi_and_coords = capacity[
        (capacity["grammar"] == "G2_and") & (capacity["n_sequences_with_neg"] >= 2)
    ]
    per_grid_nunique = multi_and_coords.groupby("grid_id")["mask_sha256"].nunique()
    n_primary_per_grid_masks = int(per_grid_nunique.sum())
    n_global_mask_strings = int(multi_and_coords["mask_sha256"].nunique())
    multi_and_summary = {
        "n_multi_seq_and_coordinates": int(len(multi_and_coords)),
        "n_single_seq_and_coordinates": int(
            (
                (capacity["grammar"] == "G2_and")
                & (capacity["n_sequences_with_neg"] == 1)
            ).sum()
        ),
        "n_primary_per_registered_grid_unique_masks": n_primary_per_grid_masks,
        "n_global_mask_strings_diagnostic": n_global_mask_strings,
        "per_grid_unique_mask_counts": {
            str(k): int(v) for k, v in per_grid_nunique.items()
        },
        "check_sum_per_grid_equals_primary": int(per_grid_nunique.sum())
        == n_primary_per_grid_masks,
        "note": (
            "primary unit = sum over grids of nunique(mask) within each grid "
            f"({n_primary_per_grid_masks}); global mask-string count "
            f"({n_global_mask_strings}) is diagnostic only and must not use per_grid_* names"
        ),
    }
    if n_primary_per_grid_masks != 8 or n_global_mask_strings != 4:
        # soft check against known Q4.5 multi-seq structure; fail if unexpected drift
        fail(
            "multi-seq mask unit check failed: "
            f"per_grid_primary={n_primary_per_grid_masks} expected 8; "
            f"global_diag={n_global_mask_strings} expected 4"
        )

    # --- 5 components + 6 dual margin ---
    component_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []

    # G1
    for (feature, direction), g in atom.groupby(["feature", "direction"], sort=True):
        grid_id = g1_grid_id(feature, direction)
        all_coords = set(int(x) for x in g["thr_index"])
        ps_coords = set(
            int(x) for x in g.loc[g["productive_safe_point"] == 1, "thr_index"]
        )
        mask_by_coord = {
            int(r.thr_index): str(r.mask_sha256)
            for r in g.itertuples()
            if int(r.productive_safe_point) == 1
        }
        comps = bfs_components_1d(ps_coords)
        for ci, comp in enumerate(comps):
            shape = classify_component_shape_1d(comp)
            masks = {mask_by_coord[c] for c in comp}
            cid = f"{grid_id}::comp{ci}"
            radii_1d = [full_neighborhood_radius_1d(x, ps_coords, 0, 86) for x in comp]
            component_rows.append(
                {
                    "component_id": cid,
                    "grammar": "G1_atom",
                    "grid_id": grid_id,
                    "n_coords": shape["n_coords"],
                    "n_unique_masks_in_component": len(masks),
                    "axis_span_a": shape["axis_span_a"],
                    "axis_span_b": shape["axis_span_b"],
                    "bounding_box_a": shape["bounding_box_a"],
                    "bounding_box_b": shape["bounding_box_b"],
                    "active_axis_count": shape["active_axis_count"],
                    "shape_class": shape["shape_class"],
                    "is_single_cell_width_strip": shape["is_single_cell_width_strip"],
                    "is_genuine_2d_thick": shape["is_genuine_2d_thick"],
                    "max_full_neighborhood_safe_radius": max(radii_1d)
                    if radii_1d
                    else 0,
                    "coords_json": json.dumps(comp),
                }
            )
            for x in comp:
                nu, cens = nearest_unsafe_1d(x, ps_coords, all_coords)
                edge_d = distance_to_lattice_edge_1d(x, 0, 86)
                radius = full_neighborhood_radius_1d(x, ps_coords, 0, 86)
                # cell_id
                cell = g.loc[g["thr_index"] == x, "atom_id"].iloc[0]
                margin_rows.append(
                    {
                        "grammar": "G1_atom",
                        "grid_id": grid_id,
                        "cell_id": cell,
                        "component_id": cid,
                        "thr_index_a": x,
                        "thr_index_b": -1,
                        "nearest_unsafe_distance": nu,
                        "nearest_unsafe_edge_censored": bool(cens or nu is None),
                        "distance_to_lattice_edge": edge_d,
                        "full_neighborhood_safe_radius": radius,
                        "edge_touches_lattice": edge_d == 0,
                        "mask_sha256": mask_by_coord[x],
                    }
                )

    # G2 AND
    keys = ["feature_a", "direction_a", "feature_b", "direction_b"]
    for key, g in and_df.groupby(keys, sort=True):
        fa, da, fb, db = key
        grid_id = g2_grid_id(fa, da, fb, db)
        all_coords = {
            (int(a), int(b)) for a, b in zip(g["thr_index_a"], g["thr_index_b"])
        }
        ps_g = g.loc[g["productive_safe_point"] == 1]
        ps_coords = {
            (int(a), int(b)) for a, b in zip(ps_g["thr_index_a"], ps_g["thr_index_b"])
        }
        if not ps_coords:
            continue
        mask_by_coord = {
            (int(r.thr_index_a), int(r.thr_index_b)): str(r.mask_sha256)
            for r in ps_g.itertuples()
        }
        cell_by_coord = {
            (int(r.thr_index_a), int(r.thr_index_b)): r.combo_id
            for r in ps_g.itertuples()
        }
        comps = bfs_components_2d(ps_coords)
        for ci, comp in enumerate(comps):
            shape = classify_component_shape_2d(comp)
            # refine genuine_2d_thick using radius
            masks = {mask_by_coord[c] for c in comp}
            cid = f"{grid_id}::comp{ci}"
            radii = [full_neighborhood_radius_2d(c, ps_coords, 0, 20) for c in comp]
            genuine = (
                shape["axis_span_a"] >= 2
                and shape["axis_span_b"] >= 2
                and max(radii) >= 1
            )
            component_rows.append(
                {
                    "component_id": cid,
                    "grammar": "G2_and",
                    "grid_id": grid_id,
                    "n_coords": shape["n_coords"],
                    "n_unique_masks_in_component": len(masks),
                    "axis_span_a": shape["axis_span_a"],
                    "axis_span_b": shape["axis_span_b"],
                    "bounding_box_a": shape["bounding_box_a"],
                    "bounding_box_b": shape["bounding_box_b"],
                    "active_axis_count": shape["active_axis_count"],
                    "shape_class": shape["shape_class"],
                    "is_single_cell_width_strip": shape["is_single_cell_width_strip"],
                    "is_genuine_2d_thick": genuine,
                    "max_full_neighborhood_safe_radius": max(radii) if radii else 0,
                    "coords_json": json.dumps(comp),
                }
            )
            for c in comp:
                nu, cens = nearest_unsafe_2d(c, ps_coords, all_coords)
                edge_d = distance_to_lattice_edge_2d(c[0], c[1], 0, 20)
                radius = full_neighborhood_radius_2d(c, ps_coords, 0, 20)
                margin_rows.append(
                    {
                        "grammar": "G2_and",
                        "grid_id": grid_id,
                        "cell_id": cell_by_coord[c],
                        "component_id": cid,
                        "thr_index_a": c[0],
                        "thr_index_b": c[1],
                        "nearest_unsafe_distance": nu,
                        "nearest_unsafe_edge_censored": bool(cens or nu is None),
                        "distance_to_lattice_edge": edge_d,
                        "full_neighborhood_safe_radius": radius,
                        "edge_touches_lattice": edge_d == 0,
                        "mask_sha256": mask_by_coord[c],
                    }
                )

    # G3 OR: no PS expected; still scan for safety
    for key, g in or_df.groupby(keys, sort=True):
        fa, da, fb, db = key
        ps_g = g.loc[g["productive_safe_point"] == 1]
        if len(ps_g) == 0:
            continue
        fail("unexpected G3 productive-safe cells during component pass")

    components = pd.DataFrame(component_rows)
    margins = pd.DataFrame(margin_rows)

    # region_stability reconciliation (quotient-level only)
    rs_recon = reconcile_region_stability(region, components, and_df, atom)

    # --- 7 G7 contract gap ---
    g7 = {
        "status": "not_derivable_from_current_artifact_contract",
        "missing": [
            "logical NOT / complement predicate identity",
            "necessary-envelope operand role",
            "support operand role",
            "N/P parameterization",
        ],
        "maximum_claim": (
            "existing G1/G2 mask-string overlap only; not G7 equivalence"
        ),
        "combinators_present": registry.get("combinators"),
        "not_a_g7_audit": True,
    }
    # optional non-G7 mask overlap
    g1_masks = set(atom["mask_sha256"].astype(str))
    g2_masks = set(and_df["mask_sha256"].astype(str))
    g1_ps_masks = set(
        atom.loc[atom["productive_safe_point"] == 1, "mask_sha256"].astype(str)
    )
    g2_ps_masks = set(
        and_df.loc[and_df["productive_safe_point"] == 1, "mask_sha256"].astype(str)
    )
    non_g7_overlap = {
        "label": "non_g7_mask_overlap",
        "not_g7_equivalence": True,
        "n_g1_masks": len(g1_masks),
        "n_g2_and_masks": len(g2_masks),
        "n_mask_string_overlap_all": len(g1_masks & g2_masks),
        "n_g1_ps_masks": len(g1_ps_masks),
        "n_g2_ps_masks": len(g2_ps_masks),
        "n_mask_string_overlap_ps": len(g1_ps_masks & g2_ps_masks),
        "ps_overlap_mask_sha256": sorted(g1_ps_masks & g2_ps_masks),
    }

    # --- headline geometry summary ---
    n_ps = len(ps)
    n_single_seq = int((capacity["n_sequences_with_neg"] == 1).sum())
    n_multi_seq = int((capacity["n_sequences_with_neg"] >= 2).sum())
    n_edge_touch = int(margins["edge_touches_lattice"].sum()) if len(margins) else 0
    n_radius_ge1 = (
        int((margins["full_neighborhood_safe_radius"] >= 1).sum())
        if len(margins)
        else 0
    )
    n_radius_0 = (
        int((margins["full_neighborhood_safe_radius"] == 0).sum())
        if len(margins)
        else 0
    )
    n_nu_gt0_r0 = (
        int(
            (
                (margins["full_neighborhood_safe_radius"] == 0)
                & (margins["nearest_unsafe_distance"].fillna(-1) > 0)
            ).sum()
        )
        if len(margins)
        else 0
    )
    shape_counts = (
        components["shape_class"].value_counts().to_dict() if len(components) else {}
    )
    n_strip = (
        int(components["is_single_cell_width_strip"].sum()) if len(components) else 0
    )
    n_genuine_2d = (
        int(components["is_genuine_2d_thick"].sum()) if len(components) else 0
    )
    # unique productive masks per grammar under per-grid micro sum
    g2_ps_masks_micro = int(
        unique_mask.loc[
            (unique_mask["grammar"] == "G2_and")
            & (unique_mask["scope"] == "per_registered_grid"),
            "n_unique_masks_productive_safe",
        ].sum()
    )
    g2_ps_masks_global = len(g2_ps_masks)

    # coordinate duplication: coords per mask within grid
    plateau = (
        capacity.groupby(["grammar", "grid_id", "mask_sha256"])
        .size()
        .reset_index(name="n_coords")
    )
    n_dup_coords = int(
        (plateau["n_coords"] > 1).sum()
    )  # masks with multi-coord plateaus
    n_coords_on_multi = int(plateau.loc[plateau["n_coords"] > 1, "n_coords"].sum())

    bounded_verdict_candidate = {
        "status": "candidate_only_pending_review",
        "terminal_b_unchanged": True,
        "descriptive_findings": {
            "n_productive_safe_coordinates": n_ps,
            "n_single_sequence_support": n_single_seq,
            "n_multi_sequence_support": n_multi_seq,
            "n_multi_seq_and_coordinates": multi_and_summary[
                "n_multi_seq_and_coordinates"
            ],
            "n_multi_seq_primary_per_registered_grid_unique_masks": multi_and_summary[
                "n_primary_per_registered_grid_unique_masks"
            ],
            "n_multi_seq_global_mask_strings_diagnostic": multi_and_summary[
                "n_global_mask_strings_diagnostic"
            ],
            "n_components": len(components),
            "shape_class_counts": shape_counts,
            "n_single_cell_width_strip_components": n_strip,
            "n_genuine_2d_thick_components": n_genuine_2d,
            "n_ps_coords_full_neighborhood_radius_ge1": n_radius_ge1,
            "n_ps_coords_full_neighborhood_radius_0": n_radius_0,
            "n_ps_coords_nearest_unsafe_gt0_and_radius_0": n_nu_gt0_r0,
            "n_ps_coords_touching_lattice_edge": n_edge_touch,
            "n_masks_with_multi_coord_plateau_per_grid": n_dup_coords,
            "n_coords_on_multi_coord_plateaus": n_coords_on_multi,
            "g2_unique_ps_masks_global_string_diagnostic": g2_ps_masks_global,
            "g2_unique_ps_masks_per_grid_micro_sum": g2_ps_masks_micro,
            "capacity_concentration": capacity_concentration,
        },
        "interpretation_candidate": (
            "Among 154 productive-safe coordinates, productivity is dominated by "
            "single-sequence support and axis-degenerate / edge-touching components; "
            "full_neighborhood_safe_radius>=1 count is zero under conservative edge policy. "
            "This is a descriptive geometry reading of the registered atlas only — not a "
            "portable safe region, production candidate, or G7 result."
        ),
        "forbidden_claims_retained": [
            "formal_or_portable_safe_region",
            "online_parameter_region_retention",
            "productive_reject_policy",
            "production_candidate",
            "g7_equivalence",
            "new_grammar_necessity",
            "threshold_path_global_falsification",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # write tables
    grammar_area_out = pd.concat(
        [grammar_area, grammar_area_agg.assign(grid_id="__GRAMMAR_AGG__")],
        ignore_index=True,
        sort=False,
    )
    grammar_area_out.to_csv(out_dir / "grammar_area_summary.csv", index=False)
    unique_mask.to_csv(out_dir / "unique_mask_summary.csv", index=False)
    capacity.to_csv(out_dir / "productive_capacity.csv", index=False)
    mask_cap.to_csv(out_dir / "productive_capacity_by_per_grid_mask.csv", index=False)
    cross_seq.to_csv(out_dir / "cross_sequence_productive_support.csv", index=False)
    components.to_csv(out_dir / "component_geometry.csv", index=False)
    margins.to_csv(out_dir / "boundary_margin.csv", index=False)

    (out_dir / "g7_contract_gap.json").write_text(
        json.dumps(g7, indent=2), encoding="utf-8"
    )
    (out_dir / "non_g7_mask_overlap.json").write_text(
        json.dumps(non_g7_overlap, indent=2), encoding="utf-8"
    )

    recon_out = {
        "revision": "T0-B-R1",
        "headline_reconciliation": recon,
        "per_sequence_cross_check": pseq_check,
        "per_grid_mask_invariance": mask_inv,
        "region_stability_quotient_reconciliation": rs_recon,
        "synthetic_margin_checks": synthetic,
        "multi_seq_and_summary": multi_and_summary,
        "capacity_concentration": capacity_concentration,
        "input_sha256": input_hashes,
        "study_git_commit": manifest.get("git_commit"),
        "study_schema": manifest.get("schema"),
    }
    (out_dir / "artifact_reconciliation.json").write_text(
        json.dumps(recon_out, indent=2), encoding="utf-8"
    )

    created = datetime.now(timezone.utc).isoformat()
    summary_out = {
        "task": "T0-B_existing_atlas_region_interpretation",
        "revision": "T0-B-R1",
        "created_utc": created,
        "input_study": str(study),
        "input_sha256": input_hashes,
        "capacity_concentration": capacity_concentration,
        "multi_seq_and_summary": multi_and_summary,
        "dual_margin_policy": {
            "nearest_unsafe_distance": (
                "shortest same-grid registered-lattice graph distance to a "
                "non-productive-safe coordinate"
            ),
            "distance_to_lattice_edge": "min steps to lattice boundary",
            "nearest_unsafe_edge_censored": (
                "true if nearest unsafe was not found on-lattice or search exhausted"
            ),
            "full_neighborhood_safe_radius": {
                "G1": "bilateral interval",
                "G2": "Manhattan / repeated 4-neighbor erosion",
                "edge_policy": "conservative: missing off-lattice neighbor => radius 0",
            },
            "edge_censored_is_not_region_thickness": True,
        },
        "mask_scope": "per_registered_grid",
        "g7": g7,
        "headline_geometry": bounded_verdict_candidate["descriptive_findings"],
        "bounded_verdict_candidate": bounded_verdict_candidate,
        "terminal_b": "isolated_safe_points_only",
        "production_preset": "unchanged",
        "evidence_ledger": "not_promoted",
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_out, indent=2), encoding="utf-8"
    )

    out_manifest = {
        "schema": "m_b1_5_t0_region_interpretation_manifest_v1",
        "created_utc": created,
        "script": str(script_path),
        "script_sha256": sha256_file(script_path) if script_path.is_file() else None,
        "input_study": str(study),
        "input_sha256": input_hashes,
        "source_manifest_git_commit": manifest.get("git_commit"),
        "source_evaluator_source_sha256": manifest.get("evaluator_source_sha256"),
        "artifact_sha256": {
            p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()
        },
        "reconciliation": recon["status"],
        "n_productive_safe": n_ps,
    }
    # recompute artifact hashes after writing summary (include all final files)
    # write manifest last without self-hash loop: hash all except manifest
    files_for_hash = [
        p for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json"
    ]
    out_manifest["artifact_sha256"] = {
        p.name: sha256_file(p) for p in sorted(files_for_hash)
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2), encoding="utf-8"
    )

    return summary_out


def reconcile_region_stability(
    region: pd.DataFrame,
    components: pd.DataFrame,
    and_df: pd.DataFrame,
    atom: pd.DataFrame,
) -> dict[str, Any]:
    """Quotient-level check only: do not treat region_id as coordinates."""
    notes: list[str] = []
    # component sizes from atlas reconstruction
    if len(components):
        max_comp = int(components["n_coords"].max())
        n_comp = len(components)
    else:
        max_comp, n_comp = 0, 0
    rs_max = int(region["component_size_coordinates"].max()) if len(region) else 0
    # region_stability n_interior should be 0
    n_interior_rs = (
        int(region["n_interior_coordinates"].sum())
        if "n_interior_coordinates" in region.columns
        else None
    )
    notes.append(
        "region_stability is mask-quotient grain; component sizes are comparable only loosely"
    )
    return {
        "n_region_stability_rows": len(region),
        "region_stability_max_component_size_coordinates": rs_max,
        "atlas_reconstructed_n_components": n_comp,
        "atlas_reconstructed_max_component_n_coords": max_comp,
        "region_stability_n_interior_coordinates_sum": n_interior_rs,
        "region_stability_has_any_interior": bool(
            region["has_interior_coordinate"].sum()
        )
        if "has_interior_coordinate" in region.columns
        else None,
        "notes": notes,
        "ok": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--study",
        type=Path,
        default=Path("out/signal_study/m_b1_5_stage2_q45_20260710"),
        help="Q4.5 runtime study directory (full atlases required)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: out/signal_study/m_b1_5_t0_region_interpretation_<utc>)",
    )
    args = ap.parse_args(argv)

    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = Path(f"out/signal_study/m_b1_5_t0_region_interpretation_{stamp}")

    script_path = Path(__file__).resolve()
    summary = interpret(args.study.resolve(), args.out.resolve(), script_path)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": "ok",
                "n_ps": summary["headline_geometry"]["n_productive_safe_coordinates"],
            },
            indent=2,
        )
    )
    print(
        "bounded_verdict_candidate:",
        summary["bounded_verdict_candidate"]["interpretation_candidate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
