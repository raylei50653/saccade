# T0-B Existing Atlas Region Interpretation Pack

<!-- doc-status: active -->
<!-- doc-promotion: none; not evidence_ledger; bounded descriptive geometry only -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: this file for T0-B derived geometry interpretation -->

**Task:** T0-B (authorized after accepted T0-A)  
**Branch:** `research/m-b1-5-t0-region-interpretation`  
**Authorize:** `4c347281` · base T0-A tip `f1981c12`  
**Thread:** [composition_grammar_safe_region.md](../../../research/threads/composition_grammar_safe_region.md)  
**Preflight (accepted):** [composition_grammar_t0_artifact_preflight_20260710.md](composition_grammar_t0_artifact_preflight_20260710.md)

> Descriptive geometry of the **existing** registered Q4.5 atlas.  
> **Not** portable safe-region proof · **not** production candidate · **not** G7 equivalence · **not** evaluator rerun.

---

## 1. Provenance

| Item | Value |
|:--|:--|
| Input study | `out/signal_study/m_b1_5_stage2_q45_20260710/` |
| Derivation script | `scripts/tools/analyze_m_b1_5_t0_region_interpretation.py` |
| Runtime outputs | `out/signal_study/m_b1_5_t0_region_interpretation_20260710/` |
| Committed evidence pack | [evidence/m_b1_5_t0_region_interpretation_20260710/](evidence/m_b1_5_t0_region_interpretation_20260710/) |
| Input atlas hashes | match Q4.5 `manifest.json` (`artifact_reconciliation.json`) |
| Headline reconcile | **PASS** — 154 = 1 G1 + 153 G2 + 0 G3 |
| `per_sequence` PS cross-check | **PASS** |
| Synthetic dual-margin checks | **PASS** (isolated / strip / 3×3 / diagonal / edge) |

```text
Terminal B:     isolated_safe_points_only (unchanged)
Production:     unchanged
evidence_ledger: not promoted
```

### Fixed dual-margin policy (locked)

| Metric | Definition |
|:--|:--|
| `nearest_unsafe_distance` | Same registered grid only; graph distance to any **non-productive-safe** registered coordinate |
| `distance_to_lattice_edge` | Min steps to lattice boundary |
| `nearest_unsafe_edge_censored` | True if on-lattice unsafe was not found / search exhausted |
| `full_neighborhood_safe_radius` | G1 bilateral interval; G2 Manhattan / repeated 4-neigh erosion |
| Edge policy | **Conservative:** missing off-lattice neighbor ⇒ radius **0** (edge-touching coords cannot claim thickness) |

Edge-censored distance is **not** region thickness.

### Units (do not collapse)

```text
coordinate unit  = registered thr_index cell within one grid
mask unit        = mask_sha256 within one registered grid (primary)
component unit   = 4-neigh (G2) / bilateral (G1) connected PS set within one grid
never join S:: ↔ P:: by atom id
never global-collapse mask_sha256 for primary ratios
```

---

## 2. Deliverables map

| # | Output | Machine artifact |
|:--|:--|:--|
| 1 | Raw-coordinate area | `grammar_area_summary.csv` |
| 2 | Per-grid unique-mask area | `unique_mask_summary.csv` |
| 3 | Productive capacity | `productive_capacity.csv` · `productive_capacity_by_per_grid_mask.csv` |
| 4 | Cross-seq productive support | `cross_sequence_productive_support.csv` |
| 5 | Component shape / axis degeneracy | `component_geometry.csv` |
| 6 | Dual boundary margin | `boundary_margin.csv` |
| 7 | G7 **contract-gap** (not equivalence) | `g7_contract_gap.json` · optional `non_g7_mask_overlap.json` |

---

## 3. Headline geometry (descriptive)

| Quantity | Value |
|:--|:--|
| Productive-safe coordinates | **154** (1 + 153 + 0) |
| Single-sequence productive support | **142** |
| Multi-sequence productive support | **12** (all G2 AND) |
| G1 coordinate productive ratio | 1 / 870 ≈ **0.00115** |
| G2 AND coordinate productive ratio | 153 / 17640 ≈ **0.00867** |
| G3 OR productive | **0** |
| G2 unique PS masks (per-grid micro sum) | **33** |
| G2 unique PS masks (global string, diagnostic only) | **15** |
| Coordinates on multi-coord plateaus (per-grid mask) | **143** / 154 |
| Distinct per-grid masks with plateau width > 1 | **23** |
| Reconstructed components | **26** |
| Shape classes | row_strip 10 · column_strip 2 · isolated_point 7 · 2d_region 7 |
| Single-cell-width strip components | **12** |
| Genuine 2D-thick components (`radius≥1` somewhere) | **0** |
| `full_neighborhood_safe_radius ≥ 1` (any PS coord) | **0** / 154 |
| `nearest_unsafe_distance > 0` while radius = 0 | **154** / 154 |
| PS coords touching lattice edge | **85** / 154 |
| Multi-seq AND coords → distinct global mask strings | **4** (diagnostic; primary scope remains per-grid) |

### Area (coordinate-weighted grammar aggregates)

| Grammar | Registered coords | Observed safe | Productive safe | Productive ratio |
|:--|--:|--:|--:|--:|
| G1_atom | 870 | 1 | 1 | 0.001149 |
| G2_and | 17640 | 153 | 153 | 0.008673 |
| G3_or | 17640 | 0 | 0 | 0 |

Do **not** compare raw G1 vs G2 cell counts without lattice-context normalization.

### Capacity

- `n_neg_captured` on PS cells: almost all **1** (142 cells); max observed **4**.
- Sequence-support count equals capacity histogram under this atlas (`n_seq` 1→142, 2→6, 3→2, 4→4).
- Multi-seq AND: **12** coordinates; **not** a large multi-seq productive body.

### Components (largest)

Largest connected PS sets are **axis-degenerate strips**, e.g.:

- `resid_mean high × score_m_bridge high` — **19** cells, column strip (1×19), 1 mask  
- `abs_log_h high × resid_mean high` — **18** cells, row strip (18×1)  
- several “2d_region” labels have both axis spans ≥2 but **still radius 0** (thin / edge / no 4-neigh interior)

`region_stability.csv` quotient max component size **19** matches atlas max component size **19** (grain differs; used as cross-check only).

### Dual margin

Under the locked conservative policy:

```text
∀ 154 PS coordinates:
  nearest_unsafe_distance = 1
  full_neighborhood_safe_radius = 0
```

So every productive-safe cell is **adjacent** to a non-PS cell (or equivalent shortest path length 1), while **none** has a full bilateral / 4-neigh safe ball of radius ≥1. This is exactly the dual-metric pattern the coverage audit warned about: nearest-unsafe > 0 does **not** imply thickness.

---

## 4. G7 contract-gap report (item 7)

```text
status: not_derivable_from_current_artifact_contract
missing:
  - logical NOT / complement predicate identity
  - necessary-envelope operand role
  - support operand role
  - N/P parameterization
maximum_claim:
  existing G1/G2 mask-string overlap only; not G7 equivalence
```

Optional diagnostic: `non_g7_mask_overlap.json` (filename and labels forbid G7 equivalence reading).

---

## 5. Answers to the T0 acceptance question

> Of the **154** productive-safe cells, how many are threshold duplicates, single-sequence islands, thin strips — and is there under-reported full-neighborhood thickness?

| Explanation channel | Reading on this atlas |
|:--|:--|
| Threshold-coordinate duplication | **Dominant:** 143/154 PS coords sit on multi-coord per-grid mask plateaus (23 masks with width>1) |
| Single-sequence support | **Dominant:** 142/154; multi-seq only **12** |
| Axis-degenerate / thin components | **Dominant:** 12 strip components; largest are 1×N / N×1; 0 genuine 2D-thick |
| Edge censoring | **Material:** 85/154 touch lattice edge; radius forced 0 at edge |
| Genuine full-neighborhood thickness | **Not observed:** 0 coords with `full_neighborhood_safe_radius≥1` |

**Bounded verdict candidate (descriptive only, pending review):**

> Among 154 productive-safe coordinates, observed productivity is explained by **threshold-coordinate mask plateaus**, **single-sequence support**, and **axis-degenerate / edge-touching components**. Under the locked conservative dual-margin policy there is **no** registered full-neighborhood thickness (`radius≥1` count = 0). This does **not** promote a portable safe region, production reject policy, online parameter-region claim, or G7 result. Terminal B remains `isolated_safe_points_only`.

---

## 6. Unresolved firewall (retained)

Still **not** claimed:

```text
formal or portable safe region
online parameter-region retention
productive reject policy
production candidate
G7 equivalence
new grammar necessity
threshold-path global falsification
```

Still **open observation gaps** (from T0-A, unchanged):

- committed Q4.5 pack alone cannot regenerate these tables (needs runtime full atlases)
- `assignment_group_key_status=invalid_frame_provenance` (out of T0 scope)
- G7 remains contract-blocked until NOT / roles exist

---

## 7. Reproduce

```bash
uv run python scripts/tools/analyze_m_b1_5_t0_region_interpretation.py \
  --study out/signal_study/m_b1_5_stage2_q45_20260710 \
  --out out/signal_study/m_b1_5_t0_region_interpretation_20260710
```

Fail-closed if full atlases missing or hashes ≠ Q4.5 manifest.

---

## 8. Next gate (not authorized here)

```text
T0-B execution (this note)
→ engineering / PR review
→ bounded research acceptance
→ only then: close line / minimal emit / region-LOO design / restricted G7 contract
```
