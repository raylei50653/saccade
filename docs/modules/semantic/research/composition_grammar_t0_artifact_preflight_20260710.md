# T0-A Artifact Preflight — Q4.5 Atlas Schema & Derivability

<!-- doc-status: active -->
<!-- doc-promotion: navigation-support; not evidence_ledger; not research verdict -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: this file for T0-A schema/derivability inventory only -->

**Task:** T0-A artifact preflight only  
**Branch:** `research/m-b1-5-t0-region-preflight`  
**Authorize commit:** `6ef90ac80d52153e02b7445736ed1c15cecf8d3e`  
**Thread:** [composition_grammar_safe_region.md](../../../research/threads/closed/composition_grammar_safe_region.md)

> Answers **what can be faithfully derived** from existing artifacts.  
> **Not** seven-output implementation · **not** terminal change · **not** T0-B · **not** evaluator rerun.

---

## 1. Base commit & inspected identities

| Item | Value |
|:--|:--|
| Working branch | `research/m-b1-5-t0-region-preflight` |
| Branch HEAD (T0-A start) | `6ef90ac80d52153e02b7445736ed1c15cecf8d3e` |
| Contains `main` tip | **yes** — ancestor includes `8f7a3700` (PR #93 merge) |
| Ahead of `origin/main` | authorize commit only (preflight note is this work) |
| Machine study | `out/signal_study/m_b1_5_stage2_q45_20260710/` |
| Study `git_commit` (manifest) | `dc758e088de9fe2bfed7e2d4d458a8360a03f712` |
| Taxonomy / schema | `stage2_q45_atlas_v4` / `m_b1_5_stage2_q45_atlas_manifest_v4` |
| Evaluator (manifest path) | `src/saccade/perception/eval/d_online_stage2_q45_atlas.py` |
| Evaluator SHA256 (manifest) | `551284f88710945dc636cb4e13f2b8401948fc717e605738f84be83b9e133643` |
| Q4.5 terminal (pack README) | `isolated_safe_points_only` (**B**, unchanged) |
| Headline reconcile | `n_productive_safe_cells=154` = 1 single + 153 AND + 0 OR ✓ |
| Artifact hashes vs manifest | atom/AND/OR parquet, `region_stability`, `per_sequence`, `threshold_registry`, `summary` **match** |

```text
Do not collapse:
  authorize commit 6ef90ac8  ≠  study-producing commit dc758e08
  runtime full atlases     ≠  committed evidence pack (subset)
```

---

## 2. Artifact inventory

### 2.1 Runtime study root (full)

`out/signal_study/m_b1_5_stage2_q45_20260710/`

| File | Role | Present |
|:--|:--|:--|
| `atom_atlas.parquet` / `.csv` | G1 single-atom lattice atlas | **runtime only** (not in committed pack) |
| `pairwise_and_atlas.parquet` / `.csv` | G2 AND lattice atlas | **runtime only** |
| `pairwise_or_atlas.parquet` / `.csv` | G3 OR lattice atlas | **runtime only** |
| `per_sequence.csv` | sparse per-(cell, sequence) counts | **runtime only** |
| `region_stability.csv` | productive-safe mask-quotient topology | runtime + pack + committed |
| `threshold_registry.json` | registered lattices / atom catalogs | runtime + pack + committed |
| `summary.json` / `manifest.json` / `reconciliation.json` / `cohort_summary.json` | run identity + counts | runtime + pack + committed |
| `nested_loso_*.csv/json`, `loo.csv`, `pareto_frontier.csv` | portability / ranking support | mixed (clause summary in pack; fold detail runtime-only) |
| `evidence_pack/*` | PR audit subset + `SHA256SUMS.json` | runtime mirror of committed |

### 2.2 Committed evidence pack

`docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/`

- Byte-identical to runtime `evidence_pack/` for all `SHA256SUMS.json` entries (**verified**).
- **Does not include** full atlases (`atom_atlas*`, `pairwise_*_atlas*`) or `per_sequence.csv`.
- Pack `README.json` states large atlases remain in parent study dir; rebuild via runner if missing.

### 2.3 Supporting (referenced, not re-run)

| Path | Use for T0 |
|:--|:--|
| `out/signal_study/m_b1_5_stage2_q1q3_20260710/d_online_events.parquet` | optional join substrate; **not required** for outputs classified derivable below |
| Stage1 freeze online dir | out of T0-A scope (selected-policy online ≠ parameter-region) |

**Presence rule for T0-B:** treat runtime full study as the **required** machine input; committed pack alone is **insufficient** for atlas-derived outputs.

---

## 3. Schema / key map

### 3.1 Atom atlas (G1) — 870 rows × 50 cols

| | |
|:--|:--|
| **PK** | `atom_id` **or** `(feature, direction, thr_index)` |
| **ID form** | `S::<feature>::<direction>::u<thr_index>` |
| **Lattice** | `lattice_kind=primary_unique_boundaries`; `thr_index∈[0,86]` (87 pts) × 5 features × 2 directions |
| **Safety / productive** | `observed_safe_point`, `productive_safe_point`, `safety_status`, `n_neg_captured`, `gt_hurt`, unknown fields |
| **Mask** | `mask_sha256` (753 unique / 870 rows; **not** a global PK) |
| **Per-seq (embedded)** | `per_sequence_neg_json`, `per_sequence_gt_json`; aggregates `n_sequences_with_neg`, … |
| **Neighbor (1D partial)** | `n_adjacent_neighbors`, `n_adjacent_also_productive_safe`, `neighbor_gt_hurts`; thr subset flags |
| **Registry join** | `threshold_registry.single_atoms[]` by `atom_id` / thr fields |

### 3.2 Pairwise AND atlas (G2) — 17 640 rows × 51 cols

| | |
|:--|:--|
| **PK** | `combo_id` |
| **ID form** | `AND::P::…::P::…` |
| **Operand IDs** | `atom_a_id`, `atom_b_id` with **`P::` namespace** (pairwise quantile atoms) — **do not** join to `S::` atom atlas IDs (overlap **0**) |
| **Coordinate** | `(feature_a, direction_a, thr_index_a, feature_b, direction_b, thr_index_b)` |
| **Lattice** | `lattice_kind=primary_quantile_lattice_q05`; `thr_index_*∈[0,20]` (21 levels); **40** axis-pairs × **441** cells = 17 640 (**complete**) |
| **Safety / productive** | same flags as atom; `empty_region`; `semantic_duplicate_mask` |
| **Mask** | `mask_sha256` (6386 unique); gate: **semantic duplicate is per-grid not global** |
| **Per-seq (embedded)** | `per_sequence_neg_json` / `_gt_json` (complete for productive cells) |
| **Neighbor / interior cols** | **absent** |
| **Registry join** | `threshold_registry.pairwise_atoms[]` by `P::` atom_id |

### 3.3 Pairwise OR atlas (G3) — 17 640 rows (same schema as AND)

- Same coordinate / lattice completeness as AND; `combinator=OR`.
- `productive_safe_point` sum = **0** (matches summary).

### 3.4 Region stability — 34 rows

| | |
|:--|:--|
| **Grain** | **mask quotient** rows (`region_id` like `mask::…` / `mask::AND::…`), **not** lattice coordinates |
| **Kinds** | 1 `single_atom_quotient` + 33 `pairwise_AND_quotient` |
| **Topology fields** | `n_coordinates`, `plateau_width_*`, `safe_component_id`, `component_size_coordinates`, `component_size_unique_masks`, `has_interior_coordinate` (all 0), `n_adjacent_other_masks`, `stability_class` |
| **Join to atlas** | **not** via `region_id` ↔ `combo_id`; use `mask_sha256` + feature/direction axes (quotient view) |
| **Join to `per_sequence`** | **0** overlap on `region_id` |

### 3.5 Per-sequence output — 19 202 rows × 5 cols

| | |
|:--|:--|
| **Schema** | `region_id`, `sequence`, `n_neg`, `n_gt`, `support` |
| **`region_id` namespaces** | `S::` (atom), `AND::`, `OR::` — **not** `mask::` |
| **Coverage** | **sparse**: not every atlas cell appears (min `support` in file = 1 among present rows; many positive-support atlas cells absent) |
| **Productive-safe coverage** | **complete**: 1/1 atom PS + 153/153 AND PS present |
| **Sequences** | 7 MOT17-*-SDP |

**Implication:** multi-seq geometry for the 154 PS cells can use either embedded JSON on atlases or `per_sequence.csv`; full-lattice per-seq mass is **not** guaranteed from `per_sequence.csv` alone.

### 3.6 Threshold registry

| Field group | Content |
|:--|:--|
| Lattices | `single_lattice_kind=primary_unique_boundaries`; `pairwise_lattice_kind` + `pairwise_quantile_lattice`; 21 quantile levels for pairwise |
| Catalogs | `single_atoms` (870), `pairwise_atoms` (210 = 5×2×21) |
| Combinators | **`["AND","OR"]` only** — no `NOT` / role / polarity metadata |
| Signals | `signals_primary` (5), `secondary_features`, `directions` high/low tail |
| Provenance flag | `assignment_group_key_status=invalid_frame_provenance` (observation only; not a T0 fix) |

### 3.7 Coordinate / grid identity (explicit)

```text
G1 grid key:  (feature, direction, thr_index)           lattice: unique boundaries 0..86
G2/G3 grid key: (feature_a, direction_a, thr_index_a,
                 feature_b, direction_b, thr_index_b)   lattice: quantile q05 0..20
S:: atoms  ⟂  P:: pairwise atoms   (disjoint ID namespaces; different thr grids)
thr_value is recorded but thr_index is the registered-grid coordinate.
```

### 3.8 `mask_sha256` quotient scope

| Observation | Consequence |
|:--|:--|
| Same mask can appear on **multiple** `(feature,direction)` atom rows (109 masks) | Global mask collapse across axes is **not** coordinate identity |
| AND: 2424/6386 masks span **multiple axis-pairs** (max 40) | Cross-grid unique-mask counts require an explicit **scope** (per axis-pair vs global) |
| Evaluator gate | `semantic_duplicate_is_per_grid_not_global: true` |
| `semantic_duplicate_mask` | present on AND/OR only |

**Fail-closed:** unique-mask ratios must declare scope; default legal scope = **per registered grid** (atom: per feature×direction; pairwise: per axis-pair). Cross-grammar mask equality is allowed only as a **mask-string compare**, not as automatic cell collapse.

---

## 4. Seven-output derivability matrix

Legend:

- **D** = `directly_derivable` (column aggregates / filters only)
- **J** = `derivable_by_deterministic_join` (grid reconstruction / declared joins; no new labels)
- **P** = `partially_derivable` (subset of intended metric; gaps must stay open)
- **N** = `not_derivable_from_current_artifact_contract`

| # | T0 output | Class | Basis | Gap / constraint |
|:--|:--|:--|:--|:--|
| 1 | raw-coordinate safe/productive area ratios | **D** | `observed_safe_point` / `productive_safe_point` over registered cell counts per grammar/grid; denominators lattice-specific | Do not compare raw counts across G1 vs G2 lattices without normalization note |
| 2 | unique-mask safe/productive area ratios | **J** | `mask_sha256` + productive/safe flags; optional `semantic_duplicate_mask` | Scope must be per-grid; global quotient is **illegal without new contract** |
| 3 | productive capacity distribution | **D** | `n_neg_captured`, `n_sequences_with_neg`, embedded `per_sequence_neg_json` on PS rows | — |
| 4 | multi-sequence productive-support geometry | **J** | PS rows + `n_sequences_with_neg` / JSON / `per_sequence.csv` (PS-complete) | Multi-seq **productive** geometry yes; do **not** restate GT0∩ as the gap. Sparse `per_sequence` cannot rebuild full non-PS support maps |
| 5 | component shape / axis degeneracy | **J** | Reconstruct 4-neigh components on complete G2 grids from `productive_safe_point`; optional cross-check `region_stability.component_size_*`, plateau widths | `region_stability` is quotient-grain; coordinate components must come from atlas grids, not `region_id` lists |
| 6 | dual margin (`nearest_unsafe_distance` vs `full_neighborhood_safe_radius`) | **J** | Complete registered coords + PS flags enable BFS / erosion **if** lattice-edge semantics are fixed | Metrics **not emitted**. AND/OR lack neighbor columns. Atom has only partial neighbor diagnostics. **84/153** AND PS cells touch thr edge {0,20} → edge policy dominates radius |
| 7 | G7 semantic equivalence audit (`¬N∧P`) | **N** (G7 form) / **P** (G1–G2 mask overlap only) | Combinators are AND/OR only; **no** operand role / negation / N-vs-P metadata | Cannot audit G7 without inventing roles or recomputing unevaluated predicates. Mask-equality among existing G1/G2 cells is **not** a G7 audit |

### 4.1 Explicit verification answers

| Check | Result |
|:--|:--|
| Coordinate / registered-grid identity | **Confirmed** — separate G1 vs G2/G3 lattices; thr_index is coordinate; S:: ≠ P:: |
| `mask_sha256` quotient scope | **Per-grid legal**; cross-grid/global collapse **not** authorized by contract |
| Per-sequence productive-support representation | **Yes** on atlas JSON + PS rows in `per_sequence.csv`; sparse file is not full atlas |
| Component adjacency reconstruction | **Yes** on complete G2 (and G1 1D) PS unions; **not** from `region_stability.region_id` alone |
| Both boundary-margin metrics exact | **Conditionally yes** after declaring: (a) unsafe = non-PS on registered grid; (b) off-lattice neighbor handling for radius; (c) 4-neigh / bilateral neighborhood definition matching coverage audit |
| G7 without inventing role metadata | **No** — fail-closed **N** for G7 form |

---

## 5. Observation gaps (fail-closed)

These remain **open** under T0-A; T0-B must not paper over them:

1. **Committed pack ≠ full atlases** — pack-only environments cannot run atlas-derived T0 outputs.
2. **`per_sequence.csv` sparsity** — absent cells ≠ proven zero support; use atlas fields for cell-level productive geometry.
3. **No emitted dual-margin columns** — any margin numbers are **derived**, not evaluator-emitted; edge convention is part of the derivation contract.
4. **`region_stability` grain mismatch** — quotient/plateau view ≠ coordinate component listing.
5. **G7 / NOT grammar absent** — no role metadata; G7 audit not in contract.
6. **Cross-grid mask identity** — same `mask_sha256` across axis-pairs is not automatic semantic merge.
7. **Assignment-group key invalid** — recorded in registry/summary; out of T0 scope; do not “fix” via region geometry.
8. **No research verdict** from this note — terminal B, production, ledger promotion unchanged.

---

## 6. Minimum proposed T0-B implementation surface

**Status:** T0-B **not authorized** until this preflight is reviewed.

### In-scope if authorized (confirmed derivable)

```text
Inputs (read-only):
  out/signal_study/m_b1_5_stage2_q45_20260710/
    atom_atlas.parquet
    pairwise_and_atlas.parquet
    pairwise_or_atlas.parquet
    region_stability.csv          # cross-check only
    per_sequence.csv              # PS join only
    threshold_registry.json       # lattice identity
    summary.json / manifest.json  # headline reconcile

Derive only:
  (1) coordinate safe/productive ratios per grammar + per registered grid
  (2) unique-mask safe/productive ratios with explicit per-grid scope
  (3) productive capacity distribution on the 154 PS cells
  (4) multi-seq productive-support geometry on PS cells
      (n_seq_neg histogram; multi-seq set; single-seq islands; worst-seq capacity)
  (5) coordinate-level components on G1/G2 PS unions + axis span / degeneracy
  (6) dual margin on those components with declared edge policy
  (7) G1/G2 mask-equality table only (optional) — labeled NOT a G7 audit

Outputs: derived tables/note under research/ or out/signal_study/ derived folder;
         no evaluator edits; no ledger promotion; no terminal change.
```

### Explicitly out of T0-B (unless a later authorize widens contract)

- G7 `¬N∧P` enumeration or “semantic equivalence” claims requiring roles  
- Evaluator rerun / lattice change / threshold search / new signals / new grammar  
- Global mask quotient as sole unique-mask denominator  
- Treating `region_stability.region_id` as coordinate keys  
- Joining `P::` pairwise atoms to `S::` atom atlas by id  
- Online region sweeps / hook / preset / production changes  
- Research terminal upgrade or `evidence_ledger` promotion  

### Assumptions T0-B must **not** make

```text
- pack-only evidence is enough for atlas metrics
- mask_sha256 is a global primary key
- per_sequence.csv covers every lattice cell
- neighbor_* columns exist on AND/OR
- dual margins are already stored
- high_tail/low_tail encodes logical negation for G7
- interior=0 implies nearest_unsafe_distance=0 (false under dual metrics)
- GT0 intersection is the multi-seq productive headline
```

---

## 7. Preflight headline (for thread)

```text
T0-A PASS (schema inventory complete):
  runtime atlases hash-match manifest; committed pack is subset (no full atlases).
  G1 S:: unique-boundary 0..86 ⟂ G2/G3 P:: quantile 0..20 complete 21×21×40.
  Outputs 1–5 derivable (D/J); dual-margin J with explicit edge policy;
  G7 form N (no role/NOT metadata). Fail-closed gaps recorded.
  Next: review gate → authorize T0-B surface only if accepted.
```

---

## 8. Next gate

1. **Chat/review** of this note (derivability + gaps).  
2. If accepted: issue **bounded T0-B** task quoting §6 surface only.  
3. T0-B evidence packet → PR review → bounded verdict → only then T1 / region-LOO / G7 / close line.

**T0-B remains unauthorized after this commit until separately dispatched.**
