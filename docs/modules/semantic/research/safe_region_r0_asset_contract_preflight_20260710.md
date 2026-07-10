# R0-A Region Asset Contract Preflight

<!-- doc-status: active -->
<!-- doc-promotion: navigation-support; not evidence_ledger; not research verdict; not A1 acceptance -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: this file for R0-A field/identity/schema derivability only -->

**Task:** R0-A Region Asset Contract Preflight only  
**Branch:** `research/composition-grammar-coverage-program`  
**Program thread:** [safe_region_assetization_20260710.md](../../../research/threads/safe_region_assetization_20260710.md)  
**A0 baseline:** [composition_grammar_t0_region_interpretation_20260710.md](composition_grammar_t0_region_interpretation_20260710.md)  
**T0-A preflight:** [composition_grammar_t0_artifact_preflight_20260710.md](composition_grammar_t0_artifact_preflight_20260710.md)

> Answers **which RegionAsset fields/grains/identities are DIRECT, DERIVABLE, contract-dependent, or blocked** from accepted G1–G3 T0/C0 truth.  
> **Not** R0-B contract acceptance · **not** R1 asset generation · **not** evaluator rerun · **not** G4–G7 · **not** maturity promotion A0→A1.

```text
Research acceptance remains chat-side after review.
This note is an evidence packet, not a verdict.
```

---

## 0. Preflight headline

```text
R0-A COMPLETE (awaiting chat-side review):

  Provenance base: Q4.5 study m_b1_5_stage2_q45_20260710 + T0-B-R1 pack
  Runtime full atlases PRESENT and hash-match accepted manifest.
  Terminal B unchanged: isolated_safe_points_only.
  Accepted C0 geometry preserved (154 PS; radius≥1 = 0/154; G3 null).

  R1 scope verdict (recommendation only; not authorized):
    DETERMINISTIC A0→A1 CONVERSION IS FEASIBLE
    conditional on R0-B accepting bounded identity/schema decisions
    and on runtime atlas availability (or equivalent sealed atlas bundle).
    No evaluator rerun required for proposed A1 core fields.

  Hard blockers for silent A1 claims:
    - T0 component_id uses enumeration ordinals (::comp0) → not stable identity
    - global mask_sha256 is not a primary key (10 PS masks span multiple grids)
    - G7 roles remain unresolved (leave open)
    - transfer / intervention fields are A2+/A3, not A1 core
    - live evaluator/script tree hashes drift from study-recorded SHAs
      (pin recorded artifact hashes; do not recompute from current tree)

  Decisions required before R0-B: D1–D12 (§11).
```

---

## 1. Truth and provenance base

### 1.1 Source studies and hashes (inspected)

#### Q4.5 atlas study (machine truth for G1–G3 lattices)

| Item | Value |
|:--|:--|
| Study root | `out/signal_study/m_b1_5_stage2_q45_20260710/` |
| Study id | `m_b1_5_stage2_q45_20260710` |
| Manifest schema | `m_b1_5_stage2_q45_atlas_manifest_v4` |
| Taxonomy | `stage2_q45_atlas_v4` |
| Study-producing git commit | `dc758e088de9fe2bfed7e2d4d458a8360a03f712` |
| Evaluator (recorded path) | `src/saccade/perception/eval/d_online_stage2_q45_atlas.py` |
| Evaluator SHA256 (recorded) | `551284f88710945dc636cb4e13f2b8401948fc717e605738f84be83b9e133643` |
| Runner SHA256 (recorded) | `376b11c0bb9c85823ca6c09eb17ed3044730b3be7bc9fec53888ffde74d5815a` |
| Source event table SHA256 | `cfca3818fd8478e6e3dcb12d3976549dab8057a0f2c2e63831f8d3e3a2fffd97` |
| Terminal | **B** `isolated_safe_points_only` |
| Production preset | unchanged |

**Runtime artifact hashes vs manifest (verified this preflight):**

| Artifact | Manifest key | SHA256 (prefix…) | Match |
|:--|:--|:--|:--|
| `atom_atlas.parquet` | `atom_atlas_parquet` | `281cb22bd92daf48…` | **yes** |
| `pairwise_and_atlas.parquet` | `pairwise_and_atlas_parquet` | `ae52bd0cb799aaa8…` | **yes** |
| `pairwise_or_atlas.parquet` | `pairwise_or_atlas_parquet` | `bc1c2938775caa1e…` | **yes** |
| `region_stability.csv` | `region_stability` | `87055ef93439ae54…` | **yes** |
| `per_sequence.csv` | `per_sequence` | `5446ac659ecada38…` | **yes** |
| `threshold_registry.json` | `threshold_registry` | `d3e3197fa7812a9e…` | **yes** |
| `summary.json` | `summary` | `a88d9dcfc5a61449…` | **yes** |
| `manifest.json` | (self / T0 input) | `4213e82e4c05a052…` | **yes** |

CSV atlas companions also recorded in manifest (`atom_atlas`, `pairwise_and_atlas`, `pairwise_or_atlas`).

#### T0-B region interpretation pack (accepted A0 descriptive geometry)

| Item | Value |
|:--|:--|
| Committed pack | `docs/modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/` |
| Revision | **T0-B-R1** |
| Schema | `m_b1_5_t0_region_interpretation_manifest_v1` |
| Note | [composition_grammar_t0_region_interpretation_20260710.md](composition_grammar_t0_region_interpretation_20260710.md) |
| PR / merge | #94 · `acd8e30e` (engineering merge ≠ research acceptance; research acceptance already recorded) |
| Reconciliation | **PASS** (154 = 1+153+0) |
| Dual-margin policy | declared in pack `summary.json` / note §1 |

**Committed T0 pack file hashes:** all 12 entries in `SHA256SUMS.json` re-verified **match** (this preflight).

Key T0 output digests (full 64-hex in pack):

| File | SHA256 prefix |
|:--|:--|
| `component_geometry.csv` | `ad2dd33b64453a49…` |
| `boundary_margin.csv` | `424fe06350000311…` |
| `productive_capacity.csv` | `84898c23c9bd7611…` |
| `productive_capacity_by_per_grid_mask.csv` | `f8c0e0e45d455a74…` |
| `cross_sequence_productive_support.csv` | `a14d045f4234e046…` |
| `grammar_area_summary.csv` | `e37d80a6f1b0bec0…` |
| `unique_mask_summary.csv` | `32c8796d34f30de0…` |
| `summary.json` | `4d12f1f24448e42d…` |
| `g7_contract_gap.json` | `77f08344360e6921…` |

#### Committed Q4.5 evidence pack (subset)

`docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/` — subset only (manifest, summary, registry, region_stability, nested_loso clause summary, etc.). **Does not include full atlases or `per_sequence.csv`.** Pack alone is insufficient for atlas-derived R1 materialization.

### 1.2 Signal family / substrate / cohort

| Field | Value | Source |
|:--|:--|:--|
| Signal family (primary) | `score_m_bridge`, `abs_log_h`, `dist_h`, `abs_ratio_m1`, `resid_mean` | `threshold_registry.signals_primary` |
| Directions | `high_tail`, `low_tail` | registry |
| Combinators present | `AND`, `OR` only | registry; **no NOT / roles** |
| Substrate id | `stage1_baudit_d_online` | Q4.5 manifest |
| Candidate universe | `online_hook_eligible` | manifest |
| Sequence set | 7× MOT17-*-SDP | manifest |
| Cohort n (primary labels) | 87 = 23 neg + 64 GT protect | registry / summary |
| Selected unresolved | 21 | summary (blocks candidacy; not primary labels) |
| D_online total events | 244 | summary |

### 1.3 Label contract

From Q4.5 `cohort_definition` / Stage 2 final:

```text
negative_class:
  resolved AND baseline_selected AND pair_label == negative

positive_protection_class:
  resolved AND baseline_selected AND pair_label == gt_consistent

excluded_from_main:
  unresolved, ambiguous, non-selected, other labels

secondary_analysis:
  resolved non-selected negative / gt_consistent
```

Productive-safe (accepted operational definition, do not reopen):

```text
productive_safe_point ⇔
  resolved GT_hurt == 0
  AND n_neg_captured > 0
  AND no unresolved contamination that blocks candidate
```

### 1.4 Unresolved policy

```text
unresolved_contaminated_blocks_candidate: true
n_selected_unresolved: 21
claims: do not treat unresolved as FP; pessimistic hurt tracking present
```

### 1.5 Lattice / coordinate contracts

| Grammar | Lattice | Coordinate key | Registered size |
|:--|:--|:--|:--|
| G1 Singleton | `primary_unique_boundaries` | `(feature, direction, thr_index)` with thr_index ∈ [0,86] | 870 = 5×2×87 |
| G2 Pairwise AND | `primary_quantile_lattice_q05` | `(feature_a, direction_a, thr_index_a, feature_b, direction_b, thr_index_b)` thr ∈ [0,20] | 17640 = 40×21×21 |
| G3 Pairwise OR | same as G2 | same as G2 | 17640 |

Hard joins:

```text
S:: single-atom IDs  ⟂  P:: pairwise-atom IDs   (disjoint namespaces)
thr_index is the registered-grid coordinate; thr_value is recorded alias material
semantic_duplicate_is_per_grid_not_global: true
```

### 1.6 Dual-margin policy (accepted T0 derivation contract)

| Metric | Definition |
|:--|:--|
| `nearest_unsafe_distance` | same-grid graph distance to a registered **non-productive-safe** coordinate |
| `distance_to_lattice_edge` | min steps to registered lattice boundary |
| `full_neighborhood_safe_radius` | G1 bilateral; G2 Manhattan / 4-neighbor erosion; **missing off-lattice neighbor ⇒ radius 0** |
| Edge-censored distance | **not** region thickness |

### 1.7 Nested LOSO / transfer observation boundary

From `nested_loso_summary.json`:

```text
clause_identity: exact_absolute_threshold_float_round12
n_clauses_nested_loso_portable: 0
n_exact_absolute_clauses_nested_loso_portable: 0
definition: exact absolute clause repeatability — NOT quantile/rank region portability
```

This is a **narrow transfer observation**, not A2 region transfer evidence.

### 1.8 Live tree provenance drift (recorded, not repaired)

| Object | Study-recorded SHA256 | Current tree SHA256 | Match |
|:--|:--|:--|:--|
| Evaluator source | `551284f8…` | `8cefab0a…` | **no** |
| T0 analysis script | `173c920d…` (T0 manifest) | `e13bcc9f…` | **no** |

**Implication for R1:** pin **artifact hashes** and study-recorded evaluator/runner SHAs inside `asset_set` provenance. Do not re-hash current source and claim equivalence. Regeneration requires either sealed historical sources or an explicit re-execution contract (out of R0-A / R1).

### 1.9 Committed vs runtime requirements

| Input class | Committed in git? | Required for R1 conversion? |
|:--|:--|:--|
| Q4.5 pack subset | yes | identity / claim context |
| T0 interpretation pack | yes | geometry, capacity, sequence support, margins |
| Full atlases (parquet/csv) | **runtime only** | **yes** for coordinate/mask membership and non-PS lattice domain |
| `per_sequence.csv` | runtime only | optional if atlas embedded JSON used; PS coverage complete on both |
| Source event table | runtime | **not** required for A1 packaging of accepted atlas |
| Live evaluator source match | no (drift) | not required if artifacts sealed |

**If runtime full atlases are unavailable:** fields that need cell-level lattice membership become `BLOCKED_BY_ARTIFACT`. This preflight environment has them present and verified.

### 1.10 Preserved C0 numbers (no new research quantities)

Only schema/cardinality checks; accepted descriptive facts reused:

```text
PS coordinates: 154 = 1 G1 + 153 G2 + 0 G3
components (coordinate PS adjacency): 26
productive per-grid mask units: 34
multi-seq coordinates: 12 → 8 primary per-grid masks (4 global strings diagnostic)
full_neighborhood_safe_radius ≥ 1: 0/154
nearest_unsafe_distance: 1 for all 154
G3 observed_safe = 0, productive_safe = 0 over 17640 registered OR cells
terminal B retained
```

No new geometric research numbers are introduced beyond identity/cardinality bookkeeping needed for the contract.

---

## 2. Field derivability matrix

Legend:

| Status | Meaning |
|:--|:--|
| `DIRECT` | Present as-is on accepted artifact columns/JSON |
| `DETERMINISTICALLY_DERIVABLE` | Computable by declared joins/digests/filters without new labels or evaluator |
| `REQUIRES_CONTRACT_DECISION` | Multiple legal schemas; R0-B must choose |
| `BLOCKED_BY_ARTIFACT` | Needed machine input missing or incomplete under current pack contract |
| `BLOCKED_BY_PROVENANCE` | Cannot trust identity without sealed hash/source identity |
| `NOT_APPLICABLE` | Outside G1–G3 A1 scope or future maturity |

Each row: `field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary`

### 2.1 Asset-set identity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `asset_set_id` | asset set | REQUIRES_CONTRACT_DECISION | R0-B + Q4.5/T0 hashes | content-address digest of truth-bearing identity tuple (§4) | D1 pack grain; D2 ID namespace version | none if hashes pinned | identity only; not maturity |
| `producer_kind` | asset set | REQUIRES_CONTRACT_DECISION | program model | constant `grammar_atlas` for this pack | literal enum | — | observation producer only |
| `producer_contract_version` | asset set | REQUIRES_CONTRACT_DECISION | R0-B | schema version string e.g. `region_asset_v0` | version policy | — | contract version ≠ research acceptance |
| `grammar_or_condition_family` | asset set | DIRECT / REQUIRES_CONTRACT_DECISION | registry + R0-B | G1/G2/G3 vs combined multi-grammar set | D1 per-grammar vs multi | — | producer family only |
| `signal_family_version` | asset set | DIRECT | registry `signals_primary` + taxonomy | frozen 5-signal list + `stage2_q45_atlas_v4` | whether secondary features enter identity (default: no) | — | family identity |
| `substrate_id` | asset set | DIRECT | Q4.5 manifest | `stage1_baudit_d_online` | — | — | not online effect proof |
| `cohort_id` | asset set | DETERMINISTICALLY_DERIVABLE | cohort_definition + sequence_set + n_primary | digest of cohort contract fields | canonical serialization of cohort dict | — | label cohort only |
| `label_contract_id` | asset set | DETERMINISTICALLY_DERIVABLE | cohort_definition | digest of neg/pos/excluded definitions | naming of contract doc id | — | not safety proof |
| `unresolved_policy_id` | asset set | DETERMINISTICALLY_DERIVABLE | evaluator gates / summary | digest of unresolved block rules | — | — | firewall identity |
| `lattice_contract_id` | asset set | DIRECT | registry lattice kinds + sizes | pair of G1/G2 lattice descriptors | multi-lattice packaging under one set | — | registered domain only |
| `evaluator_version` | asset set | DIRECT | recorded evaluator SHA + taxonomy | use **recorded** SHA, not live tree | pin recorded vs recompute | live drift if recomputed | provenance only |
| `input_artifact_hashes` | asset set | DIRECT | Q4.5 + T0 manifests | copy sealed hash maps | which hash set is authoritative | missing atlases → incomplete | hash seal |
| `study_id` | asset set | DIRECT | manifests | `m_b1_5_stage2_q45_20260710` (+ optional T0 study id) | single vs dual study reference | — | provenance |
| `terminal_letter` / `stage2_q45_terminal` | asset set | DIRECT | summary | B / `isolated_safe_points_only` | whether stored on asset_set or claim contract | — | descriptive terminal; not A1 maturity |

### 2.2 Semantic definition

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `semantic_definition_id` | semantic | DETERMINISTICALLY_DERIVABLE | grammar + operands + combinator | digest of canonical operator tree (§4, §5) | D3 operand order symmetry | — | semantic identity ≠ mask identity |
| `operator_tree` | semantic | DIRECT (G1–G3) | atlas combinator / atom form | G1: unary thr predicate; G2: AND; G3: OR | serialization form (JSON tree vs string) | — | no NOT for G7 |
| `operand_identities` | semantic | DIRECT | `feature` / `atom_*_id` / `P::`/`S::` | from atlas/registry | S:: vs P:: never joined by id | — | lattice-bound |
| `operand_roles` | semantic | DIRECT for G1–G3 equality; NOT_APPLICABLE for G7 roles | G1–G3: symmetric AND/OR leaves | G1 none; G2/G3 unordered pair leaves | D4 AND operand order canonicalization | G7 roles blocked | do not invent N/P roles |
| `predicate_direction` | semantic | DIRECT | `direction` / `direction_a/b` | high_tail / low_tail | mapping to ≥/≤ must cite thr definition | — | direction ≠ logical NOT |
| `parameter_coordinate_system` | semantic | DIRECT | lattice_kind | G1 unique-boundary index; G2/G3 quantile index | transport claims separate (A2) | — | index system only |
| `canonicalization_rules` | semantic | REQUIRES_CONTRACT_DECISION | R0-B | e.g. lexicographic atom order for AND/OR | D4 | — | affects semantic_definition_id |
| `necessary_operand_role` | semantic | NOT_APPLICABLE (G1–G3) / BLOCKED (G7) | g7_contract_gap | leave unresolved | — | no role metadata | no G7 claim |
| `support_operand_role` | semantic | NOT_APPLICABLE / BLOCKED (G7) | g7_contract_gap | leave unresolved | — | same | no G7 claim |
| `gt_envelope_semantics` | semantic | BLOCKED_BY_ARTIFACT | future G7 | not in combinators | — | absent | unresolved |
| `logical_complement_identity` | semantic | BLOCKED_BY_ARTIFACT | future G7 | NOT not registered | — | combinators AND/OR only | unresolved |
| `envelope_relative_coordinate` | semantic | NOT_APPLICABLE for A1 G1–G3 | — | absolute thr_index only here | future transport | — | not A1 core |

### 2.3 Component / region identity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `region_asset_id` | region/component | REQUIRES_CONTRACT_DECISION | T0 components + semantic + asset_set | digest(asset_set_id, semantic_definition_id, grid_id, sorted coordinate membership) | D5 region grain = component | ordinal `::compN` forbidden | A1 identity only |
| `component_coordinate_digest` | region | DETERMINISTICALLY_DERIVABLE | `component_geometry.coords_json` + grid | sorted unique coords → digest | coordinate serialization form | — | geometry identity |
| `component_mask_digest` | region | DETERMINISTICALLY_DERIVABLE | masks of member coords within grid | multiset/set of per-grid mask units | set vs multiset; plateau handling | — | not global mask key |
| `human_alias` | region | REQUIRES_CONTRACT_DECISION | thr expressions / grid | non-canonical alias string | D6 alias policy | — | alias ≠ id |
| `maturity_level` | region | REQUIRES_CONTRACT_DECISION | program model | default proposed `A0` until chat accepts A1 packaging | maturity is research gate | — | **not auto A1** |
| `bounded_status` | region | DETERMINISTICALLY_DERIVABLE | PS emptiness / C0 | `HAS_REGION` vs `NULL_RESULT` | null at set vs region grain (D7) | — | descriptive |
| T0 `component_id` (`…::comp0`) | region | DIRECT but **non-stable** | T0 pack | enumeration within grid | must be replaced for A1 | ordinal order | diagnostic join key only |

### 2.4 Per-grid mask identity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `mask_unit_id` | per-grid mask | DETERMINISTICALLY_DERIVABLE | `(grid_id, mask_sha256)` | digest(asset_set_id, grid_id, mask_sha256) | D8 confirm primary unit | using global mask alone | primary mask unit |
| `mask_sha256` | mask string | DIRECT | atlases / T0 | event-mask digest string | scope declaration required | cross-grid collapse illegal as PK | diagnostic globally |
| `grid_id` | grid | DIRECT | T0 / atlas axes | G1 `S::feat::dir`; G2/G3 `P::…__…` | normalize separators | — | registered grid |
| `coordinate_membership_digest` | mask unit | DETERMINISTICALLY_DERIVABLE | atlas rows with same (grid, mask) | sorted thr coords → digest | include only PS vs all safe | — | plateau geometry |
| `semantic_duplicate_mask` | mask unit | DIRECT (G2/G3) | atlas column | pass-through | — | absent on G1 | per-grid duplicate flag |
| global mask quotient row | mask quotient | DIRECT | `region_stability` | mask-quotient grain | never primary asset key | grain mismatch vs components | topology diagnostic |

### 2.5 Coordinate identity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `coordinate_id` | coordinate | DETERMINISTICALLY_DERIVABLE | atlas cell keys | digest(asset_set_id, grid_id, thr indices) or reuse `atom_id`/`combo_id` within set | D9 native cell_id vs new digest | — | registered cell only |
| `thr_index` / `thr_index_a/b` | coordinate | DIRECT | atlases | integer lattice index | thr_value not part of id | — | index coordinate |
| `thr_value` / `thr_value_a/b` | coordinate | DIRECT | atlases | float thresholds | alias / payload only | float equality fragile | not primary id |
| `cell_id` (`S::…` / `AND::…` / `OR::…`) | coordinate | DIRECT | atlases / T0 | native evaluator cell id | may be human alias of coordinate_id | — | within grammar namespace |
| `observed_safe_point` | coordinate | DIRECT | atlases | boolean | — | — | observed safe ≠ policy |
| `productive_safe_point` | coordinate | DIRECT | atlases | boolean | — | — | productive observation |
| `empty_region` | coordinate | DIRECT (G2/G3) | atlases | boolean | — | — | empty support |

### 2.6 Null-asset representation

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| null asset record | null asset | REQUIRES_CONTRACT_DECISION | G3 OR atlas + area summary | first-class row with `region_asset_count=0` (§7) | D7 null grain | missing-files-as-null forbidden | NULL_RESULT is a result |
| `declared_search_domain` | null | DIRECT | lattice sizes | 17640 OR cells / 40 grids | — | — | registered domain |
| `observed_safe_count` | null | DIRECT | G3 atlas / area | 0 | — | — | observation |
| `productive_safe_count` | null | DIRECT | G3 | 0 | — | — | observation |
| `null_reason` | null | REQUIRES_CONTRACT_DECISION | interpretation | e.g. `no_productive_safe_on_registered_or_lattice` | taxonomy of reasons | — | not “OR grammar invalid” |
| `forbidden_promotions` | null | DETERMINISTICALLY_DERIVABLE | claim firewall | static list for A1 null | — | — | no production claim |

### 2.7 Geometry

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| safe/productive coordinates | coordinate / region | DIRECT + DERIVABLE | atlases + components | filter flags; component coords_json | which grain stores lists | atlases missing → blocked | descriptive geometry |
| per-grid unique masks | mask / grid | DIRECT | unique_mask_summary + capacity_by_mask | pass-through | — | — | per-grid scope |
| connected components | region | DIRECT (T0) | component_geometry | T0 adjacency contract | re-derive only if atlas present | ordinal ids | A0 geometry |
| active-axis count / spans | region | DIRECT | component_geometry | pass-through | — | — | shape only |
| plateau / duplicate structure | mask / region | DIRECT / DERIVABLE | region_stability + n_coords per mask | — | dual storage of plateau | quotient≠component | plateau ≠ thickness |
| `nearest_unsafe_distance` | coordinate | DIRECT (T0) | boundary_margin | pass-through | edge policy locked | — | dual metric 1 |
| `full_neighborhood_safe_radius` | coordinate | DIRECT (T0) | boundary_margin | pass-through | — | — | dual metric 2; all 0 |
| edge-censoring metadata | coordinate | DIRECT | boundary_margin | `edge_touches_lattice`, `nearest_unsafe_edge_censored` | — | — | not thickness |
| raw coordinate area ratios | grammar/grid | DIRECT | grammar_area_summary | pass-through | cross-lattice compare forbidden | — | descriptive ratios |
| unique-mask area ratios | grammar/grid | DIRECT | unique_mask_summary | per_registered_grid scope | — | global quotient illegal | descriptive |

### 2.8 Productive capacity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `n_neg_captured` | coordinate | DIRECT | atlas / productive_capacity | pass-through | — | — | mass observation |
| mask unit capacity | mask unit | DIRECT | productive_capacity_by_per_grid_mask | `mask_n_neg` (not × plateau width) | — | — | unique-mask capacity |
| component capacity | region | DETERMINISTICALLY_DERIVABLE | join coords→masks | aggregate unique mask capacities **or** sum coords with double-count policy | D10 capacity aggregation | double-count risk | must declare unit |
| capacity concentration | pack | DIRECT | T0 summary | top-1/3/5 shares | — | — | descriptive |
| `min_positive_sequence*` | coord/mask | DIRECT | capacity tables | named minimum among **positive** sequences | never rename to all-seven worst | — | not global worst-case |
| productive floor contract | region | REQUIRES_CONTRACT_DECISION | future A2 | not A1 required | A2 gate | — | intervention later |

### 2.9 Sequence applicability

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| productive sequences JSON | coord/mask | DIRECT | capacity / cross_sequence tables | pass-through | — | — | evidence slice |
| single vs multi-seq class | coord/mask | DIRECT | cross_sequence_productive_support | support_class | — | — | not policy feature |
| multi-seq intersection/union | region | DETERMINISTICALLY_DERIVABLE | set ops on sequence maps | optional | D11 region-level sequence set definition | — | observation |
| sequence dominance | coord/mask | DIRECT | max_neg_sequence_share | pass-through | — | — | descriptive |
| scene/condition support | — | NOT_APPLICABLE | no scene owner in G1–G3 | — | — | no scene ontology | do not invent |
| abstention/unknown surface | coordinate | PARTIAL DIRECT | n_unresolved_selected, safety_status | pass-through | full abstention model is A3 | — | firewall only |
| unresolved contamination | coordinate | DIRECT | atlas fields | pass-through | — | — | blocks candidacy |

### 2.10 Transfer qualification

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| exact-absolute nested LOSO portable flag | clause | DIRECT | nested_loso / region_stability | 0 portable clauses | — | — | **not** region transfer |
| LOO deletion consistency | coordinate | DIRECT | atlas loo_* columns | pass-through | do not promote to portability | — | deletion only |
| fixed absolute threshold transport | — | NOT_APPLICABLE (A1) | needs A2 study | — | — | no A2 evidence | production_forbidden |
| train quantile / rank-CDF transport | — | NOT_APPLICABLE (A1) | — | — | — | not studied as region transport | A2+ |
| component retention under LOO | — | BLOCKED_BY_ARTIFACT / NOT_APPLICABLE | no region-level LOO | — | — | not authorized | A2 |
| online substrate support as transfer | — | DIRECT context only | Stage1 null effect | `triggered=0` history | — | — | null intervention coverage |

### 2.11 Action contract

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `action_state` | region/set | REQUIRES_CONTRACT_DECISION | maturity model | default `observation_only` for A1 | D12 action enum | — | A1 non-actionable |
| `shadow_decision` eligibility | — | NOT_APPLICABLE until A2 | — | — | — | no transfer | forbidden at A1 |
| `condition_model_candidate` | — | NOT_APPLICABLE until A2 | — | — | — | — | forbidden at A1 |
| `offline_filter_candidate` | — | NOT_APPLICABLE until A2/A3 | — | — | — | — | forbidden at A1 |
| `default_off_intervention_candidate` | — | NOT_APPLICABLE until A3 | — | — | — | Stage1 null | forbidden |
| `production_forbidden` | set/region | DETERMINISTICALLY_DERIVABLE | always true for A1 pack | constant true | — | — | hard firewall |

### 2.12 Claim firewall

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `maximum_research_claim` | set | DETERMINISTICALLY_DERIVABLE | T0 accepted verdict text | descriptive atlas packaging only | wording freeze | — | not portable safe region |
| `forbidden_promotions` | set | DIRECT | T0 summary forbidden list + terminal B | copy list | — | — | explicit denylist |
| `not_a_safe_rule` | coordinate/mask | DIRECT | atlases / region_stability | pass-through | — | — | always 1 on PS set here |
| assignment-group key status | set | DIRECT | registry/summary | `invalid_frame_provenance` | out of R1 fix scope | ranking blocked | observation limitation |

### 2.13 Maturity status

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| current maturity | set | DIRECT (program) | assetization thread | **A0 descriptive atlas** | R1 packaging ≠ A1 acceptance | — | A1 requires chat acceptance |
| A1 readiness (engineering) | set | DETERMINISTICALLY_DERIVABLE | this preflight | feasible if D1–D12 closed + atlases sealed | — | ordinal ids / decisions | engineering ≠ research |
| A2–A4 fields | set | NOT_APPLICABLE | future stages | — | — | missing evidence | do not populate as true |

### 2.14 Generation provenance

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| generation script id | set | REQUIRES_CONTRACT_DECISION | future R1 tool | script path + SHA at generation | R1 implementation | live drift | reproducibility |
| deterministic rerun identity | set | DETERMINISTICALLY_DERIVABLE | input hashes + script + contract version | digest | — | script drift | rebuild seal |
| T0 derivation script SHA | set | BLOCKED_BY_PROVENANCE if live recompute | T0 manifest vs tree | pin recorded SHA | do not trust live tree alone | tree drifted | pin pack outputs |

### 2.15 Derivability summary counts

| Status | Approx. field coverage (A1-relevant) |
|:--|:--|
| DIRECT | majority of geometry, capacity, sequence, lattice, label, terminal fields |
| DETERMINISTICALLY_DERIVABLE | content digests, null counts, claim lists, component capacity with declared rule |
| REQUIRES_CONTRACT_DECISION | all stable IDs, pack grain, alias, maturity/action defaults, capacity aggregation |
| BLOCKED_BY_ARTIFACT | full non-PS lattice if atlases absent; G7 semantics; region-level LOO |
| BLOCKED_BY_PROVENANCE | live source re-hash identity without seal |
| NOT_APPLICABLE | A2–A4 transfer/action fields; scene ontology; G7 roles on G1–G3 |

---

## 3. Grain and relation analysis

### 3.1 Grains (must remain distinct)

```text
asset set
  └─ region/component  (connected PS set within one grid, or null record)
       ├─ per-grid mask unit  (mask_sha256 within one registered grid)
       │    └─ coordinate  (registered thr-index cell)
       └─ coordinate  (member cells; may span multiple mask units)
null asset  (first-class; not “missing files”)
```

Do **not** collapse into one flat row type.

### 3.2 Parent/child and cardinalities (G1–G3 accepted)

| Relation | Cardinality (this study) | Notes |
|:--|:--|:--|
| asset set → grammars | 1 → {G1,G2,G3} or 3 sets (D1) | contract |
| grammar G1 → region assets | 1 non-null component | 1 PS coordinate |
| grammar G2 → region assets | 25 components | from 153 PS coords |
| grammar G3 → region assets | **0** + **1 null asset** | first-class null |
| components total | **26** | sum coords = 154 |
| region → per-grid mask units | 1 → many | e.g. multi-mask 2d_region components |
| per-grid mask unit → coordinates | 1 → many | plateaus: 143/154 coords on multi-coord plateaus |
| productive per-grid mask units | **34** | primary capacity unit |
| global distinct mask strings among those 34 | **24** (10 strings multi-grid) | diagnostic only |
| coordinate → component | many → 1 within grid | adjacency partition |
| coordinate → mask unit | many → 1 within grid | by (grid, mask_sha256) |

### 3.3 Keys

| Grain | Proposed primary key | Foreign keys |
|:--|:--|:--|
| asset set | `asset_set_id` | — |
| semantic definition | `semantic_definition_id` | optional `asset_set_id` scope |
| region/component | `region_asset_id` | `asset_set_id`, `semantic_definition_id`, `grid_id` |
| per-grid mask unit | `mask_unit_id` | `asset_set_id`, optional `region_asset_id` (M:N via bridge) |
| coordinate | `coordinate_id` | `asset_set_id`, `grid_id`, optional `region_asset_id`, `mask_unit_id` |
| null asset | `asset_set_id` + grammar key (or dedicated `null_asset_id`) | `asset_set_id` |

**Region may contain multiple per-grid mask units:** **yes** (observed).  
**Same global mask string in different grids:** **multiple mask units** (observed: e.g. one mask on 9 grids).  
**Coordinates relate to masks and components independently:** coordinate ∈ exactly one component per grammar grid partition; coordinate ∈ exactly one per-grid mask unit.

### 3.4 Null assets vs empty files vs missing runs

| Representation | Legal? |
|:--|:--|
| First-class null record with domain counts and `null_reason` | **required** for G3 |
| Empty `region_components.csv` without null record | **illegal** as sole null encoding |
| Missing atlas / failed run | `BLOCKED_BY_ARTIFACT` / incomplete pack — **not** NULL_RESULT |
| Zero-byte mask file | **illegal** null encoding |

### 3.5 Cross-grain join warnings

```text
region_stability.region_id (mask::…)  ≠  component_id  ≠  combo_id/atom_id
per_sequence.region_id uses S::/AND::/OR:: namespaces (cell grain), not mask::
S:: atom ids must not join to P:: pairwise atom ids
```

---

## 4. Stable identity proposal

### 4.1 Digest policy

```text
algorithm: SHA-256
encoding: lowercase hex
version tag: id_scheme = "region_asset_id_v0"
namespace: "saccade:region_asset/v0"
canonical form: UTF-8 JSON with:
  - object keys sorted lexicographically
  - arrays in declared sort order
  - no insignificant whitespace (separators=(',', ':'))
  - floats only when explicitly allowed (default: thr_index ints only in IDs)
  - absent optional fields omitted (not null) unless schema requires null
```

IDs:

```text
<prefix> || sha256(canonical_json)[:32]   # 128-bit truncation proposal
# full 256-bit retained in provenance sidecar if collision ever observed
```

**Collision handling:** if full 256-bit collides across different payloads (astronomical), fail closed and extend to full digest; do not silently merge. Truncation is a display/storage choice; equality checks for critical merges should use full digest fields in R0-B if preferred.

**Regeneration:** same inputs + same `id_scheme` ⇒ same IDs. Row reorder, component enumeration order, and unrelated grammar addition must not change existing IDs (see §4.4).

**Human-readable aliases:** optional strings; **never** primary keys. Threshold value strings are aliases only.

### 4.2 Identity definitions

#### `asset_set_id`

Canonical inputs (truth-bearing):

```json
{
  "id_scheme": "region_asset_id_v0",
  "kind": "asset_set",
  "producer_kind": "grammar_atlas",
  "producer_contract_version": "region_asset_v0",
  "taxonomy_version": "stage2_q45_atlas_v4",
  "study_id": "m_b1_5_stage2_q45_20260710",
  "substrate_id": "stage1_baudit_d_online",
  "signal_family": ["abs_log_h","abs_ratio_m1","dist_h","resid_mean","score_m_bridge"],
  "sequence_set": ["MOT17-02-SDP", "...sorted..."],
  "label_contract": { "...canonical cohort_definition..." },
  "unresolved_policy": {
    "unresolved_contaminated_blocks_candidate": true
  },
  "lattice_contract": {
    "single_lattice_kind": "primary_unique_boundaries",
    "pairwise_lattice_kind": "primary_quantile_lattice_q05",
    "combinators": ["AND","OR"]
  },
  "evaluator_source_sha256": "551284f8…",
  "input_artifact_hashes": { "...sealed Q4.5 + T0 hashes..." },
  "grammar_scope": "G1_G2_G3"  
}
```

`grammar_scope` depends on D1 (combined vs per-grammar sets). Changing any truth-bearing field changes `asset_set_id`.

#### `semantic_definition_id`

```json
{
  "id_scheme": "region_asset_id_v0",
  "kind": "semantic_definition",
  "grammar": "G1_atom" | "G2_and" | "G3_or",
  "operator": "ATOM" | "AND" | "OR",
  "operands": [
    {"feature":"...","direction":"...","lattice_kind":"...","atom_namespace":"S|P"}
  ],
  "roles": null,
  "parameter_system": "unique_boundary_index" | "quantile_index_q05",
  "symmetry": "operands_sorted_lexicographic"
}
```

For G2/G3, operands are **sorted lexicographically by (feature, direction)** so `A AND B` ≡ `B AND A` under this scheme (D4 default). Operand **roles** remain null for G1–G3; G7 must not invent roles here.

**Does not include** threshold indices (those are coordinates, not grammar semantics).

#### `region_asset_id` (non-null)

```json
{
  "id_scheme": "region_asset_id_v0",
  "kind": "region_asset",
  "asset_set_id": "...",
  "semantic_definition_id": "...",
  "grid_id": "...",
  "adjacency": "G1_bilateral" | "G2_4neighbor",
  "membership": "productive_safe",
  "coordinate_digest": "sha256 of sorted coordinate keys"
}
```

Coordinate keys:

- G1: `["u86"]` or `[86]` — pick one in R0-B; recommend `thr_index` ints.
- G2: sorted `[[i,j], ...]` with fixed axis order matching grid_id atom order after D4 sort **or** fixed feature_a/feature_b as registered in grid_id (must be locked).

**Forbidden inputs:** component enumeration index, row number, write timestamp, thr_value floats.

#### `mask_unit_id`

```json
{
  "id_scheme": "region_asset_id_v0",
  "kind": "mask_unit",
  "asset_set_id": "...",
  "grid_id": "...",
  "mask_sha256": "..."
}
```

Primary. Global mask string alone is **not** sufficient.

#### `coordinate_id`

Preferred: reuse native cell identity scoped by asset set:

```json
{
  "id_scheme": "region_asset_id_v0",
  "kind": "coordinate",
  "asset_set_id": "...",
  "cell_id": "S::..." | "AND::..." | "OR::..."
}
```

Native `cell_id` already encodes grammar + axes + thr indices and is stable under row reorder.

### 4.3 Human-readable alias policy

Examples (aliases only):

```text
G1: resid_mean high_tail thr_index=86 (thr≈0.5647)
G2: abs_log_h↑ AND resid_mean↓ on P::abs_log_h::high_tail__resid_mean::low_tail
G3 null: Hard OR registered lattice — NULL_RESULT
```

Rules:

- aliases may include thr_value for humans;
- aliases may change without changing digests;
- aliases must not be used as join keys.

### 4.4 Explicit stability tests (semantic)

| Scenario | Expected ID behavior |
|:--|:--|
| Row reorder in atlas CSV | all digests **unchanged** |
| Component reorder / different enumeration (`comp0`↔`comp1`) | `region_asset_id` **unchanged** (content membership); T0 ordinal ids **must not** be used |
| Different component enumeration order across tools | same if membership+adjacency contract identical |
| Added unrelated grammar (e.g. future G4) under new asset_set | existing G1–G3 IDs **unchanged** if asset_set_id grammar_scope fixed; if combined set expands truth-bearing scope, asset_set_id **must** change |
| Regenerated equivalent artifact (same hashes) | IDs **unchanged** |
| Same global `mask_sha256` in different grids | **different** `mask_unit_id` |
| Same event mask with different operand roles | **different** `semantic_definition_id` / `region_asset_id` when roles exist; for G1–G3 roles null, AND vs OR already differ by operator |

Hard constraints satisfied by construction:

```text
component_03 / ::compN is not a stable identity
global mask_sha256 is not the primary mask-unit identity
human-readable threshold strings are aliases, not canonical IDs
```

---

## 5. Semantic preservation

### 5.1 What can be preserved for G1/G2 (DIRECT)

| Semantic element | G1 | G2 AND |
|:--|:--|:--|
| Grammar / operator tree | unary atom | binary AND |
| Operand identity | feature + direction + S:: lattice | two P:: atoms + features/directions |
| Operand role | n/a (single) | symmetric leaves (no N/P) |
| Predicate direction | high_tail / low_tail | per operand |
| Parameter coordinate system | unique boundary thr_index 0..86 | quantile thr_index 0..20 |
| Symmetry / canonicalization | n/a | D4 lex sort of operands |
| Mask equivalence | `mask_sha256` | `mask_sha256` + `semantic_duplicate_mask` |
| Semantic equivalence | operator+operands+lattice | same; **≠** mask equality |

### 5.2 G3

Operator OR with same operand identity machinery as G2; productive/observed safe empty ⇒ null asset, semantics still recorded.

### 5.3 Mask equivalence vs semantic equivalence

```text
mask-equal ⇏ semantically equal
  counterexamples requiring future role-aware grammars; also AND vs OR
  can share event masks across different grids/operands

semantically equal ⇏ same region_asset
  different grids / different coordinate memberships
```

Cross-grammar mask-string overlap (T0 `non_g7_mask_overlap.json`) is **diagnostic only**, not G7 equivalence.

### 5.4 Future G7 — leave unresolved

Do **not** infer from mask overlap:

```text
necessary operand role
support operand role
GT-envelope semantics
logical complement identity
envelope-relative coordinate
```

Status remains `not_derivable_from_current_artifact_contract` per accepted `g7_contract_gap.json`.

---

## 6. G1–G3 A0→A1 gap analysis

Preserved C0 conclusion:

> Within registered Q4.5 G1–G3 lattices, 154 PS coordinates are thin/edge-dominated; `full_neighborhood_safe_radius ≥ 1` is **0/154**; terminal **B** retained. **Not** portable safe region / production candidate / G7 equivalence.

### 6.1 G1 Singleton

| A1 dimension | Status |
|:--|:--|
| Already exists | 1 PS cell; component geometry; capacity; sequence support; dual margins; mask unit; atom semantics |
| Deterministically repackagable | yes, if atlases sealed + IDs accepted |
| Needs R0-B decision | region_asset_id scheme; asset_set packing; alias |
| Blocked | none for A1 core |
| Evaluator rerun required? | **no** |
| Non-null assets? | **yes** (1 isolated component) |
| Null asset? | no |

### 6.2 G2 Pairwise AND

| A1 dimension | Status |
|:--|:--|
| Already exists | 153 PS; 25 components; 33 productive per-grid masks (micro); thin/strip geometry; dual margins all radius 0 |
| Deterministically repackagable | yes |
| Needs R0-B decision | multi-mask component capacity aggregation (D10); semantic operand order (D4); region↔mask M:N bridge |
| Blocked | transfer/action fields (correctly out of A1) |
| Evaluator rerun required? | **no** |
| Non-null assets? | **yes** (thin/isolated; zero thick) |
| Null asset? | no |

### 6.3 G3 Hard OR

| A1 dimension | Status |
|:--|:--|
| Already exists | complete empty PS/OS on 17640 cells; grammar area zeros |
| Deterministically repackagable | as **first-class null asset** only |
| Needs R0-B decision | null record schema (D7); null_reason taxonomy |
| Blocked if null only by missing files | **policy-blocked** — must use §7 |
| Evaluator rerun required? | **no** |
| Non-null assets? | **no** |
| Null asset? | **yes** (required) |

### 6.4 R1 feasibility statement

```text
R1 can be scoped as deterministic A0→A1 conversion packaging:
  inputs = sealed Q4.5 full atlases + T0-B-R1 pack + accepted R0-B schema
  outputs = proposed §8 files
  no evaluator modification/rerun
  no new geometry research claims
  maturity field remains non-accepted until chat-side A1 acceptance
  action_state = observation_only; production_forbidden = true
```

Engineering packaging readiness ≠ research A1 acceptance ≠ terminal change.

---

## 7. G3 null-asset contract

Concrete case: **G3 Hard OR** on registered pairwise OR lattice.

### 7.1 Proposed first-class null representation

```text
asset_set_id:              <digest with grammar_scope including G3 or G3-only set>
producer_kind:             grammar_atlas
grammar_id:                G3_or
semantic_definition_id:    <OR operator tree; operands = pairwise atom family; roles null>
declared_search_domain:
  lattice_kind:            primary_quantile_lattice_q05
  combinator:              OR
  n_registered_coordinates: 17640
  n_registered_grids:      40
  thr_index_range:         [0,20] × [0,20]
registered_lattice_denominator: 17640
observed_safe_count:       0
productive_safe_count:     0
region_asset_count:        0
null_reason:               no_observed_or_productive_safe_on_registered_or_lattice
bounded_status:            NULL_RESULT
maturity_level:            A0 (until A1 packaging accepted chat-side)
action_state:              observation_only
production_forbidden:      true
provenance:
  study_id:                m_b1_5_stage2_q45_20260710
  pairwise_or_atlas_parquet_sha256: bc1c2938775caa1e8f262fb083131e5f678cf612950e2400516b6d7baeb93323
  t0_grammar_area_summary_sha256:   e37d80a6f1b0bec04a1b03ecfb6b6f53b0441c198a38a4db74a824792ab3908b
  terminal:                B isolated_safe_points_only
claim_boundary:
  maximum_claim:           registered OR lattice contains no productive-safe cell under accepted label/unresolved contracts
  forbidden_promotions:
    - treat_missing_files_as_null_result
    - claim_OR_grammar_universally_useless_off_lattice
    - production_gate
    - portable_safe_region
    - G7_equivalence
```

### 7.2 Forbidden encodings

```text
❌ omit G3 from pack
❌ empty components file only
❌ absent pairwise_or_atlas
❌ null_reason = "not computed"
```

### 7.3 What null does *not* mean

```text
≠ evaluator failure
≠ artifact missing
≠ “no OR semantics”
≠ authorization to skip domain registration
```

---

## 8. Proposed R1 machine schemas

Contract proposal only — **do not emit these files in R0-A**.

### 8.1 `region_asset_manifest.json`

| | |
|:--|:--|
| Grain | single pack / asset set |
| PK | `asset_set_id` |
| Required | schema_version, asset_set_id, producer_*, study_id, substrate_id, signal_family, sequence_set, label_contract, unresolved_policy, lattice_contract, evaluator_source_sha256, input_artifact_hashes, terminal_letter, stage2_q45_terminal, maturity_declared, action_state_default, production_forbidden, grammar_scope, counts (`n_region_assets`, `n_null_assets`, `n_mask_units`, `n_coordinates_ps`), generation block |
| Optional | human title, notes, t0_revision |
| Derivability | DIRECT/DERIVABLE from §1 |
| Invariants | production_forbidden==true for A1; maturity_declared≠A1 unless research acceptance recorded externally; hash map complete for sealed inputs |

### 8.2 `grammar_region_summary.csv`

| | |
|:--|:--|
| Grain | one row per grammar (G1/G2/G3) within asset_set |
| PK | `(asset_set_id, grammar)` |
| Required | n_registered_coords, n_observed_safe, n_productive_safe, n_components, n_null_assets, n_per_grid_mask_units_ps, coordinate_productive_area_ratio, unique_mask_productive_ratio_micro, max_full_neighborhood_safe_radius_over_ps, terminal_local |
| Optional | notes |
| Source | grammar_area_summary + component_geometry + unique_mask_summary + boundary_margin aggregates |
| Invariants | G3: n_productive_safe==0, n_components==0, n_null_assets==1; G1+G2+G3 PS sums to 154 for this study |

### 8.3 `region_assets.csv`

| | |
|:--|:--|
| Grain | region/component **or** null asset row |
| PK | `region_asset_id` (null uses dedicated id) |
| FK | asset_set_id, semantic_definition_id, grid_id nullable for grammar-level null |
| Required | grammar, bounded_status, maturity_level, action_state, production_forbidden, n_coords, n_mask_units, shape_class, is_genuine_2d_thick, max_full_neighborhood_safe_radius, human_alias |
| Optional | null_reason, declared_search_domain_json |
| Source | component_geometry + claim defaults; G3 null synthetic |
| Invariants | bounded_status=NULL_RESULT ⇒ n_coords==0 and null_reason set; HAS_REGION ⇒ n_coords≥1; no ::comp ordinal as PK |

### 8.4 `region_components.csv`

| | |
|:--|:--|
| Grain | same as non-null region (geometry detail) |
| PK | `region_asset_id` |
| Required | grid_id, adjacency_contract, axis_span_*, bounding_box_*, active_axis_count, coords_digest, coords_json, t0_component_id_alias |
| Optional | — |
| Source | component_geometry |
| Invariants | coords_json sorted; digest matches; t0 alias not used as PK |

### 8.5 `region_masks.csv`

| | |
|:--|:--|
| Grain | per-grid mask unit |
| PK | `mask_unit_id` |
| FK | asset_set_id, grid_id; bridge to regions via `region_mask_link` optional columns `region_asset_ids_json` or separate link table |
| Required | grammar, mask_sha256, n_coords, mask_n_neg, mask_n_gt, n_sequences_with_neg, productive_sequences_json, coordinate_membership_digest |
| Optional | semantic_duplicate_mask, min_positive_sequence* |
| Source | productive_capacity_by_per_grid_mask + atlas membership |
| Invariants | PK includes grid_id; global mask uniqueness not assumed |

### 8.6 `region_coordinates.csv`

| | |
|:--|:--|
| Grain | productive-safe coordinate (A1 minimum); optional extension to all registered cells out of scope |
| PK | `coordinate_id` |
| FK | asset_set_id, region_asset_id, mask_unit_id, grid_id |
| Required | grammar, cell_id, thr indices, observed_safe_point, productive_safe_point, n_neg_captured, gt_hurt, safety_status, mask_sha256, nearest_unsafe_distance, full_neighborhood_safe_radius, edge_touches_lattice |
| Optional | thr_value*, per_sequence_neg_json, loo_* flags, n_unresolved_selected |
| Source | atlases + boundary_margin + productive_capacity |
| Invariants | PS rows only unless schema version allows full lattice; thr_value not in coordinate_id |

### 8.7 `region_capacity.csv`

| | |
|:--|:--|
| Grain | declare `unit` column: `coordinate` \| `mask_unit` \| `region_asset` |
| PK | `(unit, unit_id)` |
| Required | n_neg_captured_or_mask_n_neg, n_sequences_with_neg, productive_sequences_json, min_positive_sequence, min_positive_sequence_n_neg, double_count_policy |
| Source | productive_capacity* ; region aggregate per D10 |
| Invariants | mask unit capacity not multiplied by plateau width; region aggregate policy explicit |

### 8.8 `region_sequence_support.csv`

| | |
|:--|:--|
| Grain | `unit` ∈ {coordinate, mask_unit, region_asset} × sequence support summary |
| PK | `(unit, unit_id)` for summary rows (as T0) or long format `(unit, unit_id, sequence)` |
| Required | support_class, n_sequences_with_productive_support, productive_sequences_json |
| Source | cross_sequence_productive_support (+ optional region rollup) |
| Invariants | sequence names are evidence slices, not policy features |

### 8.9 `region_margin.csv`

| | |
|:--|:--|
| Grain | coordinate (dual metrics) |
| PK | `coordinate_id` |
| Required | nearest_unsafe_distance, full_neighborhood_safe_radius, distance_to_lattice_edge, nearest_unsafe_edge_censored, edge_policy_id |
| Source | boundary_margin |
| Invariants | dual metrics both present; edge-censored ≠ thickness |

### 8.10 `region_claim_contract.json`

| | |
|:--|:--|
| Grain | asset set |
| PK | asset_set_id |
| Required | maximum_research_claim, forbidden_promotions[], action_states_allowed[], action_states_forbidden[], maturity_gate_definitions A0–A4 refs, terminal_b, not_a_safe_rule=true, g7_status, nested_loso_portability_note, production_preset=unchanged, evidence_ledger=not_promoted |
| Source | T0 verdict + Stage2 firewall + this program |
| Invariants | A1 does not include transfer or intervention permissions |

### 8.11 Validation invariants (pack-level)

```text
sum_PS(G1,G2,G3) == 154 for this sealed study
G3 null present
no region_asset_id depends on row/component ordinal
every mask_unit_id includes grid_id
every region with HAS_REGION has ≥1 coordinate FK
production_forbidden == true
action_state == observation_only for all A1 rows
evaluator not rerun; input hashes match seal
```

---

## 9. Maturity and claim firewall

### 9.1 Maturity levels — evidence and permitted actions

| Level | Required evidence | Permitted actions | Forbidden |
|:--|:--|:--|:--|
| **A0** descriptive atlas | point/region enumeration or null; accepted T0/C0 | describe; compare informally | consume as stable transferable asset |
| **A1** region asset | stable IDs+provenance; semantic def; grain relations; dual area/topology; capacity; sequence support; null form; claim firewall | compare, rank, diff, reproduce | transfer claims; intervention; production |
| **A2** validated applicability | A1 + per-seq contraction + explicit transport + region LOO/equiv + held-out harm + productive floor + unresolved firewall | condition modeling / shadow eval candidates | default intervention; production |
| **A3** intervention asset | A2 + action-time observables + frozen representative + default-off path + online support + control evidence + monitoring/rollback | application validation default-off | production default |
| **A4** production-approved | separate formal promotion + ops ownership | production use under governance | silent auto-promotion from R4 |

### 9.2 Independent gates (never collapse)

```text
artifact generated
engineering ready
asset maturity accepted          ← chat/research
research conclusion accepted     ← chat/research
intervention qualified
production approved
```

### 9.3 Action / claim states

| State | Min maturity | G1–G3 now |
|:--|:--|:--|
| `observation_only` | A0/A1 | **required default** |
| `shadow_decision` | A2 | forbidden |
| `condition_model_candidate` | A2 | forbidden |
| `offline_filter_candidate` | A2/A3 | forbidden |
| `default_off_intervention_candidate` | A3 | forbidden |
| `production_forbidden` | any | **true** |

### 9.4 A1 non-transferable / non-actionable

```text
A1 packaging may exist as engineering output after R0-B/R1 authorization
without implying:
  portable region
  nested-LOSO portability (0 exact absolute portable clauses)
  online reject policy
  production gate
  G7 necessity
```

---

## 10. Cross-family boundary

### 10.1 Core RegionAsset fields (shareable)

Suitable for grammar, occ-exit, association, relink producers:

```text
asset_set_id, producer_kind, producer_contract_version
substrate_id, cohort_id, label_contract_id, unresolved_policy_id
evaluator/provenance hashes
region_asset_id, semantic_definition_id (opaque tree ref)
mask_unit_id (if mask-like), coordinate_id (if lattice-like)
geometry dual margins (when defined), capacity, sequence support
maturity_level, action_state, production_forbidden
claim maximum + forbidden promotions
null asset record shape
```

### 10.2 Producer-specific opaque adapter payload

Stay out of A1 core columns; store in `adapter_payload_json` or side files:

```text
grammar: operator tree details, lattice_kind, thr registries, combinator catalogs
occ-exit: episode ids, gap paths, cover features, local enable predicates
association: cost terms, gate knobs, auction parameters
relink: bridge geometry, appearance veto thresholds, bank keys
```

No inheritance framework, runtime API, or shared code object is authorized here.

### 10.3 Intervention-time fields (not A1 core)

```text
action-time observable binding
control/treatment assignment
online trigger/decision_changed counters as effect proof
rollback/monitoring contracts
default-on production flags
transport retention metrics (quantile/rank/envelope LOO)
```

These belong to A2–A4 qualification packets, not the A1 region asset core.

---

## 11. Decisions required before R0-B

| decision_id | question | recommended_default | alternatives | evidence | falsifier | consequence_for_R1 | owner_required |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **D1** | One multi-grammar asset_set vs per-grammar sets? | **One** `grammar_scope=G1_G2_G3` pack with grammar column | 3 independent sets | single study/hashes | if grammars gain independent cohort/lattice seals | single manifest vs three | research owner |
| **D2** | ID scheme version pin? | `region_asset_id_v0` + SHA-256 truncated 32 hex | full 256-bit only | §4 | collision or scheme bug | all PK generation | research owner |
| **D3** | Is `semantic_definition_id` threshold-free? | **yes** (operator+operands+lattice only) | include thr families | G1/G2 share grammar across thr | if thr-specific semantics required | region ids depend on membership not thr family | research owner |
| **D4** | AND/OR operand order canonicalization? | **lexicographic (feature,direction)** | preserve evaluator atom_a/b order | symmetry of hard AND/OR | if asymmetric roles introduced | semantic_id stability | research owner |
| **D5** | Primary reusable region grain? | **connected PS component within grid** | mask unit as primary; dual primary | T0 components 26; program model | if components unstable under adjacency change | region_assets.csv grain | research owner |
| **D6** | Human alias format? | compact grid+shape+n_coords; thr_value optional | thr-value-first strings | alias≠id rule | alias used as join in tools | docs only | engineering+research |
| **D7** | Null grain packaging? | **grammar-level null row** in region_assets + summary counts | asset_set-level only | G3 empty | if partial-grid nulls needed later | G3 R1 rows | research owner |
| **D8** | Confirm primary mask unit? | **(grid_id, mask_sha256)** | global mask (reject) | 10 multi-grid masks | any tool joins on global mask only | region_masks PK | research owner |
| **D9** | coordinate_id native cell_id vs pure digest? | **asset_set-scoped native cell_id** | pure digest of indices | native ids stable & unique per grammar | namespace collision across studies without asset_set scope | coordinates PK | research owner |
| **D10** | Component capacity aggregation? | **sum unique per-grid mask capacities in component** (no plateau double count) | sum coordinate n_neg (double counts plateaus) | T0 mask capacity notes | if coordinate-additive mass intended | region_capacity region rows | research owner |
| **D11** | Region-level sequence support definition? | union of member coordinate productive sequences | intersection-only | multi-seq 12 coords / 8 masks | if intersection claimed without evidence | sequence_support region rows | research owner |
| **D12** | Default action_state at packaging? | `observation_only` + `production_forbidden=true` | none | A1 firewall | any auto shadow/intervention flag | claim_contract | research owner |

### 11.1 Decision classes

| Class | IDs |
|:--|:--|
| Acceptable from existing evidence | D3, D5 (supported), D8, D10 (supported by T0 notes), D12 |
| Naming/schema choices | D1, D2, D4, D6, D7, D9, D11 |
| Unresolved research semantics | G7 roles (not D-table; remain open) |
| Artifact/provenance blockers | runtime atlas presence for environments without `out/`; live evaluator/script drift if regeneration claimed |

### 11.2 Not decisions — fixed by accepted truth

```text
terminal B retained
per-grid mask primary unit
no global mask PK
no component ordinal identity
dual margin policy already fixed in T0
G7 roles not inventable
A1 non-transferable / non-actionable
no evaluator rerun for R1
```

---

## 12. R1 readiness recommendation (not authorization)

```text
IF R0-B accepts D1–D12 defaults (or explicit alternates)
AND full atlases remain hash-sealed
THEN R1 = deterministic conversion packaging A0→(candidate A1 files)
ELSE IF atlases missing THEN R1 BLOCKED_BY_ARTIFACT
ELSE IF identity decisions unresolved THEN R1 blocked on contract only

R1 authorization still requires separate chat-side gate after R0-B.
A1 research acceptance is a further gate after R1 engineering review.
```

---

## 13. Acceptance checks (R0-A self-audit)

```text
[x] every proposed field has a bounded derivability status
[x] asset/component/mask/coordinate/null grains remain distinct
[x] IDs do not depend on row or component ordinal
[x] per-grid mask unit remains primary
[x] semantic role identity is not collapsed into mask identity
[x] G3 has a complete first-class null-asset proposal
[x] A0–A4 maturity and action firewalls are explicit
[x] R1 scoped as deterministic conversion conditional on R0-B + sealed atlases
[x] no evaluator rerun or modification
[x] no R1 asset files generated
[x] no G4–G7 implementation
[x] terminal B unchanged
[x] production/presets unchanged
[x] evidence ledger unchanged
```

---

## 14. Explicit non-claims

```text
no evaluator rerun
no asset pack generated
no research verdict self-accepted
no R0-A acceptance declared
no R0-B / R1 authorized
no A0→A1 maturity promotion
no PR opened by this note alone
no evidence_ledger promotion
```

---

## 15. Next gate

```text
R0-A preflight complete
→ chat-side review of this note
→ only if accepted: authorize R0-B final RegionAsset contract
→ only after R0-B acceptance: authorize R1 deterministic packaging
```
