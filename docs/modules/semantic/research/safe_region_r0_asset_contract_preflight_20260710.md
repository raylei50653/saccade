# R0-A Region Asset Contract Preflight

<!-- doc-status: active -->
<!-- doc-promotion: navigation-support; not evidence_ledger; not research verdict; not A1 acceptance -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: this file for R0-A field/identity/schema derivability only -->

**Task:** R0-A Region Asset Contract Preflight · revision **R0-A-R1** (contract correction)  
**Branch:** `research/composition-grammar-coverage-program`  
**Program thread:** [safe_region_assetization_20260710.md](../../../research/threads/safe_region_assetization_20260710.md)  
**A0 baseline:** [composition_grammar_t0_region_interpretation_20260710.md](composition_grammar_t0_region_interpretation_20260710.md)  
**T0-A preflight:** [composition_grammar_t0_artifact_preflight_20260710.md](composition_grammar_t0_artifact_preflight_20260710.md)

> Answers **which RegionAsset fields/grains/identities are DIRECT, DERIVABLE, contract-dependent, or blocked** from accepted G1–G3 T0/C0 truth.  
> **Not** R0-B contract acceptance · **not** R1 asset generation · **not** evaluator rerun · **not** G4–G7 · **not** maturity promotion A0→A1.

```text
Research acceptance remains chat-side after review.
This note is an evidence packet, not a verdict.
R0-A-R1 corrects five blocking contract issues from chat-side review.
Prior R0-A packet: 762adf9a (provenance/grain/firewall portions retained).
```

---

## 0. Preflight headline

```text
R0-A-R1 CONTRACT CORRECTION COMPLETE (awaiting chat-side re-review):

  Provenance base: Q4.5 study m_b1_5_stage2_q45_20260710 + T0-B-R1 pack
  Runtime full atlases PRESENT and hash-match accepted manifest.
  Terminal B unchanged: isolated_safe_points_only.
  Accepted C0 geometry preserved (154 PS; radius≥1 = 0/154; G3 null).
  Maturity: A0 retained.

  Five review blockers addressed (CR1–CR5):
    CR1 identity layers split: truth_context ≠ pack ≠ content ≠ evidence_record
    CR2 component capacity = distribution over members; sum retracted;
        event-union capacity = BLOCKED_BY_ARTIFACT
    CR3 sequence union AND intersection + incidence; neither = A2 applicability
    CR4 G3 null = grammar/search-domain record; semantic_definition_id nullable
    CR5 region↔mask M:N authoritative via coordinate FKs (+ optional link table)

  R1 scope verdict (recommendation only; not authorized):
    DETERMINISTIC A0→A1 CONVERSION IS FEASIBLE
    conditional on R0-B accepting remaining naming decisions
    and sealed atlas availability.
    No evaluator rerun required for proposed A1 core fields.

  Hard constraints retained:
    - ::compN ordinals are not stable identity
    - global mask_sha256 is not primary mask-unit identity
    - G7 roles unresolved
    - transfer/intervention = A2+/A3
    - live evaluator/script tree drift pinned at provenance layer only
```

### R0-A-R1 change log (vs `762adf9a`)

| ID | Correction |
|:--|:--|
| CR1 | Decouple `truth_context_id` / `pack_id` / content IDs / optional `evidence_record_id` |
| CR2 | Retract additive component capacity; distribution semantics; event-union blocked |
| CR3 | Explicit sequence incidence, union, intersection, min/max support counts |
| CR4 | Grammar/search-domain null grain; nullable `semantic_definition_id`; count names |
| CR5 | Coordinate FKs authoritative for region↔mask; normalized link optional derived |

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
| PR / merge | #94 · `acd8e30e` (engineering merge ≠ research acceptance) |
| Reconciliation | **PASS** (154 = 1+153+0) |
| Dual-margin policy | declared in pack `summary.json` / note §1 |

**Committed T0 pack file hashes:** all 12 entries in `SHA256SUMS.json` re-verified **match**.

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

`docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/` — subset only. **Does not include full atlases or `per_sequence.csv`.** Pack alone is insufficient for atlas-derived R1 materialization.

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
| Selected unresolved | 21 | summary |
| D_online total events | 244 | summary |

### 1.3 Label contract

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
| G1 Singleton | `primary_unique_boundaries` | `(feature, direction, thr_index)` thr ∈ [0,86] | 870 = 5×2×87 |
| G2 Pairwise AND | `primary_quantile_lattice_q05` | 6-tuple thr ∈ [0,20] | 17640 = 40×21×21 |
| G3 Pairwise OR | same as G2 | same as G2 | 17640 |

```text
S:: single-atom IDs  ⟂  P:: pairwise-atom IDs
thr_index is the registered-grid coordinate; thr_value is alias material
semantic_duplicate_is_per_grid_not_global: true
```

### 1.6 Dual-margin policy (accepted T0 derivation contract)

| Metric | Definition |
|:--|:--|
| `nearest_unsafe_distance` | same-grid graph distance to a registered **non-productive-safe** coordinate |
| `distance_to_lattice_edge` | min steps to registered lattice boundary |
| `full_neighborhood_safe_radius` | G1 bilateral; G2 Manhattan / 4-neighbor erosion; off-lattice neighbor ⇒ radius 0 |
| Edge-censored distance | **not** region thickness |

### 1.7 Nested LOSO / transfer observation boundary

```text
clause_identity: exact_absolute_threshold_float_round12
n_clauses_nested_loso_portable: 0
definition: exact absolute clause repeatability — NOT quantile/rank region portability
```

Narrow transfer observation only — not A2 region transfer evidence.

### 1.8 Live tree provenance drift (recorded, not repaired)

| Object | Study-recorded SHA256 | Current tree SHA256 | Match |
|:--|:--|:--|:--|
| Evaluator source | `551284f8…` | `8cefab0a…` | **no** |
| T0 analysis script | `173c920d…` (T0 manifest) | `e13bcc9f…` | **no** |

**Implication:** pin **artifact hashes** and study-recorded evaluator/runner SHAs inside truth-context provenance. Do not re-hash current source and claim equivalence.

### 1.9 Committed vs runtime requirements

| Input class | Committed in git? | Required for R1 conversion? |
|:--|:--|:--|
| Q4.5 pack subset | yes | identity / claim context |
| T0 interpretation pack | yes | geometry, capacity, sequence, margins |
| Full atlases (parquet/csv) | **runtime only** | **yes** for coordinate/mask membership |
| `per_sequence.csv` | runtime only | optional if atlas embedded JSON used |
| Source event table | runtime | **not** required for A1 packaging of accepted atlas |
| Event membership bitsets | **absent** | required only for event-union capacity (§2.8, CR2) |
| Live evaluator source match | no (drift) | not required if artifacts sealed |

### 1.10 Preserved C0 numbers (no new research quantities)

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

Cardinality bookkeeping only (R0-A-R1 recheck for CR5):

```text
per-grid mask units spanning >1 component (this study): 0 observed
components spanning >1 mask unit: 4 observed
→ M:N still required in schema (do not assume mask→region many-to-one)
```

---

## 2. Field derivability matrix

Legend:

| Status | Meaning |
|:--|:--|
| `DIRECT` | Present as-is on accepted artifact columns/JSON |
| `DETERMINISTICALLY_DERIVABLE` | Computable by declared joins/digests/filters without new labels or evaluator |
| `REQUIRES_CONTRACT_DECISION` | Multiple legal schemas; R0-B must choose |
| `BLOCKED_BY_ARTIFACT` | Needed machine input missing under current pack contract |
| `BLOCKED_BY_PROVENANCE` | Cannot trust identity without sealed hash/source identity |
| `NOT_APPLICABLE` | Outside G1–G3 A1 scope or future maturity |

### 2.1 Identity layer fields (CR1)

Four **distinct** identity layers — never collapse:

```text
truth_context_id          # study truth / cohort / lattice / sealed input hashes
pack_id                   # materialization / schema / grammar_scope of a pack emission
region_asset_id           # stable content identity (local; not pack-scoped)
mask_unit_id / coordinate_id
evidence_record_id        # optional pack-specific row instance (if needed)
```

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `truth_context_id` | truth context | DETERMINISTICALLY_DERIVABLE | Q4.5+T0 seals | digest of truth-bearing study identity only (§4.2) | naming prefix | live re-hash | identity only |
| `pack_id` | pack/materialization | REQUIRES_CONTRACT_DECISION | R0-B schema | digest(truth_context_id, producer_contract_version, grammar_scope, schema_version, pack file list hashes) | D1 pack composition | — | pack emission only |
| `producer_kind` | pack | REQUIRES_CONTRACT_DECISION | program model | constant `grammar_atlas` | literal enum | — | observation producer |
| `producer_contract_version` | pack | REQUIRES_CONTRACT_DECISION | R0-B | e.g. `region_asset_v0` | version policy | — | changes `pack_id` **not** content IDs |
| `grammar_scope` | pack | REQUIRES_CONTRACT_DECISION | R0-B | e.g. `G1_G2_G3` | multi vs per-grammar packs | — | pack membership only |
| `signal_family_version` | truth context | DIRECT | registry + taxonomy | frozen 5-signal list + `stage2_q45_atlas_v4` | secondary features out | — | family identity |
| `substrate_id` | truth context | DIRECT | Q4.5 manifest | `stage1_baudit_d_online` | — | — | not online effect proof |
| `cohort_id` | truth context | DETERMINISTICALLY_DERIVABLE | cohort_definition + sequence_set | digest of cohort contract | serialization | — | label cohort |
| `label_contract_id` | truth context | DETERMINISTICALLY_DERIVABLE | cohort_definition | digest of neg/pos/excluded | — | — | not safety proof |
| `unresolved_policy_id` | truth context | DETERMINISTICALLY_DERIVABLE | evaluator gates | digest of unresolved block rules | — | — | firewall identity |
| `lattice_contract_id` | truth context | DIRECT | registry lattice kinds | G1+G2 lattice descriptors | — | — | registered domain |
| `evaluator_version` | truth context | DIRECT | recorded evaluator SHA + taxonomy | use **recorded** SHA | pin recorded | live drift if recomputed | provenance |
| `input_artifact_hashes` | truth context | DIRECT | Q4.5 + T0 manifests | sealed hash maps for **truth inputs only** | do not fold derived pack files into truth_context | missing atlases | hash seal |
| `study_id` | truth context | DIRECT | manifests | `m_b1_5_stage2_q45_20260710` | dual study refs optional | — | provenance |
| `terminal_letter` | truth context / claim | DIRECT | summary | B / `isolated_safe_points_only` | storage location | — | descriptive terminal |
| legacy `asset_set_id` (R0-A) | — | **SUPERSEDED** | — | split into truth_context_id + pack_id | do not use as parent of content IDs | conflates pack & truth | retired |

### 2.2 Semantic definition

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `semantic_definition_id` | semantic (concrete) | DETERMINISTICALLY_DERIVABLE | grammar + concrete operands + combinator | digest of canonical **concrete** operator tree (§4) | D4 operand order | — | ≠ mask identity |
| `semantic_family_id` / `search_domain_id` | semantic family / domain | DETERMINISTICALLY_DERIVABLE | grammar + lattice + combinator family | digest without concrete thr operands | naming | — | G3 null uses this |
| `operator_tree` | semantic | DIRECT (G1–G2 concrete) | atlas combinator / atom form | G1 unary; G2 AND pair; G3 family OR | serialization | G3 null has no single concrete tree | no NOT for G7 |
| `operand_identities` | semantic | DIRECT | feature / atom ids | from atlas/registry for concrete cells | S:: ⟂ P:: | — | lattice-bound |
| `operand_roles` | semantic | DIRECT for G1–G3 equality; NOT_APPLICABLE for G7 | G1–G3 symmetric leaves | roles null | D4 | G7 blocked | do not invent N/P |
| `predicate_direction` | semantic | DIRECT | direction fields | high_tail / low_tail | ≥/≤ mapping cites thr def | — | ≠ logical NOT |
| `parameter_coordinate_system` | semantic | DIRECT | lattice_kind | unique-boundary vs quantile index | transport = A2 | — | index system only |
| `canonicalization_rules` | semantic | REQUIRES_CONTRACT_DECISION | R0-B | lex operand order for AND/OR | D4 | — | affects semantic_id |
| G7 role / envelope / complement fields | semantic | BLOCKED_BY_ARTIFACT / NOT_APPLICABLE | g7_contract_gap | leave unresolved | — | no role metadata | no G7 claim |

### 2.3 Component / region identity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `region_asset_id` | region/component | DETERMINISTICALLY_DERIVABLE | content + truth_context | digest(**truth_context_id**, semantic_definition_id, grid_id, adjacency, sorted coord membership) — **not pack_id** | D5 grain; D2 id scheme | ordinal `::compN` | A1 content identity |
| `component_coordinate_digest` | region | DETERMINISTICALLY_DERIVABLE | coords_json | sorted unique coords → digest | serialization form | — | geometry identity |
| `component_mask_digest` | region | DETERMINISTICALLY_DERIVABLE | member mask units | sorted set of mask_unit_ids | set policy | — | not global mask key |
| `human_alias` | region | REQUIRES_CONTRACT_DECISION | thr expressions | non-canonical alias | D6 | — | alias ≠ id |
| `maturity_level` | region | REQUIRES_CONTRACT_DECISION | program | default `A0` until chat accepts A1 packaging | research gate | — | **not auto A1** |
| `bounded_status` | region | DETERMINISTICALLY_DERIVABLE | PS emptiness | `HAS_REGION` vs null records use separate null grain | D7 | — | descriptive |
| T0 `component_id` (`…::comp0`) | region | DIRECT but **non-stable** | T0 pack | enumeration within grid | replace for A1 | ordinal order | diagnostic alias only |
| `evidence_record_id` | pack row | DETERMINISTICALLY_DERIVABLE | pack_id + region_asset_id | optional instance id for a pack emission | whether required | — | pack-local only |

### 2.4 Per-grid mask identity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `mask_unit_id` | per-grid mask | DETERMINISTICALLY_DERIVABLE | (truth_context, grid_id, mask_sha256) | digest(**truth_context_id**, grid_id, mask_sha256) — **not pack_id** | D8 | global mask alone | primary mask unit |
| `mask_sha256` | mask string | DIRECT | atlases / T0 | event-mask digest string | scope declaration | cross-grid collapse illegal as PK | diagnostic globally |
| `grid_id` | grid | DIRECT | T0 / atlas axes | G1 `S::…`; G2/G3 `P::…__…` | separators | — | registered grid |
| `coordinate_membership_digest` | mask unit | DETERMINISTICALLY_DERIVABLE | atlas rows same (grid, mask) | sorted thr coords | PS-only vs all safe | — | plateau geometry |
| `semantic_duplicate_mask` | mask unit | DIRECT (G2/G3) | atlas | pass-through | — | absent on G1 | per-grid flag |

### 2.5 Coordinate identity

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `coordinate_id` | coordinate | DETERMINISTICALLY_DERIVABLE | native cell_id + truth_context | digest(**truth_context_id**, cell_id) or equivalent — **not pack_id** | D9 native vs pure digest | — | registered cell |
| `thr_index` / `thr_index_a/b` | coordinate | DIRECT | atlases | integer lattice index | thr_value not in id | — | index coordinate |
| `thr_value*` | coordinate | DIRECT | atlases | float | alias/payload only | float equality fragile | not primary id |
| `cell_id` | coordinate | DIRECT | atlases / T0 | native evaluator cell id | may be alias of coordinate_id | — | grammar namespace |
| `observed_safe_point` / `productive_safe_point` | coordinate | DIRECT | atlases | boolean | — | — | observation |
| `region_asset_id` FK | coordinate | DETERMINISTICALLY_DERIVABLE | component membership | join via adjacency partition | — | — | **authoritative** region link (CR5) |
| `mask_unit_id` FK | coordinate | DETERMINISTICALLY_DERIVABLE | (grid, mask_sha256) | — | — | — | **authoritative** mask link (CR5) |

### 2.6 Null-asset representation (CR4)

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| null record | null (grammar/search-domain) | REQUIRES_CONTRACT_DECISION | G3 OR atlas + area | first-class row; **not** a concrete region_asset | D7 structure | missing-files-as-null forbidden | NULL_RESULT is a result |
| `null_record_id` | null | DETERMINISTICALLY_DERIVABLE | truth_context + search_domain | digest(truth_context_id, search_domain_id, null_reason_class) | — | — | ≠ region_asset_id |
| `search_domain_id` / `semantic_family_id` | null | DETERMINISTICALLY_DERIVABLE | G3 OR lattice family | combinator=OR + pairwise quantile lattice + signal family | — | — | domain identity |
| `semantic_definition_id` | null | **nullable / NOT_APPLICABLE** for grammar-level null | — | **null** — no single concrete operand tree | never fake “pairwise atom family” as one tree | — | CR4 |
| `declared_search_domain` | null | DIRECT | lattice sizes | 17640 OR cells / 40 grids | — | — | registered domain |
| `observed_safe_count` / `productive_safe_count` | null | DIRECT | G3 atlas / area | 0 / 0 | — | — | observation |
| `n_non_null_region_assets` | grammar summary | DIRECT | component count | **0** for G3 | — | — | not contradict null rows |
| `n_null_records` | grammar summary | DETERMINISTICALLY_DERIVABLE | null rows | **1** grammar-level (default) | per-grid nulls optional | — | separate from region count |
| `null_reason` | null | REQUIRES_CONTRACT_DECISION | interpretation | e.g. `no_productive_safe_on_registered_or_lattice` | taxonomy | — | not “OR invalid” |
| optional per-grid null summary | grid | DETERMINISTICALLY_DERIVABLE | per-grid area zeros | optional child rows | whether emit | — | still not concrete semantic_id |

### 2.7 Geometry

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| safe/productive coordinates | coordinate / region | DIRECT + DERIVABLE | atlases + components | filter flags; coords_json | — | atlases missing | descriptive |
| per-grid unique masks | mask / grid | DIRECT | unique_mask_summary | pass-through | — | — | per-grid scope |
| connected components | region | DIRECT (T0) | component_geometry | T0 adjacency | re-derive needs atlas | ordinal ids | A0 geometry |
| active-axis / spans / shape | region | DIRECT | component_geometry | pass-through | — | — | shape only |
| dual margins | coordinate | DIRECT (T0) | boundary_margin | pass-through | edge policy locked | — | dual metrics |
| area ratios | grammar/grid | DIRECT | grammar_area / unique_mask | pass-through | cross-lattice compare forbidden | — | descriptive |

### 2.8 Productive capacity (CR2)

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `n_neg_captured` | coordinate | DIRECT | atlas / productive_capacity | pass-through | — | — | mass observation |
| mask unit capacity `mask_n_neg` | mask unit | DIRECT | productive_capacity_by_per_grid_mask | **not** × plateau width | — | — | unique-mask capacity |
| **component capacity sum of mask_n_neg** | region | **INVALID / retracted** | — | **must not be used** | D10 retracted | double-count risk | **not a legal metric** |
| component capacity **distribution** | region | DETERMINISTICALLY_DERIVABLE | member coord and/or mask capacities | min, max, median, quantiles, robust floor over member values | D10b which member grain | — | descriptive alternatives |
| component event-union capacity | region | **BLOCKED_BY_ARTIFACT** | would need event membership bitsets | cannot invert `mask_sha256` to event sets | sealed reconstruction contract absent | no event ids on atlas rows | do not emit fake union |
| capacity concentration (pack) | pack | DIRECT | T0 summary top-k | per-grid mask unit shares | — | — | descriptive |
| `min_positive_sequence*` | coord/mask | DIRECT | capacity tables | among **positive** sequences only | never “all-seven worst” | — | not global worst-case |
| productive floor contract | region | REQUIRES_CONTRACT_DECISION | future A2 | not A1 required | A2 gate | — | intervention later |

### 2.9 Sequence applicability geometry (CR3)

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| coordinate/mask × sequence incidence | incidence | DIRECT / DERIVABLE | productive_sequences_json / per_sequence | long-format preferred for A1 | D11 format | sparse per_sequence for non-PS | **evidence slice only** |
| component sequence **union** | region | DETERMINISTICALLY_DERIVABLE | union of member productive sequence sets | explicit name `sequence_support_union` | — | — | “some parameter worked somewhere” |
| component sequence **intersection** | region | DETERMINISTICALLY_DERIVABLE | intersection of member sets | `sequence_support_intersection` | — | empty intersection legal | “all members support” |
| min/max n_sequences across members | region | DETERMINISTICALLY_DERIVABLE | member support counts | min/max | — | — | descriptive |
| sequence dominance / islands | coord/mask | DIRECT | max_neg_sequence_share, support_class | pass-through | — | — | descriptive |
| scene/condition support | — | NOT_APPLICABLE | no scene owner | — | — | — | do not invent |
| **A2 applicability claim from union** | region | **NOT_APPLICABLE / forbidden at A1** | — | requires representative/transport contract | — | — | CR3 |
| unresolved contamination | coordinate | DIRECT | atlas fields | pass-through | — | — | blocks candidacy |

### 2.10 Transfer qualification

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| exact-absolute nested LOSO portable | clause | DIRECT | nested_loso | 0 portable | — | — | **not** region transfer |
| LOO deletion consistency | coordinate | DIRECT | atlas loo_* | pass-through | not portability | — | deletion only |
| absolute / quantile / rank / envelope transport | — | NOT_APPLICABLE (A1) | needs A2 | — | — | not studied | production_forbidden |
| component retention under LOO | — | NOT_APPLICABLE / BLOCKED | no region LOO | — | — | not authorized | A2 |

### 2.11 Action contract

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `action_state` | region/pack | REQUIRES_CONTRACT_DECISION | maturity model | default `observation_only` | D12 | — | A1 non-actionable |
| shadow / condition / offline / default-off states | — | NOT_APPLICABLE until A2/A3 | — | — | — | — | forbidden at A1 |
| `production_forbidden` | pack/region | DETERMINISTICALLY_DERIVABLE | always true for A1 | constant true | — | — | hard firewall |

### 2.12 Claim firewall

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `maximum_research_claim` | pack | DETERMINISTICALLY_DERIVABLE | T0 verdict | descriptive packaging only | wording freeze | — | not portable safe region |
| `forbidden_promotions` | pack | DIRECT | T0 + terminal B | copy list | — | — | denylist |
| `not_a_safe_rule` | coordinate/mask | DIRECT | atlases | pass-through | — | — | always 1 here |
| assignment-group key status | truth | DIRECT | registry/summary | `invalid_frame_provenance` | out of R1 fix | ranking blocked | observation limitation |

### 2.13 Maturity status

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| current maturity | program | DIRECT | thread | **A0** | R1 packaging ≠ A1 acceptance | — | chat gate |
| A1 engineering readiness | pack | DETERMINISTICALLY_DERIVABLE | this preflight | feasible if remaining decisions closed + atlases sealed | — | CR1–CR5 were blockers; now corrected | ≠ research |
| A2–A4 fields | — | NOT_APPLICABLE | future | — | — | missing evidence | do not populate as true |

### 2.14 Generation provenance

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| generation script id | pack | REQUIRES_CONTRACT_DECISION | future R1 tool | path + SHA at generation | R1 implementation | live drift | reproducibility |
| deterministic rerun identity | pack | DETERMINISTICALLY_DERIVABLE | truth seals + pack schema + script | digest → pack_id inputs | — | script drift | rebuild seal |
| T0 derivation script SHA | truth/pack | BLOCKED_BY_PROVENANCE if live recompute | T0 manifest | pin recorded | — | tree drifted | pin pack outputs |

### 2.15 Derivability summary

| Status | A1-relevant coverage |
|:--|:--|
| DIRECT | geometry, coord/mask capacity, sequence incidence inputs, lattices, labels, terminal |
| DETERMINISTICALLY_DERIVABLE | content digests, capacity distributions, sequence union/intersection, claim lists |
| REQUIRES_CONTRACT_DECISION | pack composition, alias, null taxonomy, action default, some serializations |
| BLOCKED_BY_ARTIFACT | event-union capacity; full lattice if atlases absent; G7; region LOO |
| BLOCKED_BY_PROVENANCE | live source re-identity without seal |
| NOT_APPLICABLE | A2–A4 transfer/action; scene ontology; G7 roles; additive component capacity (invalid) |

---

## 3. Grain and relation analysis

### 3.1 Grains (must remain distinct)

```text
truth context
pack / materialization
semantic definition (concrete operator tree)     # nullable on grammar-level null
semantic family / search domain                  # G3 null owner
region/component  (connected PS set within one grid)
per-grid mask unit
coordinate
null record  (grammar/search-domain; first-class)
evidence record  (optional pack-local instance)
```

Do **not** collapse into one flat row type.  
Do **not** parent content IDs under `pack_id`.

### 3.2 Parent/child and cardinalities (G1–G3 accepted)

| Relation | Cardinality (this study) | Notes |
|:--|:--|:--|
| truth context → packs | 1 → many | schema/scope variants |
| pack → grammars | 1 → {G1,G2,G3} (D1) | pack membership only |
| grammar G1 → non-null regions | 1 | 1 PS coordinate |
| grammar G2 → non-null regions | 25 | from 153 PS coords |
| grammar G3 → non-null regions | **0** | |
| grammar G3 → null records | **1** (grammar-level default) | CR4 |
| components total | **26** | sum coords = 154 |
| region → mask units | 1 → many | 4 multi-mask components observed |
| mask unit → regions | many → many **in schema** | 0 multi-region masks observed **here**; still do not assume 1:1 |
| mask unit → coordinates | 1 → many | plateaus |
| productive per-grid mask units | **34** | primary capacity unit |
| coordinate → region | many → 1 within grid | adjacency partition |
| coordinate → mask unit | many → 1 within grid | (grid, mask_sha256) |

### 3.3 Keys (corrected)

| Grain | Primary key | Foreign keys / notes |
|:--|:--|:--|
| truth context | `truth_context_id` | — |
| pack | `pack_id` | `truth_context_id` |
| semantic definition | `semantic_definition_id` | concrete trees only |
| search domain / family | `search_domain_id` | used by G3 null |
| region/component | `region_asset_id` | `truth_context_id`, `semantic_definition_id`, `grid_id` — **not pack_id** |
| per-grid mask unit | `mask_unit_id` | `truth_context_id`, `grid_id` — **not pack_id** |
| coordinate | `coordinate_id` | `truth_context_id`, `grid_id`, **`region_asset_id`**, **`mask_unit_id`** |
| null record | `null_record_id` | `truth_context_id`, `search_domain_id`; `semantic_definition_id` **NULL** |
| evidence record (opt.) | `evidence_record_id` | `pack_id` + content id |

### 3.4 Authoritative region↔mask relation (CR5)

```text
AUTHORITATIVE derivation source:
  region_coordinates.csv
    coordinate_id → region_asset_id
    coordinate_id → mask_unit_id

OPTIONAL derived table (not a second truth):
  region_mask_link.csv
    PK (region_asset_id, mask_unit_id)
    derived by DISTINCT projection from coordinates
    may store n_coords in link for convenience

FORBIDDEN as authoritative truth:
  region_masks.region_asset_ids_json
  any embedded JSON list of partner IDs
```

Invariant: tools may materialize `region_mask_link` for convenience, but if it disagrees with coordinate FKs, **coordinates win**.

### 3.5 Null assets vs empty files vs missing runs

| Representation | Legal? |
|:--|:--|
| First-class null record + domain counts + `null_reason` | **required** for G3 |
| `n_non_null_region_assets=0` **and** `n_null_records=1` | **required** disambiguation |
| Empty components file without null record | **illegal** |
| Missing atlas / failed run | `BLOCKED_BY_ARTIFACT` — **not** NULL_RESULT |
| Zero-byte mask file | **illegal** null encoding |

### 3.6 Cross-grain join warnings

```text
region_stability.region_id (mask::…)  ≠  component_id  ≠  combo_id/atom_id
per_sequence.region_id uses S::/AND::/OR:: namespaces (cell grain), not mask::
S:: atom ids must not join to P:: pairwise atom ids
pack_id must not be required to resolve region_asset_id
```

---

## 4. Stable identity proposal (CR1)

### 4.1 Digest policy

```text
algorithm: SHA-256
encoding: lowercase hex
id_scheme: "region_asset_id_v1"    # bumped from v0 due to identity-layer split
namespace: "saccade:region_asset/v1"
canonical form: UTF-8 JSON
  - object keys sorted lexicographically
  - arrays in declared sort order
  - separators=(',', ':')
  - thr_index ints only in content IDs (no thr_value floats)
  - absent optional fields omitted unless schema requires null
```

```text
display_id = <prefix> || sha256(canonical_json)[:32]
full_digest retained in provenance for collision fail-closed
```

**Human-readable aliases:** optional; never PKs; thr_value strings are aliases only.

### 4.2 Layer definitions

#### `truth_context_id` (truth/evidence context)

Truth-bearing only — **excludes** pack schema, grammar_scope of emission, derived pack file hashes, producer_contract_version of materialization:

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "truth_context",
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
  "truth_input_artifact_hashes": {
    "atom_atlas_parquet": "281cb22b…",
    "pairwise_and_atlas_parquet": "ae52bd0c…",
    "pairwise_or_atlas_parquet": "bc1c2938…",
    "threshold_registry": "d3e3197f…",
    "summary": "a88d9dcf…",
    "t0_component_geometry": "ad2dd33b…",
    "t0_boundary_margin": "424fe063…",
    "t0_summary": "4d12f1f2…"
  }
}
```

Changes when study truth changes. Does **not** change when an unrelated grammar is added to a **different** pack, or when materialization schema version changes.

#### `pack_id` (asset-pack / materialization identity)

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "pack",
  "truth_context_id": "...",
  "producer_kind": "grammar_atlas",
  "producer_contract_version": "region_asset_v0",
  "schema_version": "region_asset_tables_v0",
  "grammar_scope": "G1_G2_G3",
  "generation_script_sha256": null
}
```

Changes when packaging choices change. **Must not** be an input to `region_asset_id` / `mask_unit_id` / `coordinate_id`.

#### `semantic_definition_id` (concrete only)

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "semantic_definition",
  "grammar": "G1_atom" | "G2_and",
  "operator": "ATOM" | "AND",
  "operands": [
    {"feature":"...","direction":"...","lattice_kind":"...","atom_namespace":"S|P"}
  ],
  "roles": null,
  "parameter_system": "unique_boundary_index" | "quantile_index_q05",
  "symmetry": "operands_sorted_lexicographic"
}
```

For G2, operands sorted lexicographically by `(feature, direction)` (D4).  
**Does not include** threshold indices.  
**Not used** as sole identity for G3 grammar-level null (CR4).

#### `search_domain_id` (semantic family / declared search domain)

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "search_domain",
  "grammar": "G3_or",
  "combinator": "OR",
  "lattice_kind": "primary_quantile_lattice_q05",
  "signal_family": ["abs_log_h","abs_ratio_m1","dist_h","resid_mean","score_m_bridge"],
  "n_registered_grids": 40,
  "n_registered_coordinates": 17640
}
```

#### `region_asset_id` (stable region **content** identity)

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "region_asset",
  "truth_context_id": "...",
  "semantic_definition_id": "...",
  "grid_id": "...",
  "adjacency": "G1_bilateral" | "G2_4neighbor",
  "membership": "productive_safe",
  "coordinate_digest": "sha256 of sorted thr-index keys"
}
```

**Forbidden inputs:** `pack_id`, grammar_scope of pack, producer_contract_version, component ordinal, thr_value floats, unrelated grammar membership.

#### `mask_unit_id`

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "mask_unit",
  "truth_context_id": "...",
  "grid_id": "...",
  "mask_sha256": "..."
}
```

#### `coordinate_id`

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "coordinate",
  "truth_context_id": "...",
  "cell_id": "S::..." | "AND::..." | "OR::..."
}
```

#### `null_record_id`

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "null_record",
  "truth_context_id": "...",
  "search_domain_id": "...",
  "null_reason_class": "no_productive_safe_on_registered_domain"
}
```

#### `evidence_record_id` (optional pack-specific)

```json
{
  "id_scheme": "region_asset_id_v1",
  "kind": "evidence_record",
  "pack_id": "...",
  "content_id": "<region_asset_id or null_record_id or mask_unit_id>",
  "content_kind": "region_asset" | "null_record" | "mask_unit" | "coordinate"
}
```

Use only when a pack needs a distinct row instance identity. Never substitute for content IDs.

### 4.3 Human-readable alias policy

```text
G1: resid_mean high_tail thr_index=86 (thr≈0.5647)
G2: abs_log_h↑ AND resid_mean↓ @ P::abs_log_h::high_tail__resid_mean::low_tail
G3 null: Hard OR registered domain — NULL_RESULT (search_domain)
```

Aliases may change without changing digests; must not be join keys.

### 4.4 Explicit stability tests (CR1)

| Scenario | `truth_context_id` | `pack_id` | `region_asset_id` / `mask_unit_id` / `coordinate_id` |
|:--|:--|:--|:--|
| Row reorder in atlas | unchanged | unchanged | **unchanged** |
| Component ordinal reorder (`comp0`↔`comp1`) | unchanged | unchanged | **unchanged** (content membership) |
| Added unrelated grammar G4 in a **new** pack | unchanged | **new pack** | **unchanged** |
| Same truth, grammar_scope expanded in a new combined pack | unchanged | **new pack** | **unchanged** |
| Schema / producer_contract_version migration only | unchanged | **changes** | **unchanged** |
| New derived file added to pack emission | unchanged | may change if pack file seal in pack_id | **unchanged** |
| Regenerated equivalent sealed artifacts | unchanged | unchanged | **unchanged** |
| Truth input hash changes (new atlas) | **changes** | changes | **changes** (correct) |
| Same global `mask_sha256` in different grids | — | — | **different** `mask_unit_id` |
| Same mask, different operand roles (future) | — | — | different `semantic_definition_id` / region |

Hard constraints:

```text
component_03 / ::compN is not a stable identity
global mask_sha256 is not the primary mask-unit identity
human-readable threshold strings are aliases, not canonical IDs
pack_id is not a parent key of content identity
```

### 4.5 Synthetic identity examples

**Example A — unrelated grammar must not move G1 region id**

```text
truth_context_id = TC0  (Q4.5+T0 seals)
region_asset_id(G1 resid_mean high_tail {86}) = R_G1

pack_A: grammar_scope=G1_G2_G3, schema=v0  → pack_id=P_A
pack_B: grammar_scope=G1_G2_G3_G4, schema=v0 → pack_id=P_B ≠ P_A

assert region_asset_id in pack_A == region_asset_id in pack_B == R_G1
assert only pack_id / evidence_record_id differ
```

**Example B — schema migration**

```text
producer_contract_version v0 → v1
pack_id changes
R_G1, mask_unit_id, coordinate_id unchanged
```

**Example C — same mask string, two grids**

```text
mask_sha256 = M
grid_id = Gx, Gy
mask_unit_id(TC0, Gx, M) ≠ mask_unit_id(TC0, Gy, M)
```

---

## 5. Semantic preservation

### 5.1 G1/G2 DIRECT

| Semantic element | G1 | G2 AND |
|:--|:--|:--|
| Grammar / operator tree | unary atom | binary AND |
| Operand identity | feature + direction + S:: | two P:: atoms |
| Operand role | n/a | symmetric leaves (no N/P) |
| Predicate direction | high_tail / low_tail | per operand |
| Parameter coordinate system | unique boundary thr_index | quantile thr_index |
| Symmetry | n/a | D4 lex sort |
| Mask equivalence | `mask_sha256` | + `semantic_duplicate_mask` |
| Semantic equivalence | operator+operands+lattice | **≠** mask equality |

### 5.2 G3

```text
Concrete cell semantics: OR(atom_a, atom_b) at a coordinate — available on atlas rows
but there are no PS/OS cells.

Grammar-level null: search_domain / semantic_family of registered OR lattice.
semantic_definition_id is NULL on the null record (CR4).
Optional per-grid null summaries still do not invent one family-wide concrete tree.
```

### 5.3 Mask vs semantic equivalence

```text
mask-equal ⇏ semantically equal
semantically equal ⇏ same region_asset
cross-grammar mask-string overlap = diagnostic only (not G7)
```

### 5.4 Future G7 — leave unresolved

```text
necessary operand role
support operand role
GT-envelope semantics
logical complement identity
envelope-relative coordinate
```

Do not infer from mask overlap.

---

## 6. G1–G3 A0→A1 gap analysis

Preserved C0 conclusion:

> Within registered Q4.5 G1–G3 lattices, 154 PS coordinates are thin/edge-dominated; `full_neighborhood_safe_radius ≥ 1` is **0/154**; terminal **B** retained. **Not** portable safe region / production candidate / G7 equivalence.

### 6.1 G1 Singleton

| A1 dimension | Status |
|:--|:--|
| Already exists | 1 PS cell; geometry; capacity; sequence; dual margins; mask; atom semantics |
| Deterministically repackagable | yes (sealed atlases + content IDs) |
| Needs R0-B decision | pack composition; alias |
| Blocked | none for A1 core |
| Evaluator rerun? | **no** |
| Non-null assets? | **yes** (1) |
| Null record? | no |

### 6.2 G2 Pairwise AND

| A1 dimension | Status |
|:--|:--|
| Already exists | 153 PS; 25 components; multi-mask regions; thin geometry; margins radius 0 |
| Deterministically repackagable | yes |
| Needs R0-B decision | D4 operand order; distribution summary quantiles to emit |
| Blocked / invalid | **additive** component capacity (retracted); event-union capacity |
| Evaluator rerun? | **no** |
| Non-null assets? | **yes** (thin; zero thick) |

### 6.3 G3 Hard OR

| A1 dimension | Status |
|:--|:--|
| Already exists | complete empty PS/OS on 17640 cells |
| Deterministically repackagable | as **grammar/search-domain null record** |
| Needs R0-B decision | null_reason taxonomy; optional per-grid null children |
| Blocked if missing-files-as-null | policy-blocked |
| Evaluator rerun? | **no** |
| `n_non_null_region_assets` | **0** |
| `n_null_records` | **1** (default) |

### 6.4 R1 feasibility statement

```text
R1 can be scoped as deterministic A0→A1 conversion packaging:
  inputs = sealed Q4.5 full atlases + T0-B-R1 pack + accepted R0-B schema
  outputs = proposed §8 files
  no evaluator modification/rerun
  no new geometry research claims
  content IDs independent of pack_id
  component capacity = distributions only
  sequence union/intersection = descriptive only
  G3 = null_record with nullable semantic_definition_id
  region↔mask via coordinate FKs
  maturity field remains non-accepted until chat-side A1 acceptance
  action_state = observation_only; production_forbidden = true
```

---

## 7. G3 null-asset contract (CR4)

Concrete case: **G3 Hard OR** registered pairwise OR lattice (40 grids, 17640 cells).

### 7.1 Grammar / search-domain null record

```text
null_record_id:            <digest(truth_context_id, search_domain_id, null_reason_class)>
truth_context_id:          TC0
pack_id:                   <pack emission; does not define null_record_id>
producer_kind:             grammar_atlas
grammar_id:                G3_or
search_domain_id:          <OR + primary_quantile_lattice_q05 + signal family + 40×441>
semantic_definition_id:    NULL
semantic_family_id:        same as search_domain_id (or 1:1 alias)
declared_search_domain:
  lattice_kind:            primary_quantile_lattice_q05
  combinator:              OR
  n_registered_coordinates: 17640
  n_registered_grids:      40
  thr_index_range:         [0,20] × [0,20]
registered_lattice_denominator: 17640
observed_safe_count:       0
productive_safe_count:     0
n_non_null_region_assets:  0
n_null_records:            1
region_asset_count:        0          # synonym of n_non_null_region_assets only
null_reason:               no_observed_or_productive_safe_on_registered_or_lattice
bounded_status:            NULL_RESULT
maturity_level:            A0
action_state:              observation_only
production_forbidden:      true
provenance:
  pairwise_or_atlas_parquet_sha256: bc1c2938…
  t0_grammar_area_summary_sha256:   e37d80a6…
  terminal:                B isolated_safe_points_only
claim_boundary:
  maximum_claim:
    registered OR lattice contains no productive-safe cell under accepted
    label/unresolved contracts (search-domain null; not one concrete operator tree)
  forbidden_promotions:
    - treat_missing_files_as_null_result
    - invent_single_concrete_semantic_definition_for_all_40_grids
    - claim_OR_grammar_universally_useless_off_lattice
    - production_gate
    - portable_safe_region
    - G7_equivalence
```

### 7.2 Count disambiguation

```text
n_non_null_region_assets = 0   # no HAS_REGION rows for G3
n_null_records           = 1   # one grammar/search-domain null row
#region_asset_count      = 0   # == n_non_null_region_assets; does NOT count null rows
```

Having `n_null_records=1` does **not** contradict `region_asset_count=0`.

### 7.3 Optional per-grid null summaries

May emit 40 child rows with `grid_id`, zero counts, FK `null_record_id` parent — still **no** concrete `semantic_definition_id` unless a future contract defines per-grid concrete empty semantics separately.

### 7.4 Forbidden encodings

```text
❌ omit G3 from pack
❌ empty components file only
❌ absent pairwise_or_atlas interpreted as null
❌ semantic_definition_id pointing at "pairwise atom family" fake tree
❌ null_reason = "not computed"
```

---

## 8. Proposed R1 machine schemas

Contract proposal only — **do not emit these files**.

### 8.1 `region_asset_manifest.json`

| | |
|:--|:--|
| Grain | one pack emission |
| PK | `pack_id` |
| Required | schema_version, pack_id, truth_context_id, producer_*, study_id, substrate_id, signal_family, sequence_set, label_contract, unresolved_policy, lattice_contract, evaluator_source_sha256, truth_input_artifact_hashes, terminal_letter, maturity_declared, action_state_default, production_forbidden, grammar_scope, counts (`n_non_null_region_assets`, `n_null_records`, `n_mask_units`, `n_coordinates_ps`), generation block |
| Optional | human title, notes, t0_revision |
| Invariants | production_forbidden==true; maturity_declared≠A1 unless external acceptance; content IDs not functions of pack_id |

### 8.2 `grammar_region_summary.csv`

| | |
|:--|:--|
| Grain | one row per grammar within pack |
| PK | `(pack_id, grammar)` |
| Required | n_registered_coords, n_observed_safe, n_productive_safe, **n_non_null_region_assets**, **n_null_records**, n_per_grid_mask_units_ps, coordinate_productive_area_ratio, unique_mask_productive_ratio_micro, max_full_neighborhood_safe_radius_over_ps |
| Invariants | G3: n_productive_safe==0, n_non_null_region_assets==0, n_null_records≥1; G1+G2+G3 PS sums to 154 |

### 8.3 `region_assets.csv`

| | |
|:--|:--|
| Grain | **non-null** region/component only |
| PK | `region_asset_id` |
| FK | truth_context_id, semantic_definition_id, grid_id; pack listing via evidence_record or pack membership table |
| Required | grammar, bounded_status=HAS_REGION, n_coords, n_mask_units, shape_class, is_genuine_2d_thick, max_full_neighborhood_safe_radius, human_alias, action_state, production_forbidden |
| Invariants | n_coords≥1; no ::comp ordinal as PK; pack_id not required to compute PK |

### 8.4 `null_records.csv` (explicit; not folded into region_assets)

| | |
|:--|:--|
| Grain | grammar/search-domain null |
| PK | `null_record_id` |
| FK | truth_context_id, search_domain_id |
| Required | grammar, semantic_definition_id **NULL**, declared_search_domain fields, observed_safe_count, productive_safe_count, null_reason, bounded_status=NULL_RESULT, action_state, production_forbidden |
| Optional | parent for per-grid null children |
| Invariants | semantic_definition_id is null; counts match domain zeros |

### 8.5 `region_components.csv`

| | |
|:--|:--|
| Grain | geometry detail for non-null regions |
| PK | `region_asset_id` |
| Required | grid_id, adjacency_contract, axis_span_*, bounding_box_*, active_axis_count, coords_digest, coords_json, t0_component_id_alias |
| Invariants | coords sorted; digest matches; t0 alias not PK |

### 8.6 `region_masks.csv`

| | |
|:--|:--|
| Grain | per-grid mask unit |
| PK | `mask_unit_id` |
| FK | truth_context_id, grid_id |
| Required | grammar, mask_sha256, n_coords, mask_n_neg, mask_n_gt, n_sequences_with_neg, productive_sequences_json, coordinate_membership_digest |
| **Forbidden columns as authority** | `region_asset_ids_json` |
| Optional diagnostic | global_mask_multi_grid_count |
| Invariants | PK includes grid_id via construction |

### 8.7 `region_coordinates.csv` (**authoritative M:N derivation source** — CR5)

| | |
|:--|:--|
| Grain | productive-safe coordinate (A1 minimum) |
| PK | `coordinate_id` |
| FK | truth_context_id, **region_asset_id**, **mask_unit_id**, grid_id |
| Required | grammar, cell_id, thr indices, observed/productive flags, n_neg_captured, gt_hurt, safety_status, mask_sha256, dual margins, edge flags |
| Optional | thr_value*, per_sequence_neg_json, loo_*, n_unresolved_selected |
| Invariants | every PS coord has exactly one region_asset_id and one mask_unit_id within grid; thr_value not in coordinate_id |

### 8.8 `region_mask_link.csv` (optional derived)

| | |
|:--|:--|
| Grain | (region, mask_unit) pair |
| PK | `(region_asset_id, mask_unit_id)` |
| Required | n_coords_in_link (count of coordinates realizing the pair) |
| Source | `SELECT DISTINCT region_asset_id, mask_unit_id, COUNT(*) FROM region_coordinates GROUP BY 1,2` |
| Invariants | must equal projection from coordinates; never hand-authored |

### 8.9 `region_capacity.csv`

| | |
|:--|:--|
| Grain | `unit` ∈ {coordinate, mask_unit, region_asset} |
| PK | `(unit, unit_id, metric_name)` or wide region row with metric columns |
| Coordinate/mask required | n_neg (mask_n_neg), n_sequences_with_neg, productive_sequences_json, min_positive_sequence* |
| Region required metrics | **distribution only**: `member_capacity_min`, `member_capacity_max`, `member_capacity_median`, `member_capacity_q25`, `member_capacity_q75`, `member_capacity_robust_floor`, `member_grain` ∈ {coordinate, mask_unit}, `n_members` |
| Forbidden | `member_capacity_sum` as primary capacity; plateau-multiplied capacity |
| Event-union metric | **omit** or emit with status `BLOCKED_BY_ARTIFACT` only |
| Invariants | region metrics are not additive across alternative parameters |

### 8.10 `region_sequence_support.csv` (CR3)

Two complementary tables (or one long + one summary):

**A. Incidence (authoritative long format)**

| | |
|:--|:--|
| Grain | `(unit, unit_id, sequence)` with unit ∈ {coordinate, mask_unit} |
| PK | `(unit, unit_id, sequence)` |
| Required | n_neg, support flag |
| Source | expand productive_sequences_json / per_sequence |

**B. Region descriptive summary**

| | |
|:--|:--|
| Grain | region_asset |
| PK | `region_asset_id` |
| Required | `sequence_support_union_json`, `sequence_support_intersection_json`, `n_seq_union`, `n_seq_intersection`, `min_member_n_sequences`, `max_member_n_sequences`, `dominance_note` |
| Forbidden claim | treating union as A2 applicability / common parameter choice |
| Invariants | intersection ⊆ union; both may be empty for degenerate cases |

### 8.11 `region_margin.csv`

| | |
|:--|:--|
| Grain | coordinate |
| PK | `coordinate_id` |
| Required | nearest_unsafe_distance, full_neighborhood_safe_radius, distance_to_lattice_edge, nearest_unsafe_edge_censored, edge_policy_id |
| Invariants | dual metrics both present |

### 8.12 `region_claim_contract.json`

| | |
|:--|:--|
| Grain | pack |
| PK | pack_id |
| Required | maximum_research_claim, forbidden_promotions[], action_states_allowed/forbidden, maturity gates A0–A4 refs, terminal_b, not_a_safe_rule, g7_status, nested_loso_portability_note, production_preset=unchanged, evidence_ledger=not_promoted, identity_layer_policy (content ⟂ pack), capacity_policy (non-additive), sequence_policy (union≠applicability) |
| Invariants | A1 excludes transfer/intervention permissions |

### 8.13 Pack-level validation invariants

```text
sum_PS(G1,G2,G3) == 154 for this sealed study
G3: n_non_null_region_assets==0 AND n_null_records>=1
no region_asset_id depends on pack_id, row order, or component ordinal
every mask_unit_id includes grid_id in construction
region_mask_link == projection(region_coordinates) if present
no additive component capacity metric
sequence union/intersection both present for multi-member regions
production_forbidden == true
action_state == observation_only
evaluator not rerun; truth hashes match seal
```

---

## 9. Maturity and claim firewall

### 9.1 Maturity levels

| Level | Required evidence | Permitted actions | Forbidden |
|:--|:--|:--|:--|
| **A0** | enumeration or null; accepted T0/C0 | describe | consume as stable transferable asset |
| **A1** | stable **content** IDs+provenance; semantic def; grain relations; dual geometry; capacity distributions; sequence incidence+union+intersection; null form; claim firewall | compare, rank, diff, reproduce | transfer; intervention; production |
| **A2** | A1 + contraction + transport + region LOO/equiv + held-out harm + productive floor + unresolved firewall | condition modeling / shadow candidates | default intervention; production |
| **A3** | A2 + action-time observables + frozen representative + default-off + online support + control + monitoring/rollback | application validation default-off | production default |
| **A4** | separate formal promotion + ops ownership | production under governance | silent auto-promotion |

### 9.2 Independent gates

```text
artifact generated
engineering ready
asset maturity accepted
research conclusion accepted
intervention qualified
production approved
```

### 9.3 Action states

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
sequence_support_union ≠ applicability
component capacity distribution ≠ simultaneous multi-mask action
null_record ≠ concrete semantic definition
pack_id change ≠ content identity change
```

---

## 10. Cross-family boundary

### 10.1 Core shareable fields

```text
truth_context_id, pack_id (emission), producer_kind, producer_contract_version
substrate_id, cohort_id, label_contract_id, unresolved_policy_id
provenance hashes
region_asset_id, semantic_definition_id (when concrete), search_domain_id
mask_unit_id, coordinate_id
geometry dual margins, capacity distributions, sequence incidence/union/intersection
maturity_level, action_state, production_forbidden
claim maximum + forbidden promotions
null_record shape
```

### 10.2 Opaque adapter payloads

```text
grammar: operator trees, lattice_kind, thr registries
occ-exit: episode ids, gap paths, cover features
association: cost terms, gate knobs
relink: bridge geometry, appearance veto, bank keys
```

No inheritance framework or runtime API.

### 10.3 Intervention-time fields (not A1 core)

```text
action-time observable binding
control/treatment assignment
online trigger effect proof
rollback/monitoring contracts
transport retention metrics
```

---

## 11. Decisions required before R0-B

| decision_id | question | recommended_default | alternatives | evidence | falsifier | consequence_for_R1 | owner_required |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **D1** | Pack grammar_scope composition? | One pack `G1_G2_G3` with grammar column | 3 packs | single study | if independent seals needed | pack_id only | research |
| **D2** | ID scheme pin after CR1? | `region_asset_id_v1` + 32-hex display | full 256-bit only | §4 | collision | all PKs | research |
| **D3** | semantic_definition threshold-free? | **yes** | include thr families | grammar shared across thr | thr-specific semantics | region ids use membership | research |
| **D4** | AND operand order? | lex `(feature,direction)` | preserve atom_a/b order | symmetry | asymmetric roles | semantic_id | research |
| **D5** | Primary region grain? | connected PS component within grid | mask unit primary | T0 26 components | adjacency instability | region_assets | research |
| **D6** | Human alias format? | grid+shape+n_coords | thr-value-first | alias≠id | alias used as join | docs only | eng+research |
| **D7** | G3 null structure? | **one grammar/search-domain null_record** + nullable semantic_definition_id + optional per-grid children | per-grid nulls only | CR4; 40 grids | if concrete empty trees required | null_records.csv | research |
| **D8** | Primary mask unit? | (truth_context, grid_id, mask_sha256) | global mask (**reject**) | multi-grid masks | global-only joins | mask PK | research |
| **D9** | coordinate_id form? | truth_context-scoped native cell_id | pure index digest | native stable | cross-study collision without context | coordinates PK | research |
| **D10** | Component capacity? | **distribution** over member coordinate **or** mask capacities (min/max/median/quantiles/robust floor); **no sum** | sum (**rejected**) | CR2; multi-mask e.g. caps {1,2,3,4} | if event-union bitsets appear | region_capacity | research |
| **D10u** | Event-union capacity? | **omit / BLOCKED_BY_ARTIFACT** | sealed membership reconstruction (future) | no event bitsets on atlas | if membership pack sealed | no fake union metric | research |
| **D11** | Sequence region summary? | emit **incidence long-format** + region **union and intersection** + min/max member support counts | union-only (**reject**) | CR3 | if union treated as applicability | sequence tables | research |
| **D12** | Default action_state? | `observation_only` + `production_forbidden=true` | none | A1 firewall | auto shadow flag | claim_contract | research |
| **D13** | Region↔mask authority? | **coordinate FKs authoritative**; optional derived `region_mask_link` | link-only authority (**weaker**) | CR5; 4 multi-mask components | JSON list authority | join model | research |

### 11.1 Decision classes

| Class | IDs |
|:--|:--|
| Closed by R0-A-R1 contract correction (defaults locked for review) | CR1 layers; D10 non-additive; D10u blocked; D11 union∩intersection; D7 null grain; D13 coordinate authority |
| Remaining naming/schema choices for R0-B | D1, D2, D3, D4, D5, D6, D8, D9, D12, optional quantile list in D10 |
| Unresolved research semantics | G7 roles |
| Artifact/provenance blockers | atlases absent in some envs; event membership for union capacity; live tree drift if regeneration claimed |

### 11.2 Fixed by accepted truth (not reopenable here)

```text
terminal B retained
per-grid mask primary unit
no global mask PK
no component ordinal identity
dual margin policy fixed in T0
G7 roles not inventable
A1 non-transferable / non-actionable
no evaluator rerun for R1
content identity ⟂ pack identity
```

---

## 12. Synthetic aggregation examples (CR2, CR3)

### 12.1 Component capacity — multi-mask strip (observed pattern)

Observed (T0): component  
`P::abs_log_h::high_tail__score_m_bridge::low_tail::comp0`  
has **12** coordinates and **4** distinct per-grid mask units with `mask_n_neg ∈ {1,2,3,4}`.

```text
LEGAL descriptive metrics (member_grain = mask_unit):
  n_members = 4
  min = 1, max = 4, median = 2.5, q25 = 1.5, q75 = 3.5
  robust_floor = min = 1   # example floor definition

ILLEGAL:
  sum = 1+2+3+4 = 10   # treats alternative masks as disjoint simultaneous mass

EVENT-UNION capacity:
  status = BLOCKED_BY_ARTIFACT
  reason = atlas stores counts + mask_sha256, not invertible event membership
```

Same component with `member_grain = coordinate` uses the 12 coordinate `n_neg_captured` values analogously — still **not** a sum-as-region-mass claim without stating double-count policy, and plateau-adjacent cells often share masks so coordinate-sum also overcounts unique-mask mass.

### 12.2 Sequence union ≠ applicability

```text
member coordinates productive sequences:
  c1: {MOT17-10-SDP}
  c2: {MOT17-10-SDP, MOT17-13-SDP}
  c3: {MOT17-13-SDP}

union        = {MOT17-10-SDP, MOT17-13-SDP}   # some thr worked somewhere
intersection = {}                             # no sequence supported by all members
min_n_seq    = 1
max_n_seq    = 2

A1 claim: descriptive geometry only
A2 claim: FORBIDDEN without representative selection / transport contract
```

### 12.3 G3 counts

```text
n_non_null_region_assets = 0
n_null_records = 1
semantic_definition_id on null row = NULL
search_domain_id present
```

---

## 13. R1 readiness recommendation (not authorization)

```text
IF R0-B accepts remaining D* naming defaults
AND full atlases remain hash-sealed
AND CR1–CR5 corrections remain in force
THEN R1 = deterministic conversion packaging A0 → candidate A1 files
ELSE IF atlases missing THEN R1 BLOCKED_BY_ARTIFACT
ELSE IF event-union capacity demanded THEN that metric BLOCKED_BY_ARTIFACT (rest may proceed)

R1 authorization still requires separate chat-side gate after R0-B.
A1 research acceptance is a further gate after R1 engineering review.
R0-A-R1 itself is not self-accepted.
```

---

## 14. Acceptance checks (R0-A-R1 self-audit)

```text
[x] every proposed field has a bounded derivability status
[x] asset/component/mask/coordinate/null grains remain distinct
[x] content IDs do not depend on row/component ordinal
[x] content IDs do not depend on pack_id / unrelated grammar scope / schema version
[x] per-grid mask unit remains primary
[x] semantic role identity is not collapsed into mask identity
[x] component capacity is non-additive distribution; sum retracted
[x] event-union capacity BLOCKED_BY_ARTIFACT without membership
[x] sequence incidence + union + intersection preserved; union ≠ applicability
[x] G3 null is search-domain grain with nullable semantic_definition_id
[x] n_non_null_region_assets vs n_null_records disambiguated
[x] region↔mask M:N authoritative via coordinate FKs
[x] A0–A4 maturity and action firewalls explicit
[x] R1 scoped as deterministic conversion conditional on R0-B + seals
[x] no evaluator rerun or modification
[x] no R1 asset files generated
[x] no G4–G7 implementation
[x] terminal B unchanged
[x] production/presets unchanged
[x] evidence ledger unchanged
[x] maturity A0 retained
```

---

## 15. Explicit non-claims

```text
no evaluator rerun
no asset pack generated
no research verdict self-accepted
no R0-A / R0-A-R1 acceptance declared
no R0-B / R1 authorized
no A0→A1 maturity promotion
no PR opened by this note alone
no evidence_ledger promotion
```

---

## 16. Next gate

```text
R0-A-R1 contract correction complete
→ chat-side re-review of this note
→ only if accepted: treat R0-A as accepted and authorize R0-B final contract
→ only after R0-B acceptance: authorize R1 deterministic packaging
```
