# R0-A Region Asset Contract Preflight

<!-- doc-status: active -->
<!-- doc-promotion: navigation-support; not evidence_ledger; not research verdict; not A1 acceptance -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->
<!-- fact-owner: this file for R0-A field/identity/schema derivability only -->

**Task:** R0-A Region Asset Contract Preflight · revision **R0-A-R2** (mathematical/identity normalization)  
**Branch:** `research/composition-grammar-coverage-program`  
**Program thread:** [safe_region_assetization_20260710.md](../../../research/threads/safe_region_assetization_20260710.md)  
**Mathematical contract (canonical):** [statistical_robust_feasible_set_estimation_under_asymmetric_loss.md](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)  
**A0 baseline:** [composition_grammar_t0_region_interpretation_20260710.md](composition_grammar_t0_region_interpretation_20260710.md)  
**T0-A preflight:** [composition_grammar_t0_artifact_preflight_20260710.md](composition_grammar_t0_artifact_preflight_20260710.md)

> Answers **which RegionAsset fields/grains/identities are DIRECT, DERIVABLE, contract-dependent, or blocked** from accepted G1–G3 T0/C0 truth, under the shared feasible-set framework.  
> **Not** R0-B acceptance · **not** R1 asset generation · **not** evaluator rerun · **not** G4–G7 · **not** A0→A1 promotion.

```text
Research acceptance remains chat-side after review.
This note is an evidence packet, not a verdict.

Revision chain:
  R0-A     762adf9a  initial preflight
  R0-A-R1  136841a8  CR1–CR5 structural (PASS retained)
  R0-A-R2  (this)    CR6–CR9 math/identity normalization
```

---

## 0. Preflight headline

```text
R0-A-R2 MATH/IDENTITY NORMALIZATION COMPLETE (awaiting chat-side re-review):

  Provenance seals: Q4.5 + T0-B-R1 hash-verified (unchanged).
  Terminal B unchanged: isolated_safe_points_only.
  Maturity: A0 retained.
  Statistical claim ceiling (this study): L1 in-sample registered-lattice region geometry
    (not L2+ held-out region; not population risk zero).

  CR1–CR5: PASS (retained; must not regress).
  CR6–CR9: addressed in this revision.

  CR6: truth_contract_id (normalized) ⟂ evidence_bundle_id (raw seals)
       content IDs parent only truth_contract_id + local content
  CR7: feasibility_contract_id first-class; A0–A4 ⟂ L0–L6 ladders
  CR8: search_domain membership enumerates all 40 concrete OR grid semantics
  CR9: machine authority tables for all FKs; dual capacity distributions

  R1 recommendation (not authorized):
    deterministic A0→A1 packaging FEASIBLE after R0-B + sealed atlases
    no evaluator rerun for proposed A1 core
```

### Change log

| Rev | Scope |
|:--|:--|
| R0-A-R1 | pack ⟂ content; non-additive capacity; sequence union∩; G3 null grain; coord FK M:N |
| R0-A-R2 | truth_contract ⟂ evidence_bundle; feasibility contract; domain membership; FK machine tables; dual capacity distributions; math framework terminology |

---

## 1. Truth and provenance base

### 1.0 Canonical mathematical framework (input)

Shared contract (not semantic-private):

[docs/research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)

Maps this preflight’s objects to framework terms:

| Framework | This study (Q4.5 G1–G3) |
|:--|:--|
| $\Theta$ parameter space | registered thr-index lattices (G1 unique-boundary; G2/G3 quantile) |
| $\Omega$ candidate universe | `online_hook_eligible` on substrate `stage1_baudit_d_online` |
| $L_{\mathrm{GT}}$ | resolved `gt_hurt` / GT exposure under unresolved firewall |
| $\varepsilon$ | exact-zero count contract: `N_GT,hurt = 0` on resolved sample |
| $G_{\mathrm{FP}}$ / productivity | count surrogate `n_neg_captured > 0` (rate optional) |
| $\widehat{\mathcal S}$ productive-safe set | cells with `productive_safe_point` |
| coordinate / mask / semantic area | registered cells; per-grid `mask_sha256`; operator trees |
| region geometry | T0 components + dual margins |
| claim ladder L0–L6 | orthogonal to asset maturity A0–A4 |

### 1.1 Source studies and hashes (evidence seal material)

#### Q4.5 atlas study

| Item | Value |
|:--|:--|
| Study root | `out/signal_study/m_b1_5_stage2_q45_20260710/` |
| Study id | `m_b1_5_stage2_q45_20260710` |
| Manifest schema | `m_b1_5_stage2_q45_atlas_manifest_v4` |
| Taxonomy | `stage2_q45_atlas_v4` |
| Study-producing git commit | `dc758e088de9fe2bfed7e2d4d458a8360a03f712` |
| Evaluator SHA256 (recorded) | `551284f88710945dc636cb4e13f2b8401948fc717e605738f84be83b9e133643` |
| Runner SHA256 (recorded) | `376b11c0bb9c85823ca6c09eb17ed3044730b3be7bc9fec53888ffde74d5815a` |
| Source event table SHA256 | `cfca3818fd8478e6e3dcb12d3976549dab8057a0f2c2e63831f8d3e3a2fffd97` |
| Terminal | **B** `isolated_safe_points_only` |
| Production preset | unchanged |

**Runtime hashes vs manifest (verified; belong to `evidence_bundle_id`, not content IDs):**

| Artifact | Manifest key | SHA256 prefix | Match |
|:--|:--|:--|:--|
| `atom_atlas.parquet` | `atom_atlas_parquet` | `281cb22bd92daf48…` | yes |
| `pairwise_and_atlas.parquet` | `pairwise_and_atlas_parquet` | `ae52bd0cb799aaa8…` | yes |
| `pairwise_or_atlas.parquet` | `pairwise_or_atlas_parquet` | `bc1c2938775caa1e…` | yes |
| `region_stability.csv` | `region_stability` | `87055ef93439ae54…` | yes |
| `per_sequence.csv` | `per_sequence` | `5446ac659ecada38…` | yes |
| `threshold_registry.json` | `threshold_registry` | `d3e3197fa7812a9e…` | yes |
| `summary.json` | `summary` | `a88d9dcfc5a61449…` | yes |
| `manifest.json` | self / T0 input | `4213e82e4c05a052…` | yes |

#### T0-B pack (A0 geometry)

| Item | Value |
|:--|:--|
| Committed pack | `docs/modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/` |
| Revision | **T0-B-R1** |
| SHA256SUMS | 12/12 match |
| Dual-margin policy | pack `summary.json` / T0 note §1 |

### 1.2 Signal family / substrate / cohort

| Field | Value |
|:--|:--|
| Signals | `score_m_bridge`, `abs_log_h`, `dist_h`, `abs_ratio_m1`, `resid_mean` |
| Directions | `high_tail`, `low_tail` |
| Combinators | `AND`, `OR` only |
| Substrate | `stage1_baudit_d_online` |
| Candidate universe | `online_hook_eligible` |
| Sequences | 7× MOT17-*-SDP |
| Primary cohort n | 87 = 23 neg + 64 GT protect |
| Selected unresolved | 21 |
| D_online events | 244 |

### 1.3 Label + unresolved (operational)

```text
negative_class:
  resolved ∧ baseline_selected ∧ pair_label == negative
  → n_fp_exposed = 23 (primary)

positive_protection_class:
  resolved ∧ baseline_selected ∧ pair_label == gt_consistent
  → n_gt_exposed = 64 (primary)

excluded_from_main: unresolved, ambiguous, non-selected, other
unresolved_contaminated_blocks_candidate: true
```

Productive-safe (do not reopen):

```text
productive_safe_point ⇔
  resolved GT_hurt == 0
  AND n_neg_captured > 0
  AND no unresolved contamination that blocks candidate
```

### 1.4 Lattice / coordinate contracts

| Grammar | Lattice | Registered size |
|:--|:--|:--|
| G1 | `primary_unique_boundaries` thr_index ∈ [0,86] | 870 |
| G2 AND | `primary_quantile_lattice_q05` thr ∈ [0,20] | 17640 = 40×441 |
| G3 OR | same as G2 | 17640 |

```text
S::  ⟂  P::
semantic_duplicate_is_per_grid_not_global: true
```

### 1.5 Dual-margin / nested LOSO (retained)

Dual margin: T0 conservative edge policy; radius≥1 = 0/154.  
Nested LOSO: 0 exact-absolute portable clauses — **not** region transfer (L2+).

### 1.6 Live tree drift

Evaluator/T0 script live SHAs ≠ study-recorded SHAs.  
**Evidence-bundle** pins recorded SHAs; content IDs do not re-hash live tree.

### 1.7 Committed vs runtime

| Input | Committed? | Role |
|:--|:--|:--|
| Q4.5 pack subset | yes | evidence seal subset |
| T0 pack | yes | geometry/capacity/sequence |
| Full atlases | runtime | required for R1 lattice membership |
| Event membership bitsets | **absent** | event-union capacity blocked |
| Live evaluator match | no | not required if artifacts sealed |

### 1.8 Preserved C0 numbers

```text
PS: 154 = 1 G1 + 153 G2 + 0 G3
components: 26
productive per-grid mask units: 34
radius ≥ 1: 0/154
G3 OS=PS=0 on 17640
terminal B retained
mask units spanning >1 component (observed): 0
components with >1 mask unit: 4
→ still model M:N in schema
```

---

## 2. Field derivability matrix

Statuses: `DIRECT` | `DETERMINISTICALLY_DERIVABLE` | `REQUIRES_CONTRACT_DECISION` | `BLOCKED_BY_ARTIFACT` | `BLOCKED_BY_PROVENANCE` | `NOT_APPLICABLE`

### 2.1 Identity layers (CR1 + CR6)

```text
truth_contract_id       # normalized semantic/data truth (order-insensitive)
evidence_bundle_id      # exact raw artifacts + recorded source seals
feasibility_contract_id # asymmetric-loss / productive-safe definition (CR7)
pack_id                 # materialization/schema/generator
region_asset_id         # content: truth_contract + local membership
mask_unit_id / coordinate_id
semantic_definition_id / search_domain_id
null_record_id
evidence_record_id      # optional pack-local instance
```

**Retired:** R0-A `asset_set_id`; R0-A-R1 `truth_context_id` as sole parent that embedded raw file SHAs.

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `truth_contract_id` | truth contract | DETERMINISTICALLY_DERIVABLE | normalized contracts + content digests | §4.2 — **no raw file byte SHAs** | digest field set | — | semantic identity |
| `evidence_bundle_id` | evidence seal | DIRECT / DERIVABLE | Q4.5+T0 manifests | raw hashes + recorded evaluator/runner SHAs | — | missing atlases | provenance only |
| `pack_id` | pack | REQUIRES_CONTRACT_DECISION | R0-B | truth_contract + feasibility + schema + grammar_scope + generator | D1 | — | emission only |
| `feasibility_contract_id` | feasibility | DETERMINISTICALLY_DERIVABLE | labels + PS def + denominators + edge policy | §4.2 / §7a | D14 freeze fields | — | L-level owner |
| `producer_kind` | pack | REQUIRES_CONTRACT_DECISION | program | `grammar_atlas` | enum | — | observation |
| `producer_contract_version` | pack | REQUIRES_CONTRACT_DECISION | R0-B | e.g. `region_asset_v0` | — | — | changes pack_id only |
| `grammar_scope` | pack | REQUIRES_CONTRACT_DECISION | R0-B | e.g. `G1_G2_G3` | multi vs split packs | — | pack membership |
| `signal_family` / lattices / substrate / cohort / label / unresolved | truth contract | DIRECT | registry/manifest | normalized sorted fields | — | — | truth |
| `normalized_data_content_digest` | truth contract | DETERMINISTICALLY_DERIVABLE | atlases+T0 under order-insensitive rules | §4.3 | field list | atlases missing | truth without raw order |
| raw `input_artifact_hashes` | evidence bundle | DIRECT | manifests | full SHA map | — | — | **not** content parent |
| live evaluator re-hash | — | BLOCKED_BY_PROVENANCE if used as identity | tree drift | pin recorded | — | drift | do not recompute |

### 2.2 Semantic definition + search domain (CR4 + CR8)

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `semantic_definition_id` | concrete tree | DETERMINISTICALLY_DERIVABLE | operands+operator+lattice | §4; D4 lex sort | D4 | — | ≠ mask |
| `search_domain_id` | search domain | DETERMINISTICALLY_DERIVABLE | **ordered membership** of concrete grids/semantics | digest of sorted member rows (§4.2) | — | summary-only fields insufficient | CR8 |
| `grid_domain_id` | registered grid | DETERMINISTICALLY_DERIVABLE | axes + lattice | e.g. digest of grid_id + lattice_kind | — | — | domain member |
| `search_domain_members` | membership | DETERMINISTICALLY_DERIVABLE | G1 10 grids / G2 40 / G3 40 | rows: domain→grid→concrete semantic→denominator | — | — | CR8 authority |
| G7 roles / envelope / complement | — | BLOCKED / NOT_APPLICABLE | g7 gap | leave open | — | no roles | no G7 claim |

### 2.3 Feasibility contract fields (CR7)

| field_name | grain | status | source_owner | derivation_rule | required_contract_decision | blocker | claim_boundary |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `parameter_or_policy_space_id` | feasibility | DETERMINISTICALLY_DERIVABLE | lattice contracts | G1/G2/G3 Θ descriptors | multi-space packaging | — | registered Θ only |
| `candidate_universe_id` | feasibility | DIRECT | manifest | `online_hook_eligible` | — | — | Ω |
| `safety_loss_definition` | feasibility | DIRECT | atlas PS flags + labels | resolved GT_hurt==0 + unresolved firewall | wording freeze | — | L_GT operational |
| `productivity_definition` | feasibility | DIRECT | atlas | `n_neg_captured > 0` count surrogate | rate vs count | — | G_FP surrogate |
| `epsilon` | feasibility | DIRECT | exact-zero contract | `epsilon_kind=exact_zero_count` (not rate ε) | — | — | finite-sample |
| `g_min` | feasibility | DIRECT | count surrogate | `g_min_kind=count_ge_1` | — | — | not rate floor |
| `n_gt_exposed` | feasibility | DIRECT | summary | **64** primary GT protect | secondary analyses separate | — | denominator |
| `n_fp_exposed` | feasibility | DIRECT | summary | **23** primary negative | — | — | denominator |
| `selection_scope` | feasibility | DIRECT | study design | `in_sample_searched_and_evaluated` | — | — | not held-out freeze |
| `finite_sample_statement` | feasibility | DETERMINISTICALLY_DERIVABLE | framework | observed GT0 ≠ population risk 0 | — | — | claim bound |
| `metric_adjacency_edge_policy` | feasibility | DIRECT | T0 dual-margin policy | bilateral/4-neigh; off-lattice ⇒ radius 0 | — | — | geometry |
| `claim_level_max` | feasibility / asset | DETERMINISTICALLY_DERIVABLE | evidence | **L1** for G1/G2 PS geometry; G3 null at L0 domain-empty | — | no L2 region | ⟂ A-level |
| population confidence bound | — | NOT_APPLICABLE / absent | — | **not established** | — | — | forbidden claim |

### 2.4 Region / mask / coordinate (CR1 retained; parents updated CR6)

| field_name | grain | status | derivation | claim_boundary |
|:--|:--|:--|:--|:--|
| `region_asset_id` | region | DETERMINISTICALLY_DERIVABLE | digest(**truth_contract_id**, feasibility_contract_id?, semantic_definition_id, grid_id, adjacency, coord membership) — **not** pack_id, **not** evidence_bundle_id | content |
| `mask_unit_id` | mask unit | DETERMINISTICALLY_DERIVABLE | digest(truth_contract_id, grid_id, mask_sha256) | primary mask unit |
| `coordinate_id` | coordinate | DETERMINISTICALLY_DERIVABLE | digest(truth_contract_id, cell_id) | registered cell |
| T0 `::compN` | alias | DIRECT non-stable | diagnostic only | not PK |
| `region_asset_id` / `mask_unit_id` FKs on coordinates | link | DETERMINISTICALLY_DERIVABLE | CR5 authoritative | M:N |

### 2.5 Null asset (CR4 + CR8)

| field_name | grain | status | notes |
|:--|:--|:--|:--|
| `null_record_id` | null | DETERMINISTICALLY_DERIVABLE | truth_contract + search_domain + null_reason_class |
| `semantic_definition_id` | null row | **NULL** | grammar-level null |
| member concrete semantics | members | DIRECT | 40 OR grids each have concrete definition |
| `n_non_null_region_assets` | summary | DIRECT | G3 = 0 |
| `n_null_records` | summary | DERIVABLE | G3 = 1 default |
| missing files as null | — | **illegal** | BLOCKED_BY_ARTIFACT |

### 2.6 Geometry (retained)

DIRECT from T0/atlases: areas, components, dual margins, plateaus. Cross-lattice raw count compare forbidden.

### 2.7 Productive capacity (CR2 + CR9)

| field_name | status | notes |
|:--|:--|:--|
| coordinate `n_neg_captured` | DIRECT | — |
| mask `mask_n_neg` | DIRECT | not × plateau |
| region capacity **sum** | **INVALID** | retracted CR2 |
| region **coordinate-member distribution** | DETERMINISTICALLY_DERIVABLE | min/max/median/quantiles/floor over member coords |
| region **mask-member distribution** | DETERMINISTICALLY_DERIVABLE | same over distinct per-grid mask units in region |
| event-union capacity | **BLOCKED_BY_ARTIFACT** | no event bitsets |
| pack concentration | DIRECT | T0 top-k on mask units |

**Both** coordinate-member and mask-member distributions are required (CR9); they answer different questions and must not be collapsed or sold as event-union.

### 2.8 Sequence geometry (CR3 retained)

Incidence long-format + region union + intersection + min/max member n_seq.  
Union ≠ A2 applicability.

### 2.9 Transfer / action / claim / maturity

| Topic | status |
|:--|:--|
| Nested LOSO portable | DIRECT 0; not region transfer |
| Absolute/quantile/region LOO | NOT_APPLICABLE A1 |
| action_state default | `observation_only` |
| production_forbidden | true |
| maturity current | **A0** |
| claim_level current max | **L1** (in-sample region geometry) |
| A-level vs L-level | **orthogonal** — A1 packaging may still be L1 only |

### 2.10 Derivability summary

| Status | Coverage |
|:--|:--|
| DIRECT | geometry, capacities, incidence inputs, denominators, seals |
| DETERMINISTICALLY_DERIVABLE | normalized digests, distributions, domain membership, feasibility id |
| REQUIRES_CONTRACT_DECISION | pack composition, alias, some serialization freezes |
| BLOCKED_BY_ARTIFACT | event-union; atlases absent; G7; region LOO |
| BLOCKED_BY_PROVENANCE | live re-hash as identity |
| NOT_APPLICABLE | A2–A4 actions; population risk; L2+ without evidence |

---

## 3. Grain and relation analysis

### 3.1 Distinct grains

```text
truth_contract
evidence_bundle
feasibility_contract
pack / materialization
semantic_definition (concrete)
search_domain + search_domain_members + grid_domain
region/component
per-grid mask unit
coordinate
null_record
evidence_record / pack_membership (pack-local)
```

### 3.2 Cardinalities (accepted study)

| Relation | Cardinality |
|:--|:--|
| truth_contract → evidence_bundles | 1 → many (re-runs/formats) |
| truth_contract → packs | 1 → many |
| G1 non-null regions | 1 |
| G2 non-null regions | 25 |
| G3 non-null regions | 0 |
| G3 null_records | 1 (grammar-level) |
| G3 search_domain_members | **40** concrete grids |
| components | 26 (154 coords) |
| productive mask units | 34 |
| region → masks | 1→many (4 multi-mask comps) |
| mask → regions | M:N in schema (0 multi-region masks observed) |

### 3.3 Keys

| Grain | PK | Parents (must / must-not) |
|:--|:--|:--|
| truth_contract | `truth_contract_id` | — |
| evidence_bundle | `evidence_bundle_id` | may reference truth_contract_id |
| feasibility_contract | `feasibility_contract_id` | may reference truth_contract_id |
| pack | `pack_id` | truth_contract, feasibility, evidence_bundle |
| semantic_definition | `semantic_definition_id` | not pack |
| search_domain | `search_domain_id` | membership digest |
| grid_domain | `grid_domain_id` | lattice |
| region | `region_asset_id` | **truth_contract** (+ feasibility, semantic, grid, membership); **not** pack/evidence_bundle |
| mask_unit | `mask_unit_id` | truth_contract + grid + mask_sha256 |
| coordinate | `coordinate_id` | truth_contract + cell_id; FKs to region + mask |
| null_record | `null_record_id` | truth_contract + search_domain; semantic_definition **NULL** |
| pack_membership | `(pack_id, content_id, content_kind)` | pack → content |

### 3.4 Authoritative relations (CR5 retained)

```text
region ↔ mask:
  AUTHORITATIVE = region_coordinates (region_asset_id, mask_unit_id FKs)
  DERIVED       = region_mask_link projection
  FORBIDDEN     = embedded partner JSON lists as authority

search_domain ↔ concrete semantics:
  AUTHORITATIVE = search_domain_members.csv
```

### 3.5 Null vs missing

| Encoding | Legal? |
|:--|:--|
| null_record + domain counts | required for G3 |
| n_non_null=0 and n_null_records=1 | required |
| missing atlas as null | **no** → BLOCKED_BY_ARTIFACT |

---

## 4. Stable identity proposal (CR1 + CR6)

### 4.1 Digest policy

```text
id_scheme: "region_asset_id_v2"   # bump: CR6 truth_contract vs evidence_bundle
algorithm: SHA-256
canonical JSON: sorted keys; separators=(',', ':'); thr_index ints only in content IDs
display: prefix || hex[:32]; full digest for fail-closed equality
```

### 4.2 Layer definitions

#### `truth_contract_id` — normalized semantic/data truth

**Includes** (order-insensitive / field-declared):

```json
{
  "id_scheme": "region_asset_id_v2",
  "kind": "truth_contract",
  "taxonomy_version": "stage2_q45_atlas_v4",
  "substrate_id": "stage1_baudit_d_online",
  "candidate_universe_id": "online_hook_eligible",
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
  "normalized_data_content_digest": "<order-insensitive digest §4.3>"
}
```

**Excludes:** raw file byte SHAs; CSV/Parquet row order; pack schema; generator script SHA; human study_id alone (study_id may be recorded on evidence_bundle, not as sole truth identity).

#### `evidence_bundle_id` — exact run/artifact seal

```json
{
  "id_scheme": "region_asset_id_v2",
  "kind": "evidence_bundle",
  "truth_contract_id": "...",
  "study_id": "m_b1_5_stage2_q45_20260710",
  "study_git_commit": "dc758e08…",
  "evaluator_source_sha256": "551284f8…",
  "runner_source_sha256": "376b11c0…",
  "source_event_table_sha256": "cfca3818…",
  "raw_artifact_sha256": {
    "atom_atlas_parquet": "281cb22b…",
    "pairwise_and_atlas_parquet": "ae52bd0c…",
    "pairwise_or_atlas_parquet": "bc1c2938…",
    "threshold_registry": "d3e3197f…",
    "summary": "a88d9dcf…",
    "manifest": "4213e82e…",
    "t0_component_geometry": "ad2dd33b…",
    "t0_boundary_margin": "424fe063…",
    "t0_summary": "4d12f1f2…"
  }
}
```

Row reorder or format rematerialization that changes bytes → **new evidence_bundle_id**.  
If normalized content digests match → **same truth_contract_id** and **same content IDs**.

#### `feasibility_contract_id` — CR7

```json
{
  "id_scheme": "region_asset_id_v2",
  "kind": "feasibility_contract",
  "truth_contract_id": "...",
  "parameter_or_policy_space": {
    "spaces": [
      {"grammar":"G1_atom","lattice":"primary_unique_boundaries","n_coords":870},
      {"grammar":"G2_and","lattice":"primary_quantile_lattice_q05","n_coords":17640},
      {"grammar":"G3_or","lattice":"primary_quantile_lattice_q05","n_coords":17640}
    ]
  },
  "candidate_universe_id": "online_hook_eligible",
  "safety_loss_definition": {
    "name": "resolved_gt_hurt_exact_zero",
    "predicate": "resolved GT_hurt == 0",
    "unresolved_firewall": "unresolved_contaminated_blocks_candidate"
  },
  "productivity_definition": {
    "name": "n_neg_captured_ge_1",
    "predicate": "n_neg_captured > 0",
    "kind": "count_surrogate"
  },
  "epsilon": {"kind": "exact_zero_count", "value": 0},
  "g_min": {"kind": "count_ge_1", "value": 1},
  "n_gt_exposed": 64,
  "n_fp_exposed": 23,
  "denominator_owner": "primary_resolved_baseline_selected_cohort",
  "selection_scope": "in_sample_searched_and_evaluated",
  "finite_sample_statement": "observed_GT0_is_not_population_risk_zero",
  "metric_adjacency_edge_policy": {
    "G1": "bilateral",
    "G2": "4neighbor_manhattan_erosion",
    "unsafe": "non_productive_safe_on_registered_grid",
    "off_lattice_neighbor": "fails_full_neighborhood"
  },
  "claim_level_max_supported": "L1"
}
```

#### `pack_id`

```json
{
  "kind": "pack",
  "truth_contract_id": "...",
  "feasibility_contract_id": "...",
  "evidence_bundle_id": "...",
  "producer_kind": "grammar_atlas",
  "producer_contract_version": "region_asset_v0",
  "schema_version": "region_asset_tables_v0",
  "grammar_scope": "G1_G2_G3"
}
```

#### `semantic_definition_id` (concrete)

Operator + sorted operands + lattice + roles=null. No thr indices.  
G2/G3 concrete: `AND`/`OR` of two P:: feature×direction leaves.

#### `search_domain_id` (CR8)

```json
{
  "kind": "search_domain",
  "truth_contract_id": "...",
  "grammar": "G3_or",
  "membership_digest": "sha256 of sorted search_domain_members canonical rows"
}
```

Members (each of 40):

```json
{
  "grid_domain_id": "...",
  "grid_id": "P::feat_a::dir_a__feat_b::dir_b",
  "semantic_definition_id": "<concrete OR tree for that operand pair>",
  "n_registered_coordinates": 441,
  "lattice_kind": "primary_quantile_lattice_q05"
}
```

`search_domain_id` changes if any grid/operand is added, removed, or semantically changed — not merely if `n_grids` summary stays 40 with different members.

#### `region_asset_id` / `mask_unit_id` / `coordinate_id`

Parents: **truth_contract_id** (+ local content).  
**Never** pack_id, evidence_bundle_id, raw file SHAs, component ordinals.

#### `null_record_id`

truth_contract + search_domain + null_reason_class; semantic_definition_id NULL on the null row.

### 4.3 Normalized data-content digest (order-insensitive)

Declared procedure (R1 must implement; not executed as asset emission here):

```text
For each atlas table T in {atom, pairwise_and, pairwise_or}:
  project declared identity+truth columns only
  sort rows by native PK (atom_id / combo_id)
  serialize each row as canonical JSON
  digest_T = sha256(concat of row digests in sorted PK order)

For each T0 geometry table used as truth (component coords, PS flags alignment):
  similarly sort by stable key

normalized_data_content_digest = sha256(sorted map of table_name → digest_T)
```

Equivalent regeneration with same logical rows different physical order → same normalized digest → same truth_contract_id.  
Byte-different files → different evidence_bundle_id.

### 4.4 Stability tests

| Scenario | truth_contract | evidence_bundle | region/mask/coord IDs |
|:--|:--|:--|:--|
| Row reorder same logical content | unchanged | **may change** | **unchanged** |
| CSV→Parquet same logical content | unchanged | **may change** | **unchanged** |
| Schema/pack version only | unchanged | unchanged (if raw same) | **unchanged** |
| Unrelated grammar in new pack | unchanged | unchanged | **unchanged** |
| Cohort/label/signal/lattice change | **changes** | changes | **changes** |
| Region membership change | **changes** (content digest / membership) | may change | **changes** affected IDs |
| Same mask_sha256 two grids | — | — | **different** mask_unit_id |
| Different 40-grid OR family, same counts | search_domain **changes** | — | null_record **changes** |

### 4.5 Synthetic identity examples

**A — reorder**

```text
atlas_v1 bytes H1, atlas_v2 row-permuted bytes H2 ≠ H1
normalized content equal
⇒ evidence_bundle_id changes; truth_contract_id stable; R_G1 stable
```

**B — pack/schema**

```text
pack schema v0 → v1 ⇒ pack_id changes; R_G1 stable
```

**C — CR8 domain**

```text
domain A: 40 grids including (abs_log_h↑, resid_mean↓)
domain B: 40 grids replacing that pair with another
summary n_grids=40 both
⇒ search_domain_id(A) ≠ search_domain_id(B)
```

---

## 5. Semantic preservation

### 5.1 G1/G2 concrete (DIRECT)

Operator trees, operand identity, directions, lattice coordinate systems, mask vs semantic inequality — as R0-A-R1.

### 5.2 G3

```text
Grammar-level null: search_domain + null_record; semantic_definition_id NULL
Each of 40 grids: concrete OR(semantic_definition_id) in search_domain_members
Zero PS cells does not erase concrete grid semantics
```

### 5.3 G7 — leave unresolved

necessary/support roles, GT-envelope, logical complement, envelope-relative coordinates — not inferred from masks.

---

## 6. G1–G3 A0→A1 gap + claim levels

C0 geometry retained; terminal B retained.

| Grammar | Non-null assets | Null | Maturity now | Max L-level now |
|:--|:--|:--|:--|:--|
| G1 | 1 isolated | no | A0 | L1 (in-sample region geometry; thin) |
| G2 | 25 thin components | no | A0 | L1 |
| G3 | 0 | 1 domain null | A0 | L0 domain-empty observation (no PS set) |

Nested LOSO 0 portable ≠ L2/L3.  
Online Stage1 null effect ≠ L5.

Evaluator rerun: **not required** for A1 packaging of existing sealed evidence.

---

## 7. G3 null-asset contract (CR4 + CR8)

```text
null_record_id
truth_contract_id
evidence_bundle_id          # seal only; not identity parent of null_record_id
feasibility_contract_id
search_domain_id            # membership-digest over 40 concrete grids
semantic_definition_id:     NULL
n_non_null_region_assets:   0
n_null_records:             1
observed_safe_count:        0
productive_safe_count:      0
declared_search_domain:     17640 coords / 40 grids / OR / q05 lattice
null_reason:                no_observed_or_productive_safe_on_registered_or_lattice
bounded_status:             NULL_RESULT
action_state:               observation_only
production_forbidden:       true
claim_level_max:            L0 (empty productive-safe set on declared domain)
finite_sample_statement:    domain-empty observation ≠ proof OR useless off-lattice
```

**Members (required machine table):** 40 rows each with concrete `semantic_definition_id` for that operand-pair OR.

---

## 7a. Feasibility contract for Q4.5 (CR7) — operational freeze

```text
safety:
  resolved GT_hurt == 0
  under unresolved-contamination firewall

productivity:
  n_neg_captured > 0

selection:
  searched and evaluated in-sample on registered lattices

population safety:
  not established

denominators:
  n_gt_exposed = 64 (primary positive-protection)
  n_fp_exposed = 23 (primary negative)

epsilon:
  exact-zero count (not a positive rate allowance)

g_min:
  count surrogate ≥ 1 negative captured

geometry metric:
  T0 dual-margin policy
```

### Independent ladders

```text
Asset maturity A0–A4          | Statistical/transfer claim L0–L6
----------------------------- | --------------------------------
A0 descriptive atlas          | L0 observed safe point
A1 region asset (package)     | L1 in-sample safe region
A2 validated applicability    | L2 held-out point / L3 held-out region
A3 intervention asset         | L4 substrate / L5 online
A4 production-approved        | L6 production-safe candidate
```

```text
A1 engineering packaging ⇏ L2+
L1 claim ⇏ A1 maturity accepted
neither ladder implies the other
```

Current sealed G1–G3 pack, if later materialised at A1 engineering readiness, remains **L1 max** until transfer evidence exists.

---

## 8. Proposed R1 machine schemas (CR9)

Contract only — **do not emit files**.

### 8.0 Authority tables (new / explicit)

#### `truth_contract.json`

| | |
|:--|:--|
| Grain | one normalized truth contract |
| PK | `truth_contract_id` |
| Required | taxonomy, substrate, candidate_universe, signal_family, sequence_set, label_contract, unresolved_policy, lattice_contract, normalized_data_content_digest, id_scheme |
| Forbidden | raw file SHA map as identity input |
| Invariants | order-insensitive content digest declared |

#### `evidence_bundle.json`

| | |
|:--|:--|
| Grain | one exact seal |
| PK | `evidence_bundle_id` |
| FK | truth_contract_id |
| Required | study_id, recorded evaluator/runner/event SHAs, raw_artifact_sha256 map, terminal_letter, created_utc optional |
| Invariants | every truth-used raw artifact present |

#### `feasibility_contract.json`

| | |
|:--|:--|
| Grain | one feasibility definition |
| PK | `feasibility_contract_id` |
| FK | truth_contract_id |
| Required | parameter spaces, candidate_universe_id, safety_loss_definition, productivity_definition, epsilon, g_min, n_gt_exposed, n_fp_exposed, selection_scope, finite_sample_statement, metric_adjacency_edge_policy, claim_level_max_supported |
| Invariants | population bound absent or explicitly null; L-level ≤ evidence |

#### `semantic_definitions.csv` | `jsonl`

| | |
|:--|:--|
| Grain | one concrete operator tree |
| PK | `semantic_definition_id` |
| Required | grammar, operator, operands_json (sorted), lattice_kind, roles_json (null), parameter_system |
| Invariants | G3 grammar-level null is **not** a row here; per-grid OR trees **are** |

#### `search_domains.csv` | `jsonl`

| | |
|:--|:--|
| Grain | one search domain |
| PK | `search_domain_id` |
| Required | truth_contract_id, grammar, combinator, lattice_kind, n_members, n_registered_coordinates_sum, membership_digest |
| Invariants | membership_digest matches members table |

#### `search_domain_members.csv`

| | |
|:--|:--|
| Grain | domain × concrete grid |
| PK | `(search_domain_id, grid_domain_id)` |
| FK | search_domain_id, grid_domain_id, **semantic_definition_id** (concrete, non-null) |
| Required | grid_id, n_registered_coordinates, feature/direction axes |
| Invariants | G3 has **40** members; unique semantic per distinct operand-pair; membership change ⇒ domain id change |

#### `pack_membership.csv`

| | |
|:--|:--|
| Grain | pack × content object |
| PK | `(pack_id, content_kind, content_id)` |
| FK | pack_id → content ids (region, null_record, mask, etc.) |
| Required | content_kind enum |
| Invariants | every content row in pack listed; content ids exist in authority tables |

### 8.1 Pack emission tables (retained + fixed)

| File | Grain | PK | Notes |
|:--|:--|:--|:--|
| `region_asset_manifest.json` | pack | pack_id | FKs: truth_contract, evidence_bundle, feasibility; counts n_non_null / n_null_records |
| `grammar_region_summary.csv` | pack×grammar | (pack_id, grammar) | n_non_null_region_assets, n_null_records |
| `region_assets.csv` | non-null region | region_asset_id | FK truth_contract, feasibility, semantic_definition, grid |
| `null_records.csv` | null | null_record_id | semantic_definition_id NULL; FK search_domain |
| `region_components.csv` | region geometry | region_asset_id | coords_json + digest |
| `region_masks.csv` | mask unit | mask_unit_id | no partner JSON authority |
| `region_coordinates.csv` | PS coord | coordinate_id | **authoritative** region_asset_id + mask_unit_id FKs |
| `region_mask_link.csv` | optional derived | (region, mask) | projection only |
| `region_capacity.csv` | see §8.2 | — | dual distributions |
| `region_sequence_support.csv` | incidence + region summary | — | union∩ CR3 |
| `region_margin.csv` | coord | coordinate_id | dual margins |
| `region_claim_contract.json` | pack | pack_id | A-ladder + L-ladder + forbidden promotions |

### 8.2 Capacity dual distributions (CR9)

`region_capacity.csv` (or twin files) **must** include for each multi-member region:

```text
metric_family = coordinate_member_capacity_distribution
  member_grain = coordinate
  n_members, min, max, median, q25, q75, robust_floor

metric_family = mask_member_capacity_distribution
  member_grain = per_grid_mask_unit
  n_members, min, max, median, q25, q75, robust_floor
```

```text
FORBIDDEN:
  sum as simultaneous multi-mask mass
  presenting either distribution as event-union capacity
  omitting one family when the other is emitted for multi-member regions
```

Event-union: omit or status=`BLOCKED_BY_ARTIFACT`.

### 8.3 Pack invariants

```text
sum_PS = 154 for this seal
G3: n_non_null=0, n_null_records≥1, search_domain_members=40
content IDs independent of pack_id and evidence_bundle_id
region_mask_link == projection(coordinates) if present
both capacity distributions present for multi-member regions
claim_level_max ≤ L1 unless transfer tables exist
production_forbidden=true; action_state=observation_only
every FK has an authority file in §8.0–8.1
no evaluator rerun
```

---

## 9. Maturity (A) and claim (L) firewalls

### 9.1 A0–A4 (asset maturity)

| Level | Meaning | G1–G3 now |
|:--|:--|:--|
| A0 | descriptive atlas / null | **current** |
| A1 | stable package + firewall | engineering-feasible after R0-B; **not accepted** |
| A2 | validated applicability | no |
| A3 | intervention | no |
| A4 | production-approved | no |

### 9.2 L0–L6 (statistical/transfer claim)

| Level | Meaning | G1–G3 now |
|:--|:--|:--|
| L0 | observed safe point / domain observation | G3 empty domain; isolated points |
| L1 | in-sample safe **region** geometry | **max** for G1/G2 atlas geometry |
| L2 | held-out retained point | not established as region program |
| L3 | held-out retained region | no (nested LOSO portable clauses = 0) |
| L4–L6 | substrate / online / production | no |

### 9.3 Action states

`observation_only` required; all intervention/shadow/production states forbidden at current evidence.

### 9.4 Independent gates

```text
artifact generated
≠ engineering ready
≠ asset maturity accepted (A*)
≠ statistical claim level (L*)
≠ research conclusion accepted
≠ intervention qualified
≠ production approved
```

---

## 10. Cross-family boundary

Core shareable: truth_contract, evidence_bundle, feasibility_contract, content IDs, capacity distributions, sequence incidence/union∩, null_record, A/L ladders, action firewall.  
Opaque adapters: grammar/occ-exit/association/relink specifics.  
Intervention-time fields: A3/L5+ only.

---

## 11. Decisions before R0-B

| id | question | recommended_default | consequence |
|:--|:--|:--|:--|
| D1 | pack grammar_scope | one G1_G2_G3 pack | pack_id only |
| D2 | id_scheme | `region_asset_id_v2` | all PKs |
| D4 | AND operand order | lex (feature,direction) | semantic_id |
| D5 | region grain | connected PS component | region_assets |
| D6 | alias format | grid+shape+n_coords | non-PK |
| D7 | G3 null | one domain null + 40 members | null_records + members |
| D8 | mask unit | (truth_contract, grid, mask_sha256) | mask PK |
| D9 | coordinate_id | truth_contract + native cell_id | coord PK |
| D10 | capacity | **both** coord-member and mask-member distributions; no sum | region_capacity |
| D10u | event-union | BLOCKED_BY_ARTIFACT | omit |
| D11 | sequence | incidence + union + intersection | sequence tables |
| D12 | action default | observation_only + production_forbidden | claim_contract |
| D13 | region↔mask | coordinate FKs authority | joins |
| D14 | feasibility freeze | §7a Q4.5 operational text | feasibility_contract_id |
| D15 | claim_level_max | L1 until transfer evidence | claim_contract |
| D16 | normalized content digest field list | §4.3 declared columns | truth_contract stability |

**Closed by R0-A-R2 for review:** CR6 layers; CR7 feasibility+ladders; CR8 membership; CR9 authority tables + dual distributions.  
**CR1–CR5:** remain closed/pass.

---

## 12. Synthetic examples (CR6–CR9)

### 12.1 CR6 reorder

```text
H_raw changes, H_normalized stable
⇒ evidence_bundle_id' ≠ evidence_bundle_id
⇒ truth_contract_id' = truth_contract_id
⇒ region_asset_id stable
```

### 12.2 CR7 ladders

```text
Asset A1 packaging accepted (hypothetical future)
claim_level_max still L1
⇏ L3 held-out region
⇏ population L_GT ≤ ε
```

### 12.3 CR8 domain membership

```text
G3 members = 40
each: grid_id + concrete OR semantic_definition_id + 441 denominator
search_domain_id = digest(sorted members)
null_record.semantic_definition_id = NULL
```

### 12.4 CR9 dual capacity

Multi-mask component with mask_n_neg ∈ {1,2,3,4}:

```text
mask_member: n=4, min=1, max=4, ...
coordinate_member: n=12, distribution over cell n_neg
sum=10 ILLEGAL as region mass
event_union BLOCKED
```

---

## 13. R1 readiness (not authorization)

```text
IF R0-B accepts remaining naming freezes
AND atlases sealed
AND CR1–CR9 remain in force
THEN R1 = deterministic packaging of A0 evidence into A1-shaped files
     with claim_level_max=L1 and maturity field still pending chat acceptance
ELSE blocked on missing seals / open decisions

R0-A-R2 is not self-accepted.
R0-B / R1 unauthorized.
```

---

## 14. Acceptance checks (R0-A-R2 self-audit)

```text
[x] CR1–CR5 retained (pack⟂content; non-additive; union∩; G3 null grain; coord FKs)
[x] truth_contract_id excludes raw byte hashes; evidence_bundle_id carries them
[x] content IDs parent truth_contract (+ local content), not evidence_bundle/pack
[x] row reorder / equivalent regeneration: evidence may change; content IDs stable if normalized truth stable
[x] feasibility_contract_id with denominators, ε, g_min, selection_scope, finite-sample statement
[x] A0–A4 ⟂ L0–L6; current max L1; A0 maturity
[x] G3 search_domain membership includes all concrete grid semantics (40)
[x] grammar-level null semantic_definition_id NULL
[x] authority tables proposed for all FKs
[x] dual capacity distributions required; event-union blocked
[x] no evaluator rerun; no asset pack; no G4–G7; terminal B; ledger unchanged
[x] math framework cited as canonical method/evidence-semantics contract
```

---

## 15. Explicit non-claims

```text
no evaluator rerun
no asset pack generated
no research verdict self-accepted
no R0-A / R0-A-R1 / R0-A-R2 acceptance declared
no R0-B / R1 authorized
no A0→A1 maturity promotion
no L1→L2+ claim promotion
no PR opened by this note alone
no evidence_ledger promotion
```

---

## 16. Next gate

```text
R0-A-R2 complete
→ chat-side re-review
→ only if accepted: close R0-A line and authorize R0-B final contract
→ only after R0-B: authorize R1 deterministic packaging
```
