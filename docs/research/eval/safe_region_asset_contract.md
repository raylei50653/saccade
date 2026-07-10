# Safe-Region Asset Contract (R0-B Draft)

<!-- doc-status: active -->
<!-- doc-promotion: none; method/schema contract only; not evidence_ledger -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: cross -->

**Role:** Normative cross-cutting **RegionAsset** contract draft (R0-B).  
**Status:** **DRAFT** — awaiting chat-side acceptance. Not self-accepted.  
**Does not authorize:** R1 asset generation · A0→A1 maturity · transfer · intervention · production · ledger promotion.

**Parents**

| Document | Role |
|:--|:--|
| [statistical_robust_feasible_set_estimation_under_asymmetric_loss.md](statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) | mathematical / claim-ladder semantics |
| [safe_region_r0_asset_contract_preflight_20260710.md](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md) | accepted R0-A derivability packet (CR1–CR9) |
| [safe_region_assetization_20260710.md](../threads/safe_region_assetization_20260710.md) | program ownership thread |

**Contract version:** `region_asset_contract_v0`  
**ID scheme:** `region_asset_id_v2`  
**Digest:** SHA-256; display may truncate to 32 hex; equality fail-closed on full digest.

---

## 1. Scope and non-claims

### In scope

- Normative identity layers, grains, machine schemas, validation invariants.
- Mapping of accepted Q4.5 G1–G3 / T0 evidence into a future deterministic pack.
- Separation of mathematical feasibility definition from observed claim level.

### Out of scope / non-claims

```text
≠ accepted A1 maturity
≠ population risk bound
≠ held-out / LOO / LOSO region transfer (L2+)
≠ online retention (L5) or production (L6 / A4)
≠ G4–G7 implementation or G7 role invention
≠ evaluator rerun or modification
≠ production preset / hook / shadow policy change
≠ evidence_ledger promotion
```

Current sealed study remains **A0** descriptive atlas with terminal **B** `isolated_safe_points_only`.

---

## 2. Mathematical object

A RegionAsset packages an estimated **productive-safe set** under asymmetric loss on a declared parameter space and candidate universe.

### 2.1 Symbols (operational)

| Symbol | Contract field | Q4.5 G1–G3 freeze |
|:--|:--|:--|
| $\Theta$ | `parameter_or_policy_space` | registered thr-index lattices (G1 unique-boundary 870; G2/G3 q05 17640 each) |
| $\Omega$ | `candidate_universe_id` | `online_hook_eligible` |
| $L_{\mathrm{GT}}$ | `safety_loss_definition` | resolved `GT_hurt == 0` under unresolved-contamination firewall |
| $\varepsilon$ | `epsilon` | `exact_zero_count` (count hurt $=0$, not a positive rate allowance) |
| $G_{\mathrm{FP}}$ / productivity | `productivity_definition` | count surrogate `n_neg_captured > 0` |
| $g_{\min}$ | `g_min` | `count_ge_1` |
| $N_{\mathrm{GT,exposed}}$ | `n_gt_exposed` | **64** primary positive-protection |
| $N_{\mathrm{FP,exposed}}$ | `n_fp_exposed` | **23** primary negative |
| metric / adjacency / edge | `metric_adjacency_edge_policy` | T0 dual-margin: G1 bilateral; G2 4-neighbor; unsafe = non-PS on registered grid; off-lattice neighbor fails full-neighborhood radius |

### 2.2 Productive-safe predicate (operational)

```text
productive_safe ⇔
  safety_loss satisfied (resolved GT_hurt == 0 under unresolved firewall)
  AND productivity satisfied (n_neg_captured > 0)
```

Safe-only cells (GT0 with zero negatives) are not productive-safe.  
Observed sample GT0 **does not** establish population $L_{\mathrm{GT}}\le\varepsilon$.

### 2.3 Selection scope

```text
selection_scope = in_sample_searched_and_evaluated
```

Search and evaluation use the same registered cohort/lattices. This is not a frozen held-out selection.

---

## 3. Identity layers

Layers are **independent**. Child content IDs must not depend on pack materialization or raw file byte order.

| Layer | ID | Owns | Must not own |
|:--|:--|:--|:--|
| Normalized truth | `truth_contract_id` | substrate, universe, signals, sequences, label/unresolved, lattices, **order-insensitive data content digest** | raw file SHAs; pack schema; claim L-level |
| Exact evidence seal | `evidence_bundle_id` | study_id, recorded evaluator/runner SHAs, **raw artifact SHA map** | semantic content identity |
| Feasibility definition | `feasibility_contract_id` | $\Theta$, $\Omega$, $L_{\mathrm{GT}}$, productivity, $\varepsilon$, $g_{\min}$, exposure **definitions and declared denominators**, selection_scope, finite-sample statement, geometry metric/edge policy | **supported claim_level** (outcome) |
| Pack / materialization | `pack_id` | producer kind/version, schema version, grammar_scope, FKs to truth/feasibility/evidence | local region/mask/coord content |
| Concrete semantics | `semantic_definition_id` | operator tree + operand leaves + lattice kind + roles | thr indices; masks |
| Grid domain | `grid_domain_id` | one registered feature×direction (pair) lattice domain | pack |
| Search domain | `search_domain_id` | membership digest over concrete grid members | summary counts alone |
| Region content | `region_asset_id` | connected PS component membership within one grid | pack_id; evidence_bundle_id; `::compN` |
| Mask unit | `mask_unit_id` | `(truth_contract_id, grid_id, mask_sha256)` | global mask alone |
| Coordinate | `coordinate_id` | `(truth_contract_id, cell_id)` | thr_value floats as PK |
| Null record | `null_record_id` | search-domain empty result | concrete semantic_definition on grammar-level null |
| Evidence / claim outcome | `evidence_claim_id` (row) | observed exposures, counts, **claim_level**, finite-sample notes tied to a sealed bundle | redefining feasibility math |
| Pack membership | `(pack_id, content_kind, content_id)` | which content objects appear in an emission | redefining content IDs |

**Retired names:** `asset_set_id` (R0-A); using `truth_context_id` that embedded raw SHAs as content parent.

---

## 4. Claim ownership (A-ladder ⟂ L-ladder)

### 4.1 Feasibility contract excludes claim outcomes

`feasibility_contract_id` **must not** include:

```text
claim_level_max_supported
claim_level
n_productive_safe observed counts as identity inputs
pack-wide maximum L-level
```

Those belong to **evidence/claim records** (`evidence_claims` and/or `region_claim_contract.json` + per-object claim fields).

Feasibility **may** include declared denominators (`n_gt_exposed`, `n_fp_exposed`) when they define the rate/count contract for the study population definition — not the observed hurt/removed outcomes of a particular region.

### 4.2 Observed evidence owner

| Field | Owner |
|:--|:--|
| Observed `n_gt_hurt`, `n_neg_captured`, per-object geometry | content / capacity / margin tables + evidence_bundle seal |
| `selection_scope` realization note | evidence_claims / claim contract |
| `finite_sample_statement` restatement | claim contract (definition text may be shared) |
| **`claim_level` per object** | evidence_claims / region_assets.claim_level / null_records.claim_level |
| **`pack_claim_ceiling`** | region_claim_contract.json / pack manifest |

### 4.3 Q4.5 per-object claim levels (locked)

| Object class | claim_level | Rationale |
|:--|:--|:--|
| G1 non-null region (single isolated PS coordinate) | **L0** | observed safe/productive point; not multi-coordinate region thickness claim |
| G2 non-null region (multi-coordinate in-sample component geometry) | **L1** | in-sample safe-region geometry under registered adjacency |
| G3 grammar/search-domain null | **L0** | declared-domain empty productive-safe observation |
| Pack aggregate ceiling | **L1** | max over objects; **not** inherited by every object |

```text
pack_claim_ceiling = L1
⇏ every region_asset.claim_level = L1
```

### 4.4 Orthogonal ladders

```text
A0–A4  asset maturity / packaging / action readiness
L0–L6  statistical / transfer claim strength
```

Neither ladder implies the other. A future A1 pack of this study remains **L0/L1 only**.

---

## 5. Object grains and relations

```text
truth_contract
evidence_bundle ──FK──► truth_contract
feasibility_contract ──FK──► truth_contract
pack ──FK──► truth_contract, feasibility_contract, evidence_bundle

semantic_definition          (concrete operator tree)
grid_domain                  (registered lattice domain)
search_domain
search_domain_member ──FK──► search_domain, grid_domain, semantic_definition
region_asset ──FK──► truth_contract, semantic_definition, grid_domain
null_record ──FK──► truth_contract, search_domain; semantic_definition = NULL
mask_unit ──FK──► truth_contract, grid_domain
coordinate ──FK──► truth_contract, region_asset, mask_unit, grid_domain
pack_membership ──FK──► pack, content
evidence_claim ──FK──► evidence_bundle, optional content_id
```

### 5.1 Primary grains

| Grain | Definition |
|:--|:--|
| Region (non-null) | Connected **productive-safe** component within **one** registered grid under declared adjacency |
| Per-grid mask unit | `mask_sha256` within one `grid_id` (primary); global mask string is diagnostic only |
| Coordinate | Registered thr-index cell (`cell_id`) |
| Null record | First-class empty productive-safe result on a **search domain**, not missing files |
| Search domain | Ordered membership of concrete grid domains + concrete semantics |

### 5.2 Cardinality rules

- One region may contain many mask units; one mask unit may appear in many regions **in schema** (M:N). Observed Q4.5: multi-mask regions exist; multi-region masks not observed — still do not assume many-to-one.
- **Authoritative** region↔mask relation: projection from `region_coordinates` FKs. Optional `region_mask_link` is derived only.
- G3 sealed study: `n_non_null_region_assets = 0`, `n_null_records = 1`, `search_domain_members = 40`.

### 5.3 Forbidden identity inputs

```text
component ordinals (::comp0, component_03, …)
global mask_sha256 alone as PK
pack_id / producer_contract_version / schema_version inside content digests
raw file SHA maps inside content digests
human thr_value strings as PKs (aliases only)
```

---

## 6. Stable-ID canonicalization

### 6.1 Canonical JSON

```text
UTF-8 JSON object
keys sorted lexicographically
separators = (',', ':')
arrays in declared sort order
no insignificant whitespace
thr_index as JSON integers
thr_value never used in content IDs
```

### 6.2 Content ID inputs (normative)

**`region_asset_id`**

```json
{
  "id_scheme": "region_asset_id_v2",
  "kind": "region_asset",
  "truth_contract_id": "<hex>",
  "semantic_definition_id": "<hex>",
  "grid_domain_id": "<hex>",
  "adjacency": "G1_bilateral" | "G2_4neighbor",
  "membership": "productive_safe",
  "coordinate_digest": "<sha256 of sorted thr-index keys>"
}
```

Coordinate keys: G1 sorted `[thr_index]`; G2/G3 sorted `[[i,j],…]` with axis order fixed by `grid_domain` operand order after lex sort of leaves.

**`mask_unit_id`:** `truth_contract_id` + `grid_id` + `mask_sha256`.  
**`coordinate_id`:** `truth_contract_id` + native `cell_id` (`S::…` / `AND::…` / `OR::…`).  
**`null_record_id`:** `truth_contract_id` + `search_domain_id` + `null_reason_class`.  
**`semantic_definition_id`:** grammar, operator, lex-sorted operands `(feature,direction)`, lattice_kind, `roles=null`.  
**`search_domain_id`:** `truth_contract_id` + `membership_digest` over canonical member rows.  
**`grid_domain_id`:** lattice_kind + ordered axis descriptors (G1: feature,direction; pairwise: two leaves lex-sorted for domain identity of the **grid**, while combinator is property of the semantic definition / search domain).

### 6.3 AND/OR operand symmetry

For symmetric hard AND/OR leaves without roles: sort operands lexicographically by `(feature, direction)` before digesting `semantic_definition_id`.

### 6.4 Stability requirements

| Event | evidence_bundle | truth_contract | content IDs |
|:--|:--|:--|:--|
| Row reorder, same logical rows | may change | stable | stable |
| CSV↔Parquet rematerialization, same logical rows | may change | stable | stable |
| Pack/schema version only | stable (if raw same) | stable | stable |
| Unrelated grammar in another pack | — | stable | stable |
| Cohort / label / signal / lattice / membership change | changes | changes | affected IDs change |

### 6.5 Aliases

Human thr expressions and T0 `::compN` strings are **aliases only**, never join keys.

---

## 7. Normalized truth digest

`truth_contract.normalized_data_content_digest` is the sole logical-data input that makes truth identity independent of byte order.

### 7.1 Algorithm

```text
for each declared truth table T:
  project only truth-bearing columns (§7.2)
  drop excluded columns (§7.3)
  apply missing/float/duplicate rules (§7.4–7.5)
  sort rows by stable row key ascending
  for each row: row_digest = SHA256(canonical_json(row))
  table_digest[T] = SHA256(concat(row_digests in sorted order))

normalized_data_content_digest =
  SHA256(canonical_json({T: table_digest[T] for T sorted by name}))
```

### 7.2 Truth-bearing tables and stable row keys

| Table (logical) | Stable row key | Truth-bearing columns (ordered list is for documentation; JSON keys still sorted) |
|:--|:--|:--|
| `atom_atlas` | `atom_id` | `atom_id`, `feature`, `direction`, `thr_index`, `lattice_kind`, `observed_safe_point`, `productive_safe_point`, `gt_hurt`, `n_neg_captured`, `n_gt_captured`, `n_unresolved_selected`, `safety_status`, `mask_sha256`, `per_sequence_neg_json`, `per_sequence_gt_json` |
| `pairwise_and_atlas` | `combo_id` | `combo_id`, `combinator`, `atom_a_id`, `atom_b_id`, `feature_a`, `direction_a`, `thr_index_a`, `feature_b`, `direction_b`, `thr_index_b`, `lattice_kind`, `observed_safe_point`, `productive_safe_point`, `gt_hurt`, `n_neg_captured`, `n_gt_captured`, `n_unresolved_selected`, `safety_status`, `mask_sha256`, `semantic_duplicate_mask`, `empty_region`, `per_sequence_neg_json`, `per_sequence_gt_json` |
| `pairwise_or_atlas` | `combo_id` | same column set as AND |
| `threshold_registry_meta` | single logical row | `taxonomy_version`, `signals_primary` (sorted), `directions` (sorted), `combinators` (sorted), `single_lattice_kind`, `pairwise_lattice_kind`, `n_single_atoms`, `n_pairwise_atoms`, `assignment_group_key_status` |
| `cohort_contract` | single logical row | canonical `cohort_definition` object + sorted `sequence_set` + `n_primary_negative` + `n_primary_positive_protect` |
| `t0_component_membership` | `grid_id` + sorted `coords_json` normalized | `grammar`, `grid_id`, `adjacency`, sorted coordinate key list (not T0 `component_id` ordinal) |

Implementations may read parquet or csv; **logical projection** above is authoritative.

### 7.3 Excluded from truth digest

```text
raw file paths, timestamps, write order
thr_value / thr_value_a / thr_value_b   # alias material; thr_index is coordinate
display aliases, nested_loso clause float strings used only as labels
T0 component_id ordinals (::compN)
row numbers, parquet row groups
evaluator live source path
pack schema fields
loo_* portability marketing flags (may appear on evidence tables, not truth digest)
```

### 7.4 Missing values and types

| Kind | Canonical form in JSON |
|:--|:--|
| Integer flags 0/1 | JSON numbers `0` / `1` (not booleans, not `"0"`) |
| Integer counts / thr_index | JSON numbers (no leading zeros) |
| Missing optional string | key **omitted** (not `null`, not `""`) unless column is required |
| Required empty string disallowed | fail closed |
| JSON map fields (`per_sequence_*_json`) | parse → object with **sorted keys** → re-emit canonical JSON string value **or** nested object with sorted keys (implementation must pick **nested object** form) |

**Locked choice:** sequence maps are nested objects with sorted sequence-name keys and integer counts as JSON numbers.

### 7.5 Float policy

- **Truth digest columns avoid free floats** (thr_value excluded).
- If a future truth column requires floats: use shortest round-trip decimal via IEEE-754 binary64 → unique shortest decimal that recovers the same float; reject NaN/±Inf in truth identity columns.
- ±0: normalize to `0`.

### 7.6 Duplicate rows

- Duplicate stable row keys with identical projected payloads: collapse to one.
- Duplicate keys with conflicting payloads: **fail closed** (invalid truth input).

---

## 8. Geometry and capacity

### 8.1 Geometry

- Dual area: coordinate registered counts vs per-grid unique-mask counts; **never** compare raw G1 vs G2 counts without lattice note.
- Dual margin: `nearest_unsafe_distance` and `full_neighborhood_safe_radius` both required; edge-censored distance ≠ thickness.
- Components: connected PS under §2 metric; membership from coordinates, not `region_stability` mask-quotient rows alone.
- Global mask collapse across grids is diagnostic only.

### 8.2 Capacity (non-additive)

| Metric family | Member grain | Required stats |
|:--|:--|:--|
| `coordinate_member_capacity_distribution` | PS coordinates in region | n_members, min, max, median, q25, q75, robust_floor |
| `mask_member_capacity_distribution` | distinct per-grid mask units in region | same stats |

```text
FORBIDDEN as region mass:
  sum of member mask_n_neg
  sum of coordinate n_neg_captured sold as unique-event mass
  plateau_width × mask capacity
  event-union capacity without sealed event membership  → BLOCKED_BY_ARTIFACT
```

Both distribution families are required for multi-member regions. Neither is event-union capacity.

---

## 9. Sequence / applicability descriptions

| Field | Meaning | Promotion ban |
|:--|:--|:--|
| Incidence `(unit, sequence)` | coordinate or mask productive support | not a policy feature |
| `sequence_support_union` | some member choice had productive mass on sequence | **≠** region applicability |
| `sequence_support_intersection` | every member supports sequence | still descriptive |
| min/max member `n_sequences` | range across alternative parameters | descriptive |
| dominance / islands | max share, single-seq flags | descriptive |

A2 applicability / transfer requires a separate representative or transport contract (not this A0 pack).

---

## 10. Null-result contract

### 10.1 Required representation

A null result is a first-class `null_records` row:

```text
bounded_status = NULL_RESULT
semantic_definition_id = NULL          # grammar/search-domain null
search_domain_id = <membership digest>
n_non_null_region_assets (grammar summary) = 0
n_null_records ≥ 1
observed_safe_count / productive_safe_count on domain
null_reason
claim_level = L0 for Q4.5 G3
```

### 10.2 G3 sealed instance

```text
combinator OR · lattice primary_quantile_lattice_q05
40 members · 441 coords each · 17640 total
each member: grid_domain_id + concrete OR semantic_definition_id
productive_safe_count = 0 · observed_safe_count = 0
```

### 10.3 Illegal encodings

```text
missing atlas / failed run → BLOCKED_BY_ARTIFACT, not NULL_RESULT
empty components file without null_records row
summary-only domain id without member table
fake single semantic_definition for all 40 grids on the null row
```

---

## 11. Machine schemas and referential integrity

### 11.1 Authority files (normative)

Every FK target below is an authority. No FK may point to an undefined object.

#### `truth_contract.json`

| | |
|:--|:--|
| PK | `truth_contract_id` |
| Required | id_scheme, taxonomy_version, substrate_id, candidate_universe_id, signal_family[], sequence_set[], label_contract, unresolved_policy, lattice_contract, normalized_data_content_digest |
| Forbidden in identity inputs | raw_artifact_sha256 map |

#### `evidence_bundle.json`

| | |
|:--|:--|
| PK | `evidence_bundle_id` |
| FK | truth_contract_id |
| Required | study_id, study_git_commit, evaluator_source_sha256, runner_source_sha256, source_event_table_sha256, raw_artifact_sha256{}, terminal_letter |

#### `feasibility_contract.json`

| | |
|:--|:--|
| PK | `feasibility_contract_id` |
| FK | truth_contract_id |
| Required | parameter_or_policy_space, candidate_universe_id, safety_loss_definition, productivity_definition, epsilon, g_min, n_gt_exposed, n_fp_exposed, denominator_owner, selection_scope, finite_sample_statement, metric_adjacency_edge_policy |
| **Forbidden** | claim_level, claim_level_max_supported, observed region counts |

#### `evidence_claims.csv` | `jsonl`

| | |
|:--|:--|
| PK | `evidence_claim_id` |
| FK | evidence_bundle_id; optional content_id + content_kind |
| Required | claim_level (L0–L6), claim_scope (`object`\|`pack`\|`grammar`), selection_scope_note, finite_sample_statement, observed summary fields as applicable |
| Notes | Pack ceiling row: claim_scope=pack, claim_level=L1 for this study; object rows carry G1=L0, G2=L1, G3=L0 |

#### `semantic_definitions.csv` | `jsonl`

| | |
|:--|:--|
| PK | `semantic_definition_id` |
| Required | grammar, operator, operands_json, lattice_kind, roles_json, parameter_system |
| Notes | Concrete trees only; includes all 40 G3 OR pair definitions |

#### `grid_domains.csv` | `jsonl`

| | |
|:--|:--|
| PK | `grid_domain_id` |
| Required | lattice_kind, grid_id, grammar_family (`G1`\|`pairwise`), axis descriptors, n_registered_coordinates |
| Invariants | G1: 10 feature×direction grids × 87; pairwise: 40 axis-pairs × 441 |

#### `search_domains.csv` | `jsonl`

| | |
|:--|:--|
| PK | `search_domain_id` |
| FK | truth_contract_id |
| Required | grammar, combinator, lattice_kind, n_members, n_registered_coordinates_sum, membership_digest |

#### `search_domain_members.csv`

| | |
|:--|:--|
| PK | `(search_domain_id, grid_domain_id)` |
| FK | search_domain_id, **grid_domain_id**, **semantic_definition_id** (concrete, non-null) |
| Required | grid_id, n_registered_coordinates |
| Invariants | G3 sealed: exactly 40 rows; membership_digest matches |

#### `region_assets.csv`

| | |
|:--|:--|
| PK | `region_asset_id` |
| FK | truth_contract_id, semantic_definition_id, grid_domain_id |
| Required | grammar, bounded_status=HAS_REGION, n_coords, n_mask_units, shape fields, **claim_level**, action_state, production_forbidden |
| Notes | Non-null only; claim_level from evidence ladder (G1→L0, G2→L1) |

#### `null_records.csv`

| | |
|:--|:--|
| PK | `null_record_id` |
| FK | truth_contract_id, search_domain_id |
| Required | grammar, semantic_definition_id **NULL**, null_reason, domain counts, **claim_level=L0**, action_state, production_forbidden |

#### `region_masks.csv`

| | |
|:--|:--|
| PK | `mask_unit_id` |
| FK | truth_contract_id, grid_domain_id |
| Required | grid_id, mask_sha256, n_coords, mask_n_neg, sequence fields |
| Forbidden as authority | region_asset_ids_json |

#### `region_coordinates.csv`

| | |
|:--|:--|
| PK | `coordinate_id` |
| FK | truth_contract_id, **region_asset_id**, **mask_unit_id**, grid_domain_id |
| Required | cell_id, thr indices, PS flags, capacity, dual margins |
| **Authority** | sole authoritative region↔mask derivation source |

#### `pack_membership.csv`

| | |
|:--|:--|
| PK | `(pack_id, content_kind, content_id)` |
| Required | content_kind ∈ {region_asset, null_record, mask_unit, coordinate, semantic_definition, …} |

#### `region_claim_contract.json`

| | |
|:--|:--|
| PK | pack_id (emission) |
| Required | pack_claim_ceiling, maturity_declared (A0), action_states allowed/forbidden, production_forbidden, forbidden_promotions[], terminal_b, g7_status, identity_layer_policy, capacity_policy, sequence_policy, claim_ownership_policy (feasibility ⟂ claim_level) |

### 11.2 Auxiliary / derived (optional)

| File | Authority source |
|:--|:--|
| `region_components.csv` | region_assets + coordinates |
| `region_mask_link.csv` | **derived from** region_coordinates |
| `region_capacity.csv` | coordinates + masks; dual distributions |
| `region_sequence_support.csv` | incidence expansion + region union/intersection |
| `region_margin.csv` | coordinates / T0 boundary_margin |
| `grammar_region_summary.csv` | aggregates |
| `region_asset_manifest.json` | pack emission header |

### 11.3 Pack emission defaults (locked)

```text
grammar_scope: G1_G2_G3
producer_kind: grammar_atlas
action_state: observation_only
production_forbidden: true
maturity_declared: A0
pack_claim_ceiling: L1
```

### 11.4 Validation invariants

```text
every FK resolves to exactly one authority row
content IDs independent of pack_id and evidence_bundle_id
region_mask_link == DISTINCT(region_asset_id, mask_unit_id) FROM coordinates
both capacity distributions present for multi-member regions
no additive region capacity metric
sequence union and intersection both present when multi-member
G3: members=40, n_non_null=0, n_null_records≥1, null.semantic_definition_id IS NULL
claim_level: G1 objects L0; G2 objects L1; G3 null L0; pack ceiling L1
feasibility_contract has no claim_level fields
truth_contract has no raw SHA map
production_forbidden=true; action_state=observation_only for this study pack
```

---

## 12. Maturity / action / claim firewall

### 12.1 Asset maturity A0–A4

| Level | Permission |
|:--|:--|
| A0 | describe (current) |
| A1 | compare/rank/diff/reproduce after research acceptance of pack |
| A2 | condition/shadow candidates after transfer evidence |
| A3 | default-off intervention validation |
| A4 | production-approved (separate governance) |

### 12.2 Claim ladder L0–L6

As defined in the mathematical framework. Engineering merge ≠ L advance.  
This study: **L0/L1 only**.

### 12.3 Action states

| State | This study pack |
|:--|:--|
| observation_only | **required** |
| shadow_decision / condition_model / offline_filter / default_off_intervention | **forbidden** |
| production_forbidden | **true** |

### 12.4 Forbidden promotions (non-exhaustive)

```text
safe_region ⇒ gate or production policy
union sequences ⇒ applicability
component capacity sum ⇒ simultaneous multi-mask action
observed GT0 ⇒ population risk zero
pack L1 ceiling ⇒ every object L1
A1 packaging ⇒ transferable or actionable
missing files ⇒ NULL_RESULT
mask equality ⇒ semantic / G7 equivalence
```

---

## 13. R1 conversion boundary

### Authorized only after chat-side R0-B acceptance (not now)

Deterministic packaging:

```text
inputs:  sealed Q4.5 full atlases + T0-B-R1 pack + this contract
outputs: authority + emission tables in §11
process: no evaluator modification/rerun; no new research geometry claims
```

### Remains blocked even after R0-B accept until further gates

```text
A1 research maturity acceptance
L2+ claim levels
event-union capacity
G4–G7 / G7 roles
LOO/shadow/hook/preset/production
evidence_ledger promotion
```

### R1 readiness (engineering, not authorization)

Existing sealed evidence is sufficient for deterministic conversion under this contract once R0-B is accepted. Runtime full atlases must remain hash-sealed for lattice membership.

---

## 14. Q4.5 instance snapshot (normative freeze, not new results)

```text
truth: stage2_q45_atlas_v4 · substrate stage1_baudit_d_online · universe online_hook_eligible
cohort: n_gt_exposed=64 · n_fp_exposed=23 · unresolved firewall on
PS: 154 = 1 G1 + 153 G2 + 0 G3
components: 26 · productive mask units: 34
radius≥1: 0/154 · terminal B
claims: G1 L0 · G2 L1 · G3 L0 · pack ceiling L1 · maturity A0
```

---

## 15. Open decisions (non-blocking for R0-B draft review)

These may be fixed at acceptance without reopening R0-A:

| ID | Topic | Default if silent-accept |
|:--|:--|:--|
| O1 | Display truncation length for IDs | 32 hex + full digest column |
| O2 | Whether `evidence_claims` is csv or jsonl | jsonl |
| O3 | Robust floor definition | min member capacity |
| O4 | Whether G1 low-population isolated points may carry optional L0+geometry tags | tags ≠ L1 |

No open decision may re-merge claim_level into feasibility_contract_id or drop `grid_domains` authority.

---

## 16. Explicit non-authorization

```text
R0-B draft is not self-accepted
R1 not authorized
no evaluator rerun
no asset pack generated
no A0→A1 promotion
no PR required for this draft alone
evidence_ledger unchanged
production/presets unchanged
```
