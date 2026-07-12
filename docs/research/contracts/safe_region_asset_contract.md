# Safe-Region Asset Contract (R0-B)

<!-- doc-status: active -->
<!-- doc-promotion: none; method/schema contract only; not evidence_ledger -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: cross -->

**Role:** Normative cross-cutting **RegionAsset** contract (R0-B · revision **R0-B-R3** + editorial **E1**).  
**Status:** **ACCEPTED** — chat-side final review of delivery tip `f92340b7` (acceptance commit `01a2ec37`); RB1–RB9 **PASS**.  
**R1 authorization:** deterministic G1–G3 packaging of sealed evidence into an **A0 observation-only pack candidate** only.  
**Still not authorized:** A0→A1 maturity · transfer · intervention · production · ledger promotion · evaluator rerun · new threshold/geometry search · research verdict self-acceptance.

```text
R0-B-R1 corrections (retained):
  RB1 object claim_level from geometry (not grammar-wide G2=L1)
  RB2 region_asset_manifest.json is authoritative pack row
  RB3 region/null IDs + claims bind feasibility_contract_id (model A)
  RB4 pairwise leaf a/b full-field canonicalize before truth digest

R0-B-R2 corrections (retained):
  RB5 stable coordinate/mask identity vs feasibility-bound membership
  RB6 candidate-universe + predicate-edge machine authorities
  RB7 canonical executable Boolean policy identity (≠ observed mask)

R0-B-R3 corrections (retained):
  RB8 sealed candidate-universe contract vs instance membership identity
  RB9 threshold-registry authority + concrete threshold-bound policy grain

Editorial E1 (non-substantive; does not reopen RB8/RB9):
  §2.4 inverted positive implications → explicit non-implications / forbidden inferences
```

**Parents**

| Document | Role |
|:--|:--|
| [boolean_composition_semantics_contract.md](boolean_composition_semantics_contract.md) | **normative** Boolean / Ω / predicate / policy AST semantics |
| [statistical_robust_feasible_set_estimation_under_asymmetric_loss.md](statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) | mathematical / claim-ladder semantics |
| [safe_region_r0_asset_contract_preflight_20260710.md](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md) | accepted R0-A derivability packet (CR1–CR9) |
| [safe_region_assetization_20260710.md](../threads/closed/safe_region_assetization_20260710.md) | program ownership thread |

**Contract version:** `region_asset_contract_v0`  
**ID scheme:** `region_asset_id_v2`  
**Digest:** SHA-256; display may truncate to 32 hex; equality fail-closed on full digest.

---

## 1. Scope and non-claims

### In scope

- Normative identity layers, grains, machine schemas, validation invariants.
- Mapping of accepted Q4.5 G1–G3 / T0 evidence into a future deterministic pack.
- Separation of mathematical feasibility definition from observed claim level.
- Separation of truth-level coordinate/mask identity from feasibility-bound realization.
- Typed candidate-universe, predicate-edge, and executable Boolean policy authorities.
- Sealed candidate-universe **instance** identity (membership digest) vs generator **contract**.
- Threshold-registry authority and concrete threshold-bound policy instance reconstruction.

### Out of scope / non-claims

```text
≠ accepted A1 maturity
≠ population risk bound
≠ held-out / LOO / LOSO region transfer (L2+)
≠ online retention (L5) or production (L6 / A4)
≠ G4–G7 implementation or G7 role invention
≠ NOT / complement as reject authorization
≠ single-step intervention or closed-loop policy safety
≠ evaluator rerun or modification
≠ production preset / hook / shadow policy change
≠ evidence_ledger promotion
```

Current sealed study remains **A0** descriptive atlas with terminal **B** `isolated_safe_points_only`.

### Typed-space boundary (from Boolean semantics)

```text
candidate / reject set in Ω
≠ policy / feasible set in Θ

truth-level coordinate or mask identity
≠ feasibility-bound region membership / outcome

canonical policy semantics
≠ observed-mask equivalence

observational Boolean composition
≠ single-step intervention
≠ closed-loop policy safety

candidate-universe generator contract
≠ sealed candidate-universe instance membership

parameterized policy family (AST / grammar)
≠ concrete threshold-bound executable policy instance

raw source_event_table_sha256
≠ normalized universe_membership_digest

threshold index alone
≠ reconstructible threshold value without registry authority
```

---

## 2. Mathematical object

A RegionAsset packages an estimated **productive-safe set** under asymmetric loss on a declared parameter space and candidate universe.

### 2.1 Symbols (operational)

| Symbol | Contract field | Q4.5 G1–G3 freeze |
|:--|:--|:--|
| $\Theta$ | `parameter_or_policy_space` | registered thr-index lattices (G1 unique-boundary 870; G2/G3 q05 17640 each) |
| $\Omega$ | `candidate_universe_instance_id` → **`candidate_universe_instances`** (+ contract row) | sealed `online_hook_eligible` **instance** (contract + membership digest); not bare string / generator-only |
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

### 2.4 Composition level freeze (Q4.5)

```text
composition_level = observational
same declared universe and pre-decision state
reject-only G1/G2/G3 AND/OR
no NOT/complement authorization
unknown never maps to reject
no cross-universe composition without transport
```

**Forbidden inferences (editorial E1 — non-implications):**

```text
generator-contract equality ⇏ same sealed universe instance
source_event_table_sha256 ⇏ universe_membership_digest
policy family ⇏ concrete threshold-executable policy
thr_index without registry ⇏ reconstructible thr_value
```

These four implications are **rejected** by the accepted RB8/RB9 model and by §12.4 firewalls. Implementations must treat them as invalid inferences, not as reconstruction rules.

Observational mask algebra on frozen rows **does not** establish single-step intervention or closed-loop policy composition.

---

## 3. Identity layers

Layers are **independent**. Child content IDs must not depend on pack materialization or raw file byte order.

| Layer | ID | Owns | Must not own |
|:--|:--|:--|:--|
| Normalized truth | `truth_contract_id` | substrate, **universe instance** FK, signals, sequences, label/unresolved, lattices, threshold_registry FK, **order-insensitive data content digest** | raw file SHAs; pack schema; claim L-level |
| Universe generator contract | `candidate_universe_contract_id` | substrate, hook, builder, prefilters, key schema, label/exposure owner, time/frame range, pre-decision state **schema** | sealed membership; claim outcomes |
| Universe sealed instance | `candidate_universe_instance_id` | contract_id + **normalized `universe_membership_digest`** | generator-only metadata sold as same-universe equality; raw event-table SHA as sole ID |
| Predicate edge | `predicate_id` | signal, unit, domain/codomain, unknown/missing/NaN/Inf, comparator, endpoint, tie, quantile method, clipping | observed masks; thr_index alone as full identity; threshold **value** (registry owns value) |
| Threshold registry | `threshold_registry_id` (+ entry keys) | sealed thr_index → `threshold_value_repr` / thr_value mapping per feature×direction×lattice | coordinate PK floats; claim outcomes |
| Policy family (parameterized) | `policy_family_definition_id` | grammar + truth_semantics versions, **canonical parameterized AST** (predicate/role leaves **without** thr values), universe **instance** requirement, composition_level, NOT/complement metadata when applicable | thr_index bindings; observed_mask_hash; evidence outcomes |
| Concrete policy instance | `policy_instance_id` | family + `threshold_registry_id` + **ordered threshold bindings** (thr_index per axis/leaf) | parameterized family alone; observed_mask as identity |
| Exact evidence seal | `evidence_bundle_id` | study_id, recorded evaluator/runner SHAs, **raw artifact SHA map** including `source_event_table_sha256` | semantic content identity; substitute for universe membership digest |
| Feasibility definition | `feasibility_contract_id` | $\Theta$, **$\Omega$ instance** FK, $L_{\mathrm{GT}}$, productivity, $\varepsilon$, $g_{\min}$, exposure **definitions and declared denominators**, selection_scope, finite-sample statement, geometry metric/edge policy | **supported claim_level** (outcome) |
| Pack / materialization | `pack_id` | producer kind/version, schema version, grammar_scope, FKs to truth/feasibility/evidence/**universe instance**/threshold_registry | local region/mask/coord content |
| Grid domain | `grid_domain_id` | one registered feature×direction (pair) lattice domain + registry lattice_kind | pack |
| Search domain | `search_domain_id` | membership digest over concrete grid members | summary counts alone |
| Region content (feasible-set outcome) | `region_asset_id` | connected PS component within one grid **under one** `feasibility_contract_id` (model A) + **policy_family** FK | pack_id; evidence_bundle_id; `::compN`; thr values as PK |
| Mask unit (truth-level) | `mask_unit_id` | `(truth_contract_id, grid_id, mask_sha256)` — **feasibility-independent** | PS membership counts that change with feasibility; region FKs as identity |
| Coordinate (truth-level) | `coordinate_id` | `(truth_contract_id, threshold_registry_id, canonical_cell_key)` with **per-axis registry entry FKs** — **feasibility-independent** | productive-safe flags; capacity under a feasibility; region FK; thr_value as PK |
| Membership realization | `(region_asset_id, coordinate_id)` | feasibility-bound PS membership, observed capacity, dual margins, sequence incidence; may cite `policy_instance_id` | redefining coordinate/mask content IDs |
| Null record (feasible-set outcome) | `null_record_id` | search-domain empty under **one** `feasibility_contract_id` | concrete policy on grammar-level null |
| Evidence / claim outcome | `evidence_claim_id` (row) | observed outcomes + **claim_level** bound to **feasibility_contract_id + evidence_bundle_id** | redefining feasibility math |
| Pack membership | `(pack_id, content_kind, content_id)` | emission listing | redefining content IDs |

**Retired / forbidden patterns:**

```text
asset_set_id (R0-A name)
truth_context_id that embeds raw SHAs as content parent
feasibility-independent coordinate_id as sole PK of a table that stores PS flags / margins / capacity
bare string candidate_universe_id without contract + instance authority rows
generator-only universe ID used for same-universe Boolean / mask comparison
source_event_table_sha256 used as universe_membership_digest substitute
fabricated universe_membership_digest from dataset name without candidate rows
policy_family_definition_id labeled as concrete executable threshold policy
semantic_definition_id that digests only operator+leaves+lattice without executable AST / roles / truth semantics
observed_mask_hash as policy identity
thr_index without threshold_registry entry FK as reconstructible threshold
```

**Naming aliases (locked semantics):**

```text
candidate_universe_id  (when used on pack / feasibility / same-universe compare)
  ≝ candidate_universe_instance_id     # NOT the generator contract alone

policy_definition_id / semantic_definition_id
  without threshold bindings  ≝ policy_family_definition_id
  with ordered thr bindings + registry  ≝ policy_instance_id

legacy single-field "universe_hash"
  ≝ universe_membership_digest on the instance row (name may remain as column alias)
```

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
| Observed `n_gt_hurt`, `n_neg_captured`, per-object geometry | membership / capacity / margin tables + evidence_bundle seal |
| `selection_scope` realization note | evidence_claims / claim contract |
| `finite_sample_statement` restatement | claim contract (definition text may be shared) |
| **`claim_level` per object** | evidence_claims / region_assets.claim_level / null_records.claim_level |
| **`pack_claim_ceiling`** | region_claim_contract.json / pack manifest |

### 4.3 Object claim_level derivation (RB1 — not grammar-hardcoded)

**Rule (fail-closed):** derive `claim_level` from **object geometry / evidence shape**, never from grammar name alone.

```text
IF object is grammar/search-domain NULL_RESULT:
  claim_level = L0

ELSE IF non-null region with n_coords == 1
     OR shape_class == isolated_point
     OR not a multi-coordinate connected PS component:
  claim_level = L0   # observed productive-safe point only

ELSE IF non-null region with n_coords >= 2
     AND connected PS component under declared adjacency
     AND in-sample selection_scope only:
  claim_level = L1   # in-sample multi-coordinate region geometry

ELSE:
  fail closed or leave claim_level unset until shape is declared
```

**Forbidden:** `G2 → L1` as a grammar-wide constant. Isolated G2 components are **L0**.

### 4.4 Q4.5 sealed counts (from T0 `component_geometry.csv`)

| Object class | Count | claim_level |
|:--|--:|:--|
| G1 non-null (isolated, n_coords=1) | 1 | **L0** |
| G2 isolated (`isolated_point`, n_coords=1) | **6** | **L0** |
| G2 multi-coordinate (n_coords≥2) | **19** | **L1** |
| G3 grammar/search-domain null | 1 | **L0** |
| Pack aggregate ceiling | — | **L1** (max over objects only) |

```text
G1: 1 × L0
G2: 6 × L0 isolated + 19 × L1 multi-coordinate
G3: 1 × L0 domain null
pack_claim_ceiling = L1
⇏ every region_asset.claim_level = L1
⇏ every G2 region_asset.claim_level = L1
```

### 4.5 Orthogonal ladders

```text
A0–A4  asset maturity / packaging / action readiness
L0–L6  statistical / transfer claim strength
```

Neither ladder implies the other. A future A1 pack of this study remains **L0/L1 only**.

---

## 5. Object grains and relations

```text
truth_contract
candidate_universe_contract
candidate_universe_instance ──FK──► contract; referenced by truth/feasibility/pack/policy
threshold_registry (+ entries) ──referenced by──► truth_contract, coordinates, policy instances
predicate_definitions ──referenced by──► policy family AST leaves
policy_family_definitions ──FK──► universe instance, predicate leaves
policy_instances ──FK──► policy_family, threshold_registry, ordered thr bindings
evidence_bundle ──FK──► truth_contract   # owns source_event_table_sha256 (raw seal)
feasibility_contract ──FK──► truth_contract, candidate_universe_instance
pack (region_asset_manifest.json) ──FK──► truth, feasibility, evidence, universe instance, threshold_registry

grid_domain                  (registered lattice domain; axes after pairwise canonicalize)
search_domain
search_domain_member ──FK──► search_domain, grid_domain, policy_family_definition

coordinates                  # truth-level; PK coordinate_id; per-axis registry entry FKs
mask_units                   # truth-level; PK mask_unit_id  (file may be region_masks.csv)

region_asset ──FK──► truth, feasibility, policy_family, grid_domain
null_record ──FK──► truth, feasibility, search_domain; policy_family = NULL
region_coordinate_membership ──FK──► region_asset, coordinate, mask_unit
  # optional policy_instance_id derived from family + coordinate thr bindings
  # feasibility-bound realization; sole region↔mask authority projection source

pack_membership ──FK──► pack, content
evidence_claim ──FK──► feasibility_contract, evidence_bundle, optional content_id
```

**RB3 model A (locked):** feasible-set **outcomes** (`region_asset_id`, `null_record_id`) include `feasibility_contract_id` in their identity digests. Changing $\varepsilon$, $g_{\min}$, loss, universe instance, or denominators yields new outcome IDs.

**RB5 (locked):** `mask_unit_id` and `coordinate_id` remain feasibility-independent and **must not** own fields that change with feasibility. Those fields live only on membership/realization rows.

**RB8 (locked, model B):** generator **contract** and sealed **instance** are distinct authorities. Packs, feasibility contracts, evidence claims, and same-universe Boolean/mask comparisons bind **`candidate_universe_instance_id`**.

**RB9 (locked):** threshold values come only from `threshold_registry` entries. Parameterized **family** ≠ concrete **instance** (family + ordered thr bindings + registry).

### 5.1 Primary grains

| Grain | Definition |
|:--|:--|
| Region (non-null) | Connected **productive-safe** component within **one** registered grid under declared adjacency, one feasibility contract, and one **policy family** |
| Coordinate (truth-level) | Registered thr-index cell (`canonical_cell_key`) under one truth contract and one threshold registry, with per-axis entry FKs |
| Mask unit (truth-level) | Distinct `mask_sha256` within one `grid_id` under one truth contract |
| Membership realization | `(region_asset_id, coordinate_id)` with optional `mask_unit_id` + observed PS/capacity/margin fields under that region’s feasibility; concrete `policy_instance_id` is reconstructible |
| Null record | First-class empty productive-safe result on a **search domain**, not missing files |
| Search domain | Ordered membership of concrete grid domains + policy **families** |
| Policy family | Parameterized canonical Boolean AST + grammar/truth semantics + roles + universe instance (no thr values) |
| Policy instance | Family + threshold_registry + ordered thr_index bindings — concrete executable threshold policy |
| Universe contract / instance | Generator schema vs sealed membership (RB8 model B) |
| Threshold registry | Sealed thr_index → threshold_value_repr authority for reconstruction |

### 5.2 Cardinality rules

- One region may contain many mask units; one mask unit may appear in many regions **in schema** (M:N via membership). Observed Q4.5: multi-mask regions exist; multi-region masks not observed — still do not assume many-to-one.
- **Authoritative** region↔mask relation: projection from **`region_coordinate_membership`** FKs. Optional `region_mask_link` is derived only.
- One truth-level `coordinate_id` **may** appear under multiple `region_asset_id`s across different feasibility contracts without row conflict, because membership tables are separate from coordinate authority.
- G3 sealed study: `n_non_null_region_assets = 0`, `n_null_records = 1`, `search_domain_members = 40`.

### 5.3 Forbidden identity inputs

```text
component ordinals (::comp0, component_03, …)
global mask_sha256 alone as PK
pack_id / producer_contract_version / schema_version inside content digests
raw file SHA maps inside content digests
human thr_value strings as PKs (aliases only)
observed_mask_hash inside policy_family / policy_instance / semantic_definition digests
feasibility-dependent PS flags inside coordinate_id or mask_unit_id digests
source_event_table_sha256 inside candidate_universe_instance_id
dataset name alone as universe_membership_digest
thr_value float as coordinate_id PK
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
thr_value never used in coordinate_id / region_asset_id PKs
threshold_value_repr required on registry entries for execution reconstruction
```

### 6.2 Content ID inputs (normative)

**`region_asset_id` (RB3 model A)**

```json
{
  "id_scheme": "region_asset_id_v2",
  "kind": "region_asset",
  "truth_contract_id": "<hex>",
  "feasibility_contract_id": "<hex>",
  "policy_family_definition_id": "<hex>",
  "grid_domain_id": "<hex>",
  "adjacency": "G1_bilateral" | "G2_4neighbor",
  "membership": "productive_safe",
  "coordinate_digest": "<sha256 of sorted canonical thr-index keys>"
}
```

`policy_family_definition_id` is the parameterized Boolean/grammar identity (§6.6). Concrete cell-level executable identity is `policy_instance_id` (§6.7). Legacy column name `policy_definition_id` / `semantic_definition_id` is allowed only as a synonym of **family** when threshold bindings are absent.

Coordinate keys: G1 sorted `[thr_index]`; G2 sorted `[[i,j],…]` with axes in **canonical leaf order** after §6.3 pairwise normalization (not raw evaluator a/b order).

**`mask_unit_id` (truth-level, RB5):** `truth_contract_id` + `grid_id` + `mask_sha256` — **no** feasibility, **no** PS counts, **no** region FK.  
**`coordinate_id` (truth-level, RB5 + RB9):** `truth_contract_id` + `threshold_registry_id` + **canonical_cell_key** + **per-axis `threshold_registry_entry_id` FKs** — **no** feasibility, **no** PS flags, **no** capacity/margins; **no thr_value in PK**.  
**`null_record_id` (RB3 model A):**

```json
{
  "id_scheme": "region_asset_id_v2",
  "kind": "null_record",
  "truth_contract_id": "<hex>",
  "feasibility_contract_id": "<hex>",
  "search_domain_id": "<hex>",
  "null_reason_class": "no_productive_safe_on_registered_domain"
}
```

Changing $\varepsilon$ / $g_{\min}$ / loss / universe / denominators ⇒ new `null_record_id` and new `region_asset_id`s even if geometry coordinate keys match.

**`search_domain_id`:** `truth_contract_id` + `membership_digest` over canonical member rows.  
**`grid_domain_id`:** lattice_kind + **canonical** ordered axis descriptors (pairwise: leaves sorted by `(feature,direction)`).  
**`pack_id`:** digest of pack authority row fields (or explicit assigned id recorded in `region_asset_manifest.json`).  
**`candidate_universe_contract_id` / `candidate_universe_instance_id` / `universe_membership_digest`:** §6.5 (RB8).  
**`predicate_id`:** §6.5.  
**`policy_family_definition_id`:** §6.6 (RB7 family grain).  
**`policy_instance_id` / `threshold_registry_id`:** §6.7 (RB9).

### 6.3 Pairwise operand full-field canonicalization (RB4)

For symmetric hard AND/OR **without differentiated roles**, before any of: policy AST digest, semantic digest, grid_domain axes, truth-row digest, coordinate keys, region membership coordinates:

```text
leaf_a = (atom_id, feature, direction, thr_index, role)_a
leaf_b = (atom_id, feature, direction, thr_index, role)_b

if (feature_a, direction_a) > (feature_b, direction_b)  # lexicographic
  swap leaf_a ↔ leaf_b entirely
  # after swap, all of: atom_id, feature, direction, thr_index, role move together
```

**Role provenance survives the swap** (RB7). Roles are not dropped during commutative sorting.

Tie-break if `(feature, direction)` equal (should not occur for distinct pairwise axes): sort by `(atom_id, thr_index, role)`.

If roles differ across commutative children, sorting is still by the full leaf key after binding roles; role fields remain on each child in the canonical AST.

After swap:

| Use | Canonical form |
|:--|:--|
| Semantic / policy operands | sorted leaves; thr_index excluded from policy leaf identity when thr is grid axis (policy leaf refs predicate + direction; thr is coordinate) |
| Grid axes / `grid_id` logical form | `P::{feat0}::{dir0}__{feat1}::{dir1}` with feat0/dir0 ≤ feat1/dir1 |
| Truth row fields | store as `atom_0_id`, `feature_0`, `direction_0`, `thr_index_0`, `atom_1_id`, … (or a/b **after** swap) |
| Coordinate key | `(thr_index_0, thr_index_1)` in that axis order |
| Region membership | same remapped indices |

**Native evaluator `combo_id` / raw `cell_id`:** may be retained as **alias** only.  
**`canonical_cell_key` for identity:**

```text
G1: "S::{feature}::{direction}::u{thr_index}"
G2/G3: "{AND|OR}::{atom_0_id}::{atom_1_id}"   # atoms already thr-indexed P::…::qK after leaf swap
```

Equivalent `A AND B` ↔ `B AND A` raw rows must produce **identical** truth digests, policy_ids, grid_domain_ids, coordinate_ids, and region membership digests **when roles and other non-order fields match**.

### 6.4 Stability requirements

| Event | evidence_bundle | truth_contract | region/null outcome IDs | mask/coord IDs | membership rows |
|:--|:--|:--|:--|:--|:--|
| Row reorder, same logical rows | may change | stable | stable | stable | stable (content) |
| Symmetric a/b operand swap | may change raw bytes | **stable** (after RB4) | stable | stable | stable |
| CSV↔Parquet rematerialization | may change | stable | stable | stable | stable |
| Pack/schema version only | stable (if raw same) | stable | stable | stable | stable |
| Feasibility $\varepsilon$/$g_{\min}$/loss change | — | stable | **change** | **stable** | **new rows / new region FKs** |
| Cohort / label / signal / lattice / membership change | changes | changes | change | change | change |

### 6.5 Candidate universe contract vs instance (RB6 + RB8 model B)

**Locked model B:** split generator contract from sealed instance.

#### 6.5.1 `candidate_universe_contract_id`

Digests **generator / schema** fields only (no membership rows):

```json
{
  "kind": "candidate_universe_contract",
  "substrate_id": "...",
  "hook_id": "...",
  "candidate_builder_id": "...",
  "candidate_builder_version": "...",
  "prefilter_contract_id": "...",
  "eligibility_contract": "...",
  "candidate_key_schema": {
    "primary_key_columns": ["..."],
    "column_types": {"...": "string|int|..."}
  },
  "label_exposure_contract_id": "...",
  "label_exposure_columns": ["..."],
  "observation_time_or_frame_range": "...",
  "predecision_state_snapshot_contract": "..."
}
```

Same contract_id may apply to multiple sealed instances.

#### 6.5.2 Normalized `universe_membership_digest` algorithm

**Fail-closed.** Do not invent membership from a dataset name or bare `online_hook_eligible` string.

```text
REQUIRE candidate-level rows that reconstruct the sealed observation set
  (from sealed study artifacts). If unavailable → BLOCKED_BY_ARTIFACT.

project columns =
  candidate_key_schema.primary_key_columns
  ∪ label_exposure_columns
  ∪ any membership-critical columns declared by predecision_state_snapshot_contract

for each candidate row:
  project only those columns
  missing required column → fail closed
  missing optional → omit key
  integer/bool → JSON numbers / 0|1 as elsewhere
  floats (if any membership column) → shortest round-trip decimal (§7.5)
  row_obj = canonical_json(projected object)   # sorted keys
  row_digest = SHA256(UTF-8 bytes of row_obj)  # lowercase hex

sort unique row_digests lexicographically as hex strings
  duplicate PK + identical projected payload → collapse to one
  duplicate PK + conflicting payload → fail closed

universe_membership_digest =
  SHA256( UTF-8( join(sorted_row_digests, "\n") + "\n" ) )
```

**Separation:**

```text
source_event_table_sha256   ∈ evidence_bundle   # exact raw bytes seal
universe_membership_digest  ∈ universe instance # normalized logical membership
```

Changing candidate membership under the same generator contract **must** change `universe_membership_digest` and therefore `candidate_universe_instance_id`.

#### 6.5.3 `candidate_universe_instance_id`

```json
{
  "kind": "candidate_universe_instance",
  "candidate_universe_contract_id": "<hex>",
  "universe_membership_digest": "<hex>"
}
```

Column alias `universe_hash` ≝ `universe_membership_digest`.  
Field name `candidate_universe_id` on pack/feasibility/policy/**same-universe compare** **must** resolve to **`candidate_universe_instance_id`**.

#### 6.5.4 Predicate IDs (RB6 retained)

**`predicate_id`** digests at least:

```json
{
  "kind": "predicate",
  "signal_identity": "...",
  "signal_unit": "...",
  "predicate_domain": "...",
  "predicate_codomain": "{T,F,U}" | "{T,F}",
  "unknown_value_policy": "...",
  "final_unknown_action": "no_reject" | "reject" | "...",
  "comparator": ">" | ">=" | "<" | "<=" | "...",
  "endpoint_policy": "...",
  "tie_policy": "...",
  "nan_policy": "...",
  "posinf_policy": "...",
  "neginf_policy": "...",
  "missing_value_policy": "...",
  "quantile_method": "...",
  "floating_point_tolerance": "...",
  "clipping_domain": "..."
}
```

Two-valued codomain `{T,F}` must be **declared and justified**, never inferred from a sample that happened to contain no unknowns.  
**Threshold numeric values are not part of `predicate_id`** — they come from the threshold registry (RB9).

### 6.6 Policy family identity (RB7, parameterized grain)

**`policy_family_definition_id`** digests **parameterized** canonical executable semantics — operator tree, predicate refs, roles, grammar bounds — **without** thr_index / thr_value bindings:

```json
{
  "kind": "policy_family_definition",
  "grammar_version": "...",
  "truth_semantics_version": "...",
  "composition_level": "observational" | "single_step" | "closed_loop",
  "candidate_universe_instance_id": "<hex>",
  "operator_precedence": "NOT>AND>OR",
  "maximum_nesting_depth": <int or null-bound>,
  "maximum_operands_per_node": <int>,
  "not_scope": "none" | "atom_only" | "arbitrary_subtree",
  "mixed_role_policy": "...",
  "canonical_policy_ast": { "...": "typed AST with roles and predicate refs; no thr values" },
  "canonical_policy_ast_hash": "<sha256 of canonical AST JSON>"
}
```

**Three distinct equivalence fields** (never collapsed into one ID):

| Field | Meaning |
|:--|:--|
| `canonical_policy_ast_hash` / `policy_family_definition_id` | parameterized syntactic / grammar identity |
| `logical_equivalence_status` | policies agree for every legal input under declared truth semantics (optional evidence annotation) |
| `observed_mask_hash` / `observed_mask_equivalence_status` | sample agreement only — attach to **policy instance** / coordinate, not family alone |

```text
observed_mask equality ≠ logical equivalence ≠ policy family identity ≠ policy instance identity
```

**Role rules (retained):**

- Every atom and subtree carries a declared role.
- Q4.5 sealed operands that are not role-qualified use bounded role  
  `untyped_observation` (or an explicit pack-declared `observational_reject_candidate` alias) — **not** silent `sufficient_reject` upgrade, and **not** G7 `necessary_envelope` / `support`.
- Role metadata must not disappear during commutative sorting or operand canonicalization.
- Current G1–G3 serialization **must not invent** G7 necessary/support roles.
- `NOT` / complement metadata are required only when NOT appears; Q4.5 freezes `not_scope=none` and does not authorize complement reject policies.

**RegionAsset rows** FK to the **`policy_family_definition_id`** of the atlas grammar family.  
**Observed masks and concrete execution** bind to **`policy_instance_id`** (or the reconstructible composite below).

### 6.7 Threshold registry and concrete policy instance (RB9)

#### 6.7.1 `threshold_registry_id` and entries

Sealed Q4.5 authority: evidence  
`docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/threshold_registry.json`  
(file sha256 in study `SHA256SUMS.json`: `d3e3197fa7812a9ec5f9b06cc2286dcce52d49cf805eba6527c3b24b62a585f4`).

**`threshold_registry_id`** digests at least:

```json
{
  "kind": "threshold_registry",
  "taxonomy_version": "stage2_q45_atlas_v4",
  "single_lattice_kind": "primary_unique_boundaries",
  "pairwise_lattice_kind": "primary_quantile_lattice_q05",
  "signals_primary": ["score_m_bridge", "abs_log_h", "dist_h", "abs_ratio_m1", "resid_mean"],
  "directions": ["high_tail", "low_tail"],
  "combinators": ["AND", "OR"],
  "entries_digest": "<sha256 over sorted entry digests>"
}
```

**Entry grain** (`threshold_registry_entry_id` or composite PK):

```text
(threshold_registry_id, lattice_kind, feature, direction, thr_index)
```

Required entry fields:

```text
atom_id
feature, direction, lattice_kind
thr_index                 # integer lattice coordinate
threshold_value_repr      # required reconstruction string (shortest round-trip of thr_value)
thr_value                 # numeric, for execution; excluded from coordinate PK
quantile_lattice_point    # when pairwise q05 lattice; omit when N/A
scope                     # single_atom | pairwise_atom
```

`entries_digest` algorithm: for each entry, `SHA256(canonical_json(entry without optional display-only fields))`, sort hex digests, hash concat as in §6.5.2.

Two independent packers **must** reconstruct identical `threshold_value_repr` and entry IDs from the sealed registry without guessing floats.

#### 6.7.2 Coordinate ↔ registry binding

Every axis of every coordinate **must** resolve to exactly one registry entry:

```text
G1 axis: (single_lattice_kind, feature, direction, thr_index) → entry
G2/G3 axes: for each ordered leaf after §6.3,
  (pairwise_lattice_kind, feature_i, direction_i, thr_index_i) → entry_i
```

#### 6.7.3 `policy_instance_id` (concrete executable threshold policy)

```json
{
  "kind": "policy_instance",
  "policy_family_definition_id": "<hex>",
  "threshold_registry_id": "<hex>",
  "threshold_bindings": [
    {"axis": 0, "feature": "...", "direction": "...", "thr_index": 0, "threshold_registry_entry_id": "<hex>"},
    {"axis": 1, "...": "..."}
  ]
}
```

Bindings are ordered in **canonical leaf / axis order** after §6.3.  
Equivalent reconstructible composite (must yield the same identity when used):

```text
(policy_family_definition_id, coordinate_id, threshold_registry_id)
```

when `coordinate_id` already embeds the ordered thr_index keys and registry entry FKs.

```text
FORBIDDEN: treat policy_family_definition_id alone as a concrete threshold-executable policy
FORBIDDEN: attach observed_mask_hash identity only to the family without instance/coordinate
```

### 6.8 Aliases

Human thr expressions and T0 `::compN` strings are **aliases only**, never join keys.  
Raw evaluator `combo_id` / pre-swap a/b order are aliases only after §6.3.  
`threshold_value_repr` may appear as a human alias column; identity uses registry entry IDs + thr_index, not free thr_value floats as PKs.

---

## 7. Normalized truth digest

`truth_contract.normalized_data_content_digest` is the sole logical-data input that makes truth identity independent of byte order.

### 7.1 Algorithm

```text
for each declared truth table T:
  project only truth-bearing columns (§7.2)
  drop excluded columns (§7.3)
  if T is pairwise_and_atlas or pairwise_or_atlas:
    apply §6.3 full-field leaf swap to every row  # RB4 — before key/digest
    replace raw combo_id with canonical_cell_key for identity fields
  apply missing/float/duplicate rules (§7.4–7.5)
  sort rows by stable row key ascending
  for each row: row_digest = SHA256(canonical_json(row))
  table_digest[T] = SHA256(concat(row_digests in sorted order))

normalized_data_content_digest =
  SHA256(canonical_json({T: table_digest[T] for T sorted by name}))
```

### 7.2 Truth-bearing tables and stable row keys

| Table (logical) | Stable row key | Truth-bearing columns |
|:--|:--|:--|
| `atom_atlas` | `atom_id` | `atom_id`, `feature`, `direction`, `thr_index`, `lattice_kind`, `observed_safe_point`, `productive_safe_point`, `gt_hurt`, `n_neg_captured`, `n_gt_captured`, `n_unresolved_selected`, `safety_status`, `mask_sha256`, nested `per_sequence_neg`, nested `per_sequence_gt` |
| `pairwise_and_atlas` | **`canonical_cell_key`** (after §6.3) | `canonical_cell_key`, `combinator`, `atom_0_id`, `atom_1_id`, `feature_0`, `direction_0`, `thr_index_0`, `feature_1`, `direction_1`, `thr_index_1`, `lattice_kind`, `observed_safe_point`, `productive_safe_point`, `gt_hurt`, `n_neg_captured`, `n_gt_captured`, `n_unresolved_selected`, `safety_status`, `mask_sha256`, `semantic_duplicate_mask`, `empty_region`, nested sequence maps |
| `pairwise_or_atlas` | **`canonical_cell_key`** (after §6.3) | same column set as AND |
| `threshold_registry_meta` | single logical row | `taxonomy_version`, `signals_primary` (sorted), `directions` (sorted), `combinators` (sorted), `single_lattice_kind`, `pairwise_lattice_kind`, `n_single_atoms`, `n_pairwise_atoms`, `assignment_group_key_status` |
| `cohort_contract` | single logical row | canonical `cohort_definition` object + sorted `sequence_set` + `n_primary_negative` + `n_primary_positive_protect` |
| `t0_component_membership` | `grid_domain_id` + sorted canonical coord keys | `grammar`, canonical `grid_id`/axes, `adjacency`, sorted coordinate key list (**not** T0 `component_id`; pairwise coords remapped via §6.3) |

**Excluded from pairwise truth payload as identity inputs:** raw `combo_id`, pre-swap `atom_a_id`/`atom_b_id` order, pre-swap feature/direction/thr a/b. Those may appear only as non-identity aliases outside the digest.

Truth digest captures **observed atlas tables**. It does **not** replace `policy_family_definition_id`, `policy_instance_id`, `candidate_universe_instance_id`, or `threshold_registry` authorities.

Implementations may read parquet or csv; **logical projection after RB4 canonicalize** is authoritative.

### 7.3 Excluded from truth digest

```text
raw file paths, timestamps, write order
thr_value / thr_value_a / thr_value_b   # alias material; thr_index is coordinate
raw combo_id and pre-canonical a/b field order (aliases only after §6.3)
display aliases, nested_loso clause float strings used only as labels
T0 component_id ordinals (::compN)
row numbers, parquet row groups
evaluator live source path
pack schema fields
loo_* portability marketing flags (may appear on evidence tables, not truth digest)
observed_mask_hash as if it were truth contract identity
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
- Dual margin: `nearest_unsafe_distance` and `full_neighborhood_safe_radius` both required on **membership realization** rows (or a derived margin table keyed by membership); edge-censored distance ≠ thickness.
- Components: connected PS under §2 metric; membership from **membership coordinates**, not `region_stability` mask-quotient rows alone.
- Global mask collapse across grids is diagnostic only.
- Truth-level coordinate/mask rows do not themselves assert productive-safe geometry under a feasibility contract.

### 8.2 Capacity (non-additive)

| Metric family | Member grain | Required stats |
|:--|:--|:--|
| `coordinate_member_capacity_distribution` | PS coordinates in region (via membership) | n_members, min, max, median, q25, q75, robust_floor |
| `mask_member_capacity_distribution` | distinct per-grid mask units in region | same stats |

```text
FORBIDDEN as region mass:
  sum of member mask_n_neg
  sum of coordinate n_neg_captured sold as unique-event mass
  plateau_width × mask capacity
  event-union capacity without sealed event membership  → BLOCKED_BY_ARTIFACT
```

Both distribution families are required for multi-member regions. Neither is event-union capacity.

Mask fields that are conditional on productive-safe membership (e.g. mask capacity under a changed $\varepsilon$) belong on realization/aggregate tables, not on truth-level `mask_units` identity rows. Truth-level mask rows may store only feasibility-invariant facts (mask bitstring hash, coordinate support of that mask on the registered grid as a static signature if sealed as invariant).

---

## 9. Sequence / applicability descriptions

| Field | Meaning | Promotion ban |
|:--|:--|:--|
| Incidence `(unit, sequence)` | coordinate or mask productive support under a realization | not a policy feature |
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
policy_family_definition_id = NULL   # grammar/search-domain null
feasibility_contract_id = <required; in null_record_id digest>
search_domain_id = <membership digest>
n_non_null_region_assets (grammar summary) = 0
n_null_records ≥ 1
observed_safe_count / productive_safe_count on domain
null_reason
claim_level = L0 for Q4.5 G3
```

Null realization is relative to $(\varepsilon,g_{\min},L_{\mathrm{GT}},\Omega,$ denominators$)$. A new feasibility contract **must not** reuse the prior `null_record_id`.

### 10.2 G3 sealed instance

```text
combinator OR · lattice primary_quantile_lattice_q05
40 members · 441 coords each · 17640 total
each member: grid_domain_id + concrete OR policy_family_definition_id
productive_safe_count = 0 · observed_safe_count = 0
```

### 10.3 Illegal encodings

```text
missing atlas / failed run → BLOCKED_BY_ARTIFACT, not NULL_RESULT
empty components file without null_records row
summary-only domain id without member table
fake single policy_family for all 40 grids on the null row
```

---

## 11. Machine schemas and referential integrity

### 11.1 Authority files (normative)

Every FK target below is an authority. No FK may point to an undefined object.

#### `truth_contract.json`

| | |
|:--|:--|
| PK | `truth_contract_id` |
| Required | id_scheme, taxonomy_version, substrate_id, **candidate_universe_instance_id** (FK), **threshold_registry_id** (FK), signal_family[], sequence_set[], label_contract, unresolved_policy, lattice_contract, normalized_data_content_digest |
| Forbidden in identity inputs | raw_artifact_sha256 map |

#### `candidate_universe_contracts.json` | `jsonl` (**RB6/RB8 generator authority**)

| | |
|:--|:--|
| PK | `candidate_universe_contract_id` |
| Required | substrate_id, hook_id, candidate_builder_id, candidate_builder_version, prefilter_contract_id / eligibility_contract, **candidate_key_schema** (primary_key_columns + types), label_exposure_contract_id / label_exposure_columns, observation_time_or_frame_range, predecision_state_snapshot_contract |
| Notes | Generator schema only; may be shared by many instances |

#### `candidate_universe_instances.json` | `jsonl` (**RB8 sealed instance authority**)

| | |
|:--|:--|
| PK | `candidate_universe_instance_id` |
| FK | **candidate_universe_contract_id** |
| Required | **universe_membership_digest** (alias column `universe_hash` allowed), membership_digest_algorithm_version, n_candidates (when known), membership_status (`SEALED` \| `BLOCKED_BY_ARTIFACT`) |
| Optional | transport_id (null for same-universe sealed studies) |
| Invariants | bare string names without this row are **not** machine-sufficient; every pack/feasibility/**same-universe** reference must resolve here; generator-only contract_id is **insufficient** for Boolean/mask comparison; if candidate rows unavailable → `BLOCKED_BY_ARTIFACT` and no fabricated digest |
| Q4.5 freeze | contract for `online_hook_eligible` on substrate `stage1_baudit_d_online`; instance digests sealed membership from study candidate rows (or blocks); `source_event_table_sha256` stays on evidence_bundle only |

#### `threshold_registry.json` | `jsonl` (**RB9 authority**)

| | |
|:--|:--|
| PK | `threshold_registry_id` |
| Required | taxonomy_version, single_lattice_kind, pairwise_lattice_kind, signals_primary[], directions[], combinators[], entries_digest, n_single_atoms, n_pairwise_atoms |
| Entry table | `threshold_registry_entries.csv` \| embedded entries: PK `(threshold_registry_id, lattice_kind, feature, direction, thr_index)` or `threshold_registry_entry_id`; required `atom_id`, `threshold_value_repr`, `thr_value`, optional `quantile_lattice_point` |
| Invariants | every coordinate axis FK resolves to exactly one entry; thr_value excluded from coordinate PK; reconstruction without guessing floats |
| Q4.5 freeze | sealed evidence `threshold_registry.json` (`taxonomy_version=stage2_q45_atlas_v4`; 870 single + 210 pairwise atoms; lattices `primary_unique_boundaries` / `primary_quantile_lattice_q05`); raw file sha256 `d3e3197fa7812a9ec5f9b06cc2286dcce52d49cf805eba6527c3b24b62a585f4` |

#### `predicate_definitions.csv` | `jsonl` (**RB6 authority**)

| | |
|:--|:--|
| PK | `predicate_id` |
| Required | signal_identity, signal_unit, predicate_domain, predicate_codomain (`{T,F,U}` default unless total proven), unknown_value_policy, final_unknown_action, comparator, endpoint_policy, tie_policy, nan_policy, posinf_policy, neginf_policy, missing_value_policy, quantile_method (if quantile lattice), floating_point_tolerance, clipping_domain |
| Forbidden defaults | silent `U → reject`; silent total two-valued codomain; embedding thr_value as predicate identity |
| Q4.5 freeze | three-valued/fail-safe serialization unless the sealed evaluator contract explicitly proves totality; **final_unknown_action = no_reject**; no NOT predicates authorized |

#### `policy_family_definitions.csv` | `jsonl` (**RB7 parameterized authority**)

| | |
|:--|:--|
| PK | `policy_family_definition_id` |
| FK | **candidate_universe_instance_id**; leaf predicate_ids inside AST |
| Required | grammar_version, truth_semantics_version, composition_level, operator_precedence, maximum_nesting_depth, maximum_operands_per_node, not_scope, mixed_role_policy, **canonical_policy_ast** (no thr values), **canonical_policy_ast_hash**, role annotations on every atom/subtree, parameter_system / lattice_kind when atlas-bound |
| Separate non-identity fields | logical_equivalence_status, role_validity_result |
| Notes | filename synonym `policy_definitions` / `semantic_definitions` allowed **only** if grain is family (no thr bindings) and required fields are present |
| Q4.5 freeze | G1 atom / G2 binary AND / G3 binary OR families; `composition_level=observational`; `not_scope=none`; roles = `untyped_observation`; no G7 roles; no complement_contract |

#### `policy_instances.csv` | `jsonl` (**RB9 concrete authority**)

| | |
|:--|:--|
| PK | `policy_instance_id` |
| FK | **policy_family_definition_id**, **threshold_registry_id**, ordered threshold_registry_entry_ids |
| Required | threshold_bindings_json (canonical ordered axes), optional coordinate_id when instance 1:1 with a cell |
| Separate non-identity fields | observed_mask_hash, observed_mask_equivalence_status |
| Invariants | observed masks attach here (or via coordinate) — **not** to family alone; family without bindings is not a concrete executable threshold policy |

#### `evidence_bundle.json`

| | |
|:--|:--|
| PK | `evidence_bundle_id` |
| FK | truth_contract_id |
| Required | study_id, study_git_commit, evaluator_source_sha256, runner_source_sha256, **source_event_table_sha256** (raw seal only), raw_artifact_sha256{}, terminal_letter |
| Notes | `source_event_table_sha256` **≠** `universe_membership_digest` |

#### `feasibility_contract.json`

| | |
|:--|:--|
| PK | `feasibility_contract_id` |
| FK | truth_contract_id, **candidate_universe_instance_id** |
| Required | parameter_or_policy_space, candidate_universe_instance_id, safety_loss_definition, productivity_definition, epsilon, g_min, n_gt_exposed, n_fp_exposed, denominator_owner, selection_scope, finite_sample_statement, metric_adjacency_edge_policy |
| **Forbidden** | claim_level, claim_level_max_supported, observed region counts |

#### `evidence_claims.csv` | `jsonl`

| | |
|:--|:--|
| PK | `evidence_claim_id` |
| FK | **`feasibility_contract_id` (required)**, **`evidence_bundle_id` (required)**; optional content_id + content_kind (`region_asset` \| `null_record` \| …) |
| Required | claim_level (L0–L6), claim_scope (`object`\|`pack`\|`grammar`), selection_scope_note, finite_sample_statement, observed summary fields as applicable; composition_level note when claim is composition-sensitive |
| Invariants | every claim resolves to exactly one feasibility + one evidence bundle; object claim_level follows §4.3 derivation |
| Notes | Pack ceiling row: claim_scope=pack, claim_level=L1; object rows: G1 L0; G2 isolated L0 / multi L1; G3 null L0 |

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
| FK | search_domain_id, **grid_domain_id**, **policy_family_definition_id** (non-null family for that member grid) |
| Required | grid_id, n_registered_coordinates |
| Invariants | G3 sealed: exactly 40 rows; membership_digest matches |

#### `region_assets.csv`

| | |
|:--|:--|
| PK | `region_asset_id` |
| FK | truth_contract_id, **feasibility_contract_id**, **policy_family_definition_id**, grid_domain_id |
| Required | grammar, bounded_status=HAS_REGION, n_coords, n_mask_units, shape fields, **claim_level** (from §4.3), action_state, production_forbidden |
| Notes | Non-null only; claim_level from geometry derivation — **not** grammar-wide G2=L1; family FK is the parameterized grammar; concrete cells via coordinates/policy_instances |

#### `null_records.csv`

| | |
|:--|:--|
| PK | `null_record_id` |
| FK | truth_contract_id, **feasibility_contract_id**, search_domain_id |
| Required | grammar, policy_family_definition_id **NULL**, null_reason, domain counts, **claim_level=L0**, action_state, production_forbidden |

#### `coordinates.csv` (**RB5 truth-level authority + RB9 registry binding**)

| | |
|:--|:--|
| PK | `coordinate_id` |
| FK | truth_contract_id, grid_domain_id, **threshold_registry_id**, **per-axis threshold_registry_entry_id** |
| Required | **canonical_cell_key**, thr indices in canonical axis order, registry entry FKs for every axis, optional raw `cell_id` / `combo_id` aliases; optional reconstructible `policy_instance_id` |
| **Forbidden on this table** | region_asset_id, productive_safe flags, capacity observations, dual margins, sequence incidence under a feasibility, feasibility_contract_id, thr_value as PK |
| Notes | Reusable across feasibility contracts without contradictory rows; threshold **values** recovered only via registry entries |

#### `mask_units.csv` (**RB5 truth-level authority**; filename may remain `region_masks.csv` if grain matches)

| | |
|:--|:--|
| PK | `mask_unit_id` |
| FK | truth_contract_id, grid_domain_id |
| Required | grid_id, mask_sha256; optional feasibility-invariant mask signature fields |
| **Forbidden on this table** | region_asset_ids_json as authority; feasibility-conditional PS capacity as identity; claim_level |
| Notes | If legacy filename `region_masks.csv` is kept, schema grain must still be truth-level only |

#### `region_coordinate_membership.csv` (**RB5 feasibility-bound realization authority**)

| | |
|:--|:--|
| PK | `(region_asset_id, coordinate_id)` |
| FK | **region_asset_id**, **coordinate_id**, **mask_unit_id**, (region implies feasibility) |
| Required | productive-safe membership flags as applicable, observed capacity fields, dual margins (or FK to margin table keyed by this PK), sequence incidence fields as applicable |
| **Authority** | sole authoritative region↔mask derivation source:  
  `region_mask_link == DISTINCT(region_asset_id, mask_unit_id) FROM region_coordinate_membership` |
| Notes | Filename may remain `region_coordinates.csv` **only if** grain and composite PK are this membership relation — not truth-level coordinate identity |

#### `region_asset_manifest.json` (**AUTHORITATIVE pack row — RB2**)

| | |
|:--|:--|
| Grain | **exactly one pack emission row** |
| PK | `pack_id` |
| FK | truth_contract_id, evidence_bundle_id, feasibility_contract_id, **candidate_universe_instance_id**, **threshold_registry_id** |
| Required | pack_id, producer_kind, producer_contract_version, schema_version, grammar_scope, maturity_declared, pack_claim_ceiling, action_state_default, production_forbidden, counts (`n_non_null_region_assets`, `n_null_records`, …), terminal_letter, composition_level |
| Invariants | sole authority for `pack_id`; every `pack_membership.pack_id` and `region_claim_contract.pack_id` **must** resolve here |
| Notes | Not optional/derived. If a multi-pack future needs a table form, `packs.jsonl` may supersede with the same PK/FK fields — until then this file is normative. |

#### `pack_membership.csv`

| | |
|:--|:--|
| PK | `(pack_id, content_kind, content_id)` |
| FK | **pack_id → region_asset_manifest.json** |
| Required | content_kind ∈ {region_asset, null_record, mask_unit, coordinate, membership, policy_family, policy_instance, predicate_definition, candidate_universe_contract, candidate_universe_instance, threshold_registry, …} |

#### `region_claim_contract.json`

| | |
|:--|:--|
| PK | pack_id (**FK → region_asset_manifest.json**) |
| Required | pack_claim_ceiling, maturity_declared (A0), action_states allowed/forbidden, production_forbidden, forbidden_promotions[], terminal_b, g7_status, identity_layer_policy, capacity_policy, sequence_policy, claim_ownership_policy (feasibility ⟂ claim_level), claim_level_derivation_policy (§4.3), composition_level_policy (observational only for this study), policy_equivalence_policy (family AST ≠ instance ≠ mask), realization_vs_content_policy (RB5), universe_instance_policy (RB8), threshold_registry_policy (RB9) |

### 11.2 Auxiliary / derived (optional)

| File | Authority source |
|:--|:--|
| `region_components.csv` | region_assets + membership |
| `region_mask_link.csv` | **derived from** region_coordinate_membership |
| `region_capacity.csv` | membership + mask units; dual distributions |
| `region_sequence_support.csv` | incidence expansion + region union/intersection |
| `region_margin.csv` | membership / T0 boundary_margin (keyed by region+coordinate) |
| `grammar_region_summary.csv` | aggregates |
| `semantic_definitions.*` / `policy_definitions.*` | only if synonym of `policy_family_definitions` with full family fields |
| `policy_instances.*` | family + registry bindings (RB9) |

### 11.3 Pack emission defaults (locked)

```text
grammar_scope: G1_G2_G3
producer_kind: grammar_atlas
action_state: observation_only
production_forbidden: true
maturity_declared: A0
pack_claim_ceiling: L1
composition_level: observational
not_scope: none
g7_roles: not_inferred
final_unknown_action: no_reject
```

### 11.4 Validation invariants

```text
every FK resolves to exactly one authority row
pack_id FK → region_asset_manifest.json only (not optional headers)

# RB2 / RB3
region_asset and null_record rows carry feasibility_contract_id;
  their IDs digest includes feasibility_contract_id
evidence_claim rows require feasibility_contract_id + evidence_bundle_id
feasibility_contract has no claim_level fields

# RB5
coordinate_id and mask_unit_id are feasibility-independent
coordinates / mask_units tables contain no feasibility-bound outcome fields
region_coordinate_membership PK = (region_asset_id, coordinate_id)
region_mask_link == DISTINCT(region_asset_id, mask_unit_id) FROM membership
one coordinate_id may appear under multiple feasibility-bound regions without conflict
content outcome IDs independent of pack_id and evidence_bundle_id

# RB6 + RB8
every pack / feasibility / same-universe compare resolves to one candidate_universe_instance row
every instance FK resolves to one candidate_universe_contract row
universe_membership_digest uses §6.5.2; no fabrication from dataset name
source_event_table_sha256 remains evidence-only raw seal
predicate missing/unknown/comparator/endpoint behavior is machine-reconstructible
unknown never defaults to reject for this study
cross-universe-instance composition without transport is invalid

# RB7 + RB9
every non-null region and every search_domain_member resolves to one policy_family_definition
policy_family digests parameterized canonical AST (+ grammar/truth semantics/roles/universe instance)
every coordinate axis resolves to one threshold_registry entry
threshold_value_repr reconstructible from registry without guessing floats
policy_instance_id = family + registry + ordered thr bindings (or equivalent composite)
observed_mask_hash attaches to policy instance / coordinate — not family alone
role provenance survives commutative canonicalize
no G7 necessary/support roles inferred for Q4.5
composition_level=observational for this study pack

# RB1 / geometry claims
claim_level derived per §4.3:
  G1: 1×L0; G2: 6×L0 isolated + 19×L1 multi; G3 null L0; pack ceiling L1
both capacity distributions present for multi-member regions
no additive region capacity metric
sequence union and intersection both present when multi-member
G3: members=40, n_non_null=0, n_null_records≥1, null.policy_family_definition_id IS NULL

# RB4 / truth
pairwise truth rows digest only after §6.3 leaf canonicalize
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
grammar G2 ⇒ every G2 object L1
A1 packaging ⇒ transferable or actionable
missing files ⇒ NULL_RESULT
mask equality ⇒ semantic / G7 equivalence
observed_mask equality ⇒ policy identity or logical equivalence
untyped observational role ⇒ G7 necessary/support or sufficient_reject authorization
observational composition ⇒ single-step or closed-loop safety
unknown ⇒ reject
cross-universe composition without transport
generator-contract equality ⇒ same sealed universe instance
source_event_table_sha256 ⇒ universe_membership_digest
policy family ⇒ concrete threshold-executable policy
thr_index without registry ⇒ reconstructible thr_value
```

---

## 13. R1 conversion boundary

### Authorized now (chat-side R0-B accept + R1 gate)

Deterministic packaging only:

```text
inputs:  sealed Q4.5 full atlases + T0-B-R1 pack + this contract (E1 applied)
         + boolean_composition_semantics_contract
         + sealed threshold_registry.json (+ SHA256SUMS)
         + candidate-level rows for universe_membership_digest
outputs: authority + emission tables in §11 under declared out root
         A0 observation-only pack candidate (not A1)
process: no evaluator modification/rerun; no new threshold search;
         no new research geometry claims; no self-promotion to A1
```

### Remains blocked even after successful R1 emission until further gates

```text
A1 research maturity acceptance (separate chat-side gate)
L2+ claim levels
event-union capacity
G4–G7 / G7 roles
NOT/complement reject policies
single-step / closed-loop composition claims
LOO/shadow/hook/preset/production
evidence_ledger promotion
```

### R1 engineering requirements

Packaging must materialize the authorities (universe contract+instance with membership digest or explicit BLOCKED_BY_ARTIFACT, threshold_registry+entries, predicate_definitions, policy_family_definitions, policy_instances, coordinates, mask_units, membership). Runtime full atlases must remain hash-sealed for lattice membership. Two clean converter runs must yield identical authority content and IDs. No new threshold search.

---

## 14. Q4.5 instance snapshot (normative freeze, not new results)

```text
truth: stage2_q45_atlas_v4 · substrate stage1_baudit_d_online
universe: contract online_hook_eligible + sealed instance membership digest (RB8 model B)
  source_event_table_sha256 = cfca3818… (evidence seal only; ≠ membership digest)
threshold_registry: sealed Q4.5 registry (sha d3e3197f…; 870 single + 210 pairwise)
cohort: n_gt_exposed=64 · n_fp_exposed=23 · unresolved firewall on
PS: 154 = 1 G1 + 153 G2 + 0 G3
components: 26 = 1 G1 + 25 G2 (6 isolated + 19 multi) · productive mask units: 34
radius≥1: 0/154 · terminal B
claims: G1 1×L0 · G2 6×L0 + 19×L1 · G3 1×L0 · pack ceiling L1 · maturity A0
composition_level: observational · not_scope: none · U→no_reject · no G7 roles
RB3: region/null IDs bind feasibility_contract_id (model A)
RB2: pack authority = region_asset_manifest.json
RB4: pairwise a/b full-field canonicalize before truth digest
RB5: coordinates/mask_units truth-level; membership feasibility-bound
RB6: predicate_definitions + universe authorities required
RB7: policy_family canonical AST ≠ observed_mask_hash
RB8: universe contract ≠ instance; membership digest algorithm locked
RB9: threshold_registry authority; policy_instance = family + thr bindings
```

---

## 15. Open decisions (non-blocking)

| ID | Topic | Default if silent-accept |
|:--|:--|:--|
| O1 | Display truncation length for IDs | 32 hex + full digest column |
| O2 | Whether `evidence_claims` is csv or jsonl | jsonl |
| O3 | Robust floor definition | min member capacity |
| O4 | Optional L0 geometry tags on isolated points | tags ≠ L1 |
| O5 | Future multi-pack table `packs.jsonl` | only if multi-pack needed; same fields as manifest |
| O6 | Keep filename `region_coordinates.csv` vs rename to `region_coordinate_membership.csv` | either OK if grain = membership composite PK |
| O7 | Keep filename `region_masks.csv` vs `mask_units.csv` | either OK if grain = truth-level mask |
| O8 | Synonym `semantic_definitions` vs rename to `policy_definitions` | synonym allowed only with full RB7 fields |
| O9 | Exact Q4.5 role token string | `untyped_observation` |
| O10 | RB8 model A (instance-only PK) vs B (contract+instance) | **B locked** for R0-B-R3 |
| O11 | Store policy_instances table vs derive from (family, coordinate, registry) | either OK if identity equal |
| O12 | Q4.5 candidate primary-key column list | take from sealed event schema / study contract; if rows unavailable → BLOCKED_BY_ARTIFACT |

No open decision may: re-merge claim_level into feasibility_contract_id; drop `grid_domains`; reintroduce grammar-wide G2=L1; drop pack authority; unbind null/region outcomes from feasibility; skip pairwise leaf canonicalize; put feasibility-bound fields on coordinate/mask content IDs; omit candidate-universe contract/instance or predicate authorities; collapse universe contract into instance without membership digest; use source_event_table_sha256 as membership digest; omit threshold_registry; treat policy family as concrete threshold policy; digest observed masks into family identity; invent G7 roles; map unknown to reject; promote observational composition to intervention safety; run new threshold search.

---

## 16. Explicit non-authorization / residual gates

```text
R0-B contract: ACCEPTED (chat-side; not agent self-accept)
R1: authorized for deterministic A0 pack candidate only
R1 pack: NOT self-accepted as A1
no evaluator rerun or modification
no new threshold / policy / geometry search
no A0→A1 promotion by the converter
R1 conversion did not itself grant research acceptance or merge authorization
engineering delivery and review now proceed through the active implementation PR
evidence_ledger unchanged
production/presets unchanged
no G4–G7 implementation
no research verdict self-acceptance
```
