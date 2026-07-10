---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** Convert safe-region observations into stable, comparable, transferable, and action-bounded research assets. Current sole task is **R0-A completed — awaiting chat-side review**. No asset generation, grammar extension, LOO, shadow policy, hook, production change, or ledger promotion is authorized yet.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-A completed — awaiting chat-side review** |
| Existing G1–G3 T0/C0 result | **A0 descriptive baseline**; A1-derivable evidence exists, but no accepted reusable asset object |
| R0-A preflight note | **DELIVERED** · not self-accepted · [safe_region_r0_asset_contract_preflight_20260710.md](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md) |
| R0 asset contract | **PREFLIGHT COMPLETE; R0-B NOT AUTHORIZED** |
| R1 G1–G3 asset conversion | **NOT AUTHORIZED** (recommended as deterministic conversion after R0-B only) |
| R2 grammar asset extension | **CONDITIONAL** on accepted R0/R1 and declared asset-increment hypothesis |
| R3 transfer qualification | **CONDITIONAL** on A1 assets worth transferring |
| R4 intervention qualification | **CONDITIONAL** on A2 assets |
| A4 production approval | **outside automatic program flow; separate evidence acceptance** |
| Occ-exit conditional modeling | **PARKED adapter / future producer-consumer family** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |
| Maturity | **A0 retained** (no A0→A1 promotion) |

## Program objective

The research object is not a Boolean rule or a coverage-table cell. It is a reusable region object:

```text
signal / grammar semantics
→ parameterized coordinates
→ region identity
→ geometry
→ productive capacity
→ applicability
→ transfer qualification
→ action contract
```

The core question is:

> How can an observed safe/productive set be packaged as a research asset with stable identity, explicit semantics, geometry, capacity, applicability, transfer evidence, and action boundaries—without silently promoting it into a policy?

Composition grammars are **region producers**. Occ-exit, association, relink, and other conditional interventions may later be producers or consumers of the same asset language.

## Why this supersedes grammar-coverage completion

The grammar-coverage map remains useful as a producer inventory, but it is not the main research object.

A new grammar is justified only when it has a bounded hypothesis that it may add at least one asset increment:

```text
new unique productive masks
more full-neighborhood thickness
more multi-sequence support
higher productive capacity
better transfer retention
clearer intervention semantics
```

A grammar with no asset increment may close early as `NULL_RESULT`, `NOT_JUSTIFIED`, `BLOCKED_BY_PROVENANCE`, or `NOT_APPLICABLE`. It does not need to reach online merely to fill a table.

The former [Composition Grammar Coverage Completion Program](composition_grammar_coverage_program_20260710.md) is retained as a superseded design record. Its useful grammar/gate content is absorbed here under R2–R4.

## Fixed boundaries

- Frozen current signal family for the first asset pack: `score_m_bridge`, `abs_log_h`, `dist_h`, `abs_ratio_m1`, `resid_mean`.
- Frozen Stage 2 primary decision cohort, label contract, and unresolved firewall.
- G1–G3 truth remains limited to registered Q4.5 lattices; no continuous-domain overclaim.
- Existing T0 units remain canonical starting points:
  - coordinate = registered threshold coordinate within one grid;
  - primary mask unit = `mask_sha256` within one registered grid;
  - component = grammar-specific lattice adjacency within one grid.
- Operand roles must survive serialization; equal event masks do not imply equal research semantics.
- Dual area and dual margin must remain separate.
- Engineering readiness, asset maturity, research acceptance, intervention qualification, and production approval are separate gates.
- Assetization must not manufacture causal or transfer evidence that the source artifacts do not contain.
- Research progress is measured by accepted asset contracts and bounded maturity transitions, not schema/tool completion.

## Region asset object model — contract target

R0 must decide the exact grain. The starting model is four identity layers plus a pack manifest.

### 1. Asset-set identity

One immutable study/producer context:

```text
asset_set_id
producer_kind              # grammar_atlas | occurrence_condition | other future adapter
producer_contract_version
grammar_or_condition_family
signal_family_version
substrate_id
cohort_id
label_contract_id
unresolved_policy_id
lattice_contract_id
evaluator_version
input_artifact_hashes
```

`asset_set_id` must change when any truth-bearing identity above changes.

### 2. Region/component identity

The primary reusable region object is expected to be a connected component or an explicitly declared null asset, not an arbitrary row and not an ordinal label.

Candidate fields:

```text
region_asset_id
asset_set_id
semantic_definition_id
component_coordinate_digest
component_mask_digest
maturity_level
bounded_status
human_alias
```

Hard rule:

```text
component_03 is not a stable identity
```

Component IDs must derive from canonical semantic/context identity plus canonical content, not enumeration order. Human-readable threshold expressions are aliases only.

### 3. Per-grid mask-unit identity

```text
mask_unit_id
region_asset_id
grid_id
mask_sha256
coordinate_membership_digest
```

The same global mask string in two registered grids remains two primary mask units. Global mask-string collapse is diagnostic only unless a later contract explicitly changes the unit.

### 4. Coordinate identity

```text
coordinate_id
asset_set_id
grid_id
canonical_parameter_coordinate
```

Coordinates must preserve the grammar-specific parameter system: absolute threshold, quantile index, envelope-relative coordinate, rank/CDF coordinate, or another declared transport system. These systems must not be silently mixed.

### 5. Null-region asset

A null result is a first-class asset record, not absence of output:

```text
bounded_status: NULL_RESULT
region_asset_count: 0
declared_search_domain
observed_safe_count
productive_safe_count
null_reason
claim_boundary
```

G3 Hard OR is the first expected null-asset conversion candidate.

## Required asset dimensions

### Identity and provenance

- stable IDs and human aliases;
- producer/grammar semantic contract version;
- signal family, substrate, cohort, labels, unresolved policy;
- lattice/coordinate contract;
- evaluator and source-artifact hashes;
- generation script/version and deterministic rerun identity.

### Semantic definition

Preserve:

```text
operator tree
operand identities
operand roles
parameter coordinates
canonicalization / symmetry rules
```

Examples:

```text
Singleton: resid_mean >= theta
Pairwise AND: abs_log_h >= theta_1 AND dist_h >= theta_2
Necessary + support:
  necessary_violation(theta_N) AND productive_support(theta_P)
```

Two assets may currently emit the same mask but remain semantically distinct when operand roles or transport coordinates differ.

### Region geometry

- safe and productive coordinates;
- per-grid unique masks;
- connected components and adjacency contract;
- active-axis count and axis widths;
- plateau/duplicate structure;
- `nearest_unsafe_distance`;
- `full_neighborhood_safe_radius`;
- edge-censoring metadata.

### Productive capacity

- negative mass captured;
- capacity per coordinate, mask unit, and component;
- capacity distribution and concentration;
- minimum positive-sequence capacity;
- productive floor contract;
- explicit distinction between coordinate-duplicated and unique-mask capacity.

### Applicability geometry

- supported/productive sequences;
- sequence-specific islands;
- multi-sequence intersection/union;
- sequence dominance;
- scene/condition support only when grounded by a separate semantic owner;
- abstention/unknown surface;
- unresolved contamination.

Sequence names may be evidence slices or comparators. They are not automatically valid policy features or grammar semantics.

### Transfer contract

Each transport is a separate qualification and claim:

```text
fixed absolute threshold
train quantile
rank / CDF coordinate
GT-envelope-relative coordinate
component retention
nested LOSO
online substrate support
```

Current exact-absolute-clause LOO is one narrow transfer observation, not general region transferability.

### Action and claim contract

Allowed action states include:

```text
observation_only
shadow_decision
condition_model_candidate
offline_filter_candidate
default_off_intervention_candidate
production_forbidden
```

The asset must separately state maximum research claim and forbidden promotions. Seeing `safe_region` must never be sufficient to create a gate.

## Asset maturity model

### A0 — Descriptive atlas

Has point-level enumeration or a bounded null result, but no accepted reusable region object.

```text
can describe
cannot be consumed as a stable asset
```

Current G1–G3 T0/C0 truth is classified here. Much of the evidence needed for A1 is derivable, but identity/schema/claim packaging has not yet been accepted.

### A1 — Region asset

Requires accepted:

- stable identity and provenance;
- semantic definition with operand roles;
- coordinate/mask/component relations;
- dual area and topology;
- capacity and cross-sequence support;
- null-asset representation when applicable;
- claim/action firewall.

```text
can compare, rank, diff, and reproduce
not yet transferable
```

### A2 — Validated applicability asset

Requires declared qualification for:

- per-sequence contraction;
- one or more explicit transport modes;
- region-level LOO or equivalent transfer evidence;
- held-out harm bound;
- productive floor;
- unresolved firewall.

```text
may support condition modeling or shadow-policy evaluation
not yet an intervention asset
```

### A3 — Intervention asset

Requires:

- action-time observable matching inputs;
- frozen representative or matching contract;
- default-off action path;
- online substrate support;
- control/treatment evidence;
- monitoring, drift, abstention, and rollback contracts.

```text
may enter application validation
still not a production default
```

### A4 — Production-approved policy

Requires a separate formal promotion decision, production evidence acceptance, operational monitoring, rollback ownership, and preset/default governance.

A4 is not automatically reached by completing R4.

## Program stages

### R0 — Region Asset Contract

Define the minimal trustworthy schema, identities, maturity gates, claim firewall, and deterministic materialization contract.

R0 is split:

```text
R0-A contract/artifact preflight
→ chat-side review
→ R0-B accepted RegionAsset contract
```

No asset files are generated before R0-B acceptance.

### R1 — G1–G3 Asset Conversion

Convert the accepted T0/C0 outputs into the first A1-compatible pack without rerunning or changing the evaluator unless R0 proves a required field is not derivable.

Target outputs after authorization:

```text
region_asset_manifest.json
grammar_region_summary.csv
region_assets.csv
region_components.csv
region_masks.csv
region_coordinates.csv
region_capacity.csv
region_sequence_support.csv
region_margin.csv
region_claim_contract.json
```

R1 expected coverage:

- G1 Singleton: non-null A1 asset(s), likely isolated;
- G2 Pairwise AND: component/mask assets preserving thin/edge geometry;
- G3 Hard OR: first-class null asset.

R1 is not authorized by R0-A.

### R2 — Grammar Asset Extension

Extend G4–G7 only grammar-by-grammar and only with an explicit asset-increment hypothesis.

Default research order remains advisory:

```text
G7 necessary-violation + support
→ G4 atom-count
→ G6 extreme / consensus
→ G5 family-count
```

Required first questions for each grammar:

```text
does the semantic contract exist?
does it add new per-grid productive masks?
does it add capacity, thickness, or multi-sequence support?
does it clarify intervention/transfer semantics?
```

No giant G4–G7 evaluator PR. Stop when no asset increment appears.

### R3 — Transfer Qualification

Promote selected A1 assets toward A2 through explicit, non-collapsed transport studies:

- per-sequence contraction;
- sequence-island versus shared-region analysis;
- declared region selection unit;
- absolute / quantile / rank-CDF / envelope-relative transport as separate claims;
- held-out harm, productive mass, and geometry retention.

No best-clause headline may substitute for region transfer.

### R4 — Intervention Qualification

Promote selected A2 assets toward A3:

- observation-only online matching first when appropriate;
- shadow decision and expected action logging;
- action-time observable contract;
- frozen representative/default-off A/B;
- support/boundary/capacity drift monitoring;
- abstention and rollback.

Occ-exit episode modeling can later reuse this stage by emitting or consuming RegionAssets rather than inventing a separate applicability abstraction.

## Current step — R0-A completed — awaiting chat-side review

**Working branch:** `research/composition-grammar-coverage-program` (legacy branch name retained; program identity is this thread).

### Deliverable

```text
docs/modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md
```

### Preflight outcome (summary; not acceptance)

```text
Runtime Q4.5 full atlases PRESENT and hash-match accepted manifest.
T0-B-R1 committed pack hashes verified.
Terminal B unchanged; maturity remains A0.

R1 recommendation: deterministic A0→A1 conversion is FEASIBLE
  after R0-B accepts identity/schema decisions D1–D12
  and atlases remain sealed. No evaluator rerun required for A1 core.

Hard findings:
  - T0 ::compN ordinals are not stable region_asset_id
  - global mask_sha256 is not primary (multi-grid mask strings observed)
  - G3 requires first-class null asset (not missing files)
  - live evaluator/script tree SHAs drift from study-recorded SHAs
    → pin artifact hashes; do not re-hash live tree for identity
  - transfer/intervention fields remain A2+/A3 (NOT_APPLICABLE for A1)
  - G7 roles remain unresolved
```

### Blocking decisions before R0-B

Primary contract decisions D1–D12 in the preflight note, especially:

| ID | Topic |
|:--|:--|
| D1 | multi-grammar pack vs per-grammar asset sets |
| D4 | AND/OR operand order canonicalization |
| D5 | primary region grain = connected PS component |
| D7 | G3 null packaging |
| D8 | per-grid mask unit primary key |
| D10 | component capacity aggregation without plateau double-count |
| D12 | default `observation_only` + `production_forbidden` |

### Not authorized

```text
R0-A research acceptance (chat-side only)
R0-B final contract
R1 asset file generation
A0→A1 maturity promotion
evaluator rerun / modification
G4–G7, LOO, shadow, hooks, presets, ledger promotion
```

### Must not (still in force)

- Modify or rerun the Q4.5 evaluator.
- Generate R1 asset files.
- Invent missing semantic roles, causal labels, transfer evidence, or online support.
- Use global `mask_sha256` as the primary asset key.
- Use component ordinals as stable identity.
- Collapse coordinates, masks, components, and assets into one row type.
- Treat A1 as transferable or actionable.
- Implement G4–G7 enumeration.
- Implement generic RegionAsset runtime/framework code.
- Add hooks, flags, presets, or production behavior.
- Promote evidence to the ledger.
- Open an R1 implementation PR.
- Self-accept R0-A or authorize R0-B/R1 from this thread alone.

## Review sequence

```text
T0/C0 accepted A0 baseline
→ R0-A asset-contract preflight  ← completed; awaiting chat-side review
→ chat-side review
→ R0-B final RegionAsset contract
→ chat-side review
→ R1 G1–G3 deterministic asset conversion
→ engineering review
→ A1 research acceptance
→ authorize at most one R2 grammar asset hypothesis
→ later R3 transfer / R4 intervention only through maturity gates
```

## Occ-exit disposition

Occ-exit remains parked, but its future relationship is now explicit:

```text
composition grammar / threshold atlas
  = one RegionAsset producer

occ-exit episode ledger / condition model
  = future RegionAsset producer and intervention consumer
```

The accepted #55 evidence remains unchanged: global audit harmful, one local enable candidate, Cheb-GR log-only, no gate or production promotion.

## History

- 2026-07-10: G1–G3 T0-A/B/R1 accepted; PR #94 merged as `acd8e30e`.
- 2026-07-10: Composition Grammar Coverage Completion design opened, mapping T0/C0 and proposing C1–C6.
- 2026-07-10: owner reframed the missing layer as reusable safe-region assets rather than grammar-table completion.
- 2026-07-10: Safe-Region Assetization Program supersedes coverage completion; R0-A becomes semantic sole active; no asset generation authorized.
- 2026-07-10: R0-A preflight note delivered; hashes verified; R1 scoped as conditional deterministic conversion; **awaiting chat-side review** (not accepted; R0-B/R1 unauthorized; A0 retained).
