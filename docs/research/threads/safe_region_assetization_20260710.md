---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A engineering/preflight packet delivered at `762adf9a`, but chat-side review is **CHANGES_REQUESTED**. Current sole task is **R0-A-R1 contract correction**. R0-B, R1 asset generation, grammar extension, transfer, intervention, production, and ledger promotion remain unauthorized.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-A-R1 contract correction** |
| Existing G1–G3 T0/C0 result | **A0 descriptive baseline**; terminal B retained |
| R0-A packet | **DELIVERED** at `762adf9a`; provenance/grain/firewall portions pass |
| Chat-side review | **CHANGES_REQUESTED** — four blocking contract issues |
| R0-B final contract | **NOT AUTHORIZED** |
| R1 G1–G3 asset conversion | **NOT AUTHORIZED** |
| R2 grammar asset extension | **CONDITIONAL** after accepted R0/R1 and an asset-increment hypothesis |
| R3 transfer qualification | **CONDITIONAL** on selected A1 assets |
| R4 intervention qualification | **CONDITIONAL** on selected A2 assets |
| Current maturity | **A0 retained** |
| Occ-exit conditional modeling | **PARKED** future RegionAsset producer/consumer |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

The research object is a reusable `RegionAsset`, not a rule and not a coverage-table cell:

```text
producer semantics
→ parameter coordinates
→ stable region identity
→ geometry
→ productive capacity
→ applicability geometry
→ transfer qualification
→ action contract
```

Hard separation remains:

```text
artifact generated
≠ engineering ready
≠ asset maturity accepted
≠ research conclusion accepted
≠ intervention qualified
≠ production approved
```

No assetization step may manufacture causal, transfer, or online evidence absent from the source artifacts.

## Read first

1. [R0-A preflight packet](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
2. [Closed A0/T0 thread](composition_grammar_safe_region.md)
3. [T0 artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
4. [T0 region interpretation](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)
5. [Superseded grammar-coverage design](composition_grammar_coverage_program_20260710.md)

## Accepted A0 baseline

```text
154 productive-safe coordinates = 1 G1 + 153 G2 + 0 G3
142 single-sequence · 12 multi-sequence
34 productive per-grid mask units
26 coordinate components
full_neighborhood_safe_radius >= 1: 0/154
G3 Hard OR: registered-lattice NULL_RESULT
terminal B: isolated_safe_points_only
```

This baseline is descriptive only. It is not transferable, actionable, or production-ready.

## Program stages

```text
R0 Region Asset Contract
  R0-A preflight
  R0-A-R1 correction  ← current
  R0-B final accepted contract

R1 G1–G3 deterministic asset conversion
R2 grammar-by-grammar asset extension
R3 transfer qualification toward A2
R4 intervention qualification toward A3
A4 separate production approval
```

## R0-A review result

### Pass

The packet correctly established:

- sealed Q4.5/T0 provenance and live-tree drift handling;
- distinct asset-set, component, per-grid-mask, coordinate, and null grains;
- ordinal component IDs are invalid;
- per-grid mask identity is primary;
- semantic equivalence is not mask equivalence;
- G3 requires a first-class null result;
- A0–A4 maturity and action firewalls;
- no evaluator rerun is required for the currently derivable A1 core.

### Blocking corrections

#### CR1 — Stable region identity must not depend on whole-pack identity

Current proposal scopes `region_asset_id`, `mask_unit_id`, and `coordinate_id` through an `asset_set_id` that includes combined grammar scope, output contract version, and broad input hash maps. This can change existing G1–G3 IDs when an unrelated grammar, schema/materialization version, or derived file is added.

R0-A-R1 must separate at least:

```text
truth/evidence context identity
asset-pack/materialization identity
stable region content identity
pack-specific evidence record identity (if needed)
```

Stable region/mask/coordinate identity must depend on its truth-bearing context plus local semantic/content identity, not unrelated pack membership or component order. Explicitly test added grammar, schema migration, repackaging, and equivalent regeneration.

#### CR2 — Component capacity is not additive across alternative parameter coordinates

A connected component is a set of alternative threshold coordinates. Member masks are not simultaneous actions. Therefore:

```text
sum(unique mask_n_neg within component)
```

is not a valid component capacity and may double-count the same event across different masks.

R0-A-R1 must:

- retract D10's additive recommendation;
- define component capacity as a declared distribution over member coordinate/mask capacities (for example min/max/median/quantiles and robust floor), not a sum;
- treat union-of-captured-events as a separate optional metric;
- mark event-union capacity `BLOCKED_BY_ARTIFACT` unless event membership/bitsets or an explicit sealed reconstruction contract are available;
- keep plateau width from multiplying capacity.

#### CR3 — Sequence union is not region applicability

Taking the union of productive sequences across member coordinates only means that some parameter choice works somewhere. It does not establish one stable component-wide applicability region.

R0-A-R1 must preserve at least:

```text
per-coordinate / per-mask sequence incidence
component sequence union
component sequence intersection
min/max sequence-support count across member choices
sequence dominance / island diagnostics
```

Union and intersection must be named explicitly. Neither may be promoted to A2 transfer/applicability without a representative/transport contract.

#### CR4 — G3 grammar-level null and concrete semantic definitions are different grains

G3's null result spans 40 concrete operand-pair grids. A single grammar-level null record cannot point to one concrete `semantic_definition_id` pretending that “pairwise atom family” is one operator tree.

R0-A-R1 must choose and document one valid structure, such as:

```text
one grammar/search-domain null record
  + nullable concrete semantic_definition_id
  + search_domain_id / semantic_family_id
  + optional per-grid null summaries
```

or an explicitly justified per-grid null representation. Also disambiguate:

```text
n_non_null_region_assets = 0
n_null_records = 1 (or declared per-grid count)
```

Missing files remain `BLOCKED_BY_ARTIFACT`, never `NULL_RESULT`.

#### CR5 — Region↔mask M:N relation must have one authoritative representation

Do not use an embedded `region_asset_ids_json` list as the authoritative relation. Use coordinate foreign keys as the derivation source or propose an explicit normalized link table. The same per-grid mask unit may not be assumed to belong to only one component without proof.

## Current step — R0-A-R1

Revise the existing preflight note in place:

```text
docs/modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md
```

Required changes:

1. correct the derivability matrix for component capacity and event-union capacity;
2. replace D1/D2/D9 identity design with a context-vs-pack separation that preserves local IDs under unrelated additions;
3. replace D10 with non-additive region-capacity distribution semantics;
4. replace D11 with explicit union/intersection/incidence semantics;
5. repair G3 null semantic grain and count names;
6. make region↔mask linking authoritative and normalized/derivable;
7. update proposed R1 schemas, invariants, readiness statement, and self-audit accordingly;
8. add explicit synthetic identity/aggregation examples demonstrating the corrected behavior.

## Acceptance for R0-A-R1

- existing region IDs do not change merely because an unrelated grammar is added to another pack;
- schema/materialization version changes do not silently redefine stable region content identity;
- component capacity never sums alternative mask capacities as if they were disjoint actions;
- event-union capacity is separately named and blocked unless membership is available;
- sequence union and intersection are both preserved and bounded;
- G3 null has a valid grammar/search-domain identity without fake concrete operands;
- null record counts and non-null region counts are unambiguous;
- region↔mask M:N relation has one authoritative representation;
- all prior provenance, per-grid-mask, semantic-role, maturity, and firewall gates remain intact;
- R0-B and R1 remain unauthorized.

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset pack files;
- implement G4–G7;
- implement generic runtime/framework code;
- infer G7 roles;
- add LOO, shadow, hook, preset, or production work;
- change terminal B or maturity A0;
- promote to evidence ledger;
- open an R1 implementation PR.

## History

- 2026-07-10: T0-A/B/R1 accepted; PR #94 merged; G1–G3 retained as A0 baseline.
- 2026-07-10: Safe-Region Assetization Program opened; R0-A authorized.
- 2026-07-10: R0-A delivered at `762adf9a`; artifact hashes and document checks passed.
- 2026-07-10: chat-side review requested R0-A-R1 for identity, capacity, applicability, null-grain, and relation corrections.
