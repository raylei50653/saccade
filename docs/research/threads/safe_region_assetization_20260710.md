---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A is **ACCEPTED**. RB1–RB4 remain **PASS**. R0-B-R2 delivered a typed realization integration draft on the RegionAsset contract (RB5–RB7). **Chat-side re-review required** — not self-accepted. R1 asset generation, A0→A1, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-B-R2 delivered — awaiting chat-side re-review** |
| R0-A research review | **ACCEPTED** |
| CR1–CR9 | **PASS** (retain) |
| R0-B draft | `75dec59a` CHANGES_REQUESTED |
| R0-B-R1 | `e02a5367` — **RB1–RB4 PASS** (retain) |
| R0-B-R2 | **delivered** — RB5–RB7 typed realization integration; **not self-accepted** |
| Boolean semantics patch | **NORMATIVE** — [boolean_composition_semantics_contract.md](../eval/boolean_composition_semantics_contract.md) |
| R1 | **NOT AUTHORIZED** |
| Maturity | **A0 retained** |
| Claims | G1 **1×L0**; G2 **6×L0 isolated + 19×L1 multi**; G3 **1×L0**; pack ceiling **L1** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

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
```

Hard separation remains:

```text
artifact generated
≠ engineering ready
≠ asset maturity accepted (A0–A4)
≠ statistical claim level (L0–L6)
≠ research conclusion accepted
≠ intervention qualified
≠ production approved
```

## Read first

1. [R0-B / R0-B-R1 RegionAsset contract](../eval/safe_region_asset_contract.md)
2. [Boolean Composition Semantics Contract](../eval/boolean_composition_semantics_contract.md)
3. [Mathematical framework](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
4. [Accepted R0-A preflight](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
5. [T0 component geometry evidence](../../modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/component_geometry.csv)

## Accepted A0 baseline

```text
154 PS = 1 G1 + 153 G2 + 0 G3
26 components = 1 G1 + 25 G2 (6 isolated + 19 multi)
radius≥1: 0/154 · terminal B · G3 domain null
claims: G1 1×L0 · G2 6×L0 + 19×L1 · G3 1×L0 · pack ceiling L1
```

This is finite-sample, searched in-sample, registered-lattice evidence. It is not population safety, held-out retention, intervention evidence, or a production candidate.

## Program stages

```text
R0-A preflight                         # ACCEPTED
R0-B draft                             # CHANGES_REQUESTED
R0-B-R1 RB1–RB4                        # PASS (retain)
R0-B-R2 typed realization integration  # delivered; chat-side re-review
chat-side final contract review

R1 G1–G3 deterministic conversion     # not authorized
R2–R4 conditional maturity stages
A4/L6 separate production approval
```

## R0-B-R1 integrated review

### Pass — RB1–RB4

- **RB1:** claim level is geometry-derived: G1 1×L0; G2 6×L0 isolated + 19×L1 multi; G3 1×L0; pack ceiling L1 only as aggregate.
- **RB2:** `region_asset_manifest.json` is the authoritative one-row `pack_id` owner.
- **RB3:** model A binds `region_asset_id`, `null_record_id`, and evidence claims to `feasibility_contract_id`; masks and coordinate identities stay feasibility-independent.
- **RB4:** pairwise leaves swap all `(atom, feature, direction, thr_index)` fields before truth digest, axes, coordinate keys, and membership geometry.

These corrections are accepted and must not regress.

## Blocking corrections — R0-B-R2

### RB5 — Separate stable coordinate/mask identity from feasibility-bound realization

The current contract keeps `coordinate_id` and `mask_unit_id` feasibility-independent, but `region_coordinates.csv` uses `coordinate_id` as its sole PK while carrying `region_asset_id`, productive-safe flags, capacity, and margins. The same coordinate can have different membership/outcome fields under another feasibility contract, so one feasibility-independent PK cannot authoritatively own those changing fields.

R0-B-R2 must normalize the relation, for example:

```text
coordinates.csv
  PK coordinate_id
  truth-level fields only: truth_contract_id, grid_domain_id,
  canonical_cell_key, threshold indices / aliases

mask_units.csv (or renamed region_masks authority)
  PK mask_unit_id
  truth-level mask fields only: truth_contract_id, grid_domain_id,
  mask_sha256 and invariant mask facts

region_coordinate_membership.csv
  PK (region_asset_id, coordinate_id)
  FK region_asset_id, coordinate_id, mask_unit_id
  feasibility-bound fields: productive-safe membership,
  capacity observation, dual margins, sequence incidence as applicable
```

The filename may remain `region_coordinates.csv`, but its grain and composite PK must be the feasibility-bound membership relation, and a separate coordinate authority must exist.

Requirements:

- no feasibility-independent content ID may own fields that can change with `epsilon`, `g_min`, loss, universe, or denominators;
- region↔mask authority remains the projection of the feasibility-bound membership rows;
- mask/coordinate content identities remain reusable without creating contradictory rows across packs;
- any mask field whose value is conditional on productive-safe membership must move to a realization/membership or aggregate table.

### RB6 — Add an authoritative candidate-universe and predicate-edge contract

The Boolean semantics contract requires the observation space and partial predicate behavior to be reconstructible. A bare string such as `online_hook_eligible` is insufficient as the only machine authority.

R0-B-R2 must define an authority such as:

```text
candidate_universe.json|jsonl
  universe_id, universe_hash
  substrate_id, hook_id, candidate_builder_id/version
  prefilter/eligibility contract
  candidate key schema
  label/exposure owner
  time/frame range
  pre-decision state-snapshot contract
```

It must also define predicate/atom semantics, either in a new authority or by extending a uniquely named existing authority:

```text
predicate_id
signal_identity / unit
predicate_domain / codomain
unknown_value_policy and final_unknown_action
comparator, endpoint, tie, NaN, ±Inf, missing policy
quantile method, tolerance, clipping domain
```

For the current Q4.5 pack, freeze only what is supported:

```text
composition_level = observational
same declared universe and pre-decision state
reject-only G1/G2/G3 AND/OR
no NOT/complement authorization
unknown never maps to reject
no cross-universe composition without transport
```

Do not infer total two-valued predicates unless the source contract establishes them; otherwise serialize the explicit three-valued/fail-safe behavior.

### RB7 — Make executable Boolean policy identity first-class

`semantic_definition_id = operator + leaves + lattice` is not sufficient under the normative Boolean contract. The final machine contract must preserve executable semantics independently from the observed mask.

Define a policy/semantic authority that owns at least:

```text
grammar_version
truth_semantics_version
canonical_policy_ast
operator precedence and grammar bounds
predicate references / threshold-edge contract
operand and subtree roles
universe requirement
composition level
NOT/complement metadata when applicable
canonical_policy_ast_hash
```

Requirements:

- `semantic_definition_id` or a dedicated `policy_definition_id` must digest the canonical executable semantics, not `observed_mask_hash` or evidence outcomes;
- observed-mask equivalence, syntactic identity, and logical-equivalence status remain separate fields;
- role metadata must not disappear during commutative sorting or operand canonicalization;
- current G1–G3 serialization must not invent G7 necessary/support roles;
- if current search operands are not role-qualified, declare the bounded untyped/observational role rather than silently granting reject authorization;
- RegionAsset / coordinate realization rows must resolve to the exact policy semantics that generated them.

## Current step — R0-B-R2 (delivered; re-review)

Revised in place:

```text
docs/research/eval/safe_region_asset_contract.md
```

### Delivered corrections

1. **RB5:** truth-level `coordinates.csv` + `mask_units.csv` (feasibility-independent) vs feasibility-bound `region_coordinate_membership.csv` with PK `(region_asset_id, coordinate_id)`; region↔mask projects from membership.
2. **RB6:** `candidate_universe.json|jsonl` and `predicate_definitions` authorities; Q4.5 freeze observational / U→no_reject / no cross-universe without transport.
3. **RB7:** `policy_definitions` with canonical AST hash; AST ≠ observed_mask ≠ logical equivalence; roles survive commutative sort; Q4.5 roles = `untyped_observation`; no G7 inference.
4. Boolean semantics contract cited as normative parent; PK/FK map, invariants, Q4.5 freeze, R1 boundary updated.
5. RB1–RB4 retained; no asset files generated.

## Acceptance for R0-B-R2 (chat-side checklist)

- one stable `coordinate_id` can be reused across feasibility contracts without contradictory authority rows;
- every feasibility-bound membership/outcome row resolves to exactly one region and feasibility contract;
- region↔mask remains derivable from one declared membership authority;
- every pack resolves to one fully identified candidate universe and pre-decision state contract;
- predicate missing/unknown/comparator/endpoint behavior is machine-reconstructible;
- current pack is explicitly observational and cannot be mistaken for single-step or closed-loop evidence;
- policy identity uses a canonical AST and remains separate from observed-mask equality;
- role provenance survives canonicalization and no G7 role is inferred;
- RB1–RB4 and CR1–CR9 remain intact;
- every FK resolves to one authority;
- R1 remains unauthorized pending final chat-side acceptance.

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset data files;
- implement a generic runtime/framework library;
- implement G4–G7, LOO, shadow, hook, preset, or production behavior;
- infer G7 roles or map unknown to reject;
- compare/compose different universes without transport;
- promote observational algebra to intervention safety;
- change terminal B, maturity A0, or accepted T0 numbers;
- promote to evidence ledger;
- self-accept R0-B-R2 or authorize R1.

## History

- 2026-07-10: R0-A accepted; R0-B draft `75dec59a`.
- 2026-07-10: R0-B review requested RB1–RB4.
- 2026-07-10: R0-B-R1 delivered at `e02a5367`; RB1–RB4 pass on re-review.
- 2026-07-10: Boolean composition semantics contract added at branch tip after `e02a5367`.
- 2026-07-10: integrated review requested RB5–RB7; sole active → R0-B-R2; R1/A1 remain unauthorized.
- 2026-07-10: R0-B-R2 delivered (RB5–RB7 typed realization integration); awaiting chat-side re-review; R1/A1 still unauthorized.
