---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A preflight is **ACCEPTED**. R0-B draft at `75dec59a` is **CHANGES_REQUESTED**. Current sole task is **R0-B-R1 minimal contract correction**. R1 asset generation, A0→A1, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**; corrected pack claim ceiling remains **L1**.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-B-R1 minimal contract correction** |
| Existing G1–G3 T0/C0 result | **A0 descriptive baseline**; terminal B retained |
| R0-A research review | **ACCEPTED** — derivability/provenance/math preflight closed |
| CR1–CR9 | **PASS at preflight level**; retain without regression |
| R0-B draft | **DELIVERED** at `75dec59a`; **CHANGES_REQUESTED** on four normative blockers |
| R1 G1–G3 asset conversion | **NOT AUTHORIZED** |
| Current maturity | **A0 retained** |
| Corrected statistical claims | G1 region **L0**; G2 **6 isolated L0 + 19 multi-coordinate L1**; G3 null **L0**; pack ceiling **L1** |
| Mathematical framework | [Statistical Robust Feasible-Set Estimation under Asymmetric Loss](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

```text
producer semantics
→ parameter / policy space and candidate universe
→ feasibility contract
→ estimated productive-safe set
→ normalized truth identity + exact evidence seal
→ feasibility-bound region / null realization
→ geometry + capacity + sequence descriptions
→ transfer qualification
→ action contract
```

Hard separations:

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

1. [R0-B RegionAsset contract draft](../eval/safe_region_asset_contract.md)
2. [Statistical Robust Feasible-Set Estimation under Asymmetric Loss](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
3. [Accepted R0-A preflight packet](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
4. [T0 component geometry evidence](../../modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/component_geometry.csv)
5. [Closed A0/T0 thread](composition_grammar_safe_region.md)

## Accepted A0 baseline

```text
154 productive-safe coordinates = 1 G1 + 153 G2 + 0 G3
26 coordinate components · 34 productive per-grid mask units
G1: 1 isolated component
G2: 25 components = 6 isolated + 19 multi-coordinate
full_neighborhood_safe_radius >= 1: 0/154
G3 Hard OR: registered-lattice NULL_RESULT
terminal B: isolated_safe_points_only
```

This is finite-sample, searched in-sample, registered-lattice evidence. It is not population safety, held-out retention, intervention evidence, or a production candidate.

## Program stages

```text
R0 Region Asset Contract
  R0-A preflight                         # ACCEPTED
  R0-B draft                             # CHANGES_REQUESTED
  R0-B-R1 minimal correction            ← current
  chat-side final contract review

R1 G1–G3 deterministic asset conversion # not authorized
R2–R4 conditional maturity stages
A4/L6 separate production approval
```

## R0-B review result

### Pass

The draft correctly freezes:

- mathematical feasible/productive-safe definition and asymmetric-loss boundaries;
- claim ownership outside `feasibility_contract_id`;
- normalized truth vs exact evidence seal;
- grid/search-domain semantics and G3 40-member null domain;
- non-additive dual capacity distributions;
- sequence incidence/union/intersection without A2 promotion;
- coordinate FKs as region↔mask authority;
- A/L/action/production firewalls.

### Blocking corrections

#### RB1 — Per-object claim level must follow object geometry, not grammar label

T0 evidence contains six G2 `isolated_point` components with `n_coords=1`; these are **L0**, not L1. The remaining nineteen multi-coordinate G2 components may carry L1 in-sample geometry. Lock:

```text
G1: 1 × L0
G2: 6 × L0 isolated + 19 × L1 multi-coordinate
G3: 1 × L0 domain null
pack_claim_ceiling: L1 aggregate only
```

Do not encode `G2 → L1` as a grammar-wide constant. Derive object claim level from declared evidence shape, with explicit fail-closed rules.

#### RB2 — Every `pack_id` FK needs one authoritative pack object

`pack_membership` and `region_claim_contract` reference `pack_id`, but `region_asset_manifest.json` is currently listed as optional/derived. Make one file authoritative, for example:

```text
packs.json|jsonl
```

or declare `region_asset_manifest.json` as the normative one-row pack authority. It must own `pack_id` and the truth/evidence/feasibility/schema/producer FKs. No FK may target an optional derived header.

#### RB3 — Feasible-set realizations and claims must bind to the feasibility contract

A region component and especially a null result are outcomes relative to `(epsilon, g_min, loss, universe, denominators)`. The current `region_asset_id` / `null_record_id` and `evidence_claims` do not fully bind that context.

R0-B-R1 must choose one consistent model:

```text
A. include feasibility_contract_id in region_asset_id and null_record_id
```

or:

```text
B. keep reusable geometry/content IDs feasibility-independent,
   but introduce an authoritative feasibility_realization/evidence_record
   linking feasibility_contract_id + evidence_bundle_id + content/null object,
   and move HAS_REGION/NULL_RESULT/claim_level to that realization grain.
```

Whichever model is chosen:

- every observed claim must resolve to exactly one `feasibility_contract_id` and one `evidence_bundle_id`;
- G3 null identity/realization must differ when epsilon, productivity floor, candidate universe, or loss definition changes;
- pure mask/coordinate identities may remain feasibility-independent;
- no outcome field may float without its mathematical contract.

#### RB4 — Pairwise normalized truth must canonicalize symmetric operand order

The contract lex-sorts AND/OR leaves for semantic IDs, but §7 currently digests raw `combo_id`, `atom_a_id`, `atom_b_id`, feature/direction a/b, and threshold indices. Before stable-key generation and row digesting, pairwise rows must canonicalize the two leaves and swap all associated fields together:

```text
(atom_id, feature, direction, thr_index)_a
↔
(atom_id, feature, direction, thr_index)_b
```

The canonical row key must be derived after that normalization. Equivalent symmetric operand swaps must preserve normalized truth, grid/semantic identity, coordinate identity, and region membership coordinates after axis remapping.

## Current step — R0-B-R1

Revise in place:

```text
docs/research/eval/safe_region_asset_contract.md
```

Required output:

1. correct per-object L-level rule and Q4.5 counts;
2. define the authoritative pack schema;
3. bind region/null outcomes and claims to feasibility + evidence using one explicit model;
4. canonicalize pairwise operands in normalized truth and coordinate/grid derivation;
5. update authority tables, PK/FK rules, validation invariants, snapshot, and non-claims;
6. retain all R0-A and R0-B passing boundaries;
7. do not generate asset data files.

## Acceptance for R0-B-R1

- G2 isolated components are L0 and multi-coordinate components are L1; pack ceiling remains aggregate L1;
- object claim level is evidence-derived, not grammar-hardcoded;
- every `pack_id` FK resolves to one authoritative pack row;
- every feasible-set outcome/claim resolves to one feasibility contract and one evidence bundle;
- a change in epsilon/g_min/loss/universe cannot silently reuse a null-result realization;
- symmetric pairwise operand swaps preserve normalized logical identity after full a/b field and axis normalization;
- all CR1–CR9, capacity, sequence, null, action, and production firewalls remain intact;
- R1 remains unauthorized pending final chat-side acceptance.

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset data files;
- implement a generic runtime/framework library;
- implement G4–G7, LOO, shadow, hook, preset, or production behavior;
- infer G7 roles;
- change terminal B, maturity A0, or accepted T0 numbers;
- promote to evidence ledger;
- self-accept R0-B-R1 or authorize R1.

## History

- 2026-07-10: T0-A/B/R1 accepted; PR #94; G1–G3 A0 baseline.
- 2026-07-10: R0-A → R0-A-R1 → R0-A-R2; CR1–CR9 pass; R0-A accepted.
- 2026-07-10: R0-B draft delivered at `75dec59a`.
- 2026-07-10: chat-side R0-B review **CHANGES_REQUESTED** on RB1–RB4; current sole active → R0-B-R1; R1/A1 remain unauthorized.
