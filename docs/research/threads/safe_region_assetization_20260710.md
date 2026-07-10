---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A-R1 correctly resolves CR1–CR5, but mathematical-contract re-review is **CHANGES_REQUESTED**. Current sole task is **R0-A-R2 mathematical/identity normalization**. R0-B, R1 asset generation, grammar extension, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-A-R2 mathematical/identity normalization** |
| Existing G1–G3 T0/C0 result | **A0 descriptive baseline**; terminal B retained |
| R0-A packet | original `762adf9a`; R0-A-R1 correction `136841a8` |
| CR1–CR5 | **PASS after R0-A-R1** |
| Mathematical-contract re-review | **CHANGES_REQUESTED** — CR6–CR9 |
| R0-B final contract | **NOT AUTHORIZED** |
| R1 G1–G3 asset conversion | **NOT AUTHORIZED** |
| R2 grammar asset extension | **CONDITIONAL** after accepted R0/R1 and an asset-increment hypothesis |
| R3 transfer qualification | **CONDITIONAL** on selected A1 assets |
| R4 intervention qualification | **CONDITIONAL** on selected A2 assets |
| Current maturity | **A0 retained** |
| Maximum current mathematical claim | registered-lattice, in-sample descriptive feasible-set geometry only |
| Occ-exit conditional modeling | **PARKED** future RegionAsset producer/consumer |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

The research object is a reusable `RegionAsset`, not a rule and not a coverage-table cell:

```text
producer semantics
→ parameter / decision representation
→ estimated feasible/productive-safe set
→ stable region identity
→ geometry and decision-mask quotient
→ productive-capacity distribution
→ applicability geometry
→ transfer qualification
→ action contract
```

Hard separation remains:

```text
artifact generated
≠ engineering ready
≠ asset maturity accepted
≠ statistical claim level
≠ research conclusion accepted
≠ intervention qualified
≠ production approved
```

No assetization step may manufacture population safety, causal, transfer, or online evidence absent from source artifacts.

## Read first

1. [Statistical Robust Feasible-Set Estimation under Asymmetric Loss](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) — canonical mathematical/evidence-semantics contract
2. [R0-A / R0-A-R1 preflight packet](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
3. [Closed A0/T0 thread](composition_grammar_safe_region.md)
4. [T0 artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
5. [T0 region interpretation](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)
6. [Superseded grammar-coverage design](composition_grammar_coverage_program_20260710.md)

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

This baseline is an observed, registered-lattice result under the declared cohort and unresolved firewall. It is not a population-risk guarantee, transferable region, actionable policy, or production candidate.

## Program stages

```text
R0 Region Asset Contract
  R0-A preflight
  R0-A-R1 structural correction          # CR1–CR5 pass
  R0-A-R2 math/identity normalization    ← current
  R0-B final accepted contract

R1 G1–G3 deterministic asset conversion
R2 grammar-by-grammar asset extension
R3 transfer qualification toward A2
R4 intervention qualification toward A3
A4 separate production approval
```

## R0-A-R1 re-review result

### Pass — CR1–CR5

The corrected packet now properly preserves:

- truth context, pack/materialization, content identity, and evidence-record layers as separate concepts;
- non-additive component-capacity distributions;
- event-union capacity as blocked without membership evidence;
- sequence incidence, union, intersection, and min/max member support;
- grammar/search-domain null distinct from concrete region semantics;
- authoritative region↔mask derivation through coordinate foreign keys;
- all prior provenance, per-grid-mask, semantic-role, maturity, and production firewalls.

These corrections are retained and must not regress.

## Blocking corrections — R0-A-R2

### CR6 — Stable semantic identity must not depend on byte-level file ordering

The corrected proposal still puts raw atlas/T0 file SHA maps inside `truth_context_id`, then claims that row reordering and equivalent regeneration leave content IDs unchanged. Those claims cannot both be true: byte-level row reordering changes file hashes and therefore changes `truth_context_id` and all child content IDs.

R0-A-R2 must separate:

```text
truth_contract_id
  = normalized semantic/data contract identity
  = substrate + cohort semantics + label/unresolved contract
    + lattice/coordinate semantics + normalized data-content identity

evidence_bundle_id
  = exact run/artifact seal
  = manifest + raw file hashes + recorded source hashes

pack_id
  = materialization/schema/generator identity

region/mask/coordinate content IDs
  = truth_contract_id + local normalized semantic/content identity
```

Requirements:

- raw file SHA maps remain mandatory provenance but do not define semantic content identity;
- normalized table/content digests must be order-insensitive and field-declared;
- a row reorder or byte-equivalent format migration changes `evidence_bundle_id` when appropriate, but not `truth_contract_id` or region IDs if normalized truth is unchanged;
- a changed cohort, label, signal value, lattice, unresolved policy, or region membership must change the relevant semantic/content ID;
- do not use only a human study name as normalized truth identity.

### CR7 — RegionAsset needs an explicit mathematical feasibility contract

The new canonical framework defines the object as an estimated feasible/productive-safe set under asymmetric loss. R0-A-R1 preserves geometry but does not yet make the mathematical constraint itself a first-class identity.

R0-A-R2 must add a `feasibility_contract_id` covering at least:

```text
parameter_or_policy_space_id        # Θ and representation grain
candidate_universe_id               # Ω / exposed population
safety_loss_definition              # operational L_GT
productivity_definition             # operational G_FP or count surrogate
epsilon                             # rate/count bound; exact-zero contract explicit
g_min                               # or explicit n_neg_captured > 0 surrogate
n_gt_exposed / denominator owner
n_fp_exposed / denominator owner
selection_scope                     # in-sample searched / frozen held-out / other
finite_sample_statement             # observed GT0 ≠ population risk zero
metric_adjacency_edge_policy        # geometry contract
```

For current Q4.5 G1–G3, encode exactly what exists:

```text
safety: resolved GT_hurt == 0 under unresolved-contamination firewall
productivity: n_neg_captured > 0
GT exposure denominator: declared primary positive-protection cohort
negative exposure denominator: declared primary negative cohort
selection scope: searched and evaluated on the registered in-sample cohort
population confidence bound: not established
```

Also keep two independent ladders:

```text
asset maturity: A0–A4
statistical/transfer claim: L0–L6
```

A mechanically valid A1 asset may still carry only an L1 in-sample-region claim. These ladders must not be inferred from each other.

### CR8 — G3 search-domain identity must enumerate the actual concrete semantic domain

The grammar-level G3 null record is valid, but its current `search_domain_id` example includes only OR, lattice kind, signal family, and counts. Different sets of 40 concrete operand grids could share those summary fields.

R0-A-R2 must define:

```text
search_domain_id
  = digest of canonical ordered membership in concrete grid domains

search_domain_members
  = normalized rows linking search_domain_id
    → grid_domain_id
    → concrete semantic_definition_id
    → registered coordinate denominator
```

For G3, each registered operand-pair grid has directly available concrete OR semantics even though it contains zero productive-safe coordinates. Therefore:

- the grammar-level null record keeps `semantic_definition_id = NULL`;
- each search-domain member/grid links to its concrete OR `semantic_definition_id`;
- optional per-grid null summaries, if emitted, reuse that concrete semantic definition;
- domain identity must change if a grid/operand family is added, removed, or semantically changed, not merely when the total count changes.

### CR9 — Proposed R1 schemas must close referential integrity

The corrected schemas introduce foreign keys for semantic definitions, search domains, truth context, evidence bundles, and pack membership, but do not yet define authoritative machine files for all of them.

R0-A-R2 must propose explicit grains, PKs, FKs, and invariants for at least:

```text
truth_contract.json                 # or one-row table
evidence_bundle.json                # exact hashes / source seal
semantic_definitions.csv|jsonl      # concrete operator trees
search_domains.csv|jsonl            # grammar/search-domain objects
search_domain_members.csv           # domain → concrete grids/semantics
pack_membership.csv                 # pack → content objects, or normalized evidence_records
```

The final R1 file map may rename these, but no FK may point to an undefined authority.

Capacity must also retain both non-collapsed views:

```text
region member-capacity distribution by coordinate
region member-capacity distribution by per-grid mask unit
```

They answer different questions and must be separately named; neither is an event-union capacity.

## Current step — R0-A-R2

Revise the same note in place:

```text
docs/modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md
```

Required changes:

1. add the canonical mathematical framework to inputs and terminology;
2. replace raw-hash-scoped `truth_context_id` with normalized `truth_contract_id` plus exact `evidence_bundle_id`;
3. update all stable content-ID definitions and synthetic reorder/regeneration tests;
4. add `feasibility_contract_id`, exposure/productivity/epsilon fields, and A0–A4 × L0–L6 separation;
5. make G3 search-domain membership concrete and auditable across all registered grids;
6. add missing authoritative machine schemas and referential-integrity invariants;
7. emit separate coordinate-member and mask-member capacity-distribution contracts;
8. update readiness recommendation and self-audit;
9. do not generate any asset files.

## Acceptance for R0-A-R2

- semantic content IDs survive row reorder, serialization changes, and unrelated pack/schema additions when normalized truth is unchanged;
- exact raw artifact seals remain preserved under a distinct evidence-bundle identity;
- changes in actual cohort/signal/lattice/label/membership truth change the appropriate IDs;
- every RegionAsset/null result points to an explicit feasibility contract with denominators and asymmetric-loss semantics;
- observed GT0 is explicitly bounded as finite-sample/in-sample evidence;
- A0–A4 maturity and L0–L6 claim level remain orthogonal;
- G3 domain identity includes all concrete registered grid semantics;
- all FK authorities and pack membership are machine-defined;
- coordinate-level and per-grid-mask-level capacity distributions are both retained and never summed as simultaneous actions;
- CR1–CR5 remain intact;
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
- open an R1 implementation PR;
- self-accept R0-A-R2 or authorize R0-B/R1 from this thread alone.

## History

- 2026-07-10: T0-A/B/R1 accepted; PR #94 merged; G1–G3 retained as A0 baseline.
- 2026-07-10: Safe-Region Assetization Program opened; R0-A authorized.
- 2026-07-10: R0-A delivered at `762adf9a`; first review requested CR1–CR5.
- 2026-07-10: R0-A-R1 delivered at `136841a8`; CR1–CR5 pass on re-review.
- 2026-07-10: cross-cutting mathematical framework added under `docs/research/eval/`.
- 2026-07-10: mathematical-contract re-review requested CR6–CR9; sole active → R0-A-R2; R0-B/R1 remain unauthorized.
