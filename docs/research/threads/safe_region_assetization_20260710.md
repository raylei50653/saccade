---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A is **ACCEPTED**. R0-B-R1 RB1–RB4 and R0-B-R2 RB5–RB7 are **PASS**. Final contract review is **CHANGES_REQUESTED** only on **RB8–RB9 identity/reconstruction closure**. Current sole task is **R0-B-R3**. R1 asset generation, A0→A1, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-B-R3 identity/reconstruction closure** |
| R0-A / CR1–CR9 | **ACCEPTED / PASS** |
| R0-B-R1 | `e02a5367` — **RB1–RB4 PASS** |
| R0-B-R2 | `34eab247` — **RB5–RB7 PASS** |
| R0-B final contract | **CHANGES_REQUESTED** on RB8–RB9 only |
| Boolean semantics patch | **NORMATIVE** — [boolean_composition_semantics_contract.md](../eval/boolean_composition_semantics_contract.md) |
| R1 | **NOT AUTHORIZED** |
| Maturity | **A0 retained** |
| Claims | G1 **1×L0**; G2 **6×L0 isolated + 19×L1 multi**; G3 **1×L0**; pack ceiling **L1** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

```text
candidate-universe generator contract
≠ sealed candidate-universe instance membership

parameterized policy / grammar definition
≠ concrete threshold-bound executable policy

raw source-event-table SHA
≠ normalized candidate-membership digest

threshold index
≠ reconstructible threshold value without registry authority
```

All earlier boundaries remain:

```text
candidate / reject set in Ω ≠ policy / feasible set in Θ
truth-level coordinate/mask identity ≠ feasibility-bound realization
canonical policy semantics ≠ observed-mask equivalence
observational composition ≠ single-step intervention ≠ closed-loop safety
artifact generated ≠ A1 accepted ≠ intervention qualified ≠ production approved
```

## Read first

1. [R0-B RegionAsset contract](../eval/safe_region_asset_contract.md)
2. [Boolean Composition Semantics Contract](../eval/boolean_composition_semantics_contract.md)
3. [Mathematical framework](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
4. [Accepted R0-A preflight](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
5. [Sealed Q4.5 threshold registry](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/threshold_registry.json)
6. [Q4.5 evidence manifest](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/manifest.json)
7. [Q4.5 SHA inventory](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/SHA256SUMS.json)

## Accepted A0 baseline

```text
154 PS = 1 G1 + 153 G2 + 0 G3
26 components = 1 G1 + 25 G2 (6 isolated + 19 multi)
radius≥1: 0/154 · terminal B · G3 domain null
claims: G1 1×L0 · G2 6×L0 + 19×L1 · G3 1×L0 · pack ceiling L1
```

Finite-sample, searched in-sample, registered-lattice evidence only. No population, held-out, intervention, or production claim.

## Program stages

```text
R0-A preflight                         # ACCEPTED
R0-B-R1 RB1–RB4                        # PASS
R0-B-R2 RB5–RB7                        # PASS
R0-B-R3 identity/reconstruction       ← current
chat-side final contract review

R1 G1–G3 deterministic conversion     # not authorized
R2–R4 conditional maturity stages
A4/L6 separate production approval
```

## Accepted contract corrections — retain without regression

### RB1–RB4

- object claim level derives from geometry: G1 1×L0; G2 6×L0 isolated + 19×L1 multi; G3 1×L0; pack ceiling L1 aggregate only;
- `region_asset_manifest.json` is the authoritative `pack_id` row;
- region/null outcome identities and claims bind `feasibility_contract_id`; masks/coordinates remain feasibility-independent;
- pairwise leaves canonicalize all `(atom, feature, direction, thr_index, role)` fields before truth digest, axes, coordinate keys, and membership.

### RB5–RB7

- truth-level `coordinates` / `mask_units` are separated from feasibility-bound `region_coordinate_membership`;
- candidate-universe and predicate-edge machine authorities are first-class; Q4.5 is observational, `U→no_reject`, no cross-universe composition without transport;
- canonical Boolean policy/grammar AST identity is separated from observed-mask and logical-equivalence status; role provenance survives canonicalization; no G7 roles are inferred.

These items are **PASS** and must not be reopened by R0-B-R3.

## Blocking corrections — R0-B-R3

### RB8 — Bind sealed candidate membership into universe instance identity

The R0-B-R2 draft defines `candidate_universe_id` primarily from substrate/hook/builder/prefilter/state metadata and stores `universe_hash` beside it. This can assign the same ID to two runs with identical generator metadata but different actual candidate membership.

For direct Boolean composition and mask comparison, the sealed observation set—not only its generator contract—must be identical.

R0-B-R3 must choose one explicit model:

```text
A. candidate_universe_id is the sealed instance ID and digests
   normalized universe membership / universe_hash;
```

or:

```text
B. split:
   candidate_universe_contract_id  # builder/hook/prefilter/key/state schema
   candidate_universe_instance_id  # contract + normalized membership digest
```

Requirements:

- packs, feasibility contracts, evidence claims, and same-universe comparisons bind the **instance** identity;
- define the exact normalized universe digest: candidate primary-key columns, exposure/label-bearing columns, canonical sort, missing encoding, duplicate/conflict handling, and hash algorithm;
- `source_event_table_sha256` remains an exact evidence seal, not a substitute for normalized logical membership identity;
- if candidate-level rows are unavailable, emit a fail-closed status such as `BLOCKED_BY_ARTIFACT`; do not fabricate a logical universe hash from a dataset name;
- same generator contract with changed candidate membership must produce a different instance ID/hash.

### RB9 — Add threshold-registry authority and concrete policy reconstruction grain

The sealed Q4.5 evidence already contains `threshold_registry.json` with `thr_index`, `thr_value`, lattice, feature, direction, and quantile-point mappings. The R0-B-R2 output schema does not make that registry an authority: `coordinates.csv` requires only indices and treats threshold aliases/values as optional.

Therefore `policy_definition_id` currently identifies a parameterized AST/family, not by itself a concrete executable threshold policy.

R0-B-R3 must define:

```text
threshold_registry.json|jsonl
  PK threshold_registry_id (or registry + entry composite key)
  exact thr_index → threshold_value_repr mapping
  feature, direction, lattice_kind, signal unit
  quantile lattice point/method and canonical float representation
```

and bind coordinates to exact registry entries for every axis.

It must also explicitly define the concrete policy grain, for example:

```text
policy_family_definition_id   # canonical parameterized AST / grammar
policy_instance_id            # family + ordered threshold bindings
```

or declare an equivalent canonical composite such as:

```text
(policy_definition_id, coordinate_id, threshold_registry_id)
```

as the serialized concrete executable policy identity.

Requirements:

- two independent packers reconstruct identical threshold values and concrete policy IDs;
- `threshold_value_repr` is required for execution/reproduction, while float values remain excluded from coordinate PKs as previously locked;
- comparator/endpoint/unknown semantics come from predicate authority; threshold values come from the registry; axis binding comes from the coordinate/grid authority;
- observed masks attach to concrete policy instances/coordinates, not to the parameterized family as identity;
- Q4.5 uses the existing sealed threshold registry; no evaluator rerun or new threshold search.

## Current step — R0-B-R3

Revise in place:

```text
docs/research/eval/safe_region_asset_contract.md
```

Required output:

1. retain CR1–CR9 and RB1–RB7 unchanged;
2. close sealed universe contract-vs-instance identity (RB8);
3. add threshold-registry authority and concrete threshold-bound policy grain (RB9);
4. update identity layers, PK/FK map, validation invariants, pack manifest, Q4.5 freeze, and R1 boundary;
5. use the existing Q4.5 threshold registry and evidence seals only;
6. generate no asset data files.

## Acceptance for R0-B-R3

- a changed sealed candidate membership cannot reuse the same universe instance ID;
- universe normalization is explicit enough for two implementations to agree;
- exact raw event-table SHA and normalized universe membership identity remain separate;
- every coordinate axis resolves to one sealed threshold-registry entry;
- every concrete threshold policy is reconstructible from machine authorities without guessing float values;
- parameterized policy-family identity is not mislabeled as a concrete executable policy;
- policy/coordinate/mask relations preserve the existing Boolean, feasibility, and observed-mask separations;
- RB1–RB7 and CR1–CR9 remain intact;
- every FK resolves to exactly one authority;
- R1 remains unauthorized pending final chat-side acceptance.

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset data files;
- perform new threshold search or recompute research conclusions;
- implement G4–G7, LOO, shadow, hook, preset, intervention, or production behavior;
- infer G7 roles or map unknown to reject;
- compare/compose different universe instances without transport;
- change terminal B, maturity A0, accepted counts, or claim levels;
- promote to evidence ledger;
- self-accept R0-B-R3 or authorize R1.

## History

- 2026-07-10: R0-A accepted; CR1–CR9 pass.
- 2026-07-10: R0-B-R1 delivered; RB1–RB4 pass.
- 2026-07-10: Boolean composition semantics contract added.
- 2026-07-10: R0-B-R2 delivered at `34eab247`; RB5–RB7 pass.
- 2026-07-10: final R0-B-R2 review requested RB8–RB9 identity/reconstruction closure; sole active → R0-B-R3; R1/A1 remain unauthorized.
