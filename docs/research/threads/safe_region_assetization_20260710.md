---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-B RegionAsset contract is **ACCEPTED** at reviewed delivery `f92340b7` with RB1–RB9 **PASS**. Current sole task is **R1 deterministic G1–G3 asset conversion**. R1 may package the sealed evidence into an **A0 observation-only pack candidate**; it does not grant A1, L2+, intervention, transfer, production, or ledger promotion.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R1 deterministic asset conversion** |
| R0-A / CR1–CR9 | **ACCEPTED / PASS** |
| R0-B-R1 | `e02a5367` — **RB1–RB4 PASS** |
| R0-B-R2 | `34eab247` — **RB5–RB7 PASS** |
| R0-B-R3 | `f92340b7` — **RB8–RB9 PASS** |
| R0-B final contract | **ACCEPTED** |
| R1 | **AUTHORIZED — deterministic conversion only** |
| Current maturity | **A0 retained** |
| R1 output maturity | **A0 pack candidate; separate chat-side acceptance required for A1** |
| Claims | G1 **1×L0**; G2 **6×L0 isolated + 19×L1 multi**; G3 **1×L0**; pack ceiling **L1** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Acceptance record

Chat-side final review of `f92340b7d5f95b449297fbc141fa028de60a8b87` accepts:

```text
CR1–CR9
RB1–RB9
R0-B RegionAsset contract
```

### Editorial erratum E1

`safe_region_asset_contract.md` §2.4 contains four editorially inverted positive implications:

```text
generator-contract equality ⇒ same sealed universe instance
source_event_table_sha256 ⇒ universe_membership_digest
policy family ⇒ concrete threshold-executable policy
thr_index without registry ⇒ reconstructible thr_value
```

They are rejected by the accepted model and by the contract's own §12.4 firewall. Before generating a valid R1 pack, the implementation change must replace them with explicit **non-implications / forbidden inferences**. This is a non-substantive text correction; it does not reopen RB8 or RB9.

## Current boundary

```text
contract accepted
≠ asset pack generated

asset pack generated
≠ engineering/research acceptance as A1

A0 pack candidate
≠ transferable or actionable policy

observational composition
≠ single-step intervention
≠ closed-loop safety

engineering merge
≠ research conclusion promotion
```

R1 is authorized to transform sealed inputs into deterministic authorities and emission tables only.

## Read first

1. [Agent execution dispatch](safe_region_assetization_20260710.dispatch.yaml)
2. [Accepted R0-B RegionAsset contract](../eval/safe_region_asset_contract.md)
3. [Boolean Composition Semantics Contract](../eval/boolean_composition_semantics_contract.md)
4. [Mathematical framework](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
5. [Accepted R0-A preflight](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
6. [Q4.5 artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
7. [Sealed Q4.5 threshold registry](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/threshold_registry.json)
8. [Q4.5 manifest](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/manifest.json)
9. [Q4.5 SHA inventory](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/SHA256SUMS.json)
10. [T0 interpretation evidence](../../modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/manifest.json)

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
R0-B-R3 RB8–RB9                        # PASS
R0-B final contract                    # ACCEPTED

R1 deterministic G1–G3 conversion    ← current
chat-side R1 pack review / A1 gate
R2–R4 conditional maturity stages
A4/L6 separate production approval
```

## R1 authorized work

R1 may:

- implement a deterministic, research-only converter/validator;
- consume the existing sealed Q4.5 runtime full atlases, committed Q4.5 seals, T0 evidence, threshold registry, and source candidate rows when present;
- emit the authority and membership files required by contract §11;
- generate a RegionAsset pack under `out/`;
- validate stable IDs, normalized digests, PK/FK integrity, canonical pair ordering, claim derivation, and byte/content determinism;
- write an R1 conversion note and update navigation/status mirrors;
- apply editorial erratum E1 and mark the contract accepted before valid pack emission.

R1 must run without evaluator modification, evaluator rerun, threshold search, geometry recomputation, or new research claims.

### Artifact availability

Runtime full atlases are required machine inputs and are not all in the committed evidence subset. Candidate-level rows are required for `universe_membership_digest`.

```text
required artifact present → convert deterministically
required artifact absent  → BLOCKED_BY_ARTIFACT
```

On `BLOCKED_BY_ARTIFACT`, emit only a bounded preflight/block report. Do not rebuild the evaluator outputs, fabricate IDs, substitute dataset names, or use the raw source-event-table SHA as the logical universe digest.

## R1 acceptance checklist

A delivered R1 pack is reviewable only when:

- E1 is corrected in the accepted contract;
- inputs match recorded SHA seals;
- two clean converter runs produce identical authority content and IDs;
- all PKs are unique and every FK resolves exactly once;
- universe contract and sealed instance are distinct, with normalized membership digest;
- threshold-registry entries reconstruct every coordinate's concrete threshold values;
- policy family and concrete policy instance identities remain separate;
- region/mask relations derive only from feasibility-bound membership rows;
- object claims remain G1 1×L0, G2 6×L0 + 19×L1, G3 null L0, pack ceiling L1;
- manifest remains `maturity_declared=A0`, `composition_level=observational`, `production_forbidden=true`;
- no evaluator/threshold search/production/ledger changes occurred.

Passing engineering validation creates an **A0 pack candidate**. Chat-side review must separately decide A0→A1; R1 may not self-promote it.

## Must not

- modify or rerun the Q4.5 evaluator;
- run a new threshold search or recompute accepted research conclusions;
- weaken the contract or tests to make conversion pass;
- invent candidate rows, universe membership, G7 roles, or missing threshold values;
- implement G4–G7, NOT authorization, LOO, shadow, hook, preset, intervention, or production behavior;
- compare/compose different universe instances without transport;
- change terminal B, accepted counts, claim levels, or production defaults;
- promote to evidence ledger;
- self-accept the generated pack as A1 or authorize later stages;
- open a PR unless separately authorized.

## History

- 2026-07-10: R0-A accepted; CR1–CR9 pass.
- 2026-07-10: R0-B-R1 delivered; RB1–RB4 pass.
- 2026-07-10: Boolean composition semantics contract added.
- 2026-07-10: R0-B-R2 delivered at `34eab247`; RB5–RB7 pass.
- 2026-07-10: R0-B-R3 delivered at `f92340b7`; RB8–RB9 pass.
- 2026-07-10: chat-side final review accepted R0-B and authorized R1 deterministic conversion only; A0 retained; A1 remains a separate gate.
