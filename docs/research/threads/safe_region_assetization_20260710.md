---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A is **ACCEPTED**. R0-B-R1 minimal contract correction is **delivered** and **awaiting chat-side re-review**. R1 asset generation, A0→A1, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**; pack claim ceiling **L1** with G1=1×L0 · G2=6×L0+19×L1 · G3=1×L0.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-B-R1 completed — awaiting chat-side re-review** |
| R0-A research review | **ACCEPTED** |
| CR1–CR9 | **PASS** (retained) |
| R0-B draft | `75dec59a` CHANGES_REQUESTED → **R0-B-R1 delivered** (RB1–RB4) · not self-accepted |
| Contract | [safe_region_asset_contract.md](../eval/safe_region_asset_contract.md) |
| Boolean semantics | [boolean_composition_semantics_contract.md](../eval/boolean_composition_semantics_contract.md) — normative Ω/Θ, partial-predicate, role, grammar, threshold, universe and closed-loop boundary |
| R1 | **NOT AUTHORIZED** |
| Maturity | **A0 retained** |
| Claims | G1 **1×L0**; G2 **6×L0 isolated + 19×L1 multi**; G3 **1×L0**; pack ceiling **L1** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

```text
artifact generated
≠ engineering ready
≠ asset maturity accepted (A0–A4)
≠ statistical claim level (L0–L6)
≠ research conclusion accepted
≠ intervention qualified
≠ production approved
```

Boolean set algebra is additionally bounded to same-universe, same-pre-decision-state composition. Observational mask equality does not imply logical equivalence or closed-loop intervention equivalence.

## Read first

1. [R0-B / R0-B-R1 RegionAsset contract draft](../eval/safe_region_asset_contract.md)
2. [Mathematical framework](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
3. [Boolean composition semantics contract](../eval/boolean_composition_semantics_contract.md)
4. [Accepted R0-A preflight](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)

## Accepted A0 baseline

```text
154 PS = 1 G1 + 153 G2 + 0 G3
26 components = 1 G1 + 25 G2 (6 isolated + 19 multi)
radius≥1: 0/154 · terminal B · G3 domain null
```

## Program stages

```text
R0-A ACCEPTED
R0-B draft CHANGES_REQUESTED
R0-B-R1 delivered ← awaiting re-review
R1 not authorized
```

## R0-B-R1 correction summary (RB1–RB4)

| ID | Fix |
|:--|:--|
| RB1 | claim_level from geometry: G2 6×L0 + 19×L1; no grammar-wide G2=L1 |
| RB2 | `region_asset_manifest.json` authoritative pack row for all `pack_id` FKs |
| RB3 | model A: `region_asset_id` / `null_record_id` + claims bind `feasibility_contract_id` |
| RB4 | pairwise full leaf swap before truth digest / coords / grid axes; must also preserve typed Boolean AST/universe/role semantics |

R0-B-R1 re-review must treat the Boolean semantics contract as normative for any AST, operand-role, threshold-edge, observed-mask-equivalence, NOT/complement, or online-composition field. It must not authorize G7 or closed-loop intervention merely because those semantics are now specified.

### Not authorized

```text
R0-B / R0-B-R1 acceptance (chat-side only)
R1 asset generation
A0→A1 · L2+ · evaluator rerun · G4–G7 · hooks · ledger
```

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset data files;
- implement framework / G4–G7 / LOO / shadow / hook / preset / production;
- infer missing/unknown predicates as reject;
- compare or compose masks across different universes without a transport contract;
- promote observational Boolean algebra to single-step or closed-loop safety;
- change terminal B or maturity A0;
- promote to evidence ledger;
- self-accept R0-B-R1 or authorize R1.

## History

- 2026-07-10: R0-A accepted; R0-B draft `75dec59a`.
- 2026-07-10: R0-B review CHANGES_REQUESTED (RB1–RB4).
- 2026-07-10: R0-B-R1 delivered; **awaiting chat-side re-review**.
- 2026-07-10: typed Boolean composition semantics contract added and bound into R0-B-R1 review; program verdict unchanged.
