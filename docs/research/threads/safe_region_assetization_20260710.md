---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A preflight is **ACCEPTED**. R0-B Final RegionAsset Contract Draft is **delivered** and **awaiting chat-side review**. R1 asset generation, A0→A1, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**; pack claim ceiling **L1** with G1=L0 · G2=L1 · G3=L0.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-B completed — awaiting chat-side review** |
| Existing G1–G3 T0/C0 result | **A0 descriptive baseline**; terminal B retained |
| R0-A preflight chain | `762adf9a` → R1 `136841a8` → R2 `0f5799f4` |
| R0-A research review | **ACCEPTED** |
| CR1–CR9 | **PASS** (frozen into R0-B draft) |
| R0-B final contract draft | **DELIVERED** · not self-accepted · [safe_region_asset_contract.md](../eval/safe_region_asset_contract.md) |
| R1 G1–G3 asset conversion | **NOT AUTHORIZED** |
| Current maturity | **A0 retained** |
| Statistical claim | pack ceiling **L1**; G1=**L0**, G2=**L1**, G3=**L0** |
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
→ region / mask / coordinate / null identities
→ geometry + capacity + sequence applicability descriptions
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
4. [Closed A0/T0 thread](composition_grammar_safe_region.md)

## Accepted A0 baseline

```text
154 productive-safe coordinates = 1 G1 + 153 G2 + 0 G3
26 coordinate components · 34 productive per-grid mask units
full_neighborhood_safe_radius >= 1: 0/154
G3 Hard OR: registered-lattice NULL_RESULT
terminal B: isolated_safe_points_only
claim levels: G1=L0 · G2=L1 · G3=L0 · pack ceiling=L1
```

## Program stages

```text
R0 Region Asset Contract
  R0-A preflight                         # ACCEPTED
  R0-B final contract draft             ← delivered; awaiting review
  chat-side contract review

R1 G1–G3 deterministic asset conversion # not authorized
R2–R4 conditional maturity stages
A4/L6 separate production approval
```

## R0-B draft summary (locked in contract)

| Topic | Lock |
|:--|:--|
| Claim ownership | L-level on evidence/claim records — **not** inside `feasibility_contract_id` |
| Per-object L-level | G1=L0, G2=L1, G3=L0; pack ceiling L1 aggregate only |
| Grid authority | `grid_domains` table required; FK target of members |
| Normalized digest | exact truth columns, row keys, missing/float/duplicate rules |
| Identity layers | truth_contract ⟂ evidence_bundle ⟂ feasibility ⟂ pack ⟂ content |
| Capacity | dual distributions; no sum; event-union blocked |
| Sequence | incidence + union + intersection; no A2 inference |
| Region↔mask | coordinate FKs authoritative |
| Action | observation_only; production_forbidden |

### Not authorized

```text
R0-B research acceptance (chat-side only)
R1 asset file generation
A0→A1 maturity promotion
L1→L2+ claim promotion
evaluator rerun / modification
G4–G7, LOO, shadow, hooks, presets, ledger promotion
```

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset data files;
- implement a generic runtime/framework library;
- implement G4–G7, LOO, shadow, hook, preset, or production behavior;
- infer G7 roles;
- change terminal B, maturity A0, or accepted T0 numbers;
- promote to evidence ledger;
- self-accept R0-B or authorize R1.

## History

- 2026-07-10: T0-A/B/R1 accepted; PR #94; G1–G3 A0 baseline.
- 2026-07-10: Safe-Region Assetization opened; R0-A authorized.
- 2026-07-10: R0-A → R0-A-R1 → R0-A-R2; CR1–CR9 pass at preflight level.
- 2026-07-10: chat-side review **accepted R0-A** and authorized R0-B contract draft only.
- 2026-07-10: R0-B draft delivered at `docs/research/eval/safe_region_asset_contract.md`; **awaiting chat-side review** (not accepted; R1 unauthorized; A0 retained).
