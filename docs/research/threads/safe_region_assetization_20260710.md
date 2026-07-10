---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-B RegionAsset contract is **ACCEPTED** (`f92340b7`, RB1–RB9). **R1 engineering delivery** exists on the research branch as an **A0 observation-only pack candidate**. Engineering/A1 review proceeds through a **PR** (not direct-agent dispatch). **A1 not granted** — research acceptance remains a separate chat-side / research-owner gate. No production/ledger promotion.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Delivery model | **PR-driven** (direct-agent dispatch **retired**) |
| Semantic sole active | **R1 engineering delivery complete — A0 pack candidate; A1 not accepted** |
| R0-A / CR1–CR9 | **ACCEPTED / PASS** |
| R0-B-R1 | `e02a5367` — **RB1–RB4 PASS** |
| R0-B-R2 | `34eab247` — **RB5–RB7 PASS** |
| R0-B-R3 | `f92340b7` — **RB8–RB9 PASS** |
| R0-B final contract | **ACCEPTED** (+ editorial E1) |
| R1 engineering delivery | **COMPLETE** on branch — converter, tests, A0 pack candidate |
| Engineering review | via **pull request** against `main` (PR metadata = head/base/CI/files) |
| Research asset acceptance (A1) | **not accepted** — separate from PR engineering review |
| Current maturity | **A0 retained** |
| R1 output maturity | **A0 pack candidate; R1 does not self-promote A0→A1** |
| Pack root | `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/` |
| Conversion note | [safe_region_asset_r1_conversion_20260710.md](../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md) |
| Claims | G1 **1×L0**; G2 **6×L0 isolated + 19×L1 multi**; G3 **1×L0**; pack ceiling **L1** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Gate separation (normative)

Keep these distinct:

| Gate | Owns | Authority |
|:--|:--|:--|
| **Engineering delivery** | Converter, tests, pack emission, docs mirrors | implementation branch + PR |
| **Engineering review** | Code quality, CI, contract fidelity of implementation | PR reviewers / CI |
| **Research asset acceptance (A1)** | Whether the A0 pack may be treated as accepted research asset | chat-side / research-owner |
| **Maturity promotion** | A0→A1 (and later stages) | research documents + owner gate only |
| **Merge** | Landing engineering on `main` | PR merge (does **not** imply research acceptance) |
| **Next-stage authorization** | R2+ / production / ledger | research-owner; not implied by merge |

```text
R1 implementation exists on the branch
A0 pack candidate emitted locally
engineering / A1 review should occur through a PR
R1 delivery does not self-promote A0→A1
PR merge ≠ research acceptance
engineering-ready ≠ evidence promotion
```

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

R1 transforms sealed inputs into deterministic authorities and emission tables only. It does not authorize maturity promotion.

## Read first

1. [Accepted R0-B RegionAsset contract](../eval/safe_region_asset_contract.md)
2. [Boolean Composition Semantics Contract](../eval/boolean_composition_semantics_contract.md)
3. [Mathematical framework](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
4. [R1 conversion note](../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md)
5. [Accepted R0-A preflight](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
6. [Q4.5 artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
7. [Sealed Q4.5 threshold registry](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/threshold_registry.json)
8. [Q4.5 manifest](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/manifest.json)
9. [Q4.5 SHA inventory](../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/SHA256SUMS.json)
10. [T0 interpretation evidence](../../modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/manifest.json)
11. Current implementation PR on GitHub (when open) — live head SHA, base, CI, files

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

R1 deterministic G1–G3 conversion    # ENGINEERING DELIVERED (A0 pack candidate)
PR engineering review                ← current engineering path
chat-side R1 pack review / A1 gate   ← separate research path (not self-granted)
R2–R4 conditional maturity stages    # unauthorized until A1 + owner gate
A4/L6 separate production approval
```

## R1 delivery record

R1 engineering delivery:

- editorial E1 applied; contract marked ACCEPTED;
- deterministic converter `scripts/tools/convert_safe_region_asset_r1.py`;
- pack root `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/`;
- two-run authority fingerprint match; PK/FK/claims/firewall validation PASS;
- conversion note [safe_region_asset_r1_conversion_20260710.md](../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md);
- unit tests `tests/unit/test_safe_region_asset_r1_conversion.py` PASS.

No evaluator modification, evaluator rerun, threshold search, geometry recomputation, or new research claims.

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

Passing engineering validation creates an **A0 pack candidate**. Chat-side / research-owner review must separately decide A0→A1; R1 may not self-promote it. PR merge does not decide A1.

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
- treat PR merge as research acceptance or maturity promotion;
- recreate a direct-agent `*.dispatch.yaml` execution authority.

## History

- 2026-07-10: R0-A accepted; CR1–CR9 pass.
- 2026-07-10: R0-B-R1 delivered; RB1–RB4 pass.
- 2026-07-10: Boolean composition semantics contract added.
- 2026-07-10: R0-B-R2 delivered at `34eab247`; RB5–RB7 pass.
- 2026-07-10: R0-B-R3 delivered at `f92340b7`; RB8–RB9 pass.
- 2026-07-10: chat-side final review accepted R0-B and authorized R1 deterministic conversion only; A0 retained; A1 remains a separate gate.
- 2026-07-10: R1 conversion delivered — A0 pack candidate emitted; two-run determinism + PK/FK/claims PASS; awaiting research A1 gate (not self-accepted).
- 2026-07-10: **Retired direct-agent dispatch sidecar**; delivery model switched to **PR-driven engineering** + separate research acceptance. Historical note only: Chat-copied hash start protocols and same-name dispatch sidecars are no longer execution authority and must not be recreated.
