---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-B contract **ACCEPTED**; R1 engineering **MERGED** via [PR #95](https://github.com/raylei50653/saccade/pull/95). **A0 retained.** Current sole active gate = **A1 research asset acceptance** (research-consumption, not engineering re-check). R2–R4 unauthorized. No production/ledger promotion.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Delivery model | **PR-driven** (direct-agent dispatch **retired**) |
| Semantic sole active | **A1 research asset acceptance** (A0 pack; A1 not accepted) |
| R0-A / CR1–CR9 | **ACCEPTED / PASS** |
| R0-B-R1 | `e02a5367` — **RB1–RB4 PASS** |
| R0-B-R2 | `34eab247` — **RB5–RB7 PASS** |
| R0-B-R3 | `f92340b7` — **RB8–RB9 PASS** |
| R0-B final contract | **ACCEPTED** (+ editorial E1) |
| R1 engineering delivery | **MERGED** on `main` (converter, tests, A0 pack tooling) |
| PR #95 engineering review | **COMPLETE** (history pointer only) |
| Current active gate | **A1 research asset acceptance** (chat-side / research-owner) |
| Research asset acceptance (A1) | **not accepted** — open gate; separate from engineering merge |
| Current maturity | **A0 retained** |
| R1 output maturity | **A0 pack candidate; R1 does not self-promote A0→A1** |
| R2–R4 | **unauthorized** (exist only after A1 verdict + owner gate) |
| Pack root | `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/` |
| Conversion note | [safe_region_asset_r1_conversion_20260710.md](../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md) |
| Claims | G1 **1×L0**; G2 **6×L0 isolated + 19×L1 multi**; G3 **1×L0**; pack ceiling **L1** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

```text
R1 engineering delivery: MERGED
PR #95 engineering review: COMPLETE
current active gate: A1 research asset acceptance
current maturity: A0 retained
R2–R4: unauthorized
```

## Gate separation (normative)

Keep these distinct:

| Gate | Owns | Authority | Status |
|:--|:--|:--|:--|
| **Engineering delivery** | Converter, tests, pack emission, docs mirrors | implementation branch + PR | **done** |
| **Engineering review** | Code quality, CI, contract fidelity of implementation | PR reviewers / CI | **COMPLETE** ([#95](https://github.com/raylei50653/saccade/pull/95)) |
| **Research asset acceptance (A1)** | Whether the A0 pack is a consumable research asset | chat-side / research-owner | **open** |
| **Maturity promotion** | A0→A1 (and later stages) | research documents + owner gate only | not performed |
| **Merge** | Landing engineering on `main` | PR merge (does **not** imply research acceptance) | **done** |
| **Next-stage authorization** | R2+ / production / ledger | research-owner; not implied by merge | unauthorized |

```text
engineering merge on main
≠ research acceptance
≠ A0→A1 maturity promotion
≠ evidence promotion
≠ R2 authorization
```

## Acceptance record

Chat-side final review of `f92340b7d5f95b449297fbc141fa028de60a8b87` accepts:

```text
CR1–CR9
RB1–RB9
R0-B RegionAsset contract
```

### Editorial erratum E1

The contract previously contained four inverted implications in §2.4:

```text
generator-contract equality ⇒ same sealed universe instance
source_event_table_sha256 ⇒ universe_membership_digest
policy family ⇒ concrete threshold-executable policy
thr_index without registry ⇒ reconstructible thr_value
```

E1 was corrected before R1 pack emission: those statements were replaced with explicit **non-implications / forbidden inferences**, matching the accepted model and the contract's own §12.4 firewall. The correction was editorial and did not reopen RB8 or RB9.

## Current boundary

```text
contract accepted
≠ asset pack generated

asset pack generated + engineering merged
≠ A1 research acceptance

A0 pack candidate
≠ transferable or actionable policy

observational composition
≠ single-step intervention
≠ closed-loop safety

engineering merge
≠ research conclusion promotion
```

R1 transforms sealed inputs into deterministic authorities and emission tables only. It does not authorize maturity promotion. This write-back is **ownership / status correction only** — not a new research stage.

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
11. R1 delivery / engineering review history: [PR #95](https://github.com/raylei50653/saccade/pull/95) (MERGED)

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

R1 deterministic G1–G3 conversion    # ENGINEERING DELIVERED
PR #95 engineering review + merge    # COMPLETE (history)
A1 research asset acceptance         ← current active gate
R2–R4 conditional maturity stages    # unauthorized until A1 + owner gate
A4/L6 separate production approval
```

## R1 delivery record

R1 engineering delivery (complete; landed via PR #95):

- editorial E1 applied; contract marked ACCEPTED;
- deterministic converter `scripts/tools/convert_safe_region_asset_r1.py`;
- pack root `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/`;
- two-run authority fingerprint match; PK/FK/claims/firewall validation PASS;
- conversion note [safe_region_asset_r1_conversion_20260710.md](../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md);
- unit tests `tests/unit/test_safe_region_asset_r1_conversion.py` PASS;
- engineering review + merge: [PR #95](https://github.com/raylei50653/saccade/pull/95) **COMPLETE / MERGED**.

No evaluator modification, evaluator rerun, threshold search, geometry recomputation, or new research claims.

### Artifact availability

Runtime full atlases are required machine inputs and are not all in the committed evidence subset. Candidate-level rows are required for `universe_membership_digest`.

```text
required artifact present → convert deterministically
required artifact absent  → BLOCKED_BY_ARTIFACT
```

On `BLOCKED_BY_ARTIFACT`, emit only a bounded preflight/block report. Do not rebuild the evaluator outputs, fabricate IDs, substitute dataset names, or use the raw source-event-table SHA as the logical universe digest.

## Engineering prerequisites (already passed — not A1)

The following are **engineering / contract-fidelity prerequisites**. They **already passed** on R1 delivery and PR #95. **Do not re-run them as the A1 research gate:**

```text
determinism (two-run authority fingerprint)
PK uniqueness + FK resolution
SHA seals / input seal match
ID / threshold reconstruction
manifest flags (A0, observational, production_forbidden)
claim-ceiling non-escalation
```

Passing those creates an **A0 pack candidate** only. PR merge does not decide A1.

## A1 research asset acceptance (current gate)

A1 answers **research consumption**, not engineering re-validation.

### A1 questions (formal)

```text
1. semantic fidelity
   RegionAsset 能否無歧義回指原始 atlas / policy / mask / component？

2. research query utility
   pack 能否直接回答 topology、capacity、sequence support、
   duplicate mask、null asset 等研究問題？

3. decision utility
   pack 能否導出 retain / close / transfer-candidate /
   further-observation-needed 的 bounded classification？

4. reusable abstraction
   後續 grammar 或 intervention applicability 是否能沿用，
   而不需重建另一套物件模型？
```

### A1 terminals (fixed)

```text
A1_ACCEPTED
  pack 已成為可信且可消費的研究資產

A1_ACCEPTED_WITH_LIMITS
  部分資產可消費；必須明列哪些研究查詢仍需回讀原始 artifacts

A1_REJECTED
  semantic fidelity 或研究效用不足，需要有限修正
```

`WITH_LIMITS` **must** enumerate residual raw-artifact queries. A1 verdict does **not** by itself authorize R2–R4, transfer, LOO, production, or ledger promotion — those remain separate owner gates after A1.

### A1 out of scope (this gate)

```text
no R0-C / R1.1 contract refinement as primary progress
no selecting transfer candidates as a committed next stage
no R2 task creation until after A1 terminal
no terminal B change
no A0 maturity self-promotion
no evidence_ledger promotion
no RegionAsset contract expansion
```

Suggested first consumption probe after/with A1 (not auto-authorized as R2): portfolio interpretation over 26 components / 34 masks into retain / transfer-candidate / null / close — only as evidence for A1 decision utility, not as stage advancement.

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
- treat engineering re-checks (determinism / PK-FK / seals / reconstruction / flags) as A1 research acceptance;
- expand R0/R1 contract as the primary next action;
- create R2 tasks or authorize transfer/LOO before an A1 terminal;
- recreate a direct-agent `*.dispatch.yaml` execution authority.

## History

- 2026-07-10: R0-A accepted; CR1–CR9 pass.
- 2026-07-10: R0-B-R1 delivered; RB1–RB4 pass.
- 2026-07-10: Boolean composition semantics contract added.
- 2026-07-10: R0-B-R2 delivered at `34eab247`; RB5–RB7 pass.
- 2026-07-10: R0-B-R3 delivered at `f92340b7`; RB8–RB9 pass.
- 2026-07-10: chat-side final review accepted R0-B and authorized R1 deterministic conversion only; A0 retained; A1 remains a separate gate.
- 2026-07-10: R1 conversion delivered — A0 pack candidate emitted; two-run determinism + PK/FK/claims PASS; research A1 gate open (not self-accepted).
- 2026-07-10: **Retired direct-agent dispatch sidecar**; delivery model switched to **PR-driven engineering** + separate research acceptance. Historical note only: Chat-copied hash start protocols and same-name dispatch sidecars are no longer execution authority and must not be recreated.
- 2026-07-10: PR #95 engineering review + merge **COMPLETE** on `main` (`d963fe21`). Engineering path closed; #95 retained as delivery/history pointer only.
- 2026-07-10: **Status write-back** — sole active gate reframed to **A1 research asset acceptance** (semantic fidelity · query utility · decision utility · reusable abstraction). Engineering prerequisites marked already-passed. Terminals fixed: `A1_ACCEPTED` / `A1_ACCEPTED_WITH_LIMITS` / `A1_REJECTED`. No R2 authorization, no maturity change, no ledger promotion.
