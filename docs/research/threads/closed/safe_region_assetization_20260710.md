---
doc-status: closed
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
closed: 2026-07-11
closed-verdict: A1_ACCEPTED_WITH_LIMITS
---

# Safe-Region Assetization Program

> **One-line:** **`A1_ACCEPTED_WITH_LIMITS` recorded (2026-07-11) — A1 gate CLOSED.** Acceptance unit = conversion pack `1a180620bc…`; maturity **A1** with 5 enumerated `acceptance_limits` (see terminal record). Audit 26/26 PASS + mutation sensitivity 5/5. R1/R1.1 remain diagnostic overlay; R2–R4 / production / ledger remain **fail-closed** (accepting terminal alone does not authorize them). Program returns to research mainline: owner picks the next scientific uncertainty.

## Status

| Item | Status |
|:--|:--|
| Program | **CLOSED** — A1 gate closed; mainline handed off → [gt_support_morphology](../gt_support_morphology_20260711.md) |
| Delivery model | **PR-driven** (direct-agent dispatch **retired**) |
| Semantic sole active | **handed off 2026-07-11** → [GT-Support Morphology](../gt_support_morphology_20260711.md)（owner 選定的 next scientific uncertainty） |
| A1 acceptance unit | **locked**: conversion pack `1a180620bc050e70dd4a673f991ab24410e78ab02d2cbe49346414800ca631a7` — probe/R1.1 roots are **overlay, not A1 objects** |
| A1 terminal | **`A1_ACCEPTED_WITH_LIMITS` recorded 2026-07-11** (research owner) — gate **CLOSED** · [terminal record](#a1-terminal-record-2026-07-11) |
| A1 audit trail | S0/S1/Q1/N1 **26/26 PASS** + mutation sensitivity **5/5** (`tests/unit/test_safe_region_a1_audit.py`) · [audit note](../../../modules/semantic/research/safe_region_a1_audit_20260711.md) |
| R0-A / CR1–CR9 | **ACCEPTED / PASS** |
| R0-B-R1 | `e02a5367` — **RB1–RB4 PASS** |
| R0-B-R2 | `34eab247` — **RB5–RB7 PASS** |
| R0-B-R3 | `f92340b7` — **RB8–RB9 PASS** |
| R0-B final contract | **ACCEPTED** (+ editorial E1) |
| R1 pack conversion eng. | **MERGED** on `main` (converter, tests, A0 pack tooling) |
| PR #95 engineering path | **closed** — delivery merged; CI PASS; history pointer only |
| R1 capacity probe | **overlay** (downgraded 2026-07-10) · [note](../../../modules/semantic/research/safe_region_assetization_r1_20260710.md) · study `out/signal_study/safe_region_assetization_r1_20260710/` |
| Probe verdict | **V-C as heuristic-specific descriptive failure** — LOO candidate pool is global-label-screened (`r1.py:1010,1625,1372`), so **not** an inductive class null; pooled 4→8 in-sample witness stands |
| Identity hygiene | **34 grid-local mask assets ≠ 15 unique prediction masks** (reconciled) |
| LOO protocol name | `transductive_globally_registered_basis_LOO` **with global-label-screened candidate pool** (notes' "labels do not select registry" claim was false at the pool layer) |
| R2 grammar distillation | **NOT authorized** (requires V-A) |
| R1.1 attribution | **overlay** (downgraded) · [note](../../../modules/semantic/research/safe_region_assetization_r11_20260710.md) · study `out/signal_study/safe_region_assetization_r11_20260710/` |
| R1.1 net contribution | **2 unique harmful AND events located + 3 descriptive symptoms** (role reversal / weak retention / margin contraction); "primary F3" **rejected** (post-hoc floors, K-duplicated reversal count, alias-ambiguous predicate) |
| A1 audit | **26/26 PASS** — S0 unit lock · S1 crosswalk 0 mismatches · Q1 pack-only · N1 controls · `out/signal_study/safe_region_a1_audit_20260711/` |
| Next research | **owner picks next scientific uncertainty** — R1.1's four lines are candidate directions only, none committed; asset-management validation is done |
| Current maturity | **A1** (`A1_ACCEPTED_WITH_LIMITS`; 5 enumerated limits — see terminal record); probe `A1_region_asset` tags = overlay labels only (not pack maturity) |
| R2–R4 | **unauthorized** (fail-closed; V-A not met) |
| Pack root (conversion) | `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/` |
| Pack root (probe R1) | `out/signal_study/safe_region_assetization_r1_20260710/` |
| Conversion note | [safe_region_asset_r1_conversion_20260710.md](../../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md) |
| Probe note | [safe_region_assetization_r1_20260710.md](../../../modules/semantic/research/safe_region_assetization_r1_20260710.md) |
| Claims | G1 **1×L0**; G2 **6×L0 isolated + 19×L1 multi**; G3 **1×L0**; pack ceiling **L1** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

```text
R1 engineering delivery: MERGED (#95); audit + overlay code MERGED (#97)
A1 acceptance unit: conversion pack 1a180620bc… (locked; probe/R1.1 = overlay)
A1 terminal: A1_ACCEPTED_WITH_LIMITS (recorded 2026-07-11) — gate CLOSED
current maturity: A1 + 5 enumerated acceptance_limits
R2–R4: unauthorized (fail-closed — accepting terminal exists, but each named
       stage still requires explicit owner authorization)
next: research mainline — owner picks next scientific uncertainty
```

## Gate separation (normative)

Keep these distinct:

| Gate | Owns | Authority | Status |
|:--|:--|:--|:--|
| **Engineering delivery** | Converter, tests, pack emission, docs mirrors | implementation branch + PR | **done** |
| **Engineering path close** | Landed engineering + CI; closes eng. workstream | merge + CI ([#95](https://github.com/raylei50653/saccade/pull/95)) | **closed** (not a formal review-record claim) |
| **Research asset acceptance (A1)** | Whether the A0 pack is a consumable research asset; **terminal binds maturity** | chat-side / research-owner | **closed** — `A1_ACCEPTED_WITH_LIMITS` (2026-07-11) |
| **Merge** | Landing engineering on `main` | PR merge (does **not** imply research acceptance) | **done** |
| **Next-stage authorization** | Named R2+ / production / ledger stages | research-owner; **fail-closed** (see below) | unauthorized |

```text
engineering merge on main
≠ research acceptance
≠ A0→A1 maturity promotion
≠ evidence promotion
≠ R2 authorization
```

### R2–R4 authorization (fail-closed)

```text
R2–R4 remain unauthorized unless
  A1 terminal ∈ {A1_ACCEPTED, A1_ACCEPTED_WITH_LIMITS}
  AND the research owner explicitly authorizes the named next stage.

A1_REJECTED retains A0 and authorizes only bounded R1 repair/re-review.
Any A1 terminal alone does not authorize R2–R4, transfer, LOO,
production, or ledger promotion.
```

## A1 terminal record (2026-07-11)

Recorded by the research owner after: acceptance-unit lock → read-only S0/S1/Q1/N1 audit **26/26 PASS** → mutation sensitivity **5/5** (audit FAILs under component-alias tamper, per-sequence-support tamper, forbidden-promotion removal, self-promoted maturity) → repository-wide state consistency audit. No new evidence was produced for this terminal; no post-hoc D1 was run.

```text
A1_ACCEPTED_WITH_LIMITS

maturity:
  A1

accepted scope:
  - semantic crosswalk fidelity
  - fixed pack-only topology/capacity/support/grain/null queries
  - encoded negative-control and claim-boundary metadata

acceptance_limits:
  - decision utility not demonstrated; no pre-declared D1 trace
  - reusable abstraction not demonstrated; no independent second consumer
  - event-mass queries require raw event artifacts
  - non-productive-cell queries require raw artifacts
  - predicate-alias semantic resolution requires raw/contract-level artifacts

non-authorizations:
  - R1/R1.1 remain diagnostic overlay only
  - no class-level transfer null
  - no R2–R4 authorization
  - no production or evidence-ledger promotion
```

Meaning: the pack is a trusted, **boundedly consumable** research asset; it has **not** demonstrated direct decision-making capability or reuse by a second research workflow. Reusable abstraction is validated later, usage-based, when an independent second consumer appears naturally — not by manufacturing one.

The pack's own `maturity_declared=A0` / `review_status=A0_PACK_CANDIDATE_AWAITING_CHAT_REVIEW` fields are **sealed producer-side self-declarations** (the converter must never self-promote); this terminal record is the authority that binds effective maturity **A1**.

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

1. [Accepted R0-B RegionAsset contract](../../eval/safe_region_asset_contract.md)
2. [Boolean Composition Semantics Contract](../../eval/boolean_composition_semantics_contract.md)
3. [Mathematical framework](../../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
4. [R1 conversion note](../../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md)
5. [A1 acceptance-unit lock + audit (2026-07-11)](../../../modules/semantic/research/safe_region_a1_audit_20260711.md)
6. [Accepted R0-A preflight](../../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
7. [Q4.5 artifact preflight](../../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
8. [Sealed Q4.5 threshold registry](../../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/threshold_registry.json)
9. [Q4.5 manifest](../../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/manifest.json)
10. [Q4.5 SHA inventory](../../../modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/SHA256SUMS.json)
11. [T0 interpretation evidence](../../../modules/semantic/research/evidence/m_b1_5_t0_region_interpretation_20260710/manifest.json)
12. R1 delivery history: [PR #95](https://github.com/raylei50653/saccade/pull/95) (MERGED; CI PASS; engineering path closed)

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
PR #95 merge + CI                    # MERGED; path closed (history)
A1 research asset acceptance         # CLOSED — A1_ACCEPTED_WITH_LIMITS
  (terminal binds maturity → A1)       (recorded 2026-07-11)
R2–R4 named stages                   ← next decision point: unauthorized
                                       until owner explicitly authorizes a
                                       named stage (accepting terminal alone
                                       is not authorization)
A4/L6 separate production approval
```

## R1 delivery record

R1 engineering delivery (complete; landed via PR #95):

- editorial E1 applied; contract marked ACCEPTED;
- deterministic converter `scripts/tools/convert_safe_region_asset_r1.py`;
- pack root `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/`;
- two-run authority fingerprint match; PK/FK/claims/firewall validation PASS;
- conversion note [safe_region_asset_r1_conversion_20260710.md](../../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md);
- unit tests `tests/unit/test_safe_region_asset_r1_conversion.py` PASS;
- [PR #95](https://github.com/raylei50653/saccade/pull/95): **engineering delivery merged; CI PASS; engineering path closed** (history pointer; not a formal review-record claim).

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

### A1 questions (restructured by 2026-07-10 chat review)

The original four-in-one framing was rejected as an acceptance test: only the
first two are verifiable against the pack today.

```text
A1-S  semantic fidelity (hard gate)
      RegionAsset 能否無歧義回指原始 atlas / policy / mask / component？
      → audited 2026-07-11: S1 crosswalk 0 mismatches (11/11 PASS)

A1-Q  fixed, pack-only research query battery
      topology · dual capacity · duplicate-mask grain · G3 null —
      pack-only 精確回答（predeclared goldens）；
      sequence union/intersection — pack-only **computable**
      （驗證可計算性＋封閉性；未建 per-region golden，不宣稱精確回答）
      → audited 2026-07-11: 5/5 PASS · N1 negative controls 6/6 PASS

D1    decision utility — 只驗證「事前固定」的 bounded decision trace；
      規則必須先由 owner 固定，post-hoc trace 不算數 → NOT yet run

R-later reusable abstraction — usage-based maturity；
      需要第二個獨立 consumer 沿用 RegionAsset IDs/relations 才可驗證，
      在此之前不進 A1 判準
```

### A1 terminals (fixed) — bind maturity

A0–A4 are **asset maturity** levels in the accepted RegionAsset contract. The A1 terminal **directly decides** maturity; do not leave `A1_ACCEPTED` while still marking A0.

```text
A1_PENDING_VALIDATION          (non-terminal; added 2026-07-10)
  受驗物已鎖定但 audit / terminal 未完成
  → A0 retained; 不表示 pack 有 defect
  → exited 2026-07-11: terminal A1_ACCEPTED_WITH_LIMITS recorded

A1_ACCEPTED
  pack 已成為可信且可消費的研究資產
  → maturity = A1

A1_ACCEPTED_WITH_LIMITS
  部分資產可消費；必須明列哪些研究查詢仍需回讀原始 artifacts
  → maturity = A1 + explicit acceptance_limits

A1_REJECTED
  semantic fidelity 或研究效用不足，需要有限修正
  → maturity = A0 retained
  → authorizes only bounded R1 repair / re-review
  → does NOT open R2–R4
```

`WITH_LIMITS` **must** enumerate residual raw-artifact queries as `acceptance_limits`. Per the 2026-07-10 review: `A1_ACCEPTED` additionally requires S0/S1/Q1/N1 all PASS with zero semantic mismatch (met 2026-07-11); `A1_REJECTED` applies on non-unique acceptance object, any critical semantic mismatch, any false-promoted negative control, or limits describable only vaguely. Semantic errors must **not** be packaged as limits. Known candidate limits: no D1 trace · reusable abstraction unproven (usage-based) · event-mass / non-productive-cell / predicate-alias queries need raw artifacts.

D1 binding (disambiguated 2026-07-11 — closing A1 must not spawn a decision experiment):

```text
A1_ACCEPTED:
  requires pre-declared D1 only if decision utility is claimed.

A1_ACCEPTED_WITH_LIMITS:
  may record absence of D1 as an explicit acceptance_limit;
  no post-hoc D1 is required to close A1.
```

Even after an accepting terminal (`A1_ACCEPTED` or `A1_ACCEPTED_WITH_LIMITS`), R2–R4 / transfer / LOO / production / ledger remain **unauthorized** until the research owner **explicitly authorizes the named next stage**. Maturity promotion to A1 is bound by the terminal; next-stage work is not.

### A1 out of scope (historical — while the gate was open; gate closed 2026-07-11)

```text
no pre-written A1 terminal or maturity promotion
no R0-C / R1.1 contract refinement as primary progress
no selecting transfer candidates as a committed next stage
no R2 task creation under A1_REJECTED or without accepting terminal
no terminal B change
no evidence_ledger promotion
no RegionAsset contract expansion
```

Suggested first consumption probe for the A1 decision (not auto-authorized as R2): portfolio interpretation over 26 components / 34 masks into retain / transfer-candidate / null / close — only as evidence for A1 decision utility, not as stage advancement.

## Must not

- modify or rerun the Q4.5 evaluator;
- run a new threshold search or recompute accepted research conclusions;
- weaken the contract or tests to make conversion pass;
- invent candidate rows, universe membership, G7 roles, or missing threshold values;
- implement G4–G7, NOT authorization, LOO, shadow, hook, preset, intervention, or production behavior;
- compare/compose different universe instances without transport;
- change terminal B, accepted counts, claim levels, or production defaults;
- promote to evidence ledger;
- self-accept the generated pack as A1 without a recorded A1 terminal;
- leave maturity=A0 after `A1_ACCEPTED` / `A1_ACCEPTED_WITH_LIMITS`, or set maturity=A1 after `A1_REJECTED`;
- treat PR merge as research acceptance or maturity promotion;
- treat engineering re-checks (determinism / PK-FK / seals / reconstruction / flags) as A1 research acceptance;
- expand R0/R1 contract as the primary next action;
- treat any A1 terminal (including `A1_REJECTED`) as sufficient for R2–R4;
- create R2 tasks or authorize transfer/LOO without
  (accepting A1 terminal ∈ {`A1_ACCEPTED`, `A1_ACCEPTED_WITH_LIMITS`}
   AND explicit owner authorization of the named stage);
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
- 2026-07-10: PR #95 **merged** on `main` (`d963fe21`); CI PASS; engineering path closed. #95 retained as delivery/history pointer only (not a formal GitHub review-record claim).
- 2026-07-10: **Status write-back** — sole active gate reframed to **A1 research asset acceptance** (semantic fidelity · query utility · decision utility · reusable abstraction). Engineering prerequisites marked already-passed. Terminals fixed with maturity binding: `A1_ACCEPTED`→A1 · `A1_ACCEPTED_WITH_LIMITS`→A1+limits · `A1_REJECTED`→A0 + bounded repair only. R2–R4 fail-closed (accepting terminal + explicit owner auth). No A1 verdict pre-written, no ledger promotion.
- 2026-07-10: **PR #96 review follow-up** — fail-closed R2 wording; terminal↔maturity binding; soften #95 "engineering review COMPLETE" to merged/CI/path-closed.
- 2026-07-10: **R1 capacity probe delivered** — Phase A G1–G3 region asset tables + Phase B sparse NN hard-safety probe; L3 pooled n_neg≤8 multi-seq in-sample but nested LOO holds GT hurt → **verdict V-C**; R2 unauthorized; terminal B retained; production/ledger unchanged.
- 2026-07-10: **Research acceptance of V-C** — refined as `in-sample grammar-limited` + `cross-sequence invariance-limited` (not blanket signal-limited). Phase A ACCEPTED after 34-vs-15 identity reconciliation (`n_unique_prediction_masks=15`, `n_grid_local_mask_assets=34`). LOO named transductive global label-free basis registry + train-only supervised fit. R2/G4–G7/hook/ledger closed. Suggested next: R1.1 transfer-failure attribution (F1–F5), not grammar search.
- 2026-07-10: **R1.1 AUTHORIZED and COMPLETE** — Transfer Failure Attribution Pack. Primary **F3** (train-neg bases fire holdout GT on MOT17-05/11; score_m_bridge-low AND patterns). Secondary **F4** (weak holdout productive retention) + **F1** (margin sign flip 4/14 folds). F2 moderate Jaccard only; F5 rejected (pooled n_neg=8). Decision mapping: no global gate; R2 still closed; owner chooses next among conditional applicability / new signal family / narrow coord-transport spec / formal mainline close.
- 2026-07-10: **Chat review — A1 framing invalidated; R1/R1.1 downgraded to overlay.** Seven findings, all verified against code/artifacts: (1) acceptance object non-unique (probe writes `A1_region_asset` tags while pack is A0); (2) R1/R1.1 never consume the pack (raw Q4.5/events only); (3) LOO candidate pool is global-label-screened top-48, reused per fold, truncated to top-24 (`r1.py:1010,1625,1372`) — "labels do not select registry" false at the pool layer; V-C survives only as heuristic-specific descriptive failure; (4) F1/F3/F4 hard-floored post-hoc to 40/55/50 (`r11.py:584,598,605`); (5) F3's ≥3-reversal trigger double-counts K2/K5 — 2 unique (basis,event) pairs; (6) harmful basis `b2:07de9243…` has 12 observed-mask aliases — predicate identity not unique; (7) probe `linear_probe_models.csv` writes `multi_sequence_productive_coordinates` (12) into `n_productive_sequences` in a 7-seq cohort (`r1.py:1574`) — semantic counterexample **in the overlay, not the pack**. New non-terminal state **A1_PENDING_VALIDATION**; A1 split into A1-S (hard gate) + A1-Q (pack-only battery); decision utility → pre-declared D1; reusable abstraction → usage-based. R1.1's four next-lines deferred until terminal.
- 2026-07-11: **A1 acceptance unit locked + read-only audit PASS.** Unit = conversion pack `1a180620bc…` (probe/R1.1 = overlay). `scripts/tools/run_safe_region_a1_audit.py` → **26/26**: S0 4/4 · S1 11/11 (154 memberships, 26↔26 components, 34→15 masks, 1080 registry entries reconstruct sealed values, 0 mismatches) · Q1 5/5 pack-only · N1 6/6. D1 not run (no pre-declared rule). [Audit note](../../../modules/semantic/research/safe_region_a1_audit_20260711.md). Maturity stays A0; terminal (e.g. `A1_ACCEPTED_WITH_LIMITS` with enumerated limits) awaits research owner.
- 2026-07-11: **PR #97 merged** (audit runner + overlay evidence code + docs write-back; CI green). Follow-up review fixes landed in-branch: Q1.sequence claim softened to pack-only *computable* (no per-region goldens predeclared); D1 binding disambiguated (`A1_ACCEPTED` needs pre-declared D1 only if decision utility claimed; `WITH_LIMITS` records absence of D1 as a limit — no post-hoc D1); mutation tests added (`tests/unit/test_safe_region_a1_audit.py`).
- 2026-07-11: **A1 terminal recorded: `A1_ACCEPTED_WITH_LIMITS` — gate CLOSED.** Post-merge closure sequence per owner: mutation sensitivity **5/5** on `main` → terminal record (see [A1 terminal record](#a1-terminal-record-2026-07-11): maturity A1; accepted scope = crosswalk fidelity + fixed pack-only queries + encoded negative controls; 5 acceptance_limits incl. no D1 trace and no second consumer; non-authorizations unchanged) → repository-wide state consistency audit (no stale "A1 open / A0 retained / terminal pending"; R1/R1.1 still overlay; R2–R4/production/ledger still fail-closed). No new evidence produced; no post-hoc D1. Program returns to research mainline — owner picks the next scientific uncertainty.
- 2026-07-11: **Next-line pick recorded** — owner selected **GT-Support Morphology** as the next scientific uncertainty（[thread](../gt_support_morphology_20260711.md)）；R1.1 的 role-reversal 症狀由該線的 escape-tail 機制化吸收。本 thread 轉為 CLOSED-state 導航（A1 terminal / limits / R2–R4 fail-closed 不變）。
