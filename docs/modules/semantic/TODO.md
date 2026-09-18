# Semantic Relink — 模組 TODO

> **WIP register only**（O0）：只鎖 decision-changing mainline charter。任務與排序在 GitHub issue（milestone [`Now`](https://github.com/raylei50653/saccade/milestone/1)）。高階型別與 owner 路由見 [research control plane](../../research/README.md#research-control-plane)；Expected state / probe 在 linked charter；**state / verdict / terminal 的唯一 fact-owner 是 [claim-state registry](../../research/contracts/claim_state_registry.md)**，本檔不複述。
> 開發路由：[DEVELOPMENT action cards](../../../DEVELOPMENT.md#agent-action-cards) · 規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

- **OWDL — WAITING_OWNER_SEAL; implementation complete; no formal outcome access authorized.**
  owner decision → [#442](https://github.com/raylei50653/saccade/issues/442)（`Now`）·
  [thread](../../research/threads/observability_weighted_directional_likelihood_task.md) ·
  [declaration](research/observability_weighted_directional_likelihood_declaration_20260827.md).
- GCTM/H0 runtime slots remain **無 active**; OWDL neither activates nor
  satisfies their runtime-substrate and compatibility gates.

## Registry-owned lines（state 在 registry；此處只留指標）

- **H0** — five mechanical `H0_PROVENANCE_INVALID` terminals（route-1 + re-entries + R4 `S=a76efffa…` + R5 `S=6fdb060c…`）；all `S` permanently spent; no faithful capture; no actual H0 guarantee; Phase B forbidden; registry `admissible_units: []`. Any re-entry = new owner decision + fresh repair/seal/authorization chain. Evidence roots `research/evidence/h0_phase_a_<I>/` + execution witnesses; detail in registry.
- **H2**（successor to H0's identity layer only; card `proposed` / non-WIP）— Phase A **CLOSED 2026-08-04**, owner verdict `ACCEPT WITH NAMED LIMITS — H2 measurement closure`（[admitted packet](research/evidence/h2_measure_envelope_c570dd9202498f390083dd02503d5675f900e027/)，PR #321）。Phase B not established, equivalence `unproven`, no I/F/S, no seal, **no automatic next**; named limits held only by registry `successor_formal_measurement_executed.owner_review`. Navigation: [task charter](../../research/threads/h2_behavioral_identity_capture_task.md) · [declaration](research/headline_bridge_behavioral_identity_capture_declaration_20260725.md).
- **GCTM** — consumer re-charter `GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE` and registration-v3 `H0_REGISTRATION_V3_CONTRACT_SEALABLE`（2026-07-24, WIP released at terminal; **not** H0 implementation authority, **not** actual guarantee）→ [closed charters](../../research/threads/closed/).
- Frozen-substrate consequence: `tracker_gpu.hpp` / `tracker_gpu.cu` are strict sha256 inputs of the CLOSED packets; evolution path = [#436](https://github.com/raylei50653/saccade/issues/436)（`Now`）; portable OR-tail hook accept/dispose = [#438](https://github.com/raylei50653/saccade/issues/438)（`Later`, blocked by #436）.

## Proposed（non-WIP）

- **H0 route-5 B1 — runtime-grounded consumer-faithful operating curve** —
  `H0_ROUTE5_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; distinct from and
  coexists with `GCTM_B1` → [machine identity decision](../../research/contracts/gctm_b1_slot_identity_decision_v1.json)
- **GCTM B1 — runtime-grounded offline attribution and score-ranking evaluation** —
  `GCTM_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; does not alias or
  supersede `H0_ROUTE5_B1` → [task charter](../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
- **GCTM O1 — online score intervention and system-efficacy evaluation** → [task charter](../../research/threads/gctm_o1_online_intervention_efficacy_task.md)

## Parked

- Score temporal-to-stable-domain → [charter](../../research/threads/score_temporal_to_stable_domain_20260712.md)
- GT-support morphology → [charter](../../research/threads/gt_support_morphology_20260711.md)
- Occ-exit intervention modeling → [charter](../../research/threads/occ_exit_audit_20260709.md)
- Sparse key-embedding bank → [research note](research/sparse_key_embedding_bank_20260704.md)

## Navigation（不佔 WIP）

- [Research threads index](../../research/threads/README.md)
- [Module research index](README.md)
- [Claim-state registry](../../research/contracts/claim_state_registry.md)
- [NO-GO registry](../../reference/no_go_registry.md)
- Evidence governance → [#404](https://github.com/raylei50653/saccade/issues/404)

## Done / closed

See the module research index, closed threads index, evidence ledger, and NO-GO registry; terminal details do not live in TODO.
