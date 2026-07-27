# Semantic Relink — 模組 TODO

> **WIP register only**（O0）：只鎖 decision-changing mainline charter。高階型別與 owner 路由見 [research control plane](../../research/README.md#research-control-plane)；Expected state / probe 在 linked charter；事實與結論在 [research](research/) / [README](README.md)。
> 開發路由：[DEVELOPMENT action cards](../../../DEVELOPMENT.md#agent-action-cards) · 規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

- **無 active**
- H0 R4 repair closed at `H0_R4_REPAIR_QUALIFIED_SEALABLE`
  （`h0_authority_overlay_runtime_binding_split_v1` / Amendment 10）；WIP released.
  Seal PR #277 landed exact S=`a76efffa…`. See
  [repair evidence](research/evidence/h0_r4_authority_overlay_runtime_binding_split_20260724/).
- **H0-R4 Phase-A executed once under sealed S（2026-07-24）** — owner-accepted
  truthful-negative terminal only（PR #279 merge `55d2da47…`）:
  - `I=2a233387…` / `F=ced4a4cc…` / `S=a76efffa…`
  - authorization `h0_r4_phase_a_exactly_once_authorization_20260724`（Issue #278；consumed）
  - controller `result=provenance_invalid` → `H0_PROVENANCE_INVALID`; verifier `valid=true`
  - evidence:
    [h0_phase_a_2a233387…](research/evidence/h0_phase_a_2a233387a6a321dd43570e2e30dc718571b3b4f4/)
    + [execution witness](research/evidence/h0_r4_phase_a_execution_witness_20260724/)
  - S4 permanently spent; no faithful capture; no actual H0 guarantee; Phase B forbidden;
    no future reentry authorization from R4 alone.
  Detail: [claim-state registry](../../research/contracts/claim_state_registry.md).
- H0 closure baseline（**five** mechanical `H0_PROVENANCE_INVALID` terminals including
  R5 S5; no faithful capture; no actual H0 guarantee; Phase B forbidden）:
  state fact-owner 見
  [claim-state registry](../../research/contracts/claim_state_registry.md)。
- **H0-R5 Repair closed at `H0_R5_ATTESTATION_QUALIFIED_SEALABLE`**
  （`h0_extension_plugin_runtime_attestation_closure_v1` / Issue #280）。
  Controlled-host qualification passed; later tool_runtime independent-expansion parity
  repair landed at `I=524f7e3b…`. See
  [repair evidence](research/evidence/h0_r5_extension_plugin_attestation_closure_20260724/).
- **H0-R5 Phase-A executed once under sealed S（2026-07-25）** — mechanical
  truthful-negative terminal:
  - `I=524f7e3b…` / `F=6e425dc6…` / `S=6fdb060c…`
  - authorization `h0_r5_phase_a_exactly_once_authorization_20260725`（Issue #283；consumed）
  - controller `result=provenance_invalid` → `H0_PROVENANCE_INVALID`; verifier `valid=true`
  - failure: `extension/plugin load is absent from runtime attestation` at `extension_load`
  - evidence:
    [h0_phase_a_524f7e3b…](research/evidence/h0_phase_a_524f7e3b88f73bc366d467d53a2c393a7d3ba937/)
    + [execution witness](research/evidence/h0_r5_phase_a_execution_witness_20260725/)
  - S5 permanently spent; no faithful capture; no actual H0 guarantee; Phase B forbidden;
    no future reentry authorization from R5 alone.
  Detail: [claim-state registry](../../research/contracts/claim_state_registry.md).
- 2026-07-24 consumer re-charter closed at
  `GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`（`gctm_runtime_native_candidate_universe_v1`）；
  WIP released at terminal；**not** H0 implementation authority. See
  [closed charter](../../research/threads/closed/gctm_runtime_native_candidate_universe_task.md).
- 2026-07-24 registration-v3 closed at
  `H0_REGISTRATION_V3_CONTRACT_SEALABLE`（`h0_gctm_guarantee_registration_v3` /
  `quantity.h0_native_universe_completeness_registration`）；
  WIP released at terminal；**not** actual guarantee / capture / re-entry.
  H0-R4 Phase-A did **not** establish an actual registration-v3 guarantee.
  See [closed charter](../../research/threads/closed/h0_gctm_guarantee_registration_v3_universe_completeness_20260724.md).

## Proposed（non-WIP）

- **H0 route-5 B1 — runtime-grounded consumer-faithful operating curve** —
  `H0_ROUTE5_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; it is distinct
  from and coexists with `GCTM_B1` → [machine identity decision](../../research/contracts/gctm_b1_slot_identity_decision_v1.json)
- **GCTM B1 — runtime-grounded offline attribution and score-ranking evaluation** —
  `GCTM_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; it does not alias
  or supersede `H0_ROUTE5_B1` → [task charter](../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
- **GCTM O1 — online score intervention and system-efficacy evaluation** → [task charter](../../research/threads/gctm_o1_online_intervention_efficacy_task.md)
- **H2 — bridge-decision capture under behavioral runtime identity** — `proposed`,
  non-WIP; successor to H0's **identity layer only**（declaration §0.1 boundary;
  H0 sealed history / 五個 spent `S` / permanent ledger 全部不變）；authorizes no
  capture, no `I`/`F`/`S`, no re-entry；Phase-A Layer-M controller implemented,
  contract-tested, reviewed and merged（[PR #295](https://github.com/raylei50653/saccade/pull/295)，
  merge `b2f3c23f…`）⇒ S4 code-review gate closed；剩餘 gate 依序（Acceptance
  items 4→5→6）= seal-candidate head 的 coordinate/probe current + controlled-host
  綠 → 同一 head 的 Layer-P certificate + independent verification → separate
  owner exactly-once authorization；do not seal or execute →
  [task charter](../../research/threads/h2_behavioral_identity_capture_task.md)
  · [declaration](research/headline_bridge_behavioral_identity_capture_declaration_20260725.md)
  · owner decision surface [#286](https://github.com/raylei50653/saccade/issues/286)（四項決策；不授權任何執行）
- **(next only via separate owner decision)** any future H0 re-entry requires a new
  repair/seal/authorization chain; exact S=`6fdb060c…` is permanently spent.

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

## Done / closed

See the module research index, closed threads index, evidence ledger, and NO-GO registry; terminal details do not live in TODO.
