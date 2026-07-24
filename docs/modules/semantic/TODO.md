# Semantic Relink — 模組 TODO

> **WIP register only**（O0）：只鎖 decision-changing mainline charter。高階型別與 owner 路由見 [research control plane](../../research/README.md#research-control-plane)；Expected state / probe 在 linked charter；事實與結論在 [research](research/) / [README](README.md)。
> 開發路由：[DEVELOPMENT action cards](../../../DEVELOPMENT.md#agent-action-cards) · 規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

- **H0 R4 repair — authority-overlay / runtime-binding split** —
  `h0_authority_overlay_runtime_binding_split_v1`（Amendment 10）；decision-relevant
  WIP held for the Repair only. Positive: exact qualified I may proceed to a
  **separate Seal PR**. Negative: capture remains forbidden; owner chooses
  further repair or ABI-delta charter. Does **not** select I, create F/S,
  authorize execution/capture, register an actual guarantee, or activate
  B1/O1. See
  [repair evidence](research/evidence/h0_r4_authority_overlay_runtime_binding_split_20260724/).
- H0 closure baseline（三個 owner-accepted ordered terminal `H0_PROVENANCE_INVALID`;
  faithful capture = none; actual H0 guarantee = none; Phase B forbidden;
  S3=`3a6a9ec6…` permanently spent）：state fact-owner 見
  [claim-state registry](../../research/contracts/claim_state_registry.md)。
- 2026-07-24 consumer re-charter closed at
  `GCTM_RUNTIME_UNIVERSE_CONTRACT_SEALABLE`（`gctm_runtime_native_candidate_universe_v1`）；
  WIP released at terminal；**not** H0 implementation authority. See
  [closed charter](../../research/threads/closed/gctm_runtime_native_candidate_universe_task.md).
- 2026-07-24 registration-v3 closed at
  `H0_REGISTRATION_V3_CONTRACT_SEALABLE`（`h0_gctm_guarantee_registration_v3` /
  `quantity.h0_native_universe_completeness_registration`）；
  WIP released at terminal；**not** actual guarantee / capture / re-entry.
  Downstream target of a successful future Phase-A packet under this Repair.
  See [closed charter](../../research/threads/closed/h0_gctm_guarantee_registration_v3_universe_completeness_20260724.md).

## Proposed（non-WIP）

- **H0 route-5 B1 — runtime-grounded consumer-faithful operating curve** —
  `H0_ROUTE5_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; it is distinct
  from and coexists with `GCTM_B1` → [machine identity decision](../../research/contracts/gctm_b1_slot_identity_decision_v1.json)
- **GCTM B1 — runtime-grounded offline attribution and score-ranking evaluation** —
  `GCTM_B1`, `proposed`, `blocked_by: h0_runtime_substrate`; it does not alias
  or supersede `H0_ROUTE5_B1` → [task charter](../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
- **GCTM O1 — online score intervention and system-efficacy evaluation** → [task charter](../../research/threads/gctm_o1_online_intervention_efficacy_task.md)
- **(next owner choice after R4 terminal)** separate Seal PR on the exact
  qualified repair head, or exact ABI-delta charter, or further repair contract

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
