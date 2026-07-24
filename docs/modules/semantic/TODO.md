# Semantic Relink — 模組 TODO

> **WIP register only**（O0）：只鎖 decision-changing mainline charter。高階型別與 owner 路由見 [research control plane](../../research/README.md#research-control-plane)；Expected state / probe 在 linked charter；事實與結論在 [research](research/) / [README](README.md)。
> 開發路由：[DEVELOPMENT action cards](../../../DEVELOPMENT.md#agent-action-cards) · 規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

- **無 active**
- H0 R4 repair closed at `H0_R4_REPAIR_QUALIFIED_SEALABLE`
  （`h0_authority_overlay_runtime_binding_split_v1` / Amendment 10）；WIP released.
  Seal PR #277 landed exact S=`a76efffa…`. See
  [repair evidence](research/evidence/h0_r4_authority_overlay_runtime_binding_split_20260724/).
- **H0-R4 Phase-A executed once under sealed S（2026-07-24）** — facts only, not
  owner terminal acceptance:
  - `I=2a233387…` / `F=ced4a4cc…` / `S=a76efffa…`
  - authorization `h0_r4_phase_a_exactly_once_authorization_20260724`（Issue #278）
  - first+second launch-hygiene = `clear`; invocation count = 1; authorization consumed
  - controller `result=provenance_invalid`; mechanical disposition
    `H0_PROVENANCE_INVALID`; independent verifier `valid=true`, rc=0
  - evidence:
    [h0_phase_a_2a233387…](research/evidence/h0_phase_a_2a233387a6a321dd43570e2e30dc718571b3b4f4/)
    + [execution witness](research/evidence/h0_r4_phase_a_execution_witness_20260724/)
  - exact S permanently spent; retry/resume/second invocation forbidden; Phase B
    not authorized; actual guarantee = none; runtime compatibility = none;
    B1/O1 not activated. Owner acceptance of the truthful-negative terminal is
    this evidence PR merge surface only. No repair/reseal/new re-entry authorized.
  Detail: [claim-state registry `reentry_terminal_history` re-entry #4 / H0-R4](../../research/contracts/claim_state_registry.md).
- H0 closure baseline（三個 prior owner-accepted ordered terminal
  `H0_PROVENANCE_INVALID` + H0-R4 mechanical `H0_PROVENANCE_INVALID` pending this
  PR’s owner acceptance surface; faithful capture = none; actual H0 guarantee =
  none; Phase B forbidden; S3=`3a6a9ec6…` and S4=`a76efffa…` permanently spent）：
  state fact-owner 見
  [claim-state registry](../../research/contracts/claim_state_registry.md)。
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
- **(next owner choice after H0-R4 evidence PR)** terminal closeout acceptance of
  the truthful-negative `H0_PROVENANCE_INVALID` packet only; **or** a separate
  owner decision for any later repair / reseal / re-entry. This record does **not**
  authorize Phase B, actual registration-v3 guarantee audit as accepted, runtime
  compatibility, or B1/O1 activation.

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
