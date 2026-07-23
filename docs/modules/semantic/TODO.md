# Semantic Relink — 模組 TODO

> **WIP register only**（O0）：只鎖 decision-changing mainline charter。高階型別與 owner 路由見 [research control plane](../../research/README.md#research-control-plane)；Expected state / probe 在 linked charter；事實與結論在 [research](research/) / [README](README.md)。
> 開發路由：[DEVELOPMENT action cards](../../../DEVELOPMENT.md#agent-action-cards) · 規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

- ⏸️ **無 active**（2026-07-23 起）。GCTM charter 已於 owner 接受 ordered terminal
  `GCTM_MODEL_SPEC_SEALABLE` 後關閉 → [closed charter](../../research/threads/closed/gap_conditioned_stochastic_transition_model_task.md)
  · [Issue #175](https://github.com/raylei50653/saccade/issues/175);terminal 與
  limits 以該卡 *Final status* 與 [claim-state registry](../../research/contracts/claim_state_registry.md) 為準
  （本欄只作 WIP 投影,不複述 terminal 內容）。**該 seal 是 diagnostic-only**,
  不授權 data／fitting／H0／B1／O1／runtime／online／production,亦不改 H0 狀態;
  B1／O1 仍是 proposed,各自另有 gate。
- H0 closure（三個 owner-accepted ordered terminal `H0_PROVENANCE_INVALID`;
  faithful capture = none;actual H0 guarantee = none;Phase B forbidden;
  任何未來 re-entry 前置＝machine-checked launch-hygiene pre-authorization
  gate＋fresh I→F→S＋另行 exactly-once 授權）：state fact-owner 見
  [claim-state registry](../../research/contracts/claim_state_registry.md)。

## Proposed（non-WIP）

- **GCTM B1 — runtime-grounded offline attribution and score-ranking evaluation** → [task charter](../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
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

## Done / closed

See the module research index, closed threads index, evidence ledger, and NO-GO registry; terminal details do not live in TODO.
