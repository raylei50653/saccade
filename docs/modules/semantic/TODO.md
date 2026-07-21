# Semantic Relink — 模組 TODO

> **WIP register only**（O0）：只鎖 decision-changing mainline charter。高階型別與 owner 路由見 [research control plane](../../research/README.md#research-control-plane)；Expected state / probe 在 linked charter；事實與結論在 [research](research/) / [README](README.md)。
> 開發路由：[DEVELOPMENT action cards](../../../DEVELOPMENT.md#agent-action-cards) · 規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

- **none**。H0 re-entry repair（#224 amendment：build-tool binding → #227 repair：
  canonical controller-input member parity）於 2026-07-21 由 owner 授權 exactly-once
  執行,到達第二個同型 owner-accepted ordered terminal `H0_PROVENANCE_INVALID`
  （I₂=`31c9eee8`→F₂=`46539a2d`→S₂=`0da082a9`；死因＝preflight launch hygiene:
  `build/h0_phase_a exists at controller launch`,T0 未進、零 capture;independent
  verifier `valid=true`）。#224／#227 已關閉;歷史 I/F/S 與 evidence 不可變;
  controller retry／second invocation under S₂／Phase B forbidden;GCTM #175 維持
  PARKED（不因此自動啟動）;actual H0 guarantee = none。永久留帳結論與 re-entry
  precondition（先建 machine-checked launch-hygiene pre-authorization gate,再 fresh
  I→F→S＋另行 exactly-once 授權）見 [claim-state registry
  `quantity.bridge_capture_provenance`](../../research/contracts/claim_state_registry.md)。
- **none**（re-entry #3）。owner 於 2026-07-21 授權 exactly-once 執行 re-entry #3
  （I₃=`5a2d1de5`→F₃=`7895704c`→S₃=`3a6a9ec6`），到達第三個同型 owner-accepted
  ordered terminal `H0_PROVENANCE_INVALID`（PR #235 comment 5032610430;independent
  verifier `result=provenance_invalid, valid=true, rc=0`）。死因不同＝extension_load
  confinement-plan construction（capture 前;`provenance_ok=false` 為唯一 false predicate;
  capture children NOT_RUN;T0/T1 completed、T2a_0→T4 not_reached）;根因＝
  seal-event／runtime-binding identity incompatibility。invocation count=1、exactly-once
  授權永久消耗、exact S 永久 spent、retry／resume／second invocation forbidden。
  **faithful capture = none;actual H0 guarantee = none;candidate／guarantee sets 空;
  Phase B FORBIDDEN;GCTM #175 PARKED;無 repair／新 re-entry 授權**。詳見 [claim-state
  registry `reentry_terminal_history` re-entry #3](../../research/contracts/claim_state_registry.md)。

## Proposed（non-WIP）

- **GCTM B1 — runtime-grounded offline attribution and score-ranking evaluation** → [task charter](../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
- **GCTM O1 — online score intervention and system-efficacy evaluation** → [task charter](../../research/threads/gctm_o1_online_intervention_efficacy_task.md)

## Parked

- **PARKED — Gap-conditioned stochastic transition model task** → [Issue #175](https://github.com/raylei50653/saccade/issues/175) · activation: accepted H0 ordered terminal + owner scheduling · [task charter](../../research/threads/gap_conditioned_stochastic_transition_model_task.md)
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
