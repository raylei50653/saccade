# Semantic Relink — 模組 TODO

> **WIP register only**（O0）。任務敘事 → [threads/](../../research/threads/)；事實 → [research/](research/) · [README](README.md)。  
> Dashboard：[DEVELOPMENT.md 模組現狀總覽](../../../DEVELOPMENT.md)。規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

🔄 **Safe-Region Assetization — R0-B-R2 typed realization integration**

- Thread: [safe_region_assetization_20260710.md](../../research/threads/safe_region_assetization_20260710.md)
- Accepted preflight: [research/safe_region_r0_asset_contract_preflight_20260710.md](research/safe_region_r0_asset_contract_preflight_20260710.md) (R0-A ACCEPTED; CR1–CR9)
- Mathematical contract: [Statistical Robust Feasible-Set Estimation under Asymmetric Loss](../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
- Boolean semantics: [Boolean Composition Semantics Contract](../../research/eval/boolean_composition_semantics_contract.md)
- Deliverable: [safe_region_asset_contract.md](../../research/eval/safe_region_asset_contract.md) (RB1–RB4 PASS; RB5–RB7 current)
- Boundary: R1/A1 unauthorized; A0 retained; claims G1 1×L0, G2 6×L0+19×L1, G3 1×L0, pack ceiling L1; no asset pack, evaluator, grammar extension, LOO, shadow, hook, production, or ledger work

## Parked

- Occ-exit conditional intervention modeling — WP1–WP3 complete; future RegionAsset producer/intervention consumer; resume only after assetization owner gate → [occ-exit thread](../../research/threads/occ_exit_audit_20260709.md)
- Sparse key-embedding bank — C++ async sidecar（#57 禁 sync）→ [sparse_key_embedding_bank_20260704.md](research/sparse_key_embedding_bank_20260704.md)

## Related research threads（不佔 WIP 鎖）

- [safe-region assetization](../../research/threads/safe_region_assetization_20260710.md) — **ACTIVE** · R0-B-R2 typed realization integration
- [composition grammar coverage program](../../research/threads/composition_grammar_coverage_program_20260710.md) — **SUPERSEDED** · producer gates absorbed into R2–R4
- [composition grammar × safe-region A0](../../research/threads/composition_grammar_safe_region.md) — **CLOSED** · T0-A/B/R1 · terminal B retained
  - A0 source note: [composition_grammar_t0_region_interpretation_20260710.md](research/composition_grammar_t0_region_interpretation_20260710.md)
  - Coverage audit: [composition_grammar_safe_region_coverage_audit_20260710.md](research/composition_grammar_safe_region_coverage_audit_20260710.md)
- [m_b1 online hook](../../research/threads/m_b1_online_hook_20260709.md) — **CLOSED** · S1+S2 Q4.5 B · ranking deferred (invalid assignment-group key)
  - Offline history: [m_b1_research_history_20260709_20260710.md](research/m_b1_research_history_20260709_20260710.md)
  - S1 final: [m_b1_stage1_online_hook_final_20260710.md](research/m_b1_stage1_online_hook_final_20260710.md)
- [association recovery registry](../../research/threads/association_recovery_registry_20260709.md)

## Done / closed

See [module research index](README.md) · [chebgr signal map](research/chebgr_handover_signal_map_20260704.md) · [no_go_registry](../../reference/no_go_registry.md) · [online sparse handoff](research/online_sparse_reid_handoff_20260704.md)。

> Code path note: Cheb-GR / relink may live under `perception/reid/` / `tracking/`; **doc home = semantic**. ReID feature unlock → [reid TODO](../reid/TODO.md)。
