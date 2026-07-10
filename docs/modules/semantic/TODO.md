# Semantic Relink — 模組 TODO

> **WIP register only**（O0）。任務敘事 → [threads/](../../research/threads/)；事實 → [research/](research/) · [README](README.md)。  
> Dashboard：[DEVELOPMENT.md 模組現狀總覽](../../../DEVELOPMENT.md)。規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

🔄 **Safe-Region Assetization — A1 research asset acceptance (A0 retained; eng. MERGED)**

- Thread: [safe_region_assetization_20260710.md](../../research/threads/safe_region_assetization_20260710.md)
- Delivery path: **PR-driven** (direct-agent dispatch retired)
- R1 engineering: **MERGED** — converter + unit tests + A0 pack tooling; history [PR #95](https://github.com/raylei50653/saccade/pull/95) (**review COMPLETE**)
- Current active gate: **A1 research asset acceptance** (chat-side / research-owner) — research-consumption, **not** engineering re-check
- A1 questions: semantic fidelity · research query utility · decision utility · reusable abstraction
- A1 terminals: `A1_ACCEPTED` / `A1_ACCEPTED_WITH_LIMITS` / `A1_REJECTED` (`WITH_LIMITS` must list residual raw-artifact queries)
- Engineering prerequisites already passed (do **not** re-run as A1): determinism · PK/FK · SHA seals · ID/threshold reconstruction · manifest flags
- Primary deliverable (landed): accepted R0-B RegionAsset contract + deterministic R1 packer (`scripts/tools/convert_safe_region_asset_r1.py`) + conversion note
- Accepted contract: [safe_region_asset_contract.md](../../research/eval/safe_region_asset_contract.md) — **R0-B: ACCEPTED**; RB1–RB9 PASS; E1 applied
- R1 conversion note: [research/safe_region_asset_r1_conversion_20260710.md](research/safe_region_asset_r1_conversion_20260710.md)
- Pack candidate: `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/` (**A0** observation-only; **not A1**)
- Preserved state:
  ```text
  R0-B: ACCEPTED
  R1 engineering delivery: MERGED
  PR #95 engineering review: COMPLETE
  current active gate: A1 research asset acceptance
  current maturity: A0 retained
  A1: not accepted
  R2–R4: unauthorized
  terminal B: unchanged
  production / ledger: unchanged
  ```
- Boundary: A0→A1 self-promotion, evaluator rerun/modification, threshold search, new research claims, G4–G7, LOO, shadow, hook, production, ledger, R0-C/R1.1 contract expansion, and R2 task creation remain unauthorized until after an A1 terminal + owner gate. PR merge does not grant research acceptance.

## Parked

- Occ-exit conditional intervention modeling — WP1–WP3 complete; future RegionAsset producer/intervention consumer; resume only after assetization owner gate → [occ-exit thread](../../research/threads/occ_exit_audit_20260709.md)
- Sparse key-embedding bank — C++ async sidecar（#57 禁 sync）→ [sparse_key_embedding_bank_20260704.md](research/sparse_key_embedding_bank_20260704.md)

## Related research threads（不佔 WIP 鎖）

- [safe-region assetization](../../research/threads/safe_region_assetization_20260710.md) — **ACTIVE** · R0-B accepted · R1 eng. MERGED · A0 retained · **A1 gate open** · [PR #95](https://github.com/raylei50653/saccade/pull/95) history
- [composition grammar coverage program](../../research/threads/composition_grammar_coverage_program_20260710.md) — **SUPERSEDED** · coverage map absorbed into assetization R2–R4
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
