# Semantic Relink — 模組 TODO

> **WIP register only**（O0）。任務敘事 → [threads/](../../research/threads/)；事實 → [research/](research/) · [README](README.md)。  
> Dashboard：[DEVELOPMENT.md 模組現狀總覽](../../../DEVELOPMENT.md)。規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

🔄 **Safe-Region Assetization — A1 terminal decision (audit 26/26 PASS; owner records terminal)**

- Thread: [safe_region_assetization_20260710.md](../../research/threads/safe_region_assetization_20260710.md)
- **A1 acceptance unit locked** (2026-07-10 chat review): conversion pack `1a180620bc…` (`out/signal_study/m_b1_5_safe_region_asset_r1_20260710/`)
- **Read-only S0/S1/Q1/N1 audit PASS 26/26** (2026-07-11): [research/safe_region_a1_audit_20260711.md](research/safe_region_a1_audit_20260711.md) · `scripts/tools/run_safe_region_a1_audit.py` · `out/signal_study/safe_region_a1_audit_20260711/`
- **R1 / R1.1 downgraded to external diagnostic overlay** (not A1 objects; not pack consumers):
  - R1 V-C = heuristic-specific descriptive failure (LOO pool global-label-screened; class null retracted) — [note](research/safe_region_assetization_r1_20260710.md)
  - R1.1 = 2 unique harmful AND events + 3 descriptive symptoms; "primary F3" rejected (post-hoc floors, K-duplicated count, alias-ambiguous predicate) — [note](research/safe_region_assetization_r11_20260710.md)
- Preserved state:
  ```text
  A1 acceptance unit: conversion pack 1a180620bc… (locked)
  A1 state: A1_PENDING_VALIDATION → audit 26/26 PASS
  maturity: A0 retained (terminal not yet recorded)
  next: owner records A1 terminal
    (A1_ACCEPTED_WITH_LIMITS supportable iff limits enumerated:
     no D1 trace · abstraction usage-based · event-mass/alias queries need raw)
  R1.1 four next-lines: deferred until terminal
  R2 / grammar search / hook / ledger: closed
  terminal B: retained (never rested on the overlay)
  ```
- **Do not** restart grammar search, LOO-tune the probe, or treat probe `A1_region_asset` tags as pack maturity.

## Parked

- Occ-exit conditional intervention modeling — WP1–WP3 complete; future RegionAsset producer/intervention consumer; resume only after assetization owner gate → [occ-exit thread](../../research/threads/occ_exit_audit_20260709.md)
- Sparse key-embedding bank — C++ async sidecar（#57 禁 sync）→ [sparse_key_embedding_bank_20260704.md](research/sparse_key_embedding_bank_20260704.md)

## Related research threads（不佔 WIP 鎖）

- [safe-region assetization](../../research/threads/safe_region_assetization_20260710.md) — **ACTIVE** · R0-B accepted · R1 eng. MERGED · A0 retained · **A1 audit PASS, terminal pending** · [PR #95](https://github.com/raylei50653/saccade/pull/95) history
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
