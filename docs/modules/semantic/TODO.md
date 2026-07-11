# Semantic Relink — 模組 TODO

> **WIP register only**（O0）。任務敘事 → [threads/](../../research/threads/)；事實 → [research/](research/) · [README](README.md)。  
> Dashboard：[DEVELOPMENT.md 模組現狀總覽](../../../DEVELOPMENT.md)。規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

🔄 **Gap-conditioned probabilistic motion probe — E3 `E3_SIGNALS_SEALED`; D0 fail-closed (`not_fidelity_aligned` / capture unavailable); Phase B unauthorized**

- Thread: [gap_conditioned_probabilistic_motion_probe_20260711.md](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)
- E0: [note](research/gap_conditioned_motion_e0_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e0_20260711/manifest.json)
- E1: [note](research/gap_conditioned_motion_e1_m0_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e1_m0_20260711/manifest.json)
- E2: [note](research/gap_conditioned_motion_e2_family_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e2_family_20260711/manifest.json)
- E3: [note](research/gap_conditioned_motion_e3_signals_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json) — 7 folds · 28 parameter + 7 selection · full cube 679,952 rows (`evaluation_role=held_out|train`, A6-complete) · no A1–A8
- **D0**（[Issue #112](https://github.com/raylei50653/saccade/issues/112) **incomplete**）: [note](research/d0_bridge_estimator_fidelity_20260711.md) · [packet](research/evidence/d0_bridge_estimator_fidelity_20260711/manifest.json) — status `D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE` · terminal **`not_fidelity_aligned`** · primary=`runtime_capture_unavailable` · reconstruction diagnostics only · production unchanged
- Substrate mapping（PR #111, canonical precondition）: [note](research/production_substrate_mapping_20260711.md) — headline=\(S_A=[1,26]\)（consumer A）· D0 gate fail-closed as above
- Phase B design（predeclared, execution unauthorized）: [A1–A8 protocol](research/gap_conditioned_motion_phase_b_design_20260711.md) — frozen criteria + V1–V5 rule; ordering = design seal → E3 signals sealed → **owner authorization** → single run
- Boundary: frozen pair table only · E3 signals sealed · D0 fail-closed · Phase B / A1–A8 + joint/velocity fail-closed · no production/default/global-closure change

## Previous line（closed; nav only）

**Safe-Region Assetization — A1 CLOSED (`A1_ACCEPTED_WITH_LIMITS`)**

- Thread: [safe_region_assetization_20260710.md](../../research/threads/closed/safe_region_assetization_20260710.md)
- **A1 terminal recorded 2026-07-11**: `A1_ACCEPTED_WITH_LIMITS` → **maturity A1**; 5 enumerated limits (no D1 trace · no second consumer · event-mass / non-productive-cell / predicate-alias queries need raw) — [terminal record](../../research/threads/closed/safe_region_assetization_20260710.md#a1-terminal-record-2026-07-11)
- Acceptance unit: conversion pack `1a180620bc…` (`out/signal_study/m_b1_5_safe_region_asset_r1_20260710/`); audit **26/26 PASS** + mutation sensitivity **5/5** — [audit note](research/safe_region_a1_audit_20260711.md)
- **R1 / R1.1 remain external diagnostic overlay** (not A1 objects; not pack consumers):
  - R1 V-C = heuristic-specific descriptive failure (LOO pool global-label-screened; class null retracted) — [note](research/safe_region_assetization_r1_20260710.md)
  - R1.1 = 2 unique harmful AND events + 3 descriptive symptoms; "primary F3" rejected — [note](research/safe_region_assetization_r11_20260710.md)
- Preserved state:
  ```text
  A1: CLOSED — A1_ACCEPTED_WITH_LIMITS (2026-07-11) · maturity A1
  next-line pick: DONE 2026-07-11 → GT-Support Morphology（見 Sole active）
    (R1.1 four lines = candidate directions; escape-tail 機制化吸收其
     role-reversal 症狀)
  R2–R4: unauthorized (fail-closed; accepting terminal ≠ stage authorization)
  grammar search / hook / production / ledger: closed / unchanged
  terminal B: retained (never rested on the overlay)
  ```
- **Do not** restart grammar search, LOO-tune the probe, treat probe `A1_region_asset` tags as pack maturity, or invent a post-hoc D1 rule.

## Parked

- GT-Support Morphology — PR-D / #107 `ACCEPTED_WITH_LIMITS`; restricted-closure prototype not started; accepted global=`{dist_h, log_h_ratio}` boundary preserved → [thread](../../research/threads/gt_support_morphology_20260711.md)
- Occ-exit conditional intervention modeling — WP1–WP3 complete; future RegionAsset producer/intervention consumer; resume only after assetization owner gate → [occ-exit thread](../../research/threads/occ_exit_audit_20260709.md)
- Sparse key-embedding bank — C++ async sidecar（#57 禁 sync）→ [sparse_key_embedding_bank_20260704.md](research/sparse_key_embedding_bank_20260704.md)

## Related research threads（不佔 WIP 鎖）

- [gap-conditioned probabilistic motion](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) — **ACTIVE**（sole active）· E0–E2 `ACCEPTED_WITH_LIMITS` · E3 `E3_SIGNALS_SEALED` · D0 fail-closed (`not_fidelity_aligned` / capture unavailable; Issue #112 incomplete) · Phase B unauthorized · A1–A8 not executed · probabilistic verdict `NOT_YET_EVALUATED`
- [gt-support morphology](../../research/threads/gt_support_morphology_20260711.md) — **PARKED** · PR-A/B/C sealed · PR-D / #107 **`ACCEPTED_WITH_LIMITS`** · restricted-closure prototype not started
- [safe-region assetization](../../research/threads/closed/safe_region_assetization_20260710.md) — **A1 CLOSED (`A1_ACCEPTED_WITH_LIMITS`, maturity A1)** · mainline handed off → gt-support morphology · [PR #95](https://github.com/raylei50653/saccade/pull/95)/[#97](https://github.com/raylei50653/saccade/pull/97) history
- [composition grammar coverage program](../../research/threads/closed/composition_grammar_coverage_program_20260710.md) — **SUPERSEDED** · coverage map absorbed into assetization R2–R4
- [composition grammar × safe-region A0](../../research/threads/closed/composition_grammar_safe_region.md) — **CLOSED** · T0-A/B/R1 · terminal B retained
  - A0 source note: [composition_grammar_t0_region_interpretation_20260710.md](research/composition_grammar_t0_region_interpretation_20260710.md)
  - Coverage audit: [composition_grammar_safe_region_coverage_audit_20260710.md](research/composition_grammar_safe_region_coverage_audit_20260710.md)
- [m_b1 online hook](../../research/threads/closed/m_b1_online_hook_20260709.md) — **CLOSED** · S1+S2 Q4.5 B · ranking deferred (invalid assignment-group key)
  - Offline history: [m_b1_research_history_20260709_20260710.md](research/m_b1_research_history_20260709_20260710.md)
  - S1 final: [m_b1_stage1_online_hook_final_20260710.md](research/m_b1_stage1_online_hook_final_20260710.md)
- [association recovery registry](../../research/threads/association_recovery_registry_20260709.md)

## Done / closed

See [module research index](README.md) · [chebgr signal map](research/chebgr_handover_signal_map_20260704.md) · [no_go_registry](../../reference/no_go_registry.md) · [online sparse handoff](research/online_sparse_reid_handoff_20260704.md)。

> Code path note: Cheb-GR / relink may live under `perception/reid/` / `tracking/`; **doc home = semantic**. ReID feature unlock → [reid TODO](../reid/TODO.md)。
