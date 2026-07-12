# Semantic Relink — 模組 TODO

> **WIP register only**（O0）。任務敘事 → [threads/](../../research/threads/)；事實 → [research/](research/) · [README](README.md)。  
> Dashboard：[DEVELOPMENT.md 模組現狀總覽](../../../DEVELOPMENT.md)。規則：[DOC_MAINTENANCE § WIP](../../DOC_MAINTENANCE.md) · [契約 C7](../../ownership/doc_structure_contract.md)。

## Sole active

**Score temporal-to-stable-domain modeling — discrete-\(M\) declaration in `draft`, owner seal pending**

- Thread: [score_temporal_to_stable_domain_20260712.md](../../research/threads/score_temporal_to_stable_domain_20260712.md)
- Canonical: [research charter](research/score_temporal_to_stable_domain_20260712.md)
- **R1 closed** at terminal [`R1_FAITHFUL`](research/r1_temporal_reduction_capture_results_20260712.md)
  ([packet](research/evidence/r1_temporal_reduction_capture_20260712/manifest.json)) under sealed
  Consumer-A headline adaptive-anchor config + seven-seq MOT17 support.
- Only current work: owner review / seal of the
  [discrete-\(M\) declaration](research/discrete_m_capability_declaration_20260712.md)
  (`m0_state_capture_v1` per-frame capture; \(z_{t+1}\approx Mz_t+c\) on the
  lost-side \(z^{R}\); scope = **anchor propagation**, terminals `ANCHOR_*`).
  Rev. 2 after owner `REQUEST_CHANGES`: eligibility-before-ceiling, all-horizon
  ceiling, finite-horizon \(M^k\) stability, pinned float64 SVD solver, and 0.40
  demoted to a **heuristic** ceiling (no `bdist` quantity-equivalence claim).
  **Nothing may be captured, exported, or fitted before the seal event
  (declaration § 12)**; no score, gate, preset, or \(e^{A\Delta t}\) work is
  authorized by R1 alone.

## Previous line 0（closed; nav only）

**Ambiguous-band ranking-power probe (Door 0) — `T2 NO_USABLE_RANKING_POWER_IN_CLASS` ACCEPTED**

- Thread（closed）: [ambiguous_band_ranking_power_probe_20260712.md](../../research/threads/closed/ambiguous_band_ranking_power_probe_20260712.md)
- Declaration（sealed via PR #135, main `f864a6e2`）: [declaration doc](research/ambiguous_band_ranking_power_probe_declaration_20260712.md)
- **Results（accepted `9ec583c7`）**: [results note](research/door0_ranking_probe_results_20260712.md) · [packet](research/evidence/door0_ranking_probe_20260712/manifest.json) — V1–V5 PASS · H not triggered（baseline PWA 0.878/top-1 0.590;P3=84/205）· 12/12 fail（best ΔPWA=+0.001097,距 B1 約 18×）· unconditional reachable-set caveat
- **Closure in effect（step ⑤ class-scoped）**: 12 members 不得在相同 family/substrate 重跑;9 個未測 AND pair/其他 quantile/連續訊號/有限 λ/learned score 不被耗盡;step ④ 未開
- Boundary: production / preset / ledger unchanged · 單次授權執行已用畢,無宣告偏離

## Previous line（Phase B V5 recorded; D0 fail-closed pending [#112](https://github.com/raylei50653/saccade/issues/112); nav only）

**Gap-conditioned probabilistic motion probe — Phase B `V5 ACCEPTED_WITH_LIMITS` (representation / level 1); D0 fail-closed; no production authorization**

- Thread: [gap_conditioned_probabilistic_motion_probe_20260711.md](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md)
- E0: [note](research/gap_conditioned_motion_e0_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e0_20260711/manifest.json)
- E1: [note](research/gap_conditioned_motion_e1_m0_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e1_m0_20260711/manifest.json)
- E2: [note](research/gap_conditioned_motion_e2_family_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e2_family_20260711/manifest.json)
- E3: [note](research/gap_conditioned_motion_e3_signals_20260711.md) · [packet](research/evidence/gap_conditioned_motion_e3_signals_20260711/manifest.json) — 7 folds · 28 parameter + 7 selection · full cube 679,952 rows (`evaluation_role=held_out|train`, A6-complete) · no A1–A8
- **Phase B**: [note](research/gap_conditioned_motion_phase_b_20260711.md) · [packet](research/evidence/gap_conditioned_motion_phase_b_20260711/manifest.json) — A1–A8 single sealed-cube run → **`V5 ACCEPTED_WITH_LIMITS`** · representation / level-1 ceiling · no five-box contract · production unchanged / unauthorized
- **D0**（[Issue #112](https://github.com/raylei50653/saccade/issues/112) **incomplete**）: [note](research/d0_bridge_estimator_fidelity_20260711.md) · [packet](research/evidence/d0_bridge_estimator_fidelity_20260711/manifest.json) — status `D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE` · terminal **`not_fidelity_aligned`** · primary=`runtime_capture_unavailable` · reconstruction diagnostics only · production unchanged
- Substrate mapping（PR #111, canonical precondition）: [note](research/production_substrate_mapping_20260711.md) — headline=\(S_A=[1,26]\)（consumer A）· D0 gate fail-closed as above
- Phase B design（predeclared; executed without criterion deviation）: [A1–A8 protocol](research/gap_conditioned_motion_phase_b_design_20260711.md) — frozen criteria + V1–V5 rule; sealed entrypoint verified
- Boundary: frozen pair table only · E3 signals sealed · Phase B `V5 ACCEPTED_WITH_LIMITS` · D0 fail-closed · joint/velocity fail-closed · no production/default/global-closure change

## Previous line 2（closed; nav only）

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

- [ambiguous-band ranking-power probe](../../research/threads/closed/ambiguous_band_ranking_power_probe_20260712.md) — **CLOSED** · Door 0 · **`T2 NO_USABLE_RANKING_POWER_IN_CLASS` ACCEPTED**（#135 seal／#136 acceptance）· step ⑤ class-scoped closure 生效
- [gap-conditioned probabilistic motion](../../research/threads/gap_conditioned_probabilistic_motion_probe_20260711.md) — **ACTIVE**（不佔 WIP;剩 D0/#112 engineering follow-up）· E0–E2 `ACCEPTED_WITH_LIMITS` · E3 `E3_SIGNALS_SEALED` · Phase B **`V5 ACCEPTED_WITH_LIMITS`** (representation / level 1) · D0 fail-closed · no production change
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
