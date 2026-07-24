# Closed research threads

**定位：** 已結案 thread 卡的**檔案家**。仍是 navigation-only，不是 evidence home。

```text
threads/         = proposed · active · parked
threads/closed/  = closed（本目錄）
docs/archive/    = archived（很少用；不再需要 threads 導航）
```

**索引權威：** 父層 [../README.md](../README.md) 的 **Closed** 表（terminal one-liner + close date）。  
本 README **只列檔名**，不複製 close date / terminal，避免第二真相。

## Cards

| File |
|:--|
| [gctm_runtime_native_candidate_universe_task.md](gctm_runtime_native_candidate_universe_task.md) |
| [h0_gctm_interface_static_feasibility_audit_20260723.md](h0_gctm_interface_static_feasibility_audit_20260723.md) |
| [gctm_d1_substrate_agnostic_ranking_diagnostic_task.md](gctm_d1_substrate_agnostic_ranking_diagnostic_task.md) |
| [gap_conditioned_stochastic_transition_model_task.md](gap_conditioned_stochastic_transition_model_task.md) |
| [bridge_frozen_evidence_o0_routing_20260716.md](bridge_frozen_evidence_o0_routing_20260716.md) |
| [gap_conditioned_probabilistic_motion_probe_20260711.md](gap_conditioned_probabilistic_motion_probe_20260711.md) |
| [safe_region_assetization_20260710.md](safe_region_assetization_20260710.md) |
| [composition_grammar_coverage_program_20260710.md](composition_grammar_coverage_program_20260710.md) |
| [composition_grammar_safe_region.md](composition_grammar_safe_region.md) |
| [m_b1_online_hook_20260709.md](m_b1_online_hook_20260709.md) |
| [runtime_faithful_safe_domain_20260712.md](runtime_faithful_safe_domain_20260712.md) |

## On move-in

見父層 [How to close a thread](../README.md#how-to-close-a-thread)：

1. frontmatter `doc-status: closed` + `closed: YYYY-MM-DD`
2. body Final status / History close line
3. `git mv` 進本目錄 + 修相對連結 + 全庫 path rewrite
4. 父 README **Closed** 表加行（date · terminal 只寫那裡）
5. 本表只加檔名一行

**Same-PR：** 結案移動必須連同 link 修復一併提交。
