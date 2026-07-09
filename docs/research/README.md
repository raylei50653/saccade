# Experiments

活躍的實驗設計與分析。一次性數據報告歸檔至 [../archive/](../archive)。

## Workstream WIP（一模主一目標）

| 文件 | 內容 |
|------|------|
| [../DOC_MAINTENANCE.md](../DOC_MAINTENANCE.md) § Workstream WIP | **WIP=1** 規則：每模組 sole active；開第二目標須先 park/收合 |
| [../../DEVELOPMENT.md](../../DEVELOPMENT.md) §6 | 模組現狀 dashboard（每 🔄 一目標） |
| [tracker-decision/status_2026-07-09.md](tracker-decision/status_2026-07-09.md) | 已結案線入口（P0–P8 closed；勿與 dual-stability 並開） |

## pipeline/

| 文件 | 內容 |
|------|------|
| [mot17_mamba_whole_graph_m_sdp_double_buffer.md](pipeline/mot17_mamba_whole_graph_m_sdp_double_buffer.md) | `mamba_whole_graph_m` + SDP + double-buffer 實際 runtime 路徑 / 檔案對應 runbook |
| [gpu_pipeline_m4b_identity_resolver.md](pipeline/gpu_pipeline_m4b_identity_resolver.md) | M4b native identity resolver 設計 |
| [perf_attribution_whole_graph_m.md](pipeline/perf_attribution_whole_graph_m.md) | whole_graph_m 每幀開銷歸因 + 優化 backlog（nsys, 2026-07） |

## Research narrative (P8)

| 文件 | 內容 |
|------|------|
| [tracker-decision/status_2026-07-09.md](tracker-decision/status_2026-07-09.md) | 一頁式 production / contract / dual-stab / NO-GO 總結 |
| [paper_outline.md](paper_outline.md) | arXiv / technical report **skeleton only** |
| [evidence_ledger.md](evidence_ledger.md) | 可引用 metrics + decision outcomes 總表 |

## tracker-decision/

| 文件 | 內容 |
|------|------|
| [README.md](tracker-decision/README.md) | Tracker 決策層範圍、與 pipeline 分工、文件索引 |
| [status_2026-07-09.md](tracker-decision/status_2026-07-09.md) | P8 一頁式現況（keep both + m cost nuance） |
| [scoring_semantics.md](tracker-decision/scoring_semantics.md) | Association cost / gate / weight / identity 語意 |
| [assoc_knobs.md](tracker-decision/assoc_knobs.md) | 決策旋鈕卡片（ACTIVE + NO-GO） |
| [relink_bridge.md](tracker-decision/relink_bridge.md) | Geometry-only bridge relink |
| [kalman_gmc_motion.md](tracker-decision/kalman_gmc_motion.md) | Kalman / GMC 對 matching 的假設 |
| [failure_modes.md](tracker-decision/failure_modes.md) | Geometry 側失敗模式 |
| [audit/config_surface.md](tracker-decision/audit/config_surface.md) | 跨模組決策參數面 |
| [audit/callpoints.md](tracker-decision/audit/callpoints.md) | schema → inject → native 對照 |
| [audit/native_bridge.md](tracker-decision/audit/native_bridge.md) | Python↔CUDA bridge / rename 風險 |
| [audit/math_model_drift_2026-07-09.md](tracker-decision/audit/math_model_drift_2026-07-09.md) | math_model.md 靜態 drift audit（P3） |
| [audit/dual_stability_ablation_results_2026-07-09.md](tracker-decision/audit/dual_stability_ablation_results_2026-07-09.md) | P7 dual stability A–D（s+m） |

## tracking/

| 文件 | 內容 |
|------|------|
| [fp_fn_recovery_and_gmc.md](../modules/geometry/research/fp_fn_recovery_and_gmc.md) | FP/FN recovery + GMC 診斷 |
| [tentative_confirmed_state.md](../modules/lifecycle/research/tentative_confirmed_state.md) | Tentative/Confirmed 狀態機設計 |

## reid/

| 文件 | 內容 |
|------|------|
| [dynamic_trigger.md](../modules/trigger/research/dynamic_trigger.md) | 動態 ReID 觸發機制設計 |
| [last_vit_integration_analysis.md](../modules/reid/research/last_vit_integration_analysis.md) | LaSt-ViT frequency-domain 診斷 |
| [semantic_relink_and_crop.md](../modules/reid/research/semantic_relink_and_crop.md) | Semantic relink + SigLIP2 crop 實驗 |

## eval/

| 文件 | 內容 |
|------|------|
| [fp_classifier_external_only_plan.md](eval/fp_classifier_external_only_plan.md) | 0-shot FP classifier 設計 |
| [wsl2_d2h_pinned_memory_leak_20260517.md](eval/wsl2_d2h_pinned_memory_leak_20260517.md) | WSL2 D2H memory leak post-mortem |

## 規範

- 一次性數據報告完成後移入 `archive/`
- 設計探索文件保留於此（仍在持續參考中）
