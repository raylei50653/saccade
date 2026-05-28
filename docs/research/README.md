# Experiments

活躍的實驗設計與分析。一次性數據報告歸檔至 [../archive/](../archive/)。

## pipeline/

| 文件 | 內容 |
|------|------|
| [gpu_pipeline_m4b_identity_resolver.md](pipeline/gpu_pipeline_m4b_identity_resolver.md) | M4b native identity resolver 設計 |

## tracking/

| 文件 | 內容 |
|------|------|
| [fp_fn_recovery_and_gmc.md](tracking/fp_fn_recovery_and_gmc.md) | FP/FN recovery + GMC 診斷 |
| [tentative_confirmed_state.md](tracking/tentative_confirmed_state.md) | Tentative/Confirmed 狀態機設計 |

## reid/

| 文件 | 內容 |
|------|------|
| [dynamic_trigger.md](reid/dynamic_trigger.md) | 動態 ReID 觸發機制設計 |
| [last_vit_integration_analysis.md](reid/last_vit_integration_analysis.md) | LaSt-ViT frequency-domain 診斷 |
| [semantic_relink_and_crop.md](reid/semantic_relink_and_crop.md) | Semantic relink + SigLIP2 crop 實驗 |

## eval/

| 文件 | 內容 |
|------|------|
| [fp_classifier_external_only_plan.md](eval/fp_classifier_external_only_plan.md) | 0-shot FP classifier 設計 |
| [wsl2_d2h_pinned_memory_leak_20260517.md](eval/wsl2_d2h_pinned_memory_leak_20260517.md) | WSL2 D2H memory leak post-mortem |

## 規範

- 一次性數據報告完成後移入 `archive/`
- 設計探索文件保留於此（仍在持續參考中）
