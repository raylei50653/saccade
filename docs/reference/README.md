# Reference

規格、設定、benchmark 數值、資料流圖。

| 文件 | 內容 |
|------|------|
| [saccade_module_reference.md](saccade_module_reference.md) | 模組化前背景筆記：既有能力、封裝與依賴邊界、native delivery、failure semantics 與 public/runtime surface；非設計方案 |
| [mot17_default_config.md](mot17_default_config.md) | MOT17 目前推薦 baseline（`mamba_whole_graph`）與 raw CLI fallback |
| [pipeline_flow.md](pipeline_flow.md) | 現行 eval stage 名稱與 source map；細節見 [DATAFLOW.md](../DATAFLOW.md) |
| [production_pipeline_code_map.md](production_pipeline_code_map.md) | 806c52cf source 閱讀快照：production call graph、tensor／演算法表、association passes 與同步邊界 |
| [math_model.md](math_model.md) | 現行 baseline 的全局數學模型：GMC、Kalman、成本、auction、bridge relink |
| [math_model_implementation.md](math_model_implementation.md) | 修改模型時的實作流程、invariants、測試與文檔 checklist |
| [PIPELINE_REFERENCE.md](PIPELINE_REFERENCE.md) | 2026-05 legacy pipeline snapshot / module delta ledger（非現行 baseline） |
| [no_go_registry.md](no_go_registry.md) | NO-GO / parked / revived 方向索引：用途、訊號判定、證據連結 |
| [no_go_registry_details.md](no_go_registry_details.md) | NO-GO registry 長版歷史筆記與實驗細節保存 |
| [api_spec.md](../modules/storage/api_spec.md) | Redis 事件 / Chroma metadata / API contract |
| [concurrent_eval.md](concurrent_eval.md) | Concurrent eval 架構與限制 |
