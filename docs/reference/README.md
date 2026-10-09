# Reference

規格、設定、benchmark 數值、資料流圖。

| 文件 | 內容 |
|------|------|
| [ci_timing_baseline_509.md](ci_timing_baseline_509.md) | #509 前段：remote main/PR CI job/step、cache 與本機 pytest/build 的可追溯耗時基線；不承載跨 Issue 排程 |
| [saccade_module_reference.md](saccade_module_reference.md) | 模組化前背景筆記：既有能力、封裝與依賴邊界、native delivery、failure semantics 與 public/runtime surface；非設計方案 |
| [mot17_default_config.md](mot17_default_config.md) | MOT17 目前推薦 baseline（`mamba_whole_graph`）與 raw CLI fallback |
| [pipeline_flow.md](pipeline_flow.md) | 現行 eval stage 名稱與 source map；細節見 [DATAFLOW.md](../DATAFLOW.md) |
| [production_pipeline_code_map.md](production_pipeline_code_map.md) | 806c52cf source 閱讀快照：production call graph、tensor／演算法表、association passes 與同步邊界 |
| [native_runtime_packaging_audit.md](native_runtime_packaging_audit.md) | #465 Phase A：headline runtime 的 Python／native 邊界、依賴清單、預編譯包形式比較與 verdict（`native_package_blocked_by_bounded_python_surface`） |
| [native_runtime_shipping_boundary.md](native_runtime_shipping_boundary.md) | #465 Phase A.5：shipping／developer／eval 範圍凍結；固定 shipping entrypoint、U1–U6 與 Python 擁有語義的 scope matrix、shared_boundary 抽取提案、Phase B PR 順序 |
| [native_runtime_head_artifact.md](native_runtime_head_artifact.md) | #465 Phase B PR-1：headline Mamba head 的 TRT artifact（ONNX＋scan plugin，head-only、FP32）、fail-closed lineage 輸入、可重建檢查與已知限制 |
| [native_runtime_resolved_config.md](native_runtime_resolved_config.md) | #465 Phase B PR-3：headline resolved shipping config（`configs/shipping/mamba_whole_graph.resolved.json`）的 schema、值的來源（執行 oracle 自身敘述＋native 替身）、fail-closed 覆蓋檢查、匯出時的發現與限制 |
| [native_runtime_head_parity_declaration.md](native_runtime_head_parity_declaration.md) | #465 Phase B PR-2 預宣告：head parity 的凍結輸入、L1 tensor gross-error screen、L2 7-seq MOT 容差（以 `--no-compile` reference 定尺度，含 floor／cap）、validity gate 與窮盡 terminal |
| [native_runtime_phase_c_scope.md](native_runtime_phase_c_scope.md) | #465 Phase C：owner 決策 C-D1–C-D4（bundle attested 第三方集合、只支援 sm_120、relocatable tarball、Ubuntu 24.04 baseline）、授權讀法、已知技術問題、PR-C1–C4 拆分 |
| [native_runtime_closeout.md](native_runtime_closeout.md) | #465 收尾（#546）：`ENGINEERING_COMPLETE`／`PUBLIC_DISTRIBUTION_READY` 的定義、架構與信任邊界圖、宣稱 → PR／正式證據／限制／測試的驗收矩陣、公開散佈的未解項目與延後工作；建議 verdict `ENGINEERING_COMPLETE — PUBLIC_DISTRIBUTION_DEFERRED`（待 owner review） |
| [math_model.md](math_model.md) | 現行 baseline 的全局數學模型：GMC、Kalman、成本、auction、bridge relink |
| [math_model_implementation.md](math_model_implementation.md) | 修改模型時的實作流程、invariants、測試與文檔 checklist |
| [PIPELINE_REFERENCE.md](PIPELINE_REFERENCE.md) | 2026-05 legacy pipeline snapshot / module delta ledger（非現行 baseline） |
| [no_go_registry.md](no_go_registry.md) | NO-GO / parked / revived 方向索引：用途、訊號判定、證據連結 |
| [no_go_registry_details.md](no_go_registry_details.md) | NO-GO registry 長版歷史筆記與實驗細節保存 |
| [api_spec.md](../modules/storage/api_spec.md) | Redis 事件 / Chroma metadata / API contract |
| [concurrent_eval.md](concurrent_eval.md) | Concurrent eval 架構與限制 |
