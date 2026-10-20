# Reference

規格、設定、benchmark 數值、資料流圖。

| 文件 | 內容 |
|------|------|
| [model_runtime_inventory_549.md](model_runtime_inventory_549.md) | #549 S0：native unchanged-model freeze 與其他 runtime 模型入口的逐檔 bytes／SHA、producer／consumer、權利路由盤點（[JSON](model_runtime_inventory_549.json)）；非 S1 bundle schema |
| [native_runtime_package_repin_536.md](native_runtime_package_repin_536.md) | #536 package re-pin：固定 entrypoint bytes、installed Completion／Gate A 控制與正式 package GPU parity 的契約／證據 |
| [model_bundle_s2_1_549.md](model_bundle_s2_1_549.md) | #549 S2-1：`--model-bundle` manifest mode、TR-1b allowlist（空）、resolved bindings、Gate B `load_verification`、journal v3／report v5 的 source-level 實作（[#576](https://github.com/raylei50653/saccade/pull/576) merged）與分層驗證紀錄（[coverage](model_bundle_s2_1_coverage_549.json)）；非 installed package re-pin |
| [model_bundle_s2_2_549.md](model_bundle_s2_2_549.md) | #549 S2-2：engine／head 從 buffer 載入、manifest 模式的版本／SM／TRT I/O gate 的 source-level 實作（[#582](https://github.com/raylei50653/saccade/pull/582) merged）與分層驗證紀錄（[evidence](model_bundle_s2_2_549.evidence.json)）；S2-2 PARTIAL |
| [model_bundle_s2_2_n03_offline_proposal_549.md](model_bundle_s2_2_n03_offline_proposal_549.md) | #549 S2-2 N03：P-N03 rev 3d 離線 admission＋compiled pin 提案（owner 已接受 O-N03-1..3） |
| [model_bundle_s2_2_n03_phase2_549.md](model_bundle_s2_2_n03_phase2_549.md) | #549 S2-2 N03 Phase 2：sealed-memfd operator 載入的實作與證據（[#584](https://github.com/raylei50653/saccade/pull/584) merged，[evidence](model_bundle_s2_2_n03_phase2_549.evidence.json)）；MB-41 OPEN、整體 `hashed_before_load` |
| [model_fetch_610.md](model_fetch_610.md) | #610 第二片：`scripts/model/model_fetch.py` 的 CLI 契約（使用者選定的 https 來源、expected sha256、原子發布、離線 `verify`）、操作說明與測試矩陣；不是來源認證、不是 Gate A／`--check-bundle`，Ultralytics 隱性下載尚未處理 |
| [model_bundle_mb41_f1_549.md](model_bundle_mb41_f1_549.md) | #549 MB-41：N02／LibTorch 2.11.0+cu130 的 fail-closed head admission、五個 G-1 對照與歷史證據；#590 已裁決，F-1＋F-2 由 [#591](https://github.com/raylei50653/saccade/pull/591) 合併（`c78f1319`），#592 為獨立 coordinate 審查紀錄；MB-41 依有界宣告 CLOSED，manifest aggregate `loaded_buffer` |
| [model_bundle_mb41_549.md](model_bundle_mb41_549.md) | #549 MB-41：歷史條件核對（G-1）與 #590 已確認的有界契約／整合路徑；新候選驗證與 main 狀態分開，歷史結果不驗證 #591 的 F-1＋F-2 |
| [native_identity_observability_549.md](native_identity_observability_549.md) | #549 S2-1a：legacy checksum identity 的 source-level 實作（[#572](https://github.com/raylei50653/saccade/pull/572) merged）、六項 binding／SHA source、journal v2／report v4 與分層驗證紀錄；非 installed package re-pin |
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
