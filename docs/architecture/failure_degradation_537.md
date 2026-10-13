<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-13 -->
<!-- doc-module: cross -->

# 失敗／fallback／降級現況基線（#537 C0）

本稿是 [#537](https://github.com/raylei50653/saccade/issues/537) 的 **C0 checkpoint**：只**固定現況與失敗證據**，分 profile 記下「能力缺失、受控降級、監控 unknown」目前實際會產生什麼結果，並用 repo 內的特徵測試把這些結果釘住。**本稿不修復任何行為，也不選定失敗語義**；已採納的方向只引用 §4 的 owner 裁決，由後續 PR 各自實作。閱讀入口是 [architecture README](README.md)。

**Source baseline**：`main@8350f6f7bc591f313d9a37996b33d7e77f66a61f`（#594 merge，2026-10-13）。§2／§3 標 `reproduced` 的列，都以 §5 的特徵測試在此 baseline 執行過；除 C0-01 需要 CUDA 外都不需要 GPU。標 `source-inspected` 的列只讀過 source（其中 `main.py` 的建構順序另有靜態 AST 測試釘住，但沒有執行），沒有執行完整 service、GPU 或 package。

**本稿不擁有的內容**（只引用）：CAP／REQ 與支持裁決歸 [#535 ledger](capability_requirements_535.md)；節點／邊 ID 與 B0 發現歸 [跨 profile 總覽](system_overview_536.md)；shipping run 狀態值與 completion 接口歸 [CC-536-01-01](ship_export_contracts_536.md#cc-536-01-01)（#537 只擁有它的轉換語義，C0 未重查）；freshness、writer restart 與資源所有權歸 [#538](https://github.com/raylei50653/saccade/issues/538)；tracker public surface 歸 [#441](https://github.com/raylei50653/saccade/issues/441)；requirement↔check 對應與 as-built 驗收歸 [#541](https://github.com/raylei50653/saccade/issues/541)。狀態用語沿用[總覽 §1](system_overview_536.md#1-狀態用語)。

## 1. 要區分的五種結果

#537 要求的五種結果，在 C0 baseline 中能否分辨：

| 結果 | C0 baseline 能否分辨 | 主要缺口（§2 的 ID） |
|:--|:--|:--|
| 真的沒有 tracks（合法空幀） | 不能和「能力缺失」分辨 | C0-01 |
| 能力不可用 | 依路徑不同：回空、拋例外，或延到第一個有 detection 的 batch 才失敗 | C0-01、C0-02、C0-03 |
| 受控降級 | 有四個 level，但轉換和文件不一致 | C0-04 |
| 監控 unknown | 沒有 unknown 狀態，NVML 失敗與 SHM 缺失都表現為 NORMAL | C0-05、C0-06、C0-09 |
| 功能已退役 | 不在本單範圍，歸 #539 | — |

## 2. Capability／failure 矩陣（observed as-is）

| ID | Profile／節點 | 失敗條件 | 觀察到的結果 | 可觀測性 | 驗證 | REQ／下游 |
|:--|:--|:--|:--|:--|:--|:--|
| C0-01 | public-library、eval／SN-PL2 | native 擴充缺失，有 CUDA | facade `GPUByteTracker.update()` 對 3 個或 0 個 detection 都回 `[]` | 只有 `is_cuda == False`；呼叫端若不查就看不出差別 | reproduced（gpu 標記） | REQ-535-02-01、02-03；#441 |
| C0-02 | public-library、eval／SN-PL2 | native 擴充缺失，沒有 CUDA | `update()` 在進入 stub 前就拋 `RuntimeError`；訊息依 driver 環境而不同（例如 `No CUDA GPUs are available` 或 `Found no NVIDIA driver ...`），測試只釘住例外型別 | 例外 | reproduced | REQ-535-02-01 |
| C0-03 | online-service／SN-ON3（B0-F02） | native 擴充缺失 | `AsyncDispatcher` 直接 import 失敗後，名稱綁成 `typing.Any`；`get_tracker()` 拋 `TypeError: Any cannot be instantiated`。`_process_batch` 只在 batch 有 detection 時才建 tracker，所以空幀 batch 不會觸發這個錯誤 | 例外，出現時間依輸入而定 | reproduced（`get_tracker`）；觸發時間為 source-inspected | REQ-535-02-04 |
| C0-04 | online-service／SN-ON7 | VRAM usage 接近門檻 | 轉換表見 §3：FAST_PATH 實質上沒有遲滯，EMERGENCY 的轉換不單調，在 91～92 會一次跳兩級 | dispatcher `get_stats()["current_level"]`；轉換時 stdout 有一行 print | reproduced | REQ-535-07-02；D-537-2 |
| C0-05 | online-service／SN-ON7 | NVML init 失敗 | `get_stats()` 全部回 0。目前若為 EMERGENCY，連續 3 次決策會依序降為 FAST_PATH → REDUCED → NORMAL | 只有 init 時一行 print「Degradation disabled」；level 顯示 NORMAL | reproduced | REQ-535-07-01 |
| C0-06 | online-service／SE-23、SE-24 | 沒有 writer（SHM segment 不存在） | `VRAMLevelReader.read()` 回 NORMAL；writer 之後才建立時，reader 能重新附掛 | 無 | reproduced | REQ-535-07-01 |
| C0-07 | online-service／SE-24 | segment 內的 byte 超出 0～3 | `read()` 拋 `ValueError`。這個呼叫位於 orchestrator `handle_cognitive_event` 內，對事件處理的影響未追 | 例外 | reproduced（reader）；consumer 影響未驗證 | REQ-535-07-01 |
| C0-08 | online-service／SE-23、SE-24（**B0-F07**） | 前次崩潰留下 segment，而且 reader 比 writer 先附掛 | writer unlink 並重建後，舊 reader 一直讀到舊值（測試中為 EMERGENCY）；新建立的 reader 讀得到新值；writer 關閉後，舊 reader 仍讀到最後的值。`main.py --mode full` 把 `PipelineOrchestrator()` 當 `gather()` 的參數，依 source 判讀 reader 會比 `dispatcher.start()` 先建立。卡在 EMERGENCY 時，`handle_cognitive_event()` 對非 anomaly 事件不寫入 ChromaDB 就返回，`process_event_batch()` 仍 ACK 整批，這些事件不會重送 | 無；level 看起來有效 | SHM 行為 reproduced；`main.py` 建構順序為 source-inspected（AST 測試靜態釘住，沒有執行 `run_full()`）；「orchestrator 持續丟棄非 anomaly 事件且仍 ACK」為 source-inspected（`src/saccade/cognition/orchestrator.py` 的 `handle_cognitive_event()`／`process_event_batch()`，沒有端到端測試） | REQ-535-07-01；#538 判準；D-537-3 |
| C0-09 | online-service／SN-ON3 | WorkbenchPool 路徑 | `get_stats()` 固定回 `current_level: "NORMAL"`，這條路徑沒有接 resource manager | stats 看起來健康 | source-inspected | REQ-535-07-02 |
| C0-10 | shipping／SN-SH2、SN-SH4 | run 中途失敗 | 狀態值與 completion journal 由 CC-536-01-01 定義並已實作（source-level）；轉換語義歸 #537，C0 沒有重查 | journal `state`、exit code | 引用原卡 | REQ-535-01-01 |

## 3. 降級轉換表：現況與已採納目標

單次 `decide_degradation_level()` 的結果。升級門檻是嚴格大於（85／92／96）；目標欄依 D-537-2（§4）：升級立即生效，可以跨級；恢復時每次決策只降一級，而且 usage 必須嚴格低於目前級別的門檻 −5。

| 目前級別 | usage 範圍 | 現況 | D-537-2 目標 | 是否一致 |
|:--|:--|:--|:--|:--|
| NORMAL | ≤85／(85, 92]／(92, 96]／>96 | NORMAL／REDUCED／FAST_PATH／EMERGENCY | 相同 | 一致 |
| REDUCED | <80／[80, 92]／(92, 96]／>96 | NORMAL／REDUCED／FAST_PATH／EMERGENCY | 相同 | 一致 |
| FAST_PATH | <87 | REDUCED | REDUCED | 一致 |
| FAST_PATH | [87, 92] | **REDUCED** | FAST_PATH | **不一致**（issue 反例 @90） |
| FAST_PATH | (92, 96]／>96 | FAST_PATH／EMERGENCY | 相同 | 一致 |
| EMERGENCY | <91 | FAST_PATH | FAST_PATH | 一致 |
| EMERGENCY | [91, 92] | **REDUCED**（跳兩級） | EMERGENCY | **不一致** |
| EMERGENCY | (92, 96] | **FAST_PATH** | EMERGENCY | **不一致**（issue 反例 @94） |
| EMERGENCY | >96 | EMERGENCY | EMERGENCY | 一致 |

C0 測試釘住的是「現況」欄。「目標」欄只是記錄已採納的方向，不是 C0 的斷言；對應的目標測試和實作屬於後續 C2 PR。

## 4. Owner 裁決（引用）

以下來自 [#537 C0 起步](https://github.com/raylei50653/saccade/issues/537#issuecomment-6139603145)，是 2026-10-13 工作 session 的轉錄；同一天也記入 [#550](https://github.com/raylei50653/saccade/issues/550#issuecomment-6139619832)。

- **D-537-1**：C0 為一條 Draft PR，只含文件矩陣和無 GPU 特徵測試，只固定現況與失敗證據，**不混入修復**；不改 runtime 行為；沒有 merge 授權。
- **D-537-2**：C2 的目標語義採 [degradation 文件 §5](../modules/resource/architecture_degradation.md)，內容如 §3。
- **D-537-3**：第一個修復目標是 B0-F07（C0-08），另開小 PR。具體方案要先給 owner 核准；freshness／restart 判準歸 #538，#537 只選 consumer policy。

## 5. 證據

| 測試 | 涵蓋 | 環境 |
|:--|:--|:--|
| [test_resource_degradation_observed_537.py](../../tests/unit/test_resource_degradation_observed_537.py) | C0-04～C0-08：§3 的現況欄（19 個單步案例，含兩個 issue 反例）、NVML 失敗、SHM 缺失／非法值、B0-F07 的 SHM 行為；`main.py --mode full` 的建構順序只以 AST 靜態檢查 | 不需 GPU；NVML 以 mock 取代；SHM 使用每個測試程序私有的 segment 名稱 |
| [test_native_missing_observed_537.py](../../tests/unit/test_native_missing_observed_537.py) | C0-01～C0-03 | 子程序中封鎖 `saccade_tracking_ext`；C0-02／C0-03 隱藏 CUDA；C0-01 標 `gpu`，在 CI 會略過 |

這兩個檔的 function 標籤都是 `diagnostic`：只觀察、不構成正式保證。後續修復 PR 若改變某列的行為，要在同一個 PR 更新對應測試與本稿，並說明這項改變是哪一條裁決授權的。

**限制**：沒有執行完整 online service、Redis／Chroma 或真實 VRAM 壓力；C0-07、C0-08 對 orchestrator 事件的影響、C0-08 的 `--mode full` 建構順序，以及 C0-03 的觸發時間，都沒有執行驗證。C0-01 需要 CUDA，只在本機執行過。沒有新增 runtime／GPU／installed-package qualification 或 equivalence 證據。

## 6. 未裁決項目與下一步

| 項目 | 需要的裁決 | 歸屬 |
|:--|:--|:--|
| C0-01～C0-03 的拒絕點 | 各 profile 在哪一步拒絕（import、construct 或 update），以及怎麼回報 | REQ-535-02-01／02-03／02-04；#441 |
| C0-05、C0-06 | NVML 不可用、沒有 writer 時，consumer 應該看到什麼（unknown、停用降級，或其他具名狀態） | REQ-535-07-01 |
| C0-07 | 超出範圍的 level 怎麼處理 | REQ-535-07-01 |
| C0-08 | freshness／restart 判準（#538）與 consumer policy（#537）；方案要先經 owner 核准 | D-537-3 |
| C0-09 | WorkbenchPool 是否支援 resource degradation | REQ-535-07-02 |
| C2 其他參數 | 重啟時的初值、非法 usage（NaN、負值、>100）、決策頻率 | D-537-2 未涵蓋 |
| health CLI、orchestrator 錯誤後的 ACK | 報告 unknown 的方式；ACK 語義 | REQ-535-07-03；REQ-535-06-01（#538） |

建議順序：先提出 C0-08（B0-F07）的修復方案給 owner，方案要包含事件持久化與 ACK 的負控制（非 anomaly 事件在舊 reader 下不得未寫入就被 ACK）；C2 依 D-537-2 另開 PR，附上目標測試和負控制。每個修復 PR 只處理一類行為。
