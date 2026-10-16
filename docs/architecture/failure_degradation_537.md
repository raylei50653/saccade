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
| C0-06 | online-service／SE-23、SE-24 | 沒有 writer（SHM segment 不存在） | `VRAMLevelReader.read()` 回 NORMAL；writer 之後才建立時，reader 能重新附掛 | 無 | reproduced | REQ-535-07-01；**已由 F07 修復，見 §7** |
| C0-07 | online-service／SE-24 | segment 內的 byte 超出 0～3 | `read()` 拋 `ValueError`。這個呼叫位於 orchestrator `handle_cognitive_event` 內，對事件處理的影響未追 | 例外 | reproduced（reader）；consumer 影響未驗證 | REQ-535-07-01；**已由 F07 修復，見 §7** |
| C0-08 | online-service／SE-23、SE-24（**B0-F07**） | 前次崩潰留下 segment，而且 reader 比 writer 先附掛 | writer unlink 並重建後，舊 reader 一直讀到舊值（測試中為 EMERGENCY）；新建立的 reader 讀得到新值；writer 關閉後，舊 reader 仍讀到最後的值。`main.py --mode full` 把 `PipelineOrchestrator()` 當 `gather()` 的參數，依 source 判讀 reader 會比 `dispatcher.start()` 先建立。卡在 EMERGENCY 時，`handle_cognitive_event()` 對非 anomaly 事件不寫入 ChromaDB 就返回，`process_event_batch()` 仍 ACK 整批，這些事件不會重送 | 無；level 看起來有效 | SHM 行為 reproduced；`main.py` 建構順序為 source-inspected（AST 測試靜態釘住，沒有執行 `run_full()`）；「orchestrator 持續丟棄非 anomaly 事件且仍 ACK」為 source-inspected（`src/saccade/cognition/orchestrator.py` 的 `handle_cognitive_event()`／`process_event_batch()`，沒有端到端測試） | REQ-535-07-01；#538 判準；D-537-3；**已由 F07 修復，見 §7** |
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

F07 修復後，第一個檔的 SHM 測試（C0-06～08）已改為目標語義，其餘維持現況特徵；F07 的目標測試見 §7。這兩個檔的 function 標籤都是 `diagnostic`：只觀察、不構成正式保證。後續修復 PR 若改變某列的行為，要在同一個 PR 更新對應測試與本稿，並說明這項改變是哪一條裁決授權的。

**限制**：沒有執行完整 online service、Redis／Chroma 或真實 VRAM 壓力；C0-07、C0-08 對 orchestrator 事件的影響、C0-08 的 `--mode full` 建構順序，以及 C0-03 的觸發時間，都沒有執行驗證。C0-01 需要 CUDA，只在本機執行過。沒有新增 runtime／GPU／installed-package qualification 或 equivalence 證據。

## 6. 未裁決項目與下一步

| 項目 | 需要的裁決 | 歸屬 |
|:--|:--|:--|
| C0-01～C0-03 的拒絕點 | 各 profile 在哪一步拒絕（import、construct 或 update），以及怎麼回報 | REQ-535-02-01／02-03／02-04；#441 |
| C0-05 | NVML 不可用時，consumer 應該看到什麼（unknown、停用降級，或其他具名狀態）；F07 後仍發佈 FRESH NORMAL | REQ-535-07-01 |
| C0-06～C0-08 | 已由 F07 修復（§7）；T 預設值、多 writer、shutdown 仍待 #538 | #538 |
| C0-09 | WorkbenchPool 是否支援 resource degradation | REQ-535-07-02 |
| C2 其他參數 | 重啟時的初值、非法 usage（NaN、負值、>100）、決策頻率 | D-537-2 未涵蓋 |
| health CLI、orchestrator 錯誤後的 ACK | 報告 unknown 的方式；ACK 語義 | REQ-535-07-03；REQ-535-06-01（#538） |

建議順序：先提出 C0-08（B0-F07）的修復方案給 owner，方案要包含事件持久化與 ACK 的負控制（非 anomaly 事件在舊 reader 下不得未寫入就被 ACK）；C2 依 D-537-2 另開 PR，附上目標測試和負控制。每個修復 PR 只處理一類行為。

## 7. B0-F07 修復（C0-06／07／08 收斂）

本節記錄 C0 之後的 B0-F07 修復 PR，§2 的矩陣保留為修復前的基線。方案經 Codex 獨立審查後，由 owner 在 2026-10-16 的工作 session 授權實作，並授權在審查與檢查都通過時合併（D-537-3）。新契約的完整內容（記錄格式、狀態、freshness、ownership、consumer policy、部署規則）在 [degradation 文件 §6](../modules/resource/architecture_degradation.md#6-vram-level-跨程序快照契約537-b0-f07)；本節只記狀態變化與殘留。

**觸發條件修正**：C0-08 的前提寫成「前次崩潰留下 segment，而且 reader 先附掛」。跨程序探針另外重現了兩個序列（Python 3.12，沒有頻率資料）：

- **S-B**：兩程序部署時，若 orchestrator 的 reader 已附掛而且仍在運作，perception 程序重啟（被 SIGKILL 而 tracker 存活，或正常停止後重啟），reader 會一直停在最後的值。
- **S-C**：orchestrator 程序結束時，它的 `resource_tracker` 會 unlink writer 正在使用的 segment。之後啟動的 reader 找不到 segment，回報 NORMAL，即使 writer 仍處於 EMERGENCY。

**修復後的行為**：

| ID | 修復前（§2） | 修復後 |
|:--|:--|:--|
| C0-06 | 沒有快照時讀為 NORMAL | 讀為 MISSING；所有事件都嘗試持久化，不啟動新的事件觸發 RAG，計入 `unknown_events` |
| C0-07 | 非法 level 拋 `ValueError`，整批中斷而且沒有 ACK | 讀為 INVALID，reader 不拋例外；整批照常持久化並 ACK |
| C0-08／S-A／S-B | 已附掛的 reader 一直讀舊 segment 的值 | 同一個 reader 的下一次讀取就會跟上新 writer：新 writer 啟動時先發佈「尚未決策」（STALE），第一次決策後讀為 FRESH。writer 正常停止時發佈「尚未決策」（STALE）；writer 死亡留下的快照，在上限 T 內仍讀為 FRESH，之後變成 STALE |
| S-C | reader 程序結束時刪除 writer 的 segment | reader 不持有任何資源，結束時不影響 writer |

只有 FRESH EMERGENCY 會卸載非 anomaly 事件，並計入 `shed_events`；其他狀態都不會卸載。`main.py --mode full` 的建構順序沒有改（AST 測試保留），因為 reader 不再持有狀態，這個順序已不會造成問題。

**證據**（都不需要 GPU）：

| 測試 | 涵蓋 |
|:--|:--|
| [test_vram_level_snapshot_contract_537.py](../../tests/unit/test_vram_level_snapshot_contract_537.py) | 四種狀態與邊界、各種不合法記錄（含 pad、FIFO）、reader 不拋例外、重啟與殘留、`close()` 與繼任 writer 交錯、發佈的每個階段失敗、`CLOCK_BOOTTIME`、dispatcher 在決策前取時戳 |
| [test_orchestrator_vram_policy_537.py](../../tests/unit/test_orchestrator_vram_policy_537.py) | 同一個 consumer 依序經過 FRESH EMERGENCY → STALE → MISSING → INVALID → reader 出錯 → FRESH NORMAL，逐步檢查持久化、ACK、卸載、RAG 與計數；狀態轉換 log；storage 失敗仍 ACK（現況） |
| [test_vram_level_snapshot_xproc_537.py](../../tests/integration/test_vram_level_snapshot_xproc_537.py) | 真實子程序：S-C、writer 被 SIGKILL 後先老化再重啟（S-A／S-B，老化以注入的時鐘偏移控制）、SIGKILL 後立即重啟、writer 在暫存檔寫完而尚未 rename 時死亡、舊版 SharedMemory attacher 的 tracker 刪除名稱後恢復、並行發佈與讀取（以 handshake 結束；要求 0 個 INVALID、0 個錯配、0 個倒退、至少 1000 次 FRESH 讀取，而且子程序成功結束）。子程序的等待都有上限（readiness 逐 byte 讀取並共用同一個 deadline），子程序一律回收 |
| [test_resource_degradation_observed_537.py](../../tests/unit/test_resource_degradation_observed_537.py) | C0-06～08 的四個測試改為目標語義 |

負控制的輸出附在 PR 描述：同一個與版本無關的 harness，在修復前的 `c0a29e47` 上 4 個情境都因行為斷言失敗（不是 import 失敗），修復後全部通過；5 個 mutation（忽略 age、沿用上一個 level、把 unknown 當 NORMAL、決策後才取時戳、不經 rename 直接覆寫）都會讓對應的測試失敗。

**殘留與交接**：

- **C0-05**：NVML 失敗時，writer 仍發佈 FRESH NORMAL（REQ-535-07-01）。
- 「嘗試持久化」不保證不遺失資料：storage 失敗時仍會 ACK，PEL 也沒有回收（REQ-535-06-01，#538）。
- 不保證 unknown 狀態下不用 GPU：Chroma 預設 embedding 沒有指定 provider；已經在執行的 RAG task 不受影響。
- T 預設 1 s 是推估值，沒有量測，待 #538 裁決；單一 writer 沒有強制機制；`dispatcher.stop()` 只在 CancelledError 時呼叫。以上都交 #538。
- dispatcher 閒置時讀為 STALE，屬於預期行為（不加 heartbeat）。
- 沒有在真實 Redis、Chroma 或 VRAM 壓力下做端到端驗證。
- 部署：新舊版本不能混用，升級或 rollback 時要先停掉兩端。
