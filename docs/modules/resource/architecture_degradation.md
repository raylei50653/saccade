# L6: 資源層 (Resource - Resource Management)

## 1. 定義與目標
L6 是 Saccade 的「決策大腦」，負責高層級的 VRAM 資源監控與系統平衡。目標是確保系統在資源極限環境下透過自適應策略優雅降級，優先保證核心感知（L1）。

## 2. 核心組件
- **ResourceManager** (`src/saccade/resource/resource_manager.py`): 透過 NVML 實時監測 VRAM 負載，輸出降級指令。
- **FrameSelector** (`src/saccade/resource/frame_selector.py`): 基於 L2 漂移分數動態調整 L1 偵測頻率。

## 3. 資料流向
- **Input**: VRAM Stats (NVML)、Latency Spike (L1)、Drift Score (L2)。
- **Output**: DegradationLevel（NORMAL / REDUCED / FAST_PATH / EMERGENCY）。

## 4. 階梯式降級策略

| Level | VRAM 閾值 | 動作 |
|---|---|---|
| NORMAL | < 85% | 正常運行 |
| REDUCED | > 85% | 縮減 5-Buffer Pool 大小 |
| FAST_PATH | > 92% | 暫停 L2（SigLIP 2）與 L5（RAG） |
| EMERGENCY | > 96% | 解析度 640→320、Target Culling（Confidence < 0.4）、track_buffer 30→10 |

## 5. Hysteresis（遲滯保護）
- 升級門檻：85% / 92% / 96%
- 降級恢復：需降至觸發點 **-5%** 才恢復上一級，防止臨界點頻繁切換（Thrashing）。

## 6. VRAM-level 跨程序快照契約（#537 B0-F07）

Dispatcher 端的 `VRAMLevelWriter` 把每次決策發佈為一份完整快照；orchestrator 端的 `VRAMLevelReader` 每次讀取都重新讀取這份快照。程式在 `src/saccade/resource/resource_manager.py`，失敗矩陣與修復前的行為見 [#537 C0 §7](../../architecture/failure_degradation_537.md#7-b0-f07-修復c0-060708-收斂)。

**發佈方式**：writer 先把 24-byte 記錄寫到私有暫存檔 `/dev/shm/saccade_vram_level.tmp.<pid>.<generation>`，再 `rename()` 覆蓋 `/dev/shm/saccade_vram_level`。reader 每次讀取都依名稱 `open` → `pread` → `close`，所以一定讀到「整份舊快照」或「整份新快照」，下一次讀取就會跟上重啟後的 writer。兩端都不使用 `SharedMemory`，因此沒有 `resource_tracker` 會刪除別的程序的快照。

**記錄格式 v1**（little-endian，24 bytes）：

| offset | 型別 | 欄位 | 說明 |
|---|---|---|---|
| 0 | u8 | level | 0～3 |
| 1 | u8 | version | 1 |
| 2–3 | pad | — | 必須為 0 |
| 4 | u32 | crc32 | 把本欄（offset 4–7）設為 0 後，對全部 24 bytes 計算（含 pad） |
| 8 | u64 | generation | writer 啟動時產生的隨機值，只用於觀測重啟 |
| 16 | u64 | decided_at_ns | 決策**開始前**讀取的 `CLOCK_BOOTTIME`；0 表示尚未決策 |

**讀取狀態**（`VRAMLevelSnapshot.status`；reader 不拋例外）：

| status | 判定 | `level` |
|---|---|---|
| FRESH | 記錄有效，且 0 ≤ age ≤ `max_age_ns`（預設 1 s，由建構參數設定） | 0～3 |
| STALE | 記錄有效，但 `decided_at_ns == 0`，或 age 超過上限 | None |
| MISSING | 快照不存在 | None |
| INVALID | 大小不是 24 bytes、version 不符、pad 不為 0、CRC 不符、level > 3、不是一般檔案（以 `O_NONBLOCK` 開檔，名稱上的 FIFO 不會讓讀取阻塞）、age < 0，或其他讀取錯誤 | None |

**Freshness 的意義**：「這個 level 是在 `max_age_ns` 內決定的」，不是「writer 還活著」。dispatcher 只在取到 frame 時才決策，所以閒置時 reader 會讀到 STALE；本契約不加 heartbeat，因為 heartbeat 會改變決策頻率（C2 範圍）。決策時間在 dispatcher 呼叫 `decide_degradation_level()` 之前讀取，並透過 `write(level, decided_at_ns)` 傳入。這樣即使 writer 在決策後、發佈前停住，舊決策也不會被蓋上新的時間。`CLOCK_BOOTTIME` 在 suspend 期間持續計時；reader 與 writer 必須在同一個 time namespace 內。

**Ownership 與重啟**：

- 每台主機只能有一個 writer（由 dispatcher 持有）。這條規則沒有強制機制：第二個 writer 會覆蓋第一個的快照。
- writer 啟動時，會先刪除擁有者程序已不存在的暫存檔，然後發佈一份「尚未決策」的記錄。前一個 writer 留下的舊 level 因此立刻變成 STALE，不會延續。
- writer `close()` 不刪除公開名稱：公開快照仍屬於自己（generation 相同）時，改發佈一份「尚未決策」記錄，所以正常停止後 reader 立刻讀到 STALE。若繼任 writer 剛好在 ownership 檢查與發佈之間寫入（兩個 writer 同時存活，不支援），繼任者的快照會讀為 STALE，直到它下一次決策；不會變成 MISSING，也不會讀到錯誤的 level。
- writer 被 SIGKILL 時，快照會留下來：它最後的值在 `max_age_ns` 內仍讀為 FRESH，之後變成 STALE。這就是 freshness 上限的定義。
- 發佈失敗（例如 `/dev/shm` 已滿）時，`write()` 回 False，只在第一次失敗時記一行 log；reader 讀到的上一份快照會依 age 變成 STALE。

**Consumer policy**（`PipelineOrchestrator.handle_cognitive_event`）：

| 狀態 | 非 anomaly 事件 | anomaly 事件 | 新的事件觸發 RAG |
|---|---|---|---|
| FRESH NORMAL／REDUCED | 嘗試持久化 | 嘗試持久化 | 依現有條件 |
| FRESH FAST_PATH | 嘗試持久化 | 嘗試持久化 | 不啟動 |
| FRESH EMERGENCY | **刻意卸載**，計入 `shed_events` | 嘗試持久化 | 不啟動 |
| STALE／MISSING／INVALID，或 reader 本身出錯 | 嘗試持久化，計入 `unknown_events` | 嘗試持久化，計入 `unknown_events` | 不啟動 |

計數單位是「處理過的事件」，可由 `PipelineOrchestrator.vram_policy_stats()` 讀取（`shed_events`、`unknown_events`、`last_status`）。status 每次改變時，stdout 記錄一行 `VRAM level status <舊> -> <新>`，內容包含兩個計數器。每次讀取各自判定，consumer 不沿用先前讀到的 level。

**這個 policy 不保證的事**：

- 「嘗試持久化」不等於不遺失資料。`add_memory` 失敗時例外仍被吞掉，整批照樣 ACK；PEL 也沒有回收（REQ-535-06-01，#538）。
- 不保證 unknown 狀態下不使用 GPU。Chroma 的預設 embedding 沒有指定 provider，可能用到 GPU；已經在執行的 RAG task，以及啟動時的 RAG 初始化，都不受這個 gate 控制。
- NVML 初始化失敗時，writer 仍會發佈 FRESH NORMAL（C0-05），本契約不涵蓋這種情況（REQ-535-07-01）。

**部署規則**：v1 記錄與修復前的 1-byte segment 不能混用。新 reader 會把舊 segment 判為 INVALID，但舊 reader 不會檢查 freshness、會保留舊 mapping，而且它結束時，它的 `resource_tracker` 會刪除公開名稱。升級或 rollback 時：(1) 停掉 perception（writer）與 orchestrator（reader）兩端；(2) 確認兩端程序與它們的 `multiprocessing.resource_tracker` 子程序都已結束（例如以服務使用者執行 `pgrep -f multiprocessing.resource_tracker` 沒有結果）；(3) 再啟動新版本。如果漏掉第 (2) 步，殘留 tracker 刪除的只是名稱：reader 會讀到 MISSING（不會讀到錯誤的 level），writer 下一次發佈就會恢復（整合測試 `test_legacy_attacher_tracker_removal_heals_on_next_decision`）。
