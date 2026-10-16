# Resource Module (資源與健康度)

## 📐 模組職責
負責實時顯存 (VRAM) 監控、多進程 Named Shared Memory 狀態廣播與三階降級遲滯保護。

## 🟢 目前現況
* **ResourceManager 階梯降級邏輯**：
  * 使用 `pynvml` 連接 GPU 驅動，監控顯存佔用百分比。
  * 降級閾值設定為：NORMAL (<85%) ➡️ REDUCED (85%~92%，壓縮緩衝池) ➡️ FAST_PATH (92%~96%，跳過 ReID/RAG) ➡️ EMERGENCY (>96%，縮小解析度至 320p、丟棄非異常影格)。
  * **遲滯保護 (Hysteresis)**：下降門檻比上升門檻低 $5\%$，即升級到 EMERGENCY (96%) 後，必須降到 $91\%$ 以下才恢復為 FAST_PATH；FAST_PATH (92%) 必須降到 $87\%$ 以下恢復成 REDUCED，以此防止顯存處於臨界值時系統抖動。
* **跨進程廣播設計 (`saccade_vram_level`)**：
  * `VRAMLevelWriter` 把每次決策寫成 `/dev/shm/saccade_vram_level` 的 24-byte 快照（level、決策時間、generation、CRC），以「寫暫存檔再 rename」原子發佈。
  * `VRAMLevelReader.read_snapshot()` 每次都依名稱重讀，回傳 FRESH／STALE／MISSING／INVALID，不拋例外；writer 重啟後的下一次讀取就會跟上。只有 FRESH 的 level 可以拿來卸載工作。契約見 [degradation §6](architecture_degradation.md#6-vram-level-跨程序快照契約537-b0-f07)。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | 橫切（cross-cutting，VRAM 廣播）；見 [pipeline_flow.md](../../reference/pipeline_flow.md) §4.3 |
| **輸入** | `pynvml` VRAM 佔用 % |
| **輸出** | `DegradationLevel` 快照與決策時間（`/dev/shm/saccade_vram_level`，見 §6 契約）→ cognition orchestrator（目前唯一的 reader） |
| **上游 → 下游** | `pynvml → ResourceManager (hysteresis) → VRAMLevelWriter(/dev/shm 快照) → cognition orchestrator`（目前唯一的 reader） |

## ⚖️ GO / NO-GO 決策

🟢 穩定運行，無 active ablation。OOM 維運見 [runbooks/vram_oom.md](runbooks/vram_oom.md)。

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
