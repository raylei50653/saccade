# Resource Module (資源與健康度)

## 📐 模組職責
負責實時顯存 (VRAM) 監控、多進程 Named Shared Memory 狀態廣播與三階降級遲滯保護。

## 🟢 目前現況
* **ResourceManager 階梯降級邏輯**：
  * 使用 `pynvml` 連接 GPU 驅動，監控顯存佔用百分比。
  * 降級閾值設定為：NORMAL (<85%) ➡️ REDUCED (85%~92%，壓縮緩衝池) ➡️ FAST_PATH (92%~96%，跳過 ReID/RAG) ➡️ EMERGENCY (>96%，縮小解析度至 320p、丟棄非異常影格)。
  * **遲滯保護 (Hysteresis)**：下降門檻比上升門檻低 $5\%$，即升級到 EMERGENCY (96%) 後，必須降到 $91\%$ 以下才恢復為 FAST_PATH；FAST_PATH (92%) 必須降到 $87\%$ 以下恢復成 REDUCED，以此防止顯存處於臨界值時系統抖動。
* **跨進程廣播設計 (`saccade_vram_level`)**：
  * `VRAMLevelWriter` 與 `VRAMLevelReader` 透過 **POSIX Named Shared Memory** 共享 1-Byte 的 `DegradationLevel` 數值。
  * Writer 每次重啟時自動 unlink staled segment（清除上次崩潰留下的殘存段），Reader 附掛時若找不到共享記憶體則 fallback 回 NORMAL 狀態，確保異步主進程在獨立啟動時無死鎖風險。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | 橫切（cross-cutting，VRAM 廣播）；見 [pipeline_flow.md](../../reference/pipeline_flow.md) §4.3 |
| **輸入** | `pynvml` VRAM 佔用 % |
| **輸出** | 1-Byte `DegradationLevel`（POSIX named shm `saccade_vram_level`）→ 各模組 reader |
| **上游 → 下游** | `pynvml → ResourceManager (hysteresis) → VRAMLevelWriter(shm) → reid / cognition / streaming readers` |

## ⚖️ GO / NO-GO 決策

🟢 穩定運行，無 active ablation。OOM 維運見 [runbooks/vram_oom.md](runbooks/vram_oom.md)。

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
