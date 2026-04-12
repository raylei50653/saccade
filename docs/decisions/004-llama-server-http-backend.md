# ADR 004: 採用 llama-server 作為非同步推理後端

## 狀態
已通過

## 背景 (Context)
原本規劃直接在 Python 進程中透過 `llama-cpp-python` 載入模型。但在雙軌管線中，慢路徑推理時間較長（數百毫秒至數秒），若在主進程中管理權重，會增加資源調度難度，且推理崩潰會直接導致整個系統掛掉。

## 決策 (Decision)
改為採用 `llama-server` 作為獨立的 HTTP 推理伺服器，Python 端透過非同步 `httpx` 客戶端進行通訊。

## 取捨 (Consequences / Trade-offs)
- **優點**: 
  - **進程隔離**: 推理伺服器與感知管線獨立，互不干擾。
  - **穩定性**: 支援 Systemd 自動重啟，VRAM OOM 時主程式可持續運行。
  - **併發管理**: `llama-server` 內建高效的 Request Queue 與 PagedAttention。
- **缺點**:
  - **通訊延遲**: 增加了極小（<1ms）的 HTTP 本地環回延遲，在慢路徑場景下可忽略不計。
  - **部署複雜度**: 需要管理額外的 Systemd 服務單元。
