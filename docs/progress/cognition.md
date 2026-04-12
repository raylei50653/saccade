# Cognition 模組進度核對 (2026-04-12)

## 模組狀態：進行中 (llama-server 優先策略)

## 1. LLM 引擎 (llm_engine.py)
- [x] **基礎實作**: 非同步 `httpx` 客戶端封裝。
- [x] **伺服器後端**: 採用 `llama-server` (llama.cpp) 提供 HTTP API。
- [x] **Systemd 整合**: 完成 `yolo-cognition.service` 服務配置。
- [x] **超時管理**: 實作了 60 秒異步超時機制，確保不阻塞管線。

## 2. 視覺推理 (vlm_engine.py)
- [ ] **多模態對接**: 整合 `llama-server` 的視覺提示詞介面 (OpenAI Vision API)。
- [ ] **視覺提示詞**: 實作將影格編碼為 Base64 並傳輸至推理伺服器。

## 3. 幀選取器 (frame_selector.py)
- [ ] **資訊熵介面**: 接收來自 `perception/entropy.py` 的觸發信號。
- [ ] **動態抓幀**: 實作從 MediaMTX 抓取高價值關鍵幀並傳遞給 VLM。

## 4. 資源調度器 (resource_manager.py)
- [ ] **VRAM 動態管理**: 透過 `nvml` 監測 VRAM，並在必要時透過 Systemd 重啟 `llama-server` 以釋放空間或調整 `-ngl`。

## 已完成里程碑
- [x] **ADR 004**: 確立以獨立進程 (HTTP API) 模式執行慢路徑推理。
- [x] **CI/CD**: `httpx` 加入依賴清單。

## 最後更新
2026-04-12
