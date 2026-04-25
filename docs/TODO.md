# Saccade TODO — 具體實作清單

> 基於代碼審查（2026-04-25）。每條都對應具體文件與行為落差，不是架構願景。

---

## ✅ 已完成（2026-04-25）

### P0 — GPUByteTracker 核心強化（ADR 013）
- [x] **ReID 融合代價矩陣**：`tracker_gpu.cu` cost matrix 改為 `(1-w)*IoU + w*CosSim`，預設 w=0.5，crowded 場景 w=0.8
- [x] **Strong ReID Gate**：CosSim > 0.75 時強制配對，對抗相機劇烈晃動
- [x] **GMC 全域運動補償**：Python 層 optical flow → 仿射矩陣傳入 C++ `gmc_kernel`，同步修正 Kalman 狀態與協方差
- [x] **Light Compensation**：`light_factor` 動態調整 R 矩陣，穩定夜間軌跡
- [x] **Saccade Heartbeat 間隔修正**：`% 30` → `% 10`，對齊 ADR 013 規格

### P1 — 媒體層穩定性
- [x] **RTSP 斷線自動恢復**：`mediamtx_client.py` 加入 `watchdog_loop()`，`_is_alive()` 偵測超時後呼叫 `_restart_pipeline()` 重建 GStreamer pipeline

### P2 — 儲存層
- [x] **Redis Micro-batching**：`MicroBatcher` 整合於 `RedisCache.publish_event()`，100ms 視窗聚合，Redis QPS ~300 → ~30

### P3 — 認知層（ADR 014）
- [x] **LlamaIndex RAG 接入**：`orchestrator.py` 連接 ChromaDB → LlamaIndex，使用 `BAAI/bge-small-en-v1.5` local embedding + Ollama llama3
- [x] **事件觸發式查詢**：`entropy > 0.9` 或 `is_anomaly=True` 時才觸發 RAG，避免每幀呼叫
- [x] **Visual Re-query (視覺重查)**：在 `orchestrator.py` 中註冊 `visual_requery` Tool 給 ReAct Agent，允許 LLM 發起 ChromaDB 純向量搜尋 (Image-to-Image 語義比對)。
- [x] **跨鏡頭 Re-ID**：重構 `FeatureBank` 支援 `stream_map`，實作 `find_cross_camera_matches` 矩陣運算，讓多路串流可共享特徵索引並進行跨畫面比對。

### P4 — 基礎設施與維運（Infrastructure & Maintenance）
- [x] **ChromaDB 冷備份**：於 `ChromaStore` 中實作 `backup()` 函數，利用 `shutil.make_archive` 定期壓縮並 snapshot 向量資料庫，防止長期記憶遺失。
- [x] **串流身分驗證**：修改 `infra/mediamtx.yml`，為發布 (publish) 與讀取 (read) 動作加入帳號密碼保護。
- [x] **Redis 自動清理**：實作 `cleanup_expired_objects()`，監控 Redis 記憶體使用量，當超過閾值時強制刪除最舊的 `saccade:obj:*` 快取，避免記憶體溢出。
- [x] **智慧影格抽樣策略**：在 `MediaMTXClient` (含 C++ 及 Python 回調) 中實作像素差異比對 (SAD < 2.0)，即時丟棄低資訊幀，降低無效計算負載。

---

## 🎯 專案里程碑：全部核心與周邊待辦事項已清空！

最後更新：2026-04-25
