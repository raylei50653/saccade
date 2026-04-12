# Storage 模組進度 (2026-04-12)

## 模組狀態：待啟動

## 1. 向量記憶體 (chroma_store.py)
- [ ] **ChromaDB 初始化**: 建立 Collection 結構。
- [ ] **Embedding 對接**: 整合 SigLIP/CLIP 向量生成與儲存。
- [ ] **相似性檢索**: 實現長短期記憶查詢介面。

## 2. 快取與事件佇列 (redis_cache.py)
- [ ] **即時狀態**: 儲存 Perception 模組的最新目標位置與 ID。
- [ ] **事件發布**: 發送觸發慢路徑 (Cognition) 的非同步事件。
- [ ] **全域變數**: 分散式系統狀態同步。

## 待處理
- [ ] 設定本地 ChromaDB 與 Redis 的自動啟動腳本。
- [ ] 測試大量數據下的向量檢索延遲。
