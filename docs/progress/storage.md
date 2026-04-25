# Storage & Buffering (L3-L4) 進度核對

## 系統架構層次
- [x] **L3: Streaming/Buffering (串流緩衝層)** - 已完成
- [x] **L4: Storage (時空記憶層)** - 已完成

## 模組狀態：穩定運行

## L3: Streaming/Buffering (串流實作)
- [x] **Redis RPUSH 整合**: 實作基礎的事件串流化，將語義特徵異步寫入 Redis 隊列。
- [x] **[重大更新] Redis Pipelining (2026-04-15)**:
    - [x] **批次寫入**: 透過 Redis Pipeline 減少 90% 的寫入系統調用，支撐極高吞吐量需求。
    - [x] **實作位置**: `storage/redis_cache.py` 中的 `add_to_stream_batch`。
- [x] **儲存雜訊消除**: 
    - 語義去重：僅在 Drift Score 超過閾值時才觸發「強記憶（儲存）」，過濾無意義的重複背景。
- [x] **Micro-batching**: `MicroBatcher` 整合於 `RedisCache`，100ms 滑動窗口聚合，Redis QPS 從 ~300 降至 ~30。
- [x] **目標狀態同步**: 儲存追蹤目標的最後位置與過期管理。
- [x] **事件發布**: 提供統一的事件推播介面，供後端 API 消費。

## L4: Storage (記憶實作)
- [x] **ChromaDB 初始化**: 建立 Collection 結構，支持結構化數據與非結構化描述。
- [x] **Hybrid Query**: 實現語義搜尋與時間/屬性範圍查詢之混和介面。
- [x] **Embedding 管理**: 整合特徵儲存與高效索引機制。

- [x] **Redis 自動清理**: `cleanup_expired_objects()` 監控記憶體使用量，超過閾值時強制刪除最舊的 `saccade:obj:*` 快取。
- [x] **ChromaDB 冷備份**: `ChromaStore.backup()` 利用 `shutil.make_archive` 定期壓縮 snapshot，防止長期記憶遺失。

## 待處理

（已清空）

## 已完成里程碑
- [x] **儲存優化機制實裝**: 完成語義去重 (Deduplication)，過濾重複背景。
- [x] **向量化記憶**: 成功整合 ChromaDB 實現結構化數據與非結構化描述的關聯儲存。
- [x] **Redis 抽象層**: 建立統一的快取與佇列操作介面。
- [x] **Micro-batching 實裝**: `MicroBatcher` 將 Redis 寫入改為批次 pipeline，QPS ~300 → ~30。
- [x] **維運機制完備**: Redis 自動清理 + ChromaDB 定期 snapshot。

最後更新：2026-04-25
