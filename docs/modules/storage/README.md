# Storage Module (長期記憶與儲存)

## 📐 模組職責
負責 Redis 軌跡快取與微批次寫入、ChromaDB 向量長期記憶存儲與混合語意檢索。

## 🟢 目前現況
* **Redis L3 Shock Absorber 實現**：
  * 使用 Redis Streams 進行高頻感知事件（saccade:stream）的非同步緩衝，連接池設為 `max_connections=32`。
  * 利用 `xadd` 的 `approximate=True` 與 `maxlen=10000` 進行極限寫入，防止 Stream 無限增長並消除阻礙。
  * `MicroBatcher` 緩衝器：保留給一般 Redis List 批次寫入；cognition event 主路徑使用 Stream。
* **ChromaStore L4 向量存儲實現**：
  * 使用 `PersistentClient` 連接 runtime-created storage/chroma_db。
  * 實作 `hybrid_query` 方法，支援合併語意搜尋、視覺向量（SigLIP 2）、時間戳（`$gte` 語意限制）、以及 `is_anomaly` 等多維 metadata 過濾。
  * 支援 `backup()` 冷備份，自動將向量庫目錄壓縮成 zip Snapshot，並寫入 runtime-created storage/backups。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | 外圍慢路徑（見 [pipeline_flow.md](../../reference/pipeline_flow.md) §4.1） |
| **輸入** | track / scene events（含 SigLIP 2 視覺向量、metadata） |
| **輸出** | Redis Stream `saccade:stream`、ChromaDB 向量持久化 |
| **上游 → 下游** | `perception events → RedisCache.add_to_stream() → Redis Stream → RedisCache.read_stream_batch() → ChromaStore.add_memory() / hybrid_query()`；下游由 [cognition](../cognition/README.md) 讀取 |

## ⚖️ GO / NO-GO 決策

🟢 穩定運行，無 active ablation。Event / schema 規格見 [api_spec.md](api_spec.md)。

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
