# L4: 儲存層 (Storage Layer)

## 1. 定義與目標
L4 是 Saccade 的「長期記憶 (Long-term Memory)」，負責大規模向量檢索與屬性過濾。目標是提供語義維度的時空檢索。

## 2. 核心組件
- **向量庫 (ChromaStore)**: 封裝 ChromaDB 實作，負責 HNSW 索引與向量儲存。
- **後端存儲 (HNSW / DuckDB)**: 負責底層磁碟持久化。
- **中繼資料管理 (Metadata Management)**: 儲存 BBox, Timestamp, Confidence, TrackID 等。

## 3. 資料流向
- **Input**: 聚合後的物件特徵與中繼資料。
- **Output**: 檢索結果 (IDs, Distance, Metadata)。

## 4. 關鍵優化
- **Async DB Client**: 所有的資料庫操作皆為非同步，不阻塞感知管線。
- **Batch Insertion**: 定期 (如每 1s) 或定量 (如 50 條) 進行批量寫入，提升儲存性能。
