# Saccade API & Event Specification (L1-L5 Edition)

Saccade 採用非同步事件驅動與向量檢索架構。通訊發生在 Redis 實時事件流與 ChromaDB 語義檢索介面。

---

## 1. 內部事件流 (L3: Redis Streams)
Perception (快路徑) 在觸發事件時，應推送到 Redis List (saccade:events)，未來將升級為 Redis Streams。

- **Key:** `saccade:events`
- **Format:** JSON
- **TTL:** 1 Hour (確保快路徑不因過期數據溢出)

### 事件結構範例 (L1 -> L3)
```json
{
  "event_id": "uuid-v4",
  "timestamp": 1712918400.123,
  "type": "entropy_trigger",
  "metadata": {
    "entropy_value": 0.85,
    "source_path": "local_cam",
    "frame_id": 4502,
    "objects": ["person", "backpack"],
    "is_anomaly": 0
  }
}
```

---

## 2. 向量檢索介面 (L4/L5: ChromaDB)
Saccade 的應用層透過語義特徵進行關聯搜尋。

### 語義與 Metadata 混合查詢
- **Query Type:** Vector + Metadata Filter
- **Schema:**
```python
{
  "query_text": "person with knife",
  "n_results": 5,
  "where": {
    "$and": [
      {"timestamp": {"$gte": 1712900000.0}},
      {"is_anomaly": 1}
    ]
  }
}
```

---

## 3. 健康檢查接口 (Health API)
`pipeline/health.py` 依賴此規範來判定系統狀態。

- **Redis Health:** `PING` (Expected: PONG)
- **Vector DB Health:** `client.heartbeat()` (Expected: Valid timestamp)
- **System Metrics:** 
    - `VRAM_Usage`: 偵測 GPU 記憶體是否超過 85% 閾值。
    - `Latency_L1`: 感知層單影格處理延遲 (目標: < 15ms)。

---

## 4. 開發約定 (Coding Standards)

### 非同步與併發 (Concurrency)
- **Redis 推送**: 必須使用 `redis.asyncio` 的 `rpush`。
- **寫入頻率控制**: `Orchestrator` 使用 `asyncio.Semaphore(32)` 控制併發，避免 I/O 阻塞。

### 數據流向 (Data Type)
- **影像傳輸**: 禁止使用 Base64。影像資料應留在 GPU Tensor 或存儲於本地快取路徑供 VLM/LLM 按需讀取。
- **特徵向量**: 固定為 768 或 1024 維的 `float32` 陣列。
