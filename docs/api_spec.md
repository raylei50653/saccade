# Saccade API & Event Specification

Saccade 採用非同步事件驅動架構，主要通訊發生在 Redis 事件佇列與 llama-server HTTP 介面之間。

## 1. 內部事件佇列 (Redis)
Perception (快路徑) 在觸發事件時，應推送到 Redis List。

- **Key:** `saccade:events`
- **Format:** JSON

### 事件結構範例
```json
{
  "event_id": "uuid-v4",
  "timestamp": 1712918400.123,
  "type": "entropy_trigger",
  "metadata": {
    "entropy_value": 0.85,
    "source_path": "rtsp://camera1/live",
    "frame_id": 4502,
    "objects": ["person", "backpack"]
  }
}
```

---

## 2. 認知推理介面 (llama-server)
Saccade 的慢路徑透過 HTTP 呼叫 `llama-server`。

### 視覺推理 (VLM)
- **Endpoint:** `POST /completion`
- **Payload 規範:**
```json
{
  "prompt": "USER:[image_0]\nDescribe the activity in this frame.\nASSISTANT:",
  "image_data": [
    {"data": "base64_encoded_string", "id": 0}
  ],
  "n_predict": 256,
  "stream": false
}
```

---

## 3. 健康檢查接口 (Health API)
`pipeline/health.py` 依賴此規範來判定系統狀態。

- **LLM Health:** `GET /health` (Expected: 200 OK)
- **System Metrics:** `GET /metrics` (Prometheus format, optional)

---

## 4. 開發約定 (Coding Standards)

### 非同步呼叫
所有對外通訊必須使用 `httpx.AsyncClient` 並明確設定 `timeout`：
- **LLM Inference:** 60.0s
- **Health Check:** 2.0s
- **Redis Ping:** 3.0s

### 型別檢查
所有 API 回傳的字典 (Dict) 必須映射到 `typing.TypedDict` 或 `dataclasses`，禁止在邏輯層直接操作裸字典。
