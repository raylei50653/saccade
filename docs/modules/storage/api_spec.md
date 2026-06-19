# Saccade API & Event Specification

本文件描述目前程式碼中實際存在的介面與資料合約。重點分成三類：

- RTSP / MediaMTX stream contract
- L1/L3：Perception 發出的 Redis event queue
- L3/L5：Orchestrator 使用的 Redis stream
- 外部查詢：FastAPI retrieval API

若本文件與目前主路徑程式碼衝突，以程式碼為準，並應同步修正本文件。

---

## 1. Source of Truth

此規範主要對應下列檔案：

- [src/saccade/media/rtsp.py](../../../src/saccade/media/rtsp.py)
- [src/saccade/media/mediamtx_client.py](../../../src/saccade/media/mediamtx_client.py)
- [src/saccade/media/ffmpeg_utils.py](../../../src/saccade/media/ffmpeg_utils.py)
- [src/saccade/perception/entropy.py](../../../src/saccade/perception/entropy.py)
- [src/saccade/storage/redis_cache.py](../../../src/saccade/storage/redis_cache.py)
- [src/saccade/storage/chroma_store.py](../../../src/saccade/storage/chroma_store.py)
- [src/saccade/cognition/orchestrator.py](../../../src/saccade/cognition/orchestrator.py)
- [src/saccade/api/server.py](../../../src/saccade/api/server.py)
- [src/saccade/pipeline/health.py](../../../src/saccade/pipeline/health.py)

---

## 2. RTSP / MediaMTX Contract

### 2.1 Source of Truth

- Module: `src/saccade/media/rtsp.py`
- Purpose:
  - build canonical RTSP URLs
  - define default MediaMTX host / port / credentials
  - standardize per-stream path naming

### 2.2 Canonical Defaults

- host: `127.0.0.1`
- port: `8554`
- read credentials:
  - user: `reader`
  - password: `readpass123`
- publish credentials:
  - user: `publisher`
  - password: `pubpass123`

### 2.3 Path Naming

- single-camera path: `live`
- multi-stream path pattern: `stream_<id>`
- processed output path: `detected`

Examples:

```text
rtsp://reader:readpass123@127.0.0.1:8554/live
rtsp://reader:readpass123@127.0.0.1:8554/stream_0
rtsp://publisher:pubpass123@127.0.0.1:8554/stream_7
rtsp://127.0.0.1:8554/detected
```

### 2.4 Programmatic API

- `build_rtsp_url(path, host=..., port=..., username=..., password=...)`
- `build_reader_url(path, host=..., port=..., username=..., password=...)`
- `build_publisher_url(path, host=..., port=..., username=..., password=...)`
- `build_stream_path(stream_id, prefix="stream_")`
- `RTSPEndpoint.from_url(url)`

### 2.5 Runtime Contract

- Reader side:
  - `MediaMTXClient` expects a full RTSP URL.
  - For RTSP, the stable default pipeline is TCP transport plus CPU decode to RGB appsink.
- Publisher side:
  - `RTSPStreamer` accepts a full RTSP URL.
  - FFmpeg push scripts must use `-rtsp_transport tcp`.
- Multi-stream demo:
  - preferred flags:
    - `--rtsp-host`
    - `--rtsp-port`
    - `--rtsp-user`
    - `--rtsp-password`
    - `--rtsp-path-prefix`
  - `--rtsp-prefix` is legacy / deprecated.

### 2.6 Compatibility Notes

- Passing a fully formed RTSP URL remains supported.
- New code should not hardcode `rtsp://...` strings inline when the helper API can express the same endpoint.

---

## 3. Redis Event Queue

### 3.1 Queue Key

- Key: `saccade:events`
- Type: Redis List
- Producer:
  - `EntropyTrigger.emit_event()`
  - `RedisCache.publish_event()` + `MicroBatcher`
- Retention:
  - TTL `3600s`

### 3.2 Event Schema

目前 `entropy.py` 實際發送格式如下：

```json
{
  "event_id": "uuid-v4",
  "timestamp": 1712918400.123,
  "type": "entropy_trigger",
  "metadata": {
    "entropy_value": 0.85,
    "source_path": "rtsp://127.0.0.1:8554/live",
    "frame_id": 4502,
    "objects": ["person", "backpack"]
  }
}
```

### 3.3 Field Contract

- `event_id`
  - type: `string`
  - contract: globally unique event id, currently `uuid4`
- `timestamp`
  - type: `float`
  - contract: unix timestamp in seconds
- `type`
  - type: `string`
  - current value: `entropy_trigger`
- `metadata.entropy_value`
  - type: `float`
  - range: expected `0.0 ~ 1.0`
- `metadata.source_path`
  - type: `string`
  - contract: source camera / stream identifier
- `metadata.frame_id`
  - type: `int`
  - contract: source frame index
- `metadata.objects`
  - type: `list[string]`
  - contract: detected object labels

### 3.4 Notes

- `is_anomaly` 不是目前 perception event producer 的必填欄位。
- `is_anomaly` 目前主要在 cognition / storage 寫入 Chroma metadata 時推導產生。
- 若未來要讓 perception 直接發 `is_anomaly`，需同步更新本文件與 `orchestrator.py`。

---

## 4. Redis Stream Contract

### 4.1 Stream Key

- Key: `saccade:stream`
- Type: Redis Stream
- Producer:
  - `RedisCache.add_to_stream()`
  - `RedisCache.add_to_stream_batch()`
- Consumer group:
  - group: `orchestrator_group`
  - consumer: currently `worker_1`

### 4.2 Stream Payload

`RedisCache` 目前將事件包成單欄位 payload：

```text
{
  "data": "<json-serialized-event>"
}
```

其中 `data` 反序列化後的內容，應沿用第 3 節的 event schema。

### 4.3 Consumption Contract

- reader:
  - `RedisCache.read_stream_batch(count=..., timeout_ms=...)`
- ack:
  - `RedisCache.acknowledge(message_ids)`
  - 或 orchestrator 直接 `XACK`
- retention:
  - approximate `MAXLEN=10000`

### 4.4 Notes

- 目前 repo 同時存在 Redis List 與 Redis Stream 兩條事件入口。
- 若未來正式統一為 Stream-only，需更新：
  - `entropy.py`
  - `tests/integration/test_pipeline.py`
  - 本文件

---

## 5. Chroma Memory Contract

### 5.1 Collection

- default path: storage/chroma_db (runtime-created; not committed)
- default collection: `saccade_memories`

### 5.2 Insert Contract

`ChromaStore.add_memory()` 目前接受：

```python
add_memory(
    content: str,
    metadata: dict[str, Any],
    doc_id: str | None = None,
    embedding: list[float] | None = None,
) -> str
```

### 5.3 Stored Metadata

目前 orchestrator 實際寫入的 metadata 典型欄位：

```python
{
    "frame_id": 123,
    "entropy": 0.91,
    "objects": "person, backpack",
    "is_anomaly": 1,
    "timestamp": 1712918400.123,
}
```

欄位合約：

- `frame_id`: `int`
- `entropy`: `float`
- `objects`: `string`
  - 注意：這裡目前是 comma-separated string，不是 list
- `is_anomaly`: `0 | 1`
- `timestamp`: `float`

### 5.4 Query Contract

`ChromaStore.hybrid_query()` 目前支援：

- `query_text`
- `query_embedding`
- `n_results`
- `start_time`
- `is_anomaly`
- `object_filter`

對應 where filter 範型：

```python
{
    "$and": [
        {"timestamp": {"$gte": 1712900000.0}},
        {"is_anomaly": 1},
        {"objects": {"$contains": "person"}}
    ]
}
```

---

## 6. FastAPI Retrieval API

對應檔案：[src/saccade/api/server.py](../../../src/saccade/api/server.py)

### 6.1 `GET /`

Response:

```json
{
  "status": "online",
  "system": "Saccade",
  "api_version": "1.0"
}
```

### 6.2 `GET /objects`

用途：

- 讀取目前 Redis 中仍未過期的 active object ids

Response:

```json
{
  "count": 2,
  "active_objects": [101, 102]
}
```

### 6.3 `GET /objects/{obj_id}`

用途：

- 讀取單一物件的 Redis 快取資料

目前注意事項：

- code path 會計算 `dwell_time_seconds`
- 但 `RedisCache.update_object_track()` 目前只寫 `id / label / box / timestamp`
- 若沒有 `first_seen / last_seen`，此 endpoint 的 dwell-time 假設不完整

這代表：

- `server.py` 與 `redis_cache.py` 之間目前存在資料合約缺口
- 若要正式對外使用此 endpoint，需先補齊 object history schema

### 6.4 `POST /search`

Request body:

```json
{
  "text": "person with suspicious bag",
  "n_results": 5,
  "start_time": 1712900000.0,
  "is_anomaly": true
}
```

Request contract:

- `text`: `string`, required
- `n_results`: `int | null`, default `5`
- `start_time`: `float | null`
- `is_anomaly`: `bool | null`

Response shape:

```json
{
  "query": "person with suspicious bag",
  "results": [
    {
      "id": "memory-id",
      "content": "Scene contains: 1 person, 1 backpack.",
      "metadata": {
        "frame_id": 4502,
        "entropy": 0.91,
        "objects": "person, backpack",
        "is_anomaly": 1,
        "timestamp": 1712918400.123
      },
      "distance": 0.123
    }
  ]
}
```

---

## 7. Health Contract

對應檔案：[src/saccade/pipeline/health.py](../../../src/saccade/pipeline/health.py)

### 7.1 Checked Services

- systemd user services:
  - `yolo-perception`
  - `yolo-orchestrator`
  - `mediamtx`
- redis connectivity:
  - `PING`
  - queue depth via `LLEN saccade:events`
- vram:
  - NVML memory usage
- stress metrics:
  - event loop latency
  - redis queue depth
  - torch-reported VRAM fragmentation

### 7.2 Health Output Contract

`HealthChecker.run()` returns:

- `timestamp`
- `systemd`
- `vram`
- `redis`
- `stress`
- `overall_ok`

這是 internal operational contract，不是正式對外 HTTP API。

---

## 8. Concurrency / I/O Rules

- Redis list writes:
  - use `redis.asyncio`
  - high-frequency queue writes should prefer `RedisCache.publish_event()` + `MicroBatcher`
- Redis stream writes:
  - use `RedisCache.add_to_stream()` or batch variant
- Orchestrator:
  - DB-bound tasks are limited by `asyncio.Semaphore(32)`
- Blocking RAG calls:
  - must be wrapped through executor / background task path

---

## 9. Known Gaps

- `EntropyTrigger.calculate_entropy()` is still placeholder logic and not a stable semantic contract yet.
- Redis List event path and Redis Stream path are both present; long-term canonical path is not fully unified.
- `GET /objects/{obj_id}` assumes richer object history than `RedisCache.update_object_track()` currently stores.
- `docs/architecture/README.md` and `docs/reference/pipeline_flow.md` still describe the intended system shape at a higher level; this file describes currently implemented interface contracts.

---

## 10. Update Rules

更新本文件的時機：

- 修改 event schema
- 修改 Redis key / stream name / consumer group
- 修改 Chroma metadata schema
- 修改 FastAPI request/response shape
- 修改 health checker 的輸出合約

若只是實驗性欄位掃描，先記在 [TODO.md](TODO.md) 或實驗文件；當欄位成為穩定介面時，再回寫此規範。
