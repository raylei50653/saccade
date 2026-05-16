# RTSP Contract

本文件定義本 repo 內 RTSP / MediaMTX 的標準約定。

## Canonical Defaults

- host: `127.0.0.1`
- port: `8554`
- reader:
  - user: `reader`
  - password: `readpass123`
- publisher:
  - user: `publisher`
  - password: `pubpass123`

## Path Naming

- 單路輸入：`live`
- 多路輸入：`stream_<id>`
- 推論輸出：`detected`

例子：

```text
rtsp://reader:readpass123@127.0.0.1:8554/live
rtsp://reader:readpass123@127.0.0.1:8554/stream_0
rtsp://publisher:pubpass123@127.0.0.1:8554/stream_7
rtsp://127.0.0.1:8554/detected
```

## Source Of Truth

- helper module: [src/saccade/media/rtsp.py](/home/ray/developer/ai/saccade/src/saccade/media/rtsp.py)
- server config: [infra/mediamtx.yml](/home/ray/developer/ai/saccade/infra/mediamtx.yml)

新程式不要直接手寫 RTSP URL，優先用 helper：

- `build_rtsp_url(...)`
- `build_reader_url(...)`
- `build_publisher_url(...)`
- `build_stream_path(...)`

## Local MediaMTX Policy

- 開發預設只開 RTSP，不開 HLS / WebRTC。
- `docker-compose.yml` 對 MediaMTX 採顯式 port mapping `8554:8554`。
- 避免使用 `network_mode: host`，除非你明確需要 host networking。

## Recommended Commands

啟動 MediaMTX：

```bash
docker compose up -d mediamtx
```

啟動 8 路 publisher：

```bash
bash scripts/ops/setup_8_streams.sh
```

驗證單路讀流：

```bash
ffprobe -v error rtsp://reader:readpass123@127.0.0.1:8554/stream_0
```

跑 8 路 perception demo：

```bash
uv run python scripts/ops/run_8stream_perception.py \
  --streams 8 \
  --rtsp-host 127.0.0.1 \
  --rtsp-port 8554 \
  --rtsp-user reader \
  --rtsp-password readpass123 \
  --rtsp-path-prefix stream_
```
