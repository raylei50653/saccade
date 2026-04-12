# Media 模組進度 (2026-04-12)

## 模組狀態：穩定運行 (RTSP 背景抓幀已實作)

## 1. MediaMTX 用戶端 (mediamtx_client.py)
- [x] **串流接取**: 實作 RTSP/WebRTC 協議對接。
- [x] **背景抓幀**: 實作專屬讀取線程，確保即時性並防止緩衝區爆掉。
- [x] **自動重連**: 抓幀失敗時自動嘗試重新連接 RTSP 流。
- [x] **底層穩定**: 透過 `threads;1` 鎖定，解決 FFMPEG 多執行緒 Assertion 報錯。

## 2. FFmpeg 工具集 (ffmpeg_utils.py)
- [ ] **NVENC/NVDEC**: 封裝 NVIDIA 硬體編解碼工具。
- [x] **GStreamer 整合**: 支援 OpenCv 透過 GStreamer 高效能接入。

## 待處理
- [ ] 支援多路串流同時處理。
- [ ] 整合 HLS 延遲監測。

## 最後更新
2026-04-12
