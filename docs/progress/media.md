# Media 模組進度 (2026-04-12)

## 模組狀態：待啟動

## 1. MediaMTX 用戶端 (mediamtx_client.py)
- [ ] **串流接取**: 實作 RTSP/WebRTC 協議對接。
- [ ] **緩衝管理**: 針對高延遲環境配置合適的 Jitter Buffer。
- [ ] **狀態監控**: 定期回報 MediaMTX 串流可用性。

## 2. FFmpeg 工具集 (ffmpeg_utils.py)
- [ ] **NVENC/NVDEC**: 封裝 NVIDIA 硬體編解碼工具函式。
- [ ] **零拷貝對接**: 確保解碼後的 NVMM 緩衝區能正確傳遞給 Perception 模組。
- [ ] **轉碼模板**: 提供低延遲、高壓縮率的轉碼 Profile。

## 待處理
- [ ] MediaMTX 設定檔 (`infra/mediamtx.yml`) 整合測試。
- [ ] 支援多路串流同時並行處理。
