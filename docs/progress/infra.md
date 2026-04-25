# Infrastructure (基礎設施) 進度核對

## 模組狀態：穩定，維運機制已完備

## 1. Systemd 服務管理 (infra/systemd/)
- [x] **模組化啟停**: 支援 Perception (L1-L2), Orchestrator (L5), MediaMTX 等服務的獨立生命週期。
- [x] **管理工具**: 透過 `systemctl --user` 管理各服務（saccade-perception, saccade-orchestrator, mediamtx）的啟停與健康檢查。
- [x] **User 模式**: 全面遷移至 `systemctl --user`，確保權限隔離與非 root 安全運行。

## 2. CI/CD 與自動化佈署
- [x] **整合 CI 管線 (.github/workflows/ci.yml)**: 實作 Ruff 靜態檢查、Mypy 嚴格型別檢查與 Pytest 單元測試。
- [x] **C++ 核心建構驗證 (.github/workflows/cpp_build.yml)**: 使用 NVIDIA CUDA 容器自動編譯並驗證 Python 擴充模組。
- [x] **Docker CD 管線 (.github/workflows/docker_build.yml)**: 支援自動建構鏡像並推送至 GHCR (GitHub Container Registry)。
- [x] **依賴快取優化**: 整合 `uv` 快取與 Docker Layer 快取，大幅縮短開發反饋時間。

## 3. MediaMTX 與串流配置 (infra/mediamtx.yml)
- [x] **RTSP/WebRTC 支援**: 已設定統一串流入口，供 Perception 模組接取。
- [x] **低延遲優化**: 已進行 HLS/WebRTC 緩衝調優，適配 L1 感測之即時性。
- [x] **身分驗證**: 為發布 (publish) 與讀取 (read) 動作加入帳號密碼保護。

## 4. 環境與狀態管理
- [x] **Docker 容器化整合**: 開發環境完全由 Docker 管理，確保 CUDA 與 TensorRT 版本一致。

## 待處理

（已清空）

最後更新：2026-04-25
