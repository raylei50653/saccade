# Infrastructure (基礎設施) 進度核對

## 模組狀態：穩定 (CI/CD 與 Systemd 服務管理已完備)

## 1. Systemd 服務管理 (infra/systemd/)
- [x] **模組化啟停**: 支援 Perception (L1-L2), Orchestrator (L5), MediaMTX 等服務的獨立生命週期。
- [x] **管理工具**: 透過 `scripts/saccade` 實作一鍵啟動、停止與即時健康檢查。
- [x] **User 模式**: 全面遷移至 `systemctl --user`，確保權限隔離與非 root 安全運行。

## 2. CI/CD 與自動化佈署
- [x] **整合 CI 管線 (.github/workflows/ci.yml)**: 實作 Ruff 靜態檢查、Mypy 嚴格型別檢查與 Pytest 單元測試。
- [x] **C++ 核心建構驗證 (.github/workflows/cpp_build.yml)**: 使用 NVIDIA CUDA 容器自動編譯並驗證 Python 擴充模組。
- [x] **Docker CD 管線 (.github/workflows/docker_build.yml)**: 支援自動建構鏡像並推送至 GHCR (GitHub Container Registry)。
- [x] **依賴快取優化**: 整合 `uv` 快取與 Docker Layer 快取，大幅縮短開發反饋時間。

## 3. MediaMTX 與串流配置 (infra/mediamtx.yml)
- [x] **RTSP/WebRTC 支援**: 已設定統一串流入口，供 Perception 模組接取。
- [x] **低延遲優化**: 已進行 HLS/WebRTC 緩衝調優，適配 L1 感測之即時性。
- [ ] **身分驗證**: 串流存取密鑰管理 (待落實)。

## 4. 環境與狀態管理
- [x] **Docker 容器化整合**: 開發環境完全由 Docker 管理，確保 CUDA 與 TensorRT 版本一致。
- [ ] **自動化佈署**: 完善生產環境一鍵佈署腳本。

## 待處理
- [ ] 完成針對 L3-L4 存儲負載的 Redis 自動清理腳本。
- [ ] 整合 VRAM 監控系統至 Orchestrator。

最後更新：2026-04-13
