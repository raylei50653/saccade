# Infra 模組進度 (2026-04-12)

## 模組狀態：穩定 (Systemd --user 模式已實作)

## 1. Systemd 服務 (infra/systemd/)
- [x] **熱切換單元檔**: 支援 Perception, Orchestrator, MediaMTX 等模組的獨立啟停。
- [x] **管理工具**: 透過 `scripts/saccade` 實作一鍵啟動與健康檢查。
- [x] **User 模式**: 修正權限問題，全面遷移至 `systemctl --user`。

## 2. MediaMTX 配置 (infra/mediamtx.yml)
- [x] **串流路徑**: 已設定 RTSP/WebRTC 通用 Path。
- [ ] **緩衝策略**: 針對 HLS/WebRTC 進行低延遲調優。
- [ ] **身份驗證**: 設定串流接取的密鑰。

## 待處理
- [ ] 完成與 Nix Flakes 的全自動佈署腳本。
- [ ] 本地測試環境 (Docker-compose 選項) 評估。
