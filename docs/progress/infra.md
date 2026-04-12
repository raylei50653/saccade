# Infra 模組進度 (2026-04-12)

## 模組狀態：待啟動

## 1. Systemd 服務 (systemd/)
- [ ] **熱切換單元檔**: 支援 Perception 與 Cognition 模組的獨立啟停。
- [ ] **失敗自啟動**: 偵測 GPU 驅動崩潰並嘗試重啟服務。
- [ ] **環境變數注入**: 整合 `.env` 與 Nix Shell 環境。

## 2. MediaMTX 配置 (mediamtx.yml)
- [ ] **串流路徑**: 設定 live/cognition 等不同延遲需求的 Path。
- [ ] **緩衝策略**: 針對 HLS/WebRTC 進行低延遲調優。
- [ ] **身份驗證**: 設定串流接取的密鑰。

## 待處理
- [ ] 完成與 Nix Flakes 的全自動佈署腳本。
- [ ] 本地測試環境 (Docker-compose 選項) 評估。
