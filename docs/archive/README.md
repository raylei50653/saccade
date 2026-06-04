# Archive

> 本目錄存放已結案、廢棄、或 NO-GO 的設計文檔與決策記錄，
> 保留供歷史參考。**不反映當前開發方向。**

## Contents

| 目錄 | 內容 | 封存原因 |
|------|------|----------|
| [option-d/](option-d) | Option D 設計文檔 (4 files) | 已實作訓練完成，但 gate 無實質貢獻 (∆ <0.2pp)，IDF1 31.7% vs baseline 52.0%。2026-05-19 結案 NO-GO。 |
| [adr/](adr) | 已廢棄的 ADR (2 files) | Route map 已被後續 ADR 取代：`refactor_tracker.md` → ADR 013/015/016；`yolo26_siglip2_integration.md` → ADR 005。 |
| [TODO_history.md](../TODO_history.md) | 歷史 TODO 記錄 | 從 `docs/TODO_history.md` 移出的舊結案 workstreams。 |

## 為何保留而非刪除

- Option D 的 `TrackerGateInput` + `TrackSpatialGate` 設計提供 track-conditioned detection 的完整失敗分析
- 廢棄 ADR 記錄了設計空間中被排除的路徑，避免重複評估
- 訓練產物 (`runs/`) 與評估結果 (`results/`) 的清理策略見 `docs/archive/option-d/README.md`
