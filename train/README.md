# train/

訓練腳本根目錄。複用 `src/saccade/perception/` 的模型與資料集程式碼，
不重複實作，只處理訓練流程（optimizer、scheduler、checkpoint、loop）。

## 子目錄

| 目錄 | 方向 | 狀態 |
|------|------|------|
| [temporal_yolo/](temporal_yolo/) | YOLO + Track Queries 時序追蹤 | Option C ✅  Option D 實作中 |

## 設計原則

- 每個子目錄對應一個獨立的研究方向
- 複用 `src/` 的模型程式碼，不 copy
- configs/ 存放超參數，腳本只處理訓練邏輯
- 所有腳本從專案根目錄執行：`uv run train/<subdir>/<script>.py`
