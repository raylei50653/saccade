# Ops Scripts

本目錄集中放置本地串流、服務控制與多路 demo 啟動腳本。

## 目前腳本

- `run_8stream_perception.py`
  - 啟動 8 路 RTSP 感知 demo。
- `setup_8_streams.sh`
  - 背景推送 8 路 RTSP 測試流。
- `saccade_ctl.sh`
  - 本地服務控制與健康檢查。

## 原則

- 這些腳本偏本地開發與操作，不是主 eval workflow。
- 若腳本依賴特定環境或服務名稱，請在檔頭或 README 註明。
