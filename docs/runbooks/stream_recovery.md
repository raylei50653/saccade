# Runbook: 串流恢復機制 (Stream Recovery)

Saccade 依賴穩定影像源。當 RTSP 斷流或 MediaMTX 重啟時，系統具備自動恢復機制。

## 1. 斷流判定
- `MediaMTXClient` 持續 5 秒無法透過 `grab_tensor()` 獲取影格。
- `PipelineOrchestrator` 收到 Redis 事件流中斷警報。

## 2. 恢復流程
1. **重置解碼器**: 系統會調用 `media.release()` 釋放現有的 GStreamer Pipe。
2. **指數退避重連**: `media.connect()` 會以 1s, 2s, 4s, 8s 的間隔嘗試重新建立連線。
3. **狀態清除**: 斷流期間，`SmartTracker` 會清除所有遺失超過 10 秒的物件 ID。

## 3. 測試恢復
執行 `./scripts/stream_test.sh` 模擬斷流情況，檢查系統是否在 10 秒內恢復推流。
