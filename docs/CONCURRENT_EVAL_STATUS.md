# 併發評估實現狀態

更新時間：2026-05-16

## 結論：已上線，但有明確限制

`mot17.py --workbench --threads N` 路徑功能正常，可用於**吞吐量測試**。
不適合作為 IDF1/MOTA 精度基準（見下方品質數字）。

## 已完成項目

| 項目 | 狀態 | 說明 |
|------|------|------|
| `BatchingTRTDetector` server thread | ✅ | `detector_trt.py` |
| `BatchedDetectorProxy` per-thread proxy | ✅ | `detector_trt.py` |
| `Workbench` Python wrapper | ✅ | `workbench.py` |
| Batch engine 自動選擇 | ✅ | `mot17.py` L129–153 |
| CUDA stream 雙向 event handshake | ✅ | `workbench.py` process_detections() |
| 4-thread 正確性驗證 | ✅ | 2026-05-16 測試通過 |

## 已廢棄（不存在於正式路徑）

| 項目 | 說明 |
|------|------|
| `ConcurrentDetectorProxy` | 早期設計，未上線 |
| `concurrent_mot17.py` | 早期測試腳本，使用廢棄 API |
| `TRTYoloDetector.enable_concurrent` / `create_context()` | 早期 C++ 方案，未合入主路徑 |
| ADR-018 | 描述廢棄方案，僅供歷史參考 |

## 品質數字（MOT17 7 SDP 序列）

| 模式 | IDF1 | MOTA | IDs | Rcll | Wall-clock |
|------|------|------|-----|------|-----------|
| Sequential 非 workbench（精度基準） | 52.0% | 41.6% | 475 | 55.0% | ~62s |
| Workbench 1-thread | 49.3% | 39.8% | 627 | 54.9% | 39.9s |
| Workbench 4-thread | 45.0% | 33.4% | 710 | 48.4% | 45.4s |

**Wall-clock 觀察**：4-thread 比 1-thread 慢（45.4s vs 39.9s）。
主要原因：drain_ms=2ms 每 frame 等待 overhead + MOT17-04-SDP（1050 frames）長尾效應。

**Raw TRT 吞吐量**（不含追蹤）：
- batch-1 直接：311 FPS
- batch-4 × 4 threads：568 FPS total（1.83× speedup）

## 已修復的 Bug（2026-05-16）

1. **Batch size mismatch**：`--threads 2` 但無 batch-2 engine，server 以 batch-1 engine 試圖做 batch-2 → worker 2 取得空 tensor `[0,300,6]`。
   修復：batch engine 自動選擇邏輯。

2. **CUDA stream input race**：`boxes/scores/classes` 在 calling thread stream 寫入，workbench stream 未等待即讀 raw pointer → 零值或 stale data。
   修復：`caller_event` + `self.stream.wait_event(caller_event)`。

3. **CUDA stream output race**：`_wb.process_frame_postyolo` 在 `self.stream` 寫入 output buffer，calling thread 未等待即讀 → 全零輸出。
   修復：`wb_done` + `torch.cuda.current_stream().wait_event(wb_done)`。

## 後續方向（如需改善）

- 分析 workbench 路徑 IDF1 降低 2–7pp 的根因
- 考慮 drain_ms 動態調整（長序列降低 drain 等待）
- 負載均衡（先派長序列給 worker 以減少長尾）
