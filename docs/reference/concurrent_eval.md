# Concurrent MOT17 Evaluation Guide

## 概述

Saccade 支援透過 `BatchingTRTDetector` + per-thread `Workbench` 的架構，
在單一 TRT Engine 上同時執行多個 MOT17 序列評估。

> **Note**: 早期設計文件（ADR-018）描述的 `ConcurrentDetectorProxy` / `concurrent_mot17.py`
> 方案已廢棄，從未正式上線。實際架構見本文。

## 架構

```
┌──────────────────────────────────────────────────────────────┐
│                  ThreadPoolExecutor (N workers)               │
├────────────────┬────────────────┬──────┬──────────────────── ┤
│  Thread 1      │  Thread 2      │  … N │                     │
│  Workbench     │  Workbench     │      │  ← 各有獨立 tracker  │
│  cudaStream    │  cudaStream    │      │  ← 各有獨立 stream   │
│  out_buffers   │  out_buffers   │      │  ← 各有獨立 output   │
└───────┬────────┴───────┬────────┴──┬───┘                     │
        │  submit(frame) │           │                          │
        ▼                ▼           ▼                          │
┌──────────────────────────────────────────────────────────────┤
│         BatchingTRTDetector  (single server thread)          │
│  • 等待 N 個 pending 或 drain_ms=2ms timeout                  │
│  • 組裝 batch → infer_raw_batch() → 分發 output[i:i+1]       │
│  • 每個 worker 透過 done.wait() + stream.wait_event() 取回    │
└──────────────────────────────────────────────────────────────┘
                          ↓
              ┌────────────────────────┐
              │  TRT Engine (單一載入)  │
              └────────────────────────┘
```

## 使用方法

### 基本指令

```bash
# workbench 模式，單執行緒（Workbench 路徑，無 concurrent TRT）
uv run python scripts/eval/mot17.py --preset speed --workbench --threads 1

# workbench 模式，4 執行緒（需 batch-4 或更大的 engine）
uv run python scripts/eval/mot17.py --preset speed --workbench --threads 4

# 傳統非 workbench 模式（最佳 IDF1 基準）
uv run python scripts/eval/mot17.py --preset speed
```

### Batch Engine 自動選擇

`--threads N` 啟動時會依序嘗試：

1. 同目錄下 `_batchN.engine`（完全對應）
2. 同前綴的 `_batchM.engine`，M ≥ N 中最小者
3. 全部找不到 → fallback 到原本 engine，`batch_size=1`（等同序列模式）

```
[範例] --engine yolo26s_960_batch1.engine --threads 4
  → 找到 yolo26s_960_batch4.engine (dynamic, 支援 1–4)
  → 使用 batch_size=4
```

## 效能數字（MOT17 7 SDP 序列，2026-05-16）

| 模式 | IDF1 | MOTA | IDs | Rcll | Wall-clock |
|------|------|------|-----|------|-----------|
| Sequential（非 workbench，基準） | 52.0% | 41.6% | 475 | 55.0% | ~62s |
| Workbench 1-thread | 49.3% | 39.8% | 627 | 54.9% | 39.9s |
| Workbench 4-thread | 45.0% | 33.4% | 710 | 48.4% | 45.4s |

**Raw TRT throughput**（不含追蹤）:
- batch-1 直接: 3.21ms/frame = **311 FPS**
- batch-4 via BatchingTRTDetector × 4 threads = 7.04ms/thread, **568 FPS total**

## 已知限制

### 1. 追蹤品質下降
Workbench 路徑的 IDF1/MOTA 低於非 workbench 路徑（~2–7pp）。
根本原因尚未完全釐清，可能與 `BatchedDetectorProxy` 的 stateless 性質影響
tracker 初始化時機有關。**Workbench 路徑主要用於吞吐量測試，非精度基準。**

### 2. 多執行緒不加速 Wall-clock
4-thread（45.4s）比 1-thread（39.9s）更慢，原因：
- `drain_ms=2ms` 在每 frame 等待 batch 組裝，增加延遲
- MOT17-04-SDP 有 1050 frames，是其他序列的 2× 長；4-thread 時它獨佔最後一個 worker

### 3. 非 batch-N engine 時退化
若找不到合適的 batch engine，`batch_size` 降為 1，concurrent 失去意義。

## CUDA Stream 同步

`Workbench.process_detections()` 使用雙向 event handshake：

```python
# 等 calling thread 的 stream 完成寫入再讓 workbench stream 讀
caller_event = torch.cuda.Event()
caller_event.record()                  # 記錄在 calling stream
with torch.cuda.stream(self.stream):
    self.stream.wait_event(caller_event)   # workbench 等 calling
    n = self._wb.process_frame_postyolo(...)
    wb_done = torch.cuda.Event()
    wb_done.record(self.stream)        # 記錄 workbench 完成
torch.cuda.current_stream().wait_event(wb_done)  # calling 等 workbench
```

這是必要的：不做會導致零值 box 或全空輸出（在 4-thread 測試中重現過）。

## 相關檔案

| 檔案 | 說明 |
|------|------|
| `scripts/eval/mot17.py` | 主入口，`--workbench --threads N` 路徑 |
| `src/saccade/perception/detector_trt.py` | `BatchingTRTDetector` + `BatchedDetectorProxy` |
| `src/saccade/perception/workbench.py` | `Workbench`（Python wrapper over C++ `_WorkbenchExt`） |
| `src/saccade/perception/eval/runner.py` | `run_eval()` 共用 entry point |

## 實作狀態（2026-05-16）

### 已完成
| 項目 | 說明 |
|------|------|
| `BatchingTRTDetector` server thread | `detector_trt.py` |
| `BatchedDetectorProxy` per-thread proxy | `detector_trt.py` |
| `Workbench` Python wrapper | `workbench.py` |
| Batch engine 自動選擇 | `mot17.py` L129–153 |
| CUDA stream 雙向 event handshake | `workbench.py` process_detections() |

### 已廢棄
| 項目 | 說明 |
|------|------|
| `ConcurrentDetectorProxy` | 早期設計，未上線 |
| `concurrent_mot17.py` | 早期測試腳本 |
| `TRTYoloDetector.enable_concurrent` | 早期 C++ 方案 |
| ADR-018 | 描述廢棄方案，歷史參考 |

### 已修復 Bug（2026-05-16）
1. **Batch size mismatch**：`--threads 2` 無 batch-2 engine → 自動選擇邏輯
2. **CUDA stream input race**：caller_event + wait_event 修復
3. **CUDA stream output race**：wb_done + wait_event 修復
