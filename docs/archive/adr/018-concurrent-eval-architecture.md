# ADR-018: Concurrent Evaluation Architecture

**Status**: Proposed  
**Date**: 2026-05-16  
**Deciders**: Ray (Lead Developer)

---

## Context

MOT17 evaluation 目前採用 **順序執行** 模式：
```python
for seq in cfg.seqs:
    detector.reset_tracker()
    # ... evaluate single sequence
```

- 7 個 SDP 序列總耗時約 40-50 秒
- GPU 利用率僅 20-30%（因為每次只處理一個序列）
- 開發者需要等待較長時間才能看到結果

## Decision

採用 **Thread-per-Sequence + Per-Sequence TRT Context** 的併發架構：

```
┌─────────────────────────────────────────────────────┐
│              ThreadPoolExecutor (N workers)          │
├─────────────────┬─────────────────┬─────────────────┤
│  Thread 1       │  Thread 2       │  Thread N       │
│  (Seq: 02-SDP)  │  (Seq: 04-SDP)  │  (Seq: 13-SDP)  │
├─────────────────┼─────────────────┼─────────────────┤
│  TRT Context 1  │  TRT Context 2  │  TRT Context N  │
│  cudaStream 1   │  cudaStream 2   │  cudaStream N   │
│  Tracker 1      │  Tracker 2      │  Tracker N      │
│  VRAM Buffer 1  │  VRAM Buffer 2  │  VRAM Buffer N  │
└─────────────────┴─────────────────┴─────────────────┘
                    ↕ 共享 (只讀)
          ┌─────────────────────────┐
          │  TRT Engine (single)    │  ← 不重複載入
          └─────────────────────────┘
```

## Rationale

### 為什麼選擇 Thread-per-Sequence？

1. **最小改動**：共用同一 TRT engine，只建立獨立的 context
2. **狀態隔離**：每個序列獨立 tracker + relinker 狀態
3. **VRAM 效率**：engine 只載入一次，context 開銷小
4. **線程安全**：TRT context 的 `setTensorAddress` 非線程安全，但 per-sequence context 避免了競爭

### 為什麼不選其他方案？

| 方案 | 缺點 |
|------|------|
| **Process-per-Sequence** | VRAM 重複載入 engine，進程通訊開銷大 |
| **Batch multiple sequences** | eval 需要 per-sequence 結果，batch 會混合 output |
| **Frame-level pipeline** | 複雜度高，且 eval 是離線任務，非即時需求 |
| **GPU tensor parallelism** | 模型太小（YOLO），parallelism 不划算 |

## Implementation

### 修改檔案

| 檔案 | 修改內容 |
|------|----------|
| `include/perception/trt_engine.hpp` | 增加 `create_context()` 方法宣告 |
| `src/perception/trt_engine.cpp` | 實作 `create_context()` |
| `src/perception/perception_python.cpp` | 暴露 `create_context()` 給 Python |
| `src/saccade/perception/detector_trt.py` | 增加 `enable_concurrent`、`ConcurrentDetectorProxy` |
| `src/saccade/perception/eval/concurrent_detector.py` | 併發 detector 包裝器 |
| `scripts/eval/concurrent_mot17.py` | 測試入口腳本 |

### API 使用

```python
# 方式 1：直接使用 TRTYoloDetector 的 concurrent mode
from saccade.perception.detector_trt import TRTYoloDetector

detector = TRTYoloDetector(
    engine_path="models/yolo/yolo26s_960_batch1.engine",
    enable_concurrent=True,
)

# 為每個序列取得獨立 context
for seq in sequences:
    seq_det = detector.get_seq_context(seq)
    result = evaluate_sequence(seq_det, seq_data)
```

```python
# 方式 2：使用併發評估腳本
uv run python scripts/eval/concurrent_mot17.py \
    --preset speed \
    --max-workers 8 \
    --detector SDP
```

## Expected Performance

| Metric | Sequential | Concurrent (8 workers) |
|--------|-----------|----------------------|
| Total wall time | ~45s | ~8-10s |
| Speedup | 1× | 4.5× - 5.5× |
| GPU utilization | ~25% | ~80-90% |
| VRAM peak | ~6GB | ~8-10GB |

## Trade-offs

### Pros
- 開發迭代速度大幅提升
- GPU 利用率提高
- 可選啟用（不影響現有流程）
- 每個序列結果獨立，方便 ablation

### Cons
- 需要修改 C++ extension（需重新編譯）
- VRAM 使用量增加（每個序列 ~1GB buffer）
- 複雜度略增（需管理 context 生命週期）

## Open Questions

1. **VRAM 管理**：是否需要動態根據 VRAM 調整 max_workers？
2. **Context 釋放**：目前 context 建立後何時釋放？（需要 pybind11 支援 delete）
3. **Profile 合併**：階段 profiling 結果如何合併？

## Next Steps

1. [ ] 編譯 C++ extension（測試 create_context）
2. [ ] 單元測試併發 detector
3. [ ] 效能 benchmark（順序 vs 併發）
4. [ ] 整合到 CI workflow
5. [ ] 文件更新（docs/TODO.md）

## References

- ADR-013: GPUByteTracker Saccade Heartbeat
- ADR-010: DALI GPU Preprocessing
- [concurrent_mot17.py](/scripts/eval/concurrent_mot17.py)
- [concurrent_detector.py](/src/saccade/perception/eval/concurrent_detector.py)
