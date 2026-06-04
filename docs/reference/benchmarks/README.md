# Benchmarks

效能量測數據，用於優化前後比較與容量規劃。

## 文件

| 文件 | 內容 |
|------|------|
| [latency_log.md](latency_log.md) | 端到端延遲、分段延遲紀錄 |
| [throughput.md](throughput.md) | 單路與多路吞吐量 |

## Baseline

```bash
./scripts/eval/module_benchmark.sh --mode all
```

- detector: `SDP`, sequences: `MOT17-04-SDP,MOT17-10-SDP`
- engine: `models/yolo/yolo26s_960_batch1.engine`
- validate: 65 FPS, 15.4ms

詳細規格見 [PIPELINE_REFERENCE.md](../PIPELINE_REFERENCE.md)。

## 維護

- 更新時附上 GPU、解析度、模型版本
- 重大優化同步回寫 `progress/` 或對應 ADR
