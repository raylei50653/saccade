# Benchmarks

本目錄收錄 Saccade 的效能量測結果，作為優化前後比較與容量規劃依據。

## 文件索引

- [latency_log.md](latency_log.md): 端到端延遲、分段延遲與觀測紀錄。
- [throughput.md](throughput.md): 單路與多路串流吞吐量結果。
- [vram_usage.md](vram_usage.md): 不同配置下的 VRAM 使用量與壓力觀察。
- [optimization_phases_abcd.md](optimization_phases_abcd.md): 非推理開銷清除計畫的階段性量測與結果摘要。

## 目前標準 baseline

模塊級 benchmark 的現行固定入口是：

```bash
./scripts/eval/module_benchmark.sh --mode all
```

第一組已落地並可直接比對的 baseline 是：

- `results/module_benchmark/baseline_native_960`

對應配置：

- detector: `SDP`
- sequences: `MOT17-04-SDP,MOT17-10-SDP`
- engine: `models/yolo/yolo26s_960_batch1.engine`
- tiling: `native_960`
- max_frames: `100`

摘要結果：

- validate: `65.12 FPS`, `15.36 ms`, `IDF1 9.4%`, `MOTA 4.5%`
- profile main stages: `detect 4.53 ms`, `postprocess 1.40 ms`, `track 1.10 ms`, `reid_extract 1.88 ms`, `relink_write 3.46 ms`
- current best ablation signal: `geometry mid-scale` (`IDF1 9.9%`, `MOTA 4.7%`, `IDs 14`)

詳細流程與紀錄格式請看：

- [../PIPELINE_REFERENCE.md](../PIPELINE_REFERENCE.md)
- `results/module_benchmark/baseline_native_960/notes.md`
- `results/module_benchmark/baseline_native_960/experiment_matrix.md`

## 維護原則

- 更新 benchmark 時，盡量附上測試條件，例如 GPU、輸入解析度、串流數與模型版本。
- 若數據來自重大優化，應同步回寫 `progress/` 或對應 ADR。
