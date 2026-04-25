# Benchmarks

本目錄收錄 Saccade 的效能量測結果，作為優化前後比較與容量規劃依據。

## 文件索引

- [latency_log.md](latency_log.md): 端到端延遲、分段延遲與觀測紀錄。
- [throughput.md](throughput.md): 單路與多路串流吞吐量結果。
- [vram_usage.md](vram_usage.md): 不同配置下的 VRAM 使用量與壓力觀察。

## 維護原則

- 更新 benchmark 時，盡量附上測試條件，例如 GPU、輸入解析度、串流數與模型版本。
- 若數據來自重大優化，應同步回寫 `progress/` 或對應 ADR。
