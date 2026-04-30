# Benchmark Scripts

本目錄集中放置效能量測、壓力測試與延遲拆解腳本。

## 目前腳本

- `benchmark_16_streams.py`
  - 舊版多串流壓力測試腳本。
- `benchmark_association.py`
  - GPU tracker association micro-benchmark。
- `bottleneck_annealer.py`
  - 逐步加壓探索系統瓶頸與崩潰點。
- `latency_breakdown.py`
  - 拆解 detector / feature extractor 的分段延遲。
- `latency_e2e_report.py`
  - 以 MOT17 eval path 量測端到端延遲。

## 原則

- 這些腳本不是主 workflow 入口。
- 若量測結果需要長期保留，請回寫到 `/docs/benchmarks/`。
- 若腳本已失效但仍需保留歷史脈絡，請在檔頭標註適用範圍。
