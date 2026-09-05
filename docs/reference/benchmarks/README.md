# Benchmarks

效能量測數據，用於優化前後比較與容量規劃。

## 文件

| 文件 | 內容 |
|------|------|
| [frozen_v2_ablation.md](frozen_v2_ablation.md) | **現行 headline (`frozen_v2`) 累積消融 + per-seq + 兩操作點延遲**（showcase 附錄與 ADR 018 的可追溯佐證） |
| [reid_handover_ablation_20260808.md](reid_handover_ablation_20260808.md) | **為什麼 `reid_mode: off`**（ReID 買 0.0 IDF1、付 −34% FPS）＋ offline handover +0.4 ＋ live handover −4.5 |
| [bridge_gate_stability_20260808.md](bridge_gate_stability_20260808.md) | **bridge gate 的穩定邊界與參數耦合**：`h_hi` 在 1.7 有真實不連續、出貨 m 落在錯誤側、`h_lo≈0.76` 寬平台（候選變更未套用；**已被下一列否決**） |
| [bridge_gate_cross_dataset_20260808.md](bridge_gate_cross_dataset_20260808.md) | **⛔ 上列候選變更的 cross-dataset 否決**：MOT20 中性 / DanceTrack −0.753，cliff-plateau 結構外部 0/2 不重現 ⇒ 維持出貨 0.6/1.7 |
| [frame_budget_20260905.md](frame_budget_20260905.md) | **每幀 3.02 ms 的 per-kernel 去向**＋**量測邊界**:production 小 kernel 的 exact L2 hit rate 與 per-frame DRAM ledger 在本卡 counter interface 下**不可識別**(非尚未量);ncu 干擾源已歸因為 display scanout |
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
- 重大優化同步回寫 [TODO.md](../../TODO.md)、[PIPELINE.md](../../PIPELINE.md)、對應 ADR，或本目錄下的 benchmark note
