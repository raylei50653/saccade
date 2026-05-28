# Throughput Benchmark

## 🚀 性能里程碑 (Performance Milestones)

| 日期 | 版本 | 場景 | 吞吐量 (FPS) | 延遲 (Avg) | 備註 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04-10 | v0.5 | 1080p Single Stream | 150 | 6.7ms | 初始 Python 版本 |
| 2026-04-18 | v0.8 | 10-Stream Aggregate | 1200 | 0.83ms | 遷移至 C++ Core |
| **2026-04-25** | **v1.0** | **Hybrid Association** | **2300+** | **0.42ms** | **ADR 015: Sinkhorn-Auction Fusion** |

## 📊 詳細數據 (Detailed Metrics - v1.0)

### 1. 關聯階段延遲 (Association Latency)
測試環境: NVIDIA Blackwell GPU, Max Objects: 1024

| 偵測框數量 (Dets) | P50 (ms) | P95 (ms) | P99 (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- |
| 100 | 0.36 | 0.64 | 1.75 | 2358 |
| 300 | 0.38 | 0.71 | 1.85 | 2233 |
| 500 | 0.38 | 0.64 | 1.72 | 2242 |
| 1000 | 0.41 | 0.69 | 1.94 | 2111 |

### 2. 分析與結論
- **零拷貝優勢**: 通過完全消除 Device-to-Host 同步，關聯延遲降低了約 50%。
- **運算融合**: Fused Sinkhorn-TopK Kernel 成功掩蓋了 $O(N \cdot M)$ 的計算開銷，使 100 與 1000 目標規模下的性能差異小於 15%。
- **壓力承受度**: Parallel Auction 在 1000 目標下依然展現出極高的吞吐量，未發生顯著的原子操作競爭阻塞。

## MOT17 SDP End-to-End Tracking A/B (2026-04-26)

評估命令使用 `scripts/eval/mot17.py`（CLI wrapper，主邏輯位於 `perception/eval/runner.py`），YOLO engine 為 `models/yolo/yolo26s_batch4.engine`，ReID 關閉，序列為 MOT17 train split 的 7 個 SDP sequence。

| Config | Output | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA | FPS |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| Baseline off | `results/reid_sdp_off` | 44.6% | 44.4% | 80.1% | 12,432 | 62,396 | 604 | 32.8% | 154.67 |
| Tentative default probe (`mid=0.25`, `confirm_score=0.40`) | `results/tentative_sdp_off` | 45.8% | 48.5% | 75.2% | 17,959 | 57,872 | 656 | 31.9% | 151.75 |
| Tentative tuned default (`mid=0.40`, `confirm_score=0.50`) | `results/tentative_sdp_m040_c050` | 45.5% | 45.9% | 78.6% | 14,005 | 60,786 | 575 | 32.9% | 149.94 |
| Tentative geometry scale (`mid=0.40`, `confirm_score=0.50`, median-height scale) | `results/tentative_sdp_geometry` | 45.0% | 45.5% | 78.8% | 13,724 | 61,199 | 548 | 32.8% | 153.29 |
| Tentative geometry EMA (`beta=0.80`, `max_step=0.05`) | `results/tentative_sdp_geometry_ema` | 44.8% | 45.3% | 78.9% | 13,610 | 61,408 | 551 | 32.7% | 146.88 |
| Tentative median EMA (`beta=0.80`, `loosen=0.08`, `tighten=0.03`, `min_samples=5`) | `results/tentative_sdp_geometry_median_ema` | 44.7% | 45.3% | 78.9% | 13,609 | 61,460 | 545 | 32.7% | 144.77 |
| Adaptive confirmation, fixed threshold | `results/tentative_sdp_adaptive_fixed` | 45.3% | 45.5% | 78.7% | 13,863 | 61,172 | 563 | 32.7% | 148.94 |
| Adaptive confirmation + geometry | `results/tentative_sdp_geometry_adaptive_v2` | 44.7% | 45.1% | 78.9% | 13,567 | 61,697 | 541 | 32.5% | 144.52 |

結論：收緊後的 Tentative/Confirmed 狀態機小幅提升 MOTA、提升 IDF1，並降低 ID switches；原始 `mid=0.25` 能提高 Recall，但 FP 增幅過大，不適合作為預設值。
幾何縮放模式可作為 ID-stability profile：IDs 最低、吞吐接近 baseline，但 IDF1/MOTA 較 tuned default 低。
EMA + step clamp 版本的 IDs 仍低，但 MOTA/IDF1/FPS 均弱於 raw geometry；目前不作預設 accuracy profile。
median EMA + asymmetric step 是最保守的 ID-stability profile：IDs 最低，但 FN/FPS 代價最高。
Adaptive confirmation 進一步降低 IDs，但目前 FN/MOTA 代價不划算；保留為 `--adaptive-confirmation` 實驗開關。

### Lazy ReID Arbiter Profiling

`MOT17-09-SDP --max-frames 80 --profile-stages --profile-lazy-reid-embeddings --reid-model transreid`

| 指標 | 數值 |
| :--- | ---: |
| Lazy ReID candidates | `0.20/frame` (`16 total`) |
| Lazy ReID crops | `0.44/frame` |
| Self-consistency pairs | `17` |
| Mean cosine | `0.852` |
| Pass@0.85 | `58.8%` |
| Arbiter dry-run approve | `10 / 16` (`62.5%`) |
| Synchronous profile cost | `2.32 ms/frame` |

結論：Lazy ReID arbiter 的候選面很小，且 dry-run 顯示有實際判別力；它比較像 Tentative 轉正的延遲驗證器，而不是全量 ReID 關聯替代品。

**2026-04-26 修正**：SmartTracker 改為非阻塞 Heartbeat ReID。`_submit_reid_async` 在獨立 `_reid_stream` 上提交 crop + extract，主 stream 立刻返回；下一幀 `_poll_reid()` 以 `event.query()`（non-blocking）確認完成後才搬移到 `_ready_reid`。FeatureBank 和 C++ tracker 的 reference feature 更新始終在主 stream 進行，無 race condition。代價：association 使用前一 heartbeat 的 embeddings（lag ≤ 1 frame / 10ms @100fps），在 10-frame heartbeat interval 下影響可忽略。

## Kernel 優化 (2026-04-26)

### TopK + Auction 改版

| 改動 | Before | After |
|------|--------|-------|
| `fused_sinkhorn_topk_kernel` 每 block | 1 warp (32 threads) | 4 warps (128 threads) + shared mem tree reduction |
| `parallel_auction_kernel` atomic 策略 | 全 track 直接 global `atomicMax` | Level-1 intra-block shared mem atomicMax → Level-2 block winner 才寫 global |

**理論效益**：
- 128-thread TopK：block occupancy 從 1 warp 提升到 4 warps，GPU scheduler 調度抖動降低；static shared mem `s_vals[3][128]` / `s_idxs[3][128]`，用 7 階 tree reduction 取代 5 階 warp shuffle reduction。
- Two-level auction：惡化情境（多 track 競標同一目標）下 global atomicMax 次數從 O(n_trk) 降為 O(n_blocks)；shared mem price cache 消除 L2 反覆讀寫。

RTX 5070 Ti Laptop 量測（full update() 含 D2H，非純 kernel）：

| Dets | P50 | P95 | P99 | P99/P50 |
|------|-----|-----|-----|---------|
| 100  | 0.729 ms | 1.518 ms | 2.194 ms | 3.0× |
| 300  | 1.235 ms | 2.091 ms | 2.823 ms | 2.3× |
| 500  | 2.317 ms | 3.254 ms | 4.151 ms | 1.8× |
| 1000 | 4.623 ms | 5.625 ms | 6.405 ms | 1.4× |

## Person-Only Top-K Detector Experiment (2026-05-06)

Goal: test whether filtering to `person` before YOLO end-to-end top-k, and raising the detector cap from `300` to `1000`, improves crowded-scene tracking enough to justify a default switch.

Artifacts:

- Base engine: `models/yolo/yolo26s_batch4.engine`
- Experimental engine: `models/yolo/yolo26s_person_topk1000_batch4.engine`
- Export helper: `scripts/model/export_yolo_person.py`

### Detector-only latency

Batch-4 benchmark (`scripts/eval/bench_yolo_batch.py`):

| Engine | Mean latency | FPS |
| :--- | ---: | ---: |
| `yolo26s_batch4.engine` | `12.63 ms` | `316.7` |
| `yolo26s_person_topk1000_batch4.engine` | `13.71 ms` | `291.8` |

The person-only top-k1000 engine is about `8.5%` slower on detector-only throughput.

### Crowded-frame detector effect

On a crowded MOT20-08 frame (`000601.jpg`, 640×640 resize, `conf=0.001`):

| Engine | Total detections | Person detections |
| :--- | ---: | ---: |
| `yolo26s_batch4.engine` | `300` | `169` |
| `yolo26s_person_topk1000_batch4.engine` | `499` | `499` |

So the structural detector fix works: the old engine is capped by the global top-k, while the new engine preserves many additional low-score person boxes.

### End-to-end tracking summary

Short MOT17 crowded-sequence check (`MOT17-04-SDP,MOT17-10-SDP`, first 100 frames each, `--reid-mode off`):

| Config | IDF1 | MOTA | FP | FN | IDs | FPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| New engine, default thresholds | `8.5%` | `3.7%` | `606` | `57,528` | `27` | `60.67` |
| New engine, global low thresholds (`0.02/0.25`) | `8.9%` | `3.7%` | `795` | `57,307` | `33` | `43.18` |
| New engine, crowd-aware Python switch (`trigger=25`) | `8.9%` | `3.7%` | `782` | `57,316` | `39` | `65.22` |

### Decision

- Keep the person-only top-k1000 engine as an experimental artifact.
- Do not switch the default detector engine or default thresholds yet.
- The detector-side fix is real, but most recovered boxes remain low-score under current score calibration.
- Future work, if resumed, should move crowded-scene handling into tracker-internal logic instead of per-frame Python parameter switching.

## 📈 未來優化方向
- 探索 8-bit 量化代價矩陣以減少暫存器壓力。
- `compute_cost_matrix_kernel`：加入 shared memory tiling 降低 trk_embeds 非合并讀取。
