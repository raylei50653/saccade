# MOT17 Default Configuration

> **最後更新：2026-05-11**
>
> 本文檔記錄 MOT17 evaluation pipeline 的實際默認值與推薦配置。
> 所有數值均來自 `mot17_args.py`（CLI defaults）及 `config.py`（parse_eval_config fallbacks）。
>
> **2026-05-08 更新**：CLI defaults 已改為新 baseline（`tracker_core_gmc`）：
> `--reid-mode` = `off`，`--appearance-bank` = `False`，`--semantic-bank-inject` = `False`。
> **2026-05-10 更新**：`--interpolate-tracklets`、`--fp-hard-filter-enabled` 設為 default（True）；
> `--async-reid`、`--pipeline-relink` 也為 default True。

---

## 1. CLI Defaults 總覽

| 參數 | CLI Default | 備註 |
|:-----|:----------:|:-----|
| `--match-thresh` | **0.75** | — |
| `--new-track-thresh` | **0.35** | — |
| `--high-thresh` | 0.45 | — |
| `--track-thresh` | 0.05 | — |
| `--mid-thresh` | 0.10 | — |
| `--conf-threshold` | 0.05 | — |
| `--gmc` | **True** | 預設開啟 |
| `--gmc-mode` | gpu | — |
| `--detection-quality-scaling` | **True** | 預設開啟 |
| `--cross-tile-merge` | **True** | 預設開啟 |
| `--appearance-bank` | **False** | 預設關閉（新 baseline） |
| `--semantic-bank-inject` | **False** | 預設關閉（新 baseline） |
| `--id-stability-filter` | **True** | 預設開啟 |
| `--per-seq-adapt` | **True** | 預設開啟 |
| `--geometry-suspect-support` | **True** | 預設開啟 |
| `--async-reid` | **True** | 預設開啟（2026-05-07） |
| `--pipeline-relink` | **True** | 預設開啟（2026-05-07） |
| `--reid-mode` | **off** | 預設關閉 ReID（新 baseline） |
| `--interpolate-tracklets` | **True** | 預設開啟（2026-05-10） |
| `--interpolate-max-gap` | 20 | — |
| `--interpolate-min-track-len` | 5 | — |
| `--fp-hard-filter-enabled` | **True** | 預設開啟（2026-05-10） |
| `--fp-hard-filter-max-suspicious-area` | 40000 | — |
| `--tiling` | native_960 | — |
| `--engine` | models/yolo/yolo26s_960_batch1.engine | — |

> **注意**：直接執行 `mot17.py --detector SDP` 使用 CLI defaults（match=0.75, ntt=0.35）。
> 使用 `--preset speed/baseline/accuracy` 可套用調參後的最佳值（match=0.66, ntt=0.28）。

---

## 2. 推薦 Baseline 配置（`tracker_core_gmc`）

> **CLI defaults 已更新為新 baseline（2026-05-08）。**
> `uv run scripts/eval/mot17.py --detector SDP` 預設即為 `tracker_core_gmc`。

```bash
uv run python scripts/eval/mot17.py --detector SDP
```

### 預期結果（7-seq SDP）

| Metric | 數值 |
|:-------|:-----|
| IDF1 | **48.8%** |
| MOTA | **40.6%** |
| IDs | **570** |
| FP | 11019 |
| FN | 55100 |
| FPS | 112.0 |

---

## 3. Full Pipeline 配置（`full_default`）

啟用完整 pipeline（GMC + semantic relink + appearance bank + async_reid + pipeline_relink）。

```bash
uv run python scripts/eval/mot17.py \
  --detector SDP \
  --reid-mode semantic \
  --appearance-bank \
  --semantic-bank-inject \
  --async-reid \
  --pipeline-relink \
  --semantic-threshold 0.93
```

> 注意：`--gmc`、`--cross-tile-merge`、`--id-stability-filter` 等已為 CLI 默認值，無需手動指定。

### 預期結果（7-seq SDP）

| Metric | 數值 |
|:-------|:-----|
| IDF1 | 48.8% |
| MOTA | 40.6% |
| IDs | 564 |
| FP | 11072 |
| FN | 55074 |
| FPS | 107.4 |

### Full vs Baseline 差異

| Metric | Baseline | Full Pipeline | Δ |
|:-------|:--------:|:-------------:|:-:|
| IDF1 | 48.8% | 48.8% | ~0pp |
| MOTA | 40.6% | 40.6% | ~0pp |
| IDs | 570 | 564 | -6 |
| FPS | 112.0 | 107.4 | -4.6 |

> **結論**：semantic relink + async_reid + pipeline_relink 的總和增量僅 ~0.1pp IDF1，性價比大幅下降。

---

## 4. 參數詳細說明

### 4.1 關聯（Association）參數

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--track-thresh` | 0.05 | 低閾值關聯下界 |
| `--mid-thresh` | 0.10 | 中間閾值 |
| `--high-thresh` | 0.45 | 高閾值匹配 |
| `--new-track-thresh` | **0.35** | 新 track 分數閾值 |
| `--match-thresh` | **0.75** | 關聯相似度門控 |
| `--confirm-streak` | 1 | 確認 track 所需命中次數 |
| `--confirm-score-thresh` | 0.0 | 確認所需最小分數 |

### 4.2 GMC（Global Motion Compensation）

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--gmc` | **True** | 啟用 GMC |
| `--gmc-mode` | gpu | GPU (cuFFT) 或 CPU (OpenCV LK) |
| `--gmc-downscale` | 8 | GMC 估計下採樣因子 |
| `--gmc-fg-mask` | False | GMC 前景遮罩 |
| `--gmc-pcr-uncertain-thresh` | 8.0 | PCR 不確定性閾值 |

### 4.3 ReID / Appearance

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--reid-mode` | semantic | off/tracker/semantic/hybrid |
| `--reid-model` | siglip2 | embedding model |
| `--reid-budget` | 0.2 | ReID 每幀最大處理比例 |
| `--reid-interval` | 20 | ReID 固定心跳間隔 |
| `--reid-cos-threshold` | 0.90 | 餘弦相似度門控 |
| `--reid-weight` | 0.80 | appearance 在匹配成本中的權重 |
| `--async-reid` | **True** | 非同步 ReID 提取 |
| `--pipeline-relink` | **True** | pipeline relink |

### 4.4 Semantic Relink

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--semantic-threshold` | 0.91 | semantic relink 相似度門控 |
| `--semantic-ttl` | 45 | lost track 在 semantic memory 中的存活幀數 |
| `--semantic-ema` | 0.83 | semantic appearance EMA 衰減 |
| `--semantic-spatial-gate` | 0.20 | spatial gate for relink candidates |
| `--semantic-min-lost-frames` | 2 | lost frames 閾值才考慮 relink |
| `--semantic-min-iou` | 0.20 | semantic relink 最小 IoU |
| `--semantic-buffer-size` | 10 | semantic reference buffer size |
| `--appearance-bank` | **False** | appearance bank（新 baseline 關閉） |
| `--appearance-bank-size` | 5 | per-track bank sample count |
| `--appearance-bank-min-score` | 0.45 | bank sample 最小 detection score |
| `--appearance-bank-min-iou` | 0.35 | bank sample 最小 IoU |
| `--semantic-bank-inject` | **False** | track death 時 inject bank reference（新 baseline 關閉） |

### 4.5 Detection / Tiling

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--engine` | models/yolo/yolo26s_960_batch1.engine | detector engine |
| `--tiling` | native_960 | 推理 tiling preset |
| `--conf-threshold` | 0.05 | detector confidence floor |
| `--cross-tile-merge` | **True** | tile boundary 重複 detection merge |
| `--cross-tile-score-penalty` | 1.0 | cross-tile merge score penalty |
| `--nms-iou-threshold` | None | override detector NMS IoU |
| `--detection-quality-scaling` | **True** | detection quality factor scaling |

### 4.6 Geometry Priors

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--person-geometry-prior` | True | hard geometric filtering |
| `--person-min-height-ratio` | 0.018 | 最小 bbox height 比例 |
| `--person-min-aspect` | 1.0 | 最小 h/w aspect ratio |
| `--person-max-aspect` | 5.5 | 最大 h/w aspect ratio |
| `--person-min-area-ratio` | 0.00006 | 最小 bbox area 比例 |
| `--geometry-suspect-support` | True | suspect geometry 額外支援 |
| `--kalman-r-scale` | 0.75 | Kalman measurement noise scale |

### 4.7 ID Stability

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--id-stability-filter` | **True** | unstable ID handoff 過濾 |
| `--id-stability-min-hits` | 2 | stability check 最小命中數 |
| `--id-stability-min-iou` | 0.05 | stable continuation 最小 IoU |
| `--id-stability-max-center-shift` | 2.0 | stable continuation 最大中心偏移 |
| `--id-stability-max-gap` | 1 | short-term stable 最大 gap |
| `--id-stability-score-ema` | 0.70 | stability confidence EMA |
| `--id-stability-min-score-ema` | 0.15 | stability 最小 EMA score |

### 4.8 Lifecycle Merge

| 參數 | 默認值 | 說明 |
|:-----|:------:|:-----|
| `--lifecycle-merge` | **False** | pre-output lifecycle merge |
| `--lifecycle-ttl` | 45 | mergeable dead track 存活幀數 |
| `--lifecycle-min-gap` | 2 | lifecycle merge 最小 gap |
| `--lifecycle-spatial-gate` | 0.08 | lifecycle merge spatial gate |
| `--lifecycle-sim-threshold` | 0.90 | lifecycle merge 相似度閾值 |
| `--post-lifecycle-merge` | **False** | post-output lifecycle merge |

---

## 5. 快速參考：Baseline vs Full Pipeline

| 模組 | Baseline (`tracker_core_gmc`) | Full Pipeline (`full_default`) |
|:-----|:---:|:---:|
| GMC | ✅ | ✅ |
| Semantic Relink | ❌ | ✅ |
| Appearance Bank | ❌ | ✅ |
| Bank Inject | ❌ | ✅ |
| Async ReID | ✅ | ✅ |
| Pipeline Relink | ✅ | ✅ |
| Lifecycle Merge | ❌ | ❌ |
| Post Lifecycle Merge | ❌ | ❌ |
| **IDF1** | **48.8%** | 48.8% |
| **MOTA** | **40.6%** | 40.6% |
| **IDs** | **570** | 564 |
| **FPS** | **112.0** | 107.4 |
| **CLI 命令** | `mot17.py --detector SDP` | `--reid-mode semantic --appearance-bank --semantic-bank-inject --async-reid --pipeline-relink` |

---

## 6. 開發建議

1. **調參時以 `tracker_core_gmc` 為基準線**，避免調出一套只在無 GMC 時有效的參數。
2. **GMC 之外的高 ROI 方向**：Pose biometric、Tracklet merge、Detection 品質微調。
3. **Semantic relink 診斷結論**：在 GMC 開啟下，`reject_age` = 86.8%，semantic relink 基本是冗余的。
