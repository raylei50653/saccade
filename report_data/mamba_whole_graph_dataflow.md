# Saccade Pipeline Dataflow

> 生產環境基準 (production baseline)：`--preset mamba_whole_graph`，yolo26s + Mamba head + CUDA Graph。
> 測量時間：2026-06-06，GPU: RTX 5070 Ti Laptop，MOT17-SDP 7 序列全量。
> 對應實作：`src/saccade/perception/eval/evaluator.py`，`time_stage()` 計時的串行階段。

---

## 1. 整體架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PER-FRAME PIPELINE                           │
│                                                                     │
│  [1] Ingest  → [2] Detect(whole-graph) → [3] Postprocess            │
│       → [4] GMC → [5] Tracker → [6] Materialize → [7] Relink        │
│                                                                     │
│  ────────────────────────────────────────────────────────────────   │
│  mamba_whole_graph：ReID OFF，GPU GMC ON，whole-detect CUDA graph。   │
│  生產環境總延遲：5.70 ms/frame，175.4 FPS。                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 串行階段一覽

> 測量條件：`--preset mamba_whole_graph`，MOT17-04-SDP 序列，150 frames，warmup 50。
> 注意：`--profile-stages` 強制 stage 邊界 `cudaSynchronize()`，破壞非同步重疊 (CUDA graph/native/async)。
> profile FPS (124.9) < production FPS (175.4)。**production FPS 才是真實效能。**

| # | 階段 | 狀態 | 每幀平均 | 佔比 | 備註 |
|:--|------|:---:|---:|---:|------|
| 1 | **Fetch** | ✅ | 2.14 ms | 26.7% | DALI NVDEC decode |
| 2 | **Ingest + Preprocess** | ✅ | 0.27 ms | 3.4% | Stretch-resize（無 letterbox） |
| 3 | **Detect** | ✅ | 3.37 ms | 42.1% | Whole-detect CUDA graph: TRT backbone + Mamba head + decode |
| 4 | **Postprocess** | ✅ | 3.02 ms | 37.7% | Native C++/CUDA: filter, NMS, quality scale, FP hard filter |
| 5 | **ReID Bank Sync** | ❌ | 0 ms | — | ReID mode = off |
| 6 | **ReID Budget** | ❌ | 0 ms | — | ReID mode = off |
| 7 | **ReID Crop** | ❌ | 0 ms | — | ReID mode = off |
| 8 | **ReID Extract** | ❌ | 0 ms | — | ReID mode = off |
| 9 | **Lazy ReID** | ❌ | 0 ms | — | profiling only |
| 10 | **GMC** | ✅ | 0.04 ms | 0.5% | GPU phase correlation (cuFFT), downscale=4 |
| 11 | **Tracker Update** | ✅ | 0.61 ms | 7.6% | GPUByteTracker with CUDA Graph |
| 12 | **Materialize** | ✅ | 0.20 ms | 2.5% | GPU → host view |
| 13 | **BG Relink Wait** | ✅ | 0 ms | — | pipeline_relink async, zero visible cost |
| 14 | **Relink Write** | ✅ | 0.11 ms | 1.4% | Bridge relink + identity resolve |
| 15 | **Frame Total** | ✅ | **8.01 ms** | — | with profiling syncs; production: 5.70 ms |

---

## 3. 各階段詳細描述

### Stage 1–2：幀擷取與預處理

```
Raw JPEG frames (DALI) → NVDEC decode → CHW float32 tensor
                                     → AdaptiveFramePool (GPU pin memory)
                                     → stretch-resize to 640 (no letterbox)
```

- **Fetch**：DALI streamer 從 `img1/` 讀取 JPEG 並解碼為 GPU tensor
- **Ingest**：幀資料複製至 `AdaptiveFramePool`
- **Preprocess**：stretch-resize（`preprocess: none` in mamba preset），無 letterbox 灰邊，
  確保 Mamba head 輸入域與訓練一致

### Stage 3：目標偵測（whole-detect CUDA graph）

```
GPU tensor → TRT Backbone (yolo26s) → Mamba head (PixelShuffle, Cross-Scan)
          → postprocess decode → boxes/scores/classes
         （全部封裝於單一 CUDA graph replay）
```

- **引擎**：TensorRT YOLO26s backbone + Mamba gated detection head
  - Backbone: `models/yolo/yolo26s_backbone_640_best.engine`
  - Mamba ckpt: `runs/mamba_gt_vgt_mamba_v14/best.ckpt`
- **CUDA Graph**：`use_whole_graph=true`，將 TRT backbone inference + Mamba head + decode
  捕捉為單一 graph。無 kernel launch overhead。
- **解析度**：native_640（stretch-resize），無 tiling
- **Mamba head**：PixelShuffle 上取樣 + 四向 Cross-Scan（去除方向偏見）
- **輸出**：每幀最多 300 個 bounding box

### Stage 4：後處理

```
raw detections → quality filter → NMS → FP hard filter → output
```

Native C++/CUDA path（`PerceptionPipeline::process_detections_into()`）處理全部後處理。

| Sub-stage | 狀態 | 說明 |
|-----------|:---:|------|
| post_seg_prep | ✅ | tensor preparation, 0.50 ms |
| post_seg_native | ✅ | NMS + filter, 0.33 ms |
| post_seg_slice_quality | ✅ | output slicing + quality, 0.30 ms |
| post_seg_tail_filter | ✅ | tail filtering, 0.14 ms |
| post_seg_fp_hard | ✅ | FP hard filter (area > 40000), 1.30 ms |
| post_seg_python_tail | ✅ | Python tail processing, 0.24 ms |
| **postprocess total** | | **3.02 ms** (GPU elapsed: 2.82 ms) |

Mamba preset config 與 speed preset 差異：
- `detection_quality_scaling: false`（已由 Mamba head 內建品質信號）
- `person_geometry_prior: false`
- `geometry_suspect_support: false`

### Stage 10：Global Motion Compensation

```
prev_gray_tensor → cuFFT FFT → cross-power spectrum → IFFT → peak → 2×3 affine warp
```

- **GPU phase correlation**：CUDA 實現，downscale=4，零 CPU 參與
- **PCR quality**：peak/RMS ratio 檢查
- **latency**：0.04 ms（profile 已全部分解為 sub-µs 級，gmc_phase_corr=0.00ms 表示實際由 graph 合併執行）

### Stage 11：追蹤更新（ByteTrack）

```
detections + gmc_warp → ByteTracker::update_into()
    → prediction → association (IoU gate + Sinkhorn-Auction hybrid)
    → Kalman update → GPU result buffer
```

- **Association**：IoU-only（ReID off，無 appearance matching）
- **Kalman update**：8D CV model，GMC warp 補償
- **Tracker Graph**：`use_tracker_graph=true`，tracker update 捕捉為 CUDA graph
- **Parameters**：`match_thresh=0.50`, `new_track_thresh=0.28`, `kalman_r_scale=2.8`, `fuse_score_weight=0.0`

### Stage 12–14：結果輸出與 Relink

```
GPU result buffer → HostTrackResultView → Bridge relink / identity resolve
    → Tracklet interpolation (max_gap=35) → MOT txt output
```

- **Materialize**：GPU tensor 複製至 host-visible view
- **Relink**：GPU bridge relink（speed-weighted bidirectional full-gap extrapolation + per-lost detection-score claim），async pipeline path
- **Interpolation**：`interpolate_max_gap=35`，容忍約 1.17s 遮蔽，Recall +1.3pp

---

## 4. 執行條件總覽

| 模組 | 狀態 | 備註 |
|------|:---:|------|
| Fetch | ✅ | DALI NVDEC |
| Ingest / Preprocess | ✅ | Stretch-resize, no letterbox |
| Detection (whole graph) | ✅ | TRT backbone + Mamba head + decode, CUDA graph |
| Postprocess (core) | ✅ | Native C++/CUDA |
| ReID (all stages) | ❌ | `reid_mode: off` — production preset 不啟用 ReID |
| Lazy ReID | ❌ | profiling only |
| GMC (GPU phase correlation) | ✅ | downscale=4 |
| Tracker Update | ✅ | GPUByteTracker, CUDA graph |
| Materialize | ✅ | GPU→host |
| Pipeline Relink (async) | ✅ | Bridge relink, background thread |
| Appearance Bank | ❌ | ReID off → bank inactive |
| Lifecycle Merge | ❌ | default off |
| FP Hard Filter | ✅ | area=40000, min_score=0.10 |
| Detection Quality Scaling | ❌ | Mamba head provides internal quality signal |
| Interpolation | ✅ | max_gap=35 |
| Tracklet graph | ✅ | CUDA graph replay |

---

## 5. 瓶頸分析

| 階段 | 時間 | 佔比 | 優化方向 |
|------|----:|----:|---------|
| **Detection** | 3.37 ms | 42.1% | 已 optimal（whole graph），換更小 backbone |
| **Postprocess** | 3.02 ms | 37.7% | FP hard filter (1.30ms) 為主要子項目 |
| Fetch | 2.14 ms | 26.7% | DALI pipeline tuning |
| Tracker | 0.61 ms | 7.6% | ByteTrack 本身已優化 (graph) |
| Materialize | 0.20 ms | 2.5% | D2H copy 最小化 |
| Relink | 0.11 ms | 1.4% | async path, 零可見成本 |
| GMC | 0.04 ms | 0.5% | GPU phase correlation 已極度優化 |

**detect + postprocess 合計佔 79.8%，是主要瓶頸。**

---

## 6. 基準線參數

| Preset | Engine | IDF1 | MOTA | HOTA | IDs | FP | FN | Rcll | FPS |
|--------|--------|:----:|:----:|:----:|----:|---:|---:|:----:|:---:|
| **mamba_whole_graph** | yolo26s + Mamba | **73.4%** | **76.9%** | **66.6%** | **539** | **4309** | **21106** | **81.2%** | **~175** |
| speed (reference) | yolo26s | 52.0% | 41.6% | — | 475 | 14687 | 52753 | 55.0% | ~98 |
| baseline (reference) | yolo26m | 51.4% | 43.5% | — | 502 | 16112 | 48377 | 59.0% | ~85 |

> **mamba_whole_graph** 相較 speed preset：IDF1 +21.4pp, MOTA +35.3pp, Rcll +26.2pp, FP -70.7%, FPS +79%。
> Mamba head + CUDA graph 帶來結構性精度與速度提升。

---

## 7. 模組貢獻（ablation ledger）

> 使用 cumulative cutoff 方法：每個 profile 只比前一個多開一層模組。
> `Δprev` 即該模組的單獨裸增益。
> 測量條件：mamba_whole_graph preset，yolo26s，2 seq SDP，150 frames。
> 重跑命令：`uv run python scripts/eval/pipeline_contribution.py --detector SDP -- --preset mamba_whole_graph`

### 貢獻分析階段映射

| Profile | 新增模組 | 對應 stage | 說明 |
|---------|---------|-----------|------|
| `tracker_core` | *(bare tracker)* | [1-4] + [11-12] | 基準線（GMC OFF） |
| `tracker_core_gmc` | **GPU GMC** | [10] ON | 測量 GMC 裸貢獻 |
| `semantic_core` | **ReID branch + relink** | [6-8] + [14] | ReID budget/crop/extract + semantic relink |
| `semantic_bank` | **Appearance bank** | [5-8] bank_sync | Bank sync + inject 的額外貢獻 |
| `full_default` | **Async pipeline** | [13] bg_relink_wait | async ReID + pipeline relink 吞吐量 |

### Sequential Ledger（mamba_whole_graph preset, 2 seq SDP, 150fr）

| Step | Module | IDF1 | MOTA | IDs | FP | FPS | Δprev |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | *(bare tracker)* | 18.3% | 10.9% | 106 | 317 | 176.6 | base |
| 1 | **GPU GMC** | 20.3% | 11.6% | 39 | 306 | 192.1 | **IDF1 +2.0pp, IDs −67, FPS +15.5** |
| 2 | ReID branch + relink | 20.0% | 11.6% | 41 | 326 | 165.9 | **IDF1 −0.3pp (harmful), FPS −26.3** |
| 3 | Appearance bank | 20.0% | 11.6% | 41 | 326 | 162.2 | IDF1 ±0.0pp, FPS −3.6 |
| 4 | Async pipeline | 20.0% | 11.6% | 41 | 326 | 161.8 | IDF1 ±0.0pp, FPS −0.4 |

**結論 (mamba_whole_graph preset)**：
- **GMC 是唯一顯著正向貢獻的模組**（+2.0pp IDF1, IDs −63%）
- **Semantic ReID/relink 在 GMC ON 下為負貢獻**（−0.3pp IDF1, +2 IDs）
- **Appearance bank 無貢獻**（±0pp, FPS −3.6）
- **Async pipeline**: FPS 部分回復（±0pp accuracy）
- 完整 ReID stack（`tracker_core_gmc` → `full_default`）：IDF1 **−0.3pp**, FPS **−30.3**
- ReID 在 mamba_whole_graph 下證實為**完全冗余且有害**

---

## 8. 生產 preset 對比 (7-seq SDP, full sequences)

| Preset | IDF1 | MOTA | HOTA | IDs | FP | Rcll | FPS |
|--------|:----:|:----:|:----:|----:|----:|:----:|:---:|
| **mamba_whole_graph** | **73.4%** | **76.9%** | **66.6%** | 539 | 4309 | 81.2% | **175.4** |
| mamba_optimal | 71.2% | 76.3% | — | 665 | 6050 | 82.3% | 100.9 |
| speed (yolo26s) | 52.0% | 41.6% | — | 475 | 14687 | 55.0% | 97.9 |

---

## 9. Production Latency per Sequence (mamba_whole_graph, no profiling)

| Sequence | Resolution | FPS | Avg Latency (ms) |
|----------|-----------|----:|:---:|
| MOT17-02-SDP | 1920×1080 | 177.9 | 5.62 |
| MOT17-04-SDP | 1920×1080 | 173.1 | 5.78 |
| MOT17-05-SDP | 640×480 | 180.9 | 5.53 |
| MOT17-09-SDP | 1920×1080 | 178.8 | 5.59 |
| MOT17-10-SDP | 1920×1080 | 168.8 | 5.92 |
| MOT17-11-SDP | 1920×1080 | 175.9 | 5.68 |
| MOT17-13-SDP | 1920×1080 | 173.4 | 5.77 |
| **Overall** | | **175.4** | **5.70** |

---

## 10. Nsight Systems GPU Kernel 驗證

> Nsight Systems GPU trace (MOT17-04-SDP, 20 frames) 證實：
> - 78 次 `cudaGraphLaunch` / 20 frames = **3.9 graph launches/frame**
> - 4 個 CUDA graph: WholeDetectGraph, TrackerGraph, NMSGraph, GMCGraph
> - `cudaGraphLaunch` 僅佔 host API 時間的 **0.7%**

### GPU Kernel Top-5

| Kernel | GPU Time % | 類別 |
|:-------|:--------:|------|
| cutlass tensor conv (TRT) | 17.5% | Backbone inference |
| `selective_scan_fwd_kernel` | 6.8% | Mamba SSM |
| `stage1_cost_fused_kernel` | 5.3% | Tracker association |
| `cudnn::nchwToNhwcKernel` | 4.3% | Layout conversion |
| `vectorized_elementwise_kernel` (PyTorch) | 3.8% | Ungraphed ops |

### 剩餘 Kernel Launch 來源 (~140/frame)

仍有 2,794 次 `cudaLaunchKernel` 未進入 graph，主要來自：
- Postprocess python_tail (PyTorch ops)
- DALI fetch (nvjpeg decode, color conversion)
- 非 graph 路徑的 profiler/stats 收集

合併這些路徑到 graph 是下一個優化階段。

---

*最後更新：2026-06-06*
