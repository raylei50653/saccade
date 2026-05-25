# Saccade Pipeline Dataflow

> 本文檔提供精確的演算法資料流描述，適合用於報告、簡報或對外展示。
> 對應實作：`src/saccade/perception/eval/evaluator.py`，`time_stage()` 計時的 16 個串行階段。

---

## 1. 整體架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PER-FRAME PIPELINE                           │
│                                                                     │
│  [1] Ingest  → [2] Detect → [3] Postprocess → [4] ReID Branch       │
│       → [5] Lazy ReID → [6] GMC → [7] Tracker → [8] Materialize     │
│       → [9] Relink → [10] Output                                    │
│                                                                     │
│  ────────────────────────────────────────────────────────────────   │
│  每幀固定路徑：[1][2][3][6][7][8]（約 10ms）                          │
│  每幀條件路徑：[4][5][9]（依 CLI 開關決定）                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 串行階段一覽

| # | 階段 | 預設狀態 | 代碼行 | 每幀平均 | 佔比 |
|---|------|:---:|---:|---:|---:|
| 1 | **Fetch** | ✅ | `evaluator.py:1226` | 0.48 ms | 3.5% |
| 2 | **Ingest + Preprocess** | ✅ | `evaluator.py:1236` | 0.94 ms | 6.8% |
| 3 | **Detection** | ✅ | `evaluator.py:1254` | 5.06 ms | 41.8% |
| 4 | **Postprocess** | ✅ | `evaluator.py:1333` | 3.16 ms | 26.1% |
| 5 | **ReID Bank Sync** | ❌ | `evaluator.py:2013` | 0 ms | — |
| 6 | **ReID Budget** | ❌ | `evaluator.py:2041` | 0 ms | — |
| 7 | **ReID Crop** | ❌ | `evaluator.py:2157` | 0 ms | — |
| 8 | **ReID Extract** | ❌ | `evaluator.py:2108` | 0 ms | — |
| 9 | **Lazy ReID** | ❌ | `evaluator.py:2474` | 0 ms | — |
| 10 | **GMC** | ✅ | `evaluator.py:2320` | 0.87 ms | 6.3% |
| 11 | **Tracker Update** | ✅ | `evaluator.py:2342` | 0.78 ms | 5.6% |
| 12 | **Materialize** | ✅ | `evaluator.py:2356` | 0.38 ms | 2.7% |
| 13 | **BG Relink Wait** | ❌ | `evaluator.py:1958` | 0 ms | — |
| 14 | **Relink Write** | 部分 | `evaluator.py:2617` | 0.87 ms | 7.2% |
| 15 | **Frame Total** | ✅ | — | **12.10 ms** | — |

> **測量條件**：`--preset speed`（yolo26s），MOT17-04-SDP 序列，150 frames，warmup 50。
> 所有 `OFF` 模組以 `if cfg.X` 完全 guard，不入程式路徑，零執行成本。

---

## 3. 各階段詳細描述

### Stage 1–2：幀擷取與預處理

```
Raw JPEG frames (DALI) → NVDEC decode → CHW float32 tensor
                                    → AdaptiveFramePool (GPU pin memory)
                                    → letterbox / gamma / contrast 預處理
```

- **Fetch**：DALI streamer 從 `img1/` 讀取 JPEG 並解碼為 GPU tensor
- **Ingest**：幀資料複製至 `AdaptiveFramePool`（固定大小 GPU buffer）
- **Preprocess**：gamma 校正、contrast 調整、letterbox 填充

### Stage 3：目標檢測（佔比 41.8%，最大瓶頸）

```
GPU tensor → TRT YOLO26 推論 → raw boxes(300) / scores / classes
```

- **引擎**：TensorRT 加速的 YOLO26s（`--preset speed`）或 YOLO26m（`--preset baseline`）
- **解析度**：`native_960`（預設），或 tiled 模式（`960p_2x2` / `960p_3x2`）
- **輸出**：每幀最多 300 個 bounding box
- **Pose sidecar**：可選啟用 `--pose-engine`，同時輸出姿勢 keypoints

### Stage 4：後處理（佔比 26.1%）

```
raw detections → quality boost → filter → NMS → cross-tile merge → gating
```

**Native path**（預設）：`PerceptionPipeline::process_detections_into()` 在 C++/CUDA 內完成全部後處理。

**Python fallback**：逐步執行以下 sub-stages（全部計入 `postprocess` 總時間）：

| Sub-stage | 條件 | 動作 |
|-----------|------|------|
| **Quality boost** | `detection_quality_scaling` | 依中心偏移、長寬比、面積動態提升分數 |
| **Filter** | 固定 | 分數閾值、類別過濾、幾何可疑框標記 |
| **NMS** | 固定 | Intersection-over-Union 非極大值抑制 |
| **Cross-tile merge** | `cross_tile_merge` | Tile-based 推論時，合併跨 tile 的重複檢測（seam-aware） |
| **Score penalty** | `cross_tile_score_penalty < 1.0` | 合併後的框降低分數（position uncertainty） |
| **Crowd mode** | `crowd_low_score_mode` | 高密度場景下降低閾值以保留更多檢測 |
| **FP hard filter** | `--fp-hard-filter` | 移除低分 + 大面積的可疑 FP |
| **Detection cap** | `--per-frame-detection-cap > 0` | 限制每幀檢測數量 |
| **Stage2 quality gate** | `--stage2-quality-gate` | 移除 mid-score 區間幾何不良的框 |
| **Consecutive birth gate** | `--consecutive-birth-gate` | 跨幀出現的 sub-threshold 框提升分數 |
| **Birth quality gate** | `--birth-quality-gate` | 高品質 sub-threshold 框提升分數 |
| **Multi-birth manager** | `--multi-birth` | 多信號（score × streak × motion × geometry）出生策略 |

### Stage 5–9：ReID 分支（條件路徑）

> ReID 分支在 GMC 之前執行。所有子階段受同一個 `_do_reid` flag 控制。

```
_do_reid = (reid_work_enabled) AND (has detections) AND (MIN_REID_GAP met)

If need_reid_enabled:
    _do_reid = DynamicReIDController.should_reid()
    # 或 need_reid_frame(prev_track_ids, after_merge_count)
Else:
    _do_reid = (frame_id % reid_interval == 0)
```

| 子階段 | 動作 |
|--------|------|
| **Bank Sync** | `appearance_bank.representatives()` → `tracker.set_reference_features_from_bank()` |
| **Budget** | 根據框數、GMC uncertainty、dynamic history 決定取樣的 detection 數量 |
| **Crop** | ROI crop（native path 內建於 `PerceptionPipeline`，Python fallback 用 `ZeroCopyCropper`） |
| **Extract** | SigLIP2 / 其他 backbone 提取 appearance embedding |
| **Lazy ReID** | Profiling only：對 tentative candidates 計算 self-sim embedding（`--profile-lazy-reid`） |

### Stage 10：全局運動補償

```
prev_gray_tensor → GMC estimator → warp_matrix(6 params) + uncertainty flag
```

- **GPU path**（預設）：CUDA 實現的相位相關（phase correlation）
- **FG mask**：可選在 FFT 前 zero out 偵測區域（`--gmc-fg-mask`），目前未啟用
- **PCR quality**：`0 < pcr < threshold` 標記為 uncertain，影響 ReID budget 的覆蓋範圍
- **計時 sub-stages**：`gray_downscale` → `fg_mask` → `phase_corr` → `handoff`

### Stage 11：追蹤更新（ByteTrack）

```
detections + embeddings + gmc_warp + mid_thresh_scale
    → ByteTracker::update_into()
    → prediction → association → update
    → result → GPU result buffer
```

- **Association**：IoU + appearance matching（若有 embeddings）
- **Kalman update**：狀態預測 + GMC warp 補償
- **Mid-threshold scale**：geometry-aware 的動態中閾值調整

### Stage 12：結果 Materialize

```
GPU result buffer → HostTrackResultView
```

- 將 GPU tensor 複製到 host-visible view
- 包含 keypoints 與 embedding 資訊（如有）

### Stage 13–14：Relink 與身份解析

**Async 路徑**（`--pipeline-relink`）：
```
Frame N:  準備 host_batch + motion snapshots → submit to thread pool
Frame N+1: bg_relink_wait（等 Frame N 的 background thread 完成）
```

**Sync 路徑**（預設）：
```
_prepare_track_candidates()
    → _resolve_frame_tracks()  ← IdentityResolver.resolve_pass()
        ├─ semantic relink (appearance matching)
        ├─ lifecycle merge
        └─ output tracklets
    → _emit_resolved_tracks()
    → post_lifecycle_appearance_bank (可選)
```

### Stage 15：輸出

```
MOT txt / metrics / debug artifacts
```

- 標準 MOT challenge 格式輸出
- Debug：框圖、stage 計時、tile 診斷

---

## 4. 執行條件總覽

| 模組 | 預設 | 覆蓋 CLI | OFF 行為 |
|------|:---:|---------|----------|
| Fetch | ✅ | 無 | 無可選 |
| Ingest / Preprocess | ✅ | 無 | 無可選 |
| Detection | ✅ | `--preset` / `--tiling` | 無可選 |
| Postprocess (core) | ✅ | 無 | 無可選 |
| ReID Bank Sync | ❌ | `--appearance-bank` | 完全跳過 |
| ReID Budget / Crop / Extract | ❌ | `--reid-mode off/semantic/hybrid` | 完全跳過 |
| Lazy ReID | ❌ | `--profile-lazy-reid` | 完全跳過 |
| GMC | ✅ | `--gmc` / `--no-gmc` | 無可選 |
| Tracker Update | ✅ | 無 | 無可選 |
| Materialize | ✅ | 無 | 無可選 |
| BG Relink Wait | ❌ | `--pipeline-relink` | 完全跳過 |
| Semantic Relink | ❌ | `--reid-mode semantic` | 完全跳過 |
| Appearance Bank | ❌ | `--appearance-bank` | 完全跳過 |
| Lifecycle Merge | ❌ | `--lifecycle-merge` | 完全跳過 |
| FP Hard Filter | ❌ | `--fp-hard-filter` | 完全跳過 |
| Detection Cap | ❌ | `--per-frame-detection-cap 0` | 完全跳過 |
| Stage2 Quality Gate | ❌ | `--stage2-quality-gate` | 完全跳過 |
| Birth Gates | ❌ | `--consecutive-birth-gate` / `--birth-quality-gate` | 完全跳過 |
| Multi-birth Manager | ❌ | `--multi-birth` | 完全跳過 |
| Scene Adapt | ❌ | `--scene-adapt` | 完全跳過 |

> **關鍵原則**：所有 OFF 模組以 `if cfg.X` 完全 guard，不入程式路徑，**零執行成本**。

---

## 5. 瓶頸分析

| 階段 | 時間 | 佔比 | 優化方向 |
|------|----:|----:|---------|
| **Detection** | 5.06 ms | 41.8% | 換引擎（s→n）、TRT optimization |
| **Postprocess** | 3.16 ms | 26.1% | NMS 降複雜度、減少 raw_boxes 數量 |
| Ingest | 0.94 ms | 6.8% | DALI pipeline tuning |
| GMC | 0.87 ms | 6.3% | 基本固定，低延遲 |
| Tracker | 0.78 ms | 5.6% | ByteTrack 本身已優化 |
| Relink | 0.87 ms | 7.2% | Async path 已疊加 |
| Materialize | 0.38 ms | 2.7% | D2H copy 最小化 |

**detect + postprocess 合計佔 67.9%，是唯一可觀的優化空間。**

---

## 6. 基準線參數

| Preset | Engine | match | ntt | IDF1 | MOTA | IDs | FP | FN | Rcll | FPS |
|--------|--------|:---:|:---:|----:|----:|----:|---:|---:|----:|---:|
| **speed** | yolo26s | 0.66 | 0.28 | **51.2%** | 40.8% | **541** | 13139 | 52753 | **53.0%** | **~110** |
| **baseline** | yolo26m | 0.66 | 0.28 | 50.3% | **42.0%** | 589 | 16112 | **48377** | **56.9%** | ~100 |

- **speed**：優先 IDF1、IDs、FP、FPS（identity tracking 精度導向）
- **baseline**：優先 MOTA、Rcll、FN（漏偵最小化導向）

---

## 7. 模組貢獻（ablation ledger）

> 使用 cumulative cutoff 方法：每個 profile 只比前一個多開一層模組。
> `Δprev` 即該模組的單獨裸增益。
> 重跑命令：`uv run python scripts/eval/pipeline_contribution.py --detector SDP`

### 貢獻分析階段映射

| Profile | 新增模組 | 對應 stage | 說明 |
|---------|---------|-----------|------|
| `tracker_core` | *(bare tracker)* | [1-4] + [10] + [11-12] | 基準線（GMC OFF） |
| `tracker_core_gmc` | **GPU GMC** | [10] ON | 測量 GMC 裸貢獻 |
| `semantic_core` | **ReID branch + relink** | [6-8] + [14] | ReID budget/crop/extract + semantic relink |
| `semantic_bank` | **Appearance bank** | [5-8] bank_sync | Bank sync + inject 的額外貢獻 |
| `full_default` | **Async pipeline** | [13] bg_relink_wait | async ReID + pipeline relink 吞吐量 |

> ⚠️ `reid_bank_sync [5]` 只在 appearance_bank ON 時執行，所以其貢獻
>    與 bank inject 合併在 `semantic_bank` 步驟中測量。

### Sequential Ledger（P3 baseline, m=0.66, ntt=0.28, yolo26s，2 seq）

| Step | Module | Profile | IDF1 | MOTA | IDs | FP | FN | FPS | Δprev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | *(bare tracker)* | `tracker_core` | `51.6` | `43.1` | `279` | `6726` | `27348` | `106.14` | base |
| 1 | **GPU GMC** | `tracker_core_gmc` | `54.4` | `44.7` | `146` | `6272` | `26998` | `99.29` | **IDF1 +2.8pp, IDs −133, FPS −6.9** |
| 2 | ReID branch + relink | `semantic_core` | `54.8` | `44.7` | `146` | `6448` | `26826` | `97.05` | IDF1 +0.4pp, IDs ±0, FPS −2.2 |
| 3 | Appearance bank | `semantic_bank` | `54.8` | `44.4` | `145` | `6434` | `26989` | `79.71` | IDF1 ±0.0pp, IDs −1, **FPS −17.3** |
| 4 | Async pipeline | `full_default` | `54.8` | `44.6` | `145` | `6345` | `26991` | `76.08` | IDF1 ±0.0pp, IDs ±0, FPS −3.6 |

**結論**：
- **GMC 是 pipeline 中唯一顯著貢獻的模組**（+2.8pp IDF1，IDs 降 133）
- **ReID branch + relink**：+0.4pp IDF1、FPS −2.2。邊際正益，cost 低。
- **Appearance bank**：IDF1 ±0、FPS **−17.3**。零增益高代價，不應設為 default ON。
- **Async pipeline**：FPS 部分回復（+3.6 FPS），IDF1 不變。
- 完整 ReID stack（`tracker_core_gmc` → `full_default`）：IDF1 +0.4pp，FPS **−23.2**。代價遠高於收益。
- 在 GMC ON 下，semantic relink 被診斷為基本冗余（reject 86.8% 因 age）

---

*最後更新：2026-05-13*
