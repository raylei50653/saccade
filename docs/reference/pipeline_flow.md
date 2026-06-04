# Saccade Pipeline Flow

本文件描述 **目前實作主路徑** 的資料流，重點是對應現有 `runner.py` 與 evaluation path，而不是較早期的完整產品願景圖。

- 穩定架構邊界：看 [README.md](../architecture/README.md)
- 開發入口與 source-of-truth：看 [DEVELOPMENT.md](../../DEVELOPMENT.md)
- 事件 / API / storage schema：看 [api_spec.md](../modules/storage/api_spec.md)

---

## 1. 主路徑範圍

本文件主要對應：

- [scripts/eval/mot17.py](/scripts/eval/mot17.py)
- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py)（實作主體）
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py)（entry point wrapper）

這條路徑目前是 repo 最活躍的 perception / tracking / relink 主流程。

---

## 2. 高層流程圖

> 以下對應 `evaluator.py` 中 `time_stage()` 的實際串行順序（共 16 個 stage）。
> ⚠️ 注意：ReID 分支（bank_sync → budget → crop → extract）**完全在 GMC 之前**。

```text
[1] fetch                      -> DALI / NVDEC read
[2] ingest_preprocess          -> AdaptiveFramePool + preprocess
[3] detect                     -> yolo26s/m native_960 or tiled
[4] postprocess                -> filter -> NMS -> cross-tile merge
                                 -> quality scaling + gating (6 sub-stages)
[5] reid_bank_sync             -> appearance bank → tracker [OFF]
[6] reid_budget                -> budget selection [OFF]
[7] reid_crop                  -> ROI crop (Python only) [OFF]
[8] reid_extract               -> siglip2 / other backbones [OFF]
[9] lazy_reid                  -> self-sim profiling [OFF, profiling only]
[10] gmc                       -> GPU/PMC warp estimation [ON]
[11] track                     -> association + Kalman update [ON]
[12] materialize               -> GPU -> host view [ON]
[13] bg_relink_wait            -> wait bg relink thread [OFF]
[14] relink_write              -> semantic relink + identity resolve [部分 ON]
[15] frame_total               -> 本幀總時間
```

### 階段對應表

| 階段編號 | 名稱 | 是否每幀執行 | 備註 |
|:---:|:---|:---:|:---|
| 1-4 | 偵測前處理 | ✅ | 固定主路徑 |
| 5-8 | ReID 分支 | ❌ | 僅 `reid_work_enabled` 時執行 |
| 9 | Lazy ReID | ❌ | profiling only，不影響主路徑 |
| 10 | GMC | ✅ | baseline 核心 |
| 11-12 | 追蹤 | ✅ | 固定主路徑 |
| 13-14 | Relink | 部分 | `pipeline_relink` 時走 bg 路徑，否則同步 |
| 15 | 計時 | ✅ | 僅 profiling 模式 |

### Postprocess 內部 Sub-stages

`postprocess` 階段內含 **6 個隱藏 sub-stages**（均計入 `postprocess` 總時間）：

| Sub-stage | 條件 | 說明 |
|:---|:---|:---|
| `post_filter` | 僅 Python path | confidence / class / geometry filter |
| `post_nms` | 僅 Python path, tiled | NMS (native path 已內建於 PerceptionPipeline) |
| `post_merge` | `cross_tile_merge=True` | cross-tile duplicate merge |
| `fp_hard_filter` | `--fp-hard-filter` | 移除可疑低分大面積偵測 |
| `detection_cap` | `--per-frame-detection-cap > 0` | 限制每幀檢測數 |
| `stage2_quality_gate` | `--stage2-quality-gate` | 移除 mid-score 區間幾何不良偵測 |
| `consecutive_birth_gate` | `--consecutive-birth-gate` | 跨幀出生門控 |
| `birth_quality_gate` | `--birth-quality-gate` | 出生品質門控 |
| `multi_birth` | `--multi-birth` | 多信號出生策略 |

---

## 3. 逐階段流程

### 3.1 Frame Ingest / Preprocess

責任：

- 從 frame source 取出影格
- 做基礎 preprocess
- 準備給 detector 的 tensor

主要位置：

- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:1226)

說明：

- 在 evaluation path 中，frame 先進 `AdaptiveFramePool`
- preprocess 層目前是主流程的一部分，但不是目前主要演算法焦點

### 3.2 Detection

責任：

- 跑 detector
- 產出 raw `boxes / scores / classes`

主要位置：

- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:1254)

說明：

- detector output 之後不直接進 tracker
- 會先經過 postprocess 與 geometry / tile merge 清理
- evaluation path 目前可走：
  - `native_960`：單張 `960x960` 推論
  - `960p_2x2` / `960p_3x2`：tile-based detection + cross-tile duplicate merge

### 3.3 Detection Postprocess

責任：

- confidence / class / geometry filter
- suspect box 標記
- NMS
- cross-tile duplicate merge

主要位置：

- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:1333)
- [src/saccade/perception/eval/detection.py](/src/saccade/perception/eval/detection.py)
- [src/tracking/pipeline.cpp](/src/tracking/pipeline.cpp)

說明：

- 若 native `PerceptionPipeline` 可用，主流程優先走 `process_detections_into()`
- 否則走 Python wrapper / fallback path
- tiled 路徑的 merge 現在是 seam-aware：
  - 對 seam-near pair 放寬 duplicate 判定
  - 對 seam boxes 降低座標融合權重
  - 輸出框使用「偏向非 seam 候選」的融合框，而非單純硬選 best detection
- `runner.py` 可額外開 `--tile-diagnostics`，追蹤 seam 汙染是否真的被 merge 掉
- 目前 `cross-tile merge` 不再被視為「穩定增益來源」；它是 tiled path 的必要補救，但在高密場景仍是主要風險點之一

### 3.4 ReID Trigger Decision

責任：

- 決定這一幀是否值得做 embedding extraction

主要位置：

- [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py:76)
- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:2045)

目前流程：

- 若 `reid_work_enabled` 且有 detections，才進 trigger decision
- 目前路徑可能使用：
  - fixed interval
  - `need_reid_frame()`
  - `DynamicReIDController.should_reid()`
- 並額外受 `MIN_REID_GAP` 限制

說明：

- 這裡是當前主要演算法優化熱點之一
- 下一步方向是 track-level / budgeted ReID，而不是單純再加更多 frame-level heuristic

### 3.5 ReID Crop / Extract

責任：

- 依 box 裁切 ROI
- 提取 appearance embedding

主要位置：

- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:2093)

目前流程：

- 若 native pipeline 可用，優先走 `perception_pipeline.extract_reid()`
- 否則走 Python cropper + extractor

說明：

- ReID extraction 不是每幀必做
- 它是受 trigger 控制的昂貴步驟

### 3.6 GMC（獨立階段）

責任：

- 計算幀間運動補償 warp
- FG mask（optional）
- PCR quality feedback

主要位置：

- [src/tracking/tracker_gpu.cu](/src/tracking/tracker_gpu.cu)
- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:2250)

目前流程：

- 若 `gmc_estimator` 存在，先計算 `gmc_warp` + `gmc_uncertain`
- 可選 FG mask：在 FFT 前 zero out 偵測區域
- PCR quality：`0 < pcr < threshold` 標記為 uncertain（影響 ReID budget）

說明：

- GMC 是推薦 baseline 核心模組（`--gmc`）
- 在 ReID 分支完成後、tracker update 前執行
- `gmc_fg_mask` 目前未啟用（恆為 0ms）

### 3.7 Lazy ReID（Profiling Only）

責任：

- 對 tentative candidates 做 self-sim embedding profiling
- 不影響主路徑，僅收集統計數據

主要位置：

- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:2427)

目前流程：

- 若 `--profile-lazy-reid`，從 tracker 取出 tentative candidates
- 計算 past embedding vs current embedding 的 cosine sim
- 收集 pairs/pass/sim_sum 等統計

說明：

- 這是 profiling 階段，不影響追蹤結果
- 計入 `lazy_reid` stage 時間

### 3.8 Tracker Update

責任：

- tracker state predict / update
- association
- optional appearance-aware matching（via embeddings）
- GMC warp consumption

主要位置：

- [src/tracking/tracker_gpu.cu](/src/tracking/tracker_gpu.cu)
- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:2342)

目前流程：

- 先準備 `mid_thresh_scale`（geometry-aware）
- 呼叫 `detector.tracker.update_into()`
  - 傳入：fused_boxes/scores/classes, embeddings, gmc_warp, mid_thresh_scale
  - result 留在 GPU buffer（`tracker_result_buffers`）

說明：

- 現在的 tracker 熱路徑已大幅 native 化
- result 優先留在 GPU result buffer，再在必要邊界 materialize
- GMC warp 在此階段被 tracker 消耗

### 3.9 Semantic Relink / Identity Resolve

責任：

- 將 local track ids 解決成較穩定的 identity output
- 視需要使用 appearance / motion / lifecycle merge

主要位置：

- [src/saccade/perception/eval/relink.py](/src/saccade/perception/eval/relink.py)
- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:2502)

目前流程：

- 若 `pipeline_relink`（async 路徑）：
  - 準備 host_batch + motion snapshots，提交至背景線程
  - 主線程在下一幀開頭 `bg_relink_wait`
- 否則（sync 路徑）：
  - 同步呼叫 `_prepare_track_candidates()` → `_resolve_frame_tracks()` → `_emit_resolved_tracks()`
  - `IdentityResolver.resolve_pass()` 執行 semantic relink + lifecycle merge

說明：

- 這一層目前的核心議題是：
  - reference quality
  - false accept filtering
  - unified association / relink scoring
- 在 GMC ON 下，semantic relink 被診斷為基本冗余

### 3.10 Post-Merge Cleanup

責任：

- 對 output tracklets 做 optional offline stitching
- 視需要過濾低品質 tracklet

主要位置：

- [src/saccade/perception/eval/evaluator.py](/src/saccade/perception/eval/evaluator.py:2597)

說明：

- `post_merge_output_tracklets()` 屬於 optional cleanup，不是 primary online decision path
- `filter_low_quality_tracklets()` 用於移除短命 / 低分 output IDs

---

## 4. 外圍事件與慢路徑

主 evaluation path 之外，repo 仍保留 event / storage / cognition 子系統。

### 4.1 Event Queue / Stream

主要位置：

- [src/saccade/storage/redis_cache.py](/src/saccade/storage/redis_cache.py)
- [src/saccade/perception/entropy.py](/src/saccade/perception/entropy.py)

說明：

- 目前同時存在：
  - Redis List `saccade:events`
  - Redis Stream `saccade:stream`
- schema 以 [api_spec.md](../modules/storage/api_spec.md) 為準

### 4.2 Cognition / Memory

主要位置：

- [src/saccade/cognition/orchestrator.py](/src/saccade/cognition/orchestrator.py)
- [src/saccade/storage/chroma_store.py](/src/saccade/storage/chroma_store.py)

說明：

- orchestrator 讀取 Redis stream batch
- 轉成 scene description 與 metadata
- 寫入 Chroma memory
- 必要時觸發 RAG query

### 4.3 Health

主要位置：

- [src/saccade/pipeline/health.py](/src/saccade/pipeline/health.py)

說明：

- health 屬於 operational path
- 它不是 perception 熱路徑的一部分

---

## 5. 目前主瓶頸與熱點

目前主要演算法空間集中在：

- ReID trigger quality
- tiled detector 的 seam duplicate handling
- `native_960` 與 tiled 路徑的召回 / FP 統計差異
- association / relink unified scoring
- reference quality gate
- GMC quality-aware handling
- post-merge V2 cost

這些方向的近期排序以 [TODO.md](../TODO.md) 為準。

---

## 6. 不屬於本文件的內容

本文件不負責：

- 穩定架構責任邊界的完整定義
  - 看 [README.md](../architecture/README.md)
- 事件 / API schema 細節
  - 看 [api_spec.md](../modules/storage/api_spec.md)
- 實驗結果與 backlog 排序
  - 看 [TODO.md](../TODO.md)

最後更新：2026-05-13
