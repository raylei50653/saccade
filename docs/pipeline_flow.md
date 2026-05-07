# Saccade Pipeline Flow

本文件描述 **目前實作主路徑** 的資料流，重點是對應現有 `runner.py` 與 evaluation path，而不是較早期的完整產品願景圖。

- 穩定架構邊界：看 [architecture.md](/docs/architecture.md:1)
- 開發入口與 source-of-truth：看 [../DEVELOPMENT.md](/DEVELOPMENT.md:1)
- 事件 / API / storage schema：看 [api_spec.md](/docs/api_spec.md:1)

---

## 1. 主路徑範圍

本文件主要對應：

- [scripts/eval/mot17.py](/scripts/eval/mot17.py:1)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1)

這條路徑目前是 repo 最活躍的 perception / tracking / relink 主流程。

---

## 2. 高層流程圖

```text
Frame Source
  -> ingest / preprocess
  -> detection
  -> filter / NMS / cross-tile merge
  -> optional ReID trigger decision
  -> optional crop / embedding extract
  -> tracker update
  -> semantic relink / identity resolve
  -> optional post-merge cleanup
  -> optional low-quality tracklet filter
  -> eval output
```

---

## 3. 逐階段流程

### 3.1 Frame Ingest / Preprocess

責任：

- 從 frame source 取出影格
- 做基礎 preprocess
- 準備給 detector 的 tensor

主要位置：

- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1849)

說明：

- 在 evaluation path 中，frame 先進 `AdaptiveFramePool`
- preprocess 層目前是主流程的一部分，但不是目前主要演算法焦點

### 3.2 Detection

責任：

- 跑 detector
- 產出 raw `boxes / scores / classes`

主要位置：

- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1884)

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

- [src/saccade/perception/eval/detection.py](/src/saccade/perception/eval/detection.py:1)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1910)
- [src/tracking/pipeline.cpp](/src/tracking/pipeline.cpp:1)

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
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:2068)

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

- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:2121)

目前流程：

- 若 native pipeline 可用，優先走 `perception_pipeline.extract_reid()`
- 否則走 Python cropper + extractor

說明：

- ReID extraction 不是每幀必做
- 它是受 trigger 控制的昂貴步驟

### 3.6 Tracker Update

責任：

- tracker state predict / update
- association
- optional appearance-aware matching
- GMC warp consumption

主要位置：

- [src/tracking/tracker_gpu.cu](/src/tracking/tracker_gpu.cu:1)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:2229)

目前流程：

- runner 先準備 `mid_thresh_scale`
- 如有 GMC estimator，先算 `gmc_warp`
- 再呼叫 tracker update path

說明：

- 現在的 tracker 熱路徑已大幅 native 化
- result 優先留在 GPU result buffer，再在必要邊界 materialize

### 3.7 Semantic Relink / Identity Resolve

責任：

- 將 local track ids 解決成較穩定的 identity output
- 視需要使用 appearance / motion / lifecycle merge

主要位置：

- [src/saccade/perception/eval/relink.py](/src/saccade/perception/eval/relink.py:1)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:692)

目前流程：

- semantic relink 與 lifecycle merge 已收斂成 `IdentityResolver.resolve_pass()`
- 若可用，優先走 C++ path

說明：

- 這一層目前的核心議題是：
  - reference quality
  - false accept filtering
  - unified association / relink scoring

### 3.8 Post-Merge Cleanup

責任：

- 對 output tracklets 做 optional offline stitching
- 視需要過濾低品質 tracklet

主要位置：

- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1144)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1295)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:2453)

說明：

- `post_merge_output_tracklets()` 屬於 optional cleanup，不是 primary online decision path
- `filter_low_quality_tracklets()` 用於移除短命 / 低分 output IDs

---

## 4. 外圍事件與慢路徑

主 evaluation path 之外，repo 仍保留 event / storage / cognition 子系統。

### 4.1 Event Queue / Stream

主要位置：

- [src/saccade/storage/redis_cache.py](/src/saccade/storage/redis_cache.py:1)
- [src/saccade/perception/entropy.py](/src/saccade/perception/entropy.py:1)

說明：

- 目前同時存在：
  - Redis List `saccade:events`
  - Redis Stream `saccade:stream`
- schema 以 [api_spec.md](/docs/api_spec.md:1) 為準

### 4.2 Cognition / Memory

主要位置：

- [src/saccade/cognition/orchestrator.py](/src/saccade/cognition/orchestrator.py:1)
- [src/saccade/storage/chroma_store.py](/src/saccade/storage/chroma_store.py:1)

說明：

- orchestrator 讀取 Redis stream batch
- 轉成 scene description 與 metadata
- 寫入 Chroma memory
- 必要時觸發 RAG query

### 4.3 Health

主要位置：

- [src/saccade/pipeline/health.py](/src/saccade/pipeline/health.py:1)

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

這些方向的近期排序以 [docs/TODO.md](/docs/TODO.md:1) 為準。

---

## 6. 不屬於本文件的內容

本文件不負責：

- 穩定架構責任邊界的完整定義
  - 看 [architecture.md](/docs/architecture.md:1)
- 事件 / API schema 細節
  - 看 [api_spec.md](/docs/api_spec.md:1)
- 實驗結果與 backlog 排序
  - 看 [docs/TODO.md](/docs/TODO.md:1)

最後更新：2026-04-30
