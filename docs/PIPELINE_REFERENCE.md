# Saccade Pipeline 核心技術參考文檔 (Implementation-Aligned)

本文件整合了 Saccade Pipeline 的全流程資料流 (Dataflow)、算法實作、關鍵參數以及程式碼對照，內容已與 `src/saccade/perception/eval/runner.py` 及相關模塊的實作完全對齊。

---

## 1. 全流程模組化資料流 (Full Modular Dataflow)

```text
[ Ingest Module ] (DALI NVDEC) [C++] (Separate Process/Thread)
   | <Raw Image / Video Stream>
   | 1. NVDEC Hardware Decoding -> NHWC (uint8)
   | 2. Tensor Permutation -> CHW
   | 3. FP32 Normalization (/ 255.0)
   v
   +=== [Async Queue: Raw Frames] ===+
   |
   v
[ Preprocess Module ] (apply_frame_preprocess) [C++ / CUDA]
   | <pool.frame_buffer (CHW, Float32, 0~1)>
   | 1. Letterbox / Resize (Adaptive)
   | 2. Gamma Correction (Luma Thresholded)
   | 3. Contrast Adjustment
   v
[ Detection Module ] (YOLO TRT: Native 960 or Tiled) [C++ / TensorRT]
   | <Preprocessed Frame>
   | 1. Route to native_960 or adaptive tiling (2x2 / 3x2)
   | 2. TensorRT YOLO Inference
   | 3. Bounding Box Decoding / Tile Coordinate Alignment
   v
   +=== [Async Boundary: Inference -> Post-proc] ===+
   |
   v
[ Post-proc Module ] (Native / Python) [Mixed]
   | <raw_boxes, raw_scores, raw_classes>
   | 1. Geometry Prior Filtering (Aspect/Area)
   | 2. Class-aware NMS
   | 3. Cross-tile Duplicate Merging
   | 4. Detection Quality Scaling
   v
[ GPU Tensor ] fused_boxes, fused_scores (幾何品質加權)
   |
   +-----------------------+--------------------------+-----------------------+
   |                       |                          |                       |
[ ReID Trigger ]        [ Motion: GMC ]            [ Internal States ]        |
[Python]                [C++ / CUDA]               [C++ / CUDA]               |
   | (Decision)            | (Homography)             | (Kalman Filter)       |
   | 1. Eval Thresholds    | 1. Foreground Masking    | 1. State Predict      |
   | 2. Budget Alloc.      | 2. cuFFT Phase Corr.     | 2. Covariance Update  |
   v                       v                          v                       |
[ should_reid? ]        [ gmc_warp ]               [ d_states (8xN) ]         |
   |                       |                          |                       |
   +----------+------------+                          |                       |
              |                                       |                       |
[ ReID Extract & 1D-FFT Top-K (LaSt-ViT) ]            |                       |
[C++ / CUDA]                                          |                       |
   | <ROI Cropped Tensors>                            |                       |
   | 1. cuFFT R2C (Channel-wise)                      |                       |
   | 2. Gaussian Low-Pass Filter                      |                       |
   | 3. cuFFT C2R                                     |                       |
   | 4. Stability Scoring                             |                       |
   | 5. Top-K Pooling & L2 Normalization              |                       |
   v                                                  |                       |
[ Embeddings ] (N x D) -------------------------------+                       |
                                                      |                       |
                                                      v                       |
                                       [ GPU Association Module ]             |
                                       [C++ / CUDA]                           |
                                       | (Sinkhorn-Auction Hybrid)            |
                                       | 1. Cost Matrix (IoU + ReID + Maha)   |
                                       | 2. Sinkhorn Cost Sparsification      |
                                       | 3. Auction Parallel Assignment       |
                                       | 4. Kalman State Update               |
                                       v                                      |
                               [ GPU Buffer ] tracker_result_buffers          |
                                                      |                       |
   +--------------------------------------------------+                       |
   |                                                                          |
   v                                                                          |
[ Materialize Module ] (_materialize_gpu_track_results) [C++ -> Python]       |
   | <D2H Sync: GPU -> CPU>                                                   |
   | 1. Stream Synchronization                                                |
   | 2. Zero-Copy Tensor Transfer                                             |
   v                                                                          |
[ Identity Resolve Module ] (IdentityResolver) [Python]                       |
   | 1. Semantic Relinking (Cosine Sim)                                       |
   | 2. Lifecycle Merging (Missing Links)                                     |
   v                                                                          |
[ Global Identity Map ] (GlobalTrackIdMapper) [Python]                        |
   |                                                                          |
   v                                                                          |
   +=== [Async Output Queue] ===+                                             |
   |                                                                          |
   v                                                                          |
[ Output & Feedback Module ] (_finalize_frame_side_effects) [Python/IO]       |
   | 1. Top-K Appearance Bank Update                                          |
   | 2. Post-Merge Offline Stitching                                          |
   | 3. Metrics / CSV Emit                                                    |
   v                                                                          |
[ Final Results ] (mot_results.txt / Redis Stream) (Separate Thread/Process)  |
```

### 1.1 Dataflow 與觀測能力對照

上面的 dataflow 圖描述的是責任分層，不等於每個 module 都已經有獨立的
runtime metric。就目前 evaluation path 而言，觀測能力分三類：

- **Directly measured**: 有對應 stage metric，可直接看 latency / jitter / count delta。
- **Indirectly measured**: 沒有獨立 stage，但可以透過上層 stage 或下游品質指標側推。
- **Not separately measured yet**: 目前沒有穩定獨立指標，只能視為後續 instrumentation gap。

目前對照：

| Dataflow Module | Current visibility | Current metric mapping |
| :--- | :--- | :--- |
| Ingest Module | Directly measured | `fetch` |
| Preprocess Module | Directly measured | `ingest_preprocess` |
| Detection Module | Directly measured | `detect` |
| Post-proc Module | Directly measured | `postprocess`, `post_filter`, `post_nms`, `post_merge`, `raw_boxes`, `after_filter`, `after_nms`, `after_merge` |
| ReID Trigger | Directly measured | `reid_budget`, `lazy_reid` |
| Motion: GMC | Directly measured | `gmc`, plus downstream `IDs / FN / MOTA` |
| Internal States | Indirectly measured | mostly folded into `track` |
| ReID Extract | Directly measured | `reid_crop`, `reid_extract`, `native_reid_*` |
| GPU Association Module | Directly measured | `track` |
| Materialize Module | Directly measured | `materialize` |
| Identity Resolve Module | Indirectly measured | mostly folded into `relink_write` plus `IDs / IDF1` |
| Global Identity Map | Not separately measured yet | no stable standalone metric |
| Output & Feedback Module | Indirectly measured | mostly folded into `relink_write` / `frame_total` |

---

## 2. 實作細節對照 (Implementation Mapping)

### (1) Ingest & Preprocess
*   **影像讀取：** `DALIStreamerStream` 輸出 `NHWC (uint8)`。
*   **轉換：** `frame_gpu.permute(2, 0, 1).float() / 255.0`。
*   **預處理：** `apply_frame_preprocess(buffer, modes, gamma, gamma_luma_threshold, contrast)`。

### (2) Detection & Post-process
*   **推理：** `detect_fn(detector, pool, h_orig, w_orig, preprocess_modes)`。
*   **Native 路徑：** `perception_pipeline.process_detections_into(...)` 整合了過濾、NMS 與幾何檢查。
*   **Python 路徑：**
    *   `detect_native_960`: 單張 `960x960` 推理控制組。
    *   `detect_adaptive_960_tiled`: `960p_2x2` / `960p_3x2` tiled detector。
    *   `filter_detections_fast`: 幾何先驗過濾 (`person_geometry_prior`)。
    *   `nms_fast`: 類別感知 NMS。
    *   `merge_cross_tile_duplicates_fast`: 處理重疊切片重複目標。
    *   `_compute_detection_quality_batch`: 根據 Aspect, Center, Area 計算品質分數。
*   **Diagnostics：** `runner.py` 可透過 `--tile-diagnostics` 輸出 `pre_merge_seam`、`post_merge_seam`、`merged_clusters`、`compression`，用來判讀 tiled seam 汙染是否被有效消除。

### (3) ReID 策略與特徵萃取 (Orchestration & LaSt-ViT)
*   **觸發機制：** `dynamic_reid.should_reid(after_merge_count)` 或 `need_reid_frame`。
*   **預算分配：** `_budget_reid_candidates` 根據優先級選取前 K 個目標提取特徵。
*   **提取與純化 (1D-FFT Top-K)：** `perception_pipeline.extract_reid` (Native) 或 `extractor.extract` (Python)。
    *   實作於 `src/perception/preprocessor_gpu.cu` (如 `launch_last_vit_refinement`)。
    *   利用 `cuFFT` 進行通道維度一維變換 (R2C -> 高斯濾波 -> C2R)。
    *   計算穩定性評分後，根據 `top_k_ratio` 執行 Top-K Pooling，過濾背景並產生高純度前景 Embedding。

### (4) GPU 追蹤關聯 (Tracker Core)
*   **主入口：** `detector.tracker.update_into(...)`。
*   **參數對應：**
    *   `fused_boxes`: 過濾後的檢測框。
    *   `embeddings`: 選中的 ReID 特徵（其餘為 0）。
    *   `gmc`: 運動補償單應性矩陣。
    *   `mid_thresh_scale`: 幾何自適應動態門檻。
*   **算法：** 在 `tracker_gpu.cu` 實作 Sinkhorn 先驗與 Auction 平行匹配。

### (5) 身份解析與後處理 (Relink & Cleanup)
*   **全域解析：** `IdentityResolver` 調度 `relinker.resolve_pass` 與 `lifecycle_merger`。
*   **外觀庫同步：** `_finalize_frame_side_effects` 將高品質 Embedding 存入 `primary_appearance_bank` 並反饋給 `relinker`。
*   **最後拼接：** `post_merge_output_tracklets` (Offline Stitching) 與 `filter_low_quality_tracklets`。

---

## 3. 重要參數對應 (Key Parameters)

| 參數名稱 | 來源 | 預設/典型值 | 作用 |
| :--- | :--- | :--- | :--- |
| `track_thresh` | `runner.py` | 0.05 | ByteTrack 低分框門檻。 |
| `match_thresh` | `runner.py` | 0.80 | 第一階段關聯 IoU 門檻。 |
| `reid_weight` | `tracker_gpu.py` | 0.80 | 外觀與運動代價融合權重。 |
| `semantic_threshold`| `relink.py` | 0.91 | Semantic Relink 相似度門檻。 |
| `reid_budget` | `runner.py` | 0.0 (Auto) | 單幀特徵提取上限。 |

---

## 4. 資源管理決策

*   **Zero-Copy 緩衝：** 使用 `allocate_result_buffers` 預分配 GPU 空間。
*   **GMC 遮罩：** `set_fg_mask_boxes` 在計算背景運動時排除當前檢測到的目標，避免前景干擾。
*   **自適應門檻：** `geometry_mid_thresh_scale` 根據行人平均高度動態調整檢測靈敏度。
*   **Tracker LRU Eviction（Production Path）：** `AsyncDispatcher` 以 `OrderedDict` 維護最多 `max_streams`（default 8）個活躍 `GPUByteTracker`；超限時驅逐最舊 tracker 並觸發 `~GPUByteTracker → cudaFree`，防止串流無限累積 GPU buffer。
*   **跨進程 VRAM 廣播（Production Path）：** Dispatcher 進程透過 POSIX named shared memory（`saccade_vram_level`，1 byte）廣播 `DegradationLevel`；Orchestrator 進程讀取後，在 FAST_PATH (>92%) 時停止 RAG embedding 呼叫，在 EMERGENCY (>96%) 時丟棄非異常 frame。

最後更新：2026-05-05

---

## 5. 960p_2x2 Tiled Detection Dataflow

`960p_2x2` Tiled Detection 機制旨在於高解析度場景下維持特徵粒度，並透過重疊採樣與接縫感知融合（Seam-Aware Fusion）緩解邊緣截斷（Truncation）造成的座標扭曲。

目前狀態補充：

- 這條路徑已實作 seam-aware cross-tile duplicate merge 與代表框融合。
- 但在 `MOT17-04-SDP / MOT17-10-SDP` 的最新控制實驗中，`native_960` 仍明顯優於 `960p_2x2 tiled`，尤其在 `FN / Recall / MOTA`。
- 因此這一節描述的是「tiled path 現行實作與補救策略」，不是代表 tiled 已經追平 `native_960`。

### Phase 1: Tensor Preparation & Tiling (張量預備與切片)
負責建構 Batched 推理張量：
1. **Adaptive Scaling**: 計算縮放係數 `r = 960.0 / max(h_orig, w_orig)`，將 Frame 映射至最長邊為 `960` 的空間。
2. **Canvas Projection**: 將縮放後的 Tensor 投影至均值填充 (`114/255`) 的 `960x960` 背景畫布 (`pool.canvas_960p`) 中心，產生偏移量 `x_off`, `y_off`。
3. **Overlapped Patching (50% Stride)**: 採 Stride `320` 於畫布提取 4 塊 `640x640` 張量，建構 `[4, C, H, W]` 的推理 Batch (`pool.tiles_batch4`)。
   - 空間映射：`[0:640, 0:640]`, `[0:640, 320:960]`, `[320:960, 0:640]`, `[320:960, 320:960]`

*(註：若原始分辨率 `max(h, w) <= 960`，動態路由至 `detect_single_patch_640` 單一張量推理路徑以降低 Overhead)*

### Phase 2: Forward Pass (模型推理)
- 透過 TensorRT engine 執行 Batched Inference (`detector.detect_raw`)，輸出具獨立空間座標系的 Tensor `[4, N, 6]` (Boxes, Scores, Classes)。

### Phase 3: Spatial Alignment (空間座標對齊)
將局部 Patch 座標映射回原圖空間：
1. **Patch Offset Translation**: 將框座標加回其在 `960x960` 畫布上的相對偏移 (`pool.tile_dx`, `pool.tile_dy`)。
2. **Canvas Inverse Projection**: 扣除中心投影的偏移 (`x_off`, `y_off`) 並除以縮放係數 `r`。
3. **Tensor Flattening**: 對齊維度，重塑為 `[N_total, 4]` 之聚合張量。

### Phase 4: Seam-Aware Cross-Tile Fusion (接縫感知去重融合)
調用硬體加速之 `merge_cross_tile_duplicates_fast`，處理因 50% 覆蓋率產生的跨 Tile 冗餘檢測。
1. **Seam Detection**: 透過 `seam_margin_canvas_px` (預設 ~24px) 劃定切片邊界干擾區。落於此區的檢測框被標記為 `cluster_seam`。
2. **Topology-Relaxed Matching**: 針對受截斷影響的邊界物件，動態放寬同質性約束：
   - 歐氏距離容忍度放大 (`seam_center_scale` = 1.8x)。
   - 面積相似度下界降低 (`seam_area_ratio_threshold` = 0.30)。
   - 空間重疊 IoU 閾值降低 (`seam_min_overlap_ratio` = 0.45)。
3. **Representative Box Fusion**: 分數仍保留 cluster 中較可信的候選，但輸出框不再直接硬選單一 best box。
   - 對 seam box 的座標權重降級 (`tiled_seam_coord_weight`)。
   - 將降權後的加權平均框與較可信候選做輕度 blend (`tiled_best_blend`)。
   - 目的不是單純壓 FP，而是降低截斷框直接污染 tracker 的機率，同時避免過度 merge 吞掉真陽性。

### Phase 4.5: Tile Diagnostics (切片診斷)
若啟用 `--tile-diagnostics`，evaluation path 會額外輸出：
1. `pre_merge_seam`: merge 前 seam-near detections / frame。
2. `post_merge_seam`: merge 後 seam-near detections / frame。
3. `merged_clusters`: 每幀成功 merge 的 duplicate cluster 數。
4. `compression`: duplicate cluster 的壓縮率。

這些值是目前診斷 tiled path 是否存在結構性 seam 汙染的主要依據。

### Phase 5: Morphological Filtering & NMS (型態過濾與抑制)
1. **Confidence Thresholding**: 套用 `score_threshold` 與 `track_thresh` 進行第一階概率截斷。
2. **Intra-Tile NMS**: 以 `nms_iou_threshold = 0.5` 消除單一 Patch 內的冗餘預測。
3. **Heuristic Geometry Gating**: 根據人類生物特徵先驗（如 `person_max_aspect`, `person_min_area_ratio`），濾除異常型態之 False Positives。

### Phase 6: Downstream Pipeline Injection (下游管線注入)
將正規化後的感知張量推送至 `PerceptionPipeline`：
1. **ReID Extraction Routing**: 結合 Track 狀態與動態預算 (`dynamic_reid`)，決定是否啟動 LaSt-ViT 進行 ROI 裁剪與特徵提取。
2. **Tracker Binding**: 將觀測狀態與 Embeddings 推入 GPU Tracker，進入 Sinkhorn / Auction 關聯分配生命週期。

---

## 6. 模塊級指標與優化流程 (Module-by-Module Measurement Plan)

如果目標是「把每一個模塊都測清楚，再判斷是否真的有提升」，不要只看最後的 `MOTA`。建議把評測拆成三層：

1. **模塊內指標**：這個模塊自己有沒有更快、更穩、更少副作用。
2. **相鄰下游指標**：這個模塊的變化，有沒有改善下一段輸入品質。
3. **端到端指標**：最後 `IDF1 / MOTA / FP / FN / IDs` 是否真的跟著變好。

### 6.1 Module -> Metric Contract

這張表的目的不是把所有數字都攤平，而是固定每個 module 做實驗時的
「主指標、delta 解讀、可觀測性與缺口」。

| Module | Current Metric(s) | What It Means | Good Delta | Bad Delta / Risk | Current Visibility | Gap / Next Instrumentation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ingest | `fetch` | 取 frame、前段搬運與 queue 交握成本 | `fetch` mean / P95 降，且 `frame_total` 一起更穩 | `fetch` 降但 `frame_total` 不變，代表不在熱點 | Directly measured | 若未來要拆 decode 與 queue wait，需補更細分 stage |
| Preprocess | `ingest_preprocess` | resize / letterbox / gamma / contrast 成本 | latency 降，且 `detect` / `frame_total` 不惡化 | latency 降但 detector recall 掉，代表前處理破壞輸入品質 | Directly measured | 若要分 resize / gamma / contrast，需再拆 stage |
| Detection | `detect`, `raw_boxes`, `after_filter`, `after_nms`, `after_merge` | detector 推理成本與召回 funnel | `detect` 降，或 `FN / Recall` 改善且 funnel 更健康 | `detect` 降但 `FN` 升；或 `after_merge` 掉太多代表召回被吃掉 | Directly measured | 若要分 native vs tiled routing overhead，需補 detector 內子階段 |
| Post-proc | `postprocess`, `post_filter`, `post_nms`, `post_merge`, `pre_merge_seam`, `post_merge_seam`, `merged_clusters`, `compression` | 過濾、NMS、cross-tile merge 是否減噪而不吞真陽性 | `postprocess` 降，且 `FP` 降或 seam diagnostics 改善 | `compression` 高但 `FN` 上升；`FP` 降但 recall 也掉 | Directly measured | 可再補 representative box / quality scaling 子階段 |
| ReID Trigger | `reid_budget`, `lazy_reid` | 觸發決策與預算配置成本 | latency 降，或相同成本下 `IDs / IDF1` 更好 | trigger 次數大增但 `IDF1` 沒提升，表示預算浪費 | Directly measured | 可補「每幀候選數 / 真正 extract 數」固定輸出 |
| Motion: GMC | `gmc`, downstream via `track`, `IDs`, `FN`, `MOTA` | 背景運動估計是否幫助關聯穩定 | `gmc` 不膨脹，且 `IDs / FN` 降、`track` 不惡化 | `gmc` 變慢但 `IDs / FN` 沒改善 | Directly measured | 下一步可補 GMC 品質摘要，例如 uncertain-frame / PCR 統計 |
| Internal States | folded into `track` | Kalman predict / state update / covariance 更新成本 | `track` 降且穩定性不變 | `track` 降但 `IDs` / `FN` 變差 | Indirectly measured | 若 state logic 成為瓶頸，可拆 predict/update |
| ReID Extract | `reid_crop`, `reid_extract`, `native_reid_crop`, `native_reid_pre_normalize`, `native_reid_trt_enqueue`, `native_reid_l2_normalize` | ROI crop、embedding TRT、normalize 成本 | latency 降，且 `IDs / IDF1` 不退化 | latency 降但追蹤品質沒受益，表示只是省時不是有效特徵改善 | Directly measured | 若換 backbone，需固定 embedding dim / compare contract |
| GPU Association | `track` | cost matrix、gating、matching、Kalman update 總成本 | `track` 降，或 `IDs / FN / MOTA` 改善 | `MOTA` 微升但 `IDs` 暴增；或 `track` 變慢沒有品質收益 | Directly measured | 可補 cost build / assign / update breakdown |
| Materialize | `materialize` | GPU -> CPU materialize 與同步成本 | `materialize` 降且不影響 resolve | `materialize` 上升但 `track` / 品質沒收益，代表純尾端成本膨脹 | Directly measured | 下一步可補 D2H sync 與 host view 組裝的細分 |
| Identity Resolve | mostly `relink_write`, plus `IDs`, `IDF1` | semantic relink / lifecycle merge 是否真的修復 ID | `relink_write` 降，或 `IDs / IDF1` 改善 | `relink_write` 降但 `IDF1` 掉；或 `IDs` 降但 `FP` / 錯誤 merge 上升 | Indirectly measured | 可拆 semantic / lifecycle / finalize 子階段 |
| Global Identity Map | no standalone metric | local track id 映到 global id 的穩定性 | 目前只能透過 `IDs` 側推 | 無法區分 mapper 問題還是上游 resolve 問題 | Not separately measured yet | 若 global-id churn 成問題，需補 map update / reuse 統計 |
| Output & Feedback | mostly `relink_write` / `frame_total` | bank update、offline stitch、metrics emit 尾端成本 | 尾端成本降，且 bank / output 品質不退化 | 尾端變慢但整體品質無收益；或 bank update 改壞後續穩定性 | Indirectly measured | 可補 bank update / emit / stitch 三段細分 |

### 6.2 模塊與對應指標矩陣

| 模塊 | 主要看什麼 | 下游連動指標 | 建議工具 / 輸出 |
| :--- | :--- | :--- | :--- |
| Ingest / Preprocess | `ingest_preprocess` latency、`frame_total` jitter | detection latency、總 FPS | `scripts/eval/mot17.py --profile-stages`、`scripts/benchmarks/latency_e2e_report.py` |
| Detection | `detect` latency、`raw_boxes/frame` | `after_filter`、`after_nms`、`after_merge`、Recall / FN | `scripts/eval/mot17.py`、`scripts/eval/ablation_mot17.py --category detection` |
| Post-filter / NMS / Cross-tile Merge | `post_filter`、`post_nms`、`post_merge` latency | FP / FN、`compression`、`merged_clusters`、seam 汙染 | `--profile-stages` + `--tile-diagnostics` |
| ReID Trigger / Budget | `reid_budget` latency、lazy ReID candidates/frame | `reid_crop` 次數、`reid_extract` 次數、IDs | `--profile-stages`、lazy ReID profiling 輸出 |
| ReID Crop / Extract | `reid_crop`、`reid_extract` latency | IDF1、IDs、relink 成功率 | `--profile-stages`、native ReID breakdown |
| GMC / Tracker Association | `track` latency | MOTA、IDs、FN、軌跡穩定性 | `ablation_mot17.py --category association,geometry`、`scripts/benchmarks/benchmark_association.py` |
| Relink / Lifecycle | `relink_write` latency | IDF1、IDs、短軌跡數、斷軌修復效果 | `ablation_mot17.py --category semantic,lifecycle` |
| Output / Materialize | `frame_total` 尾端抖動 | 整體吞吐與 async overlap 效果 | `--profile-stages`、`_fps_summary.txt` |

### 6.3 目前主流程已能直接觀測的指標

`src/saccade/perception/eval/runner.py` 已經能輸出下列量，這些應該成為模塊優化的第一層基準：

- **Top-level stage latency**:
  `fetch`、`ingest_preprocess`、`detect`、`postprocess`、`reid_bank_sync`、`reid_budget`、`reid_crop`、`reid_extract`、`lazy_reid`、`track`、`relink_write`、`frame_total`
- **Postprocess breakdown**:
  `post_filter`、`post_nms`、`post_merge`
- **Detection box count funnel**:
  `raw_boxes`、`after_filter`、`after_nms`、`after_merge`
- **Tiled detection diagnostics**:
  `pre_merge_seam`、`post_merge_seam`、`merged_clusters`、`compression`
- **Native ReID breakdown**:
  `native_reid_crop`、`native_reid_pre_normalize`、`native_reid_trt_enqueue`、`native_reid_l2_normalize`
- **End-to-end tracking metrics**:
  `IDF1`、`Recall`、`Precision`、`MOTA`、`IDs`、`FP`、`FN`

也就是說，現在最缺的不是「再發明一套指標」，而是把這些現有輸出固定成每次模塊改動都要看的對照面板。

### 6.4 建議的 A/B 測試規則

每次只改一類模塊，其他條件固定：

- 固定 `engine`、`tiling`、`sequence`、`split`、`max_frames`
- 固定是否開 `ReID / relink / GMC`
- 先做 **short profiling run** 看 stage 與 box funnel
- 再做 **same-sequence end-to-end run** 看 `IDF1 / MOTA / FP / FN / IDs`
- 若是 detector / tiled merge 類改動，再額外看 `tile diagnostics`
- 若是 ReID / relink 類改動，再額外看 `IDs` 與 `IDF1`，不要只看 `MOTA`

建議至少保留兩組對照：

- **局部敏感序列**：例如 crowded / seam-heavy / ID-switch-heavy 的 MOT17 序列
- **回歸控制序列**：至少一組較穩定序列，避免只在單一場景 overfit

### 6.5 現行執行順序

這套流程現在已經收斂成固定入口：

```bash
./scripts/eval/module_benchmark.sh --mode all
```

這個 wrapper 會固定做三件事：

1. `profile`: `mot17.py --profile-stages`
2. `ablation`: `ablation_mot17.py` grouped sweep
3. `validate`: `mot17.py` non-profile end-to-end run

同一個 output root 會自動產生：

- `summary.txt`: 這次 run 的完整配置
- `commands.txt`: 實際發出的命令
- `notes.md`: 假設 / 發現 / 決策摘要
- `experiment_matrix.md`: 對齊下方表格格式的紀錄模板

#### A. 建立 baseline

目前建議先用已驗證的 native-960 baseline：

```bash
./scripts/eval/module_benchmark.sh \
  --mode all \
  --output-root results/module_benchmark/baseline_native_960
```

對應配置：

- detector: `SDP`
- sequences: `MOT17-04-SDP,MOT17-10-SDP`
- split: `train`
- engine: `models/yolo/yolo26s_960_batch1.engine`
- tiling: `native_960`
- max_frames: `100`

#### B. 只提升有訊號的 category

現行做法不是每次都重跑全量 sweep，而是：

1. 先看 `ablation` summary table。
2. 只挑出少數有正向訊號的變體。
3. 再為該變體跑獨立 validate。

例如目前 baseline 中唯一明顯有訊號的是：

- `geometry mid-scale`

#### C. 再做 focused candidate validate

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/eval/mot17.py \
  --output results/module_benchmark/candidate_geometry_mid_scale \
  --detector SDP \
  --sequences MOT17-04-SDP,MOT17-10-SDP \
  --max-frames 100 \
  --geometry-mid-scale
```

比對重點：

- `IDF1` 是否提升
- `MOTA` 是否提升
- `FN` 是否下降
- `FP` / `IDs` 是否沒有用過高代價換來表面提升

### 6.6 各模塊優化時應優先盯的訊號

#### Detection / Tiling

- 主看：`Recall`、`FN`、`raw_boxes -> after_merge` funnel
- 次看：`pre_merge_seam` 是否高、`post_merge_seam` 是否有效下降
- 風險訊號：`compression` 很高但 `FN` 上升，通常代表 merge 過頭

#### Postprocess

- 主看：`post_filter / post_nms / post_merge` latency
- 次看：`after_filter / after_nms / after_merge`
- 風險訊號：`FP` 下降但 `FN` 明顯上升，代表過濾規則太 aggressive

#### ReID Trigger / Budget

- 主看：`reid_budget`、lazy candidates/frame、`reid_crop` 次數
- 次看：`IDs`、`IDF1`
- 風險訊號：提特徵次數大增但 `IDF1` 沒提升，代表預算浪費或 trigger 錯誤

#### ReID Extract

- 主看：`reid_extract` latency 與 native breakdown
- 次看：`IDF1`、`IDs`
- 風險訊號：`reid_extract` 變慢但 `IDs` 沒下降，表示純化或模型切換沒有產生實益

#### Association / Geometry / GMC

- 主看：`track` latency、`IDs`、`FN`
- 次看：`MOTA`
- 風險訊號：`MOTA` 微升但 `IDs` 暴增，通常是 match / geometry gating 失衡

#### Semantic Relink / Lifecycle

- 主看：`IDF1`、`IDs`、短軌跡修復效果
- 次看：`relink_write` latency
- 風險訊號：`IDF1` 升但 `FP` 或錯誤 merge 增加，代表 relink 門檻過鬆

### 6.7 建議補上的實驗紀錄格式

每次模塊優化，至少記一份表：

| Experiment | Module | Sequence | Key Flags | IDF1 | MOTA | FP | FN | IDs | detect ms | post ms | track ms | reid ms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

建議規則：

- `detect ms` 取 `detect`
- `post ms` 可取 `postprocess`，必要時再拆 `post_filter/nms/merge`
- `reid ms` 取 `reid_crop + reid_extract`
- 每個實驗都標註 category 與主要 flags，避免之後無法回推原因

目前這張表已經由 `scripts/eval/module_benchmark.sh` 自動生成在每次 run 的
`experiment_matrix.md`，建議直接在該檔案上維護，而不是另外手抄。

### 6.7.1 已落地 baseline

第一次完整落地的基線在：

- `results/module_benchmark/baseline_native_960`

核心結果：

| Experiment | Module | Sequence | Key Flags | IDF1 | MOTA | FP | FN | IDs | detect ms | post ms | track ms | reid ms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_native_960_validate` | baseline | `MOT17-04-SDP,MOT17-10-SDP` | `engine=models/yolo/yolo26s_960_batch1.engine; tiling=native_960; max_frames=100` | `9.4%` | `4.5%` | `458` | `57223` | `18` | `4.53` | `1.40` | `1.10` | `1.88` |
| `baseline_native_960_ablation_geometry_mid_scale` | geometry | `MOT17-04-SDP,MOT17-10-SDP` | `engine=models/yolo/yolo26s_960_batch1.engine; tiling=native_960; max_frames=100; geometry-mid-scale` | `9.9%` | `4.7%` | `509` | `57062` | `14` | - | - | - | - |

目前觀察：

- `geometry mid-scale` 是唯一明顯正向的 ablation 候選。
- `trigger` / `semantic` 類旋鈕大致持平。
- `reid tracker_mode` / `hybrid_mode` 明顯退化。
- `reid osnet` 目前因 `512` vs `768` embedding 維度不相容而不可比。

### 6.8 實務結論

這條 pipeline 的優化不應再只用「總分有沒有變高」判斷。現行穩定做法是：

1. 先用 `module_benchmark.sh` 建立 profile / ablation / validate baseline。
2. 再用 stage latency、box funnel、ReID 統計看中間品質。
3. 只把有明顯訊號的候選提升成 focused validate。
4. 最後才決定是否保留該變體。

這樣才能分辨：

- 是 detector 真的召回變好，
- 還是 postprocess 只是把框壓少了，
- 還是 ReID / relink 在掩蓋前段問題，
- 或只是把某個序列 tuning 到過擬合。
