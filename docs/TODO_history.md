# Saccade TODO History

> 從 [docs/TODO.md](/docs/TODO.md) 拆出的歷史/脈絡內容。保留已完成項、設計規範與 C++ 化路線圖，避免主 TODO 被長篇歷史淹沒。

---

## ❌ 放棄 / Deferred（2026-05-05）

### D3：Tiled Detector 流程修補（2026-05-05）

- **`native_960` control 已建立**：`scripts/eval/mot17.py` 新增 `--tiling native_960`，對照 engine `models/yolo/yolo26s_960_batch1.engine`。
- **兩序列控制結果確認 tiled 流程問題**：
  - `960p_2x2 tiled` baseline：`IDF1 43.6 / MOTA 31.6 / IDs 187 / FP 8516 / FN 32584 / Rcll 46.0`
  - `native_960`：`IDF1 47.4 / MOTA 41.4 / IDs 151 / FP 4222 / FN 31007 / Rcll 48.7`
  - 結論：這不是單純 threshold tradeoff，而是 tiled detection path 本身引入 seam duplicate / truncation / score calibration 汙染。
- **已完成 seam-aware duplicate merge 與診斷工具**：
  - 新增 `--tile-diagnostics`、`--tile-seam-margin-canvas-px`
  - 新增 `--cross-tile-seam-center-scale`
  - 新增 `--cross-tile-seam-area-ratio-threshold`
  - 新增 `--cross-tile-seam-min-overlap-ratio`
  - native CUDA/C++ path 已支援 seam-aware duplicate merge
- **已驗證無效或不值得繼續的方向**：
  - `tile-seam-score-penalty`：能降 FP，但會一起打掉真陽性，FN/Recall 變差
  - 單純放寬 seam duplicate gate：merge 更積極，但會開始吞真陽性，MOTA 反降
- **目前保留但尚未收斂的方向**：
  - 代表框邏輯已從「硬選單一 best box」改為「偏向非 seam 候選的融合框」
  - fused representative box 相比 native seam-aware baseline 只帶來小幅改善：
    - baseline：`IDF1 42.9 / MOTA 32.5 / IDs 197 / FP 7746 / FN 32845 / Rcll 45.6`
    - fused box：`IDF1 43.6 / MOTA 32.2 / IDs 192 / FP 8022 / FN 32729 / Rcll 45.8`
  - 解讀：方向比繼續掃 gate 更合理，但仍未接近 `native_960`

### B1 P3：Bank Retrieval 長期方向

- **P3-A（Full CUDA Bank T×K×D tensor）**：deferred，僅在 T≥500 時有性能意義；目前 T 不達門檻，不排期。
- **P3-B（Dormant Bank + HNSW 長期 ReID）**：D1 診斷確認放棄。match_gap ≥ 91f 僅佔 26/664（3.9%）ID switches，天花板過低，實作成本不值。
- B1 P0→P2 Bank Zero-Copy 已完成（5.8×/11.9× 增益）；P3 確認不列入近期計畫。

---

## ✅ 已完成（2026-04-25）

### P0 — GPUByteTracker 核心強化（ADR 013）
- [x] **ReID 融合代價矩陣**：`tracker_gpu.cu` cost matrix 改為 `(1-w)*IoU + w*CosSim`，預設 w=0.5，crowded 場景 w=0.8
- [x] **Strong ReID Gate**：CosSim > 0.75 時強制配對，對抗相機劇烈晃動
- [x] **GMC 全域運動補償**：Python 層 optical flow → 仿射矩陣傳入 C++ `gmc_kernel`，同步修正 Kalman 狀態與協方差
- [x] **Light Compensation**：`light_factor` 動態調整 R 矩陣，穩定夜間軌跡
- [x] **Saccade Heartbeat 間隔修正**：`% 30` → `% 10`，對齊 ADR 013 規格

### P1 — 媒體層穩定性
- [x] **RTSP 斷線自動恢復**：`mediamtx_client.py` 加入 `watchdog_loop()`，`_is_alive()` 偵測超時後呼叫 `_restart_pipeline()` 重建 GStreamer pipeline

### P2 — 儲存層
- [x] **Redis Micro-batching**：`MicroBatcher` 整合於 `RedisCache.publish_event()`，100ms 視窗聚合，Redis QPS ~300 → ~30

### P3 — 認知層（ADR 014）
- [x] **LlamaIndex RAG 接入**：`orchestrator.py` 連接 ChromaDB → LlamaIndex，使用 `BAAI/bge-small-en-v1.5` local embedding + Ollama llama3
- [x] **事件觸發式查詢**：`entropy > 0.9` 或 `is_anomaly=True` 時才觸發 RAG，避免每幀呼叫
- [x] **Visual Re-query (視覺重查)**：在 `orchestrator.py` 中註冊 `visual_requery` Tool 給 ReAct Agent，允許 LLM 發起 ChromaDB 純向量搜尋 (Image-to-Image 語義比對)。
- [x] **跨鏡頭 Re-ID**：重構 `FeatureBank` 支援 `stream_map`，實作 `find_cross_camera_matches` 矩陣運算，讓多路串流可共享特徵索引並進行跨畫面比對。

### P4 — 基礎設施與維運（Infrastructure & Maintenance）
- [x] **ChromaDB 冷備份**：於 `ChromaStore` 中實作 `backup()` 函數，利用 `shutil.make_archive` 定期壓縮並 snapshot 向量資料庫，防止長期記憶遺失。
- [x] **串流身分驗證**：修改 `infra/mediamtx.yml`，為發布 (publish) 與讀取 (read) 動作加入帳號密碼保護。
- [x] **Redis 自動清理**：實作 `cleanup_expired_objects()`，監控 Redis 記憶體使用量，當超過閾值時強制刪除最舊的 `saccade:obj:*` 快取，避免記憶體溢出。
- [x] **智慧影格抽樣策略**：在 `MediaMTXClient` (含 C++ 及 Python 回調) 中實作像素差異比對 (SAD < 2.0)，即時丟棄低資訊幀，降低無效計算負載。

---

## 精度提升 — 算法改進 (2026-04-27)

基於 Saccade vs Ultralytics 對比分析，Prcn 落後 ~4pp，FP 過多。以下為針對性算法改進。

### A — Offline Tracklet Quality Filter ✅
- [x] **離線軌跡品質過濾**：在 `post_merge_output_tracklets` 之後，過濾掉長度不足或平均分數過低的軌跡段，消滅短命 FP 軌跡。
  - `perception/eval/tracking.py`: 新增 `filter_low_quality_tracklets(lines, *, min_len, min_score)`
  - `perception/eval/runner.py`: 接在 `post_merge_output_tracklets` 之後呼叫
  - `scripts/eval/mot17.py`: `--min-tracklet-len`（預設 1）、`--min-tracklet-score`（預設 0.0）

### B — NSA-Kalman（Noise Scale Adaptive Kalman）✅
- [x] **分數自適應測量雜訊**：對每個偵測結果，依信心分數縮放 Kalman R 矩陣：`nsa = max(0.05, (1-score)²)`，讓高信心偵測有更強的更新效果，低信心偵測保持更保守。
  - `include/tracking/kalman_gpu.cuh`: `get_R()` 加入 `nsa_multiplier` 參數；`update()` 加入 `nsa_multiplier` 參數
  - `src/tracking/tracker_gpu.cu`: `inline_kalman_update_kernel` 接受 `det_scores` + `nsa_kalman` 旗標
  - `include/tracking/tracker_gpu.hpp` + `src/tracking/tracker_gpu.cu` + `src/tracking/tracker_gpu_python.cpp`: `set_params` 加入 `nsa_kalman` bool
  - `perception/tracking/tracker_gpu.py` + `perception/eval/runner.py` + `scripts/eval/mot17.py`: 對應 Python 層露出 `--nsa-kalman` 旗標

### C — Mahalanobis Gating（已併入 Phase 2）
- [x] 將 cost matrix kernel 中的固定空間閘替換為馬氏距離閘；後續實作與結論見下方「ReID 強化與 Association 重構 / Phase 2」。

### D — Score Decay（關閉：無效）
- [x] ~~Score Decay~~ — 分析確認 unmatched confirmed tracks 的 `age>0` 條件已阻止其輸出，Score Decay 對 FP 無影響，放棄。

### E — Semantic Relinker 閾值調優 ✅
- [x] **ReID 預設基線重設**：`--semantic-threshold 0.90` 與 primary appearance bank (`--appearance-bank-consistency-threshold 0.75`) 已升格為新 base；最新 `scripts/eval/ablation_relink.py --detector SDP` 重跑結果為 IDF1 45.4%、MOTA 34.7%、IDs 837、FP 16,687、FN 55,765。
- [x] **Post-lifecycle merge（motion-only）確認有害**：IDF1 最多跌 2.5pp，設計缺陷為缺乏 appearance gate，不建議啟用。

### F — 後續延伸（已完成 / 已收斂）
- [x] **Appearance-gated PostMerge**：在現有 `post_merge_output_tracklets` 的 Hungarian 匹配候選上加 cosine 相似度確認，避免 motion-only PostMerge 錯合不同人。
  - `perception/eval/runner.py`: 新增 `OutputAppearanceBank`，保留每個 output ID 的 Top-K 高分 embedding，PostMerge 候選需通過 appearance threshold / consistency / min-samples gate
  - `scripts/eval/mot17.py`: 新增 `--post-lifecycle-appearance-gate` 與 threshold / samples / score / consistency 參數
- [x] **Per-sequence 參數自適應**：讀取 `seqinfo.ini` 的 `frameRate`，自動 scale `reid_interval`（∝ fps/30）與 `track_buffer`；`--per-seq-adapt` / `--no-per-seq-adapt` flag（預設啟用）。
- [x] **換 ReID 模型（OSNet / FastReID TRT）framework 支援**：已加入 `ModelType.OSNET` / `ModelType.FASTREID` enum（C++ + Python），預設 engine 路徑已設定，ImageNet normalize 共用 DINOv2/TransReID 路徑；crop size 自動 256×128。
  - OSNet build path 已整理：預設 ONNX `models/embedding/osnet_x1_0_256x128.onnx`，預設 engine `models/embedding/osnet_x1_0_256x128.engine`
  - OSNet ONNX / TensorRT engine 已建立，`scripts/model/build_osnet.py` 可重建；`scripts/eval/ablation_reid_models.py` 已完成 `siglip2 / transreid / osnet` backbone 對比

---

## ReID 強化與 Association 重構（規範 2026-04-27）

目標：IDF1 ↑ 3~6pp、IDs ↓ 20~40%。核心：將 ReID 從「事後修正」提升為「決策核心」，確保進入 ReID 的 embedding 乾淨且穩定。

設計原則（強制）：
- ReID 必須參與 primary association，不只做 relink
- Noisy embedding 不得進入身份決策
- 不允許 pure motion-based merge
- 不允許 global weighted cost（IoU + CosSim）
- 所有 fallback 必須穩定（IoU only）

### Phase 1 — 必要（核心功能）

- [x] **Top-K Appearance Bank**：取代 EMA embedding，每 track 維護 `List[AppearanceSample]`（K=5）。
  - 收樣條件：`det_score >= 0.45 and iou >= 0.35 and geometry_clean and not suspect_box`
  - 排名：`0.5*det_score + 0.3*iou + 0.2*recency_weight`，保留最高 K 個
  - Matching：`max cosine(det.emb, s.emb)` over bank（透過 `d_features_` 傳入 C++ kernel）
  - 實作：`perception/tracking/tracker_gpu.py` `AppearanceSample` + `TrackAppearanceBank`；`perception/eval/runner.py` bank 更新與 `set_reference_features_from_bank`

- [x] **ReID 觸發機制（need_reid）**：取代固定 heartbeat，依情境決定是否提取 embedding。
  - 觸發（任一成立）：active tracks 數 > det 數、任一 det.score < 0.45、det 數 > active tracks 數
  - 無 active tracks / 無 det → False（不提取）
  - 實作：`perception/tracking/tracker_gpu.py` `need_reid_frame()`；`perception/eval/runner.py` 替換 `frame_id % reid_interval`；`--need-reid` / `--no-need-reid` flag

- [x] **Conditional Appearance Matching（兩階段 Association）**：取代全域加權 cost matrix。
  - Stage 1 IoU Gate：`IoU > 0.3`（hard gate，失敗 → cost=1.0）
  - Stage 2 條件：`candidate_count[t] >= 2 AND has_clean_embedding[t]` → `1-(0.55*CosSim+0.30*IoU+0.15*score)`；否則 `1-IoU`
  - `update_reference_features` 已實作（原 stub 已補齊）；`set_clean_embedding_flags` 同步 bank 狀態至 C++
  - 實作：`src/tracking/tracker_gpu.cu` `count_stage1_candidates_kernel` + `compute_conditional_cost_kernel`；`--appearance-bank` / `--appearance-bank-size` flag

### Phase 2 — 強化

- [x] **Appearance Consistency Gate**：防止污染的 track 參與 ReID / relink。
  - `bank_consistency = mean(cosine(bank[i], bank[j]) for all i<j pairs)`
  - `consistency < 0.82` → `track.disable_reid = True`（不參與 relink，fallback IoU）
  - 修改：`perception/eval/runner.py` — relinker call 前檢查 `is_consistent()`，不一致則傳 `None` embedding

- [x] **Mahalanobis Gating（替換固定空間閘）**：對應 C 項，Phase 2 落實。
  - `d² = (z-Hx)^T S^-1 (z-Hx)`，需 Kalman 預測後計算 covariance S
  - Gate：`MAHA_GATE = 9.4877`（chi-square df=4, 95% confidence），僅用於 Stage 1 gating，不進入 cost function
  - 修改：`include/tracking/kalman_gpu.cuh` 暴露 `compute_S_inv()`；`src/tracking/tracker_gpu.cu` 新增 `compute_innovation_sinv_kernel` + `mahal_sq_det()`；Stage 1 gate = `IoU > 0.3 OR Mahal² < 9.4877`

### Phase 3 — 選用（條件式啟用）

- [x] **Appearance-gated PostMerge v2**：預設關閉（motion-only 已確認有害）。
  - 啟用前提：Top-K Bank ✅ + Consistency Gate ✅ + appearance similarity > 0.9
  - 不允許 pure motion-based merge
  - 實作：`perception/eval/runner.py` 強制檢查 — `--post-lifecycle-merge` 啟用時若 `--post-lifecycle-appearance-gate` 未設，自動啟用並印出警告，禁止 pure motion-based merge

---

## C++ 化路線圖（2026-04-27）

目標：Python 保留 CLI / experiment orchestration / motmetrics 報表，per-frame perception 熱路徑逐步移到 C++/CUDA。

### CXX-1 — SemanticRelinker C++ 化 ✅
- [x] 在 `saccade_tracking_ext` 暴露 C++ `SemanticRelinker`
- [x] Python `perception/eval/relink.py` 優先使用 C++ relinker，失敗時 fallback 原 Python 實作
- [x] 保持現有 runner API：`resolve(raw_id, emb, box, score, frame_id, w, h, assigned)`
- [x] 跑 Python/C++ 行為 smoke，並跑 MOT17 short eval

### CXX-2 — Detection Postprocess C++/CUDA 化 ✅
- [x] 搬 confidence/class filter、person geometry prior、suspect support 到 C++/CUDA binding；CUDA fast path 使用 caller stream，CPU/Python 保留 fallback
- [x] 搬 NMS / remaining postprocess compaction 到 C++/CUDA；`nms_fast` 支援 class-agnostic / class-aware parallel bitmask/block NMS，cross-tile merge 已走 CUDA fast path
- [x] Python `perception/eval/detection.py` 的 postprocess 熱路徑保留 wrapper/fallback，主路徑走 `saccade_tracking_ext`

### CXX-3 — GMC C++ 化 ✅
- [x] 將 `SparseOpticalFlowGMC` 搬到 C++ OpenCV
- [x] 保持 affine warp 輸出格式與 tracker `gmc_kernel` 介面一致

### CXX-4 — ReID Crop / Embedding Pipeline C++ 化 ✅
- [x] C++/CUDA cropper
- [x] C++ TensorRT ReID extractor
- [x] embedding 直接供 tracker/relinker 使用，減少 Python tensor orchestration
  - `TrackResult.det_idx`: tracker 直接暴露每個 track 的匹配 detection index，消除 runner.py 的 O(n_tracks×n_dets) IoU 重計算迴圈
  - `FeatureExtractor::extract_parts_fused()`: 3-part crop 的加權融合 [0.5,0.3,0.2] + L2-normalize 移至 CUDA kernel，Python 端只需一行 call
  - OSNet / FastReID model type 支援已加入 C++ ModelType enum 與 Python wrapper；此處保留的是當時擴充範圍的歷史記錄

### CXX-5 — Eval / Product Pipeline Facade ✅
- [x] 建立 C++ `PerceptionPipeline::process_frame`
  - `include/tracking/pipeline.hpp` + `src/tracking/pipeline.cpp`：封裝 filter + NMS + ReID crop/extract 為單一 C++ facade
  - Python binding 於 `saccade_tracking_ext.PerceptionPipeline`；`PerceptionPipelineConfig` 暴露所有 filter/NMS 參數
- [x] Python MOT runner 只呼叫 C++ facade
  - runner.py 的 parts-fusion 路徑改呼叫 C++ `extract_parts_fused()`；det_idx 消除 IoU 迴圈
  - PerceptionPipeline 可作為 process_detections + extract_reid 的統一入口
- [x] 視需要新增 `saccade_eval_mot17` binary（評估結果仍依賴 Python motmetrics，不建立獨立 binary）

---

## ✅ B1 Bank Zero-Copy + A1–A8 Ablation Series（2026-05-01 ~ 2026-05-05）

### B1：Bank Retrieval Zero-Copy Optimization

- [x] **P0（C++ backend）**：`h_tid_to_slot_` map + `ensure_slot_map()` 共享懶同步（完成 2026-05-04）
  - `Impl` 新增 `h_tid_to_slot_: unordered_map<int,int>` + `h_slot_map_dirty_` flag
  - `ensure_slot_map()`：只在 dirty 時 D2H `d_active_` + `d_track_ids_`，重建 map；兩次呼叫共享同一次 D2H
  - 改寫 `update_reference_features_impl`：map O(1) lookup + D2D scatter，消滅 O(N×2048) 巢狀迴圈
  - 改寫 `set_clean_embedding_flags`：同上，消滅重複 D2H
  - `update()` / `update_into()` 結尾設 `h_slot_map_dirty_ = true`

- [x] **P1（Python bank）**：`_representatives` / `_high_quality_reps` 改為 GPU Tensor（完成 2026-05-04）
  - `_refresh_track()`：representative 計算後加 `.cuda()`，消滅 `set_reference_features_from_bank` 內的 H2D
  - `set_reference_features_from_bank()`：`torch.stack` 已在 GPU → `.to(device=device)` 變 no-op

- [x] **P2（slot-indexed zero-copy）**：`bind_features_buffer` + `get_active_tid_slot_pairs` + 批次 GPU scatter（完成 2026-05-04）
  - C++ `Impl`：`bind_external_features_buffer()`、`get_active_tid_slot_pairs()`、`d_features_owned_` flag
  - Python `GPUByteTracker.__init__`：預先配置 `_bank_slot_features[MAX_OBJ, D]`，呼叫 `bind_features_buffer`
  - Python `set_reference_features_from_bank`：改用 `_bank_slot_features[slots] = stacked`，完全消滅 N×D2D scatter
  - 實測增益（T=50）：P0→P2 **5.8×**；T=200：**11.9×**；固定成本壓到 ~175µs/frame（D=768）

### 多進程記憶體管理（已完成 2026-05-04）

- [x] **Option A — Dispatcher GPU Tracker LRU Eviction**
  - `AsyncDispatcher.trackers` 改為 `OrderedDict`，加入 `max_streams=8` 上限。
  - `get_tracker()` 命中時 `move_to_end`；超限時 `popitem(last=False)` + `del`，觸發 `~GPUByteTracker → cudaFree`。
  - 新增 `deregister_stream()`（主動釋放）與 `stop()` 清空全部 tracker。
- [x] **Option B — 跨進程 VRAM 狀態廣播（POSIX SharedMemory）**
  - `resource_manager.py` 新增 `VRAMLevelWriter` / `VRAMLevelReader`（具名 `saccade_vram_level`，1 byte）。
  - Dispatcher `_worker_loop` 每幀廣播 `DegradationLevel`；Writer 啟動時自動清除 stale segment。
  - Orchestrator `handle_cognitive_event` 依等級 gate：FAST_PATH (>92%) 跳過 RAG，EMERGENCY (>96%) 丟棄非異常 frame。
  - 採 `multiprocessing.shared_memory.SharedMemory` 而非 `multiprocessing.Value`（兩進程獨立啟動，無共同父進程）。

### A1：Unified Association Score Ablation（已完成 2026-05-01）

- [x] 替換了 `tracker_gpu.cu` 與 `relink.py` 中的硬性閾值，改用 `w_sim_base`, `w_iou_base`, `w_maha_base` 與動態調整。
- [x] 加入了 `shift_ambiguity` 與 `shift_lost_age`。
- [x] 於 `ablation_mot17.py` 整合 Optuna 進行貝氏最佳化，支援 `--optuna a1`。
- 結論：`w_sim/iou/maha` 動態權重整合完成，ReID 資源根據 track risk 分配。

### A2：Reference Quality Gate Sweep（已完成 2026-05-04）

- [x] 掃描維度：`clean_margin_ratio`（0, 0.02, 0.05, 0.08）、`clean_min_aspect`（1.0, 1.2, 1.5, 2.0）、`clean_max_aspect`（3.5, 4.5, 6.0）及組合，共 13 variants（7-seq SDP）。
- 結論：所有 variant 差距 ≤ 0.2pp IDF1，在 run-to-run noise 範圍內，無系統性增益。
  - **根本原因**：`clean_score_threshold=0.65` 已捕捉主要觀測品質訊號；frame-edge margin 與 aspect bounds 在 MOT17 密集行人場景下不是有效的 false-accept lever。
  - **不納入 default**，保持 `margin_ratio=0.0`、`min_aspect=1.2`、`max_aspect=4.5`。

### A2-L：Pre-hoc Embedding Quality — LaSt-ViT CUDA Kernel（CLOSED No-Go 2026-05-02）

- [x] **結論：** SigLIP2 未以 LaSt-ViT 目標訓練，`last_hidden_state` 的前景/背景穩定性無法區分（stab ~0.12 均勻，p=0.386），inference-time post-processing 的 IDF1 增益僅 **+0.09pp**（MOT17-04-SDP 全序列），遠低於 +1.0pp go/no-go 門檻。
- 落地產出（保留）：CUDA kernel `preprocessor_gpu.cu`、C++ API `FeatureExtractor::extract_with_stability()`、cuFFT 正規化 bug fix、驗證腳本 `scripts/eval/validate_last_vit_phase0.py`、`tests/test_last_vit_cpp_vs_python.py`（18 tests）。
- **根本限制：** LaSt-ViT 增益來自訓練期骨幹校正，非 inference formula。若要啟用需重訓 SigLIP2 with LaSt-ViT 聚合層。
- 詳細分析：`docs/experiments/reid/last_vit_integration_analysis.md` §9

### A3：Track-Level Budgeted ReID Sweep（已完成 2026-05-01）

- [x] 比較現行 frame-level `DynamicReIDController`、track-level candidate prioritization、固定 budget 與動態比例。
- 結論：**Dynamic Ratio 0.2 (`--reid-budget 0.2`)** 是目前最強配置。300 幀基準測試中，20% 預算帶來 **+24% FPS**，同時保持（甚至微幅提升）IDF1。

### A4：GMC Quality / Background Mask Ablation（已完成 2026-05-03）

- [x] **PCR score exposure**：`find_peak_subpixel_kernel` 現在將 peak/RMS ratio 寫入 `d_pcr_score_` buffer；C++ 公開 `GMC::pcr_score()`；Python binding 已加入。
- [x] **PCR → ReID feedback**：runner.py 在 `gmc_warp` valid 但 `pcr_score < --gmc-pcr-uncertain-thresh`（default 8.0）時，設 `gmc_uncertain=True`，對應 `_budget_reid_candidates` 將所有 track 優先度 ×1.5。
- [x] **Foreground mask kernel**：新增 `zero_rects_kernel` + `launch_zero_fg_rects`；C++ `set_fg_mask_boxes()`；Python binding；runner.py `--gmc-fg-mask` flag。
- 結論：`--gmc-fg-mask` 無效（MOT17 背景紋理足以主導 Phase Correlation peak），不納入 default。PCR uncertain threshold（default 8.0）作為保護性機制保留。

### A5：Post-Merge V2 Cost Ablation（已完成 2026-05-03）

- [x] `post_merge_output_tracklets` 加入 `appearance_weight`（soft cost）、`gap_uncertainty_weight`、`consistency_weight`、`missing_appearance_cost`。
- [x] runner.py 讀取對應 kwargs；當 `appearance_weight > 0` 時不強制 appearance gate。
- 結論：`max_cost=0.8` 是目前 appearance-weighted post-merge 的安全邊界：FP -104，IDF1 +0.1pp（noise range）。
  - Post-merge 從「有害」變「中性偏正」。**不納入 default**（`--post-lifecycle-merge` 預設仍為 off）。
  - 若未來 online tracking IDF1 提升，post-merge 可再評估是否帶來 additional gain。

### A6：Detection / Bank Sample Quality Scoring Ablation（已完成 2026-05-03）

- [x] 實作 `_compute_detection_quality_batch`：將 `aspect / center / area` 收斂成 [0, 1] 品質因子。
- [x] 支援 `--detection-quality-scaling`：對所有觀測進行 soft-scaling，取代原本的 binary suspect capping。
- [x] 強化 `_compute_bank_quality_score`：加入 area penalty 並暴露權重至 CLI。
- [x] **CUDA 移植**：`apply_detection_quality_scaling_kernel` 落地，納入 `GPUByteTracker` 核心更新路徑。
- 結論（7 序列 SDP）：Detection Quality Scaling 帶來 MOTA **+1.9pp**（31.4%→33.3%），FP **-28.8%**（20635→14689），IDs **-23.5%**（827→633）。已成為系統預設值。

### A7：Quality-Aware Sinkhorn / SelectMOT Integration（已完成 2026-05-02）

- [x] 最終方案：`v2_aspect_only_soft`（對極端長寬比進行機率衰減）。
- 結論：IDs 738→722（-2.2%），MOTA 32.6%（持平），FN/Recall 基本無損。成功在不犧牲 Recall 的情況下抑制遮擋引起的錯誤關聯。已永久整合至 `src/tracking/tracker_gpu.cu`。

### A8：Uniform CMC & 2D MMD / UCMCTrack Integration（已完成 2026-05-03）

- [x] **純 GPU GMC**：實作基於 `cuFFT` 的相位相關（Phase Correlation）演算法，取代 OpenCV LK，達成 100% Zero-Copy。
- [x] **2D MMD 基礎建設**：在 `tracker_gpu.cu` 中實作透視變換投影，支援 Bbox 底部中點映射至地平面進行距離計算。
- [x] 三個 bug fix（PCR 缺失、complex buffer h/w 對調、per-frame cudaMalloc → class pre-allocate）。
- 結論：GPU GMC 43.4% IDF1 > CPU GMC 41.9% > no GMC 41.7%（SDP 7 序列）。FPS 提升約 5~10%。

### Rerank Phase 3：Reference Quality + False-Accept Filtering（已完成 2026-05-01）

- [x] `TrackAppearanceBank` 高品質過濾實作完成。
- [x] `PythonSemanticRelinker` 與 C++ `SemanticRelinkerCpp` 品質門檻同步完成。
- [x] A2 Optuna 掃描完成（30 trials）。
- 最終結論：最佳參數 `clean_score_threshold=0.65 / strict_sim_threshold=0.0 / high_quality_min_score=0.75`。
  - 效果：IDF1 43.8% / MOTA 35.3%（2026-05-04 重測，A6+working GMC 環境）。
  - 發現：`strict_sim_threshold=0.74` 語義反轉 bug；正確 default 為 0.0（fallback 到 sim_threshold）。

### Online Association / Semantic Relink 統一打分（已完成 2026-05-01）

- [x] 在 A1 Ablation 中完成，把 `appearance + motion + quality` 收斂成單一 calibrated score。
- [x] track age / lost age / candidate ambiguity / observation quality 進入權重。

### Dynamic ReID Trigger V2：Track-Level / Budgeted ReID（已完成 2026-05-01）

- [x] 實作 `DynamicReIDController.get_priorities()` 與 `get_last_boxes()`。
- [x] 在 `runner.py` 整合 `_budget_reid_candidates` 優先級排序。
- [x] 支援 `--reid-budget` 固定 budget 限制與 track-level prioritization。
- 結論：ReID 資源可根據 track risk（new/lost/unstable）進行優先分配，支援與 GMC 結合的空間優先級預測。

### Low-Priority Code TODO（已完成 2026-05-04）

- [x] `src/saccade/perception/entropy.py`：實作 Shannon entropy（類別分佈正規化）+ Object Density（線性縮放）各佔 0.5 組合。支援 `class_id` / `label` 屬性物件或字串標籤。

### Pipeline GPU 化 85%+（已完成 2026-04-30）

- [x] `M1`、`M2`、`M3`、`M3.5`、`M4` 已完成。
- 結論：runner 熱路徑的 identity resolve 已整合為 C++ pass；deterministic assignment / GMC stream / GPU-native preprocess 已落地。
