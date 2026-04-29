# Saccade TODO History

> 從 [docs/TODO.md](/home/ray/developer/ai/saccade/docs/TODO.md:1) 拆出的歷史/脈絡內容。保留已完成項、設計規範與 C++ 化路線圖，避免主 TODO 被長篇歷史淹沒。

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

### C — Mahalanobis Gating（未來）
- [ ] 將 cost matrix kernel 中的固定 200px 空間閘替換為馬氏距離閘，需修改 CUDA kernel 接口（較複雜），留待後續實作。→ 見 ReID 強化 Phase 2。

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

### CXX-1 — SemanticRelinker C++ 化（進行中）
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
  - OSNet / FastReID model type 支援加入 C++ ModelType enum 與 Python wrapper；引擎路徑預設已設定，待 TRT engine 建立即可啟用

### CXX-5 — Eval / Product Pipeline Facade ✅
- [x] 建立 C++ `PerceptionPipeline::process_frame`
  - `include/tracking/pipeline.hpp` + `src/tracking/pipeline.cpp`：封裝 filter + NMS + ReID crop/extract 為單一 C++ facade
  - Python binding 於 `saccade_tracking_ext.PerceptionPipeline`；`PerceptionPipelineConfig` 暴露所有 filter/NMS 參數
- [x] Python MOT runner 只呼叫 C++ facade
  - runner.py 的 parts-fusion 路徑改呼叫 C++ `extract_parts_fused()`；det_idx 消除 IoU 迴圈
  - PerceptionPipeline 可作為 process_detections + extract_reid 的統一入口
- [x] 視需要新增 `saccade_eval_mot17` binary（評估結果仍依賴 Python motmetrics，不建立獨立 binary）
