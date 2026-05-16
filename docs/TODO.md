# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦、近期 ablation 結論與下一步方向。已完成項、設計規範與 C++ 路線圖已移至 [docs/TODO_history.md](/docs/TODO_history.md)。

---

## 歸檔標準

- 主 TODO 只保留三類內容：
  - 目前真的還要做的事項
  - 近期仍會影響決策的 ablation 結論
  - 下一輪已排定的實驗 / 實作 backlog
- 內容應移入 [docs/TODO_history.md](/docs/TODO_history.md) 的情況：
  - 已完成，且後續不再需要逐步追蹤
  - 已收斂並明確放棄，不再作為近期 default 候選
  - 已被新方向取代，只需保留背景與結論
  - 屬於長篇實作過程、舊路線圖或階段性 milestone，而不是當前待辦
- 歸檔時原則：
  - 主 TODO 保留高訊號摘要與最終結論
  - 細節、過程、舊參數掃描與已結案子項移入 history
  - 若某方向之後重新啟動，再從 history 摘回主 TODO，而不是在主 TODO 長期保留已結案脈絡

---

## 當前 Baseline（2026-05-11，最後更新）

| preset | IDF1 | MOTA | IDs | Rcll | FPS |
|--------|------|------|-----|------|-----|
| **speed**（yolo26s） | **52.0%** | **41.6%** | **475** | 55.0% | **97.9** |
| **baseline**（yolo26m） | **51.4%** | **43.5%** | **502** | 59.0% | ~85 |

已 default 的 flag：`fuse_score_weight=0.4`、`interp`、`fp_hard_filter`（area=40000）、`kalman_r_scale=0.75`、`async_reid`、`pipeline_relink`、`gmc gpu`、`detection_quality_scaling`。

---

## 待辦事項

| 優先 | 項目 | 行動 | 預期收益 |
|------|------|------|---------|
| P2 | **測試覆蓋率提升（56% → 70%+）** | 見下方覆蓋率任務清單 | 穩定性、CI 保護、開發信心 |

### 測試覆蓋率任務清單（P2）

> 詳細報告：[docs/TEST_COVERAGE.md](/docs/TEST_COVERAGE.md)

| 優先 | 模組 | 覆蓋率 | 未覆蓋行 | 狀態 |
|------|------|--------|----------|------|
| ~~P2-1~~ | ~~`perception/eval/evaluator.py`~~ | ~~40%~~ | ~~734~~ | **待實作** |
| ~~P2-2~~ | ~~`perception/dispatcher.py`~~ | ~~0%~~ | ~~116~~ | **✅ 已完成（22 tests, 94% coverage）** |
| ~~P2-3~~ | ~~`perception/eval/detection.py`~~ | ~~40%~~ | ~~336~~ | **✅ 已完成（39 tests, 49% coverage）** |
| ~~P2-4~~ | ~~`perception/eval/relink.py`~~ | ~~51%~~ | ~~228~~ | **✅ 已完成（73 tests, 88% coverage）** |
| ~~P2-5~~ | ~~`perception/drift_handler.py`~~ | ~~0%~~ | ~~71~~ | **✅ 已完成（43 tests, 100% coverage）** |
| ~~P2-6~~ | ~~`storage/redis_cache.py`~~ | ~~27%~~ | ~~106~~ | **✅ 已完成（53 tests, 99% coverage）** |
| ~~P2-7~~ | ~~`perception/calibrator.py`~~ | ~~0%~~ | ~~45~~ | **✅ 已完成（17 tests, 96% coverage）** |
| ~~P2-8~~ | ~~`perception/cropper.py`~~ | ~~23%~~ | ~~81~~ | **✅ 已完成（32 tests, 77% coverage）** |

**目標**：
- ✅ 短期 v1：`dispatcher.py` (94%)、`helpers.py` (91%) — **已完成**
- ✅ 短期 v2：`perception/eval/detection.py` (49%) — **已完成**
- ✅ 短期 v3：`perception/eval/relink.py` (88%) — **已完成**
- 🔄 短期 v4：`perception/eval/evaluator.py` (40%)
- ✅ 短期 v5：`storage/redis_cache.py` (99%) — **已完成**
- ✅ 短期 v6：`perception/calibrator.py` (96%) — **已完成**
- ✅ 短期 v7：`perception/cropper.py` (77%) — **已完成**
- ✅ 短期 v8：`perception/eval/quality.py` (100%) — **已完成**
- ✅ 短期 v9：`perception/eval/reporting.py` (93%) — **已完成**
- 📋 中期：`perception/eval/evaluator.py` (40%), `perception/eval/detection.py` (49%)
- 📋 長期：API 模組、media 模組、native 測試

**覆蓋率成長**：
- ✅ P2-1 + P2-2：新增 33 tests（dispatcher 22 + helpers 11）
  - `dispatcher.py` 從 0% → **94%**
  - `helpers.py` 從 18% → **91%**
  - 總覆蓋率 56% → **58%**
- ✅ P2-3：新增 39 tests
  - 涵蓋：`_decode_detector_boxes`, `expand_boxes_with_ankle_keypoints`, `match_keypoints_to_boxes`, `_tile_seam_mask_for_boxes`, `_get_detector_static_batch_size`
  - `detection.py` 從 40% → **49%**
- ✅ P2-4：新增 73 tests
  - 涵蓋：`__init__` 參數驗證、`_spatial_metrics`、`_measurement`、`_mahalanobis`、`_motion_box`、`_buffer_mean/consistency/sim`
  - `resolve()` 涵蓋：相似性匹配、年齡閾值拒絕、空間閾值拒絕、Mahalanobis 拒絕、margin 拒絕、生物特徵拒絕
  - 涵蓋：unified score mode、legacy joint score mode、appearance-first mode、quality filtering
  - `IdentityResolver.resolve_pass()` 涵蓋三種 API 路徑
  - `relink.py` 從 51% → **88%**
- ✅ P2-5：新增 43 tests
  - 涵蓋：`__init__`、`_get_dynamic_alpha`（所有 count 範圍 × 所有 DegradationLevel）、`calculate_drift`（新 track、相似/不同、所有降級級別）
  - 涵蓋：`filter_for_batch`（NORMAL/REDUCED/FAST_PATH/EMERGENCY 批次限制、優先級排序、面積 tiebreak）
  - 涵蓋：`update_history`（新 track 播種、EMA 更新、batch 多軌道、time.time 戳記、detach/clone、L2 normalization）
  - 涵蓋：`prune_expired_centroids`（過期/未過期、自訂 timeout、全部過期）
  - 涵蓋：`clear_history`（單軌道清除、不存在的 ID 安全處理、部分清除）
  - 涵蓋：完整生命週期整合測試（播種→更新→漂移檢查→清理）
  - `drift_handler.py` 從 0% → **100%**
  - 總覆蓋率 59% → **60%**
- ✅ P2-5：新增 43 tests
  - 涵蓋：`__init__`、`_get_dynamic_alpha`（所有 count 範圍 × 所有 DegradationLevel）、`calculate_drift`（新 track、相似/不同、所有降級級別）
  - 涵蓋：`filter_for_batch`（NORMAL/REDUCED/FAST_PATH/EMERGENCY 批次限制、優先級排序、面積 tiebreak）
  - 涵蓋：`update_history`（新 track 播種、EMA 更新、batch 多軌道、time.time 戳記、detach/clone、L2 normalization）
  - 涵蓋：`prune_expired_centroids`（過期/未過期、自訂 timeout、全部過期）
  - 涵蓋：`clear_history`（單軌道清除、不存在的 ID 安全處理、部分清除）
  - 涵蓋：完整生命週期整合測試（播種→更新→漂移檢查→清理）
  - `drift_handler.py` 從 0% → **100%**
  - `relink.py` 從 51% → **88%**
- ✅ P2-6：新增 53 tests（修復 10+ NameError + 重複 decorator bug）
  - 涵蓋：`MicroBatcher.__init__`、`add`（buffer 不足/max_size 觸發/超過）、`flush`（正常/含 timer/空 buffer）、timer 排程與取消、JSON 序列化
  - 涵蓋：`RedisCache.__init__`（default/url/env）、`connect`（建立 client/跳過已有/建立 stream group/BUSYGROUP 處理/其他錯誤）
  - 涵蓋：`add_to_stream`（xadd 呼叫/auto-connect）、`add_to_stream_batch`（pipeline/空 list/auto-connect）
  - 涵蓋：`read_stream_batch`（空 streams/解析事件/參數傳遞/auto-connect）
  - 涵蓋：`acknowledge`（xack 呼叫/空 list/auto-connect）
  - 涵蓋：`disconnect`（flush batchers/關閉 client/無 client/無 batchers）
  - 涵蓋：`cleanup_expired_objects`（低於閾值/超過閾值/無 keys/error handling/auto-connect）
  - 涵蓋：`update_object_track`（set key with TTL/auto-connect）、`get_active_objects`（返回 ID/跳過無效/空/auto-connect）
  - 涵蓋：`get_object_history`（返回資料/缺失/返回 None/auto-connect）
  - 涵蓋：`publish_event`（建立新 batcher/重用舊 batcher）
  - 涵蓋：attribute tests（queue/max_size/window_ms/stream_name/max_len）
  - `redis_cache.py` 從 27% → **99%**
  - 總覆蓋率 60% → **63%**
- ✅ P2-7：新增 17 tests
  - 涵蓋：`__init__`（cache_file/batch_size/input_shape/default 值/200 圖片上限/非 jpg 過濾）
  - 涵蓋：`get_batch_size`
  - 涵蓋：`read_calibration_cache` / `write_calibration_cache`（寫入+讀取/不存在的檔案/覆蓋寫入/建立檔案）
  - 涵蓋：`get_batch`（耗盡時返回 None/無有效圖片時返回 None/index 遞增）
  - 涵蓋：CUDA 記憶體分配、`current_batch` 初始值
  - `calibrator.py` 從 0% → **96%**
  - 總覆蓋率 63% → **64%**
- ✅ P2-9：新增 22 tests（`test_quality.py`）
  - 涵蓋：`compute_detection_quality_batch`（空 box/中心框/aspect 質量/center 質量/area 質量/多框/邊界檢查/自訂權重/CUDA/零面積/大幀）
  - 涵蓋：`compute_bank_quality_score`（基本計算/det 分數影響/IoU 影響/ideal aspect/unknown aspect/center bias/area ratio/自訂權重/邊界/負座標/大幀）
  - `quality.py` 從 11% → **100%**
  - 總覆蓋率 64% → **64%**
- ✅ P2-10：新增 30 tests（`test_reporting_extended.py`）
  - 涵蓋：`_print_stage_waterfall`（空 stages/零幀總時間/有 stages/排序/過濾零值/unaccounted 顯示與否/閾值行為）
  - 涵蓋：`print_overall_summary`（JSON profile 輸出/JSON 包含 stats/lazy reid candidates/lazy reid embeddings/lazy reid lines/birth CSV/GMC breakdown/post counts/breakdown stage/FPS summary 含/不含 latency）
  - 涵蓋：`print_sequence_summary`（tile diagnostics 打印/禁用/post counts/Native ReID breakdown/GMC/blank line/overall totals 累積/post counts 累積/GMC samples 累積）
  - `reporting.py` 從 49% → **93%**
  - 總覆蓋率 64% → **66%**
- ✅ P2-8：新增 32 tests
  - 涵蓋：`__init__`（default/custom output_size/padding/mode 驗證/invalid mode raise）
  - 涵蓋：`process`（None boxes/空 boxes/通道數保持/roi_align fallback）
  - 涵蓋：`_prepare_boxes`（tight 無 padding/expand 正方形/expand 正方形 mean/邊界 clamp/寬度 clamp/最小框尺寸/多框）
  - 涵蓋：`_fill_extra_with_mean`（非 square_mean 不修改/空 tensor/square_mean 填充）
  - 涵蓋：`process_parts`（None boxes/空 boxes/3x 輸出）
  - 涵蓋：`cpp_ptr`（raise when no C++）
  - `cropper.py` 從 23% → **77%**
  - 總覆蓋率 64% → **64%**

---

| 優先 | 項目 | 行動 | 預期收益 |
|------|------|------|---------|
| ~~P1~~ | ~~FPS anomaly 根因（match ≥ 0.73）~~ | **✅ 已結案（2026-05-11）**：全量實測（7 seq + profile-stages）確認 match=0.72 vs 0.73 差距僅 0.16ms/frame（82.3 vs 81.3 FPS），`bg_relink_wait=0.00ms`，**anomaly 已消失**。推測 2026-05-07 的 async_reid + pipeline_relink + fused letterbox 等優化將 frame total 從 ~8.4ms 壓至 12ms（其中 detect 佔 6ms），原本的臨界條件不再觸發。**`match=0.72` 安全邊界解除**。 | ✅ 已完成 |
| P3 | **Detector 訓練資料改善** | pred_h = 61.4% of gt_h，77% 近似 FP 有真實 GT；需補足腿/腳標注 | 根本解決 FN 問題；目前所有 score-gate 手段天花板已見 |

---

## Recent Ablation Conclusions（2026-05-10 ~ 05-11）

- **fuse_score_weight=0.4（2026-05-11）✅ 已設為 default（baseline preset）**
  - botsort-style：cost = 1 − IoU × (1 − 0.4 × score)，讓低信心 det 更難匹配
  - 7-seq SDP baseline preset：IDF1 +1.7pp → 51.4%，MOTA +1.6pp → 43.5%，FP **-13%**，Rcll -0.6pp，IDs -1
  - 0.4 為 Pareto 最優；≥0.75 會讓 IDs 翻倍

- **FP 模組 A：stage2_match_thresh（2026-05-11）❌ 語意反向，no-go for tuning**
  - `max_cost` 是上限；提高 = 更寬鬆 = FP 增加。0.5 在 fuse_score_weight=0.4 下已適當。
  - 參數已暴露（`--stage2-match-thresh`），保留為可調，但不調整 default。

- **FP 模組 B：birth_low_score_thresh（2026-05-11）微小增益，不設 default**
  - 出生分數 < thresh 的 track 需多一次 confirm；最佳 blst=0.28 → FP -63、IDs -3、IDF1 +0.1pp
  - 效益太小；`--birth-low-score-thresh`（default 0.0=off）保留為可調。

- **FP 模組 C：bank_weighted_mean（待測）**
  - 以 quality_score 加權 bank 樣本均值，降低低品質 embedding 影響
  - 只在 `--appearance-bank` 開啟時有效；ReID stack 測試待排。

- **P5-4 Per-sequence scene adaptive policy（2026-05-11）❌ NO-GO（narrow_bonus 策略）**
  - `SceneAdaptivePolicy` 分類邏輯正確（只有 MOT17-02 觸發 `crowded_narrow`：avg_aspect=1.97，最低）
  - 7-seq SDP speed preset：IDF1 -0.1pp，MOTA -0.2pp，IDs +5，FP +201，FN -107，FPS **-8**
  - MOT17-02 單序列：IDF1 -0.6pp（預期 +1.4pp）— 根因：speed preset `new_track_thresh=0.28` 低，bonus=0.05 把低品質框推過門檻
  - 舊版 +1.4pp 是在 `new_track_thresh=0.35` 下測得，不適用 speed preset
  - 框架保留 `--scene-adapt-enabled`（預設 off），classification 邏輯可供未來其他策略重用

- **P5-1 Multi-signal birth policy（2026-05-11）❌ NO-GO（2-run 確認）**
  - 2026-05-11 初測 7-seq：IDF1 ±0，IDs +12，FP +453，FN -530，FPS **-20**
  - 根因：Python-side O(K×C) IoU matching 每幀開銷；sub-threshold TP 比例太低
  - `--multi-birth-enabled` 保留（預設 off）
  - **2026-05-14 2-run 深度掃描（MOT17-02/10-SDP）：33 configs, 5-run avg 驗證**
    - 所有配置 MOTA 集中在 33.8–34.1%，run-to-run 波動 ±0.15–0.25%（在誤差內）
    - 最佳單策略：A `iou=0.90 ratio=1.03`（FPS 117.8，不降）、B `e=0.75 r=0.85`（FPS 92，-22%）
    - 組合策略（A+B, A+C）不如單獨使用（重疊處理同一批檢測）
    - **結論：後處理層面提升 <0.2%，不抵銷開銷。需從檢測層面下手。**
  - 新增 debug 工具：`--debug-birth-csv` + `label_boosted_birth_rows.py`（56 測試通過）
  - 856 boosted rows 全部 dropped（17.8% 真正 missed，需進一步分析）

- **P5-2 Stage 2 Quality Gate（2026-05-10）❌ NO-GO**
  - 與 `detection_quality_scaling=True` 完全重疊，零效果。只在 `--no-detection-quality-scaling` 時有意義。

- **P5-3 Consecutive-Frame Birth Gate（2026-05-10）❌ NO-GO**
  - Motion gate 有效過濾靜態 FP，但最佳結果統計中性（IDF1 +0.1pp）。sub-threshold TP 比例太低。

- **Tracklet Interpolation（2026-05-10）✅ 已設為 default**
  - `max_gap=20, min_track_len=5`：speed IDF1 +0.3pp，IDs -34，Rcll +2.0pp，FPS 不變

- **Pose-Guided Box Expansion（2026-05-10）❌ No-Go**
  - IDF1 不退但 FPS -60%（pose engine 每幀都跑），box 擴展引發 ID switches。detector training data 才是根本解。

- **FPS anomaly：match ≥ 0.73 時 FPS 驟降（2026-05-09）✅ 已結案（2026-05-11）**
  - 原症狀：`match=0.72` ~119 FPS；`match ≥ 0.73` FPS -28（+2.65ms），全 seq 受影響
  - Step 1（bench）：孤立 C++ tracker 無異常（1.3–1.5ms 全平）→ 根因在 Python
  - Step 2（實測）：7 seq `--profile-stages` 確認 0.72 vs 0.73 差距僅 0.16ms/frame，`bg_relink_wait=0.00ms`
  - **結論：anomaly 已被 2026-05-07 pipeline 優化消除（async_reid + fused letterbox）。`match=0.72` 安全邊界已解除。**

---

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖：[docs/TODO_history.md](/docs/TODO_history.md)
- Tracking base 與 relink sweep：[docs/experiments/tracking/fp_fn_recovery_and_gmc.md](/docs/experiments/tracking/fp_fn_recovery_and_gmc.md)
- ReID backbone refresh 歸檔：[docs/experiments/reid/semantic_relink_and_crop.md](/docs/experiments/reid/semantic_relink_and_crop.md)
