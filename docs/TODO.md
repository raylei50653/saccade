# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦、近期 ablation 結論與下一步方向。已完成項、設計規範與 C++ 路線圖已移至 [docs/TODO_history.md](/docs/TODO_history.md:1)。

---

## 歸檔標準

- 主 TODO 只保留三類內容：
  - 目前真的還要做的事項
  - 近期仍會影響決策的 ablation 結論
  - 下一輪已排定的實驗 / 實作 backlog
- 內容應移入 [docs/TODO_history.md](/docs/TODO_history.md:1) 的情況：
  - 已完成，且後續不再需要逐步追蹤
  - 已收斂並明確放棄，不再作為近期 default 候選
  - 已被新方向取代，只需保留背景與結論
  - 屬於長篇實作過程、舊路線圖或階段性 milestone，而不是當前待辦
- 歸檔時原則：
  - 主 TODO 保留高訊號摘要與最終結論
  - 細節、過程、舊參數掃描與已結案子項移入 history
  - 若某方向之後重新啟動，再從 history 摘回主 TODO，而不是在主 TODO 長期保留已結案脈絡

---

## Current TODO

- [x] **Rerank Phase 3：Reference Quality + False-Accept Filtering（已完成 A2 驗證）**
  - 已完成（2026-05-01）：
    - `TrackAppearanceBank` 高品質過濾實作完成。
    - `PythonSemanticRelinker` 與 C++ `SemanticRelinkerCpp` 品質門檻同步完成。
    - A2 Optuna 掃描完成（30 trials）。
  - 最終結論：
    - 最佳參數：`clean_score_threshold=0.65 / strict_sim_threshold=0.74 / high_quality_min_score=0.75`。
    - 效果：IDF1 達到 **46.29%**，成功將 Unified Score 提升至先前硬閾值 baseline 的水位。
    - 發現：低品質觀測需要較鬆的相似度門檻（0.74 vs 0.91）來維持連貫性，而誤配由高品質參考庫機制抑制。

- [x] **Online Association / Semantic Relink 統一打分**
  - 已完成（2026-05-01）：
    - 在 A1 Ablation 中完成，把 `appearance + motion + quality` 收斂成單一 calibrated score。
    - track age / lost age / candidate ambiguity / observation quality 進入權重。

- [x] **Dynamic ReID Trigger V2: Track-Level / Budgeted ReID**
  - 背景：目前 `DynamicReIDController` 已有 `score_ema` 路徑，但仍是 frame-level heuristic，且有固定 `MIN_REID_GAP`。
  - 參考設計：[docs/experiments/reid/dynamic_trigger.md](/docs/experiments/reid/dynamic_trigger.md:1)
  - 主要位置：
    - [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py:76)
    - [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:2075)
  - 已完成（2026-05-01）：
    - 實作 `DynamicReIDController.get_priorities()` 與 `get_last_boxes()`。
    - 在 `runner.py` 整合 `_budget_reid_candidates` 優先級排序。
    - 支援 `--reid-budget` 固定 budget 限制與 track-level prioritization。
  - 結論：
    - ReID 資源可根據 track risk (new/lost/unstable) 進行優先分配。
    - 支援與 GMC 結合的空間優先級預測。

- [ ] **GMC Quality-Aware / Background-Aware 補強**
  - 問題：目前 GMC 仍是 sparse LK + affine，對 crowd / foreground-dominant scene 偏脆弱。
  - 主要位置：[src/saccade/perception/eval/gmc.py](/src/saccade/perception/eval/gmc.py:1)
  - 下一步：
    - 只用 background feature points 算 GMC，避免人群主導 warp。
    - 為 GMC 增加品質評分與 fallback policy（affine / translation / identity）。
    - 把 GMC quality 回饋給 tracker 與 dynamic ReID trigger。

- [ ] **Post-Merge Tracklet Stitching V2**
  - 問題：目前 post-merge 仍以 `spatial + motion + time + direction` 為主，appearance 大多只是 gate，不是完整 cost。
  - 主要位置：[src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1144)
  - 下一步：
    - 把 appearance similarity、reference consistency、gap uncertainty 納入 merge cost。
    - 避免單靠 motion 在長 gap / crowded turn / camera motion case 誤合併。
    - 保持 offline merge 為 optional cleanup，不反過來污染 default online path 結論。

- [ ] **Detection / Reference Quality Scoring**
  - 問題：目前 detection quality 與 bank sample quality 仍大量依賴 fixed thresholds。
  - 主要位置：
    - [src/saccade/perception/eval/detection.py](/src/saccade/perception/eval/detection.py:41)
    - [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py:421)
  - 下一步：
    - 將 `score / truncation / aspect / area / center bias / motion smoothness` 收斂成 quality score，而不是單純 keep-or-drop。
    - 讓 appearance bank sample ranking 使用更完整的 quality score，降低 reference contamination。

- [ ] **Low-Priority Code TODO**
  - [src/saccade/perception/entropy.py](/src/saccade/perception/entropy.py:43)
  - `entropy` 目前仍是 placeholder；此項不阻塞 MOT17 default path，但之後若要做事件驅動 cognition，應補完實際 Shannon entropy / object density 指標。

## Recent Ablation Conclusions

- **MOT17 current default remains unchanged（2026-04-30）**
  - 最佳設定仍為：
    - `--cross-tile-merge`
    - `--match-thresh 0.78`
    - `--semantic-threshold 0.91`
  - SDP 7 序列最佳結果：
    - `IDF1 44.9% -> 46.3% (+1.4pp)`
    - `MOTA 34.2% -> 35.1% (+0.9pp)`
    - `IDs 962 -> 857 (-105)`

- **MOT17-a：semantic ambiguity suppression**
  - 位置：[src/saccade/perception/eval/relink.py](/src/saccade/perception/eval/relink.py:1)
  - 已完成：
    - `iou_weight`
    - `mahalanobis_weight`
    - `dynamic_margin_crowd`
    - `dynamic_margin_age`
  - 結論：
    - `iou_weight=0.10 + thr=0.92` 有局部增益，但整體仍低於 C++ `thr=0.91` default。
    - `crowd/age margin` 在 `thr=0.91` 時系統性拉低 IDF1，不適合 default。
    - C++ vs Python 差異已確認主要來自先前 auction atomics 的 run-to-run noise，不是 relinker 演算法 bug。

- **MOT17-b：cross-tile merge confidence propagation**
  - 位置：
    - [src/saccade/perception/eval/detection.py](/src/saccade/perception/eval/detection.py:41)
    - [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:2031)
  - 已驗證：
    - `cross_tile_score_penalty=0.95 / 0.90` 皆劣化。
  - 結論：
    - 目前 CUDA path 無法可靠暴露有效 `merge_counts` 給 default path。
    - 此方向暫停，不列入近期 default tuning。

- **MOT17-c：semantic memory / rerank default validation**
  - 位置：
    - [src/saccade/perception/eval/relink.py](/src/saccade/perception/eval/relink.py:1)
    - [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1386)
  - 已驗證：
    - `buf=2 top2_mean` 雖有整體 `IDs` 改善，但在 `MOT17-02-SDP` 出現 `+93 IDs` 嚴重回歸。
    - 所有 tested buffer/rerank variants 的整體 IDF1 仍低於 C++ default `46.3%`。
  - 結論：
    - 不升級 default。
    - 此方向暫停，除非後續 reference quality gate 改變了 reference contamination 分布。

- **Rerank Phase 2（Bank Inject + Reciprocal Margin）**
  - 結論：
    - `buf=1 EMA` 基線上，`inject + margin=0.05` 是 stripped reference path 的最佳組合。
    - 但在 current documented default base 上，沒有任何 reciprocal margin 能乾淨取代 `thr=0.91`。
    - 下一步不再做 reciprocal margin default tuning，正式轉向 Phase 3 reference-quality / false-accept filtering。

## New Ablation Backlog

- [x] **A1：Unified Association Score Ablation**
  - 已完成（2026-05-01）：
    - 替換了 `tracker_gpu.cu` 與 `relink.py` 中的硬性閾值，改用 `w_sim_base`, `w_iou_base`, `w_maha_base` 與動態調整。
    - 加入了 `shift_ambiguity` 與 `shift_lost_age`。
    - 於 `ablation_mot17.py` 整合 Optuna 進行貝氏最佳化，支援 `--optuna a1`。
  - 待做（Optuna 掃描）：
    - 執行 `--optuna a1` 實驗，尋找最佳的權重組合。
  - 預期：比單純調 `semantic_threshold` 更有機會穩定改善 default。

- [x] **A7：Quality-Aware Sinkhorn (SelectMOT Integration)**
  - 參考：ADR 017
  - **狀態：已完成 (2026-05-02)**
  - **最終方案**：`v2_aspect_only_soft` (對極端長寬比進行機率衰減)。
  - **量化結果**：
    - IDs: **738 -> 722 (-2.2%)**
    - MOTA: **32.6% (持平)**
    - FN/Recall: 基本無損。
  - **結論**：成功在不犧牲 Recall 的情況下抑制了遮擋引起的錯誤關聯。已永久整合至 `src/tracking/tracker_gpu.cu`。

- [x] **A8：Uniform CMC & 2D MMD (UCMCTrack Integration)**
  - 參考：ADR 017
  - **狀態：已完成 (2026-05-02)**
  - **實作內容**：
    - **純 GPU GMC**：實作了基於 `cuFFT` 的相位相關 (Phase Correlation) 演算法，取代 OpenCV LK，達成 100% Zero-Copy。
    - **2D MMD 基礎建設**：在 `tracker_gpu.cu` 中實作了透視變換投影，支援將 Bbox 底部中點映射至地平面進行距離計算。
  - **效益**：
    - 移除 D2H 同步瓶頸，FPS 提升約 **5~10%**。
    - 支援透過 `--homography-root` 載入序列專屬單應矩陣。

- [ ] **A2：Reference Quality Gate Sweep**
  - 比較：
    - `clean_score_threshold`
    - `clean_margin_ratio`
    - `clean_min_aspect / clean_max_aspect`
    - `strict_sim_threshold`（建議初始值 0.60，範圍 0.55–0.65）
  - 目標：
    - 找到可穩定降低 false accept 的 default 組合。
    - 驗證 `IDs / FP` 改善是否以可接受的 `FN` 成本換來。
  - [x] **A2-L：Pre-hoc Embedding Quality (LaSt-ViT CUDA Kernel 整合) — CLOSED No-Go (2026-05-02)**
    - **結論：** SigLIP2 未以 LaSt-ViT 目標訓練，`last_hidden_state` 的前景/背景穩定性無法區分（stab ~0.12 均勻，p=0.386），inference-time post-processing 的 IDF1 增益僅 **+0.09pp**（MOT17-04-SDP 全序列），遠低於 +1.0pp go/no-go 門檻。背景預處理（Gaussian mask / mean-fill）亦無改善，反而使 image_embeds baseline gap 退步。
    - **落地產出（保留，可供未來參考）：**
      - CUDA kernel：`preprocessor_gpu.cu` — `launch_last_vit_refinement`（cuFFT R2C/C2R + 5 kernels）
      - C++ API：`FeatureExtractor::extract_with_stability()` + PyBind11 binding
      - cuFFT 正規化 bug fix（`build_gauss_weights` ÷C）：Phase 2A 18/18 tests passed
      - Phase 2B：V1 (per-patch Top-K) > V4 > V3 > V2 (paper strict voting)，V2 FG/BG p 反向 (0.9963)
      - Phase 2C：R0=44.85% → R1=44.94% (+0.09pp)；背景 mask sweep 7 種全部不如 none
      - 驗證腳本：`scripts/eval/validate_last_vit_phase0.py`（支援 `--variant-compare`、`--bg-mask-sweep`）
      - 測試：`tests/test_last_vit_cpp_vs_python.py`（18 tests）
    - **根本限制：** LaSt-ViT 的增益來自訓練期骨幹校正，非 inference formula。若要啟用需重訓 SigLIP2 with LaSt-ViT 聚合層——超出 A2-L 範疇。
    - 詳細分析：`docs/experiments/reid/last_vit_integration_analysis.md` §9

- [x] **A3：Track-Level Budgeted ReID Sweep**
  - 比較：
    - 現行 frame-level `DynamicReIDController`
    - track-level candidate prioritization
    - 固定 budget 與動態比例（ratio）
  - 重點 sequence：
    - `MOT17-04-SDP`
    - `MOT17-02-SDP`
    - `MOT17-10-SDP`
  - 結論（2026-05-01）：
    - **Dynamic Ratio 0.2 (`--reid-budget 0.2`)** 是目前最強配置。
    - 在 300 幀基準測試中，20% 預算帶來了 **+24% FPS**，同時保持（甚至微幅提升）了 IDF1。
    - 證實了「高風險優先級分配」能有效抗拒背景雜訊，達到「少即是多」的效果。

- [ ] **A4：GMC Quality / Background Mask Ablation**
  - 比較：
    - 現行 sparse LK affine
    - background-only points
    - affine 失敗 fallback translation / identity
    - trigger 中納入 GMC quality guard
  - 目標：降低 moving-camera 下的 false instability 與 over-trigger。

- [ ] **A5：Post-Merge V2 Cost Ablation**
  - 比較：
    - 現行 motion-heavy merge
    - 加入 appearance similarity
    - 加入 reference consistency / gap uncertainty
  - 目標：確認 offline cleanup 是否仍能提供額外 `IDs` 改善，而不污染 online default 結論。

- [ ] **A6：Detection / Bank Sample Quality Scoring Ablation**
  - 比較：
    - 現行 threshold-based keep / suspect
    - quality score-based sample ranking
  - 目標：確認 reference contamination 是否能在 bank 入口就被壓低。

## Completed But Still Important

- [x] **Pipeline GPU 化 85%+**
  - `M1`、`M2`、`M3`、`M3.5`、`M4` 已完成。
  - 結論：
    - runner 熱路徑的 identity resolve 已整合為 C++ pass。
    - deterministic assignment / GMC stream / GPU-native preprocess 已落地。
    - 下一步不再是大規模 pipeline plumbing，而是回到演算法品質。

## Active Context

- 目前 `siglip2` 仍是最高 IDF1 ceiling。
- `transreid` 與 `osnet` 已完成對比，但都未超過當前 `siglip2` base。
- OSNet engine 已建立，可用 `uv run python scripts/model/build_osnet.py` 重建。
- Phase 1 multi-sample rerank 結果已存 `results/ablation_rerank/`，可用 `--skip-run` 重新讀取。
- Phase 2 ablation script 已修正，直接 `--fast` 執行即可。

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖： [docs/TODO_history.md](/docs/TODO_history.md:1)
- Tracking base 與 relink sweep： [docs/experiments/tracking/fp_fn_recovery_and_gmc.md](/docs/experiments/tracking/fp_fn_recovery_and_gmc.md:1)
- ReID backbone refresh 歸檔： [docs/experiments/reid/semantic_relink_and_crop.md](/docs/experiments/reid/semantic_relink_and_crop.md:252)

---

## 🎯 專案里程碑

最後更新：2026-04-30

**MOT17-a/b/c ablation + confound + 診斷全部完成（2026-04-30）**：
- MOT17-a：確認無 C++ relinker 演算法 bug；`thr=0.92` 的暴衝主要來自 run-to-run noise，不是系統性問題。
- MOT17-b：score penalty 方案放棄（default path 無可靠 `merge_counts`）。
- MOT17-c：`buf=2 top2_mean` 因 `MOT17-02` 嚴重回歸而取消 default 候選資格。
- 當前最佳配置維持：C++ relinker + `match=0.78` + `thr=0.91`。

**Pipeline GPU 化主線完成（2026-04-30）**：
- identity resolve 熱路徑已整合。
- deterministic assignment 已落地。
- GMC 同步成本已下降。
- 主 TODO 已轉回 reference quality / false-accept filtering 與下一輪 ablation。
