# Saccade 開發指南

本文件是目前的開發主入口。目標是讓開發者只看這一份，就能知道：

- 專案現在的主架構與主路徑
- 哪些文件 / 程式碼是 source of truth
- 現在優先在解的問題是什麼
- 改動時應更新哪裡、驗證什麼

---

## 1. 開發原則

本專案採用 **架構與合約驅動為主，TODO 驅動為輔**。

- 架構與合約先回答：
  - 哪一層負責什麼
  - 資料怎麼流
  - 哪些邊界與 fallback 不能破
  - 什麼指標代表成功
- TODO 只承接：
  - 當前真的要做的工作
  - 近期仍影響決策的 ablation 結論
  - 下一輪已排定的實驗與實作 backlog

一句話：
**先用架構與合約決定什麼值得做，再用 TODO 排順序。**

---

## 2. Source of Truth 順序

當文件彼此衝突時，請依下列優先順序判讀：

1. **目前主路徑程式碼**
   - `src/saccade/perception/`
   - `src/tracking/`
   - `src/`
   - `scripts/eval/mot17.py`
   - `src/saccade/perception/eval/runner.py`
2. **本文件 `DEVELOPMENT.md`**
   - 用於快速理解目前開發方向、模組責任與文件更新規則
3. **穩定架構 / 合約文件**
   - [docs/architecture.md](/docs/architecture.md:1)
   - [docs/api_spec.md](/docs/api_spec.md:1)
   - `docs/decisions/*.md`
4. **當前待辦與近期結論**
   - [docs/TODO.md](/docs/TODO.md:1)
5. **歷史脈絡**
   - [docs/TODO_history.md](/docs/TODO_history.md:1)
   - `docs/progress/`
   - `docs/experiments/`

`docs/TODO_history.md` 與 `docs/progress/` 是歷史與過程紀錄，不是目前行為合約的最高權威。

---

## 3. 系統現況摘要

Saccade 目前以 **MOT17-centered evaluation path** 為最活躍主線，核心是一條 GPU 優先的 perception/tracking/relink pipeline。

### 3.1 邏輯分層

- **L1 Perception**
  - YOLO + detection postprocess + GPU tracker
  - 主要位置：`src/saccade/perception/`, `src/tracking/`
- **L2 Appearance / ReID**
  - crop / embedding / appearance bank / semantic relink
  - 主要位置：`src/saccade/perception/tracking/`, `src/saccade/perception/eval/relink.py`
- **L3-L4 Streaming / Storage**
  - Redis / Chroma / microbatch
  - 主要位置：`src/saccade/storage/`, `src/saccade/pipeline/`
- **L5-L6 Cognition / Resource**
  - orchestrator / resource manager / entropy trigger
  - 主要位置：`src/saccade/cognition/`, `src/saccade/resource/`

### 3.2 目前主開發路徑

如果你要改 MOT / tracking / relink / ReID，先看這些檔案：

- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1)
- [src/saccade/perception/eval/relink.py](/src/saccade/perception/eval/relink.py:1)
- [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py:1)
- [src/tracking/tracker_gpu.cu](/src/tracking/tracker_gpu.cu:1)
- [src/saccade/perception/eval/detection.py](/src/saccade/perception/eval/detection.py:1)
- [src/saccade/perception/eval/gmc.py](/src/saccade/perception/eval/gmc.py:1)
- [scripts/eval/mot17.py](/scripts/eval/mot17.py:1)
- [scripts/eval/ablation_mot17.py](/scripts/eval/ablation_mot17.py:1)

---

## 4. 當前主合約

這些不是單純實作細節，而是改動時應優先守住的系統合約。

### 4.1 Pipeline 合約

- 主熱路徑優先走 GPU / native facade。
- 新增路徑若引入 CPU roundtrip，必須能說明必要性與影響。
- Python 可以負責 orchestration、評估、輸出整理，但不應輕易接回每幀大量資料面工作。

### 4.2 Tracking / Association 合約

- ambiguous case 不應只靠單一固定 threshold 決策。
- fallback 必須穩定且可解釋；目前最低保證仍是 `IoU-only fallback`。
- appearance、motion、quality 的責任邊界要清楚，避免同一訊號在多層重複加權、互相污染。

### 4.3 ReID / Reference 合約

- noisy reference 不應進 bank。
- low-quality observation 不應用與 clean observation 相同 accept 條件。
- ReID 是稀缺算力資源，不應無差別全幀觸發。

### 4.4 Documentation 合約

- 若改的是穩定行為或責任邊界，要更新架構 / ADR / API 文件。
- 若改的是近期工作方向與實驗排序，要更新 [docs/TODO.md](/docs/TODO.md:1)。
- 已完成且不再需要逐步追蹤的內容，移到 [docs/TODO_history.md](/docs/TODO_history.md:1)。

---

## 5. 目前最佳主路徑設定

截至 2026-05-01，MOT17 SDP 7 序列的 current documented default 為：

- `--cross-tile-merge`
- `--match-thresh 0.78`
- **A1 Unified Score (動態權重)**:
  - `--semantic-w-sim-base 0.80`
  - `--semantic-w-iou-base 0.34`
  - `--semantic-w-maha-base 0.31`
  - `--semantic-shift-ambiguity 0.34`
  - `--semantic-shift-lost-age 0.18`
- **A2 Quality Gate (品質門檻)**:
  - `--semantic-clean-score-threshold 0.65`
  - `--semantic-strict-sim-threshold 0.74`
  - `--appearance-bank-high-quality-min-score 0.75`

近期主結論：

- `A1 + A2` 組合已達到 **IDF1 46.3%**，成功追平硬閾值紀錄，且在擁擠場景 (Ambiguity) 與長遺失 (Lost Age) 下具有更好的自動適應能力。
- `strict_sim_threshold` (0.74) 低於預設的 0.91，說明對於低品質觀測，適度放寬外觀門檻能有效減少 Fragmentations (FN)，而錯誤匹配則由 A1 的動態權重與 A2 的高品質參考庫控住。
- 下一步主軸是 **Track-Level Budgeted ReID (A3)**，控住運算資源同時維持此高 IDF1 水位。

---

## 6. 目前最重要的開發方向

如果沒有更高優先需求，請優先朝這些方向開發：

### P0：2026 高 MOTA 整合：品質感知關聯 (SelectMOT)
- **位置**：`src/tracking/tracker_gpu.cu`、`include/tracking/pipeline.hpp`
- **目標**：實作 ADR 017 規劃的品質加權。在 `Phase 0` 根據 YOLO Confidence 與 bbox 幾何變化動態產生 Quality Score，並在 `Phase 1` 的 Dense Sinkhorn 中作為邊際分佈的先驗權重，抑制遮擋框的競價能力。這是目前最能立即提升擁擠場景 IDF1/MOTA 的方向。

### P1：2026 高 MOTA 整合：純 GPU 均勻相機補償 (UCMCTrack)
- **位置**：`include/tracking/gmc.hpp`、`src/tracking/kalman_gpu.cuh`
- **目標**：實作 ADR 017 規劃的 Uniform CMC 與 2D MMD。將目前依賴 OpenCV 的稀疏光流替換為純 CUDA 實作，並在計算卡爾曼距離時引入地平面透視變換，達成真正的全管線 Zero-Copy。

### ~~P0：Pre-hoc Embedding Quality (LaSt-ViT CUDA Kernel)~~ — CLOSED No-Go (2026-05-02)

實作已完成（CUDA kernels + C++ API + PyBind11），但 MOT17 驗證結果僅 +0.09pp IDF1（低於 +1.0pp 門檻）。
根本限制：SigLIP2 未以 LaSt-ViT 目標訓練，`last_hidden_state` 穩定分數無前景/背景區分力。
API 保留供未來使用：`FeatureExtractor::extract_with_stability()`，`mot17.py --last-vit-embed`。
詳見 `docs/experiments/reid/last_vit_integration_analysis.md` §10。

### ~~P0：Reference Quality + False-Accept Filtering~~ — DONE (2026-05-01)

bank 高品質 tier + inject gate + false-accept filter 已實作。詳見 `docs/TODO.md` A2 Phase 3 結論。

### ~~P1：Unified Association / Relink Scoring~~ — DONE (2026-05-01)

`w_sim_base/w_iou_base/w_maha_base/shift_ambiguity/shift_lost_age` 動態調整已整合（A1）。

### ~~P1：Track-Level / Budgeted ReID~~ — DONE (2026-05-01)

`--reid-budget 0.2`（Dynamic Ratio 0.2）已成為預設最佳配置（A3，+24% FPS）。

### P2：GMC Quality-Aware 補強

- 位置：`src/saccade/perception/eval/gmc.py`
- 目標：
  - 避免 foreground 主導 GMC
  - 把 GMC quality 回饋到 trigger / tracking 決策

### P2：Post-Merge V2 與 Detection/Bank Quality Scoring

- 位置：
  - `src/saccade/perception/eval/runner.py`
  - `src/saccade/perception/eval/detection.py`
  - `src/saccade/perception/tracking/tracker_gpu.py`

---

## 7. 檔案對應指南

### 7.1 你要改 tracking / association

先看：

- `src/tracking/tracker_gpu.cu`
- `include/tracking/tracker_gpu.hpp`
- `src/saccade/perception/tracking/tracker_gpu.py`

常見改動：

- cost matrix
- gating
- tracker state lifecycle
- appearance bank 與 clean embedding flags

### 7.2 你要改 semantic relink / identity resolve

先看：

- `src/saccade/perception/eval/relink.py`
- `src/saccade/perception/eval/runner.py`
- `src/tracking/tracker_gpu_python.cpp`

常見改動：

- semantic threshold logic
- reciprocal / dynamic margin
- quality gate
- lifecycle merge / resolve pass

### 7.3 你要改 detection postprocess / tile merge / suspect logic

先看：

- `src/saccade/perception/eval/detection.py`
- `src/tracking/tracker_gpu.cu`
- `include/tracking/pipeline.hpp`

### 7.4 你要改 evaluation / ablation

先看：

- `scripts/eval/mot17.py`
- `scripts/eval/ablation_mot17.py`
- `src/saccade/perception/eval/runner.py`
- `scripts/eval/summarize_ablation_mot17.py`

---

## 8. 開發流程

### 8.1 改動前

先判斷這次改動屬於哪一類：

- **架構 / 合約改動**
  - 更新 `docs/decisions/`、`docs/architecture.md`、`docs/api_spec.md`
- **近期方向 / 實驗排序改動**
  - 更新 [docs/TODO.md](/docs/TODO.md:1)
- **單純實作細節或 bug fix**
  - 至少在 commit / PR 記錄 why

### 8.2 改動中

- 優先維持主熱路徑的 GPU-first 原則
- 避免未說明的 `.cpu()` / `numpy()` / host materialization
- 若引入 fallback，需明確定義何時觸發與退回何種行為

### 8.3 改動後

至少做以下其中之一：

- 單元測試
- parity test
- MOT17 short eval
- 對應 ablation / benchmark

若改動會影響 default path，應補：

- `IDs / IDF1 / MOTA / FP / FN / FPS` 的比較
- 哪些 sequence 改善，哪些 sequence 退化
- 是否屬於 run-to-run noise 還是系統性差異

---

## 9. 常用驗證

依改動範圍選擇：

```bash
uv run mypy .
uv run pytest
scripts/test_native.sh
```

若是 MOT / tracking / relink 相關，優先補：

```bash
uv run python scripts/eval/mot17.py ...
uv run python scripts/eval/ablation_mot17.py ...
```

原則：

- 不要只看單次 `IDs`
- 要區分單序列現象與 7-seq aggregate
- 若差異接近既有 noise，不能直接宣稱演算法勝出

---

## 10. Scripts 與 Tests

- 主 workflow 腳本集中在 `scripts/eval/`
- 核心入口仍是：
  - `scripts/eval/mot17.py`
  - `scripts/eval/ablation_mot17.py`
- 效能量測與壓力測試集中在 `scripts/benchmarks/`
- native build / coverage helpers 集中在 `scripts/native/`
- 本地串流與服務控制集中在 `scripts/ops/`
- 一般 Python 測試放在 `tests/test_*.py`
- native tests 放在 `tests/native/`
- benchmarks 放在 `tests/benchmarks/bench_*.py`
- 詳細規範、命名、目錄分工與最低驗證要求，見 [docs/TESTING.md](/docs/TESTING.md:1)

---

## 11. 文件更新規則

### 主 TODO

[docs/TODO.md](/docs/TODO.md:1) 只保留：

- 目前真的還要做的事項
- 近期仍影響決策的 ablation 結論
- 下一輪已排定的 backlog

### 歷史 TODO

[docs/TODO_history.md](/docs/TODO_history.md:1) 保留：

- 已完成項
- 已收斂並放棄的方向
- 舊路線圖與長篇過程紀錄

### 歸檔原則

- 主 TODO 保留高訊號摘要
- 細節、過程、舊掃描結果移入 history
- 某方向若重新啟動，再從 history 摘回主 TODO，不要讓主 TODO 長期堆積舊脈絡

---

## 12. 補充入口

若需要更深背景，再查這些文件：

- [docs/architecture.md](/docs/architecture.md:1)
- [docs/api_spec.md](/docs/api_spec.md:1)
- [docs/TODO.md](/docs/TODO.md:1)
- [docs/decisions/016-rerank-phase3-reference-quality.md](/docs/decisions/016-rerank-phase3-reference-quality.md:1)
- [docs/experiments/reid/dynamic_trigger.md](/docs/experiments/reid/dynamic_trigger.md:1)
- [docs/layers/gpubytetracker_deep_dive.md](/docs/layers/gpubytetracker_deep_dive.md:1)

如果你發現這份文件與主路徑程式碼不一致，先修這份文件，再決定是否需要補 ADR / TODO / history。
