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

### 1.1 代碼工程規範

- **單文件長度限制**：單一原始碼文件（.py, .cpp, .cu, .hpp 等）原則上 **不超過 1000 行**。
  - 若超過此限制，應優先考慮將邏輯拆分至多個子模組、組件或更小的類別中。
  - 此規範旨在提升代碼可讀性、降低維護成本與合併衝突風險，並強化關注點分離 (Separation of Concerns)。

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
   - [docs/architecture.md](docs/architecture.md)
   - [docs/PIPELINE_REFERENCE.md](docs/PIPELINE_REFERENCE.md)
   - [docs/api_spec.md](docs/api_spec.md)
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

### 4.5 Detection / Tiling 合約

- `native_960` 現在是 detector path 的 control；所有 tiled 修補都應先和它對照。
- tiled path 若引入 seam duplicate / truncation 汙染，應盡量在進 tracker 前消化，不要把不乾淨的 observation 留給 association / relink / bank update。
- `cross-tile merge` 是 tiled path 的必要補救，不代表 tiled 已等價於 `native_960`。

---

## 5. 目前最佳主路徑設定

截至 2026-05-06，MOT17 SDP 7 序列的 current documented live default 為：

- `--gmc --gmc-mode gpu`
- `--reid-trigger-mode event_any`
- `--cross-tile-merge`
- `--match-thresh 0.78`
- `--semantic-threshold 0.91`
- `--detection-quality-scaling`
- `--reid-budget 0.2`
- `--new-track-thresh 0.45`
- D2-C CUDA tentative isolation（已納入 default）

最近一輪 SDP 7 序列結果（2026-05-06，`mot17.py --detector SDP`）：

- **IDF1 47.9%**
- **MOTA 40.7%**
- **IDs 648**
- **FP 10,821**
- **FN 55,103**
- **Recall 50.9%**
- **Eval FPS 51.5**（mean 19.4ms/frame）

目前重要補充：

- `new-track-thresh 0.55` 的舊方向已不再是當前代碼默認；CLI 與 runner fallback 都已對齊為 `0.45`。
- `native_960` 在 `MOT17-04-SDP / MOT17-10-SDP` 上明顯優於 `960p_2x2 tiled`，因此 tiled 現在被視為需要持續診斷的流程風險點，而不是更健康的 detector path。

---

## 6. 目前最重要的開發方向

如果沒有更高優先需求，請優先朝這些方向開發：

> 已完成與已結案項目見 [docs/TODO.md](docs/TODO.md) 與 [docs/TODO_history.md](docs/TODO_history.md)。

### P1：窄人低分框保留（7-seq 驗證待完成）

- 根因已定位：`MOT17-02-SDP` 的 FN 主要來自 `post_filter` 前後對窄人低分框的淘汰，不是 relink 問題。
- 目前候選設定（單序列最佳）：
  - `--narrow-person-score-bonus 0.05`
  - `--narrow-person-max-width-ratio 0.015`
  - `--narrow-person-min-aspect 2.4`
- 單序列結果：`IDF1 32.2 / MOTA 27.5 / FP 1757 / FN 11547 / IDs 164`（vs baseline `IDF1 30.8 / MOTA 26.5`）
- **下一步：跑 7-seq SDP 驗證，確認無 regression 後考慮納入 default。**

### P2：native_960 Tracker Threshold 重評

- `native_960` 已確認為主 baseline（tiled 停止調參）。
- 目前 `--match-thresh 0.78 / --new-track-thresh 0.45` 是在舊流程下調出的參數，尚未以 `native_960` 系統性重掃。
- 目標：在 `native_960` 上突破目前 IDF1 47.9% / MOTA 40.7% 的 7-seq 上限。
- 入口：`scripts/eval/ablation_mot17.py`，先跑 local metric，確認 signal 再做全序列。

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
- `scripts/eval/mot17.py`

常見改動：

- detector routing（`native_960` vs tiled）
- tile diagnostics
- seam-aware duplicate merge / representative box fusion

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

### 8.4 實驗入口與對比層級

MOT / tracking / relink 類改動，不要把所有驗證都壓成「最後 `IDF1 / MOTA` 有沒有變好」。
現行做法是分三層：

1. **baseline**
   - 固定對照面：`./scripts/eval/module_benchmark.sh --mode all`
   - 目前標準 baseline：`results/module_benchmark/baseline_native_960`
2. **module-local**
   - 先看你改的那個模組自己的主指標
   - 例如：
   - preprocess：`ingest_preprocess`
   - detection：`detect`、`raw_boxes -> after_merge`
   - postprocess：`post_filter` / `post_nms` / `post_merge`
   - reid：`reid_budget`、`reid_extract`
   - lifecycle / relink：`relink_write`
3. **promotion validate**
   - 只有當 local signal 明確，才升級去看整體 `IDF1 / MOTA / FP / FN / IDs / FPS`

判讀規則：

- `local improved`, `downstream neutral`, `e2e neutral`
  - 可以先標記為局部優化，不必硬說整體演算法勝出
- `local improved`, `downstream worse`
  - 視為 regression，不保留
- `local better`, `downstream better`, `e2e better`
  - 才是強候選，值得進一步推成 default

實驗紀錄原則：

- 每次候選實驗都應明確寫出：
  - 跟哪個 baseline 比
  - 改的是哪個 module / knob
  - local metric 是什麼
  - downstream metric 是什麼
  - 是否值得升級成 promotion validate

什麼時候要打開 `docs/PIPELINE_REFERENCE.md`：

- 你要知道某個 pipeline module 現在對應哪個 metric
- 你要判讀某個 delta 到底算改善、noise 還是 regression
- 你要確認某個 module 是 `directly measured`、`indirectly measured`，還是目前仍是 gap
- 你要做 module-local experiment，而不是只跑整體 `mot17.py` 結果

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
./scripts/eval/module_benchmark.sh --mode all
uv run python scripts/eval/mot17.py ...
uv run python scripts/eval/ablation_mot17.py ...
```

原則：

- 不要只看單次 `IDs`
- 要區分單序列現象與 7-seq aggregate
- 若差異接近既有 noise，不能直接宣稱演算法勝出
- 先看 local metric，再看 downstream，再看 end-to-end

---

## 10. Scripts 與 Tests

- 主 workflow 腳本集中在 `scripts/eval/`
- 核心入口仍是：
  - `scripts/eval/module_benchmark.sh`
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

- [docs/architecture.md](docs/architecture.md)
- [docs/PIPELINE_REFERENCE.md](docs/PIPELINE_REFERENCE.md)
- [docs/api_spec.md](docs/api_spec.md)
- [docs/TODO.md](docs/TODO.md)
- [docs/decisions/016-rerank-phase3-reference-quality.md](/docs/decisions/016-rerank-phase3-reference-quality.md:1)
- [docs/experiments/reid/dynamic_trigger.md](/docs/experiments/reid/dynamic_trigger.md:1)
- [docs/layers/gpubytetracker_deep_dive.md](/docs/layers/gpubytetracker_deep_dive.md:1)

如果你發現這份文件與主路徑程式碼不一致，先修這份文件，再決定是否需要補 ADR / TODO / history。
