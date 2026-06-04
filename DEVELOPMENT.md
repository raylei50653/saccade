# Saccade 開發指南

開發者入口。看這一份就能知道架構責任、合約邊界、當前設定、改動驗證方式。

---

## 1. 開發原則

- 架構與合約決定什麼值得做，TODO 排優先順序。
- 單一原始碼檔案（.py/.cpp/.cu/.hpp）原則上不超過 **1000 行**；超過則拆模組。
- 主熱路徑優先走 GPU / native facade；不應輕易引入未說明的 CPU roundtrip。

---

## 2. Source of Truth 順序

當文件彼此衝突時，依下列優先順序判讀：

1. **主路徑程式碼**（`src/saccade/perception/`、`src/tracking/`、`scripts/eval/mot17.py`）
2. **本文件 DEVELOPMENT.md**
3. **穩定架構 / 合約文件**（`docs/architecture/README.md`、`docs/reference/PIPELINE_REFERENCE.md`、`docs/reference/api_spec.md`、`docs/decisions/`）
4. **當前待辦**（`docs/TODO.md`）
5. **歷史脈絡**（`docs/TODO_history.md`、`docs/research/`）

---

## 3. 系統分層

| 層 | 責任 | 主要路徑 |
|---|---|---|
| **L1 Perception** | YOLO + postprocess + GPU tracker | `src/saccade/perception/`, `src/tracking/` |
| **L2 Appearance / ReID** | crop / embedding / bank / semantic relink | `src/saccade/perception/tracking/`, `eval/relink.py` |
| **L3–L4 Streaming / Storage** | Redis / Chroma / microbatch | `src/saccade/storage/`, `src/saccade/pipeline/` |
| **L5–L6 Cognition / Resource** | orchestrator / resource manager | `src/saccade/cognition/`, `src/saccade/resource/` |

MOT / tracking / relink 主線開發從這些檔案出發：

- `src/saccade/perception/eval/evaluator.py`
- `src/saccade/perception/eval/relink.py`
- `src/saccade/perception/tracking/tracker_gpu.py`
- `src/tracking/tracker_gpu.cu`
- `src/saccade/perception/eval/detection.py`
- `scripts/eval/mot17.py`

---

## 4. 系統合約（改動時應守住）

### 4.1 Pipeline 合約
- 主熱路徑優先走 GPU / native。新路徑引入 CPU roundtrip 必須說明必要性。
- Python 負責 orchestration / 評估 / 整理；不應接回每幀大量資料面工作。

### 4.2 Tracking / Association 合約
- ambiguous case 不靠單一固定 threshold；fallback 必須穩定且可解釋（最低保證：IoU-only）。
- appearance、motion、quality 的責任邊界要清楚，避免同一訊號多層重複加權。

### 4.3 ReID / Reference 合約
- noisy reference 不進 bank；low-quality observation 不用與 clean 相同 accept 條件。
- ReID 是稀缺算力資源，不無差別全幀觸發。

### 4.4 Detection / Tiling 合約
- `native_960` 是 detector path 主控；tiled path 改動先對照 `native_960`。
- seam duplicate / truncation 汙染在進 tracker 前消化；不把不乾淨的 observation 留給 association / bank。

### 4.5 Documentation 合約
- 穩定行為 / 責任邊界改動 → 更新架構 / ADR / API 文件。
- 模組待辦 / 實驗排序 → 更新該 `docs/modules/<m>/TODO.md`；跨模組 / 全局排序 → `docs/TODO.md`。
- ablation 收斂 → 該 module README `⚖️ GO/NO-GO` 加一行，細節進 `docs/TODO_history.md`。
- 每個事實只有一個家、其餘只連結（詳見 §11）。

---

## 5. 當前 Baseline

**當前生產 baseline = `mamba_whole_graph`**（Option F Mamba 偵測頭 + 整圖 CUDA graph，**ReID off**）。整圖 graph 與 `mamba_optimal` 同路線、純加速 detect（7.4→3.1ms），精度持平、FPS 大幅提升。

> 詳細 CLI 默認值見 [docs/reference/mot17_default_config.md](docs/reference/mot17_default_config.md)。完整 baseline 矩陣（含各 Option 歷史）見 [docs/TODO.md](docs/TODO.md)。
> mamba preset 自帶 override，不吃 yolo26 path 的部分 default：`reid_mode=off`、`kalman_r_scale=2.8`、`match_thresh=0.50`、`fuse_score_weight=0.0`、`gmc_downscale=4`、`interpolate_max_gap=35`。

| preset | engine | IDF1 | MOTA | HOTA | IDs | FPS |
|--------|--------|------|------|------|-----|-----|
| **mamba_whole_graph**（當前 baseline，ReID off） | mamba v14 + yolo26s_640 | **73.3%** | **77.1%** | **66.7%** | **536** | **157.1** |
| mamba_optimal（head-only graph，前身） | mamba v14 + yolo26s_640 | 73.4% | 77.1% | 66.7% | 533 | 116.7 |
| speed（舊 yolo26 路線，參考） | yolo26s_960 | 52.3% | 41.8% | — | 473 | 138.9 |
| baseline（舊 yolo26 路線，參考） | yolo26m_960 | 51.4% | 43.5% | — | 501 | 113.2 |

`mamba_whole_graph` 完整指標：DetA 69.9% / AssA 63.9% / FP 3797 / FN 21333 / Rcll 81.0% / Prcn 96.0%（2026-06-03 SDP train，AssA 為瓶頸）。

使用方式：
```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP   # 當前 baseline
uv run scripts/eval/mot17.py --preset mamba_optimal --detector SDP       # head-only graph 前身
uv run scripts/eval/mot17.py --preset speed --detector SDP               # 舊 yolo26 路線（參考）
```

預設會將結果登錄到 MLflow（`--mlflow-uri http://localhost:5000`）。若 MLflow 未啟動會自動跳過。

調參時以 `--preset mamba_whole_graph` 為基準線。

---

## 6. 目前開發方向

詳細待辦、近期 ablation 結論、backlog 見 **[docs/TODO.md](docs/TODO.md)**。

### 模組現狀總覽

> 入口快照：一眼看現狀。每模組完整待辦見各 `docs/modules/<name>/TODO.md`，全局矩陣 / Baseline 見 [docs/TODO.md](docs/TODO.md)。

| 模組 | 狀態 | active 待辦 | TODO |
|------|------|-------------|------|
| 🔍 detection | 🔄 active | VGT-Mamba（訓練中）、Hybrid Mamba-ViT、資料集補強 | [↗](docs/modules/detection/TODO.md) |
| 📐 geometry | 🔄 active | GMC Warp 精度驗證（支援 VGT-Mamba） | [↗](docs/modules/geometry/TODO.md) |
| 🧬 reid | ⏸️ 暫緩 | Appearance Bank 尋回（待時序 YOLO 驗證） | [↗](docs/modules/reid/TODO.md) |
| 🔄 lifecycle | 📋 待辦 | evaluator.py lifecycle 切片測試覆蓋率 | [↗](docs/modules/lifecycle/TODO.md) |
| 🌀 motion | 🟢 收斂 | — | [↗](docs/modules/motion/TODO.md) |
| 🤝 semantic | 🟢 收斂 | — | [↗](docs/modules/semantic/TODO.md) |
| ⚡ trigger | 🟢 收斂 | — | [↗](docs/modules/trigger/TODO.md) |
| 🖥️ streaming | 🟢 收斂 | — | [↗](docs/modules/streaming/TODO.md) |
| 💾 storage | 🟢 收斂 | — | [↗](docs/modules/storage/TODO.md) |
| 🧠 cognition | 🟢 收斂 | — | [↗](docs/modules/cognition/TODO.md) |
| ⚙️ resource | 🟢 收斂 | — | [↗](docs/modules/resource/TODO.md) |

跨模組待辦（測試覆蓋率）見 [docs/TODO.md](docs/TODO.md) 的「跨模組待辦」節。

### 🔄 Option F — Mamba SSM Detection Head

- 設計文件：[docs/modules/detection/option-f-mamba-head.md](docs/modules/detection/option-f-mamba-head.md)
- 訓練腳本：`scripts/train/temporal_yolo/`
- 評估命令：`uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP`（當前 baseline；`mamba_optimal` 為 head-only graph 前身）

### ✅ Option E — GatedYOLODetector（baseline, IDF1 57.2%）

- 設計文件：[docs/modules/detection/option-e-v2-design.md](docs/modules/detection/option-e-v2-design.md)

### ❌ Option D — Track-Conditioned YOLO（NO-GO, 2026-05-19）

- 設計文件移至 [docs/archive/option-d/](docs/archive/option-d) 保留供參考

其餘已結案或延後的方向以 `docs/TODO.md` / `docs/TODO_history.md` 為準。

---

## 7. 檔案對應指南

### 改 tracking / association
`src/tracking/tracker_gpu.cu` → `include/tracking/tracker_gpu.hpp` → `src/saccade/perception/tracking/tracker_gpu.py`

### 改 semantic relink / identity resolve
`src/saccade/perception/eval/relink.py` → `src/tracking/tracker_gpu_python.cpp`

### 改 detection postprocess / tile merge
`src/saccade/perception/eval/detection.py` → `src/tracking/tracker_gpu.cu` → `scripts/eval/mot17.py`

### 改 eval config / ablation params
`scripts/eval/config/` → `src/saccade/perception/eval/config.py` → `src/saccade/perception/eval/evaluator.py`

### 改 evaluation / ablation 流程
`scripts/eval/mot17.py` → `src/saccade/perception/eval/evaluator.py`

### DB / Experiment Tracking
需要查詢時參考：
- `scripts/ops/mlflow_server.sh` — MLflow tracking server 啟動
- `scripts/eval/mlflow_logger.py` — MLflow logging 共用工具
- `scripts/tools/compare_trials.py` — Optuna trial 對比
- `scripts/tools/average_top_trials.py` — Optuna top-trial 平均

---

## 8. 開發流程

### 8.1 實驗追蹤

三個 PostgreSQL 資料庫透過 `docker compose up db -d` 提供：

| 資料庫 | 用途 | 介面 |
|--------|------|------|
| `saccade` | 預留（目前空） | — |
| `mlflow` | 實驗追蹤 (params / metrics / tags) | MLflow UI → `http://localhost:5000` |
| `optuna` | 超參數調優 (trials / studies) | `scripts/tools/compare_trials.py` / `optuna-dashboard` |

**啟動 MLflow：**
```bash
./scripts/ops/mlflow_server.sh
```

**跑 eval 自動登錄：**
```bash
uv run python scripts/eval/mot17.py --preset speed --detector SDP \
    --mlflow-uri http://localhost:5000 \
    --mlflow-experiment mot17
```

**跑 ablation 自動登錄每個 variant：**
```bash
uv run python -m scripts.eval.ablation_mot17 \
    --category detection \
    --mlflow-experiment mot17-ablation
```

**查詢過往 trial：**
```bash
uv run python scripts/tools/compare_trials.py \
    --study A1_Unified_Score --trials 0 1 2
```

MLflow 或 Optuna 未啟動時不影嚮 eval — `log_eval_run()` 會印 warning 並退場。

### 改動前
判斷改動類型：
- **架構 / 合約** → 更新 `docs/decisions/`、`docs/architecture.md`
- **近期方向 / 實驗** → 更新 `docs/TODO.md`
- **bug fix / 實作細節** → commit message 記錄 why

開分支前先確認工作項已落到 `docs/TODO.md`。
沒有進 TODO 的工作，不直接開新分支。

### 改動中
- 維持主熱路徑 GPU-first；避免未說明的 `.cpu()` / `numpy()` / host materialization
- 引入 fallback 時明確定義觸發條件與回退行為

### 改動後
若影響 default path，應補：
- `IDF1 / MOTA / IDs / FP / FN / FPS` 與 baseline 對比
- 哪些 sequence 改善 / 退化
- 是否屬於 run-to-run noise 還是系統性差異

### 實驗層級
不要把所有驗證壓成「最後 IDF1 有沒有變好」。分三層：
1. **baseline**：`./scripts/eval/module_benchmark.sh --mode all`
2. **module-local**：先看你改的那個模組的主指標
3. **promotion validate**：local signal 明確後，才升級看整體 e2e 指標

判讀規則：`local improved + downstream worse` = regression，不保留；`local + downstream + e2e` 全改善才是強候選。

---

## 9. Branching Policy

- `main`：單一主分支。所有功能開發、修正、文件更新最終都合回 `main`。
- `feat/*`、`fix/*`、`perf/*`：工作分支。從 `main` 拉出，完成後以 PR 合回 `main`，合併後立即刪除。

### 分支工作流

1. 先在 `docs/TODO.md` 寫清楚要做的工作、目的與驗證方式。
2. 從 `main` 建立工作分支。
3. 在工作分支開發與提交。
4. 推到遠端後，開 PR 指向 `main`。
5. CI 通過後合併到 `main`，並刪除工作分支。

### 規則

- 不直接 push 到 `main`。
- 預設所有改動都走 PR，即使是單人倉庫也保留 CI 與變更紀錄。
- 調整 CI 觸發分支時，必須同步更新 `.github/workflows/` 的 `push` / `pull_request` 觸發條件。
- 工作樹有未提交變更時，不做分支整理、批次 merge 或預設分支切換。

---

## 10. 本地驗證

### 推送前（鏡像 CI）

```bash
git config core.hooksPath .githooks  # 一次性設定：commit 前自動 ruff format/check staged Python files
bash scripts/pre_push.sh          # lockfile + ruff + mypy + pytest + C++ build 偵測
bash scripts/pre_push.sh --fix    # 同上，自動 ruff fix / format 後再檢查
```

`pre-commit` hook 只處理已暫存的 Python 檔案：先 `ruff format`，再 `ruff check`，最後自動重新 `git add`。`mypy / pytest / C++ build` 仍保留在 `pre_push`，避免每次 commit 過重。

CI 跑兩個 ruff 步驟（`ruff check` 和 `ruff format --check`），腳本一併涵蓋。有 `src/`、`include/`、`CMakeLists.txt` 改動且 `build/` 存在時，自動跑 `make saccade_tracking_ext`。

### 依改動範圍選擇

```bash
uv run pytest tests/ --ignore=tests/benchmarks
scripts/test_native.sh
./scripts/eval/module_benchmark.sh --mode all
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP
uv run scripts/eval/mot17.py --preset speed --detector SDP
```

---

## 11. 文件更新規則

**核心原則：每個事實只有一個家，其餘只連結、不複製。** 模組文檔多檔結構容易 drift，靠單一來源 + 連結檢查維持一致。

| 事實類型 | 唯一的家（single source of truth） |
|------|---------|
| 模組狀態 dashboard（🔄/✅/❌ 一覽） | 本文件 §6 模組現狀總覽 |
| 模組 active 待辦（要做什麼） | `docs/modules/<m>/TODO.md` |
| 模組職責 / 現況 / I/O & dataflow / GO·NO-GO 短結論 | `docs/modules/<m>/README.md` |
| 全局矩陣、Baseline、跨模組待辦、模組 TODO 索引 | `docs/TODO.md` |
| GO/NO-GO 細節、過程、舊參數掃描、舊路線圖 | `docs/TODO_history.md` |
| 架構決策紀錄（ADR） | `docs/decisions/` |
| 全局 dataflow 串接（16-stage） | `docs/reference/pipeline_flow.md` |
| pipeline module 與 metric 對應 | `docs/reference/PIPELINE_REFERENCE.md` |

**維護時機（綁 ablation 流程，避免 drift）：**
- ablation 收斂 → 在該 module README `⚖️ GO/NO-GO` 加 **一行**（日期＋項目＋結論）；長文進 `TODO_history.md`。
- 開新工作 → 寫進 `docs/modules/<m>/TODO.md`（**不是**主 TODO）。
- §6 dashboard 的狀態 icon → **只在階段轉換**（🔄→✅/❌）時改，不隨小改同步。
- 高頻動作是「加一行 ledger」；不做全套五處同步。連結正確性由 `scripts/pre_push.sh` 的 doc link check（`scripts/tools/check_doc_links.py`）自動把關。

歸檔原則：主 TODO 保留高訊號摘要；細節、過程、舊參數掃描移入 history。某方向重新啟動再從 history 摘回，不在主 TODO 長期保留已結案脈絡。

---

## 12. 補充入口

- [docs/architecture/README.md](docs/architecture/README.md)
- [docs/reference/PIPELINE_REFERENCE.md](docs/reference/PIPELINE_REFERENCE.md)
- [docs/TODO.md](docs/TODO.md)
- [docs/reference/mot17_default_config.md](docs/reference/mot17_default_config.md)
- [MLflow UI](http://localhost:5000)（需先啟動 `scripts/ops/mlflow_server.sh`）
- [scripts/eval/mlflow_logger.py](scripts/eval/mlflow_logger.py)（MLflow logging 工具）
- [scripts/tools/compare_trials.py](scripts/tools/compare_trials.py)（Optuna trial 對比）
- [scripts/tools/average_top_trials.py](scripts/tools/average_top_trials.py)（Optuna top-trial 平均）
