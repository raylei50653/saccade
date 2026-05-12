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
3. **穩定架構 / 合約文件**（`docs/architecture.md`、`docs/PIPELINE_REFERENCE.md`、`docs/api_spec.md`、`docs/decisions/`）
4. **當前待辦**（`docs/TODO.md`）
5. **歷史脈絡**（`docs/TODO_history.md`、`docs/progress/`、`docs/experiments/`）

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
- 近期方向 / 實驗排序 → 更新 `docs/TODO.md`。
- 已完成且不需再追蹤的內容 → 移到 `docs/TODO_history.md`。

---

## 5. 當前 Baseline

> 詳細 CLI 默認值見 [docs/mot17_default_config.md](docs/mot17_default_config.md)。已 default 的 flag：`fuse_score_weight=0.4`、`interp`、`fp_hard_filter`、`kalman_r_scale=0.75`、`async_reid`、`pipeline_relink`、`gmc gpu`、`detection_quality_scaling`。

| preset | engine | IDF1 | MOTA | IDs | FPS |
|--------|--------|------|------|-----|-----|
| **speed** | yolo26s_960 | **52.0%** | **41.6%** | **475** | **97.9** |
| **baseline** | yolo26m_960 | 51.4% | 43.5% | 502 | ~85 |
| **accuracy** | yolo26l_960 | — | — | — | — |

使用方式：
```bash
uv run scripts/eval/mot17.py --preset speed --detector SDP
uv run scripts/eval/mot17.py --preset baseline --detector SDP
```

調參時以 `--preset speed` 為基準線；需要更高 Recall 時用 `--preset baseline`。

---

## 6. 目前開發方向

詳細待辦、近期 ablation 結論、backlog 見 **[docs/TODO.md](docs/TODO.md)**。

目前無高優先未完成項；下一個有明確收益的方向是 **P3：Detector 訓練資料改善**（補足腿/腳標注，根本解決 FN 問題）。

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

---

## 8. 開發流程

### 改動前
判斷改動類型：
- **架構 / 合約** → 更新 `docs/decisions/`、`docs/architecture.md`
- **近期方向 / 實驗** → 更新 `docs/TODO.md`
- **bug fix / 實作細節** → commit message 記錄 why

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

## 9. 本地驗證

### 推送前（鏡像 CI）

```bash
bash scripts/pre_push.sh          # lockfile + ruff + mypy + pytest + C++ build 偵測
bash scripts/pre_push.sh --fix    # 同上，自動 ruff fix / format 後再檢查
```

CI 跑兩個 ruff 步驟（`ruff check` 和 `ruff format --check`），腳本一併涵蓋。有 `src/`、`include/`、`CMakeLists.txt` 改動且 `build/` 存在時，自動跑 `make saccade_tracking_ext`。

### 依改動範圍選擇

```bash
uv run pytest tests/ --ignore=tests/benchmarks
scripts/test_native.sh
./scripts/eval/module_benchmark.sh --mode all
uv run scripts/eval/mot17.py --preset speed --detector SDP
```

---

## 10. 文件更新規則

| 文件 | 保留什麼 |
|------|---------|
| `docs/TODO.md` | 目前待辦、近期 ablation 結論、下一輪 backlog |
| `docs/TODO_history.md` | 已完成、已收斂放棄、舊路線圖與過程紀錄 |
| `docs/decisions/` | 架構決策紀錄（ADR） |
| `docs/PIPELINE_REFERENCE.md` | pipeline module 與 metric 對應 |

歸檔原則：主 TODO 保留高訊號摘要；細節、過程、舊參數掃描移入 history。某方向重新啟動再從 history 摘回，不在主 TODO 長期保留已結案脈絡。

---

## 11. 補充入口

- [docs/architecture.md](docs/architecture.md)
- [docs/PIPELINE_REFERENCE.md](docs/PIPELINE_REFERENCE.md)
- [docs/TODO.md](docs/TODO.md)
- [docs/mot17_default_config.md](docs/mot17_default_config.md)
