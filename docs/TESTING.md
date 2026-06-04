# Scripts 與 Tests 規範

本文件定義目前 repo 中 `scripts/` 與 `tests/` 的角色、命名、放置位置與最低驗證要求。

---

## 1. Scripts 規範

### 1.1 主入口

主 workflow 腳本集中在 `scripts/eval/`。

目前只有兩個核心入口：

- `/scripts/eval/mot17.py`
- `/scripts/eval/ablation_mot17.py`

### 1.2 腳本分類

其餘腳本分三類：

- 支援工具
  - 例如 `calculate_mota.py`、`convert_mot17.py`
- 替代流程
  - 例如 `mot17_public.py`、`ultralytics_official_mot17.py`
- 效能工具
  - 例如 `bench_yolo_batch.py`

目前目錄分層建議如下：

- `/scripts/eval/`
  - 主 eval / ablation workflow
- `/scripts/benchmarks/`
  - 延遲、吞吐量、壓力測試腳本
- `/scripts/native/`
  - native build / rebuild / coverage helpers
- `/scripts/model/`
  - 模型匯出、建模與 engine 工具
- `/scripts/ops/`
  - 本地串流、服務控制、demo 操作腳本
- `/scripts/tools/`
  - 環境檢查與輔助 shell / utility
- `/scripts/`
  - 只保留少量通用入口或 build / test shell scripts

### 1.3 命名與放置

- 新腳本一律使用 `snake_case.py`
- 新主流程腳本若會長期保留，優先放進 `scripts/eval/`
- 新 benchmark / latency / pressure 腳本，優先放進 `scripts/benchmarks/`
- 不要再新增零散 ad-hoc grid search / one-off ablation 入口作為主路徑

### 1.4 變更後同步更新

新增、移動、刪除腳本後，至少同步更新：

- `/scripts/eval/README.md`
- `/README.md`
- 必要時：
  - `/docs/README.md`
  - `/DEVELOPMENT.md`
  - `/docs/TODO.md`

---

## 2. Tests 規範

### 2.1 目錄分工

- `tests/test_*.py`
  - 單元 / 整合 / parity / e2e 測試
- `tests/native/`
  - C++ / native tests
- `tests/benchmarks/`
  - benchmark 與壓力測試，不屬於一般單元測試

### 2.2 命名規則

- 一般測試檔名必須符合 `test_*.py`
- benchmark 檔名使用 `bench_*.py`
- 新 benchmark 不應放進 `tests/test_*.py`

### 2.3 優先補哪些測試

若改 tracking / relink / runner 主路徑，優先補：

- parity test
- runner helper / materialization test
- 必要時 MOT17 short eval

---

## 3. 測試覆蓋率

目前總體 **66%**（647 passing tests，7,751 statements）。

| 模組 | 覆蓋率 |
|------|--------|
| `perception/eval/` | ~40% |
| `perception/tracking/` | ~85% |
| `perception/temporal_yolo/` | ~50% |
| `storage/` | ~80% |

### 查看覆蓋率

```bash
# 執行測試並顯示覆蓋率
uv run pytest --cov=saccade --cov-report=term-missing

# 產生 HTML 報告
uv run coverage html
# 開啟：htmlcov/index.html
```

### 覆蓋率成長

| 日期 | 覆蓋率 |
|------|--------|
| 2026-05-15 | 56% |

## 4. 現行工具設定

目前工具設定以 `/pyproject.toml` 為準。

### 3.1 Pytest

- `testpaths = ["tests"]`
- `python_files = ["test_*.py"]`
- `asyncio_mode = "strict"`
- 預設帶 coverage

### 3.2 Mypy

- 目前排除 `tests/`

### 3.3 Ruff

- 目前排除 `tests/benchmarks/archive`

---

## 4. 最低驗證要求

### 改 Python 主邏輯

至少執行：

```bash
uv run pytest
uv run mypy .
```

### 改 native / tracking 主路徑

至少執行：

```bash
scripts/test_native.sh
```

並補：

- 相關 parity tests
- 必要時 `scripts/eval/mot17.py`

### 改 benchmark / eval 腳本

至少：

- 跑對應腳本一次
- 確認 CLI 沒漂移
- 確認 README / docs 入口沒有失效

---

## 5. 常用指令

```bash
uv run pytest
uv run mypy .
scripts/test_native.sh
uv run python scripts/eval/mot17.py ...
uv run python scripts/eval/ablation_mot17.py ...
```

---

## 6. 原則

- 不要只新增腳本，不更新入口文件
- 不要把 benchmark 混進一般單元測試
- 不要把一次性實驗腳本長期留在主 workflow 路徑
- 若改動會影響 default path，驗證不應只停在單元測試
