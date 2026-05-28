# Saccade

高效率邊緣視覺感知與 MOT 風格評估系統，整合 TensorRT detection、
GPU tracking、appearance-aware ReID，以及 native-heavy tracking
infrastructure。

## Start Here

如果你要開發這個 repo，請用這個閱讀順序進入：

1. [DEVELOPMENT.md](/DEVELOPMENT.md)
2. [docs/architecture.md](/docs/architecture.md)
3. [docs/pipeline_flow.md](/docs/pipeline_flow.md)
4. [docs/api_spec.md](/docs/api_spec.md)
5. [docs/TODO.md](/docs/TODO.md)

這個 repo 目前以 **MOT17 為中心的 evaluation 與 tuning workflow** 為主，
線上 perception stack 仍保留在周邊。

## What This Repo Is Now

這個 repo 現在有兩個實用中心：

- **GPU-first perception / tracking stack**
  - detection、postprocess、GMC、Kalman、association、relink
  - 主要 hot path 在 `src/`、`include/`、`src/saccade/perception/`

- **離線 MOT evaluation 與 tuning**
  - `scripts/eval/mot17.py` 是主要 evaluation 入口
  - `scripts/eval/ablation_mot17.py` 是目前的 grouped ablation runner

目前記錄中的 default path 與活躍優化方向，請看
[docs/TODO.md](docs/TODO.md)。

## 0-shot Policy

這個 repo 目前對 `FP/TP classifier` 的原則是：

- **不接受**用最終 `MOT17` eval/test sequence 標註回訓 classifier
- `MOT17` 應只作 inference / evaluation
- 若要做 classifier 路線，應走 `external-only` 訓練資料，例如 `CrowdHuman / CityPersons`

正式方向與實作計畫見
[docs/research/eval/fp_classifier_external_only_plan.md](docs/research/eval/fp_classifier_external_only_plan.md)。

## Main Code Areas

- `src/saccade/perception/`
  - detector、preprocessing、ReID、relink、tracker coordination、eval runner

- `src/` and `include/`
  - C++ / CUDA tracking 與效能敏感的 native components

- `src/saccade/storage/`
  - Redis / Chroma eventing 與 persistence

- `src/saccade/cognition/`
  - orchestrator 與 slow-path cognition

- `src/saccade/api/`
  - retrieval API

- `scripts/eval/`
  - MOT17 evaluation、ablation、metrics、comparison helpers

- `tests/`
  - unit、parity、e2e、benchmark coverage

更多文件入口請看 [docs/README.md](/docs/README.md)。

## Environment

- Python：`3.12`
- 套件管理：`uv`
- 常見執行期相依：
  - `torch`
  - `torchvision`
  - `tensorrt-cu12`
  - `nvidia-dali-cuda120`
  - `motmetrics`

安裝相依：

```bash
uv sync
```

如果你要使用 native C++ / CUDA extension，也需要為你的機器完成對應 build。

## Main Workflows

### 執行 MOT17 Evaluation

```bash
uv run python scripts/eval/mot17.py \
  --detector SDP \
  --output results/MOT17_eval
```

目前 evaluation flow 主要圍繞：

- detection and preprocessing
- association
- geometry / ID stability
- ReID backbone and trigger policy
- semantic relink
- lifecycle merge / cleanup

### 執行 Ablation

```bash
uv run python scripts/eval/ablation_mot17.py --category detection
uv run python scripts/eval/ablation_mot17.py --category association,semantic
uv run python scripts/eval/ablation_mot17.py --category all
```

目前支援的 category：

- `detection`
- `association`
- `geometry`
- `reid`
- `semantic`
- `trigger`
- `lifecycle`

目前 script map 請看 [scripts/eval/README.md](/scripts/eval/README.md)。

### 重算 Metrics

```bash
uv run python scripts/eval/calculate_mota.py --results results/MOT17_eval
```

### 計算 Detector mAP

```bash
uv run python scripts/eval/detection_map.py \
  --model models/yolo/yolo26s_960_batch1.engine \
  --sequences MOT17-04-SDP \
  --max-frames 100
```

### 與外部 Baseline 比較

```bash
uv run python scripts/eval/compare_framework_ultralytics.py \
  --saccade results/MOT17_eval \
  --ultralytics results/MOT17_ultralytics_eval \
  --gt-root datasets/MOT17/train \
  --detector SDP
```

## Alternative Paths

- `scripts/eval/mot17_public.py`
  - 用 MOT17 public detections 跑 tracking

- `scripts/eval/ultralytics_official_mot17.py`
  - Ultralytics baseline path

- `scripts/eval/bench_yolo_batch.py`
  - detector batch throughput / latency

## Development and Documentation

- [DEVELOPMENT.md](/DEVELOPMENT.md)
  - 開發主入口

- [docs/architecture.md](/docs/architecture.md)
  - 穩定架構邊界

- [docs/pipeline_flow.md](/docs/pipeline_flow.md)
  - 目前實作主路徑 flow

- [docs/api_spec.md](/docs/api_spec.md)
  - API / event / storage contract

- [docs/TODO.md](/docs/TODO.md)
  - 當前工作與 ablation backlog

## Tests

執行 Python 測試（覆蓋率 56%）：

```bash
uv run pytest
```

詳細覆蓋率報告：[docs/TEST_COVERAGE.md](/docs/TEST_COVERAGE.md)

常用驗證指令：

```bash
uv run mypy .
scripts/test_native.sh
```

產出 HTML 覆蓋率報告：

```bash
uv run coverage html  # 開啟 htmlcov/index.html
```

部分 benchmark 與 eval flow 需要：

- CUDA-capable hardware
- TensorRT engines
- local MOT datasets

## Status

如果這份 README 與程式碼衝突，請以
`src/saccade/perception/`、`src/tracking/`、`scripts/eval/`、`tests/`
下的主路徑程式碼為準，並同步更新文件。
