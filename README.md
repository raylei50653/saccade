# Saccade

高效率邊緣視覺感知與 MOT 風格評估系統，整合 TensorRT detection、
GPU tracking、appearance-aware ReID，以及 native-heavy tracking
infrastructure。

## Start Here

如果你要開發這個 repo，請用這個閱讀順序進入：

1. [DEVELOPMENT.md](DEVELOPMENT.md)
2. [docs/architecture/README.md](/docs/architecture/README.md)
3. [docs/reference/pipeline_flow.md](/docs/reference/pipeline_flow.md)
4. [docs/modules/storage/api_spec.md](/docs/modules/storage/api_spec.md)
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

## Current Benchmark — YOLO26s + Mamba + GPU Tracker

目前 headline 是 **2026-06-21 `frozen_v2` run**：MOT17 train / SDP
七個 sequence 的 GT-weighted internal evaluation。這不是 MOTChallenge test-server
leaderboard，也不應與不同 engine、profiling 或 subset 的吞吐數字直接比較。

**Stack:** YOLO26s TensorRT backbone + Mamba v14-replica T3→T1 head + C++/CUDA
`GPUByteTracker`（GMC、auction association、bidirectional bridge relink）。實際
preset 名稱為 `mamba_whole_graph`。

這個 checkpoint 屬於 v14-replica training lineage；其 teacher/cache/distillation
流程使用 MOT17 資料，歷史 replica 設定更涵蓋全部七個 sequence。因此這些數字是
**in-domain train evaluation**，不是 holdout generalization 結果。訓練資料範圍見
[v14 replication protocol](docs/modules/detection/mamba-v14-replication-protocol.md)。

| HOTA | IDF1 | MOTA | DetA | AssA | IDs | Eval FPS |
|---:|---:|---:|---:|---:|---:|---:|
| **70.2** | **78.2** | **78.4** | **70.9** | **69.7** | **413** | **269.47** |

重現指令：

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --double-buffer \
  --output out/frozen_v2
```

`frozen_v2` 的量測環境：**NVIDIA GeForce RTX 5070 Ti Laptop
GPU（12 GB）**、Driver `610.62`、CUDA UMD `13.3`。GPU 型號與量測協定會直接影響
FPS／latency；不要把此數字與不同 GPU 或不同 profiling setup 的結果並列比較。

完整 preset 與 metric protocol 見
[MOT17 Evaluation Configuration](docs/reference/mot17_default_config.md)。

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

- [DEVELOPMENT.md](DEVELOPMENT.md)
  - 開發主入口

- [docs/architecture/README.md](/docs/architecture/README.md)
  - 穩定架構邊界

- [docs/reference/pipeline_flow.md](/docs/reference/pipeline_flow.md)
  - 目前實作主路徑 flow

- [docs/modules/storage/api_spec.md](/docs/modules/storage/api_spec.md)
  - API / event / storage contract

- [docs/TODO.md](/docs/TODO.md)
  - 當前工作與 ablation backlog

## Tests

執行 Python 測試（覆蓋率 56%）：

```bash
uv run pytest
```

詳細覆蓋率報告：[docs/TESTING.md](/docs/TESTING.md)

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
