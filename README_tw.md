# Saccade

高效率邊緣視覺感知系統，整合 TensorRT detector、GPU tracking、
appearance-based ReID，以及 MOT 風格的評估工具。

> **🌍 [English Version](README.md)**

## 目前這個 Repo 的重心

這個 repo 目前主要有兩個實用中心：

- **線上感知管線**
  - TensorRT 偵測與前處理
  - GPU tracking、GMC、Kalman、appearance fusion
  - 非同步 embedding / storage / orchestration 元件

- **離線 MOT 評估與調參**
  - `scripts/eval/mot17.py` 是主要的 MOT17 評估入口
  - `scripts/eval/ablation_mot17.py` 是依參數分群的統一 ablation 入口

這份 README 以目前仍在使用的程式結構與工作流為主。

## 核心目錄

- `perception/`
  - Detector、preprocessing、ReID、relink、tracker 協調與 eval runner。

- `src/` 與 `include/`
  - C++ / CUDA tracking 與效能敏感的 extension。

- `media/`
  - 串流接入與視訊管線整合。

- `pipeline/`
  - perception、storage、cognition 之上的高階 orchestration。

- `storage/`
  - Redis / Chroma 相關的持久化路徑。

- `scripts/eval/`
  - MOT17 evaluation、ablation、conversion、comparison 與 metric 工具。

- `tests/`
  - 單元測試與 benchmark。

更完整的架構背景可參考 [docs/architecture.md](/home/ray/developer/ai/saccade/docs/architecture.md:1) 與 [docs/README.md](/home/ray/developer/ai/saccade/docs/README.md:1)。

## 環境

- Python：`3.12`
- 套件管理：`uv`
- 主要執行期相依：
  - `torch`
  - `torchvision`
  - `tensorrt-cu12`
  - `nvidia-dali-cuda120`
  - `motmetrics`

安裝相依可使用：

```bash
uv sync
```

如果你需要使用 C++ / CUDA extension，也要另外為你的機器完成 native targets 的建置。

## 主要工作流

### 1. 執行 MOT17 評估

```bash
uv run python scripts/eval/mot17.py \
  --detector SDP \
  --output results/MOT17_eval
```

`mot17.py` 目前將參數分為以下能力區塊：

- detection and preprocessing
- association
- geometry and ID stability
- ReID backbone
- semantic relink
- dynamic ReID trigger policy
- lifecycle merge and cleanup

### 2. 執行 Ablation Study

```bash
uv run python scripts/eval/ablation_mot17.py --category detection
uv run python scripts/eval/ablation_mot17.py --category detection,geometry
uv run python scripts/eval/ablation_mot17.py --category all
```

目前支援的分類：

- `detection`
- `association`
- `geometry`
- `reid`
- `semantic`
- `trigger`
- `lifecycle`

`scripts/eval/` 目前的腳本用途可參考 [scripts/eval/README.md](/home/ray/developer/ai/saccade/scripts/eval/README.md:1)。

### 3. 單獨重算 Tracking Metrics

```bash
uv run python scripts/eval/calculate_mota.py --results results/MOT17_eval
```

### 4. 與外部 Baseline 比較

```bash
uv run python scripts/eval/compare_framework_ultralytics.py \
  --saccade results/MOT17_eval \
  --ultralytics results/MOT17_ultralytics_eval \
  --gt-root datasets/MOT17/train \
  --detector SDP
```

## 其他評估路徑

- `scripts/eval/mot17_public.py`
  - 使用 MOT17 `det/det.txt` 的 public detections 來跑 tracking。

- `scripts/eval/ultralytics_official_mot17.py`
  - 使用 Ultralytics 官方 tracking 作為外部 baseline。

- `scripts/eval/bench_yolo_batch.py`
  - 測量 detector engine 在不同 batch size 下的 throughput / latency。

## 開發備註

- 目前 worktree 仍可能包含 tracking、GMC、ReID、evaluation 的持續實驗。
- MOT evaluation 流程最近已收斂到：
  - `scripts/eval/mot17.py`
  - `scripts/eval/ablation_mot17.py`
- 舊的 ad-hoc grid search 與分散式 ablation 入口已移除，以降低 drift。

## 測試

執行 Python 測試：

```bash
uv run pytest
```

部分 benchmark 與 evaluation 腳本仍需要：

- 可用的 CUDA 硬體
- TensorRT engines
- 本地 MOT dataset

## 狀態

如果這份 README 與程式碼不一致，請以 `perception/`、`scripts/eval/`、`tests/` 內的實作為準。這份文件已更新為目前以 MOT17 為中心的 evaluation workflow，但整個 repo 仍在持續迭代中。
