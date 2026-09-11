# Saccade

高效率邊緣視覺感知與 MOT 評估系統：YOLO26s TensorRT detection + Mamba-FPN
detection head + C++/CUDA `GPUByteTracker`（GMC、auction association、bidirectional
bridge relink），ReID-free 即時多目標追蹤。

目前以 **MOT17 為中心的 evaluation 與 tuning workflow** 為主，線上 perception
stack 保留在周邊。

![Saccade architecture — YOLO26s detection → Mamba-FPN head → GPUByteTracker](docs/reference/math_model_architecture.svg)

完整數學模型與架構說明見 [docs/reference/math_model.md](docs/reference/math_model.md)。

## Current Benchmark — YOLO26s + Mamba-FPN + GPU Tracker

headline 為 **2026-06-21 `frozen_v2` run**：MOT17 train / SDP 七個 sequence 的
GT-weighted internal evaluation。這是 in-domain train evaluation，**不是**
MOTChallenge test-server leaderboard，也不應與不同 engine / profiling / subset
的吞吐數字並列比較。

<!-- fact-owner: current-baseline = docs/TODO.md -->
> 下表為 headline 快照；baseline 數字的唯一事實來源是 [docs/TODO.md](docs/TODO.md)「當前 Baseline」節。

| HOTA | IDF1 | MOTA | DetA | AssA | IDs | Eval FPS |
|---:|---:|---:|---:|---:|---:|---:|
| **70.2** | **78.2** | **78.4** | **70.9** | **69.7** | **413** | **269.47** |

量測環境：**NVIDIA GeForce RTX 5070 Ti Laptop GPU（12 GB）**、Driver `610.62`、
CUDA UMD `13.3`。GPU 型號與量測協定會直接影響 FPS／latency。

重現：

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --double-buffer \
  --output out/frozen_v2
```

完整 preset 與 metric protocol 見
[MOT17 Evaluation Configuration](docs/reference/mot17_default_config.md)；training
資料範圍見
[v14 replication protocol](docs/modules/detection/mamba-v14-replication-protocol.md)。

## Quick Start

```bash
uv sync                                                                # Python 3.12 + uv (no DALI)
uv sync --extra dali                                                   # GPU + NVDEC media path only
uv run python scripts/eval/mot17.py --detector SDP --output results/MOT17_eval
```

常見執行期相依：`torch`、`torchvision`、`tensorrt-cu12`、`motmetrics`。  
**DALI**（`nvidia-dali-cuda120`）是 **optional extra** — 無 GPU 雲端 / CI / C++ core
build 不裝；本機 GPU 解碼路徑再 `uv sync --extra dali`。  
native C++ / CUDA extension 需另為你的機器 build；部分 benchmark／eval flow 需要
CUDA hardware、TensorRT engines 與 local MOT datasets。

所有 evaluation / ablation / metrics / mAP / baseline 比較指令見
**[scripts/eval/README.md](scripts/eval/README.md)**。

## Repo Orientation

從實體目錄找到實作：[src/README.md](src/README.md)。

- `src/saccade/perception/` — detector、preprocessing、ReID、relink、tracker
  coordination、eval runner
- `src/`、`include/` — C++ / CUDA tracking 與效能敏感 native components
- `src/saccade/storage/` — Redis / Chroma eventing 與 persistence
- `src/saccade/cognition/`、`src/saccade/api/` — orchestrator、retrieval API
- `scripts/eval/`、`tests/` — evaluation runner、unit / parity / e2e / benchmark

開發前建議閱讀順序：

1. [DEVELOPMENT.md](DEVELOPMENT.md) — 開發薄入口（需求層級 D0–D4 → action cards → 文檔組合 / 命令）
2. [docs/architecture/README.md](docs/architecture/README.md) — 穩定架構邊界
3. [src/README.md](src/README.md) — `src/` 目錄地圖與跨語言入口
4. [docs/reference/pipeline_flow.md](docs/reference/pipeline_flow.md) — 主路徑 flow
5. [docs/TODO.md](docs/TODO.md) — 當前工作與 ablation backlog

更多文件入口見 [docs/README.md](docs/README.md)。

## Tests

```bash
uv run pytest          # Python 測試（覆蓋率見 docs/TESTING.md）
uv run mypy .
scripts/test_native.sh # native C++ / CUDA
```

詳細覆蓋率見 [docs/TESTING.md](docs/TESTING.md)；推送前驗證流程見
[DEVELOPMENT.md](DEVELOPMENT.md) §10。

## Status

如果這份 README 與程式碼衝突，以 `src/saccade/perception/`、`src/tracking/`、
`scripts/eval/`、`tests/` 下的主路徑程式碼為準，並同步更新文件。

## License

本專案自身的程式碼與文件以 **Apache License 2.0** 授權，全文見 [LICENSE](LICENSE)，
歸屬與第三方聲明見 [NOTICE](NOTICE)。

repo 內另有兩份直接收錄的第三方程式碼，各自沿用原授權：

- `third_party/TrackEval/` — MIT（[LICENSE](third_party/TrackEval/LICENSE)）
- `src/saccade/perception/eval/_torch_graphs.py` — 自 PyTorch 2.11.0 的
  `torch/cuda/graphs.py` vendored，BSD 3-Clause
  （[_torch_graphs.LICENSE](src/saccade/perception/eval/_torch_graphs.LICENSE)）

### 散佈時的相依性授權注意事項

`ultralytics` 是 **AGPL-3.0**，而且被 detector 路徑直接 import
（`src/saccade/perception/temporal_yolo/`、`scripts/eval/` 共數十個檔案）。
本 repo 的 Apache-2.0 只涵蓋我們自己寫的部分，不改變任何相依性自身的授權條款。

依 [Ultralytics 官方 licensing guidance](https://www.ultralytics.com/license)，
把含 `ultralytics` 的 combined work 散佈出去、或以網路服務形式提供，可能被要求
遵守 AGPL-3.0（包含提供對應原始碼），或改為取得 Ultralytics 商業授權。這裡陳述
的是 upstream 自己的授權立場，不是本 repo 對 AGPL 邊界的法律判斷；實際要走哪條
路請自行評估或徵詢法律意見。
