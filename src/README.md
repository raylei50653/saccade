# `src/` — 從目錄找到架構與實作

本檔只回答一件事：**給一個路徑或一個改動，實作在哪、接著讀哪份文件。**

- 系統分層、責任邊界、主資料流 → [docs/architecture/README.md](../docs/architecture/README.md)
- 演算法現行最優 / GO / NO-GO → [docs/PIPELINE.md](../docs/PIPELINE.md)
- 開發需求層級與熱路徑命令 → [DEVELOPMENT.md](../DEVELOPMENT.md)

目錄表以 Git tracked 原始碼為準。headline 評測是 MOT17 + preset `mamba_whole_graph`（ReID off）；工業 RTSP / Redis / RAG 是另一條執行路徑，不是退役碼。

---

## 1. 範圍：為什麼有兩套 perception / tracking

`pyproject.toml` 把 Python 套件根設在 `src/`，所以產品碼在 `src/saccade/`。效能敏感的 C++/CUDA 與它並列，public header 在 `include/`。

```text
src/
├── perception/     native 檢測 / 預處理 / ReID 提取     → saccade_perception_ext
├── tracking/       native tracker + 後處理 / GMC / eval  → saccade_tracking_ext, saccade_eval_ext
├── media/          native GStreamer / buffer pool        → saccade_media_ext
├── main.cpp        C++ perception demo（saccade_node）
└── saccade/        Python 套件
    ├── perception/ eval 編排、Mamba 檢測棧、tracker facade、工業 dispatcher
    ├── media/      RTSP / DALI / GStreamer 客戶端
    ├── api/ cognition/ resource/ storage/ pipeline/
    └── …

include/{perception,tracking,media,saccade,utils}/   對應 public headers
```

同名不表示同一層：

| 看到這個名字 | 先分清 |
|:--|:--|
| `src/perception/` | C++ TensorRT / Mamba 引擎 |
| `src/saccade/perception/` | Python 編排、訓練、eval、工業膠水 |
| `src/tracking/` | C++ GPUByteTracker 與同目錄的後處理 / eval / scan |
| `src/saccade/perception/tracking/` | Python facade（`tracker_gpu.py`） |

改 tracker association／Kalman 先進 native；改 MOT 幀循環先進 `saccade/perception/eval/`；改 RTSP 先進 `saccade/media/`。不要只靠目錄名。

---

## 2. 目錄導航

### Native

| 路徑 | 實際職責 | 主要入口 | 命名例外 |
|:--|:--|:--|:--|
| [`perception/`](perception/) | TRT YOLO、Mamba gated detector、letterbox、ReID `FeatureExtractor` | [`perception_python.cpp`](perception/perception_python.cpp) → `saccade_perception_ext`；header [`../include/perception/`](../include/perception/) | `batched_mamba_detector.cpp` 的類名是 `BatchedBackbone`。`nv12_kernel.cu` / `rgb_to_nv12_kernel.cu` 有源碼，見 [§5](#5-易走錯的位置) |
| [`tracking/`](tracking/) | `GPUByteTracker`、GMC、`PerceptionPipeline`（filter/NMS/crop）、C++ eval pool、relink gate、Mamba scan、ReID crop 環 | [`tracker_gpu.cu`](tracking/tracker_gpu.cu)、[`tracker_gpu_python.cpp`](tracking/tracker_gpu_python.cpp) → `saccade_tracking_ext`；[`eval_python.cpp`](tracking/eval_python.cpp) → `saccade_eval_ext`；header [`../include/tracking/`](../include/tracking/) | 目錄比「tracker」寬。子目錄 [`CMakeLists.txt`](tracking/CMakeLists.txt) 只編 FPN ReID，**不是** GPUByteTracker（那在根 [`CMakeLists.txt`](../CMakeLists.txt)） |
| [`media/`](media/) | `GstClient`、GPU `BufferPool` | [`gst_client_python.cpp`](media/gst_client_python.cpp) → `saccade_media_ext`；header [`../include/media/`](../include/media/) | Python 預設 `SACCADE_MEDIA_USE_CPP=0`，C++ 路徑是 opt-in |
| [`main.cpp`](main.cpp) | Gst + `TRTEngine` + `Preprocessor` 的 demo node | CMake `saccade_node` | 未接 tracker；不是 MOT 或工業主入口 |

### Python 套件 `saccade/`

| 路徑 | 實際職責 | 主要入口 | 命名例外 |
|:--|:--|:--|:--|
| [`saccade/perception/eval/`](saccade/perception/eval/) | MOT17 評測編排：幀循環、stage、config、detect 後處理、GMC 工廠、relink、metrics | [`evaluator.py`](saccade/perception/eval/evaluator.py) `run_eval`；狀態袋 [`pipeline.py`](saccade/perception/eval/pipeline.py)；幀級 [`stages.py`](saccade/perception/eval/stages.py) | [`runner.py`](saccade/perception/eval/runner.py) 只 re-export `run_eval`。[`tracking.py`](saccade/perception/eval/tracking.py) 只有 global MOT id mapper。C++ 類 `PerceptionPipeline` 不在這裡 |
| [`saccade/perception/temporal_yolo/`](saccade/perception/temporal_yolo/) | YOLO26 + Mamba 檢測訓練／推論；舊 Option B–E 仍在同目錄 | 現行檢測：[`mamba_gated_detector.py`](saccade/perception/temporal_yolo/mamba_gated_detector.py)、[`mamba_head.py`](saccade/perception/temporal_yolo/mamba_head.py) | 目錄名與 [`__init__.py`](saccade/perception/temporal_yolo/__init__.py) 仍是 Temporal YOLO Hybrid，**不 re-export** Mamba |
| [`saccade/perception/tracking/`](saccade/perception/tracking/) | GPUByteTracker Python 包裝 | [`tracker_gpu.py`](saccade/perception/tracking/tracker_gpu.py) | 同目錄有 `dynamic_reid.py`、`fpn_reid*.py`、`reorder.py`（工業亂序緩衝），不是 Kalman/GMC 本體 |
| [`saccade/perception/reid/`](saccade/perception/reid/) | Cheb-GR graph re-ranking | [`cheb_gr.py`](saccade/perception/reid/cheb_gr.py) | 不是 crop / SigLIP / FeatureBank。那些在 perception 根層；關聯政策歸 [semantic 模組](../docs/modules/semantic/README.md) |
| [`saccade/perception/`](saccade/perception/) 根層 | 工業 dispatcher、zero-copy、TRT YOLO、workbench、可選 ReID 提取 | [`dispatcher.py`](saccade/perception/dispatcher.py)、[`workbench.py`](saccade/perception/workbench.py)、[`detector_trt.py`](saccade/perception/detector_trt.py)、[`feature_extractor.py`](saccade/perception/feature_extractor.py) | headline 檢測不走 `detector_trt.py`，走 `temporal_yolo/mamba_gated_detector.py`。`drift_handler.py` 是語義質心，不是 GMC |
| [`saccade/media/`](saccade/media/) | RTSP URL、MediaMTX/GStreamer 解碼、DALI | [`rtsp.py`](saccade/media/rtsp.py)、[`mediamtx_client.py`](saccade/media/mediamtx_client.py) | [`ffmpeg_utils.py`](saccade/media/ffmpeg_utils.py) 實際是 `RTSPStreamer`。MOT eval 拉 JPEG 不走這裡 |
| [`saccade/api/`](saccade/api/) | FastAPI 物件檢索 / 混合搜尋 | [`server.py`](saccade/api/server.py) | 不是系統控制 API，不啟動 perception |
| [`saccade/cognition/`](saccade/cognition/) | Redis → Chroma → 可選 RAG | [`orchestrator.py`](saccade/cognition/orchestrator.py) `PipelineOrchestrator` | 類名帶 pipeline，職責是慢路徑 cognition |
| [`saccade/pipeline/`](saccade/pipeline/) | 服務健康檢查（systemd / VRAM / Redis） | [`health.py`](saccade/pipeline/health.py) | **目前承載健康檢查**，不是幀 pipeline |
| [`saccade/resource/`](saccade/resource/) | VRAM 三階降級 + 跨進程 SHM | [`resource_manager.py`](saccade/resource/resource_manager.py) | [`frame_selector.py`](saccade/resource/frame_selector.py) 目前是空檔 |
| [`saccade/storage/`](saccade/storage/) | Redis Streams、Chroma memory | [`redis_cache.py`](saccade/storage/redis_cache.py)、[`chroma_store.py`](saccade/storage/chroma_store.py) | MOT eval 不用這層 |

---

## 3. 執行入口

### 從任務找入口

下表是**建議閱讀順序**（先打開哪個檔），不是呼叫鏈。實際呼叫見下方執行路徑。

| 我要改 | 建議閱讀順序 | 不要停在 | 接著讀 |
|:--|:--|:--|:--|
| **Detector**（headline Mamba） | [`temporal_yolo/mamba_gated_detector.py`](saccade/perception/temporal_yolo/mamba_gated_detector.py) → [`perception/mamba_gated_detector.cpp`](perception/mamba_gated_detector.cpp) | `temporal_yolo/__init__.py`、`detector_trt.py` | [detection 模組](../docs/modules/detection/README.md) |
| **Association / tracker** | [`tracking/tracker_gpu.cu`](tracking/tracker_gpu.cu) → [`include/tracking/tracker_gpu.hpp`](../include/tracking/tracker_gpu.hpp) → [`tracking/tracker_gpu.py`](saccade/perception/tracking/tracker_gpu.py) | [`eval/tracking.py`](saccade/perception/eval/tracking.py) | [tracker deep dive](../docs/modules/geometry/tracker_deep_dive.md) |
| **Eval 編排** | [`../scripts/eval/mot17.py`](../scripts/eval/mot17.py) → [`evaluator.py`](saccade/perception/eval/evaluator.py) → [`pipeline.py`](saccade/perception/eval/pipeline.py) → [`stages.py`](saccade/perception/eval/stages.py) | [`eval/runner.py`](saccade/perception/eval/runner.py) | [pipeline_flow.md](../docs/reference/pipeline_flow.md) |
| **RTSP / 工業 ingest** | [`saccade/media/`](saccade/media/) → [`dispatcher.py`](saccade/perception/dispatcher.py) → [`zero_copy.py`](saccade/perception/zero_copy.py) | [`eval/streaming.py`](saccade/perception/eval/streaming.py)（那是 MOT JPEG/DALI 拉流） | [streaming 模組](../docs/modules/streaming/README.md)、[rtsp_contract](../docs/modules/streaming/runbooks/rtsp_contract.md) |
| **Native build** | 根 [`../CMakeLists.txt`](../CMakeLists.txt) → [`../scripts/native/rebuild.sh`](../scripts/native/rebuild.sh) | [`tracking/CMakeLists.txt`](tracking/CMakeLists.txt)（FPN ReID 子工程） | [scripts/native/README.md](../scripts/native/README.md)、[`../scripts/test_native.sh`](../scripts/test_native.sh) |

### 三條執行路徑

**MOT eval（headline）**

```text
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP
        │
        ├─ 預設：eval.runner.run_eval  →  evaluator.run_eval（實作在 evaluator.py）
        └─ --cpp-threads N：evaluator.run_eval_cpp → saccade_eval_ext
```

Preset：[`configs/presets/mamba_whole_graph.yaml`](../configs/presets/mamba_whole_graph.yaml)。CLI 欄位在 `scripts/eval/config/`，執行期投影在 [`eval/config.py`](saccade/perception/eval/config.py)。

資料流（stage 名以 `evaluator.py` 為準）：

```text
eval/streaming.py  JPEG/DALI/nvJPEG
  → ingest_preprocess  eval/pool.py
  → detect             temporal_yolo/mamba_gated_detector.py
  → postprocess        eval/detection.py + saccade_tracking_ext.PerceptionPipeline
  → gmc                saccade_tracking_ext.GMC
  → track              tracking/tracker_gpu.py → tracker_gpu.cu
  → materialize / relink_write / MOT emit
  → metrics            eval/metrics.py
```

headline 上 ReID 分支關閉；開啟時才走 crop / `feature_extractor.py`。

**工業串流**

```text
python main.py --mode perception|orchestrator|full
scripts/ops/run_8stream_perception.py
        → saccade.media + dispatcher + workbench + resource
        → Redis / Chroma → cognition → api
```

**Native demo**

```text
saccade_node   ← src/main.cpp（感知節點，未接 tracker）
```

---

## 4. 跨語言對照

改 Python facade 不夠時，沿這張表進 extension 與 header。

| Python | Extension | Native | Header |
|:--|:--|:--|:--|
| [`tracking/tracker_gpu.py`](saccade/perception/tracking/tracker_gpu.py) | `saccade_tracking_ext.GPUByteTracker` | [`tracker_gpu.cu`](tracking/tracker_gpu.cu) | [`tracker_gpu.hpp`](../include/tracking/tracker_gpu.hpp) |
| [`workbench.py`](saccade/perception/workbench.py) | `saccade_tracking_ext.Workbench` | [`workbench.cpp`](tracking/workbench.cpp) + [`pipeline.cpp`](tracking/pipeline.cpp) | [`workbench.hpp`](../include/tracking/workbench.hpp)、[`pipeline.hpp`](../include/tracking/pipeline.hpp) |
| [`eval/cpp_runner.py`](saccade/perception/eval/cpp_runner.py) | `saccade_eval_ext` | [`eval_pool.cpp`](tracking/eval_pool.cpp)、[`seq_runner.cpp`](tracking/seq_runner.cpp) | [`eval_pool.hpp`](../include/tracking/eval_pool.hpp) |
| [`eval/relink.py`](saccade/perception/eval/relink.py) | `saccade_tracking_ext`（`SemanticRelinker`、`relink_gate_batch`） | [`relink_gate.cu`](tracking/relink_gate.cu) | [`relink_gate.hpp`](../include/tracking/relink_gate.hpp) |
| GMC（eval `_build_gmc_estimator`） | `saccade_tracking_ext.GMC` | [`gmc.cpp`](tracking/gmc.cpp)、[`gmc_kernel.cu`](tracking/gmc_kernel.cu) | [`gmc.hpp`](../include/tracking/gmc.hpp) |
| [`mamba_gated_detector.py`](saccade/perception/temporal_yolo/mamba_gated_detector.py) | `saccade_perception_ext.MambaGatedDetector` | [`mamba_gated_detector.cpp`](perception/mamba_gated_detector.cpp) | [`mamba_gated_detector.hpp`](../include/perception/mamba_gated_detector.hpp) |
| [`mamba_head.py`](saccade/perception/temporal_yolo/mamba_head.py) scan | `saccade_tracking_ext.selective_scan_*` | [`mamba_scan.cu`](tracking/mamba_scan.cu)；TRT plugin [`mamba_scan_plugin.cpp`](tracking/mamba_scan_plugin.cpp) | [`mamba_scan.cuh`](../include/tracking/mamba_scan.cuh) |
| [`detector_trt.py`](saccade/perception/detector_trt.py) | `saccade_perception_ext.TRTEngine` | [`trt_engine.cpp`](perception/trt_engine.cpp) | [`trt_engine.hpp`](../include/perception/trt_engine.hpp) |
| [`feature_extractor.py`](saccade/perception/feature_extractor.py) | `saccade_perception_ext.FeatureExtractor` | [`feature_extractor.cpp`](perception/feature_extractor.cpp) | [`feature_extractor.hpp`](../include/perception/feature_extractor.hpp) |
| [`cropper.py`](saccade/perception/cropper.py) | `saccade_perception_ext.Cropper` | [`preprocessor.cpp`](perception/preprocessor.cpp) | [`preprocessor.hpp`](../include/perception/preprocessor.hpp) |
| [`mediamtx_client.py`](saccade/media/mediamtx_client.py)（`SACCADE_MEDIA_USE_CPP=1`） | `saccade_media_ext.GstClient` | [`gst_client.cpp`](media/gst_client.cpp) | [`gst_client.hpp`](../include/media/gst_client.hpp) |

兩個額外 extension 的建構入口不同，不要從同一份 CMake 找：

- `saccade_cheb_gr_online_ext`：由**根** [`CMakeLists.txt`](../CMakeLists.txt) 建構（[`cheb_gr_online.cpp`](tracking/cheb_gr_online.cpp)）。eval 呼叫端是 Python [`eval/cheb_gr_online.py`](saccade/perception/eval/cheb_gr_online.py)（import `..reid.cheb_gr`，不 import 該 ext）。
- `saccade_fpn_reid_cuda`：由 [`tracking/CMakeLists.txt`](tracking/CMakeLists.txt) **子工程**建構，根 CMake 沒有 `add_subdirectory`。可選 FPN ReID，不是 headline tracker。

同名雙路徑（改一邊不會自動改另一邊）：Python [`eval/gmc.py`](saccade/perception/eval/gmc.py) vs C++ `GMC`；Python [`dynamic_reid.py`](saccade/perception/tracking/dynamic_reid.py) vs [`dynamic_reid_controller.cpp`](tracking/dynamic_reid_controller.cpp)；Python [`eval/multi_birth.py`](saccade/perception/eval/multi_birth.py) vs native `temporal_birth_pool` / `multi_signal_birth_pool`（編進 `saccade_tracking`，native test 在 `tests/native/`）。

---

## 5. 易走錯的位置

- **`temporal_yolo/`** — 現行檢測是 `mamba_gated_detector.py`。套件表面仍是舊 Hybrid。
- **`saccade/pipeline/`** — 只有 `health.py`。幀循環在 `eval/evaluator.py`；C++ 後處理是 `include/tracking/pipeline.hpp` 的 `PerceptionPipeline`；`eval/pipeline.py` 是序列狀態袋。
- **`eval/tracking.py`** — 不是 tracker。Association 在 `tracker_gpu.cu`。
- **`eval/runner.py`** — `mot17.py` 由此 import，實作在 `evaluator.py`。
- **`perception/reid/`** — Cheb-GR re-ranking。SigLIP / crop / FeatureBank 在 `feature_extractor.py`、`cropper.py`、`feature_bank.py`。
- **`eval/streaming.py` vs `saccade/media/`** — 前者 MOT JPEG；後者工業 RTSP。
- **`detector_trt.py`** — 工業與非-Mamba eval 的 TRT YOLO。headline 是 Mamba gated detector。
- **NV12 fused kernel** — 源碼在 [`nv12_kernel.cu`](perception/nv12_kernel.cu)、[`rgb_to_nv12_kernel.cu`](perception/rgb_to_nv12_kernel.cu)。根 CMake 的 `saccade_perception` / `saccade_tracking` 目前未列入這兩個檔；`saccade_tracking_ext` 有 `letterbox_gpu`，沒有 `nv12_*`。Python [`eval/pool.py`](saccade/perception/eval/pool.py) / [`dispatcher.py`](saccade/perception/dispatcher.py) 向該 ext 取符號，`ImportError` 時走 PyTorch fallback。改 NV12 要同時看源碼、CMake、pybind、Python loader。
- **`eval/_cuda/`** — 研究用 bridge replay，不是 production tracker。見該目錄 [README](saccade/perception/eval/_cuda/README.md)。

---

## 6. 深入文件

| 問題 | 文件 |
|:--|:--|
| 邏輯分層 L1–L6、主路徑合約 | [docs/architecture/README.md](../docs/architecture/README.md) |
| Stage 名與 source map | [docs/reference/pipeline_flow.md](../docs/reference/pipeline_flow.md)、[docs/DATAFLOW.md](../docs/DATAFLOW.md) |
| 演算法結論 | [docs/PIPELINE.md](../docs/PIPELINE.md) |
| 模組設計（detection / geometry / reid / semantic / streaming …） | [docs/README.md](../docs/README.md) 模組地圖 |
| Tracker 內部 | [docs/modules/geometry/tracker_deep_dive.md](../docs/modules/geometry/tracker_deep_dive.md) |
| Storage / API schema | [docs/modules/storage/api_spec.md](../docs/modules/storage/api_spec.md) |
| Native 重建與測試 | [scripts/native/README.md](../scripts/native/README.md)、[`scripts/test_native.sh`](../scripts/test_native.sh) |
| 頂層目錄（`src/` 以外） | [REPO_LAYOUT.md](../REPO_LAYOUT.md) |
| 專案方向（非數字） | [docs/PROJECT_DIRECTION.md](../docs/PROJECT_DIRECTION.md) |
