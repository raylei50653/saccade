# Native runtime packaging audit（#465 Phase A）

> 狀態：Phase A 調查結論；**不改** runtime 行為、preset、threshold、weights、benchmark claim 或 packaging policy。
> 固定 source commit：`7be2f51f`（main）。證據方法：靜態讀碼＋`mot17.py --help` 的 `-X importtime` import closure＋對既有 build 產物的 `readelf -d` / `cuobjdump`。
> 沒有執行 inference、benchmark 或 packaging 實驗；本文不發任何效能或精度數字。
> build 產物來自主 checkout `build/`（2026-09-24 建置、分支 `perf/459-merge-sparse-exact`）；該分支相對 `7be2f51f` 在 `CMakeLists.txt`、`src/perception/`、`src/tracking/tracker_gpu_python.cpp` 無差異，也就是受檢 target 的 **source 與 main 等價**。但 ELF／link／CUDA 性質也取決於 CMake cache、build 選項與 build 環境，所以 §3.2、§8 觀察到的性質**只描述這一個實測 build configuration**（見 §3.2 的 cache 摘錄），不主張對 main 的每一種 build configuration 都成立。

本文回答一個問題：**supported headline runtime 能不能做成終端使用者免裝 Python 的預編譯包？不能的話差多少。**

相關但不同的 owner（link-don't-relabel，照抄其自稱）：

- [ADR 025](../decisions/025-native-extension-delivery.md) — native extension 由 consumer 從 checkout 建置；§5 列出「prebuilt wheel 線的前置條件」。本文不重複其結論，只在 §7 引用。
- #441 — public tracker-core API / distribution channel；明確不含 native wheel packaging。
- [production_pipeline_code_map.md](production_pipeline_code_map.md) — 同一條 headline 路徑的演算法／tensor 閱讀快照（`806c52cf`）。本文只取其中的**所有權邊界**，不重述演算法。

---

## 0. 結論先講

**Verdict：`native_package_blocked_by_bounded_python_surface`**

- 目前的 headline runtime **不是**一個產品執行檔，而是 Python eval harness：`scripts/eval/mot17.py` → `evaluator.run_eval`。每一幀的排程、GPU 記憶體配置、CUDA stream／event、**所有** CUDA graph capture，以及 Mamba detection head 本身，都由 Python/PyTorch 擁有。
- native 端（tracker、NMS／private continuation、GMC）已經是 raw-pointer API，演算法語義在 C++/CUDA 裡，**不需要重新設計**。
- 擋路的是一份**可以列舉完的** Python 擁有語義（§5，11 項），其中真正需要驗證成本的只有一項：**headline Mamba head 目前沒有經過 parity 驗證的 native 形式**。
- 現成的 C++ 路徑（`--cpp-threads` → `saccade_eval_ext`）**不能**代表 headline：它拒絕 `private_continuation`（`mot17.py:127-129`），沒有 OAO 設定欄位（`cpp_runner.py` 註解），用 `cv::imread` 在 CPU 解碼，而且它的 TorchScript head（`models/yolo/mamba_head_best.pt`）沒有綁到 headline checkpoint 的 lineage 紀錄。
- 什麼情況會把 verdict 升級成 `requires_runtime_redesign`：若 §8 的 U1 證明 headline head 無法以 native 形式（TRT engine 或 LibTorch）在已宣告的容差內重現，就代表 detector 語義非改不可，benchmark claim 也得重新建立基準。這一點目前**未知**，不是已知。

---

## 1. Entrypoint

| 問題 | 答案 | 證據 |
|:--|:--|:--|
| headline 怎麼啟動 | `uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer --output …` | [README.md](../../README.md) 重現段；[mot17_default_config.md](mot17_default_config.md) |
| 實際呼叫 | `mot17.py` → `build_mamba_gated_detector(...)`（`mot17.py:371`）→ `run_eval(**eval_kwargs)`（`mot17.py:445`） | `mot17.py` |
| 有沒有 native 執行檔 | `saccade_node`（`src/main.cpp`）存在，但只是 Gst + `TRTEngine` + `Preprocessor` 的 demo，**沒接 tracker**，寫死 `yolo26n_native.engine`；不是 headline | `src/main.cpp`；[src/README.md](../../src/README.md) §3 |
| 另一條 C++ 路徑 | `--cpp-threads N` → `evaluator.run_eval_cpp` → `saccade_eval_ext`（`seq_runner.cpp`）。**headline preset 開了 `private_continuation_enabled`，這條路徑會被 parser 拒絕** | `mot17.py:127-129` |
| 工業串流入口 | `main.py --mode perception…` → dispatcher 用 `TRTYoloDetector`，不是 Mamba detector；不是 headline | `src/saccade/perception/dispatcher.py:12` |

也就是說，**repo 裡沒有任何「端到端產品 runtime」入口**：唯一能跑 headline 語義的是 eval harness，它的輸入是 MOT17 目錄（`img1/*.jpg` + `seqinfo.ini`），輸出是每個 sequence 一份 MOT txt 加上 metrics。

---

## 2. 執行鏈與每個邊界的所有權

`P` = Python（含 PyTorch op）；`N` = native C++/CUDA，透過 pybind 以 raw pointer 呼叫；`P→N` = Python 持有 buffer／stream／graph，呼叫 native kernel。

| # | 段落 | 擁有者 | 實作 | 對應的 native twin（若有）與差異 |
|:--|:--|:--|:--|:--|
| 0 | 設定解析：argparse 預設 → base YAML → module YAML → preset → CLI → `config.py` 最終 fallback；`configure_runtime_env` 寫入 `SACCADE_DOUBLE_BUFFER` / `SACCADE_DETECT_BARRIER` / `SACCADE_GPU_DECODE` / `SACCADE_MAIN_NMS_GRAPHED` | P | `scripts/eval/mot17.py`、`scripts/eval/config/*`、`mot17_args.py:167`、`src/saccade/perception/eval/config.py` | 無。native 只收已解析的參數 |
| 1 | 讀檔＋JPEG 解碼（背景 thread 預取） | P | `TorchvisionGpuStreamer`（`eval/streaming.py:131`），`torchvision.io.decode_jpeg(device="cuda")` → nvJPEG | `seq_runner.cpp:152` 用 `cv::imread`（CPU），解碼器不同 |
| 2 | Ingest：HWC u8 → CHW f32 /255 寫入 pool | P（torch op） | `eval/pool.py`、`stages.py` `_run_detect` | 無同義 twin（native `Preprocessor` 做 letterbox，headline 是 stretch） |
| 3 | Resize 640 → TRT backbone → Mamba head → decode／top-k，整段包成一個 whole-detect CUDA graph | P→N（TRT）＋P（head/decode） | `mamba_gated_detector.py` `_whole_graph_fn:1097`；backbone 用 **Python `tensorrt` bindings**（`TRTYoloBackbone:382`）；head = PyTorch `MambaDetectionHead._forward_eager`，selective scan 走 `saccade_tracking_ext.selective_scan_fwd`；預設 `torch.compile`（`mot17.py` 的 `set_postprocess_compile(True)`、`set_head_compile`、`set_block_compile`） | C++ `MambaGatedDetector`（`src/perception/mamba_gated_detector.cpp`）＝`TRTEngine` + **LibTorch TorchScript** head，資產 `mamba_head_best.pt`，lineage 未綁到 headline ckpt（見 §3.4）；TRT head（`TRTMambaHead` + `libsaccade_scan_plugin.so`）的 parity 依 [mamba_whole_graph_analysis.md §8.3](../modules/detection/mamba_whole_graph_analysis.md) 仍是「尚未完成」 |
| 4 | Track priors、main NMS、private continuation NMS + append | P→N | `stages.py` `_run_nms` → `saccade_tracking_ext.PerceptionPipeline`（`pipeline.cpp`）；graph capture 由 Python 做（`stages.py:906,1009,1101`） | kernel 在 native；capture／buffer 生命週期在 Python |
| 5 | FP hard filter（score mask，保持 shape） | P（torch op） | `detection_filters.py` `_fp_hard_reject_mask:828` | `seq_runner.cpp:426` 有 C++ 版，未驗 parity |
| 6 | GMC（cuFFT phase correlation） | P→N | `PyGraphedGMC`（`eval/gmc.py:250`，以 `graphed_callables` capture）→ `saccade_tracking_ext.GMC` | native 完整 |
| 7 | Tracker update：Kalman、OAO／occ、cost、五個 pass、birth、geometric bridge、compact | **N**（語義）／P（capture＋buffer） | `GraphedTrackerUpdate`（`tracker_gpu.py:1740`，`torch.cuda.make_graphed_callables`）→ `update_into` → `tracker_gpu.cu` `run_update_device`；參數由 `GPUByteTracker.set_params`（`tracker_gpu.py:587`）從 config 映射 | native 完整；只有 config→`set_*` 的映射在 Python |
| 8 | Pinned D2H、per-frame MOT emit、local→global ID 映射 | P | `evaluator.py` emit 區段；`GlobalTrackIdMapper`（`eval/tracking.py:1`） | `cpp_runner.py:275` 有另一份 emit |
| 9 | Sequence tail：`interpolate_tracklets`（max_gap 35、min_len 5）、寫 `<seq>.txt` | P（**pandas** + numpy） | `post_merge.py:359`（`pd.read_csv`，輸出格式 `:.2f`／`:.4f`）；`evaluator.py:3201-3210` | 無 |
| 10 | Double-buffer 排程（side stream detect、event barrier） | P | `evaluator.py` `_launch_double_buffer_detect`；`SACCADE_DETECT_BARRIER=event` | 無 |
| 11 | Metrics（TrackEval／motmetrics）、MLflow、run manifest | P | `eval/metrics.py`、`scripts/eval/mlflow_logger.py`、`scripts/provenance/run_manifest.py` | 不屬於 runtime |

**總結**：native 擁有 tracker、NMS、GMC 的**演算法語義**；Python 擁有**執行宿主**（frame loop、排程、stream、allocator、graph capture）、**detector head**、**輸入解碼**，以及**輸出語義**（ID 空間、插值、文字格式）。

---

## 3. Runtime 依賴清單

### 3.1 Python interpreter 與套件

直譯器：CPython 3.12（`requires-python = ">=3.12, <3.13"`；native build 用 `find_package(Python3 3.12 EXACT)`）。

`mot17.py --help` 的 import closure（`-X importtime`，只載入模組、不跑 frame）會載入：`torch`（含 `torch.export` → `pydot`、`torch.cuda` → `pynvml`）、`numpy`、`pandas`（→ `pyarrow`）、`yaml`、`cv2`、`tensorrt`、`nvidia.dali`（在 try 區塊內）、`saccade_tracking_ext`、`saccade_perception_ext`、`tqdm`。frame loop 執行時還會 lazy 載入 `torchvision`（decode）與 `triton`（`torch.compile` 的 backend）。

### 3.2 Native libraries（`readelf -d` NEEDED）

實測 build configuration（主 checkout `build/CMakeCache.txt` 與 `CMakeFiles/4.4.3/`）：

| 項目 | 值 | 備註 |
|:--|:--|:--|
| CMake | 4.4.3 | |
| `CMAKE_BUILD_TYPE` | `Debug` | 全域 flags 仍另外加 `-O3`（`CMakeLists.txt` §1） |
| `CMAKE_CUDA_ARCHITECTURES`（cache） | `75` | 被 `CMakeLists.txt:102` 的一般變數 `"native"` 蓋掉；實測 SASS 只有 `sm_120`，與這個解讀一致 |
| CUDA compiler | nvcc 13.3.73（venv） | host compiler `/usr/bin/g++-15` |
| C++ compiler | `/usr/sbin/g++-13`（13.4.1） | |
| `ENABLE_NATIVE_TESTS` / `ENABLE_NATIVE_COVERAGE` / `SACCADE_ENABLE_NVTX` | ON / OFF / ON | |
| `Torch_DIR` / TensorRT libs | 主 checkout 的 `.venv`（`torch/share/cmake/Torch`、`tensorrt_libs/libnvinfer{,_plugin}.so.10`） | 絕對 RUNPATH 的來源 |
| `OpenCV_DIR` | `/usr/lib/cmake/opencv5`（系統） | |

| 產物 | NEEDED（去掉 libc/libstdc++/libm/libgcc） | 備註 |
|:--|:--|:--|
| `saccade_tracking_ext` | `libcudart.so.13`、`libcublas.so.13`、`libcublasLt.so.13`、`libcufft.so.12`、`libnvinfer.so.10`、`libtorch{,_cpu,_cuda}.so`、`libc10{,_cuda}.so`、`libopencv_{core,imgproc,video,features,geometry}.so.500` | tracker 程式碼本身**不用** torch／TRT：`saccade_tracking` 靜態庫連到 `saccade_perception`（`CMakeLists.txt:336-341`），所以一起被帶進來。OpenCV 只用在非 headline 的 sparse-flow GMC 模式（`gmc.cpp`：`goodFeaturesToTrack`、`calcOpticalFlowPyrLK`…）；headline GMC 走 cuFFT |
| `saccade_perception_ext` | 同上 + **`libtorch_python.so`**、`libnvinfer_plugin.so.10` | headline 只在 import 時載入它（`detector_trt.py:10` 在模組層 import `TRTEngine`，`mot17.py` 為了 libjpeg 載入順序先 import `detector_trt`）；`--cpp-threads` 以外不建 C++ detector |
| `saccade_eval_ext` | 同上 + `libopencv_imgcodecs` | 非 headline |
| `libsaccade_scan_plugin.so` | `libnvinfer.so.10`、`libcudart.so.13` | TRT Mamba head 的 plugin；headline 未用 |
| `saccade_node` | GStreamer／GLib、TRT、libtorch、OpenCV、cudart | 示範：一個 native 執行檔**本來就能**不帶 libpython 連結起來 |

在這個實測 build 裡，RUNPATH 全部是**絕對路徑**，指向 `<checkout>/build`、`<venv>/…/tensorrt_libs`、`<venv>/…/torch/lib`、`build/cuda_devlink`、`build/cuda_shim_root/lib64`；搬動 build 或 venv 就會斷。

`libtorch*.so` 是 C++ 函式庫，本身**不需要** libpython；只有 `libtorch_python.so` 需要。所以「保留 LibTorch」和「免裝 Python」並不衝突，只是套件會變大。

### 3.3 CUDA／TensorRT

| 元件 | 目前來源 | 版本 |
|:--|:--|:--|
| CUDA runtime／cuBLAS／cuFFT | venv 的 `nvidia-*` cu13 wheels（torch 帶進來的） | runtime 13.0.x；cuFFT 12.0；cuBLAS 13.1 |
| nvJPEG（decode） | torchvision 內建 | — |
| TensorRT | `tensorrt-cu12==10.16.1.11` wheel（`tensorrt_libs`） | **cu12 build 的 TRT 跑在 cu13 process 裡** |
| nvcc（只在 build 時） | `nvidia-cuda-nvcc==13.3.73` | build_install_only |
| Triton | `triton==3.6.0`（torch 依賴） | 首次 `torch.compile` 會 JIT；`triton/runtime/build.py:26-33` 要找系統 C 編譯器（`CC`／`clang`／`gcc`）來建 launcher stub → **headline 預設設定在 runtime 要有 C 編譯器**（有 cache 時除外） |
| Driver | 使用者系統 | 量測主機 driver 616.92（README headline 當時為 610.62） |

### 3.4 模型／engine 資產（headline s）

| 資產 | 格式 | 載入者 | 可攜性 |
|:--|:--|:--|:--|
| `models/yolo/yolo26s_backbone_640_best.engine` | TRT engine | Python `tensorrt` | 綁 GPU SM + TRT 版本；不能跨機器搬 |
| `runs/mamba_gt_v14replica_t3_t1/best.ckpt` | PyTorch pickle（`torch.load(weights_only=False)`，`mamba_gated_detector.py:726`） | Python | 需要 Python 反序列化；架構參數存在 `mamba_args` 裡 |
| `models/yolo/yolo26s.pt`（`--mamba-yolo-weights` 預設） | 只拿來做 SHA-256 lineage 比對（`mamba_gated_detector.py:729-736`） | Python | runtime 用不到它的內容，只用 hash |
| `models/yolo/mamba_head_best.pt` | TorchScript | C++ `MambaGatedDetector`（非 headline） | `scripts/model/export_mamba_head.py` 預設從 **legacy** `runs/mamba_gt_vgt_mamba_v14/best.ckpt` 匯出；[training_lineage_inventory.md](../research/training/training_lineage_inventory.md) 只記了 hash，沒有記 source ckpt |
| `models/yolo/mamba_head.engine` / `.onnx` | TRT／ONNX | `TRTMambaHead`（非 headline） | parity 未驗 |

### 3.5 設定

`configs/presets/mamba_whole_graph.yaml` 本身**不是**完整設定：生效值由 §2 第 0 列的四層預設疊出來，部分 tracker 開關還有只在環境變數裡的 escape hatch（例如 `SACCADE_ENABLE_DDA`，native 預設 ON，`tracker_gpu.cu:3256`）。Python 端在 `eval/`、`tracking/`、`temporal_yolo/` 三個目錄共讀 24 個不同的 `SACCADE_*`，native 在 `src/` 讀 6 個 `getenv`。

### 3.6 路徑假設

- `mot17.py` 用 `Path(__file__).parents` 找 checkout，並把 `src/`、`build/` 插進 `sys.path`。
- 套件內統一經 `saccade/paths.py`：`SACCADE_BUILD_PATH` → checkout `build/`；`SACCADE_TRACKEVAL_ROOT`。
- 模型、資料路徑相對於 **cwd**（`paths.runtime_input()`）。
- extension 靠 venv 裡的 `saccade_build.pth` 註冊（ADR 025）。

### 3.7 Subprocess／helper script

headline frame 路徑上**沒有** subprocess。只有非 runtime 的：`mlflow_logger.py` 和 `run_manifest.py` 會呼叫 `git`；`evaluator.py:2844` 的 D0 capture（opt-in）；`SACCADE_NV12_BUFFER=1` 時 `mot17.py` 會用 `LD_PRELOAD` `os.execve` 重啟自己（非 headline）；還有 Triton JIT 時會呼叫的 C 編譯器（§3.3）。

---

## 4. Python 依賴分類

定義：

- `runtime_required`：目前在 headline 路徑上擁有 runtime 語義，而且沒有經過 parity 驗證的 native 等價物。
- `replaceable_orchestration`：語義完全由 native 程式碼加上已解析的設定決定，Python 只負責膠合（持有 buffer、呼叫、capture），可以機械式換掉。
- `fallback_only`：只在非預設旗標或 import 失敗時才用到。
- `eval_research_only`：metrics、紀錄、研究功能；headline 關閉，或與「產出 tracks」無關。
- `build_install_only`：只在建置或產生資產時用到。

| Python 依賴 | 分類 | 理由 |
|:--|:--|:--|
| CPython 3.12 | `runtime_required` | 宿主 |
| `torch`（eager op、caching allocator、stream／event） | `runtime_required` | 擁有所有 GPU buffer 和 stream；ingest、FP filter、decode／top-k 都是 torch op |
| `torch.cuda.graph` / `make_graphed_callables`（`eval/_torch_graphs.py`、`eval/cuda_capture.py`） | `replaceable_orchestration` | capture policy 在 Python，但被 capture 的是 native kernel 或 TRT enqueue；native `cudaStreamBeginCapture` 可以取代。detect graph 裡的 torch op 屬於下一列 |
| `MambaDetectionHead`（`mamba_head.py`）＋ `mamba_gated_detector.py` whole-graph 函式 | `runtime_required` | **detector 語義**：head 前向、resize、anchor decode、top-k、座標還原 |
| `torch.compile` + `triton` | `runtime_required`（依預設設定） | headline 命令沒帶 `--no-compile`；compiled 與 eager 的輸出是否相同，本文沒有證據 |
| `torch.load` pickle ckpt + `yolo26s.pt` SHA 檢查 | `runtime_required` | 模型載入格式；lineage gate |
| `tensorrt` Python bindings（`TRTYoloBackbone`） | `replaceable_orchestration` | native `TRTEngine`（`trt_engine.cpp`）已存在；語義就是 engine 本身 |
| `torchvision.io.decode_jpeg` | `runtime_required` | 解碼後的像素決定下游一切；native 端目前只有 `cv::imread`（CPU，不同解碼器） |
| `nvidia.dali`（`DALIStreamerStream`） | `fallback_only` | `--no-gpu-decode` 才用；import 包在 try 裡 |
| PyTorch 版 NV12 路徑 | `fallback_only` | `SACCADE_NV12_BUFFER`，非 headline |
| `eval/stages.py` NMS／private continuation 呼叫 | `replaceable_orchestration` | 語義在 `pipeline.cpp`／`tracker_gpu.cu` |
| `detection_filters._fp_hard_reject_mask` | `runtime_required`（小） | 幾個 elementwise 比較；C++ twin 在 `seq_runner.cpp`，未驗 parity |
| `PyGraphedGMC` | `replaceable_orchestration` | 語義在 `gmc.cpp`／`gmc_kernel.cu` |
| `GPUByteTracker` / `GraphedTrackerUpdate` | `replaceable_orchestration` | 語義在 `tracker_gpu.cu`；**config→`set_params` 映射**是這層唯一帶語義的膠合，要一起搬走 |
| `evaluator.run_eval` frame loop、double-buffer 排程 | `runtime_required` | 決定 stage 順序、barrier、prior 取自哪一幀；沒有 native 宿主 |
| `GlobalTrackIdMapper`、MOT emit | `runtime_required` | 輸出 ID 空間與文字格式 |
| `interpolate_tracklets` + `pandas` | `runtime_required` | headline 開啟；會改變輸出列 |
| `yaml` + 四層設定解析 + `configure_runtime_env` | `replaceable_orchestration` | 可以預先解析成一份扁平設定（見 U2） |
| `cv2`（Python） | `eval_research_only` | import-time 被 `eval/pipeline.py` 帶進來；headline frame 路徑不用 |
| `pandas`（`post_merge` 其餘功能）、Cheb-GR／semantic relink／lifecycle merge／D0 capture | `eval_research_only` | headline 關閉；但 `post_merge` 模組 import 時會帶進 pandas |
| `eval/metrics.py`、TrackEval、motmetrics、MLflow、`run_manifest.py` | `eval_research_only` | 只在產出 tracks 之後 |
| `saccade/paths.py`、`saccade_build.pth` | `build_install_only` | extension 定位 |
| `pybind11`、`nvidia-cuda-nvcc`/`nvvm`/`crt`/`cccl`、cmake | `build_install_only` | `native-build` extra |
| `scripts/model/*`（`build_yolo.py`、`export_mamba_head_onnx.py`、`build_mamba_head_trt.py`、`export_mamba_head.py`） | `build_install_only` | 產生 engine／ONNX／TorchScript |

---

## 5. 目前由 Python 擁有、阻止純 native binary 的 runtime 語義

依「搬走的驗證成本」排序：

1. **Mamba detection head 前向**（PyTorch eager，預設經 `torch.compile`）。沒有經過驗證的 native 形式。
2. **Whole-detect 後段**：640 stretch resize（`F.interpolate` bilinear）、anchor decode、sigmoid／class max、top-k、座標縮放（`_postprocess_mamba_fixed`）。
3. **JPEG 解碼**：nvJPEG via torchvision。換解碼器可能改變像素。
4. **Ingest normalize**：`/255` 轉 f32 CHW。
5. **FP hard filter** mask。
6. **Frame loop 與 double-buffer 排程**：stage 順序、private continuation 用的是上一幀的 tracker state、event barrier。
7. **GPU 資源所有權**：caching allocator、stream／event、CUDA graph capture 與重新 capture 的 key（shape／原圖尺寸／input slot）。
8. **Config→native 參數映射**（`set_params` 等）與四層預設解析。
9. **Local→global track ID 映射與 MOT emit 格式**。
10. **Sequence-tail 插值**（pandas）。
11. **模型載入與 lineage gate**（pickle ckpt、`mamba_args` 決定網路形狀、SHA 比對）。

第 1、2 項決定 detector 語義，其餘是宿主與 I/O。tracker、NMS、private continuation、GMC、bridge relink 的**演算法**都已經在 native，不在這份清單上。

---

## 6. 最小 native boundary

能形成 `input → detector → tracker → output` 的最窄邊界：

```text
saccade_track (C++ executable, no libpython)
 ├─ ingest:   file/dir reader → nvJPEG decode → normalize            [新；取代 §5-3,4]
 ├─ detect:   TRTEngine(backbone) → head(native, §8 U1) → decode/top-k  [TRTEngine 已有；head、decode 待定]
 ├─ post:     PerceptionPipeline main NMS + private continuation + FP mask  [已有 native；FP mask 需 parity]
 ├─ gmc:      GMC::estimate_into_direct (cuFFT)                         [已有]
 ├─ track:    GPUByteTracker::update_into                               [已有]
 ├─ emit:     compact → D2H → global ID map → MOT rows                  [新；取代 §5-9]
 ├─ tail:     interpolate_tracklets                                     [新；取代 §5-10]
 └─ host:     frame loop + streams + native CUDA graph capture          [新；取代 §5-6,7]
config: 一份預先解析好的扁平設定檔                                     [取代 §5-8]
```

邊界上的決定：

- `saccade_tracking` 不該再連 `saccade_perception`（CMake 目標拆開），這樣 tracker 就不會被迫帶 libtorch、TRT、OpenCV。
- 如果 head 選 TRT 形式，整個執行檔可以**完全不依賴 libtorch**；選 LibTorch TorchScript 的話，要 ship `libtorch*.so`（不需要 Python，但很大）。
- OpenCV 只服務非 headline 的 GMC 模式，可以改成條件編譯。

---

## 7. 四種交付形式比較（不實作）

| 形式 | 可行的前提 | 實際阻礙 | 驗證成本 |
|:--|:--|:--|:--|
| **A. Pure native archive／installer**（單一靜態或近靜態執行檔） | §5 全部搬走；head 為 TRT；靜態連結 | TRT、cudart 在實務上都是動態連結（授權與 driver 相容性）；TRT engine 綁 SM＋TRT 版本；glibc baseline | 最高：除了 B 的全部驗證，還要做靜態連結的相容性測試 |
| **B. Native executable + shared libs + configs/models** | §5 全部搬走；`$ORIGIN` RUNPATH；明確的 SM 清單 | 與 A 相同的語義阻礙，但把 CUDA/TRT 留成「系統提供或另外決定」 | 中高：MOT txt 對 Python headline 的 parity（同一機器）、乾淨機器上的載入測試 |
| **C. Python wheel + bundled native extension** | ADR 025 §5 的五條前置條件 | **仍然需要 Python**；也就是 ADR 025 已經記錄的 wheel 線 | 中：屬於 ADR 025／#441 的範圍，不回答本 issue |
| **D. Transitional bundled-Python package**（例如內嵌 CPython + venv 的 archive） | 不需要搬任何語義 | 體積（torch＋cu13 wheels＋TRT＋triton 動輒數 GB）；Triton JIT 在 runtime 需要 C 編譯器（或關掉 compile，這會改變 kernel 集合）；絕對 RUNPATH 要改寫；§3.6 的 checkout 路徑假設（`mot17.py` 插 `sys.path`、`paths.py` 要 checkout）要處理；它是 eval harness，不是產品 CLI | 中：功能上只需比對同機器 MOT txt，但散佈、授權、更新成本高；滿足「使用者不需自行安裝 Python」，但 runtime 仍然包含 Python，不算 Python-free |

建議的目標形式是 **B**；D 可以當作不動語義的過渡方案：bundled-Python 滿足「終端使用者不需自行安裝 Python」這個部署目標（#465 的字面需求），但不滿足後續 phase 追求的、更強的「runtime 本身 Python-free」目標。兩個目標在本文中分開計算。要不要做 D 是 owner 的決定，本文不做決定。

---

## 8. 部署假設（現況紀錄，不是政策）

| 軸 | 現況 | 證據 |
|:--|:--|:--|
| OS | Linux x86_64 是唯一驗證過的平台。量測主機是 **WSL2**（kernel `6.18.x-microsoft-standard-WSL2`、Arch Linux userland）。原生 Windows 沒有 build 路徑（GStreamer／pkg-config／`.so`／POSIX 路徑），ADR 025 矩陣明確把它列在範圍外 | `uname`；ADR 025 §0 |
| glibc | build host 是 glibc 2.44；在這台機器建出來的 binary 要求執行端 glibc ≥ build 端 → 發佈版需要較舊的 baseline 建置環境 | `ldd --version` |
| GPU arch | `CMAKE_CUDA_ARCHITECTURES "native"`（`CMakeLists.txt:102`）；在這個實測 build 裡，產物只有 `sm_120` SASS、**沒有 PTX** → 只能在 CC 12.x 的 GPU 上跑，更舊的 GPU 不能 JIT。`native` 表示 arch 由 build host 的 GPU 決定 | `cuobjdump --list-elf` / `--list-ptx` |
| TRT engine | 在 build host 上建置，綁 SM＋TRT 10.16；換 GPU 型號就要重建 engine | TRT 慣例；preset 註解 |
| CUDA | runtime 13.0（wheel），nvcc 13.3；driver 必須支援 CUDA 13 | pyproject、ADR 025 |
| TensorRT | 10.16.1.11 **cu12** wheel 和 cu13 runtime 放在同一個 process | pyproject |
| CUDA／TRT 是 bundle 還是系統提供 | **未決定**。目前兩者都從 venv wheel 取得，不是系統安裝。本 PR 不做決定（hard boundary） | — |

---

## 9. 候選預編譯包目錄 layout（形式 B）

```text
saccade-<version>-linux-x86_64-cuda13-trt10.16-sm<list>/
├── bin/
│   └── saccade_track                 # 單一 CLI：input dir|video → MOT txt
├── lib/
│   ├── libsaccade_runtime.so         # tracker + pipeline + gmc + detector host（RUNPATH=$ORIGIN）
│   ├── libsaccade_scan_plugin.so     # 只有 head 走 TRT 時需要
│   └── vendor/                       # 只有在 owner 決定 bundle 時才存在：libnvinfer*, libcudart*, libcufft*, libnvjpeg*
├── share/saccade/
│   ├── configs/
│   │   └── mamba_whole_graph.resolved.json   # U2 產出的扁平設定；含來源 preset hash
│   ├── models/
│   │   ├── yolo26s_backbone_640.onnx          # 可攜；首次執行時在本機建 engine
│   │   └── mamba_head_s.onnx                  # 或 TorchScript（若 U1 選 LibTorch）
│   └── engine_cache/                          # 空；第一次執行寫入 <sm>-<trt>.engine
├── licenses/                          # Apache-2.0、NOTICE、第三方（TRT／CUDA 若 bundle）
├── MANIFEST.json                      # 每個檔案的 sha256、版本 pin、SM 清單、source commit
└── README.txt                         # driver 需求、GPU 需求、用法
```

設計重點：engine 不隨包散佈（不可攜），改為 ship ONNX＋首次執行時建 engine，或者按 SM ship 多份。這個取捨屬於 Phase C，這裡只標出來。

---

## 10. 後續單元（最小、依序；本 PR 不實作）

每個單元都是一個可以單獨 review 的 PR，驗收條件都是**與同機器 Python headline 的對照**，不發新的 benchmark claim。

| # | 單元 | 內容 | 驗收 | 依賴 |
|:--|:--|:--|:--|:--|
| **U1** | Headline head 的 native 形式＋parity gate | 對 `mamba_gt_v14replica_t3_t1` 產生 TRT head（沿用 `export_mamba_head_onnx.py` → `build_mamba_head_trt.py` + scan plugin）或 TorchScript（並記錄 lineage）；比對 whole-graph detection tensor（eager、compile 兩種）以及 7-seq MOT 輸出 | 事先宣告的容差；結果若不是 bit-exact，要讓 owner 決定能不能接受（會影響 benchmark claim）。**失敗 ⇒ verdict 升級成 `requires_runtime_redesign`** | — |
| **U2** | Resolved runtime config export | Python 把四層預設＋env hatch 解析成一份扁平 JSON；加 contract test：Python 從這份 JSON 跑的結果和從 preset 跑的逐位元相同 | 同機器 MOT txt byte-identical | — |
| **U3a** | Native post-detector host（replay） | C++ 執行檔讀 Python dump 出來的 detector 輸出 → FP mask → NMS／private continuation → GMC → tracker → emit；serial、eager，不做 graph | 同樣的 detection 輸入下，MOT txt 與 Python（serial 組態）比對；重現性是 per-configuration 的，判準要用該組態的 observed range | U2 |
| **U3b** | Native ingest + detector | nvJPEG decode、normalize、TRT backbone、U1 head、decode／top-k 搬進 U3a 的宿主 | 對 Python 的 detection tensor 做 parity；解碼像素差異要單獨量 | U1、U3a |
| **U4** | Native sequence tail | global ID 映射、`interpolate_tracklets`、MOT 文字格式（`:.2f`／`:.4f`） | golden test，byte-identical | U3a |
| **U5** | Native graph capture + double-buffer | 用 native `cudaStreamBeginCapture` 取代 torch capture；event barrier 排程 | 輸出 parity＋同 session 的 `--control` FPS 對照（只能說「相同或不同」，不能改 headline claim） | U3b |
| **U6** | Link surface 與可攜性 | 拆開 `saccade_tracking`／`saccade_perception`；讓 OpenCV 變成可選；`$ORIGIN` RUNPATH；明確的 SM 清單＋PTX；glibc baseline | `readelf` 沒有 libpython／torch（若 U1 選 TRT）；在乾淨容器裡能載入 | U3b |
| **—** | Phase C／owner 決策 | CUDA／TRT 要 bundle 還是用系統的（授權）；engine 分發策略；支援的平台（是否納入原生 Windows） | owner 決定；不屬於工程單元 | U6 |

U1 和 U2 互不依賴，可以平行做；U1 是整條線的**風險閘門**，應該先做。

---

## 11. 本文刻意沒有做的事

- 沒有執行 headline、沒有量 FPS 或精度、沒有驗證任何 parity。
- 沒有判斷 `torch.compile` 與 eager 的輸出是否相同（列為 U1 的一部分）。
- 沒有決定要不要 bundle CUDA／TRT、要不要支援原生 Windows、要不要做形式 D。
- 沒有修改任何 runtime 程式碼、preset、CMake 或 packaging 設定。
