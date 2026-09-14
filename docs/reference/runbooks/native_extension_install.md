# Native Extension 第三方安裝 Runbook

Date: 2026-09-14（[ADR 025](../../decisions/025-native-extension-delivery.md)）

`saccade` 的 Python package 用 pip 裝；tracker 的 C++/CUDA extension
`saccade_tracking_ext` **由你自己從 source checkout 建**，對準你要載入它的那個
venv，再把 build 目錄登記給該 venv。本文是**唯一受支援的第三方路徑**；沒有
install-time build，沒有 prebuilt wheel（原因與前置條件見 ADR 025 §2 / §5）。

> `pip install saccade` 單獨**不會**得到可用的 `GPUByteTracker`：它能 construct，但
> `is_cuda` 為 `False`（純 Python stub）。要 native tracker，走完 §2 全部步驟。

## 0. 支援矩陣

| 軸 | 值 |
|---|---|
| OS / arch | Linux x86_64 |
| Python | 3.12 |
| torch | 2.11.0（PyPI Linux 預設 wheel 即 `+cu130`） |
| TensorRT | 10.16.1.11（`tensorrt-cu12`，隨 `saccade` 自動安裝） |
| CUDA runtime | 13.0（torch 自帶的 `nvidia-*` cu13 wheels） |
| nvcc | 13.3.73（`saccade[native-build]`；**不**使用系統 `/opt/cuda` / `/usr/local/cuda`） |
| GPU | build host 的 SM（`CMAKE_CUDA_ARCHITECTURES=native`）；已驗證 sm_120 |
| host C++ | GNU g++，major ≤ 15（CMake 自動挑 `c++` 或 `/usr/bin/g++-N`；或設 `CUDAHOSTCXX`） |
| 系統套件 | `cmake` ≥ 3.18、OpenCV dev（驗證於 5.0.0）、GStreamer 1.0 dev（`gstreamer-1.0`、`gstreamer-app-1.0`、`gstreamer-video-1.0`）+ `pkg-config`、`git` |
| 網路 | 第一次 configure 需要（FetchContent 抓 TensorRT OSS headers 與 Eigen） |

矩陣外的組合未驗證、不支援。

## 1. 名詞

- `<venv>`：你要在裡面 `import saccade` 的 virtualenv。
- `<checkout>`：`git clone` 下來的 saccade 原始碼；只用來 build，`<venv>` 不需要
  能看到它。
- `<build>`：CMake build 目錄，**可以在任何地方**（不必在 `<checkout>` 內）。建好的
  `.so` 就留在這裡，這裡就是部署位置——它的 `RUNPATH` 指向 `<build>` 與 `<venv>`；
  搬走或刪掉 `<build>` 等於解除安裝，`<venv>` 換位置要重建。

## 2. 步驟

```bash
# 1. venv（Python 3.12）
python3.12 -m venv <venv>            # 或 uv venv --python 3.12 <venv>

# 2. Python package + build toolchain（非 editable）
<venv>/bin/pip install 'saccade[native-build] @ file://<checkout>'
#   來源之後也可以是 git URL 或 index；ADR 025 未選定，本文不假設。

# 3. 對準 <venv> 建 extension
cmake -S <checkout> -B <build> \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_NATIVE_TESTS=OFF \
      -DPYTHON_EXECUTABLE=<venv>/bin/python
cmake --build <build> --target saccade_tracking_ext --parallel

# 4. 登記 <build> 給 <venv>（持久；所有 import site 都走一般 import）
echo "<build>" > "$(<venv>/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/saccade_build.pth"

# 5. 驗證（在 <checkout> 以外的目錄跑）
cd /
<venv>/bin/python -c "import saccade, saccade_tracking_ext; t = saccade.GPUByteTracker(); print(t.is_cuda, saccade_tracking_ext.__file__)"
#   期望：True <build>/saccade_tracking_ext.cpython-312-x86_64-linux-gnu.so
```

`-DPYTHON_EXECUTABLE` 是**必要**輸入：CMake 從它推導 site-packages，nvcc、cuda
headers/libs、`tensorrt_libs`、`torch/lib`、pybind11 全部從那個 venv 拿，且 fail-closed
（nvcc 不在 `<venv>/…/nvidia/cu13/bin/` 就 `FATAL_ERROR`）。不給的話 CMake 退回
`<checkout>/.venv/bin/python3`——那是 repo 開發者的工作流，不是本文路徑。

### 2.1 每程序替代：`SACCADE_BUILD_PATH`

不想動 site-packages 時，可改用環境變數：

```bash
SACCADE_BUILD_PATH=<build> <venv>/bin/python your_script.py
```

`saccade.paths.build_dir()` 會回傳它（就算路徑不存在也照回，指錯要看得到），
`saccade.perception.tracking.tracker_gpu` 在一般 import 失敗時把它插進 `sys.path`。
**只有 tracker surface 保證吃這條**；eval / detector 模組的 top-level
`from saccade_tracking_ext import …` 只在 `tracker_gpu` 已先被 import 時受惠。要跑
eval / benchmark 請用 `.pth`。

### 2.2 其他 target

同一個 `<build>` 也能建 `saccade_eval_ext`、`saccade_perception_ext`、
`saccade_cheb_gr_online_ext`、`saccade_scan_plugin`、`saccade_media_ext`
（`cmake --build <build>` 不帶 `--target` 全建）。discovery 相同。ADR 025 只驗證了
tracker target；其他 target 的第三方支援未主張。

## 3. 驗證腳本

[`scripts/native/verify_consumer_install.sh`](../../../scripts/native/verify_consumer_install.sh)
在 checkout 之外把 §2 整條重做一次（新 venv → 非 editable install → cmake →
`--target saccade_tracking_ext` → `.pth` smoke → `SACCADE_BUILD_PATH` smoke →
兩者皆無的 control），並輸出 `report.json`（`NEEDED` 清單、載入路徑、`is_cuda`、
device）：

```bash
scripts/native/verify_consumer_install.sh --work /some/where/outside/the/checkout
```

smoke 會在下列任一情況失敗：`sys.path` 或 cwd 含 checkout、`saccade` 不是從
site-packages 載入、`saccade.paths.source_checkout_root()` 不是 `None`、`.pth` 模式下
`saccade.paths.build_dir()` 不是 `None`、extension 不是從指定 `<build>` 載入、
`GPUByteTracker().is_cuda` 為假、一次 `update()` 拋例外。改到 `CMakeLists.txt` 的
interpreter / toolchain 解析或 `saccade.paths` 之後請重跑。

## 4. 故障排除

| 症狀 | 原因 / 處置 |
|---|---|
| `Frozen venv CUDA compiler is absent: …/nvidia/cu13/bin/nvcc` | `<venv>` 沒裝 `native-build` extra，或 `-DPYTHON_EXECUTABLE` 指到別的 interpreter |
| `CUDA compiler escapes the frozen venv toolkit` | 環境有 `CUDACXX` 或 cache 裡有系統 nvcc；清 `<build>` 重 configure，不要設 `CUDACXX` |
| `No host C++ compiler within the frozen nvcc's supported GNU range (<= 15)` | 裝 `g++-15`（或更低），或 `CUDAHOSTCXX=/usr/bin/g++-15` |
| `Could not find a package configuration file provided by "OpenCV"` / `gstreamer-1.0 not found` | 裝系統 OpenCV / GStreamer dev 套件（發行版套件名各異） |
| `ImportError: libnvinfer.so.10: cannot open shared object file` | `<venv>` 少了 `tensorrt-cu12`（它是 default 相依，通常表示 `saccade` 不是裝在這個 venv），或 `<venv>` 搬過位置——`RUNPATH` 是絕對路徑，重建 |
| `import saccade` 成功但 `GPUByteTracker().is_cuda` 為 `False` | `<build>` 未登記（沒有 `.pth`、沒有 `SACCADE_BUILD_PATH`），或 `.so` 是為另一個 venv / 另一個 Python 版本建的 |
| `undefined symbol` 於 import | `<venv>` 的 torch 或 TensorRT 版本與 build 時不同（例如之後 `pip install -U torch`）；重建 |
| 換了 GPU 世代 | `native` arch 只含 build host 的 SM；重建 |

## 5. Repo 開發者

Repo 內的工作流不變：`uv sync`（`dev` group 含 `native-build`）→
`scripts/native/rebuild.sh`（對 `<checkout>/.venv` build 進 `<checkout>/build/`，
並寫 `.pth`）。`SACCADE_BUILD_PATH` 與 `paths.build_dir()` 的優先序見
[`src/README.md`](../../../src/README.md)。

## 相關

- [ADR 025: `saccade_tracking_ext` 的第三方交付形式](../../decisions/025-native-extension-delivery.md)
- [scripts/native/README.md](../../../scripts/native/README.md)
- `tests/contract/test_package_native_delivery.py`、`tests/contract/test_package_dependency_surface.py`
