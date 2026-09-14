<!-- doc-status: accepted -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-14 -->
<!-- doc-module: cross -->

# ADR 025: `saccade_tracking_ext` 的第三方交付形式

## Status

**Accepted** (2026-09-14)

**Terminal verdict: `consumer_cmake_build`** — native extension 由**使用者自己**從
source checkout 用 CMake 建，對準**它將被載入的那個 venv**；`pip install saccade`
只交付 Python package 與 load-time 相依，**不**交付可用的 native tracker。

對照時點 2026-09-14，`main` = `069a5be5`（PR #414 / #415 之後）。本文接續
public package 線第二步（PR #414，core / extras 拆分）留下的「native substrate 已分類、
未決定」。

本文**不**決定 Python package 走 PyPI 還是 git（[§6](#6-未決不授權)）、**不**改
tracker / detector 行為、**不**改 benchmark / eval 語義、**不**清理無關相依。

---

## 1. Context：extension 現在長什麼樣

`saccade_tracking_ext` 是根 [`CMakeLists.txt`](../../CMakeLists.txt) 的
`pybind11_add_module`，把 `saccade_tracking` + `saccade_perception` 兩個 static
library 連進一個 `.so`。以 `main` 上的建置產物量到的事實（`readelf -d`、
`cuobjdump --list-elf`，2026-09-14）：

| 面向 | 觀測值 | 對交付形式的含義 |
|---|---|---|
| 直接 `NEEDED` | `libcudart.so.13`、`libcublas.so.13`、`libcublasLt.so.13`、`libcufft.so.12`（venv `nvidia/cu13`）；**`libnvinfer.so.10`**（venv `tensorrt_libs`）；`libtorch.so`、`libtorch_cpu.so`、`libtorch_cuda.so`、`libc10.so`、`libc10_cuda.so`（venv `torch/lib`）；**十個 `libopencv_*.so.500`**（**系統** OpenCV 5.0，含 CUDA modules，再拉 `libcudnn.so.9` 等系統庫） | 三組 pip wheel 相依可以用版本 pin 綁死；系統 OpenCV 的 C++ ABI **不能** |
| device code | `CMAKE_CUDA_ARCHITECTURES "native"` ⇒ 只有 build host 的 SM（本機 `sm_120`） | 產物只對同 SM 的 GPU 有效 |
| `RUNPATH` | build 目錄、build 目錄的 `cuda_devlink/`、該 venv 的 `tensorrt_libs` 與 `torch/lib`，全是絕對路徑 | 產物綁定「這個 build 目錄 + 這個 venv」 |
| Python ABI | `cpython-312-x86_64-linux-gnu` | `requires-python = ">=3.12, <3.13"` 已相符 |
| 載入方式 | top-level module（不在 `saccade/` package 內）；`import saccade_tracking_ext` 靠 `sys.path`：site-packages 的 `saccade_build.pth`，或 `saccade.paths.build_dir()`（`SACCADE_BUILD_PATH` → checkout `build/` → `None`）由 `tracker_gpu.py` 在 import 失敗時 fallback | discovery 已經有一條不依賴 checkout 的路（PR #409） |
| build 需求 | venv 的 nvcc 13.3（`nvidia-cuda-nvcc` pip wheel，fail-closed 不接受系統 CUDA，issue #214）、GNU host C++ ≤ nvcc crt 上限、系統 OpenCV dev、系統 GStreamer dev + `pkg-config`、cmake ≥ 3.18、第一次 configure 需網路（FetchContent：TensorRT OSS headers、Eigen） | 這些是 host 條件，pip 給不了 |

---

## 2. 三種交付形式的比較

| 形式 | 能不能現在做 | 為什麼 |
|---|---|---|
| **(a) install-time source build**（PEP 517 backend 在 `pip install` 時跑 CMake） | ✗ 不採 | extension 連 libtorch C++ ABI，configure 要 `import torch` ⇒ 只能 `--no-build-isolation` 或把 torch 塞進 `build-system.requires`；加上系統 OpenCV / GStreamer dev 缺一就 fail ⇒ 純 Python 的部分（`ReorderingBuffer`、`EvalConfig`、`paths`）會被 native build 失敗一起拖垮。這是把 §1 的 host 條件變成整個 package 的安裝條件 |
| **(b) prebuilt wheel** | ✗ 現在不可能；前置條件列在 [§5](#5-wheel-線的前置條件不是決定) | `native` arch、系統 OpenCV 5 動態連結（十個 SONAME + CUDA modules + cudnn）、絕對 `RUNPATH` 三件事任一都讓 wheel 只能在 build host 上跑。把 OpenCV 靜態進去或換掉是另一條 native 工程線，不在本 ADR 範圍 |
| **(c) consumer-run CMake after install** | ✓ **採用** | §1 的每一項 host 條件都由使用者的機器滿足；build 對準使用者的 venv，`RUNPATH` / SM / torch / TensorRT 自然一致；package 的 Python 部分不受 native 影響。代價是使用者要有 source checkout 與 C++ toolchain |

(c) 就是 repo 內部一直以來的做法（`scripts/native/rebuild.sh`），差別只在**把「哪個
venv」從 `<checkout>/.venv` 的隱含假設變成明確輸入**（`-DPYTHON_EXECUTABLE`），
並把它寫成受支援的第三方路徑。

---

## 3. Decision

### 3.1 交付形式

1. **`pip install saccade`** 交付 Python package + load-time 相依（`numpy`、
   `torch==2.11.0`、`tensorrt-cu12==10.16.1.11`）。**它單獨不會產生可用的
   `GPUByteTracker`**：`saccade.GPUByteTracker()` 可以 construct，但
   `is_cuda == False`（`tracker_gpu.py` 既有的 stub fallback，本 ADR 不改它）。
2. **native extension 由使用者建**：`pip install 'saccade[native-build]'` 把
   compiler line 裝進**目標 venv**，然後
   `cmake -S <checkout> -B <build> -DPYTHON_EXECUTABLE=<venv>/bin/python`、
   `cmake --build <build> --target saccade_tracking_ext`。
3. **placement**：`.so` 留在 `<build>`，不複製進 `saccade/` package、沒有
   `cmake --install`。`<build>` 就是部署位置（`RUNPATH` 指向它與 venv）；搬走或刪掉
   它等於解除安裝。
4. **discovery**（兩條，都不含 checkout）：
   - 持久：在目標 venv 的 site-packages 寫 `saccade_build.pth`，內容一行 `<build>`。
     所有 `import saccade_tracking_ext` 的 call site 都走一般 import。
   - 每程序：`SACCADE_BUILD_PATH=<build>`。只有 tracker surface
     （`tracker_gpu.py` 的 fallback）保證吃它；eval / detector 路徑的 top-level
     import 只在 `tracker_gpu` 已先被 import 時受惠。runbook 以 `.pth` 為主。
5. **支援矩陣（v1）**：

   | 軸 | 支援值 | 來源 |
   |---|---|---|
   | OS / arch | Linux x86_64 | 只有這裡有 build 與驗證 |
   | Python | 3.12（`cpython-312`） | `requires-python` |
   | torch | 2.11.0（PyPI Linux 預設即 `+cu130`） | `dependencies` pin；libtorch ABI |
   | TensorRT | 10.16.1.11（`tensorrt-cu12`） | `dependencies` pin；`libnvinfer.so.10` |
   | CUDA runtime | 13.0（torch 拉的 `nvidia-cuda-runtime` 等 cu13 wheels） | link 目標 |
   | nvcc | 13.3.73（`native-build` extra；nvcc/nvvm/crt 同一 matched line） | issue #214 |
   | GPU | build host 的 SM（`native`）；已驗證 `sm_120` | `CMAKE_CUDA_ARCHITECTURES` |
   | host C++ | GNU，major ≤ crt 上限（13.3 crt 為 15）；CMake 自動選 `c++` 或 `/usr/bin/g++-N` | `CMakeLists.txt` |
   | 系統庫 | OpenCV dev（驗證於 5.0.0）、GStreamer 1.0 dev + `pkg-config`、cmake ≥ 3.18 | `find_package` / `pkg_check_modules` |

   矩陣外的組合（其他 Python 版本、其他 torch / TensorRT、aarch64、Windows、
   容器內無 GPU 的 build）**未驗證、不支援**，不是「應該也可以」。

### 3.2 分類

| 項目 | 分類 | 落點 | 證據 |
|---|---|---|---|
| `tensorrt-cu12` | **runtime loader dependency** | `[project].dependencies` | `libnvinfer.so.10` 是 extension 的直接 `NEEDED`，經 `RUNPATH` 解析到該 venv 的 `tensorrt_libs`；tracker 路徑不 `import tensorrt` 但 `.so` 載不起來 |
| `nvidia-cuda-nvcc` / `nvidia-nvvm` / `nvidia-cuda-crt` / `nvidia-cuda-cccl` | **build-only** | `[project.optional-dependencies].native-build` | `src/saccade` 無 import；`NEEDED` 無 `libnvvm` 等；只被 `CMakeLists.txt` 消費 |
| `pybind11` | **build-only** | 同上 | header-only；`NEEDED` 無對應 SONAME |
| native `.so` placement | build 目錄即部署位置 | 見 §3.1 (3) | `RUNPATH` 含 `<build>` 與 `<build>/cuda_devlink` |
| discovery | `.pth`（主）／`SACCADE_BUILD_PATH`（每程序） | 見 §3.1 (4) | 兩條都在 [§4](#4-驗證) 實跑 |
| `pip install saccade` 單獨 ⇒ 可用 `GPUByteTracker`？ | **否**，by design | README / runbook 明寫 | §4 的 control run：無 `.pth`、無 env ⇒ `is_cuda == False` |

`dev` group 納入 `native-build`，所以 `uv sync` 仍是完整 repo 環境、`rebuild.sh`
不變；`tests/contract/test_package_dependency_surface.py` 把「build-only extra 不擁有
任何 module」寫成規則，`tests/contract/test_package_native_delivery.py` 把本節綁到
`pyproject.toml` / `CMakeLists.txt` / runbook / 驗證腳本。

### 3.3 CMake 的唯一改動

`Python_ROOT_DIR` 從寫死的 `${PROJECT_SOURCE_DIR}/.venv` 改為由目標 interpreter 的
`sys.prefix` 推導。這是 supported path 上僅存的 checkout-relative 假設；
`<checkout>/.venv/bin/python3` 仍是**沒給** `-DPYTHON_EXECUTABLE` 時的 fallback
（repo 工作流），不是第三方路徑的一部分。

---

## 4. 驗證

[`scripts/native/verify_consumer_install.sh`](../../scripts/native/verify_consumer_install.sh)
在 checkout **之外**重做整條路徑：新 venv → 非 editable `pip install
'saccade[native-build]'` → `cmake` 對準該 venv → `--target saccade_tracking_ext` →
`.pth` 與 `SACCADE_BUILD_PATH` 各跑一次 smoke → 兩者皆無的 control。smoke 自己
fail-closed：`sys.path` 與 cwd 不得含 checkout、`saccade.__file__` 必須在
site-packages、`paths.source_checkout_root() is None`、`.pth` 模式下
`paths.build_dir() is None`、extension 必須從指定的 build 目錄載入、
`GPUByteTracker().is_cuda` 為真、一次 `update()` 不拋例外。

2026-09-14 於本機（checkout `069a5be5` + 本 PR 工作樹）實跑一次，exit 0：

| 步驟 | 結果 |
|---|---|
| venv | `uv venv --python 3.12` → CPython 3.12.13 |
| install | 39 packages（`torch==2.11.0`、`tensorrt-cu12==10.16.1.11`、`nvidia-cuda-nvcc==13.3.73`、`pybind11==3.1.0` …）；`saccade` 從 checkout 建 wheel 安裝，非 editable |
| configure | `Found CUDAToolkit … venv/nvidia/cu13 (13.3.73)`、`Found OpenCV: /usr (5.0.0)`、gstreamer 1.28.7、`Found Python3: …/venv/bin/python (3.12.13)`、host compiler `/usr/bin/g++-15`；約 1 分鐘（含 FetchContent） |
| build | `saccade_tracking_ext.cpython-312-x86_64-linux-gnu.so`，8 個 `sm_120` cubin；`-j32` 約 80 秒 |
| smoke（`.pth`） | `is_cuda: true`、extension 從 `<work>/native` 載入、`build_dir() is None`、`torch 2.11.0+cu130`、RTX 5070 Ti Laptop (12,0) |
| smoke（`SACCADE_BUILD_PATH`） | 同上，`build_dir() == <work>/native` |
| control（皆無） | `is_cuda: false` |
| `NEEDED` | 與 §1 逐項相同（含 `libnvinfer.so.10`，無 `libnvvm*`） |

這是**一台 host、一組矩陣值**的驗證；矩陣其他格子沒有跑過就沒有主張。

---

## 5. Wheel 線的前置條件（不是決定）

要開 prebuilt wheel，下列每項都得先成立，且各自是獨立工程：

1. `CMAKE_CUDA_ARCHITECTURES` 從 `native` 改為明確的 SM 清單（fat binary），並量
   build 時間 / 體積。
2. 系統 OpenCV 的動態連結消失：靜態連結最小 OpenCV、或把用到的功能移出
   extension。十個 `libopencv_*.so.500` + CUDA modules + `libcudnn.so.9` 沒有
   manylinux 路徑。
3. `RUNPATH` 改為 `$ORIGIN`-relative，並決定 torch / TensorRT 庫如何被找到（不能
   bundle 進 wheel，只能依賴同 venv 的 pin）。
4. 一個 wheel 對應一組 (Python, torch, TensorRT, CUDA) pin；每次 bump 都是新 wheel。
5. `auditwheel` / manylinux 相容性與 GStreamer 相依（media ext）的處置。

本 ADR 只把這五條寫下來；達成任一條不改變本文結論，五條全達成才值得開新 ADR。

---

## 6. 未決（不授權）

- **PyPI vs git**：native 交付已定形（消費者必有 source checkout），Python package
  的分發來源是獨立問題，本文不選。§4 的驗證用 `file://<checkout>` 安裝，兩種來源
  都能重跑同一腳本。
- `saccade_eval_ext` / `saccade_perception_ext` / `saccade_cheb_gr_online_ext` /
  `libsaccade_scan_plugin.so` / `saccade_media_ext`：同一 CMake、同一 build 目錄、
  同一 discovery；本文只驗證 tracker target，其他 target 的第三方支援矩陣未主張。
- 讓 GStreamer / OpenCV 對 tracker-only build 變成 optional：會縮小 host 條件，但
  是 CMake 結構改動，未做。

---

## 7. Consequences

- `pyproject.toml`：build toolchain 移入 `native-build` extra；`tensorrt-cu12` 留在
  default 並寫明理由；`dev` group 納入 `native-build`。`uv.lock` 同 260 packages、
  同版本，只有 requirement 出處移動。
- `CMakeLists.txt`：`Python_ROOT_DIR` 推導改動（§3.3）；缺 nvcc 的錯誤訊息指向
  `saccade[native-build]`。
- `environment` 軸（recipe 半邊）因此移動 ⇒ runtime coordinate 依
  [runbook](../reference/runbooks/runtime_identity_republication.md) 重新出版
  （stacked provenance PR，atomic co-land）。
- 新增 [`docs/reference/runbooks/native_extension_install.md`](../reference/runbooks/native_extension_install.md)
  （第三方安裝路徑）、`scripts/native/verify_consumer_install.sh`、
  `tests/contract/test_package_native_delivery.py`。
- README 的 native 段落改為指向 runbook。

## 相關

- [ADR 023](023-ultralytics-runtime-decouple.md) — default install 去 AGPL 的前提
- PR #409 / #410 — `saccade.paths` 唯一 resolver；PR #414 / #415 — core / extras 拆分
- issue #214 — 凍結 build-CUDA toolchain
