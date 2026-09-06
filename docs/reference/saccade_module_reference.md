# Saccade 模塊化前參考筆記

> 目的：整理目前 Saccade 專案已觀察到的結構特徵、可重用能力與模塊化前可能遇到的問題。  
> 本文不是設計方案，也不定義未來 API；僅作為後續拆分與封裝時的背景資料。

## 1. 專案目前的基本形態

Saccade 已經不是單一腳本式研究專案，而是具有明確 Python package 與 C++/CUDA native extension 的混合型系統。

目前 Python 採標準 `src/` layout：

```text
src/saccade/
├── api/
├── cognition/
├── media/
├── perception/
├── pipeline/
├── resource/
└── storage/
```

`pyproject.toml` 已將 package root 設為 `src/`，套件名稱為 `saccade`。

這代表「包成模塊」不需要從零建立 package 結構；真正需要處理的是邊界、依賴與 public surface。

---

## 2. Tracking 核心已具備相對獨立的入口

目前：

```python
from saccade.perception.tracking import GPUByteTracker
```

已是可用的 Python import surface。

`src/saccade/perception/tracking/__init__.py` 目前直接 export：

```text
GPUByteTracker
ReorderingBuffer
```

native 端則已有：

```text
include/tracking/tracker_gpu.hpp
src/tracking/tracker_gpu.cu
src/tracking/tracker_gpu_python.cpp
```

並透過 pybind11 建立：

```text
saccade_tracking_ext
```

所以 tracker 本體已經具有：

- C++ public class
- CUDA implementation
- Python binding
- Python wrapper

這是目前最明顯可以獨立封裝的能力區塊。

---

## 3. Production tracker 已明顯偏向 geometry-first

目前 production 方向已經從 ReID-centered 轉為 geometry-first。

核心 identity / association 能力主要來自：

- Kalman motion
- GMC-compensated motion
- geometry gating
- stability terms
- relink / bridge
- lost-track lifecycle
- short-occlusion recovery
- birth / confirm / death policy
- GPU-side association

ReID 已被降為 experimental / optional 路徑，不再是 production tracker 的核心依賴。

這對模塊化是有利的，因為 production tracker 的核心功能已比早期更集中，不必把整套 embedding / crop / ReID bank 一起綁進主要使用路徑。

---

## 4. 效能已足以成為模塊本身的特徵

目前專案已經有多組可重現的高吞吐結果。

已記錄的 headline / production-family 結果包括：

- geometry-first 約 350 FPS 級
- whole-graph 約 290 FPS 級
- 近期 full-suite paired A/B 中約 322 → 329 FPS
- fused ingest 單一改動帶來約 +2.15% throughput
- 多次測試中品質輸出可以做到 bit-identical / deterministic

因此這個模塊不只是「功能可用」，而是具有很明確的工程定位：

> GPU-first、低延遲、高吞吐 multi-object tracking / video perception component。

但不同 benchmark 的 measurement boundary 不完全相同，之後不能只拿一個 FPS 數字當統一規格。

---

## 5. Determinism 是目前很重要的特徵

在 `--no-gpu-decode` 等受控條件下，部分 eval 已做到：

- 多次重複品質指標 stdev = 0
- MOT output byte-identical
- A/B 可用 N=1 做品質比較
- production optimization 可用 paired ABBA 做 throughput 驗證

這代表目前 tracker / evaluation stack 的可重現性已經高於一般研究原型。

這是未來封裝時很有價值的既有資產，因為可以用來做：

- regression
- compatibility validation
- native extension verification
- packaging 前後等價性檢查

---

## 6. Python tracking wrapper 仍混有大量研究時期功能

`src/saccade/perception/tracking/tracker_gpu.py` 並不是單純的 thin wrapper。

目前同一檔案中還包含：

- appearance sample
- ReID frame stats
- ReID observation
- track appearance bank
- EMA embedding
- geometry-clean / suspect-box logic
- experimental velocity-aligned appearance update
- 多種 optional / historical behavior

因此雖然 `GPUByteTracker` 已可 import，但目前 wrapper 的責任範圍仍偏大。

這表示：

> 「已經有 import surface」不等於「已經有乾淨 module boundary」。

---

## 7. Native extension 的載入方式目前帶有 repo-local 假設

`tracker_gpu.py` 載入 `saccade_tracking_ext` 失敗時，會嘗試：

```text
<repo>/build
```

並手動插入 `sys.path`。

這對開發 repo 很方便，但透露出目前 extension loading 還依賴：

- repository layout
- 本地 `build/`
- 開發環境中的 native artifact

若未來離開 repository 安裝，這個假設可能不成立。

---

## 8. Native extension 載入失敗時目前可能靜默退化

目前如果 `saccade_tracking_ext` 最終 import 失敗，Python 端會建立 fallback stub。

其中 fallback `update()` 直接回傳空 list。

也就是存在這種可能：

```text
native extension unavailable
        ↓
Python import 仍成功
        ↓
tracker 看似可建立
        ↓
update() 回空結果
```

這對研究環境中的 optional import 有便利性，但對獨立模塊而言可能形成很難診斷的 failure mode。

這是目前最需要特別記住的 packaging/runtime 問題之一。

---

## 9. 目前 package runtime dependencies 非常寬

`pyproject.toml` 的主 dependencies 同時包含：

- torch / torchvision
- TensorRT
- ONNX Runtime GPU
- Ultralytics
- OpenCV
- GStreamer Python binding
- Transformers
- Accelerate
- timm
- FastAPI / Uvicorn
- Redis
- ChromaDB
- SQLAlchemy / psycopg2
- MLflow
- Optuna
- MOT metrics
- pycocotools
- pytest / ruff / mypy
- CUDA compiler/toolchain packages
- 其他 research / server / storage 套件

所以目前的 `saccade` 安裝單位實際上綁住了：

```text
tracker
+ detector
+ media
+ server
+ research
+ experiment tooling
+ evaluation
+ development tooling
```

這是目前「作為一個獨立 tracking module」最大的結構性問題之一。

---

## 10. 開發依賴與 runtime 依賴目前混在一起

目前主 dependencies 中也有：

- pytest
- pytest-cov
- ruff
- mypy
- pytest-asyncio
- pytest-anyio

這些對開發與 CI 很重要，但不是 tracker runtime 本身需要的功能。

因此現有 dependency declaration 比「實際執行 tracker 所需依賴」大很多。

---

## 11. CUDA / Torch / TensorRT 版本綁定偏強

目前環境明確 pin：

```text
Python >=3.12,<3.13
torch==2.11.0
torchvision==0.26.0
TensorRT 10.16.1.11
CUDA toolchain 13.x
```

而且 native extension 的 build/runtime identity 已經證明：

- toolchain 版本會影響 artifact identity
- build directory 可能影響 binary hash / build-id
- runtime loaded closure 並不單純等於 source tree
- CUDA / Torch / native extension 的組合需要精確匹配

所以 packaging 真正困難的地方很可能不是 Python wheel 本身，而是：

> 如何穩定交付 native CUDA extension 與其 ABI / runtime matrix。

---

## 12. CMake 目前負責多個 native component

目前 CMake 不只建 tracker，也包含其他 media / native targets。

tracking extension 本身透過：

```text
pybind11_add_module(saccade_tracking_ext ...)
```

建立，並連到：

```text
saccade_tracking
CUDA::cudart
```

這表示 native tracker 本身已有可識別 build target，但整個 build system 還是 project-level build，而不是明確的 standalone distribution build。

---

## 13. Detector 與 Tracker 在研究 repo 中耦合較深，但概念上已可分離

目前 benchmark / production pipeline 常以：

```text
detector
→ preprocess
→ tracker
→ postprocess
```

一起測量。

但 tracker 本身已有 detections-based 更新介面，因此從功能角色上，它不是一定要綁特定 detector 才能存在。

需要注意的是：

- 部分 geometry / lifecycle tuning 來自目前 detector 分布
- 某些 threshold 與 detection characteristics 可能有 dataset / detector dependence
- production benchmark 數字不能直接等同「任何 detector 接上去都相同」

所以「程式介面可分離」與「性能/accuracy claim 可分離」是兩回事。

---

## 14. 部分 tracker 行為仍具有大量 configuration surface

目前 runtime / evaluation stack 的 resolved parameter 數量已超過 400。

其中很多不是 end-user 需要知道的參數，而是：

- research knobs
- diagnostic switches
- historical compatibility knobs
- experimental association / ReID switches
- evaluator / runtime controls
- test / tracing / measurement controls

如果未來把這些全部視為模塊 API，實際上會把研究 harness 一起暴露出去。

因此目前要記住的一個現況是：

> Saccade 的 internal configurable surface 很大，但真正穩定的 production surface 其實小得多。

---

## 15. Evaluation path 與 runtime path 仍有部分特殊邏輯

近期已經出現一個實例：

原先 frame-budget 文件引用了 `evaluator.py` 中一個 expression，但該 call site 在 headline preset 下實際不可達；真正執行的是 `stages.py` 中的 live ingest path。

這說明目前 codebase 中：

- evaluator path
- workbench path
- headline path
- optional runtime path

之間不是完全等價。

模塊化時若直接從 eval code 抽 interface，可能會把 dead / non-production path 當成正式行為。

---

## 16. Output-layer postprocessing 仍有研究性質

目前 offline handover、Cheb-GR tracklet merge 等操作是在 sequence-level output 上額外執行。

近期實驗顯示：

- handover only 有小幅正收益
- merge only 的收益更高
- handover + merge 並不互補
- stage order 本身會改變結果
- postprocess wall-time 並沒有完全納入 tracker FPS

因此這些操作目前比較像：

```text
tracker core
+
optional sequence-level repair
```

而不是一個完全不可拆的單體演算法。

---

## 17. 目前最強的 accuracy 結果未必就是 production core 本身

例如近期 merge-only 結果可達：

```text
IDF1 81.3
IDs 313
```

高於 base：

```text
IDF1 80.4
IDs 344
```

但該結果來自 output-layer postprocess，而且目前仍屬研究 / draft context。

所以未來如果說「Saccade module 的 accuracy」，必須先釐清：

- core online tracker
- online tracker + optional postprocess
- full evaluation stack

否則同一個專案會出現多個合理但不同的 headline。

---

## 18. Runtime identity / provenance 基礎設施已經很成熟，但也很重

目前已有：

- runtime identity axes
- implementation digest
- environment digest
- decision surface digest
- runtime-input binding
- behavior probe
- source attestation
- experiment manifest
- asset producer registry
- fail-closed checks

這些對研究證據非常有價值。

但如果要把 tracker 當一般模塊使用，這些機制未必全部需要成為 runtime dependency。

目前應把它理解成：

> Saccade repo 擁有很強的研究 / audit infrastructure，但它與最小可用 tracking functionality 並不是同一層。

---

## 19. 目前測試體系是模塊化的重要資產

目前已有多層測試：

### Python
- pytest
- pytest-cov
- async tests
- contract tests

### Native
- CMake
- CTest
- CUDA / C++ tests

### 靜態 / 契約
- ruff
- mypy
- project-specific contracts
- runtime identity staleness
- source attestation
- structure checks

而且 pre-push 測試量已超過 3000 tests。

這表示後續若做 packaging，最大優勢是：

> 可以用既有 regression / contract system 驗證拆分前後是否真的保持行為。

---

## 20. 目前 repository 還帶有大量 research-only surface

例如：

```text
docs/research/
docs/modules/semantic/research/
results/
runs/
out/
scripts/eval/
scripts/train/
scripts/provenance/
```

其中很多是：

- study declarations
- evidence
- contracts
- benchmark tooling
- experimental sweeps
- research-only scripts
- model / claim governance

它們是 repo 的重要價值，但不是 tracker 作為 library 時一定要一起存在的部分。

---

# 目前看到的主要問題清單

## A. Packaging / distribution

- native CUDA extension 尚未證明可以脫離 repo-local `build/` 安裝
- extension loader 帶有 repository path fallback
- CUDA / Torch / TensorRT version matrix 很窄
- project-level CMake 與 standalone native package 邊界尚未分開

## B. Dependency surface

- runtime / research / server / evaluation / dev dependencies 混在同一組
- 安裝 `saccade` 目前會帶入很多與 tracker 無關的 package
- Python 版本鎖定 3.12.x

## C. Failure semantics

- native extension 缺失時存在 silent stub fallback
- tracker 可能表面正常、實際輸出空結果
- optional path 很多，錯誤組態可能不容易從 import 階段看出

## D. Public API

- 已經有 `GPUByteTracker` import surface
- 但 wrapper 內部責任仍很廣
- internal configuration surface 遠大於真正 production surface
- eval / runtime / research knobs 混合

## E. Runtime behavior boundary

- 不同 execution path 可能經過不同 call site
- evaluator path 不一定等於 production path
- benchmark 數字存在 decode / postprocess / full-suite 等不同 measurement boundary

## F. Algorithm boundary

- geometry-first core 已相對穩定
- ReID 已降為 optional / experimental
- sequence-level handover / merge 屬額外 repair
- core tracker 與 output-level repair 的能力邊界仍需在使用時明確區分

## G. Claims / benchmark

- 不同 benchmark command 的 FPS 不能直接互比
- MOT17 上的 accuracy 不代表外部資料集泛化
- detector 與 tracker 雖能在程式上分離，但現有 accuracy tuning 並非 detector-independent
- current best accuracy 可能依賴 optional postprocess

---

# 可以視為既有優勢的部分

目前 Saccade 已具有幾個對模塊化很有利的條件：

1. 已使用標準 Python `src/` package layout。
2. tracker 已有 Python import surface。
3. native tracker 已有 C++ public class。
4. 已有 pybind11 CUDA extension。
5. geometry-first production core 已經相對明確。
6. ReID 不再是核心依賴。
7. 已有高吞吐實測。
8. deterministic evaluation 很成熟。
9. regression / contract tests 很完整。
10. 大量 runtime / provenance 問題已經被實際踩過並留下記錄。

換句話說，目前的主要問題不是「沒有可以包的核心」，而是：

> 核心已經存在，但仍被整個研究系統、開發環境、native build 假設與大量 configuration surface 包圍。

---

# 後續查看時建議優先參考的檔案

```text
pyproject.toml

src/saccade/perception/tracking/__init__.py
src/saccade/perception/tracking/tracker_gpu.py

include/tracking/tracker_gpu.hpp
src/tracking/tracker_gpu.cu
src/tracking/tracker_gpu_python.cpp

CMakeLists.txt

docs/decisions/019-demote-reid-geometry-first-production-tracker.md
docs/reference/math_model.md
docs/reference/benchmarks/frame_budget_20260905.md
docs/reference/benchmarks/reid_handover_ablation_20260808.md

scripts/pre_push.sh
scripts/tools/h2_path_partition.py
```

這些檔案大致分別代表：

```text
package / dependency
Python wrapper
native API
native implementation
Python binding
build
production architecture decision
implementation model
performance boundary
ReID / postprocess evidence
regression gate
runtime path classification
```

---

## 一句話總結

Saccade 現在已經具備「高效 GPU tracker 模塊」所需的大部分核心能力；真正尚未收斂的是 **distribution boundary、dependency boundary、native extension delivery、failure semantics 與 public/runtime surface 的分離**。
