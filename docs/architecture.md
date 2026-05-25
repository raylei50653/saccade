# Saccade 系統架構說明書

本文件描述 **目前穩定的系統形狀與責任邊界**。它不是實驗日誌，也不是待辦清單。

- 近期工作方向與 ablation backlog：看 [docs/TODO.md](/docs/TODO.md)
- 開發入口與 source-of-truth 規則：看 [DEVELOPMENT.md](/DEVELOPMENT.md)
- 事件 / API / storage schema：看 [reference/api_spec.md](reference/api_spec.md)

---

## 1. 系統目標

Saccade 目前以 **GPU-first 的 MOT / tracking / relink pipeline** 為核心，外圍再接事件、儲存與 cognition。

系統主要追求：

- 讓 perception 熱路徑盡量留在 GPU / native path
- 讓 tracking / relink 的決策可解釋、可評估
- 讓 storage / cognition 與 perception 解耦，不阻塞主循環

---

## 2. 邏輯分層

| 層級 | 名稱 | 主要責任 | 主要位置 |
| :--- | :--- | :--- | :--- |
| **L1** | **Perception / Tracking** | detection、postprocess、tracking、GMC、association | `src/saccade/perception/`, `src/tracking/`, `include/tracking/` |
| **L2** | **Appearance / ReID** | crop、embedding、appearance bank、semantic relink | `src/saccade/perception/tracking/`, `src/saccade/perception/eval/relink.py` |
| **L3** | **Streaming / Buffering** | Redis queue / stream、microbatch、event buffering | `src/saccade/storage/`, `src/saccade/pipeline/` |
| **L4** | **Vector Storage** | Chroma memory、metadata filter、hybrid query | `src/saccade/storage/` |
| **L5** | **Cognition / Retrieval** | orchestrator、RAG trigger、query / visual requery | `src/saccade/cognition/`, `src/saccade/api/` |
| **L6** | **Resource / Health** | VRAM 監控、階梯降級、跨進程 VRAM 狀態廣播、service health | `src/saccade/resource/`, `src/saccade/pipeline/health.py` |

---

## 3. 當前主路徑

目前最活躍、最常被維護的主路徑是 **MOT17-centered evaluation path**：

- [scripts/eval/mot17.py](/scripts/eval/mot17.py)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py)

在這條路徑上，主要資料流如下：

```text
Frame Source
  -> preprocess / detection
  -> detection postprocess
  -> optional ReID extract
  -> GPU tracker update
  -> semantic relink / identity resolve
  -> optional post-merge cleanup
  -> eval output
```

與較早期文件相比，現在的重點不是「完整產品型多服務拓樸」，而是：

- tracking / relink / trigger 的決策品質
- native / GPU path 的穩定性
- default evaluation path 的可重現性

---

## 4. 核心元件

### 4.1 Detection / Postprocess

責任：

- 接收 detector 輸出
- 做 filter / suspect / NMS / cross-tile merge
- 產出後續 tracker 使用的 box / score / class

主要位置：

- [src/saccade/perception/eval/detection.py](/src/saccade/perception/eval/detection.py)
- [include/tracking/pipeline.hpp](/include/tracking/pipeline.hpp)
- [src/tracking/pipeline.cpp](/src/tracking/pipeline.cpp)

目前架構重點：

- 盡量走 native facade / CUDA fast path
- letterbox / resize 已換為 fused CUDA kernel（`src/perception/letterbox_kernel.cu`），單次 detect 節省 ~1ms
- Python wrapper 保留 orchestration 與 fallback
- detection quality 仍有進一步演算法空間，但責任邊界已固定

### 4.2 GPU Tracker

責任：

- track state lifecycle
- motion prediction / Kalman
- association
- optional appearance-aware matching
- GMC warp consumption

主要位置：

- [src/tracking/tracker_gpu.cu](/src/tracking/tracker_gpu.cu)
- [include/tracking/tracker_gpu.hpp](/include/tracking/tracker_gpu.hpp)
- [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py)

目前架構重點：

- result path 優先走 GPU-side buffers，再在必要邊界 materialize
- GMC（Global Motion Compensation）走 GPU phase correlation；`estimate_into()` 直寫 device buffer，避免 host roundtrip；sub-stage profiling 可追蹤 FFT / cross_power / IFFT / peak_find 分段耗時
- association 允許 appearance 參與，但仍保留穩定 fallback
- deterministic assignment 與 native identity resolve 已完成收斂

### 4.3 Appearance / ReID

責任：

- crop / embedding extraction
- appearance bank 維護
- dynamic ReID trigger
- semantic relink / identity resolve

主要位置：

- [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py)
- [src/saccade/perception/eval/relink.py](/src/saccade/perception/eval/relink.py)
- [src/saccade/perception/feature_extractor.py](/src/saccade/perception/feature_extractor.py)

目前架構重點：

- reference quality 與 false-accept filtering 是當前主優化方向
- ReID 不是每幀必做；它是受 trigger / budget 控制的昂貴決策資源（`async_reid=True` 為預設，走 side CUDA stream，不阻塞主追蹤循環）
- inter-frame relink 預設走 `pipeline_relink=True`（ThreadPoolExecutor overlap）
- noisy reference 不應污染 bank

### 4.4 Storage / Eventing

責任：

- perception event queue / stream
- Redis microbatch
- Chroma memory insert / hybrid query

主要位置：

- [src/saccade/storage/redis_cache.py](/src/saccade/storage/redis_cache.py)
- [src/saccade/storage/chroma_store.py](/src/saccade/storage/chroma_store.py)

目前架構重點：

- perception 與 cognition 透過 Redis/Chroma 解耦
- event queue / stream 屬於較外圍層，不應反向影響 perception 熱路徑
- 具體 schema 以 [reference/api_spec.md](reference/api_spec.md) 為準

### 4.5 Resource / Memory Management

責任：

- VRAM 使用率監控（pynvml，85/92/96% 三階 hysteresis）
- 跨進程 VRAM 狀態廣播（POSIX named shared memory）
- Dispatcher 端 GPU tracker 生命週期管理（LRU eviction）

主要位置：

- [src/saccade/resource/resource_manager.py](/src/saccade/resource/resource_manager.py)
- [src/saccade/perception/dispatcher.py](/src/saccade/perception/dispatcher.py)

目前架構重點：

**跨進程 VRAM 狀態同步**

`AsyncDispatcher`（dispatcher 進程）持有 `VRAMLevelWriter`，每次 `decide_degradation_level()` 後將 `DegradationLevel`（0–3）寫入 POSIX named shared memory `saccade_vram_level`（1 byte）。`PipelineOrchestrator`（獨立進程）持有 `VRAMLevelReader`，在每個 `handle_cognitive_event` 入口讀取：

- `FAST_PATH (>92%)`：跳過 RAG 分析（停止 HuggingFaceEmbedding GPU 呼叫）
- `EMERGENCY (>96%)`：丟棄非異常 frame，不寫 ChromaDB

兩進程獨立啟動（無共同父進程），故用具名 shared memory 而非 `multiprocessing.Value`。Writer 啟動時自動清除前次崩潰的 stale segment；Reader 找不到 segment 時 fallback 為 NORMAL，不阻塞。

**Dispatcher GPU Tracker LRU**

`AsyncDispatcher.trackers` 改為 `OrderedDict`，加入 `max_streams`（default 8）上限。`get_tracker()` 命中時 `move_to_end` 刷新 LRU 順序；超限時 `popitem(last=False)` 取出最舊 tracker 並 `del`，觸發 C++ `~GPUByteTracker` 釋放所有 CUDA buffer。`deregister_stream()` 支援串流正常結束時的主動釋放。`stop()` 清空全部 tracker。

### 4.6 Cognition / API / Health

責任：

- 讀取事件
- 寫入語義記憶
- 提供 retrieval API
- 監控 service / Redis / VRAM 健康狀態

主要位置：

- [src/saccade/cognition/orchestrator.py](/src/saccade/cognition/orchestrator.py)
- [src/saccade/api/server.py](/src/saccade/api/server.py)
- [src/saccade/pipeline/health.py](/src/saccade/pipeline/health.py)

目前架構重點：

- cognition 是慢路徑，不應阻塞 L1/L2
- health contract 是 operational contract，不等同於對外產品 API

---

## 5. 系統合約

### 5.1 Pipeline 合約

- 主熱路徑優先走 GPU / native facade。
- 引入 CPU roundtrip 時，必須有明確理由與可接受的成本。
- Python 可以做 orchestration，但不應承接大規模每幀資料面工作。

### 5.2 Tracking / Association 合約

- fallback 必須穩定可解釋。
- ambiguous case 不應只靠單一固定 threshold 決策。
- appearance、motion、quality 不應在多層被不透明地重複加權。

### 5.3 ReID / Reference 合約

- low-quality observation 不應和 clean observation 用相同 accept 規則。
- noisy reference 不應進 bank。
- ReID 是受觸發控制的資源，不是預設全量工作。

### 5.4 Storage / API 合約

- Redis key / stream / event schema / Chroma metadata 屬於明確合約，變更時需同步更新 `reference/api_spec.md`
- 外部查詢 API 與 internal health output contract 不可混用

---

## 6. 目前已收斂的架構決策

以下方向已在主路徑上基本收斂：

- Pipeline GPU 化主線完成，主熱路徑已大量 native 化
- deterministic assignment 已落地
- current documented default path 維持：
  - `--cross-tile-merge`
  - `--match-thresh 0.78`
  - `--semantic-threshold 0.91`
  - `async_reid=True`、`pipeline_relink=True`（非同步 side-stream + inter-frame relink overlap）
- GPU GMC 已收斂：phase correlation pipeline 全段 GPU；peak_find 改為 256-thread parallel reduction（原 single-thread O(N) → 12.5× 加速）；frame total 0.71 → 0.28 ms

這些屬於目前穩定系統形狀的一部分，不應在日常小改動中隨意漂移。

---

## 7. 不屬於本文件的內容

以下內容不應長期堆在 `architecture.md`：

- 詳細 ablation 過程
- 單次實驗數字掃描
- 已完成的長篇路線圖
- 細碎 TODO

這些應分別放在：

- [docs/TODO.md](/docs/TODO.md)
- [docs/TODO_history.md](/docs/TODO_history.md)
- `docs/experiments/`

---

## 8. 相關文件

- 開發入口：[DEVELOPMENT.md](/DEVELOPMENT.md)
- API / event / storage contract：[reference/api_spec.md](reference/api_spec.md)
- 模組 delta ledger：[reference/PIPELINE_REFERENCE.md](reference/PIPELINE_REFERENCE.md)
- 全流程敘事版資料流：[reference/pipeline_flow.md](reference/pipeline_flow.md)
- Tracker 深入說明：[docs/layers/gpubytetracker_deep_dive.md](/docs/layers/gpubytetracker_deep_dive.md)

最後更新：2026-05-07
