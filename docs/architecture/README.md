# Saccade 系統架構說明書

本文件描述 **目前穩定的系統形狀與責任邊界**。它不是實驗日誌，也不是待辦清單。

- 近期工作方向與 ablation backlog：看 [TODO.md](../TODO.md)
- 開發入口與 source-of-truth 規則：看 [DEVELOPMENT.md](../../DEVELOPMENT.md)
- 事件 / API / storage schema：看 [api_spec.md](../modules/storage/api_spec.md)

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
| **L1** | **Eval Hot Path / Perception** | frame ingest、detection、postprocess、GMC、tracking、MOT output | `src/saccade/perception/eval/`, `scripts/eval/`, `src/tracking/`, `include/tracking/` |
| **L2** | **Tracker Core / Native Facades** | GPUByteTracker、Kalman、auction/Sinkhorn、native postprocess/GMC/ReID facades | `src/tracking/`, `include/tracking/`, `src/saccade/perception/tracking/` |
| **L3** | **Appearance / ReID / Relink Options** | crop、embedding、appearance bank、semantic relink、optional identity resolve | `src/saccade/perception/tracking/`, `src/saccade/perception/eval/relink.py`, `src/saccade/perception/eval/output_bank.py` |
| **L4** | **Runtime Streaming / Dispatch** | multistream dispatcher、zero-copy helpers、frame pool / eval streaming adapters | `src/saccade/perception/dispatcher.py`, `src/saccade/perception/zero_copy.py`, `src/saccade/perception/eval/streaming.py`, `src/saccade/perception/eval/pool.py` |
| **L5** | **Storage / Event Memory** | Redis microbatch、Chroma memory、metadata filter、hybrid query | `src/saccade/storage/` |
| **L6** | **Cognition / API / Resource Health** | orchestrator、RAG trigger、API、VRAM degradation、service health | `src/saccade/cognition/`, `src/saccade/api/`, `src/saccade/resource/`, `src/saccade/pipeline/health.py` |

---

## 3. 當前主路徑

目前最活躍、最常被維護的主路徑是 **MOT17-centered evaluation path**：

- [scripts/eval/mot17.py](../../scripts/eval/mot17.py)
- [src/saccade/perception/eval/evaluator.py](../../src/saccade/perception/eval/evaluator.py)

[runner.py](../../src/saccade/perception/eval/runner.py) remains as a compatibility
shim that re-exports `run_eval`; do not treat it as the implementation source of
truth.

在這條路徑上，主要資料流如下：

```text
Frame Source
  -> fetch / ingest_preprocess
  -> detect
  -> postprocess
  -> optional ReID budget / crop / extract
  -> GMC
  -> GPU tracker update
  -> materialize
  -> fast MOT emit or optional identity resolve
  -> eval output
```

與較早期文件相比，現在的重點不是「完整產品型多服務拓樸」，而是：

- tracking / relink / trigger 的決策品質
- native / GPU path 的穩定性
- default evaluation path 的可重現性

Current headline preset is `mamba_whole_graph`: `native_640`, whole-detect CUDA
graph, GPU GMC, tracker graph, bidirectional bridge relink, ReID off. Raw parser
defaults and older module configs are still useful for ablations, but they are
not the headline architecture.

---

## 4. 核心元件

### 4.1 Detection / Postprocess

責任：

- 接收 detector 輸出
- 做 filter / suspect / NMS / cross-tile merge
- 產出後續 tracker 使用的 box / score / class

主要位置：

- [src/saccade/perception/eval/detection.py](../../src/saccade/perception/eval/detection.py)
- [include/tracking/pipeline.hpp](../../include/tracking/pipeline.hpp)
- [src/tracking/pipeline.cpp](../../src/tracking/pipeline.cpp)

目前架構重點：

- 盡量走 native facade / CUDA fast path
- `mamba_whole_graph` 以 `native_640` + `preprocess: none` 作 headline path；tiled / `native_960` 是 legacy comparison 或 ablation path
- native `PerceptionPipeline` 可承接 tensor prep / filter / NMS；Python wrapper 保留 orchestration、debug dump 與 fallback
- `detection_quality_scaling`、`person_geometry_prior`、`geometry_suspect_support` 在 current headline preset 關閉，避免和 Mamba 分佈重校準重疊

### 4.2 GPU Tracker

責任：

- track state lifecycle
- motion prediction / Kalman
- association
- optional appearance-aware matching
- GMC warp consumption

主要位置：

- [src/tracking/tracker_gpu.cu](../../src/tracking/tracker_gpu.cu)
- [include/tracking/tracker_gpu.hpp](../../include/tracking/tracker_gpu.hpp)
- [src/saccade/perception/tracking/tracker_gpu.py](../../src/saccade/perception/tracking/tracker_gpu.py)

目前架構重點：

- result path 優先走 GPU-side buffers，再在必要邊界 materialize
- GMC（Global Motion Compensation）走 GPU phase correlation；`estimate_into()` 直寫 device buffer，避免 host roundtrip；sub-stage profiling 可追蹤 FFT / cross_power / IFFT / peak_find 分段耗時
- current preset 透過 `set_params()`、`set_oao_params()`、`set_occ_params()`、`set_relink_params()` 將 YAML 值下到 C++ tracker
- association 允許 appearance 參與，但 current headline preset 是 ReID off；主要 identity 修復在 tracker-core bidirectional bridge relink
- result buffers 由 `track` 寫入，`materialize` 才在 MOT output boundary 做必要 readback

### 4.3 Appearance / ReID

責任：

- crop / embedding extraction
- appearance bank 維護
- dynamic ReID trigger
- semantic relink / identity resolve

主要位置：

- [src/saccade/perception/tracking/tracker_gpu.py](../../src/saccade/perception/tracking/tracker_gpu.py)
- [src/saccade/perception/eval/relink.py](../../src/saccade/perception/eval/relink.py)
- [src/saccade/perception/feature_extractor.py](../../src/saccade/perception/feature_extractor.py)

目前架構重點：

- reference quality 與 false-accept filtering 是保留能力，但不是 current headline preset 的精度來源
- ReID 不是每幀必做；它是受 trigger / budget 控制的昂貴決策資源。當 ReID 開啟且 native backend 可用時，`async_reid=True` 會走 side CUDA stream，並和 GMC overlap，再於 `track` 前同步
- inter-frame `pipeline_relink=True` 只在完整 emit pipeline active 且沒有 `--profile-stages` 時把 `relink_write` 丟到 background executor
- noisy reference 不應污染 bank

### 4.4 Storage / Eventing

責任：

- perception event queue / stream
- Redis microbatch
- Chroma memory insert / hybrid query

主要位置：

- [src/saccade/storage/redis_cache.py](../../src/saccade/storage/redis_cache.py)
- [src/saccade/storage/chroma_store.py](../../src/saccade/storage/chroma_store.py)

目前架構重點：

- perception 與 cognition 透過 Redis/Chroma 解耦
- event queue / stream 屬於較外圍層，不應反向影響 perception 熱路徑
- 具體 schema 以 [api_spec.md](../modules/storage/api_spec.md) 為準

### 4.5 Resource / Memory Management

責任：

- VRAM 使用率監控（pynvml，85/92/96% 三階 hysteresis）
- 跨進程 VRAM 狀態廣播（POSIX named shared memory）
- Dispatcher 端 GPU tracker 生命週期管理（LRU eviction）

主要位置：

- [src/saccade/resource/resource_manager.py](../../src/saccade/resource/resource_manager.py)
- [src/saccade/perception/dispatcher.py](../../src/saccade/perception/dispatcher.py)

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

- [src/saccade/cognition/orchestrator.py](../../src/saccade/cognition/orchestrator.py)
- [src/saccade/api/server.py](../../src/saccade/api/server.py)
- [src/saccade/pipeline/health.py](../../src/saccade/pipeline/health.py)

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

- Redis key / stream / event schema / Chroma metadata 屬於明確合約，變更時需同步更新 [api_spec.md](../modules/storage/api_spec.md)
- 外部查詢 API 與 internal health output contract 不可混用

---

## 6. 目前已收斂的架構決策

以下方向已在主路徑上基本收斂：

- Pipeline GPU 化主線完成，主熱路徑已大量 native 化。
- deterministic assignment / GPUByteTracker hot path 已落地。
- current headline preset 維持 `mamba_whole_graph`：`native_640`、whole-detect CUDA graph、GPU GMC、tracker graph、bidirectional bridge relink、ReID off。
- current tracker preset 值：`match_thresh=0.50`、`new_track_thresh=0.28`、`kalman_r_scale=2.8`、`oao_tau=0.50`、`oao_ramp_frames=25`、`multiplicative_cost=true`。
- GPU GMC 已收斂：phase correlation pipeline 全段 GPU；C++ GMC / PyGraphedGMC / SparseOpticalFlow fallback 順序由 `_build_gmc_estimator()` 決定。
- `async_reid`、`pipeline_relink` 是 optional throughput mechanisms；current headline preset ReID off，因此它們不是 headline 精度來源。

這些屬於目前穩定系統形狀的一部分，不應在日常小改動中隨意漂移。

---

## 7. 不屬於本文件的內容

以下內容不應長期堆在 `architecture.md`：

- 詳細 ablation 過程
- 單次實驗數字掃描
- 已完成的長篇路線圖
- 細碎 TODO

這些應分別放在：

- [TODO.md](../TODO.md)
- [TODO_history.md](../TODO_history.md)
- `../research/`

---

## 8. 相關文件

- 開發入口：[DEVELOPMENT.md](../../DEVELOPMENT.md)
- API / event / storage contract：[api_spec.md](../modules/storage/api_spec.md)
- 模組 delta ledger：[PIPELINE_REFERENCE.md](../reference/PIPELINE_REFERENCE.md)
- 全流程敘事版資料流：[pipeline_flow.md](../reference/pipeline_flow.md)
- Tracker 深入說明：[gpubytetracker_deep_dive.md](../modules/geometry/tracker_deep_dive.md)

最後更新：2026-05-07
