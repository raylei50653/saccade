# Saccade 系統架構說明書

本文件描述 **目前穩定的系統形狀與責任邊界**。它不是實驗日誌，也不是待辦清單。

- 近期工作方向與 ablation backlog：看 [docs/TODO.md](/docs/TODO.md:1)
- 開發入口與 source-of-truth 規則：看 [DEVELOPMENT.md](/DEVELOPMENT.md:1)
- 事件 / API / storage schema：看 [docs/api_spec.md](/docs/api_spec.md:1)

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
| **L6** | **Resource / Health** | VRAM、service health、degrade signals | `src/saccade/resource/`, `src/saccade/pipeline/health.py` |

---

## 3. 當前主路徑

目前最活躍、最常被維護的主路徑是 **MOT17-centered evaluation path**：

- [scripts/eval/mot17.py](/scripts/eval/mot17.py:1)
- [src/saccade/perception/eval/runner.py](/src/saccade/perception/eval/runner.py:1)

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

- [src/saccade/perception/eval/detection.py](/src/saccade/perception/eval/detection.py:1)
- [include/tracking/pipeline.hpp](/include/tracking/pipeline.hpp:1)
- [src/tracking/pipeline.cpp](/src/tracking/pipeline.cpp:1)

目前架構重點：

- 盡量走 native facade / CUDA fast path
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

- [src/tracking/tracker_gpu.cu](/src/tracking/tracker_gpu.cu:1)
- [include/tracking/tracker_gpu.hpp](/include/tracking/tracker_gpu.hpp:1)
- [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py:1)

目前架構重點：

- result path 優先走 GPU-side buffers，再在必要邊界 materialize
- association 允許 appearance 參與，但仍保留穩定 fallback
- deterministic assignment 與 native identity resolve 已完成收斂

### 4.3 Appearance / ReID

責任：

- crop / embedding extraction
- appearance bank 維護
- dynamic ReID trigger
- semantic relink / identity resolve

主要位置：

- [src/saccade/perception/tracking/tracker_gpu.py](/src/saccade/perception/tracking/tracker_gpu.py:1)
- [src/saccade/perception/eval/relink.py](/src/saccade/perception/eval/relink.py:1)
- [src/saccade/perception/feature_extractor.py](/src/saccade/perception/feature_extractor.py:1)

目前架構重點：

- reference quality 與 false-accept filtering 是當前主優化方向
- ReID 不是每幀必做；它是受 trigger / budget 控制的昂貴決策資源
- noisy reference 不應污染 bank

### 4.4 Storage / Eventing

責任：

- perception event queue / stream
- Redis microbatch
- Chroma memory insert / hybrid query

主要位置：

- [src/saccade/storage/redis_cache.py](/src/saccade/storage/redis_cache.py:1)
- [src/saccade/storage/chroma_store.py](/src/saccade/storage/chroma_store.py:1)

目前架構重點：

- perception 與 cognition 透過 Redis/Chroma 解耦
- event queue / stream 屬於較外圍層，不應反向影響 perception 熱路徑
- 具體 schema 以 [docs/api_spec.md](/docs/api_spec.md:1) 為準

### 4.5 Cognition / API / Health

責任：

- 讀取事件
- 寫入語義記憶
- 提供 retrieval API
- 監控 service / Redis / VRAM 健康狀態

主要位置：

- [src/saccade/cognition/orchestrator.py](/src/saccade/cognition/orchestrator.py:1)
- [src/saccade/api/server.py](/src/saccade/api/server.py:1)
- [src/saccade/pipeline/health.py](/src/saccade/pipeline/health.py:1)

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

- Redis key / stream / event schema / Chroma metadata 屬於明確合約，變更時需同步更新 `api_spec.md`
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

這些屬於目前穩定系統形狀的一部分，不應在日常小改動中隨意漂移。

---

## 7. 不屬於本文件的內容

以下內容不應長期堆在 `architecture.md`：

- 詳細 ablation 過程
- 單次實驗數字掃描
- 已完成的長篇路線圖
- 細碎 TODO

這些應分別放在：

- [docs/TODO.md](/docs/TODO.md:1)
- [docs/TODO_history.md](/docs/TODO_history.md:1)
- `docs/experiments/`

---

## 8. 相關文件

- 開發入口：[DEVELOPMENT.md](/DEVELOPMENT.md:1)
- API / event / storage contract：[docs/api_spec.md](/docs/api_spec.md:1)
- 當前待辦與近期結論：[docs/TODO.md](/docs/TODO.md:1)
- 全流程敘事版資料流：[docs/pipeline_flow.md](/docs/pipeline_flow.md:1)
- Tracker 深入說明：[docs/layers/gpubytetracker_deep_dive.md](/docs/layers/gpubytetracker_deep_dive.md:1)

最後更新：2026-04-30
