# Saccade 系統架構說明書

本文件描述 **目前穩定的系統形狀與責任邊界**。它不是實驗日誌，也不是待辦清單。

- **架構圖集**：本 README 是唯一閱讀入口。[跨 profile 系統總覽](system_overview_536.md) 提供各入口、責任、主要依賴、執行邊界與節點／邊追溯；[S-SHIP／S-EXPORT 展開圖與契約](ship_export_contracts_536.md) 維護 shipping／export 的細節與設計狀態。圖集組織規則見 [S-SHIP §6](ship_export_contracts_536.md#6-本稿之後)，各文件的狀態由原文件維護。
- 從 `src/` 目錄找到實作與跨語言入口：看 [src/README.md](../../src/README.md)
- 近期工作方向與 ablation backlog：看 [TODO.md](../TODO.md)
- 開發入口（需求層級 → 文檔組合）與 source-of-truth：看 [DEVELOPMENT.md](../../DEVELOPMENT.md)
- 事件 / API / storage schema：看 [api_spec.md](../modules/storage/api_spec.md)
- 能力／REQ／Profile、具名 owner、inherited scope／proposed delta 與代表鏈狀態：看 [capability_requirements_535.md](capability_requirements_535.md)（#535；批准與待決狀態只在該檔維護）
- S-SHIP／S-EXPORT 的展開圖（現況與目標）、completion／identity／export 接口契約卡與 ID 索引：看 [ship_export_contracts_536.md](ship_export_contracts_536.md)（#536；設計狀態只在該檔維護）
- 失敗／fallback／降級（分 profile 矩陣、owner 裁決、已落地的 F07／C2／R-1 與 D2 host lifecycle；D2-GPU 仍 OPEN）：看 [failure_degradation_537.md](failure_degradation_537.md)；NVML 失敗的下一切片與待核准方案見 [§17](failure_degradation_537.md#17-c0-05-下一切片nvml-失敗的重現與待核准方案)。
- 安裝後 package 的 Completion／Gate A、entrypoint re-pin 與驗證證據：看 [package re-pin](../reference/native_runtime_package_repin_536.md)（#536；不擴大支持範圍）
- 模型 bundle 與 native runtime 的分離（現況／目標架構圖、信任根、責任分工、A／B／C 比較）：看 [ADR 028](../decisions/028-model-bundle-runtime-separation.md)；bundle manifest 契約、fail-closed 規則、G01–G07 處置與控制矩陣：看 [model_bundle_contract_549.md](model_bundle_contract_549.md)（#549 S1；設計狀態只在該檔維護）

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

L1～L6 從 Python 套件與它載入的 native 擴充來劃分；L1／L2 列的 `src/tracking/`、`include/tracking/` 是這些擴充背後的 native source。同一份 source 也編成 static library 連進 `shipping/` 的 native 執行檔；`shipping/` 執行檔、static library 的連結關係與 build／export 期的產物交付不在這張表，見 [系統總圖](system_overview_536.md#2-系統總圖)。

### 2.1 能力與使用 profile

L1～L6 是責任分層；它們在不同 entrypoints 中組成不同路徑。支持範圍須同時讀
capability 與 profile，不能用一條 eval 或 shipping 路徑的證據概括全部模組。
[CAP/REQ 盤點與支持矩陣](capability_requirements_535.md) 保留既有支持義務、
source-inspected 現況與新候選需求；owner、批准與待決狀態只在該檔的
[decision 紀錄](capability_requirements_535.md#decision-535-first-slice)維護，本節不重述。

| 使用者／profile | 能力與邊界；細節及限制由 linked owner 擁有 |
|:--|:--|
| 本機 native 使用者／shipping | CAP-535-01：installed `saccade_track` 的 JPEG-sequence tracking/MOT output；依 [#465 shipping boundary](../reference/native_runtime_shipping_boundary.md) 與 [closeout](../reference/native_runtime_closeout.md)，保留同機 A_L、sm_120／Ubuntu 24.04 限制；公開發行由 #547 裁決 |
| 外部 Python caller／public-library | CAP-535-02：五個現有 lazy exports；[ADR 025](../decisions/025-native-extension-delivery.md) 已決定 native delivery，#441 擁有 tracker/reorder public surface／compatibility／core channel；eval/config/detector API 支持由 #535/#536 與原 owner 裁決 |
| 評測維護者／eval | CAP-535-03：MOT17/headline、dataset adapters、detector/metric/comparison 入口；既有 headline config 不為所有組合承諾品質 |
| 研究與量測維護者／research | CAP-535-04、08-T：measurement/controls/custody、training recipes；原 research contracts/evidence owners 保有驗收與 promotion，`saccade_track_measurement` 與 native probes 不 install 到 shipping |
| 串流／記憶服務使用者及維運者／online-service | CAP-535-05～07：ingest/dispatch、event memory/RAG/retrieval、resource/health；source producer/callback/deployment 缺口與未知 consumers 見 ledger E05～E07 |
| 開發者／build-debug | CAP-535-04、08-E/B、09：observer/native measurement、artifact export、native build/install/check、perception demo；demo 未接 tracker，工具支持不從 shipping parity 推出 |

能力生命週期由 #539 裁決；未批准、default-off、未找到 consumer 均不等於可退役。

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
- VRAM 狀態廣播（POSIX named shared memory；分開啟動時跨進程）
- Dispatcher 端 GPU tracker 生命週期管理（LRU eviction）

主要位置：

- [src/saccade/resource/resource_manager.py](../../src/saccade/resource/resource_manager.py)
- [src/saccade/perception/dispatcher.py](../../src/saccade/perception/dispatcher.py)

目前架構重點：

**VRAM 狀態同步（具名 shared memory）**

`AsyncDispatcher`（perception 端；分開啟動時為獨立進程，見下段）持有 `VRAMLevelWriter`，每次 `decide_degradation_level()` 後把 `DegradationLevel`（0–3）與決策開始時間寫成 `/dev/shm/saccade_vram_level` 的快照（寫暫存檔再 rename）。`PipelineOrchestrator`（分開啟動時為獨立進程，見下段）持有 `VRAMLevelReader`，在每個 `handle_cognitive_event` 入口重新讀取；只有 FRESH 的 level 會生效：

- `FAST_PATH`（usage >92% 進入，降到 87% 以下才離開）：跳過 RAG 分析（停止 HuggingFaceEmbedding GPU 呼叫）
- `EMERGENCY`（usage >96% 進入，降到 91% 以下才離開）：丟棄非異常 frame，不寫 ChromaDB
- 不是 FRESH（STALE／MISSING／INVALID）：所有事件都嘗試寫入，不啟動新的 RAG

分開以 `main.py --mode perception`／`--mode orchestrator` 啟動時，兩者是沒有共同父進程的兩個 process，所以用 `/dev/shm` 下的具名快照而非 `multiprocessing.Value`；`--mode full` 則在同一 process 的同一 event loop 跑兩者，level 仍經同一份快照傳遞（見[系統總圖](system_overview_536.md#52-本次追查的發現) B0-F01）。修復前，reader 會一直附掛在舊 segment 上（B0-F07，[C0-08](failure_degradation_537.md#2-capabilityfailure-矩陣observed-as-is)）；修復後，reader 每次都依名稱重讀，writer 重啟後的下一次讀取就會跟上，找不到快照時讀為 MISSING 而不是 NORMAL（[C0 §7](failure_degradation_537.md#7-b0-f07-修復c0-060708-收斂)、[契約](../modules/resource/architecture_degradation.md#6-vram-level-跨程序快照契約537-b0-f07)）。三階 hysteresis 修復前的實際轉換和文件不一致（[§3 轉換表](failure_degradation_537.md#3-降級轉換表現況與已採納目標)）；C2 修復後，進入門檻與離開門檻分開：升級立即生效、可跨級，恢復每次只降一級，而且要嚴格低於門檻 −5。例如進入 EMERGENCY 後 usage 降到 94%，level 仍是 EMERGENCY，FRESH consumer 會繼續卸載非 anomaly 事件（[C0 §8](failure_degradation_537.md#8-c2-遲滯修復c0-04-收斂)、[degradation §5](../modules/resource/architecture_degradation.md#5-hysteresis遲滯保護)）。

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

- 目錄地圖（從路徑找到實作）：[src/README.md](../../src/README.md)
- 開發入口：[DEVELOPMENT.md](../../DEVELOPMENT.md)
- API / event / storage contract：[api_spec.md](../modules/storage/api_spec.md)
- 跨 profile 系統總圖與節點／邊追溯索引：[system_overview_536.md](system_overview_536.md)
- 現行 full stack 各機制的 present-day role 與 interaction map：[module_interactions.md](module_interactions.md)
- 模組 delta ledger：[PIPELINE_REFERENCE.md](../reference/PIPELINE_REFERENCE.md)
- 全流程敘事版資料流：[pipeline_flow.md](../reference/pipeline_flow.md)
- Tracker 深入說明：[gpubytetracker_deep_dive.md](../modules/geometry/tracker_deep_dive.md)

> 本文件只描述**穩定架構形狀與責任邊界**，不內嵌會隨實驗漂移的數字。
> baseline 數字見 [TODO.md](../TODO.md)「當前 Baseline」節；模組 live WIP 見各 module `TODO.md`（入口在 [DEVELOPMENT.md 模組現狀總覽](../../DEVELOPMENT.md#模組現狀總覽)）。
