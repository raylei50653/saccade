<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-26 -->
<!-- doc-module: cross -->

# 非同步 edge 與 buffer ownership 盤點（#538 D1）

本稿的 §1～§9 是 [#538](https://github.com/raylei50653/saccade/issues/538) 的 **D1**：依 #535 的 profiles，盤點主要的非同步 edge，逐條記下誰持有資料、何時算完成、何時可以重用或釋放、容量與溢出、失敗怎麼傳、取消與重啟時怎麼收尾，以及支撐這些說法的證據等級。D1 只盤點與路由，不修改行為；Owner 對發布、路由與 #340 trace 的裁決見 §9（決策來源是 #538 的 comment）。[§10](#d3-f02) 另記 D3-F02 的候選修復與驗證。閱讀入口是 [architecture README](README.md)。

**D1 source baseline（§1～§9）**：`main@3375e7d4`（#629 merge，2026-10-25）。除 §5 標 `reproduced` 的兩項與 §1 註明的 DALI 建構觀察外，D1 新增的判讀都是 `source-inspected`；引用的既有證據保留原文件的等級與座標，D1 沒有重跑。§10 的修復與驗證另列自己的基線。

**本稿不擁有的內容**（只引用）：CAP／REQ 與支持裁決歸 [#535 ledger](capability_requirements_535.md)；節點／邊 ID 歸 [跨 profile 總覽](system_overview_536.md)（本稿的 `AE-*` 掛在它的 `SN-*`／`SE-*` 上，見 [§6](system_overview_536.md#6-給-537538539-的共同邊界)）；失敗狀態與 run 轉換語義歸 [#537](failure_degradation_537.md)；shipping 的排程、graph 與 parity 歸 #465（[resolved config §14](../reference/native_runtime_resolved_config.md#14-native-double-buffercuda-graphpr-10u5)），本稿只引用、不重驗、不推廣成全 repo 的保證；VRAM 快照契約歸 [degradation §6](../modules/resource/architecture_degradation.md#6-vram-level-跨程序快照契約537-b0-f07)；功能保留或退役歸 #539。

## 1. 範圍與用語

**涵蓋的 edge 家族**（#538 D1 指定）與對應的 #535 profile：

| 家族 | Profile | 掛載點（#536） | 本稿 ID |
|:--|:--|:--|:--|
| native double-buffer／CUDA graphs | shipping（CAP-01） | SN-SH3 | AE-S1～S3 |
| eval 的 decode 預取、double-buffer、延後輸出、選用的 async ReID／relink、程序與 thread 分流 | eval（CAP-03） | SN-EV2、SN-EV3 | AE-E1～E9 |
| online ingest、dispatcher、workbench | online-service（CAP-05） | SE-17、SE-18、SE-19、SE-49、SN-ON3 | AE-O1～O6 |
| 跨程序 resource state | online-service（CAP-07） | SE-23、SE-24 | AE-R1 |
| storage／cognition queue | online-service（CAP-06） | SE-21、SE-22、SE-25、SN-ON4 | AE-M1～M3 |

**刻意不列**：`MultiStreamRunner`（[multi_stream.py](../../src/saccade/perception/eval/multi_stream.py)，沒有 in-tree 入口，只有 evaluator 的註解提到它）；explicit stream probe（`_pp_streams`，env 開啟的 research 診斷）；`tools/resctl.py` 的 worktree lease（開發工具，不屬任何 runtime profile）。它們不代表沒有風險，只是不在 D1 的主要 profile 內。

**證據等級**（由強到弱，彼此不互相替代）：

| 等級 | 意義 |
|:--|:--|
| cited（#465） | #465 已接受的排程／parity 證據，限其同機座標；本稿未重跑 |
| GPU-validated | 在本機 GPU 執行過的契約測試或正控制（D2-GPU 等） |
| reproduced | 本稿或既有文件以可重播的程序重現了具體行為 |
| host-only | CPU fake 的 lifecycle 測試；不涵蓋 CUDA 排序或 buffer lifetime |
| documentation-supported | 依第三方套件的文件與 source，GPU-unverified：本 repo 沒有 runtime 驗證，不能取代 GPU 端到端測試 |
| source-inspected | 只讀過本 repo 的 source |
| unknown | 沒有讀到足以判斷的 source，或屬 C++ 內部未查 |

**DALI 2.0.0 的建構觀察**：`DALIVideoPipeline` 與 eval 的 `_JpgPipe` 都沒有關閉預設的 dynamic executor。在本機只建構不 build 的 `DALIVideoPipeline` 上讀到 `exec_dynamic=True`。這條件下 `DALIGenericIterator.__next__` 走 `torch.from_dlpack`（零拷貝），DALI 的 `TensorGPU.__dlpack__` 文件說預設執行模型的輸出可以無限期持有，並會在呼叫端 stream 插入同步。這是 documentation-supported，**不是**本 repo 的 runtime 驗證（AE-O2、AE-E1b）。

**欄位**（#538 D1 要求的 12 欄，拆成兩張表，以 ID 對應）：表 A 是 producer／consumer、buffer owner、snapshot／version、publish／completion、reuse／free、ordering；表 B 是 capacity、overflow／backpressure、failure propagation、cancel／drain、restart／reset、evidence。

## 2. 表 A：ownership、completion 與 ordering

### 2.1 shipping（引用 #465，不重驗）

| ID | producer → consumer | buffer owner | snapshot／version | publish／completion | reuse／free 條件 | ordering |
|:--|:--|:--|:--|:--|:--|:--|
| AE-S1 | host thread 的 JPEG decode(k+1) → detect stream 的 normalize | `IngestHost`（每個 sequence 一個，2 個 pool，依 `(k−1) % 2` 選） | parity | decode 在 host 返回時已完成 | host 在 decode 進某個 pool 前，先等該 pool 上次 normalize 的 event | host 順序 decode；detect stream 等 main stream 的 `input_ready(k+1)` |
| AE-S2 | detect stream 的 whole-detect graph(k) → main stream 的 post-detector／tracker(k) | runtime 的兩組 device detection buffer（每個 run），依 parity | parity；graph static output 拷進 parity buffer | detect stream 記 `ready(k)`，main stream 等它 | launch(k+1) 重用 k−1 的 frame／detection buffer；它們最後的讀者（GMC frame 拷貝、copy_pad）在 main stream 上排在 `input_ready(k+1)` 之前 | 同時只有一個 detection 在飛：detect(k+1) ∥ tracker(k) |
| AE-S3 | 四個 CUDA graph 的 static input／output | whole-detect graph 依形狀＋dims＋NMS pad 快取（跨 sequence）；NMS、GMC、tracker graph 每個 sequence 一個 | recapture key | replay 前把輸入拷進 static input | 下一次 replay 前，輸出已拷出（AE-S2） | replay 都在 runtime 的 stream 上 |

說明見 [double_buffer_runtime.hpp](../../shipping/include/saccade_shipping/double_buffer_runtime.hpp) 的檔頭與 [resolved config §14.1](../reference/native_runtime_resolved_config.md#141-照抄的-oracle-事實排程與-graph)。

### 2.2 eval

| ID | producer → consumer | buffer owner | snapshot／version | publish／completion | reuse／free 條件 | ordering |
|:--|:--|:--|:--|:--|:--|:--|
| AE-E1a | `TorchvisionGpuStreamer` decode worker（`SACCADE_GPU_DECODE=1`）→ frame loop | caching allocator；frame 在 worker 的 decode stream 上配置 | 每個 worker generation 綁自己的 queue 與 decode stream | 每幀在 decode stream 記 `ready`，consumer stream `wait_event` | consumer 對 frame `record_stream`，丟掉引用後才可回收 | queue FIFO；一個 sequence 一個 worker generation |
| AE-E1b | eval `DALIStreamerStream`（預設）→ frame loop | DALI（dynamic executor）經 DLPack 交給 torch | — | `from_dlpack` 在 consumer 的 current stream 插入同步（DALI 文件） | DALI 文件：預設執行模型的輸出可以無限期持有 | iterator 順序 |
| AE-E2 | double-buffer side stream 的 detect(k+1) → main stream 的 frame(k+1) | `EvalPipeline`：兩個 `AdaptiveFramePool`、一條 side stream、兩組 event | parity；graph 輸出在 side stream 上 clone | `input_ready` 記在 main、side 等它；clone 後記 `ready_event`，main 等它 | 輸出 tensor 對 main stream `record_stream`；`frame_gpu` 對 side stream `record_stream` | 一個 detection 在飛（同 AE-S2，是它的 oracle） |
| AE-E3 | double-buffer 的 tracker 輸出 D2H(k) → 下一幀開頭的 emit／relink(k) | 兩組 pinned buffer，依 `frame_id % 2` | parity | main stream 上 non-blocking copy 後記 event；`_flush_db_tracker_out` 先 `ev.synchronize()` 再 clone 出 pinned | 下一個同 parity 的 copy 在兩幀之後，flush 已在其前完成 | flush 在 tracker(k+1) 之前，保留 relink→tracker 順序 |
| AE-E4 | 非 double-buffer 的延後輸出（`_defer_emit`）→ 下一幀開頭的 emit | 單組 pinned buffer | — | 專用 copy stream 等 `compute_done` 後拷貝並記 event；`_flush_deferred_emit` 在 `_run_frame` 開頭 host 等 event | 下一次 tracker update 排入前已 host 同步；pinned 在下一次 copy 前已讀完 | copy stream 與 main stream 只經 event 與 host 同步連接 |
| AE-E5 | 選用的 relink 背景寫入（`pipeline_relink`，不是 headline）→ 下一幀或 sequence 結尾 | 單 thread 的 `ThreadPoolExecutor`（每個 `EvalPipeline` lazy 建立）；輸入先以 `.cpu()` 同步拷到 host | — | `future.result()` | 背景 thread 改的共享狀態（relinker、appearance bank、dynamic ReID），main 在下一次存取前先取 `result()` | 至多一個 future 在飛 |
| AE-E6 | 選用的 async ReID（`async_reid`；headline `reid_mode: off`）side stream → tracker update 前的 embedding 注入 | Python 引用（`frame_hwc`、`budgeted_boxes`、`budget_embeddings`）持有到同一函式內的 host 同步；native 收 raw `data_ptr` | — | side stream 等 `reid_main_ready`，結束記 `reid_side_event`；注入前 host `synchronize()` | 沒有 `record_stream`；安全來自引用在 host 同步之後才丟 | 與 main 上的 GMC 重疊 |
| AE-E7 | `mot17.py --processes N --sequences …` 的 parent → 每個 sequence 一個子程序 | 每個子程序自己的 GPU context 與輸出檔 | run manifest 由 parent claim、子程序 join | `subprocess.run` 的 exit code | — | parent 依提交順序取結果，之後才算 metrics |
| AE-E8 | `--cpp-threads N` → C++ `CppEvaluatorPool` | unknown（C++ 內部未查） | unknown | unknown | unknown | 各 sequence 平行，之後在 Python 做後處理 |
| AE-E9 | `concurrent_mot17.py` 的多 worker → 共享 `BatchingTRTDetector` | 同 AE-O5 的 batcher | — | 同 AE-O5 | 同 AE-O5 | 4 workers 時在 NMS CUDA graph capture segfault（[§18](failure_degradation_537.md#18-538-d2-gpu-1cuda-排序輸出所有權與-batcher-失敗的修復) 的範圍外發現） |

### 2.3 online：ingest、dispatcher、workbench

| ID | producer → consumer | buffer owner | snapshot／version | publish／completion | reuse／free 條件 | ordering |
|:--|:--|:--|:--|:--|:--|:--|
| AE-O1 | MediaMTX（GStreamer callback thread，或 C++ GstClient callback）→ producer task 的 `grab_tensor()` | 每個 sample 新配置的 tensor；C++ 路徑在 buffer 專屬 stream 上 `clone()` 後歸還 pool slot | `_frame_generation`；`grab_tensor` 不回傳同一 generation 兩次 | slot 在 `_lock` 下替換 | slot 只保留最新一幀，舊的由 GC 回收（[GE-08](failure_degradation_537.md#17-538-d2-gpu-0cuda-排序輸出所有權與-batcher-失敗現況)） | latest-wins；Python 路徑的 `.to("cuda")` 在 callback thread 的 legacy stream 0 |
| AE-O2 | `DALIMediaClient.grab_tensor()` → `main.py` producer → `AsyncDispatcher` queue | DALI（dynamic executor）經 DLPack；producer 放進 queue 的是 `squeeze(0)` 的 view，沒有 clone | — | `from_dlpack` 在 current stream 插入同步（DALI 文件） | DALI 文件：輸出可以無限期持有；本 repo 未驗證（GE-07 收窄） | queue FIFO |
| AE-O3 | producer task → `AsyncDispatcher` worker → `on_track_result` | queue 中的 `_QueuedFrame` 持有 frame 引用；`torch.stack` 拷進新的 batch tensor | — | 同一 event loop thread、同一 legacy stream；callback 被 `await` | callback 拿到新 tensor，不被後續幀重用（[GE-06](failure_degradation_537.md#17-538-d2-gpu-0cuda-排序輸出所有權與-batcher-失敗現況)） | 單一 FIFO 跨 stream；同一 stream 內保序 |
| AE-O4 | 8stream `--workbench` producer → `WorkbenchPool` 每 stream 的 queue → `_WorkbenchWorker` | queue 持有 frame 引用；worker 前處理產生 `padded` | pool generation（重啟換新 queue 與 worker） | worker 在前處理後 `workbench.stream.wait_stream(producer)`（G1 修復） | 同 GE-02／GE-03：正常路徑靠 C++ host 同步 | 每 stream 一個 queue、一個 worker |
| AE-O5 | `_WorkbenchWorker.submit()` → `BatchingTRTDetector` server thread | server 的 batch 與輸出 slice | — | `frame_event`／`infer_event`；server 失敗時喚醒全部等待者（G3 修復） | 同 GE-03 | server 依到達順序組 batch |
| AE-O6 | `_WorkbenchWorker` → `on_result`（`call_soon_threadsafe` → event loop） | 交出前 clone 的 ids、boxes、classes（G2 修復） | generation 檢查；舊 generation 的晚到 callback 丟棄並計數 | clone 排在已等過 `wb_done` 的 stream | consumer 擁有 clone | 同 stream 內保序 |

### 2.4 跨程序 resource state 與 storage／cognition

| ID | producer → consumer | buffer owner | snapshot／version | publish／completion | reuse／free 條件 | ordering |
|:--|:--|:--|:--|:--|:--|:--|
| AE-R1 | `AsyncDispatcher` 的 `VRAMLevelWriter` → orchestrator 的 `VRAMLevelReader` | writer 擁有快照檔；reader 不持有資源 | writer generation（63-bit 亂數）＋`decided_at_ns`＋CRC | 暫存檔寫完後 rename | 新 writer 啟動先發佈「尚未決策」；reader 每次讀都重新開檔 | rename 原子；freshness 上限 T（預設 1 s，推估值） |
| AE-M1 | 線上 producer → Redis stream（XADD） | Redis | message id | XADD 回傳 | `MAXLEN ~10000`（approximate）由 Redis 修剪 | **沒有線上 producer**：`entropy.py` 只在自己的 `__main__` 呼叫，`main.py` 的 callback 是 no-op（SE-21） |
| AE-M2 | Redis stream（XREADGROUP `>`，consumer 固定 `worker_1`）→ orchestrator 的背景 batch task → ACK | Redis PEL 持有未 ACK 的訊息 | message id | `gather` 全部 handler 之後 XACK | ACK 後由 Redis 移出 PEL | 每批一個背景 task；批與批之間沒有順序保證 |
| AE-M3 | `AsyncEmbeddingDispatcher` 的 `submit`／`put_crops` → worker → callback／Redis | 呼叫端的 crops；worker 在自己的 stream 上推理 | — | 輪詢 event 完成 | 結果 `clone()` 後交出 | **未接線**：#537 R-1 後 online 入口不啟動它，只有 `scripts/benchmarks/` 會建構 |

## 3. 表 B：容量、失敗與生命週期

| ID | capacity | overflow／backpressure | failure propagation | cancel／drain | restart／reset | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| AE-S1～S3 | 1 個 detection 在飛；2 個 pool；2 組 detection buffer | 不丟幀：host 決定節奏，每幀都處理 | 例外 → `RunCompletion` journal（[§23](failure_degradation_537.md#23-c0-10shipping-run-journal-的轉換語義)、CC-536-01-01） | signal／中斷語義不在 D1 重查（歸 CC-536-01-01） | track id、pool、NMS／GMC／tracker graph 每個 sequence 重建；whole-detect graph 依 key 沿用 | cited（#465）：[§14.4](../reference/native_runtime_resolved_config.md#144-驗收) detector 5316 幀、MOT 7/7、graph capture 7/7 EXACT；三個 measurement mutation 都 `DIFFERS` |
| AE-E1a | queue `prefetch+1`（預設 3） | 阻塞的 `put`：producer 等 consumer | `_DecodeFailure` 經 queue 送出，`__next__` 拋 `RuntimeError`，與 end-of-sequence 分開 | `_stop` → 清 queue → `join(3 s)`；逾時的 worker 只能寫進自己那一代的 queue | 每次 `__iter__` 換新的 worker、queue 與 decode stream | reproduced／GPU：[test_decode_stream_ordering.py](../../tests/unit/eval/test_decode_stream_ordering.py)（含正控制）；host-only：[test_decode_producer_contract.py](../../tests/unit/eval/test_decode_producer_contract.py)（#340 Phase 2B） |
| AE-E1b | DALI `prefetch_queue_depth=2` | 阻塞（iterator 拉取） | DALI 例外直接從 `next()` 拋出 | 檔案清單的暫存檔只在 `StopIteration` 時刪除 | `auto_reset=True` | documentation-supported |
| AE-E2～E4 | 1 個 detection／1 份延後輸出在飛 | 不丟幀 | 同 thread 例外直接拋出 | 不適用（同步 frame loop）；sequence 結尾 flush 最後一份延後輸出 | 每個 sequence 一個 `EvalPipeline` | AE-E2：cited（#465 oracle pins 與 parity）；AE-E3／E4：source-inspected |
| AE-E5 | 1 個 future | `result()` 阻塞 main | 背景例外在下一次 `result()` 時於 main 重拋 | main 先出錯時，未取的 future 不會被等待；executor 沒有 `shutdown()`（D1-F03） | 每個 `EvalPipeline` 一個新的 executor，舊的 thread 留到 interpreter 結束 | source-inspected |
| AE-E6 | 1 次在飛 | 注入前 host 阻塞 | native 例外直接拋出 | 不適用 | 每幀重設 | source-inspected |
| AE-E7 | 最多 N 個子程序 | `ThreadPoolExecutor(N)` 排隊 | **子程序失敗只印出，parent 仍算 metrics 並 `exit(0)`**（D1-F01） | parent 被中斷時子程序的處置未查 | 每個 sequence 一個新程序 | reproduced（§7） |
| AE-E8 | `n_threads` | unknown | unknown | unknown | unknown | unknown |
| AE-E9 | `--max-workers` | 同 AE-O5 | 同 AE-O5 | 同 AE-O5 | — | GPU：`--max-workers 1` 修復前後位元相同；4 workers segfault（§18） |
| AE-O1 | 1 格（最新一幀） | latest-wins；中間幀與「smart sampling」（像素差 < 2.0）**靜默丟棄，沒有計數** | callback 例外只印出；watchdog 例外由 main producer 向上傳遞（§11） | main 先取消並等待 watchdog／in-flight control，再以 thread 執行 `release()`；native call 沒有硬時限（§11） | `watchdog_loop` 5 s 無幀時重連；阻塞的 connect／restart／release 移出 event loop，失敗後繼續 backoff 重試（§11） | host contract＋真實 Python GStreamer／CUDA ingest（§11）；GE-08 不因此關閉 |
| AE-O2 | DALI 預取深度（預設） | 阻塞拉取 | `grab_tensor` 吃掉例外並回 `(False, None)` | `release()` 刪除 iterator 與 pipeline；main 的 connect 失敗／取消也進入 cleanup（§11） | `auto_reset=True`；`DALIMediaClient` 沒有 `watchdog_loop`，main 不對它建立 watchdog（§11） | documentation-supported；GE-07；§11 的 caller fixture 不提供 DALI GPU 證據 |
| AE-O3 | `asyncio.Queue(maxsize=100)` | 丟**新**幀，每幀計數 `dropped_queue_full` | worker 例外 → FAILED，保留原始錯誤（[§11](failure_degradation_537.md#11-538-d2-adasyncdispatcher-worker-失敗與停止)） | `stop()` 清 queue 並計數 `discarded_on_stop` | trackers 依 `stream_id` 存放，超過 `max_streams=8` 時 LRU 淘汰，該 stream 的 track 狀態**靜默重設**（只印 log） | host-only（§11）；GE-06 GPU-validated（fake detector） |
| AE-O4 | 每 stream `queue.Queue(maxsize=4)` | 滿時**清空整個 queue** 再放入新幀；`drops` 每次溢出 +1，不是每丟一幀 +1（D1-F02） | worker 例外 → FAILED（§13） | 有界 `stop(timeout)`，worker 存活時保留 batcher（§13） | 重啟換新 queue 與 worker | host-only（§12、§13）；GPU（§18） |
| AE-O5 | server 的 batch 上限 | `submit()` 阻塞等結果 | server 例外 → batcher FAILED，等待者全部收到失敗，新的 `submit()` 立即失敗（§18） | 同 AE-O4 | — | GPU-validated（§18；fake TRT） |
| AE-O6 | event loop 的 callback queue | 無上限 | `on_result` 在 event loop 上執行，例外由 loop 的 exception handler 記錄；pool 不進入 FAILED，也不讓 worker 結束（source 判讀） | 晚到 callback 丟棄並計數（§13） | generation | GPU-validated（§18）；host-only（§13） |
| AE-R1 | 單一快照 | 不適用（覆寫） | reader 不拋例外，回報 MISSING／STALE／INVALID | writer 正常停止時發佈「尚未決策」 | 新 writer 換 generation | reproduced（跨程序測試，[§7](failure_degradation_537.md#7-b0-f07-修復c0-060708-收斂)）；T 未量測；單一 writer 無強制 |
| AE-M1 | `MAXLEN ~10000` | Redis 修剪最舊的，**不看是否已 ACK** | XADD 失敗由呼叫端處理 | — | — | source-inspected；沒有線上 producer |
| AE-M2 | handler 並行上限 `Semaphore(32)`；**背景 batch task 數沒有上限**：loop 不等前一批就讀下一批 | 沒有往 Redis 的 backpressure | storage 例外被 handler 吃掉後仍 ACK；handler 外的例外讓 `gather` 失敗、不 ACK，訊息留在 PEL，`>` 讀取不會再交付（REQ-535-06-01）；`read_stream_batch` 裡 `json.loads` 失敗時，整批都已交付卻不會 ACK，同樣留在 PEL | `run()` 的 `finally` 只 `disconnect()`，未完成的背景 task 不取消也不等待 | consumer 名稱固定，兩個 orchestrator 會共用身分 | source-inspected；ACK 行為另有 [test_orchestrator_vram_policy_537.py](../../tests/unit/test_orchestrator_vram_policy_537.py) |
| AE-M3 | `asyncio.Queue(maxsize=64)` | `await put` 阻塞 | 例外時只對最後一個 `future` 設例外，同批其他 future 不會完成，也不呼叫 `task_done`；worker 的 stream 沒有等呼叫端的 stream | `stop()` 取消 task，不等待 | — | source-inspected；未接線 |

## 4. Frame／sequence／stream identity 與丟幀政策

各 profile 的政策不同，本稿只記現況，不要求一致。

| Profile | identity | 允許的亂序 | 允許的丟幀 | 序列／stream 切換時清除的狀態 |
|:--|:--|:--|:--|:--|
| shipping | sequence 目錄＋frame k＝1..`frame_end` | 不允許 | 不允許；列出的幀比 `seqLength` 少時寫出截短的 sequence，不算失敗（[resolved config 的幀數規則](../reference/native_runtime_resolved_config.md)） | track id、pool、NMS／GMC／tracker graph |
| eval | sequence 名稱＋`frame_id` | 不允許；延後輸出只延一幀，emit 順序不變 | 不允許；**缺 `seqinfo.ini` 的 sequence 被靜默略過**（D1-F06） | 每個 sequence 新的 `EvalPipeline`、decode worker generation、relink executor |
| online AsyncDispatcher | `stream_id`＋producer 的 wall-clock `timestamp`；沒有 frame 序號 | 不同 stream 交錯；同 stream 內 FIFO | queue 滿時丟新幀（每幀計數）；MediaMTX 層 latest-wins 與 smart sampling 丟幀沒有計數 | tracker 依 `stream_id`；超過 8 個 stream 時 LRU 淘汰並重設 |
| online WorkbenchPool | `stream_0..n-1`＋`timestamp` | 不同 stream 平行；同 stream 內 FIFO | 溢出時清空整個 queue（≤4 幀），計數單位是溢出事件 | 重啟換 generation；晚到 callback 丟棄 |
| cognition | Redis message id | 批與批之間沒有順序保證 | trim 會刪除未 ACK 的訊息；storage 失敗仍 ACK；FRESH EMERGENCY 時刻意卸載非 anomaly 事件（計數） | 無 |

## 5. 本次新發現

都沒有修復。`reproduced` 的原始輸出見 §7；其餘是 source 判讀，不推論發生率。

| ID | 發現 | 等級 | 影響的 D3 驗收項 | 路由（§9 已裁決者註明） |
|:--|:--|:--|:--|:--|
| D1-F01 | `mot17.py --processes`：子程序以非零碼結束時 parent 只印 `❌`，接著用剩下的輸出算 metrics 並 `exit(0)` | reproduced | error 不得偽裝成正常完成 | **已裁決**：#537 擁有修復契約，#538 只保留追溯連結；修復見 [failure_degradation_537.md](failure_degradation_537.md) §25 |
| D1-F02 | `WorkbenchPool.put_frame` 溢出時清空整個 queue（最多丟 4 幀），`drops` 只加 1；`AsyncDispatcher` 則是丟 1 個新幀、加 1。兩邊的計數單位不同，`drops` 不是遺失的幀數 | source-inspected（D1）；修復前重現與候選驗證見 §10 | overflow、丟幀與計數語義一致 | **已裁決**：#538 獨立 D3 切片；保留 drain-all，計數改為實際丟棄的幀數（§9）；候選修復見 [§10](#d3-f02) |
| D1-F03 | relink 背景寫入（`pipeline_relink`）的 executor 沒有 `shutdown()`；main 先出錯時，未取的 future 不會被等待，背景 thread 可能仍在改 relinker／bank | source-inspected | cancel／drain | #539（選用分支是否保留）後再決定是否修 |
| D1-F04 | `main.py` 的 producer 以 `asyncio.create_task(media.watchdog_loop())` 啟動 watchdog（8stream 沒有啟動 watchdog），沒有保留引用、不在 `supervise` 內、producer 結束時不取消；它的例外不會被收集。重連路徑的 `time.sleep(1)` 與 `connect()`（最多等 3 s 第一幀）在 event loop 上阻塞，同時卡住所有 stream 與 dispatcher worker | source-inspected | cancel／drain；error 不得被吞 | **已裁決**：留在 #538（event loop 阻塞與 task ownership） |
| D1-F05 | MediaMTX 的 latest-wins 與 smart sampling 丟幀沒有計數，`AsyncDispatcher` 的 `dropped_queue_full` 看不到 ingest 層的丟幀 | source-inspected | overflow 與計數語義 | 未個別裁決；F02 的計數原則是否涵蓋 ingest 層，待 F02 切片確認 |
| D1-F06 | `run_eval` 略過沒有 `seqinfo.ini` 的 sequence，不報錯；單一程序與 `--processes` 都 `exit(0)`，沒有輸出 | reproduced | error 不得偽裝成正常完成 | **已裁決**：同 D1-F01；修復見 [failure_degradation_537.md](failure_degradation_537.md) §26 |
| D1-F07 | Cognition 背景 batch task 數沒有上限；`run()` 結束時不取消也不等待；trim 不看 PEL；consumer 名稱固定 | source-inspected | backpressure、cancel／drain | #538（與既有的 PEL 不回收 REQ-535-06-01 一起），線上 producer 接通之前不急 |

**收窄的既有 OPEN**：GE-07（DALI buffer 所有權）由「未知」收窄為 documentation-supported、GPU-unverified（§1；owner 裁決見 §9）。仍然沒有在本 repo 以影片實跑，不能當成 GPU 證據；online qualification 前需要一個 GPU 測試（持有多幀的 DLPack tensor，確認之後的 `next()` 不改它們的內容）。

## 6. #340 trace 的證據依賴與處置選項

#538 原文要求 D1 盤點完成前不清除 `~/.local/state/saccade/perf/capture-race-incidence-20260909/traces/`。本節只回答「D1 的 edge 是否依賴這批 trace」，處置由 owner 裁決。

**內容**（2026-10-25 只讀檢查）：`traces/` 73 GiB，200 個 run 目錄（A／B 各 100），每個目錄有 observer 的 `cuda.jsonl`（約 430 MB）、`python.jsonl`、`maps.txt`、`manifest.json`、`observer_build.json`、`source.patch` 與 log。campaign 的其他部分很小：`runs/` 966 MB、`sealed/` 804 KB、`logs/` 2.4 MB，加上 `manifest.json`、`runs.jsonl`。

**依賴判讀**：

- [closure](../research/pipeline/closed/capture_race_incidence_closure_20260909.md) 的結論只用到 `runs.jsonl`／`sealed/`／`manifest.json` 的 verdict 計數。200 筆的 `attribution_status` 全部是 `not_applicable`：沒有 failure，第二層分析（[analyze.py](../../scripts/tools/capture_attribution/analyze.py) 讀 `cuda.jsonl`、`python.jsonl`、`maps.txt`）從未執行。
- 本稿沒有任何一列引用這批 trace。和 #340 有關的 AE-E1a，證據是 Phase 2B 的測試與 decode worker 的 docstring（cudart tally 來自另一個 attribution campaign），不是 incidence campaign 的 trace。
- 結論：**D1 不依賴這批 trace**。它們的剩餘價值只有：日後在同一座標上重新分析 CUDA API 序列（例如 legacy stream 的足跡），或發生類似的 900／901／906 時當作對照。

**處置選項**（沿用 #538 原文的封存前提：先保留 manifest、`runs.jsonl`、`sealed/`、`logs/` 與原始 SHA-256／檔案對照，解壓後要逐位元組相同，並確認 `analyze.py` 對原始路徑的需求）：

| 選項 | 空間 | 代價 |
|:--|:--|:--|
| 保留原狀 | 73 GiB | 無 |
| 無損壓縮 `cuda.jsonl` | 單檔實測 zstd `-19` 約 45 倍（430 MB → 9.6 MB，29 s）；全部約 1.6 GiB 是外推，未驗證 | 約 100 分鐘 CPU；重新分析前要解壓，`analyze.py` 讀的是解壓後的固定檔名 |
| 刪除 `traces/`，保留其餘 | 約 1 GiB | 不能再做第二層分析；closure 的計數不受影響 |

**已裁決**：`.tar.zst` 冷封存，一次性處置，流程見 §9。驗證成功之前不刪除任何原始檔。

## 7. 證據與重現

**本稿新增的探針**（原始輸出、指令與 SHA-256 在 repo 外：`~/.local/state/saccade/evidence/538-d1/20261025T110804/`，`README.md` 是索引）：

| 探針 | 指令（摘要） | 結果 |
|:--|:--|:--|
| D1-F01 | 兩個 sequence 目錄只放缺少 `[Sequence]` 的 `seqinfo.ini`；`mot17.py --processes 2 --sequences BAD-A-SDP,BAD-B-SDP --data-root <probe>/data --split train --output <probe>/out` | 兩個子程序都是 `failed with code 1`（`configparser.NoSectionError`），parent `exit=0` |
| D1-F06 | sequence 名稱不存在：`--processes 2` 與單一程序各一次 | 都是 `exit=0`、沒有 MOT 輸出；`--processes` 時子程序回報成功 |
| DALI 建構 | 建構 `DALIVideoPipeline`（不 build，不需影片），讀 `exec_dynamic` | `True` |
| #340 trace | 對 `traces/0000-A/cuda.jsonl` 執行 `zstd -19 -T0 -c … \| wc -c`（只讀） | 430,856,370 → 9,555,505 bytes |

環境：RTX 5070 Ti Laptop、WSL2、Python 3.12、DALI 2.0.0、`main@3375e7d4`。D1-F01／F06 會載入 TRT engine，需要本機的 `models/yolo/` 與 GPU；CI 沒有對應測試。

**引用的既有證據**：shipping 見 [resolved config §14](../reference/native_runtime_resolved_config.md#14-native-double-buffercuda-graphpr-10u5)；AsyncDispatcher、WorkbenchPool、caller lifecycle 與 D2-GPU 見 [failure_degradation_537.md](failure_degradation_537.md) §11～§18；VRAM 快照見同檔 §7 與 [degradation §6](../modules/resource/architecture_degradation.md#6-vram-level-跨程序快照契約537-b0-f07)。

## 8. 對照 D3 驗收（現況，不是結案宣告）

- [x] 主要 async edge 有 ownership／completion／release 規則，或具名的未知與後續 owner：§2～§3 已逐條列出；未知集中在 AE-E8（C++ eval pool）與 AE-O2／AE-E1b（DALI，documentation-supported），路由見 §5。Owner 已裁決 D1 inventory completed（§9）。
- [x] WorkbenchPool 的有限狀態契約先於修復固定（D2-WB-0 → WB-1，已合併）。
- [x] 至少一個故障注入重現了具體違約（D2）；本稿另外重現 D1-F01、D1-F06，其餘標 source-inspected。
- [~] overflow、丟幀與計數語義一致：F02 的候選修復與待完成門檻見 [§10](#d3-f02)；D1-F05 的 ingest 丟幀仍未處理。
- [~] error／cancel／timeout 不偽裝成正常完成：D1-F01／F06 已由 #537 修復，F04 的 owner／drain 契約見 §11；D1-F03、D1-F07 仍待處理。
- [x] host-only、GPU-validated、documentation-supported 與未驗證的證據分開記錄（§1、§3）。
- [x] #465 的 native schedule、capture freeze 與 trace／measurement 邊界沒有改動；本稿只引用。G 的 contract／test／limit 映射尚未交付。

## 9. Owner 裁決（2026-10-25，引用）

決策來源是 [#538 comment](https://github.com/raylei50653/saccade/issues/538#issuecomment-6393037794)；本節只摘錄要點，以 comment 為準。

- **D1**：inventory completed。批准發布文件 Draft PR；文件與證據審核後可合併，不等待 §5 的缺陷全部修復。不授權藉此擴大 online 支援範圍。
- **D1-F02**：不同 profile 可保留不同的丟幀策略，但計數單位統一為**實際丟棄的幀數**，禁止沒有記錄的批次丟棄。WorkbenchPool 保留 latest-wins／drain-all。新統計契約：
  - `drops`／`workbench_drops`：實際丟棄的幀數；
  - `overflow_events`：觸發溢出的次數；
  - `discarded_on_stop`：獨立計算，不混入溢出；
  - 每條 stream 保留 frame identity 與可觀察的 frame gap。

  例：清掉 4 幀再放入新幀 ⇒ `drops += 4`、`overflow_events += 1`。修復要保證並行 producer 下 drain 與 enqueue 不會產生沒有記錄的丟幀，測試涵蓋競態、實際丟棄數與 telemetry 一致性。修復是 #538 獨立的 D3 小切片。
- **D1-F01／F06**：交 #537，修復契約由 #537 維護；本稿只保留追溯。
- **D1-F04**（watchdog）：event loop 阻塞與 task ownership 留在 #538。
- **#340 trace**：`.tar.zst` 冷封存，是一次性處置，不建立自動封存政策：
  1. 保留 manifest、`runs.jsonl`、`sealed/`、`logs/`，建立原始路徑與 SHA-256 清單；
  2. 只對 trace 建立 `.tar.zst`，記錄工具版本與建立資訊；
  3. 完整還原並逐檔核對 SHA-256，確認逐位元組相同；
  4. 驗證成功後才刪除未壓縮的 trace。

  全量的壓縮率與耗時以首次封存的實測為準。
- **GE-07**：source／文件支持、GPU-unverified，不能取代 GPU 端到端測試（stream completion、PyTorch consumer、實際 pipeline 組態）。online qualification 維持未通過。
- **執行順序**：D1 審核並合併 → #537 F01／F06、#538 F02 與 watchdog 分別處理 → 獨立執行 #340 封存。

<a id="d3-f02"></a>

## 10. D3-F02：WorkbenchPool 溢出計數與停止競態

**範圍與狀態**：依 §9 的 F02 裁決接續 `22c01fcd5a3ab5ff01c7f248f768c4be3eae9769` WIP；基線是 `main@aa69c3c1caf502ecc5aeeda92f90d8db6871a6c3`。已由 #637 原子合併為 `6ca331d6`，獨立 publication #636 同時落地；[完成證據](https://github.com/raylei50653/saccade/issues/538#issuecomment-6409963417) 包含 PR／main CI、最終 review 與 strict attested。只修 `WorkbenchPool` 的 host queue admission、計數與對應停止競態；不修改 CUDA 排程、模型、tracker 參數、AsyncDispatcher 的丟幀策略或線上支持範圍。D1 的 F02 表格保留原盤點座標，現況以本節為準。

### 10.1 計數與 identity

| 欄位 | 定義與範圍 |
|:--|:--|
| `drops` | 實際從滿 queue 清掉的幀數，從最近一次 `reset_stats()` 起累積 |
| `overflow_events` | `put_frame()` 第一次 enqueue 遇到滿 queue 的次數；即使 consumer 搶先取走全部幀而清掉 0 幀，仍是一個事件 |
| `workbench_drops`／`workbench_overflow_events` | 相同事件的 telemetry；由 telemetry 物件累積，pool 的 `reset_stats()` 不重設它 |
| `streams[stream_id]` | pool 物件生命期的 `submitted`、`dequeued`、`dropped`、`overflow_events`、`discarded_on_stop`、`last_dequeued_seq`、`gaps`、`gap_frames` 及當下 `queued`；stop/start 與 `reset_stats()` 不清掉它們 |
| `discarded_on_stop` | stop 清掉的 queued 幀，與 overflow drop 分開，重試 stop 不重複計數 |

每條 stream 在接受 frame 時配置從 1 開始、單調遞增的 `seq`。worker 取出 frame 時記 `last_dequeued_seq`；跳過的 identity 數量計入 `gap_frames`，每段缺口計入 `gaps`。這裡的 `dequeued` 表示交給處理，不表示成功處理或 callback 已交付；處理失敗與 late callback 仍走既有契約。現有 callback 的 stream ID／timestamp 介面不變。

所有 producer／worker 安靜後，每條 stream 滿足：

```text
submitted = dequeued + dropped + discarded_on_stop + queued
```

仍在執行中的 `get_stats()` 是逐 stream 的觀測，不是整個 pool 與 telemetry 的原子快照。比較總數時須保持相同統計區間：全域 `drops` 可 reset，per-stream 與 telemetry 則累積。尾端被丟棄的幀在下一個 frame 被 worker 取出之前，尚不形成已觀察的 gap。

### 10.2 並行 admission 與有界停止

- 保留 latest-wins／drain-all、每 stream queue 容量 4。per-stream lock 涵蓋第二次 RUNNING 檢查、identity 配置及完整 drain-and-enqueue；worker 仍可以取走舊幀。計數以實際 `get_nowait()` 取出的數量為準。
- producer 在 lock 外已看過 RUNNING，取得 lock 後仍須再查一次；停止開始後才取得 lock 的 producer 會被拒絕，不增加 `submitted`。已通過第二次檢查的 admission 必須完成，再由 worker 或 stop 收帳。
- `stop(timeout)` 的 worker joins 與等待 stream lock 共用 deadline。lock 逾時時維持 STOPPING、保留 batcher、增加 `stop_timeouts` 並拋 `WorkbenchPoolStopTimeout`；之後可以重試。每條 stream 一清完就更新全域與該 stream 的 `discarded_on_stop`，後面的 stream 逾時不會遺失前面的計數。batcher shutdown 仍另有既有上限（[#537 §13](failure_degradation_537.md#13-538-d2-wb-1workbenchpool-有界停止worker-failure-與資源釋放)）。
- lock 會改變競爭下的交錯順序，因此可能改變實際交付哪些幀；保留的是上述有界 queue／latest-wins 契約，沒有宣稱修復前後的丟幀樣本、輸出、吞吐或精度等價。不同 stream 各有自己的 lock。

**審查發現與處置**：對原 WIP `22c01fcd` 重現一項停止回歸：producer 持有 stream lock 時，`stop(timeout=0.1)` 等到 1 秒後才返回，且回報正常停止。原因是 worker joins 之後的 lock acquisition 沒有上限。候選改為上述 deadline 內取得 lock，並補停止後拒絕、保留 batcher、重試及部分 drain 的測試。

### 10.3 證據與尚未完成項

契約測試：[test_workbench_pool_overflow_contract_538.py](../../tests/unit/test_workbench_pool_overflow_contract_538.py)。CPU 情境沿用 D2 的獨立子程序 harness 與外部 timeout；GPU 情境沿用 D2-GPU 的真實 CUDA Workbench／batcher，只替換 TRT inference。

- 修復前重現：送 6 幀，worker 收到 `[0.0, 5.0]`，實際丟 4 幀卻報 `drops=1`。重播使用的 dispatcher git blob 與基線 `aa69c3c1` 相同。
- 已提交 head `a06f56abaefb6e5ea4ef86abad2ea282afffa6af`、本次新 build：四份相關契約測試 **43 passed／0 skipped**（30 個 CPU、13 個 GPU），RTX 5070 Ti Laptop。這是上述局部測試後，以新 native extension 做的完整相關回歸。
- CPU 範圍：兩 stream／16 producer threads 的 6,400 次提交、overflow 與 telemetry 對帳、stop discard 分離、重啟 identity gap、admission／stop 競態及部分清理後的重試。
- CUDA 範圍：既有 D2-GPU 契約與新增 F02 案例。新案例固定第一幀正在 inference，再排入 5 幀：`drops=4`、`overflow_events=1`、callback timestamps `[0.0, 5.0]`，並核對真實 CUDA 前處理交給 batcher 的兩份 input 值。
- 6 個隔離 mutation 都被行為斷言攔住：把 drops 算成事件、telemetry 算成事件、取消 stop lock timeout、遺失部分 stop 計數、重啟重設 identity、移除 admission 的第二次狀態檢查。
- 第一次失敗分類：新增的 stop 回歸測試在原 WIP 上失敗（`returned`，應為 `timeout`）；修復後通過。沙箱內 NVML／CUDA 被作業系統阻擋；已在沙箱外核對裝置並執行上述 GPU 測試，沒有把 skip 當通過。
- 完整 pre-push 首次執行於 combined head `756dde4ebe580f2f551ed3ea04162b54e818f1a4`：6,241 passed、195 skipped、105 deselected、5 xfailed、1 failed。唯一失敗是既有 `test_h2_import_witness.py::test_the_recorder_records_dependencies_rather_than_filtering_them`：驗證 checkout 的 `.venv` 是指向 repo 外的 symlink，`ImportRecorder.note_file()` 依既有契約先 `resolve()` 再判斷是否位於 repo 內，所以沒有記錄該依賴。該 recorder 與測試相對基線完全未改；同一測試在原工作區通過。改以同一 head 的 `.worktrees/550-f02-validation`、repo 內的實體 `.venv` 目錄（同磁碟 hard links，禁止 sync 與 bytecode 寫入）驗證後，整份 import-witness suite **34 passed**。此項分類為已定位的驗證環境問題，沒有改寫或跳過測試。
- 修正環境後，**同一 combined head** 的完整 `scripts/pre_push.sh` exit 0：**6,242 passed、195 skipped、105 deselected、5 xfailed、0 failed**。格式、mypy（605 files）、文件與其餘適用檢查也通過；這是本機結果，沒有遠端 CI 結果。以下收尾只增加文件中的驗證紀錄，不改實作、測試或 identity 檔案內容。

本機原始輸出與 mutation runner 在 `/tmp/saccade-550-f02-evidence/`；來源與摘要檔 SHA-256：

重播指令（在候選 worktree；GPU 測試需要可存取裝置的 host）：

```bash
PYTHONPATH=build/runtime_identity:src .venv/bin/python -m pytest -q \
  tests/unit/test_workbench_pool_overflow_contract_538.py \
  tests/unit/test_workbench_pool_lifecycle_contract_538.py \
  tests/unit/test_workbench_pool_lifecycle_observed_538.py \
  tests/unit/test_workbench_pool_gpu_contract_538.py
.venv/bin/python /tmp/saccade-550-f02-evidence/run_mutations.py
```

| 檔案 | SHA-256 |
|:--|:--|
| `src/saccade/perception/dispatcher.py` | `c0b9a68eac09ef67b7a8e9df3380fc98cebdb909b6779459c6ae138a1af53afe` |
| `tests/unit/test_workbench_pool_overflow_contract_538.py` | `e29c073237f4bbcd7937011132f30e7e1e386e57fdadc774e421a78a020849a4` |
| `run_mutations.py` | `21a7ec01f720484a72070bdfa99bb5997169392108beb8c064d3fa9986dfb5d4` |
| `mutations.json` | `2a0d106a6ff8db957a7ddad5b895aad240c0ae9aef1a03de96d13d786d68c25c` |
| `final-contracts.log` | `3a69d4032550ea4bcc409ca9c2eb915b2a31a68ba407000de9e2de3823343b96` |
| `final-contracts.xml` | `4c3b1f1daec87805add348a84c2c4fca3868b42d1c537d727838acc888e03e94` |
| `pre-push.log`（首次失敗） | `f37f3683d01bcd4388c9d77ca14f8e0978501028b8ba54d4e8400536d4e05483` |
| `pre-push-local-venv.log`（環境修正後通過） | `e28174382d9597fd174efa575ff7f5cc599987c3e580ab6f2b7bdbab17d1422e` |
| `attested-strict.log` | `6f66d4d2da35c315d08ac546f9d406dc828bf9378a7619db41ebb586d58e750d` |

**Runtime identity 候選**：在 clean `a06f56ab` 上獨立 build `saccade_tracking_ext`／`saccade_scan_plugin`，依 runbook 重跑 MOT17-09 identity probe：

```bash
PYTHONPATH=src .venv/bin/python scripts/tools/build_runtime_identity.py \
  --run-probe --build-dir build/runtime_identity \
  --emit /tmp/saccade-550-f02-evidence/runtime_identity.candidate.json \
  --require-complete
```

候選 `publication_complete=true`；只有 implementation 改變（`11263f7c… → 0cb15155…`，唯一變動 member 是 `dispatcher.py`），其餘四軸相同。probe 是 fresh run，digest 仍為 `2dabed0b…`；`equivalence=unproven`。候選檔 SHA-256：`f36316235124b7edde843e262b46248299b99077f25aa47c50e1784d51bdb5b2`；capture log：`b52b5757db0b1d98785ad4ed2939e31fc2e4e27fd87c3d8bbd2d7cc396f8cd1e`。候選只寫入 scratch，publication 應以 canonical＋archive 的獨立 review surface 與實作原子落地。

本機 publication 分支已準備 canonical＋archive 的獨立兩檔 commit，archive 與 `main@aa69c3c1` 的原 canonical 逐位元組相同（SHA-256 `d740889a2fe15e58bd868a9af5b8e9ae23b033c76cca1caa068562e65407d85b`）。另以 `h2_behavioral_identity.py --identity-mode` 重跑驗證 probe、`h2_runtime_inputs.py --build-dir build/runtime_identity` 重算完整 manifest，再執行 `check_runtime_identity_staleness.py --mode attested --strict --probe-from <verification-probe.json> --runtime-inputs-from <runtime-inputs.json>`，exit 0；這不是只驗 portable axes 的結果。

完整 pre-push 的環境保護參數為 `UV_NO_SYNC=1`、`PYTHONDONTWRITEBYTECODE=1`；`UV_CACHE_DIR` 指向 `/tmp/saccade-550-uv-cache`。`PYTHONPATH` 依序指定本次新 build、原工作區其他 native extensions，以及驗證 checkout 的 `src`。這些參數不跳過任何檢查。

**當時的交付門檻**：獨立 publication review、最終 head review、PR／required CI 與 atomic co-land 後續已完成，見本節開頭的完成證據。WorkbenchPool online qualification、D1-F05 ingest 丟幀、多 worker NMS capture 與其餘 D1／D2 殘留仍 OPEN；watchdog（F04）另見 §11；沒有清除 #340 trace。

回退需要同時回退實作並依 [runtime identity runbook](../reference/runbooks/runtime_identity_republication.md) 重發座標；不能只把 canonical publication 換回舊檔。

<a id="d3-f04"></a>

## 11. D3-F04：watchdog 與 media control 的 ownership

**範圍**：基線 `main@6ca331d60e8fbb002fdfbd72487c60de6aee161b`，依 §9 owner 裁決與 [§0A 方案紀錄](https://github.com/raylei50653/saccade/issues/538#issuecomment-6436053501) 修復 main 的 MediaMTX watchdog。只修 host control／task lifecycle；8stream 仍沒有 watchdog，沒有變更 CUDA 算子、drop policy、tracker 或 online qualification。D1 的 F04 列記錄修復前的盤點；現況以本節為準。

### 11.1 控制與清理契約

- main 持有 pump 與 watchdog task，以既有 `supervise` 收集失敗。watchdog 拋例外或意外正常返回都讓 producer 失敗，進一步由上層取消其他 producer；不再留下無人收集的背景失敗。
- `connect_async`、watchdog restart、`release_async` 用 thread 執行阻塞的 GStreamer 控制。每個呼叫持有並 shield 自己的控制 task；取消及重複取消都先等待該工作真正結束，再傳出取消。不能用「coroutine 已取消」推斷 native thread 已停止。
- producer 的正常結束、錯誤、取消、初始 connect 失敗與部分 task 建立失敗，都先取消並等待 pump／watchdog，再 release。清理例外不取代原始錯誤；取消期間控制工作另失敗時，錯誤被收集並附在取消的 note。
- watchdog 的生命週期由 owner 取消並 await 結束。同步 `release()` 仍供同步 caller 與 restart 內部使用，**不再是 watchdog task 的停止訊號**。不得在仍有 watchdog／control 的時候並行呼叫 release。第一次 reconnect 失敗雖將 `_running` 清為 false，watchdog 仍繼續原有 backoff（設定值 1、2、4…至 30 s）；健康檢查間隔仍為 5 s。connect 會重設 freshness 時間，因此實際嘗試間隔還包含 freshness／poll 等待，並非恰好等於 backoff 設定值。
- main 只為 MediaMTX 建立 watchdog；DALI connect／release 留在原 thread，沒有替 DALI 增加監控或 GPU 保證。真正存在的本機檔走 DALI；明確 RTSP URL 傳入 `rtsp_url`；不存在的預設影片維持預設 RTSP fallback，但不再把它當作 `dummy_video` 而停用監控。
- native connect／release 若永久卡死，取消也必須繼續等它；本切片沒有硬停止時限。需要終止這種工作時應使用 process supervision，不能同時釋放仍被控制工作使用的管線。實際 GPU 前處理仍可能造成 loop 延遲，本節不宣稱 latency SLA。

### 11.2 測試與真實環境證據

契約：[test_watchdog_contract_538.py](../../tests/unit/media/test_watchdog_contract_538.py)，**17 passed／0 skipped**。每個情境使用獨立 interpreter 與外部 timeout，涵蓋 initial／reconnect 的 heartbeat、失敗後繼續重試、取消與重複取消時的 control drain、task／錯誤收集、來源建構、DALI 清理及原始錯誤保留。既有 caller lifecycle 與 MediaMTX 測試另有 **33 passed**。這些 controlled fixture 只證明 host 契約。

基線 `6ca331d6` 的三個負控制都攔住既有缺陷：同步 reconnect 阻塞 heartbeat、第一次 reconnect 失敗後 watchdog 退出、DALI connect 拋例外後沒有 release。新增測試的首次失敗是 harness 用 `gather(return_exceptions=True)` 取得被重建的 `CancelledError`，看不到原 note；改為直接 await／catch 原 task 後通過，沒有為此改 runtime 或放寬斷言。

另有兩個隔離 mutation：移除 shield／drain，測試發現 connect thread 還被 gate 擋住時取消已返回；改成 unowned watchdog，測試發現 release 早於 watchdog 結束及 1 個 leftover task，**2/2 DETECTED**。後者第一次逃過 fixture：fake watchdog 等待沒有外部引用的 bare Future，孤立 task 可被 GC 並執行 finally。已把 tracked fixture 改為真實 watchdog 同樣會使用的 timer-held sleep，再以相同測試攔住 mutation；runtime 沒有因這項修正而變動。

真實驗證在 RTX 5070 Ti Laptop、Python 3.12.13、Torch 2.11.0+cu130、GStreamer 1.28.8、MediaMTX 1.19.1 上執行。使用自有 localhost 18554 server、ffmpeg H.264 publisher 與真正的 main producer，檢查 CUDA tensor；dispatcher 只記錄與丟棄輸入，不執行 detector／tracker。沒有干預既有 8554 服務。

| 故障注入 | 觀察與適用範圍 |
|:--|:--|
| 停掉 publisher，再恢復 | 真實 reconnect 為 false、false、true，之後恢復交付；initial connect 有 198 次 heartbeat，三次 reconnect 各有 104／105／203 次（10 ms 排程） |
| initial connect 的真實 first-frame 等待中重複取消 | 等控制完成約 3.001 s 才最後 release；期間 297 次 heartbeat，沒有之後重新開啟管線 |
| reconnect 的真實 first-frame 等待中取消 | 等控制完成約 3.008 s 才最後 release，沒有重新開啟；取消前已交付 76 個 CUDA frame |
| 真實交付影格後讓 watchdog 拋 OSError | 經 producer 傳出，sibling 被取消、media 釋放；沒有遺留 asyncio task |

自有 server／publisher 全部已停止。上述不提供 C++ media opt-in、DALI buffer 所有權、detector／tracker、完整服務或永久卡死 native call 的驗證。

本機重播指令（probe／負控制原始檔在 repo 外）：

```bash
PYTHONPATH=build/runtime_identity:build:src .venv/bin/python -m pytest -q \
  tests/unit/media/test_watchdog_contract_538.py \
  tests/unit/media/test_mediamtx_client.py tests/unit/test_caller_lifecycle_538.py
timeout 130s .venv/bin/python -u /tmp/saccade-538-f04-evidence/live_media_probe.py
.venv/bin/python /tmp/saccade-538-f04-contract-evidence/baseline_negative.py
```

| 檔案 | SHA-256 |
|:--|:--|
| `main.py` | `f1af650605394defe2d665dda3532f0589ef8ee8f5f1b251169bf300259139ac` |
| `src/saccade/media/mediamtx_client.py` | `3a305c34e287ad4010b312fdbe32569cc8c4698b64b7d2415468ec4dbc69f623` |
| `tests/unit/media/test_watchdog_contract_538.py` | `9afad06419139ac54f0664cee6e9bc989fc053d2ef49f424a3bd10d1eca44bcc` |
| `/tmp/saccade-538-f04-contract-evidence/contracts.log` | `751b79c339cf15423abe4fceb88e10efe4d01ff2bef769b4a75c4618dd10b48e` |
| `baseline_negative.py` | `e98d1ab7371d910c8759529adbd994e00b67656898a1e988474672827c526eae` |
| `baseline_negative.log` | `432d95eb876ccbe9f2bc5027f2bd1ece9b1d7337c3b4af7129a86d402425cc02` |
| `mutations.py` | `0f91abbb6dcfd9a26c50e086338a7b12c2dae0b450e0a9b41d02c19b7c443e4a` |
| `mutations.json` | `6e5b4d9eac8f593e474a65079c4df6b1e7bf4ca18a5972b7aeab7541990d5ab2` |
| `/tmp/saccade-538-f04-evidence/live_media_probe.py` | `12cbf0138d1e27f043c0ee7b6967069464d77e66f3475a90eae0dcdafe8de50b` |
| `live-media-results.json` | `4edde223b4af9f654bbcc6506579f09a937b1910634648aaa1252baf22733e7b` |
| `live-media.log` | `96c506a0cc27ac77f0e3cb95a987370ccf342014199c33781d722e3359cdc330` |

### 11.3 交付與剩餘邊界

在 clean implementation `596838f8` 上重新 build native extension／scan plugin，執行 fresh MOT17-09 identity probe 並建立完整 runtime-input manifest。候選 `publication_complete=true`，只有 implementation 軸 `0cb15155… → 6237a7ff…` 改變（`main.py`、`mediamtx_client.py`），其餘四軸相同；probe digest 仍為 `2dabed0b…`，`equivalence=unproven`。候選 SHA-256 `30d793a83dfc6678df230c345142e40cedd11919c1ce74c34664ed5a09ea8635`；capture log `a44dfd1e84c5bd5033c80f445725fe92b037ecfd516b6742381fc5df4db97243`。其後的 fixture／文件 delta 不改任何 identity 軸。

implementation 與 runtime identity 依 runbook §3.2 經獨立 review surface 原子落地；完整本機回歸、publication、最終 head review 與 PR／main CI 的可追溯結果記在本切片 PR 及 #538 checkpoint。runtime identity 的相同 probe 不證明 semantic equivalence。

F05 ingest 丟幀、F03／F07、GE-07、多 worker NMS capture、`update_into`／NV12 與 online qualification 仍 OPEN；distribution 維持 local-only、equivalence 維持 unproven。沒有清除 #340 trace；後續依 §9 另做一次性封存。回退必須包含實作與依 runbook 重發的座標，不能只還原 canonical。
