<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-10-25 -->
<!-- doc-module: cross -->

# 非同步 edge 與 buffer ownership 盤點（#538 D1）

本稿是 [#538](https://github.com/raylei50653/saccade/issues/538) 的 **D1**：依 #535 的 profiles，盤點主要的非同步 edge，逐條記下誰持有資料、何時算完成、何時可以重用或釋放、容量與溢出、失敗怎麼傳、取消與重啟時怎麼收尾，以及支撐這些說法的證據等級。**本稿不修改行為，也不選定政策**；新發現的缺口只列出並路由（§5）。Owner 對發布、路由與 #340 trace 的裁決見 §9（決策來源是 #538 的 comment）。閱讀入口是 [architecture README](README.md)。

**Source baseline**：`main@3375e7d4`（#629 merge，2026-10-25）。除 §5 標 `reproduced` 的兩項與 §1 註明的 DALI 建構觀察外，本稿新增的判讀都是 `source-inspected`；引用的既有證據保留原文件的等級與座標，本稿沒有重跑。

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
| AE-O1 | 1 格（最新一幀） | latest-wins；中間幀與「smart sampling」（像素差 < 2.0）**靜默丟棄，沒有計數** | callback 例外只印出 | `release()`：pipeline 設 NULL、`mainloop.quit()`、`join(1 s)` | `watchdog_loop` 5 s 無幀時重連；重連用阻塞的 `time.sleep(1)` 與 `connect()`，跑在 event loop 上（D1-F04） | source-inspected；GE-08 |
| AE-O2 | DALI 預取深度（預設） | 阻塞拉取 | `grab_tensor` 吃掉例外並回 `(False, None)` | `release()` 刪除 iterator 與 pipeline | `auto_reset=True`；**`DALIMediaClient` 沒有 `watchdog_loop`**，`main.py` 的 DALI 分支在 `create_task` 時就失敗（ledger C4） | documentation-supported；GE-07 |
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
| D1-F01 | `mot17.py --processes`：子程序以非零碼結束時 parent 只印 `❌`，接著用剩下的輸出算 metrics 並 `exit(0)` | reproduced | error 不得偽裝成正常完成 | **已裁決**：#537 擁有修復契約，#538 只保留追溯連結 |
| D1-F02 | `WorkbenchPool.put_frame` 溢出時清空整個 queue（最多丟 4 幀），`drops` 只加 1；`AsyncDispatcher` 則是丟 1 個新幀、加 1。兩邊的計數單位不同，`drops` 不是遺失的幀數 | source-inspected | overflow、丟幀與計數語義一致 | **已裁決**：#538 獨立 D3 切片；保留 drain-all，計數改為實際丟棄的幀數（§9） |
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
- [~] overflow、丟幀與計數語義一致：**不一致**（D1-F02、D1-F05）。F02 的方向已裁決，修復尚未實作。
- [~] error／cancel／timeout 不偽裝成正常完成：D1-F01、D1-F06 違反；D1-F03、D1-F04、D1-F07 待處理。
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
