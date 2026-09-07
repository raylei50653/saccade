# #340 原始 failure provenance 恢復

調查 checkout：`0d757e17`。範圍為歷史證據恢復與 attribution harness；不重跑
failure-rate，不修改 capture semantics。這份恢復不是完整 runtime attestation。

## 已恢復的 primary evidence

原始 GPU-decode `cudaErrorStreamCaptureImplicit` 來自 **第一次、lazy-capture
版本的 `b2_1_B`**，不是現在同名的 run，也不是後來的 `tmp/f3c-nopre` arm C。
證據來自當時工具輸出的 transcript：
`/home/ray/.claude/projects/-home-ray-developer-ai-saccade/4b613efa-080a-4ba6-8e0c-e20d0ae89ab7.jsonl`。
可用 [recover_failure.py](../../../scripts/tools/capture_attribution/recover_failure.py)
抽取精確 tool ID、時間與原始內容至一個新的目錄；不執行 inference。

| UTC 時間 | 直接觀察 | 證據 tool ID |
|:--|:--|:--|
| 02:03:51 | GPU = RTX 5070 Ti Laptop；nvidia-smi driver = 616.56 | `toolu_01BJExtLyMEY8vQ8UpiTnemz` |
| 02:15:23 | amend 後顯示 `806c52cf`；清除前一輪 A/B 輸出 | `toolu_01RETdTPD64ru7h9iSdFSvFU` |
| 02:16:06 | 啟動 `ab.py 12` | `toolu_01HYW4W44Lb5eYr8iSBNWq2K` |
| 02:28:23 | `b2_1_B` 為異常列，output digest prefix `320867b9ac`，IDF1 78.8 | `toolu_01BebPzotydyAfmiYF8pAcjV` |
| 02:28:48 | 從 `runs/b2_1_B.log` 讀到 leased slot 0/1 與 Implicit traceback | `toolu_01PM2BKuiHAJrCqRM1yi3QiN` |
| 02:30:31 | 當時 HEAD 輸出 `806c52cf` | `toolu_01RvN6GSgA7iQyvuWD4qy2MH` |
| 02:30:56 | worker traceback，`currentStreamCaptureStatusMayInitCtx` | `toolu_01KzYmHcpjgNy33nMybRREez` |
| 02:32:51–02:34:16 | 才加入 precapture 並 amend | `toolu_0194m4enxjhaYYQcBXGqbjrt`、`toolu_01TdJ4fx4HE4JyFeWjDwkBFU` |
| 02:36:42 | 清除 `runs/`、`ab_results.jsonl` 等，重用 run IDs 重跑 | `toolu_01U6r5AC5S2eymbsUJFXMgu6` |

保留的 `ab.py` 指向命令：

```text
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m
  --detector SDP --double-buffer
  --output /home/ray/.local/state/saccade/perf/f3c-20260906/runs/b2_1_B
```

沒有 `--no-gpu-decode`。保留 harness 的 `toggle.py` 只切換 `pipeline.py`、
`pool.py`、`mamba_gated_detector.py`，不是切換整個 checkout。
`806c52cf8ced0836c80606559f7c38a5fcc546a3` 是時間線支持的 candidate source，
不能當作該 run 所有 source、native libraries、engine、checkpoint 的 byte identity。

## 會改變 attribution 設計的線索

保留 traceback 的原 log 行 193：

```text
Exception raised from currentStreamCaptureStatusMayInitCtx at
.../torch/include/c10/cuda/CUDAGraphsC10Utils.h:72
```

本機 PyTorch 同名函式呼叫 `cudaStreamIsCapturing(current_stream, &status)`。
這是失敗位置的強線索，但尚無原始 CUDA API return trace，不能把本機 header
視為原 run 的 library byte attestation。正常 decode 的 WaitEvent tally 不能證明
此失敗發生於 WaitEvent。Harness 必須涵蓋 status query、WaitEvent 與 begin/end。
[NVIDIA stream API 文件](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
亦說明 legacy stream 上的 `cudaStreamIsCapturing` 在 blocking stream capture 期間
可回傳 Implicit，且該 query 本身不會 invalidate blocking capture。

原摘錄包含 overall throughput 與非 baseline metrics；所以「sequence 一定直接
中止，結果不可能看似成功」不能當作原始事件的完整描述。Worker traceback 是可見的，
但 run 是否以 nonzero exit 結束、輸出截短範圍及確切失敗 sequence，尚未恢復。

## 未恢復與停止條件

- 原完整 log／JSONL 已被刪除並重用名稱；現在同名檔案不得接替原證據。
- 缺少 per-run dirty tree、source hashes、完整環境與套件／native library／資產 hashes。
- 尚無失敗 API 的 numeric return、stream handle/flags、thread/capture timeline。
- 鄰近日期的 runtime identity 或現在的環境只能當背景，不能回填成失敗當時測量。

Provenance 狀態為 **partial recovery**。Attribution harness 可針對上述缺口建置及用
獨立正／負控制驗證；任何 future trace 必須保留自己的版本、命令、載入模組與 artifact
hashes，不能宣稱重建了原 run。在 provenance 與 harness coverage 未完成前，
failure-rate 與新的 capture-semantics 修改維持不執行。#340 不因此關閉。

## Bounded production-path trace(production-01)

2026-09-06 19:28–19:29(local)執行了一次 bounded production-path topology trace:
HEAD `0d757e17`(production source 未變更;harness 未 commit,以 `harness_sha256`
逐檔 hash 記錄),preset `mamba_whole_graph_m`、MOT17-02-SDP、GPU decode、
double-buffer、64 frames、單一 process、無 incidence loop、無 failure-rate 量測。
Plan 存於本機 `production-01-plan.json`。Observer 為 `production-observer/observer.so`
(sha256 `6485ada3…`,與該次 manifest 一致),先通過固定六種控制的 qualification
(`production-observer-qualification/qualification.json`,全部 passed)後才執行。
Workload 正常完成(316.46 FPS;total 2.10 s),`target_returned`。

觀察(25,262 CUDA rows;runtime／driver 兩 domain 的同一操作並存):

- 4 個 python capture site 全數歸類:`detector.whole`(mode global)、
  `nms.main_nocopyback`(mode thread-local)、`gmc.direct`(mode thread-local)、
  `tracker.update`(mode global);同一 context、同一 non-blocking stream(flags=1,
  source=create)、main thread;無 unclassified capture。
- ~~Capture 期間無任何其他 stream 的 WaitEvent／EventRecord／Synchronize／
  IsCapturing 進入;667 個 record→wait event edges 全部落在 capture 之外。
  此 snapshot 未觀察到跨 stream 加入 capture 或 blocking participant。~~
  **此條於 2026-09-07 撤回;retained rows 反證之,見下方「更正」。實際為:667 條
  edges 中 27 條落在 `detector.whole` capture window 內,四個 blocking stream
  (flags=0)參與。**
- `cudaStreamIsCapturing`／`cuStreamIsCapturing` 11,684 次(每 domain 5,842)全數
  rc 0;8 次回傳 Active(＝4 captures × runtime＋driver),其餘 None;decode worker
  thread 的 query 亦全為 None。
- 無任何 900–907 capture error(無 906)。其餘非零 rc 僅啟動期
  `cuCtxGetDevice` 201(device uninitialized)×6 與 `cudaEventQuery` 600
  (not ready)×54,皆非 capture error。

Structure check 未通過(exit 1),依 harness 定義屬 evidence gap,不是負面證據:

- `observer_shutdown_incomplete_or_workers_alive`:停止時 decode worker、
  InductorSubproc、Thread-2 仍存活。
- `missing_final_manifest`:最終 manifest(artifacts／source-drift／
  mapped-library／cupti-stop 欄位)未寫入 — process 在 emit `harness_stopped`
  (cupti_rc 0)之後、mapped-library hashing 尾段結束。`cuda.jsonl` 在 process
  結束前完整,但收尾欄位缺漏使 tamper-evidence 無法核對。

結論:此 trace 是單一 bounded snapshot,不是原始 failure 的重建、不是 failure-rate
或 throughput 證據、未關閉 root cause。~~它顯示目前 production 的 whole-graph
capture 為單 stream 自足、capture 內無跨 stream join;~~ **「單 stream 自足／
capture 內無跨 stream join」同於 2026-09-07 撤回(見下方「更正」);** blocking
stream 經 event join 參與 capture 已在 production topology 中被觀察到,但那四條
stream 的 owning component、以及它與真正 900/901/906 failure 的關係,仍未辨識
(見下方「更正」)。Provenance 與 harness coverage 未完成前,failure-rate 與
capture-semantics 修改維持不執行;#340 不因此關閉。
分析輸出為本機證據:同根目錄下 `production-01-analysis/analysis.json`。

## 更正:production-01 的 join／participant negative claim 撤回(2026-09-07)

重新檢視 production-01 **自己保留的** `cuda.jsonl` 後,上一節原本兩處
negative claim 被其自身 rows 反證,於此撤回。

**撤回的敘述**

1. 「Capture 期間無任何其他 stream 的 WaitEvent／EventRecord 進入;667 個
   record→wait event edges 全部落在 capture 之外。」
2. 「此 snapshot 未觀察到跨 stream 加入 capture 或 blocking participant。」
3. 「production 的 whole-graph capture 為單 stream 自足、capture 內無跨 stream
   join。」

**retained rows 實際顯示的內容**

667 條 record→wait event edges 中,**27 條落在 `detector.whole` capture window
內**(640 條在外);該 window 為 driver domain `ns=[31457106091053,
31457108876388]`、`seq=[15675,15859]`、origin stream `0x4de0c810`(flags=1)、
mode global。27 條之中:

| 面向 | 數量 |
|:--|--:|
| wait 由 blocking stream(flags=0)發出 | 12 |
| wait 的 event 由 blocking stream record | 17 |
| 至少一端為 blocking stream | 27 |

參與的 blocking stream 有 **4 個**:`0x4df68dc0`、`0x4e184bb0`、`0x4e184f50`、
`0x4e184f70`,全部 `flags=0, source=create`,皆由 capturing thread(tid 415149)
發出。連同 `cuEventRecord`,window 內共 28 個 row 來自這四個 blocking stream。
最早一筆距 window 起點 1.02 ms、最後一筆距終點 1.08 ms(window 寬 2.79 ms),
不是邊界四捨五入的產物。其餘三個 capture window(`nms.main_nocopyback`、
`gmc.direct`、`tracker.update`)window 內無 event edge。

**原分析錯在哪裡**

原結論是從 status query 推出來的:trace 中只有 origin stream 曾被觀察到
`status=Active`,於是被讀成「沒有其他 stream 參與」。但 harness 只在
application 自己呼叫 status query 時才記錄 status,而那四個 blocking stream
的 status **從未被查詢過**。沒有 Active 觀察屬於 evidence gap,不是 negative
evidence — 這正是本 repo `README.md`「Missing metadata … are evidence gaps,
never negative evidence」所禁止的推論。同一節下方的控制早已指出「所有
BeginCapture stream 都 non-blocking」不足以排除 blocking participant;錯誤在於
上一節仍把 origin set 當成 participant set。

**不撤回的部分**

「所有觀察到的 BeginCapture origin 都是 non-blocking」**維持成立**:retained
rows 中唯一的 BeginCapture origin 是 `0x4de0c810`,`flags=1`。撤回的是從
origin 推導到 participant set 的推論,不是這句 origin 敘述本身。
「無任何 900–907 capture error」亦維持成立。

**此更正不改變 production-01 的 disposition**

production-01 **仍是 structure-invalid**,原因未變且不因本次重新分析而改變:
停止時仍有 live worker(decode worker、InductorSubproc、Thread-2)、最終
manifest 未寫入、tamper／provenance closure 不完整。#352 對 harness teardown
的修正只讓**日後**的 trace 能避免同類 structure failure,**不追溯性地**
qualify 這個舊 trace。

因此 retained rows 的效力僅限於**撤回原本錯誤的 negative claim**。它們
**不**被升格為 clean positive attribution evidence:不得據此宣稱
production-01 證明了 Rule B、定位了 #340 root cause、等同於 production-02、
提供了可引用的 clean production attribution、驗證了 #344,或重建了原始 failure。
production-01 的四個 blocking stream 是 structure-invalid trace 中的觀察,不是
可引用的 production attribution 結果。

**與 production-02 的關係**

觸發本次重新檢視的是後續那次 structure-clean 的 production-02 trace。
在 production-02 中後來觀察到**相同性質的 join pattern**(`detector.whole`
window 內有 blocking stream 經 event 加入)。這是**另一個 run 的獨立證據**,
與 production-01 各自成立;production-02 的 clean status **不**替 production-01
背書,production-01 也不因為與它相似而被 qualify。

因此 topology 層面的敘述要更新:**blocking stream 經 event dependency 參與
capture,已經在 production topology 中被直接觀察到**,不再只是合成控制裡的
機制。原先「blocking-join 機制尚未在 production site 定位」這種寫法已經太強,
一併修正。仍然未辨識的是:

- 那四條 blocking stream 的 **owning component**;
- decode 對 legacy stream 的 status query 與該 blocking-participant window 之間
  的 **failure-time overlap**;
- 與實際 900/901/906 的 **co-occurrence**(兩次 production trace 都是 0 capture
  error);
- 因而 **#340 的 causal mechanism / root cause** 仍未成立。

換句話說:觀察到的是 topology,不是 causality。

**證據與可重現性**

| 項目 | 值 |
|:--|:--|
| retained trace | `capture-attribution-20260906/production-01/` |
| `cuda.jsonl` sha256 | `8f775afe516a679a538667486611bf0bc4ccd58ec5c55235fb0fdf28e02afbb3` |
| `python.jsonl` sha256 | `447188e3ed1a8d9de32dae613a6e844ea7a7af247887933901b29c85c895a9c1` |
| observer sha256 | `6485ada3d6c1f58b2789262438e619b0fe11a1213bef77c1387164564f9e5744` |
| trace 當時 HEAD | `0d757e1759c2129f1166d36b58d6ba2d3f865357` |
| 重新分析時 repo HEAD | `efd3176f81d7a4e3c13a8c91e9369732dfd85158`(#352 merge) |
| 重新分析輸出 | `capture-attribution-20260906/production-01-reanalysis/` |
| 原分析輸出(未變更) | `capture-attribution-20260906/production-01-analysis/analysis.json` |
| production-02 trace(topology 敘述的依據) | `capture-attribution-340r2-20260906/production-02/` |
| production-02 `cuda.jsonl` sha256 | `9273602ef05ba7e401640010d240dc208334ec76140b2c274ebf56db04ff6e42` |
| production-02 observer sha256 | `a2fb4262c6b3890bbd1255888f5fd864c3e0054d7febe98543aa60862937090e` |
| production-02 當時 HEAD | `bab4e154fd33198b465199570c2558bb760dd7af`(#351 merge) |
| production-02 structure | `trace_structure_ok=true`、`problems=[]`、0 capture error |

重新分析只讀 retained rows,未重跑 workload、未改 analyzer／observer／harness。
capture window 的 begin/end 配對與 `analyze.py` 相同;window 內外的判定與
blocking／non-blocking 的判別另行寫在
`production-01-reanalysis/reanalyze01.py`(sha256
`d31bbd5f8aeb2b08a18a21af445c978cd049a414011ec4d48380008267bf7677`),輸出見同目錄
`reanalysis.txt`、`tally.txt`、`in_window_rows.txt`。以上為本機證據。

**尚未更正的重複處**

`src/saccade/perception/eval/cuda_capture.py` 的 module docstring 仍複述
「no in-capture joins and no blocking participant」。該檔為
`h2_path_partition` 的 `decision_relevant`,連 docstring 變更都會改動
decision-surface content digest 並要求 runtime-identity 重新 attestation,
超出本次 documentation-only 更正的範圍。該句依本節同樣**視為已撤回**,
待後續帶 runtime-identity 的變更一併修正。


## Attribution harness 與固定控制

[Harness 使用說明](../../../scripts/tools/capture_attribution/README.md) 說明 CUPTI
runtime／driver callbacks、Python begin/end site 對照、flags 來源、status queries、
event edges、錯誤與 provenance manifest。未改動 production `src/`。

2026-09-06 的獨立控制觀察到：non-blocking origin 開始 capture 後，blocking side
stream 透過 event wait 加入，該 side 的 `cudaStreamIsCapturing` 為 active；另一個
thread 對 legacy 的同一 query 回傳 906，而 origin 的 capture end 成功。
因此「所有 BeginCapture stream 都 non-blocking」不足以排除 blocking participant。
這是合成機制控制，其本身未定位 production site，也未重建原始 failure；同型的
join 後來在 production topology 中被直接觀察到（見上方「更正」），被觀察到的是
topology，不是 owning component，也不是 causality。
跨 stream 加入機制見 [NVIDIA CUDA Graphs 文件](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

本機 durable artifacts 根目錄：
`/home/ray/.local/state/saccade/perf/capture-attribution-20260906/`。
`recovery-final/recovered.json` 保留抽出的工具證據；`build-qualified/build.json`
保留 observer build 與 header hashes；`qualification-qualified/` 保留固定六種
控制的每個 process trace／manifest 與總表。production-01 一節的產出另存於
`production-observer/`、`production-observer-qualification/`、`production-01/`、
`production-01-eval/` 與 `production-01-analysis/analysis.json`。以上都是本機
證據，不是已發布的 research 結果。固定控制的通過或 production snapshot 的
觀察本身，都不表示 production capture-site attribution 已完成。
