<!-- doc-status: active -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-08 -->

# #340 capture-race incidence closure — preregistration

本文件在**第一個 rate run 之前**凍結。凍結後只能以 append-only amendment 修訂
(§11),不得 inline 編輯。權威 seal bar 是
[experiment contract §20.8](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md),
declaration 欄位是同一份契約的 §20.2;本文件引用它們,不複述、不分叉。

## 0. 這回答什麼、不回答什麼

#340 目前混著三件事:mechanism、mitigation、incidence。本研究**只做第三件**。

| | 狀態 |
|---|---|
| mechanism(哪一條 blocking stream 開了 capture) | **未解**,本研究不回答。production topology 已顯示四條 blocking stream 透過 event dependency 加入 `detector.whole`,其 owning component 仍未辨識 |
| mitigation | Phase 2A(#343,Rule A 的 `thread_local`)與 Phase 2B(#344,decode producer 離開 legacy stream)已 merge。兩者都是**移除前提**,不是根因解釋 |
| incidence | **本研究**。在 mitigation 已落地的 main 上,量 #340 capture-race failure 的發生率上界 |

本研究不因 0/100 而宣稱 #340 的 root cause 已知。它只回答:**在這個 coordinate、
這兩條 path、這個 workload 上,#340 failure 的發生率上界是多少。**

## 1. §20.2 declarations

```text
Target decision layer   none (cross-layer substrate work)
Study intent            boundary diagnostic
                        (secondary: 提供 #340 closure review 的 incidence 依據)
Design objective        n/a — 非 design evaluation。本研究不挑候選、不調參數、
                        不產生 design candidate
Selection rule          n/a — 無候選集合可排序
Validity gate           見 §4:sequence execution 開始**前**的 infrastructure
                        failure 標 invalid 並補跑;開始**後**該 run 永久進
                        denominator,且若因非 §3 原因無法跑完 7 條 sequence,
                        整個 campaign 立即 EXECUTION_INVALID。
                        任一 path 的有效 run 數未達 100 即為 validity failure
Stop condition          見 §5:sufficiency = 該 path 100 個有效 run 且 0 failure;
                        early-terminate = 第一個符合 §3 定義的 failure
Output class            diagnostic result(§20.4)。不得升格為 design candidate,
                        也不得當作 performance upper-bound candidate
Mainline transition     見 §7 terminal 表。CLOSED_BOUNDED 關閉「#340 發生率未知」
                        這個 core unknown;FAILURE_OBSERVED 不關閉任何東西,
                        直接轉入該次 failure 的 attribution
Type κ                  quantification space: 一個 path 上的完整 7-sequence run
                                             (trial unit = run,不是 sequence、
                                              不是 frame、不是 capture)
                        comparison relation: 該 run 是否命中 §3 的 failure 述詞
                                             (布林,無 tolerance)
                        decision rule: 該 path 的 failure count == 0 ⇒ 該 path pass;
                                       >= 1 ⇒ 該 path fail(§5 立即終止)
```

## 2. 凍結的執行組態

**量測對象是一個固定的 source SHA,不是 campaign 當下的 `main` HEAD。**
本文件自己 merge 就會讓 `main` 前進,harness 的 commit 也會。那些前進**不得**
進入被量測的東西,而這件事不能留給 harness 自行解釋。因此 campaign 分成兩個
worktree:

- **target worktree** — clean、detached、釘死在下表的 target source SHA。
  campaign 執行的每一個 `mot17.py` 都從這個 checkout 啟動。
- **control worktree** — 本文件、harness、campaign 記錄住在這裡,可以自由前進。
  controller 只能 spawn target checkout 裡的 `scripts/eval/mot17.py`,不得執行
  control worktree 裡的任何 production 程式碼。

**⚠️ worktree 分離不等於 import 分離。** `.venv` 是 editable install:
`__editable__.saccade-0.1.0.pth` 指向 `/home/ray/developer/ai/saccade/src`,
`saccade_build.pth` 指向 `/home/ray/developer/ai/saccade/build`。用共用 `.venv`
去跑另一個目錄的 `mot17.py`,`import saccade` 仍然解析到**那個 `.pth` 指的目錄**,
pin 就只是裝飾。所以本研究要求的是 import 層的不變量,不是目錄層的:

> **target worktree 必須就是 `.venv` 的 editable install 解析到的那個目錄。**

滿足這條最省的作法是把既有的主目錄 `/home/ray/developer/ai/saccade` 本身當
target worktree(detached 到 target source SHA;`build/` 與 `.venv/` 都被
gitignore,checkout 不會動到,attested 的 `build/h2_layer_p` 原地可用),
control worktree 另開一個。反過來擺(target 另開目錄)則必須為它建**獨立的
venv 與 build**,並先證明 environment 軸 digest 未因此移動 —— 未證明前不得開跑。

| | 值 |
|---|---|
| target source SHA | `b649de68e36ad530ed883f579478ab656a238158` |
| coordinate.implementation | `26c49eb3629ed4b703641277b9bcd177a373b3b248570cd2cb5056b346addfa7` |
| coordinate.decision_surface | `9b7faeb0f76a43483a924ac4028361bb155373027e9fe44e4cffbd5f1a2b0369` |
| coordinate.environment | `4ed7abb1c0539ba693484567a8a218366b4d0292f7a9fa3f7ebf5ce338fd05be` |
| coordinate.identity_semantics | `8fc9bd85dd651791ec552dd2d6ab0244e849d213dcee7a2b77dfb9246a7969ca` |
| coordinate.runtime_inputs | `0b839df0b89141959a4ae4762c727d446a2292016832ab468ce48334fff1a3d5` |
| probe.digest | `2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5` |
| build dir | `build/h2_layer_p`(target worktree 內) |

**每個 run 啟動前,harness 必須機械驗證下列三條,任一不符即 fail-closed 中止
campaign(§7 `EXECUTION_INVALID`),不得只記 warning、不得續跑:**

1. target worktree 的 `git rev-parse HEAD` == target source SHA
2. target worktree 的 `git status --porcelain` 為空
3. 在**即將用來跑 workload 的那個 interpreter** 裡解析 `saccade.__file__`,
   其路徑必須落在 target worktree 之下(第 3 條才是真正擋住 §2 那個 import 陷阱
   的檢查;前兩條擋不住它)

第 3 條的觀測值逐 run 記錄(§10 `saccade_import_root`),不是只做斷言。

任一 coordinate 軸在 campaign 中途移動 ⇒ 已完成的 run 全部作廢,重新開始;
不得把兩個 coordinate 的 run 併進同一個分母。把 target 釘死正是為了讓這件事
不會**因為 repository 自己前進**而發生。

**Path A(production GPU decode).** 這是 #340 原始 failure 的組態:

```text
<target>/.venv/bin/python <target>/scripts/eval/mot17.py \
  --preset mamba_whole_graph_m \
  --detector SDP --double-buffer --output <run_dir>
```

**Path B(`--no-gpu-decode` / DALI 不參與 GPU decode)。**

```text
<target>/.venv/bin/python <target>/scripts/eval/mot17.py \
  --preset mamba_whole_graph_m \
  --detector SDP --double-buffer --no-gpu-decode --output <run_dir>
```

`<target>` = target worktree 根目錄,cwd 亦設為它。
兩條 path 除 `--no-gpu-decode` 外逐字相同。`--detector SDP` 且不給
`--sequences` ⇒ 展開為 train split 全部 7 條 SDP sequence
(`MOT17-02/04/05/09/10/11/13-SDP`,合計 5,316 frames)。**一個 run = 這 7 條全跑完。**

**Interleaving.** 交錯執行 `A,B,A,B,…`,共 100 組。不得先跑完 100 個 A 再跑 B:
兩條 path 必須共享同一段時間內的 thermal、driver、host 負載條件。
交錯順序本身凍結為 A 先。

**N.** 每 path 100 個有效 run。**看到中途結果後不得加 N。**

## 3. Primary failure 定義(事前列舉)

一個 run 命中 **#340 capture-race failure**,若且唯若該 run 的 stdout/stderr 或
`runs/` log 出現下列任一 CUDA capture 錯誤,或其已知 wrapper 形式:

| CUDA code | 名稱 | 已知 wrapper 形式 |
|---|---|---|
| 900 | `cudaErrorStreamCaptureUnsupported` | `operation not permitted when stream is capturing`;torch `RuntimeError: CUDA error: operation not permitted when stream is capturing` |
| 901 | `cudaErrorStreamCaptureInvalidated` | `operation failed due to a previous error during capture`;torch 同名 `RuntimeError` |
| 906 | `cudaErrorStreamCaptureImplicit` | `legacy stream depend on a capturing blocking stream`;torch `RuntimeError: CUDA error: operation would make the legacy stream depend on a capturing blocking stream`;`currentStreamCaptureStatusMayInitCtx` 的 `c10` traceback |

述詞是**機械可判定**的:對上述字串集合做比對,命中即為 failure。判定不看
exit code — provenance 恢復已顯示原始事件是否以 nonzero exit 結束**未恢復**,
所以 exit code 不是可靠的偵測面。exit code 仍逐 run 記錄(§10),但不進述詞。

**不算 #340 failure 的東西**(即使 run 失敗):OOM、dataset/IO 錯誤、TensorRT
engine build 失敗、driver 未載入、磁碟滿、被外部訊號中止。這些一律不進
numerator。它們**進不進 denominator 取決於發生時點**,規則在 §4 —— 關鍵是
sequence execution 開始後出現這類 failure,並**不是**一個可以悄悄補跑的
per-run invalid。

## 4. 分母規則

分界點是 **sequence execution 是否已開始**(= 第一個 sequence 的第一個 frame
已進 pipeline)。這個時點逐 run 記錄(§10 `sequence_execution_started`)。

- **開始前的純 infrastructure failure**(engine build、model load、dataset 掃描、
  輸出目錄建立階段的失敗)⇒ 標 `invalid`,不進 denominator,**補跑一個 run 遞補**。
  這是**唯一**可補跑的 invalid 類別。
- **開始後,run 跑完完整 7 條 sequence** ⇒ 進 denominator。命中 §3 述詞為
  `failure`,否則為 `ok`。
- **開始後,run 因非 §3 原因無法跑完 7 條 sequence**(mid-run OOM、IO 錯誤、
  被外部訊號砍掉、host 重開機……)⇒ **整個 campaign 立即
  `EXECUTION_INVALID`(§7),停止,不補跑、不續跑。** 該 run **不得**被當成
  failure-free observation 計入分母,**也不得**被單獨移出分母。

第三條是刻意設計成沒有中間選項的。若允許「這次 OOM 就跳過、補一個」,
研究就多出一個事後可用的自由度:任何看起來不順的 run 都可以被重新描述成
環境雜訊而消失。把它升級成整個 campaign 作廢,代價是重跑,換到的是
「execution 一旦開始就不能事後剔除」這條性質**沒有例外**。這也讓 harness
作者不必替研究規格臨場做統計裁決 —— 遇到就 abort,不需要判斷。

(這條的實務含義:campaign 期間該台機器不要做別的重活。mid-run OOM 多半是
自找的,不是不可抗力。)

**記錄義務.** 每個 `invalid` 都要記錄原因與判定時點(§10),closure report
必須列出 invalid 總數。invalid 數量本身不是 verdict 輸入,但隱藏它是 report 缺陷。

## 5. 停止條件

- **Sufficiency(該 path pass).** 該 path 累積 100 個有效 run 且 failure count == 0。
- **Early terminate(該 path fail).** 出現**第一個**符合 §3 的 failure 即宣告該
  path closure fail,**立刻停止該 path**。不繼續跑到 100 去稀釋它。
  另一條 path 是否跑完由 owner 決定,但已完成的部分不得改寫成 rate。
- **Execution abort(整個 campaign).** §4 第三條(execution 開始後因非 §3 原因
  未跑完 7 條 sequence)或 §2 的三條 pre-run 驗證任一不符 ⇒ 立刻停止兩條 path,
  terminal = `EXECUTION_INVALID`。
- **Validity failure.** 因外部原因(硬體更換、driver 升級、coordinate 移動)
  無法補到 100 個有效 run ⇒ `UNRESOLVED / INVALID-STUDY`(§20.7)。

## 6. 統計與措辭

0/100 的 one-sided 95% exact binomial upper bound:

```text
1 - 0.05^(1/100) = 0.029513  →  2.95%
```

(等價於 Clopper–Pearson 的 `Beta.ppf(0.95, 1, 100) = 0.0295130`。)

**允許的措辭,逐字:**

> Path A: 0/100 observed; one-sided 95% upper bound ≈ 2.95% per 7-sequence run.

**禁止的措辭**(任一出現即為 report 缺陷):

- 「true failure rate = 0」、「發生率為零」、「不再發生」
- 「#340 已修復」、「capture race 已消除」
- 把兩條 path 的 200 個 run 併成一個分母去算更緊的 bound
  (兩條 path 是兩個獨立宣稱,各自 100,各自 2.95%)
- 把 upper bound 說成 point estimate
- 從 0/100 反推 mechanism 或宣稱 blocking capturing stream 已辨識

bound 的單位是 **per 7-sequence run**,不是 per frame、不是 per sequence、
不是 per capture。任何換算都要重新宣告分母。

## 7. Terminal 對照(§20.8 item 3:窮盡且各自具名 transition)

| Terminal | 條件 | Mainline transition |
|---|---|---|
| `CLOSED_BOUNDED` | A 與 B 皆 100 有效 run、皆 0 failure | **關閉 core unknown**「#340 在 mitigation 後的發生率未知」。#340 進 closure review。**不**關閉 mechanism |
| `FAILURE_OBSERVED_A` | Path A 出現第一個 §3 failure | 無 transition。#340 維持 open,直接轉入該次 failure 的 attribution(保留完整 log、stream flags、capture state dump)。**不得**改門檻、改定義、改組態後重跑 |
| `FAILURE_OBSERVED_B` | Path B 出現第一個 §3 failure | 同上。額外意義:failure 不需要 GPU decode ⇒ Phase 2B 的前提移除不足以涵蓋 |
| `UNRESOLVED_INVALID_STUDY` | 無法補足 100 有效 run,或 coordinate 中途移動 | 無 transition。不得報成 incidence 結果,也不得報成 signal-family exhaustion |
| `EXECUTION_INVALID` | (a) harness 自身失效(log 未寫、schema 不合、campaign 記錄不完整);(b) §2 三條 pre-run 驗證任一不符;(c) §4 第三條:sequence execution 開始後因非 §3 原因未跑完 7 條 sequence | 無 transition,fail-closed。已產生的 run 不得部分採信,**不得**從中抽出「至少這幾次沒 fail」當弱結論 |

沒有第六種結果。沒有「再多描述一點然後繼續」這個出口。

## 8. Observer 禁令

**primary rate measurement 不得開 CUPTI、不得掛 LD_PRELOAD observer、不得開
`scripts/tools/capture_attribution/` 的 harness。** 理由:observer 已知可能改變
race timing,用它量出來的 incidence 不是 production 的 incidence。

保留的:failure 發生當下既有的 production diagnostics
(`describe_capture_state`,`SACCADE_CAPTURE_DEBUG` 的 failure-time dump 為無條件)。
這些只在 failure path 觸發,不在正常 run 的熱路徑上。

一旦命中 `FAILURE_OBSERVED_*`,attribution 階段當然可以開 observer —— 但那是
另一個研究,不能回填成本研究的 rate。

## 9. Scope 之外(明列,避免事後擴張)

不量 FPS / throughput。不比較 IDF1 / HOTA / MOTA。**MOT identity 不進 verdict** ——
run 之間 MOT bytes 相不相同是 #363 的軸,與本研究無關(#363 已 closed)。
不重跑、不重新比較 F3c candidate。不做 kernel-level attribution。

production 程式碼的凍結範圍限定在 **target worktree**:它 detached、clean、
釘在 target source SHA,由 §2 的三條 pre-run 檢查機械保證。control worktree
(harness、本文件、campaign 記錄)可以在 campaign 期間繼續 commit —— 只要它
不是 execution source。這比「campaign 期間全 repo 不准動」更嚴格,因為它是
可檢查的,不是靠自律。

## 10. 凍結的 per-run 記錄 schema

每個 run 一筆 JSON,append 進 campaign 的 `runs.jsonl`。欄位凍結如下:

```json
{
  "schema": "capture_race_incidence_run_v1",
  "campaign_id": "<UTC timestamp of campaign start>",
  "seq_index": 0,
  "path": "A|B",
  "run_dir": "<absolute path>",
  "started_utc": "<ISO8601>",
  "finished_utc": "<ISO8601>",
  "argv": ["..."],
  "exit_code": 0,
  "sequence_execution_started": true,
  "sequences_completed": ["MOT17-02-SDP", "..."],
  "capture_error_hits": [],
  "verdict": "ok|failure|invalid|execution_invalid",
  "invalid_reason": null,
  "log_sha256": "<sha256 of the captured stdout+stderr>",
  "target_source_sha": "b649de68e36ad530ed883f579478ab656a238158",
  "target_head_observed": "<git rev-parse HEAD of the target worktree>",
  "target_worktree_clean": true,
  "saccade_import_root": "<dirname of saccade.__file__ in the spawned interpreter>",
  "coordinate_implementation": "26c49eb3...addfa7"
}
```

欄位名是 `target_source_sha`,**不是** `main_sha` —— 它記的是被量測的 checkout,
不是 campaign 當下 repository 的 `main` HEAD,那兩者在 campaign 期間必然不同。
`target_head_observed` / `target_worktree_clean` / `saccade_import_root` 是 §2
三條 pre-run 檢查的**觀測值**,逐 run 落盤,讓 pin 是被記錄的事實而不是斷言。

`capture_error_hits` 記錄命中的字串與其所在行號;非空 ⇒ `verdict = "failure"`。
`sequence_execution_started = false` 且非 ok ⇒ `verdict = "invalid"`(§4 第一條)。
`sequence_execution_started = true` 且未跑完 7 條 sequence 且 `capture_error_hits`
為空 ⇒ `verdict = "execution_invalid"`,campaign 停止(§4 第三條)。

原始 stdout+stderr 逐 run 落到非 scratch 的 timestamped artifact 目錄
(`~/.local/state/saccade/perf/capture-race-incidence-<campaign_id>/`),
連同 `runs.jsonl` 與一份 manifest。**raw log 不得只留在 scratchpad。**

## 11. Amendment 規則

凍結後發現任何影響 terminal 的自由度沒被釘住 ⇒ 那是 declaration defect,
以 **append-only amendment**(本文件末尾新增 `## A1`、`## A2` …,註明日期與
當時已完成的 run 數)修補,絕不 inline 改寫上文。amendment 若改變 failure 定義、
N、或 path 組態,已完成的 run 全部作廢。

## 12. §20.8 自檢

| 條 | 本文件 |
|---|---|
| 1 凍結自由度 | §2 target source SHA 與三條 pre-run 檢查、§2 組態逐字、§2 交錯順序、§3 字串集合、§4 分母三分、§5 停止點、§6 bound 公式與措辭、§10 schema |
| 2 機械可判定 | §3 是字串比對述詞;§5/§7 是計數比較。無「明顯較好」類語言 |
| 3 窮盡 terminal | §7 五個 terminal 含 validity failure 與 execution-invalid,各自具名 transition。§4 的三種時點各自對應唯一 terminal,無未定義情形 |
| 4 scoped exhaustion | §6 禁止「rate = 0」;§0 明示不宣稱 mechanism;bound 單位限定 per 7-sequence run |
| 5 joint headroom | n/a — 單一決策量(failure count),無多量權衡 |
| 6 blind→reveal | n/a — 本研究無 blind phase |

## 13. 執行順序

1. 本文件 merge(= 凍結點)
2. 備妥 target worktree:detached 於 target source SHA、clean、且滿足 §2 的
   import 不變量(`saccade.__file__` 落在其下)。control worktree 另立
3. 寫 harness:交錯排程、§2 三條 pre-run 檢查、§3 述詞、§10 schema、
   fail-closed(log 寫不出來即中止)
4. harness ↔ 本文件的 spec equivalence check:逐條對照述詞字串集合、分母三分、
   schema 欄位、terminal 對照
5. 第一個 rate run
6. closure report

**harness 尚未寫。** 在 harness 存在且其述詞與 schema 經對照本文件驗證之前,
不得開始 run 1。

## A1. `sequence_execution_started` 的可觀測面(2026-09-08,已完成 run 數 = 0)

**Defect.** §4 把分母的分界點定義成「第一個 sequence 的第一個 frame 已進
pipeline」,但**沒有釘住這件事要用什麼觀測。** 寫 harness 時實查 frozen target,
發現能用的最細觀測是 `evaluator.py` 的進度行

```text
🎬 <seq> [<frame_id>/<frame_end>]
```

它無 verbose flag 保護(三條 frame loop 都有,格式相同),但只在
`frame_id % 100 == 0` 觸發。MOT17 frame 由 1 起算,所以**第一條進度行在 frame 100**,
7 條 sequence 各 ≥525 frames ⇒ 完成側觀測充分,但**第一條 sequence 的 frame 1–99
沒有任何觀測**。

這個盲區正好落在 §4 的分界點上。天真讀法會把「frame 40 因 OOM 崩掉」判成
「execution 尚未開始」⇒ 標 invalid、悄悄補跑 —— 正是 §4 存在要堵的那個洞。

**規則(fail-closed,取代天真讀法).** `sequence_execution_started` 的**預設值是
`true`**。只有在下列兩條**同時**成立時才能記為 `false`:

1. 該 run 的 log 中無任何 `🎬 ` 進度行,且 run_dir 中無任何 `<seq>.txt`;**且**
2. 該 run 的 log 正面命中 **表 A1-1** 列舉的 setup-phase failure signature 之一。

兩條缺一 ⇒ `sequence_execution_started = true`,走 §4 第二/三條。因此**一個沒有
進度行、也不符任何已列舉 setup signature 的崩潰,會讓整個 campaign
`EXECUTION_INVALID`,而不是被補跑掉。** 不確定性只往「作廢重跑」倒,不往
「丟掉這個 run」倒。

**表 A1-1:setup-phase failure signatures(事前列舉,大小寫不敏感子字串)**

每一列是 **(context term) ∧ (failure term)** 的合取:兩側各取一個子字串,
**都**出現在同一份 log 才算命中。單邊詞不構成 signature —— 健康的 run 也會印
`checkpoint`、`TensorRT` 這類字,單邊比對會讓正常 log 命中 setup signature。
比對大小寫不敏感。

| 類別 | context term(任一) | ∧ failure term(任一) |
|---|---|---|
| `dataset` | `seqinfo.ini`、`data_root`、`MOT17-` | `No such file`、`FileNotFoundError`、`not found`、`does not exist` |
| `weights` | `checkpoint`、`state_dict`、`.ckpt`、`.pth` | `No such file`、`FileNotFoundError`、`not found` |
| `tensorrt` | `TensorRT`、`trtexec`、`engine build`、`.engine` | `failed`、`Error`、`Exception` |
| `cuda_device` | (無 —— 下列自成 signature) | `no CUDA-capable device is detected`、`CUDA driver version is insufficient`、`CUDA unknown error`、`Found no NVIDIA driver` |
| `output_dir` | 本 run 的 `--output` 路徑字串 | `Permission denied`、`Read-only file system` |

`cuda_device` 一列的 failure term 已經自帶足夠 context,不需要合取。

**表 A1-1 視為 failure 定義的一部分(§11 適用)。** 在觀察到一個它「新涵蓋」的
failure **之後**才擴充它 ⇒ 已完成的 run 全部作廢。這條是為了讓「事後把某次崩潰
重新描述成 setup 問題」付出與改 failure 定義相同的代價。事前擴充不受限。

**Preflight 失敗的 verdict 優先權.** §2 三條 pre-run 檢查任一不符 ⇒
`verdict = "execution_invalid"`,**不論** `sequence_execution_started` 為何。
preflight 失敗不是可補跑的 §4 第一類 invalid。三個 observed identity 欄位
(`target_head_observed`、`target_worktree_clean`、`saccade_import_root`)
即使在 preflight 失敗時**也必須落盤**,否則 fail-closed 這件事本身沒有證據。

**Log fidelity:`PYTHONUNBUFFERED=1`.** harness 必須在 workload 的環境變數中設定它。
理由:stdout 導向 pipe 時是 block-buffered(8KB),硬崩潰會丟掉尾端 buffer ——
包含 §3 要比對的那行 error 與最後幾條 `🎬`。沒有它,§3 述詞與 §4 分界**都是讀在
被截斷的 log 上**。這不是 §8 意義下的 observer:它只改 I/O buffering,不加
per-frame 工作(進度行每 100 frames 一條),不碰任何 CUDA path。

**§10 schema 的加性欄位.** A1 為 `capture_race_incidence_run_v1` 追加三個
**純觀測**欄位,不改動任何既有欄位的語義:

```json
{
  "progress_marker_seen": true,
  "progress_markers": 53,
  "setup_failure_signature": null
}
```

`progress_marker_seen` = 是否出現過 `🎬 ` 行;`progress_markers` = 行數;
`setup_failure_signature` = 命中的表 A1-1 類別名,未命中為 `null`。
