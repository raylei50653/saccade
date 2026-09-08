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
Validity gate           見 §4:一個 run 只有在 sequence execution 已開始後才進
                        denominator;infrastructure-only failure 標 invalid 並補跑。
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

**Coordinate binding.** 兩條 path 都在同一個 merged-main coordinate 上跑:

| | 值 |
|---|---|
| main SHA | `b649de68e36ad530ed883f579478ab656a238158` |
| coordinate.implementation | `26c49eb3629ed4b703641277b9bcd177a373b3b248570cd2cb5056b346addfa7` |
| coordinate.decision_surface | `9b7faeb0f76a43483a924ac4028361bb155373027e9fe44e4cffbd5f1a2b0369` |
| coordinate.environment | `4ed7abb1c0539ba693484567a8a218366b4d0292f7a9fa3f7ebf5ce338fd05be` |
| coordinate.identity_semantics | `8fc9bd85dd651791ec552dd2d6ab0244e849d213dcee7a2b77dfb9246a7969ca` |
| coordinate.runtime_inputs | `0b839df0b89141959a4ae4762c727d446a2292016832ab468ce48334fff1a3d5` |
| probe.digest | `2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5` |
| build dir | `build/h2_layer_p` |

任一 coordinate 軸在 campaign 中途移動 ⇒ 已完成的 run 全部作廢,重新開始;
不得把兩個 coordinate 的 run 併進同一個分母。

**Path A(production GPU decode).** 這是 #340 原始 failure 的組態:

```text
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m \
  --detector SDP --double-buffer --output <run_dir>
```

**Path B(`--no-gpu-decode` / DALI 不參與 GPU decode)。**

```text
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m \
  --detector SDP --double-buffer --no-gpu-decode --output <run_dir>
```

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
engine build 失敗、driver 未載入、磁碟滿、被外部訊號中止。這些走 §4 的
invalid 規則,不進 numerator,也不進 denominator。

## 4. 分母規則

- **workload 開始前的純 infrastructure failure**(process 未進入 sequence execution:
  engine build、model load、dataset 掃描、輸出目錄建立階段的失敗)⇒ 標 `invalid`,
  不進 denominator,**補跑一個 run 遞補**。
- **sequence execution 一旦開始**(第一個 sequence 的第一個 frame 已進 pipeline)
  ⇒ 該 run **永久進入 denominator**。之後不得因為結果不好看、因為懷疑是環境雜訊、
  或因為任何事後判斷把它移出分母。這條沒有例外。
- 每個 `invalid` 都要記錄原因與判定時點(§10),closure report 必須列出
  invalid 總數。invalid 數量本身不是 verdict 輸入,但隱藏它是 report 缺陷。

## 5. 停止條件

- **Sufficiency(該 path pass).** 該 path 累積 100 個有效 run 且 failure count == 0。
- **Early terminate(該 path fail).** 出現**第一個**符合 §3 的 failure 即宣告該
  path closure fail,**立刻停止該 path**。不繼續跑到 100 去稀釋它。
  另一條 path 是否跑完由 owner 決定,但已完成的部分不得改寫成 rate。
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
| `EXECUTION_INVALID` | harness 自身失效(log 未寫、schema 不合、campaign 記錄不完整) | 無 transition,fail-closed。已產生的 run 不得部分採信 |

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
不重跑、不重新比較 F3c candidate。不做 kernel-level attribution。不改任何
production 程式碼:本研究期間 `src/saccade/**` 與 `scripts/eval/mot17.py` 凍結
(改動會移動 coordinate,見 §2)。

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
  "verdict": "ok|failure|invalid",
  "invalid_reason": null,
  "log_sha256": "<sha256 of the captured stdout+stderr>",
  "main_sha": "b649de68e36ad530ed883f579478ab656a238158",
  "coordinate_implementation": "26c49eb3...addfa7"
}
```

`capture_error_hits` 記錄命中的字串與其所在行號;非空 ⇒ `verdict = "failure"`。
`sequence_execution_started = false` 且非 ok ⇒ `verdict = "invalid"`(§4)。

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
| 1 凍結自由度 | §2 組態逐字、§2 交錯順序、§3 字串集合、§5 停止點、§6 bound 公式與措辭、§10 schema |
| 2 機械可判定 | §3 是字串比對述詞;§5/§7 是計數比較。無「明顯較好」類語言 |
| 3 窮盡 terminal | §7 五個 terminal 含 validity failure 與 execution-invalid,各自具名 transition |
| 4 scoped exhaustion | §6 禁止「rate = 0」;§0 明示不宣稱 mechanism;bound 單位限定 per 7-sequence run |
| 5 joint headroom | n/a — 單一決策量(failure count),無多量權衡 |
| 6 blind→reveal | n/a — 本研究無 blind phase |

## 13. 執行順序

1. 本文件 merge(= 凍結點)
2. 寫 harness:交錯排程、§3 述詞、§10 schema、fail-closed(log 寫不出來即中止)
3. 第一個 rate run
4. closure report

**harness 尚未寫。** 在 harness 存在且其述詞與 schema 經對照本文件驗證之前,
不得開始 run 1。
