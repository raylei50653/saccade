<!-- doc-status: closed -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-08 -->

# #340 capture-race incidence closure — report

本文件是
[capture_race_incidence_preregistration_20260908.md](capture_race_incidence_preregistration_20260908.md)
（以下稱「預註冊」）§13 step 6 的產出。它只報告該預註冊所凍結的量與 terminal，
不新增自由度、不重新描述統計理由、不引入預註冊沒有事先列舉的判準。

## 1. Terminal

**`FAILURE_OBSERVED_B`**（預註冊 §7 五個 terminal 之一）。

依 §7：**無 mainline transition。#340 維持 open**，直接轉入該次 failure 的 attribution。
依 §5：Path B 出現第一個符合 §3 的 failure 即宣告該 path closure fail，立刻停止該 path。

依 §7 同一列的規定，以下事情**不得**發生：改門檻、改 failure 定義、改 path 組態後重跑。
下一輪必須是新的 frozen target 與新的預註冊（見 §7 本文件）。

## 2. 計數

| | 值 |
|---|---|
| campaign id | `20260908T061300Z` |
| target source SHA | `b649de68e36ad530ed883f579478ab656a238158` |
| 起訖（UTC） | 2026-09-08T06:13:00Z → 06:59:20Z |
| attempts | 94 |
| verdict `ok` | 93 |
| verdict `failure` | 1 |
| verdict `invalid` | **0** |
| verdict `execution_invalid` | 0 |
| Path A valid runs | 47（0 failure） |
| Path B valid runs | 47（含該 1 次 failure） |

`invalid` 總數為 **0**：預註冊 §4 的記錄義務要求 closure report 列出它，且本次
沒有任何 setup-invalid，因此 §A2.2 的 effective-slot 補跑機制在本 campaign 中
從未被觸發，attempt ordinal 與 slot 一一對應。

**不報告 incidence rate。** 兩條 path 都沒有達到預註冊 §2 凍結的 100 個有效 run，
§6 的 upper-bound 措辭（其成立條件是 0/100）對任一條 path 都不成立。
本文件不給 point estimate、不給 confidence bound、不把兩條 path 的 run 併成同一個分母。

## 3. 命中的那一次 run

| 欄位 | 值 |
|---|---|
| `seq_index`（attempt ordinal） | 93 |
| `slot_index` | 46 |
| `path` | **B**（`--no-gpu-decode`） |
| `verdict` | `failure` |
| `exit_code` | 1 |
| `sequence_execution_started` | `true`（`progress_markers` = 6） |
| `sequences_completed` | `["MOT17-02-SDP"]`（1/7） |
| `capture_error_hits` | 3 筆，全為 CUDA **901** |
| `log_sha256` | `d9736b669af3d2ccafa5ce7b7d801c85d825e512cbe0c4a9fd6f00b55c19c70b` |

三條 pre-run 檢查（§2）在該 run 的記錄中全部相符：
`target_head_observed` = target source SHA、`target_worktree_clean` = `true`、
`saccade_import_root` = `/home/ray/developer/ai/saccade/src`。

分類依預註冊 §3 → §4 的順序成立：capture-error 述詞命中即為 `failure`，
不看 exit code，也不因該 run 未跑完 7 條 sequence 而改判 `execution_invalid`。

失敗點在第 2 條 sequence（`MOT17-04-SDP`）的 capture：

```text
tracker_gpu.py:1889  _capture()
  → torch.cuda.make_graphed_callables
    → torch/cuda/graphs.py:265 __exit__ → capture_end()
      torch.AcceleratorError: CUDA error:
        operation failed due to a previous error during capture
        （cudaErrorStreamCaptureInvalidated）

內層先報：
  tracker_gpu.py:1856 _graph_fn → tracker_cpp.update_into
    RuntimeError: CUDA Error: operation failed due to a previous error
      during capture at src/tracking/tracker_gpu.cu:3553
```

## 4. 本輪可以保留的結論

target `b649de68` **已包含 Phase 2A（#343）與 Phase 2B（#344）**
（`fix/decode-stream-ordering` 是該 SHA 的 ancestor）。在這個 target 上：

> **capture invalidation 可以在 `--no-gpu-decode` 路徑上發生。**
> 因此 GPU decode **不是**這個 failure 的必要前提，
> **Phase 2B 所移除的 decode-side prerequisite is not necessary for the observed
> failure class。**

措辭邊界（本文件與任何引用它的文件都要守住）：

- **不要寫「Phase 2B failed」。** Phase 2B 修掉的那條具體 decode ordering 問題
  仍可能是真的；本輪證據只說它**不是完整機制**，沒有推翻它自己宣稱的內容。
- **不要寫「#340 已修復」「capture race 已消除」「發生率為零」。**（§6 禁止措辭）
- **不得從本輪結果反推 mechanism。** 本輪完全沒有指出是哪一條 stream、哪個
  operation、哪條 runtime edge 造成 invalidation。預註冊 §0 已事先聲明本研究
  不回答 mechanism，§6 明文禁止從結果反推它。

可一併記錄的**結構事實**（來自 Phase 1 audit，不是本輪的量測結論）：命中點是兩個
拿不到 `capture_error_mode`、因此仍為 `global` 模式的 `make_graphed_callables`
site 之一（tracker）。這說明 Rule A 在該處仍然活著是**既有**的已知事實，
不構成對本次 invalidation 成因的歸因。

## 5. Path A 的處置

> Path A completed 47 valid attempts with no observed failure before campaign
> termination. The preregistered 100-attempt bound was not reached; no failure
> rate or 0/100 confidence bound is reported.

依 §5，另一條 path 是否跑完由 owner 決定，且已完成的部分不得改寫成 rate。
**owner 決定：不補跑至 100，本 campaign 就此正式 closure。**

理由（owner 原話要點）：這 47 次作為 `b649de68` 的歷史結果仍然有效，但它只描述
舊的 frozen target，對下一個帶 attribution instrumentation 的 target 沒有直接
外推力；現在真正限制研究進度的是「failure 已經出現，但現場沒有留下 attribution
evidence」，不是 A 的樣本數。若日後部署風險界線仍需要 production Path A 的統計
bound，那一輪再把 A/B 都按完整預註冊數量跑滿。

## 6. 證據

campaign artifact（非 scratch，timestamped）：

```text
~/.local/state/saccade/perf/capture-race-incidence-20260908T061300Z/
  manifest.json      terminal / valid / failures / invalid 計數
  runs.jsonl         94 筆 capture_race_incidence_run_v1
  runs/<NNNN>-<P>/   每 run 的 MOT 輸出
  logs/<NNNN>-<P>.log 每 run 的 raw stdout+stderr（sha256 逐 run 記在 runs.jsonl）
```

controller stdout 另存
`~/.local/state/saccade/perf/capture-race-incidence-controller-20260908T061300Z.log`。

`manifest.json` 的 `preregistration` 欄記的是 campaign 執行當下的路徑
`docs/research/pipeline/capture_race_incidence_preregistration_20260908.md`。
本 PR 依 Doc Structure C6 把該文件移入 `closed/`，那個欄位因此是**移動前**的路徑；
manifest 是已落盤的執行紀錄，不回填。

**⚠️ attribution 證據缺口。** §7 要求命中 `FAILURE_OBSERVED_*` 時保留「完整 log、
stream flags、capture state dump」。實際保留下來的**只有 log**：失敗 run 的 log
共 73 行，其中**沒有 `describe_capture_state()` dump，也沒有 stream flags**。

原因是 §8 所稱「無條件」的 failure-time dump 實際掛在 **decode worker 的例外
路徑**上；本次是 **main eval thread** 的 capture 失敗，走不到那條路徑。
這是觀測面的缺口，不是本次 terminal 的瑕疵——terminal 只依賴 §3 的字串述詞，
該述詞的輸入（log）完整且 sha 已核對。但它使本輪**無法**移交任何 stream 級證據
給 attribution。

## 7. 下一輪的前置 blocker

以下兩項在下一輪 campaign 開跑**之前**必須完成。

**B1 — capture failure 的觀測面必須涵蓋 main eval thread。**
capture 在 main eval thread 失敗時，也必須嘗試留下 site / thread / current stream /
stream flags / capture-state diagnostic，而不能只掛在 decode-worker 的 exception
path 上。**diagnostic 自己失敗也要被記錄，而不得蓋掉原始的 CUDA 901**——診斷程式碼
吞掉它要診斷的錯誤，會讓下一次 failure 比這一次更沒有證據。

**B2 — 修掉 `setup_failure_signature` 的 fail-open。**
現行判定是「全 log 出現 `.engine` + 任意 `failed|error|exception`」的 global fuzzy
match。`.engine` 來自每個 run 都會印的 `TRT MambaHead enabled: …/mamba_head_26m.engine`
banner，因此**任何**失敗的 run 都會被標成 `tensorrt` setup signature。

本次 `setup_failure_signature = "tensorrt"` 即為此假陽性；它**沒有**影響本輪任何
判定（該 run `progress_markers`=6 且已完成 1 條 sequence，`started` 保持 `true`；
且 §3 的 capture-error 述詞優先於 A1 邊界），本輪 `invalid` 亦為 0。

但方向是 fail-open：該 signature 只在 `markers == 0` 且無完成 sequence 時把
`started` 翻成 `false`，而那正是預註冊 §4 中**唯一可以補跑**的 invalid 類別。
一個過寬的 setup signature 會擴大這個可補跑類別，正好侵蝕 §4 第三條刻意設計成
沒有中間選項的那條性質。

修法方向：改以結構化 phase + 明確 setup signature 判定；至少也要把匹配限制在
failure-local context，避免正常的 TRT banner 充當 causal context。

## 8. §13 執行順序的結案

| §13 | 狀態 |
|---|---|
| 1 prereg merge（凍結點） | 完成 `05390b2e` |
| 2 備妥 target worktree | 完成 `b649de68`，detached / clean |
| 3 寫 harness | 完成（PR #372） |
| 4 harness ↔ spec equivalence check | 完成 |
| 5 跑 campaign | 完成，於 attempt 93 依 §5 early-terminate |
| 6 closure report | 本文件 |

下一步不是本預註冊的延伸：B1 / B2 完成後，在**新的 frozen target** 上以**新的
預註冊**重新進入 #340 attribution。
