<!-- doc-status: closed -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-09 -->

# #340 capture-race incidence closure — report (2026-09-09)

本文件是
[capture_race_incidence_preregistration_20260909.md](capture_race_incidence_preregistration_20260909.md)
（以下稱「預註冊」）§13 step 6 的產出，以及 owner closure review 的落盤。
它只報告該預註冊所凍結的量與 terminal，不新增自由度、不重新描述統計理由、
不引入預註冊沒有事先列舉的判準。

## 1. Terminal

**`CLOSED_BOUNDED`**（預註冊 §7 五個 terminal 之一）。

Closure timestamp: **2026-09-09T12:31:28Z**。

依 §7：關閉 core unknown「#340 在本凍結 coordinate / observer overlay 上的
發生率未知」。#340 進 closure review。**不**關閉 mechanism，**不**關閉歷史
2026-09-08 CUDA 901。

## 2. 計數

| | 值 |
|---|---|
| campaign id | `20260909T100259Z` |
| production target source SHA | `4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f` |
| observer/control freeze | `276d8d744d7050ad272f9b69cf3c60b0c31333a5` |
| 起迄（UTC） | 2026-09-09T10:02:59Z → 12:31:28Z |
| attempts | 200 |
| verdict `ok` | 200 |
| verdict `failure` | 0 |
| verdict `invalid` | **0** |
| verdict `execution_invalid` | 0 |
| Path A valid runs | 100（0 failure） |
| Path B valid runs | 100（0 failure） |

`invalid` 總數為 **0**：預註冊 §4 的記錄義務要求 closure report 列出它。
本次沒有任何 setup-invalid，effective-slot 補跑從未被觸發，attempt ordinal
與 slot 一一對應（seq_index 0–199，slot_index 0–99，交錯 A,B）。

每一筆 run 的 `target_head_observed` 均為凍結 target SHA，
`target_worktree_clean` 均為 true，`observer_sha256_observed` 均為凍結
`observer.so`。`attribution_status` 全部為 `not_applicable`：沒有 §3
failure 可做第二層分析。

## 3. 統計與措辭（預註冊 §6，逐字）

> Path A: 0/100 observed; one-sided 95% upper bound ≈ 2.95% per 7-sequence run.

> Path B: 0/100 observed; one-sided 95% upper bound ≈ 2.95% per 7-sequence run.

這是兩個 path-specific 宣稱，**不得**併成 0/200。單位是 **per 7-sequence run**。

**本文件不寫、任何引用也不得寫：** true failure rate = 0、race 已消失、
CUDA 901 已修復、historical root cause 已辨識、#379 修好了歷史 901。

## 4. Closure review 的範圍邊界

This closure bounds incidence under the frozen coordinate / observer overlay
only. It does **not** establish that the historical 2026-09-08 CUDA 901 was
caused by a particular race, that the mechanism has been identified, that the
race has disappeared, or that #379 repaired the historical CUDA 901.

The historical CUDA 901 therefore remains recorded as **causally unresolved**,
while #379 remains classified as safety hardening / mechanism isolation rather
than root-cause proof.

If a materially similar 900/901/906 incident is observed again under a future
coordinate, treat it as a new incident / investigation and reference #340 as
prior evidence rather than automatically reopening this campaign.

## 5. 證據

campaign artifact（非 scratch，timestamped）：

```text
~/.local/state/saccade/perf/capture-race-incidence-20260909/
  manifest.json      terminal CLOSED_BOUNDED; valid A=100 B=100; failures 0; invalid 0
  runs.jsonl         200 筆 capture_race_incidence_run_v1
  runs/<NNNN>-<P>/   每 run 的 MOT 輸出
  traces/<NNNN>-<P>/ 每 run 的 observer overlay trace
  logs/<NNNN>-<P>.log 每 run 的 wrapper-retained stdout+stderr
  sealed/<NNNN>-<P>.json 每 run 的 sealed incidence row
```

controller stdout：
`~/.local/state/saccade/perf/capture-race-incidence-20260909-controller.log`。

`manifest.json` 的 `preregistration` 欄記的是 campaign 執行當下的路徑
`docs/research/pipeline/capture_race_incidence_preregistration_20260909.md`。
本 PR 依 Doc Structure C6 把該文件與 seal 移入 `closed/`，那個欄位因此是
**移動前**的路徑；manifest 是已落盤的執行紀錄，不回填。

## 6. §13 執行順序的結案

| §13 | 狀態 |
|---|---|
| 1 observer/control freeze | `276d8d744d7050ad272f9b69cf3c60b0c31333a5` |
| 2 prereg merge（凍結點） | `72adf71e6066b5f54abcbb9f263a1f8ae427b980` |
| 3 備妥 target worktree | `4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f` detached / clean |
| 4 寫 harness | `59dd4401a7d0a838d7e98913db1a7937451d52ba` |
| 5 harness ↔ spec equivalence | 完成 |
| 6 跑 campaign | 完成，200/200 `ok`，terminal `CLOSED_BOUNDED` |
| 7 closure report | 本文件 |
