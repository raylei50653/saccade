---
doc-status: closed
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-12
closed: 2026-07-12
closed-verdict: T2_NO_USABLE_RANKING_POWER_IN_CLASS
task-type: mainline-diagnostic-probe
production-impact: none
default-behavior-change: forbidden
---

# Ambiguous-band ranking-power probe (Door 0)

> **One-line (CLOSED):** terminal **`T2 NO_USABLE_RANKING_POWER_IN_CLASS`**
> RESEARCH ACCEPTED(PR #136 merged `9ec583c7`)——12-member tested class 在
> gate-retained ambiguous band 無可用 ranking power(best ΔPWA=+0.0011,距
> B1 +0.02 約 18×);step ⑤ class-scoped closure 生效;step ④ 未開;主線下一步
> =owner charter 決定。

## Final status

| Item | Status |
|:--|:--|
| Program | **CLOSED — terminal `T2 NO_USABLE_RANKING_POWER_IN_CLASS` RESEARCH ACCEPTED**;declaration sealed via PR #135(main `f864a6e2`);results accepted+merged via [PR #136](https://github.com/raylei50653/saccade/pull/136)(`9ec583c7`) |
| Terminal record | V1–V5 全 PASS · H 未觸發(baseline PWA 0.878/top-1 0.590;P3=84/205 top-1 miss)· **12/12 candidates fail boxes**(best ΔPWA=+0.0011,距 B1 +0.02 約 18×;owner 裁定非邊界案例)· motion conditions 有害(與 escape-tail forensics 一致)· [results note](../../../modules/semantic/research/door0_ranking_probe_results_20260712.md) · [packet](../../../modules/semantic/research/evidence/door0_ranking_probe_20260712/manifest.json) |
| Closure in effect | **step ⑤ class-scoped closure**:12 members 不得在相同 family/substrate 重跑;band=unexplained residual set w.r.t. 這 12 members;9 個未測 AND pair/其他 quantile/連續訊號/有限 λ/learned score **不被耗盡**;step ④ 未開;擴 class/換訊號/轉向=新 §20.2 宣告+owner charter,非本線延續 |
| Direct handoff disposition | **Tested 12-member score class:** **no receiver / no continuation.** No new research charter is opened by this terminal. |
| Cross-thread consequence (not handoff) | Door 0 T2 closes [GT-support morphology](../gt_support_morphology_20260711.md)'s score branch for this tested class; its gate branch remains parked. D0 / #112 CUDA capture remains an independent active non-WIP follow-up in [gap-conditioned motion](../gap_conditioned_probabilistic_motion_probe_20260711.md), not a Door 0 receiver. |
| Mainline position | realignment sealed 順序 **step ③ 完成**(①contract v1 PR #133 → ②bookkeeping PR #134 → ③本 probe **T2** → **⑤ class-scoped closure**;step ④ 未開,擴 class=新宣告+owner charter) |
| §20.2 declaration | [declaration doc](../../../modules/semantic/research/ambiguous_band_ranking_power_probe_declaration_20260712.md)（target layer=score-ranking · intent=capability map · output class=diagnostic result only） |
| Core question | 凍結 signal family 在 gate-retained ambiguous band 內,對 GT vs FP 的相對排序是否存在**穩定、可解釋、LOO 保留**的改善(超越 production `s0=score_m_bridge` baseline)? |
| Substrate | frozen pairs `0ae38967…`(7-seq SDP;21,789 gt_valid rows;340 GT);coarse gate=production h-window proxy;**210/205 rankable events**(gate 前/後) |
| Probe family | 6 single-atom + 6 second-order AND conditions;lexicographic demotion(parameter-free;不 fit λ);q85/q15 band quantiles |
| Terminals | T1 `RANKING_SIGNAL_PRESENT`→開 step ④ / T2 `NO_USABLE_RANKING_POWER_IN_CLASS`→只封 **12-member tested class**(step ⑤ class-scoped;9 個未測 AND pair/其他 quantile/連續訊號/有限 λ 均不被耗盡)/ T3 `NO_HEADROOM`(H1 PWA∧H2 top-1 joint)→step ④ 在 Door-0 resolution 下不必要 / T0 `UNRESOLVED/INVALID-STUDY`→只關實驗;**每個 terminal 無條件附 reachable-set scope caveat**(無方向判斷) |
| Prior risk | [幾何 AUC≈0.55 前例](../../../modules/semantic/research/boolean_closure_domain_line_20260711.md)——band 內 ranking power 可能本來就弱;此即 Door 0 存在的理由(先驗證有訊號再投資 ④) |
| Production / presets / ledger | **unchanged**(read-only probe) |

## Current step

**none — closed.**

## Read first

1. [Declaration(§20.2 block+boxes+terminals)](../../../modules/semantic/research/ambiguous_band_ranking_power_probe_declaration_20260712.md)
2. [Framework §20 contract](../../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
3. [Production substrate mapping(consumer split;s0 proxy 語意)](../../../modules/semantic/research/production_substrate_mapping_20260711.md)
4. 上游 capability maps:[GT-support morphology thread](../gt_support_morphology_20260711.md)(diagnostic;SUSPENDED prototype 與本 probe 的 re-charter score 方向相接)

## Acceptance

```text
seal: declaration PR merged by owner
run:  V1–V5 validity gate → H headroom → 12 candidates × B1–B6 boxes
out:  exactly one terminal(T0–T3)+ full 12-candidate capability map
      (no cherry-pick;P3 hard subset+reachable slice reported)
then: T1→step ④ design-evaluation charter;T2→step ⑤ closure statement;
      T3→step ④ closed as unnecessary;T0→fix validity blocker only
```

## Must not

- seal 前執行 ranking 計算(counts-only recon 已揭露於 declaration §8);
- seal 後偏離宣告(atoms/quantiles/metrics/boxes/terminals)——偏離=作廢重宣告;
- 以 best performer 取代 boxes;futility 後擴家族;
- 動 production preset、evidence_ledger、safe-region closed gates。

## History

- 2026-07-12: Realignment step ③ 開工。Counts-only recon(event 結構/pool 大小,無 ranking 指標):210 rankable events(gate 後 205,per-seq min 12);reachable slice(s0≤0.4)僅 34 events→降 descriptive-only。Declaration doc 完成,PR open = seal unit。
- 2026-07-12: **Owner seal review = SEAL BLOCKED(四組 terminal-affecting 自由度)→ Revision 2 落地**:①H/T3 改 joint(H1 PWA≥0.98 ∧ H2 top-1≥0.98),撤「算術不可能」錯誤理由,明定 H-over-boxes precedence;②凍結 rank tie policy(pessimistic against GT)與 quantile estimator(`numpy.quantile method="linear"`),B3 明定 metric-level re-aggregation 不重 fit threshold(refit 只在 B5);③B6 改機械判定(B6a fire-rate P(c|FP)>P(c|GT) ∧ B6b flip decomposition n_good>n_bad,均 strict),排除 reviewer 事後判斷;④T2 更名 `NO_USABLE_RANKING_POWER_IN_CLASS` 並限 Door-0 complexity class(擴 class=新宣告+owner charter),reachable-slice limit rule 改為適用全部 terminal。等第二輪 seal review。
- 2026-07-12: **Round-2 owner review = SEAL BLOCKED(narrow scope fix)→ Revision 3 落地**:①class 定義從「up to second-order AND」收斂為**恰好 §6 列舉的 12 members**(15 個二階 AND 中 9 個未測,不得被 T2 耗盡);②reachable-slice caveat 改**無條件**——T0–T3 每個 recorded terminal 一律附 §3 verbatim clause,無方向 trigger,消除 implementer 選擇。等最終 seal review。
- 2026-07-12: **Final review = 一個 clause-level blocker → Revision 4**:universal caveat 開頭「established on the gate-retained band」與 T0(UNRESOLVED,不建立任何結論)矛盾;改為中性措辭「study scope is the gate-retained band; this terminal establishes no claim inside the production-reachable set…」,對 T0–T3 一致成立。無 terminal/box/candidate/protocol 變更。
- 2026-07-12: **SEAL ACCEPTED → PR #135 MERGED(main `f864a6e2`)= declaration sealed。**
- 2026-07-12: **Post-seal 單次執行完成(無宣告偏離)**:runner `scripts/tools/run_door0_ranking_probe.py`(V1 SHA 拒跑;合成資料 self-test 先行,未觸 substrate)。V1–V5 全 PASS;H 未觸發;**12/12 fail → terminal `T2 NO_USABLE_RANKING_POWER_IN_CLASS`** + 無條件 caveat。關鍵機制發現:class 可見的 unsafe tail 幾乎全是 `s0` 已排後的 FP(demote 無效,good/bad=0/0 多例);殘餘 84 個 top-1-miss events 的混淆在 in-band 聯合分佈內,tail conditions 無法分離;motion conditions 因 band 內 GT fire rate 不低而淨有害。Results PR open = research-acceptance unit。
- 2026-07-12: **RESEARCH ACCEPTED — [PR #136](https://github.com/raylei50653/saccade/pull/136) MERGED(`9ec583c7`),CI 全綠。**Owner 裁定:runner 忠實實作 sealed declaration,無 terminal-affecting 偏離;封閉範圍=恰好 12 members;判決非邊界案例(best ΔPWA=+0.001097,距 +0.02 約 18×,次要 box 解讀變化不足以翻轉)。**Step ⑤ class-scoped closure 生效。**非阻擋文字校正:results note「two orders of magnitude」→ 正確約 18×(owner acceptance review 記錄;本次 close PR 同步修正)。Thread CLOSED,移入 `threads/closed/`;semantic sole-active 清空,mainline 下一步=owner charter。
